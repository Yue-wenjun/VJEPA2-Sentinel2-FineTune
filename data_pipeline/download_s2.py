"""
Sentinel-2 L2A patch downloader via Microsoft Planetary Computer (STAC API).

Pitfall checklist:
  1. Band resolution mismatch  -- 20m bands upsampled to 10m with bilinear
  2. No-data edges             -- patch sampling avoids tile borders
  3. Scale factor              -- L2A DN / 10000 → [0, 1] reflectance
  4. Cloud masking             -- SCL layer; patches with >MAX_CLOUD_FRAC discarded
  5. Temporal gaps             -- only patches with ≥MIN_VALID_FRAMES kept
  6. File size                 -- only the 6 target bands + SCL are fetched (COG windowed read)
  7. Authentication            -- planetary_computer.sign() handles token refresh automatically
  8. Rate limiting             -- tenacity retry wrapper on HTTP requests

Output layout:
    <out_dir>/
        patches/
            <tile_id>_<row>_<col>/
                <YYYY-MM-DD>.npy   # float32 [H, W, 6]  (bands: B02 B03 B04 B08 B11 B12)
                <YYYY-MM-DD>.npy
                ...
        index.csv                  # path, doy columns -- one row per (location, frame)
        sequences.csv              # one row per location; comma-sep list of frame paths

Usage:
    pip install pystac-client planetary-computer stackstac rioxarray rasterio tqdm tenacity
    python download_s2.py --bbox 116.0 39.5 116.5 40.0 --start 2022-01-01 --end 2023-12-31 --out_dir ./s2_data
"""

import argparse
import logging
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import planetary_computer
import pystac_client
import rioxarray  # noqa: F401 – registers rio accessor on xarray
import xarray as xr
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# --- constants ----------------------------------------------------------------

BANDS_10M = ["B02", "B03", "B04", "B08"]       # blue green red nir -- native 10m
BANDS_20M = ["B11", "B12"]                      # swir1 swir2 -- native 20m, upsampled
SCL_BAND = "SCL"                                # scene classification layer (20m)

# SCL classes considered cloudy / invalid
SCL_CLOUD_CLASSES = {3, 8, 9, 10}               # cloud shadow, med cloud, high cloud, cirrus
SCL_NODATA_CLASS = 0

S2_SCALE = 10_000.0                             # L2A DN → reflectance

PLANETARY_COMPUTER_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"

# --- helpers ------------------------------------------------------------------

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
def _signed_asset_url(item, band: str) -> str:
    item_signed = planetary_computer.sign(item)
    return item_signed.assets[band].href


def _cloud_fraction(scl_patch: np.ndarray) -> float:
    total = scl_patch.size
    if total == 0:
        return 1.0
    cloud = np.isin(scl_patch, list(SCL_CLOUD_CLASSES)).sum()
    nodata = (scl_patch == SCL_NODATA_CLASS).sum()
    valid = total - nodata
    if valid == 0:
        return 1.0
    return float(cloud) / valid


def _read_band_window(href: str, row: int, col: int, size: int, target_size: int) -> np.ndarray:
    """Read a spatial window from a COG href using rioxarray, return float32 [H, W]."""
    import rioxarray as rxr
    da = rxr.open_rasterio(href, masked=True).squeeze("band", drop=True)
    # window in native pixels
    patch = da.isel(x=slice(col, col + size), y=slice(row, row + size))
    arr = patch.values.astype(np.float32)
    # upsample if needed (20m → 10m: size doubled)
    if arr.shape[0] != target_size or arr.shape[1] != target_size:
        from scipy.ndimage import zoom
        scale = target_size / arr.shape[0]
        arr = zoom(arr, scale, order=1)
    return arr


# --- main download function ---------------------------------------------------

def download_patches(
    bbox,                       # (lon_min, lat_min, lon_max, lat_max)
    start_date: str,
    end_date: str,
    out_dir: Path,
    patch_size: int = 224,      # pixels at 10m → 2.24km × 2.24km
    stride: int = 224,          # stride between patches (no overlap by default)
    max_cloud_frac: float = 0.3,
    min_valid_frames: int = 4,
    border_px: int = 112,       # skip this many pixels at tile borders (avoid no-data)
):
    out_dir = Path(out_dir)
    patches_dir = out_dir / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)

    # ---- query STAC ----------------------------------------------------------
    catalog = pystac_client.Client.open(
        PLANETARY_COMPUTER_URL,
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=[COLLECTION],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": 80}},   # coarse pre-filter; patch-level below
    )
    items = list(search.items())
    log.info(f"Found {len(items)} S2 scenes for bbox={bbox}")
    if not items:
        log.warning("No scenes found. Check bbox / date range.")
        return

    # ---- group by MGRS tile so we only sample patches within one tile --------
    from collections import defaultdict
    by_tile = defaultdict(list)
    for item in items:
        tile_id = item.properties.get("s2:mgrs_tile", item.id[:5])
        by_tile[tile_id].append(item)

    all_sequence_rows = []
    all_frame_rows = []

    for tile_id, tile_items in by_tile.items():
        tile_items = sorted(tile_items, key=lambda x: x.datetime)
        log.info(f"Tile {tile_id}: {len(tile_items)} scenes")

        # Determine patch grid from the first available scene
        first_item = planetary_computer.sign(tile_items[0])
        import rioxarray as rxr
        sample_da = rxr.open_rasterio(first_item.assets["B02"].href, masked=True).squeeze()
        tile_h, tile_w = sample_da.shape
        log.info(f"  Tile size: {tile_h} × {tile_w} px (10m)")

        # Build patch grid avoiding borders
        rows = list(range(border_px, tile_h - border_px - patch_size, stride))
        cols = list(range(border_px, tile_w - border_px - patch_size, stride))
        log.info(f"  Patch grid: {len(rows)} × {len(cols)} = {len(rows)*len(cols)} patches")

        for row_idx, pr in enumerate(rows):
            for col_idx, pc in enumerate(cols):
                patch_key = f"{tile_id}_r{row_idx:04d}_c{col_idx:04d}"
                patch_out = patches_dir / patch_key
                patch_out.mkdir(exist_ok=True)

                frame_paths = []
                frame_doys = []
                frame_cloud_fracs = []

                for item in tqdm(tile_items, desc=patch_key, leave=False):
                    date_str = item.datetime.strftime("%Y-%m-%d")
                    out_npy = patch_out / f"{date_str}.npy"
                    out_meta = patch_out / f"{date_str}.meta.npy"

                    if out_npy.exists() and out_meta.exists():
                        doy = item.datetime.timetuple().tm_yday
                        cf = float(np.load(out_meta))
                        frame_paths.append(str(out_npy))
                        frame_doys.append(doy)
                        frame_cloud_fracs.append(cf)
                        continue

                    try:
                        # Read SCL first (cheap cloud check)
                        scl_href = _signed_asset_url(item, SCL_BAND)
                        # SCL is 20m → patch coords halved
                        scl_row, scl_col = pr // 2, pc // 2
                        scl_size = math.ceil(patch_size / 2)
                        scl_patch = _read_band_window(
                            scl_href, scl_row, scl_col, scl_size, scl_size
                        ).astype(np.uint8)

                        cf = _cloud_fraction(scl_patch)
                        if cf > max_cloud_frac:
                            continue

                        # Read 10m bands
                        band_arrays = []
                        for band in BANDS_10M:
                            href = _signed_asset_url(item, band)
                            arr = _read_band_window(href, pr, pc, patch_size, patch_size)
                            band_arrays.append(arr)

                        # Read 20m bands and upsample
                        for band in BANDS_20M:
                            href = _signed_asset_url(item, band)
                            arr = _read_band_window(
                                href, pr // 2, pc // 2,
                                math.ceil(patch_size / 2), patch_size
                            )
                            band_arrays.append(arr)

                        # Stack → [H, W, 6], clip invalid DN, normalize
                        stack = np.stack(band_arrays, axis=-1)          # [H, W, 6]
                        stack = np.clip(stack, 0, S2_SCALE) / S2_SCALE  # [0, 1]
                        stack = stack.astype(np.float32)

                        np.save(out_npy, stack)
                        np.save(out_meta, np.float32(cf))   # persist cloud_frac for ablation
                        doy = item.datetime.timetuple().tm_yday
                        frame_paths.append(str(out_npy))
                        frame_doys.append(doy)
                        frame_cloud_fracs.append(cf)

                    except Exception as e:
                        log.warning(f"  Failed {date_str} patch {patch_key}: {e}")
                        continue

                if len(frame_paths) >= min_valid_frames:
                    # Store POSIX-style relative paths so the CSV is portable across OS
                    rel_paths = [
                        Path(fp).relative_to(out_dir).as_posix() for fp in frame_paths
                    ]
                    all_sequence_rows.append({
                        "patch_key": patch_key,
                        "frame_paths": ",".join(rel_paths),
                        "doys": ",".join(map(str, frame_doys)),
                        "cloud_fracs": ",".join(f"{c:.4f}" for c in frame_cloud_fracs),
                        "n_frames": len(frame_paths),
                    })
                    for rp, doy, cf in zip(rel_paths, frame_doys, frame_cloud_fracs):
                        all_frame_rows.append({
                            "path": rp, "doy": doy,
                            "cloud_frac": cf, "patch_key": patch_key,
                        })

    import pandas as pd
    pd.DataFrame(all_frame_rows).to_csv(out_dir / "index.csv", index=False)
    pd.DataFrame(all_sequence_rows).to_csv(out_dir / "sequences.csv", index=False)
    log.info(f"Done. {len(all_sequence_rows)} sequences, {len(all_frame_rows)} frames → {out_dir}")


# --- CLI ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox", type=float, nargs=4,
                        metavar=("lon_min", "lat_min", "lon_max", "lat_max"),
                        default=[116.0, 39.5, 116.5, 40.0],
                        help="Bounding box in WGS84")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end",   default="2023-12-31")
    parser.add_argument("--out_dir", default="./s2_data")
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--stride",     type=int, default=224)
    parser.add_argument("--max_cloud",  type=float, default=0.3)
    parser.add_argument("--min_frames", type=int,   default=4)
    args = parser.parse_args()

    download_patches(
        bbox=tuple(args.bbox),
        start_date=args.start,
        end_date=args.end,
        out_dir=Path(args.out_dir),
        patch_size=args.patch_size,
        stride=args.stride,
        max_cloud_frac=args.max_cloud,
        min_valid_frames=args.min_frames,
    )
