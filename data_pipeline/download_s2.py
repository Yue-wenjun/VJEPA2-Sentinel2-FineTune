"""
Sentinel-2 L2A patch downloader via Microsoft Planetary Computer (STAC API).

Pitfall checklist:
  1. Band resolution mismatch  -- 20m bands upsampled to 10m with bilinear
  2. No-data edges             -- patch sampling avoids tile borders
  3. Scale factor              -- L2A DN / 10000 → [0, 1] reflectance
  4. Cloud classification      -- SCL layer → cloud_cat label (-1/0/1/2); all frames kept
  5. Temporal gaps             -- only patches with ≥MIN_VALID_FRAMES kept
  6. File size                 -- only the 6 target bands + SCL are fetched (COG windowed read)
  7. Authentication            -- sign + read merged into one retried call; expired tokens
                                  are always refreshed before each retry attempt
  8. Rate limiting             -- tenacity retry wrapper on all network operations
  9. N0500+ SCL format         -- SCL read failure is non-fatal; bands still downloaded, cloud_cat=-1
 10. Windows SSL               -- GDAL_HTTP_UNSAFESSL=YES set before rasterio import

Output layout:
    <out_dir>/
        patches/
            <tile_id>_<row>_<col>/
                <YYYY-MM-DD>.npy        # float32 [H, W, 6]  (bands: B02 B03 B04 B08 B11 B12)
                <YYYY-MM-DD>.meta.npy   # scalar float32: cloud_frac (-1.0 if SCL unavailable)
        index.csv      # path, doy, cloud_frac, cloud_cat, patch_key -- one row per frame
        sequences.csv  # patch_key, frame_paths, doys, cloud_fracs, cloud_cats, n_frames

Usage:
    pip install pystac-client planetary-computer rioxarray rasterio scipy tqdm tenacity pandas
    python download_s2.py --bbox 116.0 39.5 116.5 40.0 --start 2022-01-01 --end 2023-12-31
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path

# Must be set before GDAL/rasterio is imported.
# Windows schannel has TLS handshake failures with Azure Blob Storage;
# GDAL_HTTP_UNSAFESSL bypasses the Windows-specific TLS issue.
os.environ.setdefault("GDAL_HTTP_UNSAFESSL", "YES")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
# Disable slow directory scans on vsicurl; speeds up repeated COG opens.
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

import numpy as np
import planetary_computer
import pystac_client
import rioxarray  # noqa: F401 – registers rio accessor on xarray
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

def _cloud_category(cf: float) -> int:
    """Map cloud fraction to integer category for fast filtering.
    0 = clean   (cf < 0.10)
    1 = moderate (0.10 ≤ cf < 0.30)
    2 = cloudy  (cf ≥ 0.30)
    """
    if cf < 0.10:
        return 0
    if cf < 0.30:
        return 1
    return 2


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


@retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=1, min=2, max=60))
def _read_patch(item, band: str, row: int, col: int, size: int, target_size: int) -> np.ndarray:
    """Sign the item fresh on every attempt, then read the spatial window.

    Merging sign + read into one retried unit means an expired token is always
    refreshed before the next retry — not just at the URL-fetch step.
    Handles both token expiry (se= expired URLs) and transient COG read errors
    (TIFFReadEncodedTile, network interruptions).
    """
    import rioxarray as rxr
    href = planetary_computer.sign(item).assets[band].href
    da = rxr.open_rasterio(href, masked=True).squeeze("band", drop=True)
    patch = da.isel(x=slice(col, col + size), y=slice(row, row + size))
    arr = patch.values.astype(np.float32)
    # upsample if needed (20m → 10m: size doubled)
    if arr.shape[0] != target_size or arr.shape[1] != target_size:
        from scipy.ndimage import zoom
        scale = target_size / arr.shape[0]
        arr = zoom(arr, scale, order=1)
    return arr


@retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=1, min=2, max=60))
def _probe_tile(item) -> tuple:
    """Open B02 of the first scene to get tile pixel dimensions."""
    import rioxarray as rxr
    href = planetary_computer.sign(item).assets["B02"].href
    da = rxr.open_rasterio(href, masked=True).squeeze("band", drop=True)
    return tuple(da.shape)   # (H, W)


# --- main download function ---------------------------------------------------

def download_patches(
    bbox,                       # (lon_min, lat_min, lon_max, lat_max)
    start_date: str,
    end_date: str,
    out_dir: Path,
    patch_size: int = 384,      # pixels at 10m → 3.84km × 3.84km; matches pretrained 384px
    stride: int = 384,          # stride between patches (no overlap by default)
    min_valid_frames: int = 4,
    border_px: int = 192,       # skip this many pixels at tile borders (avoid no-data)
    patch_list: dict | None = None,  # {tile_id: [(px_row, px_col), ...]}; None = full grid
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

    # When patch_list is provided, skip tiles not in the list
    if patch_list is not None:
        by_tile = {tid: v for tid, v in by_tile.items() if tid in patch_list}
        log.info(f"patch_list active: downloading {len(by_tile)} tile(s), "
                 f"{sum(len(v) for v in patch_list.values())} patch(es)")

    all_sequence_rows = []
    all_frame_rows = []

    for tile_id, tile_items in by_tile.items():
        tile_items = sorted(tile_items, key=lambda x: x.datetime)
        log.info(f"Tile {tile_id}: {len(tile_items)} scenes")

        tile_h, tile_w = _probe_tile(tile_items[0])
        log.info(f"  Tile size: {tile_h} × {tile_w} px (10m)")

        # Build patch grid: either full uniform grid or the provided patch_list
        if patch_list is not None:
            patch_coords = patch_list[tile_id]            # [(px_row, px_col), ...]
            log.info(f"  Using {len(patch_coords)} selected patches")
        else:
            rows = list(range(border_px, tile_h - border_px - patch_size, stride))
            cols = list(range(border_px, tile_w - border_px - patch_size, stride))
            patch_coords = [(pr, pc) for pr in rows for pc in cols]
            log.info(f"  Patch grid: {len(patch_coords)} patches")

        for pr, pc in patch_coords:
            patch_key = f"{tile_id}_r{pr:05d}_c{pc:05d}"
            patch_out = patches_dir / patch_key
            patch_out.mkdir(exist_ok=True)

            frame_paths = []
            frame_doys = []
            frame_cloud_fracs = []
            frame_cloud_cats = []

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
                    frame_cloud_cats.append(_cloud_category(cf) if cf >= 0 else -1)
                    continue

                # --- Step 1: SCL cloud classification (non-fatal if unavailable) ---
                cf = -1.0
                cloud_cat = -1  # -1 = SCL unavailable (e.g. N0500+ baseline format)
                scl_size = math.ceil(patch_size / 2)
                try:
                    scl_patch = _read_patch(
                        item, SCL_BAND,
                        pr // 2, pc // 2, scl_size, scl_size
                    ).astype(np.uint8)
                    cf = _cloud_fraction(scl_patch)
                    cloud_cat = _cloud_category(cf)
                except Exception as e:
                    log.warning(f"  SCL unavailable {date_str} {patch_key}: {e}")

                # --- Step 2: spectral bands (fatal if unavailable) ---
                try:
                    band_arrays = []
                    for band in BANDS_10M:
                        arr = _read_patch(item, band, pr, pc, patch_size, patch_size)
                        band_arrays.append(arr)

                    for band in BANDS_20M:
                        arr = _read_patch(
                            item, band,
                            pr // 2, pc // 2,
                            scl_size, patch_size,
                        )
                        band_arrays.append(arr)

                    stack = np.stack(band_arrays, axis=-1)
                    stack = np.clip(stack, 0, S2_SCALE) / S2_SCALE
                    stack = stack.astype(np.float32)

                    np.save(out_npy, stack)
                    np.save(out_meta, np.float32(cf))
                    doy = item.datetime.timetuple().tm_yday
                    frame_paths.append(str(out_npy))
                    frame_doys.append(doy)
                    frame_cloud_fracs.append(cf)
                    frame_cloud_cats.append(cloud_cat)

                except Exception as e:
                    log.warning(f"  Failed bands {date_str} {patch_key}: {e}")
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
                    "cloud_cats": ",".join(map(str, frame_cloud_cats)),
                    "n_frames": len(frame_paths),
                })
                for rp, doy, cf, cc in zip(rel_paths, frame_doys, frame_cloud_fracs, frame_cloud_cats):
                    all_frame_rows.append({
                        "path": rp, "doy": doy,
                        "cloud_frac": cf, "cloud_cat": cc, "patch_key": patch_key,
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
                        help="Bounding box in WGS84 used to query STAC")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end",   default="2023-12-31")
    parser.add_argument("--out_dir", default="./s2_data")
    parser.add_argument("--patch_size", type=int, default=384)
    parser.add_argument("--stride",     type=int, default=384)
    parser.add_argument("--min_frames", type=int, default=4)
    parser.add_argument("--patches_csv", default=None,
                        help="CSV from sample_patches.py; if provided, only "
                             "download listed patches instead of the full grid")
    args = parser.parse_args()

    patch_list = None
    if args.patches_csv:
        import pandas as pd
        from collections import defaultdict
        df_p = pd.read_csv(args.patches_csv)
        patch_list = defaultdict(list)
        for _, row in df_p.iterrows():
            patch_list[row["tile_id"]].append((int(row["px_row"]), int(row["px_col"])))
        patch_list = dict(patch_list)

    download_patches(
        bbox=tuple(args.bbox),
        start_date=args.start,
        end_date=args.end,
        out_dir=Path(args.out_dir),
        patch_size=args.patch_size,
        stride=args.stride,
        min_valid_frames=args.min_frames,
        patch_list=patch_list,
    )
