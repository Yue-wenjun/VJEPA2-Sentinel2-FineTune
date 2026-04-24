"""
Stratified patch sampler for Sentinel-2 time-series dataset.

Two modes:
  full  — scan DEFAULT_BBOXES (22 regions) and select patches across all LULC classes
  rare  — read an existing CSV, find classes below --threshold, top them up using
          RARE_BBOXES (fresh 2°×2° regions, distinct from DEFAULT_BBOXES)

Two-stage scoring:
  Stage 1 — Tile-level (free, STAC metadata only):
    - s2:vegetation_percentage temporal variance → temporal change proxy
  Stage 2 — Patch-level (2 COG reads per tile):
    - ESA WorldCover full tile → dominant LULC class per patch
    - Copernicus DEM full tile → terrain roughness per patch

Bad-patch filter (new):
    - Patches where valid WorldCover pixels < --min_valid_frac are skipped
    - Each patch records wc_source ("worldcover" | "stac_fallback"); downstream
      filtering can drop stac_fallback patches for fine-grain rare classes.

Output columns:
    tile_id, px_row, px_col, lat, lon,
    lulc_class, lulc_name, terrain_roughness, ndvi_var, priority,
    wc_source, query_bbox

Usage:
    # Full run (all classes)
    python sample_patches.py --proxy

    # Supplement rare classes from existing CSV
    python sample_patches.py --mode rare --existing_csv ../selected_patches.csv --proxy

    # Custom bboxes
    python sample_patches.py --bboxes_json my_bboxes.json --proxy
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("GDAL_HTTP_UNSAFESSL", "YES")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rioxarray as rxr
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# --- Proxy -------------------------------------------------------------------

_PROXY_HTTP  = "http://127.0.0.1:33210"
_PROXY_SOCKS = "socks5://127.0.0.1:33211"

def configure_proxy() -> None:
    """Set HTTP/SOCKS proxy env vars for requests, urllib3, and GDAL COG reads.

    Python requests/urllib → HTTP proxy (33210)
    GDAL/vsicurl           → SOCKS5 proxy (33211)

    GDAL uses libcurl. HTTP CONNECT tunnels + Windows schannel cause TLS handshake
    failures. SOCKS5 forwards raw TCP transparently, so TLS goes end-to-end and
    schannel works correctly.
    """
    os.environ["HTTP_PROXY"]       = _PROXY_HTTP
    os.environ["HTTPS_PROXY"]      = _PROXY_HTTP
    os.environ["ALL_PROXY"]        = _PROXY_SOCKS
    os.environ["GDAL_HTTP_PROXY"]  = _PROXY_SOCKS   # SOCKS5: no TLS interception
    log.info(f"Proxy: requests → {_PROXY_HTTP}  GDAL/vsicurl → {_PROXY_SOCKS}")

# --- Constants ---------------------------------------------------------------

PLANETARY_COMPUTER_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_COLLECTION          = "sentinel-2-l2a"
WORLDCOVER_COLLECTION  = "esa-worldcover"
DEM_COLLECTION         = "cop-dem-glo-30"

LULC_NAMES = {
    10: "trees",     20: "shrubland", 30: "grassland",  40: "cropland",
    50: "built_up",  60: "barren",    70: "snow_ice",   80: "water",
    90: "wetland",   95: "mangrove",  100: "moss_lichen",
}

DEFAULT_TARGET_FRACS = {
    10: 0.20,   # trees
    20: 0.07,   # shrubland
    30: 0.12,   # grassland
    40: 0.20,   # cropland
    50: 0.10,   # built-up
    60: 0.08,   # barren
    70: 0.04,   # snow/ice
    80: 0.07,   # water
    90: 0.05,   # wetland
    95: 0.03,   # mangrove
    100: 0.04,  # moss/lichen
}

# Full-run bboxes (22 regions, 0.5°×0.5°, all LULC classes)
DEFAULT_BBOXES = [
    # trees
    (-60.0, -3.5, -59.5, -3.0),    # 亚马逊热带雨林
    (100.0, 25.0, 100.5, 25.5),    # 云南亚热带森林
    # shrubland
    (150.0,-34.0, 150.5,-33.5),    # 澳大利亚硬叶灌木
    ( 18.0,-34.5,  18.5,-34.0),    # 南非灌木地
    # grassland
    ( 36.5, -1.5,  37.0, -1.0),    # 肯尼亚热带稀树草原
    (-105.0,40.0,-104.5, 40.5),    # 美国科罗拉多草原
    # cropland
    (116.0, 39.5, 116.5, 40.0),    # 华北平原农田
    (104.0, 30.5, 104.5, 31.0),    # 四川盆地农业
    # built-up
    (121.3, 31.1, 121.8, 31.6),    # 上海城区
    ( 77.0, 28.4,  77.5, 28.9),    # 新德里城区
    # barren
    ( 87.0, 43.5,  87.5, 44.0),    # 新疆荒漠
    ( 25.0, 26.0,  25.5, 26.5),    # 撒哈拉沙漠（埃及）
    # snow/ice
    (-45.0, 67.0, -44.5, 67.5),    # 格陵兰冰盖
    ( 15.0, 78.0,  15.5, 78.5),    # 斯瓦尔巴群岛
    # water
    ( 32.5, -0.5,  33.0,  0.0),    # 维多利亚湖
    (-93.0, 46.5, -92.5, 47.0),    # 苏必利尔湖
    # wetland
    ( 89.5, 22.0,  90.0, 22.5),    # 孟加拉国湿地
    (-57.0,-19.0, -56.5,-18.5),    # 巴西潘塔纳尔
    # mangrove
    ( 89.0, 21.5,  89.5, 22.0),    # 桑达班红树林
    (105.0,  9.5, 105.5, 10.0),    # 湄公河三角洲
    # moss/lichen
    ( 28.0, 63.0,  28.5, 63.5),    # 芬兰苔原
    (-72.0, 68.0, -71.5, 68.5),    # 加拿大哈德逊湾苔原
]

# Rare-mode bboxes: 2°×2°, geographically distinct from DEFAULT_BBOXES
RARE_BBOXES: dict[int, list[tuple]] = {
    20: [   # shrubland
        (-70.0, -42.0, -68.0, -40.0),   # 巴塔哥尼亚
        ( -5.0,  37.0,  -3.0,  39.0),   # 西班牙地中海
        (-122.0, 37.0,-120.0,  39.0),   # 加州查帕拉尔
        (  7.0,  33.0,   9.0,  35.0),   # 突尼斯/阿尔及利亚
    ],
    50: [   # built-up
        (139.0,  35.0, 141.0,  37.0),   # 东京
        (-47.0, -24.0, -45.0, -22.0),   # 圣保罗
        (  3.0,   6.0,   5.0,   8.0),   # 拉各斯
        ( -0.5,  51.0,   1.5,  53.0),   # 伦敦
    ],
    70: [   # snow/ice
        (-148.0,  62.0,-146.0,  64.0),  # 阿拉斯加冰川
        (  86.0,  27.0,  88.0,  29.0),  # 喜马拉雅
        ( -74.0, -52.0, -72.0, -50.0),  # 巴塔哥尼亚冰原
        ( 100.0,  70.0, 102.0,  72.0),  # 西伯利亚北极
    ],
    90: [   # wetland
        ( -92.0,  29.0, -90.0,  31.0),  # 路易斯安那沼泽
        (  22.0, -20.0,  24.0, -18.0),  # 奥卡万戈三角洲
        (  73.0,  60.0,  75.0,  62.0),  # 西西伯利亚湿地
        (  28.5,  51.0,  30.5,  53.0),  # 普里皮亚季湿地
    ],
    95: [   # mangrove
        ( -16.0,  10.0, -14.0,  12.0),  # 西非几内亚
        ( 145.0, -16.0, 147.0, -14.0),  # 澳大利亚昆士兰
        ( -81.0,  25.0, -79.0,  27.0),  # 佛罗里达大沼泽
        (  98.0,   7.0, 100.0,   9.0),  # 泰国南部
    ],
    100: [  # moss/lichen/tundra
        ( 100.0,  70.0, 102.0,  72.0),  # 西伯利亚苔原
        (  18.0,  70.0,  20.0,  72.0),  # 挪威北极
        (-160.0,  68.0,-158.0,  70.0),  # 阿拉斯加苔原
        ( -95.0,  62.0, -93.0,  64.0),  # 哈德逊湾西岸苔原
    ],
}

# --- COG read helpers --------------------------------------------------------

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=60))
def _read_full_band(item, band_key: str) -> np.ndarray:
    """Read an entire band from a STAC item (re-signed on every retry)."""
    href = planetary_computer.sign(item).assets[band_key].href
    da = rxr.open_rasterio(href, masked=True).squeeze("band", drop=True)
    return da.values.astype(np.float32)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=60))
def _probe_tile(item) -> tuple:
    """Return (H, W) of the tile's B02 band."""
    href = planetary_computer.sign(item).assets["B02"].href
    da = rxr.open_rasterio(href, masked=True).squeeze("band", drop=True)
    return tuple(da.shape)


def _stac_lulc_estimate(tile_items) -> int:
    """
    Estimate dominant LULC from S2 STAC SCL metadata — tile-uniform, zero COG reads.
    Only used as fallback when WorldCover is unavailable; wc_source is flagged
    'stac_fallback' so downstream can filter these patches.
    Cannot distinguish mangrove/wetland/shrubland from trees — do not trust for
    fine-grain rare classes.
    """
    props_list = [it.properties for it in tile_items[:12]]

    def mean_prop(key):
        vals = [p[key] for p in props_list if p.get(key) is not None]
        return float(np.mean(vals)) if vals else 0.0

    veg   = mean_prop("s2:vegetation_percentage")
    water = mean_prop("s2:water_percentage")
    snow  = mean_prop("s2:snow_ice_percentage")
    nveg  = mean_prop("s2:not_vegetated_percentage")

    if water > 40:  return 80
    if snow  > 30:  return 70
    if veg   > 50:  return 10
    if veg   > 25:  return 40
    if veg   > 10:  return 30
    if nveg  > 60:  return 60
    return 50

# --- Stage-2 patch feature extraction ----------------------------------------

def _extract_patch_features(
    tile_items,
    wc_arr: np.ndarray | None,
    dem_arr: np.ndarray | None,
    tile_h: int,
    tile_w: int,
    patch_size: int,
    stride: int,
    border_px: int,
    ndvi_var_tile: float,
    fallback_lulc: int = 60,
    min_valid_frac: float = 0.7,   # bad-patch filter: skip if < this fraction valid
) -> pd.DataFrame:
    """
    Score every candidate patch in the tile grid.

    Bad-patch filter: patches where valid WorldCover pixels < min_valid_frac
    (masked edges, cloud shadow) are silently dropped.

    wc_source column:
      "worldcover"    — label from ESA WorldCover COG (trustworthy)
      "stac_fallback" — label estimated from S2 SCL metadata (tile-uniform,
                        cannot distinguish mangrove/wetland/shrubland from trees)
    """
    rows = list(range(border_px, tile_h - border_px - patch_size, stride))
    cols = list(range(border_px, tile_w - border_px - patch_size, stride))

    records = []
    skipped = 0
    for pr in rows:
        for pc in cols:
            # ── LULC + bad-patch filter ───────────────────────────────────────
            if wc_arr is not None:
                wc_patch = wc_arr[pr: pr + patch_size, pc: pc + patch_size].astype(np.uint8)
                valid_frac = np.sum(wc_patch > 0) / wc_patch.size
                if valid_frac < min_valid_frac:
                    skipped += 1
                    continue
                vals, counts = np.unique(wc_patch[wc_patch > 0], return_counts=True)
                lulc_class = int(vals[np.argmax(counts)]) if len(vals) > 0 else fallback_lulc
                wc_source = "worldcover"
            else:
                lulc_class = fallback_lulc
                wc_source = "stac_fallback"

            # ── DEM roughness ─────────────────────────────────────────────────
            roughness = 0.0
            if dem_arr is not None:
                scale = dem_arr.shape[0] / tile_h
                dr = int(pr * scale)
                dc = int(pc * scale)
                ds = max(1, int(patch_size * scale))
                dem_patch = dem_arr[dr: dr + ds, dc: dc + ds]
                valid = dem_patch[np.isfinite(dem_patch)]
                roughness = float(np.std(valid)) if valid.size > 1 else 0.0

            records.append({
                "px_row":            pr,
                "px_col":            pc,
                "lulc_class":        lulc_class,
                "lulc_name":         LULC_NAMES.get(lulc_class, "unknown"),
                "terrain_roughness": roughness,
                "ndvi_var":          ndvi_var_tile,
                "wc_source":         wc_source,
            })

    if skipped:
        log.info(f"      Skipped {skipped} bad patches "
                 f"(valid_frac < {min_valid_frac}, total={len(records)+skipped})")
    return pd.DataFrame(records)


# --- Stage-1 tile scoring ----------------------------------------------------

def _tile_ndvi_variance(tile_items) -> float:
    veg_pcts = [it.properties.get("s2:vegetation_percentage") for it in tile_items]
    veg_pcts = [v for v in veg_pcts if v is not None]
    return float(np.var(veg_pcts)) if len(veg_pcts) >= 2 else 0.0


# --- Selection ---------------------------------------------------------------

def stratified_select(
    candidates: pd.DataFrame,
    n_total: int,
    target_fracs: dict,
    min_per_class: int = 0,
    geo_bin_deg: float = 5.0,
    max_per_geo_bin: int = None,
    alpha: float = 0.6,
    beta: float = 0.4,
) -> pd.DataFrame:
    """
    LULC-stratified selection with geographic diversity cap.
    Per-class quota = max(min_per_class, round(n_total * frac)).
    """
    df = candidates.copy()
    for col in ("ndvi_var", "terrain_roughness"):
        mx = df[col].max()
        df[f"{col}_norm"] = df[col] / mx if mx > 0 else 0.0
    df["priority"] = alpha * df["ndvi_var_norm"] + beta * df["terrain_roughness_norm"]

    if max_per_geo_bin is None:
        max_per_geo_bin = max(1, n_total // 20)
    df["geo_bin"] = (
        (df["lat"] // geo_bin_deg).astype(int).astype(str) + "_" +
        (df["lon"] // geo_bin_deg).astype(int).astype(str)
    )

    selected = []
    log.info("\nStratified selection (min_per_class=%d):", min_per_class)
    for lulc_class, frac in sorted(target_fracs.items()):
        n_class = max(min_per_class, int(round(n_total * frac)))
        sub = df[df["lulc_class"] == lulc_class]
        if sub.empty:
            log.warning(f"  LULC {lulc_class:3d} ({LULC_NAMES.get(lulc_class,'?'):12s}): "
                        f"no candidates — skipped")
            continue
        geo_capped = (
            sub.sort_values("priority", ascending=False)
               .groupby("geo_bin", group_keys=False)
               .head(max_per_geo_bin)
        )
        chosen = geo_capped.nlargest(n_class, "priority")
        selected.append(chosen)
        log.info(f"  LULC {lulc_class:3d} ({LULC_NAMES.get(lulc_class,'?'):12s}): "
                 f"target={n_class:4d}  available={len(sub):6d}  selected={len(chosen):4d}")

    if not selected:
        return pd.DataFrame()
    result = pd.concat(selected, ignore_index=True)
    log.info(f"\nTotal selected: {len(result)} patches\n")
    return result


def select_top_n(candidates: pd.DataFrame, needs: dict[int, int]) -> pd.DataFrame:
    """
    Rare-mode selection: take top-priority patches up to needs[lulc_class] per class.
    """
    df = candidates.copy()
    for col in ("ndvi_var", "terrain_roughness"):
        mx = df[col].max()
        df[f"{col}_norm"] = df[col] / mx if mx > 0 else 0.0
    df["priority"] = 0.6 * df["ndvi_var_norm"] + 0.4 * df["terrain_roughness_norm"]

    parts = []
    log.info("\nRare-class top-N selection:")
    for lulc_class, n_needed in sorted(needs.items()):
        pool = df[df["lulc_class"] == lulc_class]
        chosen = pool.nlargest(n_needed, "priority")
        log.info(f"  LULC {lulc_class:3d} ({LULC_NAMES.get(lulc_class,'?'):12s}): "
                 f"need={n_needed:4d}  available={len(pool):5d}  selected={len(chosen):4d}")
        parts.append(chosen)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _count_existing(existing_csv: str) -> dict[int, int]:
    if not Path(existing_csv).exists():
        return {}
    df = pd.read_csv(existing_csv)
    return {int(k): int(v) for k, v in df.groupby("lulc_class").size().items()}


# --- Inner processing loop (shared by both modes) ----------------------------

def _process_bboxes(
    bboxes: list,
    catalog,
    start_date: str,
    end_date: str,
    patch_size: int,
    candidate_stride: int,
    border_px: int,
    min_valid_frac: float,
) -> pd.DataFrame:
    all_candidates = []

    for bbox in bboxes:
        lon_c = (bbox[0] + bbox[2]) / 2
        lat_c = (bbox[1] + bbox[3]) / 2
        log.info(f"\n{'─'*60}")
        log.info(f"BBox {bbox}  (center {lat_c:.1f}N {lon_c:.1f}E)")

        s2_items = list(catalog.search(
            collections=[S2_COLLECTION],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
            query={"eo:cloud_cover": {"lt": 80}},
        ).items())
        if not s2_items:
            log.warning("  No S2 scenes — skipping bbox")
            continue

        by_tile = defaultdict(list)
        for it in s2_items:
            tid = it.properties.get("s2:mgrs_tile", it.id[:5])
            by_tile[tid].append(it)

        wc_items = list(catalog.search(
            collections=[WORLDCOVER_COLLECTION], bbox=bbox).items())
        wc_item = wc_items[0] if wc_items else None
        if wc_item is None:
            log.warning("  WorldCover unavailable — all patches in bbox will be stac_fallback")

        dem_items = list(catalog.search(
            collections=[DEM_COLLECTION], bbox=bbox).items())
        dem_item = dem_items[0] if dem_items else None

        for tile_id, tile_items in by_tile.items():
            tile_items = sorted(tile_items, key=lambda x: x.datetime)
            log.info(f"  Tile {tile_id}: {len(tile_items)} scenes")

            try:
                tile_h, tile_w = _probe_tile(tile_items[0])
            except Exception as e:
                log.warning(f"    Cannot probe tile {tile_id}: {e}")
                continue

            ndvi_var_tile = _tile_ndvi_variance(tile_items)

            wc_arr = None
            if wc_item is not None:
                try:
                    wc_arr = _read_full_band(wc_item, "map").astype(np.uint8)
                    if wc_arr.shape != (tile_h, tile_w):
                        from scipy.ndimage import zoom
                        wc_arr = zoom(wc_arr,
                                      (tile_h / wc_arr.shape[0], tile_w / wc_arr.shape[1]),
                                      order=0)
                    # Tile-level no-data check: skip tile if mostly 0/NaN
                    tile_valid_frac = np.sum(wc_arr > 0) / wc_arr.size
                    if tile_valid_frac < 0.1:
                        log.warning(f"    WorldCover tile {tile_id} nearly empty "
                                    f"(valid={tile_valid_frac:.1%}) — skipping tile")
                        continue
                except Exception as e:
                    log.warning(f"    WorldCover read failed: {e}")

            fallback_lulc = _stac_lulc_estimate(tile_items) if wc_arr is None else 60
            if wc_arr is None:
                log.info(f"    STAC fallback LULC: "
                         f"{fallback_lulc} ({LULC_NAMES.get(fallback_lulc, '?')})")

            dem_arr = None
            if dem_item is not None:
                try:
                    dem_arr = _read_full_band(dem_item, "data")
                except Exception as e:
                    log.warning(f"    DEM read failed: {e}")

            tile_df = _extract_patch_features(
                tile_items, wc_arr, dem_arr,
                tile_h, tile_w,
                patch_size, candidate_stride, border_px,
                ndvi_var_tile,
                fallback_lulc=fallback_lulc,
                min_valid_frac=min_valid_frac,
            )
            tile_df["tile_id"]    = tile_id
            tile_df["lat"]        = lat_c
            tile_df["lon"]        = lon_c
            tile_df["query_bbox"] = json.dumps(list(bbox))

            log.info(f"    {len(tile_df)} candidate patches scored")
            all_candidates.append(tile_df)

    if not all_candidates:
        return pd.DataFrame()
    return pd.concat(all_candidates, ignore_index=True)


# --- Main pipeline ------------------------------------------------------------

def build_patch_list(
    bboxes: list,
    start_date: str,
    end_date: str,
    n_patches: int = 33000,
    target_fracs: dict = None,
    min_per_class: int = 3000,
    patch_size: int = 384,
    candidate_stride: int = 768,
    border_px: int = 192,
    min_valid_frac: float = 0.7,
    out_csv: str = "selected_patches.csv",
) -> pd.DataFrame:
    """Full-mode: scan bboxes → stratified selection."""
    target_fracs = target_fracs or DEFAULT_TARGET_FRACS

    catalog = pystac_client.Client.open(PLANETARY_COMPUTER_URL)

    candidates_df = _process_bboxes(
        bboxes, catalog, start_date, end_date,
        patch_size, candidate_stride, border_px, min_valid_frac)

    if candidates_df.empty:
        log.error("No candidate patches found.")
        return pd.DataFrame()

    log.info(f"\nTotal candidates: {len(candidates_df)}")
    selected = stratified_select(candidates_df, n_patches, target_fracs,
                                 min_per_class=min_per_class)

    _save(selected, out_csv)
    return selected


def build_rare_patch_list(
    existing_csv: str,
    target: int = 3000,
    threshold: int = 1000,
    start_date: str = "2022-01-01",
    end_date: str = "2023-12-31",
    patch_size: int = 384,
    candidate_stride: int = 768,
    border_px: int = 192,
    min_valid_frac: float = 0.7,
    out_csv: str = "selected_patches_rare.csv",
) -> pd.DataFrame:
    """Rare-mode: top up classes below threshold using dedicated 2°×2° bboxes."""
    existing_counts = _count_existing(existing_csv)
    log.info("Existing counts: %s", existing_counts)

    needs: dict[int, int] = {}
    for lulc_class in RARE_BBOXES:
        current = existing_counts.get(lulc_class, 0)
        if current < threshold:
            needs[lulc_class] = target - current
            log.info(f"  LULC {lulc_class:3d} ({LULC_NAMES.get(lulc_class,'?'):12s}): "
                     f"current={current}  need {needs[lulc_class]} more")

    if not needs:
        log.info("All classes already meet threshold — nothing to do.")
        return pd.DataFrame()

    catalog = pystac_client.Client.open(PLANETARY_COMPUTER_URL)

    # Collect bboxes for all classes that need topping up
    bboxes_flat = []
    for lulc_class in needs:
        bboxes_flat.extend(RARE_BBOXES[lulc_class])
    bboxes_flat = list(dict.fromkeys(bboxes_flat))  # deduplicate, preserve order

    candidates_df = _process_bboxes(
        bboxes_flat, catalog, start_date, end_date,
        patch_size, candidate_stride, border_px, min_valid_frac)

    if candidates_df.empty:
        log.error("No candidate patches found.")
        return pd.DataFrame()

    log.info(f"\nTotal candidates: {len(candidates_df)}")
    log.info("Candidate class distribution:\n%s",
             candidates_df.groupby(["lulc_class", "lulc_name"]).size().to_string())

    # Opportunistically fill any other rare class found in the candidates
    for lulc_class in candidates_df["lulc_class"].unique():
        lulc_class = int(lulc_class)
        if lulc_class not in needs:
            current = existing_counts.get(lulc_class, 0)
            if current < threshold:
                needs[lulc_class] = target - current

    selected = select_top_n(candidates_df, needs)
    _save(selected, out_csv)
    return selected


def _save(df: pd.DataFrame, out_csv: str) -> None:
    keep = ["tile_id", "px_row", "px_col", "lat", "lon",
            "lulc_class", "lulc_name", "terrain_roughness", "ndvi_var",
            "priority", "wc_source", "query_bbox"]
    df = df[[c for c in keep if c in df.columns]]
    df.to_csv(out_csv, index=False)
    log.info(f"Saved {len(df)} patches → {out_csv}")


# --- CLI ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stratified S2 patch sampler")
    parser.add_argument("--mode", choices=["full", "rare"], default="full",
                        help="full: scan DEFAULT_BBOXES; rare: top up rare classes")
    parser.add_argument("--existing_csv", default="selected_patches.csv",
                        help="[rare mode] existing CSV to read current counts from")
    parser.add_argument("--threshold", type=int, default=1000,
                        help="[rare mode] classes below this count are topped up")
    parser.add_argument("--bboxes_json", type=str, default=None,
                        help="[full mode] JSON file with [[lon_min,lat_min,lon_max,lat_max],...]")
    parser.add_argument("--start",           default="2022-01-01")
    parser.add_argument("--end",             default="2023-12-31")
    parser.add_argument("--n_patches",       type=int, default=33000)
    parser.add_argument("--min_per_class",   type=int, default=3000)
    parser.add_argument("--target",          type=int, default=3000,
                        help="[rare mode] target patches per class after top-up")
    parser.add_argument("--patch_size",      type=int, default=384)
    parser.add_argument("--candidate_stride",type=int, default=768)
    parser.add_argument("--min_valid_frac",  type=float, default=0.7,
                        help="Min fraction of valid WorldCover pixels to keep a patch")
    parser.add_argument("--out_csv",         default=None,
                        help="Output CSV path (default: selected_patches.csv or "
                             "selected_patches_rare.csv depending on mode)")
    parser.add_argument("--proxy", action="store_true",
                        help="Enable HTTP/SOCKS5 proxy (127.0.0.1:33210/33211)")
    args = parser.parse_args()

    if args.proxy:
        configure_proxy()

    if args.mode == "full":
        if args.bboxes_json:
            with open(args.bboxes_json) as f:
                bboxes = [tuple(b) for b in json.load(f)]
        else:
            bboxes = DEFAULT_BBOXES
        build_patch_list(
            bboxes=bboxes,
            start_date=args.start,
            end_date=args.end,
            n_patches=args.n_patches,
            min_per_class=args.min_per_class,
            patch_size=args.patch_size,
            candidate_stride=args.candidate_stride,
            min_valid_frac=args.min_valid_frac,
            out_csv=args.out_csv or "selected_patches.csv",
        )
    else:
        build_rare_patch_list(
            existing_csv=args.existing_csv,
            target=args.target,
            threshold=args.threshold,
            start_date=args.start,
            end_date=args.end,
            patch_size=args.patch_size,
            candidate_stride=args.candidate_stride,
            min_valid_frac=args.min_valid_frac,
            out_csv=args.out_csv or "selected_patches_rare.csv",
        )
