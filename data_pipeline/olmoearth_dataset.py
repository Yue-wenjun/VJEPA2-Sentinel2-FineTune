"""
OLMo-Earth Dataset wrapper for V-JEPA 2.1 fine-tuning.

Source: allenai/olmoearth_pretrain_dataset (local TAR files, webdataset format)
        285,288 samples × 12 monthly Sentinel-2 composites, global coverage.

Channel layout — depends on which sub-folder you downloaded:

  ``10_sentinel2_l2a_monthly`` (10 m bands only):
      4 bands per timestep × 12 months = 48 channels total.
      Band order: B02(0)  B03(1)  B04(2)  B08(3)
      → set n_bands_per_timestep=4  (default)

  ``20_sentinel2_l2a_monthly`` (20 m bands resampled to 10 m):
      Adds B05 B06 B07 B8A B11 B12 — not yet supported by this loader.
      To get 6 bands (B02/B03/B04/B08/B11/B12), you need both folders.

Spatial: native 2560 m tile = 256×256 px at 10 m; train with 256 px crops.
Temporal: T=12 fixed monthly composites; DOY = mid-month calendar value.

Returns (matches Sentinel2Dataset convention for MaskCollator):
    ([buffer], label, doy_tensor, clip_indices)
    buffer:       [N_BANDS, 12, crop, crop] float32 z-scored
    doy_tensor:   [12] int32  (fixed mid-month DOY)
    clip_indices: [np.arange(12)]   — for MaskCollator fpc detection
"""

from __future__ import annotations

import glob as _glob
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

# ── normalization stats (reflectance space, 0–1) ─────────────────────────────
# 4-band: B02 B03 B04 B08
_MEAN_4 = torch.tensor([0.0850, 0.0950, 0.1001, 0.2841])
_STD_4  = torch.tensor([0.0574, 0.0521, 0.0660, 0.1076])
# 6-band: B02 B03 B04 B08 B11 B12
_MEAN_6 = torch.tensor([0.0850, 0.0950, 0.1001, 0.2841, 0.2260, 0.1546])
_STD_6  = torch.tensor([0.0574, 0.0521, 0.0660, 0.1076, 0.1102, 0.0900])

_NORM_STATS: dict[int, tuple[torch.Tensor, torch.Tensor]] = {4: (_MEAN_4, _STD_4), 6: (_MEAN_6, _STD_6)}

N_MONTHS  = 12
NATIVE_PX = 256          # 2560 m / 10 m/px

_MISSING = -99999.0      # OLMo-Earth sentinel for missing pixels

# Mid-month day-of-year (calendar month index 0=Jan … 11=Dec)
_MONTHLY_DOYS = [15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]
_DOY_TENSOR   = torch.tensor(_MONTHLY_DOYS, dtype=torch.int32)
_CLIP_INDICES = [np.arange(N_MONTHS, dtype=np.int64)]


class OLMoEarthDataset(IterableDataset):
    """
    Streaming PyTorch IterableDataset over local OLMo-Earth TAR shards.

    Reads webdataset-format TAR files produced by allenai/olmoearth_pretrain_dataset.
    Multi-worker DataLoader is handled automatically via webdataset's split_by_worker.

    Args:
        tar_path:             Glob pattern or list of paths to local .tar files.
                              e.g. "/data/olmoearth/10_sentinel2_l2a_monthly/*.tar"
        n_bands_per_timestep: Bands stored per monthly timestep in each GeoTIFF.
                              4 for ``10_sentinel2_l2a_monthly`` (B02/B03/B04/B08).
        band_indices:         Which bands to extract from each timestep.
                              None = all bands (0 … n_bands_per_timestep-1).
        crop_size:            Spatial crop in px; must be ≤ NATIVE_PX (256).
        random_flip:          Random horizontal flip augmentation.
        normalize:            Apply per-band S2 z-score normalization.
        dn_scale:             Divide raw pixel values by this to convert to reflectance.
                              Run inspect_sample() to verify before training.
        max_missing_frac:     Skip sample if fraction of MISSING pixels exceeds this.
        shuffle_buffer:       webdataset shuffle buffer size (samples).
        seed:                 Base RNG seed (each worker uses seed + worker_id).
    """

    def __init__(
        self,
        tar_path: str | Sequence[str],
        n_bands_per_timestep: int = 4,
        band_indices: list[int] | None = None,
        crop_size: int = 256,
        random_flip: bool = True,
        normalize: bool = True,
        dn_scale: float = 10000.0,
        max_missing_frac: float = 0.10,
        shuffle_buffer: int = 1000,
        seed: int = 42,
        tif_key: str = "tif",        # webdataset key for the GeoTIFF bytes; verify with inspect_sample()
    ):
        super().__init__()
        if isinstance(tar_path, (list, tuple)):
            self.tar_files = list(tar_path)
        elif "*" in str(tar_path) or "?" in str(tar_path):
            self.tar_files = sorted(_glob.glob(str(tar_path)))
        else:
            self.tar_files = [str(tar_path)]

        if not self.tar_files:
            raise FileNotFoundError(f"No TAR files found: {tar_path}")

        self.n_bands_per_timestep = n_bands_per_timestep
        self.band_indices = list(band_indices) if band_indices is not None else list(range(n_bands_per_timestep))
        self.n_out_bands  = len(self.band_indices)
        self.crop_size    = crop_size
        self.random_flip  = random_flip
        self.normalize    = normalize
        self.dn_scale     = dn_scale
        self.max_missing_frac = max_missing_frac
        self.shuffle_buffer   = shuffle_buffer
        self.seed             = seed
        self.tif_key          = tif_key

        if self.n_out_bands in _NORM_STATS:
            self.s2_mean, self.s2_std = _NORM_STATS[self.n_out_bands]
        else:
            # Tile 4-band stats to cover arbitrary band counts
            m, s = _NORM_STATS[4]
            idx = torch.arange(self.n_out_bands) % 4
            self.s2_mean, self.s2_std = m[idx], s[idx]

    # ── streaming iterator ───────────────────────────────────────────────────

    def __iter__(self) -> Iterator:
        import webdataset as wds

        worker_info = torch.utils.data.get_worker_info()
        worker_id   = worker_info.id if worker_info is not None else 0
        rng = np.random.default_rng(self.seed + worker_id)

        ds = (
            wds.WebDataset(
                self.tar_files,
                shardshuffle=True,
                nodesplitter=wds.split_by_worker,
            )
            .shuffle(self.shuffle_buffer)
        )

        for sample in ds:
            result = self._process(sample, rng)
            if result is not None:
                yield result

    # ── per-sample processing ────────────────────────────────────────────────

    def _process(self, sample: dict, rng: np.random.Generator):
        tif_bytes = sample.get(self.tif_key)
        if tif_bytes is None:
            return None
        try:
            import rasterio
            from rasterio.io import MemoryFile
            with MemoryFile(tif_bytes) as mf, mf.open() as src:
                arr = src.read().astype(np.float32)   # [C_total, H, W]
        except Exception:
            return None

        expected_ch = N_MONTHS * self.n_bands_per_timestep
        if arr.shape[0] != expected_ch:
            return None

        # ── reshape & band extraction ─────────────────────────────────────
        _, H, W = arr.shape
        arr = arr.reshape(N_MONTHS, self.n_bands_per_timestep, H, W)  # [T, C_all, H, W]
        arr = arr[:, self.band_indices, :, :]                          # [T, N_BANDS, H, W]

        # ── missing-value guard ───────────────────────────────────────────
        missing_mask = arr == _MISSING
        if missing_mask.mean() > self.max_missing_frac:
            return None
        arr[missing_mask] = 0.0

        # ── DN → reflectance ──────────────────────────────────────────────
        if self.dn_scale != 1.0:
            arr = arr / self.dn_scale
        arr = np.clip(arr, 0.0, 1.0)

        # ── random spatial crop ───────────────────────────────────────────
        if H > self.crop_size:
            top  = rng.integers(0, H - self.crop_size)
            left = rng.integers(0, W - self.crop_size)
            arr  = arr[:, :, top:top + self.crop_size, left:left + self.crop_size]

        # ── random horizontal flip ────────────────────────────────────────
        if self.random_flip and rng.random() < 0.5:
            arr = arr[:, :, :, ::-1].copy()

        # ── [T, C, H, W] → [C, T, H, W] (V-JEPA convention) ─────────────
        arr = arr.transpose(1, 0, 2, 3).copy()
        buffer = torch.from_numpy(arr)

        # ── per-band z-score normalization ────────────────────────────────
        if self.normalize:
            mean = self.s2_mean.view(self.n_out_bands, 1, 1, 1)
            std  = self.s2_std.view(self.n_out_bands, 1, 1, 1)
            buffer = (buffer - mean) / std

        # clip_indices must be last: MaskCollator detects fpc via len(sample[-1][-1])
        return [buffer], 0, _DOY_TENSOR.clone(), [_CLIP_INDICES[0].copy()]


# ── factory ───────────────────────────────────────────────────────────────────

def make_olmoearth_dataloader(
    tar_path: str,
    batch_size: int,
    n_bands_per_timestep: int = 4,
    crop_size: int = 256,
    dn_scale: float = 10000.0,
    num_workers: int = 4,
    collator=None,
    drop_last: bool = True,
    pin_mem: bool = True,
) -> DataLoader:
    dataset = OLMoEarthDataset(
        tar_path=tar_path,
        n_bands_per_timestep=n_bands_per_timestep,
        crop_size=crop_size,
        dn_scale=dn_scale,
    )
    return DataLoader(
        dataset,
        collate_fn=collator,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )


# ── debug helper ──────────────────────────────────────────────────────────────

def inspect_sample(tar_path: str) -> None:
    """
    Load ONE sample from a local TAR file and print shape, value range, and
    missing-pixel count. Use this to verify dn_scale and n_bands_per_timestep.

    Run:
        python -c "from data_pipeline.olmoearth_dataset import inspect_sample; inspect_sample('/data/olmoearth/*.tar')"
    """
    import webdataset as wds
    import rasterio
    from rasterio.io import MemoryFile

    tar_files = sorted(_glob.glob(tar_path)) if ("*" in tar_path or "?" in tar_path) else [tar_path]
    if not tar_files:
        print(f"No TAR files found: {tar_path}"); return

    print(f"Found {len(tar_files)} TAR file(s). Reading first sample from: {tar_files[0]}")
    ds = wds.WebDataset(tar_files[:1])

    try:
        sample = next(iter(ds))
    except StopIteration:
        print("TAR file is empty."); return

    print("Sample keys:", [k for k in sample.keys() if not k.startswith("__")])
    print("__key__:", sample.get("__key__"))

    tif_bytes = sample.get("tif")
    if tif_bytes is None:
        print("No 'tif' key — check column names above"); return

    with MemoryFile(tif_bytes) as mf, mf.open() as src:
        arr  = src.read().astype(np.float32)
        meta = src.meta

    print(f"\nGeoTIFF shape (C, H, W): {arr.shape}")
    print(f"dtype: {meta['dtype']}   CRS: {meta.get('crs')}")

    n_ch = arr.shape[0]
    for bpt in [4, 6, 12]:
        if n_ch % N_MONTHS == 0 and n_ch // N_MONTHS == bpt:
            print(f"\n→ {n_ch} channels = {N_MONTHS} months × {bpt} bands/month")
            print(f"  Set n_bands_per_timestep={bpt} in dataset / YAML")

    valid = arr[arr != _MISSING]
    if valid.size > 0:
        vmin, vmax = valid.min(), valid.max()
        print(f"\nValue range (excl. MISSING): [{vmin:.4f}, {vmax:.4f}]")
        if vmax > 10.0:
            print("→ DN range detected — use dn_scale=10000.0")
        elif vmax <= 1.01:
            print("→ Reflectance range detected — use dn_scale=1.0")
        else:
            print(f"→ Ambiguous range ({vmax:.1f}). Check data documentation.")
    else:
        print("\nAll pixels are MISSING — check TAR contents.")

    print(f"Missing fraction: {(arr == _MISSING).mean():.3%}")
