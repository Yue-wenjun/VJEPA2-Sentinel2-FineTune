"""
Sentinel-2 Dataset compatible with V-JEPA 2.1 training pipeline.

Drop-in replacement for VideoDataset:
  - Reads .npy patch files saved by download_s2.py  [H, W, 6]
  - Assembles temporal sequences                    [T, H, W, 6]
  - Injects DOY encoding as additive positional signal
  - Returns ([buffer], label, clip_indices, doy_tensor) — same as VideoDataset plus doy_tensor

Band order in saved .npy files: B02 B03 B04 B08 B11 B12
Band statistics (per-band mean/std over S2 L2A corpus, reflectance 0-1):

    Band   mean    std
    B02    0.0850  0.0574
    B03    0.0950  0.0521
    B04    0.1001  0.0660
    B08    0.2841  0.1076
    B11    0.2260  0.1102
    B12    0.1546  0.0900
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Per-band normalization stats (reflectance space, 6 bands)
S2_MEAN = torch.tensor([0.0850, 0.0950, 0.1001, 0.2841, 0.2260, 0.1546])
S2_STD  = torch.tensor([0.0574, 0.0521, 0.0660, 0.1076, 0.1102, 0.0900])

N_BANDS = 6
DOY_DIM = 4   # sin/cos at two frequencies


def doy_encoding(doy: int) -> torch.Tensor:
    """Day-of-year → [sin(2π/365·d), cos(2π/365·d), sin(4π/365·d), cos(4π/365·d)]"""
    d = float(doy)
    return torch.tensor([
        math.sin(2 * math.pi * d / 365),
        math.cos(2 * math.pi * d / 365),
        math.sin(4 * math.pi * d / 365),
        math.cos(4 * math.pi * d / 365),
    ], dtype=torch.float32)


class Sentinel2Dataset(Dataset):
    """
    Temporal patch dataset for V-JEPA 2.1 fine-tuning.

    Args:
        sequences_csv:  path to sequences.csv produced by download_s2.py
        frames_per_clip: number of time steps per training clip (T)
        random_clip:    randomly subsample T frames from available sequence
        crop_size:      spatial crop size (H = W)
        random_flip:    horizontal flip augmentation
        normalize:      apply per-band S2 normalization
    """

    def __init__(
        self,
        sequences_csv: str,
        base_dir: str = "",
        frames_per_clip: int = 8,
        random_clip: bool = True,
        crop_size: int = 384,
        random_flip: bool = True,
        normalize: bool = True,
        cloud_cats: list | None = None,
    ):
        """
        cloud_cats: list of allowed cloud categories (0=clean, 1=moderate, 2=cloudy, -1=SCL unavailable).
                    None = no filter (use all frames including cat=-1).
                    Example: [0] = only clean frames, [0,1] = clean+moderate (excludes cloudy and SCL-unknown).
        """
        self.frames_per_clip = frames_per_clip
        self.random_clip = random_clip
        self.crop_size = crop_size
        self.random_flip = random_flip
        self.normalize = normalize
        _allowed = set(cloud_cats) if cloud_cats is not None else None  # None = no filter
        # CSV stores relative paths; base_dir (or sequences_csv's parent) resolves them
        _base = Path(base_dir) if base_dir else Path(sequences_csv).parent

        df = pd.read_csv(sequences_csv)
        self.sequences = []
        for _, row in df.iterrows():
            paths = [str(_base / p) for p in row["frame_paths"].split(",")]
            doys  = list(map(int, str(row["doys"]).split(",")))
            if _allowed is not None and "cloud_cats" in df.columns:
                cats = list(map(int, str(row["cloud_cats"]).split(",")))
                valid = [(p, d) for p, d, c in zip(paths, doys, cats) if c in _allowed]
                if not valid:
                    continue
                paths, doys = zip(*valid)
                paths, doys = list(paths), list(doys)
            if len(paths) >= frames_per_clip:
                self.sequences.append({"paths": paths, "doys": doys})

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        seq = self.sequences[index]
        paths, doys = seq["paths"], seq["doys"]
        n = len(paths)

        # Sample T frame indices
        if self.random_clip:
            start = np.random.randint(0, n - self.frames_per_clip + 1)
            indices = list(range(start, start + self.frames_per_clip))
        else:
            step = max(1, n // self.frames_per_clip)
            indices = list(range(0, n, step))[: self.frames_per_clip]

        frames, clip_doys = [], []
        for i in indices:
            arr = np.load(paths[i])       # [H, W, 6] float32
            frames.append(arr)
            clip_doys.append(doys[i])

        buffer = np.stack(frames, axis=0)   # [T, H, W, 6]
        buffer = torch.from_numpy(buffer)   # [T, H, W, 6]

        # Random crop
        T, H, W, C = buffer.shape
        if H > self.crop_size:
            top  = np.random.randint(0, H - self.crop_size)
            left = np.random.randint(0, W - self.crop_size)
        else:
            top, left = 0, 0
        buffer = buffer[:, top:top+self.crop_size, left:left+self.crop_size, :]

        # Random horizontal flip (same flip across all frames)
        if self.random_flip and np.random.rand() < 0.5:
            buffer = torch.flip(buffer, dims=[2])

        # [T H W C] → [C T H W] to match V-JEPA convention
        buffer = buffer.permute(3, 0, 1, 2).float()   # [6, T, H, W]

        # Per-band normalization
        if self.normalize:
            mean = S2_MEAN.view(N_BANDS, 1, 1, 1)
            std  = S2_STD.view(N_BANDS, 1, 1, 1)
            buffer = (buffer - mean) / std

        label = 0
        clip_indices = [np.array(indices, dtype=np.int64)]
        doy_tensor = torch.tensor([clip_doys[i] for i in range(len(indices))], dtype=torch.int32)
        return [buffer], label, clip_indices, doy_tensor

    def get_doy_encodings(self, index) -> torch.Tensor:
        """Returns DOY encodings [T, DOY_DIM] for the clip at index (for positional injection)."""
        seq = self.sequences[index]
        _, doys = seq["paths"], seq["doys"]
        return torch.stack([doy_encoding(d) for d in doys[: self.frames_per_clip]])


def make_sentinel2_dataloader(
    sequences_csv: str,
    batch_size: int,
    base_dir: str = "",
    cloud_cats: list | None = None,
    frames_per_clip: int = 8,
    crop_size: int = 384,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 4,
    collator=None,
    drop_last: bool = True,
    pin_mem: bool = True,
):
    dataset = Sentinel2Dataset(
        sequences_csv=sequences_csv,
        base_dir=base_dir,
        cloud_cats=cloud_cats,
        frames_per_clip=frames_per_clip,
        crop_size=crop_size,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )
    return dataset, loader, sampler
