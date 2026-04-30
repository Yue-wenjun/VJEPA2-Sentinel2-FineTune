"""
linear_probe.py — Frozen linear probe: EuroSAT-MS and BreizhCrops

Extracts frozen encoder features (mean-pool over all tokens), caches them
to disk as .npz files, then fits sklearn LogisticRegression and reports
top-1 accuracy + per-class report.

Dependencies:
    pip install torchgeo breizhcrops scikit-learn

Usage:
    python linear_probe.py \\
        --config  vjepa2/configs/finetune/vitl16/olmoearth-256px-12f.yaml \\
        --checkpoint /home/baai/vjepa2/checkpoints/checkpoint_final.pth \\
        [--dataset eurosat|breizhcrops|both]  (default: both) \\
        [--data_dir  /home/baai/data] \\
        [--output_dir ./probe_results] \\
        [--batch_size 16] \\
        [--no_cache]   # re-extract even if .npz already exists

BreizhCrops note:
    Parcel-level time series are tiled to a uniform spatial grid [4,12,256,256].
    This tests temporal feature quality; spatial features are not evaluated.
    Train regions: frh01+frh02+frh03  |  Test region: frh04  (standard split).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent / "vjepa2"))

import app.vjepa_2_1.models.vision_transformer as video_vit
from app.vjepa_2_1.wrappers import MultiSeqWrapper
from data_pipeline.patch_embed_6ch import build_nch_patch_embed_from_pretrained

# ── normalization stats (same as OLMo-Earth training) ────────────────────────
_MEAN4 = torch.tensor([0.0850, 0.0950, 0.1001, 0.2841])   # B02 B03 B04 B08
_STD4  = torch.tensor([0.0574, 0.0521, 0.0660, 0.1076])
_DOYS  = [15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]  # mid-month


# ── encoder utilities (mirrors finetune_main.py / visualize.py) ───────────────

def _strip_prefix(sd: dict) -> dict:
    if not sd:
        return sd
    key = next(iter(sd))
    for pfx in ("module.backbone.", "module.", "backbone."):
        if key.startswith(pfx):
            return {k[len(pfx):]: v for k, v in sd.items()}
    return sd


def _safe_load(module, state_dict: dict):
    own = module.state_dict()
    compat = {k: v for k, v in state_dict.items()
              if k not in own or own[k].shape == v.shape}
    module.load_state_dict(compat, strict=False)


def build_frozen_encoder(cfg: dict, ckpt_path: str, device: torch.device) -> MultiSeqWrapper:
    """Build encoder, load fine-tuned weights, freeze all parameters."""
    m, d = cfg["model"], cfg["data"]
    backbone = video_vit.__dict__[m["model_name"]](
        img_size=d["crop_size"],
        patch_size=d["patch_size"],
        num_frames=d["frames_per_clip"],
        tubelet_size=d["tubelet_size"],
        in_chans=m["in_chans"],
        use_doy_encoding=m.get("use_doy_encoding", True),
        use_rope=m.get("use_rope", False),
        uniform_power=m.get("uniform_power", True),
        use_sdpa=m.get("use_sdpa", True),
        use_activation_checkpointing=False,
        modality_embedding=m.get("modality_embedding", False),
        has_cls_first=m.get("has_cls_first", False),
        n_registers=m.get("n_registers", 0),
    )
    encoder = MultiSeqWrapper(backbone).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = _strip_prefix(ckpt.get("encoder", ckpt))
    _safe_load(encoder.backbone, state)
    epoch = ckpt.get("epoch", "?")
    print(f"  Checkpoint loaded (epoch={epoch}): {ckpt_path}")

    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()
    return encoder


# ── EuroSAT-MS dataset ────────────────────────────────────────────────────────

class EuroSATProbeDataset(Dataset):
    """
    Wraps torchgeo's EuroSAT (13-band, 64×64px).

    Selects B02/B03/B04/B08, resizes to 256×256, repeats across T=12
    monthly time steps with real mid-month DOY values.
    Output: (image [4,12,256,256], label int, doys [12])
    """

    TARGET_BANDS = ("B02", "B03", "B04", "B08")

    def __init__(self, root: str, split: str, download: bool = True):
        try:
            from torchgeo.datasets import EuroSAT
        except ImportError:
            raise ImportError("pip install torchgeo")

        # Request all 13 bands; we'll select 4 afterwards.
        self._ds = EuroSAT(
            root=root,
            split=split,
            download=download,
            bands=EuroSAT.all_band_names,
        )
        # Resolve band indices once
        all_bands = list(EuroSAT.all_band_names)
        self._idx = [all_bands.index(b) for b in self.TARGET_BANDS]
        self._doys = torch.tensor(_DOYS, dtype=torch.int32)

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, i):
        sample = self._ds[i]
        img = sample["image"]          # [13, 64, 64] float32 — torchgeo returns DN
        label = int(sample["label"])

        # Select 4 bands and convert DN → reflectance
        img = img[self._idx] / 10000.0   # [4, 64, 64]
        img = img.clamp(0.0, 1.0)

        # Resize 64 → 256 (bilinear)
        img = F.interpolate(img.unsqueeze(0), size=(256, 256),
                            mode="bilinear", align_corners=False)[0]   # [4, 256, 256]

        # Z-score normalisation (same as OLMo-Earth training)
        mean = _MEAN4.view(4, 1, 1)
        std  = _STD4.view(4, 1, 1)
        img  = (img - mean) / std

        # Stack T=12 copies → [4, 12, 256, 256]
        img = img.unsqueeze(1).expand(-1, 12, -1, -1).contiguous()

        return img, label, self._doys.clone()


# ── BreizhCrops dataset ───────────────────────────────────────────────────────

class BreizhCropsProbeDataset(Dataset):
    """
    Wraps breizhcrops for parcel-level monthly Sentinel-2 time series.

    Each parcel is resampled to 12 monthly composites (median per calendar month).
    The spectral vector [4] is tiled spatially to [4, 12, 256, 256].

    NOTE: No real spatial structure — all spatial tokens are identical.
          This specifically tests temporal feature quality.

    Band columns in breizhcrops X: [doy, B2, B3, B4, B5, B6, B7, B8, B8A, ...]
    We select B02=col1, B03=col2, B04=col3, B08=col7.
    """

    BAND_COLS = [1, 2, 3, 7]   # B02, B03, B04, B08 in BreizhCrops X columns

    # Month boundaries (DOY start of each month, leap-year-safe)
    _MONTH_STARTS = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366]

    def __init__(self, regions: list[str], root: str, download: bool = True):
        try:
            from breizhcrops import BreizhCrops as BC
        except ImportError:
            raise ImportError("pip install breizhcrops")

        self._samples = []   # list of (X [T,13+], label)
        for region in regions:
            print(f"  Loading BreizhCrops {region} …")
            ds = BC(region=region, root=root, year=2017,
                    download=download, classmapping="majorcrops")
            for idx in range(len(ds)):
                X, y, _ = ds[idx]
                if isinstance(X, np.ndarray) and len(X) > 0:
                    self._samples.append((X.astype(np.float32), int(y)))

        self.classes = sorted({y for _, y in self._samples})
        # Remap labels to 0-indexed contiguous integers
        self._label_map = {c: i for i, c in enumerate(self.classes)}
        print(f"  BreizhCrops: {len(self._samples)} parcels, "
              f"{len(self.classes)} classes")

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, i):
        X, label = self._samples[i]
        # X: [T_obs, ≥8] columns: [doy, B2, B3, B4, ...]
        doys_obs = X[:, 0]
        bands    = X[:, self.BAND_COLS]   # [T_obs, 4]

        # Resample to 12 monthly composites (median per calendar month)
        monthly = np.zeros((12, 4), dtype=np.float32)
        for m in range(12):
            lo, hi = self._MONTH_STARTS[m], self._MONTH_STARTS[m + 1]
            mask = (doys_obs >= lo) & (doys_obs < hi)
            if mask.sum() > 0:
                monthly[m] = np.median(bands[mask], axis=0)
            else:
                monthly[m] = np.nan

        # Forward-fill missing months (np.nan → carry previous month)
        for m in range(1, 12):
            if np.isnan(monthly[m]).any():
                monthly[m] = monthly[m - 1]
        if np.isnan(monthly[0]).any():
            monthly[0] = monthly[~np.isnan(monthly).any(axis=1)][0] if (
                ~np.isnan(monthly).any(axis=1)).any() else np.zeros(4)

        # DN → reflectance (BreizhCrops stores raw DN 0-10000)
        monthly = np.clip(monthly / 10000.0, 0.0, 1.0)

        # Z-score normalise
        mean = _MEAN4.numpy()
        std  = _STD4.numpy()
        monthly = (monthly - mean[None]) / std[None]   # [12, 4]

        # Tile spectral vector spatially → [4, 12, 256, 256]
        # Each token sees the same spectral value (no spatial structure).
        # The encoder still adds DOY positional encoding per temporal step.
        t_img = torch.from_numpy(monthly).T   # [4, 12]
        t_img = t_img[:, :, None, None].expand(-1, -1, 256, 256).contiguous()

        doys = torch.tensor(_DOYS, dtype=torch.int32)
        return t_img, self._label_map[label], doys


# ── feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(encoder: MultiSeqWrapper,
                     loader: DataLoader,
                     device: torch.device,
                     dtype: torch.dtype,
                     desc: str = "") -> tuple[np.ndarray, np.ndarray]:
    """
    Run frozen encoder over all batches; mean-pool all spatial-temporal tokens.
    Returns (features [N, embed_dim], labels [N]).
    """
    embed_dim = encoder.backbone.embed_dim
    all_feats, all_labels = [], []
    total = len(loader)

    for bi, (imgs, labels, doys) in enumerate(loader, 1):
        imgs  = imgs.to(device, dtype=dtype)    # [B, 4, 12, 256, 256]
        doys  = doys.to(device)                  # [B, 12]

        with torch.autocast(device_type=device.type, dtype=dtype):
            z = encoder([imgs], doys=doys, training_mode=False)[0]
            # z: [B, N_TOK, embed_dim] (N_TOK = 1536 for 256px/T=12/patch=16/tubelet=2)

        # Mean-pool over all spatial-temporal tokens → [B, embed_dim]
        feats = z[:, :, :embed_dim].mean(dim=1).float().cpu().numpy()
        all_feats.append(feats)
        all_labels.append(labels.numpy())

        if bi % 50 == 0 or bi == total:
            print(f"  {desc} [{bi}/{total}]  batch done", end="\r")

    print()
    return np.concatenate(all_feats), np.concatenate(all_labels)


# ── linear probe ─────────────────────────────────────────────────────────────

def run_probe(feats_tr: np.ndarray, labels_tr: np.ndarray,
              feats_te: np.ndarray, labels_te: np.ndarray,
              class_names: list[str], dataset_name: str) -> float:
    """
    StandardScale → LogisticRegression → accuracy + per-class report.
    Returns top-1 test accuracy.
    """
    print(f"\n── {dataset_name} linear probe ──")
    print(f"  Train: {len(labels_tr):,}  Test: {len(labels_te):,}  "
          f"Features: {feats_tr.shape[1]}")

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(feats_tr)
    X_te = scaler.transform(feats_te)

    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        multi_class="multinomial",
        n_jobs=-1,
        verbose=1,
    )
    clf.fit(X_tr, labels_tr)

    preds = clf.predict(X_te)
    acc = accuracy_score(labels_te, preds)
    print(f"\n  Top-1 accuracy: {acc * 100:.2f}%")
    print(classification_report(labels_te, preds, target_names=class_names,
                                 digits=3))
    return acc


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      required=True)
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--dataset",     default="both",
                        choices=["eurosat", "breizhcrops", "both"])
    parser.add_argument("--data_dir",    default="data")
    parser.add_argument("--output_dir",  default="probe_results")
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--no_cache",    action="store_true",
                        help="Re-extract features even if cache exists")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if cfg["meta"].get("dtype") == "bfloat16" else torch.float32
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}  dtype: {dtype}\n")

    # ── build frozen encoder ──────────────────────────────────────────────
    print("Building frozen encoder …")
    encoder = build_frozen_encoder(cfg, args.checkpoint, device)

    results = {}

    # ══ EuroSAT-MS ══════════════════════════════════════════════════════════
    if args.dataset in ("eurosat", "both"):
        print("\n── EuroSAT-MS ──")
        cache_tr = output_dir / "feat_eurosat_train.npz"
        cache_te = output_dir / "feat_eurosat_test.npz"

        if cache_tr.exists() and cache_te.exists() and not args.no_cache:
            print("  Loading cached features …")
            tr = np.load(cache_tr); te = np.load(cache_te)
            feats_tr, labels_tr = tr["feats"], tr["labels"]
            feats_te, labels_te = te["feats"], te["labels"]
        else:
            print("  Building EuroSAT-MS train split …")
            ds_tr = EuroSATProbeDataset(str(data_dir / "eurosat"), "train", download=True)
            ds_te = EuroSATProbeDataset(str(data_dir / "eurosat"), "test",  download=False)
            ldr_tr = DataLoader(ds_tr, batch_size=args.batch_size,
                                shuffle=False, num_workers=4, pin_memory=True)
            ldr_te = DataLoader(ds_te, batch_size=args.batch_size,
                                shuffle=False, num_workers=4, pin_memory=True)

            feats_tr, labels_tr = extract_features(encoder, ldr_tr, device, dtype, "train")
            feats_te, labels_te = extract_features(encoder, ldr_te, device, dtype, "test")

            np.savez(cache_tr, feats=feats_tr, labels=labels_tr)
            np.savez(cache_te, feats=feats_te, labels=labels_te)
            print(f"  Features cached → {cache_tr}, {cache_te}")

        from torchgeo.datasets import EuroSAT
        class_names = list(EuroSAT.classes)
        acc = run_probe(feats_tr, labels_tr, feats_te, labels_te,
                        class_names, "EuroSAT-MS")
        results["eurosat"] = acc

    # ══ BreizhCrops ══════════════════════════════════════════════════════════
    if args.dataset in ("breizhcrops", "both"):
        print("\n── BreizhCrops (temporal crop classification) ──")
        cache_tr = output_dir / "feat_breizhcrops_train.npz"
        cache_te = output_dir / "feat_breizhcrops_test.npz"

        if cache_tr.exists() and cache_te.exists() and not args.no_cache:
            print("  Loading cached features …")
            tr = np.load(cache_tr, allow_pickle=True)
            te = np.load(cache_te, allow_pickle=True)
            feats_tr, labels_tr = tr["feats"], tr["labels"]
            feats_te, labels_te = te["feats"], te["labels"]
            class_names = list(tr["class_names"])
        else:
            print("  Building BreizhCrops train split (frh01+frh02+frh03) …")
            ds_tr = BreizhCropsProbeDataset(
                regions=["frh01", "frh02", "frh03"],
                root=str(data_dir / "breizhcrops"), download=True)
            ds_te = BreizhCropsProbeDataset(
                regions=["frh04"],
                root=str(data_dir / "breizhcrops"), download=False)
            class_names = [str(c) for c in ds_tr.classes]

            ldr_tr = DataLoader(ds_tr, batch_size=args.batch_size,
                                shuffle=False, num_workers=4, pin_memory=True)
            ldr_te = DataLoader(ds_te, batch_size=args.batch_size,
                                shuffle=False, num_workers=4, pin_memory=True)

            feats_tr, labels_tr = extract_features(encoder, ldr_tr, device, dtype, "train")
            feats_te, labels_te = extract_features(encoder, ldr_te, device, dtype, "test")

            np.savez(cache_tr, feats=feats_tr, labels=labels_tr,
                     class_names=np.array(class_names))
            np.savez(cache_te, feats=feats_te, labels=labels_te,
                     class_names=np.array(class_names))
            print(f"  Features cached → {cache_tr}, {cache_te}")

        acc = run_probe(feats_tr, labels_tr, feats_te, labels_te,
                        class_names, "BreizhCrops")
        results["breizhcrops"] = acc

    # ── summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  Linear Probe Results")
    print("=" * 50)
    for name, acc in results.items():
        print(f"  {name:<20s}  {acc * 100:.2f}%")
    print("=" * 50)

    # Save summary
    summary_path = output_dir / "results.txt"
    with open(summary_path, "w") as f:
        f.write(f"Checkpoint: {args.checkpoint}\n\n")
        for name, acc in results.items():
            f.write(f"{name}: {acc * 100:.2f}%\n")
    print(f"\nSaved summary → {summary_path}")


if __name__ == "__main__":
    main()
