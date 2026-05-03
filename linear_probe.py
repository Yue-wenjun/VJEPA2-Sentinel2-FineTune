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
    Reads EuroSATallBands TIF files directly (no TorchGeo split files needed).

    Scans root recursively for *.tif files, infers class from parent directory
    name, and makes a deterministic 70/15/15 train/val/test split (seed=42).

    Selects B02/B03/B04/B08 (band indices 1,2,3,7 in 13-band EuroSAT TIFs),
    resizes to 256×256, repeats across T=12 monthly time steps.
    Output: (image [4,12,256,256], label int, doys [12])
    """

    # Band order in EuroSATallBands TIF files (0-indexed)
    # B01 B02 B03 B04 B05 B06 B07 B08 B08A B09 B10 B11 B12
    _BAND_IDX = [1, 2, 3, 7]   # B02, B03, B04, B08

    def __init__(self, root: str, split: str, download: bool = False):
        try:
            import rasterio
            self._rasterio_version = rasterio.__version__
        except ImportError:
            raise ImportError("pip install rasterio")

        root = Path(root)
        tif_files = sorted(root.rglob("*.tif"))
        if not tif_files:
            raise FileNotFoundError(
                f"No .tif files found under {root}.\n"
                "Make sure EuroSATallBands is extracted there."
            )

        from collections import defaultdict
        class_files: dict = defaultdict(list)
        for f in tif_files:
            class_files[f.parent.name].append(f)

        classes = sorted(class_files.keys())
        self.classes = classes
        self._label_map = {c: i for i, c in enumerate(classes)}

        rng = np.random.default_rng(42)
        samples: list = []
        for cls, files in sorted(class_files.items()):
            files = sorted(files)
            n = len(files)
            idx = rng.permutation(n)
            n_train = int(0.70 * n)
            n_val   = int(0.15 * n)
            if split == "train":
                chosen = idx[:n_train]
            elif split == "val":
                chosen = idx[n_train : n_train + n_val]
            else:
                chosen = idx[n_train + n_val :]
            for i in chosen:
                samples.append((files[i], self._label_map[cls]))

        self._samples = samples
        self._doys = torch.tensor(_DOYS, dtype=torch.int32)
        print(f"  EuroSAT {split}: {len(samples)} samples, {len(classes)} classes")

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, i):
        import rasterio
        path, label = self._samples[i]
        with rasterio.open(path) as src:
            img = src.read(indexes=[b + 1 for b in self._BAND_IDX]).astype(np.float32)  # [4,64,64]

        img = np.clip(img / 10000.0, 0.0, 1.0)
        img = torch.from_numpy(img)

        img = F.interpolate(img.unsqueeze(0), size=(256, 256),
                            mode="bilinear", align_corners=False)[0]   # [4, 256, 256]

        mean = _MEAN4.view(4, 1, 1)
        std  = _STD4.view(4, 1, 1)
        img  = (img - mean) / std

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
            ds = BC(region=region, root=root, year=2017)
            for idx in range(len(ds)):
                X, y, _ = ds[idx]
                if isinstance(X, torch.Tensor):
                    X = X.numpy()
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
                     pool_mode: str = "global",
                     pca=None,
                     n_spatial: int = 256,
                     n_temporal: int = 6,
                     desc: str = "") -> tuple[np.ndarray, np.ndarray]:
    """
    pool_mode="global"  : mean all 1536 tokens → [N, D]
    pool_mode="temporal": mean 6 temporal → keep 256 spatial → per-token PCA → mean → [N, pca_dim]
                          (requires pca to be pre-fitted; without pca reduces to global)
    """
    embed_dim = encoder.backbone.embed_dim
    all_feats, all_labels = [], []
    total = len(loader)

    for bi, (imgs, labels, doys) in enumerate(loader, 1):
        imgs = imgs.to(device, dtype=dtype)
        doys = doys.to(device)

        with torch.autocast(device_type=device.type, dtype=dtype):
            z = encoder([imgs], doys=doys, training_mode=False)[0]

        z = z[:, :, :embed_dim].float().cpu().numpy()   # [B, 1536, D]
        B = z.shape[0]

        if pool_mode == "temporal" and pca is not None:
            z_t = z.reshape(B, n_temporal, n_spatial, embed_dim).mean(axis=1)  # [B, 256, D]
            z_flat = z_t.reshape(-1, embed_dim)                                 # [B*256, D]
            z_proj = pca.transform(z_flat).reshape(B, n_spatial, -1).mean(axis=1)  # [B, pca_dim]
            feats = z_proj
        else:
            feats = z.mean(axis=1)   # [B, D]  (global, or temporal-no-pca which equals global)
            if pca is not None:
                feats = pca.transform(feats)   # [B, pca_dim]

        all_feats.append(feats)
        all_labels.append(labels.numpy())

        if bi % 50 == 0 or bi == total:
            print(f"  {desc} [{bi}/{total}]  batch done", end="\r")

    print()
    return np.concatenate(all_feats), np.concatenate(all_labels)


@torch.no_grad()
def fit_pca(encoder: MultiSeqWrapper,
            loader: DataLoader,
            device: torch.device,
            dtype: torch.dtype,
            pca_dim: int,
            pool_mode: str,
            n_spatial: int = 256,
            n_temporal: int = 6,
            max_vecs: int = 300_000):
    """Fit sklearn PCA on a subset of encoder tokens from the training loader."""
    from sklearn.decomposition import PCA as SklearnPCA
    embed_dim = encoder.backbone.embed_dim
    vecs = []

    for imgs, _, doys in loader:
        imgs = imgs.to(device, dtype=dtype)
        doys = doys.to(device)

        with torch.autocast(device_type=device.type, dtype=dtype):
            z = encoder([imgs], doys=doys, training_mode=False)[0]

        z = z[:, :, :embed_dim].float().cpu().numpy()
        B = z.shape[0]

        if pool_mode == "temporal":
            z_t = z.reshape(B, n_temporal, n_spatial, embed_dim).mean(axis=1)  # [B, 256, D]
            vecs.append(z_t.reshape(-1, embed_dim))   # [B*256, D]
        else:
            vecs.append(z.mean(axis=1))   # [B, D]

        if sum(v.shape[0] for v in vecs) >= max_vecs:
            break

    data = np.concatenate(vecs, axis=0)[:max_vecs]
    pca = SklearnPCA(n_components=pca_dim)
    pca.fit(data)
    var = pca.explained_variance_ratio_.sum()
    print(f"  PCA fitted on {len(data)} vectors: {embed_dim}→{pca_dim}  "
          f"explained variance={var:.3f}")
    return pca


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
    parser.add_argument("--checkpoint",  default=None,
                        help="Checkpoint path (default: <folder>/<run_tag>/checkpoint_final.pth)")
    parser.add_argument("--run_tag",     default=None,
                        help="Run subfolder name, e.g. run01; auto-fills --checkpoint and --output_dir")
    parser.add_argument("--dataset",     default="both",
                        choices=["eurosat", "breizhcrops", "both"])
    parser.add_argument("--data_dir",    default="data")
    parser.add_argument("--output_dir",  default=None,
                        help="Output dir (default: probe_results or probe_results/<run_tag>)")
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--no_cache",    action="store_true",
                        help="Re-extract features even if cache exists")
    parser.add_argument("--download",    action="store_true",
                        help="Download datasets if not present (requires internet)")
    parser.add_argument("--pool_mode",   default="global",
                        choices=["global", "temporal"],
                        help="global: mean all tokens; temporal: temporal-pool→per-token PCA→mean")
    parser.add_argument("--pca_dim",     type=int, default=None,
                        help="If set, reduce features to this dimension via PCA (e.g. 128)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_tag = args.run_tag or cfg.get("run_tag")
    folder  = Path(cfg["folder"]) / str(run_tag) if run_tag else Path(cfg["folder"])

    checkpoint = args.checkpoint or str(folder / "checkpoint_final.pth")
    output_dir = Path(args.output_dir) if args.output_dir else Path("probe_results") / str(run_tag) if run_tag else Path("probe_results")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if cfg["meta"].get("dtype") == "bfloat16" else torch.float32
    data_dir   = Path(args.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # derive n_spatial / n_temporal from config
    d = cfg["data"]
    n_spatial  = (d["crop_size"] // d["patch_size"]) ** 2          # 256
    n_temporal = d["frames_per_clip"] // d["tubelet_size"]          # 6
    pca_tag    = f"_pca{args.pca_dim}" if args.pca_dim else ""
    feat_tag   = f"_{args.pool_mode}{pca_tag}"                      # e.g. _global or _temporal_pca128

    print(f"Device: {device}  dtype: {dtype}  pool={args.pool_mode}  pca_dim={args.pca_dim}\n")

    # ── build frozen encoder ──────────────────────────────────────────────
    print("Building frozen encoder …")
    encoder = build_frozen_encoder(cfg, checkpoint, device)

    results = {}

    # ══ EuroSAT-MS ══════════════════════════════════════════════════════════
    if args.dataset in ("eurosat", "both"):
        print("\n── EuroSAT-MS ──")
        cache_tr = output_dir / f"feat_eurosat{feat_tag}_train.npz"
        cache_te = output_dir / "feat_eurosat_test.npz"

        if cache_tr.exists() and cache_te.exists() and not args.no_cache:
            print("  Loading cached features …")
            tr = np.load(cache_tr, allow_pickle=True)
            te = np.load(cache_te, allow_pickle=True)
            feats_tr, labels_tr = tr["feats"], tr["labels"]
            feats_te, labels_te = te["feats"], te["labels"]
            class_names = list(tr["class_names"])
        else:
            print("  Building EuroSAT-MS train split …")
            if not args.download and not (data_dir / "eurosat").exists():
                raise FileNotFoundError(
                    f"EuroSAT not found at {data_dir / 'eurosat'}.\n"
                    "Download manually on a machine with internet:\n"
                    "  python linear_probe.py ... --download\n"
                    "or download EuroSATallBands.zip from:\n"
                    "  https://madm.dfki.de/files/sentinel/EuroSATallBands.zip\n"
                    f"and extract to {data_dir / 'eurosat'}"
                )
            ds_tr = EuroSATProbeDataset(str(data_dir / "eurosat"), "train", download=args.download)
            ds_te = EuroSATProbeDataset(str(data_dir / "eurosat"), "test",  download=False)
            class_names = ds_tr.classes
            ldr_tr = DataLoader(ds_tr, batch_size=args.batch_size,
                                shuffle=False, num_workers=4, pin_memory=True)
            ldr_te = DataLoader(ds_te, batch_size=args.batch_size,
                                shuffle=False, num_workers=4, pin_memory=True)

            pca = fit_pca(encoder, ldr_tr, device, dtype, args.pca_dim,
                          args.pool_mode, n_spatial, n_temporal) if args.pca_dim else None
            feats_tr, labels_tr = extract_features(encoder, ldr_tr, device, dtype,
                                                   args.pool_mode, pca, n_spatial, n_temporal, "train")
            feats_te, labels_te = extract_features(encoder, ldr_te, device, dtype,
                                                   args.pool_mode, pca, n_spatial, n_temporal, "test")

            np.savez(cache_tr, feats=feats_tr, labels=labels_tr, class_names=class_names)
            np.savez(cache_te, feats=feats_te, labels=labels_te, class_names=class_names)
            print(f"  Features cached → {cache_tr}, {cache_te}")

        acc = run_probe(feats_tr, labels_tr, feats_te, labels_te,
                        class_names, "EuroSAT-MS")
        results["eurosat"] = acc

    # ══ BreizhCrops ══════════════════════════════════════════════════════════
    if args.dataset in ("breizhcrops", "both"):
        print("\n── BreizhCrops (temporal crop classification) ──")
        cache_tr = output_dir / f"feat_breizhcrops{feat_tag}_train.npz"
        cache_te = output_dir / f"feat_breizhcrops{feat_tag}_test.npz"

        if cache_tr.exists() and cache_te.exists() and not args.no_cache:
            print("  Loading cached features …")
            tr = np.load(cache_tr, allow_pickle=True)
            te = np.load(cache_te, allow_pickle=True)
            feats_tr, labels_tr = tr["feats"], tr["labels"]
            feats_te, labels_te = te["feats"], te["labels"]
            class_names = list(tr["class_names"])
        else:
            print("  Building BreizhCrops train split (frh01+frh02+frh03) …")
            if not args.download and not (data_dir / "breizhcrops").exists():
                raise FileNotFoundError(
                    f"BreizhCrops not found at {data_dir / 'breizhcrops'}.\n"
                    "Download manually on a machine with internet:\n"
                    "  python linear_probe.py ... --download"
                )
            ds_tr = BreizhCropsProbeDataset(
                regions=["frh01", "frh02", "frh03"],
                root=str(data_dir / "breizhcrops"), download=args.download)
            ds_te = BreizhCropsProbeDataset(
                regions=["frh04"],
                root=str(data_dir / "breizhcrops"), download=False)
            class_names = [str(c) for c in ds_tr.classes]

            ldr_tr = DataLoader(ds_tr, batch_size=args.batch_size,
                                shuffle=False, num_workers=4, pin_memory=True)
            ldr_te = DataLoader(ds_te, batch_size=args.batch_size,
                                shuffle=False, num_workers=4, pin_memory=True)

            pca = fit_pca(encoder, ldr_tr, device, dtype, args.pca_dim,
                          args.pool_mode, n_spatial, n_temporal) if args.pca_dim else None
            feats_tr, labels_tr = extract_features(encoder, ldr_tr, device, dtype,
                                                   args.pool_mode, pca, n_spatial, n_temporal, "train")
            feats_te, labels_te = extract_features(encoder, ldr_te, device, dtype,
                                                   args.pool_mode, pca, n_spatial, n_temporal, "test")

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
        f.write(f"Checkpoint: {checkpoint}\n\n")
        for name, acc in results.items():
            f.write(f"{name}: {acc * 100:.2f}%\n")
    print(f"\nSaved summary → {summary_path}")


if __name__ == "__main__":
    main()
