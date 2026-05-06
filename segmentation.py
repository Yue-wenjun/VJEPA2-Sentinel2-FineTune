"""
segmentation.py — End-to-end segmentation fine-tuning: AWF Land Cover / EuroSAT-MS

Fine-tunes V-JEPA encoder + lightweight segmentation decoder.
2-stage training recipe (from OlmoEarth tutorial):
  Stage 1: Freeze encoder, train decoder only   (default 10 epochs)
  Stage 2: Unfreeze encoder, train end-to-end   (default 30 epochs)

Decoder:
  temporal mean-pool → reshape [B, D, 16, 16] → 4× deconv → GELU → 1×1 conv → [B, C, 64, 64]
Loss:
  cross-entropy on spatial-mean logits (patch-level label supervision)

--- AWF data preparation ---
Dataset: allenai/olmoearth_projects_awf on HuggingFace (1.87 GB)

  # Download (on a machine with internet)
  pip install huggingface_hub
  hf download allenai/olmoearth_projects_awf \\
      --repo-type dataset --local-dir /path/to/awf_raw

  # Extract
  tar -xf /path/to/awf_raw/dataset.tar -C /path/to/awf_raw/

  # scp to server
  scp -r /path/to/awf_raw user@server:/home/baai/vjepa2/data/awf

Expected layout after extraction:
  data/awf/
    dataset/windows/spatial_split/task_{uuid}_point_{N}/
      layers/sentinel2.{1-12}/B02_B03_B04_B08/geotiff.tif
    annotation_features.geojson

Usage:
    # AWF — 2-stage fine-tuning (10 frozen + 30 unfrozen epochs)
    python segmentation.py \\
        --config vjepa2/configs/finetune/vitl16/olmoearth-256px-12f.yaml \\
        --checkpoint /home/baai/vjepa2/checkpoints/run03/checkpoint_final.pth \\
        --dataset awf --data_dir /home/baai/vjepa2/data

    # EuroSAT — quick test with existing data
    python segmentation.py \\
        --config vjepa2/configs/finetune/vitl16/olmoearth-256px-12f.yaml \\
        --checkpoint /home/baai/vjepa2/checkpoints/run03/checkpoint_final.pth \\
        --dataset eurosat --data_dir /home/baai/vjepa2/data
"""

import argparse
import importlib.util
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, classification_report
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent / "vjepa2"))
import app.vjepa_2_1.models.vision_transformer as video_vit
from app.vjepa_2_1.wrappers import MultiSeqWrapper
from data_pipeline.patch_embed_6ch import build_nch_patch_embed_from_pretrained  # noqa: F401

_MEAN4 = torch.tensor([0.0850, 0.0950, 0.1001, 0.2841])
_STD4  = torch.tensor([0.0574, 0.0521, 0.0660, 0.1076])
_DOYS  = [15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]


# ── encoder utilities ─────────────────────────────────────────────────────────

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


def build_encoder(cfg: dict, ckpt_path: str, device: torch.device) -> MultiSeqWrapper:
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
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    state = _strip_prefix(ckpt.get("encoder", ckpt))
    _safe_load(encoder.backbone, state)
    print(f"  Loaded checkpoint (epoch={ckpt.get('epoch', '?')}): {ckpt_path}")
    return encoder


# ── segmentation head ─────────────────────────────────────────────────────────

class SegHead(nn.Module):
    """
    Lightweight decoder on top of frozen/unfrozen V-JEPA encoder.

    Input:  tokens [B, n_temporal * n_spatial, D]
    Steps:  temporal mean-pool → reshape [B, D, H, W] → 4× deconv → GELU → 1×1 conv
    Output: logits [B, num_classes, H*4, W*4]  (e.g. [B, C, 64, 64] for 16×16 tokens)
    """

    def __init__(self, embed_dim: int, num_classes: int,
                 n_spatial: int = 256, n_temporal: int = 6):
        super().__init__()
        self.n_temporal = n_temporal
        self.n_spatial  = n_spatial
        self.h = self.w = int(n_spatial ** 0.5)   # 16
        hidden = 256
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, hidden, kernel_size=4, stride=4),
            nn.GELU(),
            nn.Conv2d(hidden, num_classes, kernel_size=1),
        )
        self._embed_dim = embed_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        D = self._embed_dim
        z = tokens[:, :, :D]                                              # [B, T*S, D]
        z = z.reshape(B, self.n_temporal, self.n_spatial, D).mean(dim=1) # [B, S, D]
        z = z.permute(0, 2, 1).reshape(B, D, self.h, self.w)             # [B, D, 16, 16]
        return self.decoder(z)                                             # [B, C, 64, 64]


# ── datasets ──────────────────────────────────────────────────────────────────

def _augment(img: torch.Tensor) -> torch.Tensor:
    """Random horizontal/vertical flip applied consistently across all time steps."""
    if random.random() > 0.5:
        img = torch.flip(img, dims=[-1])
    if random.random() > 0.5:
        img = torch.flip(img, dims=[-2])
    return img.contiguous()


class AWFDataset(Dataset):
    """
    rslearn-format AWF Kenya land cover (allenai/olmoearth_projects_awf).

    Layout (after extracting dataset.tar):
      root/
        dataset/windows/spatial_split/task_{uuid}_point_{N}/
          layers/sentinel2.{1-12}/B02_B03_B04_B08/geotiff.tif  (4-band, one month)
        annotation_features.geojson

    Labels: properties.oe_labels.category (int)
    Folder→label: task_{uuid}_point_{N} maps to the N-th point (0-indexed) in
                  annotation_features.geojson entries with oe_annotations_task_id=uuid.
    Split: west longitude (train) / east longitude (val), boundary = median longitude.
    """

    def __init__(self, root: str, split: str, augment: bool = False):
        import importlib
        if importlib.util.find_spec("rasterio") is None:
            raise ImportError("pip install rasterio")

        root        = Path(root)
        geojson     = root / "annotation_features.geojson"
        windows_dir = root / "dataset" / "windows" / "spatial_split"

        if not geojson.exists():
            raise FileNotFoundError(f"annotation_features.geojson not found at {root}")
        if not windows_dir.exists():
            raise FileNotFoundError(f"dataset/windows/spatial_split not found at {root}")

        # Parse geojson: task_uuid → [(category, longitude), ...]  (preserving order)
        with open(geojson) as f:
            gj = json.load(f)

        task_pts: dict = defaultdict(list)
        for feat in gj["features"]:
            tid = feat["properties"]["oe_annotations_task_id"]
            cat = feat["properties"]["oe_labels"]["category"]
            lon = feat["geometry"]["coordinates"][0]
            task_pts[tid].append((cat, lon))

        # Build class list from all unique categories (sorted)
        all_cats  = sorted({c for pts in task_pts.values() for c, _ in pts})
        cat_to_idx = {c: i for i, c in enumerate(all_cats)}
        self.classes = [f"class_{c}" for c in all_cats]

        # Scan folders and join with geojson labels
        all_samples = []
        for folder in sorted(windows_dir.iterdir()):
            name = folder.name  # "task_{uuid}_point_{N}"
            sep  = name.rfind("_point_")
            if sep == -1 or not folder.is_dir():
                continue
            task_uuid = name[5:sep]          # strip leading "task_"
            point_idx = int(name[sep + 7:])  # index after "_point_"

            pts = task_pts.get(task_uuid)
            if pts is None or point_idx >= len(pts):
                continue

            cat, lon = pts[point_idx]
            all_samples.append((folder, cat_to_idx[cat], lon))

        # Spatial split by median longitude
        lons    = [lon for _, _, lon in all_samples]
        med_lon = float(np.median(lons)) if lons else 0.0
        if split == "train":
            chosen = [(f, l) for f, l, lon in all_samples if lon <= med_lon]
        else:
            chosen = [(f, l) for f, l, lon in all_samples if lon > med_lon]

        self._samples = chosen
        self._augment = augment
        self._doys    = torch.tensor(_DOYS, dtype=torch.int32)
        print(f"  AWF {split}: {len(chosen)} samples, {len(self.classes)} classes "
              f"(lon_split={med_lon:.3f}°)")

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, i):
        import rasterio
        folder, label = self._samples[i]

        # Load 12 monthly TIFs: sentinel2.{1-12}/B02_B03_B04_B08/geotiff.tif
        monthly: list = []
        ref_shape = None
        for month in range(1, 13):
            tif = folder / "layers" / f"sentinel2.{month}" / "B02_B03_B04_B08" / "geotiff.tif"
            if tif.exists():
                with rasterio.open(tif) as src:
                    data = src.read().astype(np.float32)   # [4, H, W]
                ref_shape = data.shape
                monthly.append(data)
            else:
                monthly.append(None)

        # Fill missing months (forward then backward)
        for m in range(1, 12):
            if monthly[m] is None and monthly[m - 1] is not None:
                monthly[m] = monthly[m - 1].copy()
        for m in range(10, -1, -1):
            if monthly[m] is None and monthly[m + 1] is not None:
                monthly[m] = monthly[m + 1].copy()
        h, w = ref_shape[1:] if ref_shape else (64, 64)
        for m in range(12):
            if monthly[m] is None:
                monthly[m] = np.zeros((4, h, w), dtype=np.float32)

        imgs = np.stack(monthly, axis=0)            # [12, 4, H, W]
        imgs = np.clip(imgs / 10000.0, 0.0, 1.0)
        img  = torch.from_numpy(imgs).permute(1, 0, 2, 3).contiguous()  # [4, 12, H, W]

        if img.shape[-1] != 256 or img.shape[-2] != 256:
            flat = img.view(48, *img.shape[2:])
            flat = F.interpolate(flat.unsqueeze(0), (256, 256),
                                 mode="bilinear", align_corners=False)[0]
            img  = flat.view(4, 12, 256, 256)

        img = (img - _MEAN4.view(4, 1, 1, 1)) / _STD4.view(4, 1, 1, 1)
        if self._augment:
            img = _augment(img)
        return img, label, self._doys.clone()


class EuroSATSegDataset(Dataset):
    """EuroSAT-MS for segmentation training (same band selection as linear_probe.py)."""

    _BAND_IDX = [1, 2, 3, 7]   # B02 B03 B04 B08

    def __init__(self, root: str, split: str, augment: bool = False):
        if importlib.util.find_spec("rasterio") is None:
            raise ImportError("pip install rasterio")

        root = Path(root)
        tif_files = sorted(root.rglob("*.tif"))
        if not tif_files:
            raise FileNotFoundError(f"No .tif files found under {root}.")

        class_files: dict = defaultdict(list)
        for f in tif_files:
            class_files[f.parent.name].append(f)

        classes   = sorted(class_files.keys())
        self.classes  = classes
        label_map = {c: i for i, c in enumerate(classes)}

        rng = np.random.default_rng(42)
        samples: list = []
        for cls, files in sorted(class_files.items()):
            files = sorted(files)
            n     = len(files)
            idx   = rng.permutation(n)
            n_tr  = int(0.70 * n)
            n_va  = int(0.15 * n)
            if split == "train":
                sel = idx[:n_tr]
            elif split == "val":
                sel = idx[n_tr:n_tr + n_va]
            else:
                sel = idx[n_tr + n_va:]
            for j in sel:
                samples.append((files[j], label_map[cls]))

        self._samples = samples
        self._augment = augment
        self._doys    = torch.tensor(_DOYS, dtype=torch.int32)
        print(f"  EuroSAT {split}: {len(samples)} samples, {len(classes)} classes")

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, i):
        import rasterio
        path, label = self._samples[i]
        with rasterio.open(path) as src:
            img = src.read(indexes=[b + 1 for b in self._BAND_IDX]).astype(np.float32)

        img = np.clip(img / 10000.0, 0.0, 1.0)
        img = torch.from_numpy(img)
        img = F.interpolate(img.unsqueeze(0), (256, 256),
                            mode="bilinear", align_corners=False)[0]
        img = (img - _MEAN4.view(4, 1, 1)) / _STD4.view(4, 1, 1)
        img = img.unsqueeze(1).expand(-1, 12, -1, -1).contiguous()
        if self._augment:
            img = _augment(img)
        return img, label, self._doys.clone()


# ── training / eval ───────────────────────────────────────────────────────────

def train_epoch(encoder, head, loader, optimizer, device, dtype,
                encoder_frozen: bool = False) -> float:
    encoder.eval() if encoder_frozen else encoder.train()
    head.train()
    total_loss, n = 0.0, 0

    params_for_clip = list(head.parameters())
    if not encoder_frozen:
        params_for_clip += list(encoder.parameters())

    for imgs, labels, doys in loader:
        imgs   = imgs.to(device, dtype=dtype)
        doys   = doys.to(device)
        labels = labels.to(device)

        with torch.autocast(device_type=device.type, dtype=dtype):
            tokens = encoder([imgs], doys=doys, training_mode=False)[0]
            logits = head(tokens)                                   # [B, C, 64, 64]
            loss   = F.cross_entropy(logits.mean(dim=(-2, -1)), labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params_for_clip, max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)

    return total_loss / n


@torch.no_grad()
def run_eval(encoder, head, loader, device, dtype):
    encoder.eval()
    head.eval()
    all_preds, all_labels = [], []

    for imgs, labels, doys in loader:
        imgs = imgs.to(device, dtype=dtype)
        doys = doys.to(device)
        with torch.autocast(device_type=device.type, dtype=dtype):
            tokens = encoder([imgs], doys=doys, training_mode=False)[0]
            logits = head(tokens)
            preds  = logits.mean(dim=(-2, -1)).argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())

    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return accuracy_score(labels, preds), preds, labels


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",           required=True)
    parser.add_argument("--checkpoint",       default=None,
                        help="Checkpoint path (default: <folder>/<run_tag>/checkpoint_final.pth)")
    parser.add_argument("--run_tag",          default=None)
    parser.add_argument("--dataset",          default="awf", choices=["awf", "eurosat"])
    parser.add_argument("--data_dir",         default="data")
    parser.add_argument("--output_dir",       default=None,
                        help="Output dir (default: <folder>/<run_tag>/seg_results)")
    parser.add_argument("--batch_size",       type=int, default=4,
                        help="Reduce to 2 if OOM during stage 2 (full encoder)")
    parser.add_argument("--freeze_epochs",    type=int, default=10,
                        help="Stage 1: decoder-only training epochs")
    parser.add_argument("--unfreeze_epochs",  type=int, default=30,
                        help="Stage 2: end-to-end fine-tuning epochs")
    parser.add_argument("--lr",               type=float, default=1e-4)
    parser.add_argument("--encoder_lr_scale", type=float, default=0.1,
                        help="Encoder LR = lr * encoder_lr_scale in stage 2 (default 0.1)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_tag   = args.run_tag or cfg.get("run_tag")
    folder    = Path(cfg["folder"]) / str(run_tag) if run_tag else Path(cfg["folder"])
    ckpt_path = args.checkpoint or str(folder / "checkpoint_final.pth")
    out_dir   = (Path(args.output_dir) if args.output_dir else folder / "seg_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype    = torch.bfloat16 if cfg["meta"].get("dtype") == "bfloat16" else torch.float32
    data_dir = Path(args.data_dir)

    d          = cfg["data"]
    n_spatial  = (d["crop_size"] // d["patch_size"]) ** 2    # 256 = 16×16
    n_temporal = d["frames_per_clip"] // d["tubelet_size"]    # 6

    print(f"Device: {device}  dtype: {dtype}  dataset: {args.dataset}\n")

    # ── datasets ──────────────────────────────────────────────────────────
    if args.dataset == "awf":
        ds_tr = AWFDataset(str(data_dir / "awf"), "train", augment=True)
        ds_va = AWFDataset(str(data_dir / "awf"), "val")
    else:
        ds_tr = EuroSATSegDataset(str(data_dir / "eurosat"), "train", augment=True)
        ds_va = EuroSATSegDataset(str(data_dir / "eurosat"), "val")

    ldr_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)
    ldr_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    class_names = ds_tr.classes
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}\n")

    # ── build encoder + head ───────────────────────────────────────────────
    print("Building encoder …")
    encoder   = build_encoder(cfg, ckpt_path, device)
    embed_dim = encoder.backbone.embed_dim

    head = SegHead(embed_dim, num_classes, n_spatial, n_temporal).to(device)
    print(f"SegHead params: {sum(p.numel() for p in head.parameters()):,}\n")

    # ── stage 1: freeze encoder, train decoder only ────────────────────────
    if args.freeze_epochs > 0:
        print(f"── Stage 1: frozen encoder, {args.freeze_epochs} epochs ──")
        for p in encoder.parameters():
            p.requires_grad = False

        opt1 = AdamW(head.parameters(), lr=args.lr)
        sch1 = ReduceLROnPlateau(opt1, patience=3, factor=0.2)

        for ep in range(1, args.freeze_epochs + 1):
            loss = train_epoch(encoder, head, ldr_tr, opt1, device, dtype,
                               encoder_frozen=True)
            acc, _, _ = run_eval(encoder, head, ldr_va, device, dtype)
            sch1.step(loss)
            print(f"  ep {ep:02d}/{args.freeze_epochs}  "
                  f"loss={loss:.4f}  val_acc={acc * 100:.2f}%")

        torch.save(head.state_dict(), out_dir / "head_stage1.pth")
        print(f"  Saved → {out_dir}/head_stage1.pth\n")

    # ── stage 2: unfreeze encoder, end-to-end ─────────────────────────────
    if args.unfreeze_epochs > 0:
        print(f"── Stage 2: end-to-end fine-tune, {args.unfreeze_epochs} epochs ──")
        for p in encoder.parameters():
            p.requires_grad = True

        opt2 = AdamW([
            {"params": encoder.parameters(), "lr": args.lr * args.encoder_lr_scale},
            {"params": head.parameters(),    "lr": args.lr},
        ])
        sch2 = ReduceLROnPlateau(opt2, patience=5, factor=0.2)

        best_acc = 0.0
        for ep in range(1, args.unfreeze_epochs + 1):
            loss = train_epoch(encoder, head, ldr_tr, opt2, device, dtype,
                               encoder_frozen=False)
            acc, _, _ = run_eval(encoder, head, ldr_va, device, dtype)
            sch2.step(loss)
            marker = " ←best" if acc > best_acc else ""
            print(f"  ep {ep:02d}/{args.unfreeze_epochs}  "
                  f"loss={loss:.4f}  val_acc={acc * 100:.2f}%{marker}")
            if acc > best_acc:
                best_acc = acc
                torch.save(head.state_dict(),    out_dir / "head_best.pth")
                torch.save(encoder.state_dict(), out_dir / "encoder_best.pth")

        # restore best-of-stage
        head.load_state_dict(
            torch.load(out_dir / "head_best.pth",    map_location=device))
        encoder.load_state_dict(
            torch.load(out_dir / "encoder_best.pth", map_location=device))
        print(f"\n  Best val acc: {best_acc * 100:.2f}%")
        print(f"  Saved → {out_dir}/head_best.pth  encoder_best.pth\n")

    # ── final evaluation ───────────────────────────────────────────────────
    print("── Final Evaluation ──")
    acc, preds, labels_np = run_eval(encoder, head, ldr_va, device, dtype)
    print(f"  Val accuracy: {acc * 100:.2f}%")
    report = classification_report(labels_np, preds,
                                   target_names=class_names, digits=3)
    print(report)

    with open(out_dir / "results.txt", "w") as f:
        f.write(f"Checkpoint: {ckpt_path}\nDataset: {args.dataset}\n"
                f"Val accuracy: {acc * 100:.2f}%\n\n{report}")
    print(f"Saved summary → {out_dir}/results.txt")


if __name__ == "__main__":
    main()
