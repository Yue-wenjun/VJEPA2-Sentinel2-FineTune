"""
visualize.py — V-JEPA 2.1 × OLMo-Earth Embedding Visualizer

Paper-style patch embedding PCA figures from a fine-tuned checkpoint.
Runs on server (no display — saves PNG files).

Outputs (in --output_dir):
    embeddings_finetuned.png          always
    embeddings_pretrained.png         if --pretrained is given
    comparison.png                    if --pretrained is given

Usage:
    python visualize.py \\
        --config  vjepa2/configs/finetune/vitl16/olmoearth-256px-12f.yaml \\
        --checkpoint /home/baai/vjepa2/checkpoints/checkpoint_final.pth \\
        [--pretrained  /home/baai/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt] \\
        [--n_samples   6] \\
        [--output_dir  /home/baai/vjepa2/vis]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # no display needed on server
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent / "vjepa2"))

import app.vjepa_2_1.models.vision_transformer as video_vit
from app.vjepa_2_1.wrappers import MultiSeqWrapper
from data_pipeline.patch_embed_6ch import build_nch_patch_embed_from_pretrained
from data_pipeline.olmoearth_dataset import OLMoEarthDataset

# ── Sentinel-2 normalization stats (for un-z-scoring RGB display) ─────────────
_MEAN4 = np.array([0.0850, 0.0950, 0.1001, 0.2841])
_STD4  = np.array([0.0574, 0.0521, 0.0660, 0.1076])

TUBELET_LABELS = ["Jan–Feb", "Mar–Apr", "May–Jun",
                  "Jul–Aug", "Sep–Oct", "Nov–Dec"]


# ── model helpers ─────────────────────────────────────────────────────────────

def build_encoder(cfg: dict, device: torch.device) -> MultiSeqWrapper:
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
    return MultiSeqWrapper(backbone).to(device)


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


def load_finetuned(encoder: MultiSeqWrapper, ckpt_path: str, device: torch.device):
    """Load from finetune_main.py checkpoint — key 'encoder' holds backbone state_dict."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = _strip_prefix(ckpt.get("encoder", ckpt))
    _safe_load(encoder.backbone, state)
    epoch = ckpt.get("epoch", "?")
    print(f"  fine-tuned checkpoint loaded  (epoch={epoch}): {ckpt_path}")


def load_pretrained(encoder: MultiSeqWrapper, ckpt_path: str,
                    device: torch.device, in_chans: int):
    """Load original V-JEPA 2.1 pretrained checkpoint with N-channel patch_embed init."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    enc_state = _strip_prefix(ckpt.get("encoder", ckpt))

    new_pe = build_nch_patch_embed_from_pretrained(
        pretrained_state_dict=enc_state,
        in_chans=in_chans,
        patch_size=encoder.backbone.patch_size,
        tubelet_size=encoder.backbone.tubelet_size,
        embed_dim=encoder.backbone.embed_dim,
    ).to(device)
    encoder.backbone.patch_embed = new_pe

    filtered = {k: v for k, v in enc_state.items()
                if not k.startswith("patch_embed")}
    _safe_load(encoder.backbone, filtered)
    print(f"  pretrained checkpoint loaded: {ckpt_path}")


# ── data ──────────────────────────────────────────────────────────────────────

def collect_samples(cfg: dict, n: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Stream the first n valid (buffer, doys) pairs from the OLMo-Earth TAR files."""
    oe, d = cfg["olmoearth"], cfg["data"]
    ds = OLMoEarthDataset(
        tar_path=oe["tar_path"],
        n_bands_per_timestep=oe.get("n_bands_per_timestep", 4),
        crop_size=d["crop_size"],
        dn_scale=oe.get("dn_scale", 10000.0),
        max_missing_frac=oe.get("max_missing_frac", 0.10),
        shuffle_buffer=1,       # deterministic; no shuffle for visualization
        random_flip=False,
        seed=0,
        node_split=False,       # single-process: read all shards without distribution
    )
    samples = []
    for buffers, _, doys, _ in ds:
        samples.append((buffers[0], doys))   # buf: [C, T, H, W], doys: [T]
        if len(samples) >= n:
            break
    print(f"Collected {len(samples)} samples")
    return samples


# ── embedding extraction ──────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(encoder: MultiSeqWrapper, samples: list,
                       device: torch.device, dtype: torch.dtype) -> np.ndarray:
    """
    Returns float32 numpy array of shape [N, N_TOK, embed_dim].
    Uses training_mode=False → standard last-layer features [B, N_TOK, embed_dim].
    """
    encoder.eval()
    embed_dim = encoder.backbone.embed_dim
    all_embs = []
    for buf, doys in samples:
        x    = buf.unsqueeze(0).to(device, dtype=dtype)   # [1, C, T, H, W]
        doys = doys.unsqueeze(0).to(device)               # [1, T]
        with torch.autocast(device_type=device.type, dtype=dtype):
            # wrapper returns list[Tensor[B, N_TOK, D]], take clip 0
            z = encoder([x], doys=doys, training_mode=False)[0]
        # guard: take only first embed_dim dims (safe whether D or 4D is returned)
        all_embs.append(z[0, :, :embed_dim].float().cpu().numpy())
    return np.stack(all_embs, axis=0)   # [N, N_TOK, embed_dim]


# ── PCA ───────────────────────────────────────────────────────────────────────

def compute_pca_rgb(embs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Joint PCA fit over all N × N_TOK tokens; returns per-token RGB in [0,1].

    Args:
        embs:  [N, N_TOK, D]
    Returns:
        pca_rgb: [N, N_TOK, 3]  float32
        var_exp: [3]             variance explained ratio for PC1/PC2/PC3
    """
    N, T, D = embs.shape
    flat   = embs.reshape(N * T, D)
    flat_c = flat - flat.mean(axis=0)

    cov = flat_c.T @ flat_c / (flat_c.shape[0] - 1)   # [D, D]
    eigvals, eigvecs = np.linalg.eigh(cov)              # ascending eigenvalues
    top3    = eigvecs[:, -3:][:, ::-1]                  # [D, 3] descending
    var_exp = eigvals[-3:][::-1] / eigvals.sum()        # [3]

    proj = flat_c @ top3                                # [N*T, 3]
    pca_rgb = np.zeros_like(proj, dtype=np.float32)
    for i in range(3):
        lo = np.percentile(proj[:, i], 1)
        hi = np.percentile(proj[:, i], 99)
        pca_rgb[:, i] = np.clip((proj[:, i] - lo) / (hi - lo + 1e-8), 0.0, 1.0)

    return pca_rgb.reshape(N, T, 3), var_exp


# ── rendering helpers ─────────────────────────────────────────────────────────

def _buf_to_rgb(buf_np: np.ndarray, t: int) -> np.ndarray:
    """Un-z-score and convert [C, T, H, W] float32 → [H, W, 3] RGB at month t."""
    rgb = np.zeros((*buf_np.shape[2:], 3), dtype=np.float32)
    for ci, bi in enumerate([2, 1, 0]):   # B04→R  B03→G  B02→B
        ch = buf_np[bi, t] * _STD4[bi] + _MEAN4[bi]
        lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)
        rgb[:, :, ci] = np.clip((ch - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return rgb


def _pca_tile(pca_rgb_sample: np.ndarray, ti: int, H_PAT: int = 16, W_PAT: int = 16,
              patch_px: int = 16) -> np.ndarray:
    """Extract PCA colors for tubelet ti and upsample to pixel space."""
    tok_start = ti * H_PAT * W_PAT
    tok_end   = tok_start + H_PAT * W_PAT
    patch_colors = pca_rgb_sample[tok_start:tok_end].reshape(H_PAT, W_PAT, 3)
    return np.repeat(np.repeat(patch_colors, patch_px, axis=0), patch_px, axis=1)


# ── figure builders ───────────────────────────────────────────────────────────

def save_pca_figure(samples: list, pca_rgb: np.ndarray, var_exp: np.ndarray,
                    output_path: Path, title: str):
    """
    Grid: rows = samples (2 rows each: RGB input + PCA embedding),
          cols = 6 tubelet timepoints.
    """
    N = len(samples)
    T_PAT = 6

    fig, axes = plt.subplots(N * 2, T_PAT, figsize=(T_PAT * 2.5, N * 5.2),
                             squeeze=False)

    for si, (buf, _) in enumerate(samples):
        buf_np = buf.float().numpy()
        for ti in range(T_PAT):
            m1, m2 = ti * 2, ti * 2 + 1
            rgb = (_buf_to_rgb(buf_np, m1) + _buf_to_rgb(buf_np, m2)) / 2.0

            ax_rgb = axes[si * 2, ti]
            ax_rgb.imshow(rgb)
            ax_rgb.axis("off")
            if si == 0:
                ax_rgb.set_title(TUBELET_LABELS[ti], fontsize=9)

            ax_pca = axes[si * 2 + 1, ti]
            ax_pca.imshow(_pca_tile(pca_rgb[si], ti))
            ax_pca.axis("off")

        axes[si * 2,     0].set_ylabel(f"Sample {si+1}\nRGB",     fontsize=8,
                                        rotation=0, labelpad=55, va="center")
        axes[si * 2 + 1, 0].set_ylabel("PCA emb.", fontsize=8,
                                        rotation=0, labelpad=55, va="center")

    var_str = "PC1={:.1f}%  PC2={:.1f}%  PC3={:.1f}%".format(
        var_exp[0] * 100, var_exp[1] * 100, var_exp[2] * 100)
    fig.suptitle(f"{title}\n{var_str}", fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output_path}")


def save_comparison_figure(samples: list,
                           pca_pre: np.ndarray, var_pre: np.ndarray,
                           pca_ft:  np.ndarray, var_ft:  np.ndarray,
                           output_path: Path):
    """
    3-row-per-sample grid: RGB input | pretrained PCA | fine-tuned PCA.
    Shows how fine-tuning changes learnt feature structure.
    """
    N = len(samples)
    T_PAT = 6
    row_labels = ["RGB input", "Pretrained", "Fine-tuned"]

    fig, axes = plt.subplots(N * 3, T_PAT, figsize=(T_PAT * 2.5, N * 7.5),
                             squeeze=False)

    for si, (buf, _) in enumerate(samples):
        buf_np = buf.float().numpy()
        for ti in range(T_PAT):
            m1, m2 = ti * 2, ti * 2 + 1
            rgb = (_buf_to_rgb(buf_np, m1) + _buf_to_rgb(buf_np, m2)) / 2.0

            ax_rgb = axes[si * 3, ti]
            ax_rgb.imshow(rgb)
            ax_rgb.axis("off")
            if si == 0:
                ax_rgb.set_title(TUBELET_LABELS[ti], fontsize=9)

            axes[si * 3 + 1, ti].imshow(_pca_tile(pca_pre[si], ti))
            axes[si * 3 + 1, ti].axis("off")

            axes[si * 3 + 2, ti].imshow(_pca_tile(pca_ft[si], ti))
            axes[si * 3 + 2, ti].axis("off")

        for row_off, lbl in enumerate(row_labels):
            axes[si * 3 + row_off, 0].set_ylabel(
                f"S{si + 1}  {lbl}", fontsize=7.5,
                rotation=0, labelpad=65, va="center")

    pre_str = "Pretrained  PC1={:.1f}%  PC2={:.1f}%  PC3={:.1f}%".format(
        *[v * 100 for v in var_pre])
    ft_str = "Fine-tuned   PC1={:.1f}%  PC2={:.1f}%  PC3={:.1f}%".format(
        *[v * 100 for v in var_ft])
    fig.suptitle(f"Embedding Comparison\n{pre_str}\n{ft_str}", fontsize=10, y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate paper-style embedding PCA figures from a V-JEPA 2.1 checkpoint.")
    parser.add_argument("--config",      required=True,
                        help="YAML config (same file used for fine-tuning)")
    parser.add_argument("--checkpoint",  required=True,
                        help="Fine-tuned .pth checkpoint path")
    parser.add_argument("--pretrained",  default=None,
                        help="Original pretrained checkpoint for before/after comparison")
    parser.add_argument("--n_samples",   type=int, default=6,
                        help="Number of samples to visualize (default: 6)")
    parser.add_argument("--output_dir",  default="vis",
                        help="Directory to write PNG files (default: ./vis)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype   = torch.bfloat16 if cfg["meta"].get("dtype") == "bfloat16" else torch.float32
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}  dtype: {dtype}  output: {out_dir}\n")

    # ── samples ───────────────────────────────────────────────────────────
    print(f"Collecting {args.n_samples} samples …")
    samples = collect_samples(cfg, args.n_samples)

    # ── fine-tuned encoder ────────────────────────────────────────────────
    print("\nLoading fine-tuned encoder …")
    enc_ft = build_encoder(cfg, device)
    load_finetuned(enc_ft, args.checkpoint, device)

    print("Extracting fine-tuned embeddings …")
    embs_ft = extract_embeddings(enc_ft, samples, device, dtype)
    pca_ft, var_ft = compute_pca_rgb(embs_ft)

    save_pca_figure(
        samples, pca_ft, var_ft,
        out_dir / "embeddings_finetuned.png",
        title="Fine-tuned Encoder — Patch Embeddings PCA(3) as RGB",
    )

    # ── optional comparison with pretrained ───────────────────────────────
    if args.pretrained:
        print("\nLoading pretrained encoder for comparison …")
        enc_pre = build_encoder(cfg, device)
        load_pretrained(enc_pre, args.pretrained, device, cfg["model"]["in_chans"])

        print("Extracting pretrained embeddings …")
        embs_pre = extract_embeddings(enc_pre, samples, device, dtype)
        pca_pre, var_pre = compute_pca_rgb(embs_pre)

        save_pca_figure(
            samples, pca_pre, var_pre,
            out_dir / "embeddings_pretrained.png",
            title="Pretrained Encoder — Patch Embeddings PCA(3) as RGB",
        )
        save_comparison_figure(
            samples, pca_pre, var_pre, pca_ft, var_ft,
            out_dir / "comparison.png",
        )
        del enc_pre

    del enc_ft
    print(f"\nDone. Figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
