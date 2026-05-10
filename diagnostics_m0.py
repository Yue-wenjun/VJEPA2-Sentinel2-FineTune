"""
diagnostics_m0.py — M0 light diagnostics (Stage 0 of PFU plan).

Maps to checklist.md M0:
  0.1 effective rank / covariance spectrum of frozen V-JEPA features
  0.2 token covariance comparison: hand-crafted RGB vs Prithvi vs random adapter
  0.3 EuroSAT linear probe vs spectral-distance ε preliminary correlation

For each of three frozen encoder configurations on EuroSAT-MS:
  • extract per-sample mean-pooled features (probe-style) and per-token features
  • compute effective rank, top-k covariance spectrum
  • run a logistic-regression linear probe → top-1 accuracy
  • compute pairwise covariance distance across configurations (ε proxy)

No new training. Reuses the encoder builder + EuroSAT loader from linear_probe.py.

Usage:
  python diagnostics_m0.py \\
    --config vjepa2/configs/finetune/vitl16/olmoearth-256px-12f.yaml \\
    --checkpoint /home/baai/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt \\
    --data_dir /home/baai/data \\
    --output_dir m0_results \\
    [--max_samples 2000]   # subsample EuroSAT for fast estimates
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent / "vjepa2"))

from data_pipeline.patch_embed_6ch import build_nch_patch_embed_from_pretrained
from linear_probe import (
    EuroSATProbeDataset,
    _MEAN4,
    _STD4,
    _strip_prefix,
    build_frozen_encoder,
)


# ── Hand-crafted RGB EuroSAT (true colour: B04, B03, B02) ────────────────────

class EuroSATRGB3ChDataset(EuroSATProbeDataset):
    """
    Same EuroSAT split as EuroSATProbeDataset but returns 3-channel true-colour
    RGB tiles in the order V-JEPA's pretrained RGB patch_embed expects.

    Channel order: B04 (red), B03 (green), B02 (blue) — derived from the same
    [B02, B03, B04, B08] subset used elsewhere, just permuted to RGB and
    dropping the NIR band.
    """

    # _BAND_IDX in parent: [1, 2, 3, 7] = [B02, B03, B04, B08].
    # We re-permute to [B04, B03, B02] = parent indices [2, 1, 0].
    _RGB_FROM_PARENT = [2, 1, 0]
    _MEAN_RGB = _MEAN4[[2, 1, 0]].clone()
    _STD_RGB = _STD4[[2, 1, 0]].clone()

    def __getitem__(self, i):
        import rasterio
        path, label = self._samples[i]
        with rasterio.open(path) as src:
            img = src.read(indexes=[b + 1 for b in self._BAND_IDX]).astype(np.float32)

        img = np.clip(img / 10000.0, 0.0, 1.0)
        img = torch.from_numpy(img)                               # [4, 64, 64]
        img = img[self._RGB_FROM_PARENT]                          # [3, 64, 64]

        img = F.interpolate(img.unsqueeze(0), size=(256, 256),
                            mode="bilinear", align_corners=False)[0]
        mean = self._MEAN_RGB.view(3, 1, 1)
        std = self._STD_RGB.view(3, 1, 1)
        img = (img - mean) / std

        img = img.unsqueeze(1).expand(-1, 12, -1, -1).contiguous()  # [3, 12, 256, 256]
        return img, label, self._doys.clone()


# ── encoder builders for the three M0 configurations ─────────────────────────

def build_encoder_random(cfg, ckpt_path, device):
    """Force a random N-channel patch_embed after loading the rest of the encoder."""
    cfg_local = copy.deepcopy(cfg)
    cfg_local["model"]["patch_embed_init"] = "random"
    encoder = build_frozen_encoder(cfg_local, ckpt_path, device)
    encoder.backbone.patch_embed = build_nch_patch_embed_from_pretrained(
        pretrained_state_dict={},
        in_chans=cfg_local["model"]["in_chans"],
        patch_size=cfg_local["data"]["patch_size"],
        tubelet_size=cfg_local["data"]["tubelet_size"],
        embed_dim=encoder.backbone.embed_dim,
        init_mode="random",
    ).to(device)
    for p in encoder.backbone.patch_embed.parameters():
        p.requires_grad = False
    encoder.eval()
    return encoder


def build_encoder_prithvi(cfg, ckpt_path, device):
    """patch_embed initialised by Prithvi-style RGB-mean copy."""
    cfg_local = copy.deepcopy(cfg)
    cfg_local["model"]["patch_embed_init"] = "prithvi"
    encoder = build_frozen_encoder(cfg_local, ckpt_path, device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = _strip_prefix(ckpt.get("encoder", ckpt))
    new_pe = build_nch_patch_embed_from_pretrained(
        pretrained_state_dict=state,
        in_chans=cfg_local["model"]["in_chans"],
        patch_size=cfg_local["data"]["patch_size"],
        tubelet_size=cfg_local["data"]["tubelet_size"],
        embed_dim=encoder.backbone.embed_dim,
        init_mode="prithvi",
    ).to(device)
    encoder.backbone.patch_embed = new_pe
    for p in encoder.backbone.patch_embed.parameters():
        p.requires_grad = False
    encoder.eval()
    return encoder


def build_encoder_hand_rgb(cfg, ckpt_path, device):
    """3-channel encoder using V-JEPA's original pretrained RGB patch_embed."""
    cfg_local = copy.deepcopy(cfg)
    cfg_local["model"]["in_chans"] = 3
    return build_frozen_encoder(cfg_local, ckpt_path, device)


CONFIGS = {
    "random":    (build_encoder_random,    EuroSATProbeDataset),
    "prithvi":   (build_encoder_prithvi,   EuroSATProbeDataset),
    "hand_rgb":  (build_encoder_hand_rgb,  EuroSATRGB3ChDataset),
}


# ── token-level feature extraction (for spectrum / covariance) ───────────────

@torch.no_grad()
def extract_token_features(encoder, loader, device, dtype, max_samples=None,
                           tokens_per_sample=256, desc=""):
    """
    Returns:
      tokens   [N_sub_tokens, D]  — random subsample of tokens_per_sample tokens
                                    per sample (keeps memory bounded)
      mean_pool[N_samples, D]     — per-sample mean over ALL tokens (for probe)
      labels   [N_samples]
    """
    embed_dim = encoder.backbone.embed_dim
    rng = np.random.default_rng(42)
    tok_chunks, mp_chunks, lab_chunks = [], [], []
    n_samples = 0
    total = len(loader)
    for bi, (imgs, labels, doys) in enumerate(loader, 1):
        imgs = imgs.to(device, dtype=dtype)
        doys = doys.to(device)
        with torch.autocast(device_type=device.type, dtype=dtype):
            z = encoder([imgs], doys=doys, training_mode=False)[0]  # [B, T, D*4 or D]
        z = z[:, :, :embed_dim].float().cpu().numpy()                # [B, T, D]
        B, T, _ = z.shape
        mp_chunks.append(z.mean(axis=1))
        if tokens_per_sample is None or tokens_per_sample >= T:
            tok_chunks.append(z.reshape(-1, embed_dim))
        else:
            idx = rng.choice(T, size=tokens_per_sample, replace=False)
            tok_chunks.append(z[:, idx, :].reshape(-1, embed_dim))
        lab_chunks.append(labels.numpy())
        n_samples += B
        if bi % 25 == 0 or bi == total:
            print(f"  {desc} [{bi}/{total}]  samples={n_samples}", end="\r")
        if max_samples is not None and n_samples >= max_samples:
            break
    print()
    return (np.concatenate(tok_chunks),
            np.concatenate(mp_chunks),
            np.concatenate(lab_chunks))


# ── diagnostic metrics ───────────────────────────────────────────────────────

def token_covariance(tokens: np.ndarray) -> np.ndarray:
    """Sample covariance Σ ∈ R^{D×D} from token matrix [N, D] (centered)."""
    centered = tokens - tokens.mean(axis=0, keepdims=True)
    return (centered.T @ centered) / max(len(centered) - 1, 1)


def effective_rank(cov: np.ndarray) -> dict:
    """
    Three rank-style summaries:
      stable_rank  = ‖Σ‖_F^2 / ‖Σ‖_2^2          (Vershynin)
      entropy_rank = exp(H(p)) where p_i = λ_i / Σλ_i  (participation ratio of eigvals)
      n_eff_99     = #eigvals needed to capture 99% variance
    """
    eigvals = np.linalg.eigvalsh(cov.astype(np.float64))
    eigvals = np.clip(eigvals, 0.0, None)
    fro_sq = float((eigvals ** 2).sum())
    spec_sq = float(eigvals.max() ** 2) if eigvals.size else 1.0
    stable = fro_sq / max(spec_sq, 1e-12)

    total = float(eigvals.sum())
    if total > 0:
        p = eigvals / total
        nz = p[p > 0]
        entropy = float(-np.sum(nz * np.log(nz)))
        entropy_rank = float(np.exp(entropy))
        cum = np.cumsum(np.sort(p)[::-1])
        n_eff_99 = int(np.searchsorted(cum, 0.99) + 1)
    else:
        entropy_rank = 0.0
        n_eff_99 = 0

    return {
        "stable_rank":  stable,
        "entropy_rank": entropy_rank,
        "n_eff_99":     n_eff_99,
        "lambda_max":   float(eigvals.max()) if eigvals.size else 0.0,
        "lambda_min":   float(eigvals.min()) if eigvals.size else 0.0,
        "trace":        total,
    }


def top_k_spectrum(cov: np.ndarray, k: int = 128) -> np.ndarray:
    """Top-k eigenvalues of Σ in descending order, padded with zeros."""
    eigvals = np.linalg.eigvalsh(cov.astype(np.float64))
    sorted_eigs = np.sort(eigvals)[::-1]
    out = np.zeros(k)
    out[:min(k, sorted_eigs.size)] = sorted_eigs[:min(k, sorted_eigs.size)]
    return out


def pairwise_eps(covs: dict[str, np.ndarray]) -> dict[tuple[str, str], float]:
    """
    Frobenius distance between covariance matrices, normalised by the
    smaller trace so values are scale-invariant. Lower = closer feature
    distribution.
    """
    out = {}
    names = sorted(covs.keys())
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sa, sb = covs[a], covs[b]
            denom = min(np.trace(sa), np.trace(sb))
            denom = max(float(denom), 1e-12)
            out[(a, b)] = float(np.linalg.norm(sa - sb, ord="fro") / denom)
    return out


# ── linear probe (minimal, no caching) ───────────────────────────────────────

def linear_probe_simple(feats_tr, lab_tr, feats_te, lab_te) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(feats_tr)
    X_te = scaler.transform(feats_te)
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                             multi_class="multinomial", n_jobs=-1)
    clf.fit(X_tr, lab_tr)
    return float((clf.predict(X_te) == lab_te).mean())


# ── reporting ────────────────────────────────────────────────────────────────

def write_summary_md(path: Path, results: dict, eps: dict):
    lines = ["# M0 Diagnostics Summary", ""]
    lines.append("## Per-config metrics")
    lines.append("")
    lines.append("| Config | Probe acc | Stable rank | Entropy rank | n@99% var | trace |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, r in results.items():
        s = r["effective_rank"]
        lines.append(
            f"| {name} | {r['probe_acc']*100:.2f}% "
            f"| {s['stable_rank']:.1f} "
            f"| {s['entropy_rank']:.1f} "
            f"| {s['n_eff_99']} "
            f"| {s['trace']:.3e} |"
        )

    lines.append("")
    lines.append("## Pairwise covariance distance ε (Frobenius / min-trace)")
    lines.append("")
    lines.append("| pair | ε |")
    lines.append("|---|---:|")
    for (a, b), v in eps.items():
        lines.append(f"| {a} ↔ {b} | {v:.4f} |")

    lines.append("")
    lines.append("Notes:")
    lines.append("- Stable rank, entropy rank: higher = features span more directions (less collapse)")
    lines.append("- ε pairwise: lower = two adapters produce more similar token distributions")
    lines.append("- Probe acc: standard StandardScaler + LogisticRegression on EuroSAT-MS")
    path.write_text("\n".join(lines), encoding="utf-8")


def maybe_plot(out_dir: Path, results: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed; skipping plots")
        return

    # Spectrum overlay (log-log)
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, r in results.items():
        spec = r["spectrum"]
        ax.loglog(np.arange(1, len(spec) + 1), np.maximum(spec, 1e-12), label=name)
    ax.set_xlabel("eigenvalue rank")
    ax.set_ylabel("eigenvalue")
    ax.set_title("Token covariance spectrum (top-128)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "m0_spectrum.png", dpi=150)
    plt.close(fig)

    # Probe acc vs effective rank scatter
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, r in results.items():
        ax.scatter(r["effective_rank"]["stable_rank"], r["probe_acc"] * 100,
                   s=80, label=name)
    ax.set_xlabel("stable rank")
    ax.set_ylabel("EuroSAT linear-probe top-1 (%)")
    ax.set_title("Probe accuracy vs feature rank")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "m0_scatter.png", dpi=150)
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True,
                        help="V-JEPA 2.1 pretrained or fine-tuned checkpoint")
    parser.add_argument("--data_dir", required=True,
                        help="Parent dir containing eurosat/ subfolder")
    parser.add_argument("--output_dir", default="m0_results")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=2000,
                        help="Cap per-split EuroSAT samples for fast diagnostics")
    parser.add_argument("--tokens_per_sample", type=int, default=256,
                        help="Random subsample tokens per sample for covariance "
                             "(None or >= 1536 = use all 1536 V-JEPA tokens; "
                             "default 256 keeps cov memory bounded)")
    parser.add_argument("--spectrum_k", type=int, default=128)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()),
                        help=f"Subset of {list(CONFIGS.keys())} to evaluate")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if cfg["meta"].get("dtype") == "bfloat16" else torch.float32
    eurosat_dir = Path(args.data_dir) / "eurosat"
    if not eurosat_dir.exists():
        raise FileNotFoundError(f"EuroSAT not found at {eurosat_dir}")

    print(f"Device: {device}  dtype: {dtype}  configs: {args.configs}\n")

    results: dict = {}
    cov_per_config: dict = {}

    for name in args.configs:
        if name not in CONFIGS:
            raise ValueError(f"Unknown config {name!r}; choose from {list(CONFIGS)}")
        builder, dataset_cls = CONFIGS[name]
        print(f"── config: {name} ──")
        encoder = builder(cfg, args.checkpoint, device)

        ds_tr = dataset_cls(str(eurosat_dir), "train")
        ds_te = dataset_cls(str(eurosat_dir), "test")
        ldr_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        ldr_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

        toks_tr, mp_tr, lab_tr = extract_token_features(
            encoder, ldr_tr, device, dtype, args.max_samples,
            tokens_per_sample=args.tokens_per_sample, desc=f"{name}-train")
        _, mp_te, lab_te = extract_token_features(
            encoder, ldr_te, device, dtype, args.max_samples,
            tokens_per_sample=args.tokens_per_sample, desc=f"{name}-test")

        cov = token_covariance(toks_tr)
        cov_per_config[name] = cov
        rank = effective_rank(cov)
        spectrum = top_k_spectrum(cov, k=args.spectrum_k)
        probe_acc = linear_probe_simple(mp_tr, lab_tr, mp_te, lab_te)
        print(f"  probe acc = {probe_acc*100:.2f}%   "
              f"stable_rank = {rank['stable_rank']:.1f}   "
              f"n@99% = {rank['n_eff_99']}\n")

        results[name] = {
            "probe_acc":      probe_acc,
            "effective_rank": rank,
            "spectrum":       spectrum.tolist(),
            "n_train_tokens": int(len(toks_tr)),
        }

        # Free memory before next config
        del encoder, toks_tr, mp_tr, mp_te, lab_tr, lab_te
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    eps = pairwise_eps(cov_per_config)

    # JSON dump (eps tuple keys → string)
    eps_serial = {f"{a}__{b}": v for (a, b), v in eps.items()}
    (out_dir / "m0_results.json").write_text(json.dumps(
        {"results": results, "eps": eps_serial,
         "checkpoint": args.checkpoint, "config": args.config},
        indent=2,
    ))
    write_summary_md(out_dir / "m0_summary.md", results, eps)
    maybe_plot(out_dir, results)

    print("=" * 56)
    print("M0 diagnostics complete")
    print(f"  JSON     : {out_dir / 'm0_results.json'}")
    print(f"  Summary  : {out_dir / 'm0_summary.md'}")
    print(f"  Spectrum : {out_dir / 'm0_spectrum.png'}")
    print(f"  Scatter  : {out_dir / 'm0_scatter.png'}")
    print("=" * 56)


if __name__ == "__main__":
    main()
