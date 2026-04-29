"""
V-JEPA 2.1 × OLMo-Earth Fine-Tuning Entry Point

Usage:
    python finetune_main.py --config vjepa2/configs/finetune/vitl16/olmoearth-256px-12f.yaml

Pipeline:
  1. OLMoEarthDataset streams local TAR shards → ([buffer], label, doy_tensor, clip_indices)
  2. MaskCollator (as collate_fn) batches samples and generates JEPA masks
  3. N-channel encoder loaded from pretrained weights via Prithvi-style init
  4. DOY tensor injected into ViT forward pass for seasonal positional encoding
  5. Three-stage training: freeze → partial unfreeze → full fine-tune
"""

import argparse
import copy
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, str(Path(__file__).parent / "vjepa2"))

import app.vjepa_2_1.models.predictor as vit_pred
import app.vjepa_2_1.models.vision_transformer as video_vit
from app.vjepa_2_1.wrappers import MultiSeqWrapper, PredictorMultiSeqWrapper
from data_pipeline.patch_embed_6ch import build_nch_patch_embed_from_pretrained
from data_pipeline.olmoearth_dataset import OLMoEarthDataset
from src.masks.multiseq_multiblock3d import MaskCollator
from src.masks.utils import apply_masks
from src.utils.schedulers import CosineWDSchedule, WarmupCosineSchedule

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# PyTorch 2.1.x has a bug in collate_tensor_fn: it passes element count (not byte
# count) to untyped_storage()._new_shared(), allocating 4× too few bytes for float32,
# then calls resize_() on the resulting non-resizable file-backed shared memory.
# Fix: replace the pre-allocation path with a plain torch.stack (same result, no shm).
try:
    import torch.utils.data._utils.collate as _pt_collate
    if torch.Tensor in _pt_collate.default_collate_fn_map:
        _pt_collate.default_collate_fn_map[torch.Tensor] = (
            lambda batch, **_: torch.stack(batch, 0)
        )
except Exception:
    pass


# ── config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── model ────────────────────────────────────────────────────────────────────

def build_model(cfg: dict, device: torch.device):
    m = cfg["model"]
    d = cfg["data"]
    in_chans = m["in_chans"]

    encoder_backbone = video_vit.__dict__[m["model_name"]](
        img_size=d["crop_size"],
        patch_size=d["patch_size"],
        num_frames=d["frames_per_clip"],
        tubelet_size=d["tubelet_size"],
        in_chans=in_chans,
        use_doy_encoding=m.get("use_doy_encoding", True),
        use_rope=m.get("use_rope", False),
        uniform_power=m.get("uniform_power", True),
        use_sdpa=m.get("use_sdpa", True),
        use_activation_checkpointing=m.get("use_activation_checkpointing", False),
        modality_embedding=m.get("modality_embedding", False),
        has_cls_first=m.get("has_cls_first", False),
        n_registers=m.get("n_registers", 0),
    )

    predictor_backbone = vit_pred.__dict__["vit_predictor"](
        img_size=d["crop_size"],
        patch_size=d["patch_size"],
        num_frames=d["frames_per_clip"],
        tubelet_size=d["tubelet_size"],
        embed_dim=encoder_backbone.embed_dim,
        predictor_embed_dim=m.get("pred_embed_dim", 384),
        depth=m.get("pred_depth", 12),
        num_heads=m.get("pred_num_heads", 12),
        uniform_power=m.get("uniform_power", True),
        use_mask_tokens=m.get("use_mask_tokens", True),
        zero_init_mask_tokens=m.get("zero_init_mask_tokens", True),
        use_rope=m.get("use_rope", False),
        use_sdpa=m.get("use_sdpa", True),
        use_activation_checkpointing=m.get("use_activation_checkpointing", False),
        n_registers=m.get("n_registers_predictor", 0),
        has_cls_first=m.get("has_cls_first", False),
        modality_embedding=m.get("modality_embedding", False),
    )

    encoder = MultiSeqWrapper(encoder_backbone).to(device)
    predictor = PredictorMultiSeqWrapper(predictor_backbone).to(device)
    return encoder, predictor


def _strip_prefix(state_dict: dict) -> dict:
    """Strip DDP/wrapper key prefixes: 'module.backbone.' or 'module.' or 'backbone.'."""
    if not state_dict:
        return state_dict
    sample = next(iter(state_dict))
    for prefix in ("module.backbone.", "module.", "backbone."):
        if sample.startswith(prefix):
            return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def _safe_load(module: nn.Module, state_dict: dict):
    """load_state_dict that silently skips shape-mismatched keys (strict=False)."""
    own = module.state_dict()
    compatible = {k: v for k, v in state_dict.items()
                  if k not in own or own[k].shape == v.shape}
    return module.load_state_dict(compatible, strict=False)


def _backbone(model):
    """Unwrap DDP wrapper and return the .backbone module."""
    m = model.module if hasattr(model, "module") else model
    return m.backbone


def _make_ddp(enc, pred, local_rank, find_unused):
    """(Re-)wrap encoder and predictor in DDP; rebuilt at each stage so that
    find_unused_parameters can be turned off once all params are trainable."""
    raw_enc  = enc.module  if hasattr(enc,  "module") else enc
    raw_pred = pred.module if hasattr(pred, "module") else pred
    return (
        DistributedDataParallel(raw_enc,  device_ids=[local_rank],
                                find_unused_parameters=find_unused,
                                gradient_as_bucket_view=True),
        DistributedDataParallel(raw_pred, device_ids=[local_rank],
                                find_unused_parameters=find_unused,
                                gradient_as_bucket_view=True),
    )


def load_pretrained_weights(encoder, predictor, ckpt_path: str, device: torch.device, in_chans: int = 6):
    """
    Load V-JEPA 2.1 pretrained weights.
    Handles DDP-wrapped checkpoints (module.backbone.* prefix).
    patch_embed is re-initialised via Prithvi-style 3ch→Nch channel averaging.
    Predictor: loaded with shape-safe filter (checkpoint may differ in dist. architecture).
    doy_encoding: stays at random init (new module, not in pretrained ckpt).
    """
    log.info(f"Loading pretrained checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    enc_state = _strip_prefix(ckpt.get("encoder", ckpt))

    # Prithvi-style patch_embed init: keys are "patch_embed.proj.weight/bias"
    new_patch_embed = build_nch_patch_embed_from_pretrained(
        pretrained_state_dict=enc_state,
        in_chans=in_chans,
        patch_size=encoder.backbone.patch_size,
        tubelet_size=encoder.backbone.tubelet_size,
        embed_dim=encoder.backbone.embed_dim,
    ).to(device)
    encoder.backbone.patch_embed = new_patch_embed

    # Exclude patch_embed (shape [D,3,t,p,p] vs our [D,N,t,p,p]) and load rest.
    # doy_encoding and pos_embed (RoPE ckpt has neither) → expected missing keys.
    enc_state_filtered = {k: v for k, v in enc_state.items()
                          if not k.startswith("patch_embed")}
    msg = _safe_load(encoder.backbone, enc_state_filtered)
    missing_blocks = [k for k in msg.missing_keys
                      if not k.startswith(("doy_encoding", "patch_embed", "pos_embed"))]
    if missing_blocks:
        log.warning(f"Encoder unexpected missing keys: {missing_blocks[:5]}")
    log.info(f"Encoder loaded — missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")

    if "predictor" in ckpt:
        # Checkpoint predictor may have been built for distillation (different out_dim).
        # Load whatever shapes match; skip the rest — predictor adapts quickly.
        pred_state = _strip_prefix(ckpt["predictor"])
        msg = _safe_load(predictor.backbone, pred_state)
        log.info(f"Predictor loaded — missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")


# ── freeze / unfreeze ─────────────────────────────────────────────────────────

def set_freeze_stage(encoder, stage_cfg: dict):
    freeze = stage_cfg.get("freeze_backbone", False)
    n_unfreeze = stage_cfg.get("unfreeze_last_n_blocks", -1)
    backbone = _backbone(encoder)

    for p in backbone.parameters():
        p.requires_grad = not freeze

    # patch_embed and doy_encoding always trainable
    for p in backbone.patch_embed.parameters():
        p.requires_grad = True
    if backbone.doy_encoding is not None:
        for p in backbone.doy_encoding.parameters():
            p.requires_grad = True

    if freeze and n_unfreeze > 0:
        for blk in list(backbone.blocks)[-n_unfreeze:]:
            for p in blk.parameters():
                p.requires_grad = True

    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    total = sum(p.numel() for p in encoder.parameters())
    log.info(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


# ── optimizer ─────────────────────────────────────────────────────────────────

def build_optimizer(encoder, predictor, stage_cfg: dict, ipe: int):
    lr = stage_cfg["lr"]
    epochs = stage_cfg["epochs"]

    def _split(model):
        wd_params = [p for n, p in model.named_parameters()
                     if p.requires_grad and "bias" not in n and p.ndim != 1]
        no_wd = [p for n, p in model.named_parameters()
                 if p.requires_grad and ("bias" in n or p.ndim == 1)]
        return wd_params, no_wd

    enc_wd, enc_nowd = _split(encoder)
    pred_wd, pred_nowd = _split(predictor)

    optimizer = torch.optim.AdamW([
        {"params": enc_wd + pred_wd},
        {"params": enc_nowd + pred_nowd, "weight_decay": 0},
    ], lr=lr, betas=(0.9, 0.999), eps=1e-8,
       weight_decay=stage_cfg.get("weight_decay", 0.05))

    T = epochs * ipe
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=stage_cfg.get("warmup", 0) * ipe,
        start_lr=stage_cfg.get("start_lr", lr),
        ref_lr=lr,
        final_lr=stage_cfg.get("final_lr", 1e-6),
        T_max=T,
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=stage_cfg.get("weight_decay", 0.05),
        final_wd=stage_cfg.get("final_weight_decay", 0.05),
        T_max=T,
    )
    return optimizer, scheduler, wd_scheduler


# ── training loop ─────────────────────────────────────────────────────────────

def run_one_epoch(
    encoder, predictor, target_encoder,
    loader,
    optimizer, scheduler, wd_scheduler,
    scaler, device, dtype,
    ema_momentum: float,
    loss_exp: float,
    epoch: int,
    rank0: bool = True,
):
    encoder.train()
    predictor.train()
    target_encoder.eval()

    total_loss, n_batches = 0.0, 0

    # MaskCollator returns list of (collated_batch, masks_enc, masks_pred) per fpc
    for fpc_collations in loader:
        for collated_batch, masks_enc, masks_pred in fpc_collations:
            # collated_batch is [buffers, labels, doys, clip_indices]
            # buffers: list of [B, C, T, H, W] tensors (one per clip count)
            clips = collated_batch[0]    # list length 1 for our dataset
            doys = collated_batch[2]     # [B, T] int32

            x = clips[0].to(device, non_blocking=True)      # [B, C, T, H, W]
            doys = doys.to(device, non_blocking=True)        # [B, T]
            # MaskCollator returns [gen0[B,K0], gen1[B,K1]]; wrap in clip list: [[gen0, gen1]]
            masks_enc  = [[m.to(device) for m in masks_enc]]
            masks_pred = [[m.to(device) for m in masks_pred]]

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=dtype):
                # Context encoding (masked); training_mode=True returns hierarchical
                # features [B, N_ctx, 4*embed_dim] required by predictor_embed.
                z_ctx = encoder([x], masks=masks_enc, doys=doys, training_mode=True)

                # Target encoding (EMA, no grad, full sequence); same hierarchical
                # output so z_tgt dims match z_pred after predictor_proj.
                with torch.no_grad():
                    z_tgt_full = target_encoder([x], doys=doys, training_mode=True)
                    z_tgt = [
                        [apply_masks(z, [m]) for m in mp]
                        for z, mp in zip(z_tgt_full, masks_pred)
                    ]

                # Predictor (returns outs_pred, outs_context; only predictions needed for loss)
                z_pred, _ = predictor(z_ctx, masks_enc, masks_pred)

                # Smooth-L1 JEPA loss
                loss = 0.0
                n = 0
                for zp_list, zt_list in zip(z_pred, z_tgt):
                    for zp, zt in zip(zp_list, zt_list):
                        l = nn.functional.smooth_l1_loss(zp, zt, reduction="none")
                        if loss_exp != 1.0:
                            l = l ** loss_exp
                        loss = loss + l.mean()
                        n += 1
                loss = loss / max(n, 1)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(predictor.parameters()), 1.0
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            wd_scheduler.step()

            # EMA target encoder update
            with torch.no_grad():
                for p_enc, p_tgt in zip(encoder.parameters(), target_encoder.parameters()):
                    p_tgt.data.mul_(ema_momentum).add_((1 - ema_momentum) * p_enc.data)

            total_loss += loss.item()
            n_batches += 1
            if rank0 and n_batches % 50 == 0:
                log.info(f"  epoch {epoch:04d}  step {n_batches:5d}  loss={loss.item():.4f}")

    return total_loss / max(n_batches, 1)


# ── checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(encoder, predictor, optimizer, epoch, path):
    torch.save({
        "epoch": epoch,
        "encoder": _backbone(encoder).state_dict(),
        "predictor": _backbone(predictor).state_dict(),
        "opt": optimizer.state_dict(),
    }, path)
    log.info(f"Saved → {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # ── DDP init ──────────────────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_ddp = world_size > 1
    if is_ddp:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    rank0 = (local_rank == 0)

    cfg = load_config(args.config)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if cfg["meta"].get("dtype") == "bfloat16" else torch.float32
    folder = Path(cfg["folder"])
    if rank0:
        folder.mkdir(parents=True, exist_ok=True)

    seed = cfg["meta"].get("seed", 42)
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)

    d_cfg   = cfg["data"]
    oe_cfg  = cfg["olmoearth"]
    opt_cfg = cfg["optimization"]

    # Linear LR scaling: effective batch = batch_size × world_size.
    # Applied once here so every stage inherits the scaled values.
    if world_size > 1:
        for sname in ("stage1", "stage2", "stage3"):
            s = opt_cfg[sname]
            for k in ("lr", "start_lr", "final_lr"):
                if k in s:
                    s[k] = s[k] * world_size
        if rank0:
            log.info(f"LR linearly scaled ×{world_size} "
                     f"(effective batch {d_cfg['batch_size'] * world_size})")

    # ── dataset + collator ────────────────────────────────────────────────────
    dataset = OLMoEarthDataset(
        tar_path=oe_cfg["tar_path"],
        n_bands_per_timestep=oe_cfg.get("n_bands_per_timestep", 4),
        crop_size=d_cfg["crop_size"],
        dn_scale=oe_cfg.get("dn_scale", 10000.0),
        max_missing_frac=oe_cfg.get("max_missing_frac", 0.10),
        shuffle_buffer=oe_cfg.get("shuffle_buffer", 1000),
        seed=cfg["meta"].get("seed", 42),
    )
    log.info(f"Dataset: OLMoEarthDataset — {len(dataset.tar_files)} TAR shards")

    mask_collator = MaskCollator(
        cfgs_mask=cfg["mask"],
        dataset_fpcs=[d_cfg["frames_per_clip"]],
        crop_size=(d_cfg["crop_size"], d_cfg["crop_size"]),
        patch_size=(d_cfg["patch_size"], d_cfg["patch_size"]),
        tubelet_size=d_cfg["tubelet_size"],
    )

    num_workers = d_cfg.get("num_workers", 4)
    loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=mask_collator,
        batch_size=d_cfg["batch_size"],
        shuffle=False,       # IterableDataset: shuffling handled inside dataset
        drop_last=True,
        num_workers=num_workers,
        pin_memory=d_cfg.get("pin_mem", True),
        persistent_workers=(num_workers > 0),
    )
    log.info("DataLoader ready")

    # ── model ─────────────────────────────────────────────────────────────────
    encoder, predictor = build_model(cfg, device)
    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    ckpt_path = cfg.get("pretrained_checkpoint")
    if ckpt_path:
        load_pretrained_weights(encoder, predictor, ckpt_path, device,
                                in_chans=cfg["model"]["in_chans"])
    else:
        if rank0:
            log.warning("No pretrained_checkpoint specified — training from scratch")

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    ema = opt_cfg["ema"][0]
    loss_exp = cfg["loss"].get("loss_exp", 1.0)
    save_freq = cfg["meta"].get("save_every_freq", 10)

    # ── steps-per-epoch (IterableDataset has no __len__) ─────────────────────
    try:
        ipe = len(loader)
    except TypeError:
        ipe = d_cfg.get("ipe")
        if ipe is None:
            raise ValueError(
                "IterableDataset has no __len__. Set data.ipe in the config "
                "(e.g. ipe: 17830 for 285288 samples / batch_size 16)."
            )
        if rank0:
            log.info(f"IterableDataset: using ipe={ipe} from config (data.ipe)")
    # Each rank sees 1/world_size of the shards, so effective steps/epoch shrinks.
    ipe = max(1, ipe // world_size)

    # ── 3-stage training ──────────────────────────────────────────────────────
    global_epoch = 0
    for stage_name in ("stage1", "stage2", "stage3"):
        stage_cfg = opt_cfg[stage_name]
        if rank0:
            log.info(f"\n{'='*60}\n  {stage_name.upper()}: {stage_cfg['epochs']} epochs\n{'='*60}")

        set_freeze_stage(encoder, stage_cfg)

        # Rebuild DDP at each stage: turn off find_unused_parameters once the
        # backbone is fully unfrozen (Stage 3) to eliminate the per-step scan overhead.
        if is_ddp:
            freeze = stage_cfg.get("freeze_backbone", False)
            encoder, predictor = _make_ddp(encoder, predictor, local_rank, find_unused=freeze)

        optimizer, scheduler, wd_scheduler = build_optimizer(
            encoder, predictor, stage_cfg, ipe=ipe
        )

        # Per-stage early advancement: advance to next stage if loss stops improving.
        # Controlled by optional YAML keys per stage:
        #   early_stop_patience: N    (epochs without improvement before advancing)
        #   min_epochs_before_stop: M (don't check before epoch M, protects warmup)
        patience   = stage_cfg.get("early_stop_patience", None)
        min_ep     = stage_cfg.get("min_epochs_before_stop", 1)
        best_loss_s = float("inf")
        no_improve  = 0

        for ep_idx in range(stage_cfg["epochs"]):
            avg_loss = run_one_epoch(
                encoder, predictor, target_encoder,
                loader, optimizer, scheduler, wd_scheduler,
                scaler, device, dtype,
                ema_momentum=ema,
                loss_exp=loss_exp,
                epoch=global_epoch,
                rank0=rank0,
            )

            # Sync avg_loss across all ranks so every rank makes the identical
            # patience decision — avoids deadlock where rank0 breaks but others wait.
            if is_ddp:
                _lt = torch.tensor(avg_loss, device=device)
                dist.all_reduce(_lt, op=dist.ReduceOp.AVG)
                avg_loss = _lt.item()

            if rank0:
                log.info(f"[{stage_name}] epoch {global_epoch:04d}  avg_loss={avg_loss:.4f}")
                if global_epoch % save_freq == 0:
                    save_checkpoint(
                        encoder, predictor, optimizer, global_epoch,
                        folder / f"checkpoint_ep{global_epoch:04d}.pth",
                    )
            global_epoch += 1

            # Patience check (all ranks run identical logic → break together).
            if patience is not None and ep_idx >= min_ep - 1:
                if avg_loss < best_loss_s - 1e-4:
                    best_loss_s = avg_loss
                    no_improve  = 0
                else:
                    no_improve += 1
                if no_improve >= patience:
                    if rank0:
                        log.info(f"[{stage_name}] early advance: no improvement "
                                 f"for {no_improve} consecutive epochs")
                    break

    if rank0:
        save_checkpoint(
            encoder, predictor, optimizer, global_epoch,
            folder / "checkpoint_final.pth",
        )
        log.info("Fine-tuning complete.")

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
