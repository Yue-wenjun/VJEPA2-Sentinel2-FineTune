"""
V-JEPA 2.1 × Sentinel-2 Fine-Tuning Entry Point

Usage:
    python finetune_main.py --config vjepa2/configs/finetune/vitl16/sentinel2-224px-8f.yaml

Pipeline:
  1. Sentinel2Dataset yields ([buffer], label, clip_indices, doy_tensor)
  2. MaskCollator (as collate_fn) batches samples and generates JEPA masks
  3. 6-channel encoder loaded from pretrained weights via Prithvi-style init
  4. DOY tensor injected into ViT forward pass for seasonal positional encoding
  5. Three-stage training: freeze → partial unfreeze → full fine-tune
"""

import argparse
import copy
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "vjepa2"))

import app.vjepa_2_1.models.predictor as vit_pred
import app.vjepa_2_1.models.vision_transformer as video_vit
from app.vjepa_2_1.wrappers import MultiSeqWrapper, PredictorMultiSeqWrapper
from data_pipeline.patch_embed_6ch import build_6ch_patch_embed_from_pretrained
from data_pipeline.sentinel2_dataset import Sentinel2Dataset
from src.masks.multiseq_multiblock3d import MaskCollator
from src.masks.utils import apply_masks
from src.utils.schedulers import CosineWDSchedule, WarmupCosineSchedule

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── model ────────────────────────────────────────────────────────────────────

def build_model(cfg: dict, device: torch.device):
    m = cfg["model"]
    d = cfg["data"]
    s2 = cfg["sentinel2"]

    encoder_backbone = video_vit.__dict__[m["model_name"]](
        img_size=d["crop_size"],
        patch_size=d["patch_size"],
        num_frames=d["frames_per_clip"],
        tubelet_size=d["tubelet_size"],
        in_chans=s2["in_chans"],
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


def load_pretrained_weights(encoder, predictor, ckpt_path: str, device: torch.device):
    """
    Load V-JEPA 2.1 pretrained weights.
    enc_state keys have no "encoder." prefix (already inside ckpt["encoder"]).
    patch_embed is re-initialised via Prithvi-style 3ch→6ch channel averaging.
    doy_encoding stays at random init (new module, not in pretrained ckpt).
    """
    log.info(f"Loading pretrained checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    enc_state = ckpt.get("encoder", ckpt)

    # Prithvi-style patch_embed init: keys are "patch_embed.proj.weight/bias"
    new_patch_embed = build_6ch_patch_embed_from_pretrained(
        pretrained_state_dict=enc_state,
        patch_size=encoder.backbone.patch_size,
        tubelet_size=encoder.backbone.tubelet_size,
        embed_dim=encoder.backbone.embed_dim,
    ).to(device)
    encoder.backbone.patch_embed = new_patch_embed

    # Load remaining weights; mismatched keys (patch_embed, doy_encoding) skipped
    msg = encoder.backbone.load_state_dict(enc_state, strict=False)
    log.info(f"Encoder: {msg}")

    if "predictor" in ckpt:
        msg = predictor.backbone.load_state_dict(ckpt["predictor"], strict=False)
        log.info(f"Predictor: {msg}")


# ── freeze / unfreeze ─────────────────────────────────────────────────────────

def set_freeze_stage(encoder, stage_cfg: dict):
    freeze = stage_cfg.get("freeze_backbone", False)
    n_unfreeze = stage_cfg.get("unfreeze_last_n_blocks", -1)
    backbone = encoder.backbone

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
):
    encoder.train()
    predictor.train()
    target_encoder.eval()

    total_loss, n_batches = 0.0, 0

    # MaskCollator returns list of (collated_batch, masks_enc, masks_pred) per fpc
    for fpc_collations in loader:
        for collated_batch, masks_enc, masks_pred in fpc_collations:
            # collated_batch is [buffers, labels, clip_indices, doys]
            # buffers: list of [B, C, T, H, W] tensors (one per clip count)
            clips = collated_batch[0]    # list length 1 for our dataset
            doys = collated_batch[3]     # [B, T] int32

            x = clips[0].to(device, non_blocking=True)      # [B, 6, T, H, W]
            doys = doys.to(device, non_blocking=True)        # [B, T]
            masks_enc = [[m.to(device) for m in me] for me in masks_enc]
            masks_pred = [[m.to(device) for m in mp] for mp in masks_pred]

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=dtype):
                # Context encoding (masked)
                z_ctx = encoder([x], masks=masks_enc, doys=doys)

                # Target encoding (EMA, no grad, full sequence)
                with torch.no_grad():
                    z_tgt_full = target_encoder([x], doys=doys)
                    z_tgt = [
                        [apply_masks(z, m) for m in mp]
                        for z, mp in zip(z_tgt_full, masks_pred)
                    ]

                # Predictor
                z_pred = predictor(z_ctx, masks_enc, masks_pred)

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
            if n_batches % 50 == 0:
                log.info(f"  epoch {epoch:04d}  step {n_batches:5d}  loss={loss.item():.4f}")

    return total_loss / max(n_batches, 1)


# ── checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(encoder, predictor, optimizer, epoch, path):
    torch.save({
        "epoch": epoch,
        "encoder": encoder.backbone.state_dict(),
        "predictor": predictor.backbone.state_dict(),
        "opt": optimizer.state_dict(),
    }, path)
    log.info(f"Saved → {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if cfg["meta"].get("dtype") == "bfloat16" else torch.float32
    folder = cfg["folder"]
    os.makedirs(folder, exist_ok=True)

    seed = cfg["meta"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    d_cfg = cfg["data"]
    s2_cfg = cfg["sentinel2"]
    opt_cfg = cfg["optimization"]

    # ── dataset + collator ────────────────────────────────────────────────────
    dataset = Sentinel2Dataset(
        sequences_csv=s2_cfg["sequences_csv"],
        frames_per_clip=d_cfg["frames_per_clip"],
        crop_size=d_cfg["crop_size"],
        max_cloud_frac=s2_cfg.get("max_cloud_frac", 0.30),
    )
    log.info(f"Dataset: {len(dataset)} sequences")

    mask_collator = MaskCollator(
        cfgs_mask=cfg["mask"],
        dataset_fpcs=[d_cfg["frames_per_clip"]],
        crop_size=(d_cfg["crop_size"], d_cfg["crop_size"]),
        patch_size=(d_cfg["patch_size"], d_cfg["patch_size"]),
        tubelet_size=d_cfg["tubelet_size"],
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=mask_collator,
        batch_size=d_cfg["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=d_cfg.get("num_workers", 4),
        pin_memory=d_cfg.get("pin_mem", True),
        persistent_workers=(d_cfg.get("num_workers", 4) > 0),
    )
    log.info(f"DataLoader: {len(loader)} batches/epoch")

    # ── model ─────────────────────────────────────────────────────────────────
    encoder, predictor = build_model(cfg, device)
    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    load_pretrained_weights(encoder, predictor, s2_cfg["pretrained_checkpoint"], device)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    ema = opt_cfg["ema"][0]
    loss_exp = cfg["loss"].get("loss_exp", 1.0)
    save_freq = cfg["meta"].get("save_every_freq", 10)

    # ── 3-stage training ──────────────────────────────────────────────────────
    global_epoch = 0
    for stage_name in ("stage1", "stage2", "stage3"):
        stage_cfg = opt_cfg[stage_name]
        log.info(f"\n{'='*60}\n  {stage_name.upper()}: {stage_cfg['epochs']} epochs\n{'='*60}")

        set_freeze_stage(encoder, stage_cfg)
        optimizer, scheduler, wd_scheduler = build_optimizer(
            encoder, predictor, stage_cfg, ipe=len(loader)
        )

        for _ in range(stage_cfg["epochs"]):
            avg_loss = run_one_epoch(
                encoder, predictor, target_encoder,
                loader, optimizer, scheduler, wd_scheduler,
                scaler, device, dtype,
                ema_momentum=ema,
                loss_exp=loss_exp,
                epoch=global_epoch,
            )
            log.info(f"[{stage_name}] epoch {global_epoch:04d}  avg_loss={avg_loss:.4f}")

            if global_epoch % save_freq == 0:
                save_checkpoint(
                    encoder, predictor, optimizer, global_epoch,
                    os.path.join(folder, f"checkpoint_ep{global_epoch:04d}.pth"),
                )
            global_epoch += 1

    save_checkpoint(
        encoder, predictor, optimizer, global_epoch,
        os.path.join(folder, "checkpoint_final.pth"),
    )
    log.info("Fine-tuning complete.")


if __name__ == "__main__":
    main()
