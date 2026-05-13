# V-JEPA 2.1 × OLMo-Earth Fine-Tuning

Fine-tuning V-JEPA 2.1 on OLMo-Earth Sentinel-2 monthly composites for remote sensing representation learning.

## Quick Reference

Config file used in all commands:
```
vjepa2/configs/finetune/vitl16/olmoearth-256px-12f.yaml
```

---

## Training

### Start (8 GPU)

```bash
screen -S vjepa_train
torchrun --nproc_per_node=8 finetune_main.py \
    --config vjepa2/configs/finetune/vitl16/olmoearth-256px-12f.yaml \
    --run_tag run01 \
    2>&1 | tee train.log
# Ctrl+A D  to detach;  screen -r vjepa_train  to reattach
```

`--run_tag` saves checkpoints to `<folder>/run01/`. Change to `run02`, `run03`, … for each new run to avoid overwriting.

### Resume after disconnect

Edit the yaml:
```yaml
meta:
  load_checkpoint: true
  read_checkpoint: /workspace/checkpoints/run01/checkpoint_ep0005.pth
```
Then rerun the same `torchrun` command.

### Debug (1 GPU, 1 epoch)

```bash
torchrun --nproc_per_node=1 finetune_main.py \
    --config vjepa2/configs/finetune/vitl16/olmoearth-256px-12f.yaml \
    --run_tag debug
```

---

## M1 Ablation Configs

For M1 experiments, keep the base yaml as the template and copy one config per run:

```bash
mkdir -p experiments/configs logs
cp vjepa2/configs/finetune/vitl16/olmoearth-256px-12f.yaml \
   experiments/configs/m1_init_prithvi_doy_on.yaml
```

Useful config switches:

```yaml
model:
  patch_embed_init: prithvi   # random | rgb_mean_copy | prithvi | spectral
  use_doy_encoding: true
  doy_mode: sinusoidal        # none | sinusoidal | learnable | month_token
  modality_embedding: false

optimization:
  stage1:
    train_final_norm: false
    train_all_norms: false
    freeze_predictor: false
```

Run with a matching tag:

```bash
RUN=m1_init_prithvi_doy_on
torchrun --nproc_per_node=8 finetune_main.py \
    --config experiments/configs/${RUN}.yaml \
    --run_tag ${RUN} \
    2>&1 | tee logs/${RUN}.log
```

Record each run in `../week2文档/ablation.md`: config path, code commit/tag, checkpoint path, linear probe command, and result.

---

## M0 Diagnostics

**已迁移到 [code/PFU_Experiments/m0_cross_backbone.py](../PFU_Experiments/m0_cross_backbone.py)**(2026-05-14)。
原 `diagnostics_m0.py` 的 5 个 V-JEPA adapter builder 已 inline 到
`code/PFU_Experiments/backbones/vjepa.py`,并被多 backbone 通用脚本调用。

V-JEPA 5-adapter sweep:

```bash
# 从 code/PFU_Experiments/ 启动
cd ../PFU_Experiments

# Mac local
python m0_cross_backbone.py --backbone vjepa --vjepa_sweep \
    --vjepa_weights ../../model_weights/vjepa2.1/vjepa2_1_vitl_dist_vitG_384.pt \
    --data_dir ../../data \
    --output_dir ../../results/m0_cross/ \
    --max_samples 2000

# AutoDL
python m0_cross_backbone.py --backbone vjepa --vjepa_sweep \
    --vjepa_weights /root/autodl-tmp/model_weights/vjepa2.1/vjepa2_1_vitl_dist_vitG_384.pt \
    --data_dir /root/autodl-tmp/data \
    --output_dir /root/autodl-tmp/results/m0_cross/
```

Outputs:

| File | Content |
|------|---------|
| `results/m0_cross/m0_cross_results.json` | metrics + spectra, per backbone:init |
| `results/m0_cross/cov_<backbone>.npz` | covariance matrices for cross-ε computation |

Default configs are `random rgb_mean_copy prithvi hand_rgb spectral`.
Use `--configs random prithvi hand_rgb` to choose a subset, and `--tokens_per_sample` to control covariance memory.

---

## Visualization (PCA Embeddings)

Outputs PNG files to `./vis/<run_tag>/`. No display required (server-safe).

Always outputs `embeddings_finetuned.png`.
Add `--pretrained` to also output `embeddings_pretrained.png` + `comparison.png` (before/after comparison).

```bash
# Fine-tuned embeddings only (run_tag from yaml)
python visualize.py \
    --config vjepa2/configs/finetune/vitl16/olmoearth-256px-12f.yaml

# + before/after comparison with original pretrained weights
python visualize.py \
    --config vjepa2/configs/finetune/vitl16/olmoearth-256px-12f.yaml \
    --pretrained /workspace/vjepa2_1_vitl_dist_vitG_384.pt

# Different run (without editing yaml)
python visualize.py \
    --config vjepa2/configs/finetune/vitl16/olmoearth-256px-12f.yaml \
    --run_tag run02

# Specific checkpoint (e.g. intermediate epoch)
python visualize.py \
    --config vjepa2/configs/finetune/vitl16/olmoearth-256px-12f.yaml \
    --checkpoint /workspace/checkpoints/run01/checkpoint_ep0005.pth \
    --output_dir /workspace/vis/run01_ep5
```

---

## Linear Probe (EuroSAT-MS + BreizhCrops)

Evaluates frozen encoder features with logistic regression.

**数据集准备**：
```bash
# EuroSAT：在有网的机器下载后 scp 到服务器
wget --no-check-certificate https://madm.dfki.de/files/sentinel/EuroSATallBands.zip
mkdir -p ./data/eurosat/
unzip EuroSATallBands.zip -d ./data/eurosat/

# BreizhCrops：在有网的机器上运行，自动下载 H5 文件（~2-3 GB）
python -c "
from breizhcrops import BreizhCrops as BC
for region in ['frh01', 'frh02', 'frh03', 'frh04']:
    print(f'Downloading {region}...')
    BC(region=region, root='./breizhcrops', year=2017)
print('Done')
"

mv ~/vjepa2/breizhcrops/ /workspace/data/breizhcrops/
```

```bash
# Install dependencies (once)
pip install rasterio breizhcrops scikit-learn

# Both datasets — yaml run_tag (uses checkpoint_final.pth)
python linear_probe.py \
    --config vjepa2/configs/finetune/vitl16/olmoearth-256px-12f.yaml \
    --dataset both \
    --data_dir /workspace/data

# EuroSAT only — specific checkpoint
python linear_probe.py \
    --config vjepa2/configs/finetune/vitl16/olmoearth-256px-12f.yaml \
    --checkpoint /workspace/checkpoints/run03/checkpoint_ep0000.pth \
    --dataset eurosat \
    --data_dir /workspace/data

# BreizhCrops only — specific checkpoint
python linear_probe.py \
    --config vjepa2/configs/finetune/vitl16/olmoearth-256px-12f.yaml \
    --checkpoint /workspace/checkpoints/run03/checkpoint_ep0000.pth \
    --dataset breizhcrops \
    --data_dir /workspace/data

# Different run (without editing yaml)
python linear_probe.py \
    --config vjepa2/configs/finetune/vitl16/olmoearth-256px-12f.yaml \
    --run_tag run02 \
    --dataset both \
    --data_dir /workspace/data

# Pretrained baseline (before fine-tuning, for comparison)
python linear_probe.py \
    --config vjepa2/configs/finetune/vitl16/olmoearth-256px-12f.yaml \
    --checkpoint /workspace/vjepa2_1_vitl_dist_vitG_384.pt \
    --run_tag pretrained \
    --dataset eurosat \
    --data_dir /workspace/data

# Re-extract features (ignore .npz cache)
python linear_probe.py \
    --config vjepa2/configs/finetune/vitl16/olmoearth-256px-12f.yaml \
    --checkpoint /workspace/checkpoints/run03/checkpoint_ep0000.pth \
    --dataset both \
    --data_dir /workspace/data \
    --no_cache
```

Features are cached as `.npz` under `<checkpoint_folder>/<run_tag>/probe_results/`. Subsequent runs skip the encoder forward pass.

### Pooling / PCA Ablation

```bash
CKPT=/workspace/checkpoints/run03/checkpoint_ep0000.pth
CFG=vjepa2/configs/finetune/vitl16/olmoearth-256px-12f.yaml
DATA=/workspace/data

# global mean-pool 1024-dim (baseline)
python linear_probe.py --config $CFG --checkpoint $CKPT --dataset eurosat --data_dir $DATA

# global + PCA 128
python linear_probe.py --config $CFG --checkpoint $CKPT --dataset eurosat --data_dir $DATA \
    --pca_dim 128

# temporal pool → spatial mean → 1024-dim
python linear_probe.py --config $CFG --checkpoint $CKPT --dataset eurosat --data_dir $DATA \
    --pool_mode temporal

# temporal pool → per-token PCA 128 → spatial mean → 128-dim
python linear_probe.py --config $CFG --checkpoint $CKPT --dataset eurosat --data_dir $DATA \
    --pool_mode temporal --pca_dim 128
```

| pool_mode | pca_dim | Feature dim | EuroSAT Top-1 |
|-----------|---------|-------------|---------------|
| global | — | 1024 | 97.23% |
| global | 128 | 128 | 96.25% |
| temporal | — | 1024 | = global |
| temporal | 128 | 128 | 95.90% |

---

## Segmentation (AWF Land Cover)

End-to-end fine-tuning with a lightweight decoder head. 2-stage training:
- Stage 1 (10 epochs): freeze encoder, train decoder only
- Stage 2 (30 epochs): unfreeze encoder, end-to-end with lower encoder LR

**AWF 数据准备**：
```bash
# AWF：在有网的机器下载后 scp 到服务器
pip install huggingface_hub
hf download allenai/olmoearth_projects_awf \
    --repo-type dataset --local-dir ./awf_raw
mkdir -p ./data/awf/
tar -xf ./awf_raw/dataset.tar -C ./data/awf/
cp ./awf_raw/annotation_features.geojson ./data/awf/
```

```bash
# AWF — 完整 2-stage 训练（10 frozen + 30 unfrozen epochs）
python segmentation.py \
    --config vjepa2/configs/finetune/vitl16/olmoearth-256px-12f.yaml \
    --checkpoint /workspace/checkpoints/run03/checkpoint_ep0000.pth \
    --dataset awf \
    --data_dir /workspace/data

# AWF — 训练前先跑 kNN baseline（评估 encoder 特征质量，k=20 cosine）
python segmentation.py \
    --config vjepa2/configs/finetune/vitl16/olmoearth-256px-12f.yaml \
    --checkpoint /workspace/checkpoints/run03/checkpoint_ep0000.pth \
    --dataset awf \
    --data_dir /workspace/data \
    --eval_knn

# EuroSAT — 用已有数据快速验证 decoder 架构
python segmentation.py \
    --config vjepa2/configs/finetune/vitl16/olmoearth-256px-12f.yaml \
    --checkpoint /workspace/checkpoints/run03/checkpoint_ep0000.pth \
    --dataset eurosat \
    --data_dir /workspace/data

# 自定义 epoch / LR
python segmentation.py \
    --config vjepa2/configs/finetune/vitl16/olmoearth-256px-12f.yaml \
    --checkpoint /workspace/checkpoints/run03/checkpoint_ep0000.pth \
    --dataset awf \
    --data_dir /workspace/data \
    --freeze_epochs 10 \
    --unfreeze_epochs 30 \
    --lr 1e-4 \
    --encoder_lr_scale 0.1

# 仅 stage 1（decoder-only，快速基线）
python segmentation.py \
    --config vjepa2/configs/finetune/vitl16/olmoearth-256px-12f.yaml \
    --checkpoint /workspace/checkpoints/run03/checkpoint_ep0000.pth \
    --dataset awf \
    --data_dir /workspace/data \
    --freeze_epochs 10 \
    --unfreeze_epochs 0
```

Results saved to `<checkpoint_folder>/<run_tag>/seg_results/`.

---

## Checkpoint Path Logic

All scripts resolve the checkpoint path the same way:

| Argument | Resolved checkpoint |
|----------|-------------------|
| `--run_tag run01` | `<folder>/run01/checkpoint_final.pth` |
| `--checkpoint /path/x.pth` | `/path/x.pth` (exact) |
| neither (yaml has `run_tag`) | `<folder>/<yaml_run_tag>/checkpoint_final.pth` |

`--checkpoint` takes priority over `--run_tag`.

---

## Key Files

| File | Purpose |
|------|---------|
| `finetune_main.py` | Training entry point (3-stage freeze/unfreeze, EMA, JEPA loss) |
| `visualize.py` | PCA patch embedding figures (server, no display) |
| `linear_probe.py` | Frozen linear probe on EuroSAT-MS + BreizhCrops |
| `segmentation.py` | End-to-end segmentation fine-tuning (AWF / EuroSAT) |
| (`diagnostics_m0.py` → moved to `code/PFU_Experiments/m0_cross_backbone.py` on 2026-05-14) |
| `vjepa2/configs/finetune/vitl16/olmoearth-256px-12f.yaml` | Training config |
| `fine_tune.md` | Detailed design notes (LLRD, best-of-stage restore, data pipeline) |
