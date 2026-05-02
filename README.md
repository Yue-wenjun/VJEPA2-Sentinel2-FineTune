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
  read_checkpoint: /home/baai/vjepa2/checkpoints/run01/checkpoint_ep0005.pth
```
Then rerun the same `torchrun` command.

### Debug (1 GPU, 1 epoch)

```bash
torchrun --nproc_per_node=1 finetune_main.py \
    --config vjepa2/configs/finetune/vitl16/olmoearth-256px-12f.yaml \
    --run_tag debug
```

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
    --pretrained /home/baai/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt

# Different run (without editing yaml)
python visualize.py \
    --config vjepa2/configs/finetune/vitl16/olmoearth-256px-12f.yaml \
    --run_tag run02

# Specific checkpoint (e.g. intermediate epoch)
python visualize.py \
    --config vjepa2/configs/finetune/vitl16/olmoearth-256px-12f.yaml \
    --checkpoint /home/baai/vjepa2/checkpoints/run01/checkpoint_ep0005.pth \
    --output_dir /home/baai/vjepa2/vis/run01_ep5
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

mv ~/vjepa2/breizhcrops/ /home/baai/vjepa2/data/breizhcrops/
```

```bash
# Install dependencies (once)
pip install rasterio breizhcrops scikit-learn

# Both datasets — yaml run_tag (uses checkpoint_final.pth)
python linear_probe.py \
    --config vjepa2/configs/finetune/vitl16/olmoearth-256px-12f.yaml \
    --dataset both \
    --data_dir /home/baai/vjepa2/data

# EuroSAT only — specific checkpoint
python linear_probe.py \
    --config vjepa2/configs/finetune/vitl16/olmoearth-256px-12f.yaml \
    --checkpoint /home/baai/vjepa2/checkpoints/run03/checkpoint_ep0000.pth \
    --dataset eurosat \
    --data_dir /home/baai/vjepa2/data

# BreizhCrops only — specific checkpoint
python linear_probe.py \
    --config vjepa2/configs/finetune/vitl16/olmoearth-256px-12f.yaml \
    --checkpoint /home/baai/vjepa2/checkpoints/run03/checkpoint_ep0000.pth \
    --dataset breizhcrops \
    --data_dir /home/baai/vjepa2/data

# Different run (without editing yaml)
python linear_probe.py \
    --config vjepa2/configs/finetune/vitl16/olmoearth-256px-12f.yaml \
    --run_tag run02 \
    --dataset both \
    --data_dir /home/baai/vjepa2/data

# Pretrained baseline (before fine-tuning, for comparison)
python linear_probe.py \
    --config vjepa2/configs/finetune/vitl16/olmoearth-256px-12f.yaml \
    --checkpoint /home/baai/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt \
    --run_tag pretrained \
    --dataset eurosat \
    --data_dir /home/baai/vjepa2/data

# Re-extract features (ignore .npz cache)
python linear_probe.py \
    --config vjepa2/configs/finetune/vitl16/olmoearth-256px-12f.yaml \
    --checkpoint /home/baai/vjepa2/checkpoints/run03/checkpoint_ep0000.pth \
    --dataset both \
    --data_dir /home/baai/vjepa2/data \
    --no_cache
```

Features are cached as `.npz` under `<checkpoint_folder>/<run_tag>/probe_results/`. Subsequent runs skip the encoder forward pass.

---

## Checkpoint Path Logic

All three scripts resolve the checkpoint path the same way:

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
| `vjepa2/configs/finetune/vitl16/olmoearth-256px-12f.yaml` | Training config |
| `fine_tune.md` | Detailed design notes (LLRD, best-of-stage restore, data pipeline) |
