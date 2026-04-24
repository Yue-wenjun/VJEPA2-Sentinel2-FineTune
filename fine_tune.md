# V-JEPA 2.1 × OLMo-Earth Fine-Tuning Notes

## 目标

用 V-JEPA 2.1 预训练权重在 OLMo-Earth 全球 Sentinel-2 月度合成影像上做 fine-tune，
生成地表时序 embedding，用于下游任务（变化检测、土地分类、物候监测等）。

> Sentinel-2 自采数据 pipeline（sample_patches / download_s2 / cloud_ablation）
> 已归档至 [doc/sentinel2_pipeline/](doc/sentinel2_pipeline/)，暂不维护。

---

## 核心挑战

### 挑战1：时序稀疏与季节性编码

V-JEPA 原设计针对连续视频（~24fps）。OLMo-Earth 提供固定 12 帧月度合成影像，
帧间隔 ~30 天，语义跳变远大于视频。

**解法**：
- 用绝对时间戳（DOY，Day of Year）叠加在 patch embedding 之后
- mid-month DOY：[15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]
- 4维 sinusoidal 编码 → learned Linear 投影至 embed_dim
- 保留原始 temporal PE 不动，DOY 编码作为加法偏置

---

## 波段选择

### 当前：4 波段（10m 分辨率）

OLMo-Earth `10_sentinel2_l2a_monthly` 文件夹仅包含 10m 波段：

| 波段 | 名称 | 中心波长 | 作用 |
|------|------|---------|------|
| B02 | Blue  | 490nm  | RGB 可见光 |
| B03 | Green | 560nm  | RGB 可见光 |
| B04 | Red   | 665nm  | RGB 可见光 |
| B08 | NIR   | 842nm  | 植被动态（NDVI 基础） |

`in_chans: 4`，每 GeoTIFF 48 通道（4 × 12 月）。

### 升级路径：6 波段

B11/B12（SWIR，20m）存储在 `20_sentinel2_l2a_monthly`。
额外下载后设 `n_bands_per_timestep: 12`，`in_chans: 6`，
可获得土壤湿度、洪水、火烧迹地等强信号。

---

## 权重复用策略

### 问题

V-JEPA 2.1 的 patch embedding 层原为 3 通道（RGB 视频）。S2 有 4/6 个波段，
直接载入会报形状不匹配。

### 解法：Prithvi-style 权重初始化

```python
w3    = pretrained["patch_embed.proj.weight"]  # [D, 3, t, p, p]
w_avg = w3.mean(dim=1, keepdim=True)           # [D, 1, t, p, p]
wN    = w_avg.repeat(1, N, 1, 1, 1)           # [D, N, t, p, p]
new_patch_embed.proj.weight = nn.Parameter(wN)
```

RGB 权重取平均后复制到 N 个通道。Backbone 其余所有层权重完全复用。

实现文件：[data_pipeline/patch_embed_6ch.py](data_pipeline/patch_embed_6ch.py)
入口函数：`build_nch_patch_embed_from_pretrained(in_chans=N, ...)`

### 训练阶段策略

1. **Stage 1（20 epochs）**：冻结 ViT backbone，只训练 `patch_embed` + `doy_encoding`
2. **Stage 2（30 epochs）**：冻结 backbone，解冻后端 6 个 transformer block
3. **Stage 3（50 epochs）**：全网络 fine-tune（低学习率，lr=1e-5）

---

## OLMo-Earth 数据集

[allenai/olmoearth_pretrain_dataset](https://huggingface.co/datasets/allenai/olmoearth_pretrain_dataset)：
285,288 个全球样本，每个样本含 12 个月度 Sentinel-2 合成影像，全球覆盖，已完成去云处理。

### 子文件夹与波段

| 子文件夹 | 分辨率 | 波段 | channels/GeoTIFF |
|---------|-------|------|-----------------|
| `10_sentinel2_l2a_monthly` | 10 m | B02 B03 B04 B08 | 4 × 12 = **48** |
| `20_sentinel2_l2a_monthly` | 20 m（重采样） | B05 B06 B07 B8A B11 B12 | 6 × 12 = **72** |

> 当前已下载：`10_sentinel2_l2a_monthly`（10 个 TAR）= 4 波段模式。
> B11/B12 需额外下载 `20_sentinel2_l2a_monthly`。

### 数据坑

| # | 坑 | 处理位置 |
|---|----|---------| 
| 1 | **通道数陷阱** — `10_sentinel2_l2a_monthly` = 48ch（4×12），不是 144ch；若 `n_bands_per_timestep` 设错会静默跳过所有样本 | `_process()` 中的 `expected_ch` 检查；先跑 `inspect_sample()` 验证 |
| 2 | **MISSING 像素（-99999）** — OLMo-Earth 用 -99999.0 标记缺失（云、边缘）；不处理会污染归一化 | `max_missing_frac` 过滤（>10% → 跳过），剩余置 0 后 clip 到 [0,1] |
| 3 | **IterableDataset 无 `__len__`** — DataLoader 的 `len()` 抛 `TypeError`；调度器需要 `ipe`（steps/epoch） | `main()` try/except + `data.ipe` 配置项（YAML 已设 `ipe: 17830`） |
| 4 | **webdataset 分片不均** — TAR 文件数 < num_workers 时部分 worker 空转 | 确保 TAR 数 ≥ num_workers（10 TAR + 4 workers = OK） |

### 检查 TAR 内容

下载后先验证通道数和值域再训练：

```python
from data_pipeline.olmoearth_dataset import inspect_sample
inspect_sample("/your_data/olmoearth/10_sentinel2_l2a_monthly/*.tar")
# 输出：channels=48，4 bands/month，值域→确认 dn_scale
```

### Per-band 归一化统计（反射率空间）

| 波段 | mean   | std    |
|------|--------|--------|
| B02  | 0.0850 | 0.0574 |
| B03  | 0.0950 | 0.0521 |
| B04  | 0.1001 | 0.0660 |
| B08  | 0.2841 | 0.1076 |
| B11  | 0.2260 | 0.1102 |
| B12  | 0.1546 | 0.0900 |

> 当前为文献近似值，建议在实际数据上重新计算。

---

## 使用方法

```bash
pip install -r data_pipeline/requirements.txt
```

### 1. 检查 TAR 内容（首次必做）

```bash
python -c "
from data_pipeline.olmoearth_dataset import inspect_sample
inspect_sample('/your_data/olmoearth/10_sentinel2_l2a_monthly/*.tar')
"
```

### 2. 编辑训练配置

```yaml
# vjepa2/configs/finetune/vitl16/olmoearth-256px-12f.yaml
olmoearth:
  tar_path: "/your_data/olmoearth/10_sentinel2_l2a_monthly/*.tar"
pretrained_checkpoint: "/your_checkpoints/vjepa2_vitl.pth"
```

### 3. 启动训练

```bash
python finetune_main.py --config vjepa2/configs/finetune/vitl16/olmoearth-256px-12f.yaml
```

调试先改 yaml `stage1.epochs: 1`，跑 50 步验证 loss 下降后再提交完整任务。

---

## 文件索引

| 文件 | 用途 |
|------|------|
| [data_pipeline/olmoearth_dataset.py](data_pipeline/olmoearth_dataset.py) | OLMo-Earth webdataset 读取；本地 TAR；4/6 波段可配置；`inspect_sample()` |
| [data_pipeline/patch_embed_6ch.py](data_pipeline/patch_embed_6ch.py) | N-ch PatchEmbed3D + Prithvi-style 权重初始化 |
| [data_pipeline/requirements.txt](data_pipeline/requirements.txt) | 依赖列表 |
| [finetune_main.py](finetune_main.py) | 训练入口（3阶段冻结/解冻、EMA、JEPA 损失） |
| [vjepa2/configs/finetune/vitl16/olmoearth-256px-12f.yaml](vjepa2/configs/finetune/vitl16/olmoearth-256px-12f.yaml) | 训练配置（256px，4波段，12帧） |
| [doc/sentinel2_pipeline/](doc/sentinel2_pipeline/) | 旧版 Sentinel-2 自采 pipeline（归档，不维护） |

---

## 算力估算

### 假设条件

| 参数 | 值 | 说明 |
|------|----|------|
| 模型 | ViT-L/16 | ~307M 参数 |
| 输入 | [B, 4, 12, 256, 256] | 12帧月度合成，256px，4波段 |
| 空间 token 数/帧 | 256 = (256/16)² | |
| 总 token 数/样本 | 3,072 = 256 × 12 | |
| 数据集规模 | 285,288 样本 | OLMo-Earth 全量 |
| 批大小 | 16 | 256px token 数较少，可用较大 batch |
| 每 epoch 步数 | ≈ 17,830 | 285,288 / 16 |
| 训练总轮次 | 100 epochs | Stage1×20 + Stage2×30 + Stage3×50 |

### 各阶段耗时估算（单卡 A100 80GB）

| 阶段 | epochs | 步数 | 估计步时 | 小计 |
|------|--------|------|---------|------|
| Stage 1 | 20 | ~356,600 | ~0.25 s/step | ~25 h |
| Stage 2 | 30 | ~534,900 | ~0.50 s/step | ~74 h |
| Stage 3 | 50 | ~891,500 | ~0.70 s/step | ~173 h |
| **合计** | **100** | **~1,783,000** | | **~272 h** |

> 步数多因为数据集大（285K vs 50K）。多卡线性加速显著：4×A100 ≈ 68 h。

### 多卡估算

| 配置 | 等效批大小 | 估计总时长 |
|------|----------|----------|
| 1 × A100 80GB | 16 | ~272 h |
| 2 × A100 80GB | 32 | ~140 h |
| 4 × A100 80GB | 64 | ~70 h |

### 预训练权重下载

```bash
huggingface-cli download facebook/vjepa2 vjepa2_vitl16.pth --local-dir ./pretrained
# 若受限改用镜像：export HF_ENDPOINT=https://hf-mirror.com
```

---

## 待办

- [ ] 运行 `inspect_sample()` 确认 TAR 通道数（48ch）和值域（dn_scale=10000.0）
- [ ] 填写 yaml 中的 `tar_path` 和 `pretrained_checkpoint` 实际路径
- [ ] 用 1×A100 先跑 Stage 1 前 50 步，验证 loss 下降正常
- [ ] 确认 per-band 归一化统计值（当前为文献近似值）
- [ ] 确定下游验证任务（变化检测 / 土地分类）和评估指标
- [ ] 若需 6 波段：下载 `20_sentinel2_l2a_monthly`，更新 YAML `n_bands_per_timestep: 12`，`in_chans: 6`
