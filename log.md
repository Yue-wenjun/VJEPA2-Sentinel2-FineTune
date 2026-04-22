# 变更日志

## 2026-04-22

### 代码复核：修复 8 个 Bug

全面复核所有新增/修改文件后发现并修复以下问题：

| # | 文件 | 问题 | 修复 |
|---|------|------|------|
| 1 | `data_pipeline/patch_embed_6ch.py` | `weight_key` 默认值含 `encoder.` 前缀，但 `ckpt["encoder"]` 已去掉该前缀，导致权重找不到，退化为随机初始化 | 改为 `patch_embed.proj.weight` |
| 2 | `data_pipeline/sentinel2_dataset.py` | `__getitem__` 未返回 DOY 值，训练时无法获取真实日期 | 返回第 4 个元素 `doy_tensor: [T]` |
| 3 | `finetune_main.py` | DOY 用 `torch.zeros` 占位，从未注入真实值到模型 | 从 `collated_batch[3]` 提取 DOY |
| 4 | `finetune_main.py` | `MaskCollator` 初始化参数完全错误（传了不存在的参数名） | 改为正确签名：`cfgs_mask / dataset_fpcs / crop_size / patch_size / tubelet_size` |
| 5 | `finetune_main.py` | Collator 在训练循环内单独调用，而非作为 `collate_fn`，batch 结构错误 | 改为 `DataLoader(collate_fn=mask_collator, ...)` |
| 6 | `vjepa2/app/vjepa_2_1/models/vision_transformer.py` | DOY 注入使用 `H_patches * W_patches`，但 `handle_nonsquare_inputs=False` 时两者均为 `None`，引发 `TypeError` | 加 `H_patches is not None` guard |
| 7 | `vjepa2/src/models/vision_transformer.py` | 同上 | 同上 |
| 8 | `vjepa2/src/models/vision_transformer.py` | 从 `app/` 包导入 `DOYEncoding`，`src/` → `app/` 跨包依赖存在循环风险 | 将 `DOYEncoding` 移至 `src/models/utils/doy_encoding.py`，两处 `vision_transformer.py` 统一从 `src` 导入 |

---

## 2026-04-21

### 删除 Apache 2.0 授权文件

根据 README 中的 License 声明，以下三个文件为 Apache 2.0 授权（第三方来源），与项目主体 MIT 授权不同，予以删除：

- `vjepa2/src/datasets/utils/video/randaugment.py`（来自 Ross Wightman / timm）
- `vjepa2/src/datasets/utils/video/randerase.py`（来自 Ross Wightman / timm）
- `vjepa2/src/datasets/utils/worker_init_fn.py`（来自 Lightning AI）

三个文件均为数据增强/加载辅助功能，不涉及模型核心逻辑，可不使用。

**同步修改的引用文件（共 8 处）：**

| 文件 | 修改内容 |
|------|---------|
| `src/datasets/utils/video/transforms.py` | 移除 `rand_augment_transform` import；`create_random_augment` 中替换为 `raise NotImplementedError` |
| `src/datasets/utils/video/transforms_builder.py` | 移除 `RandomErasing` import；内联 no-op stub |
| `app/vjepa/transforms.py` | 同上 |
| `app/vjepa_2_1/transforms.py` | 同上 |
| `app/vjepa_droid/transforms.py` | 同上 |
| `evals/action_anticipation_frozen/dataloader.py` | 同上 |
| `evals/video_classification_frozen/utils.py` | 同上 |
| `evals/action_anticipation_frozen/epickitchens.py` | 移除 `pl_worker_init_function` import；替换为 `None` |

**说明**：`RandomErasing` 在 `__init__` 中无条件实例化，直接删文件会导致 `NameError`，因此保留 no-op stub。实际效果等同于功能不存在（`reprob=0` 时原本也是 no-op；`auto_augment=True` 时会 `raise NotImplementedError`）。

---

### 注入 DOYEncoding 到 VisionTransformer

**新增文件**：
- `vjepa2/app/vjepa_2_1/models/utils/doy_encoding.py` — `DOYEncoding` 模块

**修改文件**：

| 文件 | 修改内容 |
|------|---------|
| `app/vjepa_2_1/models/vision_transformer.py` | 新增 `use_doy_encoding` 参数；`__init__` 中创建 `self.doy_encoding`；`forward` 新增 `doys` 参数，在 `pos_embed` 之后、`apply_masks` 之前注入 |
| `src/models/vision_transformer.py` | 同上（v1 路径同步） |
| `app/vjepa_2_1/wrappers.py` | `MultiSeqWrapper.forward` 新增 `doys` 参数并透传给 `self.backbone` |

注入时序：`patch_embed(x)` → `+pos_embed` → **`+doy_encoding`** → `apply_masks` → transformer blocks

---

### 新增 Fine-Tune 训练配置和主脚本

**新增文件**：
- `vjepa2/configs/finetune/vitl16/sentinel2-224px-8f.yaml` — 训练配置，含 3 阶段优化参数和云层实验开关
- `finetune_main.py` — 训练入口，串联 Sentinel2Dataset、6ch 模型初始化、Prithvi-style 权重加载、3 阶段冻结/解冻、JEPA 损失、EMA target encoder、checkpoint 保存

---

### 新增 Sentinel-2 数据 Pipeline

新增文件：

| 文件 | 用途 |
|------|------|
| `data_pipeline/download_s2.py` | 通过 Microsoft Planetary Computer STAC API 下载 S2 L2A patch，输出 `.npy` 文件 + `index.csv` + `sequences.csv` |
| `data_pipeline/sentinel2_dataset.py` | PyTorch Dataset，兼容 V-JEPA 2.1 collator，支持 6 波段输入和 DOY 编码 |
| `data_pipeline/patch_embed_6ch.py` | 6 通道 `PatchEmbed3D`（Prithvi-style 权重初始化）+ `DOYEncoding` 模块 |
| `data_pipeline/requirements.txt` | 依赖列表 |

**关键设计决策：**
- 波段选择：B02/B03/B04/B08/B11/B12（6 波段），与 Prithvi 一致
- 权重初始化：对预训练 RGB 权重取均值后复制到 6 通道，保留 backbone 其余层
- 云掩码：基于 SCL 层，每帧记录 `cloud_frac`，支持对比实验动态过滤
- `index.csv` 包含 `cloud_frac` 列，供云层鲁棒性对比实验使用

---

### 新增文档

| 文件 | 内容 |
|------|------|
| `fine_tune.md` | Fine-tune 方案完整记录（挑战、波段选择、权重复用、数据 pipeline、云层对比实验设计、License 说明、待办） |
| `log.md` | 本文件，记录变更动作 |
