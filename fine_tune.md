# V-JEPA 2.1 × Sentinel-2 Fine-Tuning Notes

## 目标

用 V-JEPA 2.1 预训练权重在 Sentinel-2 L2A 多光谱时序卫星影像上做 fine-tune，
生成地表时序 embedding，用于下游任务（变化检测、土地分类、物候监测等）。

---

## 核心挑战

### 挑战1：时序稀疏（5天重访周期）

V-JEPA 原设计针对连续视频（~24fps），5天间隔的卫星影像之间语义跳变大，
原始帧序号索引无法编码真实时间间隔。

**解法**：
- 用绝对时间戳（DOY，Day of Year）替代帧序号做时间位置编码
- 保留原始 temporal PE（编码帧顺序），叠加 DOY sinusoidal 编码（编码季节信息）
- 训练时随机采样不等间隔时间窗口，强迫模型泛化时间跨度

### 挑战2：云层遮挡（有效信息稀疏）

云覆盖下光学波段完全失效，部分地区年均有效观测不足10次。

**解法**：
- 用 SCL（Scene Classification Layer）做逐像素云掩码，
  类别 3/8/9/10（云影/中云/高云/卷云）判为无效
- patch 级别过滤：单 patch 云覆盖率 > 30% 的帧直接丢弃
- 长期方向：引入 Sentinel-1 SAR 作为云下锚点（SAR 穿云）

**开放问题（需对比实验）**：见下方"云层对 Embedding 的影响：对比实验设计"

---

## 云层对 Embedding 的影响：对比实验设计

### 核心问题

云层过滤会减少可用训练数据，但保留云层帧会引入噪声。
**最优目标：即便输入帧含有云层，视频 embedding 也不受影响**——这样可以最大化数据利用率，不丢弃任何帧。

这是一个需要实验验证的开放问题，不能先验假设。

### 假说

V-JEPA 的 masked prediction 机制天然具有对局部遮挡的鲁棒性——
云层在空间上是一种"自然掩码"，模型可能学会从周边上下文及时序信息推断云下状态，
从而产生对云层鲁棒的 embedding。

### 实验组设计

| 实验组 | 训练数据 | 推理数据 | 目的 |
|--------|---------|---------|------|
| A（基线） | 云覆盖 < 10% | 云覆盖 < 10% | 无云基线 |
| B | 云覆盖 < 30% | 云覆盖 < 30% | 当前 pipeline 默认 |
| C | 全部帧（含云） | 全部帧 | 最大数据量 |
| D | 全部帧（含云） | 云覆盖 < 10% | 训练鲁棒、推理干净 |
| E | 云覆盖 < 30% | 全部帧 | 测试泛化性上界 |

### 评估指标

- **主指标**：下游任务（变化检测 / 土地分类）在各实验组的 F1 / mIoU
- **辅助指标**：含云帧与对应无云帧的 embedding cosine 相似度（应接近 1.0 表示鲁棒）
- **数据量指标**：各实验组可用训练序列数（体现数据利用率收益）

### 在代码中的体现

`sentinel2_dataset.py` 的 `Sentinel2Dataset` 支持通过 `max_cloud_frac` 控制云覆盖阈值；
`download_s2.py` 保存每帧的 SCL 云覆盖率到 `index.csv`，
以便训练时按实验组动态过滤，无需重新下载数据。

```python
# index.csv 列: path, doy, patch_key, cloud_frac
# 按实验组过滤示例:
df_A = df[df["cloud_frac"] < 0.10]   # 实验组 A
df_C = df                              # 实验组 C（全量）
```

> **结论预期**：若实验组 D ≈ 实验组 A，则说明在含云数据上训练可以显著提升数据利用率，
> 且不损失 embedding 质量——这将是本工作的核心贡献之一。

---

## 波段选择

### Sentinel-2 推荐 6 波段

| 波段 | 名称 | 中心波长 | 分辨率 | 作用 |
|------|------|---------|-------|------|
| B02 | Blue  | 490nm  | 10m | RGB 可见光 |
| B03 | Green | 560nm  | 10m | RGB 可见光 |
| B04 | Red   | 665nm  | 10m | RGB 可见光 |
| B08 | NIR   | 842nm  | 10m | 植被动态（NDVI 基础） |
| B11 | SWIR1 | 1610nm | 20m | 土壤湿度、洪水、火烧迹地 |
| B12 | SWIR2 | 2190nm | 20m | 矿物、雪/冰识别 |

与 Prithvi（IBM/NASA）采用的波段组合一致，经过充分验证。
B11/B12 原始分辨率为 20m，下载时双线性上采样至 10m。

### 为什么不只用 RGB

- NIR（B08）是植被时序变化最敏感的波段
- SWIR（B11/B12）是变化检测最强信号来源（土壤湿度、烧毁、积雪融化）
- Red Edge（B05/06/07）对作物物候期有独特响应（本期暂不纳入）
- 只用 RGB 严重欠拟合时序变化信息

### Sentinel-1 SAR（后期引入）

- 优势：全天候（穿云）、信息与光学互补
- 代价：跨模态对齐非平凡，需处理 speckle 噪声
- 策略：第一阶段先做 S2 单模态，验证 pipeline 后再引入 S1 做跨模态 JEPA

---

## 权重复用策略

### 问题

V-JEPA 2.1 的 patch embedding 层：

```python
# vjepa2/app/vjepa_2_1/models/utils/patch_embed.py
class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=16, tubelet_size=2, in_chans=3, embed_dim=768):
        self.proj = nn.Conv3d(
            in_channels=in_chans,          # ← 硬编码 3 通道
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            ...
        )
```

S2 有 6 个波段，直接载入预训练权重会报形状不匹配错误。

### 解法：Prithvi-style 权重初始化

```python
w3    = pretrained["encoder.patch_embed.proj.weight"]  # [D, 3, t, p, p]
w_avg = w3.mean(dim=1, keepdim=True)                   # [D, 1, t, p, p]
w6    = w_avg.repeat(1, 6, 1, 1, 1)                    # [D, 6, t, p, p]
new_patch_embed.proj.weight = nn.Parameter(w6)
```

- RGB 权重取平均后复制到 6 个通道
- Backbone 其余所有层权重完全复用，无需修改
- 比随机初始化显著更好（低频空间特征在 step 0 就已有效）

实现文件：[data_pipeline/patch_embed_6ch.py](data_pipeline/patch_embed_6ch.py)

### DOY 时间编码注入

保留原始 temporal PE（预训练权重），叠加 DOY sinusoidal 编码：

```python
# DOY → 4维向量
doy_enc = [sin(2π·d/365), cos(2π·d/365), sin(4π·d/365), cos(4π·d/365)]

# 注入：在 patch embedding 之后，ViT blocks 之前
x = x + temporal_pe[frame_idx]   # 复用预训练权重（帧顺序）
x = x + doy_proj(doy_enc)        # 新增小Linear层（季节信息）
```

DOY 编码模块：[data_pipeline/patch_embed_6ch.py](data_pipeline/patch_embed_6ch.py) → `DOYEncoding`

### 训练阶段策略

1. **阶段1**：冻结 ViT backbone，只训练 patch_embed + doy_proj
2. **阶段2**：解冻后端 4-6 个 transformer block
3. **阶段3**：全网络 fine-tune（低学习率）

---

## 数据 Pipeline

### 数据来源

Microsoft Planetary Computer（免费，STAC API，COG 格式）
- 集合：`sentinel-2-l2a`
- 认证：`planetary_computer.sign_inplace()` 自动处理 token 续签

### 数据坑清单

| # | 坑 | 处理位置 |
|---|----|---------| 
| 1 | **波段分辨率不一致** — B11/B12 是 20m | `scipy.zoom` 双线性上采样至 10m |
| 2 | **瓦片边缘 no-data** — S2 tile 边缘 ~100px 为 0 | `border_px=112` 跳过边缘区域 |
| 3 | **DN 值域** — L2A 值为 0-10000 | `/ 10000` → [0,1]，再 per-band normalize |
| 4 | **云掩码** — 必须用 SCL，不能依赖目视 | SCL 类别 3/8/9/10 → 云，超阈值帧丢弃 |
| 5 | **时序帧数不足** — 部分 patch 全年有效帧 < 4 | `min_valid_frames` 过滤，不足整体丢弃 |
| 6 | **文件太大** — 单景 S2 tile ~800MB | COG windowed read，只下目标 patch 的空间窗口 |
| 7 | **Token 过期** — Planetary Computer token 有 TTL | `tenacity` retry + `sign_inplace` 自动续签 |
| 8 | **Patch embed 通道不匹配** — V-JEPA 是 3 通道 | 见上方 Prithvi-style 权重初始化 |

### 输出格式

```
s2_data/
├── patches/
│   └── <tile>_r0001_c0003/
│       ├── 2022-03-15.npy   # float32 [224, 224, 6]
│       ├── 2022-03-20.npy
│       └── ...
├── index.csv                # 每帧一行：path, doy, patch_key
└── sequences.csv            # 每个 patch 一行：frame_paths, doys, n_frames
```

### Per-band 归一化统计（S2 L2A 反射率空间）

| 波段 | mean   | std    |
|------|--------|--------|
| B02  | 0.0850 | 0.0574 |
| B03  | 0.0950 | 0.0521 |
| B04  | 0.1001 | 0.0660 |
| B08  | 0.2841 | 0.1076 |
| B11  | 0.2260 | 0.1102 |
| B12  | 0.1546 | 0.0900 |

---

## 文件索引

| 文件 | 用途 |
|------|------|
| [data_pipeline/download_s2.py](data_pipeline/download_s2.py) | Sentinel-2 patch 下载脚本 |
| [data_pipeline/sentinel2_dataset.py](data_pipeline/sentinel2_dataset.py) | PyTorch Dataset，兼容 V-JEPA collator |
| [data_pipeline/patch_embed_6ch.py](data_pipeline/patch_embed_6ch.py) | 6ch PatchEmbed3D + DOY 编码模块 |
| [data_pipeline/requirements.txt](data_pipeline/requirements.txt) | 依赖列表 |
| [vjepa2/app/vjepa_2_1/models/utils/patch_embed.py](vjepa2/app/vjepa_2_1/models/utils/patch_embed.py) | 原始 PatchEmbed3D（3ch） |
| [vjepa2/src/datasets/video_dataset.py](vjepa2/src/datasets/video_dataset.py) | 原始 VideoDataset（decord 读取视频文件） |

---

## 源代码 License 说明

grep 检查结果：vjepa2 仓库中共有 **21 个 Python 文件不含 "MIT" 字样**，分为三类：

### Group A：Apache 2.0（通过根目录 APACHE-LICENSE 引用）— 16 个文件

头部写的是 `"licensed under the license found in the LICENSE file"`，
指向根目录的 `APACHE-LICENSE`（Apache 2.0），**有许可证，但不是 MIT**。

```
app/vjepa_2_1/models/predictor.py
app/vjepa_2_1/models/utils/masks_dist.py
app/vjepa_2_1/models/utils/modules.py
app/vjepa_2_1/models/utils/patch_embed.py
app/vjepa_2_1/models/utils/pos_embs.py
app/vjepa_2_1/models/vision_transformer.py
app/vjepa_2_1/train.py  app/vjepa_2_1/transforms.py
app/vjepa_2_1/utils.py  app/vjepa_2_1/wrappers.py
app/vjepa_droid/droid.py  app/vjepa_droid/train.py
app/vjepa_droid/transforms.py  app/vjepa_droid/utils.py
notebooks/utils/mpc_utils.py  notebooks/utils/world_model_wrapper.py
```

这是用户记得的"没有 MIT license 的文件"。这些文件是 **V-JEPA 2.1 的核心训练代码**，
删除会导致训练流程完全不可用，不应删除。

### Group B：Apache 2.0（显式声明，第三方来源）— ~~3 个文件~~ **已删除**

```
src/datasets/utils/video/randaugment.py   # 来自 Ross Wightman (timm) — 已删除
src/datasets/utils/video/randerase.py    # 来自 Ross Wightman (timm) — 已删除
src/datasets/utils/worker_init_fn.py     # 来自 Lightning AI         — 已删除
```

8 个引用处已同步处理：
- `randerase.RandomErasing` → 各文件内联 no-op stub（`reprob=0` 时行为等价）
- `randaugment.rand_augment_transform` → `create_random_augment()` 中替换为 `raise NotImplementedError`（`auto_augment=False` 时不触发）
- `worker_init_fn.pl_worker_init_function` → `None`（`DataLoader` 接受 `worker_init_fn=None`）

### Group C：无任何 license 头 — 2 个文件

```
evals/hub/__init__.py    # 完全空文件（0 字节）
src/hub/__init__.py      # 完全空文件（0 字节）
```

这两个文件是纯 Python 模块标记文件（空 `__init__.py`），无实质内容。
删除它们会破坏 `from evals.hub import ...` 和 `from src.hub import ...` 的导入链。
**保留，不删除**；若需补充 license 头可加一行注释，但不影响功能。

### 结论

| 分类 | 文件数 | License | 操作 |
|------|--------|---------|------|
| MIT 显式声明 | 其余文件 | MIT | 保留 |
| Apache 2.0（根目录引用） | 16 | Apache 2.0 | 保留，核心代码 |
| Apache 2.0（显式，第三方） | 3 | Apache 2.0 | **已删除**，引用处替换为 stub |
| 无 license 头（空文件） | 2 | 无 | 保留，删除会破坏 import |

---

## 待办

**Pipeline**
- [ ] 验证 download_s2.py 在小 bbox 上跑通，确认 index.csv 含 cloud_frac 列
- [ ] 确认 per-band 归一化统计值（当前为文献近似值，建议在实际下载数据上重新计算）
- [ ] 将 DOYEncoding 注入 V-JEPA ViT forward（修改 vision_transformer.py）
- [ ] 编写 fine-tune 训练配置 yaml

**云层对比实验**
- [ ] 按实验组 A/B/C/D/E 各准备数据子集（过滤 index.csv，无需重新下载）
- [ ] 训练各实验组，记录下游任务指标（F1 / mIoU）
- [ ] 计算含云帧与对应无云帧的 embedding cosine 相似度
- [ ] 根据实验结论决定最终 pipeline 的云过滤策略

**长期**
- [ ] 确定下游验证任务（变化检测 / 土地分类）和评估指标
- [ ] 阶段2：引入 Sentinel-1 SAR 跨模态融合
