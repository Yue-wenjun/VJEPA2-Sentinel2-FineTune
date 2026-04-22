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
| [finetune_main.py](finetune_main.py) | 训练入口（3阶段冻结/解冻、EMA、JEPA 损失） |
| [vjepa2/configs/finetune/vitl16/sentinel2-224px-8f.yaml](vjepa2/configs/finetune/vitl16/sentinel2-224px-8f.yaml) | 训练配置 |
| [vjepa2/src/models/utils/doy_encoding.py](vjepa2/src/models/utils/doy_encoding.py) | DOYEncoding 模块（canonical 位置） |
| [vjepa2/app/vjepa_2_1/models/utils/patch_embed.py](vjepa2/app/vjepa_2_1/models/utils/patch_embed.py) | 原始 PatchEmbed3D（3ch） |
| [vjepa2/src/datasets/video_dataset.py](vjepa2/src/datasets/video_dataset.py) | 原始 VideoDataset（decord 读取视频文件） |

---

## 算力估算

### 假设条件

| 参数 | 值 | 说明 |
|------|----|------|
| 模型 | ViT-L/16 | ~307M 参数 |
| 输入 | [B, 6, 8, 224, 224] | 8帧，224px，6波段 |
| 数据集规模 | ~50,000 序列 | 中等区域全年覆盖 |
| 训练总轮次 | 100 epochs | Stage1×10 + Stage2×20 + Stage3×70 |
| 批大小 | 16 | 单卡 A100 80GB |
| 每 epoch 步数 | 50,000 / 16 ≈ 3,125 步 | |
| 总步数 | ~312,500 步 | |

### 各阶段耗时估算（单卡 A100 80GB）

| 阶段 | epochs | 步数 | 冻结状态 | 估计步时 | 小计 |
|------|--------|------|---------|---------|------|
| Stage 1 | 10 | ~31,250 | backbone 冻结，只训 patch_embed+doy | ~0.25 s/step | ~2.2 h |
| Stage 2 | 20 | ~62,500 | 后 6 blocks 解冻 | ~0.45 s/step | ~7.8 h |
| Stage 3 | 70 | ~218,750 | 全量 fine-tune | ~0.65 s/step | ~39.5 h |
| **合计** | **100** | **~312,500** | | | **~50 h** |

### 多卡线性加速估算

| 配置 | 等效批大小 | 估计总时长 | 备注 |
|------|----------|----------|------|
| 1 × A100 80GB | 16 | ~50 h | 单机可跑，性价比最高 |
| 2 × A100 80GB | 32 | ~27 h | DDP，线性加速约 1.85× |
| 4 × A100 80GB | 64 | ~14 h | DDP，线性加速约 3.6× |
| 8 × A100 80GB | 128 | ~8 h | 一天内完成，适合快速迭代 |

> 以上为 bfloat16 + `torch.compile` 估算值。数据加载（num_workers=4）为瓶颈时实际可能慢 10-20%。
> **建议**：先用 1×A100 跑完 Stage1 验证 pipeline（约 2 h），确认 loss 下降后再决定是否多卡。

### openi.pcl.ac.cn CloudBrains 配置建议

- 初次调试：1×A100 × 1 节点（验证数据流、loss 曲线、checkpoint 保存）
- 正式训练：1×A100 或 2×A100（50K 序列规模不需要 8 卡，浪费配额）
- 数据下载：在任务容器内直接运行 `python data_pipeline/download_s2.py` 下载 S2 数据；权重用 `huggingface_hub` 命令行下载（见下方"权重下载"节），无需手动上传

### 预训练权重下载

V-JEPA 2.1 权重托管于 HuggingFace，在服务器上运行：

```bash
pip install huggingface_hub
python - <<'EOF'
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="facebook/vjepa2",
    filename="vjepa2_vitl16.pth",   # 确认实际文件名
    local_dir="./pretrained",
)
print(path)
EOF
```

或直接用 CLI：

```bash
huggingface-cli download facebook/vjepa2 vjepa2_vitl16.pth --local-dir ./pretrained
```

> 若 openi 环境访问 HuggingFace 受限，可改用镜像：
> `export HF_ENDPOINT=https://hf-mirror.com` 后再运行上述命令。

---

## 源代码 License 说明

grep 检查结果：vjepa2 仓库中共有 **3  Python 文件不含 MIT License**

### Apache 2.0（显式声明，第三方来源）— ~~3 个文件~~ **已删除**

```
src/datasets/utils/video/randaugment.py   # 来自 Ross Wightman (timm) — 已删除
src/datasets/utils/video/randerase.py    # 来自 Ross Wightman (timm) — 已删除
src/datasets/utils/worker_init_fn.py     # 来自 Lightning AI         — 已删除
```

8 个引用处已同步处理：
- `randerase.RandomErasing` → 各文件内联 no-op stub（`reprob=0` 时行为等价）
- `randaugment.rand_augment_transform` → `create_random_augment()` 中替换为 `raise NotImplementedError`（`auto_augment=False` 时不触发）
- `worker_init_fn.pl_worker_init_function` → `None`（`DataLoader` 接受 `worker_init_fn=None`）

---

## 待办

**Pipeline**
- [ ] 验证 download_s2.py 在小 bbox 上跑通，确认 index.csv 含 cloud_frac 列
- [ ] 确认 per-band 归一化统计值（当前为文献近似值，建议在实际下载数据上重新计算）
- [x] 将 DOYEncoding 注入 V-JEPA ViT forward（`app/vjepa_2_1/models/vision_transformer.py` + `src/models/vision_transformer.py`）
- [x] 编写 fine-tune 训练配置 yaml（`vjepa2/configs/finetune/vitl16/sentinel2-224px-8f.yaml`）
- [x] 编写 finetune_main.py 训练入口（3阶段冻结/解冻、MaskCollator、EMA、JEPA 损失）

**训练前准备（Stage 1 开始前）**
- [ ] 在 openi 任务环境中运行 `pip install -r data_pipeline/requirements.txt` 安装依赖
- [ ] 在 openi 任务环境中运行 `python data_pipeline/download_s2.py` 下载 S2 数据到服务器本地（或持久化存储路径）
- [ ] 用代码下载 V-JEPA 2.1 预训练权重（见下方权重下载命令）
- [ ] 用 1×A100 先跑 Stage 1 前 50 步，验证 loss 下降正常，再开始完整训练

**云层对比实验**
- [ ] 按实验组 A/B/C/D/E 各准备数据子集（过滤 index.csv，无需重新下载）
- [ ] 训练各实验组，记录下游任务指标（F1 / mIoU）
- [ ] 计算含云帧与对应无云帧的 embedding cosine 相似度
- [ ] 根据实验结论决定最终 pipeline 的云过滤策略

**长期**
- [ ] 确定下游验证任务（变化检测 / 土地分类）和评估指标
- [ ] 阶段2：引入 Sentinel-1 SAR 跨模态融合
