# TIP-Net：时序不一致性与原型对比网络

基于弱监督学习的部分伪造视频检测与帧级定位，只需视频级标签，无需帧级标注。

---

## 目录

- [项目结构](#项目结构)
- [数据组织方式](#数据组织方式)
- [数据读取流程详解](#数据读取流程详解)
- [快速开始](#快速开始)
- [训练参数说明](#训练参数说明)
- [测试与评估](#测试与评估)
- [模型架构说明](#模型架构说明)
- [显存优化建议](#显存优化建议)
- [消融实验参数](#消融实验参数)
- [常见问题](#常见问题)

---

## 项目结构

```
tipnet/
│
├── configs/
│   └── default.yaml              # 所有超参数的配置文件
│
├── data/
│   ├── __init__.py               # 留空即可
│   ├── dataset.py                # 数据集类，核心数据读取逻辑
│   └── transforms.py             # 训练/验证图像变换
│
├── model/
│   ├── __init__.py               # 留空即可
│   ├── backbone.py               # 骨干网络注册表（Xception/ViT/DINO/EfficientNet）
│   └── tipnet.py                 # TIP-Net 完整模型架构
│
├── losses/
│   ├── __init__.py               # 留空即可
│   └── losses.py                 # 所有损失函数（Lsim/Lproto/Lloc/ECL）
│
├── utils/
│   ├── __init__.py               # 留空即可
│   └── train_utils.py            # 指标计算、学习率调度、Checkpoint工具
│
├── scripts/
│   └── preprocess.py             # 视频抽帧预处理脚本（独立使用）
│
├── train.py                      # 训练入口
├── test.py                       # 测试/评估入口
├── requirements.txt              # 依赖列表
└── README.md                     # 本文件
```

---

## 数据组织方式

### 第一步：原始视频目录结构

将原始视频按真实/伪造分成两个文件夹：

```
raw_dataset/
├── real/
│   ├── id00001.mp4
│   ├── id00002.mp4
│   └── ...
└── fake/
    ├── id00001.mp4
    ├── id00002.mp4
    └── ...
```

支持格式：`.mp4` `.avi` `.mov` `.mkv` `.webm` `.flv` `.wmv`

---

### 第二步：预处理抽帧

运行 `scripts/preprocess.py`，将视频转换为帧文件夹：

```bash
python scripts/preprocess.py \
    --data_root  ./raw_dataset \
    --frames_dir ./frames \
    --max_frames 128 \
    --size       224 \
    --face_crop \
    --num_workers 8
```

**输出目录结构（训练时读取这个目录）：**

```
frames/
├── real/
│   ├── id00001/          ← 一个视频对应一个子文件夹
│   │   ├── 00000.jpg
│   │   ├── 00001.jpg
│   │   └── ...（按数字顺序）
│   ├── id00002/
│   └── ...
└── fake/
    ├── id00001/
    │   ├── 00000.jpg
    │   └── ...
    └── ...
```

**核心规则：**
- 每个视频 → 一个子文件夹，文件夹名即视频ID
- 帧文件名按数字顺序排列（`00000.jpg`, `00001.jpg`, ...）
- 只支持 `.jpg` `.jpeg` `.png` 格式
- 帧数少于 `min_frames`（默认8帧）的视频自动跳过

**抽帧参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max_frames` | 无 | 每个视频均匀采样N帧，推荐 64~256 |
| `--stride` | 4 | 不设 max_frames 时，每隔N帧取一帧 |
| `--size` | 224 | 输出帧的分辨率，Xception骨干用299 |
| `--face_crop` | 关 | 启用人脸检测裁剪（需安装 facenet-pytorch） |
| `--face_margin` | 0.3 | 人脸框外扩比例 |
| `--quality` | 90 | JPEG压缩质量 |
| `--num_workers` | 4 | 并行处理进程数 |
| `--overwrite` | 关 | 重新提取已处理的视频 |

---

## 数据读取流程详解

### 完整流程图

```
磁盘文件
frames/real/id00001/00000.jpg
frames/real/id00001/00001.jpg
...（共 N 帧）
frames/fake/id00002/00000.jpg
...
        │
        ▼  Dataset.__init__()
        扫描所有子文件夹，记录每个视频的帧路径列表
        按 split_ratio=0.8 随机划分 train/val
        │
        ▼  Dataset.__getitem__(idx)
        ┌─────────────────────────────────────────────────────┐
        │                                                     │
        │  视频有 N 帧，划分为 T=16 个时序窗口（Snippet）        │
        │                                                     │
        │  窗口大小 = N / T                                    │
        │  窗口 t: [t*window_size, (t+1)*window_size)         │
        │                                                     │
        │  每个窗口内用采样策略选 L=8 帧：                       │
        │    uniform → 窗口内均匀取L帧（默认，保留时序）           │
        │    random  → 窗口内随机取L帧（训练数据增强）            │
        │    dense   → 随机起点连续取L帧（捕捉局部运动）           │
        │                                                     │
        │  加载帧 → PIL.Image → transform → Tensor[C,H,W]    │
        │  L帧叠加 → Tensor[L,C,H,W]                          │
        │  T个Snippet叠加 → Tensor[T,L,C,H,W]                │
        └─────────────────────────────────────────────────────┘
        │
        ▼  collate_fn（DataLoader自动调用）
        B个视频拼成一个batch：
        snippets:   [B, T, L, C, H, W]   ← 主输入
        labels:     [B]                   ← 视频级标签（0/1）
        snippet_gt: [B, T]                ← 片段级标签（训练时=视频标签）
```

### 具体数字举例

假设视频有 **320 帧**，`T=16`，`L=8`：

```
窗口大小 = 320 / 16 = 20 帧/窗口

Snippet  0 → 帧 [  0~19]  → 均匀取8帧 → 第0,3,6,9,12,15,17,19帧
Snippet  1 → 帧 [ 20~39]  → 均匀取8帧
Snippet  2 → 帧 [ 40~59]  → 均匀取8帧
...
Snippet 15 → 帧 [300~319] → 均匀取8帧

最终输出 tensor 形状：[16, 8, 3, 224, 224]
                       T   L  C   H    W
```

### 片段标签（snippet_gt）的生成逻辑

```
情况1：未提供帧级标注文件（弱监督训练标准模式）
    real 视频 → snippet_gt = [0, 0, 0, ..., 0]（T个0）
    fake 视频 → snippet_gt = [1, 1, 1, ..., 1]（T个1）
    这是弱监督的核心假设：视频级标签传播到所有片段

情况2：提供了帧级标注文件（评估定位性能时使用）
    读取 annotation.json 中对应视频的帧级标签列表
    每个片段窗口内对帧标签做多数投票：
        窗口内 >50% 的帧是伪造 → 该片段标签 = 1
        否则 → 该片段标签 = 0
    用于计算 AP@0.5、mAP 等定位指标
```

### 标注文件格式（可选，仅评估时需要）

```json
{
    "id00001": {
        "frames": [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
    },
    "id00002": [0, 0, 0, 0, 0, 0]
}
```

两种格式均支持：带 `"frames"` 键的字典，或直接的标签列表。

---

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt

# 可选：人脸检测（预处理时使用）
pip install facenet-pytorch

# 可选：Xception ImageNet预训练权重
pip install pretrainedmodels
```

### 完整三步流程

```bash
# 第一步：视频抽帧
python scripts/preprocess.py \
    --data_root  ./raw_dataset \
    --frames_dir ./frames \
    --max_frames 128 \
    --size 224 \
    --face_crop \
    --num_workers 8

# 第二步：训练
python train.py \
    --frames_dir ./frames \
    --backbone   xception \
    --epochs     100 \
    --batch_size 8 \
    --save_dir   ./checkpoints \
    --log_dir    ./logs

# 第三步：测试
python test.py \
    --frames_dir ./frames \
    --checkpoint ./checkpoints/checkpoint_ep100_best.pth \
    --backbone   xception \
    --save_dir   ./results \
    --visualize
```

### 从断点恢复训练

```bash
python train.py \
    --frames_dir ./frames \
    --resume     ./checkpoints/checkpoint_ep050.pth \
    --epochs     100
```

---

## 训练参数说明

### 数据参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--frames_dir` | 必填 | 抽帧后根目录（含 real/ 和 fake/ 子文件夹） |
| `--num_snippets` | 16 | 每个视频划分的片段数 T |
| `--frames_per_snippet` | 8 | 每个片段采样帧数 L（消融实验目标） |
| `--image_size` | 224 | 帧分辨率，Xception填299 |
| `--split_ratio` | 0.8 | 训练集比例 |
| `--max_videos` | 无 | 限制视频数（调试用） |

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--backbone` | xception | 骨干网络，见支持列表 |
| `--snippet_dim` | 512 | Snippet特征维度 D |
| `--proj_dim` | 128 | 对比投影维度 D' |
| `--num_prototypes` | 8 | 真实原型数量 K（消融实验目标） |
| `--tmm_order` | 2 | 时序差分阶数 k（消融实验目标） |
| `--lstm_layers` | 2 | Bi-LSTM层数 |
| `--attention_heads` | 4 | 多头注意力头数 |

### 损失权重

| 参数 | 默认值 | 对应损失 | 设为0时效果 |
|------|--------|----------|------------|
| `--gamma1` | 0.5 | Lsim 相似度一致性 | 关闭TIM辅助监督 |
| `--gamma2` | 1.0 | Lproto 原型对比 | 关闭PCL原型学习 |
| `--gamma3` | 0.8 | Lloc 自举定位 | 关闭片段伪标签 |
| `--gamma4` | 0.3 | Lecl 重加权对比 | 关闭ECL损失 |
| `--top_k` | 5 | 伪标签选取的Top-K片段数 | — |

### 训练过程参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 100 | 训练轮数 |
| `--batch_size` | 8 | 每批视频数 |
| `--lr` | 1e-4 | 初始学习率 |
| `--min_lr` | 1e-6 | 余弦退火最小学习率 |
| `--warmup_epochs` | 5 | 线性预热轮数 |
| `--weight_decay` | 1e-4 | AdamW权重衰减 |
| `--accumulate_grad` | 2 | 梯度累积步数 |
| `--update_proto_freq` | 5 | 每N轮K-Means更新原型 |

### 支持的骨干网络

```bash
# Xception（论文默认，输入299×299）
--backbone xception

# EfficientNet系列（通过timm）
--backbone efficientnet_b4

# Vision Transformer（通过timm）
--backbone vit_base_patch16_224
--backbone vit_large_patch16_224

# DINO自监督ViT（通过torch.hub）
--backbone dino_vit_small
--backbone dino_vit_base
```

---

## 测试与评估

```bash
# 基本测试
python test.py \
    --frames_dir         ./frames \
    --checkpoint         ./checkpoints/checkpoint_ep100_best.pth \
    --backbone           xception \
    --num_snippets       16 \
    --frames_per_snippet 8 \
    --batch_size         4 \
    --threshold          0.5 \
    --save_dir           ./results \
    --visualize \
    --num_vis_samples    20

# 带帧级标注的定位评估
python test.py \
    --frames_dir      ./frames \
    --checkpoint      ./checkpoints/checkpoint_ep100_best.pth \
    --annotation_file ./annotations.json \
    --save_dir        ./results
```

### 输出内容

**终端输出：**
```
视频级指标:
  ACC         : 0.9234
  AUC         : 0.9567
  PRECISION   : 0.9123
  RECALL      : 0.9345
  F1          : 0.9233

定位指标:
  loc_prec@0.5  : 0.8734
  loc_rec@0.5   : 0.8956
  loc_f1@0.5    : 0.8844
  loc_mAP       : 0.8512
```

**保存的文件：**
```
results/
├── test_results.json       # 完整结果
└── visualizations/
    ├── id00001_Fake.png    # 时序异常分数图（含 At/mt/dt 三条曲线）
    └── id00002_Real.png
```

**JSON结构：**
```json
{
  "video_metrics":  {"acc": 0.923, "auc": 0.956, ...},
  "localisation":   {"loc_f1@0.5": 0.884, "loc_mAP": 0.851},
  "per_video_results": [
    {
      "video_id": "id00001",
      "true_label": 1,
      "pred_label": 1,
      "prob_fake": 0.934,
      "anomaly_scores": [0.12, 0.89, 0.92, ...],
      "temporal_mt":    [0.07, 0.83, 0.78, ...],
      "spatial_dt":     [0.12, 0.71, 0.68, ...]
    }
  ]
}
```

---

## 模型架构说明

### 数据维度流转

```
输入:  snippets [B, T, L, C, H, W]
       B=批大小  T=片段数(16)  L=帧数(8)  C=3  H=W=224

Step1  分块骨干编码（每次最多 frame_chunk=64 帧，防OOM）
       [B×T×L, C, H, W] → backbone → [B×T×L, backbone_dim]

Step2  帧平均池化 → 片段特征
       reshape → mean(dim=L) → [B, T, backbone_dim]

Step3  片段投影 + 时序精化
       Linear+LN+ReLU → [B, T, 512]
       1D Conv残差块  → [B, T, 512]  ← 抑制内容波动，保留时序语义

Step4  三路并行分支
       ┌── TIM（时序不一致性挖掘）
       │   [B,T,512] → 相邻余弦相似度 → [B,T-1]
       │            → k阶差分+归一化  → mt [B,T]
       │
       ├── PCL（原型对比学习）
       │   [B,T,512] → 投影头(→128) + L2归一 → zt [B,T,128]
       │            → 与K个真实原型计算相似度
       │            → dt = 1 - max_sim          → [B,T]
       │
       └── MSA-L（多源注意力定位）
           [B,T,512] + mt + dt
           → Bi-LSTM → ht [B,T,512]
           → 多头自注意力（残差）→ ht [B,T,512]
           → et = tanh(Wh·ht + Wm·mt + Wd·dt)
           → softmax → αt [B,T]
           → At = 0.5·αt + 0.5·mt  [B,T]  ← 融合异常分

Step5  片段预测头
       ht → Linear(512→256→1)+Sigmoid → snippet_scores [B,T]

Step6  MIL视频级聚合
       vagg = Σ(αt·ht) → [B,512]
       → Linear(512→256→2) → video_logits [B,2]
```

### 片段分数 → 帧级分数映射

```python
# 每个片段的分数广播到对应帧窗口
window_size = num_frames / T
for t in range(T):
    start = int(t * window_size)
    end   = int((t+1) * window_size)
    frame_scores[start:end] = snippet_scores[t]
```

### 记忆库与原型更新流程

```
每个训练 batch：
  取 label==0（真实视频）的片段投影 zt
  过滤：只保留 mt < 0.3 的片段（低突变 = 可靠真实片段）
  FIFO写入记忆库（容量 memory_size=4096）

每隔 update_proto_freq 轮：
  对记忆库特征做 K-Means（K=num_prototypes）
  聚类中心 → 归一化 → 更新真实原型
  触发条件：记忆库至少填充 K×10 个样本
```

---

## 显存优化建议

| 问题 | 解决方案 |
|------|----------|
| 显存不足（OOM） | 减小 `--batch_size`，增大 `--accumulate_grad` |
| 骨干网络太大 | 在 `tipnet.py` 中设 `frame_chunk=32` |
| 训练速度慢 | 增大 `--num_workers`，配置中开启 `amp: true` |
| 记忆库不更新 | 检查 `real/` 文件夹是否存在且有足够视频 |
| 原型始终不更新 | 需要记忆库中有 `num_prototypes × 10` 个样本 |

**8GB显存推荐配置：**
```bash
python train.py --batch_size 4 --accumulate_grad 4 \
    --frames_per_snippet 8 --num_snippets 16
```

**24GB显存推荐配置：**
```bash
python train.py --batch_size 16 --accumulate_grad 1 \
    --frames_per_snippet 8 --num_snippets 16
```

---

## 消融实验参数

| 实验目标 | 控制参数 | 推荐取值 |
|----------|----------|----------|
| 时序差分阶数 | `--tmm_order` | 0, 1, 2, 3 |
| 每片段帧数 | `--frames_per_snippet` | 4, 6, 8, 16, 32 |
| 原型数量 | `--num_prototypes` | 4, 8, 16, 32 |
| Top-K伪标签 | `--top_k` | 3, 5, 8, 10 |
| 各损失项开关 | `--gamma1/2/3/4` | 0.0=关闭 |

**逐一去掉各模块进行消融：**
```bash
# w/o TMM（关闭时序损失）
python train.py --gamma1 0.0 ...

# w/o PCL（关闭原型对比）
python train.py --gamma2 0.0 ...

# w/o Lloc（关闭自举定位）
python train.py --gamma3 0.0 ...

# w/o ECL（关闭重加权对比）
python train.py --gamma4 0.0 ...
```

---

## 常见问题

**Q：记忆库一直是 0/4096？**
检查 `frames/real/` 文件夹是否存在且非空，以及 `frames_dir` 路径是否正确。

**Q：原型更新后 Lproto 突然变大？**
正常现象，K-Means 重聚类会短暂打乱对比空间，几个 batch 后会恢复稳定。

**Q：视频帧数少于 T×L 怎么处理？**
采样时会自动重复末帧填充，不会报错。建议预处理时设 `--min_frames 16` 过滤过短视频。

**Q：ForgeryNet / FF++ / CelebDF 数据集怎么组织？**
将真实视频帧放入 `frames/real/`，伪造视频帧放入 `frames/fake/`，每个视频一个子文件夹，文件夹名即视频ID。

**Q：能不能直接读取视频文件不抽帧？**
当前不支持。抽帧是必须的，目的是：避免训练时重复解码视频浪费时间；支持人脸裁剪预处理；保证帧文件按数字顺序排列以维持时序。