# TIP-Net

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

Official PyTorch implementation of the paper:

> **Partially Manipulated DeepFake Video Detection and Localization via Weakly Supervised Learning**  
> *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2026*

---

## 📌 Overview

TIP-Net is a weakly supervised framework for **partially manipulated DeepFake video detection and localization**, requiring only **video-level annotations**.

Key characteristics:

- 🎯 Handles **partial manipulation scenarios**
- 🧠 Prototype-based representation learning
- 🔍 Temporal inconsistency modeling for localization
- 🏷️ Robust to **noisy labels** under weak supervision

---

## ⚙️ Prerequisites

- Python >= 3.8  
- PyTorch >= 1.10  
- CUDA >= 11.3  

Install dependencies:

```bash
git clone https://github.com/yyxb0627/TIP-Net.git
cd TIP-Net
pip install -r requirements.txt
```

Optional (for face preprocessing):

```bash
pip install facenet-pytorch
```

---

## 📁 Data Preparation

Organize your dataset in the following structure:

```
raw_dataset/
├── real/
│   ├── id00001.mp4
│   └── id00002.mp4
└── fake/
    ├── id00003.mp4
    └── id00004.mp4
```

Supported datasets include:

- ForgeryNet  
- FaceForensics++ (FF++)  
- Celeb-DF  

### Frame Extraction

Run preprocessing:

```bash
python scripts/preprocess.py \
    --data_root  ./raw_dataset \
    --frames_dir ./frames \
    --face_crop
```

---

## 🚀 Quick Start

### 🔹 Training

Train TIP-Net from scratch:

```bash
python train.py \
    --frames_dir ./frames \
    --save_dir   ./checkpoints \
    --log_dir    ./logs
```

Notes:

- The model automatically:
  - builds the memory bank
  - updates real prototypes dynamically
- If you encounter **OOM (Out-of-Memory)**:
  - reduce batch size
  - or use gradient accumulation

---

### 🔹 Testing & Evaluation

Evaluate trained model:

```bash
python test.py \
    --frames_dir ./frames \
    --checkpoint ./checkpoints/best_model.pth \
    --save_dir   ./results
```

Outputs include:

- Video-level metrics  
- Snippet-level localization results  



