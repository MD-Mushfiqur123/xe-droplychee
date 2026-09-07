# 🫐 xe_droplychee

**A High-Efficiency Bengali & Multilingual Foundation Model Architecture Inspired by DeepSeek-V4-Flash**

---

## 📌 Overview

`xe_droplychee` is a next-generation, parameter-efficient language model architecture optimized for high-throughput training on single-node workstation GPUs (such as **NVIDIA RTX PRO 6000 96GB VRAM**) and cloud clusters.

### Core Architectural Features:
- **Fine-Grained Mixture-of-Experts (MoE)**: 32 routed experts with Top-2 routing + 1 dedicated permanently active **Shared Expert** capturing universal syntax and grammar.
- **SqrtSoftplus Gating Router**: Independent scoring function $s_i = \sqrt{\text{softplus}(z_i)}$ preventing vanishing gradients and expert starvation without auxiliary loss penalties (`noaux_tc`).
- **SwiGLU Clamping (`swiglu_limit = 10.0`)**: Mitigates quadratic outlier explosions in expert MLPs for rock-solid numerical stability in reduced precision (`bfloat16`/`FP8`).
- **Manifold-Constrained Hyper-Connections (Mini-mHC)**: Residual stream mixing projected onto the Birkhoff Polytope via Sinkhorn-Knopp iterations, bounding signal explosion across deep layers.
- **Multi-Token Prediction (MTP)**: Built-in depth-1 future token prediction head for self-speculative decoding acceleration during inference.
- **Autonomous Training & Auto-Push Pipeline**: Native integration with **Marimo** reactive notebook and automated Hugging Face Hub release with safetensors validation.

---

## 🚀 Quickstart

### 1. Requirements & Setup
```bash
git clone https://github.com/MD-Mushfiqur123/xe-droplychee.git
cd xe-droplychee
pip install torch transformers datasets huggingface_hub safetensors bitsandbytes marimo
```

### 2. Day-1 Training (4-Hour Run on 96GB GPU)

#### Option A: Headless Autonomous Terminal (Recommended)
```bash
python train_and_push_autonomous.py \
    --dataset_name "nahid-hub/B-CORE-bengali-corpus" \
    --output_repo "MD-Mushfiqur123/xe-droplychee-v2-trained" \
    --total_steps 6500 \
    --batch_size 8 \
    --gradient_accumulation_steps 4
```

#### Option B: Marimo Reactive Interactive Dashboard
```bash
marimo edit marimo_train_and_push.py
```
Open the interactive UI in your browser and click **🚀 Start 4-Hour Training & Auto-Push**.

---

## 📂 Repository Layout

```
├── m_droplychee/                  # Custom model architecture package
│   ├── configuration_droplychee.py
│   ├── modeling_droplychee.py
│   └── __init__.py
├── hf_model/                      # Exportable Hugging Face model files
├── hf_push_pipeline.py            # Production Hugging Face auto-push engine
├── marimo_train_and_push.py       # Reactive Marimo training dashboard
├── train_and_push_autonomous.py   # Standalone 4-hour training + auto-push CLI
├── train_bengali_96gb.py          # Production training script
├── deepseek_v4_flash_research.md  # 32-subagent DeepSeek V4 Flash research report
├── deepseek_research_papers/      # 9 official DeepSeek technical reports
└── README.md
```

---

## 📜 License & Citation
Released under the **Apache 2.0 License**.
Architectural concepts inspired by DeepSeek-AI technical reports.
