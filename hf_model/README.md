---
language:
- bn
- en
license: apache-2.0
library_name: transformers
tags:
- xe_droplychee
- mla
- moe
- multi-token-prediction
- grpo
- reasoning
- causal-lm
pipeline_tag: text-generation
---

# 🫐 m-droplychee (`xe_droplychee` Architecture)
### Next-Generation Frontier MoE & Latent Attention Architecture
**Optimized for Pre-Training on 94GB/96GB VRAM (NVIDIA RTX PRO 6000)**

---

## 📌 Model Summary

**m-droplychee** is powered by the custom **`xe_droplychee`** architecture—an ultra-efficient, highly scalable design (~6.5B Total, ~450M Active parameters per token) optimized for single-node multi-gigabyte workstations.

### Core Architectural Upgrades:
1. **Multi-Head Latent Attention (MLA) with QK-Head Normalization**: Slashes KV-cache memory bandwidth consumption by **~90%** via low-rank compression ($d_{c}^{KV} = 256$) coupled with decoupled 64-dimensional Rotary Position Embeddings (RoPE). RMSNorm on Q and K heads prevents attention entropy collapse and stabilizes FP8/BF16 training.
2. **Accelerated Attention Kernels**: Direct PyTorch SDPA integration for maximum throughput and memory efficiency.
3. **Fine-Grained MoE (64 Routed + 1 Dedicated Shared Expert)**: Top-4 dynamic expert routing with 1 dedicated shared expert (5 active experts per token).
4. **Auxiliary-Loss-Free Bias Balancing with Exponential Decay**: Dynamic parameter updates ($\pm \gamma$) with exponential decay factor ($0.999$) to eliminate expert collapse without sacrificing routing representation entropy.
5. **Multi-Token Prediction (MTP) Depth-1 Module**: Simultaneous 2-token prediction during pre-training, doubling token representation density and enabling speculative decoding speedups.
6. **Native GRPO Reinforcement Learning Engine**: Native policy ratio calculation with clipped surrogate objectives and KL divergence regularizers directly in the model class—completely bypassing the need for a separate critic model.

---

## 💻 Quickstart & Usage

```python
import torch
from transformers import AutoConfig, AutoModelForCausalLM

# Load model directly from Hugging Face Hub (or local directory)
model_id = "MD-Mushfiqur123/m-droplychee"

config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).cuda()

# Autoregressive generation utilizing MLA compressed KV cache
prompt = torch.tensor([[1, 100, 200, 300]]).cuda()
output = model.generate(prompt, max_new_tokens=64, temperature=0.7)
print("Generated Sequence:", output.tolist())
```

---

## 🏋️‍♂️ Continued Pre-Training on 94GB/96GB VRAM

```bash
python continue_training.py --model_name_or_path MD-Mushfiqur123/m-droplychee --dataset_name "your_bangla_dataset"
```

## 📜 License
Apache-2.0

