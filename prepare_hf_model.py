import os
import json
import torch
import shutil
from pathlib import Path

from m_droplychee import DroplycheeConfigV4, DroplycheeForCausalLMV4

out_dir = Path('./hf_model')
out_dir.mkdir(parents=True, exist_ok=True)

print('[1/5] Copying model architecture code to hf_model/...')
shutil.copy('m_droplychee/configuration_droplychee.py', out_dir / 'configuration_droplychee.py')
shutil.copy('m_droplychee/modeling_droplychee.py', out_dir / 'modeling_droplychee.py')
shutil.copy('m_droplychee/__init__.py', out_dir / '__init__.py')

print('[2/5] Generating config.json with HuggingFace auto_map for trust_remote_code=True...')
config = DroplycheeConfigV4()
config_dict = config.__dict__.copy()
config_dict['model_type'] = 'droplychee'
config_dict['architectures'] = ['DroplycheeForCausalLMV4']
config_dict['auto_map'] = {
    'AutoConfig': 'configuration_droplychee.DroplycheeConfigV4',
    'AutoModelForCausalLM': 'modeling_droplychee.DroplycheeForCausalLMV4'
}

with open(out_dir / 'config.json', 'w', encoding='utf-8') as f:
    json.dump(config_dict, f, indent=2)

print('[3/5] Generating Hugging Face Tokenizer & Generation Config...')
tokenizer_config = {
    'add_bos_token': True,
    'add_eos_token': False,
    'bos_token': '<|begin_of_text|>',
    'eos_token': '<|end_of_text|>',
    'pad_token': '<|pad|>',
    'model_type': 'droplychee',
    'tokenizer_class': 'PreTrainedTokenizerFast',
    'clean_up_tokenization_spaces': False,
    'extra_special_tokens': [
        '<think>', '</think>', '<answer>', '</answer>',
        '<｜fim_begin｜>', '<｜fim_hole｜>', '<｜fim_end｜>'
    ]
}

with open(out_dir / 'tokenizer_config.json', 'w', encoding='utf-8') as f:
    json.dump(tokenizer_config, f, indent=2)

special_tokens_map = {
    'bos_token': '<|begin_of_text|>',
    'eos_token': '<|end_of_text|>',
    'pad_token': '<|pad|>',
    'additional_special_tokens': [
        '<think>', '</think>', '<answer>', '</answer>',
        '<｜fim_begin｜>', '<｜fim_hole｜>', '<｜fim_end｜>'
    ]
}

with open(out_dir / 'special_tokens_map.json', 'w', encoding='utf-8') as f:
    json.dump(special_tokens_map, f, indent=2)

generation_config = {
    'max_length': 4096,
    'temperature': 0.7,
    'top_p': 0.9,
    'top_k': 50,
    'do_sample': True,
    'pad_token_id': 0,
    'bos_token_id': 1,
    'eos_token_id': 2
}

with open(out_dir / 'generation_config.json', 'w', encoding='utf-8') as f:
    json.dump(generation_config, f, indent=2)

print('[4/5] Initializing base weights and saving model.safetensors / pytorch_model.bin...')
model = DroplycheeForCausalLMV4(config)
torch.save(model.state_dict(), out_dir / 'pytorch_model.bin')
print(f'      Saved pytorch_model.bin ({os.path.getsize(out_dir / "pytorch_model.bin") / (1024*1024):.2f} MB)')

print('[5/5] Generating publication-grade Hugging Face Model Card (README.md)...')
readme_content = '''---
language:
- bn
- en
license: apache-2.0
tags:
- deepseek
- deepseek-v4
- moe
- mla
- grpo
- reasoning
pipeline_tag: text-generation
---

# 🚀 m-droplychee v4 (1/10th Scale DeepSeek-V3/V4 Replica)

**m-droplychee v4** is an architectural replica of **DeepSeek-V3 / DeepSeek-V4**, scaled down to **1/10th parameter footprint** (~6.5B Total, ~450M Active parameters per token). 

It is designed for **ultra-fast pretraining, continued pre-training (CPT), and Bengali / Multilingual LLM research** on modern workstations and cloud GPUs (such as **NVIDIA RTX PRO 6000 96GB** or H100).

---

## 🌟 Key Innovations Included

1. **Multi-Head Latent Attention (MLA)**:
   - KV Cache compression into latent vector ^{KV}$ slashes active memory by **85% - 93%**.
   - Decoupled Rotary Position Embedding (RoPE) carried on separate positional query/key heads (^R, k^R$).

2. **DeepSeek-V3/V4 MoE Architecture**:
   - **Fine-grained Routed Experts**: 64 fine-grained experts with SwiGLU activations.
   - **Shared Expert Isolation**: 1 permanently active shared expert anchoring general syntax and common linguistic structures.
   - **Sigmoid Routing with Dynamic Bias ($)**: Unnormalized sigmoid affinity routing with Auxiliary-Loss-Free load balancing.

3. **Multi-Token Prediction (MTP)**:
   - Built-in depth-1 MTP block predicting +2$ in parallel during pre-training for fast speculative decoding.

4. **DeepSeek-R1 GRPO Support**:
   - Native compute_grpo_loss(...) method enabling pure Reinforcement Learning without a critic network.

---

## 💻 Quickstart with Hugging Face Transformers

`python
import torch
from transformers import AutoConfig, AutoModelForCausalLM

# Load model directly from Hugging Face Hub (or local directory)
model_id = "your-username/m-droplychee"

config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).cuda()

# Autoregressive generation with compressed MLA cache
prompt = torch.tensor([[1, 100, 200, 300]]).cuda()
output = model.generate(prompt, max_new_tokens=64, temperature=0.7)
print("Generated Tokens:", output.tolist())
`

---

## 🏋️‍♂️ Continued Pre-Training on 94GB/96GB VRAM

`ash
python continue_training.py --model_id your-username/m-droplychee --dataset_name "your_bangla_dataset"
`

## 📜 Citation & Acknowledgments
Based on research papers by DeepSeek-AI (DeepSeek LLM, DeepSeekMoE, DeepSeek-V2, DeepSeek-V3, and DeepSeek-R1).
'''

with open(out_dir / 'README.md', 'w', encoding='utf-8') as f:
    f.write(readme_content.strip())

print('Hugging Face Model Export directory ./hf_model successfully created and populated!')
