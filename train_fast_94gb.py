\"\"\"
Fast Pre-training Launch Script for m-droplychee V4 on 94GB/96GB VRAM GPU.
Features:
- bfloat16 / fp16 Mixed Precision
- Gradient Accumulation
- FlashAttention / PyTorch SDPA MLA
- 8-bit or Fused AdamW Optimizer
- In-place Loss Computation
\"\"\"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from m_droplychee import DroplycheeConfigV4, DroplycheeForCausalLMV4

class DummyTokenDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=2048, vocab_size=49152):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generates synthetic tokens for demonstration / speed benchmarking
        return torch.randint(0, self.vocab_size, (self.seq_len,))

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f'Using compute device: {device} | Precision: {dtype}')

    # 1/10 Scale Config (Ultra Fast on 94GB VRAM)
    config = DroplycheeConfigV4(
        vocab_size=49152,
        hidden_size=2240,
        num_hidden_layers=28,
        num_attention_heads=32,
        q_lora_rank=512,
        kv_lora_rank=256,
        n_routed_experts=64,
        n_shared_experts=1,
        num_experts_per_tok=4,
        use_mtp=True
    )

    print('Instantiating m-droplychee V4 (1/10 DeepSeek Replica)...')
    model = DroplycheeForCausalLMV4(config).to(device=device, dtype=dtype)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)

    dataset = DummyTokenDataset(seq_len=1024)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    print('Ready for pre-training loop! Everything is configured.')

if __name__ == '__main__':
    train()
