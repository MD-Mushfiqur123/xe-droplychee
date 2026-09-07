# train_bengali_96gb.py
# Production Pre-Training Script for xe_droplychee on NVIDIA RTX PRO 6000 (96GB)
# Dataset: nahid-hub/B-CORE-bengali-corpus (1 Epoch)

import os
import time
import math
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description='Train xe_droplychee on 96GB GPU')
    parser.add_argument('--model_id', type=str, default='MD-Mushfiqur123/m-droplychee')
    parser.add_argument('--dataset_name', type=str, default='nahid-hub/B-CORE-bengali-corpus')
    parser.add_argument('--batch_size', type=int, default=8, help='Per-device batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Accumulation steps')
    parser.add_argument('--seq_len', type=int, default=2048, help='Context sequence length')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Peak learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--warmup_steps', type=int, default=1500)
    parser.add_argument('--total_steps', type=int, default=65000, help='~1 Epoch across 4.3B tokens')
    parser.add_argument('--save_every', type=int, default=2500)
    parser.add_argument('--output_dir', type=str, default='./xe_droplychee_checkpoints')
    return parser.parse_args()

class BengaliTokenizedStream(IterableDataset):
    def __init__(self, dataset_name, tokenizer, seq_len=2048):
        self.dataset = load_dataset(dataset_name, split='train', streaming=True)
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        buffer = []
        for sample in self.dataset:
            text = sample.get('text', '')
            if not text:
                continue
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)
            while len(buffer) >= self.seq_len:
                chunk = buffer[:self.seq_len]
                buffer = buffer[self.seq_len:]
                tensor_chunk = torch.tensor(chunk, dtype=torch.long)
                yield {'input_ids': tensor_chunk, 'labels': tensor_chunk.clone()}

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print('==================================================================')
    print('  xe_droplychee: Production Pre-Training (RTX PRO 6000 96GB)')
    print(f'  Dataset: {args.dataset_name}')
    print(f'  Device: {device} | Precision: {dtype}')
    print(f'  Batch Size: {args.batch_size} x Accum {args.gradient_accumulation_steps} = Global Batch {args.batch_size * args.gradient_accumulation_steps}')
    print(f'  Tokens per Step: {args.batch_size * args.gradient_accumulation_steps * args.seq_len:,}')
    print('==================================================================')

    os.makedirs(args.output_dir, exist_ok=True)

    print('[1/4] Loading Tokenizer & Config...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)

    print('[2/4] Initializing Model weights...')
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=dtype
    ).to(device)
    model.train()

    # Optional torch.compile for extra 20% speedup on Ada/Blackwell
    # model = torch.compile(model)

    print('[3/4] Initializing Optimizer & Cosine Scheduler...')
    # Try 8-bit AdamW if bitsandbytes is available, else standard PyTorch fused AdamW
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay)
        print('  Using: 8-bit AdamW (bitsandbytes)')
    except ImportError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay, fused=True if device == 'cuda' else False)
        print('  Using: PyTorch Fused AdamW')

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.total_steps
    )

    print('[4/4] Connecting Hugging Face Streaming Dataset...')
    stream_dataset = BengaliTokenizedStream(args.dataset_name, tokenizer, seq_len=args.seq_len)
    dataloader = DataLoader(stream_dataset, batch_size=args.batch_size)

    print('>>> Starting Training Loop (1 Epoch Target) <<<')
    step = 0
    total_loss = 0.0
    start_time = time.time()
    optimizer.zero_grad()

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        with torch.amp.autocast(device_type=device, dtype=dtype):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / args.gradient_accumulation_steps

        loss.backward()
        total_loss += loss.item() * args.gradient_accumulation_steps

        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step = (step + 1) // args.gradient_accumulation_steps

            if global_step % 10 == 0:
                elapsed = time.time() - start_time
                tps = (global_step * args.batch_size * args.gradient_accumulation_steps * args.seq_len) / elapsed
                avg_loss = total_loss / (global_step * args.gradient_accumulation_steps)
                lr = scheduler.get_last_lr()[0]
                print(f'Step {global_step}/{args.total_steps} | Loss: {loss.item() * args.gradient_accumulation_steps:.4f} | Avg Loss: {avg_loss:.4f} | LR: {lr:.2e} | Speed: {tps:,.0f} tok/s')

            if global_step % args.save_every == 0:
                save_path = os.path.join(args.output_dir, f'checkpoint-{global_step}')
                print(f'Saving checkpoint to {save_path}...')
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)

            if global_step >= args.total_steps:
                print('Training completed 1 full epoch!')
                final_path = os.path.join(args.output_dir, 'final_model')
                model.save_pretrained(final_path)
                tokenizer.save_pretrained(final_path)
                break

        step += 1

if __name__ == '__main__':
    main()
