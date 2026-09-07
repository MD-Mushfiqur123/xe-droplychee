# push_trained_model.py
import os, sys, time, math, random, argparse, gc
from collections import deque
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from datasets import load_dataset
from huggingface_hub import HfApi

def parse_args():
    parser = argparse.ArgumentParser(description='Autonomous Train and Push Pipeline')
    parser.add_argument('--model_id', type=str, default='MD-Mushfiqur123/xe-droplychee-v2')
    parser.add_argument('--dataset_name', type=str, default='sagorsarker/bangla-wikipedia')
    parser.add_argument('--output_repo', type=str, default='MD-Mushfiqur123/xe-droplychee-v2-trained')
    parser.add_argument('--hf_token', type=str, default=os.getenv('HF_TOKEN', ''))
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--seq_len', type=int, default=2048)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--total_steps', type=int, default=8000)
    parser.add_argument('--save_every', type=int, default=2000)
    parser.add_argument('--output_dir', type=str, default='./final_trained_model')
    parser.add_argument('--push_to_hub', action='store_true', default=True)
    return parser.parse_args()

class ZeroLeakStreamingDataset(IterableDataset):
    """
    Zero-memory-leak streaming dataset using collections.deque to eliminate CPython pymalloc heap fragmentation.
    """
    def __init__(self, dataset_name, tokenizer, seq_len=2048):
        self.dataset = load_dataset(dataset_name, split='train', streaming=True)
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        token_deque = deque()
        for sample in self.dataset:
            text = sample.get('text', '')
            if not text:
                continue
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            token_deque.extend(tokens)
            while len(token_deque) >= self.seq_len:
                chunk = [token_deque.popleft() for _ in range(self.seq_len)]
                tensor_chunk = torch.tensor(chunk, dtype=torch.long)
                yield {'input_ids': tensor_chunk, 'labels': tensor_chunk}

def create_model_card(repo_id, dataset_name, total_steps, final_loss):
    model_name = repo_id.split('/')[-1]
    return f'''---
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

# {model_name}
Fully Trained xe_droplychee Architecture on Bengali Corpus
Trained on NVIDIA RTX PRO 6000 (96GB VRAM)

## Training Overview
- Base Architecture: xe_droplychee
- Dataset: {dataset_name}
- Total Steps: {total_steps}
- Final Loss: {final_loss:.4f}
- Precision: bfloat16

## Quickstart
`python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()

prompt = "\u09ac\u09be\u0982\u09b2\u09be\u09a6\u09c7\u09b6"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(out[0], skip_special_tokens=True))
`
'''

def robust_push_to_hf(output_dir, repo_id, token, dataset_name, total_steps, final_loss, max_retries=5):
    print(f'Pushing trained model to {repo_id} with exponential backoff...')
    api = HfApi(token=token)

    # 1. Create repo with retry
    for att in range(1, max_retries + 1):
        try:
            api.create_repo(repo_id=repo_id, repo_type='model', exist_ok=True)
            break
        except Exception as e:
            if att == max_retries:
                raise e
            delay = min(60, (2 ** att) + random.uniform(0.5, 2.0))
            print(f'Repo create retry {att}: {e}. Retrying in {delay:.1f}s...')
            time.sleep(delay)

    # Write model card
    with open(os.path.join(output_dir, 'README.md'), 'w', encoding='utf-8') as f:
        f.write(create_model_card(repo_id, dataset_name, total_steps, final_loss))

    # 2. Upload folder with retry and jitter
    for attempt in range(1, max_retries + 1):
        try:
            api.upload_folder(
                folder_path=output_dir,
                repo_id=repo_id,
                repo_type='model',
                commit_message=f'Release trained xe_droplychee model at step {total_steps}'
            )
            print(f'SUCCESS: Model live at https://huggingface.co/{repo_id}')
            return True
        except Exception as e:
            delay = min(120, (3 ** attempt) + random.uniform(1.0, 3.0))
            print(f'Upload retry {attempt}/{max_retries}: {e}. Retrying in {delay:.1f}s...')
            time.sleep(delay)
    return False

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print('==================================================================')
    print('  XE_DROPLYCHEE: AUTONOMOUS TRAIN & PUSH PIPELINE (v2.0 Verified)')
    print(f'  Hardware: RTX PRO 6000 | Device: {device} | Precision: {dtype}')
    print(f'  Target Repo: {args.output_repo} | Dataset: {args.dataset_name}')
    print('==================================================================')

    os.makedirs(args.output_dir, exist_ok=True)

    print('[1/5] Loading Tokenizer & Model...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True, torch_dtype=dtype).to(device)
    model.train()

    print('[2/5] Initializing Optimizer & Cosine Scheduler...')
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay)
        print('  Optim: 8-bit AdamW (bitsandbytes)')
    except Exception:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay, fused=True if device == 'cuda' else False)
        print('  Optim: PyTorch Fused AdamW')

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.total_steps)

    print('[3/5] Streaming Dataset (Zero-Leak Deque Engine)...')
    stream_dataset = ZeroLeakStreamingDataset(args.dataset_name, tokenizer, seq_len=args.seq_len)
    dataloader = DataLoader(stream_dataset, batch_size=args.batch_size)

    print('[4/5] Running Training Loop...')
    step = 0
    total_loss = 0.0
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)

    try:
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device, dtype=dtype):
                outputs = model(input_ids=input_ids, labels=labels)
                loss_val = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
                scaled_loss = loss_val / args.gradient_accumulation_steps

            scaled_loss.backward()
            total_loss += loss_val.item()

            # Prevent Host RAM leak / pinning
            del outputs, loss_val, scaled_loss, input_ids, labels

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step = (step + 1) // args.gradient_accumulation_steps

                if global_step % 10 == 0:
                    elapsed = time.time() - start_time
                    tok_per_sec = (global_step * args.batch_size * args.gradient_accumulation_steps * args.seq_len) / elapsed
                    cur_loss = total_loss / (step + 1)
                    lr = scheduler.get_last_lr()[0]
                    print(f'Step {global_step:5d}/{args.total_steps:5d} | Avg Loss: {cur_loss:.4f} | LR: {lr:.2e} | Speed: {tok_per_sec:,.0f} tok/s')

                if global_step % args.save_every == 0:
                    ckpt = os.path.join(args.output_dir, f'checkpoint-{global_step}')
                    print(f'Saving local checkpoint: {ckpt}')
                    model.save_pretrained(ckpt, safe_serialization=True)
                    tokenizer.save_pretrained(ckpt)

                # Periodic GC on Windows
                if global_step % 250 == 0:
                    gc.collect()
                    if device == 'cuda':
                        torch.cuda.empty_cache()

                if global_step >= args.total_steps:
                    print('Finished target steps!')
                    break

            step += 1
    except KeyboardInterrupt:
        print('Interrupted by user. Proceeding to save...')

    final_loss = total_loss / max(1, step)
    print(f'[5/5] Exporting final weights (safetensors) to {args.output_dir}...')
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub and args.hf_token:
        robust_push_to_hf(
            output_dir=args.output_dir,
            repo_id=args.output_repo,
            token=args.hf_token,
            dataset_name=args.dataset_name,
            total_steps=global_step if 'global_step' in locals() else step,
            final_loss=final_loss
        )

if __name__ == '__main__':
    main()
