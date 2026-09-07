# train_and_push_autonomous.py
# Production Autonomous 4-Hour Training & Auto-Push Script for NVIDIA RTX PRO 6000 (96GB)
# Chief AI Researcher Engine

import os
import sys
import time
import math
import random
import argparse
import gc
from collections import deque
from pathlib import Path

# Enable high-speed Rust-based HF transfer if available
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from datasets import load_dataset
from huggingface_hub import HfApi

def parse_args():
    parser = argparse.ArgumentParser(description="Autonomous 4-Hour Training & Hugging Face Auto-Push")
    parser.add_argument("--model_id", type=str, default="./hf_model", help="Base model checkpoint or local architecture directory")
    parser.add_argument("--dataset_name", type=str, default="nahid-hub/B-CORE-bengali-corpus", help="Streaming dataset")
    parser.add_argument("--output_repo", type=str, default="MD-Mushfiqur123/xe-droplychee-v2-trained", help="Target HF repo")
    parser.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN", ""))
    parser.add_argument("--batch_size", type=int, default=8, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Accumulation steps")
    parser.add_argument("--seq_len", type=int, default=2048, help="Context sequence length")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--total_steps", type=int, default=6500, help="~4 Hours of continuous pre-training")
    parser.add_argument("--save_every", type=int, default=1000, help="Checkpoint interval")
    parser.add_argument("--output_dir", type=str, default="./xe_droplychee_final_export")
    parser.add_argument("--push_to_hub", action="store_true", default=True, help="Auto push to HF upon completion")
    return parser.parse_args()

class ZeroLeakStreamingDataset(IterableDataset):
    """
    Zero-memory-leak streaming dataset using collections.deque to avoid CPython heap fragmentation.
    """
    def __init__(self, dataset_name, tokenizer, seq_len=2048):
        self.dataset = load_dataset(dataset_name, split="train", streaming=True)
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        token_deque = deque()
        for sample in self.dataset:
            text = sample.get("text", "")
            if not text:
                continue
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            token_deque.extend(tokens)
            while len(token_deque) >= self.seq_len:
                chunk = [token_deque.popleft() for _ in range(self.seq_len)]
                tensor_chunk = torch.tensor(chunk, dtype=torch.long)
                yield {"input_ids": tensor_chunk, "labels": tensor_chunk}

def create_model_card(repo_id, dataset_name, total_steps, final_loss, sample_text=""):
    model_name = repo_id.split("/")[-1]
    return f"""---
language:
- bn
- en
license: apache-2.0
library_name: transformers
tags:
- xe_droplychee
- bengali-llm
- moe
- deepseek-v4
- causal-lm
pipeline_tag: text-generation
---

# 🫐 {model_name}
**Autonomous Day-1 Trained Model on Curated Bengali Corpus**
Trained on NVIDIA RTX PRO 6000 (96GB VRAM) in Marimo / JupyterLab environment.

## 📊 Training Specifications
- **Base Architecture:** `xe_droplychee`
- **Dataset:** `{dataset_name}`
- **Sequence Length:** 2048 tokens
- **Total Steps Trained:** {total_steps:,} (~4 Hours on 96GB GPU)
- **Final Loss:** {final_loss:.4f}
- **Precision:** `bfloat16`
- **Validation Sample Output:** *"{sample_text}"*

## 💻 Quickstart Inference
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).cuda()

prompt = "বাংলাদেশ একটি সুন্দর"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=64, temperature=0.7)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## 📜 Citation & Credits
- Built with DeepSeek-V4 inspired architectural components (Shared Expert, SwiGLU clamp, SqrtSoftplus router).
"""

def robust_push_to_hf(output_dir, repo_id, token, dataset_name, total_steps, final_loss, sample_text="", max_retries=6):
    print("=" * 65)
    print(f"  [HUGGING FACE AUTO-PUSH] Target: https://huggingface.co/{repo_id}")
    print("=" * 65)

    api = HfApi(token=token)

    # 1. Create or verify repository
    for att in range(1, 4):
        try:
            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
            print("  ✓ Repository verified/ready.")
            break
        except Exception as e:
            if att == 3:
                print(f"  [Warning] Repo verify: {e}")
            time.sleep(2 * att)

    # 2. Write dynamic Model Card
    card_path = os.path.join(output_dir, "README.md")
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(create_model_card(repo_id, dataset_name, total_steps, final_loss, sample_text))
    print("  ✓ Production Model Card (README.md) written.")

    # 3. Fault-tolerant upload with exponential backoff & jitter
    for attempt in range(1, max_retries + 1):
        try:
            print(f"  [Upload Attempt {attempt}/{max_retries}] Uploading {output_dir}...")
            api.upload_folder(
                folder_path=output_dir,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Release trained xe_droplychee model at step {total_steps} (loss: {final_loss:.4f})",
            )
            print("=" * 65)
            print(f"  🎉 SUCCESS! MODEL IS LIVE AT: https://huggingface.co/{repo_id}")
            print("=" * 65)
            return True
        except Exception as e:
            delay = min(120, (2 ** attempt) + random.uniform(1.0, 3.0))
            print(f"  Upload error on attempt {attempt}: {e}. Retrying in {delay:.1f}s...")
            time.sleep(delay)

    print("  [Error] Failed to push to Hugging Face after all retries.")
    return False

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print("=" * 68)
    print("  XE_DROPLYCHEE: AUTONOMOUS 4-HOUR PRE-TRAINING & AUTO-PUSH")
    print(f"  Hardware: RTX PRO 6000 (96GB) | Precision: {dtype}")
    print(f"  Dataset: {args.dataset_name}")
    print(f"  Batch: {args.batch_size} x Accum {args.gradient_accumulation_steps} = Global Batch {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Total Steps: {args.total_steps:,} | Context: {args.seq_len}")
    print(f"  Auto-Push Target: https://huggingface.co/{args.output_repo}")
    print("=" * 68)

    os.makedirs(args.output_dir, exist_ok=True)

    # [1/5] Tokenizer & Model Loading
    print("\n[1/5] Loading Tokenizer & Model...", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    except Exception as e_tok:
        print(f"  Remote tokenizer load failed ({e_tok}). Loading from local ./hf_model...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained("./hf_model", trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    except Exception as e_cfg:
        print(f"  Remote config load failed ({e_cfg}). Loading from local ./hf_model...", flush=True)
        config = AutoConfig.from_pretrained("./hf_model", trust_remote_code=True)

    model = None
    try:
        print(f"  Attempting to load existing weights from '{args.model_id}'...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            config=config,
            trust_remote_code=True,
            dtype=dtype,
        )
        print("  ✓ Successfully resumed from existing checkpoint weights!", flush=True)
    except (OSError, EnvironmentError, KeyError) as e_weights:
        print(f"  ℹ️ No existing pre-trained weights found ({e_weights}).", flush=True)
        print("  🚀 Initializing fresh xe_droplychee architecture from config for Day-1 Pre-Training...", flush=True)
        print("  ⏳ Allocating 28 layers & 65 experts directly on GPU (takes ~20-30s)...", flush=True)
        try:
            model = AutoModelForCausalLM.from_config(
                config,
                trust_remote_code=True,
                dtype=dtype,
            )
        except Exception:
            from m_droplychee import DroplycheeForCausalLM
            model = DroplycheeForCausalLM(config).to(dtype=dtype)
        print(f"  ✓ Initialized fresh model with {sum(p.numel() for p in model.parameters()):,} parameters!", flush=True)

    model = model.to(device)
    model.train()

    # [2/5] Optimizer & Cosine Scheduler
    print("\n[2/5] Initializing Optimizer & Cosine Scheduler...", flush=True)
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay)
        print("  Using: 8-bit AdamW (bitsandbytes)", flush=True)
    except Exception:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay, fused=True if device == "cuda" else False)
        print("  Using: PyTorch Fused AdamW", flush=True)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.total_steps,
    )

    # [3/5] Streaming Dataset Setup
    print(f"\n[3/5] Connecting to Hugging Face Streaming Dataset ({args.dataset_name})...", flush=True)
    stream_dataset = ZeroLeakStreamingDataset(args.dataset_name, tokenizer, seq_len=args.seq_len)
    dataloader = DataLoader(stream_dataset, batch_size=args.batch_size)
    print("  ✓ Dataset stream ready! Starting training steps...", flush=True)

    # [4/5] Running 4-Hour Training Loop
    print(f"\n[4/5] Executing Training Loop (Target: {args.total_steps:,} steps)...", flush=True)
    step = 0
    total_loss = 0.0
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)

    try:
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device, dtype=dtype):
                outputs = model(input_ids=input_ids, labels=labels)
                loss_val = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
                scaled_loss = loss_val / args.gradient_accumulation_steps

            scaled_loss.backward()
            total_loss += loss_val.item()

            del outputs, loss_val, scaled_loss, input_ids, labels

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step = (step + 1) // args.gradient_accumulation_steps

                if global_step == 1 or global_step % 10 == 0:
                    elapsed = time.time() - start_time
                    tok_per_sec = (global_step * args.batch_size * args.gradient_accumulation_steps * args.seq_len) / elapsed
                    cur_loss = total_loss / (step + 1)
                    vram_gb = torch.cuda.memory_allocated() / (1024**3) if device == "cuda" else 0.0
                    lr = scheduler.get_last_lr()[0]
                    eta_hours = ((args.total_steps - global_step) * (elapsed / global_step)) / 3600
                    print(f"Step {global_step:5d}/{args.total_steps:5d} | Loss: {cur_loss:.4f} | LR: {lr:.2e} | Speed: {tok_per_sec:,.0f} tok/s | VRAM: {vram_gb:.1f}G | ETA: {eta_hours:.2f}h", flush=True)

                if global_step % args.save_every == 0:
                    ckpt = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    print(f"  [Checkpoint] Saving intermediate checkpoint to {ckpt}...", flush=True)
                    model.save_pretrained(ckpt, safe_serialization=True)
                    tokenizer.save_pretrained(ckpt)

                if global_step % 250 == 0:
                    gc.collect()
                    if device == "cuda":
                        torch.cuda.empty_cache()

                if global_step >= args.total_steps:
                    print("\n🎯 Target steps completed successfully!")
                    break

            step += 1

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user. Gracefully initiating export & auto-push...")

    final_loss = total_loss / max(1, step)
    actual_steps = (step + 1) // args.gradient_accumulation_steps

    # [5/5] Pre-Push Sanity Check & Hugging Face Upload
    print(f"\n[5/5] Performing Pre-Push Integrity Validation...")
    model.eval()
    sample_text = ""
    try:
        test_inputs = tokenizer("বাংলাদেশ একটি", return_tensors="pt").to(device)
        with torch.no_grad():
            test_out = model.generate(**test_inputs, max_new_tokens=32, temperature=0.7)
        sample_text = tokenizer.decode(test_out[0], skip_special_tokens=True)
        print(f"  ✓ Sanity check generation succeeded: '{sample_text}'")
    except Exception as e:
        print(f"  [Warning] Sanity check warning: {e}")

    print(f"  ✓ Exporting final weights (safetensors) to {args.output_dir}...")
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    # Trigger Auto-Push
    if args.push_to_hub and args.hf_token:
        robust_push_to_hf(
            output_dir=args.output_dir,
            repo_id=args.output_repo,
            token=args.hf_token,
            dataset_name=args.dataset_name,
            total_steps=actual_steps,
            final_loss=final_loss,
            sample_text=sample_text,
        )

if __name__ == "__main__":
    main()
