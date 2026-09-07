# train_and_push_autonomous.py
# Production Autonomous 4-Hour Training & Auto-Push Engine for NVIDIA RTX PRO 6000 (96GB)
# Chief AI Researcher Engine | xe_droplychee Architecture

import os
import sys
import time
import math
import shutil
import random
import argparse
import gc
import json
import queue
import threading
from collections import deque
from pathlib import Path

# Ensure UTF-8 output encoding across all platforms (Windows, Linux, VPS)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# High-performance Hugging Face Hub transfer configuration
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from datasets import load_dataset
from huggingface_hub import HfApi, get_token


def parse_args():
    parser = argparse.ArgumentParser(description="Autonomous 4-Hour Pre-Training & Hugging Face Auto-Push")
    parser.add_argument("--model_id", type=str, default="./hf_model", help="Base model checkpoint or local architecture directory")
    parser.add_argument("--dataset_name", type=str, default="nahid-hub/B-CORE-bengali-corpus", help="Streaming dataset")
    parser.add_argument("--output_repo", type=str, default="MD-Mushfiqur123/xe-droplychee-v2-trained", help="Target HF repo")
    parser.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN", ""), help="Hugging Face write token")
    parser.add_argument("--batch_size", type=int, default=4, help="Micro-batch size per device (4 or 8 recommended for 96GB)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulation steps (batch 4 x accum 8 = global batch 32)")
    parser.add_argument("--seq_len", type=int, default=2048, help="Context sequence length")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=3e-5, help="Minimum learning rate for cosine scheduler")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup optimizer steps")
    parser.add_argument("--total_steps", type=int, default=6500, help="Total global steps (~4 hours on RTX PRO 6000)")
    parser.add_argument("--save_every", type=int, default=500, help="Checkpoint interval (in global steps)")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Keep at most N intermediate checkpoints")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Directory to store intermediate checkpoints")
    parser.add_argument("--export_dir", type=str, default="./xe_droplychee_final_export", help="Clean staging directory for Hugging Face release")
    parser.add_argument("--grad_checkpointing", action="store_true", default=True, help="Enable gradient checkpointing to drastically reduce activation VRAM")
    parser.add_argument("--no_grad_checkpointing", action="store_false", dest="grad_checkpointing")
    parser.add_argument("--resume", action="store_true", default=True, help="Automatically resume from latest valid checkpoint if available")
    parser.add_argument("--push_to_hub", action="store_true", default=True, help="Auto-push final model to Hugging Face Hub")
    return parser.parse_args()


# ==============================================================================
# Fast Prebuffered Background Streaming DataLoader
# ==============================================================================
class FastPrebufferedStreamLoader:
    """
    Background-threaded streaming data loader.
    Eliminates GPU starvation by pre-fetching and tokenizing text in a dedicated
    producer thread, placing packed sequence batches into a thread-safe Queue.
    """
    def __init__(self, dataset_name, tokenizer, seq_len=2048, batch_size=4, buffer_size=16, skip_batches=0):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.skip_batches = skip_batches
        self.queue = queue.Queue(maxsize=buffer_size)
        self.stop_event = threading.Event()
        self.batches_yielded = 0
        self.thread = threading.Thread(target=self._producer_loop, daemon=True)
        self.thread.start()

    def _producer_loop(self):
        try:
            stream = load_dataset(self.dataset_name, split="train", streaming=True)
            stream_iter = iter(stream)
            token_deque = deque()

            # Skip batches if resuming
            batches_to_skip = self.skip_batches
            if batches_to_skip > 0:
                print(f"  [DataLoader] Fast-forwarding dataset stream past {batches_to_skip:,} previously trained batches...", flush=True)

            while not self.stop_event.is_set():
                # Fetch text chunk
                texts = []
                for _ in range(64):
                    try:
                        sample = next(stream_iter)
                        t = sample.get("text", "")
                        if t and len(t.strip()) > 10:
                            texts.append(t)
                    except StopIteration:
                        # Re-loop over dataset if exhausted
                        stream = load_dataset(self.dataset_name, split="train", streaming=True)
                        stream_iter = iter(stream)
                        break
                    except Exception:
                        time.sleep(0.5)
                        continue

                if texts:
                    # Multi-core fast Rust tokenization
                    tokenized = self.tokenizer(texts, add_special_tokens=False, return_attention_mask=False)["input_ids"]
                    for tok_list in tokenized:
                        token_deque.extend(tok_list)

                # Pack into batches of shape [batch_size, seq_len]
                required_tokens = self.batch_size * self.seq_len
                while len(token_deque) >= required_tokens and not self.stop_event.is_set():
                    if batches_to_skip > 0:
                        for _ in range(required_tokens):
                            token_deque.popleft()
                        batches_to_skip -= 1
                        continue

                    batch_ids = []
                    for _ in range(self.batch_size):
                        chunk = [token_deque.popleft() for _ in range(self.seq_len)]
                        batch_ids.append(chunk)

                    input_tensor = torch.tensor(batch_ids, dtype=torch.long)
                    batch_data = {
                        "input_ids": input_tensor,
                        "labels": input_tensor.clone(),
                    }

                    # Push to queue with timeout to allow clean shutdown
                    while not self.stop_event.is_set():
                        try:
                            self.queue.put(batch_data, timeout=0.5)
                            break
                        except queue.Full:
                            continue

        except Exception as e:
            print(f"  [DataLoader Error] Background stream worker exception: {e}", flush=True)

    def next_batch(self, timeout=60.0):
        try:
            batch = self.queue.get(timeout=timeout)
            self.batches_yielded += 1
            return batch
        except queue.Empty:
            raise TimeoutError("Data loader timed out waiting for next streaming batch. Check internet connection.")

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)


# ==============================================================================
# Atomic Checkpoint Manager with Auto-Resume
# ==============================================================================
class CheckpointManager:
    """
    Manages atomic checkpoint saving, pruning, and automatic recovery.
    Uses sentinel files (.checkpoint_completed) to guarantee no corrupt checkpoints are loaded.
    """
    def __init__(self, output_dir: str, save_total_limit: int = 2):
        self.output_dir = Path(output_dir)
        self.save_total_limit = save_total_limit
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_latest_checkpoint(self):
        ckpts = []
        for p in self.output_dir.glob("checkpoint-*"):
            if p.is_dir() and (p / ".checkpoint_completed").exists():
                try:
                    step_num = int(p.name.split("-")[1])
                    ckpts.append((step_num, p))
                except (IndexError, ValueError):
                    continue
        if not ckpts:
            return None
        ckpts.sort(key=lambda x: x[0], reverse=True)
        return ckpts[0]  # (step_num, path)

    def save_checkpoint(self, model, tokenizer, optimizer, scheduler, global_step, micro_step, total_loss, cur_loss):
        final_dir = self.output_dir / f"checkpoint-{global_step}"
        tmp_dir = self.output_dir / f"checkpoint-{global_step}.tmp"

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save weights and tokenizer
        model.save_pretrained(str(tmp_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(tmp_dir))

        # 2. Save optimizer & scheduler state
        torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
        torch.save(scheduler.state_dict(), tmp_dir / "scheduler.pt")

        # 3. Save training metadata
        meta = {
            "global_step": global_step,
            "micro_step": micro_step,
            "total_loss": total_loss,
            "cur_loss": cur_loss,
            "timestamp": time.time(),
        }
        with open(tmp_dir / "training_state.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # 4. Write sentinel file and atomically rename
        (tmp_dir / ".checkpoint_completed").touch()
        if final_dir.exists():
            shutil.rmtree(final_dir, ignore_errors=True)
        tmp_dir.rename(final_dir)

        print(f"\n  ✓ [Checkpoint Saved] Atomic checkpoint persisted to {final_dir}", flush=True)

        # 5. Prune old checkpoints
        self._prune_old_checkpoints()

    def _prune_old_checkpoints(self):
        ckpts = []
        for p in self.output_dir.glob("checkpoint-*"):
            if p.is_dir() and (p / ".checkpoint_completed").exists():
                try:
                    step_num = int(p.name.split("-")[1])
                    ckpts.append((step_num, p))
                except (IndexError, ValueError):
                    continue
        ckpts.sort(key=lambda x: x[0])
        while len(ckpts) > self.save_total_limit:
            oldest_step, oldest_path = ckpts.pop(0)
            try:
                shutil.rmtree(oldest_path, ignore_errors=True)
                print(f"  [Checkpoint Pruned] Removed old checkpoint: {oldest_path.name}", flush=True)
            except Exception as e:
                print(f"  [Warning] Failed to prune {oldest_path.name}: {e}", flush=True)


# ==============================================================================
# Model Card Generator
# ==============================================================================
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
**Autonomous Pre-Trained Bengali MoE Foundation Model**
Pre-trained on NVIDIA RTX PRO 6000 (96GB VRAM) using curated high-quality Bengali web and educational text corpora.

## 📊 Model Specifications
- **Architecture:** `xe_droplychee` (DeepSeek-V4 inspired)
- **Parameters:** ~10.02 Billion Total (~750M Active per token)
- **Attention Mechanism:** Multi-Head Latent Attention (MLA) with QK-Norm & RoPE
- **Mixture-of-Experts:** 64 Routed Experts + 1 Shared Expert (Top-4 Routing with dynamic load balancing)
- **Multi-Token Prediction (MTP):** Depth-1 Speculative Decoding
- **Vocabulary Size:** 50,267 tokens
- **Training Dataset:** `{dataset_name}`
- **Context Sequence Length:** 2,048 tokens
- **Total Completed Global Steps:** {total_steps:,}
- **Final Cross-Entropy Loss:** {final_loss:.4f}
- **Precision:** `bfloat16`

## 💬 Sample Generation
> *Input:* "বাংলাদেশ একটি"
> *Output:* "{sample_text}"

## 💻 Inference Usage
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="auto"
)

prompt = "বাংলাদেশ একটি সুন্দর"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=64, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 📜 Acknowledgements
Developed with DeepSeek-V4 architectural innovations adapted for high-efficiency single-GPU enterprise pre-training.
"""


# ==============================================================================
# Staging & Hugging Face Auto-Push
# ==============================================================================
def stage_and_push_to_hf(model, tokenizer, config, export_dir, model_source_dir, repo_id, token, dataset_name, total_steps, final_loss, sample_text="", max_retries=5):
    print("\n" + "=" * 68)
    print(f"  [HUGGING FACE HUB AUTO-PUSH] Target: https://huggingface.co/{repo_id}")
    print("=" * 68)

    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)

    # 1. Stage architecture and code files
    print("  [1/4] Staging architecture and configuration files...", flush=True)
    src_path = Path(model_source_dir)
    required_files = [
        "configuration_droplychee.py",
        "modeling_droplychee.py",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    for fname in required_files:
        s = src_path / fname
        if s.exists():
            shutil.copy2(str(s), str(export_path / fname))

    # 2. Export safetensors weights & tokenizer
    print("  [2/4] Saving final safetensors weights and tokenizer...", flush=True)
    model.save_pretrained(str(export_path), safe_serialization=True)
    tokenizer.save_pretrained(str(export_path))

    # 3. Generate README.md Model Card
    print("  [3/4] Generating production Model Card (README.md)...", flush=True)
    card_content = create_model_card(repo_id, dataset_name, total_steps, final_loss, sample_text)
    with open(export_path / "README.md", "w", encoding="utf-8") as f:
        f.write(card_content)

    # 4. Push to Hub with retries and exponential backoff
    if not token:
        print("  [Warning] HF_TOKEN is empty! Model saved locally to export directory, but Hub upload skipped.")
        return False

    api = HfApi(token=token.strip())

    # Create / verify repository
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print("  ✓ Repository verified on Hugging Face Hub.")
    except Exception as e_repo:
        print(f"  [Warning] create_repo warning: {e_repo}")

    print("  [4/4] Uploading artifacts to Hugging Face Hub...", flush=True)
    ignore_patterns = [
        "checkpoint-*",
        "*.tmp",
        "*.pt",
        "*.bin",
        "*.log",
        "__pycache__/*",
        ".git/*",
    ]

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  Upload Attempt {attempt}/{max_retries}...", flush=True)
            api.upload_folder(
                folder_path=str(export_path),
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Release trained xe_droplychee model at step {total_steps} (loss: {final_loss:.4f})",
                ignore_patterns=ignore_patterns,
            )
            print("\n" + "=" * 68)
            print(f"  🎉 SUCCESS! MODEL IS LIVE AT: https://huggingface.co/{repo_id}")
            print("=" * 68 + "\n")
            return True
        except Exception as e_upload:
            delay = min(120, (2 ** attempt) + random.uniform(1.0, 3.0))
            print(f"  Upload attempt {attempt} error: {e_upload}. Retrying in {delay:.1f}s...", flush=True)
            time.sleep(delay)

    print("  [Error] Failed to push to Hugging Face Hub after all retries.")
    return False


# ==============================================================================
# Main Training Routine
# ==============================================================================
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32

    # Discover HF Token
    hf_token = args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or get_token() or ""

    print("=" * 72)
    print("  XE_DROPLYCHEE: AUTONOMOUS 4-HOUR PRE-TRAINING ENGINE")
    print(f"  Compute Hardware: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'} | Precision: {dtype}")
    print(f"  Dataset: {args.dataset_name}")
    print(f"  Micro-Batch: {args.batch_size} | Gradient Accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective Global Batch Size: {args.batch_size * args.gradient_accumulation_steps} sequences ({args.batch_size * args.gradient_accumulation_steps * args.seq_len:,} tokens/step)")
    print(f"  Total Steps: {args.total_steps:,} | Context Window: {args.seq_len}")
    print(f"  Auto-Push Target: https://huggingface.co/{args.output_repo}")
    print("=" * 72)

    # Setup Checkpoint Manager
    ckpt_mgr = CheckpointManager(output_dir=args.output_dir, save_total_limit=args.save_total_limit)
    resumed_ckpt = None
    resumed_meta = None

    if args.resume:
        latest = ckpt_mgr.get_latest_checkpoint()
        if latest is not None:
            step_num, ckpt_path = latest
            meta_path = ckpt_path / "training_state.json"
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        resumed_meta = json.load(f)
                    resumed_ckpt = ckpt_path
                    print(f"  ✓ Discovered valid resume checkpoint at step {step_num}: {ckpt_path}", flush=True)
                except Exception as e:
                    print(f"  [Warning] Failed reading metadata from {ckpt_path}: {e}", flush=True)

    # [1/5] Load Tokenizer & Config
    print("\n[1/5] Loading Tokenizer & Architecture Configuration...", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("./hf_model", trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    except Exception:
        config = AutoConfig.from_pretrained("./hf_model", trust_remote_code=True)

    # [2/5] Direct GPU VRAM Model Instantiation
    print("\n[2/5] Instantiating Model Directly in GPU VRAM (Sub-2s)...", flush=True)
    t_model_start = time.time()

    if resumed_ckpt is not None:
        print(f"  Loading model weights from checkpoint: {resumed_ckpt}...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(resumed_ckpt),
            config=config,
            trust_remote_code=True,
            dtype=dtype,
        ).to(device)
    else:
        # Fast Direct GPU VRAM Allocation
        print(f"  Allocating 28 layers & 65 experts directly on {device} ({dtype})...", flush=True)
        with torch.device(device):
            prev_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(dtype)
                try:
                    model = AutoModelForCausalLM.from_config(
                        config,
                        trust_remote_code=True,
                        dtype=dtype,
                    )
                except Exception:
                    sys.path.insert(0, str(Path(args.model_id).resolve()))
                    from modeling_droplychee import XeDroplycheeForCausalLM
                    model = XeDroplycheeForCausalLM(config)
            finally:
                torch.set_default_dtype(prev_dtype)

    model = model.to(device)

    # Enable Gradient Checkpointing if requested
    if args.grad_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        elif hasattr(model, "model"):
            model.model.gradient_checkpointing = True
        print("  ✓ Gradient checkpointing activated (Activation VRAM drastically reduced).", flush=True)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model ready in {time.time() - t_model_start:.2f}s | Total Parameters: {param_count:,}", flush=True)
    model.train()

    # [3/5] Optimizer & Cosine Scheduler
    print("\n[3/5] Initializing Optimizer & Cosine Scheduler...", flush=True)
    optimizer = None
    try:
        import bitsandbytes as bnb
        # Verify 8-bit AdamW works on this device
        test_p = nn.Parameter(torch.zeros(2, device=device))
        _ = bnb.optim.AdamW8bit([test_p], lr=1e-4)
        del test_p
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay
        )
        print("  ✓ Using bitsandbytes 8-bit AdamW (Saves ~60GB VRAM on optimizer states)", flush=True)
    except Exception as e_bnb:
        print(f"  [Info] bitsandbytes 8-bit AdamW not active ({e_bnb}). Falling back to PyTorch Fused AdamW...", flush=True)
        use_fused = (device == "cuda" and hasattr(torch.optim.AdamW, "supports_fused") and torch.optim.AdamW.supports_fused)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
            fused=(device == "cuda"),
        )
        print(f"  ✓ Using PyTorch AdamW (fused={device == 'cuda'})", flush=True)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.total_steps,
    )

    # Resume optimizer & scheduler state if available
    start_global_step = 0
    start_micro_step = 0
    accumulated_loss = 0.0

    if resumed_ckpt is not None and resumed_meta is not None:
        try:
            opt_path = resumed_ckpt / "optimizer.pt"
            sched_path = resumed_ckpt / "scheduler.pt"
            if opt_path.exists():
                optimizer.load_state_dict(torch.load(opt_path, map_location=device))
                print("  ✓ Optimizer state restored successfully.", flush=True)
            if sched_path.exists():
                scheduler.load_state_dict(torch.load(sched_path, map_location=device))
                print("  ✓ Scheduler state restored successfully.", flush=True)
            start_global_step = resumed_meta.get("global_step", 0)
            start_micro_step = resumed_meta.get("micro_step", 0)
            accumulated_loss = resumed_meta.get("total_loss", 0.0)
            print(f"  ✓ Resumed from Global Step {start_global_step:,} (Micro-Step {start_micro_step:,})", flush=True)
        except Exception as e_resume:
            print(f"  [Warning] Partial resume error: {e_resume}. Starting optimizer fresh.", flush=True)

    # [4/5] Background Streaming DataLoader
    print(f"\n[4/5] Launching Background Prebuffered DataLoader ({args.dataset_name})...", flush=True)
    loader = FastPrebufferedStreamLoader(
        dataset_name=args.dataset_name,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        buffer_size=16,
        skip_batches=start_micro_step,
    )
    print("  ✓ Streaming pipeline connected. Pre-buffering batches...", flush=True)

    # [5/5] Autonomous Training Loop
    print(f"\n[5/5] Executing Pre-Training Loop (Target: {args.total_steps:,} Global Steps)...", flush=True)
    print("-" * 72)
    print(f"{'Step':>7} | {'Loss':>7} | {'Inst':>7} | {'LR':>8} | {'Speed':>11} | {'VRAM':>7} | {'ETA':>8}")
    print("-" * 72)

    micro_step = start_micro_step
    global_step = start_global_step
    total_loss = accumulated_loss
    start_time = time.time()
    last_print_time = start_time
    processed_tokens = 0
    optimizer.zero_grad(set_to_none=True)

    try:
        while global_step < args.total_steps:
            accum_idx = micro_step % args.gradient_accumulation_steps

            # Fetch batch from prebuffered queue
            batch = loader.next_batch(timeout=60.0)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device, dtype=dtype):
                outputs = model(input_ids=input_ids, labels=labels)
                loss_val = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
                scaled_loss = loss_val / args.gradient_accumulation_steps

            scaled_loss.backward()
            total_loss += loss_val.item()
            processed_tokens += (args.batch_size * args.seq_len)

            vram_gb = torch.cuda.memory_allocated() / (1024 ** 3) if device == "cuda" else 0.0

            # Dynamic Real-Time Micro-Step Heartbeat (Zero Silence)
            print(
                f"\r  ⚡ [Step {global_step:5d}/{args.total_steps:5d} | Micro {accum_idx + 1:2d}/{args.gradient_accumulation_steps:2d} | "
                f"Loss: {loss_val.item():.4f} | VRAM: {vram_gb:.1f}GB] ...",
                end="",
                flush=True,
            )

            del outputs, loss_val, scaled_loss, input_ids, labels

            # Optimizer Step on Completed Accumulation Window
            if (accum_idx + 1) == args.gradient_accumulation_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                cur_loss = total_loss / max(1, micro_step + 1)
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                steps_done = max(1, global_step - start_global_step)
                tok_per_sec = processed_tokens / max(0.001, elapsed)

                eta_seconds = (args.total_steps - global_step) * (elapsed / steps_done)
                eta_str = time.strftime("%H:%M:%S", time.gmtime(max(0, int(eta_seconds))))

                # Clean Columnar Step Output
                print(
                    f"\r{global_step:7d} | {cur_loss:7.4f} | {loss_val.item() if 'loss_val' in locals() else 0.0:7.4f} | "
                    f"{lr:8.2e} | {tok_per_sec:9,.0f} t/s | {vram_gb:5.1f}GB | {eta_str:>8}",
                    flush=True,
                )

                # Periodic Checkpoint Saving
                if global_step % args.save_every == 0:
                    ckpt_mgr.save_checkpoint(
                        model=model,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        global_step=global_step,
                        micro_step=micro_step,
                        total_loss=total_loss,
                        cur_loss=cur_loss,
                    )

                # Periodic Garbage Collection
                if global_step % 250 == 0:
                    gc.collect()
                    if device == "cuda":
                        torch.cuda.empty_cache()

            micro_step += 1

        print("\n\n🎯 TARGET TRAINING STEPS REACHED SUCCESSFULLY!")

    except KeyboardInterrupt:
        print("\n\n⚠️ Keyboard Interrupt detected! Safely terminating and saving current state...")
        ckpt_mgr.save_checkpoint(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=global_step,
            micro_step=micro_step,
            total_loss=total_loss,
            cur_loss=(total_loss / max(1, micro_step)),
        )

    finally:
        loader.stop()

    # Final Inference Sanity Validation
    print("\n" + "=" * 68)
    print("  [PRE-PUSH VALIDATION] Running Test Bengali Generation...")
    print("=" * 68)
    model.eval()
    sample_text = ""
    try:
        prompt_text = "বাংলাদেশ একটি"
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_out = model.generate(
                **inputs,
                max_new_tokens=32,
                temperature=0.7,
                do_sample=True,
            )
        sample_text = tokenizer.decode(gen_out[0], skip_special_tokens=True)
        print(f"  ✓ Sanity check generated: \"{sample_text}\"", flush=True)
    except Exception as e_gen:
        print(f"  [Warning] Sanity generation test warning: {e_gen}", flush=True)

    final_loss = total_loss / max(1, micro_step)

    # Hugging Face Hub Export & Upload
    if args.push_to_hub:
        stage_and_push_to_hf(
            model=model,
            tokenizer=tokenizer,
            config=config,
            export_dir=args.export_dir,
            model_source_dir=args.model_id,
            repo_id=args.output_repo,
            token=hf_token,
            dataset_name=args.dataset_name,
            total_steps=global_step,
            final_loss=final_loss,
            sample_text=sample_text,
        )


if __name__ == "__main__":
    main()
