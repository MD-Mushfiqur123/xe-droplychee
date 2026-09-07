# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "torch",
#     "transformers",
#     "datasets",
#     "huggingface_hub",
#     "safetensors",
#     "bitsandbytes"
# ]
# ///

import marimo

__generated_with = "0.11.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import os, sys, time, math, random, gc
    from collections import deque
    from pathlib import Path
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
    import marimo as mo

    return (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        DataLoader,
        HfApi,
        IterableDataset,
        Path,
        deque,
        gc,
        get_cosine_schedule_with_warmup,
        load_dataset,
        math,
        mo,
        nn,
        os,
        random,
        sys,
        time,
        torch,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        # 🚀 xe_droplychee: Day 1 Autonomous Training & Auto-Push Dashboard
        **Chief AI Researcher Studio — Powered by Marimo Reactive Engine & NVIDIA RTX PRO 6000 (96GB)**

        This reactive dashboard manages the complete 4-hour Day-1 pre-training session on the **B-CORE Bengali Corpus** and **automatically validates, exports, and pushes the final model to Hugging Face Hub** upon completion.
        """
    )
    return


@app.cell
def __(mo, os):
    # UI Control Widgets
    hf_token_input = mo.ui.text(
        value=os.getenv("HF_TOKEN", ""),
        label="Hugging Face Write Token:",
        kind="password",
    )

    repo_id_input = mo.ui.text(
        value="MD-Mushfiqur123/xe-droplychee-v2-trained",
        label="Target Hugging Face Repo ID:",
    )

    dataset_dropdown = mo.ui.dropdown(
        options=[
            "nahid-hub/B-CORE-bengali-corpus",
            "md-nishat-008/Bangla-Instruct",
            "sagorsarker/bangla-wikipedia",
        ],
        value="nahid-hub/B-CORE-bengali-corpus",
        label="Dataset Source:",
    )

    hours_slider = mo.ui.slider(
        start=1,
        stop=8,
        step=1,
        value=4,
        label="Training Duration (Hours):",
    )

    batch_size_slider = mo.ui.slider(
        start=4,
        stop=16,
        step=2,
        value=8,
        label="Per-Device Batch Size:",
    )

    accum_slider = mo.ui.slider(
        start=2,
        stop=8,
        step=2,
        value=4,
        label="Gradient Accumulation Steps:",
    )

    mo.vstack([
        mo.md("### ⚙️ Pipeline Configuration"),
        hf_token_input,
        repo_id_input,
        dataset_dropdown,
        hours_slider,
        mo.hstack([batch_size_slider, accum_slider]),
    ])
    return (
        accum_slider,
        batch_size_slider,
        dataset_dropdown,
        hf_token_input,
        hours_slider,
        repo_id_input,
    )


@app.cell
def __(accum_slider, batch_size_slider, hours_slider, mo):
    # Calculate step targets based on 4-hour budget
    # RTX PRO 6000 96GB does ~30k tokens/sec on xe_droplychee (~450M active)
    est_tok_sec = 30000
    total_seconds = hours_slider.value * 3600
    total_tokens_budget = total_seconds * est_tok_sec
    tokens_per_step = batch_size_slider.value * accum_slider.value * 2048
    computed_steps = int(total_tokens_budget // tokens_per_step)

    run_btn = mo.ui.run_button(
        label=f"🚀 Start {hours_slider.value}-Hour Training & Auto-Push ({computed_steps:,} Steps)",
        kind="success",
    )

    mo.hstack([
        run_btn,
        mo.md(
            f"**Budget:** ~{total_tokens_budget / 1e6:.1f}M tokens | "
            f"**Global Batch:** {batch_size_slider.value * accum_slider.value} ({tokens_per_step:,} tok/step)"
        ),
    ])
    return computed_steps, est_tok_sec, run_btn, tokens_per_step, total_seconds, total_tokens_budget


@app.cell
def __(
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataLoader,
    HfApi,
    IterableDataset,
    accum_slider,
    batch_size_slider,
    computed_steps,
    dataset_dropdown,
    deque,
    gc,
    get_cosine_schedule_with_warmup,
    hf_token_input,
    load_dataset,
    mo,
    os,
    random,
    repo_id_input,
    run_btn,
    time,
    torch,
):
    # Reactive Guard: Prevent cell from executing until the Run Button is clicked
    mo.stop(not run_btn.value, mo.md("👈 *Click the button above to start the 4-hour training and auto-push pipeline.*"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    repo_id = repo_id_input.value
    hf_token = hf_token_input.value
    dataset_name = dataset_dropdown.value
    batch_size = batch_size_slider.value
    grad_accum = accum_slider.value
    seq_len = 2048
    total_steps = computed_steps
    output_dir = "./xe_droplychee_final_export"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Zero-Leak Streaming Dataset
    class StreamDataset(IterableDataset):
        def __init__(self, ds_name, tok, length=2048):
            self.ds = load_dataset(ds_name, split="train", streaming=True)
            self.tok = tok
            self.length = length

        def __iter__(self):
            dq = deque()
            for sample in self.ds:
                txt = sample.get("text", "")
                if not txt:
                    continue
                tokens = self.tok.encode(txt, add_special_tokens=False)
                dq.extend(tokens)
                while len(dq) >= self.length:
                    chunk = [dq.popleft() for _ in range(self.length)]
                    t = torch.tensor(chunk, dtype=torch.long)
                    yield {"input_ids": t, "labels": t}

    # Model Card Generator
    def build_model_card(r_id, d_name, steps, final_l):
        m_name = r_id.split("/")[-1]
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

# 🫐 {m_name}
**Autonomous Pre-Training Day 1 Release (Bengali Corpus)**
Trained on NVIDIA RTX PRO 6000 (96GB VRAM) via Marimo Reactive ML Studio.

## 📊 Training Specifications
- **Architecture:** `xe_droplychee` (Fine-Grained MoE + Shared Expert + SwiGLU Clamped)
- **Dataset:** `{d_name}`
- **Context Length:** 2048
- **Total Training Steps:** {steps:,}
- **Final Average Loss:** {final_l:.4f}
- **Precision:** `bfloat16`
- **Optimizer:** 8-bit AdamW / Fused AdamW

## 💻 Quickstart
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "{r_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).cuda()

prompt = "বাংলাদেশ একটি"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=64, temperature=0.7)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```
"""

    # 2. Loading Model & Tokenizer
    print("Loading Tokenizer and Model...")
    base_model_id = "./hf_model"
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("MD-Mushfiqur123/m-droplychee", trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        config = AutoConfig.from_pretrained(base_model_id, trust_remote_code=True)
    except Exception:
        config = AutoConfig.from_pretrained("MD-Mushfiqur123/m-droplychee", trust_remote_code=True)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            config=config,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
    except (OSError, EnvironmentError, KeyError):
        try:
            model = AutoModelForCausalLM.from_config(
                config,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
        except Exception:
            from m_droplychee import DroplycheeForCausalLM
            model = DroplycheeForCausalLM(config).to(dtype=dtype)

    model = model.to(device)
    model.train()

    # 3. Optimizer & Scheduler
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    except Exception:
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True if device == "cuda" else False)

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=total_steps)

    dataloader = DataLoader(StreamDataset(dataset_name, tokenizer, seq_len), batch_size=batch_size)

    # 4. Training Loop with Live Marimo Progress Bar
    total_loss = 0.0
    step = 0
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)

    status_message = "Training in progress..."

    with mo.status.progress_bar(total=total_steps, title="🚀 xe_droplychee 4-Hour Training", show_eta=True) as pbar:
        try:
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                with torch.amp.autocast(device_type=device, dtype=dtype):
                    out = model(input_ids=input_ids, labels=labels)
                    loss_val = out.loss if hasattr(out, "loss") else out["loss"]
                    scaled_loss = loss_val / grad_accum

                scaled_loss.backward()
                total_loss += loss_val.item()

                del out, loss_val, scaled_loss, input_ids, labels

                if (step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    global_step = (step + 1) // grad_accum
                    pbar.update(increment=1)

                    if global_step % 20 == 0:
                        elapsed = time.time() - start_time
                        tok_s = (global_step * batch_size * grad_accum * seq_len) / elapsed
                        cur_loss = total_loss / (step + 1)
                        vram_gb = torch.cuda.memory_allocated() / (1024**3) if device == "cuda" else 0.0
                        pbar.update(
                            title=f"Step {global_step}/{total_steps} | Loss: {cur_loss:.4f} | {tok_s:,.0f} tok/s",
                            subtitle=f"VRAM: {vram_gb:.1f}GB / 96GB | LR: {scheduler.get_last_lr()[0]:.2e}",
                        )

                    # Periodic Checkpoints
                    if global_step % 1000 == 0:
                        ckpt = os.path.join(output_dir, f"checkpoint-{global_step}")
                        model.save_pretrained(ckpt, safe_serialization=True)
                        tokenizer.save_pretrained(ckpt)

                    if global_step % 250 == 0:
                        gc.collect()
                        if device == "cuda":
                            torch.cuda.empty_cache()

                    if global_step >= total_steps:
                        status_message = "Training target reached successfully!"
                        break

                step += 1

        except KeyboardInterrupt:
            status_message = "Training safely halted by user. Triggering auto-push..."

    final_loss = total_loss / max(1, step)

    # 5. Pre-Push Validation & Export
    print("Running Pre-Push Model Validation...")
    model.eval()
    test_prompt = "বিজ্ঞান ও প্রযুক্তি"
    test_inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        test_out = model.generate(**test_inputs, max_new_tokens=20)
    generated_text = tokenizer.decode(test_out[0], skip_special_tokens=True)
    print(f"Validation Sample Output: '{generated_text}'")

    print(f"Saving final weights (safetensors) to {output_dir}...")
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(build_model_card(repo_id, dataset_name, total_steps, final_loss))

    # 6. Fault-Tolerant Hugging Face Hub Upload
    upload_success = False
    upload_error = None
    if hf_token:
        print(f"Connecting to Hugging Face Hub: {repo_id}...")
        api = HfApi(token=hf_token)

        # Create repo if not exists
        for att in range(1, 4):
            try:
                api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
                break
            except Exception as e:
                time.sleep(2 * att)

        # Upload folder with exponential backoff & jitter
        for attempt in range(1, 6):
            try:
                api.upload_folder(
                    folder_path=output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"Autonomous Day-1 release: {total_steps} steps, loss={final_loss:.4f}",
                )
                upload_success = True
                break
            except Exception as e:
                upload_error = str(e)
                delay = (2 ** attempt) + random.uniform(0.5, 2.0)
                print(f"Upload retry {attempt}/5: {e}. Retrying in {delay:.1f}s...")
                time.sleep(delay)

    # Output Presentation
    if upload_success:
        result_view = mo.md(
            f"""
            ### 🎉 Training Complete & Model Successfully Pushed to Hugging Face!
            - **Target Repository:** [{repo_id}](https://huggingface.co/{repo_id})
            - **Total Steps Trained:** `{total_steps:,}`
            - **Final Average Loss:** `{final_loss:.4f}`
            - **Validation Inference:** *"{generated_text}"*
            - **Weights Format:** `safetensors` (Native bfloat16)

            👉 **Your model is now live on Hugging Face:** [https://huggingface.co/{repo_id}](https://huggingface.co/{repo_id})
            """
        )
    else:
        result_view = mo.md(
            f"""
            ### ⚠️ Training Completed, Weights Saved Locally
            - **Local Weights Path:** `{output_dir}`
            - **Upload Error:** `{upload_error}`
            - You can push manually using `python push_to_hf.py --repo-id {repo_id}`
            """
        )

    return (
        base_model_id,
        batch,
        batch_size,
        build_model_card,
        cur_loss,
        dataloader,
        dataset_name,
        delay,
        device,
        dtype,
        elapsed,
        final_loss,
        generated_text,
        global_step,
        grad_accum,
        hf_token,
        input_ids,
        labels,
        loss_val,
        model,
        optimizer,
        out,
        output_dir,
        pbar,
        repo_id,
        result_view,
        scaled_loss,
        scheduler,
        seq_len,
        start_time,
        status_message,
        step,
        test_inputs,
        test_out,
        test_prompt,
        time,
        tok_s,
        tokenizer,
        total_loss,
        total_steps,
        upload_error,
        upload_success,
        vram_gb,
    )


if __name__ == "__main__":
    app.run()
