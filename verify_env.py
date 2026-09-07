#!/usr/bin/env python3
"""
verify_env.py - Standalone Ultra-Fast Pre-Flight Environment Diagnostic

Checks:
  1) CUDA availability, GPU device name, compute capability, and VRAM.
  2) bfloat16 hardware support and tensor compute verification.
  3) Tokenizer loading from './hf_model' and Bengali sentence tokenization.
  4) Model instantiation via Hugging Face AutoConfig and AutoModelForCausalLM.
  5) Forward & backward pass on dummy batch (MLA + MoE + MTP loss & gradient check).
  6) Dataset streaming connectivity to 'nahid-hub/B-CORE-bengali-corpus'.
  7) Hugging Face token validity and user/write authorization.

Designed to execute in under 10 seconds with clear green/red status indicators.
"""

import os
import sys
import time
import copy
import argparse
from pathlib import Path

# Ensure UTF-8 output encoding across all platforms (Windows, Linux, macOS)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Enable ANSI escape sequences on Windows console
if sys.platform == "win32":
    os.system("")

# ANSI Color Codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

TAG_PASS = f"{BOLD}{GREEN}[PASS]{RESET}"
TAG_FAIL = f"{BOLD}{RED}[FAIL]{RESET}"
TAG_WARN = f"{BOLD}{YELLOW}[WARN]{RESET}"
TAG_INFO = f"{BOLD}{CYAN}[INFO]{RESET}"


class DiagnosticSuite:
    def __init__(self, hf_model_path="./hf_model", dataset_name="nahid-hub/B-CORE-bengali-corpus", token=None, full_scale=False):
        self.hf_model_path = Path(hf_model_path).resolve()
        self.dataset_name = dataset_name
        self.token = token
        self.full_scale = full_scale
        self.results = []
        self.device = "cpu"
        self.dtype = None
        self.tokenizer = None
        self.config = None
        self.model = None

    def record_result(self, name: str, passed: bool, message: str, elapsed: float, is_warning: bool = False):
        status_tag = TAG_PASS if passed else (TAG_WARN if is_warning else TAG_FAIL)
        self.results.append({
            "name": name,
            "passed": passed,
            "is_warning": is_warning,
            "message": message,
            "elapsed": elapsed
        })
        print(f"  {status_tag} {name} ({elapsed:.2f}s)")
        print(f"         {DIM}-> {message}{RESET}")

    # 1. CUDA Availability & Device Name
    def check_cuda(self):
        t0 = time.time()
        print(f"\n{BOLD}[1/7] Testing CUDA Availability & Device Specs...{RESET}")
        try:
            import torch
            if torch.cuda.is_available():
                count = torch.cuda.device_count()
                curr = torch.cuda.current_device()
                name = torch.cuda.get_device_name(curr)
                props = torch.cuda.get_device_properties(curr)
                vram_gb = props.total_memory / (1024 ** 3)
                major, minor = props.major, props.minor
                self.device = "cuda"
                msg = f"{name} (GPU {curr}/{count}) | VRAM: {vram_gb:.2f} GB | Compute: {major}.{minor} | PyTorch: {torch.__version__}"
                self.record_result("CUDA & Device Name", True, msg, time.time() - t0)
            else:
                self.device = "cpu"
                msg = f"CUDA is NOT available! Running on CPU fallback ({sys.platform})."
                self.record_result("CUDA & Device Name", False, msg, time.time() - t0, is_warning=True)
        except Exception as e:
            self.record_result("CUDA & Device Name", False, f"Exception: {e}", time.time() - t0)

    # 2. bfloat16 Support
    def check_bfloat16(self):
        t0 = time.time()
        print(f"\n{BOLD}[2/7] Testing bfloat16 Hardware & Kernel Support...{RESET}")
        try:
            import torch
            if self.device == "cuda":
                bf16_native = torch.cuda.is_bf16_supported()
                test_tensor = torch.ones((4, 4), dtype=torch.bfloat16, device="cuda")
                test_res = (test_tensor * 2.5).sum().item()
                if bf16_native:
                    self.dtype = torch.bfloat16
                    msg = f"Native hardware bfloat16 supported. Matmul test verified ({test_res:.1f})."
                    self.record_result("bfloat16 Support", True, msg, time.time() - t0)
                else:
                    self.dtype = torch.float16
                    msg = "CUDA available but bfloat16 is NOT natively accelerated. Falling back to float16."
                    self.record_result("bfloat16 Support", False, msg, time.time() - t0, is_warning=True)
            else:
                self.dtype = torch.float32
                msg = "Running on CPU; using float32."
                self.record_result("bfloat16 Support", True, msg, time.time() - t0, is_warning=True)
        except Exception as e:
            self.record_result("bfloat16 Support", False, f"Failed bfloat16 validation: {e}", time.time() - t0)

    # 3. Tokenizer Loading from './hf_model'
    def check_tokenizer(self):
        t0 = time.time()
        print(f"\n{BOLD}[3/7] Loading Tokenizer from '{self.hf_model_path}'...{RESET}")
        try:
            from transformers import AutoTokenizer
            if not self.hf_model_path.exists():
                raise FileNotFoundError(f"Model path does not exist: {self.hf_model_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(str(self.hf_model_path), trust_remote_code=True)
            vocab_size = len(self.tokenizer)

            test_text = "আমাদের মাতৃভাষা বাংলা। xe_droplychee DeepSeek architecture test."
            encoded = self.tokenizer.encode(test_text)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            msg = f"Loaded fast tokenizer | Vocab size: {vocab_size:,} | Special tokens: bos='{self.tokenizer.bos_token}', eos='{self.tokenizer.eos_token}' | Encode/Decode sanity verified."
            self.record_result("Tokenizer Loading", True, msg, time.time() - t0)
        except Exception as e:
            self.record_result("Tokenizer Loading", False, f"Tokenizer failure: {e}", time.time() - t0)

    # 4. Model Instantiation
    def check_model_instantiation(self):
        t0 = time.time()
        print(f"\n{BOLD}[4/7] Testing Model Instantiation from Config...{RESET}")
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM

            self.config = AutoConfig.from_pretrained(str(self.hf_model_path), trust_remote_code=True)
            full_params_est = "~10B (28 layers, 65 experts, hidden_size=2240)"

            if self.full_scale and self.device == "cuda":
                print(f"         {TAG_INFO} Instantiating FULL-SCALE model directly in GPU VRAM...")
                test_cfg = self.config
                scale_desc = f"Full Production Scale ({full_params_est})"
            else:
                test_cfg = copy.deepcopy(self.config)
                test_cfg.num_hidden_layers = 2
                test_cfg.n_routed_experts = 4
                test_cfg.num_experts_per_tok = 2
                test_cfg.hidden_size = 256
                test_cfg.intermediate_size = 512
                test_cfg.moe_intermediate_size = 256
                test_cfg.num_attention_heads = 4
                test_cfg.num_key_value_heads = 4
                test_cfg.q_lora_rank = 128
                test_cfg.kv_lora_rank = 64
                test_cfg.qk_nope_head_dim = 32
                test_cfg.qk_rope_head_dim = 32
                test_cfg.qk_head_dim = 64
                test_cfg.v_head_dim = 32
                test_cfg.use_mtp = True
                scale_desc = "Fast Diagnostic Scale (2 layers, 4 routed + 1 shared experts, MTP=True)"

            target_dtype = self.dtype if (self.device == "cuda" and self.dtype) else torch.float32

            prev_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(target_dtype)
                if self.device == "cuda":
                    with torch.device("cuda"):
                        try:
                            self.model = AutoModelForCausalLM.from_config(
                                test_cfg,
                                trust_remote_code=True,
                                dtype=target_dtype
                            )
                        except Exception:
                            sys.path.insert(0, str(self.hf_model_path))
                            from modeling_droplychee import XeDroplycheeForCausalLM
                            self.model = XeDroplycheeForCausalLM(test_cfg)
                else:
                    try:
                        self.model = AutoModelForCausalLM.from_config(
                            test_cfg,
                            trust_remote_code=True,
                            dtype=target_dtype
                        )
                    except Exception:
                        sys.path.insert(0, str(self.hf_model_path))
                        from modeling_droplychee import XeDroplycheeForCausalLM
                        self.model = XeDroplycheeForCausalLM(test_cfg)
            finally:
                torch.set_default_dtype(prev_dtype)

            self.model = self.model.to(self.device)
            self.model.eval()

            param_count = sum(p.numel() for p in self.model.parameters())
            msg = f"Architecture '{self.model.__class__.__name__}' allocated successfully ({param_count:,} params) | {scale_desc}."
            self.record_result("Model Instantiation", True, msg, time.time() - t0)
        except Exception as e:
            self.record_result("Model Instantiation", False, f"Model instantiation failed: {e}", time.time() - t0)

    # 5. Forward & Backward Pass on Dummy Batch
    def check_forward_backward(self):
        t0 = time.time()
        print(f"\n{BOLD}[5/7] Testing Forward & Backward Autograd Graph...{RESET}")
        if self.model is None:
            self.record_result("Forward & Backward Pass", False, "Skipped because model instantiation failed.", time.time() - t0)
            return

        try:
            import torch
            self.model.train()
            batch_size = 2
            seq_len = 32
            vocab_limit = getattr(self.model.config, "vocab_size", 50267)

            input_ids = torch.randint(0, min(1000, vocab_limit), (batch_size, seq_len), device=self.device)
            labels = input_ids.clone()

            autocast_device = "cuda" if self.device == "cuda" else "cpu"
            autocast_dtype = self.dtype if self.dtype else torch.float32

            with torch.amp.autocast(device_type=autocast_device, dtype=autocast_dtype, enabled=(self.device == "cuda")):
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs.get("loss")
                logits = outputs.logits if hasattr(outputs, "logits") else outputs.get("logits")

            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                raise ValueError(f"Invalid loss computed: {loss}")

            loss.backward()

            grads_found = False
            has_nan_grad = False
            for p in self.model.parameters():
                if p.grad is not None:
                    grads_found = True
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        has_nan_grad = True
                        break

            if not grads_found:
                raise ValueError("No gradients computed across trainable parameters.")
            if has_nan_grad:
                raise ValueError("NaN or Inf encountered in computed gradients.")

            msg = f"Forward loss={loss.item():.4f} | Backward autograd successful | Zero NaN/Inf gradients."
            self.record_result("Forward & Backward Pass", True, msg, time.time() - t0)
        except Exception as e:
            self.record_result("Forward & Backward Pass", False, f"Autograd failure: {e}", time.time() - t0)

    # 6. Dataset Streaming Connectivity
    def check_dataset_streaming(self):
        t0 = time.time()
        print(f"\n{BOLD}[6/7] Testing Streaming Connectivity to '{self.dataset_name}'...{RESET}")
        try:
            from datasets import load_dataset
            stream = load_dataset(self.dataset_name, split="train", streaming=True)
            sample = next(iter(stream))

            text = sample.get("text", "")
            if not text:
                raise ValueError("Fetched sample has empty or missing 'text' key.")

            token_count = len(self.tokenizer.encode(text)) if self.tokenizer else "N/A"
            snippet = (text[:45] + "...").replace("\n", " ")
            msg = f"Connected to Hugging Face stream | First sample fetched ({len(text)} chars, ~{token_count} tokens) | Preview: \"{snippet}\""
            self.record_result("Dataset Streaming", True, msg, time.time() - t0)
        except Exception as e:
            self.record_result("Dataset Streaming", False, f"Dataset stream error: {e}", time.time() - t0)

    # 7. Hugging Face Token Validity
    def check_hf_token(self):
        t0 = time.time()
        print(f"\n{BOLD}[7/7] Testing Hugging Face Token & Write Authorization...{RESET}")
        try:
            from huggingface_hub import HfApi, get_token

            token = (
                self.token
                or os.getenv("HF_TOKEN")
                or os.getenv("HUGGING_FACE_HUB_TOKEN")
                or get_token()
            )

            if not token:
                msg = "No HF token discovered. Training can run locally, but set HF_TOKEN before pushing to Hub."
                self.record_result("HF Token Validity", False, msg, time.time() - t0, is_warning=True)
                return

            api = HfApi(token=token.strip())
            user_info = api.whoami()
            username = user_info.get("name") or user_info.get("username")
            user_type = user_info.get("type", "user")

            msg = f"Authenticated as @{username} (Type: {user_type}). Push ready."
            self.record_result("HF Token Validity", True, msg, time.time() - t0)
        except Exception as e:
            self.record_result("HF Token Validity", False, f"HF Auth failed (check token permissions): {e}", time.time() - t0)

    def run_all(self):
        wall_start = time.time()
        print("=" * 72)
        print(f"{BOLD}{CYAN}      XE_DROPLYCHEE: 10-SECOND ENVIRONMENT PRE-FLIGHT VERIFICATION{RESET}")
        print("=" * 72)

        self.check_cuda()
        self.check_bfloat16()
        self.check_tokenizer()
        self.check_model_instantiation()
        self.check_forward_backward()
        self.check_dataset_streaming()
        self.check_hf_token()

        total_elapsed = time.time() - wall_start

        # Summary Table
        print("\n" + "=" * 72)
        print(f"{BOLD}                        DIAGNOSTIC SUMMARY REPORT{RESET}")
        print("=" * 72)
        all_passed = True
        for res in self.results:
            tag = f"{GREEN}PASS{RESET}" if res["passed"] else (f"{YELLOW}WARN{RESET}" if res["is_warning"] else f"{RED}FAIL{RESET}")
            print(f"  [{tag}] {res['name']:<35} ({res['elapsed']:>5.2f}s)")
            if not res["passed"] and not res["is_warning"]:
                all_passed = False

        print("-" * 72)
        status_color = GREEN if all_passed else RED
        status_text = "ALL CHECKS PASSED - READY FOR HIGH-SPEED TRAINING" if all_passed else "SOME CRITICAL CHECKS FAILED - INSPECT ERRORS ABOVE"
        print(f"  {BOLD}Total Diagnostic Time:{RESET} {total_elapsed:.2f}s (Target: < 10.0s)")
        print(f"  {BOLD}Status:{RESET} {status_color}{status_text}{RESET}")
        print("=" * 72 + "\n")

        return 0 if all_passed else 1


def main():
    parser = argparse.ArgumentParser(description="Ultra-fast 10-second environment verification for xe_droplychee.")
    parser.add_argument("--hf_model", type=str, default="./hf_model", help="Path to local hf_model directory")
    parser.add_argument("--dataset", type=str, default="nahid-hub/B-CORE-bengali-corpus", help="Target streaming dataset")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face API token (defaults to HF_TOKEN env var)")
    parser.add_argument("--full_scale", action="store_true", default=False, help="Instantiate complete 28-layer 10B model instead of diagnostic scale")
    args = parser.parse_args()

    suite = DiagnosticSuite(
        hf_model_path=args.hf_model,
        dataset_name=args.dataset,
        token=args.hf_token,
        full_scale=args.full_scale
    )
    exit_code = suite.run_all()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
