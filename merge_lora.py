# merge_lora.py
import os, sys, gc, json, shutil, argparse, torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description='Merge LoRA weights back into base model and export safetensors')
    parser.add_argument('--base_model_path', '-b', type=str, default='./hf_model', help='Path to base model directory')
    parser.add_argument('--adapter_path', '-a', type=str, required=True, help='Path to trained LoRA adapter directory')
    parser.add_argument('--output_dir', '-o', type=str, default='./merged_model', help='Target output directory')
    parser.add_argument('--torch_dtype', '-d', type=str, default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--push_to_hub', action='store_true', help='Upload merged model to Hugging Face Hub')
    parser.add_argument('--hub_repo_id', type=str, default=None, help='Hugging Face repo id')
    parser.add_argument('--hub_token', type=str, default=os.getenv('HF_TOKEN', ''))
    return parser.parse_args()

def main():
    args = parse_args()
    dtype_map = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}
    target_dtype = dtype_map[args.torch_dtype]

    print(f'Loading base model: {args.base_model_path}...')
    try:
        from peft import PeftModel
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=target_dtype,
            device_map=args.device,
            trust_remote_code=True
        )
        print(f'Attaching adapter: {args.adapter_path}...')
        peft_model = PeftModel.from_pretrained(base_model, args.adapter_path)
        print('Merging LoRA weights in-place...')
        model = peft_model.merge_and_unload()
    except Exception as e:
        print(f'Direct load fallback: {e}')
        model = AutoModelForCausalLM.from_pretrained(args.adapter_path, torch_dtype=target_dtype, trust_remote_code=True)

    print(f'Exporting merged model with safetensors to {args.output_dir}...')
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_dir)

    # Copy code files if present
    for f in ['configuration_droplychee.py', 'modeling_droplychee.py', '__init__.py']:
        src = os.path.join(args.base_model_path, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.output_dir, f))

    if args.push_to_hub and args.hub_repo_id:
        from huggingface_hub import HfApi
        print(f'Pushing to Hugging Face Hub: {args.hub_repo_id}...')
        api = HfApi(token=args.hub_token)
        api.create_repo(repo_id=args.hub_repo_id, exist_ok=True)
        api.upload_folder(folder_path=args.output_dir, repo_id=args.hub_repo_id, repo_type='model')
        print(f'Successfully pushed to https://huggingface.co/{args.hub_repo_id}')

if __name__ == '__main__':
    main()
