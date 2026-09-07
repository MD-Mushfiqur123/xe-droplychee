import os
import shutil
import json
import torch
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

from m_droplychee import DroplycheeConfigV4, DroplycheeForCausalLMV4

def build_fast_tokenizer(vocab_size: int = 49152) -> PreTrainedTokenizerFast:
    special_tokens = [
        '<unk>', '<s>', '</s>', '<pad>',
        '<think>', '</think>', '<answer>', '</answer>',
        '<｜fim_begin｜>', '<｜fim_hole｜>', '<｜fim_end｜>'
    ]
    vocab = {tok: idx for idx, tok in enumerate(special_tokens)}
    for b in range(256):
        char = chr(b)
        if char not in vocab:
            vocab[char] = len(vocab)
    for i in range(len(vocab), vocab_size):
        vocab[f'<token_{i}>'] = i

    raw_tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token='<unk>'))
    raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw_tokenizer.decoder = decoders.ByteLevel()
    raw_tokenizer.add_special_tokens(special_tokens)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        bos_token='<s>',
        eos_token='</s>',
        unk_token='<unk>',
        pad_token='<pad>',
        clean_up_tokenization_spaces=False,
        model_max_length=8192
    )
    return tokenizer

def export():
    out_dir = Path('./hf_model')
    out_dir.mkdir(parents=True, exist_ok=True)
    print('=== Exporting m-droplychee v4 to Hugging Face format ===')

    print('[1/5] Initializing Model & Tokenizer...')
    config = DroplycheeConfigV4()
    tokenizer = build_fast_tokenizer(vocab_size=config.vocab_size)

    # Save tokenizer
    print('[2/5] Saving Tokenizer to hf_model/...')
    tokenizer.save_pretrained(out_dir)

    # Copy architecture files for trust_remote_code
    print('[3/5] Copying architecture modules for dynamic remote execution...')
    shutil.copy('m_droplychee/configuration_droplychee.py', out_dir / 'configuration_droplychee.py')
    shutil.copy('m_droplychee/modeling_droplychee.py', out_dir / 'modeling_droplychee.py')
    shutil.copy('m_droplychee/__init__.py', out_dir / '__init__.py')

    # Config JSON
    print('[4/5] Generating config.json with auto_map...')
    cfg_dict = config.to_dict()
    cfg_dict['model_type'] = 'droplychee'
    cfg_dict['architectures'] = ['DroplycheeForCausalLMV4']
    cfg_dict['auto_map'] = {
        'AutoConfig': 'configuration_droplychee.DroplycheeConfigV4',
        'AutoModelForCausalLM': 'modeling_droplychee.DroplycheeForCausalLMV4'
    }
    with open(out_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(cfg_dict, f, indent=2)

    # Generation Config JSON
    gen_config = {
        'bos_token_id': 1,
        'eos_token_id': 2,
        'pad_token_id': 3,
        'do_sample': True,
        'temperature': 0.7,
        'top_p': 0.95,
        'top_k': 50,
        'max_new_tokens': 4096
    }
    with open(out_dir / 'generation_config.json', 'w', encoding='utf-8') as f:
        json.dump(gen_config, f, indent=2)

    print('[5/5] Exporting Model Card README.md...')
    readme_path = out_dir / 'README.md'
    if not readme_path.exists():
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write('# m-droplychee v4\n\nScaled 1:1 DeepSeek-V3/V4 Architecture Replica.\n')

    print('=== Hugging Face Model Export Complete at ./hf_model ===')

if __name__ == '__main__':
    export()
