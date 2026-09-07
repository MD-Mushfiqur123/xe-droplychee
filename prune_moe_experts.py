# prune_moe_experts.py
# Profiles MoE router activation frequencies, identifies underperforming experts,
# and prunes them to shrink the model size.

import torch
import torch.nn as nn
from collections import defaultdict
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

def profile_expert_usage(model, dataloader, num_layers, num_experts, device='cuda'):
    print('Hooking into MoE router layers...')
    expert_counts = {layer_idx: torch.zeros(num_experts, device=device) for layer_idx in range(num_layers)}

    hooks = []
    def get_gate_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                indices = output[1]
            else:
                indices = output
            if indices is not None:
                for idx in indices.view(-1):
                    if 0 <= idx < num_experts:
                        expert_counts[layer_idx][idx] += 1
        return hook

    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
            hooks.append(layer.mlp.gate.register_forward_hook(get_gate_hook(layer_idx)))

    print('Profiling forward passes...')
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            model(input_ids)
            if batch_idx >= 50:
                break

    for h in hooks:
        h.remove()
    return expert_counts

def select_top_experts(expert_counts, keep_n_experts=16):
    keep_indices = {}
    print(f'Pruning Selection: Keeping Top {keep_n_experts} Active Experts...')
    for layer_idx, counts in expert_counts.items():
        total_calls = counts.sum().item()
        if total_calls == 0:
            top_indices = list(range(keep_n_experts))
        else:
            sorted_indices = torch.argsort(counts, descending=True).tolist()
            top_indices = sorted_indices[:keep_n_experts]
            dropped = sorted_indices[keep_n_experts:]
            print(f'Layer {layer_idx:02d}: Keeping {len(top_indices)} | Dropping {len(dropped)} inactive experts')
        keep_indices[layer_idx] = top_indices
    return keep_indices
