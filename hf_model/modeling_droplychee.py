import math
from typing import Optional, Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel, AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

try:
    from .configuration_droplychee import XeDroplycheeConfig
except ImportError:
    from configuration_droplychee import XeDroplycheeConfig


class XeRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class XeRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = (cos.unsqueeze(1) if x.ndim == 4 else cos).to(dtype=x.dtype)
    sin = (sin.unsqueeze(1) if x.ndim == 4 else sin).to(dtype=x.dtype)
    return (x * cos) + (rotate_half(x) * sin)


class XeMLA(nn.Module):
    """
    Enhanced Multi-Head Latent Attention with QK-Norm and SDPA Acceleration.
    """
    def __init__(self, config: XeDroplycheeConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_head_dim
        self.v_head_dim = config.v_head_dim
        self.use_sdpa = getattr(config, 'use_sdpa', True)
        self.use_qk_norm = getattr(config, 'use_qk_norm', True)

        # 1. Query Projections
        if self.q_lora_rank > 0:
            self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
            self.q_a_layernorm = XeRMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_nope_head_dim, bias=False)
            self.q_rope_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_rope_head_dim, bias=False)
        else:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.qk_head_dim, bias=False)

        # 2. Key-Value Projections
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = XeRMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False
        )

        # 3. Headwise QK-Norm for High-Scale Training Stability
        if self.use_qk_norm:
            self.q_norm = XeRMSNorm(self.qk_head_dim, eps=config.rms_norm_eps)
            self.k_norm = XeRMSNorm(self.qk_head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)
        self.rotary_emb = XeRotaryEmbedding(
            self.qk_rope_head_dim, max_position_embeddings=config.max_position_embeddings, base=config.rope_theta
        )
        self.softmax_scale = 1.0 / math.sqrt(self.qk_head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape

        # Query path
        if self.q_lora_rank > 0:
            q_latent = self.q_a_proj(hidden_states)
            q_latent = self.q_a_layernorm(q_latent)
            q_nope = self.q_b_proj(q_latent).view(bsz, q_len, self.num_heads, self.qk_nope_head_dim)
            q_pe = self.q_rope_proj(q_latent).view(bsz, q_len, self.num_heads, self.qk_rope_head_dim)
        else:
            q_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.qk_head_dim)
            q_nope = q_states[..., : self.qk_nope_head_dim]
            q_pe = q_states[..., self.qk_nope_head_dim :]

        # Compressed KV path
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        c_kv = compressed_kv[..., : self.kv_lora_rank]
        k_pe = compressed_kv[..., self.kv_lora_rank :]
        c_kv_normed = self.kv_a_layernorm(c_kv)

        kv_seq_len = q_len if past_key_values is None else past_key_values[0].shape[1] + q_len
        cos, sin = self.rotary_emb(k_pe, kv_seq_len)
        cos = cos[-q_len:]
        sin = sin[-q_len:]

        q_pe = apply_rotary_pos_emb(q_pe, cos, sin)
        k_pe = apply_rotary_pos_emb(k_pe, cos, sin)

        # Cache only c_kv and k_pe
        if past_key_values is not None:
            past_c_kv, past_k_pe = past_key_values
            c_kv_normed = torch.cat([past_c_kv, c_kv_normed], dim=1)
            k_pe = torch.cat([past_k_pe, k_pe], dim=1)

        current_key_values = (c_kv_normed, k_pe) if use_cache else None

        kv_states = self.kv_b_proj(c_kv_normed)
        kv_states = kv_states.view(bsz, -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope = kv_states[..., : self.qk_nope_head_dim]
        v_states = kv_states[..., self.qk_nope_head_dim :]

        k_pe_expanded = k_pe.unsqueeze(2).expand(-1, -1, self.num_heads, -1)

        # QK Assembly & QK-Norm
        query_states = torch.cat([q_nope, q_pe], dim=-1)
        key_states = torch.cat([k_nope, k_pe_expanded], dim=-1)

        query_states = self.q_norm(query_states).transpose(1, 2)  # [bsz, heads, q_len, dim]
        key_states = self.k_norm(key_states).transpose(1, 2)      # [bsz, heads, kv_len, dim]
        value_states = v_states.transpose(1, 2)                   # [bsz, heads, kv_len, v_dim]

        # Fast Scaled Dot-Product Attention (SDPA)
        if self.use_sdpa and attention_mask is None:
            is_causal = (q_len > 1 and past_key_values is None)
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                is_causal=is_causal,
                dropout_p=self.config.attention_dropout if self.training else 0.0,
                scale=self.softmax_scale
            )
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.softmax_scale
            if attention_mask is not None:
                if attention_mask.ndim == 2:
                    expanded_mask = (1.0 - attention_mask[:, None, None, :].to(attn_weights.dtype)) * torch.finfo(attn_weights.dtype).min
                    attn_weights = attn_weights + expanded_mask
                else:
                    attn_weights = attn_weights + attention_mask
            if q_len > 1 and past_key_values is None:
                causal_mask = torch.triu(torch.full((q_len, q_len), torch.finfo(attn_weights.dtype).min, device=hidden_states.device), diagonal=1)
                attn_weights = attn_weights + causal_mask

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = F.dropout(attn_weights, p=self.config.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, current_key_values


class XeMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class XeMoE(nn.Module):
    """
    Fine-Grained MoE with Dynamic Bias Decay Scheduler.
    """
    def __init__(self, config: XeDroplycheeConfig):
        super().__init__()
        self.config = config
        self.num_routed_experts = config.n_routed_experts
        self.num_shared_experts = config.n_shared_experts
        self.top_k = config.num_experts_per_tok
        self.gamma = config.bias_gamma
        self.aux_loss_alpha = config.aux_loss_alpha

        if self.num_shared_experts > 0:
            shared_dim = self.num_shared_experts * config.moe_intermediate_size
            self.shared_experts = XeMLP(config.hidden_size, shared_dim)
        else:
            self.shared_experts = None

        self.experts = nn.ModuleList([
            XeMLP(config.hidden_size, config.moe_intermediate_size)
            for _ in range(self.num_routed_experts)
        ])

        self.gate = nn.Linear(config.hidden_size, self.num_routed_experts, bias=False)
        self.register_buffer("expert_biases", torch.zeros(self.num_routed_experts))

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        orig_shape = hidden_states.shape
        x_flat = hidden_states.view(-1, orig_shape[-1])
        num_tokens = x_flat.shape[0]

        shared_output = self.shared_experts(x_flat).to(dtype=x_flat.dtype) if self.shared_experts is not None else torch.zeros_like(x_flat)

        router_logits = self.gate(x_flat)
        affinity_scores = torch.sigmoid(router_logits)

        # Dynamic Bias Top-K Selection (ensure consistent dtype to avoid unwanted scalar upcasting)
        scores_for_selection = affinity_scores + self.expert_biases.to(dtype=affinity_scores.dtype)
        _, topk_indices = torch.topk(scores_for_selection, self.top_k, dim=-1)

        selected_scores = torch.gather(affinity_scores, dim=-1, index=topk_indices)
        weights = (selected_scores / (selected_scores.sum(dim=-1, keepdim=True) + 1e-9)).to(dtype=x_flat.dtype)

        if self.training:
            with torch.no_grad():
                mask = torch.zeros_like(affinity_scores).scatter_(-1, topk_indices, 1.0)
                tokens_per_expert = mask.sum(dim=0)
                ideal_load = (self.top_k * num_tokens) / self.num_routed_experts
                if self.gamma > 0 and torch.is_grad_enabled():
                    delta = torch.zeros_like(self.expert_biases)
                    delta[tokens_per_expert > ideal_load] -= self.gamma
                    delta[tokens_per_expert < ideal_load] += self.gamma
                    new_biases = (self.expert_biases + delta)
                    new_biases = (new_biases - new_biases.mean()).clamp(-0.5, 0.5)
                    self.expert_biases.copy_(new_biases)

            normed_affinity = affinity_scores / (affinity_scores.sum(dim=-1, keepdim=True) + 1e-6)
            f_i = (self.num_routed_experts / (self.top_k * num_tokens)) * tokens_per_expert
            p_i = normed_affinity.mean(dim=0)
            aux_loss = self.aux_loss_alpha * (f_i * p_i).sum()
        else:
            aux_loss = torch.zeros((), device=hidden_states.device, dtype=hidden_states.dtype)

        # Expert dispatch without token_mask.any() CPU-GPU synchronization and strict dtype safety
        routed_output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            token_mask = (topk_indices == i)
            token_idx, k_pos = torch.where(token_mask)
            if token_idx.numel() > 0:
                expert_weights = weights[token_idx, k_pos].unsqueeze(-1).to(dtype=routed_output.dtype)
                expert_out = expert(x_flat[token_idx]).to(dtype=routed_output.dtype)
                source = (expert_weights * expert_out).to(dtype=routed_output.dtype)
                routed_output.index_add_(0, token_idx, source)

        final_output = (shared_output + routed_output).to(dtype=x_flat.dtype).view(orig_shape)
        return final_output, aux_loss


class XeDecoderLayer(nn.Module):
    def __init__(self, config: XeDroplycheeConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = XeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = XeMLA(config)
        self.post_attention_layernorm = XeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if layer_idx < config.first_k_dense_replace:
            self.mlp = XeMLP(config.hidden_size, config.intermediate_size)
            self.is_moe = False
        else:
            self.mlp = XeMoE(config)
            self.is_moe = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, current_kv = self.self_attn(
            hidden_states, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=use_cache
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.is_moe:
            hidden_states, aux_loss = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
            aux_loss = torch.tensor(0.0, device=hidden_states.device)
        hidden_states = residual + hidden_states

        return hidden_states, current_kv, aux_loss


class XeMTPModule(nn.Module):
    """Multi-Token Prediction (MTP) with dual RMSNorm projection."""
    def __init__(self, config: XeDroplycheeConfig):
        super().__init__()
        self.mtp_norm_h = XeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mtp_norm_emb = XeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.block = XeDecoderLayer(config, layer_idx=config.num_hidden_layers)
        self.final_norm = XeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, main_hidden: torch.Tensor, token_embeddings: torch.Tensor) -> torch.Tensor:
        h_norm = self.mtp_norm_h(main_hidden)
        emb_norm = self.mtp_norm_emb(token_embeddings)
        combined = torch.cat([h_norm, emb_norm], dim=-1)
        projected = self.proj(combined)
        out, _, _ = self.block(projected)
        return self.final_norm(out)


class XeDroplycheePreTrainedModel(PreTrainedModel):
    config_class = XeDroplycheeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["XeDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", 0.006)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, XeRMSNorm):
            module.weight.data.fill_(1.0)


class XeDroplycheeModel(XeDroplycheePreTrainedModel):
    def __init__(self, config: XeDroplycheeConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            XeDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)
        ])
        self.norm = XeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]], torch.Tensor]:
        hidden_states = self.embed_tokens(input_ids)
        total_aux_loss = torch.zeros((), device=input_ids.device, dtype=hidden_states.dtype)
        next_cache = [] if use_cache else None

        for idx, layer in enumerate(self.layers):
            layer_past = past_key_values[idx] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                hidden_states, current_kv, aux_loss = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    attention_mask,
                    layer_past,
                    use_cache,
                    use_reentrant=False,
                )
            else:
                hidden_states, current_kv, aux_loss = layer(
                    hidden_states, attention_mask=attention_mask, past_key_values=layer_past, use_cache=use_cache
                )
            total_aux_loss = total_aux_loss + aux_loss
            if use_cache:
                next_cache.append(current_kv)

        hidden_states = self.norm(hidden_states)
        return hidden_states, next_cache, total_aux_loss


class XeDroplycheeForCausalLM(XeDroplycheePreTrainedModel):
    def __init__(self, config: XeDroplycheeConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config
        self.model = XeDroplycheeModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        if config.use_mtp:
            self.mtp_head = XeMTPModule(config)
        else:
            self.mtp_head = None

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> dict:
        hidden_states, next_cache, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        logits = self.lm_head(hidden_states)
        loss = None

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            main_loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            loss = main_loss + aux_loss

            if self.mtp_head is not None and input_ids.shape[1] > 2:
                token_embs = self.model.embed_tokens(input_ids[:, 1:])
                mtp_hidden = self.mtp_head(hidden_states[:, :-1], token_embs)
                mtp_logits = self.lm_head(mtp_hidden)
                mtp_shift_logits = mtp_logits[..., :-1, :].contiguous()
                mtp_shift_labels = labels[..., 2:].contiguous()
                mtp_loss = F.cross_entropy(mtp_shift_logits.view(-1, self.config.vocab_size), mtp_shift_labels.view(-1))
                loss = loss + self.config.mtp_loss_factor * mtp_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=next_cache,
            hidden_states=None,
            attentions=None
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 0.7,
        top_k: int = 50,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        past_key_values = None
        generated = input_ids

        for _ in range(max_new_tokens):
            inputs = generated[:, -1:] if past_key_values is not None else generated
            outputs = self.forward(input_ids=inputs, past_key_values=past_key_values, use_cache=True)
            logits = outputs['logits'][:, -1, :]
            past_key_values = outputs['past_key_values']

            if temperature > 0:
                logits = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated

    def compute_grpo_loss(
        self,
        policy_logits: torch.Tensor,
        old_policy_logits: torch.Tensor,
        ref_policy_logits: torch.Tensor,
        action_tokens: torch.Tensor,
        advantages: torch.Tensor,
        epsilon: float = 0.2,
        beta: float = 0.04
    ) -> torch.Tensor:
        log_prob = F.log_softmax(policy_logits, dim=-1)
        old_log_prob = F.log_softmax(old_policy_logits, dim=-1)
        ref_log_prob = F.log_softmax(ref_policy_logits, dim=-1)

        token_log_prob = log_prob.gather(-1, action_tokens.unsqueeze(-1)).squeeze(-1)
        old_token_log_prob = old_log_prob.gather(-1, action_tokens.unsqueeze(-1)).squeeze(-1)
        ref_token_log_prob = ref_log_prob.gather(-1, action_tokens.unsqueeze(-1)).squeeze(-1)

        ratio = torch.exp(token_log_prob - old_token_log_prob)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        log_ratio = ref_token_log_prob - token_log_prob
        kl_div = torch.exp(log_ratio) - log_ratio - 1.0
        kl_loss = beta * kl_div.mean()

        return policy_loss + kl_loss


DroplycheeForCausalLMV4 = XeDroplycheeForCausalLM
DroplycheeModelV4 = XeDroplycheeModel
