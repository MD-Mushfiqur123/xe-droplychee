from typing import Optional
from transformers.configuration_utils import PretrainedConfig

class XeDroplycheeConfig(PretrainedConfig):
    model_type = 'xe_droplychee'
    keys_to_ignore_at_inference = ['past_key_values']

    auto_map = {
        'AutoConfig': 'configuration_droplychee.XeDroplycheeConfig',
        'AutoModel': 'modeling_droplychee.XeDroplycheeModel',
        'AutoModelForCausalLM': 'modeling_droplychee.XeDroplycheeForCausalLM',
    }

    def __init__(
        self,
        # Scaled Model Dimensions (Optimized 1/10th scale for 94GB/96GB VRAM)
        vocab_size: int = 49152,
        hidden_size: int = 2240,
        intermediate_size: int = 4480,
        moe_intermediate_size: int = 768,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 32,
        num_key_value_heads: Optional[int] = None,

        # MLA Specifications + Improved QK-Norm
        q_lora_rank: int = 512,
        kv_lora_rank: int = 256,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        use_qk_norm: bool = True,            # IMPROVEMENT 1: Stabilizes attention logits

        # Fine-Grained MoE with Dynamic Scheduling
        n_routed_experts: int = 64,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 4,
        first_k_dense_replace: int = 1,
        norm_topk_prob: bool = True,
        routing_func: str = 'sigmoid',
        bias_gamma: float = 0.001,
        aux_loss_alpha: float = 0.0001,

        # FlashAttention & Precision Optimizations
        use_sdpa: bool = True,               # IMPROVEMENT 2: Fast Fused Scaled Dot Product Attention
        max_position_embeddings: int = 8192,
        rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,

        # DeepSeek-V3/V4 Multi-Token Prediction (MTP)
        use_mtp: bool = True,
        mtp_depth: int = 1,
        mtp_loss_factor: float = 0.3,

        initializer_range: float = 0.006,
        tie_word_embeddings: bool = True,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: Optional[int] = None,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.use_qk_norm = use_qk_norm

        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.routing_func = routing_func
        self.bias_gamma = bias_gamma
        self.aux_loss_alpha = aux_loss_alpha

        self.use_sdpa = use_sdpa
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.attention_dropout = attention_dropout

        self.use_mtp = use_mtp
        self.mtp_depth = mtp_depth
        self.mtp_loss_factor = mtp_loss_factor
        self.initializer_range = initializer_range

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            is_decoder=True,
            **kwargs
        )

DroplycheeConfigV4 = XeDroplycheeConfig
DroplycheeConfig = XeDroplycheeConfig
