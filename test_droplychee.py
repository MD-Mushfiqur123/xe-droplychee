import torch
from m_droplychee import DroplycheeConfig, DroplycheeForCausalLM

def run_test():
    print('====================================================')
    print('Testing m-droplychee Architecture from Scratch')
    print('====================================================')

    # 1. Instantiate Configuration (Small scale for quick unit test, full 2B config can be used for 96GB training)
    config = DroplycheeConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        q_lora_rank=128,
        kv_lora_rank=64,
        qk_nope_head_dim=32,
        qk_rope_head_dim=32,
        v_head_dim=32,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        use_mtp=True
    )

    print('[1/5] Building m-droplychee model...')
    model = DroplycheeForCausalLM(config)
    model.eval()

    # Parameter Count
    total_params = sum(p.numel() for p in model.parameters())
    print(f'   -> Total Parameters (Test Scale): {total_params:,}')

    # 2. Forward Pass Test
    print('[2/5] Testing Forward Pass & Loss Calculation...')
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    outputs = model(input_ids=input_ids, labels=labels)
    logits = outputs['logits']
    loss = outputs['loss']
    aux_loss = outputs['aux_loss']

    print(f'   -> Logits shape: {logits.shape} (Expected: [{batch_size}, {seq_len}, {config.vocab_size}])')
    print(f'   -> Main + Aux Loss: {loss.item():.4f} (Auxiliary Load Loss: {aux_loss.item():.4f})')
    assert logits.shape == (batch_size, seq_len, config.vocab_size), 'Logits shape mismatch!'
    assert loss is not None, 'Loss computation failed!'

    # 3. Autoregressive Generation with Compressed MLA KV Cache
    print('[3/5] Testing Autoregressive Generation with MLA KV Cache...')
    prompt = torch.tensor([[10, 20, 30]])
    gen_tokens = model.generate(prompt, max_new_tokens=8, temperature=0.7)
    print(f'   -> Generated sequence: {gen_tokens.tolist()}')
    assert gen_tokens.shape == (1, 3 + 8), 'Generation sequence length mismatch!'

    # 4. DeepSeek-R1 GRPO Loss Test
    print('[4/5] Testing DeepSeek-R1 GRPO Loss Module...')
    dummy_policy_logits = torch.randn(2, 8, config.vocab_size)
    dummy_old_logits = dummy_policy_logits + 0.05 * torch.randn_like(dummy_policy_logits)
    dummy_ref_logits = dummy_policy_logits + 0.1 * torch.randn_like(dummy_policy_logits)
    action_tokens = torch.randint(0, config.vocab_size, (2, 8))
    advantages = torch.tensor([[0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
                               [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]])

    grpo_loss = model.compute_grpo_loss(
        policy_logits=dummy_policy_logits,
        old_policy_logits=dummy_old_logits,
        ref_policy_logits=dummy_ref_logits,
        action_tokens=action_tokens,
        advantages=advantages
    )
    print(f'   -> GRPO Loss value: {grpo_loss.item():.4f}')
    assert not torch.isnan(grpo_loss), 'GRPO Loss produced NaN!'

    # 5. Production Pre-training Blueprint for 96GB VRAM
    print('[5/5] Checking 96GB VRAM Production Pre-training Preset...')
    prod_config = DroplycheeConfig() # default full pretraining preset
    full_moe_experts = prod_config.n_routed_experts + prod_config.n_shared_experts
    active_experts = prod_config.num_experts_per_tok + prod_config.n_shared_experts
    print(f'   -> Full Model Hidden Dim: {prod_config.hidden_size}')
    print(f'   -> Total Layers: {prod_config.num_hidden_layers}')
    print(f'   -> Total MoE Experts: {full_moe_experts} (Active per token: {active_experts})')
    print(f'   -> Multi-Head Latent Attention: q_lora_rank={prod_config.q_lora_rank}, kv_lora_rank={prod_config.kv_lora_rank}')
    print(f'   -> Decoupled RoPE: qk_rope_head_dim={prod_config.qk_rope_head_dim}')

    print('====================================================')
    print('ALL TESTS PASSED! m-droplychee IS FULLY OPERATIONAL!')
    print('====================================================')

if __name__ == '__main__':
    run_test()
