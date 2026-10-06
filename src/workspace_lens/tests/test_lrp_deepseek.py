"""LRP surgery on the tiny DeepSeek-V4 fixture: the exact patch set under the ``r+mhc``
preset (residual-stream norms, the clamped shared-expert MLP, every hyper-connection
and the head; the coefficient norms, routers, experts and attention untouched), the
detach-only rules bit-exact over all four streams, the clamped MLP and the mHC patchers
value-exact against the stock forwards, and the full rules within kernel dust. The second
half covers the ``all-c4+mhc`` preset: the clamped routed-experts loop, the router detach,
and the forward x4 on the shared expert (an intervention, tested as one)."""

from __future__ import annotations

import dataclasses

import pytest
import torch as t

from workspace_lens.lrp.lrp import (
    LrpRuleConfig,
    apply_lrp_rules,
    lrp_rule_config_for_mode,
)
from workspace_lens.tests.tiny_deepseek import (
    TINY_DEEPSEEK_NUM_LAYERS,
    build_tiny_deepseek,
    capture_block_output,
    random_tiny_deepseek_input_ids,
)

DETACH_ONLY_RULES = LrpRuleConfig(ln_rule=True, identity_rule=False, half_rule=True, mhc_detach=True)
FULL_RULES = lrp_rule_config_for_mode("r+mhc")


@pytest.fixture(scope="module")
def tiny_deepseek():
    return build_tiny_deepseek()


@pytest.fixture(scope="module")
def input_ids_Int_1S() -> t.Tensor:
    return random_tiny_deepseek_input_ids()


def streams_BSRN(model, input_ids_Int_1S: t.Tensor) -> t.Tensor:
    """The last block's four-stream output (before the collapse head), no grad."""
    return capture_block_output(model.model.layers[-1], lambda: model(input_ids_Int_1S, use_cache=False))


def test_r_mhc_patch_set_is_exactly_the_residual_norms_shared_experts_and_hyper_connections(tiny_deepseek):
    with apply_lrp_rules(tiny_deepseek, FULL_RULES) as report:
        pass
    num_layers = TINY_DEEPSEEK_NUM_LAYERS
    expected_norms = {"model.norm"} | {
        f"model.layers.{i}.{norm}" for i in range(num_layers) for norm in ("input_layernorm", "post_attention_layernorm")
    }
    assert set(report.ln_rule) == expected_norms  # the unweighted coefficient norms and q/kv norms are not here
    assert set(report.mlp_rule) == {f"model.layers.{i}.mlp.shared_experts" for i in range(num_layers)}
    assert set(report.mhc) == {"model.hc_head"} | {
        f"model.layers.{i}.{site}" for i in range(num_layers) for site in ("attn_hc", "ffn_hc")
    }
    assert report.moe_experts == [] and report.moe_gates == []


def test_rlens_alone_leaves_the_hyper_connections_stock(tiny_deepseek):
    with apply_lrp_rules(tiny_deepseek, lrp_rule_config_for_mode("rlens")) as report:
        assert report.mhc == []
        assert len(report.ln_rule) == 2 * TINY_DEEPSEEK_NUM_LAYERS + 1


def test_detach_only_rules_are_bit_exact_on_every_stream(tiny_deepseek, input_ids_Int_1S):
    stock_BSRN = streams_BSRN(tiny_deepseek, input_ids_Int_1S)
    with apply_lrp_rules(tiny_deepseek, DETACH_ONLY_RULES):
        patched_BSRN = streams_BSRN(tiny_deepseek, input_ids_Int_1S)
    assert patched_BSRN.shape == stock_BSRN.shape and patched_BSRN.dim() == 4
    assert t.equal(patched_BSRN, stock_BSRN)


def test_mhc_detach_alone_is_bit_exact_and_changes_only_the_coefficient_gradients(tiny_deepseek, input_ids_Int_1S):
    stock_BSRN = streams_BSRN(tiny_deepseek, input_ids_Int_1S)
    with apply_lrp_rules(tiny_deepseek, LrpRuleConfig(mhc_detach=True)) as report:
        patched_BSRN = streams_BSRN(tiny_deepseek, input_ids_Int_1S)
        assert set(report.patched_names()["mhc"]) and report.ln_rule == []
    assert t.equal(patched_BSRN, stock_BSRN)


def test_full_rules_stay_within_activation_kernel_dust(tiny_deepseek, input_ids_Int_1S):
    stock_BSRN = streams_BSRN(tiny_deepseek, input_ids_Int_1S)
    with apply_lrp_rules(tiny_deepseek, FULL_RULES):
        patched_BSRN = streams_BSRN(tiny_deepseek, input_ids_Int_1S)
    # The identity rule rewrites silu(z) as z * sigmoid(z): fp32 kernel dust only.
    assert (patched_BSRN - stock_BSRN).abs().max() < 1e-5 * stock_BSRN.abs().max()


def test_clamped_mlp_patcher_reproduces_the_clamps_where_they_bind():
    # A limit small enough that the clamps saturate on random activations: the patched
    # forward must still equal the stock forward (detach-only rules), and the identity
    # rule must act on the clamped pre-activation.
    model = build_tiny_deepseek(seed=3)
    mlp = model.model.layers[1].mlp.shared_experts
    mlp.limit = 0.05
    x_BSN = t.randn(1, 6, mlp.hidden_size) * 3.0
    with t.no_grad():
        stock_BSN = mlp(x_BSN)
        gate_BSI = mlp.gate_proj(x_BSN)
    assert (gate_BSI > mlp.limit).any(), "the fixture must make the gate clamp bind"
    with t.no_grad():
        assert (mlp.up_proj(x_BSN).abs() > mlp.limit).any(), "the fixture must make the up clamp bind"
    with apply_lrp_rules(model, DETACH_ONLY_RULES), t.no_grad():
        detach_only_BSN = mlp(x_BSN)
    assert t.equal(detach_only_BSN, stock_BSN)
    with apply_lrp_rules(model, FULL_RULES), t.no_grad():
        full_BSN = mlp(x_BSN)
    assert (full_BSN - stock_BSN).abs().max() < 1e-6


def test_mhc_detach_zeroes_the_coefficient_path_gradients_and_nothing_else(input_ids_Int_1S):
    """The mixing coefficients (pre, post, comb) are functions of the streams through the
    hyper-connection's `fn`, `base` and `scale`: under mhc_detach their parameters receive no
    gradient, while the block's other parameters still do; stock, they receive gradients."""
    model = build_tiny_deepseek(seed=7)
    hyper_connection = model.model.layers[1].attn_hc
    head = model.model.hc_head
    coefficient_parameters = [hyper_connection.fn, hyper_connection.base, hyper_connection.scale, head.hc_fn]
    content_parameter = model.model.layers[1].mlp.shared_experts.down_proj.weight
    for parameter in [*coefficient_parameters, content_parameter]:
        parameter.requires_grad_(True)

    def gradient_norms(config: LrpRuleConfig) -> tuple[list[float], float]:
        for parameter in [*coefficient_parameters, content_parameter]:
            parameter.grad = None
        with apply_lrp_rules(model, config):
            model(input_ids_Int_1S, use_cache=False).logits.float().pow(2).sum().backward()
        coefficient_norms = [0.0 if p.grad is None else float(p.grad.norm()) for p in coefficient_parameters]
        return coefficient_norms, float(content_parameter.grad.norm())

    stock_norms, stock_content = gradient_norms(LrpRuleConfig())
    detached_norms, detached_content = gradient_norms(LrpRuleConfig(mhc_detach=True))
    assert all(norm > 0 for norm in stock_norms)
    assert all(norm == 0 for norm in detached_norms)
    assert detached_content > 0 and stock_content > 0
    # And the patched hyper-connection hands back detached mixing coefficients.
    streams_BSRN = t.randn(1, 4, model.config.hc_mult, model.config.hidden_size, requires_grad=True)
    with apply_lrp_rules(model, LrpRuleConfig(mhc_detach=True)):
        post, comb, collapsed = hyper_connection(streams_BSRN)
    assert not post.requires_grad and not comb.requires_grad and collapsed.requires_grad


def test_detach_only_rules_are_bit_exact_in_bf16(input_ids_Int_1S):
    model = build_tiny_deepseek(seed=11).to(t.bfloat16)
    stock_BSRN = streams_BSRN(model, input_ids_Int_1S)
    with apply_lrp_rules(model, DETACH_ONLY_RULES):
        patched_BSRN = streams_BSRN(model, input_ids_Int_1S)
    assert patched_BSRN.dtype == t.bfloat16 and t.equal(patched_BSRN, stock_BSRN)


def test_context_exit_restores_the_stock_forward(tiny_deepseek, input_ids_Int_1S):
    stock_BSRN = streams_BSRN(tiny_deepseek, input_ids_Int_1S)
    with apply_lrp_rules(tiny_deepseek, FULL_RULES):
        pass
    assert t.equal(streams_BSRN(tiny_deepseek, input_ids_Int_1S), stock_BSRN)


def test_ln_rule_gradient_ignores_the_norm_scale_path(tiny_deepseek):
    # Under the LN rule the RMSNorm's gradient is the fixed diagonal scale weight * rsqrt(mean x^2 + eps).
    norm = tiny_deepseek.model.layers[0].input_layernorm
    x_N = t.randn(norm.weight.shape[0]) * 2.0
    with apply_lrp_rules(tiny_deepseek, LrpRuleConfig(ln_rule=True)):
        x_in = x_N.clone().requires_grad_(True)
        norm(x_in).sum().backward()
    expected_N = norm.weight * t.rsqrt(x_N.pow(2).mean() + norm.variance_epsilon)
    assert t.allclose(x_in.grad, expected_N, atol=1e-6, rtol=1e-5)


### THE all-c4+mhc RULE SET

C4_RULES = lrp_rule_config_for_mode("all-c4+mhc")
# Everything in the preset that only detaches or halves gradients: value-exact by construction. The forward x4 on
# the shared expert is an intervention and is left out here (tested on its own below).
C4_DETACH_ONLY_RULES = dataclasses.replace(C4_RULES, identity_rule=False, shared_expert_forward_scale=1.0)
FORWARD_SCALE_ONLY = LrpRuleConfig(shared_expert_forward_scale=4.0)


def test_all_c4_mhc_patch_set_adds_the_experts_and_scales_the_shared_experts(tiny_deepseek):
    """On top of r+mhc's norms, shared experts and hyper-connections, the preset patches every block's fused
    routed experts (moe_experts) and forward-scales every shared expert (shared_expert_forward_scale, the same
    DeepseekV4MLP modules as mlp_rule); no MoE block is patched (no gate, no backward scale) and attention and
    the routers' own modules stay stock (the router weights are detached inside the experts loop)."""
    with apply_lrp_rules(tiny_deepseek, C4_RULES) as report:
        pass
    num_layers = TINY_DEEPSEEK_NUM_LAYERS
    shared_experts = sorted(f"model.layers.{i}.mlp.shared_experts" for i in range(num_layers))
    assert sorted(report.moe_experts) == sorted(f"model.layers.{i}.mlp.experts" for i in range(num_layers))
    assert sorted(report.mlp_rule) == shared_experts
    assert sorted(report.shared_expert_forward_scale) == shared_experts
    assert not report.moe_gates and len(report.mhc) == 2 * num_layers + 1
    with apply_lrp_rules(tiny_deepseek, FULL_RULES) as r_mhc_report:
        pass
    assert not r_mhc_report.moe_experts and not r_mhc_report.shared_expert_forward_scale


@pytest.mark.parametrize("num_experts_per_tok", [None, 2])
def test_all_c4_mhc_detach_only_rules_are_bit_exact_on_every_stream(input_ids_Int_1S, num_experts_per_tok):
    """Router detach, the half rule, the LN and mHC detaches change gradients only: the four streams of the
    last block are bit-identical to stock, in fp32 and in bf16, on the hash-routed block and the top-k blocks,
    with every expert hit (the default fixture) and with two of four experts per token (unhit experts, partial
    masks and the `== num_experts` guard exercised)."""
    for dtype in (t.float32, t.bfloat16):
        model = build_tiny_deepseek(seed=3, num_experts_per_tok=num_experts_per_tok).to(dtype)
        stock = streams_BSRN(model, input_ids_Int_1S)
        with apply_lrp_rules(model, C4_DETACH_ONLY_RULES):
            patched = streams_BSRN(model, input_ids_Int_1S)
        assert t.equal(stock, patched), dtype


def test_forward_scale_multiplies_the_shared_expert_output_and_nothing_else(input_ids_Int_1S):
    """With only the forward scale on, each MoE block's output is exactly routed + 4 * shared, where routed and
    shared are the stock sub-module outputs on the block's own input; the whole model's last block therefore
    differs from stock (the preset fits the amplified model)."""
    model = build_tiny_deepseek(seed=17)
    block = model.model.layers[1].mlp  # a top-k block; layer 0 is hash-routed
    hidden_states_BSN = t.randn(1, 6, model.config.hidden_size)
    with t.no_grad():
        flat = hidden_states_BSN.view(-1, model.config.hidden_size)
        _, weights, indices = block.gate(hidden_states_BSN)
        routed = block.experts(flat, indices, weights).view_as(hidden_states_BSN)
        shared = block.shared_experts(hidden_states_BSN)
        with apply_lrp_rules(model, FORWARD_SCALE_ONLY):
            scaled_output = block(hidden_states_BSN)
        stock_output = block(hidden_states_BSN)
    assert t.equal(scaled_output, routed + 4.0 * shared)
    assert t.equal(stock_output, routed + shared)
    stock_streams = streams_BSRN(model, input_ids_Int_1S)
    with apply_lrp_rules(model, FORWARD_SCALE_ONLY):
        scaled_streams = streams_BSRN(model, input_ids_Int_1S)
    assert not t.allclose(stock_streams, scaled_streams)


def test_forward_scale_is_refused_where_no_deepseek_shared_expert_exists():
    """The forward scale is DeepSeek's shared-expert intervention; on a model without a DeepseekV4MLP the
    walker finds nothing to scale and refuses rather than fitting the stock model under a misleading label."""
    from workspace_lens.tests.tiny_gemma_olmo import build_tiny_olmo3

    with pytest.raises(RuntimeError, match="shared_expert_forward_scale"):
        with apply_lrp_rules(build_tiny_olmo3(), FORWARD_SCALE_ONLY):
            pass


def test_all_c4_mhc_rules_without_the_forward_scale_stay_within_activation_kernel_dust(tiny_deepseek, input_ids_Int_1S):
    """The identity rule inside the routed experts and the shared experts rewrites SiLU as x * sigmoid(x).detach():
    same value up to kernel rounding, so the preset minus its forward scale stays within dust of stock."""
    stock = streams_BSRN(tiny_deepseek, input_ids_Int_1S)
    with apply_lrp_rules(tiny_deepseek, dataclasses.replace(C4_RULES, shared_expert_forward_scale=1.0)):
        patched = streams_BSRN(tiny_deepseek, input_ids_Int_1S)
    assert t.allclose(stock, patched, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("layer", [0, 1])  # layer 0 hash-routed (fixed selection, learned mixing scores), layer 1 top-k
def test_router_detach_zeroes_both_routers_weight_gradients_and_keeps_the_experts(input_ids_Int_1S, layer):
    """Under all-c4+mhc a router's weight receives no gradient: the hash router fixes expert selection by token
    id but still mixes with learned scores, the top-k router does both, and in both cases the mixing weights are
    constants inside the experts loop. The routed experts' parameters still receive gradients; stock, the router
    weights do too."""
    model = build_tiny_deepseek(seed=11)
    block = model.model.layers[layer].mlp
    router_weight = block.gate.weight
    expert_weight = block.experts.down_proj
    for parameter in (router_weight, expert_weight):
        parameter.requires_grad_(True)

    def gradient_norms(config: LrpRuleConfig) -> tuple[float, float]:
        router_weight.grad = None
        expert_weight.grad = None
        with apply_lrp_rules(model, config):
            model(input_ids_Int_1S, use_cache=False).logits.float().pow(2).sum().backward()
        return (0.0 if router_weight.grad is None else float(router_weight.grad.norm()), float(expert_weight.grad.norm()))

    stock_router, stock_experts = gradient_norms(LrpRuleConfig())
    detached_router, detached_experts = gradient_norms(C4_RULES)
    assert stock_router > 0 and detached_router == 0
    assert stock_experts > 0 and detached_experts > 0


def test_half_rule_inside_the_routed_experts_halves_their_gate_up_gradient_at_the_last_block(input_ids_Int_1S):
    """The half rule on the routed experts' product halves the gradient reaching their gate/up weights and leaves
    the down weights' gradient alone (d loss / d down_proj is grad_out x product, untouched). At the last block the
    gradient arriving from the head is identical with and without the rule (the forward is bit-exact), so the
    ratio is exactly one half there; a patcher that forgot the half rule inside the experts would pass every
    forward test and fail here."""
    model = build_tiny_deepseek(seed=19)
    experts = model.model.layers[TINY_DEEPSEEK_NUM_LAYERS - 1].mlp.experts
    for parameter in (experts.gate_up_proj, experts.down_proj):
        parameter.requires_grad_(True)

    def gradients(config: LrpRuleConfig) -> tuple[t.Tensor, t.Tensor]:
        experts.gate_up_proj.grad = None
        experts.down_proj.grad = None
        with apply_lrp_rules(model, config):
            model(input_ids_Int_1S, use_cache=False).logits.float().pow(2).sum().backward()
        return experts.gate_up_proj.grad.clone(), experts.down_proj.grad.clone()

    with_half = dataclasses.replace(C4_DETACH_ONLY_RULES, half_rule=True)
    without_half = dataclasses.replace(C4_DETACH_ONLY_RULES, half_rule=False)
    gate_up_half, down_half = gradients(with_half)
    gate_up_full, down_full = gradients(without_half)
    assert float(gate_up_full.norm()) > 0
    assert t.allclose(gate_up_half, 0.5 * gate_up_full, rtol=1e-6, atol=0)
    assert t.allclose(down_half, down_full, rtol=1e-6, atol=0)


def test_backward_shared_scale_and_gate_detach_are_refused_on_deepseek(tiny_deepseek):
    """DeepSeek-V4's MoE block is never patched: a backward-only shared-expert scale or a shared-gate detach
    finds no module and the walker refuses (``all-c4+mhc`` uses neither; its x4 is the forward scale)."""
    for config in (
        LrpRuleConfig(ln_rule=True, routed_experts=True, shared_expert_scale=4.0),
        LrpRuleConfig(ln_rule=True, routed_experts=True, shared_gate_detach=True),
    ):
        with pytest.raises(RuntimeError, match="shared_gate_detach/shared_expert_scale"):
            with apply_lrp_rules(tiny_deepseek, config):
                pass


def test_expert_loop_reproduces_the_swiglu_clamps_where_they_bind(input_ids_Int_1S):
    """With the routed experts' ``limit`` lowered until both clamps bind on most activations, the patched
    experts loop matches the stock block output bit for bit under the detach-only rules, and stays within dust
    under the identity rule (applied to the CLAMPED gate, as the stock forward clamps before SiLU). At the
    default limit a random-init fixture never clamps, so this is the test that would catch a missing or
    misplaced clamp."""
    model = build_tiny_deepseek(seed=13)
    limit = 0.02
    for layer in model.model.layers:
        layer.mlp.experts.limit = limit
    # Count the clamps on the experts' actual inputs of the tested forward (hooked, stock rules).
    binds = 0

    def count_binding_clamps(module, args, output):
        nonlocal binds
        hidden_states, _, _ = args
        with t.no_grad():
            gate_up = t.einsum("th,ech->etc", hidden_states, module.gate_up_proj)
            gate, up = gate_up.chunk(2, dim=-1)
            binds += int((gate > limit).sum() + (up.abs() > limit).sum())

    handles = [layer.mlp.experts.register_forward_hook(count_binding_clamps) for layer in model.model.layers]
    try:
        stock = streams_BSRN(model, input_ids_Int_1S)
    finally:
        for handle in handles:
            handle.remove()
    assert binds > 0
    with apply_lrp_rules(model, C4_DETACH_ONLY_RULES):
        patched = streams_BSRN(model, input_ids_Int_1S)
    assert t.equal(stock, patched)
    with apply_lrp_rules(model, dataclasses.replace(C4_RULES, shared_expert_forward_scale=1.0)):
        with_identity_rule = streams_BSRN(model, input_ids_Int_1S)
    assert t.allclose(stock, with_identity_rule, atol=1e-6, rtol=1e-6)
