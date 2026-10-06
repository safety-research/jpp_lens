"""Tests for the LRP forward-graph surgery (workspace_lens.lrp).

What the surgery must guarantee, each pinned below:
1. Forward values are unchanged — bit-exact for the detach-only rules
   (LN-rule, half-rule), and equal up to SiLU kernel dust when the
   identity-rule rewrites ``F.silu(z)`` as ``z * sigmoid(z)``.
2. Gradients follow the intended propagation rules — checked analytically
   (LN-rule: fixed diagonal scale; half-rule: exactly half the product-node
   gradient) and against an independent reference implementation built from
   the module's own weights (identity-rule).
3. Exiting the context restores stock behaviour bit-exactly.
4. Wrong-artifact failure modes raise instead of silently half-patching:
   unknown module classes, zero modules patched, torch.compile'd blocks,
   and fla-capable environments.
"""

from __future__ import annotations

import json

import pytest
import torch as t

from jlens.tests.tiny import TinyDecoder
from workspace_lens.lrp.lrp import (
    LrpRuleConfig,
    apply_lrp_rules,
    lrp_rule_config_for_mode,
)
from workspace_lens.lrp.types import LRP_MODES
from workspace_lens.tests.tiny_qwen import (
    TINY_QWEN_VOCAB_SIZE,
    build_tiny_qwen3_5,
    last_hidden_state_BSN,
    random_tiny_qwen_input_ids,
)


@pytest.fixture(scope="module")
def tiny_qwen():
    return build_tiny_qwen3_5(seed=0)


@pytest.fixture(scope="module")
def input_ids_Int_1S() -> t.Tensor:
    return random_tiny_qwen_input_ids()


### 1. FORWARD VALUES


def test_detach_only_rules_are_bit_exact(tiny_qwen, input_ids_Int_1S):
    # LN-rule and half-rule replicate the stock ops exactly; only .detach()s
    # are inserted, which cannot change values.
    stock_BSN = last_hidden_state_BSN(tiny_qwen, input_ids_Int_1S)
    config = LrpRuleConfig(ln_rule=True, identity_rule=False, half_rule=True)
    with apply_lrp_rules(tiny_qwen, config):
        patched_BSN = last_hidden_state_BSN(tiny_qwen, input_ids_Int_1S)
    assert t.equal(stock_BSN, patched_BSN)


def test_full_rules_value_close_up_to_silu_kernel_dust(tiny_qwen, input_ids_Int_1S):
    stock_BSN = last_hidden_state_BSN(tiny_qwen, input_ids_Int_1S)
    with apply_lrp_rules(tiny_qwen, lrp_rule_config_for_mode("rlens")):
        patched_BSN = last_hidden_state_BSN(tiny_qwen, input_ids_Int_1S)
    assert t.allclose(stock_BSN, patched_BSN, rtol=1e-5, atol=1e-6)


def test_context_exit_restores_stock_bit_exact(tiny_qwen, input_ids_Int_1S):
    stock_BSN = last_hidden_state_BSN(tiny_qwen, input_ids_Int_1S)
    with apply_lrp_rules(tiny_qwen, lrp_rule_config_for_mode("rlens")):
        pass
    restored_BSN = last_hidden_state_BSN(tiny_qwen, input_ids_Int_1S)
    assert t.equal(stock_BSN, restored_BSN)


### 2. PATCH SCOPE


def test_patch_report_covers_expected_modules(tiny_qwen):
    rmsnorm_names = [
        name
        for name, module in tiny_qwen.named_modules()
        if type(module).__name__ == "Qwen3_5RMSNorm"
    ]
    qk_norm_names = [
        name for name in rmsnorm_names if name.rsplit(".", 1)[-1] in ("q_norm", "k_norm")
    ]
    assert qk_norm_names, "fixture should contain q/k norms in the softmax layers"

    with apply_lrp_rules(tiny_qwen, lrp_rule_config_for_mode("rlens")) as report:
        pass
    # q/k norms excluded by default; RMSNormGated (GatedDeltaNet) never matches.
    assert sorted(report.ln_rule) == sorted(set(rmsnorm_names) - set(qk_norm_names))
    assert len(report.mlp_rule) == tiny_qwen.config.num_hidden_layers


### 3. GRADIENTS


def test_ln_rule_gradient_is_fixed_diagonal_scale(tiny_qwen):
    # LN-rule: y = x * [rsqrt(mean(x^2) + eps)]detached * (1 + w), so the
    # input gradient must be exactly cotangent * (1 + w) * scale, with no
    # projection term from the variance.
    rmsnorm = tiny_qwen.model.layers[0].input_layernorm
    generator = t.Generator().manual_seed(2)
    x_SN = t.randn(5, 32, generator=generator, requires_grad=True)
    cotangent_SN = t.randn(5, 32, generator=generator)

    with apply_lrp_rules(
        tiny_qwen, LrpRuleConfig(ln_rule=True, identity_rule=False, half_rule=False)
    ):
        (patched_grad_SN,) = t.autograd.grad(rmsnorm(x_SN), x_SN, cotangent_SN)

    scale_S1 = t.rsqrt(x_SN.detach().pow(2).mean(-1, keepdim=True) + rmsnorm.eps)
    expected_grad_SN = cotangent_SN * (1.0 + rmsnorm.weight) * scale_S1
    assert t.allclose(patched_grad_SN, expected_grad_SN, rtol=1e-5, atol=1e-7)

    (stock_grad_SN,) = t.autograd.grad(rmsnorm(x_SN), x_SN, cotangent_SN)
    assert not t.allclose(patched_grad_SN, stock_grad_SN, rtol=1e-3, atol=1e-5)


def test_identity_rule_matches_reference_implementation(tiny_qwen):
    # Independent reference built from the module's own weights: the patched
    # MLP must equal down((z * sigmoid(z).detach()) * up(x)) in both value
    # and gradient.
    mlp = tiny_qwen.model.layers[0].mlp
    generator = t.Generator().manual_seed(3)
    x_SN = t.randn(5, 32, generator=generator, requires_grad=True)
    cotangent_SN = t.randn(5, 32, generator=generator)

    with apply_lrp_rules(
        tiny_qwen, LrpRuleConfig(ln_rule=False, identity_rule=True, half_rule=False)
    ):
        patched_SN = mlp(x_SN)
        (patched_grad_SN,) = t.autograd.grad(patched_SN, x_SN, cotangent_SN)

    z_SH = x_SN @ mlp.gate_proj.weight.T
    reference_SN = (z_SH * t.sigmoid(z_SH).detach() * (x_SN @ mlp.up_proj.weight.T)) @ (
        mlp.down_proj.weight.T
    )
    (reference_grad_SN,) = t.autograd.grad(reference_SN, x_SN, cotangent_SN)
    assert t.allclose(patched_SN, reference_SN, rtol=1e-5, atol=1e-7)
    assert t.allclose(patched_grad_SN, reference_grad_SN, rtol=1e-5, atol=1e-7)

    (stock_grad_SN,) = t.autograd.grad(mlp(x_SN), x_SN, cotangent_SN)
    assert not t.allclose(patched_grad_SN, stock_grad_SN, rtol=1e-3, atol=1e-5)


def test_half_rule_halves_the_mlp_gradient(tiny_qwen):
    # With only the half-rule on, forward values are bit-exact and every path
    # to the input runs through the halved product node, so the input
    # gradient is exactly half the stock gradient.
    mlp = tiny_qwen.model.layers[0].mlp
    generator = t.Generator().manual_seed(4)
    x_SN = t.randn(5, 32, generator=generator, requires_grad=True)
    cotangent_SN = t.randn(5, 32, generator=generator)

    stock_SN = mlp(x_SN)
    (stock_grad_SN,) = t.autograd.grad(stock_SN, x_SN, cotangent_SN)

    with apply_lrp_rules(
        tiny_qwen, LrpRuleConfig(ln_rule=False, identity_rule=False, half_rule=True)
    ):
        patched_SN = mlp(x_SN)
        (patched_grad_SN,) = t.autograd.grad(patched_SN, x_SN, cotangent_SN)

    assert t.equal(stock_SN, patched_SN)
    assert t.allclose(patched_grad_SN, 0.5 * stock_grad_SN, rtol=1e-6, atol=1e-9)


def test_graph_recorded_under_patches_survives_context_exit(tiny_qwen):
    # The whole trainer design rests on this: backward run AFTER the context
    # exits still reads the detach-edited graph.
    rmsnorm = tiny_qwen.model.layers[0].input_layernorm
    generator = t.Generator().manual_seed(5)
    x_SN = t.randn(5, 32, generator=generator, requires_grad=True)
    cotangent_SN = t.randn(5, 32, generator=generator)

    with apply_lrp_rules(
        tiny_qwen, LrpRuleConfig(ln_rule=True, identity_rule=False, half_rule=False)
    ):
        output_SN = rmsnorm(x_SN)
    (grad_after_exit_SN,) = t.autograd.grad(output_SN, x_SN, cotangent_SN)

    scale_S1 = t.rsqrt(x_SN.detach().pow(2).mean(-1, keepdim=True) + rmsnorm.eps)
    expected_grad_SN = cotangent_SN * (1.0 + rmsnorm.weight) * scale_S1
    assert t.allclose(grad_after_exit_SN, expected_grad_SN, rtol=1e-5, atol=1e-7)


### 4. GUARDS AND PRESETS


def test_unknown_rmsnorm_class_raises(tiny_qwen):
    class WeirdRMSNorm(t.nn.Module):
        def forward(self, x: t.Tensor) -> t.Tensor:
            return x

    container = t.nn.Sequential(WeirdRMSNorm())
    with pytest.raises(RuntimeError, match="No LN-rule patcher"):
        with apply_lrp_rules(container, lrp_rule_config_for_mode("rlens")):
            pass
    # a failed entry must not leave stale patches on other models
    _ = last_hidden_state_BSN(tiny_qwen, t.randint(0, TINY_QWEN_VOCAB_SIZE, (1, 8)))


def test_zero_patched_modules_raises():
    tiny_decoder = TinyDecoder()  # no RMSNorm or gated MLP anywhere
    with pytest.raises(RuntimeError, match="no module matched"):
        with apply_lrp_rules(tiny_decoder, lrp_rule_config_for_mode("rlens")):
            pass


def test_compiled_block_raises(tiny_qwen):
    original_layer = tiny_qwen.model.layers[0]
    tiny_qwen.model.layers[0] = t.compile(original_layer, backend="eager")
    try:
        with pytest.raises(RuntimeError, match="torch.compile"):
            with apply_lrp_rules(tiny_qwen, lrp_rule_config_for_mode("rlens")):
                pass
    finally:
        tiny_qwen.model.layers[0] = original_layer


def test_moe_modes_on_a_dense_model_raise(tiny_qwen):
    # The per-family guard: on a dense model the dense rules would still
    # patch, so without it r+moe would silently degrade to rlens.
    with pytest.raises(RuntimeError, match="routed_experts"):
        with apply_lrp_rules(tiny_qwen, lrp_rule_config_for_mode("r+moe")):
            pass


def test_mode_presets():
    assert lrp_rule_config_for_mode("none") == LrpRuleConfig()
    rlens = lrp_rule_config_for_mode("rlens")
    assert (rlens.ln_rule, rlens.identity_rule, rlens.half_rule) == (True, True, True)
    assert not rlens.routed_experts
    r_moe = lrp_rule_config_for_mode("r+moe")
    assert r_moe.routed_experts and r_moe.router_detach and r_moe.shared_gate_detach
    assert r_moe.shared_expert_scale == 4.0
    r_mhc = lrp_rule_config_for_mode("r+mhc")  # DeepSeek-V4: minimal RelP rules + mHC detach
    assert r_mhc.mhc_detach and (r_mhc.ln_rule, r_mhc.identity_rule, r_mhc.half_rule) == (True, True, True)
    assert not r_mhc.routed_experts and not rlens.mhc_detach
    c4 = lrp_rule_config_for_mode("all-c4+mhc")  # DeepSeek-V4: + routed experts and a forward x4
    assert c4.routed_experts and c4.router_detach and c4.mhc_detach and c4.shared_expert_forward_scale == 4.0
    assert c4.shared_expert_scale == 1.0 and not c4.shared_gate_detach  # the x4 is forward, not backward-only
    assert (c4.ln_rule, c4.identity_rule, c4.half_rule) == (True, True, True)
    assert set(LRP_MODES) == {"none", "rlens", "all-c4+mhc", "r+mhc", "r+moe"}
    with pytest.raises(ValueError, match="unknown lrp_mode"):
        lrp_rule_config_for_mode("gamma-rule")


# The rule flags recorded in the provenance (config_json) of camilablank/workspace-lenses
# deepseek-v4-flash/r-lens/lens.pt. Its one scale knob, "shared_expert_scale", is read here as our
# shared_expert_forward_scale, the forward x4 on the shared expert's output that all-c4+mhc applies.
HUB_DEEPSEEK_R_LENS_CONFIG_JSON = (
    '{"estimator": "relp", "arm": "all-c4", "rules": {"ln_rule": true, "identity_rule": true, "half_rule": true, '
    '"include_qk_norms": false, "routed_experts": true, "router_detach": true, "router_half": false, '
    '"shared_gate_detach": false, "mhc_detach": true, "shared_expert_scale": 4.0}}'
)
HUB_DEEPSEEK_WRITER_RENAMES = {"shared_expert_scale": "shared_expert_forward_scale"}


def test_all_c4_mhc_preset_matches_the_hub_deepseek_r_lens_provenance_flags():
    """Every flag recorded in the Hub DeepSeek-V4-Flash R-Lens provenance that our config also has
    agrees with the ``all-c4+mhc`` preset, with the file's "shared_expert_scale" read as
    ``shared_expert_forward_scale``; the recorded flags we do not model (router_half,
    include_qk_norms) are False there."""
    hub_rules = json.loads(HUB_DEEPSEEK_R_LENS_CONFIG_JSON)["rules"]
    hub_rules = {HUB_DEEPSEEK_WRITER_RENAMES.get(flag, flag): value for flag, value in hub_rules.items()}
    ours = lrp_rule_config_for_mode("all-c4+mhc").to_dict()
    shared_flags = set(hub_rules) & set(ours)
    assert {"ln_rule", "identity_rule", "half_rule", "routed_experts", "router_detach",
            "shared_gate_detach", "mhc_detach", "shared_expert_forward_scale"} <= shared_flags
    assert {flag: ours[flag] for flag in shared_flags} == {flag: hub_rules[flag] for flag in shared_flags}
    assert set(hub_rules) - set(ours) == {"router_half", "include_qk_norms"}
    assert hub_rules["router_half"] is False and hub_rules["include_qk_norms"] is False
