"""Tests for the Gemma 4 and OLMo 3 LRP patchers, mirroring test_lrp.py's
guarantees on these two module families:

1. Forward values are unchanged: bit-exact under the detach-only rules
   (LN-rule, half-rule); equal up to kernel dust under the identity rule,
   whose GELU-tanh rewrite ``z * gate(z)`` replaces ``F.gelu(z, approximate=
   "tanh")`` on Gemma and the SiLU rewrite on OLMo.
2. The patch report covers exactly the residual-stream norms (Gemma's four per
   block, OLMo's two post-sublayer norms, the final norm) and every block MLP,
   and leaves the attention-internal norms (q_norm, k_norm, Gemma's v_norm)
   alone.
3. Gradients follow the rules: the LN-rule gives the fixed diagonal scale with
   the plain weight (Gemma stores w, not 1 + w; OLMo applies w in fp32 before
   the cast); the GELU identity rule matches a reference built from the
   module's own weights in value and gradient.
4. Exiting the context restores stock behaviour bit-exactly.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch as t
from torch.nn import functional as F

from workspace_lens.lrp.lrp import (
    PATCH_REPORT_FAMILIES,
    LrpPatchReport,
    LrpRuleConfig,
    _gelu_tanh_with_detached_gate,
    apply_lrp_rules,
    is_attention_internal_norm,
    is_multimodal_tower_module,
    lrp_rule_config_for_mode,
)
from workspace_lens.tests.tiny_gemma_olmo import (
    TINY_D_MODEL,
    build_tiny_gemma4,
    build_tiny_gemma4_multimodal,
    build_tiny_olmo3,
)
from workspace_lens.tests.tiny_qwen import (
    last_hidden_state_BSN,
)
from workspace_lens.tests.tiny_qwen import (
    random_tiny_qwen_input_ids as random_tiny_input_ids,
)

GEMMA_BLOCK_NORMS = (
    "input_layernorm",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm",
)


def residual_norm_names(prefix: str, num_layers: int, block_norms: tuple[str, ...]) -> set[str]:
    """The residual-stream norms the LN-rule must patch: ``block_norms`` in every
    block plus the final norm, under ``prefix`` (``model`` or ``model.language_model``)."""
    return {f"{prefix}.norm"} | {
        f"{prefix}.layers.{i}.{norm}" for i in range(num_layers) for norm in block_norms
    }

DETACH_ONLY_RULES = LrpRuleConfig(ln_rule=True, identity_rule=False, half_rule=True)
FULL_RULES = lrp_rule_config_for_mode("rlens")


@pytest.fixture(scope="module")
def tiny_gemma():
    return build_tiny_gemma4()


@pytest.fixture(scope="module")
def tiny_olmo():
    return build_tiny_olmo3()


@pytest.fixture(scope="module")
def input_ids_Int_1S() -> t.Tensor:
    return random_tiny_input_ids()


@pytest.fixture(params=["gemma", "olmo"])
def tiny_model(request, tiny_gemma, tiny_olmo):
    return tiny_gemma if request.param == "gemma" else tiny_olmo


### 1. FORWARD VALUES


def test_detach_only_rules_are_bit_exact(tiny_model, input_ids_Int_1S):
    stock_BSN = last_hidden_state_BSN(tiny_model, input_ids_Int_1S)
    with apply_lrp_rules(tiny_model, DETACH_ONLY_RULES):
        patched_BSN = last_hidden_state_BSN(tiny_model, input_ids_Int_1S)
    assert t.equal(stock_BSN, patched_BSN)


def test_full_rules_value_close_up_to_activation_kernel_dust(tiny_model, input_ids_Int_1S):
    stock_BSN = last_hidden_state_BSN(tiny_model, input_ids_Int_1S)
    with apply_lrp_rules(tiny_model, FULL_RULES):
        patched_BSN = last_hidden_state_BSN(tiny_model, input_ids_Int_1S)
    assert t.allclose(stock_BSN, patched_BSN, rtol=1e-4, atol=1e-5)


def test_gelu_tanh_rewrite_matches_the_torch_kernel():
    generator = t.Generator().manual_seed(5)
    z_SN = t.randn(64, 32, generator=generator) * 3
    assert t.allclose(
        _gelu_tanh_with_detached_gate(z_SN), F.gelu(z_SN, approximate="tanh"), atol=1e-6
    )


def test_gelu_tanh_rewrite_bf16_dust_is_bounded_like_the_silu_rewrite():
    # Real models run in bf16. The rewrite's forward error against the fused kernel,
    # relative to the largest activation, must stay at bf16 rounding scale, as the
    # SiLU rewrite's does.
    generator = t.Generator().manual_seed(6)
    z_SN = (t.randn(4096, 64, generator=generator) * 3).to(t.bfloat16)
    reference_SN = F.gelu(z_SN, approximate="tanh")
    error = (_gelu_tanh_with_detached_gate(z_SN).float() - reference_SN.float()).abs().max()
    assert float(error / reference_SN.float().abs().max()) < 5e-3


def test_context_exit_restores_stock_bit_exact(tiny_model, input_ids_Int_1S):
    stock_BSN = last_hidden_state_BSN(tiny_model, input_ids_Int_1S)
    with apply_lrp_rules(tiny_model, FULL_RULES):
        pass
    assert t.equal(stock_BSN, last_hidden_state_BSN(tiny_model, input_ids_Int_1S))
    assert "forward" not in tiny_model.model.layers[0].mlp.__dict__


### 2. WHICH MODULES ARE PATCHED


def test_gemma_patch_report_covers_block_norms_and_mlps_not_attention_norms(tiny_gemma):
    with apply_lrp_rules(tiny_gemma, FULL_RULES) as report:
        pass
    num_layers = len(tiny_gemma.model.layers)
    assert set(report.ln_rule) == residual_norm_names("model", num_layers, GEMMA_BLOCK_NORMS)
    assert set(report.mlp_rule) == {f"model.layers.{i}.mlp" for i in range(num_layers)}
    # Gemma's attention holds q_norm, k_norm and a weightless v_norm, all Gemma4RMSNorm.
    attention = tiny_gemma.model.layers[0].self_attn
    assert type(attention.v_norm).__name__ == "Gemma4RMSNorm" and not attention.v_norm.with_scale
    assert not any(".self_attn." in name for name in report.ln_rule)


def test_olmo_patch_report_covers_post_sublayer_norms_and_mlps(tiny_olmo):
    with apply_lrp_rules(tiny_olmo, FULL_RULES) as report:
        pass
    num_layers = len(tiny_olmo.model.layers)
    assert set(report.ln_rule) == residual_norm_names(
        "model", num_layers, ("post_attention_layernorm", "post_feedforward_layernorm")
    )
    assert set(report.mlp_rule) == {f"model.layers.{i}.mlp" for i in range(num_layers)}


def test_patch_report_families_match_the_dataclass_fields():
    # LrpPatchReport.patched_names iterates the written-out tuple; a new family must join both.
    assert PATCH_REPORT_FAMILIES == tuple(
        f.name for f in dataclasses.fields(LrpPatchReport) if f.name != "lrp_config"
    )


def test_attention_internal_and_tower_name_rules():
    # Path segments, not substrings: the rules hold with or without a wrapper prefix.
    assert is_attention_internal_norm("model.layers.3.self_attn.v_norm")
    assert is_attention_internal_norm("layers.3.self_attn.v_norm")
    assert not is_attention_internal_norm("model.layers.3.post_attention_layernorm")
    assert not is_attention_internal_norm("model.layers.3.self_attn_like.norm")
    assert is_multimodal_tower_module("model.vision_tower.encoder.layers.0.mlp")
    assert is_multimodal_tower_module("vision_tower.encoder.layers.0.mlp")
    assert is_multimodal_tower_module("model.audio_tower.layers.0.norm_pre_attn")
    assert is_multimodal_tower_module("model.embed_vision.embedding_pre_projection_norm")
    assert not is_multimodal_tower_module("model.language_model.layers.0.mlp")


def test_multimodal_wrapper_patches_only_the_text_decoder():
    # The real gemma-4-31B is a Gemma4ForConditionalGeneration: vision and audio
    # towers (their MLP has the gated-MLP attribute shape and an unregistered
    # class) and the embed_vision / embed_audio projectors sit beside the text
    # decoder at model.language_model. The surgery must skip all of them, patch
    # exactly the decoder's residual-stream norms and MLPs, and not raise.
    wrapper = build_tiny_gemma4_multimodal()
    with apply_lrp_rules(wrapper, FULL_RULES) as report:
        input_ids_Int_1S = random_tiny_input_ids()
        with t.no_grad():
            patched_BSN = wrapper.model.language_model(
                input_ids=input_ids_Int_1S, use_cache=False
            ).last_hidden_state
    num_layers = len(wrapper.model.language_model.layers)
    prefix = "model.language_model"
    assert set(report.ln_rule) == residual_norm_names(prefix, num_layers, GEMMA_BLOCK_NORMS)
    assert set(report.mlp_rule) == {f"{prefix}.layers.{i}.mlp" for i in range(num_layers)}
    for names in report.patched_names().values():
        for name in names:
            assert not is_multimodal_tower_module(name) and not is_attention_internal_norm(name)
    with t.no_grad():
        stock_BSN = wrapper.model.language_model(
            input_ids=input_ids_Int_1S, use_cache=False
        ).last_hidden_state
    assert t.allclose(stock_BSN, patched_BSN, rtol=1e-4, atol=1e-5)


### 3. GRADIENTS


def _rmsnorm_eps(rmsnorm) -> float:
    return getattr(rmsnorm, "eps", None) or rmsnorm.variance_epsilon


def test_ln_rule_gradient_is_fixed_diagonal_scale_with_plain_weight(tiny_model):
    # LN-rule: y = x * [rsqrt(mean(x^2) + eps)]detached * w, so the input
    # gradient must be exactly cotangent * w * scale, with no projection term
    # from the variance. Both families store the plain weight w (Qwen3.5's
    # 1 + w convention does not apply).
    rmsnorm = tiny_model.model.norm
    generator = t.Generator().manual_seed(2)
    x_SN = t.randn(5, TINY_D_MODEL, generator=generator, requires_grad=True)
    cotangent_SN = t.randn(5, TINY_D_MODEL, generator=generator)

    with apply_lrp_rules(tiny_model, LrpRuleConfig(ln_rule=True)):
        (patched_grad_SN,) = t.autograd.grad(rmsnorm(x_SN), x_SN, cotangent_SN)

    scale_S1 = t.rsqrt(x_SN.detach().pow(2).mean(-1, keepdim=True) + _rmsnorm_eps(rmsnorm))
    expected_grad_SN = cotangent_SN * rmsnorm.weight * scale_S1
    assert t.allclose(patched_grad_SN, expected_grad_SN, rtol=1e-5, atol=1e-7)

    (stock_grad_SN,) = t.autograd.grad(rmsnorm(x_SN), x_SN, cotangent_SN)
    assert not t.allclose(patched_grad_SN, stock_grad_SN, rtol=1e-3, atol=1e-5)


def test_gelu_identity_rule_matches_reference_implementation(tiny_gemma):
    # Independent reference built from the module's own weights: the patched
    # MLP must equal down((z * gate(z).detach()) * up(x)) in value and gradient,
    # with gate the tanh-approximation factor of gelu_pytorch_tanh.
    mlp = tiny_gemma.model.layers[0].mlp
    generator = t.Generator().manual_seed(3)
    x_SN = t.randn(5, TINY_D_MODEL, generator=generator, requires_grad=True)
    cotangent_SN = t.randn(5, TINY_D_MODEL, generator=generator)

    with apply_lrp_rules(tiny_gemma, LrpRuleConfig(identity_rule=True)):
        patched_SN = mlp(x_SN)
        (patched_grad_SN,) = t.autograd.grad(patched_SN, x_SN, cotangent_SN)

    z_SI = mlp.gate_proj(x_SN)
    gate_SI = 0.5 * (1.0 + t.tanh(math.sqrt(2.0 / math.pi) * (z_SI + 0.044715 * z_SI.pow(3))))
    reference_SN = mlp.down_proj((z_SI * gate_SI.detach()) * mlp.up_proj(x_SN))
    (reference_grad_SN,) = t.autograd.grad(reference_SN, x_SN, cotangent_SN)
    assert t.allclose(patched_SN, reference_SN, rtol=1e-5, atol=1e-6)
    assert t.allclose(patched_grad_SN, reference_grad_SN, rtol=1e-5, atol=1e-6)

    (stock_grad_SN,) = t.autograd.grad(mlp(x_SN), x_SN, cotangent_SN)
    assert not t.allclose(patched_grad_SN, stock_grad_SN, rtol=1e-3, atol=1e-5)


def test_half_rule_halves_the_mlp_gradient(tiny_model):
    mlp = tiny_model.model.layers[0].mlp
    generator = t.Generator().manual_seed(4)
    x_SN = t.randn(5, TINY_D_MODEL, generator=generator, requires_grad=True)
    cotangent_SN = t.randn(5, TINY_D_MODEL, generator=generator)
    (stock_grad_SN,) = t.autograd.grad(mlp(x_SN), x_SN, cotangent_SN)
    with apply_lrp_rules(tiny_model, LrpRuleConfig(half_rule=True)):
        (patched_grad_SN,) = t.autograd.grad(mlp(x_SN), x_SN, cotangent_SN)
    assert t.allclose(patched_grad_SN, 0.5 * stock_grad_SN, rtol=1e-6, atol=1e-7)


### 4. BF16


def test_rmsnorm_patchers_are_bit_exact_in_bf16_with_non_unit_weights(tiny_model):
    # The models run in bf16, where the cast order (Gemma: scale in fp32 then type_as; OLMo:
    # weight applied in fp32 before the downcast) decides the bits. A wrong order would still
    # pass the fp32 tests above.
    rmsnorm = tiny_model.model.layers[0].post_attention_layernorm
    generator = t.Generator().manual_seed(7)
    with t.no_grad():
        rmsnorm.weight.copy_(t.rand(TINY_D_MODEL, generator=generator) * 2 + 0.25)
    x_SN = (t.randn(9, TINY_D_MODEL, generator=generator) * 4).to(t.bfloat16)
    stock_SN = rmsnorm(x_SN)
    with apply_lrp_rules(tiny_model, LrpRuleConfig(ln_rule=True)):
        patched_SN = rmsnorm(x_SN)
    assert patched_SN.dtype == t.bfloat16 and t.equal(stock_SN, patched_SN)
