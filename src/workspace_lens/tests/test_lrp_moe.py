"""Tests for the MoE LRP rules (the ``r+moe`` preset) and the mHC detach (workspace_lens.lrp).

MoE guarantees pinned here, on a tiny hybrid Qwen3.5-MoE built from the exact
``Qwen3_5Moe*`` classes of the Qwen3.6-35B-A3B model:
1. The detach-only MoE edits (router detach, shared-gate detach, backward
   shared-expert scale, half-rule) are bit-exact in the forward; the full
   ``r+moe`` preset is equal up to SiLU kernel dust.
2. The backward-only ``shared_expert_scale`` identity ``s + (c-1)(s-[s]c)``
   is exactly zero-correction forward and exactly x c backward.
3. Rules change gradients (router path dead under router_detach) while
   values, and therefore expert routing, are unchanged.
4. The mHC patchers run and detach on a mock module carrying the DeepSeek
   attribute layout.
"""

from __future__ import annotations

import pytest
import torch as t

from workspace_lens.lrp.lrp import (
    LrpRuleConfig,
    apply_lrp_rules,
    lrp_rule_config_for_mode,
)
from workspace_lens.tests.tiny_qwen import (
    build_tiny_qwen3_5_moe,
    last_hidden_state_BSN,
    random_tiny_qwen_input_ids,
)


@pytest.fixture(scope="module")
def tiny_moe():
    return build_tiny_qwen3_5_moe(seed=0)


@pytest.fixture(scope="module")
def input_ids_Int_1S() -> t.Tensor:
    return random_tiny_qwen_input_ids()


def test_detach_only_moe_edits_value_equal_and_scoped(tiny_moe, input_ids_Int_1S):
    # Block-level knobs (gate detach, backward scale) and ln/half are
    # bit-exact, but the experts patcher replaces a transformers dispatch
    # wrapper with the reference eager loop — reduction-order dust of ~1 ulp
    # is expected even with no rules applied inside, so the standard here is
    # tight allclose, not t.equal (pinned separately below for the block).
    stock_BSN = last_hidden_state_BSN(tiny_moe, input_ids_Int_1S)
    config = LrpRuleConfig(
        ln_rule=True,
        half_rule=True,
        routed_experts=True,
        router_detach=True,
        shared_gate_detach=True,
        shared_expert_scale=4.0,
    )
    with apply_lrp_rules(tiny_moe, config) as report:
        patched_BSN = last_hidden_state_BSN(tiny_moe, input_ids_Int_1S)
    assert t.allclose(stock_BSN, patched_BSN, rtol=1e-6, atol=1e-6)
    num_layers = tiny_moe.config.num_hidden_layers
    assert len(report.moe_experts) == num_layers
    assert len(report.moe_gates) == num_layers
    assert len(report.mlp_rule) == num_layers  # the shared experts (Qwen3_5MoeMLP)


def test_block_level_knobs_alone_are_bit_exact(tiny_moe, input_ids_Int_1S):
    # Without the experts patcher, the remaining MoE knobs are pure detaches
    # and the exact zero-correction scale identity: bitwise identical.
    stock_BSN = last_hidden_state_BSN(tiny_moe, input_ids_Int_1S)
    config = LrpRuleConfig(shared_gate_detach=True, shared_expert_scale=4.0)
    with apply_lrp_rules(tiny_moe, config):
        patched_BSN = last_hidden_state_BSN(tiny_moe, input_ids_Int_1S)
    assert t.equal(stock_BSN, patched_BSN)


def test_full_r_moe_preset_value_close_up_to_silu_dust(tiny_moe, input_ids_Int_1S):
    stock_BSN = last_hidden_state_BSN(tiny_moe, input_ids_Int_1S)
    with apply_lrp_rules(tiny_moe, lrp_rule_config_for_mode("r+moe")):
        patched_BSN = last_hidden_state_BSN(tiny_moe, input_ids_Int_1S)
    assert t.allclose(stock_BSN, patched_BSN, rtol=1e-5, atol=1e-6)


def test_shared_expert_scale_identity_is_exact():
    generator = t.Generator().manual_seed(2)
    shared_SN = t.randn(5, 8, generator=generator, requires_grad=True)
    cotangent_SN = t.randn(5, 8, generator=generator)
    scale = 4.0

    scaled_SN = shared_SN + (scale - 1.0) * (shared_SN - shared_SN.detach())
    assert t.equal(scaled_SN.detach(), shared_SN.detach())  # correction exactly 0
    (grad_SN,) = t.autograd.grad(scaled_SN, shared_SN, cotangent_SN)
    assert t.equal(grad_SN, scale * cotangent_SN)


def test_moe_rules_change_block_gradients_not_values(tiny_moe, input_ids_Int_1S):
    from jlens.hooks import ActivationRecorder

    def block_gradient() -> t.Tensor:
        with (
            ActivationRecorder(tiny_moe.model.layers, at=[0, 1], start_graph_at=0) as rec,
            t.enable_grad(),
        ):
            tiny_moe.model(input_ids=input_ids_Int_1S, use_cache=False)
            source_BSN, target_BSN = rec.activations[0], rec.activations[1]
        (grad_BSN,) = t.autograd.grad(target_BSN, source_BSN, t.ones_like(target_BSN))
        return grad_BSN

    stock_grad_BSN = block_gradient()
    config = LrpRuleConfig(routed_experts=True, router_detach=True, shared_gate_detach=True)
    with apply_lrp_rules(tiny_moe, config):
        patched_grad_BSN = block_gradient()
    assert not t.allclose(stock_grad_BSN, patched_grad_BSN, rtol=1e-4, atol=1e-7)


class _MockHyperConnection(t.nn.Module):
    """DeepseekV4HyperConnection's attribute layout, tiny and random. The
    stock forward below mirrors the patched forward minus the detaches, so
    the patch's maths is pinned without the transformers class."""

    def __init__(self, hc_mult: int = 2, d: int = 6) -> None:
        super().__init__()
        t.manual_seed(3)
        self.hc_mult = hc_mult
        self.hc_eps = 1e-5
        self.hc_sinkhorn_iters = 3
        self.input_norm = t.nn.LayerNorm(hc_mult * d)
        self.fn = t.nn.Parameter(t.randn(2 * hc_mult + hc_mult * hc_mult, hc_mult * d))
        self.base = t.nn.Parameter(t.randn(2 * hc_mult + hc_mult * hc_mult))
        self.scale = t.nn.Parameter(t.randn(3))

    def forward(self, hidden_streams: t.Tensor):
        hc = self.hc_mult
        flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
        mixed = t.nn.functional.linear(flat, self.fn.float())
        pre_w, post_w, comb_w = mixed[..., :hc], mixed[..., hc : 2 * hc], mixed[..., 2 * hc :]
        pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = self.scale.unbind(0)
        pre = t.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
        post = 2 * t.sigmoid(post_w * post_scale + post_b)
        comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(
            hc, hc
        )
        comb = t.softmax(comb_logits, dim=-1) + self.hc_eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        streams = pre.unsqueeze(-1) * hidden_streams
        collapsed = streams.sum(dim=2).to(hidden_streams.dtype)
        return post, comb, collapsed


# apply_lrp_rules matches mHC patchers by class NAME, so alias the mock.
DeepseekV4HyperConnection = type("DeepseekV4HyperConnection", (_MockHyperConnection,), {})


def test_mhc_patch_is_value_exact_and_detaches_coefficients():
    mock_mhc = DeepseekV4HyperConnection()
    for parameter in mock_mhc.parameters():
        parameter.requires_grad_(False)
    container = t.nn.Sequential(mock_mhc)
    generator = t.Generator().manual_seed(4)
    hidden_streams = t.randn(1, 3, 2, 6, generator=generator, requires_grad=True)

    stock_post, stock_comb, stock_collapsed = mock_mhc(hidden_streams)
    with apply_lrp_rules(container, LrpRuleConfig(mhc_detach=True)) as report:
        patched_post, patched_comb, patched_collapsed = mock_mhc(hidden_streams)
    assert report.mhc == ["0"]
    assert t.allclose(stock_post, patched_post, rtol=1e-6, atol=1e-8)
    assert t.allclose(stock_comb, patched_comb, rtol=1e-6, atol=1e-8)
    assert t.allclose(stock_collapsed, patched_collapsed, rtol=1e-6, atol=1e-8)
    # the detached coefficients carry no gradient; the collapsed content does
    assert patched_post.grad_fn is None and patched_comb.grad_fn is None
    assert patched_collapsed.grad_fn is not None
    assert stock_post.grad_fn is not None
