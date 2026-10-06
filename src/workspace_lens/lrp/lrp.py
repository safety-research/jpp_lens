# The LRP rules in this module are adapted from the R-Lens implementation of Camila Blank and
# Agam Bhatia (see NOTICE).
"""LRP forward-graph surgery: detach-based propagation rules for RelP fitting.

RelP (arXiv:2508.21258) replaces the raw gradient in attribution estimates
with an LRP propagation coefficient. Crucially this is implemented not as a
custom backward but as an *ordinary* autograd backward over a forward graph
with specific sub-expressions detached. :func:`apply_lrp_rules` performs that
surgery reversibly on a stock HF model by swapping ``forward`` on individual
module *instances* (the class is untouched, so other models of the same class
are unaffected):

- **LN-rule** (RMSNorm): detach the ``rsqrt(mean(x^2) + eps)`` factor, so
  backward sees a fixed per-position scaling instead of the variance term
  (the "relevance collapse" guard from CP-LRP, arXiv:2202.07304).
- **Identity-rule** (SiLU): write ``silu(z) = z * sigmoid(z)`` and detach the
  sigmoid, so backward sees a per-element linear map.
- **Half-rule** (gated-MLP product): ``p -> 0.5*p + 0.5*p.detach()``,
  splitting relevance evenly between the gate and up paths instead of
  double-counting through the bilinear product.

Detaching never changes forward *values*: a patched model produces
bit-identical activations (up to ``F.silu``-vs-``z*sigmoid(z)`` kernel dust
when the identity-rule is on) — only gradients differ. So a Jacobian-lens fit
run inside this context estimates the RelP coefficient matrix instead of the
raw mean Jacobian, on the *same* forward pass; routing and recorded
activations are unchanged. The surgery only needs to be live while the graph
is *recorded*: the detached ops persist in the retained graph after the
context exits, so backward passes run later still read RelP coefficients.

Rule scope is selected by :class:`LrpRuleConfig`, usually via a named preset
(:func:`lrp_rule_config_for_mode`, keyed by ``LensConfig.lrp_mode``). The
minimal RelP scope ("rlens") is residual-stream RMSNorms + the gated MLP only;
attention (softmax layers, q/k norms, and the GatedDeltaNet linear-attention
layers) keeps normal gradients. The ``r+moe`` preset extends the dense rules
into the fused routed experts (identity/half per expert, ``router_detach``
freezing the top-k mixing weights) and the MoE block's shared-expert branch
(``shared_gate_detach``; ``shared_expert_scale`` = a bit-exact backward-only
gradient scale). ``shared_expert_forward_scale`` is NOT a propagation rule: it multiplies the
DeepSeek shared expert's output in the forward, so the fitted model is the
amplified one (used by the ``all-c4+mhc`` preset).

Gemma 4 (``Gemma4RMSNorm``, ``Gemma4TextMLP`` with GELU-tanh) and OLMo 3
(``Olmo3RMSNorm`` on the sublayer outputs of its post-norm blocks,
``Olmo3MLP``) are covered too. The identity rule holds for any activation of
the form ``act(z) = z · g(z)`` with the gate ``g`` detached (RelP's "GELU /
SiLU Identity-rule"); :data:`_IDENTITY_RULE_ACTIVATIONS` maps each activation
class to its detached-gate rewrite. Every RMSNorm inside a ``self_attn``
module keeps standard gradients, and a multimodal wrapper's vision and audio
towers and embedders are skipped: the surgery targets the text decoder.
DeepSeek-V4 (``DeepseekV4RMSNorm``, the clamped ``DeepseekV4MLP`` shared
expert, the clamped ``DeepseekV4Experts`` loop, the ``r+mhc`` preset
detaching the hyper-connection coefficients and the ``all-c4+mhc`` preset,
which adds the routed experts and the forward shared-expert scale) is covered;
its unweighted coefficient norm stays stock (:data:`_STOCK_RMSNORM_CLASSES`).

Patchers are per-class and replicate the stock forward exactly — the supported
RMSNorm variants differ in weight placement and dtype casts, so each has its own
verified copy. Unknown RMSNorm or gated-MLP classes raise rather than
silently producing a partly patched lens on a new architecture; further
guards catch silently wrong lenses: a requested rule *family* that
patched nothing (e.g. MoE rules on a dense model), ``torch.compile``d blocks
(compiled forwards inline the stock code and bypass instance patches),
gradient checkpointing (checkpointed blocks re-run their forward during
backward, outside the surgery context), and an importable ``fla`` package
(fused GatedDeltaNet kernels that bypass the module graph).
"""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from types import MethodType
from typing import Any, Optional

import torch as t
from torch import Tensor, nn

from workspace_lens.lrp.types import LRP_MODES

_QK_NORM_NAMES = ("q_norm", "k_norm")
# Gated-MLP classes whose forward is byte-equivalent to
# down(act(gate(x)) * up(x)). Qwen3.6 checkpoints reuse the Qwen3.5 classes.
# NOTE: routed MoE experts live in fused-3D-parameter modules (no
# gate_proj/up_proj attributes) and are matched separately under the
# routed_experts flag; at dense scope they keep standard gradients.
_KNOWN_GATED_MLPS = (
    "Qwen3_5MLP",
    "Qwen2MLP",
    "Qwen3MLP",
    "Qwen3_5MoeMLP",
    "Qwen3MoeMLP",
    "Olmo3MLP",  # SiLU; forward byte-identical to Qwen3_5MLP's (transformers 5.14.1)
    "Gemma4TextMLP",  # GELU-tanh; same down(act(gate(x)) * up(x)) forward
    # DeepSeek-V4's dense / shared-expert MLP: down(act(clamp(gate(x))) * clamp(up(x)))
    # with a ``limit`` attribute (swiglu_limit); the patcher reproduces the clamps.
    "DeepseekV4MLP",
)
# Norm classes the walker leaves stock on purpose. DeepSeek-V4's unweighted RMSNorm
# normalises the flattened residual streams only to compute the hyper-connection
# mixing coefficients (and the head's collapse weights); it is not a residual-stream
# norm, and under ``mhc_detach`` the coefficients it feeds are detached anyway.
_STOCK_RMSNORM_CLASSES = ("DeepseekV4UnweightedRMSNorm",)
# nn.SiLU or transformers' ACT2FN["silu"] wrapper — both compute F.silu.
_SILU_CLASS_NAMES = ("SiLU", "SiLUActivation")
# Fused routed experts (3D gate_up/down parameters, per-expert SwiGLU loop)
# and the MoE block holding the gated shared expert.
# DeepSeek-V4's experts run the same per-expert loop with the swiglu_limit clamps (its
# ``limit`` attribute), which the patcher reproduces through clamp_swiglu_inputs.
_KNOWN_FUSED_EXPERTS = ("Qwen3_5MoeExperts", "Qwen3MoeExperts", "DeepseekV4Experts")
_KNOWN_MOE_BLOCKS = ("Qwen3_5MoeSparseMoeBlock",)
# Gated MLPs that ``shared_expert_forward_scale`` multiplies in the FORWARD: DeepSeek-V4's shared expert
# (``DeepseekV4MLP`` is instantiated only as ``shared_experts``; the routed experts are the fused class above).
_FORWARD_SCALED_MLP_CLASSES = ("DeepseekV4MLP",)
# DeepSeek-V4's MoE block itself is never patched: its shared expert has no gate, and
# ``shared_expert_forward_scale`` is applied by the gated-MLP patcher on ``DeepseekV4MLP``.
# DeepSeek-V4 mHC residual mixing (the ``r+mhc`` preset; value-exact against
# transformers 5.14.1 on the tiny DeepSeek fixture, tests/test_lrp_deepseek.py).
_KNOWN_MHC_CLASSES = ("DeepseekV4HyperConnection", "DeepseekV4HyperHead")
_MISSING = object()


def patch_instance_forward(
    saved_forwards: list[tuple[nn.Module, object]],
    module: nn.Module,
    patched_forward: Any,
) -> None:
    """Shadow ``forward`` on this module *instance* (the class is untouched, so
    other models of the same class are unaffected), recording what
    :func:`restore_instance_forwards` must put back. ``object.__setattr__``
    bypasses ``nn.Module``'s setattr bookkeeping."""
    saved_forwards.append((module, module.__dict__.get("forward", _MISSING)))
    object.__setattr__(module, "forward", MethodType(patched_forward, module))


def restore_instance_forwards(saved_forwards: list[tuple[nn.Module, object]]) -> None:
    """Undo :func:`patch_instance_forward` in LIFO order."""
    for module, original in reversed(saved_forwards):
        if original is _MISSING:
            module.__dict__.pop("forward", None)
        else:
            object.__setattr__(module, "forward", original)


@dataclass(frozen=True)
class LrpRuleConfig:
    """Which LRP rules to apply. Defaults are all-off; use
    :func:`lrp_rule_config_for_mode` for the named presets."""

    ln_rule: bool = False  # detach the RMSNorm rsqrt factor (residual-stream norms)
    identity_rule: bool = False  # detach the sigmoid factor inside SiLU
    half_rule: bool = False  # half-detach the gated-MLP product
    # --- MoE flags ---
    routed_experts: bool = False  # identity/half rules inside fused routed experts
    router_detach: bool = False  # detach top_k_weights in the expert mixing
    shared_gate_detach: bool = False  # detach sigmoid(shared_expert_gate(x))
    shared_expert_scale: float = 1.0  # backward-only grad scale on the shared branch
    # --- DeepSeek flags ---
    mhc_detach: bool = False  # detach DeepSeek-V4 mHC residual-mixing coefficients
    # FORWARD multiply on the DeepSeek shared expert's output (c * y): an activation-changing
    # intervention, not value-preserving; the fitted Jacobians are those of the amplified model.
    shared_expert_forward_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.router_detach and not self.routed_experts:
            raise ValueError(
                "router_detach requires routed_experts=True — it lives inside the "
                "patched experts forward"
            )
        if self.shared_expert_scale < 0:
            raise ValueError(
                "shared_expert_scale must be >= 0 (0 detaches the branch, 1 is stock)"
            )
        if self.shared_expert_forward_scale < 0:
            raise ValueError("shared_expert_forward_scale must be >= 0 (1 is stock)")

    def requests_any_rule(self) -> bool:
        return (
            self.ln_rule
            or self.identity_rule
            or self.half_rule
            or self.routed_experts
            or self.shared_gate_detach
            or self.shared_expert_scale != 1.0
            or self.mhc_detach
            or self.shared_expert_forward_scale != 1.0
        )

    def to_dict(self) -> dict[str, bool | float]:
        return asdict(self)


def lrp_rule_config_for_mode(lrp_mode: str) -> LrpRuleConfig:
    """The preset :class:`LrpRuleConfig` behind each ``LensConfig.lrp_mode``.

    The ``rlens`` preset is the minimal RelP scope (LN + identity +
    half). The ``r+moe`` preset adds MoE rules: routed experts treated as
    dense MLPs, the router detached, the shared-expert gate detached, and a
    backward-only gradient scale of 4 on the shared-expert branch.
    ``all-c4+mhc`` (DeepSeek-V4) applies the minimal RelP rules, the routed
    experts under the identity and half rules, the router detached, the
    hyper-connection detach, and a FORWARD x4 on the shared expert's output
    (``shared_expert_forward_scale``); DeepSeek-V4's shared expert has no gate,
    and no backward scale is applied.
    """
    base = {"ln_rule": True, "identity_rule": True, "half_rule": True}
    moe = {
        "routed_experts": True,
        "router_detach": True,
        "shared_gate_detach": True,
        "shared_expert_scale": 4.0,
    }
    presets: dict[str, LrpRuleConfig] = {
        "none": LrpRuleConfig(),
        "rlens": LrpRuleConfig(**base),
        # DeepSeek-V4: the minimal RelP rules plus a freeze of the hyper-connection
        # mixing coefficients.
        "r+mhc": LrpRuleConfig(**base, mhc_detach=True),
        # DeepSeek-V4: the minimal RelP rules, the routed experts under the same identity and half
        # rules, the router detached, the hyper-connection detach, and a FORWARD x4 on the shared
        # expert's output. tests/test_lrp.py pins these flags against the rule flags recorded in the
        # provenance of camilablank/workspace-lenses deepseek-v4-flash/r-lens/lens.pt ("arm": "all-c4"),
        # reading that file's "shared_expert_scale" as shared_expert_forward_scale.
        "all-c4+mhc": LrpRuleConfig(
            **base, routed_experts=True, router_detach=True, mhc_detach=True, shared_expert_forward_scale=4.0
        ),
        "r+moe": LrpRuleConfig(**base, **moe),
    }
    if lrp_mode not in presets:
        raise ValueError(f"unknown lrp_mode {lrp_mode!r}; expected one of {LRP_MODES}")
    return presets[lrp_mode]


# The rule families a patch report lists, one attribute each (below).
PATCH_REPORT_FAMILIES = (
    "ln_rule",
    "mlp_rule",
    "moe_experts",
    "moe_gates",
    "mhc",
    "shared_expert_forward_scale",
)


@dataclass
class LrpPatchReport:
    """Qualified names of every module each rule was applied to (for logs and
    checkpoint provenance), one list per family of :data:`PATCH_REPORT_FAMILIES`."""

    lrp_config: LrpRuleConfig
    ln_rule: list[str] = field(default_factory=list)
    mlp_rule: list[str] = field(default_factory=list)
    moe_experts: list[str] = field(default_factory=list)
    moe_gates: list[str] = field(default_factory=list)
    mhc: list[str] = field(default_factory=list)
    shared_expert_forward_scale: list[str] = field(default_factory=list)  # the forward-scaled shared experts

    def patched_names(self) -> dict[str, list[str]]:
        """``{family: names}`` over :data:`PATCH_REPORT_FAMILIES`."""
        return {family: list(getattr(self, family)) for family in PATCH_REPORT_FAMILIES}

    def patched_any(self) -> bool:
        """True when any family (:data:`PATCH_REPORT_FAMILIES`) recorded a patch."""
        return any(self.patched_names().values())


### PER-CLASS VALUE-EXACT PATCHED FORWARDS
# The local names inside these forwards (x_f, scale, z, act, hc, flat, mixed,
# pre_w, comb, streams, collapsed, ...) deliberately mirror the stock HF
# forwards line-for-line so the two can be diffed; do not rename them.


def _qwen3_5_rmsnorm_lrp_forward(self: Any, x: Tensor) -> Tensor:
    # Stock: (x_f * rsqrt(mean(x_f^2) + eps)) * (1 + w_f), then type_as(x).
    x_f = x.float()
    scale = t.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps).detach()
    output: Tensor = x_f * scale * (1.0 + self.weight.float())
    return output.type_as(x)


def _qwen2_rmsnorm_lrp_forward(self: Any, hidden_states: Tensor) -> Tensor:
    # Stock: w * (x_f * rsqrt(mean(x_f^2) + eps)).to(input_dtype) — weight
    # applied after the downcast, eps attr named variance_epsilon.
    input_dtype = hidden_states.dtype
    x_f = hidden_states.to(t.float32)
    scale = t.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.variance_epsilon).detach()
    result: Tensor = self.weight * (x_f * scale).to(input_dtype)
    return result


def _olmo3_rmsnorm_lrp_forward(self: Any, hidden_states: Tensor) -> Tensor:
    # Stock: (w * (x_f * rsqrt(mean(x_f^2) + eps))).to(input_dtype) — weight
    # applied in fp32 BEFORE the downcast (Qwen2 casts first), eps attr named
    # variance_epsilon.
    input_dtype = hidden_states.dtype
    x_f = hidden_states.to(t.float32)
    scale = t.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.variance_epsilon).detach()
    result: Tensor = (self.weight * (x_f * scale)).to(input_dtype)
    return result


def _gemma4_rmsnorm_lrp_forward(self: Any, hidden_states: Tensor) -> Tensor:
    # Stock: x_f * pow(mean(x_f^2) + eps, -0.5), then * w_f when with_scale
    # (plain weight, not 1 + w; Gemma 4 stores the full scale), then type_as.
    # Norms built with with_scale=False (Gemma's v_norm, tower norms) have no
    # weight attribute at all.
    x_f = hidden_states.float()
    scale = t.pow(x_f.pow(2).mean(-1, keepdim=True) + self.eps, -0.5).detach()
    normed_f = x_f * scale
    if self.with_scale:
        normed_f = normed_f * self.weight.float()
    output: Tensor = normed_f.type_as(hidden_states)
    return output


# Per-class LN-rule forwards. The variants differ in weight placement ((1+w)
# in fp32, w after the downcast, w in fp32 before the downcast, optional plain
# w) and eps attribute name; each entry was verified value-exact against its
# stock forward (pinned in test_lrp.py and test_lrp_gemma_olmo.py).
_RMSNORM_LRP_FORWARDS = {
    "Qwen3_5RMSNorm": _qwen3_5_rmsnorm_lrp_forward,
    # Value-identical family: zero-init (1+w), same eps attr, same cast order.
    "Qwen3_5MoeRMSNorm": _qwen3_5_rmsnorm_lrp_forward,
    "Qwen2RMSNorm": _qwen2_rmsnorm_lrp_forward,
    "Qwen3RMSNorm": _qwen2_rmsnorm_lrp_forward,  # same forward as Qwen2's
    "Qwen3MoeRMSNorm": _qwen2_rmsnorm_lrp_forward,  # byte-identical to Qwen2's
    "DeepseekV4RMSNorm": _qwen2_rmsnorm_lrp_forward,  # byte-identical to Qwen2's
    "Olmo3RMSNorm": _olmo3_rmsnorm_lrp_forward,
    "Gemma4RMSNorm": _gemma4_rmsnorm_lrp_forward,
}


def _silu_with_detached_sigmoid(z: Tensor) -> Tensor:
    """SiLU with its sigmoid factor treated as a constant (the identity
    rule). Forward bit-exact vs silu only where the kernel computes
    z*sigmoid(z) the same way — value-equal up to kernel dust otherwise."""
    return z * t.sigmoid(z).detach()


def _gelu_tanh_with_detached_gate(z: Tensor) -> Tensor:
    """GELU (tanh approximation) written as ``z * gate(z)`` with the gate
    ``0.5 (1 + tanh(√(2/π)(z + 0.044715 z³)))`` treated as a constant: the
    identity rule for Gemma's ``gelu_pytorch_tanh`` MLPs, RelP's "GELU / SiLU
    Identity-rule" applied to the tanh form. Value-equal to
    ``F.gelu(z, approximate="tanh")`` up to kernel dust, like the SiLU rewrite."""
    gate = 0.5 * (1.0 + t.tanh(math.sqrt(2.0 / math.pi) * (z + 0.044715 * z.pow(3))))
    return z * gate.detach()


# Identity-rule rewrite per activation class: the detached-gate form of the
# activation the gated MLP was built with. "GELUTanh" is transformers'
# ACT2FN["gelu_pytorch_tanh"] wrapper (F.gelu(z, approximate="tanh")).
_IDENTITY_RULE_ACTIVATIONS: dict[str, Callable[[Tensor], Tensor]] = {
    "SiLU": _silu_with_detached_sigmoid,
    "SiLUActivation": _silu_with_detached_sigmoid,
    "GELUTanh": _gelu_tanh_with_detached_gate,
}


def _halve_gradient(x: Tensor) -> Tensor:
    """Forward bit-exact identity with gradient x0.5 (0.5*y is an exact fp
    scaling and the two equal halves sum exactly)."""
    return 0.5 * x + 0.5 * x.detach()


def clamp_swiglu_inputs(gate: Tensor, up: Tensor, limit: Optional[float]) -> tuple[Tensor, Tensor]:  # noqa: UP045
    """The ``swiglu_limit`` clamps of DeepSeek-V4's MLPs (gate at most ``limit``, up
    within ``±limit``), as their stock forwards apply them; a no-op when ``limit`` is None."""
    if limit is None:
        return gate, up
    return gate.clamp(max=limit), up.clamp(min=-limit, max=limit)


def _make_gated_mlp_lrp_forward(
    identity_rule: bool,
    half_rule: bool,
    detached_gate_activation: Optional[Callable[[Tensor], Tensor]],  # noqa: UP045
    forward_scale: float = 1.0,
) -> Any:
    """The gated-MLP forward with the identity rule (``detached_gate_activation``,
    the module's entry in :data:`_IDENTITY_RULE_ACTIVATIONS`, in place of
    ``act_fn``; None, and never called, when ``identity_rule`` is off) and the
    half rule. A module with a ``limit`` attribute (DeepSeek-V4's MLP) clamps
    the gate at ``limit`` and the up branch at ``±limit`` before the product,
    exactly as its stock forward does; the identity rule then acts on the
    clamped pre-activation. ``forward_scale`` (the DeepSeek shared expert's
    ``shared_expert_forward_scale``) multiplies the OUTPUT in the forward: at 1
    the forward is bit-identical to stock, at any other value the module's
    contribution to the residual is amplified (an intervention, not a rule)."""

    def forward(self: Any, x: Tensor) -> Tensor:
        z, up = clamp_swiglu_inputs(self.gate_proj(x), self.up_proj(x), getattr(self, "limit", None))
        act = detached_gate_activation(z) if identity_rule else self.act_fn(z)
        product = act * up
        if half_rule:
            product = _halve_gradient(product)
        result: Tensor = self.down_proj(product)
        if forward_scale != 1.0:
            result = forward_scale * result
        return result

    return forward


def _make_moe_experts_lrp_forward(
    identity_rule: bool, half_rule: bool, router_detach: bool
) -> Any:
    """The reference fused-experts loop (Qwen3.5/Qwen3-MoE, and DeepSeek-V4 whose
    experts add the swiglu_limit clamps of their ``limit`` attribute) with the dense
    MLP rules applied inside each expert and, under ``router_detach``, the
    top-k mixing weights treated as constants (the CP-LRP gate treatment;
    expert *selection* is already non-differentiable).

    NOTE: unlike every other patcher this is value-equal only up to
    reduction-order dust (~1 ulp), not bit-exact — the stock class-level
    ``forward`` is a transformers dispatch wrapper that may route to a
    grouped/batched implementation rather than this eager loop. Routing
    happens upstream of the experts, so cluster assignments are unaffected."""

    def forward(
        self: Any, hidden_states: Tensor, top_k_index: Tensor, top_k_weights: Tensor
    ) -> Tensor:
        if router_detach:
            top_k_weights = top_k_weights.detach()
        final_hidden_states = t.zeros_like(hidden_states)
        swiglu_limit = getattr(self, "limit", None)  # DeepSeek-V4's experts clamp; Qwen's have no limit
        with t.no_grad():
            expert_mask = nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = t.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = t.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(
                current_state, self.gate_up_proj[expert_idx]
            ).chunk(2, dim=-1)
            gate, up = clamp_swiglu_inputs(gate, up, swiglu_limit)
            act = _silu_with_detached_sigmoid(gate) if identity_rule else self.act_fn(gate)
            product = act * up
            if half_rule:
                product = _halve_gradient(product)
            current_hidden_states = nn.functional.linear(product, self.down_proj[expert_idx])
            current_hidden_states = (
                current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            )
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )
        return final_hidden_states

    return forward


def _make_moe_block_lrp_forward(shared_gate_detach: bool, shared_expert_scale: float) -> Any:
    """Value-exact replica of Qwen3_5MoeSparseMoeBlock.forward with two knobs
    on the shared-expert branch: ``shared_gate_detach`` detaches the sigmoid
    gate factor (content path keeps gradients), and ``shared_expert_scale``
    multiplies the GRADIENT through the post-gate shared contribution by c via
    ``s + (c-1)*(s - s.detach())`` — the correction term is exactly zero in
    the forward, so values are bit-identical for any c."""

    def forward(self: Any, hidden_states: Tensor) -> Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        shared_expert_output: Tensor = self.shared_expert(hidden_states_reshaped)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        expert_output = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
        gate_sigmoid = t.sigmoid(self.shared_expert_gate(hidden_states_reshaped))
        if shared_gate_detach:
            gate_sigmoid = gate_sigmoid.detach()
        shared_expert_output = gate_sigmoid * shared_expert_output
        if shared_expert_scale != 1.0:
            shared_expert_output = shared_expert_output + (shared_expert_scale - 1.0) * (
                shared_expert_output - shared_expert_output.detach()
            )
        expert_output = expert_output + shared_expert_output
        result: Tensor = expert_output.reshape(batch_size, sequence_length, hidden_dim)
        return result

    return forward


def _deepseek_hyperconnection_lrp_forward(
    self: Any, hidden_streams: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Value-exact DeepseekV4HyperConnection.forward with the three mHC
    coefficients detached (pre collapses the parallel streams, post re-places
    the sublayer output, comb re-mixes the streams)."""
    hc = self.hc_mult
    flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
    pre_w, post_w, comb_w = nn.functional.linear(flat, self.fn.float()).split([hc, hc, hc * hc], dim=-1)
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
    streams = pre.detach().unsqueeze(-1) * hidden_streams
    collapsed = streams.sum(dim=2).to(hidden_streams.dtype)
    return post.detach(), comb.detach(), collapsed


def _deepseek_hyperhead_lrp_forward(self: Any, x: Tensor) -> Tensor:
    """Value-exact DeepseekV4HyperHead.forward (final stream collapse) with
    its pre gate detached."""
    flat = self.input_norm(x.flatten(2).float())
    mixes = nn.functional.linear(flat, self.hc_fn.float())
    pre = t.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float()) + self.eps
    result: Tensor = (pre.detach().unsqueeze(-1) * x).sum(dim=2).to(x.dtype)
    return result


### MODULE MATCHING


def _is_gated_mlp(module: nn.Module) -> bool:
    return all(
        hasattr(module, attr) for attr in ("gate_proj", "up_proj", "down_proj", "act_fn")
    )


def _is_qk_norm(qualified_name: str) -> bool:
    return qualified_name.rsplit(".", 1)[-1] in _QK_NORM_NAMES


_MULTIMODAL_MODULE_SEGMENTS = frozenset(
    {"vision_tower", "audio_tower", "embed_vision", "embed_audio"}
)


def is_attention_internal_norm(qualified_name: str) -> bool:
    """Norms kept OUT of the LN-rule: any RMSNorm inside a ``self_attn`` module
    is attention-internal (DeepSeek-V4's q_a/kv norms, Gemma 4's ``v_norm``, and
    the q/k norms, which :func:`_is_qk_norm` also matches); the residual-stream
    norms are never under ``self_attn``. Matched on path segments, so the rule
    holds whether the walk starts at a wrapper
    (``model.language_model.layers.0.self_attn.v_norm``) or at a bare decoder
    (``layers.0.self_attn.v_norm``)."""
    return "self_attn" in qualified_name.split(".")


def is_multimodal_tower_module(qualified_name: str) -> bool:
    """Modules of a multimodal wrapper that the text forward never runs (Gemma
    4's ``vision_tower``, ``audio_tower`` and the ``embed_vision`` /
    ``embed_audio`` projectors): the surgery skips them so their norms and MLPs
    (the vision MLP has the gated-MLP attribute shape) neither need patchers nor
    count toward the per-family guard. Matched on path segments."""
    return not _MULTIMODAL_MODULE_SEGMENTS.isdisjoint(qualified_name.split("."))


def _find_compiled_block(model: nn.Module) -> str | None:
    """Qualified name of the first ``torch.compile``d submodule, or None."""
    optimized_module_cls = getattr(
        getattr(t._dynamo, "eval_frame", None), "OptimizedModule", ()
    )
    for name, module in model.named_modules():
        if optimized_module_cls and isinstance(module, optimized_module_cls):
            return name or "<root>"
    return None


### THE SURGERY CONTEXT MANAGER


def hf_module_to_patch(model: object) -> nn.Module:
    """The module the surgery patches: ``model._hf_model`` when ``model`` is
    the HF :class:`LensModel` adapter, else ``model`` itself.

    Raises:
        TypeError: If neither is an ``nn.Module``.
    """
    module_to_patch = getattr(model, "_hf_model", model)
    if not isinstance(module_to_patch, nn.Module):
        raise TypeError(
            f"cannot apply LRP surgery: {type(model).__name__} is not an "
            "nn.Module and exposes no _hf_model"
        )
    return module_to_patch


@contextmanager
def apply_lrp_rules_for_mode(model: object, lrp_mode: str) -> Iterator[LrpPatchReport | None]:
    """:func:`apply_lrp_rules` with the standard entry ritual: resolve the
    preset from ``lrp_mode``, no-op for ``"none"``, and unwrap a
    :class:`LensModel` to its HF module (``model._hf_model``) when present.
    Yields the patch report, or ``None`` when no rules are requested."""
    rule_config = lrp_rule_config_for_mode(lrp_mode)
    if not rule_config.requests_any_rule():
        yield None
        return
    with apply_lrp_rules(hf_module_to_patch(model), rule_config) as report:
        yield report


@contextmanager
def apply_lrp_rules(model: nn.Module, lrp_config: LrpRuleConfig) -> Iterator[LrpPatchReport]:
    """Reversibly patch the module instances each requested rule applies to
    (RMSNorm, gated MLP, MoE experts and blocks, DeepSeek mHC) so that
    ordinary autograd computes RelP coefficients.

    Yields a :class:`LrpPatchReport` naming every patched module. On exit
    every instance ``forward`` is restored, whatever happened inside the
    block; graphs recorded while the patches were live keep the detached ops.

    Raises:
        RuntimeError: On a matched class with no exact patcher or an
            unexpected attribute layout, on a ``torch.compile``d or
            gradient-checkpointed model, on an fla-capable environment, or
            when rules were requested but no module matched (each would
            silently build a wrong artifact).
    """
    # transformers silently switches GatedDeltaNet layers to fla kernels when
    # the library is importable; those kernels bypass module-level graph
    # structure the surgery relies on. This guard lives here so every entry
    # point gets it, not just the trainer.
    if lrp_config.requests_any_rule() and importlib.util.find_spec("fla") is not None:
        raise RuntimeError(
            "the `fla` package is importable, so transformers may route "
            "GatedDeltaNet through fused kernels that bypass the LRP "
            "surgery — run RelP fits in an environment without fla"
        )
    if getattr(model, "is_gradient_checkpointing", False):
        raise RuntimeError(
            "model has gradient checkpointing enabled — checkpointed blocks "
            "re-run their forward during backward, outside the surgery "
            "context, silently producing standard gradients. Disable it "
            "(model.gradient_checkpointing_disable())."
        )

    compiled_block = _find_compiled_block(model)
    if compiled_block is not None:
        raise RuntimeError(
            f"{compiled_block} is a torch.compile'd module — compiled forwards "
            "bypass instance-level patches, silently disabling the LRP rules. "
            "Load the model without compile."
        )

    saved_forwards: list[tuple[nn.Module, object]] = []

    def _patch(module: nn.Module, patched_forward: Any) -> None:
        patch_instance_forward(saved_forwards, module, patched_forward)

    report = LrpPatchReport(lrp_config=lrp_config)
    try:
        for name, module in model.named_modules():
            if is_multimodal_tower_module(name):
                continue  # a wrapper's vision/audio towers: never on the text forward
            cls = type(module).__name__
            if cls in _KNOWN_MHC_CLASSES:
                if not lrp_config.mhc_detach:
                    continue  # default: mHC coefficients keep standard gradients
                mhc_forward = (
                    _deepseek_hyperconnection_lrp_forward
                    if cls == "DeepseekV4HyperConnection"
                    else _deepseek_hyperhead_lrp_forward
                )
                _patch(module, mhc_forward)
                report.mhc.append(name)
            elif cls.endswith("RMSNorm"):
                if not lrp_config.ln_rule or cls in _STOCK_RMSNORM_CLASSES:
                    continue
                if _is_qk_norm(name) or is_attention_internal_norm(name):
                    continue  # attention-internal norms keep standard gradients
                rmsnorm_forward = _RMSNORM_LRP_FORWARDS.get(cls)
                if rmsnorm_forward is None:
                    raise RuntimeError(
                        f"No LN-rule patcher for RMSNorm class {cls!r} at {name!r} "
                        "— add a value-exact forward to _RMSNORM_LRP_FORWARDS "
                        "before building."
                    )
                _patch(module, rmsnorm_forward)
                report.ln_rule.append(name)
            elif cls in _KNOWN_FUSED_EXPERTS:
                if not lrp_config.routed_experts:
                    continue  # minimal RelP scope: routed experts keep standard grads
                required_attributes = ("gate_up_proj", "down_proj", "act_fn", "num_experts")
                if not all(hasattr(module, attr) for attr in required_attributes):
                    raise RuntimeError(
                        f"fused-experts class {cls!r} at {name!r} lacks one of "
                        f"{required_attributes} — verify its forward matches the Qwen3.5-MoE "
                        "expert loop before registering."
                    )
                if (
                    lrp_config.identity_rule
                    and type(module.act_fn).__name__ not in _SILU_CLASS_NAMES
                ):
                    raise RuntimeError(
                        f"Identity-rule expects SiLU, found "
                        f"{type(module.act_fn).__name__} at {name!r}."
                    )
                _patch(
                    module,
                    _make_moe_experts_lrp_forward(
                        lrp_config.identity_rule, lrp_config.half_rule, lrp_config.router_detach
                    ),
                )
                report.moe_experts.append(name)
            elif cls in _KNOWN_MOE_BLOCKS:
                if not (lrp_config.shared_gate_detach or lrp_config.shared_expert_scale != 1.0):
                    continue  # shared-expert gate + content keep stock grads
                required_attributes = ("shared_expert", "shared_expert_gate", "experts", "gate")
                if not all(hasattr(module, attr) for attr in required_attributes):
                    raise RuntimeError(
                        f"MoE block class {cls!r} at {name!r} lacks one of "
                        f"{required_attributes} — verify its forward matches the Qwen3.5-MoE "
                        "block before registering."
                    )
                _patch(
                    module,
                    _make_moe_block_lrp_forward(
                        lrp_config.shared_gate_detach, lrp_config.shared_expert_scale
                    ),
                )
                report.moe_gates.append(name)
            elif _is_gated_mlp(module):
                # The forward scale is DeepSeek-V4's shared-expert intervention; every DeepseekV4MLP in the
                # decoder is a shared expert (each block is an MoE block).
                forward_scale = lrp_config.shared_expert_forward_scale if cls in _FORWARD_SCALED_MLP_CLASSES else 1.0
                if not (lrp_config.identity_rule or lrp_config.half_rule or forward_scale != 1.0):
                    continue
                if cls not in _KNOWN_GATED_MLPS:
                    raise RuntimeError(
                        f"Unknown gated-MLP class {cls!r} at {name!r} — verify its "
                        "forward matches down(act(gate(x)) * up(x)) and add it to "
                        "_KNOWN_GATED_MLPS."
                    )
                activation_class_name = type(module.act_fn).__name__
                detached_gate_activation = _IDENTITY_RULE_ACTIVATIONS.get(activation_class_name)
                if lrp_config.identity_rule and detached_gate_activation is None:
                    raise RuntimeError(
                        f"Identity-rule expects SiLU or GELU-tanh, found "
                        f"{activation_class_name} at {name!r} — the detached-gate "
                        f"rewrites cover {sorted(_IDENTITY_RULE_ACTIVATIONS)} only."
                    )
                _patch(
                    module,
                    _make_gated_mlp_lrp_forward(
                        lrp_config.identity_rule, lrp_config.half_rule, detached_gate_activation, forward_scale
                    ),
                )
                if lrp_config.identity_rule or lrp_config.half_rule:
                    report.mlp_rule.append(name)
                if forward_scale != 1.0:
                    report.shared_expert_forward_scale.append(name)

        # Per-family guard: every requested rule family must have patched at
        # least one module, else that family silently degrades to standard
        # gradients (e.g. MoE rules on a dense model, LN-rule on a model with
        # no RMSNorms). Extended alongside each new family of patchers.
        rule_families = [
            ("ln_rule", lrp_config.ln_rule, report.ln_rule),
            (
                "identity_rule/half_rule",
                lrp_config.identity_rule or lrp_config.half_rule,
                report.mlp_rule,
            ),
            ("routed_experts", lrp_config.routed_experts, report.moe_experts),
            (
                "shared_gate_detach/shared_expert_scale",
                lrp_config.shared_gate_detach or lrp_config.shared_expert_scale != 1.0,
                report.moe_gates,
            ),
            ("mhc_detach", lrp_config.mhc_detach, report.mhc),
            (
                "shared_expert_forward_scale",
                lrp_config.shared_expert_forward_scale != 1.0,
                report.shared_expert_forward_scale,
            ),
        ]
        unmatched_families = [
            family for family, requested, patched in rule_families if requested and not patched
        ]
        if unmatched_families:
            raise RuntimeError(
                f"LRP rule families {unmatched_families} were requested but no "
                f"module matched their patchers on {type(model).__name__} — the "
                "fit would silently use standard gradients for them."
            )
        yield report
    finally:
        restore_instance_forwards(saved_forwards)
