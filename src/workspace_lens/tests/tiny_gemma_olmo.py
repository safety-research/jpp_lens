"""Tiny CPU-only Gemma 3, Gemma 4 and OLMo 3 text models for the LRP-surgery tests.

Like :mod:`workspace_lens.tests.tiny_qwen`, these build the real transformers
classes from shrunk configs, so the module classes the LRP surgery patches on
the full models — ``Gemma3RMSNorm`` (zero-init 1 + w, as Qwen3.5),
``Gemma3MLP`` (GELU-tanh), ``Gemma4RMSNorm`` (plain weight, optional
``with_scale``), ``Gemma4TextMLP`` (GELU-tanh), ``Olmo3RMSNorm`` (weight in
fp32 before the cast, on the sublayer outputs of a post-norm block) and
``Olmo3MLP`` (SiLU) — exercise the per-class patchers for real. Gemma's
attention keeps its ``q_norm``, ``k_norm`` and (Gemma 4) weightless
``v_norm``, which the LN-rule must leave alone.

Two fixtures per Gemma: the text-only ``*ForCausalLM`` (module names
``model.layers.<i>...``) for the rule tests, and the multimodal
``*ForConditionalGeneration`` the full ``google/gemma-4-31B`` and
``google/gemma-3-27b-it`` load as, whose text decoder sits at
``model.language_model`` next to modules the surgery must skip by path segment
(:func:`workspace_lens.lrp.lrp.is_multimodal_tower_module`): Gemma 4's vision
and audio towers and embedders (the vision MLP has the gated-MLP attribute
shape and would otherwise raise), and Gemma 3's SigLIP tower and
``multi_modal_projector``, whose ``mm_soft_emb_norm`` is a ``Gemma3RMSNorm``.

The Gemma 4 text settings below (``hidden_size_per_layer_input=0``,
``num_kv_shared_layers=0``, ``enable_moe_block`` off, GELU-tanh) match the
published ``google/gemma-4-31B`` config.
"""

from __future__ import annotations

import torch as t
from torch import nn
from transformers import AutoConfig
from transformers.models.gemma3 import Gemma3ForCausalLM, Gemma3ForConditionalGeneration
from transformers.models.gemma3.configuration_gemma3 import Gemma3Config
from transformers.models.gemma4 import Gemma4ForCausalLM, Gemma4ForConditionalGeneration
from transformers.models.gemma4.configuration_gemma4 import Gemma4Config
from transformers.models.olmo3 import Olmo3ForCausalLM

# The input-id and last-hidden-state helpers are shared with the tiny Qwen fixture
# (workspace_lens.tests.tiny_qwen: random_tiny_qwen_input_ids, last_hidden_state_BSN);
# both vocabularies are 128 wide.
from workspace_lens.tests.tiny_qwen import TINY_QWEN_VOCAB_SIZE

TINY_D_MODEL = 32
# Three blocks: two sliding-window layers then one full-attention layer.
_TINY_ATTENTION_CONFIG_KWARGS = dict(
    hidden_size=TINY_D_MODEL,
    intermediate_size=64,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    vocab_size=TINY_QWEN_VOCAB_SIZE,
    max_position_embeddings=256,
    sliding_window=16,
    layer_types=["sliding_attention", "sliding_attention", "full_attention"],
)
_TINY_GEMMA3_TEXT_CONFIG_KWARGS = dict(
    **_TINY_ATTENTION_CONFIG_KWARGS,
    hidden_activation="gelu_pytorch_tanh",
)
_TINY_GEMMA4_TEXT_CONFIG_KWARGS = dict(
    **_TINY_GEMMA3_TEXT_CONFIG_KWARGS,
    hidden_size_per_layer_input=0,
    vocab_size_per_layer_input=TINY_QWEN_VOCAB_SIZE,
    num_kv_shared_layers=0,
    use_double_wide_mlp=False,
)
# A one-layer SigLIP tower for the Gemma 3 wrapper;
# Gemma 4's vision config takes the same keys.
_TINY_VISION_CONFIG_KWARGS = dict(
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=1,
    num_attention_heads=4,
    image_size=32,
    patch_size=16,
)


def _randomly_initialised(model_class: type[nn.Module], config, seed: int = 0) -> nn.Module:
    """``model_class(config)`` from a fixed seed, in eval mode with parameter grads off."""
    t.manual_seed(seed)
    model = model_class(config).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def build_tiny_gemma3() -> Gemma3ForCausalLM:
    """A tiny Gemma 3 text model on CPU: GELU-tanh gated MLP, four
    ``Gemma3RMSNorm`` per block plus the final norm, q/k norms in attention."""
    config = AutoConfig.for_model("gemma3_text", **_TINY_GEMMA3_TEXT_CONFIG_KWARGS)
    return _randomly_initialised(Gemma3ForCausalLM, config)  # type: ignore[return-value]


def build_tiny_gemma3_multimodal() -> Gemma3ForConditionalGeneration:
    """The same tiny text decoder inside the multimodal wrapper, with a one-layer
    SigLIP tower (LayerNorms and fc1/fc2 MLPs, nothing the surgery matches) and
    the ``multi_modal_projector``, whose ``mm_soft_emb_norm`` is a
    ``Gemma3RMSNorm`` of the residual width: the module tree the real 27B presents."""
    config = Gemma3Config(
        text_config=_TINY_GEMMA3_TEXT_CONFIG_KWARGS,
        vision_config=_TINY_VISION_CONFIG_KWARGS,
        mm_tokens_per_image=4,
    )
    return _randomly_initialised(Gemma3ForConditionalGeneration, config)  # type: ignore[return-value]


def build_tiny_gemma4() -> Gemma4ForCausalLM:
    """A tiny Gemma 4 text model on CPU: GELU-tanh gated MLP, four
    ``Gemma4RMSNorm`` per block plus the final norm, q/k/v norms in attention,
    no per-layer input embeddings and no kv-shared layers."""
    config = AutoConfig.for_model("gemma4_text", **_TINY_GEMMA4_TEXT_CONFIG_KWARGS)
    return _randomly_initialised(Gemma4ForCausalLM, config)  # type: ignore[return-value]


def build_tiny_gemma4_multimodal() -> Gemma4ForConditionalGeneration:
    """The same tiny text decoder inside the multimodal wrapper, with a one-layer
    vision tower (its MLP has gate_proj/up_proj/down_proj/act_fn) and a one-layer
    audio tower, both full of ``Gemma4RMSNorm`` modules, plus the ``embed_vision``
    / ``embed_audio`` projectors: the module tree the real 31B presents."""
    config = Gemma4Config(
        text_config=_TINY_GEMMA4_TEXT_CONFIG_KWARGS,
        vision_config=_TINY_VISION_CONFIG_KWARGS,
        audio_config=dict(hidden_size=32, num_hidden_layers=1),
    )
    return _randomly_initialised(Gemma4ForConditionalGeneration, config)  # type: ignore[return-value]


def build_tiny_olmo3() -> Olmo3ForCausalLM:
    """A tiny OLMo 3 model on CPU: SiLU gated MLP, post-attention and
    post-feedforward ``Olmo3RMSNorm`` on the sublayer outputs, q/k norms in
    attention."""
    config = AutoConfig.for_model("olmo3", **_TINY_ATTENTION_CONFIG_KWARGS, hidden_act="silu")
    return _randomly_initialised(Olmo3ForCausalLM, config)  # type: ignore[return-value]
