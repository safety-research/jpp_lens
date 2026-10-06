"""A tiny CPU-only Qwen3.5-family model for LRP-surgery tests.

Unlike :class:`jlens.tests.tiny.TinyDecoder` (plain linear residual blocks),
this builds a real ``Qwen3_5ForCausalLM`` from a shrunk text-only config, so
the exact module classes our target models instantiate — ``Qwen3_5RMSNorm``,
``Qwen3_5MLP``, ``Qwen3_5Attention``, ``Qwen3_5GatedDeltaNet``,
``Qwen3_5RMSNormGated`` — exercise the per-class LRP patchers for real. The
hybrid layer pattern (two GatedDeltaNet linear-attention layers per softmax
-attention layer, ``layer_types`` below) mirrors Qwen3.6-27B's
``full_attention_interval=4`` structure at test scale.

transformers falls back to its pure-torch GatedDeltaNet path on machines
without the fla/causal-conv1d kernels, which is differentiable on CPU —
exactly what these tests need.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch as t
from transformers import AutoConfig
from transformers.models.qwen3_5 import Qwen3_5ForCausalLM

import jlens
from jlens.hf import HFLensModel

TINY_QWEN_VOCAB_SIZE = 128


class ByteVocabTokenizer:
    """Toy tokenizer with the surface :class:`jlens.hf.HFLensModel` needs."""

    bos_token_id = None

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str = "pt",
        truncation: bool = True,
        max_length: int = 128,
    ) -> SimpleNamespace:
        input_ids = [1 + (byte % (TINY_QWEN_VOCAB_SIZE - 1)) for byte in text.encode()]
        return SimpleNamespace(input_ids=t.tensor([input_ids[:max_length]]))

    def decode(self, input_ids, **_kwargs) -> str:
        return "".join(chr(97 + int(i) % 26) for i in input_ids)


def random_tiny_qwen_input_ids(seed: int = 1, seq_len: int = 16) -> t.Tensor:
    """``[1, seq_len]`` random ids over the tiny Qwen vocab, from a local generator."""
    generator = t.Generator().manual_seed(seed)
    return t.randint(0, TINY_QWEN_VOCAB_SIZE, (1, seq_len), generator=generator)


def last_hidden_state_BSN(model, input_ids_Int_1S: t.Tensor) -> t.Tensor:
    """The final residual stream of a tiny Qwen model, no grad, no KV cache."""
    with t.no_grad():
        return model.model(input_ids=input_ids_Int_1S, use_cache=False).last_hidden_state


def _tiny_hybrid_config_kwargs(num_layers: int) -> dict:
    """The shared tiny hybrid-Qwen geometry (d_model 32, 2 linear-attention
    layers per full-attention layer) for both the dense and MoE fixtures."""
    if num_layers % 3 != 0:
        raise ValueError("num_layers must be a multiple of the 3-layer type pattern")
    return dict(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        vocab_size=TINY_QWEN_VOCAB_SIZE,
        max_position_embeddings=256,
        layer_types=["linear_attention", "linear_attention", "full_attention"]
        * (num_layers // 3),
    )


def build_tiny_qwen3_5(seed: int = 0, num_layers: int = 6) -> Qwen3_5ForCausalLM:
    """A randomly initialised 6-layer hybrid Qwen3.5 text model on CPU
    (d_model 32, ~72k parameters), eval mode, parameter grads off."""
    config = AutoConfig.for_model(
        "qwen3_5_text", **_tiny_hybrid_config_kwargs(num_layers)
    )
    t.manual_seed(seed)
    model = Qwen3_5ForCausalLM(config).eval()
    # A stable HF name so trainer resume paths can verify checkpoint/model
    # agreement (a from-config model otherwise reports name_or_path="").
    model.config.name_or_path = "tiny-qwen3_5"
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def wrap_tiny_qwen3_5(seed: int = 0, num_layers: int = 6) -> HFLensModel:
    """:func:`build_tiny_qwen3_5` wrapped as a :class:`jlens` LensModel."""
    return jlens.from_hf(build_tiny_qwen3_5(seed, num_layers), ByteVocabTokenizer())


def build_tiny_qwen3_5_moe(seed: int = 0, num_layers: int = 3):
    """A randomly initialised tiny hybrid Qwen3.5-MoE text model on CPU:
    8 routed experts top-2 with a gated shared expert per layer, so the
    r+moe patchers (fused-experts loop, shared-gate detach, backward scale)
    exercise the exact ``Qwen3_5Moe*`` classes of the Qwen3.6-35B-A3B model."""
    from transformers.models.qwen3_5_moe import Qwen3_5MoeForCausalLM

    config = AutoConfig.for_model(
        "qwen3_5_moe_text",
        **_tiny_hybrid_config_kwargs(num_layers),
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        decoder_sparse_step=1,
    )
    t.manual_seed(seed)
    model = Qwen3_5MoeForCausalLM(config).eval()
    model.config.name_or_path = "tiny-qwen3_5-moe"
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
