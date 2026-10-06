"""A tiny CPU-only DeepSeek-V4 model for the residual-stream and LRP tests.

Builds a real ``DeepseekV4ForCausalLM`` from a shrunk config, so the exact module
classes the real model instantiates exercise our code: the ``[batch, seq, hc_mult,
d_model]`` residual carried between blocks by ``DeepseekV4HyperConnection`` and
collapsed by ``DeepseekV4HyperHead``, the clamped ``DeepseekV4MLP`` shared expert,
``DeepseekV4Experts`` with a hash router (the first block) and a top-k router (the
rest), ``DeepseekV4RMSNorm`` and the unweighted coefficient norm. Uses eager attention and
the eager expert loop.

``num_experts_per_tok`` equals ``n_routed_experts`` by default so every expert is
active at every token: the forward is then smooth in the residual and finite
differences are well posed (a top-k routing flip would break them).
"""

from __future__ import annotations

import torch as t
from transformers import DeepseekV4Config, DeepseekV4ForCausalLM

import jlens
from jlens.hf import HFLensModel
from jlens.hooks import ActivationRecorder
from workspace_lens.tests.tiny_gemma_olmo import _randomly_initialised
from workspace_lens.tests.tiny_qwen import TINY_QWEN_VOCAB_SIZE, ByteVocabTokenizer

TINY_DEEPSEEK_VOCAB_SIZE = TINY_QWEN_VOCAB_SIZE
TINY_DEEPSEEK_NUM_LAYERS = 3
TINY_DEEPSEEK_HIDDEN_SIZE = 32
TINY_DEEPSEEK_HC_MULT = 4


def tiny_deepseek_config(*, num_routed_experts: int = 4, num_experts_per_tok: int | None = None) -> DeepseekV4Config:
    return DeepseekV4Config(
        vocab_size=TINY_DEEPSEEK_VOCAB_SIZE,
        hidden_size=TINY_DEEPSEEK_HIDDEN_SIZE,
        moe_intermediate_size=48,
        num_hidden_layers=TINY_DEEPSEEK_NUM_LAYERS,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        q_lora_rank=16,
        o_lora_rank=16,
        o_groups=2,
        n_routed_experts=num_routed_experts,
        num_experts_per_tok=num_routed_experts if num_experts_per_tok is None else num_experts_per_tok,
        n_shared_experts=1,
        mlp_layer_types=["hash_moe", "moe", "moe"],  # one hash-routed block, then top-k blocks
        hc_mult=TINY_DEEPSEEK_HC_MULT,
        hc_sinkhorn_iters=3,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=8,
        max_position_embeddings=128,
        sliding_window=16,
        num_nextn_predict_layers=0,
        attn_implementation="eager",
        experts_implementation="eager",
        use_cache=False,
        tie_word_embeddings=False,
    )


def build_tiny_deepseek(seed: int = 0, **config_kwargs) -> DeepseekV4ForCausalLM:
    """A randomly initialised tiny DeepSeek-V4 in fp32, eval mode, no grad on the parameters."""
    return _randomly_initialised(DeepseekV4ForCausalLM, tiny_deepseek_config(**config_kwargs), seed)  # type: ignore[return-value]


def capture_block_output(block: t.nn.Module, run_forward) -> t.Tensor:
    """The raw output of ``block`` (all streams, ``[B, S, R, N]``) during ``run_forward()``."""
    with t.no_grad(), ActivationRecorder([block], at=[0]) as recorder:
        run_forward()
    return recorder.activations[0]


def tiny_deepseek_lens_model(seed: int = 0, **config_kwargs) -> HFLensModel:
    """The tiny model behind the :class:`jlens.hf.HFLensModel` surface our fitting and
    evals use (``layers``, ``forward``, ``unembed`` through the final norm and head)."""
    lens_model = jlens.from_hf(build_tiny_deepseek(seed, **config_kwargs), ByteVocabTokenizer())
    return lens_model  # type: ignore[return-value]


def random_tiny_deepseek_input_ids(seed: int = 1, seq_len: int = 12) -> t.Tensor:
    generator = t.Generator().manual_seed(seed)
    return t.randint(0, TINY_DEEPSEEK_VOCAB_SIZE, (1, seq_len), generator=generator)
