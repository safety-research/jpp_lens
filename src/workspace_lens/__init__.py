"""Workspace lens: lenses as readouts of decoder-transformer residuals."""

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

import jlens
from jlens.hf import HFLensModel as HFLensModel
from jlens.protocol import LensModel

DEFAULT_DEVICE = "cuda" if t.cuda.is_available() else "cpu"


def get_hf_model(
    model_name: str,
    attn_implementation: str | None = None,
    device_map: str | None = None,
    experts_implementation: str | None = None,
    dequantize_fp8: bool = False,
    **from_pretrained_kwargs: object,
) -> LensModel:
    """Load ``model_name`` in bf16 on the GPU as a LensModel. ``attn_implementation``
    (e.g. "sdpa") pins the HF attention path. ``device_map`` (e.g. "auto") shards a
    model too large for one GPU across the visible devices instead of the single-GPU
    ``.cuda()``. DeepSeek-V4 needs ``attn_implementation="eager"`` (its attention
    sinks and 512-wide heads have no SDPA or flash path),
    ``experts_implementation="eager"`` (the per-expert loop instead of the grouped
    kernel) and ``dequantize_fp8=True``, which unpacks its native FP8/FP4 checkpoint
    to bf16 on load. Any further keyword argument goes to ``from_pretrained``
    unchanged."""
    if experts_implementation is not None:
        from_pretrained_kwargs["experts_implementation"] = experts_implementation
    if dequantize_fp8:
        from transformers import FineGrainedFP8Config

        # For a checkpoint that is already quantized, transformers keeps the checkpoint's own
        # quantization config (block size, activation scheme, scale format) and takes only the
        # loading attributes from this one: dequantize=True unpacks the FP8 blocks and the FP4
        # experts to bf16 on load.
        from_pretrained_kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=t.bfloat16,
        attn_implementation=attn_implementation,
        device_map=device_map,
        **from_pretrained_kwargs,
    )
    if device_map is None:
        hf_model = hf_model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = jlens.from_hf(hf_model, tokenizer)
    return model  # type: ignore[return-value]
