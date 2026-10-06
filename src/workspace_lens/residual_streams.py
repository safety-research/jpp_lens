"""Residual streams: models whose block output is several parallel residual
streams instead of one vector.

DeepSeek-V4's manifold hyper-connections carry ``[batch, seq, hc_mult, d_model]``
between blocks (four streams, re-mixed by learned coefficients in every block and
collapsed by a learned head only before the final norm). The lens's residual at a
layer is the **mean over the streams**, at every source layer and at the transport
target (the convention of the published DeepSeek-V4 lenses). A 3-D block output
passes through every function here unchanged, so single-stream models are unaffected.

Shape letter: ``R`` is the stream axis (``hc_mult`` parallel residual streams),
lowercase ``r`` where it may be absent: ``block_output_BSrN`` is a recorded block
output of either kind.

Reverse-mode semantics of the mean. The Jacobian the lens estimates is that of the
target mean ``m_T = (1/R) sum_r T_r`` with respect to a *uniform* perturbation of the
source streams (every stream moved by the same vector ``v``, so the source mean moves
by ``v``; the mean has no inverse, so this lift is a convention). On the target side
autograd does the work: the estimators differentiate ``m_T`` itself (an attached
``.mean`` over the streams), so the cotangent ``c`` on ``m_T`` reaches every target
stream as ``c/R``. On the source side the returned per-stream gradients add up,
``d m_T / d v = sum_r (d m_T / d S_r)`` because ``d S_r / d v = I`` for each stream:
:func:`sum_gradient_over_streams`.
"""

from __future__ import annotations

import torch as t

from jlens.protocol import LensModel

STREAM_DIM = 2  # [batch, seq, streams, d_model]


def has_residual_streams(block_output_BSrN: t.Tensor) -> bool:
    """True for a multi-stream block output ``[B, S, R, N]``; False for ``[B, S, N]``."""
    return block_output_BSrN.dim() == 4


def residual_mean_over_streams(block_output_BSrN: t.Tensor) -> t.Tensor:
    """The lens's residual ``[B, S, N]``: the mean over the streams of a multi-stream
    block output, or the block output itself when there is a single stream."""
    if has_residual_streams(block_output_BSrN):
        return block_output_BSrN.mean(dim=STREAM_DIM)
    return block_output_BSrN


def sum_gradient_over_streams(gradient_BSrN: t.Tensor) -> t.Tensor:
    """The gradient with respect to the source mean under a uniform perturbation of the
    streams: the per-stream gradients summed over the stream axis. The sum is kept in fp32
    (torch already accumulates a bf16 reduction in fp32 but would round the result back to
    bf16; every consumer casts to fp32 anyway, so the promotion only skips that rounding).
    A single-stream gradient is returned unchanged."""
    if has_residual_streams(gradient_BSrN):
        return gradient_BSrN.sum(dim=STREAM_DIM, dtype=t.promote_types(gradient_BSrN.dtype, t.float32))
    return gradient_BSrN


def collapse_streams_like_the_model(lens_model: LensModel, block_output_BSrN: t.Tensor) -> t.Tensor:
    """The residual the MODEL itself reads out, ``[B, S, N]``: a multi-stream block output
    collapsed by the model's own learned head (DeepSeek-V4's ``hc_head``, the module the
    text decoder applies to the last block's streams before its final norm), or the block
    output itself for a single-stream model.

    This is not the lens's residual (that is the stream mean) but the model's: use it where
    "the model's own logits" are meant, such as the correctness filter. The head is reached
    through the adapter's text module (``jlens.hf.HFLensModel._text_module``)."""
    if not has_residual_streams(block_output_BSrN):
        return block_output_BSrN
    text_module = getattr(lens_model, "_text_module", None)
    head = getattr(text_module, "hc_head", None)
    if head is None:
        raise ValueError(
            "a multi-stream block output needs the model's stream-collapse head, which the "
            "adapter's text module does not expose as `hc_head`"
        )
    collapsed_BSN: t.Tensor = head(block_output_BSrN)
    return collapsed_BSN
