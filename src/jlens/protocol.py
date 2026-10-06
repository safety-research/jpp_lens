# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
# Modified by Kola Ayonrinde, 2026.
"""The model interface the lens is typed against.

Any model can be plugged in by implementing these members.
:func:`jlens.hf.from_hf` is the HuggingFace adapter; ``tests/tiny.py`` is a
minimal from-scratch example.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import torch
from torch import nn


class LensModel(Protocol):
    """What the lens needs from a model.

    Attributes:
        n_layers: Number of residual blocks.
        d_model: Residual-stream width.
        layers: The residual blocks, indexable by integer; what
            :class:`~jlens.hooks.ActivationRecorder` hooks.
        tokenizer: Tokenizer; must provide ``decode(token_ids) -> str``.
            Fitting and :meth:`~workspace_lens.lenses.base_lens.BaseLens.apply`
            never touch it.
    """

    n_layers: int
    d_model: int
    layers: Sequence[nn.Module]
    tokenizer: Any

    def encode(self, text: str, *, max_length: int = ...) -> torch.Tensor:
        """Tokenize ``text`` to ``input_ids`` of shape ``[1, seq_len]`` on the
        model's input device."""
        ...

    def forward(self, input_ids: torch.Tensor) -> Any:
        """Run the residual stack on ``input_ids`` (no LM head). Must build an
        autograd graph through :attr:`layers` when grad is enabled, and must be
        deterministic across batch elements (eval mode, dropout off) — the
        fitting estimator replicates the prompt along the batch axis."""
        ...

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        """Map a residual-stream tensor ``[..., d_model]`` to logits
        ``[..., vocab_size]`` (final norm + LM head)."""
        ...
