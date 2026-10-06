# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
# Modified by Kola Ayonrinde, 2026.
"""A tiny CPU-only decoder for end-to-end tests.

Implements :class:`jlens.protocol.LensModel` (``n_layers``, ``d_model``,
``layers``, ``tokenizer``, ``encode``, ``forward``, ``unembed``) so
the lens fitting and :class:`jlens.hooks.ActivationRecorder` exercise their real
code paths against it. Residual blocks are ``h + 0.1 * linear(h)``: the small
gain keeps the Jacobian well-conditioned so the late-layer ``diag(J) ~= 1``
property holds.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn


class _ResidualBlock(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            self.linear.weight.mul_(0.1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.linear(hidden)


class _ByteTokenizer:
    """Toy tokenizer: just enough surface for ``encode`` and ``decode``."""

    bos_token_id = 0

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str = "pt",
        truncation: bool = True,
        max_length: int = 128,
    ):
        ids = [self.bos_token_id] + [1 + (b % 30) for b in text.encode()][
            : max_length - 1
        ]
        return SimpleNamespace(input_ids=torch.tensor([ids]))

    def decode(self, ids, **_kw) -> str:
        return "".join(chr(96 + int(i)) for i in ids)


class TinyDecoder(nn.Module):
    """``n_layers``-layer residual stack on CPU. ``vocab_size=32``, ``d_model=8``
    by default."""

    def __init__(
        self, n_layers: int = 4, d_model: int = 8, vocab_size: int = 32, seed: int = 0
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.n_layers = n_layers
        self.d_model = d_model
        self.tokenizer = _ByteTokenizer()
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([_ResidualBlock(d_model) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    @property
    def input_device(self) -> torch.device:
        return self.embed_tokens.weight.device

    def encode(self, text: str, *, max_length: int = 128) -> torch.Tensor:
        return self.tokenizer(text, max_length=max_length).input_ids.to(
            self.input_device
        )

    def forward(self, input_ids: torch.Tensor):
        hidden = self.embed_tokens(input_ids)
        for block in self.layers:
            hidden = block(hidden)
        return SimpleNamespace(last_hidden_state=hidden)

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(residual.float()))
