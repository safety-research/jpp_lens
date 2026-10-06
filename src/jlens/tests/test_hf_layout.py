# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Layout auto-detection over mock HF-shaped modules: no network, no transformers."""

import pytest
import torch
from torch import nn

from jlens.hf import Layout, _find_layout


class _MockConfig:
    def __init__(self, n_layers: int, d_model: int) -> None:
        self.num_hidden_layers = n_layers
        self.hidden_size = d_model

    def get_text_config(self):
        return self


def _block(d):
    return nn.Linear(d, d)


def _make_hf_mock(layout: Layout, *, n_layers=2, d_model=8, vocab=100) -> nn.Module:
    """Build the smallest nn.Module tree that matches ``layout``."""
    text = nn.Module()
    setattr(
        text, layout.layers, nn.ModuleList(_block(d_model) for _ in range(n_layers))
    )
    setattr(text, layout.norm, nn.LayerNorm(d_model))
    setattr(text, layout.embed, nn.Embedding(vocab, d_model))

    root = nn.Module()
    parent = root
    *parts, leaf = layout.path.split(".")
    for part in parts:
        child = nn.Module()
        setattr(parent, part, child)
        parent = child
    setattr(parent, leaf, text)
    setattr(root, layout.lm_head, nn.Linear(d_model, vocab))
    root.config = _MockConfig(n_layers, d_model)
    return root


@pytest.mark.parametrize(
    "layout",
    [
        Layout("model"),
        Layout("model.language_model"),
        Layout("model", norm="final_layernorm"),  # Phi
        Layout("transformer", layers="h", norm="ln_f", embed="wte"),  # GPT-2
        Layout(
            "gpt_neox", norm="final_layer_norm", embed="embed_in", lm_head="embed_out"
        ),
    ],
    ids=["llama", "multimodal", "phi", "gpt2", "gptneox"],
)
def test_find_layout_roundtrip(layout):
    mock = _make_hf_mock(layout)
    found = _find_layout(mock)
    assert found == layout


def test_find_layout_unknown_raises():
    bad = nn.Module()
    bad.something = nn.Module()
    with pytest.raises(ValueError, match="could not locate"):
        _find_layout(bad)


def test_from_hf_with_mock_llama():
    """End-to-end through HFLensModel on a Llama-shaped mock."""
    from jlens import from_hf

    layout = Layout("model")
    mock = _make_hf_mock(layout, n_layers=3, d_model=8)

    class Tok:
        bos_token_id = None

    lm = from_hf(mock, Tok())
    assert lm.n_layers == 3
    assert lm.d_model == 8
    assert lm.layout == layout
    logits = lm.unembed(torch.randn(1, 5, 8))
    assert logits.shape == (1, 5, 100)
