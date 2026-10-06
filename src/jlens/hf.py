# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
# Modified by Kola Ayonrinde, 2026.
"""HuggingFace adapter.

Wraps an already-loaded HF model as a :class:`~jlens.protocol.LensModel` so
the rest of the package stays model-library-agnostic. Model loading
(``from_pretrained``, device placement, dtype) stays the caller's job;
:func:`from_hf` only locates the residual stack inside whatever it's handed.

Any model library can be plugged in the same way: implement the
:class:`~jlens.protocol.LensModel` members directly (``tests/tiny.py`` is a
minimal example) and the rest of the package works unchanged.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


def _resolve_attr_path(obj: Any, dotted_path: str) -> Any:
    return functools.reduce(getattr, dotted_path.split("."), obj)


@dataclass(frozen=True)
class Layout:
    """Where the lens-relevant submodules live inside a HuggingFace model.

    Attributes:
        path: Dotted attribute path from the ``*ForCausalLM`` to the bare text
            decoder (the module to call for a hooks-visible forward pass).
        layers: Attribute name on the text decoder for the residual blocks.
        norm: Attribute name for the final pre-unembed norm.
        embed: Attribute name for the input token embedding.
        lm_head: Attribute name on the ``*ForCausalLM`` for the unembedding.
    """

    path: str
    layers: str = "layers"
    norm: str = "norm"
    embed: str = "embed_tokens"
    lm_head: str = "lm_head"


#: Known layouts, tried in order. The first whose ``path`` resolves and whose
#: text decoder has all three of ``layers``/``norm``/``embed`` wins. Covers
#: Llama / Qwen / Mistral / Gemma / OLMo / StableLM (the modern HF default),
#: their multimodal-wrapper variants, plus Phi, GPT-2, and GPT-NeoX.
_LAYOUTS: tuple[Layout, ...] = (
    Layout("model"),
    Layout("model.language_model"),
    Layout("language_model"),
    Layout("model", norm="final_layernorm"),  # Phi
    Layout("transformer", layers="h", norm="ln_f", embed="wte"),  # GPT-2
    Layout(
        "gpt_neox", norm="final_layer_norm", embed="embed_in", lm_head="embed_out"
    ),  # Pythia
)


def _find_layout(hf_model: nn.Module) -> Layout:
    """Locate the text decoder inside an HF ``*ForCausalLM`` /
    ``*ForConditionalGeneration`` by trying :data:`_LAYOUTS` in order."""
    for layout in _LAYOUTS:
        try:
            candidate = _resolve_attr_path(hf_model, layout.path)
        except AttributeError:
            continue
        if all(
            hasattr(candidate, a) for a in (layout.layers, layout.norm, layout.embed)
        ) and hasattr(hf_model, layout.lm_head):
            return layout
    raise ValueError(
        f"could not locate the text decoder inside {type(hf_model).__name__} "
        f"(tried {len(_LAYOUTS)} known layouts); pass layout= explicitly"
    )


class HFLensModel:
    """:class:`~jlens.protocol.LensModel` over a loaded HuggingFace model.

    Holds references into the caller's model; nothing is copied. The
    constructor mutates that model in place: every parameter gets
    ``requires_grad_(False)`` (the Jacobian fit needs grads only with respect
    to activations), ``compile=True`` replaces each block with a
    :func:`torch.compile` wrapper, and ``force_bos`` may set
    ``tokenizer.add_bos_token``. Pass a model you don't otherwise need.
    """

    def __init__(
        self,
        hf_model: nn.Module,
        tokenizer: Any,
        *,
        layout: Layout | None = None,
        compile: bool = False,
        force_bos: bool = True,
    ) -> None:
        self._hf_model = hf_model
        self.tokenizer = tokenizer
        if (
            force_bos
            and getattr(tokenizer, "bos_token_id", None) is not None
            and hasattr(tokenizer, "add_bos_token")
        ):
            tokenizer.add_bos_token = True

        hf_model.eval()
        for param in hf_model.parameters():
            param.requires_grad_(False)

        if layout is None:
            layout = _find_layout(hf_model)
        self.layout = layout
        self._text_module = _resolve_attr_path(hf_model, layout.path)
        self.layers: nn.ModuleList = getattr(self._text_module, layout.layers)
        self._final_norm: nn.Module = getattr(self._text_module, layout.norm)
        self._embed_tokens: nn.Module = getattr(self._text_module, layout.embed)
        self._lm_head: nn.Module = getattr(hf_model, layout.lm_head)

        text_config = hf_model.config.get_text_config()
        self.n_layers: int = text_config.num_hidden_layers
        self.d_model: int = text_config.hidden_size
        self._logit_softcap: float | None = getattr(
            text_config, "final_logit_softcapping", None
        )
        if len(self.layers) != self.n_layers:
            raise ValueError(
                f"config.num_hidden_layers={self.n_layers} but found "
                f"{len(self.layers)} blocks at {layout.path}.{layout.layers}"
            )

        # Per-layer compile: each block stays a hook boundary, so
        # ActivationRecorder still fires and the retained graph is bounded per
        # block. Whole-module compile would inline the blocks and bypass the
        # hooks.
        if compile:
            for i in range(len(self.layers)):
                self.layers[i] = torch.compile(
                    self.layers[i], mode="default", dynamic=False
                )

    def __repr__(self) -> str:
        return (
            f"HFLensModel({type(self._hf_model).__name__}, "
            f"n_layers={self.n_layers}, d_model={self.d_model})"
        )

    @property
    def input_device(self) -> torch.device:
        return self._embed_tokens.weight.device

    def encode(self, text: str, *, max_length: int = 512) -> torch.Tensor:
        encoded = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        )
        return encoded.input_ids.to(self.input_device)

    def forward(self, input_ids: torch.Tensor) -> Any:
        return self._text_module(input_ids=input_ids, use_cache=False)

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        target_device = self._lm_head.weight.device
        target_dtype = self._lm_head.weight.dtype
        logits = self._lm_head(
            self._final_norm(residual.to(target_dtype).to(target_device))
        )
        if self._logit_softcap is not None:
            logits = self._logit_softcap * torch.tanh(logits / self._logit_softcap)
        return logits


def from_hf(
    hf_model: nn.Module,
    tokenizer: Any,
    *,
    layout: Layout | None = None,
    text_module: str | None = None,
    compile: bool = False,
    force_bos: bool = True,
) -> HFLensModel:
    """Wrap a loaded HuggingFace model as a :class:`~jlens.protocol.LensModel`.

    Args:
        hf_model: A loaded ``*ForCausalLM`` (or ``*ForConditionalGeneration``),
            already on the target device and dtype.
        tokenizer: The matching HF tokenizer.
        layout: Where the residual blocks / final norm / embedding / LM head
            live inside ``hf_model``. Auto-detected for the common HF families;
            pass explicitly only for unusual layouts.
        text_module: Deprecated alias for ``layout=Layout(path=text_module)``.
        compile: Wrap each residual block in :func:`torch.compile`. Faster
            backward during lens fitting after a one-time compilation cost.
            Do not combine with ``device_map="auto"``.
        force_bos: Some instruction-tuned checkpoints ship with
            ``add_bos_token=False``; raw-text prompts are degraded without an
            attention-sink BOS, so this sets it ``True`` by default. The
            attribute may have no effect for some fast-tokenizer
            configurations.
    """
    if text_module is not None:
        if layout is not None:
            raise TypeError("pass at most one of layout= / text_module=")
        layout = Layout(path=text_module)
    return HFLensModel(
        hf_model, tokenizer, layout=layout, compile=compile, force_bos=force_bos
    )
