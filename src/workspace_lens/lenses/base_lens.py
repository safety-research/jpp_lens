# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
# Modified by Kola Ayonrinde, 2026.
"""Applying a fitted lens.

A lens holds per-layer transport parameters and reads a residual out as ``unembed(transport(h_layer))``.
:meth:`BaseLens.apply` runs a forward pass and reads out the requested layers;
:meth:`BaseLens.transport` is the bare transport for callers that already
have residuals.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import torch as t

from jlens.protocol import LensModel
from workspace_lens.config import LensConfig, LensTypes
from workspace_lens.residual_streams import (
    collapse_streams_like_the_model,
    residual_mean_over_streams,
)
from workspace_lens.utils import check_layers_fitted, record_block_outputs


@dataclass
class LensParameters:
    """Normalised per-layer lens parameters.

    Construction derives ``source_layers`` from the ``_jacobians_L_dict_FN`` keys
    when it isn't given, casts the tensors to fp32, and validates that every
    provided dict covers exactly ``source_layers`` — which also guarantees the
    dicts agree with each other on their layer keys.
    """

    _source_layers: Sequence[int] | None = None
    _jacobians_L_dict_FN: dict[int, t.Tensor] | None = None

    def __post_init__(self) -> None:
        if self._source_layers is not None:
            self._source_layers = sorted(self._source_layers)
        elif self._jacobians_L_dict_FN is not None:
            self._source_layers = sorted(self._jacobians_L_dict_FN.keys())
        else:
            raise ValueError("must provide source_layers or jacobians")

        if self._jacobians_L_dict_FN is not None:
            self._jacobians_L_dict_FN = {
                layer: jacobian_FN.float()
                for layer, jacobian_FN in self._jacobians_L_dict_FN.items()
            }

        provided_dicts = {
            "jacobians": self._jacobians_L_dict_FN,
        }
        for name, per_layer in provided_dicts.items():
            if per_layer is not None and sorted(per_layer) != self._source_layers:
                raise ValueError(
                    f"{name} covers layers {sorted(per_layer)} but "
                    f"source_layers is {self._source_layers}"
                )

        self.source_layers = self._source_layers
        self.jacobians_L_dict_FN = self._jacobians_L_dict_FN


class BaseLens(ABC):
    """A fitted lens: per-layer transport parameters and the readout method.

    Attributes:
        jacobians_L_dict_FN: ``{layer: Tensor[F, N]}`` (stored fp32), or
            ``None`` for lens types without them. ``N`` is ``d_model`` at the
            source layer and ``F`` ``d_model`` at the transport target layer
            (equal for one model; the letters say which basis a tensor is in).
        source_layers: Sorted list of fitted layer indices.
        num_prompts_trained_on: Number of prompts the lens was averaged over.
        d_model: Residual-stream width.
    """

    lens_type: ClassVar[LensTypes]
    _lens_classes: ClassVar[dict[str, type[BaseLens]]] = {}
    # Serialisation name (stable: used in save files and _from_parameters)
    # -> attribute name on the lens instance. Keep the keys as they are: they
    # are the on-disk "parameters" keys of every saved lens.
    _PARAMETER_ATTRIBUTES: ClassVar[dict[str, str]] = {
        "jacobians": "jacobians_L_dict_FN",
    }

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        # Register only classes that declare their own lens_type, so an
        # undeclared subclass can't hijack its parent's registry slot.
        if "lens_type" in cls.__dict__:
            BaseLens._lens_classes[cls.__dict__["lens_type"]] = cls

    def __init__(
        self,
        config: LensConfig,
        *,
        source_layers: Sequence[int] | None = None,
        jacobians: dict[int, t.Tensor] | None = None,
    ) -> None:
        declared_lens_type = getattr(type(self), "lens_type", None)
        if declared_lens_type is None:
            raise TypeError(f"{type(self).__name__} must declare a lens_type class attribute")
        if config.lens_type != declared_lens_type:
            raise ValueError(
                f"config.lens_type={config.lens_type!r} but "
                f"{type(self).__name__} is a {declared_lens_type!r} lens"
            )

        lens_parameters = LensParameters(
            _source_layers=source_layers,
            _jacobians_L_dict_FN=jacobians,
        )
        self.source_layers = lens_parameters.source_layers
        self.jacobians_L_dict_FN = lens_parameters.jacobians_L_dict_FN

        self.config = config
        self.num_prompts_trained_on = config.num_prompts_trained_on
        self.d_model = config.d_model

        self.relative_end_transport_layer = config.relative_end_transport_layer

    ### SHARED METHODS

    def _parameter_dicts(self) -> dict[str, dict[int, t.Tensor | float]]:
        """The per-layer parameter dicts this lens actually has, keyed by
        serialisation name — the unit that validation, :meth:`save` and
        :meth:`move_parameters_to_device` iterate over."""
        named = {
            name: getattr(self, attribute)
            for name, attribute in self._PARAMETER_ATTRIBUTES.items()
        }
        return {name: per_layer for name, per_layer in named.items() if per_layer is not None}

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(d_model={self.d_model}, "
            f"num_prompts_trained_on={self.num_prompts_trained_on}, "
            f"source_layers=[{self.source_layers[0]}..{self.source_layers[-1]}] "
            f"({len(self.source_layers)} layers))"
        )

    def move_parameters_to_device(self, device: t.device | str) -> None:
        """Move every per-layer parameter tensor to ``device``.

        A no-op for tensors already there; callers that hold residuals on the
        model's device can do this once instead of paying a host-to-device
        copy inside every :meth:`transport` call."""
        for name, per_layer in self._parameter_dicts().items():
            setattr(
                self,
                self._PARAMETER_ATTRIBUTES[name],
                {
                    layer: (
                        parameter.to(device) if isinstance(parameter, t.Tensor) else parameter
                    )
                    for layer, parameter in per_layer.items()
                },
            )

    def save(self, path: str, *, dtype: t.dtype = t.float16) -> None:
        """Save to ``path``. Floating parameters are stored as ``dtype``
        (default fp16: halves file size; entries are O(1) so the range is not
        a constraint and fp16's extra mantissa bits beat bf16 here); integer
        parameters keep their dtype."""
        t.save(
            {
                "parameters": {
                    name: {
                        layer: (
                            parameter.to(dtype)
                            if isinstance(parameter, t.Tensor) and parameter.is_floating_point()
                            else parameter
                        )
                        for layer, parameter in per_layer.items()
                    }
                    for name, per_layer in self._parameter_dicts().items()
                },
                "source_layers": self.source_layers,
                "config": self.config.to_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> BaseLens:
        """Load a lens previously written by :meth:`save`.

        The concrete class is picked from the saved ``config.lens_type``, so
        ``BaseLens.load`` works for any lens file this library writes. Files in the
        format of the released J-Lens and R-Lens files, which embed no config, load
        with :func:`workspace_lens.utils.load_lens_file`.
        """
        checkpoint = t.load(path, map_location="cpu", weights_only=True)
        if "config" not in checkpoint:
            raise ValueError(
                f"{path} embeds no config; load released-format lens files with "
                "workspace_lens.utils.load_lens_file"
            )
        if "J" in checkpoint:  # a top-level "J" key with a config: always a J-lens
            parameters = {"jacobians": checkpoint["J"]}
            source_layers = sorted(checkpoint["J"])
        elif "parameters" in checkpoint:
            parameters = checkpoint["parameters"]
            source_layers = checkpoint["source_layers"]
        else:
            raise ValueError(
                f"{path} is not a BaseLens file "
                f"(found keys {sorted(checkpoint)!r}; a fit() checkpoint?)"
            )

        config = LensConfig.from_dict(checkpoint["config"])

        declared_lens_type = getattr(cls, "lens_type", None)
        if declared_lens_type is not None and declared_lens_type != config.lens_type:
            raise ValueError(
                f"{path} holds a {config.lens_type!r} lens; load it with "
                f"BaseLens.load or the matching class, not {cls.__name__}"
            )
        lens_class = cls._lens_classes.get(config.lens_type)
        if lens_class is None:
            raise ValueError(
                f"no lens class registered for lens_type={config.lens_type!r} "
                f"(known: {sorted(cls._lens_classes)})"
            )

        return lens_class._from_parameters(source_layers, parameters, config)

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str,
        *,
        filename: str = "lens.pt",
        revision: str | None = None,
    ) -> BaseLens:
        """Load a lens from a local file, a local directory, or a HuggingFace
        Hub ``repo_id``. ``filename`` is the path inside the directory or repo
        (so one Hub repo can host lenses for many models); ignored when
        ``name_or_path`` is itself a file. ``revision`` selects a Hub branch,
        tag, or commit. Deserialisation goes through :meth:`load`
        (``weights_only=True``)."""
        if os.path.isfile(name_or_path):
            return cls.load(name_or_path)
        if not os.path.isdir(name_or_path):
            from huggingface_hub import snapshot_download

            name_or_path = snapshot_download(
                name_or_path, allow_patterns=[filename], revision=revision
            )
        return cls.load(os.path.join(name_or_path, filename))

    def _check_layers(self, model: LensModel, layers: Sequence[int]) -> None:
        out_of_range = sorted(
            layer for layer in set(layers) if not 0 <= layer < model.n_layers
        )
        if out_of_range:
            raise ValueError(
                f"layers {out_of_range} out of range for a {model.n_layers}-layer model"
            )

        check_layers_fitted(self, layers)

    @t.no_grad()
    def apply(
        self,
        model: LensModel,
        prompt: str,
        *,
        layers: Sequence[int] | None = None,
        token_positions_for_residuals: Sequence[int] | None = None,
        max_seq_len: int = 512,
    ) -> tuple[dict[int, t.Tensor], t.Tensor, t.Tensor]:
        """Run ``model`` on ``prompt`` and read out the requested layers.

        One forward pass records the residual stream; each requested layer is
        then read out as ``unembed(transport(h_layer))``. Subclasses supply
        :meth:`transport`.

        Args:
            model: The model to read out from.
            prompt: Input text.
            layers: Layers to read out at. Defaults to all of
                :attr:`source_layers`; must be a subset of it.
            token_positions_for_residuals: Token positions to read out (Python
                indexing into the sequence; negative indices count from the
                end). ``None`` reads out every position.
            max_seq_len: Truncate the prompt to this many tokens.

        Returns:
            A triple ``(lens_logits, model_logits, input_ids)``. ``lens_logits``
            maps each requested layer to a ``[n_positions, vocab_size]`` fp32
            CPU tensor; ``model_logits`` is the model's actual final-layer
            logits at the same positions (same shape); ``input_ids`` is the
            tokenized prompt, ``[1, seq_len]``. ``n_positions`` is
            ``len(token_positions_for_residuals)``, or the full sequence
            length when it is ``None``.

        Raises:
            ValueError: If any requested layer is out of range for the model
                or not in :attr:`source_layers`.
        """
        if layers is None:
            layers = self.source_layers  # default to all source layers
        self._check_layers(model, layers)

        final_layer_index = model.n_layers - 1
        layers_to_record = sorted(set(layers) | {final_layer_index})

        # 1. Run the model and collect the block outputs. The lens reads the stream mean at every layer
        # (the lens's residual); the model's own logits below come from the model's head, which on a
        # multi-stream model collapses the final block's streams itself (residual_streams.py).
        input_ids_Int_1S, block_outputs_L_dict_1SrN = record_block_outputs(
            model, prompt, max_seq_len, layers_to_record
        )
        activations_L_dict_SN = {
            layer: residual_mean_over_streams(block_output_1SrN)[0]
            for layer, block_output_1SrN in block_outputs_L_dict_1SrN.items()
        }
        model_final_residual_SF = collapse_streams_like_the_model(
            model, block_outputs_L_dict_1SrN[final_layer_index]
        )[0]

        def _residuals_from_positions_at_layer(layer: int) -> t.Tensor:
            """Residuals at the requested positions: ``[n_positions, d_model]``."""
            layer_activations_SN = activations_L_dict_SN[layer]

            if token_positions_for_residuals is not None:
                layer_activations_SN = layer_activations_SN[
                    list(token_positions_for_residuals)
                ]
            return layer_activations_SN.float()

        # 2. Compute lens logits at each requested layer
        lens_logits_L_dict_SV: dict[int, t.Tensor] = {}
        for layer in layers:
            layer_activations_SN = _residuals_from_positions_at_layer(layer)

            transported_layer_activations_SF = self.transport(layer_activations_SN, layer)

            layer_lens_logits_SV = model.unembed(transported_layer_activations_SF)
            lens_logits_L_dict_SV[layer] = layer_lens_logits_SV.float().cpu()

        # 3. Compute model logits at the final layer by applying the unembedding matrix as normal
        if token_positions_for_residuals is not None:
            model_final_residual_SF = model_final_residual_SF[list(token_positions_for_residuals)]
        model_logits_SV = model.unembed(model_final_residual_SF.float())
        model_logits_SV = model_logits_SV.float().cpu()

        return lens_logits_L_dict_SV, model_logits_SV, input_ids_Int_1S

    ### ABSTRACT METHODS

    @abstractmethod
    def transport(self, residual_BsN: t.Tensor, layer: int) -> t.Tensor:
        """Map a residual at ``layer`` into the final-layer basis.

        Args:
            residual_BsN: Tensor of shape ``[..., d_model]``.
            layer: Source layer index (must be in :attr:`source_layers`).
        """
        ...

    @classmethod
    @abstractmethod
    def _from_parameters(
        cls,
        source_layers: Sequence[int],
        parameters: dict[str, dict[int, t.Tensor | float]],
        config: LensConfig,
    ) -> BaseLens:
        """Construct this lens from named per-layer parameter dicts (used by
        :meth:`load`)."""
        ...
