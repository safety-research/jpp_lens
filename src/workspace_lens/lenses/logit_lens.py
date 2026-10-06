"""The logit lens: identity transport, no fitted parameters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import torch as t

from workspace_lens.config import LensConfig, LensTypes
from workspace_lens.lenses.base_lens import BaseLens


class LogitLens(BaseLens):
    """Reads residuals out through the unembedding directly: ``unembed(h)``.

    The transport is the identity, so the lens has no per-layer parameters —
    just the ``source_layers`` it may be read out at.
    """

    lens_type: ClassVar[LensTypes] = "logit"

    def __init__(self, source_layers: Sequence[int], *, config: LensConfig) -> None:
        super().__init__(config, source_layers=source_layers)

    @classmethod
    def _from_parameters(
        cls,
        source_layers: Sequence[int],
        parameters: dict[str, dict[int, t.Tensor | float]],
        config: LensConfig,
    ) -> LogitLens:
        return cls(source_layers, config=config)

    def transport(self, residual_BsN: t.Tensor, layer: int) -> t.Tensor:
        """Identity: the logit lens decodes the residual where it stands."""
        return residual_BsN
