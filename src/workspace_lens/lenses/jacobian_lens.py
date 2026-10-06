from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import torch as t

from workspace_lens.config import LensConfig, LensTypes
from workspace_lens.lenses.base_lens import BaseLens


class JacobianLens(BaseLens):
    """Linear transport by the corpus-averaged Jacobian: ``J_l @ h``."""

    lens_type: ClassVar[LensTypes] = "jacobian"

    def __init__(self, jacobians: dict[int, t.Tensor], *, config: LensConfig) -> None:
        super().__init__(config, jacobians=jacobians)

    @classmethod
    def _from_parameters(
        cls,
        source_layers: Sequence[int],
        parameters: dict[str, dict[int, t.Tensor | float]],
        config: LensConfig,
    ) -> JacobianLens:
        if "jacobians" not in parameters:
            raise ValueError("Missing 'jacobians' in parameters.")
        return cls(jacobians=parameters["jacobians"], config=config)  # type: ignore

    def transport(self, residual_BsN: t.Tensor, layer: int) -> t.Tensor:
        """Map a residual at ``layer`` into the final-layer basis: ``J_l @ h``.

        Args:
            residual_BsN: Tensor of shape ``[..., d_model]``.
            layer: Source layer index (must be in :attr:`source_layers`).
        """
        assert self.jacobians_L_dict_FN is not None, "Jacobian dictionary is not initialized."
        jacobian_FN = self.jacobians_L_dict_FN[layer].to(residual_BsN.device)
        transported_residual_BsF = residual_BsN @ jacobian_FN.T
        return transported_residual_BsF
