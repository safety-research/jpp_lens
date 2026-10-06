from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch as t


class Intervention(ABC):
    """One residual-stream edit built from per-layer J-lens vectors.

    Subclasses implement :meth:`_edit_selected` (fp32 in, fp32 out, same
    shape) and must NOT override :meth:`edit`, which is the template method:
    it validates the layer and d_model, applies the optional token-position
    mask, computes in fp32 on the residual's device and casts back to the
    residual's original dtype.

    Args:
        layers: Layers this intervention has vectors for.
        d_model: Residual width the vectors were built for.
        token_positions: Positions along the sequence dim to edit (Python
            indexing; negative indices count from the end), applied to every
            batch row. ``None`` edits all positions. Duplicates are rejected
            because duplicate fancy-index write-back is ill-defined.
    """

    def __init__(
        self,
        *,
        layers: Sequence[int],
        d_model: int,
        token_positions: Sequence[int] | None = None,  # if None edit all positions
    ) -> None:
        self.layers: list[int] = sorted(layers)
        self.d_model = d_model

        if token_positions is not None:
            token_positions = list(token_positions)
            if len(set(token_positions)) != len(token_positions):
                raise ValueError(f"token_positions contains duplicates: {token_positions}")

        self.token_positions: list[int] | None = token_positions

    def edit(self, residual_BSN: t.Tensor, layer: int) -> t.Tensor:
        """A new tensor: ``residual_BSN`` with this edit applied at ``layer``.

        Never edits in place; the input tensor is untouched.
        """
        if layer not in self.layers:
            raise ValueError(
                f"no vector for layer {layer}; this intervention has layers " f"{self.layers}"
            )

        if residual_BSN.shape[-1] != self.d_model:
            raise ValueError(
                f"residual has d_model {residual_BSN.shape[-1]}; intervention "
                f"vectors have {self.d_model}"
            )

        if self.token_positions is None:
            edited_BSN = self._edit_selected(residual_BSN.float(), layer)

            return edited_BSN.to(dtype=residual_BSN.dtype)

        else:
            edited_BSN = residual_BSN.clone()
            selected_BPN = residual_BSN[:, self.token_positions, :].float()

            edited_BPN = self._edit_selected(selected_BPN, layer)

            edited_BSN[:, self.token_positions, :] = edited_BPN.to(dtype=residual_BSN.dtype)

            return edited_BSN

    @abstractmethod
    def _edit_selected(self, residual_BPN: t.Tensor, layer: int) -> t.Tensor:
        """Apply the edit to fp32 residuals at the selected positions."""
        ...
