"""Residual-stream interventions built from J-lens vectors.

A J-lens vector for vocab token ``tok`` at source layer ``l`` is
``v_N = W_U[tok] @ J_l`` (see :func:`workspace_lens.utils.jlens_vectors`), so
``<h, v>`` is that token's pre-final-norm lens logit. Interventions edit the
*output* of block ``l`` — the same residual the lens Jacobians are keyed by.

The pieces:

- :class:`Intervention` — abstract base holding per-layer vectors and the
  masking/dtype plumbing; the concrete edit (:class:`Clamp`) subclasses
  it and implements only :meth:`Intervention._edit_selected`.
- :class:`InterventionHooks` — context manager registering writer forward
  hooks on ``model.layers`` so the edits apply during a forward pass.

Generation/KV-cache note: hooks fire on *every* forward, and
``token_positions`` index each forward's own sequence dimension, so
incremental decoding (one-token forwards) with explicit positions would
misbehave. These utilities are designed for single full forward passes.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch as t

from workspace_lens.interventions.base import Intervention

NEAR_PARALLEL_ABS_COSINE = 0.999  # |cos| above this: a 2-vector pinv is ill-conditioned


def abs_cosine_between(first_vector_N: t.Tensor, second_vector_N: t.Tensor) -> float:
    norm_product = first_vector_N.norm() * second_vector_N.norm()
    return float(t.dot(first_vector_N, second_vector_N).abs() / norm_product)


class Clamp(Intervention):
    """Pin the residual's coordinates on a set of J-lens atoms to given values.

    Per layer, let ``V`` be a basis of ``K`` atom directions (``[d_model, K]``)
    and ``c = pinv(V) @ h`` the coordinates of the residual ``h`` in that
    (generally non-orthogonal) basis -- the lens coordinates ``c = V^+ h``.
    The edit overwrites those coordinates with the given targets ``c*`` and
    touches nothing else:

        h <- h + V @ (c* - c)

    so afterwards ``pinv(V) @ h == c*`` (exactly, in fp32), the component of
    ``h`` orthogonal to ``span(V)`` is unchanged, and the edit is idempotent.
    The targets are fixed in advance (e.g. read off a clean forward pass).

    Basis and targets are given per layer, in the order of the *edited*
    positions (``P = len(token_positions)``, or the sequence length when
    ``token_positions`` is None):

    - atoms: ``[d_model]`` (one atom) or ``[d_model, K]`` (one basis shared by
      every edited position).
    - targets: a float or ``[K]`` (the same values at every edited position)
      or ``[P, K]`` (per position).

    Units: coordinates are in units of each atom as given (``pinv`` units): a
    coordinate of 1 on an un-normalised atom ``v`` contributes ``v``, and the
    J-lens logit ``<v, h>`` of an isolated atom becomes ``c* ||v||^2``. Pass
    unit-normalised atoms for edits of comparable magnitude across lenses, as
    the swap runner does. There is no zero-basis guard: an all-zero basis is a
    no-op.

    Args:
        atoms_L_dict_NK: Per-layer atoms, shapes as above.
        targets_L_dict_K: Per-layer targets, shapes as above; same layer keys.
        token_positions: See :class:`Intervention`.
    """

    def __init__(
        self,
        atoms_L_dict_NK: dict[int, t.Tensor],
        targets_L_dict_K: dict[int, t.Tensor | float],
        *,
        token_positions: Sequence[int] | None = None,
    ) -> None:
        if set(atoms_L_dict_NK) != set(targets_L_dict_K):
            raise ValueError(
                "atoms and targets cover different layers: "
                f"{sorted(atoms_L_dict_NK)} vs {sorted(targets_L_dict_K)}"
            )
        if not atoms_L_dict_NK:
            raise ValueError("Clamp needs at least one layer")

        self.atoms_L_dict_NK: dict[int, t.Tensor] = {}
        self.pinv_L_dict_KN: dict[int, t.Tensor] = {}
        self.targets_L_dict_K: dict[int, t.Tensor] = {}
        for layer, atoms in atoms_L_dict_NK.items():
            atoms_NK = t.as_tensor(atoms).detach().float()
            if atoms_NK.ndim == 1:
                atoms_NK = atoms_NK.unsqueeze(1)  # one atom: [d_model] -> [d_model, 1]
            if atoms_NK.ndim != 2:
                raise ValueError(
                    f"layer {layer}: atoms must be [d_model] or [d_model, K]; "
                    f"got shape {tuple(atoms_NK.shape)}"
                )
            targets_K = t.as_tensor(targets_L_dict_K[layer]).detach().float()
            if targets_K.ndim == 0:
                targets_K = targets_K.reshape(1)
            if targets_K.ndim not in (1, 2):
                raise ValueError(
                    f"layer {layer}: targets must be a float, [K] or [P, K]; got "
                    f"shape {tuple(targets_K.shape)}"
                )
            if targets_K.shape[-1] != atoms_NK.shape[-1]:
                raise ValueError(
                    f"layer {layer}: {atoms_NK.shape[-1]} atoms but "
                    f"{targets_K.shape[-1]} targets"
                )
            self.atoms_L_dict_NK[layer] = atoms_NK
            self.pinv_L_dict_KN[layer] = t.linalg.pinv(atoms_NK)
            self.targets_L_dict_K[layer] = targets_K

        d_models = {atoms_NK.shape[-2] for atoms_NK in self.atoms_L_dict_NK.values()}
        if len(d_models) != 1:
            raise ValueError(f"atoms disagree on d_model across layers: {sorted(d_models)}")

        super().__init__(
            layers=sorted(self.atoms_L_dict_NK),
            d_model=d_models.pop(),
            token_positions=token_positions,
        )

    def _edit_selected(self, residual_BPN: t.Tensor, layer: int) -> t.Tensor:
        atoms_NK = self.atoms_L_dict_NK[layer].to(residual_BPN.device)
        pinv_KN = self.pinv_L_dict_KN[layer].to(residual_BPN.device)
        targets_K = self.targets_L_dict_K[layer].to(residual_BPN.device)

        num_positions = residual_BPN.shape[-2]
        if targets_K.ndim == 2 and targets_K.shape[0] != num_positions:
            raise ValueError(
                f"layer {layer}: per-position targets cover {targets_K.shape[0]} positions "
                f"but {num_positions} positions are being edited"
            )

        coordinates_BPK = residual_BPN @ pinv_KN.T  # c = pinv(V) h, batched
        delta_BPK = targets_K - coordinates_BPK  # broadcasts [K] or [P, K]
        return residual_BPN + delta_BPK @ atoms_NK.T  # h + V (c* - c)
