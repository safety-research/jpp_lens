"""One K's per-expert Jacobians, before any expert weights are learned, and the
weighted sum that turns them into the inference lens.

:class:`ExpertJacobians` is built from a fit's sufficient statistics
(:meth:`ExpertJacobians.from_sums`), saved and loaded in fp16, and combined into a
:class:`~workspace_lens.lenses.jacobian_lens.JacobianLens` by
:meth:`ExpertJacobians.combine`. :func:`combine_expert_jacobians` is the maths
that does the combining.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass

import torch as t

from workspace_lens.config import LensConfig
from workspace_lens.fitting.types import ExpertFitSums, LayerClusterSums
from workspace_lens.fitting.utils import build_layer_expert_jacobians
from workspace_lens.lenses.jacobian_lens import JacobianLens
from workspace_lens.utils import ensure_parent_dir

logger = logging.getLogger(__name__)


### THE COMBINED LENS


def combine_expert_jacobians(experts_EFN: t.Tensor, weights_E: t.Tensor) -> t.Tensor:
    weighted_jacobian_FN = t.einsum("e,eij->ij", weights_E.to(experts_EFN), experts_EFN)
    return weighted_jacobian_FN


### EXPERT JACOBIANS


def _layer_dict_on_cpu(tensors_L_dict: Mapping[int, t.Tensor]) -> dict[int, t.Tensor]:
    return {layer: tensor.cpu() for layer, tensor in tensors_L_dict.items()}


def _build_experts_for_layer(
    layer: int, layer_sums: LayerClusterSums, *, min_kept_positions: int
) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    """:func:`~workspace_lens.fitting.utils.build_layer_expert_jacobians` for one layer,
    with the layer in the error and the fallback warning (the pure function
    does not know its layer)."""
    try:
        experts_EFN, pooled_jacobian_FN, fallback_Bool_E = build_layer_expert_jacobians(
            layer_sums, min_kept_positions=min_kept_positions
        )
    except ValueError as exc:
        raise ValueError(f"layer {layer}: {exc}") from exc

    if bool(fallback_Bool_E.any()):
        logger.warning(
            "layer %d: experts %s kept fewer than %d position(s) (counts %s); "
            "using the pooled Jacobian for them",
            layer,
            t.where(fallback_Bool_E)[0].tolist(),
            min_kept_positions,
            layer_sums.position_count_E.tolist(),
        )
    return experts_EFN, pooled_jacobian_FN, fallback_Bool_E


@dataclass
class ExpertJacobians:
    """One K's per-expert Jacobians, before the expert weights are learned.

    ``experts_L_dict_EFN[layer][e]`` is ``row_sum_e / weight_sum_e`` for every
    expert that kept at least ``min_kept_positions`` positions, and the
    pooled Jacobian ``pooled_jacobian_L_dict_FN[layer] = sum_e row_sum_e / sum_e
    weight_sum_e`` for the rest (``fallback_L_dict_Bool_E`` marks which; see
    :func:`~workspace_lens.fitting.utils.build_layer_expert_jacobians`).
    ``position_counts_L_dict_E`` / ``weight_sums_L_dict_E`` are the sums'
    per-expert counts and weight totals (equal: every position has weight 1). The two Jacobian dicts hold
    whatever dtype they were built with — fp32 from :meth:`from_sums`, fp16
    when read back by :meth:`load` (the file dtype: :meth:`save` stores them in
    fp16, as ``BaseLens.save`` does, so a multi-GB experts file stays small on
    the host). Consumers cast per layer to fp32 on their device
    (:meth:`~workspace_lens.fitting.condense_experts.fitter.ExpertWeightingFitter._expert_transports_PEF`
    / :meth:`combine`). Everything else is plain tensors / ints, so the file
    loads with ``weights_only=True``.
    """

    experts_L_dict_EFN: dict[int, t.Tensor]
    pooled_jacobian_L_dict_FN: dict[int, t.Tensor]
    position_counts_L_dict_E: dict[int, t.Tensor]
    weight_sums_L_dict_E: dict[int, t.Tensor]
    fallback_L_dict_Bool_E: dict[int, t.Tensor]
    config: LensConfig
    min_kept_positions: int

    @property
    def layers(self) -> list[int]:
        return sorted(self.experts_L_dict_EFN)

    @property
    def num_experts(self) -> int:
        return int(next(iter(self.experts_L_dict_EFN.values())).shape[0])

    @classmethod
    def from_sums(
        cls,
        fit_sums: ExpertFitSums,
        config: LensConfig,
        *,
        min_kept_positions: int = 50,
    ) -> ExpertJacobians:
        """The expert Jacobians from one K's (merged) sufficient statistics:
        :func:`~workspace_lens.fitting.utils.build_layer_expert_jacobians` per
        layer, so an expert that kept fewer than ``min_kept_positions`` positions
        (default 50) is replaced by the layer's pooled Jacobian and flagged.
        ``config`` is the trainer / merged config the sums were fitted under
        (stored as is; the experts are not a lens)."""
        experts_L_dict_EFN: dict[int, t.Tensor] = {}
        pooled_jacobian_L_dict_FN: dict[int, t.Tensor] = {}
        fallback_L_dict_Bool_E: dict[int, t.Tensor] = {}
        position_counts_L_dict_E: dict[int, t.Tensor] = {}
        weight_sums_L_dict_E: dict[int, t.Tensor] = {}
        for layer, layer_sums in fit_sums.layer_sums_L_dict.items():
            experts_EFN, pooled_jacobian_FN, fallback_Bool_E = _build_experts_for_layer(
                layer, layer_sums, min_kept_positions=min_kept_positions
            )
            experts_L_dict_EFN[layer] = experts_EFN
            pooled_jacobian_L_dict_FN[layer] = pooled_jacobian_FN
            fallback_L_dict_Bool_E[layer] = fallback_Bool_E
            position_counts_L_dict_E[layer] = layer_sums.position_count_E.clone()
            weight_sums_L_dict_E[layer] = layer_sums.weight_sum_E.float().clone()
        return cls(
            experts_L_dict_EFN=experts_L_dict_EFN,
            pooled_jacobian_L_dict_FN=pooled_jacobian_L_dict_FN,
            position_counts_L_dict_E=position_counts_L_dict_E,
            weight_sums_L_dict_E=weight_sums_L_dict_E,
            fallback_L_dict_Bool_E=fallback_L_dict_Bool_E,
            config=config,
            min_kept_positions=min_kept_positions,
        )

    def combine(
        self, weights_L_dict_E: Mapping[int, t.Tensor], *, checkpoint_name: str
    ) -> JacobianLens:
        """The inference lens: ``J_l(w) = sum_e w_e J_{l,e}`` at every layer of
        ``weights_L_dict_E`` (:func:`combine_expert_jacobians`; the layers must
        be among :attr:`layers`), combined in fp32 on the CPU whatever the
        experts' stored dtype. Takes the fitted weights per layer — pass
        :attr:`~workspace_lens.fitting.condense_experts.fitter.ExpertWeighting.weights_L_dict_E`
        of a fitted weighting, or a plain ``{layer: weights_E}`` dict. Its
        config is :attr:`config` as a Jacobian lens's with the given
        ``checkpoint_name``
        (:meth:`~workspace_lens.config.LensConfig.as_jacobian`, as
        ``build_matched_baseline_lens`` does). The weights are not stored in the
        lens (``BaseLens.save`` has no metadata slot): write them beside it with
        :meth:`~workspace_lens.fitting.condense_experts.fitter.ExpertWeighting.save`.
        Save with ``dtype=t.float32`` so the file holds the fitted map bit for
        bit (fp16 rounding of the combined Jacobian flips near-tie ranks)."""
        unknown_layers = sorted(set(weights_L_dict_E) - set(self.layers))
        if unknown_layers:
            raise ValueError(
                f"weights given for layers {unknown_layers}, but the experts cover "
                f"{self.layers}"
            )
        combined_L_dict_FN = {
            layer: combine_expert_jacobians(
                self.experts_L_dict_EFN[layer].to(dtype=t.float32), weights_E
            )
            for layer, weights_E in weights_L_dict_E.items()
        }
        return JacobianLens(
            jacobians=combined_L_dict_FN,
            config=self.config.as_jacobian(checkpoint_name=checkpoint_name),
        )

    def pooled_lens(self, *, checkpoint_name: str) -> JacobianLens:
        """The pooled lens of these experts (the R-Lens of an LRP fit): per
        layer the mean over every fit position, ``sum_e row_sum_e / sum_e weight_sum_e``
        (:attr:`pooled_jacobian_L_dict_FN`), as a plain :class:`JacobianLens` with
        :attr:`config` as a Jacobian lens's. It needs no labels and no router at
        inference. fp32 on the CPU whatever the stored dtype; :meth:`save` stores
        the pooled key in fp16, so from a *loaded* file the lens is fp16-rounded
        (near-tie ranks can flip), while from in-memory sums
        (``jpp_cli.py merge-experts --pooled-lens-out``) it is exact."""
        return JacobianLens(
            jacobians={
                layer: pooled_jacobian_FN.to(device="cpu", dtype=t.float32)
                for layer, pooled_jacobian_FN in self.pooled_jacobian_L_dict_FN.items()
            },
            config=self.config.as_jacobian(checkpoint_name=checkpoint_name),
        )

    def save(self, path: str) -> None:
        """The Jacobians in fp16, everything on the CPU whatever device the
        merge ran on (``merge-experts`` sums on the GPU when there is one)."""
        ensure_parent_dir(path)
        t.save(
            {
                "experts": {
                    layer: experts_EFN.half().cpu()
                    for layer, experts_EFN in self.experts_L_dict_EFN.items()
                },
                # "pooled" is the on-disk key of pooled_jacobian_L_dict_FN.
                "pooled": {
                    layer: pooled_jacobian_FN.half().cpu()
                    for layer, pooled_jacobian_FN in (
                        self.pooled_jacobian_L_dict_FN.items()
                    )
                },
                "position_counts": _layer_dict_on_cpu(self.position_counts_L_dict_E),
                "weight_sums": _layer_dict_on_cpu(self.weight_sums_L_dict_E),
                "fallback": _layer_dict_on_cpu(self.fallback_L_dict_Bool_E),
                "config": self.config.to_dict(),
                "min_kept_positions": self.min_kept_positions,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> ExpertJacobians:
        state = t.load(path, map_location="cpu", weights_only=True)
        return cls(
            experts_L_dict_EFN=state["experts"],  # fp16, the file dtype
            pooled_jacobian_L_dict_FN=state["pooled"],  # the on-disk key, see save
            position_counts_L_dict_E=state["position_counts"],
            weight_sums_L_dict_E=state["weight_sums"],
            fallback_L_dict_Bool_E=state["fallback"],
            config=LensConfig.from_dict(state["config"]),
            min_kept_positions=int(state["min_kept_positions"]),
        )
