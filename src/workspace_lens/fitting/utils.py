"""Expert-fit helpers: merging shard checkpoints and turning per-expert sums
into per-expert Jacobians.

Each shard's checkpoint (written by
:meth:`~workspace_lens.fitting.expert_fitting.ExpertJacobianTrainer.write_checkpoint`)
stores one K's :class:`~workspace_lens.fitting.types.ExpertFitSums` — raw
row sums, weight sums, and position counts — so merging is just
summing: it reproduces a single-machine fit over the union of the shards'
prompts exactly (pinned in ``tests/test_expert_fitting.py``).
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass

import torch as t

from workspace_lens.config import LensConfig
from workspace_lens.fitting.types import ExpertFitSums, FitStepForward, LayerClusterSums
from workspace_lens.residual_streams import sum_gradient_over_streams

logger = logging.getLogger(__name__)

# The plain-typed router description the expert trainer writes into each per-K
# checkpoint (key "router"). Every router is a per-layer
# ActivationRouterCollection, so the merge and the resume refuse a checkpoint
# stamped with anything else.
ACTIVATION_COLLECTION_STAMP: dict[str, object] = {"router_kind": "activation_collection"}


def checkpoint_router_stamp(checkpoint_state: dict) -> dict[str, object] | None:
    """The router stamp a per-K shard checkpoint was fitted under (its ``"router"``
    key; ``None`` when the key is missing). The merge and the resume compare stamps
    through this one reader."""
    return checkpoint_state.get("router")


def _check_merge_fit_settings(
    first_config: LensConfig, other_config: LensConfig, path: str
) -> None:
    """Shards must agree on the settings that determine which positions enter
    the sums: ``check_compatible`` covers the lens geometry, this adds the
    remaining estimator knobs."""
    for field_name in (
        "jacobian_rows_per_pass",
        "max_seq_len",
        "skip_first_n_positions",
    ):
        ours = getattr(first_config, field_name)
        theirs = getattr(other_config, field_name)
        if ours != theirs:
            raise ValueError(
                f"{path} was fitted with {field_name}={theirs}, expected {ours}; "
                "merging shards with different fitting settings is not exact"
            )


@dataclass
class MergedExpertCheckpoint:
    """The result of merging one K's shard checkpoints: the summed
    sufficient statistics plus the settings they were fitted under.
    ``config`` is the merged trainer config (``num_prompts_trained_on``
    summed across shards, ``next_prompt_idx`` reset)."""

    num_clusters: int
    fit_sums: ExpertFitSums
    config: LensConfig


def merge_expert_checkpoints(
    checkpoint_paths: list[str], *, device: str = "cpu"
) -> MergedExpertCheckpoint:
    """Merge one K's shard checkpoint files by summing their sufficient
    statistics — exact, because every accumulator is a plain sum over
    disjoint prompt sets.
    """
    if not checkpoint_paths:
        raise ValueError("merge needs at least one checkpoint")

    # Streamed: one checkpoint in memory at a time (a shard checkpoint holds an
    # [E, d_model, d_model] row sum per source layer). The first shard's tensors
    # become the accumulators.
    merged_sums: ExpertFitSums | None = None
    first_num_clusters: int | None = None
    first_lrp_rules: dict | None = None
    first_config: LensConfig | None = None
    total_prompts = 0

    for checkpoint_idx, path in enumerate(checkpoint_paths):
        # device="cuda" streams each shard straight to the GPU and sums there.
        # Large shard files take a while to read, so the loop logs its progress.
        logger.info(
            "merging checkpoint %d/%d: %s", checkpoint_idx + 1, len(checkpoint_paths), path
        )
        state = t.load(path, map_location=device, weights_only=True)
        config = LensConfig.from_dict(state["config"])
        total_prompts += config.num_prompts_trained_on

        # Same K and geometry do not imply the same bucketing: a shard fitted
        # under another router must not merge.
        router_stamp_of_shard = checkpoint_router_stamp(state)
        if router_stamp_of_shard != ACTIVATION_COLLECTION_STAMP:
            raise ValueError(
                f"{path} was fitted under router {router_stamp_of_shard!r}, expected "
                f"{ACTIVATION_COLLECTION_STAMP!r}; the experts are bucketed differently"
            )

        if merged_sums is None:
            first_num_clusters = state["num_clusters"]
            first_lrp_rules = state["lrp_rules"]
            first_config = config
            merged_sums = ExpertFitSums.from_state_dict(state)
        else:
            if state["num_clusters"] != first_num_clusters:
                raise ValueError(
                    f"{path} holds K={state['num_clusters']}, expected {first_num_clusters}"
                )
            # check_compatible below already rejects differing lrp_mode; this
            # additionally pins the *resolved* rule flags each shard was stamped
            # with, in case a preset's definition ever drifts between the shards'
            # code versions.
            lrp_rules = state["lrp_rules"]
            if lrp_rules != first_lrp_rules:
                raise ValueError(
                    f"{path} was fitted under LRP rules {lrp_rules!r}, "
                    f"expected {first_lrp_rules!r}; merging is not meaningful"
                )
            assert first_config is not None
            first_config.check_compatible(config)
            _check_merge_fit_settings(first_config, config, path)
            merged_sums.add_(ExpertFitSums.from_state_dict(state))
        del state

    assert (
        first_config is not None and merged_sums is not None and first_num_clusters is not None
    )
    merged_config = dataclasses.replace(
        first_config,
        num_prompts_trained_on=total_prompts,
        next_prompt_idx=0,
    )
    return MergedExpertCheckpoint(
        num_clusters=first_num_clusters,
        fit_sums=merged_sums,
        config=merged_config,
    )


def build_layer_expert_jacobians(
    layer_sums: LayerClusterSums, *, min_kept_positions: int
) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    """One layer's per-expert Jacobians from its sufficient statistics:
    ``(experts_EFN, pooled_jacobian_FN, fallback_Bool_E)`` (Jacobians are
    ``[F, N]``: target-layer rows, source-layer columns).

    Expert ``e`` is the position-weighted mean ``row_sum_e / weight_sum_e``
    when it kept at least ``min_kept_positions`` positions (and so has a
    positive weight sum: every position has weight 1, so the weight sum is zero
    exactly when the count is). Every other expert is replaced by the pooled
    Jacobian ``sum_e row_sum_e / sum_e weight_sum_e`` — the same matrix
    :func:`~workspace_lens.fitting.expert_fitting.build_matched_baseline_lens`
    uses — and flagged in ``fallback_Bool_E``. With ``min_kept_positions=1``
    only experts that saw no position fall back;
    :meth:`~workspace_lens.fitting.condense_experts.ExpertJacobians.from_sums`
    defaults to 50 so a handful of positions cannot define an expert.

    Raises:
        ValueError: If no expert kept any position at all (the pooled
            Jacobian would be 0 / 0).
    """
    row_sums_EFN = layer_sums.weighted_jacobian_row_sum_EFN
    counts_E = layer_sums.position_count_E
    weight_sums_E = layer_sums.weight_sum_E.float()
    if int(counts_E.sum()) == 0:
        raise ValueError("no positions were fitted at all")

    pooled_jacobian_FN = row_sums_EFN.sum(dim=0) / float(weight_sums_E.sum())
    fallback_Bool_E = (counts_E < min_kept_positions) | (weight_sums_E <= 0)

    experts_EFN = t.empty_like(row_sums_EFN)
    for expert_idx in range(row_sums_EFN.shape[0]):
        if bool(fallback_Bool_E[expert_idx]):
            experts_EFN[expert_idx] = pooled_jacobian_FN
        else:
            experts_EFN[expert_idx] = row_sums_EFN[expert_idx] / float(
                weight_sums_E[expert_idx]
            )
    return experts_EFN, pooled_jacobian_FN, fallback_Bool_E


def expert_checkpoint_filename(num_clusters: int) -> str:
    """The per-K checkpoint file written by :meth:`ExpertJacobianTrainer.
    write_checkpoint`."""
    return f"experts_K{num_clusters}_checkpoint.pt"


def source_gradients_summed_over_streams(
    forward_state: FitStepForward, cotangent_BSF: t.Tensor, *, retain_graph: bool
) -> list[t.Tensor]:
    """One backward pass of the fit: the gradient of the target residual under ``cotangent_BSF`` with
    respect to every source block output, each reduced to the lens's ``[B, S, N]`` gradient by
    :func:`sum_gradient_over_streams` (a no-op on a single-stream model). The per-stream gradients
    contain R times as many elements as their sums, so each is discarded after reduction.
    """
    grads_L_list_BSrN = list(
        t.autograd.grad(
            outputs=forward_state.target_residual_BSF,
            inputs=forward_state.source_block_outputs_L_list_BSrN,
            grad_outputs=cotangent_BSF,
            retain_graph=retain_graph,
        )
    )
    grads_L_list_BSN: list[t.Tensor] = []
    while grads_L_list_BSrN:  # pop as we go: the per-stream gradient is freed once summed
        grads_L_list_BSN.append(sum_gradient_over_streams(grads_L_list_BSrN.pop(0)))
    return grads_L_list_BSN

