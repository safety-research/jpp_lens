"""Fitting expert Jacobians: one Jacobian per activation cluster.

Estimator. The J-lens backward passes (shared with :class:`LensTrainer` via
``_run_fit_forward`` / ``_backward_passes``) produce, for every kept position
``p``, the gradient rows ``G_p[i, :] = d(sum_{kept p' >= p} h_target[p', i]) /
d h_l[p]`` — a per-position Jacobian estimate. The plain J-lens averages
``G_p`` over all positions and prompts; here each position is instead assigned
to a cluster ``c(p)`` (nearest centroid of a frozen
:class:`~workspace_lens.routing.router.ActivationRouterCollection`, from the same forward
pass's activation at ``p``) and averages within each cluster. Every position
enters its cluster's sums with weight 1, and each expert Jacobian is the mean::

    J_{l,k} = sum_{p: c(p) = k} G_p  /  sum_{p: c(p) = k} 1

(the denominator is stored as the expert's weight sum, which equals its
position count).

This is a *position-weighted* mean over every member position in the corpus —
the only clean per-cluster estimator, since a prompt can contribute any number
of positions to a cluster, including zero, which makes a per-prompt-mean
ill-defined per cluster. The matched single-Jacobian baseline from the same
run (:func:`build_matched_baseline_lens`) pools the same sums, so
expert-vs-pooled comparisons are like-for-like — slightly different from
:meth:`LensTrainer.fit`'s per-prompt-mean-then-prompt-mean, which weights
every prompt equally.

Routers. Each K's router is a per-layer
:class:`~workspace_lens.routing.router.ActivationRouterCollection`: a position
is routed at every layer from that layer's own activation
(:meth:`ExpertJacobianTrainer._route_positions`).

Which positions enter at all is
:func:`workspace_lens.utils.get_position_mask_with_early_skips` (the
``skip_first_n_positions`` / final-position rule): the cotangent is placed at
the kept positions, and the rows are read, routed, counted and accumulated
there. The set is computed once, in the fit forward, so the routing and the
accumulated rows see the same positions.

Cost. Identical backward passes to a plain J-lens fit: clustering only
changes the reduction over positions. The trainer therefore fits
several cluster counts K in one run for free (each K just buckets the same
gradient rows differently), and shards across GPUs by fitting disjoint prompt
slices — checkpoints store the row *sums*, weight *sums*, and
position *counts* (one :class:`~workspace_lens.fitting.types.ExpertFitSums`
per K), so summing shard checkpoints
(:func:`workspace_lens.fitting.utils.merge_expert_checkpoints`) reproduces a
single-machine fit exactly.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch as t

from jlens.protocol import LensModel
from workspace_lens import DEFAULT_DEVICE
from workspace_lens.config import LensConfig
from workspace_lens.fitting.jacobian_fitting import LensTrainer
from workspace_lens.fitting.types import ExpertFitSums, FitStepForward
from workspace_lens.fitting.utils import (
    ACTIVATION_COLLECTION_STAMP,
    checkpoint_router_stamp,
    expert_checkpoint_filename,
)
from workspace_lens.lenses.jacobian_lens import JacobianLens
from workspace_lens.lrp import lrp_rule_config_for_mode
from workspace_lens.residual_streams import residual_mean_over_streams
from workspace_lens.routing.router import ActivationRouterCollection

logger = logging.getLogger(__name__)

### TRAINER


def _source_position_activations_PN(
    block_output_BSrN: t.Tensor, source_positions_Int_P: t.Tensor
) -> t.Tensor:
    """The prompt's ``[P, N]`` fp32 residual at the kept positions of the first
    replica (every replica carries the same prompt), detached: what a router assigns
    experts from. On a multi-stream model this is the stream mean, the lens's residual
    (:mod:`workspace_lens.residual_streams`)."""
    positions_on_device_Int_P = source_positions_Int_P.to(block_output_BSrN.device)
    return residual_mean_over_streams(block_output_BSrN[:1, positions_on_device_Int_P].detach())[0].float()


class ExpertJacobianTrainer(LensTrainer):
    """Fits per-cluster Jacobians for one or more cluster counts K in one run.

    The backward passes (the entire GPU cost) are shared across Ks: each K
    just buckets the same gradient rows differently, so fitting several Ks
    costs one fit plus each K's accumulator memory. Row contributions are added
    *directly* into the running accumulators — each backward pass covers a
    disjoint output-dim slice, so no per-prompt buffer is needed and CPU
    memory stays at one copy of the accumulators.

    Checkpointing writes one file per K (:func:`expert_checkpoint_filename`,
    atomically overwritten in place), so downstream merging can load and
    merge each K independently and in parallel. Each file is stamped with its
    K's router geometry and the resolved LRP rule flags
    (:meth:`write_checkpoint`), so shards fitted through any entry point (the
    trainers directly, ``scripts/jpp_cli.py fit-shard``) merge on identical
    stamps.
    """

    def __init__(
        self,
        config: LensConfig,
        model: LensModel,
        prompts: list[str],
        *,
        router_collections_K_dict: dict[int, ActivationRouterCollection],
        device: str = DEFAULT_DEVICE,
    ) -> None:
        super().__init__(config, model, prompts, device=device)

        self._validate_router_collections(router_collections_K_dict)

        self.router_collections_K_dict = dict(sorted(router_collections_K_dict.items()))

        self.fit_sums_K_dict: dict[int, ExpertFitSums] = {
            num_clusters: ExpertFitSums.zeros(num_clusters, model.d_model, self.source_layers)
            for num_clusters in self.router_collections_K_dict
        }

    ### VALIDATION AND INITIALIZATION

    def _validate_router_collections(
        self, router_collections_K_dict: dict[int, ActivationRouterCollection]
    ) -> None:
        """Every router's K matches its key and every collection covers every
        source layer."""
        for num_clusters, router in router_collections_K_dict.items():
            if router.num_clusters != num_clusters:
                raise ValueError(
                    f"router_collections_K_dict[{num_clusters}] has "
                    f"num_clusters={router.num_clusters}"
                )
            # A per-layer collection or a duck-typed stand-in for one (it needs
            # `layer_routers_L_dict` keys and `assign(activations_PN, layer)`).
            missing_layers = set(self.source_layers) - set(router.layer_routers_L_dict)
            if missing_layers:
                raise ValueError(
                    f"router_collections_K_dict[{num_clusters}] has no router for source "
                    f"layers {sorted(missing_layers)}"
                )

    def _init_accumulators(self) -> None:
        """The base J-lens accumulator is unused here; this trainer allocates
        its own accumulators in __init__ (they need the router collections, which the
        base __init__ has not seen yet)."""
        self.jacobian_sum_L_dict_FN: dict[int, t.Tensor] = {}

    ### TRAINING

    def fit_step(  # type: ignore[override]
        self,
        prompt: str,
        *,
        prompt_idx: Optional[int] = None,  # noqa: UP045
    ) -> tuple[int, int]:
        """Accumulate one prompt directly into the running sums (all Ks).

        Returns ``(seq_len, num_source_positions)``. Side effect: appends this
        prompt's position record (:meth:`LensTrainer._append_position_record`;
        ``prompt_idx`` is the prompt's index in the trainer's list, ``None`` stand-alone).
        """
        # NOTE: this method mutates the running accumulators (counts and
        # weight sums inside compute_cluster_memberships, row sums in the pass
        # loop below). That is safe only because the sole skippable error —
        # the short-prompt ValueError that _run_prompt_loop catches — is raised by
        # _run_fit_forward, before any mutation.

        forward_state = self._run_fit_forward(prompt)
        source_positions_Int_P = forward_state.source_positions_Int_P

        cluster_membership_K_dict_L_dict_EP = self.compute_cluster_memberships(forward_state)

        ### Fit a prompt's gradient rows into the running accumulators (one backward pass per output dim slice).
        for dim_start, current_pass_batch_dim, grads_L_list_BSN in self._backward_passes(
            forward_state
        ):

            for layer, grad_BSN in zip(self.source_layers, grads_L_list_BSN, strict=True):
                positions_on_device_Int_P = source_positions_Int_P.to(
                    grad_BSN.device, non_blocking=True
                )

                source_grad_BPN = grad_BSN[
                    :current_pass_batch_dim, positions_on_device_Int_P, :
                ].float()

                for num_clusters in self.router_collections_K_dict:
                    cluster_membership_EP = cluster_membership_K_dict_L_dict_EP[num_clusters][
                        layer
                    ].to(source_grad_BPN.device)
                    weighted_jacobian_row_sums_EBN = t.einsum(
                        "ep,bpn->ebn", cluster_membership_EP, source_grad_BPN
                    )

                    # alias into the persistent accumulator: the += below mutates trainer state
                    running_layer_cluster_sums = self.fit_sums_K_dict[
                        num_clusters
                    ].layer_sums_L_dict[layer]

                    # Batch element b IS output row dim_start + b and passes
                    # cover disjoint dim slices, so `+=` into the running
                    # accumulator adds this prompt's contribution exactly once.
                    running_layer_cluster_sums.weighted_jacobian_row_sum_EFN[
                        :, dim_start : dim_start + current_pass_batch_dim, :
                    ] += weighted_jacobian_row_sums_EBN.cpu()

        self._append_position_record(forward_state, prompt_idx)
        return forward_state.seq_len, forward_state.num_source_positions

    def compute_cluster_memberships(
        self, forward_state: FitStepForward
    ) -> dict[int, dict[int, t.Tensor]]:
        """One prompt's routing, ahead of the backward passes.

        Returns ``cluster_membership_K_dict_L_dict_EP``: per (K, layer) the
        ``[E, P]`` matrix whose entry (e, p) is 1 if position ``p`` routes to
        expert ``e`` and 0 otherwise.

        Side effect: accumulates this prompt's position counts and weight
        sums into ``self.fit_sums_K_dict`` — they are per-prompt quantities,
        unlike the row sums, which :meth:`fit_step` accumulates per backward
        pass. Every position has weight 1, so the weight sums equal the counts.
        """
        # Cluster-membership matrices per (K, layer); counts and weight sums
        # accumulate immediately (they are per-prompt, not per-pass).
        cluster_membership_K_dict_L_dict_EP: dict[int, dict[int, t.Tensor]] = {}
        for num_clusters, router in self.router_collections_K_dict.items():
            cluster_membership_L_dict_EP: dict[int, t.Tensor] = {}
            assignments_L_dict_P = self._route_positions(router, forward_state)

            for layer in self.source_layers:
                assignments_Int_P = assignments_L_dict_P[layer]
                cluster_indices_Int_E = t.arange(num_clusters, device=assignments_Int_P.device)
                cluster_membership_EP = (
                    assignments_Int_P[None, :] == cluster_indices_Int_E[:, None]
                ).float()
                cluster_membership_L_dict_EP[layer] = cluster_membership_EP

                running_layer_cluster_sums = self.fit_sums_K_dict[
                    num_clusters
                ].layer_sums_L_dict[layer]

                running_layer_cluster_sums.position_count_E += (
                    cluster_membership_EP.sum(dim=1).long().cpu()
                )
                running_layer_cluster_sums.weight_sum_E += (
                    cluster_membership_EP.sum(dim=1).float().cpu()
                )

            cluster_membership_K_dict_L_dict_EP[num_clusters] = cluster_membership_L_dict_EP

        return cluster_membership_K_dict_L_dict_EP

    def _route_positions(
        self, router: ActivationRouterCollection, forward_state: FitStepForward
    ) -> dict[int, t.Tensor]:
        """Per source layer, the ``[P]`` expert index of every kept position
        under ``router``: each layer is routed from that layer's own activation at
        the position."""
        source_positions_Int_P = forward_state.source_positions_Int_P
        return {
            layer: router.assign(
                _source_position_activations_PN(source_block_output_BSrN, source_positions_Int_P),
                layer,
            )
            for layer, source_block_output_BSrN in zip(
                self.source_layers, forward_state.source_block_outputs_L_list_BSrN, strict=True
            )
        }

    def _fit_prompt(self, prompt_idx: int, prompt: str, prompt_start_time: float) -> None:
        """One prompt of :meth:`LensTrainer._run_prompt_loop`: fit_step
        accumulates directly into the running sums; only logging remains."""
        self.fit_step(prompt, prompt_idx=prompt_idx)
        logger.info(self._prompt_progress_line(prompt_idx, prompt_start_time))

    def fit(self) -> dict[int, ExpertFitSums]:  # type: ignore[override]
        """Fit all Ks over ``self.prompts``; returns each K's sufficient statistics
        (:meth:`~workspace_lens.fitting.condense_experts.ExpertJacobians.from_sums`
        turns them into expert Jacobians). The per-K checkpoints hold the same sums."""
        logger.info(self.config)
        logger.info("expert fit: Ks=%s", sorted(self.router_collections_K_dict))

        self._run_prompt_loop()
        logger.info("expert fit: done, %d prompts", self.completed_prompt_count)
        return self.fit_sums_K_dict

    ### SAVING AND LOADING

    def _extra_checkpoint_state(self) -> dict[str, object]:
        """Extra keys embedded in every per-K checkpoint, all plain
        (``weights_only=True`` safe): the resolved LRP rule flags of
        ``config.lrp_mode`` (all off for ``"none"``), stamped by every trainer
        so the merge compares shards from either trainer on the same key.
        Subclasses extend it (``**super()._extra_checkpoint_state()``)."""
        return {"lrp_rules": lrp_rule_config_for_mode(self.config.lrp_mode).to_dict()}

    def write_checkpoint(self, *, final: bool = False) -> None:
        """One file per K, atomically overwritten in place (bounded disk). Each
        file describes its own sums: the config with that K's router geometry
        (:meth:`~workspace_lens.config.LensConfig.with_router`), the router's kind
        (:data:`~workspace_lens.fitting.utils.ACTIVATION_COLLECTION_STAMP`, under
        ``"router"``) plus :meth:`_extra_checkpoint_state`, so the merge can
        compare shards on the stamps alone."""
        self.config.num_prompts_trained_on = self.completed_prompt_count
        self.config.next_prompt_idx = self.next_prompt_idx
        os.makedirs(self.config.checkpoint_path, exist_ok=True)
        for num_clusters, router_collection in self.router_collections_K_dict.items():
            checkpoint_config = self.config.with_router(
                num_clusters=num_clusters, projection_dim=router_collection.projection_dim
            )
            self._atomic_save(
                {
                    "num_clusters": num_clusters,
                    "config": checkpoint_config.to_dict(),
                    "router": ACTIVATION_COLLECTION_STAMP,
                    # Per fitted prompt: token hash and kept positions
                    # (LensTrainer._append_position_record).
                    "position_records": self.position_records,
                    **self._extra_checkpoint_state(),
                    **self.fit_sums_K_dict[num_clusters].to_state_dict(),
                },
                os.path.join(
                    self.config.checkpoint_path, expert_checkpoint_filename(num_clusters)
                ),
            )
        self._write_position_records_json()

    @classmethod
    def from_checkpoint_dir(
        cls,
        checkpoint_dir: str,
        prompts: list[str],
        *,
        router_collections_K_dict: dict[int, ActivationRouterCollection],
        model: LensModel | None = None,
        device: str = DEFAULT_DEVICE,
    ) -> ExpertJacobianTrainer:
        """Resume from the per-K checkpoint files in ``checkpoint_dir``.
        ``prompts`` and ``router_collections_K_dict`` must be the ones the run started with."""
        states: dict[int, dict] = {}
        for num_clusters in sorted(router_collections_K_dict):
            path = os.path.join(checkpoint_dir, expert_checkpoint_filename(num_clusters))
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            states[num_clusters] = t.load(path, map_location="cpu", weights_only=True)

        # A resume under a different router would mix two bucketings in one accumulator
        # and re-stamp the checkpoint, so the merge could no longer tell.
        for num_clusters, state in states.items():
            stamp_of_checkpoint = checkpoint_router_stamp(state)
            if stamp_of_checkpoint != ACTIVATION_COLLECTION_STAMP:
                raise ValueError(
                    f"{checkpoint_dir} K={num_clusters} was fitted under router "
                    f"{stamp_of_checkpoint!r}, but the resume passes "
                    f"{ACTIVATION_COLLECTION_STAMP!r}"
                )

        configs = {
            num_clusters: LensConfig.from_dict(state["config"])
            for num_clusters, state in states.items()
        }
        next_indices = {config.next_prompt_idx for config in configs.values()}
        if len(next_indices) != 1:
            raise ValueError(
                f"per-K checkpoints disagree on next_prompt_idx: {sorted(next_indices)}"
            )
        # The per-K files carry that K's router geometry (write_checkpoint);
        # the trainer's own config is K-agnostic, so reset the router fields.
        first_config = next(iter(configs.values()))
        first_config = first_config.as_jacobian(checkpoint_name=first_config.checkpoint_name)

        model = cls._resolve_model(first_config, model)

        first_state = next(iter(states.values()))

        trainer = cls(
            config=first_config,
            model=model,
            prompts=prompts,
            router_collections_K_dict=router_collections_K_dict,
            device=device,
        )
        for num_clusters, state in states.items():
            trainer.fit_sums_K_dict[num_clusters] = ExpertFitSums.from_state_dict(state)
        trainer.completed_prompt_count = first_config.num_prompts_trained_on
        trainer.next_prompt_idx = first_config.next_prompt_idx
        trainer.position_records = list(first_state["position_records"])
        logger.info(
            "  resuming expert fit from %s: %d/%d prompts processed",
            checkpoint_dir,
            trainer.next_prompt_idx,
            len(prompts),
        )
        return trainer


### THE POOLED JACOBIAN OF THE SAME SUMS


def build_matched_baseline_lens(
    fit_sums: ExpertFitSums,
    config: LensConfig,
    checkpoint_name: str,
) -> JacobianLens:
    """The position-weighted single-Jacobian baseline over the same fit:
    ``J_l = sum_k row_sums[k] / sum_k weight_sums[k]`` (see module docstring
    for how this differs from :meth:`LensTrainer.fit`'s prompt weighting)."""
    baseline_jacobians_L_dict_FN = {
        layer: layer_sums.weighted_jacobian_row_sum_EFN.sum(dim=0)
        / float(layer_sums.weight_sum_E.sum())
        for layer, layer_sums in fit_sums.layer_sums_L_dict.items()
    }
    return JacobianLens(
        jacobians=baseline_jacobians_L_dict_FN,
        config=config.as_jacobian(checkpoint_name=checkpoint_name),
    )
