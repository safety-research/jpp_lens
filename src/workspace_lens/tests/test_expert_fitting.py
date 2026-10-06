"""ExpertJacobianTrainer end-to-end on the tiny CPU model.

The pins that matter, in order of strength:

1. With one cluster the expert estimator must reproduce the plain J-lens
   estimator (and its weight sums must equal the position counts
   bit-for-bit: every position has weight 1).
2. The accumulators must match a brute-force per-cluster ``sum G_p``
   computed from exact per-position Jacobians (obtained by slicing the
   estimator's own gradient rows position-by-position).
3. The matched baseline (pooled over clusters) must be invariant to K:
   bucketing then re-pooling cannot change the pooled average.
4. A multi-K fit must equal independent single-K fits (the shared backward
   passes make this checkable).
5. Merging shard checkpoints must equal a single-machine fit over the union
   (the accumulators are plain sums), and resume from the per-K checkpoint
   files is bit-exact.
6. Checkpoint state dicts round-trip bit-exactly.
7. Every per-K checkpoint is stamped with its K's router geometry, the router
   kind and the resolved LRP rules, and a resumed trainer's config stays
   K-agnostic; shards stamped with another router kind refuse to merge or
   resume.
8. At threshold 1 every populated expert of a real fit is its own mean
   ``row_sum_e / count_e``; ``ExpertJacobians.from_sums`` applies the
   starved-expert fallback at its threshold and ``ExpertJacobians`` round-trips
   through fp16 storage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
import torch

from jlens.tests.tiny import TinyDecoder
from workspace_lens.fitting.condense_experts import ExpertJacobians
from workspace_lens.fitting.expert_fitting import (
    ExpertJacobianTrainer,
    build_matched_baseline_lens,
)
from workspace_lens.fitting.jacobian_fitting import LensTrainer
from workspace_lens.fitting.types import ExpertFitSums, LayerClusterSums
from workspace_lens.fitting.utils import (
    build_layer_expert_jacobians,
    expert_checkpoint_filename,
    merge_expert_checkpoints,
)
from workspace_lens.lrp import lrp_rule_config_for_mode
from workspace_lens.residual_streams import residual_mean_over_streams
from workspace_lens.routing.cluster import (
    collect_valid_position_activations,
)
from workspace_lens.routing.router import (
    ActivationRouterCollection,
    LayerActivationRouter,
)
from workspace_lens.tests.fixtures import (
    exact_rows_per_source_PFN,
    make_tiny_lens_config,
)

D_MODEL = 8
SOURCE_LAYERS = [0, 1, 2]
PROMPTS = [
    "abcdefghij " * 5,
    "klmnopqrst " * 5,
    "uvwxyzabcd " * 5,
    "efghijklmn " * 5,
]


@pytest.fixture(scope="module")
def model() -> TinyDecoder:
    return TinyDecoder(n_layers=4, d_model=D_MODEL)


@pytest.fixture(scope="module")
def fitted_router_collections_K_dict(
    model: TinyDecoder,
) -> dict[int, ActivationRouterCollection]:
    """K=2 and K=3 router collections fitted on the test prompts' real activations."""
    activations_L_dict_PN = collect_valid_position_activations(
        model, PROMPTS, SOURCE_LAYERS, skip_first_n_positions=16, max_seq_len=64
    )
    fitted: dict[int, ActivationRouterCollection] = {}
    for num_clusters in (2, 3):
        router_collection = ActivationRouterCollection(num_clusters, projection_dim=4, seed=0)
        router_collection.fit(activations_L_dict_PN)
        fitted[num_clusters] = router_collection
    return fitted


def single_cluster_router_collection() -> ActivationRouterCollection:
    """K=1, identity projection: every position lands in cluster 0."""
    router_collection = ActivationRouterCollection(
        num_clusters=1, projection_dim=D_MODEL, seed=0
    )
    router_collection.layer_routers_L_dict = {
        layer: LayerActivationRouter(
            pca_mean_N=torch.zeros(D_MODEL),
            pca_components_ND=torch.eye(D_MODEL),
            centroids_ED=torch.zeros(1, D_MODEL),
        )
        for layer in SOURCE_LAYERS
    }
    return router_collection


def fit_expert_jacobians(trainer: ExpertJacobianTrainer) -> dict[int, ExpertJacobians]:
    """Run ``trainer.fit()`` and build each K's experts at threshold 1, so only an
    expert that saw no position falls back to the pooled Jacobian."""
    fit_sums_K_dict = trainer.fit()
    return {
        num_clusters: ExpertJacobians.from_sums(fit_sums, trainer.config, min_kept_positions=1)
        for num_clusters, fit_sums in fit_sums_K_dict.items()
    }


def exact_per_position_jacobians(trainer: LensTrainer, prompt: str) -> dict[int, torch.Tensor]:
    """Exact ``G_p`` for every kept source, ``{layer: [P, d, d]}`` in fp32 (no reduction):
    :func:`exact_rows_per_source_PFN` on a fresh forward."""
    return exact_rows_per_source_PFN(trainer, trainer._run_fit_forward(prompt))


def test_single_cluster_matches_base_estimator(
    model: TinyDecoder, tmp_path: Path
) -> None:
    """K=1 fit_step == LensTrainer.fit_step's position mean, per layer.

    Both run the identical forward/backward passes; only the reduction
    differs (weighted-sum-then-divide vs mean), so this pins the cluster
    bucketing and the einsum reduction against the already-verified base
    estimator. Every position has weight 1, which the weight-sum == count
    assertions check bit-for-bit.
    """
    prompt = PROMPTS[0]
    base_trainer = LensTrainer(make_tiny_lens_config(tmp_path, "base"), model, prompts=[])
    base_jacobians_L_dict_FN, _, base_n_valid = base_trainer.fit_step(prompt)

    expert_trainer = ExpertJacobianTrainer(
        make_tiny_lens_config(tmp_path, "experts-k1"),
        model,
        prompts=[],
        router_collections_K_dict={1: single_cluster_router_collection()},
    )
    _, n_valid = expert_trainer.fit_step(prompt)

    assert n_valid == base_n_valid
    for layer in SOURCE_LAYERS:
        counts_E = expert_trainer.fit_sums_K_dict[1].layer_sums_L_dict[layer].position_count_E
        assert counts_E.tolist() == [base_n_valid]
        torch.testing.assert_close(
            expert_trainer.fit_sums_K_dict[1].layer_sums_L_dict[layer].weight_sum_E,
            counts_E.float(),
            rtol=0,
            atol=0,
        )
        expert_jacobian_FN = (
            expert_trainer.fit_sums_K_dict[1]
            .layer_sums_L_dict[layer]
            .weighted_jacobian_row_sum_EFN[0]
            / base_n_valid
        )
        torch.testing.assert_close(
            expert_jacobian_FN,
            base_jacobians_L_dict_FN[layer],
            atol=1e-6,
            rtol=1e-5,
        )


def test_matched_baseline_invariant_to_clustering(
    model: TinyDecoder,
    fitted_router_collections_K_dict: dict[int, ActivationRouterCollection],
    tmp_path: Path,
) -> None:
    """Pooling the K=2 accumulators reproduces the K=1 expert exactly, and —
    for these equal-length prompts — the plain prompt-averaged J-lens too."""
    trainer = ExpertJacobianTrainer(
        make_tiny_lens_config(tmp_path, "pool-invariance"),
        model,
        prompts=list(PROMPTS),
        router_collections_K_dict={
            1: single_cluster_router_collection(),
            2: fitted_router_collections_K_dict[2],
        },
    )
    experts_K_dict = fit_expert_jacobians(trainer)
    baseline_lens = build_matched_baseline_lens(
        trainer.fit_sums_K_dict[2], trainer.config, "pool-invariance-matched"
    )

    # Per-layer counts add up to exactly every valid position of every prompt
    # (each position lands in exactly one cluster). The -16 mirrors the
    # skip_first_n_positions config default; -1 drops the final position.
    prompt_lengths = [model.encode(prompt, max_length=64).shape[1] for prompt in PROMPTS]
    # Equal lengths are what make position-weighting == prompt-weighting below.
    assert len(set(prompt_lengths)) == 1
    expected_total_positions = sum(length - 16 - 1 for length in prompt_lengths)
    for layer in SOURCE_LAYERS:
        counts_E = experts_K_dict[2].position_counts_L_dict_E[layer]
        assert counts_E.shape == (2,)
        assert int(counts_E.sum()) == expected_total_positions

    base_lens = LensTrainer(make_tiny_lens_config(tmp_path, "plain"), model, prompts=list(PROMPTS)).fit()

    for layer in SOURCE_LAYERS:
        pooled_baseline_FN = baseline_lens.jacobians_L_dict_FN[layer]
        k1_expert_FN = experts_K_dict[1].experts_L_dict_EFN[layer][0]
        torch.testing.assert_close(pooled_baseline_FN, k1_expert_FN, atol=1e-6, rtol=1e-5)
        # Equal-length prompts: position weighting == prompt weighting.
        torch.testing.assert_close(
            pooled_baseline_FN,
            base_lens.jacobians_L_dict_FN[layer],
            atol=1e-5,
            rtol=1e-4,
        )

    # TinyDecoder's blocks are position-wise linear, so the true Jacobian is
    # the same matrix at every position: on a linear model every expert must
    # equal the baseline no matter how the positions are clustered. (Experts
    # only diverge on genuinely non-linear models.)
    for layer in SOURCE_LAYERS:
        for cluster_idx in range(2):
            torch.testing.assert_close(
                experts_K_dict[2].experts_L_dict_EFN[layer][cluster_idx],
                baseline_lens.jacobians_L_dict_FN[layer],
                atol=1e-5,
                rtol=1e-4,
            )


def test_experts_diverge_on_a_nonlinear_model(tmp_path: Path) -> None:
    """On a model whose blocks are genuinely non-linear, per-cluster experts
    must differ — this would catch a routing bug where every expert silently
    receives all positions (which K=1 equivalence cannot detect)."""
    import torch.nn as nn

    class _TanhBlock(nn.Module):
        def __init__(self, d_model: int) -> None:
            super().__init__()
            self.linear = nn.Linear(d_model, d_model, bias=False)

        def forward(self, hidden_BSN: torch.Tensor) -> torch.Tensor:
            return hidden_BSN + 0.5 * torch.tanh(self.linear(hidden_BSN))

    torch.manual_seed(0)
    model = TinyDecoder(n_layers=4, d_model=D_MODEL)
    model.layers = nn.ModuleList([_TanhBlock(D_MODEL) for _ in range(4)])

    activations_L_dict_PN = collect_valid_position_activations(
        model, PROMPTS, SOURCE_LAYERS, skip_first_n_positions=16, max_seq_len=64
    )
    router_collection = ActivationRouterCollection(num_clusters=2, projection_dim=4, seed=0)
    router_collection.fit(activations_L_dict_PN)

    trainer = ExpertJacobianTrainer(
        make_tiny_lens_config(tmp_path, "nonlinear"),
        model,
        prompts=list(PROMPTS),
        router_collections_K_dict={2: router_collection},
    )
    experts = fit_expert_jacobians(trainer)[2]

    num_layers_checked = 0
    for layer in SOURCE_LAYERS:
        counts_E = experts.position_counts_L_dict_E[layer]
        if int(counts_E.min()) == 0:
            continue  # a degenerate clustering can't separate experts
        num_layers_checked += 1
        expert_difference = (
            experts.experts_L_dict_EFN[layer][0] - experts.experts_L_dict_EFN[layer][1]
        ).norm()
        # 1e-3 = well above fp accumulation noise for O(1) Jacobians (the
        # linear-model invariance test sees ~1e-7 differences).
        assert expert_difference > 1e-3, f"layer {layer}: experts identical"
    assert num_layers_checked > 0, "every layer clustering was degenerate"


def test_sums_match_bruteforce_cluster_sums(
    model: TinyDecoder,
    fitted_router_collections_K_dict: dict[int, ActivationRouterCollection],
    tmp_path: Path,
) -> None:
    """The accumulators must equal per-cluster sums of ``G_p`` computed from
    exact per-position Jacobians, and the weight sums the member counts."""
    prompt = PROMPTS[0]
    trainer = ExpertJacobianTrainer(
        make_tiny_lens_config(tmp_path, "bruteforce"),
        model,
        prompts=[],
        router_collections_K_dict={2: fitted_router_collections_K_dict[2]},
    )
    trainer.fit_step(prompt)

    forward_state = trainer._run_fit_forward(prompt)
    exact_jacobians_L_dict_PFN = exact_per_position_jacobians(trainer, prompt)

    for layer in SOURCE_LAYERS:
        source_activation_BSN = residual_mean_over_streams(
            forward_state.source_block_outputs_L_list_BSrN[SOURCE_LAYERS.index(layer)]
        )  # the routing reads this mean at the valid positions of replica 0
        assignments_Int_P = fitted_router_collections_K_dict[2].assign(
            source_activation_BSN[0, forward_state.source_positions_Int_P, :].detach().float(),
            layer,
        )
        for cluster_idx in range(2):
            member_mask_Bool_P = assignments_Int_P == cluster_idx
            expected_sum_FN = exact_jacobians_L_dict_PFN[layer][member_mask_Bool_P].sum(dim=0)
            torch.testing.assert_close(
                trainer.fit_sums_K_dict[2]
                .layer_sums_L_dict[layer]
                .weighted_jacobian_row_sum_EFN[cluster_idx],
                expected_sum_FN,
                atol=1e-5,
                rtol=1e-4,
            )
            assert float(
                trainer.fit_sums_K_dict[2].layer_sums_L_dict[layer].weight_sum_E[cluster_idx]
            ) == float(member_mask_Bool_P.sum())


def test_multi_k_fit_matches_independent_single_k_fits(
    model: TinyDecoder,
    fitted_router_collections_K_dict: dict[int, ActivationRouterCollection],
    tmp_path: Path,
) -> None:
    """One fit over K in {2, 3} must equal two independent single-K fits:
    same weighted row sums, weight sums, counts, and expert Jacobians. This pins
    that the multi-K bucketing keeps the Ks fully independent."""
    multi_k_trainer = ExpertJacobianTrainer(
        make_tiny_lens_config(tmp_path, "multi-k"),
        model,
        prompts=list(PROMPTS),
        router_collections_K_dict=fitted_router_collections_K_dict,
    )
    multi_k_experts_K_dict = fit_expert_jacobians(multi_k_trainer)
    assert sorted(multi_k_experts_K_dict) == [2, 3]

    for num_clusters, router_collection in fitted_router_collections_K_dict.items():
        single_trainer = ExpertJacobianTrainer(
            make_tiny_lens_config(tmp_path, f"single-k{num_clusters}"),
            model,
            prompts=list(PROMPTS),
            router_collections_K_dict={num_clusters: router_collection},
        )
        single_experts_K_dict = fit_expert_jacobians(single_trainer)
        for layer in SOURCE_LAYERS:
            torch.testing.assert_close(
                multi_k_trainer.fit_sums_K_dict[num_clusters]
                .layer_sums_L_dict[layer]
                .weighted_jacobian_row_sum_EFN,
                single_trainer.fit_sums_K_dict[num_clusters]
                .layer_sums_L_dict[layer]
                .weighted_jacobian_row_sum_EFN,
                atol=1e-6,
                rtol=1e-5,
            )
            torch.testing.assert_close(
                multi_k_trainer.fit_sums_K_dict[num_clusters]
                .layer_sums_L_dict[layer]
                .weight_sum_E,
                single_trainer.fit_sums_K_dict[num_clusters]
                .layer_sums_L_dict[layer]
                .weight_sum_E,
                atol=1e-6,
                rtol=1e-5,
            )
            assert (
                multi_k_trainer.fit_sums_K_dict[num_clusters]
                .layer_sums_L_dict[layer]
                .position_count_E
                == single_trainer.fit_sums_K_dict[num_clusters]
                .layer_sums_L_dict[layer]
                .position_count_E
            ).all()
            torch.testing.assert_close(
                multi_k_experts_K_dict[num_clusters].experts_L_dict_EFN[layer],
                single_experts_K_dict[num_clusters].experts_L_dict_EFN[layer],
                atol=1e-6,
                rtol=1e-5,
            )


def test_shard_merge_equals_single_run(
    model: TinyDecoder,
    fitted_router_collections_K_dict: dict[int, ActivationRouterCollection],
    tmp_path: Path,
) -> None:
    """Two 2-prompt shards merged == one 4-prompt fit (same sums, weight
    sums, and counts), and the merged accumulators build identical expert
    Jacobians and matched baselines."""
    full_trainer = ExpertJacobianTrainer(
        make_tiny_lens_config(tmp_path, "full"),
        model,
        prompts=list(PROMPTS),
        router_collections_K_dict=fitted_router_collections_K_dict,
    )
    full_experts_K_dict = fit_expert_jacobians(full_trainer)

    shard_dirs: list[str] = []
    for shard_idx, shard_prompts in enumerate([PROMPTS[:2], PROMPTS[2:]]):
        shard_trainer = ExpertJacobianTrainer(
            make_tiny_lens_config(tmp_path, f"shard{shard_idx}"),
            model,
            prompts=list(shard_prompts),
            router_collections_K_dict=fitted_router_collections_K_dict,
        )
        shard_trainer.fit()
        shard_dirs.append(shard_trainer.config.checkpoint_path)

    for num_clusters in fitted_router_collections_K_dict:
        checkpoint_paths = [
            str(Path(shard_dir) / expert_checkpoint_filename(num_clusters))
            for shard_dir in shard_dirs
        ]
        merged = merge_expert_checkpoints(checkpoint_paths)
        assert merged.num_clusters == num_clusters
        assert merged.config.num_prompts_trained_on == len(PROMPTS)
        for layer in SOURCE_LAYERS:
            merged_layer_sums = merged.fit_sums.layer_sums_L_dict[layer]
            full_layer_sums = full_trainer.fit_sums_K_dict[num_clusters].layer_sums_L_dict[
                layer
            ]
            torch.testing.assert_close(
                merged_layer_sums.weighted_jacobian_row_sum_EFN,
                full_layer_sums.weighted_jacobian_row_sum_EFN,
                atol=1e-6,
                rtol=1e-5,
            )
            torch.testing.assert_close(
                merged_layer_sums.weight_sum_E,
                full_layer_sums.weight_sum_E,
                atol=1e-6,
                rtol=1e-5,
            )
            assert (
                merged_layer_sums.position_count_E == full_layer_sums.position_count_E
            ).all()

        merged_experts = ExpertJacobians.from_sums(
            merged.fit_sums, merged.config, min_kept_positions=1
        )
        merged_baseline = build_matched_baseline_lens(
            merged.fit_sums, merged.config, "merged-matched"
        )
        full_baseline = build_matched_baseline_lens(
            full_trainer.fit_sums_K_dict[num_clusters], full_trainer.config, "full-matched"
        )
        for layer in SOURCE_LAYERS:
            torch.testing.assert_close(
                merged_experts.experts_L_dict_EFN[layer],
                full_experts_K_dict[num_clusters].experts_L_dict_EFN[layer],
                atol=1e-6,
                rtol=1e-5,
            )
            torch.testing.assert_close(
                merged_baseline.jacobians_L_dict_FN[layer],
                full_baseline.jacobians_L_dict_FN[layer],
                atol=1e-6,
                rtol=1e-5,
            )


def test_checkpoint_resume_bit_exact(
    model: TinyDecoder,
    fitted_router_collections_K_dict: dict[int, ActivationRouterCollection],
    tmp_path: Path,
) -> None:
    """Resuming from the per-K checkpoint files reproduces the uninterrupted
    fit exactly (same accumulation order)."""
    uninterrupted_trainer = ExpertJacobianTrainer(
        make_tiny_lens_config(tmp_path, "uninterrupted", checkpoint_every_n_prompts=2),
        model,
        prompts=list(PROMPTS),
        router_collections_K_dict=fitted_router_collections_K_dict,
    )
    uninterrupted_experts_K_dict = fit_expert_jacobians(uninterrupted_trainer)

    # Rebuild a fresh partial run to produce the mid-fit checkpoint state.
    partial_trainer = ExpertJacobianTrainer(
        make_tiny_lens_config(tmp_path, "partial", checkpoint_every_n_prompts=2),
        model,
        prompts=list(PROMPTS[:2]),
        router_collections_K_dict=fitted_router_collections_K_dict,
    )
    partial_trainer.fit()

    resumed_trainer = ExpertJacobianTrainer.from_checkpoint_dir(
        partial_trainer.config.checkpoint_path,
        list(PROMPTS),
        router_collections_K_dict=fitted_router_collections_K_dict,
        model=model,
    )
    assert resumed_trainer.next_prompt_idx == 2
    resumed_experts_K_dict = fit_expert_jacobians(resumed_trainer)

    for num_clusters in fitted_router_collections_K_dict:
        for layer in SOURCE_LAYERS:
            torch.testing.assert_close(
                resumed_experts_K_dict[num_clusters].experts_L_dict_EFN[layer],
                uninterrupted_experts_K_dict[num_clusters].experts_L_dict_EFN[layer],
                rtol=0,
                atol=0,
            )


def test_empty_cluster_falls_back_to_baseline(
    model: TinyDecoder, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An expert that saw zero positions gets the pooled Jacobian — the
    matched baseline of build_matched_baseline_lens — (and a warning), rather
    than a meaningless zero matrix. ``build_layer_expert_jacobians`` at threshold 1 and
    ``ExpertJacobians.from_sums(min_kept_positions=1)`` implement the same rule and
    flag exactly that expert. ``fit()`` returns the trainer's own per-K sums."""
    # Centroid 0 is unreachably far in projected space; every position -> 1.
    router_collection = ActivationRouterCollection(
        num_clusters=2, projection_dim=D_MODEL, seed=0
    )
    router_collection.layer_routers_L_dict = {
        layer: LayerActivationRouter(
            pca_mean_N=torch.zeros(D_MODEL),
            pca_components_ND=torch.eye(D_MODEL),
            centroids_ED=torch.stack([torch.full((D_MODEL,), 1e6), torch.zeros(D_MODEL)]),
        )
        for layer in SOURCE_LAYERS
    }

    trainer = ExpertJacobianTrainer(
        make_tiny_lens_config(tmp_path, "empty-cluster"),
        model,
        prompts=list(PROMPTS[:2]),
        router_collections_K_dict={2: router_collection},
    )
    fit_sums = trainer.fit()[2]
    assert fit_sums is trainer.fit_sums_K_dict[2]
    with caplog.at_level(logging.WARNING):
        experts = ExpertJacobians.from_sums(fit_sums, trainer.config, min_kept_positions=1)

    assert "pooled Jacobian" in caplog.text
    baseline_lens = build_matched_baseline_lens(
        fit_sums, trainer.config, "empty-cluster-matched"
    )
    for layer in SOURCE_LAYERS:
        counts_E = experts.position_counts_L_dict_E[layer]
        assert counts_E[0] == 0 and counts_E[1] > 0
        torch.testing.assert_close(
            experts.experts_L_dict_EFN[layer][0],
            baseline_lens.jacobians_L_dict_FN[layer],
            atol=1e-6,
            rtol=1e-5,
        )
        # The populated expert equals the baseline too (it holds all the data).
        torch.testing.assert_close(
            experts.experts_L_dict_EFN[layer][1],
            baseline_lens.jacobians_L_dict_FN[layer],
            atol=1e-6,
            rtol=1e-5,
        )
        experts_EFN, pooled_jacobian_FN, fallback_Bool_E = build_layer_expert_jacobians(
            fit_sums.layer_sums_L_dict[layer], min_kept_positions=1
        )
        assert fallback_Bool_E.tolist() == [True, False]
        assert torch.equal(pooled_jacobian_FN, baseline_lens.jacobians_L_dict_FN[layer])
        assert torch.equal(experts.experts_L_dict_EFN[layer], experts_EFN)
        assert experts.fallback_L_dict_Bool_E[layer].tolist() == [True, False]


def test_invalid_construction_rejected(
    model: TinyDecoder,
    fitted_router_collections_K_dict: dict[int, ActivationRouterCollection],
    tmp_path: Path,
) -> None:
    def make_trainer(name: str, **overrides: Any) -> ExpertJacobianTrainer:
        kwargs: dict[str, Any] = dict(
            router_collections_K_dict={2: fitted_router_collections_K_dict[2]},
        )
        kwargs.update(overrides)
        return ExpertJacobianTrainer(
            make_tiny_lens_config(tmp_path, name), model, prompts=[], **kwargs
        )

    # The router_collections_K_dict key must match the collection's own K.
    with pytest.raises(ValueError, match="num_clusters"):
        make_trainer(
            "bad-k", router_collections_K_dict={5: fitted_router_collections_K_dict[2]}
        )

    # A router collection that lacks a router for a source layer is rejected.
    partial_router_collection = ActivationRouterCollection(
        num_clusters=2, projection_dim=4, seed=0
    )
    partial_router_collection.layer_routers_L_dict = {
        layer: fitted_router_collections_K_dict[2].layer_routers_L_dict[layer]
        for layer in [0, 1]
    }
    with pytest.raises(ValueError, match="no router for source layers"):
        make_trainer("bad-layers", router_collections_K_dict={2: partial_router_collection})


def test_state_dict_round_trip(
    model: TinyDecoder,
    fitted_router_collections_K_dict: dict[int, ActivationRouterCollection],
    tmp_path: Path,
) -> None:
    """Checkpoint state dicts round-trip bit-exactly, directly and through a
    merge of the written shard file."""
    trainer = ExpertJacobianTrainer(
        make_tiny_lens_config(tmp_path, "round-trip"),
        model,
        prompts=list(PROMPTS[:2]),
        router_collections_K_dict={2: fitted_router_collections_K_dict[2]},
    )
    trainer.fit()
    fit_sums = trainer.fit_sums_K_dict[2]

    def assert_sums_equal(actual: ExpertFitSums, expected: ExpertFitSums) -> None:
        assert sorted(actual.layer_sums_L_dict) == sorted(expected.layer_sums_L_dict)
        for layer, expected_layer_sums in expected.layer_sums_L_dict.items():
            actual_layer_sums = actual.layer_sums_L_dict[layer]
            torch.testing.assert_close(
                actual_layer_sums.weighted_jacobian_row_sum_EFN,
                expected_layer_sums.weighted_jacobian_row_sum_EFN,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                actual_layer_sums.weight_sum_E,
                expected_layer_sums.weight_sum_E,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                actual_layer_sums.position_count_E,
                expected_layer_sums.position_count_E,
                rtol=0,
                atol=0,
            )

    assert_sums_equal(ExpertFitSums.from_state_dict(fit_sums.to_state_dict()), fit_sums)

    checkpoint_path = Path(trainer.config.checkpoint_path) / expert_checkpoint_filename(2)
    assert_sums_equal(merge_expert_checkpoints([str(checkpoint_path)]).fit_sums, fit_sums)


def test_checkpoint_stamps_each_k_geometry_and_lrp_rules(
    model: TinyDecoder,
    fitted_router_collections_K_dict: dict[int, ActivationRouterCollection],
    tmp_path: Path,
) -> None:
    """Every per-K file carries the config with that K's router geometry
    (``LensConfig.with_router``; still a ``"jacobian"`` lens type) and the resolved
    rule flags of the config's ``lrp_mode`` (all off here) — the same stamps
    whichever trainer or entry point wrote the shard — while the trainer's own
    config stays K-agnostic, on resume too."""
    trainer = ExpertJacobianTrainer(
        make_tiny_lens_config(tmp_path, "stamps"),
        model,
        prompts=list(PROMPTS[:1]),
        router_collections_K_dict=fitted_router_collections_K_dict,
    )
    trainer.fit()
    assert trainer.config.lens_type == "jacobian"
    assert trainer.config.num_clusters == 0
    for num_clusters, router_collection in fitted_router_collections_K_dict.items():
        state = torch.load(
            Path(trainer.config.checkpoint_path) / expert_checkpoint_filename(num_clusters),
            weights_only=True,
        )
        assert state["config"]["lens_type"] == "jacobian"
        assert state["config"]["num_clusters"] == num_clusters
        assert state["config"]["cluster_projection_dim"] == router_collection.projection_dim
        assert state["lrp_rules"] == lrp_rule_config_for_mode("none").to_dict()

    resumed = ExpertJacobianTrainer.from_checkpoint_dir(
        trainer.config.checkpoint_path,
        list(PROMPTS[:1]),
        router_collections_K_dict=fitted_router_collections_K_dict,
        model=model,
    )
    assert resumed.config.lens_type == "jacobian"
    assert resumed.config.num_clusters == 0
    assert resumed.config.checkpoint_path == trainer.config.checkpoint_path


def test_router_stamp_is_written_and_checked_by_merge_and_resume(
    model: TinyDecoder,
    fitted_router_collections_K_dict: dict[int, ActivationRouterCollection],
    tmp_path: Path,
) -> None:
    """Every per-K checkpoint carries the per-layer collection's router stamp and
    resumes with its position records; a shard stamped with another router kind
    refuses to merge (even alone or with shards of the same stamp) or resume,
    since its experts are bucketed differently."""
    routers = {2: fitted_router_collections_K_dict[2]}
    trainer = ExpertJacobianTrainer(
        make_tiny_lens_config(tmp_path, "router-stamp"),
        model,
        prompts=list(PROMPTS[:1]),
        router_collections_K_dict=routers,
    )
    trainer.fit()
    checkpoint_path = Path(trainer.config.checkpoint_path) / expert_checkpoint_filename(2)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert state["router"] == {"router_kind": "activation_collection"}

    resumed = ExpertJacobianTrainer.from_checkpoint_dir(
        str(checkpoint_path.parent),
        list(PROMPTS[:1]),
        router_collections_K_dict=routers,
        model=model,
    )
    assert resumed.position_records == trainer.position_records

    other_router_path = tmp_path / "other_router" / expert_checkpoint_filename(2)
    other_router_path.parent.mkdir()
    other_router_state = {**state, "router": {"router_kind": "other_router"}}
    torch.save(other_router_state, other_router_path)
    with pytest.raises(ValueError, match="bucketed differently"):
        merge_expert_checkpoints([str(checkpoint_path), str(other_router_path)])
    with pytest.raises(ValueError, match="bucketed differently"):
        merge_expert_checkpoints([str(other_router_path), str(other_router_path)])
    with pytest.raises(ValueError, match="resume passes"):
        ExpertJacobianTrainer.from_checkpoint_dir(
            str(other_router_path.parent),
            list(PROMPTS[:1]),
            router_collections_K_dict=routers,
            model=model,
        )


### EXPERTS FROM SUMS


def test_threshold_one_experts_are_per_expert_means_on_a_real_fit(
    model: TinyDecoder,
    fitted_router_collections_K_dict: dict[int, ActivationRouterCollection],
    tmp_path: Path,
) -> None:
    """On a real fit, ``ExpertJacobians.from_sums(min_kept_positions=1)`` makes every
    populated expert its own mean ``row_sum_e / count_e`` (computed here from the
    sums), equals ``build_layer_expert_jacobians`` at threshold 1, pools to the
    matched baseline, and flags exactly the experts with ``count == 0``."""
    trainer = ExpertJacobianTrainer(
        make_tiny_lens_config(tmp_path, "per-expert-means"),
        model,
        prompts=list(PROMPTS),
        router_collections_K_dict=fitted_router_collections_K_dict,
    )
    experts_K_dict = fit_expert_jacobians(trainer)
    for num_clusters in fitted_router_collections_K_dict:
        fit_sums = trainer.fit_sums_K_dict[num_clusters]
        experts = experts_K_dict[num_clusters]
        baseline_lens = build_matched_baseline_lens(fit_sums, trainer.config, "matched")
        for layer in SOURCE_LAYERS:
            layer_sums = fit_sums.layer_sums_L_dict[layer]
            counts_E = layer_sums.position_count_E
            populated_Bool_E = counts_E > 0
            expected_means_EFN = (
                layer_sums.weighted_jacobian_row_sum_EFN / counts_E.float()[:, None, None]
            )
            assert torch.equal(
                experts.experts_L_dict_EFN[layer][populated_Bool_E],
                expected_means_EFN[populated_Bool_E],
            )
            experts_EFN, pooled_jacobian_FN, fallback_Bool_E = (
                build_layer_expert_jacobians(layer_sums, min_kept_positions=1)
            )
            assert torch.equal(experts.experts_L_dict_EFN[layer], experts_EFN)
            assert torch.equal(
                pooled_jacobian_FN, baseline_lens.jacobians_L_dict_FN[layer]
            )
            assert torch.equal(fallback_Bool_E, counts_E == 0)
            assert torch.equal(experts.fallback_L_dict_Bool_E[layer], counts_E == 0)


def synthetic_layer_sums(generator: torch.Generator) -> LayerClusterSums:
    """Three experts: one with 3 kept positions, one with 60, one empty."""
    row_sums_EFN = torch.randn(3, 4, 4, generator=generator)
    row_sums_EFN[2] = 0.0
    return LayerClusterSums(
        weighted_jacobian_row_sum_EFN=row_sums_EFN,
        weight_sum_E=torch.tensor([2.5, 55.0, 0.0]),
        position_count_E=torch.tensor([3, 60, 0]),
    )


def test_build_experts_applies_min_kept_positions_fallback() -> None:
    """Experts below the kept-position threshold (or with zero weight sum)
    become the pooled Jacobian ``sum_e row_sum_e / sum_e weight_sum_e`` and
    are flagged; the threshold is inclusive (a count equal to it is kept);
    a layer with no positions at all raises."""
    layer_sums = synthetic_layer_sums(torch.Generator().manual_seed(0))
    row_sums_EFN = layer_sums.weighted_jacobian_row_sum_EFN
    expected_pooled_jacobian_FN = row_sums_EFN.sum(dim=0) / 57.5

    experts_EFN, pooled_jacobian_FN, fallback_Bool_E = build_layer_expert_jacobians(
        layer_sums, min_kept_positions=50
    )
    assert fallback_Bool_E.tolist() == [True, False, True]
    torch.testing.assert_close(pooled_jacobian_FN, expected_pooled_jacobian_FN)
    assert torch.equal(experts_EFN[0], pooled_jacobian_FN)
    assert torch.equal(experts_EFN[2], pooled_jacobian_FN)
    torch.testing.assert_close(experts_EFN[1], row_sums_EFN[1] / 55.0)

    _, _, fallback_at_1_Bool_E = build_layer_expert_jacobians(
        layer_sums, min_kept_positions=1
    )
    assert fallback_at_1_Bool_E.tolist() == [False, False, True]
    _, _, fallback_at_3_Bool_E = build_layer_expert_jacobians(
        layer_sums, min_kept_positions=3
    )
    assert fallback_at_3_Bool_E.tolist() == [False, False, True]
    _, _, fallback_at_4_Bool_E = build_layer_expert_jacobians(
        layer_sums, min_kept_positions=4
    )
    assert fallback_at_4_Bool_E.tolist() == [True, False, True]

    empty_layer_sums = LayerClusterSums(
        weighted_jacobian_row_sum_EFN=torch.zeros(3, 4, 4),
        weight_sum_E=torch.zeros(3),
        position_count_E=torch.zeros(3, dtype=torch.long),
    )
    with pytest.raises(ValueError, match="no positions"):
        build_layer_expert_jacobians(empty_layer_sums, min_kept_positions=1)


def test_expert_jacobians_from_sums_and_round_trip(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """``ExpertJacobians.from_sums`` applies the fallback at its threshold on every
    layer, records counts / weight sums / flags and the config, warns about
    the replaced experts; ``ExpertJacobians`` saves fp16 Jacobians and plain
    tensors (``weights_only=True``-loadable) and loads them back as stored
    (fp16, cast per layer by the consumers)."""
    generator = torch.Generator().manual_seed(1)
    fit_sums = ExpertFitSums(
        layer_sums_L_dict={
            0: synthetic_layer_sums(generator),
            2: synthetic_layer_sums(generator),
        }
    )
    config = make_tiny_lens_config(tmp_path, "synthetic-experts", d_model=4, num_prompts_trained_on=7)

    with caplog.at_level(logging.WARNING):
        experts = ExpertJacobians.from_sums(fit_sums, config)  # default min_kept=50
    assert "pooled Jacobian" in caplog.text
    assert experts.min_kept_positions == 50
    assert experts.layers == [0, 2]
    assert experts.num_experts == 3
    assert experts.config is config
    for layer, layer_sums in fit_sums.layer_sums_L_dict.items():
        expected_experts_EFN, expected_pooled_jacobian_FN, expected_fallback_Bool_E = (
            build_layer_expert_jacobians(layer_sums, min_kept_positions=50)
        )
        assert torch.equal(experts.experts_L_dict_EFN[layer], expected_experts_EFN)
        assert torch.equal(
            experts.pooled_jacobian_L_dict_FN[layer], expected_pooled_jacobian_FN
        )
        assert torch.equal(experts.fallback_L_dict_Bool_E[layer], expected_fallback_Bool_E)
        assert torch.equal(
            experts.position_counts_L_dict_E[layer], layer_sums.position_count_E
        )
        assert torch.equal(experts.weight_sums_L_dict_E[layer], layer_sums.weight_sum_E)

    path = str(tmp_path / "expert_jacobians.pt")
    experts.save(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert state["experts"][0].dtype == torch.float16
    assert state["pooled"][0].dtype == torch.float16
    assert state["min_kept_positions"] == 50

    loaded = ExpertJacobians.load(path)
    assert loaded.min_kept_positions == 50
    assert loaded.layers == [0, 2]
    assert loaded.config.to_dict() == config.to_dict()
    for layer in experts.layers:
        assert loaded.experts_L_dict_EFN[layer].dtype == torch.float16
        assert loaded.pooled_jacobian_L_dict_FN[layer].dtype == torch.float16
        assert torch.equal(
            loaded.experts_L_dict_EFN[layer], experts.experts_L_dict_EFN[layer].half()
        )
        assert torch.equal(
            loaded.pooled_jacobian_L_dict_FN[layer],
            experts.pooled_jacobian_L_dict_FN[layer].half(),
        )
        assert torch.equal(
            loaded.position_counts_L_dict_E[layer], experts.position_counts_L_dict_E[layer]
        )
        assert torch.equal(
            loaded.weight_sums_L_dict_E[layer], experts.weight_sums_L_dict_E[layer]
        )
        assert torch.equal(
            loaded.fallback_L_dict_Bool_E[layer], experts.fallback_L_dict_Bool_E[layer]
        )
