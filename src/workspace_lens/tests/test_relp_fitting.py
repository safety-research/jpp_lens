"""Tests for ExpertJacobianRelPTrainer (workspace_lens.fitting.relp_fitting).

The trainer's contract: identical to ExpertJacobianTrainer in every respect
except that the retained graph is built under the LRP surgery. Pinned here:
- ``lrp_mode="none"`` reproduces ExpertJacobianTrainer bit-exactly (same row
  sums, weight sums, counts);
- ``lrp_mode="rlens"`` changes the accumulated gradient rows but not the
  routing (position counts identical — the forward is value-identical);
- checkpoints embed the resolved rule flags and the mode survives a resume;
- merging shards across modes raises.
"""

from __future__ import annotations

import dataclasses
import os

import pytest
import torch as t

from workspace_lens.config import LensConfig
from workspace_lens.fitting.expert_fitting import ExpertJacobianTrainer
from workspace_lens.fitting.jacobian_fitting import LensTrainer
from workspace_lens.fitting.relp_fitting import ExpertJacobianRelPTrainer
from workspace_lens.fitting.utils import (
    expert_checkpoint_filename,
    merge_expert_checkpoints,
)
from workspace_lens.tests.fixtures import fit_random_router_collection
from workspace_lens.tests.tiny_qwen import wrap_tiny_qwen3_5

SOURCE_LAYERS = [1, 3]
NUM_CLUSTERS = 2
PROMPTS = [
    "the quick brown fox jumps over the lazy dog",
    "pack my box with five dozen liquor jugs today",
    "how vexingly quick daft zebras jump over fences",
]


@pytest.fixture(scope="module")
def tiny_lens_model():
    return wrap_tiny_qwen3_5(seed=0)


@pytest.fixture(scope="module")
def router_collections_K_dict(tiny_lens_model):
    return {
        NUM_CLUSTERS: fit_random_router_collection(
            tiny_lens_model.d_model, source_layers=SOURCE_LAYERS, num_clusters=NUM_CLUSTERS
        )
    }


def _make_config(tmp_path, lrp_mode: str, checkpoint_name: str) -> LensConfig:
    return LensConfig(
        hf_model_name="tiny-qwen3_5",
        checkpoint_name=checkpoint_name,
        artifacts_base_dir=str(tmp_path),
        checkpoint_every_n_prompts=None,
        source_layers=SOURCE_LAYERS,
        relative_end_transport_layer=-2,
        jacobian_rows_per_pass=8,
        max_seq_len=32,
        skip_first_n_positions=2,
        lrp_mode=lrp_mode,
    )


def _fit_sums(trainer) -> dict:
    trainer.fit()
    return trainer.fit_sums_K_dict[NUM_CLUSTERS].layer_sums_L_dict


def _make_trainer(
    trainer_class, tmp_path, lrp_mode, checkpoint_name, tiny_lens_model, routers, **overrides
):
    kwargs = dict(
        config=_make_config(tmp_path, lrp_mode, checkpoint_name),
        model=tiny_lens_model,
        prompts=list(PROMPTS),
        router_collections_K_dict=routers,
        device="cpu",
    )
    kwargs.update(overrides)
    return trainer_class(**kwargs)


def test_mode_none_reproduces_expert_trainer_bit_exact(
    tmp_path, tiny_lens_model, router_collections_K_dict
):
    standard_sums = _fit_sums(
        _make_trainer(
            ExpertJacobianTrainer, tmp_path, "none", "std",
            tiny_lens_model, router_collections_K_dict,
        )
    )
    relp_none_sums = _fit_sums(
        _make_trainer(
            ExpertJacobianRelPTrainer, tmp_path, "none", "relp-none",
            tiny_lens_model, router_collections_K_dict,
        )
    )
    for layer in SOURCE_LAYERS:
        assert t.equal(
            standard_sums[layer].weighted_jacobian_row_sum_EFN,
            relp_none_sums[layer].weighted_jacobian_row_sum_EFN,
        )
        assert t.equal(standard_sums[layer].weight_sum_E, relp_none_sums[layer].weight_sum_E)
        assert t.equal(
            standard_sums[layer].position_count_E, relp_none_sums[layer].position_count_E
        )


def test_rlens_mode_changes_rows_but_not_routing(
    tmp_path, tiny_lens_model, router_collections_K_dict
):
    standard_trainer = _make_trainer(
        ExpertJacobianTrainer, tmp_path, "none", "std2",
        tiny_lens_model, router_collections_K_dict,
    )
    relp_trainer = _make_trainer(
        ExpertJacobianRelPTrainer, tmp_path, "rlens", "relp",
        tiny_lens_model, router_collections_K_dict,
    )
    standard_sums = _fit_sums(standard_trainer)
    relp_sums = _fit_sums(relp_trainer)

    assert relp_trainer.last_patch_report is not None
    assert relp_trainer.last_patch_report.patched_any()
    for layer in SOURCE_LAYERS:
        # Value-identical forward => identical routing and counts.
        assert t.equal(
            standard_sums[layer].position_count_E, relp_sums[layer].position_count_E
        )
        # RelP coefficients != raw gradients => different accumulated rows.
        assert not t.allclose(
            standard_sums[layer].weighted_jacobian_row_sum_EFN,
            relp_sums[layer].weighted_jacobian_row_sum_EFN,
            rtol=1e-3,
            atol=1e-6,
        )


def test_checkpoint_embeds_rules_and_resume_restores_mode(
    tmp_path, tiny_lens_model, router_collections_K_dict
):
    trainer = _make_trainer(
        ExpertJacobianRelPTrainer, tmp_path, "rlens", "relp-ckpt",
        tiny_lens_model, router_collections_K_dict,
    )
    trainer.fit()
    checkpoint_path = os.path.join(
        trainer.config.checkpoint_path, expert_checkpoint_filename(NUM_CLUSTERS)
    )
    state = t.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert state["lrp_rules"] == trainer.lrp_rule_config.to_dict()
    assert state["config"]["lrp_mode"] == "rlens"

    resumed = ExpertJacobianRelPTrainer.from_checkpoint_dir(
        trainer.config.checkpoint_path,
        list(PROMPTS),
        router_collections_K_dict=router_collections_K_dict,
        model=tiny_lens_model,
        device="cpu",
    )
    assert resumed.config.lrp_mode == "rlens"
    assert resumed.lrp_rule_config == trainer.lrp_rule_config


def test_merging_across_modes_raises(
    tmp_path, tiny_lens_model, router_collections_K_dict
):
    checkpoint_paths = {}
    for lrp_mode, trainer_class in [
        ("none", ExpertJacobianTrainer),
        ("rlens", ExpertJacobianRelPTrainer),
    ]:
        trainer = _make_trainer(
            trainer_class, tmp_path, lrp_mode, f"merge-{lrp_mode}",
            tiny_lens_model, router_collections_K_dict,
        )
        trainer.fit()
        checkpoint_paths[lrp_mode] = os.path.join(
            trainer.config.checkpoint_path, expert_checkpoint_filename(NUM_CLUSTERS)
        )

    with pytest.raises(ValueError, match="LRP rules"):
        merge_expert_checkpoints([checkpoint_paths["none"], checkpoint_paths["rlens"]])


def test_merging_relp_shards_is_exact(
    tmp_path, tiny_lens_model, router_collections_K_dict
):
    # Two disjoint one-and-two prompt shards must merge to the full fit.
    shard_prompt_slices = [PROMPTS[:1], PROMPTS[1:]]
    shard_checkpoint_paths = []
    for shard_idx, shard_prompts in enumerate(shard_prompt_slices):
        config = _make_config(tmp_path, "rlens", f"relp-shard{shard_idx}")
        trainer = ExpertJacobianRelPTrainer(
            config=config,
            model=tiny_lens_model,
            prompts=shard_prompts,
            router_collections_K_dict=router_collections_K_dict,
            device="cpu",
        )
        trainer.fit()
        shard_checkpoint_paths.append(
            os.path.join(config.checkpoint_path, expert_checkpoint_filename(NUM_CLUSTERS))
        )

    full_trainer = _make_trainer(
        ExpertJacobianRelPTrainer, tmp_path, "rlens", "relp-full",
        tiny_lens_model, router_collections_K_dict,
    )
    full_sums = _fit_sums(full_trainer)

    merged = merge_expert_checkpoints(shard_checkpoint_paths)
    assert merged.config.lrp_mode == "rlens"
    for layer in SOURCE_LAYERS:
        merged_layer_sums = merged.fit_sums.layer_sums_L_dict[layer]
        assert t.allclose(
            merged_layer_sums.weighted_jacobian_row_sum_EFN,
            full_sums[layer].weighted_jacobian_row_sum_EFN,
            rtol=1e-6,
            atol=1e-8,
        )
        assert t.equal(
            merged_layer_sums.position_count_E, full_sums[layer].position_count_E
        )


def test_checkpoint_config_carries_lrp_mode(
    tmp_path, tiny_lens_model, router_collections_K_dict
):
    """The per-K checkpoint's config reads back with the fit's LRP mode and its K
    (a ``"jacobian"`` lens type), and a config that differs only in the mode is
    incompatible with it."""
    trainer = _make_trainer(
        ExpertJacobianRelPTrainer, tmp_path, "rlens", "relp-experts",
        tiny_lens_model, router_collections_K_dict,
    )
    trainer.fit()
    checkpoint_path = os.path.join(
        trainer.config.checkpoint_path, expert_checkpoint_filename(NUM_CLUSTERS)
    )
    state = t.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_config = LensConfig.from_dict(state["config"])
    assert checkpoint_config.lrp_mode == "rlens"
    assert checkpoint_config.lens_type == "jacobian"
    assert checkpoint_config.num_clusters == NUM_CLUSTERS

    incompatible_config = dataclasses.replace(checkpoint_config, lrp_mode="none")
    with pytest.raises(ValueError, match="lrp_mode"):
        checkpoint_config.check_compatible(incompatible_config)


def test_plain_trainer_rejects_relp_lrp_mode(tmp_path, tiny_lens_model) -> None:
    """config.lrp_mode is provenance: a trainer that does not apply the
    surgery must refuse a non-"none" mode (else the artifact would claim
    RelP over standard gradients)."""
    config = _make_config(tmp_path, lrp_mode="rlens", checkpoint_name="guard")
    with pytest.raises(ValueError, match="does not apply the LRP surgery"):
        LensTrainer(config, tiny_lens_model, ["ab"])
    config_none = _make_config(tmp_path, lrp_mode="none", checkpoint_name="guard2")
    LensTrainer(config_none, tiny_lens_model, ["ab"])
