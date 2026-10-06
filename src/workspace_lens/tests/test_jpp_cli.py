"""The J++ CLI stages on tiny CPU models (``scripts/jpp_cli.py``).

Pinned here:

- ``fit_router`` fits one router per requested layer with the requested K and
  projection dim, deterministically given the seed, and the saved file loads
  back identical.
- ``fit_shard`` writes each shard's per-K checkpoint to its own directory
  (``<checkpoint_name>/shard<i>of<n>``) with the router's K and projection dim
  in its config; and the merged shard sums equal a one-shot fit over all the
  prompts (the accumulators are plain sums over disjoint prompt sets).
- ``fit_shard`` with several routers of distinct K writes one checkpoint per
  K, each bit-identical to the single-router fit of that K (the same passes,
  another bucketing); two routers with the same K are rejected.
- ``fit_shard`` dispatches on ``config.lrp_mode``: the RelP trainer exactly
  when the mode is not ``"none"`` (checked via the trainer class it logs;
  both trainers stamp the same ``lrp_rules`` and router geometry, so a CLI
  shard merges with shards fitted through either trainer directly). Run on
  the tiny Qwen model because the LRP surgery needs real RMSNorm / gated-MLP
  modules, which TinyDecoder lacks.
- ``merge_experts`` at ``min_kept_positions=1`` makes every populated expert its
  own mean ``row_sum_e / count_e`` of the merged sums, and at a high threshold
  replaces every expert by the pooled Jacobian.
- ``fit_weights`` on the merged experts and fake labelled items (a character
  tokenizer makes single letters single-token candidates) yields the weighted
  Jacobian lens ``sum_e w_e J_e``; ``evaluate`` on it returns the rank rows
  and the pair table with the macro row; ``select_items``'s fitting and held-out
  splits partition the scoreable items.
- ``shard_prompt_slice`` covers ``range(num_prompts)`` exactly once, with the
  remainder on the first shards.
- Every subcommand parses (``--help`` exits 0) and runs end to end through
  ``main`` on the tiny models (loaders monkeypatched), writing its stamps; the
  default fit-weights and evaluate item splits are disjoint.
- ``write_stamp`` writes JSON carrying the provenance keys.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch as t

from jlens.tests.tiny import TinyDecoder
from lens_evals.readout_evals.readout_eval_items import (
    DEFAULT_FIT_FRACTION,
    MACRO_EVALS,
    ReadoutEvalItem,
    non_semantic_token_ids,
)
from workspace_lens.config import LensConfig
from workspace_lens.fitting.condense_experts import (
    CondenseConfig,
    ExpertJacobians,
    ExpertWeighting,
)
from workspace_lens.fitting.expert_fitting import (
    ExpertJacobianTrainer,
    build_matched_baseline_lens,
)
from workspace_lens.fitting.relp_fitting import ExpertJacobianRelPTrainer
from workspace_lens.fitting.utils import (
    MergedExpertCheckpoint,
    expert_checkpoint_filename,
    merge_expert_checkpoints,
)
from workspace_lens.lenses.base_lens import BaseLens
from workspace_lens.lenses.jacobian_lens import JacobianLens
from workspace_lens.lrp import lrp_rule_config_for_mode
from workspace_lens.routing.router import ActivationRouterCollection
from workspace_lens.tests.fixtures import (
    fit_random_router_collection,
    make_tiny_decoder,
    make_tiny_lens_config,
    write_readout_eval_json,
)
from workspace_lens.tests.tokenizers import CharTokenizer

# scripts/ is not a package (the CLI is run as `python scripts/jpp_cli.py`), so
# the module is imported by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import jpp_cli  # noqa: E402
from jpp_cli import (  # noqa: E402
    evaluate,
    fit_router,
    fit_shard,
    fit_weights,
    main,
    merge_experts,
    readout_runner,
    select_items,
    shard_prompt_slice,
    write_stamp,
)

from workspace_lens.tests.tiny_qwen import wrap_tiny_qwen3_5
from workspace_lens.utils import vocab_size_of

D_MODEL = 8
LAYERS = [0, 1, 2]
NUM_CLUSTERS = 2
PROJECTION_DIM = 4
SKIP_FIRST_N_POSITIONS = 16
MAX_SEQ_LEN = 64
READOUT_MAX_SEQ_LEN = 32
PROMPTS = [
    "abcdefghij " * 5,
    "klmnopqrst " * 5,
    "uvwxyzabcd " * 5,
    "efghijklmn " * 5,
]
STAGES = [
    "fit-router",
    "fit-shard",
    "merge-experts",
    "pooled-lens",
    "fit-weights",
    "evaluate",
    "probe-swap",
]

# Four labelled items per macro eval; every prompt holds a newline (poetry
# reads out at the last one) and its single-letter intermediates.
FAKE_PROMPTS_AND_INTERMEDIATES = [
    ("ab\ncd", ("d", "b")),
    ("ac\nde", ("e",)),
    ("bd\nca", ("a", "c")),
    ("cb\nad", ("d",)),
]
NUM_FAKE_PAIRS_PER_EVAL = 6
UNSCOREABLE_ITEM = ReadoutEvalItem("typo", "typo-unscoreable", "ab\ncd", None, ("xy",))


def fake_recipe_items() -> list[ReadoutEvalItem]:
    return [
        ReadoutEvalItem(
            eval_slug,
            f"{eval_slug}-{idx}",
            prompt,
            intermediates[0],
            intermediates,
        )
        for eval_slug in MACRO_EVALS
        for idx, (prompt, intermediates) in enumerate(FAKE_PROMPTS_AND_INTERMEDIATES)
    ]


def item_names(items: list[ReadoutEvalItem]) -> list[str]:
    return [item.name for item in items]


@pytest.fixture(scope="module")
def model() -> TinyDecoder:
    return TinyDecoder(n_layers=4, d_model=D_MODEL)


@pytest.fixture(scope="module")
def eval_model() -> TinyDecoder:
    """The fitting model's weights (same seed) with a character tokenizer, so
    single letters are single-token candidates for the labelled items."""
    return make_tiny_decoder(CharTokenizer())


@pytest.fixture(scope="module")
def router(model: TinyDecoder) -> ActivationRouterCollection:
    return fit_router(
        model,
        PROMPTS,
        layers=LAYERS,
        num_clusters=NUM_CLUSTERS,
        projection_dim=PROJECTION_DIM,
        skip_first_n_positions=SKIP_FIRST_N_POSITIONS,
        max_seq_len=MAX_SEQ_LEN,
    )


@pytest.fixture(scope="module")
def shard_dirs(
    model: TinyDecoder,
    router: ActivationRouterCollection,
    tmp_path_factory: pytest.TempPathFactory,
) -> list[str]:
    """PROMPTS as two 2-prompt shards through fit_shard."""
    artifacts_dir = tmp_path_factory.mktemp("shards")
    config = make_tiny_lens_config(artifacts_dir, "experts")
    return [
        fit_shard(
            model,
            PROMPTS,
            routers=[router],
            shard_idx=shard_idx,
            num_shards=2,
            config=config,
        )
        for shard_idx in range(2)
    ]


@pytest.fixture(scope="module")
def experts(shard_dirs: list[str]) -> ExpertJacobians:
    return merge_experts(
        shard_dirs,
        num_clusters=NUM_CLUSTERS,
        min_kept_positions=1,
        checkpoint_name="merged_experts",
    )


@pytest.fixture(scope="module")
def excluded_token_ids(eval_model: TinyDecoder) -> list[int]:
    return non_semantic_token_ids(
        eval_model.tokenizer,
        vocab_size=vocab_size_of(eval_model),
        items=fake_recipe_items(),
    )


def shard_checkpoint_path(shard_dir: str) -> str:
    return os.path.join(shard_dir, expert_checkpoint_filename(NUM_CLUSTERS))


def load_shard_state(shard_dir: str) -> dict:
    return t.load(shard_checkpoint_path(shard_dir), weights_only=True)


def merge_shard_sums(shard_dirs: list[str]) -> MergedExpertCheckpoint:
    """What ``merge_experts`` merges before dividing into experts."""
    return merge_expert_checkpoints([shard_checkpoint_path(d) for d in shard_dirs])


def write_fake_eval_data(data_dir: Path, *, hf_model_name: str) -> Path:
    """The six lens-eval-<slug>.json files (order-ops empty) for the fake
    items, and a correctness CSV marking every item correct. Returns the CSV."""
    items_by_eval: dict[str, list[dict]] = {slug: [] for slug in (*MACRO_EVALS, "order-ops")}
    for item in fake_recipe_items():
        items_by_eval[item.eval_slug].append(
            {
                "name": item.name,
                "prompt": item.prompt,
                "intermediates": list(item.intermediates),
                "target": item.target,
            }
        )
    for eval_slug, raw_items in items_by_eval.items():
        write_readout_eval_json(data_dir, eval_slug, raw_items)
    correctness_csv = data_dir / "model_correctness.csv"
    csv_rows = [f"eval,item,{hf_model_name}"] + [
        f"{item.eval_slug},{item.name},True" for item in fake_recipe_items()
    ]
    correctness_csv.write_text("\n".join(csv_rows) + "\n")
    return correctness_csv


### fit_router


def test_fit_router_fits_every_layer_and_round_trips(
    router: ActivationRouterCollection, tmp_path: Path
) -> None:
    assert router.num_clusters == NUM_CLUSTERS
    assert router.projection_dim == PROJECTION_DIM
    assert sorted(router.layer_routers_L_dict) == LAYERS
    for layer_router in router.layer_routers_L_dict.values():
        assert layer_router.pca_mean_N.shape == (D_MODEL,)
        assert layer_router.pca_components_ND.shape == (D_MODEL, PROJECTION_DIM)
        assert layer_router.centroids_ED.shape == (NUM_CLUSTERS, PROJECTION_DIM)

    router_path = tmp_path / "router.pt"
    router.save(str(router_path))
    loaded = ActivationRouterCollection.load(str(router_path))
    assert loaded.num_clusters == NUM_CLUSTERS
    assert loaded.projection_dim == PROJECTION_DIM
    for layer in LAYERS:
        assert t.equal(
            loaded.layer_routers_L_dict[layer].centroids_ED,
            router.layer_routers_L_dict[layer].centroids_ED,
        )
        assert t.equal(
            loaded.layer_routers_L_dict[layer].pca_components_ND,
            router.layer_routers_L_dict[layer].pca_components_ND,
        )


def test_fit_router_is_deterministic_given_seed(
    model: TinyDecoder, router: ActivationRouterCollection
) -> None:
    refit = fit_router(
        model,
        PROMPTS,
        layers=LAYERS,
        num_clusters=NUM_CLUSTERS,
        projection_dim=PROJECTION_DIM,
        skip_first_n_positions=SKIP_FIRST_N_POSITIONS,
        max_seq_len=MAX_SEQ_LEN,
    )
    for layer in LAYERS:
        assert t.equal(
            refit.layer_routers_L_dict[layer].centroids_ED,
            router.layer_routers_L_dict[layer].centroids_ED,
        )


### fit_shard


def test_fit_shard_writes_one_directory_per_shard(shard_dirs: list[str]) -> None:
    assert len(set(shard_dirs)) == 2
    for shard_idx, shard_dir in enumerate(shard_dirs):
        assert shard_dir.endswith(os.path.join("experts", f"shard{shard_idx}of2"))
        state = load_shard_state(shard_dir)
        assert state["config"]["num_prompts_trained_on"] == 2


def test_fit_shard_config_carries_the_router_geometry(
    shard_dirs: list[str], experts: ExpertJacobians
) -> None:
    """The shard config holds the router's K and projection dim
    (``LensConfig.with_router``; the lens type stays ``"jacobian"``), and so does
    the merged experts' config."""
    for shard_dir in shard_dirs:
        config_dict = load_shard_state(shard_dir)["config"]
        assert config_dict["lens_type"] == "jacobian"
        assert config_dict["num_clusters"] == NUM_CLUSTERS
        assert config_dict["cluster_projection_dim"] == PROJECTION_DIM
    assert experts.config.lens_type == "jacobian"
    assert experts.config.num_clusters == NUM_CLUSTERS
    assert experts.config.cluster_projection_dim == PROJECTION_DIM
    assert experts.config.d_model == D_MODEL


def test_fit_shard_rejects_out_of_range_shard(
    model: TinyDecoder, router: ActivationRouterCollection, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="shard_idx"):
        fit_shard(
            model,
            PROMPTS,
            routers=[router],
            shard_idx=2,
            num_shards=2,
            config=make_tiny_lens_config(tmp_path, "experts"),
        )


def test_fit_shard_with_two_routers_writes_one_checkpoint_per_k_bit_for_bit(
    model: TinyDecoder,
    router: ActivationRouterCollection,
    shard_dirs: list[str],
    tmp_path: Path,
) -> None:
    """Two routers with distinct K in one shard: the shard directory holds one
    ``experts_K<K>_checkpoint.pt`` per router, and each equals the single-router
    fit of that K exactly (the same forward and backward passes; only the
    bucketing over positions differs)."""
    router_k3 = fit_router(
        model,
        PROMPTS,
        layers=LAYERS,
        num_clusters=3,
        projection_dim=PROJECTION_DIM,
        skip_first_n_positions=SKIP_FIRST_N_POSITIONS,
        max_seq_len=MAX_SEQ_LEN,
    )
    two_router_dir = fit_shard(
        model,
        PROMPTS,
        routers=[router, router_k3],
        shard_idx=0,
        num_shards=2,
        config=make_tiny_lens_config(tmp_path, "experts-two-routers"),
    )
    single_k3_dir = fit_shard(
        model,
        PROMPTS,
        routers=[router_k3],
        shard_idx=0,
        num_shards=2,
        config=make_tiny_lens_config(tmp_path, "experts-k3"),
    )
    # One checkpoint per K plus the position-records sidecar.
    assert sorted(os.listdir(two_router_dir)) == sorted(
        [
            expert_checkpoint_filename(NUM_CLUSTERS),
            expert_checkpoint_filename(3),
            "position_records.json",
        ]
    )
    with open(os.path.join(two_router_dir, "position_records.json")) as handle:
        sidecar_records = json.load(handle)
    for num_clusters, single_router_dir in (
        (NUM_CLUSTERS, shard_dirs[0]),
        (3, single_k3_dir),
    ):
        two_router_state = t.load(
            os.path.join(two_router_dir, expert_checkpoint_filename(num_clusters)),
            weights_only=True,
        )
        single_router_state = t.load(
            os.path.join(single_router_dir, expert_checkpoint_filename(num_clusters)),
            weights_only=True,
        )
        assert two_router_state["config"]["num_clusters"] == num_clusters
        # Every per-K file carries its router stamp and the same position records as
        # the sidecar (the single-router fit of the same prompts wrote identical ones).
        assert two_router_state["router"] == single_router_state["router"]
        assert two_router_state["position_records"] == sidecar_records
        assert single_router_state["position_records"] == sidecar_records
        for layer in LAYERS:
            for field_name in (
                "weighted_jacobian_row_sum_EFN",
                "weight_sum_E",
                "position_count_E",
            ):
                assert t.equal(
                    two_router_state["layer_sums"][layer][field_name],
                    single_router_state["layer_sums"][layer][field_name],
                )


def test_fit_shard_rejects_routers_with_the_same_k(
    model: TinyDecoder, router: ActivationRouterCollection, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="distinct num_clusters"):
        fit_shard(
            model,
            PROMPTS,
            routers=[router, router],
            shard_idx=0,
            num_shards=1,
            config=make_tiny_lens_config(tmp_path, "experts"),
        )


@pytest.mark.parametrize(
    "lrp_mode, expected_trainer_name",
    [("none", "ExpertJacobianTrainer"), ("rlens", "ExpertJacobianRelPTrainer")],
)
def test_fit_shard_dispatches_trainer_on_lrp_mode(
    tmp_path: Path,
    lrp_mode: str,
    expected_trainer_name: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """fit_shard logs the trainer class it constructed; both trainers stamp the
    resolved rules of the config's ``lrp_mode`` and the router geometry, so
    the checkpoint does not depend on which one ran."""
    tiny_qwen = wrap_tiny_qwen3_5(seed=0)
    qwen_router = fit_random_router_collection(tiny_qwen.d_model, source_layers=[1, 3])
    config = LensConfig(
        hf_model_name="tiny-qwen3_5",
        checkpoint_name=f"experts-{lrp_mode}",
        artifacts_base_dir=str(tmp_path),
        checkpoint_every_n_prompts=None,
        source_layers=sorted(qwen_router.layer_routers_L_dict),
        relative_end_transport_layer=-2,
        jacobian_rows_per_pass=8,
        max_seq_len=32,
        skip_first_n_positions=2,
        lrp_mode=lrp_mode,
    )
    with caplog.at_level(logging.INFO, logger="jpp_cli"):
        checkpoint_dir = fit_shard(
            tiny_qwen,
            ["the quick brown fox jumps over the lazy dog", "pack my box with five jugs"],
            routers=[qwen_router],
            shard_idx=0,
            num_shards=1,
            config=config,
        )
    assert f"{expected_trainer_name} shard 0/1" in caplog.text
    state = load_shard_state(checkpoint_dir)
    assert state["lrp_rules"] == lrp_rule_config_for_mode(lrp_mode).to_dict()
    assert state["config"]["lrp_mode"] == lrp_mode
    assert state["config"]["lens_type"] == "jacobian"
    assert state["config"]["num_clusters"] == qwen_router.num_clusters
    assert state["config"]["num_prompts_trained_on"] == 2


def test_cli_shard_merges_with_library_shards(
    model: TinyDecoder,
    router: ActivationRouterCollection,
    tmp_path: Path,
    shard_dirs: list[str],
) -> None:
    """A fit_shard checkpoint and one written by either trainer on a plain
    config (no router fields set) carry the same stamps, so they merge — to
    the sums of two fit_shard shards bit for bit."""
    cli_merged = merge_shard_sums(shard_dirs)
    shard1_prompts = list(PROMPTS[shard_prompt_slice(len(PROMPTS), 2, 1)])
    for trainer_class, name in (
        (ExpertJacobianTrainer, "library-experts"),
        (ExpertJacobianRelPTrainer, "library-relp-mode-none"),
    ):
        trainer = trainer_class(
            make_tiny_lens_config(tmp_path, name),
            model,
            prompts=shard1_prompts,
            router_collections_K_dict={NUM_CLUSTERS: router},
        )
        trainer.fit()
        merged = merge_expert_checkpoints(
            [
                shard_checkpoint_path(shard_dirs[0]),
                shard_checkpoint_path(trainer.config.checkpoint_path),
            ]
        )
        assert merged.config.lens_type == "jacobian"
        assert merged.config.num_clusters == NUM_CLUSTERS
        for layer in LAYERS:
            merged_layer_sums = merged.fit_sums.layer_sums_L_dict[layer]
            cli_layer_sums = cli_merged.fit_sums.layer_sums_L_dict[layer]
            assert t.equal(
                merged_layer_sums.weighted_jacobian_row_sum_EFN,
                cli_layer_sums.weighted_jacobian_row_sum_EFN,
            )
            assert t.equal(merged_layer_sums.weight_sum_E, cli_layer_sums.weight_sum_E)
            assert t.equal(
                merged_layer_sums.position_count_E, cli_layer_sums.position_count_E
            )


### merge_experts


def test_merged_shard_sums_equal_one_shot_fit(
    model: TinyDecoder,
    router: ActivationRouterCollection,
    tmp_path: Path,
    shard_dirs: list[str],
) -> None:
    """Two 2-prompt shards merged == one 4-prompt ExpertJacobianTrainer fit
    (row sums, weight sums, counts)."""
    merged = merge_shard_sums(shard_dirs)
    assert merged.num_clusters == NUM_CLUSTERS
    assert merged.config.num_prompts_trained_on == len(PROMPTS)

    full_trainer = ExpertJacobianTrainer(
        make_tiny_lens_config(tmp_path, "full"),
        model,
        prompts=list(PROMPTS),
        router_collections_K_dict={NUM_CLUSTERS: router},
    )
    full_trainer.fit()
    for layer in LAYERS:
        merged_layer_sums = merged.fit_sums.layer_sums_L_dict[layer]
        full_layer_sums = full_trainer.fit_sums_K_dict[NUM_CLUSTERS].layer_sums_L_dict[layer]
        t.testing.assert_close(
            merged_layer_sums.weighted_jacobian_row_sum_EFN,
            full_layer_sums.weighted_jacobian_row_sum_EFN,
            atol=1e-6,
            rtol=1e-5,
        )
        t.testing.assert_close(
            merged_layer_sums.weight_sum_E,
            full_layer_sums.weight_sum_E,
            atol=1e-6,
            rtol=1e-5,
        )
        assert t.equal(merged_layer_sums.position_count_E, full_layer_sums.position_count_E)
        # Every position enters with weight 1.
        assert t.equal(
            merged_layer_sums.weight_sum_E, merged_layer_sums.position_count_E.float()
        )


def test_merge_experts_at_min_kept_one_gives_per_expert_means(
    shard_dirs: list[str],
    experts: ExpertJacobians,
) -> None:
    """Every populated expert is its own mean ``row_sum_e / count_e`` of the merged
    shard sums, bit for bit what ``ExpertJacobians.from_sums`` gives on those sums;
    only an expert with no position falls back."""
    assert isinstance(experts, ExpertJacobians)
    assert experts.num_experts == NUM_CLUSTERS
    assert experts.layers == LAYERS
    assert experts.min_kept_positions == 1
    assert experts.config.checkpoint_name == "merged_experts"
    assert experts.config.num_prompts_trained_on == len(PROMPTS)

    merged = merge_shard_sums(shard_dirs)
    expected = ExpertJacobians.from_sums(merged.fit_sums, merged.config, min_kept_positions=1)
    for layer in LAYERS:
        layer_sums = merged.fit_sums.layer_sums_L_dict[layer]
        counts_E = layer_sums.position_count_E
        populated_Bool_E = counts_E > 0
        expected_means_EFN = (
            layer_sums.weighted_jacobian_row_sum_EFN / counts_E.float()[:, None, None]
        )
        assert t.equal(
            experts.experts_L_dict_EFN[layer][populated_Bool_E],
            expected_means_EFN[populated_Bool_E],
        )
        assert t.equal(experts.experts_L_dict_EFN[layer], expected.experts_L_dict_EFN[layer])
        assert t.equal(experts.fallback_L_dict_Bool_E[layer], counts_E == 0)
        assert t.equal(experts.position_counts_L_dict_E[layer], counts_E)


def test_merge_experts_applies_pooled_fallback_at_high_threshold(
    shard_dirs: list[str],
) -> None:
    """A threshold above every expert's count makes every expert the pooled
    Jacobian, i.e. the matched baseline over the same sums."""
    all_fallback = merge_experts(
        shard_dirs,
        num_clusters=NUM_CLUSTERS,
        min_kept_positions=10**6,
        checkpoint_name="all-fallback",
    )
    merged = merge_shard_sums(shard_dirs)
    baseline = build_matched_baseline_lens(merged.fit_sums, merged.config, "matched")
    for layer in LAYERS:
        assert all_fallback.fallback_L_dict_Bool_E[layer].all()
        for expert_idx in range(NUM_CLUSTERS):
            assert t.equal(
                all_fallback.experts_L_dict_EFN[layer][expert_idx],
                all_fallback.pooled_jacobian_L_dict_FN[layer],
            )
        assert t.equal(
            all_fallback.pooled_jacobian_L_dict_FN[layer],
            baseline.jacobians_L_dict_FN[layer],
        )


def test_experts_file_round_trips_through_the_library_loader(
    experts: ExpertJacobians, tmp_path: Path
) -> None:
    """The CLI reads ExpertJacobians files only: a saved file loads back, and a
    file in another format raises."""
    library_path = tmp_path / "experts.pt"
    experts.save(str(library_path))
    loaded = ExpertJacobians.load(str(library_path))
    assert loaded.layers == LAYERS
    assert loaded.config.checkpoint_name == "merged_experts"  # the file's own
    for layer in LAYERS:
        assert t.equal(
            loaded.experts_L_dict_EFN[layer], experts.experts_L_dict_EFN[layer].half()
        )

    unknown_path = tmp_path / "unknown.pt"
    t.save({"experts": {}}, unknown_path)
    with pytest.raises(KeyError):
        ExpertJacobians.load(str(unknown_path))


### fit_weights / evaluate


def test_fit_weights_end_to_end_on_tiny_model(
    eval_model: TinyDecoder,
    experts: ExpertJacobians,
    excluded_token_ids: list[int],
    tmp_path: Path,
) -> None:
    """Residuals recorded (and cached) -> one unit weight vector per layer ->
    the lens is sum_e w_e J_e with the router fields reset; a fit on the
    fitting split, by item names, runs off the same cache."""
    items = fake_recipe_items() + [UNSCOREABLE_ITEM]
    cache_path = tmp_path / "readout_residuals.pt"
    weighting, lens = fit_weights(
        eval_model,
        experts,
        items,
        layers=LAYERS,
        excluded_token_ids=excluded_token_ids,
        cache_path=str(cache_path),
        checkpoint_name="jpp",
        hf_model_name="tiny",
        max_seq_len=READOUT_MAX_SEQ_LEN,
        condense_config=CondenseConfig(pass_k=3, top_n=NUM_CLUSTERS, steps=5),
    )
    assert cache_path.exists()
    assert list(weighting.learned_weights_L_dict) == LAYERS
    for learned in weighting.learned_weights_L_dict.values():
        assert learned.weights_E.shape == (NUM_CLUSTERS,)
        assert learned.weights_E.norm().item() == pytest.approx(1.0, abs=1e-5)
        assert 0.0 <= learned.train_macro_pass <= 1.0

    assert isinstance(lens, JacobianLens)
    assert lens.source_layers == LAYERS
    assert lens.config.checkpoint_name == "jpp"
    assert lens.config.lens_type == "jacobian"
    assert lens.config.num_clusters == 0
    assert lens.config.cluster_projection_dim == 0
    for layer in LAYERS:
        expected_FN = t.einsum(
            "e,enm->nm",
            weighting.learned_weights_L_dict[layer].weights_E,
            experts.experts_L_dict_EFN[layer],
        )
        t.testing.assert_close(lens.jacobians_L_dict_FN[layer], expected_FN)

    # The fitting split, selected by name over the scoreable items, off the cache.
    scoreable = readout_runner(
        eval_model, layers=LAYERS, max_seq_len=READOUT_MAX_SEQ_LEN
    ).scoreable_items(items)
    train_names = item_names(select_items(scoreable, "fit:0", fit_fraction=0.5))
    train_weighting, train_lens = fit_weights(
        eval_model,
        experts,
        items,
        layers=LAYERS,
        excluded_token_ids=excluded_token_ids,
        cache_path=str(cache_path),
        checkpoint_name="jpp-train",
        hf_model_name="tiny",
        max_seq_len=READOUT_MAX_SEQ_LEN,
        fitting_item_names=train_names,
        condense_config=CondenseConfig(pass_k=3, top_n=NUM_CLUSTERS, steps=5),
    )
    assert list(train_weighting.learned_weights_L_dict) == LAYERS
    assert train_lens.config.checkpoint_name == "jpp-train"


def test_fit_weights_rejects_layers_outside_the_experts_before_recording(
    eval_model: TinyDecoder,
    experts: ExpertJacobians,
    excluded_token_ids: list[int],
    tmp_path: Path,
) -> None:
    """A layer the experts do not cover is refused up front — before the
    residual pass runs or a cache keyed to the wrong layers is written."""
    cache_path = tmp_path / "never_written.pt"
    with pytest.raises(ValueError, match="not among the experts' layers"):
        fit_weights(
            eval_model,
            experts,
            fake_recipe_items(),
            layers=[*LAYERS, 99],
            excluded_token_ids=excluded_token_ids,
            cache_path=str(cache_path),
            checkpoint_name="jpp",
            hf_model_name="tiny",
            max_seq_len=READOUT_MAX_SEQ_LEN,
        )
    assert not cache_path.exists()


def test_evaluate_rejects_a_lens_fitted_at_another_width(
    eval_model: TinyDecoder, experts: ExpertJacobians, excluded_token_ids: list[int]
) -> None:
    """A lens whose config says another ``d_model`` was not fitted on this
    model (check_model_matches_config; same-width mixes of another HF identity
    are caught by name, which test models lack)."""
    lens = experts.combine(
        {layer: t.ones(NUM_CLUSTERS) for layer in LAYERS}, checkpoint_name="jpp"
    )
    wider_lens = JacobianLens(
        jacobians=lens.jacobians_L_dict_FN,
        config=dataclasses.replace(lens.config, d_model=D_MODEL + 1),
    )
    with pytest.raises(ValueError, match="d_model"):
        evaluate(
            eval_model,
            {"jpp": wider_lens},
            fake_recipe_items(),
            layers=LAYERS,
            excluded_token_ids=excluded_token_ids,
            max_seq_len=READOUT_MAX_SEQ_LEN,
        )


def test_evaluate_returns_ranks_and_recall_table_with_macro_rows(
    eval_model: TinyDecoder, experts: ExpertJacobians, excluded_token_ids: list[int]
) -> None:
    items = fake_recipe_items()
    _, lens = fit_weights(
        eval_model,
        experts,
        items,
        layers=LAYERS,
        excluded_token_ids=excluded_token_ids,
        cache_path=None,
        checkpoint_name="jpp",
        hf_model_name="tiny",
        max_seq_len=READOUT_MAX_SEQ_LEN,
        condense_config=CondenseConfig(pass_k=3, top_n=NUM_CLUSTERS, steps=2),
    )
    ranks_df, pass_df = evaluate(
        eval_model,
        {"jpp": lens},
        items,
        layers=LAYERS,
        excluded_token_ids=excluded_token_ids,
        ks=(1, 10),
        max_seq_len=READOUT_MAX_SEQ_LEN,
    )
    assert list(ranks_df.columns) == [
        "eval",
        "lens",
        "item",
        "intermediate",
        "layer",
        "rank",
        "n_candidates",
    ]
    assert set(ranks_df["lens"]) == {"jpp"}
    assert set(ranks_df["eval"]) == set(MACRO_EVALS)
    assert len(ranks_df) == len(MACRO_EVALS) * NUM_FAKE_PAIRS_PER_EVAL * len(LAYERS)
    assert ranks_df["rank"].min() >= 1

    assert set(pass_df["eval"]) == {*MACRO_EVALS, "macro"}
    assert set(pass_df["k"]) == {1, 10}
    assert set(pass_df["weighting"]) == {"item", "pair"}
    for weighting in ("item", "pair"):
        weighted_df = pass_df[pass_df["weighting"] == weighting]
        for k in (1, 10):
            rows_k = weighted_df[weighted_df["k"] == k].set_index("eval")["pass_at_k"]
            assert 0.0 <= rows_k["macro"] <= 1.0
            assert rows_k["macro"] == pytest.approx(rows_k[list(MACRO_EVALS)].mean())
        macro_by_k = weighted_df[weighted_df["eval"] == "macro"].set_index("k")["pass_at_k"]
        assert macro_by_k[1] <= macro_by_k[10]


def test_select_items_partitions_the_scoreable_items(eval_model: TinyDecoder) -> None:
    items = fake_recipe_items() + [UNSCOREABLE_ITEM]
    scoreable = readout_runner(
        eval_model, layers=LAYERS, max_seq_len=READOUT_MAX_SEQ_LEN
    ).scoreable_items(items)
    assert item_names(scoreable) == item_names(fake_recipe_items())
    assert select_items(scoreable, "all", fit_fraction=DEFAULT_FIT_FRACTION) == scoreable

    # Four items per eval: the default fraction keeps one fitting item per eval, a half
    # keeps two.
    for fit_fraction, num_fitting_per_eval in ((DEFAULT_FIT_FRACTION, 1), (0.5, 2)):
        fitting = select_items(scoreable, "fit:0", fit_fraction=fit_fraction)
        held_out = select_items(scoreable, "held-out:0", fit_fraction=fit_fraction)
        assert not set(item_names(fitting)) & set(item_names(held_out))
        assert sorted(item_names(fitting) + item_names(held_out)) == sorted(
            item_names(scoreable)
        )
        for eval_slug in MACRO_EVALS:
            assert sum(item.eval_slug == eval_slug for item in fitting) == num_fitting_per_eval

    for bad_spec in ("half", "valid:0", "fit", "fit:x", "train:0:0"):
        with pytest.raises(ValueError):
            select_items(scoreable, bad_spec, fit_fraction=DEFAULT_FIT_FRACTION)


### prompt helpers


@pytest.mark.parametrize(
    "num_prompts, num_shards", [(4, 2), (7, 3), (5, 8), (64, 16), (1000, 16)]
)
def test_shard_prompt_slice_partitions_prompts_contiguously(
    num_prompts: int, num_shards: int
) -> None:
    prompt_indices = list(range(num_prompts))
    shard_slices = [
        prompt_indices[shard_prompt_slice(num_prompts, num_shards, shard_idx)]
        for shard_idx in range(num_shards)
    ]
    assert [idx for shard in shard_slices for idx in shard] == prompt_indices
    shard_sizes = [len(shard) for shard in shard_slices]
    assert max(shard_sizes) - min(shard_sizes) <= 1
    # The remainder goes to the first shards.
    assert shard_sizes == sorted(shard_sizes, reverse=True)


def test_shard_prompt_slice_layout_for_1000_prompts_over_16_shards() -> None:
    """1000 prompts over 16 shards: 63 each for shards 0-7, 62 after."""
    assert shard_prompt_slice(1000, 16, 0) == slice(0, 63)
    assert shard_prompt_slice(1000, 16, 7) == slice(441, 504)
    assert shard_prompt_slice(1000, 16, 8) == slice(504, 566)
    assert shard_prompt_slice(1000, 16, 15) == slice(938, 1000)


### command line


@pytest.mark.parametrize("stage", STAGES)
def test_main_help_parses_for_every_stage(
    stage: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main([stage, "--help"])
    assert exit_info.value.code == 0
    assert stage in capsys.readouterr().out


def test_main_without_a_stage_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main([])
    assert exit_info.value.code == 2


def test_fit_shard_argv_parses_layers_and_defaults() -> None:
    from jpp_cli import build_parser

    args = build_parser().parse_args(
        ["fit-shard", "--router-path", "router.pt", "--layers", "8,16", "--num-shards", "2"]
    )
    assert args.stage == "fit-shard"
    assert args.router_path == ["router.pt"]
    assert args.layers == [8, 16]
    assert args.shard_idx == 0
    assert args.num_shards == 2
    # The defaults: RelP rows, the final block as the transport target, 64 prompts.
    assert args.lrp_mode == "rlens"
    assert args.target_offset == -1
    assert args.num_prompts == 64


def test_fit_weights_and_evaluate_argv_parse_defaults() -> None:
    from jpp_cli import (
        DEFAULT_PASS_KS,
        RECIPE_LAYERS,
        RECIPE_READOUT_MAX_SEQ_LEN,
        build_parser,
    )

    fit_args = build_parser().parse_args(
        ["fit-weights", "--experts", "experts.pt", "--out", "lens.pt"]
    )
    assert fit_args.stage == "fit-weights"
    assert fit_args.items == "fit:0"
    assert fit_args.fit_fraction == DEFAULT_FIT_FRACTION
    assert fit_args.layers == RECIPE_LAYERS
    assert fit_args.readout_max_seq_len == RECIPE_READOUT_MAX_SEQ_LEN
    assert fit_args.cache_path is None

    eval_args = build_parser().parse_args(["evaluate", "--lens", "a.pt", "--out", "ranks.csv"])
    assert eval_args.items == "held-out:0"
    assert eval_args.fit_fraction == DEFAULT_FIT_FRACTION
    eval_args = build_parser().parse_args(
        [
            "evaluate",
            "--lens",
            "a.pt",
            "b.pt",
            "--items",
            "held-out:1",
            "--fit-fraction",
            "0.5",
            "--out",
            "ranks.csv",
        ]
    )
    assert eval_args.stage == "evaluate"
    assert eval_args.lens == ["a.pt", "b.pt"]
    assert eval_args.items == "held-out:1"
    assert eval_args.fit_fraction == 0.5
    assert eval_args.ks == DEFAULT_PASS_KS


def test_main_fit_router_and_fit_shard_end_to_end_on_tiny_model(
    model: TinyDecoder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two fitting handlers, with the HF model loader and the WikiText
    loader swapped for the tiny fixtures: router file + stamp, then a shard
    checkpoint + stamp."""
    monkeypatch.setattr(jpp_cli, "get_hf_model", lambda name, **_kwargs: model)
    monkeypatch.setattr(jpp_cli, "load_fit_prompts", lambda num_prompts: PROMPTS)

    router_path = tmp_path / "router.pt"
    assert (
        main(
            [
                "fit-router",
                "--hf-model-name",
                "tiny",
                "--layers",
                "0,1,2",
                "--num-prompts",
                "4",
                "--num-clusters",
                str(NUM_CLUSTERS),
                "--projection-dim",
                str(PROJECTION_DIM),
                "--max-seq-len",
                str(MAX_SEQ_LEN),
                "--out",
                str(router_path),
            ]
        )
        == 0
    )
    assert ActivationRouterCollection.load(str(router_path)).num_clusters == NUM_CLUSTERS
    router_stamp = json.loads((tmp_path / "router_stamp.json").read_text())
    assert router_stamp["stage"] == "fit-router"
    assert router_stamp["args"]["layers"] == LAYERS

    assert (
        main(
            [
                "fit-shard",
                "--hf-model-name",
                "tiny",
                "--router-path",
                str(router_path),
                "--layers",
                "0,1,2",
                "--num-prompts",
                "4",
                "--shard-idx",
                "1",
                "--num-shards",
                "2",
                "--lrp-mode",
                "none",
                "--jacobian-rows-per-pass",
                "4",
                "--max-seq-len",
                str(MAX_SEQ_LEN),
                "--checkpoint-every-n-prompts",
                "1",
                "--artifacts-base-dir",
                str(tmp_path),
                "--checkpoint-name",
                "cli_experts",
            ]
        )
        == 0
    )
    shard_dirs = list(tmp_path.glob("*/cli_experts/shard1of2"))
    assert len(shard_dirs) == 1
    assert (shard_dirs[0] / expert_checkpoint_filename(NUM_CLUSTERS)).exists()
    shard_stamp = json.loads((shard_dirs[0] / "fit_shard_stamp.json").read_text())
    assert shard_stamp["stage"] == "fit-shard"
    assert shard_stamp["args"]["shard_idx"] == 1


def test_merge_experts_argv_defaults_device_to_the_gpu_when_available() -> None:
    args = jpp_cli.build_parser().parse_args(
        ["merge-experts", "--checkpoint-dirs", "shard0", "--out", "experts.pt"]
    )
    assert args.device == jpp_cli.DEFAULT_MERGE_DEVICE
    assert jpp_cli.DEFAULT_MERGE_DEVICE == ("cuda" if t.cuda.is_available() else "cpu")


def test_model_loading_stages_accept_a_device_map() -> None:
    """``--device-map`` (``auto`` spreads a large model over several GPUs) is optional on
    every model-loading stage and defaults to the single-GPU load."""
    parser = jpp_cli.build_parser()
    for argv in (
        ["fit-router", "--out", "router.pt"],
        ["fit-shard", "--router-path", "router.pt"],
        ["fit-weights", "--experts", "experts.pt", "--out", "lens.pt"],
        ["evaluate", "--lens", "lens.pt", "--out", "ranks.csv"],
    ):
        assert parser.parse_args(argv).device_map is None
        assert parser.parse_args([*argv, "--device-map", "auto"]).device_map == "auto"
        # The DeepSeek-V4 load settings ride on the same flags at every stage.
        parsed = parser.parse_args(argv)
        assert (parsed.attn_implementation, parsed.experts_implementation, parsed.dequantize_fp8) == ("sdpa", None, False)
        parsed = parser.parse_args(
            [*argv, "--attn-implementation", "eager", "--experts-implementation", "eager", "--dequantize-fp8"]
        )
        assert (parsed.attn_implementation, parsed.experts_implementation, parsed.dequantize_fp8) == ("eager", "eager", True)


def test_main_merge_experts_saves_experts_and_stamp(
    tmp_path: Path, shard_dirs: list[str]
) -> None:
    experts_path = tmp_path / "experts" / "recipe_experts.pt"
    assert (
        main(
            [
                "merge-experts",
                "--checkpoint-dirs",
                *shard_dirs,
                "--num-clusters",
                str(NUM_CLUSTERS),
                "--min-kept-positions",
                "1",
                "--checkpoint-name",
                "cli_merged",
                "--out",
                str(experts_path),
            ]
        )
        == 0
    )
    loaded = ExpertJacobians.load(str(experts_path))
    assert loaded.num_experts == NUM_CLUSTERS
    assert loaded.layers == LAYERS
    assert loaded.min_kept_positions == 1
    assert loaded.config.checkpoint_name == "cli_merged"
    stamp = json.loads((tmp_path / "experts" / "recipe_experts_stamp.json").read_text())
    assert stamp["stage"] == "merge-experts"
    assert stamp["num_prompts_trained_on"] == len(PROMPTS)
    assert sorted(int(layer) for layer in stamp["fallback_experts"]) == LAYERS


def test_main_fit_weights_and_evaluate_end_to_end_on_tiny_model(
    eval_model: TinyDecoder,
    experts: ExpertJacobians,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """fit-weights on its default split (fit:0) writes an fp32 lens, the weights
    JSON and a stamp; evaluate on its default split (held-out:0) writes ranks, the
    pair table with the macro row and a stamp; the two splits are disjoint and
    cover every item. Scoring the lens on items it was fitted on logs a warning."""
    monkeypatch.setattr(jpp_cli, "get_hf_model", lambda name, **_kwargs: eval_model)
    data_dir = tmp_path / "evals"
    data_dir.mkdir()
    correctness_csv = write_fake_eval_data(data_dir, hf_model_name="tiny")
    experts_path = tmp_path / "experts.pt"
    experts.save(str(experts_path))
    shared_argv = [
        "--hf-model-name",
        "tiny",
        "--layers",
        "0,1,2",
        "--readout-max-seq-len",
        str(READOUT_MAX_SEQ_LEN),
        "--eval-data-dir",
        str(data_dir),
        "--correctness-csv",
        str(correctness_csv),
    ]

    lens_path = tmp_path / "lens" / "jpp.pt"
    assert (
        main(
            [
                "fit-weights",
                *shared_argv,
                "--experts",
                str(experts_path),
                "--cache-path",
                str(tmp_path / "readout_residuals.pt"),
                "--checkpoint-name",
                "cli_jpp",
                "--k",
                "3",
                "--top-n",
                str(NUM_CLUSTERS),
                "--steps",
                "3",
                "--out",
                str(lens_path),
            ]
        )
        == 0
    )
    saved_lens = BaseLens.load(str(lens_path))
    assert isinstance(saved_lens, JacobianLens)
    assert saved_lens.config.checkpoint_name == "cli_jpp"
    assert saved_lens.jacobians_L_dict_FN[0].dtype == t.float32
    saved_weighting = ExpertWeighting.load(tmp_path / "lens" / "jpp_expert_weights.json")
    assert saved_weighting.layers == LAYERS
    # The CLI's --k / --top-n / --steps flags are what the search ran under.
    assert saved_weighting.condense_config == CondenseConfig(pass_k=3, top_n=NUM_CLUSTERS, steps=3)
    weights_L_dict_E = saved_weighting.weights_L_dict_E
    # The saved (fp32) lens is exactly the JSON weights applied to the experts
    # the CLI read (the experts file stores fp16, so reload it rather than
    # reusing the in-memory fp32 fixture).
    recombined = ExpertJacobians.load(str(experts_path)).combine(
        weights_L_dict_E, checkpoint_name="check"
    )
    for layer in LAYERS:
        assert t.equal(
            saved_lens.jacobians_L_dict_FN[layer], recombined.jacobians_L_dict_FN[layer]
        )
    fit_stamp = json.loads((tmp_path / "lens" / "jpp_stamp.json").read_text())
    assert fit_stamp["stage"] == "fit-weights"
    # The default fraction keeps one of each eval's four items for fitting.
    assert len(fit_stamp["fitting_item_names"]) == len(MACRO_EVALS)
    assert sorted(int(layer) for layer in fit_stamp["train_macro_pass"]) == LAYERS

    ranks_path = tmp_path / "eval" / "ranks.csv"
    assert (
        main(
            [
                "evaluate",
                *shared_argv,
                "--lens",
                str(lens_path),
                "--ks",
                "1,10",
                "--out",
                str(ranks_path),
            ]
        )
        == 0
    )
    ranks_df = pd.read_csv(ranks_path)
    assert set(ranks_df["lens"]) == {"jpp"}
    pass_df = pd.read_csv(tmp_path / "eval" / "ranks_pass_at_k.csv")
    assert set(pass_df["eval"]) == {*MACRO_EVALS, "macro"}
    eval_stamp = json.loads((tmp_path / "eval" / "ranks_stamp.json").read_text())
    assert eval_stamp["stage"] == "evaluate"
    assert eval_stamp["lenses"] == {"jpp": str(lens_path)}
    scored_names = eval_stamp["scored_item_names"]
    assert set(ranks_df["item"]) == set(scored_names)
    assert not set(scored_names) & set(fit_stamp["fitting_item_names"])
    assert sorted(scored_names + fit_stamp["fitting_item_names"]) == sorted(
        item_names(fake_recipe_items())
    )
    assert "in-sample" not in caplog.text

    # Every item includes the fitting split: one in-sample warning naming the count.
    all_items_path = tmp_path / "eval" / "all_items_ranks.csv"
    assert (
        main(
            [
                "evaluate",
                *shared_argv,
                "--lens",
                str(lens_path),
                "--items",
                "all",
                "--out",
                str(all_items_path),
            ]
        )
        == 0
    )
    in_sample_warnings = [
        record.getMessage() for record in caplog.records if "in-sample" in record.getMessage()
    ]
    assert in_sample_warnings == [
        f"{lens_path}: {len(MACRO_EVALS)} of the {len(fake_recipe_items())} scored items were in "
        "its expert-weight fitting split, so its scores on them are in-sample"
    ]


### pooled lens


def test_merge_experts_pooled_lens_out_is_the_fp32_matched_baseline(
    shard_dirs: list[str], tmp_path: Path
) -> None:
    """``merge-experts --pooled-lens-out`` writes the position-weighted pooled
    Jacobian of the merged sums as an fp32 JacobianLens, equal to
    ``build_matched_baseline_lens`` on the same sums; the ``pooled-lens`` stage
    from the saved experts file gives the fp16-rounded version of the same map
    (the file stores the pooled key in fp16), so it agrees to fp16 precision
    but not bit for bit."""
    experts_path = tmp_path / "experts.pt"
    pooled_path = tmp_path / "pooled_fp32.pt"
    main(
        [
            "merge-experts",
            "--checkpoint-dirs",
            *shard_dirs,
            "--num-clusters",
            str(NUM_CLUSTERS),
            "--device",
            "cpu",
            "--out",
            str(experts_path),
            "--pooled-lens-out",
            str(pooled_path),
        ]
    )
    merged = merge_shard_sums(shard_dirs)
    reference = build_matched_baseline_lens(merged.fit_sums, merged.config, "reference")
    fp32_pooled = BaseLens.load(str(pooled_path))
    assert isinstance(fp32_pooled, JacobianLens)
    for layer in LAYERS:
        assert t.equal(
            fp32_pooled.jacobians_L_dict_FN[layer].float(),
            reference.jacobians_L_dict_FN[layer].float(),
        )

    from_file_path = tmp_path / "pooled_from_file.pt"
    main(["pooled-lens", "--experts", str(experts_path), "--out", str(from_file_path)])
    from_file = BaseLens.load(str(from_file_path))
    assert isinstance(from_file, JacobianLens)
    for layer in LAYERS:
        assert t.allclose(
            from_file.jacobians_L_dict_FN[layer].float(),
            reference.jacobians_L_dict_FN[layer].float(),
            rtol=1e-3,
            atol=1e-5,
        )
    assert (tmp_path / "pooled_from_file_stamp.json").exists()


### stamps


def test_write_stamp_writes_provenance_json(tmp_path: Path) -> None:
    stamp_path = tmp_path / "router_stamp.json"
    write_stamp(
        str(stamp_path),
        {"stage": "fit-router", "args": {"num_prompts": 4, "layers": [0, 1]}},
    )
    stamp = json.loads(stamp_path.read_text())
    assert stamp["stage"] == "fit-router"
    assert stamp["args"] == {"num_prompts": 4, "layers": [0, 1]}
    assert {"git_hash", "timestamp", "library_version"} <= set(stamp)
    assert stamp["git_hash"] is None or len(stamp["git_hash"]) == 40


def test_load_fit_prompts_from_json_takes_the_prefix_and_checks_the_count(tmp_path) -> None:
    pool_path = tmp_path / "pool.json"
    pool_path.write_text(json.dumps(["p0", "p1", "p2"]))
    assert jpp_cli.load_fit_prompts_from_json(str(pool_path), 2) == ["p0", "p1"]
    with pytest.raises(ValueError, match="3 prompts; 4 requested"):
        jpp_cli.load_fit_prompts_from_json(str(pool_path), 4)
