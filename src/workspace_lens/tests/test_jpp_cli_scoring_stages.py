"""The CLI's scoring stages on tiny CPU models: the Logit Lens and released-format lens files
(``--lens``), scoring with and without Readout Filtering (``--no-readout-filter``), the
probe-swap stage and the quickstart readout."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch as t

from jlens.tests.tiny import TinyDecoder
from lens_evals.causal_evals import SwapTrial
from lens_evals.readout_evals.readout_eval_items import (
    DEFAULT_FIT_FRACTION,
    MACRO_EVALS,
    ReadoutEvalItem,
    non_semantic_token_ids,
)
from workspace_lens.fitting.condense_experts import CondenseConfig, ExpertJacobians
from workspace_lens.lenses.base_lens import BaseLens
from workspace_lens.lenses.jacobian_lens import JacobianLens
from workspace_lens.lenses.logit_lens import LogitLens
from workspace_lens.tests.fixtures import (
    make_tiny_decoder,
    make_tiny_lens_config,
    write_readout_eval_json,
)
from workspace_lens.tests.tokenizers import CharTokenizer
from workspace_lens.utils import load_lens_file, vocab_size_of

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import jpp_cli  # noqa: E402

LAYERS = [0, 1, 2]
NUM_CLUSTERS = 2
READOUT_MAX_SEQ_LEN = 32
PROMPTS = ["abcdefghij " * 5, "klmnopqrst " * 5, "uvwxyzabcd " * 5, "efghijklmn " * 5]
# Four items per macro eval; single letters are single tokens under the character tokenizer.
FAKE_PROMPTS_AND_INTERMEDIATES = [
    ("ab\ncd", ("d", "b")),
    ("ac\nde", ("e",)),
    ("bd\nca", ("a", "c")),
    ("cb\nad", ("d",)),
]


def fake_recipe_items() -> list[ReadoutEvalItem]:
    return [
        ReadoutEvalItem(slug, f"{slug}-{idx}", prompt, intermediates[0], intermediates)
        for slug in MACRO_EVALS
        for idx, (prompt, intermediates) in enumerate(FAKE_PROMPTS_AND_INTERMEDIATES)
    ]


@pytest.fixture(scope="module")
def eval_model() -> TinyDecoder:
    return make_tiny_decoder(CharTokenizer())


@pytest.fixture(scope="module")
def experts(tmp_path_factory: pytest.TempPathFactory) -> ExpertJacobians:
    model = TinyDecoder(n_layers=4, d_model=8)
    router = jpp_cli.fit_router(
        model,
        PROMPTS,
        layers=LAYERS,
        num_clusters=NUM_CLUSTERS,
        projection_dim=4,
        skip_first_n_positions=16,
        max_seq_len=64,
    )
    config = make_tiny_lens_config(tmp_path_factory.mktemp("shards"), "experts")
    shard_dir = jpp_cli.fit_shard(
        model, PROMPTS, routers=[router], shard_idx=0, num_shards=1, config=config
    )
    return jpp_cli.merge_experts(
        [shard_dir],
        num_clusters=NUM_CLUSTERS,
        min_kept_positions=1,
        checkpoint_name="experts",
        device="cpu",
    )


@pytest.fixture(scope="module")
def excluded_token_ids(eval_model: TinyDecoder) -> list[int]:
    return non_semantic_token_ids(
        eval_model.tokenizer, vocab_size=vocab_size_of(eval_model), items=fake_recipe_items()
    )


@pytest.fixture(scope="module")
def scoreable_items(eval_model: TinyDecoder) -> list[ReadoutEvalItem]:
    runner = jpp_cli.readout_runner(eval_model, layers=LAYERS, max_seq_len=READOUT_MAX_SEQ_LEN)
    return runner.scoreable_items(fake_recipe_items())


@pytest.fixture(scope="module")
def lens(
    eval_model: TinyDecoder,
    experts: ExpertJacobians,
    excluded_token_ids: list[int],
    scoreable_items: list[ReadoutEvalItem],
) -> JacobianLens:
    """A J++ Lens whose expert weights were fitted on the default fitting split."""
    fitting_names = [
        item.name
        for item in jpp_cli.select_items(
            scoreable_items, "fit:0", fit_fraction=DEFAULT_FIT_FRACTION
        )
    ]
    _, lens = jpp_cli.fit_weights(
        eval_model,
        experts,
        fake_recipe_items(),
        layers=LAYERS,
        excluded_token_ids=excluded_token_ids,
        cache_path=None,
        checkpoint_name="jpp",
        hf_model_name="tiny",
        max_seq_len=READOUT_MAX_SEQ_LEN,
        fitting_item_names=fitting_names,
        condense_config=CondenseConfig(pass_k=3, top_n=NUM_CLUSTERS, steps=3),
    )
    return lens


def test_load_lenses_builds_the_logit_lens_and_reads_both_file_formats(
    eval_model: TinyDecoder, lens: JacobianLens, tmp_path: Path
) -> None:
    lens_path = tmp_path / "jpp.pt"
    lens.save(str(lens_path), dtype=t.float32)
    # Doubled maps, so the two files hold different lenses.
    released_jacobians_L_dict_FN = {
        layer: 2 * lens.jacobians_L_dict_FN[layer] for layer in LAYERS
    }
    released_format_path = tmp_path / "released_j_lens.pt"
    t.save(
        {
            "J": released_jacobians_L_dict_FN,
            "n_prompts": 4,
            "source_layers": LAYERS,
            "d_model": eval_model.d_model,
        },
        released_format_path,
    )
    lenses = jpp_cli.load_lenses(
        ["logit", str(lens_path), str(released_format_path)],
        eval_model,
        hf_model_name="tiny",
        layers=LAYERS,
    )
    assert list(lenses) == ["logit", "jpp", "released_j_lens"]
    assert isinstance(lenses["logit"], LogitLens)
    assert lenses["logit"].source_layers == LAYERS
    for name, expected_L_dict_FN in (
        ("jpp", lens.jacobians_L_dict_FN),
        ("released_j_lens", released_jacobians_L_dict_FN),
    ):
        assert isinstance(lenses[name], JacobianLens)
        for layer in LAYERS:
            assert t.equal(lenses[name].jacobians_L_dict_FN[layer], expected_L_dict_FN[layer])
    with pytest.raises(ValueError, match="share the name"):
        jpp_cli.load_lenses(
            [str(lens_path), str(lens_path)], eval_model, hf_model_name="tiny", layers=LAYERS
        )


def write_fake_eval_data(data_dir: Path) -> Path:
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
    rows = ["eval,item,tiny"] + [
        f"{item.eval_slug},{item.name},True" for item in fake_recipe_items()
    ]
    correctness_csv.write_text("\n".join(rows) + "\n")
    return correctness_csv


def test_main_evaluate_writes_recall_tables_with_and_without_readout_filtering(
    eval_model: TinyDecoder,
    lens: JacobianLens,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """evaluate with the Logit Lens and a saved lens writes ranks, the item- and pair-weighted
    recall table and a stamp; Readout Filtering is on unless --no-readout-filter is passed."""
    monkeypatch.setattr(jpp_cli, "get_hf_model", lambda name, **_kwargs: eval_model)
    data_dir = tmp_path / "evals"
    data_dir.mkdir()
    correctness_csv = write_fake_eval_data(data_dir)
    lens_path = str(tmp_path / "jpp.pt")
    lens.save(lens_path, dtype=t.float32)
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
        "--ks",
        "1,3",
    ]

    raw_path = tmp_path / "raw" / "ranks.csv"
    assert (
        jpp_cli.main(
            [
                "evaluate",
                *shared_argv,
                "--lens",
                "logit",
                lens_path,
                "--no-readout-filter",
                "--out",
                str(raw_path),
            ]
        )
        == 0
    )
    raw_ranks_df = pd.read_csv(raw_path)
    assert set(raw_ranks_df["lens"]) == {"logit", "jpp"}
    raw_stamp = json.loads((tmp_path / "raw" / "ranks_stamp.json").read_text())
    assert raw_stamp["num_excluded_token_ids"] == 0
    raw_pass_df = pd.read_csv(tmp_path / "raw" / "ranks_pass_at_k.csv")
    assert set(raw_pass_df["weighting"]) == {"item", "pair"}
    assert {"macro", *MACRO_EVALS} == set(raw_pass_df["eval"])

    filtered_path = tmp_path / "filtered" / "ranks.csv"
    assert (
        jpp_cli.main(
            ["evaluate", *shared_argv, "--lens", lens_path, "--out", str(filtered_path)]
        )
        == 0
    )
    filtered_stamp = json.loads((tmp_path / "filtered" / "ranks_stamp.json").read_text())
    assert filtered_stamp["num_excluded_token_ids"] > 0
    assert set(pd.read_csv(filtered_path)["item"]) == set(filtered_stamp["scored_item_names"])


def test_load_lens_file_refuses_a_file_that_is_not_a_lens(tmp_path: Path) -> None:
    router_like_path = tmp_path / "router.pt"
    t.save({"centroids": t.zeros(2, 2)}, router_like_path)
    with pytest.raises(ValueError, match="not a lens file"):
        load_lens_file(str(router_like_path), hf_model_name="tiny")
    released_format_path = tmp_path / "released.pt"
    t.save(
        {"J": {0: t.eye(8)}, "n_prompts": 1, "source_layers": [0], "d_model": 8},
        released_format_path,
    )
    with pytest.raises(ValueError, match="embeds no config"):
        BaseLens.load(str(released_format_path))
    assert isinstance(load_lens_file(str(released_format_path), hf_model_name="tiny"), JacobianLens)


def test_main_probe_swap_writes_trials_tables_and_stamp(
    eval_model: TinyDecoder,
    lens: JacobianLens,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The probe-swap stage end to end on a tiny model: the trial run is replaced by synthetic rows
    (concept names are not single tokens under the character tokenizer), everything else runs."""
    monkeypatch.setattr(jpp_cli, "get_hf_model", lambda name, **_kwargs: eval_model)
    multihop_items = [item for item in fake_recipe_items() if item.eval_slug == "multihop"]
    trials = [
        SwapTrial(
            eval_slug="probe-swap",
            name=f"p{i}",
            prompt=item.prompt,
            source="a",
            target="b",
            success_surfaces=("b",),
            baseline_surfaces=("a",),
        )
        for i, item in enumerate(multihop_items)
    ]

    def fake_run_probe_swap(model, lenses, *, layers, scales, max_seq_len):
        rows = [
            {
                "variant": "main",
                "lens": lens_name,
                "item": trial.name,
                "scale": scale,
                "clean_baseline_word_rank": 1,
                "clean_success_word_rank": 9,
                "success_word_rank": 1,
            }
            for lens_name in lenses
            for trial in trials
            for scale in scales
        ]
        return pd.DataFrame(rows), []

    monkeypatch.setattr(jpp_cli, "run_probe_swap", fake_run_probe_swap)
    lens_path = str(tmp_path / "jpp.pt")
    lens.save(lens_path, dtype=t.float32)
    out_path = tmp_path / "probe" / "trials.parquet"
    argv = [
        "probe-swap",
        "--hf-model-name",
        "tiny",
        "--layers",
        "0,1,2",
        "--lens",
        "logit",
        lens_path,
        "--scales",
        "1,2",
        "--out",
        str(out_path),
    ]
    assert jpp_cli.main(argv) == 0
    trials_df = pd.read_parquet(out_path)
    assert set(trials_df["lens"]) == {"logit", "jpp"}
    best_df = pd.read_csv(tmp_path / "probe" / "trials_best_scale.csv")
    assert set(best_df["lens"]) == {"logit", "jpp"}
    assert (best_df["scale"] == 1.0).all()  # every scale ties, so the smaller one wins
    success_df = pd.read_csv(tmp_path / "probe" / "trials_success.csv")
    assert set(success_df["scale"]) == {1.0, 2.0}
    stamp = json.loads((tmp_path / "probe" / "trials_stamp.json").read_text())
    assert stamp["stage"] == "probe-swap"


def test_quickstart_reads_out_the_final_position_with_readout_filtering(
    eval_model: TinyDecoder, lens: JacobianLens
) -> None:
    spec = importlib.util.spec_from_file_location(
        "quickstart", Path(__file__).resolve().parents[3] / "scripts" / "quickstart.py"
    )
    quickstart = importlib.util.module_from_spec(spec)
    sys.modules["quickstart"] = quickstart  # dataclasses look their module up
    spec.loader.exec_module(quickstart)
    prompt = "ab\ncd"
    lens_logits, model_logits, _ = lens.apply(
        eval_model, prompt, layers=LAYERS, token_positions_for_residuals=[-1]
    )
    excluded = lens_logits[1][0].topk(2).indices.tolist()  # the two best tokens at layer 1
    readouts = quickstart.read_out_prompt(
        eval_model, lens, prompt, layers=LAYERS, top_k=3, excluded_token_ids=excluded
    )
    assert list(readouts.lens_top_ids_L_dict) == LAYERS
    assert not set(readouts.lens_top_ids_L_dict[1]) & set(excluded)
    expected_layer_1 = lens_logits[1][0].clone()
    expected_layer_1[excluded] = float("-inf")
    assert readouts.lens_top_ids_L_dict[1] == expected_layer_1.topk(3).indices.tolist()
    assert readouts.model_top_ids == model_logits[0].topk(3).indices.tolist()
