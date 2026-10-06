"""Causal swap evals: the swap runner and the probe-swap loader on TinyDecoder.

TinyDecoder's own tokenizer (``_ByteTokenizer``) does not accept
``add_special_tokens``; the runner calls the tokenizer the way the real Qwen
tokenizer is called, so tests attach ``tokenizers.CharTokenizer`` instead.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import torch as t

from jlens.tests.tiny import TinyDecoder
from lens_evals.causal_evals import (
    SwapEvalRunner,
    SwapTrial,
    load_probe_swap_trials,
)
from lens_evals.eval_utils import resolve_swap_token
from workspace_lens.lenses.jacobian_lens import JacobianLens
from workspace_lens.lenses.logit_lens import LogitLens
from workspace_lens.tests.fixtures import (
    BAND_LAYERS,
    make_band_decoder,
    make_logit_lens,
    make_random_band_jacobian_lens,
)
from workspace_lens.tests.tokenizers import CharTokenizer
from workspace_lens.utils import get_unembed_matrix, readout_vectors


@pytest.fixture
def model() -> TinyDecoder:
    return make_band_decoder(CharTokenizer())


@pytest.fixture
def jlens(tmp_path: Path) -> JacobianLens:
    return make_random_band_jacobian_lens(tmp_path, seed=1, checkpoint_name="jac")


@pytest.fixture
def logit_lens(tmp_path: Path) -> LogitLens:
    return make_logit_lens(BAND_LAYERS, tmp_path)


### resolve_swap_token / helpers


def test_resolve_swap_token_prefers_single_token(model: TinyDecoder) -> None:
    # Under the char tokenizer only single letters are single-token; " d" is two
    # tokens, so the bare "d" surface is chosen.
    token_id, surface = resolve_swap_token(model.tokenizer, "d")
    assert surface == "d"
    assert token_id == model.tokenizer("d", add_special_tokens=False).input_ids[0]


def test_resolve_swap_token_drops_multitoken(model: TinyDecoder) -> None:
    token_id, surface = resolve_swap_token(model.tokenizer, "cat")
    assert token_id is None and surface is None


### logit-lens readout vectors are raw W_U rows


def test_logit_readout_vectors_are_unembedding_rows(
    model: TinyDecoder, logit_lens: LogitLens
) -> None:
    unembed_VF = get_unembed_matrix(model)
    vectors = readout_vectors(logit_lens, model, 5, layers=BAND_LAYERS)
    for layer in BAND_LAYERS:
        t.testing.assert_close(vectors[layer], unembed_VF[5].float())


def test_jacobian_readout_vectors_use_jacobian(
    model: TinyDecoder, jlens: JacobianLens
) -> None:
    unembed_VF = get_unembed_matrix(model)
    vectors = readout_vectors(jlens, model, 5, layers=BAND_LAYERS)
    for layer in BAND_LAYERS:
        expected = unembed_VF[5].float() @ jlens.jacobians_L_dict_FN[layer]
        t.testing.assert_close(vectors[layer], expected)


### the runner end-to-end


def test_runner_swaps_and_records(
    model: TinyDecoder, jlens: JacobianLens, logit_lens: LogitLens
) -> None:
    runner = SwapEvalRunner(
        model,
        {"jlens": jlens, "logit": logit_lens},
        band_layers=BAND_LAYERS,
        scales=(1.0, 2.0),
        max_seq_len=64,
    )
    trials = [
        SwapTrial(
            eval_slug="probe-swap",
            name="t1",
            prompt="abcdefghij",
            source="c",
            target="d",
            success_surfaces=("d", " d"),
            baseline_surfaces=("c", " c"),
            category="x",
        )
    ]
    df = runner.run(trials)
    assert len(df) == 4  # 2 lenses x 2 scales
    for column in (
        "success_rank",
        "edit_norm_ratio",
        "delta_logprob",
        "success_top5",
        "clean_success_rank",
    ):
        assert column in df.columns
    # A stronger swap perturbs the residual more.
    for lens_name in ("jlens", "logit"):
        rows = df[df["lens"] == lens_name].sort_values("scale")
        assert (
            rows["edit_norm_ratio"].iloc[1] > rows["edit_norm_ratio"].iloc[0]
        ), lens_name
    assert not runner.dropped_concepts

    # Coverage: only the bare letter is single-token under the char tokenizer,
    # so each surface set contributes exactly one vocab id.
    assert (df["n_success_ids"] == 1).all()
    assert (df["n_baseline_ids"] == 1).all()
    # The edit applies to every position but the sink (prompt is BOS + 10 chars).
    assert (df["n_swap_positions"] == 10).all()


def test_runner_drops_multitoken_concept_once_across_lenses_and_scales(
    model: TinyDecoder, jlens: JacobianLens, logit_lens: LogitLens
) -> None:
    runner = SwapEvalRunner(
        model,
        {"jlens": jlens, "logit": logit_lens},
        band_layers=BAND_LAYERS,
        scales=(1.0, 2.0),
        max_seq_len=64,
    )
    df = runner.run(
        [
            SwapTrial(
                "probe-swap",
                "multi",
                "abcdefghij",
                source="cat",  # multi-token -> dropped
                target="d",
                success_surfaces=("d",),
                baseline_surfaces=("c",),
            )
        ]
    )
    assert df.empty
    # run_trial revisits the trial 2 lenses x 2 scales = 4 times; the casualty
    # is recorded exactly once.
    assert runner.dropped_concepts == [("probe-swap", "multi", "cat")]


def test_runner_records_empty_success_surfaces_as_dropped(
    model: TinyDecoder, jlens: JacobianLens, logit_lens: LogitLens
) -> None:
    runner = SwapEvalRunner(
        model,
        {"jlens": jlens, "logit": logit_lens},
        band_layers=BAND_LAYERS,
        scales=(1.0, 2.0),
        max_seq_len=64,
    )
    df = runner.run(
        [
            SwapTrial(
                "probe-swap",
                "nosuccess",
                "abcdefghij",
                source="c",
                target="d",
                success_surfaces=("cat", "dog", "emu", "fox"),  # all multi-token
                baseline_surfaces=("c",),
            )
        ]
    )
    assert df.empty
    # Visible in dropped_concepts (once), with the first three surfaces named.
    assert runner.dropped_concepts == [("probe-swap", "nosuccess", "success:cat|dog|emu")]


def test_runner_empty_baseline_is_ungradeable_not_dropped(
    model: TinyDecoder,
    jlens: JacobianLens,
    logit_lens: LogitLens,
    caplog: pytest.LogCaptureFixture,
) -> None:
    runner = SwapEvalRunner(
        model,
        {"jlens": jlens, "logit": logit_lens},
        band_layers=BAND_LAYERS,
        scales=(1.0, 2.0),
        max_seq_len=64,
    )
    trials = [
        SwapTrial("probe-swap", "graded", "abcdefghij", "c", "d", ("d",), ("c",)),
        SwapTrial(
            "probe-swap",
            "ungraded",
            "abcdefghij",
            "c",
            "d",
            success_surfaces=("d",),
            baseline_surfaces=("cat",),  # multi-token -> no baseline ids
        ),
    ]
    with caplog.at_level(logging.WARNING, logger="lens_evals.causal_evals.runner"):
        df = runner.run(trials)
    assert len(df) == 8  # neither trial is dropped
    assert not runner.dropped_concepts

    ungraded = df[df["item"] == "ungraded"]
    graded = df[df["item"] == "graded"]
    assert (ungraded["n_baseline_ids"] == 0).all()
    assert (ungraded["clean_baseline_rank"] == -1).all()
    assert (ungraded["baseline_rank"] == -1).all()
    assert ungraded["baseline_logprob"].isna().all()
    assert (graded["n_baseline_ids"] == 1).all()
    assert (graded["clean_baseline_rank"] >= 1).all()

    # Warned once for the trial, not once per (lens, scale); plus the run summary.
    per_trial_warnings = [
        record for record in caplog.records if "baseline ranks will be -1" in record.message
    ]
    assert len(per_trial_warnings) == 1
    assert any("1/2 trials have no single-token baseline" in r.message for r in caplog.records)


### the probe-swap loader against the real JSON (data only, no GPU)


def test_load_probe_swap_trials_shapes() -> None:
    trials = load_probe_swap_trials()
    variants = {trial.variant for trial in trials}
    assert variants == {"main", "answer_swap", "random_null"}
    main = [trial for trial in trials if trial.variant == "main"]
    raw = json.loads(Path("data/jlens/experiments/probe-swap.json").read_text())["items"]
    assert len(main) == len(raw)
    first = main[0]
    assert first.source == raw[0]["intermediate"]
    assert first.target == raw[0]["swap_to"]


def test_load_probe_swap_no_controls() -> None:
    trials = load_probe_swap_trials(include_controls=False)
    assert {trial.variant for trial in trials} == {"main"}


### word ranks, qualitative columns, argument checks


def _one_trial(name: str = "t1", **overrides) -> SwapTrial:
    fields = dict(
        eval_slug="probe-swap",
        name=name,
        prompt="abcdefghij",
        source="c",
        target="d",
        success_surfaces=("d",),
        baseline_surfaces=("c",),
    )
    fields.update(overrides)
    return SwapTrial(**fields)


def test_word_ranks_and_top_tokens_columns(
    model: TinyDecoder, jlens: JacobianLens
) -> None:
    # Everything except letters counts as formatting for the char tokenizer, so
    # the word rank can only be better (lower) than the full-vocab rank.
    runner = SwapEvalRunner(
        model,
        {"jlens": jlens},
        band_layers=BAND_LAYERS,
        scales=(1.0,),
        max_seq_len=64,
        formatting_token_ids=[0, 27, 28, 29, 30, 31],
    )
    df = runner.run([_one_trial()])
    assert (df["success_word_rank"] <= df["success_rank"]).all()
    assert (df["clean_success_word_rank"] <= df["clean_success_rank"]).all()
    for column in ("clean_top_tokens", "swapped_top_tokens", "swapped_top_word_tokens"):
        tokens = json.loads(df[column].iloc[0])
        assert len(tokens) == 5
    # A formatting token is never among the word-only top tokens.
    assert not set(json.loads(df["swapped_top_word_tokens"].iloc[0])) & {"", "\n", " ", "?"}


def test_runner_rejects_formatting_token_as_scored_surface(
    model: TinyDecoder, jlens: JacobianLens
) -> None:
    runner = SwapEvalRunner(
        model, {"jlens": jlens}, band_layers=BAND_LAYERS, scales=(1.0,), max_seq_len=64,
        formatting_token_ids=[28],  # the space
    )
    with pytest.raises(ValueError, match="formatting"):
        runner.run([_one_trial(success_surfaces=(" ",))])
