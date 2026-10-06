"""The probe-swap library functions: formatting ids, lens slicing, the run
wrapper, the success table and the best scale.

The run wrapper is checked on TinyDecoder (with ``tokenizers.CharTokenizer``,
under which only single letters are single-token, so the trials are swapped
for synthetic letter trials); the scoring functions are checked on hand-built
trial tables whose counts are worked out in the comments.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch as t

from lens_evals.causal_evals import (
    SwapEvalRunner,
    SwapTrial,
    best_scale_rows,
    probe_swap_formatting_token_ids,
    probe_swap_success_table,
    run_probe_swap,
    slice_lens_to_layers,
)
from lens_evals.causal_evals import probe_swap as probe_swap_module
from lens_evals.causal_evals.probe_swap import SUCCESS_TABLE_COLUMNS
from workspace_lens.tests.fixtures import (
    BAND_LAYERS,
    make_band_decoder,
    make_logit_lens,
    make_random_band_jacobian_lens,
)
from workspace_lens.tests.tokenizers import CharTokenizer

### RUNNING


def test_formatting_token_ids_are_the_non_word_tokens() -> None:
    # CharTokenizer: letters are ids 1-26; 0 (BOS) and 30 (EOS) are special;
    # 27 "\n", 28 " ", 29 and 31 "?" decode to no letter or digit.
    model = make_band_decoder(CharTokenizer())
    assert probe_swap_formatting_token_ids(model) == [0, 27, 28, 29, 30, 31]


def test_slice_lens_to_layers_keeps_only_those_layers(tmp_path: Path) -> None:
    lens = make_random_band_jacobian_lens(tmp_path, seed=1, checkpoint_name="jac")
    sliced = slice_lens_to_layers(lens, [2, 6])
    assert sliced.source_layers == [2, 6]
    assert sliced.config.source_layers == [2, 6]
    assert sliced.config.checkpoint_name == "jac"
    assert sliced.jacobians_L_dict_FN is not None and lens.jacobians_L_dict_FN is not None
    for layer in (2, 6):
        assert t.equal(sliced.jacobians_L_dict_FN[layer], lens.jacobians_L_dict_FN[layer])

    with pytest.raises(ValueError, match=r"lacks layers \[3\]"):
        slice_lens_to_layers(lens, [2, 3])


def test_run_probe_swap_equals_a_direct_clamp_swap_runner_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run_probe_swap`` is the clamp-swap runner on the loaded trials with its
    default settings; slicing the J-lens to ``layers`` leaves the rows
    unchanged. One trial has a multi-token source, so it is dropped."""
    trials = [
        SwapTrial("probe-swap", "t1", "abcdefghij", "c", "d", ("d",), ("c",), category="x"),
        SwapTrial("probe-swap", "t2", "klmnopqrst", "m", "q", ("q",), ("m",), category="x"),
        SwapTrial("probe-swap", "multi", "abcdefghij", "cat", "d", ("d",), ("c",)),
    ]
    loader_calls: list[bool] = []

    def fake_loader(*, include_controls: bool) -> list[SwapTrial]:
        loader_calls.append(include_controls)
        return trials

    monkeypatch.setattr(probe_swap_module, "load_probe_swap_trials", fake_loader)
    model = make_band_decoder(CharTokenizer())
    jlens = make_random_band_jacobian_lens(tmp_path, seed=1, checkpoint_name="jac")
    logit_lens = make_logit_lens(BAND_LAYERS, tmp_path)
    layers = [2, 4]

    trials_df, dropped_concepts = run_probe_swap(
        model, {"jlens": jlens, "logit": logit_lens}, layers=layers, scales=(1.0, 2.0),
        max_seq_len=64,
    )

    assert loader_calls == [False]
    assert dropped_concepts == [("probe-swap", "multi", "cat")]
    direct_runner = SwapEvalRunner(
        model,
        {"jlens": jlens, "logit": logit_lens},
        band_layers=layers,
        scales=(1.0, 2.0),
        max_seq_len=64,
    )
    pd.testing.assert_frame_equal(trials_df, direct_runner.run(trials))
    assert len(trials_df) == 8  # 2 trials x 2 lenses x 2 scales
    assert (trials_df["n_swapped_layers"] == len(layers)).all()


### SCORING


def trial_rows(
    lens: str, scale: float, ranks: list[tuple[str, int, int, int]], variant: str = "main"
) -> list[dict[str, object]]:
    """Rows of ``(item, clean_baseline_word_rank, clean_success_word_rank,
    success_word_rank)`` for one lens and scale."""
    return [
        {
            "item": item,
            "variant": variant,
            "lens": lens,
            "scale": scale,
            "clean_baseline_word_rank": clean_baseline,
            "clean_success_word_rank": clean_success,
            "success_word_rank": success,
        }
        for item, clean_baseline, clean_success, success in ranks
    ]


def test_success_table_counts_eligible_main_trials_on_word_ranks() -> None:
    trials_df = pd.DataFrame(
        trial_rows(
            "a",
            1.0,
            [
                ("i1", 1, 10, 1),  # top-1 hit; top-5 eligible, hit
                ("i2", 1, 3, 2),  # top-1 miss; swapped answer already clean top 5
                ("i3", 2, 20, 1),  # clean answer wrong: not eligible
                ("i4", 1, 6, 5),  # top-1 miss; top-5 hit
                ("i5", 1, 50, 7),  # top-1 miss; top-5 miss
                ("i6", -1, 50, 1),  # ungradeable baseline: not eligible
            ],
        )
        + trial_rows("a", 1.0, [("i3", 1, 50, 1)], variant="random_null")  # ignored
        + trial_rows(
            "a",
            2.0,
            [
                ("i1", 1, 10, 3),
                ("i2", 1, 3, 1),
                ("i3", 2, 20, 1),
                ("i4", 1, 6, 1),
                ("i5", 1, 50, 1),
            ],
        )
        + trial_rows("b", 1.0, [("i1", 2, 10, 1)])  # nothing eligible
    )

    table = probe_swap_success_table(trials_df)

    assert list(table.columns) == list(SUCCESS_TABLE_COLUMNS)
    rows = table.set_index(["lens", "scale"])
    counts = ["top1_hits", "top1_trials", "top5_hits", "top5_trials"]
    # Scale 1: top-1 eligible i1, i2, i4, i5 (hit i1); top-5 eligible i1, i4, i5 (hits i1, i4).
    assert rows.loc[("a", 1.0), counts].tolist() == [1, 4, 2, 3]
    assert rows.loc[("a", 1.0), "top1_rate"] == 0.25
    assert rows.loc[("a", 1.0), "top5_rate"] == pytest.approx(2 / 3)
    # Scale 2: top-1 hits i2, i4, i5; top-5 hits i1, i4, i5.
    assert rows.loc[("a", 2.0), counts].tolist() == [3, 4, 3, 3]
    assert rows.loc[("b", 1.0), ["top1_hits", "top1_trials"]].tolist() == [0, 0]
    assert pd.isna(rows.loc[("b", 1.0), "top1_rate"])


def test_best_scale_rows_take_the_highest_top1_rate_with_ties_to_the_smaller_scale() -> None:
    success_table = pd.DataFrame(
        [
            ("a", 0.5, 1, 4, 0.25, 0, 3, 0.0),
            ("a", 1.0, 2, 4, 0.5, 1, 3, 1 / 3),
            ("a", 2.0, 2, 4, 0.5, 3, 3, 1.0),  # ties scale 1 on top-1: scale 1 wins
            ("a", 4.0, 0, 4, 0.0, 0, 3, 0.0),
            ("b", 0.5, 3, 10, 0.3, 2, 8, 0.25),
            ("b", 1.0, 2, 10, 0.2, 5, 8, 0.625),
        ],
        columns=list(SUCCESS_TABLE_COLUMNS),
    )

    best = best_scale_rows(success_table)

    assert best["lens"].tolist() == ["a", "b"]
    assert best["scale"].tolist() == [1.0, 0.5]
    # The top-5 columns are those at the best top-1 scale, not the best top-5.
    assert best["top5_hits"].tolist() == [1, 2]
