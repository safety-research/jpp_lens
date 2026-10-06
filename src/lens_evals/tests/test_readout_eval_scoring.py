"""Readout-eval scoring: the item- and pair-weighted pass@k (recall@k) with their
macro rows and the stacked table of both, pinned on small hand-built rank tables;
plus a check of both weightings against an independent reference implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lens_evals.readout_evals.readout_eval_items import MACRO_EVALS
from lens_evals.readout_evals.readout_eval_scoring import (
    pair_pass_at_k,
    pass_at_k_by_weighting,
    pass_at_k_over_layers,
)


def _ranks_df(rows: list[tuple[str, str, str, str, int, int]]) -> pd.DataFrame:
    """A rank table from ``(eval, lens, item, intermediate, layer, rank)``
    tuples, with the runner's columns."""
    return pd.DataFrame(
        [
            {
                "eval": eval_slug,
                "lens": lens,
                "item": item,
                "intermediate": intermediate,
                "layer": layer,
                "rank": rank,
                "n_candidates": 1,
            }
            for eval_slug, lens, item, intermediate, layer, rank in rows
        ]
    )


def _hand_ranks_df() -> pd.DataFrame:
    """Two layers, one lens: i1 has intermediates x, y; i2 has x only.

    ranks: i1/x: L0=3, L1=1;  i1/y: L0=12, L1=8;  i2/x: L0=1, L1=2.
    """
    return _ranks_df(
        [
            ("e", "A", "i1", "x", 0, 3),
            ("e", "A", "i1", "x", 1, 1),
            ("e", "A", "i1", "y", 0, 12),
            ("e", "A", "i1", "y", 1, 8),
            ("e", "A", "i2", "x", 0, 1),
            ("e", "A", "i2", "x", 1, 2),
        ]
    )


### ITEM-WEIGHTED PASS@K


def test_item_pass_at_k_hand_checked() -> None:
    ranks_df = _hand_ranks_df()
    item_table = pass_at_k_over_layers(ranks_df, ks=(1, 10))
    assert list(item_table.columns) == ["eval", "lens", "k", "pass_at_k"]
    by_eval_k = item_table.set_index(["eval", "k"])["pass_at_k"]
    # min ranks: i1/x=1, i1/y=8, i2/x=1.
    # pass@1: i1 -> 0.5 (x hits, y doesn't), i2 -> 1.0 => 0.75.
    assert by_eval_k[("e", 1)] == pytest.approx(0.75)
    assert by_eval_k[("e", 10)] == pytest.approx(1.0)
    # "e" is not a macro eval and the five macro evals are absent: NaN macro.
    assert pd.isna(by_eval_k[("macro", 1)]) and pd.isna(by_eval_k[("macro", 10)])


def test_item_pass_at_k_macro_over_the_five_evals_only() -> None:
    """Item-weighted macro: each eval's recall averages its items' hit
    fractions, the macro is the unweighted mean of the five, order-ops gets
    its own row but does not enter it, and a lens missing one macro eval gets
    NaN rather than a narrower mean."""
    rows: list[tuple[str, str, str, str, int, int]] = []
    # Lens "A", k = 10, min over layers 8 and 9:
    # multihop: i1 has x (hit), y (miss), z (miss) -> 1/3; i2 has x (hit) -> 1;
    #   recall (1/3 + 1) / 2 = 2/3 (pair-weighted would be 2/4).
    rows += [
        ("multihop", "A", "i1", "x", 8, 50),
        ("multihop", "A", "i1", "x", 9, 3),
        ("multihop", "A", "i1", "y", 8, 11),
        ("multihop", "A", "i1", "y", 9, 40),
        ("multihop", "A", "i1", "z", 8, 99),
        ("multihop", "A", "i1", "z", 9, 12),
        ("multihop", "A", "i2", "x", 8, 1),
        ("multihop", "A", "i2", "x", 9, 20),
    ]
    # multilingual: one item, both intermediates hit -> 1.
    rows += [
        ("multilingual", "A", "i1", "x", 8, 10),
        ("multilingual", "A", "i1", "y", 9, 2),
    ]
    # typo: i1 misses -> 0; i2 hits -> 1; recall 1/2.
    rows += [("typo", "A", "i1", "x", 8, 11), ("typo", "A", "i2", "x", 8, 4)]
    # association: i1 has x (hit), y (miss) -> 1/2; recall 1/2.
    rows += [("association", "A", "i1", "x", 9, 7), ("association", "A", "i1", "y", 9, 30)]
    # poetry: one miss -> 0.
    rows += [("poetry", "A", "i1", "x", 8, 15)]
    # order-ops always hits (would lift the macro if counted).
    rows += [("order-ops", "A", "i1", "x", 8, 1)]
    # Lens "B": poetry missing, everything else hits.
    rows += [
        (eval_slug, "B", "i1", "x", 8, 1)
        for eval_slug in ("multihop", "multilingual", "typo", "association")
    ]
    table = pass_at_k_over_layers(_ranks_df(rows), ks=(10,)).set_index(["lens", "eval"])[
        "pass_at_k"
    ]
    assert table[("A", "multihop")] == pytest.approx(2 / 3)
    assert table[("A", "multilingual")] == pytest.approx(1.0)
    assert table[("A", "typo")] == pytest.approx(1 / 2)
    assert table[("A", "association")] == pytest.approx(1 / 2)
    assert table[("A", "poetry")] == pytest.approx(0.0)
    # (2/3 + 1 + 1/2 + 1/2 + 0) / 5 = (8/3) / 5 = 8/15.
    assert table[("A", "macro")] == pytest.approx(8 / 15)
    assert table[("A", "order-ops")] == pytest.approx(1.0)
    assert pd.isna(table[("B", "macro")])
    assert table[("B", "multihop")] == pytest.approx(1.0)


def _reference_recall_by_eval(
    ranks: pd.DataFrame, k: int, rank_column: str = "min_rank", weighting: str = "item"
) -> pd.Series:
    """An independent reference implementation of per-eval recall: recall per
    eval from one row per (eval, item, intermediate); the macro is the mean of
    this Series."""
    hits = ranks[rank_column] <= k
    if weighting == "pair":
        return hits.groupby(ranks["eval"]).mean()
    item_scores = hits.groupby([ranks["eval"], ranks["item"]]).mean()
    return item_scores.groupby(level="eval").mean()


def _synthetic_ranks_df(seed: int) -> pd.DataFrame:
    """Two lenses x the five macro evals x 12 items each with 1 to 4
    intermediates, ranks drawn from 1..30 at three readout layers."""
    generator = np.random.default_rng(seed)
    rows: list[tuple[str, str, str, str, int, int]] = []
    for lens in ("A", "B"):
        for eval_slug in MACRO_EVALS:
            for item_idx in range(12):
                num_intermediates = int(generator.integers(1, 5))
                for intermediate_idx in range(num_intermediates):
                    for layer in (8, 16, 24):
                        rank = int(generator.integers(1, 31))
                        rows.append(
                            (
                                eval_slug,
                                lens,
                                f"item-{item_idx}",
                                f"intermediate-{intermediate_idx}",
                                layer,
                                rank,
                            )
                        )
    return _ranks_df(rows)


def test_item_pass_at_k_matches_a_reference_recall() -> None:
    """On a random rank table whose items carry 1 to 4 intermediates, the item
    rows equal the reference item-weighted recall per eval and their mean (the
    macro), the pair rows equal its pair-weighted recall, and the two
    weightings differ (so the comparison can tell them apart)."""
    ranks_df = _synthetic_ranks_df(seed=0)
    ks = (1, 5, 10)
    item_table = pass_at_k_over_layers(ranks_df, ks).set_index(["lens", "k", "eval"])
    pair_table = pair_pass_at_k(ranks_df, ks).set_index(["lens", "k", "eval"])
    min_rank_df = (
        ranks_df.groupby(["eval", "lens", "item", "intermediate"], as_index=False)["rank"]
        .min()
        .rename(columns={"rank": "min_rank"})
    )
    weightings_differ = False
    for lens in ("A", "B"):
        lens_min_rank_df = min_rank_df[min_rank_df["lens"] == lens]
        for k in ks:
            reference_item = _reference_recall_by_eval(lens_min_rank_df, k, weighting="item")
            reference_pair = _reference_recall_by_eval(lens_min_rank_df, k, weighting="pair")
            assert sorted(reference_item.index) == sorted(MACRO_EVALS)
            for eval_slug in MACRO_EVALS:
                assert item_table.loc[(lens, k, eval_slug), "pass_at_k"] == pytest.approx(
                    reference_item[eval_slug], abs=1e-12
                )
                assert pair_table.loc[(lens, k, eval_slug), "pass_at_k"] == pytest.approx(
                    reference_pair[eval_slug], abs=1e-12
                )
            assert item_table.loc[(lens, k, "macro"), "pass_at_k"] == pytest.approx(
                float(reference_item.mean()), abs=1e-12
            )
            assert pair_table.loc[(lens, k, "macro"), "pass_at_k"] == pytest.approx(
                float(reference_pair.mean()), abs=1e-12
            )
            weightings_differ |= abs(reference_item.mean() - reference_pair.mean()) > 1e-6
    assert weightings_differ


### PAIR-WEIGHTED PASS@K


def test_pair_pass_at_k_hand_checked() -> None:
    """Pair-level: the three (item, intermediate) pairs count equally, so
    pass@1 = 2/3 where the item-weighted pass@k gives 0.75."""
    ranks_df = _hand_ranks_df()
    pair_table = pair_pass_at_k(ranks_df, ks=(1, 10))
    assert list(pair_table.columns) == ["eval", "lens", "k", "pass_at_k"]
    by_eval_k = pair_table.set_index(["eval", "k"])["pass_at_k"]
    assert by_eval_k[("e", 1)] == pytest.approx(2 / 3)
    assert by_eval_k[("e", 10)] == pytest.approx(1.0)
    item_level = pass_at_k_over_layers(ranks_df, ks=(1,)).set_index("eval").loc["e", "pass_at_k"]
    assert item_level == pytest.approx(0.75) != by_eval_k[("e", 1)]
    # "e" is not a macro eval and the five macro evals are absent: NaN macro.
    assert pd.isna(by_eval_k[("macro", 1)]) and pd.isna(by_eval_k[("macro", 10)])



def test_pair_pass_at_k_macro_over_the_five_evals_only() -> None:
    """The macro is the unweighted mean over MACRO_EVALS: order-ops gets its
    own row but does not enter it, and a lens missing one macro eval gets
    NaN rather than a narrower mean."""
    rows: list[tuple[str, str, str, str, int, int]] = []
    # Lens "A": one pair per eval, hit iff the eval is multihop or typo
    # (macro = 2/5); order-ops always hits (would lift the macro if counted).
    for eval_slug in (*MACRO_EVALS, "order-ops"):
        rank = 1 if eval_slug in ("multihop", "typo", "order-ops") else 50
        rows.append((eval_slug, "A", "i", "x", 8, rank))
    # Lens "B": poetry missing.
    for eval_slug in ("multihop", "multilingual", "typo", "association"):
        rows.append((eval_slug, "B", "i", "x", 8, 1))
    table = pair_pass_at_k(_ranks_df(rows), ks=(10,)).set_index(["lens", "eval"])["pass_at_k"]
    assert table[("A", "macro")] == pytest.approx(2 / 5)
    assert table[("A", "order-ops")] == pytest.approx(1.0)
    assert pd.isna(table[("B", "macro")])
    assert table[("B", "multihop")] == pytest.approx(1.0)


### BOTH WEIGHTINGS IN ONE TABLE


def test_pass_at_k_by_weighting_stacks_the_two_tables() -> None:
    ranks_df = _synthetic_ranks_df(seed=1)
    ks = (1, 10)
    table = pass_at_k_by_weighting(ranks_df, ks)
    assert list(table.columns) == ["eval", "lens", "k", "weighting", "pass_at_k"]
    # One row per (eval incl. macro, lens, k, weighting): 6 x 2 x 2 x 2.
    assert len(table) == 6 * 2 * 2 * 2
    assert not table.duplicated(["eval", "lens", "k", "weighting"]).any()
    for weighting, expected_df in (
        ("item", pass_at_k_over_layers(ranks_df, ks)),
        ("pair", pair_pass_at_k(ranks_df, ks)),
    ):
        rows_df = table[table["weighting"] == weighting].drop(columns="weighting")
        pd.testing.assert_frame_equal(rows_df.reset_index(drop=True), expected_df)
