"""Summaries over the rank rows of :class:`~lens_evals.readout_evals.readout_evals.ReadoutEvalRunner`.

Two pass@k (recall@k) weightings live here, both with a ``macro`` row over
:data:`~lens_evals.readout_evals.readout_eval_items.MACRO_EVALS`.
:func:`pass_at_k_over_layers` is item-weighted: mean over items of the
fraction of intermediates whose min-over-layers rank <= k.
:func:`pair_pass_at_k` is pair-weighted: the hit rate over (item,
intermediate) pairs. :func:`pass_at_k_by_weighting` stacks the two into one
table.
"""

from collections.abc import Sequence

import pandas as pd

from lens_evals.readout_evals.readout_eval_items import MACRO_EVALS

### SUMMARIES (pure functions over the ranks DataFrame)


def _item_mean_pass_at_k(
    ranks_df: pd.DataFrame, ks: Sequence[int], group_columns: list[str]
) -> pd.DataFrame:
    """pass@k averaged within each item first, then over an eval's items: for
    each k, ``hit = rank <= k`` is meaned per ``(*group_columns, item)`` and
    then per ``group_columns``. One row per ``(*group_columns, k)``."""
    frames: list[pd.DataFrame] = []
    for k in ks:
        per_item = (
            ranks_df.assign(hit=ranks_df["rank"] <= k)
            .groupby([*group_columns, "item"])["hit"]
            .mean()
        )
        per_group = per_item.groupby(group_columns).mean()
        frames.append(per_group.rename("pass_at_k").reset_index().assign(k=k))
    return pd.concat(frames, ignore_index=True)[[*group_columns, "k", "pass_at_k"]]


def pass_at_k_over_layers(ranks_df: pd.DataFrame, ks: Sequence[int]) -> pd.DataFrame:
    """The data README's metric: pass@k = mean over items of the fraction of
    intermediates whose min-over-layers rank <= k. One row per
    (eval, lens, k), plus ``eval == "macro"`` rows: per (lens, k), the
    unweighted mean over :data:`MACRO_EVALS`, NaN when any of the five is
    missing for that lens (the rule of :func:`pair_pass_at_k`). Differs from
    :func:`pair_pass_at_k` on items with several intermediates (this averages
    within the item first)."""
    min_rank_df = (
        ranks_df.groupby(["eval", "lens", "item", "intermediate"])["rank"].min().reset_index()
    )
    per_eval_df = _item_mean_pass_at_k(min_rank_df, ks, ["eval", "lens"])
    per_eval_wide = per_eval_df.set_index(["lens", "k", "eval"])["pass_at_k"].unstack("eval")
    macro = per_eval_wide.reindex(columns=list(MACRO_EVALS)).mean(axis=1, skipna=False)
    macro_df = macro.rename("pass_at_k").reset_index().assign(eval="macro")
    return pd.concat([per_eval_df, macro_df], ignore_index=True)[
        ["eval", "lens", "k", "pass_at_k"]
    ]


def pair_pass_at_k(ranks_df: pd.DataFrame, ks: Sequence[int]) -> pd.DataFrame:
    """Pair-weighted pass@k: the unit is an (item, intermediate) pair, a hit is
    a pair whose min-over-layers rank <= k, and pass@k is the hit rate over an
    eval's pairs. One row per (eval, lens, k), plus ``eval == "macro"`` rows: the
    unweighted mean over :data:`MACRO_EVALS`, NaN when any of the five is
    missing for that lens — never a silently narrower macro. Evals outside the
    macro (order-ops) get their own rows only."""
    group_columns = ["lens"]
    pair_ranks_df = ranks_df.groupby(
        ["eval", "lens", "item", "intermediate"], as_index=False
    )["rank"].min()

    frames: list[pd.DataFrame] = []
    for k in ks:
        hit_rate = (
            pair_ranks_df.assign(hit=pair_ranks_df["rank"] <= k)
            .groupby([*group_columns, "eval"])["hit"]
            .mean()
        )
        per_eval_wide = hit_rate.unstack("eval")
        macro = per_eval_wide.reindex(columns=list(MACRO_EVALS)).mean(axis=1, skipna=False)
        per_eval_rows = hit_rate.rename("pass_at_k").reset_index()
        macro_rows = macro.rename("pass_at_k").reset_index().assign(eval="macro")
        frame = pd.concat([per_eval_rows, macro_rows], ignore_index=True).assign(k=k)
        frames.append(frame[["eval", *group_columns, "k", "pass_at_k"]])
    return pd.concat(frames, ignore_index=True)


def pass_at_k_by_weighting(ranks_df: pd.DataFrame, ks: Sequence[int]) -> pd.DataFrame:
    """Both weightings in one tidy table: the rows of
    :func:`pass_at_k_over_layers` with ``weighting == "item"`` followed by the
    rows of :func:`pair_pass_at_k` with ``weighting == "pair"``. One row per
    (eval incl. ``macro``, lens, k, weighting)."""
    item_df = pass_at_k_over_layers(ranks_df, ks).assign(weighting="item")
    pair_df = pair_pass_at_k(ranks_df, ks).assign(weighting="pair")
    return pd.concat([item_df, pair_df], ignore_index=True)[
        ["eval", "lens", "k", "weighting", "pass_at_k"]
    ]
