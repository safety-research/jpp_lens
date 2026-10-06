"""The eval items themselves: the prompt files in ``data/jlens/evaluations``,
their per-eval settings, the correctness filter, the default item list, the
candidate vocab ids of an item's intermediates, the Readout Filtering exclusion
mask and the split into fitting and held-out items.

Everything here reads prompt JSONs, the correctness CSV and the tokenizer only:
no lens, no forward pass. The runner that reads lenses out on these items is
:mod:`lens_evals.readout_evals.readout_evals`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

from lens_evals.eval_utils import (
    candidate_token_ids,
    non_semantic_token_ids_sparing,
)

logger = logging.getLogger(__name__)

READOUT_EVALS_DATA_DIR = "data/jlens/evaluations"
MODEL_CORRECTNESS_CSV = f"{READOUT_EVALS_DATA_DIR}/model_correctness.csv"
# The share of each eval's items the expert weights are fitted on; the rest are held out
# for scoring.
DEFAULT_FIT_FRACTION = 0.2
# The default model: the only column of the shipped correctness CSV.
RECIPE_HF_MODEL_NAME = "Qwen/Qwen3.6-27B"

# The evals the macro averages over. order-ops stays in READOUT_EVAL_SPECS
# and the data dir but out of the default macro (the digit-splitting
# tokenizer makes its scoring unreliable).
MACRO_EVALS = ("multihop", "multilingual", "typo", "association", "poetry")


@dataclass(frozen=True)
class ReadoutEvalSpec:
    """Per-eval settings that the prompt JSONs deliberately do not carry."""

    # Where the lens is read out: the final prompt token, or the last newline
    # token (poetry: the end of the couplet's first line).
    readout_rule: Literal["final_token", "last_newline"]
    # Whether intermediates are order-ops keys expanded to synonym sets.
    expand_order_ops_synonyms: bool = False
    # Where the greedy-correctness target comes from; None: not gradeable.
    target_source: Optional[Literal["target", "first_intermediate"]] = None  # noqa: UP045


READOUT_EVAL_SPECS: dict[str, ReadoutEvalSpec] = {
    "multihop": ReadoutEvalSpec("final_token", target_source="target"),
    "multilingual": ReadoutEvalSpec("final_token", target_source="target"),
    "order-ops": ReadoutEvalSpec(
        "final_token",
        expand_order_ops_synonyms=True,
        target_source="target",
    ),
    "poetry": ReadoutEvalSpec("last_newline", target_source="first_intermediate"),
    "association": ReadoutEvalSpec("final_token"),
    "typo": ReadoutEvalSpec("final_token"),
}


@dataclass(frozen=True)
class ReadoutEvalItem:
    eval_slug: str
    name: str
    prompt: str
    target: Optional[str]  # noqa: UP045
    intermediates: tuple[str, ...]

    def correctness_target(self) -> Optional[str]:  # noqa: UP045
        """The string the greedy next token is graded against; ``None`` when
        the eval poses no question (association, typo)."""
        target_source = READOUT_EVAL_SPECS[self.eval_slug].target_source
        if target_source == "target":
            return self.target
        if target_source == "first_intermediate":
            return self.intermediates[0]
        return None


def load_readout_eval_items(
    data_dir: str = READOUT_EVALS_DATA_DIR,
    slugs: Sequence[str] | None = None,
) -> list[ReadoutEvalItem]:
    """Load ``lens-eval-{slug}.json`` items for every requested eval."""
    items: list[ReadoutEvalItem] = []
    for slug in slugs if slugs is not None else READOUT_EVAL_SPECS:
        raw_items = json.loads((Path(data_dir) / f"lens-eval-{slug}.json").read_text())[
            "items"
        ]
        for raw_item in raw_items:
            items.append(
                ReadoutEvalItem(
                    eval_slug=slug,
                    name=raw_item["name"],
                    prompt=raw_item["prompt"],
                    target=raw_item.get("target"),
                    intermediates=tuple(raw_item["intermediates"]),
                )
            )
    return items


def filter_items_by_correctness(
    items: Sequence[ReadoutEvalItem],
    *,
    hf_model_name: str,
    correctness_csv: str = MODEL_CORRECTNESS_CSV,
) -> list[ReadoutEvalItem]:
    """Items the model answered correctly, per the grading CSV written by
    :meth:`~lens_evals.readout_evals.readout_evals.ReadoutEvalRunner.grade_model_correctness`;
    ungradeable evals (``NA`` cells — association, typo) are always kept. Items
    with no row in the CSV are dropped with a warning (they were never graded).
    Reads only the CSV, never the model.

    Raises:
        ValueError: If the CSV has no column for ``hf_model_name``.
    """
    correctness_df = pd.read_csv(correctness_csv)
    if hf_model_name not in correctness_df.columns:
        raise ValueError(f"{correctness_csv} has no column {hf_model_name!r}")

    correct_or_ungradeable_item_keys = {
        (row["eval"], row["item"])
        for _, row in correctness_df.iterrows()
        if pd.isna(row[hf_model_name]) or row[hf_model_name] == True  # noqa: E712
    }
    graded_item_keys = {
        (row["eval"], row["item"]) for _, row in correctness_df.iterrows()
    }
    num_ungraded = sum(
        1 for item in items if (item.eval_slug, item.name) not in graded_item_keys
    )
    if num_ungraded:
        logger.warning(
            "dropping %d items with no correctness row in %s",
            num_ungraded,
            correctness_csv,
        )
    filtered_items = [
        item
        for item in items
        if (item.eval_slug, item.name) in correct_or_ungradeable_item_keys
    ]
    logger.info("correctness filter kept %d/%d items", len(filtered_items), len(items))
    return filtered_items


def load_recipe_items(
    *,
    data_dir: str = READOUT_EVALS_DATA_DIR,
    correctness_csv: str = MODEL_CORRECTNESS_CSV,
    hf_model_name: str = RECIPE_HF_MODEL_NAME,
) -> list[ReadoutEvalItem]:
    """The default item list: the five :data:`MACRO_EVALS` (in that order),
    correctness-filtered with the recorded CSV."""
    return filter_items_by_correctness(
        load_readout_eval_items(data_dir, slugs=MACRO_EVALS),
        hf_model_name=hf_model_name,
        correctness_csv=correctness_csv,
    )


def item_candidate_token_ids(tokenizer: Any, item: ReadoutEvalItem) -> set[int]:
    """Every candidate vocab id for an item's intermediates, under the eval's
    synonym-expansion setting. Callers building ``excluded_token_ids`` masks
    must spare these ids: an excluded candidate can never be ranked, so
    excluding one silently penalises every lens that promotes it (e.g. the
    order-ops symbol synonyms under a non-semantic-token filter)."""
    spec = READOUT_EVAL_SPECS[item.eval_slug]
    return {
        token_id
        for intermediate in item.intermediates
        for token_id in candidate_token_ids(
            tokenizer,
            intermediate,
            expand_order_ops_synonyms=spec.expand_order_ops_synonyms,
        )
    }


def non_semantic_token_ids(
    tokenizer: Any,
    *,
    vocab_size: int,
    items: Sequence[ReadoutEvalItem] | None = None,
) -> list[int]:
    """The Readout Filtering exclusion mask: every id in ``range(vocab_size)`` whose
    decoded string has no letter or digit
    (:func:`~lens_evals.eval_utils.is_non_semantic_token`; whitespace,
    punctuation, and LM-head padding rows that decode to ``""``), minus the
    candidate ids of ``items`` so no intermediate becomes unrankable.
    Special ids are not excluded (they decode word-like). ``vocab_size`` is
    the LM-head width, not ``len(tokenizer)``. ``items`` defaults to all six
    evals' unfiltered items."""
    if items is None:
        items = load_readout_eval_items()
    candidate_ids: set[int] = set().union(
        *(item_candidate_token_ids(tokenizer, item) for item in items)
    )
    return non_semantic_token_ids_sparing(
        tokenizer, vocab_size, candidate_ids, context="non-semantic exclusion"
    )


def split_fitting_items(
    items: Sequence[ReadoutEvalItem], *, seed: int, fit_fraction: float = DEFAULT_FIT_FRACTION
) -> tuple[list[ReadoutEvalItem], list[ReadoutEvalItem]]:
    """Split the items into a fitting split (for the expert weights) and a held-out split
    (for scoring), stratified by eval: one ``numpy.random.default_rng(seed)`` shared across the
    evals in sorted slug order; within each eval of ``n`` items the items are permuted and the
    first ``max(1, floor(fit_fraction * n))`` of the permutation go to the fitting split.
    Returns ``(fitting_items, held_out_items)``, each in the input order.

    An item's split depends on the whole list (every eval's item count advances the generator),
    so pass exactly the items the experiment uses: the scoreable items
    (:meth:`~lens_evals.readout_evals.readout_evals.ReadoutEvalRunner.scoreable_items`), not the raw
    :func:`load_recipe_items` output.
    """
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError(f"fit_fraction must be in (0, 1), got {fit_fraction}")
    item_keys = [(item.eval_slug, item.name) for item in items]
    if len(set(item_keys)) != len(item_keys):
        raise ValueError("items must have unique (eval, name) keys")

    items_by_eval: dict[str, list[ReadoutEvalItem]] = {}
    for item in items:
        items_by_eval.setdefault(item.eval_slug, []).append(item)

    rng = np.random.default_rng(seed)
    fitting_keys: set[tuple[str, str]] = set()
    for eval_slug in sorted(items_by_eval):
        eval_items = items_by_eval[eval_slug]
        shuffled_indices = rng.permutation(len(eval_items)).tolist()
        num_fitting = max(1, int(fit_fraction * len(eval_items)))
        for item_idx in shuffled_indices[:num_fitting]:
            fitting_keys.add((eval_items[item_idx].eval_slug, eval_items[item_idx].name))

    fitting_items = [item for item in items if (item.eval_slug, item.name) in fitting_keys]
    held_out_items = [item for item in items if (item.eval_slug, item.name) not in fitting_keys]
    return fitting_items, held_out_items
