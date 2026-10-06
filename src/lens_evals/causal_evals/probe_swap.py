"""The probe-swap eval: are unspoken intermediates causally used?

Two-hop factual prompts ("the language spoken in the country where the Amazon
River ends is ") activate an unspoken bridge entity (Brazil) in the workspace;
swapping Brazil->Mexico across the band should flip the answer
Portuguese->Spanish. Each JSON item is fully specified
(``intermediate``/``swap_to``/``answer``/``swap_answer``/``category``); the
readout is the final prompt token.

We emit three trial variants per item:

- ``main``: swap ``intermediate`` -> ``swap_to``; success = ``swap_answer``.
- ``answer_swap``: swap ``answer`` -> ``swap_answer`` directly. A positive-
  control *ceiling*: if even this does not reach top-1, the intervention is too
  weak on this model and the ``main`` result is uninterpretable.
- ``random_null``: swap ``intermediate`` -> an unrelated token; success is still
  scored against ``swap_answer`` and should stay ~0. Vectors are unit-normalised
  in the runner, so this is a matched-norm random-direction null.

Trailing-space prompts (29/90 end in a space, which blocks the leading-space
answer token under BPE) are kept as written; the loader does not strip them.

:func:`run_probe_swap` runs the ``main`` trials (and optionally the controls)
with the runner's clamp swap; :func:`probe_swap_success_table` and
:func:`best_scale_rows` score them.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import random
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from jlens.protocol import LensModel
from lens_evals.causal_evals.runner import SwapEvalRunner, SwapTrial
from lens_evals.eval_utils import formatting_token_ids_of, surface_forms
from workspace_lens.lenses.base_lens import BaseLens
from workspace_lens.lenses.jacobian_lens import JacobianLens
from workspace_lens.utils import vocab_size_of

logger = logging.getLogger(__name__)

PROBE_SWAP_PATH = "data/jlens/experiments/probe-swap.json"
PROBE_SWAP_SCALES: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)

SUCCESS_TABLE_COLUMNS: tuple[str, ...] = (
    "lens",
    "scale",
    "top1_hits",
    "top1_trials",
    "top1_rate",
    "top5_hits",
    "top5_trials",
    "top5_rate",
)

# Unrelated common nouns for the matched-norm random-direction null. Chosen to be
# semantically distant from the probe-swap concepts (countries, animals, ...); the
# runner unit-normalises so only the direction matters.
_RANDOM_NULL_POOL: tuple[str, ...] = (
    "table",
    "window",
    "pencil",
    "carpet",
    "engine",
    "ladder",
    "button",
    "pillow",
    "basket",
    "mirror",
)


def load_probe_swap_trials(
    data_path: str = PROBE_SWAP_PATH,
    *,
    include_controls: bool = True,
    random_seed: int = 0,
) -> list[SwapTrial]:
    """Build the probe-swap swap trials from the prompt JSON.

    Args:
        data_path: Path to ``probe-swap.json``.
        include_controls: Also emit the ``answer_swap`` ceiling and
            ``random_null`` control trials.
        random_seed: Seeds the deterministic choice of random-null targets.
    """
    items = json.loads(Path(data_path).read_text())["items"]
    rng = random.Random(random_seed)
    trials: list[SwapTrial] = []
    for item in items:
        name = item["name"]
        prompt = item["prompt"]
        category = item["category"]
        answer_surfaces = tuple(surface_forms(item["answer"]))
        swap_answer_surfaces = tuple(surface_forms(item["swap_answer"]))

        trials.append(
            SwapTrial(
                eval_slug="probe-swap",
                name=name,
                prompt=prompt,
                source=item["intermediate"],
                target=item["swap_to"],
                success_surfaces=swap_answer_surfaces,
                baseline_surfaces=answer_surfaces,
                category=category,
                variant="main",
            )
        )
        if not include_controls:
            continue

        trials.append(
            SwapTrial(
                eval_slug="probe-swap",
                name=name,
                prompt=prompt,
                source=item["answer"],
                target=item["swap_answer"],
                success_surfaces=swap_answer_surfaces,
                baseline_surfaces=answer_surfaces,
                category=category,
                variant="answer_swap",
            )
        )
        random_target = rng.choice(_RANDOM_NULL_POOL)
        trials.append(
            SwapTrial(
                eval_slug="probe-swap",
                name=name,
                prompt=prompt,
                source=item["intermediate"],
                target=random_target,
                success_surfaces=swap_answer_surfaces,
                baseline_surfaces=answer_surfaces,
                category=category,
                variant="random_null",
            )
        )
    return trials


### RUNNING THE PROBE SWAP


def probe_swap_formatting_token_ids(model: LensModel) -> list[int]:
    """The model's formatting-token ids (non-semantic, special and added
    control tokens), removed from its next-token distribution for the swap
    runner's word ranks."""
    formatting_token_ids = formatting_token_ids_of(model.tokenizer, vocab_size_of(model))
    logger.info("formatting tokens: %d ids", len(formatting_token_ids))
    return formatting_token_ids


def slice_lens_to_layers(lens: JacobianLens, layers: Sequence[int]) -> JacobianLens:
    """A JacobianLens restricted to ``layers`` (saves memory compared with
    holding every fitted layer). Raises ``ValueError`` if the lens lacks any of
    ``layers``."""
    assert lens.jacobians_L_dict_FN is not None
    missing = sorted(set(layers) - set(lens.jacobians_L_dict_FN))
    if missing:
        raise ValueError(
            f"lens {lens.config.checkpoint_name!r} lacks layers {missing}; "
            f"fitted layers are {sorted(lens.jacobians_L_dict_FN)}"
        )
    return JacobianLens(
        jacobians={layer: lens.jacobians_L_dict_FN[layer] for layer in layers},
        config=dataclasses.replace(lens.config, source_layers=list(layers)),
    )


def run_probe_swap(
    model: LensModel,
    lenses: dict[str, BaseLens],
    *,
    layers: Sequence[int],
    scales: Sequence[float] = PROBE_SWAP_SCALES,
    max_seq_len: int = 512,
    include_controls: bool = False,
) -> tuple[pd.DataFrame, list[tuple[str, str, str]]]:
    """Run the probe-swap trials through :class:`SwapEvalRunner`.

    Every JacobianLens is sliced to ``layers`` (other lenses, e.g. the logit
    lens, are passed as given). The clamp swaps are applied at ``layers`` on
    every position but the attention sink, along unit-normalised lens vectors,
    and the word ranks drop :func:`probe_swap_formatting_token_ids`.

    Returns ``(trials_df, dropped_concepts)``: the runner's rows (one per
    trial, lens and scale; see :meth:`SwapEvalRunner.run`) and the
    ``(eval, item, concept)`` triples it skipped for having no single-token
    surface.
    """
    trials = load_probe_swap_trials(include_controls=include_controls)
    sliced_lenses: dict[str, BaseLens] = {
        name: slice_lens_to_layers(lens, layers) if isinstance(lens, JacobianLens) else lens
        for name, lens in lenses.items()
    }
    runner = SwapEvalRunner(
        model,
        sliced_lenses,
        band_layers=layers,
        scales=scales,
        max_seq_len=max_seq_len,
        formatting_token_ids=probe_swap_formatting_token_ids(model),
    )
    trials_df = runner.run(trials)
    return trials_df, list(runner.dropped_concepts)


### SCORING


def _rate(hits: int, trials: int) -> float:
    return hits / trials if trials else float("nan")


def probe_swap_success_table(trials_df: pd.DataFrame) -> pd.DataFrame:
    """Top-1 and top-5 successes per (lens, scale) over the ``main`` trials,
    on word ranks.

    A trial is top-1 eligible when the clean pass answers correctly (clean
    baseline word rank 1); it succeeds when the edited pass ranks the swapped
    answer first (success word rank 1). A trial is top-5 eligible when it is
    top-1 eligible and the swapped answer is not already in the clean top 5
    (clean success word rank > 5); it succeeds at success word rank <= 5.

    Columns: ``lens``, ``scale``, then ``top1_hits``, ``top1_trials``,
    ``top1_rate`` and the same for top 5 (a rate is nan when no trial is
    eligible).
    """
    main_trials_df = trials_df[trials_df["variant"] == "main"]
    rows: list[dict[str, object]] = []
    for (lens, scale), at_scale_df in main_trials_df.groupby(["lens", "scale"]):
        top1_eligible_df = at_scale_df[at_scale_df["clean_baseline_word_rank"] == 1]
        top5_eligible_df = top1_eligible_df[top1_eligible_df["clean_success_word_rank"] > 5]
        top1_hits = int((top1_eligible_df["success_word_rank"] <= 1).sum())
        top5_hits = int((top5_eligible_df["success_word_rank"] <= 5).sum())
        rows.append(
            {
                "lens": lens,
                "scale": scale,
                "top1_hits": top1_hits,
                "top1_trials": len(top1_eligible_df),
                "top1_rate": _rate(top1_hits, len(top1_eligible_df)),
                "top5_hits": top5_hits,
                "top5_trials": len(top5_eligible_df),
                "top5_rate": _rate(top5_hits, len(top5_eligible_df)),
            }
        )
    return pd.DataFrame(rows, columns=list(SUCCESS_TABLE_COLUMNS))


def best_scale_rows(success_table: pd.DataFrame) -> pd.DataFrame:
    """One row of ``success_table`` per lens: the scale with the highest top-1
    rate (ties go to the smaller scale), so the top-5 columns are reported at
    the best top-1 scale."""
    ranked_df = success_table.sort_values(
        ["lens", "top1_rate", "scale"], ascending=[True, False, True]
    )
    return ranked_df.drop_duplicates("lens", keep="first").reset_index(drop=True)
