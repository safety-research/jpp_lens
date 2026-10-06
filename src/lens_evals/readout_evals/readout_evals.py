"""The runner that reads lenses out on the prompt-distribution lens evals in
``data/jlens/evaluations``.

The items themselves — the prompt files, the per-eval specs, the correctness
filter, the default item list, the exclusion mask and the held-out split —
live in :mod:`lens_evals.readout_evals.readout_eval_items`; import them from there.

Each eval item is a prompt in which some latent concept (the ``intermediates``)
should be present in the workspace before it is ever emitted. For every item,
the model runs once (a shared forward pass whose recorded residuals every lens
transports), each lens is read out at one position across layers, and each
intermediate's lens rank is the min over its single-token candidate surfaces
(see :mod:`lens_evals.eval_utils`). Lens logits are ranked raw; the only
scoring knob is ``excluded_token_ids`` (tokens forced to ``-inf``).

:meth:`ReadoutEvalRunner.grade_model_correctness` implements the correctness
filter: one forward pass per item, correct iff the greedy next token is the
target or a leading subword of it. Only evals with a defined target are
gradeable (multihop, multilingual, order-ops via ``target``; poetry via its
rhyme-word intermediate); association and typo prompts pose no question, so
they grade as ``NA`` and are always evaluated.

The runner's other two item-level methods are
:meth:`ReadoutEvalRunner.scoreable_items` (the items with a single-token candidate)
and :meth:`ReadoutEvalRunner.record_residuals` (the labelled-item residuals the
expert-weight fit consumes, recorded at the same position and with the same
candidates as :meth:`ReadoutEvalRunner.run`).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch as t

from jlens.protocol import LensModel
from lens_evals.eval_utils import (
    candidate_token_ids,
    check_candidates_not_all_excluded,
    next_token_matches_target,
)
from lens_evals.readout_evals.readout_eval_items import (
    READOUT_EVAL_SPECS,
    ReadoutEvalItem,
)
from workspace_lens.lenses.base_lens import BaseLens
from workspace_lens.utils import (
    check_layers_fitted,
    ensure_parent_dir,
    final_position_model_logits,
    hf_model_name_of,
    masked_lens_logits,
    model_device_of,
    record_activations,
    token_id_mask_Bool_V,
    vocab_size_of,
)

logger = logging.getLogger(__name__)


def readout_position(input_ids_Int_1S: t.Tensor, tokenizer: Any, rule: str) -> int:
    """The position the lens is read out at, under an eval spec's rule:
    ``final_token``, the last prompt token; ``last_newline``, the last token
    whose decoded text contains a newline (poetry: end of the couplet's first
    line)."""
    seq_len = input_ids_Int_1S.shape[1]
    if rule == "final_token":
        return seq_len - 1
    if rule == "last_newline":
        newline_positions = [
            position
            for position in range(seq_len)
            if "\n" in tokenizer.decode([int(input_ids_Int_1S[0, position])])
        ]
        if not newline_positions:
            raise ValueError("last_newline readout: prompt has no newline token")
        return newline_positions[-1]
    raise ValueError(f"unknown readout rule {rule!r}")


def min_rank_over_candidates(logits_V: t.Tensor, candidate_ids: Sequence[int]) -> int:
    """The best (lowest) rank any candidate token achieves under ``logits_V``.

    Rank = 1 + number of strictly larger logits (ties resolve in the lens's
    favour); the min over candidates is the rank of the highest-logit candidate.
    """
    best_candidate_logit = logits_V[list(candidate_ids)].max()
    return int((logits_V > best_candidate_logit).sum().item()) + 1


### READOUT POSITION AND CANDIDATES (the runner's, shared by all its methods)


@dataclass(frozen=True)
class _ItemReadout:
    """Where one item is read out and which vocab ids score each of its
    intermediates."""

    input_ids_Int_1S: t.Tensor
    readout_position: int
    # Intermediates with no single-token candidate surface are absent here
    # and listed in dropped_intermediates.
    candidate_ids_by_intermediate: dict[str, list[int]]
    dropped_intermediates: tuple[str, ...]


def _item_readout_position(
    input_ids_Int_1S: t.Tensor, tokenizer: Any, item: ReadoutEvalItem
) -> int:
    """:func:`readout_position` under the item's eval spec rule."""
    return readout_position(
        input_ids_Int_1S, tokenizer, READOUT_EVAL_SPECS[item.eval_slug].readout_rule
    )


def _item_readout_and_candidates(
    model: LensModel, item: ReadoutEvalItem, max_seq_len: int
) -> _ItemReadout | None:
    """Tokenize the item (no forward pass), resolve its readout position, and
    collect each intermediate's candidate ids under the eval's synonym
    setting. Intermediates with no single-token surface are dropped with a
    warning; when none is left the item is skipped (``None``) with a warning.
    :meth:`ReadoutEvalRunner.run` and :meth:`ReadoutEvalRunner.record_residuals` both
    go through here, so they agree on positions and candidates by construction.
    """
    spec = READOUT_EVAL_SPECS[item.eval_slug]
    candidate_ids_by_intermediate: dict[str, list[int]] = {}
    dropped_intermediates: list[str] = []
    for intermediate in item.intermediates:
        candidate_ids = candidate_token_ids(
            model.tokenizer,
            intermediate,
            expand_order_ops_synonyms=spec.expand_order_ops_synonyms,
        )
        if not candidate_ids:
            dropped_intermediates.append(intermediate)
            logger.warning(
                "dropping %s/%s intermediate %r: no single-token surface",
                item.eval_slug,
                item.name,
                intermediate,
            )
            continue
        candidate_ids_by_intermediate[intermediate] = candidate_ids
    if not candidate_ids_by_intermediate:
        logger.warning("skipping %s/%s: no scoreable intermediates", item.eval_slug, item.name)
        return None

    input_ids_Int_1S = model.encode(item.prompt, max_length=max_seq_len)
    return _ItemReadout(
        input_ids_Int_1S=input_ids_Int_1S,
        readout_position=_item_readout_position(input_ids_Int_1S, model.tokenizer, item),
        candidate_ids_by_intermediate=candidate_ids_by_intermediate,
        dropped_intermediates=tuple(dropped_intermediates),
    )


### READOUT RESIDUALS (the labelled-item inputs of the expert-weight fit)


def _cache_header(
    hf_model_name: str,
    layers: Sequence[int],
    max_seq_len: int,
    item_keys: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    """What a residuals cache is keyed on: model name, sorted layers, the eval
    prompt truncation and the kept items' ``(eval, name)`` keys (as lists, for
    the plain-types file)."""
    return {
        "hf_model_name": hf_model_name,
        "layers": sorted(layers),
        "max_seq_len": max_seq_len,
        "item_keys": [[eval_slug, name] for eval_slug, name in item_keys],
    }


@dataclass
class ReadoutResiduals:
    """The residual stream of labelled eval items at their readout position,
    at each recorded layer: ``residuals_L_dict_PN[layer]`` is fp32 CPU
    ``[num_items, d_model]``, row ``p`` belonging to item
    ``(eval_slugs[p], item_names[p])`` whose scoreable intermediates map to
    candidate vocab ids in ``candidate_ids_P_list[p]``. Items with no
    scoreable intermediate are absent
    (:meth:`ReadoutEvalRunner.scoreable_items`).
    ``save``/``load`` round-trip through ``t.save`` with plain types; the
    header (:func:`_cache_header`) is what
    :meth:`ReadoutEvalRunner.record_residuals` checks a cache against."""

    eval_slugs: list[str]
    item_names: list[str]
    residuals_L_dict_PN: dict[int, t.Tensor]
    candidate_ids_P_list: list[dict[str, list[int]]]
    hf_model_name: str
    max_seq_len: int

    @property
    def layers(self) -> list[int]:
        return sorted(self.residuals_L_dict_PN)

    @property
    def num_items(self) -> int:
        return len(self.item_names)

    @property
    def item_keys(self) -> list[tuple[str, str]]:
        return list(zip(self.eval_slugs, self.item_names, strict=True))

    def cache_header(self) -> dict[str, Any]:
        return _cache_header(self.hf_model_name, self.layers, self.max_seq_len, self.item_keys)

    def save(self, path: str | Path) -> None:
        ensure_parent_dir(path)
        t.save(
            {
                "header": self.cache_header(),
                "residuals_L_dict_PN": {
                    layer: residuals_PN.float().cpu()
                    for layer, residuals_PN in self.residuals_L_dict_PN.items()
                },
                "candidate_ids_by_item": self.candidate_ids_P_list,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> ReadoutResiduals:
        payload = t.load(path, map_location="cpu", weights_only=True)
        header = payload["header"]
        return cls(
            eval_slugs=[eval_slug for eval_slug, _ in header["item_keys"]],
            item_names=[name for _, name in header["item_keys"]],
            residuals_L_dict_PN={
                int(layer): residuals_PN
                for layer, residuals_PN in payload["residuals_L_dict_PN"].items()
            },
            candidate_ids_P_list=payload["candidate_ids_by_item"],
            hf_model_name=header["hf_model_name"],
            max_seq_len=int(header["max_seq_len"]),
        )


### RUNNER


class ReadoutEvalRunner:
    """Runs the lens evals for several lenses under identical conditions.

    Args:
        model: The model to read out from.
        lenses: ``{name: lens}``; one forward pass per item records the
            residuals, which every lens then transports (see :meth:`run`).
            May be empty for the item-level methods that read out no lens
            (:meth:`scoreable_items`, :meth:`record_residuals`), and then
            ``layers`` is required.
        layers: Layers to read out at; ``None`` uses the first lens's
            ``source_layers``. Must be a subset of every lens's fitted layers.
        max_seq_len: Prompt truncation for both grading and readout.
        excluded_token_ids: Token ids removed from every lens ranking: their
            scores are forced to ``-inf`` before any rank is
            computed — an excluded token can never outrank anything
            (:func:`~lens_evals.readout_evals.readout_eval_items.non_semantic_token_ids` is
            the Readout Filtering list). :meth:`run` raises ``ValueError`` if all of an
            intermediate's candidate ids are excluded, rather than silently
            mis-ranking it. Model correctness grading is never filtered.
    """

    def __init__(
        self,
        model: LensModel,
        lenses: dict[str, BaseLens],
        *,
        layers: Sequence[int] | None = None,
        max_seq_len: int = 512,
        excluded_token_ids: Sequence[int] | None = None,
    ) -> None:
        self.model = model
        self.lenses = lenses
        self.max_seq_len = max_seq_len

        if layers is not None:
            self.layers = sorted(layers)
        elif lenses:
            self.layers = next(iter(lenses.values())).source_layers
        else:
            raise ValueError("layers is required when lenses is empty")
        for lens_name, lens in lenses.items():
            check_layers_fitted(lens, self.layers, lens_name=lens_name)
        for lens in lenses.values():
            lens.move_parameters_to_device(model_device_of(model))

        # Excluded tokens: a [vocab_size] boolean mask (moved to the logits'
        # device when applied), plus the id set for run()'s all-candidates-
        # excluded guard.
        self.excluded_token_ids: frozenset[int] = frozenset(
            int(token_id) for token_id in (excluded_token_ids or ())
        )
        self.excluded_mask_Bool_V: t.Tensor | None = None
        if self.excluded_token_ids:
            self.excluded_mask_Bool_V = token_id_mask_Bool_V(
                self.excluded_token_ids, vocab_size_of(model)
            )

        # (eval, item, intermediate) triples with no single-token candidate
        # surface, skipped by run(); flag these wherever results are reported.
        self.dropped_intermediates: list[tuple[str, str, str]] = []

    def _lens_logits(self, lens: BaseLens, residual_1N: t.Tensor, layer: int) -> t.Tensor:
        """The lens's logits at one readout, fp32 on CPU, with excluded tokens
        forced to ``-inf``."""
        return masked_lens_logits(
            self.model, lens, residual_1N, layer, self.excluded_mask_Bool_V
        )[0].cpu()

    def scoreable_items(self, items: Sequence[ReadoutEvalItem]) -> list[ReadoutEvalItem]:
        """``items`` minus those with no single-token candidate — exactly the
        items :meth:`record_residuals` keeps and :meth:`run` scores (all
        through :func:`_item_readout_and_candidates`; tokenizes only, no
        forward pass). Both sides of
        :func:`~lens_evals.readout_evals.readout_eval_items.split_fitting_items` must be
        taken over this list, because an item's side depends on the whole list."""
        return [
            item
            for item in items
            if _item_readout_and_candidates(self.model, item, self.max_seq_len) is not None
        ]

    @t.no_grad()
    def grade_model_correctness(
        self, items: Sequence[ReadoutEvalItem], *, hf_model_name: str
    ) -> pd.DataFrame:
        """One greedy step per item: is the next token the target (or a
        leading subword of it)?

        Returns one row per item: ``eval``, ``item``, and an ``hf_model_name``
        column holding True/False, or ``NA`` for ungradeable evals.
        """
        rows: list[dict] = []
        for item in items:
            target = item.correctness_target()
            if target is None:
                correct: Any = pd.NA
            else:
                logits_V = final_position_model_logits(self.model, item.prompt, self.max_seq_len)
                next_token = self.model.tokenizer.decode([int(logits_V.argmax())])
                correct = next_token_matches_target(next_token, target)
            rows.append({"eval": item.eval_slug, "item": item.name, hf_model_name: correct})
        return pd.DataFrame(rows)

    @t.no_grad()
    def run(self, items: Sequence[ReadoutEvalItem]) -> pd.DataFrame:
        """Per-layer lens ranks for every (lens, item, intermediate).

        Returns a long DataFrame: ``eval``, ``lens``, ``item``,
        ``intermediate``, ``layer``, ``rank``, ``n_candidates``. One forward
        pass per item records the residuals, which every lens then
        transports; rank = 1 + strictly-greater count over the
        exclusion-masked logits. Intermediates with no single-token
        candidate surface are skipped and recorded in
        :attr:`dropped_intermediates`.
        """
        # Columnar accumulation: the output has one row per (lens, item,
        # intermediate, layer), and a list of per-row dicts is severalfold
        # larger in memory.
        column_values: dict[str, list] = {
            column: []
            for column in (
                "eval",
                "lens",
                "item",
                "intermediate",
                "layer",
                "rank",
                "n_candidates",
            )
        }
        for item_idx, item in enumerate(items):
            item_readout = _item_readout_and_candidates(self.model, item, self.max_seq_len)
            item_dropped_intermediates = (
                item.intermediates
                if item_readout is None
                else item_readout.dropped_intermediates
            )
            self.dropped_intermediates.extend(
                (item.eval_slug, item.name, intermediate)
                for intermediate in item_dropped_intermediates
            )
            if item_readout is None:
                continue
            candidate_ids_by_intermediate = item_readout.candidate_ids_by_intermediate
            check_candidates_not_all_excluded(
                item.eval_slug, item.name, candidate_ids_by_intermediate, self.excluded_token_ids
            )

            # One forward pass per item; every lens transports the same
            # recorded residuals (the model logits a per-lens apply() would
            # also produce are unused here).
            _, activations_L_dict_SN = record_activations(
                self.model, item.prompt, self.max_seq_len, list(self.layers)
            )
            position = item_readout.readout_position

            for lens_name, lens in self.lenses.items():
                for layer in self.layers:
                    residual_1N = activations_L_dict_SN[layer][[position]].float()
                    logits_V = self._lens_logits(lens, residual_1N, layer)
                    for (
                        intermediate,
                        candidate_ids,
                    ) in candidate_ids_by_intermediate.items():
                        column_values["eval"].append(item.eval_slug)
                        column_values["lens"].append(lens_name)
                        column_values["item"].append(item.name)
                        column_values["intermediate"].append(intermediate)
                        column_values["layer"].append(layer)
                        column_values["rank"].append(
                            min_rank_over_candidates(logits_V, candidate_ids)
                        )
                        column_values["n_candidates"].append(len(candidate_ids))
            if (item_idx + 1) % 25 == 0 or item_idx + 1 == len(items):
                logger.info("scored item %d/%d", item_idx + 1, len(items))
        return pd.DataFrame(column_values)

    @t.no_grad()
    def record_residuals(
        self,
        items: Sequence[ReadoutEvalItem],
        *,
        cache_path: str | Path | None = None,
        hf_model_name: str | None = None,
    ) -> ReadoutResiduals:
        """One forward pass per item, recording the residual at the item's
        readout position (:func:`_item_readout_and_candidates`, shared with
        :meth:`run`) at every layer of :attr:`layers`. Items with no scoreable
        intermediate are skipped (:meth:`scoreable_items`). When ``cache_path``
        exists it is loaded instead and its header must match (model name,
        layers, ``max_seq_len`` and the kept items' keys); otherwise the
        recording is saved there. ``hf_model_name`` defaults to the wrapped HF
        model's name and must be given for models without one (test models).
        """
        if hf_model_name is None:
            hf_model_name = hf_model_name_of(self.model)
            if hf_model_name is None:
                raise ValueError("model has no HF name; pass hf_model_name explicitly")
        layers = self.layers

        kept_items = self.scoreable_items(items)
        expected_header = _cache_header(
            hf_model_name,
            layers,
            self.max_seq_len,
            [(item.eval_slug, item.name) for item in kept_items],
        )

        if cache_path is not None and Path(cache_path).exists():
            cached = ReadoutResiduals.load(cache_path)
            if cached.cache_header() != expected_header:
                raise ValueError(
                    f"{cache_path} was recorded for another model, layer set, "
                    f"max_seq_len or item list (cached model "
                    f"{cached.hf_model_name!r}, layers {cached.layers}, "
                    f"max_seq_len {cached.max_seq_len}, {cached.num_items} items; "
                    f"requested {hf_model_name!r}, {layers}, {self.max_seq_len}, "
                    f"{len(kept_items)} items); delete it to re-record"
                )
            logger.info("loaded %d readout residuals from %s", cached.num_items, cache_path)
            return cached

        residual_rows_L_dict_list_N: dict[int, list[t.Tensor]] = {
            layer: [] for layer in layers
        }
        candidate_ids_P_list: list[dict[str, list[int]]] = []
        for item_idx, item in enumerate(kept_items):
            item_readout = _item_readout_and_candidates(
                self.model, item, self.max_seq_len
            )
            assert item_readout is not None  # scoreable_items kept it
            candidate_ids_P_list.append(item_readout.candidate_ids_by_intermediate)
            _, activations_L_dict_SN = record_activations(
                self.model, item.prompt, self.max_seq_len, layers
            )
            for layer in layers:
                # .float() is the runner's cast; .clone() so the result holds
                # only the [N] vector, not a view into the [S, N] recording.
                residual_rows_L_dict_list_N[layer].append(
                    activations_L_dict_SN[layer][item_readout.readout_position]
                    .float()
                    .cpu()
                    .clone()
                )
            if (item_idx + 1) % 25 == 0 or item_idx + 1 == len(kept_items):
                logger.info("recorded item %d/%d", item_idx + 1, len(kept_items))

        residuals = ReadoutResiduals(
            eval_slugs=[item.eval_slug for item in kept_items],
            item_names=[item.name for item in kept_items],
            residuals_L_dict_PN={
                layer: t.stack(rows_list_N)
                for layer, rows_list_N in residual_rows_L_dict_list_N.items()
            },
            candidate_ids_P_list=candidate_ids_P_list,
            hf_model_name=hf_model_name,
            max_seq_len=self.max_seq_len,
        )
        if cache_path is not None:
            residuals.save(cache_path)
            logger.info("saved %d readout residuals to %s", residuals.num_items, cache_path)
        return residuals
