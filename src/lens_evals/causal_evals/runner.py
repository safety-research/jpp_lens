"""The causal swap-eval runner.

Each trial exchanges the residual-stream coordinate of one concept for another
along the lens's per-token directions during a single forward pass, then scores
the *model's own* next-token distribution. This is the causal complement to
:mod:`lens_evals.readout_evals.readout_evals`, which only reads concepts out
observationally: a lens can score well there with directions the model never
uses, but a swap only moves the output if the model actually computes through
the swapped direction.

The runner is a mechanism: a caller (the probe-swap loader,
:func:`~lens_evals.causal_evals.probe_swap.load_probe_swap_trials`) builds
:class:`SwapTrial` specs and the runner executes them identically for every
lens and scale. Design points:

- **Any lens, including the logit lens.** Swap vectors come from
  :func:`workspace_lens.utils.readout_vectors`, which returns
  ``W_U[token] @ J_l`` for a Jacobian lens and the raw unembedding row
  ``W_U[token]`` for the logit lens (the natural null basis).
- **Equal-magnitude edits.** Source and target vectors are unit-normalised per
  layer before any edit, so a lens does not win merely because its raw vectors
  are larger.
- **Per-layer edits.** One intervention per band layer, so a near-parallel
  layer is dropped individually (recorded) instead of killing the whole trial.
- **Clean pass cached per prompt**, shared across lenses and scales.

The edit is a clamp swap at the band layers on every prompt position but the
attention sink (position 0), with the unit vectors ``u_s, u_t``: a
:class:`~workspace_lens.interventions.interventions.Clamp` on the two-atom
basis ``[u_s, u_t]`` pins the coordinates at every (layer, position) to the
*clean pass's* values with source and target exchanged (times the scale).

Two rank conventions are recorded for every scored surface set: the rank in
the full vocabulary, and the *word rank* -- the rank after the model's
formatting tokens (whitespace, punctuation, special tokens;
:func:`~lens_evals.eval_utils.formatting_token_ids_of`) are removed
from the distribution. The model often emits ``"\\n\\n"`` or a ``<think>``
block before its answer, so without the word rank the rank-1 slot can be a
formatting token.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import pandas as pd
import torch as t

from jlens.hooks import ActivationRecorder
from jlens.protocol import LensModel
from lens_evals.eval_utils import (
    formatting_token_mask_Bool_V,
    resolve_swap_token,
    single_token_ids,
)
from lens_evals.readout_evals.readout_evals import min_rank_over_candidates
from workspace_lens.interventions.base import Intervention
from workspace_lens.interventions.hooks import InterventionHooks
from workspace_lens.interventions.interventions import (
    NEAR_PARALLEL_ABS_COSINE,
    Clamp,
    abs_cosine_between,
)
from workspace_lens.lenses.base_lens import BaseLens
from workspace_lens.residual_streams import (
    collapse_streams_like_the_model,
    residual_mean_over_streams,
)
from workspace_lens.utils import (
    check_layers_fitted,
    model_device_of,
    readout_vectors,
)

logger = logging.getLogger(__name__)


NUM_TOP_TOKENS = 5  # decoded top tokens kept per pass for qualitative examples


@dataclass(frozen=True)
class SwapTrial:
    """One swap to run: exchange ``source`` for ``target`` and score ``success``.

    ``source``/``target``/``baseline`` are concept strings resolved to single
    vocab ids by :func:`resolve_swap_token`. ``success_surfaces`` and
    ``baseline_surfaces`` are the surface-form sets whose min-rank / summed
    probability is scored at the readout position (a set so casing / leading
    space never penalise a hit). ``variant`` labels the trial: ``"main"``,
    ``"answer_swap"`` (the probe-swap ceiling control), ``"random_null"``, etc.
    """

    eval_slug: str
    name: str
    prompt: str
    source: str
    target: str
    success_surfaces: tuple[str, ...]
    baseline_surfaces: tuple[str, ...]
    category: str | None = None
    variant: str = "main"


@dataclass(frozen=True)
class SwapTrialResult:
    """Everything one edited forward yields; one row of the runner's output."""

    success_rank: int
    success_logprob: float
    baseline_rank: int
    baseline_logprob: float
    clean_success_rank: int
    clean_success_logprob: float
    clean_baseline_rank: int
    clean_baseline_logprob: float
    # Ranks with the model's formatting tokens removed from the distribution.
    success_word_rank: int
    baseline_word_rank: int
    clean_success_word_rank: int
    clean_baseline_word_rank: int
    edit_norm_ratio: float  # median over band layers of ||intervened - clean|| / ||clean||
    next_token: str
    swapped_layers: tuple[int, ...]
    skipped_layers: tuple[int, ...]
    n_swap_positions: int  # sequence positions actually edited
    # Single-token vocab ids actually scored for each surface set. A baseline set
    # of size 0 makes the baseline ranks -1 and the baseline logprobs nan.
    n_success_ids: int
    n_baseline_ids: int
    source_token_id: int
    target_token_id: int
    source_surface: str
    target_surface: str
    # Qualitative examples: JSON lists of the decoded top tokens, full vocab and
    # word tokens only, for the clean and the edited pass.
    clean_top_tokens: str
    swapped_top_tokens: str
    clean_top_word_tokens: str
    swapped_top_word_tokens: str


@dataclass
class _CleanPass:
    """The un-edited forward of one prompt, cached on the CPU in fp32 and shared
    by every (lens, scale): the final-position model logits and the band-layer
    residuals at every position (the clamp swap's targets; their last row, the
    final-token residual, feeds the edit-norm diagnostic)."""

    logits_V: t.Tensor
    band_residuals_L_dict_SN: dict[int, t.Tensor]  # the last position is the final-token residual


def _swapped_coordinate_clamp(
    atoms: t.Tensor,
    clean_residuals_PN: t.Tensor,
    *,
    scale: float,
    layer: int,
    positions: Sequence[int],
) -> Clamp:
    """The clamp swap's clamp: the clean pass's least-squares coordinates on
    ``atoms`` ``[N, 2]`` (columns 0 and 1 are the source and target
    directions), exchanged and multiplied by ``scale``."""
    pinv = t.linalg.pinv(atoms)  # [K, N]
    clean_coordinates_PK = clean_residuals_PN @ pinv.T
    targets_PK = clean_coordinates_PK.clone()
    targets_PK[:, 0] = scale * clean_coordinates_PK[:, 1]
    targets_PK[:, 1] = scale * clean_coordinates_PK[:, 0]
    return Clamp({layer: atoms}, {layer: targets_PK}, token_positions=positions)


def _summed_logprob(logits_V: t.Tensor, token_ids: Sequence[int]) -> float:
    """log of the total probability mass on ``token_ids`` under ``logits_V``."""
    log_probs_V = t.log_softmax(logits_V, dim=-1)
    return float(t.logsumexp(log_probs_V[list(token_ids)], dim=0))


class SwapEvalRunner:
    """Runs swap trials for several lenses under identical conditions.

    Args:
        model: The model to intervene on and read out from.
        lenses: ``{name: lens}``; every lens must be fitted at ``band_layers``.
        band_layers: The layers the edits are applied at. Must be a subset of
            every lens's ``source_layers``.
        scales: Edit strengths swept.
        max_seq_len: Prompt truncation.
        formatting_token_ids: Ids removed from the model's distribution for
            the word ranks; defaults to
            :func:`~lens_evals.eval_utils.formatting_token_ids_of`
            of the model's tokenizer (pass a precomputed list to share the
            vocab scan between runners).
    """

    def __init__(
        self,
        model: LensModel,
        lenses: dict[str, BaseLens],
        *,
        band_layers: Sequence[int],
        scales: Sequence[float] = (1.0, 2.0),
        max_seq_len: int = 512,
        formatting_token_ids: Sequence[int] | None = None,
    ) -> None:
        self.model = model
        self.lenses = lenses
        self.band_layers = sorted(band_layers)
        self.scales = tuple(scales)
        self.max_seq_len = max_seq_len

        for lens_name, lens in lenses.items():
            check_layers_fitted(lens, self.band_layers, lens_name=lens_name)

        self.device = model_device_of(model)
        for lens in lenses.values():
            lens.move_parameters_to_device(self.device)

        self.final_layer = model.n_layers - 1

        # Word ranks: the model's formatting tokens masked to -inf.
        self._formatting_mask_Bool_V = formatting_token_mask_Bool_V(model, formatting_token_ids)
        self.formatting_token_ids: frozenset[int] = frozenset(
            self._formatting_mask_Bool_V.nonzero().flatten().tolist()
        )

        # (eval, item, concept) with no single-token surface: skipped, flagged.
        # Each casualty appears once even though run_trial revisits the trial
        # once per (lens, scale). A success-surface set with no single-token
        # member is recorded as "success:<first surfaces>".
        self.dropped_concepts: list[tuple[str, str, str]] = []
        # (eval, item) already warned about for an empty baseline id set, so the
        # warning fires once per trial rather than once per (lens, scale).
        self._empty_baseline_warned: set[tuple[str, str]] = set()

    ### VECTORS

    def _atom_vectors(self, token_id: int, lens: BaseLens) -> dict[int, t.Tensor]:
        """Per-band-layer readout vectors for ``token_id`` under ``lens``
        (:func:`readout_vectors`, ``W_U[token] @ J_l``), unit-normalised."""
        vectors_L_dict_N = readout_vectors(
            lens, self.model, token_id, layers=self.band_layers
        )
        return {
            layer: (vector_N / vector_N.norm()).to(self.device)
            for layer, vector_N in vectors_L_dict_N.items()
        }

    ### FORWARD PASSES

    @t.no_grad()
    def _forward_recording_band(
        self, prompt: str, interventions: Sequence[Intervention]
    ) -> tuple[t.Tensor, dict[int, t.Tensor]]:
        """Run one forward under ``interventions``; return the final-position
        model logits (fp32, CPU) and the band-layer residuals at every position
        (fp32, on the device).

        The recorder is entered *after* the intervention hooks so it observes
        edited residuals (final-layer edits included)."""
        input_ids_Int_1S = self.model.encode(prompt, max_length=self.max_seq_len)
        layers_to_record = sorted(set(self.band_layers) | {self.final_layer})
        with (
            InterventionHooks(self.model, interventions),
            ActivationRecorder(self.model.layers, at=layers_to_record) as recorder,
        ):
            self.model.forward(input_ids_Int_1S)
            band_residuals_L_dict_SN = {
                layer: residual_mean_over_streams(recorder.activations[layer])[0].detach().float()
                for layer in self.band_layers
            }
            # The model's own logits: the final block collapsed by the model's head, not the lens's mean.
            final_residual_F = (
                collapse_streams_like_the_model(self.model, recorder.activations[self.final_layer])[0, -1]
                .detach()
                .float()
            )
        model_logits_V = self.model.unembed(final_residual_F[None, :])[0].float().cpu()
        return model_logits_V, band_residuals_L_dict_SN

    def _clean_pass(self, prompt: str, clean_cache: dict[str, _CleanPass] | None) -> _CleanPass:
        if clean_cache is not None and prompt in clean_cache:
            return clean_cache[prompt]
        logits_V, band_residuals_L_dict_SN = self._forward_recording_band(prompt, [])
        clean = _CleanPass(
            logits_V=logits_V,
            band_residuals_L_dict_SN={
                layer: residuals_SN.cpu()
                for layer, residuals_SN in band_residuals_L_dict_SN.items()
            },
        )
        if clean_cache is not None:
            clean_cache[prompt] = clean
        return clean

    ### INTERVENTIONS

    def _build_interventions(
        self,
        *,
        source_vectors_L_dict_N: dict[int, t.Tensor],
        target_vectors_L_dict_N: dict[int, t.Tensor],
        clean: _CleanPass,
        scale: float,
        positions: Sequence[int],
    ) -> tuple[list[Intervention], list[int]]:
        """One clamp swap per band layer; near-parallel source/target layers
        are skipped and returned separately."""
        interventions: list[Intervention] = []
        skipped_layers: list[int] = []
        positions_Int_P = t.tensor(list(positions), dtype=t.long)
        for layer in self.band_layers:
            unit_source_N = source_vectors_L_dict_N[layer]
            unit_target_N = target_vectors_L_dict_N[layer]
            # A near-parallel pair makes the two-atom pinv ill-conditioned, so the
            # layer is skipped (and recorded).
            near_parallel = abs_cosine_between(unit_source_N, unit_target_N) > NEAR_PARALLEL_ABS_COSINE
            if near_parallel:
                skipped_layers.append(layer)
                continue

            # Clean residuals at the edited positions, in edited-position order.
            clean_residuals_PN = clean.band_residuals_L_dict_SN[layer][positions_Int_P].to(self.device)

            basis_N2 = t.stack([unit_source_N, unit_target_N], dim=1)
            interventions.append(
                _swapped_coordinate_clamp(
                    basis_N2, clean_residuals_PN, scale=scale, layer=layer, positions=positions
                )
            )
        return interventions, skipped_layers

    ### TRIAL BOOKKEEPING

    def _record_dropped_concept(self, trial: SwapTrial, concept: str) -> bool:
        """Append ``(eval, item, concept)`` to :attr:`dropped_concepts` unless it
        is already there. Returns True when newly recorded, so the caller logs
        each casualty once rather than once per (lens, scale)."""
        entry = (trial.eval_slug, trial.name, concept)
        if entry in self.dropped_concepts:
            return False
        self.dropped_concepts.append(entry)
        return True

    def _resolve_concept_token(
        self, trial: SwapTrial, role: str, concept: str
    ) -> tuple[int, str] | None:
        """``(token_id, surface)`` for the trial's ``role`` (``"source"`` /
        ``"target"``) concept, or ``None`` (recorded and logged once) when it has
        no single-token surface."""
        token_id, chosen = resolve_swap_token(self.model.tokenizer, concept)
        if token_id is None or chosen is None:
            if self._record_dropped_concept(trial, concept):
                logger.warning(
                    "dropping %s/%s %s concept %r: no single-token surface",
                    trial.eval_slug,
                    trial.name,
                    role,
                    concept,
                )
            return None
        return token_id, chosen

    def _check_not_formatting(self, trial: SwapTrial, role: str, ids: Sequence[int]) -> None:
        clashing = sorted(set(ids) & self.formatting_token_ids)
        if clashing:
            raise ValueError(
                f"{trial.eval_slug}/{trial.name}: {role} ids {clashing} are formatting "
                "tokens, so their word rank is undefined; a scored surface must be a "
                "word token"
            )

    def _word_logits(self, logits_V: t.Tensor) -> t.Tensor:
        return logits_V.masked_fill(self._formatting_mask_Bool_V, float("-inf"))

    def _top_tokens_json(self, logits_V: t.Tensor) -> str:
        top_ids = logits_V.topk(NUM_TOP_TOKENS).indices.tolist()
        return json.dumps([self.model.tokenizer.decode([int(i)]) for i in top_ids])

    ### RUNNING

    @t.no_grad()
    def run_trial(
        self,
        trial: SwapTrial,
        lens_name: str,
        *,
        scale: float,
        clean_cache: dict[str, _CleanPass] | None = None,
    ) -> SwapTrialResult | None:
        """Execute one (trial, lens, scale). Returns ``None`` if the source,
        the target, or every success surface has no single-token form (each
        recorded once in :attr:`dropped_concepts`), or if every band layer is
        near-parallel. An empty *baseline* surface set does not drop the trial:
        its baseline ranks are -1 and logprobs nan (``n_baseline_ids == 0``).
        ``lens_name`` is the key of :attr:`lenses`."""
        lens = self.lenses[lens_name]

        source = self._resolve_concept_token(trial, "source", trial.source)
        target = self._resolve_concept_token(trial, "target", trial.target)
        if source is None or target is None:
            return None
        source_token_id, source_surface = source
        target_token_id, target_surface = target

        success_ids = single_token_ids(self.model.tokenizer, trial.success_surfaces)
        baseline_ids = single_token_ids(self.model.tokenizer, trial.baseline_surfaces)
        if not success_ids:
            descriptor = f"success:{'|'.join(trial.success_surfaces[:3])}"
            if self._record_dropped_concept(trial, descriptor):
                logger.warning(
                    "dropping %s/%s: no single-token success surface in %s",
                    trial.eval_slug,
                    trial.name,
                    trial.success_surfaces,
                )
            return None
        if not baseline_ids:
            trial_key = (trial.eval_slug, trial.name)
            if trial_key not in self._empty_baseline_warned:
                self._empty_baseline_warned.add(trial_key)
                logger.warning(
                    "%s/%s: no single-token baseline surface in %s; baseline ranks "
                    "will be -1 and baseline logprobs nan",
                    trial.eval_slug,
                    trial.name,
                    trial.baseline_surfaces,
                )
        self._check_not_formatting(trial, "success", success_ids)
        self._check_not_formatting(trial, "baseline", baseline_ids)

        input_ids_Int_1S = self.model.encode(trial.prompt, max_length=self.max_seq_len)
        seq_len = input_ids_Int_1S.shape[1]
        # Every position but 0, the attention sink, whose large-norm residual gives
        # unstable pinv coordinates.
        positions = list(range(1, seq_len))

        clean = self._clean_pass(trial.prompt, clean_cache)

        source_vectors_L_dict_N = self._atom_vectors(source_token_id, lens)
        target_vectors_L_dict_N = self._atom_vectors(target_token_id, lens)
        interventions, skipped_layers = self._build_interventions(
            source_vectors_L_dict_N=source_vectors_L_dict_N,
            target_vectors_L_dict_N=target_vectors_L_dict_N,
            clean=clean,
            scale=scale,
            positions=positions,
        )
        if not interventions:
            logger.warning(
                "%s/%s lens=%s: every band layer near-parallel; skipping",
                trial.eval_slug,
                trial.name,
                lens_name,
            )
            return None

        swapped_logits_V, swapped_band_L_dict_SN = self._forward_recording_band(
            trial.prompt, interventions
        )

        # Edit size at the final position, relative to the clean residual there.
        edit_norm_ratios_L_list = [
            float(
                (swapped_band_L_dict_SN[layer][-1].cpu() - clean.band_residuals_L_dict_SN[layer][-1]).norm()
                / clean.band_residuals_L_dict_SN[layer][-1].norm()
            )
            for layer in self.band_layers
        ]
        next_token = self.model.tokenizer.decode([int(swapped_logits_V.argmax())])
        clean_logits_V = clean.logits_V
        clean_word_logits_V = self._word_logits(clean_logits_V)
        swapped_word_logits_V = self._word_logits(swapped_logits_V)

        def rank_or_sentinel(logits_V: t.Tensor, ids: list[int]) -> int:
            return min_rank_over_candidates(logits_V, ids) if ids else -1

        def logprob_or_nan(logits_V: t.Tensor, ids: list[int]) -> float:
            return _summed_logprob(logits_V, ids) if ids else float("nan")

        return SwapTrialResult(
            success_rank=min_rank_over_candidates(swapped_logits_V, success_ids),
            success_logprob=_summed_logprob(swapped_logits_V, success_ids),
            baseline_rank=rank_or_sentinel(swapped_logits_V, baseline_ids),
            baseline_logprob=logprob_or_nan(swapped_logits_V, baseline_ids),
            clean_success_rank=min_rank_over_candidates(clean_logits_V, success_ids),
            clean_success_logprob=_summed_logprob(clean_logits_V, success_ids),
            clean_baseline_rank=rank_or_sentinel(clean_logits_V, baseline_ids),
            clean_baseline_logprob=logprob_or_nan(clean_logits_V, baseline_ids),
            success_word_rank=min_rank_over_candidates(swapped_word_logits_V, success_ids),
            baseline_word_rank=rank_or_sentinel(swapped_word_logits_V, baseline_ids),
            clean_success_word_rank=min_rank_over_candidates(clean_word_logits_V, success_ids),
            clean_baseline_word_rank=rank_or_sentinel(clean_word_logits_V, baseline_ids),
            edit_norm_ratio=float(
                sorted(edit_norm_ratios_L_list)[len(edit_norm_ratios_L_list) // 2]
            ),
            next_token=next_token,
            swapped_layers=tuple(
                layer for layer in self.band_layers if layer not in skipped_layers
            ),
            skipped_layers=tuple(skipped_layers),
            n_swap_positions=len(positions),
            n_success_ids=len(success_ids),
            n_baseline_ids=len(baseline_ids),
            source_token_id=source_token_id,
            target_token_id=target_token_id,
            source_surface=source_surface,
            target_surface=target_surface,
            clean_top_tokens=self._top_tokens_json(clean_logits_V),
            swapped_top_tokens=self._top_tokens_json(swapped_logits_V),
            clean_top_word_tokens=self._top_tokens_json(clean_word_logits_V),
            swapped_top_word_tokens=self._top_tokens_json(swapped_word_logits_V),
        )

    @t.no_grad()
    def run(self, trials: Sequence[SwapTrial]) -> pd.DataFrame:
        """Run every (trial, lens, scale) and return a long DataFrame.

        The clean pass is computed once per distinct prompt and shared across
        lenses and scales. One row per (trial, lens, scale); dropped concepts
        and all-near-parallel trials are skipped and logged.

        Columns:

        - ``eval``, ``item``, ``category``, ``variant``, ``lens``, ``scale``.
        - ``n_swap_positions``: positions actually edited.
        - ``success_rank`` / ``success_logprob``, ``baseline_rank`` /
          ``baseline_logprob`` under the edit, and their ``clean_*`` twins from
          the un-edited pass; ``*_word_rank`` twins with formatting tokens
          removed from the distribution. ``n_success_ids`` / ``n_baseline_ids``:
          how many single-token vocab ids each surface set contributed (0
          baseline ids gives ``baseline_rank == clean_baseline_rank == -1`` and
          nan logprobs).
        - ``edit_norm_ratio``, ``next_token``, ``n_swapped_layers``,
          ``n_skipped_layers``, ``source_surface``, ``target_surface``.
        - Qualitative: ``clean_top_tokens``, ``swapped_top_tokens`` and their
          ``*_word_tokens`` twins (JSON lists of decoded tokens, best first).
        - Derived: ``success`` (rank 1), ``success_top5`` (rank <= 5),
          ``success_word_top1``, ``success_word_top5``, ``delta_logprob``
          (success minus baseline logprob under the edit).
        """

        clean_cache: dict[str, _CleanPass] = {}
        rows: list[dict] = []
        trials_with_empty_baseline: set[tuple[str, str]] = set()
        for trial_idx, trial in enumerate(trials):
            for lens_name in self.lenses:
                for scale in self.scales:
                    result = self.run_trial(trial, lens_name, scale=scale, clean_cache=clean_cache)
                    if result is None:
                        continue
                    if result.n_baseline_ids == 0:
                        trials_with_empty_baseline.add((trial.eval_slug, trial.name))
                    result_fields = asdict(result)
                    swapped_layers = result_fields.pop("swapped_layers")
                    skipped_layers = result_fields.pop("skipped_layers")
                    result_fields.pop("source_token_id")
                    result_fields.pop("target_token_id")
                    rows.append(
                        {
                            "eval": trial.eval_slug,
                            "item": trial.name,
                            "category": trial.category,
                            "variant": trial.variant,
                            "lens": lens_name,
                            "scale": scale,
                            **result_fields,
                            "n_swapped_layers": len(swapped_layers),
                            "n_skipped_layers": len(skipped_layers),
                            "success": result.success_rank == 1,
                            "success_top5": result.success_rank <= 5,
                            "success_word_top1": result.success_word_rank == 1,
                            "success_word_top5": result.success_word_rank <= 5,
                            "delta_logprob": result.success_logprob - result.baseline_logprob,
                        }
                    )
            if (trial_idx + 1) % 20 == 0 or trial_idx + 1 == len(trials):
                logger.info("ran trial %d/%d", trial_idx + 1, len(trials))
        if trials_with_empty_baseline:
            logger.warning(
                "%d/%d trials have no single-token baseline surface "
                "(clean_baseline_rank == -1)",
                len(trials_with_empty_baseline),
                len(trials),
            )
        return pd.DataFrame(rows)

