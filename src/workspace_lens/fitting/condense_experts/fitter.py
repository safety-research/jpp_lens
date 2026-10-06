"""Fitting the expert weights: the search, its result and the workflow around
the model.

Objective. Every layer is fitted on its own: the objective is that layer's
pair-level macro pass@k on the fitting items (:mod:`rank_objective`). A hit is
an (item, intermediate) pair whose rank *at this layer alone* is ``<= k``; the
hit rate is taken per eval and the macro is the unweighted mean over evals. The
readout evals' pass@k (:mod:`~lens_evals.readout_evals.readout_eval_scoring`)
takes the min rank over layers before thresholding, so the per-layer objective
is a surrogate for it.

Search. The exact objective is piecewise constant in ``w``, so
:func:`fit_expert_weights` runs Adam on a smooth surrogate — the sigmoid of the
margin between the best candidate logit and the k-th largest logit, in units of
the logit spread — from several starts (uniform, the best single expert alone,
the top-n indicator; :func:`build_search_starting_points` from
:func:`~workspace_lens.fitting.condense_experts.rank_objective.expert_pair_pass_scores`),
evaluates the exact macro along the way and keeps the weights with the highest
exact macro. Weights are scale-free (unit L2 during the search) and may be
negative. :class:`CondenseConfig` holds the search's hyperparameters.

Result. :class:`LearnedWeights` is one layer's winning iterate and
:class:`ExpertWeighting` is the per-layer collection returned by
:meth:`ExpertWeightingFitter.fit` — the thing to save beside the combined lens
(:meth:`ExpertWeighting.save`) and to hand to
:meth:`~workspace_lens.fitting.condense_experts.expert_jacobians.ExpertJacobians.combine`
(as :attr:`ExpertWeighting.weights_L_dict_E`).
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch as t

from jlens.protocol import LensModel
from workspace_lens.fitting.condense_experts.expert_jacobians import ExpertJacobians
from workspace_lens.fitting.condense_experts.rank_objective import (
    PairTable,
    build_pair_table,
    combined_logits,
    exact_macro_pass,
    expert_logits,
    expert_pair_pass_scores,
    hit_margins,
    soft_macro_pass,
)
from workspace_lens.utils import (
    check_model_matches_config,
    ensure_parent_dir,
    token_id_mask_Bool_V,
    vocab_size_of,
)

if TYPE_CHECKING:
    from lens_evals.readout_evals.readout_evals import ReadoutResiduals

logger = logging.getLogger(__name__)


### THE SEARCH


@dataclass(frozen=True)
class CondenseConfig:
    """Hyperparameters of the expert-weight search."""

    pass_k: int = 10  # the eval's k: a pair is a hit when its rank <= pass_k
    top_n: int = 3  # experts in the mean_top start
    steps: int = 200  # Adam steps per start
    learning_rate: float = 0.05
    temperature: float = 0.1  # of the per-pair logit spread, in the sigmoid surrogate
    exact_every: int = 5  # evaluate the exact macro every this many steps


# The default search of ExpertWeightingFitter.fit (a frozen dataclass instance,
# so sharing one is safe; a call in a default trips B008).
DEFAULT_CONDENSE_CONFIG = CondenseConfig()


def build_search_starting_points(
    scores_E: t.Tensor, *, top_n: int
) -> dict[str, t.Tensor]:
    """The search's starting points from the stand-alone per-expert scores
    (:func:`~workspace_lens.fitting.condense_experts.rank_objective.expert_pair_pass_scores`):
    ``uniform`` (all ones), ``select_best`` (the highest-scoring expert alone)
    and ``mean_top`` (the indicator of the ``top_n`` highest). Ties go to the
    lower index: experts are ordered by ``(-score, index)``."""
    num_experts = int(scores_E.numel())
    if not 1 <= top_n <= num_experts:
        raise ValueError(f"top_n must be in 1..{num_experts}, got {top_n}")
    expert_scores_list_E = [float(score) for score in scores_E.tolist()]
    by_score_then_index = sorted(
        range(num_experts), key=lambda e: (-expert_scores_list_E[e], e)
    )
    select_best_E = t.zeros(num_experts)
    select_best_E[by_score_then_index[0]] = 1.0
    mean_top_E = t.zeros(num_experts)
    mean_top_E[by_score_then_index[:top_n]] = 1.0
    return {
        "uniform": t.ones(num_experts),
        "select_best": select_best_E,
        "mean_top": mean_top_E,
    }


@dataclass(frozen=True)
class LearnedWeights:
    weights_E: t.Tensor  # unit L2, fp32 on the CPU
    train_macro_pass: float  # exact, on the fitting pairs
    init_name: str
    step: int

    def signed_shares_E(self) -> t.Tensor:
        """The weights scaled to unit L1 (``sum |w_e| = 1``), signs kept — for
        display."""
        return self.weights_E / self.weights_E.abs().sum()


def fit_expert_weights(
    logits_PEV: t.Tensor,
    pairs: PairTable,
    *,
    excluded_mask_Bool_V: t.Tensor | None,
    initial_weightings: dict[str, t.Tensor],
    condense_config: CondenseConfig,
) -> LearnedWeights:
    """Adam on the surrogate from every init; the returned weights are the
    iterate (over all starting points and evaluated steps) with the highest exact macro
    pass on ``pairs``, ties broken by the higher surrogate value."""
    device = logits_PEV.device
    best_scored_weights: tuple[float, float, LearnedWeights] | None = None
    for init_name, initial_weights_E in initial_weightings.items():
        weights_E = (
            (initial_weights_E.float() / initial_weights_E.float().norm())
            .to(device)
            .clone()
        )
        weights_E.requires_grad_(True)
        optimizer = t.optim.Adam([weights_E], lr=condense_config.learning_rate)
        for step in range(condense_config.steps + 1):
            unit_weights_E = weights_E / weights_E.norm()
            margin_Q, spread_Q = hit_margins(
                combined_logits(logits_PEV, unit_weights_E),
                pairs,
                excluded_mask_Bool_V=excluded_mask_Bool_V,
                k=condense_config.pass_k,
            )
            surrogate_macro_score = soft_macro_pass(
                margin_Q, spread_Q, pairs, temperature=condense_config.temperature
            )
            if step % condense_config.exact_every == 0 or step == condense_config.steps:
                exact_macro_score = exact_macro_pass(margin_Q.detach(), pairs)
                learned_weights = LearnedWeights(
                    unit_weights_E.detach().cpu().clone(),
                    exact_macro_score,
                    init_name,
                    step,
                )
                candidate = (
                    exact_macro_score,
                    float(surrogate_macro_score.detach()),
                    learned_weights,
                )
                if (
                    best_scored_weights is None
                    or candidate[:2] > best_scored_weights[:2]
                ):
                    best_scored_weights = candidate
            if step == condense_config.steps:
                break
            optimizer.zero_grad()
            (-surrogate_macro_score).backward()
            optimizer.step()
    assert best_scored_weights is not None
    return best_scored_weights[2]


### THE FITTED WEIGHTING


@dataclass
class ExpertWeighting:
    """The fitted expert weights of a lens: one :class:`LearnedWeights` per
    layer, plus the :class:`CondenseConfig` they were fitted under
    (``condense_config``).

    :attr:`weights_L_dict_E` is what
    :meth:`~workspace_lens.fitting.condense_experts.expert_jacobians.ExpertJacobians.combine`
    takes; :meth:`save` / :meth:`load` are the JSON that travels beside the
    combined lens.
    """

    learned_weights_L_dict: dict[int, LearnedWeights]
    condense_config: CondenseConfig

    @property
    def layers(self) -> list[int]:
        return sorted(self.learned_weights_L_dict)

    @property
    def weights_L_dict_E(self) -> dict[int, t.Tensor]:
        """``{layer: weights_E}`` (unit L2) — the fitted map's weights."""
        return {
            layer: learned.weights_E
            for layer, learned in self.learned_weights_L_dict.items()
        }

    @property
    def signed_shares_L_dict_E(self) -> dict[int, t.Tensor]:
        """``{layer: signed_shares_E}`` (unit L1, signs kept) — for display."""
        return {
            layer: learned.signed_shares_E()
            for layer, learned in self.learned_weights_L_dict.items()
        }

    def save(self, path: str | Path) -> None:
        """The weighting as JSON beside the combined lens: the search's
        ``condense_config`` as a nested dict under ``config``, and under ``by_layer`` one entry per layer
        (key: the layer as a string) with ``weights_unit_l2`` (the fitted vector
        — what
        :meth:`~workspace_lens.fitting.condense_experts.expert_jacobians.ExpertJacobians.combine`
        takes), ``signed_shares`` (unit L1, for display), and the winning
        iterate's ``train_macro_pass``, ``init`` and ``step``."""
        payload = {
            "config": dataclasses.asdict(self.condense_config),
            "by_layer": {  # on-disk key; the attribute is learned_weights_L_dict
                str(layer): {
                    "weights_unit_l2": learned.weights_E.tolist(),
                    "signed_shares": learned.signed_shares_E().tolist(),
                    "train_macro_pass": learned.train_macro_pass,
                    "init": learned.init_name,
                    "step": learned.step,
                }
                for layer, learned in sorted(self.learned_weights_L_dict.items())
            },
        }
        ensure_parent_dir(path)
        Path(path).write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> ExpertWeighting:
        """The weighting back from a :meth:`save` file, ``LearnedWeights`` and
        all (fp32 CPU weights). ``tolist`` / JSON keep fp32 values exactly, so
        the round trip is bit for bit."""
        payload = json.loads(Path(path).read_text())
        return cls(
            learned_weights_L_dict={
                int(layer): LearnedWeights(
                    weights_E=t.tensor(entry["weights_unit_l2"], dtype=t.float32),
                    train_macro_pass=entry["train_macro_pass"],
                    init_name=entry["init"],
                    step=entry["step"],
                )
                for layer, entry in payload["by_layer"].items()
            },
            condense_config=CondenseConfig(**payload["config"]),
        )


### FITTING THE WEIGHTS


class ExpertWeightingFitter:
    """Fits one expert-weight vector per layer on labelled readout items.

    Holds the model whose unembed defines the ranking, the
    :class:`~workspace_lens.fitting.condense_experts.expert_jacobians.ExpertJacobians`,
    the items' :class:`~lens_evals.readout_evals.readout_evals.ReadoutResiduals` and the
    token ids the ranking ignores, and derives once: the model's device, its
    LM-head width (:func:`~workspace_lens.utils.vocab_size_of`) and the
    ``Bool[V]`` exclusion mask. The model must be the one the experts were
    fitted on (:func:`~workspace_lens.utils.check_model_matches_config`:
    ``d_model`` and, where it exposes one, its HF name), else the weights
    would be ranked through another model's unembed. The rank rule of
    :func:`~workspace_lens.fitting.condense_experts.rank_objective.hit_margins`
    assumes candidates are never excluded
    (:func:`~lens_evals.readout_evals.readout_eval_items.non_semantic_token_ids` spares them),
    so an excluded id among the candidates of any recorded item is an error
    here — checked over every item, which covers every fitting subset
    :meth:`fit` can be given.
    """

    def __init__(
        self,
        model: LensModel,
        expert_jacobians: ExpertJacobians,
        residuals: ReadoutResiduals,
        *,
        excluded_token_ids: Sequence[int],
    ) -> None:
        check_model_matches_config(model, expert_jacobians.config)
        self.model = model
        self.expert_jacobians = expert_jacobians
        self.residuals = residuals
        self.excluded_token_ids = excluded_token_ids
        self.device: t.device | str = getattr(model, "input_device", t.device("cpu"))
        self.vocab_size = vocab_size_of(model)
        self.excluded_mask_Bool_V = self._build_exclusion_mask()

    def _build_exclusion_mask(self) -> t.Tensor:
        """``Bool[V]``, true at the ids the ranking ignores; an id that is also
        a candidate of one of the recorded items breaks the rank rule's
        premise, so that is an error."""
        excluded_mask_Bool_V = token_id_mask_Bool_V(
            self.excluded_token_ids, self.vocab_size
        ).to(self.device)
        # One entry per candidate id of every (item, intermediate) pair, flattened.
        flat_candidate_ids_Int = t.as_tensor(
            [
                candidate_id
                for candidates in self.residuals.candidate_ids_P_list
                for candidate_ids in candidates.values()
                for candidate_id in candidate_ids
            ],
            dtype=t.long,
            device=self.device,
        )
        flat_candidate_excluded_Bool = excluded_mask_Bool_V[flat_candidate_ids_Int]
        if bool(flat_candidate_excluded_Bool.any()):
            excluded_candidate_ids = sorted(
                set(flat_candidate_ids_Int[flat_candidate_excluded_Bool].tolist())
            )
            raise ValueError(
                "excluded_token_ids contains candidate ids of the recorded items "
                f"({excluded_candidate_ids[:10]}); spare the candidates "
                "(readout_eval_items.non_semantic_token_ids does)"
            )
        return excluded_mask_Bool_V

    def _fitting_item_indices(self, item_names: Sequence[str] | None) -> list[int]:
        """Indices into the residuals of the items to fit on: all of them, or
        those named in ``item_names`` (e.g. the fitting split from
        ``readout_eval_items.split_fitting_items``), in residual order. Item names are
        unique across the evals; a name the residuals do not hold is an error
        rather than a silently smaller fitting set."""
        if item_names is None:
            return list(range(len(self.residuals.item_names)))
        wanted = set(item_names)
        missing = sorted(wanted - set(self.residuals.item_names))
        if missing:
            raise ValueError(
                f"{len(missing)} requested item(s) are not in the residuals: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        return [idx for idx, name in enumerate(self.residuals.item_names) if name in wanted]

    def _expert_transports_PEF(
        self, layer: int, item_indices_Int_P: t.Tensor
    ) -> t.Tensor:
        """``[P, E, F]``: ``T_pe = J_e x_p``, every expert's transport of the
        readout residual at ``layer`` of the items ``item_indices_Int_P``
        (``x @ J_e^T`` per expert), fp32 on the model's device (the experts are
        cast from the file dtype per layer here). The einsum covers every
        recorded item and is then indexed, so its shape does not depend on the
        fitting subset."""
        residuals_PN = self.residuals.residuals_L_dict_PN[layer].to(
            device=self.device, dtype=t.float32
        )
        experts_EFN = self.expert_jacobians.experts_L_dict_EFN[layer].to(
            device=self.device, dtype=t.float32
        )
        return t.einsum("pj,eij->pei", residuals_PN, experts_EFN)[item_indices_Int_P]

    def fitting_pairs(
        self, item_names: Optional[Sequence[str]]  # noqa: UP045
    ) -> tuple[PairTable, t.Tensor]:
        """The pair table of the fitting items (:meth:`_fitting_item_indices`) and
        those items' residual indices as a ``[P]`` long tensor on the device."""
        fitting_item_indices = self._fitting_item_indices(item_names)
        pairs = build_pair_table(
            self.residuals.candidate_ids_P_list,
            self.residuals.eval_slugs,
            fitting_item_indices,
            device=self.device,
        )
        fitting_item_indices_Int_P = t.tensor(
            fitting_item_indices, dtype=t.long, device=self.device
        )
        return pairs, fitting_item_indices_Int_P

    def _layer_logits_and_single_expert_scores(
        self,
        layer: int,
        pairs: PairTable,
        fitting_item_indices_Int_P: t.Tensor,
        *,
        pass_k: int,
    ) -> tuple[t.Tensor, t.Tensor]:
        """One layer's ``[P, E, V]`` expert logits (the one pass through the
        model's unembed) and each expert's stand-alone pair-level macro
        pass@``pass_k`` on the fitting pairs, ``[E]``. The caller frees the
        logits before the next layer."""
        transports_PEF = self._expert_transports_PEF(layer, fitting_item_indices_Int_P)
        logits_PEV = expert_logits(self.model.unembed, transports_PEF)
        del transports_PEF
        scores_E = expert_pair_pass_scores(
            logits_PEV, pairs, excluded_mask_Bool_V=self.excluded_mask_Bool_V, k=pass_k
        )
        return logits_PEV, scores_E

    def single_expert_pass_scores(
        self,
        layers: Sequence[int],
        *,
        item_names: Sequence[str] | None = None,
        pass_k: int = DEFAULT_CONDENSE_CONFIG.pass_k,
    ) -> dict[int, t.Tensor]:
        """Per layer, ``[E]``: each expert's stand-alone pair-level macro
        pass@``pass_k`` on the fitting items (the same
        :func:`~workspace_lens.fitting.condense_experts.rank_objective.expert_pair_pass_scores`
        that :meth:`fit` logs and starts from). Its per-layer argmax picks one
        fixed expert per layer, with no weight search. The ``[P, E, V]`` logits
        are freed before the next layer."""
        pairs, fitting_item_indices_Int_P = self.fitting_pairs(item_names)
        scores_L_dict_E: dict[int, t.Tensor] = {}
        for layer in layers:
            logits_PEV, scores_L_dict_E[layer] = self._layer_logits_and_single_expert_scores(
                layer, pairs, fitting_item_indices_Int_P, pass_k=pass_k
            )
            del logits_PEV
        return scores_L_dict_E

    def fit(
        self,
        layers: Sequence[int],
        *,
        item_names: Sequence[str] | None = None,
        condense_config: CondenseConfig = DEFAULT_CONDENSE_CONFIG,
    ) -> ExpertWeighting:
        """One learned weight vector per layer, fitted on the labelled items of
        the residuals (all of them, or the ``item_names`` subset — the fitting
        split — in residual order; :meth:`_fitting_item_indices`).

        Objective: each layer's own pair-level macro pass@``condense_config.pass_k`` on
        the fitting items, the excluded ids removed from the ranking — a
        per-layer surrogate for the eval's min rank over layers (module
        docstring). Per layer: :meth:`_expert_transports_PEF` →
        :func:`~workspace_lens.fitting.condense_experts.rank_objective.expert_logits`
        (the one pass through the model's unembed) →
        :func:`~workspace_lens.fitting.condense_experts.rank_objective.expert_pair_pass_scores`
        → :func:`build_search_starting_points` (the best expert alone and the
        ``condense_config.top_n`` highest; ties → the lower index) →
        :func:`fit_expert_weights`. The ``[P, E, V]`` logits are freed before the
        next layer.
        """
        pairs, fitting_item_indices_Int_P = self.fitting_pairs(item_names)

        learned_weights_L_dict: dict[int, LearnedWeights] = {}
        for layer in layers:
            logits_PEV, scores_E = self._layer_logits_and_single_expert_scores(
                layer, pairs, fitting_item_indices_Int_P, pass_k=condense_config.pass_k
            )
            learned_weights_L_dict[layer] = fit_expert_weights(
                logits_PEV,
                pairs,
                excluded_mask_Bool_V=self.excluded_mask_Bool_V,
                initial_weightings=build_search_starting_points(
                    scores_E, top_n=condense_config.top_n
                ),
                condense_config=condense_config,
            )
            del logits_PEV  # [P, E, V] fp32: the layer's big tensor
            logger.info(
                "layer %d: single-expert pass@%d %s -> learned %.3f (%s, step %d)",
                layer,
                condense_config.pass_k,
                [round(score, 3) for score in scores_E.tolist()],
                learned_weights_L_dict[layer].train_macro_pass,
                learned_weights_L_dict[layer].init_name,
                learned_weights_L_dict[layer].step,
            )
        return ExpertWeighting(
            learned_weights_L_dict=learned_weights_L_dict, condense_config=condense_config
        )
