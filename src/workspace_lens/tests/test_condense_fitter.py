"""CPU tests for ``fitting/condense_experts/fitter.py``:

1. The search finds a planted expert from starts that exclude it.
2. ``build_search_starting_points`` breaks ties towards the lower index.
3. ``ExpertWeighting.save`` / ``load`` round-trip bit for bit, config included.
4. ``ExpertWeightingFitter._expert_transports_PEF`` equals the explicit
   ``J_e x_p`` loop.
5. ``ExpertWeightingFitter.fit`` end to end on ``TinyDecoder``: the named subset
   in residual order, the starts follow the tie rule on a planted tie, and with
   ``steps=0`` the result is the best of the three starts by exact train macro.

The model is ``TinyDecoder`` with the fake experts and residuals of
``fixtures.py`` (shared with ``test_expert_jacobians.py``). Its unembed is
LayerNorm-based, so no rank exactness is asserted here
(``test_rank_objective.py`` pins that with a fake RMSNorm unembed); these tests
pin that the pipeline is self-consistent.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch as t

from jlens.tests.tiny import TinyDecoder
from lens_evals.readout_evals.readout_evals import ReadoutResiduals
from workspace_lens.fitting.condense_experts import (
    CondenseConfig,
    ExpertJacobians,
    ExpertWeighting,
    ExpertWeightingFitter,
    LearnedWeights,
)
from workspace_lens.fitting.condense_experts import fitter as condense_fitter
from workspace_lens.fitting.condense_experts.fitter import (
    build_search_starting_points,
    fit_expert_weights,
)
from workspace_lens.fitting.condense_experts.rank_objective import (
    build_pair_table,
    combined_logits,
    exact_macro_pass,
    expert_logits,
    expert_pair_pass_scores,
    hit_margins,
)
from workspace_lens.tests.fixtures import (
    CONDENSE_D_MODEL,
    CONDENSE_LAYERS,
    CONDENSE_NUM_EXPERTS,
    CONDENSE_VOCAB_SIZE,
    FAKE_EXCLUDED_TOKEN_IDS,
    FAKE_ITEM_NAMES,
    all_item_expert_transports,
    make_fake_expert_jacobians,
    make_fake_readout_residuals,
    make_planted_expert_problem,
)


def test_fit_finds_a_planted_expert() -> None:
    pairs, logits_PEV = make_planted_expert_problem()
    # Starts from scores that exclude the planted expert 2.
    inits = build_search_starting_points(t.tensor([0.9, 0.5, 0.0, 0.5]), top_n=3)
    assert inits["uniform"].tolist() == [1.0] * CONDENSE_NUM_EXPERTS
    assert inits["select_best"].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert inits["mean_top"].tolist() == [1.0, 1.0, 0.0, 1.0]
    learned = fit_expert_weights(
        logits_PEV,
        pairs,
        excluded_mask_Bool_V=None,
        initial_weightings=inits,
        condense_config=CondenseConfig(pass_k=1, steps=60, learning_rate=0.1),
    )
    assert learned.train_macro_pass == pytest.approx(1.0)
    assert int(learned.weights_E.abs().argmax()) == 2
    assert learned.weights_E.norm() == pytest.approx(1.0, abs=1e-5)
    assert learned.signed_shares_E().abs().sum() == pytest.approx(1.0, abs=1e-5)
    assert learned.init_name in inits and 0 <= learned.step <= 60


def _starts(scores: list[float], top_n: int) -> tuple[list[float], list[float]]:
    inits = build_search_starting_points(t.tensor(scores), top_n=top_n)
    assert set(inits) == {"uniform", "select_best", "mean_top"}
    assert inits["uniform"].tolist() == [1.0] * len(scores)
    return inits["select_best"].tolist(), inits["mean_top"].tolist()


def test_build_search_starting_points_breaks_ties_towards_the_lower_index() -> None:
    assert _starts([0.5, 0.7, 0.7, 0.2], top_n=2) == ([0, 1, 0, 0], [0, 1, 1, 0])
    assert _starts([0.5, 0.7, 0.7, 0.2], top_n=1) == ([0, 1, 0, 0], [0, 1, 0, 0])
    assert _starts([0.4, 0.4, 0.4, 0.4], top_n=3) == ([1, 0, 0, 0], [1, 1, 1, 0])
    # Top-n by (-score, index).
    assert _starts([0.1, 0.9, 0.2, 0.9], top_n=3) == ([0, 1, 0, 0], [0, 1, 1, 1])
    for bad_top_n in (0, 5):
        with pytest.raises(ValueError):
            build_search_starting_points(
                t.tensor([0.1, 0.9, 0.2, 0.9]), top_n=bad_top_n
            )


def test_expert_weighting_json_round_trip(tmp_path: Path) -> None:
    raw_1_E = t.tensor([0.3, -0.4, 0.5, 0.7])
    raw_2_E = t.tensor([1.0, 0.1, -0.1, 0.0])
    weighting = ExpertWeighting(
        learned_weights_L_dict={
            2: LearnedWeights(raw_2_E / raw_2_E.norm(), 0.625, "mean_top", 35),
            1: LearnedWeights(raw_1_E / raw_1_E.norm(), 0.5, "uniform", 0),
        },
        condense_config=CondenseConfig(pass_k=3, top_n=2, steps=40),
    )
    assert weighting.layers == [1, 2]
    assert set(weighting.weights_L_dict_E) == {1, 2}
    assert float(weighting.signed_shares_L_dict_E[1].abs().sum()) == pytest.approx(1.0)

    path = tmp_path / "out" / "expert_weights.json"
    weighting.save(path)

    payload = json.loads(path.read_text())
    assert list(payload["by_layer"]) == ["1", "2"]  # sorted by layer
    assert set(payload["by_layer"]["2"]) == {
        "weights_unit_l2",
        "signed_shares",
        "train_macro_pass",
        "init",
        "step",
    }
    assert payload["by_layer"]["2"]["init"] == "mean_top"
    assert payload["by_layer"]["2"]["step"] == 35
    assert payload["by_layer"]["2"]["train_macro_pass"] == 0.625
    assert sum(
        abs(share) for share in payload["by_layer"]["2"]["signed_shares"]
    ) == pytest.approx(1.0)
    assert payload["config"]["pass_k"] == 3 and payload["config"]["steps"] == 40

    loaded = ExpertWeighting.load(path)
    assert loaded.condense_config == weighting.condense_config
    assert set(loaded.learned_weights_L_dict) == {1, 2}
    for layer, learned in weighting.learned_weights_L_dict.items():
        loaded_learned = loaded.learned_weights_L_dict[layer]
        assert loaded_learned.weights_E.dtype == t.float32
        assert t.equal(loaded_learned.weights_E, learned.weights_E)
        assert loaded_learned.train_macro_pass == learned.train_macro_pass
        assert loaded_learned.init_name == learned.init_name
        assert loaded_learned.step == learned.step


def test_expert_transports_match_the_explicit_loop(tmp_path: Path) -> None:
    generator = t.Generator().manual_seed(9)
    model = TinyDecoder(n_layers=4, d_model=CONDENSE_D_MODEL, vocab_size=CONDENSE_VOCAB_SIZE)
    experts = make_fake_expert_jacobians(generator, tmp_path)
    residuals = make_fake_readout_residuals(generator)
    for layer in CONDENSE_LAYERS:
        transports_PEF = all_item_expert_transports(model, experts, residuals, layer)
        assert transports_PEF.shape == (len(FAKE_ITEM_NAMES), CONDENSE_NUM_EXPERTS, CONDENSE_D_MODEL)
        assert transports_PEF.dtype == t.float32
        for item_idx in range(len(FAKE_ITEM_NAMES)):
            for expert_idx in range(CONDENSE_NUM_EXPERTS):
                expected_F = (
                    experts.experts_L_dict_EFN[layer][expert_idx]
                    @ residuals.residuals_L_dict_PN[layer][item_idx]
                )
                assert t.allclose(transports_PEF[item_idx, expert_idx], expected_F, atol=1e-5)


def test_fitting_item_indices_keep_residual_order_and_reject_unknown_names(
    tmp_path: Path,
) -> None:
    generator = t.Generator().manual_seed(11)
    model = TinyDecoder(n_layers=4, d_model=CONDENSE_D_MODEL, vocab_size=CONDENSE_VOCAB_SIZE)
    residuals = make_fake_readout_residuals(generator)
    fitter = ExpertWeightingFitter(
        model,
        make_fake_expert_jacobians(generator, tmp_path),
        residuals,
        excluded_token_ids=FAKE_EXCLUDED_TOKEN_IDS,
    )
    assert fitter._fitting_item_indices(None) == [0, 1, 2, 3, 4, 5]
    assert fitter._fitting_item_indices(["item3", "item0", "item5", "item2"]) == [
        0,
        2,
        3,
        5,
    ]
    with pytest.raises(ValueError, match="not in the residuals"):
        fitter._fitting_item_indices(["item0", "nope"])


def test_fitter_rejects_experts_fitted_at_another_width(tmp_path: Path) -> None:
    """The experts' config must describe the model whose unembed ranks
    (``check_model_matches_config``): a ``d_model`` mismatch is refused up
    front rather than at the transport einsum."""
    generator = t.Generator().manual_seed(13)
    model = TinyDecoder(n_layers=4, d_model=CONDENSE_D_MODEL, vocab_size=CONDENSE_VOCAB_SIZE)
    experts = make_fake_expert_jacobians(generator, tmp_path)
    wider_experts = dataclasses.replace(
        experts, config=dataclasses.replace(experts.config, d_model=CONDENSE_D_MODEL + 1)
    )
    with pytest.raises(ValueError, match="d_model"):
        ExpertWeightingFitter(
            model,
            wider_experts,
            make_fake_readout_residuals(generator),
            excluded_token_ids=FAKE_EXCLUDED_TOKEN_IDS,
        )


def test_fit_end_to_end_on_tiny_decoder(tmp_path: Path) -> None:
    model = TinyDecoder(n_layers=4, d_model=CONDENSE_D_MODEL, vocab_size=CONDENSE_VOCAB_SIZE)
    generator = t.Generator().manual_seed(12)
    experts = make_fake_expert_jacobians(generator, tmp_path)
    residuals = make_fake_readout_residuals(generator)
    fitter = ExpertWeightingFitter(
        model, experts, residuals, excluded_token_ids=FAKE_EXCLUDED_TOKEN_IDS
    )
    config = CondenseConfig(pass_k=3, steps=10)
    weighting = fitter.fit(CONDENSE_LAYERS, condense_config=config)
    assert isinstance(weighting, ExpertWeighting)
    assert list(weighting.learned_weights_L_dict) == CONDENSE_LAYERS
    assert weighting.condense_config == config
    for learned in weighting.learned_weights_L_dict.values():
        assert learned.weights_E.shape == (CONDENSE_NUM_EXPERTS,)
        assert learned.weights_E.dtype == t.float32
        assert learned.weights_E.norm() == pytest.approx(1.0, abs=1e-5)
        assert 0.0 <= learned.train_macro_pass <= 1.0
        assert learned.init_name in {"uniform", "select_best", "mean_top"}
        assert 0 <= learned.step <= 10

    # A candidate among the excluded ids breaks the rank rule's premise.
    with pytest.raises(ValueError, match="candidate ids"):
        ExpertWeightingFitter(model, experts, residuals, excluded_token_ids=[3])


def _plant_perfect_expert(
    model: TinyDecoder,
    experts: ExpertJacobians,
    residuals: ReadoutResiduals,
    *,
    expert_indices: Sequence[int],
) -> None:
    """Overwrite ``expert_indices`` at every layer with one Jacobian that maps
    each item's residual onto the unembedding row of its first candidate, so
    the candidate tops the logits: identical experts, so identical scores (a
    planted tie), and — as the best experts — a tie at the top."""
    head_VF = model.lm_head.weight.detach()
    first_candidates = [
        next(iter(candidates.values()))[0]
        for candidates in residuals.candidate_ids_P_list
    ]
    targets_PN = head_VF[first_candidates] * 20.0
    for layer in CONDENSE_LAYERS:
        residuals_PN = residuals.residuals_L_dict_PN[layer]
        # J x_p = h_p for every item: J = H^T (X X^T)^{-1} X (P <= N rows).
        planted_FN = targets_PN.T @ t.linalg.solve(residuals_PN @ residuals_PN.T, residuals_PN)
        for expert_idx in expert_indices:
            experts.experts_L_dict_EFN[layer][expert_idx] = planted_FN


def test_fit_subset_starts_and_steps_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On the named subset (residual order): the scores passed to
    ``build_search_starting_points`` are the stand-alone per-expert scores, whose
    starts follow the tie rule (experts 0 and 2 are planted identical and best),
    and with ``steps=0`` the result is the best of the three starts by exact
    train macro."""
    model = TinyDecoder(n_layers=4, d_model=CONDENSE_D_MODEL, vocab_size=CONDENSE_VOCAB_SIZE)
    generator = t.Generator().manual_seed(13)
    experts = make_fake_expert_jacobians(generator, tmp_path)
    residuals = make_fake_readout_residuals(generator)
    _plant_perfect_expert(model, experts, residuals, expert_indices=(0, 2))

    recorded_scores: list[t.Tensor] = []
    original_build_search_starting_points = build_search_starting_points

    def recording_build_search_starting_points(
        scores_E: t.Tensor, *, top_n: int
    ) -> dict[str, t.Tensor]:
        recorded_scores.append(scores_E.clone())
        return original_build_search_starting_points(scores_E, top_n=top_n)

    monkeypatch.setattr(
        condense_fitter,
        "build_search_starting_points",
        recording_build_search_starting_points,
    )

    subset_names = ["item3", "item0", "item5", "item2"]
    subset_indices = [0, 2, 3, 5]
    config = CondenseConfig(pass_k=3, top_n=2, steps=0)
    fitter = ExpertWeightingFitter(
        model, experts, residuals, excluded_token_ids=FAKE_EXCLUDED_TOKEN_IDS
    )
    weighting = fitter.fit(CONDENSE_LAYERS, item_names=subset_names, condense_config=config)
    assert len(recorded_scores) == len(CONDENSE_LAYERS)

    pairs = build_pair_table(
        residuals.candidate_ids_P_list,
        residuals.eval_slugs,
        subset_indices,
        device="cpu",
    )
    excluded_Bool_V = t.zeros(CONDENSE_VOCAB_SIZE, dtype=t.bool)
    excluded_Bool_V[FAKE_EXCLUDED_TOKEN_IDS] = True
    for layer, scores_E in zip(CONDENSE_LAYERS, recorded_scores, strict=True):
        transports_PEF = all_item_expert_transports(model, experts, residuals, layer)[subset_indices]
        logits_PEV = expert_logits(model.unembed, transports_PEF)
        expected_scores_E = expert_pair_pass_scores(
            logits_PEV, pairs, excluded_mask_Bool_V=excluded_Bool_V, k=config.pass_k
        )
        assert t.equal(scores_E, expected_scores_E)
        # The planted tie sits at the top, so the tie rule decides the starts:
        # expert 0 alone, and {0, 2} for the top two.
        scores = scores_E.tolist()
        assert scores[0] == scores[2] > max(scores[1], scores[3])
        inits = original_build_search_starting_points(scores_E, top_n=config.top_n)
        assert inits["select_best"].tolist() == [1.0, 0.0, 0.0, 0.0]
        assert inits["mean_top"].tolist() == [1.0, 0.0, 1.0, 0.0]

        # steps=0: the returned weights are the start with the highest exact
        # macro on the subset's pairs, unit-normalised.
        exact_by_init: dict[str, float] = {}
        for init_name, init_E in inits.items():
            margin_Q, _ = hit_margins(
                combined_logits(logits_PEV, init_E / init_E.norm()),
                pairs,
                excluded_mask_Bool_V=excluded_Bool_V,
                k=config.pass_k,
            )
            exact_by_init[init_name] = exact_macro_pass(margin_Q, pairs)
        learned = weighting.learned_weights_L_dict[layer]
        assert learned.step == 0
        assert learned.train_macro_pass == pytest.approx(max(exact_by_init.values()))
        assert exact_by_init[learned.init_name] == pytest.approx(learned.train_macro_pass)
        chosen_init_E = inits[learned.init_name]
        assert t.allclose(learned.weights_E, chosen_init_E / chosen_init_E.norm())


def test_single_expert_pass_scores_match_the_stand_alone_scores(tmp_path: Path) -> None:
    """``single_expert_pass_scores`` returns, per layer, the stand-alone macro
    pass@k of every expert on the fitting items: identical to
    ``expert_pair_pass_scores`` on the explicit transports (the numbers ``fit``
    starts from), with the planted identical experts 0 and 2 tied at the top,
    and on a named item subset the same computation restricted to those items."""
    model = TinyDecoder(n_layers=4, d_model=CONDENSE_D_MODEL, vocab_size=CONDENSE_VOCAB_SIZE)
    generator = t.Generator().manual_seed(13)
    experts = make_fake_expert_jacobians(generator, tmp_path)
    residuals = make_fake_readout_residuals(generator)
    _plant_perfect_expert(model, experts, residuals, expert_indices=(0, 2))
    fitter = ExpertWeightingFitter(
        model, experts, residuals, excluded_token_ids=FAKE_EXCLUDED_TOKEN_IDS
    )
    excluded_Bool_V = t.zeros(CONDENSE_VOCAB_SIZE, dtype=t.bool)
    excluded_Bool_V[FAKE_EXCLUDED_TOKEN_IDS] = True
    subset_names = ["item3", "item0", "item5", "item2"]
    subset_indices = [0, 2, 3, 5]
    for item_names, item_indices in ((None, list(range(6))), (subset_names, subset_indices)):
        scores_L_dict_E = fitter.single_expert_pass_scores(
            CONDENSE_LAYERS, item_names=item_names, pass_k=3
        )
        assert list(scores_L_dict_E) == CONDENSE_LAYERS
        pairs = build_pair_table(
            residuals.candidate_ids_P_list, residuals.eval_slugs, item_indices, device="cpu"
        )
        for layer, scores_E in scores_L_dict_E.items():
            transports_PEF = all_item_expert_transports(model, experts, residuals, layer)[
                item_indices
            ]
            expected_E = expert_pair_pass_scores(
                expert_logits(model.unembed, transports_PEF),
                pairs,
                excluded_mask_Bool_V=excluded_Bool_V,
                k=3,
            )
            assert t.equal(scores_E, expected_E)
            scores = scores_E.tolist()
            assert scores[0] == scores[2] > max(scores[1], scores[3])
