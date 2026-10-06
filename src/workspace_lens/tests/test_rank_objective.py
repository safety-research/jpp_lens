"""CPU tests for ``fitting/condense_experts/rank_objective.py`` (fake tensors,
no model):

1. The pair table + margin rule reproduce the eval's rank rule (brute force,
   with and without an exclusion mask).
2. The smooth surrogate tends to the exact macro as the temperature vanishes.
3. Through a fake RMSNorm-then-linear unembed (with ``eps``), the cached
   per-expert logits rank *exactly* like the exact unembed of the weighted
   prediction, including negative weights (the cache is fp32, so no near-tie
   swaps are tolerated).
4. ``expert_pair_pass_scores`` equals the brute-force score of every single
   expert alone.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest
import torch as t

from workspace_lens.fitting.condense_experts.rank_objective import (
    build_pair_table,
    combined_logits,
    exact_macro_pass,
    expert_logits,
    expert_pair_pass_scores,
    hit_margins,
    soft_macro_pass,
)
from workspace_lens.tests.fixtures import (
    CONDENSE_D_MODEL,
    CONDENSE_NUM_EXPERTS,
    CONDENSE_VOCAB_SIZE,
    make_planted_expert_problem,
)

# Four fake items: (eval slug, {intermediate: candidate ids}). The last has no
# scoreable intermediate and contributes no pairs.
EVAL_SLUGS = ["poetry", "typo", "poetry", "typo"]
CANDIDATE_IDS_BY_ITEM: list[dict[str, list[int]]] = [
    {"a": [3], "b": [5, 6]},
    {"c": [7]},
    {"d": [9, 10, 11]},
    {},
]


def _brute_force_macro(
    logits_PV: t.Tensor,
    candidate_ids_P_list: Sequence[dict[str, Sequence[int]]],
    eval_slugs: Sequence[str],
    item_indices: Sequence[int],
    excluded_mask_Bool_V: t.Tensor | None,
    k: int,
) -> float:
    """The eval's rule written out: rank = 1 + #(masked logits > best candidate);
    macro = mean over evals of the pair hit rate."""
    hits_by_eval: dict[str, list[float]] = {}
    for local_idx, item_idx in enumerate(item_indices):
        logits_V = logits_PV[local_idx].clone()
        if excluded_mask_Bool_V is not None:
            logits_V[excluded_mask_Bool_V] = float("-inf")
        for candidate_ids in candidate_ids_P_list[item_idx].values():
            best = max(float(logits_V[c]) for c in candidate_ids)
            rank = 1 + int((logits_V > best).sum())
            hits_by_eval.setdefault(eval_slugs[item_idx], []).append(float(rank <= k))
    return sum(sum(h) / len(h) for h in hits_by_eval.values()) / len(hits_by_eval)


def _exclusion_mask() -> t.Tensor:
    excluded_Bool_V = t.zeros(CONDENSE_VOCAB_SIZE, dtype=t.bool)
    excluded_Bool_V[[0, 1, 2, 20, 21]] = True
    return excluded_Bool_V


def test_pair_table_and_exact_macro_match_the_evals_rank_rule() -> None:
    item_indices = [0, 1, 2, 3]
    pairs = build_pair_table(CANDIDATE_IDS_BY_ITEM, EVAL_SLUGS, item_indices, device="cpu")
    assert pairs.num_pairs == 4 and pairs.group_eval_slugs == ("poetry", "typo")
    assert pairs.item_index_Int_Q.tolist() == [0, 0, 1, 2]
    assert pairs.eval_index_Int_Q.tolist() == [0, 0, 1, 0]
    # Padded by repeating each pair's first id.
    assert pairs.candidate_ids_Int_QC.tolist() == [
        [3, 3, 3],
        [5, 6, 5],
        [7, 7, 7],
        [9, 10, 11],
    ]
    generator = t.Generator().manual_seed(0)
    for trial in range(20):
        logits_PV = t.randn(len(item_indices), CONDENSE_VOCAB_SIZE, generator=generator) * 3
        for mask in (None, _exclusion_mask()):
            for k in (1, 3, 10):
                margin_Q, _ = hit_margins(logits_PV, pairs, excluded_mask_Bool_V=mask, k=k)
                expected = _brute_force_macro(
                    logits_PV, CANDIDATE_IDS_BY_ITEM, EVAL_SLUGS, item_indices, mask, k
                )
                assert exact_macro_pass(margin_Q, pairs) == pytest.approx(expected), (
                    trial,
                    mask is None,
                    k,
                )
    with pytest.raises(ValueError):
        build_pair_table(CANDIDATE_IDS_BY_ITEM, EVAL_SLUGS, [3], device="cpu")


def test_soft_macro_tends_to_exact_as_temperature_vanishes() -> None:
    pairs = build_pair_table(CANDIDATE_IDS_BY_ITEM, EVAL_SLUGS, [0, 1, 2], device="cpu")
    logits_PV = t.randn(3, CONDENSE_VOCAB_SIZE, generator=t.Generator().manual_seed(1))
    margin_Q, spread_Q = hit_margins(logits_PV, pairs, excluded_mask_Bool_V=None, k=5)
    exact = exact_macro_pass(margin_Q, pairs)
    soft_cold = float(soft_macro_pass(margin_Q, spread_Q, pairs, temperature=1e-4))
    soft_warm = float(soft_macro_pass(margin_Q, spread_Q, pairs, temperature=10.0))
    assert soft_cold == pytest.approx(exact, abs=1e-3)
    assert 0.3 < soft_warm < 0.7  # a warm surrogate is near 1/2 everywhere


def _rmsnorm_unembed(
    head_FV: t.Tensor, gain_F: t.Tensor, *, eps: float
) -> Callable[[t.Tensor], t.Tensor]:
    """A fake model unembed: RMSNorm (with the real models' ``eps``) then a
    linear head — the structure the cached logits rely on."""

    def unembed(residual_BN: t.Tensor) -> t.Tensor:
        mean_square_B1 = residual_BN.float().pow(2).mean(dim=-1, keepdim=True)
        return (gain_F * residual_BN.float() / (mean_square_B1 + eps).sqrt()) @ head_FV

    return unembed


def test_linear_fast_path_ranks_like_the_unembed_of_the_weighted_prediction() -> None:
    """Through RMSNorm + linear head, the logits of sum_e w_e T_e are a positive
    multiple of sum_e w_e L_e (L_e = unembed(T_e) * rms(T_e)): identical ranks
    for any weights, including negative ones."""
    generator = t.Generator().manual_seed(2)
    head_FV = t.randn(CONDENSE_D_MODEL, CONDENSE_VOCAB_SIZE, generator=generator)
    gain_F = t.rand(CONDENSE_D_MODEL, generator=generator) + 0.5
    unembed = _rmsnorm_unembed(head_FV, gain_F, eps=1e-6)
    # Experts with very different transport scales, so the folded-in rms matters.
    expert_scales_E = t.tensor([1.0, 10.0, 0.1, 3.0])
    transports_PEF = (
        t.randn(5, CONDENSE_NUM_EXPERTS, CONDENSE_D_MODEL, generator=generator) * expert_scales_E[None, :, None]
    )
    logits_PEV = expert_logits(unembed, transports_PEF, batch_size=2)
    assert logits_PEV.shape == (5, CONDENSE_NUM_EXPERTS, CONDENSE_VOCAB_SIZE)
    assert logits_PEV.dtype == t.float32
    # The cache is the unembed's logits with the transport's rms multiplied back in.
    rms_PE = transports_PEF.pow(2).mean(dim=-1).sqrt()
    unembedded_PEV = unembed(transports_PEF.reshape(-1, CONDENSE_D_MODEL)).reshape(
        5, CONDENSE_NUM_EXPERTS, CONDENSE_VOCAB_SIZE
    )
    assert t.allclose(logits_PEV, unembedded_PEV * rms_PE[:, :, None], atol=1e-5)
    for weights_E in (
        t.tensor([0.5, 0.2, -0.4, 0.1]),
        t.tensor([0.0, 1.0, 0.0, 0.0]),
        t.ones(CONDENSE_NUM_EXPERTS),
    ):
        fast_PV = combined_logits(logits_PEV, weights_E)
        exact_PV = unembed(t.einsum("pen,e->pn", transports_PEF, weights_E))
        # An fp32 cache leaves no near-tie swaps: the orderings are identical.
        assert t.equal(fast_PV.argsort(dim=1), exact_PV.argsort(dim=1))
        assert t.equal(fast_PV.argmax(dim=1), exact_PV.argmax(dim=1))
        # And the fast logits are a positive rescaling of the exact ones, per
        # item (least-squares scale; fp32 leaves ~1e-5 relative noise).
        scale_P1 = (fast_PV * exact_PV).sum(dim=1, keepdim=True) / exact_PV.pow(2).sum(
            dim=1, keepdim=True
        )
        assert bool((scale_P1 > 0).all())
        assert t.allclose(fast_PV, scale_P1 * exact_PV, atol=1e-4 * float(fast_PV.std()))


def test_expert_pair_pass_scores_match_brute_force_per_expert() -> None:
    generator = t.Generator().manual_seed(4)
    item_indices = [0, 1, 2, 3]
    pairs = build_pair_table(CANDIDATE_IDS_BY_ITEM, EVAL_SLUGS, item_indices, device="cpu")
    excluded_Bool_V = _exclusion_mask()
    for trial in range(10):
        logits_PEV = (
            t.randn(len(item_indices), CONDENSE_NUM_EXPERTS, CONDENSE_VOCAB_SIZE, generator=generator) * 3
        )
        for mask in (None, excluded_Bool_V):
            for k in (1, 3, 10):
                scores_E = expert_pair_pass_scores(
                    logits_PEV, pairs, excluded_mask_Bool_V=mask, k=k
                )
                assert scores_E.shape == (CONDENSE_NUM_EXPERTS,)
                for expert_idx in range(CONDENSE_NUM_EXPERTS):
                    # Expert e alone is the one-hot combination, i.e. exactly
                    # the slice the function scores.
                    expected = _brute_force_macro(
                        logits_PEV[:, expert_idx],
                        CANDIDATE_IDS_BY_ITEM,
                        EVAL_SLUGS,
                        item_indices,
                        mask,
                        k,
                    )
                    assert float(scores_E[expert_idx]) == pytest.approx(expected), (
                        trial,
                        mask is None,
                        k,
                        expert_idx,
                    )
    # On the planted problem the planted expert is the only perfect one at k=1.
    planted_pairs, planted_logits_PEV = make_planted_expert_problem()
    planted_scores_E = expert_pair_pass_scores(
        planted_logits_PEV, planted_pairs, excluded_mask_Bool_V=None, k=1
    )
    assert planted_scores_E.tolist() == [0.0, 0.0, 1.0, 0.0]
