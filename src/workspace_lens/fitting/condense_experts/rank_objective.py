"""The readout eval's rank rule as tensor maths, so a weight search can score
thousands of candidate weightings per layer without touching the model again.

Everything here is a pure function of tensors: the pair table, the cached
per-expert logits, the hit margins, and the macro means (exact and smooth).

What the eval scores. A pair is an (item, intermediate); its rank under a lens
is ``1 + #(logits > best candidate logit)`` with the excluded ids at ``-inf``
(candidates are never excluded, see
:class:`~workspace_lens.fitting.condense_experts.fitter.ExpertWeightingFitter`),
and the
metric is the unweighted mean over evals of the pair hit rate ``rank <= k``.
:func:`hit_margins` writes that rule as ``best candidate logit - k-th largest
logit >= 0``, which is differentiable in the logits everywhere the rank is
constant — the handle the smooth surrogate :func:`soft_macro_pass` needs.

Why one cached tensor is enough. Under weights ``w`` the lens prediction for
item ``p`` is ``x_p(w) = sum_e w_e J_e x_p = sum_e w_e T_pe`` (``T`` = the
expert transports, computed once per layer). The model's unembed is RMSNorm
followed by a linear head,

    unembed(x) = W_U (g * x) / rms(x),        rms(x) = sqrt(mean(x^2) + eps),

so multiplying it back by ``rms`` removes the only nonlinearity::

    unembed(x) * rms(x) = W_U diag(g) x       — linear in x.

:func:`expert_logits` therefore caches ``L_pe = unembed(T_pe) * rms(T_pe)``
(``[P, E, V]``), and by linearity

    L_p(w) = sum_e w_e L_pe = W_U diag(g) x_p(w) = unembed(x_p(w)) * rms(x_p(w)),

which is the logits of the weighted prediction times the positive per-item
scalar ``rms(x_p(w))``. Ranks ignore positive rescalings, so
:func:`combined_logits` scores exactly what the eval would score, and the
objective is *linear* in ``w`` — no separate rms tensor and no per-weight model
pass.

Two notes on the numerics:

- ``rms`` as computed here omits the norm's ``eps``: the exact per-term factor
  is ``sqrt(mean(T^2) + eps)``, so each term carries a relative error of
  ``O(eps / mean(T^2))`` — at ``eps = 1e-6`` and residual-scale ``T``, far
  below the gaps that decide ranks.
- The cache is fp32 with ``P`` x ``E`` x ``V`` entries per layer. It is freed
  before the next layer (:meth:`ExpertWeightingFitter.fit`).

There is no runtime check that the unembed really is RMSNorm-then-linear;
``tests/test_rank_objective.py`` pins the exactness with a
fake RMSNorm model (``TinyDecoder`` has LayerNorm, for which the argument
fails).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch as t

### PAIRS


@dataclass(frozen=True)
class PairTable:
    """The (item, intermediate) pairs the eval scores, for a chosen list of items:
    ``item_index_Int_Q`` indexes that list, ``candidate_ids_Int_QC`` is padded by
    repeating each pair's first id (a duplicate cannot change the max), and
    ``eval_index_Int_Q`` groups pairs by eval for the macro mean."""

    item_index_Int_Q: t.Tensor
    candidate_ids_Int_QC: t.Tensor
    eval_index_Int_Q: t.Tensor
    group_eval_slugs: tuple[str, ...]

    @property
    def num_pairs(self) -> int:
        return int(self.item_index_Int_Q.numel())


def build_pair_table(
    candidate_ids_P_list: Sequence[Mapping[str, Sequence[int]]],
    eval_slugs: Sequence[str],
    item_indices: Sequence[int],
    *,
    device: t.device | str,
) -> PairTable:
    """The pairs of the items ``item_indices`` (``item_index`` counts within
    ``item_indices``, so it indexes tensors already restricted to those items).
    ``candidate_ids_P_list[i]`` maps each of item ``i``'s intermediates to its
    candidate token ids and ``eval_slugs[i]`` is its eval; evals are numbered in
    sorted slug order."""
    group_eval_slugs = tuple(sorted({eval_slugs[i] for i in item_indices}))
    item_index: list[int] = []
    candidate_lists: list[list[int]] = []
    eval_index: list[int] = []
    for local_idx, item_idx in enumerate(item_indices):
        for candidate_ids in candidate_ids_P_list[item_idx].values():
            item_index.append(local_idx)
            candidate_lists.append(list(candidate_ids))
            eval_index.append(group_eval_slugs.index(eval_slugs[item_idx]))
    if not candidate_lists:
        raise ValueError("no scoreable (item, intermediate) pairs among the chosen items")
    max_candidates = max(len(ids) for ids in candidate_lists)
    padded = [ids + [ids[0]] * (max_candidates - len(ids)) for ids in candidate_lists]
    return PairTable(
        item_index_Int_Q=t.tensor(item_index, dtype=t.long, device=device),
        candidate_ids_Int_QC=t.tensor(padded, dtype=t.long, device=device),
        eval_index_Int_Q=t.tensor(eval_index, dtype=t.long, device=device),
        group_eval_slugs=group_eval_slugs,
    )


### THE FAST RANK PATH


@t.no_grad()
def expert_logits(
    unembed: Callable[[t.Tensor], t.Tensor],
    transports_PEF: t.Tensor,
    *,
    batch_size: int = 32,
) -> t.Tensor:
    """``[P, E, V]`` fp32: ``L_pe = unembed(T_pe) * rms(T_pe)``, the model's
    logits of every expert's transport with the RMSNorm scaling multiplied back
    in (module docstring), so that ``sum_e w_e L_pe`` is a positive rescaling of
    the logits of ``sum_e w_e T_pe``. The rms is computed in fp32 over
    ``d_model`` and the unembed runs in ``batch_size``-item chunks."""
    num_items, num_experts, d_model = transports_PEF.shape
    rms_PE = transports_PEF.float().pow(2).mean(dim=-1).sqrt()
    logit_chunks_list_BEV: list[t.Tensor] = []
    for start in range(0, num_items, batch_size):
        transports_chunk_BEF = transports_PEF[start : start + batch_size]
        # unembed every (item, expert) transport as one row, then restore [B, E, V]
        logits_BEV = (
            unembed(transports_chunk_BEF.reshape(-1, d_model))
            .reshape(transports_chunk_BEF.shape[0], num_experts, -1)
            .float()
        )
        # A model sharded over several GPUs (device_map) keeps its unembedding on the last device while
        # the transports live on the first: scale on the logits' device, then return the chunk on the
        # transports' device so the pair table and the weight fit downstream see one device.
        rms_chunk_BE1 = rms_PE[start : start + batch_size, :, None].to(logits_BEV.device)
        logit_chunks_list_BEV.append((logits_BEV * rms_chunk_BE1).to(transports_PEF.device))
    return t.cat(logit_chunks_list_BEV)


def combined_logits(logits_PEV: t.Tensor, weights_E: t.Tensor) -> t.Tensor:
    """``[P, V]`` fp32: ``sum_e w_e L_pe`` — the rank-equivalent logits of the
    weighted prediction, differentiable in ``weights_E``. Summed expert by
    expert so only one ``[P, V]`` slice is live at a time."""
    num_items, num_experts, vocab_size = logits_PEV.shape
    weights_on_cache_E = weights_E.to(logits_PEV)
    logits_PV = t.zeros(num_items, vocab_size, device=logits_PEV.device)
    for expert_idx in range(num_experts):
        logits_PV = logits_PV + weights_on_cache_E[expert_idx] * logits_PEV[:, expert_idx]
    return logits_PV


def hit_margins(
    logits_PV: t.Tensor,
    pairs: PairTable,
    *,
    excluded_mask_Bool_V: t.Tensor | None,
    k: int,
) -> tuple[t.Tensor, t.Tensor]:
    """Per pair: ``(best candidate logit − k-th largest logit, logit std over
    the vocab)``. The eval's rank is ``1 + #(logits > best candidate)`` with
    excluded ids at ``-inf`` (candidates are never excluded), so a pair is a
    hit (rank <= k) exactly when the margin is >= 0. The spread is that of the
    unmasked logits; both are computed on the ``P`` item rows and then indexed
    out per pair."""
    spread_P = logits_PV.std(dim=1).detach()
    if excluded_mask_Bool_V is not None:
        logits_PV = logits_PV.masked_fill(
            excluded_mask_Bool_V.to(logits_PV.device)[None, :], float("-inf")
        )
    logits_QV = logits_PV[pairs.item_index_Int_Q]
    best_candidate_Q = logits_QV.gather(1, pairs.candidate_ids_Int_QC).amax(dim=1)
    kth_largest_Q = logits_QV.topk(k, dim=1).values[:, -1]
    return best_candidate_Q - kth_largest_Q, spread_P[pairs.item_index_Int_Q]


def per_eval_mean(values_Q: t.Tensor, pairs: PairTable) -> t.Tensor:
    """``[G]`` (one entry per eval group, in ``pairs.group_eval_slugs`` order): the
    mean of ``values_Q`` within each eval."""
    num_evals = len(pairs.group_eval_slugs)
    sums_G = t.zeros(num_evals, device=values_Q.device, dtype=values_Q.dtype)
    sums_G.index_add_(0, pairs.eval_index_Int_Q, values_Q)
    counts_G = t.zeros(num_evals, device=values_Q.device, dtype=values_Q.dtype)
    counts_G.index_add_(0, pairs.eval_index_Int_Q, t.ones_like(values_Q))
    return sums_G / counts_G


def exact_macro_pass(margin_Q: t.Tensor, pairs: PairTable) -> float:
    """The eval's macro pass@k: the unweighted mean over evals of the hit rate."""
    return float(per_eval_mean((margin_Q >= 0).float(), pairs).mean())


def soft_macro_pass(
    margin_Q: t.Tensor, spread_Q: t.Tensor, pairs: PairTable, *, temperature: float
) -> t.Tensor:
    """The smooth surrogate: ``sigmoid(margin / (temperature * spread))`` in
    place of the hit indicator, then the same macro mean."""
    return per_eval_mean(t.sigmoid(margin_Q / (temperature * spread_Q)), pairs).mean()


@t.no_grad()
def expert_pair_pass_scores(
    logits_PEV: t.Tensor,
    pairs: PairTable,
    *,
    excluded_mask_Bool_V: t.Tensor | None,
    k: int,
) -> t.Tensor:
    """``[E]``: each expert's stand-alone exact macro pass@k on ``pairs`` — the
    score of the single-expert lens ``J_i`` at this layer, whose logits are
    ``logits_PEV[:, i]`` (:func:`combined_logits` with the one-hot weighting
    returns exactly that slice). Its argmax and top-n are the ``select_best`` /
    ``mean_top`` starts of
    :func:`~workspace_lens.fitting.condense_experts.fitter.build_search_starting_points`."""
    scores: list[float] = []
    for expert_idx in range(logits_PEV.shape[1]):
        margin_Q, _ = hit_margins(
            logits_PEV[:, expert_idx],
            pairs,
            excluded_mask_Bool_V=excluded_mask_Bool_V,
            k=k,
        )
        scores.append(exact_macro_pass(margin_Q, pairs))
    return t.tensor(scores)
