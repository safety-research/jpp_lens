"""Shared builders for the CPU-only ``workspace_lens`` tests: the 4-layer
``TinyDecoder`` with its fitted J-lens and lens config, lenses built directly
from given matrices (Jacobian, logit), the 8-layer "band" decoder with a
near-identity random band lens for the eval runners, a random router
collection, the fake condense-experts problem, and the readout-eval JSON
writer. Every builder that makes a ``LensConfig`` takes
``tmp_path`` so no checkpoint path points into the repo. These are plain
functions, not fixtures: each test module keeps its own module- or
function-scoped fixtures and calls these from them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch as t

from jlens.tests.tiny import TinyDecoder
from lens_evals.readout_evals.readout_evals import ReadoutResiduals
from workspace_lens.config import LensConfig
from workspace_lens.fitting.condense_experts import (
    ExpertJacobians,
    ExpertWeightingFitter,
)
from workspace_lens.fitting.condense_experts.rank_objective import (
    PairTable,
    build_pair_table,
)
from workspace_lens.fitting.expert_fitting import ExpertJacobianTrainer
from workspace_lens.fitting.jacobian_fitting import LensTrainer
from workspace_lens.fitting.types import FitStepForward
from workspace_lens.lenses.jacobian_lens import JacobianLens
from workspace_lens.lenses.logit_lens import LogitLens
from workspace_lens.routing.router import ActivationRouterCollection

### THE 4-LAYER TINY DECODER AND ITS FITTED J-LENS

TINY_D_MODEL = 8
TINY_SOURCE_LAYERS = [0, 1, 2]
TINY_FIT_PROMPTS = ["abcdefghij " * 5, "klmnopqrst " * 5]
# TinyDecoder's byte tokenizer maps byte b to id 1 + b % 30 and decodes id i to
# chr(96 + i), so ids 27-30 decode to the non-alphanumeric '{', '|', '}', '~'.
# The bytes ':', ';', '8', '9', 't', 'u', 'v', 'w' land on those ids, so this
# prompt mixes semantic and non-semantic positions (the tests that use it check
# this rather than assume it).
MIXED_SEMANTIC_PROMPT = "abc: def; 89 tuvw " * 3


def make_tiny_decoder(
    tokenizer: Any | None = None,
    *,
    n_layers: int = 4,
    vocab_size: int = 32,
    seed: int = 0,
) -> TinyDecoder:
    """``TinyDecoder(n_layers, d_model=8, vocab_size, seed)``. When ``tokenizer``
    is given it replaces the decoder's ByteTokenizer: the eval runners and
    loaders call the tokenizer the way the real Qwen tokenizer is called (with
    ``add_special_tokens``), which ByteTokenizer does not accept."""
    decoder = TinyDecoder(
        n_layers=n_layers, d_model=TINY_D_MODEL, vocab_size=vocab_size, seed=seed
    )
    if tokenizer is not None:
        decoder.tokenizer = tokenizer
    return decoder


def make_tiny_lens_config(tmp_path: Path, checkpoint_name: str, **overrides: Any) -> LensConfig:
    """A LensConfig for the 4-layer TinyDecoder with checkpoints under
    ``tmp_path``: layers 0-2 transported to the final block, and no
    intermediate checkpoints unless a test asks for them. ``jacobian_rows_per_pass`` 4 is
    below ``d_model`` 8 on purpose: the Jacobian estimator then needs two
    backward passes, so its multi-pass ``dim_start`` slicing is exercised."""
    defaults: dict[str, Any] = dict(
        hf_model_name="tiny",
        checkpoint_name=checkpoint_name,
        artifacts_base_dir=str(tmp_path),
        source_layers=list(TINY_SOURCE_LAYERS),
        jacobian_rows_per_pass=4,
        max_seq_len=64,
        checkpoint_every_n_prompts=None,
    )
    defaults.update(overrides)
    return LensConfig(**defaults)


def fit_tiny_jacobian_lens(
    model: TinyDecoder, tmp_path: Path, checkpoint_name: str
) -> JacobianLens:
    """A J-lens fitted on ``model`` at layers 0-2 over ``TINY_FIT_PROMPTS``."""
    config = make_tiny_lens_config(tmp_path, checkpoint_name)
    return LensTrainer(config, model, prompts=list(TINY_FIT_PROMPTS)).fit()


def make_jacobian_lens(
    jacobians_L_dict_FN: dict[int, t.Tensor],
    tmp_path: Path,
    *,
    num_prompts: int,
    d_model: int,
    checkpoint_name: str = "mini",
) -> JacobianLens:
    """A JacobianLens built directly from given matrices (no fitting)."""
    config = LensConfig(
        hf_model_name="tiny",
        checkpoint_name=checkpoint_name,
        artifacts_base_dir=str(tmp_path),
        d_model=d_model,
        num_prompts_trained_on=num_prompts,
    )
    return JacobianLens(jacobians=jacobians_L_dict_FN, config=config)


def make_logit_lens(
    source_layers: list[int],
    tmp_path: Path,
    *,
    checkpoint_name: str = "logit",
    d_model: int = TINY_D_MODEL,
) -> LogitLens:
    """A LogitLens readable at ``source_layers`` (identity transport)."""
    config = LensConfig(
        hf_model_name="tiny",
        checkpoint_name=checkpoint_name,
        lens_type="logit",
        artifacts_base_dir=str(tmp_path),
        d_model=d_model,
    )
    return LogitLens(source_layers=source_layers, config=config)


### THE 8-LAYER "BAND" DECODER AND A RANDOM BAND LENS FOR THE EVAL RUNNERS

# Interior layers of the 8-layer band decoder, read out together by the swap eval
# runner tests.
BAND_LAYERS = [2, 4, 6]


def make_band_decoder(tokenizer: Any, *, vocab_size: int = 32, seed: int = 0) -> TinyDecoder:
    """An 8-layer TinyDecoder (so ``BAND_LAYERS`` are interior layers) with the
    given tokenizer attached."""
    return make_tiny_decoder(tokenizer, n_layers=8, vocab_size=vocab_size, seed=seed)


def make_random_band_jacobian_lens(
    tmp_path: Path,
    *,
    seed: int,
    checkpoint_name: str,
    num_prompts_trained_on: int = 10,
) -> JacobianLens:
    """A J-lens on ``BAND_LAYERS`` whose Jacobians are the identity plus small
    Gaussian noise (``I + 0.05 * N(0, 1)``, drawn from a generator seeded with
    ``seed`` in layer order): a non-trivial but well-conditioned transport for
    the eval runners' brute-force oracles."""
    generator = t.Generator().manual_seed(seed)
    jacobians_L_dict_FN = {
        layer: t.eye(TINY_D_MODEL)
        + 0.05 * t.randn(TINY_D_MODEL, TINY_D_MODEL, generator=generator)
        for layer in BAND_LAYERS
    }
    config = LensConfig(
        hf_model_name="tiny",
        checkpoint_name=checkpoint_name,
        artifacts_base_dir=str(tmp_path),
        d_model=TINY_D_MODEL,
        source_layers=list(BAND_LAYERS),
        num_prompts_trained_on=num_prompts_trained_on,
    )
    return JacobianLens(jacobians=jacobians_L_dict_FN, config=config)


### A ROUTER COLLECTION FITTED ON GAUSSIAN POINTS


def fit_random_router_collection(
    d_model: int,
    *,
    source_layers: list[int],
    num_clusters: int = 2,
    projection_dim: int = 4,
    seed: int = 6,
) -> ActivationRouterCollection:
    """A router per ``source_layers`` fitted on 64 Gaussian points each: the
    routing itself is not under test, it only has to be deterministic."""
    generator = t.Generator().manual_seed(seed)
    pooled_activations_L_dict_PN = {
        layer: t.randn(64, d_model, generator=generator) for layer in source_layers
    }
    router_collection = ActivationRouterCollection(num_clusters, projection_dim=projection_dim)
    router_collection.fit(pooled_activations_L_dict_PN)
    return router_collection


### THE FAKE CONDENSE-EXPERTS PROBLEM (fake tensors in the real dataclasses)

CONDENSE_VOCAB_SIZE = 40
CONDENSE_D_MODEL = 6
CONDENSE_NUM_EXPERTS = 4
CONDENSE_LAYERS = [1, 2]
CONDENSE_NUM_PROMPTS = 12

# Six fake items over two evals; items 0 and 5 have two intermediates, items
# 0, 2 and 4 have multi-candidate intermediates. All ids are below
# CONDENSE_VOCAB_SIZE and disjoint from FAKE_EXCLUDED_TOKEN_IDS.
FAKE_EVAL_SLUGS = ["poetry", "typo", "poetry", "typo", "poetry", "typo"]
FAKE_ITEM_NAMES = [f"item{idx}" for idx in range(6)]
FAKE_CANDIDATE_IDS_BY_ITEM: list[dict[str, list[int]]] = [
    {"a": [3], "b": [5, 6]},
    {"c": [7]},
    {"d": [9, 10, 11]},
    {"e": [12]},
    {"f": [13, 14]},
    {"g": [15], "h": [16]},
]
FAKE_EXCLUDED_TOKEN_IDS = [0, 1, 2, 20, 21]


def make_fake_expert_jacobians(generator: t.Generator, tmp_path: Path) -> ExpertJacobians:
    """Random fp32 experts at ``CONDENSE_LAYERS`` with an expert-fit config."""
    config = LensConfig(
        hf_model_name="tiny",
        checkpoint_name="fake-experts",
        artifacts_base_dir=str(tmp_path),
        source_layers=list(CONDENSE_LAYERS),
        relative_end_transport_layer=-1,
        d_model=CONDENSE_D_MODEL,
        num_prompts_trained_on=CONDENSE_NUM_PROMPTS,
        num_clusters=CONDENSE_NUM_EXPERTS,
        cluster_projection_dim=4,
        lrp_mode="rlens",
    )
    experts_L_dict_EFN = {
        layer: t.randn(CONDENSE_NUM_EXPERTS, CONDENSE_D_MODEL, CONDENSE_D_MODEL, generator=generator)
        for layer in CONDENSE_LAYERS
    }
    return ExpertJacobians(
        experts_L_dict_EFN=experts_L_dict_EFN,
        pooled_jacobian_L_dict_FN={
            layer: experts_EFN.mean(dim=0) for layer, experts_EFN in experts_L_dict_EFN.items()
        },
        position_counts_L_dict_E={
            layer: t.full((CONDENSE_NUM_EXPERTS,), 100, dtype=t.long) for layer in CONDENSE_LAYERS
        },
        weight_sums_L_dict_E={
            layer: t.full((CONDENSE_NUM_EXPERTS,), 100.0) for layer in CONDENSE_LAYERS
        },
        fallback_L_dict_Bool_E={
            layer: t.zeros(CONDENSE_NUM_EXPERTS, dtype=t.bool) for layer in CONDENSE_LAYERS
        },
        config=config,
        min_kept_positions=50,
    )


def make_fake_readout_residuals(generator: t.Generator) -> ReadoutResiduals:
    """Random readout residuals for the six fake items at ``CONDENSE_LAYERS``."""
    return ReadoutResiduals(
        eval_slugs=list(FAKE_EVAL_SLUGS),
        item_names=list(FAKE_ITEM_NAMES),
        residuals_L_dict_PN={
            layer: t.randn(len(FAKE_ITEM_NAMES), CONDENSE_D_MODEL, generator=generator) * 3
            for layer in CONDENSE_LAYERS
        },
        candidate_ids_P_list=[dict(ids) for ids in FAKE_CANDIDATE_IDS_BY_ITEM],
        hf_model_name="tiny",
        max_seq_len=64,
    )


def all_item_expert_transports(
    model: TinyDecoder,
    experts: ExpertJacobians,
    residuals: ReadoutResiduals,
    layer: int,
) -> t.Tensor:
    """``[P, E, F]`` for every recorded item, through the fitter (no ids are
    excluded, so the candidate check cannot fire)."""
    fitter = ExpertWeightingFitter(model, experts, residuals, excluded_token_ids=[])
    return fitter._expert_transports_PEF(layer, t.arange(residuals.num_items))


def make_planted_expert_problem() -> tuple[PairTable, t.Tensor]:
    """Four single-intermediate items; expert 2 alone puts every pair's candidate
    on top, every other expert puts a decoy token on top."""
    generator = t.Generator().manual_seed(3)
    eval_slugs = ["poetry", "poetry", "typo", "typo"]
    candidate_ids_P_list = [{"x": [4]}, {"y": [8, 9]}, {"z": [12]}, {"w": [15]}]
    pairs = build_pair_table(candidate_ids_P_list, eval_slugs, [0, 1, 2, 3], device="cpu")
    logits_PEV = t.randn(4, CONDENSE_NUM_EXPERTS, CONDENSE_VOCAB_SIZE, generator=generator)
    decoy = 30
    for item_idx, candidate in enumerate((4, 9, 12, 15)):
        logits_PEV[item_idx, :, decoy] += 6.0  # every expert loves the decoy...
        logits_PEV[item_idx, 2, candidate] += 9.0  # ...expert 2 loves the candidate more
        logits_PEV[item_idx, 2, decoy] -= 6.0
    return pairs, logits_PEV


### READOUT-EVAL DATA FILES


def write_readout_eval_json(data_dir: Path, slug: str, items: list[dict]) -> None:
    """Write ``lens-eval-<slug>.json`` in the readout-eval data layout."""
    (data_dir / f"lens-eval-{slug}.json").write_text(json.dumps({"items": items}))


def exact_rows_per_source_PFN(
    trainer: LensTrainer, forward_state: FitStepForward, *, dtype: t.dtype = t.float32
) -> dict[int, t.Tensor]:
    """``{layer: [P, F, N]}``: the trainer's own gradient rows at every kept source,
    unreduced, from its backward passes (``_backward_passes``, the all-targets-at-once
    cotangent), cast to ``dtype`` without the fp32 mean ``fit_step`` applies. Row ``p``
    is the estimator's ``G_p``. Consumes the retained graph of ``forward_state``."""
    d_model = trainer.model.d_model
    rows_L_dict_PFN = {
        layer: t.zeros(forward_state.num_source_positions, d_model, d_model, dtype=dtype)
        for layer in trainer.source_layers
    }
    for dim_start, current, grads_L_list_BSN in trainer._backward_passes(forward_state):
        for layer, grad_BSN in zip(trainer.source_layers, grads_L_list_BSN, strict=True):
            rows_BPN = grad_BSN[:current, forward_state.source_positions_Int_P, :].to(dtype)
            rows_L_dict_PFN[layer][:, dim_start : dim_start + current, :] = rows_BPN.permute(
                1, 0, 2
            )
    return rows_L_dict_PFN


def make_tiny_expert_trainer(
    model: TinyDecoder,
    tmp_path: Path,
    checkpoint_name: str,
    routers: dict[int, ActivationRouterCollection],
    **config_overrides: Any,
) -> ExpertJacobianTrainer:
    """An expert-Jacobian trainer over the tiny fit prompts with ``routers`` (``{K: router}``,
    per-layer collections);
    ``config_overrides`` go to :func:`make_tiny_lens_config` (e.g. ``source_layers``)."""
    return ExpertJacobianTrainer(
        make_tiny_lens_config(tmp_path, checkpoint_name, **config_overrides),
        model,
        TINY_FIT_PROMPTS,
        router_collections_K_dict=routers,
    )
