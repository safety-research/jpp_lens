# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
# Modified by Kola Ayonrinde, 2026.
"""Shared helpers.

This module is deliberately a *leaf* of the package import graph (it imports
only ``jlens``, ``config``, ``residual_streams`` and third-party packages), so
any module, including those the lenses import, can use it at the top of the file
without an import cycle. Lens classes are imported under ``TYPE_CHECKING`` or at
call time only.

Sections: which prompt positions a fit uses; recording residuals
and reading lenses out; introspecting a ``LensModel`` (names, sizes, the unembedding
matrix, token ids); J-lens vectors for one token; reading the released J-Lens and
R-Lens files; file-system helpers.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch as t
from transformers import AutoConfig

from jlens.hooks import ActivationRecorder
from jlens.protocol import LensModel
from workspace_lens.config import LensConfig
from workspace_lens.residual_streams import (
    collapse_streams_like_the_model,
    residual_mean_over_streams,
)

if TYPE_CHECKING:
    from workspace_lens.lenses.base_lens import BaseLens
    from workspace_lens.lenses.jacobian_lens import JacobianLens

logger = logging.getLogger(__name__)


### WHICH PROMPT POSITIONS A FIT USES


def get_position_mask_with_early_skips(seq_len: int, skip_first_n_positions: int) -> t.Tensor:
    """Boolean mask (``[seq_len]``) over sequence positions included in every
    estimator.

    The single definition of "valid position", shared by Jacobian fitting and
    activation clustering: the first
    ``skip_first_n_positions`` positions are excluded (attention sinks with
    atypical residual statistics) and so is the final position. The estimator
    itself could use the final position; it is excluded for parity with the
    original jacobian-lens convention and with next-token evaluation, which
    has no target there.

    Raises:
        ValueError: If the prompt is too short to leave any valid positions.
    """
    mask_Bool_S = t.zeros(seq_len, dtype=t.bool)
    mask_Bool_S[skip_first_n_positions : seq_len - 1] = True

    if mask_Bool_S.sum() == 0:
        raise ValueError(
            f"prompt too short: seq_len={seq_len}, need > {skip_first_n_positions + 1} tokens"
        )

    return mask_Bool_S


def _token_id_bytes(input_ids_Int_S: t.Tensor) -> bytes:
    """A prompt's token ids as int64 bytes: the prompt identity behind :func:`token_ids_sha256`."""
    return input_ids_Int_S.detach().cpu().to(t.int64).numpy().tobytes()


def token_ids_sha256(input_ids_Int_S: t.Tensor) -> str:
    """SHA-256 hex digest of a prompt's token ids (int64 bytes): the prompt identity
    a fit's position records store."""
    return hashlib.sha256(_token_id_bytes(input_ids_Int_S)).hexdigest()


### RECORDING RESIDUALS


def record_activations(
    model: LensModel,
    prompt: str,
    max_seq_len: int,
    layers_to_record: list[int],
) -> tuple[t.Tensor, dict[int, t.Tensor]]:
    """One forward pass on ``prompt``; the lens's residual at each requested
    layer as ``{layer: [seq_len, d_model]}``, plus the input ids ``[1, S]``.

    Uses no lens state, so multi-lens evaluators can record once and share the
    residuals across lenses.
    """
    input_ids_Int_1S, block_outputs_L_dict_1SrN = record_block_outputs(model, prompt, max_seq_len, layers_to_record)
    # A multi-stream block output (DeepSeek-V4) is read as its stream mean, the lens's residual.
    activations_L_dict_SN = {
        layer: residual_mean_over_streams(block_output_1SrN)[0]
        for layer, block_output_1SrN in block_outputs_L_dict_1SrN.items()
    }
    return input_ids_Int_1S, activations_L_dict_SN


def record_block_outputs(
    model: LensModel,
    prompt: str,
    max_seq_len: int,
    layers_to_record: list[int],
) -> tuple[t.Tensor, dict[int, t.Tensor]]:
    """One forward pass on ``prompt``; each requested block's raw output
    ``{layer: [1, S, N]}`` (``[1, S, R, N]`` on a multi-stream model), plus the input
    ids ``[1, S]``. Runs under ``no_grad``: the recordings are read-only inputs to
    evals and fits, never differentiated through, so building the autograd graph
    would only cost memory. :func:`record_activations` reduces these to the lens's
    residual."""
    input_ids_Int_1S = model.encode(prompt, max_length=max_seq_len)
    with t.no_grad(), ActivationRecorder(model.layers, at=layers_to_record) as recorder:
        _ = model.forward(input_ids_Int_1S)
        # [1, S, N], or [1, S, R, N] on a multi-stream model; callers collapse the streams themselves (the lens's
        # mean, or the model's own head), so the axis is kept here.
        block_outputs_L_dict_1SrN = {layer: recorder.activations[layer] for layer in layers_to_record}
    return input_ids_Int_1S, block_outputs_L_dict_1SrN


def final_position_model_logits(model: LensModel, prompt: str, max_seq_len: int) -> t.Tensor:
    """The model's own next-token logits at the last prompt position, fp32 on
    the CPU: one forward pass, the final block's last residual through the
    unembedding. On a multi-stream model the final block's streams are collapsed
    by the model's own head first (:func:`collapse_streams_like_the_model`), so
    these are the model's logits and not the lens's stream-mean readout."""
    final_layer = model.n_layers - 1
    _, block_outputs_L_dict_1SrN = record_block_outputs(model, prompt, max_seq_len, [final_layer])
    final_residual_F = collapse_streams_like_the_model(model, block_outputs_L_dict_1SrN[final_layer])[0, -1]
    return model.unembed(final_residual_F.float()[None, :])[0].float().cpu()


def masked_lens_logits(
    model: LensModel,
    lens: BaseLens,
    residual_PN: t.Tensor,
    layer: int,
    excluded_mask_Bool_V: t.Tensor | None,
) -> t.Tensor:
    """``unembed(transport(h))`` at every position, fp32 on the residual's
    device, with the excluded ids (if any) forced to ``-inf`` — the one
    readout every ranking eval scores."""
    logits_PV = model.unembed(lens.transport(residual_PN, layer)).float()
    if excluded_mask_Bool_V is not None:
        logits_PV = logits_PV.masked_fill(
            excluded_mask_Bool_V.to(logits_PV.device), float("-inf")
        )
    return logits_PV


def check_layers_fitted(
    lens: BaseLens, layers: Sequence[int], *, lens_name: str | None = None
) -> None:
    """Raise ``ValueError`` unless every layer in ``layers`` is one of the
    lens's ``source_layers`` (``lens_name`` names the lens in the message when
    several are being checked)."""
    unknown = sorted(set(layers) - set(lens.source_layers))
    if unknown:
        which_lens = f" of lens {lens_name!r}" if lens_name is not None else ""
        raise ValueError(
            f"layers {unknown} not in source_layers{which_lens}; "
            f"fitted layers are {lens.source_layers}"
        )


### MODEL INTROSPECTION: NAMES, SIZES, THE UNEMBEDDING, TOKEN IDS


def model_device_of(model: LensModel) -> t.device | str:
    """The device the model takes its inputs on (``input_device`` on the HF
    adapter; the CPU for models that don't say)."""
    return getattr(model, "input_device", t.device("cpu"))


def hf_model_name_of(model: LensModel) -> str | None:
    """Best-effort HF name of a wrapped model; ``None`` if it has none
    (e.g. test models)."""
    hf_config = getattr(getattr(model, "_hf_model", None), "config", None)
    # "" on models built from a config rather than a checkpoint: no name either.
    return getattr(hf_config, "name_or_path", None) or None


def check_model_matches_config(model: LensModel, config: LensConfig) -> None:
    """Raise ``ValueError`` unless ``model`` is the model ``config`` was fitted
    on: ``d_model`` must agree and, where the model exposes an HF name
    (:func:`hf_model_name_of`), so must ``hf_model_name``. A model with no HF
    name (a test model) passes on ``d_model`` alone, with a warning. The one
    check for pairing a saved artifact (checkpoint, experts, lens) with a
    freshly loaded model: a same-width model of another identity would
    otherwise be accepted silently."""
    if model.d_model != config.d_model:
        raise ValueError(
            f"model has d_model={model.d_model} but {config.checkpoint_name!r} was "
            f"fitted with d_model={config.d_model}"
        )
    model_name = hf_model_name_of(model)
    if model_name is None:
        logger.warning(
            "model exposes no HF name; cannot verify it matches %r (d_model matches)",
            config.hf_model_name,
        )
    elif model_name != config.hf_model_name:
        raise ValueError(
            f"model is {model_name!r} but {config.checkpoint_name!r} was fitted on "
            f"{config.hf_model_name!r}"
        )


def resolve_target_layer(num_layers: int, relative_end_transport_layer: int) -> int:
    """The absolute block index a lens transports to: ``num_layers +
    relative_end_transport_layer`` (``-1`` is the final block).

    Raises:
        ValueError: If the result is not a layer of the model.
    """
    target_layer = num_layers + relative_end_transport_layer
    if not 0 <= target_layer < num_layers:
        raise ValueError(f"target_layer={target_layer} out of range for {num_layers} layers")
    return target_layer


def num_hidden_layers(hf_model_name: str) -> int:
    """The block count of ``hf_model_name`` from its HF config (the text
    config on multimodal models); downloads no weights."""

    return int(AutoConfig.from_pretrained(hf_model_name).get_text_config().num_hidden_layers)


@t.no_grad()
def vocab_size_of(model: LensModel) -> int:
    """The LM-head width — the ``vocab_size`` every exclusion mask spans
    (wider than the tokenizer's vocabulary on padded heads: 248 320 on
    Qwen3.6-27B), probed with one zero residual through the unembedding because the
    ``LensModel`` protocol exposes no ``vocab_size``."""
    return int(model.unembed(t.zeros(1, model.d_model, device=model_device_of(model))).shape[-1])


def token_id_mask_Bool_V(token_ids: Iterable[int], vocab_size: int) -> t.Tensor:
    """A ``[vocab_size]`` boolean mask that is True at ``token_ids`` (CPU)."""
    mask_Bool_V = t.zeros(vocab_size, dtype=t.bool)
    unique_token_ids = sorted({int(token_id) for token_id in token_ids})
    if unique_token_ids:
        mask_Bool_V[unique_token_ids] = True
    return mask_Bool_V


def get_unembed_matrix(model: LensModel) -> t.Tensor:
    """Return the bare unembedding weight ``W_U`` as ``[vocab_size, d_model]``.

    The ``LensModel`` protocol only exposes ``unembed()`` (final norm + LM
    head as a black box), so this reaches for the head module directly:
    ``lm_head`` on :class:`jlens.tests.tiny.TinyDecoder`, ``_lm_head`` on the
    HF adapter. The final norm is deliberately *not* included.

    Raises:
        TypeError: If no LM head attribute with a weight is found.
        ValueError: If the weight's width disagrees with ``model.d_model``.
    """
    for attribute_name in ("lm_head", "_lm_head"):
        lm_head_module = getattr(model, attribute_name, None)
        weight_VF = getattr(lm_head_module, "weight", None)

        if weight_VF is None:
            continue
        if weight_VF.ndim != 2 or weight_VF.shape[1] != model.d_model:
            raise ValueError(
                f"{attribute_name}.weight has shape {tuple(weight_VF.shape)}; expected "
                f"[vocab_size, d_model={model.d_model}]"
            )

        return weight_VF.detach()

    raise TypeError(
        f"cannot locate an unembedding matrix on {type(model).__name__}: "
        "expected an `lm_head` or `_lm_head` attribute with a `.weight`"
    )


def resolve_token_id(model: LensModel, token: str | int) -> int:
    """Resolve a concept token to a vocab id.

    An ``int`` is range-checked against :func:`vocab_size_of` and returned
    as-is. A ``str`` is tokenized with ``model.encode``; a leading
    BOS token (when the tokenizer exposes ``bos_token_id``) is stripped, and
    the result must be exactly one token.

    Raises:
        ValueError: If an int id is out of range, or a string does not encode
            to exactly one non-BOS token.
    """
    vocab_size = vocab_size_of(model)

    if isinstance(token, int):
        if not 0 <= token < vocab_size:
            raise ValueError(f"token id {token} out of range for vocab size {vocab_size}")
        return token

    token_ids: list[int] = model.encode(token)[0].tolist()
    bos_token_id = getattr(model.tokenizer, "bos_token_id", None)

    if bos_token_id is not None and token_ids and token_ids[0] == bos_token_id:
        token_ids = token_ids[1:]

    if len(token_ids) != 1:
        raise ValueError(
            f"{token!r} encodes to {len(token_ids)} tokens {token_ids}; "
            "interventions and probes need a single-token concept"
        )
    return token_ids[0]


### J-LENS VECTORS FOR ONE TOKEN


def jlens_vectors(
    lens: BaseLens,
    model: LensModel,
    token: str | int,
    *,
    layers: Sequence[int] | None = None,
) -> dict[int, t.Tensor]:
    """Per-layer J-lens vectors for one vocab token: ``{layer: v_N}``.

    ``v_N = W_U[token_id].float() @ J_layer`` — the ``token_id`` row of the
    J-space dictionary ``W_U @ J_l``, computed without materialising the full
    ``[vocab, d_model]`` dictionary. By construction ``<h, v>`` is the token's
    pre-final-norm lens logit at ``layer``: residuals transport forward with
    ``h @ J.T`` (``JacobianLens.transport``), so the unembedding row pulls *back*
    with ``row @ J`` — no transpose. Returned fp32, on each Jacobian's device.

    ``layers`` defaults to ``lens.source_layers`` and must be a subset of it.
    """
    layers, unembed_row_F = _readout_layers_and_unembed_row(lens, model, token, layers)
    assert lens.jacobians_L_dict_FN is not None, "Jacobian dictionary is not initialized."
    jvectors_L_dict_N: dict[int, t.Tensor] = {}
    for layer in layers:
        jacobian_FN = lens.jacobians_L_dict_FN[layer]
        jvectors_L_dict_N[layer] = unembed_row_F.to(jacobian_FN.device) @ jacobian_FN

    return jvectors_L_dict_N


def readout_vectors(
    lens: BaseLens,
    model: LensModel,
    token: str | int,
    *,
    layers: Sequence[int] | None = None,
) -> dict[int, t.Tensor]:
    """Per-layer **pre-final-norm** readout vectors for one token, for a
    Jacobian-family lens or the logit lens.

    For a Jacobian-family lens this is :func:`jlens_vectors`
    (``v_l = W_U[token] @ J_l``). For the logit lens (identity transport) the
    direction at every layer is the unembedding row ``W_U[token]``; swapping
    along these rows is the natural null baseline for the causal swap evals
    ("does the model use the raw unembedding direction the way it uses the
    Jacobian-transported one?").

    Convention: these are directions for the token's *pre-final-norm* lens
    logit. The model's actual unembedding is ``lm_head(final_norm(h))`` and the
    final RMSNorm carries a per-dimension gain ``g``, so the direction whose inner
    product with ``h`` gives the true logit is ``g * W_U[token]`` (and
    ``(g * W_U[token]) @ J_l``), not ``W_U[token]``; these vectors omit ``g``.
    Ranks are unaffected by the scalar ``1/rms(h)``; the omission of ``g`` is the
    only deviation.

    ``layers`` defaults to ``lens.source_layers`` and must be a subset of it.
    """
    if lens.jacobians_L_dict_FN is not None:
        return jlens_vectors(lens, model, token, layers=layers)

    layers, unembed_row_F = _readout_layers_and_unembed_row(lens, model, token, layers)
    return {layer: unembed_row_F.clone() for layer in layers}


def _readout_layers_and_unembed_row(
    lens: BaseLens, model: LensModel, token: str | int, layers: Sequence[int] | None
) -> tuple[Sequence[int], t.Tensor]:
    """The shared prologue of :func:`jlens_vectors` and :func:`readout_vectors`:
    ``layers`` defaulted to the lens's source layers and checked against them,
    and the token's unembedding row ``W_U[token]`` as fp32."""
    if layers is None:
        layers = lens.source_layers
    check_layers_fitted(lens, layers)
    token_id = resolve_token_id(model, token)
    unembed_row_F = get_unembed_matrix(model)[token_id].detach().float()
    return layers, unembed_row_F


### READING THE RELEASED J-LENS AND R-LENS FILES


def load_legacy_format_lens(
    lens_path: str, hf_model_name: str, *, num_layers: int | None = None
) -> JacobianLens:
    """Build a :class:`JacobianLens` from a lens file in the format of the
    released J-Lens and R-Lens files on the Hugging Face Hub.

    The format stores ``{"J", "n_prompts", "source_layers", "d_model"}`` with no
    embedded config or model name, so the caller supplies ``hf_model_name``.
    ``BaseLens.load`` cannot read these files (no ``"config"`` key); this
    converter is the path for e.g. the Hub-hosted Qwen lens
    (``neuronpedia/jacobian-lens``).

    The released R-Lens files (e.g. ``camilablank/workspace-lenses``)
    additionally embed a ``"provenance"`` dict. Its ``target_layer`` (absolute
    block index, e.g. 62) is converted to ``relative_end_transport_layer``
    using ``num_layers`` — fetched from the HF config of ``hf_model_name`` when
    not passed — because those files target the penultimate block, not ``-1``.
    Its ``config_json`` LRP flags label the lens's ``lrp_mode`` ("rlens" for the
    minimal RelP rule set).
    """
    from workspace_lens.lenses.jacobian_lens import JacobianLens  # leaf-module rule

    checkpoint = t.load(lens_path, map_location="cpu", weights_only=True)
    provenance = checkpoint.get("provenance")

    if provenance is not None and "target_layer" in provenance:
        if num_layers is None:
            num_layers = num_hidden_layers(hf_model_name)
        relative_end_transport_layer = int(provenance["target_layer"]) - num_layers
    else:
        relative_end_transport_layer = -1  # files without provenance target the final block

    config = LensConfig(
        hf_model_name=hf_model_name,
        checkpoint_name=Path(lens_path).stem,
        lens_type="jacobian",  # J-Lens and R-Lens files both hold one Jacobian per layer
        source_layers=checkpoint["source_layers"],
        d_model=checkpoint["d_model"],
        num_prompts_trained_on=checkpoint["n_prompts"],
        relative_end_transport_layer=relative_end_transport_layer,
        lrp_mode=_legacy_provenance_lrp_mode(provenance),
    )
    return JacobianLens(jacobians=checkpoint["J"], config=config)


def load_lens_file(path: str, *, hf_model_name: str) -> BaseLens:
    """Load a lens file of either format: one written by this library
    (:meth:`~workspace_lens.lenses.base_lens.BaseLens.load`), or one in the format
    of the released J-Lens and R-Lens files on the Hugging Face Hub, which embed no
    config (:func:`load_legacy_format_lens`, labelled with ``hf_model_name``)."""
    from workspace_lens.lenses.base_lens import BaseLens  # leaf-module rule

    keys = t.load(path, map_location="cpu", weights_only=True, mmap=True).keys()
    if "config" in keys:
        return BaseLens.load(path)
    if "J" in keys:
        return load_legacy_format_lens(path, hf_model_name)
    raise ValueError(f"{path} is not a lens file (keys {sorted(keys)!r})")


def _legacy_provenance_lrp_mode(provenance: dict | None) -> str:
    """Best-effort ``lrp_mode`` label from the provenance dict of a released
    R-Lens or J-Lens file (e.g. ``camilablank/workspace-lenses``). Its
    ``config_json`` is ``{"estimator": "relp"|"standard", "rules": {...flags...}}``
    (the rule flags are nested under ``"rules"``). Mapping: RelP estimator ->
    "rlens" (the minimal RelP rule set), or "r+moe" when the routed-expert knobs
    are on; standard estimator / missing / unparseable -> "none". (A
    standard-estimator file fitted with a shared-expert scale matches no preset
    here and also labels "none"; set the label from the caller if it matters.)"""
    if provenance is None:
        return "none"
    try:
        payload = json.loads(provenance.get("config_json", "{}"))
    except (TypeError, ValueError):
        return "none"
    if not isinstance(payload, dict) or payload.get("estimator") != "relp":
        return "none"
    rules = payload.get("rules")
    if isinstance(rules, dict) and rules.get("routed_experts"):
        return "r+moe"
    return "rlens"


### FILES


def ensure_parent_dir(path: str | Path) -> None:
    """Create the directory ``path`` will be written into (``mkdir -p``)."""
    Path(path).absolute().parent.mkdir(parents=True, exist_ok=True)
