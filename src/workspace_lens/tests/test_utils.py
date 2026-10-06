# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
# Modified by Kola Ayonrinde, 2026.
"""Unembedding extraction, reading lens files in the format of the released
J-Lens and R-Lens files, and the position masks shared by every estimator."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from jlens.tests.tiny import TinyDecoder
from workspace_lens.config import LensConfig
from workspace_lens.utils import (
    check_model_matches_config,
    get_position_mask_with_early_skips,
    get_unembed_matrix,
    load_legacy_format_lens,
    record_activations,
    vocab_size_of,
)


def test_record_activations_runs_the_forward_without_grad() -> None:
    model = TinyDecoder(n_layers=2, d_model=8)
    grad_enabled_during_forward: list[bool] = []
    handle = model.layers[0].register_forward_hook(
        lambda module, args, output: grad_enabled_during_forward.append(torch.is_grad_enabled())
    )
    try:
        _, activations_L_dict_SN = record_activations(model, "a b c", 8, [0, 1])
    finally:
        handle.remove()

    assert grad_enabled_during_forward and not any(grad_enabled_during_forward)
    assert not any(activations_SN.requires_grad for activations_SN in activations_L_dict_SN.values())


def test_vocab_size_of_is_the_lm_head_width() -> None:
    assert vocab_size_of(TinyDecoder(n_layers=2, d_model=8, vocab_size=40)) == 40


def test_check_model_matches_config(tmp_path: Path) -> None:
    """``d_model`` must agree, and so must the HF name when the model exposes
    one — otherwise a same-width model of another identity would pass. A model
    with no HF name (a test model) is checked on ``d_model`` alone."""
    config = LensConfig(
        hf_model_name="org/model-a",
        checkpoint_name="lens",
        artifacts_base_dir=str(tmp_path),
        d_model=8,
    )

    class NamedModel:
        d_model = 8
        _hf_model = SimpleNamespace(config=SimpleNamespace(name_or_path="org/model-a"))

    class OtherNamedModel(NamedModel):
        _hf_model = SimpleNamespace(config=SimpleNamespace(name_or_path="org/model-b"))

    check_model_matches_config(NamedModel(), config)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fitted on 'org/model-a'"):
        check_model_matches_config(OtherNamedModel(), config)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="d_model=8"):
        check_model_matches_config(TinyDecoder(n_layers=2, d_model=4), config)
    check_model_matches_config(TinyDecoder(n_layers=2, d_model=8), config)


def test_get_unembed_matrix() -> None:
    """Extracts the bare lm_head weight (no final norm) from TinyDecoder, and
    fails loudly on models without a recognisable head."""
    model = TinyDecoder(n_layers=4, d_model=8)
    weight_VF = get_unembed_matrix(model)
    assert weight_VF.shape == (32, model.d_model)
    torch.testing.assert_close(weight_VF, model.lm_head.weight.detach())

    class NoHead:
        d_model = 8

    with pytest.raises(TypeError, match="unembedding"):
        get_unembed_matrix(NoHead())  # type: ignore[arg-type]

    class WrongWidthHead:
        d_model = 8
        lm_head = SimpleNamespace(weight=torch.randn(32, 4))

    with pytest.raises(ValueError, match="expected"):
        get_unembed_matrix(WrongWidthHead())  # type: ignore[arg-type]


def test_load_legacy_format_lens(tmp_path: Path) -> None:
    """Converts a file in the released J-Lens format ({"J", "n_prompts",
    "source_layers", "d_model"}, no "config" key) into a JacobianLens that
    transports to the final block."""
    torch.manual_seed(0)
    jacobians = {0: torch.randn(4, 4), 2: torch.randn(4, 4)}
    lens_file = {
        "J": jacobians,
        "n_prompts": 7,
        "source_layers": [0, 2],
        "d_model": 4,
    }
    path = tmp_path / "qwen_lens_n7.pt"
    torch.save(lens_file, str(path))

    lens = load_legacy_format_lens(str(path), hf_model_name="tiny")
    assert lens.source_layers == [0, 2]
    assert lens.d_model == 4
    assert lens.num_prompts_trained_on == 7
    assert lens.relative_end_transport_layer == -1
    assert lens.config.hf_model_name == "tiny"
    assert lens.config.checkpoint_name == "qwen_lens_n7"
    for layer, jacobian_FN in jacobians.items():
        torch.testing.assert_close(lens.jacobians_L_dict_FN[layer], jacobian_FN)


def test_legacy_provenance_sets_lrp_mode_and_target_layer(tmp_path: Path) -> None:
    """The released R-Lens files nest the rule flags under config_json["rules"]
    with an explicit "estimator" field; the reader maps them to the matching
    lrp_mode, and converts the provenance's absolute target_layer to
    relative_end_transport_layer."""
    import json

    def lens_with_provenance(name: str, config_json: str):
        lens_file = {
            "J": {0: torch.zeros(2, 2)},
            "n_prompts": 1,
            "source_layers": [0],
            "d_model": 2,
            "provenance": {"target_layer": 62, "config_json": config_json},
        }
        path = tmp_path / f"{name}.pt"
        torch.save(lens_file, str(path))
        return load_legacy_format_lens(str(path), hf_model_name="tiny", num_layers=64)

    relp_dense = json.dumps(
        {"estimator": "relp", "rules": {"ln_rule": True, "identity_rule": True}}
    )
    rlens = lens_with_provenance("rlens", relp_dense)
    assert rlens.config.lrp_mode == "rlens"
    assert rlens.relative_end_transport_layer == -2  # block 62 of 64

    relp_moe = json.dumps(
        {"estimator": "relp", "rules": {"ln_rule": True, "routed_experts": True}}
    )
    assert lens_with_provenance("rmoe", relp_moe).config.lrp_mode == "r+moe"

    standard = json.dumps({"estimator": "standard"})
    assert lens_with_provenance("std", standard).config.lrp_mode == "none"


### POSITION MASKS


def test_position_mask_skips_the_first_positions_and_the_last() -> None:
    """The valid positions are every position after the first
    ``skip_first_n_positions`` except the final one; a prompt left with none is
    a "too short" error."""
    mask_Bool_S = get_position_mask_with_early_skips(8, 2)
    assert mask_Bool_S.dtype == torch.bool
    assert mask_Bool_S.tolist() == [False, False, True, True, True, True, True, False]

    with pytest.raises(ValueError, match="too short"):
        get_position_mask_with_early_skips(3, 2)
