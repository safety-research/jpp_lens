# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
# Modified by Kola Ayonrinde, 2026.
"""JacobianLens apply and from_pretrained on the tiny CPU model."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch

from jlens.tests.tiny import TinyDecoder
from workspace_lens.config import LensConfig
from workspace_lens.lenses.base_lens import BaseLens, LensParameters
from workspace_lens.lenses.jacobian_lens import JacobianLens
from workspace_lens.lenses.logit_lens import LogitLens
from workspace_lens.tests.fixtures import fit_tiny_jacobian_lens, make_jacobian_lens


@pytest.fixture(scope="module")
def model() -> TinyDecoder:
    return TinyDecoder(n_layers=4, d_model=8)


@pytest.fixture(scope="module")
def lens(model: TinyDecoder, tmp_path_factory: pytest.TempPathFactory) -> JacobianLens:
    """A lens fitted on the module-scoped tiny model (layers 0-2)."""
    return fit_tiny_jacobian_lens(model, tmp_path_factory.mktemp("artifacts"), "test-lenses")


def test_apply_round_trip(model: TinyDecoder, lens: JacobianLens, tmp_path: Path) -> None:
    """apply() on a saved-and-reloaded lens returns correctly shaped logits,
    honours ``token_positions_for_residuals`` (including negative indices),
    and rejects unfitted or out-of-range layers.

    The strongest check: TinyDecoder's last block is exactly linear, so the
    transported readout from layer 2 must equal the model's own logits — the
    lens map is exact, not approximate, on this model.
    """
    path = str(tmp_path / "lens.pt")
    lens.save(path)
    reloaded = JacobianLens.load(path)
    assert reloaded.source_layers == [0, 1, 2]
    assert reloaded.num_prompts_trained_on == 2

    lens_logits_dict_SV, model_logits_SV, input_ids_1S = reloaded.apply(
        model, "the quick brown fox jumps", layers=[0, 2]
    )
    assert set(lens_logits_dict_SV) == {0, 2}
    vocab_size = model.lm_head.out_features
    seq_len = input_ids_1S.shape[1]
    # token_positions_for_residuals=None -> every position.
    assert model_logits_SV.shape == (seq_len, vocab_size)
    for logits_SV in lens_logits_dict_SV.values():
        assert logits_SV.shape == (seq_len, vocab_size)
    # Exactly linear lens map: transported readout == model logits (atol: fp16 save).
    torch.testing.assert_close(lens_logits_dict_SV[2], model_logits_SV, rtol=0, atol=1e-2)

    # Explicit positions (negative indices allowed) -> that many rows, in order.
    sub_logits_dict_SV, sub_model_logits_SV, _ = reloaded.apply(
        model,
        "the quick brown fox jumps",
        layers=[0, 2],
        token_positions_for_residuals=[0, -1],
    )
    assert sub_model_logits_SV.shape == (2, vocab_size)
    torch.testing.assert_close(sub_model_logits_SV[1], model_logits_SV[-1])
    for layer in [0, 2]:
        assert sub_logits_dict_SV[layer].shape == (2, vocab_size)
        torch.testing.assert_close(sub_logits_dict_SV[layer][0], lens_logits_dict_SV[layer][0])

    # Unfitted layer is rejected.
    with pytest.raises(ValueError, match="not in source_layers"):
        reloaded.apply(model, "x" * 30, layers=[3])
    # Out-of-range layers are rejected.
    with pytest.raises(ValueError, match="out of range"):
        reloaded.apply(model, "x" * 30, layers=[99])


def test_from_pretrained_local(tmp_path: Path) -> None:
    """from_pretrained() resolves all three local layouts: a direct file
    path, a directory containing ``lens.pt``, and a directory with the lens
    at a subpath (Hub-repo style, one repo hosting lenses for many models)."""
    lens = make_jacobian_lens(
        {0: torch.randn(6, 6), 1: torch.randn(6, 6)}, tmp_path, num_prompts=3, d_model=6
    )
    # File path -> load() directly.
    single = tmp_path / "single.pt"
    lens.save(str(single))
    for layer in [0, 1]:
        torch.testing.assert_close(
            JacobianLens.from_pretrained(str(single)).jacobians_L_dict_FN[layer],
            lens.jacobians_L_dict_FN[layer],
            rtol=0,
            atol=2e-3,
        )  # fp16 round-trip
    # Directory containing lens.pt.
    one_dir = tmp_path / "one"
    one_dir.mkdir()
    lens.save(str(one_dir / "lens.pt"))
    assert JacobianLens.from_pretrained(str(one_dir)).num_prompts_trained_on == 3
    # filename= may be a subpath inside the directory.
    deep = tmp_path / "hubrepo"
    sub = deep / "gemma-2-27b" / "jlens" / "wikitext"
    sub.mkdir(parents=True)
    lens.save(str(sub / "lens.pt"))
    reloaded = JacobianLens.from_pretrained(
        str(deep), filename="gemma-2-27b/jlens/wikitext/lens.pt"
    )
    assert reloaded.num_prompts_trained_on == 3 and reloaded.d_model == 6


def test_check_compatible_reports_the_most_fundamental_disagreement() -> None:
    """When several fields disagree at once, ``check_compatible`` reports the most
    fundamental one first: ``d_model`` before the model name and the layers."""
    config = LensConfig(hf_model_name="tiny", checkpoint_name="a", d_model=4, source_layers=[0])
    other = LensConfig(hf_model_name="other", checkpoint_name="b", d_model=6, source_layers=[1])
    with pytest.raises(ValueError, match="d_model"):
        config.check_compatible(other)
    with pytest.raises(ValueError, match="hf_model_name"):
        config.check_compatible(dataclasses.replace(other, d_model=4))


### GENERIC PARAMETERS: PER-LENS-TYPE CONSTRUCTION, SERIALISATION, DISPATCH


def test_save_load_dispatches_on_lens_type(tmp_path: Path) -> None:
    """BaseLens.load reconstructs the concrete class saved in config.lens_type;
    loading with a mismatched concrete class is rejected."""
    jacobian_lens = make_jacobian_lens(
        {0: torch.eye(4) * 2.0, 2: torch.eye(4)}, tmp_path, num_prompts=1, d_model=4
    )
    jacobian_path = str(tmp_path / "jacobian.pt")
    jacobian_lens.save(jacobian_path)
    loaded_jacobian = BaseLens.load(jacobian_path)
    assert isinstance(loaded_jacobian, JacobianLens)
    assert loaded_jacobian.source_layers == [0, 2]

    logit_lens = LogitLens(
        [1, 3],
        config=LensConfig(
            hf_model_name="tiny", checkpoint_name="logit-mini", lens_type="logit"
        ),
    )
    logit_path = str(tmp_path / "logit.pt")
    logit_lens.save(logit_path)
    loaded_logit = BaseLens.load(logit_path)
    assert isinstance(loaded_logit, LogitLens)
    assert loaded_logit.source_layers == [1, 3]  # survives without parameters

    with pytest.raises(ValueError, match="holds a 'logit' lens"):
        JacobianLens.load(logit_path)


def test_load_reads_a_top_level_j_dict_with_a_config(tmp_path: Path) -> None:
    """A file with a top-level "J" dict and a config loads as a J-lens."""
    jacobians = {0: torch.eye(4), 1: torch.eye(4) * 2.0}
    config = LensConfig(hf_model_name="tiny", checkpoint_name="top-level-j", d_model=4)
    path = str(tmp_path / "top_level_j.pt")
    torch.save({"J": jacobians, "config": config.to_dict()}, path)

    loaded = BaseLens.load(path)
    assert isinstance(loaded, JacobianLens)
    assert loaded.source_layers == [0, 1]
    torch.testing.assert_close(loaded.jacobians_L_dict_FN[1], jacobians[1])


def test_constructor_validation() -> None:
    """A lens rejects a config carrying a different lens_type, and parameter
    dicts must cover exactly the source layers."""
    logit_config = LensConfig(
        hf_model_name="tiny", checkpoint_name="mismatch", lens_type="logit"
    )
    with pytest.raises(ValueError, match="lens_type"):
        JacobianLens(jacobians={0: torch.eye(2)}, config=logit_config)

    with pytest.raises(ValueError, match="jacobians covers layers"):
        LensParameters(
            _source_layers=[0, 1],
            _jacobians_L_dict_FN={0: torch.eye(2)},  # layer 1 missing
        )


def test_lens_parameters_normalisation() -> None:
    """LensParameters derives source_layers from the jacobian keys and requires
    one of the two."""
    parameters = LensParameters(
        _jacobians_L_dict_FN={2: torch.eye(3), 0: torch.eye(3)},
    )
    assert parameters.source_layers == [0, 2]

    with pytest.raises(ValueError, match="source_layers or jacobians"):
        LensParameters()


def test_move_parameters_to_device_updates_the_suffixed_attributes(tmp_path: Path) -> None:
    """move_parameters_to_device rebuilds the suffixed attributes in place and
    creates no unsuffixed ones."""
    lens = make_jacobian_lens({0: torch.eye(4) * 2.0}, tmp_path, num_prompts=1, d_model=4)
    jacobian_before_FN = lens.jacobians_L_dict_FN[0]

    lens.move_parameters_to_device("cpu")

    torch.testing.assert_close(lens.jacobians_L_dict_FN[0], jacobian_before_FN)
    assert not hasattr(lens, "jacobians")
