# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
# Modified by Kola Ayonrinde, 2026.
"""LensTrainer end-to-end on the tiny CPU model: position masking, the
Jacobian estimator itself, layer-index validation, fit, checkpoint/resume,
and save/load."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from jlens.tests.tiny import TinyDecoder
from workspace_lens.fitting.jacobian_fitting import LensTrainer
from workspace_lens.lenses.jacobian_lens import JacobianLens
from workspace_lens.tests.fixtures import make_tiny_lens_config

PROMPTS = [
    "abcdefghij " * 5,
    "klmnopqrst " * 5,
    "uvwxyzabcd " * 5,
    "efghijklmn " * 5,
]


@pytest.fixture(scope="module")
def model() -> TinyDecoder:
    return TinyDecoder(n_layers=4, d_model=8)


def test_position_mask_basic(model: TinyDecoder, tmp_path: Path) -> None:
    """The fit forward reads the rows at, and places the cotangent at, every
    position except the first ``skip_first_n_positions`` (attention sinks) and
    the final one (no next-token target)."""
    trainer = LensTrainer(
        make_tiny_lens_config(tmp_path, "mask", skip_first_n_positions=4), model, prompts=[]
    )
    prompt = "abcdefghij " * 3
    seq_len = model.encode(prompt, max_length=64).shape[1]
    forward_state = trainer._run_fit_forward(prompt)
    expected_positions_Int_P = torch.arange(4, seq_len - 1)
    assert torch.equal(forward_state.source_positions_Int_P.cpu(), expected_positions_Int_P)
    assert forward_state.num_source_positions == seq_len - 4 - 1


def test_position_mask_too_short(model: TinyDecoder, tmp_path: Path) -> None:
    """A prompt with no valid positions after masking raises (so fit() skips
    it) instead of silently fitting on an empty average."""
    trainer = LensTrainer(
        make_tiny_lens_config(tmp_path, "mask-short", skip_first_n_positions=8),
        model,
        prompts=[],
    )
    with pytest.raises(ValueError, match="too short"):
        trainer._run_fit_forward("abcd")


def test_position_records_survive_checkpoint(model: TinyDecoder, tmp_path: Path) -> None:
    """The position records round-trip through a checkpoint resume and match the
    JSON sidecar beside the checkpoints."""
    config = make_tiny_lens_config(tmp_path, "records-ckpt", checkpoint_every_n_prompts=2)
    LensTrainer(config, model, prompts=PROMPTS[:2]).fit()
    resumed = LensTrainer.from_checkpoint(
        os.path.join(config.checkpoint_path, "2_checkpoint.pt"), PROMPTS, model=model
    )
    assert [record["prompt_idx"] for record in resumed.position_records] == [0, 1]
    for record in resumed.position_records:
        assert set(record) == {
            "prompt_idx",
            "token_ids_sha256",
            "seq_len",
            "num_source_positions",
            "source_positions",
        }
        expected_positions = list(range(config.skip_first_n_positions, record["seq_len"] - 1))
        assert record["source_positions"] == expected_positions
        assert record["num_source_positions"] == len(expected_positions)
    # The same records as a JSON sidecar beside the checkpoints.
    with open(os.path.join(config.checkpoint_path, "position_records.json")) as handle:
        sidecar_records = json.load(handle)
    assert sidecar_records == resumed.position_records


def test_fit_step_exact_jacobian(tmp_path: Path) -> None:
    """The estimator computes the true input-output Jacobian, in the right
    orientation, with the right shapes.

    TinyDecoder's blocks are ``h + c*W*h`` (position-wise linear), so the
    Jacobian through the last block alone is known in closed form:
    ``J_2 == I + W_3`` exactly. That equality pins down transposition,
    row/column ordering, and layer indexing — the self-consistency tests
    below (resume) would all pass even if the estimator were wrong.

    Runs with every parameter ``requires_grad=False``: the recorder's
    ``start_graph_at`` must root the autograd graph itself.
    """
    model = TinyDecoder(n_layers=4, d_model=8)
    for param in model.parameters():
        param.requires_grad_(False)
    trainer = LensTrainer(make_tiny_lens_config(tmp_path, "exact"), model, prompts=[])

    prompt = "the quick brown fox " * 4  # > skip_first_n_positions chars
    jacobians_L_dict_FN, seq_len, n_valid = trainer.fit_step(prompt)

    assert set(jacobians_L_dict_FN) == {0, 1, 2}
    for jacobian_FN in jacobians_L_dict_FN.values():
        assert jacobian_FN.shape == (8, 8) and jacobian_FN.dtype == torch.float32
    assert n_valid > 0 and seq_len > n_valid
    # Residual block is h + 0.1*W*h, so J_{n_layers-2} = I + 0.1*W -> diag ~= 1.
    diag_late_F = jacobians_L_dict_FN[2].diag()
    assert (diag_late_F - 1.0).abs().max() < 0.2
    # Earlier layers compound through more blocks -> further from identity.
    assert (jacobians_L_dict_FN[0] - torch.eye(8)).norm() > (
        jacobians_L_dict_FN[2] - torch.eye(8)
    ).norm()
    # Block 3 is h + W_3 h, so J_2 == I + W_3 exactly — pins orientation/indexing.
    expected_J2_FN = torch.eye(8) + model.layers[3].linear.weight.detach()
    torch.testing.assert_close(jacobians_L_dict_FN[2], expected_J2_FN, rtol=0, atol=1e-5)


def test_negative_source_layers_normalised(model: TinyDecoder, tmp_path: Path) -> None:
    """Negative source-layer indices count from the end and produce exactly
    the same Jacobians as their positive equivalents, and the fitted lens
    reports the normalised (non-negative) indices."""
    prompt = "the quick brown fox " * 4
    trainer_neg = LensTrainer(
        make_tiny_lens_config(tmp_path, "neg", source_layers=[-4, -3]), model, prompts=[]
    )
    trainer_pos = LensTrainer(
        make_tiny_lens_config(tmp_path, "pos", source_layers=[0, 1]), model, prompts=[]
    )
    assert trainer_neg.source_layers == [0, 1]
    jac_neg_L_dict_FN, _, _ = trainer_neg.fit_step(prompt)
    jac_pos_L_dict_FN, _, _ = trainer_pos.fit_step(prompt)
    for layer in (0, 1):
        torch.testing.assert_close(jac_neg_L_dict_FN[layer], jac_pos_L_dict_FN[layer])

    lens = LensTrainer(
        make_tiny_lens_config(tmp_path, "neg-fit", source_layers=[-4]), model, prompts=[prompt]
    ).fit()
    assert lens.source_layers == [0]


def test_out_of_range_layers_rejected(model: TinyDecoder, tmp_path: Path) -> None:
    """Invalid layer configs fail loudly at trainer construction: source
    layers beyond the model, source layers at/after the target layer, and a
    ``relative_end_transport_layer`` that resolves outside the model."""
    with pytest.raises(ValueError, match="out of range"):
        LensTrainer(make_tiny_lens_config(tmp_path, "oor", source_layers=[0, 7]), model, prompts=[])
    with pytest.raises(ValueError, match="must all be < target_layer"):
        LensTrainer(make_tiny_lens_config(tmp_path, "src-tgt", source_layers=[-1]), model, prompts=[])
    with pytest.raises(ValueError, match="target_layer"):
        LensTrainer(
            make_tiny_lens_config(tmp_path, "tgt", relative_end_transport_layer=-5),
            model,
            prompts=[],
        )


def test_fit_sets_lens_metadata(model: TinyDecoder, tmp_path: Path) -> None:
    """The lens returned by fit() carries the prompt count, model width, and
    fitted layers — metadata that save/load relies on."""
    config = make_tiny_lens_config(tmp_path, "meta")
    lens = LensTrainer(config, model, prompts=PROMPTS).fit()
    assert lens.num_prompts_trained_on == len(PROMPTS)
    assert lens.d_model == model.d_model
    assert lens.source_layers == [0, 1, 2]


def test_checkpoint_files_are_step_numbered(model: TinyDecoder, tmp_path: Path) -> None:
    """Each periodic checkpoint is a new ``<step>_checkpoint.pt`` file (so
    earlier progress is never overwritten) and a ``final_checkpoint.pt`` is
    always written at the end."""
    config = make_tiny_lens_config(tmp_path, "steps", checkpoint_every_n_prompts=2)
    LensTrainer(config, model, prompts=PROMPTS).fit()
    files = sorted(os.listdir(config.checkpoint_path))
    assert "2_checkpoint.pt" in files
    assert "4_checkpoint.pt" in files
    assert "final_checkpoint.pt" in files


def test_resume_matches_uninterrupted_fit(model: TinyDecoder, tmp_path: Path) -> None:
    """A run interrupted mid-way and resumed via from_checkpoint() produces
    bit-for-bit the same Jacobians as a single uninterrupted fit — i.e. the
    checkpoint captures the full running state (sum, count, position)."""
    full_config = make_tiny_lens_config(tmp_path, "full")
    full_lens = LensTrainer(full_config, model, prompts=PROMPTS).fit()

    # "Interrupted" run: fit only the first two prompts.
    part_config = make_tiny_lens_config(tmp_path, "part", checkpoint_every_n_prompts=2)
    LensTrainer(part_config, model, prompts=PROMPTS[:2]).fit()

    resumed = LensTrainer.from_checkpoint(
        os.path.join(part_config.checkpoint_path, "2_checkpoint.pt"),
        PROMPTS,
        model=model,
    )
    assert resumed.next_prompt_idx == 2
    assert resumed.completed_prompt_count == 2
    resumed_lens = resumed.fit()

    assert resumed_lens.num_prompts_trained_on == len(PROMPTS)
    for layer in full_lens.source_layers:
        torch.testing.assert_close(
            resumed_lens.jacobians_L_dict_FN[layer], full_lens.jacobians_L_dict_FN[layer]
        )


def test_resume_after_skip_does_not_double_count(model: TinyDecoder, tmp_path: Path) -> None:
    """Resume after a too-short prompt was skipped must not double-count.

    The success count and the list position diverge after a skip, and the
    resume must follow the list position, or the prompt after a skip is
    processed twice. Here the skip happens *during* the resumed run; the result must
    still exactly match an uninterrupted fit over the same prompt list, with
    the skipped prompt excluded from the count.
    """
    long_a = "abcdefghij " * 5
    short = "x"  # tokenizes to 2 tokens -> ValueError -> skip
    long_b = "klmnopqrst " * 5
    prompts = [long_a, short, long_b]

    reference_lens = LensTrainer(
        make_tiny_lens_config(tmp_path, "skip-ref"), model, prompts=prompts
    ).fit()
    assert reference_lens.num_prompts_trained_on == 2  # short was skipped

    # "Interrupted" after the first prompt; resume must skip `short` exactly
    # once and process `long_b` exactly once.
    part_config = make_tiny_lens_config(tmp_path, "skip-part", checkpoint_every_n_prompts=1)
    LensTrainer(part_config, model, prompts=[long_a]).fit()
    resumed = LensTrainer.from_checkpoint(
        os.path.join(part_config.checkpoint_path, "1_checkpoint.pt"),
        prompts,
        model=model,
    )
    resumed_lens = resumed.fit()

    assert resumed_lens.num_prompts_trained_on == 2
    for layer in reference_lens.source_layers:
        torch.testing.assert_close(
            resumed_lens.jacobians_L_dict_FN[layer], reference_lens.jacobians_L_dict_FN[layer]
        )


def test_save_load_roundtrip(model: TinyDecoder, tmp_path: Path) -> None:
    """save()/load() round-trips the Jacobians (to fp16 precision) and the
    full config, including ``hf_model_name``."""
    config = make_tiny_lens_config(tmp_path, "roundtrip")
    lens = LensTrainer(config, model, prompts=PROMPTS).fit()

    path = str(tmp_path / "lens.pt")
    lens.save(path)
    loaded = JacobianLens.load(path)

    assert loaded.config.hf_model_name == "tiny"
    assert loaded.num_prompts_trained_on == lens.num_prompts_trained_on
    assert loaded.d_model == lens.d_model
    assert loaded.source_layers == lens.source_layers
    for layer in lens.source_layers:
        torch.testing.assert_close(
            loaded.jacobians_L_dict_FN[layer],
            lens.jacobians_L_dict_FN[layer],
            atol=1e-3,  # save() stores fp16
            rtol=1e-3,
        )


def test_from_checkpoint_rejects_mismatched_model(model: TinyDecoder, tmp_path: Path) -> None:
    """Passing the wrong model to from_checkpoint() is caught: a d_model
    mismatch always raises, and an HF-name mismatch raises whenever the
    model exposes a name (models with no name only get a logged warning)."""
    config = make_tiny_lens_config(tmp_path, "verify", checkpoint_every_n_prompts=2)
    LensTrainer(config, model, prompts=PROMPTS[:2]).fit()
    checkpoint = os.path.join(config.checkpoint_path, "2_checkpoint.pt")

    # Wrong width is always caught.
    with pytest.raises(ValueError, match="d_model"):
        LensTrainer.from_checkpoint(
            checkpoint, PROMPTS, model=TinyDecoder(n_layers=4, d_model=16)
        )

    # Wrong HF name is caught when the model exposes one.
    class _NamedModel:
        d_model = 8

        class _hf_model:
            class config:
                name_or_path = "other-model"

    with pytest.raises(ValueError, match="other-model"):
        LensTrainer.from_checkpoint(checkpoint, PROMPTS, model=_NamedModel())
