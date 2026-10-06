"""Tests for J-lens vector utilities and residual-stream interventions.

Covers ``jlens_vectors``/``resolve_token_id`` (pinning the transpose
convention against ``transport``), the Clamp edit maths, and the hook plumbing
(ordering, tuple outputs, cleanup). The hook tests use ``AddVector``, a
test-only intervention that adds a fixed vector.
"""

from __future__ import annotations

import pytest
import torch as t

from jlens.hooks import ActivationRecorder
from jlens.tests.tiny import TinyDecoder
from workspace_lens.interventions.base import Intervention
from workspace_lens.interventions.hooks import InterventionHooks
from workspace_lens.interventions.interventions import Clamp
from workspace_lens.lenses.jacobian_lens import JacobianLens
from workspace_lens.tests.fixtures import fit_tiny_jacobian_lens
from workspace_lens.utils import get_unembed_matrix, jlens_vectors, resolve_token_id

D_MODEL = 8


class AddVector(Intervention):
    """Test-only intervention: ``h <- h + vector`` at each of its layers."""

    def __init__(self, vectors_L_dict_N: dict[int, t.Tensor]) -> None:
        self.vectors_L_dict_N = vectors_L_dict_N
        super().__init__(layers=list(vectors_L_dict_N), d_model=D_MODEL)

    def _edit_selected(self, residual_BPN: t.Tensor, layer: int) -> t.Tensor:
        return residual_BPN + self.vectors_L_dict_N[layer]


@pytest.fixture(scope="module")
def model() -> TinyDecoder:
    return TinyDecoder(n_layers=4, d_model=D_MODEL)


@pytest.fixture(scope="module")
def lens(model: TinyDecoder, tmp_path_factory: pytest.TempPathFactory) -> JacobianLens:
    """A lens fitted on the module-scoped tiny model (layers 0-2)."""
    return fit_tiny_jacobian_lens(model, tmp_path_factory.mktemp("artifacts"), "test-interventions")


# ---------------------------------------------------------------------------
# jlens_vectors / resolve_token_id
# ---------------------------------------------------------------------------


def test_jlens_vectors_match_transport(model: TinyDecoder, lens: JacobianLens) -> None:
    """Pins the transpose convention: ``<h, v>`` with ``v = W_U[tok] @ J_l``
    must equal reading the transported residual through the unembedding row."""
    t.manual_seed(0)
    h_N = t.randn(D_MODEL)
    token_id = 5
    vectors_L_dict_N = jlens_vectors(lens, model, token_id)
    assert sorted(vectors_L_dict_N) == [0, 1, 2]

    unembed_row_F = get_unembed_matrix(model).float()[token_id]
    for layer in [0, 1, 2]:
        lens_logit = h_N @ vectors_L_dict_N[layer]
        transported_logit = lens.transport(h_N, layer) @ unembed_row_F
        t.testing.assert_close(lens_logit, transported_logit)


def test_resolve_token_id(model: TinyDecoder) -> None:
    # Int ids pass straight through when in range.
    assert resolve_token_id(model, 7) == 7
    with pytest.raises(ValueError, match="out of range"):
        resolve_token_id(model, 32)  # vocab_size is 32
    with pytest.raises(ValueError, match="out of range"):
        resolve_token_id(model, -1)
    # Byte tokenizer: 'a' -> [BOS=0, 1 + (97 % 30)] -> BOS stripped -> 8.
    assert resolve_token_id(model, "a") == 8
    with pytest.raises(ValueError, match="single-token"):
        resolve_token_id(model, "ab")


def test_jlens_vectors_layer_validation(model: TinyDecoder, lens: JacobianLens) -> None:
    with pytest.raises(ValueError, match="not in source_layers"):
        jlens_vectors(lens, model, 5, layers=[3])


# ---------------------------------------------------------------------------
# The Intervention base class (exercised through Clamp)
# ---------------------------------------------------------------------------


def test_duplicate_token_positions_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        Clamp({0: t.randn(D_MODEL)}, {0: 1.0}, token_positions=[1, 1])


def test_edit_error_paths() -> None:
    t.manual_seed(6)
    clamp = Clamp({1: t.randn(D_MODEL)}, {1: 1.0})
    with pytest.raises(ValueError, match="no vector for layer"):
        clamp.edit(t.randn(1, 3, D_MODEL), 0)
    with pytest.raises(ValueError, match="d_model"):
        clamp.edit(t.randn(1, 3, D_MODEL // 2), 1)


# ---------------------------------------------------------------------------
# Clamp
# ---------------------------------------------------------------------------


def lens_coordinate(residual_BSN: t.Tensor, vector_N: t.Tensor) -> t.Tensor:
    """The lens coordinate ``c = pinv(v) @ h = <v, h> / ||v||^2``; ``[B, S]``."""
    return (residual_BSN @ vector_N) / vector_N.dot(vector_N)


def orthogonal_complement(residual_BSN: t.Tensor, vector_N: t.Tensor) -> t.Tensor:
    """``h`` minus its projection onto ``v``."""
    unit_vector_N = vector_N / vector_N.norm()
    return residual_BSN - (residual_BSN @ unit_vector_N).unsqueeze(-1) * unit_vector_N


def test_clamp_sets_coordinate_and_keeps_orthogonal_complement() -> None:
    t.manual_seed(11)
    vector_N = 3.0 * t.randn(D_MODEL)  # non-unit norm, so the coordinate units matter
    residual_BSN = t.randn(2, 3, D_MODEL)
    residual_copy_BSN = residual_BSN.clone()
    target_coordinate = 2.5

    clamp = Clamp({1: vector_N}, {1: target_coordinate})
    edited_BSN = clamp.edit(residual_BSN, 1)

    # The coordinate along v equals the target at every position, measured
    # with the pseudoinverse definition c = V^+ h, V = [v]...
    pinv_1N = t.linalg.pinv(vector_N.unsqueeze(1))
    t.testing.assert_close(
        (edited_BSN @ pinv_1N.T).squeeze(-1), t.full((2, 3), target_coordinate)
    )
    # ...the edit is exactly h + (target - c) v, parallel to v...
    coordinate_BS1 = lens_coordinate(residual_BSN, vector_N).unsqueeze(-1)
    t.testing.assert_close(
        edited_BSN - residual_BSN, (target_coordinate - coordinate_BS1) * vector_N
    )
    # ...so the component orthogonal to v is unchanged.
    t.testing.assert_close(
        orthogonal_complement(edited_BSN, vector_N),
        orthogonal_complement(residual_BSN, vector_N),
    )

    assert edited_BSN.dtype == residual_BSN.dtype
    assert edited_BSN.device == residual_BSN.device
    # The input tensor is never modified in place.
    assert t.equal(residual_BSN, residual_copy_BSN)


def test_clamp_uses_each_layers_own_vector() -> None:
    """Two layers with differently scaled vectors: each layer must reach the
    target along *its own* vector (a mutant ignoring ``layer`` in the per-layer
    lookups would clamp along the wrong vector at one of them)."""
    t.manual_seed(15)
    vectors_L_dict_N = {0: t.randn(D_MODEL), 2: 5.0 * t.randn(D_MODEL)}
    residual_BSN = t.randn(2, 3, D_MODEL)
    target_coordinate = -0.4

    clamp = Clamp(vectors_L_dict_N, dict.fromkeys(vectors_L_dict_N, target_coordinate))
    assert clamp.layers == [0, 2]
    for layer, vector_N in vectors_L_dict_N.items():
        edited_BSN = clamp.edit(residual_BSN, layer)
        t.testing.assert_close(
            lens_coordinate(edited_BSN, vector_N), t.full((2, 3), target_coordinate)
        )


def test_clamp_is_idempotent_and_zero_removes_the_component() -> None:
    t.manual_seed(12)
    vector_N = t.randn(D_MODEL)
    residual_BSN = t.randn(2, 3, D_MODEL)

    clamp = Clamp({2: vector_N}, {2: -1.5})
    edited_BSN = clamp.edit(residual_BSN, 2)
    # Idempotent: the coordinate is already at the target, so the update is ~0.
    t.testing.assert_close(clamp.edit(edited_BSN, 2), edited_BSN)
    # A residual already at the target coordinate is left alone.
    already_clamped_BSN = orthogonal_complement(residual_BSN, vector_N) - 1.5 * vector_N
    t.testing.assert_close(clamp.edit(already_clamped_BSN, 2), already_clamped_BSN)

    # target_coordinate=0 removes the component along v.
    t.testing.assert_close(
        Clamp({2: vector_N}, {2: 0.0}).edit(residual_BSN, 2),
        orthogonal_complement(residual_BSN, vector_N),
    )


def test_clamp_target_coordinate_is_in_units_of_the_vector() -> None:
    """Pins the convention: the clamped quantity is the pinv coordinate
    c = <v, h> / ||v||^2 (c = V^+ h), so the J-lens logit <v, h>
    becomes target * ||v||^2, and scaling v by k is undone by target / k."""
    t.manual_seed(13)
    vector_N = t.randn(D_MODEL)
    residual_BSN = t.randn(2, 3, D_MODEL)
    target_coordinate = 0.75

    edited_BSN = Clamp({0: vector_N}, {0: target_coordinate}).edit(
        residual_BSN, 0
    )
    t.testing.assert_close(
        edited_BSN @ vector_N,
        t.full((2, 3), target_coordinate) * vector_N.dot(vector_N),
    )

    doubled_vector_BSN = Clamp({0: 2.0 * vector_N}, {0: target_coordinate / 2.0}
    ).edit(residual_BSN, 0)
    t.testing.assert_close(doubled_vector_BSN, edited_BSN)


def test_clamp_token_positions_and_dtype_round_trip() -> None:
    t.manual_seed(14)
    vector_N = t.randn(D_MODEL)
    residual_BSN = t.randn(2, 5, D_MODEL)
    target_coordinate = 1.0
    clamp = Clamp({0: vector_N}, {0: target_coordinate}, token_positions=[0, -1]
    )

    edited_BSN = clamp.edit(residual_BSN, 0)
    # Selected positions are clamped in both batch rows...
    t.testing.assert_close(
        lens_coordinate(edited_BSN[:, [0, -1], :], vector_N), t.ones(2, 2)
    )
    # ...and every unselected position is untouched.
    assert t.equal(edited_BSN[:, 1:4, :], residual_BSN[:, 1:4, :])

    # fp16 residuals: computed in fp32, cast back. Unselected positions are
    # bit-identical; selected positions match h_perp + target * v computed in
    # fp32 from the same fp16 values, then rounded to half.
    residual_fp16_BSN = residual_BSN.half()
    edited_fp16_BSN = clamp.edit(residual_fp16_BSN, 0)
    assert edited_fp16_BSN.dtype == t.float16
    assert t.equal(edited_fp16_BSN[:, 1:4, :], residual_fp16_BSN[:, 1:4, :])
    selected_fp32_BPN = residual_fp16_BSN.float()[:, [0, -1], :]
    expected_fp16_BPN = (
        orthogonal_complement(selected_fp32_BPN, vector_N)
        + target_coordinate * vector_N
    ).half()
    t.testing.assert_close(edited_fp16_BSN[:, [0, -1], :], expected_fp16_BPN)


def test_clamp_end_to_end_through_hooks(model: TinyDecoder, lens: JacobianLens) -> None:
    """Clamp real J-lens vectors at every fitted layer (0-2) during a live
    forward pass. The residual recorded *after* the hooks must sit at the
    target coordinate at each edited layer, even though the clamp at layer 0
    already changes what blocks 1 and 2 compute."""
    token_id = 5
    vectors_L_dict_N = jlens_vectors(lens, model, token_id)
    edited_layers = sorted(vectors_L_dict_N)
    target_coordinate = 0.3
    clamp = Clamp(vectors_L_dict_N, dict.fromkeys(vectors_L_dict_N, target_coordinate))
    input_ids_Int_1S = model.encode("clamp me")
    seq_len = input_ids_Int_1S.shape[1]

    with t.no_grad(), ActivationRecorder(model.layers, at=edited_layers) as recorder:
        model.forward(input_ids_Int_1S)
        clean_1SN_by_layer = {
            layer: recorder.activations[layer].detach().float().clone()
            for layer in edited_layers
        }

    with (
        t.no_grad(),
        InterventionHooks(model, [clamp]),
        # Entered after the hooks, so it sees the edited residual.
        ActivationRecorder(model.layers, at=edited_layers) as recorder,
    ):
        model.forward(input_ids_Int_1S)
        clamped_1SN_by_layer = {
            layer: recorder.activations[layer].detach().float().clone()
            for layer in edited_layers
        }

    target_1S = t.full((1, seq_len), target_coordinate)
    for layer in edited_layers:
        vector_N = vectors_L_dict_N[layer]
        # The clean pass is not already at the target (so the check is not vacuous)...
        assert not t.allclose(
            lens_coordinate(clean_1SN_by_layer[layer], vector_N), target_1S
        )
        # ...and the live forward's residual is clamped at every position.
        t.testing.assert_close(
            lens_coordinate(clamped_1SN_by_layer[layer], vector_N), target_1S
        )


def test_clamp_two_atoms_pins_both_coordinates() -> None:
    """A two-atom basis: both pinv coordinates land on their targets and the
    component orthogonal to span(V) is untouched."""
    t.manual_seed(21)
    atoms_NK = t.randn(D_MODEL, 2)
    residual_BSN = t.randn(2, 3, D_MODEL)
    targets_K = t.tensor([0.7, -1.2])

    edited_BSN = Clamp({0: atoms_NK}, {0: targets_K}).edit(residual_BSN, 0)

    pinv_KN = t.linalg.pinv(atoms_NK)
    t.testing.assert_close(edited_BSN @ pinv_KN.T, targets_K.expand(2, 3, 2))
    projector_NN = atoms_NK @ pinv_KN  # onto span(V)

    def complement(residual: t.Tensor) -> t.Tensor:
        return residual - residual @ projector_NN.T

    t.testing.assert_close(complement(edited_BSN), complement(residual_BSN))


def test_clamp_per_position_targets() -> None:
    """A shared basis with a different target at every edited position (the
    swap runner's shape); unselected positions are untouched."""
    t.manual_seed(22)
    seq_len, num_atoms = 4, 2
    positions = [1, 3]
    atoms_NK = t.randn(D_MODEL, num_atoms)
    targets_PK = t.randn(len(positions), num_atoms)
    residual_BSN = t.randn(2, seq_len, D_MODEL)

    clamp = Clamp({0: atoms_NK}, {0: targets_PK}, token_positions=positions)
    edited_BSN = clamp.edit(residual_BSN, 0)

    pinv_KN = t.linalg.pinv(atoms_NK)
    for row, position in enumerate(positions):
        t.testing.assert_close(
            edited_BSN[:, position, :] @ pinv_KN.T, targets_PK[row].expand(2, num_atoms)
        )
    assert t.equal(edited_BSN[:, [0, 2], :], residual_BSN[:, [0, 2], :])

    # Per-position targets must match the number of edited positions.
    with pytest.raises(ValueError, match="positions"):
        Clamp({0: atoms_NK}, {0: targets_PK}).edit(residual_BSN, 0)


def test_clamp_to_swapped_clean_coordinates_exchanges_the_coordinates() -> None:
    """The swap runner's clamp: pinning (c_s, c_t) to the clean pass's
    scale * (c_t, c_s) at every position is, on the clean residual itself, the
    swap edit h + V (sigma(c) - c), sigma(c) = scale * (c_t, c_s)."""
    t.manual_seed(24)
    source_N, target_N = t.randn(D_MODEL), t.randn(D_MODEL)
    residual_BSN = t.randn(1, 5, D_MODEL)
    scale = 2.0

    basis_NK = t.stack([source_N, target_N], dim=1)
    clean_coords_SK = residual_BSN[0] @ t.linalg.pinv(basis_NK).T
    swapped_targets_SK = scale * clean_coords_SK[:, [1, 0]]

    clamp = Clamp({0: basis_NK}, {0: swapped_targets_SK})
    expected_BSN = residual_BSN + ((swapped_targets_SK - clean_coords_SK) @ basis_NK.T)[None]
    t.testing.assert_close(clamp.edit(residual_BSN, 0), expected_BSN)


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


def test_hooks_apply_in_list_order(model: TinyDecoder) -> None:
    """Two interventions on one layer apply in list order; the same vector for
    both makes the orders provably different: ablate(steer(h)) kills the
    steering component, steer(ablate(h)) keeps it."""
    t.manual_seed(7)
    vector_N = t.randn(D_MODEL)
    layer = 1
    steering = AddVector({layer: 2.0 * vector_N})
    ablation = Clamp({layer: vector_N}, {layer: 0.0})
    prompt = "order test"
    input_ids_Int_1S = model.encode(prompt)

    with t.no_grad(), ActivationRecorder(model.layers, at=[layer]) as recorder:
        model.forward(input_ids_Int_1S)
        clean_residual_1SN = recorder.activations[layer].detach().clone()

    observed_1SN_by_order: dict[str, t.Tensor] = {}
    for label, ordered_interventions in (
        ("steer_then_ablate", [steering, ablation]),
        ("ablate_then_steer", [ablation, steering]),
    ):
        with (
            t.no_grad(),
            InterventionHooks(model, ordered_interventions),
            # Entered after the hooks, so it sees the edited residual.
            ActivationRecorder(model.layers, at=[layer]) as recorder,
        ):
            model.forward(input_ids_Int_1S)
            observed_1SN_by_order[label] = recorder.activations[layer].detach().clone()

    t.testing.assert_close(
        observed_1SN_by_order["steer_then_ablate"],
        ablation.edit(steering.edit(clean_residual_1SN, layer), layer),
    )
    t.testing.assert_close(
        observed_1SN_by_order["ablate_then_steer"],
        steering.edit(ablation.edit(clean_residual_1SN, layer), layer),
    )
    assert not t.allclose(
        observed_1SN_by_order["steer_then_ablate"],
        observed_1SN_by_order["ablate_then_steer"],
    )


def test_hook_rebuilds_tuple_output(model: TinyDecoder) -> None:
    """Some HF blocks (other families / older transformers) return
    ``(hidden, present_kv, ...)`` tuples; TinyDecoder and current Qwen/Llama
    decoder layers return bare tensors, so this unit test is the tuple
    branch's only coverage."""
    t.manual_seed(8)
    vector_N = t.randn(D_MODEL)
    steering = AddVector({1: 2.0 * vector_N})
    hook = InterventionHooks(model, [steering])._make_hook(1, [steering])
    hidden_BSN = t.randn(1, 4, D_MODEL)

    tuple_output = hook(model.layers[1], (), (hidden_BSN, "kv"))
    assert isinstance(tuple_output, tuple)
    assert tuple_output[1] == "kv"
    assert t.equal(tuple_output[0], steering.edit(hidden_BSN, 1))

    tensor_output = hook(model.layers[1], (), hidden_BSN)
    assert t.is_tensor(tensor_output)
    assert t.equal(tensor_output, steering.edit(hidden_BSN, 1))


def test_hooks_removed_after_exception(model: TinyDecoder) -> None:
    t.manual_seed(9)
    steering = AddVector({1: 5.0 * t.randn(D_MODEL)})
    input_ids_Int_1S = model.encode("cleanup test")
    with t.no_grad():
        clean_logits_1SV = model.unembed(model.forward(input_ids_Int_1S).last_hidden_state)

    with pytest.raises(RuntimeError, match="boom"):
        with InterventionHooks(model, [steering]):
            raise RuntimeError("boom")

    for block in model.layers:
        assert len(block._forward_hooks) == 0
    with t.no_grad():
        after_logits_1SV = model.unembed(model.forward(input_ids_Int_1S).last_hidden_state)
    assert t.equal(after_logits_1SV, clean_logits_1SV)


def test_hooks_reentry_guard_and_distinct_instance_nesting(model: TinyDecoder) -> None:
    """Re-entering one active InterventionHooks instance would register every
    hook twice (edits applied twice) and the inner exit would strip the outer
    context's hooks — so same-instance re-entry raises. Distinct instances
    nest correctly: edits compose while both are active, and the inner exit
    removes only its own handles.
    """
    t.manual_seed(10)
    steering = AddVector({1: 3.0 * t.randn(D_MODEL)})
    input_ids_Int_1S = model.encode("reentry test")

    def layer_one_output_1SN() -> t.Tensor:
        with ActivationRecorder(model.layers, at=[1]) as recorder, t.no_grad():
            model.forward(input_ids_Int_1S)
        return recorder.activations[1].detach().clone()

    clean_1SN = layer_one_output_1SN()

    # Same-instance re-entry raises; the instance stays usable afterwards.
    hooks = InterventionHooks(model, [steering])
    with hooks:
        with pytest.raises(RuntimeError, match="already active"):
            with hooks:
                pass
    with hooks:
        pass
    for block in model.layers:
        assert len(block._forward_hooks) == 0

    # Distinct instances nest: both apply while active, and the inner exit
    # leaves the outer instance's hook in place.
    outer_hooks = InterventionHooks(model, [steering])
    inner_hooks = InterventionHooks(model, [steering])
    with outer_hooks:
        with inner_hooks:
            edited_twice_1SN = layer_one_output_1SN()
        edited_once_1SN = layer_one_output_1SN()
    after_1SN = layer_one_output_1SN()

    expected_once_1SN = steering.edit(clean_1SN, 1)
    t.testing.assert_close(edited_once_1SN, expected_once_1SN)
    t.testing.assert_close(edited_twice_1SN, steering.edit(expected_once_1SN, 1))
    assert t.equal(after_1SN, clean_1SN)


def test_hooks_out_of_range_layer_raises(model: TinyDecoder) -> None:
    steering = AddVector({99: t.ones(D_MODEL)})
    with pytest.raises(ValueError, match="out of range"):
        InterventionHooks(model, [steering])
