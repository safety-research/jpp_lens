"""CPU tests for ``fitting/condense_experts/expert_jacobians.py``:

1. ``combine_expert_jacobians`` is the explicit weighted sum, and its transport
   equals the weighted sum of the per-expert transports (linearity).
2. ``ExpertJacobians.combine`` transports as ``sum_e w_e J_e x`` (fp16-stored
   experts combine as their fp32 casts), resets the router config fields,
   rejects unknown layers, round-trips through ``BaseLens.save(dtype=float32)``
   / ``load`` bit for bit.

The fake ``ExpertJacobians`` / ``ReadoutResiduals`` (fake tensors in the real
dataclasses, ``TinyDecoder`` as the model) come from ``fixtures.py`` and are
shared with ``test_condense_fitter.py``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch as t

from jlens.tests.tiny import TinyDecoder
from workspace_lens.fitting.condense_experts.expert_jacobians import (
    combine_expert_jacobians,
)
from workspace_lens.lenses.base_lens import BaseLens
from workspace_lens.lenses.jacobian_lens import JacobianLens
from workspace_lens.tests.fixtures import (
    CONDENSE_D_MODEL,
    CONDENSE_LAYERS,
    CONDENSE_NUM_EXPERTS,
    CONDENSE_VOCAB_SIZE,
    all_item_expert_transports,
    make_fake_expert_jacobians,
    make_fake_readout_residuals,
)


def test_combine_expert_jacobians_is_the_weighted_sum_and_commutes_with_transport() -> None:
    generator = t.Generator().manual_seed(5)
    experts_EFN = t.randn(CONDENSE_NUM_EXPERTS, CONDENSE_D_MODEL, CONDENSE_D_MODEL, generator=generator)
    weights_E = t.tensor([0.5, -0.2, 0.0, 1.5])
    combined_FN = combine_expert_jacobians(experts_EFN, weights_E)
    explicit_FN = t.zeros(CONDENSE_D_MODEL, CONDENSE_D_MODEL)
    for expert_idx in range(CONDENSE_NUM_EXPERTS):
        explicit_FN += float(weights_E[expert_idx]) * experts_EFN[expert_idx]
    assert combined_FN.shape == (CONDENSE_D_MODEL, CONDENSE_D_MODEL)
    assert t.allclose(combined_FN, explicit_FN, atol=1e-6)
    # Linearity: (sum_e w_e J_e) x == sum_e w_e (J_e x), so the inference lens's
    # transport is the weighted prediction the weights were fitted on.
    residuals_PN = t.randn(7, CONDENSE_D_MODEL, generator=generator)
    transports_PEF = t.einsum("pj,eij->pei", residuals_PN, experts_EFN)
    assert t.allclose(
        residuals_PN @ combined_FN.T,
        t.einsum("pen,e->pn", transports_PEF, weights_E),
        atol=1e-5,
    )


### WIRING: ExpertJacobians / ReadoutResiduals / the combined lens

def test_combine_builds_the_weighted_jacobian_lens(tmp_path: Path) -> None:
    generator = t.Generator().manual_seed(10)
    model = TinyDecoder(n_layers=4, d_model=CONDENSE_D_MODEL, vocab_size=CONDENSE_VOCAB_SIZE)
    experts = make_fake_expert_jacobians(generator, tmp_path)
    residuals = make_fake_readout_residuals(generator)
    weights_L_dict_E = {
        1: t.tensor([0.5, -0.2, 0.0, 1.5]),
        2: t.tensor([0.3, 0.3, 0.3, 0.3]),
    }
    lens = experts.combine(weights_L_dict_E, checkpoint_name="jpp_lens")
    assert isinstance(lens, JacobianLens)
    assert lens.source_layers == CONDENSE_LAYERS
    for layer in CONDENSE_LAYERS:
        residuals_PN = residuals.residuals_L_dict_PN[layer]
        expected_PF = t.einsum(
            "pen,e->pn",
            all_item_expert_transports(model, experts, residuals, layer),
            weights_L_dict_E[layer],
        )
        assert t.allclose(lens.transport(residuals_PN, layer), expected_PF, atol=1e-4)

    # Experts stored in fp16 (as ExpertJacobians.load returns them) combine as
    # their fp32 casts, in fp32.
    half_experts = dataclasses.replace(
        experts,
        experts_L_dict_EFN={
            layer: experts_EFN.half()
            for layer, experts_EFN in experts.experts_L_dict_EFN.items()
        },
    )
    half_lens = half_experts.combine(weights_L_dict_E, checkpoint_name="h")
    for layer in CONDENSE_LAYERS:
        assert half_lens.jacobians_L_dict_FN[layer].dtype == t.float32
        assert t.equal(
            half_lens.jacobians_L_dict_FN[layer],
            combine_expert_jacobians(
                experts.experts_L_dict_EFN[layer].half().float(), weights_L_dict_E[layer]
            ),
        )

    # The router fields are reset, the name replaced, everything else kept —
    # as build_matched_baseline_lens does.
    config, source_config = lens.config, experts.config
    assert config.lens_type == "jacobian"
    assert config.num_clusters == 0
    assert config.cluster_projection_dim == 0
    assert config.checkpoint_name == "jpp_lens"
    for field_name in (
        "hf_model_name",
        "source_layers",
        "relative_end_transport_layer",
        "d_model",
        "num_prompts_trained_on",
        "lrp_mode",
        "creation_time",
        "artifacts_base_dir",
    ):
        assert getattr(config, field_name) == getattr(source_config, field_name)

    # A subset of the experts' layers is fine; an unknown layer is not.
    assert experts.combine({1: weights_L_dict_E[1]}, checkpoint_name="one").source_layers == [
        1
    ]
    with pytest.raises(ValueError, match=r"layers \[5\]"):
        experts.combine(
            {1: weights_L_dict_E[1], 5: weights_L_dict_E[1]},
            checkpoint_name="x",
        )

    # fp32 save / load reproduces the combined Jacobians bit for bit.
    path = str(tmp_path / "jpp_lens.pt")
    lens.save(path, dtype=t.float32)
    loaded = BaseLens.load(path)
    assert isinstance(loaded, JacobianLens)
    assert loaded.source_layers == CONDENSE_LAYERS
    assert loaded.jacobians_L_dict_FN is not None and lens.jacobians_L_dict_FN is not None
    for layer in CONDENSE_LAYERS:
        assert t.equal(loaded.jacobians_L_dict_FN[layer], lens.jacobians_L_dict_FN[layer])
    assert loaded.config.to_dict() == lens.config.to_dict()
