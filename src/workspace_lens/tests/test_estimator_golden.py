"""Regression test: the Jacobian estimator, the k-means routers and the expert-Jacobian
trainer reproduce pinned reference tensors.

The reference tensors live in ``estimator_golden_tensors.pt`` beside this module:
``LensTrainer.fit_step``'s per-layer Jacobians and the expert trainer's per-K, per-layer
sufficient statistics on the two tiny fit prompts and the mixed-semantic prompt, plus the
K = 2 and K = 3 routers' centroids (fp32, one CPU thread), as computed by
:func:`compute_golden_tensors`.

The tests compare with a tolerance (``TOLERANCES`` below, per group; integer counts
exactly) rather than by hash: the same torch build gives different last bits on different
CPUs, so bit identity is a criterion for one machine, not for CI.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch as t

from jlens.tests.tiny import TinyDecoder
from workspace_lens.fitting.jacobian_fitting import LensTrainer
from workspace_lens.routing.cluster import collect_valid_position_activations
from workspace_lens.routing.router import ActivationRouterCollection
from workspace_lens.tests.fixtures import (
    MIXED_SEMANTIC_PROMPT,
    TINY_FIT_PROMPTS,
    make_tiny_expert_trainer,
    make_tiny_lens_config,
)

ROUTER_FIT_PROMPTS = [*TINY_FIT_PROMPTS, "uvwxyzabcd " * 5, "efghijklmn " * 5]
SOURCE_LAYERS = [0, 1, 2]
GOLDEN_PROMPTS = [*TINY_FIT_PROMPTS, MIXED_SEMANTIC_PROMPT]
GOLDEN_TENSORS_PATH = Path(__file__).with_name("estimator_golden_tensors.pt")
GOLDEN_GROUPS = ("jacobian", "router", "experts")
# (rtol, atol) per group. fp32 kernel differences between CPUs sit around 1e-7 relative. The
# k-means centroids compound that noise over their iterations, so they get ten times the room;
# a flipped assignment would move a centroid by about 1e-2, far outside either bound.
TOLERANCES: dict[str, tuple[float, float]] = {
    "jacobian": (1e-5, 1e-6),
    "router": (1e-4, 1e-5),
    "experts": (1e-5, 1e-6),
}

GoldenTensors = dict[str, dict[str, t.Tensor]]


def make_golden_model() -> TinyDecoder:
    """The fixture model: one CPU thread, seed 0, the 4-layer d_model-8 TinyDecoder."""
    t.set_num_threads(1)
    t.manual_seed(0)
    return TinyDecoder(n_layers=4, d_model=8)


def _plain_copy(tensor: t.Tensor) -> t.Tensor:
    return tensor.detach().cpu().contiguous().clone()


def compute_golden_tensors(model: TinyDecoder, tmp_path: Path) -> GoldenTensors:
    """The fixture flow behind the golden file:
    ``"jacobian"``: ``LensTrainer.fit_step``'s per-layer Jacobians on the golden prompts;
    ``"router"``: the K = 2 and K = 3 routers' centroids, refit on the expert-fitting
    test's prompts; ``"experts"``: the expert trainer's row sums, weight sums and position
    counts after the golden prompts."""
    tensors: GoldenTensors = {group: {} for group in GOLDEN_GROUPS}
    trainer = LensTrainer(
        make_tiny_lens_config(tmp_path, "golden-jacobian"), model, prompts=[]
    )
    for prompt_idx, prompt in enumerate(GOLDEN_PROMPTS):
        jacobians_L_dict_FN, _, _ = trainer.fit_step(prompt)
        for layer in SOURCE_LAYERS:
            tensors["jacobian"][f"prompt{prompt_idx}_layer{layer}"] = _plain_copy(
                jacobians_L_dict_FN[layer]
            )

    activations_L_dict_PN = collect_valid_position_activations(
        model, ROUTER_FIT_PROMPTS, SOURCE_LAYERS, skip_first_n_positions=16, max_seq_len=64
    )
    routers: dict[int, ActivationRouterCollection] = {}
    for num_clusters in (2, 3):
        router = ActivationRouterCollection(num_clusters, projection_dim=4, seed=0)
        router.fit(activations_L_dict_PN)
        routers[num_clusters] = router
        for layer in SOURCE_LAYERS:
            tensors["router"][f"K{num_clusters}_layer{layer}_centroids"] = _plain_copy(
                router.layer_routers_L_dict[layer].centroids_ED
            )

    expert_trainer = make_tiny_expert_trainer(model, tmp_path, "golden-experts", routers)
    for prompt in GOLDEN_PROMPTS:
        expert_trainer.fit_step(prompt)
    for num_clusters, fit_sums in expert_trainer.fit_sums_K_dict.items():
        for layer, layer_sums in fit_sums.layer_sums_L_dict.items():
            prefix = f"K{num_clusters}_layer{layer}"
            tensors["experts"][f"{prefix}_row_sum"] = _plain_copy(
                layer_sums.weighted_jacobian_row_sum_EFN
            )
            tensors["experts"][f"{prefix}_weight_sum"] = _plain_copy(layer_sums.weight_sum_E)
            tensors["experts"][f"{prefix}_position_count"] = _plain_copy(
                layer_sums.position_count_E
            )
    return tensors


def load_golden_tensors(path: Path = GOLDEN_TENSORS_PATH) -> GoldenTensors:
    """The three tensor groups of the golden file."""
    state = t.load(path, map_location="cpu", weights_only=True)
    return {group: dict(state[group]) for group in GOLDEN_GROUPS}


def assert_group_close(
    actual: dict[str, t.Tensor], expected: dict[str, t.Tensor], group: str
) -> None:
    """Same keys, and every tensor within the group's tolerances of its golden (integer
    tensors exactly: the tolerances are below one count)."""
    assert sorted(actual) == sorted(expected), (group, sorted(actual), sorted(expected))
    rtol, atol = TOLERANCES[group]
    for key in sorted(expected):
        t.testing.assert_close(
            actual[key],
            expected[key],
            rtol=rtol,
            atol=atol,
            msg=lambda m, key=key: f"{group}/{key}: {m}",
        )


@pytest.fixture(scope="module")
def computed_golden_tensors(tmp_path_factory: pytest.TempPathFactory) -> Iterator[GoldenTensors]:
    # make_golden_model pins one thread (the golden file was written that way); restored for
    # the later modules.
    previous_num_threads = t.get_num_threads()
    tensors = compute_golden_tensors(make_golden_model(), tmp_path_factory.mktemp("golden"))
    t.set_num_threads(previous_num_threads)
    yield tensors


@pytest.fixture(scope="module")
def golden_tensors() -> GoldenTensors:
    return load_golden_tensors()


def test_jacobian_fit_step_matches_golden_tensors(
    computed_golden_tensors: GoldenTensors, golden_tensors: GoldenTensors
) -> None:
    """``LensTrainer.fit_step`` reproduces the pinned per-layer Jacobians on all three golden
    prompts (same keys, values within tolerance)."""
    assert_group_close(computed_golden_tensors["jacobian"], golden_tensors["jacobian"], "jacobian")


def test_expert_sums_match_golden_tensors(
    computed_golden_tensors: GoldenTensors, golden_tensors: GoldenTensors
) -> None:
    """The K = 2 and K = 3 routers refit on the expert-fitting test's prompts have the
    pinned centroids, and the expert trainer's row sums, weight sums and position counts
    after the three golden prompts match the pinned values within tolerance."""
    assert_group_close(computed_golden_tensors["router"], golden_tensors["router"], "router")
    assert_group_close(computed_golden_tensors["experts"], golden_tensors["experts"], "experts")
