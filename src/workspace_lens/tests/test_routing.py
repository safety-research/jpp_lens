"""Clustering for the expert fit's activation router: PCA orientation, k-means recovery
of separated blobs, fit/assign consistency, save/load, and the activation pool's positions."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from jlens.tests.tiny import TinyDecoder
from workspace_lens.routing.cluster import (
    collect_valid_position_activations,
    fit_pca,
    kmeans,
    unit_normalize,
)
from workspace_lens.routing.router import ActivationRouterCollection
from workspace_lens.tests.fixtures import MIXED_SEMANTIC_PROMPT
from workspace_lens.utils import (
    get_position_mask_with_early_skips,
    record_activations,
)


def test_unit_normalize_rows() -> None:
    points_PN = torch.randn(10, 6, generator=torch.Generator().manual_seed(0)) * 5
    normalized_PN = unit_normalize(points_PN)
    torch.testing.assert_close(normalized_PN.norm(dim=-1), torch.ones(10))
    # Direction preserved: normalised rows are positive multiples of the originals.
    cosine_P = torch.nn.functional.cosine_similarity(points_PN, normalized_PN, dim=-1)
    torch.testing.assert_close(cosine_P, torch.ones(10))


def test_fit_pca_recovers_dominant_direction() -> None:
    """Points spread along a known axis: the first component must align with
    it, projection must have the right shape, and components are orthonormal."""
    generator = torch.Generator().manual_seed(0)
    direction_N = torch.zeros(8)
    direction_N[3] = 1.0
    # Large variance along `direction_N`, small isotropic noise elsewhere.
    points_PN = (
        torch.randn(500, 1, generator=generator) * 10 * direction_N
        + torch.randn(500, 8, generator=generator) * 0.1
    )

    mean_N, components_ND = fit_pca(points_PN, projection_dim=3)
    assert mean_N.shape == (8,)
    assert components_ND.shape == (8, 3)
    torch.testing.assert_close(
        components_ND.T @ components_ND, torch.eye(3), atol=1e-5, rtol=0
    )
    # First component aligns with the planted direction (up to sign).
    alignment = (components_ND[:, 0] @ direction_N).abs()
    assert alignment > 0.99

    with pytest.raises(ValueError, match="projection_dim"):
        fit_pca(points_PN, projection_dim=9)


def test_kmeans_recovers_separated_blobs() -> None:
    """Three well-separated blobs: k-means must put one centroid near each
    blob mean and assign every point to its own blob's centroid."""
    generator = torch.Generator().manual_seed(0)
    blob_means_ED = torch.tensor([[10.0, 0.0], [-10.0, 0.0], [0.0, 10.0]])
    points_PD = torch.cat(
        [mean_D + torch.randn(50, 2, generator=generator) * 0.3 for mean_D in blob_means_ED]
    )

    centroids_ED, assignments_Int_P = kmeans(points_PD, num_clusters=3, seed=0)

    # Each blob mean has a centroid within noise distance.
    distances_EE = torch.cdist(blob_means_ED, centroids_ED)
    assert (distances_EE.min(dim=1).values < 0.5).all()
    # Points in the same blob share a label; different blobs differ.
    labels_per_blob = [assignments_Int_P[i * 50 : (i + 1) * 50] for i in range(3)]
    for blob_labels_P in labels_per_blob:
        assert (blob_labels_P == blob_labels_P[0]).all()
    assert len({int(blob_labels_P[0]) for blob_labels_P in labels_per_blob}) == 3


def test_kmeans_deterministic_given_seed() -> None:
    points_PD = torch.randn(100, 4, generator=torch.Generator().manual_seed(1))
    centroids_a_ED, assignments_a_P = kmeans(points_PD, num_clusters=4, seed=7)
    centroids_b_ED, assignments_b_P = kmeans(points_PD, num_clusters=4, seed=7)
    torch.testing.assert_close(centroids_a_ED, centroids_b_ED, rtol=0, atol=0)
    assert (assignments_a_P == assignments_b_P).all()


def test_kmeans_too_few_points() -> None:
    with pytest.raises(ValueError, match="points"):
        kmeans(torch.randn(2, 3), num_clusters=5)


def test_router_collection_fit_assign_roundtrip(tmp_path: Path) -> None:
    """fit() then assign() reproduces the training assignments (nearest
    centroid is stable), works across layers, and survives save/load."""
    generator = torch.Generator().manual_seed(0)
    # Two direction-separated groups per layer (normalisation keeps direction).
    group_a_PN = torch.randn(80, 8, generator=generator) + torch.tensor(
        [8.0, 0, 0, 0, 0, 0, 0, 0]
    )
    group_b_PN = torch.randn(80, 8, generator=generator) + torch.tensor(
        [0, 0, 0, 0, 0, 0, 0, 8.0]
    )
    activations_L_dict_PN = {
        3: torch.cat([group_a_PN, group_b_PN]),
        5: torch.cat([group_b_PN, group_a_PN]),
    }

    router_collection = ActivationRouterCollection(num_clusters=2, projection_dim=4, seed=0)
    router_collection.fit(activations_L_dict_PN)
    assert set(router_collection.layer_routers_L_dict) == {3, 5}

    assignments_Int_P = router_collection.assign(activations_L_dict_PN[3], layer=3)
    assert assignments_Int_P.shape == (160,)
    # The two groups land in different clusters, uniformly within each group.
    group_a_labels_P, group_b_labels_P = assignments_Int_P[:80], assignments_Int_P[80:]
    assert (group_a_labels_P == group_a_labels_P[0]).all()
    assert (group_b_labels_P == group_b_labels_P[0]).all()
    assert int(group_a_labels_P[0]) != int(group_b_labels_P[0])

    # Leading batch dims are preserved.
    batched_assignments_Int_BS = router_collection.assign(
        activations_L_dict_PN[3].reshape(2, 80, 8), layer=3
    )
    assert batched_assignments_Int_BS.shape == (2, 80)
    assert (batched_assignments_Int_BS.reshape(-1) == assignments_Int_P).all()

    path = str(tmp_path / "router_collection.pt")
    router_collection.save(path)
    reloaded = ActivationRouterCollection.load(path)
    assert reloaded.num_clusters == 2 and reloaded.projection_dim == 4
    reloaded_assignments_Int_P = reloaded.assign(activations_L_dict_PN[3], layer=3)
    assert (reloaded_assignments_Int_P == assignments_Int_P).all()


def test_collect_valid_position_activations_pools_the_valid_positions() -> None:
    """The pool holds every valid position of every prompt (the skip/final
    rule the fit uses), in prompt order."""
    model = TinyDecoder(n_layers=4, d_model=8)
    prompts = [MIXED_SEMANTIC_PROMPT, "klmnopqrst " * 3]
    layers = [0, 2]

    plain_L_dict_PN = collect_valid_position_activations(
        model, prompts, layers, skip_first_n_positions=4, max_seq_len=64
    )

    expected_plain_L_dict_list_PN: dict[int, list[torch.Tensor]] = {
        layer: [] for layer in layers
    }
    for prompt in prompts:
        input_ids_Int_1S, activations_L_dict_SN = record_activations(model, prompt, 64, layers)
        plain_Bool_S = get_position_mask_with_early_skips(input_ids_Int_1S.shape[1], 4)
        for layer in layers:
            expected_plain_L_dict_list_PN[layer].append(
                activations_L_dict_SN[layer][plain_Bool_S]
            )

    for layer in layers:
        expected_plain_PN = torch.cat(expected_plain_L_dict_list_PN[layer]).float()
        assert torch.equal(plain_L_dict_PN[layer], expected_plain_PN)
