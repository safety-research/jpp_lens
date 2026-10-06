"""Per-layer clustering of residual-stream activations for expert-Jacobian fits.

Pipeline per layer, fit once on a corpus of recorded activations and then
frozen: unit-normalise each activation (cluster on *direction* — residual
norms grow smoothly with sequence position and would otherwise dominate the
top principal components), centre, project onto the top ``projection_dim``
principal components, then k-means (k-means++ init) in the projected space.

Assignment at fit or inference time repeats the same
normalise -> centre -> project transformation and picks the nearest centroid
by Euclidean distance. All parameters are plain tensors, so the fitted state
saves with ``torch.save`` and loads with ``torch.load(weights_only=True)``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch as t

from jlens.protocol import LensModel
from workspace_lens.utils import get_position_mask_with_early_skips, record_activations

logger = logging.getLogger(__name__)

### PURE COMPUTATIONS


def unit_normalize(points_PN: t.Tensor) -> t.Tensor:
    """Scale each row to unit L2 norm (a zero row would produce NaNs; residual
    stream activations are not zero in practice)."""
    return points_PN / points_PN.norm(dim=-1, keepdim=True)


def normalize_center_project(
    points_BsN: t.Tensor, pca_mean_N: t.Tensor, pca_components_ND: t.Tensor
) -> t.Tensor:
    """The routing transformation, normalise -> centre -> project:
    ``[..., d_model] -> [..., D]``. The single definition shared by router
    fitting and inference-time assignment, so the two can never diverge."""
    return (unit_normalize(points_BsN.float()) - pca_mean_N) @ pca_components_ND


def fit_pca(points_PN: t.Tensor, projection_dim: int) -> tuple[t.Tensor, t.Tensor]:
    """Top principal components of ``points_PN`` via the covariance eigendecomposition.

    Returns ``(mean_N, components_ND)``: the column ``components_ND[:, j]`` is
    the ``j``-th principal direction (descending eigenvalue order). Projection
    is ``(x - mean_N) @ components_ND``. The covariance route keeps memory at
    ``[N, N]`` regardless of the number of points.
    """
    num_points, d_model = points_PN.shape
    if not 0 < projection_dim <= d_model:
        raise ValueError(
            f"projection_dim must be in [1, d_model={d_model}], got {projection_dim}"
        )

    mean_N = points_PN.mean(dim=0)
    centered_PN = points_PN - mean_N
    covariance_NN = (centered_PN.T @ centered_PN) / max(num_points - 1, 1)

    # eigh returns eigenvalues ascending; take the trailing columns, reversed.
    _, eigenvectors_NN = t.linalg.eigh(covariance_NN)
    components_ND = eigenvectors_NN[:, -projection_dim:].flip(dims=[1])
    return mean_N, components_ND


def _kmeans_plusplus_init(
    points_PD: t.Tensor, num_clusters: int, generator: t.Generator
) -> t.Tensor:
    """k-means++ seeding: first centroid uniform, each next one sampled with
    probability proportional to squared distance to the nearest chosen centroid."""
    num_points = points_PD.shape[0]
    first_index = int(
        t.randint(num_points, (1,), generator=generator, device=points_PD.device)
    )
    centroids_E_list_D: list[t.Tensor] = [points_PD[first_index]]

    for _ in range(num_clusters - 1):
        chosen_ED = t.stack(centroids_E_list_D)
        squared_distance_P = t.cdist(points_PD, chosen_ED).pow(2).min(dim=1).values
        # clamp: multinomial rejects an all-zero weight vector, which occurs
        # only if every point coincides with a chosen centroid (duplicate
        # activations); the clamp degrades that case to uniform sampling.
        next_index = int(
            t.multinomial(squared_distance_P.clamp(min=1e-12), 1, generator=generator)
        )
        centroids_E_list_D.append(points_PD[next_index])
    return t.stack(centroids_E_list_D)


def kmeans(
    points_PD: t.Tensor,
    num_clusters: int,
    *,
    seed: int = 0,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> tuple[t.Tensor, t.Tensor]:
    """Lloyd's k-means with k-means++ init. Returns ``(centroids_ED,
    assignments_Int_P)``.

    A cluster that loses every point is re-seeded to the point currently
    farthest from its assigned centroid. Deterministic given ``seed`` on a
    fixed device (the generator lives on the points' device).
    """
    num_points = points_PD.shape[0]
    if num_points < num_clusters:
        raise ValueError(f"need >= {num_clusters} points, got {num_points}")

    generator = t.Generator(device=points_PD.device).manual_seed(seed)
    centroids_ED = _kmeans_plusplus_init(points_PD, num_clusters, generator)

    assignments_Int_P = t.zeros(num_points, dtype=t.long, device=points_PD.device)
    for _ in range(max_iterations):
        distances_PE = t.cdist(points_PD, centroids_ED)
        assignments_Int_P = distances_PE.argmin(dim=1)

        new_centroid_list_D: list[t.Tensor] = []
        for cluster_idx in range(num_clusters):
            member_mask_Bool_P = assignments_Int_P == cluster_idx
            if member_mask_Bool_P.any():
                new_centroid_list_D.append(points_PD[member_mask_Bool_P].mean(dim=0))
            else:
                # Empty cluster: re-seed to the point farthest from its
                # centroid. (If several clusters empty in one iteration they
                # all re-seed to the same point and coincide until a later
                # iteration separates them — accepted as a rare non-event.)
                farthest_index = int(
                    distances_PE.gather(1, assignments_Int_P[:, None]).argmax()
                )
                new_centroid_list_D.append(points_PD[farthest_index])
        new_centroids_ED = t.stack(new_centroid_list_D)

        centroid_shift = (new_centroids_ED - centroids_ED).norm(dim=1).max()
        centroids_ED = new_centroids_ED
        if centroid_shift < tolerance:
            break

    # Re-assign against the *final* centroids so the returned pair is
    # consistent (the in-loop assignments used the previous iteration's).
    assignments_Int_P = t.cdist(points_PD, centroids_ED).argmin(dim=1)
    return centroids_ED, assignments_Int_P


@t.no_grad()
def collect_valid_position_activations(
    model: LensModel,
    prompts: Sequence[str],
    layers: Sequence[int],
    *,
    skip_first_n_positions: int,
    max_seq_len: int,
) -> dict[int, t.Tensor]:
    """Pool residual activations for clustering: one forward pass per prompt,
    keeping exactly the positions the fit will route
    (:func:`workspace_lens.utils.get_position_mask_with_early_skips`). Returns
    ``{layer: [total_positions, d_model]}`` as fp32 CPU tensors; prompts
    too short to leave any valid position are skipped with a warning.
    """
    collected_activations_L_dict_list_PN: dict[int, list[t.Tensor]] = {
        layer: [] for layer in layers
    }
    for prompt_idx, prompt in enumerate(prompts):
        input_ids_Int_1S, activations_L_dict_SN = record_activations(
            model, prompt, max_seq_len, list(layers)
        )
        try:
            mask_Bool_S = get_position_mask_with_early_skips(
                input_ids_Int_1S.shape[1], skip_first_n_positions
            )
        except ValueError as exc:
            logger.warning("  skipping clustering prompt %d: %s", prompt_idx, exc)
            continue
        source_positions_Int_P = t.where(mask_Bool_S)[0]
        for layer in layers:
            collected_activations_L_dict_list_PN[layer].append(
                activations_L_dict_SN[layer][source_positions_Int_P].float().cpu()
            )

    if not collected_activations_L_dict_list_PN[layers[0]]:
        raise ValueError("no prompt was long enough to collect activations from")
    return {
        layer: t.cat(collected_activations_L_dict_list_PN[layer]) for layer in layers
    }
