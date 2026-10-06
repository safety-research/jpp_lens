from dataclasses import dataclass

import torch as t

from workspace_lens.routing.cluster import (
    fit_pca,
    kmeans,
    normalize_center_project,
    unit_normalize,
)


@dataclass
class LayerActivationRouter:
    """The frozen clustering for one layer: PCA projection plus centroids."""

    pca_mean_N: t.Tensor
    pca_components_ND: t.Tensor
    centroids_ED: t.Tensor

    def project(self, residual_BsN: t.Tensor) -> t.Tensor:
        """:func:`normalize_center_project` with this layer's parameters."""
        return normalize_center_project(
            residual_BsN,
            self.pca_mean_N.to(residual_BsN.device),
            self.pca_components_ND.to(residual_BsN.device),
        )

    def assign(self, residual_BsN: t.Tensor) -> t.Tensor:
        """Nearest-centroid cluster index for each activation: ``[...]`` (long)."""
        projected_BsD = self.project(residual_BsN)

        centroids_ED = self.centroids_ED.to(residual_BsN.device)

        flat_PD = projected_BsD.reshape(-1, projected_BsD.shape[-1])
        assignments_Int_P = t.cdist(flat_PD, centroids_ED).argmin(dim=1)
        return assignments_Int_P.reshape(projected_BsD.shape[:-1])


class ActivationRouterCollection:
    """Per-layer PCA + k-means clustering of residual activations.

    Fit once (:meth:`fit`) on pooled valid-position activations, then frozen:
    :class:`~workspace_lens.fitting.expert_fitting.ExpertJacobianTrainer` routes fitting
    positions through :meth:`assign`, one expert Jacobian per cluster. The router is
    used only while fitting: the J++ Lens combines the experts into one map per layer,
    so inference needs no router.
    """

    def __init__(
        self,
        num_clusters: int,
        projection_dim: int,
        *,
        seed: int = 0,
        kmeans_max_iterations: int = 100,
    ) -> None:
        self.num_clusters = num_clusters
        self.projection_dim = projection_dim
        self.seed = seed
        self.kmeans_max_iterations = kmeans_max_iterations
        self.layer_routers_L_dict: dict[int, LayerActivationRouter] = {}

    def fit(self, activations_L_dict_PN: dict[int, t.Tensor]) -> None:
        """Fit each layer's PCA and centroids from pooled activations
        ``{layer: [num_points, d_model]}``."""

        for layer, activations_PN in activations_L_dict_PN.items():

            normalized_PN = unit_normalize(activations_PN.float())

            pca_mean_N, pca_components_ND = fit_pca(normalized_PN, self.projection_dim)

            projected_PD = normalize_center_project(
                activations_PN, pca_mean_N, pca_components_ND
            )

            centroids_ED, _ = kmeans(
                projected_PD,
                self.num_clusters,
                seed=self.seed,
                max_iterations=self.kmeans_max_iterations,
            )

            self.layer_routers_L_dict[layer] = LayerActivationRouter(
                pca_mean_N=pca_mean_N,
                pca_components_ND=pca_components_ND,
                centroids_ED=centroids_ED,
            )

    def assign(self, residual_BsN: t.Tensor, layer: int) -> t.Tensor:
        """Cluster index for each activation at ``layer``: ``[...]`` (long)."""
        return self.layer_routers_L_dict[layer].assign(residual_BsN)

    ### SAVING AND LOADING (plain tensors only: weights_only=True safe)

    def save(self, path: str) -> None:
        t.save(
            {
                "num_clusters": self.num_clusters,
                "projection_dim": self.projection_dim,
                "seed": self.seed,
                "kmeans_max_iterations": self.kmeans_max_iterations,
                "pca_means": {
                    layer: layer_router.pca_mean_N.cpu()
                    for layer, layer_router in self.layer_routers_L_dict.items()
                },
                "pca_components": {
                    layer: layer_router.pca_components_ND.cpu()
                    for layer, layer_router in self.layer_routers_L_dict.items()
                },
                "centroids": {
                    layer: layer_router.centroids_ED.cpu()
                    for layer, layer_router in self.layer_routers_L_dict.items()
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "ActivationRouterCollection":
        state = t.load(path, map_location="cpu", weights_only=True)
        router_collection = cls(
            num_clusters=state["num_clusters"],
            projection_dim=state["projection_dim"],
            seed=state["seed"],
            kmeans_max_iterations=state["kmeans_max_iterations"],
        )
        router_collection.layer_routers_L_dict = {
            layer: LayerActivationRouter(
                pca_mean_N=state["pca_means"][layer],
                pca_components_ND=state["pca_components"][layer],
                centroids_ED=state["centroids"][layer],
            )
            for layer in state["centroids"]
        }
        return router_collection
