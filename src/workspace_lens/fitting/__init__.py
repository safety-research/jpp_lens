"""Lens fitting.

Implementations live in the submodules — ``jacobian_fitting`` (the J-lens
estimator and ``LensTrainer``), ``expert_fitting`` (``ExpertJacobianTrainer``, one Jacobian
sum per router cluster), ``relp_fitting`` (the same under the LRP backward pass), ``types``
(``FitStepForward``, ``LayerClusterSums``, ``ExpertFitSums``), ``utils``
(shard-checkpoint merging and ``build_layer_expert_jacobians``), and
``condense_experts`` (expert Jacobians and the expert-weight fit:
``CondenseConfig``, ``ExpertJacobians``, ``ExpertWeightingFitter``,
``ExpertWeighting`` / ``LearnedWeights`` and the combined ``JacobianLens``).
The public names are re-exported here.
"""

from workspace_lens.fitting.condense_experts import (
    CondenseConfig,
    ExpertJacobians,
    ExpertWeighting,
    ExpertWeightingFitter,
    LearnedWeights,
)
from workspace_lens.fitting.expert_fitting import (
    ExpertJacobianTrainer,
    build_matched_baseline_lens,
)
from workspace_lens.fitting.jacobian_fitting import LensTrainer
from workspace_lens.fitting.relp_fitting import ExpertJacobianRelPTrainer
from workspace_lens.fitting.types import (
    ExpertFitSums,
    FitStepForward,
    LayerClusterSums,
)
from workspace_lens.fitting.utils import (
    MergedExpertCheckpoint,
    build_layer_expert_jacobians,
    expert_checkpoint_filename,
    merge_expert_checkpoints,
)

__all__ = [
    "CondenseConfig",
    "ExpertJacobians",
    "ExpertWeighting",
    "ExpertWeightingFitter",
    "FitStepForward",
    "LayerClusterSums",
    "LearnedWeights",
    "LensTrainer",
    "MergedExpertCheckpoint",
    "ExpertFitSums",
    "ExpertJacobianRelPTrainer",
    "ExpertJacobianTrainer",
    "build_matched_baseline_lens",
    "merge_expert_checkpoints",
    "build_layer_expert_jacobians",
    "expert_checkpoint_filename",
]
