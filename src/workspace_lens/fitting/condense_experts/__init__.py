"""Condensing an expert fit's expert Jacobians into one linear map per layer:
``J_l(w) = sum_e w_e J_{l,e}``, with the weights fitted to maximise the readout
eval's pair-level pass@k on labelled items — the last fitting step of the J++
Lens.

Three modules:

- :mod:`expert_jacobians` — :class:`ExpertJacobians`, one K's per-expert
  Jacobians (the fit-time artifact, before any weights are learned), and the
  weighted sum that turns them into the inference
  :class:`~workspace_lens.lenses.jacobian_lens.JacobianLens`.
- :mod:`rank_objective` — the eval's rank rule as tensor maths: the pair table,
  the cached per-expert logits (linear in the weights, so a candidate weighting
  is scored without touching the model), the hit margins and the macro means.
- :mod:`fitter` — the workflow: :class:`CondenseConfig` (the search's
  hyperparameters), :class:`ExpertWeightingFitter` (holds the model, the
  experts and the labelled items' residuals), and its result
  :class:`ExpertWeighting` (one :class:`LearnedWeights` per layer, saved as
  JSON beside the combined lens).
"""

from workspace_lens.fitting.condense_experts.expert_jacobians import ExpertJacobians
from workspace_lens.fitting.condense_experts.fitter import (
    CondenseConfig,
    ExpertWeighting,
    ExpertWeightingFitter,
    LearnedWeights,
)

__all__ = [
    "CondenseConfig",
    "ExpertJacobians",
    "ExpertWeighting",
    "ExpertWeightingFitter",
    "LearnedWeights",
]
