"""Residual-stream interventions built from J-lens vectors.

Public surface of the ``workspace_lens.interventions`` package; implementations
live in the submodules (``base``, ``hooks``, ``interventions``).
"""

from workspace_lens.interventions.base import Intervention
from workspace_lens.interventions.hooks import InterventionHooks
from workspace_lens.interventions.interventions import Clamp

__all__ = [
    "Clamp",
    "Intervention",
    "InterventionHooks",
]
