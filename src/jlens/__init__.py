# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
# Modified by Kola Ayonrinde, 2026.
"""Jacobian lens: model-loading and hook utilities.

Fitting and applying lenses lives in :mod:`workspace_lens` (see
:class:`workspace_lens.fitting.LensTrainer` and
:mod:`workspace_lens.lenses`)."""

from jlens._logging import configure_logging
from jlens.hf import HFLensModel, Layout, from_hf
from jlens.hooks import ActivationRecorder
from jlens.protocol import LensModel

__all__ = [
    "ActivationRecorder",
    "HFLensModel",
    "Layout",
    "LensModel",
    "configure_logging",
    "from_hf",
]
