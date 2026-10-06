"""Fitting expert Jacobians through a RelP (LRP-modified) backward graph.

:class:`ExpertJacobianRelPTrainer` is :class:`~workspace_lens.fitting.expert_fitting.
ExpertJacobianTrainer` with exactly one change: the retained autograd graph is
built while :func:`workspace_lens.lrp.apply_lrp_rules` has swapped the model's
RMSNorm / gated-MLP forwards for detach-edited copies. Ordinary backward over
that graph computes RelP propagation coefficients instead of raw gradients, so
every downstream consumer is inherited unchanged and self-consistently:

- the Jacobian-row backward passes accumulate RelP rows into the per-cluster
  sums (the fitted experts are per-cluster mean RelP coefficient matrices);
- routing sees the recorded activations, which the surgery leaves
  value-identical (up to ``F.silu`` rounding noise), so cluster assignments and
  position counts match a standard fit on the same prompts;
- checkpointing and merging are unchanged: ``ExpertJacobianTrainer.write_checkpoint``
  embeds the resolved rule dict of ``config.lrp_mode`` in every checkpoint
  (for every trainer, so mode-none shards from either trainer merge) and the
  merge checks it (``config.lrp_mode`` is already in
  ``LensConfig.check_compatible``).

The surgery context only spans the forward pass: the detached ops persist in
the retained graph after the module forwards are restored, so the backward
passes that run later still read RelP coefficients. ``lrp_mode="none"`` makes
this trainer bit-identical to ``ExpertJacobianTrainer`` (pinned in
``tests/test_relp_fitting.py``).
"""

from __future__ import annotations

import logging

from jlens.protocol import LensModel
from workspace_lens import DEFAULT_DEVICE
from workspace_lens.config import LensConfig
from workspace_lens.fitting.expert_fitting import ExpertJacobianTrainer
from workspace_lens.fitting.types import FitStepForward
from workspace_lens.lrp import (
    LrpPatchReport,
    LrpRuleConfig,
    apply_lrp_rules_for_mode,
    lrp_rule_config_for_mode,
)
from workspace_lens.lrp.lrp import hf_module_to_patch
from workspace_lens.routing.router import ActivationRouterCollection

logger = logging.getLogger(__name__)


class ExpertJacobianRelPTrainer(ExpertJacobianTrainer):
    """Expert-Jacobian fitting with LRP surgery around the fit forward pass.

    The rule set is resolved from ``config.lrp_mode`` by
    :func:`workspace_lens.lrp.apply_lrp_rules_for_mode` (which also unwraps an
    HF adapter to its ``_hf_model``), so the mode is recorded in every
    checkpoint and built lens via the config.
    """

    applies_lrp_rules = True

    def __init__(
        self,
        config: LensConfig,
        model: LensModel,
        prompts: list[str],
        *,
        router_collections_K_dict: dict[int, ActivationRouterCollection],
        device: str = DEFAULT_DEVICE,
    ) -> None:
        super().__init__(
            config,
            model,
            prompts,
            router_collections_K_dict=router_collections_K_dict,
            device=device,
        )
        hf_module_to_patch(model)  # a model the surgery cannot patch fails here, not at the first fit step
        self.lrp_rule_config: LrpRuleConfig = lrp_rule_config_for_mode(config.lrp_mode)
        self.last_patch_report: LrpPatchReport | None = None

    def _run_fit_forward(self, prompt: str) -> FitStepForward:
        """The stock recorded forward, run under the LRP surgery so the
        retained graph carries the detach-edited ops. Restoring the module
        forwards on exit does not touch the recorded graph."""
        first_patch = self.last_patch_report is None
        with apply_lrp_rules_for_mode(self.model, self.config.lrp_mode) as report:
            forward_state = super()._run_fit_forward(prompt)
        if report is None:  # lrp_mode == "none": nothing was patched
            return forward_state
        self.last_patch_report = report
        if first_patch:
            logger.info(
                "LRP surgery (lrp_mode=%s): patched %d RMSNorms, %d gated MLPs",
                self.config.lrp_mode,
                len(report.ln_rule),
                len(report.mlp_rule),
            )
        return forward_state
