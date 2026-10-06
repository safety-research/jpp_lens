"""LRP/RelP rules: detach-edited module forwards so that ordinary autograd
over the recorded graph computes RelP propagation coefficients. Rule
surgery in ``lrp.py``."""

from workspace_lens.lrp.lrp import (
    LrpPatchReport,
    LrpRuleConfig,
    apply_lrp_rules,
    apply_lrp_rules_for_mode,
    lrp_rule_config_for_mode,
)
from workspace_lens.lrp.types import LRP_MODES, LrpMode

__all__ = [
    "LRP_MODES",
    "LrpMode",
    "LrpPatchReport",
    "LrpRuleConfig",
    "apply_lrp_rules",
    "apply_lrp_rules_for_mode",
    "lrp_rule_config_for_mode",
]
