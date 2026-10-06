"""Causal swap evals: does the model compute *through* the lens's directions?

The runner (:class:`SwapEvalRunner`) exchanges one concept's residual-stream
coordinate for another along a lens's per-token directions during a single
forward pass, then scores the model's own next-token distribution. The
probe-swap loader (:func:`load_probe_swap_trials`) builds :class:`SwapTrial`
specs.
"""

from lens_evals.causal_evals.probe_swap import (
    best_scale_rows,
    load_probe_swap_trials,
    probe_swap_formatting_token_ids,
    probe_swap_success_table,
    run_probe_swap,
    slice_lens_to_layers,
)
from lens_evals.causal_evals.runner import (
    SwapEvalRunner,
    SwapTrial,
    SwapTrialResult,
)

__all__ = [
    "SwapEvalRunner",
    "SwapTrial",
    "SwapTrialResult",
    "best_scale_rows",
    "load_probe_swap_trials",
    "probe_swap_formatting_token_ids",
    "probe_swap_success_table",
    "run_probe_swap",
    "slice_lens_to_layers",
]
