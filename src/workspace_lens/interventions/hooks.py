from collections.abc import Callable, Sequence

import torch as t
from torch import nn

from jlens.protocol import LensModel
from workspace_lens.interventions.base import Intervention
from workspace_lens.residual_streams import has_residual_streams


class InterventionHooks:
    """Applies interventions during forward passes via writer forward hooks.

    Registers one hook per targeted layer on ``model.layers[layer]``; the hook
    *replaces* block ``layer``'s output (the vendored
    :class:`jlens.hooks.ActivationRecorder` only reads). Interventions sharing
    a layer are applied in the order given in ``interventions``. Bare-tensor
    block outputs (``TinyDecoder``, and Qwen/Llama decoder layers in current
    transformers) are returned edited directly; tuple outputs are rebuilt as
    ``(edited_hidden, *rest)`` — a compatibility path for other model families
    or older transformers versions, not exercised by this repo's target models
    (unit-test coverage only).

    Hooks on one module run in registration order and each sees the previous
    hook's returned output, so enter this context *before* an
    ``ActivationRecorder`` that should observe edited residuals.

    A single instance must not be entered twice concurrently (re-entry raises
    ``RuntimeError``); nesting *distinct* instances is fine — each removes
    only its own handles on exit.
    """

    def __init__(self, model: LensModel, interventions: Sequence[Intervention]) -> None:
        self.model = model
        self.interventions = list(interventions)
        self._interventions_by_layer: dict[int, list[Intervention]] = {}
        for intervention in self.interventions:
            for layer in intervention.layers:
                if not 0 <= layer < model.n_layers:
                    raise ValueError(
                        f"layer {layer} out of range for a " f"{model.n_layers}-layer model"
                    )
                self._interventions_by_layer.setdefault(layer, []).append(intervention)
        self._handles: list[t.utils.hooks.RemovableHandle] = []

    def _make_hook(
        self, layer: int, layer_interventions: list[Intervention]
    ) -> Callable[..., t.Tensor | tuple]:
        def hook(
            module: nn.Module, inputs: tuple, output: t.Tensor | tuple
        ) -> t.Tensor | tuple:
            block_output_BSrN = output if t.is_tensor(output) else output[0]
            if has_residual_streams(block_output_BSrN):
                # A multi-stream residual (DeepSeek-V4) would need a lift from the lens's mean residual
                # back to the streams (add the mean's change to every stream); not implemented.
                raise NotImplementedError("interventions do not lift to a multi-stream residual yet")
            residual_BSN = block_output_BSrN  # single stream: the block output is the residual
            for intervention in layer_interventions:
                residual_BSN = intervention.edit(residual_BSN, layer)
            if t.is_tensor(output):
                return residual_BSN
            return (residual_BSN, *output[1:])

        return hook

    def __enter__(self) -> "InterventionHooks":
        if self._handles:
            raise RuntimeError(
                "this InterventionHooks is already active; exit it first or "
                "create a new instance (re-entry would register every hook "
                "twice and the inner exit would strip the outer context's hooks)"
            )
        try:
            for layer, layer_interventions in sorted(self._interventions_by_layer.items()):
                self._handles.append(
                    self.model.layers[layer].register_forward_hook(
                        self._make_hook(layer, layer_interventions)
                    )
                )
        except Exception:
            for handle in self._handles:
                handle.remove()
            self._handles = []
            raise
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
