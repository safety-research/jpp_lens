from __future__ import annotations

from dataclasses import dataclass

import torch as t


@dataclass
class FitStepForward:
    """One recorded fit-step forward pass: everything the backward passes in
    :meth:`LensTrainer._backward_passes` need, plus the prompt's token ids. The
    activation tensors are still attached to the retained autograd graph. The
    stream axis of a multi-stream model (DeepSeek-V4) is explained in
    :mod:`workspace_lens.residual_streams` and on the two tensor fields below.

    One position set: the kept positions ``[P]``
    (:func:`workspace_lens.utils.get_position_mask_with_early_skips`) are where
    the one-hot cotangent is placed and where the gradient rows are read, routed
    and averaged, so the row at position ``p`` sums the per-pair Jacobians over
    the kept positions ``p' >= p``.

    Devices: ``source_positions_Int_P`` lives on the target residual's device (the
    cotangent is indexed there); ``input_ids_Int_S`` is wherever ``model.encode``
    put it. Consumers move what they read.
    """

    seq_len: int
    num_source_positions: int
    source_positions_Int_P: t.Tensor
    # The lens's target residual: on a multi-stream model the (attached) mean over the streams, so never an R axis.
    target_residual_BSF: t.Tensor
    # The autograd inputs, as the model produced them: [B, S, N], or [B, S, R, N] with the streams intact on a
    # multi-stream model. Collapsing them here would cut the graph; the backward sums each gradient over R instead.
    source_block_outputs_L_list_BSrN: list[t.Tensor]
    # The prompt's token ids, ``[seq_len]`` long (hashed into the prompt's position record).
    input_ids_Int_S: t.Tensor


@dataclass
class LayerClusterSums:
    """One layer of one K's expert fit: the per-expert sufficient statistics.

    ``weighted_jacobian_row_sum_EFN`` (``[E, F, N]``: one ``[F, N]`` Jacobian
    row sum per expert) is the running ``sum_p G_p`` over the positions routed
    to each expert; ``position_count_E`` is how many positions routed to each
    expert (int64: the empty-expert fallback signal, also reported on the built
    lens). Every kept position enters with weight one, so ``weight_sum_E``, the
    matching ``sum_p 1`` in fp32, equals the count. Dividing row sums by weight
    sums gives the expert Jacobians; shards merge by summing all three fields.
    """

    weighted_jacobian_row_sum_EFN: t.Tensor
    weight_sum_E: t.Tensor
    position_count_E: t.Tensor


@dataclass
class ExpertFitSums:
    """One K's sufficient statistics over all source layers — everything
    :meth:`~workspace_lens.fitting.condense_experts.ExpertJacobians.from_sums` needs.

    Checkpoints store this layer-major, mirroring the class
    (``{"layer_sums": {layer: {field: tensor}}}``).
    """

    layer_sums_L_dict: dict[int, LayerClusterSums]

    @classmethod
    def zeros(cls, num_experts: int, d_model: int, layers: list[int]) -> ExpertFitSums:
        return cls(
            layer_sums_L_dict={
                layer: LayerClusterSums(
                    weighted_jacobian_row_sum_EFN=t.zeros(
                        num_experts, d_model, d_model, dtype=t.float32
                    ),
                    weight_sum_E=t.zeros(num_experts, dtype=t.float32),
                    position_count_E=t.zeros(num_experts, dtype=t.long),
                )
                for layer in layers
            }
        )

    def add_(self, other: ExpertFitSums) -> None:
        """Accumulate another shard's sums in place (exact: every field is a
        plain sum over disjoint prompt sets)."""
        for layer, layer_sums in self.layer_sums_L_dict.items():
            other_sums = other.layer_sums_L_dict[layer]
            layer_sums.weighted_jacobian_row_sum_EFN += other_sums.weighted_jacobian_row_sum_EFN
            layer_sums.weight_sum_E += other_sums.weight_sum_E
            layer_sums.position_count_E += other_sums.position_count_E

    def to_state_dict(self) -> dict:
        """Plain nested dicts of tensors, safe for ``weights_only=True``."""
        return {
            "layer_sums": {
                layer: {
                    "weighted_jacobian_row_sum_EFN": layer_sums.weighted_jacobian_row_sum_EFN,
                    "weight_sum_E": layer_sums.weight_sum_E,
                    "position_count_E": layer_sums.position_count_E,
                }
                for layer, layer_sums in self.layer_sums_L_dict.items()
            }
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> ExpertFitSums:
        """Read a checkpoint's sums (the :meth:`to_state_dict` layout)."""
        return cls(
            layer_sums_L_dict={
                layer: LayerClusterSums(**field_tensors)
                for layer, field_tensors in state["layer_sums"].items()
            }
        )
