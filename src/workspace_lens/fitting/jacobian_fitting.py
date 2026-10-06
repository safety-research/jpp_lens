# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
# Modified by Kola Ayonrinde, 2026.
"""Fitting the Jacobian lens.

The lens reads out an early-layer residual ``h_l`` by linearly transporting it
into the final-layer basis with the average input-output Jacobian, then
decoding with the model's own unembedding::

    lens_l(h) = unembed( J_l @ h )

Estimator (:meth:`LensTrainer.fit_step`): for each output dimension, inject a
one-hot cotangent at *every kept position at once* and backprop. The kept
positions are the valid positions
(:func:`workspace_lens.utils.get_position_mask_with_early_skips`). The gradient
at a kept position ``p`` is then ``sum_{kept p' >= p} dh_final[p'] / dh_l[p]``,
the sum over the kept positions at or after ``p``, and ``J_l`` is its mean over
the kept positions ``p``. A per-position estimator (``dh_final[p] / dh_l[p]``
averaged over ``p``) gives a slightly different ``J_l``; both work as a lens.

Cost: one forward pass and ``ceil(d_model / jacobian_rows_per_pass)`` backward passes per
prompt.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Optional, Self

import torch as t

from jlens.hooks import ActivationRecorder
from jlens.protocol import LensModel
from workspace_lens import DEFAULT_DEVICE, get_hf_model
from workspace_lens.config import LensConfig
from workspace_lens.fitting.types import FitStepForward
from workspace_lens.fitting.utils import source_gradients_summed_over_streams
from workspace_lens.lenses.jacobian_lens import JacobianLens
from workspace_lens.residual_streams import (
    residual_mean_over_streams,
)
from workspace_lens.utils import (
    check_model_matches_config,
    get_position_mask_with_early_skips,
    resolve_target_layer,
    token_ids_sha256,
)

logger = logging.getLogger(__name__)


class LensTrainer:
    # Subclasses that run the LRP surgery themselves (ExpertJacobianRelPTrainer)
    # set this True; everyone else must not carry a non-"none" lrp_mode,
    # which would stamp RelP provenance onto a standard-gradient fit.
    applies_lrp_rules: bool = False

    def __init__(
        self,
        config: LensConfig,
        model: LensModel,
        prompts: list[str],
        *,
        device: str = DEFAULT_DEVICE,
    ) -> None:
        if config.lrp_mode != "none" and not self.applies_lrp_rules:
            raise ValueError(
                f"config.lrp_mode={config.lrp_mode!r} but "
                f"{type(self).__name__} does not apply the LRP surgery — the "
                "artifact would carry RelP provenance over standard "
                "gradients. Fit with ExpertJacobianRelPTrainer."
            )
        self.config = config
        self.model = model
        self.prompts = prompts
        self.device = device

        self.config.d_model = model.d_model
        self.sqrt_d_model = math.sqrt(model.d_model)

        self.target_layer = resolve_target_layer(
            model.n_layers, config.relative_end_transport_layer
        )
        self.source_layers = self._resolve_source_layers()

        # Two independent facts about this accumulator: (1) from_checkpoint
        # overwrites it after construction (resume); (2) subclasses with their
        # own accumulators override _init_accumulators to skip allocating the
        # (large) J-lens sum they never use.
        self.jacobian_sum_L_dict_FN: dict[int, t.Tensor] = {}
        self._init_accumulators()

        self.completed_prompt_count = 0
        self.next_prompt_idx = 0
        # One plain dict per fitted prompt (:meth:`_append_position_record`): the
        # kept positions, stored in every checkpoint under "position_records" and
        # in a JSON sidecar; the merge does not read them.
        self.position_records: list[dict[str, object]] = []

    def _init_accumulators(self) -> None:
        self.jacobian_sum_L_dict_FN = {
            layer: t.zeros(self.model.d_model, self.model.d_model, dtype=t.float32)
            for layer in self.source_layers
        }

    ### INITIALISATION

    def _resolve_source_layers(self) -> list[int]:
        """``config.source_layers`` as sorted absolute indices (negative
        indices count from the end); ``None`` means every layer before the
        target. Every source layer must lie strictly before the target."""
        num_layers = self.model.n_layers
        source_layers = self.config.source_layers

        if source_layers is None:
            return list(range(self.target_layer))

        resolved_source_layers = sorted(
            {layer + num_layers if layer < 0 else layer for layer in source_layers}
        )

        if (
            not resolved_source_layers
            or resolved_source_layers[0] < 0
            or resolved_source_layers[-1] >= num_layers
        ):
            raise ValueError(
                f"source_layers {sorted(source_layers)} out of range for {num_layers} layers"
            )
        elif resolved_source_layers[-1] >= self.target_layer:
            raise ValueError(
                f"source_layers must all be < target_layer={self.target_layer}; "
                f"got max={resolved_source_layers[-1]}"
            )

        return resolved_source_layers

    ### TRAINING

    def fit_step(
        self,
        prompt: str,
        *,
        prompt_idx: Optional[int] = None,  # noqa: UP045
    ) -> tuple[dict[int, t.Tensor], int, int]:
        """Compute the per-layer Jacobian estimator ``J_l`` for one prompt.

        Runs one forward pass on the prompt replicated ``jacobian_rows_per_pass`` times along
        the batch axis, retains the graph, then runs ``ceil(d_model / jacobian_rows_per_pass)``
        backward passes against it. Each backward computes ``jacobian_rows_per_pass`` rows of
        ``J_l`` at once: batch element ``b`` carries a one-hot cotangent at output
        dimension ``dim_start + b``, set at every kept position; the rows are read
        and averaged at the same positions. See the module docstring for
        the resulting estimator and how it relates to a strict per-position Jacobian.
        Side effect: appends this prompt's position record (:meth:`_append_position_record`).

        Args:
            prompt: Input text.
            prompt_idx: The prompt's index in the trainer's list, stored in the record
                (``None`` for a stand-alone call).

        Returns:
            ``(jacobians, seq_len, num_source_positions)``. ``jacobians`` maps each
            source layer to a ``[F, N]`` fp32 CPU tensor (rows: target-layer
            dims, columns: source-layer dims).
        """
        d_model = self.model.d_model

        forward_state = self._run_fit_forward(prompt)

        jacobians_L_dict_FN = {
            layer: t.zeros(d_model, d_model, dtype=t.float32) for layer in self.source_layers
        }

        for dim_start, current_pass_batch_dim, grads_L_list_BSN in self._backward_passes(
            forward_state
        ):
            for layer, grad_BSN in zip(self.source_layers, grads_L_list_BSN, strict=True):
                # mean over the kept positions -> jacobian_rows_per_pass rows of J_l
                positions_on_device_Int_P = forward_state.source_positions_Int_P.to(
                    grad_BSN.device, non_blocking=True
                )
                source_grad_BPN = grad_BSN[
                    :current_pass_batch_dim, positions_on_device_Int_P, :
                ].float()
                jacobians_L_dict_FN[layer][
                    dim_start : dim_start + current_pass_batch_dim, :
                ] = source_grad_BPN.mean(dim=1).cpu()

        self._append_position_record(forward_state, prompt_idx)
        return jacobians_L_dict_FN, forward_state.seq_len, forward_state.num_source_positions

    def _run_fit_forward(self, prompt: str) -> FitStepForward:
        """One recorded forward pass on ``prompt`` replicated ``jacobian_rows_per_pass``
        times, with the autograd graph retained from ``min(source_layers)``
        onward, plus the kept positions.

        Raises:
            ValueError: If the prompt is too short to leave any kept
                position (callers skip-and-warn, as :meth:`fit` does).
        """
        input_ids_Int_1S = self.model.encode(prompt, max_length=self.config.max_seq_len)
        _, seq_len = input_ids_Int_1S.shape

        # The cotangent is placed at the valid positions and the rows are read there.
        position_mask_Bool_S = get_position_mask_with_early_skips(
            seq_len, self.config.skip_first_n_positions
        )
        source_positions_Int_P = t.where(position_mask_Bool_S)[0]

        with (
            ActivationRecorder(
                self.model.layers,
                at=[*self.source_layers, self.target_layer],
                start_graph_at=min(self.source_layers),
            ) as recorder,
            t.enable_grad(),
        ):
            # One forward on the prompt replicated jacobian_rows_per_pass times. The retained
            # graph is reused for every backward pass.
            batched_input_ids_Int_BS = input_ids_Int_1S.expand(self.config.jacobian_rows_per_pass, -1)

            _ = self.model.forward(batched_input_ids_Int_BS)

            # The lens's residual is the stream mean on a multi-stream model (the mean node
            # stays attached, so autograd spreads the target cotangent over the streams).
            target_residual_BSF = residual_mean_over_streams(recorder.activations[self.target_layer])
            # The sources keep their stream axis ([B, S, R, N] on a multi-stream model): they are the autograd
            # inputs below, and a mean taken here would not be a leaf the Jacobian can be taken with respect to.
            source_block_outputs_L_list_BSrN = [
                recorder.activations[layer] for layer in self.source_layers
            ]

        return FitStepForward(
            seq_len=seq_len,
            num_source_positions=int(source_positions_Int_P.numel()),
            source_positions_Int_P=source_positions_Int_P.to(target_residual_BSF.device),
            target_residual_BSF=target_residual_BSF,
            source_block_outputs_L_list_BSrN=source_block_outputs_L_list_BSrN,
            input_ids_Int_S=input_ids_Int_1S[0],
        )

    def _backward_passes(
        self, forward_state: FitStepForward
    ) -> Iterator[tuple[int, int, tuple[t.Tensor, ...]]]:
        """The estimator's backward passes over ``forward_state``'s retained
        graph, one yield per pass: ``(dim_start, current_pass_batch_dim,
        grads_L_list_BSN)``.

        Batch element ``b`` of each pass carries a one-hot cotangent at output
        dimension ``dim_start + b``, set at every kept position (and nowhere
        else), so ``grads_L_list_BSN[i][b, p, :]`` is row ``dim_start + b`` of the
        position-``p`` Jacobian estimate at ``source_layers[i]``, summed over the
        kept positions ``p' >= p``. Consumers reduce over the kept positions
        however they like (mean for the J-lens, per-cluster sums for the expert
        Jacobians). The number of passes does not depend on the positions. The graph
        is freed on the final pass, so the generator must be run to exhaustion.

        On a multi-stream model the target is the stream mean (autograd spreads
        the cotangent over the streams) and each source gradient is summed over
        its streams, so the rows are those of the Jacobian of the target mean with
        respect to a uniform perturbation of the source streams
        (:mod:`residual_streams`).
        """
        d_model = self.model.d_model
        jacobian_rows_per_pass = self.config.jacobian_rows_per_pass
        num_passes_for_step = math.ceil(d_model / jacobian_rows_per_pass)

        target_residual_BSF = forward_state.target_residual_BSF
        positions_Int_P = forward_state.source_positions_Int_P
        batch_indices_Int_B = t.arange(jacobian_rows_per_pass, device=target_residual_BSF.device)

        cotangent_BSF = t.zeros_like(target_residual_BSF)

        for pass_idx, dim_start in enumerate(range(0, d_model, jacobian_rows_per_pass)):
            current_pass_batch_dim = min(jacobian_rows_per_pass, d_model - dim_start)
            # One-hot cotangent at dim (dim_start + b) for batch element b,
            # at every kept position. Yields rows dim_start..+n of J_l.
            cotangent_BSF.zero_()
            cotangent_BSF[
                batch_indices_Int_B[:current_pass_batch_dim, None],
                positions_Int_P[None, :],
                dim_start + batch_indices_Int_B[:current_pass_batch_dim, None],
            ] = 1.0

            # Per-stream gradients [B, S, R, N] become the lens's gradient [B, S, N] by summing over R: the
            # derivative of the stream mean under a uniform shift of the streams (residual_streams.py).
            grads_L_list_BSN = tuple(
                source_gradients_summed_over_streams(
                    forward_state, cotangent_BSF, retain_graph=(pass_idx < num_passes_for_step - 1)
                )
            )

            yield dim_start, current_pass_batch_dim, grads_L_list_BSN

            del grads_L_list_BSN

            if pass_idx % 100 == 0 or pass_idx == num_passes_for_step - 1:
                logger.debug(
                    "    pass %d/%d (dims %d-%d)",
                    pass_idx + 1,
                    num_passes_for_step,
                    dim_start,
                    dim_start + current_pass_batch_dim,
                )

    def _fit_prompt(self, prompt_idx: int, prompt: str, prompt_start_time: float) -> None:
        """One prompt of the fitting loop: run the estimator, log, accumulate.
        Subclasses override this (with their own fit_step and accumulators)
        while sharing :meth:`_run_prompt_loop`'s scaffolding."""
        prompt_jacobians_L_dict_FN, _, _ = self.fit_step(prompt, prompt_idx=prompt_idx)
        self._diagnostic_logging(prompt_idx, prompt_start_time, prompt_jacobians_L_dict_FN)
        for layer in self.source_layers:
            self.jacobian_sum_L_dict_FN[layer] += prompt_jacobians_L_dict_FN[layer]

    def _append_position_record(
        self,
        forward_state: FitStepForward,
        prompt_idx: Optional[int],  # noqa: UP045
    ) -> None:
        """Record one fitted prompt's kept positions as plain types (checkpoint key
        ``"position_records"`` and the ``position_records.json`` sidecar): the prompt's
        index, token-id hash and length, and the kept positions and their count."""
        self.position_records.append(
            {
                "prompt_idx": prompt_idx,
                "token_ids_sha256": token_ids_sha256(forward_state.input_ids_Int_S),
                "seq_len": forward_state.seq_len,
                "num_source_positions": forward_state.num_source_positions,
                "source_positions": forward_state.source_positions_Int_P.cpu().tolist(),
            }
        )

    def _prompt_progress_line(self, prompt_idx: int, prompt_start_time: float) -> str:
        """The shared head of every trainer's per-prompt log line, formatted from the
        prompt's position record (the one :meth:`fit_step` just appended); a trainer
        logs its own metrics after it."""
        record = self.position_records[-1]
        return (
            f"  prompt {prompt_idx + 1}/{len(self.prompts)}  seq_len={record['seq_len']} "
            f"num_source_positions={record['num_source_positions']}  "
            f"{time.perf_counter() - prompt_start_time:.0f}s"
        )

    def _run_prompt_loop(self) -> None:
        """The fitting loop shared by every trainer: resume skip, skip-and-warn
        on too-short prompts, success/resume counters, the checkpoint cadence,
        and the final checkpoint. ``next_prompt_idx`` is tracked separately
        from ``completed_prompt_count`` so a too-short prompt that was skipped
        is not re-processed on resume.

        The only ValueError expected from :meth:`_fit_prompt` is the too-short
        -prompt one raised by ``_run_fit_forward`` before any accumulator
        mutation, so catching it here cannot leave partial state behind.

        Raises:
            ValueError: If no prompt was long enough to fit on.
        """
        for prompt_idx, prompt in enumerate(self.prompts):
            if prompt_idx < self.next_prompt_idx:
                continue
            prompt_start_time = time.perf_counter()
            try:
                self._fit_prompt(prompt_idx, prompt, prompt_start_time)
            except ValueError as exc:
                logger.warning("  skipping prompt %d: %s", prompt_idx, exc)
                self.next_prompt_idx = prompt_idx + 1
                continue

            self.completed_prompt_count += 1
            self.next_prompt_idx = prompt_idx + 1
            if (
                self.config.checkpoint_every_n_prompts is not None
                and self.next_prompt_idx % self.config.checkpoint_every_n_prompts == 0
            ):
                self.write_checkpoint()

        self.write_checkpoint(final=True)
        if self.completed_prompt_count == 0:
            raise ValueError("no prompts were long enough to fit on")

    def fit(self) -> JacobianLens:
        """Fit ``J_l`` over a list of prompts and return a :class:`JacobianLens`.

        Per-prompt Jacobians from :meth:`fit_step` are accumulated as a running
        sum and divided by the prompt count at the end. Every
        ``config.checkpoint_every_n_prompts`` prompts, the running sum
        is written (atomically) to ``<config.checkpoint_path>/<N>_checkpoint.pt``;
        a final ``final_checkpoint.pt`` is always written. Resume from a
        checkpoint file with :meth:`from_checkpoint`. Checkpoints can be large
        (``len(source_layers) * d_model**2 * 4`` bytes each), so raise
        ``checkpoint_every_n_prompts`` for large models.

        Returns:
            The fitted :class:`JacobianLens`.
        """
        logger.info(self.config)
        self._run_prompt_loop()
        logger.info("fit: done, %d prompts", self.completed_prompt_count)

        jacobian_mean_L_dict_FN = {
            layer: self.jacobian_sum_L_dict_FN[layer] / self.completed_prompt_count
            for layer in self.source_layers
        }

        # The lens reads these off the config; set them here explicitly rather
        # than relying on write_checkpoint's side effect.
        self.config.num_prompts_trained_on = self.completed_prompt_count
        self.config.next_prompt_idx = self.next_prompt_idx

        return JacobianLens(
            jacobians=jacobian_mean_L_dict_FN,
            config=self.config,
        )

    ### SAVING AND LOADING

    @classmethod
    def _resolve_model(
        cls,
        config: LensConfig,
        model: Optional[LensModel],  # noqa: UP045
    ) -> LensModel:
        """The model to resume with: loaded from ``config.hf_model_name`` when
        ``model`` is ``None``, otherwise the passed model after verifying it
        matches the checkpoint
        (:func:`~workspace_lens.utils.check_model_matches_config`). Shared by
        every trainer's resume path."""
        if model is None:
            return get_hf_model(config.hf_model_name)
        check_model_matches_config(model, config)
        return model

    def _atomic_write(self, path: str, write: Callable[[str], object]) -> None:
        """``write(temporary_path)`` then ``os.replace`` so a crash never leaves a
        half-written file behind. ``write``'s return value is ignored
        (``Path.write_text`` returns a count)."""
        temporary_path = f"{path}.tmp.{os.getpid()}"
        write(temporary_path)
        os.replace(temporary_path, path)

    def _atomic_save(self, obj: object, path: str) -> None:
        self._atomic_write(path, lambda temporary_path: t.save(obj, temporary_path))

    def _write_position_records_json(self) -> None:
        """``<checkpoint_path>/position_records.json``: the position records as a
        small sidecar beside the (multi-GB) checkpoint, so every shard's kept
        positions can be checked without loading its tensors
        (``merge_expert_checkpoints`` does not read the records)."""
        self._atomic_write(
            os.path.join(self.config.checkpoint_path, "position_records.json"),
            lambda temporary_path: Path(temporary_path).write_text(
                json.dumps(self.position_records)
            ),
        )

    def write_checkpoint(self, *, final: bool = False) -> None:
        self.config.num_prompts_trained_on = self.completed_prompt_count
        self.config.next_prompt_idx = self.next_prompt_idx

        os.makedirs(self.config.checkpoint_path, exist_ok=True)
        filename = "final_checkpoint.pt" if final else f"{self.next_prompt_idx}_checkpoint.pt"

        self._atomic_save(
            {
                "jacobian_sum": self.jacobian_sum_L_dict_FN,
                "config": self.config.to_dict(),
                "position_records": self.position_records,
            },
            os.path.join(self.config.checkpoint_path, filename),
        )
        self._write_position_records_json()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        prompts: list[str],
        *,
        model: Optional[LensModel] = None,  # noqa: UP045
        device: str = DEFAULT_DEVICE,
    ) -> Self:
        """Resume training from a checkpoint file written by :meth:`write_checkpoint`.

        ``prompts`` must be the same list (same order) the checkpoint was fitted
        on: the checkpoint stores an index into it. ``model`` skips reloading
        when the caller already has the model in memory; it must be the same
        model as ``config.hf_model_name`` (verified via ``d_model`` and, where
        the model exposes one, its HF name).
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"checkpoint {checkpoint_path} not found")

        state: dict[str, Any] = t.load(checkpoint_path, map_location="cpu", weights_only=True)

        config = LensConfig.from_dict(state["config"])
        jacobian_sum_L_dict_FN = state["jacobian_sum"]

        model = cls._resolve_model(config, model)

        logger.info(
            "  resuming from checkpoint: %d/%d prompts processed",
            config.next_prompt_idx,
            len(prompts),
        )

        trainer = cls(
            config=config,
            model=model,
            prompts=prompts,
            device=device,
        )

        trainer.jacobian_sum_L_dict_FN = jacobian_sum_L_dict_FN
        trainer.completed_prompt_count = config.num_prompts_trained_on
        trainer.next_prompt_idx = config.next_prompt_idx
        trainer.position_records = list(state["position_records"])

        return trainer

    ### LOGGING

    def _diagnostic_logging(
        self,
        prompt_idx: int,
        prompt_start_time: float,
        prompt_jacobians_L_dict_FN: dict[int, t.Tensor],
    ) -> None:
        # Per-prompt diagnostics, max over source layers: the prompt's own
        # Jacobian norm flags heavy-tailed outliers, and the relative shift
        # in the running mean tracks convergence (falls ~1/n once settled).
        # The (n+1) divisor below comes from the running-mean identity
        # new_mean - old_mean = (J_prompt - old_mean) / (n+1).

        max_normalised_jacobian_norm = (
            max(prompt_jacobians_L_dict_FN[layer].norm().item() for layer in self.source_layers)
            / self.sqrt_d_model
        )

        if self.completed_prompt_count > 0:
            max_running_mean_relative_change = max(
                (
                    (
                        prompt_jacobians_L_dict_FN[layer]
                        - self.jacobian_sum_L_dict_FN[layer] / self.completed_prompt_count
                    ).norm()
                    / (
                        (self.completed_prompt_count + 1)
                        * (
                            self.jacobian_sum_L_dict_FN[layer] / self.completed_prompt_count
                        ).norm()
                    )
                ).item()
                for layer in self.source_layers
            )
        else:
            max_running_mean_relative_change = float("nan")

        logger.info(
            "%s  max||J||/sqrt(d)=%.3f  max_d_mean=%.2e",
            self._prompt_progress_line(prompt_idx, prompt_start_time),
            max_normalised_jacobian_norm,
            max_running_mean_relative_change,
        )
