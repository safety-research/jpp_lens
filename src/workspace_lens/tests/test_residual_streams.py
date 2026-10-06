"""The stream-mean residual on the tiny DeepSeek-V4 fixture: the helpers, the read
sites, and the Jacobian estimator against finite differences.

The estimator test runs the real ``LensTrainer`` forward and backward passes with
``lrp_mode="none"`` in float64 on a model where every routed expert is active at every
token and no MLP clamp binds, so the forward is smooth and central differences are well
posed. Both sides measure the same object: the sum over valid positions of the Jacobian of
the target stream-mean with respect to a uniform perturbation of the source streams (the
estimator's per-position rows summed over the positions, against a perturbation added to
every stream at those positions)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch as t

from workspace_lens.fitting.jacobian_fitting import LensTrainer
from workspace_lens.residual_streams import (
    collapse_streams_like_the_model,
    has_residual_streams,
    residual_mean_over_streams,
    sum_gradient_over_streams,
)
from workspace_lens.tests.fixtures import make_tiny_lens_config
from workspace_lens.tests.tiny_deepseek import (
    TINY_DEEPSEEK_HC_MULT,
    TINY_DEEPSEEK_HIDDEN_SIZE,
    capture_block_output,
    tiny_deepseek_lens_model,
)
from workspace_lens.utils import (
    final_position_model_logits,
    record_activations,
    record_block_outputs,
)

PROMPT = "the quick brown fox jumps over the lazy dog"
SOURCE_LAYER = 0
TARGET_LAYER = 2
SKIP_FIRST = 2


def test_helpers_pass_single_stream_tensors_through_and_reduce_streams():
    residual_BSN = t.randn(2, 5, 8)
    assert not has_residual_streams(residual_BSN)
    assert residual_mean_over_streams(residual_BSN) is residual_BSN
    assert sum_gradient_over_streams(residual_BSN) is residual_BSN

    streams_BSRN = t.randn(2, 5, 4, 8)
    assert has_residual_streams(streams_BSRN)
    assert t.allclose(residual_mean_over_streams(streams_BSRN), streams_BSRN.mean(dim=2))
    assert t.allclose(sum_gradient_over_streams(streams_BSRN), streams_BSRN.sum(dim=2))
    summed_BSN = sum_gradient_over_streams(streams_BSRN.to(t.bfloat16))
    assert summed_BSN.dtype == t.float32
    assert t.equal(summed_BSN, streams_BSRN.to(t.bfloat16).float().sum(dim=2))


def test_record_activations_returns_the_stream_mean_of_the_raw_block_output():
    model = tiny_deepseek_lens_model()
    activations_L_dict_SN: dict[int, t.Tensor] = {}

    def run() -> None:
        activations_L_dict_SN.update(record_activations(model, PROMPT, 16, [1])[1])

    raw_1SRN = capture_block_output(model.layers[1], run)
    assert raw_1SRN.shape[2] == TINY_DEEPSEEK_HC_MULT
    assert activations_L_dict_SN[1].shape == (raw_1SRN.shape[1], TINY_DEEPSEEK_HIDDEN_SIZE)
    assert t.allclose(activations_L_dict_SN[1], raw_1SRN[0].mean(dim=1))


def test_model_logits_go_through_the_models_own_collapse_head():
    model = tiny_deepseek_lens_model()
    input_ids_Int_1S = model.encode(PROMPT, max_length=16)
    with t.no_grad():
        expected_V = model._hf_model(input_ids=input_ids_Int_1S, use_cache=False).logits[0, -1].float()  # type: ignore[attr-defined]
    logits_V = final_position_model_logits(model, PROMPT, 16)
    assert t.allclose(logits_V, expected_V, atol=1e-5, rtol=1e-4)


def test_collapse_helper_uses_the_head_and_refuses_a_model_without_one():
    model = tiny_deepseek_lens_model()
    streams_1SRN = t.randn(1, 3, TINY_DEEPSEEK_HC_MULT, TINY_DEEPSEEK_HIDDEN_SIZE)
    with t.no_grad():
        expected_1SN = model._text_module.hc_head(streams_1SRN)  # type: ignore[attr-defined]
        assert t.equal(collapse_streams_like_the_model(model, streams_1SRN), expected_1SN)
    single_1SN = t.randn(1, 3, TINY_DEEPSEEK_HIDDEN_SIZE)
    assert collapse_streams_like_the_model(object(), single_1SN) is single_1SN
    with pytest.raises(ValueError, match="stream-collapse head"):
        collapse_streams_like_the_model(object(), streams_1SRN)


def test_estimator_matches_central_finite_differences_of_the_stream_mean(tmp_path: Path):
    t.set_default_dtype(t.float64)
    try:
        model = tiny_deepseek_lens_model(seed=5, num_routed_experts=4, num_experts_per_tok=4)
        model._hf_model.double()  # type: ignore[attr-defined]
        for block in model.layers:  # no clamp binds: the forward is smooth everywhere
            block.mlp.shared_experts.limit = 1e6
        d_model = model.d_model
        config = make_tiny_lens_config(
            tmp_path, "streams",
            source_layers=[SOURCE_LAYER], relative_end_transport_layer=-1, jacobian_rows_per_pass=8,
            max_seq_len=16, skip_first_n_positions=SKIP_FIRST, lrp_mode="none",
        )
        trainer = LensTrainer(config, model, prompts=[PROMPT])
        assert trainer.target_layer == TARGET_LAYER

        # The estimator: rows of the Jacobian at every valid position, summed over the positions.
        forward_state = trainer._run_fit_forward(PROMPT)
        valid_positions_Int_P = forward_state.source_positions_Int_P
        jacobian_estimated_FN = t.zeros(d_model, d_model)
        for dim_start, rows_this_pass, grads_L_list_BSN in trainer._backward_passes(forward_state):
            grad_BSN = grads_L_list_BSN[0]
            jacobian_estimated_FN[dim_start : dim_start + rows_this_pass] = (
                grad_BSN[:rows_this_pass, valid_positions_Int_P, :].sum(dim=1)
            )

        # Central differences: add delta * e_j to EVERY stream of the source block's output at the
        # valid positions (a uniform perturbation of the source mean) and read the target stream-mean
        # summed over the same positions.
        input_ids_Int_1S = model.encode(PROMPT, max_length=16)
        # The hyper-connection coefficients run in fp32 inside the model (its own `.float()` casts),
        # so the differences carry fp32 rounding of order 1e-7 / delta: at delta 1e-3 the central
        # difference is good to about 1e-4 relative, while a wrong stream semantics (a missing 1/R
        # on the cotangent, a missing sum over the source streams) is off by a factor of R.
        delta = 1e-3

        def target_mean_sum_N(perturbation_N: t.Tensor) -> t.Tensor:
            def add_to_source(module, inputs, output_1SRN):
                shifted_1SRN = output_1SRN.clone()
                shifted_1SRN[0, valid_positions_Int_P, :, :] += perturbation_N
                return shifted_1SRN

            handle = model.layers[SOURCE_LAYER].register_forward_hook(add_to_source)
            try:
                _, activations_L_dict_SN = record_activations(model, PROMPT, 16, [TARGET_LAYER])
            finally:
                handle.remove()
            return activations_L_dict_SN[TARGET_LAYER][valid_positions_Int_P].sum(dim=0)

        jacobian_fd_FN = t.zeros(d_model, d_model)
        for j in range(d_model):
            unit_N = t.zeros(d_model)
            unit_N[j] = delta
            jacobian_fd_FN[:, j] = (target_mean_sum_N(unit_N) - target_mean_sum_N(-unit_N)) / (2 * delta)

        scale = jacobian_fd_FN.abs().max()
        assert scale > 0
        assert (jacobian_estimated_FN - jacobian_fd_FN).abs().max() / scale < 1e-3
        assert (jacobian_estimated_FN - jacobian_fd_FN).norm() / jacobian_fd_FN.norm() < 1e-3
        assert input_ids_Int_1S.shape[1] > SKIP_FIRST + 1
    finally:
        t.set_default_dtype(t.float32)


def test_cli_fit_router_fit_shard_and_merge_run_on_the_multi_stream_model(tmp_path: Path, monkeypatch) -> None:
    """The CLI's router fit, a RelP shard under ``r+mhc`` (routing on the stream mean, the mHC
    detach patched in) and the expert merge run end to end on the tiny
    DeepSeek fixture, with the HF loader and the WikiText loader swapped for the fixtures."""
    import json
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    import jpp_cli

    from workspace_lens.fitting.utils import expert_checkpoint_filename
    from workspace_lens.routing.router import ActivationRouterCollection

    model = tiny_deepseek_lens_model(seed=3)
    prompts = ["the quick brown fox jumps over the lazy dog again", "pack my box with five dozen liquor jugs"] * 2
    monkeypatch.setattr(jpp_cli, "get_hf_model", lambda name, **_kwargs: model)
    monkeypatch.setattr(jpp_cli, "load_fit_prompts", lambda num_prompts: prompts[:num_prompts])

    router_path = tmp_path / "router.pt"
    argv = [
        "fit-router", "--hf-model-name", "tiny-deepseek", "--layers", "0,1", "--num-prompts", "4",
        "--num-clusters", "2", "--projection-dim", "4", "--max-seq-len", "64", "--out", str(router_path),
    ]
    assert jpp_cli.main(argv) == 0
    router = ActivationRouterCollection.load(str(router_path))
    assert router.num_clusters == 2

    argv = [
        "fit-shard", "--hf-model-name", "tiny-deepseek", "--router-path", str(router_path), "--layers", "0,1",
        "--num-prompts", "4", "--shard-idx", "0", "--num-shards", "2", "--lrp-mode", "r+mhc",
        "--jacobian-rows-per-pass", "8", "--max-seq-len", "64",
        "--checkpoint-every-n-prompts", "1", "--artifacts-base-dir", str(tmp_path), "--checkpoint-name", "ds_experts",
    ]
    assert jpp_cli.main(argv) == 0
    shard_dirs = list(tmp_path.glob("*/ds_experts/shard0of2"))
    assert len(shard_dirs) == 1
    checkpoint = t.load(shard_dirs[0] / expert_checkpoint_filename(2), weights_only=True)
    stamp = json.loads((shard_dirs[0] / "fit_shard_stamp.json").read_text())
    assert stamp["args"]["lrp_mode"] == "r+mhc"
    # The shard's per-layer sums are finite and the mHC detach was part of the recorded rules.
    def tensors_in(value):
        if t.is_tensor(value):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from tensors_in(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from tensors_in(item)

    layer_sum_tensors = list(tensors_in(checkpoint["layer_sums"]))
    assert layer_sum_tensors and all(t.isfinite(value).all() for value in layer_sum_tensors)
    assert checkpoint["lrp_rules"]["mhc_detach"] is True and checkpoint["lrp_rules"]["ln_rule"] is True

    experts_path = tmp_path / "experts.pt"
    argv = [
        "merge-experts", "--checkpoint-dirs", str(shard_dirs[0]), "--num-clusters", "2", "--min-kept-positions", "1",
        "--checkpoint-name", "ds_experts", "--out", str(experts_path),
    ]
    assert jpp_cli.main(argv) == 0
    assert experts_path.exists()


def test_apply_model_logits_come_from_the_models_own_head(tmp_path: Path) -> None:
    """``BaseLens.apply``'s model logits on a multi-stream model are the model's (its head collapses the final
    block's streams), not the lens's stream-mean readout."""
    from workspace_lens.config import LensConfig
    from workspace_lens.lenses.logit_lens import LogitLens

    model = tiny_deepseek_lens_model()
    lens = LogitLens(
        [0],
        config=LensConfig(hf_model_name="tiny-deepseek", checkpoint_name="logit", lens_type="logit", d_model=model.d_model,
                          num_prompts_trained_on=1, artifacts_base_dir=str(tmp_path)),
    )
    lens_logits_L_dict_SV, model_logits_SV, input_ids_Int_1S = lens.apply(model, PROMPT, layers=[0], max_seq_len=16)
    with t.no_grad():
        expected_SV = model._hf_model(input_ids=input_ids_Int_1S, use_cache=False).logits[0].float()  # type: ignore[attr-defined]
    assert t.allclose(model_logits_SV, expected_SV.cpu(), atol=1e-5, rtol=1e-4)
    # The stream-mean readout differs from the head's by more than the tolerance above (about 2e-4 on the tiny
    # fixture, whose head starts near uniform), so model logits read from the stream mean would fail the
    # assertion above.
    _, block_outputs = record_block_outputs(model, PROMPT, 16, [model.n_layers - 1])
    mean_readout_SV = model.unembed(residual_mean_over_streams(block_outputs[model.n_layers - 1])[0].float()).float()
    assert not t.allclose(mean_readout_SV.cpu(), expected_SV.cpu(), atol=1e-5, rtol=1e-4)


def test_expert_fit_routes_each_position_on_the_stream_mean(tmp_path: Path) -> None:
    """The expert trainer's fit-time routing sees the lens's residual, the mean over the streams, at the valid
    positions: its per-cluster position counts equal the histogram of the router's assignments over the stock
    forward's stream means (routing on a single stream would pass the rest of the suite)."""
    from workspace_lens.fitting.relp_fitting import ExpertJacobianRelPTrainer
    from workspace_lens.routing.router import ActivationRouterCollection
    model = tiny_deepseek_lens_model(seed=3)
    prompts = ["the quick brown fox jumps over the lazy dog again", "pack my box with five dozen liquor jugs"]
    layers = [0, 1]
    # The tiny fixture's four streams sit within about 2% of their mean, so routing on any one stream would give
    # the same histogram as routing on the mean and the test would be vacuous. Pull stream 0 away from the others
    # at both source blocks (a forward hook on the block, ahead of every recorder), then check below that routing
    # on stream 0 alone does differ from routing on the mean.
    def pull_stream_zero(module, inputs, output):
        block_output_BSRN = output if t.is_tensor(output) else output[0]
        pulled_BSRN = block_output_BSRN.clone()
        pulled_BSRN[:, :, 0, :] = pulled_BSRN[:, :, 0, :] * 4.0 + 1.0
        return pulled_BSRN if t.is_tensor(output) else (pulled_BSRN, *output[1:])

    handles = [model.layers[layer].register_forward_hook(pull_stream_zero) for layer in layers]
    router = ActivationRouterCollection(num_clusters=2, projection_dim=4, seed=0)
    router.fit({
        layer: t.cat([record_activations(model, prompt, 64, layers)[1][layer].float() for prompt in prompts])
        for layer in layers
    })
    config = make_tiny_lens_config(
        tmp_path, "route-on-mean", hf_model_name="tiny-deepseek", source_layers=layers, lrp_mode="r+mhc",
        jacobian_rows_per_pass=8,
    )
    trainer = ExpertJacobianRelPTrainer(
        config, model, prompts=[], router_collections_K_dict={2: router}
    )
    trainer.fit_step(prompts[0])
    valid_positions_Int_P = trainer._run_fit_forward(prompts[0]).source_positions_Int_P.cpu()
    _, activations_L_dict_SN = record_activations(model, prompts[0], 64, layers)
    _, block_outputs_L_dict_1SRN = record_block_outputs(model, prompts[0], 64, layers)
    for handle in handles:
        handle.remove()
    one_stream_differs = False
    for layer in layers:
        expected_counts_E = t.bincount(
            router.assign(activations_L_dict_SN[layer][valid_positions_Int_P].float(), layer), minlength=2
        )
        stream_zero_counts_E = t.bincount(
            router.assign(block_outputs_L_dict_1SRN[layer][0, valid_positions_Int_P, 0, :].float(), layer), minlength=2
        )
        one_stream_differs |= stream_zero_counts_E.tolist() != expected_counts_E.tolist()
        counts_E = trainer.fit_sums_K_dict[2].layer_sums_L_dict[layer].position_count_E.cpu()
        assert counts_E.tolist() == expected_counts_E.tolist(), (layer, counts_E, expected_counts_E)
    assert one_stream_differs, "fixture guard: routing on stream 0 must differ from routing on the mean somewhere"
    assert sum(trainer.fit_sums_K_dict[2].layer_sums_L_dict[0].position_count_E.tolist()) == len(valid_positions_Int_P)
