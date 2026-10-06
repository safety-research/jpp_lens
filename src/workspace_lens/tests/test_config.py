"""``LensConfig.from_dict``: round trips, and the config saved in the released lens."""

import re
from datetime import UTC, datetime
from pathlib import Path

from workspace_lens.config import TIMESTAMP_FORMAT, LensConfig

# Keys of the released lens's config that name no LensConfig field; from_dict ignores them.
IGNORED_CONFIG_KEYS = (
    "num_s2_fitting_prompts",
    "mask_non_semantic_fit_positions",
)

# The released lens's source layers: all 63 layers before the final block.
SAVED_SOURCE_LAYERS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]

# The config saved in the released lens.
RELEASED_LENS_CONFIG: dict = {
    "hf_model_name": "Qwen/Qwen3.6-27B",
    "checkpoint_name": "jpp_trained_lens",
    "lens_type": "jacobian",
    "checkpoint_every_n_steps": 100,
    "creation_time": "2026-09-07_23-46-18",
    "artifacts_base_dir": "artifacts",
    "checkpoint_path": "artifacts/2026-09-07/jpp_trained_lens",
    "source_layers": SAVED_SOURCE_LAYERS,
    "relative_end_transport_layer": -1,
    "d_model": 5120,
    "num_prompts_trained_on": 64,
    "next_prompt_idx": 0,
    "batch_dim": 8,
    "max_seq_len": 128,
    "num_s2_fitting_prompts": 10,
    "skip_first_n_positions": 16,
    "mask_non_semantic_fit_positions": False,
    "num_clusters": 0,
    "cluster_projection_dim": 0,
    "lrp_mode": "rlens",
}


def _config(tmp_path: Path) -> LensConfig:
    return LensConfig(
        hf_model_name="tiny",
        checkpoint_name="probe",
        artifacts_base_dir=str(tmp_path),
        creation_time="2026-01-02_03-04-05",
        source_layers=[0, 2],
        d_model=8,
        jacobian_rows_per_pass=4,
        checkpoint_every_n_prompts=None,
        lrp_mode="rlens",
    )


def test_from_dict_round_trips(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert LensConfig.from_dict(config.to_dict()) == config


def test_from_dict_reads_the_saved_key_spellings_and_ignores_unknown_keys(tmp_path: Path) -> None:
    config_dict = _config(tmp_path).to_dict()
    # The keys the released lens's config uses for two fields ...
    config_dict["batch_dim"] = config_dict.pop("jacobian_rows_per_pass")
    config_dict["checkpoint_every_n_steps"] = config_dict.pop("checkpoint_every_n_prompts")
    # ... a stored path (the path is always re-derived from creation_time), and a
    # key that names no field.
    config_dict["checkpoint_path"] = "somewhere/else"
    config_dict["unknown_key"] = 1

    restored = LensConfig.from_dict(config_dict)

    assert restored.jacobian_rows_per_pass == 4
    assert restored.checkpoint_every_n_prompts is None
    assert restored.checkpoint_path == str(tmp_path / "2026-01-02" / "probe")
    assert restored == _config(tmp_path)


def test_from_dict_reads_the_released_lens_config() -> None:
    """The config saved in the released lens loads and round-trips: its keys for
    ``jacobian_rows_per_pass`` and ``checkpoint_every_n_prompts`` are read, its keys
    that name no field are ignored, and the rebuilt config's ``to_dict`` does not
    carry them."""
    config = LensConfig.from_dict(RELEASED_LENS_CONFIG)

    assert config.jacobian_rows_per_pass == RELEASED_LENS_CONFIG["batch_dim"]
    assert config.checkpoint_every_n_prompts == RELEASED_LENS_CONFIG["checkpoint_every_n_steps"]
    for field_name in (
        "hf_model_name",
        "checkpoint_name",
        "lens_type",
        "source_layers",
        "relative_end_transport_layer",
        "d_model",
        "num_prompts_trained_on",
        "max_seq_len",
        "skip_first_n_positions",
        "num_clusters",
        "cluster_projection_dim",
        "lrp_mode",
    ):
        assert getattr(config, field_name) == RELEASED_LENS_CONFIG[field_name], field_name
    assert config.checkpoint_path == RELEASED_LENS_CONFIG["checkpoint_path"]
    assert not set(IGNORED_CONFIG_KEYS) & set(config.to_dict())
    assert LensConfig.from_dict(config.to_dict()) == config


def test_to_dict_carries_no_ignored_keys(tmp_path: Path) -> None:
    assert not set(IGNORED_CONFIG_KEYS) & set(_config(tmp_path).to_dict())


def test_creation_time_is_a_utc_timestamp_and_dates_the_checkpoint_dir(tmp_path) -> None:
    """A new config is stamped now, in UTC; the date part of ``creation_time`` names its
    checkpoint directory, so a config restored with the same ``creation_time`` resolves to the
    same directory."""
    config = LensConfig(hf_model_name="tiny", checkpoint_name="probe", artifacts_base_dir=str(tmp_path))
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", config.creation_time)
    stamped = datetime.strptime(config.creation_time, TIMESTAMP_FORMAT).replace(tzinfo=UTC)
    assert abs((datetime.now(UTC) - stamped).total_seconds()) < 60
    restored = LensConfig(
        hf_model_name="tiny",
        checkpoint_name="probe",
        artifacts_base_dir=str(tmp_path),
        creation_time="2026-01-02_03-04-05",
    )
    assert restored.checkpoint_path == str(tmp_path / "2026-01-02" / "probe")
    assert LensConfig.from_dict(restored.to_dict()).checkpoint_path == restored.checkpoint_path
