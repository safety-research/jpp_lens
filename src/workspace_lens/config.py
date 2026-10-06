import os
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import Any, Literal, Optional

from workspace_lens.lrp import LRP_MODES

LensTypes = Literal["jacobian", "logit"]

# The creation_time format (UTC); its date part names the checkpoint directory.
TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"

# Per-model layer sets shared by fitting and the eval runners. Names with RECIPE
# (here, in scripts/jpp_cli.py and lens_evals/readout_evals/readout_eval_items.py) refer to the
# J++ Lens configuration for Qwen3.6-27B.
# The readout layers: every eighth block of Qwen3.6-27B's 64, the layers the
# J++ Lens is fitted and scored on. The probe-swap eval edits at these too.
QWEN3_6_27B_RECIPE_READOUT_LAYERS: list[int] = [8, 16, 24, 32, 40, 48, 56]

# Field name -> the key it is saved under in configs such as the released lens's.
LEGACY_CONFIG_KEYS: dict[str, str] = {
    "jacobian_rows_per_pass": "batch_dim",
    "checkpoint_every_n_prompts": "checkpoint_every_n_steps",
}


def _read_renamed_field(config_dict: dict, field_name: str) -> Any:
    """``config_dict[field_name]``, falling back to the field's key in
    :data:`LEGACY_CONFIG_KEYS`."""
    if field_name in config_dict:
        return config_dict[field_name]
    return config_dict[LEGACY_CONFIG_KEYS[field_name]]


@dataclass
class LensConfig:
    hf_model_name: str

    # Checkpoint params
    checkpoint_name: str
    lens_type: LensTypes = "jacobian"
    # None: write only the final checkpoint.
    checkpoint_every_n_prompts: Optional[int] = 10  # noqa: UP045
    # None: stamped now, in UTC.
    creation_time: Optional[str] = None  # noqa: UP045
    artifacts_base_dir: str = "artifacts"
    # Derived: <artifacts_base_dir>/<date of creation_time>/<checkpoint_name>
    checkpoint_path: str = ""

    # Layer indices
    source_layers: Optional[list[int]] = None  # noqa: UP045
    relative_end_transport_layer: int = -1

    # Trained lens params
    d_model: int = 0
    num_prompts_trained_on: int = 0
    next_prompt_idx: int = 0

    # Training hyperparameters
    # Rows of J_l computed per backward pass (the B axis of the fit's cotangent);
    # not a data batch size.
    jacobian_rows_per_pass: int = 8
    max_seq_len: int = 128
    # Positions before this index are excluded from the Jacobian average; early
    # positions act as attention sinks and have atypical residual statistics.
    skip_first_n_positions: int = 16

    # The router of an expert-Jacobian fit (0 for a lens): clusters (experts) per
    # layer and the PCA dimension the routing operates in.
    num_clusters: int = 0
    cluster_projection_dim: int = 0

    # Backward-pass rule set the lens was fitted under: "none" is the standard
    # gradient; every other mode is an LRP (RelP) preset resolved by
    # workspace_lens.lrp.lrp_rule_config_for_mode.
    lrp_mode: str = "none"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "LensConfig":
        """Rebuild a config from :meth:`to_dict` output, field by field.

        ``checkpoint_path`` is deliberately not read back: ``__post_init__``
        re-derives it from ``artifacts_base_dir``, ``creation_time`` and
        ``checkpoint_name``. ``jacobian_rows_per_pass`` and
        ``checkpoint_every_n_prompts`` are also read under their keys in
        :data:`LEGACY_CONFIG_KEYS`. Keys that name no field are ignored.
        """
        return cls(
            hf_model_name=config_dict["hf_model_name"],
            checkpoint_name=config_dict["checkpoint_name"],
            lens_type=config_dict["lens_type"],
            checkpoint_every_n_prompts=_read_renamed_field(
                config_dict, "checkpoint_every_n_prompts"
            ),
            creation_time=config_dict["creation_time"],
            artifacts_base_dir=config_dict["artifacts_base_dir"],
            source_layers=config_dict["source_layers"],
            relative_end_transport_layer=config_dict["relative_end_transport_layer"],
            d_model=config_dict["d_model"],
            num_prompts_trained_on=config_dict["num_prompts_trained_on"],
            next_prompt_idx=config_dict["next_prompt_idx"],
            jacobian_rows_per_pass=_read_renamed_field(config_dict, "jacobian_rows_per_pass"),
            max_seq_len=config_dict["max_seq_len"],
            skip_first_n_positions=config_dict["skip_first_n_positions"],
            num_clusters=config_dict["num_clusters"],
            cluster_projection_dim=config_dict["cluster_projection_dim"],
            lrp_mode=config_dict["lrp_mode"],
        )

    def with_router(self, *, num_clusters: int, projection_dim: int) -> "LensConfig":
        """This config with the router geometry of an expert-Jacobian fit set:
        ``num_clusters`` and ``cluster_projection_dim`` (from a router collection, so
        callers need not pre-set them on a trainer or merged config)."""
        return replace(self, num_clusters=num_clusters, cluster_projection_dim=projection_dim)

    def as_jacobian(self, *, checkpoint_name: str) -> "LensConfig":
        """This config for a lens with one map per layer (a combined or pooled lens built from
        expert Jacobians): the router fields reset to 0 and the given ``checkpoint_name``
        (whose derived ``checkpoint_path`` ``__post_init__`` recomputes)."""
        return replace(
            self,
            checkpoint_name=checkpoint_name,
            num_clusters=0,
            cluster_projection_dim=0,
        )

    def check_compatible(self, other: "LensConfig") -> None:
        """Raise ``ValueError`` if two configs describe lenses that cannot be
        combined (e.g. merged).

        Fields are checked most-important first, so the error reports the
        most fundamental disagreement: ``d_model`` (different residual
        widths), then ``hf_model_name`` (different models), then the layer
        geometry.
        """
        for field_name in (
            "d_model",
            "hf_model_name",
            "relative_end_transport_layer",
            "source_layers",
            "lens_type",
            "num_clusters",
            "cluster_projection_dim",
            "lrp_mode",
        ):
            ours, theirs = getattr(self, field_name), getattr(other, field_name)
            if ours != theirs:
                raise ValueError(f"configs disagree on {field_name}: {ours!r} != {theirs!r}")

    def __post_init__(self):

        if self.lrp_mode not in LRP_MODES:
            raise ValueError(f"lrp_mode must be one of {LRP_MODES}, got {self.lrp_mode!r}")
        if self.skip_first_n_positions < 0:
            raise ValueError(
                f"skip_first_n_positions must be >= 0, got {self.skip_first_n_positions}"
            )

        if self.creation_time is None:
            self.creation_time = datetime.now(UTC).strftime(TIMESTAMP_FORMAT)

        # Deterministic given creation_time, so a config restored from a
        # checkpoint resolves to the same directory it was saved in.
        date = self.creation_time.split("_")[0]
        self.checkpoint_path = os.path.join(
            self.artifacts_base_dir, date, self.checkpoint_name
        )
