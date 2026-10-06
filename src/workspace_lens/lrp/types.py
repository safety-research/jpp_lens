from typing import Literal, get_args

LrpMode = Literal[
    "none",
    "rlens",
    "r+mhc",
    "all-c4+mhc",
    "r+moe",
]
LRP_MODES: tuple[LrpMode, ...] = get_args(LrpMode)
