"""Read out a prompt with the released J++ Lens for Qwen3.6-27B.

Downloads the lens from the Hugging Face Hub (or reads a local file), loads the model on a GPU,
and prints the top tokens the lens reads out at each readout layer for the prompt's final
position, next to the model's own top tokens. Run from the repository root::

    python scripts/quickstart.py
    python scripts/quickstart.py --prompt "Fact: The currency of the country shaped like a boot is"
    python scripts/quickstart.py --lens path/to/lens.pt --no-readout-filter --out readouts.json

The default prompt asks, without naming it, how many chambers the heart has: the answer is 4,
and the bridging concept ("heart") should appear in the middle layers.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass

import torch as t

from jlens.protocol import LensModel
from lens_evals.readout_evals.readout_eval_items import (
    load_readout_eval_items,
    non_semantic_token_ids,
)
from workspace_lens import get_hf_model
from workspace_lens.config import QWEN3_6_27B_RECIPE_READOUT_LAYERS
from workspace_lens.lenses.base_lens import BaseLens
from workspace_lens.utils import check_model_matches_config, load_lens_file, vocab_size_of

DEFAULT_LENS_REPO = "koayon/jpp-lenses"
DEFAULT_LENS_FILENAME = "qwen3.6-27b/lens.pt"
DEFAULT_MODEL = "Qwen/Qwen3.6-27B"
DEFAULT_PROMPT = (
    "Fact: In humans, the organ that pumps blood through the body has this many chambers: "
)


@dataclass
class PromptReadouts:
    """Top token ids at the prompt's final position: the lens's at each layer, and the
    model's own."""

    lens_top_ids_L_dict: dict[int, list[int]]
    model_top_ids: list[int]


def top_token_ids(logits_V: t.Tensor, top_k: int) -> list[int]:
    return logits_V.topk(top_k).indices.tolist()


def read_out_prompt(
    model: LensModel,
    lens: BaseLens,
    prompt: str,
    *,
    layers: Sequence[int],
    top_k: int,
    excluded_token_ids: Sequence[int],
) -> PromptReadouts:
    """Top-``top_k`` readouts at the prompt's final position, with ``excluded_token_ids`` never
    ranked by the lens (Readout Filtering)."""
    lens_logits_L_dict_1V, model_logits_1V, _ = lens.apply(
        model, prompt, layers=layers, token_positions_for_residuals=[-1]
    )
    excluded_mask_Bool_V = t.zeros(lens_logits_L_dict_1V[layers[0]].shape[-1], dtype=t.bool)
    excluded_mask_Bool_V[list(excluded_token_ids)] = True
    return PromptReadouts(
        lens_top_ids_L_dict={
            layer: top_token_ids(
                logits_1V[0].masked_fill(excluded_mask_Bool_V, float("-inf")), top_k
            )
            for layer, logits_1V in lens_logits_L_dict_1V.items()
        },
        model_top_ids=top_token_ids(model_logits_1V[0], top_k),
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--lens",
        default=DEFAULT_LENS_REPO,
        help="Hub repo id (a lens saved by this library) or a local lens file (either this "
        "library's format or that of the released J-Lens and R-Lens files). Default %(default)s.",
    )
    parser.add_argument(
        "--filename",
        default=DEFAULT_LENS_FILENAME,
        help="Lens file inside the Hub repo. Default %(default)s.",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="Model the lens was fitted on. Default %(default)s."
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--layers",
        type=lambda text: [int(layer) for layer in text.split(",")],
        default=list(QWEN3_6_27B_RECIPE_READOUT_LAYERS),
        help="Comma-separated layers to read out. Default %(default)s.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--no-readout-filter",
        action="store_true",
        help="Rank the full vocabulary instead of dropping tokens with no letter or digit.",
    )
    parser.add_argument(
        "--out", default=None, help="Optional JSON path for the token ids and strings."
    )
    args = parser.parse_args(argv)

    lens = (
        load_lens_file(args.lens, hf_model_name=args.model)
        if os.path.isfile(args.lens)
        else BaseLens.from_pretrained(args.lens, filename=args.filename)
    )
    model = get_hf_model(args.model, attn_implementation="sdpa")
    check_model_matches_config(model, lens.config)
    excluded_token_ids = (
        []
        if args.no_readout_filter
        else non_semantic_token_ids(
            model.tokenizer, vocab_size=vocab_size_of(model), items=load_readout_eval_items()
        )
    )
    readouts = read_out_prompt(
        model,
        lens,
        args.prompt,
        layers=args.layers,
        top_k=args.top_k,
        excluded_token_ids=excluded_token_ids,
    )

    def decode(token_ids: list[int]) -> list[str]:
        return [model.tokenizer.decode([token_id]) for token_id in token_ids]

    print(f"prompt: {args.prompt!r}")
    for layer, token_ids in readouts.lens_top_ids_L_dict.items():
        print(f"layer {layer:>2}: {decode(token_ids)}")
    print(f"model output: {decode(readouts.model_top_ids)}")
    if args.out is not None:
        token_ids = {"lens": readouts.lens_top_ids_L_dict, "model": readouts.model_top_ids}
        tokens = {
            "lens": {layer: decode(ids) for layer, ids in readouts.lens_top_ids_L_dict.items()},
            "model": decode(readouts.model_top_ids),
        }
        with open(args.out, "w") as file:
            json.dump(
                {
                    "prompt": args.prompt,
                    "token_ids": token_ids,
                    "tokens": tokens,
                    "num_excluded_token_ids": len(excluded_token_ids),
                },
                file,
                indent=1,
                ensure_ascii=False,
            )


if __name__ == "__main__":
    main()
