"""Token-level utilities for the lens evals in ``data/jlens/evaluations``.

``ORDER_OPS_SYNONYMS`` implements the expansion the data README describes for
``lens-eval-order-ops`` ("numbers -> digit and word forms; operations ->
symbol and word forms").
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping, Sequence
from typing import Any

import torch as t

from jlens.protocol import LensModel
from workspace_lens.utils import token_id_mask_Bool_V, vocab_size_of

logger = logging.getLogger(__name__)

ORDER_OPS_SYNONYMS: dict[str, tuple[str, ...]] = {
    # Numbers appearing as order-ops intermediates: digit and word forms.
    "3": ("3", "three"),
    "4": ("4", "four"),
    "5": ("5", "five"),
    "6": ("6", "six"),
    "7": ("7", "seven"),
    "8": ("8", "eight"),
    "9": ("9", "nine"),
    "10": ("10", "ten"),
    "11": ("11", "eleven"),
    "12": ("12", "twelve", "dozen"),
    "13": ("13", "thirteen"),
    "15": ("15", "fifteen"),
    "16": ("16", "sixteen"),
    "20": ("20", "twenty"),
    "24": ("24", "twenty-four"),
    # Operations: word and symbol forms.
    "addition": ("addition", "add", "plus", "sum", "and", "+"),
    "subtraction": ("subtraction", "subtract", "minus", "difference", "take", "diff", "-"),
    "multiplication": (
        "multiplication",
        "multiply",
        "times",
        "product",
        "mult",
        "prod",
        "by",
        "*",
        "×",
    ),
    "division": ("division", "divide", "quotient", "/", "÷"),
    "mod": ("mod", "modulo", "remainder", "%"),
    "squared": ("squared", "square", "power", "exponent", "^", "**"),
}


def surface_forms(word: str) -> list[str]:
    """The tokenizer surfaces scored for one synonym: with/without a leading
    space, in the original, capitalised, and lowercase casings (so e.g.
    "Brazil" also scores " brazil" — a lens is never penalised for reading a
    concept out in the "wrong" case)."""
    cased = {word, word.capitalize(), word.lower()}
    return [prefix + form for form in sorted(cased) for prefix in ("", " ")]


def candidate_token_ids(
    tokenizer: Any,
    intermediate: str,
    *,
    expand_order_ops_synonyms: bool = False,
) -> list[int]:
    """Vocab ids scored for an intermediate: every single-token surface form.

    An intermediate's lens rank is the min over these candidates, so surface
    variation (leading space, capitalisation) and — for order-ops — synonym
    choice never penalise a lens. Multi-token surfaces are dropped; the
    returned list is empty when *no* surface is a single token (e.g. "24"
    under a digit-splitting tokenizer), in which case the intermediate cannot
    be scored and callers should skip and log it.
    """
    if expand_order_ops_synonyms and intermediate in ORDER_OPS_SYNONYMS:
        synonyms = ORDER_OPS_SYNONYMS[intermediate]
    else:
        synonyms = (intermediate,)

    return single_token_ids(
        tokenizer, [surface for synonym in synonyms for surface in surface_forms(synonym)]
    )


def single_token_ids(tokenizer: Any, surfaces: Sequence[str]) -> list[int]:
    """Single-token vocab ids among literal surface strings; multi-token
    surfaces are skipped, so the list is empty when nothing is a single token.
    The one place this loop lives: every eval scores a concept as the min rank
    over these ids."""
    token_ids: set[int] = set()
    for surface in surfaces:
        surface_token_ids = tokenizer(surface, add_special_tokens=False).input_ids
        if len(surface_token_ids) == 1:
            token_ids.add(int(surface_token_ids[0]))
    return sorted(token_ids)


def non_semantic_token_ids_sparing(
    tokenizer: Any, vocab_size: int, spared_ids: set[int], *, context: str
) -> list[int]:
    """The Readout Filtering exclusion list: :func:`all_non_semantic_token_ids` minus
    ``spared_ids`` (the ids an eval scores, so no scored concept becomes
    unrankable), logged under ``context``."""
    non_semantic_ids = all_non_semantic_token_ids(tokenizer, vocab_size)
    excluded_ids = sorted(non_semantic_ids - spared_ids)
    logger.info(
        "%s: %d token ids excluded (%d spared as scored surfaces)",
        context, len(excluded_ids), len(non_semantic_ids & spared_ids),
    )
    return excluded_ids


def is_non_semantic_token(decoded_token: str) -> bool:
    """Whitespace-only or pure punctuation/symbol tokens (no letters or
    digits); ids that decode to the empty string (embedding-matrix padding)
    also count. The single definition of "non-semantic" in this library."""
    return not any(char.isalnum() for char in decoded_token)


def decode_each_token_id(tokenizer: Any, vocab_size: int) -> list[str]:
    """``[tokenizer.decode([i]) for i in range(vocab_size)]``, via the fast
    ``batch_decode`` when the tokenizer has one (the HF tokenizers do; the
    test tokenizers only implement ``decode``)."""
    if hasattr(tokenizer, "batch_decode"):
        return list(tokenizer.batch_decode([[token_id] for token_id in range(vocab_size)]))
    return [tokenizer.decode([token_id]) for token_id in range(vocab_size)]


def all_non_semantic_token_ids(tokenizer: Any, vocab_size: int) -> set[int]:
    """Every id in ``range(vocab_size)`` whose decoded string has no letter or
    digit (:func:`is_non_semantic_token`: whitespace, punctuation, and LM-head
    padding rows that decode to ``""``). The one vocab scan behind every
    non-semantic exclusion mask; callers subtract the ids they must spare.
    ``vocab_size`` is the LM-head width, not ``len(tokenizer)``."""
    return {
        token_id
        for token_id, decoded_token in enumerate(decode_each_token_id(tokenizer, vocab_size))
        if is_non_semantic_token(decoded_token)
    }


def added_token_ids(tokenizer: Any) -> set[int]:
    """Every id the tokenizer registers as an *added* token, special or not.
    Qwen's chat-control tokens split into two groups: ``<|im_end|>`` and the
    ``<|endoftext|>`` family are special (in ``all_special_ids``), while
    ``<think>``, ``</think>``, ``<tool_call>`` and the FIM markers are added but
    not special, so ``all_special_ids`` alone misses them. Tokenizers without
    an added vocabulary (the test tokenizers) contribute nothing."""
    get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
    if get_added_vocab is None:
        return set()
    return {int(token_id) for token_id in get_added_vocab().values()}


def formatting_token_ids_of(tokenizer: Any, vocab_size: int) -> list[int]:
    """Ids of the model's *formatting* tokens: the non-semantic ids plus every
    special or added control token (``<|im_end|>``, ``<think>``, ``<tool_call>``,
    ...). Masking these out of the model's own next-token distribution ranks
    it over word tokens only -- the "word rank" the causal evals record beside
    the full-vocab rank, because the model often emits ``"\n\n"`` or a
    ``<think>`` block before its answer, which would otherwise put a control
    token in the rank-1 slot."""
    special_ids = {int(i) for i in (getattr(tokenizer, "all_special_ids", None) or [])}
    return sorted(all_non_semantic_token_ids(tokenizer, vocab_size) | special_ids | added_token_ids(tokenizer))


def is_non_semantic_token_id(token_id: int, tokenizer: Any) -> bool:
    """A special id, or an id whose decode has no letter or digit."""
    special_token_ids = getattr(tokenizer, "all_special_ids", None) or []
    return token_id in special_token_ids or is_non_semantic_token(tokenizer.decode([token_id]))


def is_semantic_token_id(token_id: int, tokenizer: Any) -> bool:
    """``not`` :func:`is_non_semantic_token_id`: a letter or digit in the
    decode, and not a special id."""
    return not is_non_semantic_token_id(token_id, tokenizer)


def check_candidates_not_all_excluded(
    eval_slug: str,
    item_name: str,
    candidates_by_label: Mapping[str, Sequence[int]],
    excluded_ids: Collection[int] | None,
    *,
    lens_name: str | None = None,
) -> None:
    """Raise ``ValueError`` if ``excluded_ids`` would make one of the item's
    labels (intermediates / tracked tokens) unrankable: every one of its
    candidate ids excluded. An excluded id is forced to ``-inf`` before
    ranking, so such a label's rank would be meaningless."""
    if not excluded_ids:
        return
    for label, candidate_ids in candidates_by_label.items():
        if candidate_ids and all(candidate_id in excluded_ids for candidate_id in candidate_ids):
            for_lens = f" for lens {lens_name!r}" if lens_name is not None else ""
            raise ValueError(
                f"{eval_slug}/{item_name} {label!r}: every candidate token id "
                f"({list(candidate_ids)}) is in excluded_token_ids{for_lens}; its rank "
                "would be meaningless. Spare the scored candidate ids when building the "
                "exclusion list (readout_eval_items.non_semantic_token_ids does)."
            )


def next_token_matches_target(decoded_next_token: str, target: str) -> bool:
    """Greedy-correctness rule: the model's next token (decoded) must be the
    target or a leading subword of it, ignoring leading whitespace.

    E.g. next token " Atl" matches target "Atlantic"; "2" matches "20".
    Case-sensitive; a whitespace-only next token never matches.
    """
    stripped = decoded_next_token.lstrip()
    return bool(stripped) and target.strip().startswith(stripped)


def resolve_swap_token(
    tokenizer: Any, concept: str
) -> tuple[int, str] | tuple[None, None]:
    """The single vocab id used to build a swap/steering vector for ``concept``.

    A swap needs one concrete direction ``W_U[id]``, so — unlike
    :func:`candidate_token_ids`, which scores *all* surface forms — we must
    commit to one surface. We prefer the leading-space form in the concept's
    own casing (`" Brazil"`), because that is how a concept token appears
    mid-sentence and is the direction the model computes through; we fall back
    to the bare form and then to lowercase variants. Returns
    ``(token_id, surface)`` for the first single-token surface, or
    ``(None, None)`` if no surface is single-token (caller drops-and-logs).
    """
    stripped = concept.strip()
    ordered_surfaces = [
        " " + stripped,
        stripped,
        " " + stripped.lower(),
        stripped.lower(),
        " " + stripped.capitalize(),
        stripped.capitalize(),
    ]
    seen: set[str] = set()
    for surface in ordered_surfaces:
        if surface in seen:
            continue
        seen.add(surface)
        surface_token_ids = tokenizer(surface, add_special_tokens=False).input_ids
        if len(surface_token_ids) == 1:
            return int(surface_token_ids[0]), surface
    return None, None


def formatting_token_mask_Bool_V(
    model: LensModel, token_ids: Sequence[int] | None = None
) -> t.Tensor:
    """``[vocab_size]`` bool, true at the model's formatting tokens
    (:func:`~lens_evals.eval_utils.formatting_token_ids_of` of its
    tokenizer, or the given ``token_ids``): the ids masked out for word ranks."""
    vocab_size = vocab_size_of(model)
    if token_ids is None:
        token_ids = formatting_token_ids_of(model.tokenizer, vocab_size)
    return token_id_mask_Bool_V(token_ids, vocab_size)
