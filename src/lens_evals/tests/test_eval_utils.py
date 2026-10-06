"""Token-level rules of ``eval_utils`` that no eval item is needed for, on the
character-level stub tokenizer. The item-flavoured ``eval_utils`` tests
(surface forms, candidate ids, the greedy-correctness rule) live in
``test_readout_evals``."""

from __future__ import annotations

from lens_evals.eval_utils import (
    added_token_ids,
    all_non_semantic_token_ids,
    formatting_token_ids_of,
    is_non_semantic_token,
    is_semantic_token_id,
)
from workspace_lens.tests.tokenizers import CharTokenizer


def test_is_semantic_token_id() -> None:
    tokenizer = CharTokenizer()
    assert is_semantic_token_id(2, tokenizer)  # 'b'
    assert not is_semantic_token_id(29, tokenizer)  # punctuation
    assert not is_semantic_token_id(28, tokenizer)  # space
    # <eos> decodes to letters, so only the all_special_ids check excludes it.
    assert not is_non_semantic_token(tokenizer.decode([30]))
    assert not is_semantic_token_id(30, tokenizer)


def test_formatting_token_ids_are_non_semantic_plus_special() -> None:
    tokenizer = CharTokenizer()
    non_semantic = all_non_semantic_token_ids(tokenizer, 32)
    assert {27, 28, 29} <= non_semantic  # newline, space, punctuation
    assert 2 not in non_semantic  # 'b'
    formatting = formatting_token_ids_of(tokenizer, 32)
    assert set(formatting) == non_semantic | set(tokenizer.all_special_ids)
    assert 30 in formatting  # <eos>: word-like decode, excluded via all_special_ids
    assert 2 not in formatting


class _ThinkTokenizer(CharTokenizer):
    """The char tokenizer plus one *added but not special* control token:
    id 31 decodes to the word-like ``"<think>"`` and is registered in the
    added vocabulary only, as Qwen's ``<think>`` is."""

    def get_added_vocab(self) -> dict[str, int]:
        return {"<think>": 31}

    def decode(self, token_ids: list[int]) -> str:
        return "".join("<think>" if token_id == 31 else super(_ThinkTokenizer, self).decode([token_id]) for token_id in token_ids)


def test_formatting_token_ids_include_added_control_tokens() -> None:
    tokenizer = _ThinkTokenizer()
    assert added_token_ids(tokenizer) == {31}
    assert added_token_ids(CharTokenizer()) == set()  # no added vocabulary
    assert 31 not in all_non_semantic_token_ids(tokenizer, 32)  # word-like decode
    assert 31 not in tokenizer.all_special_ids
    assert 31 in formatting_token_ids_of(tokenizer, 32)
