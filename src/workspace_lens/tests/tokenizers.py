"""Toy tokenizers for the CPU-only eval tests (TinyDecoder fixtures)."""

from __future__ import annotations

from types import SimpleNamespace

import torch as t


class CharTokenizer:
    """Case-insensitive character tokenizer: a-z -> 1..26, newline -> 27,
    space -> 28, anything else -> 29; BOS 0. Multi-character strings are
    multi-token, so only single letters have single-token surfaces. Id 30 is
    an EOS that ``__call__`` never emits; it decodes to the word-like
    ``"<eos>"`` so tests can tell the special-id check apart from the
    punctuation check. Ids past 30 decode to ``"?"`` (vocab padding)."""

    bos_token_id = 0
    all_special_ids = [0, 30]

    def _char_id(self, char: str) -> int:
        char = char.lower()
        if "a" <= char <= "z":
            return 1 + ord(char) - 97
        if char == "\n":
            return 27
        if char == " ":
            return 28
        return 29

    def __call__(
        self,
        text: str,
        return_tensors: str | None = None,
        truncation: bool = True,
        max_length: int = 128,
        add_special_tokens: bool = True,
    ) -> SimpleNamespace:
        ids = ([0] if add_special_tokens else []) + [
            self._char_id(char) for char in text
        ][: max_length - 1]
        # TinyDecoder.encode omits return_tensors but expects a tensor (the
        # real ByteTokenizer defaults to "pt"); candidate_token_ids passes
        # add_special_tokens=False and expects a plain list (HF default).
        if return_tensors == "pt" or add_special_tokens:
            return SimpleNamespace(input_ids=t.tensor([ids]))
        return SimpleNamespace(input_ids=ids)

    def decode(self, ids, **_kwargs) -> str:
        special = {0: "", 27: "\n", 28: " ", 30: "<eos>"}
        return "".join(
            special.get(int(i), chr(96 + int(i)) if 1 <= int(i) <= 26 else "?")
            for i in ids
        )

    def batch_decode(self, batch_ids, **_kwargs) -> list[str]:
        return [self.decode(ids) for ids in batch_ids]
