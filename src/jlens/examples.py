# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
# Modified by Kola Ayonrinde, 2026: removed the slice-visualisation example prompts.
"""A WikiText loader for lens-fitting prompts."""

from __future__ import annotations


def load_wikitext_prompts(n_prompts: int, *, min_chars: int = 600) -> list[str]:
    """Return the first ``n_prompts`` WikiText-103 records of at least
    ``min_chars`` characters, streamed from the HuggingFace Hub (requires
    ``datasets``)."""
    if n_prompts <= 0:
        return []
    from datasets import load_dataset

    dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True
    )
    prompts: list[str] = []
    for record in dataset:
        text = record["text"]
        if len(text.strip()) >= min_chars:
            prompts.append(text)
            if len(prompts) == n_prompts:
                break
    return prompts
