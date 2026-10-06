# Experiments

Prompt sets for the global-workspace experiments. Each `{slug}.json` is prompts only.

## Conventions

Unless a section says otherwise:

- **Lens readout** — at each (layer, token position) the Jacobian lens
  returns a ranked list of vocabulary tokens.
- **Workspace band** — the contiguous mid-network layer range where
  workspace content is read; experiments report over this band, not
  individual layers.
- **Hit** — a target token is a *hit* if it appears at lens rank 1 at any
  (layer, position) in the band over the scored span.
- **Swap** — clamping a lens coordinate replaces one token's direction with
  another's at every band layer at the specified positions, then samples
  the continuation.
- Prompts that span multiple turns are given as
  `[{"role": "user"|"assistant", "content": ...}]`.

## probe-swap

[`probe-swap.json`](probe-swap.json)

90 two-hop factual prompts. `items[*].prompt` ends just before the answer; `intermediate` is the bridge entity, `swap_to` the replacement. Baseline: greedy next-token == `answer`. Swap: replace the `intermediate` representation (linear-probe direction) with `swap_to` across the band at every prompt token position; score next-token at the final position == `swap_answer`. `category` groups items by relation type for the per-category breakdown.
