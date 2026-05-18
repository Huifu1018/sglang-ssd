# Method Notes

`sglang-group` exposes one SGLang algorithm name, `SGLANG_GROUP`, with multiple
runtime methods.

## `itl`

TokenTiming-style route:

```text
target ids -> target text
target text -> draft ids
draft model proposes draft ids
draft ids + lookbehind -> draft text
draft text -> target ids
suffix alignment -> new target proxy ids
DTW alignment diagnostics
target model verifies proxy ids
```

This method is useful for high-temperature sampling in the current MiniMax
benchmarks. By default it uses the same UAG-style tokenizer bridge as SLEM so
raw Qwen ids are never passed to the MiniMax verifier as if the vocabularies were
shared. Set `SGLANG_GROUP_TOKENIZER_BRIDGE=segment` only for controlled
experiments with new-text-only retokenization.

## `itl-base-slem`

SLEM/UAG-style string re-tokenization route:

```text
target ids -> target text
target text -> draft ids
draft model proposes draft ids
draft ids + lookbehind -> draft text
draft text -> target ids
suffix alignment -> new target proxy ids
target model verifies proxy ids
```

This method is greedy-only and is the default `auto` route for
`temperature=0`.

## `itl-base-tli`

TLI shared-token route:

```text
assistant vocab token string == target vocab token string
assistant id -> target id
draft logits restricted to shared token strings
draft probabilities mapped into target vocabulary rows
target verifier runs speculative rejection sampling
```

This method is the default `auto` route for mid-temperature sampling.

## `auto`

Default routing:

```text
temperature == 0       -> itl-base-slem
0 < temperature < 0.9  -> itl-base-tli
temperature >= 0.9     -> itl
```

The threshold and route methods are configurable from the launch wrapper.

## Verifier and overlap

`SGLANG_GROUP_VERIFY_BACKEND=auto` enables the SGLANG_GROUP greedy verifier fast
path. Linear rows use the local Triton/torch verifier. Tree rows use the CUDA
tree decision kernel directly when available, with a local tree-walk fallback.
`SGLANG_GROUP_VERIFY_BACKEND=sglang` forces the upstream NGRAM verifier.

`SGLANG_GROUP_TREE_BRANCH_FACTOR>1` enables branch/tree candidate rows. The
current proposer adds draft top-k root siblings and preserves the greedy branch
as the first child. `SGLANG_GROUP_TREE_MAX_DEPTH` can cap branch depth.

When `SGLANG_GROUP_SSD_MODE` is not `off`, draft proposal workers run on a
dedicated CUDA stream by default. Ready-only mode consumes only proposals whose
CUDA event has completed; sync-fallback mode makes the current stream wait on
the proposal event before verifier input uses GPU-resident draft probability
rows.
