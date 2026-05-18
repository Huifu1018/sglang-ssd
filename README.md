# sglang-ssd

SGLang SSD-style speculative decoding for heterogeneous target/draft models.

This repository contains a deep SGLang 0.5.9 integration for running:

- target: `cyankiwi/MiniMax-M2.7-AWQ-4bit`
- draft: `Qwen/Qwen2.5-1.5B-Instruct`
- tokenizer relation: heterogeneous vocabularies

The package name and Python module are still `sglang-group` / `sglang_group`
for compatibility with the existing implementation, but this repository is the
SSD-oriented version.

## What Is Implemented

This is not an outer wrapper around SGLang serving. The implementation patches
and extends SGLang's speculative decoding path:

- Native `SGLANG_GROUP` speculative algorithm registration for SGLang 0.5.9.
- Scheduler post-commit hooks after prefill/decode output processing.
- Worker-level SSD async proposal service.
- `async-hit` and `async-sync-fallback` proposal policies.
- CUDA stream/event proposal overlap.
- Ready/miss batch split so ready requests still use speculative verify.
- Heterogeneous tokenizer bridge for MiniMax target ids and Qwen draft ids.
- TLI shared-token probability path for sampling.
- Custom greedy verifier fast path.
- Branch/tree candidate rows with first-child/sibling verifier metadata.
- CUDA tree verifier path through `sgl_kernel.verify_tree_greedy`.
- Local torch tree-walk fallback for non-CUDA tests.

Current branch generation supports root top-k siblings plus the greedy branch.
Deeper multi-level branch expansion and full benchmark tuning are the next
optimization layer.

## Install

```bash
git clone https://github.com/Huifu1018/sglang-ssd.git
cd sglang-ssd
pip install -e ".[sglang]"
```

The `sglang` extra pins:

```text
sglang==0.5.9
```

For development:

```bash
pip install -e ".[dev]"
python -m unittest discover -s tests -p "test_*.py"
```

## Patch SGLang Source

Install the source-level SGLang integration before production-style runs:

```bash
sglang-group-install-sglang-patch
sglang-group-install-sglang-patch --check
```

This patches:

- `sglang/srt/speculative/spec_info.py`
- `sglang/srt/server_args.py`
- `sglang/srt/managers/scheduler_output_processor_mixin.py`

Backups are written with `.sglang-group.bak`. After patching, SGLang accepts:

```bash
--speculative-algorithm SGLANG_GROUP
```

without rewriting it to `NGRAM`.

For a local SGLang source tree:

```bash
PYTHONPATH=/path/to/sglang-ssd \
python -m sglang_group.cli.install_sglang_patch \
  --sglang-root /path/to/sglang
```

## Run MiniMax + Qwen SSD Mode

Recommended first run for the AWQ target:

```bash
CUDA_VISIBLE_DEVICES=0 sglang-group-launch \
  --model-path cyankiwi/MiniMax-M2.7-AWQ-4bit \
  --host 0.0.0.0 \
  --port 30000 \
  --trust-remote-code \
  --speculative-algorithm SGLANG_GROUP \
  --speculative-draft-model-path Qwen/Qwen2.5-1.5B-Instruct \
  --speculative-num-steps 4 \
  --speculative-num-draft-tokens 5 \
  --sglang-group-method auto \
  --sglang-group-draft-backend sglang \
  --sglang-group-max-context-tokens 8192 \
  --sglang-group-ssd-mode async-hit \
  --sglang-group-verify-backend auto \
  --sglang-group-tokenizer-bridge uag \
  --sglang-group-tree-branch-factor 4 \
  --sglang-group-tree-max-depth 3 \
  --sglang-group-metrics-log-interval 5
```

Use `async-sync-fallback` when you want to preserve speculative decoding on
proposal misses by synchronously filling the proposal:

```bash
--sglang-group-ssd-mode async-sync-fallback
```

Use `async-hit` when you want draft work off the target critical path. Misses
fall back to a safe root-only verify path, while ready requests continue through
speculative verify.

## Baseline Comparison

Run a target-only SGLang server on a second port:

```bash
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path cyankiwi/MiniMax-M2.7-AWQ-4bit \
  --host 0.0.0.0 \
  --port 30001 \
  --trust-remote-code
```

Compare with the same prompts, sampling settings, context length, tensor
parallel size, and memory fraction.

## Key Runtime Flags

| Flag | Default | Purpose |
| --- | --- | --- |
| `--sglang-group-method` | `auto` | `itl`, `itl-base-slem`, `itl-base-tli`, or temperature-based auto routing. |
| `--sglang-group-draft-backend` | `sglang` | Runs the draft model through SGLang's low-level `ModelRunner`. |
| `--sglang-group-ssd-mode` | `off` | `off`, `async-hit`, or `async-sync-fallback`. |
| `--sglang-group-ssd-prefetch-workers` | `1` | Background proposal worker count. |
| `--sglang-group-ssd-max-prefetch` | `256` | Completed async proposal cache size. |
| `--sglang-group-verify-backend` | `auto` | `auto`, `sglang`, `torch`, or `triton`. |
| `--sglang-group-tokenizer-bridge` | `uag` | UAG/lookbehind retokenization or segment-only retokenization. |
| `--sglang-group-tree-branch-factor` | `1` | Root top-k branch factor. Use `4` for branch/tree testing. |
| `--sglang-group-tree-max-depth` | unset | Optional cap on verifier tree depth. |
| `--no-sglang-group-cuda-overlap` | disabled | Turns off draft proposal CUDA stream/event overlap. |
| `--sglang-group-max-context-tokens` | unset | Caps draft-side context length. |
| `--sglang-group-metrics-log-interval` | `60` | Periodic metrics log interval. Use `0` to disable. |

## Method Routing

`auto` routes by request temperature:

```text
temperature == 0       -> itl-base-slem
0 < temperature < 0.9  -> itl-base-tli
temperature >= 0.9     -> itl
```

`itl-base-slem` and `itl` retokenize Qwen draft ids into MiniMax target ids
before verification. `itl-base-tli` restricts draft probabilities to shared
token strings and scatters probability rows into the target vocabulary.

## Metrics To Watch

Set `--sglang-group-metrics-log-interval 5` during tests. Important fields:

- `acceptance_rate`
- `draft_prepare_ms_per_verify_batch`
- `target_verify_ms_per_verify_batch`
- `verify_postprocess_ms_per_verify_batch`
- `proposal_cache_hits`
- `accepted_on_proposal_cache_hit`
- `draft_cache_hits`
- `async.cache_size`
- `async.inflight`
- `async.cuda_overlap`
- `tree_proposals`
- `tree_nodes`

The current implementation should be treated as a high-performance SGLang SSD
engine path, but final "extreme performance" claims require real GPU benchmark
results for throughput, latency, acceptance rate, and overlap utilization.

## Limitations

- Targets SGLang 0.5.9.
- No pipeline parallel support yet.
- No DP attention support yet.
- Branch generation currently adds root top-k siblings; deeper multi-level fork
  expansion is future work.
- Sampling/TLI branch probability rows are still linear; branch/tree mode is
  intended first for greedy/text proposal paths.
- SGLang-native accepted-context draft KV cache is experimental and disabled by
  default.

## Validation

```bash
python -m compileall -q sglang_group
python -m unittest discover -s tests -p "test_*.py"
```

