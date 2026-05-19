# sglang-ssd

面向 SGLang 的 SSD 风格推测解码实现，重点支持异构 target/draft 模型组合。

当前重点适配：

- target 模型：`cyankiwi/MiniMax-M2.7-AWQ-4bit`
- draft 模型：`Qwen/Qwen2.5-1.5B-Instruct`
- tokenizer 关系：MiniMax 与 Qwen 使用不同 tokenizer / vocab

为了兼容已有实现，Python 包名和模块名仍保留为 `sglang-group` / `sglang_group`，但这个仓库是面向 SSD 深改路线的版本。

## 项目定位

这个项目不是在 SGLang 外层套一层普通 wrapper，而是对 SGLang 0.5.9 的 speculative decoding 路径做内部集成：

- 注册原生 `SGLANG_GROUP` speculative algorithm。
- 在 scheduler 的 prefill/decode output processing 后挂接 post-commit hook。
- 在 worker 层加入 SSD 异步 proposal 服务。
- 支持 `async-hit` 与 `async-sync-fallback` 两种 proposal 策略。
- 支持 draft/target CUDA stream 与 event 级 overlap。
- 支持 ready/miss batch split，让 ready 请求继续走 speculative verify。
- 支持 MiniMax target ids 与 Qwen draft ids 的异构 tokenizer bridge。
- 支持 TLI shared-token probability path。
- 支持 greedy verifier fast path。
- 支持 branch/tree candidate rows。
- 支持 first-child / next-sibling verifier metadata。
- 支持通过 `sgl_kernel.verify_tree_greedy` 进入 CUDA tree verifier 路径。
- 支持非 CUDA 环境下的 torch tree-walk fallback，方便本地单测。

当前 branch generation 已支持 root top-k siblings + greedy branch。更深层的多级分支展开和完整 benchmark 调参仍属于下一层优化。

## 安装

```bash
git clone https://github.com/Huifu1018/sglang-ssd.git
cd sglang-ssd
pip install -e ".[sglang]"
```

`sglang` extra 当前固定：

```text
sglang==0.5.9
```

开发环境：

```bash
pip install -e ".[dev]"
python -m unittest discover -s tests -p "test_*.py"
```

## 安装 SGLang 源码补丁

正式运行前需要把 SGLang 侧的源码级集成补丁打进去：

```bash
sglang-group-install-sglang-patch
sglang-group-install-sglang-patch --check
```

补丁会修改：

- `sglang/srt/speculative/spec_info.py`
- `sglang/srt/server_args.py`
- `sglang/srt/managers/scheduler_output_processor_mixin.py`

原文件会备份为 `.sglang-group.bak`。打完补丁后，SGLang 可以直接接受：

```bash
--speculative-algorithm SGLANG_GROUP
```

不会再被重写成 `NGRAM`。

如果使用本地 SGLang 源码树：

```bash
PYTHONPATH=/path/to/sglang-ssd \
python -m sglang_group.cli.install_sglang_patch \
  --sglang-root /path/to/sglang
```

## MiniMax + Qwen 启动方式

AWQ target 的推荐起步命令：

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

如果希望 proposal miss 时仍尽量保持 speculative decoding，可以使用同步补齐模式：

```bash
--sglang-group-ssd-mode async-sync-fallback
```

如果希望 draft 尽量不阻塞 target 关键路径，使用：

```bash
--sglang-group-ssd-mode async-hit
```

在 `async-hit` 下，proposal cache hit 的请求会走 speculative verify，miss 的请求会走安全的 root-only verify fallback。

注意：如果使用 `--tp-size > 1` 且 draft backend 是 `sglang`，后台 draft 线程会涉及 tensor-parallel collective。当前实现会为 draft ModelRunner 创建独立 TP group/communicator，避免 draft collective 和 target collective 共用同一个 TP communicator 后乱序卡死。独立 draft TP group 创建失败时会启动失败，不会自动降级到 `async-sync-fallback`。

## 基线对比

建议同时启动一个 target-only SGLang server 作为基线：

```bash
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path cyankiwi/MiniMax-M2.7-AWQ-4bit \
  --host 0.0.0.0 \
  --port 30001 \
  --trust-remote-code
```

对比时需要保持以下条件一致：

- prompt 集合
- sampling 参数
- context length
- tensor parallel size
- GPU memory fraction
- batch / concurrency 配置

## 关键参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--sglang-group-method` | `auto` | `itl`、`itl-base-slem`、`itl-base-tli` 或按 temperature 自动路由。 |
| `--sglang-group-draft-backend` | `sglang` | draft 模型通过 SGLang 低层 `ModelRunner` 执行。 |
| `--sglang-group-ssd-mode` | `off` | `off`、`async-hit` 或 `async-sync-fallback`。 |
| `--sglang-group-ssd-prefetch-workers` | `1` | 后台 proposal worker 数量。 |
| `--sglang-group-ssd-max-prefetch` | `256` | 异步 proposal 完成缓存大小。 |
| `--sglang-group-verify-backend` | `auto` | `auto`、`sglang`、`torch` 或 `triton`。 |
| `--sglang-group-tokenizer-bridge` | `uag` | UAG/lookbehind retokenization 或 segment-only retokenization。 |
| `--sglang-group-tree-branch-factor` | `1` | root top-k branch factor；branch/tree 测试建议设为 `4`。 |
| `--sglang-group-tree-max-depth` | 未设置 | verifier tree 最大深度上限。 |
| `--no-sglang-group-cuda-overlap` | 默认关闭该开关 | 关闭 draft proposal CUDA stream/event overlap。 |
| `--sglang-group-max-context-tokens` | 未设置 | 限制 draft 侧上下文长度。 |
| `--sglang-group-metrics-log-interval` | `60` | 周期性 metrics log 间隔，设为 `0` 可关闭。 |

## 方法路由

`auto` 会根据请求 temperature 自动选择方法：

```text
temperature == 0       -> itl-base-slem
0 < temperature < 0.9  -> itl-base-tli
temperature >= 0.9     -> itl-base-tli
```

说明：

- `itl-base-slem` 与 `itl` 会把 Qwen draft ids retokenize 成 MiniMax target ids 后再 verify。
- `itl-base-tli` 会限制在共享 token string 上计算 draft probability，再 scatter 到 target vocab。
- 非 greedy sampling 会强制路由到 `itl-base-tli`，避免 `itl` / `itl-base-slem` 缺少 draft probability 时破坏输出分布。
- 异构 tokenizer 场景下，`--sglang-group-tokenizer-bridge uag` 是推荐起点。

## 需要重点观察的指标

测试时建议设置：

```bash
--sglang-group-metrics-log-interval 5
```

重点看：

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

## 当前限制

- 目标版本是 SGLang 0.5.9。
- 暂不支持 pipeline parallel。
- 暂不支持 DP attention。
- branch generation 当前主要是 root top-k siblings + greedy branch，更深层多级 fork expansion 还需要继续开发。
- sampling/TLI branch probability rows 当前仍偏线性，branch/tree 模式优先服务 greedy/text proposal path。
- SGLang-native accepted-context draft KV cache 仍是实验功能，默认关闭。
- 是否达到最终极限性能，需要在目标 GPU、真实 batch/concurrency 和真实业务 prompt 上用 benchmark 验证，不能只靠代码路径判断。

## 验证

```bash
python -m compileall -q sglang_group tests
PYTHONPATH=. python -m unittest discover -s tests -p "test_*.py"
```

当前本地验证结果：

```text
Ran 52 tests
OK (skipped=5)
```
