# gpt3-tiny, gpt3-small, and gpt3-base

`gpt3-tiny`, `gpt3-small`, and `gpt3-base` (`llm::presets::gpt3_tiny`, `gpt3_small`, and `gpt3_base`) are the GPT-2 dense decoder with `max_seq_len` 2048. Positions are learned absolute. Norms are LayerNorm with epsilon `1e-5`. The MLP is erf GELU with `d_ff = 4 * d_model`. Every linear has a bias. Attention is causal and dense in every layer. A diagram of `gpt3_base` is in [gpt3-base-architecture.md](gpt3-base-architecture.md).

The GPT-3 paper alternates dense attention with locally banded sparse attention. This library has no sparse attention. The extra parameters versus `gpt2-*` are the longer position table: `1024 * d_model` rows.

Train this shape from scratch. OpenAI did not release GPT-3 weights, and `gguf from` does not load a public GPT-2 GGUF. The checkpoint stays in `model.bin` or an `nnllm` GGUF.

## Sizes

| Preset | Layers | `d_model` | Heads | `d_ff` | Parameters at vocab 16000 |
|---|---:|---:|---:|---:|---:|
| gpt3_tiny | 4 | 256 | 4 | 1024 | 11,891,840 (11.89M) |
| gpt3_small | 6 | 512 | 8 | 2048 | 36,363,904 (36.36M) |
| gpt3_base | 12 | 768 | 12 | 3072 | 111,220,864 (111.22M) |

Head dim is 64. KV heads match the query heads. GELU's first linear writes `d_ff`. Dropout is 0. Embeddings are untied. `max_seq_len` is 2048.

`train_data` saves a version-2 checkpoint with AdamW moments. `save` without an optimizer writes version 1. The file is the same version as a `gpt2-*` checkpoint.

## Usage

```bash
./build.sh amd       # or: ./build.sh opencl, ./build.sh cuda, ./build.sh arc
cmake --build build -j --target train_data

./build/train_data --files books.txt --out run \
  --preset gpt3-tiny --vocab 1024 --seq-len 512 --batch 4 --steps 1000 \
  --prompt "The morning after"
```

`--preset` is `gpt3-tiny`, `gpt3-small`, or `gpt3-base`. `--seq-len` must be at most 2048. `--vocab` must be at least 259. Attention memory grows with the length. The longer position table is cheap beside that. On a 12 GB GPU prefer `--seq-len` 512 or below for `gpt3-base`. `gpt3-tiny` is the width that can spend the extra context on a 12 GB card. The notes are one GPU, full attention, and no activation checkpointing. They were sized for float32 activations. Training stores retained activations as bfloat16 and keeps parameters, Adam state, and checkpoints in float32, so the activation working set is about half of these notes.

The run prints a continuation. `chat` uses the 2048-token window and drops the oldest `User:` / `Assistant:` turn until the prompt plus `--max-new-tokens` fits. `finetune` keeps training on a new file list. Its `--seq-len` must also stay at or below 2048.

`gguf export` writes an `nnllm` GGUF that keeps GELU, the linear biases, and the 2048-row position table. `gguf llama` rejects the checkpoint because of GELU and linear bias.

```cpp
nn::TransformerConfig config = llm::presets::gpt3_tiny(vocab_size);
nn::Transformer model(ctx, config, seed);
```

`vocab_size` must be at least 1. `gpt3_small` and `gpt3_base` take the same argument.

## Benefits

- Context is 2048, so a training window and a `chat` transcript can be twice as long as `gpt2-*`.
- The block matches `gpt2-*`: learned positions, LayerNorm epsilon `1e-5`, erf GELU, biases, and `d_ff = 4 * d_model`.
- The checkpoint is version 2, the same façade version as `gpt2-*`, `tiny`, and `llama-*`.
- An `nnllm` GGUF round-trips the longer position table.

## Applications

- Book windows up to 2048 tokens.
- Dialogue whose history needs more than 1024 tokens before `chat` starts dropping turns.
- A from-scratch dense decoder at GPT-3's context length, sampled with `chat` or the continuation `train_data` prints.
- More text on the same tokenizer through `finetune`, with `--seq-len` raised toward 2048 when the GPU allows it.
- A comparison with `gpt2-*` at the same depth when the only change you want is the position table and the context limit.

## See also

- [gpt3-base-architecture.md](gpt3-base-architecture.md) — diagram and specs for `gpt3_base`
- [preset-gpt2.md](preset-gpt2.md) — the same block with context 1024
- [preset-bert.md](preset-bert.md) — the same dense block, bidirectional, context 512
- [README.md](../README.md) — preset table and parameter counts
- [USAGE.md](../USAGE.md) — library notes
