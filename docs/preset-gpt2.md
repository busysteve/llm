# gpt2-tiny, gpt2-small, gpt2-base, gpt2-base-medium, gpt2-base-large, gpt2-medium, and gpt2-large

`gpt2-tiny`, `gpt2-small`, `gpt2-base`, `gpt2-base-medium`, `gpt2-base-large`, `gpt2-medium`, and `gpt2-large` (`llm::presets::gpt2_tiny`, `gpt2_small`, `gpt2_base`, `gpt2_base_medium`, `gpt2_base_large`, `gpt2_medium`, and `gpt2_large`) are the GPT-2 decoder. Positions are learned absolute. Norms are LayerNorm with epsilon `1e-5`. The MLP is erf GELU with `d_ff = 4 * d_model`. Every linear has a bias, including the LM head. Attention is causal. Context is 1024. A diagram of `gpt2_base` is in [gpt2-base-architecture.md](gpt2-base-architecture.md).

Train this shape from scratch. `gguf from` does not load a public GPT-2 GGUF. The checkpoint stays in `model.bin` or an `nnllm` GGUF.

## Sizes

| Preset | Layers | `d_model` | Heads | `d_ff` | Parameters at vocab 16000 |
|---|---:|---:|---:|---:|---:|
| gpt2_tiny | 4 | 256 | 4 | 1024 | 11,629,696 (11.63M) |
| gpt2_small | 6 | 512 | 8 | 2048 | 35,839,616 (35.84M) |
| gpt2_base | 12 | 768 | 12 | 3072 | 110,434,432 (110.43M) |
| gpt2_base_medium | 18 | 768 | 12 | 3072 | 152,961,664 (152.96M) |
| gpt2_base_large | 20 | 768 | 12 | 3072 | 167,137,408 (167.14M) |
| gpt2_medium | 24 | 1024 | 16 | 4096 | 336,144,000 (336.14M) |
| gpt2_large | 36 | 1280 | 20 | 5120 | 750,677,120 (750.68M) |

Head dim is 64. KV heads match the query heads. GELU's first linear writes `d_ff`. Dropout is 0. Embeddings are untied. `nn` rejects tied embeddings, so the token table and the LM head stay separate. `max_seq_len` is 1024.

`train_data` saves a version-2 checkpoint with AdamW moments. `save` without an optimizer writes version 1.

## Usage

```bash
./build.sh amd       # or: ./build.sh opencl, ./build.sh cuda, ./build.sh arc
cmake --build build -j --target train_data

./build/train_data --files books.txt --out run \
  --preset gpt2-tiny --vocab 1024 --seq-len 256 --batch 4 --steps 1000 \
  --prompt "The morning after"
```

`--preset` is `gpt2-tiny`, `gpt2-small`, `gpt2-base`, `gpt2-base-medium`, `gpt2-base-large`, `gpt2-medium`, or `gpt2-large`. `--seq-len` must be at most 1024. `--vocab` must be at least 259. Memory matches the default preset of the same depth: `gpt2-tiny` fits in 12 GB, `gpt2-small` prefers 24 GB at `(4, 512)`, and `gpt2-base` is a 24 GB target. On a 12 GB GPU keep the base width at `--seq-len` 512 or below. `gpt2-base-medium` is 18 layers at the base width, and `gpt2-base-large` is 20 layers at the base width. `gpt2-medium` is 24 layers at width 1024, and `gpt2-large` is 36 layers at width 1280. `gpt2-base-medium`, `gpt2-base-large`, `gpt2-medium`, and `gpt2-large` are past these notes. The notes are one GPU, full attention, and no activation checkpointing. They were sized for float32 activations. Training stores retained activations as bfloat16 and keeps parameters, Adam state, and checkpoints in float32, so the activation working set is about half of these notes.

The run prints a continuation. `chat` continues a transcript inside the 1024-token window. `finetune` keeps training on a new file list.

`gguf export` writes an `nnllm` GGUF that keeps GELU, the linear biases, and the position table. `gguf import` reads that file back. `gguf llama` rejects the checkpoint because of GELU and linear bias.

```cpp
nn::TransformerConfig config = llm::presets::gpt2_tiny(vocab_size);
nn::Transformer model(ctx, config, seed);
```

`vocab_size` must be at least 1. `gpt2_small`, `gpt2_base`, `gpt2_base_medium`, `gpt2_base_large`, `gpt2_medium`, and `gpt2_large` take the same argument.

## Benefits

- The position table, the LayerNorm epsilon `1e-5`, the erf GELU, and a bias on every linear are in the checkpoint.
- The MLP width is `4 * d_model`. The first linear writes that width once.
- `chat` and `Generator` sample from the checkpoint. Context is 1024.
- An `nnllm` GGUF round-trips the block, including biases.

## Applications

- A from-scratch GPT-2-shaped model on a book list, with the continuation `train_data` prints.
- Dialogue through `chat` when each training file is one `User:` / `Assistant:` exchange.
- More text on the same tokenizer through `finetune`.
- A comparison with `gpt3-*`, which is this block at context 2048, or with `bert-*`, which is this dense block run in both directions at context 512.
- An `nnllm` archive of the weights for a loader that accepts this library's GGUF.

## See also

- [gpt2-base-architecture.md](gpt2-base-architecture.md) — diagram and specs for `gpt2_base`
- [preset-gpt3.md](preset-gpt3.md) — the same block with context 2048
- [preset-bert.md](preset-bert.md) — the same dense block, bidirectional, context 512
- [preset-default.md](preset-default.md) — SwiGLU, bias off, learned positions
- [README.md](../README.md) — preset table and parameter counts
- [USAGE.md](../USAGE.md) — library notes
