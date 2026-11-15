# qwen-tiny, qwen-small, and qwen-base

`qwen-tiny`, `qwen-small`, and `qwen-base` (`llm::presets::qwen_tiny`, `qwen_small`, and `qwen_base`) are the Qwen3 decoder. Attention is causal and grouped. Head dim is 128. Q and K each pass an RMSNorm of width 128 before RoPE. The RoPE base is `1e6`. The RMSNorm epsilon is `1e-6`. Bias is off. A diagram of `qwen_base` is in [qwen-base-architecture.md](qwen-base-architecture.md).

Train this shape from scratch. The preset does not load published Qwen weights. A public Qwen2 or Qwen3 GGUF comes in through `gguf from`.

## Sizes

| Preset | Layers | `d_model` | Heads | KV heads | Head dim | `d_ff` | Parameters at vocab 16000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen_tiny | 4 | 256 | 4 | 2 | 128 | 768 | 12,127,488 (12.13M) |
| qwen_small | 6 | 512 | 8 | 2 | 128 | 1536 | 38,412,288 (38.41M) |
| qwen_base | 12 | 768 | 12 | 4 | 128 | 2048 | 118,970,112 (118.97M) |

Q is `num_heads * 128` wide, so the query projection is twice `d_model`. K and V stay at `num_kv_heads * 128`. SwiGLU's first linear writes `2 * d_ff`. Dropout is 0. Embeddings are untied. `max_seq_len` is 1024.

`train_books` saves a version-4 checkpoint with AdamW moments. `save` without an optimizer writes version 3.

## Usage

```bash
./build.sh opencl   # or: ./build.sh cuda
cmake --build build -j --target train_books

./build/train_books --files books.txt --out run \
  --preset qwen-tiny --vocab 1024 --seq-len 256 --batch 4 --steps 1000 \
  --prompt "The morning after"
```

`--preset` is `qwen-tiny`, `qwen-small`, or `qwen-base`. `--seq-len` must be at most 1024. `--vocab` must be at least 259. The extra parameters versus `llama-*` are the wider Q projection and the QK-norm weights. Treat memory as the matching default width, and give `qwen-base` 24 GB. On a 12 GB GPU keep that width at `--seq-len` 512 or below. The notes are float32, one GPU, full attention, and no activation checkpointing.

The run prints a continuation. `chat` continues a transcript inside the 1024-token window. `finetune` keeps training on a new file list.

`gguf llama` writes architecture `qwen3`, with the wider head, the grouped K and V, QK-norm, RoPE base `1e6`, and RMSNorm epsilon `1e-6`. llama.cpp runs that block. `gguf from` loads a public Qwen2 or Qwen3 GGUF into a directory `chat` can run. A Qwen2 file has no QK-norm. A Qwen3 file has it. That import does not use `--preset`.

```cpp
nn::TransformerConfig config = llm::presets::qwen_tiny(vocab_size);
nn::Transformer model(ctx, config, seed);
```

`vocab_size` must be at least 1. `qwen_small` and `qwen_base` take the same argument.

## Benefits

- Two or four KV heads serve the query heads, so the K and V cache is narrower than a square-head model of the same `d_model`.
- Head dim 128, QK-norm, RoPE base `1e6`, and RMSNorm epsilon `1e-6` are stored in the checkpoint and in the `qwen3` export.
- `gguf llama` writes architecture `qwen3`. A public Qwen2 or Qwen3 GGUF of this block family loads through `gguf from`.
- `chat` and `Generator` sample from a checkpoint trained with this preset. Context is 1024.

## Applications

- A from-scratch Qwen3-shaped model on a book list or a dialogue list.
- Sampling with `chat`, then the same weights in llama.cpp as `qwen3`.
- More next-token training on a new file list through `finetune`.
- A published Qwen2 or Qwen3 GGUF run locally through `gguf from` and `chat`.

## See also

- [qwen-base-architecture.md](qwen-base-architecture.md) — diagram and specs for `qwen_base`
- [preset-llama.md](preset-llama.md) — square heads, RoPE base 10000, no QK-norm
- [preset-default.md](preset-default.md) — learned positions at these depths
- [README.md](../README.md) — preset table and parameter counts
- [USAGE.md](../USAGE.md) — library notes
- [gguf](../examples/gguf.md) — `qwen3` export and `gguf from`
