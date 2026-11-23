# llama-tiny, llama-small, llama-base, llama-medium, and llama-large

`llama-tiny`, `llama-small`, `llama-base`, `llama-medium`, and `llama-large` (`llm::presets::llama_tiny`, `llama_small`, `llama_base`, `llama_medium`, and `llama_large`) are the default widths with RoPE and RMSNorm. Attention is causal. Head dim is 64, KV heads match the query heads, the RoPE base is 10000, and the RMSNorm epsilon is `1e-5`. The checkpoint has no position table and no norm biases. A diagram of `llama_base` is in [llama-base-architecture.md](llama-base-architecture.md).

Train this shape from scratch. A checkpoint that was trained with learned positions stays a learned-position model.

## Sizes

| Preset | Layers | `d_model` | Heads | `d_ff` | Parameters at vocab 16000 |
|---|---:|---:|---:|---:|---:|
| llama_tiny | 4 | 256 | 4 | 768 | 11,602,176 (11.60M) |
| llama_small | 6 | 512 | 8 | 1536 | 36,837,888 (36.84M) |
| llama_base | 12 | 768 | 12 | 2048 | 109,529,856 (109.53M) |
| llama_medium | 24 | 1024 | 16 | 2816 | 341,099,520 (341.10M) |
| llama_large | 36 | 1280 | 20 | 3584 | 772,435,200 (772.44M) |

SwiGLU's first linear writes `2 * d_ff`. Dropout is 0. Embeddings are untied. Bias is off. `max_seq_len` is 1024.

`train_data` saves a version-2 checkpoint with AdamW moments. `save` without an optimizer writes version 1.

## Usage

```bash
./build.sh amd       # or: ./build.sh opencl, ./build.sh cuda, ./build.sh arc
cmake --build build -j --target train_data

./build/train_data --files books.txt --out run \
  --preset llama-tiny --vocab 1024 --seq-len 256 --batch 4 --steps 1000 \
  --prompt "The morning after"
```

`--preset` is `llama-tiny`, `llama-small`, `llama-base`, `llama-medium`, or `llama-large`. `--seq-len` must be at most 1024. `--vocab` must be at least 259. Memory follows the matching default width: `llama-tiny` fits in 12 GB, `llama-small` prefers 24 GB at `(4, 512)`, and `llama-base` is a 24 GB target. On a 12 GB GPU keep the base width at `--seq-len` 512 or below. `llama-medium` is 24 layers at width 1024, and `llama-large` is 36 layers at width 1280. Both are past these notes. The notes are one GPU, full attention, and no activation checkpointing. They were sized for float32 activations. Training stores retained activations as bfloat16 and keeps parameters, Adam state, and checkpoints in float32, so the activation working set is about half of these notes.

The run prints a continuation. `chat` loads the directory and continues a `User:` / `Assistant:` transcript inside the 1024-token window. `finetune` keeps training on a new file list.

`gguf llama` writes architecture `llama`. The file uses this library's tensors under llama.cpp names and splits SwiGLU `fc1` into `ffn_gate` and `ffn_up`. llama.cpp runs the same block, so the export does not print the learned-position warning. A public llama.cpp GGUF of this RoPE / RMSNorm / SwiGLU block loads through `gguf from` into a directory `chat` can run. That import is separate from `--preset`.

```cpp
nn::TransformerConfig config = llm::presets::llama_tiny(vocab_size);
nn::Transformer model(ctx, config, seed);
```

`vocab_size` must be at least 1. `llama_small`, `llama_base`, `llama_medium`, and `llama_large` take the same argument.

## Benefits

- The block is the one llama.cpp runs: adjacent-pair RoPE, base 10000, RMSNorm epsilon `1e-5`, square attention, SwiGLU, and no linear bias.
- `gguf llama` keeps every trained tensor. The position table and the LayerNorm biases are absent because this preset never allocated them.
- Widths match `tiny`, `small`, `base`, `medium`, and `large`, with a smaller parameter count because those tables are gone.
- `chat` and `Generator` sample from the checkpoint. Context is 1024.

## Applications

- A from-scratch decoder you will run in this library and then in llama.cpp.
- Book continuation through `train_data`, and dialogue through `chat`.
- More text on the same tokenizer through `finetune`.
- A public Llama-style GGUF brought in with `gguf from` and sampled with `chat`.
- A width-matched comparison with `tiny`, `small`, `base`, `medium`, and `large`, which keep a learned position table.

## See also

- [llama-base-architecture.md](llama-base-architecture.md) — diagram and specs for `llama_base`
- [preset-default.md](preset-default.md) — the same widths with learned positions
- [preset-qwen.md](preset-qwen.md) — grouped-query attention, head dim 128, QK-norm
- [README.md](../README.md) — preset table and parameter counts
- [USAGE.md](../USAGE.md) — library notes
- [gguf](../examples/gguf.md) — llama.cpp export and `gguf from`
