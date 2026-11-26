# tiny, small, base, base-medium, base-large, medium, and large

`tiny`, `small`, `base`, `base-medium`, `base-large`, `medium`, and `large` are the default decoder presets (`llm::presets::tiny`, `small`, `base`, `base_medium`, `base_large`, `medium`, and `large`). They are a bias-free SwiGLU stack with learned absolute positions, LayerNorm, and context 1024. `train_data` uses `tiny` when `--preset` is omitted. `examples/train_tiny.cpp` trains `tiny`. `base-medium` and `base-large` keep the base width.

## Sizes

| Preset | Layers | `d_model` | Heads | `d_ff` | Parameters at vocab 16000 |
|---|---:|---:|---:|---:|---:|
| tiny | 4 | 256 | 4 | 768 | 11,866,624 (11.87M) |
| small | 6 | 512 | 8 | 1536 | 37,368,832 (37.37M) |
| base | 12 | 768 | 12 | 2048 | 110,335,488 (110.34M) |
| base_medium | 18 | 768 | 12 | 2048 | 152,821,248 (152.82M) |
| base_large | 20 | 768 | 12 | 2048 | 166,983,168 (166.98M) |
| medium | 24 | 1024 | 16 | 2816 | 342,198,272 (342.20M) |
| large | 36 | 1280 | 20 | 3584 | 773,839,360 (773.84M) |

Head dim is 64. KV heads match the query heads. SwiGLU's first linear writes `2 * d_ff`. Dropout is 0. Embeddings are untied. Bias is off. `max_seq_len` is 1024.

`train_data` saves a version-2 checkpoint with AdamW moments. `save` without an optimizer writes version 1.

## Usage

```bash
./build.sh amd       # or: ./build.sh opencl, ./build.sh cuda, ./build.sh arc, ./build.sh agx
cmake --build build -j --target train_data

./build/train_data --files books.txt --out run \
  --preset tiny --vocab 1024 --seq-len 256 --batch 4 --steps 1000 \
  --prompt "The morning after"
```

`--preset` is `tiny`, `small`, `base`, `base-medium`, `base-large`, `medium`, or `large`. `--seq-len` must be at most 1024. `--vocab` must be at least 259. The notes below are one GPU, full attention, and no activation checkpointing. They were sized for float32 activations. On nnCUDA, nnCL, and nnROCm, training stores retained activations as bfloat16 and keeps parameters, Adam state, and checkpoints in float32, so the activation working set is about half of these notes. nnArc keeps activations in float32.

`tiny` fits in 12 GB. Suggested batches are `(8, 512)` and `(16, 256)`. `small` is tight in 12 GB at `(4, 512)`; 24 GB is comfortable. `base` is a 24 GB target. On a 12 GB GPU keep `base` at `--seq-len` 512 or below. `base-medium` is 18 layers at the base width, and `base-large` is 20 layers at the base width. `medium` is 24 layers at width 1024, and `large` is 36 layers at width 1280. `base-medium`, `base-large`, `medium`, and `large` are past these notes.

The run writes `tokenizer.bpe`, `tokens.bin`, `tokens.meta`, and `model.bin`, then prints a continuation. Load that directory with `chat`:

```bash
cmake --build build -j --target chat
./build/chat --model run
```

The encoded prompt plus `--max-new-tokens` has to fit in 1024. `chat` drops the oldest `User:` / `Assistant:` turn until it does. `finetune` keeps training the checkpoint on a new file list. Walkthroughs: [train_data](../examples/train_data.md), [chat](../examples/chat.md), [finetune](../examples/finetune.md).

`gguf export` writes an `nnllm` GGUF that `gguf import` reads back. `gguf llama` writes a llama.cpp file and omits the position table and the LayerNorm biases. llama.cpp then applies RoPE and RMSNorm, so those tokens differ from this runtime. The program prints that warning and exits 0.

```cpp
nn::TransformerConfig config = llm::presets::tiny(vocab_size);
nn::Transformer model(ctx, config, seed);
```

`vocab_size` must be at least 1. `small`, `base`, `base_medium`, `base_large`, `medium`, and `large` take the same argument.

## Benefits

- Each token has a learned position row. The checkpoint stores `pos_emb.weight`, and the forward pass adds it.
- `tiny` is the shape `train_tiny` accepts, and it is the 12 GB starting point. `small`, `base`, `medium`, and `large` step up in depth and width. `base-medium` and `base-large` keep the base width at 18 and 20 layers.
- Linears are bias-free and the norms are LayerNorm, so the file stays on the version-2 façade path.
- `chat` and `Generator` sample from the checkpoint. Context is 1024.

## Applications

- A first training run on a book list, followed by the continuation `train_data` prints.
- Dialogue when each file is one `User:` / `Assistant:` exchange and `chat` continues the transcript. [train_chat](../examples/train_chat.md) trains `tiny` on four of those exchanges and samples one reply.
- More text on the same tokenizer through `finetune`.
- An `nnllm` GGUF for another loader in this family.
- A width-matched comparison with `llama-*`, which uses RoPE and RMSNorm at the same depth and width.

## See also

- [preset-llama.md](preset-llama.md) — the same widths with RoPE and RMSNorm
- [preset-qwen.md](preset-qwen.md) — the Qwen3 block
- [preset-gpt2.md](preset-gpt2.md) — the GPT-2 block
- [preset-gpt3.md](preset-gpt3.md) — that dense block at context 2048
- [preset-bert.md](preset-bert.md) — the bidirectional encoder
- [train_chat](../examples/train_chat.md) — train `tiny` beside `bert_tiny`, then one reply and one masked token
- [train_bert_chat](../examples/train_bert_chat.md) — train `bert_tiny` on four exchanges, then one reply
- [README.md](../README.md) — preset table and parameter counts
- [USAGE.md](../USAGE.md) — library notes
