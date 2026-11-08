# How to convert a checkpoint to GGUF

`gguf` converts a transformer `model.bin` to a [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) version 3 file and converts that file back. The source is [`gguf.cpp`](gguf.cpp). It does not link `llm` or CUDA, and it is a separate program from `llm_tests`.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target gguf
./build/gguf export run/model.bin run/model.gguf
./build/gguf import run/model.gguf run-again/model.bin
```

`export` reads a family-6 checkpoint written by `train_books`, `finetune`, or `Transformer::save`. It keeps the float32 weights and writes architecture `nnllm`. `import` reads that file and writes a version-1 checkpoint with the weights and no optimizer block. `LoadMode::Inference` accepts that file. A later `finetune` run builds a fresh AdamW from it. `finetune --resume` needs the original checkpoint, because AdamW moments are not in the GGUF.

`tokenizer.bpe` is not inside either file. Keep it next to the checkpoint you train or chat with.

## The nnllm architecture

`nnllm` is the architecture name inside a standard GGUF version 3 file. It is not a second container. `general.architecture` is the string `nnllm`, and the hyperparameters use the `nnllm.` prefix.

A checkpoint from this library is a pre-norm transformer. The norm is LayerNorm. Positions are a learned absolute embedding. Q, K, and V are separate matrices. That is a different block from the `llama` and `gpt2` architectures, so those names would describe the wrong math. GGUF carries the bytes. `nnllm` tells a reader which model they belong to. `llama.cpp` does not implement `nnllm`, so it will not run these files. `gguf import` is the reader that turns one back into `model.bin`.

The file is little-endian. `general.alignment` is 32. `export` writes float32, so `general.file_type` is 0. `import` also accepts `nnllm` tensors stored as float16 (`general.file_type` 1) and expands them to float32.

| Key | Meaning |
|---|---|
| `nnllm.vocab_size` | vocabulary size |
| `nnllm.embedding_length` | `d_model` |
| `nnllm.block_count` | layer count |
| `nnllm.feed_forward_length` | `d_ff`, the width after the SwiGLU gate |
| `nnllm.context_length` | `max_seq_len` |
| `nnllm.attention.head_count` | head count |
| `nnllm.dropout` | dropout stored in the checkpoint |
| `nnllm.bias` | whether the linear layers have a bias |
| `nnllm.tie_embeddings` | always false |
| `nnllm.activation` | `swiglu` or `gelu` |
| `nnllm.positional` | `learned_absolute` |

Integer counts are `uint32`. `nnllm.dropout` is `float32`. The two flags are GGUF bools. Activation and position encoding are strings.

## Tensors

Names match the checkpoint, in parameter order:

| Tensor | Shape |
|---|---|
| `tok_emb.weight` | `(vocab, d_model)` |
| `pos_emb.weight` | `(max_seq_len, d_model)` |
| `blocks.N.ln1.weight`, `blocks.N.ln1.bias` | `(d_model)` |
| `blocks.N.attn.q.weight`, and the same for `k`, `v`, and `out` | `(d_model, d_model)` |
| `blocks.N.ln2.weight`, `blocks.N.ln2.bias` | `(d_model)` |
| `blocks.N.fc1.weight` | `(2 * d_ff, d_model)` for SwiGLU, `(d_ff, d_model)` for GELU |
| `blocks.N.fc2.weight` | `(d_model, d_ff)` |
| `final_ln.weight`, `final_ln.bias` | `(d_model)` |
| `lm_head.weight` | `(vocab, d_model)` |

`N` is the block index starting at 0. LayerNorm always has a weight and a bias. When `nnllm.bias` is true, each linear name also has a `.bias` whose length is the output size. With SwiGLU, `fc1.weight` holds the gate and the up projection in that one matrix. The float values stay in this library's row-major order. The GGUF dimension list stores the contiguous dimension first, so a weight stored here as `(out, in)` is recorded as dimensions `(in, out)`.

The GGUF does not contain `tokenizer.bpe` or the AdamW moments. `import` writes a version-1 checkpoint. `LoadMode::Inference` loads it, and a later `finetune` starts a fresh AdamW. `finetune --resume` needs the original `model.bin`.

## What this tool rejects

This runtime is a float32 pre-norm transformer with LayerNorm and learned absolute positions. A GGUF whose `general.architecture` is `llama`, `gpt2`, or anything other than `nnllm` is rejected with `gguf: architecture`. Quantized tensor types are rejected with `gguf: quantized`. `llama.cpp` does not implement the `nnllm` architecture, so it will not run these files.

## Check the converter

`gguf self-test` builds a small checkpoint, writes GGUF, reads it back, and checks a float16 round trip. It does not need a GPU.

```bash
./build/gguf self-test
```

## When it fails

Errors print `gguf: <reason>` on stderr and the process exits 1.

| Reason | Meaning |
|---|---|
| `magic` | the file is not a checkpoint or a GGUF |
| `version` | the checkpoint version is outside 1..2, or the GGUF version is not 3 |
| `family` | the checkpoint is not a transformer |
| `architecture` | `general.architecture` is not `nnllm` |
| `quantized` | a tensor is neither float32 nor float16 |
| `tensor <name>` | a weight is missing, duplicated, or the wrong shape |
| `tie_embeddings` | the config asks for a tied embedding, which this runtime rejects |
| `positional` | positions are not learned absolute |
| `read` or `write` | the path could not be opened |
