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

## What is in the GGUF

The file is little-endian GGUF version 3. `general.architecture` is `nnllm`. `general.alignment` is 32. `general.file_type` is 0, which means float32. The hyperparameters use the `nnllm.` prefix: vocab size, embedding length, block count, feed-forward length, context length, head count, dropout, bias, `tie_embeddings` (always false), activation (`swiglu` or `gelu`), and position encoding (`learned_absolute`).

Tensor names match the checkpoint: `tok_emb.weight`, `pos_emb.weight`, `blocks.N.ln1.weight`, `blocks.N.attn.q.weight`, `blocks.N.fc1.weight`, `final_ln.weight`, `lm_head.weight`, and the bias tensors this model actually has. The bytes stay in this library's row-major order. The GGUF dimension list stores the contiguous dimension first, so a weight stored here as `(out, in)` is recorded as dimensions `(in, out)`.

`import` also accepts `nnllm` tensors stored as float16 and expands them to float32.

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
