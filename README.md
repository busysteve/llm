# llm

A C++17 library for small decoder-only language models, built beside [`nn`](../nn). This tree is the tokenizer, the token dataset, transformer presets, the trainer, and the generator.

Public headers live under `include/llm/` and do not include device headers. The `llm` target links `nn`. The default checkout is the sibling [`nn`](../nn). Set `NN_SOURCE` to another tree that provides the CMake target `nn`, such as [`nnCL`](../nnCL).

## Requirements

- C++17 and CMake 3.24 or newer
- An `nn` checkout. The default is `../nn`. Device toolkits are that project's requirement.

`./build.sh` configures against `NN_SOURCE`, or `../nn` when that variable is unset. `add_subdirectory(... EXCLUDE_FROM_ALL)` builds `nn` only as a dependency of `llm`, so that project's tests are not part of `all`.

```bash
./build.sh
```

The script configures `build/`, then builds and runs `llm_tests` and `train_tiny`, and builds `train_books`, `chat`, and `gguf`. To build one target after configuring:

```bash
cmake --build build -j --target llm_tests
./build/llm_tests
cmake --build build -j --target train_tiny
./build/train_tiny
cmake --build build -j --target train_books
cmake --build build -j --target chat
cmake --build build -j --target finetune
cmake --build build -j --target gguf
```

`train_tiny` is the acceptance example. It is not part of `llm_tests`. `train_books` trains from a list of text files, writes `tokenizer.bpe` and `model.bin`, then loads those files and generates. `chat` loads that directory and continues a `User:` / `Assistant:` transcript. `finetune` keeps training one of those checkpoints on a new file list. `gguf` converts `model.bin` to an `nnllm` GGUF and back, and can write a llama.cpp GGUF from the same checkpoint. Walkthroughs: [train_tiny](examples/train_tiny.md), [train_books](examples/train_books.md), [chat](examples/chat.md), [finetune](examples/finetune.md), [gguf](examples/gguf.md). Library details are in [USAGE.md](USAGE.md).

## Layout

```
include/llm/tokenizer.hpp   Tokenizer
include/llm/bpe.hpp         BpeTokenizer
include/llm/data.hpp        TextDataset and token files
include/llm/presets.hpp     tiny / small / base and llama configs, AdamW groups
include/llm/train.hpp       learning-rate schedule and Trainer
include/llm/generate.hpp    Generator
src/bpe.cpp                 byte-level BPE, train, and the LPB1 file
src/data.cpp                token file and next-token batches
src/presets.cpp             preset configs and adamw_groups
src/train.cpp               warmup/cosine schedule and the fit loop
src/generate.cpp            prefill, sample, and decode_step
examples/train_tiny.cpp     acceptance train and generate
examples/train_tiny.md      how to run train_tiny
examples/train_books.cpp    train from a list of book files, save, and generate
examples/train_books.md     how to train from a file list
examples/chat.cpp           continue a User/Assistant transcript
examples/chat.md            how to converse with a trained model
examples/finetune.cpp       fine-tune an existing checkpoint on new text
examples/finetune.md        how to fine-tune an existing model
examples/gguf.cpp           convert model.bin to GGUF, and to a llama.cpp GGUF
examples/gguf.md            how to convert a checkpoint to GGUF
tests/test_bpe.cpp          tokenizer tests
tests/test_data.cpp         dataset tests
tests/test_train.cpp        presets, trainer, and generator tests
```

## Tokenizer

`BpeTokenizer::train` learns a byte-level vocabulary. Ids `0..255` are raw bytes. Learned merges start at `256`. The last three ids are pad, bos, and eos, in that order, so `vocab_size = 256 + n_merges + 3`.

`encode` splits the string into bytes, applies merges in the order they were learned, then optionally prepends bos and appends eos. It does not insert pad. `decode` turns byte and merge ids back into bytes. Pad, bos, and eos add no bytes, and an unknown id throws. With bos and eos off, `decode(encode(text))` equals `text`.

Training counts adjacent pairs inside each string, merges the most frequent pair, and repeats until the vocab is full. Equal counts take the smaller `(left, right)` pair, so the same corpus and vocab size train the same merges.

The on-disk form is little-endian with magic `LPB1`. The byte layout is in [USAGE.md](USAGE.md).

## Presets

`llm::presets::tiny`, `small`, and `base` return an `nn::TransformerConfig`. `vocab_size` must be positive. Dropout is 0, bias is off, embeddings are not tied, the activation is SwiGLU, positions are learned absolute, and `max_seq_len` is 1024. `llama_tiny`, `llama_small`, and `llama_base` are the same widths with `positional` set to RoPE, which selects RMSNorm and adjacent-pair RoPE. A learned-position checkpoint cannot be turned into that block. Train a llama preset from scratch.

| Preset | Layers | `d_model` | Heads | `d_ff` | Positions |
|---|---:|---:|---:|---:|---|
| tiny | 4 | 256 | 4 | 768 | learned absolute |
| small | 6 | 512 | 8 | 1536 | learned absolute |
| base | 12 | 768 | 12 | 2048 | learned absolute |
| llama_tiny | 4 | 256 | 4 | 768 | RoPE, RMSNorm |
| llama_small | 6 | 512 | 8 | 1536 | RoPE, RMSNorm |
| llama_base | 12 | 768 | 12 | 2048 | RoPE, RMSNorm |

`llm::adamw_groups` walks `parameters()` in order. Rank-2 tensors whose names do not refer to LayerNorm take the given weight decay. Biases, names containing `ln`, and every other rank-1 tensor take 0. Groups are contiguous runs of that choice, so concatenating them is still `parameters()` order. A NaN or negative `weight_decay` throws `weight_decay`.

Untied parameter counts at vocab 16000 are tiny 11866624 (11.87M), small 37368832 (37.37M), and base 110335488 (110.34M). The llama presets drop the position table and the norm biases: llama_tiny 11602176 (11.60M), llama_small 36837888 (36.84M), and llama_base 109529856 (109.53M). The GPU notes below are float32, one GPU, full attention, and no activation checkpointing.

tiny fits in 12 GB. Suggested batches are `(8, 512)` and `(16, 256)`. small is tight in 12 GB at `(4, 512)`; 24 GB is comfortable. base is a 24 GB target, not a 12 GB target at `T=512`. Keep `T <= 1024` for base, and prefer `T <= 512` on a 12 GB GPU.

`examples/train_tiny.cpp` trains `presets::tiny` at the byte-level BPE vocab, a few million parameters, so that acceptance run fits easily. That preset stays learned absolute. `train_books --preset llama-tiny` is the RoPE and RMSNorm shape.

## Trainer

`Trainer::fit` runs `max_steps` AdamW updates. Each step is 1-based. The learning rate warms up linearly, then follows a cosine down to `min_lr`. `learning_rate_at` is that schedule. Micro-batches accumulate gradients on device (`Tensor::scale` by `1 / grad_accum_steps`) before one clip and one `opt.step`. A non-finite loss or grad norm throws when `abort_on_non_finite` is set. Eval, when a validation set is passed, prints mean cross-entropy and perplexity. A positive `early_stop_patience` stops after that many validation evals with no improvement and restores the best step, including AdamW moments. A finite `min_loss` of at least 0 stops when a step's training loss is at or below that value and keeps that step. Checkpoints use `Transformer::save(path, &adamw)`, so the file is version 2 and includes embeddings and AdamW moments.

## Generator

`Generator::generate` encodes the prompt with no extra bos or eos, prefills the cache, and samples new tokens with `decode_step`. Temperature 0 is greedy. Otherwise logits are divided by temperature, masked to `top_k` (`0` skips top-k), softmaxed, then nucleus-filtered on probabilities (`top_p == 1` skips nucleus), and drawn. An eos id stops the string without being appended. The return value is `decode` of the new ids only.
