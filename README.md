# llm

A C++17 library for small decoder-only language models and a BERT encoder, built beside [`nnCUDA`](../nnCUDA). This tree is the tokenizer, the token dataset, transformer presets, the trainer, and the generator.

Public headers live under `include/llm/` and do not include CUDA, HIP, OpenCL, or SYCL headers. The `llm` target links `nn` and pthread. That library is the CUDA tree [`nnCUDA`](../nnCUDA), the OpenCL tree [`nnCL`](../nnCL), the ROCm tree [`nnROCm`](../nnROCm), or the Intel Arc tree [`nnArc`](../nnArc).

## Requirements

- C++17 and CMake 3.24 or newer
- One backend checkout next to this tree:
  - CUDA: `../nnCUDA`, plus the CUDA Toolkit and cuDNN that `nn` needs. `CMAKE_CUDA_ARCHITECTURES` defaults to `86`. If the default GCC is newer than 15, `./build.sh` uses `g++-15`.
  - OpenCL: `../nnCL`, plus an OpenCL ICD and a GPU. Do not set `RUSTICL_ENABLE`.
  - ROCm: `../nnROCm`, plus hipBLAS and rocBLAS. `CMAKE_HIP_ARCHITECTURES` defaults to `gfx1152`. The HIP compiler is `amdclang++`, not `hipcc`.
  - Arc: `../nnArc`, plus Intel oneAPI DPC++ (`icpx`) and oneMKL SYCL BLAS. Kernels are JIT SPIR-V. There is no architecture flag. `./build.sh arc` exits if `icpx` is missing.

`./build.sh cuda` configures against `../nnCUDA`. `./build.sh opencl` configures against `../nnCL`. `./build.sh amd` configures against `../nnROCm`. `./build.sh arc` configures against `../nnArc`. With no argument, `NN_SOURCE` is used when it is set; otherwise OpenCL is chosen when `../nnCL` is present and `nvcc` is not. A build directory configured for a different sibling is cleared first. `add_subdirectory(... EXCLUDE_FROM_ALL)` builds `nn` only as a dependency of `llm`, so that project's tests are not part of `all`.

```bash
./build.sh amd
# or: ./build.sh opencl
# or: ./build.sh cuda
# or: ./build.sh arc
```

The script configures `build/`, then builds and runs `llm_tests` and `train_tiny`, and builds `train_books`, `chat`, and `gguf`. To build one target after configuring:

```bash
cmake --build build -j --target llm_tests
./build/llm_tests
cmake --build build -j --target train_tiny
./build/train_tiny
cmake --build build -j --target train_chat
./build/train_chat
cmake --build build -j --target train_bert_chat
./build/train_bert_chat
cmake --build build -j --target train_books
cmake --build build -j --target chat
cmake --build build -j --target finetune
cmake --build build -j --target gguf
```

`train_tiny` is the acceptance example. It is not part of `llm_tests`. `train_books` trains from a list of text files, writes `tokenizer.bpe` and `model.bin`, then loads those files and generates. SIGINT saves that checkpoint, `--resume` continues the same run, and `--reuse-tokens` starts a new model on the saved tokens. `chat` loads that directory and continues a `User:` / `Assistant:` transcript, or with `--plain` continues the line as a prompt. `finetune` keeps training one of those checkpoints on a new file list. `gguf` converts `model.bin` to an `nnllm` GGUF and back, can write a llama.cpp GGUF from the same checkpoint, and can convert a public llama.cpp GGUF into a `chat` directory. Walkthroughs: [train_tiny](examples/train_tiny.md), [train_chat](examples/train_chat.md), [train_bert_chat](examples/train_bert_chat.md), [train_books](examples/train_books.md), [chat](examples/chat.md), [finetune](examples/finetune.md), [gguf](examples/gguf.md). Library details are in [USAGE.md](USAGE.md).

## Layout

```
include/llm/tokenizer.hpp   Tokenizer
include/llm/bpe.hpp         BpeTokenizer
include/llm/data.hpp        TextDataset and token files
include/llm/presets.hpp     tiny / small / base, llama, qwen, gpt2, gpt3, and bert configs, AdamW groups
include/llm/train.hpp       learning-rate schedule and Trainer
include/llm/generate.hpp    Generator
src/bpe.cpp                 byte-level BPE, train, and the LPB1 file
src/data.cpp                token file and next-token batches
src/presets.cpp             preset configs and adamw_groups
src/train.cpp               warmup/cosine schedule and the fit loop
src/generate.cpp            prefill, sample, decode_step, and a masked forward for an encoder
examples/train_tiny.cpp     acceptance train and generate
examples/train_tiny.md      how to run train_tiny
examples/train_chat.cpp     train tiny for a reply and bert_tiny for one masked token
examples/train_chat.md      how to run train_chat
examples/train_bert_chat.cpp train bert_tiny on dialogue and print one reply
examples/train_bert_chat.md  how to run train_bert_chat
examples/train_books.cpp    train from a list of book files, save, and generate
examples/train_books.md     how to train from a file list
examples/chat.cpp           continue a User/Assistant transcript, or a plain prompt
examples/chat.md            how to converse with a trained model
examples/finetune.cpp       fine-tune an existing checkpoint on new text
examples/finetune.md        how to fine-tune an existing model
examples/gguf.cpp           convert model.bin to GGUF, to a llama.cpp GGUF, and from a public GGUF
examples/gguf.md            how to convert a checkpoint to GGUF, and a public GGUF to chat files
tests/test_bpe.cpp          tokenizer tests
tests/test_data.cpp         dataset tests
tests/test_train.cpp        presets, trainer, and generator tests
docs/llama-base-architecture.md  Mermaid diagram of llama_base
docs/qwen-base-architecture.md   Mermaid diagram of qwen_base
docs/gpt2-base-architecture.md   Mermaid diagram of gpt2_base
docs/gpt3-base-architecture.md   Mermaid diagram of gpt3_base
docs/bert-base-architecture.md   Mermaid diagram of bert_base
docs/preset-default.md           usage, benefits, and applications for tiny, small, and base
docs/preset-llama.md             usage, benefits, and applications for llama-*
docs/preset-qwen.md              usage, benefits, and applications for qwen-*
docs/preset-gpt2.md              usage, benefits, and applications for gpt2-*
docs/preset-gpt3.md              usage, benefits, and applications for gpt3-*
docs/preset-bert.md              usage, benefits, and applications for bert-*
```

## Tokenizer

`BpeTokenizer::train` learns a byte-level vocabulary. Ids `0..255` are raw bytes. Learned merges start at `256`. The last three ids are pad, bos, and eos, in that order, so `vocab_size = 256 + n_merges + 3`.

`encode` splits the string into pieces, then into bytes, applies merges in the order they were learned, and optionally prepends bos and appends eos. A piece is one run of word bytes, one run of whitespace, or one run of punctuation. Word bytes are ASCII letters and digits and every byte at or above 128, so a UTF-8 sequence stays inside one piece. Whitespace is space, tab, and the newline family. Everything else, including punctuation, is its own run. Merges do not cross a piece boundary. `encode` does not insert pad. `decode` turns byte and merge ids back into bytes. Pad, bos, and eos add no bytes, and an unknown id throws. With bos and eos off, `decode(encode(text))` equals `text`.

Training counts adjacent pairs inside each piece, merges the most frequent pair, and repeats until the vocab is full. Pair counts and each merge rewrite run on all CPU threads. Equal counts take the smaller `(left, right)` pair, so the same corpus and vocab size train the same merges on one thread or many. A pair that crosses whitespace, punctuation, or two training strings is not a pair. The `LPB1` file does not store a pre-tokenization flag, so a file trained before this split must be trained again.

The on-disk form is little-endian with magic `LPB1`. The byte layout is in [USAGE.md](USAGE.md).

## Presets

`llm::presets::tiny`, `small`, and `base` return an `nn::TransformerConfig`. `vocab_size` must be positive. Dropout is 0, bias is off, embeddings are not tied, the activation is SwiGLU, positions are learned absolute, and `max_seq_len` is 1024. `llama_tiny`, `llama_small`, and `llama_base` are the same widths with `positional` set to RoPE, which selects RMSNorm and adjacent-pair RoPE. `qwen_tiny`, `qwen_small`, and `qwen_base` are the Qwen3 block: RoPE base `1e6`, RMSNorm epsilon `1e-6`, head dim 128, grouped-query attention, and a QK-norm of width 128 before RoPE. `gpt2_tiny`, `gpt2_small`, and `gpt2_base` are the GPT-2 block at those depths and widths: learned absolute positions, LayerNorm, GELU, `d_ff = 4 * d_model`, and a bias on every linear. `gpt3_tiny`, `gpt3_small`, and `gpt3_base` are that same dense block with `max_seq_len` 2048. `bert_tiny`, `bert_small`, and `bert_base` are that GPT-2 block with `max_seq_len` 512, bidirectional attention, and a segment table of size 2. Single-sequence training uses segment 0. There is no next-sentence head and no WordPiece tokenizer. `nn` rejects tied embeddings, so the token table and the LM head stay separate. The GPT-3 paper's alternating sparse attention is not in this library. A learned-position checkpoint cannot be turned into a llama or qwen block, and a RoPE checkpoint cannot be turned into a GPT block. A BERT checkpoint generates by appending the pad id and reading that position from `forward`. `decode_step` and `prefill` still throw `causal`. Train each preset from scratch.

| Preset | Layers | `d_model` | Heads | KV heads | Head dim | `d_ff` | Positions |
|---|---:|---:|---:|---:|---:|---:|---|
| tiny | 4 | 256 | 4 | 4 | 64 | 768 | learned absolute |
| small | 6 | 512 | 8 | 8 | 64 | 1536 | learned absolute |
| base | 12 | 768 | 12 | 12 | 64 | 2048 | learned absolute |
| llama_tiny | 4 | 256 | 4 | 4 | 64 | 768 | RoPE, RMSNorm |
| llama_small | 6 | 512 | 8 | 8 | 64 | 1536 | RoPE, RMSNorm |
| llama_base | 12 | 768 | 12 | 12 | 64 | 2048 | RoPE, RMSNorm |
| qwen_tiny | 4 | 256 | 4 | 2 | 128 | 768 | Qwen3 |
| qwen_small | 6 | 512 | 8 | 2 | 128 | 1536 | Qwen3 |
| qwen_base | 12 | 768 | 12 | 4 | 128 | 2048 | Qwen3 |
| gpt2_tiny | 4 | 256 | 4 | 4 | 64 | 1024 | GPT-2, context 1024 |
| gpt2_small | 6 | 512 | 8 | 8 | 64 | 2048 | GPT-2, context 1024 |
| gpt2_base | 12 | 768 | 12 | 12 | 64 | 3072 | GPT-2, context 1024 |
| gpt3_tiny | 4 | 256 | 4 | 4 | 64 | 1024 | GPT-2 block, context 2048 |
| gpt3_small | 6 | 512 | 8 | 8 | 64 | 2048 | GPT-2 block, context 2048 |
| gpt3_base | 12 | 768 | 12 | 12 | 64 | 3072 | GPT-2 block, context 2048 |
| bert_tiny | 4 | 256 | 4 | 4 | 64 | 1024 | BERT, context 512 |
| bert_small | 6 | 512 | 8 | 8 | 64 | 2048 | BERT, context 512 |
| bert_base | 12 | 768 | 12 | 12 | 64 | 3072 | BERT, context 512 |

A Mermaid diagram of the `llama_base` architecture (with sizes and specs) is in [docs/llama-base-architecture.md](docs/llama-base-architecture.md). The `qwen_base` diagram is in [docs/qwen-base-architecture.md](docs/qwen-base-architecture.md). The `gpt2_base` diagram is in [docs/gpt2-base-architecture.md](docs/gpt2-base-architecture.md). The `gpt3_base` diagram is in [docs/gpt3-base-architecture.md](docs/gpt3-base-architecture.md). The `bert_base` diagram is in [docs/bert-base-architecture.md](docs/bert-base-architecture.md). Usage, benefits, and applications for each preset type are in [docs/preset-default.md](docs/preset-default.md), [docs/preset-llama.md](docs/preset-llama.md), [docs/preset-qwen.md](docs/preset-qwen.md), [docs/preset-gpt2.md](docs/preset-gpt2.md), [docs/preset-gpt3.md](docs/preset-gpt3.md), and [docs/preset-bert.md](docs/preset-bert.md).

`llm::adamw_groups` walks `parameters()` in order. Rank-2 tensors whose names do not refer to LayerNorm take the given weight decay. Biases, names containing `ln`, and every other rank-1 tensor take 0. Groups are contiguous runs of that choice, so concatenating them is still `parameters()` order. A NaN or negative `weight_decay` throws `weight_decay`.

Untied parameter counts at vocab 16000 are tiny 11866624 (11.87M), small 37368832 (37.37M), and base 110335488 (110.34M). The llama presets drop the position table and the norm biases: llama_tiny 11602176 (11.60M), llama_small 36837888 (36.84M), and llama_base 109529856 (109.53M). The qwen presets add the wider Q projection and the QK-norm weights: qwen_tiny 12127488 (12.13M), qwen_small 38412288 (38.41M), and qwen_base 118970112 (118.97M). The GPT-2 presets add linear biases and use a GELU width of `4 * d_model`: gpt2_tiny 11629696 (11.63M), gpt2_small 35839616 (35.84M), and gpt2_base 110434432 (110.43M). The GPT-3 presets are that block with a 2048-row position table: gpt3_tiny 11891840 (11.89M), gpt3_small 36363904 (36.36M), and gpt3_base 111220864 (111.22M). The BERT presets shorten that position table to 512 rows and add a 2-row segment table: bert_tiny 11499136 (11.50M), bert_small 35578496 (35.58M), and bert_base 110042752 (110.04M). The GPU notes below are float32, one GPU, full attention, and no activation checkpointing.

tiny fits in 12 GB. Suggested batches are `(8, 512)` and `(16, 256)`. small is tight in 12 GB at `(4, 512)`; 24 GB is comfortable. base is a 24 GB target, not a 12 GB target at `T=512`. Keep `T <= 1024` for base, and prefer `T <= 512` on a 12 GB GPU. `gpt2_*` is the same memory class as the matching width. `gpt3_*` allows `T` up to 2048; the extra cost is the attention, not the position table. Prefer `T <= 512` for `gpt3_base` on a 12 GB GPU. `bert_*` is the same memory class as the matching `gpt2_*` width, and its context is already 512.

`examples/train_tiny.cpp` trains `presets::tiny` at the byte-level BPE vocab, a few million parameters, so that acceptance run fits easily. That preset stays learned absolute. `train_books --preset llama-tiny` is the RoPE and RMSNorm shape. `train_books --preset qwen-tiny` is the Qwen3 shape. `train_books --preset gpt2-tiny` is the GPT-2 block, and `--preset gpt3-tiny` is that block with context 2048. `train_books --preset bert-tiny` trains masked tokens on the bidirectional block. The pad id is the mask. `chat` continues a transcript from that checkpoint by scoring the pad id. `examples/train_bert_chat.cpp` trains `bert_tiny` on four exchanges, saves the pair `chat` loads, and prints one reply. `examples/train_chat.cpp` trains `tiny` and `bert_tiny` on one dialogue tokenizer, samples one reply from `tiny`, and scores one masked token on `bert_tiny`. A public Qwen2 or Qwen3 GGUF loads through `gguf from` and `chat`. `gguf llama` rejects GELU, linear bias, bidirectional attention, and a segment table, so a gpt2, gpt3, or bert checkpoint stays in `model.bin` or an nnllm GGUF. It is not a llama.cpp file, and `gguf from` does not load a public GPT-2 or BERT GGUF.

## Trainer

`Trainer::fit` runs `max_steps` AdamW updates. Each step is 1-based. The learning rate warms up linearly, then follows a cosine down to `min_lr`. `learning_rate_at` is that schedule. Micro-batches accumulate gradients on device (`Tensor::scale` by `1 / grad_accum_steps`) before one clip and one `opt.step`. A non-finite loss or grad norm throws when `abort_on_non_finite` is set. Eval, when a validation set is passed, prints mean cross-entropy and perplexity. A positive `early_stop_patience` stops after that many validation evals with no improvement and restores the best step, including AdamW moments. A finite `min_loss` of at least 0 stops when a step's training loss is at or below that value and keeps that step. Checkpoints use `Transformer::save(path, &adamw)`, so the file is version 2, version 4 for a Qwen-style block, or version 6 for a bidirectional block or a segment table, and includes embeddings and AdamW moments. When `masked_lm` is set, each step picks positions with probability `mask_prob` (default 0.15). Of those, 80% become `mask_token`, 10% become a random id, and 10% stay. The loss is the original id at the chosen positions and ignore index `-100` elsewhere.

## Generator

`Generator::generate` encodes the prompt with no extra bos or eos. A causal model prefills the cache and samples new tokens with `decode_step`. A bidirectional model appends the tokenizer pad id, which is the mask, and reads that position from `forward`. Segment ids stay 0. Temperature 0 is greedy. Otherwise logits are divided by temperature, masked to `top_k` (`0` skips top-k), softmaxed, then nucleus-filtered on probabilities (`top_p == 1` skips nucleus), and drawn. An eos id stops the string without being appended. The return value is `decode` of the new ids only.
