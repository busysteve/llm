# llm

A C++17 library for small decoder-only language models and a BERT encoder, built beside [`nnCUDA`](../nnCUDA). This tree is the tokenizer, the token dataset, transformer presets, the trainer, an optional reinforcement-learning trainer, and the generator.

Public headers live under `include/llm/` and do not include CUDA, HIP, OpenCL, Metal, or SYCL headers. The `llm` target links `nn` and pthread. That library is the CUDA tree [`nnCUDA`](../nnCUDA), the OpenCL tree [`nnCL`](../nnCL), the ROCm tree [`nnROCm`](../nnROCm), the Intel Arc tree [`nnArc`](../nnArc), or the Apple AGX tree [`nnAGX`](../nnAGX).

## Requirements

- C++17 and CMake 3.24 or newer
- One backend checkout next to this tree:
  - CUDA: `../nnCUDA`, plus the CUDA Toolkit and cuDNN that `nn` needs. `CMAKE_CUDA_ARCHITECTURES` defaults to `86`. If the default GCC is newer than 15, `./build.sh` uses `g++-15`.
  - OpenCL: `../nnCL`, plus an OpenCL ICD and a GPU. Do not set `RUSTICL_ENABLE`.
  - ROCm: `../nnROCm`, plus hipBLAS and rocBLAS. `CMAKE_HIP_ARCHITECTURES` defaults to `gfx1152`. The HIP compiler is `amdclang++`, not `hipcc`.
  - Arc: `../nnArc`, plus Intel oneAPI DPC++ (`icpx`) and oneMKL SYCL BLAS. Kernels are JIT SPIR-V. There is no architecture flag. `./build.sh arc` exits if `icpx` is missing.
  - AGX: `../nnAGX`, plus Apple Metal. `./build.sh agx` exits on a non-Apple host. The Mac needs Xcode or the Metal command-line tools.

`./build.sh cuda` configures against `../nnCUDA`. `./build.sh opencl` configures against `../nnCL`. `./build.sh amd` configures against `../nnROCm`. `./build.sh arc` configures against `../nnArc`. `./build.sh agx` configures against `../nnAGX`. With no argument, `NN_SOURCE` is used when it is set; otherwise OpenCL is chosen when `../nnCL` is present and `nvcc` is not. A build directory configured for a different sibling is cleared first. `add_subdirectory(... EXCLUDE_FROM_ALL)` builds `nn` only as a dependency of `llm`, so that project's tests are not part of `all`.

```bash
./build.sh amd
# or: ./build.sh opencl
# or: ./build.sh cuda
# or: ./build.sh arc
# or: ./build.sh agx
```

The script configures `build/`, builds every example and `llm_tests`, then runs `llm_tests` and `train_tiny`. To build one target after configuring:

```bash
cmake --build build -j --target llm_tests
./build/llm_tests
cmake --build build -j --target train_tiny
./build/train_tiny
cmake --build build -j --target train_bench
./build/train_bench
cmake --build build -j --target train_chat
./build/train_chat
cmake --build build -j --target train_bert_chat
./build/train_bert_chat
cmake --build build -j --target train_data
cmake --build build -j --target chat
cmake --build build -j --target finetune
cmake --build build -j --target reinforce
cmake --build build -j --target gguf
cmake --build build -j --target hf
```

`train_tiny` is the acceptance example. It is not part of `llm_tests`. `train_bench` times eight steps of `presets::tiny` after one warmup step and prints tokens per second. It is not part of `llm_tests`. `train_data` trains from a list of text files or from every regular file under a directory, writes `tokenizer.bpe` and `model.bin`, then loads those files and generates. SIGINT saves that checkpoint, `--resume` continues the same run, `--lr-new` with `--resume` starts a new learning-rate schedule at step 1, `--reuse-tokens` starts a new model on the saved tokens, `--tokenize-only` writes those tokens and exits before the model, `--dir` reads every regular file under a directory, and `--stdin` reads one document from standard input. `--layers`, `--d-model`, `--heads`, `--d-ff`, `--max-seq`, `--kv-heads`, and `--head-dim` override the selected preset's sizes. `chat` loads that directory and continues a `User:` / `Assistant:` transcript, or with `--plain` continues the line as a prompt. `finetune` keeps training one of those checkpoints on a new file list. `reinforce` updates a causal checkpoint from a caller-supplied reward. `gguf` converts `model.bin` to an `nnllm` GGUF and back, can write a llama.cpp GGUF from the same checkpoint, and can convert a public llama.cpp GGUF into a `chat` directory. `hf` converts a Hugging Face snapshot to `model.bin`. Walkthroughs: [train_tiny](examples/train_tiny.md), [train_bench](examples/train_bench.md), [train_chat](examples/train_chat.md), [train_bert_chat](examples/train_bert_chat.md), [train_data](examples/train_data.md), [chat](examples/chat.md), [finetune](examples/finetune.md), [reinforce](examples/reinforce.md), [gguf](examples/gguf.md), [hf](examples/hf.md). Library details are in [USAGE.md](USAGE.md). Terms, acronyms, and processes are in [docs/glossary.md](docs/glossary.md). The source layout is in [docs/code-structure.md](docs/code-structure.md).

## Layout

```
include/llm/tokenizer.hpp   Tokenizer
include/llm/chat_template.hpp  ChatWrap for dialogue, plain, and ChatML
include/llm/hf_tokenizer.hpp   Hugging Face tokenizer.json
include/llm/bpe.hpp         BpeTokenizer
include/llm/data.hpp        TextDataset and token files
include/llm/presets.hpp     tiny through large, including base-medium and base-large, plus llama, qwen, gpt2, gpt3, and bert configs, AdamW groups
include/llm/train.hpp       learning-rate schedule and Trainer
include/llm/rl.hpp          policy gradient and RlTrainer
include/llm/generate.hpp    Generator and sample_token
src/bpe.cpp                 byte-level BPE, train, and the LPB1 file
src/hf_tokenizer.cpp        Hugging Face tokenizer.json byte-level BPE
src/data.cpp                token file and next-token batches
src/presets.cpp             preset configs and adamw_groups
src/train.cpp               warmup/cosine schedule and the fit loop
src/rl.cpp                  host policy gradient and the reinforcement-learning fit loop
src/generate.cpp            prefill, sample, decode_step, and a masked forward for an encoder
examples/train_tiny.cpp     acceptance train and generate
examples/train_tiny.md      how to run train_tiny
examples/train_bench.cpp    time eight steps of presets::tiny
examples/train_bench.md     how to run train_bench
examples/train_chat.cpp     train tiny for a reply and bert_tiny for one masked token
examples/train_chat.md      how to run train_chat
examples/train_bert_chat.cpp train bert_tiny on dialogue and print one reply
examples/train_bert_chat.md  how to run train_bert_chat
examples/train_data.cpp     train from a file list or a directory, save, and generate
examples/train_data.md      how to train from a file list or a directory
examples/chat.cpp           continue a User/Assistant transcript, or a plain prompt
examples/chat.md            how to converse with a trained model
examples/finetune.cpp       fine-tune an existing checkpoint on new text
examples/finetune.md        how to fine-tune an existing model
examples/reinforce.cpp      update a causal checkpoint from a string reward
examples/reinforce.md       how to run reinforce
examples/gguf.cpp           convert model.bin to GGUF, to a llama.cpp GGUF, and from a public GGUF
examples/gguf.md            how to convert a checkpoint to GGUF, and a public GGUF to chat files
examples/hf.cpp             convert a Hugging Face snapshot to model.bin
examples/hf.md              how to convert Qwen3 and Llama 2 weights
tests/test_bpe.cpp          tokenizer tests
tests/test_hf_tokenizer.cpp Hugging Face tokenizer and ChatML wrap tests
tests/test_data.cpp         dataset tests
tests/test_train.cpp        presets, trainer, and generator tests
tests/test_rl.cpp           policy gradient and RlTrainer tests
docs/llama-base-architecture.md  Mermaid diagram of llama_base
docs/qwen-base-architecture.md   Mermaid diagram of qwen_base
docs/gpt2-base-architecture.md   Mermaid diagram of gpt2_base
docs/gpt3-base-architecture.md   Mermaid diagram of gpt3_base
docs/bert-base-architecture.md   Mermaid diagram of bert_base
docs/preset-default.md           usage, benefits, and applications for tiny through large
docs/preset-llama.md             usage, benefits, and applications for llama-*
docs/preset-qwen.md              usage, benefits, and applications for qwen-*
docs/preset-gpt2.md              usage, benefits, and applications for gpt2-*
docs/preset-gpt3.md              usage, benefits, and applications for gpt3-*
docs/preset-bert.md              usage, benefits, and applications for bert-*
docs/glossary.md                 terms, acronyms, and processes
docs/code-structure.md           how the sources fit together, and how they call nn
```

## Tokenizer

`BpeTokenizer::train` learns a byte-level vocabulary. Ids `0..255` are raw bytes. Learned merges start at `256`. The last three ids are pad, bos, and eos, in that order, so `vocab_size = 256 + n_merges + 3`.

`encode` splits the string into pieces, then into bytes, applies merges in the order they were learned, and optionally prepends bos and appends eos. A piece is one run of word bytes, one run of whitespace, or one run of punctuation. Word bytes are ASCII letters and digits and every byte at or above 128, so a UTF-8 sequence stays inside one piece. Whitespace is space, tab, and the newline family. Everything else, including punctuation, is its own run. Merges do not cross a piece boundary. `encode` does not insert pad. `decode` turns byte and merge ids back into bytes. Pad, bos, and eos add no bytes, and an unknown id throws. With bos and eos off, `decode(encode(text))` equals `text`.

Training counts adjacent pairs inside each piece, merges the most frequent pair, and repeats until the vocab is full. The first count runs on all CPU threads. Each merge rewrites only the pieces that contain that pair and updates those counts. Equal counts take the smaller `(left, right)` pair, so the same corpus and vocab size train the same merges on one thread or many. A pair that crosses whitespace, punctuation, or two training strings is not a pair. The `LPB1` file does not store a pre-tokenization flag, so a file trained before this split must be trained again.

The on-disk form is little-endian with magic `LPB1`. The byte layout is in [USAGE.md](USAGE.md).

## Presets

`llm::presets::tiny`, `small`, `base`, `base_medium`, `base_large`, `medium`, and `large` return an `nn::TransformerConfig`. `vocab_size` must be positive. Dropout is 0, bias is off, embeddings are not tied, the activation is SwiGLU, positions are learned absolute, and `max_seq_len` is 1024. `llama_tiny`, `llama_small`, `llama_base`, `llama_base_medium`, `llama_base_large`, `llama_medium`, and `llama_large` are the same widths with `positional` set to RoPE, which selects RMSNorm and adjacent-pair RoPE. `qwen_tiny`, `qwen_small`, `qwen_base`, `qwen_base_medium`, `qwen_base_large`, `qwen_medium`, and `qwen_large` are the Qwen3 block: RoPE base `1e6`, RMSNorm epsilon `1e-6`, head dim 128, grouped-query attention, and a QK-norm of width 128 before RoPE. `gpt2_tiny`, `gpt2_small`, `gpt2_base`, `gpt2_base_medium`, `gpt2_base_large`, `gpt2_medium`, and `gpt2_large` are the GPT-2 block at those depths and widths: learned absolute positions, LayerNorm, GELU, `d_ff = 4 * d_model`, and a bias on every linear. `gpt3_tiny`, `gpt3_small`, `gpt3_base`, `gpt3_base_medium`, `gpt3_base_large`, `gpt3_medium`, and `gpt3_large` are that same dense block with `max_seq_len` 2048. `bert_tiny`, `bert_small`, `bert_base`, `bert_base_medium`, `bert_base_large`, `bert_medium`, and `bert_large` are that GPT-2 block with `max_seq_len` 512, bidirectional attention, and a segment table of size 2. Single-sequence training uses segment 0. There is no next-sentence head and no WordPiece tokenizer. `nn` rejects tied embeddings, so the token table and the LM head stay separate. The GPT-3 paper's alternating sparse attention is not in this library. A learned-position checkpoint cannot be turned into a llama or qwen block, and a RoPE checkpoint cannot be turned into a GPT block. A BERT checkpoint generates by appending the pad id and reading that position from `forward`. `decode_step` and `prefill` still throw `causal`. Train each preset from scratch.

| Preset | Layers | `d_model` | Heads | KV heads | Head dim | `d_ff` | Positions | Size |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| tiny | 4 | 256 | 4 | 4 | 64 | 768 | learned absolute | 47.47 MB |
| small | 6 | 512 | 8 | 8 | 64 | 1536 | learned absolute | 149.48 MB |
| base | 12 | 768 | 12 | 12 | 64 | 2048 | learned absolute | 441.34 MB |
| base_medium | 18 | 768 | 12 | 12 | 64 | 2048 | learned absolute | 582.97 MB |
| base_large | 20 | 768 | 12 | 12 | 64 | 2048 | learned absolute | 636.99 MB |
| medium | 24 | 1024 | 16 | 16 | 64 | 2816 | learned absolute | 1368.79 MB |
| large | 36 | 1280 | 20 | 20 | 64 | 3584 | learned absolute | 3095.36 MB |
| llama_tiny | 4 | 256 | 4 | 4 | 64 | 768 | RoPE, RMSNorm | 46.41 MB |
| llama_small | 6 | 512 | 8 | 8 | 64 | 1536 | RoPE, RMSNorm | 147.35 MB |
| llama_base | 12 | 768 | 12 | 12 | 64 | 2048 | RoPE, RMSNorm | 438.12 MB |
| llama_base_medium | 18 | 768 | 12 | 12 | 64 | 2048 | RoPE, RMSNorm | 579.86 MB |
| llama_base_large | 20 | 768 | 12 | 12 | 64 | 2048 | RoPE, RMSNorm | 633.87 MB |
| llama_medium | 24 | 1024 | 16 | 16 | 64 | 2816 | RoPE, RMSNorm | 1364.40 MB |
| llama_large | 36 | 1280 | 20 | 20 | 64 | 3584 | RoPE, RMSNorm | 3089.74 MB |
| qwen_tiny | 4 | 256 | 4 | 2 | 128 | 768 | Qwen3 | 48.51 MB |
| qwen_small | 6 | 512 | 8 | 2 | 128 | 1536 | Qwen3 | 153.65 MB |
| qwen_base | 12 | 768 | 12 | 4 | 128 | 2048 | Qwen3 | 475.88 MB |
| qwen_base_medium | 18 | 768 | 12 | 4 | 128 | 2048 | Qwen3 | 633.88 MB |
| qwen_base_large | 20 | 768 | 12 | 4 | 128 | 2048 | Qwen3 | 693.89 MB |
| qwen_medium | 24 | 1024 | 16 | 4 | 128 | 2816 | Qwen3 | 1465.09 MB |
| qwen_large | 36 | 1280 | 20 | 5 | 128 | 3584 | Qwen3 | 3325.71 MB |
| gpt2_tiny | 4 | 256 | 4 | 4 | 64 | 1024 | GPT-2, context 1024 | 46.52 MB |
| gpt2_small | 6 | 512 | 8 | 8 | 64 | 2048 | GPT-2, context 1024 | 143.36 MB |
| gpt2_base | 12 | 768 | 12 | 12 | 64 | 3072 | GPT-2, context 1024 | 441.74 MB |
| gpt2_base_medium | 18 | 768 | 12 | 12 | 64 | 3072 | GPT-2, context 1024 | 583.50 MB |
| gpt2_base_large | 20 | 768 | 12 | 12 | 64 | 3072 | GPT-2, context 1024 | 637.58 MB |
| gpt2_medium | 24 | 1024 | 16 | 16 | 64 | 4096 | GPT-2, context 1024 | 1344.58 MB |
| gpt2_large | 36 | 1280 | 20 | 20 | 64 | 5120 | GPT-2, context 1024 | 3002.71 MB |
| gpt3_tiny | 4 | 256 | 4 | 4 | 64 | 1024 | GPT-2 block, context 2048 | 47.57 MB |
| gpt3_small | 6 | 512 | 8 | 8 | 64 | 2048 | GPT-2 block, context 2048 | 145.46 MB |
| gpt3_base | 12 | 768 | 12 | 12 | 64 | 3072 | GPT-2 block, context 2048 | 444.88 MB |
| gpt3_base_medium | 18 | 768 | 12 | 12 | 64 | 3072 | GPT-2 block, context 2048 | 586.50 MB |
| gpt3_base_large | 20 | 768 | 12 | 12 | 64 | 3072 | GPT-2 block, context 2048 | 640.58 MB |
| gpt3_medium | 24 | 1024 | 16 | 16 | 64 | 4096 | GPT-2 block, context 2048 | 1348.77 MB |
| gpt3_large | 36 | 1280 | 20 | 20 | 64 | 5120 | GPT-2 block, context 2048 | 3007.95 MB |
| bert_tiny | 4 | 256 | 4 | 4 | 64 | 1024 | BERT, context 512 | 46.00 MB |
| bert_small | 6 | 512 | 8 | 8 | 64 | 2048 | BERT, context 512 | 142.31 MB |
| bert_base | 12 | 768 | 12 | 12 | 64 | 3072 | BERT, context 512 | 440.17 MB |
| bert_base_medium | 18 | 768 | 12 | 12 | 64 | 3072 | BERT, context 512 | 582.01 MB |
| bert_base_large | 20 | 768 | 12 | 12 | 64 | 3072 | BERT, context 512 | 636.08 MB |
| bert_medium | 24 | 1024 | 16 | 16 | 64 | 4096 | BERT, context 512 | 1342.49 MB |
| bert_large | 36 | 1280 | 20 | 20 | 64 | 5120 | BERT, context 512 | 3000.10 MB |

A Mermaid diagram of the `llama_base` architecture (with sizes and specs) is in [docs/llama-base-architecture.md](docs/llama-base-architecture.md). The `qwen_base` diagram is in [docs/qwen-base-architecture.md](docs/qwen-base-architecture.md). The `gpt2_base` diagram is in [docs/gpt2-base-architecture.md](docs/gpt2-base-architecture.md). The `gpt3_base` diagram is in [docs/gpt3-base-architecture.md](docs/gpt3-base-architecture.md). The `bert_base` diagram is in [docs/bert-base-architecture.md](docs/bert-base-architecture.md). Each page also draws the layer and head dimensions and adds them up to the float32 weight footprint in the Size column. Usage, benefits, and applications for each preset type are in [docs/preset-default.md](docs/preset-default.md), [docs/preset-llama.md](docs/preset-llama.md), [docs/preset-qwen.md](docs/preset-qwen.md), [docs/preset-gpt2.md](docs/preset-gpt2.md), [docs/preset-gpt3.md](docs/preset-gpt3.md), and [docs/preset-bert.md](docs/preset-bert.md).

`llm::adamw_groups` walks `parameters()` in order. Rank-2 tensors whose names do not refer to LayerNorm take the given weight decay. Biases, names containing `ln`, and every other rank-1 tensor take 0. Groups are contiguous runs of that choice, so concatenating them is still `parameters()` order. A NaN or negative `weight_decay` throws `weight_decay`.

The Size column is the float32 parameter storage at vocab 16000, four bytes per parameter. Untied parameter counts at vocab 16000 are tiny 11866624 (11.87M), small 37368832 (37.37M), base 110335488 (110.34M), base_medium 152821248 (152.82M), base_large 166983168 (166.98M), medium 342198272 (342.20M), and large 773839360 (773.84M). The llama presets drop the position table and the norm biases: llama_tiny 11602176 (11.60M), llama_small 36837888 (36.84M), llama_base 109529856 (109.53M), llama_base_medium 152006400 (152.01M), llama_base_large 166165248 (166.17M), llama_medium 341099520 (341.10M), and llama_large 772435200 (772.44M). The qwen presets add the wider Q projection and the QK-norm weights: qwen_tiny 12127488 (12.13M), qwen_small 38412288 (38.41M), qwen_base 118970112 (118.97M), qwen_base_medium 166166784 (166.17M), qwen_base_large 181899008 (181.90M), qwen_medium 366271488 (366.27M), and qwen_large 831426816 (831.43M). The GPT-2 presets add linear biases and use a GELU width of `4 * d_model`: gpt2_tiny 11629696 (11.63M), gpt2_small 35839616 (35.84M), gpt2_base 110434432 (110.43M), gpt2_base_medium 152961664 (152.96M), gpt2_base_large 167137408 (167.14M), gpt2_medium 336144000 (336.14M), and gpt2_large 750677120 (750.68M). The GPT-3 presets are that block with a 2048-row position table: gpt3_tiny 11891840 (11.89M), gpt3_small 36363904 (36.36M), gpt3_base 111220864 (111.22M), gpt3_base_medium 153748096 (153.75M), gpt3_base_large 167923840 (167.92M), gpt3_medium 337192576 (337.19M), and gpt3_large 751987840 (751.99M). The BERT presets shorten that position table to 512 rows and add a 2-row segment table: bert_tiny 11499136 (11.50M), bert_small 35578496 (35.58M), bert_base 110042752 (110.04M), bert_base_medium 152569984 (152.57M), bert_base_large 166745728 (166.75M), bert_medium 335621760 (335.62M), and bert_large 750024320 (750.02M). The GPU notes below are one GPU, full attention, and no activation checkpointing. They were sized for float32 activations. On nnCUDA, nnCL, and nnROCm, training stores retained activations as bfloat16 and keeps parameters, Adam state, and checkpoints in float32, so the activation working set is about half of these notes. nnArc keeps activations in float32.

tiny fits in 12 GB. Suggested batches are `(8, 512)` and `(16, 256)`. small is tight in 12 GB at `(4, 512)`; 24 GB is comfortable. base is a 24 GB target, not a 12 GB target at `T=512`. base-medium is 18 layers at the base width, and base-large is 20 layers at the base width. medium is 24 layers at width 1024, and large is 36 layers at width 1280. base-medium, base-large, medium, and large are past these notes. Keep `T <= 1024` for base, and prefer `T <= 512` on a 12 GB GPU. `gpt2_*` is the same memory class as the matching width. `gpt3_*` allows `T` up to 2048; the extra cost is the attention, not the position table. Prefer `T <= 512` for `gpt3_base` on a 12 GB GPU. `bert_*` is the same memory class as the matching `gpt2_*` width, and its context is already 512.

`examples/train_tiny.cpp` trains `presets::tiny` at the byte-level BPE vocab, a few million parameters, so that acceptance run fits easily. That preset stays learned absolute. `train_data --preset llama-tiny` is the RoPE and RMSNorm shape. `train_data --preset qwen-tiny` is the Qwen3 shape. `train_data --preset gpt2-tiny` is the GPT-2 block, and `--preset gpt3-tiny` is that block with context 2048. `train_data --preset bert-tiny` trains masked tokens on the bidirectional block. The pad id is the mask. `chat` continues a transcript from that checkpoint by scoring the pad id. `examples/train_bert_chat.cpp` trains `bert_tiny` on four exchanges, saves the pair `chat` loads, and prints one reply. `examples/train_chat.cpp` trains `tiny` and `bert_tiny` on one dialogue tokenizer, samples one reply from `tiny`, and scores one masked token on `bert_tiny`. A public Qwen2 or Qwen3 GGUF loads through `gguf from` and `chat`. Pre `qwen2` or `qwen35` keeps the GGUF token ids and writes `tokenizer.json`. `chat --template qwen` wraps that directory as ChatML. Other pre names are still remapped to LPB1. Architecture `qwen35` is rejected. Qwen2 QKV bias is copied as stored and the checkpoint is version 7. `hf import` reads a Hugging Face `qwen3`, `qwen2`, or Llama 2 `llama` snapshot and writes `model.bin`. Q and K are interleaved so this library's adjacent-pair RoPE matches the published rotate-half weights. A Qwen2 file with `attention_bias` is version 7 and keeps the three QKV biases. Llama 3 RoPE scaling, Mistral, GPT-2, and BERT are rejected until those graphs land. Tokenizer files are copied. `chat` loads `tokenizer.json` when `tokenizer.bpe` is absent and keeps the Hugging Face ids. A Llama 2 directory still needs `tokenizer.model`, which `chat` does not read yet. `gguf llama` rejects GELU, linear bias, bidirectional attention, and a segment table, so a gpt2, gpt3, or bert checkpoint stays in `model.bin` or an nnllm GGUF. It is not a llama.cpp file, and `gguf from` does not load a public GPT-2 or BERT GGUF.

## Trainer

`Trainer::fit` runs `max_steps` AdamW updates. Each step is 1-based. The learning rate warms up linearly, then follows a cosine down to `min_lr`. `learning_rate_at` is that schedule. Micro-batches accumulate gradients on device (`Tensor::scale` by `1 / grad_accum_steps`) before one clip and one `opt.step`. A non-finite loss or grad norm throws when `abort_on_non_finite` is set. Eval, when a validation set is passed, prints mean cross-entropy and perplexity. A positive `early_stop_patience` stops after that many validation evals with no improvement and restores the best step, including AdamW moments. A finite `min_loss` of at least 0 stops when a step's training loss is at or below that value and keeps that step. Checkpoints use `Transformer::save(path, &adamw)`, so the file is version 2, version 4 for a Qwen-style block, version 6 for a bidirectional block or a segment table, or version 8 for a projection-bias block, and includes embeddings and AdamW moments. When `masked_lm` is set, each step picks positions with probability `mask_prob` (default 0.15). Of those, 80% become `mask_token`, 10% become a random id, and 10% stay. The loss is the original id at the chosen positions and ignore index `-100` elsewhere.

`RlTrainer::fit` is a separate loop for a causal decoder. It samples a group of completions, scores them with a caller-supplied reward, and backpropagates a host policy gradient through `Transformer::backward`. `group_size` 1 uses the raw reward. A larger group uses reward minus the group mean. A positive `kl_coef` adds the full-vocabulary KL against a frozen reference model the caller loads and does not train. Prompt tokens and padding stay on ignore index `-100`. The learning rate is `learning_rate_at` on a `TrainConfig` filled with the schedule fields. `Trainer::fit` is unchanged. `examples/reinforce.cpp` is that loop with a reward that is 1 when the completion contains a string.

## Generator

`sample_token` draws one id from a logit row. `Generator::generate` calls it. `generate` encodes the prompt with `add_bos_default()` and `add_eos_default()`. Both are false for an `LPB1` tokenizer. A causal model prefills the cache and samples new tokens with `decode_step`. A bidirectional model appends the tokenizer pad id, which is the mask, and reads that position from `forward`. Segment ids stay 0. Temperature 0 is greedy. `repetition_penalty` defaults to 1, which leaves the logits unchanged. A value above 1 is applied before temperature, including the greedy pick, once per distinct id in the prompt and the ids already sampled. A positive logit is divided by the penalty. Any other logit is multiplied by it. A temperature above 0 then divides the logits, masks them to `top_k` (`0` skips top-k), softmaxes, nucleus-filters on probabilities (`top_p == 1` skips nucleus), and draws. Sampling stops when the id is in `stop_ids`. An empty `stop_ids` stops on `eos_id` when that id is non-negative. The stopping id is not appended. The return value is `decode` of the new ids only.
