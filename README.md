# llm

A C++17 library for small decoder-only language models, built beside [`nn`](../nn). This tree is the tokenizer, the token dataset, and the CMake scaffold. It does not include a trainer, generator, or transformer presets.

Public headers live under `include/llm/` and do not include CUDA headers. The `llm` target still links `nn`, and `nn` uses CUDA.

## Requirements

- C++17 and CMake 3.24 or newer
- The CUDA Toolkit and cuDNN setup that `nn` needs
- A checkout of `nn` at `../nn`, or pass `-DNN_SOURCE=` to that checkout

`CMAKE_CUDA_ARCHITECTURES` defaults to `86`. `add_subdirectory(... EXCLUDE_FROM_ALL)` builds `nn` only as a dependency of `llm`, so `nn`'s tests are not part of `all`.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j --target llm_tests
./build/llm_tests
```

## Layout

```
include/llm/tokenizer.hpp   Tokenizer
include/llm/bpe.hpp         BpeTokenizer
include/llm/data.hpp        TextDataset and token files
src/bpe.cpp                 byte-level BPE, train, and the LPB1 file
src/data.cpp                token file and next-token batches
tests/test_bpe.cpp          tokenizer tests
tests/test_data.cpp         dataset tests
```

## Tokenizer

`BpeTokenizer::train` learns a byte-level vocabulary. Ids `0..255` are raw bytes. Learned merges start at `256`. The last three ids are pad, bos, and eos, in that order, so `vocab_size = 256 + n_merges + 3`.

`encode` splits the string into bytes, applies merges in the order they were learned, then optionally prepends bos and appends eos. It does not insert pad. `decode` turns byte and merge ids back into bytes. Pad, bos, and eos add no bytes, and an unknown id throws. With bos and eos off, `decode(encode(text))` equals `text`.

Training counts adjacent pairs inside each string, merges the most frequent pair, and repeats until the vocab is full. Equal counts take the smaller `(left, right)` pair, so the same corpus and vocab size train the same merges.

The on-disk form is little-endian with magic `LPB1`. The byte layout is in [USAGE.md](USAGE.md).
