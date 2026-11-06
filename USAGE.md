# Usage

Programmer's guide to the tokenizer in namespace `llm`. Headers under `include/llm/` are C++17 and do not include CUDA. Link the CMake target `llm`. It links `nn` publicly, so a program that links `libllm.a` by hand also needs `nn` and the CUDA libraries `nn` was built against.

`train` throws `std::invalid_argument("vocab_size")` when `vocab_size` is below 259, does not fit in `int32`, or the corpus runs out of pairs before that vocab is full. `decode` throws `std::invalid_argument("id")` for an id outside `0 .. vocab_size-1`. `load` throws `std::runtime_error` for a missing file, a bad magic, or a file that does not match the layout below.

## Train, encode, decode

Header: `llm/bpe.hpp`

```cpp
#include "llm/bpe.hpp"

std::unique_ptr<llm::BpeTokenizer> tok = llm::BpeTokenizer::train(
    {"to be or not to be", "that is the question"}, /*vocab_size=*/300);

std::vector<int32_t> ids = tok->encode("to be");
std::string text = tok->decode(ids);  // "to be"

llm::EncodeOptions opt;
opt.add_bos = true;
opt.add_eos = true;
ids = tok->encode("to be", opt);  // bos, pieces, eos
text = tok->decode(ids);          // specials contribute no bytes
```

`vocab_size()`, `pad_id()`, `bos_id()`, and `eos_id()` follow one layout:

| Ids | Meaning |
|---|---|
| `0 .. 255` | Raw bytes of the string |
| `256 .. vocab_size - 4` | Merges, in the order they were learned |
| `vocab_size - 3` | pad |
| `vocab_size - 2` | bos |
| `vocab_size - 1` | eos |

`encode` never writes pad. Bos and eos are written only when `EncodeOptions` asks for them. Merges are applied in learn order, lowest rank first, left to right, and a pair does not overlap itself. The same rules are used while training, on the current segmentation of each string. Pairs are not counted across string boundaries.

`decode(encode(text))` equals `text` when both flags are off, including for empty strings and for bytes that are not UTF-8. Pad, bos, and eos in the middle of an id list are skipped.

## Save and load

```cpp
tok->save("tokenizer.bpe");
std::unique_ptr<llm::Tokenizer> loaded = llm::Tokenizer::load("tokenizer.bpe");
```

`load` returns a `BpeTokenizer` when the magic matches. The file is little-endian. `save` writes a temporary file next to the destination and renames it into place.

| Offset | Type | Field |
|---:|---|---|
| 0 | 4 bytes | Magic `LPB1` |
| 4 | uint32 | `vocab_size` |
| 8 | uint32 | `n_merges` (`vocab_size - 259`) |
| 12 | uint32 | `pad_id` |
| 16 | uint32 | `bos_id` |
| 20 | uint32 | `eos_id` |
| 24 | `n_merges` records | `uint32 left`, `uint32 right`, in learn order |

`left` and `right` are ids that already exist when that merge is learned: a byte, or an earlier merge. They are not pad, bos, or eos.
