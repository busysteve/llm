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

## Token file

Header: `llm/data.hpp`

`tokens.bin` is a raw little-endian `int32` stream. There is no header inside the bin. The sidecar meta file is UTF-8 text, two lines: `numel`, then `vocab_size`. `write_token_file` ends each line with `\n`.

```cpp
#include "llm/data.hpp"

llm::write_token_file(tokens_bin, meta_path, ids, vocab_size);
llm::TextDataset data(tokens_bin, meta_path);
```

`write_token_file` throws `std::invalid_argument("vocab_size")` when `vocab_size < 1`, and `std::invalid_argument("id")` when an id is negative or `>= vocab_size`. Those checks happen before either file is opened, so a rejected call leaves an existing pair untouched.

`TextDataset` reads both files into host memory. `numel()` and `vocab_size()` return the meta values. The constructor throws `std::runtime_error("token meta")` for a missing meta file or a file that is not those two integers (each `>= 0` for `numel`, `vocab_size` in `1 .. 2^31-1`, and no third line). It throws `std::runtime_error("token read")` for a missing bin, a short read, or a byte length that is not a multiple of 4. It throws `std::runtime_error("numel")` when that length is not `numel` int32s, and `std::runtime_error("vocab_size")` when a stored id is negative or `>= vocab_size`.

## next()

```cpp
nn::Context ctx;
std::mt19937 rng(1);
llm::Batch batch = data.next(ctx, /*batch_size=*/4, /*seq_len=*/16, rng);
```

`batch.input_ids` and `batch.labels` are `nn::IndexTensor` values of shape `(batch_size, seq_len)` and dtype int64. Storage is contiguous and row-major: element `(n, t)` is at `n * seq_len + t`.

`next` throws `std::invalid_argument("batch_size")` when `batch_size < 1`, `std::invalid_argument("seq_len")` when `seq_len < 1`, and `std::invalid_argument("short corpus")` when `numel < seq_len + 1`.

One example needs `seq_len + 1` tokens. A start `s` is in `[0, numel - seq_len)`. The input row is `tokens[s, s + seq_len)` and the label row is `tokens[s + 1, s + seq_len + 1)`. Every label is a real next token. Nothing is written as an ignore index, and there is no padding mask.

Draw order, so a test can recover `s`: each `next` call builds one `std::uniform_int_distribution<int64_t>(0, numel - seq_len - 1)` and draws once per row, row 0 first. Rows are independent. Copy the `std::mt19937` before the call and replay that same distribution to get each start. Each id is widened with `int64_t{id}` and uploaded with `IndexTensor::copy_from_host`. `next` synchronizes the context before it returns, so the device values are ready.
