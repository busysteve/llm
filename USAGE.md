# Usage

Programmer's guide to the tokenizer in namespace `llm`. Headers under `include/llm/` are C++17 and do not include CUDA, HIP, or OpenCL. Link the CMake target `llm`. It links `nn` and pthread publicly, so a program that links `libllm.a` by hand also needs `nn`, pthread, and the backend libraries that tree was built against (CUDA and cuDNN, OpenCL, or hipBLAS and rocBLAS).

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

`encode` never writes pad. Bos and eos are written only when `EncodeOptions` asks for them. Merges are applied in learn order, lowest rank first, left to right, and a pair does not overlap itself. The same rules are used while training. Pre-tokenization splits each string into maximal runs of one class before any merge: word bytes (ASCII letters, ASCII digits, and bytes 128..255), whitespace (` `, `\t`, `\n`, `\r`, `\v`, `\f`), or anything else (punctuation and the remaining control bytes). Pairs are not counted across a run, and they are not counted across string boundaries. Concatenating the runs restores the original bytes. The `LPB1` file has no flag for this split. A file trained before it must be trained again; `load` still accepts the old layout, but `encode` now splits.

`train` counts pairs and applies each merge across pieces on `std::thread::hardware_concurrency()` workers. Pass a positive `n_threads` to pick the count; `0` or a negative value uses the hardware count. Pair counts are summed across workers, so the learned merges depend only on the corpus and `vocab_size`. `encode` splits a long string across those workers by piece.

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

`TextDataset` reads both files into host memory. `numel()` and `vocab_size()` return the meta values. `ids()` returns those host ids. The constructor throws `std::runtime_error("token meta")` for a missing meta file or a file that is not those two integers (each `>= 0` for `numel`, `vocab_size` in `1 .. 2^31-1`, and no third line). It throws `std::runtime_error("token read")` for a missing bin, a short read, or a byte length that is not a multiple of 4. It throws `std::runtime_error("numel")` when that length is not `numel` int32s, and `std::runtime_error("vocab_size")` when a stored id is negative or `>= vocab_size`.

`TextDataset(ids, vocab_size)` takes host ids and does not touch the files. It throws `std::invalid_argument("vocab_size")` when `vocab_size < 1`, and `std::invalid_argument("id")` when an id is negative or `>= vocab_size`. `numel()` is `ids.size()`.

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

## Presets and AdamW groups

Header: `llm/presets.hpp`

```cpp
nn::TransformerConfig config = llm::presets::tiny(vocab_size);
nn::Transformer model(ctx, config, /*seed=*/1);
std::vector<nn::AdamWGroup> groups = llm::adamw_groups(model, /*weight_decay=*/0.1f);
nn::AdamW opt(std::move(groups), 3e-4f, 0.9f, 0.95f, 1e-8f);
```

Usage, benefits, and applications for each preset type are in [docs/preset-default.md](docs/preset-default.md), [docs/preset-llama.md](docs/preset-llama.md), [docs/preset-qwen.md](docs/preset-qwen.md), [docs/preset-gpt2.md](docs/preset-gpt2.md), [docs/preset-gpt3.md](docs/preset-gpt3.md), and [docs/preset-bert.md](docs/preset-bert.md).

`tiny`, `small`, `base`, `llama_tiny`, `llama_small`, `llama_base`, `qwen_tiny`, `qwen_small`, `qwen_base`, `gpt2_tiny`, `gpt2_small`, `gpt2_base`, `gpt3_tiny`, `gpt3_small`, `gpt3_base`, `bert_tiny`, `bert_small`, and `bert_base` throw `std::invalid_argument("vocab_size")` when `vocab_size < 1`. Dropout is `0` and `tie_embeddings` is false on every preset. `nn` rejects tied embeddings. The first three use learned-absolute positions, LayerNorm, bias off, SwiGLU, and `max_seq_len` 1024. The `llama_*` presets set `positional` to RoPE, which selects RMSNorm and adjacent-pair RoPE on Q and K. The `qwen_*` presets use that RoPE block with head dim 128, fewer KV heads than query heads, QK-norm, RoPE base `1e6`, and RMSNorm epsilon `1e-6`. The `gpt2_*` presets are the GPT-2 block: learned absolute positions, LayerNorm epsilon `1e-5`, GELU (the erf form, not the tanh approximation), `d_ff = 4 * d_model`, bias on every linear, and `max_seq_len` 1024. The `gpt3_*` presets are that dense block with `max_seq_len` 2048. The GPT-3 paper's alternating dense and locally banded sparse attention is not implemented. The `bert_*` presets are that GPT-2 block with `max_seq_len` 512, `causal` false, and `type_vocab_size` 2. `forward(tokens)` fills every segment id with 0. `forward(tokens, type_ids)` supplies the ids. There is no next-sentence head. A learned-position checkpoint does not become a llama or qwen block, and a RoPE checkpoint does not become a GPT block. A BERT checkpoint generates by appending the pad id and reading that position from `forward`. `decode_step` and `prefill` still throw `causal`.

| Preset | `num_layers` | `d_model` | `num_heads` | KV heads | Head dim | `d_ff` | Positions |
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

Untied parameter counts at vocab 16000 are tiny 11866624 (11.87M), small 37368832 (37.37M), and base 110335488 (110.34M). The llama presets are llama_tiny 11602176 (11.60M), llama_small 36837888 (36.84M), and llama_base 109529856 (109.53M). The qwen presets are qwen_tiny 12127488 (12.13M), qwen_small 38412288 (38.41M), and qwen_base 118970112 (118.97M). The GPT-2 presets are gpt2_tiny 11629696 (11.63M), gpt2_small 35839616 (35.84M), and gpt2_base 110434432 (110.43M). The GPT-3 presets are gpt3_tiny 11891840 (11.89M), gpt3_small 36363904 (36.36M), and gpt3_base 111220864 (111.22M). The BERT presets are bert_tiny 11499136 (11.50M), bert_small 35578496 (35.58M), and bert_base 110042752 (110.04M). The GPU notes below are one GPU, full attention, and no activation checkpointing. They were sized for float32 activations. Training stores retained activations as bfloat16 and keeps parameters, Adam state, and checkpoints in float32, so the activation working set is about half of these notes.

tiny fits in 12 GB. Suggested batches are `(8, 512)` and `(16, 256)`. small is tight in 12 GB at `(4, 512)`; 24 GB is comfortable. base is a 24 GB target, not a 12 GB target at `T=512`. Keep `T <= 1024` for base, and prefer `T <= 512` on a 12 GB GPU. `gpt2_*` matches that memory class. `gpt3_*` allows `T` up to 2048. Prefer `T <= 512` for `gpt3_base` on a 12 GB GPU. The extra cost is attention, not the longer position table. `bert_*` matches the `gpt2_*` memory class, and its context is 512.

`examples/train_tiny.cpp` trains `presets::tiny` at the byte-level BPE vocab, a few million parameters, so that acceptance run fits easily. That preset stays learned absolute. The walkthrough is [examples/train_tiny.md](examples/train_tiny.md).

`examples/train_chat.cpp` trains `presets::tiny` and `presets::bert_tiny` on one byte-level BPE tokenizer and four `User:` / `Assistant:` exchanges. It samples one reply from the tiny model and scores one masked token on the BERT model. Each model has its own weights. The walkthrough is [examples/train_chat.md](examples/train_chat.md).

`examples/train_bert_chat.cpp` trains `presets::bert_tiny` on those same four exchanges with masked tokens. It writes `tokenizer.bpe` and a version-6 `model.bin`, loads them with `LoadMode::Inference`, and prints one reply. The mask id is the pad id. The walkthrough is [examples/train_bert_chat.md](examples/train_bert_chat.md).

## Train from a list of books

`examples/train_data.cpp` is the file-list example. It is not part of `llm_tests`. The walkthrough is [examples/train_data.md](examples/train_data.md).

```bash
cmake --build build -j --target train_data
./build/train_data --files books.txt --out run
```

`books.txt` is one path per line. Blank lines and lines whose first character is `#` are ignored. A relative path is resolved from the directory that contains the list. Each file is read in full, trained as its own BPE string, and encoded with eos appended so a book boundary is a token. Pairs do not cross from one file into the next. The raw text is released before the network is built. `--dir DIR` recurses `DIR` and reads every regular file, in lexicographic path order, instead of `--files`. Each of those files is one BPE string with eos appended. Directory symlinks are not followed, and a symbolic link to a file is skipped. `--stdin` reads one document from standard input and omits `--files` and `--dir`. That document is one BPE string with one eos token appended. It can feed a training run or `--tokenize-only`.

The run writes `run/tokenizer.bpe`, `run/tokens.bin`, `run/tokens.meta`, and `run/model.bin`. `model.bin` is a version-2 façade checkpoint, version 4 for a `qwen-*` preset, or version 6 for a `bert-*` preset. After training a decoder, the process drops the training context, loads the tokenizer and `model.bin` with `LoadMode::Inference`, and prints a continuation. The default prompt is the opening of the first book. Generation stops on eos. A `bert-*` run trains masked tokens, prints `encoder: masked lm, mask token N`, and returns without sampling. `N` is the pad id. `chat` loads that directory and scores the pad id with `forward`.

```bash
./build/train_data --files books.txt --out run \
  --preset tiny --vocab 4096 --seq-len 256 --batch 4 --steps 1000 \
  --prompt "The morning after"
```

`--preset` is `tiny`, `small`, `base`, `llama-tiny`, `llama-small`, `llama-base`, `qwen-tiny`, `qwen-small`, `qwen-base`, `gpt2-tiny`, `gpt2-small`, `gpt2-base`, `gpt3-tiny`, `gpt3-small`, `gpt3-base`, `bert-tiny`, `bert-small`, or `bert-base`. The `llama-*` names train the RoPE and RMSNorm block from scratch. The `qwen-*` names train the Qwen3 block from scratch. The `gpt2-*` names train the GPT-2 block from scratch. The `gpt3-*` names train that dense block with context 2048. The `bert-*` names train masked tokens on the bidirectional block with context 512. `--seq-len` must be at most that preset's `max_seq_len` (1024, 2048 for `gpt3-*`, or 512 for `bert-*`). `--vocab` must be at least 259. Tokenizer training scans every byte once per merge, so a vocab of 4096 on several long books is much slower than the GPU steps. `--vocab 1024` is a faster first run. Defaults are `--preset tiny`, `--seq-len 256`, `--batch 4`, which fits a 12 GB GPU. `small` prefers 24 GB. `base` is a 24 GB model; keep the sequence at 512 or below on a smaller card. The same limit applies to `gpt2-base`, `gpt3-base`, and `bert-base`. `--temperature 0` is greedy. `--help` prints the flags. A gpt2 or gpt3 checkpoint is version 2. A bert checkpoint saved by this example is version 6. `gguf llama` rejects GELU, linear bias, bidirectional attention, and a segment table.

`--early-stop N` with `N > 0` holds out the last `--valid-fraction` (default `0.1`) of the token stream, trains on the prefix, and stops after `N` validation evals with no drop in cross-entropy. The saved checkpoint is that best step. `tokens.bin` stays the full stream. `--early-stop 0` trains on the whole stream and does not evaluate. `--min-loss FLOAT` stops when a step's training loss is at or below `FLOAT` and keeps that step on the whole stream. Omitting `--min-loss` leaves the threshold off. When that threshold stops the run, the saved checkpoint is that step even if `--early-stop` had stored an earlier validation best.

SIGINT or SIGTERM finishes the current update, writes `model.bin` with that step's weights and AdamW moments, prints `stop step N`, and exits before the continuation. A second signal aborts. `--resume` continues that directory. It keeps the tokenizer, the token file, the weights, and the AdamW moments, and it does not encode `--files` or `--dir` again. `--steps` is the total update count. The next update is the checkpoint step plus one, and its learning rate is `learning_rate_at` for that step under the `--lr`, `--warmup`, and `--steps` of this command. A checkpoint that has already reached `--steps` is not trained further. `--lr-new` with `--resume` keeps those weights and moments and starts the warmup and cosine at step 1 under this `--lr`, `--warmup`, and `--steps`. The AdamW step is kept, so the next plain `--resume` still counts from it. Batch windows are drawn again from `--seed`. `--early-stop` patience starts again. `finetune --resume` is a new run: it keeps moments and starts its own schedule at step 1. `--tokenize-only` trains the tokenizer, writes `tokenizer.bpe`, `tokens.bin`, and `tokens.meta`, and exits before the model. It does not write `model.bin`. `--stdin` can supply that text. `--reuse-tokens` is a new `train_data` run on the tokenizer and token file already in `--out`. It does not load `model.bin`. The schedule starts at step 1, and the finished run replaces `model.bin`.

## Talk to a trained model

`examples/chat.cpp` loads `tokenizer.bpe` and `model.bin` from a `train_data` directory or from `gguf from`. It is not part of `llm_tests`. Training stays in `train_data`. The walkthrough is [examples/chat.md](examples/chat.md). Put one exchange in each file. `train_data` appends eos after each file, so that token ends the assistant turn:

```
User: What is the harbor bell for?
Assistant: It marks the hour.
```

```bash
cmake --build build -j --target chat
./build/train_data --files dialogues.txt --out chat-run --vocab 1024
./build/chat --model chat-run
```

`chat` prints `you>`, reads a line, and builds a prompt from the transcript plus `User: `, the line, a newline, and `Assistant:`. The space after the colon is part of the reply the model generates. `model>` prints that reply with one leading space removed. The transcript keeps the space. A decoded `\nUser:` ends the turn for both the screen and the transcript. `/reset` clears the transcript. `/quit` or end of input exits. `--once TEXT` prints one reply and exits.

`--plain` leaves those tags off. The prompt is the transcript then the line, which matches a book checkpoint. The reply is printed as generated. A decoded `\nUser:` does not end it.

Most presets use `max_seq_len` 1024. `gpt3-*` uses 2048. The encoded prompt plus `--max-new-tokens` has to fit. `chat` drops the oldest turn until it does. A line that still does not fit prints `max_seq_len` and leaves the transcript as it was. A causal reply prefills the whole prompt. A bidirectional checkpoint, including `bert-*`, appends the pad id and scores that position with `forward`. There is no cache carried from one turn to the next.

Defaults are `--max-new-tokens 128`, `--temperature 0.8`, `--top-k 40`, `--top-p 0.9`, and `--seed 1`. `--temperature 0` is greedy. `--help` prints the flags.

## Fine-tune an existing checkpoint

`examples/finetune.cpp` loads `tokenizer.bpe` and `model.bin` from a `train_data` or `finetune` directory, encodes a new file list with that tokenizer, and keeps training. It is not part of `llm_tests`. The walkthrough is [examples/finetune.md](examples/finetune.md).

```bash
cmake --build build -j --target finetune
./build/finetune --from run --files more.txt --out run-ft --steps 200 --lr 1e-4
```

The tokenizer is not trained again. The checkpoint vocab and model shape stay as they are. `--out` must be a different directory from `--from`. The new directory gets its own `tokenizer.bpe`, token file, and `model.bin`. By default the run builds a fresh AdamW. `--resume` keeps the checkpoint moments and optimizer step. In both cases the learning-rate schedule starts at step 1 of this run. The default peak is `1e-4`, warmup is `0`, and the cosine floor is one tenth of `--lr`. After the last step a causal checkpoint is loaded from `--out` with `LoadMode::Inference` and a continuation is printed. A bidirectional checkpoint trains masked tokens, prints `encoder: masked lm, mask token N`, and does not sample.

## Convert a checkpoint to GGUF

`examples/gguf.cpp` converts `model.bin` to a GGUF version 3 file and back, writes a llama.cpp GGUF, and converts a public llama.cpp GGUF into a `chat` directory. It does not link `llm` or a device backend. It is not part of `llm_tests`. The walkthrough is [examples/gguf.md](examples/gguf.md).

```bash
cmake --build build -j --target gguf
./build/gguf export run/model.bin run/model.gguf
./build/gguf import run/model.gguf run/model.bin
./build/gguf llama run/model.bin run/model-llama.gguf
./build/gguf llama run/model.bin run/model-llama.gguf run/tokenizer.bpe
./build/gguf from model.gguf chat-run
./build/chat --model chat-run
```

`export` writes architecture `nnllm` and float32 weights. `import` reads `nnllm` files whose tensors are float32 or float16 and writes a checkpoint with no optimizer. That file is version 1, version 3 when the block is Qwen-style, or version 5 when attention is bidirectional or a segment table is present. `nnllm.positional` is `learned_absolute` or `rope`. `nnllm.causal` is written only when it is false. `nnllm.type_vocab_size` is written only when it is positive. `llama` reads a checkpoint or an `nnllm` GGUF and writes a file llama.cpp can load. A classic block is architecture `llama`. A Qwen-style block is `qwen3` when QK-norm is on, and `qwen2` otherwise. That file uses llama.cpp tensor names and splits SwiGLU `fc1` into `ffn_gate` and `ffn_up`. A learned-position checkpoint omits the position table and the LayerNorm biases, and llama.cpp then applies RoPE and RMSNorm, so those tokens differ. A RoPE checkpoint already uses that block, so the export keeps every tensor and does not print the mismatch warning. A `tokenizer.bpe` beside the checkpoint, or a path given after the output, is embedded so `llama-cli` can encode a prompt. The pre-tokenizer is `superbpe`: a merge that crosses a digit boundary can get a different id than this library's encoder. `from` reads a public llama.cpp GGUF, including Qwen2 and Qwen3, and writes a `chat` directory: `model.bin`, `tokenizer.bpe`, `tokens.bin`, and `tokens.meta`. GPT-2 BPE is remapped to LPB1. Quantized tensors expand to float32. Grouped-query attention, a head dim other than `d_model / heads`, QK-norm, the file's RoPE base, and its RMSNorm epsilon are kept. That checkpoint is version 1, or version 3 for a Qwen-style block. `import` rejects a GGUF with another architecture, or with quantized tensors. `llama` rejects GELU, linear bias, bidirectional attention, and a segment table. `from` rejects partial RoPE, MoE, a fused QKV matrix, GELU, and linear bias. `from` does not load a public BERT GGUF. The metadata keys, tensor names, and dimension order are in [examples/gguf.md](examples/gguf.md).

`adamw_groups` throws `std::invalid_argument("weight_decay")` when `weight_decay` is NaN, infinite, or negative. Decay applies only to rank-2 parameters whose names do not contain `ln` and do not end in `.bias`. Every other parameter, including LayerNorm weight and bias, is decay 0. A new group starts only when that choice changes, so the groups stay in `parameters()` order. `Transformer::save` rejects any other order.

## Trainer

Header: `llm/train.hpp`

```cpp
llm::TrainConfig cfg;
cfg.batch_size = 8;
cfg.seq_len = 512;
cfg.max_steps = 10000;
cfg.checkpoint_path = "model.bin";
llm::Trainer trainer(ctx, model, opt, cfg);
trainer.fit(train, &valid, rng);
```

`TrainConfig` defaults are `learning_rate` `3e-4`, `weight_decay` `0.1`, `beta1` `0.9`, `beta2` `0.95`, `batch_size` 8, `seq_len` 512, `max_steps` 10000, `start_step` 1, `grad_accum_steps` 1, `warmup_steps` 100, `min_lr` `3e-5`, `log_every` 50, `eval_every` 500, `eval_batches` 20, `ckpt_every` 1000, `grad_clip_norm` 1, `abort_on_non_finite` true, `early_stop_patience` 0, `early_stop_min_delta` 0, `min_loss` NaN, `masked_lm` false, `mask_prob` `0.15`, `mask_token` `-1`, an empty `checkpoint_path`, and a null `stop_flag`. The constructor throws `std::invalid_argument` with the field name when `batch_size`, `seq_len`, `max_steps`, `start_step`, or `grad_accum_steps` is not positive; when `start_step` is greater than `max_steps`; when `warmup_steps`, `log_every`, `eval_every`, `eval_batches`, `ckpt_every`, or `early_stop_patience` is negative; when `learning_rate` or `min_lr` is NaN, infinite, or negative; when `early_stop_min_delta` is NaN, infinite, or negative; when `min_loss` is infinite or negative; or when `grad_clip_norm` is NaN, infinite, or negative. When `masked_lm` is true, `mask_prob` must be finite and inside `(0, 1]`, and `mask_token` must be at least 0. A `mask_token` at or above the model vocabulary throws `mask_token` on the step that applies it. `min_loss` NaN leaves the threshold off.

`weight_decay`, `beta1`, and `beta2` are the values to use when building `AdamW`. `fit` does not rebuild the optimizer. It calls `set_learning_rate` from `learning_rate_at`.

`learning_rate_at(cfg, step)` uses a 1-based step. While `step <= warmup_steps` and warmup is positive, the rate is `learning_rate * step / warmup_steps`. After that, if `max_steps > warmup_steps`, progress is `(step - warmup_steps) / (max_steps - warmup_steps)` and the rate is `min_lr + 0.5 * (learning_rate - min_lr) * (1 + cos(pi * progress))`. Otherwise the rate stays `learning_rate`. Step `warmup_steps` is exactly `learning_rate`. Step `max_steps` is exactly `min_lr` on the cosine branch.

Each optimizer step calls `train` and `zero_grad` once, then `grad_accum_steps` micro-batches. A micro-batch is `next`, `forward`, and `sequence_cross_entropy` with ignore index `-100`. When `masked_lm` is set, the micro-batch overwrites that next-token pair before `forward`. Positions are chosen with probability `mask_prob`. A chosen position is labeled with its original id. Of those positions, 80% are replaced by `mask_token`, 10% by a random id in `[0, vocab)`, and 10% stay. Every other position is labeled `-100`. Validation uses the same replacement. The logit gradient is scaled in place by `1 / grad_accum_steps` and passed to `backward`. `opt.step` runs after the micro-batches. A non-finite micro-batch loss throws `std::runtime_error("non-finite loss")` before that backward when `abort_on_non_finite` is set. `grad_clip_norm > 0` clips with `clip_grad_norm`. A non-finite norm throws `std::runtime_error("non-finite grad")`. `grad_clip_norm == 0` does not clip.

`log_every > 0` prints step, loss, learning rate, and grad norm when `step % log_every == 0`. Every line `fit` prints starts with the local time `YYYY-MM-DD hh:mm:ss.sss` and one space. The printed loss is the mean of the micro-batch losses. When `valid` is non-null, `eval_every > 0`, and `eval_batches > 0`, the same cadence runs `eval_batches` forwards under `eval`, prints the mean cross-entropy and `exp` of that mean, then returns to `train`. When `ckpt_every > 0` and `checkpoint_path` is non-empty, the same cadence calls `model.save(checkpoint_path, &opt)`. That file is checkpoint version 2, version 4 for a Qwen-style block, or version 6 when attention is bidirectional or a segment table is present, with AdamW moments and the façade embeddings.

`fit` returns the last step that ran. That is `max_steps` when no stop trips. The loop starts at `start_step` instead of 1, and `learning_rate_at` uses that same step number. When patience trips, the return is the step that exhausted it, which is greater than the restored step. When `min_loss` trips, the return is the step whose training loss reached the threshold, and that step stays loaded. When `stop_flag` is non-null and nonzero after a step, grads are zeroed, a non-empty `checkpoint_path` is saved, `fit` prints `stop step <step>`, and the return is that step. `min_loss`, validation, and that step's `ckpt_every` write do not run. The flag is not cleared.

When `early_stop_patience` is positive, `fit` throws `std::invalid_argument("early_stop_patience")` when `valid` is null, `eval_every` or `eval_batches` is not positive, or `eval_every` is greater than `max_steps`. After each eval, a finite mean cross-entropy that is lower than the best by more than `early_stop_min_delta` becomes the new best. The first finite eval is always the best. The weights, AdamW moments, `step_count`, and learning rate are copied to the host at that step. An eval that does not improve adds one to the wait count. A non-finite eval mean throws `std::runtime_error("non-finite loss")` when `abort_on_non_finite` is set, and otherwise counts as no improvement. When the wait count reaches `early_stop_patience`, `fit` stops. At the end, if a best step was stored, those weights and moments are copied back, grads are zeroed, and a non-empty `checkpoint_path` is saved. A `ckpt_every` write during the loop can still store a later step; the save at the end replaces it with the best step. `fit` prints `early_stop step <last> best_step <best> best_eval_ce <ce>`. `<last>` is the returned step. `<best>` is the restored step.

A finite `min_loss` of at least 0 is checked after each optimizer step, against the same mean of micro-batch losses that `log_every` prints. A finite mean at or below `min_loss` stops the loop before validation and before that step's `ckpt_every` write. Grads are zeroed, a non-empty `checkpoint_path` is saved, and `fit` prints `min_loss step <step> loss <loss>`. An earlier validation snapshot is left unused. A non-finite step loss does not meet the threshold.

## Generator

Header: `llm/generate.hpp`

```cpp
llm::Generator generator(ctx, model, *tokenizer);
llm::GenerateConfig cfg;
cfg.max_new_tokens = 32;
cfg.temperature = 0.f;
cfg.top_k = 1;
cfg.eos_id = tokenizer->eos_id();
std::string text = generator.generate(prompt, cfg, rng);
```

`generate` throws `std::invalid_argument("max_new_tokens")` when `max_new_tokens < 1`, `("temperature")` when temperature is NaN, infinite, or negative, `("top_k")` when `top_k < 0`, and `("top_p")` when `top_p` is outside `(0, 1]`. It calls `model.eval()`, encodes the prompt with bos and eos off, and throws `("prompt")` when that id list is empty. `prompt_len + max_new_tokens` above `max_seq_len` throws `("max_seq_len")`.

The prompt ids are copied into an `IndexTensor` of shape `(1, prompt_len)` on the causal path, and the context is synchronized before that host buffer is destroyed. A causal model then uses `prefill` and `decode_step` only. It does not call `forward`. Each sampled id is fed back as shape `(1, 1)`.

A bidirectional model does not call `prefill` or `decode_step`. Each new token appends `pad_id`, uploads the whole sequence, synchronizes, and calls `forward`. Segment ids stay 0. The logits at the appended position are the row that is sampled. The pad id is then replaced by that sample. `pad_id` outside `[0, vocab)` throws `std::invalid_argument("mask_token")`. A `forward` result whose length is not `seq * vocab` throws `("logits")`.

Temperature `0` picks the argmax. Any other temperature divides the logits, keeps the top `top_k` when `top_k > 0` (`top_k == 0` skips top-k), softmaxes, then applies nucleus filtering on those probabilities when `top_p < 1` (`top_p == 1` skips nucleus), and draws with `rng`. A sampled id equal to `eos_id` stops the loop when `eos_id >= 0`, and that id is not appended. Other ids are kept until `max_new_tokens`. `decode` is called on the new ids only.
