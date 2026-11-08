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

## Presets and AdamW groups

Header: `llm/presets.hpp`

```cpp
nn::TransformerConfig config = llm::presets::tiny(vocab_size);
nn::Transformer model(ctx, config, /*seed=*/1);
std::vector<nn::AdamWGroup> groups = llm::adamw_groups(model, /*weight_decay=*/0.1f);
nn::AdamW opt(std::move(groups), 3e-4f, 0.9f, 0.95f, 1e-8f);
```

`tiny`, `small`, `base`, `llama_tiny`, `llama_small`, and `llama_base` throw `std::invalid_argument("vocab_size")` when `vocab_size < 1`. The other fields are fixed: dropout `0`, `bias` false, `tie_embeddings` false, SwiGLU, `max_seq_len` 1024. The first three use learned-absolute positions and LayerNorm. The `llama_*` presets set `positional` to RoPE, which selects RMSNorm and adjacent-pair RoPE on Q and K. A checkpoint trained with learned positions does not become that block.

| Preset | `num_layers` | `d_model` | `num_heads` | `d_ff` | Positions |
|---|---:|---:|---:|---:|---|
| tiny | 4 | 256 | 4 | 768 | learned absolute |
| small | 6 | 512 | 8 | 1536 | learned absolute |
| base | 12 | 768 | 12 | 2048 | learned absolute |
| llama_tiny | 4 | 256 | 4 | 768 | RoPE, RMSNorm |
| llama_small | 6 | 512 | 8 | 1536 | RoPE, RMSNorm |
| llama_base | 12 | 768 | 12 | 2048 | RoPE, RMSNorm |

Untied parameter counts at vocab 16000 are tiny 11866624 (11.87M), small 37368832 (37.37M), and base 110335488 (110.34M). The llama presets are llama_tiny 11602176 (11.60M), llama_small 36837888 (36.84M), and llama_base 109529856 (109.53M). The GPU notes below are float32, one GPU, full attention, and no activation checkpointing.

tiny fits in 12 GB. Suggested batches are `(8, 512)` and `(16, 256)`. small is tight in 12 GB at `(4, 512)`; 24 GB is comfortable. base is a 24 GB target, not a 12 GB target at `T=512`. Keep `T <= 1024` for base, and prefer `T <= 512` on a 12 GB GPU.

`examples/train_tiny.cpp` trains `presets::tiny` at the byte-level BPE vocab, a few million parameters, so that acceptance run fits easily. That preset stays learned absolute. The walkthrough is [examples/train_tiny.md](examples/train_tiny.md).

## Train from a list of books

`examples/train_books.cpp` is the file-list example. It is not part of `llm_tests`. The walkthrough is [examples/train_books.md](examples/train_books.md).

```bash
cmake --build build -j --target train_books
./build/train_books --files books.txt --out run
```

`books.txt` is one path per line. Blank lines and lines whose first character is `#` are ignored. A relative path is resolved from the directory that contains the list. Each file is read in full, trained as its own BPE string, and encoded with eos appended so a book boundary is a token. Pairs do not cross from one file into the next. The raw text is released before the network is built.

The run writes `run/tokenizer.bpe`, `run/tokens.bin`, `run/tokens.meta`, and `run/model.bin`. `model.bin` is a version-2 façade checkpoint. After training, the process drops the training context, loads the tokenizer and `model.bin` with `LoadMode::Inference`, and prints a continuation. The default prompt is the opening of the first book. Generation stops on eos.

```bash
./build/train_books --files books.txt --out run \
  --preset tiny --vocab 4096 --seq-len 256 --batch 4 --steps 1000 \
  --prompt "The morning after"
```

`--preset` is `tiny`, `small`, `base`, `llama-tiny`, `llama-small`, or `llama-base`. The `llama-*` names train the RoPE and RMSNorm block from scratch. `--seq-len` must be at most 1024. `--vocab` must be at least 259. Tokenizer training scans every byte once per merge, so a vocab of 4096 on several long books is much slower than the GPU steps. `--vocab 1024` is a faster first run. Defaults are `--preset tiny`, `--seq-len 256`, `--batch 4`, which fits a 12 GB GPU. `small` prefers 24 GB. `base` is a 24 GB model; keep the sequence at 512 or below on a smaller card. `--temperature 0` is greedy. `--help` prints the flags.

## Talk to a trained model

`examples/chat.cpp` loads `tokenizer.bpe` and `model.bin` from a `train_books` directory. It is not part of `llm_tests`. Training stays in `train_books`. The walkthrough is [examples/chat.md](examples/chat.md). Put one exchange in each file. `train_books` appends eos after each file, so that token ends the assistant turn:

```
User: What is the harbor bell for?
Assistant: It marks the hour.
```

```bash
cmake --build build -j --target chat
./build/train_books --files dialogues.txt --out chat-run --vocab 1024
./build/chat --model chat-run
```

`chat` prints `you>`, reads a line, and builds a prompt from the transcript plus `User: `, the line, a newline, and `Assistant:`. The space after the colon is part of the reply the model generates. `model>` prints that reply with one leading space removed. The transcript keeps the space. A decoded `\nUser:` ends the turn for both the screen and the transcript. `/reset` clears the transcript. `/quit` or end of input exits. `--once TEXT` prints one reply and exits.

Presets use `max_seq_len` 1024. The encoded prompt plus `--max-new-tokens` has to fit. `chat` drops the oldest `User:` / `Assistant:` turn until it does. A line that still does not fit prints `max_seq_len` and leaves the transcript as it was. Each reply prefills the whole prompt. There is no cache carried from one turn to the next.

Defaults are `--max-new-tokens 128`, `--temperature 0.8`, `--top-k 40`, `--top-p 0.9`, and `--seed 1`. `--temperature 0` is greedy. `--help` prints the flags.

## Fine-tune an existing checkpoint

`examples/finetune.cpp` loads `tokenizer.bpe` and `model.bin` from a `train_books` or `finetune` directory, encodes a new file list with that tokenizer, and keeps training. It is not part of `llm_tests`. The walkthrough is [examples/finetune.md](examples/finetune.md).

```bash
cmake --build build -j --target finetune
./build/finetune --from run --files more.txt --out run-ft --steps 200 --lr 1e-4
```

The tokenizer is not trained again. The checkpoint vocab and model shape stay as they are. `--out` must be a different directory from `--from`. The new directory gets its own `tokenizer.bpe`, token file, and `model.bin`. By default the run builds a fresh AdamW. `--resume` keeps the checkpoint moments and optimizer step. In both cases the learning-rate schedule starts at step 1 of this run. The default peak is `1e-4`, warmup is `0`, and the cosine floor is one tenth of `--lr`. After the last step the process loads `--out` with `LoadMode::Inference` and prints a continuation.

## Convert a checkpoint to GGUF

`examples/gguf.cpp` converts `model.bin` to a GGUF version 3 file and back. It does not link `llm` or CUDA. It is not part of `llm_tests`. The walkthrough is [examples/gguf.md](examples/gguf.md).

```bash
cmake --build build -j --target gguf
./build/gguf export run/model.bin run/model.gguf
./build/gguf import run/model.gguf run/model.bin
./build/gguf llama run/model.bin run/model-llama.gguf
```

`export` writes architecture `nnllm` and float32 weights. `import` reads `nnllm` files whose tensors are float32 or float16 and writes a version-1 checkpoint with no optimizer. `nnllm.positional` is `learned_absolute` or `rope`. `llama` reads a checkpoint or an `nnllm` GGUF and writes architecture `llama` for llama.cpp. That file uses llama.cpp tensor names and splits SwiGLU `fc1` into `ffn_gate` and `ffn_up`. A learned-position checkpoint omits the position table and the LayerNorm biases, and llama.cpp then applies RoPE and RMSNorm, so those tokens differ. A RoPE checkpoint already uses that block, so the export keeps every tensor and does not print the mismatch warning. `tokenizer.bpe` stays a separate file. `import` rejects a GGUF with another architecture, or with quantized tensors. `llama` rejects GELU and linear bias. The metadata keys, tensor names, and dimension order are in [examples/gguf.md](examples/gguf.md).

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

`TrainConfig` defaults are `learning_rate` `3e-4`, `weight_decay` `0.1`, `beta1` `0.9`, `beta2` `0.95`, `batch_size` 8, `seq_len` 512, `max_steps` 10000, `grad_accum_steps` 1, `warmup_steps` 100, `min_lr` `3e-5`, `log_every` 50, `eval_every` 500, `eval_batches` 20, `ckpt_every` 1000, `grad_clip_norm` 1, `abort_on_non_finite` true, and an empty `checkpoint_path`. The constructor throws `std::invalid_argument` with the field name when `batch_size`, `seq_len`, `max_steps`, or `grad_accum_steps` is not positive; when `warmup_steps`, `log_every`, `eval_every`, `eval_batches`, or `ckpt_every` is negative; when `learning_rate` or `min_lr` is NaN, infinite, or negative; or when `grad_clip_norm` is NaN, infinite, or negative.

`weight_decay`, `beta1`, and `beta2` are the values to use when building `AdamW`. `fit` does not rebuild the optimizer. It calls `set_learning_rate` from `learning_rate_at`.

`learning_rate_at(cfg, step)` uses a 1-based step. While `step <= warmup_steps` and warmup is positive, the rate is `learning_rate * step / warmup_steps`. After that, if `max_steps > warmup_steps`, progress is `(step - warmup_steps) / (max_steps - warmup_steps)` and the rate is `min_lr + 0.5 * (learning_rate - min_lr) * (1 + cos(pi * progress))`. Otherwise the rate stays `learning_rate`. Step `warmup_steps` is exactly `learning_rate`. Step `max_steps` is exactly `min_lr` on the cosine branch.

Each optimizer step calls `train` and `zero_grad` once, then `grad_accum_steps` micro-batches. A micro-batch is `next`, `forward`, and `sequence_cross_entropy` with ignore index `-100`. The logit gradient is scaled in place by `1 / grad_accum_steps` and passed to `backward`. `opt.step` runs after the micro-batches. A non-finite micro-batch loss throws `std::runtime_error("non-finite loss")` before that backward when `abort_on_non_finite` is set. `grad_clip_norm > 0` clips with `clip_grad_norm`. A non-finite norm throws `std::runtime_error("non-finite grad")`. `grad_clip_norm == 0` does not clip.

`log_every > 0` prints step, loss, learning rate, and grad norm when `step % log_every == 0`. The printed loss is the mean of the micro-batch losses. When `valid` is non-null, `eval_every > 0`, and `eval_batches > 0`, the same cadence runs `eval_batches` forwards under `eval`, prints the mean cross-entropy and `exp` of that mean, then returns to `train`. When `ckpt_every > 0` and `checkpoint_path` is non-empty, the same cadence calls `model.save(checkpoint_path, &opt)`. That file is checkpoint version 2 with AdamW moments and the façade embeddings.

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

The prompt is uploaded as an `IndexTensor` of shape `(1, prompt_len)`. The context is synchronized before that host buffer is destroyed. Generation then uses `prefill` and `decode_step` only. It does not call `forward` or `network().forward`.

Temperature `0` picks the argmax. Any other temperature divides the logits, keeps the top `top_k` when `top_k > 0` (`top_k == 0` skips top-k), softmaxes, then applies nucleus filtering on those probabilities when `top_p < 1` (`top_p == 1` skips nucleus), and draws with `rng`. A sampled id equal to `eos_id` stops the loop when `eos_id >= 0`, and that id is not appended. Other ids are appended and fed back as shape `(1, 1)` until `max_new_tokens`. `decode` is called on the new ids only.
