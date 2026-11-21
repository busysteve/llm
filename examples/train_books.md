# How to train from a list of books

`train_books` is the file-list example in [`train_books.cpp`](train_books.cpp). It reads a list of text files, trains a byte-level BPE tokenizer and a preset transformer, writes the tokenizer and the checkpoint, then loads both and prints a continuation. A `bert-*` preset trains masked tokens and does not print a continuation. `--early-stop` can hold out a suffix of the tokens and stop when that validation loss stops falling. `--min-loss` can stop when a step's training loss reaches a threshold. SIGINT or SIGTERM saves the current step, `--resume` continues that same run, and `--reuse-tokens` starts a new model on the saved tokens. It is a separate program from `llm_tests`.

Build from the repository root. Requirements are in the [README](../README.md).

```bash
./build.sh amd       # or: ./build.sh opencl, ./build.sh cuda, ./build.sh arc
cmake --build build -j --target train_books
```

Use text you have the rights to train on. The sample commands below assume those files are already on disk.

## Prepare the list

`books.txt` is one path per line. Blank lines are ignored. A line whose first character is `#` is ignored. A relative path is resolved from the directory that contains the list.

```
# one work per line
novels/harbor.txt
novels/tide.txt
```

Each file is read in full and trained as its own BPE string. Pairs do not cross from one file into the next. The encoder then appends an eos token, so a file boundary is a token in the training stream. The raw text is released before the network is built.

The whole token array stays in host memory. A corpus shorter than `--seq-len + 1` tokens fails with `train_books: short corpus`.

## Run

A first run with a smaller vocab spends less time in the tokenizer. Tokenizer training scans the corpus once per merge, in parallel on the CPU, so `--vocab 4096` on long books is the slow part. The GPU steps come after that.

```bash
./build/train_books --files books.txt --out run \
  --preset tiny --vocab 1024 --seq-len 256 --batch 4 --steps 1000 \
  --prompt "The morning after"
```

Progress is printed on stderr: each file and its size, the vocab and byte count, the token count, and the paths that were saved. With `--early-stop` the stderr line `holdout train=<n> valid=<n> early_stop=<n> eval_every=<n>` follows the token count. Training steps are printed on stdout. Each of those lines starts with the local time:

```
YYYY-MM-DD hh:mm:ss.sss step 10 loss <loss> lr <lr> grad_norm <norm>
```

When a signal stops the run, stdout prints `YYYY-MM-DD hh:mm:ss.sss stop step <step>` and the process exits before the continuation. When `--resume` continues a checkpoint, stderr prints `resuming step <N>` and the next stdout step is `N + 1`. When that checkpoint has already reached `--steps`, stderr prints `resume step <N> reached steps <steps>` and no further update runs. When `--reuse-tokens` restarts on the saved stream, stderr prints `reusing <N> tokens` and the next stdout step is `1`.

When `--min-loss` trips, stdout also prints:

```
YYYY-MM-DD hh:mm:ss.sss min_loss step <step> loss <loss>
```

`<step>` is the update that reached the threshold, and `<loss>` is that step's training loss. With `--early-stop`, validation lines are printed on the same stream:

```
YYYY-MM-DD hh:mm:ss.sss step 50 eval_ce <loss> perplexity <exp(loss)>
YYYY-MM-DD hh:mm:ss.sss early_stop step <last> best_step <best> best_eval_ce <loss>
```

When a decoder finishes, stdout ends with:

```
prompt: The morning after
continuation: <new text>
```

The default prompt, when `--prompt` is omitted, is the opening of the first file: leading whitespace is skipped, then up to 48 Unicode code points are taken, and a newline ends the excerpt early. Generation stops when eos is sampled. That id is not part of the continuation.

A `bert-tiny`, `bert-small`, or `bert-base` run does not sample. After the save it prints one line and exits 0:

```
encoder: masked lm, mask token <pad id>
```

The pad id is the mask. `encode` does not insert it. About 15% of positions are trained: 80% become that id, 10% become a random id, and 10% keep the original token. `chat` loads the checkpoint and continues a `User:` / `Assistant:` transcript by scoring that id with `forward`. [train_bert_chat](train_bert_chat.md) is the short in-process run of the same objective.

## Files written

`run/` receives four files.

| File | Contents |
|---|---|
| `tokenizer.bpe` | byte-level BPE, magic `LPB1` |
| `tokens.bin` | token ids, int32 little-endian |
| `tokens.meta` | vocab size and length for `TextDataset` |
| `model.bin` | version-2 façade checkpoint, version 4 for a `qwen-*` preset, or version 6 for a `bert-*` preset, including AdamW moments |

With `--steps` of 500 or more, `model.bin` is also written every 500 steps. The final save always runs after the last step. `--steps` below 500 writes the checkpoint once, at the end, and also when a signal stops an earlier step. With `--early-stop`, that final file is the best validation step, including that step's AdamW moments, even when a 500-step write was a later step. When `--min-loss` stops the run, `model.bin` is the step whose training loss reached the threshold, including that step's AdamW moments. A signal writes the update that just finished, including its AdamW moments, and does not copy back an earlier validation best.

Load `tokenizer.bpe` and `model.bin` together. `chat` does that for a dialogue corpus. `--plain` feeds the line as the prompt with no `User:` / `Assistant:` tags, which matches a book continuation. See [chat](chat.md).

## Flags

`--files` and `--out` are required. `--help` prints this list and exits 0.

| Flag | Default | Role |
|---|---|---|
| `--preset tiny\|small\|base\|llama-tiny\|llama-small\|llama-base\|qwen-tiny\|qwen-small\|qwen-base\|gpt2-tiny\|gpt2-small\|gpt2-base\|gpt3-tiny\|gpt3-small\|gpt3-base\|bert-tiny\|bert-small\|bert-base` | `tiny` | model shape |
| `--vocab N` | 4096 | BPE size, minimum 259 |
| `--seq-len N` | 256 | tokens per example, at most the preset `max_seq_len` |
| `--batch N` | 4 | examples per step |
| `--steps N` | 1000 | AdamW updates |
| `--accum N` | 1 | micro-batches per update |
| `--warmup N` | 100 | linear warmup steps |
| `--log-every N` | 10 | print every Nth step |
| `--early-stop N` | 0 | stop after N validation evals with no loss drop. 0 keeps training for `--steps` |
| `--min-loss FLOAT` | off | stop when a step's training loss is at or below `FLOAT` and keep that step |
| `--eval-every N` | 0, or `min(50, --steps)` when `--early-stop` is positive | steps between validation evals |
| `--valid-fraction FLOAT` | 0.1 | suffix of the token stream held out when `--early-stop` is positive |
| `--lr FLOAT` | 3e-4 | peak learning rate |
| `--prompt TEXT` | opening of the first file | text to continue |
| `--max-new-tokens N` | 80 | tokens to sample after training |
| `--temperature FLOAT` | 0.8 | `0` is greedy |
| `--top-k N` | 40 | `0` disables top-k |
| `--top-p FLOAT` | 0.9 | nucleus threshold |
| `--seed N` | 1 | model init and batch order. `--resume` keeps the checkpoint weights and draws windows from this seed again. `--reuse-tokens` initializes a new model from this seed |
| `--resume` | off | continue `--out` from its checkpoint. `--steps` is the total update count |
| `--reuse-tokens` | off | train a new model on the tokenizer and token file already in `--out` |

Weight decay stays 0.1. The cosine floor is one tenth of `--lr`. Eval is off unless `--early-stop` is positive. Defaults `--preset tiny`, `--seq-len 256`, and `--batch 4` fit a 12 GB GPU. `small` prefers 24 GB. `base` is a 24 GB model. On a smaller card keep `--seq-len` at 512 or below.

`tiny`, `small`, and `base` use learned absolute positions and LayerNorm. `llama-tiny`, `llama-small`, and `llama-base` are the same widths with RoPE and RMSNorm, which is the block llama.cpp runs. `qwen-tiny`, `qwen-small`, and `qwen-base` are the Qwen3 block: head dim 128, grouped-query attention, QK-norm, RoPE base `1e6`, and RMSNorm epsilon `1e-6`. `gpt2-tiny`, `gpt2-small`, and `gpt2-base` are the GPT-2 block: GELU, linear bias, `d_ff = 4 * d_model`, learned positions, and context 1024. `gpt3-tiny`, `gpt3-small`, and `gpt3-base` are that dense block with context 2048. `bert-tiny`, `bert-small`, and `bert-base` are that dense block with bidirectional attention, a segment table of size 2, and context 512. They train masked tokens. The pad id is the mask. Train a llama, qwen, GPT, or BERT preset from scratch. A checkpoint that was trained with learned positions does not become a llama or qwen block. The saved `qwen-*` checkpoint is version 4. A `gpt2-*` or `gpt3-*` checkpoint is version 2. A `bert-*` checkpoint is version 6. `gguf llama` rejects GELU, linear bias, bidirectional attention, and a segment table. Usage, benefits, and applications for each preset type are in [preset-default](../docs/preset-default.md), [preset-llama](../docs/preset-llama.md), [preset-qwen](../docs/preset-qwen.md), [preset-gpt2](../docs/preset-gpt2.md), [preset-gpt3](../docs/preset-gpt3.md), and [preset-bert](../docs/preset-bert.md). [train_chat](train_chat.md) trains `tiny` and `bert-tiny` on four exchanges, then prints one reply and one masked token. [train_bert_chat](train_bert_chat.md) trains `bert_tiny` on those exchanges and prints one reply.

## Early stop

`--early-stop N` with `N > 0` holds out the last `--valid-fraction` of the token stream and trains on the prefix. `tokens.bin` stays the full stream. A window does not cross the cut. The suffix is about that fraction of the tokens, and it grows or shrinks so both sides still have at least `--seq-len + 1` tokens. A corpus that cannot be split that way fails with `short valid`.

Every `--eval-every` steps the run measures validation cross-entropy on 20 windows. The first finite measurement is the baseline. Any lower value is an improvement. An equal or higher value is not. After `N` measurements with no improvement, training stops. The weights, AdamW moments, and learning rate from the best measurement are copied back, and that is the `model.bin` at the end. Those tensors are kept on the host until the run ends.

`<last>` in the `early_stop` line is the step that exhausted patience, or `--steps` when patience never tripped. `<best>` is the restored step.

```bash
./build/train_books --files books.txt --out run \
  --preset tiny --vocab 1024 --seq-len 256 --batch 4 --steps 1000 \
  --early-stop 3
```

## Loss threshold

`--min-loss FLOAT` stops after an AdamW update whose training loss is at or below `FLOAT`. That loss is the mean next-token cross-entropy of the step's micro-batches, the same number printed as `loss`. It is separate from `eval_ce`. The threshold is off when the flag is omitted. `0` is a real threshold.

The whole token stream stays in the training set. The weights, AdamW moments, and learning rate of the step that reached the threshold stay in place, and that is the `model.bin` at the end. A 500-step write from earlier in the run is replaced. When `--early-stop` is also set and this threshold trips first, the finished checkpoint is this step.

```bash
./build/train_books --files books.txt --out run \
  --preset tiny --vocab 1024 --seq-len 256 --batch 4 --steps 1000 \
  --min-loss 2.5
```

## Stop and resume

SIGINT or SIGTERM finishes the AdamW update already running, writes `model.bin`, prints `YYYY-MM-DD hh:mm:ss.sss stop step <step>`, and exits 0. It does not sample a continuation. A second signal aborts the process. The handler only sets a flag. The flag is checked after the update, not inside a GPU call.

`--resume` reads `tokenizer.bpe`, `tokens.bin`, `tokens.meta`, and `model.bin` from `--out`. The tokenizer is not trained again, and `--files` is not encoded again. `--files` is still required. When `--prompt` is omitted, the continuation prompt is the opening of the first file. `--preset` and `--vocab` have to match the checkpoint and the tokenizer. The saved token stream stays the training stream, including the holdout split when `--early-stop` is set.

`--steps` counts every update from the start of the run, including updates already in the checkpoint. The next update is that AdamW step plus one. Its learning rate is the warmup and cosine value for that step under this `--warmup`, `--lr`, and `--steps`. The schedule does not start over at step 1. A checkpoint whose step is already at least `--steps` is left as it is. stderr prints `resume step <N> reached steps <steps>`, and a causal run still prints a continuation. A larger `--steps` trains through the new total. Changing `--steps`, `--warmup`, or `--lr` changes the rate of the updates that remain. The AdamW moments stay as saved.

Batch windows are drawn again from `--seed`. They are not the next draws from the interrupted process. `--early-stop` patience starts again. If the previous process ended on `--early-stop`, `model.bin` is that best step, and resume continues from it. A run without `--resume` or `--reuse-tokens` replaces the tokenizer and the token file.

```bash
./build/train_books --files books.txt --out run \
  --preset tiny --vocab 1024 --seq-len 256 --batch 4 --steps 1000
# interrupt with Ctrl-C, then:
./build/train_books --files books.txt --out run \
  --preset tiny --vocab 1024 --seq-len 256 --batch 4 --steps 1000 \
  --resume
```

[`finetune`](finetune.md) `--resume` keeps moments for a new file list and starts that run's schedule at step 1.

## Restart on saved tokens

`--reuse-tokens` reads `tokenizer.bpe`, `tokens.bin`, and `tokens.meta` from `--out` and trains a new model. It does not load `model.bin`. The finished run replaces `model.bin`. The tokenizer is not trained again, and `--files` is not encoded again. `--files` is still required. When `--prompt` is omitted, the continuation prompt is the opening of the first file. `--vocab` has to match the tokenizer. `--preset` chooses the new model. The learning-rate schedule starts at step 1 under this command's `--lr`, `--warmup`, and `--steps`. `--seed` initializes that model and draws the batch windows. stderr prints `reusing <N> tokens`. Combining `--reuse-tokens` with `--resume` fails with `reuse-tokens`.

```bash
./build/train_books --files books.txt --out run \
  --preset tiny --vocab 1024 --seq-len 256 --batch 4 --steps 1000 \
  --reuse-tokens
```

## When it fails

Errors print `train_books: <reason>` on stderr and the process exits 1. Common reasons:

| Reason | Meaning |
|---|---|
| `files` | the list is missing, empty, or names a path that is not a file |
| `vocab_size` | `--vocab` is below 259, the corpus runs out of pairs before the vocab is full, or `--resume` or `--reuse-tokens` and `--vocab`, the token file, or the checkpoint disagrees with the tokenizer |
| `short corpus` | the encoded files have fewer than `--seq-len + 1` tokens |
| `short valid` | `--early-stop` is positive and the stream cannot be split into two spans of `--seq-len + 1` tokens |
| `seq_len` | `--seq-len` is above the preset `max_seq_len` (1024, 2048 for `gpt3-*`, or 512 for `bert-*`) |
| `early_stop` | `--early-stop` is negative |
| `min_loss` | `--min-loss` is negative or infinite |
| `eval_every` | `--early-stop` is positive and `--eval-every` is not in `1 .. --steps` |
| `valid_fraction` | `--valid-fraction` is not inside `(0, 1)` |
| `preset` | the name is not `tiny`, `small`, `base`, `llama-tiny`, `llama-small`, `llama-base`, `qwen-tiny`, `qwen-small`, `qwen-base`, `gpt2-tiny`, `gpt2-small`, `gpt2-base`, `gpt3-tiny`, `gpt3-small`, `gpt3-base`, `bert-tiny`, `bert-small`, or `bert-base`, or `--resume` and that preset does not match the checkpoint |
| `resume` | `--resume` is set and `--out` is missing `tokenizer.bpe`, `tokens.bin`, `tokens.meta`, or `model.bin` |
| `reuse-tokens` | `--reuse-tokens` is combined with `--resume`, or `--out` is missing `tokenizer.bpe`, `tokens.bin`, or `tokens.meta` |
| `adamw` | `--resume` found a checkpoint whose optimizer is not AdamW |
| `optimizer block missing` | `--resume` found a checkpoint with no optimizer |
| a flag name | the flag is unknown, or its value is missing or not a number |

Library details are in [USAGE.md](../USAGE.md). The one-paragraph acceptance run is [train_tiny](train_tiny.md). `--resume` continues this run. To keep training the checkpoint on more text, see [finetune](finetune.md).
