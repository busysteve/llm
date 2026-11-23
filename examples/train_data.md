# How to train from a list of books

`train_data` is the file-list example in [`train_data.cpp`](train_data.cpp). It reads a list of text files, or every regular file under a directory, trains a byte-level BPE tokenizer and a preset transformer, writes the tokenizer and the checkpoint, then loads both and prints a continuation. A `bert-*` preset trains masked tokens and does not print a continuation. `--early-stop` can hold out a suffix of the tokens and stop when that validation loss stops falling. `--min-loss` can stop when a step's training loss reaches a threshold. SIGINT or SIGTERM saves the current step, `--resume` continues that same run, `--lr-new` with `--resume` starts a new learning-rate schedule at step 1, `--reuse-tokens` starts a new model on the saved tokens, and `--tokenize-only` writes that tokenizer and token file and exits before the model. `--dir` recurses a directory instead of `--files`. `--stdin` reads one document from standard input instead of `--files` or `--dir`. It is a separate program from `llm_tests`.

Build from the repository root. Requirements are in the [README](../README.md).

```bash
./build.sh amd       # or: ./build.sh opencl, ./build.sh cuda, ./build.sh arc
cmake --build build -j --target train_data
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

The whole token array stays in host memory. A corpus shorter than `--seq-len + 1` tokens fails with `train_data: short corpus`.

## Directory

`--dir` recurses a directory and reads every regular file. Paths are sorted, so the first file is the first path in lexicographic order. Each file is one BPE string, and encoding appends eos, the same as a line in `--files`. Directory symlinks are not followed. A symbolic link to a file is skipped. A path that is not a directory, and a directory with no regular files, fail with `dir`. An empty file fails with that path.

```bash
./build/train_data --dir novels --out run \
  --preset tiny --vocab 1024 --seq-len 256 \
  --tokenize-only
```

Combining `--dir` with `--files` fails with `dir`. Combining `--dir` with `--stdin` fails with `stdin`.

## Standard input

`--stdin` reads one document from standard input and replaces `--files` and `--dir`. The document is one BPE string. One eos token is appended. A pipe of several books does not insert a boundary between them. Use `--files` or `--dir` when each file should end with eos.

```bash
cat novels/harbor.txt | ./build/train_data --stdin --out run \
  --preset tiny --vocab 1024 --seq-len 256 \
  --tokenize-only
```

stderr prints `read stdin (<bytes> bytes)`. Empty standard input fails with `stdin`. Combining `--stdin` with `--files`, `--dir`, `--resume`, or `--reuse-tokens` fails with `stdin`.

## Run

A first run with a smaller vocab spends less time in the tokenizer. Tokenizer training scans the corpus once per merge, in parallel on the CPU, so `--vocab 4096` on long books is the slow part. The GPU steps come after that.

```bash
./build/train_data --files books.txt --out run \
  --preset tiny --vocab 1024 --seq-len 256 --batch 4 --steps 1000 \
  --prompt "The morning after"
```

Progress is printed on stderr: each file and its size, or `read stdin (<bytes> bytes)`, then the vocab and byte count, the token count, and the paths that were saved. With `--early-stop` the stderr line `holdout train=<n> valid=<n> early_stop=<n> eval_every=<n>` follows the token count. Training steps are printed on stdout. Each of those lines starts with the local time:

```
YYYY-MM-DD hh:mm:ss.sss step 10 loss <loss> lr <lr> grad_norm <norm>
```

When a signal stops the run, stdout prints `YYYY-MM-DD hh:mm:ss.sss stop step <step>` and the process exits before the continuation. When `--resume` continues a checkpoint, stderr prints `resuming step <N>` and the next stdout step is `N + 1`. When that checkpoint has already reached `--steps`, stderr prints `resume step <N> reached steps <steps>` and no further update runs. When `--lr-new` restarts the schedule, stderr prints `lr-new step 1 (AdamW step <N>)` and the next stdout step is `1`. When `--reuse-tokens` restarts on the saved stream, stderr prints `reusing <N> tokens` and the next stdout step is `1`. When `--tokenize-only` finishes, stderr prints `tokenize only` and the process exits before the model.

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

The default prompt, when `--prompt` is omitted, is the opening of the first file: leading whitespace is skipped, then up to 48 Unicode code points are taken, and a newline ends the excerpt early. With `--dir`, that file is the first path in lexicographic order. Generation stops when eos is sampled. That id is not part of the continuation.

A `bert-tiny`, `bert-small`, `bert-base`, `bert-base-medium`, `bert-base-large`, `bert-medium`, or `bert-large` run does not sample. After the save it prints one line and exits 0:

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

`--tokenize-only` writes the first three files and does not write `model.bin`. An existing `model.bin` is left in place.

With `--steps` of 500 or more, `model.bin` is also written every 500 steps. The final save always runs after the last step. `--steps` below 500 writes the checkpoint once, at the end, and also when a signal stops an earlier step. With `--early-stop`, that final file is the best validation step, including that step's AdamW moments, even when a 500-step write was a later step. When `--min-loss` stops the run, `model.bin` is the step whose training loss reached the threshold, including that step's AdamW moments. A signal writes the update that just finished, including its AdamW moments, and does not copy back an earlier validation best.

Load `tokenizer.bpe` and `model.bin` together. `chat` does that for a dialogue corpus. `--plain` feeds the line as the prompt with no `User:` / `Assistant:` tags, which matches a book continuation. See [chat](chat.md).

## Flags

`--out` is required. Pass `--files`, `--dir`, or `--stdin`. `--help` prints this list and exits 0.

| Flag | Default | Role |
|---|---|---|
| `--preset tiny\|small\|base\|base-medium\|base-large\|medium\|large\|llama-tiny\|llama-small\|llama-base\|llama-base-medium\|llama-base-large\|llama-medium\|llama-large\|qwen-tiny\|qwen-small\|qwen-base\|qwen-base-medium\|qwen-base-large\|qwen-medium\|qwen-large\|gpt2-tiny\|gpt2-small\|gpt2-base\|gpt2-base-medium\|gpt2-base-large\|gpt2-medium\|gpt2-large\|gpt3-tiny\|gpt3-small\|gpt3-base\|gpt3-base-medium\|gpt3-base-large\|gpt3-medium\|gpt3-large\|bert-tiny\|bert-small\|bert-base\|bert-base-medium\|bert-base-large\|bert-medium\|bert-large` | `tiny` | block and starting shape |
| `--layers N` | preset | layer count |
| `--d-model N` | preset | model width |
| `--heads N` | preset | query-head count |
| `--d-ff N` | preset | feed-forward width |
| `--max-seq N` | preset | context cap stored in the checkpoint |
| `--kv-heads N` | preset | KV heads. `0` uses one KV head per query head |
| `--head-dim N` | preset | head width. `0` uses `d_model / heads` |
| `--vocab N` | 4096 | BPE size, minimum 259 |
| `--seq-len N` | 256 | tokens per example, at most the context cap |
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
| `--prompt TEXT` | opening of the first file, or of the stdin document | text to continue. With `--dir`, the first file is the first path in lexicographic order |
| `--max-new-tokens N` | 80 | tokens to sample after training |
| `--temperature FLOAT` | 0.8 | `0` is greedy |
| `--top-k N` | 40 | `0` disables top-k |
| `--top-p FLOAT` | 0.9 | nucleus threshold |
| `--seed N` | 1 | model init and batch order. `--resume` keeps the checkpoint weights and draws windows from this seed again. `--reuse-tokens` initializes a new model from this seed |
| `--resume` | off | continue `--out` from its checkpoint. `--steps` is the total update count |
| `--lr-new` | off | with `--resume`, start the learning-rate schedule at step 1. `--lr` is the new peak. The AdamW step is kept |
| `--reuse-tokens` | off | train a new model on the tokenizer and token file already in `--out` |
| `--tokenize-only` | off | train the tokenizer, write the token file, and exit before the model |
| `--dir DIR` | | recurse `DIR` and read every regular file instead of `--files` |
| `--stdin` | off | read one document from standard input instead of `--files` or `--dir` |

Weight decay stays 0.1. The cosine floor is one tenth of `--lr`. Eval is off unless `--early-stop` is positive. Defaults `--preset tiny`, `--seq-len 256`, and `--batch 4` fit a 12 GB GPU. `small` prefers 24 GB. `base` is a 24 GB model. On a smaller card keep `--seq-len` at 512 or below. `base-medium` is 18 layers at the base width, and `base-large` is 20 layers at the base width. `medium` is 24 layers at width 1024, and `large` is 36 layers at width 1280.

`tiny`, `small`, `base`, `base-medium`, `base-large`, `medium`, and `large` use learned absolute positions and LayerNorm. `base-medium` is 18 layers at the base width. `base-large` is 20 layers at the base width. `medium` is 24 layers at width 1024. `large` is 36 layers at width 1280. `llama-tiny`, `llama-small`, `llama-base`, `llama-base-medium`, `llama-base-large`, `llama-medium`, and `llama-large` are the same widths with RoPE and RMSNorm, which is the block llama.cpp runs. `qwen-tiny`, `qwen-small`, `qwen-base`, `qwen-base-medium`, `qwen-base-large`, `qwen-medium`, and `qwen-large` are the Qwen3 block: head dim 128, grouped-query attention, QK-norm, RoPE base `1e6`, and RMSNorm epsilon `1e-6`. `gpt2-tiny`, `gpt2-small`, `gpt2-base`, `gpt2-base-medium`, `gpt2-base-large`, `gpt2-medium`, and `gpt2-large` are the GPT-2 block: GELU, linear bias, `d_ff = 4 * d_model`, learned positions, and context 1024. `gpt3-tiny`, `gpt3-small`, `gpt3-base`, `gpt3-base-medium`, `gpt3-base-large`, `gpt3-medium`, and `gpt3-large` are that dense block with context 2048. `bert-tiny`, `bert-small`, `bert-base`, `bert-base-medium`, `bert-base-large`, `bert-medium`, and `bert-large` are that dense block with bidirectional attention, a segment table of size 2, and context 512. They train masked tokens. The pad id is the mask. Train a llama, qwen, GPT, or BERT preset from scratch. A checkpoint that was trained with learned positions does not become a llama or qwen block. The saved `qwen-*` checkpoint is version 4. A `gpt2-*` or `gpt3-*` checkpoint is version 2. A `bert-*` checkpoint is version 6. `gguf llama` rejects GELU, linear bias, bidirectional attention, and a segment table. Usage, benefits, and applications for each preset type are in [preset-default](../docs/preset-default.md), [preset-llama](../docs/preset-llama.md), [preset-qwen](../docs/preset-qwen.md), [preset-gpt2](../docs/preset-gpt2.md), [preset-gpt3](../docs/preset-gpt3.md), and [preset-bert](../docs/preset-bert.md). [train_chat](train_chat.md) trains `tiny` and `bert-tiny` on four exchanges, then prints one reply and one masked token. [train_bert_chat](train_bert_chat.md) trains `bert_tiny` on those exchanges and prints one reply.

## Custom shape

`--preset` selects the block and the starting sizes. `--layers`, `--d-model`, `--heads`, `--d-ff`, `--max-seq`, `--kv-heads`, and `--head-dim` replace individual fields. An omitted flag keeps the preset value. `--d-model 512` on `tiny` leaves 4 heads and `d_ff` 768.

`--max-seq` is the context cap stored in the checkpoint. `--seq-len` is the training window and must be at most that cap. On a learned-position preset, `--max-seq` sets the position-table rows. On `llama-*` and `qwen-*` it adds no parameters. Attention memory grows with the square of `--seq-len`.

On `qwen-*`, `--head-dim` sets the Q, K, V, and QK-norm width. `--kv-heads` sets the grouped-query width. QK-norm, RoPE base `1e6`, and RMSNorm epsilon `1e-6` stay with the preset. `--kv-heads 0` stores one KV head per query head. `--head-dim 0` stores `d_model / num_heads`.

Activation, bias, RoPE or learned positions, QK-norm, causal attention, and the segment table stay with `--preset`. Resume repeats the same `--preset` and the same size flags. A different depth, width, or context fails with `preset`. `--reuse-tokens` trains the new shape on the tokenizer and token file already in `--out`.

When any size flag is set, stderr prints one line before training:

```
spec layers=8 d_model=256 heads=4 kv_heads=0 head_dim=0 d_ff=768 max_seq=2048
```

`chat` reads the cap from the checkpoint.

```bash
./build/train_data --files books.txt --out run \
  --preset tiny --layers 8 --max-seq 2048 --seq-len 512 \
  --vocab 1024 --batch 4 --steps 1000
```

## Early stop

`--early-stop N` with `N > 0` holds out the last `--valid-fraction` of the token stream and trains on the prefix. `tokens.bin` stays the full stream. A window does not cross the cut. The suffix is about that fraction of the tokens, and it grows or shrinks so both sides still have at least `--seq-len + 1` tokens. A corpus that cannot be split that way fails with `short valid`.

Every `--eval-every` steps the run measures validation cross-entropy on 20 windows. The first finite measurement is the baseline. Any lower value is an improvement. An equal or higher value is not. After `N` measurements with no improvement, training stops. The weights, AdamW moments, and learning rate from the best measurement are copied back, and that is the `model.bin` at the end. Those tensors are kept on the host until the run ends.

`<last>` in the `early_stop` line is the step that exhausted patience, or `--steps` when patience never tripped. `<best>` is the restored step.

```bash
./build/train_data --files books.txt --out run \
  --preset tiny --vocab 1024 --seq-len 256 --batch 4 --steps 1000 \
  --early-stop 3
```

## Loss threshold

`--min-loss FLOAT` stops after an AdamW update whose training loss is at or below `FLOAT`. That loss is the mean next-token cross-entropy of the step's micro-batches, the same number printed as `loss`. It is separate from `eval_ce`. The threshold is off when the flag is omitted. `0` is a real threshold.

The whole token stream stays in the training set. The weights, AdamW moments, and learning rate of the step that reached the threshold stay in place, and that is the `model.bin` at the end. A 500-step write from earlier in the run is replaced. When `--early-stop` is also set and this threshold trips first, the finished checkpoint is this step.

```bash
./build/train_data --files books.txt --out run \
  --preset tiny --vocab 1024 --seq-len 256 --batch 4 --steps 1000 \
  --min-loss 2.5
```

## Stop and resume

SIGINT or SIGTERM finishes the AdamW update already running, writes `model.bin`, prints `YYYY-MM-DD hh:mm:ss.sss stop step <step>`, and exits 0. It does not sample a continuation. A second signal aborts the process. The handler only sets a flag. The flag is checked after the update, not inside a GPU call.

`--resume` reads `tokenizer.bpe`, `tokens.bin`, `tokens.meta`, and `model.bin` from `--out`. The tokenizer is not trained again, and `--files` or `--dir` is not encoded again. `--files` or `--dir` is still required. When `--prompt` is omitted, the continuation prompt is the opening of the first file. `--preset`, the size flags, and `--vocab` have to match the checkpoint and the tokenizer. An omitted size flag keeps that preset's value, so a custom checkpoint is resumed with the same overrides. The saved token stream stays the training stream, including the holdout split when `--early-stop` is set.

`--steps` counts every update from the start of the run, including updates already in the checkpoint. The next update is that AdamW step plus one. Its learning rate is the warmup and cosine value for that step under this `--warmup`, `--lr`, and `--steps`. The schedule does not start over at step 1. A checkpoint whose step is already at least `--steps` is left as it is. stderr prints `resume step <N> reached steps <steps>`, and a causal run still prints a continuation. A larger `--steps` trains through the new total. Changing `--steps`, `--warmup`, or `--lr` changes the rate of the updates that remain. The AdamW moments stay as saved.

Batch windows are drawn again from `--seed`. They are not the next draws from the interrupted process. `--early-stop` patience starts again. If the previous process ended on `--early-stop`, `model.bin` is that best step, and resume continues from it. A run without `--resume` or `--reuse-tokens` replaces the tokenizer and the token file.

```bash
./build/train_data --files books.txt --out run \
  --preset tiny --vocab 1024 --seq-len 256 --batch 4 --steps 1000
# interrupt with Ctrl-C, then:
./build/train_data --files books.txt --out run \
  --preset tiny --vocab 1024 --seq-len 256 --batch 4 --steps 1000 \
  --resume
```

[`finetune`](finetune.md) `--resume` keeps moments for a new file list and starts that run's schedule at step 1.

## New learning rate

`--lr-new` is used with `--resume`. It loads the same weights and AdamW moments, then runs a new warmup and cosine from step 1 through `--steps`. The peak is `--lr` and the floor is one tenth of that peak. Warmup is `--warmup`. A checkpoint that has already reached the old `--steps` is trained again. stderr prints `lr-new step 1 (AdamW step <N>)`, and the next stdout step is `1`.

The AdamW step stored in `model.bin` is not reset. It stays the bias-correction timestep and keeps incrementing. The next plain `--resume` still counts from that step. Pass `--lr-new` again to start another schedule. Combining `--lr-new` with a run that is not `--resume` fails with `lr-new`.

```bash
./build/train_data --files books.txt --out run \
  --preset tiny --vocab 1024 --seq-len 256 --batch 4 --steps 1000 \
  --resume --lr-new --lr 1e-4 --warmup 0
```

## Tokenize only

`--tokenize-only` trains the byte-level BPE tokenizer, encodes each file with eos appended, writes `tokenizer.bpe`, `tokens.bin`, and `tokens.meta`, and exits before the model. It does not create a training context and it does not write `model.bin`. An existing `model.bin` in `--out` is left in place. stderr prints `saved <tokenizer.bpe>` and then `tokenize only`. `--dir` reads the text from that directory. `--stdin` reads that text from standard input. See the directory and standard input sections.

The encoded stream still has to hold at least `--seq-len + 1` tokens, and `--seq-len` still has to fit the preset. `--preset` is checked and is not trained. `--steps`, `--early-stop`, `--min-loss`, and the sampling flags are not used. A later `--reuse-tokens` run trains a new model on this directory. Combining `--tokenize-only` with `--resume` or `--reuse-tokens` fails with `tokenize-only`.

```bash
./build/train_data --files books.txt --out run \
  --preset tiny --vocab 1024 --seq-len 256 \
  --tokenize-only

./build/train_data --files books.txt --out run \
  --preset tiny --vocab 1024 --seq-len 256 --batch 4 --steps 1000 \
  --reuse-tokens
```

## Restart on saved tokens

`--reuse-tokens` reads `tokenizer.bpe`, `tokens.bin`, and `tokens.meta` from `--out` and trains a new model. It does not load `model.bin`. The finished run replaces `model.bin`. The tokenizer is not trained again, and `--files` or `--dir` is not encoded again. `--files` or `--dir` is still required. When `--prompt` is omitted, the continuation prompt is the opening of the first file. `--vocab` has to match the tokenizer. `--preset` and the size flags choose the new model. The learning-rate schedule starts at step 1 under this command's `--lr`, `--warmup`, and `--steps`. `--seed` initializes that model and draws the batch windows. stderr prints `reusing <N> tokens`. Combining `--reuse-tokens` with `--resume` fails with `reuse-tokens`.

```bash
./build/train_data --files books.txt --out run \
  --preset tiny --vocab 1024 --seq-len 256 --batch 4 --steps 1000 \
  --reuse-tokens
```

## When it fails

Errors print `train_data: <reason>` on stderr and the process exits 1. Common reasons:

| Reason | Meaning |
|---|---|
| `files` | the list is missing, empty, or names a path that is not a file, or none of `--files`, `--dir`, and `--stdin` was passed |
| `dir` | `--dir` is combined with `--files`, the path is not a directory, or the directory has no regular files |
| `vocab_size` | `--vocab` is below 259, the corpus runs out of pairs before the vocab is full, or `--resume` or `--reuse-tokens` and `--vocab`, the token file, or the checkpoint disagrees with the tokenizer |
| `short corpus` | the encoded files have fewer than `--seq-len + 1` tokens |
| `short valid` | `--early-stop` is positive and the stream cannot be split into two spans of `--seq-len + 1` tokens |
| `layers` | `--layers` is below 1 |
| `d-model` | `--d-model` is below 1 |
| `heads` | `--heads` is below 1 |
| `d-ff` | `--d-ff` is below 1 |
| `max-seq` | `--max-seq` is below 1 |
| `kv-heads` | `--kv-heads` is below 0 |
| `head-dim` | `--head-dim` is below 0 |
| `num_heads` | `d_model` is not divisible by the query heads while head dim is 0 |
| `num_kv_heads` | the KV heads do not divide the query heads |
| `head_dim` | the head width is illegal |
| `use_rope` | a RoPE preset (`llama-*` or `qwen-*`) has an odd head dim |
| `d_ff` | the feed-forward width is illegal for this block |
| `seq_len` | `--seq-len` is above the context cap (`--max-seq`, or the preset cap: 1024, 2048 for `gpt3-*`, or 512 for `bert-*`) |
| `early_stop` | `--early-stop` is negative |
| `min_loss` | `--min-loss` is negative or infinite |
| `eval_every` | `--early-stop` is positive and `--eval-every` is not in `1 .. --steps` |
| `valid_fraction` | `--valid-fraction` is not inside `(0, 1)` |
| `preset` | the name is not one of the preset names in the flag table, or `--resume` and the preset plus size flags do not match the checkpoint |
| `resume` | `--resume` is set and `--out` is missing `tokenizer.bpe`, `tokens.bin`, `tokens.meta`, or `model.bin` |
| `lr-new` | `--lr-new` is set without `--resume` |
| `reuse-tokens` | `--reuse-tokens` is combined with `--resume`, or `--out` is missing `tokenizer.bpe`, `tokens.bin`, or `tokens.meta` |
| `tokenize-only` | `--tokenize-only` is combined with `--resume` or `--reuse-tokens` |
| `stdin` | `--stdin` is combined with `--files`, `--dir`, `--resume`, or `--reuse-tokens`, or standard input is empty or cannot be read |
| `adamw` | `--resume` found a checkpoint whose optimizer is not AdamW |
| `optimizer block missing` | `--resume` found a checkpoint with no optimizer |
| a flag name | the flag is unknown, or its value is missing or not a number |

Library details are in [USAGE.md](../USAGE.md). The one-paragraph acceptance run is [train_tiny](train_tiny.md). `--resume` continues this run. To keep training the checkpoint on more text, see [finetune](finetune.md).
