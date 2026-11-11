# How to train from a list of books

`train_books` is the file-list example in [`train_books.cpp`](train_books.cpp). It reads a list of text files, trains a byte-level BPE tokenizer and a preset transformer, writes the tokenizer and the checkpoint, then loads both and prints a continuation. `--early-stop` can hold out a suffix of the tokens and stop when that validation loss stops falling. `--min-loss` can stop when a step's training loss reaches a threshold. It is a separate program from `llm_tests`.

Build from the repository root. Requirements are in the [README](../README.md).

```bash
./build.sh opencl   # or: ./build.sh cuda
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

A first run with a smaller vocab spends less time in the tokenizer. Tokenizer training scans the corpus once per merge, so `--vocab 4096` on long books is the slow part. The GPU steps come after that.

```bash
./build/train_books --files books.txt --out run \
  --preset tiny --vocab 1024 --seq-len 256 --batch 4 --steps 1000 \
  --prompt "The morning after"
```

Progress is printed on stderr: each file and its size, the vocab and byte count, the token count, and the paths that were saved. With `--early-stop` the stderr line `holdout train=<n> valid=<n> early_stop=<n> eval_every=<n>` follows the token count. Training steps are printed on stdout:

```
step 10 loss <loss> lr <lr> grad_norm <norm>
```

When `--min-loss` trips, stdout also prints:

```
min_loss step <step> loss <loss>
```

`<step>` is the update that reached the threshold, and `<loss>` is that step's training loss. With `--early-stop`, validation lines are printed on the same stream:

```
step 50 eval_ce <loss> perplexity <exp(loss)>
early_stop step <last> best_step <best> best_eval_ce <loss>
```

When training finishes, stdout ends with:

```
prompt: The morning after
continuation: <new text>
```

The default prompt, when `--prompt` is omitted, is the opening of the first file: leading whitespace is skipped, then up to 48 Unicode code points are taken, and a newline ends the excerpt early. Generation stops when eos is sampled. That id is not part of the continuation.

## Files written

`run/` receives four files.

| File | Contents |
|---|---|
| `tokenizer.bpe` | byte-level BPE, magic `LPB1` |
| `tokens.bin` | token ids, int32 little-endian |
| `tokens.meta` | vocab size and length for `TextDataset` |
| `model.bin` | version-2 façade checkpoint, including AdamW moments |

With `--steps` of 500 or more, `model.bin` is also written every 500 steps. The final save always runs after the last step. `--steps` below 500 writes the checkpoint once, at the end. With `--early-stop`, that final file is the best validation step, including that step's AdamW moments, even when a 500-step write was a later step. When `--min-loss` stops the run, `model.bin` is the step whose training loss reached the threshold, including that step's AdamW moments.

Load `tokenizer.bpe` and `model.bin` together. `chat` does that for a dialogue corpus. See [chat](chat.md).

## Flags

`--files` and `--out` are required. `--help` prints this list and exits 0.

| Flag | Default | Role |
|---|---|---|
| `--preset tiny\|small\|base\|llama-tiny\|llama-small\|llama-base` | `tiny` | model shape |
| `--vocab N` | 4096 | BPE size, minimum 259 |
| `--seq-len N` | 256 | tokens per example, at most 1024 |
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
| `--seed N` | 1 | model init and batch order |
| `--gpu N` | 0 | CUDA device, or the OpenCL device this build accepts |

Weight decay stays 0.1. The cosine floor is one tenth of `--lr`. Eval is off unless `--early-stop` is positive. Defaults `--preset tiny`, `--seq-len 256`, and `--batch 4` fit a 12 GB GPU. `small` prefers 24 GB. `base` is a 24 GB model. On a smaller card keep `--seq-len` at 512 or below.

`--gpu N` is the device index. `0` is the default, and the chosen name is printed on stderr before the books are read. A CUDA build uses CUDA device N. An OpenCL build uses the Nth device that build accepts: `NNCL_OPENCL=auto` counts GPUs whose work-group size is at least 256, and `intel` can count the CPU device when no Intel GPU is listed. A negative index, or an index past that list, fails with `gpu`.

`tiny`, `small`, and `base` use learned absolute positions and LayerNorm. `llama-tiny`, `llama-small`, and `llama-base` are the same widths with RoPE and RMSNorm, which is the block llama.cpp runs. Train one of those from scratch. A checkpoint that was trained with learned positions does not become that block.

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

## When it fails

Errors print `train_books: <reason>` on stderr and the process exits 1. Common reasons:

| Reason | Meaning |
|---|---|
| `files` | the list is missing, empty, or names a path that is not a file |
| `vocab_size` | `--vocab` is below 259, or the corpus runs out of pairs before the vocab is full |
| `short corpus` | the encoded files have fewer than `--seq-len + 1` tokens |
| `short valid` | `--early-stop` is positive and the stream cannot be split into two spans of `--seq-len + 1` tokens |
| `seq_len` | `--seq-len` is above 1024 |
| `early_stop` | `--early-stop` is negative |
| `min_loss` | `--min-loss` is negative or infinite |
| `eval_every` | `--early-stop` is positive and `--eval-every` is not in `1 .. --steps` |
| `valid_fraction` | `--valid-fraction` is not inside `(0, 1)` |
| `preset` | the name is not `tiny`, `small`, `base`, `llama-tiny`, `llama-small`, or `llama-base` |
| `gpu` | `--gpu` is negative, or it is past the devices this build can see |
| a flag name | the flag is unknown, or its value is missing or not a number |

Library details are in [USAGE.md](../USAGE.md). The one-paragraph acceptance run is [train_tiny](train_tiny.md). To keep training that checkpoint on more text, see [finetune](finetune.md).
