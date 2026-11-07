# How to train from a list of books

`train_books` is the file-list example in [`train_books.cpp`](train_books.cpp). It reads a list of text files, trains a byte-level BPE tokenizer and a preset transformer, writes the tokenizer and the checkpoint, then loads both and prints a continuation. It is a separate program from `llm_tests`.

Build from the repository root. Requirements are in the [README](../README.md).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
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

Progress is printed on stderr: each file and its size, the vocab and byte count, the token count, and the paths that were saved. Training steps are printed on stdout:

```
step 10 loss <loss> lr <lr> grad_norm <norm>
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

With `--steps` of 500 or more, `model.bin` is also written every 500 steps. The final save always runs after the last step. `--steps` below 500 writes the checkpoint once, at the end.

Load `tokenizer.bpe` and `model.bin` together. `chat` does that for a dialogue corpus. See [chat](chat.md).

## Flags

`--files` and `--out` are required. `--help` prints this list and exits 0.

| Flag | Default | Role |
|---|---|---|
| `--preset tiny\|small\|base` | `tiny` | model shape |
| `--vocab N` | 4096 | BPE size, minimum 259 |
| `--seq-len N` | 256 | tokens per example, at most 1024 |
| `--batch N` | 4 | examples per step |
| `--steps N` | 1000 | AdamW updates |
| `--accum N` | 1 | micro-batches per update |
| `--warmup N` | 100 | linear warmup steps |
| `--log-every N` | 10 | print every Nth step |
| `--lr FLOAT` | 3e-4 | peak learning rate |
| `--prompt TEXT` | opening of the first file | text to continue |
| `--max-new-tokens N` | 80 | tokens to sample after training |
| `--temperature FLOAT` | 0.8 | `0` is greedy |
| `--top-k N` | 40 | `0` disables top-k |
| `--top-p FLOAT` | 0.9 | nucleus threshold |
| `--seed N` | 1 | model init and batch order |

Weight decay stays 0.1. The cosine floor is one tenth of `--lr`. Eval is off. Defaults `--preset tiny`, `--seq-len 256`, and `--batch 4` fit a 12 GB GPU. `small` prefers 24 GB. `base` is a 24 GB model. On a smaller card keep `--seq-len` at 512 or below.

## When it fails

Errors print `train_books: <reason>` on stderr and the process exits 1. Common reasons:

| Reason | Meaning |
|---|---|
| `files` | the list is missing, empty, or names a path that is not a file |
| `vocab_size` | `--vocab` is below 259, or the corpus runs out of pairs before the vocab is full |
| `short corpus` | the encoded files have fewer than `--seq-len + 1` tokens |
| `seq_len` | `--seq-len` is above 1024 |
| `preset` | the name is not `tiny`, `small`, or `base` |
| a flag name | the flag is unknown, or its value is missing or not a number |

Library details are in [USAGE.md](../USAGE.md). The one-paragraph acceptance run is [train_tiny](train_tiny.md).
