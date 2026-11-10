# How to fine-tune an existing model

`finetune` is the continued-training example in [`finetune.cpp`](finetune.cpp). It loads `tokenizer.bpe` and `model.bin` from a directory written by [`train_books`](train_books.md) or by an earlier `finetune` run, encodes a new file list with that tokenizer, and keeps training. It writes a new directory and prints a continuation. It is a separate program from `llm_tests`.

The checkpoint's vocab and model shape stay as they are. `finetune` reads this library's `model.bin` and `tokenizer.bpe`.

Build from the repository root. Requirements are in the [README](../README.md).

```bash
./build.sh opencl   # or: ./build.sh cuda
cmake --build build -j --target finetune
```

## Prepare the new text

The list has the same shape as a `train_books` list. Blank lines are ignored. A line whose first character is `#` is ignored. A relative path is resolved from the directory that contains the list.

```
# more text for the harbor model
notes/fog.txt
notes/pier.txt
```

Each file is read in full and encoded with the existing tokenizer. An eos token is appended so a file boundary is a token. The tokenizer is not trained again. A corpus shorter than `--seq-len + 1` tokens fails with `finetune: short corpus`.

Use text you have the rights to train on.

## Run

`--from` is read only. `--out` has to be a different directory. The new directory receives `tokenizer.bpe`, `tokens.bin`, `tokens.meta`, and `model.bin`, so [`chat`](chat.md) can load `--out` the same way it loads a `train_books` directory.

```bash
./build/finetune --from run --files more.txt --out run-ft \
  --seq-len 256 --batch 4 --steps 200 --lr 1e-4 \
  --prompt "The morning after"
```

The default peak learning rate is `1e-4`. Warmup defaults to 0. The cosine floor is one tenth of `--lr`. The schedule starts at step 1 of this run. With `--warmup 0` that first step is already on the cosine, and step `--steps` is the floor. The schedule does not continue from the checkpoint's optimizer step.

Without `--resume`, the run builds a fresh AdamW at `--lr` and `--wd` (default weight decay `0.1`). With `--resume`, AdamW moments and the optimizer step stored in the checkpoint are kept, and `--wd` is ignored. The learning-rate flags still apply.

```bash
./build/finetune --from run --files more.txt --out run-ft --resume --lr 1e-4
```

Progress on stderr names each file, the token count, the loaded vocab, `max_seq_len`, layers, and `d_model`, then either `fresh AdamW` or `resuming AdamW step N`. Steps print on stdout. The process then drops the training context, loads `--out` with `LoadMode::Inference`, and prints:

```
prompt: The morning after
continuation: <new text>
```

The default prompt is the opening of the first new file: leading whitespace is skipped, then up to 48 Unicode code points, and a newline ends the excerpt early. Generation stops on eos.

With `--steps` of 100 or more, `model.bin` is also written every 100 steps. The final save always runs after the last step.

## Flags

`--from`, `--files`, and `--out` are required. `--help` prints this list and exits 0.

| Flag | Default | Role |
|---|---|---|
| `--from DIR` | | checkpoint directory, read only |
| `--files LIST` | | new text, one path per line |
| `--out DIR` | | new directory, different from `--from` |
| `--seq-len N` | 256 | tokens per example, at most the checkpoint maximum |
| `--batch N` | 4 | examples per step |
| `--steps N` | 200 | AdamW updates |
| `--accum N` | 1 | micro-batches per update |
| `--warmup N` | 0 | linear warmup steps |
| `--log-every N` | 10 | print every Nth step |
| `--lr FLOAT` | 1e-4 | peak learning rate for this run |
| `--wd FLOAT` | 0.1 | weight decay for a fresh AdamW |
| `--resume` | off | keep checkpoint AdamW moments |
| `--prompt TEXT` | opening of the first new file | text to continue |
| `--max-new-tokens N` | 80 | tokens to sample after training |
| `--temperature FLOAT` | 0.8 | `0` is greedy |
| `--top-k N` | 40 | `0` disables top-k |
| `--top-p FLOAT` | 0.9 | nucleus threshold |
| `--seed N` | 1 | batch order and sampling |

`--seq-len 256` and `--batch 4` match the `train_books` tiny preset defaults. A `small` or `base` checkpoint still prefers the GPU sizes in [train_books](train_books.md).

## When it fails

Errors print `finetune: <reason>` on stderr and the process exits 1.

| Reason | Meaning |
|---|---|
| `from` | `--from` is missing, or it has no `tokenizer.bpe` or no `model.bin` |
| `files` | the list is missing, empty, or names a path that is not a file |
| `out` | `--out` is missing or is the same directory as `--from` |
| `short corpus` | the encoded files have fewer than `--seq-len + 1` tokens |
| `vocab_size` | the tokenizer vocab differs from the checkpoint vocab |
| `seq_len` | `--seq-len` is above the checkpoint `max_seq_len` |
| `adamw` | `--resume` found a checkpoint whose optimizer is not AdamW |
| a flag name | the flag is unknown, or its value is missing or not a number |

A checkpoint whose tensors do not match the model built from its header fails with `checkpoint tensor mismatch`. Library details are in [USAGE.md](../USAGE.md).
