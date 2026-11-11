# How to run train_tiny

`train_tiny` is the acceptance example in [`train_tiny.cpp`](train_tiny.cpp). It trains a tiny transformer on one short paragraph, checks that the loss fell, and generates eight tokens. It takes no arguments. It is a separate program from `llm_tests`.

Build from the repository root. Requirements are in the [README](../README.md).

```bash
./build.sh
cmake --build build -j --target train_tiny
./build/train_tiny
```

## What it trains on

The corpus is this paragraph, compiled into the binary:

```
A tiny model can learn a short paragraph. The same words come back, so the next token is easier. Loss should fall after a few dozen steps.
```

Byte-level BPE is trained on that one string with vocab size 320. The token file is written under the system temp directory in `llm-train-tiny/`, then loaded as a `TextDataset`.

## Training

The run uses `presets::tiny` with learned absolute positions. The schedule is fixed:

| Setting | Value |
|---|---|
| Steps | 30 |
| Sequence length | 32 |
| Batch size | 4 |
| Learning rate | 3e-3, held constant |
| Warmup | 0 |
| Weight decay | 0.1 |
| Gradient accumulation | 1 |
| Eval | off |
| Checkpoints | off |

Every step is printed to stdout:

```
step 1 loss <loss> lr 0.003 grad_norm <norm>
```

The learning-rate column stays at `0.003`. After step 30 the program prints the first loss, the last loss, and one line of generated text:

```
first_loss <step 1 loss>
last_loss <step 30 loss>
<eight tokens, or fewer if eos is sampled>
```

The prompt is `A tiny model`. Sampling is greedy: temperature 0 and top-k 1. A successful run exits 0. The last loss has to be finite and lower than the first.

The temp directory is removed before the process exits. This example leaves no `tokenizer.bpe` and no `model.bin`. To keep a model, use [train_books](train_books.md).

## When it fails

A failed run prints a reason on stderr and exits 1.

| Message | Meaning |
|---|---|
| `missing step loss` | the log has no step 1 line or no step 30 line |
| `loss did not fall` | the last loss is not lower than the first |
| `generate failed: ...` | sampling threw |
| `train_tiny: ...` | tokenizer training, the dataset, or the fit loop threw |

Library details for the preset, the trainer, and the generator are in [USAGE.md](../USAGE.md).
