# How to run train_chat

`train_chat` is the example in [`train_chat.cpp`](train_chat.cpp). It trains two models on one byte-level BPE tokenizer and four dialogue exchanges. `presets::tiny` is the causal model, and the program samples one `User:` / `Assistant:` reply from it. `presets::bert_tiny` is the encoder, and the program replaces one token with the pad id and prints the argmax piece from `forward`. Each model has its own weights. The program takes no arguments. It is a separate program from `llm_tests`.

Build from the repository root. Requirements are in the [README](../README.md).

```bash
./build.sh amd       # or: ./build.sh opencl, ./build.sh cuda, ./build.sh arc
cmake --build build -j --target train_chat
./build/train_chat
```

For the CUDA backend, `CMAKE_CUDA_ARCHITECTURES` defaults to `86`. Set it to the GPU you have before `./build.sh cuda`. For ROCm, `CMAKE_HIP_ARCHITECTURES` defaults to `gfx1152`. Set it before `./build.sh amd`. For Arc, `./build.sh arc` uses `icpx` and JIT SPIR-V. There is no architecture flag. For Apple AGX, `./build.sh agx` uses Metal and exits on a non-Apple host.

## What it trains on

The corpus is these four exchanges, compiled into the binary. Each exchange is its own BPE string. Encoding appends eos, so each exchange ends on that token.

```
User: What is the harbor bell for?
Assistant: It marks the hour.

User: Where does the bell stand?
Assistant: The bell stands on the harbor wall.

User: When does the harbor bell ring?
Assistant: The harbor bell rings at noon.

User: Who rings the harbor bell?
Assistant: The keeper rings it when the tide turns.
```

Byte-level BPE is trained on those four strings with vocab size 320. The token ids stay in memory as a `TextDataset`. Fit logs go under the system temp directory in `llm-train-chat/`.

## Training

Both models use the same schedule. The tiny model trains next-token prediction. The BERT model trains masked tokens: probability 0.15, and the mask id is the tokenizer pad id.

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

The learning-rate column stays at `0.003`. The program prints `chat` before the tiny log and `bert` before the encoder log. After each log it prints that model's first and last loss.

The chat prompt is the first question, in the same shape `chat` uses:

```
User: What is the harbor bell for?
Assistant:
```

Sampling is greedy: temperature 0 and top-k 1. Eos stops the reply and is left off the string. A decoded newline and `User:` ends the printed reply. One leading space is hidden. The line is:

```
chat> <reply>
```

The masked sentence is `The harbor bell rings at noon.` Token index 2 is replaced with the pad id. `forward` fills every segment id with 0. The program prints the original piece at that index, the argmax id, and `decode` of that id:

```
bert_mask_at 2
bert_original <piece>
bert_guess_id <id>
bert_guess <piece>
```

Pad, bos, and eos decode to no bytes, so `bert_guess` is empty when that id wins.

A 30-step run on four exchanges is a shape check. The reply and the guess are whatever those weights produce. A longer chat checkpoint is a causal `train_data` run, then [`chat`](chat.md). A longer BERT checkpoint is `train_data --preset bert-tiny`. [`finetune`](finetune.md) continues either checkpoint.

A successful run exits 0. Each last loss has to be finite and lower than that model's first loss.

The temp directory is removed before the process exits. This example leaves no `tokenizer.bpe` and no `model.bin`.

## When it fails

A failed run prints a reason on stderr and exits 1.

| Message | Meaning |
|---|---|
| `missing chat step loss` | the tiny log has no step 1 line or no step 30 line |
| `missing bert step loss` | the encoder log has no step 1 line or no step 30 line |
| `chat loss did not fall` | the tiny model's last loss is not lower than its first |
| `bert loss did not fall` | the encoder's last loss is not lower than its first |
| `generate failed: ...` | sampling the tiny model threw |
| `train_chat: ...` | tokenizer training, the dataset, the fit loop, or the masked forward threw |

This program constructs `Generator` with the tiny model only. The encoder is scored with `forward` in `print_mask`. [train_bert_chat](train_bert_chat.md) is the run that prints a reply from `bert_tiny`.

Library details for the presets, the trainer, and the generator are in [USAGE.md](../USAGE.md). The preset pages are [preset-default](../docs/preset-default.md) and [preset-bert](../docs/preset-bert.md).
