# How to run train_bert_chat

`train_bert_chat` is the example in [`train_bert_chat.cpp`](train_bert_chat.cpp). It trains `presets::bert_tiny` on one byte-level BPE tokenizer and four dialogue exchanges, saves `tokenizer.bpe` and `model.bin`, loads them the way [`chat`](chat.md) does, and prints one `User:` / `Assistant:` reply. The program takes no arguments. It is a separate program from `llm_tests`.

Build from the repository root. Requirements are in the [README](../README.md).

```bash
./build.sh opencl   # or: ./build.sh cuda
cmake --build build -j --target train_bert_chat
./build/train_bert_chat
```

For the CUDA backend, `CMAKE_CUDA_ARCHITECTURES` defaults to `86`. Set it to the GPU you have before `./build.sh cuda`.

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

Byte-level BPE is trained on those four strings with vocab size 320. The token ids stay in memory as a `TextDataset`. Fit logs, `tokenizer.bpe`, and `model.bin` go under the system temp directory in `llm-train-bert-chat/`.

## Training

The model is `presets::bert_tiny`. Attention is bidirectional. The objective is masked tokens: probability 0.15, and the mask id is the tokenizer pad id. Of the positions chosen for the loss, 80% become that id, 10% become a random id, and 10% keep the original id. Every other position is labeled `-100`.

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

The learning-rate column stays at `0.003`. After the log the program prints the first and last loss.

The saved file is a version-6 checkpoint with AdamW moments. The process then loads `tokenizer.bpe` and `model.bin` with `LoadMode::Inference`, which drops the moments. That is the pair `chat` reads.

The prompt is the first question, in the same shape `chat` uses:

```
User: What is the harbor bell for?
Assistant:
```

Sampling is greedy: temperature 0 and top-k 1. Each new token is the pad id. `forward` fills every segment id with 0 and the program reads the logits at that position. Eos stops the reply and is left off the string. A decoded newline and `User:` ends the printed reply. One leading space is hidden. The line is:

```
model> <reply>
```

A 30-step run on four exchanges is a shape check. The reply is whatever those weights produce. Pad, bos, and eos decode to no bytes, so `model>` is empty when every sampled id is one of those. A longer checkpoint is `train_books --preset bert-tiny` with one exchange per file, then [`chat`](chat.md) `--model` on that directory. [`finetune`](finetune.md) continues the masked-token objective. `decode_step` and `prefill` still throw `causal`.

A successful run exits 0. The last loss has to be finite and lower than the first loss.

The temp directory is removed before the process exits.

## When it fails

A failed run prints a reason on stderr and exits 1.

| Message | Meaning |
|---|---|
| `missing step loss` | the log has no step 1 line or no step 30 line |
| `loss did not fall` | the last loss is not lower than the first |
| `generate failed: ...` | saving, loading, or the masked reply threw |
| `train_bert_chat: ...` | tokenizer training, the dataset, or the fit loop threw |

`generate failed: mask_token` means the pad id is outside the model vocabulary. `generate failed: max_seq_len` means the prompt plus 16 new tokens does not fit in context 512.

Library details for the preset, the trainer, and the generator are in [USAGE.md](../USAGE.md). The preset page is [preset-bert](../docs/preset-bert.md).
