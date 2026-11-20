# How to converse with a trained model

`chat` is the dialogue example in [`chat.cpp`](chat.cpp). It loads `tokenizer.bpe` and `model.bin` from a directory written by [`train_books`](train_books.md) or [`gguf from`](gguf.md) and continues a `User:` / `Assistant:` transcript. `--plain` continues the line as a prompt with those tags left off, which matches a book checkpoint. It does not train. It is a separate program from `llm_tests`.

Build from the repository root. Requirements are in the [README](../README.md).

```bash
./build.sh opencl   # or: ./build.sh cuda
cmake --build build -j --target train_books
cmake --build build -j --target chat
```

## Train on exchanges

Put one exchange in each file. `train_books` appends eos after each file, and that token marks the end of the assistant turn. A trailing newline on the file matches the transcript `chat` stores.

`dialogues/bell.txt`:

```
User: What is the harbor bell for?
Assistant: It marks the hour and calls the boats home.
```

`dialogues/tide.txt`:

```
User: Who rings the harbor bell?
Assistant: The keeper rings it when the tide turns.
```

`dialogues.txt`:

```
bell.txt
tide.txt
```

Paths in the list are resolved from the list's directory.

```bash
./build/train_books --files dialogues.txt --out chat-run \
  --preset tiny --vocab 1024 --seq-len 256 --batch 4 --steps 1000
```

Two short exchanges and a few steps are enough to prove the loop. They are not enough for a reply that answers the question. A usable transcript model needs many exchanges and a full training run. The same tiny preset memorizes a small corpus rather than learning general conversation.

The same file list trains an encoder. `--seq-len` must be at most 512. `train_books` does not sample after this run. `chat` does. Each new token is the pad id.

```bash
./build/train_books --files dialogues.txt --out chat-bert \
  --preset bert-tiny --vocab 1024 --seq-len 256 --batch 4 --steps 1000
./build/chat --model chat-bert
```

[train_bert_chat](train_bert_chat.md) is the short in-process run of that objective.

`chat-run/` must contain `tokenizer.bpe` and `model.bin`. The other `train_books` files are not read by `chat`. A public llama.cpp GGUF, including Qwen2 and Qwen3, becomes that directory with `gguf from MODEL.GGUF chat-run`. A model trained with `--preset qwen-tiny`, `qwen-small`, or `qwen-base` loads the same way. See [gguf](gguf.md).

## Talk

```bash
./build/chat --model chat-run
```

The session looks like this. The terminal echoes the line you type after `you>`.

```
you> What is the harbor bell for?
model> It marks the hour and calls the boats home.
you> /quit
```

| Input | Effect |
|---|---|
| any other line | one assistant reply, then the prompt returns |
| `/reset` | clear the transcript and keep the model loaded |
| `/quit` | exit 0 |
| end of input | exit 0 |

An empty line is ignored. One leading space on the reply is hidden after `model>`. The transcript keeps that space, because the training text has a space after `Assistant:`.

`--once` prints a single reply and exits. The same sampling flags apply.

```bash
./build/chat --model chat-run --once "Who rings it?" --temperature 0 --top-k 1
```

```
model> <reply>
```

## A book checkpoint

A model trained on prose has no `User:` / `Assistant:` lines. `--plain` feeds the transcript then the line you typed, and prints the continuation as generated.

```bash
./build/chat --model run --plain
```

```
you> The morning after
model> <continuation>
you> /quit
```

`--once` works with `--plain`.

```bash
./build/chat --model run --plain --once "The morning after"
```

```
model> <continuation>
```

## How a turn is built

Completed turns are stored as `User: ...\nAssistant:...\n`. The next prompt is that transcript, then `User: `, the line you typed, a newline, and `Assistant:`. The model generates the reply, including the space after the colon.

Sampling stops on the tokenizer eos id. That id is not appended. If the decoded reply contains a newline followed by `User:`, the turn is cut there so the model cannot start the next user line.

With `--plain`, a completed turn is the line, the reply, and a newline. The next prompt is that transcript then the line. A decoded newline and `User:` does not end the reply. One leading space stays on screen.

Each call encodes the whole prompt. A causal model prefills it. A bidirectional model runs `forward` once per new token. Nothing is kept in the KV cache between turns.

Most presets use `max_seq_len` 1024. `gpt3-tiny`, `gpt3-small`, and `gpt3-base` use 2048. `bert-tiny`, `bert-small`, and `bert-base` are encoders with context 512. A causal checkpoint prefills the prompt and samples with `decode_step`. A bidirectional checkpoint appends the pad id, which is the mask, and `forward` scores that position. Segment ids stay 0. The sampling flags are the same, and eos stops the reply. [train_bert_chat](train_bert_chat.md) trains `bert_tiny` on four exchanges and prints one reply through that path. [train_chat](train_chat.md) trains a `tiny` model for one reply and a `bert_tiny` model for one masked token in the same process. The encoded prompt plus `--max-new-tokens` has to fit. `chat` drops the oldest completed turn until it does. A line that still does not fit prints `chat: max_seq_len` and leaves the transcript unchanged. In `--once` mode that same case exits 1.

## Flags

`--model` is required. `--help` prints this list and exits 0.

| Flag | Default | Role |
|---|---|---|
| `--model DIR` | | directory with `tokenizer.bpe` and `model.bin` |
| `--max-new-tokens N` | 128 | tokens to sample for the reply |
| `--temperature FLOAT` | 0.8 | `0` is greedy |
| `--top-k N` | 40 | `0` disables top-k |
| `--top-p FLOAT` | 0.9 | nucleus threshold |
| `--seed N` | 1 | sampling rng |
| `--once TEXT` | | one reply, then exit |
| `--plain` | | prompt is the line, with no `User:` / `Assistant:` tags |

## When it fails

Errors print `chat: <reason>` on stderr. In an interactive session the process stays up and prints `you>` again. `--once` exits 1.

| Reason | Meaning |
|---|---|
| `model` | `--model` is missing, or the directory has no `tokenizer.bpe` or no `model.bin` |
| `prompt` | `--once` was given an empty string |
| `max_seq_len` | the line plus `--max-new-tokens` is longer than the model allows |
| `mask_token` | a bidirectional checkpoint has a pad id outside the model vocabulary |
| `logits` | `forward` did not return one row per token |
| `max_new_tokens`, `temperature`, `top_k`, `top_p` | that flag is out of range |

Training flags stay on `train_books`. See [train_books](train_books.md) and [USAGE.md](../USAGE.md). To keep training a dialogue checkpoint on more exchanges, see [finetune](finetune.md).
