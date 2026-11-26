# How to converse with a trained model

`chat` is the dialogue example in [`chat.cpp`](chat.cpp). It loads `model.bin` and a tokenizer from a directory written by [`train_data`](train_data.md), [`gguf from`](gguf.md), or [`hf import`](hf.md), and continues a `User:` / `Assistant:` transcript. `--plain` continues the line as a prompt with those tags left off, which matches a book checkpoint. `--template qwen` wraps each turn as ChatML for a Qwen directory from `gguf from`. It does not train. It is a separate program from `llm_tests`.

Build from the repository root. Requirements are in the [README](../README.md).

```bash
./build.sh amd       # or: ./build.sh opencl, ./build.sh cuda, ./build.sh arc, ./build.sh agx
cmake --build build -j --target train_data
cmake --build build -j --target chat
```

## Train on exchanges

Put one exchange in each file. `train_data` appends eos after each file, and that token marks the end of the assistant turn. A trailing newline on the file matches the transcript `chat` stores.

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
./build/train_data --files dialogues.txt --out chat-run \
  --preset tiny --vocab 1024 --seq-len 256 --batch 4 --steps 1000
```

Two short exchanges and a few steps are enough to prove the loop. They are not enough for a reply that answers the question. A usable transcript model needs many exchanges and a full training run. The same tiny preset memorizes a small corpus rather than learning general conversation.

The same file list trains an encoder. `--seq-len` must be at most 512. `train_data` does not sample after this run. `chat` does. Each new token is the pad id.

```bash
./build/train_data --files dialogues.txt --out chat-bert \
  --preset bert-tiny --vocab 1024 --seq-len 256 --batch 4 --steps 1000
./build/chat --model chat-bert
```

[train_bert_chat](train_bert_chat.md) is the short in-process run of that objective.

`chat-run/` must contain `model.bin` and `tokenizer.bpe`. The other `train_data` files are not read by `chat`. A public llama.cpp GGUF, including Qwen2 and Qwen3, becomes that directory with `gguf from MODEL.GGUF chat-run`. A model trained with `--preset qwen-tiny`, `qwen-small`, or `qwen-base` loads the same way. See [gguf](gguf.md).

An `hf import` directory is loaded when it has `model.bin` and `tokenizer.json`. `tokenizer.bpe` wins when both files are present. The Hugging Face ids stay as they are. `tokenizer_config.json` supplies the default bos and eos flags, and `generation_config.json` supplies the stop ids. `add_bos_token` prepends bos on each prompt. Sampling stops on every id in `eos_token_id`, which may be one integer or a list, and that id is left off the reply. A directory that has only `tokenizer.model` is not a chat directory yet.

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

Sampling stops on the tokenizer eos id. A Hugging Face directory with `eos_token_id` stops on each id in that list instead. The stopping id is not appended. If the decoded reply contains a newline followed by `User:`, the turn is cut there so the model cannot start the next user line.

`--repeat-penalty` defaults to 1, which leaves the logits unchanged. A value above 1 is applied before temperature, including a greedy pick, once per distinct id in the prompt and the reply so far. A positive logit is divided by the penalty. Any other logit is multiplied by it.

With `--plain`, a completed turn is the line, the reply, and a newline. The next prompt is that transcript then the line. A decoded newline and `User:` does not end the reply. One leading space stays on screen.

`--template qwen` writes a completed turn as `<|im_start|>user`, a newline, the line, `<|im_end|>`, a newline, `<|im_start|>assistant`, a newline, the reply, and `<|im_end|>`. The next prompt ends at `<|im_start|>assistant` and a newline, with no reply yet. The reply is cut at `<|im_end|>` or at a newline followed by `<|im_start|>`. One leading space stays on screen. `--plain` and `--template qwen` together fail with `template`. There is no Jinja template.

Each call encodes the whole prompt. A causal model prefills it. A bidirectional model runs `forward` once per new token. Nothing is kept in the KV cache between turns.

Most presets use `max_seq_len` 1024. Every `gpt3-*` preset uses 2048. Every `bert-*` preset is an encoder with context 512. A causal checkpoint prefills the prompt and samples with `decode_step`. A bidirectional checkpoint appends the pad id, which is the mask, and `forward` scores that position. Segment ids stay 0. The sampling flags are the same, and eos stops the reply. [train_bert_chat](train_bert_chat.md) trains `bert_tiny` on four exchanges and prints one reply through that path. [train_chat](train_chat.md) trains a `tiny` model for one reply and a `bert_tiny` model for one masked token in the same process. The encoded prompt plus `--max-new-tokens` has to fit. `chat` drops the oldest completed turn until it does. A line that still does not fit prints `chat: max_seq_len` and leaves the transcript unchanged. In `--once` mode that same case exits 1.

## Flags

`--model` is required. `--help` prints this list and exits 0.

| Flag | Default | Role |
|---|---|---|
| `--model DIR` | | directory with `model.bin` and `tokenizer.bpe`, or `model.bin` and `tokenizer.json` |
| `--max-new-tokens N` | 128 | tokens to sample for the reply |
| `--temperature FLOAT` | 0.8 | `0` is greedy |
| `--top-k N` | 40 | `0` disables top-k |
| `--top-p FLOAT` | 0.9 | nucleus threshold |
| `--repeat-penalty FLOAT` | 1 | `1` leaves logits unchanged. Above 1 discourages ids already in the prompt and the reply |
| `--seed N` | 1 | sampling rng |
| `--once TEXT` | | one reply, then exit |
| `--plain` | | prompt is the line, with no `User:` / `Assistant:` tags |
| `--template qwen` | | ChatML `<|im_start|>user` / `<|im_start|>assistant` turns |

## When it fails

Errors print `chat: <reason>` on stderr. In an interactive session the process stays up and prints `you>` again. `--once` exits 1.

| Reason | Meaning |
|---|---|
| `model` | `--model` is missing, `model.bin` is missing, or the directory has neither `tokenizer.bpe` nor `tokenizer.json` |
| `tokenizer` | `tokenizer.json` is not a byte-level BPE this build accepts |
| `template` | `--template` is not `qwen`, or it is combined with `--plain` |
| `prompt` | `--once` was given an empty string |
| `max_seq_len` | the line plus `--max-new-tokens` is longer than the model allows |
| `mask_token` | a bidirectional checkpoint has a pad id outside the model vocabulary |
| `logits` | `forward` did not return one row per token |
| `max_new_tokens`, `temperature`, `top_k`, `top_p`, `repetition_penalty` | that flag is out of range |

Training flags stay on `train_data`. See [train_data](train_data.md) and [USAGE.md](../USAGE.md). To keep training a dialogue checkpoint on more exchanges, see [finetune](finetune.md).
