# bert-tiny, bert-small, and bert-base

`bert-tiny`, `bert-small`, and `bert-base` (`llm::presets::bert_tiny`, `bert_small`, and `bert_base`) are an encoder. The dense block matches GPT-2: learned absolute positions, LayerNorm epsilon `1e-5`, erf GELU, `d_ff = 4 * d_model`, and a bias on every linear. Attention is bidirectional. Context is 512. A segment table has 2 rows. A diagram of `bert_base` is in [bert-base-architecture.md](bert-base-architecture.md).

Train this shape from scratch. The preset does not load published BERT weights. There is no WordPiece tokenizer, no `<mask>` piece, and no next-sentence pooler. `Generator` and `chat` append the pad id and read that position from `forward`. `decode_step` and `prefill` throw `causal`.

## Sizes

| Preset | Layers | `d_model` | Heads | `d_ff` | Parameters at vocab 16000 |
|---|---:|---:|---:|---:|---:|
| bert_tiny | 4 | 256 | 4 | 1024 | 11,499,136 (11.50M) |
| bert_small | 6 | 512 | 8 | 2048 | 35,578,496 (35.58M) |
| bert_base | 12 | 768 | 12 | 3072 | 110,042,752 (110.04M) |

Head dim is 64. KV heads match the query heads. GELU's first linear writes `d_ff`. Dropout is 0. Embeddings are untied. `max_seq_len` is 512. `type_vocab_size` is 2.

`train_books` saves a version-6 checkpoint with AdamW moments. `save` without an optimizer writes version 5.

## Usage

```bash
./build.sh opencl   # or: ./build.sh cuda
cmake --build build -j --target train_books

./build/train_books --files books.txt --out run \
  --preset bert-tiny --vocab 1024 --seq-len 256 --batch 4 --steps 1000
```

`--preset` is `bert-tiny`, `bert-small`, or `bert-base`. `--seq-len` must be at most 512. `--vocab` must be at least 259. Memory matches `gpt2-*` at the same depth, and the context is already 512. `bert-tiny` fits in 12 GB. `bert-small` prefers 24 GB at batch 4 and length 512. `bert-base` is a 24 GB target. The notes are float32, one GPU, full attention, and no activation checkpointing.

`train_books` sets masked language modeling because the config is non-causal. The mask id is the tokenizer pad id. `encode` does not insert that id. Of the positions chosen for the loss (15% by default), 80% become the pad id, 10% become a random id, and 10% keep the original id. Every other position is labeled `-100`. After the save the process prints one line and exits 0:

```
encoder: masked lm, mask token <pad id>
```

`--prompt` and the sampling flags are unused. `--early-stop` and `--min-loss` still apply. `finetune` continues the masked-token objective and prints the same line.

`forward(tokens)` fills every segment id with 0. `forward(tokens, type_ids)` supplies the ids for a pair of segments. `decode_step` and `prefill` throw `causal`. [train_bert_chat](../examples/train_bert_chat.md) trains this encoder on four `User:` / `Assistant:` exchanges and prints one reply. [chat](../examples/chat.md) loads a `bert-*` directory from `train_books` and continues the transcript the same way. [train_chat](../examples/train_chat.md) trains this encoder beside a causal `tiny` model, samples one reply from `tiny`, and reads one masked position from `forward`.

`gguf export` writes an `nnllm` GGUF with `nnllm.causal` false and `nnllm.type_vocab_size` 2. `gguf import` of that file is checkpoint version 5 and drops the AdamW moments. `gguf llama` rejects the file because attention is bidirectional and a segment table is present. `finetune --resume` needs the original `model.bin`.

```cpp
nn::TransformerConfig config = llm::presets::bert_tiny(vocab_size);
nn::Transformer model(ctx, config, seed);

llm::TrainConfig cfg;
cfg.masked_lm = true;
cfg.mask_token = tokenizer.pad_id();
cfg.mask_prob = 0.15f;
cfg.seq_len = 256;  // at most config.max_seq_len (512)
```

`vocab_size` must be at least 1. `mask_token` must be inside the vocabulary when `masked_lm` is set. `bert_small` and `bert_base` take the same vocab argument.

## Benefits

- Every token attends to the whole window, up to 512 positions.
- The loss is the original id at the masked positions, so the step trains recovery of those tokens from both sides.
- The segment table is in the checkpoint. Single-sequence training uses segment 0, and `forward(tokens, type_ids)` is the hook for a second segment.
- The position table is 512 rows, so the parameter count sits just under the matching `gpt2-*` width. The 2-row segment table is included in that count.
- `finetune` keeps the masked-token objective, the pad id, and the AdamW moments when you pass `--resume`.

## Applications

- Masked-token training on a text list you have the rights to use, then more of the same through `finetune`.
- A caller that builds token ids, optionally fills segment ids, and reads logits from `forward`.
- A short in-process run, `train_bert_chat`, that trains `bert_tiny` on four exchanges and prints one reply.
- `chat` on a `bert-*` directory from `train_books`. Each new token is the pad id, scored by `forward`.
- A short in-process run, `train_chat`, that trains `bert_tiny` beside a causal `tiny` model and reads one masked position.
- An `nnllm` GGUF for a loader that accepts bidirectional attention and a segment table of size 2.
- A comparison with `gpt2-*` when you want the same dense block as a causal decoder at context 1024.

## See also

- [bert-base-architecture.md](bert-base-architecture.md) — diagram and specs for `bert_base`
- [preset-gpt2.md](preset-gpt2.md) — the same dense block as a causal decoder
- [preset-gpt3.md](preset-gpt3.md) — that decoder at context 2048
- [train_bert_chat](../examples/train_bert_chat.md) — train `bert_tiny` on four exchanges, then one reply
- [chat](../examples/chat.md) — continue a transcript, including a `bert-*` checkpoint
- [train_chat](../examples/train_chat.md) — train `tiny` and `bert_tiny` on one tokenizer, then one reply and one masked token
- [train_books](../examples/train_books.md) — the file-list run, including the encoder line
- [finetune](../examples/finetune.md) — continued masked-token training
- [README.md](../README.md) — preset table and parameter counts
- [USAGE.md](../USAGE.md) — library notes
