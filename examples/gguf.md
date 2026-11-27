# How to convert a checkpoint to GGUF

`gguf` converts a transformer `model.bin` to a [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) version 3 file and converts that file back. It can write a llama.cpp file from the same checkpoint, and it can turn a public llama.cpp GGUF into a directory `chat` loads. The source is [`gguf.cpp`](gguf.cpp). It does not link `llm` or a device backend, and it is a separate program from `llm_tests`. Configuring the tree still selects `nn`, `nnCL`, `nnROCm`, `nnArc`, or `nnAGX`, the same way as the other targets.

```bash
./build.sh amd       # or: ./build.sh opencl, ./build.sh cuda, ./build.sh arc, ./build.sh agx
cmake --build build -j --target gguf
./build/gguf export run/model.bin run/model.gguf
./build/gguf import run/model.gguf run-again/model.bin
./build/gguf llama run/model.bin run/model-llama.gguf
./build/gguf llama run/model.bin run/model-llama.gguf run/tokenizer.bpe
./build/gguf from model.gguf chat-run
./build/chat --model chat-run
```

`export` reads a family-6 checkpoint written by `train_data`, `finetune`, or `Transformer::save`. It keeps the float32 weights and writes architecture `nnllm`. `import` reads that file and writes a checkpoint with the weights and no optimizer block. A classic block stays version 1. A Qwen-style block (grouped-query attention, a head dim other than `d_model / heads`, QK-norm, a RoPE base other than 10000, or an RMSNorm epsilon other than `1e-5`) is version 3. `LoadMode::Inference` accepts that file. A later `finetune` run builds a fresh AdamW from it. `finetune --resume` needs the original checkpoint, because AdamW moments are not in the GGUF.

An `export` or `import` file does not contain `tokenizer.bpe`. Keep that file next to the checkpoint you train or chat with. `gguf llama` embeds a native `tokenizer.bpe` or a Qwen `tokenizer.json` when it sits beside the checkpoint or you pass its path.

## A public GGUF to a chat directory

`gguf from MODEL.GGUF OUTDIR [--context N]` reads a llama.cpp GGUF and writes a directory `chat` loads. Architecture `nnllm` uses `import` for the weights and copies a sibling `tokenizer.bpe`. Architecture must be `llama`, `qwen2`, `qwen3`, or `mistral`. Anything else, including `qwen35`, stops with `architecture` before the weights are read. Those four are mapped onto the RoPE / RMSNorm / SwiGLU block.

When `tokenizer.ggml.pre` is `qwen2` or `qwen35`, token ids stay in GGUF order. `vocab_size` is the length of `tokenizer.ggml.tokens`. `token_embd` and `output` rows are copied straight. The converter writes `tokenizer.json`, `tokenizer_config.json`, and `generation_config.json`. It does not write `tokenizer.bpe`, `tokens.bin`, or `tokens.meta`. `qwen2` selects the Qwen scanner, one digit at a time. `qwen35` selects that scanner with combining marks kept in the letter run. Control and user-defined pieces become `added_tokens`. `chat --template qwen` wraps the prompt as ChatML.

Any other `tokenizer.ggml.pre`, including `superbpe`, is remapped to this library's LPB1 layout: ids `0..255` are raw bytes, later ids are merges in learn order, and the last three ids are pad, bos, and eos. Extra special tokens are dropped. Embedding and `lm_head` rows follow that map. A missing GPT-2 byte piece, including the Qwen hole at byte 173 (`ń`), still throws `tokenizer` on that path. `tokens.bin` is the encoded string `User: hello\nAssistant: hello\n` plus eos.

Quantized tensors (float16, bfloat16, Q8_0, Q8_1, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K, Q6_K, Q8_K) expand to float32. The written `model.bin` is version 1, version 3 when the block is Qwen-style, or version 7 when Q, K, and V have a bias and the other linears do not. Grouped-query attention keeps the native K and V width, `num_kv_heads * head_dim`. A head dim other than `d_model / heads` is stored as that width, so Q is `(num_heads * head_dim, d_model)`. QK-norm weights are kept. The file's `rope.freq_base` and `attention.layer_norm_rms_epsilon` are stored in the checkpoint. A base other than 10000 prints that fact and still converts. Tied embeddings copy `token_embd` onto `lm_head`. `--context N` sets `max_seq_len`. The default is `min(source context, 1024)`.

Partial RoPE (`rope.dimension_count` other than the key length), MoE expert tensors, a fused QKV matrix, GELU, attention-output bias, MLP bias, an LM-head bias, and a partial set of Q, K, and V biases stop the conversion. The Qwen Q/K projection weights, Q/K biases, and Q/K norm scales are reordered for adjacent-pair RoPE. SentencePiece tokenizers stop it too. GPT-2 BPE is accepted. `qwen2` and `qwen35` keep the published ids. Other pre names still need every GPT-2 byte piece.

## The nnllm architecture

`nnllm` is the architecture name inside a standard GGUF version 3 file. It is not a second container. `general.architecture` is the string `nnllm`, and the hyperparameters use the `nnllm.` prefix.

A checkpoint from this library is a pre-norm transformer. Q, K, and V are separate matrices. Learned absolute positions use LayerNorm and a position table. RoPE uses RMSNorm and rotates Q and K, with no position table and no norm bias. GGUF carries the bytes. `nnllm` tells a reader which of those two blocks they belong to. `llama.cpp` does not implement `nnllm`, so it will not run an `export` file. `gguf import` is the reader that turns one back into `model.bin`. The [`llama` command](#a-llamacpp-file) writes a second file that llama.cpp can load.

The file is little-endian. `general.alignment` is 32. `export` writes float32, so `general.file_type` is 0. `import` also accepts `nnllm` tensors stored as float16 (`general.file_type` 1) and expands them to float32.

| Key | Meaning |
|---|---|
| `nnllm.vocab_size` | vocabulary size |
| `nnllm.embedding_length` | `d_model` |
| `nnllm.block_count` | layer count |
| `nnllm.feed_forward_length` | `d_ff`, the width after the SwiGLU gate |
| `nnllm.context_length` | `max_seq_len` |
| `nnllm.attention.head_count` | head count |
| `nnllm.attention.head_count_kv` | KV heads, only when the block is Qwen-style |
| `nnllm.attention.key_length` | head dim, only when the block is Qwen-style |
| `nnllm.attention.value_length` | the same head dim |
| `nnllm.qk_norm` | whether Q and K have an RMSNorm of width `head_dim` |
| `nnllm.rope.freq_base` | RoPE base, `1e6` for Qwen |
| `nnllm.attention.layer_norm_rms_epsilon` | RMSNorm epsilon, `1e-6` for Qwen |
| `nnllm.dropout` | dropout stored in the checkpoint |
| `nnllm.bias` | whether every linear follows one bias flag. `nnllm.qkv_bias`, `nnllm.attn_out_bias`, `nnllm.mlp_bias`, and `nnllm.lm_head_bias` are written only when that projection differs. A missing key follows `nnllm.bias` |
| `nnllm.tie_embeddings` | always false |
| `nnllm.activation` | `swiglu` or `gelu` |
| `nnllm.positional` | `learned_absolute` or `rope` |

Integer counts are `uint32`. `nnllm.dropout` is `float32`. The two flags are GGUF bools. Activation and position encoding are strings.

## Tensors

Names match the checkpoint, in parameter order:

| Tensor | Shape |
|---|---|
| `tok_emb.weight` | `(vocab, d_model)` |
| `pos_emb.weight` | `(max_seq_len, d_model)` |
| `blocks.N.ln1.weight`, `blocks.N.ln1.bias` | `(d_model)` |
| `blocks.N.attn.q.weight` | `(num_heads * head_dim, d_model)`. Square when `head_dim` is `d_model / heads` |
| `blocks.N.attn.k.weight`, `blocks.N.attn.v.weight` | `(num_kv_heads * head_dim, d_model)`. Square when there is one KV head per query head and the head is `d_model / heads` |
| `blocks.N.attn.out.weight` | `(d_model, num_heads * head_dim)` |
| `blocks.N.attn.q_norm.weight`, `blocks.N.attn.k_norm.weight` | `(head_dim)`, only when QK-norm is on, after `out` |
| `blocks.N.ln2.weight`, `blocks.N.ln2.bias` | `(d_model)` |
| `blocks.N.fc1.weight` | `(2 * d_ff, d_model)` for SwiGLU, `(d_ff, d_model)` for GELU |
| `blocks.N.fc2.weight` | `(d_model, d_ff)` |
| `final_ln.weight`, `final_ln.bias` | `(d_model)` |
| `lm_head.weight` | `(vocab, d_model)` |

`N` is the block index starting at 0. A learned-absolute file has a position table and a bias on every norm. A RoPE file omits `pos_emb.weight` and every norm bias. A classic file also omits the Qwen keys above, and its Q, K, V, and out matrices stay `(d_model, d_model)`. When `nnllm.bias` is true, each linear name also has a `.bias` whose length is the output size. With SwiGLU, `fc1.weight` holds the gate and the up projection in that one matrix. The float values stay in this library's row-major order. The GGUF dimension list stores the contiguous dimension first, so a weight stored here as `(out, in)` is recorded as dimensions `(in, out)`.

The GGUF does not contain `tokenizer.bpe` or the AdamW moments. `import` writes version 1, version 3 for a Qwen-style block, or version 5 when attention is bidirectional or a segment table is present. `nnllm.causal` is stored only when it is false. `nnllm.type_vocab_size` is stored only when it is positive, so an older file still imports. `LoadMode::Inference` loads it, and a later `finetune` starts a fresh AdamW. `finetune --resume` needs the original `model.bin`.

## A llama.cpp file

`gguf llama MODEL MODEL.GGUF [TOKENIZER.BPE|TOKENIZER.JSON]` reads a family-6 checkpoint or an `nnllm` GGUF and writes float32. A classic block is architecture `llama`. A Qwen-style block is `qwen3` when QK-norm is on, and `qwen2` otherwise. The hyperparameter prefix matches that architecture. `MODEL` is `model.bin` or the file from `gguf export`. Without a tokenizer path, the tool looks beside `MODEL` for `tokenizer.bpe`. For Qwen it also looks for `tokenizer.json`. Bidirectional attention and a segment table are rejected with `causal` before the GELU and bias checks. A `bert-*` checkpoint stays in `model.bin` or an nnllm GGUF.

| nnllm tensor | llama.cpp tensor |
|---|---|
| `tok_emb.weight` | `token_embd.weight` |
| `blocks.N.ln1.weight` | `blk.N.attn_norm.weight` |
| `blocks.N.attn.q.weight` | `blk.N.attn_q.weight`, shape `(num_heads * head_dim, d_model)` |
| `blocks.N.attn.k.weight`, `blocks.N.attn.v.weight` | `blk.N.attn_k.weight`, `attn_v`, shape `(num_kv_heads * head_dim, d_model)` |
| `blocks.N.attn.out.weight` | `blk.N.attn_output.weight` |
| `blocks.N.attn.q.bias`, `k.bias`, `v.bias` | `blk.N.attn_q.bias`, `attn_k.bias`, `attn_v.bias` for Qwen2 checkpoints with QKV bias |
| `blocks.N.attn.q_norm.weight`, `blocks.N.attn.k_norm.weight` | `blk.N.attn_q_norm.weight`, `attn_k_norm`, only when QK-norm is on |
| `blocks.N.ln2.weight` | `blk.N.ffn_norm.weight` |
| `blocks.N.fc1.weight` rows `[0, d_ff)` | `blk.N.ffn_gate.weight` |
| `blocks.N.fc1.weight` rows `[d_ff, 2 * d_ff)` | `blk.N.ffn_up.weight` |
| `blocks.N.fc2.weight` | `blk.N.ffn_down.weight` |
| `final_ln.weight` | `output_norm.weight` |
| `lm_head.weight` | `output.weight` |

A learned-absolute checkpoint leaves out `pos_emb.weight` and every LayerNorm bias. A RoPE checkpoint has neither tensor, so nothing is dropped. The file does not contain rope frequency tensors or expert tensors. For Qwen, the exporter puts Q/K projection rows, Q/K biases, and Q/K norm scales in the split-half order llama.cpp uses. V keeps its original order.

| Key | Value |
|---|---|
| `general.architecture` | `llama`, or `qwen2` / `qwen3` for a Qwen-style block |
| `general.name` | Output filename stem for Qwen, `nnllm` for Llama |
| `general.alignment` | 32 |
| `general.file_type` | 0, float32 |
| `general.description` | Checkpoint conversion description |
| `tokenizer.ggml.model` | `gpt2` when a BPE tokenizer is embedded, otherwise `none` |
| `tokenizer.ggml.pre` | `superbpe` for LPB1, or the Qwen pre-tokenizer from JSON |
| `tokenizer.ggml.add_bos_token`, `tokenizer.ggml.add_eos_token` | False for LPB1, otherwise set from tokenizer configuration |
| `tokenizer.ggml.tokens` | Byte-level BPE pieces and special tokens in checkpoint ID order |
| `tokenizer.ggml.merges` | `"left right"` in learn order |
| `tokenizer.ggml.token_type` | Normal, control, user-defined, or unused |
| `tokenizer.ggml.bos_token_id`, `tokenizer.ggml.eos_token_id`, `tokenizer.ggml.padding_token_id` | IDs from the embedded tokenizer, when defined |
| `tokenizer.chat_template` | Qwen chat template, when provided |
| `llama.vocab_size` | vocabulary size |
| `llama.context_length` | `max_seq_len` |
| `llama.embedding_length` | `d_model` |
| `llama.block_count` | layer count |
| `llama.feed_forward_length` | `d_ff` |
| `llama.attention.head_count` | head count. The prefix is `qwen2` or `qwen3` when that is the architecture |
| `llama.attention.head_count_kv` | KV heads. A classic file uses the query head count |
| `llama.attention.key_length` | head dim |
| `llama.attention.value_length` | the same head dim |
| `llama.attention.layer_norm_rms_epsilon` | `1e-5` for a classic file, the checkpoint epsilon otherwise |
| `llama.rope.freq_base` | `10000` for a classic file, the checkpoint base otherwise |
| `llama.rope.dimension_count` | the head dim. A classic file uses `d_model / heads` |

Counts are `uint32`. The epsilon and the RoPE base are `float32`. The epsilon is the LayerNorm epsilon from this library. llama.cpp still uses it inside RMSNorm. Weights are stored in row-major order. The GGUF dimension list stores the contiguous dimension first, so `token_embd.weight` is recorded as `(d_model, vocab)`, `ffn_gate.weight` and `ffn_up.weight` as `(d_model, d_ff)`, and `ffn_down.weight` as `(d_ff, d_model)`.

llama.cpp loads a classic file and runs it as a LLaMA block: adjacent-pair RoPE on Q and K, base 10000, and RMSNorm with epsilon `1e-5`. A checkpoint trained with a `llama-*` preset uses that same block, so the graph matches. A `qwen-*` checkpoint is written as `qwen3`, with the wider head, grouped-query K and V, QK-norm, RoPE base `1e6`, and RMSNorm epsilon `1e-6`. A checkpoint trained with learned positions and LayerNorm does not match the RoPE block llama.cpp applies. The tokens from that file differ.

When a `tokenizer.bpe` is embedded, llama.cpp can encode a text prompt. The vocab is this library's byte-level BPE: ids 0 through 255 are raw bytes, later ids are merges, and the last three ids are pad, bos, and eos. Pieces are stored with llama.cpp's GPT-2 byte map so the BPE ranks line up. `add_bos_token` and `add_eos_token` stay false, matching `train_data`, which appends eos only while it is building the training spans. The pre-tokenizer is `superbpe`. It keeps ordinary text, including newlines, as one piece, and it isolates each run of digits. `BpeTokenizer` instead keeps letters, digits, and bytes above 127 in one run, and it splits whitespace and punctuation into their own runs. A merge that crosses a space, a punctuation mark, or a digit boundary therefore gets a different id than `BpeTokenizer::encode`, and the ids match again after that boundary. The eos piece is `<|endoftext|>`, which llama.cpp treats as an end-of-generation token.

Without a tokenizer, the export writes `tokenizer.ggml.model=none` and llama.cpp cannot encode a text prompt.

`gguf import` still accepts only architecture `nnllm`. A GELU checkpoint stops with `gguf: activation`. Qwen2 QKV-only bias is supported. Other linear bias patterns stop with `gguf: bias`.

On success with a learned-position checkpoint the program prints a warning on stderr and exits 0. A RoPE checkpoint exits 0 without that warning:

```text
gguf: llama.cpp applies RoPE and RMSNorm. This checkpoint was trained with learned positions and LayerNorm. The file loads and the tokens differ. LayerNorm biases and the position table are omitted.
```

## Share a Qwen GGUF on Hugging Face

From the `llm` directory, export a trained `qwen-*` checkpoint or a directory produced by [`hf import`](hf.md).

```bash
cmake --build build -j --target gguf
./build/gguf llama run/model.bin My-Qwen-F32.gguf
llama-completion -m My-Qwen-F32.gguf -no-cnv -p 'Hello' -n 16
```

For a Hugging Face tokenizer, keep `tokenizer_config.json`, `generation_config.json`, and `chat_template.jinja` beside `tokenizer.json` when the source provides them. The GGUF keeps the original token IDs, BPE merge order, special tokens, and supplied chat template. Unassigned model vocabulary rows are marked unused. For parity with Hugging Face tokenizers that apply NFC, normalize decomposed Unicode input to NFC before passing it to llama.cpp. Re-import earlier Hugging Face Qwen3 conversions before export to pick up the Q/K normalization layout fix. For checkpoints made by an older `gguf from`, re-import the original upstream Qwen GGUF to pick up the corresponding layout fix. GGUF files previously exported by this tool can still be read by `gguf from`.

After creating a model repository on the Hub, upload the verified file. Add a model card describing the training data, license, prompt format, and any base model. [Hugging Face GGUF guide](https://huggingface.co/docs/hub/gguf) · [Uploading models](https://huggingface.co/docs/hub/models-uploading)

```bash
hf upload YOUR_ACCOUNT/My-Qwen-GGUF My-Qwen-F32.gguf
```

## What this tool rejects

`import` accepts an `nnllm` file whose positions are `learned_absolute` or `rope`. It rejects a GGUF whose `general.architecture` is `llama`, `gpt2`, or anything other than `nnllm` with `gguf: architecture`. Quantized tensor types on that path are rejected with `gguf: quantized`. `llama.cpp` does not implement the `nnllm` architecture, so it will not run an `export` file.

`from` accepts architecture `nnllm` when a sibling `tokenizer.bpe` is present, and accepts `llama`, `qwen2`, `qwen3`, and `mistral` when the tensors match the RoPE / RMSNorm / SwiGLU block. Any other architecture stops with `architecture` before a weight is read. It expands the quantized types listed above. A SentencePiece tokenizer, partial RoPE, MoE, a fused QKV matrix, GELU, attention-output bias, MLP bias, an LM-head bias, and a partial QKV bias set stop that conversion. QKV bias uses checkpoint version 7. Qwen Q/K biases are reordered with their projection rows. Pre `qwen2` and `qwen35` keep GGUF ids and write `tokenizer.json`. Other pre names are remapped to LPB1.

## Check the converter

`gguf self-test` builds a small checkpoint, writes GGUF, reads it back, and checks a float16 round trip. It also writes the llama.cpp file from both the checkpoint and the `nnllm` GGUF, checks the `fc1` split, and checks that GELU and unsupported linear biases are rejected by `llama`. It also checks Qwen split-half RoPE order and round-trips embedded HF tokenizer metadata. It checks a version-7 QKV-bias round trip through `nnllm` and through `from`, and checks that a partial QKV set, an attention-output bias, an MLP bias, and an LM-head bias are rejected. It writes a tiny `qwen2` GGUF whose vocab omits `ń` and whose pre name is `qwen2`, checks that `from` keeps row 0, writes version 7, writes `tokenizer.json` with the Qwen regex and the ChatML specials, and does not write `tokenizer.bpe`. It checks that a `qwen35` header with an `ssm_conv1d` tensor exits `architecture`. It writes a tiny bidirectional checkpoint with a segment table, checks that the file is version 5 and that `type_emb.weight` reloads, and checks that `gguf llama` rejects it with `causal`. It embeds a 260-id `tokenizer.bpe` with one merge and checks the GPT-2 token list. `from` round-trips that llama.cpp file, and a small Qwen3 file, into a chat directory. The test checks Q8_0 dequant, the GQA repeat helper, and GPT-2 byte remapping. It does not need a GPU.

```bash
./build/gguf self-test
```

## When it fails

Errors print `gguf: <reason>` on stderr and the process exits 1.

| Reason | Meaning |
|---|---|
| `magic` | the file is not a checkpoint or a GGUF |
| `version` | the checkpoint version is outside 1..8, or the GGUF version is not 3 |
| `family` | the checkpoint is not a transformer |
| `architecture` | `import` saw a `general.architecture` other than `nnllm`, or `from` saw one other than `nnllm`, `llama`, `qwen2`, `qwen3`, or `mistral` |
| `quantized` | `import` saw a tensor that is neither float32 nor float16, or `from` saw an unsupported quant type |
| `activation` | the activation string is unknown, or `gguf llama` was given GELU |
| `tensor <name>` | a weight is missing, duplicated, or the wrong shape |
| `tie_embeddings` | the config asks for a tied embedding, which this runtime rejects |
| `positional` | positions are not `learned_absolute` or `rope` |
| `tokenizer` | `tokenizer.bpe` is not an LPB1 file, a piece is duplicated, a special name is already a piece, `from` has no GPT-2 BPE with merges, or a `qwen2` / `qwen35` file is missing token types, bos, eos, or pad |
| `vocab` | the `tokenizer.bpe` vocab size does not match the checkpoint |
| `qk_norm` | a layer is missing `attn_q_norm` or `attn_k_norm`, or one of those tensors is present when the model has no QK-norm |
| `head_dim` | the key and value lengths differ, or the length is missing and `d_model` is not divisible by the head count |
| `rope_dim` | `rope.dimension_count` is not the key length |
| `head_count_kv` | KV heads do not divide query heads |
| `gelu` | `from` did not find `ffn_gate.weight` |
| `bias` | `from` saw a partial QKV bias, an attention-output bias, an MLP bias, or an LM-head bias, or `gguf llama` was given a linear bias pattern other than Qwen2 QKV-only bias |
| `causal` | `gguf llama` was given bidirectional attention or a segment table |
| `type_vocab` | a segment-table size is negative |
| `moe` | `from` saw expert tensors |
| `qkv` | `from` saw a fused QKV matrix |
| `context` | `--context` is missing or not a positive integer |
| `use_rope` | a RoPE file has an odd head dimension |
| `read` or `write` | the path could not be opened |
