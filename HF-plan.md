# Hugging Face compatibility for llm

Load a Hugging Face model directory and run it in `chat` so that token ids match that model's tokenizer and the logits match the published graph, within float32 tolerance. The converter stays a host tool, like `examples/gguf.cpp`. It does not link `llm` or a device backend, and it does not wrap llama.cpp or PyTorch.

`nn` keeps adjacent-pair RoPE. The converter interleaves Q and K so that layout matches rotate-half checkpoints. Existing `model.bin` files stay valid.

## Models this covers

| `model_type` | Examples | What has to land first |
|---|---|---|
| `qwen3` | Qwen3 | Converter and byte-level BPE. The block is already version 3. |
| `llama` with no RoPE scaling and no attention bias | Llama 2 7B and 13B | Converter. Prompts wait on SentencePiece. 7B is a version-1 RoPE checkpoint (`rope_theta` 10000, RMSNorm eps `1e-5`, one KV head per query head). |
| `qwen2` | Qwen2, Qwen2.5 | QKV-only bias. Same byte-level BPE as Qwen3. |
| `llama` with `rope_scaling.rope_type = llama3` | Llama 3, Llama 3.1, Llama 3.2 | RoPE frequency table. Tokenizer is `tokenizer.json` BPE. |
| `mistral` | Mistral | SentencePiece plus a sliding-window mask. |
| `gpt2` | GPT-2 | Tanh GELU, Conv1D transpose, fused QKV split, LM-head bias off. |
| `bert` | BERT base, uncased and cased | Post-norm block, embedding LayerNorm at `1e-12`, no extra final norm, masked-LM head, WordPiece. |

Anything else stops with a named reason: MoE (`mixtral` and expert tensors), Gemma (residual RMSNorm, GeGLU, embedding scale, logit softcap), GPTQ/AWQ, unigram SentencePiece, and an unrecognized pre-tokenizer.

## Decisions

- **Ids stay in Hugging Face order.** LPB1 cannot represent these vocabs. `gguf from` remaps rows and then encodes with a different split, so that path is not how these models are loaded. `BpeTokenizer` stays the trainer for this library's own runs.
- **One new checkpoint generation.** Versions 1–6 are unchanged. A file that needs a new flag is version 7 with no optimizer, or version 8 with AdamW. Old readers reject the new magic version instead of misreading the tail.
- **Defaults keep today's bytes.** Version 7 is written only when a new flag differs from the historical value. A Qwen3 or Llama 2 import still writes version 1 or 3.
- **Four bias switches, not one.** `qkv_bias`, `attn_out_bias`, `mlp_bias`, and `lm_head_bias`. Versions 1–6 set all four from `TransformerConfig::bias`. Qwen2 sets only `qkv_bias`. GPT-2 sets every switch except `lm_head_bias`.
- **RoPE kernel reads a frequency table.** The default table is today's formula, `theta ** (-2i / head_dim)` on adjacent pairs, so current tests stay bit-identical. Llama 3 replaces that table. Partial RoPE rotates the first `rope_dims` coordinates and copies the rest.
- **No Jinja.** `chat --plain` takes a prompt the caller already wrapped. A fixed `--template qwen` or `--template llama3` applies that family's header. A general chat-template interpreter is out of scope.
- **Tied embeddings are copied.** `nn` still rejects `tie_embeddings`. Inference matches. A later `finetune` updates the token table and `lm_head` as two matrices.
- **Weights are float32 on disk.** Safetensors F32, F16, and BF16 widen while streaming. Quantized safetensors are rejected. There is no quantized matmul in this work. A 7B model is about 28 GB of weights plus the KV cache.
- **The same `nn` diff lands on every sibling in one change.** `checkpoint.cpp` is the contract shared by nnCUDA, nnCL, nnROCm, and nnArc. nnAGX gets the same patch when that tree is in the build. Kernels differ per backend: CUDA `.cu`, ROCm `.hip`, OpenCL `kernels.cl`, Arc SYCL, AGX Metal.

## New checkpoint fields (version 7 and 8)

Appended after the version-5 tail, then `pad8`:

| Field | Type | Default from versions 1–6 |
|---|---|---|
| `qkv_bias`, `attn_out_bias`, `mlp_bias`, `lm_head_bias` | four u32 | all equal to `bias` |
| `gelu_tanh` | u32 | 0 (erf, the current `gelu_forward`) |
| `rope_dims` | i32 | 0 means the whole head |
| `sliding_window` | i32 | 0 means a strict causal mask |
| `rope_scaling` | u32 | 0 none, 1 llama3 |
| llama3 `factor`, `low_freq_factor`, `high_freq_factor` | f32 | unused when scaling is 0 |
| `rope_original_context` | i32 | unused when scaling is 0 |
| `post_norm` | u32 | 0 |
| `embed_norm` | u32 | 0 |
| `embed_norm_eps` | f32 | `1e-5` |
| `final_norm` | u32 | 1 |
| `mlm_head` | u32 | 0 |

`TransformerConfig` and `TransformerBlockSpec` grow the same fields. `validate` rejects a llama3 table with a non-positive factor, a `rope_dims` that is odd or wider than the head, a negative window, and `mlm_head` on a causal model.

Parameter names when `mlm_head` is set, after `final_ln` is omitted: `mlm.dense.weight`, `mlm.dense.bias`, `mlm.ln.weight`, `mlm.ln.bias`, `mlm.decoder.weight`, `mlm.bias`. The decoder is a dense copy of the token table. It is not tied.

## Converter

New target `hf` in [`llm/examples/hf.cpp`](llm/examples/hf.cpp), same CMake shape as `gguf`. No `llm` and no `nn` link.

```text
hf import MODEL_DIR OUT_DIR [--context N]
hf self-test
```

`MODEL_DIR` is a snapshot directory: `config.json`, either `model.safetensors` or `model.safetensors.index.json`, and a tokenizer (`tokenizer.json` or `tokenizer.model`). `OUT_DIR` receives `model.bin`, the tokenizer files copied through, and `generation_config.json` when it exists.

Host pieces inside that file:

- A small JSON reader for objects, arrays, strings, numbers, and bools. Enough for `config.json` and the safetensors header.
- Safetensors: u64 header length, JSON header, then the payload. Read one tensor at a time. Shard map comes from `weight_map` in the index.
- Widen F16 and BF16 to F32. Reject other dtypes with `dtype`.
- Config map: `hidden_size` → `d_model`, `intermediate_size` → `d_ff`, `num_attention_heads`, `num_key_value_heads`, `head_dim` (0 when absent and `hidden_size` divides the heads), `rms_norm_eps`, `rope_theta`, `max_position_embeddings`, `vocab_size`, `hidden_act`, `attention_bias`, `partial_rotary_factor`, `sliding_window`, `rope_scaling`, `tie_word_embeddings`, `layer_norm_eps`, `model_type`.
- `--context N` overrides `max_seq_len`. The default is the file's context. Do not cap at 1024.

Tensor layout, Llama and Qwen names:

| Hugging Face | Checkpoint |
|---|---|
| `model.embed_tokens.weight` | `tok_emb.weight` |
| `model.layers.N.input_layernorm.weight` | `blocks.N.ln1.weight` |
| `q_proj`, `k_proj`, `v_proj`, `o_proj` | `attn.q/k/v/out.weight` |
| `q_proj.bias`, `k_proj.bias`, `v_proj.bias` | `attn.q/k/v.bias` when `attention_bias` |
| `q_norm.weight`, `k_norm.weight` | `attn.q_norm.weight`, `attn.k_norm.weight` |
| `post_attention_layernorm.weight` | `blocks.N.ln2.weight` |
| `mlp.gate_proj` then `mlp.up_proj` | `fc1.weight` rows `[0, d_ff)` then `[d_ff, 2*d_ff)` |
| `mlp.down_proj` | `fc2.weight` |
| `model.norm.weight` | `final_ln.weight` |
| `lm_head.weight`, or a copy of the token table | `lm_head.weight` |

`nn.Linear` weights are already `(out, in)`. Q and K (and their biases) are interleaved per head before they are written:

```text
(n_head, 2, head_dim/2, hidden) → swap axes 1 and 2 → (n_head, head_dim/2, 2, hidden)
```

For K, `n_head` is the KV head count. This is the permute llama.cpp applies so adjacent-pair RoPE matches rotate-half weights.

GPT-2 names use Conv1D storage `(in, out)`. Transpose, then split `c_attn` into Q, K, V along the output axis. `wpe` is `pos_emb.weight`. `wte` is `tok_emb.weight`. Missing `lm_head` copies `wte` and writes no LM-head bias. GELU MLP width is `n_inner` or `4 * n_embd`, one projection, not `2 * d_ff`.

BERT names: word, position, and token-type tables; embedding LayerNorm; per-layer query, key, value, output dense, intermediate, output dense; the two post-norms stored as the block norms. Skip `bert.pooler`. The MLM head is `cls.predictions.transform` plus `cls.predictions.decoder` plus `cls.predictions.bias`. `final_norm` is 0 because the last block already ends in LayerNorm.

Reject, with `hf: <reason>` and exit 1:

| Reason | When |
|---|---|
| `architecture` | `model_type` is not in the table above |
| `moe` | expert or router tensors |
| `quantized` | a dtype other than F32, F16, BF16 |
| `tensor` | missing, duplicate, or wrong shape |
| `rope` | a scaling type other than absent or `llama3` |
| `activation` | not `silu` for the decoder families, and not GELU for GPT-2 and BERT |
| `tokenizer` | the tokenizer file the family needs is absent |

`self-test` builds a tiny safetensors file in memory, imports it, and checks the permute, the gate/up pack, a BF16 widen, a tied `lm_head` copy, and the version byte. It does not need a GPU or a downloaded model.

## Tokenizer

`Tokenizer::load` keeps LPB1 as it is. A new `HfTokenizer` loads the copied Hugging Face files. `chat` looks for `tokenizer.json` or `tokenizer.model` when `tokenizer.bpe` is absent.

### Byte-level BPE (`tokenizer.json`)

Covers Qwen2, Qwen3, Llama 3, and GPT-2.

- Vocab is the JSON map, ids unchanged. Merges are the pair list, applied in order inside one pre-token piece.
- Specials and added tokens are matched longest-first on the raw string and never go through BPE.
- Pre-tokenizer is selected by the regex string in the file. Implement three scanners, using a Unicode letter/number table rather than ICU:
  - GPT-2: contractions, then letters, numbers, punctuation, whitespace.
  - Llama 3: the same shape, numbers in groups of at most three.
  - Qwen: the same shape, digits one at a time.
- An unrecognized regex, normalizer, or decoder fails with `tokenizer`. Do not silently fall back to the word/space split in `BpeTokenizer`.
- Byte-level decode uses the GPT-2 byte map, then the byte-level decoder. `BpeTokenizer::decode` stays on its own pieces.
- `add_bos_token` and `add_eos_token` from `tokenizer_config.json` are exposed as `add_bos_default()` and `add_eos_default()`. The LPB1 tokenizer returns false. `chat`, `finetune`, and generation pass those defaults. `EncodeOptions` still overrides them when the caller sets the flags.

### SentencePiece (`tokenizer.model`)

Covers Llama 2 and Mistral. Hand-parse the model protobuf far enough to read `pieces` (text, score, type). Accept SentencePiece BPE. Reject unigram with `tokenizer`. Pre-tokenization is the SentencePiece space-prefix rule (`▁`), not the byte-level regex. Byte fallback pieces follow the model.

### WordPiece (`vocab.txt`)

Covers BERT. Greedy longest match, `##` continuations, `[UNK]` when nothing matches. Uncased models lowercase and strip accents before the match. Specials are `[CLS]`, `[SEP]`, `[PAD]`, `[MASK]`, `[UNK]`.

### Stop ids

`GenerateConfig` gains `stop_ids`. Empty means "use `eos_id` only". `HfTokenizer::stop_ids()` reads `eos_token_id` from `generation_config.json`, which may be one int or a list. `Generator` stops when the sample is in that set and leaves those ids off the decoded string. `BpeTokenizer` is unchanged: one eos id.

## Graph work in nn

Each item is implemented in the four local siblings, with tests that fail if the default path changes.

1. **Per-projection bias.** `Linear` construction for Q, K, V, attention out, `fc1`, `fc2`, and `lm_head` takes its own flag. The checkpoint writer emits version 7 only when the flags differ from `bias`. A Qwen2 file has the three QKV biases and no other linear bias.

2. **RoPE table.** Build `inv_freq` on the host and pass it to the kernel. Llama 3 adjusts that table with `factor`, `low_freq_factor`, `high_freq_factor`, and `original_max_position_embeddings` before any token is scored, including short prompts. `rope_dims` shorter than the head rotates a prefix of each head. `gguf from` currently rejects a partial dimension; once the field exists it stores `rope.dimension_count` instead of rejecting it.

3. **Sliding window.** The causal mask hides keys older than `sliding_window` as well as future keys. Window 0 keeps the current triangle. The KV cache still stores the whole prompt; the mask is what changes.

4. **Tanh GELU.** Second device function, `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`, selected by `gelu_tanh`. Erf GELU remains the default. GPT-2 presets in this library stay on erf. Only an imported GPT-2 sets the flag.

5. **BERT block.** `post_norm` applies the residual and then the norm. `embed_norm` LayerNorms the sum of token, position, and segment embeddings with the file's epsilon (`1e-12` for BERT) before the first block. `final_norm` 0 skips the extra norm in front of the MLM head. The MLM head is dense, GELU, LayerNorm, decoder, bias. `forward` of that config returns vocab logits at every position. Masked-LM training already ignores `-100`. The imported head replaces the single `lm_head` linear.

`gguf export` / `import` learn the same fields under the `nnllm.` prefix so a version-7 checkpoint round-trips. `gguf llama` keeps rejecting GELU, a bias pattern llama.cpp cannot store, post-norm, and a segment table.

## chat and the other examples

- `chat --model DIR` loads `model.bin` plus whichever tokenizer is present. HF defaults supply bos and the stop list.
- `--template qwen` wraps the line as ChatML (`<|im_start|>user` / `<|im_start|>assistant`). `--template llama3` wraps it with the Llama 3 header and `<|eot_id|>`. `--plain` and the existing `User:` / `Assistant:` transcript stay the default, so directories this library trained do not change.
- `finetune` can keep training an imported checkpoint. Document the untied copy. `--resume` still needs an optimizer block the converter does not write, so the first finetune builds a fresh AdamW.
- Presets, `train_data`, and LPB1 training are untouched.

## Tests

Host tests, no download, no GPU:

- `hf self-test`: permute, pack, BF16, tied copy, version byte, and one rejection for `mixtral` and for a bad dtype.
- `tests/test_hf_tokenizer.cpp`: three checked-in fixtures (GPT-2 regex, Qwen regex, Llama 3 regex). Each fixture is a tiny vocab, the merges, and the ids for a handful of strings including a special token and a non-ASCII byte. The ids are frozen in the test. A fourth fixture covers SentencePiece BPE with `▁`, and a fifth covers uncased WordPiece.
- Version-7 round-trip in the nn checkpoint tests: QKV-only bias, llama3 factors, a window, tanh GELU, and a BERT-shaped header. A version-1 file still loads and still writes version 1.
- Kernel tests: one llama3 frequency value computed by hand, one partial-RoPE tail left unchanged, tanh GELU at `x = 1`, and a windowed mask that hides a key inside the context but outside the window.

A full logit comparison against upstream Transformers needs a downloaded model and a GPU. That stays a manual check described in `examples/hf.md`, not a ctest.

## Docs

- [`llm/examples/hf.md`](llm/examples/hf.md) for the command, the families, the rejections, and the memory note.
- [`llm/README.md`](llm/README.md) and [`llm/USAGE.md`](llm/USAGE.md): one short section and the `chat --template` flags.
- [`llm/docs/glossary.md`](llm/docs/glossary.md): version 7 and 8.
- nn README checkpoint section in each sibling: the new tail.

## Order

Each step is one reviewable change. Later steps depend on earlier ones. nn steps include all four local siblings.

1. **`hf import` for the current graph.** Safetensors, `config.json`, Q/K interleave, gate/up pack, BF16 widen, tied copy. Writes version 1 or 3. Accepts a Qwen3 or Llama 2 shape and rejects everything else, including a missing tokenizer match that this step does not load yet. Weights only.
2. **Byte-level `HfTokenizer` and stop ids.** `chat` loads `tokenizer.json`. Qwen3 can be imported and prompted. Llama 3 weights still wait on the RoPE table; the tokenizer can land first.
3. **Per-projection bias in nn, then the converter writes it.** Qwen2 and Qwen2.5 import. `gguf from` accepts QKV bias on a `qwen2` file and writes version 7.
4. **RoPE frequency table in nn, then llama3 metadata in the converter.** Llama 3 weights import. Partial `rope_dims` is the same kernel change.
5. **SentencePiece BPE.** Llama 2 and Mistral prompts encode. Mistral logits still wait on the window.
6. **Sliding-window mask.** Mistral imports. A Qwen2 file that sets `sliding_window` uses it too.
7. **Tanh GELU and the GPT-2 layout.** Conv1D transpose, fused QKV, position table, copied `wte`, `lm_head_bias` off. GPT-2 prompts already use the tokenizer from step 2.
8. **BERT block and WordPiece.** Post-norm, embedding LayerNorm, MLM head, uncased normalizer. `chat` keeps scoring the pad id on a bidirectional checkpoint.
9. **`--template` and the docs pass.** Fixed Qwen and Llama 3 wrappers. Glossary and `hf.md`.

## Out of scope

MoE, Gemma and Gemma 2, vision or multimodal checkpoints, quantized matmul, unigram SentencePiece, a Jinja engine, and true tied embeddings. `gguf from` remains the GGUF path. It is not rewritten into the safetensors reader. Finetune of a copied tied head is allowed and is not numerically the same as a tied update.


