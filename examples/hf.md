# How to convert a Hugging Face snapshot

`hf` reads a Hugging Face model directory and writes a transformer `model.bin`. The source is [`hf.cpp`](hf.cpp). It does not link `llm` or a device backend. Configuring the tree still selects `nn`, the same way as the other targets.

```bash
./build.sh cuda       # or opencl, amd, arc
cmake --build build -j --target hf
./build/hf import snapshot out
./build/hf import snapshot out --context 2048
./build/hf self-test
```

`import` reads `config.json` and either `model.safetensors` or `model.safetensors.index.json`. F32, F16, and BF16 tensors widen to float32. The written checkpoint has no optimizer. `LoadMode::Inference` can load it.

This step accepts three families, the ones whose block this library already computes:

| `model_type` | Checkpoint | What is kept |
|---|---|---|
| `qwen3` | version 3 | grouped-query attention, a head dim other than `d_model / heads`, QK-norm, the file's RoPE base, and its RMSNorm epsilon |
| `qwen2` | version 7 when `attention_bias` is true, otherwise version 1 or 3 | the three Q, K, and V biases, and no other linear bias |
| `llama` with no `rope_scaling`, no attention bias, and no sliding window | version 1 when the heads are square and the RoPE base is 10000 | separate Q, K, and V, RMSNorm, SwiGLU |

Q and K weights are interleaved per head. Hugging Face stores them for rotate-half RoPE. This runtime rotates adjacent pairs, so the weights are rearranged before they are written. A Qwen2 Q or K bias is interleaved the same way. The V bias is copied as stored. `gate_proj` and `up_proj` are packed into `fc1`, gate in the first `d_ff` rows. A tied token table is copied onto `lm_head`. The file still records `tie_embeddings` as false, because this runtime rejects a tied embedding. A later finetune updates those two matrices separately.

`--context N` sets `max_seq_len`. The default is `max_position_embeddings`.

`qwen3` and `qwen2` require `tokenizer.json` in the snapshot. `llama` requires `tokenizer.model`. The converter copies those files, plus `tokenizer_config.json`, `generation_config.json`, `special_tokens_map.json`, and `vocab.txt` when they are present. It does not parse them. `chat` loads `tokenizer.json` from the output directory and keeps the Hugging Face ids. A Llama 2 directory still needs `tokenizer.model`, which `chat` does not read yet.

A float32 checkpoint of a 7B model is about 28 GB plus the KV cache. The converter keeps one tensor in flight, then writes the whole `model.bin`.

## What this step rejects

Errors print `hf: <reason>` on stderr and the process exits 1.

| Reason | Meaning |
|---|---|
| `architecture` | `model_type` is not `qwen3`, `qwen2`, or `llama` |
| `activation` | `hidden_act` is not `silu` |
| `rope` | `rope_scaling` is set, `partial_rotary_factor` is not 1, or the head dim is odd |
| `bias` | `qwen3` or `llama` has `attention_bias` or any linear bias; `qwen2` has a linear bias other than `q_proj`, `k_proj`, and `v_proj`, or those tensors are present while `attention_bias` is false |
| `window` | a sliding window is on |
| `tokenizer` | the family's tokenizer file is missing |
| `dtype` | a tensor is not F32, F16, or BF16 |
| `moe` | an expert or router tensor is present |
| `tensor` | a weight is missing, duplicated, or the wrong shape |
| `config` | a required hyperparameter is missing or not a positive integer |
| `context` | `--context` is missing or not a positive integer |
| `read` or `write` | a path could not be opened |
| `json` | `config.json` or a safetensors header is not the JSON this tool reads |

Llama 3 RoPE scaling, Mistral, GPT-2, and BERT stop here. Their graphs are not in the checkpoint this tool writes yet. Qwen2 with `attention_bias` writes version 7 and the three QKV biases.

## Check the converter

`hf self-test` builds a tiny sharded Qwen3 snapshot, a single-file Llama snapshot, and a Qwen2 snapshot with QKV bias. It checks the Q/K interleave, the gate/up pack, a BF16 widen, an F16 widen, a tied `lm_head` copy, the version byte, and that the Qwen2 file is version 7 with interleaved Q and K biases and a straight V bias. It also checks that Mixtral, Llama 3 scaling, attention bias on Qwen3 and Llama, a Qwen2 output bias, a missing QKV bias, a sliding window, a missing tokenizer, a bad dtype, and an expert tensor are rejected. It does not need a GPU or a downloaded model.

```bash
./build/hf self-test
```
