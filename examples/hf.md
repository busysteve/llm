# How to convert a Hugging Face snapshot

`hf` reads a Hugging Face model directory and writes a transformer `model.bin`. The source is [`hf.cpp`](hf.cpp). It does not link `llm` or a device backend. Configuring the tree still selects `nn`, the same way as the other targets.

```bash
./build.sh cuda       # or opencl, amd, arc, agx
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
| `qwen2` | version 7 when QKV bias is present, otherwise version 1 or 3 | the three Q, K, and V biases, and no other linear bias |
| `llama` with no `rope_scaling`, no attention bias, and no sliding window | version 1 when the heads are square and the RoPE base is 10000 | separate Q, K, and V, RMSNorm, SwiGLU |

Q and K weights are interleaved per head. Hugging Face stores them for rotate-half RoPE. This runtime rotates adjacent pairs, so the weights are rearranged before they are written. Qwen3 Q/K normalization scales and Qwen2 Q/K biases are interleaved the same way. The V bias is copied as stored. `gate_proj` and `up_proj` are packed into `fc1`, gate in the first `d_ff` rows. A tied token table is copied onto `lm_head`. The file still records `tie_embeddings` as false, because this runtime rejects a tied embedding. A later finetune updates those two matrices separately.

`--context N` sets `max_seq_len`. The default is `max_position_embeddings`.

`qwen3` and `qwen2` require `tokenizer.json` in the snapshot. `llama` requires `tokenizer.model`. The converter copies those files, plus `tokenizer_config.json`, `generation_config.json`, `special_tokens_map.json`, and `vocab.txt` and `chat_template.jinja` when they are present. It does not parse them. `chat` loads `tokenizer.json` from the output directory and keeps the Hugging Face ids. A Llama 2 directory still needs `tokenizer.model`, which `chat` does not read yet.

A float32 checkpoint of a 7B model is about 28 GB plus the KV cache. The converter keeps one tensor in flight, then writes the whole `model.bin`.

## Run Qwen from a pinned snapshot

Run these commands from the `llm` directory. `--context` sets the imported checkpoint's context limit. The C++ converter and runtime use the local files.

```bash
hf download Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca --local-dir qwen3-hf
hf download Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775 --local-dir qwen25-hf
./build/hf import qwen3-hf qwen3-run --context 2048
./build/hf import qwen25-hf qwen25-run --context 2048
./build/chat --model qwen3-run --template qwen3 --thinking off --once 'Hello'
./build/chat --model qwen25-run --template qwen2.5 --once 'Hello'
```

Use `--plain` for completion. The Qwen templates cover user and assistant text turns. Qwen3 thinking mode defaults to on. `finetune` and `reinforce` copy the tokenizer and its metadata into their output. The first finetune starts a fresh AdamW. Subsequent runs can resume from its saved optimizer state. See [finetune](finetune.md), [reinforce](reinforce.md), and [GGUF export](gguf.md#share-a-qwen-gguf-on-hugging-face).

### Compare against Transformers

Build `llm_tests` on an available backend. In a Python environment with PyTorch, NumPy and Transformers 4.51.3, run the comparison with the downloaded snapshots.

```bash
cmake --build build -j --target llm_tests
python tests/verify_qwen.py qwen3-hf qwen3-run
python tests/verify_qwen.py qwen25-hf qwen25-run
```

The comparison checks Unicode and ChatML token IDs, both chat history lengths, generation stop IDs, prefill logits, and three cached decode steps. It uses float32 activations where the backend exposes that setting. Raw logits must match at absolute tolerance `1e-3` and relative tolerance `1e-4`. Greedy predictions must match over tokenizer-defined IDs. The official snapshots are required for this comparison.

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

Llama 3 RoPE scaling, Mistral, GPT-2, and BERT stop here. Their graphs are not in the checkpoint this tool writes yet. For Qwen2, the converter detects QKV bias tensors when `attention_bias` is absent. A complete set writes version 7. See [Share a Qwen GGUF on Hugging Face](gguf.md#share-a-qwen-gguf-on-hugging-face) to export the result.

## Check the converter

`hf self-test` builds a tiny sharded Qwen3 snapshot, a single-file Llama snapshot, and a Qwen2 snapshot with QKV bias. It checks the Q/K interleave, the gate/up pack, a BF16 widen, an F16 widen, a tied `lm_head` copy, the version byte, and that the Qwen2 file is version 7 with interleaved Q and K biases and a straight V bias. It also checks that Mixtral, Llama 3 scaling, attention bias on Qwen3 and Llama, a Qwen2 output bias, a missing QKV bias, a sliding window, a missing tokenizer, a bad dtype, and an expert tensor are rejected. It does not need a GPU or a downloaded model.

```bash
./build/hf self-test
```
