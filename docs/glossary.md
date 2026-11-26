# Glossary

Terms, acronyms, and processes for the models in this repository, the neural net they are built from, and the compute they run on. A name in `code` font is an identifier in `llm` or `nn`. Byte layouts and call sequences are in [USAGE.md](../USAGE.md). The source layout is in [code-structure.md](code-structure.md). Preset shapes are in the other files in this directory.

## Processes

- Text becomes ids in [pretokenization](#pretokenization), [BPE](#bpe), [encode](#encode), and [decode (tokenizer)](#decode-tokenizer).
- A supervised run is a [training step](#training-step): [forward](#forward), [cross-entropy](#cross-entropy), [backpropagation](#backpropagation), [gradient clipping](#gradient-clipping), and an [AdamW](#adamw) update. [Masked language modeling](#masked-language-modeling) replaces the labels first. [Validation](#validation) and [early stopping](#early-stopping) watch a held-out loss.
- A causal model continues text by [prefill](#prefill), [decode_step](#decode_step), and [sampling](#sampling).
- A reward run is a [reinforcement-learning step](#reinforcement-learning-step).
- Weights leave the process through a [checkpoint](#checkpoint) or a [GGUF](#gguf) file.

## A

### Activation

The number a layer writes for each position. Inside a block that is the residual stream of width [d_model](#d_model). At the end of the network the activation is the [logits](#logits). On nnCUDA, nnCL, and nnROCm, retained activations use the context's [dtype](#dtype). The default there is [bfloat16](#bfloat16). nnArc keeps activations in [float32](#float32).

### Activation checkpointing

Dropping stored activations during the [forward](#forward) and computing them again on the [backward](#backpropagation). The GPU notes in the README assume this is off, and they assume [full attention](#full-attention) on one [GPU](#gpu).

### AdamW

The optimizer `Trainer` and `RlTrainer` step. It keeps a first moment and a second moment for every parameter, then applies [weight decay](#weight-decay) to the weights themselves. Moments and the step count are stored in an even [checkpoint](#checkpoint) version. `llm::adamw_groups` assigns the decay: rank-2 tensors take it, and biases, LayerNorm parameters, and every other rank-1 tensor take 0.

### Advantage

The scalar that multiplies the token [log-probability](#log-softmax) in a [policy gradient](#policy-gradient). `group_size` 1 uses the raw [reward](#reward). A larger group uses the reward minus the mean reward of that group. Equal rewards in the group produce advantage 0. The mean is not divided by the group standard deviation.

### AGX

The Apple GPU backend. `./build.sh agx` builds against the sibling tree `nnAGX` and [Metal](#metal). The script exits on a host that is not a Mac.

### Arc

The Intel GPU backend. `./build.sh arc` builds against the sibling tree `nnArc` with [oneAPI](#oneapi) DPC++ and oneMKL. Kernels are [JIT](#jit) [SPIR-V](#spir-v). There is no GPU-architecture flag.

### Attention

The mix, at one position, of values from other positions. This library uses scaled dot-product attention. Each [head](#head) forms scores from the [query](#query) and the [key](#key), divides by the square root of the [head dimension](#head-dimension), applies [softmax](#softmax), and takes that weighted sum of [values](#value). A [causal](#causal) mask limits the sum to the current position and earlier ones. [Bidirectional attention](#bidirectional-attention) uses the whole sequence.

### Autoregressive

A model that writes the next [token](#token) from the tokens already written, then repeats. The [decoder-only](#decoder-only) presets train that way. Generation is the same loop, one new id at a time.

## B

### Backpropagation

The pass that carries the [loss](#loss) back through the layers. Each layer takes the gradient of the loss with respect to its output and writes the gradient with respect to its input and its [parameters](#parameter). Gradients accumulate until [zero_grad](#zero_grad). `Transformer::backward` starts from a gradient of the [logits](#logits).

### Batch

How many sequences the device scores together. `TrainConfig::batch_size` is that count. A row is one window of [sequence length](#sequence) tokens. Rows in `TextDataset::next` are independent random spans.

### BERT

An encoder with [bidirectional attention](#bidirectional-attention). The `bert-*` presets use the GPT-2 block at context 512, plus a [segment embedding](#segment-embedding) table of two rows. Training is [masked language modeling](#masked-language-modeling). There is no next-sentence head and no WordPiece tokenizer. The mask id used by `chat` and the examples is the [pad](#pad) id.

### BFloat16

A 16-bit floating type with the exponent range of [float32](#float32) and a shorter fraction. On nnCUDA, nnCL, and nnROCm, `DType::BFloat16` is the default storage for retained [activations](#activation). `Tensor::to_host` widens it to float32. Parameters, Adam moments, and checkpoints stay float32. nnArc has no `DType` and keeps activations in float32.

### Bias

An added vector inside a linear or a [LayerNorm](#layernorm). The default, llama, and qwen presets turn linear bias off. The GPT-2, GPT-3, and BERT presets put a bias on every linear, including the [LM head](#lm-head).

### Bidirectional attention

Every position may attend to every position in the sequence. `TransformerConfig::causal` is false. `prefill` and `decode_step` throw `causal` on that block. `Generator` appends the [pad](#pad) id and reads that position from a full [forward](#forward). A checkpoint with this attention, or with a [segment embedding](#segment-embedding) table, is version 5 or version 6.

### BOS

Beginning of sequence. The id is `vocab_size - 2`. `encode` writes it only when `EncodeOptions` asks. `decode` adds no bytes for it.

### BPE

Byte-pair encoding. Training repeatedly merges the most frequent adjacent pair inside a [piece](#piece) until the [vocab](#vocab) is full. The learned merges are the vocabulary. `BpeTokenizer::train` counts pairs on every CPU thread. Each merge rewrites only the pieces that contain the winning pair and updates those counts. An equal count keeps the smaller `(left, right)` pair, so one thread and many threads learn the same merges.

### Byte

The base symbol of the tokenizer. Ids `0..255` are the raw bytes. A UTF-8 sequence is bytes at or above 128, and [pretokenization](#pretokenization) keeps those bytes in the same piece as ASCII letters and digits.

## C

### Causal

A position may attend only to itself and to earlier positions. `TransformerConfig::causal` defaults to true. [Prefill](#prefill) and [decode_step](#decode_step) require a causal model that is in [eval mode](#eval-mode). [RlTrainer](#reinforcement-learning-step) rejects a non-causal policy.

### Chat template

`ChatWrap` in `llm/chat_template.hpp`. `Dialogue` is `User:` / `Assistant:`. `Plain` is the line alone. `Qwen` is ChatML (`<|im_start|>user` / `<|im_start|>assistant`). `chat` selects it with `--template qwen` or `--plain`. Those two flags together throw `template`. The helpers are `chat_prompt_suffix`, `chat_completed_turn`, `chat_cut_reply`, and `chat_shown_reply`.

### Checkpoint

`model.bin`, written by `Transformer::save`. A projection-bias block is version 7, or version 8 with [AdamW](#adamw) moments. That shape is a Q, K, V, attention-output, MLP, or [LM head](#lm-head) bias that differs from the single bias flag, or `gelu_tanh`, a nonzero `rope_dims`, `sliding_window`, or `rope_scaling`, `post_norm`, `embed_norm`, `final_norm` left off, or `mlm_head`. An imported Qwen2 file sets the bias on Q, K, and V only. Otherwise a bidirectional block, or a block with a [segment embedding](#segment-embedding) table, is version 5, or version 6 when moments are stored. Otherwise a wide [head](#head), grouped KV heads, [QK-norm](#qk-norm), a [RoPE](#rope) base other than 10000, or an [RMSNorm](#rmsnorm) epsilon other than `1e-5` is version 3, or version 4 with moments. Every other block is version 1, or version 2 with moments. The default, llama, GPT-2, and GPT-3 presets take version 2. Qwen takes version 4. BERT takes version 6. The file holds the token table, the position table when the block has one, the block weights, and the [LM head](#lm-head). `LoadMode::Resume` restores the moments. `LoadMode::Inference` loads weights for generation.

### Completion

The tokens a policy samples after a [prompt](#prompt). In `RlTrainer`, an eos id that stopped the sample is part of the completion and is a trained label. The reward sees `decode` of those ids with that trailing eos removed.

### Compute capability

The CUDA architecture number. `CMAKE_CUDA_ARCHITECTURES` defaults to `86`, which is compute capability 8.6. The ROCm target is a separate name: `CMAKE_HIP_ARCHITECTURES` defaults to `gfx1152`.

### Context (device)

`nn::Context` is the handle to one [device](#device) and one [stream](#stream). Tensors and models built from it allocate on that device. On CUDA, `Context()` uses the current device and does not call `cudaSetDevice`. On the other backends, `Context()` opens device 0. `Context(int)` and `Context(string)` select a device. `synchronize` waits until work queued on that context has finished. Two models may share a context. The policy and its [reference model](#reference-model) do. A second context has its own stream and its own allocations. Those allocations come out of whichever device that context opened. They do not enlarge the first device.

### Context length

The most tokens the position table or the attention mask can hold. `TransformerConfig::max_seq_len` is that limit. The default, llama, qwen, and GPT-2 presets use 1024. GPT-3 uses 2048. BERT uses 512. A training window and a generation of `prompt length + max_new_tokens` both have to fit.

### Cosine schedule

The learning-rate curve after [warmup](#warmup). Progress is `(step - warmup_steps) / (max_steps - warmup_steps)`. The rate is `min_lr + 0.5 * (learning_rate - min_lr) * (1 + cos(pi * progress))`. Step `max_steps` returns `min_lr` exactly. When `max_steps` equals `warmup_steps`, every step stays at `learning_rate`. `learning_rate_at` implements this. `RlTrainer` copies those four fields into a `TrainConfig` and calls the same function.

### CPU

The host processor. Tokenization, the host [policy gradient](#policy-gradient), and sampling run here. The model math runs on the [device](#device). BPE training and long encodes use `std::thread::hardware_concurrency` workers. The `llm` target links pthread.

### Cross-entropy

The supervised loss. At each trained position it is the negative [log-softmax](#log-softmax) of the label id. `sequence_cross_entropy` averages that over positions whose label is not the [ignore index](#ignore-index). [Perplexity](#perplexity) is the exponential of that mean. With advantage 1 and `kl_coef` 0, the [policy gradient](#policy-gradient) matches this gradient.

### CUDA

The NVIDIA GPU platform. `./build.sh cuda` builds against the sibling tree `nnCUDA`, which uses the CUDA Toolkit, [cuBLAS](#cublas), and [cuDNN](#cudnn). Public `llm` headers do not include CUDA headers.

### cuBLAS

The CUDA library for matrix multiplication. `nnCUDA` binds a cuBLAS handle to the context [stream](#stream).

### cuDNN

The CUDA library for deep-learning primitives. `nnCUDA` binds a cuDNN handle to the same stream.

## D

### Data-parallel training

The same weights on each GPU, a different [batch](#batch) on each, and an average of the [gradients](#gradient). That raises tokens per second. Each GPU still holds a full copy of the [parameters](#parameter), the gradients, and the [AdamW](#adamw) moments, so the model that fits is still the model that fits on one card. This library keeps the updated model on one [context](#context-device).

### Decode (tokenizer)

`Tokenizer::decode` turns ids into bytes. Byte ids and merge ids expand to the bytes they were built from. [Pad](#pad), [bos](#bos), and [eos](#eos) add no bytes. An id outside the vocab throws `id`. With bos and eos off, `decode(encode(text))` equals `text`.

### decode_step

One generation step on a causal model. The caller feeds the id just sampled, shape `(batch, 1)`, and the layer appends [keys](#key) and [values](#value) to the [KV cache](#kv-cache). The new logit row has length 1. The call throws unless the model is causal and in [eval mode](#eval-mode). The reinforcement-learning score does not backward through this cache. It runs a fresh [forward](#forward).

### Decoder-only

A stack of causal blocks that predicts the next token. There is no separate encoder. The default, llama, qwen, GPT-2, and GPT-3 presets are this shape. BERT is an encoder.

### Device

The GPU that owns a tensor's memory. `Tensor::data` is a device pointer. A copy from another context is completed on the source, then queued on the destination. Memory on one device is not usable as memory on another. `device_count`, `device_name`, and `detect_devices` list the devices the backend can see.

### d_ff

The width of the [feed-forward](#feed-forward-network) after its nonlinearity. For [SwiGLU](#swiglu) the first linear writes `2 * d_ff`, and the gate reduces that to `d_ff`. For [GELU](#gelu) the first linear writes `d_ff`. The default and llama presets use a SwiGLU width below `4 * d_model`. The GPT-2, GPT-3, and BERT presets set `d_ff` to `4 * d_model`.

### d_model

The width of the residual stream, the token [embedding](#embedding), and the [LM head](#lm-head) input. [Head](#head) count times [head dimension](#head-dimension) equals this width, except on the Qwen presets, where the [query](#query) projection is `num_heads * head_dim` and is wider than `d_model`.

### Dropout

A layer that, in [train mode](#train-mode), sets activations to zero with probability `p` and rescales the rest. Every preset sets `dropout` to 0, so the train forward and the eval forward match. `decode_step` still requires eval mode.

### DType

`nn::DType` on nnCUDA, nnCL, and nnROCm. `Float32` is the storage of [parameters](#parameter), gradients, Adam moments, and checkpoints. `BFloat16` is the default storage of retained [activations](#activation). `Context::set_activation_dtype` accepts those two and throws `dtype` otherwise. Call it before creating tensors or layers. nnArc has no `DType`. Its activations stay float32.

## E

### Early stopping

`early_stop_patience` counts [validation](#validation) evals that do not improve the mean [cross-entropy](#cross-entropy) by more than `early_stop_min_delta`. When the count is reached, `fit` restores the weights and Adam moments from the best eval and saves them. A finite `min_loss` is a different stop: it ends the run when the training loss itself is at or below that value, and it keeps that step.

### Embedding

A learned row for each [token](#token) id, of width [d_model](#d_model). The [forward](#forward) looks up the rows for the input ids. A learned-absolute preset also adds a row from a position table of length [context length](#context-length). [RoPE](#rope) stores no position table. The token table and the [LM head](#lm-head) are separate. See [untied embeddings](#untied-embeddings).

### Encode

`Tokenizer::encode` splits the string into [pieces](#piece), turns each piece into bytes, and applies [BPE](#bpe) merges in learn order. It optionally writes [bos](#bos) and [eos](#eos). It does not write [pad](#pad). Training a model encodes each document and appends eos in the example programs.

### EOS

End of sequence. The id is `vocab_size - 1`. Generation stops when this id is sampled, and the returned string does not contain it. When `TextDataset` is given this id, a window stays inside the document that ends on it. The id may be the last label. The input tokens of that window come from earlier in the document. In [RlTrainer](#reinforcement-learning-step) the sampled eos is stored in `Episode::token_ids` and is trained. The reward text omits it.

### Epoch

One pass through the corpus. `Trainer::fit` does not count epochs. It runs a fixed number of [training steps](#training-step), and each step draws new random windows from the token array. `train_data` prints `one pass is K steps at batch B seq S`, where `K` is the training token count divided by `batch * seq_len`, rounded up.

### Eval mode

`model.eval()`. [Dropout](#dropout) is off, and [prefill](#prefill) and [decode_step](#decode_step) accept the model. Validation forwards and both reinforcement-learning forwards run in eval. `Trainer::fit` returns to [train mode](#train-mode) after a validation pass.

## F

### Façade

The `Transformer` object that owns the token [embeddings](#embedding) and the [LM head](#lm-head) around the stack of blocks. A façade [checkpoint](#checkpoint) stores those outer tables with the block weights. `Transformer::save` writes that file.

### Feed-forward network

The position-wise MLP inside each block, after the second [pre-norm](#pre-norm). The hidden width is [d_ff](#d_ff). The default, llama, and qwen blocks use [SwiGLU](#swiglu). The GPT-2, GPT-3, and BERT blocks use [GELU](#gelu).

### Fine-tuning

More supervised steps on a checkpoint that already exists. `finetune` reuses `tokenizer.bpe` and the model shape, encodes a new file list, and writes a new directory. The default optimizer is a fresh [AdamW](#adamw). `--resume` keeps the moments and the optimizer step. The [learning rate](#learning-rate) still starts at step 1 of that run.

### Float16

IEEE half precision. `gguf import` accepts an `nnllm` tensor stored as float16 or [float32](#float32) and writes a float32 [checkpoint](#checkpoint). Training does not store weights as float16.

### Float32

The 32-bit floating type. Parameters, gradients, Adam moments, checkpoints, and the host [policy gradient](#policy-gradient) use it. The Size column in the README is four bytes times the parameter count at vocab 16000.

### Forward

The pass from token ids to [logits](#logits). Look up [embeddings](#embedding), run each block, apply the final norm, and multiply by the [LM head](#lm-head). The result has shape `(batch, seq, vocab)`. The returned tensor is owned by the layer. The next forward overwrites it.

### Full attention

Every position allowed by the mask is scored. The library has no sliding window and no sparse pattern. The alternating sparse attention from the GPT-3 paper is not implemented. The `gpt3-*` presets are the dense GPT-2 block with context 2048.

## G

### GELU

Gaussian Error Linear Unit, the nonlinearity in the GPT-2, GPT-3, and BERT feed-forward. Those presets use the erf form. The first linear writes [d_ff](#d_ff) values, GELU is applied, and the second linear returns to [d_model](#d_model).

### Generate

`Generator::generate`. It encodes the [prompt](#prompt) with `add_bos_default()` and `add_eos_default()`. Both are false for an [LPB1](#lpb1) tokenizer. A causal model [prefills](#prefill) and then samples with [decode_step](#decode_step). A bidirectional model appends [pad](#pad) and samples that position from [forward](#forward). The return value is [decode](#decode-tokenizer) of the new ids. Sampling stops when the id is in `stop_ids`. An empty `stop_ids` stops on [eos](#eos) when that id is non-negative. The stopping id is not appended.

### GGUF

The file format used by `examples/gguf.cpp`. `export` writes architecture `nnllm` and float32 weights. `import` reads float32 or [float16](#float16) `nnllm` tensors and writes a checkpoint with no optimizer. `llama` writes a llama.cpp file: architecture `llama` for a classic block, `qwen3` when [QK-norm](#qk-norm) is on, and `qwen2` otherwise. A learned-position export omits the position table, and llama.cpp then applies [RoPE](#rope), so those tokens differ. Quantized tensors, and a public GPT-2 or BERT GGUF, stay outside this converter. `tokenizer.bpe` is not stored in the GGUF.

### GPU

The graphics processor that runs the network. One [context](#context-device) uses one GPU. This tree's backends are [CUDA](#cuda), [OpenCL](#opencl), [ROCm](#rocm), [Arc](#arc), and [AGX](#agx).

### GPU memory

The storage on one [device](#device). A tensor allocated on one GPU is not in the memory of another. Training holds float32 [parameters](#parameter), float32 gradients, and two float32 Adam moment buffers. nnCUDA, nnCL, and nnROCm also hold [bfloat16](#bfloat16) retained activations, so the activation working set is about half of the README's float32 fit notes. nnArc keeps those activations in float32.

### Gradient

The derivative of the [loss](#loss) with respect to a tensor. Parameter gradients accumulate across [micro-batches](#micro-batch). `zero_grad` clears them. A non-finite gradient throws `non-finite grad` when `abort_on_non_finite` is set.

### Gradient accumulation

Several [micro-batches](#micro-batch) add into the same parameter gradients, and [AdamW](#adamw) steps once. `Trainer` scales each logit gradient by `1 / grad_accum_steps` before [backward](#backpropagation). `RlTrainer` scales by `1 / batch_prompts`.

### Gradient clipping

`clip_grad_norm` rescales the parameter gradients so their global norm is at most `grad_clip_norm`. The default cap is 1. A cap of 0 skips the rescale. With `abort_on_non_finite`, a cap of 0 still reads the norm so a non-finite value can throw.

### Gradient descent

The update that moves each parameter against its [gradient](#gradient). [AdamW](#adamw) is the form used here: a per-parameter step size from the two moments, a global [learning rate](#learning-rate), and [weight decay](#weight-decay).

### Grouped-query attention

Several [query](#query) heads share one [key](#key) and one [value](#value) head. `num_kv_heads` is the number of key and value heads. Zero means one KV head per query head, which is ordinary [multi-head attention](#multi-head-attention). The qwen presets set a smaller KV count. The K and V projections are `num_kv_heads * head_dim` wide.

## H

### Head

One attention slice. `num_heads` query heads each have width [head dimension](#head-dimension). Scores and the weighted sum happen inside the head, and the head outputs are joined before the output projection.

### Head dimension

The width of one [head](#head). When `head_dim` is 0 it is `d_model / num_heads`. The default, llama, GPT-2, GPT-3, and BERT presets use 64. The qwen presets use 128, and the query projection is wider than [d_model](#d_model).

### HIP

The AMD GPU language that ROCm compiles. `./build.sh amd` uses `amdclang++`. The device link of a binary that contains HIP code uses the HIP linker. `hipcc` is not the compiler this script selects.

### Host

CPU memory, as opposed to [device](#device) memory. `copy_from_host` queues an upload. The caller synchronizes before freeing or reusing that buffer. `to_host` synchronizes and returns a `std::vector<float>`, widening [bfloat16](#bfloat16). Sampling and `policy_gradient` read those host copies.

## I

### ICD

Installable Client Driver, the loader [OpenCL](#opencl) uses to find a vendor's GPU implementation. `./build.sh opencl` needs an ICD and a GPU.

### Ignore index

The label `-100`. [Cross-entropy](#cross-entropy) and the [policy gradient](#policy-gradient) skip it, and it is absent from the mean. Prompt tokens, padding, and the positions a masked-LM step did not pick are ignore. The denominator is the count of labels that remain.

### Inference

A forward that does not update weights. `LoadMode::Inference` loads a [checkpoint](#checkpoint) without Adam moments. `chat` and the continuation printed by `train_data` load that way. The model is put in [eval mode](#eval-mode).

## J

### JIT

Just-in-time compilation. The [Arc](#arc) backend compiles its kernels to [SPIR-V](#spir-v) when the program runs, so one binary covers the Arc devices the driver accepts. The [OpenCL](#opencl) backend compiles `kernels.cl` at runtime for the device's OpenCL C version.

## K

### Kernel

A function that runs on the [device](#device). The `llm` library calls `nn` operations. It does not ship its own CUDA, HIP, OpenCL, SYCL, or Metal kernels. The host [policy gradient](#policy-gradient) is ordinary C++.

### Key

The vector a [query](#query) is scored against. One key is stored per position per [KV head](#kv-head). [RoPE](#rope) and, on Qwen3, [QK-norm](#qk-norm) are applied to keys before the score.

### KL divergence

Kullback-Leibler divergence. For two distributions over the [vocab](#vocab) it is the sum of `p(v) * (log p(v) - log p_ref(v))`. `RlTrainer` adds `kl_coef` times that full-vocabulary value at each completion position. `kl_coef` 0 skips the [reference model](#reference-model). The reported `kl` is the mean of those values over completion positions.

### KV cache

The [keys](#key) and [values](#value) kept so a later token does not recompute the [prompt](#prompt). [Prefill](#prefill) fills it. [decode_step](#decode_step) appends one position. The cache is for generation. The training [forward](#forward) and the reinforcement-learning score compute keys and values for the whole window.

### KV head

A key and value head shared by one or more query heads. See [grouped-query attention](#grouped-query-attention).

## L

### Layer

One block in the stack, counted by `num_layers`. Each layer is a [pre-norm](#pre-norm), [attention](#attention), [residual](#residual-connection), a second pre-norm, a [feed-forward](#feed-forward-network), and a second residual. A final norm sits after the last layer.

### LayerNorm

Normalization across the features of one position: subtract the mean, divide by the root of the variance plus epsilon, then scale and shift. The default, GPT-2, GPT-3, and BERT blocks use it. GPT-2 and BERT set epsilon to `1e-5`. The llama and qwen blocks use [RMSNorm](#rmsnorm) instead. Names containing `ln` take [weight decay](#weight-decay) 0.

### Learning rate

The step size `AdamW` uses. It changes by step. [Warmup](#warmup) raises it linearly from a small value to `learning_rate`. The [cosine schedule](#cosine-schedule) then lowers it to `min_lr`. `fit` calls `set_learning_rate` at the start of each step. It does not rebuild the optimizer.

### LLM

A large language model: a [transformer](#transformer) trained to predict tokens. The models in this tree use that block at a small size. `tiny` is about 12 million parameters at vocab 16000. `large` is under a billion. The training loop is the same shape at either size.

### LM head

The last linear, from [d_model](#d_model) to [vocab](#vocab). Its outputs are the [logits](#logits). It does not share weights with the token [embedding](#embedding). GPT-2, GPT-3, and BERT give it a [bias](#bias). The other presets do not.

### LoadMode

How `Transformer::load` reads a [checkpoint](#checkpoint). `Inference` restores weights for a forward. `Resume` also restores [AdamW](#adamw) moments and the optimizer step.

### Logits

The raw scores at the [LM head](#lm-head), one per vocab id, before [softmax](#softmax). Shape `(batch, sequence, vocab)` after a full forward, and `(batch, 1, vocab)` after [decode_step](#decode_step). Sampling and the host policy gradient copy them to the [host](#host).

### Log-softmax

The log of a [softmax](#softmax) probability. [Cross-entropy](#cross-entropy) is the negative log-softmax of the correct id. The [KL divergence](#kl-divergence) is a difference of log-softmax values, weighted by the policy probability.

### Loss

The number the [training step](#training-step) descends. Supervised training uses [cross-entropy](#cross-entropy). A masked step uses that loss on the masked positions. A reinforcement-learning step uses the [policy gradient](#policy-gradient) loss, which adds an optional KL term. A non-finite loss throws `non-finite loss` when `abort_on_non_finite` is set.

### LPB1

The magic bytes of `tokenizer.bpe`. The file is little-endian: magic, `vocab_size`, merge count, pad id, bos id, eos id, then each merge as a pair of ids. The merge count is `vocab_size - 259`. The file has no flag for the [pretokenization](#pretokenization) rules. A file trained before the current piece split has to be trained again.

## M

### Masked language modeling

The BERT training loss, and `TrainConfig::masked_lm`. Each position is chosen with probability `mask_prob` (default 0.15). A chosen position is labeled with its original id. Of those positions, 80% are replaced by `mask_token`, 10% by a random id, and 10% are left as they are. Every other label is the [ignore index](#ignore-index). The examples use the [pad](#pad) id as the mask.

### Merge

One learned BPE rule: two ids become one new id. Merges are numbered from 256 upward, in the order they were learned. `encode` applies them in that order, inside a single [piece](#piece). A merge cannot cite [pad](#pad), [bos](#bos), or [eos](#eos).

### Metal

Apple's GPU API. The [AGX](#agx) backend uses it, together with the Foundation framework.

### Micro-batch

One [forward](#forward) and [backward](#backpropagation) inside a [training step](#training-step). `grad_accum_steps` micro-batches add their gradients before one clip and one optimizer step. The logged loss is the mean of the micro-batch losses.

### Model

The [parameters](#parameter) plus the computation that maps token ids to [logits](#logits) or to the next token. In this library that object is `nn::Transformer`, built from a `TransformerConfig`. A [preset](#preset) fills the config in. The weights on disk are the [checkpoint](#checkpoint).

### Multi-head attention

[Attention](#attention) run in several [heads](#head), then joined. `num_kv_heads` of 0 gives each query head its own key and value head. A smaller count is [grouped-query attention](#grouped-query-attention).

## N

### Next-token prediction

The decoder objective. The input at position `t` is token `t`, and the label is token `t + 1`. `TextDataset::next` builds that shift. The last position of a window has no label inside the window, so the stored length is `seq_len + 1` tokens and the tensors are `seq_len` long.

### nn

The neural-net library this tree links. The CMake target is `nn`. The source tree is one of [nnCUDA](#cuda), `nnCL`, `nnROCm`, `nnArc`, or `nnAGX`, chosen by `./build.sh`. `llm` calls it for tensors, layers, the transformer, the loss, and [AdamW](#adamw).

### Nucleus sampling

Top-p sampling. After [softmax](#softmax), tokens are kept in descending probability until the cumulative mass reaches `top_p`, and the draw is among those tokens. `top_p` of 1 keeps the whole distribution. `sample_token` applies it after [top-k](#top-k).

## O

### oneAPI

Intel's compiler stack for [Arc](#arc). `./build.sh arc` needs `icpx` from oneAPI DPC++ and oneMKL's SYCL BLAS. The script exits when `icpx` is missing.

### OpenCL

The portable GPU API used by `nnCL`. `./build.sh opencl` selects that sibling. The vendor filter is fixed at configure time (`auto`, `radeon`, `intel`, `apple`, or `nvidia`). A device is kept when its max work-group size is at least 256. Do not set `RUSTICL_ENABLE`.

### Optimizer

The algorithm that applies [gradients](#gradient) to [parameters](#parameter). Training and reinforcement learning use [AdamW](#adamw). `opt.step` reads the current [learning rate](#learning-rate) and the accumulated gradients.

### Overfitting

The training [loss](#loss) keeps falling while the [validation](#validation) loss rises or stalls. The weights are memorizing the training windows. [Early stopping](#early-stopping) is the library's response: it restores the step with the best validation cross-entropy. A training-loss threshold, `min_loss`, does not see that gap.

## P

### Pad

The padding id, `vocab_size - 3`. `encode` does not write it. Right-padding in a reinforcement-learning batch writes it on the input and the [ignore index](#ignore-index) on the label. A finished generation row is fed pad on later [decode_step](#decode_step) calls, and that pad is not stored. BERT generation appends pad and reads that position. `decode` adds no bytes for pad.

### Parameter

A trained number: an embedding row, a linear weight, a norm scale, a bias. `parameters()` lists them in the order [AdamW](#adamw) steps. The README Size column counts them at vocab 16000 and stores each as four [float32](#float32) bytes. Activations are not parameters.

### Perplexity

The exponential of the mean [cross-entropy](#cross-entropy). Validation prints both. It is the geometric average of one over the probability the model assigned to the correct token. A lower cross-entropy is a lower perplexity.

### Piece

One run of a single byte class, produced before any [merge](#merge). Word bytes are ASCII letters, ASCII digits, and bytes at or above 128. Whitespace is space, tab, and the newline family. Everything else, including punctuation, is its own class. Merges do not cross a piece, and they do not cross from one training string into the next.

### Pipeline parallelism

Different [layers](#layer) of one model placed on different [GPUs](#gpu). Activations pass from one device to the next. This library keeps every layer on the same [context](#context-device).

### Policy

The distribution the model defines over the next [token](#token): [softmax](#softmax) of the [logits](#logits). In `RlTrainer` the policy is the causal model that is updated. Sampling and the scoring forward both run in [eval mode](#eval-mode), so the scored distribution is the one that drew the tokens.

### Policy gradient

The reinforcement-learning [loss](#loss) and its logit gradient, computed by `policy_gradient` on the [host](#host). At each completion position the gradient is `advantage * (softmax - one_hot(token))`, plus `kl_coef * softmax * (log_softmax - log_softmax_ref - KL)` when a reference is present. The result is divided by the number of completion labels. The trainer copies that gradient into a float32 tensor and calls `Transformer::backward`.

### Positional encoding

How the model knows the index of a token. Learned absolute adds a row from a table of length [context length](#context-length). [RoPE](#rope) rotates [query](#query) and [key](#key) by an angle that depends on the index, and stores no table. A learned-position checkpoint and a RoPE checkpoint are different blocks. One cannot be edited into the other.

### PPO

Proximal Policy Optimization. It clips the ratio of the new token probability to the probability under the policy that sampled the token. `RlTrainer` has no ratio and no clip. Its update is the [advantage](#advantage) term plus an optional [KL](#kl-divergence) penalty.

### Prefill

The first generation call on a causal model. It runs the whole [prompt](#prompt) and fills the [KV cache](#kv-cache). Later tokens use [decode_step](#decode_step). The call requires [eval mode](#eval-mode) and a causal block.

### Pre-norm

The norm is applied to the copy of the residual stream that enters [attention](#attention) or the [feed-forward](#feed-forward-network). The branch output is then added back to the un-normalized stream. Every preset uses this order. See [residual connection](#residual-connection).

### Preset

A function in `llm::presets` that fills a `TransformerConfig`. The widths are tiny, small, base, base-medium, base-large, medium, and large. The families are the default block, `llama-*`, `qwen-*`, `gpt2-*`, `gpt3-*`, and `bert-*`. `train_data` can replace layers, `d_model`, heads, `d_ff`, context, KV heads, and head dim on the selected preset. Dropout stays 0 and embeddings stay untied.

### Pretokenization

The split of a string into [pieces](#piece) before [BPE](#bpe). It is a class run, not a linguistic word segmenter. The `LPB1` file does not record the rules. Encode uses the rules in the code that loads the file.

### Prompt

The text the model is given before it writes. `generate` encodes it with `add_bos_default()` and `add_eos_default()`. Both are false for an [LPB1](#lpb1) tokenizer. `RlTrainer` leaves prompt targets on the [ignore index](#ignore-index). The last prompt position is kept, because its logit predicts the first completion token.

## Q

### QK-norm

An [RMSNorm](#rmsnorm) on the [query](#query) and the [key](#key) of each head, of width [head dimension](#head-dimension), applied before [RoPE](#rope). The qwen presets turn it on. `gguf llama` writes architecture `qwen3` when it is on and `qwen2` when a Qwen-style block has it off.

### Quantization

Storing a weight in fewer than 32 bits. This trainer stores [float32](#float32). `gguf import` also accepts [float16](#float16) `nnllm` tensors and writes float32. Quantized GGUF block types are rejected.

### Query

The vector that asks [attention](#attention) which positions to read. There is one query per position per [head](#head). On the qwen presets the query projection is `num_heads * 128`, which is wider than [d_model](#d_model).

## R

### Reference model

A second `Transformer`, loaded with `LoadMode::Inference`, whose token probabilities are the `p_ref` in the [KL divergence](#kl-divergence). The caller owns it and does not train it. `RlTrainer` requires it when `kl_coef > 0` and rejects it when `kl_coef` is 0. Its vocab, width, layers, heads, context, and causal flag must match the policy. Its forward runs first, and those logits are copied to the host before the policy forward.

### REINFORCE

The policy gradient that multiplies the log-probability of the sampled token by the [reward](#reward). `group_size` 1 is that update. A larger group subtracts the group mean from the reward first, which is a baseline. There is no learned value function.

### Residual connection

The sum that carries the block input forward. Attention's output is added to the block input. The feed-forward's output is added to that sum. The [pre-norm](#pre-norm) sits on the branch, so the skip path itself is not normalized.

### Reward

The float a caller returns for one [completion](#completion). `RlTrainer` turns rewards into [advantages](#advantage). A non-finite reward throws `non-finite loss` when `abort_on_non_finite` is set. `examples/reinforce.cpp` uses 1 when the completion contains a string and 0 otherwise. That string test is the example program. The library takes any function with that signature.

### Reinforcement-learning step

One `RlTrainer::fit` update. The policy is set to [eval mode](#eval-mode) and its gradients are cleared. The step draws prompts with replacement, samples `group_size` completions through [prefill](#prefill) and [decode_step](#decode_step), scores the [reward](#reward), builds a right-padded batch, and runs a full [forward](#forward). An optional [reference model](#reference-model) is forwarded first. `policy_gradient` writes a float32 logit gradient, `backward` runs, the gradient is [clipped](#gradient-clipping), and [AdamW](#adamw) steps. A step with no completion tokens is counted and does not step the optimizer. `Trainer::fit` is a separate loop.

### Repetition penalty

`GenerateConfig::repetition_penalty`. The default is 1, which leaves the logits unchanged. A value above 1 is applied before temperature, once per distinct id in the prompt and the ids already sampled. A positive logit is divided by the penalty. Any other logit is multiplied by it. The value has to be finite and greater than 0. `chat --repeat-penalty` sets it. `RlTrainer` samples with the default.

### RMSNorm

Root-mean-square normalization. Each position is divided by the root of the mean square plus epsilon, then scaled. There is no shift. The llama presets use epsilon `1e-5` and [RoPE](#rope) base 10000. The qwen presets use epsilon `1e-6` and base `1e6`.

### ROCm

AMD's GPU platform. `./build.sh amd` builds against `nnROCm` and links hipBLAS and rocBLAS. See [HIP](#hip).

### RoPE

Rotary position embedding. Pairs of neighboring coordinates in the [query](#query) and the [key](#key) are rotated by an angle that depends on the token position. The base angle comes from `rope_theta`. The llama presets use adjacent-pair RoPE with base 10000. The qwen presets use base `1e6`. No position table is stored.

## S

### Sampling

Drawing the next id from one logit row. `sample_token` does the draw, and `Generator::generate` calls it. A [repetition penalty](#repetition-penalty) above 1 is applied first. Temperature 0 returns the argmax. Any other temperature divides the logits, keeps the [top-k](#top-k) entries when `top_k > 0`, applies [softmax](#softmax), applies [nucleus sampling](#nucleus-sampling) when `top_p < 1`, and draws with the caller's `std::mt19937`.

### Seed

The integer passed to the `Transformer` constructor and the seed of the `std::mt19937` used for windows and samples. The same model seed builds the same initial weights. The generator used for data and sampling is separate from that constructor argument.

### Segment embedding

A learned row added from a token-type id. `type_vocab_size` of 0 omits the table. The BERT presets use a table of two rows. Single-sequence training and generation use segment 0. A checkpoint with this table is version 5 or version 6.

### Sequence

The ordered tokens of one example. `seq_len` is the training window. The dataset needs at least `seq_len + 1` tokens. With `eos_id` left at -1, spans are chosen uniformly from `[0, numel - seq_len)`. Passing the eos id keeps each window inside one document. The document ends on that id, or on the last token when the tail has none. Eos may be the last label. The input tokens stay earlier in the document. A document shorter than `seq_len + 1` contributes no starts, and `next` throws `short corpus` when no start remains. `train_data`, `finetune`, `train_chat`, and `train_bert_chat` pass the tokenizer eos id.

### Softmax

The map from [logits](#logits) to probabilities that are positive and sum to 1. Subtracting the row maximum before the exponential keeps the math finite. [Attention](#attention) softmaxes scores. The [LM head](#lm-head) softmaxes token logits. Sampling and the losses use that map.

### SPIR-V

The binary form of a GPU kernel used by [SYCL](#sycl). The [Arc](#arc) backend JITs it, so the CMake build has no per-chip architecture list.

### Step

One optimizer update, numbered from 1. `max_steps` is how many run. `start_step` resumes the [learning rate](#learning-rate) at a later index. The Adam step count stored in a [checkpoint](#checkpoint) is the number of `opt.step` calls, which matches this index when every step updated the weights.

### Stream

The device queue on a [context](#context-device). Each context has one queue, and work on it runs in order. The CUDA backend binds its cuBLAS and cuDNN handles to a non-default stream. OpenCL uses one in-order command queue. `synchronize` waits for that queue.

### SwiGLU

The gated feed-forward used by the default, llama, and qwen presets. The first linear writes two halves of width [d_ff](#d_ff). The output is `silu(gate) * value`, where `silu(x)` is `x * sigmoid(x)`. The second linear maps `d_ff` back to [d_model](#d_model). A llama.cpp export splits that first matrix into `ffn_gate` and `ffn_up`.

### SYCL

The C++ accelerator model used by [oneAPI](#oneapi). `./build.sh arc` compiles the device code with `-fsycl`.

### Synchronize

`Context::synchronize` waits until every operation queued on that [context](#context-device) has finished. `to_host`, parameter initialization, and checkpoint save synchronize for you. `copy_from_host` only queues the copy.

## T

### Temperature

The divisor applied to [logits](#logits) before [sampling](#sampling). 0 selects the maximum logit. A value below 1 sharpens the distribution. A value above 1 flattens it. The field must be finite and at least 0.

### Tensor

A multidimensional array on a [context](#context-device). `nn::Tensor` holds floating values. `nn::IndexTensor` holds the int64 token ids. Shapes used here are `(batch, seq)` for ids and `(batch, seq, vocab)` for logits. Storage is contiguous and row-major.

### Tensor parallelism

Shards of a single [layer](#layer) placed on several [GPUs](#gpu), with a reduction after the shard. This library keeps each matrix on one [device](#device).

### Token

One integer id the model reads or writes. A byte, a [merge](#merge), [pad](#pad), [bos](#bos), and [eos](#eos) are all tokens. The text of a merge is the concatenation of the bytes of the ids it was built from.

### Token file

`tokens.bin` is a raw little-endian `int32` stream of ids, with no header. `tokens.meta` is two text lines: the count, then the [vocab](#vocab) size. `TextDataset` loads both into host memory. `--reuse-tokens` trains on that file and does not [encode](#encode) the documents again.

### Tokenizer

The map between text and ids. `BpeTokenizer` trains and loads the [LPB1](#lpb1) file `tokenizer.bpe`. `HfTokenizer` loads a Hugging Face `tokenizer.json` byte-level BPE and keeps that file's ids. `chat` loads `tokenizer.bpe` when the directory has one, and `tokenizer.json` otherwise. The checkpoint vocab and the tokenizer are a pair.

### Top-k

The sampling cut that keeps the `top_k` largest logits and drops the rest before [softmax](#softmax). `top_k` of 0 keeps every logit. It runs before [nucleus sampling](#nucleus-sampling).

### Train mode

`model.train()`. [Dropout](#dropout) would be active when its probability is positive. `Trainer::fit` trains in this mode. [Prefill](#prefill) and [decode_step](#decode_step) throw while it is set. The transformer blocks do not use BatchNorm, so eval and train differ only where dropout is non-zero. Preset dropout is 0.

### Training step

One supervised update. Set the [learning rate](#learning-rate), switch to [train mode](#train-mode), and [zero_grad](#zero_grad). For each [micro-batch](#micro-batch), draw a window, [forward](#forward), compute [cross-entropy](#cross-entropy), scale the logit gradient, and [backward](#backpropagation). Then [clip](#gradient-clipping) and `opt.step`. [Validation](#validation), the [checkpoint](#checkpoint) write, [early stopping](#early-stopping), and `min_loss` run on their own cadences after the step.

### Transformer

The network: token [embeddings](#embedding), `num_layers` blocks, a final norm, and an [LM head](#lm-head). `nn::Transformer` owns those embeddings outside the inner `Network`. `Family::Transformer` is that checkpoint family.

## U

### Untied embeddings

The token table and the [LM head](#lm-head) are two parameters. `tie_embeddings` is false on every preset, and `nn` rejects tied embeddings. Tying would store one matrix and use its transpose as the head.

## V

### Validation

A set of [forwards](#forward) in [eval mode](#eval-mode) on held-out tokens. `Trainer::fit` runs `eval_batches` of them every `eval_every` steps, prints the mean [cross-entropy](#cross-entropy) and the [perplexity](#perplexity), and returns to train mode. [Early stopping](#early-stopping) reads that mean. The reinforcement-learning loop has no validation split. Its reward is not comparable across steps the way this cross-entropy is.

### Value

The vector [attention](#attention) mixes. Each position contributes its value, multiplied by the [softmax](#softmax) weight of that position's score. Values are stored in the [KV cache](#kv-cache) during generation. This is the attention value. It is not a reinforcement-learning value function. This library does not train one of those.

### Vocab

The set of ids. Size is `256 + n_merges + 3`, so the minimum is 259: the 256 bytes plus [pad](#pad), [bos](#bos), and [eos](#eos). The model `vocab_size` and the tokenizer vocab are the same number. The embedding has that many rows, and the [LM head](#lm-head) has that many outputs.

## W

### Warmup

The opening steps of the [learning rate](#learning-rate). While `step` is below `warmup_steps`, the rate is `learning_rate * step / warmup_steps`. Step `warmup_steps` is `learning_rate` exactly. A warmup of 0 skips that ramp, and the [cosine schedule](#cosine-schedule) starts at step 1.

### Weight decay

A pull of each decayed parameter toward zero, applied by [AdamW](#adamw) beside the gradient update. `adamw_groups` turns it on for rank-2 weights and off for biases and norm parameters. A zero advantage and a zero KL still step AdamW, so decay still moves those weights.

### Weight tying

Using one matrix as both the token [embedding](#embedding) and the [LM head](#lm-head). This library keeps them [untied](#untied-embeddings).

## Z

### zero_grad

Clears the parameter [gradients](#gradient) so the next [micro-batch](#micro-batch) starts from zero. `fit` calls it once per step, before the micro-batches. A stop or an early-stop restore zeros the gradients again before returning.
