# Optional reinforcement learning in llm

Add a separate trainer, `RlTrainer`, that updates a causal decoder from a caller-supplied reward. `Trainer::fit` stays next-token cross-entropy and masked-token cross-entropy. No nn backend changes.

## Objective

One optimizer step samples a group of completions for each chosen prompt, scores them, and minimizes

```text
L = mean over completion tokens of
    -advantage * log π(token) + kl_coef * KL(π || π_ref)
```

- `group_size == 1`: advantage is the raw reward. That is REINFORCE with no baseline.
- `group_size > 1`: advantage is `reward - mean(rewards in the group)`. Equal rewards produce advantage 0. There is no division by the group standard deviation.
- Prompt tokens and padding are ignore index `-100` and drop out of the mean. The denominator is the count of completion labels, matching `sequence_cross_entropy`.
- `kl_coef == 0` skips the reference model. The KL is the full-vocab categorical KL at each completion position, not a sampled-token estimate.

The gradient at a completion position, divided by that count, is

```text
advantage * (softmax - one_hot(token))
+ kl_coef * softmax * (log_softmax - log_softmax_ref - KL)
```

With advantage 1 and `kl_coef` 0, that row equals the `sequence_cross_entropy` gradient. That identity is the numerical check.

Label layout matches `TextDataset::next`. For a row `full = prompt + completion`, input is `full` without its last token, and `labels[t]` is `full[t + 1]`. Labels whose target still lies inside the prompt are `-100`. The last prompt position is kept: its logit predicts the first completion token. An eos id sampled as the stop token is part of the completion, so its log-probability is trained. The reward text is `decode` of the new ids with that eos removed, the same string `Generator::generate` returns.

## Why the loss stays on the host

`Transformer::backward` already accepts any `grad_logits` tensor. `Generator` already copies logits to the host to sample. A device kernel would have to be added to nnCUDA, nnCL, nnROCm, nnArc, and nnAGX.

`policy_gradient` runs on the host. The trainer copies the gradient into a float tensor and calls `backward`. A later device kernel can replace that function without changing `RlTrainer`.

v1 requires float32 logits. A BFloat16 policy throws `std::invalid_argument("dtype")`.

## API

New header `include/llm/rl.hpp`. Implementation `src/rl.cpp`, added to the `llm` static library.

```cpp
struct Episode {
  std::string_view prompt;
  std::string_view text;                  // eos omitted
  const std::vector<int32_t>& token_ids;  // completion ids, eos included when sampled
};

using RewardFn = std::function<float(const Episode&)>;

struct PolicyGrad {
  float loss = 0.f;
  float kl = 0.f;  // mean full-vocab KL over completion positions; 0 when kl_coef is 0
  std::vector<float> grad_logits;
};

// logits, and ref_logits when kl_coef > 0, are row-major (N, T, V).
// labels are (N, T) int64 with ignore -100.
// advantages has length N and applies to every kept label in that row.
PolicyGrad policy_gradient(const float* logits, const float* ref_logits,
                           const int64_t* labels, const float* advantages,
                           int rows, int seq, int vocab, float kl_coef);

struct RlConfig {
  float learning_rate = 1e-5f;
  float min_lr = 1e-6f;
  int warmup_steps = 0;
  int max_steps = 100;
  int start_step = 1;
  int batch_prompts = 1;
  int group_size = 4;
  int max_new_tokens = 64;
  float temperature = 1.f;
  int top_k = 0;
  float top_p = 1.f;
  float kl_coef = 0.f;
  int log_every = 1;
  int ckpt_every = 0;
  float grad_clip_norm = 1.f;
  bool abort_on_non_finite = true;
  std::filesystem::path checkpoint_path;
  volatile std::sig_atomic_t* stop_flag = nullptr;
};

class RlTrainer {
 public:
  // reference is required when kl_coef > 0 and must be null when kl_coef == 0.
  // The caller owns it, loads it in LoadMode::Inference, and does not train it.
  RlTrainer(nn::Context& ctx, nn::Transformer& policy, nn::AdamW& opt, RlConfig cfg,
            nn::Transformer* reference);
  int fit(const Tokenizer& tok, const std::vector<std::string>& prompts, RewardFn reward,
          std::mt19937& rng);
};
```

`fit` returns the last step that ran. The learning rate uses the existing cosine: copy the schedule fields into a `TrainConfig` and call `learning_rate_at`. `TrainConfig` and `Trainer::fit` are not edited.

Constructor validation follows the trainer. Throw `std::invalid_argument` whose message is the field name:

- `max_steps`, `start_step`, `batch_prompts`, `group_size`, `max_new_tokens` not positive, or `start_step > max_steps`
- `warmup_steps`, `log_every`, `ckpt_every` negative
- `learning_rate`, `min_lr`, `kl_coef`, or `grad_clip_norm` NaN, infinite, or negative
- sampling fields fail the same checks as `GenerateConfig` (`temperature`, `top_k`, `top_p`)
- policy config is not causal (`"causal"`)
- `kl_coef > 0` with a null reference, or `kl_coef == 0` with a reference (`"reference"`)
- reference config differs in `vocab_size`, `d_model`, `num_layers`, `num_heads`, `max_seq_len`, or `causal` (`"reference"`)

`fit` also throws `std::invalid_argument("prompt")` on an empty prompt list or a prompt that encodes empty, and `"max_seq_len"` when `prompt_length + max_new_tokens` exceeds the policy context. A non-finite reward or loss throws `std::runtime_error("non-finite loss")` when `abort_on_non_finite` is set.

## One step

Both models share the caller's `nn::Context`.

1. Set the learning rate. `policy.eval()` and `policy.zero_grad()`. `decode_step` throws unless the model is in eval.
2. Draw `batch_prompts` prompts with replacement. For each prompt, sample `group_size` completions as one batch through `prefill` and `decode_step`.
3. Call the reward on each episode. For `group_size == 1`, advantage is the reward. Otherwise subtract the group mean.
4. Build a right-padded batch. Shorter rows use `Tokenizer::pad_id()` on the input and `-100` on the labels. Rows that stopped on the first token add no completion labels.
5. If `kl_coef > 0`, `reference->eval()`, `reference->forward`, and copy those logits to the host before the policy forward.
6. `policy.forward` on the same tokens, still in eval, so the scored distribution is the one that sampled. Preset dropout is already 0. `decode_step`'s cache is not used for the backward.
7. `policy_gradient`, scale the gradient by `1 / batch_prompts`, `copy_from_host` into a tensor shaped like the policy logits, `policy.backward`.
8. Clip with `clip_grad_norm` the same way `Trainer::fit` does, then `opt.step`.

If a step has no completion tokens, skip `backward` and `opt.step`, log loss 0, and still count the step. A zero advantage with `kl_coef == 0` is a real zero gradient; AdamW still steps, so weight decay still applies.

Log lines use the same local-time prefix as `fit`:

```text
step <n> loss <L> reward <mean reward> kl <mean KL> lr <lr> grad_norm <norm>
```

`ckpt_every` and `stop_flag` match `Trainer`: save `checkpoint_path` with the policy AdamW, and a nonzero stop flag saves after the step and returns that step. No validation early-stop and no `min_loss`. The RL loss is not comparable across steps the way validation cross-entropy is.

## Sampling

Move the private `pick_id` path in `src/generate.cpp` to a public function on `llm/generate.hpp`:

```cpp
int sample_token(std::vector<float> logits, const GenerateConfig& cfg, std::mt19937& rng);
```

`Generator::generate` calls it. Temperature 0 stays greedy. Top-k, nucleus, and eos behavior stay as they are. `RlTrainer` splits each batched logit row and calls `sample_token`. A finished row is fed `pad_id` on later decode steps and those pad ids are not stored. `GenerateConfig::eos_id` is the policy eos (`Tokenizer::eos_id()`).

## Example

`examples/reinforce.cpp`, CMake target `reinforce`, plus `examples/reinforce.md`. It is not part of `llm_tests`.

```bash
./build/reinforce --from run --prompts prompts.txt --contains "red" --out run-rl \
  --steps 50 --group 4 --lr 1e-5 --kl 0.02 --max-new 32
```

`--from` is a `train_data` or `finetune` directory (`tokenizer.bpe`, `model.bin`). The policy loads `LoadMode::Inference` and gets a fresh AdamW. `--kl` greater than 0 loads a second Inference copy as the frozen reference. `--contains` is the toy reward: 1 when the completion contains that string, 0 otherwise. The library reward is the `RewardFn`; the flag is only this example. `--out` must differ from `--from` and receives `tokenizer.bpe` and `model.bin`.

Wire the target through `CMakeLists.txt`, including the ROCm linker-language list and the Arc `-fsycl` loop. Document it in `README.md` and `USAGE.md` beside `Trainer` and `finetune`.

## Tests

Add `tests/test_rl.cpp` to the `llm_tests` sources. Use the same `CHECK` style and the 1-layer `d_model` 16 config already in `tests/test_train.cpp`.

- Advantage 1 and `kl_coef` 0: `policy_gradient` matches `sequence_cross_entropy` on the same logits and labels, including a row of `-100`.
- A constant advantage scales that gradient.
- KL finite-difference on vocab 3 with advantage 0.
- `kl_coef` 0 leaves the gradient unchanged when reference logits are omitted.
- Constructor rejects a bidirectional config, a missing reference when `kl_coef > 0`, and a reference when `kl_coef == 0`.
- One greedy step (`temperature` 0, `group_size` 1, reward always 1, weight decay 0) changes a parameter, writes a finite log line, and saves a version-2 checkpoint whose AdamW step is 1.
- `sample_token` at temperature 0 returns the argmax. Existing `generate` behavior is unchanged.

`train_tiny` stays the supervised acceptance run. `reinforce` is not required to show a reward increase; the toy string reward on a random tiny model is not a stable test.

## Left unchanged

- `Trainer`, `TrainConfig`, masked LM, `finetune`, and `chat`.
- BERT and every non-causal preset. `RlTrainer` rejects them.
- nnCUDA, nnCL, nnROCm, nnArc, and nnAGX.
- Learned reward models, value heads, PPO clipping, and a probability ratio against an old policy.
