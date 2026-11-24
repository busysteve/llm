#pragma once

#include "llm/tokenizer.hpp"
#include "nn/context.hpp"
#include "nn/optim.hpp"
#include "nn/transformer.hpp"

#include <cstdint>
#include <csignal>
#include <filesystem>
#include <functional>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace llm {

// One sampled completion. prompt and text are valid only for the reward call.
// token_ids includes a trailing eos when sampling stopped on eos. text is
// decode of those ids with that trailing eos removed.
struct Episode {
  std::string_view prompt;
  std::string_view text;
  const std::vector<int32_t>& token_ids;
};

using RewardFn = std::function<float(const Episode&)>;

struct PolicyGrad {
  float loss = 0.f;
  float kl = 0.f;
  std::vector<float> grad_logits;
};

// logits and ref_logits are row-major (rows, seq, vocab). ref_logits is read
// only when kl_coef > 0. labels are (rows, seq) with ignore index -100.
// advantages has one value per row and applies to every kept label in that row.
// The gradient matches sequence_cross_entropy when every advantage is 1 and
// kl_coef is 0: (softmax - one_hot) / n_valid.
PolicyGrad policy_gradient(const float* logits, const float* ref_logits, const int64_t* labels,
                           const float* advantages, int rows, int seq, int vocab, float kl_coef);

struct RlConfig {
  float learning_rate = 1e-5f;
  float min_lr = 1e-6f;
  int warmup_steps = 0;
  int max_steps = 100;
  int start_step = 1;
  int batch_prompts = 1;
  // 1 uses the raw reward. Above 1, the advantage is reward minus the group mean.
  int group_size = 4;
  int max_new_tokens = 64;
  float temperature = 1.f;
  int top_k = 0;
  float top_p = 1.f;
  // 0 skips the reference model. Above 0, the loss adds this times the
  // full-vocabulary KL against that frozen model.
  float kl_coef = 0.f;
  int log_every = 1;
  int ckpt_every = 0;
  float grad_clip_norm = 1.f;
  bool abort_on_non_finite = true;
  std::filesystem::path checkpoint_path;
  volatile std::sig_atomic_t* stop_flag = nullptr;
};

// Causal decoder only. reference is required when kl_coef > 0 and must be null
// when kl_coef is 0. The caller owns the reference, and fit does not train it.
class RlTrainer {
 public:
  RlTrainer(nn::Context& ctx, nn::Transformer& policy, nn::AdamW& opt, RlConfig cfg,
            nn::Transformer* reference);
  int fit(const Tokenizer& tok, const std::vector<std::string>& prompts, RewardFn reward,
          std::mt19937& rng);

 private:
  nn::Context* ctx_ = nullptr;
  nn::Transformer* policy_ = nullptr;
  nn::AdamW* opt_ = nullptr;
  nn::Transformer* reference_ = nullptr;
  RlConfig cfg_;
};

}  // namespace llm
