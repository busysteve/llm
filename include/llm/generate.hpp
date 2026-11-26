#pragma once

#include "llm/tokenizer.hpp"
#include "nn/context.hpp"
#include "nn/transformer.hpp"

#include <cstdint>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace llm {

struct GenerateConfig {
  int max_new_tokens = 128;
  float temperature = 1.f;
  int top_k = 0;
  float top_p = 1.f;
  // 1 leaves logits unchanged. Above 1 divides a positive logit of a seen id
  // and multiplies a negative one, once per distinct id.
  float repetition_penalty = 1.f;
  int32_t eos_id = -1;
  // Empty stops on eos_id. A non-empty list stops on those ids and leaves them
  // out of the decoded string.
  std::vector<int32_t> stop_ids;
};

inline bool is_stop_id(const GenerateConfig& cfg, int32_t id) {
  if (!cfg.stop_ids.empty()) {
    for (int32_t stop : cfg.stop_ids) {
      if (stop == id) return true;
    }
    return false;
  }
  return cfg.eos_id >= 0 && id == cfg.eos_id;
}

// Temperature 0 returns the argmax. Otherwise logits are divided by temperature,
// restricted to top_k (0 skips that cut), softmaxed, nucleus-filtered, and drawn.
// seen is the prompt and the ids already sampled. A null seen skips the penalty.
int sample_token(std::vector<float> logits, const GenerateConfig& cfg, std::mt19937& rng,
                 const int32_t* seen = nullptr, size_t n_seen = 0);

class Generator {
 public:
  Generator(nn::Context& ctx, nn::Transformer& model, const Tokenizer& tok);
  std::string generate(std::string_view prompt, GenerateConfig cfg, std::mt19937& rng);

 private:
  nn::Context* ctx_ = nullptr;
  nn::Transformer* model_ = nullptr;
  const Tokenizer* tok_ = nullptr;
};

}  // namespace llm
