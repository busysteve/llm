#pragma once

#include "llm/tokenizer.hpp"
#include "nn/context.hpp"
#include "nn/transformer.hpp"

#include <cstdint>
#include <random>
#include <string>
#include <string_view>

namespace llm {

struct GenerateConfig {
  int max_new_tokens = 128;
  float temperature = 1.f;
  int top_k = 0;
  float top_p = 1.f;
  int32_t eos_id = -1;
};

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
