#include "llm/generate.hpp"

#include "nn/index_tensor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace llm {
namespace {

void validate(const GenerateConfig& cfg) {
  if (cfg.max_new_tokens < 1) throw std::invalid_argument("max_new_tokens");
  if (!std::isfinite(cfg.temperature) || cfg.temperature < 0.f) throw std::invalid_argument("temperature");
  if (cfg.top_k < 0) throw std::invalid_argument("top_k");
  if (!(cfg.top_p > 0.f && cfg.top_p <= 1.f)) throw std::invalid_argument("top_p");
}

int argmax_id(const std::vector<float>& logits) {
  size_t best = 0;
  for (size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > logits[best]) best = i;
  }
  return static_cast<int>(best);
}

void mask_top_k(std::vector<float>& logits, int top_k) {
  if (top_k <= 0 || top_k >= static_cast<int>(logits.size())) return;
  std::vector<int> order(logits.size());
  std::iota(order.begin(), order.end(), 0);
  std::nth_element(order.begin(), order.begin() + (top_k - 1), order.end(),
                   [&](int a, int b) { return logits[static_cast<size_t>(a)] > logits[static_cast<size_t>(b)]; });
  std::vector<char> keep(logits.size(), 0);
  for (int i = 0; i < top_k; ++i) keep[static_cast<size_t>(order[static_cast<size_t>(i)])] = 1;
  for (size_t i = 0; i < logits.size(); ++i) {
    if (!keep[i]) logits[i] = -std::numeric_limits<float>::infinity();
  }
}

std::vector<double> softmax(const std::vector<float>& logits) {
  float peak = -std::numeric_limits<float>::infinity();
  for (float value : logits) peak = std::max(peak, value);
  std::vector<double> probs(logits.size(), 0.0);
  if (!std::isfinite(peak)) return probs;
  double sum = 0.0;
  for (size_t i = 0; i < logits.size(); ++i) {
    const double p = std::exp(static_cast<double>(logits[i] - peak));
    probs[i] = p;
    sum += p;
  }
  if (sum > 0.0) {
    for (double& p : probs) p /= sum;
  }
  return probs;
}

void mask_top_p(std::vector<double>& probs, float top_p) {
  if (!(top_p < 1.f)) return;
  std::vector<int> order(probs.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](int a, int b) { return probs[static_cast<size_t>(a)] > probs[static_cast<size_t>(b)]; });
  double cumulative = 0.0;
  size_t keep = 0;
  for (; keep < order.size(); ++keep) {
    cumulative += probs[static_cast<size_t>(order[keep])];
    if (cumulative >= static_cast<double>(top_p)) {
      ++keep;
      break;
    }
  }
  for (size_t i = keep; i < order.size(); ++i) probs[static_cast<size_t>(order[i])] = 0.0;
  double sum = 0.0;
  for (double p : probs) sum += p;
  if (sum > 0.0) {
    for (double& p : probs) p /= sum;
  }
}

int sample_id(const std::vector<double>& probs, std::mt19937& rng) {
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  const double draw = dist(rng);
  double cumulative = 0.0;
  int last = static_cast<int>(probs.size()) - 1;
  for (size_t i = 0; i < probs.size(); ++i) {
    if (probs[i] > 0.0) last = static_cast<int>(i);
    cumulative += probs[i];
    if (draw <= cumulative) return static_cast<int>(i);
  }
  return last;
}

int pick_id(std::vector<float> logits, const GenerateConfig& cfg, std::mt19937& rng) {
  if (cfg.temperature == 0.f) return argmax_id(logits);
  for (float& value : logits) value /= cfg.temperature;
  mask_top_k(logits, cfg.top_k);
  std::vector<double> probs = softmax(logits);
  mask_top_p(probs, cfg.top_p);
  return sample_id(probs, rng);
}

}  // namespace

Generator::Generator(nn::Context& ctx, nn::Transformer& model, const Tokenizer& tok)
    : ctx_(&ctx), model_(&model), tok_(&tok) {}

std::string Generator::generate(std::string_view prompt, GenerateConfig cfg, std::mt19937& rng) {
  validate(cfg);
  model_->eval();

  const std::vector<int32_t> encoded = tok_->encode(prompt);
  if (encoded.empty()) throw std::invalid_argument("prompt");
  if (encoded.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("max_seq_len");
  }
  const int prompt_len = static_cast<int>(encoded.size());
  if (cfg.max_new_tokens > std::numeric_limits<int>::max() - prompt_len) {
    throw std::invalid_argument("max_seq_len");
  }
  const int t_max = prompt_len + cfg.max_new_tokens;
  const nn::TransformerConfig& model_cfg = std::get<nn::TransformerConfig>(model_->network().config());
  if (t_max > model_cfg.max_seq_len) throw std::invalid_argument("max_seq_len");

  std::vector<int64_t> host(encoded.size());
  for (size_t i = 0; i < encoded.size(); ++i) host[i] = int64_t{encoded[i]};
  nn::IndexTensor prompt_ids(*ctx_, {1, prompt_len});
  prompt_ids.copy_from_host(host.data(), prompt_ids.numel());
  ctx_->synchronize();

  nn::Transformer::GenerateState state;
  const nn::Tensor* logits = &model_->prefill(prompt_ids, state, t_max);

  std::vector<int32_t> generated;
  generated.reserve(static_cast<size_t>(cfg.max_new_tokens));
  for (int n = 0; n < cfg.max_new_tokens; ++n) {
    const int32_t id = static_cast<int32_t>(pick_id(logits->to_host(), cfg, rng));
    if (cfg.eos_id >= 0 && id == cfg.eos_id) break;
    generated.push_back(id);
    if (n + 1 == cfg.max_new_tokens) break;

    std::vector<int64_t> one{int64_t{id}};
    nn::IndexTensor next_ids(*ctx_, {1, 1});
    next_ids.copy_from_host(one.data(), next_ids.numel());
    ctx_->synchronize();
    logits = &model_->decode_step(next_ids, state);
  }
  return tok_->decode(generated);
}

}  // namespace llm
