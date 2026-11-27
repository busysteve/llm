#include "llm/rl.hpp"

#include "llm/generate.hpp"
#include "llm/train.hpp"

#include <algorithm>
#include <cstdarg>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <limits>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace llm {
namespace {

constexpr int64_t kIgnore = -100;

void log_line(const char* fmt, ...) {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(now);
  const std::time_t seconds = static_cast<std::time_t>(millis.count() / 1000);
  int ms = static_cast<int>(millis.count() % 1000);
  if (ms < 0) ms += 1000;
  std::tm local{};
  if (localtime_r(&seconds, &local) == nullptr) {
    std::fprintf(stdout, "0000-00-00 00:00:00.000 ");
  } else {
    std::fprintf(stdout, "%04d-%02d-%02d %02d:%02d:%02d.%03d ", local.tm_year + 1900,
                 local.tm_mon + 1, local.tm_mday, local.tm_hour, local.tm_min, local.tm_sec, ms);
  }
  va_list args;
  va_start(args, fmt);
  std::vfprintf(stdout, fmt, args);
  va_end(args);
  std::fflush(stdout);
}

void require_positive(int value, const char* name) {
  if (value <= 0) throw std::invalid_argument(name);
}

void require_non_negative(int value, const char* name) {
  if (value < 0) throw std::invalid_argument(name);
}

void require_rate(float value, const char* name) {
  if (!std::isfinite(value) || value < 0.f) throw std::invalid_argument(name);
}

const nn::TransformerConfig& config_of(const nn::Transformer& model) {
  return std::get<nn::TransformerConfig>(model.network().config());
}

bool same_reference(const nn::TransformerConfig& policy, const nn::TransformerConfig& reference) {
  return policy.vocab_size == reference.vocab_size && policy.d_model == reference.d_model &&
         policy.num_layers == reference.num_layers && policy.num_heads == reference.num_heads &&
         policy.max_seq_len == reference.max_seq_len && policy.causal == reference.causal;
}

void validate(const RlConfig& cfg, const nn::Transformer& policy, const nn::Transformer* reference) {
  require_rate(cfg.learning_rate, "learning_rate");
  require_rate(cfg.min_lr, "min_lr");
  require_rate(cfg.kl_coef, "kl_coef");
  require_rate(cfg.grad_clip_norm, "grad_clip_norm");
  require_positive(cfg.max_steps, "max_steps");
  require_positive(cfg.start_step, "start_step");
  if (cfg.start_step > cfg.max_steps) throw std::invalid_argument("start_step");
  require_positive(cfg.batch_prompts, "batch_prompts");
  require_positive(cfg.group_size, "group_size");
  require_positive(cfg.max_new_tokens, "max_new_tokens");
  require_non_negative(cfg.warmup_steps, "warmup_steps");
  require_non_negative(cfg.log_every, "log_every");
  require_non_negative(cfg.ckpt_every, "ckpt_every");
  if (!std::isfinite(cfg.temperature) || cfg.temperature < 0.f) throw std::invalid_argument("temperature");
  if (cfg.top_k < 0) throw std::invalid_argument("top_k");
  if (!(cfg.top_p > 0.f && cfg.top_p <= 1.f)) throw std::invalid_argument("top_p");

  const nn::TransformerConfig& policy_cfg = config_of(policy);
  if (!policy_cfg.causal) throw std::invalid_argument("causal");
  for (const nn::NamedParam& param : policy.parameters()) {
    if (param.value == nullptr || param.value->dtype() != nn::DType::Float32) {
      throw std::invalid_argument("dtype");
    }
  }
  const bool need_reference = cfg.kl_coef > 0.f;
  if (need_reference != (reference != nullptr) || reference == &policy) {
    throw std::invalid_argument("reference");
  }
  if (reference != nullptr && !same_reference(policy_cfg, config_of(*reference))) {
    throw std::invalid_argument("reference");
  }
}

struct RowSample {
  std::vector<int32_t> ids;
};

// decode_step refuses train mode. Sampling and the scoring forward both stay
// in eval so the log-probabilities belong to the distribution that drew the tokens.
std::vector<RowSample> sample_group(nn::Context& ctx, nn::Transformer& policy,
                                    const std::vector<int32_t>& prompt, int group, int max_new,
                                    int vocab, int32_t pad, const GenerateConfig& gen,
                                    const std::vector<int32_t>& unavailable, std::mt19937& rng) {
  const int prompt_len = static_cast<int>(prompt.size());
  const int t_max = prompt_len + max_new;
  std::vector<int64_t> host(static_cast<size_t>(group) * static_cast<size_t>(prompt_len));
  for (int row = 0; row < group; ++row) {
    for (int t = 0; t < prompt_len; ++t) {
      host[static_cast<size_t>(row) * static_cast<size_t>(prompt_len) + static_cast<size_t>(t)] =
          int64_t{prompt[static_cast<size_t>(t)]};
    }
  }
  nn::IndexTensor prompt_ids(ctx, {group, prompt_len});
  prompt_ids.copy_from_host(host.data(), prompt_ids.numel());
  ctx.synchronize();

  nn::Transformer::GenerateState state;
  const nn::Tensor* logits = &policy.prefill(prompt_ids, state, t_max);
  std::vector<RowSample> rows(static_cast<size_t>(group));
  std::vector<char> done(static_cast<size_t>(group), 0);
  int active = group;
  for (int n = 0; n < max_new; ++n) {
    const std::vector<float> flat = logits->to_host();
    if (flat.size() != static_cast<size_t>(group) * static_cast<size_t>(vocab)) {
      throw std::invalid_argument("logits");
    }
    std::vector<int64_t> next(static_cast<size_t>(group), int64_t{pad});
    for (int row = 0; row < group; ++row) {
      if (done[static_cast<size_t>(row)]) continue;
      const size_t begin = static_cast<size_t>(row) * static_cast<size_t>(vocab);
      std::vector<float> one(flat.begin() + static_cast<std::ptrdiff_t>(begin),
                             flat.begin() + static_cast<std::ptrdiff_t>(begin + static_cast<size_t>(vocab)));
      const int32_t id = static_cast<int32_t>(sample_token(std::move(one), gen, rng, nullptr, 0, unavailable));
      rows[static_cast<size_t>(row)].ids.push_back(id);
      if (is_stop_id(gen, id)) {
        done[static_cast<size_t>(row)] = 1;
        --active;
      } else {
        next[static_cast<size_t>(row)] = int64_t{id};
      }
    }
    if (active == 0 || n + 1 == max_new) break;
    nn::IndexTensor next_ids(ctx, {group, 1});
    next_ids.copy_from_host(next.data(), next_ids.numel());
    ctx.synchronize();
    logits = &policy.decode_step(next_ids, state);
  }
  return rows;
}

struct GroupResult {
  float loss = 0.f;
  float kl = 0.f;
  float reward_sum = 0.f;
  bool updated = false;
};

// input is full without its last token. labels[t] is full[t + 1], and a target
// that is still inside the prompt is ignore. Right-padding uses pad on the
// input and ignore on the label. A causal mask keeps that padding out of the
// real positions.
GroupResult score_group(nn::Context& ctx, nn::Transformer& policy, nn::Transformer* reference,
                        const std::string& prompt_text, const std::vector<int32_t>& prompt,
                        const std::vector<RowSample>& rows, const float* advantages, float reward_sum,
                        int vocab, int32_t pad, float kl_coef, float grad_scale, bool abort,
                        const std::vector<char>& available) {
  GroupResult result;
  result.reward_sum = reward_sum;
  const int group = static_cast<int>(rows.size());
  const int prompt_len = static_cast<int>(prompt.size());
  int max_full = prompt_len;
  int n_valid = 0;
  for (const RowSample& row : rows) {
    const int full = prompt_len + static_cast<int>(row.ids.size());
    max_full = std::max(max_full, full);
    n_valid += static_cast<int>(row.ids.size());
  }
  if (n_valid == 0 || max_full < 2) return result;

  const int seq = max_full - 1;
  const size_t count = static_cast<size_t>(group) * static_cast<size_t>(seq);
  std::vector<int64_t> input(count, int64_t{pad});
  std::vector<int64_t> labels(count, kIgnore);
  for (int row = 0; row < group; ++row) {
    std::vector<int32_t> full = prompt;
    full.insert(full.end(), rows[static_cast<size_t>(row)].ids.begin(),
                rows[static_cast<size_t>(row)].ids.end());
    const int length = static_cast<int>(full.size());
    for (int t = 0; t < seq; ++t) {
      const size_t at = static_cast<size_t>(row) * static_cast<size_t>(seq) + static_cast<size_t>(t);
      if (t < length - 1) input[at] = int64_t{full[static_cast<size_t>(t)]};
      if (t + 1 < length && t + 1 >= prompt_len) labels[at] = int64_t{full[static_cast<size_t>(t + 1)]};
    }
  }

  nn::IndexTensor tokens(ctx, {group, seq});
  tokens.copy_from_host(input.data(), tokens.numel());
  ctx.synchronize();

  std::vector<float> ref_host;
  const float* ref_ptr = nullptr;
  if (kl_coef > 0.f) {
    reference->eval();
    const nn::Tensor& ref_logits = reference->forward(tokens);
    ref_host = ref_logits.to_host();
    if (ref_host.size() != count * static_cast<size_t>(vocab)) throw std::invalid_argument("logits");
    ref_ptr = ref_host.data();
  }
  const nn::Tensor& logits = policy.forward(tokens);
  const std::vector<float> host = logits.to_host();
  if (host.size() != count * static_cast<size_t>(vocab)) throw std::invalid_argument("logits");

  PolicyGrad grad = policy_gradient(host.data(), ref_ptr, labels.data(), advantages, group, seq, vocab, kl_coef, available);
  if (abort && !std::isfinite(grad.loss)) throw std::runtime_error("non-finite loss");
  for (float& value : grad.grad_logits) value *= grad_scale;

  nn::Tensor grad_logits(ctx, logits.shape());
  if (static_cast<size_t>(grad_logits.numel()) != grad.grad_logits.size()) {
    throw std::invalid_argument("logits");
  }
  grad_logits.copy_from_host(grad.grad_logits);
  ctx.synchronize();
  policy.backward(grad_logits);
  result.loss = grad.loss;
  result.kl = grad.kl;
  result.updated = true;
  return result;
}

}  // namespace

PolicyGrad policy_gradient(const float* logits, const float* ref_logits, const int64_t* labels,
                           const float* advantages, int rows, int seq, int vocab, float kl_coef,
                           const std::vector<char>& available) {
  if (rows < 1) throw std::invalid_argument("rows");
  if (seq < 1) throw std::invalid_argument("seq");
  if (vocab <= 1) throw std::invalid_argument("vocab");
  if (!available.empty() && available.size() != static_cast<size_t>(vocab)) {
    throw std::invalid_argument("available");
  }
  auto valid = [&](int64_t id) { return available.empty() || available[static_cast<size_t>(id)]; };
  require_rate(kl_coef, "kl_coef");
  if (logits == nullptr) throw std::invalid_argument("logits");
  if (labels == nullptr) throw std::invalid_argument("labels");
  if (advantages == nullptr) throw std::invalid_argument("advantages");
  if (kl_coef > 0.f && ref_logits == nullptr) throw std::invalid_argument("reference");
  const int64_t positions = static_cast<int64_t>(rows) * seq;
  if (positions > std::numeric_limits<int64_t>::max() / vocab) throw std::invalid_argument("logits");
  const int64_t count = positions * vocab;

  PolicyGrad out;
  out.grad_logits.assign(static_cast<size_t>(count), 0.f);

  int n_valid = 0;
  for (int row = 0; row < rows; ++row) {
    if (!std::isfinite(advantages[row])) throw std::invalid_argument("advantages");
    for (int t = 0; t < seq; ++t) {
      const int64_t label = labels[static_cast<int64_t>(row) * seq + t];
      if (label == kIgnore) continue;
      if (label < 0 || label >= vocab || !valid(label)) throw std::invalid_argument("label");
      ++n_valid;
    }
  }
  if (n_valid == 0) return out;

  const double inv = 1.0 / static_cast<double>(n_valid);
  double loss_sum = 0.0;
  double kl_sum = 0.0;
  std::vector<double> probs(static_cast<size_t>(vocab));
  std::vector<double> log_probs(static_cast<size_t>(vocab));
  std::vector<double> ref_log_probs(static_cast<size_t>(vocab));

  for (int row = 0; row < rows; ++row) {
    const double advantage = static_cast<double>(advantages[row]);
    for (int t = 0; t < seq; ++t) {
      const int64_t pos = static_cast<int64_t>(row) * seq + t;
      const int64_t label = labels[pos];
      if (label == kIgnore) continue;
      const float* row_logits = logits + pos * vocab;
      float peak = -std::numeric_limits<float>::infinity();
      for (int v = 0; v < vocab; ++v) {
        if (valid(v)) peak = std::max(peak, row_logits[v]);
      }
      double sum = 0.0;
      for (int v = 0; v < vocab; ++v) {
        if (!valid(v)) continue;
        const double prob = std::exp(static_cast<double>(row_logits[v]) - static_cast<double>(peak));
        probs[static_cast<size_t>(v)] = prob;
        sum += prob;
      }
      const double log_sum = std::log(sum);
      for (int v = 0; v < vocab; ++v) {
        if (!valid(v)) continue;
        log_probs[static_cast<size_t>(v)] =
            static_cast<double>(row_logits[v]) - static_cast<double>(peak) - log_sum;
        probs[static_cast<size_t>(v)] /= sum;
      }

      double kl = 0.0;
      if (kl_coef > 0.f) {
        const float* row_ref = ref_logits + pos * vocab;
        float ref_peak = -std::numeric_limits<float>::infinity();
        for (int v = 0; v < vocab; ++v) {
          if (valid(v)) ref_peak = std::max(ref_peak, row_ref[v]);
        }
        double ref_sum = 0.0;
        for (int v = 0; v < vocab; ++v) {
          if (!valid(v)) continue;
          ref_sum += std::exp(static_cast<double>(row_ref[v]) - static_cast<double>(ref_peak));
        }
        const double ref_log_sum = std::log(ref_sum);
        for (int v = 0; v < vocab; ++v) {
          if (!valid(v)) continue;
          ref_log_probs[static_cast<size_t>(v)] =
              static_cast<double>(row_ref[v]) - static_cast<double>(ref_peak) - ref_log_sum;
          kl += probs[static_cast<size_t>(v)] *
                (log_probs[static_cast<size_t>(v)] - ref_log_probs[static_cast<size_t>(v)]);
        }
      }

      loss_sum += -advantage * log_probs[static_cast<size_t>(label)] + static_cast<double>(kl_coef) * kl;
      kl_sum += kl;
      float* row_grad = out.grad_logits.data() + pos * vocab;
      for (int v = 0; v < vocab; ++v) {
        if (!valid(v)) continue;
        const double indicator = static_cast<int64_t>(v) == label ? 1.0 : 0.0;
        double grad_v = advantage * (probs[static_cast<size_t>(v)] - indicator);
        if (kl_coef > 0.f) {
          const double log_ratio =
              log_probs[static_cast<size_t>(v)] - ref_log_probs[static_cast<size_t>(v)] - kl;
          grad_v += static_cast<double>(kl_coef) * probs[static_cast<size_t>(v)] * log_ratio;
        }
        row_grad[v] = static_cast<float>(grad_v * inv);
      }
    }
  }
  out.loss = static_cast<float>(loss_sum * inv);
  out.kl = static_cast<float>(kl_sum * inv);
  return out;
}

RlTrainer::RlTrainer(nn::Context& ctx, nn::Transformer& policy, nn::AdamW& opt, RlConfig cfg,
                     nn::Transformer* reference)
    : ctx_(&ctx), policy_(&policy), opt_(&opt), reference_(reference), cfg_(std::move(cfg)) {
  validate(cfg_, policy, reference_);
}

int RlTrainer::fit(const Tokenizer& tok, const std::vector<std::string>& prompts, RewardFn reward,
                   std::mt19937& rng) {
  if (prompts.empty()) throw std::invalid_argument("prompt");
  const nn::TransformerConfig& policy_cfg = config_of(*policy_);
  const int vocab = policy_cfg.vocab_size;
  const auto unavailable = tok.unavailable_ids(vocab);
  std::vector<char> available;
  if (!unavailable.empty()) {
    available.assign(static_cast<size_t>(vocab), 1);
    for (int32_t id : unavailable) available[id] = 0;
  }
  if (!tok.has_token(tok.pad_id())) throw std::invalid_argument("pad_id");
  const auto stops = tok.stop_ids();
  if (stops.empty() && !tok.has_token(tok.eos_id())) throw std::invalid_argument("eos_id");

  std::vector<std::vector<int32_t>> encoded;
  encoded.reserve(prompts.size());
  for (const std::string& prompt : prompts) {
    std::vector<int32_t> ids = tok.encode(prompt, {tok.add_bos_default(), tok.add_eos_default()});
    for (int32_t id : ids) {
      if (!tok.has_token(id)) throw std::invalid_argument("prompt token id");
    }
    if (ids.empty()) throw std::invalid_argument("prompt");
    if (ids.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument("max_seq_len");
    }
    const int prompt_len = static_cast<int>(ids.size());
    if (cfg_.max_new_tokens > std::numeric_limits<int>::max() - prompt_len ||
        prompt_len + cfg_.max_new_tokens > policy_cfg.max_seq_len) {
      throw std::invalid_argument("max_seq_len");
    }
    encoded.push_back(std::move(ids));
  }

  GenerateConfig gen;
  gen.max_new_tokens = cfg_.max_new_tokens;
  gen.temperature = cfg_.temperature;
  gen.top_k = cfg_.top_k;
  gen.top_p = cfg_.top_p;
  gen.eos_id = tok.eos_id();
  gen.stop_ids = stops;

  TrainConfig schedule;
  schedule.learning_rate = cfg_.learning_rate;
  schedule.min_lr = cfg_.min_lr;
  schedule.warmup_steps = cfg_.warmup_steps;
  schedule.max_steps = cfg_.max_steps;

  std::uniform_int_distribution<size_t> pick(0, prompts.size() - 1);
  const float grad_scale = 1.f / static_cast<float>(cfg_.batch_prompts);

  for (int step = cfg_.start_step; step <= cfg_.max_steps; ++step) {
    const float lr = learning_rate_at(schedule, step);
    opt_->set_learning_rate(lr);
    policy_->eval();
    policy_->zero_grad();

    float loss_sum = 0.f;
    float kl_sum = 0.f;
    float reward_sum = 0.f;
    int episodes = 0;
    bool updated = false;
    for (int micro = 0; micro < cfg_.batch_prompts; ++micro) {
      const size_t which = pick(rng);
      const std::vector<RowSample> rows =
          sample_group(*ctx_, *policy_, encoded[which], cfg_.group_size, cfg_.max_new_tokens, vocab,
                       tok.pad_id(), gen, unavailable, rng);

      std::vector<float> rewards(static_cast<size_t>(cfg_.group_size));
      float micro_reward = 0.f;
      for (int row = 0; row < cfg_.group_size; ++row) {
        const std::vector<int32_t>& ids = rows[static_cast<size_t>(row)].ids;
        std::vector<int32_t> shown = ids;
        if (!shown.empty() && is_stop_id(gen, shown.back())) shown.pop_back();
        const std::string text = tok.decode(shown);
        const Episode episode{prompts[which], text, ids};
        const float value = reward(episode);
        if (cfg_.abort_on_non_finite && !std::isfinite(value)) {
          throw std::runtime_error("non-finite loss");
        }
        rewards[static_cast<size_t>(row)] = value;
        micro_reward += value;
      }
      std::vector<float> advantages(rewards.size());
      if (cfg_.group_size == 1) {
        advantages[0] = rewards[0];
      } else {
        const float mean = micro_reward / static_cast<float>(cfg_.group_size);
        for (size_t row = 0; row < rewards.size(); ++row) advantages[row] = rewards[row] - mean;
      }

      const GroupResult group =
          score_group(*ctx_, *policy_, reference_, prompts[which], encoded[which], rows,
                      advantages.data(), micro_reward, vocab, tok.pad_id(), cfg_.kl_coef, grad_scale,
                      cfg_.abort_on_non_finite, available);
      loss_sum += group.loss;
      kl_sum += group.kl;
      reward_sum += group.reward_sum;
      episodes += cfg_.group_size;
      updated = updated || group.updated;
    }

    const float loss_value = loss_sum / static_cast<float>(cfg_.batch_prompts);
    const float kl_value = kl_sum / static_cast<float>(cfg_.batch_prompts);
    const float reward_value =
        episodes > 0 ? reward_sum / static_cast<float>(episodes) : 0.f;

    float grad_norm = 0.f;
    if (updated) {
      if (cfg_.grad_clip_norm > 0.f) {
        grad_norm = nn::clip_grad_norm(policy_->parameters(), cfg_.grad_clip_norm);
      } else if (cfg_.abort_on_non_finite) {
        grad_norm = nn::clip_grad_norm(policy_->parameters(), std::numeric_limits<float>::max());
      }
      if (cfg_.abort_on_non_finite && !std::isfinite(grad_norm)) {
        throw std::runtime_error("non-finite grad");
      }
      opt_->step();
    }

    if (cfg_.log_every > 0 && step % cfg_.log_every == 0) {
      log_line("step %d loss %.6f reward %.6f kl %.6f lr %.8g grad_norm %.6f\n", step, loss_value,
               reward_value, kl_value, lr, grad_norm);
    }

    if (cfg_.stop_flag != nullptr && *cfg_.stop_flag != 0) {
      policy_->zero_grad();
      if (!cfg_.checkpoint_path.empty()) policy_->save(cfg_.checkpoint_path, opt_);
      log_line("stop step %d\n", step);
      return step;
    }

    if (cfg_.ckpt_every > 0 && step % cfg_.ckpt_every == 0 && !cfg_.checkpoint_path.empty()) {
      policy_->save(cfg_.checkpoint_path, opt_);
    }
  }
  return cfg_.max_steps;
}

}  // namespace llm
