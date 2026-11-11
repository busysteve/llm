#include "llm/train.hpp"

#include "nn/loss.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace llm {
namespace {

constexpr float kPi = 3.14159265358979323846f;

void require_positive(int value, const char* name) {
  if (value <= 0) throw std::invalid_argument(name);
}

void require_non_negative(int value, const char* name) {
  if (value < 0) throw std::invalid_argument(name);
}

void validate(const TrainConfig& cfg) {
  if (!std::isfinite(cfg.learning_rate) || cfg.learning_rate < 0.f) {
    throw std::invalid_argument("learning_rate");
  }
  if (!std::isfinite(cfg.min_lr) || cfg.min_lr < 0.f) throw std::invalid_argument("min_lr");
  require_positive(cfg.batch_size, "batch_size");
  require_positive(cfg.seq_len, "seq_len");
  require_positive(cfg.max_steps, "max_steps");
  require_positive(cfg.grad_accum_steps, "grad_accum_steps");
  require_non_negative(cfg.warmup_steps, "warmup_steps");
  require_non_negative(cfg.log_every, "log_every");
  require_non_negative(cfg.eval_every, "eval_every");
  require_non_negative(cfg.eval_batches, "eval_batches");
  require_non_negative(cfg.ckpt_every, "ckpt_every");
  require_non_negative(cfg.early_stop_patience, "early_stop_patience");
  if (!std::isfinite(cfg.early_stop_min_delta) || cfg.early_stop_min_delta < 0.f) {
    throw std::invalid_argument("early_stop_min_delta");
  }
  // NaN leaves the threshold off. Inf would stop on every finite loss.
  if (!std::isnan(cfg.min_loss) && (!std::isfinite(cfg.min_loss) || cfg.min_loss < 0.f)) {
    throw std::invalid_argument("min_loss");
  }
  if (!std::isfinite(cfg.grad_clip_norm) || cfg.grad_clip_norm < 0.f) {
    throw std::invalid_argument("grad_clip_norm");
  }
}

struct BestStep {
  std::vector<std::vector<float>> params;
  std::vector<std::vector<float>> m;
  std::vector<std::vector<float>> v;
  uint64_t opt_step = 0;
  float learning_rate = 0.f;
  int step = 0;
  float eval_ce = 0.f;
};

void capture_best(BestStep& best, const nn::Transformer& model, const nn::AdamW& opt, int step,
                  float eval_ce) {
  const std::vector<nn::NamedParam> params = model.parameters();
  best.params.clear();
  best.params.reserve(params.size());
  for (const nn::NamedParam& param : params) best.params.push_back(param.value->to_host());
  best.m.clear();
  best.v.clear();
  best.m.reserve(opt.m().size());
  best.v.reserve(opt.v().size());
  for (const nn::Tensor& moment : opt.m()) best.m.push_back(moment.to_host());
  for (const nn::Tensor& moment : opt.v()) best.v.push_back(moment.to_host());
  best.opt_step = opt.step_count();
  best.learning_rate = opt.learning_rate();
  best.step = step;
  best.eval_ce = eval_ce;
}

void restore_best(const BestStep& best, nn::Context& ctx, nn::Transformer& model, nn::AdamW& opt) {
  const std::vector<nn::NamedParam> params = model.parameters();
  if (params.size() != best.params.size() || opt.m().size() != best.m.size() ||
      opt.v().size() != best.v.size()) {
    throw std::runtime_error("early_stop");
  }
  for (size_t i = 0; i < params.size(); ++i) params[i].value->copy_from_host(best.params[i]);
  for (size_t i = 0; i < best.m.size(); ++i) {
    opt.m()[i].copy_from_host(best.m[i]);
    opt.v()[i].copy_from_host(best.v[i]);
  }
  opt.set_step_count(best.opt_step);
  opt.set_learning_rate(best.learning_rate);
  // copy_from_host is async on CUDA. The host snapshot has to outlive that copy.
  ctx.synchronize();
  model.zero_grad();
  model.train();
}

float mean_loss(TextDataset& data, nn::Context& ctx, nn::Transformer& model, int batches, int batch_size,
                int seq_len, std::mt19937& rng) {
  double sum = 0.0;
  for (int i = 0; i < batches; ++i) {
    Batch batch = data.next(ctx, batch_size, seq_len, rng);
    const nn::Tensor& logits = model.forward(batch.input_ids);
    const nn::LossResult loss = nn::sequence_cross_entropy(ctx, logits, batch.labels, -100);
    sum += static_cast<double>(loss.loss);
  }
  return static_cast<float>(sum / static_cast<double>(batches));
}

}  // namespace

float learning_rate_at(const TrainConfig& cfg, int step) {
  if (cfg.warmup_steps > 0 && step <= cfg.warmup_steps) {
    // Endpoints are exact. lr * n / n and cos(pi) do not round-trip in float.
    if (step == cfg.warmup_steps) return cfg.learning_rate;
    return cfg.learning_rate * static_cast<float>(step) / static_cast<float>(cfg.warmup_steps);
  }
  if (cfg.max_steps > cfg.warmup_steps) {
    if (step == cfg.max_steps) return cfg.min_lr;
    const float progress = static_cast<float>(step - cfg.warmup_steps) /
                           static_cast<float>(cfg.max_steps - cfg.warmup_steps);
    const float cosine = std::cos(kPi * progress);
    return cfg.min_lr + 0.5f * (cfg.learning_rate - cfg.min_lr) * (1.f + cosine);
  }
  return cfg.learning_rate;
}

Trainer::Trainer(nn::Context& ctx, nn::Transformer& model, nn::AdamW& opt, TrainConfig cfg)
    : ctx_(&ctx), model_(&model), opt_(&opt), cfg_(std::move(cfg)) {
  validate(cfg_);
}

int Trainer::fit(TextDataset& train, TextDataset* valid, std::mt19937& rng) {
  if (cfg_.early_stop_patience > 0 &&
      (valid == nullptr || cfg_.eval_every < 1 || cfg_.eval_batches < 1 ||
       cfg_.eval_every > cfg_.max_steps)) {
    throw std::invalid_argument("early_stop_patience");
  }

  BestStep best;
  bool have_best = false;
  int waited = 0;
  int last_step = cfg_.max_steps;
  bool hit_min_loss = false;
  float stop_loss = 0.f;

  for (int step = 1; step <= cfg_.max_steps; ++step) {
    const float lr = learning_rate_at(cfg_, step);
    opt_->set_learning_rate(lr);
    model_->train();
    model_->zero_grad();

    float loss_sum = 0.f;
    for (int micro = 0; micro < cfg_.grad_accum_steps; ++micro) {
      Batch batch = train.next(*ctx_, cfg_.batch_size, cfg_.seq_len, rng);
      const nn::Tensor& logits = model_->forward(batch.input_ids);
      nn::LossResult loss = nn::sequence_cross_entropy(*ctx_, logits, batch.labels, -100);
      if (cfg_.abort_on_non_finite && !std::isfinite(loss.loss)) {
        throw std::runtime_error("non-finite loss");
      }
      loss.grad_logits.scale(1.f / static_cast<float>(cfg_.grad_accum_steps));
      model_->backward(loss.grad_logits);
      loss_sum += loss.loss;
    }
    const float loss_value = loss_sum / static_cast<float>(cfg_.grad_accum_steps);

    float grad_norm = 0.f;
    if (cfg_.grad_clip_norm > 0.f) {
      grad_norm = nn::clip_grad_norm(model_->parameters(), cfg_.grad_clip_norm);
    } else if (cfg_.abort_on_non_finite) {
      // clip_grad_norm is the only grad reduction. A huge limit leaves a finite norm unscaled.
      grad_norm = nn::clip_grad_norm(model_->parameters(), std::numeric_limits<float>::max());
    }
    if (cfg_.abort_on_non_finite && !std::isfinite(grad_norm)) {
      throw std::runtime_error("non-finite grad");
    }

    opt_->step();

    if (cfg_.log_every > 0 && step % cfg_.log_every == 0) {
      std::printf("step %d loss %.6f lr %.8g grad_norm %.6f\n", step, loss_value, lr, grad_norm);
      std::fflush(stdout);
    }

    // The logged loss is this step's mean, measured before the update.
    // Keep the weights from that update and skip an older validation restore.
    if (std::isfinite(cfg_.min_loss) && std::isfinite(loss_value) && loss_value <= cfg_.min_loss) {
      hit_min_loss = true;
      stop_loss = loss_value;
      last_step = step;
      break;
    }

    bool stop = false;
    if (valid != nullptr && cfg_.eval_every > 0 && cfg_.eval_batches > 0 && step % cfg_.eval_every == 0) {
      model_->eval();
      const float mean_ce = mean_loss(*valid, *ctx_, *model_, cfg_.eval_batches, cfg_.batch_size,
                                      cfg_.seq_len, rng);
      const float perplexity = std::exp(mean_ce);
      std::printf("step %d eval_ce %.6f perplexity %.6f\n", step, mean_ce, perplexity);
      std::fflush(stdout);
      model_->train();

      if (cfg_.early_stop_patience > 0) {
        // The first finite eval is the baseline. A later drop has to beat min_delta.
        const bool improved = std::isfinite(mean_ce) &&
                              (!have_best || (best.eval_ce - mean_ce) > cfg_.early_stop_min_delta);
        if (!std::isfinite(mean_ce) && cfg_.abort_on_non_finite) {
          throw std::runtime_error("non-finite loss");
        }
        if (improved) {
          capture_best(best, *model_, *opt_, step, mean_ce);
          have_best = true;
          waited = 0;
        } else if (++waited >= cfg_.early_stop_patience) {
          stop = true;
        }
      }
    }

    if (stop) {
      last_step = step;
      break;
    }

    if (cfg_.ckpt_every > 0 && step % cfg_.ckpt_every == 0 && !cfg_.checkpoint_path.empty()) {
      model_->save(cfg_.checkpoint_path, opt_);
    }
  }

  if (hit_min_loss) {
    model_->zero_grad();
    model_->train();
    if (!cfg_.checkpoint_path.empty()) model_->save(cfg_.checkpoint_path, opt_);
    std::printf("min_loss step %d loss %.6f\n", last_step, stop_loss);
    std::fflush(stdout);
    return last_step;
  }

  if (have_best) {
    // Moments are restored with the weights so a resumed AdamW matches that step.
    restore_best(best, *ctx_, *model_, *opt_);
    if (!cfg_.checkpoint_path.empty()) model_->save(cfg_.checkpoint_path, opt_);
    std::printf("early_stop step %d best_step %d best_eval_ce %.6f\n", last_step, best.step,
                best.eval_ce);
    std::fflush(stdout);
  }
  return last_step;
}

}  // namespace llm
