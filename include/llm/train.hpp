#pragma once

#include "llm/data.hpp"
#include "nn/optim.hpp"
#include "nn/transformer.hpp"

#include <filesystem>
#include <random>

namespace llm {

struct TrainConfig {
  float learning_rate = 3e-4f;
  float weight_decay = 0.1f;
  float beta1 = 0.9f;
  float beta2 = 0.95f;
  int batch_size = 8;
  int seq_len = 512;
  int max_steps = 10000;
  int grad_accum_steps = 1;
  int warmup_steps = 100;
  float min_lr = 3e-5f;
  int log_every = 50;
  int eval_every = 500;
  int eval_batches = 20;
  int ckpt_every = 1000;
  float grad_clip_norm = 1.f;
  bool abort_on_non_finite = true;
  std::filesystem::path checkpoint_path;
};

// step is 1-based. This is the schedule fit uses.
float learning_rate_at(const TrainConfig& cfg, int step);

class Trainer {
 public:
  Trainer(nn::Context& ctx, nn::Transformer& model, nn::AdamW& opt, TrainConfig cfg);
  void fit(TextDataset& train, TextDataset* valid, std::mt19937& rng);

 private:
  nn::Context* ctx_ = nullptr;
  nn::Transformer* model_ = nullptr;
  nn::AdamW* opt_ = nullptr;
  TrainConfig cfg_;
};

}  // namespace llm
