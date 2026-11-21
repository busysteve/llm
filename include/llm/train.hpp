#pragma once

#include "llm/data.hpp"
#include "nn/optim.hpp"
#include "nn/transformer.hpp"

#include <csignal>
#include <filesystem>
#include <limits>
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
  // First step fit runs. 1-based, and at most max_steps. A resumed run sets
  // this to the checkpoint AdamW step plus one so the schedule continues.
  int start_step = 1;
  int grad_accum_steps = 1;
  int warmup_steps = 100;
  float min_lr = 3e-5f;
  int log_every = 50;
  int eval_every = 500;
  int eval_batches = 20;
  int ckpt_every = 1000;
  float grad_clip_norm = 1.f;
  bool abort_on_non_finite = true;
  // 0 runs every step. A positive count stops after that many validation
  // evals with no improvement and restores the best step.
  int early_stop_patience = 0;
  float early_stop_min_delta = 0.f;
  // NaN is off. A finite value >= 0 stops when a step's training loss is at
  // or below it and keeps that step.
  float min_loss = std::numeric_limits<float>::quiet_NaN();
  // Masked language modeling. Labels are the original ids at masked positions
  // and -100 elsewhere. 80% of chosen positions become mask_token, 10% a
  // random id, and 10% stay. mask_token must be inside the model vocabulary.
  bool masked_lm = false;
  float mask_prob = 0.15f;
  int mask_token = -1;
  std::filesystem::path checkpoint_path;
  // Polled after each optimizer step. A nonzero value saves checkpoint_path
  // and returns that step. A signal handler may set the flag.
  volatile std::sig_atomic_t* stop_flag = nullptr;
};

// step is 1-based. This is the schedule fit uses.
float learning_rate_at(const TrainConfig& cfg, int step);

class Trainer {
 public:
  Trainer(nn::Context& ctx, nn::Transformer& model, nn::AdamW& opt, TrainConfig cfg);
  // Last step that ran. Less than max_steps when patience, min_loss, or the
  // stop flag ends the run before the last scheduled step.
  int fit(TextDataset& train, TextDataset* valid, std::mt19937& rng);

 private:
  nn::Context* ctx_ = nullptr;
  nn::Transformer* model_ = nullptr;
  nn::AdamW* opt_ = nullptr;
  TrainConfig cfg_;
};

}  // namespace llm
