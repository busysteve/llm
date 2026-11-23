#include "llm/data.hpp"
#include "llm/presets.hpp"
#include "llm/train.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include <random>

namespace {

constexpr int kVocab = 320;
constexpr int kCorpus = 4096;
constexpr int kSeqLen = 64;
constexpr int kBatchSize = 4;
constexpr int kWarmupSteps = 1;
constexpr int kTimedSteps = 8;
constexpr int kLastStep = kWarmupSteps + kTimedSteps;
constexpr float kLearningRate = 3e-4f;
constexpr float kWeightDecay = 0.1f;

// fit prints the step loss and does not return it.
class StdoutCapture {
 public:
  explicit StdoutCapture(const std::filesystem::path& path) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) throw std::runtime_error("stdout");
    std::fflush(stdout);
    saved_ = ::dup(fileno(stdout));
    if (saved_ < 0 || ::dup2(fd, fileno(stdout)) < 0) {
      if (saved_ >= 0) ::close(saved_);
      saved_ = -1;
      ::close(fd);
      throw std::runtime_error("stdout");
    }
    ::close(fd);
  }

  ~StdoutCapture() { restore(); }

  void restore() {
    if (saved_ < 0) return;
    std::fflush(stdout);
    ::dup2(saved_, fileno(stdout));
    ::close(saved_);
    saved_ = -1;
  }

 private:
  int saved_ = -1;
};

std::string read_text(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("fit log");
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

bool loss_at(const std::string& log, int step, float& loss) {
  std::istringstream in(log);
  std::string line;
  bool found = false;
  while (std::getline(in, line)) {
    const char* text = line.c_str();
    int year = 0;
    int month = 0;
    int day = 0;
    int hour = 0;
    int minute = 0;
    int second = 0;
    int millis = 0;
    int consumed = 0;
    if (std::sscanf(text, "%d-%d-%d %d:%d:%d.%d %n", &year, &month, &day, &hour, &minute, &second,
                    &millis, &consumed) == 7 &&
        consumed > 0) {
      text += consumed;
    }
    int got = 0;
    float value = 0.f;
    if (std::sscanf(text, "step %d loss %f", &got, &value) == 2 && got == step) {
      loss = value;
      found = true;
    }
  }
  return found;
}

llm::TrainConfig make_cfg(int start_step, int max_steps, int log_every) {
  llm::TrainConfig cfg;
  cfg.learning_rate = kLearningRate;
  cfg.min_lr = kLearningRate;
  cfg.weight_decay = kWeightDecay;
  cfg.beta1 = 0.9f;
  cfg.beta2 = 0.95f;
  cfg.batch_size = kBatchSize;
  cfg.seq_len = kSeqLen;
  cfg.max_steps = max_steps;
  cfg.start_step = start_step;
  cfg.grad_accum_steps = 1;
  cfg.warmup_steps = 0;
  cfg.log_every = log_every;
  cfg.eval_every = 0;
  cfg.ckpt_every = 0;
  return cfg;
}

std::vector<int32_t> corpus() {
  std::mt19937 rng(1);
  std::uniform_int_distribution<int> ids(0, kVocab - 1);
  std::vector<int32_t> out(static_cast<size_t>(kCorpus));
  for (int32_t& id : out) id = static_cast<int32_t>(ids(rng));
  return out;
}

}  // namespace

int main() {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "llm-train-bench";
  try {
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::filesystem::path fit_log = dir / "fit.log";

    llm::TextDataset data(corpus(), kVocab);
    nn::Context ctx;
    const nn::TransformerConfig config = llm::presets::tiny(kVocab);
    nn::Transformer model(ctx, config, /*seed=*/1);
    nn::AdamW opt(llm::adamw_groups(model, kWeightDecay), kLearningRate, 0.9f, 0.95f, 1e-8f);

    std::mt19937 rng(1);
    {
      llm::Trainer warmup(ctx, model, opt, make_cfg(1, kWarmupSteps, 0));
      if (warmup.fit(data, nullptr, rng) != kWarmupSteps) {
        std::fprintf(stderr, "warmup\n");
        std::filesystem::remove_all(dir);
        return 1;
      }
    }
    ctx.synchronize();

    const std::vector<nn::NamedParam> params = model.parameters();
    if (params.empty() || params.front().name != "tok_emb.weight") {
      std::fprintf(stderr, "weights\n");
      std::filesystem::remove_all(dir);
      return 1;
    }
    const std::vector<float> before = params.front().value->to_host();

    const auto t0 = std::chrono::steady_clock::now();
    std::string log;
    {
      StdoutCapture capture(fit_log);
      llm::Trainer timed(ctx, model, opt, make_cfg(kWarmupSteps + 1, kLastStep, 1));
      if (timed.fit(data, nullptr, rng) != kLastStep) {
        capture.restore();
        std::fprintf(stderr, "steps\n");
        std::filesystem::remove_all(dir);
        return 1;
      }
    }
    ctx.synchronize();
    const auto t1 = std::chrono::steady_clock::now();
    log = read_text(fit_log);
    std::fputs(log.c_str(), stdout);
    if (!log.empty() && log.back() != '\n') std::fputc('\n', stdout);

    const std::vector<float> after = model.parameters().front().value->to_host();
    float last = 0.f;
    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    const double tokens =
        static_cast<double>(kTimedSteps) * static_cast<double>(kBatchSize) * static_cast<double>(kSeqLen);
    const double rate = tokens / seconds;
    const bool changed = before.size() == after.size() &&
                         std::memcmp(before.data(), after.data(), before.size() * sizeof(float)) != 0;
    if (!loss_at(log, kLastStep, last) || !std::isfinite(last) || !(seconds > 0.0) ||
        !std::isfinite(rate) || !(rate > 0.0) || opt.step_count() != static_cast<uint64_t>(kLastStep) ||
        !changed) {
      std::fprintf(stderr, "benchmark\n");
      std::filesystem::remove_all(dir);
      return 1;
    }

    std::printf(
        "benchmark preset=tiny layers=%d d_model=%d heads=%d d_ff=%d vocab=%d batch=%d seq=%d "
        "steps=%d tokens=%.0f seconds=%.6f tokens_per_s=%.1f\n",
        config.num_layers, config.d_model, config.num_heads, config.d_ff, kVocab, kBatchSize, kSeqLen,
        kTimedSteps, tokens, seconds, rate);
    std::fflush(stdout);
    std::filesystem::remove_all(dir);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "train_bench: %s\n", error.what());
    std::filesystem::remove_all(dir);
    return 1;
  }
}
