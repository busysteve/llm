#include "llm/bpe.hpp"
#include "llm/data.hpp"
#include "llm/generate.hpp"
#include "llm/presets.hpp"
#include "llm/train.hpp"

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

constexpr int kVocabSize = 320;
constexpr int kSeqLen = 32;
constexpr int kBatchSize = 4;
constexpr int kMaxSteps = 30;
constexpr float kLearningRate = 3e-3f;
constexpr float kWeightDecay = 0.1f;
constexpr char kParagraph[] =
    "A tiny model can learn a short paragraph. "
    "The same words come back, so the next token is easier. "
    "Loss should fall after a few dozen steps. ";

// train throws vocab_size when the paragraph runs out of pairs. A longer copy still has pairs.
std::unique_ptr<llm::BpeTokenizer> train_tokenizer(std::string& corpus, std::vector<int32_t>& ids) {
  corpus.clear();
  for (int copies = 0; copies < 64; ++copies) {
    corpus += kParagraph;
    try {
      std::unique_ptr<llm::BpeTokenizer> tok = llm::BpeTokenizer::train({corpus}, kVocabSize);
      ids = tok->encode(corpus);
      if (ids.size() >= static_cast<size_t>(kSeqLen) + 1) return tok;
    } catch (const std::invalid_argument& error) {
      if (std::strcmp(error.what(), "vocab_size") != 0) throw;
    }
  }
  throw std::runtime_error("vocab_size");
}

// fit prints the step loss and does not return it. Compare step 1 with the last step from that log.
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
    int got = 0;
    float value = 0.f;
    if (std::sscanf(line.c_str(), "step %d loss %f", &got, &value) == 2 && got == step) {
      loss = value;
      found = true;
    }
  }
  return found;
}

}  // namespace

int main() {
  try {
    std::string corpus;
    std::vector<int32_t> ids;
    const std::unique_ptr<llm::BpeTokenizer> tok = train_tokenizer(corpus, ids);

    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "llm-train-tiny";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::filesystem::path bin = dir / "tokens.bin";
    const std::filesystem::path meta = dir / "tokens.meta";
    const std::filesystem::path fit_log = dir / "fit.log";
    llm::write_token_file(bin, meta, ids, static_cast<int32_t>(tok->vocab_size()));
    llm::TextDataset data(bin, meta);

    nn::Context ctx;
    nn::Transformer model(ctx, llm::presets::tiny(tok->vocab_size()), /*seed=*/1);
    nn::AdamW opt(llm::adamw_groups(model, kWeightDecay), kLearningRate, 0.9f, 0.95f, 1e-8f);

    llm::TrainConfig cfg;
    cfg.learning_rate = kLearningRate;
    cfg.min_lr = kLearningRate;
    cfg.weight_decay = kWeightDecay;
    cfg.beta1 = 0.9f;
    cfg.beta2 = 0.95f;
    cfg.batch_size = kBatchSize;
    cfg.seq_len = kSeqLen;
    cfg.max_steps = kMaxSteps;
    cfg.grad_accum_steps = 1;
    cfg.warmup_steps = 0;
    cfg.log_every = 1;
    cfg.eval_every = 0;
    cfg.ckpt_every = 0;

    std::mt19937 rng(1);
    {
      StdoutCapture capture(fit_log);
      llm::Trainer trainer(ctx, model, opt, cfg);
      trainer.fit(data, nullptr, rng);
    }
    const std::string log = read_text(fit_log);
    std::fputs(log.c_str(), stdout);
    if (!log.empty() && log.back() != '\n') std::fputc('\n', stdout);

    float first = 0.f;
    float last = 0.f;
    if (!loss_at(log, 1, first) || !loss_at(log, kMaxSteps, last)) {
      std::fprintf(stderr, "missing step loss\n");
      std::filesystem::remove_all(dir);
      return 1;
    }
    std::printf("first_loss %.6f\nlast_loss %.6f\n", first, last);
    std::fflush(stdout);

    llm::Generator generator(ctx, model, *tok);
    llm::GenerateConfig gen;
    gen.max_new_tokens = 8;
    gen.temperature = 0.f;
    gen.top_k = 1;
    const std::string prompt = "A tiny model";
    std::string text;
    try {
      text = generator.generate(prompt, gen, rng);
    } catch (const std::exception& error) {
      std::fprintf(stderr, "generate failed: %s\n", error.what());
      std::filesystem::remove_all(dir);
      return 1;
    }
    std::printf("%s\n", text.c_str());
    std::fflush(stdout);
    std::filesystem::remove_all(dir);

    if (!std::isfinite(first) || !std::isfinite(last) || !(last < first)) {
      std::fprintf(stderr, "loss did not fall\n");
      return 1;
    }
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "train_tiny: %s\n", error.what());
    return 1;
  }
}
