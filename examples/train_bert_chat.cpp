#include "llm/bpe.hpp"
#include "llm/data.hpp"
#include "llm/generate.hpp"
#include "llm/presets.hpp"
#include "llm/train.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace {

constexpr int kVocabSize = 320;
constexpr int kSeqLen = 32;
constexpr int kBatchSize = 4;
constexpr int kMaxSteps = 30;
constexpr float kLearningRate = 3e-3f;
constexpr float kWeightDecay = 0.1f;
constexpr char kQuestion[] = "What is the harbor bell for?";

constexpr const char* kExchanges[] = {
    "User: What is the harbor bell for?\n"
    "Assistant: It marks the hour.\n",
    "User: Where does the bell stand?\n"
    "Assistant: The bell stands on the harbor wall.\n",
    "User: When does the harbor bell ring?\n"
    "Assistant: The harbor bell rings at noon.\n",
    "User: Who rings the harbor bell?\n"
    "Assistant: The keeper rings it when the tide turns.\n",
};

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

struct RemoveAll {
  std::filesystem::path path;
  ~RemoveAll() {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
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

void print_log(const std::string& log) {
  std::fputs(log.c_str(), stdout);
  if (!log.empty() && log.back() != '\n') std::fputc('\n', stdout);
  std::fflush(stdout);
}

std::vector<int32_t> encode_corpus(const llm::Tokenizer& tok) {
  llm::EncodeOptions opt;
  opt.add_eos = true;
  std::vector<int32_t> ids;
  for (const char* exchange : kExchanges) {
    const std::vector<int32_t> piece = tok.encode(exchange, opt);
    ids.insert(ids.end(), piece.begin(), piece.end());
  }
  return ids;
}

// Prompt shape matches chat: "User: " + line + newline + "Assistant:".
std::string chat_reply(nn::Context& ctx, nn::Transformer& model, const llm::Tokenizer& tok,
                       std::mt19937& rng) {
  llm::Generator generator(ctx, model, tok);
  llm::GenerateConfig gen;
  gen.max_new_tokens = 16;
  gen.temperature = 0.f;
  gen.top_k = 1;
  gen.eos_id = tok.eos_id();
  std::string prompt = "User: ";
  prompt += kQuestion;
  prompt += "\nAssistant:";
  std::string reply = generator.generate(prompt, gen, rng);
  const size_t cut = reply.find("\nUser:");
  if (cut != std::string::npos) reply.resize(cut);
  if (!reply.empty() && reply.front() == ' ') reply.erase(reply.begin());
  return reply;
}

}  // namespace

int main() {
  try {
    const std::vector<std::string> exchanges(std::begin(kExchanges), std::end(kExchanges));
    const std::unique_ptr<llm::BpeTokenizer> tok = llm::BpeTokenizer::train(exchanges, kVocabSize);
    const std::vector<int32_t> ids = encode_corpus(*tok);
    llm::TextDataset data(ids, static_cast<int32_t>(tok->vocab_size()), tok->eos_id());

    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "llm-train-bert-chat";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    RemoveAll cleanup{dir};

    nn::Context ctx;
    nn::Transformer model(ctx, llm::presets::bert_tiny(tok->vocab_size()), /*seed=*/1);
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
    cfg.masked_lm = true;
    cfg.mask_token = tok->pad_id();
    cfg.mask_prob = 0.15f;

    std::mt19937 rng(1);
    const std::filesystem::path fit_log = dir / "fit.log";
    {
      StdoutCapture capture(fit_log);
      llm::Trainer trainer(ctx, model, opt, cfg);
      trainer.fit(data, nullptr, rng);
    }
    const std::string log = read_text(fit_log);
    float first = 0.f;
    float last = 0.f;
    if (!loss_at(log, 1, first) || !loss_at(log, kMaxSteps, last)) {
      std::fprintf(stderr, "missing step loss\n");
      return 1;
    }
    print_log(log);
    std::printf("first_loss %.6f\nlast_loss %.6f\n", first, last);
    std::fflush(stdout);

    const std::filesystem::path tok_path = dir / "tokenizer.bpe";
    const std::filesystem::path model_path = dir / "model.bin";
    tok->save(tok_path);
    model.save(model_path, &opt);

    std::string reply;
    try {
      nn::Context infer;
      const std::unique_ptr<llm::Tokenizer> loaded_tok = llm::Tokenizer::load(tok_path);
      nn::Transformer::Loaded loaded =
          nn::Transformer::load(infer, model_path, nn::LoadMode::Inference);
      std::mt19937 reply_rng(1);
      reply = chat_reply(infer, loaded.model, *loaded_tok, reply_rng);
    } catch (const std::exception& error) {
      std::fprintf(stderr, "generate failed: %s\n", error.what());
      return 1;
    }
    std::printf("model> %s\n", reply.c_str());
    std::fflush(stdout);

    if (!std::isfinite(first) || !std::isfinite(last) || !(last < first)) {
      std::fprintf(stderr, "loss did not fall\n");
      return 1;
    }
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "train_bert_chat: %s\n", error.what());
    return 1;
  }
}
