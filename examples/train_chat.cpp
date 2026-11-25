#include "llm/bpe.hpp"
#include "llm/data.hpp"
#include "llm/generate.hpp"
#include "llm/presets.hpp"
#include "llm/train.hpp"

#include "nn/index_tensor.hpp"

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
constexpr int kMaskAt = 2;
constexpr char kQuestion[] = "What is the harbor bell for?";
constexpr char kMaskSentence[] = "The harbor bell rings at noon.";

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

llm::TrainConfig make_config() {
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
  return cfg;
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

std::string fit_log(nn::Context& ctx, nn::Transformer& model, llm::TextDataset& data,
                    const llm::TrainConfig& cfg, std::mt19937& rng,
                    const std::filesystem::path& path) {
  nn::AdamW opt(llm::adamw_groups(model, cfg.weight_decay), cfg.learning_rate, cfg.beta1, cfg.beta2,
                1e-8f);
  {
    StdoutCapture capture(path);
    llm::Trainer trainer(ctx, model, opt, cfg);
    trainer.fit(data, nullptr, rng);
  }
  return read_text(path);
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

// Token index 2 of kMaskSentence becomes the pad id. forward uses segment 0.
void print_mask(nn::Context& ctx, nn::Transformer& model, const llm::Tokenizer& tok) {
  model.eval();
  const std::vector<int32_t> encoded = tok.encode(kMaskSentence);
  if (static_cast<int>(encoded.size()) <= kMaskAt) throw std::runtime_error("mask sentence");
  const int32_t original = encoded[static_cast<size_t>(kMaskAt)];
  std::vector<int64_t> host(encoded.size());
  for (size_t i = 0; i < encoded.size(); ++i) host[i] = encoded[i];
  host[static_cast<size_t>(kMaskAt)] = tok.pad_id();

  nn::IndexTensor tokens(ctx, {1, static_cast<int>(host.size())});
  tokens.copy_from_host(host.data(), static_cast<int64_t>(host.size()));
  ctx.synchronize();
  const std::vector<float> logits = model.forward(tokens).to_host();
  const int vocab = tok.vocab_size();
  if (logits.size() != host.size() * static_cast<size_t>(vocab)) throw std::runtime_error("logits");
  const float* row = logits.data() + static_cast<size_t>(kMaskAt) * static_cast<size_t>(vocab);
  int best = 0;
  for (int i = 0; i < vocab; ++i) {
    if (!std::isfinite(row[i])) throw std::runtime_error("logits");
    if (row[i] > row[best]) best = i;
  }
  const std::string original_piece = tok.decode({original});
  const std::string guess = tok.decode({best});
  std::printf("bert_mask_at %d\nbert_original %s\nbert_guess_id %d\nbert_guess %s\n", kMaskAt,
              original_piece.c_str(), best, guess.c_str());
  std::fflush(stdout);
}

bool loss_fell(float first, float last) {
  return std::isfinite(first) && std::isfinite(last) && last < first;
}

}  // namespace

int main() {
  try {
    const std::vector<std::string> exchanges(std::begin(kExchanges), std::end(kExchanges));
    const std::unique_ptr<llm::BpeTokenizer> tok = llm::BpeTokenizer::train(exchanges, kVocabSize);
    const std::vector<int32_t> ids = encode_corpus(*tok);
    llm::TextDataset data(ids, static_cast<int32_t>(tok->vocab_size()), tok->eos_id());

    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "llm-train-chat";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    RemoveAll cleanup{dir};

    nn::Context ctx;
    // Sampling uses the causal tiny model. BERT keeps its own weights and is scored with forward.
    nn::Transformer chat_model(ctx, llm::presets::tiny(tok->vocab_size()), /*seed=*/1);
    nn::Transformer bert_model(ctx, llm::presets::bert_tiny(tok->vocab_size()), /*seed=*/1);
    std::mt19937 rng(1);

    const std::string chat_text = fit_log(ctx, chat_model, data, make_config(), rng, dir / "chat.log");
    float chat_first = 0.f;
    float chat_last = 0.f;
    if (!loss_at(chat_text, 1, chat_first) || !loss_at(chat_text, kMaxSteps, chat_last)) {
      std::fprintf(stderr, "missing chat step loss\n");
      return 1;
    }
    std::fputs("chat\n", stdout);
    print_log(chat_text);
    std::printf("chat_first_loss %.6f\nchat_last_loss %.6f\n", chat_first, chat_last);
    std::fflush(stdout);

    std::string reply;
    try {
      reply = chat_reply(ctx, chat_model, *tok, rng);
    } catch (const std::exception& error) {
      std::fprintf(stderr, "generate failed: %s\n", error.what());
      return 1;
    }
    std::printf("chat> %s\n", reply.c_str());
    std::fflush(stdout);

    llm::TrainConfig bert_cfg = make_config();
    bert_cfg.masked_lm = true;
    bert_cfg.mask_token = tok->pad_id();
    bert_cfg.mask_prob = 0.15f;
    const std::string bert_text = fit_log(ctx, bert_model, data, bert_cfg, rng, dir / "bert.log");
    float bert_first = 0.f;
    float bert_last = 0.f;
    if (!loss_at(bert_text, 1, bert_first) || !loss_at(bert_text, kMaxSteps, bert_last)) {
      std::fprintf(stderr, "missing bert step loss\n");
      return 1;
    }
    std::fputs("bert\n", stdout);
    print_log(bert_text);
    std::printf("bert_first_loss %.6f\nbert_last_loss %.6f\n", bert_first, bert_last);
    std::fflush(stdout);
    print_mask(ctx, bert_model, *tok);

    int failed = 0;
    if (!loss_fell(chat_first, chat_last)) {
      std::fprintf(stderr, "chat loss did not fall\n");
      failed = 1;
    }
    if (!loss_fell(bert_first, bert_last)) {
      std::fprintf(stderr, "bert loss did not fall\n");
      failed = 1;
    }
    return failed;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "train_chat: %s\n", error.what());
    return 1;
  }
}
