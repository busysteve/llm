#include "llm/bpe.hpp"
#include "llm/generate.hpp"
#include "llm/presets.hpp"
#include "llm/rl.hpp"

#include "nn/loss.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace {

#define CHECK(cond)                                                                  \
  do {                                                                               \
    if (!(cond)) {                                                                   \
      std::fprintf(stderr, "CHECK failed: %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      std::exit(1);                                                                  \
    }                                                                                \
  } while (0)

template <typename F>
void check_invalid(F&& call, const char* message) {
  try {
    call();
  } catch (const std::invalid_argument& error) {
    CHECK(std::strcmp(error.what(), message) == 0);
    return;
  }
  CHECK(false);
}

bool near(float left, float right) {
  return std::isfinite(left) && std::isfinite(right) && std::fabs(left - right) <= 1e-4f;
}

bool same_bits(const std::vector<float>& left, const std::vector<float>& right) {
  if (left.size() != right.size()) return false;
  return std::memcmp(left.data(), right.data(), left.size() * sizeof(float)) == 0;
}

nn::TransformerConfig tiny_config(int vocab) {
  nn::TransformerConfig config;
  config.vocab_size = vocab;
  config.d_model = 16;
  config.num_heads = 2;
  config.num_layers = 1;
  config.d_ff = 32;
  config.max_seq_len = 32;
  config.dropout = 0.f;
  config.tie_embeddings = false;
  config.bias = false;
  config.activation = nn::TransformerActivation::SwiGLU;
  config.positional = nn::PositionalEncoding::LearnedAbsolute;
  return config;
}

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

void check_policy_gradient(nn::Context& ctx) {
  constexpr int kRows = 2;
  constexpr int kSeq = 3;
  constexpr int kVocab = 4;
  std::vector<float> logits(static_cast<size_t>(kRows * kSeq * kVocab));
  for (size_t i = 0; i < logits.size(); ++i) {
    logits[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.5f;
  }
  // The second row is ignore, so its gradient stays zero.
  const std::vector<int64_t> labels = {1, -100, 2, -100, -100, -100};
  const std::vector<float> unit = {1.f, 1.f};
  const llm::PolicyGrad got =
      llm::policy_gradient(logits.data(), nullptr, labels.data(), unit.data(), kRows, kSeq, kVocab, 0.f);

  nn::Tensor device(ctx, {kRows, kSeq, kVocab});
  device.copy_from_host(logits);
  nn::IndexTensor label_tensor(ctx, {kRows, kSeq});
  label_tensor.copy_from_host(labels.data(), label_tensor.numel());
  ctx.synchronize();
  const nn::LossResult ce = nn::sequence_cross_entropy(ctx, device, label_tensor, -100);
  const std::vector<float> ce_grad = ce.grad_logits.to_host();
  CHECK(ce_grad.size() == got.grad_logits.size());
  CHECK(near(ce.loss, got.loss));
  for (size_t i = 0; i < ce_grad.size(); ++i) CHECK(near(ce_grad[i], got.grad_logits[i]));

  const std::vector<float> twice = {2.f, 2.f};
  const llm::PolicyGrad scaled = llm::policy_gradient(logits.data(), nullptr, labels.data(), twice.data(),
                                                      kRows, kSeq, kVocab, 0.f);
  CHECK(scaled.grad_logits.size() == got.grad_logits.size());
  for (size_t i = 0; i < got.grad_logits.size(); ++i) {
    CHECK(near(scaled.grad_logits[i], 2.f * got.grad_logits[i]));
  }

  const std::vector<float> junk(logits.size(), 4.f);
  const llm::PolicyGrad ignored = llm::policy_gradient(logits.data(), junk.data(), labels.data(), unit.data(),
                                                       kRows, kSeq, kVocab, 0.f);
  CHECK(ignored.grad_logits.size() == got.grad_logits.size());
  CHECK(std::memcmp(ignored.grad_logits.data(), got.grad_logits.data(),
                    got.grad_logits.size() * sizeof(float)) == 0);
  CHECK(ignored.kl == 0.f);

  const std::vector<float> ref = {0.5f, 0.1f, -0.3f};
  const std::vector<int64_t> one_label = {1};
  const std::vector<float> zero = {0.f};
  const llm::PolicyGrad kl = llm::policy_gradient(logits.data(), ref.data(), one_label.data(), zero.data(), 1,
                                                  1, 3, 1.f);
  CHECK(kl.grad_logits.size() == 3);
  constexpr float kEps = 1e-3f;
  for (int v = 0; v < 3; ++v) {
    std::vector<float> plus = logits;
    std::vector<float> minus = logits;
    plus[static_cast<size_t>(v)] += kEps;
    minus[static_cast<size_t>(v)] -= kEps;
    // Only the first three logits are the row under test.
    plus.resize(3);
    minus.resize(3);
    const float high =
        llm::policy_gradient(plus.data(), ref.data(), one_label.data(), zero.data(), 1, 1, 3, 1.f).loss;
    const float low =
        llm::policy_gradient(minus.data(), ref.data(), one_label.data(), zero.data(), 1, 1, 3, 1.f).loss;
    const float numeric = (high - low) / (2.f * kEps);
    CHECK(std::fabs(numeric - kl.grad_logits[static_cast<size_t>(v)]) < 1e-2f);
  }
}

void check_constructor(nn::Context& ctx) {
  const nn::TransformerConfig causal = tiny_config(32);
  nn::Transformer model(ctx, causal, 1);
  nn::AdamW opt(llm::adamw_groups(model, 0.f), 1e-2f, 0.9f, 0.95f, 1e-8f);
  llm::RlConfig cfg;
  llm::RlTrainer ok(ctx, model, opt, cfg, nullptr);

  llm::RlConfig missing = cfg;
  missing.kl_coef = 0.1f;
  check_invalid([&] { llm::RlTrainer trainer(ctx, model, opt, missing, nullptr); }, "reference");

  nn::Transformer same(ctx, causal, 2);
  check_invalid([&] { llm::RlTrainer trainer(ctx, model, opt, cfg, &same); }, "reference");
  llm::RlConfig with_kl = cfg;
  with_kl.kl_coef = 0.1f;
  llm::RlTrainer paired(ctx, model, opt, with_kl, &same);

  nn::TransformerConfig other = causal;
  other.vocab_size = 64;
  nn::Transformer mismatch(ctx, other, 3);
  check_invalid([&] { llm::RlTrainer trainer(ctx, model, opt, with_kl, &mismatch); }, "reference");

  nn::TransformerConfig bidirectional = causal;
  bidirectional.causal = false;
  nn::Transformer encoder(ctx, bidirectional, 4);
  nn::AdamW encoder_opt(llm::adamw_groups(encoder, 0.f), 1e-2f, 0.9f, 0.95f, 1e-8f);
  check_invalid([&] { llm::RlTrainer trainer(ctx, encoder, encoder_opt, cfg, nullptr); }, "causal");
}

void check_sample_token() {
  llm::GenerateConfig cfg;
  cfg.temperature = 0.f;
  std::mt19937 rng(1);
  const std::vector<float> logits = {1.f, 3.f, -1.f};
  CHECK(llm::sample_token(logits, cfg, rng) == 1);
  cfg.repetition_penalty = 4.f;
  const int32_t seen = 1;
  // 3 / 4 = 0.75, which loses to the untouched logit 1.
  CHECK(llm::sample_token(logits, cfg, rng, &seen, 1) == 0);
  cfg.repetition_penalty = 0.f;
  check_invalid([&] { llm::sample_token(logits, cfg, rng, &seen, 1); }, "repetition_penalty");
}

uint32_t read_u32(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  CHECK(static_cast<bool>(in));
  unsigned char bytes[8] = {};
  in.read(reinterpret_cast<char*>(bytes), 8);
  CHECK(in.gcount() == 8);
  auto load = [&](int offset) {
    return static_cast<uint32_t>(bytes[offset]) | (static_cast<uint32_t>(bytes[offset + 1]) << 8) |
           (static_cast<uint32_t>(bytes[offset + 2]) << 16) |
           (static_cast<uint32_t>(bytes[offset + 3]) << 24);
  };
  CHECK(load(0) == 0x314E4E43u);
  return load(4);
}

void check_fit(nn::Context& ctx) {
  const std::unique_ptr<llm::BpeTokenizer> tok = llm::BpeTokenizer::train({"hello"}, 259);
  nn::TransformerConfig config = tiny_config(tok->vocab_size());
  nn::Transformer model(ctx, config, 1);
  nn::AdamW opt(llm::adamw_groups(model, 0.f), 1e-1f, 0.9f, 0.95f, 1e-8f);
  const std::vector<float> before = model.parameters().front().value->to_host();

  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "llm-rl-test";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const std::filesystem::path ckpt = dir / "model.bin";
  const std::filesystem::path log_path = dir / "fit.log";

  llm::RlConfig cfg;
  cfg.learning_rate = 1e-1f;
  cfg.min_lr = 1e-1f;
  cfg.group_size = 1;
  cfg.max_new_tokens = 2;
  cfg.temperature = 0.f;
  cfg.max_steps = 1;
  cfg.log_every = 1;
  cfg.ckpt_every = 1;
  cfg.checkpoint_path = ckpt;
  cfg.grad_clip_norm = 1.f;

  llm::RlTrainer trainer(ctx, model, opt, cfg, nullptr);
  std::mt19937 rng(1);
  {
    StdoutCapture capture(log_path);
    CHECK(trainer.fit(*tok, {"hello"}, [](const llm::Episode&) { return 1.f; }, rng) == 1);
  }
  const std::string log = read_text(log_path);
  const size_t at = log.find("step 1 loss ");
  CHECK(at != std::string::npos);
  float loss = 0.f;
  CHECK(std::sscanf(log.c_str() + at, "step 1 loss %f", &loss) == 1);
  CHECK(std::isfinite(loss));
  CHECK(!same_bits(before, model.parameters().front().value->to_host()));
  CHECK(read_u32(ckpt) == 2u);

  nn::Context loaded_ctx;
  nn::Transformer::Loaded loaded = nn::Transformer::load(loaded_ctx, ckpt, nn::LoadMode::Resume);
  CHECK(loaded.adamw.has_value());
  CHECK(loaded.adamw->step_count() == 1);

  nn::Transformer::Loaded frozen = nn::Transformer::load(ctx, ckpt, nn::LoadMode::Inference);
  llm::RlConfig with_kl = cfg;
  with_kl.kl_coef = 0.1f;
  with_kl.ckpt_every = 0;
  with_kl.checkpoint_path.clear();
  with_kl.log_every = 0;
  llm::RlTrainer kl_trainer(ctx, model, opt, with_kl, &frozen.model);
  CHECK(kl_trainer.fit(*tok, {"hello"}, [](const llm::Episode&) { return 1.f; }, rng) == 1);
  std::filesystem::remove_all(dir);
}

}  // namespace

void run_rl_tests() {
  check_sample_token();
  nn::Context ctx;
  check_policy_gradient(ctx);
  check_constructor(ctx);
  check_fit(ctx);
}
