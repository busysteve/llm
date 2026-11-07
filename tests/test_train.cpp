#include "llm/bpe.hpp"
#include "llm/data.hpp"
#include "llm/generate.hpp"
#include "llm/presets.hpp"
#include "llm/train.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

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

bool same_bits(const std::vector<float>& left, const std::vector<float>& right) {
  if (left.size() != right.size()) return false;
  return std::memcmp(left.data(), right.data(), left.size() * sizeof(float)) == 0;
}

int64_t param_numel(const nn::Transformer& model) {
  int64_t n = 0;
  for (const nn::NamedParam& param : model.parameters()) n += param.value->numel();
  return n;
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

void check_groups(const nn::Transformer& model) {
  const std::vector<nn::NamedParam> params = model.parameters();
  const std::vector<nn::AdamWGroup> groups = llm::adamw_groups(model, 0.1f);
  std::vector<nn::NamedParam> flat;
  for (const nn::AdamWGroup& group : groups) {
    flat.insert(flat.end(), group.params.begin(), group.params.end());
  }
  CHECK(flat.size() == params.size());
  bool saw_tok = false;
  bool saw_bias = false;
  bool saw_final = false;
  for (size_t i = 0; i < params.size(); ++i) {
    CHECK(flat[i].value == params[i].value);
    CHECK(flat[i].name == params[i].name);
  }
  for (const nn::AdamWGroup& group : groups) {
    for (const nn::NamedParam& param : group.params) {
      if (param.name == "tok_emb.weight") {
        CHECK(group.weight_decay == 0.1f);
        saw_tok = true;
      }
      if (param.name.size() >= 5 && param.name.compare(param.name.size() - 5, 5, ".bias") == 0) {
        CHECK(group.weight_decay == 0.f);
        saw_bias = true;
      }
      if (param.name == "final_ln.weight") {
        CHECK(group.weight_decay == 0.f);
        saw_final = true;
      }
    }
  }
  CHECK(saw_tok);
  CHECK(saw_bias);
  CHECK(saw_final);
  check_invalid([&] { llm::adamw_groups(model, -0.1f); }, "weight_decay");
  check_invalid([&] { llm::adamw_groups(model, std::numeric_limits<float>::quiet_NaN()); },
                "weight_decay");
}

void check_schedule() {
  llm::TrainConfig cfg;
  cfg.learning_rate = 3e-4f;
  cfg.min_lr = 3e-5f;
  cfg.warmup_steps = 10;
  cfg.max_steps = 100;
  CHECK(llm::learning_rate_at(cfg, 1) == cfg.learning_rate / 10.f);
  CHECK(llm::learning_rate_at(cfg, 10) == cfg.learning_rate);
  const float mid = llm::learning_rate_at(cfg, 50);
  CHECK(mid < cfg.learning_rate);
  CHECK(mid >= cfg.min_lr);
  CHECK(llm::learning_rate_at(cfg, cfg.max_steps) == cfg.min_lr);
}

void check_fit(nn::Context& ctx) {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "llm-train-test";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const std::filesystem::path bin = dir / "tokens.bin";
  const std::filesystem::path meta = dir / "tokens.meta";
  const std::filesystem::path ckpt = dir / "model.bin";

  constexpr int32_t kVocab = 32;
  std::vector<int32_t> ids(64);
  for (size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<int32_t>(i % static_cast<size_t>(kVocab));
  llm::write_token_file(bin, meta, ids, kVocab);
  llm::TextDataset data(bin, meta);

  nn::Transformer model(ctx, tiny_config(kVocab), 1);
  check_groups(model);
  std::vector<nn::AdamWGroup> groups = llm::adamw_groups(model, 0.1f);
  nn::AdamW opt(std::move(groups), 1e-2f, 0.9f, 0.95f, 1e-8f);

  llm::TrainConfig cfg;
  cfg.learning_rate = 1e-2f;
  cfg.weight_decay = 0.1f;
  cfg.beta1 = 0.9f;
  cfg.beta2 = 0.95f;
  cfg.batch_size = 2;
  cfg.seq_len = 4;
  cfg.max_steps = 2;
  cfg.grad_accum_steps = 2;
  cfg.warmup_steps = 1;
  cfg.eval_every = 1;
  cfg.eval_batches = 1;
  cfg.ckpt_every = 1;
  cfg.checkpoint_path = ckpt;

  llm::TrainConfig bad = cfg;
  bad.batch_size = 0;
  check_invalid([&] { llm::Trainer trainer(ctx, model, opt, bad); }, "batch_size");
  bad = cfg;
  bad.seq_len = 0;
  check_invalid([&] { llm::Trainer trainer(ctx, model, opt, bad); }, "seq_len");
  bad = cfg;
  bad.max_steps = 0;
  check_invalid([&] { llm::Trainer trainer(ctx, model, opt, bad); }, "max_steps");
  bad = cfg;
  bad.grad_accum_steps = 0;
  check_invalid([&] { llm::Trainer trainer(ctx, model, opt, bad); }, "grad_accum_steps");
  bad = cfg;
  bad.warmup_steps = -1;
  check_invalid([&] { llm::Trainer trainer(ctx, model, opt, bad); }, "warmup_steps");
  bad = cfg;
  bad.learning_rate = std::numeric_limits<float>::quiet_NaN();
  check_invalid([&] { llm::Trainer trainer(ctx, model, opt, bad); }, "learning_rate");
  bad = cfg;
  bad.min_lr = -1.f;
  check_invalid([&] { llm::Trainer trainer(ctx, model, opt, bad); }, "min_lr");
  bad = cfg;
  bad.grad_clip_norm = -1.f;
  check_invalid([&] { llm::Trainer trainer(ctx, model, opt, bad); }, "grad_clip_norm");

  const std::vector<float> before = model.parameters().front().value->to_host();
  llm::Trainer trainer(ctx, model, opt, cfg);
  std::mt19937 rng(1);
  trainer.fit(data, &data, rng);
  const std::vector<nn::NamedParam> trained = model.parameters();
  CHECK(!same_bits(before, trained.front().value->to_host()));

  nn::Context loaded_ctx;
  nn::Transformer::Loaded loaded = nn::Transformer::load(loaded_ctx, ckpt, nn::LoadMode::Resume);
  CHECK(loaded.adamw.has_value());
  CHECK(!loaded.optimizer.has_value());
  CHECK(loaded.adamw->step_count() == 2);
  CHECK(loaded.step == 2);
  const std::vector<nn::NamedParam> restored = loaded.model.parameters();
  CHECK(restored.size() == trained.size());
  CHECK(restored.front().name == "tok_emb.weight");
  CHECK(same_bits(trained.front().value->to_host(), restored.front().value->to_host()));

  std::filesystem::remove_all(dir);
}

void check_generate(nn::Context& ctx) {
  const auto tok = llm::BpeTokenizer::train({"ab"}, 259);
  nn::Transformer model(ctx, tiny_config(tok->vocab_size()), 2);
  llm::Generator generator(ctx, model, *tok);

  llm::GenerateConfig cfg;
  cfg.max_new_tokens = 3;
  cfg.temperature = 0.f;
  cfg.top_k = 1;
  std::mt19937 first(7);
  std::mt19937 second(7);
  const std::string a = generator.generate("ab", cfg, first);
  const std::string b = generator.generate("ab", cfg, second);
  CHECK(a == b);

  check_invalid([&] { generator.generate("", cfg, first); }, "prompt");
  llm::GenerateConfig bad = cfg;
  bad.max_new_tokens = 0;
  check_invalid([&] { generator.generate("ab", bad, first); }, "max_new_tokens");
  bad = cfg;
  bad.temperature = std::numeric_limits<float>::quiet_NaN();
  check_invalid([&] { generator.generate("ab", bad, first); }, "temperature");
  bad = cfg;
  bad.top_k = -1;
  check_invalid([&] { generator.generate("ab", bad, first); }, "top_k");
  bad = cfg;
  bad.top_p = 0.f;
  check_invalid([&] { generator.generate("ab", bad, first); }, "top_p");
}

void check_presets() {
  check_invalid([] { llm::presets::tiny(0); }, "vocab_size");
  check_invalid([] { llm::presets::base(-3); }, "vocab_size");
  {
    nn::Context ctx;
    nn::Transformer model(ctx, llm::presets::tiny(16000), 1);
    CHECK(param_numel(model) == 11866624);
  }
  {
    nn::Context ctx;
    nn::Transformer model(ctx, llm::presets::small(16000), 1);
    CHECK(param_numel(model) == 37368832);
  }
  {
    nn::Context ctx;
    nn::Transformer model(ctx, llm::presets::base(16000), 1);
    CHECK(param_numel(model) == 110335488);
  }
}

}  // namespace

void run_train_tests() {
  check_schedule();
  {
    nn::Context ctx;
    check_fit(ctx);
    check_generate(ctx);
  }
  check_presets();
}
