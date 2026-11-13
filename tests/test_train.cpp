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

std::vector<std::vector<float>> take_params(const nn::Transformer& model) {
  std::vector<std::vector<float>> out;
  const std::vector<nn::NamedParam> params = model.parameters();
  out.reserve(params.size());
  for (const nn::NamedParam& param : params) out.push_back(param.value->to_host());
  return out;
}

std::vector<std::vector<float>> take_tensors(const std::vector<nn::Tensor>& tensors) {
  std::vector<std::vector<float>> out;
  out.reserve(tensors.size());
  for (const nn::Tensor& tensor : tensors) out.push_back(tensor.to_host());
  return out;
}

void check_same_params(const std::vector<std::vector<float>>& want, const nn::Transformer& model) {
  const std::vector<nn::NamedParam> params = model.parameters();
  CHECK(params.size() == want.size());
  for (size_t i = 0; i < want.size(); ++i) CHECK(same_bits(want[i], params[i].value->to_host()));
}

void check_same_tensors(const std::vector<std::vector<float>>& want,
                        const std::vector<nn::Tensor>& got) {
  CHECK(got.size() == want.size());
  for (size_t i = 0; i < want.size(); ++i) CHECK(same_bits(want[i], got[i].to_host()));
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
  bad = cfg;
  bad.early_stop_patience = -1;
  check_invalid([&] { llm::Trainer trainer(ctx, model, opt, bad); }, "early_stop_patience");
  bad = cfg;
  bad.early_stop_min_delta = -1.f;
  check_invalid([&] { llm::Trainer trainer(ctx, model, opt, bad); }, "early_stop_min_delta");
  bad = cfg;
  bad.early_stop_min_delta = std::numeric_limits<float>::infinity();
  check_invalid([&] { llm::Trainer trainer(ctx, model, opt, bad); }, "early_stop_min_delta");
  bad = cfg;
  bad.min_loss = -1.f;
  check_invalid([&] { llm::Trainer trainer(ctx, model, opt, bad); }, "min_loss");
  bad = cfg;
  bad.min_loss = std::numeric_limits<float>::infinity();
  check_invalid([&] { llm::Trainer trainer(ctx, model, opt, bad); }, "min_loss");

  const std::vector<float> before = model.parameters().front().value->to_host();
  llm::Trainer trainer(ctx, model, opt, cfg);
  std::mt19937 rng(1);
  CHECK(trainer.fit(data, &data, rng) == 2);
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

void check_early_stop(nn::Context& ctx) {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "llm-early-stop-test";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const std::filesystem::path bin = dir / "tokens.bin";
  const std::filesystem::path meta = dir / "tokens.meta";
  const std::filesystem::path ckpt = dir / "model.bin";

  constexpr int32_t kVocab = 32;
  std::vector<int32_t> ids(64);
  for (size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<int32_t>(i % static_cast<size_t>(kVocab));
  llm::write_token_file(bin, meta, ids, kVocab);
  llm::TextDataset train(bin, meta);
  llm::TextDataset valid(bin, meta);

  // min_lr == learning_rate keeps step 1 identical across different max_steps.
  llm::TrainConfig base;
  base.learning_rate = 1e-2f;
  base.min_lr = 1e-2f;
  base.weight_decay = 0.1f;
  base.batch_size = 2;
  base.seq_len = 4;
  base.grad_accum_steps = 1;
  base.warmup_steps = 0;
  base.log_every = 0;
  base.eval_every = 1;
  base.eval_batches = 1;
  base.grad_clip_norm = 1.f;

  {
    nn::Transformer model(ctx, tiny_config(kVocab), 1);
    nn::AdamW opt(llm::adamw_groups(model, 0.1f), base.learning_rate, 0.9f, 0.95f, 1e-8f);
    llm::TrainConfig bad = base;
    bad.max_steps = 4;
    bad.early_stop_patience = 1;
    bad.eval_every = 0;
    llm::Trainer trainer(ctx, model, opt, bad);
    std::mt19937 rng(1);
    check_invalid([&] { trainer.fit(train, nullptr, rng); }, "early_stop_patience");
    bad.eval_every = 1;
    bad.eval_batches = 0;
    llm::Trainer trainer_batches(ctx, model, opt, bad);
    check_invalid([&] { trainer_batches.fit(train, &valid, rng); }, "early_stop_patience");
    bad.eval_batches = 1;
    bad.eval_every = 8;
    llm::Trainer trainer_gap(ctx, model, opt, bad);
    check_invalid([&] { trainer_gap.fit(train, &valid, rng); }, "early_stop_patience");
  }

  std::vector<std::vector<float>> want_params;
  std::vector<std::vector<float>> want_m;
  std::vector<std::vector<float>> want_v;
  {
    nn::Transformer model(ctx, tiny_config(kVocab), 1);
    nn::AdamW opt(llm::adamw_groups(model, 0.1f), base.learning_rate, 0.9f, 0.95f, 1e-8f);
    llm::TrainConfig cfg = base;
    cfg.max_steps = 1;
    cfg.ckpt_every = 0;
    llm::Trainer trainer(ctx, model, opt, cfg);
    std::mt19937 rng(1);
    CHECK(trainer.fit(train, &valid, rng) == 1);
    CHECK(opt.step_count() == 1);
    want_params = take_params(model);
    want_m = take_tensors(opt.m());
    want_v = take_tensors(opt.v());
  }

  {
    nn::Transformer model(ctx, tiny_config(kVocab), 1);
    nn::AdamW opt(llm::adamw_groups(model, 0.1f), base.learning_rate, 0.9f, 0.95f, 1e-8f);
    llm::TrainConfig cfg = base;
    cfg.max_steps = 4;
    cfg.early_stop_patience = 1;
    cfg.early_stop_min_delta = 1e6f;
    cfg.ckpt_every = 1;
    cfg.checkpoint_path = ckpt;
    llm::Trainer trainer(ctx, model, opt, cfg);
    std::mt19937 rng(1);
    // Step 1 is the baseline. Step 2 cannot beat it by 1e6, so patience trips.
    CHECK(trainer.fit(train, &valid, rng) == 2);
    CHECK(opt.step_count() == 1);
    check_same_params(want_params, model);
    check_same_tensors(want_m, opt.m());
    check_same_tensors(want_v, opt.v());

    nn::Context loaded_ctx;
    nn::Transformer::Loaded loaded = nn::Transformer::load(loaded_ctx, ckpt, nn::LoadMode::Resume);
    CHECK(loaded.adamw.has_value());
    CHECK(loaded.adamw->step_count() == 1);
    CHECK(loaded.step == 1);
    check_same_params(want_params, loaded.model);
    check_same_tensors(want_m, loaded.adamw->m());
    check_same_tensors(want_v, loaded.adamw->v());
  }

  {
    nn::Transformer model(ctx, tiny_config(kVocab), 1);
    nn::AdamW opt(llm::adamw_groups(model, 0.1f), base.learning_rate, 0.9f, 0.95f, 1e-8f);
    llm::TrainConfig cfg = base;
    cfg.max_steps = 5;
    cfg.early_stop_patience = 2;
    cfg.early_stop_min_delta = 1e6f;
    cfg.ckpt_every = 0;
    llm::Trainer trainer(ctx, model, opt, cfg);
    std::mt19937 rng(1);
    // One missed eval is not enough. The second missed eval stops on step 3.
    CHECK(trainer.fit(train, &valid, rng) == 3);
    CHECK(opt.step_count() == 1);
    check_same_params(want_params, model);
    check_same_tensors(want_m, opt.m());
    check_same_tensors(want_v, opt.v());
  }

  std::filesystem::remove_all(dir);
}

void check_min_loss(nn::Context& ctx) {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "llm-min-loss-test";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const std::filesystem::path bin = dir / "tokens.bin";
  const std::filesystem::path meta = dir / "tokens.meta";
  const std::filesystem::path ckpt = dir / "model.bin";

  constexpr int32_t kVocab = 32;
  std::vector<int32_t> ids(64);
  for (size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<int32_t>(i % static_cast<size_t>(kVocab));
  llm::write_token_file(bin, meta, ids, kVocab);

  // min_lr == learning_rate keeps step 1 identical across different max_steps.
  llm::TrainConfig base;
  base.learning_rate = 1e-2f;
  base.min_lr = 1e-2f;
  base.weight_decay = 0.1f;
  base.batch_size = 2;
  base.seq_len = 4;
  base.grad_accum_steps = 1;
  base.warmup_steps = 0;
  base.log_every = 0;
  base.eval_every = 0;
  base.grad_clip_norm = 1.f;

  std::vector<std::vector<float>> want_params;
  std::vector<std::vector<float>> want_m;
  std::vector<std::vector<float>> want_v;
  {
    llm::TextDataset train(bin, meta);
    nn::Transformer model(ctx, tiny_config(kVocab), 1);
    nn::AdamW opt(llm::adamw_groups(model, 0.1f), base.learning_rate, 0.9f, 0.95f, 1e-8f);
    llm::TrainConfig cfg = base;
    cfg.max_steps = 1;
    llm::Trainer trainer(ctx, model, opt, cfg);
    std::mt19937 rng(1);
    CHECK(trainer.fit(train, nullptr, rng) == 1);
    CHECK(opt.step_count() == 1);
    want_params = take_params(model);
    want_m = take_tensors(opt.m());
    want_v = take_tensors(opt.v());
  }

  {
    llm::TextDataset train(bin, meta);
    llm::TextDataset valid(bin, meta);
    nn::Transformer model(ctx, tiny_config(kVocab), 1);
    nn::AdamW opt(llm::adamw_groups(model, 0.1f), base.learning_rate, 0.9f, 0.95f, 1e-8f);
    llm::TrainConfig cfg = base;
    cfg.max_steps = 4;
    cfg.min_loss = 1e6f;
    cfg.early_stop_patience = 1;
    cfg.early_stop_min_delta = 1e6f;
    cfg.eval_every = 1;
    cfg.eval_batches = 1;
    cfg.ckpt_every = 0;
    cfg.checkpoint_path = ckpt;
    llm::Trainer trainer(ctx, model, opt, cfg);
    std::mt19937 rng(1);
    // Any real loss is below 1e6, so the threshold stops on step 1.
    // Patience would have continued to step 2 and restored that baseline.
    CHECK(trainer.fit(train, &valid, rng) == 1);
    CHECK(opt.step_count() == 1);
    check_same_params(want_params, model);
    check_same_tensors(want_m, opt.m());
    check_same_tensors(want_v, opt.v());

    nn::Context loaded_ctx;
    nn::Transformer::Loaded loaded = nn::Transformer::load(loaded_ctx, ckpt, nn::LoadMode::Resume);
    CHECK(loaded.adamw.has_value());
    CHECK(loaded.adamw->step_count() == 1);
    CHECK(loaded.step == 1);
    check_same_params(want_params, loaded.model);
    check_same_tensors(want_m, loaded.adamw->m());
    check_same_tensors(want_v, loaded.adamw->v());
  }

  {
    llm::TextDataset train(bin, meta);
    nn::Transformer model(ctx, tiny_config(kVocab), 1);
    nn::AdamW opt(llm::adamw_groups(model, 0.1f), base.learning_rate, 0.9f, 0.95f, 1e-8f);
    llm::TrainConfig cfg = base;
    cfg.max_steps = 2;
    cfg.min_loss = 0.f;
    llm::Trainer trainer(ctx, model, opt, cfg);
    std::mt19937 rng(1);
    // Cross-entropy of an untrained step stays above 0, so the run finishes.
    CHECK(trainer.fit(train, nullptr, rng) == 2);
    CHECK(opt.step_count() == 2);
  }

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
  check_invalid([] { llm::presets::llama_tiny(0); }, "vocab_size");
  const nn::TransformerConfig llama = llm::presets::llama_tiny(16000);
  CHECK(llama.positional == nn::PositionalEncoding::Rope);
  CHECK(llama.d_model == 256);
  CHECK(llama.num_layers == 4);
  CHECK(llama.num_heads == 4);
  CHECK(llama.d_ff == 768);
  CHECK(llama.bias == false);
  CHECK(llm::presets::llama_small(8).positional == nn::PositionalEncoding::Rope);
  CHECK(llm::presets::llama_small(8).num_layers == 6);
  CHECK(llm::presets::llama_base(8).d_model == 768);
  CHECK(llm::presets::llama_base(8).d_ff == 2048);
  {
    nn::Context ctx;
    nn::Transformer model(ctx, llama, 1);
    CHECK(param_numel(model) == 11602176);
    for (const nn::NamedParam& param : model.parameters()) {
      CHECK(param.name.find("pos_emb") == std::string::npos);
      CHECK(param.name.size() < 5 || param.name.compare(param.name.size() - 5, 5, ".bias") != 0);
    }
  }
  check_invalid([] { llm::presets::qwen_tiny(0); }, "vocab_size");
  const nn::TransformerConfig qwen = llm::presets::qwen_tiny(16000);
  CHECK(qwen.positional == nn::PositionalEncoding::Rope);
  CHECK(qwen.num_layers == 4);
  CHECK(qwen.d_model == 256);
  CHECK(qwen.num_heads == 4);
  CHECK(qwen.num_kv_heads == 2);
  CHECK(qwen.head_dim == 128);
  CHECK(qwen.qk_norm);
  CHECK(qwen.rope_theta == 1000000.f);
  CHECK(qwen.rms_eps == 1e-6f);
  CHECK(qwen.d_ff == 768);
  CHECK(qwen.bias == false);
  CHECK(llm::presets::qwen_small(8).num_layers == 6);
  CHECK(llm::presets::qwen_small(8).num_heads == 8);
  CHECK(llm::presets::qwen_small(8).num_kv_heads == 2);
  CHECK(llm::presets::qwen_small(8).head_dim == 128);
  CHECK(llm::presets::qwen_base(8).d_model == 768);
  CHECK(llm::presets::qwen_base(8).num_heads == 12);
  CHECK(llm::presets::qwen_base(8).num_kv_heads == 4);
  CHECK(llm::presets::qwen_base(8).d_ff == 2048);
  {
    nn::Context ctx;
    nn::Transformer model(ctx, qwen, 1);
    CHECK(param_numel(model) == 12127488);
    bool saw_q = false;
    bool saw_k = false;
    for (const nn::NamedParam& param : model.parameters()) {
      CHECK(param.name.find("pos_emb") == std::string::npos);
      if (param.name == "blocks.0.attn.q_norm.weight") {
        saw_q = true;
        CHECK(param.value->shape() == std::vector<int>{128});
      }
      if (param.name == "blocks.0.attn.k_norm.weight") {
        saw_k = true;
        CHECK(param.value->shape() == std::vector<int>{128});
      }
      if (param.name == "blocks.0.attn.q.weight") CHECK((param.value->shape() == std::vector<int>{512, 256}));
      if (param.name == "blocks.0.attn.k.weight") CHECK((param.value->shape() == std::vector<int>{256, 256}));
    }
    CHECK(saw_q);
    CHECK(saw_k);
  }
  {
    nn::Context ctx;
    nn::Transformer model(ctx, llm::presets::qwen_small(16000), 1);
    CHECK(param_numel(model) == 38412288);
  }
  {
    nn::Context ctx;
    nn::Transformer model(ctx, llm::presets::qwen_base(16000), 1);
    CHECK(param_numel(model) == 118970112);
  }
  {
    const auto tok = llm::BpeTokenizer::train({"ab"}, 259);
    nn::TransformerConfig config = llm::presets::qwen_tiny(tok->vocab_size());
    config.num_layers = 1;
    config.d_model = 32;
    config.num_heads = 4;
    config.num_kv_heads = 2;
    config.head_dim = 16;
    config.d_ff = 64;
    config.max_seq_len = 32;
    nn::Context ctx;
    nn::Transformer model(ctx, config, 3);
    llm::Generator generator(ctx, model, *tok);
    llm::GenerateConfig cfg;
    cfg.max_new_tokens = 2;
    cfg.temperature = 0.f;
    cfg.top_k = 1;
    std::mt19937 rng(1);
    const std::string text = generator.generate("ab", cfg, rng);
    const auto dir = std::filesystem::temp_directory_path() / "llm-qwen-preset";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const auto path = dir / "model.bin";
    model.save(path, nullptr);
    nn::Context loaded_ctx;
    nn::Transformer::Loaded loaded = nn::Transformer::load(loaded_ctx, path, nn::LoadMode::Inference);
    llm::Generator again(loaded_ctx, loaded.model, *tok);
    std::mt19937 rng2(1);
    CHECK(again.generate("ab", cfg, rng2) == text);
    std::filesystem::remove_all(dir);
  }
}

}  // namespace

void run_train_tests() {
  check_schedule();
  {
    nn::Context ctx;
    check_fit(ctx);
    check_early_stop(ctx);
    check_min_loss(ctx);
    check_generate(ctx);
  }
  check_presets();
}
