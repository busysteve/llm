#include "llm/bpe.hpp"
#include "llm/data.hpp"
#include "llm/generate.hpp"
#include "llm/presets.hpp"
#include "llm/train.hpp"

#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
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

  {
    nn::TransformerConfig config = tiny_config(tok->vocab_size());
    config.causal = false;
    config.type_vocab_size = 2;
    nn::Transformer bert(ctx, config, 2);
    llm::Generator bert_gen(ctx, bert, *tok);
    llm::GenerateConfig masked = cfg;
    masked.max_new_tokens = 2;
    std::mt19937 left_rng(7);
    std::mt19937 right_rng(7);
    const std::string left = bert_gen.generate("ab", masked, left_rng);
    const std::string right = bert_gen.generate("ab", masked, right_rng);
    CHECK(left == right);
    const auto dir = std::filesystem::temp_directory_path() / "llm-bert-generate";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const auto path = dir / "model.bin";
    bert.save(path, nullptr);
    nn::Context loaded_ctx;
    nn::Transformer::Loaded loaded = nn::Transformer::load(loaded_ctx, path, nn::LoadMode::Inference);
    llm::Generator again(loaded_ctx, loaded.model, *tok);
    std::mt19937 loaded_rng(7);
    CHECK(again.generate("ab", masked, loaded_rng) == left);
    std::filesystem::remove_all(dir);
  }
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
  CHECK(llm::presets::medium(8).num_layers == 24);
  CHECK(llm::presets::medium(8).d_model == 1024);
  CHECK(llm::presets::medium(8).num_heads == 16);
  CHECK(llm::presets::medium(8).d_ff == 2816);
  CHECK(llm::presets::large(8).num_layers == 36);
  CHECK(llm::presets::large(8).d_model == 1280);
  CHECK(llm::presets::large(8).num_heads == 20);
  CHECK(llm::presets::large(8).d_ff == 3584);
  {
    nn::Context ctx;
    nn::Transformer model(ctx, llm::presets::medium(16000), 1);
    CHECK(param_numel(model) == 342198272);
  }
  {
    nn::Context ctx;
    nn::Transformer model(ctx, llm::presets::large(16000), 1);
    CHECK(param_numel(model) == 773839360);
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
  CHECK(llm::presets::llama_medium(8).num_layers == 24);
  CHECK(llm::presets::llama_medium(8).d_model == 1024);
  CHECK(llm::presets::llama_medium(8).d_ff == 2816);
  CHECK(llm::presets::llama_medium(8).positional == nn::PositionalEncoding::Rope);
  CHECK(llm::presets::llama_large(8).num_layers == 36);
  CHECK(llm::presets::llama_large(8).d_model == 1280);
  CHECK(llm::presets::llama_large(8).d_ff == 3584);
  {
    nn::Context ctx;
    nn::Transformer model(ctx, llama, 1);
    CHECK(param_numel(model) == 11602176);
    for (const nn::NamedParam& param : model.parameters()) {
      CHECK(param.name.find("pos_emb") == std::string::npos);
      CHECK(param.name.size() < 5 || param.name.compare(param.name.size() - 5, 5, ".bias") != 0);
    }
  }
  {
    nn::Context ctx;
    nn::Transformer model(ctx, llm::presets::llama_medium(16000), 1);
    CHECK(param_numel(model) == 341099520);
  }
  {
    nn::Context ctx;
    nn::Transformer model(ctx, llm::presets::llama_large(16000), 1);
    CHECK(param_numel(model) == 772435200);
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
  CHECK(llm::presets::qwen_medium(8).num_layers == 24);
  CHECK(llm::presets::qwen_medium(8).d_model == 1024);
  CHECK(llm::presets::qwen_medium(8).num_heads == 16);
  CHECK(llm::presets::qwen_medium(8).num_kv_heads == 4);
  CHECK(llm::presets::qwen_medium(8).d_ff == 2816);
  CHECK(llm::presets::qwen_large(8).num_layers == 36);
  CHECK(llm::presets::qwen_large(8).d_model == 1280);
  CHECK(llm::presets::qwen_large(8).num_heads == 20);
  CHECK(llm::presets::qwen_large(8).num_kv_heads == 5);
  CHECK(llm::presets::qwen_large(8).d_ff == 3584);
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
    nn::Context ctx;
    nn::Transformer model(ctx, llm::presets::qwen_medium(16000), 1);
    CHECK(param_numel(model) == 366271488);
  }
  {
    nn::Context ctx;
    nn::Transformer model(ctx, llm::presets::qwen_large(16000), 1);
    CHECK(param_numel(model) == 831426816);
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
  check_invalid([] { llm::presets::gpt2_tiny(0); }, "vocab_size");
  check_invalid([] { llm::presets::gpt3_base(-1); }, "vocab_size");
  const nn::TransformerConfig gpt2 = llm::presets::gpt2_tiny(16000);
  CHECK(gpt2.activation == nn::TransformerActivation::Gelu);
  CHECK(gpt2.positional == nn::PositionalEncoding::LearnedAbsolute);
  CHECK(gpt2.bias);
  CHECK(gpt2.tie_embeddings == false);
  CHECK(gpt2.dropout == 0.f);
  CHECK(gpt2.max_seq_len == 1024);
  CHECK(gpt2.num_layers == 4);
  CHECK(gpt2.d_model == 256);
  CHECK(gpt2.num_heads == 4);
  CHECK(gpt2.d_ff == 1024);
  CHECK(llm::presets::gpt2_small(8).num_layers == 6);
  CHECK(llm::presets::gpt2_small(8).d_model == 512);
  CHECK(llm::presets::gpt2_small(8).num_heads == 8);
  CHECK(llm::presets::gpt2_small(8).d_ff == 2048);
  CHECK(llm::presets::gpt2_small(8).max_seq_len == 1024);
  CHECK(llm::presets::gpt2_base(8).num_layers == 12);
  CHECK(llm::presets::gpt2_base(8).d_model == 768);
  CHECK(llm::presets::gpt2_base(8).d_ff == 3072);
  CHECK(llm::presets::gpt2_medium(8).num_layers == 24);
  CHECK(llm::presets::gpt2_medium(8).d_model == 1024);
  CHECK(llm::presets::gpt2_medium(8).num_heads == 16);
  CHECK(llm::presets::gpt2_medium(8).d_ff == 4096);
  CHECK(llm::presets::gpt2_large(8).num_layers == 36);
  CHECK(llm::presets::gpt2_large(8).d_model == 1280);
  CHECK(llm::presets::gpt2_large(8).num_heads == 20);
  CHECK(llm::presets::gpt2_large(8).d_ff == 5120);
  const nn::TransformerConfig gpt3 = llm::presets::gpt3_tiny(16000);
  CHECK(gpt3.activation == nn::TransformerActivation::Gelu);
  CHECK(gpt3.positional == nn::PositionalEncoding::LearnedAbsolute);
  CHECK(gpt3.bias);
  CHECK(gpt3.max_seq_len == 2048);
  CHECK(gpt3.num_layers == 4);
  CHECK(gpt3.d_model == 256);
  CHECK(gpt3.d_ff == 1024);
  CHECK(llm::presets::gpt3_small(8).max_seq_len == 2048);
  CHECK(llm::presets::gpt3_small(8).d_model == 512);
  CHECK(llm::presets::gpt3_small(8).d_ff == 2048);
  CHECK(llm::presets::gpt3_base(8).num_layers == 12);
  CHECK(llm::presets::gpt3_base(8).d_model == 768);
  CHECK(llm::presets::gpt3_base(8).d_ff == 3072);
  CHECK(llm::presets::gpt3_medium(8).num_layers == 24);
  CHECK(llm::presets::gpt3_medium(8).d_model == 1024);
  CHECK(llm::presets::gpt3_medium(8).max_seq_len == 2048);
  CHECK(llm::presets::gpt3_medium(8).d_ff == 4096);
  CHECK(llm::presets::gpt3_large(8).num_layers == 36);
  CHECK(llm::presets::gpt3_large(8).d_model == 1280);
  CHECK(llm::presets::gpt3_large(8).d_ff == 5120);
  {
    nn::Context ctx;
    nn::Transformer model(ctx, gpt2, 1);
    CHECK(param_numel(model) == 11629696);
    bool saw_pos = false;
    bool saw_fc1_bias = false;
    for (const nn::NamedParam& param : model.parameters()) {
      if (param.name == "pos_emb.weight") {
        saw_pos = true;
        CHECK((param.value->shape() == std::vector<int>{1024, 256}));
      }
      if (param.name == "blocks.0.fc1.weight") {
        CHECK((param.value->shape() == std::vector<int>{1024, 256}));
      }
      if (param.name == "blocks.0.fc1.bias") {
        saw_fc1_bias = true;
        CHECK((param.value->shape() == std::vector<int>{1024}));
      }
    }
    CHECK(saw_pos);
    CHECK(saw_fc1_bias);
  }
  {
    nn::Context ctx;
    nn::Transformer model(ctx, llm::presets::gpt2_small(16000), 1);
    CHECK(param_numel(model) == 35839616);
  }
  {
    nn::Context ctx;
    nn::Transformer model(ctx, llm::presets::gpt2_base(16000), 1);
    CHECK(param_numel(model) == 110434432);
  }
  {
    nn::Context ctx;
    nn::Transformer model(ctx, llm::presets::gpt2_medium(16000), 1);
    CHECK(param_numel(model) == 336144000);
  }
  {
    nn::Context ctx;
    nn::Transformer model(ctx, llm::presets::gpt2_large(16000), 1);
    CHECK(param_numel(model) == 750677120);
  }
  {
    nn::Context ctx;
    nn::Transformer model(ctx, gpt3, 1);
    CHECK(param_numel(model) == 11891840);
    bool saw_pos = false;
    for (const nn::NamedParam& param : model.parameters()) {
      if (param.name == "pos_emb.weight") {
        saw_pos = true;
        CHECK((param.value->shape() == std::vector<int>{2048, 256}));
      }
    }
    CHECK(saw_pos);
  }
  {
    nn::Context ctx;
    nn::Transformer model(ctx, llm::presets::gpt3_small(16000), 1);
    CHECK(param_numel(model) == 36363904);
  }
  {
    nn::Context ctx;
    nn::Transformer model(ctx, llm::presets::gpt3_base(16000), 1);
    CHECK(param_numel(model) == 111220864);
  }
  {
    nn::Context ctx;
    nn::Transformer model(ctx, llm::presets::gpt3_medium(16000), 1);
    CHECK(param_numel(model) == 337192576);
  }
  {
    nn::Context ctx;
    nn::Transformer model(ctx, llm::presets::gpt3_large(16000), 1);
    CHECK(param_numel(model) == 751987840);
  }
  check_invalid([] { llm::presets::bert_tiny(0); }, "vocab_size");
  const nn::TransformerConfig bert = llm::presets::bert_tiny(16000);
  CHECK(bert.causal == false);
  CHECK(bert.type_vocab_size == 2);
  CHECK(bert.activation == nn::TransformerActivation::Gelu);
  CHECK(bert.positional == nn::PositionalEncoding::LearnedAbsolute);
  CHECK(bert.bias);
  CHECK(bert.max_seq_len == 512);
  CHECK(bert.num_layers == 4);
  CHECK(bert.d_model == 256);
  CHECK(bert.d_ff == 1024);
  CHECK(llm::presets::bert_small(8).num_layers == 6);
  CHECK(llm::presets::bert_small(8).d_model == 512);
  CHECK(llm::presets::bert_small(8).d_ff == 2048);
  CHECK(llm::presets::bert_small(8).max_seq_len == 512);
  CHECK(llm::presets::bert_base(8).num_layers == 12);
  CHECK(llm::presets::bert_base(8).d_model == 768);
  CHECK(llm::presets::bert_base(8).d_ff == 3072);
  CHECK(llm::presets::bert_medium(8).num_layers == 24);
  CHECK(llm::presets::bert_medium(8).d_model == 1024);
  CHECK(llm::presets::bert_medium(8).d_ff == 4096);
  CHECK(llm::presets::bert_medium(8).max_seq_len == 512);
  CHECK(llm::presets::bert_medium(8).causal == false);
  CHECK(llm::presets::bert_large(8).num_layers == 36);
  CHECK(llm::presets::bert_large(8).d_model == 1280);
  CHECK(llm::presets::bert_large(8).d_ff == 5120);
  {
    nn::Context ctx;
    nn::Transformer model(ctx, bert, 1);
    CHECK(param_numel(model) == 11499136);
    bool saw_type = false;
    for (const nn::NamedParam& param : model.parameters()) {
      if (param.name == "type_emb.weight") {
        saw_type = true;
        CHECK((param.value->shape() == std::vector<int>{2, 256}));
      }
      if (param.name == "pos_emb.weight") {
        CHECK((param.value->shape() == std::vector<int>{512, 256}));
      }
    }
    CHECK(saw_type);
  }
  {
    nn::Context ctx;
    nn::Transformer model(ctx, llm::presets::bert_small(16000), 1);
    CHECK(param_numel(model) == 35578496);
  }
  {
    nn::Context ctx;
    nn::Transformer model(ctx, llm::presets::bert_base(16000), 1);
    CHECK(param_numel(model) == 110042752);
  }
  {
    nn::Context ctx;
    nn::Transformer model(ctx, llm::presets::bert_medium(16000), 1);
    CHECK(param_numel(model) == 335621760);
  }
  {
    nn::Context ctx;
    nn::Transformer model(ctx, llm::presets::bert_large(16000), 1);
    CHECK(param_numel(model) == 750024320);
  }
}

void check_masked(nn::Context& ctx) {
  nn::TransformerConfig config;
  config.vocab_size = 32;
  config.d_model = 32;
  config.num_heads = 4;
  config.num_layers = 1;
  config.d_ff = 64;
  config.max_seq_len = 8;
  config.bias = true;
  config.activation = nn::TransformerActivation::Gelu;
  config.causal = false;
  config.type_vocab_size = 2;
  nn::Transformer model(ctx, config, 1);
  nn::AdamW opt(llm::adamw_groups(model, 0.f), 1e-3f, 0.9f, 0.95f, 1e-8f);
  std::vector<int32_t> ids(40);
  for (int i = 0; i < 40; ++i) ids[static_cast<size_t>(i)] = (i % 30) + 1;
  llm::TextDataset data(std::move(ids), 32);
  llm::TrainConfig cfg;
  cfg.learning_rate = 1e-3f;
  cfg.min_lr = 1e-4f;
  cfg.weight_decay = 0.f;
  cfg.batch_size = 2;
  cfg.seq_len = 8;
  cfg.max_steps = 1;
  cfg.warmup_steps = 0;
  cfg.log_every = 0;
  cfg.masked_lm = true;
  cfg.mask_prob = 1.f;
  cfg.mask_token = 0;
  std::mt19937 rng(1);
  llm::Trainer trainer(ctx, model, opt, cfg);
  CHECK(trainer.fit(data, nullptr, rng) == 1);
  const auto path = std::filesystem::temp_directory_path() / "llm-bert-mask.bin";
  model.save(path, nullptr);
  std::ifstream in(path, std::ios::binary);
  in.seekg(4);
  uint32_t version = 0;
  in.read(reinterpret_cast<char*>(&version), 4);
  CHECK(version == 5u);
  std::filesystem::remove(path);
  llm::TrainConfig bad = cfg;
  bad.mask_token = -1;
  check_invalid([&] { llm::Trainer discarded(ctx, model, opt, bad); }, "mask_token");
}

void check_resume(nn::Context& ctx) {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "llm-resume-test";
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
  CHECK(data.ids().size() == ids.size());
  CHECK(data.ids()[3] == ids[3]);

  auto make_cfg = [&](int start, int steps) {
    llm::TrainConfig cfg;
    cfg.learning_rate = 1e-2f;
    cfg.min_lr = 1e-3f;
    cfg.weight_decay = 0.1f;
    cfg.batch_size = 2;
    cfg.seq_len = 4;
    cfg.max_steps = steps;
    cfg.start_step = start;
    cfg.grad_accum_steps = 1;
    cfg.warmup_steps = 1;
    cfg.log_every = 0;
    cfg.eval_every = 0;
    cfg.ckpt_every = 0;
    cfg.checkpoint_path = ckpt;
    return cfg;
  };

  nn::Transformer model(ctx, tiny_config(kVocab), 1);
  nn::AdamW opt(llm::adamw_groups(model, 0.1f), 1e-2f, 0.9f, 0.95f, 1e-8f);
  llm::TrainConfig bad = make_cfg(0, 4);
  check_invalid([&] { llm::Trainer trainer(ctx, model, opt, bad); }, "start_step");
  bad = make_cfg(5, 4);
  check_invalid([&] { llm::Trainer trainer(ctx, model, opt, bad); }, "start_step");

  // The flag is already set, so the first update is saved and the loop returns.
  volatile std::sig_atomic_t flag = 1;
  llm::TrainConfig cfg = make_cfg(1, 4);
  cfg.stop_flag = &flag;
  llm::Trainer trainer(ctx, model, opt, cfg);
  std::mt19937 rng(1);
  CHECK(trainer.fit(data, nullptr, rng) == 1);
  CHECK(opt.step_count() == 1);
  CHECK(opt.learning_rate() == llm::learning_rate_at(cfg, 1));
  const std::vector<std::vector<float>> stopped_params = take_params(model);
  const std::vector<std::vector<float>> stopped_m = take_tensors(opt.m());
  const std::vector<std::vector<float>> stopped_v = take_tensors(opt.v());

  {
    nn::Context loaded_ctx;
    nn::Transformer::Loaded loaded = nn::Transformer::load(loaded_ctx, ckpt, nn::LoadMode::Resume);
    CHECK(loaded.adamw.has_value());
    CHECK(loaded.adamw->step_count() == 1);
    CHECK(loaded.adamw->learning_rate() == llm::learning_rate_at(cfg, 1));
    check_same_params(stopped_params, loaded.model);
    check_same_tensors(stopped_m, loaded.adamw->m());
    check_same_tensors(stopped_v, loaded.adamw->v());
  }

  flag = 0;
  cfg = make_cfg(2, 4);
  cfg.stop_flag = &flag;
  llm::Trainer trainer2(ctx, model, opt, cfg);
  CHECK(trainer2.fit(data, nullptr, rng) == 4);
  CHECK(opt.step_count() == 4);
  CHECK(opt.learning_rate() == cfg.min_lr);

  // Same seed, same windows, same schedule: a stop after step 1 then steps 2..4
  // matches one uninterrupted run of 4 steps.
  nn::Transformer full(ctx, tiny_config(kVocab), 1);
  nn::AdamW full_opt(llm::adamw_groups(full, 0.1f), 1e-2f, 0.9f, 0.95f, 1e-8f);
  llm::TrainConfig full_cfg = make_cfg(1, 4);
  full_cfg.checkpoint_path.clear();
  llm::Trainer full_trainer(ctx, full, full_opt, full_cfg);
  std::mt19937 full_rng(1);
  CHECK(full_trainer.fit(data, nullptr, full_rng) == 4);
  CHECK(full_opt.step_count() == 4);
  CHECK(full_opt.learning_rate() == full_cfg.min_lr);
  check_same_params(take_params(full), model);
  check_same_tensors(take_tensors(full_opt.m()), opt.m());
  check_same_tensors(take_tensors(full_opt.v()), opt.v());

  std::filesystem::remove_all(dir);
}

}  // namespace

void run_train_tests() {
  check_schedule();
  {
    nn::Context ctx;
    check_fit(ctx);
    check_early_stop(ctx);
    check_min_loss(ctx);
    check_resume(ctx);
    check_generate(ctx);
    check_masked(ctx);
  }
  check_presets();
}
