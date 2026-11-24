#include "llm/presets.hpp"
#include "llm/rl.hpp"
#include "llm/tokenizer.hpp"

#include "nn/transformer.hpp"

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace {

struct Options {
  std::filesystem::path from;
  std::filesystem::path prompts;
  std::filesystem::path out;
  std::string contains;
  int steps = 100;
  int group = 4;
  int batch_prompts = 1;
  int max_new = 64;
  int warmup = 0;
  int log_every = 1;
  float lr = 1e-5f;
  float kl = 0.f;
  float wd = 0.1f;
  float temperature = 1.f;
  int top_k = 0;
  float top_p = 1.f;
  int seed = 1;
};

void usage() {
  std::fputs(
      "usage: reinforce --from DIR --prompts FILE --contains TEXT --out DIR [options]\n"
      "\n"
      "DIR is a train_data or finetune directory with tokenizer.bpe and model.bin.\n"
      "FILE is one prompt per line. Blank lines and lines starting with # are ignored.\n"
      "\n"
      "Samples a group of completions for each prompt and updates the causal\n"
      "checkpoint from a reward that is 1 when the completion contains TEXT and 0\n"
      "otherwise. The library reward is a callback; TEXT is only this example.\n"
      "\n"
      "  --from DIR                 required, read-only\n"
      "  --prompts FILE             required\n"
      "  --contains TEXT            required\n"
      "  --out DIR                  required, must differ from --from\n"
      "  --steps N                  default 100\n"
      "  --group N                  default 4 (1 is the raw reward)\n"
      "  --batch-prompts N          default 1\n"
      "  --max-new N                default 64\n"
      "  --warmup N                 default 0\n"
      "  --log-every N              default 1\n"
      "  --lr FLOAT                 default 1e-5\n"
      "  --kl FLOAT                 default 0, frozen reference when positive\n"
      "  --wd FLOAT                 default 0.1\n"
      "  --temperature FLOAT        default 1 (0 is greedy)\n"
      "  --top-k N                  default 0 (disables)\n"
      "  --top-p FLOAT              default 1\n"
      "  --seed N                   default 1\n"
      "\n"
      "The policy loads without optimizer state and builds a fresh AdamW. The\n"
      "cosine floor is one tenth of --lr. A positive --kl loads a second copy of\n"
      "the checkpoint and does not train it.\n",
      stderr);
}

std::string need(int& i, int argc, char** argv) {
  if (i + 1 >= argc) throw std::runtime_error(argv[i]);
  ++i;
  return argv[i];
}

int need_int(int& i, int argc, char** argv) {
  const std::string text = need(i, argc, argv);
  char* end = nullptr;
  const long value = std::strtol(text.c_str(), &end, 10);
  if (end == text.c_str() || *end != '\0') throw std::runtime_error(argv[i - 1]);
  return static_cast<int>(value);
}

float need_float(int& i, int argc, char** argv) {
  const std::string text = need(i, argc, argv);
  char* end = nullptr;
  const float value = std::strtof(text.c_str(), &end);
  if (end == text.c_str() || *end != '\0') throw std::runtime_error(argv[i - 1]);
  return value;
}

Options parse(int argc, char** argv) {
  Options opt;
  bool help = false;
  for (int i = 1; i < argc; ++i) {
    const std::string_view flag = argv[i];
    if (flag == "--help" || flag == "-h") help = true;
    else if (flag == "--from") opt.from = need(i, argc, argv);
    else if (flag == "--prompts") opt.prompts = need(i, argc, argv);
    else if (flag == "--contains") opt.contains = need(i, argc, argv);
    else if (flag == "--out") opt.out = need(i, argc, argv);
    else if (flag == "--steps") opt.steps = need_int(i, argc, argv);
    else if (flag == "--group") opt.group = need_int(i, argc, argv);
    else if (flag == "--batch-prompts") opt.batch_prompts = need_int(i, argc, argv);
    else if (flag == "--max-new") opt.max_new = need_int(i, argc, argv);
    else if (flag == "--warmup") opt.warmup = need_int(i, argc, argv);
    else if (flag == "--log-every") opt.log_every = need_int(i, argc, argv);
    else if (flag == "--lr") opt.lr = need_float(i, argc, argv);
    else if (flag == "--kl") opt.kl = need_float(i, argc, argv);
    else if (flag == "--wd") opt.wd = need_float(i, argc, argv);
    else if (flag == "--temperature") opt.temperature = need_float(i, argc, argv);
    else if (flag == "--top-k") opt.top_k = need_int(i, argc, argv);
    else if (flag == "--top-p") opt.top_p = need_float(i, argc, argv);
    else if (flag == "--seed") opt.seed = need_int(i, argc, argv);
    else throw std::runtime_error(argv[i]);
  }
  if (help) {
    usage();
    std::exit(0);
  }
  if (opt.from.empty() || opt.prompts.empty() || opt.contains.empty() || opt.out.empty()) {
    usage();
    if (opt.from.empty()) throw std::runtime_error("from");
    if (opt.prompts.empty()) throw std::runtime_error("prompts");
    if (opt.contains.empty()) throw std::runtime_error("contains");
    throw std::runtime_error("out");
  }
  return opt;
}

std::string trim(std::string line) {
  while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back()))) line.pop_back();
  size_t i = 0;
  while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i]))) ++i;
  return line.substr(i);
}

std::vector<std::string> read_prompts(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("prompts");
  std::vector<std::string> prompts;
  std::string line;
  while (std::getline(in, line)) {
    line = trim(std::move(line));
    if (line.empty() || line[0] == '#') continue;
    prompts.push_back(std::move(line));
  }
  if (prompts.empty()) throw std::runtime_error("prompts");
  return prompts;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options opt = parse(argc, argv);
    const std::filesystem::path tok_path = opt.from / "tokenizer.bpe";
    const std::filesystem::path from_model = opt.from / "model.bin";
    if (!std::filesystem::is_regular_file(tok_path) || !std::filesystem::is_regular_file(from_model)) {
      throw std::runtime_error("from");
    }
    std::filesystem::create_directories(opt.out);
    if (std::filesystem::weakly_canonical(opt.from) == std::filesystem::weakly_canonical(opt.out)) {
      throw std::runtime_error("out");
    }

    const std::vector<std::string> prompts = read_prompts(opt.prompts);
    const std::unique_ptr<llm::Tokenizer> tok = llm::Tokenizer::load(tok_path);
    const std::filesystem::path out_tok = opt.out / "tokenizer.bpe";
    const std::filesystem::path model_path = opt.out / "model.bin";
    tok->save(out_tok);

    nn::Context ctx;
    nn::Transformer::Loaded policy = nn::Transformer::load(ctx, from_model, nn::LoadMode::Inference);
    const nn::TransformerConfig& config =
        std::get<nn::TransformerConfig>(policy.model.network().config());
    if (tok->vocab_size() != config.vocab_size) throw std::runtime_error("vocab_size");
    if (!config.causal) throw std::runtime_error("causal");
    std::fprintf(stderr, "loaded vocab=%d max_seq_len=%d layers=%d d_model=%d prompts=%zu\n",
                 config.vocab_size, config.max_seq_len, config.num_layers, config.d_model,
                 prompts.size());

    std::optional<nn::Transformer::Loaded> frozen;
    nn::Transformer* reference = nullptr;
    if (opt.kl > 0.f) {
      frozen.emplace(nn::Transformer::load(ctx, from_model, nn::LoadMode::Inference));
      reference = &frozen->model;
      std::fprintf(stderr, "frozen reference\n");
    }

    nn::AdamW optimizer(llm::adamw_groups(policy.model, opt.wd), opt.lr, 0.9f, 0.95f, 1e-8f);
    llm::RlConfig cfg;
    cfg.learning_rate = opt.lr;
    cfg.min_lr = opt.lr * 0.1f;
    cfg.warmup_steps = opt.warmup;
    cfg.max_steps = opt.steps;
    cfg.batch_prompts = opt.batch_prompts;
    cfg.group_size = opt.group;
    cfg.max_new_tokens = opt.max_new;
    cfg.temperature = opt.temperature;
    cfg.top_k = opt.top_k;
    cfg.top_p = opt.top_p;
    cfg.kl_coef = opt.kl;
    cfg.log_every = opt.log_every;
    cfg.ckpt_every = opt.steps >= 100 ? 100 : 0;
    cfg.checkpoint_path = model_path;

    const std::string contains = opt.contains;
    std::mt19937 rng(static_cast<uint32_t>(opt.seed));
    llm::RlTrainer trainer(ctx, policy.model, optimizer, cfg, reference);
    trainer.fit(*tok, prompts,
                [&](const llm::Episode& episode) {
                  return episode.text.find(contains) != std::string_view::npos ? 1.f : 0.f;
                },
                rng);
    policy.model.save(model_path, &optimizer);
    std::fprintf(stderr, "saved %s\n", model_path.string().c_str());
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "reinforce: %s\n", error.what());
    return 1;
  }
}
