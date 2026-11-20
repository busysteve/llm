#include "llm/generate.hpp"
#include "llm/tokenizer.hpp"

#include "nn/transformer.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace {

struct Options {
  std::filesystem::path model;
  int max_new_tokens = 128;
  float temperature = 0.8f;
  int top_k = 40;
  float top_p = 0.9f;
  int seed = 1;
  std::optional<std::string> once;
  bool plain = false;
};

struct Turn {
  std::string line;
  std::string reply;
};

void usage() {
  std::fputs(
      "usage: chat --model DIR [options]\n"
      "\n"
      "DIR is a train_books output directory with tokenizer.bpe and model.bin.\n"
      "Train one exchange per file so that file's eos ends the assistant turn:\n"
      "\n"
      "  User: What is the harbor bell for?\n"
      "  Assistant: It marks the hour.\n"
      "\n"
      "chat reads a line after you>. /quit and end of input exit. /reset clears\n"
      "the transcript. --once TEXT prints one reply and exits. --plain continues\n"
      "the line with no User or Assistant tags.\n"
      "\n"
      "  --model DIR                required\n"
      "  --max-new-tokens N         default 128\n"
      "  --temperature FLOAT        default 0.8 (0 is greedy)\n"
      "  --top-k N                  default 40 (0 disables)\n"
      "  --top-p FLOAT              default 0.9\n"
      "  --seed N                   default 1\n"
      "  --once TEXT                one reply, then exit\n"
      "  --plain                    no User:/Assistant: tags\n"
      "\n"
      "The prompt is the transcript, then \"User: \", the line, a newline, and\n"
      "\"Assistant:\". Older turns drop off the front while the encoded prompt plus\n"
      "--max-new-tokens is above max_seq_len. A decoded newline and \"User:\" ends\n"
      "the reply. One leading space is hidden on screen and kept in the transcript.\n"
      "\n"
      "--plain leaves those tags off. The prompt is the transcript then the line.\n"
      "The reply is printed as generated. A decoded newline and \"User:\" does not\n"
      "end it.\n"
      "\n"
      "A causal checkpoint prefills the prompt and samples with decode_step.\n"
      "A bidirectional checkpoint, including bert-*, appends the pad id and\n"
      "scores that position with forward. The pad id is the mask. Sampling\n"
      "flags are the same. Eos stops the reply and is left off the string.\n",
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
    else if (flag == "--model") opt.model = need(i, argc, argv);
    else if (flag == "--max-new-tokens") opt.max_new_tokens = need_int(i, argc, argv);
    else if (flag == "--temperature") opt.temperature = need_float(i, argc, argv);
    else if (flag == "--top-k") opt.top_k = need_int(i, argc, argv);
    else if (flag == "--top-p") opt.top_p = need_float(i, argc, argv);
    else if (flag == "--seed") opt.seed = need_int(i, argc, argv);
    else if (flag == "--once") opt.once = need(i, argc, argv);
    else if (flag == "--plain") opt.plain = true;
    else throw std::runtime_error(argv[i]);
  }
  if (help) {
    usage();
    std::exit(0);
  }
  if (opt.model.empty()) {
    usage();
    throw std::runtime_error("model");
  }
  return opt;
}

std::string completed_turn(const Turn& turn, bool plain) {
  std::string text;
  if (plain) {
    text += turn.line;
    text += turn.reply;
    text += '\n';
    return text;
  }
  text += "User: ";
  text += turn.line;
  text += "\nAssistant:";
  text += turn.reply;
  text += '\n';
  return text;
}

std::string make_prompt(const std::vector<Turn>& turns, const std::string& line, bool plain) {
  std::string prompt;
  for (const Turn& turn : turns) prompt += completed_turn(turn, plain);
  if (plain) {
    prompt += line;
    return prompt;
  }
  prompt += "User: ";
  prompt += line;
  prompt += "\nAssistant:";
  return prompt;
}

bool exceeds_context(const llm::Tokenizer& tok, std::string_view prompt, int max_new_tokens,
                     int max_seq_len) {
  if (max_new_tokens < 1 || max_seq_len < 1) return false;
  const size_t n = tok.encode(prompt).size();
  return n + static_cast<size_t>(max_new_tokens) > static_cast<size_t>(max_seq_len);
}

std::string stop_at_next_user(std::string reply) {
  const size_t cut = reply.find("\nUser:");
  if (cut != std::string::npos) reply.resize(cut);
  return reply;
}

std::string shown_reply(std::string_view reply, bool plain) {
  if (!plain && !reply.empty() && reply.front() == ' ') reply.remove_prefix(1);
  return std::string(reply);
}

bool take_turn(const llm::Tokenizer& tok, llm::Generator& generator, llm::GenerateConfig cfg,
               std::mt19937& rng, std::vector<Turn>& turns, const std::string& line, int max_seq_len,
               bool plain) {
  std::vector<Turn> kept = turns;
  std::string prompt;
  for (;;) {
    prompt = make_prompt(kept, line, plain);
    if (!exceeds_context(tok, prompt, cfg.max_new_tokens, max_seq_len)) break;
    if (kept.empty()) return false;
    kept.erase(kept.begin());
  }

  std::string reply = generator.generate(prompt, cfg, rng);
  if (!plain) reply = stop_at_next_user(std::move(reply));
  turns = std::move(kept);
  turns.push_back(Turn{line, reply});
  std::printf("model> %s\n", shown_reply(reply, plain).c_str());
  std::fflush(stdout);
  return true;
}

void interact(const llm::Tokenizer& tok, llm::Generator& generator, llm::GenerateConfig cfg,
              std::mt19937& rng, int max_seq_len, bool plain) {
  std::vector<Turn> turns;
  for (;;) {
    std::fputs("you> ", stdout);
    std::fflush(stdout);
    std::string line;
    if (!std::getline(std::cin, line)) break;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line == "/quit") break;
    if (line == "/reset") {
      turns.clear();
      continue;
    }
    if (line.empty()) continue;
    try {
      if (!take_turn(tok, generator, cfg, rng, turns, line, max_seq_len, plain)) {
        std::fprintf(stderr, "chat: max_seq_len\n");
      }
    } catch (const std::exception& error) {
      std::fprintf(stderr, "chat: %s\n", error.what());
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options opt = parse(argc, argv);
    const std::filesystem::path tok_path = opt.model / "tokenizer.bpe";
    const std::filesystem::path model_path = opt.model / "model.bin";
    if (!std::filesystem::is_regular_file(tok_path) || !std::filesystem::is_regular_file(model_path)) {
      throw std::runtime_error("model");
    }

    nn::Context ctx;
    const std::unique_ptr<llm::Tokenizer> tok = llm::Tokenizer::load(tok_path);
    nn::Transformer::Loaded loaded =
        nn::Transformer::load(ctx, model_path, nn::LoadMode::Inference);
    const nn::TransformerConfig& config =
        std::get<nn::TransformerConfig>(loaded.model.network().config());
    llm::Generator generator(ctx, loaded.model, *tok);
    const int max_seq_len = config.max_seq_len;

    llm::GenerateConfig cfg;
    cfg.max_new_tokens = opt.max_new_tokens;
    cfg.temperature = opt.temperature;
    cfg.top_k = opt.top_k;
    cfg.top_p = opt.top_p;
    cfg.eos_id = tok->eos_id();
    std::mt19937 rng(static_cast<uint32_t>(opt.seed));

    if (opt.once) {
      if (opt.once->empty()) throw std::runtime_error("prompt");
      std::vector<Turn> turns;
      if (!take_turn(*tok, generator, cfg, rng, turns, *opt.once, max_seq_len, opt.plain)) {
        throw std::runtime_error("max_seq_len");
      }
      return 0;
    }

    interact(*tok, generator, cfg, rng, max_seq_len, opt.plain);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "chat: %s\n", error.what());
    return 1;
  }
}
