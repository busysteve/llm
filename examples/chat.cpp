#include "llm/generate.hpp"
#include "llm/tokenizer.hpp"

#include "nn/context.hpp"
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

namespace {

struct Options {
  std::filesystem::path model;
  int max_new_tokens = 128;
  float temperature = 0.8f;
  int top_k = 40;
  float top_p = 0.9f;
  int seed = 1;
  int gpu = 0;
  std::optional<std::string> once;
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
      "the transcript. --once TEXT prints one reply and exits.\n"
      "\n"
      "  --model DIR                required\n"
      "  --max-new-tokens N         default 128\n"
      "  --temperature FLOAT        default 0.8 (0 is greedy)\n"
      "  --top-k N                  default 40 (0 disables)\n"
      "  --top-p FLOAT              default 0.9\n"
      "  --seed N                   default 1\n"
      "  --gpu N                    default 0. CUDA device, or the OpenCL\n"
      "                             device this build accepts.\n"
      "  --once TEXT                one reply, then exit\n"
      "\n"
      "The prompt is the transcript, then \"User: \", the line, a newline, and\n"
      "\"Assistant:\". Older turns drop off the front while the encoded prompt plus\n"
      "--max-new-tokens is above max_seq_len. A decoded newline and \"User:\" ends\n"
      "the reply. One leading space is hidden on screen and kept in the transcript.\n",
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
    else if (flag == "--gpu") opt.gpu = need_int(i, argc, argv);
    else if (flag == "--once") opt.once = need(i, argc, argv);
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
  if (opt.gpu < 0 || opt.gpu >= nn::device_count()) throw std::runtime_error("gpu");
  return opt;
}

// Completed turns are "User: ...\nAssistant:...\n". The next one begins at "\nUser: ".
void drop_oldest_turn(std::string& transcript) {
  const size_t pos = transcript.find("\nUser: ");
  if (pos == std::string::npos) {
    transcript.clear();
    return;
  }
  transcript.erase(0, pos + 1);
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

std::string shown_reply(std::string_view reply) {
  if (!reply.empty() && reply.front() == ' ') reply.remove_prefix(1);
  return std::string(reply);
}

bool take_turn(const llm::Tokenizer& tok, llm::Generator& generator, llm::GenerateConfig cfg,
               std::mt19937& rng, std::string& transcript, const std::string& line,
               int max_seq_len) {
  std::string kept = transcript;
  std::string prompt;
  for (;;) {
    prompt = kept;
    prompt += "User: ";
    prompt += line;
    prompt += "\nAssistant:";
    if (!exceeds_context(tok, prompt, cfg.max_new_tokens, max_seq_len)) break;
    if (kept.empty()) return false;
    drop_oldest_turn(kept);
  }

  const std::string reply = stop_at_next_user(generator.generate(prompt, cfg, rng));
  transcript = std::move(kept);
  transcript += "User: ";
  transcript += line;
  transcript += "\nAssistant:";
  transcript += reply;
  transcript += '\n';
  std::printf("model> %s\n", shown_reply(reply).c_str());
  std::fflush(stdout);
  return true;
}

void interact(const llm::Tokenizer& tok, llm::Generator& generator, llm::GenerateConfig cfg,
              std::mt19937& rng, int max_seq_len) {
  std::string transcript;
  for (;;) {
    std::fputs("you> ", stdout);
    std::fflush(stdout);
    std::string line;
    if (!std::getline(std::cin, line)) break;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line == "/quit") break;
    if (line == "/reset") {
      transcript.clear();
      continue;
    }
    if (line.empty()) continue;
    try {
      if (!take_turn(tok, generator, cfg, rng, transcript, line, max_seq_len)) {
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
    std::fprintf(stderr, "gpu %d: %s\n", opt.gpu, nn::device_name(opt.gpu).c_str());
    const std::filesystem::path tok_path = opt.model / "tokenizer.bpe";
    const std::filesystem::path model_path = opt.model / "model.bin";
    if (!std::filesystem::is_regular_file(tok_path) || !std::filesystem::is_regular_file(model_path)) {
      throw std::runtime_error("model");
    }

    nn::Context ctx(opt.gpu);
    const std::unique_ptr<llm::Tokenizer> tok = llm::Tokenizer::load(tok_path);
    nn::Transformer::Loaded loaded =
        nn::Transformer::load(ctx, model_path, nn::LoadMode::Inference);
    llm::Generator generator(ctx, loaded.model, *tok);
    const int max_seq_len =
        std::get<nn::TransformerConfig>(loaded.model.network().config()).max_seq_len;

    llm::GenerateConfig cfg;
    cfg.max_new_tokens = opt.max_new_tokens;
    cfg.temperature = opt.temperature;
    cfg.top_k = opt.top_k;
    cfg.top_p = opt.top_p;
    cfg.eos_id = tok->eos_id();
    std::mt19937 rng(static_cast<uint32_t>(opt.seed));

    if (opt.once) {
      if (opt.once->empty()) throw std::runtime_error("prompt");
      std::string transcript;
      if (!take_turn(*tok, generator, cfg, rng, transcript, *opt.once, max_seq_len)) {
        throw std::runtime_error("max_seq_len");
      }
      return 0;
    }

    interact(*tok, generator, cfg, rng, max_seq_len);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "chat: %s\n", error.what());
    return 1;
  }
}
