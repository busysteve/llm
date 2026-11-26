#include "llm/chat_template.hpp"
#include "llm/generate.hpp"
#include "llm/hf_tokenizer.hpp"
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
  float repetition_penalty = 1.f;
  int seed = 1;
  std::optional<std::string> once;
  llm::ChatWrap wrap = llm::ChatWrap::Dialogue;
};

struct Turn {
  std::string line;
  std::string reply;
};

void usage() {
  std::fputs(
      "usage: chat --model DIR [options]\n"
      "\n"
      "DIR has model.bin and tokenizer.bpe, or model.bin and tokenizer.json\n"
      "from hf import. tokenizer.bpe is used when that file is present.\n"
      "Train one exchange per file so that file's eos ends the assistant turn:\n"
      "\n"
      "  User: What is the harbor bell for?\n"
      "  Assistant: It marks the hour.\n"
      "\n"
      "chat reads a line after you>. /quit and end of input exit. /reset clears\n"
      "the transcript. --once TEXT prints one reply and exits. --plain continues\n"
      "the line with no User or Assistant tags. --template qwen wraps each turn\n"
      "as ChatML for a Qwen GGUF.\n"
      "\n"
      "  --model DIR                required\n"
      "  --max-new-tokens N         default 128\n"
      "  --temperature FLOAT        default 0.8 (0 is greedy)\n"
      "  --top-k N                  default 40 (0 disables)\n"
      "  --top-p FLOAT              default 0.9\n"
      "  --repeat-penalty FLOAT     default 1 (1 disables)\n"
      "  --seed N                   default 1\n"
      "  --once TEXT                one reply, then exit\n"
      "  --plain                    no User:/Assistant: tags\n"
      "  --template qwen            ChatML <|im_start|>user / assistant turns\n"
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
      "--template qwen writes <|im_start|>user, the line, <|im_end|>, and\n"
      "<|im_start|>assistant. A completed turn closes the assistant text with\n"
      "<|im_end|>. The reply is cut at <|im_end|> or a newline and <|im_start|>.\n"
      "The leading space stays on screen. --plain and --template qwen together\n"
      "are rejected.\n"
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
    else if (flag == "--repeat-penalty") opt.repetition_penalty = need_float(i, argc, argv);
    else if (flag == "--seed") opt.seed = need_int(i, argc, argv);
    else if (flag == "--once") opt.once = need(i, argc, argv);
    else if (flag == "--plain") {
      if (opt.wrap == llm::ChatWrap::Qwen) throw std::runtime_error("template");
      opt.wrap = llm::ChatWrap::Plain;
    } else if (flag == "--template") {
      const std::string name = need(i, argc, argv);
      if (name != "qwen") throw std::runtime_error("template");
      if (opt.wrap == llm::ChatWrap::Plain) throw std::runtime_error("template");
      opt.wrap = llm::ChatWrap::Qwen;
    } else throw std::runtime_error(argv[i]);
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

std::string make_prompt(const std::vector<Turn>& turns, const std::string& line, llm::ChatWrap wrap) {
  std::string prompt;
  for (const Turn& turn : turns) prompt += llm::chat_completed_turn(turn.line, turn.reply, wrap);
  prompt += llm::chat_prompt_suffix(line, wrap);
  return prompt;
}

llm::EncodeOptions prompt_options(const llm::Tokenizer& tok) {
  llm::EncodeOptions opt;
  opt.add_bos = tok.add_bos_default();
  opt.add_eos = tok.add_eos_default();
  return opt;
}

bool exceeds_context(const llm::Tokenizer& tok, std::string_view prompt, int max_new_tokens,
                     int max_seq_len) {
  if (max_new_tokens < 1 || max_seq_len < 1) return false;
  const size_t n = tok.encode(prompt, prompt_options(tok)).size();
  return n + static_cast<size_t>(max_new_tokens) > static_cast<size_t>(max_seq_len);
}

bool take_turn(const llm::Tokenizer& tok, llm::Generator& generator, llm::GenerateConfig cfg,
               std::mt19937& rng, std::vector<Turn>& turns, const std::string& line, int max_seq_len,
               llm::ChatWrap wrap) {
  std::vector<Turn> kept = turns;
  std::string prompt;
  for (;;) {
    prompt = make_prompt(kept, line, wrap);
    if (!exceeds_context(tok, prompt, cfg.max_new_tokens, max_seq_len)) break;
    if (kept.empty()) return false;
    kept.erase(kept.begin());
  }

  std::string reply = generator.generate(prompt, cfg, rng);
  reply = llm::chat_cut_reply(std::move(reply), wrap);
  turns = std::move(kept);
  turns.push_back(Turn{line, reply});
  std::printf("model> %s\n", llm::chat_shown_reply(reply, wrap).c_str());
  std::fflush(stdout);
  return true;
}

void interact(const llm::Tokenizer& tok, llm::Generator& generator, llm::GenerateConfig cfg,
              std::mt19937& rng, int max_seq_len, llm::ChatWrap wrap) {
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
      if (!take_turn(tok, generator, cfg, rng, turns, line, max_seq_len, wrap)) {
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
    const std::filesystem::path bpe_path = opt.model / "tokenizer.bpe";
    const std::filesystem::path hf_path = opt.model / "tokenizer.json";
    const std::filesystem::path model_path = opt.model / "model.bin";
    if (!std::filesystem::is_regular_file(model_path) ||
        (!std::filesystem::is_regular_file(bpe_path) && !std::filesystem::is_regular_file(hf_path))) {
      throw std::runtime_error("model");
    }

    nn::Context ctx;
    const std::unique_ptr<llm::Tokenizer> tok = std::filesystem::is_regular_file(bpe_path)
                                                    ? llm::Tokenizer::load(bpe_path)
                                                    : llm::HfTokenizer::load(opt.model);
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
    cfg.repetition_penalty = opt.repetition_penalty;
    cfg.eos_id = tok->eos_id();
    cfg.stop_ids = tok->stop_ids();
    std::mt19937 rng(static_cast<uint32_t>(opt.seed));

    if (opt.once) {
      if (opt.once->empty()) throw std::runtime_error("prompt");
      std::vector<Turn> turns;
      if (!take_turn(*tok, generator, cfg, rng, turns, *opt.once, max_seq_len, opt.wrap)) {
        throw std::runtime_error("max_seq_len");
      }
      return 0;
    }

    interact(*tok, generator, cfg, rng, max_seq_len, opt.wrap);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "chat: %s\n", error.what());
    return 1;
  }
}
