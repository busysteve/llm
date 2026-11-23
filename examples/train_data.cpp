#include "llm/bpe.hpp"
#include "llm/data.hpp"
#include "llm/generate.hpp"
#include "llm/presets.hpp"
#include "llm/train.hpp"

#include "nn/transformer.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <random>

namespace {

struct Options {
  std::filesystem::path files;
  std::filesystem::path out;
  std::string preset = "tiny";
  int vocab = 4096;
  int seq_len = 256;
  int batch = 4;
  int steps = 1000;
  int accum = 1;
  int warmup = 100;
  int log_every = 10;
  int early_stop = 0;
  int eval_every = 0;
  float valid_fraction = 0.1f;
  float min_loss = std::numeric_limits<float>::quiet_NaN();
  float lr = 3e-4f;
  int max_new_tokens = 80;
  float temperature = 0.8f;
  int top_k = 40;
  float top_p = 0.9f;
  int seed = 1;
  bool resume = false;
  bool reuse_tokens = false;
  bool tokenize_only = false;
  bool from_stdin = false;
  std::string prompt;
};

volatile std::sig_atomic_t g_stop = 0;

void on_stop(int sig) {
  if (g_stop != 0) {
    std::signal(sig, SIG_DFL);
    std::raise(sig);
    return;
  }
  g_stop = 1;
}

// Installs the stop handler for one training call. A second signal aborts.
struct StopGuard {
  StopGuard() {
    g_stop = 0;
    std::signal(SIGINT, on_stop);
    std::signal(SIGTERM, on_stop);
  }
  ~StopGuard() {
    std::signal(SIGINT, SIG_DFL);
    std::signal(SIGTERM, SIG_DFL);
  }
  StopGuard(const StopGuard&) = delete;
  StopGuard& operator=(const StopGuard&) = delete;
};

void usage() {
  std::fputs(
      "usage: train_data (--files LIST | --stdin) --out DIR [options]\n"
      "\n"
      "LIST is one book path per line. Blank lines and lines starting with # are\n"
      "ignored. A relative path is resolved from the directory that contains LIST.\n"
      "--stdin reads one document from standard input instead of LIST. That\n"
      "document is one tokenizer string, with one eos token appended.\n"
      "\n"
      "Trains a byte-level BPE tokenizer on those files, trains a preset\n"
      "transformer, writes tokenizer.bpe and model.bin, then loads both and\n"
      "generates a continuation.\n"
      "\n"
      "SIGINT or SIGTERM finishes the current update, writes model.bin, and\n"
      "exits without sampling. --resume continues that checkpoint. Weights,\n"
      "AdamW moments, and the learning-rate step are kept. --steps is the\n"
      "total update count from the start of the run. --reuse-tokens starts a\n"
      "new model on the tokenizer and token file already in --out.\n"
      "--tokenize-only writes that tokenizer and token file and exits before\n"
      "the model. --stdin can supply the text for that tokenization.\n"
      "\n"
      "  --preset tiny|small|base|llama-tiny|llama-small|llama-base|\n"
      "          qwen-tiny|qwen-small|qwen-base|\n"
      "          gpt2-tiny|gpt2-small|gpt2-base|\n"
      "          gpt3-tiny|gpt3-small|gpt3-base|\n"
      "          bert-tiny|bert-small|bert-base\n"
      "                             default tiny\n"
      "  --vocab N                  default 4096 (minimum 259)\n"
      "  --seq-len N                default 256, at most the preset maximum\n"
      "  --batch N                  default 4\n"
      "  --steps N                  default 1000\n"
      "  --accum N                  default 1\n"
      "  --warmup N                 default 100\n"
      "  --log-every N              default 10\n"
      "  --early-stop N             default 0 (off). Stop after N validation\n"
      "                             evals with no loss drop and keep the best\n"
      "                             step, so later updates do not overfit.\n"
      "  --min-loss FLOAT           default off. Stop when a step's training\n"
      "                             loss is at or below FLOAT and keep that\n"
      "                             step.\n"
      "  --eval-every N             default 0, or min(50, --steps) when\n"
      "                             --early-stop is positive\n"
      "  --valid-fraction FLOAT     default 0.1. Suffix held out for that\n"
      "                             validation loss. Used when --early-stop\n"
      "                             is positive.\n"
      "  --lr FLOAT                 default 3e-4\n"
      "  --prompt TEXT              default: opening of the first book,\n"
      "                             or of the stdin document\n"
      "  --max-new-tokens N         default 80\n"
      "  --temperature FLOAT        default 0.8 (0 is greedy)\n"
      "  --top-k N                  default 40 (0 disables)\n"
      "  --top-p FLOAT              default 0.9\n"
      "  --seed N                   default 1\n"
      "  --resume                   continue --out. Requires tokenizer.bpe,\n"
      "                             tokens.bin, tokens.meta, and model.bin.\n"
      "  --reuse-tokens             restart training on the tokens in --out.\n"
      "                             Requires tokenizer.bpe, tokens.bin, and\n"
      "                             tokens.meta. Does not load model.bin.\n"
      "  --tokenize-only            write tokenizer.bpe, tokens.bin, and\n"
      "                             tokens.meta, then exit before the model.\n"
      "  --stdin                    read one document from standard input\n"
      "                             instead of --files.\n"
      "\n"
      "Each book is one tokenizer string, then encoded with an eos token appended.\n"
      "Tokenizer training scans the corpus once per merge, so a large vocab on\n"
      "long books is the slow part. --vocab 1024 is a faster first run.\n"
      "tiny fits in 12 GB at this default batch and length. small prefers 24 GB.\n"
      "base is a 24 GB model; prefer --seq-len 512 or less on a smaller card.\n"
      "llama-* is that width with RoPE and RMSNorm. qwen-* is the Qwen3 block:\n"
      "grouped-query attention, head dim 128, QK-norm, RoPE base 1e6, and\n"
      "RMSNorm epsilon 1e-6. Train a qwen preset from scratch.\n"
      "gpt2-* is the GPT-2 block: GELU, linear bias, d_ff = 4 * d_model, and\n"
      "learned positions, context 1024. gpt3-* is that dense block with context\n"
      "2048. The GPT-3 paper's sparse attention is not in this library.\n"
      "bert-* is that dense block with bidirectional attention, a segment\n"
      "table of size 2, and context 512. It trains masked tokens. The pad id is\n"
      "the mask. This program does not sample from a bert checkpoint. chat\n"
      "loads it and scores the pad id with forward.\n",
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
  bool eval_every_set = false;
  bool min_loss_set = false;
  for (int i = 1; i < argc; ++i) {
    const std::string_view flag = argv[i];
    if (flag == "--help" || flag == "-h") help = true;
    else if (flag == "--files") opt.files = need(i, argc, argv);
    else if (flag == "--out") opt.out = need(i, argc, argv);
    else if (flag == "--preset") opt.preset = need(i, argc, argv);
    else if (flag == "--vocab") opt.vocab = need_int(i, argc, argv);
    else if (flag == "--seq-len") opt.seq_len = need_int(i, argc, argv);
    else if (flag == "--batch") opt.batch = need_int(i, argc, argv);
    else if (flag == "--steps") opt.steps = need_int(i, argc, argv);
    else if (flag == "--accum") opt.accum = need_int(i, argc, argv);
    else if (flag == "--warmup") opt.warmup = need_int(i, argc, argv);
    else if (flag == "--log-every") opt.log_every = need_int(i, argc, argv);
    else if (flag == "--early-stop") opt.early_stop = need_int(i, argc, argv);
    else if (flag == "--min-loss") {
      opt.min_loss = need_float(i, argc, argv);
      min_loss_set = true;
    } else if (flag == "--eval-every") {
      opt.eval_every = need_int(i, argc, argv);
      eval_every_set = true;
    } else if (flag == "--valid-fraction") opt.valid_fraction = need_float(i, argc, argv);
    else if (flag == "--lr") opt.lr = need_float(i, argc, argv);
    else if (flag == "--prompt") opt.prompt = need(i, argc, argv);
    else if (flag == "--max-new-tokens") opt.max_new_tokens = need_int(i, argc, argv);
    else if (flag == "--temperature") opt.temperature = need_float(i, argc, argv);
    else if (flag == "--top-k") opt.top_k = need_int(i, argc, argv);
    else if (flag == "--top-p") opt.top_p = need_float(i, argc, argv);
    else if (flag == "--seed") opt.seed = need_int(i, argc, argv);
    else if (flag == "--resume") opt.resume = true;
    else if (flag == "--reuse-tokens") opt.reuse_tokens = true;
    else if (flag == "--tokenize-only") opt.tokenize_only = true;
    else if (flag == "--stdin") opt.from_stdin = true;
    else throw std::runtime_error(argv[i]);
  }
  if (help) {
    usage();
    std::exit(0);
  }
  if (!eval_every_set) {
    opt.eval_every = (opt.early_stop > 0 && opt.steps > 0) ? std::min(opt.steps, 50) : 0;
  }
  if (opt.early_stop < 0) throw std::runtime_error("early_stop");
  if (min_loss_set && (!std::isfinite(opt.min_loss) || opt.min_loss < 0.f)) {
    throw std::runtime_error("min_loss");
  }
  if (!std::isfinite(opt.valid_fraction) || opt.valid_fraction <= 0.f || opt.valid_fraction >= 1.f) {
    throw std::runtime_error("valid_fraction");
  }
  if (opt.early_stop > 0 && opt.steps > 0 &&
      (opt.eval_every < 1 || opt.eval_every > opt.steps)) {
    throw std::runtime_error("eval_every");
  }
  if (opt.tokenize_only && (opt.resume || opt.reuse_tokens)) {
    throw std::runtime_error("tokenize-only");
  }
  if (opt.resume && opt.reuse_tokens) throw std::runtime_error("reuse-tokens");
  if (opt.from_stdin && (opt.resume || opt.reuse_tokens || !opt.files.empty())) {
    throw std::runtime_error("stdin");
  }
  if (opt.out.empty() || (opt.files.empty() && !opt.from_stdin)) {
    usage();
    throw std::runtime_error("files");
  }
  return opt;
}

std::string trim(std::string line) {
  while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back()))) line.pop_back();
  size_t i = 0;
  while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i]))) ++i;
  return line.substr(i);
}

std::vector<std::filesystem::path> read_list(const std::filesystem::path& list_path) {
  std::ifstream in(list_path);
  if (!in) throw std::runtime_error("files");
  const std::filesystem::path base = list_path.parent_path();
  std::vector<std::filesystem::path> paths;
  std::string line;
  while (std::getline(in, line)) {
    line = trim(std::move(line));
    if (line.empty() || line[0] == '#') continue;
    std::filesystem::path path(line);
    if (path.is_relative()) path = base / path;
    if (!std::filesystem::is_regular_file(path)) throw std::runtime_error(path.string());
    paths.push_back(std::move(path));
  }
  if (paths.empty()) throw std::runtime_error("files");
  return paths;
}

std::string read_stdin() {
  std::string text;
  char buf[65536];
  while (true) {
    const std::size_t n = std::fread(buf, 1, sizeof buf, stdin);
    if (n > 0) text.append(buf, n);
    if (std::ferror(stdin)) throw std::runtime_error("stdin");
    if (n < sizeof buf) break;
  }
  if (text.empty()) throw std::runtime_error("stdin");
  return text;
}

std::string read_book(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error(path.string());
  in.seekg(0, std::ios::end);
  const std::streamoff n = in.tellg();
  if (n <= 0) throw std::runtime_error(path.string());
  in.seekg(0);
  std::string text(static_cast<size_t>(n), '\0');
  in.read(text.data(), n);
  if (!in) throw std::runtime_error(path.string());
  return text;
}

std::string opening_of(const std::string& book) {
  size_t i = 0;
  while (i < book.size() && std::isspace(static_cast<unsigned char>(book[i]))) ++i;
  size_t end = i;
  int points = 0;
  while (end < book.size() && book[end] != '\n' && points < 48) {
    const auto lead = static_cast<unsigned char>(book[end]);
    if ((lead & 0xC0) == 0x80) {
      ++end;
      continue;
    }
    ++points;
    ++end;
    while (end < book.size() && (static_cast<unsigned char>(book[end]) & 0xC0) == 0x80) ++end;
  }
  if (end == i) throw std::runtime_error("prompt");
  return book.substr(i, end - i);
}

struct Holdout {
  std::vector<int32_t> train;
  std::vector<int32_t> valid;
};

// Suffix of the token stream. Both spans still hold one window of seq_len.
Holdout holdout_suffix(const std::vector<int32_t>& ids, int seq_len, float fraction) {
  const int64_t n = static_cast<int64_t>(ids.size());
  const int64_t min_span = static_cast<int64_t>(seq_len) + 1;
  if (min_span < 2 || n < min_span * 2) throw std::runtime_error("short valid");
  int64_t valid_n =
      static_cast<int64_t>(std::llround(static_cast<double>(n) * static_cast<double>(fraction)));
  if (valid_n < min_span) valid_n = min_span;
  if (n - valid_n < min_span) valid_n = n - min_span;
  if (valid_n < min_span || n - valid_n < min_span) throw std::runtime_error("short valid");
  const size_t cut = static_cast<size_t>(n - valid_n);
  Holdout out;
  out.train.assign(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(cut));
  out.valid.assign(ids.begin() + static_cast<std::ptrdiff_t>(cut), ids.end());
  return out;
}

nn::TransformerConfig preset_config(const std::string& name, int vocab) {
  if (name == "tiny") return llm::presets::tiny(vocab);
  if (name == "small") return llm::presets::small(vocab);
  if (name == "base") return llm::presets::base(vocab);
  if (name == "llama-tiny") return llm::presets::llama_tiny(vocab);
  if (name == "llama-small") return llm::presets::llama_small(vocab);
  if (name == "llama-base") return llm::presets::llama_base(vocab);
  if (name == "qwen-tiny") return llm::presets::qwen_tiny(vocab);
  if (name == "qwen-small") return llm::presets::qwen_small(vocab);
  if (name == "qwen-base") return llm::presets::qwen_base(vocab);
  if (name == "gpt2-tiny") return llm::presets::gpt2_tiny(vocab);
  if (name == "gpt2-small") return llm::presets::gpt2_small(vocab);
  if (name == "gpt2-base") return llm::presets::gpt2_base(vocab);
  if (name == "gpt3-tiny") return llm::presets::gpt3_tiny(vocab);
  if (name == "gpt3-small") return llm::presets::gpt3_small(vocab);
  if (name == "gpt3-base") return llm::presets::gpt3_base(vocab);
  if (name == "bert-tiny") return llm::presets::bert_tiny(vocab);
  if (name == "bert-small") return llm::presets::bert_small(vocab);
  if (name == "bert-base") return llm::presets::bert_base(vocab);
  throw std::runtime_error("preset");
}

bool same_config(const nn::TransformerConfig& a, const nn::TransformerConfig& b) {
  return a.vocab_size == b.vocab_size && a.d_model == b.d_model && a.num_heads == b.num_heads &&
         a.num_layers == b.num_layers && a.d_ff == b.d_ff && a.max_seq_len == b.max_seq_len &&
         a.dropout == b.dropout && a.tie_embeddings == b.tie_embeddings && a.bias == b.bias &&
         a.activation == b.activation && a.positional == b.positional && a.head_dim == b.head_dim &&
         a.num_kv_heads == b.num_kv_heads && a.qk_norm == b.qk_norm && a.rope_theta == b.rope_theta &&
         a.rms_eps == b.rms_eps && a.causal == b.causal && a.type_vocab_size == b.type_vocab_size;
}

bool is_file(const std::filesystem::path& path) { return std::filesystem::is_regular_file(path); }

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options opt = parse(argc, argv);
    const std::vector<std::filesystem::path> paths =
        opt.from_stdin ? std::vector<std::filesystem::path>{} : read_list(opt.files);

    const std::filesystem::path tok_path = opt.out / "tokenizer.bpe";
    const std::filesystem::path model_path = opt.out / "model.bin";
    const std::filesystem::path bin_path = opt.out / "tokens.bin";
    const std::filesystem::path meta_path = opt.out / "tokens.meta";
    const bool keep_tokens = opt.resume || opt.reuse_tokens;
    if (opt.resume && (!is_file(tok_path) || !is_file(model_path) || !is_file(bin_path) ||
                       !is_file(meta_path))) {
      throw std::runtime_error("resume");
    }
    if (opt.reuse_tokens && (!is_file(tok_path) || !is_file(bin_path) || !is_file(meta_path))) {
      throw std::runtime_error("reuse-tokens");
    }
    std::filesystem::create_directories(opt.out);

    std::unique_ptr<llm::Tokenizer> tok;
    std::string prompt;
    std::vector<int32_t> ids;
    std::optional<llm::TextDataset> resume_tokens;
    if (keep_tokens) {
      tok = llm::Tokenizer::load(tok_path);
      if (tok->vocab_size() != opt.vocab) throw std::runtime_error("vocab_size");
      resume_tokens.emplace(bin_path, meta_path);
      if (resume_tokens->vocab_size() != tok->vocab_size()) throw std::runtime_error("vocab_size");
      if (resume_tokens->numel() < static_cast<int64_t>(opt.seq_len) + 1) {
        throw std::runtime_error("short corpus");
      }
      prompt = opt.prompt.empty() ? opening_of(read_book(paths.front())) : opt.prompt;
      std::fprintf(stderr, opt.resume ? "loaded %zu tokens\n" : "reusing %zu tokens\n",
                   resume_tokens->ids().size());
    } else {
      std::vector<std::string> books;
      int64_t bytes = 0;
      if (opt.from_stdin) {
        books.push_back(read_stdin());
        bytes = static_cast<int64_t>(books.back().size());
        std::fprintf(stderr, "read stdin (%zu bytes)\n", books.back().size());
      } else {
        books.reserve(paths.size());
        for (const std::filesystem::path& path : paths) {
          books.push_back(read_book(path));
          bytes += static_cast<int64_t>(books.back().size());
          std::fprintf(stderr, "read %s (%zu bytes)\n", path.string().c_str(), books.back().size());
        }
      }
      prompt = opt.prompt.empty() ? opening_of(books.front()) : opt.prompt;

      std::fprintf(stderr, "training tokenizer vocab=%d on %zu books, %lld bytes\n", opt.vocab,
                   books.size(), static_cast<long long>(bytes));
      tok = llm::BpeTokenizer::train(books, opt.vocab);

      llm::EncodeOptions encode_opt;
      encode_opt.add_eos = true;
      for (const std::string& book : books) {
        const std::vector<int32_t> part = tok->encode(book, encode_opt);
        ids.insert(ids.end(), part.begin(), part.end());
      }
      books.clear();
      books.shrink_to_fit();
      std::fprintf(stderr, "encoded %zu tokens\n", ids.size());
      if (static_cast<int64_t>(ids.size()) < static_cast<int64_t>(opt.seq_len) + 1) {
        throw std::runtime_error("short corpus");
      }
    }

    const nn::TransformerConfig config = preset_config(opt.preset, tok->vocab_size());
    if (opt.seq_len > config.max_seq_len) throw std::runtime_error("seq_len");
    const bool encoder = !config.causal;

    // Tokens are on the host. Exit before the training context allocates a model.
    if (opt.tokenize_only) {
      llm::write_token_file(bin_path, meta_path, ids, static_cast<int32_t>(tok->vocab_size()));
      tok->save(tok_path);
      std::fprintf(stderr, "saved %s\n", tok_path.string().c_str());
      std::fprintf(stderr, "tokenize only\n");
      return 0;
    }

    std::vector<int32_t> train_ids;
    std::vector<int32_t> valid_ids;
    if (opt.early_stop > 0) {
      if (opt.seq_len < 1) throw std::runtime_error("seq_len");
      const std::vector<int32_t>& source = keep_tokens ? resume_tokens->ids() : ids;
      Holdout held = holdout_suffix(source, opt.seq_len, opt.valid_fraction);
      std::fprintf(stderr, "holdout train=%zu valid=%zu early_stop=%d eval_every=%d\n",
                   held.train.size(), held.valid.size(), opt.early_stop, opt.eval_every);
      train_ids = std::move(held.train);
      valid_ids = std::move(held.valid);
      resume_tokens.reset();
    }

    if (!keep_tokens) {
      llm::write_token_file(bin_path, meta_path, ids, static_cast<int32_t>(tok->vocab_size()));
      tok->save(tok_path);
      std::fprintf(stderr, "saved %s\n", tok_path.string().c_str());
    }
    ids.clear();
    ids.shrink_to_fit();

    bool stopped = false;
    bool saved_model = false;
    {
      // Release the training context before the inference load allocates another model.
      std::optional<llm::TextDataset> train_mem;
      std::optional<llm::TextDataset> valid_mem;
      std::optional<llm::TextDataset> file_data;
      llm::TextDataset* train_data = nullptr;
      llm::TextDataset* valid_data = nullptr;
      if (opt.early_stop > 0) {
        const int32_t vocab = static_cast<int32_t>(tok->vocab_size());
        train_mem.emplace(std::move(train_ids), vocab);
        valid_mem.emplace(std::move(valid_ids), vocab);
        train_data = &*train_mem;
        valid_data = &*valid_mem;
      } else if (resume_tokens) {
        file_data.emplace(std::move(*resume_tokens));
        resume_tokens.reset();
        train_data = &*file_data;
      } else {
        file_data.emplace(bin_path, meta_path);
        train_data = &*file_data;
      }

      nn::Context ctx;
      std::optional<nn::Transformer> fresh;
      std::optional<nn::Transformer::Loaded> loaded;
      nn::Transformer* model = nullptr;
      std::optional<nn::AdamW> fresh_opt;
      nn::AdamW* optimizer = nullptr;
      uint64_t done = 0;
      if (opt.resume) {
        loaded.emplace(nn::Transformer::load(ctx, model_path, nn::LoadMode::Resume));
        model = &loaded->model;
        const nn::TransformerConfig& got = std::get<nn::TransformerConfig>(model->network().config());
        if (got.vocab_size != tok->vocab_size()) throw std::runtime_error("vocab_size");
        if (!same_config(got, config)) throw std::runtime_error("preset");
        if (!loaded->adamw) throw std::runtime_error("adamw");
        optimizer = &*loaded->adamw;
        done = optimizer->step_count();
      } else {
        fresh.emplace(ctx, config, static_cast<uint32_t>(opt.seed));
        model = &*fresh;
        fresh_opt.emplace(llm::adamw_groups(*model, 0.1f), opt.lr, 0.9f, 0.95f, 1e-8f);
        optimizer = &*fresh_opt;
      }

      // --steps is the total schedule length. A finished checkpoint is left as it is.
      const bool train_more = !opt.resume || done < static_cast<uint64_t>(opt.steps);
      if (train_more) {
        if (done > static_cast<uint64_t>(std::numeric_limits<int>::max()) - 1) {
          throw std::runtime_error("resume");
        }
        if (opt.resume) {
          std::fprintf(stderr, "resuming step %llu\n", static_cast<unsigned long long>(done));
        }
        llm::TrainConfig cfg;
        cfg.learning_rate = opt.lr;
        cfg.min_lr = opt.lr * 0.1f;
        cfg.weight_decay = 0.1f;
        cfg.batch_size = opt.batch;
        cfg.seq_len = opt.seq_len;
        cfg.max_steps = opt.steps;
        cfg.start_step = static_cast<int>(done) + 1;
        cfg.grad_accum_steps = opt.accum;
        cfg.warmup_steps = opt.warmup;
        cfg.log_every = opt.log_every;
        cfg.eval_every = 0;
        cfg.ckpt_every = opt.steps >= 500 ? 500 : 0;
        cfg.checkpoint_path = model_path;
        cfg.min_loss = opt.min_loss;
        cfg.stop_flag = &g_stop;
        if (opt.early_stop > 0) {
          cfg.eval_every = opt.eval_every;
          cfg.eval_batches = 20;
          cfg.early_stop_patience = opt.early_stop;
        }
        if (encoder) {
          cfg.masked_lm = true;
          cfg.mask_token = tok->pad_id();
        }

        StopGuard guard;
        std::mt19937 rng(static_cast<uint32_t>(opt.seed));
        llm::Trainer trainer(ctx, *model, *optimizer, cfg);
        trainer.fit(*train_data, valid_data, rng);
        // A signal already saved this step inside fit. Skip the second write.
        if (g_stop == 0) model->save(model_path, optimizer);
        stopped = g_stop != 0;
        saved_model = true;
      } else {
        std::fprintf(stderr, "resume step %llu reached steps %d\n",
                     static_cast<unsigned long long>(done), opt.steps);
      }
    }
    if (saved_model) std::fprintf(stderr, "saved %s\n", model_path.string().c_str());
    if (stopped) return 0;

    if (encoder) {
      std::printf("encoder: masked lm, mask token %d\n", tok->pad_id());
      return 0;
    }

    nn::Context ctx;
    const std::unique_ptr<llm::Tokenizer> loaded_tok = llm::Tokenizer::load(tok_path);
    nn::Transformer::Loaded loaded =
        nn::Transformer::load(ctx, model_path, nn::LoadMode::Inference);
    llm::Generator generator(ctx, loaded.model, *loaded_tok);
    llm::GenerateConfig gen;
    gen.max_new_tokens = opt.max_new_tokens;
    gen.temperature = opt.temperature;
    gen.top_k = opt.top_k;
    gen.top_p = opt.top_p;
    gen.eos_id = loaded_tok->eos_id();
    std::mt19937 rng(static_cast<uint32_t>(opt.seed) + 1);
    const std::string text = generator.generate(prompt, gen, rng);
    std::printf("prompt: %s\ncontinuation: %s\n", prompt.c_str(), text.c_str());
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "train_data: %s\n", error.what());
    return 1;
  }
}
