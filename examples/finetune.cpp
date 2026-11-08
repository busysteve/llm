#include "llm/data.hpp"
#include "llm/generate.hpp"
#include "llm/presets.hpp"
#include "llm/tokenizer.hpp"
#include "llm/train.hpp"

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
  std::filesystem::path files;
  std::filesystem::path out;
  int seq_len = 256;
  int batch = 4;
  int steps = 200;
  int accum = 1;
  int warmup = 0;
  int log_every = 10;
  float lr = 1e-4f;
  float wd = 0.1f;
  bool resume = false;
  int max_new_tokens = 80;
  float temperature = 0.8f;
  int top_k = 40;
  float top_p = 0.9f;
  int seed = 1;
  std::string prompt;
};

void usage() {
  std::fputs(
      "usage: finetune --from DIR --files LIST --out DIR [options]\n"
      "\n"
      "DIR is a train_books or finetune directory with tokenizer.bpe and model.bin.\n"
      "LIST is one text path per line. Blank lines and lines starting with # are\n"
      "ignored. A relative path is resolved from the directory that contains LIST.\n"
      "\n"
      "Encodes LIST with that tokenizer, keeps training the checkpoint, and writes\n"
      "a new directory. The vocab and the model shape stay as they are in the\n"
      "checkpoint. The learning-rate schedule starts at step 1 of this run.\n"
      "\n"
      "  --from DIR                 required, read-only\n"
      "  --files LIST               required\n"
      "  --out DIR                  required, must differ from --from\n"
      "  --seq-len N                default 256, at most the checkpoint max\n"
      "  --batch N                  default 4\n"
      "  --steps N                  default 200\n"
      "  --accum N                  default 1\n"
      "  --warmup N                 default 0\n"
      "  --log-every N              default 10\n"
      "  --lr FLOAT                 default 1e-4\n"
      "  --wd FLOAT                 default 0.1, fresh AdamW only\n"
      "  --resume                   keep AdamW moments from the checkpoint\n"
      "  --prompt TEXT              default: opening of the first new file\n"
      "  --max-new-tokens N         default 80\n"
      "  --temperature FLOAT        default 0.8 (0 is greedy)\n"
      "  --top-k N                  default 40 (0 disables)\n"
      "  --top-p FLOAT              default 0.9\n"
      "  --seed N                   default 1\n"
      "\n"
      "Each new file is encoded with an eos token appended. Without --resume the\n"
      "run builds a new AdamW. With --resume the checkpoint moments and optimizer\n"
      "step are kept, and --wd is ignored. The cosine floor is one tenth of --lr.\n",
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
    else if (flag == "--files") opt.files = need(i, argc, argv);
    else if (flag == "--out") opt.out = need(i, argc, argv);
    else if (flag == "--seq-len") opt.seq_len = need_int(i, argc, argv);
    else if (flag == "--batch") opt.batch = need_int(i, argc, argv);
    else if (flag == "--steps") opt.steps = need_int(i, argc, argv);
    else if (flag == "--accum") opt.accum = need_int(i, argc, argv);
    else if (flag == "--warmup") opt.warmup = need_int(i, argc, argv);
    else if (flag == "--log-every") opt.log_every = need_int(i, argc, argv);
    else if (flag == "--lr") opt.lr = need_float(i, argc, argv);
    else if (flag == "--wd") opt.wd = need_float(i, argc, argv);
    else if (flag == "--resume") opt.resume = true;
    else if (flag == "--prompt") opt.prompt = need(i, argc, argv);
    else if (flag == "--max-new-tokens") opt.max_new_tokens = need_int(i, argc, argv);
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
  if (opt.from.empty()) {
    usage();
    throw std::runtime_error("from");
  }
  if (opt.files.empty()) {
    usage();
    throw std::runtime_error("files");
  }
  if (opt.out.empty()) {
    usage();
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

    const std::vector<std::filesystem::path> paths = read_list(opt.files);
    const std::unique_ptr<llm::Tokenizer> tok = llm::Tokenizer::load(tok_path);
    std::vector<std::string> books;
    books.reserve(paths.size());
    int64_t bytes = 0;
    for (const std::filesystem::path& path : paths) {
      books.push_back(read_book(path));
      bytes += static_cast<int64_t>(books.back().size());
      std::fprintf(stderr, "read %s (%zu bytes)\n", path.string().c_str(), books.back().size());
    }
    const std::string prompt = opt.prompt.empty() ? opening_of(books.front()) : opt.prompt;

    llm::EncodeOptions encode_opt;
    encode_opt.add_eos = true;
    std::vector<int32_t> ids;
    for (const std::string& book : books) {
      const std::vector<int32_t> part = tok->encode(book, encode_opt);
      ids.insert(ids.end(), part.begin(), part.end());
    }
    books.clear();
    books.shrink_to_fit();
    std::fprintf(stderr, "encoded %zu tokens from %lld bytes, vocab=%d\n", ids.size(),
                 static_cast<long long>(bytes), tok->vocab_size());
    if (static_cast<int64_t>(ids.size()) < static_cast<int64_t>(opt.seq_len) + 1) {
      throw std::runtime_error("short corpus");
    }

    const std::filesystem::path out_tok = opt.out / "tokenizer.bpe";
    const std::filesystem::path model_path = opt.out / "model.bin";
    const std::filesystem::path bin_path = opt.out / "tokens.bin";
    const std::filesystem::path meta_path = opt.out / "tokens.meta";
    llm::write_token_file(bin_path, meta_path, ids, static_cast<int32_t>(tok->vocab_size()));
    ids.clear();
    ids.shrink_to_fit();
    tok->save(out_tok);

    {
      llm::TextDataset data(bin_path, meta_path);
      nn::Context ctx;
      nn::Transformer::Loaded loaded = nn::Transformer::load(
          ctx, from_model, opt.resume ? nn::LoadMode::Resume : nn::LoadMode::Inference);
      const nn::TransformerConfig& config =
          std::get<nn::TransformerConfig>(loaded.model.network().config());
      if (tok->vocab_size() != config.vocab_size) throw std::runtime_error("vocab_size");
      if (opt.seq_len > config.max_seq_len) throw std::runtime_error("seq_len");
      std::fprintf(stderr, "loaded vocab=%d max_seq_len=%d layers=%d d_model=%d\n",
                   config.vocab_size, config.max_seq_len, config.num_layers, config.d_model);

      std::optional<nn::AdamW> fresh;
      nn::AdamW* optimizer = nullptr;
      if (opt.resume) {
        if (!loaded.adamw) throw std::runtime_error("adamw");
        std::fprintf(stderr, "resuming AdamW step %llu\n",
                     static_cast<unsigned long long>(loaded.adamw->step_count()));
        optimizer = &*loaded.adamw;
      } else {
        fresh.emplace(llm::adamw_groups(loaded.model, opt.wd), opt.lr, 0.9f, 0.95f, 1e-8f);
        optimizer = &*fresh;
        std::fprintf(stderr, "fresh AdamW\n");
      }

      llm::TrainConfig cfg;
      cfg.learning_rate = opt.lr;
      cfg.min_lr = opt.lr * 0.1f;
      cfg.weight_decay = opt.wd;
      cfg.batch_size = opt.batch;
      cfg.seq_len = opt.seq_len;
      cfg.max_steps = opt.steps;
      cfg.grad_accum_steps = opt.accum;
      cfg.warmup_steps = opt.warmup;
      cfg.log_every = opt.log_every;
      cfg.eval_every = 0;
      cfg.ckpt_every = opt.steps >= 100 ? 100 : 0;
      cfg.checkpoint_path = model_path;

      std::mt19937 rng(static_cast<uint32_t>(opt.seed));
      llm::Trainer trainer(ctx, loaded.model, *optimizer, cfg);
      trainer.fit(data, nullptr, rng);
      loaded.model.save(model_path, optimizer);
    }
    std::fprintf(stderr, "saved %s\n", model_path.string().c_str());

    nn::Context ctx;
    const std::unique_ptr<llm::Tokenizer> loaded_tok = llm::Tokenizer::load(out_tok);
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
    std::fprintf(stderr, "finetune: %s\n", error.what());
    return 1;
  }
}
