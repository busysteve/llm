#include "llm/bpe.hpp"
#include "llm/data.hpp"
#include "llm/generate.hpp"
#include "llm/presets.hpp"
#include "llm/train.hpp"

#include "nn/transformer.hpp"

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
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
  float lr = 3e-4f;
  int max_new_tokens = 80;
  float temperature = 0.8f;
  int top_k = 40;
  float top_p = 0.9f;
  int seed = 1;
  std::string prompt;
};

void usage() {
  std::fputs(
      "usage: train_books --files LIST --out DIR [options]\n"
      "\n"
      "LIST is one book path per line. Blank lines and lines starting with # are\n"
      "ignored. A relative path is resolved from the directory that contains LIST.\n"
      "\n"
      "Trains a byte-level BPE tokenizer on those files, trains a preset\n"
      "transformer, writes tokenizer.bpe and model.bin, then loads both and\n"
      "generates a continuation.\n"
      "\n"
      "  --preset tiny|small|base|llama-tiny|llama-small|llama-base\n"
      "                             default tiny\n"
      "  --vocab N                  default 4096 (minimum 259)\n"
      "  --seq-len N                default 256, at most 1024\n"
      "  --batch N                  default 4\n"
      "  --steps N                  default 1000\n"
      "  --accum N                  default 1\n"
      "  --warmup N                 default 100\n"
      "  --log-every N              default 10\n"
      "  --lr FLOAT                 default 3e-4\n"
      "  --prompt TEXT              default: opening of the first book\n"
      "  --max-new-tokens N         default 80\n"
      "  --temperature FLOAT        default 0.8 (0 is greedy)\n"
      "  --top-k N                  default 40 (0 disables)\n"
      "  --top-p FLOAT              default 0.9\n"
      "  --seed N                   default 1\n"
      "\n"
      "Each book is one tokenizer string, then encoded with an eos token appended.\n"
      "Tokenizer training scans the corpus once per merge, so a large vocab on\n"
      "long books is the slow part. --vocab 1024 is a faster first run.\n"
      "tiny fits in 12 GB at this default batch and length. small prefers 24 GB.\n"
      "base is a 24 GB model; prefer --seq-len 512 or less on a smaller card.\n",
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
    else if (flag == "--lr") opt.lr = need_float(i, argc, argv);
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
  if (opt.files.empty() || opt.out.empty()) {
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

nn::TransformerConfig preset_config(const std::string& name, int vocab) {
  if (name == "tiny") return llm::presets::tiny(vocab);
  if (name == "small") return llm::presets::small(vocab);
  if (name == "base") return llm::presets::base(vocab);
  if (name == "llama-tiny") return llm::presets::llama_tiny(vocab);
  if (name == "llama-small") return llm::presets::llama_small(vocab);
  if (name == "llama-base") return llm::presets::llama_base(vocab);
  throw std::runtime_error("preset");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options opt = parse(argc, argv);
    const std::vector<std::filesystem::path> paths = read_list(opt.files);

    std::vector<std::string> books;
    books.reserve(paths.size());
    int64_t bytes = 0;
    for (const std::filesystem::path& path : paths) {
      books.push_back(read_book(path));
      bytes += static_cast<int64_t>(books.back().size());
      std::fprintf(stderr, "read %s (%zu bytes)\n", path.string().c_str(), books.back().size());
    }
    const std::string prompt = opt.prompt.empty() ? opening_of(books.front()) : opt.prompt;

    std::fprintf(stderr, "training tokenizer vocab=%d on %zu books, %lld bytes\n", opt.vocab,
                 books.size(), static_cast<long long>(bytes));
    const std::unique_ptr<llm::BpeTokenizer> tok = llm::BpeTokenizer::train(books, opt.vocab);

    llm::EncodeOptions encode_opt;
    encode_opt.add_eos = true;
    std::vector<int32_t> ids;
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

    std::filesystem::create_directories(opt.out);
    const std::filesystem::path tok_path = opt.out / "tokenizer.bpe";
    const std::filesystem::path model_path = opt.out / "model.bin";
    const std::filesystem::path bin_path = opt.out / "tokens.bin";
    const std::filesystem::path meta_path = opt.out / "tokens.meta";
    llm::write_token_file(bin_path, meta_path, ids, static_cast<int32_t>(tok->vocab_size()));
    ids.clear();
    ids.shrink_to_fit();
    tok->save(tok_path);
    std::fprintf(stderr, "saved %s\n", tok_path.string().c_str());

    const nn::TransformerConfig config = preset_config(opt.preset, tok->vocab_size());
    if (opt.seq_len > config.max_seq_len) throw std::runtime_error("seq_len");

    {
      // Release the training context before the inference load allocates another model.
      llm::TextDataset data(bin_path, meta_path);
      nn::Context ctx;
      nn::Transformer model(ctx, config, static_cast<uint32_t>(opt.seed));
      nn::AdamW optimizer(llm::adamw_groups(model, 0.1f), opt.lr, 0.9f, 0.95f, 1e-8f);

      llm::TrainConfig cfg;
      cfg.learning_rate = opt.lr;
      cfg.min_lr = opt.lr * 0.1f;
      cfg.weight_decay = 0.1f;
      cfg.batch_size = opt.batch;
      cfg.seq_len = opt.seq_len;
      cfg.max_steps = opt.steps;
      cfg.grad_accum_steps = opt.accum;
      cfg.warmup_steps = opt.warmup;
      cfg.log_every = opt.log_every;
      cfg.eval_every = 0;
      cfg.ckpt_every = opt.steps >= 500 ? 500 : 0;
      cfg.checkpoint_path = model_path;

      std::mt19937 rng(static_cast<uint32_t>(opt.seed));
      llm::Trainer trainer(ctx, model, optimizer, cfg);
      trainer.fit(data, nullptr, rng);
      model.save(model_path, &optimizer);
    }
    std::fprintf(stderr, "saved %s\n", model_path.string().c_str());

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
    std::fprintf(stderr, "train_books: %s\n", error.what());
    return 1;
  }
}
