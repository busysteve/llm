#include "llm/data.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace llm {
namespace {

// Stage each file beside its destination so publishing uses a local rename.
class StagedFile {
 public:
  explicit StagedFile(const std::filesystem::path& destination) {
    std::string pattern = (destination.parent_path() / ".tokens-XXXXXX").string();
    if (::mkdtemp(pattern.data()) == nullptr) throw std::runtime_error("token write");
    directory_ = std::move(pattern);
  }

  StagedFile(const StagedFile&) = delete;
  StagedFile& operator=(const StagedFile&) = delete;

  ~StagedFile() {
    std::error_code error;
    std::filesystem::remove_all(directory_, error);
  }

  std::filesystem::path path() const { return std::filesystem::path(directory_) / "data"; }

 private:
  std::string directory_;
};

int64_t parse_i64(const std::string& line) {
  if (line.empty()) throw std::runtime_error("token meta");
  int64_t value = 0;
  const char* const begin = line.data();
  const char* const end = begin + line.size();
  const std::from_chars_result parsed = std::from_chars(begin, end, value);
  if (parsed.ec != std::errc() || parsed.ptr != end) throw std::runtime_error("token meta");
  return value;
}

struct Meta {
  int64_t numel = 0;
  int32_t vocab_size = 0;
};

Meta read_meta(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("token meta");
  std::string line1;
  std::string line2;
  if (!std::getline(in, line1) || !std::getline(in, line2)) throw std::runtime_error("token meta");
  if (in.peek() != EOF) throw std::runtime_error("token meta");

  const int64_t numel = parse_i64(line1);
  const int64_t vocab = parse_i64(line2);
  if (numel < 0) throw std::runtime_error("token meta");
  if (vocab < 1 || vocab > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
    throw std::runtime_error("token meta");
  }
  return Meta{numel, static_cast<int32_t>(vocab)};
}

std::vector<int32_t> read_ids(const std::filesystem::path& path, int64_t numel, int32_t vocab_size) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("token read");
  in.seekg(0, std::ios::end);
  const std::streamoff size = in.tellg();
  if (size < 0) throw std::runtime_error("token read");
  if (size % 4 != 0) throw std::runtime_error("token read");
  const int64_t count = static_cast<int64_t>(size / 4);
  if (count != numel) throw std::runtime_error("numel");
  in.seekg(0, std::ios::beg);

  std::vector<int32_t> ids(static_cast<size_t>(count));
  if (count == 0) return ids;

  std::array<unsigned char, 16 * 1024> bytes;
  for (size_t start = 0; start < ids.size();) {
    const size_t batch = std::min(ids.size() - start, bytes.size() / 4);
    in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(batch * 4));
    if (!in) throw std::runtime_error("token read");
    for (size_t i = 0; i < batch; ++i) {
      const size_t offset = i * 4;
      const uint32_t bits = static_cast<uint32_t>(bytes[offset]) |
                            (static_cast<uint32_t>(bytes[offset + 1]) << 8) |
                            (static_cast<uint32_t>(bytes[offset + 2]) << 16) |
                            (static_cast<uint32_t>(bytes[offset + 3]) << 24);
      if (bits >= static_cast<uint32_t>(vocab_size)) throw std::runtime_error("vocab_size");
      ids[start + i] = static_cast<int32_t>(bits);
    }
    start += batch;
  }
  return ids;
}

}  // namespace

TextDataset::TextDataset(std::filesystem::path tokens_bin, std::filesystem::path meta_path) {
  const Meta meta = read_meta(meta_path);
  ids_ = read_ids(tokens_bin, meta.numel, meta.vocab_size);
  numel_ = meta.numel;
  vocab_size_ = meta.vocab_size;
}

TextDataset::TextDataset(std::vector<int32_t> ids, int32_t vocab_size) {
  if (vocab_size < 1) throw std::invalid_argument("vocab_size");
  for (int32_t id : ids) {
    if (id < 0 || id >= vocab_size) throw std::invalid_argument("id");
  }
  numel_ = static_cast<int64_t>(ids.size());
  vocab_size_ = vocab_size;
  ids_ = std::move(ids);
}

int64_t TextDataset::numel() const { return numel_; }

int32_t TextDataset::vocab_size() const { return vocab_size_; }

Batch TextDataset::next(nn::Context& ctx, int batch_size, int seq_len, std::mt19937& rng) {
  if (batch_size < 1) throw std::invalid_argument("batch_size");
  if (seq_len < 1) throw std::invalid_argument("seq_len");
  // One example is tokens[s, s+T) and the following token, so the corpus needs T+1.
  if (numel_ < static_cast<int64_t>(seq_len) + 1) throw std::invalid_argument("short corpus");

  const int64_t n_starts = numel_ - static_cast<int64_t>(seq_len);
  // One start per row, row 0 first. A copied engine replays this same distribution.
  std::uniform_int_distribution<int64_t> start_dist(0, n_starts - 1);

  const size_t rows = static_cast<size_t>(batch_size);
  const size_t cols = static_cast<size_t>(seq_len);
  const size_t count = rows * cols;
  std::vector<int64_t> input(count);
  std::vector<int64_t> labels(count);

  for (int row = 0; row < batch_size; ++row) {
    const int64_t start = start_dist(rng);
    for (int t = 0; t < seq_len; ++t) {
      const size_t dst = static_cast<size_t>(row) * cols + static_cast<size_t>(t);
      const size_t at = static_cast<size_t>(start + static_cast<int64_t>(t));
      input[dst] = int64_t{ids_[at]};
      labels[dst] = int64_t{ids_[at + 1]};
    }
  }

  Batch batch{
      nn::IndexTensor(ctx, {batch_size, seq_len}),
      nn::IndexTensor(ctx, {batch_size, seq_len}),
  };
  batch.input_ids.copy_from_host(input.data(), static_cast<int64_t>(count));
  batch.labels.copy_from_host(labels.data(), static_cast<int64_t>(count));
  // copy_from_host is async. These buffers have to stay alive until the stream finishes.
  ctx.synchronize();
  return batch;
}

int64_t write_token_batches(const std::filesystem::path& tokens_bin,
                            const std::filesystem::path& meta_path,
                            int32_t vocab_size, int64_t min_tokens,
                            const std::function<void(const TokenBatchSink&)>& produce) {
  if (vocab_size < 1) throw std::invalid_argument("vocab_size");

  const StagedFile staged_bin(tokens_bin);
  std::ofstream out(staged_bin.path(), std::ios::binary);
  if (!out) throw std::runtime_error("token write");
  std::array<unsigned char, 16 * 1024> bytes;
  size_t used = 0;
  int64_t count = 0;
  const auto flush = [&] {
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(used));
    if (!out) throw std::runtime_error("token write");
    used = 0;
  };

  produce([&](const std::vector<int32_t>& ids) {
    for (int32_t id : ids) {
      if (id < 0 || id >= vocab_size) throw std::invalid_argument("id");
      if (count == std::numeric_limits<int64_t>::max()) throw std::runtime_error("numel");
      const uint32_t bits = static_cast<uint32_t>(id);
      bytes[used++] = static_cast<unsigned char>(bits);
      bytes[used++] = static_cast<unsigned char>(bits >> 8);
      bytes[used++] = static_cast<unsigned char>(bits >> 16);
      bytes[used++] = static_cast<unsigned char>(bits >> 24);
      if (used == bytes.size()) flush();
      ++count;
    }
  });
  if (count < min_tokens) throw std::runtime_error("short corpus");
  if (used != 0) flush();
  out.close();
  if (!out) throw std::runtime_error("token write");

  const StagedFile staged_meta(meta_path);
  std::ofstream meta(staged_meta.path());
  if (!meta) throw std::runtime_error("token write");
  meta << count << '\n' << vocab_size << '\n';
  meta.close();
  if (!meta) throw std::runtime_error("token write");

  std::filesystem::rename(staged_bin.path(), tokens_bin);
  std::filesystem::rename(staged_meta.path(), meta_path);
  return count;
}

void write_token_file(const std::filesystem::path& tokens_bin,
                      const std::filesystem::path& meta_path,
                      const std::vector<int32_t>& ids,
                      int32_t vocab_size) {
  write_token_batches(tokens_bin, meta_path, vocab_size, 0,
                      [&](const TokenBatchSink& emit) { emit(ids); });
}

}  // namespace llm
