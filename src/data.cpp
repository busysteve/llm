#include "llm/data.hpp"

#include <charconv>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace llm {
namespace {

void write_bytes(const std::filesystem::path& path, const std::vector<unsigned char>& bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("token write");
  if (!bytes.empty()) {
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
  }
  out.close();
  if (!out) throw std::runtime_error("token write");
}

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

  std::vector<unsigned char> bytes(static_cast<size_t>(size));
  in.read(reinterpret_cast<char*>(bytes.data()), size);
  if (!in) throw std::runtime_error("token read");

  for (int64_t i = 0; i < count; ++i) {
    const size_t offset = static_cast<size_t>(i) * 4;
    const uint32_t bits = static_cast<uint32_t>(bytes[offset]) |
                          (static_cast<uint32_t>(bytes[offset + 1]) << 8) |
                          (static_cast<uint32_t>(bytes[offset + 2]) << 16) |
                          (static_cast<uint32_t>(bytes[offset + 3]) << 24);
    const int32_t id = static_cast<int32_t>(bits);
    // Meta vocab is the only range check. A stored id outside it is a mismatch.
    if (id < 0 || id >= vocab_size) throw std::runtime_error("vocab_size");
    ids[static_cast<size_t>(i)] = id;
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

void write_token_file(const std::filesystem::path& tokens_bin,
                      const std::filesystem::path& meta_path,
                      const std::vector<int32_t>& ids,
                      int32_t vocab_size) {
  if (vocab_size < 1) throw std::invalid_argument("vocab_size");
  for (int32_t id : ids) {
    if (id < 0 || id >= vocab_size) throw std::invalid_argument("id");
  }

  std::vector<unsigned char> bytes(ids.size() * 4);
  for (size_t i = 0; i < ids.size(); ++i) {
    const uint32_t bits = static_cast<uint32_t>(ids[i]);
    bytes[i * 4] = static_cast<unsigned char>(bits);
    bytes[i * 4 + 1] = static_cast<unsigned char>(bits >> 8);
    bytes[i * 4 + 2] = static_cast<unsigned char>(bits >> 16);
    bytes[i * 4 + 3] = static_cast<unsigned char>(bits >> 24);
  }
  write_bytes(tokens_bin, bytes);

  std::ofstream meta(meta_path, std::ios::trunc);
  if (!meta) throw std::runtime_error("token write");
  meta << static_cast<long long>(ids.size()) << '\n' << vocab_size << '\n';
  meta.close();
  if (!meta) throw std::runtime_error("token write");
}

}  // namespace llm
