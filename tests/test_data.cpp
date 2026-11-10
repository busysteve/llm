#include "llm/data.hpp"

#if defined(LLM_NN_CUDA)
#include <cuda_runtime.h>
#endif

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
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

template <typename F>
void check_runtime(F&& call, const char* message) {
  try {
    call();
  } catch (const std::runtime_error& error) {
    CHECK(std::strcmp(error.what(), message) == 0);
    return;
  }
  CHECK(false);
}

std::string read_text(const std::filesystem::path& path) {
  std::ifstream in(path);
  CHECK(static_cast<bool>(in));
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

std::vector<unsigned char> read_bytes(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  CHECK(static_cast<bool>(in));
  in.seekg(0, std::ios::end);
  const auto size = static_cast<size_t>(in.tellg());
  in.seekg(0, std::ios::beg);
  std::vector<unsigned char> bytes(size);
  if (size > 0) in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
  CHECK(in.gcount() == static_cast<std::streamsize>(size));
  return bytes;
}

void write_text(const std::filesystem::path& path, const std::string& text) {
  std::ofstream out(path, std::ios::trunc);
  CHECK(out.good());
  out << text;
  out.close();
  CHECK(out.good());
}

std::vector<int64_t> index_to_host(const nn::IndexTensor& tensor) {
#if defined(LLM_NN_CUDA)
  tensor.context().synchronize();
  std::vector<int64_t> host(static_cast<size_t>(tensor.numel()));
  const cudaError_t err = cudaMemcpy(host.data(), tensor.data(), host.size() * sizeof(int64_t),
                                      cudaMemcpyDeviceToHost);
  CHECK(err == cudaSuccess);
  return host;
#else
  return tensor.to_host();
#endif
}

// Same draw as TextDataset::next: one uniform start per row, row 0 first.
std::vector<int64_t> replay_starts(std::mt19937& rng, int batch_size, int64_t n_starts) {
  std::uniform_int_distribution<int64_t> start_dist(0, n_starts - 1);
  std::vector<int64_t> starts(static_cast<size_t>(batch_size));
  for (int row = 0; row < batch_size; ++row) starts[static_cast<size_t>(row)] = start_dist(rng);
  return starts;
}

void expect_batch(const std::vector<int32_t>& ids, const llm::Batch& batch,
                  const std::vector<int64_t>& starts, int batch_size, int seq_len) {
  CHECK(batch.input_ids.shape().size() == 2);
  CHECK(batch.labels.shape().size() == 2);
  CHECK(batch.input_ids.shape()[0] == batch_size);
  CHECK(batch.input_ids.shape()[1] == seq_len);
  CHECK(batch.labels.shape()[0] == batch_size);
  CHECK(batch.labels.shape()[1] == seq_len);
  CHECK(batch.input_ids.numel() == static_cast<int64_t>(batch_size) * seq_len);
  CHECK(batch.labels.numel() == batch.input_ids.numel());

  const std::vector<int64_t> input = index_to_host(batch.input_ids);
  const std::vector<int64_t> labels = index_to_host(batch.labels);
  CHECK(static_cast<int>(starts.size()) == batch_size);
  for (int row = 0; row < batch_size; ++row) {
    const int64_t start = starts[static_cast<size_t>(row)];
    CHECK(start >= 0);
    CHECK(start + seq_len < static_cast<int64_t>(ids.size()));
    for (int t = 0; t < seq_len; ++t) {
      const size_t flat = static_cast<size_t>(row) * static_cast<size_t>(seq_len) +
                          static_cast<size_t>(t);
      const int64_t token = int64_t{ids[static_cast<size_t>(start + t)]};
      const int64_t next = int64_t{ids[static_cast<size_t>(start + t + 1)]};
      CHECK(input[flat] == token);
      CHECK(labels[flat] == next);
      if (t + 1 < seq_len) CHECK(labels[flat] == input[flat + 1]);
    }
  }
}

}  // namespace

void run_data_tests() {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "llm-data-test";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const std::filesystem::path bin = dir / "tokens.bin";
  const std::filesystem::path meta = dir / "tokens.meta";

  const std::vector<int32_t> ids = {0, 1, 0x01020304, 65536, 49999};
  llm::write_token_file(bin, meta, ids, 20000000);
  CHECK(read_text(meta) == "5\n20000000\n");
  const std::vector<unsigned char> bytes = read_bytes(bin);
  CHECK(bytes.size() == 20);
  // 49999 is 0xC34F, so its little-endian bytes are 4f c3 00 00.
  const unsigned char want[] = {
      0, 0, 0, 0, 1, 0, 0, 0, 0x04, 0x03, 0x02, 0x01, 0, 0, 1, 0, 0x4f, 0xc3, 0, 0,
  };
  CHECK(std::memcmp(bytes.data(), want, sizeof(want)) == 0);

  const llm::TextDataset loaded(bin, meta);
  CHECK(loaded.numel() == 5);
  CHECK(loaded.vocab_size() == 20000000);

  const std::vector<unsigned char> bin_before = bytes;
  const std::string meta_before = read_text(meta);
  check_invalid([&] { llm::write_token_file(bin, meta, {0, -1}, 10); }, "id");
  check_invalid([&] { llm::write_token_file(bin, meta, {0, 10}, 10); }, "id");
  check_invalid([&] { llm::write_token_file(bin, meta, {}, 0); }, "vocab_size");
  check_invalid([&] { llm::write_token_file(bin, meta, {0}, -3); }, "vocab_size");
  CHECK(read_bytes(bin) == bin_before);
  CHECK(read_text(meta) == meta_before);

  write_text(meta, "5\n49999\n");
  check_runtime([&] { llm::TextDataset(bin, meta); }, "vocab_size");
  write_text(meta, "4\n20000000\n");
  check_runtime([&] { llm::TextDataset(bin, meta); }, "numel");
  write_text(meta, "nope\n20000000\n");
  check_runtime([&] { llm::TextDataset(bin, meta); }, "token meta");
  write_text(meta, "5\n20000000\nextra\n");
  check_runtime([&] { llm::TextDataset(bin, meta); }, "token meta");
  check_runtime([&] { llm::TextDataset(bin, dir / "missing.meta"); }, "token meta");
  write_text(meta, "5\n20000000\n");
  check_runtime([&] { llm::TextDataset(dir / "missing.bin", meta); }, "token read");

  std::vector<unsigned char> short_bytes = bytes;
  short_bytes.pop_back();
  {
    std::ofstream out(bin, std::ios::binary | std::ios::trunc);
    CHECK(out.good());
    out.write(reinterpret_cast<const char*>(short_bytes.data()),
              static_cast<std::streamsize>(short_bytes.size()));
    out.close();
    CHECK(out.good());
  }
  check_runtime([&] { llm::TextDataset(bin, meta); }, "token read");

  llm::write_token_file(bin, meta, ids, 20000000);
  llm::TextDataset data(bin, meta);
  nn::Context ctx;
  std::mt19937 rng(12345);
  check_invalid([&] { data.next(ctx, 0, 1, rng); }, "batch_size");
  check_invalid([&] { data.next(ctx, -2, 1, rng); }, "batch_size");
  check_invalid([&] { data.next(ctx, 1, 0, rng); }, "seq_len");
  check_invalid([&] { data.next(ctx, 1, -4, rng); }, "seq_len");
  check_invalid([&] { data.next(ctx, 1, 5, rng); }, "short corpus");
  check_invalid([&] { data.next(ctx, 2, 8, rng); }, "short corpus");

  {
    std::mt19937 replay = rng;
    const int batch_size = 3;
    const int seq_len = 4;
    llm::Batch batch = data.next(ctx, batch_size, seq_len, rng);
    const std::vector<int64_t> starts = replay_starts(replay, batch_size, data.numel() - seq_len);
    expect_batch(ids, batch, starts, batch_size, seq_len);
    // Both engines consumed the same draws, so the next sample still matches.
    CHECK(replay() == rng());
  }

  std::vector<int32_t> corpus(64);
  for (int i = 0; i < 64; ++i) corpus[static_cast<size_t>(i)] = (i * 10007) % 50000;
  corpus.front() = 0;
  corpus.back() = 49999;
  llm::write_token_file(dir / "corpus.bin", dir / "corpus.meta", corpus, 50000);
  llm::TextDataset corpus_data(dir / "corpus.bin", dir / "corpus.meta");
  CHECK(corpus_data.numel() == 64);
  CHECK(corpus_data.vocab_size() == 50000);

  std::mt19937 corpus_rng(7);
  std::mt19937 corpus_replay = corpus_rng;
  const int batch_size = 4;
  const int seq_len = 8;
  llm::Batch first = corpus_data.next(ctx, batch_size, seq_len, corpus_rng);
  llm::Batch second = corpus_data.next(ctx, batch_size, seq_len, corpus_rng);
  const int64_t n_starts = corpus_data.numel() - seq_len;
  expect_batch(corpus, first, replay_starts(corpus_replay, batch_size, n_starts), batch_size,
               seq_len);
  expect_batch(corpus, second, replay_starts(corpus_replay, batch_size, n_starts), batch_size,
               seq_len);

  llm::write_token_file(dir / "empty.bin", dir / "empty.meta", {}, 8);
  CHECK(read_text(dir / "empty.meta") == "0\n8\n");
  CHECK(read_bytes(dir / "empty.bin").empty());
  llm::TextDataset empty(dir / "empty.bin", dir / "empty.meta");
  CHECK(empty.numel() == 0);
  CHECK(empty.vocab_size() == 8);
  check_invalid([&] { empty.next(ctx, 1, 1, rng); }, "short corpus");

  std::filesystem::remove_all(dir);
}
