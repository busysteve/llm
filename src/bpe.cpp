#include "llm/bpe.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace llm {
namespace {

constexpr char kMagic[4] = {'L', 'P', 'B', '1'};

void require_vocab_size(long long vocab_size) {
  if (vocab_size < 259 || vocab_size > static_cast<long long>(std::numeric_limits<int32_t>::max())) {
    throw std::invalid_argument("vocab_size");
  }
}

std::vector<int32_t> bytes_of(std::string_view text) {
  // char may be signed; ids 0..255 are the raw bytes, not code points.
  std::vector<int32_t> ids;
  ids.reserve(text.size());
  for (char ch : text) ids.push_back(static_cast<int32_t>(static_cast<unsigned char>(ch)));
  return ids;
}

// Non-overlapping left to right, so a run of the same symbol collapses in pairs.
std::vector<int32_t> merge_pair(const std::vector<int32_t>& ids, int32_t left, int32_t right,
                                int32_t merged) {
  std::vector<int32_t> out;
  out.reserve(ids.size());
  size_t i = 0;
  while (i < ids.size()) {
    if (i + 1 < ids.size() && ids[i] == left && ids[i + 1] == right) {
      out.push_back(merged);
      i += 2;
    } else {
      out.push_back(ids[i]);
      ++i;
    }
  }
  return out;
}

void append_u32(std::vector<uint8_t>& out, uint32_t value) {
  out.push_back(static_cast<uint8_t>(value));
  out.push_back(static_cast<uint8_t>(value >> 8));
  out.push_back(static_cast<uint8_t>(value >> 16));
  out.push_back(static_cast<uint8_t>(value >> 24));
}

class ByteReader {
 public:
  explicit ByteReader(std::vector<uint8_t> bytes) : bytes_(std::move(bytes)) {}

  uint32_t u32() {
    if (pos_ + 4 > bytes_.size()) throw std::runtime_error("bad tokenizer");
    const uint32_t value = static_cast<uint32_t>(bytes_[pos_]) |
                           (static_cast<uint32_t>(bytes_[pos_ + 1]) << 8) |
                           (static_cast<uint32_t>(bytes_[pos_ + 2]) << 16) |
                           (static_cast<uint32_t>(bytes_[pos_ + 3]) << 24);
    pos_ += 4;
    return value;
  }

  bool done() const { return pos_ == bytes_.size(); }
  size_t size() const { return bytes_.size(); }
  const uint8_t* data() const { return bytes_.data(); }

 private:
  std::vector<uint8_t> bytes_;
  size_t pos_ = 0;
};

std::vector<uint8_t> read_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("tokenizer read");
  in.seekg(0, std::ios::end);
  const std::streamoff size = in.tellg();
  if (size < 0) throw std::runtime_error("tokenizer read");
  in.seekg(0, std::ios::beg);
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  if (size > 0) {
    in.read(reinterpret_cast<char*>(bytes.data()), size);
    if (!in) throw std::runtime_error("tokenizer read");
  }
  return bytes;
}

void write_all(int fd, const uint8_t* data, size_t n) {
  while (n > 0) {
    const ssize_t wrote = ::write(fd, data, n);
    if (wrote < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error(std::string("tokenizer write: ") + std::strerror(errno));
    }
    data += wrote;
    n -= static_cast<size_t>(wrote);
  }
}

// Rename publishes a complete file. A crash mid-write leaves only the .tmp.
void write_atomic(const std::filesystem::path& path, const std::vector<uint8_t>& bytes) {
  const std::filesystem::path tmp = path.string() + ".tmp";
  int fd = ::open(tmp.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
  if (fd < 0) throw std::runtime_error(std::string("tokenizer write: ") + std::strerror(errno));
  try {
    write_all(fd, bytes.data(), bytes.size());
    if (::fsync(fd) != 0) {
      throw std::runtime_error(std::string("tokenizer write: ") + std::strerror(errno));
    }
    if (::close(fd) != 0) {
      fd = -1;
      throw std::runtime_error(std::string("tokenizer write: ") + std::strerror(errno));
    }
    fd = -1;
    std::filesystem::rename(tmp, path);
  } catch (...) {
    if (fd >= 0) ::close(fd);
    ::unlink(tmp.c_str());
    throw;
  }
}

}  // namespace

BpeTokenizer::BpeTokenizer(int vocab_size, std::vector<Merge> merges)
    : vocab_size_(vocab_size),
      pad_id_(static_cast<int32_t>(vocab_size - 3)),
      bos_id_(static_cast<int32_t>(vocab_size - 2)),
      eos_id_(static_cast<int32_t>(vocab_size - 1)),
      merges_(std::move(merges)) {
  if (static_cast<int>(merges_.size()) != vocab_size_ - 259) throw std::runtime_error("bad tokenizer");
  build_pieces();
}

void BpeTokenizer::build_pieces() {
  pieces_.assign(static_cast<size_t>(vocab_size_), std::string());
  for (int i = 0; i < 256; ++i) {
    pieces_[static_cast<size_t>(i)] = std::string(1, static_cast<char>(i));
  }
  for (size_t i = 0; i < merges_.size(); ++i) {
    const int32_t left = merges_[i].left;
    const int32_t right = merges_[i].right;
    const int32_t id = static_cast<int32_t>(256 + i);
    // A merge may only cite symbols that already exist. That excludes specials.
    if (left < 0 || right < 0 || left >= id || right >= id) throw std::runtime_error("bad tokenizer");
    pieces_[static_cast<size_t>(id)] =
        pieces_[static_cast<size_t>(left)] + pieces_[static_cast<size_t>(right)];
  }
}

std::unique_ptr<BpeTokenizer> BpeTokenizer::train(const std::vector<std::string>& texts, int vocab_size) {
  require_vocab_size(vocab_size);
  const int n_merges = vocab_size - 259;

  std::vector<std::vector<int32_t>> words;
  words.reserve(texts.size());
  for (const std::string& text : texts) words.push_back(bytes_of(text));

  std::vector<Merge> merges;
  merges.reserve(static_cast<size_t>(n_merges));
  for (int i = 0; i < n_merges; ++i) {
    const int32_t new_id = static_cast<int32_t>(256 + i);
    // Learned ids occupy 256 .. vocab_size-4. Pad, bos, and eos stay reserved.
    if (new_id > static_cast<int32_t>(vocab_size) - 4) throw std::invalid_argument("vocab_size");

    // Count inside each string. A pair that crosses two training strings is not a pair.
    std::map<std::pair<int32_t, int32_t>, int64_t> counts;
    for (const std::vector<int32_t>& word : words) {
      for (size_t j = 0; j + 1 < word.size(); ++j) {
        counts[{word[j], word[j + 1]}] += 1;
      }
    }
    if (counts.empty()) throw std::invalid_argument("vocab_size");

    std::pair<int32_t, int32_t> best{0, 0};
    int64_t best_count = -1;
    for (const auto& [pair, count] : counts) {
      // Higher count wins. Equal counts take the smaller (left, right).
      if (count > best_count || (count == best_count && pair < best)) {
        best = pair;
        best_count = count;
      }
    }

    merges.push_back(Merge{best.first, best.second});
    for (std::vector<int32_t>& word : words) {
      word = merge_pair(word, best.first, best.second, new_id);
    }
  }

  return std::unique_ptr<BpeTokenizer>(new BpeTokenizer(vocab_size, std::move(merges)));
}

int BpeTokenizer::vocab_size() const { return vocab_size_; }

int32_t BpeTokenizer::bos_id() const { return bos_id_; }

int32_t BpeTokenizer::eos_id() const { return eos_id_; }

int32_t BpeTokenizer::pad_id() const { return pad_id_; }

std::vector<int32_t> BpeTokenizer::encode(std::string_view text, EncodeOptions opt) const {
  std::vector<int32_t> ids = bytes_of(text);
  // Learn order is lowest rank first. Later merges cannot recreate an earlier pair,
  // so one pass per merge is the usual greedy BPE apply.
  for (size_t i = 0; i < merges_.size(); ++i) {
    if (ids.size() < 2) break;
    ids = merge_pair(ids, merges_[i].left, merges_[i].right, static_cast<int32_t>(256 + i));
  }
  if (opt.add_bos) ids.insert(ids.begin(), bos_id_);
  if (opt.add_eos) ids.push_back(eos_id_);
  return ids;
}

std::string BpeTokenizer::decode(const std::vector<int32_t>& ids) const {
  std::string text;
  for (int32_t id : ids) {
    if (id < 0 || id >= vocab_size_) throw std::invalid_argument("id");
    text += pieces_[static_cast<size_t>(id)];
  }
  return text;
}

void BpeTokenizer::save(const std::filesystem::path& path) const {
  std::vector<uint8_t> bytes;
  bytes.reserve(24 + merges_.size() * 8);
  bytes.insert(bytes.end(), kMagic, kMagic + 4);
  append_u32(bytes, static_cast<uint32_t>(vocab_size_));
  append_u32(bytes, static_cast<uint32_t>(merges_.size()));
  append_u32(bytes, static_cast<uint32_t>(pad_id_));
  append_u32(bytes, static_cast<uint32_t>(bos_id_));
  append_u32(bytes, static_cast<uint32_t>(eos_id_));
  for (const Merge& merge : merges_) {
    append_u32(bytes, static_cast<uint32_t>(merge.left));
    append_u32(bytes, static_cast<uint32_t>(merge.right));
  }
  write_atomic(path, bytes);
}

std::unique_ptr<Tokenizer> Tokenizer::load(const std::filesystem::path& path) {
  ByteReader reader(read_file(path));
  if (reader.size() < 4) throw std::runtime_error("bad tokenizer");
  if (std::memcmp(reader.data(), kMagic, 4) != 0) throw std::runtime_error("bad tokenizer magic");
  reader.u32();  // magic bytes, consumed as a little-endian word only to advance

  const uint32_t vocab_u32 = reader.u32();
  const uint32_t n_merges = reader.u32();
  const uint32_t pad = reader.u32();
  const uint32_t bos = reader.u32();
  const uint32_t eos = reader.u32();

  if (vocab_u32 < 259 || vocab_u32 > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
    throw std::runtime_error("bad tokenizer");
  }
  if (n_merges != vocab_u32 - 259) throw std::runtime_error("bad tokenizer");
  if (pad != vocab_u32 - 3 || bos != vocab_u32 - 2 || eos != vocab_u32 - 1) {
    throw std::runtime_error("bad tokenizer");
  }
  const uint64_t expected = 24ull + static_cast<uint64_t>(n_merges) * 8ull;
  if (reader.size() != expected) throw std::runtime_error("bad tokenizer");

  std::vector<BpeTokenizer::Merge> merges;
  merges.reserve(n_merges);
  for (uint32_t i = 0; i < n_merges; ++i) {
    const uint32_t left = reader.u32();
    const uint32_t right = reader.u32();
    const uint32_t new_id = 256u + i;
    if (left >= new_id || right >= new_id) throw std::runtime_error("bad tokenizer");
    merges.push_back(BpeTokenizer::Merge{static_cast<int32_t>(left), static_cast<int32_t>(right)});
  }
  if (!reader.done()) throw std::runtime_error("bad tokenizer");

  return std::unique_ptr<Tokenizer>(new BpeTokenizer(static_cast<int>(vocab_u32), std::move(merges)));
}

}  // namespace llm
