#include "llm/bpe.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <fstream>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace llm {
namespace {

constexpr char kMagic[4] = {'L', 'P', 'B', '1'};
constexpr size_t kEncodeChunkBytes = 64 * 1024;

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

enum class ByteClass { Word, Space, Other };

ByteClass byte_class(unsigned char c) {
  if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f') return ByteClass::Space;
  // Bytes at or above 128 stay with letters so a UTF-8 sequence is one piece.
  if ((c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c >= 128) {
    return ByteClass::Word;
  }
  return ByteClass::Other;
}

// Maximal runs of one byte class. Concatenating them restores the original bytes.
std::string_view take_piece(std::string_view& text) {
  const ByteClass kind = byte_class(static_cast<unsigned char>(text.front()));
  size_t end = 1;
  while (end < text.size() && byte_class(static_cast<unsigned char>(text[end])) == kind) ++end;
  const std::string_view piece = text.substr(0, end);
  text.remove_prefix(end);
  return piece;
}

// Move a proposed split to the end of its run so merge rules cannot cross it.
size_t piece_boundary(std::string_view text, size_t offset) {
  const ByteClass kind = byte_class(static_cast<unsigned char>(text[offset - 1]));
  while (offset < text.size() && byte_class(static_cast<unsigned char>(text[offset])) == kind) {
    ++offset;
  }
  return offset;
}

// Non-overlapping left to right, so a run of the same symbol collapses in pairs.
void merge_pair(std::vector<int32_t>& ids, int32_t left, int32_t right, int32_t merged) {
  const size_t n = ids.size();
  if (n < 2) return;
  size_t w = 0;
  size_t i = 0;
  while (i < n) {
    if (i + 1 < n && ids[i] == left && ids[i + 1] == right) {
      ids[w++] = merged;
      i += 2;
    } else {
      ids[w++] = ids[i++];
    }
  }
  ids.resize(w);
}

uint64_t pack_pair(int32_t left, int32_t right) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(left)) << 32) |
         static_cast<uint32_t>(right);
}

unsigned choose_thread_count(int requested, size_t work_items) {
  unsigned n = 1;
  if (requested > 0) {
    n = static_cast<unsigned>(requested);
  } else {
    n = std::thread::hardware_concurrency();
    if (n == 0) n = 1;
  }
  if (work_items == 0) return 1;
  if (static_cast<size_t>(n) > work_items) n = static_cast<unsigned>(work_items);
  return n;
}

void split_range(unsigned index, unsigned n_threads, size_t n_items, size_t& begin, size_t& end) {
  begin = n_items * static_cast<size_t>(index) / static_cast<size_t>(n_threads);
  end = n_items * static_cast<size_t>(index + 1) / static_cast<size_t>(n_threads);
}

// One reusable worker set. train() issues two jobs per merge, so creating
// threads inside the loop would dominate a large vocab.
class RangePool {
 public:
  explicit RangePool(unsigned n_threads) : n_threads_(n_threads < 1 ? 1u : n_threads) {
    if (n_threads_ == 1) return;
    workers_.reserve(n_threads_ - 1);
    for (unsigned i = 1; i < n_threads_; ++i) {
      workers_.emplace_back([this, i] { worker_loop(i); });
    }
  }

  RangePool(const RangePool&) = delete;
  RangePool& operator=(const RangePool&) = delete;

  ~RangePool() {
    {
      std::lock_guard<std::mutex> guard(mu_);
      stop_ = true;
    }
    cv_start_.notify_all();
    for (std::thread& worker : workers_) worker.join();
  }

  unsigned n_threads() const { return n_threads_; }

  void for_ranges(size_t n_items, const std::function<void(unsigned, size_t, size_t)>& fn) {
    if (n_items == 0) return;
    if (n_threads_ == 1 || n_items == 1) {
      fn(0, 0, n_items);
      return;
    }

    {
      std::lock_guard<std::mutex> guard(mu_);
      fn_ = &fn;
      n_items_ = n_items;
      remaining_ = static_cast<unsigned>(workers_.size());
      error_ = nullptr;
      ++epoch_;
    }
    cv_start_.notify_all();
    invoke(0, n_items, fn);
    std::unique_lock<std::mutex> lock(mu_);
    cv_done_.wait(lock, [&] { return remaining_ == 0; });
    fn_ = nullptr;
    if (error_) {
      std::exception_ptr error = error_;
      error_ = nullptr;
      std::rethrow_exception(error);
    }
  }

 private:
  void invoke(unsigned index, size_t n_items, const std::function<void(unsigned, size_t, size_t)>& fn) {
    size_t begin = 0;
    size_t end = 0;
    split_range(index, n_threads_, n_items, begin, end);
    if (begin >= end) return;
    try {
      fn(index, begin, end);
    } catch (...) {
      std::lock_guard<std::mutex> guard(mu_);
      if (!error_) error_ = std::current_exception();
    }
  }

  void worker_loop(unsigned index) {
    unsigned seen = 0;
    for (;;) {
      std::unique_lock<std::mutex> lock(mu_);
      cv_start_.wait(lock, [&] { return stop_ || epoch_ != seen; });
      if (stop_ && epoch_ == seen) return;
      seen = epoch_;
      const std::function<void(unsigned, size_t, size_t)>* fn = fn_;
      const size_t n_items = n_items_;
      lock.unlock();
      if (fn != nullptr) invoke(index, n_items, *fn);
      lock.lock();
      if (--remaining_ == 0) cv_done_.notify_one();
    }
  }

  unsigned n_threads_ = 1;
  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable cv_start_;
  std::condition_variable cv_done_;
  const std::function<void(unsigned, size_t, size_t)>* fn_ = nullptr;
  size_t n_items_ = 0;
  unsigned epoch_ = 0;
  unsigned remaining_ = 0;
  bool stop_ = false;
  std::exception_ptr error_;
};

void run_parallel(unsigned n_threads, size_t n_items,
                  const std::function<void(unsigned, size_t, size_t)>& fn) {
  if (n_items == 0) return;
  unsigned n = n_threads < 1 ? 1u : n_threads;
  if (static_cast<size_t>(n) > n_items) n = static_cast<unsigned>(n_items);
  if (n == 1) {
    fn(0, 0, n_items);
    return;
  }
  std::mutex mu;
  std::exception_ptr error;
  auto wrap = [&](unsigned index) {
    size_t begin = 0;
    size_t end = 0;
    split_range(index, n, n_items, begin, end);
    if (begin >= end) return;
    try {
      fn(index, begin, end);
    } catch (...) {
      std::lock_guard<std::mutex> guard(mu);
      if (!error) error = std::current_exception();
    }
  };
  std::vector<std::thread> workers;
  workers.reserve(n - 1);
  for (unsigned t = 1; t < n; ++t) workers.emplace_back(wrap, t);
  wrap(0);
  for (std::thread& worker : workers) worker.join();
  if (error) std::rethrow_exception(error);
}

void append_u32(std::vector<uint8_t>& out, uint32_t value) {
  out.push_back(static_cast<uint8_t>(value));
  out.push_back(static_cast<uint8_t>(value >> 8));
  out.push_back(static_cast<uint8_t>(value >> 16));
  out.push_back(static_cast<uint8_t>(value >> 24));
}

uint32_t read_u32(const std::vector<uint8_t>& bytes, size_t& offset) {
  if (offset + 4 > bytes.size()) throw std::runtime_error("bad tokenizer");
  const uint32_t value = static_cast<uint32_t>(bytes[offset]) |
                         (static_cast<uint32_t>(bytes[offset + 1]) << 8) |
                         (static_cast<uint32_t>(bytes[offset + 2]) << 16) |
                         (static_cast<uint32_t>(bytes[offset + 3]) << 24);
  offset += 4;
  return value;
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
  merge_ids_.reserve(merges_.size());
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
    merge_ids_.emplace(pack_pair(left, right), id);
    pieces_[static_cast<size_t>(id)] =
        pieces_[static_cast<size_t>(left)] + pieces_[static_cast<size_t>(right)];
  }
}

std::unique_ptr<BpeTokenizer> BpeTokenizer::train(const std::vector<std::string>& texts, int vocab_size,
                                                  int n_threads) {
  require_vocab_size(vocab_size);
  const int n_merges = vocab_size - 259;

  struct Word {
    std::vector<int32_t> ids;
    int64_t frequency;
  };
  std::vector<Word> words;
  {
    std::unordered_map<std::string_view, size_t> distinct;
    for (std::string_view text : texts) {
      while (!text.empty()) {
        const std::string_view piece = take_piece(text);
        const auto [entry, inserted] = distinct.try_emplace(piece, words.size());
        if (inserted) words.push_back({bytes_of(piece), 1});
        else ++words[entry->second].frequency;
      }
    }
  }

  RangePool pool(choose_thread_count(n_threads, words.size()));
  std::vector<std::unordered_map<uint64_t, int64_t>> locals(pool.n_threads());

  std::vector<Merge> merges;
  merges.reserve(static_cast<size_t>(n_merges));
  for (int i = 0; i < n_merges; ++i) {
    const int32_t new_id = static_cast<int32_t>(256 + i);
    // Learned ids occupy 256 .. vocab_size-4. Pad, bos, and eos stay reserved.
    if (new_id > static_cast<int32_t>(vocab_size) - 4) throw std::invalid_argument("vocab_size");

    for (auto& local : locals) local.clear();
    // Count inside each piece. A pair that crosses whitespace, punctuation, or two
    // training strings is not a pair. Workers fill disjoint slices; the maps
    // are summed before the winner is chosen.
    pool.for_ranges(words.size(), [&](unsigned worker, size_t begin, size_t end) {
      auto& counts = locals[worker];
      for (size_t w = begin; w < end; ++w) {
        const std::vector<int32_t>& word = words[w].ids;
        for (size_t j = 0; j + 1 < word.size(); ++j) {
          counts[pack_pair(word[j], word[j + 1])] += words[w].frequency;
        }
      }
    });

    std::unordered_map<uint64_t, int64_t> counts;
    for (auto& local : locals) {
      if (local.empty()) continue;
      if (counts.empty()) {
        counts = std::move(local);
        continue;
      }
      for (const auto& [key, count] : local) counts[key] += count;
    }
    if (counts.empty()) throw std::invalid_argument("vocab_size");

    uint64_t best = 0;
    int64_t best_count = -1;
    for (const auto& [key, count] : counts) {
      // Higher count wins. Equal counts take the smaller packed (left, right).
      if (count > best_count || (count == best_count && key < best)) {
        best = key;
        best_count = count;
      }
    }

    const int32_t best_left = static_cast<int32_t>(best >> 32);
    const int32_t best_right = static_cast<int32_t>(static_cast<uint32_t>(best));
    merges.push_back(Merge{best_left, best_right});
    pool.for_ranges(words.size(), [&](unsigned, size_t begin, size_t end) {
      for (size_t w = begin; w < end; ++w) {
        merge_pair(words[w].ids, best_left, best_right, new_id);
      }
    });
  }

  return std::unique_ptr<BpeTokenizer>(new BpeTokenizer(vocab_size, std::move(merges)));
}

int BpeTokenizer::vocab_size() const { return vocab_size_; }

int32_t BpeTokenizer::bos_id() const { return bos_id_; }

int32_t BpeTokenizer::eos_id() const { return eos_id_; }

int32_t BpeTokenizer::pad_id() const { return pad_id_; }

void BpeTokenizer::encode_range(std::string_view text, std::vector<int32_t>& out) const {
  constexpr size_t kShortPiece = 32;
  constexpr int32_t kNoMerge = std::numeric_limits<int32_t>::max();
  const auto merge_id = [&](int32_t left, int32_t right) {
    const auto it = merge_ids_.find(pack_pair(left, right));
    return it == merge_ids_.end() ? kNoMerge : it->second;
  };

  struct Link {
    size_t prev;
    size_t next;
  };
  // Lexicographic order gives the lowest rank, then the leftmost occurrence.
  using Candidate = std::pair<int32_t, size_t>;
  const std::greater<Candidate> later;
  std::vector<int32_t> ids;
  std::vector<Link> links;
  std::vector<Candidate> heap;
  while (!text.empty()) {
    const std::string_view piece = take_piece(text);
    if (piece.size() == 1) {
      out.push_back(static_cast<unsigned char>(piece.front()));
      continue;
    }
    ids.clear();
    for (unsigned char byte : piece) ids.push_back(byte);
    if (ids.size() <= kShortPiece) {
      while (ids.size() > 1) {
        int32_t best = kNoMerge;
        for (size_t i = 1; i < ids.size(); ++i) {
          best = std::min(best, merge_id(ids[i - 1], ids[i]));
        }
        if (best == kNoMerge) break;
        const Merge& merge = merges_[best - 256];
        merge_pair(ids, merge.left, merge.right, best);
      }
      out.insert(out.end(), ids.begin(), ids.end());
      continue;
    }

    const size_t end = ids.size();
    heap.clear();
    const auto add_pair = [&](size_t left, size_t right) {
      if (left == end || right == end) return false;
      const int32_t id = merge_id(ids[left], ids[right]);
      if (id == kNoMerge) return false;
      heap.emplace_back(id, left);
      return true;
    };
    for (size_t i = 1; i < end; ++i) add_pair(i - 1, i);
    if (heap.empty()) {
      out.insert(out.end(), ids.begin(), ids.end());
      continue;
    }

    links.resize(end);
    for (size_t i = 0; i < end; ++i) links[i] = {i == 0 ? end : i - 1, i + 1};
    std::make_heap(heap.begin(), heap.end(), later);
    while (!heap.empty()) {
      std::pop_heap(heap.begin(), heap.end(), later);
      const auto [id, left] = heap.back();
      heap.pop_back();
      const Merge& merge = merges_[id - 256];
      const size_t right = links[left].next;
      // The entries for pairs affected by earlier merges can still be in the heap.
      if (right == end || ids[left] != merge.left || ids[right] != merge.right) continue;

      const size_t next = links[right].next;
      ids[left] = id;
      ids[right] = -1;
      links[left].next = next;
      if (next != end) links[next].prev = left;
      if (add_pair(links[left].prev, left)) std::push_heap(heap.begin(), heap.end(), later);
      if (add_pair(left, next)) std::push_heap(heap.begin(), heap.end(), later);
    }
    for (size_t i = 0; i != end; i = links[i].next) out.push_back(ids[i]);
  }
}

std::vector<int32_t> BpeTokenizer::encode(std::string_view text, EncodeOptions opt) const {
  std::vector<int32_t> ids;
  if (text.size() < 2 * kEncodeChunkBytes) {
    if (opt.add_bos) ids.push_back(bos_id_);
    encode_range(text, ids);
    if (opt.add_eos) ids.push_back(eos_id_);
  } else {
    encode_chunks(text, opt, [&](const std::vector<int32_t>& part) {
      ids.insert(ids.end(), part.begin(), part.end());
    });
  }
  return ids;
}

void BpeTokenizer::encode_chunks(
    std::string_view text, EncodeOptions opt,
    const std::function<void(const std::vector<int32_t>&)>& emit) const {
  constexpr size_t kWorkersPerBatch = 8;
  const unsigned workers = choose_thread_count(0, kWorkersPerBatch);
  std::vector<std::string_view> ranges;
  ranges.reserve(workers);
  std::vector<std::vector<int32_t>> parts(workers);
  if (opt.add_bos) parts.front().push_back(bos_id_);

  while (!text.empty()) {
    ranges.clear();
    while (!text.empty() && ranges.size() < workers) {
      const size_t end = text.size() <= kEncodeChunkBytes
                             ? text.size()
                             : piece_boundary(text, kEncodeChunkBytes);
      ranges.push_back(text.substr(0, end));
      text.remove_prefix(end);
    }
    run_parallel(workers, ranges.size(), [&](unsigned, size_t begin, size_t end) {
      for (size_t i = begin; i < end; ++i) encode_range(ranges[i], parts[i]);
    });
    for (size_t i = 0; i < ranges.size(); ++i) {
      if (text.empty() && i + 1 == ranges.size() && opt.add_eos) parts[i].push_back(eos_id_);
      emit(parts[i]);
      parts[i].clear();
    }
    if (text.empty()) return;
  }
  if (opt.add_eos) parts.front().push_back(eos_id_);
  if (!parts.front().empty()) emit(parts.front());
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
  const std::vector<uint8_t> bytes = read_file(path);
  if (bytes.size() < 4) throw std::runtime_error("bad tokenizer");
  if (std::memcmp(bytes.data(), kMagic, 4) != 0) throw std::runtime_error("bad tokenizer magic");

  size_t offset = 0;
  read_u32(bytes, offset);  // magic bytes; advance past the four-byte header

  const uint32_t vocab_u32 = read_u32(bytes, offset);
  const uint32_t n_merges = read_u32(bytes, offset);
  const uint32_t pad = read_u32(bytes, offset);
  const uint32_t bos = read_u32(bytes, offset);
  const uint32_t eos = read_u32(bytes, offset);

  if (vocab_u32 < 259 || vocab_u32 > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
    throw std::runtime_error("bad tokenizer");
  }
  if (n_merges != vocab_u32 - 259) throw std::runtime_error("bad tokenizer");
  if (pad != vocab_u32 - 3 || bos != vocab_u32 - 2 || eos != vocab_u32 - 1) {
    throw std::runtime_error("bad tokenizer");
  }
  const uint64_t expected = 24ull + static_cast<uint64_t>(n_merges) * 8ull;
  if (bytes.size() != expected) throw std::runtime_error("bad tokenizer");

  std::vector<BpeTokenizer::Merge> merges;
  merges.reserve(n_merges);
  for (uint32_t i = 0; i < n_merges; ++i) {
    const uint32_t left = read_u32(bytes, offset);
    const uint32_t right = read_u32(bytes, offset);
    const uint32_t new_id = 256u + i;
    if (left >= new_id || right >= new_id) throw std::runtime_error("bad tokenizer");
    merges.push_back(BpeTokenizer::Merge{static_cast<int32_t>(left), static_cast<int32_t>(right)});
  }
  if (offset != bytes.size()) throw std::runtime_error("bad tokenizer");

  return std::unique_ptr<Tokenizer>(new BpeTokenizer(static_cast<int>(vocab_u32), std::move(merges)));
}

}  // namespace llm
