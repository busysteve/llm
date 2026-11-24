#pragma once

#include "llm/tokenizer.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace llm {

class BpeTokenizer : public Tokenizer {
 public:
  // n_threads <= 0 uses std::thread::hardware_concurrency(), at least 1.
  static std::unique_ptr<BpeTokenizer> train(const std::vector<std::string>& texts, int vocab_size,
                                             int n_threads = 0);

  int vocab_size() const override;
  std::vector<int32_t> encode(std::string_view text, EncodeOptions opt = {}) const override;
  // Emits ordered batches. Each batch is valid for the duration of the callback.
  void encode_chunks(std::string_view text, EncodeOptions opt,
                     const std::function<void(const std::vector<int32_t>&)>& emit) const;
  std::string decode(const std::vector<int32_t>& ids) const override;
  int32_t bos_id() const override;
  int32_t eos_id() const override;
  int32_t pad_id() const override;
  void save(const std::filesystem::path& path) const override;

 private:
  friend class Tokenizer;

  struct Merge {
    int32_t left = 0;
    int32_t right = 0;
  };

  BpeTokenizer(int vocab_size, std::vector<Merge> merges);
  void build_pieces();
  void encode_range(std::string_view text, std::vector<int32_t>& out) const;

  int vocab_size_ = 0;
  int32_t pad_id_ = 0;
  int32_t bos_id_ = 0;
  int32_t eos_id_ = 0;
  std::vector<Merge> merges_;
  std::unordered_map<uint64_t, int32_t> merge_ids_;
  // Byte expansion of every id. Pad, bos, and eos stay empty.
  std::vector<std::string> pieces_;
};

}  // namespace llm
