#pragma once

#include "llm/tokenizer.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace llm {

class BpeTokenizer : public Tokenizer {
 public:
  static std::unique_ptr<BpeTokenizer> train(const std::vector<std::string>& texts, int vocab_size);

  int vocab_size() const override;
  std::vector<int32_t> encode(std::string_view text, EncodeOptions opt = {}) const override;
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

  int vocab_size_ = 0;
  int32_t pad_id_ = 0;
  int32_t bos_id_ = 0;
  int32_t eos_id_ = 0;
  std::vector<Merge> merges_;
  // Byte expansion of every id. Pad, bos, and eos stay empty.
  std::vector<std::string> pieces_;
};

}  // namespace llm
