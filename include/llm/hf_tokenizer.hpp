#pragma once

#include "llm/tokenizer.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace llm {

// Byte-level BPE from a Hugging Face directory: tokenizer.json, and when present
// tokenizer_config.json and generation_config.json. Ids stay in file order.
// Tokenizer::load still reads LPB1 only.
class HfTokenizer : public Tokenizer {
 public:
  static std::unique_ptr<HfTokenizer> load(const std::filesystem::path& dir);
  static void copy_directory(const std::filesystem::path& from, const std::filesystem::path& to);

  int vocab_size() const override;
  std::vector<int32_t> encode(std::string_view text, EncodeOptions opt = {}) const override;
  std::string decode(const std::vector<int32_t>& ids) const override;
  int32_t bos_id() const override;
  int32_t eos_id() const override;
  int32_t pad_id() const override;
  bool add_bos_default() const override;
  bool add_eos_default() const override;
  std::vector<int32_t> stop_ids() const override;
  bool has_token(int32_t id) const override;
  bool supports_model_vocab(int model_vocab) const override { return model_vocab >= vocab_size_; }
  void save(const std::filesystem::path& path) const override;

 private:
  enum class Pretok { Gpt2, Llama3, Qwen, Qwen35 };

  struct Added {
    std::string content;
    int32_t id = 0;
    bool lstrip = false;
    bool rstrip = false;
    bool single_word = false;
  };

  struct MergeHash {
    size_t operator()(const std::pair<std::string, std::string>& key) const noexcept {
      const size_t left = std::hash<std::string>{}(key.first);
      const size_t right = std::hash<std::string>{}(key.second);
      return left ^ (right + 0x9e3779b97f4a7c15ULL + (left << 6) + (left >> 2));
    }
  };

  struct Hit {
    int32_t id = 0;
    size_t end = 0;
  };

  HfTokenizer() = default;

  std::optional<Hit> match_special(std::string_view text, size_t at) const;
  void encode_gap(std::string_view gap, std::vector<int32_t>& ids) const;
  std::vector<int32_t> bpe_piece(std::string_view piece) const;

  int vocab_size_ = 0;
  int32_t bos_id_ = -1;
  int32_t eos_id_ = -1;
  int32_t pad_id_ = -1;
  bool normalize_nfc_ = false;
  bool add_bos_ = false;
  bool add_eos_ = false;
  bool add_prefix_space_ = false;
  bool strip_prefix_space_ = false;
  bool ignore_merges_ = false;
  Pretok pretok_ = Pretok::Gpt2;
  std::vector<int32_t> stop_ids_;
  std::unordered_map<std::string, int32_t> vocab_;
  std::vector<std::string> id_to_token_;
  std::unordered_map<std::pair<std::string, std::string>, int, MergeHash> merge_rank_;
  std::vector<Added> added_;
};

}  // namespace llm
