#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace llm {

struct EncodeOptions {
  bool add_bos = false;
  bool add_eos = false;
};

class Tokenizer {
 public:
  virtual ~Tokenizer() = default;

  virtual int vocab_size() const = 0;
  virtual std::vector<int32_t> encode(std::string_view text, EncodeOptions opt = {}) const = 0;
  virtual std::string decode(const std::vector<int32_t>& ids) const = 0;
  virtual int32_t bos_id() const = 0;
  virtual int32_t eos_id() const = 0;
  virtual int32_t pad_id() const = 0;
  virtual void save(const std::filesystem::path& path) const = 0;

  static std::unique_ptr<Tokenizer> load(const std::filesystem::path& path);
};

}  // namespace llm
