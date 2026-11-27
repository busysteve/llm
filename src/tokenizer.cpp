#include "llm/tokenizer.hpp"

#include "llm/hf_tokenizer.hpp"

#include <array>
#include <stdexcept>

namespace llm {

std::unique_ptr<Tokenizer> Tokenizer::load_directory(const std::filesystem::path& dir) {
  if (std::filesystem::is_regular_file(dir / "tokenizer.bpe")) return load(dir / "tokenizer.bpe");
  return HfTokenizer::load(dir);
}

void Tokenizer::copy_directory(const std::filesystem::path& from, const std::filesystem::path& to) {
  if (std::filesystem::is_regular_file(from / "tokenizer.bpe")) {
    std::filesystem::copy_file(from / "tokenizer.bpe", to / "tokenizer.bpe",
                               std::filesystem::copy_options::overwrite_existing);
    return;
  }
  HfTokenizer::copy_directory(from, to);
}

void HfTokenizer::copy_directory(const std::filesystem::path& from,
                                 const std::filesystem::path& to) {
  if (std::filesystem::exists(to / "tokenizer.bpe")) throw std::invalid_argument("output tokenizer.bpe");
  if (std::filesystem::equivalent(from, to)) return;
  constexpr std::array names{"tokenizer.json", "tokenizer_config.json", "generation_config.json",
                             "special_tokens_map.json", "chat_template.jinja"};
  for (const char* name : names) {
    if (!std::filesystem::is_regular_file(from / name) && std::filesystem::exists(to / name)) {
      throw std::invalid_argument(std::string("stale output ") + name);
    }
  }
  for (const char* name : names) {
    if (std::filesystem::is_regular_file(from / name)) {
      std::filesystem::copy_file(from / name, to / name, std::filesystem::copy_options::overwrite_existing);
    }
  }
}

std::vector<int32_t> Tokenizer::unavailable_ids(int model_vocab) const {
  if (!supports_model_vocab(model_vocab)) throw std::invalid_argument("vocab_size");
  for (int32_t id : {bos_id(), eos_id(), pad_id()}) {
    if (id != -1 && !has_token(id)) throw std::invalid_argument("special token id");
  }
  for (int32_t id : stop_ids()) {
    if (!has_token(id)) throw std::invalid_argument("stop token id");
  }
  std::vector<int32_t> unavailable;
  for (int32_t id = 0; id < model_vocab; ++id) {
    if (!has_token(id)) unavailable.push_back(id);
  }
  if (unavailable.size() == static_cast<size_t>(model_vocab)) throw std::invalid_argument("vocab_size");
  return unavailable;
}

}  // namespace llm
