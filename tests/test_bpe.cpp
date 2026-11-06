#include "llm/bpe.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

void run_data_tests();

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

std::vector<std::string> sample_corpus() {
  return {
      "to be or not to be that is the question",
      "whether tis nobler in the mind to suffer",
      "the slings and arrows of outrageous fortune",
      "or to take arms against a sea of troubles",
  };
}

uint32_t read_u32(const std::vector<uint8_t>& bytes, size_t offset) {
  CHECK(offset + 4 <= bytes.size());
  return static_cast<uint32_t>(bytes[offset]) | (static_cast<uint32_t>(bytes[offset + 1]) << 8) |
         (static_cast<uint32_t>(bytes[offset + 2]) << 16) |
         (static_cast<uint32_t>(bytes[offset + 3]) << 24);
}

std::vector<uint8_t> read_bytes(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  CHECK(static_cast<bool>(in));
  in.seekg(0, std::ios::end);
  const auto size = static_cast<size_t>(in.tellg());
  in.seekg(0, std::ios::beg);
  std::vector<uint8_t> bytes(size);
  if (size > 0) in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
  CHECK(in.gcount() == static_cast<std::streamsize>(size));
  return bytes;
}

void write_bytes(const std::filesystem::path& path, const std::vector<uint8_t>& bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  CHECK(out.good());
  if (!bytes.empty()) {
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  }
  out.close();
  CHECK(out.good());
}

void check_round_trip(const llm::Tokenizer& tok, std::string_view text) {
  const std::vector<int32_t> ids = tok.encode(text);
  CHECK(tok.decode(ids) == text);
  for (int32_t id : ids) CHECK(id < tok.pad_id());
}

void check_vocab(const llm::Tokenizer& tok, int vocab_size) {
  CHECK(tok.vocab_size() == vocab_size);
  CHECK(tok.pad_id() == vocab_size - 3);
  CHECK(tok.bos_id() == vocab_size - 2);
  CHECK(tok.eos_id() == vocab_size - 1);
  CHECK(tok.pad_id() == 256 + (vocab_size - 259));
}

}  // namespace

int main() {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "llm-bpe-test";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);

  const std::vector<std::string> corpus = sample_corpus();
  const auto first = llm::BpeTokenizer::train(corpus, 300);
  const auto second = llm::BpeTokenizer::train(corpus, 300);
  check_vocab(*first, 300);
  check_vocab(*second, 300);

  for (const std::string& text : corpus) {
    CHECK(first->encode(text) == second->encode(text));
    check_round_trip(*first, text);
  }
  check_round_trip(*first, "unseen xyz");
  check_round_trip(*first, "");
  std::string raw("\0A", 2);
  raw.push_back(static_cast<char>(0xff));
  check_round_trip(*first, raw);

  const std::vector<int32_t> plain = first->encode("to be");
  const std::vector<int32_t> with_bos = first->encode("to be", llm::EncodeOptions{true, false});
  const std::vector<int32_t> with_eos = first->encode("to be", llm::EncodeOptions{false, true});
  const std::vector<int32_t> with_both = first->encode("to be", llm::EncodeOptions{true, true});
  CHECK(with_bos.size() == plain.size() + 1);
  CHECK(with_eos.size() == plain.size() + 1);
  CHECK(with_both.size() == plain.size() + 2);
  CHECK(with_bos.front() == first->bos_id());
  CHECK(with_eos.back() == first->eos_id());
  CHECK(with_both.front() == first->bos_id());
  CHECK(with_both.back() == first->eos_id());
  CHECK(first->decode(with_both) == "to be");
  CHECK(first->decode({first->pad_id(), first->bos_id(), 65, first->eos_id()}) == "A");
  for (int32_t id : plain) CHECK(id != first->pad_id());

  const auto bytes_only = llm::BpeTokenizer::train(corpus, 259);
  check_vocab(*bytes_only, 259);
  const std::vector<int32_t> raw_ids = bytes_only->encode("Ab");
  CHECK(raw_ids.size() == 2);
  CHECK(raw_ids[0] == 65);
  CHECK(raw_ids[1] == 98);

  // Equal counts: the smaller (left, right) is learned first. 'ab' is 97,98; 'cd' is 99,100.
  const auto one_merge = llm::BpeTokenizer::train({"ab", "cd"}, 260);
  check_vocab(*one_merge, 260);
  CHECK((one_merge->encode("ab") == std::vector<int32_t>{256}));
  CHECK((one_merge->encode("cd") == std::vector<int32_t>{99, 100}));
  const auto two_merges = llm::BpeTokenizer::train({"ab", "cd"}, 261);
  CHECK((two_merges->encode("ab") == std::vector<int32_t>{256}));
  CHECK((two_merges->encode("cd") == std::vector<int32_t>{257}));

  // Three (a,a) pairs outrank one (b,b).
  const auto frequent = llm::BpeTokenizer::train({"aaaa", "bb"}, 260);
  CHECK((frequent->encode("aa") == std::vector<int32_t>{256}));
  CHECK((frequent->encode("bb") == std::vector<int32_t>{98, 98}));
  CHECK((frequent->encode("aaaa") == std::vector<int32_t>{256, 256}));

  // Overlapping pairs collapse left to right, then the new symbol can merge.
  const auto triple = llm::BpeTokenizer::train({"aaa"}, 261);
  CHECK((triple->encode("aa") == std::vector<int32_t>{256}));
  CHECK((triple->encode("aaa") == std::vector<int32_t>{257}));
  const auto quad = llm::BpeTokenizer::train({"aaaa"}, 261);
  CHECK((quad->encode("aaaa") == std::vector<int32_t>{257}));
  CHECK((quad->encode("aaa") == std::vector<int32_t>{256, 97}));

  const std::filesystem::path model = dir / "tok.bpe";
  first->save(model);
  first->save(model);
  second->save(dir / "tok2.bpe");
  CHECK(read_bytes(model) == read_bytes(dir / "tok2.bpe"));

  const std::filesystem::path tie_path = dir / "tie.bpe";
  one_merge->save(tie_path);
  const std::vector<uint8_t> tie = read_bytes(tie_path);
  CHECK(tie.size() == 32);
  CHECK(std::memcmp(tie.data(), "LPB1", 4) == 0);
  CHECK(read_u32(tie, 4) == 260);
  CHECK(read_u32(tie, 8) == 1);
  CHECK(read_u32(tie, 12) == 257);
  CHECK(read_u32(tie, 16) == 258);
  CHECK(read_u32(tie, 20) == 259);
  CHECK(read_u32(tie, 24) == 97);
  CHECK(read_u32(tie, 28) == 98);

  const std::vector<uint8_t> saved = read_bytes(model);
  CHECK(saved.size() == 24u + 41u * 8u);
  CHECK(std::memcmp(saved.data(), "LPB1", 4) == 0);
  CHECK(read_u32(saved, 4) == 300);
  CHECK(read_u32(saved, 8) == 41);
  CHECK(read_u32(saved, 12) == 297);
  CHECK(read_u32(saved, 16) == 298);
  CHECK(read_u32(saved, 20) == 299);

  std::unique_ptr<llm::Tokenizer> loaded = llm::Tokenizer::load(model);
  CHECK(dynamic_cast<llm::BpeTokenizer*>(loaded.get()) != nullptr);
  check_vocab(*loaded, 300);
  CHECK(loaded->encode("to be", llm::EncodeOptions{true, true}) == with_both);
  check_round_trip(*loaded, corpus[2]);
  CHECK(loaded->decode(loaded->encode(raw)) == raw);

  std::unique_ptr<llm::Tokenizer> loaded_tie = llm::Tokenizer::load(tie_path);
  CHECK((loaded_tie->encode("ab") == std::vector<int32_t>{256}));
  CHECK(loaded_tie->decode({256}) == "ab");

  check_invalid([] { llm::BpeTokenizer::train({"ab", "cd"}, 258); }, "vocab_size");
  check_invalid([] { llm::BpeTokenizer::train({"ab", "cd"}, 0); }, "vocab_size");
  check_invalid([] { llm::BpeTokenizer::train({"ab", "cd"}, -1); }, "vocab_size");
  check_invalid([] { llm::BpeTokenizer::train({"a"}, 300); }, "vocab_size");
  check_invalid([&] { loaded->decode({-1}); }, "id");
  check_invalid([&] { loaded->decode({loaded->vocab_size()}); }, "id");
  check_invalid([&] { loaded->decode({100000}); }, "id");

  write_bytes(dir / "bad-magic.bpe", { 'X', 'X', 'X', 'X', 0, 0, 0, 0 });
  check_runtime([&] { llm::Tokenizer::load(dir / "bad-magic.bpe"); }, "bad tokenizer magic");
  write_bytes(dir / "truncated.bpe", { 'L', 'P', 'B', '1' });
  check_runtime([&] { llm::Tokenizer::load(dir / "truncated.bpe"); }, "bad tokenizer");
  check_runtime([&] { llm::Tokenizer::load(dir / "missing.bpe"); }, "tokenizer read");

  std::filesystem::remove_all(dir);
  run_data_tests();
  std::printf("ok\n");
  return 0;
}
