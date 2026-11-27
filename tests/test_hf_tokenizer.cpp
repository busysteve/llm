#include "llm/bpe.hpp"
#include "llm/chat_template.hpp"
#include "llm/generate.hpp"
#include "llm/hf_tokenizer.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef LLM_FIXTURE_DIR
#error "LLM_FIXTURE_DIR is not set"
#endif

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

void expect_ids(const llm::HfTokenizer& tok, const std::string& text, const std::vector<int32_t>& want) {
  const std::vector<int32_t> got = tok.encode(text);
  if (got != want) {
    std::fprintf(stderr, "ids for [%s]:", text.c_str());
    for (int32_t id : got) std::fprintf(stderr, " %d", id);
    std::fprintf(stderr, "\n");
    CHECK(false);
  }
  CHECK(tok.decode(got) == text);
}

void write_file(const std::filesystem::path& path, const std::string& text) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  CHECK(static_cast<bool>(out));
  out << text;
}

void check_family(const char* name, const std::vector<int32_t>& twelve,
                  const std::vector<int32_t>& digits) {
  const std::filesystem::path dir = std::filesystem::path(LLM_FIXTURE_DIR) / "hf" / name;
  const std::unique_ptr<llm::HfTokenizer> tok = llm::HfTokenizer::load(dir);
  CHECK(tok->vocab_size() == 102);
  expect_ids(*tok, "hi", {15});
  expect_ids(*tok, "12", twelve);
  expect_ids(*tok, "hi hi", {15, 16, 7});
  expect_ids(*tok, "12345", digits);
  expect_ids(*tok, "\xC3\xA9", {8, 9});
  expect_ids(*tok, "<|end|>", {100});
  std::vector<int32_t> end_twelve{100};
  end_twelve.insert(end_twelve.end(), twelve.begin(), twelve.end());
  expect_ids(*tok, "<|end|>12", end_twelve);
  check_runtime([&] { llm::Tokenizer::load(dir / "tokenizer.json"); }, "bad tokenizer magic");
}

}  // namespace

void run_hf_tokenizer_tests() {
  check_family("gpt2", {11}, {14});
  check_family("llama3", {11}, {12, 13});
  // Qwen keeps one digit per piece, so "1" and "2" never meet a merge.
  check_family("qwen", {1, 2}, {1, 2, 3, 4, 5});

  // e + U+0301 is one piece when the letter run includes marks, and two when it does not.
  const std::string accent = "e\xCC\x81";
  const std::filesystem::path qwen35_dir = std::filesystem::path(LLM_FIXTURE_DIR) / "hf" / "qwen35";
  const std::unique_ptr<llm::HfTokenizer> qwen35 = llm::HfTokenizer::load(qwen35_dir);
  expect_ids(*qwen35, "12", {1, 2});
  expect_ids(*qwen35, accent, {8});
  expect_ids(*qwen35, "\xCC\x81", {7});
  const std::filesystem::path qwen_mark = std::filesystem::temp_directory_path() / "llm-hf-qwen-mark";
  std::filesystem::remove_all(qwen_mark);
  std::filesystem::create_directories(qwen_mark);
  std::ifstream qwen35_in(qwen35_dir / "tokenizer.json", std::ios::binary);
  CHECK(static_cast<bool>(qwen35_in));
  std::string qwen_text((std::istreambuf_iterator<char>(qwen35_in)), std::istreambuf_iterator<char>());
  const std::string marks = "[\\\\p{L}\\\\p{M}]+";
  const std::string letters = "\\\\p{L}+";
  const size_t mark_at = qwen_text.find(marks);
  CHECK(mark_at != std::string::npos);
  qwen_text.replace(mark_at, marks.size(), letters);
  const std::string punct_marks = "[^\\\\s\\\\p{L}\\\\p{M}\\\\p{N}]";
  const std::string punct = "[^\\\\s\\\\p{L}\\\\p{N}]";
  const size_t punct_at = qwen_text.find(punct_marks);
  CHECK(punct_at != std::string::npos);
  qwen_text.replace(punct_at, punct_marks.size(), punct);
  write_file(qwen_mark / "tokenizer.json", qwen_text);
  const std::unique_ptr<llm::HfTokenizer> qwen_split = llm::HfTokenizer::load(qwen_mark);
  expect_ids(*qwen_split, "12", {1, 2});
  expect_ids(*qwen_split, accent, {6, 7});
  std::filesystem::remove_all(qwen_mark);

  CHECK(llm::chat_prompt_suffix("hi", llm::ChatWrap::Qwen) ==
        "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n");
  CHECK(llm::chat_completed_turn("hi", "ok", llm::ChatWrap::Qwen) ==
        "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nok<|im_end|>\n");
  CHECK(llm::chat_completed_turn("hi", "ok", llm::ChatWrap::Qwen) +
            llm::chat_prompt_suffix("next", llm::ChatWrap::Qwen) ==
        "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nok<|im_end|>\n"
        "<|im_start|>user\nnext<|im_end|>\n<|im_start|>assistant\n");
  CHECK(llm::chat_cut_reply("ok<|im_end|>\n<|im_start|>user\nno", llm::ChatWrap::Qwen) == "ok");
  CHECK(llm::chat_cut_reply("ok\n<|im_start|>user\nno", llm::ChatWrap::Qwen) == "ok");
  CHECK(llm::chat_shown_reply(" hi", llm::ChatWrap::Qwen) == " hi");
  CHECK(llm::chat_shown_reply(" hi", llm::ChatWrap::Dialogue) == "hi");
  CHECK(llm::chat_completed_turn("hi", " ok", llm::ChatWrap::Dialogue) == "User: hi\nAssistant: ok\n");
  CHECK(llm::chat_prompt_suffix("hi", llm::ChatWrap::Dialogue) == "User: hi\nAssistant:");
  CHECK(llm::chat_prompt({}, "hi", llm::ChatWrap::Qwen25).find("<|im_start|>system\n") == 0);
  CHECK(llm::chat_prompt({{"hi", "<think>\nReasoning\n</think>\n\nAnswer"}}, "next",
                         llm::ChatWrap::Qwen3, false) ==
        "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nAnswer<|im_end|>\n"
        "<|im_start|>user\nnext<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");


  const std::filesystem::path qwen_dir = std::filesystem::path(LLM_FIXTURE_DIR) / "hf" / "qwen";
  const std::unique_ptr<llm::HfTokenizer> qwen = llm::HfTokenizer::load(qwen_dir);
  CHECK(qwen->add_bos_default());
  CHECK(!qwen->add_eos_default());
  CHECK(qwen->bos_id() == 101);
  CHECK(qwen->eos_id() == 100);
  CHECK(qwen->pad_id() == 100);
  CHECK((qwen->stop_ids() == std::vector<int32_t>{100, 14}));
  llm::EncodeOptions with_bos;
  with_bos.add_bos = qwen->add_bos_default();
  CHECK((qwen->encode("12", with_bos) == std::vector<int32_t>{101, 1, 2}));
  CHECK((qwen->encode("12") == std::vector<int32_t>{1, 2}));
  check_invalid([&] { qwen->decode({0}); }, "id");
  check_invalid([&] { qwen->decode({-1}); }, "id");
  check_invalid([&] { qwen->decode({qwen->vocab_size()}); }, "id");
  check_runtime([&] { qwen->save("unused.json"); }, "tokenizer");

  const std::unique_ptr<llm::HfTokenizer> gpt2 =
      llm::HfTokenizer::load(std::filesystem::path(LLM_FIXTURE_DIR) / "hf" / "gpt2");
  CHECK(!gpt2->add_bos_default());
  CHECK(!gpt2->add_eos_default());
  CHECK(gpt2->eos_id() == -1);
  CHECK(gpt2->pad_id() == -1);
  CHECK(gpt2->stop_ids().empty());

  const std::unique_ptr<llm::BpeTokenizer> bpe = llm::BpeTokenizer::train({"ab"}, 259);
  CHECK(!bpe->add_bos_default());
  CHECK(!bpe->add_eos_default());
  CHECK(bpe->stop_ids().empty());

  llm::GenerateConfig stop;
  stop.eos_id = 3;
  CHECK(llm::is_stop_id(stop, 3));
  CHECK(!llm::is_stop_id(stop, 4));
  stop.stop_ids = {5, 6};
  CHECK(llm::is_stop_id(stop, 5));
  CHECK(llm::is_stop_id(stop, 6));
  CHECK(!llm::is_stop_id(stop, 3));
  stop.stop_ids.clear();
  stop.eos_id = -1;
  CHECK(!llm::is_stop_id(stop, 3));

  const std::filesystem::path temp = std::filesystem::temp_directory_path() / "llm-hf-tokenizer-test";
  std::filesystem::remove_all(temp);
  std::filesystem::create_directories(temp);
  std::filesystem::copy(qwen_dir, temp / "single", std::filesystem::copy_options::recursive);
  write_file(temp / "single" / "generation_config.json", "{\"eos_token_id\": 100}\n");
  const std::unique_ptr<llm::HfTokenizer> one = llm::HfTokenizer::load(temp / "single");
  CHECK((one->stop_ids() == std::vector<int32_t>{100}));
  auto unavailable = one->unavailable_ids(104);
  CHECK(unavailable.front() == 0 && unavailable.back() == 103);
  check_invalid([&] { one->unavailable_ids(101); }, "vocab_size");
  std::mt19937 rng(1);
  llm::GenerateConfig greedy;
  greedy.temperature = 0;
  CHECK(llm::sample_token({100.f, 2.f, 1.f, 99.f}, greedy, rng, nullptr, 0, {0, 3}) == 1);
  write_file(temp / "single" / "generation_config.json", R"({"eos_token_id":[100, 103]})");
  const auto invalid_stop = llm::Tokenizer::load_directory(temp / "single");
  check_invalid([&] { invalid_stop->unavailable_ids(104); }, "stop token id");

  write_file(temp / "single" / "generation_config.json", R"({"eos_token_id":100,"eos_token_id":14})");
  check_runtime([&] { llm::HfTokenizer::load(temp / "single"); }, "tokenizer");

  write_file(temp / "single" / "generation_config.json", R"({"eos_token_id":[100, 14]})");
  write_file(temp / "single" / "chat_template.jinja", "{{ messages[0].content }}");
  std::filesystem::create_directories(temp / "copy");
  llm::Tokenizer::copy_directory(temp / "single", temp / "copy");
  const auto copied = llm::Tokenizer::load_directory(temp / "copy");
  CHECK((copied->stop_ids() == std::vector<int32_t>{100, 14}));
  CHECK(std::filesystem::is_regular_file(temp / "copy" / "chat_template.jinja"));

  std::ifstream source(qwen_dir / "tokenizer.json");
  std::string nfc_json((std::istreambuf_iterator<char>(source)), std::istreambuf_iterator<char>());
  const std::string no_normalizer = "\"normalizer\": null";
  const size_t normalizer_pos = nfc_json.find(no_normalizer);
  CHECK(normalizer_pos != std::string::npos);
  nfc_json.replace(normalizer_pos, no_normalizer.size(), R"("normalizer": {"type":"NFC"})");
  write_file(temp / "nfc" / "tokenizer.json", nfc_json);
  const auto nfc = llm::Tokenizer::load_directory(temp / "nfc");
  CHECK((nfc->encode(accent) == std::vector<int32_t>{8, 9}));
  CHECK(nfc->decode(nfc->encode(accent)) == "\xC3\xA9");


  write_file(temp / "pairs" / "tokenizer.json",
             "{"
             "\"normalizer\":null,"
             "\"pre_tokenizer\":{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"use_regex\":true},"
             "\"decoder\":{\"type\":\"ByteLevel\",\"add_prefix_space\":false},"
             "\"model\":{\"type\":\"BPE\",\"vocab\":{\"h\":1,\"i\":2,\"hi\":3},\"merges\":[[\"h\",\"i\"]]}"
             "}");
  const std::unique_ptr<llm::HfTokenizer> pairs = llm::HfTokenizer::load(temp / "pairs");
  CHECK((pairs->encode("hi") == std::vector<int32_t>{3}));
  CHECK(pairs->decode({3}) == "hi");

  write_file(temp / "prefix" / "tokenizer.json",
             "{"
             "\"normalizer\":null,"
             "\"pre_tokenizer\":{\"type\":\"ByteLevel\",\"add_prefix_space\":true,\"use_regex\":true},"
             "\"decoder\":{\"type\":\"ByteLevel\",\"add_prefix_space\":true},"
             "\"model\":{\"type\":\"BPE\",\"vocab\":{"
             "\"h\":1,\"i\":2,\"\\u0120\":3,\"hi\":4,\"\\u0120h\":5"
             "},\"merges\":[\"\\u0120 h\",\"h i\"]}"
             "}");
  const std::unique_ptr<llm::HfTokenizer> prefix = llm::HfTokenizer::load(temp / "prefix");
  CHECK((prefix->encode("hi") == std::vector<int32_t>{5, 2}));
  CHECK(prefix->decode(prefix->encode("hi")) == "hi");

  write_file(temp / "lstrip" / "tokenizer.json",
             "{"
             "\"normalizer\":null,"
             "\"added_tokens\":[{\"id\":9,\"content\":\"<|end|>\",\"lstrip\":true,\"special\":true}],"
             "\"pre_tokenizer\":{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"use_regex\":true},"
             "\"decoder\":{\"type\":\"ByteLevel\"},"
             "\"model\":{\"type\":\"BPE\",\"vocab\":{\"\\u0120\":3},\"merges\":[]}"
             "}");
  const std::unique_ptr<llm::HfTokenizer> stripped = llm::HfTokenizer::load(temp / "lstrip");
  CHECK((stripped->encode(" <|end|>") == std::vector<int32_t>{9}));

  check_runtime([&] { llm::HfTokenizer::load(temp / "missing"); }, "tokenizer");
  write_file(temp / "unigram" / "tokenizer.json", "{\"model\":{\"type\":\"Unigram\"}}");
  check_runtime([&] { llm::HfTokenizer::load(temp / "unigram"); }, "tokenizer");
  write_file(temp / "norm" / "tokenizer.json",
             "{\"normalizer\":{\"type\":\"NFC\"},\"model\":{\"type\":\"BPE\"}}");
  check_runtime([&] { llm::HfTokenizer::load(temp / "norm"); }, "tokenizer");
  write_file(temp / "regex" / "tokenizer.json",
             "{"
             "\"normalizer\":null,"
             "\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":["
             "{\"type\":\"Split\",\"pattern\":{\"Regex\":\"abc\"},\"behavior\":\"Isolated\"},"
             "{\"type\":\"ByteLevel\",\"use_regex\":false}]},"
             "\"decoder\":{\"type\":\"ByteLevel\"},"
             "\"model\":{\"type\":\"BPE\",\"vocab\":{\"h\":1},\"merges\":[]}"
             "}");
  check_runtime([&] { llm::HfTokenizer::load(temp / "regex"); }, "tokenizer");
  write_file(temp / "decoder" / "tokenizer.json",
             "{"
             "\"normalizer\":null,"
             "\"pre_tokenizer\":{\"type\":\"ByteLevel\",\"use_regex\":true},"
             "\"decoder\":{\"type\":\"WordPiece\"},"
             "\"model\":{\"type\":\"BPE\",\"vocab\":{\"h\":1},\"merges\":[]}"
             "}");
  check_runtime([&] { llm::HfTokenizer::load(temp / "decoder"); }, "tokenizer");

  std::filesystem::remove_all(temp);
}
