#include "llm/hf_tokenizer.hpp"

#include <unicode/normalizer2.h>
#include <unicode/unistr.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace llm {
namespace {

[[noreturn]] void fail() { throw std::runtime_error("tokenizer"); }

#include "unicode_cats.inc"

bool in_ranges(uint32_t cp, const CpRange* rows, size_t n) {
  size_t lo = 0;
  size_t hi = n;
  while (lo < hi) {
    const size_t mid = (lo + hi) / 2;
    if (cp < rows[mid].lo) hi = mid;
    else if (cp > rows[mid].hi) lo = mid + 1;
    else return true;
  }
  return false;
}

bool is_letter(uint32_t cp) {
  return in_ranges(cp, kLetterRanges, sizeof(kLetterRanges) / sizeof(kLetterRanges[0]));
}

bool is_number(uint32_t cp) {
  return in_ranges(cp, kNumberRanges, sizeof(kNumberRanges) / sizeof(kNumberRanges[0]));
}

bool is_mark(uint32_t cp) {
  return in_ranges(cp, kMarkRanges, sizeof(kMarkRanges) / sizeof(kMarkRanges[0]));
}

// Unicode White_Space, which is what the tokenizers regex calls \s.
bool is_ws(uint32_t cp) {
  switch (cp) {
    case 0x09:
    case 0x0A:
    case 0x0B:
    case 0x0C:
    case 0x0D:
    case 0x20:
    case 0x85:
    case 0xA0:
    case 0x1680:
    case 0x2028:
    case 0x2029:
    case 0x202F:
    case 0x205F:
    case 0x3000:
      return true;
    default:
      return cp >= 0x2000 && cp <= 0x200A;
  }
}

bool is_other(uint32_t cp) { return !is_ws(cp) && !is_letter(cp) && !is_number(cp); }

// Qwen 3.5 punctuation excludes combining marks. A mark is is_other, so the
// GPT-2 and Qwen2 punctuation class would swallow it.
bool is_punct_qwen35(uint32_t cp) {
  return !is_ws(cp) && !is_letter(cp) && !is_mark(cp) && !is_number(cp);
}

bool is_word(uint32_t cp) { return cp == '_' || is_letter(cp) || is_number(cp); }

bool decode_utf8(std::string_view text, size_t i, uint32_t& cp, size_t& len) {
  if (i >= text.size()) return false;
  const auto c0 = static_cast<unsigned char>(text[i]);
  if (c0 < 0x80) {
    cp = c0;
    len = 1;
    return true;
  }
  if ((c0 & 0xE0) == 0xC0 && i + 1 < text.size()) {
    const auto c1 = static_cast<unsigned char>(text[i + 1]);
    if ((c1 & 0xC0) != 0x80) return false;
    cp = (static_cast<uint32_t>(c0 & 0x1F) << 6) | (c1 & 0x3F);
    if (cp < 0x80) return false;
    len = 2;
    return true;
  }
  if ((c0 & 0xF0) == 0xE0 && i + 2 < text.size()) {
    const auto c1 = static_cast<unsigned char>(text[i + 1]);
    const auto c2 = static_cast<unsigned char>(text[i + 2]);
    if ((c1 & 0xC0) != 0x80 || (c2 & 0xC0) != 0x80) return false;
    cp = (static_cast<uint32_t>(c0 & 0x0F) << 12) | (static_cast<uint32_t>(c1 & 0x3F) << 6) | (c2 & 0x3F);
    if (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF)) return false;
    len = 3;
    return true;
  }
  if ((c0 & 0xF8) == 0xF0 && i + 3 < text.size()) {
    const auto c1 = static_cast<unsigned char>(text[i + 1]);
    const auto c2 = static_cast<unsigned char>(text[i + 2]);
    const auto c3 = static_cast<unsigned char>(text[i + 3]);
    if ((c1 & 0xC0) != 0x80 || (c2 & 0xC0) != 0x80 || (c3 & 0xC0) != 0x80) return false;
    cp = (static_cast<uint32_t>(c0 & 0x07) << 18) | (static_cast<uint32_t>(c1 & 0x3F) << 12) |
         (static_cast<uint32_t>(c2 & 0x3F) << 6) | (c3 & 0x3F);
    if (cp < 0x10000 || cp > 0x10FFFF) return false;
    len = 4;
    return true;
  }
  return false;
}

void append_utf8(std::string& out, uint32_t cp) {
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

void append_utf8_lossy(std::string& out, std::string_view bytes) {
  size_t i = 0;
  while (i < bytes.size()) {
    uint32_t cp = 0;
    size_t len = 0;
    if (decode_utf8(bytes, i, cp, len)) {
      append_utf8(out, cp);
      i += len;
    } else {
      append_utf8(out, 0xFFFD);
      ++i;
    }
  }
}

struct Unit {
  uint32_t cp = 0;
  uint32_t begin = 0;
  uint32_t end = 0;
};

std::vector<Unit> units_of(std::string_view text) {
  if (text.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) fail();
  std::vector<Unit> units;
  size_t i = 0;
  while (i < text.size()) {
    uint32_t cp = 0;
    size_t len = 0;
    if (!decode_utf8(text, i, cp, len)) {
      cp = static_cast<unsigned char>(text[i]);
      len = 1;
    }
    units.push_back(Unit{cp, static_cast<uint32_t>(i), static_cast<uint32_t>(i + len)});
    i += len;
  }
  return units;
}

// GPT-2, Llama 3, Qwen2/Qwen3, and Qwen 3.5. The file's regex string selects one.
// Alternation is first-match, and \s+(?!\S) keeps every space but the last
// when a non-space follows. Qwen 3.5 letter runs also consume combining marks.
constexpr char kGpt2[] =
    R"re('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)re";
constexpr char kLlama3[] =
    R"re((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)re";
constexpr char kQwen[] =
    R"re((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)re";
constexpr char kQwen35[] =
    R"re((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)re";

enum Kind { kKindGpt2 = 0, kKindLlama = 1, kKindQwen = 2, kKindQwen35 = 3 };

std::optional<size_t> contraction(const std::vector<Unit>& u, size_t i, bool fold) {
  if (i >= u.size() || u[i].cp != '\'') return std::nullopt;
  auto eq = [&](size_t k, char ch) {
    if (i + k >= u.size()) return false;
    uint32_t cp = u[i + k].cp;
    if (fold && cp >= 'A' && cp <= 'Z') cp = cp - 'A' + 'a';
    return cp == static_cast<uint32_t>(ch);
  };
  if (eq(1, 's') || eq(1, 't') || eq(1, 'm') || eq(1, 'd')) return i + 2;
  if ((eq(1, 'r') && eq(2, 'e')) || (eq(1, 'v') && eq(2, 'e')) || (eq(1, 'l') && eq(2, 'l'))) return i + 3;
  return std::nullopt;
}

std::optional<size_t> run_with_space(const std::vector<Unit>& u, size_t i, bool (*pred)(uint32_t)) {
  if (i >= u.size()) return std::nullopt;
  size_t start = i;
  if (u[i].cp == ' ') {
    if (i + 1 >= u.size() || !pred(u[i + 1].cp)) return std::nullopt;
    start = i + 1;
  } else if (!pred(u[i].cp)) {
    return std::nullopt;
  }
  size_t j = start + 1;
  while (j < u.size() && pred(u[j].cp)) ++j;
  return j;
}

std::optional<size_t> prefixed_letters(const std::vector<Unit>& u, size_t i) {
  if (i >= u.size()) return std::nullopt;
  size_t start = i;
  const uint32_t cp = u[i].cp;
  if (cp != '\r' && cp != '\n' && !is_letter(cp) && !is_number(cp)) {
    if (i + 1 < u.size() && is_letter(u[i + 1].cp)) start = i + 1;
    else return std::nullopt;
  }
  if (start >= u.size() || !is_letter(u[start].cp)) return std::nullopt;
  size_t j = start + 1;
  while (j < u.size() && is_letter(u[j].cp)) ++j;
  return j;
}

// [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+  The current scalar is always consumed when
// the run starts. Following letters and marks stay in the same piece.
std::optional<size_t> prefixed_letters_marks(const std::vector<Unit>& u, size_t i) {
  if (i >= u.size()) return std::nullopt;
  const uint32_t cp = u[i].cp;
  if (cp == '\r' || cp == '\n' || is_number(cp)) return std::nullopt;
  const bool cur = is_letter(cp) || is_mark(cp);
  const bool next = i + 1 < u.size() && (is_letter(u[i + 1].cp) || is_mark(u[i + 1].cp));
  if (!cur && !next) return std::nullopt;
  size_t j = i + 1;
  while (j < u.size() && (is_letter(u[j].cp) || is_mark(u[j].cp))) ++j;
  return j;
}

std::optional<size_t> numbers(const std::vector<Unit>& u, size_t i, size_t cap) {
  if (i >= u.size() || !is_number(u[i].cp)) return std::nullopt;
  size_t j = i + 1;
  while (j < u.size() && (j - i) < cap && is_number(u[j].cp)) ++j;
  return j;
}

std::optional<size_t> punct_then_newline(const std::vector<Unit>& u, size_t i, bool (*pred)(uint32_t)) {
  if (i >= u.size()) return std::nullopt;
  size_t k = i;
  if (u[k].cp == ' ') ++k;
  if (k >= u.size() || !pred(u[k].cp)) return std::nullopt;
  ++k;
  while (k < u.size() && pred(u[k].cp)) ++k;
  while (k < u.size() && (u[k].cp == '\r' || u[k].cp == '\n')) ++k;
  return k;
}

std::optional<size_t> whitespace_through_newline(const std::vector<Unit>& u, size_t i) {
  if (i >= u.size() || !is_ws(u[i].cp)) return std::nullopt;
  size_t j = i;
  size_t last_nl = static_cast<size_t>(-1);
  while (j < u.size() && is_ws(u[j].cp)) {
    if (u[j].cp == '\r' || u[j].cp == '\n') last_nl = j;
    ++j;
  }
  if (last_nl == static_cast<size_t>(-1)) return std::nullopt;
  return last_nl + 1;
}

std::optional<size_t> whitespace_not_before_graph(const std::vector<Unit>& u, size_t i) {
  if (i >= u.size() || !is_ws(u[i].cp)) return std::nullopt;
  size_t j = i;
  while (j < u.size() && is_ws(u[j].cp)) ++j;
  if (j == u.size()) return j;
  if (j - i >= 2) return j - 1;
  return std::nullopt;
}

std::optional<size_t> whitespace_run(const std::vector<Unit>& u, size_t i) {
  if (i >= u.size() || !is_ws(u[i].cp)) return std::nullopt;
  size_t j = i;
  while (j < u.size() && is_ws(u[j].cp)) ++j;
  return j;
}

size_t next_piece(int kind, const std::vector<Unit>& u, size_t i) {
  std::optional<size_t> end;
  if (kind == kKindGpt2) {
    end = contraction(u, i, false);
    if (!end) end = run_with_space(u, i, is_letter);
    if (!end) end = run_with_space(u, i, is_number);
    if (!end) end = run_with_space(u, i, is_other);
  } else {
    end = contraction(u, i, true);
    if (!end) end = kind == kKindQwen35 ? prefixed_letters_marks(u, i) : prefixed_letters(u, i);
    if (!end) end = numbers(u, i, kind == kKindLlama ? 3 : 1);
    if (!end) end = punct_then_newline(u, i, kind == kKindQwen35 ? is_punct_qwen35 : is_other);
    if (!end) end = whitespace_through_newline(u, i);
  }
  if (!end) end = whitespace_not_before_graph(u, i);
  if (!end) end = whitespace_run(u, i);
  if (!end || *end <= i) fail();
  return *end;
}

struct ByteMap {
  uint32_t to_cp[256] = {};
  int16_t to_byte[324];

  ByteMap() {
    for (int16_t& slot : to_byte) slot = -1;
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
    std::vector<int> cs = bs;
    bool seen[256] = {};
    for (int b : bs) seen[b] = true;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
      if (seen[b]) continue;
      bs.push_back(b);
      cs.push_back(256 + n);
      ++n;
    }
    for (size_t i = 0; i < bs.size(); ++i) {
      to_cp[bs[i]] = static_cast<uint32_t>(cs[i]);
      to_byte[cs[i]] = static_cast<int16_t>(bs[i]);
    }
  }
};

const ByteMap& byte_map() {
  static const ByteMap map;
  return map;
}

std::string byte_symbol(unsigned char byte) {
  std::string out;
  append_utf8(out, byte_map().to_cp[byte]);
  return out;
}

struct Json {
  enum class Type { Null, Bool, Number, String, Array, Object };
  Type type = Type::Null;
  bool b = false;
  std::string text;
  std::vector<Json> arr;
  std::vector<std::pair<std::string, Json>> obj;
};

class JsonParser {
 public:
  explicit JsonParser(std::string_view in) : in_(in) {}

  Json parse() {
    skip();
    Json value = parse_value(0);
    skip();
    if (i_ != in_.size()) fail();
    return value;
  }

 private:
  std::string_view in_;
  size_t i_ = 0;

  void skip() {
    while (i_ < in_.size()) {
      const char c = in_[i_];
      if (c != ' ' && c != '\t' && c != '\n' && c != '\r') break;
      ++i_;
    }
  }

  char peek() {
    if (i_ >= in_.size()) fail();
    return in_[i_];
  }

  Json parse_value(int depth) {
    if (depth > 64) fail();
    skip();
    const char c = peek();
    if (c == 'n') return literal("null", Json::Type::Null);
    if (c == 't') return literal("true", Json::Type::Bool, true);
    if (c == 'f') return literal("false", Json::Type::Bool, false);
    if (c == '"') return parse_string();
    if (c == '[') return parse_array(depth);
    if (c == '{') return parse_object(depth);
    if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
    fail();
  }

  Json literal(const char* word, Json::Type type, bool bit = false) {
    const size_t n = std::strlen(word);
    if (in_.substr(i_, n) != word) fail();
    i_ += n;
    Json value;
    value.type = type;
    value.b = bit;
    return value;
  }

  Json parse_string() {
    if (peek() != '"') fail();
    ++i_;
    std::string out;
    while (i_ < in_.size()) {
      const char c = in_[i_++];
      if (c == '"') {
        Json value;
        value.type = Json::Type::String;
        value.text = std::move(out);
        return value;
      }
      if (c == '\\') {
        if (i_ >= in_.size()) fail();
        const char e = in_[i_++];
        if (e == '"' || e == '\\' || e == '/') out.push_back(e);
        else if (e == 'b') out.push_back('\b');
        else if (e == 'f') out.push_back('\f');
        else if (e == 'n') out.push_back('\n');
        else if (e == 'r') out.push_back('\r');
        else if (e == 't') out.push_back('\t');
        else if (e == 'u') append_utf8(out, parse_u());
        else fail();
      } else if (static_cast<unsigned char>(c) < 0x20) {
        fail();
      } else {
        out.push_back(c);
      }
    }
    fail();
  }

  uint32_t parse_u() {
    const uint32_t cp = hex4();
    if (cp >= 0xDC00 && cp <= 0xDFFF) fail();
    if (cp < 0xD800 || cp > 0xDBFF) return cp;
    if (i_ + 6 > in_.size() || in_[i_] != '\\' || in_[i_ + 1] != 'u') fail();
    i_ += 2;
    const uint32_t lo = hex4();
    if (lo < 0xDC00 || lo > 0xDFFF) fail();
    return 0x10000 + (((cp - 0xD800) << 10) | (lo - 0xDC00));
  }

  uint32_t hex4() {
    if (i_ + 4 > in_.size()) fail();
    uint32_t v = 0;
    for (int k = 0; k < 4; ++k) {
      const char c = in_[i_++];
      v <<= 4;
      if (c >= '0' && c <= '9') v += static_cast<uint32_t>(c - '0');
      else if (c >= 'a' && c <= 'f') v += static_cast<uint32_t>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') v += static_cast<uint32_t>(c - 'A' + 10);
      else fail();
    }
    return v;
  }

  Json parse_number() {
    const size_t begin = i_;
    if (peek() == '-') ++i_;
    if (i_ >= in_.size() || in_[i_] < '0' || in_[i_] > '9') fail();
    if (in_[i_] == '0') ++i_;
    else while (i_ < in_.size() && in_[i_] >= '0' && in_[i_] <= '9') ++i_;
    if (i_ < in_.size() && in_[i_] == '.') {
      ++i_;
      if (i_ >= in_.size() || in_[i_] < '0' || in_[i_] > '9') fail();
      while (i_ < in_.size() && in_[i_] >= '0' && in_[i_] <= '9') ++i_;
    }
    if (i_ < in_.size() && (in_[i_] == 'e' || in_[i_] == 'E')) {
      ++i_;
      if (i_ < in_.size() && (in_[i_] == '+' || in_[i_] == '-')) ++i_;
      if (i_ >= in_.size() || in_[i_] < '0' || in_[i_] > '9') fail();
      while (i_ < in_.size() && in_[i_] >= '0' && in_[i_] <= '9') ++i_;
    }
    Json value;
    value.type = Json::Type::Number;
    value.text = std::string(in_.substr(begin, i_ - begin));
    return value;
  }

  Json parse_array(int depth) {
    ++i_;
    Json value;
    value.type = Json::Type::Array;
    skip();
    if (peek() == ']') {
      ++i_;
      return value;
    }
    for (;;) {
      value.arr.push_back(parse_value(depth + 1));
      skip();
      const char c = peek();
      if (c == ',') {
        ++i_;
        continue;
      }
      if (c == ']') {
        ++i_;
        return value;
      }
      fail();
    }
  }

  Json parse_object(int depth) {
    ++i_;
    Json value;
    value.type = Json::Type::Object;
    skip();
    if (peek() == '}') {
      ++i_;
      return value;
    }
    std::unordered_set<std::string> keys;
    for (;;) {
      skip();
      Json key = parse_string();
      skip();
      if (peek() != ':') fail();
      ++i_;
      if (!keys.insert(key.text).second) fail();
      value.obj.emplace_back(std::move(key.text), parse_value(depth + 1));
      skip();
      const char c = peek();
      if (c == ',') {
        ++i_;
        continue;
      }
      if (c == '}') {
        ++i_;
        return value;
      }
      fail();
    }
  }
};

const Json* find_key(const Json& obj, const char* key) {
  if (obj.type != Json::Type::Object) return nullptr;
  for (const auto& field : obj.obj) {
    if (field.first == key) return &field.second;
  }
  return nullptr;
}

const Json& need_key(const Json& obj, const char* key) {
  const Json* value = find_key(obj, key);
  if (!value) fail();
  return *value;
}

bool as_bool(const Json& value) {
  if (value.type != Json::Type::Bool) fail();
  return value.b;
}

int32_t as_i32(const Json& value) {
  if (value.type != Json::Type::Number) fail();
  if (value.text.empty() || value.text.size() > 10) fail();
  for (char c : value.text) {
    if (c < '0' || c > '9') fail();
  }
  std::int64_t n = 0;
  for (char c : value.text) n = n * 10 + (c - '0');
  if (n > static_cast<std::int64_t>(std::numeric_limits<int32_t>::max())) fail();
  return static_cast<int32_t>(n);
}

const std::string& as_string(const Json& value) {
  if (value.type != Json::Type::String) fail();
  return value.text;
}

std::string read_text(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) fail();
  std::string data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (data.size() >= 3 && static_cast<unsigned char>(data[0]) == 0xEF &&
      static_cast<unsigned char>(data[1]) == 0xBB && static_cast<unsigned char>(data[2]) == 0xBF) {
    data.erase(0, 3);
  }
  return data;
}

Json parse_json(const std::filesystem::path& path) {
  return JsonParser(read_text(path)).parse();
}

bool flag_or(const Json& obj, const char* key, bool fallback) {
  const Json* value = find_key(obj, key);
  if (!value || value->type == Json::Type::Null) return fallback;
  return as_bool(*value);
}

std::optional<std::string> token_name(const Json* value) {
  if (!value || value->type == Json::Type::Null) return std::nullopt;
  if (value->type == Json::Type::String) return value->text;
  if (value->type == Json::Type::Object) return as_string(need_key(*value, "content"));
  fail();
}

void require_null_or_false(const Json& obj, const char* key) {
  const Json* value = find_key(obj, key);
  if (!value || value->type == Json::Type::Null) return;
  if (value->type == Json::Type::Bool && !value->b) return;
  fail();
}

void require_null_or_empty(const Json& obj, const char* key) {
  const Json* value = find_key(obj, key);
  if (!value || value->type == Json::Type::Null) return;
  if (value->type == Json::Type::String && value->text.empty()) return;
  fail();
}

void require_null_or_zero(const Json& obj, const char* key) {
  const Json* value = find_key(obj, key);
  if (!value || value->type == Json::Type::Null) return;
  if (value->type == Json::Type::Number && value->text == "0") return;
  fail();
}

int kind_from_regex(const std::string& pattern) {
  if (pattern == kGpt2) return kKindGpt2;
  if (pattern == kLlama3) return kKindLlama;
  if (pattern == kQwen) return kKindQwen;
  if (pattern == kQwen35) return kKindQwen35;
  fail();
}

bool byte_level_prefix(const Json& node, bool allow_regex) {
  if (as_string(need_key(node, "type")) != "ByteLevel") fail();
  const Json* use_regex = find_key(node, "use_regex");
  const bool regex = !use_regex || use_regex->type == Json::Type::Null ? true : as_bool(*use_regex);
  if (regex != allow_regex) fail();
  return flag_or(node, "add_prefix_space", false);
}

}  // namespace

std::unique_ptr<HfTokenizer> HfTokenizer::load(const std::filesystem::path& dir) {
  if (!std::filesystem::is_directory(dir)) fail();
  const Json root = parse_json(dir / "tokenizer.json");
  if (root.type != Json::Type::Object) fail();

  auto tok = std::unique_ptr<HfTokenizer>(new HfTokenizer());

  const Json* normalizer = find_key(root, "normalizer");
  if (normalizer && normalizer->type != Json::Type::Null) {
    if (as_string(need_key(*normalizer, "type")) != "NFC") fail();
    tok->normalize_nfc_ = true;
  }

  const Json& model = need_key(root, "model");
  if (as_string(need_key(model, "type")) != "BPE") fail();
  require_null_or_zero(model, "dropout");
  require_null_or_empty(model, "continuing_subword_prefix");
  require_null_or_empty(model, "end_of_word_suffix");
  require_null_or_false(model, "byte_fallback");
  const Json* ignore = find_key(model, "ignore_merges");
  if (ignore && ignore->type != Json::Type::Null) tok->ignore_merges_ = as_bool(*ignore);

  auto note_id = [&](const std::string& text, int32_t id, bool override_piece) {
    if (id < 0) fail();
    if (static_cast<size_t>(id) >= tok->id_to_token_.size()) {
      tok->id_to_token_.resize(static_cast<size_t>(id) + 1);
    }
    if (!tok->id_to_token_[static_cast<size_t>(id)].empty() &&
        tok->id_to_token_[static_cast<size_t>(id)] != text && !override_piece) {
      fail();
    }
    tok->id_to_token_[static_cast<size_t>(id)] = text;
    tok->vocab_[text] = id;
    if (id >= tok->vocab_size_) tok->vocab_size_ = id + 1;
  };

  const Json& vocab = need_key(model, "vocab");
  if (vocab.type != Json::Type::Object || vocab.obj.empty()) fail();
  for (const auto& field : vocab.obj) note_id(field.first, as_i32(field.second), false);

  const Json* added = find_key(root, "added_tokens");
  if (added && added->type != Json::Type::Null) {
    if (added->type != Json::Type::Array) fail();
    for (const Json& item : added->arr) {
      Added row;
      row.id = as_i32(need_key(item, "id"));
      row.content = as_string(need_key(item, "content"));
      if (row.content.empty()) fail();
      row.lstrip = flag_or(item, "lstrip", false);
      row.rstrip = flag_or(item, "rstrip", false);
      row.single_word = flag_or(item, "single_word", false);
      note_id(row.content, row.id, true);
      tok->added_.push_back(std::move(row));
    }
  }
  std::stable_sort(tok->added_.begin(), tok->added_.end(), [](const Added& a, const Added& b) {
    return a.content.size() > b.content.size();
  });

  const Json& merges = need_key(model, "merges");
  if (merges.type != Json::Type::Array) fail();
  for (size_t rank = 0; rank < merges.arr.size(); ++rank) {
    const Json& row = merges.arr[rank];
    std::string left;
    std::string right;
    if (row.type == Json::Type::String) {
      const size_t cut = row.text.find(' ');
      if (cut == std::string::npos || cut == 0 || cut + 1 >= row.text.size()) fail();
      left = row.text.substr(0, cut);
      right = row.text.substr(cut + 1);
    } else if (row.type == Json::Type::Array && row.arr.size() == 2) {
      left = as_string(row.arr[0]);
      right = as_string(row.arr[1]);
    } else {
      fail();
    }
    if (left.empty() || right.empty()) fail();
    tok->merge_rank_.emplace(std::make_pair(left, right), static_cast<int>(rank));
  }

  const Json& pre = need_key(root, "pre_tokenizer");
  const std::string pre_type = as_string(need_key(pre, "type"));
  if (pre_type == "ByteLevel") {
    tok->add_prefix_space_ = byte_level_prefix(pre, true);
    tok->pretok_ = Pretok::Gpt2;
  } else if (pre_type == "Sequence") {
    const Json& parts = need_key(pre, "pretokenizers");
    if (parts.type != Json::Type::Array || parts.arr.size() != 2) fail();
    const Json& split = parts.arr[0];
    if (as_string(need_key(split, "type")) != "Split") fail();
    if (as_string(need_key(split, "behavior")) != "Isolated") fail();
    if (flag_or(split, "invert", false)) fail();
    const Json& pattern = need_key(split, "pattern");
    const int kind = kind_from_regex(as_string(need_key(pattern, "Regex")));
    tok->add_prefix_space_ = byte_level_prefix(parts.arr[1], false);
    tok->pretok_ = kind == kKindLlama    ? Pretok::Llama3
                   : kind == kKindQwen   ? Pretok::Qwen
                   : kind == kKindQwen35 ? Pretok::Qwen35
                                         : Pretok::Gpt2;
  } else {
    fail();
  }

  const Json& decoder = need_key(root, "decoder");
  const std::string decoder_type = as_string(need_key(decoder, "type"));
  if (decoder_type == "ByteLevel") {
    tok->strip_prefix_space_ = tok->add_prefix_space_ && flag_or(decoder, "add_prefix_space", false);
  } else if (decoder_type == "Sequence") {
    const Json& parts = need_key(decoder, "decoders");
    if (parts.type != Json::Type::Array || parts.arr.size() != 1) fail();
    if (as_string(need_key(parts.arr[0], "type")) != "ByteLevel") fail();
    tok->strip_prefix_space_ = tok->add_prefix_space_ && flag_or(parts.arr[0], "add_prefix_space", false);
  } else {
    fail();
  }

  const std::filesystem::path config_path = dir / "tokenizer_config.json";
  if (std::filesystem::is_regular_file(config_path)) {
    const Json config = parse_json(config_path);
    if (config.type != Json::Type::Object) fail();
    tok->add_bos_ = flag_or(config, "add_bos_token", false);
    tok->add_eos_ = flag_or(config, "add_eos_token", false);
    auto resolve = [&](const char* key) -> int32_t {
      const std::optional<std::string> name = token_name(find_key(config, key));
      if (!name) return -1;
      const auto it = tok->vocab_.find(*name);
      if (it == tok->vocab_.end()) fail();
      return it->second;
    };
    tok->bos_id_ = resolve("bos_token");
    tok->eos_id_ = resolve("eos_token");
    tok->pad_id_ = resolve("pad_token");
  }

  const std::filesystem::path gen_path = dir / "generation_config.json";
  if (std::filesystem::is_regular_file(gen_path)) {
    const Json gen = parse_json(gen_path);
    if (gen.type != Json::Type::Object) fail();
    auto fallback_id = [&](int32_t& id, const char* key) {
      const Json* value = find_key(gen, key);
      if (id == -1 && value && value->type != Json::Type::Null) id = as_i32(*value);
    };
    fallback_id(tok->bos_id_, "bos_token_id");
    fallback_id(tok->pad_id_, "pad_token_id");
    const Json* eos = find_key(gen, "eos_token_id");
    if (eos && eos->type != Json::Type::Null) {
      if (eos->type == Json::Type::Number) tok->stop_ids_.push_back(as_i32(*eos));
      else if (eos->type == Json::Type::Array) {
        for (const Json& item : eos->arr) tok->stop_ids_.push_back(as_i32(item));
      } else {
        fail();
      }
    }
  }
  if (tok->eos_id_ < 0 && !tok->stop_ids_.empty()) tok->eos_id_ = tok->stop_ids_.front();
  if ((tok->add_bos_ && !tok->has_token(tok->bos_id_)) ||
      (tok->add_eos_ && !tok->has_token(tok->eos_id_))) fail();
  return tok;
}

bool HfTokenizer::has_token(int32_t id) const {
  return id >= 0 && static_cast<size_t>(id) < id_to_token_.size() && !id_to_token_[id].empty();
}

int HfTokenizer::vocab_size() const { return vocab_size_; }
int32_t HfTokenizer::bos_id() const { return bos_id_; }
int32_t HfTokenizer::eos_id() const { return eos_id_; }
int32_t HfTokenizer::pad_id() const { return pad_id_; }
bool HfTokenizer::add_bos_default() const { return add_bos_; }
bool HfTokenizer::add_eos_default() const { return add_eos_; }
std::vector<int32_t> HfTokenizer::stop_ids() const {
  if (stop_ids_.empty() && eos_id_ >= 0) return {eos_id_};
  return stop_ids_;
}

void HfTokenizer::save(const std::filesystem::path&) const { fail(); }

std::optional<HfTokenizer::Hit> HfTokenizer::match_special(std::string_view text, size_t at) const {
  for (const Added& tok : added_) {
    size_t content_at = at;
    if (tok.lstrip) {
      size_t j = at;
      while (j < text.size()) {
        uint32_t cp = 0;
        size_t len = 0;
        if (!decode_utf8(text, j, cp, len) || !is_ws(cp)) break;
        j += len;
      }
      content_at = j;
    }
    if (content_at + tok.content.size() > text.size()) continue;
    if (text.compare(content_at, tok.content.size(), tok.content) != 0) continue;
    size_t end = content_at + tok.content.size();
    if (tok.rstrip) {
      while (end < text.size()) {
        uint32_t cp = 0;
        size_t len = 0;
        if (!decode_utf8(text, end, cp, len) || !is_ws(cp)) break;
        end += len;
      }
    }
    if (tok.single_word) {
      auto cp_before = [&]() -> uint32_t {
        if (at == 0) return 0;
        size_t k = at;
        while (k > 0 && (static_cast<unsigned char>(text[k - 1]) & 0xC0) == 0x80) --k;
        if (k == 0 && (static_cast<unsigned char>(text[0]) & 0xC0) == 0x80) return static_cast<unsigned char>(text[at - 1]);
        uint32_t cp = 0;
        size_t len = 0;
        if (!decode_utf8(text, k == at ? at - 1 : k, cp, len)) return static_cast<unsigned char>(text[k]);
        return cp;
      };
      if (at > 0 && is_word(cp_before())) continue;
      uint32_t after = 0;
      size_t after_len = 0;
      if (end < text.size()) {
        if (!decode_utf8(text, end, after, after_len)) after = static_cast<unsigned char>(text[end]);
        if (is_word(after)) continue;
      }
    }
    return Hit{tok.id, end};
  }
  return std::nullopt;
}

std::vector<int32_t> HfTokenizer::bpe_piece(std::string_view piece) const {
  std::vector<std::string> symbols;
  symbols.reserve(piece.size());
  for (unsigned char byte : piece) symbols.push_back(byte_symbol(byte));
  if (!ignore_merges_) {
    while (symbols.size() >= 2) {
      int best = std::numeric_limits<int>::max();
      std::string left;
      std::string right;
      for (size_t i = 0; i + 1 < symbols.size(); ++i) {
        const auto it = merge_rank_.find(std::make_pair(symbols[i], symbols[i + 1]));
        if (it == merge_rank_.end() || it->second >= best) continue;
        best = it->second;
        left = symbols[i];
        right = symbols[i + 1];
      }
      if (left.empty()) break;
      std::vector<std::string> merged;
      merged.reserve(symbols.size() - 1);
      for (size_t i = 0; i < symbols.size();) {
        if (i + 1 < symbols.size() && symbols[i] == left && symbols[i + 1] == right) {
          merged.push_back(left + right);
          i += 2;
        } else {
          merged.push_back(std::move(symbols[i]));
          ++i;
        }
      }
      symbols = std::move(merged);
    }
  }
  std::vector<int32_t> ids;
  ids.reserve(symbols.size());
  for (const std::string& symbol : symbols) {
    const auto it = vocab_.find(symbol);
    if (it == vocab_.end()) fail();
    ids.push_back(it->second);
  }
  return ids;
}

void HfTokenizer::encode_gap(std::string_view gap, std::vector<int32_t>& ids) const {
  std::string normalized;
  if (normalize_nfc_ && !gap.empty()) {
    if (gap.size() > static_cast<size_t>(INT32_MAX)) fail();
    UErrorCode error = U_ZERO_ERROR;
    const icu::Normalizer2* nfc = icu::Normalizer2::getNFCInstance(error);
    if (U_FAILURE(error)) fail();
    const auto text = icu::UnicodeString::fromUTF8(
        icu::StringPiece(gap.data(), static_cast<int32_t>(gap.size())));
    nfc->normalize(text, error).toUTF8String(normalized);
    if (U_FAILURE(error)) fail();
    gap = normalized;
  }
  if (gap.empty()) return;
  std::string owned;
  std::string_view work = gap;
  if (add_prefix_space_ && gap.front() != ' ') {
    owned.push_back(' ');
    owned.append(gap);
    work = owned;
  }
  const int kind = pretok_ == Pretok::Gpt2     ? kKindGpt2
                   : pretok_ == Pretok::Llama3  ? kKindLlama
                   : pretok_ == Pretok::Qwen35  ? kKindQwen35
                                                : kKindQwen;
  const std::vector<Unit> units = units_of(work);
  size_t i = 0;
  while (i < units.size()) {
    const size_t end = next_piece(kind, units, i);
    const std::vector<int32_t> piece =
        bpe_piece(work.substr(units[i].begin, units[end - 1].end - units[i].begin));
    ids.insert(ids.end(), piece.begin(), piece.end());
    i = end;
  }
}

std::vector<int32_t> HfTokenizer::encode(std::string_view text, EncodeOptions opt) const {
  std::vector<int32_t> ids;
  if (opt.add_bos) {
    if (bos_id_ < 0) fail();
    ids.push_back(bos_id_);
  }
  if (added_.empty()) {
    encode_gap(text, ids);
  } else {
    size_t i = 0;
    while (i < text.size()) {
      if (const std::optional<Hit> hit = match_special(text, i)) {
        ids.push_back(hit->id);
        i = hit->end;
        continue;
      }
      size_t j = i + 1;
      while (j < text.size() && !match_special(text, j)) ++j;
      encode_gap(text.substr(i, j - i), ids);
      i = j;
    }
  }
  if (opt.add_eos) {
    if (eos_id_ < 0) fail();
    ids.push_back(eos_id_);
  }
  return ids;
}

std::string HfTokenizer::decode(const std::vector<int32_t>& ids) const {
  std::string bytes;
  std::string out;
  auto flush = [&]() {
    append_utf8_lossy(out, bytes);
    bytes.clear();
  };
  for (int32_t id : ids) {
    if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) throw std::invalid_argument("id");
    const std::string& token = id_to_token_[static_cast<size_t>(id)];
    if (token.empty()) throw std::invalid_argument("id");
    std::string piece;
    bool mapped = true;
    size_t i = 0;
    while (i < token.size()) {
      uint32_t cp = 0;
      size_t len = 0;
      if (!decode_utf8(token, i, cp, len) || cp >= 324 || byte_map().to_byte[cp] < 0) {
        mapped = false;
        break;
      }
      piece.push_back(static_cast<char>(byte_map().to_byte[cp]));
      i += len;
    }
    if (!mapped || i != token.size()) {
      flush();
      out += token;
    } else {
      bytes += piece;
    }
  }
  flush();
  if (strip_prefix_space_ && !out.empty() && out.front() == ' ') out.erase(out.begin());
  return out;
}

}  // namespace llm
