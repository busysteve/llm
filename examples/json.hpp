#pragma once

#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace llm_example {

struct Json {
  enum class Kind { Null, Bool, Number, String, Array, Object };
  Kind kind = Kind::Null;
  bool b = false;
  double n = 0;
  std::string s;
  std::vector<Json> a;
  std::map<std::string, Json> o;

  const Json* find(const std::string& key) const {
    if (kind != Kind::Object) return nullptr;
    const auto it = o.find(key);
    return it == o.end() ? nullptr : &it->second;
  }
};

class JsonParser {
 public:
  explicit JsonParser(const std::string& text) : text_(text) {}

  Json parse() {
    Json value = parse_value();
    skip();
    if (i_ != text_.size()) throw std::runtime_error("json");
    return value;
  }

 private:
  const std::string& text_;
  size_t i_ = 0;

  static bool ws(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }
  static bool digit(char c) { return c >= '0' && c <= '9'; }

  void skip() {
    while (i_ < text_.size() && ws(text_[i_])) ++i_;
  }

  char peek() {
    skip();
    if (i_ >= text_.size()) throw std::runtime_error("json");
    return text_[i_];
  }

  Json parse_value(unsigned depth = 0) {
    if (depth > 64) throw std::runtime_error("json");
    const char c = peek();
    if (c == 'n') return literal("null", Json{});
    if (c == 't') {
      Json value;
      value.kind = Json::Kind::Bool;
      value.b = true;
      return literal("true", value);
    }
    if (c == 'f') {
      Json value;
      value.kind = Json::Kind::Bool;
      value.b = false;
      return literal("false", value);
    }
    if (c == '"') return parse_string();
    if (c == '[') return parse_array(depth);
    if (c == '{') return parse_object(depth);
    if (c == '-' || digit(c)) return parse_number();
    throw std::runtime_error("json");
  }

  Json literal(const char* word, Json value) {
    for (const char* p = word; *p; ++p) {
      if (i_ >= text_.size() || text_[i_] != *p) throw std::runtime_error("json");
      ++i_;
    }
    return value;
  }

  Json parse_string() {
    if (peek() != '"') throw std::runtime_error("json");
    ++i_;
    Json value;
    value.kind = Json::Kind::String;
    while (i_ < text_.size() && text_[i_] != '"') {
      unsigned char c = static_cast<unsigned char>(text_[i_++]);
      if (c == '\\') {
        if (i_ >= text_.size()) throw std::runtime_error("json");
        const char e = text_[i_++];
        if (e == '"' || e == '\\' || e == '/') value.s.push_back(e);
        else if (e == 'b') value.s.push_back('\b');
        else if (e == 'f') value.s.push_back('\f');
        else if (e == 'n') value.s.push_back('\n');
        else if (e == 'r') value.s.push_back('\r');
        else if (e == 't') value.s.push_back('\t');
        else if (e == 'u') {
          uint32_t code = hex4();
          if (code >= 0xd800 && code <= 0xdbff) {
            if (text_.compare(i_, 2, "\\u") != 0) throw std::runtime_error("json");
            i_ += 2;
            const uint32_t low = hex4();
            if (low < 0xdc00 || low > 0xdfff) throw std::runtime_error("json");
            code = 0x10000 + ((code - 0xd800) << 10) + (low - 0xdc00);
          } else if (code >= 0xdc00 && code <= 0xdfff) {
            throw std::runtime_error("json");
          }
          value.s += utf8(code);
        } else {
          throw std::runtime_error("json");
        }
      } else if (c < 0x20) {
        throw std::runtime_error("json");
      } else {
        value.s.push_back(static_cast<char>(c));
      }
    }
    if (i_ >= text_.size() || text_[i_] != '"') throw std::runtime_error("json");
    ++i_;
    return value;
  }

  uint32_t hex4() {
    uint32_t code = 0;
    for (int n = 0; n < 4; ++n) {
      if (i_ >= text_.size()) throw std::runtime_error("json");
      const char c = text_[i_++];
      code <<= 4;
      if (c >= '0' && c <= '9') code |= static_cast<uint32_t>(c - '0');
      else if (c >= 'a' && c <= 'f') code |= static_cast<uint32_t>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') code |= static_cast<uint32_t>(c - 'A' + 10);
      else throw std::runtime_error("json");
    }
    return code;
  }

  static std::string utf8(uint32_t code) {
    std::string out;
    if (code < 0x80) {
      out.push_back(static_cast<char>(code));
    } else if (code < 0x800) {
      out.push_back(static_cast<char>(0xc0 | (code >> 6)));
      out.push_back(static_cast<char>(0x80 | (code & 0x3f)));
    } else if (code < 0x10000) {
      out.push_back(static_cast<char>(0xe0 | (code >> 12)));
      out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3f)));
      out.push_back(static_cast<char>(0x80 | (code & 0x3f)));
    } else {
      out.push_back(static_cast<char>(0xf0 | (code >> 18)));
      out.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3f)));
      out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3f)));
      out.push_back(static_cast<char>(0x80 | (code & 0x3f)));
    }
    return out;
  }

  Json parse_number() {
    const size_t start = i_;
    if (text_[i_] == '-') ++i_;
    if (i_ >= text_.size() || !digit(text_[i_])) throw std::runtime_error("json");
    while (i_ < text_.size() && digit(text_[i_])) ++i_;
    if (i_ < text_.size() && text_[i_] == '.') {
      ++i_;
      if (i_ >= text_.size() || !digit(text_[i_])) throw std::runtime_error("json");
      while (i_ < text_.size() && digit(text_[i_])) ++i_;
    }
    if (i_ < text_.size() && (text_[i_] == 'e' || text_[i_] == 'E')) {
      ++i_;
      if (i_ < text_.size() && (text_[i_] == '+' || text_[i_] == '-')) ++i_;
      if (i_ >= text_.size() || !digit(text_[i_])) throw std::runtime_error("json");
      while (i_ < text_.size() && digit(text_[i_])) ++i_;
    }
    Json value;
    value.kind = Json::Kind::Number;
    try {
      value.n = std::stod(text_.substr(start, i_ - start));
    } catch (const std::exception&) {
      throw std::runtime_error("json");
    }
    if (!std::isfinite(value.n)) throw std::runtime_error("json");
    return value;
  }

  Json parse_array(unsigned depth) {
    if (peek() != '[') throw std::runtime_error("json");
    ++i_;
    Json value;
    value.kind = Json::Kind::Array;
    skip();
    if (i_ < text_.size() && text_[i_] == ']') {
      ++i_;
      return value;
    }
    while (true) {
      value.a.push_back(parse_value(depth + 1));
      const char c = peek();
      if (c == ',') {
        ++i_;
        continue;
      }
      if (c == ']') {
        ++i_;
        return value;
      }
      throw std::runtime_error("json");
    }
  }

  Json parse_object(unsigned depth) {
    if (peek() != '{') throw std::runtime_error("json");
    ++i_;
    Json value;
    value.kind = Json::Kind::Object;
    skip();
    if (i_ < text_.size() && text_[i_] == '}') {
      ++i_;
      return value;
    }
    while (true) {
      Json key = parse_string();
      if (peek() != ':') throw std::runtime_error("json");
      ++i_;
      Json child = parse_value(depth + 1);
      if (!value.o.emplace(std::move(key.s), std::move(child)).second) throw std::runtime_error("json");
      const char c = peek();
      if (c == ',') {
        ++i_;
        continue;
      }
      if (c == '}') {
        ++i_;
        return value;
      }
      throw std::runtime_error("json");
    }
  }
};

inline Json parse_json(const std::string& text) { return JsonParser(text).parse(); }

}  // namespace llm_example
