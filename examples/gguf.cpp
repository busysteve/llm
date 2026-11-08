#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t kCheckpointMagic = 0x314E4E43u;
constexpr uint32_t kGgufMagic = 0x46554747u;
constexpr uint32_t kGgufVersion = 3u;
constexpr uint32_t kGgmlF32 = 0u;
constexpr uint32_t kGgmlF16 = 1u;
constexpr uint32_t kDefaultAlign = 32u;
constexpr char kArchitecture[] = "nnllm";

enum Meta : uint32_t {
  kU8 = 0,
  kI8 = 1,
  kU16 = 2,
  kI16 = 3,
  kU32 = 4,
  kI32 = 5,
  kF32 = 6,
  kBool = 7,
  kString = 8,
  kArray = 9,
  kU64 = 10,
  kI64 = 11,
  kF64 = 12
};

struct Config {
  int vocab = 0;
  int d_model = 0;
  int heads = 0;
  int layers = 0;
  int d_ff = 0;
  int max_seq = 0;
  float dropout = 0.f;
  bool bias = false;
  bool swiglu = true;
};

struct Tensor {
  std::string name;
  std::vector<int> shape;
  std::vector<float> data;
};

struct Model {
  Config config;
  std::vector<Tensor> tensors;
};

uint16_t f32_to_f16(float value);

void usage() {
  std::fputs(
      "usage: gguf export MODEL.BIN MODEL.GGUF\n"
      "       gguf import MODEL.GGUF MODEL.BIN\n"
      "       gguf self-test\n"
      "\n"
      "export writes a GGUF version 3 file, architecture nnllm, float32.\n"
      "Tensor bytes stay in this library's row-major order. The GGUF dimension\n"
      "list stores the contiguous dimension first. import reads that file back,\n"
      "including float16 tensors, and writes a version-1 checkpoint with no\n"
      "optimizer. AdamW moments are not stored.\n"
      "\n"
      "tokenizer.bpe is not part of either file. A GGUF whose architecture is\n"
      "not nnllm, or whose tensors are quantized, is rejected.\n",
      stderr);
}

class Bytes {
 public:
  void u8(uint8_t value) { bytes_.push_back(value); }
  void u16(uint16_t value) {
    u8(static_cast<uint8_t>(value));
    u8(static_cast<uint8_t>(value >> 8));
  }
  void u32(uint32_t value) {
    u16(static_cast<uint16_t>(value));
    u16(static_cast<uint16_t>(value >> 16));
  }
  void u64(uint64_t value) {
    u32(static_cast<uint32_t>(value));
    u32(static_cast<uint32_t>(value >> 32));
  }
  void i32(int32_t value) { u32(static_cast<uint32_t>(value)); }
  void f32(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    u32(bits);
  }
  void pad8() {
    while (bytes_.size() % 8 != 0) bytes_.push_back(0);
  }
  void raw(const uint8_t* data, size_t n) { bytes_.insert(bytes_.end(), data, data + n); }
  void zeros(size_t n) { bytes_.insert(bytes_.end(), n, 0); }
  const std::vector<uint8_t>& bytes() const { return bytes_; }

 private:
  std::vector<uint8_t> bytes_;
};

class Reader {
 public:
  explicit Reader(std::vector<uint8_t> bytes) : bytes_(std::move(bytes)) {}
  size_t left() const { return bytes_.size() - pos_; }
  size_t pos() const { return pos_; }
  uint8_t u8() {
    need(1);
    return bytes_[pos_++];
  }
  uint16_t u16() {
    const uint16_t value = u8() | (static_cast<uint16_t>(u8()) << 8);
    return value;
  }
  uint32_t u32() {
    const uint32_t value = u16() | (static_cast<uint32_t>(u16()) << 16);
    return value;
  }
  uint64_t u64() {
    const uint64_t low = u32();
    return low | (static_cast<uint64_t>(u32()) << 32);
  }
  int32_t i32() { return static_cast<int32_t>(u32()); }
  float f32() {
    const uint32_t bits = u32();
    float value = 0.f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }
  void pad8() {
    while (pos_ % 8 != 0) {
      need(1);
      ++pos_;
    }
  }
  std::string str_u32() {
    const uint32_t n = u32();
    if (n > 256) throw std::runtime_error("name");
    return take(n);
  }
  std::string str_u64() {
    const uint64_t n = u64();
    if (n > 65535) throw std::runtime_error("name");
    return take(static_cast<size_t>(n));
  }
  void skip(size_t n) {
    need(n);
    pos_ += n;
  }
  const uint8_t* at(size_t offset, size_t n) const {
    if (offset > bytes_.size() || n > bytes_.size() - offset) throw std::runtime_error("truncated");
    return bytes_.data() + offset;
  }
  size_t size() const { return bytes_.size(); }

 private:
  void need(size_t n) const {
    if (pos_ + n > bytes_.size()) throw std::runtime_error("truncated");
  }
  std::string take(size_t n) {
    need(n);
    std::string value(reinterpret_cast<const char*>(bytes_.data() + pos_), n);
    pos_ += n;
    return value;
  }
  std::vector<uint8_t> bytes_;
  size_t pos_ = 0;
};

std::vector<uint8_t> read_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("read");
  in.seekg(0, std::ios::end);
  const std::streamoff n = in.tellg();
  if (n < 0) throw std::runtime_error("read");
  in.seekg(0);
  std::vector<uint8_t> bytes(static_cast<size_t>(n));
  if (n > 0) {
    in.read(reinterpret_cast<char*>(bytes.data()), n);
    if (!in) throw std::runtime_error("read");
  }
  return bytes;
}

void write_file(const std::filesystem::path& path, const std::vector<uint8_t>& bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("write");
  if (!bytes.empty()) {
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  }
  if (!out) throw std::runtime_error("write");
}

int64_t product(const std::vector<int>& shape) {
  int64_t n = 1;
  for (int dim : shape) {
    if (dim < 1) return -1;
    if (n > INT64_C(0x7fffffffffffffff) / dim) return -1;
    n *= dim;
  }
  return n;
}

std::vector<Tensor> expected_tensors(const Config& config) {
  const int d = config.d_model;
  const int fc1 = config.swiglu ? config.d_ff * 2 : config.d_ff;
  std::vector<Tensor> tensors;
  auto add = [&](std::string name, std::vector<int> shape) {
    tensors.push_back(Tensor{std::move(name), std::move(shape), {}});
  };
  auto linear = [&](const std::string& name, int out, int in) {
    add(name + ".weight", {out, in});
    if (config.bias) add(name + ".bias", {out});
  };
  auto norm = [&](const std::string& name) {
    add(name + ".weight", {d});
    add(name + ".bias", {d});
  };
  add("tok_emb.weight", {config.vocab, d});
  add("pos_emb.weight", {config.max_seq, d});
  for (int layer = 0; layer < config.layers; ++layer) {
    const std::string prefix = "blocks." + std::to_string(layer) + ".";
    norm(prefix + "ln1");
    linear(prefix + "attn.q", d, d);
    linear(prefix + "attn.k", d, d);
    linear(prefix + "attn.v", d, d);
    linear(prefix + "attn.out", d, d);
    norm(prefix + "ln2");
    linear(prefix + "fc1", fc1, d);
    linear(prefix + "fc2", d, config.d_ff);
  }
  norm("final_ln");
  linear("lm_head", config.vocab, d);
  return tensors;
}

void require_config(const Config& config) {
  if (config.vocab < 1 || config.d_model < 1 || config.heads < 1 || config.layers < 1 ||
      config.d_ff < 1 || config.max_seq < 1) {
    throw std::runtime_error("config");
  }
  if (config.d_model % config.heads != 0) throw std::runtime_error("num_heads");
  if (!(config.dropout >= 0.f) || !(config.dropout < 1.f)) throw std::runtime_error("dropout");
  if (config.swiglu && config.d_ff > 0x7fffffff / 2) throw std::runtime_error("d_ff");
}

Model read_checkpoint(const std::filesystem::path& path) {
  Reader reader(read_file(path));
  if (reader.u32() != kCheckpointMagic) throw std::runtime_error("magic");
  const uint32_t version = reader.u32();
  if (version < 1 || version > 2) throw std::runtime_error("version");
  if (reader.u32() != 6) throw std::runtime_error("family");
  const uint32_t training = reader.u32();
  reader.u64();
  const uint32_t optimizer = reader.u32();
  if (reader.u32() != 0 || training > 1 || optimizer > 1) throw std::runtime_error("header");

  Config config;
  config.vocab = reader.i32();
  config.d_model = reader.i32();
  config.heads = reader.i32();
  config.layers = reader.i32();
  config.d_ff = reader.i32();
  config.max_seq = reader.i32();
  config.dropout = reader.f32();
  if (reader.u32() != 0) throw std::runtime_error("tie_embeddings");
  config.bias = reader.u32() != 0;
  const uint32_t activation = reader.u32();
  const uint32_t positional = reader.u32();
  reader.pad8();
  if (activation == 1) config.swiglu = false;
  else if (activation == 2) config.swiglu = true;
  else throw std::runtime_error("activation");
  if (positional != 1) throw std::runtime_error("positional");
  require_config(config);

  const uint32_t count = reader.u32();
  if (reader.u32() != 0 || count > 1000000u) throw std::runtime_error("truncated");
  std::map<std::string, Tensor> got;
  for (uint32_t i = 0; i < count; ++i) {
    Tensor tensor;
    tensor.name = reader.str_u32();
    reader.pad8();
    const uint32_t rank = reader.u32();
    if (rank != 1 && rank != 2) throw std::runtime_error("rank");
    tensor.shape.resize(rank);
    for (uint32_t d = 0; d < rank; ++d) tensor.shape[d] = reader.i32();
    reader.pad8();
    if (reader.u32() != 1) throw std::runtime_error("dtype");
    const uint32_t numel = reader.u32();
    if (product(tensor.shape) != static_cast<int64_t>(numel)) throw std::runtime_error("tensor " + tensor.name);
    tensor.data.resize(numel);
    for (uint32_t n = 0; n < numel; ++n) tensor.data[n] = reader.f32();
    reader.pad8();
    if (!got.emplace(tensor.name, std::move(tensor)).second) throw std::runtime_error("tensor");
  }

  Model model;
  model.config = config;
  for (Tensor& slot : expected_tensors(config)) {
    const auto found = got.find(slot.name);
    if (found == got.end() || found->second.shape != slot.shape) {
      throw std::runtime_error("tensor " + slot.name);
    }
    slot.data = std::move(found->second.data);
    got.erase(found);
    model.tensors.push_back(std::move(slot));
  }
  if (!got.empty()) throw std::runtime_error("tensor " + got.begin()->first);
  return model;
}

void write_checkpoint(const std::filesystem::path& path, const Model& model) {
  require_config(model.config);
  Bytes out;
  out.u32(kCheckpointMagic);
  out.u32(1);
  out.u32(6);
  out.u32(0);
  out.u64(0);
  out.u32(0);
  out.u32(0);
  const Config& config = model.config;
  out.i32(config.vocab);
  out.i32(config.d_model);
  out.i32(config.heads);
  out.i32(config.layers);
  out.i32(config.d_ff);
  out.i32(config.max_seq);
  out.f32(config.dropout);
  out.u32(0);
  out.u32(config.bias ? 1u : 0u);
  out.u32(config.swiglu ? 2u : 1u);
  out.u32(1);
  out.pad8();
  out.u32(static_cast<uint32_t>(model.tensors.size()));
  out.u32(0);
  for (const Tensor& tensor : model.tensors) {
    if (tensor.name.size() > 256) throw std::runtime_error("name");
    out.u32(static_cast<uint32_t>(tensor.name.size()));
    out.raw(reinterpret_cast<const uint8_t*>(tensor.name.data()), tensor.name.size());
    out.pad8();
    out.u32(static_cast<uint32_t>(tensor.shape.size()));
    for (int dim : tensor.shape) out.i32(dim);
    out.pad8();
    const int64_t numel = product(tensor.shape);
    if (numel < 0 || static_cast<size_t>(numel) != tensor.data.size()) {
      throw std::runtime_error("tensor " + tensor.name);
    }
    out.u32(1);
    out.u32(static_cast<uint32_t>(numel));
    for (float value : tensor.data) out.f32(value);
    out.pad8();
  }
  out.u32(0);
  out.u32(0);
  write_file(path, out.bytes());
}

void gguf_string(Bytes& out, const std::string& value) {
  out.u64(value.size());
  out.raw(reinterpret_cast<const uint8_t*>(value.data()), value.size());
}

void gguf_kv_string(Bytes& out, const char* key, const std::string& value) {
  gguf_string(out, key);
  out.u32(kString);
  gguf_string(out, value);
}

void gguf_kv_u32(Bytes& out, const char* key, uint32_t value) {
  gguf_string(out, key);
  out.u32(kU32);
  out.u32(value);
}

void gguf_kv_f32(Bytes& out, const char* key, float value) {
  gguf_string(out, key);
  out.u32(kF32);
  out.f32(value);
}

void gguf_kv_bool(Bytes& out, const char* key, bool value) {
  gguf_string(out, key);
  out.u32(kBool);
  out.u8(value ? 1u : 0u);
}

uint64_t align_up(uint64_t value, uint64_t alignment) {
  const uint64_t mask = alignment - 1;
  return (value + mask) & ~mask;
}

void write_gguf(const std::filesystem::path& path, const Model& model, bool half = false) {
  require_config(model.config);
  for (const Tensor& tensor : model.tensors) {
    if (tensor.name.size() > 64) throw std::runtime_error("name");
  }
  const Config& config = model.config;
  const uint32_t dtype = half ? kGgmlF16 : kGgmlF32;
  const uint64_t width = half ? 2u : 4u;
  Bytes header;
  header.u32(kGgufMagic);
  header.u32(kGgufVersion);
  header.u64(model.tensors.size());
  header.u64(16);
  gguf_kv_string(header, "general.architecture", kArchitecture);
  gguf_kv_string(header, "general.name", kArchitecture);
  gguf_kv_u32(header, "general.alignment", kDefaultAlign);
  gguf_kv_u32(header, "general.file_type", half ? 1u : 0u);
  gguf_kv_string(header, "general.description",
                 "nn llm checkpoint weights. learned absolute positions, layer norm.");
  gguf_kv_u32(header, "nnllm.vocab_size", static_cast<uint32_t>(config.vocab));
  gguf_kv_u32(header, "nnllm.embedding_length", static_cast<uint32_t>(config.d_model));
  gguf_kv_u32(header, "nnllm.block_count", static_cast<uint32_t>(config.layers));
  gguf_kv_u32(header, "nnllm.feed_forward_length", static_cast<uint32_t>(config.d_ff));
  gguf_kv_u32(header, "nnllm.context_length", static_cast<uint32_t>(config.max_seq));
  gguf_kv_u32(header, "nnllm.attention.head_count", static_cast<uint32_t>(config.heads));
  gguf_kv_f32(header, "nnllm.dropout", config.dropout);
  gguf_kv_bool(header, "nnllm.bias", config.bias);
  gguf_kv_bool(header, "nnllm.tie_embeddings", false);
  gguf_kv_string(header, "nnllm.activation", config.swiglu ? "swiglu" : "gelu");
  gguf_kv_string(header, "nnllm.positional", "learned_absolute");

  uint64_t data_offset = 0;
  for (const Tensor& tensor : model.tensors) {
    data_offset = align_up(data_offset, kDefaultAlign);
    gguf_string(header, tensor.name);
    header.u32(static_cast<uint32_t>(tensor.shape.size()));
    for (auto it = tensor.shape.rbegin(); it != tensor.shape.rend(); ++it) {
      header.u64(static_cast<uint64_t>(*it));
    }
    header.u32(dtype);
    header.u64(data_offset);
    const int64_t numel = product(tensor.shape);
    data_offset += static_cast<uint64_t>(numel) * width;
  }
  const size_t info_end = header.bytes().size();
  const size_t data_begin = static_cast<size_t>(align_up(info_end, kDefaultAlign));
  header.zeros(data_begin - info_end);

  data_offset = 0;
  for (const Tensor& tensor : model.tensors) {
    const uint64_t aligned = align_up(data_offset, kDefaultAlign);
    header.zeros(static_cast<size_t>(aligned - data_offset));
    for (float value : tensor.data) {
      if (half) header.u16(f32_to_f16(value));
      else header.f32(value);
    }
    data_offset = aligned + static_cast<uint64_t>(tensor.data.size()) * width;
  }
  write_file(path, header.bytes());
}

void skip_meta(Reader& reader, uint32_t type, int depth) {
  if (depth > 8) throw std::runtime_error("metadata");
  switch (type) {
    case kU8:
    case kI8:
    case kBool:
      reader.skip(1);
      return;
    case kU16:
    case kI16:
      reader.skip(2);
      return;
    case kU32:
    case kI32:
    case kF32:
      reader.skip(4);
      return;
    case kU64:
    case kI64:
    case kF64:
      reader.skip(8);
      return;
    case kString: {
      const uint64_t n = reader.u64();
      if (n > 65535) throw std::runtime_error("metadata");
      reader.skip(static_cast<size_t>(n));
      return;
    }
    case kArray: {
      const uint32_t elem = reader.u32();
      const uint64_t n = reader.u64();
      if (n > 1000000u) throw std::runtime_error("metadata");
      for (uint64_t i = 0; i < n; ++i) skip_meta(reader, elem, depth + 1);
      return;
    }
    default:
      throw std::runtime_error("metadata");
  }
}

uint32_t meta_u32(Reader& reader, uint32_t type) {
  if (type == kU32) return reader.u32();
  if (type == kU64) {
    const uint64_t value = reader.u64();
    if (value > 0xffffffffu) throw std::runtime_error("metadata");
    return static_cast<uint32_t>(value);
  }
  if (type == kI32) {
    const int32_t value = reader.i32();
    if (value < 0) throw std::runtime_error("metadata");
    return static_cast<uint32_t>(value);
  }
  throw std::runtime_error("metadata");
}

float f16_to_f32(uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
  const uint32_t exp = (bits >> 10) & 0x1fu;
  const uint32_t mant = bits & 0x3ffu;
  uint32_t out = 0;
  if (exp == 0) {
    if (mant == 0) {
      out = sign;
    } else {
      uint32_t value = mant;
      uint32_t exp32 = 127 - 15 + 1;
      while ((value & 0x400u) == 0) {
        value <<= 1;
        --exp32;
      }
      value &= 0x3ffu;
      out = sign | (exp32 << 23) | (value << 13);
    }
  } else if (exp == 31) {
    out = sign | 0x7f800000u | (mant << 13);
  } else {
    out = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
  }
  float value = 0.f;
  std::memcpy(&value, &out, sizeof(value));
  return value;
}

struct MetaField {
  bool seen = false;
  uint32_t type = 0;
  size_t pos = 0;
};

Model read_gguf(const std::filesystem::path& path) {
  Reader reader(read_file(path));
  if (reader.size() < 24 || reader.u32() != kGgufMagic) throw std::runtime_error("magic");
  if (reader.u32() != kGgufVersion) throw std::runtime_error("version");
  const uint64_t tensor_count = reader.u64();
  const uint64_t kv_count = reader.u64();
  if (tensor_count > 1000000u || kv_count > 100000u) throw std::runtime_error("truncated");

  std::map<std::string, MetaField> fields;
  uint32_t alignment = kDefaultAlign;
  bool saw_alignment = false;
  for (uint64_t i = 0; i < kv_count; ++i) {
    const std::string key = reader.str_u64();
    const uint32_t type = reader.u32();
    if (key == "general.alignment") {
      alignment = meta_u32(reader, type);
      saw_alignment = true;
      continue;
    }
    MetaField field;
    field.seen = true;
    field.type = type;
    field.pos = reader.pos();
    fields[key] = field;
    skip_meta(reader, type, 0);
  }
  if (saw_alignment) {
    if (alignment < 8 || alignment > 4096 || (alignment & (alignment - 1)) != 0) {
      throw std::runtime_error("alignment");
    }
  }

  auto string_at = [&](const char* key) {
    const auto found = fields.find(key);
    if (found == fields.end() || found->second.type != kString) throw std::runtime_error(key);
    const uint8_t* bytes = reader.at(found->second.pos, 8);
    uint64_t n = 0;
    for (int i = 0; i < 8; ++i) n |= static_cast<uint64_t>(bytes[i]) << (8 * i);
    if (n > 65535) throw std::runtime_error(key);
    const uint8_t* text = reader.at(found->second.pos + 8, static_cast<size_t>(n));
    return std::string(reinterpret_cast<const char*>(text), static_cast<size_t>(n));
  };
  auto u32_at = [&](const char* key) {
    const auto found = fields.find(key);
    if (found == fields.end()) throw std::runtime_error(key);
    if (found->second.type == kU32 || found->second.type == kI32) {
      const uint8_t* bytes = reader.at(found->second.pos, 4);
      uint32_t value = 0;
      for (int i = 0; i < 4; ++i) value |= static_cast<uint32_t>(bytes[i]) << (8 * i);
      if (found->second.type == kI32 && static_cast<int32_t>(value) < 0) throw std::runtime_error(key);
      return value;
    }
    if (found->second.type == kU64) {
      const uint8_t* bytes = reader.at(found->second.pos, 8);
      uint64_t value = 0;
      for (int i = 0; i < 8; ++i) value |= static_cast<uint64_t>(bytes[i]) << (8 * i);
      if (value > 0xffffffffu) throw std::runtime_error(key);
      return static_cast<uint32_t>(value);
    }
    throw std::runtime_error(key);
  };
  auto f32_at = [&](const char* key) {
    const auto found = fields.find(key);
    if (found == fields.end() || found->second.type != kF32) throw std::runtime_error(key);
    const uint8_t* bytes = reader.at(found->second.pos, 4);
    uint32_t bits = 0;
    for (int i = 0; i < 4; ++i) bits |= static_cast<uint32_t>(bytes[i]) << (8 * i);
    float value = 0.f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  };
  auto bool_at = [&](const char* key) {
    const auto found = fields.find(key);
    if (found == fields.end() || found->second.type != kBool) throw std::runtime_error(key);
    const uint8_t value = *reader.at(found->second.pos, 1);
    if (value > 1) throw std::runtime_error(key);
    return value != 0;
  };

  if (string_at("general.architecture") != kArchitecture) throw std::runtime_error("architecture");
  Config config;
  config.vocab = static_cast<int>(u32_at("nnllm.vocab_size"));
  config.d_model = static_cast<int>(u32_at("nnllm.embedding_length"));
  config.layers = static_cast<int>(u32_at("nnllm.block_count"));
  config.d_ff = static_cast<int>(u32_at("nnllm.feed_forward_length"));
  config.max_seq = static_cast<int>(u32_at("nnllm.context_length"));
  config.heads = static_cast<int>(u32_at("nnllm.attention.head_count"));
  config.dropout = f32_at("nnllm.dropout");
  if (bool_at("nnllm.tie_embeddings")) throw std::runtime_error("tie_embeddings");
  config.bias = bool_at("nnllm.bias");
  const std::string activation = string_at("nnllm.activation");
  if (activation == "swiglu") config.swiglu = true;
  else if (activation == "gelu") config.swiglu = false;
  else throw std::runtime_error("activation");
  if (string_at("nnllm.positional") != "learned_absolute") throw std::runtime_error("positional");
  require_config(config);

  struct Stored {
    std::string name;
    std::vector<int> shape;
    uint32_t dtype = 0;
    uint64_t offset = 0;
    int64_t numel = 0;
  };
  std::vector<Stored> stored;
  stored.reserve(static_cast<size_t>(tensor_count));
  for (uint64_t i = 0; i < tensor_count; ++i) {
    Stored tensor;
    tensor.name = reader.str_u64();
    if (tensor.name.size() > 64) throw std::runtime_error("name");
    const uint32_t rank = reader.u32();
    if (rank < 1 || rank > 4) throw std::runtime_error("rank");
    std::vector<uint64_t> dims(rank);
    for (uint32_t d = 0; d < rank; ++d) dims[d] = reader.u64();
    tensor.shape.resize(rank);
    for (uint32_t d = 0; d < rank; ++d) {
      const uint64_t dim = dims[rank - 1 - d];
      if (dim < 1 || dim > static_cast<uint64_t>(0x7fffffff)) throw std::runtime_error("tensor " + tensor.name);
      tensor.shape[d] = static_cast<int>(dim);
    }
    tensor.dtype = reader.u32();
    tensor.offset = reader.u64();
    if (tensor.offset % alignment != 0) throw std::runtime_error("alignment");
    tensor.numel = product(tensor.shape);
    if (tensor.numel < 0) throw std::runtime_error("tensor " + tensor.name);
    stored.push_back(std::move(tensor));
  }
  const size_t data_begin = static_cast<size_t>(align_up(reader.pos(), alignment));
  std::map<std::string, Tensor> got;
  for (const Stored& tensor : stored) {
    if (tensor.dtype != kGgmlF32 && tensor.dtype != kGgmlF16) throw std::runtime_error("quantized");
    const uint64_t width = tensor.dtype == kGgmlF32 ? 4u : 2u;
    const uint64_t nbytes = static_cast<uint64_t>(tensor.numel) * width;
    if (data_begin > reader.size() || tensor.offset > reader.size() - data_begin ||
        nbytes > reader.size() - data_begin - tensor.offset) {
      throw std::runtime_error("truncated");
    }
    const uint8_t* raw =
        reader.at(data_begin + static_cast<size_t>(tensor.offset), static_cast<size_t>(nbytes));
    Tensor out;
    out.name = tensor.name;
    out.shape = tensor.shape;
    out.data.resize(static_cast<size_t>(tensor.numel));
    if (tensor.dtype == kGgmlF32) {
      for (int64_t n = 0; n < tensor.numel; ++n) {
        uint32_t bits = 0;
        for (int b = 0; b < 4; ++b) {
          bits |= static_cast<uint32_t>(raw[static_cast<size_t>(n) * 4 + b]) << (8 * b);
        }
        std::memcpy(&out.data[static_cast<size_t>(n)], &bits, sizeof(bits));
      }
    } else {
      for (int64_t n = 0; n < tensor.numel; ++n) {
        const uint16_t bits = static_cast<uint16_t>(raw[static_cast<size_t>(n) * 2]) |
                              (static_cast<uint16_t>(raw[static_cast<size_t>(n) * 2 + 1]) << 8);
        out.data[static_cast<size_t>(n)] = f16_to_f32(bits);
      }
    }
    if (!got.emplace(out.name, std::move(out)).second) throw std::runtime_error("tensor");
  }

  Model model;
  model.config = config;
  for (Tensor& slot : expected_tensors(config)) {
    const auto found = got.find(slot.name);
    if (found == got.end() || found->second.shape != slot.shape) {
      throw std::runtime_error("tensor " + slot.name);
    }
    slot.data = std::move(found->second.data);
    got.erase(found);
    model.tensors.push_back(std::move(slot));
  }
  if (!got.empty()) throw std::runtime_error("tensor " + got.begin()->first);
  return model;
}

uint16_t f32_to_f16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u;
  int32_t exp = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
  uint32_t mant = bits & 0x7fffffu;
  if (((bits >> 23) & 0xffu) == 0xffu) {
    return static_cast<uint16_t>(sign | 0x7c00u | (mant ? 0x200u : 0));
  }
  if (exp <= 0) {
    if (exp < -10) return static_cast<uint16_t>(sign);
    mant |= 0x800000u;
    const uint32_t shift = static_cast<uint32_t>(1 - exp);
    uint32_t half = mant >> (shift + 13);
    const uint32_t rest = mant & ((1u << (shift + 13)) - 1u);
    if (rest > (1u << (shift + 12)) || (rest == (1u << (shift + 12)) && (half & 1u))) ++half;
    return static_cast<uint16_t>(sign | half);
  }
  if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
  uint32_t half = static_cast<uint32_t>(exp << 10) | (mant >> 13);
  if ((mant & 0x1fffu) > 0x1000u || ((mant & 0x1fffu) == 0x1000u && (half & 1u))) ++half;
  return static_cast<uint16_t>(sign | half);
}

void write_bad_architecture(const std::filesystem::path& path) {
  Bytes out;
  out.u32(kGgufMagic);
  out.u32(kGgufVersion);
  out.u64(0);
  out.u64(1);
  gguf_kv_string(out, "general.architecture", "llama");
  const size_t info_end = out.bytes().size();
  out.zeros(static_cast<size_t>(align_up(info_end, kDefaultAlign) - info_end));
  write_file(path, out.bytes());
}

int self_test() {
  Config config;
  config.vocab = 32;
  config.d_model = 8;
  config.heads = 2;
  config.layers = 1;
  config.d_ff = 12;
  config.max_seq = 16;
  config.dropout = 0.f;
  config.bias = false;
  config.swiglu = true;
  Model model;
  model.config = config;
  model.tensors = expected_tensors(config);
  uint32_t n = 1;
  for (Tensor& tensor : model.tensors) {
    const int64_t numel = product(tensor.shape);
    tensor.data.resize(static_cast<size_t>(numel));
    for (int64_t i = 0; i < numel; ++i) {
      const uint32_t bits = 0x3f800000u + n + static_cast<uint32_t>(i);
      std::memcpy(&tensor.data[static_cast<size_t>(i)], &bits, sizeof(bits));
    }
    ++n;
  }
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "gguf-self-test";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const auto bin = dir / "model.bin";
  const auto gguf = dir / "model.gguf";
  const auto back = dir / "back.bin";
  write_checkpoint(bin, model);
  const Model loaded = read_checkpoint(bin);
  if (loaded.tensors.size() != model.tensors.size()) throw std::runtime_error("self-test load");
  write_gguf(gguf, loaded);
  const Model from_gguf = read_gguf(gguf);
  write_checkpoint(back, from_gguf);
  const Model round = read_checkpoint(back);
  if (round.tensors.size() != model.tensors.size()) throw std::runtime_error("self-test round");
  for (size_t i = 0; i < model.tensors.size(); ++i) {
    if (round.tensors[i].name != model.tensors[i].name || round.tensors[i].shape != model.tensors[i].shape ||
        round.tensors[i].data != model.tensors[i].data) {
      std::string detail = "self-test round " + model.tensors[i].name + " got";
      for (int dim : round.tensors[i].shape) detail += " " + std::to_string(dim);
      detail += " want";
      for (int dim : model.tensors[i].shape) detail += " " + std::to_string(dim);
      detail += " n " + std::to_string(round.tensors[i].data.size());
      if (!round.tensors[i].data.empty() && !model.tensors[i].data.empty() &&
          round.tensors[i].data[0] != model.tensors[i].data[0]) {
        detail += " data";
      }
      throw std::runtime_error(detail);
    }
  }
  bool saw_tok = false;
  Reader check(read_file(gguf));
  check.u32();
  check.u32();
  check.u64();
  const uint64_t kv = check.u64();
  for (uint64_t i = 0; i < kv; ++i) {
    check.str_u64();
    skip_meta(check, check.u32(), 0);
  }
  const uint64_t ntensor = model.tensors.size();
  for (uint64_t i = 0; i < ntensor; ++i) {
    const std::string name = check.str_u64();
    const uint32_t rank = check.u32();
    std::vector<uint64_t> dims(rank);
    for (uint32_t d = 0; d < rank; ++d) dims[d] = check.u64();
    check.u32();
    check.u64();
    if (name == "tok_emb.weight") {
      if (rank != 2 || dims[0] != static_cast<uint64_t>(config.d_model) ||
          dims[1] != static_cast<uint64_t>(config.vocab)) {
        throw std::runtime_error("self-test dims");
      }
      saw_tok = true;
    }
  }
  if (!saw_tok) throw std::runtime_error("self-test tok");

  const std::vector<float> half_values = {0.f, 1.f, -2.f, 0.5f};
  for (Tensor& tensor : model.tensors) {
    for (size_t i = 0; i < tensor.data.size(); ++i) tensor.data[i] = half_values[i % half_values.size()];
  }
  write_gguf(dir / "half.gguf", model, true);
  const Model from_half = read_gguf(dir / "half.gguf");
  for (size_t t = 0; t < model.tensors.size(); ++t) {
    if (from_half.tensors[t].data.size() != model.tensors[t].data.size()) throw std::runtime_error("f16");
    for (size_t i = 0; i < model.tensors[t].data.size(); ++i) {
      if (from_half.tensors[t].data[i] != model.tensors[t].data[i]) throw std::runtime_error("f16");
    }
  }

  write_bad_architecture(dir / "llama.gguf");
  bool rejected = false;
  try {
    read_gguf(dir / "llama.gguf");
  } catch (const std::runtime_error& error) {
    rejected = std::string(error.what()) == "architecture";
  }
  if (!rejected) throw std::runtime_error("architecture");
  std::filesystem::remove_all(dir);
  std::printf("self-test ok\n");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
      usage();
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "self-test") return self_test();
    if (argc != 4) {
      usage();
      throw std::runtime_error("usage");
    }
    const std::string command = argv[1];
    if (command == "export") {
      write_gguf(argv[3], read_checkpoint(argv[2]));
      return 0;
    }
    if (command == "import") {
      write_checkpoint(argv[3], read_gguf(argv[2]));
      return 0;
    }
    usage();
    throw std::runtime_error(command);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "gguf: %s\n", error.what());
    return 1;
  }
}
