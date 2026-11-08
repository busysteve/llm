#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
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
  bool rope = false;
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
      "       gguf llama MODEL MODEL.GGUF [TOKENIZER.BPE]\n"
      "       gguf self-test\n"
      "\n"
      "export writes a GGUF version 3 file, architecture nnllm, float32.\n"
      "Tensor bytes stay in this library's row-major order. The GGUF dimension\n"
      "list stores the contiguous dimension first. import reads that file back,\n"
      "including float16 tensors, and writes a version-1 checkpoint with no\n"
      "optimizer. AdamW moments are not stored.\n"
      "\n"
      "llama reads a transformer checkpoint or an nnllm GGUF and writes a GGUF\n"
      "with architecture llama. SwiGLU fc1 is split into ffn_gate and ffn_up.\n"
      "A learned-position checkpoint omits the position table and LayerNorm\n"
      "biases. llama.cpp then applies RoPE and RMSNorm, so those tokens differ.\n"
      "A RoPE checkpoint already uses that block, and the file keeps every\n"
      "tensor. A tokenizer.bpe next to the checkpoint, or a path passed after\n"
      "the output, is written into the file so llama-cli can encode a prompt.\n"
      "GELU and linear-bias models are rejected.\n"
      "\n"
      "import rejects a GGUF whose architecture is not nnllm, and rejects\n"
      "quantized tensors.\n",
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
    if (!config.rope) add(name + ".bias", {d});
  };
  add("tok_emb.weight", {config.vocab, d});
  if (!config.rope) add("pos_emb.weight", {config.max_seq, d});
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
  if (config.rope && ((config.d_model / config.heads) % 2 != 0)) throw std::runtime_error("use_rope");
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
  if (positional == 2) config.rope = true;
  else if (positional != 1) throw std::runtime_error("positional");
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
  out.u32(config.rope ? 2u : 1u);
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

struct Kv {
  Bytes body;
  uint64_t count = 0;
  void str(const char* key, const std::string& value) {
    gguf_kv_string(body, key, value);
    ++count;
  }
  void u32(const char* key, uint32_t value) {
    gguf_kv_u32(body, key, value);
    ++count;
  }
  void f32(const char* key, float value) {
    gguf_kv_f32(body, key, value);
    ++count;
  }
  void boolean(const char* key, bool value) {
    gguf_kv_bool(body, key, value);
    ++count;
  }
  void strs(const char* key, const std::vector<std::string>& values) {
    gguf_string(body, key);
    body.u32(kArray);
    body.u32(kString);
    body.u64(values.size());
    for (const std::string& value : values) gguf_string(body, value);
    ++count;
  }
  void i32s(const char* key, const std::vector<int32_t>& values) {
    gguf_string(body, key);
    body.u32(kArray);
    body.u32(kI32);
    body.u64(values.size());
    for (int32_t value : values) body.i32(value);
    ++count;
  }
};

void append_gguf_tensors(Bytes& header, const std::vector<Tensor>& tensors, bool half) {
  const uint32_t dtype = half ? kGgmlF16 : kGgmlF32;
  const uint64_t width = half ? 2u : 4u;
  for (const Tensor& tensor : tensors) {
    if (tensor.name.size() > 64) throw std::runtime_error("name");
    const int64_t numel = product(tensor.shape);
    if (numel < 0 || static_cast<size_t>(numel) != tensor.data.size()) {
      throw std::runtime_error("tensor " + tensor.name);
    }
  }
  uint64_t data_offset = 0;
  for (const Tensor& tensor : tensors) {
    data_offset = align_up(data_offset, kDefaultAlign);
    gguf_string(header, tensor.name);
    header.u32(static_cast<uint32_t>(tensor.shape.size()));
    for (auto it = tensor.shape.rbegin(); it != tensor.shape.rend(); ++it) {
      header.u64(static_cast<uint64_t>(*it));
    }
    header.u32(dtype);
    header.u64(data_offset);
    data_offset += static_cast<uint64_t>(product(tensor.shape)) * width;
  }
  const size_t info_end = header.bytes().size();
  const size_t data_begin = static_cast<size_t>(align_up(info_end, kDefaultAlign));
  header.zeros(data_begin - info_end);

  data_offset = 0;
  for (const Tensor& tensor : tensors) {
    const uint64_t aligned = align_up(data_offset, kDefaultAlign);
    header.zeros(static_cast<size_t>(aligned - data_offset));
    for (float value : tensor.data) {
      if (half) header.u16(f32_to_f16(value));
      else header.f32(value);
    }
    data_offset = aligned + static_cast<uint64_t>(tensor.data.size()) * width;
  }
}

void write_gguf_container(const std::filesystem::path& path, const std::vector<Tensor>& tensors, const Kv& kv,
                          bool half) {
  Bytes header;
  header.u32(kGgufMagic);
  header.u32(kGgufVersion);
  header.u64(tensors.size());
  header.u64(kv.count);
  header.raw(kv.body.bytes().data(), kv.body.bytes().size());
  append_gguf_tensors(header, tensors, half);
  write_file(path, header.bytes());
}

void write_gguf(const std::filesystem::path& path, const Model& model, bool half = false) {
  require_config(model.config);
  const Config& config = model.config;
  Kv kv;
  kv.str("general.architecture", kArchitecture);
  kv.str("general.name", kArchitecture);
  kv.u32("general.alignment", kDefaultAlign);
  kv.u32("general.file_type", half ? 1u : 0u);
  kv.str("general.description", config.rope ? "nn llm checkpoint weights. rope positions, rms norm."
                                         : "nn llm checkpoint weights. learned absolute positions, layer norm.");
  kv.u32("nnllm.vocab_size", static_cast<uint32_t>(config.vocab));
  kv.u32("nnllm.embedding_length", static_cast<uint32_t>(config.d_model));
  kv.u32("nnllm.block_count", static_cast<uint32_t>(config.layers));
  kv.u32("nnllm.feed_forward_length", static_cast<uint32_t>(config.d_ff));
  kv.u32("nnllm.context_length", static_cast<uint32_t>(config.max_seq));
  kv.u32("nnllm.attention.head_count", static_cast<uint32_t>(config.heads));
  kv.f32("nnllm.dropout", config.dropout);
  kv.boolean("nnllm.bias", config.bias);
  kv.boolean("nnllm.tie_embeddings", false);
  kv.str("nnllm.activation", config.swiglu ? "swiglu" : "gelu");
  kv.str("nnllm.positional", config.rope ? "rope" : "learned_absolute");
  write_gguf_container(path, model.tensors, kv, half);
}

const Tensor& named_tensor(const Model& model, const std::string& name) {
  for (const Tensor& tensor : model.tensors) {
    if (tensor.name == name) return tensor;
  }
  throw std::runtime_error("tensor " + name);
}

Tensor copy_tensor(const Model& model, const std::string& name, std::vector<int> shape, std::string out_name) {
  const Tensor& src = named_tensor(model, name);
  if (src.shape != shape || static_cast<int64_t>(src.data.size()) != product(shape)) {
    throw std::runtime_error("tensor " + name);
  }
  Tensor out;
  out.name = std::move(out_name);
  out.shape = std::move(shape);
  out.data = src.data;
  return out;
}

Tensor split_rows(const Tensor& src, int row0, int rows, std::string out_name) {
  if (src.shape.size() != 2 || row0 < 0 || rows < 1 || row0 > src.shape[0] - rows) {
    throw std::runtime_error("tensor " + src.name);
  }
  const int in = src.shape[1];
  const size_t begin = static_cast<size_t>(row0) * static_cast<size_t>(in);
  const size_t count = static_cast<size_t>(rows) * static_cast<size_t>(in);
  if (begin > src.data.size() || count > src.data.size() - begin) throw std::runtime_error("tensor " + src.name);
  Tensor out;
  out.name = std::move(out_name);
  out.shape = {rows, in};
  out.data.assign(src.data.begin() + static_cast<std::ptrdiff_t>(begin),
                  src.data.begin() + static_cast<std::ptrdiff_t>(begin + count));
  return out;
}

std::vector<Tensor> llama_tensors(const Model& model) {
  const Config& config = model.config;
  const int d = config.d_model;
  std::vector<Tensor> tensors;
  tensors.push_back(copy_tensor(model, "tok_emb.weight", {config.vocab, d}, "token_embd.weight"));
  for (int layer = 0; layer < config.layers; ++layer) {
    const std::string src = "blocks." + std::to_string(layer) + ".";
    const std::string dst = "blk." + std::to_string(layer) + ".";
    tensors.push_back(copy_tensor(model, src + "ln1.weight", {d}, dst + "attn_norm.weight"));
    tensors.push_back(copy_tensor(model, src + "attn.q.weight", {d, d}, dst + "attn_q.weight"));
    tensors.push_back(copy_tensor(model, src + "attn.k.weight", {d, d}, dst + "attn_k.weight"));
    tensors.push_back(copy_tensor(model, src + "attn.v.weight", {d, d}, dst + "attn_v.weight"));
    tensors.push_back(copy_tensor(model, src + "attn.out.weight", {d, d}, dst + "attn_output.weight"));
    tensors.push_back(copy_tensor(model, src + "ln2.weight", {d}, dst + "ffn_norm.weight"));
    const Tensor& fc1 = named_tensor(model, src + "fc1.weight");
    if (fc1.shape != std::vector<int>{config.d_ff * 2, d} ||
        static_cast<int64_t>(fc1.data.size()) != product(fc1.shape)) {
      throw std::runtime_error("tensor " + fc1.name);
    }
    tensors.push_back(split_rows(fc1, 0, config.d_ff, dst + "ffn_gate.weight"));
    tensors.push_back(split_rows(fc1, config.d_ff, config.d_ff, dst + "ffn_up.weight"));
    tensors.push_back(copy_tensor(model, src + "fc2.weight", {d, config.d_ff}, dst + "ffn_down.weight"));
  }
  tensors.push_back(copy_tensor(model, "final_ln.weight", {d}, "output_norm.weight"));
  tensors.push_back(copy_tensor(model, "lm_head.weight", {config.vocab, d}, "output.weight"));
  return tensors;
}

std::string utf8_cpt(uint32_t cpt) {
  std::string out;
  if (cpt <= 0x7fu) {
    out.push_back(static_cast<char>(cpt));
  } else if (cpt <= 0x7ffu) {
    out.push_back(static_cast<char>(0xc0u | (cpt >> 6)));
    out.push_back(static_cast<char>(0x80u | (cpt & 0x3fu)));
  } else {
    throw std::runtime_error("tokenizer");
  }
  return out;
}

// llama.cpp GPT-2 bytes_to_unicode. Printable bytes stay themselves. The rest
// become code points at 256 and up, so every byte is a valid Unicode scalar.
std::array<std::string, 256> gpt2_byte_tokens() {
  std::array<std::string, 256> map;
  std::array<bool, 256> used{};
  auto put = [&](int byte) {
    map[static_cast<size_t>(byte)] = utf8_cpt(static_cast<uint32_t>(byte));
    used[static_cast<size_t>(byte)] = true;
  };
  for (int byte = 0x21; byte <= 0x7e; ++byte) put(byte);
  for (int byte = 0xa1; byte <= 0xac; ++byte) put(byte);
  for (int byte = 0xae; byte <= 0xff; ++byte) put(byte);
  uint32_t next = 0;
  for (int byte = 0; byte < 256; ++byte) {
    if (!used[static_cast<size_t>(byte)]) {
      map[static_cast<size_t>(byte)] = utf8_cpt(256u + next);
      ++next;
    }
  }
  return map;
}

struct Tokenizer {
  int vocab = 0;
  int pad = 0;
  int bos = 0;
  int eos = 0;
  std::vector<std::string> tokens;
  std::vector<std::string> merges;
  std::vector<int32_t> types;
};

Tokenizer load_bpe(const std::filesystem::path& path) {
  Reader reader(read_file(path));
  if (reader.size() < 24 || reader.u32() != 0x3142504cu) throw std::runtime_error("tokenizer");
  const uint32_t vocab = reader.u32();
  const uint32_t n_merges = reader.u32();
  const uint32_t pad = reader.u32();
  const uint32_t bos = reader.u32();
  const uint32_t eos = reader.u32();
  if (vocab < 259u || n_merges != vocab - 259u) throw std::runtime_error("tokenizer");
  if (pad != vocab - 3u || bos != vocab - 2u || eos != vocab - 1u) throw std::runtime_error("tokenizer");
  if (reader.left() != static_cast<size_t>(n_merges) * 8u) throw std::runtime_error("tokenizer");

  std::vector<std::pair<uint32_t, uint32_t>> pairs;
  pairs.reserve(n_merges);
  for (uint32_t i = 0; i < n_merges; ++i) {
    const uint32_t left = reader.u32();
    const uint32_t right = reader.u32();
    const uint32_t id = 256u + i;
    if (left >= id || right >= id) throw std::runtime_error("tokenizer");
    pairs.emplace_back(left, right);
  }

  const std::array<std::string, 256> byte_tok = gpt2_byte_tokens();
  std::vector<std::string> raw(256);
  for (int i = 0; i < 256; ++i) raw[static_cast<size_t>(i)] = std::string(1, static_cast<char>(i));
  for (const auto& pair : pairs) {
    raw.push_back(raw[pair.first] + raw[pair.second]);
  }

  Tokenizer tok;
  tok.vocab = static_cast<int>(vocab);
  tok.pad = static_cast<int>(pad);
  tok.bos = static_cast<int>(bos);
  tok.eos = static_cast<int>(eos);
  tok.tokens.reserve(vocab);
  std::set<std::string> seen;
  for (const std::string& piece : raw) {
    std::string encoded;
    encoded.reserve(piece.size() * 2);
    for (unsigned char byte : piece) encoded += byte_tok[byte];
    if (encoded.find(' ') != std::string::npos || !seen.insert(encoded).second) {
      throw std::runtime_error("tokenizer");
    }
    tok.tokens.push_back(std::move(encoded));
  }
  auto special = [&](std::initializer_list<const char*> names) {
    for (const char* name : names) {
      if (seen.insert(name).second) return std::string(name);
    }
    throw std::runtime_error("tokenizer");
  };
  // <|endoftext|> is on llama.cpp's end-of-generation list, so a prompt can stop.
  tok.tokens.push_back(special({"<pad>"}));
  tok.tokens.push_back(special({"<bos>"}));
  tok.tokens.push_back(special({"<|endoftext|>", "<eos>", "</s>"}));
  if (tok.tokens.size() != vocab) throw std::runtime_error("tokenizer");

  tok.merges.reserve(pairs.size());
  for (const auto& pair : pairs) {
    tok.merges.push_back(tok.tokens[pair.first] + " " + tok.tokens[pair.second]);
  }
  tok.types.assign(vocab, 1);
  tok.types[pad] = 3;
  tok.types[bos] = 3;
  tok.types[eos] = 3;
  return tok;
}

void write_llama_gguf(const std::filesystem::path& path, const Model& model, const Tokenizer* tokenizer = nullptr) {
  require_config(model.config);
  if (!model.config.swiglu) throw std::runtime_error("activation");
  if (model.config.bias) throw std::runtime_error("bias");
  if (tokenizer != nullptr && tokenizer->vocab != model.config.vocab) throw std::runtime_error("vocab");
  const std::vector<Tensor> tensors = llama_tensors(model);
  const Config& config = model.config;
  Kv kv;
  kv.str("general.architecture", "llama");
  kv.str("general.name", kArchitecture);
  kv.u32("general.alignment", kDefaultAlign);
  kv.u32("general.file_type", 0);
  kv.str("general.description",
         config.rope ? "Converted from an nnllm checkpoint trained with RoPE and RMSNorm. "
                       "llama.cpp runs the same block."
                     : "Converted from an nnllm checkpoint trained with learned absolute positions and LayerNorm. "
                       "llama.cpp applies RoPE and RMSNorm, so the tokens differ. "
                       "LayerNorm biases and the position table are omitted.");
  if (tokenizer == nullptr) {
    kv.str("tokenizer.ggml.model", "none");
  } else {
    kv.str("tokenizer.ggml.model", "gpt2");
    kv.str("tokenizer.ggml.pre", "superbpe");
    kv.boolean("tokenizer.ggml.add_bos_token", false);
    kv.boolean("tokenizer.ggml.add_eos_token", false);
    kv.boolean("tokenizer.ggml.add_space_prefix", false);
    kv.u32("tokenizer.ggml.bos_token_id", static_cast<uint32_t>(tokenizer->bos));
    kv.u32("tokenizer.ggml.eos_token_id", static_cast<uint32_t>(tokenizer->eos));
    kv.u32("tokenizer.ggml.padding_token_id", static_cast<uint32_t>(tokenizer->pad));
    kv.strs("tokenizer.ggml.tokens", tokenizer->tokens);
    kv.strs("tokenizer.ggml.merges", tokenizer->merges);
    kv.i32s("tokenizer.ggml.token_type", tokenizer->types);
  }
  kv.u32("llama.vocab_size", static_cast<uint32_t>(config.vocab));
  kv.u32("llama.context_length", static_cast<uint32_t>(config.max_seq));
  kv.u32("llama.embedding_length", static_cast<uint32_t>(config.d_model));
  kv.u32("llama.block_count", static_cast<uint32_t>(config.layers));
  kv.u32("llama.feed_forward_length", static_cast<uint32_t>(config.d_ff));
  kv.u32("llama.attention.head_count", static_cast<uint32_t>(config.heads));
  kv.u32("llama.attention.head_count_kv", static_cast<uint32_t>(config.heads));
  kv.f32("llama.attention.layer_norm_rms_epsilon", 1e-5f);
  kv.f32("llama.rope.freq_base", 10000.f);
  kv.u32("llama.rope.dimension_count", static_cast<uint32_t>(config.d_model / config.heads));
  write_gguf_container(path, tensors, kv, false);
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
  const std::string positional = string_at("nnllm.positional");
  if (positional == "rope") config.rope = true;
  else if (positional != "learned_absolute") throw std::runtime_error("positional");
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

Model read_model(const std::filesystem::path& path) {
  const std::vector<uint8_t> bytes = read_file(path);
  if (bytes.size() < 4) throw std::runtime_error("magic");
  const uint32_t magic = static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8) |
                         (static_cast<uint32_t>(bytes[2]) << 16) | (static_cast<uint32_t>(bytes[3]) << 24);
  if (magic == kCheckpointMagic) return read_checkpoint(path);
  if (magic == kGgufMagic) return read_gguf(path);
  throw std::runtime_error("magic");
}

struct ViewTensor {
  std::vector<uint64_t> dims;
  std::vector<float> data;
};

struct GgufView {
  std::map<std::string, std::string> strings;
  std::map<std::string, uint32_t> u32s;
  std::map<std::string, float> f32s;
  std::map<std::string, bool> bools;
  std::map<std::string, std::vector<std::string>> strs;
  std::map<std::string, std::vector<int32_t>> i32s;
  std::map<std::string, ViewTensor> tensors;
};

GgufView read_gguf_view(const std::filesystem::path& path) {
  Reader reader(read_file(path));
  if (reader.size() < 24 || reader.u32() != kGgufMagic) throw std::runtime_error("self-test llama magic");
  if (reader.u32() != kGgufVersion) throw std::runtime_error("self-test llama version");
  const uint64_t tensor_count = reader.u64();
  const uint64_t kv_count = reader.u64();
  GgufView view;
  for (uint64_t i = 0; i < kv_count; ++i) {
    const std::string key = reader.str_u64();
    const uint32_t type = reader.u32();
    if (type == kString) view.strings[key] = reader.str_u64();
    else if (type == kU32) view.u32s[key] = reader.u32();
    else if (type == kF32) view.f32s[key] = reader.f32();
    else if (type == kBool) view.bools[key] = reader.u8() != 0;
    else if (type == kArray) {
      const uint32_t elem = reader.u32();
      const uint64_t n = reader.u64();
      if (n > 1000000u) throw std::runtime_error("metadata");
      if (elem == kString) {
        std::vector<std::string> values;
        values.reserve(static_cast<size_t>(n));
        for (uint64_t j = 0; j < n; ++j) values.push_back(reader.str_u64());
        view.strs.emplace(key, std::move(values));
      } else if (elem == kI32) {
        std::vector<int32_t> values;
        values.reserve(static_cast<size_t>(n));
        for (uint64_t j = 0; j < n; ++j) values.push_back(reader.i32());
        view.i32s.emplace(key, std::move(values));
      } else {
        for (uint64_t j = 0; j < n; ++j) skip_meta(reader, elem, 1);
      }
    } else skip_meta(reader, type, 0);
  }
  struct Info {
    std::string name;
    std::vector<uint64_t> dims;
    uint32_t dtype = 0;
    uint64_t offset = 0;
  };
  std::vector<Info> infos;
  for (uint64_t i = 0; i < tensor_count; ++i) {
    Info info;
    info.name = reader.str_u64();
    const uint32_t rank = reader.u32();
    info.dims.resize(rank);
    for (uint32_t d = 0; d < rank; ++d) info.dims[d] = reader.u64();
    info.dtype = reader.u32();
    info.offset = reader.u64();
    infos.push_back(std::move(info));
  }
  const size_t data_begin = static_cast<size_t>(align_up(reader.pos(), kDefaultAlign));
  for (const Info& info : infos) {
    if (info.dtype != kGgmlF32) throw std::runtime_error("self-test llama dtype");
    int64_t numel = 1;
    for (uint64_t dim : info.dims) {
      if (dim < 1 || numel > INT64_C(0x7fffffffffffffff) / static_cast<int64_t>(dim)) {
        throw std::runtime_error("self-test llama " + info.name);
      }
      numel *= static_cast<int64_t>(dim);
    }
    const size_t nbytes = static_cast<size_t>(numel) * 4u;
    const uint8_t* raw = reader.at(data_begin + static_cast<size_t>(info.offset), nbytes);
    ViewTensor tensor;
    tensor.dims = info.dims;
    tensor.data.resize(static_cast<size_t>(numel));
    for (int64_t n = 0; n < numel; ++n) {
      uint32_t bits = 0;
      for (int b = 0; b < 4; ++b) {
        bits |= static_cast<uint32_t>(raw[static_cast<size_t>(n) * 4 + b]) << (8 * b);
      }
      std::memcpy(&tensor.data[static_cast<size_t>(n)], &bits, sizeof(bits));
    }
    if (!view.tensors.emplace(info.name, std::move(tensor)).second) {
      throw std::runtime_error("self-test llama " + info.name);
    }
  }
  return view;
}

void check_llama_file(const Model& model, const std::filesystem::path& path, bool rope_src = false) {
  const GgufView view = read_gguf_view(path);
  const auto architecture = view.strings.find("general.architecture");
  const auto tokenizer = view.strings.find("tokenizer.ggml.model");
  const auto description = view.strings.find("general.description");
  if (architecture == view.strings.end() || architecture->second != "llama") {
    throw std::runtime_error("self-test llama arch");
  }
  if (tokenizer == view.strings.end() || tokenizer->second != "none") {
    throw std::runtime_error("self-test llama vocab");
  }
  if (description == view.strings.end() || description->second.find("RoPE") == std::string::npos ||
      description->second.find("RMSNorm") == std::string::npos) {
    throw std::runtime_error("self-test llama description");
  }
  if (rope_src) {
    if (description->second.find("LayerNorm") != std::string::npos) {
      throw std::runtime_error("self-test llama description");
    }
  } else if (description->second.find("LayerNorm") == std::string::npos ||
             description->second.find("position") == std::string::npos) {
    throw std::runtime_error("self-test llama description");
  }
  const Config& config = model.config;
  auto need_u32 = [&](const char* key, uint32_t want) {
    const auto found = view.u32s.find(key);
    if (found == view.u32s.end() || found->second != want) throw std::runtime_error(std::string("self-test llama ") + key);
  };
  need_u32("llama.vocab_size", static_cast<uint32_t>(config.vocab));
  need_u32("llama.context_length", static_cast<uint32_t>(config.max_seq));
  need_u32("llama.embedding_length", static_cast<uint32_t>(config.d_model));
  need_u32("llama.block_count", static_cast<uint32_t>(config.layers));
  need_u32("llama.feed_forward_length", static_cast<uint32_t>(config.d_ff));
  need_u32("llama.attention.head_count", static_cast<uint32_t>(config.heads));
  need_u32("llama.attention.head_count_kv", static_cast<uint32_t>(config.heads));
  need_u32("llama.rope.dimension_count", static_cast<uint32_t>(config.d_model / config.heads));
  const auto eps = view.f32s.find("llama.attention.layer_norm_rms_epsilon");
  const auto base = view.f32s.find("llama.rope.freq_base");
  if (eps == view.f32s.end() || eps->second != 1e-5f || base == view.f32s.end() || base->second != 10000.f) {
    throw std::runtime_error("self-test llama rope");
  }
  if (view.tensors.size() != static_cast<size_t>(9 * config.layers + 3)) {
    throw std::runtime_error("self-test llama count");
  }
  for (const auto& item : view.tensors) {
    if (item.first.find("pos_emb") != std::string::npos || item.first.find("position_embd") != std::string::npos ||
        item.first.find(".bias") != std::string::npos) {
      throw std::runtime_error("self-test llama extra");
    }
  }
  auto same = [&](const std::string& llama_name, const std::string& src_name, std::vector<uint64_t> dims) {
    const auto found = view.tensors.find(llama_name);
    const Tensor& src = named_tensor(model, src_name);
    if (found == view.tensors.end() || found->second.dims != dims || found->second.data != src.data) {
      throw std::runtime_error("self-test llama " + llama_name);
    }
  };
  const uint64_t d = static_cast<uint64_t>(config.d_model);
  const uint64_t vocab = static_cast<uint64_t>(config.vocab);
  const uint64_t ff = static_cast<uint64_t>(config.d_ff);
  same("token_embd.weight", "tok_emb.weight", {d, vocab});
  same("output_norm.weight", "final_ln.weight", {d});
  same("output.weight", "lm_head.weight", {d, vocab});
  for (int layer = 0; layer < config.layers; ++layer) {
    const std::string src = "blocks." + std::to_string(layer) + ".";
    const std::string dst = "blk." + std::to_string(layer) + ".";
    same(dst + "attn_norm.weight", src + "ln1.weight", {d});
    same(dst + "attn_q.weight", src + "attn.q.weight", {d, d});
    same(dst + "attn_k.weight", src + "attn.k.weight", {d, d});
    same(dst + "attn_v.weight", src + "attn.v.weight", {d, d});
    same(dst + "attn_output.weight", src + "attn.out.weight", {d, d});
    same(dst + "ffn_norm.weight", src + "ln2.weight", {d});
    same(dst + "ffn_down.weight", src + "fc2.weight", {ff, d});
    const Tensor& fc1 = named_tensor(model, src + "fc1.weight");
    const size_t half = static_cast<size_t>(config.d_ff) * static_cast<size_t>(config.d_model);
    const auto gate = view.tensors.find(dst + "ffn_gate.weight");
    const auto up = view.tensors.find(dst + "ffn_up.weight");
    if (gate == view.tensors.end() || up == view.tensors.end() || gate->second.dims != std::vector<uint64_t>{d, ff} ||
        up->second.dims != std::vector<uint64_t>{d, ff} || fc1.data.size() != half * 2 ||
        gate->second.data.size() != half || up->second.data.size() != half) {
      throw std::runtime_error("self-test llama ffn");
    }
    const std::vector<float> gate_want(fc1.data.begin(), fc1.data.begin() + static_cast<std::ptrdiff_t>(half));
    const std::vector<float> up_want(fc1.data.begin() + static_cast<std::ptrdiff_t>(half), fc1.data.end());
    if (gate->second.data != gate_want || up->second.data != up_want) throw std::runtime_error("self-test llama ffn");
  }
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

  const Model from_bin = read_model(bin);
  const Model from_nnllm = read_model(gguf);
  const auto llama_bin = dir / "llama-bin.gguf";
  const auto llama_nnllm = dir / "llama-nnllm.gguf";
  write_llama_gguf(llama_bin, from_bin);
  write_llama_gguf(llama_nnllm, from_nnllm);
  if (read_file(llama_bin) != read_file(llama_nnllm)) throw std::runtime_error("self-test llama bytes");
  check_llama_file(model, llama_bin);
  Model gelu = model;
  gelu.config.swiglu = false;
  bool gelu_rejected = false;
  try {
    write_llama_gguf(dir / "gelu.gguf", gelu);
  } catch (const std::runtime_error& error) {
    gelu_rejected = std::string(error.what()) == "activation";
  }
  if (!gelu_rejected) throw std::runtime_error("self-test llama activation");
  Model biased = model;
  biased.config.bias = true;
  bool bias_rejected = false;
  try {
    write_llama_gguf(dir / "bias.gguf", biased);
  } catch (const std::runtime_error& error) {
    bias_rejected = std::string(error.what()) == "bias";
  }
  if (!bias_rejected) throw std::runtime_error("self-test llama bias");

  Config rope_config = config;
  rope_config.rope = true;
  Model rope_model;
  rope_model.config = rope_config;
  rope_model.tensors = expected_tensors(rope_config);
  uint32_t rope_n = 1;
  for (Tensor& tensor : rope_model.tensors) {
    if (tensor.name.find("pos_emb") != std::string::npos || tensor.name.find(".bias") != std::string::npos) {
      throw std::runtime_error("self-test rope tensors");
    }
    const int64_t numel = product(tensor.shape);
    tensor.data.resize(static_cast<size_t>(numel));
    for (int64_t i = 0; i < numel; ++i) {
      const uint32_t bits = 0x3f800000u + rope_n + static_cast<uint32_t>(i);
      std::memcpy(&tensor.data[static_cast<size_t>(i)], &bits, sizeof(bits));
    }
    ++rope_n;
  }
  const auto rope_bin = dir / "rope.bin";
  const auto rope_gguf = dir / "rope.gguf";
  const auto rope_back = dir / "rope-back.bin";
  write_checkpoint(rope_bin, rope_model);
  const Model rope_loaded = read_checkpoint(rope_bin);
  if (!rope_loaded.config.rope || rope_loaded.tensors.size() != rope_model.tensors.size()) {
    throw std::runtime_error("self-test rope load");
  }
  write_gguf(rope_gguf, rope_loaded);
  const Model rope_from_gguf = read_gguf(rope_gguf);
  if (!rope_from_gguf.config.rope) throw std::runtime_error("self-test rope positional");
  write_checkpoint(rope_back, rope_from_gguf);
  const Model rope_round = read_checkpoint(rope_back);
  if (rope_round.tensors.size() != rope_model.tensors.size()) throw std::runtime_error("self-test rope round");
  for (size_t i = 0; i < rope_model.tensors.size(); ++i) {
    if (rope_round.tensors[i].name != rope_model.tensors[i].name ||
        rope_round.tensors[i].shape != rope_model.tensors[i].shape ||
        rope_round.tensors[i].data != rope_model.tensors[i].data) {
      throw std::runtime_error("self-test rope round");
    }
  }
  const auto rope_llama = dir / "rope-llama.gguf";
  write_llama_gguf(rope_llama, read_model(rope_bin));
  check_llama_file(rope_model, rope_llama, true);
  rope_config.d_model = 6;
  rope_config.heads = 2;
  bool odd_rejected = false;
  try {
    require_config(rope_config);
  } catch (const std::runtime_error& error) {
    odd_rejected = std::string(error.what()) == "use_rope";
  }
  if (!odd_rejected) throw std::runtime_error("self-test rope odd");

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

  Config tok_config;
  tok_config.vocab = 260;
  tok_config.d_model = 8;
  tok_config.heads = 2;
  tok_config.layers = 1;
  tok_config.d_ff = 8;
  tok_config.max_seq = 16;
  tok_config.dropout = 0.f;
  tok_config.bias = false;
  tok_config.swiglu = true;
  tok_config.rope = true;
  Model tok_model;
  tok_model.config = tok_config;
  tok_model.tensors = expected_tensors(tok_config);
  uint32_t tok_n = 1;
  for (Tensor& tensor : tok_model.tensors) {
    const int64_t numel = product(tensor.shape);
    tensor.data.assign(static_cast<size_t>(numel), 0.f);
    if (!tensor.data.empty()) {
      const uint32_t bits = 0x3f800000u + tok_n;
      std::memcpy(&tensor.data[0], &bits, sizeof(bits));
    }
    ++tok_n;
  }
  Bytes bpe;
  bpe.u32(0x3142504cu);
  bpe.u32(260);
  bpe.u32(1);
  bpe.u32(257);
  bpe.u32(258);
  bpe.u32(259);
  bpe.u32(97);
  bpe.u32(98);
  const auto bpe_path = dir / "tokenizer.bpe";
  write_file(bpe_path, bpe.bytes());
  const Tokenizer tok = load_bpe(bpe_path);
  const auto tok_gguf = dir / "tok-llama.gguf";
  write_llama_gguf(tok_gguf, tok_model, &tok);
  const GgufView tok_view = read_gguf_view(tok_gguf);
  auto need_str = [&](const char* key, const std::string& want) {
    const auto found = tok_view.strings.find(key);
    if (found == tok_view.strings.end() || found->second != want) {
      throw std::runtime_error(std::string("self-test tokenizer ") + key);
    }
  };
  auto need_bool = [&](const char* key, bool want) {
    const auto found = tok_view.bools.find(key);
    if (found == tok_view.bools.end() || found->second != want) {
      throw std::runtime_error(std::string("self-test tokenizer ") + key);
    }
  };
  auto need_id = [&](const char* key, uint32_t want) {
    const auto found = tok_view.u32s.find(key);
    if (found == tok_view.u32s.end() || found->second != want) {
      throw std::runtime_error(std::string("self-test tokenizer ") + key);
    }
  };
  need_str("tokenizer.ggml.model", "gpt2");
  need_str("tokenizer.ggml.pre", "superbpe");
  need_bool("tokenizer.ggml.add_bos_token", false);
  need_bool("tokenizer.ggml.add_eos_token", false);
  need_bool("tokenizer.ggml.add_space_prefix", false);
  need_id("tokenizer.ggml.padding_token_id", 257);
  need_id("tokenizer.ggml.bos_token_id", 258);
  need_id("tokenizer.ggml.eos_token_id", 259);
  const auto tokens = tok_view.strs.find("tokenizer.ggml.tokens");
  const auto merges = tok_view.strs.find("tokenizer.ggml.merges");
  const auto types = tok_view.i32s.find("tokenizer.ggml.token_type");
  if (tokens == tok_view.strs.end() || tokens->second.size() != 260 || tokens->second[97] != "a" ||
      tokens->second[256] != "ab" || tokens->second[257] != "<pad>" || tokens->second[258] != "<bos>" ||
      tokens->second[259] != "<|endoftext|>") {
    throw std::runtime_error("self-test tokenizer tokens");
  }
  if (merges == tok_view.strs.end() || merges->second.size() != 1 || merges->second[0] != "a b") {
    throw std::runtime_error("self-test tokenizer merges");
  }
  if (types == tok_view.i32s.end() || types->second.size() != 260 || types->second[0] != 1 ||
      types->second[256] != 1 || types->second[257] != 3 || types->second[258] != 3 || types->second[259] != 3) {
    throw std::runtime_error("self-test tokenizer types");
  }
  Model mismatch = tok_model;
  mismatch.config.vocab = 261;
  bool vocab_rejected = false;
  try {
    write_llama_gguf(dir / "vocab.gguf", mismatch, &tok);
  } catch (const std::runtime_error& error) {
    vocab_rejected = std::string(error.what()) == "vocab";
  }
  if (!vocab_rejected) throw std::runtime_error("self-test tokenizer vocab");
  bool bad_bpe = false;
  try {
    load_bpe(dir / "missing.bpe");
  } catch (const std::runtime_error& error) {
    bad_bpe = std::string(error.what()) == "read";
  }
  if (!bad_bpe) throw std::runtime_error("self-test tokenizer read");

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
    if (argc < 2) {
      usage();
      throw std::runtime_error("usage");
    }
    const std::string command = argv[1];
    if (command == "llama" ? (argc != 4 && argc != 5) : argc != 4) {
      usage();
      throw std::runtime_error("usage");
    }
    if (command == "export") {
      write_gguf(argv[3], read_checkpoint(argv[2]));
      return 0;
    }
    if (command == "import") {
      write_checkpoint(argv[3], read_gguf(argv[2]));
      return 0;
    }
    if (command == "llama") {
      const Model model = read_model(argv[2]);
      std::filesystem::path tokenizer_path;
      if (argc == 5) {
        tokenizer_path = argv[4];
      } else {
        const auto sibling = std::filesystem::path(argv[2]).parent_path() / "tokenizer.bpe";
        if (std::filesystem::is_regular_file(sibling)) tokenizer_path = sibling;
      }
      if (tokenizer_path.empty()) {
        write_llama_gguf(argv[3], model);
        std::fputs("gguf: no tokenizer.bpe beside the checkpoint. llama-cli stops on a text prompt.\n", stderr);
      } else {
        const Tokenizer tok = load_bpe(tokenizer_path);
        write_llama_gguf(argv[3], model, &tok);
        std::fputs("gguf: embedded tokenizer.bpe.\n", stderr);
      }
      if (!model.config.rope) {
        std::fputs(
            "gguf: llama.cpp applies RoPE and RMSNorm. This checkpoint was trained with learned positions and "
            "LayerNorm. The file loads and the tokens differ. LayerNorm biases and the position table are omitted.\n",
            stderr);
      }
      return 0;
    }
    usage();
    throw std::runtime_error(command);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "gguf: %s\n", error.what());
    return 1;
  }
}
