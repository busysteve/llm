#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t kCheckpointMagic = 0x314E4E43u;
constexpr uint32_t kGgufMagic = 0x46554747u;
constexpr uint32_t kGgufVersion = 3u;
constexpr uint32_t kGgmlF32 = 0u;
constexpr uint32_t kGgmlF16 = 1u;
constexpr uint32_t kGgmlQ4_0 = 2u;
constexpr uint32_t kGgmlQ4_1 = 3u;
constexpr uint32_t kGgmlQ5_0 = 6u;
constexpr uint32_t kGgmlQ5_1 = 7u;
constexpr uint32_t kGgmlQ8_0 = 8u;
constexpr uint32_t kGgmlQ8_1 = 9u;
constexpr uint32_t kGgmlQ2_K = 10u;
constexpr uint32_t kGgmlQ3_K = 11u;
constexpr uint32_t kGgmlQ4_K = 12u;
constexpr uint32_t kGgmlQ5_K = 13u;
constexpr uint32_t kGgmlQ6_K = 14u;
constexpr uint32_t kGgmlQ8_K = 15u;
constexpr uint32_t kGgmlBF16 = 30u;
constexpr uint32_t kDefaultAlign = 32u;
constexpr int kQK = 32;
constexpr int kQK_K = 256;
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
  int kv_heads = 0;       // 0 => heads
  int head_dim = 0;       // 0 => d_model / heads
  bool qk_norm = false;
  float rope_theta = 10000.f;
  float rms_eps = 1e-5f;
};

int config_head_dim(const Config& config) {
  return config.head_dim > 0 ? config.head_dim : config.d_model / config.heads;
}

int config_kv_heads(const Config& config) {
  return config.kv_heads > 0 ? config.kv_heads : config.heads;
}

bool config_extended(const Config& config) {
  const bool wide = config.head_dim > 0 &&
                    (config.d_model % config.heads != 0 || config.head_dim != config.d_model / config.heads);
  const bool grouped = config.kv_heads > 0 && config.kv_heads != config.heads;
  return wide || grouped || config.qk_norm || config.rope_theta != 10000.f || config.rms_eps != 1e-5f;
}

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
float f16_to_f32(uint16_t bits);

void usage() {
  std::fputs(
      "usage: gguf export MODEL.BIN MODEL.GGUF\n"
      "       gguf import MODEL.GGUF MODEL.BIN\n"
      "       gguf llama MODEL MODEL.GGUF [TOKENIZER.BPE]\n"
      "       gguf from MODEL.GGUF OUTDIR [--context N]\n"
      "       gguf self-test\n"
      "\n"
      "export writes a GGUF version 3 file, architecture nnllm, float32.\n"
      "Tensor bytes stay in this library's row-major order. The GGUF dimension\n"
      "list stores the contiguous dimension first. import reads that file back,\n"
      "including float16 tensors, and writes a checkpoint with no optimizer.\n"
      "That file is version 1, or version 3 for a Qwen-style block. AdamW\n"
      "moments are not stored.\n"
      "\n"
      "llama reads a transformer checkpoint or an nnllm GGUF and writes a GGUF\n"
      "llama.cpp can load. A classic block is architecture llama. A Qwen-style\n"
      "block is qwen3 when QK-norm is on, and qwen2 otherwise. SwiGLU fc1 is\n"
      "split into ffn_gate and ffn_up.\n"
      "A learned-position checkpoint omits the position table and LayerNorm\n"
      "biases. llama.cpp then applies RoPE and RMSNorm, so those tokens differ.\n"
      "A RoPE checkpoint already uses that block, and the file keeps every\n"
      "tensor. A tokenizer.bpe next to the checkpoint, or a path passed after\n"
      "the output, is written into the file so llama-cli can encode a prompt.\n"
      "GELU and linear-bias models are rejected.\n"
      "\n"
      "from reads a public llama.cpp GGUF (architecture llama, qwen2, qwen3,\n"
      "mistral, and the same tensor layout) and writes a chat directory:\n"
      "model.bin, tokenizer.bpe, tokens.bin, and tokens.meta. GPT-2 BPE is\n"
      "remapped to this library's LPB1 layout (bytes 0..255, merges, then\n"
      "pad/bos/eos). Quantized tensors are expanded to float32. Grouped-query\n"
      "attention, a head dim other than d_model/heads, and QK-norm are kept.\n"
      "Tied embeddings copy token_embd onto lm_head. --context N sets\n"
      "max_seq_len (default min(source, 1024)). MoE, GELU, linear bias, and a\n"
      "partial RoPE dimension are rejected.\n"
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
  explicit Reader(std::vector<uint8_t> bytes)
      : owned_(std::move(bytes)), bytes_(owned_.data()), size_(owned_.size()) {}
  Reader(const uint8_t* bytes, size_t size) : bytes_(bytes), size_(size) {}
  size_t left() const { return size_ - pos_; }
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
  std::string str_u64() { return str_u64_max(65535); }
  std::string str_u64_max(uint64_t max_n) {
    const uint64_t n = u64();
    if (n > max_n) throw std::runtime_error("name");
    return take(static_cast<size_t>(n));
  }
  void skip(size_t n) {
    need(n);
    pos_ += n;
  }
  const uint8_t* at(size_t offset, size_t n) const {
    if (offset > size_ || n > size_ - offset) throw std::runtime_error("truncated");
    return bytes_ + offset;
  }
  size_t size() const { return size_; }

 private:
  void need(size_t n) const {
    if (pos_ + n > size_) throw std::runtime_error("truncated");
  }
  std::string take(size_t n) {
    need(n);
    std::string value(reinterpret_cast<const char*>(bytes_ + pos_), n);
    pos_ += n;
    return value;
  }
  std::vector<uint8_t> owned_;
  const uint8_t* bytes_ = nullptr;
  size_t size_ = 0;
  size_t pos_ = 0;
};

class MappedFile {
 public:
  explicit MappedFile(const std::filesystem::path& path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("read");
    struct stat st {};
    if (::fstat(fd_, &st) != 0 || st.st_size < 0) {
      ::close(fd_);
      fd_ = -1;
      throw std::runtime_error("read");
    }
    size_ = static_cast<size_t>(st.st_size);
    if (size_ == 0) {
      data_ = nullptr;
      return;
    }
    void* mapped = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped == MAP_FAILED) {
      ::close(fd_);
      fd_ = -1;
      throw std::runtime_error("read");
    }
    data_ = static_cast<const uint8_t*>(mapped);
  }
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  ~MappedFile() {
    if (data_ != nullptr && size_ > 0) ::munmap(const_cast<uint8_t*>(data_), size_);
    if (fd_ >= 0) ::close(fd_);
  }
  const uint8_t* data() const { return data_; }
  size_t size() const { return size_; }

 private:
  int fd_ = -1;
  const uint8_t* data_ = nullptr;
  size_t size_ = 0;
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
  const int dh = config_head_dim(config);
  const int kv = config_kv_heads(config);
  const int q_dim = config.heads * dh;
  const int kv_dim = kv * dh;
  add("tok_emb.weight", {config.vocab, d});
  if (!config.rope) add("pos_emb.weight", {config.max_seq, d});
  for (int layer = 0; layer < config.layers; ++layer) {
    const std::string prefix = "blocks." + std::to_string(layer) + ".";
    norm(prefix + "ln1");
    linear(prefix + "attn.q", q_dim, d);
    linear(prefix + "attn.k", kv_dim, d);
    linear(prefix + "attn.v", kv_dim, d);
    linear(prefix + "attn.out", d, q_dim);
    if (config.qk_norm) {
      add(prefix + "attn.q_norm.weight", {dh});
      add(prefix + "attn.k_norm.weight", {dh});
    }
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
  if (config.head_dim == 0) {
    if (config.d_model % config.heads != 0) throw std::runtime_error("num_heads");
  } else if (config.head_dim < 1) {
    throw std::runtime_error("head_dim");
  }
  const int dh = config_head_dim(config);
  const int kv = config_kv_heads(config);
  if (kv < 1 || kv > config.heads || config.heads % kv != 0) throw std::runtime_error("head_count_kv");
  if (dh > 0x7fffffff / config.heads || dh > 0x7fffffff / kv) throw std::runtime_error("head_dim");
  if (config.rope && dh % 2 != 0) throw std::runtime_error("use_rope");
  if (!(config.rope_theta > 0.f) || !std::isfinite(config.rope_theta)) throw std::runtime_error("rope_theta");
  if (!(config.rms_eps > 0.f) || !std::isfinite(config.rms_eps)) throw std::runtime_error("rms_eps");
  if (!(config.dropout >= 0.f) || !(config.dropout < 1.f)) throw std::runtime_error("dropout");
  if (config.swiglu && config.d_ff > 0x7fffffff / 2) throw std::runtime_error("d_ff");
}

Model read_checkpoint(const std::filesystem::path& path) {
  Reader reader(read_file(path));
  if (reader.u32() != kCheckpointMagic) throw std::runtime_error("magic");
  const uint32_t version = reader.u32();
  if (version < 1 || version > 4) throw std::runtime_error("version");
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
  if (activation == 1) config.swiglu = false;
  else if (activation == 2) config.swiglu = true;
  else throw std::runtime_error("activation");
  if (positional == 2) config.rope = true;
  else if (positional != 1) throw std::runtime_error("positional");
  if (version >= 3) {
    config.head_dim = reader.i32();
    config.kv_heads = reader.i32();
    config.qk_norm = reader.u32() != 0;
    config.rope_theta = reader.f32();
    config.rms_eps = reader.f32();
  }
  reader.pad8();
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
  out.u32(config_extended(model.config) ? 3u : 1u);
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
  if (config_extended(config)) {
    out.i32(config.head_dim);
    out.i32(config.kv_heads);
    out.u32(config.qk_norm ? 1u : 0u);
    out.f32(config.rope_theta);
    out.f32(config.rms_eps);
  }
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
  void str(const std::string& key, const std::string& value) {
    gguf_kv_string(body, key.c_str(), value);
    ++count;
  }
  void u32(const std::string& key, uint32_t value) {
    gguf_kv_u32(body, key.c_str(), value);
    ++count;
  }
  void f32(const std::string& key, float value) {
    gguf_kv_f32(body, key.c_str(), value);
    ++count;
  }
  void boolean(const std::string& key, bool value) {
    gguf_kv_bool(body, key.c_str(), value);
    ++count;
  }
  void strs(const std::string& key, const std::vector<std::string>& values) {
    gguf_string(body, key);
    body.u32(kArray);
    body.u32(kString);
    body.u64(values.size());
    for (const std::string& value : values) gguf_string(body, value);
    ++count;
  }
  void i32s(const std::string& key, const std::vector<int32_t>& values) {
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
  if (config_extended(config)) {
    kv.u32("nnllm.attention.head_count_kv", static_cast<uint32_t>(config_kv_heads(config)));
    kv.u32("nnllm.attention.key_length", static_cast<uint32_t>(config_head_dim(config)));
    kv.u32("nnllm.attention.value_length", static_cast<uint32_t>(config_head_dim(config)));
    kv.boolean("nnllm.qk_norm", config.qk_norm);
    kv.f32("nnllm.rope.freq_base", config.rope_theta);
    kv.f32("nnllm.attention.layer_norm_rms_epsilon", config.rms_eps);
  }
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
  const int dh = config_head_dim(config);
  const int kv = config_kv_heads(config);
  const int q_dim = config.heads * dh;
  const int kv_dim = kv * dh;
  std::vector<Tensor> tensors;
  tensors.push_back(copy_tensor(model, "tok_emb.weight", {config.vocab, d}, "token_embd.weight"));
  for (int layer = 0; layer < config.layers; ++layer) {
    const std::string src = "blocks." + std::to_string(layer) + ".";
    const std::string dst = "blk." + std::to_string(layer) + ".";
    tensors.push_back(copy_tensor(model, src + "ln1.weight", {d}, dst + "attn_norm.weight"));
    tensors.push_back(copy_tensor(model, src + "attn.q.weight", {q_dim, d}, dst + "attn_q.weight"));
    tensors.push_back(copy_tensor(model, src + "attn.k.weight", {kv_dim, d}, dst + "attn_k.weight"));
    tensors.push_back(copy_tensor(model, src + "attn.v.weight", {kv_dim, d}, dst + "attn_v.weight"));
    tensors.push_back(copy_tensor(model, src + "attn.out.weight", {d, q_dim}, dst + "attn_output.weight"));
    if (config.qk_norm) {
      tensors.push_back(copy_tensor(model, src + "attn.q_norm.weight", {dh}, dst + "attn_q_norm.weight"));
      tensors.push_back(copy_tensor(model, src + "attn.k_norm.weight", {dh}, dst + "attn_k_norm.weight"));
    }
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
  const bool qwen = config_extended(config);
  const char* arch = !qwen ? "llama" : (config.qk_norm ? "qwen3" : "qwen2");
  const std::string prefix = std::string(arch) + ".";
  Kv kv;
  kv.str("general.architecture", arch);
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
  kv.u32(prefix + "vocab_size", static_cast<uint32_t>(config.vocab));
  kv.u32(prefix + "context_length", static_cast<uint32_t>(config.max_seq));
  kv.u32(prefix + "embedding_length", static_cast<uint32_t>(config.d_model));
  kv.u32(prefix + "block_count", static_cast<uint32_t>(config.layers));
  kv.u32(prefix + "feed_forward_length", static_cast<uint32_t>(config.d_ff));
  kv.u32(prefix + "attention.head_count", static_cast<uint32_t>(config.heads));
  kv.u32(prefix + "attention.head_count_kv", static_cast<uint32_t>(config_kv_heads(config)));
  kv.u32(prefix + "attention.key_length", static_cast<uint32_t>(config_head_dim(config)));
  kv.u32(prefix + "attention.value_length", static_cast<uint32_t>(config_head_dim(config)));
  kv.f32(prefix + "attention.layer_norm_rms_epsilon", config.rms_eps);
  kv.f32(prefix + "rope.freq_base", config.rope_theta);
  kv.u32(prefix + "rope.dimension_count", static_cast<uint32_t>(config_head_dim(config)));
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
  auto has_key = [&](const char* key) { return fields.find(key) != fields.end(); };
  if (has_key("nnllm.attention.head_count_kv")) {
    config.kv_heads = static_cast<int>(u32_at("nnllm.attention.head_count_kv"));
  }
  if (has_key("nnllm.attention.key_length")) {
    config.head_dim = static_cast<int>(u32_at("nnllm.attention.key_length"));
  }
  if (has_key("nnllm.qk_norm")) config.qk_norm = bool_at("nnllm.qk_norm");
  if (has_key("nnllm.rope.freq_base")) config.rope_theta = f32_at("nnllm.rope.freq_base");
  if (has_key("nnllm.attention.layer_norm_rms_epsilon")) {
    config.rms_eps = f32_at("nnllm.attention.layer_norm_rms_epsilon");
  }
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

int ggml_block_size(uint32_t dtype) {
  switch (dtype) {
    case kGgmlF32:
    case kGgmlF16:
    case kGgmlBF16:
      return 1;
    case kGgmlQ4_0:
    case kGgmlQ4_1:
    case kGgmlQ5_0:
    case kGgmlQ5_1:
    case kGgmlQ8_0:
    case kGgmlQ8_1:
      return kQK;
    case kGgmlQ2_K:
    case kGgmlQ3_K:
    case kGgmlQ4_K:
    case kGgmlQ5_K:
    case kGgmlQ6_K:
    case kGgmlQ8_K:
      return kQK_K;
    default:
      return 0;
  }
}

size_t ggml_type_size(uint32_t dtype) {
  switch (dtype) {
    case kGgmlF32:
      return 4;
    case kGgmlF16:
    case kGgmlBF16:
      return 2;
    case kGgmlQ4_0:
      return 18;
    case kGgmlQ4_1:
      return 20;
    case kGgmlQ5_0:
      return 22;
    case kGgmlQ5_1:
      return 24;
    case kGgmlQ8_0:
      return 34;
    case kGgmlQ8_1:
      return 36;
    case kGgmlQ2_K:
      return 84;
    case kGgmlQ3_K:
      return 110;
    case kGgmlQ4_K:
      return 144;
    case kGgmlQ5_K:
      return 176;
    case kGgmlQ6_K:
      return 210;
    case kGgmlQ8_K:
      return 292;
    default:
      return 0;
  }
}

uint16_t load_u16(const uint8_t* p) {
  return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

float load_f32_bits(const uint8_t* p) {
  uint32_t bits = 0;
  for (int b = 0; b < 4; ++b) bits |= static_cast<uint32_t>(p[b]) << (8 * b);
  float value = 0.f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

void dequantize(uint32_t dtype, const uint8_t* raw, int64_t numel, std::vector<float>& out) {
  out.resize(static_cast<size_t>(numel));
  const int blck = ggml_block_size(dtype);
  const size_t tsize = ggml_type_size(dtype);
  if (blck < 1 || tsize < 1) throw std::runtime_error("quantized");
  if (numel % blck != 0) throw std::runtime_error("quantized");
  const int64_t nb = numel / blck;
  if (dtype == kGgmlF32) {
    for (int64_t i = 0; i < numel; ++i) out[static_cast<size_t>(i)] = load_f32_bits(raw + static_cast<size_t>(i) * 4);
    return;
  }
  if (dtype == kGgmlF16) {
    for (int64_t i = 0; i < numel; ++i) {
      out[static_cast<size_t>(i)] = f16_to_f32(load_u16(raw + static_cast<size_t>(i) * 2));
    }
    return;
  }
  if (dtype == kGgmlBF16) {
    for (int64_t i = 0; i < numel; ++i) {
      const uint32_t bits = static_cast<uint32_t>(load_u16(raw + static_cast<size_t>(i) * 2)) << 16;
      std::memcpy(&out[static_cast<size_t>(i)], &bits, sizeof(bits));
    }
    return;
  }
  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t* block = raw + static_cast<size_t>(i) * tsize;
    float* y = out.data() + static_cast<size_t>(i) * static_cast<size_t>(blck);
    if (dtype == kGgmlQ8_0) {
      const float d = f16_to_f32(load_u16(block));
      for (int j = 0; j < kQK; ++j) y[j] = static_cast<int8_t>(block[2 + j]) * d;
    } else if (dtype == kGgmlQ8_1) {
      const float d = f16_to_f32(load_u16(block));
      for (int j = 0; j < kQK; ++j) y[j] = static_cast<int8_t>(block[4 + j]) * d;
    } else if (dtype == kGgmlQ4_0) {
      const float d = f16_to_f32(load_u16(block));
      for (int j = 0; j < kQK / 2; ++j) {
        y[j] = (static_cast<int>(block[2 + j] & 0x0F) - 8) * d;
        y[j + kQK / 2] = (static_cast<int>(block[2 + j] >> 4) - 8) * d;
      }
    } else if (dtype == kGgmlQ4_1) {
      const float d = f16_to_f32(load_u16(block));
      const float m = f16_to_f32(load_u16(block + 2));
      for (int j = 0; j < kQK / 2; ++j) {
        y[j] = static_cast<int>(block[4 + j] & 0x0F) * d + m;
        y[j + kQK / 2] = static_cast<int>(block[4 + j] >> 4) * d + m;
      }
    } else if (dtype == kGgmlQ5_0) {
      const float d = f16_to_f32(load_u16(block));
      uint32_t qh = 0;
      std::memcpy(&qh, block + 2, 4);
      for (int j = 0; j < kQK / 2; ++j) {
        const uint8_t xh0 = static_cast<uint8_t>(((qh >> (j + 0)) << 4) & 0x10);
        const uint8_t xh1 = static_cast<uint8_t>((qh >> (j + 12)) & 0x10);
        y[j] = (static_cast<int>((block[6 + j] & 0x0F) | xh0) - 16) * d;
        y[j + kQK / 2] = (static_cast<int>((block[6 + j] >> 4) | xh1) - 16) * d;
      }
    } else if (dtype == kGgmlQ5_1) {
      const float d = f16_to_f32(load_u16(block));
      const float m = f16_to_f32(load_u16(block + 2));
      uint32_t qh = 0;
      std::memcpy(&qh, block + 4, 4);
      for (int j = 0; j < kQK / 2; ++j) {
        const uint8_t xh0 = static_cast<uint8_t>(((qh >> (j + 0)) << 4) & 0x10);
        const uint8_t xh1 = static_cast<uint8_t>((qh >> (j + 12)) & 0x10);
        y[j] = static_cast<int>((block[8 + j] & 0x0F) | xh0) * d + m;
        y[j + kQK / 2] = static_cast<int>((block[8 + j] >> 4) | xh1) * d + m;
      }
    } else if (dtype == kGgmlQ4_K) {
      const float d = f16_to_f32(load_u16(block));
      const float min = f16_to_f32(load_u16(block + 2));
      const uint8_t* scales = block + 4;
      const uint8_t* q = block + 16;
      float* yp = y;
      int is = 0;
      for (int j = 0; j < kQK_K; j += 64) {
        uint8_t sc = 0, m = 0;
        get_scale_min_k4(is + 0, scales, &sc, &m);
        const float d1 = d * sc;
        const float m1 = min * m;
        get_scale_min_k4(is + 1, scales, &sc, &m);
        const float d2 = d * sc;
        const float m2 = min * m;
        for (int l = 0; l < 32; ++l) *yp++ = d1 * (q[l] & 0xF) - m1;
        for (int l = 0; l < 32; ++l) *yp++ = d2 * (q[l] >> 4) - m2;
        q += 32;
        is += 2;
      }
    } else if (dtype == kGgmlQ5_K) {
      const float d = f16_to_f32(load_u16(block));
      const float min = f16_to_f32(load_u16(block + 2));
      const uint8_t* scales = block + 4;
      const uint8_t* qh = block + 16;
      const uint8_t* ql = block + 48;
      float* yp = y;
      int is = 0;
      uint8_t u1 = 1, u2 = 2;
      for (int j = 0; j < kQK_K; j += 64) {
        uint8_t sc = 0, m = 0;
        get_scale_min_k4(is + 0, scales, &sc, &m);
        const float d1 = d * sc;
        const float m1 = min * m;
        get_scale_min_k4(is + 1, scales, &sc, &m);
        const float d2 = d * sc;
        const float m2 = min * m;
        for (int l = 0; l < 32; ++l) *yp++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
        for (int l = 0; l < 32; ++l) *yp++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
        ql += 32;
        is += 2;
        u1 = static_cast<uint8_t>(u1 << 2);
        u2 = static_cast<uint8_t>(u2 << 2);
      }
    } else if (dtype == kGgmlQ6_K) {
      const uint8_t* ql = block;
      const uint8_t* qh = block + 128;
      const int8_t* sc = reinterpret_cast<const int8_t*>(block + 192);
      const float d = f16_to_f32(load_u16(block + 208));
      float* yp = y;
      for (int n = 0; n < kQK_K; n += 128) {
        for (int l = 0; l < 32; ++l) {
          const int is = l / 16;
          const int8_t q1 = static_cast<int8_t>((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
          const int8_t q2 = static_cast<int8_t>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
          const int8_t q3 = static_cast<int8_t>((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
          const int8_t q4 = static_cast<int8_t>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
          yp[l + 0] = d * sc[is + 0] * q1;
          yp[l + 32] = d * sc[is + 2] * q2;
          yp[l + 64] = d * sc[is + 4] * q3;
          yp[l + 96] = d * sc[is + 6] * q4;
        }
        yp += 128;
        ql += 64;
        qh += 32;
        sc += 8;
      }
    } else if (dtype == kGgmlQ8_K) {
      const float d = load_f32_bits(block);
      const int8_t* qs = reinterpret_cast<const int8_t*>(block + 4);
      for (int j = 0; j < kQK_K; ++j) y[j] = d * qs[j];
    } else {
      throw std::runtime_error("quantized");
    }
  }
}

struct PublicTensor {
  std::string name;
  std::vector<int> shape;
  uint32_t dtype = 0;
  uint64_t offset = 0;
  int64_t numel = 0;
};

struct PublicGguf {
  std::string architecture;
  uint32_t alignment = kDefaultAlign;
  size_t data_begin = 0;
  std::map<std::string, std::string> strings;
  std::map<std::string, uint32_t> u32s;
  std::map<std::string, float> f32s;
  std::map<std::string, bool> bools;
  std::map<std::string, std::vector<std::string>> strs;
  std::map<std::string, PublicTensor> tensors;
};

uint32_t utf8_decode_one(const std::string& s, size_t& i) {
  if (i >= s.size()) throw std::runtime_error("tokenizer");
  const unsigned char b0 = static_cast<unsigned char>(s[i++]);
  if (b0 < 0x80u) return b0;
  if ((b0 & 0xe0u) == 0xc0u) {
    if (i >= s.size()) throw std::runtime_error("tokenizer");
    const unsigned char b1 = static_cast<unsigned char>(s[i++]);
    if ((b1 & 0xc0u) != 0x80u) throw std::runtime_error("tokenizer");
    return (static_cast<uint32_t>(b0 & 0x1fu) << 6) | (b1 & 0x3fu);
  }
  if ((b0 & 0xf0u) == 0xe0u) {
    if (i + 1 >= s.size()) throw std::runtime_error("tokenizer");
    const unsigned char b1 = static_cast<unsigned char>(s[i++]);
    const unsigned char b2 = static_cast<unsigned char>(s[i++]);
    if ((b1 & 0xc0u) != 0x80u || (b2 & 0xc0u) != 0x80u) throw std::runtime_error("tokenizer");
    return (static_cast<uint32_t>(b0 & 0x0fu) << 12) | (static_cast<uint32_t>(b1 & 0x3fu) << 6) | (b2 & 0x3fu);
  }
  throw std::runtime_error("tokenizer");
}

std::optional<std::string> gpt2_to_bytes(const std::string& piece, const std::array<int, 1024>& cpt_to_byte) {
  std::string raw;
  size_t i = 0;
  while (i < piece.size()) {
    const uint32_t cpt = utf8_decode_one(piece, i);
    if (cpt >= cpt_to_byte.size() || cpt_to_byte[cpt] < 0) return std::nullopt;
    raw.push_back(static_cast<char>(cpt_to_byte[cpt]));
  }
  return raw;
}

std::array<int, 1024> gpt2_cpt_to_byte() {
  std::array<int, 1024> map;
  map.fill(-1);
  const std::array<std::string, 256> tokens = gpt2_byte_tokens();
  for (int byte = 0; byte < 256; ++byte) {
    size_t i = 0;
    const uint32_t cpt = utf8_decode_one(tokens[static_cast<size_t>(byte)], i);
    if (i != tokens[static_cast<size_t>(byte)].size() || cpt >= map.size()) {
      throw std::runtime_error("tokenizer");
    }
    map[cpt] = byte;
  }
  return map;
}

struct ConvertedBpe {
  Tokenizer tok;
  std::vector<int32_t> orig_to_new;
  std::vector<int32_t> new_to_orig;
};

ConvertedBpe bpe_from_gpt2(const std::vector<std::string>& tokens, const std::vector<std::string>& merges,
                           int pad_orig, int bos_orig, int eos_orig) {
  if (tokens.size() < 259) throw std::runtime_error("tokenizer");
  const std::array<int, 1024> cpt_to_byte = gpt2_cpt_to_byte();
  std::map<std::string, int> token_index;
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (!token_index.emplace(tokens[i], static_cast<int>(i)).second) throw std::runtime_error("tokenizer");
  }
  std::vector<int> byte_orig(256, -1);
  for (size_t i = 0; i < tokens.size(); ++i) {
    const std::optional<std::string> raw = gpt2_to_bytes(tokens[i], cpt_to_byte);
    if (raw && raw->size() == 1) {
      const int b = static_cast<unsigned char>((*raw)[0]);
      if (byte_orig[static_cast<size_t>(b)] >= 0) throw std::runtime_error("tokenizer");
      byte_orig[static_cast<size_t>(b)] = static_cast<int>(i);
    }
  }
  for (int b = 0; b < 256; ++b) {
    if (byte_orig[static_cast<size_t>(b)] < 0) throw std::runtime_error("tokenizer");
  }

  ConvertedBpe out;
  out.orig_to_new.assign(tokens.size(), -1);
  const int n_merges = static_cast<int>(merges.size());
  const int vocab = 256 + n_merges + 3;
  if (vocab < 259) throw std::runtime_error("tokenizer");
  out.new_to_orig.assign(static_cast<size_t>(vocab), -1);
  for (int b = 0; b < 256; ++b) {
    const int orig = byte_orig[static_cast<size_t>(b)];
    out.orig_to_new[static_cast<size_t>(orig)] = b;
    out.new_to_orig[static_cast<size_t>(b)] = orig;
  }

  std::vector<std::pair<int, int>> pairs;
  pairs.reserve(static_cast<size_t>(n_merges));
  for (int i = 0; i < n_merges; ++i) {
    const std::string& merge = merges[static_cast<size_t>(i)];
    const size_t sp = merge.find(' ');
    if (sp == std::string::npos || merge.find(' ', sp + 1) != std::string::npos) {
      throw std::runtime_error("tokenizer");
    }
    const std::string left = merge.substr(0, sp);
    const std::string right = merge.substr(sp + 1);
    const auto left_it = token_index.find(left);
    const auto right_it = token_index.find(right);
    const auto result_it = token_index.find(left + right);
    if (left_it == token_index.end() || right_it == token_index.end() || result_it == token_index.end()) {
      throw std::runtime_error("tokenizer");
    }
    const int left_new = out.orig_to_new[static_cast<size_t>(left_it->second)];
    const int right_new = out.orig_to_new[static_cast<size_t>(right_it->second)];
    const int new_id = 256 + i;
    if (left_new < 0 || right_new < 0 || left_new >= new_id || right_new >= new_id) {
      throw std::runtime_error("tokenizer");
    }
    if (out.orig_to_new[static_cast<size_t>(result_it->second)] >= 0) throw std::runtime_error("tokenizer");
    out.orig_to_new[static_cast<size_t>(result_it->second)] = new_id;
    out.new_to_orig[static_cast<size_t>(new_id)] = result_it->second;
    pairs.emplace_back(left_new, right_new);
  }

  auto special_orig = [&](int orig, int fallback) {
    if (orig >= 0 && orig < static_cast<int>(tokens.size())) return orig;
    return fallback;
  };
  const int pad = special_orig(pad_orig, static_cast<int>(tokens.size()) - 3);
  const int bos = special_orig(bos_orig, static_cast<int>(tokens.size()) - 2);
  const int eos = special_orig(eos_orig, static_cast<int>(tokens.size()) - 1);
  out.new_to_orig[static_cast<size_t>(vocab - 3)] = pad;
  out.new_to_orig[static_cast<size_t>(vocab - 2)] = bos;
  out.new_to_orig[static_cast<size_t>(vocab - 1)] = eos;
  out.orig_to_new[static_cast<size_t>(pad)] = vocab - 3;
  out.orig_to_new[static_cast<size_t>(bos)] = vocab - 2;
  out.orig_to_new[static_cast<size_t>(eos)] = vocab - 1;

  out.tok.vocab = vocab;
  out.tok.pad = vocab - 3;
  out.tok.bos = vocab - 2;
  out.tok.eos = vocab - 1;
  out.tok.merges.reserve(pairs.size());
  for (const auto& pair : pairs) {
    out.tok.merges.push_back(std::to_string(pair.first) + " " + std::to_string(pair.second));
  }
  out.tok.tokens.clear();
  out.tok.types.assign(static_cast<size_t>(vocab), 1);
  out.tok.types[static_cast<size_t>(out.tok.pad)] = 3;
  out.tok.types[static_cast<size_t>(out.tok.bos)] = 3;
  out.tok.types[static_cast<size_t>(out.tok.eos)] = 3;
  return out;
}

struct BpePairs {
  ConvertedBpe map;
  std::vector<std::pair<uint32_t, uint32_t>> pairs;
};

BpePairs convert_gpt2_bpe(const std::vector<std::string>& tokens, const std::vector<std::string>& merges,
                          int pad_orig, int bos_orig, int eos_orig) {
  BpePairs out;
  out.map = bpe_from_gpt2(tokens, merges, pad_orig, bos_orig, eos_orig);
  out.pairs.reserve(merges.size());
  for (const std::string& merge : out.map.tok.merges) {
    const size_t sp = merge.find(' ');
    out.pairs.emplace_back(static_cast<uint32_t>(std::stoul(merge.substr(0, sp))),
                           static_cast<uint32_t>(std::stoul(merge.substr(sp + 1))));
  }
  return out;
}

void write_bpe_file(const std::filesystem::path& path, const BpePairs& bpe) {
  Bytes out;
  out.u32(0x3142504cu);
  out.u32(static_cast<uint32_t>(bpe.map.tok.vocab));
  out.u32(static_cast<uint32_t>(bpe.pairs.size()));
  out.u32(static_cast<uint32_t>(bpe.map.tok.pad));
  out.u32(static_cast<uint32_t>(bpe.map.tok.bos));
  out.u32(static_cast<uint32_t>(bpe.map.tok.eos));
  for (const auto& pair : bpe.pairs) {
    out.u32(pair.first);
    out.u32(pair.second);
  }
  write_file(path, out.bytes());
}

std::vector<int32_t> encode_lpb1(const BpePairs& bpe, std::string_view text, bool add_eos) {
  std::vector<int32_t> ids;
  ids.reserve(text.size());
  for (char ch : text) ids.push_back(static_cast<int32_t>(static_cast<unsigned char>(ch)));
  for (size_t i = 0; i < bpe.pairs.size(); ++i) {
    if (ids.size() < 2) break;
    const int32_t left = static_cast<int32_t>(bpe.pairs[i].first);
    const int32_t right = static_cast<int32_t>(bpe.pairs[i].second);
    const int32_t merged = static_cast<int32_t>(256 + i);
    std::vector<int32_t> next;
    next.reserve(ids.size());
    size_t j = 0;
    while (j < ids.size()) {
      if (j + 1 < ids.size() && ids[j] == left && ids[j + 1] == right) {
        next.push_back(merged);
        j += 2;
      } else {
        next.push_back(ids[j]);
        ++j;
      }
    }
    ids = std::move(next);
  }
  if (add_eos) ids.push_back(bpe.map.tok.eos);
  return ids;
}

void write_token_files(const std::filesystem::path& bin_path, const std::filesystem::path& meta_path,
                       const std::vector<int32_t>& ids, int vocab) {
  Bytes bin;
  for (int32_t id : ids) {
    if (id < 0 || id >= vocab) throw std::runtime_error("id");
    bin.i32(id);
  }
  write_file(bin_path, bin.bytes());
  const std::string meta = std::to_string(ids.size()) + "\n" + std::to_string(vocab) + "\n";
  write_file(meta_path, std::vector<uint8_t>(meta.begin(), meta.end()));
}

Tensor permute_rows(const Tensor& src, const std::vector<int32_t>& new_to_orig, int new_vocab) {
  if (src.shape.size() != 2 || src.shape[0] < 1 || src.shape[1] < 1) throw std::runtime_error("tensor " + src.name);
  const int old_vocab = src.shape[0];
  const int d = src.shape[1];
  if (static_cast<int64_t>(src.data.size()) != product(src.shape)) throw std::runtime_error("tensor " + src.name);
  Tensor out;
  out.name = src.name;
  out.shape = {new_vocab, d};
  out.data.assign(static_cast<size_t>(new_vocab) * static_cast<size_t>(d), 0.f);
  for (int row = 0; row < new_vocab; ++row) {
    const int orig = new_to_orig[static_cast<size_t>(row)];
    if (orig < 0 || orig >= old_vocab) continue;
    const float* from = src.data.data() + static_cast<size_t>(orig) * static_cast<size_t>(d);
    float* to = out.data.data() + static_cast<size_t>(row) * static_cast<size_t>(d);
    std::memcpy(to, from, static_cast<size_t>(d) * sizeof(float));
  }
  return out;
}

Tensor repeat_kv_heads(const Tensor& src, int heads, int kv_heads, int d_model) {
  if (kv_heads < 1 || heads < kv_heads || heads % kv_heads != 0) throw std::runtime_error("head_count_kv");
  const int dh = d_model / heads;
  const int src_out = kv_heads * dh;
  if (src.shape != std::vector<int>{src_out, d_model}) throw std::runtime_error("tensor " + src.name);
  if (heads == kv_heads) {
    Tensor out = src;
    out.shape = {d_model, d_model};
    return out;
  }
  const int repeat = heads / kv_heads;
  Tensor out;
  out.name = src.name;
  out.shape = {d_model, d_model};
  out.data.resize(static_cast<size_t>(d_model) * static_cast<size_t>(d_model));
  for (int h = 0; h < kv_heads; ++h) {
    for (int r = 0; r < repeat; ++r) {
      const int dst_h = h * repeat + r;
      for (int d = 0; d < dh; ++d) {
        const float* from =
            src.data.data() + (static_cast<size_t>(h) * static_cast<size_t>(dh) + static_cast<size_t>(d)) *
                                  static_cast<size_t>(d_model);
        float* to = out.data.data() + (static_cast<size_t>(dst_h) * static_cast<size_t>(dh) + static_cast<size_t>(d)) *
                                          static_cast<size_t>(d_model);
        std::memcpy(to, from, static_cast<size_t>(d_model) * sizeof(float));
      }
    }
  }
  return out;
}

PublicGguf parse_public_gguf(Reader& reader) {
  if (reader.size() < 24 || reader.u32() != kGgufMagic) throw std::runtime_error("magic");
  if (reader.u32() != kGgufVersion) throw std::runtime_error("version");
  const uint64_t tensor_count = reader.u64();
  const uint64_t kv_count = reader.u64();
  if (tensor_count > 1000000u || kv_count > 100000u) throw std::runtime_error("truncated");
  PublicGguf view;
  for (uint64_t i = 0; i < kv_count; ++i) {
    const std::string key = reader.str_u64_max(65535);
    const uint32_t type = reader.u32();
    if (key == "general.alignment") {
      view.alignment = meta_u32(reader, type);
      continue;
    }
    if (type == kString) {
      view.strings[key] = reader.str_u64_max(1u << 20);
    } else if (type == kU32) {
      view.u32s[key] = reader.u32();
    } else if (type == kI32) {
      const int32_t value = reader.i32();
      if (value < 0) throw std::runtime_error("metadata");
      view.u32s[key] = static_cast<uint32_t>(value);
    } else if (type == kU64) {
      const uint64_t value = reader.u64();
      if (value > 0xffffffffu) throw std::runtime_error("metadata");
      view.u32s[key] = static_cast<uint32_t>(value);
    } else if (type == kF32) {
      view.f32s[key] = reader.f32();
    } else if (type == kBool) {
      const uint8_t value = reader.u8();
      if (value > 1) throw std::runtime_error("metadata");
      view.bools[key] = value != 0;
    } else if (type == kArray) {
      const uint32_t elem = reader.u32();
      const uint64_t n = reader.u64();
      if (n > 2000000u) throw std::runtime_error("metadata");
      if (elem == kString) {
        std::vector<std::string> values;
        values.reserve(static_cast<size_t>(n));
        for (uint64_t j = 0; j < n; ++j) values.push_back(reader.str_u64_max(65535));
        view.strs.emplace(key, std::move(values));
      } else {
        for (uint64_t j = 0; j < n; ++j) skip_meta(reader, elem, 1);
      }
    } else {
      skip_meta(reader, type, 0);
    }
  }
  if (view.alignment < 8 || view.alignment > 4096 || (view.alignment & (view.alignment - 1)) != 0) {
    throw std::runtime_error("alignment");
  }
  for (uint64_t i = 0; i < tensor_count; ++i) {
    PublicTensor tensor;
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
    if (tensor.offset % view.alignment != 0) throw std::runtime_error("alignment");
    tensor.numel = product(tensor.shape);
    if (tensor.numel < 0) throw std::runtime_error("tensor " + tensor.name);
    if (!view.tensors.emplace(tensor.name, tensor).second) throw std::runtime_error("tensor " + tensor.name);
  }
  view.data_begin = static_cast<size_t>(align_up(reader.pos(), view.alignment));
  const auto arch = view.strings.find("general.architecture");
  if (arch == view.strings.end() || arch->second.empty()) throw std::runtime_error("architecture");
  view.architecture = arch->second;
  return view;
}

uint32_t need_u32_key(const PublicGguf& view, const std::string& key) {
  const auto found = view.u32s.find(key);
  if (found == view.u32s.end()) throw std::runtime_error(key);
  return found->second;
}

uint32_t optional_u32_key(const PublicGguf& view, const std::string& key, uint32_t fallback) {
  const auto found = view.u32s.find(key);
  if (found == view.u32s.end()) return fallback;
  return found->second;
}

Tensor load_public_tensor(const PublicGguf& view, const uint8_t* data, size_t size, const std::string& name,
                          std::vector<int> want) {
  const auto found = view.tensors.find(name);
  if (found == view.tensors.end()) throw std::runtime_error("tensor " + name);
  const PublicTensor& info = found->second;
  if (info.shape != want) throw std::runtime_error("tensor " + name);
  const int blck = ggml_block_size(info.dtype);
  const size_t tsize = ggml_type_size(info.dtype);
  if (blck < 1 || tsize < 1) throw std::runtime_error("quantized");
  const uint64_t nbytes = static_cast<uint64_t>(info.numel / blck) * tsize;
  if (view.data_begin > size || info.offset > size - view.data_begin ||
      nbytes > size - view.data_begin - info.offset) {
    throw std::runtime_error("truncated");
  }
  Tensor out;
  out.name = name;
  out.shape = std::move(want);
  dequantize(info.dtype, data + view.data_begin + static_cast<size_t>(info.offset), info.numel, out.data);
  return out;
}

bool has_tensor(const PublicGguf& view, const std::string& name) {
  return view.tensors.find(name) != view.tensors.end();
}

void require_no_tensor(const PublicGguf& view, const std::string& name, const char* reason) {
  if (has_tensor(view, name)) throw std::runtime_error(reason);
}

Model public_to_nnllm(const PublicGguf& view, const uint8_t* data, size_t size, const BpePairs& bpe, int max_seq) {
  if (view.architecture == "nnllm") throw std::runtime_error("architecture");
  const std::string p = view.architecture + ".";
  Config config;
  config.d_model = static_cast<int>(need_u32_key(view, p + "embedding_length"));
  config.layers = static_cast<int>(need_u32_key(view, p + "block_count"));
  config.d_ff = static_cast<int>(need_u32_key(view, p + "feed_forward_length"));
  config.heads = static_cast<int>(need_u32_key(view, p + "attention.head_count"));
  const int kv_heads = static_cast<int>(optional_u32_key(view, p + "attention.head_count_kv",
                                                        static_cast<uint32_t>(config.heads)));
  const int src_ctx = static_cast<int>(need_u32_key(view, p + "context_length"));
  config.max_seq = max_seq > 0 ? max_seq : (src_ctx > 1024 ? 1024 : src_ctx);
  config.vocab = bpe.map.tok.vocab;
  config.dropout = 0.f;
  config.bias = false;
  config.swiglu = true;
  config.rope = true;
  config.kv_heads = kv_heads;
  const int natural = config.d_model % config.heads == 0 ? config.d_model / config.heads : 0;
  const int key_length =
      static_cast<int>(optional_u32_key(view, p + "attention.key_length", static_cast<uint32_t>(natural)));
  const int value_length =
      static_cast<int>(optional_u32_key(view, p + "attention.value_length", static_cast<uint32_t>(key_length)));
  if (key_length < 1 || value_length != key_length) throw std::runtime_error("head_dim");
  config.head_dim = key_length;
  const auto rope_base = view.f32s.find(p + "rope.freq_base");
  if (rope_base != view.f32s.end()) config.rope_theta = rope_base->second;
  const auto rms_eps = view.f32s.find(p + "attention.layer_norm_rms_epsilon");
  if (rms_eps != view.f32s.end()) config.rms_eps = rms_eps->second;
  const auto rope_dim = view.u32s.find(p + "rope.dimension_count");
  if (rope_dim != view.u32s.end() && static_cast<int>(rope_dim->second) != key_length) {
    throw std::runtime_error("rope_dim");
  }
  if (rope_base != view.f32s.end() && rope_base->second != 10000.f) {
    std::fprintf(stderr, "gguf: rope.freq_base is %g; the checkpoint stores that base.\n",
                 static_cast<double>(rope_base->second));
  }
  config.qk_norm = has_tensor(view, "blk.0.attn_q_norm.weight");
  require_config(config);
  const int dh = config_head_dim(config);
  const int q_dim = config.heads * dh;
  const int kv_dim = kv_heads * dh;

  for (int layer = 0; layer < config.layers; ++layer) {
    const std::string blk = "blk." + std::to_string(layer) + ".";
    if (config.qk_norm) {
      if (!has_tensor(view, blk + "attn_q_norm.weight") || !has_tensor(view, blk + "attn_k_norm.weight")) {
        throw std::runtime_error("qk_norm");
      }
    } else {
      require_no_tensor(view, blk + "attn_q_norm.weight", "qk_norm");
      require_no_tensor(view, blk + "attn_k_norm.weight", "qk_norm");
    }
    require_no_tensor(view, blk + "ffn_gate_exps.weight", "moe");
    require_no_tensor(view, blk + "attn_qkv.weight", "qkv");
    if (!has_tensor(view, blk + "ffn_gate.weight")) throw std::runtime_error("gelu");
    require_no_tensor(view, blk + "attn_q.bias", "bias");
    require_no_tensor(view, blk + "attn_k.bias", "bias");
    require_no_tensor(view, blk + "attn_v.bias", "bias");
    require_no_tensor(view, blk + "attn_output.bias", "bias");
    require_no_tensor(view, blk + "ffn_gate.bias", "bias");
    require_no_tensor(view, blk + "ffn_up.bias", "bias");
    require_no_tensor(view, blk + "ffn_down.bias", "bias");
  }

  const int src_vocab = has_tensor(view, "token_embd.weight") ? view.tensors.at("token_embd.weight").shape[0]
                                                             : bpe.map.new_to_orig.size();
  Tensor tok = load_public_tensor(view, data, size, "token_embd.weight", {src_vocab, config.d_model});
  tok = permute_rows(tok, bpe.map.new_to_orig, config.vocab);
  tok.name = "tok_emb.weight";

  Tensor head;
  if (has_tensor(view, "output.weight")) {
    head = load_public_tensor(view, data, size, "output.weight", {src_vocab, config.d_model});
    head = permute_rows(head, bpe.map.new_to_orig, config.vocab);
  } else {
    head = tok;
  }
  head.name = "lm_head.weight";

  Model model;
  model.config = config;
  model.tensors.push_back(std::move(tok));
  for (int layer = 0; layer < config.layers; ++layer) {
    const std::string src = "blk." + std::to_string(layer) + ".";
    const std::string dst = "blocks." + std::to_string(layer) + ".";
    Tensor ln1 = load_public_tensor(view, data, size, src + "attn_norm.weight", {config.d_model});
    ln1.name = dst + "ln1.weight";
    model.tensors.push_back(std::move(ln1));

    Tensor q = load_public_tensor(view, data, size, src + "attn_q.weight", {q_dim, config.d_model});
    q.name = dst + "attn.q.weight";
    model.tensors.push_back(std::move(q));

    Tensor k = load_public_tensor(view, data, size, src + "attn_k.weight", {kv_dim, config.d_model});
    k.name = dst + "attn.k.weight";
    model.tensors.push_back(std::move(k));

    Tensor v = load_public_tensor(view, data, size, src + "attn_v.weight", {kv_dim, config.d_model});
    v.name = dst + "attn.v.weight";
    model.tensors.push_back(std::move(v));

    Tensor o = load_public_tensor(view, data, size, src + "attn_output.weight", {config.d_model, q_dim});
    o.name = dst + "attn.out.weight";
    model.tensors.push_back(std::move(o));

    if (config.qk_norm) {
      Tensor qn = load_public_tensor(view, data, size, src + "attn_q_norm.weight", {dh});
      qn.name = dst + "attn.q_norm.weight";
      model.tensors.push_back(std::move(qn));
      Tensor kn = load_public_tensor(view, data, size, src + "attn_k_norm.weight", {dh});
      kn.name = dst + "attn.k_norm.weight";
      model.tensors.push_back(std::move(kn));
    }

    Tensor ln2 = load_public_tensor(view, data, size, src + "ffn_norm.weight", {config.d_model});
    ln2.name = dst + "ln2.weight";
    model.tensors.push_back(std::move(ln2));

    Tensor gate = load_public_tensor(view, data, size, src + "ffn_gate.weight", {config.d_ff, config.d_model});
    Tensor up = load_public_tensor(view, data, size, src + "ffn_up.weight", {config.d_ff, config.d_model});
    Tensor fc1;
    fc1.name = dst + "fc1.weight";
    fc1.shape = {config.d_ff * 2, config.d_model};
    fc1.data = gate.data;
    fc1.data.insert(fc1.data.end(), up.data.begin(), up.data.end());
    model.tensors.push_back(std::move(fc1));

    Tensor down = load_public_tensor(view, data, size, src + "ffn_down.weight", {config.d_model, config.d_ff});
    down.name = dst + "fc2.weight";
    model.tensors.push_back(std::move(down));
  }
  Tensor final_ln = load_public_tensor(view, data, size, "output_norm.weight", {config.d_model});
  final_ln.name = "final_ln.weight";
  model.tensors.push_back(std::move(final_ln));
  model.tensors.push_back(std::move(head));
  return model;
}

void convert_from_public(const std::filesystem::path& gguf_path, const std::filesystem::path& out_dir, int context) {
  MappedFile mapped(gguf_path);
  Reader reader(mapped.data(), mapped.size());
  PublicGguf view = parse_public_gguf(reader);

  if (view.architecture == kArchitecture) {
    const Model model = read_gguf(gguf_path);
    const auto sibling = gguf_path.parent_path() / "tokenizer.bpe";
    if (!std::filesystem::is_regular_file(sibling)) throw std::runtime_error("tokenizer");
    const Tokenizer tok = load_bpe(sibling);
    if (tok.vocab != model.config.vocab) throw std::runtime_error("vocab");
    std::filesystem::create_directories(out_dir);
    write_checkpoint(out_dir / "model.bin", model);
    std::filesystem::copy_file(sibling, out_dir / "tokenizer.bpe", std::filesystem::copy_options::overwrite_existing);
    BpePairs bpe;
    bpe.map.tok = tok;
    Reader bpe_reader(read_file(sibling));
    if (bpe_reader.u32() != 0x3142504cu) throw std::runtime_error("tokenizer");
    bpe_reader.u32();
    const uint32_t n_merges = bpe_reader.u32();
    bpe_reader.u32();
    bpe_reader.u32();
    bpe_reader.u32();
    for (uint32_t i = 0; i < n_merges; ++i) {
      const uint32_t left = bpe_reader.u32();
      const uint32_t right = bpe_reader.u32();
      bpe.pairs.emplace_back(left, right);
    }
    const std::vector<int32_t> ids = encode_lpb1(bpe, "User: hello\nAssistant: hello\n", true);
    write_token_files(out_dir / "tokens.bin", out_dir / "tokens.meta", ids, tok.vocab);
    return;
  }

  const auto tok_model = view.strings.find("tokenizer.ggml.model");
  if (tok_model == view.strings.end() || tok_model->second != "gpt2") throw std::runtime_error("tokenizer");
  const auto tokens = view.strs.find("tokenizer.ggml.tokens");
  const auto merges = view.strs.find("tokenizer.ggml.merges");
  if (tokens == view.strs.end() || merges == view.strs.end()) throw std::runtime_error("tokenizer");
  const int pad = static_cast<int>(optional_u32_key(view, "tokenizer.ggml.padding_token_id",
                                                   tokens->second.empty() ? 0 : static_cast<uint32_t>(tokens->second.size() - 3)));
  const int bos = static_cast<int>(optional_u32_key(view, "tokenizer.ggml.bos_token_id",
                                                   tokens->second.empty() ? 0 : static_cast<uint32_t>(tokens->second.size() - 2)));
  const int eos = static_cast<int>(optional_u32_key(view, "tokenizer.ggml.eos_token_id",
                                                   tokens->second.empty() ? 0 : static_cast<uint32_t>(tokens->second.size() - 1)));
  const BpePairs bpe = convert_gpt2_bpe(tokens->second, merges->second, pad, bos, eos);
  Model model = public_to_nnllm(view, mapped.data(), mapped.size(), bpe, context);
  std::filesystem::create_directories(out_dir);
  write_bpe_file(out_dir / "tokenizer.bpe", bpe);
  const std::vector<int32_t> ids = encode_lpb1(bpe, "User: hello\nAssistant: hello\n", true);
  write_token_files(out_dir / "tokens.bin", out_dir / "tokens.meta", ids, bpe.map.tok.vocab);
  write_checkpoint(out_dir / "model.bin", model);
}

void write_q8_0(Bytes& out, const std::vector<float>& values) {
  if (values.size() % static_cast<size_t>(kQK) != 0) throw std::runtime_error("self-test q8");
  const size_t nb = values.size() / static_cast<size_t>(kQK);
  for (size_t i = 0; i < nb; ++i) {
    float amax = 0.f;
    for (int j = 0; j < kQK; ++j) {
      const float a = values[i * static_cast<size_t>(kQK) + static_cast<size_t>(j)];
      if (a > amax || -a > amax) amax = a < 0.f ? -a : a;
    }
    const float d = amax / 127.f;
    const float id = d != 0.f ? 1.f / d : 0.f;
    out.u16(f32_to_f16(d));
    for (int j = 0; j < kQK; ++j) {
      const float x = values[i * static_cast<size_t>(kQK) + static_cast<size_t>(j)] * id;
      int q = static_cast<int>(x >= 0.f ? x + 0.5f : x - 0.5f);
      if (q > 127) q = 127;
      if (q < -128) q = -128;
      out.u8(static_cast<uint8_t>(q));
    }
  }
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
  {
    Reader ver(read_file(rope_bin));
    if (ver.u32() != kCheckpointMagic || ver.u32() != 1u) throw std::runtime_error("self-test rope version");
  }
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

  convert_from_public(tok_gguf, dir / "from-out", 16);
  const Tokenizer from_tok = load_bpe(dir / "from-out" / "tokenizer.bpe");
  if (from_tok.vocab != 260 || from_tok.tokens[97] != "a" || from_tok.tokens[256] != "ab" ||
      from_tok.merges.size() != 1 || from_tok.merges[0] != "a b") {
    throw std::runtime_error("self-test from tokenizer");
  }
  const Model from_loaded = read_checkpoint(dir / "from-out" / "model.bin");
  if (!from_loaded.config.rope || from_loaded.config.vocab != 260 || from_loaded.tensors.size() != tok_model.tensors.size()) {
    throw std::runtime_error("self-test from model");
  }
  for (size_t i = 0; i < tok_model.tensors.size(); ++i) {
    if (from_loaded.tensors[i].name != tok_model.tensors[i].name ||
        from_loaded.tensors[i].shape != tok_model.tensors[i].shape ||
        from_loaded.tensors[i].data != tok_model.tensors[i].data) {
      throw std::runtime_error("self-test from " + tok_model.tensors[i].name);
    }
  }
  if (!std::filesystem::is_regular_file(dir / "from-out" / "tokens.bin") ||
      !std::filesystem::is_regular_file(dir / "from-out" / "tokens.meta")) {
    throw std::runtime_error("self-test from tokens");
  }

  Config qwen;
  qwen.vocab = 260;
  qwen.d_model = 8;
  qwen.heads = 4;
  qwen.layers = 1;
  qwen.d_ff = 8;
  qwen.max_seq = 16;
  qwen.dropout = 0.f;
  qwen.bias = false;
  qwen.swiglu = true;
  qwen.rope = true;
  qwen.kv_heads = 2;
  qwen.head_dim = 6;
  qwen.qk_norm = true;
  qwen.rope_theta = 1000000.f;
  qwen.rms_eps = 1e-6f;
  Model qwen_model;
  qwen_model.config = qwen;
  qwen_model.tensors = expected_tensors(qwen);
  uint32_t qwen_n = 1;
  for (Tensor& tensor : qwen_model.tensors) {
    const int64_t numel = product(tensor.shape);
    tensor.data.assign(static_cast<size_t>(numel), 0.f);
    if (!tensor.data.empty()) {
      const uint32_t bits = 0x3f800000u + qwen_n;
      std::memcpy(&tensor.data[0], &bits, sizeof(bits));
    }
    ++qwen_n;
  }
  const auto qwen_bin = dir / "qwen.bin";
  write_checkpoint(qwen_bin, qwen_model);
  {
    Reader ver(read_file(qwen_bin));
    if (ver.u32() != kCheckpointMagic || ver.u32() != 3u) throw std::runtime_error("self-test qwen version");
  }
  const Model qwen_loaded = read_checkpoint(qwen_bin);
  if (qwen_loaded.config.head_dim != 6 || qwen_loaded.config.kv_heads != 2 || !qwen_loaded.config.qk_norm ||
      qwen_loaded.config.rope_theta != 1000000.f || qwen_loaded.config.rms_eps != 1e-6f || !qwen_loaded.config.rope ||
      qwen_loaded.tensors.size() != qwen_model.tensors.size()) {
    throw std::runtime_error("self-test qwen config");
  }
  for (size_t i = 0; i < qwen_model.tensors.size(); ++i) {
    if (qwen_loaded.tensors[i].name != qwen_model.tensors[i].name ||
        qwen_loaded.tensors[i].shape != qwen_model.tensors[i].shape ||
        qwen_loaded.tensors[i].data != qwen_model.tensors[i].data) {
      throw std::runtime_error("self-test qwen " + qwen_model.tensors[i].name);
    }
  }
  write_gguf(dir / "qwen.gguf", qwen_loaded);
  const Model qwen_from_gguf = read_gguf(dir / "qwen.gguf");
  if (qwen_from_gguf.config.head_dim != 6 || qwen_from_gguf.config.kv_heads != 2 || !qwen_from_gguf.config.qk_norm ||
      qwen_from_gguf.config.rope_theta != 1000000.f || qwen_from_gguf.config.rms_eps != 1e-6f) {
    throw std::runtime_error("self-test qwen gguf");
  }
  const auto qwen_llama = dir / "qwen-llama.gguf";
  write_llama_gguf(qwen_llama, qwen_model, &tok);
  const GgufView qwen_view = read_gguf_view(qwen_llama);
  const auto qarch = qwen_view.strings.find("general.architecture");
  if (qarch == qwen_view.strings.end() || qarch->second != "qwen3") throw std::runtime_error("self-test qwen arch");
  auto need_qu = [&](const char* key, uint32_t want) {
    const auto found = qwen_view.u32s.find(key);
    if (found == qwen_view.u32s.end() || found->second != want) {
      throw std::runtime_error(std::string("self-test qwen ") + key);
    }
  };
  need_qu("qwen3.attention.head_count", 4);
  need_qu("qwen3.attention.head_count_kv", 2);
  need_qu("qwen3.attention.key_length", 6);
  need_qu("qwen3.attention.value_length", 6);
  need_qu("qwen3.rope.dimension_count", 6);
  const auto qbase = qwen_view.f32s.find("qwen3.rope.freq_base");
  const auto qeps = qwen_view.f32s.find("qwen3.attention.layer_norm_rms_epsilon");
  if (qbase == qwen_view.f32s.end() || qbase->second != 1000000.f || qeps == qwen_view.f32s.end() ||
      qeps->second != 1e-6f) {
    throw std::runtime_error("self-test qwen rope");
  }
  auto need_dims = [&](const char* name, std::vector<uint64_t> dims) {
    const auto found = qwen_view.tensors.find(name);
    if (found == qwen_view.tensors.end() || found->second.dims != dims) {
      throw std::runtime_error(std::string("self-test qwen ") + name);
    }
  };
  need_dims("blk.0.attn_q.weight", {8, 24});
  need_dims("blk.0.attn_k.weight", {8, 12});
  need_dims("blk.0.attn_v.weight", {8, 12});
  need_dims("blk.0.attn_output.weight", {24, 8});
  need_dims("blk.0.attn_q_norm.weight", {6});
  need_dims("blk.0.attn_k_norm.weight", {6});
  convert_from_public(qwen_llama, dir / "qwen-from", 16);
  const Model qwen_back = read_checkpoint(dir / "qwen-from" / "model.bin");
  if (qwen_back.config.head_dim != 6 || qwen_back.config.kv_heads != 2 || !qwen_back.config.qk_norm ||
      qwen_back.config.rope_theta != 1000000.f || qwen_back.config.rms_eps != 1e-6f ||
      qwen_back.tensors.size() != qwen_model.tensors.size()) {
    throw std::runtime_error("self-test qwen from");
  }
  for (size_t i = 0; i < qwen_model.tensors.size(); ++i) {
    if (qwen_back.tensors[i].name != qwen_model.tensors[i].name ||
        qwen_back.tensors[i].shape != qwen_model.tensors[i].shape ||
        qwen_back.tensors[i].data != qwen_model.tensors[i].data) {
      throw std::runtime_error("self-test qwen from " + qwen_model.tensors[i].name);
    }
  }

  Config qwen2 = qwen;
  qwen2.qk_norm = false;
  qwen2.head_dim = 0;
  Model qwen2_model;
  qwen2_model.config = qwen2;
  qwen2_model.tensors = expected_tensors(qwen2);
  for (Tensor& tensor : qwen2_model.tensors) {
    const int64_t numel = product(tensor.shape);
    tensor.data.assign(static_cast<size_t>(numel), 0.f);
  }
  const auto qwen2_llama = dir / "qwen2-llama.gguf";
  write_llama_gguf(qwen2_llama, qwen2_model);
  const GgufView qwen2_view = read_gguf_view(qwen2_llama);
  const auto q2arch = qwen2_view.strings.find("general.architecture");
  if (q2arch == qwen2_view.strings.end() || q2arch->second != "qwen2") {
    throw std::runtime_error("self-test qwen2 arch");
  }
  const auto q2k = qwen2_view.tensors.find("blk.0.attn_k.weight");
  if (q2k == qwen2_view.tensors.end() || q2k->second.dims != std::vector<uint64_t>{8, 4}) {
    throw std::runtime_error("self-test qwen2 k");
  }
  if (qwen2_view.tensors.find("blk.0.attn_q_norm.weight") != qwen2_view.tensors.end()) {
    throw std::runtime_error("self-test qwen2 norm");
  }

  std::vector<float> q8_src(32);
  for (int i = 0; i < 32; ++i) q8_src[static_cast<size_t>(i)] = static_cast<float>(i - 8) * 0.25f;
  Bytes q8_bytes;
  write_q8_0(q8_bytes, q8_src);
  std::vector<float> q8_back;
  dequantize(kGgmlQ8_0, q8_bytes.bytes().data(), 32, q8_back);
  if (q8_back.size() != 32) throw std::runtime_error("self-test q8 size");
  for (int i = 0; i < 32; ++i) {
    const float err = q8_back[static_cast<size_t>(i)] - q8_src[static_cast<size_t>(i)];
    if (err > 0.03f || err < -0.03f) throw std::runtime_error("self-test q8");
  }

  Tensor kv_weight;
  kv_weight.name = "k";
  kv_weight.shape = {4, 8};
  kv_weight.data.resize(32);
  for (int i = 0; i < 32; ++i) kv_weight.data[static_cast<size_t>(i)] = static_cast<float>(i + 1);
  const Tensor kv_rep = repeat_kv_heads(kv_weight, 4, 2, 8);
  if (kv_rep.shape != std::vector<int>{8, 8}) throw std::runtime_error("self-test gqa shape");
  for (int d = 0; d < 2; ++d) {
    for (int col = 0; col < 8; ++col) {
      const float a = kv_weight.data[static_cast<size_t>(d) * 8 + static_cast<size_t>(col)];
      if (kv_rep.data[static_cast<size_t>(d) * 8 + static_cast<size_t>(col)] != a ||
          kv_rep.data[static_cast<size_t>(2 + d) * 8 + static_cast<size_t>(col)] != a) {
        throw std::runtime_error("self-test gqa");
      }
    }
  }

  const std::array<std::string, 256> byte_tok = gpt2_byte_tokens();
  std::vector<int> gpt2_order;
  std::array<bool, 256> used{};
  auto push_byte = [&](int b) {
    gpt2_order.push_back(b);
    used[static_cast<size_t>(b)] = true;
  };
  for (int b = 0x21; b <= 0x7e; ++b) push_byte(b);
  for (int b = 0xa1; b <= 0xac; ++b) push_byte(b);
  for (int b = 0xae; b <= 0xff; ++b) push_byte(b);
  for (int b = 0; b < 256; ++b) {
    if (!used[static_cast<size_t>(b)]) gpt2_order.push_back(b);
  }
  std::vector<std::string> shuffled;
  shuffled.reserve(259);
  for (int b : gpt2_order) shuffled.push_back(byte_tok[static_cast<size_t>(b)]);
  shuffled.push_back("<pad>");
  shuffled.push_back("<bos>");
  shuffled.push_back("<|endoftext|>");
  const BpePairs remapped = convert_gpt2_bpe(shuffled, {}, 256, 257, 258);
  if (remapped.map.tok.vocab != 259 || !remapped.pairs.empty() || gpt2_order[0] != 33 ||
      remapped.map.orig_to_new[0] != 33 || remapped.map.new_to_orig[33] != 0 ||
      remapped.map.orig_to_new[256] != 256 || remapped.map.new_to_orig[256] != 256) {
    throw std::runtime_error("self-test remap");
  }

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
    if (command == "from") {
      if (argc < 4) {
        usage();
        throw std::runtime_error("usage");
      }
      int context = 0;
      for (int i = 4; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--context") {
          if (i + 1 >= argc) throw std::runtime_error("usage");
          ++i;
          char* end = nullptr;
          const long value = std::strtol(argv[i], &end, 10);
          if (end == argv[i] || *end != '\0' || value < 1 || value > 0x7fffffff) {
            throw std::runtime_error("context");
          }
          context = static_cast<int>(value);
        } else {
          throw std::runtime_error(argv[i]);
        }
      }
      convert_from_public(argv[2], argv[3], context);
      return 0;
    }
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
