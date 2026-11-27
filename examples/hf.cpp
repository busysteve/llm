#include "json.hpp"

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t kCheckpointMagic = 0x314E4E43u;
constexpr uint64_t kMaxHeader = 64ull << 20;

using llm_example::Json;
using llm_example::parse_json;

const Json& need_key(const Json& obj, const char* key) {
  const Json* found = obj.find(key);
  if (!found) throw std::runtime_error("config");
  return *found;
}

int as_i32(const Json& value, const char* reason, int min_value) {
  if (value.kind != Json::Kind::Number || !std::isfinite(value.n) || std::floor(value.n) != value.n) {
    throw std::runtime_error(reason);
  }
  if (value.n < min_value || value.n > 2147483647.0) throw std::runtime_error(reason);
  return static_cast<int>(value.n);
}

float as_f32(const Json& value, const char* reason) {
  if (value.kind != Json::Kind::Number || !std::isfinite(value.n)) throw std::runtime_error(reason);
  return static_cast<float>(value.n);
}

std::string read_text(const std::filesystem::path& path) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("read");
  std::string out;
  char buf[1 << 15];
  while (true) {
    const ssize_t n = ::read(fd, buf, sizeof(buf));
    if (n < 0) {
      if (errno == EINTR) continue;
      ::close(fd);
      throw std::runtime_error("read");
    }
    if (n == 0) break;
    out.append(buf, static_cast<size_t>(n));
  }
  ::close(fd);
  return out;
}

std::vector<uint8_t> pread_exact(int fd, uint64_t offset, size_t n) {
  std::vector<uint8_t> buf(n);
  size_t got = 0;
  while (got < n) {
    const ssize_t r = ::pread(fd, buf.data() + got, n - got, static_cast<off_t>(offset + got));
    if (r < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error("read");
    }
    if (r == 0) throw std::runtime_error("read");
    got += static_cast<size_t>(r);
  }
  return buf;
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

float bf16_to_f32(uint16_t bits) {
  const uint32_t out = static_cast<uint32_t>(bits) << 16;
  float value = 0.f;
  std::memcpy(&value, &out, sizeof(value));
  return value;
}

struct StTensor {
  std::string dtype;
  std::vector<int64_t> shape;
  uint64_t begin = 0;
  uint64_t end = 0;
  int fd = -1;
  uint64_t data_off = 0;
};

bool qkv_proj_bias(const std::string& name) {
  const std::string prefix = "model.layers.";
  const std::string marker = ".self_attn.";
  if (name.compare(0, prefix.size(), prefix) != 0) return false;
  const size_t mark = name.rfind(marker);
  if (mark == std::string::npos || mark <= prefix.size()) return false;
  for (size_t i = prefix.size(); i < mark; ++i) {
    if (name[i] < '0' || name[i] > '9') return false;
  }
  const std::string tail = name.substr(mark + marker.size());
  return tail == "q_proj.bias" || tail == "k_proj.bias" || tail == "v_proj.bias";
}

bool moe_name(const std::string& name) {
  return name.find(".experts") != std::string::npos || name.find("block_sparse_moe") != std::string::npos ||
         name.find(".shared_expert") != std::string::npos || name.find(".router") != std::string::npos ||
         name.find(".mlp.gate.weight") != std::string::npos;
}

bool ignored_name(const std::string& name) {
  const std::string suffix = "inv_freq";
  return name.size() >= suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
}

int64_t shape_product(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t dim : shape) {
    if (dim < 1 || n > INT64_MAX / dim) throw std::runtime_error("tensor");
    n *= dim;
  }
  return n;
}

class Bundle {
 public:
  explicit Bundle(const std::filesystem::path& dir) {
    const auto index_path = dir / "model.safetensors.index.json";
    if (std::filesystem::is_regular_file(index_path)) {
      const Json index = parse_json(read_text(index_path));
      const Json& map = need_key(index, "weight_map");
      if (map.kind != Json::Kind::Object || map.o.empty()) throw std::runtime_error("tensor");
      std::map<std::string, int> seen;
      for (const auto& entry : map.o) {
        if (entry.second.kind != Json::Kind::String) throw std::runtime_error("json");
        seen.emplace(entry.second.s, 0);
      }
      for (const auto& entry : seen) load_shard(dir / entry.first);
    } else if (std::filesystem::is_regular_file(dir / "model.safetensors")) {
      load_shard(dir / "model.safetensors");
    } else {
      throw std::runtime_error("read");
    }
  }

  ~Bundle() {
    for (int fd : fds_) {
      if (fd >= 0) ::close(fd);
    }
  }

  Bundle(const Bundle&) = delete;
  Bundle& operator=(const Bundle&) = delete;

  std::vector<float> take(const std::string& name, const std::vector<int64_t>& want) {
    const auto it = tensors_.find(name);
    if (it == tensors_.end() || it->second.used || it->second.tensor.shape != want) throw std::runtime_error("tensor");
    it->second.used = true;
    return read_tensor(it->second.tensor);
  }

  bool contains(const std::string& name) const { return tensors_.find(name) != tensors_.end(); }

  void reject_bias(bool allow_qkv) const {
    for (const auto& entry : tensors_) {
      const std::string& name = entry.first;
      if (name.size() < 5 || name.compare(name.size() - 5, 5, ".bias") != 0) continue;
      if (allow_qkv && qkv_proj_bias(name)) continue;
      throw std::runtime_error("bias");
    }
  }

  void finish() const {
    for (const auto& entry : tensors_) {
      if (entry.second.used || ignored_name(entry.first)) continue;
      throw std::runtime_error("tensor");
    }
  }

 private:
  struct Stored {
    StTensor tensor;
    bool used = false;
  };

  std::map<std::string, Stored> tensors_;
  std::vector<int> fds_;

  void load_shard(const std::filesystem::path& path) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("read");
    fds_.push_back(fd);
    const std::vector<uint8_t> len_bytes = pread_exact(fd, 0, 8);
    uint64_t header_len = 0;
    for (int i = 0; i < 8; ++i) header_len |= static_cast<uint64_t>(len_bytes[static_cast<size_t>(i)]) << (8 * i);
    if (header_len == 0 || header_len > kMaxHeader) throw std::runtime_error("read");
    const std::vector<uint8_t> header_bytes = pread_exact(fd, 8, static_cast<size_t>(header_len));
    const std::string header(reinterpret_cast<const char*>(header_bytes.data()), header_bytes.size());
    const Json root = parse_json(header);
    if (root.kind != Json::Kind::Object) throw std::runtime_error("json");
    for (const auto& entry : root.o) {
      if (entry.first == "__metadata__") continue;
      if (moe_name(entry.first)) throw std::runtime_error("moe");
      if (entry.second.kind != Json::Kind::Object) throw std::runtime_error("json");
      const Json& dtype = need_key(entry.second, "dtype");
      if (dtype.kind != Json::Kind::String) throw std::runtime_error("json");
      if (dtype.s != "F32" && dtype.s != "F16" && dtype.s != "BF16") throw std::runtime_error("dtype");
      const Json& shape = need_key(entry.second, "shape");
      const Json& offsets = need_key(entry.second, "data_offsets");
      if (shape.kind != Json::Kind::Array || offsets.kind != Json::Kind::Array || offsets.a.size() != 2) {
        throw std::runtime_error("tensor");
      }
      StTensor tensor;
      tensor.dtype = dtype.s;
      tensor.fd = fd;
      tensor.data_off = 8 + header_len;
      for (const Json& dim : shape.a) {
        const int dim_i = as_i32(dim, "tensor", 1);
        tensor.shape.push_back(dim_i);
      }
      if (tensor.shape.size() != 1 && tensor.shape.size() != 2) throw std::runtime_error("tensor");
      const double begin = offsets.a[0].kind == Json::Kind::Number ? offsets.a[0].n : -1;
      const double end = offsets.a[1].kind == Json::Kind::Number ? offsets.a[1].n : -1;
      if (std::floor(begin) != begin || std::floor(end) != end || begin < 0 || end < begin) {
        throw std::runtime_error("tensor");
      }
      tensor.begin = static_cast<uint64_t>(begin);
      tensor.end = static_cast<uint64_t>(end);
      const int64_t numel = shape_product(tensor.shape);
      const uint64_t width = tensor.dtype == "F32" ? 4u : 2u;
      if (tensor.end - tensor.begin != static_cast<uint64_t>(numel) * width) throw std::runtime_error("tensor");
      if (!tensors_.emplace(entry.first, Stored{std::move(tensor), false}).second) throw std::runtime_error("tensor");
    }
  }

  static std::vector<float> read_tensor(const StTensor& tensor) {
    const int64_t numel = shape_product(tensor.shape);
    if (numel > 0xffffffffll) throw std::runtime_error("tensor");
    const std::vector<uint8_t> raw = pread_exact(tensor.fd, tensor.data_off + tensor.begin,
                                                 static_cast<size_t>(tensor.end - tensor.begin));
    std::vector<float> out(static_cast<size_t>(numel));
    if (tensor.dtype == "F32") {
      for (int64_t i = 0; i < numel; ++i) {
        float value = 0.f;
        std::memcpy(&value, raw.data() + static_cast<size_t>(i) * 4, 4);
        out[static_cast<size_t>(i)] = value;
      }
    } else if (tensor.dtype == "F16") {
      for (int64_t i = 0; i < numel; ++i) {
        uint16_t bits = 0;
        std::memcpy(&bits, raw.data() + static_cast<size_t>(i) * 2, 2);
        out[static_cast<size_t>(i)] = f16_to_f32(bits);
      }
    } else {
      for (int64_t i = 0; i < numel; ++i) {
        uint16_t bits = 0;
        std::memcpy(&bits, raw.data() + static_cast<size_t>(i) * 2, 2);
        out[static_cast<size_t>(i)] = bf16_to_f32(bits);
      }
    }
    return out;
  }
};

// Hugging Face rotate-half layout is (head, 2, head_dim/2, hidden).
// Adjacent-pair RoPE reads (head, head_dim/2, 2, hidden).
std::vector<float> interleave_heads(const std::vector<float>& src, int n_heads, int head_dim, int hidden) {
  const int half = head_dim / 2;
  std::vector<float> dst(src.size());
  for (int head = 0; head < n_heads; ++head) {
    for (int part = 0; part < 2; ++part) {
      for (int pair = 0; pair < half; ++pair) {
        const size_t src_row = (static_cast<size_t>(head) * 2 + static_cast<size_t>(part)) * static_cast<size_t>(half) +
                               static_cast<size_t>(pair);
        const size_t dst_row = (static_cast<size_t>(head) * static_cast<size_t>(half) + static_cast<size_t>(pair)) * 2 +
                               static_cast<size_t>(part);
        std::memcpy(dst.data() + dst_row * static_cast<size_t>(hidden),
                    src.data() + src_row * static_cast<size_t>(hidden),
                    static_cast<size_t>(hidden) * sizeof(float));
      }
    }
  }
  return dst;
}

struct ModelConfig {
  int vocab = 0;
  int d_model = 0;
  int heads = 0;
  int layers = 0;
  int d_ff = 0;
  int max_seq = 0;
  int head_dim = 0;
  int kv_heads = 0;
  int resolved_head = 0;
  int resolved_kv = 0;
  bool qk_norm = false;
  float rope_theta = 10000.f;
  float rms_eps = 1e-5f;
  bool tie = false;
  bool qwen3 = false;
  bool qwen2 = false;
  bool qkv_bias = false;
};

bool extended_config(const ModelConfig& config) {
  const bool wide = config.head_dim > 0 &&
                    (config.d_model % config.heads != 0 || config.head_dim != config.d_model / config.heads);
  const bool grouped = config.kv_heads > 0 && config.kv_heads != config.heads;
  return wide || grouped || config.qk_norm || config.rope_theta != 10000.f || config.rms_eps != 1e-5f;
}

void reject_optional_object(const Json& cfg, const char* key, const char* reason) {
  const Json* value = cfg.find(key);
  if (value && value->kind != Json::Kind::Null) throw std::runtime_error(reason);
}

ModelConfig read_config(const Json& cfg, std::optional<int> context) {
  const Json& model_type = need_key(cfg, "model_type");
  if (model_type.kind != Json::Kind::String) throw std::runtime_error("architecture");
  ModelConfig config;
  if (model_type.s == "qwen3") config.qwen3 = true;
  else if (model_type.s == "qwen2") config.qwen2 = true;
  else if (model_type.s != "llama") throw std::runtime_error("architecture");

  const Json* act = cfg.find("hidden_act");
  if (act && act->kind != Json::Kind::Null) {
    if (act->kind != Json::Kind::String || act->s != "silu") throw std::runtime_error("activation");
  }
  reject_optional_object(cfg, "rope_scaling", "rope");
  const Json* partial = cfg.find("partial_rotary_factor");
  if (partial && partial->kind != Json::Kind::Null && as_f32(*partial, "rope") != 1.f) {
    throw std::runtime_error("rope");
  }
  const Json* attention_bias = cfg.find("attention_bias");
  if (attention_bias && attention_bias->kind != Json::Kind::Null) {
    if (attention_bias->kind != Json::Kind::Bool) throw std::runtime_error("bias");
    if (attention_bias->b && !config.qwen2) throw std::runtime_error("bias");
    config.qkv_bias = attention_bias->b;
  }
  bool window_off = false;
  const Json* use_window = cfg.find("use_sliding_window");
  if (use_window && use_window->kind != Json::Kind::Null) {
    if (use_window->kind != Json::Kind::Bool) throw std::runtime_error("window");
    if (use_window->b) throw std::runtime_error("window");
    window_off = true;
  }
  if (!window_off) {
    const Json* window = cfg.find("sliding_window");
    if (window && window->kind != Json::Kind::Null && as_i32(*window, "window", 0) > 0) {
      throw std::runtime_error("window");
    }
  }

  config.vocab = as_i32(need_key(cfg, "vocab_size"), "config", 1);
  config.d_model = as_i32(need_key(cfg, "hidden_size"), "config", 1);
  config.heads = as_i32(need_key(cfg, "num_attention_heads"), "config", 1);
  config.layers = as_i32(need_key(cfg, "num_hidden_layers"), "config", 1);
  config.d_ff = as_i32(need_key(cfg, "intermediate_size"), "config", 1);
  config.max_seq = context ? *context : as_i32(need_key(cfg, "max_position_embeddings"), "config", 1);
  if (config.layers > 100000 || config.d_ff > 2147483647 / 2) throw std::runtime_error("config");

  const Json* head = cfg.find("head_dim");
  int resolved = 0;
  if (head && head->kind != Json::Kind::Null) {
    resolved = as_i32(*head, "head_dim", 1);
  } else {
    if (config.d_model % config.heads != 0) throw std::runtime_error("head_dim");
    resolved = config.d_model / config.heads;
  }
  if (resolved % 2 != 0) throw std::runtime_error("rope");
  config.resolved_head = resolved;
  config.head_dim = (config.d_model % config.heads == 0 && resolved == config.d_model / config.heads) ? 0 : resolved;

  const Json* kv = cfg.find("num_key_value_heads");
  const int kv_count = kv && kv->kind != Json::Kind::Null ? as_i32(*kv, "head_count_kv", 1) : config.heads;
  if (kv_count > config.heads || config.heads % kv_count != 0) throw std::runtime_error("head_count_kv");
  config.resolved_kv = kv_count;
  config.kv_heads = kv_count == config.heads ? 0 : kv_count;

  const Json* theta = cfg.find("rope_theta");
  if (theta && theta->kind != Json::Kind::Null) config.rope_theta = as_f32(*theta, "rope");
  const Json* eps = cfg.find("rms_norm_eps");
  if (eps && eps->kind != Json::Kind::Null) config.rms_eps = as_f32(*eps, "config");
  if (!(config.rope_theta > 0.f) || !std::isfinite(config.rope_theta)) throw std::runtime_error("rope");
  if (!(config.rms_eps > 0.f) || !std::isfinite(config.rms_eps)) throw std::runtime_error("config");
  config.qk_norm = config.qwen3;

  const Json* tie = cfg.find("tie_word_embeddings");
  if (tie && tie->kind != Json::Kind::Null) {
    if (tie->kind != Json::Kind::Bool) throw std::runtime_error("config");
    config.tie = tie->b;
  }
  if (resolved > 2147483647 / config.heads || resolved > 2147483647 / config.resolved_kv) {
    throw std::runtime_error("head_dim");
  }
  return config;
}

class OutFile {
 public:
  explicit OutFile(std::filesystem::path dest)
      : dest_(std::move(dest)), tmp_(dest_.string() + ".tmp") {
    fd_ = ::open(tmp_.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (fd_ < 0) throw std::runtime_error("write");
  }

  ~OutFile() {
    if (fd_ >= 0) ::close(fd_);
    if (!committed_) ::unlink(tmp_.c_str());
  }

  OutFile(const OutFile&) = delete;
  OutFile& operator=(const OutFile&) = delete;

  void u32(uint32_t value) {
    uint8_t bytes[4] = {static_cast<uint8_t>(value), static_cast<uint8_t>(value >> 8),
                        static_cast<uint8_t>(value >> 16), static_cast<uint8_t>(value >> 24)};
    raw(bytes, 4);
  }
  void u64(uint64_t value) {
    u32(static_cast<uint32_t>(value));
    u32(static_cast<uint32_t>(value >> 32));
  }
  void i32(int value) { u32(static_cast<uint32_t>(value)); }
  void f32(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    u32(bits);
  }
  void raw(const uint8_t* data, size_t n) {
    if (n >= (1u << 20)) {
      flush();
      size_t off = 0;
      while (off < n) {
        const ssize_t wrote = ::write(fd_, data + off, n - off);
        if (wrote < 0) {
          if (errno == EINTR) continue;
          throw std::runtime_error("write");
        }
        if (wrote == 0) throw std::runtime_error("write");
        off += static_cast<size_t>(wrote);
      }
      written_ += n;
      return;
    }
    buf_.insert(buf_.end(), data, data + n);
    if (buf_.size() >= (1u << 20)) flush();
  }
  void f32s(const std::vector<float>& values) {
    raw(reinterpret_cast<const uint8_t*>(values.data()), values.size() * sizeof(float));
  }
  void str(const std::string& value) {
    if (value.size() > 256) throw std::runtime_error("tensor");
    u32(static_cast<uint32_t>(value.size()));
    raw(reinterpret_cast<const uint8_t*>(value.data()), value.size());
    pad8();
  }
  void pad8() {
    while (pos() % 8 != 0) {
      const uint8_t zero = 0;
      raw(&zero, 1);
    }
  }
  void tensor(const std::string& name, const std::vector<int>& shape, const std::vector<float>& values) {
    int64_t numel = 1;
    for (int dim : shape) {
      if (dim < 1 || numel > 0xffffffffll / dim) throw std::runtime_error("tensor");
      numel *= dim;
    }
    if (static_cast<size_t>(numel) != values.size()) throw std::runtime_error("tensor");
    str(name);
    u32(static_cast<uint32_t>(shape.size()));
    for (int dim : shape) i32(dim);
    pad8();
    u32(1);
    u32(static_cast<uint32_t>(numel));
    f32s(values);
    pad8();
  }
  void commit() {
    flush();
    if (::fsync(fd_) != 0) throw std::runtime_error("write");
    if (::close(fd_) != 0) {
      fd_ = -1;
      throw std::runtime_error("write");
    }
    fd_ = -1;
    if (::rename(tmp_.c_str(), dest_.c_str()) != 0) throw std::runtime_error("write");
    committed_ = true;
  }

 private:
  uint64_t pos() const { return written_ + buf_.size(); }
  void flush() {
    size_t off = 0;
    while (off < buf_.size()) {
      const ssize_t n = ::write(fd_, buf_.data() + off, buf_.size() - off);
      if (n < 0) {
        if (errno == EINTR) continue;
        throw std::runtime_error("write");
      }
      if (n == 0) throw std::runtime_error("write");
      off += static_cast<size_t>(n);
    }
    written_ += buf_.size();
    buf_.clear();
  }

  std::filesystem::path dest_;
  std::filesystem::path tmp_;
  int fd_ = -1;
  bool committed_ = false;
  uint64_t written_ = 0;
  std::vector<uint8_t> buf_;
};

void copy_beside(const std::filesystem::path& from, const std::filesystem::path& to) {
  if (!std::filesystem::is_regular_file(from)) return;
  if (std::filesystem::exists(to)) {
    std::error_code same;
    if (std::filesystem::equivalent(from, to, same) && !same) return;
  }
  std::error_code error;
  std::filesystem::copy_file(from, to, std::filesystem::copy_options::overwrite_existing, error);
  if (error) throw std::runtime_error("write");
}

void import_snapshot(const std::filesystem::path& model_dir, const std::filesystem::path& out_dir,
                     std::optional<int> context) {
  const Json cfg = parse_json(read_text(model_dir / "config.json"));
  if (cfg.kind != Json::Kind::Object) throw std::runtime_error("json");
  ModelConfig config = read_config(cfg, context);
  const char* required = (config.qwen3 || config.qwen2) ? "tokenizer.json" : "tokenizer.model";
  if (!std::filesystem::is_regular_file(model_dir / required)) throw std::runtime_error("tokenizer");

  Bundle bundle(model_dir);
  const Json* attention_bias = cfg.find("attention_bias");
  if (config.qwen2 && (!attention_bias || attention_bias->kind == Json::Kind::Null)) {
    config.qkv_bias = bundle.contains("model.layers.0.self_attn.q_proj.bias") ||
                      bundle.contains("model.layers.0.self_attn.k_proj.bias") ||
                      bundle.contains("model.layers.0.self_attn.v_proj.bias");
  }
  bundle.reject_bias(config.qkv_bias);
  const int q_dim = config.heads * config.resolved_head;
  const int kv_dim = config.resolved_kv * config.resolved_head;
  if (q_dim < 1 || kv_dim < 1) throw std::runtime_error("head_dim");

  std::vector<std::pair<std::string, std::vector<int>>> slots;
  auto add = [&](std::string name, std::vector<int> shape) {
    slots.push_back({std::move(name), std::move(shape)});
  };
  add("tok_emb.weight", {config.vocab, config.d_model});
  for (int layer = 0; layer < config.layers; ++layer) {
    const std::string prefix = "blocks." + std::to_string(layer) + ".";
    add(prefix + "ln1.weight", {config.d_model});
    add(prefix + "attn.q.weight", {q_dim, config.d_model});
    if (config.qkv_bias) add(prefix + "attn.q.bias", {q_dim});
    add(prefix + "attn.k.weight", {kv_dim, config.d_model});
    if (config.qkv_bias) add(prefix + "attn.k.bias", {kv_dim});
    add(prefix + "attn.v.weight", {kv_dim, config.d_model});
    if (config.qkv_bias) add(prefix + "attn.v.bias", {kv_dim});
    add(prefix + "attn.out.weight", {config.d_model, q_dim});
    if (config.qk_norm) {
      add(prefix + "attn.q_norm.weight", {config.resolved_head});
      add(prefix + "attn.k_norm.weight", {config.resolved_head});
    }
    add(prefix + "ln2.weight", {config.d_model});
    add(prefix + "fc1.weight", {config.d_ff * 2, config.d_model});
    add(prefix + "fc2.weight", {config.d_model, config.d_ff});
  }
  add("final_ln.weight", {config.d_model});
  add("lm_head.weight", {config.vocab, config.d_model});

  std::error_code mkdir_error;
  std::filesystem::create_directories(out_dir, mkdir_error);
  if (mkdir_error) throw std::runtime_error("write");

  OutFile out(out_dir / "model.bin");
  const bool ext = extended_config(config);
  const uint32_t version = config.qkv_bias ? 7u : (ext ? 3u : 1u);
  out.u32(kCheckpointMagic);
  out.u32(version);
  out.u32(6);
  out.u32(0);
  out.u64(0);
  out.u32(0);
  out.u32(0);
  out.i32(config.vocab);
  out.i32(config.d_model);
  out.i32(config.heads);
  out.i32(config.layers);
  out.i32(config.d_ff);
  out.i32(config.max_seq);
  out.f32(0.f);
  out.u32(0);
  out.u32(0);
  out.u32(2);
  out.u32(2);
  if (version >= 7) {
    out.i32(config.head_dim);
    out.i32(config.kv_heads);
    out.u32(config.qk_norm ? 1u : 0u);
    out.f32(config.rope_theta);
    out.f32(config.rms_eps);
    out.u32(1);
    out.i32(0);
    out.u32(config.qkv_bias ? 1u : 0u);
    out.u32(0);
    out.u32(0);
    out.u32(0);
    out.u32(0);
    out.i32(0);
    out.i32(0);
    out.u32(0);
    out.f32(0.f);
    out.f32(0.f);
    out.f32(0.f);
    out.i32(0);
    out.u32(0);
    out.u32(0);
    out.f32(1e-5f);
    out.u32(1);
    out.u32(0);
  } else if (ext) {
    out.i32(config.head_dim);
    out.i32(config.kv_heads);
    out.u32(config.qk_norm ? 1u : 0u);
    out.f32(config.rope_theta);
    out.f32(config.rms_eps);
  }
  out.pad8();
  out.u32(static_cast<uint32_t>(slots.size()));
  out.u32(0);

  std::vector<float> tied;
  for (const auto& slot : slots) {
    std::vector<float> values;
    if (slot.first == "tok_emb.weight") {
      values = bundle.take("model.embed_tokens.weight",
                           {static_cast<int64_t>(config.vocab), static_cast<int64_t>(config.d_model)});
      if (!bundle.contains("lm_head.weight")) {
        if (!config.tie) throw std::runtime_error("tensor");
        tied = values;
      }
    } else if (slot.first == "lm_head.weight") {
      if (bundle.contains("lm_head.weight")) {
        values = bundle.take("lm_head.weight",
                             {static_cast<int64_t>(config.vocab), static_cast<int64_t>(config.d_model)});
      } else {
        values = tied;
        tied.clear();
      }
    } else if (slot.first == "final_ln.weight") {
      values = bundle.take("model.norm.weight", {static_cast<int64_t>(config.d_model)});
    } else {
      const size_t dot = slot.first.find('.', 7);
      if (dot == std::string::npos) throw std::runtime_error("tensor");
      const int layer = std::stoi(slot.first.substr(7, dot - 7));
      const std::string rest = slot.first.substr(dot + 1);
      const std::string base = "model.layers." + std::to_string(layer) + ".";
      auto take2 = [&](const std::string& name, std::initializer_list<int64_t> shape) {
        return bundle.take(base + name, std::vector<int64_t>(shape));
      };
      if (rest == "ln1.weight") {
        values = take2("input_layernorm.weight", {config.d_model});
      } else if (rest == "ln2.weight") {
        values = take2("post_attention_layernorm.weight", {config.d_model});
      } else if (rest == "attn.q.weight") {
        values = interleave_heads(take2("self_attn.q_proj.weight", {q_dim, config.d_model}), config.heads,
                                  config.resolved_head, config.d_model);
      } else if (rest == "attn.k.weight") {
        values = interleave_heads(take2("self_attn.k_proj.weight", {kv_dim, config.d_model}), config.resolved_kv,
                                  config.resolved_head, config.d_model);
      } else if (rest == "attn.v.weight") {
        values = take2("self_attn.v_proj.weight", {kv_dim, config.d_model});
      } else if (rest == "attn.q.bias") {
        values = interleave_heads(take2("self_attn.q_proj.bias", {q_dim}), config.heads, config.resolved_head, 1);
      } else if (rest == "attn.k.bias") {
        values = interleave_heads(take2("self_attn.k_proj.bias", {kv_dim}), config.resolved_kv, config.resolved_head, 1);
      } else if (rest == "attn.v.bias") {
        values = take2("self_attn.v_proj.bias", {kv_dim});
      } else if (rest == "attn.out.weight") {
        values = take2("self_attn.o_proj.weight", {config.d_model, q_dim});
      } else if (rest == "attn.q_norm.weight") {
        values = interleave_heads(take2("self_attn.q_norm.weight", {config.resolved_head}),
                                  1, config.resolved_head, 1);
      } else if (rest == "attn.k_norm.weight") {
        values = interleave_heads(take2("self_attn.k_norm.weight", {config.resolved_head}),
                                  1, config.resolved_head, 1);
      } else if (rest == "fc1.weight") {
        values = take2("mlp.gate_proj.weight", {config.d_ff, config.d_model});
        std::vector<float> up = take2("mlp.up_proj.weight", {config.d_ff, config.d_model});
        values.insert(values.end(), up.begin(), up.end());
      } else if (rest == "fc2.weight") {
        values = take2("mlp.down_proj.weight", {config.d_model, config.d_ff});
      } else {
        throw std::runtime_error("tensor");
      }
    }
    if (static_cast<int64_t>(values.size()) != [&] {
          int64_t n = 1;
          for (int dim : slot.second) n *= dim;
          return n;
        }()) {
      throw std::runtime_error("tensor");
    }
    out.tensor(slot.first, slot.second, values);
  }
  bundle.finish();
  out.u32(0);
  out.u32(0);
  out.commit();

  for (const char* name : {"tokenizer.json", "tokenizer_config.json", "tokenizer.model", "generation_config.json",
                           "special_tokens_map.json", "vocab.txt", "chat_template.jinja"}) {
    copy_beside(model_dir / name, out_dir / name);
  }
  std::printf("hf: wrote %s (version %u)\n", (out_dir / "model.bin").string().c_str(), version);
}

struct Loaded {
  uint32_t version = 0;
  int vocab = 0;
  int d_model = 0;
  int heads = 0;
  int layers = 0;
  int d_ff = 0;
  int max_seq = 0;
  int head_dim = 0;
  int kv_heads = 0;
  bool qk_norm = false;
  float rope_theta = 10000.f;
  float rms_eps = 1e-5f;
  bool qkv_bias = false;
  bool attn_out_bias = false;
  bool mlp_bias = false;
  bool lm_head_bias = false;
  int sliding_window = 0;
  std::map<std::string, std::vector<float>> tensors;
};

class MemReader {
 public:
  explicit MemReader(std::vector<uint8_t> bytes) : bytes_(std::move(bytes)) {}
  uint32_t u32() {
    if (pos_ + 4 > bytes_.size()) throw std::runtime_error("read");
    uint32_t value = static_cast<uint32_t>(bytes_[pos_]) | (static_cast<uint32_t>(bytes_[pos_ + 1]) << 8) |
                     (static_cast<uint32_t>(bytes_[pos_ + 2]) << 16) | (static_cast<uint32_t>(bytes_[pos_ + 3]) << 24);
    pos_ += 4;
    return value;
  }
  uint64_t u64() {
    const uint64_t low = u32();
    return low | (static_cast<uint64_t>(u32()) << 32);
  }
  int i32() { return static_cast<int>(u32()); }
  float f32() {
    const uint32_t bits = u32();
    float value = 0.f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }
  void pad8() {
    while (pos_ % 8 != 0) {
      if (pos_ >= bytes_.size()) throw std::runtime_error("read");
      ++pos_;
    }
  }
  std::string str() {
    const uint32_t len = u32();
    if (len > 256 || pos_ + len > bytes_.size()) throw std::runtime_error("read");
    std::string value(reinterpret_cast<const char*>(bytes_.data() + pos_), len);
    pos_ += len;
    pad8();
    return value;
  }

 private:
  std::vector<uint8_t> bytes_;
  size_t pos_ = 0;
};

Loaded read_written(const std::filesystem::path& path) {
  const std::string text = read_text(path);
  MemReader in(std::vector<uint8_t>(text.begin(), text.end()));
  if (in.u32() != kCheckpointMagic) throw std::runtime_error("magic");
  Loaded loaded;
  loaded.version = in.u32();
  if (in.u32() != 6 || in.u32() != 0 || in.u64() != 0 || in.u32() != 0 || in.u32() != 0) {
    throw std::runtime_error("header");
  }
  loaded.vocab = in.i32();
  loaded.d_model = in.i32();
  loaded.heads = in.i32();
  loaded.layers = in.i32();
  loaded.d_ff = in.i32();
  loaded.max_seq = in.i32();
  in.f32();
  if (in.u32() != 0 || in.u32() != 0 || in.u32() != 2 || in.u32() != 2) throw std::runtime_error("header");
  if (loaded.version == 3 || loaded.version == 7) {
    loaded.head_dim = in.i32();
    loaded.kv_heads = in.i32();
    loaded.qk_norm = in.u32() != 0;
    loaded.rope_theta = in.f32();
    loaded.rms_eps = in.f32();
    if (loaded.version == 7) {
      if (in.u32() != 1) throw std::runtime_error("header");
      in.i32();
      loaded.qkv_bias = in.u32() != 0;
      loaded.attn_out_bias = in.u32() != 0;
      loaded.mlp_bias = in.u32() != 0;
      loaded.lm_head_bias = in.u32() != 0;
      in.u32();
      in.i32();
      loaded.sliding_window = in.i32();
      in.u32();
      in.f32();
      in.f32();
      in.f32();
      in.i32();
      in.u32();
      in.u32();
      in.f32();
      in.u32();
      in.u32();
    }
  } else if (loaded.version != 1) {
    throw std::runtime_error("version");
  }
  in.pad8();
  const uint32_t count = in.u32();
  if (in.u32() != 0) throw std::runtime_error("header");
  for (uint32_t i = 0; i < count; ++i) {
    const std::string name = in.str();
    const uint32_t rank = in.u32();
    int64_t numel = 1;
    for (uint32_t d = 0; d < rank; ++d) numel *= in.i32();
    in.pad8();
    if (in.u32() != 1) throw std::runtime_error("dtype");
    const uint32_t n = in.u32();
    if (static_cast<int64_t>(n) != numel) throw std::runtime_error("tensor");
    std::vector<float> values(n);
    for (uint32_t k = 0; k < n; ++k) values[k] = in.f32();
    in.pad8();
    loaded.tensors.emplace(name, std::move(values));
  }
  return loaded;
}

void write_text(const std::filesystem::path& path, const std::string& text) {
  const int fd = ::open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
  if (fd < 0) throw std::runtime_error("write");
  size_t off = 0;
  while (off < text.size()) {
    const ssize_t n = ::write(fd, text.data() + off, text.size() - off);
    if (n < 0) {
      if (errno == EINTR) continue;
      ::close(fd);
      throw std::runtime_error("write");
    }
    off += static_cast<size_t>(n);
  }
  ::close(fd);
}

std::vector<uint8_t> f32_bytes(const std::vector<float>& values) {
  std::vector<uint8_t> out(values.size() * 4);
  std::memcpy(out.data(), values.data(), out.size());
  return out;
}

std::vector<uint8_t> bf16_bytes(const std::vector<float>& values) {
  std::vector<uint8_t> out(values.size() * 2);
  for (size_t i = 0; i < values.size(); ++i) {
    uint32_t bits = 0;
    std::memcpy(&bits, &values[i], 4);
    const uint16_t half = static_cast<uint16_t>(bits >> 16);
    std::memcpy(out.data() + i * 2, &half, 2);
  }
  return out;
}

struct RawTensor {
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
  std::vector<uint8_t> bytes;
};

void write_safetensors(const std::filesystem::path& path, const std::vector<RawTensor>& tensors) {
  std::string header = "{";
  std::vector<uint8_t> data;
  for (size_t t = 0; t < tensors.size(); ++t) {
    while (data.size() % 8 != 0) data.push_back(0);
    const uint64_t begin = data.size();
    data.insert(data.end(), tensors[t].bytes.begin(), tensors[t].bytes.end());
    const uint64_t end = data.size();
    if (t) header += ',';
    header += '"' + tensors[t].name + "\":{\"dtype\":\"" + tensors[t].dtype + "\",\"shape\":[";
    for (size_t d = 0; d < tensors[t].shape.size(); ++d) {
      if (d) header += ',';
      header += std::to_string(tensors[t].shape[d]);
    }
    header += "],\"data_offsets\":[" + std::to_string(begin) + ',' + std::to_string(end) + "]}";
  }
  header += '}';
  std::vector<uint8_t> file(8 + header.size() + data.size());
  for (int i = 0; i < 8; ++i) file[static_cast<size_t>(i)] = static_cast<uint8_t>(header.size() >> (8 * i));
  std::memcpy(file.data() + 8, header.data(), header.size());
  std::memcpy(file.data() + 8 + header.size(), data.data(), data.size());
  const int fd = ::open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
  if (fd < 0) throw std::runtime_error("write");
  size_t off = 0;
  while (off < file.size()) {
    const ssize_t n = ::write(fd, file.data() + off, file.size() - off);
    if (n < 0) {
      if (errno == EINTR) continue;
      ::close(fd);
      throw std::runtime_error("write");
    }
    off += static_cast<size_t>(n);
  }
  ::close(fd);
}

void expect(bool cond, const char* what) {
  if (!cond) throw std::runtime_error(what);
}

void expect_reason(const std::filesystem::path& dir, const char* reason) {
  try {
    import_snapshot(dir, dir / "out", std::nullopt);
  } catch (const std::runtime_error& error) {
    if (std::string(error.what()) == reason) return;
    throw std::runtime_error(std::string("self-test reason ") + error.what());
  }
  throw std::runtime_error(std::string("self-test missed ") + reason);
}

int self_test() {
  const std::filesystem::path root = std::filesystem::temp_directory_path() / "hf-self-test";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);

  const std::filesystem::path qwen = root / "qwen";
  std::filesystem::create_directories(qwen);
  write_text(qwen / "config.json",
             "{"
             "\"model_type\":\"qwen3\","
             "\"hidden_act\":\"silu\","
             "\"hidden_size\":8,"
             "\"intermediate_size\":4,"
             "\"num_hidden_layers\":1,"
             "\"num_attention_heads\":4,"
             "\"num_key_value_heads\":2,"
             "\"head_dim\":4,"
             "\"vocab_size\":6,"
             "\"max_position_embeddings\":4,"
             "\"rms_norm_eps\":1e-6,"
             "\"rope_theta\":1000000,"
             "\"attention_bias\":false,"
             "\"use_sliding_window\":false,"
             "\"sliding_window\":32,"
             "\"rope_scaling\":null,"
             "\"tie_word_embeddings\":true"
             "}");
  write_text(qwen / "tokenizer.json", "{}");
  std::vector<float> embed(6 * 8, 3.f);
  std::vector<float> q(16 * 8, 0.f);
  for (int row = 0; row < 16; ++row) q[static_cast<size_t>(row) * 8] = static_cast<float>(row);
  std::vector<float> k(8 * 8, 0.f);
  for (int row = 0; row < 8; ++row) k[static_cast<size_t>(row) * 8] = 100.f + static_cast<float>(row);
  std::vector<float> ln = {1.f, 1.5f, 1.f, 1.5f, 1.f, 1.5f, 1.f, 1.5f};
  std::vector<float> gate(4 * 8, 1.f);
  std::vector<float> up(4 * 8, 2.f);
  std::vector<RawTensor> shard_a = {
      {"model.embed_tokens.weight", "F32", {6, 8}, f32_bytes(embed)},
  };
  std::vector<RawTensor> shard_b = {
      {"model.layers.0.input_layernorm.weight", "BF16", {8}, bf16_bytes(ln)},
      {"model.layers.0.self_attn.q_proj.weight", "F32", {16, 8}, f32_bytes(q)},
      {"model.layers.0.self_attn.k_proj.weight", "F32", {8, 8}, f32_bytes(k)},
      {"model.layers.0.self_attn.v_proj.weight", "F32", {8, 8}, f32_bytes(std::vector<float>(64, 0.f))},
      {"model.layers.0.self_attn.o_proj.weight", "F32", {8, 16}, f32_bytes(std::vector<float>(128, 0.f))},
      {"model.layers.0.self_attn.q_norm.weight", "F32", {4}, f32_bytes({1.f, 2.f, 3.f, 4.f})},
      {"model.layers.0.self_attn.k_norm.weight", "F32", {4}, f32_bytes({5.f, 6.f, 7.f, 8.f})},
      {"model.layers.0.post_attention_layernorm.weight", "F32", {8}, f32_bytes(std::vector<float>(8, 0.f))},
      {"model.layers.0.mlp.gate_proj.weight", "F32", {4, 8}, f32_bytes(gate)},
      {"model.layers.0.mlp.up_proj.weight", "F32", {4, 8}, f32_bytes(up)},
      {"model.layers.0.mlp.down_proj.weight", "F32", {8, 4}, f32_bytes(std::vector<float>(32, 0.f))},
      {"model.norm.weight", "F32", {8}, f32_bytes(std::vector<float>(8, 0.f))},
  };
  write_safetensors(qwen / "model-00001-of-00002.safetensors", shard_a);
  write_safetensors(qwen / "model-00002-of-00002.safetensors", shard_b);
  std::string index = "{\"weight_map\":{";
  bool first = true;
  auto add_index = [&](const std::vector<RawTensor>& tensors, const char* file) {
    for (const RawTensor& tensor : tensors) {
      if (!first) index += ',';
      first = false;
      index += '"' + tensor.name + "\":\"" + file + '"';
    }
  };
  add_index(shard_a, "model-00001-of-00002.safetensors");
  add_index(shard_b, "model-00002-of-00002.safetensors");
  index += "}}";
  write_text(qwen / "model.safetensors.index.json", index);

  const std::filesystem::path qwen_out = root / "qwen-out";
  import_snapshot(qwen, qwen_out, std::nullopt);
  const Loaded qwen_loaded = read_written(qwen_out / "model.bin");
  expect(qwen_loaded.version == 3, "self-test qwen version");
  expect(qwen_loaded.qk_norm, "self-test qwen qk");
  expect(qwen_loaded.head_dim == 4, "self-test qwen head");
  expect(qwen_loaded.kv_heads == 2, "self-test qwen kv");
  expect(qwen_loaded.rope_theta == 1000000.f, "self-test qwen rope");
  expect(qwen_loaded.rms_eps == 1e-6f, "self-test qwen eps");
  expect(qwen_loaded.tensors.count("lm_head.weight") == 1, "self-test qwen tied");
  expect(qwen_loaded.tensors.at("lm_head.weight") == embed, "self-test qwen tied values");
  expect(qwen_loaded.tensors.at("tok_emb.weight") == embed, "self-test qwen embed");
  const std::vector<float>& q_out = qwen_loaded.tensors.at("blocks.0.attn.q.weight");
  expect(q_out.size() == 128, "self-test qwen q size");
  expect(q_out[0] == 0.f && q_out[8] == 2.f && q_out[16] == 1.f && q_out[24] == 3.f, "self-test qwen q permute");
  expect(q_out[32] == 4.f && q_out[40] == 6.f && q_out[48] == 5.f && q_out[56] == 7.f, "self-test qwen q head1");
  const std::vector<float>& k_out = qwen_loaded.tensors.at("blocks.0.attn.k.weight");
  expect(k_out[0] == 100.f && k_out[8] == 102.f && k_out[16] == 101.f && k_out[24] == 103.f, "self-test qwen k permute");
  expect(qwen_loaded.tensors.at("blocks.0.attn.q_norm.weight") == std::vector<float>{1.f, 3.f, 2.f, 4.f},
         "self-test qwen q norm permute");
  expect(qwen_loaded.tensors.at("blocks.0.attn.k_norm.weight") == std::vector<float>{5.f, 7.f, 6.f, 8.f},
         "self-test qwen k norm permute");
  const std::vector<float>& fc1 = qwen_loaded.tensors.at("blocks.0.fc1.weight");
  expect(fc1.size() == 64, "self-test qwen fc1");
  expect(fc1.front() == 1.f && fc1[31] == 1.f && fc1[32] == 2.f && fc1.back() == 2.f, "self-test qwen pack");
  const std::vector<float>& ln_out = qwen_loaded.tensors.at("blocks.0.ln1.weight");
  expect(ln_out == ln, "self-test qwen bf16");
  expect(std::filesystem::is_regular_file(qwen_out / "tokenizer.json"), "self-test qwen tokenizer");

  const std::filesystem::path llama = root / "llama";
  std::filesystem::create_directories(llama);
  write_text(llama / "config.json",
             "{"
             "\"model_type\":\"llama\","
             "\"hidden_size\":8,"
             "\"intermediate_size\":4,"
             "\"num_hidden_layers\":1,"
             "\"num_attention_heads\":2,"
             "\"num_key_value_heads\":2,"
             "\"vocab_size\":6,"
             "\"max_position_embeddings\":4,"
             "\"tie_word_embeddings\":false"
             "}");
  write_text(llama / "tokenizer.model", "dummy");
  std::vector<float> lq(8 * 8, 0.f);
  for (int row = 0; row < 8; ++row) lq[static_cast<size_t>(row) * 8] = static_cast<float>(row);
  std::vector<float> lm(6 * 8, 4.f);
  write_safetensors(llama / "model.safetensors",
                    {{"model.embed_tokens.weight", "F32", {6, 8}, f32_bytes(std::vector<float>(48, 1.f))},
                     {"model.layers.0.input_layernorm.weight", "F32", {8}, f32_bytes(std::vector<float>(8, 1.f))},
                     {"model.layers.0.self_attn.q_proj.weight", "F32", {8, 8}, f32_bytes(lq)},
                     {"model.layers.0.self_attn.k_proj.weight", "F32", {8, 8}, f32_bytes(std::vector<float>(64, 0.f))},
                     {"model.layers.0.self_attn.v_proj.weight", "F32", {8, 8}, f32_bytes(std::vector<float>(64, 0.f))},
                     {"model.layers.0.self_attn.o_proj.weight", "F32", {8, 8}, f32_bytes(std::vector<float>(64, 0.f))},
                     {"model.layers.0.post_attention_layernorm.weight", "F32", {8}, f32_bytes(std::vector<float>(8, 1.f))},
                     {"model.layers.0.mlp.gate_proj.weight", "F32", {4, 8}, f32_bytes(std::vector<float>(32, 0.f))},
                     {"model.layers.0.mlp.up_proj.weight", "F32", {4, 8}, f32_bytes(std::vector<float>(32, 0.f))},
                     {"model.layers.0.mlp.down_proj.weight", "F32", {8, 4}, f32_bytes(std::vector<float>(32, 0.f))},
                     {"model.norm.weight", "F16", {8}, [] {
                        std::vector<uint8_t> bytes(16);
                        for (int i = 0; i < 8; ++i) {
                          bytes[static_cast<size_t>(i) * 2] = 0x00;
                          bytes[static_cast<size_t>(i) * 2 + 1] = 0x3c;
                        }
                        return bytes;
                      }()},
                     {"lm_head.weight", "F32", {6, 8}, f32_bytes(lm)}});
  const std::filesystem::path llama_out = root / "llama-out";
  import_snapshot(llama, llama_out, 9);
  const Loaded llama_loaded = read_written(llama_out / "model.bin");
  expect(llama_loaded.version == 1, "self-test llama version");
  expect(llama_loaded.max_seq == 9, "self-test llama context");
  expect(!llama_loaded.qk_norm && llama_loaded.head_dim == 0 && llama_loaded.kv_heads == 0, "self-test llama classic");
  const std::vector<float>& llama_q = llama_loaded.tensors.at("blocks.0.attn.q.weight");
  expect(llama_q[0] == 0.f && llama_q[8] == 2.f && llama_q[16] == 1.f && llama_q[24] == 3.f, "self-test llama q");
  expect(llama_loaded.tensors.at("lm_head.weight") == lm, "self-test llama head");
  expect(llama_loaded.tensors.at("final_ln.weight") == std::vector<float>(8, 1.f), "self-test llama f16");
  expect(std::filesystem::is_regular_file(llama_out / "tokenizer.model"), "self-test llama tokenizer");

  const std::filesystem::path qwen2 = root / "qwen2";
  std::filesystem::create_directories(qwen2);
  write_text(qwen2 / "config.json",
             "{"
             "\"model_type\":\"qwen2\","
             "\"hidden_act\":\"silu\","
             "\"hidden_size\":8,"
             "\"intermediate_size\":4,"
             "\"num_hidden_layers\":1,"
             "\"num_attention_heads\":2,"
             "\"num_key_value_heads\":2,"
             "\"vocab_size\":6,"
             "\"max_position_embeddings\":4,"
             "\"rms_norm_eps\":1e-6,"
             "\"rope_theta\":1000000,"
             "\"use_sliding_window\":false,"
             "\"sliding_window\":32,"
             "\"tie_word_embeddings\":false"
             "}");
  write_text(qwen2 / "tokenizer.json", "{}");
  std::vector<float> q_bias(8);
  std::vector<float> k_bias(8);
  std::vector<float> v_bias(8);
  for (int i = 0; i < 8; ++i) {
    q_bias[static_cast<size_t>(i)] = static_cast<float>(i);
    k_bias[static_cast<size_t>(i)] = 100.f + static_cast<float>(i);
    v_bias[static_cast<size_t>(i)] = 10.f + static_cast<float>(i);
  }
  write_safetensors(qwen2 / "model.safetensors",
                    {{"model.embed_tokens.weight", "F32", {6, 8}, f32_bytes(std::vector<float>(48, 1.f))},
                     {"model.layers.0.input_layernorm.weight", "F32", {8}, f32_bytes(std::vector<float>(8, 1.f))},
                     {"model.layers.0.self_attn.q_proj.weight", "F32", {8, 8}, f32_bytes(std::vector<float>(64, 0.f))},
                     {"model.layers.0.self_attn.q_proj.bias", "F32", {8}, f32_bytes(q_bias)},
                     {"model.layers.0.self_attn.k_proj.weight", "F32", {8, 8}, f32_bytes(std::vector<float>(64, 0.f))},
                     {"model.layers.0.self_attn.k_proj.bias", "F32", {8}, f32_bytes(k_bias)},
                     {"model.layers.0.self_attn.v_proj.weight", "F32", {8, 8}, f32_bytes(std::vector<float>(64, 0.f))},
                     {"model.layers.0.self_attn.v_proj.bias", "F32", {8}, f32_bytes(v_bias)},
                     {"model.layers.0.self_attn.o_proj.weight", "F32", {8, 8}, f32_bytes(std::vector<float>(64, 0.f))},
                     {"model.layers.0.post_attention_layernorm.weight", "F32", {8}, f32_bytes(std::vector<float>(8, 1.f))},
                     {"model.layers.0.mlp.gate_proj.weight", "F32", {4, 8}, f32_bytes(std::vector<float>(32, 0.f))},
                     {"model.layers.0.mlp.up_proj.weight", "F32", {4, 8}, f32_bytes(std::vector<float>(32, 0.f))},
                     {"model.layers.0.mlp.down_proj.weight", "F32", {8, 4}, f32_bytes(std::vector<float>(32, 0.f))},
                     {"model.norm.weight", "F32", {8}, f32_bytes(std::vector<float>(8, 1.f))},
                     {"lm_head.weight", "F32", {6, 8}, f32_bytes(std::vector<float>(48, 2.f))}});
  const std::filesystem::path qwen2_out = root / "qwen2-out";
  import_snapshot(qwen2, qwen2_out, std::nullopt);
  const Loaded qwen2_loaded = read_written(qwen2_out / "model.bin");
  expect(qwen2_loaded.version == 7, "self-test qwen2 version");
  expect(qwen2_loaded.qkv_bias && !qwen2_loaded.attn_out_bias && !qwen2_loaded.mlp_bias && !qwen2_loaded.lm_head_bias,
         "self-test qwen2 flags");
  expect(!qwen2_loaded.qk_norm && qwen2_loaded.head_dim == 0 && qwen2_loaded.kv_heads == 0, "self-test qwen2 shape");
  expect(qwen2_loaded.rope_theta == 1000000.f && qwen2_loaded.rms_eps == 1e-6f, "self-test qwen2 rope");
  expect(qwen2_loaded.sliding_window == 0, "self-test qwen2 window");
  expect(qwen2_loaded.tensors.at("blocks.0.attn.q.bias") == std::vector<float>({0, 2, 1, 3, 4, 6, 5, 7}),
         "self-test qwen2 q bias");
  expect(qwen2_loaded.tensors.at("blocks.0.attn.k.bias") ==
             std::vector<float>({100, 102, 101, 103, 104, 106, 105, 107}),
         "self-test qwen2 k bias");
  expect(qwen2_loaded.tensors.at("blocks.0.attn.v.bias") == v_bias, "self-test qwen2 v bias");
  expect(qwen2_loaded.tensors.count("blocks.0.attn.out.bias") == 0, "self-test qwen2 out");
  expect(qwen2_loaded.tensors.count("blocks.0.fc1.bias") == 0 && qwen2_loaded.tensors.count("blocks.0.fc2.bias") == 0 &&
             qwen2_loaded.tensors.count("lm_head.bias") == 0,
         "self-test qwen2 other bias");

  const std::filesystem::path bad = root / "bad";
  std::filesystem::create_directories(bad);
  write_text(bad / "config.json", "{\"model_type\":\"mixtral\"}");
  expect_reason(bad, "architecture");
  write_text(bad / "config.json",
             "{\"model_type\":\"llama\",\"hidden_size\":8,\"intermediate_size\":4,\"num_hidden_layers\":1,"
             "\"num_attention_heads\":2,\"vocab_size\":6,\"max_position_embeddings\":4,"
             "\"rope_scaling\":{\"rope_type\":\"llama3\",\"factor\":8}}");
  expect_reason(bad, "rope");
  write_text(bad / "config.json",
             "{\"model_type\":\"llama\",\"hidden_size\":8,\"intermediate_size\":4,\"num_hidden_layers\":1,"
             "\"num_attention_heads\":2,\"vocab_size\":6,\"max_position_embeddings\":4,\"attention_bias\":true}");
  expect_reason(bad, "bias");
  write_text(bad / "config.json",
             "{\"model_type\":\"qwen3\",\"hidden_act\":\"silu\",\"hidden_size\":8,\"intermediate_size\":4,"
             "\"num_hidden_layers\":1,\"num_attention_heads\":2,\"vocab_size\":6,\"max_position_embeddings\":4,"
             "\"attention_bias\":true}");
  expect_reason(bad, "bias");
  write_text(bad / "config.json",
             "{\"model_type\":\"qwen3\",\"hidden_act\":\"silu\",\"hidden_size\":8,\"intermediate_size\":4,"
             "\"num_hidden_layers\":1,\"num_attention_heads\":2,\"vocab_size\":6,\"max_position_embeddings\":4,"
             "\"use_sliding_window\":true}");
  expect_reason(bad, "window");
  write_text(bad / "config.json",
             "{\"model_type\":\"llama\",\"hidden_size\":8,\"intermediate_size\":4,\"num_hidden_layers\":1,"
             "\"num_attention_heads\":2,\"vocab_size\":6,\"max_position_embeddings\":4}");
  expect_reason(bad, "tokenizer");
  write_text(bad / "tokenizer.model", "dummy");
  write_safetensors(bad / "model.safetensors", {{"bad", "I8", {1}, std::vector<uint8_t>{0}}});
  expect_reason(bad, "dtype");
  write_safetensors(bad / "model.safetensors",
                    {{"model.layers.0.mlp.experts.0.weight", "F32", {1, 1}, f32_bytes(std::vector<float>{0.f})}});
  expect_reason(bad, "moe");
  write_text(bad / "config.json",
             "{\"model_type\":\"qwen2\",\"hidden_act\":\"silu\",\"hidden_size\":8,\"intermediate_size\":4,"
             "\"num_hidden_layers\":1,\"num_attention_heads\":2,\"vocab_size\":6,\"max_position_embeddings\":4,"
             "\"attention_bias\":true}");
  write_text(bad / "tokenizer.json", "{}");
  write_safetensors(bad / "model.safetensors",
                    {{"model.layers.0.self_attn.o_proj.bias", "F32", {8}, f32_bytes(std::vector<float>(8, 0.f))}});
  expect_reason(bad, "bias");
  write_safetensors(bad / "model.safetensors",
                    {{"model.embed_tokens.weight", "F32", {6, 8}, f32_bytes(std::vector<float>(48, 0.f))},
                     {"model.layers.0.input_layernorm.weight", "F32", {8}, f32_bytes(std::vector<float>(8, 0.f))},
                     {"model.layers.0.self_attn.q_proj.weight", "F32", {8, 8}, f32_bytes(std::vector<float>(64, 0.f))},
                     {"model.layers.0.self_attn.k_proj.weight", "F32", {8, 8}, f32_bytes(std::vector<float>(64, 0.f))},
                     {"model.layers.0.self_attn.v_proj.weight", "F32", {8, 8}, f32_bytes(std::vector<float>(64, 0.f))},
                     {"model.layers.0.self_attn.o_proj.weight", "F32", {8, 8}, f32_bytes(std::vector<float>(64, 0.f))},
                     {"model.layers.0.post_attention_layernorm.weight", "F32", {8}, f32_bytes(std::vector<float>(8, 0.f))},
                     {"model.layers.0.mlp.gate_proj.weight", "F32", {4, 8}, f32_bytes(std::vector<float>(32, 0.f))},
                     {"model.layers.0.mlp.up_proj.weight", "F32", {4, 8}, f32_bytes(std::vector<float>(32, 0.f))},
                     {"model.layers.0.mlp.down_proj.weight", "F32", {8, 4}, f32_bytes(std::vector<float>(32, 0.f))},
                     {"model.norm.weight", "F32", {8}, f32_bytes(std::vector<float>(8, 0.f))},
                     {"lm_head.weight", "F32", {6, 8}, f32_bytes(std::vector<float>(48, 0.f))}});
  expect_reason(bad, "tensor");

  std::filesystem::remove_all(root);
  std::printf("self-test ok\n");
  return 0;
}

void usage() {
  std::fputs(
      "usage: hf import MODEL_DIR OUT_DIR [--context N]\n"
      "       hf self-test\n"
      "\n"
      "import reads config.json and model.safetensors (or a sharded index)\n"
      "and writes model.bin. model_type qwen3, qwen2, and llama with no\n"
      "rope_scaling and no sliding window are accepted. qwen3 and llama reject\n"
      "attention bias. qwen2 with QKV bias writes version 7 and the three\n"
      "Q, K, and V biases. Q and K, weights and biases, are interleaved per head\n"
      "so adjacent-pair RoPE matches rotate-half weights. V bias is copied\n"
      "straight. SwiGLU gate and up are packed into fc1. F16 and BF16 widen to\n"
      "float32. A tied token table is copied onto lm_head. The checkpoint\n"
      "records tie_embeddings as false.\n"
      "\n"
      "qwen3 and qwen2 require tokenizer.json. llama requires tokenizer.model.\n"
      "Those files are copied. chat loads tokenizer.json when tokenizer.bpe is\n"
      "not in the output directory.\n"
      "--context N sets max_seq_len. The default is max_position_embeddings.\n"
      "\n"
      "Llama 3 rope scaling, Mistral, GPT-2, BERT, MoE, and any other dtype\n"
      "are rejected. A qwen2 file stores only the QKV biases.\n",
      stderr);
}

int parse_context(const std::string& text) {
  if (text.empty() || text.size() > 9) throw std::runtime_error("context");
  int value = 0;
  for (char c : text) {
    if (c < '0' || c > '9') throw std::runtime_error("context");
    value = value * 10 + (c - '0');
  }
  if (value < 1) throw std::runtime_error("context");
  return value;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      usage();
      return 2;
    }
    const std::string cmd = argv[1];
    if (cmd == "self-test") {
      if (argc != 2) {
        usage();
        return 2;
      }
      return self_test();
    }
    if (cmd == "import") {
      if (argc != 4 && argc != 6) {
        usage();
        return 2;
      }
      std::optional<int> context;
      if (argc == 6) {
        if (std::string(argv[4]) != "--context") {
          usage();
          return 2;
        }
        context = parse_context(argv[5]);
      }
      import_snapshot(argv[2], argv[3], context);
      return 0;
    }
    usage();
    return 2;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "hf: %s\n", error.what());
    return 1;
  }
}
