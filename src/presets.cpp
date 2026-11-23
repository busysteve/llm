#include "llm/presets.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace llm {
namespace {

nn::TransformerConfig make_config(int vocab_size, int layers, int d_model, int heads, int d_ff) {
  if (vocab_size <= 0) throw std::invalid_argument("vocab_size");
  nn::TransformerConfig config;
  config.vocab_size = vocab_size;
  config.d_model = d_model;
  config.num_heads = heads;
  config.num_layers = layers;
  config.d_ff = d_ff;
  config.max_seq_len = 1024;
  config.dropout = 0.f;
  config.tie_embeddings = false;
  config.bias = false;
  config.activation = nn::TransformerActivation::SwiGLU;
  config.positional = nn::PositionalEncoding::LearnedAbsolute;
  return config;
}

nn::TransformerConfig qwen_width(int vocab_size, int layers, int d_model, int heads, int kv_heads, int d_ff) {
  nn::TransformerConfig config = make_config(vocab_size, layers, d_model, heads, d_ff);
  config.positional = nn::PositionalEncoding::Rope;
  config.num_kv_heads = kv_heads;
  config.head_dim = 128;
  config.qk_norm = true;
  config.rope_theta = 1000000.f;
  config.rms_eps = 1e-6f;
  return config;
}

// GPT-2 block: learned positions, LayerNorm, GELU, and a bias on every linear.
// d_ff is 4 * d_model. GELU's first linear writes d_ff, not 2 * d_ff.
// Embeddings stay untied: nn rejects tie_embeddings.
nn::TransformerConfig gpt_width(int vocab_size, int layers, int d_model, int heads, int max_seq_len) {
  nn::TransformerConfig config = make_config(vocab_size, layers, d_model, heads, d_model * 4);
  config.max_seq_len = max_seq_len;
  config.bias = true;
  config.activation = nn::TransformerActivation::Gelu;
  return config;
}

bool apply_decay(const nn::NamedParam& param) {
  if (param.value == nullptr || param.value->rank() != 2) return false;
  if (param.name.find("ln") != std::string::npos) return false;
  if (param.name.size() >= 5 && param.name.compare(param.name.size() - 5, 5, ".bias") == 0) {
    return false;
  }
  return true;
}

}  // namespace

namespace presets {

nn::TransformerConfig tiny(int vocab_size) { return make_config(vocab_size, 4, 256, 4, 768); }

nn::TransformerConfig small(int vocab_size) { return make_config(vocab_size, 6, 512, 8, 1536); }

nn::TransformerConfig base(int vocab_size) { return make_config(vocab_size, 12, 768, 12, 2048); }

// SwiGLU width is (8/3) * d_model, rounded up to a multiple of 256.
nn::TransformerConfig medium(int vocab_size) { return make_config(vocab_size, 24, 1024, 16, 2816); }

nn::TransformerConfig large(int vocab_size) { return make_config(vocab_size, 36, 1280, 20, 3584); }

nn::TransformerConfig llama_tiny(int vocab_size) {
  nn::TransformerConfig config = tiny(vocab_size);
  config.positional = nn::PositionalEncoding::Rope;
  return config;
}

nn::TransformerConfig llama_small(int vocab_size) {
  nn::TransformerConfig config = small(vocab_size);
  config.positional = nn::PositionalEncoding::Rope;
  return config;
}

nn::TransformerConfig llama_base(int vocab_size) {
  nn::TransformerConfig config = base(vocab_size);
  config.positional = nn::PositionalEncoding::Rope;
  return config;
}

nn::TransformerConfig llama_medium(int vocab_size) {
  nn::TransformerConfig config = medium(vocab_size);
  config.positional = nn::PositionalEncoding::Rope;
  return config;
}

nn::TransformerConfig llama_large(int vocab_size) {
  nn::TransformerConfig config = large(vocab_size);
  config.positional = nn::PositionalEncoding::Rope;
  return config;
}

nn::TransformerConfig qwen_tiny(int vocab_size) { return qwen_width(vocab_size, 4, 256, 4, 2, 768); }

nn::TransformerConfig qwen_small(int vocab_size) { return qwen_width(vocab_size, 6, 512, 8, 2, 1536); }

nn::TransformerConfig qwen_base(int vocab_size) { return qwen_width(vocab_size, 12, 768, 12, 4, 2048); }

nn::TransformerConfig qwen_medium(int vocab_size) {
  return qwen_width(vocab_size, 24, 1024, 16, 4, 2816);
}

nn::TransformerConfig qwen_large(int vocab_size) {
  return qwen_width(vocab_size, 36, 1280, 20, 5, 3584);
}

nn::TransformerConfig gpt2_tiny(int vocab_size) { return gpt_width(vocab_size, 4, 256, 4, 1024); }

nn::TransformerConfig gpt2_small(int vocab_size) { return gpt_width(vocab_size, 6, 512, 8, 1024); }

nn::TransformerConfig gpt2_base(int vocab_size) { return gpt_width(vocab_size, 12, 768, 12, 1024); }

nn::TransformerConfig gpt2_medium(int vocab_size) { return gpt_width(vocab_size, 24, 1024, 16, 1024); }

nn::TransformerConfig gpt2_large(int vocab_size) { return gpt_width(vocab_size, 36, 1280, 20, 1024); }

nn::TransformerConfig gpt3_tiny(int vocab_size) { return gpt_width(vocab_size, 4, 256, 4, 2048); }

nn::TransformerConfig gpt3_small(int vocab_size) { return gpt_width(vocab_size, 6, 512, 8, 2048); }

nn::TransformerConfig gpt3_base(int vocab_size) { return gpt_width(vocab_size, 12, 768, 12, 2048); }

nn::TransformerConfig gpt3_medium(int vocab_size) { return gpt_width(vocab_size, 24, 1024, 16, 2048); }

nn::TransformerConfig gpt3_large(int vocab_size) { return gpt_width(vocab_size, 36, 1280, 20, 2048); }

// BERT block: the GPT-2 dense stack with bidirectional attention, a segment
// table of size 2, and context 512. The tokenizer pad id is the mask token.
nn::TransformerConfig bert_width(int vocab_size, int layers, int d_model, int heads) {
  nn::TransformerConfig config = gpt_width(vocab_size, layers, d_model, heads, 512);
  config.causal = false;
  config.type_vocab_size = 2;
  return config;
}

nn::TransformerConfig bert_tiny(int vocab_size) { return bert_width(vocab_size, 4, 256, 4); }

nn::TransformerConfig bert_small(int vocab_size) { return bert_width(vocab_size, 6, 512, 8); }

nn::TransformerConfig bert_base(int vocab_size) { return bert_width(vocab_size, 12, 768, 12); }

nn::TransformerConfig bert_medium(int vocab_size) { return bert_width(vocab_size, 24, 1024, 16); }

nn::TransformerConfig bert_large(int vocab_size) { return bert_width(vocab_size, 36, 1280, 20); }

}  // namespace presets

std::vector<nn::AdamWGroup> adamw_groups(const nn::Transformer& model, float weight_decay) {
  if (!std::isfinite(weight_decay) || weight_decay < 0.f) throw std::invalid_argument("weight_decay");
  std::vector<nn::AdamWGroup> groups;
  for (const nn::NamedParam& param : model.parameters()) {
    const float decay = apply_decay(param) ? weight_decay : 0.f;
    if (groups.empty() || groups.back().weight_decay != decay) {
      nn::AdamWGroup group;
      group.weight_decay = decay;
      groups.push_back(std::move(group));
    }
    groups.back().params.push_back(param);
  }
  return groups;
}

}  // namespace llm
