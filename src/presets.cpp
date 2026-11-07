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
