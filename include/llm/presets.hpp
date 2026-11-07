#pragma once

#include "nn/optim.hpp"
#include "nn/transformer.hpp"

#include <vector>

namespace llm {

namespace presets {

nn::TransformerConfig tiny(int vocab_size);
nn::TransformerConfig small(int vocab_size);
nn::TransformerConfig base(int vocab_size);

}  // namespace presets

// Contiguous decay / no-decay runs. Concatenating group.params is parameters() order.
std::vector<nn::AdamWGroup> adamw_groups(const nn::Transformer& model, float weight_decay);

}  // namespace llm
