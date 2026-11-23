#pragma once

#include "nn/optim.hpp"
#include "nn/transformer.hpp"

#include <vector>

namespace llm {

namespace presets {

nn::TransformerConfig tiny(int vocab_size);
nn::TransformerConfig small(int vocab_size);
nn::TransformerConfig base(int vocab_size);
nn::TransformerConfig base_medium(int vocab_size);
nn::TransformerConfig base_large(int vocab_size);
nn::TransformerConfig medium(int vocab_size);
nn::TransformerConfig large(int vocab_size);
nn::TransformerConfig llama_tiny(int vocab_size);
nn::TransformerConfig llama_small(int vocab_size);
nn::TransformerConfig llama_base(int vocab_size);
nn::TransformerConfig llama_base_medium(int vocab_size);
nn::TransformerConfig llama_base_large(int vocab_size);
nn::TransformerConfig llama_medium(int vocab_size);
nn::TransformerConfig llama_large(int vocab_size);
nn::TransformerConfig qwen_tiny(int vocab_size);
nn::TransformerConfig qwen_small(int vocab_size);
nn::TransformerConfig qwen_base(int vocab_size);
nn::TransformerConfig qwen_base_medium(int vocab_size);
nn::TransformerConfig qwen_base_large(int vocab_size);
nn::TransformerConfig qwen_medium(int vocab_size);
nn::TransformerConfig qwen_large(int vocab_size);
nn::TransformerConfig gpt2_tiny(int vocab_size);
nn::TransformerConfig gpt2_small(int vocab_size);
nn::TransformerConfig gpt2_base(int vocab_size);
nn::TransformerConfig gpt2_base_medium(int vocab_size);
nn::TransformerConfig gpt2_base_large(int vocab_size);
nn::TransformerConfig gpt2_medium(int vocab_size);
nn::TransformerConfig gpt2_large(int vocab_size);
nn::TransformerConfig gpt3_tiny(int vocab_size);
nn::TransformerConfig gpt3_small(int vocab_size);
nn::TransformerConfig gpt3_base(int vocab_size);
nn::TransformerConfig gpt3_base_medium(int vocab_size);
nn::TransformerConfig gpt3_base_large(int vocab_size);
nn::TransformerConfig gpt3_medium(int vocab_size);
nn::TransformerConfig gpt3_large(int vocab_size);
nn::TransformerConfig bert_tiny(int vocab_size);
nn::TransformerConfig bert_small(int vocab_size);
nn::TransformerConfig bert_base(int vocab_size);
nn::TransformerConfig bert_base_medium(int vocab_size);
nn::TransformerConfig bert_base_large(int vocab_size);
nn::TransformerConfig bert_medium(int vocab_size);
nn::TransformerConfig bert_large(int vocab_size);

}  // namespace presets

// Contiguous decay / no-decay runs. Concatenating group.params is parameters() order.
std::vector<nn::AdamWGroup> adamw_groups(const nn::Transformer& model, float weight_decay);

}  // namespace llm
