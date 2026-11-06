#pragma once

#include "nn/context.hpp"
#include "nn/index_tensor.hpp"

#include <cstdint>
#include <filesystem>
#include <random>
#include <vector>

namespace llm {

struct Batch {
  nn::IndexTensor input_ids;  // (N, T) int64
  nn::IndexTensor labels;     // (N, T) int64
};

class TextDataset {
 public:
  TextDataset(std::filesystem::path tokens_bin, std::filesystem::path meta_path);
  int64_t numel() const;
  int32_t vocab_size() const;
  Batch next(nn::Context& ctx, int batch_size, int seq_len, std::mt19937& rng);

 private:
  std::vector<int32_t> ids_;
  int64_t numel_ = 0;
  int32_t vocab_size_ = 0;
};

void write_token_file(const std::filesystem::path& tokens_bin,
                      const std::filesystem::path& meta_path,
                      const std::vector<int32_t>& ids,
                      int32_t vocab_size);

}  // namespace llm
