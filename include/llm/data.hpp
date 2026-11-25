#pragma once

#include "nn/context.hpp"
#include "nn/index_tensor.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <random>
#include <vector>

namespace llm {

struct Batch {
  nn::IndexTensor input_ids;  // (N, T) int64
  nn::IndexTensor labels;     // (N, T) int64
};

class TextDataset {
 public:
  // eos_id < 0 draws any span. A non-negative eos_id keeps each window inside
  // one document: eos may be the last label, and it is not an input token.
  TextDataset(std::filesystem::path tokens_bin, std::filesystem::path meta_path,
              int32_t eos_id = -1);
  // Ids already on the host. Checks match write_token_file.
  TextDataset(std::vector<int32_t> ids, int32_t vocab_size, int32_t eos_id = -1);
  int64_t numel() const;
  int32_t vocab_size() const;
  int32_t eos_id() const { return eos_id_; }
  const std::vector<int32_t>& ids() const { return ids_; }
  Batch next(nn::Context& ctx, int batch_size, int seq_len, std::mt19937& rng);

 private:
  void build_starts(int seq_len);
  struct StartSpan {
    int64_t origin = 0;
    int64_t global = 0;
    int64_t count = 0;
  };

  std::vector<int32_t> ids_;
  int64_t numel_ = 0;
  int32_t vocab_size_ = 0;
  int32_t eos_id_ = -1;
  int cached_seq_ = -1;
  int64_t n_starts_ = 0;
  std::vector<StartSpan> starts_;
};

// Whole documents, each one ending on eos_id when that id is present.
// The valid side is a suffix of those documents, near fraction of the tokens,
// and each side has a document long enough for one window.
struct TokenHoldout {
  std::vector<int32_t> train;
  std::vector<int32_t> valid;
  size_t train_files = 0;
  size_t valid_files = 0;
};

TokenHoldout holdout_documents(const std::vector<int32_t>& ids, int seq_len, float fraction,
                               int32_t eos_id);

// The producer emits batches in token order. Files are published after validation.
using TokenBatchSink = std::function<void(const std::vector<int32_t>&)>;
int64_t write_token_batches(const std::filesystem::path& tokens_bin,
                            const std::filesystem::path& meta_path,
                            int32_t vocab_size, int64_t min_tokens,
                            const std::function<void(const TokenBatchSink&)>& produce);

void write_token_file(const std::filesystem::path& tokens_bin,
                      const std::filesystem::path& meta_path,
                      const std::vector<int32_t>& ids,
                      int32_t vocab_size);

}  // namespace llm
