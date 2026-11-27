#include "llm/chat_template.hpp"
#include "llm/tokenizer.hpp"
#include "nn/index_tensor.hpp"
#include "nn/transformer.hpp"

#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace {

template <typename T>
void write_row(const std::vector<T>& row) {
  for (const T value : row) std::cout << value << ' ';
  std::cout << '\n';
}

void score(const std::filesystem::path& dir, const llm::Tokenizer& tok, int prompt_len) {
  nn::Context ctx;
#ifdef LLM_NN_ACTIVATION_DTYPE
  ctx.set_activation_dtype(nn::DType::Float32);
#endif
  auto loaded = nn::Transformer::load(ctx, dir / "model.bin", nn::LoadMode::Inference);
  auto& model = loaded.model;
  const auto& cfg = std::get<nn::TransformerConfig>(model.network().config());
  if (!cfg.causal) throw std::invalid_argument("causal");
  tok.unavailable_ids(cfg.vocab_size);
  std::vector<int64_t> ids;
  for (int32_t id; std::cin >> id;) {
    if (!tok.has_token(id)) throw std::invalid_argument("token id");
    ids.push_back(id);
  }
  if (!std::cin.eof() || prompt_len < 1 || static_cast<size_t>(prompt_len) > ids.size() ||
      ids.size() > static_cast<size_t>(cfg.max_seq_len)) {
    throw std::invalid_argument("prompt length");
  }
  model.eval();
  nn::Transformer::GenerateState cache;
  nn::IndexTensor prompt(ctx, {1, prompt_len});
  prompt.copy_from_host(ids.data(), prompt_len);
  ctx.synchronize();
  std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
  write_row(model.prefill(prompt, cache, static_cast<int>(ids.size())).to_host());
  if (ids.size() == static_cast<size_t>(prompt_len)) return;
  nn::IndexTensor next(ctx, {1, 1});
  for (size_t i = static_cast<size_t>(prompt_len); i < ids.size(); ++i) {
    next.copy_from_host(ids.data() + i, 1);
    ctx.synchronize();
    write_row(model.decode_step(next, cache).to_host());
  }
}

}  // namespace

int run_hf_probe(int argc, char** argv) {
  try {
    if (argc < 2) throw std::invalid_argument("usage: --hf-probe MODE MODEL_DIR ...");
    const std::string_view mode = argv[0];
    const std::filesystem::path dir = argv[1];
    const auto tok = llm::Tokenizer::load_directory(dir);
    if (mode == "text" && argc == 3) {
      write_row(tok->encode(argv[2]));
    } else if (mode == "chat" && (argc == 5 || argc == 7)) {
      const std::string_view family = argv[2];
      if (family != "qwen3" && family != "qwen2.5") throw std::invalid_argument("family");
      const std::string_view thinking = argv[3];
      if (thinking != "on" && thinking != "off") throw std::invalid_argument("thinking");
      const auto wrap = family == "qwen3" ? llm::ChatWrap::Qwen3 : llm::ChatWrap::Qwen25;
      std::vector<llm::ChatTurn> turns;
      if (argc == 7) turns.push_back({argv[5], argv[6]});
      write_row(tok->encode(llm::chat_prompt(turns, argv[4], wrap, thinking == "on")));
    } else if (mode == "stops" && argc == 2) {
      const auto stops = tok->stop_ids();
      if (!stops.empty()) write_row(stops);
      else if (tok->eos_id() >= 0) write_row(std::vector<int32_t>{tok->eos_id()});
    } else if (mode == "logits" && argc == 3) {
      score(dir, *tok, std::stoi(argv[2]));
    } else {
      throw std::invalid_argument("probe arguments");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "hf probe: " << error.what() << '\n';
    return 1;
  }
}
