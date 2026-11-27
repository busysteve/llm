#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace llm {

// How chat wraps a typed line. Dialogue is User:/Assistant:. Plain is the line
// alone. Qwen is ChatML, which a Qwen GGUF expects.
enum class ChatWrap { Dialogue, Plain, Qwen, Qwen25, Qwen3 };

inline bool chat_uses_chatml(ChatWrap wrap) {
  return wrap == ChatWrap::Qwen || wrap == ChatWrap::Qwen25 || wrap == ChatWrap::Qwen3;
}

inline std::string_view chat_prompt_prefix(ChatWrap wrap) {
  if (wrap == ChatWrap::Qwen25) {
    return "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n";
  }
  return {};
}

inline std::string_view qwen3_answer(std::string_view reply) {
  const size_t end = reply.rfind("</think>");
  if (end != std::string_view::npos) {
    reply.remove_prefix(end + std::string_view("</think>").size());
    while (!reply.empty() && reply.front() == '\n') reply.remove_prefix(1);
  }
  return reply;
}

inline std::string chat_completed_turn(std::string_view line, std::string_view reply, ChatWrap wrap) {
  std::string text;
  if (wrap == ChatWrap::Plain) {
    text.append(line);
    text.append(reply);
    text.push_back('\n');
    return text;
  }
  if (chat_uses_chatml(wrap)) {
    text += "<|im_start|>user\n";
    text.append(line);
    text += "<|im_end|>\n<|im_start|>assistant\n";
    text.append(wrap == ChatWrap::Qwen3 ? qwen3_answer(reply) : reply);
    text += "<|im_end|>\n";
    return text;
  }
  text += "User: ";
  text.append(line);
  text += "\nAssistant:";
  text.append(reply);
  text.push_back('\n');
  return text;
}

inline std::string chat_prompt_suffix(std::string_view line, ChatWrap wrap, bool thinking = true) {
  if (wrap == ChatWrap::Plain) return std::string(line);
  if (chat_uses_chatml(wrap)) {
    std::string text = "<|im_start|>user\n";
    text.append(line);
    text += "<|im_end|>\n<|im_start|>assistant\n";
    if (wrap == ChatWrap::Qwen3 && !thinking) text += "<think>\n\n</think>\n\n";
    return text;
  }
  std::string text = "User: ";
  text.append(line);
  text += "\nAssistant:";
  return text;
}

struct ChatTurn {
  std::string line;
  std::string reply;
};

inline std::string chat_prompt(const std::vector<ChatTurn>& turns, std::string_view line, ChatWrap wrap,
                               bool thinking = true) {
  std::string prompt(chat_prompt_prefix(wrap));
  for (const auto& turn : turns) prompt += chat_completed_turn(turn.line, turn.reply, wrap);
  prompt += chat_prompt_suffix(line, wrap, thinking);
  return prompt;
}

inline std::string chat_cut_reply(std::string reply, ChatWrap wrap) {
  if (chat_uses_chatml(wrap)) {
    const size_t end_tok = reply.find("<|im_end|>");
    const size_t next_turn = reply.find("\n<|im_start|>");
    size_t cut = std::string::npos;
    if (end_tok != std::string::npos) cut = end_tok;
    if (next_turn != std::string::npos && (cut == std::string::npos || next_turn < cut)) cut = next_turn;
    if (cut != std::string::npos) reply.resize(cut);
    return reply;
  }
  if (wrap == ChatWrap::Dialogue) {
    const size_t cut = reply.find("\nUser:");
    if (cut != std::string::npos) reply.resize(cut);
  }
  return reply;
}

inline std::string chat_shown_reply(std::string_view reply, ChatWrap wrap) {
  if (wrap == ChatWrap::Dialogue && !reply.empty() && reply.front() == ' ') reply.remove_prefix(1);
  return std::string(reply);
}

}  // namespace llm
