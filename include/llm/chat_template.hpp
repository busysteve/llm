#pragma once

#include <string>
#include <string_view>

namespace llm {

// How chat wraps a typed line. Dialogue is User:/Assistant:. Plain is the line
// alone. Qwen is ChatML, which a Qwen GGUF expects.
enum class ChatWrap { Dialogue, Plain, Qwen };

inline std::string chat_completed_turn(std::string_view line, std::string_view reply, ChatWrap wrap) {
  std::string text;
  if (wrap == ChatWrap::Plain) {
    text.append(line);
    text.append(reply);
    text.push_back('\n');
    return text;
  }
  if (wrap == ChatWrap::Qwen) {
    text += "<|im_start|>user\n";
    text.append(line);
    text += "<|im_end|>\n<|im_start|>assistant\n";
    text.append(reply);
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

inline std::string chat_prompt_suffix(std::string_view line, ChatWrap wrap) {
  if (wrap == ChatWrap::Plain) return std::string(line);
  if (wrap == ChatWrap::Qwen) {
    std::string text = "<|im_start|>user\n";
    text.append(line);
    text += "<|im_end|>\n<|im_start|>assistant\n";
    return text;
  }
  std::string text = "User: ";
  text.append(line);
  text += "\nAssistant:";
  return text;
}

inline std::string chat_cut_reply(std::string reply, ChatWrap wrap) {
  if (wrap == ChatWrap::Qwen) {
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
