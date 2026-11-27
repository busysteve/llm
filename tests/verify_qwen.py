"""Compare an imported dense Qwen checkpoint against a local Transformers snapshot."""

import argparse
from pathlib import Path
import subprocess

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def probe(executable, model_dir, *args, stdin=None):
    mode, *mode_args = args
    return subprocess.run(
        [str(executable), "--hf-probe", mode, str(model_dir), *mode_args],
        input=stdin,
        text=True,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def probe_ids(executable, model_dir, *args):
    return [int(id_) for id_ in probe(executable, model_dir, *args).split()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("imported", type=Path)
    parser.add_argument("--probe", type=Path, default=Path("build/llm_tests"))
    args = parser.parse_args()
    executable = args.probe.resolve()

    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.snapshot, local_files_only=True, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    family = {"qwen2": "qwen2.5", "qwen3": "qwen3"}[model.config.model_type]
    stops = model.generation_config.eos_token_id
    if isinstance(stops, int):
        stops = [stops]
    assert probe_ids(executable, args.imported, "stops") == stops

    for text in ("Cafe\u0301, 世界! 12", " It's 42.\n\n  Next line.", "😀 <|im_end|>"):
        expected = tokenizer.encode(text, add_special_tokens=False)
        assert probe_ids(executable, args.imported, "text", text) == expected, repr(text)

    question = "What is the capital of France?"
    user, answer = "Name a fruit.", "An apple."
    if family == "qwen3":
        answer = "<think>\nA familiar example.\n</think>\n\n" + answer
    for thinking in (("on", "off") if family == "qwen3" else ("on",)):
        for history in ([], [user, answer]):
            messages = []
            if history:
                messages.extend([{"role": "user", "content": user}, {"role": "assistant", "content": answer}])
            messages.append({"role": "user", "content": question})
            options = {"enable_thinking": thinking == "on"} if family == "qwen3" else {}
            expected = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **options)
            got = probe_ids(executable, args.imported, "chat", family, thinking, question, *history)
            assert got == expected, (thinking, bool(history))

    # Use the same sequence for both decoders. Drive cached decode with tokens
    # selected by the reference, and separately check the C++ greedy IDs.
    mode = "off" if family == "qwen3" else "on"
    prompt = probe_ids(executable, args.imported, "chat", family, mode, question)
    allowed = np.array(sorted(set(tokenizer.get_vocab().values())))
    inputs = torch.tensor([prompt])
    rows, continuation = [], []
    with torch.inference_mode():
        cache = None
        for _ in range(4):
            result = model(inputs, past_key_values=cache, use_cache=True)
            row = result.logits[0, -1].cpu().numpy()
            next_id = int(allowed[row[allowed].argmax()])
            rows.append(row)
            continuation.append(next_id)
            inputs = torch.tensor([[next_id]])
            cache = result.past_key_values
    ids = " ".join(map(str, prompt + continuation[:-1]))
    output = probe(executable, args.imported, "logits", str(len(prompt)), stdin=ids)
    actual = np.array([np.fromstring(row, dtype=np.float32, sep=" ") for row in output.splitlines()])
    np.testing.assert_allclose(actual, np.array(rows), atol=1e-3, rtol=1e-4)
    assert list(allowed[actual[:, allowed].argmax(axis=1)]) == continuation
    print(f"{family}: token IDs, chat templates, stop IDs, prefill, and cached decode passed")


if __name__ == "__main__":
    main()
