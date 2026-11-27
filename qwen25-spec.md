# Qwen2.5 fine-tuning dataset specification

For Qwen2.5 supervised fine-tuning with Hugging Face TRL, use **one JSON object per line, with one target reply per object**. Put everything the model sees in `prompt`, and put the reply you want it to learn in `completion`. Both fields are lists of chat messages. TRL applies Qwen’s chat template and computes training loss on the completion when `completion_only_loss=True`. [TRL dataset formats](https://huggingface.co/docs/trl/en/dataset_formats)

## `train.jsonl`

```jsonl
{"prompt":[{"role":"user","content":"Route this ticket. Return JSON with a team field. Ticket: I was charged twice."}],"completion":[{"role":"assistant","content":"{\"team\":\"billing\"}"}]}
{"prompt":[{"role":"user","content":"Route this ticket. Return JSON with a team field. Ticket: My password reset link expired."}],"completion":[{"role":"assistant","content":"{\"team\":\"account_access\"}"}]}
{"prompt":[{"role":"user","content":"Route this ticket. Return JSON with a team field. Ticket: Something went wrong."}],"completion":[{"role":"assistant","content":"{\"team\":\"needs_review\"}"}]}
```

Use the same structure for `validation.jsonl`, with different cases. If you already have data in a `messages` array, put the final assistant message in `completion` and all preceding messages in `prompt`. That preserves system instructions and prior turns as context. For structured output, put the serialized JSON inside the assistant’s `content` string, as shown above.

## What makes a good example

- **Use realistic inputs.** Include the wording, typos, missing details, and ambiguity you expect in production.
- **Write the exact desired output.** If your application expects a JSON object with a `team` field, use that schema consistently. Include examples of how to handle unclear requests.
- **Cover each behavior you care about.** For ticket routing, include every supported team and cases that are easy to confuse, such as a failed payment versus a duplicate charge.
- **Split by source before making variations.** Keep turns from the same conversation and paraphrases of the same example in one split. This makes validation a better measure of performance on new inputs.
- **Check token lengths after chat formatting.** If you set `max_length=1024`, keep the formatted prompt and completion within 1,024 tokens so the target reply survives truncation. [TRL SFT configuration](https://huggingface.co/docs/trl/en/sft_trainer#trl.SFTConfig)
