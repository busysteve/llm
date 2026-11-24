# How to update a model from a reward

`reinforce` is the reinforcement-learning example in [`reinforce.cpp`](reinforce.cpp). It loads `tokenizer.bpe` and `model.bin` from a directory written by [`train_data`](train_data.md) or [`finetune`](finetune.md), samples completions for a prompt list, and keeps training a causal checkpoint. The reward is 1 when a completion contains a string you choose, and 0 otherwise. It writes a new directory. It is a separate program from `llm_tests`.

A bidirectional checkpoint, including `bert-*`, is rejected. The library call is `RlTrainer`. The string reward is only this program. A caller of `RlTrainer` passes its own reward function.

Build from the repository root. Requirements are in the [README](../README.md).

```bash
./build.sh amd       # or: ./build.sh opencl, ./build.sh cuda, ./build.sh arc
cmake --build build -j --target reinforce
```

## Prompts

One prompt per line. Blank lines are ignored. A line whose first character is `#` is ignored.

```
# short prompts for the reward run
The morning was
A small boat
```

Use text you have the rights to train on.

## Run

`--from` is read only. `--out` has to be a different directory. The new directory receives `tokenizer.bpe` and `model.bin`, so [`chat`](chat.md) can load `--out`.

```bash
./build/reinforce --from run --prompts prompts.txt --contains "red" --out run-rl \
  --steps 50 --group 4 --lr 1e-5 --kl 0.02 --max-new 32
```

The default peak learning rate is `1e-5`. Warmup defaults to 0. The cosine floor is one tenth of `--lr`. The schedule starts at step 1. With `--warmup 0` and `--steps` greater than 0, step `--steps` is the floor when warmup is shorter than the run.

`--group 1` uses the raw reward. A larger group subtracts that group's mean reward, so equal rewards in the group produce advantage 0. There is no division by the group standard deviation.

`--kl 0` trains the policy alone. A positive `--kl` loads a second copy of `--from` and adds that coefficient times the full-vocabulary KL. That copy is not trained.

The run builds a fresh AdamW. Weight decay defaults to `0.1`. The checkpoint's old moments are not kept.
