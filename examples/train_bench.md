# How to run train_bench

`train_bench` is the training benchmark in [`train_bench.cpp`](train_bench.cpp). It times eight optimizer steps of `presets::tiny` after one warmup step and prints tokens per second. The run has to finish with a finite loss and a changed token embedding. It takes no arguments. It is a separate program from `llm_tests`.

Build from the repository root. Requirements are in the [README](../README.md).

```bash
./build.sh amd       # or: ./build.sh opencl, ./build.sh cuda, ./build.sh arc, ./build.sh agx
cmake --build build -j --target train_bench
./build/train_bench
```

For the CUDA backend, `CMAKE_CUDA_ARCHITECTURES` defaults to `86`. Set it to the GPU you have before `./build.sh cuda`. For ROCm, `CMAKE_HIP_ARCHITECTURES` defaults to `gfx1152`. Set it before `./build.sh amd`. For Arc, `./build.sh arc` uses `icpx` and JIT SPIR-V. There is no architecture flag. For Apple AGX, `./build.sh agx` uses Metal and exits on a non-Apple host.

## What it trains on

The corpus is 4096 token ids drawn from vocab 320 with seed 1. They stay in host memory. There is no tokenizer and no token file.

## Training

The model is `presets::tiny` at vocab 320: 4 layers, width 256, 4 heads, feed-forward width 768, learned positions, context 1024. Activations follow the library default. The schedule holds the learning rate at 3e-4.

| Setting | Value |
|---|---|
| Warmup steps | 1, not timed |
| Timed steps | 8 |
| Sequence length | 64 |
| Batch size | 4 |
| Gradient accumulation | 1 |
| Learning rate | 3e-4 |
| Weight decay | 0.1 |
| Eval | off |
| Checkpoints | off |

The warmup step launches the kernels. The clock starts after that step has finished on the device and stops after the eighth timed step has finished. Each timed step scores `batch * seq` tokens, so the run scores 2048 tokens. The step log is printed, then one summary line:

```
benchmark preset=tiny layers=4 d_model=256 heads=4 d_ff=768 vocab=320 batch=4 seq=64 steps=8 tokens=2048 seconds=<seconds> tokens_per_s=<rate>
```

A successful run exits 0. The temp directory `llm-train-bench/` is removed before the process exits. This program leaves no `model.bin`.

## When it fails

A failed run prints a reason on stderr and exits 1.

| Message | Meaning |
|---|---|
| `warmup` | the warmup step did not finish |
| `steps` | the timed run did not finish at step 9 |
| `weights` | the token embedding is missing, or it did not change |
| `benchmark` | the last loss is missing or not finite, the clock did not advance, or the AdamW step is not 9 |
| `train_bench: ...` | the dataset or the fit loop threw |

Library details for the preset and the trainer are in [USAGE.md](../USAGE.md).
