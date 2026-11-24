# gpt2-base Architecture

The `gpt2_base` preset in the `llm` library (`llm::presets::gpt2_base`) defines a decoder-only Transformer with the GPT-2 block: learned absolute positions, pre-norm LayerNorm, causal attention, and a GELU MLP of width `4 * d_model`.

## Specs

| Spec | Value |
|------|-------|
| Layers (`num_layers`) | 12 |
| Model dimension (`d_model`) | 768 |
| Attention heads (`num_heads`) | 12 (head dim = 64) |
| Feed-forward intermediate (`d_ff`) | 3072 (GELU; fc1 writes 3072, not 6144) |
| Max sequence length | 1024 |
| Positional encoding | learned absolute (`pos_emb`, 1024 × 768) |
| Normalization | LayerNorm, epsilon `1e-5` |
| Activation | GELU, the erf form (`0.5 * x * (1 + erf(x / sqrt(2)))`), not the tanh approximation |
| Bias | true, on Q, K, V, the attention output, both MLP linears, and the LM head |
| Tie embeddings | false (`nn` rejects `tie_embeddings`; OpenAI ties `wte` to the LM head) |
| Dropout | 0 (OpenAI trained GPT-2 with dropout 0.1; every preset in this library uses 0) |
| Parameter count (untied, vocab ≈ 16000) | 110,434,432 (~110.43 M) |

These match the values returned by `llm::presets::gpt2_base(vocab_size)` and the counts documented in the README / USAGE.md. `gpt2_tiny` is 4 × 256 with `d_ff` 1024. `gpt2_small` is 6 × 512 with `d_ff` 2048. Both keep context 1024. `train_data --preset gpt2-base` trains this shape and saves a version-2 checkpoint.

This is the block to train. It does not load OpenAI's GPT-2 weights. `gguf from` still rejects a public GPT-2 GGUF, and `gguf llama` rejects a checkpoint that uses GELU or linear bias. An nnllm GGUF keeps both.

## Architecture Diagram

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '14px'}}}%%
graph TD
    Tokens["Input Token IDs<br/>(batch, seq_len)"] --> Embed["Token Embeddings<br/>vocab × 768<br/>(untied)"]
    Pos["Position Embeddings<br/>1024 × 768"] --> AddPos["+"]
    Embed --> AddPos

    AddPos --> Blocks

    subgraph Blocks["12 × Transformer Blocks"]
        direction TB
        B1["Block 1"] --> BN["... Block 12"]
    end

    subgraph BlockDetail["Transformer Block (repeated 12×)"]
        direction TB
        X["Input (N, T, 768)"] --> LN1["LayerNorm<br/>ε = 1e-5"]
        LN1 --> Attn["Causal attention<br/>12 heads × dim 64<br/>Q, K, V, output have bias"]
        Attn --> Add1["+ Residual"]
        Add1 --> LN2["LayerNorm<br/>ε = 1e-5"]
        LN2 --> FC1["Linear<br/>768 → 3072<br/>(bias)"]
        FC1 --> Gelu["GELU"]
        Gelu --> FC2["Linear<br/>3072 → 768<br/>(bias)"]
        FC2 --> Add2["+ Residual"]
        Add2 --> Y["Output (N, T, 768)"]
    end

    BN --> Final["Final LayerNorm<br/>ε = 1e-5"]
    Final --> LMHead["LM Head<br/>Linear 768 → vocab<br/>(untied, bias)"]
    LMHead --> Logits["Logits<br/>(batch, seq_len, vocab)"]

    classDef spec fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    class Tokens,Embed,Pos,Attn,FC1,Gelu,FC2,LMHead spec
```

The diagram shows the data flow for a forward pass. Token embeddings and learned position embeddings are added, then each of the 12 identical blocks applies pre-norm LayerNorm → causal attention → residual, then pre-norm LayerNorm → GELU MLP → residual. There is no RoPE and no RMSNorm. Embeddings are not tied to the LM head.

## Layer and head footprint

Each weight is float32, so each parameter is four bytes. The vocab in this count is 16000, the same vocab as the README Size column. Activations and Adam moments are not drawn. LayerNorm stores a weight and a bias. Every linear stores a bias, including the LM head.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '14px'}}}%%
graph TD
    Tok["tok_emb.weight<br/>16000 × 768 = 12,288,000<br/>49,152,000 bytes"] --> Add
    Pos["pos_emb.weight<br/>1024 × 768 = 786,432<br/>3,145,728 bytes"] --> Add
    Add["added, no new weights"] --> Blocks

    subgraph Blocks["12 blocks · 7,087,872 each · 85,054,464 params · 340,217,856 bytes"]
        direction TB
        LN1["ln1.weight + ln1.bias<br/>768 + 768 = 1,536<br/>6,144 bytes"] --> Q
        LN1 --> K
        LN1 --> V
        Q["attn.q.weight + bias<br/>12 heads × 64 = 768<br/>768 × 768 + 768 = 590,592<br/>2,362,368 bytes"] --> Out
        K["attn.k.weight + bias<br/>12 heads × 64 = 768<br/>768 × 768 + 768 = 590,592<br/>2,362,368 bytes"] --> Out
        V["attn.v.weight + bias<br/>12 heads × 64 = 768<br/>768 × 768 + 768 = 590,592<br/>2,362,368 bytes"] --> Out
        Out["attn.out.weight + bias<br/>768 × 768 + 768 = 590,592<br/>2,362,368 bytes"] --> LN2
        LN2["ln2.weight + ln2.bias<br/>768 + 768 = 1,536<br/>6,144 bytes"] --> FC1
        FC1["fc1.weight + bias<br/>GELU 3072<br/>3072 × 768 + 3072 = 2,362,368<br/>9,449,472 bytes"] --> FC2
        FC2["fc2.weight + bias<br/>768 × 3072 + 768 = 2,360,064<br/>9,440,256 bytes"]
    end

    Blocks --> Final["final_ln.weight + bias<br/>768 + 768 = 1,536<br/>6,144 bytes"]
    Final --> LM["lm_head.weight + bias<br/>16000 × 768 + 16000 = 12,304,000<br/>49,216,000 bytes"]
    LM --> Total["float32 weights<br/>110,434,432 parameters<br/>441,737,728 bytes<br/>441.74 MB"]

    classDef spec fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    class Tok,Pos,Q,K,V,Out,FC1,FC2,LM,Total spec
```

One block is `1,536 + 4 × 590,592 + 1,536 + 2,362,368 + 2,360,064 = 7,087,872` parameters. Twelve blocks are 85,054,464. The token table is 12,288,000, the position table is 786,432, the final LayerNorm is 1,536, and the LM head with its bias is 12,304,000. The sum is 110,434,432 parameters, 441,737,728 bytes, 441.74 MB. Each of the 12 heads has dimension 64, so Q, K, V, and the output projection are square in `d_model`. GELU writes `d_ff` once, not `2 × d_ff`.

See also:

- [preset-gpt2.md](preset-gpt2.md) — usage, benefits, and applications
- [README.md](../README.md) — preset table and parameter counts
- [USAGE.md](../USAGE.md) — full programmer notes
- [gpt3-base-architecture.md](gpt3-base-architecture.md) — the same block with context 2048
- [bert-base-architecture.md](bert-base-architecture.md) — the same dense block with bidirectional attention and context 512
- `include/llm/presets.hpp` / `src/presets.cpp` — source of the config
