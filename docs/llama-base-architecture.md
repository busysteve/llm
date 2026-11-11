# llama-base Architecture

The `llama_base` preset in the `llm` library (`llm::presets::llama_base`) defines a decoder-only Transformer with RoPE and RMSNorm, matching the Llama-style block used by the library.

## Specs

| Spec | Value |
|------|-------|
| Layers (`num_layers`) | 12 |
| Model dimension (`d_model`) | 768 |
| Attention heads (`num_heads`) | 12 (head dim = 64) |
| Feed-forward intermediate (`d_ff`) | 2048 (SwiGLU; fc1 expands to 4096) |
| Max sequence length | 1024 |
| Positional encoding | RoPE (adjacent-pair, base 10000) |
| Normalization | RMSNorm |
| Activation | SwiGLU |
| Bias | false |
| Tie embeddings | false |
| Dropout | 0 |
| Parameter count (untied, vocab ≈ 16000) | 109,529,856 (~109.53 M) |

These match the values returned by `llm::presets::llama_base(vocab_size)` and the counts documented in the README / USAGE.md.

## Architecture Diagram

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '14px'}}}%%
graph TD
    Tokens["Input Token IDs<br/>(batch, seq_len)"] --> Embed["Token Embeddings<br/>vocab × 768<br/>(untied)"]

    Embed --> Blocks

    subgraph Blocks["12 × Transformer Blocks"]
        direction TB
        B1["Block 1"] --> BN["... Block 12"]
    end

    subgraph BlockDetail["Transformer Block (repeated 12×)"]
        direction TB
        X["Input (N, T, 768)"] --> RMS1["RMSNorm"]
        RMS1 --> Attn["Multi-Head Attention<br/>12 heads × dim 64<br/>RoPE on Q & K<br/>no bias, no dropout"]
        Attn --> Add1["+ Residual"]
        Add1 --> RMS2["RMSNorm"]
        RMS2 --> FC1["Linear<br/>768 → 4096<br/>(no bias)"]
        FC1 --> Gate["SwiGLU<br/>(gate → 2048)"]
        Gate --> FC2["Linear<br/>2048 → 768<br/>(no bias)"]
        FC2 --> Add2["+ Residual"]
        Add2 --> Y["Output (N, T, 768)"]
    end

    BN --> Final["Final RMSNorm"]
    Final --> LMHead["LM Head<br/>Linear 768 → vocab<br/>(untied, no bias)"]
    LMHead --> Logits["Logits<br/>(batch, seq_len, vocab)"]

    classDef spec fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    class Tokens,Embed,Attn,FC1,Gate,FC2,LMHead spec
```

The diagram shows the data flow for a forward pass. Each of the 12 identical blocks applies pre-norm RMSNorm → attention (with RoPE) → residual, then pre-norm RMSNorm → SwiGLU FFN → residual. There is no learned absolute position table (those parameters are omitted compared with the non-llama `base` preset). All linear layers are bias-free, and embeddings are not tied to the LM head.

See also:
- [README.md](../README.md) — preset table and parameter counts
- [USAGE.md](../USAGE.md) — full programmer notes
- `include/llm/presets.hpp` / `src/presets.cpp` — source of the config
