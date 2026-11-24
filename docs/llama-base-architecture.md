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

## Layer and head footprint

Each weight is float32, so each parameter is four bytes. The vocab in this count is 16000, the same vocab as the README Size column. Activations and Adam moments are not drawn.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '14px'}}}%%
graph TD
    Tok["tok_emb.weight<br/>16000 × 768 = 12,288,000<br/>49,152,000 bytes"] --> Blocks

    subgraph Blocks["12 blocks · 7,079,424 each · 84,953,088 params · 339,812,352 bytes"]
        direction TB
        LN1["ln1.weight<br/>768<br/>3,072 bytes"] --> Q
        LN1 --> K
        LN1 --> V
        Q["attn.q.weight<br/>12 heads × 64 = 768<br/>768 × 768 = 589,824<br/>2,359,296 bytes"] --> Out
        K["attn.k.weight<br/>12 heads × 64 = 768<br/>768 × 768 = 589,824<br/>2,359,296 bytes"] --> Out
        V["attn.v.weight<br/>12 heads × 64 = 768<br/>768 × 768 = 589,824<br/>2,359,296 bytes"] --> Out
        Out["attn.out.weight<br/>768 × 768 = 589,824<br/>2,359,296 bytes"] --> LN2
        LN2["ln2.weight<br/>768<br/>3,072 bytes"] --> FC1
        FC1["fc1.weight<br/>SwiGLU 2 × 2048 = 4096<br/>4096 × 768 = 3,145,728<br/>12,582,912 bytes"] --> FC2
        FC2["fc2.weight<br/>768 × 2048 = 1,572,864<br/>6,291,456 bytes"]
    end

    Blocks --> Final["final_ln.weight<br/>768<br/>3,072 bytes"]
    Final --> LM["lm_head.weight<br/>16000 × 768 = 12,288,000<br/>49,152,000 bytes"]
    LM --> Total["float32 weights<br/>109,529,856 parameters<br/>438,119,424 bytes<br/>438.12 MB"]

    classDef spec fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    class Tok,Q,K,V,Out,FC1,FC2,LM,Total spec
```

One block is `768 + 4 × 589,824 + 768 + 3,145,728 + 1,572,864 = 7,079,424` parameters. Twelve blocks are 84,953,088. The token table and the LM head are 12,288,000 each, and the final RMSNorm is 768. The sum is 109,529,856 parameters, 438,119,424 bytes, 438.12 MB. Query, key, and value each hold 12 heads of dimension 64, so each of those matrices is square in `d_model`. RoPE stores no table.

See also:
- [preset-llama.md](preset-llama.md) — usage, benefits, and applications
- [README.md](../README.md) — preset table and parameter counts
- [USAGE.md](../USAGE.md) — full programmer notes
- `include/llm/presets.hpp` / `src/presets.cpp` — source of the config
