# qwen-base Architecture

The `qwen_base` preset in the `llm` library (`llm::presets::qwen_base`) defines a decoder-only Transformer with the Qwen3 block: grouped-query attention, head dim 128, QK-norm, RoPE base `1e6`, and RMSNorm epsilon `1e-6`.

## Specs

| Spec | Value |
|------|-------|
| Layers (`num_layers`) | 12 |
| Model dimension (`d_model`) | 768 |
| Query heads (`num_heads`) | 12 |
| KV heads (`num_kv_heads`) | 4 (each repeated ×3) |
| Head dim (`head_dim`) | 128 |
| Q projection | 768 → 1536 (`12 × 128`) |
| K and V projections | 768 → 512 (`4 × 128`) |
| Output projection | 1536 → 768 |
| QK-norm | RMSNorm, width 128, epsilon `1e-6`, before RoPE |
| Feed-forward intermediate (`d_ff`) | 2048 (SwiGLU; fc1 expands to 4096) |
| Max sequence length | 1024 |
| Positional encoding | RoPE (adjacent-pair, base `1e6`) |
| Normalization | RMSNorm, epsilon `1e-6` |
| Activation | SwiGLU |
| Bias | false |
| Tie embeddings | false |
| Dropout | 0 |
| Parameter count (untied, vocab ≈ 16000) | 118,970,112 (~118.97 M) |

These match the values returned by `llm::presets::qwen_base(vocab_size)` and the counts documented in the README / USAGE.md. `train_data --preset qwen-base` trains this shape and saves a version-4 checkpoint.

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
        X["Input (N, T, 768)"] --> RMS1["RMSNorm<br/>ε = 1e-6"]
        RMS1 --> Attn["Grouped-query attention<br/>12 query heads × dim 128<br/>4 KV heads, repeated ×3<br/>QK-norm, then RoPE base 1e6"]
        Attn --> Add1["+ Residual"]
        Add1 --> RMS2["RMSNorm<br/>ε = 1e-6"]
        RMS2 --> FC1["Linear<br/>768 → 4096<br/>(no bias)"]
        FC1 --> Gate["SwiGLU<br/>(gate → 2048)"]
        Gate --> FC2["Linear<br/>2048 → 768<br/>(no bias)"]
        FC2 --> Add2["+ Residual"]
        Add2 --> Y["Output (N, T, 768)"]
    end

    subgraph AttnDetail["Attention (inside each block)"]
        direction TB
        H["Hidden (N, T, 768)"] --> Q["Q linear<br/>768 → 1536"]
        H --> K["K linear<br/>768 → 512"]
        H --> V["V linear<br/>768 → 512"]
        Q --> QN["Q RMSNorm<br/>width 128"]
        K --> KN["K RMSNorm<br/>width 128"]
        QN --> Rope["RoPE on Q and K<br/>adjacent pairs, base 1e6"]
        KN --> RepK["Repeat K ×3<br/>4 heads → 12"]
        RepK --> Rope
        V --> RepV["Repeat V ×3<br/>4 heads → 12"]
        Rope --> Scores["Scores and output<br/>1536 → 768"]
        RepV --> Scores
    end

    BN --> Final["Final RMSNorm<br/>ε = 1e-6"]
    Final --> LMHead["LM Head<br/>Linear 768 → vocab<br/>(untied, no bias)"]
    LMHead --> Logits["Logits<br/>(batch, seq_len, vocab)"]

    classDef spec fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    class Tokens,Embed,Attn,FC1,Gate,FC2,LMHead,Q,K,V,QN,KN,Rope,Scores spec
```

The diagram shows the data flow for a forward pass. Each of the 12 identical blocks applies pre-norm RMSNorm → attention → residual, then pre-norm RMSNorm → SwiGLU FFN → residual. Inside attention the order is the Q, K, and V linears, a split into heads, RMSNorm on Q and on K, a repeat of each KV head across the three query heads in its group, RoPE on Q and on the repeated K, then the output projection. Positions come from that RoPE, so the checkpoint has no position table. All linear layers are bias-free, and embeddings are not tied to the LM head.

## Layer and head footprint

Each weight is float32, so each parameter is four bytes. The vocab in this count is 16000, the same vocab as the README Size column. Activations and Adam moments are not drawn. Repeating a KV head across its three query heads adds no weights.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '14px'}}}%%
graph TD
    Tok["tok_emb.weight<br/>16000 × 768 = 12,288,000<br/>49,152,000 bytes"] --> Blocks

    subgraph Blocks["12 blocks · 7,866,112 each · 94,393,344 params · 377,573,376 bytes"]
        direction TB
        LN1["ln1.weight<br/>768<br/>3,072 bytes"] --> Q
        LN1 --> K
        LN1 --> V
        Q["attn.q.weight<br/>12 heads × 128 = 1536<br/>1536 × 768 = 1,179,648<br/>4,718,592 bytes"] --> QN
        K["attn.k.weight<br/>4 heads × 128 = 512<br/>512 × 768 = 393,216<br/>1,572,864 bytes"] --> KN
        V["attn.v.weight<br/>4 heads × 128 = 512<br/>512 × 768 = 393,216<br/>1,572,864 bytes"] --> Out
        QN["attn.q_norm.weight<br/>128<br/>512 bytes"] --> Out
        KN["attn.k_norm.weight<br/>128<br/>512 bytes"] --> Out
        Out["attn.out.weight<br/>1536 × 768 = 1,179,648<br/>4,718,592 bytes"] --> LN2
        LN2["ln2.weight<br/>768<br/>3,072 bytes"] --> FC1
        FC1["fc1.weight<br/>SwiGLU 2 × 2048 = 4096<br/>4096 × 768 = 3,145,728<br/>12,582,912 bytes"] --> FC2
        FC2["fc2.weight<br/>768 × 2048 = 1,572,864<br/>6,291,456 bytes"]
    end

    Blocks --> Final["final_ln.weight<br/>768<br/>3,072 bytes"]
    Final --> LM["lm_head.weight<br/>16000 × 768 = 12,288,000<br/>49,152,000 bytes"]
    LM --> Total["float32 weights<br/>118,970,112 parameters<br/>475,880,448 bytes<br/>475.88 MB"]

    classDef spec fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    class Tok,Q,K,V,QN,KN,Out,FC1,FC2,LM,Total spec
```

One block is `768 + 1,179,648 + 2 × 393,216 + 1,179,648 + 128 + 128 + 768 + 3,145,728 + 1,572,864 = 7,866,112` parameters. Twelve blocks are 94,393,344. The token table and the LM head are 12,288,000 each, and the final RMSNorm is 768. The sum is 118,970,112 parameters, 475,880,448 bytes, 475.88 MB. Q is wider than `d_model` because each of the 12 heads has dimension 128. K and V stay at 4 heads of that same dimension.

See also:
- [preset-qwen.md](preset-qwen.md) — usage, benefits, and applications
- [README.md](../README.md) — preset table and parameter counts
- [USAGE.md](../USAGE.md) — full programmer notes
- [llama-base-architecture.md](llama-base-architecture.md) — the same width with square heads and RoPE base 10000
- `include/llm/presets.hpp` / `src/presets.cpp` — source of the config
