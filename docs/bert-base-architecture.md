# bert-base Architecture

The `bert_base` preset in the `llm` library (`llm::presets::bert_base`) defines an encoder Transformer with the GPT-2 dense block and BERT's attention pattern: learned absolute positions, a segment table, pre-norm LayerNorm, bidirectional attention, and a GELU MLP of width `4 * d_model`.

## Specs

| Spec | Value |
|------|-------|
| Layers (`num_layers`) | 12 |
| Model dimension (`d_model`) | 768 |
| Attention heads (`num_heads`) | 12 (head dim = 64) |
| Feed-forward intermediate (`d_ff`) | 3072 (GELU; fc1 writes 3072, not 6144) |
| Max sequence length | 512 |
| Positional encoding | learned absolute (`pos_emb`, 512 × 768) |
| Segment embeddings | `type_emb`, 2 × 768. `forward(tokens)` uses segment 0 |
| Normalization | LayerNorm, epsilon `1e-5` |
| Activation | GELU, the erf form (`0.5 * x * (1 + erf(x / sqrt(2)))`), not the tanh approximation |
| Attention | bidirectional (`causal` false). Every token attends to the whole sequence |
| Bias | true, on Q, K, V, the attention output, both MLP linears, and the LM head |
| Tie embeddings | false (`nn` rejects `tie_embeddings`) |
| Dropout | 0 |
| Parameter count (untied, vocab ≈ 16000) | 110,042,752 (~110.04 M) |

These match the values returned by `llm::presets::bert_base(vocab_size)` and the counts documented in the README / USAGE.md. `bert_tiny` is 4 × 256 with `d_ff` 1024. `bert_small` is 6 × 512 with `d_ff` 2048. Both keep context 512 and a segment table of size 2.

`train_data --preset bert-base` trains masked tokens and saves a version-6 checkpoint (AdamW). `save(path, nullptr)` of the same shape is version 5. The mask id is the tokenizer pad id. This library's byte-level BPE has no `<mask>` piece and does not insert pad during `encode`. Of the positions chosen for the loss (15% by default), 80% become that pad id, 10% become a random id, and 10% keep the original id. Other positions use ignore index `-100`.

This is the block to train from scratch. It does not load published BERT weights. There is no WordPiece tokenizer, no next-sentence pooler, and no sentence-pair corpus format. `forward(tokens, type_ids)` is the hook for a later pair of segments. `chat` and `Generator` refuse this checkpoint. `gguf llama` rejects bidirectional attention and a segment table.

## Architecture Diagram

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '14px'}}}%%
graph TD
    Tokens["Input Token IDs<br/>(batch, seq_len)"] --> Embed["Token Embeddings<br/>vocab × 768<br/>(untied)"]
    Pos["Position Embeddings<br/>512 × 768"] --> AddPos["+"]
    Types["Segment Embeddings<br/>2 × 768<br/>segment 0 unless type ids are passed"] --> AddType["+"]
    Embed --> AddPos
    AddPos --> AddType

    AddType --> Blocks

    subgraph Blocks["12 × Transformer Blocks"]
        direction TB
        B1["Block 1"] --> BN["... Block 12"]
    end

    subgraph BlockDetail["Transformer Block (repeated 12×)"]
        direction TB
        X["Input (N, T, 768)"] --> LN1["LayerNorm<br/>ε = 1e-5"]
        LN1 --> Attn["Bidirectional attention<br/>12 heads × dim 64<br/>Q, K, V, output have bias"]
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
    class Tokens,Embed,Pos,Types,Attn,FC1,Gelu,FC2,LMHead spec
```

The diagram shows the data flow for a forward pass. Token embeddings, learned position embeddings, and segment embeddings are added, then each of the 12 identical blocks applies pre-norm LayerNorm → bidirectional attention → residual, then pre-norm LayerNorm → GELU MLP → residual. There is no causal mask, no RoPE, and no RMSNorm. Embeddings are not tied to the LM head. `decode_step` and `prefill` throw `causal`. `Generator` appends the pad id and reads that position from `forward`.

## Layer and head footprint

Each weight is float32, so each parameter is four bytes. The vocab in this count is 16000, the same vocab as the README Size column. Activations and Adam moments are not drawn. The block matches `gpt2_base`. The position table is 512 rows, and a segment table of 2 rows is added. Bidirectional attention does not change the weight shapes.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '14px'}}}%%
graph TD
    Tok["tok_emb.weight<br/>16000 × 768 = 12,288,000<br/>49,152,000 bytes"] --> Add
    Pos["pos_emb.weight<br/>512 × 768 = 393,216<br/>1,572,864 bytes"] --> Add
    Types["type_emb.weight<br/>2 × 768 = 1,536<br/>6,144 bytes"] --> Add
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
    LM --> Total["float32 weights<br/>110,042,752 parameters<br/>440,171,008 bytes<br/>440.17 MB"]

    classDef spec fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    class Tok,Pos,Types,Q,K,V,Out,FC1,FC2,LM,Total spec
```

One block is `1,536 + 4 × 590,592 + 1,536 + 2,362,368 + 2,360,064 = 7,087,872` parameters. Twelve blocks are 85,054,464. The token table is 12,288,000, the position table is 393,216, the segment table is 1,536, the final LayerNorm is 1,536, and the LM head with its bias is 12,304,000. The sum is 110,042,752 parameters, 440,171,008 bytes, 440.17 MB. Each of the 12 heads has dimension 64.

See also:

- [preset-bert.md](preset-bert.md) — usage, benefits, and applications
- [train_chat](../examples/train_chat.md) — train `tiny` and `bert_tiny`, then one reply and one masked token
- [README.md](../README.md) — preset table and parameter counts
- [USAGE.md](../USAGE.md) — full programmer notes
- [gpt2-base-architecture.md](gpt2-base-architecture.md) — the same dense block as a causal decoder with context 1024
- `include/llm/presets.hpp` / `src/presets.cpp` — source of the config
