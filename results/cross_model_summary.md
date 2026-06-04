# Cross-model diagnostic summary

| Model | Params | Linear layers | Median 4→3 jump | Max jump | Layers >5x | κ-vs-jump ρ | Top-κ layer |
|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | 1.54B | 196 | 3.92x | 4.02x | 0 | -0.360 | L1.mlp.down_proj |
| Qwen2.5-0.5B | 0.49B | 168 | 3.85x | 4.06x | 0 | -0.262 | L16.self_attn.o_proj |
