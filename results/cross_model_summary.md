# Cross-model AWQ-benefit summary

How much AWQ's activation-aware scaling reduces 3-bit output error, by family.

| Model | Params | Linear layers | Top-κ layer | AWQ reduction `down_proj` | `o_proj` | others | max |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 0.49B | 168 | L16.self_attn.o_proj | 2.88x | 1.77x | 1.20x | 28.9x |
| Qwen2.5-1.5B | 1.54B | 196 | L1.mlp.down_proj | 2.31x | 1.63x | 1.19x | 25.9x |
