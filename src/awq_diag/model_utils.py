"""Model loading and Transformer-layer bookkeeping helpers."""
from __future__ import annotations

import random
import re
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

# `model.layers.<idx>.<...>.<module_type>`
_LAYER_RE = re.compile(r"layers\.(\d+)\.")
# the trailing module name, e.g. q_proj / k_proj / gate_proj / down_proj
_KNOWN_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_model_and_tokenizer(
    model_name: str,
    dtype: str = "bfloat16",
    device: str = "auto",
) -> Tuple[nn.Module, "AutoTokenizer", torch.device]:
    """Load a causal LM + tokenizer in eval mode."""
    dev = resolve_device(device)
    torch_dtype = _DTYPES[dtype] if dev.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch_dtype,
        device_map=str(dev) if dev.type == "cuda" else None,
        attn_implementation="eager",
    )
    if dev.type != "cuda":
        model = model.to(dev)
    model.eval()
    return model, tokenizer, dev


def layer_idx_from_name(name: str) -> Optional[int]:
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def module_type_from_name(name: str) -> str:
    """Return the projection family (q_proj, down_proj, ...) or 'other'."""
    leaf = name.split(".")[-1]
    return leaf if leaf in _KNOWN_MODULES else "other"


def iter_block_linears(model: nn.Module) -> Iterator[Tuple[str, nn.Linear]]:
    """Yield (name, module) for every nn.Linear inside a Transformer block.

    過濾掉 embedding / lm_head，只保留 `model.layers.*` 內的 Linear。
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "layers." in name:
            yield name, module


def get_model_info(model: nn.Module) -> Dict[str, int]:
    cfg = model.config
    return {
        "num_params": int(sum(p.numel() for p in model.parameters())),
        "num_layers": int(getattr(cfg, "num_hidden_layers", len(model.model.layers))),
        "hidden_size": int(cfg.hidden_size),
        "num_attention_heads": int(cfg.num_attention_heads),
        "num_key_value_heads": int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)),
        "intermediate_size": int(cfg.intermediate_size),
    }
