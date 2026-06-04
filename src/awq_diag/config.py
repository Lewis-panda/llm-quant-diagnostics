"""Configuration for an AWQ-Diag diagnostic run.

設定一次診斷實驗所需的所有參數 (model / bit-widths / calibration / 輸出路徑)。
Everything that controls *what* we measure lives here so a run is reproducible
from a single object.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

# Repo root = two levels up from this file (src/awq_diag/config.py -> repo root)
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DiagConfig:
    # --- model ---
    model_name: str = "Qwen/Qwen2.5-1.5B"
    dtype: str = "bfloat16"            # bfloat16 | float16 | float32
    device: str = "auto"              # auto | cuda | cpu

    # --- experiment ---
    bit_widths: Tuple[int, ...] = (8, 6, 4, 3, 2)
    max_calibration_length: int = 512
    outlier_sigma: float = 6.0        # |x| > k*sigma counts as an outlier
    jump_hi_bit: int = 4              # phase-transition window: hi_bit -> lo_bit
    jump_lo_bit: int = 3
    seed: int = 0

    # Layers whose raw activations we keep (for qualitative inspection only).
    save_raw_layers: Tuple[str, ...] = (
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.mlp.gate_proj",
    )

    # --- output ---
    results_dir: Path = field(default=REPO_ROOT / "results")
    figures_dir: Path = field(default=REPO_ROOT / "figures")

    def __post_init__(self) -> None:
        self.results_dir = Path(self.results_dir)
        self.figures_dir = Path(self.figures_dir)

    @property
    def model_slug(self) -> str:
        """`Qwen/Qwen2.5-1.5B` -> `Qwen2.5-1.5B` (safe for filenames)."""
        return self.model_name.split("/")[-1]

    @property
    def json_path(self) -> Path:
        return self.results_dir / f"diagnostic_{self.model_slug}.json"

    @property
    def model_figures_dir(self) -> Path:
        return self.figures_dir / self.model_slug
