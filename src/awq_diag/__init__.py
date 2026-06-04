"""AWQ-Diag: a diagnostic toolkit for understanding low-bit LLM quantization failure."""
from __future__ import annotations

from .config import DiagConfig
from .pipeline import run_diagnostic

__version__ = "0.1.0"
__all__ = ["DiagConfig", "run_diagnostic"]
