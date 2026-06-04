#!/usr/bin/env python
"""Run the AWQ-Diag diagnostic for one model.

Examples
--------
    python scripts/run_diagnostic.py                       # Qwen2.5-1.5B (default)
    python scripts/run_diagnostic.py --model Qwen/Qwen2.5-0.5B
    python scripts/run_diagnostic.py --model Qwen/Qwen2.5-1.5B --no-figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow running straight from the repo without installing
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from awq_diag import DiagConfig, run_diagnostic  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Run AWQ-Diag on one causal LM.")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B", help="HuggingFace model id")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--bits", type=int, nargs="+", default=[8, 6, 4, 3, 2])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-figures", action="store_true", help="skip PNG rendering")
    args = p.parse_args()

    cfg = DiagConfig(
        model_name=args.model,
        dtype=args.dtype,
        device=args.device,
        bit_widths=tuple(args.bits),
        seed=args.seed,
    )
    run_diagnostic(cfg, make_figures=not args.no_figures)


if __name__ == "__main__":
    main()
