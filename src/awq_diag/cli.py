"""Console entry point: `awq-diag --model ...` (installed via pyproject scripts)."""
from __future__ import annotations

import argparse

from .config import DiagConfig
from .pipeline import run_diagnostic


def main() -> None:
    p = argparse.ArgumentParser(prog="awq-diag", description="Run AWQ-Diag on one causal LM.")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--bits", type=int, nargs="+", default=[8, 6, 4, 3, 2])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args()

    cfg = DiagConfig(
        model_name=args.model, dtype=args.dtype, device=args.device,
        bit_widths=tuple(args.bits), seed=args.seed,
    )
    run_diagnostic(cfg, make_figures=not args.no_figures)


if __name__ == "__main__":
    main()
