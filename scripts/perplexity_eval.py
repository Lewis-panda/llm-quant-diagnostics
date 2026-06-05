#!/usr/bin/env python
"""Model-level test of search-free AWQ: WikiText-2 perplexity, real group-wise quantizer.

Quantizes the whole model (all block-internal Linears) with the deployment-style group-wise
asymmetric quantizer and compares strategies at one bit-width:

    fp16          unquantized reference
    rtn           plain round-to-nearest (no protection)
    const_awq     AWQ scaling with ONE global alpha for every layer (search-free)
    search_awq    AWQ scaling with the per-layer best alpha (the expensive search)

The question: does `const_awq` match `search_awq` on perplexity? If yes, AWQ's per-layer scale
search is unnecessary; if not, per-layer alpha matters at the model level.

    python scripts/perplexity_eval.py --model Qwen/Qwen2.5-1.5B --bit 3 --max-windows 80
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from awq_diag.config import DiagConfig  # noqa: E402
from awq_diag.data import get_calibration_texts  # noqa: E402
from awq_diag.hooks import ActivationCollector, AWQErrorCollector  # noqa: E402
from awq_diag.model_utils import (  # noqa: E402
    iter_block_linears, layer_idx_from_name, load_model_and_tokenizer, module_type_from_name, set_seed)
from awq_diag.quant import awq_channel_scales, awq_dequant_weight, pseudo_quantize_groupwise  # noqa: E402

# official AWQ shares ONE scale per group of linears that read the same input
_GROUP = {"q_proj": "attn_in", "k_proj": "attn_in", "v_proj": "attn_in", "o_proj": "attn_out",
          "gate_proj": "mlp_in", "up_proj": "mlp_in", "down_proj": "mlp_out"}


def block_group_alpha(acc: dict, bit: int, alphas) -> dict:
    """Faithful (block-level) AWQ alpha: one shared alpha per (layer, group), where q/k/v share an
    input and gate/up share an input. Argmin of the group's summed output error — the official
    grouping, but using each group's combined linear-output error (robust, no standalone attention).
    """
    groups = defaultdict(list)
    for name in acc:
        groups[(layer_idx_from_name(name), _GROUP.get(module_type_from_name(name), name))].append(name)
    out = {}
    for members in groups.values():
        errs = {a: sum(acc[m][bit][a][0] for m in members) / max(sum(acc[m][bit][a][1] for m in members), 1e-12)
                for a in alphas}
        a_star = min(errs, key=errs.get)
        for m in members:
            out[m] = a_star
    return out


@torch.no_grad()
def eval_ppl(model, enc, device, seq_len=2048, max_windows=None) -> float:
    n = enc.numel()
    starts = list(range(0, n - seq_len, seq_len))
    if max_windows:
        starts = starts[:max_windows]
    nll, ntok = 0.0, 0
    for s in starts:
        ids = enc[s:s + seq_len].unsqueeze(0).to(device)
        loss = model(ids, labels=ids).loss.float().item()
        nll += loss * (seq_len - 1)
        ntok += seq_len - 1
    return float(np.exp(nll / ntok))


@torch.no_grad()
def quantize_model(model, act_scales, bit, gs, alpha_map=None):
    """alpha_map=None -> plain RTN; else AWQ-scale each linear by its alpha (per-name)."""
    quantizer = partial(pseudo_quantize_groupwise, group_size=gs, zero_point=True)
    for name, module in iter_block_linears(model):
        Wf = module.weight.data.float()
        if alpha_map is None:
            Wq = quantizer(Wf, bit)
        else:
            s = awq_channel_scales(act_scales[name].to(Wf.device), alpha_map[name])
            Wq = awq_dequant_weight(Wf, bit, s, quantizer=quantizer)
        module.weight.data.copy_(Wq.to(module.weight.dtype))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--bit", type=int, default=3)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--n-grid", type=int, default=21)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--max-windows", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = DiagConfig(model_name=args.model, bit_widths=(args.bit,))
    quantizer = partial(pseudo_quantize_groupwise, group_size=args.group_size, zero_point=True)

    print(f"[1/5] loading {args.model} ...")
    model, tok, device = load_model_and_tokenizer(cfg.model_name, cfg.dtype, cfg.device)

    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    print(f"      wikitext-2 test: {enc.numel()} tokens, {min(args.max_windows, enc.numel()//args.seq_len)} windows")

    print("[2/5] calibrating (activation scales) ...")
    coll = ActivationCollector(model, cfg)
    coll.register()
    coll.run_calibration(
        [tok(t, return_tensors="pt", truncation=True, max_length=cfg.max_calibration_length)
         for t in get_calibration_texts()], device)
    coll.remove()
    act_scales = {n: coll.stats[n]["channel_magnitude"] for n in coll.stats}

    print(f"[3/5] per-layer alpha search (group-wise, {args.n_grid}-pt grid) ...")
    alphas = [round(a, 4) for a in np.linspace(0, 1, args.n_grid)]
    awq = AWQErrorCollector(model, cfg, act_scales, alphas=alphas, quantizer=quantizer)
    awq.register()
    awq.run_calibration(
        [tok(t, return_tensors="pt", truncation=True, max_length=cfg.max_calibration_length)
         for t in get_calibration_texts()], device)
    awq.remove()
    fin = awq.finalize()
    best_alpha = {n: fin[n]["best_alpha"][args.bit] for n in fin}
    # global constant alpha = argmin of summed layer-output error
    curves = awq.error_curves()
    tot = {a: sum(curves[n][args.bit][a] for n in curves) for a in alphas}
    const_alpha = min(tot, key=tot.get)
    print(f"      global const alpha = {const_alpha};  per-layer best alpha median "
          f"{np.median(list(best_alpha.values())):.2f}")

    # snapshot original weights
    orig = {n: m.weight.detach().cpu().clone() for n, m in iter_block_linears(model)}

    def restore():
        for n, m in iter_block_linears(model):
            m.weight.data.copy_(orig[n].to(m.weight.device, m.weight.dtype))

    const_map = {n: const_alpha for n in best_alpha}
    block_alpha = block_group_alpha(awq.acc, args.bit, alphas)   # faithful block-level AWQ

    print("[4/5] evaluating perplexity per strategy ...")
    results = {}
    amaps = {"rtn": None, "const_awq": const_map, "block_awq": block_alpha, "search_awq": best_alpha}
    strategies = ["fp16", "rtn", "const_awq", "block_awq", "search_awq"]
    for strat in strategies:
        restore()
        if strat != "fp16":
            quantize_model(model, act_scales, args.bit, args.group_size, amaps[strat])
        results[strat] = eval_ppl(model, enc, device, args.seq_len, args.max_windows)
        print(f"      {strat:<12} ppl = {results[strat]:.3f}")
    restore()

    fp16, rtn = results["fp16"], results["rtn"]
    # how much of faithful block-AWQ's ppl recovery does const-alpha capture? (vs RTN baseline)
    denom = rtn - results["block_awq"]
    captures = (rtn - results["const_awq"]) / denom * 100 if denom > 0.1 else None
    summary = {
        "model": args.model, "bit": args.bit, "group_size": args.group_size,
        "const_alpha": const_alpha, "ppl": results,
        "awq_helps_over_rtn": bool(denom > 0.1),
        "const_captures_block_awq_pct": (round(float(captures), 1) if captures is not None else None),
    }
    out = Path(__file__).resolve().parents[1] / "results"
    out.mkdir(exist_ok=True)
    slug = args.model.split("/")[-1]
    (out / f"perplexity_{slug}_{args.bit}bit.json").write_text(json.dumps(summary, indent=2))

    print("\n[5/5] summary (WikiText-2 perplexity, lower = better)")
    print(f"  {'strategy':<12} {'ppl':>9} {'Δ vs fp16':>12}")
    for s in strategies:
        print(f"  {s:<12} {results[s]:>9.3f} {results[s]-fp16:>+12.3f}")
    if captures is not None:
        print(f"\n  const-α AWQ captures {captures:.0f}% of faithful block-AWQ's recovery over RTN "
              f"(const α={const_alpha}); const-α {'≤' if results['const_awq']<=results['block_awq'] else '>'} block-AWQ")
    else:
        print(f"\n  at {args.bit}-bit AWQ barely helps over RTN (rtn {rtn:.2f} vs "
              f"block-AWQ {results['block_awq']:.2f}); const α={const_alpha}")


if __name__ == "__main__":
    main()
