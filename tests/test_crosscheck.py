# -*- coding: utf-8 -*-
"""Cross-port agreement test: this repo's δ-mem sidecar vs elimaine/delta-mem-mlx-sidecar on the
same Qwen3-4B backbone, the same released adapter and the same chat-templated token ids.

  python tests/test_crosscheck.py
  python tests/test_crosscheck.py --base <dir> --adapter <dir> --ref <clone>

Runs scripts/crosscheck.py in-process (both ports loaded, prefill only, no sampling), writes
results/crosscheck.json and asserts, for each of the three prompts,
  1. max|Δ| of the final-position logits ≤ 0.05,
  2. the top-1 token is the same.
Per-layer memory-state cosine is printed, not asserted. A max|Δ| above 0.05 but ≤ 0.5 is still
RED and gets a WARN line: that band is what bf16 rounding noise looks like (one bf16 ulp of a
logit near 32 is 0.25), as opposed to a maths divergence, which is what the 0.05 contract is for.
Exit 0 = GREEN, 1 = RED, 77 = SKIP (weights or the reference clone are not on disk).
Nothing here changes either port. A disagreement is a finding: results/crosscheck.json records
the first layer whose hidden state differs and whether it enters via the q hook, the o hook, the
recurrence or the attention body.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import crosscheck  # noqa: E402

GREEN, RED, WARN = "\033[32mGREEN\033[0m", "\033[31mRED\033[0m", "\033[33mWARN\033[0m"
PASS_TOL, WARN_TOL = 0.05, 0.5


def check(name, ok, detail=""):
    print(f"  {GREEN if ok else RED}  {name}  {detail}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=crosscheck.DEFAULT_BASE)
    ap.add_argument("--adapter", default=crosscheck.DEFAULT_ADAPTER)
    ap.add_argument("--ref", default=crosscheck.DEFAULT_REF, help=f"clone of {crosscheck.REF_URL}")
    ap.add_argument("--out", default=crosscheck.DEFAULT_OUT)
    args = ap.parse_args()

    missing = crosscheck.missing_inputs(args.base, args.adapter, args.ref)
    if missing:
        print("SKIP: not on disk:\n  " + "\n  ".join(missing))
        sys.exit(77)

    try:
        doc = crosscheck.run(args.base, args.adapter, args.ref, args.out, quiet=True)
    except Exception as e:
        traceback.print_exc()
        print(f"\nRESULT: {RED}  (the cross-check itself failed: {type(e).__name__}: {str(e)[:200]})")
        sys.exit(1)

    this, ref, ind = doc["this_repo"], doc["reference"], doc["independence"]
    print(f"\n== cross-check on {os.path.basename(args.base)} + {os.path.basename(args.adapter)} ==")
    print(f"  this repo {this['commit']}{' (dirty)' if this['dirty'] else ''}   reference {ref['commit']}   "
          f"adapter sha256 {doc['adapter']['npz_sha256']}")
    ok = True
    ok &= check("SETUP    two code paths, one adapter",
                ind["distinct_model_objects"] and ind["same_layer_set"] and ind["adapter_tensors_bit_identical_in_both_ports"],
                f"{len(ind['port_a']['layers'])} layers; A={ind['port_a']['q_proj_class']}, "
                f"B={ind['port_b']['attention_class'].split(' ')[0]}")
    for p in doc["prompts"]:
        f, bo = p["final_logits"], p["backbone_only"]
        d = f["max_abs_delta"]
        ok &= check(f"LOGITS   {p['name']:<12s} max|Δ| ≤ {PASS_TOL}", d <= PASS_TOL,
                    f"max|Δ| = {d:.4g}   (backbone only, sidecar silenced: {bo['max_abs_delta']:.4g}; "
                    f"sidecar effect: {bo['sidecar_effect_max_abs_delta_a']:.3g})")
        ok &= check(f"TOP-1    {p['name']:<12s} equal", f["top1_match"],
                    f"A={f['top1_text_a']!r} B={f['top1_text_b']!r}")
        if PASS_TOL < d <= WARN_TOL:
            print(f"  {WARN}   {p['name']}: max|Δ| = {d:.4g} is above {PASS_TOL} but within {WARN_TOL} — the "
                  f"{f['dtype'].split('.')[-1]} rounding-noise band, not a maths divergence; still RED under the 0.05 contract")
        cos = [r["state_cosine"] for r in p["layers"] if r["state_cosine"] is not None]
        if cos:
            ds = max(r["state_max_abs_delta"] for r in p["layers"] if r["state_max_abs_delta"] is not None)
            print(f"           memory state: {len(cos)} layers, cosine min {min(cos):.9f}, max|ΔS| {ds:.3g}")
        else:
            print("           memory state: not exposed")
        if d > PASS_TOL or not f["top1_match"]:
            print(f"           first divergence: {p['divergence']['attribution']}")

    print(f"\nRESULT: {GREEN if ok else RED}   (details: {args.out})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
