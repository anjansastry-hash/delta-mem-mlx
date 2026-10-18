# -*- coding: utf-8 -*-
"""Milestone-2 loader test: the authors' trained δ-mem sidecar (community MLX conversion)
on the real Qwen3-4B-Instruct-2507 4-bit backbone.

  python tests/test_release_adapter.py
  python tests/test_release_adapter.py --base <dir> --adapter <dir>

Tests
  1. KEYS     every sidecar parameter the model expects is in the .npz (zero missing).
  2. KEYS     every .npz tensor lands on a model parameter (zero unexpected, including
              tensors the release regex would silently drop).
  3. SHAPES   every matched tensor has the model's shape.
  4. LOADED   the model's sidecar values are bit-identical to the .npz values.
  5. FORWARD  a prefill with the sidecar live completes with finite logits.
  6. IDENTITY gate scale 0 → logits bit-for-bit equal to the vanilla model (real weights).
Exit 0 = GREEN, 1 = RED, 77 = SKIP (weight folders not on disk).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from delta_mem_mlx import (load_release_adapter, release_weights, sidecar_parameters,  # noqa: E402
                           count_parameters, set_gate_scale, reset)
from delta_mem_mlx.sidecar import (_RELEASE_RE, _RELEASE_TO_OURS, _layer_prefix,  # noqa: E402
                                   _layers, _attn, _linear_dims)

GREEN, RED = "\033[32mGREEN\033[0m", "\033[31mRED\033[0m"
DEFAULT_BASE = os.path.expanduser("~/mlx-models/Qwen3-4B-Instruct-2507-4bit")
DEFAULT_ADAPTER = os.path.expanduser("~/mlx-models/delta-mem-qwen3-4b-adapter")


def check(name, ok, detail=""):
    print(f"  {GREEN if ok else RED}  {name}  {detail}")
    return ok


def maxdiff(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


def expected_shapes(model, r):
    """Sidecar parameter name → shape, derived from the backbone's q_proj dims and r, NOT from
    the loaded values (the loader is strict=False, so a wrong shape would already be in the tree)."""
    prefix = _layer_prefix(model)
    out = {}
    for i, layer in enumerate(_layers(model)):
        attn = _attn(layer)
        if not hasattr(attn, "delta_mem"):
            continue
        d_model, q_dim = _linear_dims(attn.q_proj.base)
        for leaf, shape in (("wq_m.weight", (r, d_model)), ("wk_m.weight", (r, d_model)),
                            ("wv_m.weight", (r, d_model)), ("w_beta.weight", (r, d_model)),
                            ("w_beta.bias", (r,)), ("a_q.weight", (q_dim, r)), ("a_o.weight", (d_model, r))):
            out[f"{prefix}.{i}.self_attn.delta_mem.{leaf}"] = shape
    return out


def key_report(model, npz_path, r):
    """(missing, unexpected, shape_mismatch, landed, n_expected, n_provided) between what the
    hooked model needs and what the .npz carries, computed independently of load_weights."""
    raw = mx.load(npz_path)
    unmatched = sorted(k for k in raw if not _RELEASE_RE.search(k))
    prefix = _layer_prefix(model)
    provided = {}
    for i, g in release_weights(npz_path).items():
        for rel, v in g.items():
            ours = _RELEASE_TO_OURS.get(rel)
            key = f"{prefix}.{i}.self_attn.delta_mem.{ours}" if ours else f"<unmapped> {prefix}.{i}.self_attn.{rel}"
            provided[key] = v
    expected = expected_shapes(model, r)
    missing = sorted(set(expected) - set(provided))
    unexpected = sorted(set(provided) - set(expected)) + unmatched
    both = sorted(set(expected) & set(provided))
    mismatch = [(k, expected[k], tuple(provided[k].shape)) for k in both if expected[k] != tuple(provided[k].shape)]
    in_model = sidecar_parameters(model)
    landed = all(k in in_model and bool(mx.array_equal(in_model[k], provided[k].astype(mx.float32))) for k in both)
    return missing, unexpected, mismatch, landed, len(expected), len(provided)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--prompt", default="Alarm at 03:12 on site 6021: handover failures toward PCI 71, neighbour PCI 101.")
    args = ap.parse_args()

    npz = os.path.join(args.adapter, "delta_mem_adapter_mlx.npz")
    cfg_json = os.path.join(args.adapter, "delta_mem_config.json")
    if not (os.path.isdir(args.base) and os.path.isfile(npz) and os.path.isfile(cfg_json)):
        print(f"SKIP: need base model at {args.base} and adapter (.npz + config) at {args.adapter}")
        sys.exit(77)

    from mlx_lm import load
    t0 = time.time()
    model, tok = load(args.base)
    ids = mx.array([tok.encode(args.prompt)])
    print(f"\n== release adapter on {os.path.basename(args.base)}  (prompt {ids.shape[1]} tokens, load {time.time()-t0:.1f}s) ==")
    vanilla = model(ids); mx.eval(vanilla)

    try:
        cfg = load_release_adapter(model, args.adapter)
    except Exception as e:  # the loader refuses the .npz outright (e.g. a missing release tensor)
        check("LOAD     load_release_adapter", False, f"{type(e).__name__}: {e}")
        print(f"\nRESULT: {RED}")
        sys.exit(1)
    n_layers, n_params = len(model._delta_mem_boxes), count_parameters(model)
    print(f"  attached: {n_layers} layers, {n_params:,} sidecar params, r={cfg.r}, alpha={cfg.alpha}")

    ok = True
    missing, unexpected, mismatch, landed, n_exp, n_prov = key_report(model, npz, cfg.r)
    ok &= check("KEYS     zero missing   ", not missing, f"{len(missing)} missing of {n_exp} expected  {missing}")
    ok &= check("KEYS     zero unexpected", not unexpected, f"{len(unexpected)} unexpected of {n_prov} provided  {unexpected}")
    ok &= check("SHAPES   all equal      ", not mismatch, f"{len(mismatch)} mismatched  {mismatch}")
    ok &= check("LOADED   values == .npz ", landed)

    try:
        set_gate_scale(model, 1.0); reset(model)
        live = model(ids); mx.eval(live)
        finite = bool(mx.all(mx.isfinite(live)))
        ok &= check("FORWARD  sidecar live   ", finite and live.shape == vanilla.shape,
                    f"max|Δ vs vanilla| = {maxdiff(live, vanilla):.4g}, finite={finite}")

        set_gate_scale(model, 0.0); reset(model)
        ident = model(ids); mx.eval(ident)
        d = maxdiff(ident, vanilla)
        ok &= check("IDENTITY gate 0 == vanilla", d == 0.0, f"max|Δ| = {d:.4g}")
    except Exception as e:  # a silently mis-shaped tensor only surfaces here (loader is strict=False)
        ok &= check("FORWARD  sidecar live   ", False, f"{type(e).__name__}: {str(e)[:160]}")

    print(f"\nRESULT: {GREEN if ok else RED}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
