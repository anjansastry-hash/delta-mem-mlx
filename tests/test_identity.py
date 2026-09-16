# -*- coding: utf-8 -*-
"""Milestone-1 tests for delta-mem-mlx.

Two ways to run:

  python tests/test_identity.py                      # tiny random Qwen2 + Qwen3, no download, ~10 s
  python tests/test_identity.py --model <path|repo>  # a real mlx-lm model (e.g. the Qwen2.5-7B already on disk)

Tests
  1. IDENTITY  β ≡ 0  → logits bit-for-bit equal to the vanilla model (prefill AND step-by-step decode).
  2. LIVE      β free → logits differ from vanilla, finite, and the state is non-zero.
  3. RESET     reset() then the same prompt → identical logits to the first live run.
  4. SHIFT     the same token after different histories → different logits (the state carries history).
  5. DETACH    detach() → vanilla again.
  6. DECODE    with a KV cache, token-by-token decode equals one-shot prefill (state threads through),
               judged against the vanilla model's own prefill-vs-decode kernel gap on the same device.
  7. RELEASE   save in the community .npz layout, reload through load_release_adapter → identical.
Exit code 0 = GREEN, 1 = RED. Every number printed comes from this run.
"""
from __future__ import annotations

import argparse
import sys
import os
import time

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from delta_mem_mlx import (DeltaMemConfig, attach, detach, reset, set_gate_scale,  # noqa: E402
                           get_states, count_parameters)

GREEN, RED = "\033[32mGREEN\033[0m", "\033[31mRED\033[0m"


# --- tiny random models (no download) ---------------------------------------------

def tiny_qwen2():
    from mlx_lm.models import qwen2
    args = qwen2.ModelArgs(model_type="qwen2", hidden_size=64, num_hidden_layers=3,
                           intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
                           rms_norm_eps=1e-6, vocab_size=257, rope_theta=10000.0, tie_word_embeddings=True)
    m = qwen2.Model(args)
    return _randomise(m, seed=1), "tiny-qwen2"


def tiny_qwen3():
    from mlx_lm.models import qwen3
    args = qwen3.ModelArgs(model_type="qwen3", hidden_size=64, num_hidden_layers=3,
                           intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
                           head_dim=16, rms_norm_eps=1e-6, vocab_size=257, rope_theta=10000.0,
                           max_position_embeddings=2048, tie_word_embeddings=True)
    m = qwen3.Model(args)
    return _randomise(m, seed=2), "tiny-qwen3"


def _randomise(m, seed):
    from mlx.utils import tree_flatten, tree_unflatten
    mx.random.seed(seed)
    new = [(k, mx.random.normal(v.shape) * 0.05) for k, v in tree_flatten(m.parameters())]
    m.update(tree_unflatten(new))
    m.set_dtype(mx.float32)
    mx.eval(m.parameters())
    return m


def real_model(path):
    from mlx_lm import load
    m, tok = load(path)
    return m, tok, path


# --- helpers ----------------------------------------------------------------------------

def logits(model, ids):
    out = model(ids)
    mx.eval(out)
    return out


def decode_logits(model, ids):
    """Token-by-token with a KV cache; returns the last-position logits after the full prompt."""
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    last = None
    for t in range(ids.shape[1]):
        last = model(ids[:, t:t + 1], cache=cache)
        mx.eval(last)
    return last[:, -1, :]


def maxdiff(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


def check(name, ok, detail=""):
    print(f"  {GREEN if ok else RED}  {name}  {detail}")
    return ok


# --- the suite ----------------------------------------------------------------------------

def run_suite(model, name, ids, ids_alt, tol_identity=0.0):
    print(f"\n== {name}  (prompt {ids.shape[1]} tokens) ==")
    ok = True
    vanilla = logits(model, ids)
    vanilla_dec = decode_logits(model, ids)

    cfg = DeltaMemConfig(r=8, alpha=16.0)
    t0 = time.time()
    attach(model, cfg)
    n = count_parameters(model)
    print(f"  attached: {len(model._delta_mem_boxes)} layers, {n:,} sidecar params, {time.time()-t0:.2f}s")

    # 1. identity
    set_gate_scale(model, 0.0); reset(model)
    ident = logits(model, ids)
    d = maxdiff(vanilla, ident)
    ok &= check("IDENTITY prefill  (β≡0 == vanilla)", d <= tol_identity, f"max|Δ| = {d:.3g}")
    reset(model)
    ident_dec = decode_logits(model, ids)
    d = maxdiff(vanilla_dec, ident_dec)
    ok &= check("IDENTITY decode   (β≡0 == vanilla, KV cache)", d <= tol_identity, f"max|Δ| = {d:.3g}")

    # 2. live
    set_gate_scale(model, 1.0); reset(model)
    live = logits(model, ids)
    d = maxdiff(vanilla, live)
    finite = bool(mx.all(mx.isfinite(live)))
    s_norm = max(float(mx.max(mx.abs(s))) for s in get_states(model) if s is not None)
    ok &= check("LIVE              (β free → differs, finite, S≠0)", d > 0 and finite and s_norm > 0,
                f"max|Δ| = {d:.3g}, max|S| = {s_norm:.3g}")

    # 3. reset reproducibility
    reset(model)
    live2 = logits(model, ids)
    d = maxdiff(live, live2)
    ok &= check("RESET             (same prompt after reset → identical)", d == 0.0, f"max|Δ| = {d:.3g}")

    # 4. history dependence: same last token, different history
    reset(model); a = logits(model, ids)[:, -1, :]
    reset(model); b = logits(model, ids_alt)[:, -1, :]
    # compare to the vanilla difference for the same two prompts (attention already makes them differ);
    # what we want is that the STATE contributes: run the same token alone with and without prior state.
    reset(model); _ = logits(model, ids[:, :-1])          # write history into S, discard output
    from mlx_lm.models.cache import make_prompt_cache
    # fresh KV (no attention to history) but S carries history:
    with_state = model(ids[:, -1:], cache=make_prompt_cache(model))[:, -1, :]
    reset(model)
    without_state = model(ids[:, -1:], cache=make_prompt_cache(model))[:, -1, :]
    mx.eval(with_state, without_state)
    d = maxdiff(with_state, without_state)
    ok &= check("SHIFT             (state alone changes the next token)", d > 0, f"max|Δ| = {d:.3g}")

    # 6. decode == prefill with the sidecar live. Prefill (L>1) and decode (L=1) use different
    # kernels on Metal, so even the vanilla model has a gap; the sidecar must not add to it
    # beyond fp32 noise. Reference gap = vanilla prefill vs vanilla decode, same prompt.
    vanilla_gap = maxdiff(vanilla[:, -1, :], vanilla_dec)
    reset(model); live_pre = logits(model, ids)[:, -1, :]
    reset(model); live_dec = decode_logits(model, ids)
    d = maxdiff(live_pre, live_dec)
    ok &= check("DECODE            (token-by-token == prefill, β free)", d <= max(1e-3, 2.0 * vanilla_gap) + tol_identity,
                f"max|Δ| = {d:.3g}   (vanilla's own prefill-vs-decode gap on this device: {vanilla_gap:.3g})")

    # 7. release-format round trip: save in the community .npz layout, reload, same logits
    import tempfile, json as _json
    from delta_mem_mlx import sidecar_parameters, load_release_adapter
    reset(model); ref = logits(model, ids)
    inv = {"wq_m.weight": "memory_q_proj", "wk_m.weight": "memory_k_proj", "wv_m.weight": "memory_v_proj",
           "a_q.weight": "delta_q_proj", "a_o.weight": "delta_o_proj", "w_beta.weight": "beta_proj",
           "w_beta.bias": "beta_bias"}
    arrays = {}
    for k, v in sidecar_parameters(model).items():
        i = k.split("layers.")[1].split(".")[0]; leaf = k.split(".delta_mem.")[1]
        arrays[f"layers.{i}.self_attn.{inv[leaf]}"] = v
    with tempfile.TemporaryDirectory() as d:
        mx.savez(os.path.join(d, "delta_mem_adapter_mlx.npz"), **arrays)
        with open(os.path.join(d, "delta_mem_config.json"), "w") as f:
            _json.dump({"rank": cfg.r, "alpha": cfg.alpha, "delta_heads": ["q", "o"], "normalize_qk": True,
                        "couple_lambda": True, "state_update_mode": "standard"}, f)
        detach(model)
        load_release_adapter(model, d)
    reset(model); rt = logits(model, ids)
    d = maxdiff(ref, rt)
    ok &= check("RELEASE FORMAT    (.npz round trip → identical)", d == 0.0, f"max|Δ| = {d:.3g}")

    # 5. detach
    detach(model)
    back = logits(model, ids)
    d = maxdiff(vanilla, back)
    ok &= check("DETACH            (vanilla again)", d == 0.0, f"max|Δ| = {d:.3g}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="mlx-lm model path or HF repo; default = tiny random models")
    ap.add_argument("--prompt", default="Alarm at 03:12 on site 6021: handover failures toward PCI 71, neighbour PCI 101.")
    args = ap.parse_args()

    all_ok = True
    if args.model is None:
        for build in (tiny_qwen2, tiny_qwen3):
            model, name = build()
            mx.random.seed(7)
            ids = mx.random.randint(0, 257, (1, 24))
            ids_alt = mx.concatenate([mx.random.randint(0, 257, (1, 23)), ids[:, -1:]], axis=1)
            all_ok &= run_suite(model, name, ids, ids_alt)
    else:
        model, tok, name = real_model(args.model)
        ids = mx.array([tok.encode(args.prompt)])
        alt = tok.encode("Site 2210 reports a sleeping cell after the 02:00 software reset, no traffic since.")
        ids_alt = mx.array([alt[:-1] + [ids[0, -1].item()]])
        # quantized / bf16 real models: identity must still be exact (Δ is exactly zero when β≡0)
        all_ok &= run_suite(model, name, ids, ids_alt, tol_identity=0.0)

    print(f"\nRESULT: {GREEN if all_ok else RED}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
