# -*- coding: utf-8 -*-
"""Cross-check of two independent MLX ports of δ-mem: same backbone, same adapter, same tokens.

  python scripts/crosscheck.py
  python scripts/crosscheck.py --base <dir> --adapter <dir> --ref <clone> --out results/crosscheck.json

Port A is this repo: delta_mem_mlx.load_release_adapter() (attach + strict load; hooks on
q_proj / o_proj). Port B is elimaine/delta-mem-mlx-sidecar, cloned at --ref: its server loads
the backbone with mlx_lm.load and attaches the adapter through MlxBackboneRuntime._load()
(delta-mem-sidecar/src/delta_mem_sidecar/mlx_runtime.py), which replaces every layer's
self_attn with its MlxDeltaAttention wrapper; benchmarks/full_delta_mem_bench.py drives that
server over HTTP. Here the same loader is called in-process — no server, no HTTP — exactly as
the reference's own benchmarks/mlx_delta_diagnostics.py does.

Per prompt the SAME chat-templated token ids go through both models (prefill only: one
forward, no sampling, no KV cache) and the script records
  * max|Δ| of the final-position logits and whether the top-1 token matches,
  * per-layer cosine similarity of the 8×8 memory state (both ports expose it), plus the
    per-layer readout, Δq, Δo and post-layer hidden state, so that a difference can be placed
    (which layer; q hook, o hook, recurrence, or the attention body),
  * a backbone-only pass (sidecar silenced in both ports: gate 0 here, write disabled there)
    that isolates the reference's re-implemented attention body from the δ-mem maths.
Everything is written to results/crosscheck.json with the prompts, ids, adapter sha256 and
both commit hashes. This script measures; tests/test_crosscheck.py judges.
Exit 0 = ran, 77 = SKIP (weights or reference clone absent), 2 = harness error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from typing import Dict, List, Optional

import mlx.core as mx
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_BASE = os.path.expanduser("~/mlx-models/Qwen3-4B-Instruct-2507-4bit")
DEFAULT_ADAPTER = os.path.expanduser("~/mlx-models/delta-mem-qwen3-4b-adapter")
DEFAULT_REF = os.path.expanduser("~/ror/_ref/delta-mem-mlx-sidecar")
DEFAULT_OUT = os.path.join(ROOT, "results", "crosscheck.json")
REF_URL = "https://github.com/elimaine/delta-mem-mlx-sidecar"
REF_SRC = os.path.join("delta-mem-sidecar", "src")

# The three prompts. Each is one user turn, rendered with the model's chat template.
PROMPTS = [
    {"name": "fact", "kind": "one-line fact",
     "text": "The capital of France is Paris."},
    {"name": "paragraph", "kind": "40-token paragraph",
     "text": "Site 6021 raised handover failures toward PCI 71 at 03:12. The on-call engineer "
             "restarted the baseband unit and traffic recovered fully within ten minutes."},
    {"name": "instruction", "kind": "short instruction",
     "text": "Give three synonyms for the word quick."},
]

# release tensor name → this repo's parameter leaf (for the "same weights landed" check)
RELEASE_TO_OURS = {"memory_q_proj": "wq_m.weight", "memory_k_proj": "wk_m.weight",
                   "memory_v_proj": "wv_m.weight", "delta_q_proj": "a_q.weight",
                   "delta_o_proj": "a_o.weight", "beta_proj": "w_beta.weight", "beta_bias": "w_beta.bias"}


# --- preconditions / provenance ----------------------------------------------------------

def missing_inputs(base: str, adapter: str, ref: str) -> List[str]:
    """What is absent on disk (empty list = everything needed is there)."""
    out = []
    if not os.path.isfile(os.path.join(base, "config.json")):
        out.append(f"base model (config.json) at {base}")
    for f in ("delta_mem_adapter_mlx.npz", "delta_mem_config.json"):
        if not os.path.isfile(os.path.join(adapter, f)):
            out.append(f"adapter file {os.path.join(adapter, f)}")
    if not os.path.isfile(os.path.join(ref, REF_SRC, "delta_mem_sidecar", "mlx_runtime.py")):
        out.append(f"reference clone at {ref}  (git clone {REF_URL} {ref})")
    return out


def git_info(path: str) -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", "-C", path, *args], capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception:
            return None
    status = run("status", "--porcelain")
    return {"path": path, "commit": run("rev-parse", "HEAD"),
            "dirty": None if status is None else bool(status),
            "head": run("log", "-1", "--format=%h %s")}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- numerics ---------------------------------------------------------------------------

def f32(a: mx.array) -> mx.array:
    return a.astype(mx.float32)


def maxabs(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(f32(a) - f32(b))))


def cosine(a: Optional[mx.array], b: Optional[mx.array]) -> Optional[float]:
    """Cosine similarity of two tensors flattened, in float64. None if either norm is zero."""
    if a is None or b is None:
        return None
    va = np.asarray(f32(a)).astype(np.float64).reshape(-1)
    vb = np.asarray(f32(b)).astype(np.float64).reshape(-1)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0.0 or nb == 0.0:
        return None
    return float(np.dot(va, vb) / (na * nb))


def last_pos(logits: mx.array) -> mx.array:
    return f32(logits[0, -1, :])


def topk(v: mx.array, k: int = 5) -> List[int]:
    return [int(i) for i in np.argsort(-np.asarray(v))[:k]]


# --- prompts ------------------------------------------------------------------------------

def render_and_encode(tok, text: str):
    """Chat template exactly as the reference's _render_prompt applies it (tokenize=False,
    add_generation_prompt=True), then the token rule mlx_lm.generate uses for a str prompt
    (mlx_lm 0.32 stream_generate: add special tokens unless the text already starts with BOS;
    Qwen3 has no BOS, and its tokenizer adds nothing either way)."""
    rendered = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                       add_generation_prompt=True)
    add_special = tok.bos_token is None or not rendered.startswith(tok.bos_token)
    ids = [int(i) for i in tok.encode(rendered, add_special_tokens=add_special)]
    n_content = len(tok.encode(text, add_special_tokens=False))
    return rendered, ids, n_content


# --- the two ports ------------------------------------------------------------------------

class PortA:
    """This repo: hooks on q_proj / o_proj, state in per-layer boxes."""
    name = "A: this repo — delta_mem_mlx (hooks on q_proj / o_proj)"

    def __init__(self, base: str, adapter: str):
        from mlx_lm import load
        import delta_mem_mlx
        from delta_mem_mlx.sidecar import _QHook, _layers, _attn
        self.lib = delta_mem_mlx
        t0 = time.time()
        self.model, self.tok = load(base)
        self.cfg = delta_mem_mlx.load_release_adapter(self.model, adapter)
        self.load_s = time.time() - t0
        self.attn = {i: _attn(layer) for i, layer in enumerate(_layers(self.model))
                     if isinstance(_attn(layer).q_proj, _QHook)}
        self.boxes = {i: a.q_proj._box for i, a in self.attn.items()}
        self.module_file = delta_mem_mlx.__file__

    def reset(self):
        self.lib.reset(self.model)

    def silence(self, on: bool):
        self.lib.set_gate_scale(self.model, 0.0 if on else 1.0)   # β ≡ 0 → S stays 0 → Δq = Δo = 0

    def states(self) -> Dict[int, Optional[mx.array]]:
        return {i: b.state for i, b in self.boxes.items()}

    def readouts(self) -> Dict[int, Optional[mx.array]]:
        return {i: b.readout for i, b in self.boxes.items()}

    def delta_q(self, i: int, dtype) -> Optional[mx.array]:
        b = self.boxes[i]
        return None if b.readout is None else b.sidecar.delta_q(b.readout, dtype)

    def delta_o(self, i: int, dtype) -> Optional[mx.array]:
        b = self.boxes[i]
        return None if b.readout is None else b.sidecar.delta_o(b.readout, dtype)

    def weight(self, i: int, release_name: str) -> mx.array:
        mod, leaf = RELEASE_TO_OURS[release_name].split(".")
        return getattr(getattr(self.attn[i].delta_mem, mod), leaf)

    def describe(self) -> dict:
        a0 = self.attn[min(self.attn)]
        return {"module": self.module_file,
                "attention_class": f"{type(a0).__module__}.{type(a0).__name__} (untouched mlx-lm attention)",
                "q_proj_class": f"{type(a0.q_proj).__module__}.{type(a0.q_proj).__name__}",
                "o_proj_class": f"{type(a0.o_proj).__module__}.{type(a0.o_proj).__name__}",
                "layers": sorted(self.attn), "r": self.cfg.r, "alpha": self.cfg.alpha,
                "scaling": self.cfg.scaling, "normalize_qk": self.cfg.normalize_qk, "load_s": round(self.load_s, 2)}


class PortB:
    """The reference port: layer.self_attn replaced by MlxDeltaAttention, loaded through the
    server's own runtime object."""
    name = "B: elimaine/delta-mem-mlx-sidecar — MlxBackboneRuntime._load() → MlxDeltaAttention"

    def __init__(self, base: str, adapter: str, ref: str):
        src = os.path.join(ref, REF_SRC)
        if src not in sys.path:
            sys.path.insert(0, src)
        import delta_mem_sidecar
        from delta_mem_sidecar.mlx_runtime import MlxBackboneRuntime
        from delta_mem_sidecar import mlx_delta_attention as mda
        self.mda = mda
        t0 = time.time()
        # the path the server takes: mlx_lm.load → load_mlx_delta_adapter → _validate_adapter_compatible
        # → wrap_qwen_attention_layers (mlx_runtime.py, MlxBackboneRuntime._load)
        self.runtime = MlxBackboneRuntime(model_path=base, model_id="crosscheck-reference", adapter_dir=adapter)
        self.model, self.tok = self.runtime._load()
        self.load_s = time.time() - t0
        if not self.runtime._delta_enabled:
            raise RuntimeError("reference runtime loaded the backbone but did not enable its δ-mem path")
        self.mods = dict(mda.iter_mlx_delta_attention_modules(self.model))
        self.module_file = delta_mem_sidecar.__file__

    def reset(self):
        self.mda.reset_mlx_delta_states(self.model)

    def silence(self, on: bool):
        for a in self.mods.values():     # write disabled + state reset to zeros → reads = 0 → Δq = Δo = 0
            a.set_write_enabled(not on)

    def states(self) -> Dict[int, Optional[mx.array]]:
        return {i: a.state_snapshot() for i, a in self.mods.items()}

    def readouts(self) -> Dict[int, Optional[mx.array]]:
        return {i: a.last_reads for i, a in self.mods.items()}

    def delta_q(self, i: int, dtype) -> Optional[mx.array]:
        a = self.mods[i]
        if a.last_reads is None:
            return None
        return a._project_delta(a.last_reads, a.weights.delta_q_proj, "q").astype(dtype)

    def delta_o(self, i: int, dtype) -> Optional[mx.array]:
        a = self.mods[i]
        if a.last_reads is None:
            return None
        return a._project_delta(a.last_reads, a.weights.delta_o_proj, "o").astype(dtype)

    def weight(self, i: int, release_name: str) -> mx.array:
        return getattr(self.mods[i].weights, release_name)

    def describe(self) -> dict:
        a0 = self.mods[min(self.mods)]
        c = a0.delta_config
        return {"module": self.module_file,
                "attention_class": f"{type(a0).__module__}.{type(a0).__name__} (re-implements the attention body)",
                "wrapped_base_class": f"{type(a0.base).__module__}.{type(a0.base).__name__}",
                "layers": sorted(self.mods), "r": c.rank, "alpha": c.alpha, "scaling": c.delta_scaling,
                "normalize_qk": c.normalize_qk, "couple_lambda": c.couple_lambda,
                "state_update_mode": c.state_update_mode, "load_s": round(self.load_s, 2)}


# --- forward passes -----------------------------------------------------------------------

def prefill(model, ids: mx.array) -> mx.array:
    out = model(ids)
    mx.eval(out)
    return out


def layerwise_hidden(model, ids: mx.array):
    """Qwen3Model.__call__ step by step (same ops, same order) so every layer's output is visible.
    Returns (hidden states after each layer, logits). The caller checks logits == model(ids)."""
    from mlx_lm.models.base import create_attention_mask
    inner = model.model
    h = inner.embed_tokens(ids)
    mask = create_attention_mask(h, None)
    hs = []
    for layer in inner.layers:
        h = layer(h, mask, None)
        hs.append(h)
    out = inner.norm(h)
    logits = inner.embed_tokens.as_linear(out) if model.args.tie_word_embeddings else model.lm_head(out)
    mx.eval(logits, *hs)
    return hs, logits


def _eval_all(*dicts):
    arrays = [v for d in dicts for v in d.values() if v is not None]
    if arrays:
        mx.eval(*arrays)


def compare_prompt(A: PortA, B: PortB, p: dict, tok_text) -> dict:
    ids = mx.array([p["ids"]])
    act_dtype = None

    # 1. live prefill through both ports, fresh state
    A.reset(); A.silence(False)
    t0 = time.time(); la = prefill(A.model, ids); ta = time.time() - t0
    sa, ra = A.states(), A.readouts(); _eval_all(sa, ra)
    B.reset(); B.silence(False)
    t0 = time.time(); lb = prefill(B.model, ids); tb = time.time() - t0
    sb, rb = B.states(), B.readouts(); _eval_all(sb, rb)
    act_dtype = la.dtype

    fa, fb = last_pos(la), last_pos(lb)
    top_a, top_b = int(mx.argmax(fa)), int(mx.argmax(fb))
    all_a, all_b = f32(la[0]), f32(lb[0])
    argmax_agree = float(np.mean(np.asarray(mx.argmax(all_a, axis=-1)) == np.asarray(mx.argmax(all_b, axis=-1))))
    final = {
        "dtype": str(la.dtype), "shape": list(la.shape),
        "max_abs_delta": maxabs(fa, fb), "mean_abs_delta": float(mx.mean(mx.abs(fa - fb))),
        "top1_a": top_a, "top1_b": top_b, "top1_match": top_a == top_b,
        "top1_text_a": tok_text(top_a), "top1_text_b": tok_text(top_b),
        "top5_a": topk(fa), "top5_b": topk(fb),
        "top1_logit_a": float(fa[top_a]), "top1_logit_b": float(fb[top_b]),
        "all_positions_max_abs_delta": maxabs(all_a, all_b),
        "all_positions_argmax_agreement": argmax_agree,
        "forward_s_a": round(ta, 3), "forward_s_b": round(tb, 3),
    }

    # 2. per-layer memory state, readout, Δq, Δo
    layers = []
    for i in sorted(set(sa) | set(sb)):
        s_a, s_b, r_a, r_b = sa.get(i), sb.get(i), ra.get(i), rb.get(i)
        row = {"layer": i,
               "state_exposed_a": s_a is not None, "state_exposed_b": s_b is not None,
               "state_cosine": cosine(s_a, s_b) if (s_a is not None and s_b is not None) else None,
               "state_max_abs_delta": maxabs(s_a, s_b) if (s_a is not None and s_b is not None) else None,
               "state_max_abs_a": float(mx.max(mx.abs(s_a))) if s_a is not None else None,
               "readout_max_abs_delta": maxabs(r_a, r_b) if (r_a is not None and r_b is not None) else None}
        if i in A.boxes and i in B.mods and r_a is not None and r_b is not None:
            row["delta_q_max_abs_delta_f32"] = maxabs(A.delta_q(i, mx.float32), B.delta_q(i, mx.float32))
            row["delta_o_max_abs_delta_f32"] = maxabs(A.delta_o(i, mx.float32), B.delta_o(i, mx.float32))
            row["delta_q_max_abs_delta_act"] = maxabs(A.delta_q(i, act_dtype), B.delta_q(i, act_dtype))
            row["delta_o_max_abs_delta_act"] = maxabs(A.delta_o(i, act_dtype), B.delta_o(i, act_dtype))
        layers.append(row)

    # 3. backbone only: sidecar silenced in both ports (A: gate 0; B: write disabled on a zero state)
    A.reset(); A.silence(True); va = prefill(A.model, ids); A.silence(False); A.reset()
    B.reset(); B.silence(True); vb = prefill(B.model, ids); B.silence(False); B.reset()
    fva, fvb = last_pos(va), last_pos(vb)
    backbone_only = {
        "max_abs_delta": maxabs(fva, fvb), "top1_match": int(mx.argmax(fva)) == int(mx.argmax(fvb)),
        "all_positions_max_abs_delta": maxabs(va[0], vb[0]),
        "sidecar_effect_max_abs_delta_a": maxabs(fa, fva),   # how much the live sidecar moves the logits
        "sidecar_effect_max_abs_delta_b": maxabs(fb, fvb),
        "vanilla_top1_a": int(mx.argmax(fva)), "vanilla_top1_b": int(mx.argmax(fvb)),
    }

    # 4. where does the live forward first differ? hidden state after every layer
    A.reset(); hs_a, lw_a = layerwise_hidden(A.model, ids)
    B.reset(); hs_b, lw_b = layerwise_hidden(B.model, ids)
    diag = {"layerwise_pass_matches_direct_a": maxabs(lw_a, la) == 0.0,
            "layerwise_pass_matches_direct_b": maxabs(lw_b, lb) == 0.0,
            "hidden_dtype": str(hs_a[0].dtype),
            "hidden_max_abs_delta_per_layer": [maxabs(x, y) for x, y in zip(hs_a, hs_b)]}
    first = next((i for i, d in enumerate(diag["hidden_max_abs_delta_per_layer"]) if d > 0.0), None)
    diag["first_divergent_layer"] = first
    if first is None:
        diag["attribution"] = "none: every layer's hidden state is bit-identical between the ports"
    else:
        row = next((r for r in layers if r["layer"] == first), {})
        dq, do = row.get("delta_q_max_abs_delta_act"), row.get("delta_o_max_abs_delta_act")
        ro = row.get("readout_max_abs_delta")
        ro_s = "n/a" if ro is None else f"{ro:.3g}"
        if dq is not None and dq > 0.0:
            where = f"q hook: Δq differs after the cast to {act_dtype} (readout max|Δ| in fp32 = {ro_s})"
        elif do is not None and do > 0.0:
            where = f"o hook: Δo differs after the cast to {act_dtype} (readout max|Δ| in fp32 = {ro_s})"
        elif ro is not None and ro > 0.0:
            where = (f"recurrence: readout differs in fp32 by {ro_s} but Δq/Δo are identical after the cast — "
                     f"the hidden difference comes from the attention body, not the sidecar")
        else:
            where = ("attention body: sidecar readout, Δq and Δo are bit-identical at this layer; the reference "
                     "re-implements the attention forward (see backbone_only)")
        diag["attribution"] = f"layer {first}: {where}"
    for r in layers:
        r["hidden_max_abs_delta_after_layer"] = diag["hidden_max_abs_delta_per_layer"][r["layer"]] \
            if r["layer"] < len(diag["hidden_max_abs_delta_per_layer"]) else None
    A.reset(); B.reset()
    mx.clear_cache()
    return {**p, "final_logits": final, "layers": layers, "backbone_only": backbone_only, "divergence": diag}


# --- main ---------------------------------------------------------------------------------

def run(base: str = DEFAULT_BASE, adapter: str = DEFAULT_ADAPTER, ref: str = DEFAULT_REF,
        out: str = DEFAULT_OUT, prompts: Optional[List[dict]] = None, quiet: bool = False) -> dict:
    """Load both ports, run the prompts, write the JSON, return the results dict."""
    import mlx_lm
    say = (lambda *a, **k: None) if quiet else print
    prompts = [dict(p) for p in (prompts or PROMPTS)]
    npz, cfg_json = os.path.join(adapter, "delta_mem_adapter_mlx.npz"), os.path.join(adapter, "delta_mem_config.json")

    say(f"== δ-mem cross-check: this repo vs {REF_URL} ==")
    say(f"  base    {base}\n  adapter {adapter}\n  ref     {ref}")
    A = PortA(base, adapter)
    say(f"  port A loaded: {len(A.attn)} layers hooked, {A.load_s:.1f}s")
    B = PortB(base, adapter, ref)
    say(f"  port B loaded: {len(B.mods)} layers wrapped, {B.load_s:.1f}s")

    # the two code paths are distinct objects and the same adapter tensors landed in both
    same_layers = sorted(A.attn) == sorted(B.mods)
    weights_identical = same_layers and all(
        bool(mx.array_equal(A.weight(i, n), B.weight(i, n))) for i in A.attn for n in RELEASE_TO_OURS)
    independence = {
        "port_a": A.describe(), "port_b": B.describe(),
        "distinct_model_objects": A.model is not B.model,
        "distinct_backbone_weights": A.attn[min(A.attn)].q_proj.base is not B.mods[min(B.mods)].q_proj,
        "same_layer_set": same_layers,
        "adapter_tensors_bit_identical_in_both_ports": weights_identical,
        "reference_server_system_preamble_omitted": True,   # app.py prepends a system message; the loader does not
    }
    say(f"  same layer set: {same_layers}; adapter tensors identical in both ports: {weights_identical}")

    # prompts: rendered once per tokenizer, must agree
    for p in prompts:
        ra, ids_a, n_a = render_and_encode(A.tok, p["text"])
        rb, ids_b, n_b = render_and_encode(B.tok, p["text"])
        if ra != rb or ids_a != ids_b:
            raise RuntimeError(f"prompt {p['name']}: the two tokenizers rendered/encoded differently")
        p.update({"rendered": ra, "ids": ids_a, "n_tokens": len(ids_a), "n_content_tokens": n_a})

    tok_text = lambda i: A.tok.decode([i])  # noqa: E731
    results = []
    for p in prompts:
        say(f"\n-- {p['name']} ({p['kind']}, {p['n_content_tokens']} content tokens, {p['n_tokens']} with template) --")
        r = compare_prompt(A, B, p, tok_text)
        f, bo, dg = r["final_logits"], r["backbone_only"], r["divergence"]
        say(f"  final-position logits: max|Δ| = {f['max_abs_delta']:.4g}   mean|Δ| = {f['mean_abs_delta']:.3g}   "
            f"top-1 {'MATCH' if f['top1_match'] else 'DIFFER'}  A={f['top1_text_a']!r} B={f['top1_text_b']!r}")
        say(f"  all positions: max|Δ| = {f['all_positions_max_abs_delta']:.4g}, argmax agreement {f['all_positions_argmax_agreement']:.3f}")
        say(f"  backbone only (sidecar silenced in both): max|Δ| = {bo['max_abs_delta']:.4g}, top-1 "
            f"{'match' if bo['top1_match'] else 'differ'};  sidecar effect on logits: A {bo['sidecar_effect_max_abs_delta_a']:.3g}, B {bo['sidecar_effect_max_abs_delta_b']:.3g}")
        say(f"  layer  cos(S)        max|ΔS|     max|Δread|  max|Δq|{str(f['dtype']).split('.')[-1]:<9s} max|Δo|{str(f['dtype']).split('.')[-1]:<9s} max|Δh| after layer")
        for row in r["layers"]:
            cs = "not exposed" if row["state_cosine"] is None and not (row["state_exposed_a"] and row["state_exposed_b"]) \
                else ("undefined" if row["state_cosine"] is None else f"{row['state_cosine']:.9f}")
            g = lambda k: ("   -   " if row.get(k) is None else f"{row[k]:.3g}")  # noqa: E731
            say(f"  {row['layer']:>5d}  {cs:<13s} {g('state_max_abs_delta'):<11s} {g('readout_max_abs_delta'):<11s} "
                f"{g('delta_q_max_abs_delta_act'):<16s} {g('delta_o_max_abs_delta_act'):<16s} {g('hidden_max_abs_delta_after_layer')}")
        say(f"  first divergence: {dg['attribution']}")
        if not (dg["layerwise_pass_matches_direct_a"] and dg["layerwise_pass_matches_direct_b"]):
            say("  (note: the layer-by-layer pass did not reproduce the direct forward bit-for-bit; treat the per-layer hidden deltas as approximate)")
        results.append(r)

    cosines = [row["state_cosine"] for r in results for row in r["layers"] if row["state_cosine"] is not None]
    summary = {
        "max_abs_delta_final_logits": max(r["final_logits"]["max_abs_delta"] for r in results),
        "all_top1_match": all(r["final_logits"]["top1_match"] for r in results),
        "min_state_cosine": min(cosines) if cosines else None,
        "max_state_abs_delta": max((row["state_max_abs_delta"] for r in results for row in r["layers"]
                                    if row["state_max_abs_delta"] is not None), default=None),
        "backbone_only_max_abs_delta": max(r["backbone_only"]["max_abs_delta"] for r in results),
        "first_divergent_layer": {r["name"]: r["divergence"]["first_divergent_layer"] for r in results},
        "attribution": {r["name"]: r["divergence"]["attribution"] for r in results},
    }
    doc = {
        "kind": "delta-mem-mlx crosscheck: this repo vs elimaine/delta-mem-mlx-sidecar",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "machine": {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
                    "mlx": mx.__version__, "mlx_lm": mlx_lm.__version__, "device": str(mx.default_device())},
        "base_model": {"path": base, "config": json.load(open(os.path.join(base, "config.json")))},
        "adapter": {"dir": adapter, "npz": os.path.basename(npz), "npz_sha256": sha256(npz),
                    "config_sha256": sha256(cfg_json), "config": json.load(open(cfg_json))},
        "this_repo": {**git_info(ROOT), "note": "dirty = uncommitted changes in the working tree at run time"},
        "reference": {**git_info(ref), "url": REF_URL,
                      "loader_path": "delta_mem_sidecar.mlx_runtime.MlxBackboneRuntime._load (in-process, no server)"},
        "method": {"forward": "prefill only: one model(ids) call per prompt, no sampling, no KV cache, fresh state",
                   "chat_template": "apply_chat_template(messages, tokenize=False, add_generation_prompt=True) → encode",
                   "same_ids_fed_to_both_ports": True},
        "independence": independence,
        "prompts": results,
        "summary": summary,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
    say(f"\n== summary ==\n  max|Δ| final-position logits over prompts: {summary['max_abs_delta_final_logits']:.4g}"
        f"\n  top-1 equal on every prompt: {summary['all_top1_match']}"
        f"\n  memory state: min cosine {summary['min_state_cosine']}, max|ΔS| {summary['max_state_abs_delta']}"
        f"\n  backbone-only max|Δ| (attention body, no sidecar): {summary['backbone_only_max_abs_delta']:.4g}"
        f"\n  this repo {doc['this_repo']['commit']}{' (dirty)' if doc['this_repo']['dirty'] else ''}; "
        f"reference {doc['reference']['commit']}; adapter sha256 {doc['adapter']['npz_sha256'][:16]}…"
        f"\n  written: {out}")
    return doc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--ref", default=DEFAULT_REF, help=f"clone of {REF_URL}")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    missing = missing_inputs(args.base, args.adapter, args.ref)
    if missing:
        print("SKIP: missing on disk:\n  " + "\n  ".join(missing))
        return 77
    try:
        run(args.base, args.adapter, args.ref, args.out)
    except Exception:
        traceback.print_exc()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
