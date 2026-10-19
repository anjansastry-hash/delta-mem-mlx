# -*- coding: utf-8 -*-
"""delta-mem-mlx — milestone 1: the δ-mem mechanism as a sidecar on a frozen mlx-lm model.

What this is
------------
δ-mem (arXiv 2605.12357, declare-lab/delta-Mem) adds, to every attention layer of a
frozen backbone, a tiny state S (r×r, r = 8 by default) that is rewritten on every
token by a gated delta rule, and whose readout nudges the attention query (Δq) and the
attention output (Δo). The backbone never changes. This module implements exactly that
mechanism on top of any mlx-lm decoder whose attention has `q_proj` and `o_proj`
sub-modules (Qwen2/2.5, Qwen3, Llama, Mistral, Gemma... — anything with that shape).

How it is attached
------------------
No model class is subclassed. For each layer we replace

    attn.q_proj  ->  _QHook(base=q_proj)   :  q = q_proj(x) + (α/r)·A_q(r_t)     [READ + WRITE]
    attn.o_proj  ->  _OHook(base=o_proj)   :  o = o_proj(a) + (α/r)·A_o(r_t)     [READ]

where r_t = S_{t-1} · W_q^m x_t is the memory readout for token t, computed in the q hook
and handed to the o hook of the same forward pass. The write happens in the q hook,
after the read (read-then-write ordering, per token):

    k = W_k^m x,  v = W_v^m x,  β = σ(W_β x + b),  λ = 1 − β
    S ← Diag(λ) S + Diag(β) (v − S k) kᵀ

The sidecar and the state are float32 and the state lives OUTSIDE the parameter tree (it is activation
state, like the KV cache, not a weight). One S per sequence in the batch.

What the community MLX port settled (elimaine/delta-mem-mlx-sidecar, ofthetrees adapter)
-------------------------------------------------------------------------------------------
Read against that code on 2026-09-16 — the mechanism here matches it line for line:
* Δq is added to the raw q_proj output, BEFORE reshape / q_norm (Qwen3) / RoPE. ✓
* Δo is added AFTER o_proj. ✓
* Read-then-write per token; write = keep·S − erase·(Sk)kᵀ + write·v kᵀ with
  keep = λ = 1 − β, erase = write = β  ("standard" mode)  ⇔  λS + β(v − Sk)kᵀ. ✓
* Memory q and k are tanh'd then L2-normalised before use (normalize_qk = True). ✓ (added)
* Two low-rank maps delta_q_proj: r → q_dim and delta_o_proj: r → d_model, scaled α/r. ✓
* Tensor names in the converted .npz: layers.{i}.self_attn.{memory_q_proj, memory_k_proj,
  memory_v_proj, delta_q_proj, delta_o_proj, beta_proj, beta_bias[, lambda_proj, lambda_bias]}.
Not implemented here: uncoupled λ (lambda_proj), "lambda_outside"/"no_lambda" modes,
SSW/MSW schedules, token masks for padded batches. Nothing here is verified against the
CUDA reference yet — that is milestone 2 and it needs a GPU run for the ground truth.
"""
from __future__ import annotations

import json
import os
import re
import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

__all__ = ["DeltaMemConfig", "Sidecar", "attach", "detach", "reset", "get_states",
           "set_states", "set_gate_scale", "sidecar_parameters", "save_sidecar",
           "load_sidecar", "count_parameters", "load_release_adapter", "release_weights"]


@dataclass
class DeltaMemConfig:
    r: int = 8            # memory rank: S is r×r per layer
    alpha: float = 16.0   # LoRA-style scaling on the corrections: (alpha / r)
    layers: Optional[List[int]] = None  # None = every layer
    normalize_qk: bool = True  # tanh + L2-normalise the memory query/key (release default)

    @property
    def scaling(self) -> float:
        return self.alpha / self.r

    @classmethod
    def from_release_json(cls, raw: dict) -> "DeltaMemConfig":
        """Config keys as written by the community MLX conversion (ofthetrees/…-mlx-adapter):
        rank, alpha, delta_heads, normalize_qk, couple_lambda, state_update_mode."""
        if not raw.get("couple_lambda", True) or raw.get("state_update_mode", "standard") != "standard":
            raise NotImplementedError("only couple_lambda=True, state_update_mode=standard is implemented")
        if set(raw.get("delta_heads", ("q", "o"))) != {"q", "o"}:
            raise NotImplementedError("only delta_heads = (q, o) is implemented")
        return cls(r=int(raw["rank"]), alpha=float(raw.get("alpha", raw["rank"])),
                   normalize_qk=bool(raw.get("normalize_qk", True)))


class _Box:
    """Plain python object (not an nn.Module, not a dict/list) so that nn.Module does not
    pull its contents into the parameter tree. Carries the per-layer runtime state."""

    def __init__(self, sidecar: "Sidecar"):
        self.sidecar = sidecar
        self.state: Optional[mx.array] = None   # (B, r, r) float32, or None = zeros
        self.readout: Optional[mx.array] = None  # (B, L, r) from the last q hook
        self.gate_scale: float = 1.0            # 0.0 => β ≡ 0 => S never changes => identity


class Sidecar(nn.Module):
    """The trained part of δ-mem for ONE layer: six projections, 4.87M params total on
    Qwen3-4B (paper). Nothing in here is the state; the state lives in the _Box."""

    def __init__(self, d_model: int, q_dim: int, cfg: DeltaMemConfig):
        super().__init__()
        r = cfg.r
        self.wq_m = nn.Linear(d_model, r, bias=False)
        self.wk_m = nn.Linear(d_model, r, bias=False)
        self.wv_m = nn.Linear(d_model, r, bias=False)
        self.w_beta = nn.Linear(d_model, r, bias=True)
        self.a_q = nn.Linear(r, q_dim, bias=False)
        self.a_o = nn.Linear(r, d_model, bias=False)
        self._scaling = cfg.scaling
        self._r = r
        self._normalize_qk = cfg.normalize_qk

    # --- the mechanism -------------------------------------------------------------

    def read_and_write(self, x: mx.array, state: Optional[mx.array], gate_scale: float):
        """x: (B, L, D). Returns (readout (B, L, r) float32, new_state (B, r, r) float32).

        Sequential over L by construction (token t reads S_{t-1}, then writes S_t). The
        projections are batched over L; only the r×r update is a python loop.
        """
        B, L, _ = x.shape
        r = self._r
        xf = x.astype(mx.float32)   # sidecar runs in fp32: the r×r recurrence amplifies bf16
                                    # kernel noise, and prefill (L>1) vs decode (L=1) use
                                    # different Metal matmul kernels — fp32 keeps them equal
        qm = self.wq_m(xf).astype(mx.float32)          # (B, L, r)
        km = self.wk_m(xf).astype(mx.float32)
        vm = self.wv_m(xf).astype(mx.float32)
        if self._normalize_qk:                         # tanh then unit-norm, as in the release
            qm, km = _tanh_l2(qm), _tanh_l2(km)
        beta = mx.sigmoid(self.w_beta(xf).astype(mx.float32))
        if gate_scale != 1.0:
            beta = beta * gate_scale                   # gate_scale = 0 → β ≡ 0 → identity
        S = state if state is not None else mx.zeros((B, r, r), dtype=mx.float32)
        outs = []
        for t in range(L):
            q_t = qm[:, t, :]                          # (B, r)
            r_t = (S @ q_t[..., None])[..., 0]         # read:  r_t = S q_m
            outs.append(r_t)
            k_t, v_t, b_t = km[:, t, :], vm[:, t, :], beta[:, t, :]
            Sk = (S @ k_t[..., None])[..., 0]          # what the state predicts for k
            err = v_t - Sk                             # the residual
            outer = err[:, :, None] * k_t[:, None, :]  # (B, r, r)  (v − S k) kᵀ
            S = (1.0 - b_t)[:, :, None] * S + b_t[:, :, None] * outer   # Diag(λ)S + Diag(β)(...)
        readout = mx.stack(outs, axis=1)               # (B, L, r)
        return readout, S

    def delta_q(self, readout: mx.array, dtype) -> mx.array:
        return (self._scaling * self.a_q(readout)).astype(dtype)

    def delta_o(self, readout: mx.array, dtype) -> mx.array:
        return (self._scaling * self.a_o(readout)).astype(dtype)


def _tanh_l2(x: mx.array) -> mx.array:
    x = mx.tanh(x)
    n = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True))
    return x / mx.maximum(n, mx.array(1e-6, dtype=x.dtype))


class _QHook(nn.Module):
    def __init__(self, base: nn.Module, box: _Box):
        super().__init__()
        self.base = base
        self._box = box

    def __call__(self, x: mx.array) -> mx.array:
        q = self.base(x)
        box = self._box
        readout, new_state = box.sidecar.read_and_write(x, box.state, box.gate_scale)
        box.state = new_state
        box.readout = readout
        return q + box.sidecar.delta_q(readout, q.dtype)


class _OHook(nn.Module):
    def __init__(self, base: nn.Module, box: _Box):
        super().__init__()
        self.base = base
        self._box = box

    def __call__(self, a: mx.array) -> mx.array:
        o = self.base(a)
        box = self._box
        if box.readout is None:
            return o
        return o + box.sidecar.delta_o(box.readout, o.dtype)


# --- attach / detach --------------------------------------------------------------

def _layers(model):
    inner = getattr(model, "model", model)
    return inner.layers


def _attn(layer):
    for name in ("self_attn", "attention", "attn"):
        if hasattr(layer, name):
            return getattr(layer, name)
    raise AttributeError("no attention module found on layer (tried self_attn/attention/attn)")


def _linear_dims(lin: nn.Module):
    """(in_features, out_features) for nn.Linear or nn.QuantizedLinear."""
    w = lin.weight
    if isinstance(lin, nn.QuantizedLinear):
        # quantized weight is packed: (out, in * bits / 32) uint32
        out_f = w.shape[0]
        in_f = w.shape[1] * 32 // lin.bits
        return in_f, out_f
    out_f, in_f = w.shape
    return in_f, out_f


def attach(model, cfg: DeltaMemConfig = DeltaMemConfig(), init_scale: float = 0.02):
    """Wrap every (selected) attention layer of `model` in place. Returns the list of
    per-layer _Box objects (state carriers). Sidecar weights are randomly initialised
    (normal, std=init_scale) — replace them with load_sidecar() at milestone 2."""
    boxes: List[_Box] = []
    for i, layer in enumerate(_layers(model)):
        if cfg.layers is not None and i not in cfg.layers:
            continue
        attn = _attn(layer)
        if isinstance(attn.q_proj, _QHook):
            raise RuntimeError(f"layer {i} already has a δ-mem sidecar attached")
        d_model, q_dim = _linear_dims(attn.q_proj)
        sc = Sidecar(d_model, q_dim, cfg)
        # deterministic small init so that runs are reproducible
        params = {}
        for k, v in tree_flatten(sc.parameters()):
            key = mx.random.key(zlib.crc32(f"{i}:{k}".encode()))
            params[k] = (mx.random.normal(v.shape, key=key) * init_scale).astype(v.dtype)
        sc.update(tree_unflatten(list(params.items())))
        sc.set_dtype(mx.float32)   # sidecar stays fp32 (19 MB for Qwen3-4B); Δq/Δo are cast to the backbone dtype
        attn.delta_mem = sc
        box = _Box(sc)
        attn.q_proj = _QHook(attn.q_proj, box)
        attn.o_proj = _OHook(attn.o_proj, box)
        boxes.append(box)
    model._delta_mem_boxes = boxes
    mx.eval(model.parameters())
    return boxes


def _param_dtype(lin: nn.Module):
    if isinstance(lin, nn.QuantizedLinear):
        return lin.scales.dtype
    return lin.weight.dtype


def detach(model):
    """Undo attach(): restore the original q_proj / o_proj and drop the sidecars."""
    for layer in _layers(model):
        attn = _attn(layer)
        if isinstance(attn.q_proj, _QHook):
            attn.q_proj = attn.q_proj.base
        if isinstance(attn.o_proj, _OHook):
            attn.o_proj = attn.o_proj.base
        if hasattr(attn, "delta_mem"):
            del attn["delta_mem"]
    if hasattr(model, "_delta_mem_boxes"):
        del model._delta_mem_boxes


def _boxes(model) -> List[_Box]:
    boxes = getattr(model, "_delta_mem_boxes", None)
    if not boxes:
        raise RuntimeError("no δ-mem sidecar attached — call attach(model) first")
    return boxes


def reset(model):
    """Start a new shift: zero every layer's state. Call at every conversation boundary."""
    for b in _boxes(model):
        b.state = None
        b.readout = None


def get_states(model) -> List[Optional[mx.array]]:
    """The per-layer S tensors ((B, r, r) float32 or None). Serialise these to make the
    state outlive the process — an engineering choice the paper does not evaluate."""
    return [b.state for b in _boxes(model)]


def set_states(model, states: List[Optional[mx.array]]):
    boxes = _boxes(model)
    assert len(states) == len(boxes)
    for b, s in zip(boxes, states):
        b.state = s
        b.readout = None


def set_gate_scale(model, scale: float):
    """0.0 forces β ≡ 0 on every layer: the state never moves from zero, the readout is
    zero, Δq = Δo = 0 and the model must be bit-for-bit the vanilla model."""
    for b in _boxes(model):
        b.gate_scale = float(scale)


# --- weights -------------------------------------------------------------------------

def sidecar_parameters(model) -> Dict[str, mx.array]:
    return {k: v for k, v in tree_flatten(model.parameters()) if ".delta_mem." in k}


def count_parameters(model) -> int:
    return sum(v.size for v in sidecar_parameters(model).values())


def save_sidecar(model, path: str, cfg: DeltaMemConfig):
    os.makedirs(path, exist_ok=True)
    mx.save_safetensors(os.path.join(path, "sidecar.safetensors"), sidecar_parameters(model))
    with open(os.path.join(path, "delta_mem_config.json"), "w") as f:
        json.dump({"r": cfg.r, "alpha": cfg.alpha, "layers": cfg.layers}, f, indent=2)


# Tensor names in the community MLX conversion of the authors' checkpoint
# (ofthetrees/delta-mem-qwen3-4b-instruct-mlx-adapter → delta_mem_adapter_mlx.npz):
#   [prefix.]layers.{i}.self_attn.{memory_q_proj|memory_k_proj|memory_v_proj|
#                                  delta_q_proj|delta_o_proj|beta_proj|beta_bias
#                                  [|lambda_proj|lambda_bias]}
_RELEASE_RE = re.compile(r"(?:^|\.)layers\.(?P<layer>\d+)\.self_attn\.(?P<name>memory_q_proj|memory_k_proj|"
                         r"memory_v_proj|delta_q_proj|delta_o_proj|beta_proj|beta_bias|lambda_proj|lambda_bias)$")
_RELEASE_TO_OURS = {"memory_q_proj": "wq_m.weight", "memory_k_proj": "wk_m.weight",
                    "memory_v_proj": "wv_m.weight", "delta_q_proj": "a_q.weight",
                    "delta_o_proj": "a_o.weight", "beta_proj": "w_beta.weight", "beta_bias": "w_beta.bias"}


def release_weights(npz_path: str) -> Dict[int, Dict[str, mx.array]]:
    """Group a release .npz by layer → {release_name: array}."""
    out: Dict[int, Dict[str, mx.array]] = {}
    for k, v in mx.load(npz_path).items():
        m = _RELEASE_RE.search(k)
        if m:
            out.setdefault(int(m.group("layer")), {})[m.group("name")] = v
    if not out:
        raise ValueError(f"no δ-mem tensors found in {npz_path}")
    return out


def load_release_adapter(model, adapter_dir: str):
    """Attach + load the community MLX conversion of the authors' Qwen3-4B sidecar.
    Expects delta_mem_config.json and delta_mem_adapter_mlx.npz in adapter_dir.
    Milestone 2: NOT yet verified against the CUDA reference — see README."""
    with open(os.path.join(adapter_dir, "delta_mem_config.json")) as f:
        raw = json.load(f)
    cfg = DeltaMemConfig.from_release_json(raw)
    groups = release_weights(os.path.join(adapter_dir, "delta_mem_adapter_mlx.npz"))
    if any("lambda_proj" in g for g in groups.values()):
        raise NotImplementedError("adapter carries lambda_proj; only coupled λ = 1 − β is implemented")
    cfg.layers = sorted(groups)
    attach(model, cfg)
    weights = {}
    for i, g in groups.items():
        for rel, ours in _RELEASE_TO_OURS.items():
            if rel not in g:
                raise ValueError(f"layer {i}: missing {rel}")
            weights[f"{_layer_prefix(model)}.{i}.self_attn.delta_mem.{ours}"] = g[rel].astype(mx.float32)
    _load_sidecar_weights(model, weights)
    return cfg


def _layer_prefix(model) -> str:
    return "model.layers" if hasattr(model, "model") else "layers"


def load_sidecar(model, path: str):
    """Load sidecar weights saved by save_sidecar(). Converting the authors' PyTorch
    checkpoint into this layout is milestone 2 (scripts/convert_authors_checkpoint.py,
    not written yet)."""
    weights = mx.load(os.path.join(path, "sidecar.safetensors"))
    _load_sidecar_weights(model, dict(weights))


def _load_sidecar_weights(model, weights: Dict[str, mx.array]):
    """Strict load of {full parameter name: array} into the attached sidecars.

    model.load_weights(..., strict=True) cannot be used for this: strict checks the given
    keys against the model's WHOLE parameter tree, and a sidecar file never carries the
    backbone, so it would always fail with "Missing N parameters". So the check is made
    against the sidecar parameters only, and every layer's Sidecar module then loads its own
    tensors with strict=True. The error message is the explicit report — every missing,
    unexpected and mis-shaped key, over all layers — and nothing is loaded unless the whole
    set is right.
    """
    expected = sidecar_parameters(model)   # empty if attach() was not called → everything is unexpected
    missing = sorted(set(expected) - set(weights))
    unexpected = sorted(set(weights) - set(expected))
    mismatch = sorted((k, tuple(expected[k].shape), tuple(weights[k].shape))
                      for k in set(expected) & set(weights) if expected[k].shape != weights[k].shape)
    if missing or unexpected or mismatch:
        raise ValueError(f"sidecar weights do not match the attached sidecars: "
                         f"{len(missing)} missing of {len(expected)} expected {missing}; "
                         f"{len(unexpected)} unexpected of {len(weights)} provided {unexpected}; "
                         f"{len(mismatch)} shape mismatches {mismatch}")
    for i, layer in enumerate(_layers(model)):
        attn = _attn(layer)
        if hasattr(attn, "delta_mem"):
            prefix = f"{_layer_prefix(model)}.{i}.self_attn.delta_mem."
            attn.delta_mem.load_weights([(k[len(prefix):], v) for k, v in weights.items()
                                         if k.startswith(prefix)], strict=True)
    mx.eval(model.parameters())
