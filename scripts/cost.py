# -*- coding: utf-8 -*-
"""Part D — what the sidecar costs on this Mac.

  python scripts/cost.py                       # benchmark → results/cost.json, then docs/milestone2.md
  python scripts/cost.py --doc-only            # regenerate docs/milestone2.md from the three results files

Two configurations in one process, in this order:
  vanilla   the 4-bit backbone as mlx-lm loads it
  sidecar   the same model object after load_release_adapter() (attach + strict load of the
            authors' Qwen3-4B sidecar, community MLX conversion)
For each configuration, after one untimed warm-up (kernel compilation), three repeats of
  prefill   one forward over a PROMPT_TOKENS-token prompt into a fresh KV cache; tokens/second
  decode    DECODE_TOKENS greedy tokens through the KV cache after that prefill; tokens/second
            counts only the decode loop, not the prefill
and peak memory (mx.get_peak_memory, mx.reset_peak_memory before every timed run; the
configuration's figure is the max over its runs; it includes the loaded weights).
Reported figures are medians; every raw value is kept in the JSON.
The prompt is a fixed English paragraph repeated and cut to exactly PROMPT_TOKENS ids, so both
configurations see the same ids.  Greedy = argmax, no sampling, no EOS stop (fixed length).

docs/milestone2.md is rendered from results/cost.json, results/crosscheck.json and
results/persistence.json; every number in it is read from those files.
Exit 0 = ran, 77 = SKIP (weights absent), 2 = error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback

import mlx.core as mx

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_BASE = os.path.expanduser("~/mlx-models/Qwen3-4B-Instruct-2507-4bit")
DEFAULT_ADAPTER = os.path.expanduser("~/mlx-models/delta-mem-qwen3-4b-adapter")
RESULTS = os.path.join(ROOT, "results")
COST_JSON = os.path.join(RESULTS, "cost.json")
CROSSCHECK_JSON = os.path.join(RESULTS, "crosscheck.json")
PERSISTENCE_JSON = os.path.join(RESULTS, "persistence.json")
DOC = os.path.join(ROOT, "docs", "milestone2.md")

PROMPT_TOKENS = 1024
DECODE_TOKENS = 128
REPEATS = 3

PROMPT_SEED = (
    "The maintenance log for the coastal relay station records every inspection in the same "
    "order: power supply, antenna alignment, cable integrity, firmware version, and the "
    "signature of the technician on duty. Entries are kept for seven years and copied to the "
    "regional archive each quarter. When a reading falls outside its tolerance band the entry "
    "is flagged, a work order is opened, and the next inspection is brought forward by a week. "
)


# --- helpers --------------------------------------------------------------------------------

def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info(path: str) -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", "-C", path, *args], capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None
    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"), "dirty": None if status is None else bool(status),
            "head": run("log", "-1", "--format=%h %s"),
            "note": "dirty = uncommitted changes in the working tree at run time"}


def sysctl(key: str):
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def machine_info() -> dict:
    import mlx_lm
    mem = sysctl("hw.memsize")
    return {"platform": platform.platform(), "machine": platform.machine(),
            "chip": sysctl("machdep.cpu.brand_string"), "cpu_cores": int(sysctl("hw.ncpu") or 0),
            "memory_bytes": int(mem) if mem else None, "memory_gib": round(int(mem) / 2**30, 1) if mem else None,
            "python": platform.python_version(), "mlx": mx.__version__, "mlx_lm": mlx_lm.__version__,
            "device": str(mx.default_device()),
            "mlx_memory_limit_bytes": int(mx.get_memory_limit())}


def build_prompt_ids(tok, n: int) -> list:
    text = PROMPT_SEED
    ids = [int(i) for i in tok.encode(text, add_special_tokens=False)]
    while len(ids) < n:
        text += PROMPT_SEED
        ids = [int(i) for i in tok.encode(text, add_special_tokens=False)]
    return ids[:n]


def median(xs):
    return float(statistics.median(xs))


# --- the benchmark --------------------------------------------------------------------------

def bench(model, ids: list, label: str) -> dict:
    """Warm-up, then REPEATS × (prefill, decode). Returns raw values and medians."""
    from mlx_lm.models.cache import make_prompt_cache

    def one(reset_state: bool):
        if reset_state:
            from delta_mem_mlx import reset
            reset(model)
        mx.reset_peak_memory()
        cache = make_prompt_cache(model)
        x = mx.array([ids])
        t0 = time.perf_counter()
        logits = model(x, cache=cache)
        nxt = mx.argmax(logits[0, -1, :].astype(mx.float32))
        mx.eval(nxt, *[c.state for c in cache if hasattr(c, "state")])
        t_prefill = time.perf_counter() - t0
        peak_prefill = int(mx.get_peak_memory())
        out = []
        t0 = time.perf_counter()
        for _ in range(DECODE_TOKENS):
            out.append(int(nxt))                       # int() forces the step; the loop is sequential
            logits = model(mx.array([[int(nxt)]]), cache=cache)
            nxt = mx.argmax(logits[0, -1, :].astype(mx.float32))
        out_last = int(nxt)
        t_decode = time.perf_counter() - t0
        peak_decode = int(mx.get_peak_memory())
        return {"prefill_s": t_prefill, "prefill_tok_s": len(ids) / t_prefill,
                "decode_s": t_decode, "decode_tok_s": DECODE_TOKENS / t_decode,
                "peak_memory_bytes": max(peak_prefill, peak_decode),
                "peak_memory_after_prefill_bytes": peak_prefill,
                "generated_ids_head": out[:8], "generated_ids_sha256": hashlib.sha256(bytes(str(out + [out_last]), "utf-8")).hexdigest()}

    has_sidecar = hasattr(model, "_delta_mem_boxes")
    print(f"  [{label}] warm-up …", flush=True)
    w = one(has_sidecar)
    print(f"  [{label}] warm-up: prefill {w['prefill_tok_s']:.0f} tok/s, decode {w['decode_tok_s']:.1f} tok/s", flush=True)
    runs = []
    for k in range(REPEATS):
        r = one(has_sidecar)
        runs.append(r)
        print(f"  [{label}] run {k+1}/{REPEATS}: prefill {r['prefill_tok_s']:.0f} tok/s ({r['prefill_s']:.3f}s)  "
              f"decode {r['decode_tok_s']:.1f} tok/s ({r['decode_s']:.2f}s)  peak {r['peak_memory_bytes']/2**30:.2f} GiB", flush=True)
    same_output = len({r["generated_ids_sha256"] for r in runs} | {w["generated_ids_sha256"]}) == 1
    return {"warmup": w, "runs": runs,
            "median": {"prefill_tok_s": median([r["prefill_tok_s"] for r in runs]),
                       "prefill_s": median([r["prefill_s"] for r in runs]),
                       "decode_tok_s": median([r["decode_tok_s"] for r in runs]),
                       "decode_s": median([r["decode_s"] for r in runs])},
            "peak_memory_bytes": max(r["peak_memory_bytes"] for r in runs),
            "peak_memory_gib": max(r["peak_memory_bytes"] for r in runs) / 2**30,
            "greedy_output_identical_across_runs": same_output}


def run(base: str, adapter: str) -> dict:
    from mlx_lm import load
    from delta_mem_mlx import load_release_adapter, count_parameters, reset

    t0 = time.time()
    model, tok = load(base)
    mx.eval(model.parameters())
    load_s = time.time() - t0
    weights_bytes = int(mx.get_active_memory())
    print(f"== cost: {os.path.basename(base)}  (load {load_s:.1f}s, {weights_bytes/2**30:.2f} GiB active) ==")

    ids = build_prompt_ids(tok, PROMPT_TOKENS)
    assert len(ids) == PROMPT_TOKENS
    print(f"  prompt: {len(ids)} ids, sha256 {hashlib.sha256(bytes(str(ids), 'utf-8')).hexdigest()[:16]}…")

    vanilla = bench(model, ids, "vanilla")

    t0 = time.time()
    cfg = load_release_adapter(model, adapter)
    mx.eval(model.parameters())
    adapter_load_s = time.time() - t0
    n_side = count_parameters(model)
    weights_with_sidecar_bytes = int(mx.get_active_memory())
    print(f"  adapter attached: {len(model._delta_mem_boxes)} layers, r={cfg.r}, alpha={cfg.alpha}, "
          f"{n_side:,} sidecar params ({adapter_load_s:.2f}s)")
    sidecar = bench(model, ids, "sidecar")
    reset(model)

    ratio = lambda k: sidecar["median"][k] / vanilla["median"][k]  # noqa: E731
    summary = {
        "prefill_tok_s": {"vanilla": vanilla["median"]["prefill_tok_s"], "sidecar": sidecar["median"]["prefill_tok_s"],
                          "sidecar_over_vanilla": ratio("prefill_tok_s")},
        "decode_tok_s": {"vanilla": vanilla["median"]["decode_tok_s"], "sidecar": sidecar["median"]["decode_tok_s"],
                         "sidecar_over_vanilla": ratio("decode_tok_s")},
        "peak_memory_gib": {"vanilla": vanilla["peak_memory_gib"], "sidecar": sidecar["peak_memory_gib"],
                            "sidecar_minus_vanilla_mib": (sidecar["peak_memory_bytes"] - vanilla["peak_memory_bytes"]) / 2**20},
    }
    print(f"\n  {'':<10s} {'prefill tok/s':>14s} {'decode tok/s':>13s} {'peak GiB':>9s}")
    for name, b in (("vanilla", vanilla), ("sidecar", sidecar)):
        print(f"  {name:<10s} {b['median']['prefill_tok_s']:>14.1f} {b['median']['decode_tok_s']:>13.2f} {b['peak_memory_gib']:>9.2f}")
    print(f"  sidecar/vanilla: prefill ×{summary['prefill_tok_s']['sidecar_over_vanilla']:.3f}  "
          f"decode ×{summary['decode_tok_s']['sidecar_over_vanilla']:.3f}  "
          f"peak {summary["peak_memory_gib"]["sidecar_minus_vanilla_mib"]:+.1f} MiB")

    npz, cfg_json = os.path.join(adapter, "delta_mem_adapter_mlx.npz"), os.path.join(adapter, "delta_mem_config.json")
    doc = {
        "kind": "delta-mem-mlx cost benchmark (Part D)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "machine": machine_info(),
        "this_repo": git_info(ROOT),
        "base_model": {"path": base, "load_s": load_s, "active_memory_after_load_bytes": weights_bytes},
        "adapter": {"dir": adapter, "npz_sha256": sha256(npz), "config_sha256": sha256(cfg_json),
                    "load_s": adapter_load_s, "layers": len(model._delta_mem_boxes), "r": cfg.r, "alpha": cfg.alpha,
                    "sidecar_params": n_side, "active_memory_after_attach_bytes": weights_with_sidecar_bytes},
        "protocol": {
            "prompt_tokens": PROMPT_TOKENS, "decode_tokens": DECODE_TOKENS, "repeats": REPEATS,
            "warmup": "one untimed prefill+decode per configuration before the timed repeats",
            "prefill": "one forward over the prompt into a fresh KV cache; tok/s = prompt_tokens / wall time (mx.eval of the argmax and the cache)",
            "decode": "greedy argmax, one token per forward through the KV cache, decode_tokens steps; tok/s = decode_tokens / wall time of the loop only",
            "peak_memory": "mx.reset_peak_memory before each run, mx.get_peak_memory after prefill and after decode; per configuration the max over the three runs; includes the loaded weights",
            "reported": "median of the three repeats",
            "prompt": {"seed_text": PROMPT_SEED, "construction": "seed repeated and cut to prompt_tokens ids; no chat template",
                       "ids_sha256": hashlib.sha256(bytes(str(ids), "utf-8")).hexdigest()},
            "order": "vanilla measured first, then load_release_adapter on the same model object, then sidecar",
        },
        "vanilla": vanilla,
        "sidecar": sidecar,
        "summary": summary,
    }
    os.makedirs(RESULTS, exist_ok=True)
    with open(COST_JSON, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
    print(f"  written: {COST_JSON}")
    return doc


# --- the document ---------------------------------------------------------------------------

def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_doc(out: str = DOC) -> str:
    cost, xc, pe = _load(COST_JSON), _load(CROSSCHECK_JSON), _load(PERSISTENCE_JSON)
    m, s = cost["machine"], cost["summary"]
    v, sc = cost["vanilla"], cost["sidecar"]
    p = cost["protocol"]
    xs, ps = xc["summary"], pe["summary"]
    rel = lambda path: os.path.relpath(path, ROOT)  # noqa: E731

    def gib(b): return f"{b/2**30:.2f} GiB"
    def arm(it, a): return it["arms"][a]

    L = []
    L.append("# Milestone 2 — the released Qwen3-4B sidecar on MLX\n")
    L.append("*Generated by `scripts/cost.py --doc-only` from `results/cost.json`, `results/crosscheck.json` and "
             "`results/persistence.json`. Every number below is read from those files; none is typed by hand. "
             "Regenerate rather than edit.*\n")
    L.append("## Setup\n")
    L.append("| | |\n|---|---|")
    L.append(f"| Machine | {m['chip']}, {m['memory_gib']} GiB, {m['cpu_cores']} cores, {m['platform']} |")
    L.append(f"| Software | Python {m['python']}, mlx {m['mlx']}, mlx-lm {m['mlx_lm']}, device {m['device']} |")
    L.append(f"| Backbone | `{os.path.basename(cost['base_model']['path'])}` ({xc['base_model']['config']['num_hidden_layers']} layers, "
             f"{xc['base_model']['config']['quantization']['bits']}-bit, group size {xc['base_model']['config']['quantization']['group_size']}) |")
    L.append(f"| Sidecar | `{os.path.basename(cost['adapter']['dir'])}`: {cost['adapter']['layers']} layers, r={cost['adapter']['r']}, "
             f"alpha={cost['adapter']['alpha']}, {cost['adapter']['sidecar_params']:,} parameters; npz sha256 `{cost['adapter']['npz_sha256'][:16]}…` |")
    L.append(f"| Repo commit (cost run) | `{cost['this_repo']['commit'][:12]}`{' (dirty tree)' if cost['this_repo']['dirty'] else ''} |")
    L.append(f"| Repo commit (crosscheck) | `{xc['this_repo']['commit'][:12]}`{' (dirty tree)' if xc['this_repo']['dirty'] else ''} |")
    L.append(f"| Repo commit (persistence) | `{pe['this_repo']['commit'][:12]}`{' (dirty tree)' if pe['this_repo']['dirty'] else ''} |")
    L.append(f"| Run timestamps | cost {cost['timestamp']}; crosscheck {xc['timestamp']}; persistence {pe['timestamp']} |\n")

    L.append("## 1. Cross-check against an independent MLX port\n")
    L.append(f"Same adapter tensors, same backbone weights, same token ids, fed to this repo's hook-based port and to "
             f"[{xc['reference']['url'].split('github.com/')[-1]}]({xc['reference']['url']}) (commit `{xc['reference']['commit'][:12]}`), "
             f"which re-implements the attention body. Prefill only, fresh state, {len(xc['prompts'])} prompts.\n")
    L.append("| Quantity | Value |\n|---|---|")
    L.append(f"| max abs Δ of final logits | {xs['max_abs_delta_final_logits']} |")
    L.append(f"| top-1 token identical on every prompt | {xs['all_top1_match']} |")
    L.append(f"| min cosine between the ports' final states | {xs['min_state_cosine']:.12f} |")
    L.append(f"| max abs Δ between the ports' final states | {xs['max_state_abs_delta']:.3e} |")
    L.append(f"| backbone-only (sidecar detached) max abs Δ | {xs['backbone_only_max_abs_delta']} |")
    L.append(f"| adapter tensors bit-identical in both ports | {xc['independence']['adapter_tensors_bit_identical_in_both_ports']} |")
    L.append(f"| distinct model objects / backbone weights | {xc['independence']['distinct_model_objects']} / {xc['independence']['distinct_backbone_weights']} |")
    L.append("")
    L.append("Per prompt, first layer whose hidden state differs between the ports:\n")
    L.append("| Prompt | Kind | Tokens | First divergent layer | Attribution |\n|---|---|---|---|---|")
    for pr in xc["prompts"]:
        n_tok = pr.get("n_tokens", pr.get("n_ids", len(pr.get("ids", [])) or "—"))
        L.append(f"| {pr['name']} | {pr['kind']} | {n_tok} | {xs['first_divergent_layer'].get(pr['name'])} | {xs['attribution'].get(pr['name'])} |")
    L.append("")

    L.append("## 2. Cost on this Mac\n")
    L.append(f"{p['prompt_tokens']}-token prompt, {p['decode_tokens']} greedy tokens, {p['repeats']} repeats after one warm-up, medians. "
             f"Peak memory from `mx.get_peak_memory` (reset before every run), max over the runs, weights included.\n")
    L.append("| Configuration | Prefill tok/s | Decode tok/s | Peak memory |\n|---|---:|---:|---:|")
    L.append(f"| vanilla | {s['prefill_tok_s']['vanilla']:.1f} | {s['decode_tok_s']['vanilla']:.2f} | {gib(v['peak_memory_bytes'])} |")
    L.append(f"| sidecar | {s['prefill_tok_s']['sidecar']:.1f} | {s['decode_tok_s']['sidecar']:.2f} | {gib(sc['peak_memory_bytes'])} |")
    L.append(f"| sidecar / vanilla | ×{s['prefill_tok_s']['sidecar_over_vanilla']:.3f} | ×{s['decode_tok_s']['sidecar_over_vanilla']:.3f} | "
             f"{s['peak_memory_gib']['sidecar_minus_vanilla_mib']:+.1f} MiB |")
    L.append("")
    L.append("Raw repeats (prefill tok/s · decode tok/s · peak):\n")
    L.append("| Configuration | Run 1 | Run 2 | Run 3 | Greedy output identical across runs |\n|---|---|---|---|---|")
    for name, b in (("vanilla", v), ("sidecar", sc)):
        cells = " | ".join(f"{r['prefill_tok_s']:.1f} · {r['decode_tok_s']:.2f} · {gib(r['peak_memory_bytes'])}" for r in b["runs"])
        L.append(f"| {name} | {cells} | {b['greedy_output_identical_across_runs']} |")
    L.append("")
    L.append(f"Model load {cost['base_model']['load_s']:.1f} s ({gib(cost['base_model']['active_memory_after_load_bytes'])} active); "
             f"adapter attach + load {cost['adapter']['load_s']:.2f} s "
             f"({gib(cost['adapter']['active_memory_after_attach_bytes'])} active afterwards).\n")

    L.append("## 3. Persistence, short context (Part C)\n")
    pp = pe["protocol"]
    L.append(f"{ps['n_facts']} invented facts. Turn 1: {pp['turn1']}. Turn 2: {pp['turn2']}. "
             f"Log-prob: {pp['logprob']}. Greedy: {pp['greedy']}.\n")
    L.append("| Fact | Answer | (a) vanilla log-p | (b) sidecar log-p | (c) reset log-p | greedy a / b / c |\n|---|---|---:|---:|---:|---|")
    for it in pe["facts"]:
        g = " / ".join("hit" if arm(it, a)["greedy_hit"] else "miss" for a in ("vanilla", "sidecar", "reset"))
        L.append(f"| {it['name']} | {it['answer']} | {arm(it,'vanilla')['logprob_first']:.3f} | {arm(it,'sidecar')['logprob_first']:.3f} | "
                 f"{arm(it,'reset')['logprob_first']:.3f} | {g} |")
    ml = ps["mean_logprob_first"]
    L.append(f"| **mean** | | {ml['vanilla']:.3f} | {ml['sidecar']:.3f} | {ml['reset']:.3f} | "
             f"{ps['greedy_hits']['vanilla']} / {ps['greedy_hits']['sidecar']} / {ps['greedy_hits']['reset']} hits |")
    L.append("")
    L.append(f"(b) > (a) on the first-token log-prob: {ps['b_gt_a']}/{ps['n_facts']}. "
             f"(b) > (c): {ps['b_gt_c']}/{ps['n_facts']}. Both: {ps['b_gt_a_and_b_gt_c']}/{ps['n_facts']}. "
             f"The (b)–(a) gap is the adapter being attached at all (compare (c), which has the adapter but no memory); "
             f"the (b)–(c) gap is the memory.\n")
    L.append(f"Item 0 (worked example from the model card): {ps['item0']}.\n")
    L.append("Greedy outputs at turn 2:\n")
    L.append("| Fact | (a) vanilla | (b) sidecar | (c) reset |\n|---|---|---|---|")
    for it in pe["facts"]:
        L.append("| " + it["name"] + " | " + " | ".join(f"`{arm(it,a)['greedy_text']!s}`".replace("\n", "⏎") for a in ("vanilla", "sidecar", "reset")) + " |")
    L.append("")
    L.append("## Sources\n")
    for path in (COST_JSON, CROSSCHECK_JSON, PERSISTENCE_JSON):
        L.append(f"- `{rel(path)}` — sha256 `{sha256(path)}`")
    L.append("")
    text = "\n".join(L)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"  written: {out}")
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--doc-only", action="store_true", help="only render docs/milestone2.md from the results files")
    args = ap.parse_args()
    try:
        if not args.doc_only:
            npz, cfg_json = os.path.join(args.adapter, "delta_mem_adapter_mlx.npz"), os.path.join(args.adapter, "delta_mem_config.json")
            if not (os.path.isdir(args.base) and os.path.isfile(npz) and os.path.isfile(cfg_json)):
                print(f"SKIP: need base model at {args.base} and adapter (.npz + config) at {args.adapter}")
                return 77
            run(args.base, args.adapter)
        for path in (COST_JSON, CROSSCHECK_JSON, PERSISTENCE_JSON):
            if not os.path.isfile(path):
                print(f"SKIP doc: missing {path}")
                return 77
        write_doc()
    except Exception:
        traceback.print_exc()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
