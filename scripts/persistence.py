# -*- coding: utf-8 -*-
"""Part C — does the δ-mem state carry a fact from one turn into a fresh prompt?

  python scripts/persistence.py
  python scripts/persistence.py --base <dir> --adapter <dir> --out results/persistence.json

Protocol (fixed before the first run; not tuned afterwards)
  Five invented facts the base model cannot know, each with a distinct multi-token answer.
  turn 1  one user message stating the fact, rendered with the chat template as a user turn
          (no assistant header), run as a prefill so the sidecar writes it into S.
  turn 2  a FRESH prompt that does not contain turn 1: the question, rendered as a user turn
          with the assistant header, scored at its last position. Fresh KV cache.
  Arms at turn 2
    (a) vanilla  no sidecar attached (measured before the adapter is loaded; nothing to carry)
    (b) sidecar  S after turn 1 carried into turn 2 via get_states → set_states
    (c) reset    sidecar attached, S reset to zero between the turns (control)
  Measurements at turn 2
    log-probability of the FIRST token of the correct answer (the answer tokenised on its own,
    as it would be generated right after the assistant header), and greedy decoding
    (max 8 tokens, stop at EOS) → does the output contain the answer (case-insensitive)?
  Item 0  the model card of declare-lab/delta-mem_qwen3_4b-instruct was checked for a worked
          example to run verbatim; see ITEM0 below.
Writes results/persistence.json and prints fact × arm → log-prob, greedy hit, then the number of
facts with (b) > (a) and (b) > (c) on the first-token log-probability.
Exit 0 = ran, 77 = SKIP (weights absent), 2 = error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
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
DEFAULT_OUT = os.path.join(ROOT, "results", "persistence.json")
MAX_NEW_TOKENS = 8

ITEM0 = {
    "source": "https://huggingface.co/declare-lab/delta-mem_qwen3_4b-instruct",
    "checked_on": "2026-10-19",
    "card_commit": "6a1a990e770974811b8fb27dab73430807f76cb5",
    "worked_example_found": False,
    "note": "the card carries a loading snippet (attach_delta_mem / load_delta_mem_adapter) and a "
            "details table, but no prompt/answer example — nothing to run verbatim",
}

FACTS = [
    {"name": "varuna-codename",
     "statement": "The codename of the Varuna project is Kestrel-7.",
     "question": "What is the codename of the Varuna project? Answer with the codename only.",
     "answer": "Kestrel-7"},
    {"name": "halvorsen-password",
     "statement": "The password for the Halvorsen archive is tangerine-moss.",
     "question": "What is the password for the Halvorsen archive? Answer with the password only.",
     "answer": "tangerine-moss"},
    {"name": "orrery-oncall",
     "statement": "The on-call engineer for the Orrery cluster is Priyanka Vashti.",
     "question": "Who is the on-call engineer for the Orrery cluster? Answer with the name only.",
     "answer": "Priyanka Vashti"},
    {"name": "meridian-serial",
     "statement": "The serial number of the Meridian relay is QX-4471-B.",
     "question": "What is the serial number of the Meridian relay? Answer with the serial number only.",
     "answer": "QX-4471-B"},
    {"name": "tamarind-train",
     "statement": "The release train for the Tamarind service is called Saffron Lantern.",
     "question": "What is the release train for the Tamarind service called? Answer with the name only.",
     "answer": "Saffron Lantern"},
]


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
            "head": run("log", "-1", "--format=%h %s")}


def render(tok, text: str, generation_prompt: bool) -> str:
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                   add_generation_prompt=generation_prompt)


def encode(tok, rendered: str) -> list:
    add_special = tok.bos_token is None or not rendered.startswith(tok.bos_token)   # mlx_lm.generate's rule
    return [int(i) for i in tok.encode(rendered, add_special_tokens=add_special)]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def last_logprobs(model, ids: list) -> mx.array:
    logits = model(mx.array([ids]))
    l = logits[0, -1, :].astype(mx.float32)
    lp = l - mx.logsumexp(l)
    mx.eval(lp)
    return lp


def greedy(model, tok, ids: list, eos: set, max_tokens: int = MAX_NEW_TOKENS):
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    logits = model(mx.array([ids]), cache=cache)
    out = []
    for _ in range(max_tokens):
        nxt = int(mx.argmax(logits[0, -1, :].astype(mx.float32)))
        if nxt in eos:
            break
        out.append(nxt)
        logits = model(mx.array([[nxt]]), cache=cache)
    return out, tok.decode(out)


# --- the experiment -------------------------------------------------------------------------

def run(base: str = DEFAULT_BASE, adapter: str = DEFAULT_ADAPTER, out: str = DEFAULT_OUT) -> dict:
    import mlx_lm
    from mlx_lm import load
    from delta_mem_mlx import load_release_adapter, reset, get_states, set_states

    t0 = time.time()
    model, tok = load(base)
    eos = {int(i) for i in (getattr(tok, "eos_token_ids", None) or [tok.eos_token_id])}
    print(f"== persistence: {os.path.basename(base)} + {os.path.basename(adapter)}  (load {time.time()-t0:.1f}s) ==")

    items = []
    for f in FACTS:
        t1, t2 = render(tok, f["statement"], False), render(tok, f["question"], True)
        ans = [int(i) for i in tok.encode(f["answer"], add_special_tokens=False)]
        if len(ans) < 2:
            raise ValueError(f"{f['name']}: answer {f['answer']!r} is a single token; the protocol needs multi-token answers")
        if f["statement"] in t2 or f["answer"] in t2:
            raise ValueError(f"{f['name']}: turn 2 must not contain turn 1 or the answer")
        items.append({**f, "turn1_rendered": t1, "turn1_ids": encode(tok, t1),
                      "turn2_rendered": t2, "turn2_ids": encode(tok, t2),
                      "answer_ids": ans, "answer_tokens": [tok.decode([i]) for i in ans], "first_answer_token": ans[0],
                      "arms": {}})
        print(f"  {f['name']:<20s} answer {f['answer']!r} → tokens {items[-1]['answer_tokens']}  (turn 1: {len(items[-1]['turn1_ids'])} ids, turn 2: {len(items[-1]['turn2_ids'])} ids)")

    def measure(it: dict) -> dict:
        lp = last_logprobs(model, it["turn2_ids"])
        first = it["first_answer_token"]
        top1 = int(mx.argmax(lp))
        return {"logprob_first": float(lp[first]), "prob_first": math.exp(float(lp[first])),
                "rank_first": int(mx.sum(lp > lp[first])) + 1,
                "top1_id": top1, "top1_text": tok.decode([top1]), "top1_logprob": float(lp[top1])}

    def decode(it: dict) -> dict:
        ids, text = greedy(model, tok, it["turn2_ids"], eos)
        return {"greedy_ids": ids, "greedy_text": text,
                "greedy_hit": norm(it["answer"]) in norm(text), "greedy_exact": text.strip() == it["answer"]}

    # (a) vanilla: nothing attached, nothing to carry — the question alone
    t0 = time.time()
    for it in items:
        it["arms"]["vanilla"] = {**measure(it), **decode(it)}
    print(f"  arm (a) vanilla done ({time.time()-t0:.1f}s)")

    # attach the released sidecar (strict load)
    cfg = load_release_adapter(model, adapter)
    print(f"  adapter attached: {len(model._delta_mem_boxes)} layers, r={cfg.r}, alpha={cfg.alpha}")

    t0 = time.time()
    for it in items:
        # turn 1: write the fact into S (prefill, sidecar live, fresh state)
        reset(model)
        o = model(mx.array([it["turn1_ids"]])); mx.eval(o)
        states = get_states(model)
        mx.eval(*[s for s in states if s is not None])
        it["state_max_abs_after_turn1"] = max(float(mx.max(mx.abs(s))) for s in states if s is not None)
        # (b) sidecar: S carried into the fresh turn-2 prompt (set before each forward; the forward moves S on)
        set_states(model, states); m = measure(it)
        set_states(model, states); d = decode(it)
        it["arms"]["sidecar"] = {**m, **d}
        # (c) reset: S zeroed between the turns
        reset(model); m = measure(it)
        reset(model); d = decode(it)
        it["arms"]["reset"] = {**m, **d}
    reset(model)
    print(f"  arms (b) sidecar and (c) reset done ({time.time()-t0:.1f}s)")

    # table
    print(f"\n  {'fact':<20s} {'answer':<16s} {'(a) vanilla':<22s} {'(b) sidecar':<22s} {'(c) reset':<22s}")
    print(f"  {'':<20s} {'':<16s} {'log-p     greedy':<22s} {'log-p     greedy':<22s} {'log-p     greedy':<22s}")
    for it in items:
        cells = []
        for arm in ("vanilla", "sidecar", "reset"):
            a = it["arms"][arm]
            cells.append(f"{a['logprob_first']:>7.3f}   {'HIT ' if a['greedy_hit'] else 'miss'}")
        print(f"  {it['name']:<20s} {it['answer']:<16s} " + "   ".join(f"{c:<20s}" for c in cells))
    print("\n  greedy outputs (max 8 tokens):")
    for it in items:
        print(f"  {it['name']:<20s} (a) {it['arms']['vanilla']['greedy_text']!r}  (b) {it['arms']['sidecar']['greedy_text']!r}  (c) {it['arms']['reset']['greedy_text']!r}")

    lp = lambda it, arm: it["arms"][arm]["logprob_first"]  # noqa: E731
    summary = {
        "n_facts": len(items),
        "b_gt_a": sum(lp(it, "sidecar") > lp(it, "vanilla") for it in items),
        "b_gt_c": sum(lp(it, "sidecar") > lp(it, "reset") for it in items),
        "b_gt_a_and_b_gt_c": sum(lp(it, "sidecar") > lp(it, "vanilla") and lp(it, "sidecar") > lp(it, "reset") for it in items),
        "mean_logprob_first": {arm: sum(lp(it, arm) for it in items) / len(items) for arm in ("vanilla", "sidecar", "reset")},
        "greedy_hits": {arm: sum(it["arms"][arm]["greedy_hit"] for it in items) for arm in ("vanilla", "sidecar", "reset")},
        "item0": "not run: " + ITEM0["note"],
    }
    print(f"\n  (b) > (a): {summary['b_gt_a']}/{len(items)}   (b) > (c): {summary['b_gt_c']}/{len(items)}   "
          f"both: {summary['b_gt_a_and_b_gt_c']}/{len(items)}   "
          f"greedy hits a/b/c: {summary['greedy_hits']['vanilla']}/{summary['greedy_hits']['sidecar']}/{summary['greedy_hits']['reset']}")
    print(f"  mean log-p of the first answer token: a {summary['mean_logprob_first']['vanilla']:.3f}  "
          f"b {summary['mean_logprob_first']['sidecar']:.3f}  c {summary['mean_logprob_first']['reset']:.3f}")
    print(f"  item 0: {summary['item0']}")

    npz, cfg_json = os.path.join(adapter, "delta_mem_adapter_mlx.npz"), os.path.join(adapter, "delta_mem_config.json")
    doc = {
        "kind": "delta-mem-mlx persistence experiment (Part C)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "machine": {"platform": platform.platform(), "python": platform.python_version(),
                    "mlx": mx.__version__, "mlx_lm": mlx_lm.__version__},
        "base_model": base,
        "adapter": {"dir": adapter, "npz_sha256": sha256(npz), "config_sha256": sha256(cfg_json)},
        "this_repo": git_info(ROOT),
        "protocol": {
            "turn1": "one user message stating the fact; chat template, user turn, no assistant header; prefill only",
            "turn2": "fresh prompt without turn 1: the question; chat template, user turn + assistant header; fresh KV cache",
            "arms": {"vanilla": "no sidecar attached (run before the adapter is loaded)",
                     "sidecar": "state after turn 1 carried into turn 2 with get_states/set_states",
                     "reset": "sidecar attached, state reset to zero between the turns"},
            "logprob": "log-softmax at the last position of turn 2, of the first token of the answer tokenised on its own",
            "greedy": f"argmax decoding, max {MAX_NEW_TOKENS} tokens, stop at EOS; hit = answer contained in the output, case-insensitive",
            "sidecar_config": {"r": cfg.r, "alpha": cfg.alpha, "layers": len(model._delta_mem_boxes)},
        },
        "item0": ITEM0,
        "facts": items,
        "summary": summary,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
    print(f"  written: {out}")
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    npz, cfg_json = os.path.join(args.adapter, "delta_mem_adapter_mlx.npz"), os.path.join(args.adapter, "delta_mem_config.json")
    if not (os.path.isdir(args.base) and os.path.isfile(npz) and os.path.isfile(cfg_json)):
        print(f"SKIP: need base model at {args.base} and adapter (.npz + config) at {args.adapter}")
        return 77
    try:
        run(args.base, args.adapter, args.out)
    except Exception:
        traceback.print_exc()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
