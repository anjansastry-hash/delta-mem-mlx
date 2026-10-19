# -*- coding: utf-8 -*-
"""Part E — long-context persistence: five facts buried in a ~2,000-token memo.

  python scripts/persistence_long.py
  python scripts/persistence_long.py --base <dir> --adapter <dir> --out results/persistence_long.json

Protocol (fixed before the first run; not tuned afterwards)
  Facts   the five of results/persistence.json (Part C): name, statement, question, answer.
  turn 1  ONE user message holding a made-up technical memo of about 2,000 tokens in which the
          five facts appear as prose sentences. The memo is written once (MEMO below), saved to
          results/memo.txt on the first run and READ BACK from that file on every later run, so
          every arm of every run sees the same text. Rendered with the chat template as a user
          turn (no assistant header); prefill only, sidecar live, fresh state → S.
  turn 2  for each fact a FRESH prompt without the memo: the Part C question, verbatim, rendered
          as a user turn with the assistant header, scored at its last position. Fresh KV cache.
  Arms at turn 2
    (a) vanilla  no sidecar attached (measured before the adapter is loaded)
    (b) sidecar  S after turn 1 carried into turn 2 via get_states → set_states
    (c) reset    sidecar attached, S reset to zero between the turns (control)
  Measurements at turn 2
    log-probability of the FIRST token of the correct answer (tokenised on its own), and greedy
    decoding (max 8 tokens, stop at EOS) → does the output contain the answer (case-insensitive)?
  OOM  if the turn-1 prefill with the sidecar fails for memory, the failure is recorded in the
       JSON (status = "oom") and the script stops.
Writes results/persistence_long.json and prints fact × arm → log-prob, greedy hit, then the number
of facts with (b) > (a) and (b) > (c) on the first-token log-probability.
Exit 0 = ran, 77 = SKIP (weights absent), 2 = error, 3 = OOM recorded.
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
RESULTS = os.path.join(ROOT, "results")
PART_C_JSON = os.path.join(RESULTS, "persistence.json")
MEMO_TXT = os.path.join(RESULTS, "memo.txt")
DEFAULT_OUT = os.path.join(RESULTS, "persistence_long.json")
MAX_NEW_TOKENS = 8

# The memo. Written once; the file results/memo.txt is authoritative after the first run.
MEMO = """INTERNAL MEMO — Platform Operations
To: Site reliability, network engineering, release management
From: Infrastructure coordination desk
Subject: Quarterly consolidation of operating notes for the Varuna, Halvorsen, Orrery, Meridian and Tamarind workstreams
Classification: internal, do not forward outside the operations group

1. Purpose

This memo consolidates the operating notes that have accumulated across five workstreams since the last quarterly review. Each section records the current state of the work, the people and identifiers that other teams keep asking for, and the open items that need an owner before the next review. The memo replaces the scattered wiki pages, chat pins and ticket comments that have served as the record so far; where this memo and an older page disagree, this memo is correct and the older page should be retired. Nothing here is new policy. It is the existing practice written down in one place so that the on-call rotation, the release managers and the archive custodians stop depending on institutional memory.

2. The Varuna project

Varuna is the replacement for the regional telemetry collector that has run on the old appliance fleet for six years. The new collector ingests the same counters over the same transport but stores them in a columnar format and serves them through a query layer that the dashboards can hit directly, which removes the nightly export that has been the source of most of the stale-dashboard complaints. Internally the Varuna project is referred to by its codename, Kestrel-7, and that codename is what appears on the change tickets, the capacity forecasts and the budget line, so anyone searching the ticket system for "Varuna" and finding nothing should search for the codename instead. The project is in its third phase. Phase one proved the ingestion path at one tenth of production volume. Phase two ran the new collector in parallel with the old appliances for eight weeks and compared the stored counters hour by hour; the comparison found two classes of discrepancy, both traced to time-zone handling in the old exporter rather than to the new collector, and both were documented and accepted. Phase three, now in progress, moves the dashboards onto the new query layer one region at a time, starting with the smallest region so that any regression is cheap to roll back. The decommissioning of the appliance fleet is scheduled for the quarter after the last region is migrated and will not begin until the archive team confirms that six years of appliance exports have been copied and verified.

3. The Halvorsen archive

The Halvorsen archive is the cold store for configuration snapshots, signed build manifests and the audit logs that the compliance programme requires us to keep for seven years. It is intentionally slow to reach: there is no API, access is through a single bastion, and the retrieval tooling asks for a reason string that is logged alongside every read. The password for the Halvorsen archive is tangerine-moss, and it is rotated on the first working day of each quarter by the archive custodian, who sends the new value through the usual secrets channel and never by mail or chat. Teams that need regular access should request a service account rather than reusing the shared credential, because the shared credential is being phased out and the audit programme has flagged shared-credential reads as a finding twice in a row. The archive currently holds four hundred and twelve thousand objects and grows by roughly nine thousand per month. Retrieval latency is measured in minutes, not milliseconds, by design, and the two tickets asking for the archive to be made faster have been closed as working-as-intended; anything that needs to be fast belongs in the warm tier, not in Halvorsen.

4. The Orrery cluster

Orrery is the batch compute cluster that runs the nightly reconciliation jobs, the weekly capacity model and the ad-hoc analysis that the finance and planning teams submit through the scheduler portal. It is the one system in this memo with a formal on-call rotation, because a missed nightly run delays the next morning's reconciliation for every downstream team. The on-call engineer for the Orrery cluster is Priyanka Vashti, who holds the rotation until the end of the quarter and should be paged, not messaged, for anything that affects the nightly window. The rotation hands over at nine in the morning on the first Monday of each quarter and the handover note must list every job that was manually re-run during the period, with the reason, so that the recurring failures can be fixed rather than nursed. Two recurring failures are already known: the capacity model occasionally exceeds its memory reservation when the input tables are larger than usual, and one of the reconciliation jobs depends on a view that is rebuilt at the same time the job starts. Fixes for both are in review. Until they land, the on-call runbook describes the manual re-run for each, and the runbook is the only sanctioned way to intervene; nobody should be editing the scheduler's job definitions by hand during an incident.

5. The Meridian relay

Meridian is the store-and-forward relay that carries event traffic between the two data halls when the primary link is congested or down. It is a single physical unit with a cold spare on the shelf next to it, and it is the only piece of hardware in this memo that is tracked by serial number rather than by hostname, because it has been re-addressed three times and the hostname has been wrong on every inventory export since the second move. The serial number of the Meridian relay is QX-4471-B, and that serial is what the vendor support contract, the asset register and the firmware advisories all key on, so a support ticket raised with the hostname instead of the serial will be bounced back by the vendor. The relay's firmware is two releases behind the vendor's current line. The upgrade has been deferred twice because the maintenance window it needs overlaps the Orrery nightly window, and the two teams have now agreed on a window in the second month of next quarter during which Orrery will run a reduced schedule. The cold spare has the same firmware as the active unit and must be upgraded in the same window so that a swap does not reintroduce the old release.

6. The Tamarind service

Tamarind is the customer-facing notification service: it takes events from the internal bus, applies each customer's delivery preferences and sends through mail, push and the webhook gateway. It ships on a fixed cadence rather than on demand, and the cadence is organised as a release train that collects every change merged before the cut, runs the full test matrix over a weekend, and deploys region by region on the following Tuesday. The release train for the Tamarind service is called Saffron Lantern, and that name is what appears on the release calendar, the change-advisory agenda and the rollback plans, so a change that misses the Saffron Lantern cut waits for the next train rather than going out on its own. Three trains have shipped this quarter. All three went out on time; one carried a rollback of a single change that failed its canary in the first region, and the rollback was completed within the same deployment window, which is the outcome the train process is designed to produce. The open question for Tamarind is whether the webhook gateway should move to its own train, because its changes are the ones most often rolled back and they hold the rest of the service to the slower canary schedule.

7. Cross-cutting notes

Three things cut across every workstream above. First, change windows: the nightly batch window, the Tamarind deployment window and the hardware maintenance windows have each been scheduled as if the others did not exist, and the Meridian deferrals are the visible cost of that. The coordination desk will publish a single calendar of protected windows next week, and from then on any new window must be booked on that calendar before it is announced to a team. Second, identifiers: every workstream has at least one identifier that other teams must quote exactly, and the recurring failure mode is that someone quotes the thing they remember rather than the thing that is recorded. The identifiers in this memo are the recorded ones; if a ticket, a vendor call or a page uses a different value, the memo wins and the other record is to be corrected. Third, ownership: the items in the next section exist because work was started without a named owner and then drifted. A named owner is one person, not a team alias and not a rotation, and the tracker will reject an item without one from the start of next quarter. None of this is new. It is written down here because the last review spent most of its time reconstructing which value, which window and which person was current, and that time is better spent on the open items themselves.

8. Items needing an owner before the next review

Varuna: confirm the archive copy of the appliance exports and set a decommissioning date. Halvorsen: finish the migration from the shared credential to service accounts and close the audit finding. Orrery: land the two fixes in review and remove the manual re-runs from the runbook. Meridian: schedule the firmware upgrade for both units in the agreed window. Tamarind: decide on the separate train for the webhook gateway. Each item should have a named owner and a target date in the tracker by the end of the month; items without an owner by then will be raised at the review as unassigned, which is the state we are trying to get out of by writing this memo at all.

9. Record keeping

This memo will be re-issued each quarter. Corrections go to the coordination desk, not to the individual sections' authors, so that there is one copy. The previous wiki pages will be marked superseded this week and deleted at the end of next quarter.
"""


# --- helpers (as in Part C) ----------------------------------------------------------------

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return sha256_bytes(f.read())


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


def is_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("out of memory", "outofmemory", "memory limit", "failed to allocate", "allocation", "oom", "insufficient memory"))


def memo_text() -> tuple[str, str]:
    """results/memo.txt is written from MEMO on the first run and read back afterwards."""
    if os.path.isfile(MEMO_TXT):
        with open(MEMO_TXT, encoding="utf-8") as fh:
            return fh.read(), "read from results/memo.txt"
    os.makedirs(RESULTS, exist_ok=True)
    with open(MEMO_TXT, "w", encoding="utf-8") as fh:
        fh.write(MEMO)
    return MEMO, "written to results/memo.txt (first run)"


# --- the experiment -------------------------------------------------------------------------

def run(base: str, adapter: str, out: str) -> int:
    import mlx_lm
    from mlx_lm import load
    from delta_mem_mlx import load_release_adapter, reset, get_states, set_states

    with open(PART_C_JSON, encoding="utf-8") as fh:
        part_c = json.load(fh)
    facts = [{k: f[k] for k in ("name", "statement", "question", "answer")} for f in part_c["facts"]]

    memo, memo_source = memo_text()
    t0 = time.time()
    model, tok = load(base)
    eos = {int(i) for i in (getattr(tok, "eos_token_ids", None) or [tok.eos_token_id])}
    print(f"== persistence (long): {os.path.basename(base)} + {os.path.basename(adapter)}  (load {time.time()-t0:.1f}s) ==")

    # turn 1: the memo as one user message
    memo_rendered = render(tok, memo, False)
    memo_ids = encode(tok, memo_rendered)
    memo_content_ids = [int(i) for i in tok.encode(memo, add_special_tokens=False)]
    print(f"  memo: {memo_source}; {len(memo_content_ids)} content tokens, {len(memo_ids)} ids rendered, "
          f"{len(memo.split())} words, sha256 {sha256_bytes(memo.encode('utf-8'))[:16]}…")
    for f in facts:
        if f["answer"] not in memo:
            raise ValueError(f"{f['name']}: answer {f['answer']!r} does not appear in the memo")

    items = []
    for f in facts:
        t2 = render(tok, f["question"], True)
        ans = [int(i) for i in tok.encode(f["answer"], add_special_tokens=False)]
        if len(ans) < 2:
            raise ValueError(f"{f['name']}: answer {f['answer']!r} is a single token; the protocol needs multi-token answers")
        if f["answer"] in t2 or f["statement"] in t2:
            raise ValueError(f"{f['name']}: turn 2 must not contain the answer or the statement")
        items.append({**f, "turn2_rendered": t2, "turn2_ids": encode(tok, t2),
                      "answer_ids": ans, "answer_tokens": [tok.decode([i]) for i in ans], "first_answer_token": ans[0],
                      "answer_in_memo": f["answer"] in memo, "statement_verbatim_in_memo": f["statement"] in memo,
                      "arms": {}})
        print(f"  {f['name']:<20s} answer {f['answer']!r} → tokens {items[-1]['answer_tokens']}  (turn 2: {len(items[-1]['turn2_ids'])} ids)")

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

    npz, cfg_json = os.path.join(adapter, "delta_mem_adapter_mlx.npz"), os.path.join(adapter, "delta_mem_config.json")
    doc = {
        "kind": "delta-mem-mlx long-context persistence experiment (Part E)",
        "status": "running",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "machine": {"platform": platform.platform(), "python": platform.python_version(),
                    "mlx": mx.__version__, "mlx_lm": mlx_lm.__version__},
        "base_model": base,
        "adapter": {"dir": adapter, "npz_sha256": sha256(npz), "config_sha256": sha256(cfg_json)},
        "this_repo": git_info(ROOT),
        "facts_from": {"path": os.path.relpath(PART_C_JSON, ROOT), "sha256": sha256(PART_C_JSON)},
        "memo": {"path": os.path.relpath(MEMO_TXT, ROOT), "source": memo_source, "sha256": sha256_bytes(memo.encode("utf-8")),
                 "words": len(memo.split()), "content_tokens": len(memo_content_ids), "rendered_ids": len(memo_ids),
                 "rendered_head": memo_rendered[:80]},
        "protocol": {
            "turn1": "one user message holding the memo; chat template, user turn, no assistant header; prefill only, sidecar live, fresh state",
            "turn2": "fresh prompt without the memo: the Part C question verbatim; chat template, user turn + assistant header; fresh KV cache",
            "arms": {"vanilla": "no sidecar attached (run before the adapter is loaded)",
                     "sidecar": "state after the memo prefill carried into turn 2 with get_states/set_states",
                     "reset": "sidecar attached, state reset to zero between the turns"},
            "logprob": "log-softmax at the last position of turn 2, of the first token of the answer tokenised on its own",
            "greedy": f"argmax decoding, max {MAX_NEW_TOKENS} tokens, stop at EOS; hit = answer contained in the output, case-insensitive",
            "fixed": "protocol and memo fixed before the first run; not tuned afterwards",
        },
        "facts": items,
    }

    def write(status: str, **extra):
        doc["status"] = status
        doc.update(extra)
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, ensure_ascii=False)
        print(f"  written: {out}  (status: {status})")

    # (a) vanilla: nothing attached, nothing to carry — the question alone
    t0 = time.time()
    for it in items:
        it["arms"]["vanilla"] = {**measure(it), **decode(it)}
    print(f"  arm (a) vanilla done ({time.time()-t0:.1f}s)")

    # attach the released sidecar (strict load)
    cfg = load_release_adapter(model, adapter)
    doc["protocol"]["sidecar_config"] = {"r": cfg.r, "alpha": cfg.alpha, "layers": len(model._delta_mem_boxes)}
    print(f"  adapter attached: {len(model._delta_mem_boxes)} layers, r={cfg.r}, alpha={cfg.alpha}")

    # turn 1 once: the memo through the sidecar, fresh state
    t0 = time.time()
    reset(model)
    mx.reset_peak_memory()
    try:
        o = model(mx.array([memo_ids])); mx.eval(o)
        states = get_states(model)
        mx.eval(*[s for s in states if s is not None])
    except Exception as e:  # noqa: BLE001
        kind = "oom" if is_oom(e) else "error"
        tb = traceback.format_exc()
        print(f"  turn 1 prefill FAILED ({kind}): {e}")
        write(kind, turn1={"ids": len(memo_ids), "elapsed_s": time.time() - t0, "error": str(e), "error_type": type(e).__name__,
                           "traceback": tb, "peak_memory_bytes": int(mx.get_peak_memory())})
        return 3 if kind == "oom" else 2
    turn1 = {"ids": len(memo_ids), "elapsed_s": time.time() - t0, "peak_memory_bytes": int(mx.get_peak_memory()),
             "state_max_abs": max(float(mx.max(mx.abs(s))) for s in states if s is not None),
             "state_mean_abs": float(sum(float(mx.mean(mx.abs(s))) for s in states if s is not None) / len(states))}
    print(f"  turn 1 (memo, sidecar): {turn1['ids']} ids in {turn1['elapsed_s']:.1f}s, peak {turn1['peak_memory_bytes']/2**30:.2f} GiB, "
          f"max|S| {turn1['state_max_abs']:.4f}")

    t0 = time.time()
    for it in items:
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
        for a in ("vanilla", "sidecar", "reset"):
            r = it["arms"][a]
            cells.append(f"{r['logprob_first']:>7.3f}   {'HIT ' if r['greedy_hit'] else 'miss'}")
        print(f"  {it['name']:<20s} {it['answer']:<16s} " + "   ".join(f"{c:<20s}" for c in cells))
    print("\n  greedy outputs (max 8 tokens):")
    for it in items:
        print(f"  {it['name']:<20s} (a) {it['arms']['vanilla']['greedy_text']!r}  (b) {it['arms']['sidecar']['greedy_text']!r}  (c) {it['arms']['reset']['greedy_text']!r}")

    lp = lambda it, a: it["arms"][a]["logprob_first"]  # noqa: E731
    summary = {
        "n_facts": len(items),
        "b_gt_a": sum(lp(it, "sidecar") > lp(it, "vanilla") for it in items),
        "b_gt_c": sum(lp(it, "sidecar") > lp(it, "reset") for it in items),
        "b_gt_a_and_b_gt_c": sum(lp(it, "sidecar") > lp(it, "vanilla") and lp(it, "sidecar") > lp(it, "reset") for it in items),
        "mean_logprob_first": {a: sum(lp(it, a) for it in items) / len(items) for a in ("vanilla", "sidecar", "reset")},
        "greedy_hits": {a: sum(it["arms"][a]["greedy_hit"] for it in items) for a in ("vanilla", "sidecar", "reset")},
    }
    print(f"\n  (b) > (a): {summary['b_gt_a']}/{len(items)}   (b) > (c): {summary['b_gt_c']}/{len(items)}   "
          f"both: {summary['b_gt_a_and_b_gt_c']}/{len(items)}   "
          f"greedy hits a/b/c: {summary['greedy_hits']['vanilla']}/{summary['greedy_hits']['sidecar']}/{summary['greedy_hits']['reset']}")
    print(f"  mean log-p of the first answer token: a {summary['mean_logprob_first']['vanilla']:.3f}  "
          f"b {summary['mean_logprob_first']['sidecar']:.3f}  c {summary['mean_logprob_first']['reset']:.3f}")
    write("ok", turn1=turn1, summary=summary)
    return 0


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
    if not os.path.isfile(PART_C_JSON):
        print(f"SKIP: need {PART_C_JSON} (Part C) for the five facts")
        return 77
    try:
        return run(args.base, args.adapter, args.out)
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
