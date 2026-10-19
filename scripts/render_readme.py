# -*- coding: utf-8 -*-
"""Render the milestone-2 numbers into README.md and CHANGELOG.md from results/*.json.

  python scripts/render_readme.py            # rewrite the marked regions in README.md and CHANGELOG.md
  python scripts/render_readme.py --diff     # print the unified diffs; write nothing
  python scripts/render_readme.py --check    # exit 1 if either file is stale (publish prep / CI)

Inputs     results/crosscheck.json, results/persistence.json, results/persistence_long.json, results/cost.json
README.md  touches exactly two things: (1) the Status table — the status cell of the milestone-2 row
           (the row starting "| 2 |"; its What cell is kept), and (2) the section between
           <!-- m2:begin --> and <!-- m2:end -->, inserted before "## Why" on the first run.
CHANGELOG.md  the region between the same markers (under the v0.2 heading): the milestone-2 commits,
           subjects read from git, and the findings, one line each.
Every number in both regions is read from the JSON files or from git; the script types none.
Names and links (model repos, the paper) are identifiers and are typed here.
Exit 0 = rendered / up to date, 1 = --check found a stale file, 2 = an input is missing or malformed.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RESULTS = os.path.join(ROOT, "results")
INPUTS = {
    "crosscheck": os.path.join(RESULTS, "crosscheck.json"),
    "persistence": os.path.join(RESULTS, "persistence.json"),
    "persistence_long": os.path.join(RESULTS, "persistence_long.json"),
    "cost": os.path.join(RESULTS, "cost.json"),
}
README = os.path.join(ROOT, "README.md")
CHANGELOG = os.path.join(ROOT, "CHANGELOG.md")
BEGIN, END = "<!-- m2:begin -->", "<!-- m2:end -->"
README_ANCHOR = "## Why"          # on the first run the section goes in just before this heading
M2_COMMITS = ("2990eac", "5747978", "7882c19")

# identifiers, not measurements
AUTHORS = ("declare-lab/delta-mem_qwen3_4b-instruct",
           "https://huggingface.co/declare-lab/delta-mem_qwen3_4b-instruct")
CONVERSION = ("ofthetrees/delta-mem-qwen3-4b-instruct-mlx-adapter",
              "https://huggingface.co/ofthetrees/delta-mem-qwen3-4b-instruct-mlx-adapter")
BASE_MODEL = ("mlx-community/Qwen3-4B-Instruct-2507-4bit",
              "https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit")
PAPER = ("arXiv 2605.12357", "https://arxiv.org/abs/2605.12357")


# --- numbers ----------------------------------------------------------------------------------

def n0(x): return f"{x:,.0f}"
def n1(x): return f"{x:,.1f}"
def n2(x): return f"{x:,.2f}"
def n3(x): return f"{x:.3f}"
def signed(x, nd=2): return f"{x:+.{nd}f}"
def yes(b): return "yes" if b else "no"


def gather(d: dict) -> dict:
    """Every figure the two documents use, pulled from the four files once."""
    xc, pc, pl, co = d["crosscheck"], d["persistence"], d["persistence_long"], d["cost"]
    xs, cs, cost = xc["summary"], pc["summary"], co["summary"]
    cfg = xc["base_model"]["config"]
    prompts = xc["prompts"]
    ls = pl.get("summary")   # absent when the long-context run stopped (status "oom" / "error")
    m = re.search(r"max (\d+) tokens", pc["protocol"]["greedy"])
    write_lens = [len(f["turn1_ids"]) for f in pc["facts"]]
    F = {
        # cross-check vs the independent port
        "xc_n_prompts": len(prompts),
        "xc_prompt_tokens": [p["n_tokens"] for p in prompts],
        "xc_max_dlogit": xs["max_abs_delta_final_logits"],
        "xc_top1": bool(xs["all_top1_match"]),
        "xc_min_cos": xs["min_state_cosine"],
        "xc_max_dS": xs["max_state_abs_delta"],
        "xc_backbone_only": xs["backbone_only_max_abs_delta"],
        "xc_sidecar_effect": max(p["backbone_only"]["sidecar_effect_max_abs_delta_a"] for p in prompts),
        "xc_all_layers_identical": all(v is None for v in xs["first_divergent_layer"].values()),
        "xc_ref_url": xc["reference"]["url"],
        "xc_ref_name": xc["reference"]["url"].split("github.com/")[-1],
        "xc_ref_commit": xc["reference"]["commit"],
        "xc_date": xc["timestamp"][:10],
        "base_layers": cfg["num_hidden_layers"],
        "base_bits": cfg["quantization"]["bits"],
        "base_group": cfg["quantization"]["group_size"],
        "adapter_sha": xc["adapter"]["npz_sha256"],
        # cost
        "chip": co["machine"]["chip"], "ram_gib": co["machine"]["memory_gib"],
        "mlx": co["machine"]["mlx"], "mlx_lm": co["machine"]["mlx_lm"],
        "prompt_tokens": co["protocol"]["prompt_tokens"], "decode_tokens": co["protocol"]["decode_tokens"],
        "repeats": co["protocol"]["repeats"],
        "prefill_v": cost["prefill_tok_s"]["vanilla"], "prefill_s": cost["prefill_tok_s"]["sidecar"],
        "prefill_ratio": cost["prefill_tok_s"]["sidecar_over_vanilla"],
        "decode_v": cost["decode_tok_s"]["vanilla"], "decode_s": cost["decode_tok_s"]["sidecar"],
        "decode_ratio": cost["decode_tok_s"]["sidecar_over_vanilla"],
        "peak_v": cost["peak_memory_gib"]["vanilla"], "peak_s": cost["peak_memory_gib"]["sidecar"],
        "peak_delta_mib": cost["peak_memory_gib"]["sidecar_minus_vanilla_mib"],
        "sc_layers": co["adapter"]["layers"], "sc_r": co["adapter"]["r"], "sc_alpha": co["adapter"]["alpha"],
        "sc_params": co["adapter"]["sidecar_params"],
        "cost_date": co["timestamp"][:10],
        # persistence, short write (Part C)
        "c_n": cs["n_facts"], "c_b_gt_a": cs["b_gt_a"], "c_b_gt_c": cs["b_gt_c"],
        "c_mean": cs["mean_logprob_first"], "c_hits": cs["greedy_hits"],
        "c_write_min": min(write_lens), "c_write_max": max(write_lens),
        "c_max_new": int(m.group(1)) if m else None,
        "c_item0_checked": pc["item0"]["checked_on"], "c_item0_found": bool(pc["item0"]["worked_example_found"]),
        # persistence, long write (Part E)
        "l_status": pl["status"],
        "l_write": pl["memo"]["rendered_ids"], "l_content_tokens": pl["memo"]["content_tokens"],
        "l_words": pl["memo"]["words"],
        "l_state_max": pl.get("turn1", {}).get("state_max_abs"),
        "l_peak_gib": pl.get("turn1", {}).get("peak_memory_bytes", 0) / 2**30,
    }
    F["c_gain"] = F["c_mean"]["sidecar"] - F["c_mean"]["reset"]            # memory contents
    F["c_adapter_effect"] = F["c_mean"]["reset"] - F["c_mean"]["vanilla"]   # adapter attached, memory zero
    if ls is not None:
        F.update({"l_n": ls["n_facts"], "l_b_gt_a": ls["b_gt_a"], "l_b_gt_c": ls["b_gt_c"],
                  "l_mean": ls["mean_logprob_first"], "l_hits": ls["greedy_hits"]})
        F["l_gain"] = F["l_mean"]["sidecar"] - F["l_mean"]["reset"]
        F["l_adapter_effect"] = F["l_mean"]["reset"] - F["l_mean"]["vanilla"]
        if F["l_n"] != F["c_n"]:
            raise ValueError(f"fact counts differ: short {F['c_n']}, long {F['l_n']}")
    F["slowdown"] = 1.0 / F["prefill_ratio"]
    F["decode_loss_pct"] = (1.0 - F["decode_ratio"]) * 100.0
    return F


def commit_info(short: str) -> dict:
    try:
        out = subprocess.run(["git", "-C", ROOT, "log", "-1", "--date=short", "--format=%h%x1f%H%x1f%ad%x1f%s", short],
                             capture_output=True, text=True, check=True).stdout.strip()
        h, full, date, subject = out.split("\x1f")
        return {"short": h, "full": full, "date": date, "subject": subject}
    except Exception:
        return {"short": short, "full": None, "date": None, "subject": "(not in this clone)"}


# --- README -----------------------------------------------------------------------------------

def status_cell(F: dict) -> str:
    """One table cell, one line. Pipes inside it are escaped for the Markdown table."""
    if F["l_status"] == "ok":
        persist = (f"Persistence under two pre-registered protocols ({F['c_write_min']}–{F['c_write_max']}-token and "
                   f"{F['l_write']:,}-token writes): sidecar beats vanilla {F['c_b_gt_a']}/{F['c_n']} and {F['l_b_gt_a']}/{F['l_n']}, "
                   f"beats the reset control {F['c_b_gt_c']}/{F['c_n']} and {F['l_b_gt_c']}/{F['l_n']}, mean memory gain "
                   f"{signed(F['c_gain'])} and {signed(F['l_gain'])} nats — **no recall effect attributable to the memory "
                   f"contents on this backbone.**")
    else:
        persist = (f"Persistence under the short pre-registered protocol ({F['c_write_min']}–{F['c_write_max']}-token write): "
                   f"sidecar beats vanilla {F['c_b_gt_a']}/{F['c_n']}, beats the reset control {F['c_b_gt_c']}/{F['c_n']}, "
                   f"mean memory gain {signed(F['c_gain'])} nats — **no recall effect attributable to the memory contents "
                   f"on this backbone.** The {F['l_write']:,}-token protocol stopped: {F['l_status']}.")
    parts = [
        "◐",
        f"Loads the authors' released sidecar: zero missing / zero unexpected keys, strict per-module "
        f"({F['sc_layers']} layers, {F['sc_params']:,} parameters; `tests/test_release_adapter.py`).",
        f"Matches [{F['xc_ref_name']}]({F['xc_ref_url']}) bit-for-bit on {F['xc_n_prompts']} prompts "
        f"(max \\|Δ logit\\| = {F['xc_max_dlogit']}; `tests/test_crosscheck.py`).",
        f"Cost on this Mac ({F['chip']}): prefill {n0(F['prefill_v'])} → {n0(F['prefill_s'])} tok/s, decode "
        f"{n1(F['decode_v'])} → {n1(F['decode_s'])} tok/s, peak memory {n2(F['peak_v'])} → {n2(F['peak_s'])} GiB "
        f"(vanilla → sidecar; {F['prompt_tokens']:,}-token prompt, {F['decode_tokens']} greedy tokens).",
        persist,
        "**Still open: the CUDA reference run.** Both MLX ports could share the same readout, so their agreement "
        "is not a check against the paper's implementation.",
    ]
    return " ".join(parts)


def m2_section(F: dict) -> str:
    L = []
    L.append("## Milestone 2 — what was measured")
    L.append("")
    L.append("*Rendered by `scripts/render_readme.py` from `results/crosscheck.json`, `results/persistence.json`, "
             "`results/persistence_long.json` and `results/cost.json`; `--check` fails when this section is stale. "
             "Full tables in `docs/milestone2.md`.*")
    L.append("")
    L.append(f"**Loader.** `load_release_adapter()` attaches the sidecar to every layer the `.npz` names and loads it "
             f"strictly, per module: {F['sc_layers']} layers, r = {F['sc_r']}, α = {F['sc_alpha']:g}, "
             f"{F['sc_params']:,} parameters, zero missing and zero unexpected keys (`tests/test_release_adapter.py`).")
    L.append("")
    L.append(f"**Cross-check against an independent MLX port.** The same adapter tensors, the same {F['base_bits']}-bit "
             f"backbone and the same token ids through this repo's hooks and through "
             f"[{F['xc_ref_name']}]({F['xc_ref_url']}) at commit `{F['xc_ref_commit'][:12]}`, which re-implements the "
             f"attention body ({F['xc_date']}, `tests/test_crosscheck.py`):")
    L.append("")
    L.append("| Prompts | max \\|Δ logit\\| | top-1 equal | backbone only, max \\|Δ\\| | min state cosine | max \\|ΔS\\| | sidecar's own effect on the logits |")
    L.append("|---|---:|---|---:|---:|---:|---:|")
    toks = " / ".join(str(t) for t in F["xc_prompt_tokens"])
    L.append(f"| {F['xc_n_prompts']} ({toks} tokens) | {F['xc_max_dlogit']} | {yes(F['xc_top1'])} | {F['xc_backbone_only']} | "
             f"{F['xc_min_cos']:.12f} | {F['xc_max_dS']:.1e} | up to {n1(F['xc_sidecar_effect'])} |")
    L.append("")
    ident = ("Every layer's hidden state is bit-identical between the ports on every prompt."
             if F["xc_all_layers_identical"] else "The ports diverge inside the network; see `results/crosscheck.json`.")
    L.append(f"{ident} The last column shows the comparison is not vacuous: the sidecar moves the logits, and both "
             f"ports move them identically. This says the two ports compute the same function of the released weights. "
             f"It does not say that function is the paper's: **the CUDA reference run is still the open question**, "
             f"and both MLX ports could share the same readout of the release.")
    L.append("")
    L.append(f"**Cost on this Mac** ({F['chip']}, {F['ram_gib']:g} GiB, mlx {F['mlx']}, mlx-lm {F['mlx_lm']}; "
             f"{F['prompt_tokens']:,}-token prompt, {F['decode_tokens']} greedy tokens, median of {F['repeats']} repeats "
             f"after one warm-up; `scripts/cost.py`, {F['cost_date']}):")
    L.append("")
    L.append("| | Prefill tok/s | Decode tok/s | Peak memory |")
    L.append("|---|---:|---:|---:|")
    L.append(f"| vanilla | {n1(F['prefill_v'])} | {n1(F['decode_v'])} | {n2(F['peak_v'])} GiB |")
    L.append(f"| sidecar | {n1(F['prefill_s'])} | {n1(F['decode_s'])} | {n2(F['peak_s'])} GiB |")
    L.append(f"| sidecar / vanilla | ×{n3(F['prefill_ratio'])} | ×{n3(F['decode_ratio'])} | {signed(F['peak_delta_mib'], 0)} MiB |")
    L.append("")
    maxnew = f" within {F['c_max_new']} greedy tokens" if F["c_max_new"] else ""
    L.append(f"**Persistence (two pre-registered protocols, fixed before the first run).** Turn 1 writes the fact(s) "
             f"into the memory through the sidecar. Turn 2 is a fresh prompt without turn 1 that asks for the fact. "
             f"Arms: (a) vanilla, no sidecar; (b) sidecar, memory carried from turn 1; (c) sidecar, memory reset — so "
             f"(c) − (a) is *adapter attached* and (b) − (c) is *memory contents*. Score: log-probability of the correct "
             f"first answer token; greedy hit = the answer appears{maxnew}. {F['c_n']} invented facts.")
    L.append("")
    L.append("| Protocol | Write | (b) > (a) | (b) > (c) | mean log-p a / b / c | adapter effect (c − a) | memory gain (b − c) | greedy hits a / b / c |")
    L.append("|---|---|---:|---:|---|---:|---:|---|")
    cm, ch = F["c_mean"], F["c_hits"]
    L.append(f"| short (`scripts/persistence.py`) | one sentence, {F['c_write_min']}–{F['c_write_max']} tokens | "
             f"{F['c_b_gt_a']}/{F['c_n']} | {F['c_b_gt_c']}/{F['c_n']} | {n2(cm['vanilla'])} / {n2(cm['sidecar'])} / {n2(cm['reset'])} | "
             f"{signed(F['c_adapter_effect'])} nats | {signed(F['c_gain'])} nats | {ch['vanilla']} / {ch['sidecar']} / {ch['reset']} |")
    if F["l_status"] == "ok":
        lm, lh = F["l_mean"], F["l_hits"]
        L.append(f"| long (`scripts/persistence_long.py`) | a {F['l_words']:,}-word memo, {F['l_write']:,} tokens | "
                 f"{F['l_b_gt_a']}/{F['l_n']} | {F['l_b_gt_c']}/{F['l_n']} | {n2(lm['vanilla'])} / {n2(lm['sidecar'])} / {n2(lm['reset'])} | "
                 f"{signed(F['l_adapter_effect'])} nats | {signed(F['l_gain'])} nats | {lh['vanilla']} / {lh['sidecar']} / {lh['reset']} |")
        state = f" The state does move (max \\|S\\| = {n2(F['l_state_max'])} after the memo); it just does not read out as the facts."
        verdict = (f"Reading: attaching the adapter moves the first-token log-probability by about {n1(F['c_adapter_effect'])} nats; "
                   f"carrying the memory adds {signed(F['c_gain'])} nats after one sentence and {signed(F['l_gain'])} nats after "
                   f"the memo, and no arm ever produces the answer. **No recall effect attributable to the memory contents on "
                   f"this backbone.**{state}")
    else:
        L.append(f"| long (`scripts/persistence_long.py`) | a {F['l_words']:,}-word memo, {F['l_write']:,} tokens | — | — | — | — | — | "
                 f"not run: {F['l_status']} |")
        verdict = (f"Reading: attaching the adapter moves the first-token log-probability by about {n1(F['c_adapter_effect'])} nats; "
                   f"carrying the memory adds {signed(F['c_gain'])} nats, and no arm produces the answer. **No recall effect "
                   f"attributable to the memory contents on this backbone.** The long protocol stopped ({F['l_status']}).")
    L.append("")
    L.append(verdict)
    L.append("")
    item0 = ("no worked example" if not F["c_item0_found"] else "a worked example")
    L.append(f"The authors' model card was checked on {F['c_item0_checked']} and carries {item0} to run verbatim.")
    L.append("")
    L.append("**Honest limits.**")
    L.append("")
    L.append(f"- Prefill is ~{F['slowdown']:.1f}× slower with the sidecar ({n0(F['prefill_s'])} vs {n0(F['prefill_v'])} tok/s) "
             f"because the delta rule is a Python loop over tokens in every layer; decode loses {F['decode_loss_pct']:.0f}%. "
             f"The lower peak memory with the sidecar ({n2(F['peak_s'])} vs {n2(F['peak_v'])} GiB) is an MLX scheduling "
             f"artefact, not a saving.")
    L.append(f"- The persistence numbers are {F['c_n']} facts on one backbone under two protocols. They rule out a large "
             f"effect; they do not measure a small one.")
    L.append("")
    L.append("### Weights and credit")
    L.append("")
    L.append(f"- **Sidecar weights**: [{AUTHORS[0]}]({AUTHORS[1]}) — the authors' trained δ-mem adapter for "
             f"Qwen3-4B-Instruct-2507 ([{PAPER[0]}]({PAPER[1]})). Not redistributed here.")
    L.append(f"- **MLX conversion**: [{CONVERSION[0]}]({CONVERSION[1]}) by ofthetrees (CC-BY-4.0) — the `.npz` this repo "
             f"loads; sha256 `{F['adapter_sha'][:16]}…`.")
    L.append(f"- **Cross-check reference**: [{F['xc_ref_name']}]({F['xc_ref_url']}) at commit `{F['xc_ref_commit']}` — "
             f"an independent MLX port, the comparison port in `tests/test_crosscheck.py`.")
    L.append(f"- **Base model**: [{BASE_MODEL[0]}]({BASE_MODEL[1]}) — Qwen3-4B-Instruct-2507, {F['base_layers']} layers, "
             f"{F['base_bits']}-bit, group size {F['base_group']}.")
    return "\n".join(L)


STATUS_ROW = re.compile(r"^\| 2 \| (?P<what>[^|\n]*?) \| (?P<status>.*) \|$", re.M)


def replace_region(text: str, body: str, insert_before: str | None, label: str) -> str:
    b, e = text.find(BEGIN), text.find(END)
    block = f"{BEGIN}\n{body.rstrip()}\n{END}"
    if b != -1 and e != -1 and e > b:
        return text[:b] + block + text[e + len(END):]
    if b != -1 or e != -1:
        raise ValueError(f"{label}: one m2 marker without the other")
    if insert_before is None:
        raise ValueError(f"{label}: markers {BEGIN} … {END} not found")
    i = text.find(insert_before)
    if i == -1:
        raise ValueError(f"{label}: markers absent and anchor {insert_before!r} not found")
    return text[:i] + block + "\n\n" + text[i:]


def render_readme(old: str, F: dict) -> str:
    if "| Milestone |" not in old:
        raise ValueError("README.md: Status table header not found")
    rows = STATUS_ROW.findall(old)
    if len(rows) != 1:
        raise ValueError(f"README.md: expected one milestone-2 row in the Status table, found {len(rows)}")
    new = STATUS_ROW.sub(lambda m: f"| 2 | {m.group('what')} | {status_cell(F)} |", old, count=1)
    return replace_region(new, m2_section(F), README_ANCHOR, "README.md")


# --- CHANGELOG --------------------------------------------------------------------------------

def changelog_region(F: dict) -> str:
    commits = [commit_info(h) for h in M2_COMMITS]
    dates = [c["date"] for c in commits if c["date"]]
    span = f" ({dates[0]} → {dates[-1]})" if dates else ""
    L = [f"Commits{span}:", ""]
    for c in commits:
        L.append(f"- `{c['short']}` {c['subject']}")
    L += ["", "Findings (numbers from `results/*.json`, rendered by `scripts/render_readme.py`):", ""]
    L.append(f"- Loader: the released sidecar loads strictly, per module — {F['sc_layers']} layers, {F['sc_params']:,} parameters, "
             f"zero missing and zero unexpected keys.")
    L.append(f"- Cross-check: bit-identical to {F['xc_ref_name']} (`{F['xc_ref_commit'][:12]}`) on {F['xc_n_prompts']} prompts — "
             f"max |Δ logit| {F['xc_max_dlogit']}, top-1 equal: {yes(F['xc_top1'])}, "
             f"{'every' if F['xc_all_layers_identical'] else 'not every'} layer's hidden state identical.")
    L.append(f"- Cost ({F['chip']}, {F['prompt_tokens']:,}-token prompt): prefill {n0(F['prefill_v'])} → {n0(F['prefill_s'])} tok/s "
             f"(×{n3(F['prefill_ratio'])}), decode {n1(F['decode_v'])} → {n1(F['decode_s'])} tok/s (×{n3(F['decode_ratio'])}), "
             f"peak memory {n2(F['peak_v'])} → {n2(F['peak_s'])} GiB (vanilla → sidecar).")
    ch = F["c_hits"]
    L.append(f"- Persistence, short write ({F['c_write_min']}–{F['c_write_max']} tokens): sidecar > vanilla {F['c_b_gt_a']}/{F['c_n']}, "
             f"sidecar > reset {F['c_b_gt_c']}/{F['c_n']}, memory gain {signed(F['c_gain'])} nats, greedy hits "
             f"{ch['vanilla']}/{ch['sidecar']}/{ch['reset']}.")
    if F["l_status"] == "ok":
        lh = F["l_hits"]
        L.append(f"- Persistence, long write ({F['l_write']:,}-token memo): sidecar > vanilla {F['l_b_gt_a']}/{F['l_n']}, "
                 f"sidecar > reset {F['l_b_gt_c']}/{F['l_n']}, memory gain {signed(F['l_gain'])} nats, greedy hits "
                 f"{lh['vanilla']}/{lh['sidecar']}/{lh['reset']}.")
    else:
        L.append(f"- Persistence, long write ({F['l_write']:,}-token memo): not run — {F['l_status']}.")
    L.append("- Conclusion: no recall effect attributable to the memory contents on this backbone. The CUDA reference run is "
             "still the open question, and both MLX ports could share the same readout.")
    return "\n".join(L)


def render_changelog(old: str, F: dict) -> str:
    return replace_region(old, changelog_region(F), None, "CHANGELOG.md")


# --- main -------------------------------------------------------------------------------------

def unified(old: str, new: str, path: str) -> str:
    rel = os.path.relpath(path, ROOT)
    return "".join(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
                                        fromfile=f"a/{rel}", tofile=f"b/{rel}"))


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--diff", action="store_true", help="print the unified diffs; write nothing")
    g.add_argument("--check", action="store_true", help="exit 1 if README.md or CHANGELOG.md is stale")
    args = ap.parse_args()

    missing = [p for p in INPUTS.values() if not os.path.isfile(p)]
    if missing:
        print("missing inputs:\n  " + "\n  ".join(os.path.relpath(p, ROOT) for p in missing))
        return 2
    try:
        data = {}
        for k, p in INPUTS.items():
            with open(p, encoding="utf-8") as fh:
                data[k] = json.load(fh)
        F = gather(data)
    except (KeyError, ValueError, TypeError) as e:
        print(f"malformed input: {type(e).__name__}: {e}")
        return 2

    stale = False
    for path, render in ((README, render_readme), (CHANGELOG, render_changelog)):
        rel = os.path.relpath(path, ROOT)
        if not os.path.isfile(path):
            print(f"missing: {rel} (README.md must exist; CHANGELOG.md needs the {BEGIN} … {END} markers)")
            return 2
        with open(path, encoding="utf-8") as fh:
            old = fh.read()
        try:
            new = render(old, F)
        except ValueError as e:
            print(f"cannot render: {e}")
            return 2
        if new == old:
            print(f"up to date: {rel}")
            continue
        stale = True
        if args.diff or args.check:
            print(unified(old, new, path), end="")
        else:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(new)
            print(f"written: {rel}")
    if args.check:
        print("STALE — run scripts/render_readme.py" if stale else "OK — README.md and CHANGELOG.md match results/*.json")
        return 1 if stale else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
