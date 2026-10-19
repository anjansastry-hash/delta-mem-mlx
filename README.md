# delta-mem-mlx

δ-mem — *"Efficient Online Memory for Large Language Models"* (arXiv 2605.12357,
[declare-lab/delta-Mem](https://github.com/declare-lab/delta-Mem)) — as a **sidecar on a
frozen [mlx-lm](https://github.com/ml-explore/mlx-lm) model**, on Apple Silicon.

An 8×8 state per attention layer, rewritten on every token by a gated delta rule, whose
readout nudges the attention query and output. The backbone never changes. Nothing is
optimised at inference; the delta rule *is* one gradient step, applied to sixty-four
numbers per layer, per token, with no optimiser.

## Status — milestone 2 in progress, not a verified port

| Milestone | What | Status |
|---|---|---|
| 1 | The mechanism on any mlx-lm decoder with `q_proj`/`o_proj` (Qwen2/2.5, Qwen3, …), randomly initialised sidecar, with a test suite that proves the hooks are in the right place: β ≡ 0 recovers the vanilla model **bit-for-bit**, in prefill and in KV-cached decode. | ✅ GREEN on tiny random Qwen2 + Qwen3 (`tests/test_identity.py`, no download, ~10 s) On Qwen2.5-7B-Instruct-4bit (28 layers, 4,817,120 sidecar params, M-series Mac, 2026-09-16): all 8 GREEN (DECODE gap 0.043 vs the vanilla model's own prefill-vs-decode kernel gap of 0.039). |
| 2 | Load the authors' trained Qwen3-4B sidecar (via the community MLX conversion, see below) and reproduce one published item against the CUDA reference. | ◐ Loads the authors' released sidecar: zero missing / zero unexpected keys, strict per-module (36 layers, 4,866,336 parameters; `tests/test_release_adapter.py`). Matches [elimaine/delta-mem-mlx-sidecar](https://github.com/elimaine/delta-mem-mlx-sidecar) bit-for-bit on 3 prompts (max \|Δ logit\| = 0.0; `tests/test_crosscheck.py`). Cost on this Mac (Apple M5 Max): prefill 6,210 → 729 tok/s, decode 158.2 → 112.4 tok/s, peak memory 3.17 → 2.69 GiB (vanilla → sidecar; 1,024-token prompt, 128 greedy tokens). Persistence under two pre-registered protocols (20–23-token and 2,010-token writes): sidecar beats vanilla 5/5 and 5/5, beats the reset control 3/5 and 3/5, mean memory gain +0.11 and +0.14 nats — **no recall effect attributable to the memory contents on this backbone.** **Still open: the CUDA reference run.** Both MLX ports could share the same readout, so their agreement is not a check against the paper's implementation. |
| 3 | Train a sidecar for a different backbone (e.g. Qwen2.5-7B) on MLX. | ⬜ Not started. A GPU-cluster SFT run in the paper; hours-to-a-day per epoch on one Mac. |

**Prior art, read before claiming anything.** This is not the first MLX port.
[elimaine/delta-mem-mlx-sidecar](https://github.com/elimaine/delta-mem-mlx-sidecar)
carries an MLX δ-mem attention wrapper (`mlx_delta_attention.py`) behind an
OpenAI-compatible server, and [ofthetrees/delta-mem-qwen3-4b-instruct-mlx-adapter](https://huggingface.co/ofthetrees/delta-mem-qwen3-4b-instruct-mlx-adapter)
is the authors' Qwen3-4B-Instruct-2507 sidecar converted to an MLX `.npz` (CC-BY-4.0).
This repo's mechanism was checked line-by-line against that wrapper on 2026-09-16 and
matches it (placement of Δq/Δo, read-then-write order, the gated update, tanh+L2 on the
memory q/k); `load_release_adapter()` reads that `.npz` layout directly. What this repo
adds is different in kind: a **model-agnostic hook** (no attention class is re-implemented,
so it survives mlx-lm upgrades and works on Qwen2.5/Llama/Mistral shapes), an
**identity test suite** that any port should pass, and — the reason it exists — the
experiment in §"Why" below.

<!-- m2:begin -->
## Milestone 2 — what was measured

*Rendered by `scripts/render_readme.py` from `results/crosscheck.json`, `results/persistence.json`, `results/persistence_long.json` and `results/cost.json`; `--check` fails when this section is stale. Full tables in `docs/milestone2.md`.*

**Loader.** `load_release_adapter()` attaches the sidecar to every layer the `.npz` names and loads it strictly, per module: 36 layers, r = 8, α = 16, 4,866,336 parameters, zero missing and zero unexpected keys (`tests/test_release_adapter.py`).

**Cross-check against an independent MLX port.** The same adapter tensors, the same 4-bit backbone and the same token ids through this repo's hooks and through [elimaine/delta-mem-mlx-sidecar](https://github.com/elimaine/delta-mem-mlx-sidecar) at commit `c946764a1fd1`, which re-implements the attention body (2026-10-19, `tests/test_crosscheck.py`):

| Prompts | max \|Δ logit\| | top-1 equal | backbone only, max \|Δ\| | min state cosine | max \|ΔS\| | sidecar's own effect on the logits |
|---|---:|---|---:|---:|---:|---:|
| 3 (15 / 48 / 16 tokens) | 0.0 | yes | 0.0 | 1.000000000000 | 1.2e-06 | up to 12.0 |

Every layer's hidden state is bit-identical between the ports on every prompt. The last column shows the comparison is not vacuous: the sidecar moves the logits, and both ports move them identically. This says the two ports compute the same function of the released weights. It does not say that function is the paper's: **the CUDA reference run is still the open question**, and both MLX ports could share the same readout of the release.

**Cost on this Mac** (Apple M5 Max, 128 GiB, mlx 0.32.3, mlx-lm 0.32.0; 1,024-token prompt, 128 greedy tokens, median of 3 repeats after one warm-up; `scripts/cost.py`, 2026-10-19):

| | Prefill tok/s | Decode tok/s | Peak memory |
|---|---:|---:|---:|
| vanilla | 6,209.9 | 158.2 | 3.17 GiB |
| sidecar | 729.2 | 112.4 | 2.69 GiB |
| sidecar / vanilla | ×0.117 | ×0.711 | -494 MiB |

**Persistence (two pre-registered protocols, fixed before the first run).** Turn 1 writes the fact(s) into the memory through the sidecar. Turn 2 is a fresh prompt without turn 1 that asks for the fact. Arms: (a) vanilla, no sidecar; (b) sidecar, memory carried from turn 1; (c) sidecar, memory reset — so (c) − (a) is *adapter attached* and (b) − (c) is *memory contents*. Score: log-probability of the correct first answer token; greedy hit = the answer appears within 8 greedy tokens. 5 invented facts.

| Protocol | Write | (b) > (a) | (b) > (c) | mean log-p a / b / c | adapter effect (c − a) | memory gain (b − c) | greedy hits a / b / c |
|---|---|---:|---:|---|---:|---:|---|
| short (`scripts/persistence.py`) | one sentence, 20–23 tokens | 5/5 | 3/5 | -11.96 / -5.63 / -5.73 | +6.23 nats | +0.11 nats | 0 / 0 / 0 |
| long (`scripts/persistence_long.py`) | a 1,675-word memo, 2,010 tokens | 5/5 | 3/5 | -11.96 / -5.59 / -5.73 | +6.23 nats | +0.14 nats | 0 / 0 / 0 |

Reading: attaching the adapter moves the first-token log-probability by about 6.2 nats; carrying the memory adds +0.11 nats after one sentence and +0.14 nats after the memo, and no arm ever produces the answer. **No recall effect attributable to the memory contents on this backbone.** The state does move (max \|S\| = 5.29 after the memo); it just does not read out as the facts.

The authors' model card was checked on 2026-10-19 and carries no worked example to run verbatim.

**Honest limits.**

- Prefill is ~8.5× slower with the sidecar (729 vs 6,210 tok/s) because the delta rule is a Python loop over tokens in every layer; decode loses 29%. The lower peak memory with the sidecar (2.69 vs 3.17 GiB) is an MLX scheduling artefact, not a saving.
- The persistence numbers are 5 facts on one backbone under two protocols. They rule out a large effect; they do not measure a small one.

### Weights and credit

- **Sidecar weights**: [declare-lab/delta-mem_qwen3_4b-instruct](https://huggingface.co/declare-lab/delta-mem_qwen3_4b-instruct) — the authors' trained δ-mem adapter for Qwen3-4B-Instruct-2507 ([arXiv 2605.12357](https://arxiv.org/abs/2605.12357)). Not redistributed here.
- **MLX conversion**: [ofthetrees/delta-mem-qwen3-4b-instruct-mlx-adapter](https://huggingface.co/ofthetrees/delta-mem-qwen3-4b-instruct-mlx-adapter) by ofthetrees (CC-BY-4.0) — the `.npz` this repo loads; sha256 `563c66de92b19ef4…`.
- **Cross-check reference**: [elimaine/delta-mem-mlx-sidecar](https://github.com/elimaine/delta-mem-mlx-sidecar) at commit `c946764a1fd116f90363f5fa0c60944a1fab4302` — an independent MLX port, the comparison port in `tests/test_crosscheck.py`.
- **Base model**: [mlx-community/Qwen3-4B-Instruct-2507-4bit](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit) — Qwen3-4B-Instruct-2507, 36 layers, 4-bit, group size 64.
<!-- m2:end -->

## Why

To put "read it once, no card in the prompt" on the same sealed evaluation as "one card on
the desk" (exemplars) and "one drill" (a LoRA overlay): the persistence ladder as an
experiment on one machine, with the same base model for every arm. δ-mem is the rung
that learns per token and forgets softly; it can carry the thread of a shift and cannot
acquire a skill. That prediction is the paper's own; this is the harness to test it on an
operations stream rather than a benchmark. See `docs/port-note.md`.

## Install

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install mlx-lm            # Apple Silicon. On Linux: pip install "mlx[cpu]" mlx-lm (CPU, for tests only)
```

## Run the tests (no download, ~10 s)

```bash
python tests/test_identity.py
```

On a real model already on disk (nothing is fetched):

```bash
python tests/test_identity.py --model mlx-community/Qwen2.5-7B-Instruct-4bit
```

Expected: every line GREEN, `max|Δ| = 0` on IDENTITY / RESET / DECODE / RELEASE FORMAT /
DETACH, and `max|Δ| > 0` on LIVE and SHIFT.

## Use

```python
from mlx_lm import load, generate
from delta_mem_mlx import DeltaMemConfig, attach, reset, get_states, set_states

model, tok = load("mlx-community/Qwen2.5-7B-Instruct-4bit")
attach(model, DeltaMemConfig(r=8, alpha=16.0))   # random sidecar: the mechanism, not the paper's memory
reset(model)                                     # start of a shift
generate(model, tok, prompt="...", max_tokens=64)
S = get_states(model)                            # per-layer (B, 8, 8) float32 — the whole memory
reset(model)                                     # end of the shift
```

With the community-converted authors' weights (milestone 2: loads and matches an independent port; not verified against the CUDA reference):

```bash
scripts/fetch.sh    # resumable; safe to re-run on a bad connection
```

```python
from delta_mem_mlx import load_release_adapter
model, tok = load("mlx-community/Qwen3-4B-Instruct-2507-4bit")
load_release_adapter(model, "weights/delta-mem-qwen3-4b-instruct-mlx-adapter")
```

## The mechanism (what `delta_mem_mlx/sidecar.py` does)

For each layer, `q_proj` and `o_proj` are wrapped; the attention class is untouched.

```
read   r_t = S_{t-1} · q_m,      q_m = tanh-L2(W_q^m x_t)
q      q_proj(x_t) + (α/r) · A_q(r_t)                         ← before q_norm / RoPE
write  k = tanh-L2(W_k^m x_t),  v = W_v^m x_t,  β = σ(W_β x_t + b),  λ = 1 − β
       S_t = Diag(λ) S_{t-1} + Diag(β) (v − S_{t-1} k) kᵀ
o      o_proj(a_t) + (α/r) · A_o(r_t)                         ← after o_proj
```

The sidecar and `S` are float32 (the r×r recurrence amplifies bf16 kernel noise; fp32 keeps
prefill and decode equal); `S` is activation state, one per sequence, kept outside the
parameter tree (like a KV cache). `set_gate_scale(model, 0.0)` forces β ≡ 0, which is the identity
test. Sidecar size for Qwen3-4B (36 layers, d = 2560, q_dim = 4096, r = 8): 135,176 per
layer, **4,866,336 total — the paper's 4.87M**, which also rules out an uncoupled λ
projection (5.60M) and a Δq in d_model space (4.42M).

## Layout

```
delta_mem_mlx/sidecar.py   the mechanism, attach/detach/reset, state get/set, .npz loader
tests/test_identity.py     the suite (tiny random models by default; --model for a real one)
scripts/fetch.sh           resumable downloads (base model + community adapter), retry loop
scripts/push.sh            git push with retries, for bad connections
docs/port-note.md          the design note this was built from
PROMPT-for-claude-code.md  what to hand Claude Code on the Mac if the real-model run goes RED
```

## Honest limits

* Milestone 1 proves the hooks are where the mechanism says they are. It proves nothing
  about memory quality — the sidecar is random until milestone 2.
* Not implemented: uncoupled λ, `lambda_outside` / `no_lambda` update modes, SSW/MSW
  write schedules, padding masks for batched prompts.
* Whether a saved `S` steers usefully in a *later* conversation was not evaluated by the
  paper. Say "lasts the shift"; do not say "persists across sessions".

## License

MIT (see LICENSE). The authors' δ-mem weights are CC-BY-4.0 and are not
redistributed here.
