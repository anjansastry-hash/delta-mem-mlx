# delta-mem-mlx

δ-mem — *"Efficient Online Memory for Large Language Models"* (arXiv 2605.12357,
[declare-lab/delta-Mem](https://github.com/declare-lab/delta-Mem)) — as a **sidecar on a
frozen [mlx-lm](https://github.com/ml-explore/mlx-lm) model**, on Apple Silicon.

An 8×8 state per attention layer, rewritten on every token by a gated delta rule, whose
readout nudges the attention query and output. The backbone never changes. Nothing is
optimised at inference; the delta rule *is* one gradient step, applied to sixty-four
numbers per layer, per token, with no optimiser.

## Status — milestone 1 of 3, not a verified port

| Milestone | What | Status |
|---|---|---|
| 1 | The mechanism on any mlx-lm decoder with `q_proj`/`o_proj` (Qwen2/2.5, Qwen3, …), randomly initialised sidecar, with a test suite that proves the hooks are in the right place: β ≡ 0 recovers the vanilla model **bit-for-bit**, in prefill and in KV-cached decode. | ✅ GREEN on tiny random Qwen2 + Qwen3 (`tests/test_identity.py`, no download, ~10 s) On Qwen2.5-7B-Instruct-4bit (28 layers, 4,817,120 sidecar params, M-series Mac, 2026-09-16): all 8 GREEN (DECODE gap 0.043 vs the vanilla model's own prefill-vs-decode kernel gap of 0.039). |
| 2 | Load the authors' trained Qwen3-4B sidecar (via the community MLX conversion, see below) and reproduce one published item against the CUDA reference. | ⬜ Loader written (`load_release_adapter`), **nothing verified**. Needs the weights and one GPU run for ground truth. |
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

With the community-converted authors' weights (milestone 2, unverified):

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
* Prefill is a python loop over tokens for the 8×8 update (the projections are batched).
  Correct, not fast; a chunked scan is the obvious next step.
* Not implemented: uncoupled λ, `lambda_outside` / `no_lambda` update modes, SSW/MSW
  write schedules, padding masks for batched prompts.
* Whether a saved `S` steers usefully in a *later* conversation was not evaluated by the
  paper. Say "lasts the shift"; do not say "persists across sessions".

## License

MIT (see LICENSE). The authors' δ-mem weights are CC-BY-4.0 and are not
redistributed here.
