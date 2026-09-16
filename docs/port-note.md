# δ-mem on MLX — a port note

What it would take to run δ-mem (arXiv 2605.12357, declare-lab/delta-Mem) on Apple
Silicon. Companion to `ttt-on-mlx.md`, `tao-on-mlx.md` and `ror-on-mlx.md`; facts about the
method are from `delta-mem-101.md` (✅ = read from the paper or repo; ⚠️ = secondary or
unverified). Written 2026-09-15.

**Status: not run — and not runnable today.** The reference release is PyTorch +
FlashAttention + DeepSpeed, CUDA bf16 only ✅. This is a *port* note: the mechanism is
small enough that the port is a few hundred lines; the blocker is not code, it is the
trained sidecar weights, which exist only for Qwen3 backbones in the authors' format.
Nothing built from this note is δ-mem; it would be **δ-mem-style**, and the numbers in
the talk stay theirs, not ours.

---

## 0. The one thing that decides the architecture

δ-mem is the one method of the four that **changes the forward pass**. TTT, TAO and RoR
all leave the model's computation alone and change what goes into it (context) or what
its weights are (a LoRA). δ-mem inserts a read and a write *inside every attention
layer, on every token*. So it cannot be bolted on from outside with `mlx-lm`'s public
API the way the other three were; it needs a modified model class.

The second consequence: the thing that learns is not the backbone and not a LoRA — it
is a per-layer **state**, an 8×8 matrix, rewritten as tokens stream past. The trained
part (the sidecar projections) is fixed after one SFT run. At inference nothing is
optimised; the delta rule *is* one gradient step, applied to the state, per token,
without an optimiser. That is why it sits at "per token · quarantined · a hunch" on the
ledger, and why it needs no training mutex, no adapter file, no rollback.

---

## 1. What we already have that carries over

| Existing piece | What it gives a δ-mem port |
|---|---|
| `mlx-lm` Qwen3 model class (`mlx_lm/models/qwen3.py`) | The attention block to subclass; RoPE, GQA and KV cache already correct. |
| `mlx_lm.lora` trainer (`_run_train` wrapper in `arm_runtime.py`) | The SFT loop for the sidecar, *if* the trainer can be pointed at a custom parameter set instead of LoRA layers (see §2.6). |
| `mlx_restricted_logits` / `evaluate_sealed` | Scoring a δ-mem-style model on the same sealed pools as the other arms — the only way the comparison means anything. |
| The Warsaw demo's alert stream | A within-interaction memory task: does a card-less δ-mem model classify 7702 correctly *after reading* the 6021 correction earlier in the same stream? That is the experiment the talk implies. |
| `mx.clear_cache()` discipline | Still applies; the state is tiny, the KV cache is not. |

**Genuinely new:** the state tensor and its update; six projections per layer; the
steering hooks into the attention query and output; a training recipe that writes the
context into the state and withholds it from the backbone.

---

## 2. Modules

### 2.1 `MemoryState` — one 8×8 per layer

`S: (n_layers, 8, 8)` in fp32 (the update is a rank-one correction; keep it out of
bf16). MSW variant: `(n_layers, 4, 8, 8)`. Lives beside the KV cache in the generation
loop; reset at the start of an interaction; optionally serialised to disk between
sessions — which the paper does not evaluate ✅, so cross-session persistence is an
engineering claim, not a result. On stage: "lasts the shift".

### 2.2 `Sidecar` — the trained projections

Per layer: `W_q^m, W_k^m, W_v^m: d_model → 8`, `W_β: d_model → 8` (+ bias), `W_λ` (or
λ = 1 − β, per the paper's gated form ✅), and the correction map `W_a^Δ` that lifts the
8-dim readout to two low-rank deltas (Δq on the attention query, Δo on the attention
output) with LoRA-style scaling α/r, α = 16, r = 8 ✅. Total 4.87M parameters for
Qwen3-4B ✅. Plain `nn.Linear`s in MLX; nothing exotic.

### 2.3 `Read` — steer, don't inject

Per token: `q_m = W_q^m x`; `r = S q_m`; `Δq, Δo = W_a^Δ(r)`; `query += Δq` before
attention, `out += Δo` after. The memory never emits text and never enters the KV
cache — it changes *where attention looks and what it emits* ✅. In MLX this is two
adds inside `Attention.__call__`; the rest of the block is untouched.

### 2.4 `Write` — the gated delta rule

`k = W_k^m x; v = W_v^m x; β = σ(W_β x + b); λ = 1 − β`
`S ← Diag(λ) S + Diag(β) (v − S k) kᵀ` ✅

Error-correcting: predict what the state would return for this key, take the error
against the real value, correct the state by it, forget a little. Sixty-four numbers per
layer, one outer product, no optimiser. Three schedules ✅: TSW (every token), SSW (one
write per message segment, on averaged hidden states), MSW (four parallel sub-states).
For a port, TSW first — it is the simplest and the one whose cross-backbone numbers are
published ✅.

Where it goes in the loop: after the layer's attention has consumed the *previous*
state's readout, write with this token's k, v — the order matters and is the first thing
a port gets wrong.

### 2.5 `Backbone` — frozen, and blind to the written context

The backbone parameters are frozen at train and test ✅. The generation loop is the
standard one with two extra tensors threaded through; prefill needs the write applied
token by token (or per segment), which means the fused prefill path must be unrolled
for the memory — the one place a port costs real speed. The paper reports decoding
slower than vanilla, memory flat ✅; expect the same shape on MLX.

### 2.6 `Train` — one SFT run, context withheld from the backbone

The recipe ✅: write the context into the state (backbone frozen, no gradient through
the backbone's weights, but gradient *through* the state update into the sidecar), then
run the query/response through the frozen backbone with the steering active, and
minimise the response's SFT loss. One epoch on the shortest 2,219-sample split of QASPER,
max sequence length 8,269 tokens ✅, on 8× A800 with ZeRO-2 ⚠️.

On MLX: `mlx_lm.lora` cannot be pointed at the sidecar as is; the trainer assumes LoRA
layers. Two routes — (a) a custom loop with `mx.value_and_grad` over the sidecar
parameters only, `nn.utils.tree_flatten` to select them; (b) register the sidecar as the
"trainable" set inside a subclassed `mlx_lm.tuner` module. Route (a) is a day's work
and keeps the paper's loss exactly. Sequence lengths of ~8K on a 128 GB machine are
fine for a 4B backbone; the wall-clock is the unknown — the authors' run is eight A800s
for one epoch, so a single M5 Max will be hours to a day per epoch, not minutes.

### 2.7 `Evaluate` — the only comparison that matters here

Not MemoryAgentBench — the demo stream. Same base for all arms (this is the catch:
δ-mem's released sidecar is Qwen3-4B; the pilot's base is Qwen2.5-7B; a fair comparison
means either training a Qwen2.5 sidecar or moving the other arms to Qwen3-4B), then: does
a δ-mem-style model, having *read* the engineer's correction earlier in the stream, get
the later alert right with nothing in the prompt? And after a reset? The prediction from
the paper's own numbers ✅ (test-time-learning subtask 26 → 47; long-range understanding
down): yes within the shift, no after it — which is the ledger's "a hunch, not a skill"
made measurable.

---

## 3. The pipeline

```mermaid
flowchart LR
    subgraph TRAIN["ONCE — the sidecar · GPU cluster in the paper · hours-to-days on one Mac"]
        Q[QASPER split<br/>2,219 samples] --> W1[write context<br/>into S]
        W1 --> L[SFT loss on response<br/>backbone frozen] --> SC[(sidecar<br/>4.87M params)]
    end
    subgraph SERVE["EVERY TOKEN — no optimiser, no file, no rollback"]
        X[token x_t] --> R[read: r = S q_m<br/>→ Δq, Δo] --> ATT[frozen attention]
        ATT --> WR[write: S ← λS + β(v − Sk)kᵀ] --> S[(S · 8×8 per layer)]
        S -. next token .-> R
    end
    SC -.-> R
    SC -.-> WR
```

Cadence: the sidecar per model (once); the state per token; nothing per release,
nothing per verified event. δ-mem has no oracle, no gate, no promotion — which is why
RoR sits *above* it on the ledger rather than beside it.

---

## 4. Commands — what exists and what doesn't

Confirm the surface before writing a line:

```bash
python3 -c "import mlx_lm, inspect; from mlx_lm.models import qwen3; print(mlx_lm.__version__); print(inspect.getsourcefile(qwen3))"
python3 -c "from mlx_lm.tuner import trainer; print('trainer ok')"
```

The reference (CUDA only ✅) — for reading, not running here:

```bash
git clone https://github.com/declare-lab/delta-Mem      # PyTorch · FlashAttention · DeepSpeed
# released sidecar: Qwen3-4B-Instruct · CC-BY-4.0 · ~8K written-token ceiling ⚠️
```

A port's first executable milestone, in order: (1) subclass `qwen3.Attention` with the
read/write hooks and a randomly initialised sidecar; assert the vanilla output is
recovered when β ≡ 0; (2) load the authors' Qwen3-4B sidecar weights (convert the
PyTorch state dict; check projection shapes and the α/r scaling) and reproduce one
LoCoMo item; (3) only then consider training a sidecar for a different backbone.
Milestone 2 is the honest test of the port; milestone 3 is a project.

---

## 5. What to measure first

1. **β ≡ 0 recovers vanilla.** Bit-for-bit on a fixed prompt. If not, the hooks are in
   the wrong place.
2. **Authors' sidecar on MLX reproduces one published item.** Same Qwen3-4B, same prompt,
   same answer as the CUDA reference (run once on any rented GPU for the ground truth).
3. **The demo stream.** Correction read at 03:12; alert at 03:22 classified with an empty
   prompt: right or wrong? Then reset the state: right or wrong?
4. Decoding speed relative to vanilla on this Mac.

Item 2 is the port. Item 3 is the talk.

---

## 6. Known hard parts

**The weights, not the code.** The mechanism is ~300 lines; the sidecar is a GPU-cluster
SFT run. Without the authors' weights (Qwen3 only) or a re-run, there is nothing to
port *to*. Any "δ-mem on the pilot's Qwen2.5-7B" claim requires training our own sidecar.

**Prefill.** The write must be applied in token order during prefill; MLX's fused prefill
does not do that. A chunked prefill with the state threaded through is the workaround
and it costs speed on long histories — the ~8K written-token ceiling ⚠️ of the released
adapters caps how long that history is anyway.

**Never merge.** There is nothing to merge; the sidecar *is* the mechanism. The
tutorial's warning that `merge_and_unload()` destroys it ⚠️ applies in spirit: any
attempt to fold the projections into the backbone removes the state path.

**Memory blending.** A fixed 64-cell state per layer must hold everything; facts,
preferences and events compete ⚠️, and long-range understanding fell in the paper's own
table ✅. On stage this is the strongest single sentence about the method's limit, and
it is theirs.

**Cross-session persistence is unproven.** Saving `S` to disk is trivial; whether a
state written under one conversation steers usefully in the next was not evaluated ✅.
Say "lasts the shift"; do not say "persists across sessions".

**Prompt-injection surface.** Anything the model *reads* is written into the state with
no verifier and no gate. A poisoned document steers every later token of the shift. RoR's
oracle is the answer to this; δ-mem has none by design.

**Why it is worth porting at all.** Not to reproduce the paper. To put "read it once,
no card" on the same sealed pools as "one card on the desk" and "one drill" — the
persistence ladder as an experiment on one machine.

---

## Change log

| Date | Change |
|---|---|
| 2026-09-15 | Created as a port note. Mechanism from `delta-mem-101.md`; MLX insertion points; training route; milestones; open problems. Not run. |
