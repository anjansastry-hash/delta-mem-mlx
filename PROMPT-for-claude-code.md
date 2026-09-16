# If `tests/test_identity.py --model …` goes RED on the Mac

Paste this into Claude Code, run from the repo root (`cd ~/ror/delta-mem-mlx && claude`):

---

You are in `delta-mem-mlx`, a δ-mem (arXiv 2605.12357) sidecar for mlx-lm. Read
`README.md` and `delta_mem_mlx/sidecar.py` first. The offline suite
(`python tests/test_identity.py`, tiny random Qwen2/Qwen3) is GREEN. The real-model run

    python tests/test_identity.py --model <MODEL>

is RED — the output is in `logs/test_real.log`. Find the cause and fix it with the
smallest change. Constraints:

1. Do not re-implement any attention class. The design is hooks on `q_proj`/`o_proj`
   only; keep it that way.
2. The IDENTITY tests must stay exact (max|Δ| = 0), including on 4-bit quantized
   models (`nn.QuantizedLinear`) and bf16. If a quantized layer makes `_linear_dims`
   or `_param_dtype` wrong, fix those helpers by reading the installed mlx-lm/mlx
   source (`python -c "import mlx.nn as nn, inspect; print(inspect.getsource(nn.QuantizedLinear))"`).
3. Do not download anything. Do not train anything.
4. When the suite is GREEN, run it twice more to confirm it is deterministic, then
   stop and print the last 20 lines of `logs/test_real.log`.

If the failure is in LIVE or SHIFT (Δ = 0 with β free), the likely cause is the sidecar
init being too small for bf16 to register: raise `init_scale` in `attach()` to 0.05 and
re-run; do not change the tests' tolerances.
