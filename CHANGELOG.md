# Changelog

Dates are commit dates. The region between the `m2` markers is rendered by
`scripts/render_readme.py` from `results/*.json` and git; edit the script, not the region.

## v0.2 — milestone 2: the authors' released Qwen3-4B sidecar on MLX

<!-- m2:begin -->
Commits (2026-10-17 → 2026-10-19):

- `2990eac` milestone 2a: release adapter loads on Qwen3-4B-Instruct-2507 (zero missing/unexpected keys); test_release_adapter.py
- `5747978` milestone 2b: strict per-module loader; cross-check vs independent MLX port (bit-identical); short-context persistence protocol (no memory effect beyond adapter)
- `7882c19` milestone 2c: cost on Apple Silicon (results/cost.json); long-context persistence protocol; docs/milestone2.md generated

Findings (numbers from `results/*.json`, rendered by `scripts/render_readme.py`):

- Loader: the released sidecar loads strictly, per module — 36 layers, 4,866,336 parameters, zero missing and zero unexpected keys.
- Cross-check: bit-identical to elimaine/delta-mem-mlx-sidecar (`c946764a1fd1`) on 3 prompts — max |Δ logit| 0.0, top-1 equal: yes, every layer's hidden state identical.
- Cost (Apple M5 Max, 1,024-token prompt): prefill 6,210 → 729 tok/s (×0.117), decode 158.2 → 112.4 tok/s (×0.711), peak memory 3.17 → 2.69 GiB (vanilla → sidecar).
- Persistence, short write (20–23 tokens): sidecar > vanilla 5/5, sidecar > reset 3/5, memory gain +0.11 nats, greedy hits 0/0/0.
- Persistence, long write (2,010-token memo): sidecar > vanilla 5/5, sidecar > reset 3/5, memory gain +0.14 nats, greedy hits 0/0/0.
- Conclusion: no recall effect attributable to the memory contents on this backbone. The CUDA reference run is still the open question, and both MLX ports could share the same readout.
<!-- m2:end -->

## v0.1 — milestone 1: the mechanism and the identity suite

- `4b5ca56` delta-mem-mlx: milestone 1 — sidecar hooks + identity suite, GREEN on Qwen2.5-7B-4bit
- `06fc5a2` README: 7B run GREEN; MIT license; fix push.sh
