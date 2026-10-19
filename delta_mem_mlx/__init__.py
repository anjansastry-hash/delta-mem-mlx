"""delta-mem-mlx — δ-mem (arXiv 2605.12357) as a sidecar on frozen mlx-lm models."""
from .sidecar import (DeltaMemConfig, Sidecar, attach, detach, reset, get_states,
                      set_states, set_gate_scale, sidecar_parameters, save_sidecar,
                      load_sidecar, count_parameters, load_release_adapter, release_weights)

__version__ = "0.2.0"
