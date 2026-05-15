import numpy as np
import torch

FEATURE_ORDER = ["tau1", "tau2", "tau3", "tau4", "p1", "p2", "p3", "p4", "g1", "g2", "g3", "g4"]
OPTIMIZABLE_FEATURES = ["tau1", "tau2", "tau3", "tau4", "p2", "p3", "p4", "g1", "g2", "g3", "g4"]

BOUNDS = {
    "tau1": (0.5,   10.0),
    "tau2": (0.5,   10.0),
    "tau3": (0.5,   10.0),
    "tau4": (0.5,   10.0),
    "p1":   (1.5,    6.0),  # derived from conservation; listed for reference only
    "p2":   (-2.0,  -0.5),
    "p3":   (-2.0,  -0.5),
    "p4":   (-2.0,  -0.5),
    "g1":   (0.05,   1.00),
    "g2":   (0.05,   1.00),
    "g3":   (0.05,   1.00),
    "g4":   (0.05,   1.00),
}

# Insertion point of p1 in the full 12-feature vector
_P1 = FEATURE_ORDER.index("p1")

# Indices of p2, p3, p4 within OPTIMIZABLE_FEATURES (the 11-feature space)
_P2 = OPTIMIZABLE_FEATURES.index("p2")
_P3 = OPTIMIZABLE_FEATURES.index("p3")
_P4 = OPTIMIZABLE_FEATURES.index("p4")

# Precomputed bound arrays in OPTIMIZABLE_FEATURES order, ready for torch conversion
LOWER = np.array([BOUNDS[f][0] for f in OPTIMIZABLE_FEATURES], dtype=np.float32)
UPPER = np.array([BOUNDS[f][1] for f in OPTIMIZABLE_FEATURES], dtype=np.float32)


def tanh_project(w: torch.Tensor, L: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    return L + (U - L) / 2.0 * (torch.tanh(w) + 1.0)


def arctanh_init(x0: torch.Tensor, L: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    v = 2.0 * (x0 - L) / (U - L) - 1.0
    # clamp away from ±1 to keep arctanh finite at boundary points
    return torch.arctanh(v.clamp(-1.0 + 1e-6, 1.0 - 1e-6))


def enforce_conservation(x_opt: torch.Tensor) -> torch.Tensor:
    # returns new tensor rather than in-place assignment to keep the autograd graph intact
    p1 = -(x_opt[..., _P2] + x_opt[..., _P3] + x_opt[..., _P4])
    return torch.cat([x_opt[..., :_P1], p1.unsqueeze(-1), x_opt[..., _P1:]], dim=-1)
