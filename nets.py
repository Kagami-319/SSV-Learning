"""
nets.py
----------------
You can keep extending with new model classes and simply register them in `make_model`.

Available utilities
- set_seed(seed: int)
- device_select(pref: str)

Available models
- FC-Net(factorized coordinate):  y = <Branch(tau), Trunk([xi1,xi2,tau])> through linear readout

Factory
- make_model(name: str, **kwargs) -> nn.Module
    name in {"fcnet-style"}

Notes
- Keep activations and widths aligned with training script defaults.
- To add a new model, implement a class and extend `make_model`.
"""

from __future__ import annotations
import random
import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "set_seed", "device_select",
    "ConcatMLP", "FCNet", "make_model",
]

# ----------------------------- utils (from fc) ----------------------------

def set_seed(seed: int = 42) -> None:
    """Set RNG seeds and enable fast matmul when available."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def device_select(pref: str = "cuda") -> torch.device:
    """Prefer CUDA if available and requested, otherwise CPU."""
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# --------------------------- building blocks (MLP) ---------------------------

def _mlp(din: int, dout: int, width: int, depth: int, act: type[nn.Module] = nn.SiLU) -> nn.Sequential:
    layers = []
    d = din
    for _ in range(depth):
        layers += [nn.Linear(d, width), act()]
        d = width
    layers += [nn.Linear(d, dout)]
    return nn.Sequential(*layers)

# --------------------------------- models -----------------------------------

class ConcatMLP(nn.Module):
    """Baseline coordinate MLP: f([xi1, xi2, tau]) -> scalar."""
    def __init__(self, in_dim: int = 3, width: int = 256, depth: int = 5, out_dim: int = 1):
        super().__init__()
        self.net = _mlp(in_dim, out_dim, width, depth)

    def forward(self, xi1: torch.Tensor, xi2: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        x = torch.stack([xi1, xi2, tau], dim=-1)  # (..., 3)
        return self.net(x).squeeze(-1)


class _FCNetCore(nn.Module):
    """
    FCNet core with separate Branch/Trunk and a linear readout on their
    elementwise product.

    y = Readout( sum_k Branch_k(branch_x) * Trunk_k(trunk_x) )
    """
    def __init__(
        self,
        branch_in: int = 1,
        trunk_in: int = 3,
        basis_dim: int = 128,
        width_branch: int = 256,
        depth_branch: int = 4,
        width_trunk: int = 256,
        depth_trunk: int = 4,
        out_dim: int = 1,
    ) -> None:
        super().__init__()
        self.branch = _mlp(branch_in, basis_dim, width_branch, depth_branch)
        self.trunk = _mlp(trunk_in, basis_dim, width_trunk, depth_trunk)
        self.readout = nn.Linear(1, out_dim)

    def forward(self, branch_x: torch.Tensor, trunk_x: torch.Tensor) -> torch.Tensor:
        B = self.branch(branch_x)           # (..., K)
        T = self.trunk(trunk_x)             # (..., K)
        z = (B * T).sum(dim=-1, keepdim=True)  # (..., 1)
        return self.readout(z).squeeze(-1)


class FCNet(nn.Module):
    """Convenience wrapper to accept (xi1, xi2, tau) directly."""
    def __init__(
        self,
        basis_dim: int = 128,
        width_branch: int = 256,
        depth_branch: int = 4,
        width_trunk: int = 256,
        depth_trunk: int = 4,
        out_dim: int = 1,
    ) -> None:
        super().__init__()
        self.core = _FCNetCore(
            branch_in=1,
            trunk_in=3,
            basis_dim=basis_dim,
            width_branch=width_branch,
            depth_branch=depth_branch,
            width_trunk=width_trunk,
            depth_trunk=depth_trunk,
            out_dim=out_dim,
        )

    def forward(self, xi1: torch.Tensor, xi2: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        branch_x = tau.unsqueeze(-1)                 # (...,1)
        trunk_x = torch.stack([xi1, xi2, tau], dim=-1)  # (...,3)
        return self.core(branch_x, trunk_x)

# ---------------------------------- factory ---------------------------------

def make_model(name: str = "concat", **kw) -> nn.Module:
    """Create a model by name. Unused kwargs are ignored.

    Parameters
    """
    name = name.lower()
    if name in {"concat", "concatmlp"}:
        return ConcatMLP(
            in_dim=3,
            width=kw.get("width", 256),
            depth=kw.get("depth", 5),
            out_dim=1,
        )
    if name in {"fcnet","fc"}:
        return FCNet(
            basis_dim=kw.get("basis_dim", 128),
            width_branch=kw.get("width_branch", 256),
            depth_branch=kw.get("depth_branch", 4),
            width_trunk=kw.get("width_trunk", 256),
            depth_trunk=kw.get("depth_trunk", 4),
            out_dim=1,
        )
    raise ValueError(f"Unknown model: {name}")
