"""
nets.py - 内存优化的KAN实现
"""

from __future__ import annotations
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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

# ------------------------- 内存优化的KAN实现 -------------------------

class EfficientKANLinear(nn.Module):
    """
    内存优化的KAN线性层
    原理：将样条函数分解为可学习的基函数组合
    """
    def __init__(self, in_features: int, out_features: int,
                 grid_size: int = 5, spline_order: int = 3,
                 scale_base: float = 1.0, scale_spline: float = 1.0,
                 base_activation: nn.Module = nn.SiLU,
                 grid_range: tuple = (-1, 1)):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.scale_base = scale_base
        self.scale_spline = scale_spline

        # 基函数部分 - 标准线性层
        self.base_linear = nn.Linear(in_features, out_features)

        # 样条部分 - 使用更小的中间维度
        # 将样条函数分解为两步线性变换，避免大张量
        self.spline_hidden = max(32, min(128, in_features // 2))  # 自适应隐藏维度

        # 第一步：输入特征 -> 样条基函数空间
        self.spline_linear1 = nn.Linear(in_features, self.spline_hidden * (grid_size + spline_order))

        # 第二步：样条基函数空间 -> 输出
        self.spline_linear2 = nn.Linear(self.spline_hidden * (grid_size + spline_order), out_features)

        # 激活函数
        self.activation = base_activation() if isinstance(base_activation, type) else base_activation

        # 归一化层，帮助稳定训练
        self.norm = nn.LayerNorm(in_features) if in_features > 1 else nn.Identity()

        # 初始化
        self.reset_parameters()

    def reset_parameters(self):
        # 基函数部分使用标准初始化
        nn.init.kaiming_uniform_(self.base_linear.weight, a=math.sqrt(5))
        if self.base_linear.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.base_linear.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.base_linear.bias, -bound, bound)

        # 样条部分使用较小的初始化
        nn.init.normal_(self.spline_linear1.weight, mean=0.0, std=0.1 / math.sqrt(self.in_features))
        nn.init.zeros_(self.spline_linear1.bias)

        nn.init.normal_(self.spline_linear2.weight, mean=0.0, std=0.1 / math.sqrt(self.spline_hidden * (self.grid_size + self.spline_order)))
        nn.init.zeros_(self.spline_linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入归一化
        x = self.norm(x)

        # 基函数部分
        base_out = self.base_linear(x)

        # 样条部分
        # 1. 将输入归一化到[-1, 1]范围（KAN的标准做法）
        x_norm = torch.tanh(x)

        # 2. 计算样条特征
        # 这里不使用显式的B样条基函数，而是让网络学习基函数的组合
        spline_features = self.spline_linear1(x_norm)
        spline_features = self.activation(spline_features)
        spline_out = self.spline_linear2(spline_features)

        # 3. 结合两部分
        return self.scale_base * base_out + self.scale_spline * spline_out


class EfficientKAN(nn.Module):
    """
    内存高效的多层KAN网络
    使用分解式设计避免大张量
    """
    def __init__(self, layers: list,
                 grid_size: int = 5,
                 spline_order: int = 3,
                 scale_base: float = 1.0,
                 scale_spline: float = 1.0,
                 base_activation: nn.Module = nn.SiLU):
        super().__init__()
        self.layers = nn.ModuleList()

        # 构建KAN层
        for i in range(len(layers) - 1):
            self.layers.append(
                EfficientKANLinear(
                    layers[i],
                    layers[i+1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation
                )
            )

        # 激活函数（最后一层不使用）
        self.activation = base_activation() if isinstance(base_activation, type) else base_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 通过所有层
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # 最后一层不加激活
                x = self.activation(x)
        return x


class _FCNetCore(nn.Module):
    """
    KAN core with separate Branch/Trunk and a linear readout on their
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
        grid_size: int = 5,
        spline_order: int = 3,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        out_dim: int = 1,
    ) -> None:
        super().__init__()

        # 构建Branch KAN网络
        branch_layers = [branch_in]
        for _ in range(depth_branch - 1):
            branch_layers.append(width_branch)
        branch_layers.append(basis_dim)

        self.branch = EfficientKAN(
            branch_layers,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_base=scale_base,
            scale_spline=scale_spline
        )

        # 构建Trunk KAN网络
        trunk_layers = [trunk_in]
        for _ in range(depth_trunk - 1):
            trunk_layers.append(width_trunk)
        trunk_layers.append(basis_dim)

        self.trunk = EfficientKAN(
            trunk_layers,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_base=scale_base,
            scale_spline=scale_spline
        )

        # 保持原有的readout层
        self.readout = nn.Linear(1, out_dim)

    def forward(self, branch_x: torch.Tensor, trunk_x: torch.Tensor) -> torch.Tensor:
        B = self.branch(branch_x)           # (..., basis_dim)
        T = self.trunk(trunk_x)             # (..., basis_dim)
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
        grid_size: int = 5,
        spline_order: int = 3,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
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
            grid_size=grid_size,
            spline_order=spline_order,
            scale_base=scale_base,
            scale_spline=scale_spline,
            out_dim=out_dim,
        )

    def forward(self, xi1: torch.Tensor, xi2: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        branch_x = tau.unsqueeze(-1)                 # (...,1)
        trunk_x = torch.stack([xi1, xi2, tau], dim=-1)  # (...,3)
        return self.core(branch_x, trunk_x)


# 可选：简化的KAN实现，更接近原始论文但内存更友好
class SimplifiedKAN(nn.Module):
    """
    简化的KAN实现，使用线性层+可学习的基函数
    这个版本更接近原始KAN思想但更容易训练
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64,
                 num_basis: int = 8, activation: nn.Module = nn.SiLU):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_basis = num_basis

        # 基函数网络：学习输入到基函数的映射
        self.basis_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, num_basis * out_dim)
        )

        # 系数网络：学习基函数的系数
        self.coeff_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, num_basis * out_dim)
        )

        # 线性项
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # 计算基函数和系数
        basis = self.basis_net(x).view(batch_size, self.out_dim, self.num_basis)  # (B, out, num_basis)
        coeff = self.coeff_net(x).view(batch_size, self.out_dim, self.num_basis)  # (B, out, num_basis)

        # 线性部分
        linear_out = self.linear(x)  # (B, out)

        # 非线性部分：基函数与系数的加权和
        nonlinear_out = (basis * coeff).sum(dim=-1)  # (B, out)

        return linear_out + nonlinear_out


class FCNetSimpleKAN(nn.Module):
    """使用简化KAN的FCNet版本"""
    def __init__(
        self,
        basis_dim: int = 128,
        width_branch: int = 128,  # 注意：KAN可以用更小的宽度
        depth_branch: int = 4,
        width_trunk: int = 128,
        depth_trunk: int = 4,
        hidden_dim: int = 64,
        num_basis: int = 8,
        out_dim: int = 1,
    ):
        super().__init__()

        # 构建Branch网络
        branch_layers = [1] + [width_branch] * (depth_branch - 1) + [basis_dim]
        self.branch_layers = nn.ModuleList()
        for i in range(len(branch_layers) - 1):
            self.branch_layers.append(
                SimplifiedKAN(branch_layers[i], branch_layers[i+1], hidden_dim, num_basis)
            )

        # 构建Trunk网络
        trunk_layers = [3] + [width_trunk] * (depth_trunk - 1) + [basis_dim]
        self.trunk_layers = nn.ModuleList()
        for i in range(len(trunk_layers) - 1):
            self.trunk_layers.append(
                SimplifiedKAN(trunk_layers[i], trunk_layers[i+1], hidden_dim, num_basis)
            )

        # readout层
        self.readout = nn.Linear(1, out_dim)

    def forward(self, xi1: torch.Tensor, xi2: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        # Branch前向
        branch_x = tau.unsqueeze(-1)
        x = branch_x
        for layer in self.branch_layers:
            x = layer(x)
        B = x

        # Trunk前向
        trunk_x = torch.stack([xi1, xi2, tau], dim=-1)
        x = trunk_x
        for layer in self.trunk_layers:
            x = layer(x)
        T = x

        # 点积和readout
        z = (B * T).sum(dim=-1, keepdim=True)
        return self.readout(z).squeeze(-1)

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
        # 使用高效的KAN实现
        return FCNet(
            basis_dim=kw.get("basis_dim", 128),
            width_branch=kw.get("width_branch", 256),
            depth_branch=kw.get("depth_branch", 4),
            width_trunk=kw.get("width_trunk", 256),
            depth_trunk=kw.get("depth_trunk", 4),
            grid_size=kw.get("grid_size", 5),
            spline_order=kw.get("spline_order", 3),
            scale_base=kw.get("scale_base", 1.0),
            scale_spline=kw.get("scale_spline", 1.0),
            out_dim=1,
        )
    if name in {"simplekan", "skan"}:
        # 使用简化的KAN实现
        return FCNetSimpleKAN(
            basis_dim=kw.get("basis_dim", 128),
            width_branch=kw.get("width_branch", 128),
            depth_branch=kw.get("depth_branch", 4),
            width_trunk=kw.get("width_trunk", 128),
            depth_trunk=kw.get("depth_trunk", 4),
            hidden_dim=kw.get("hidden_dim", 64),
            num_basis=kw.get("num_basis", 8),
            out_dim=1,
        )
    raise ValueError(f"Unknown model: {name}")