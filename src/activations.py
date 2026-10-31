"""
Unified Activation Function Library (Fused C++ Kernel & Zero-Branching Formulation)
Authors: Preecha Noiumkar and Thawatchai Chomsiri
Affiliation: Faculty of Informatics, Mahasarakham University
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F


class RectLUShifted(nn.Module):
    """Shifted RectLU (D0 discontinuity at x = L + W unless L = -W/2)."""
    def __init__(self, L: float = 0.0, W: float = 2.0):
        super().__init__()
        self.L = float(L)
        self.W = max(float(W), 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L, W = self.L, self.W
        quad = x * (x - L) / W
        lin = x - L - 0.5 * W
        return torch.where(x < L, 0.0, torch.where(x < L + W, quad, lin))


class RectLUContinuous(nn.Module):
    """
    Continuous RectLU (C0 continuous everywhere, slope jump 2 -> 1 at x = L + W):
    Using u = ReLU(x - L) and c = clamp(u, max=W):
      f(x) = u - c + (c^2) / W
    """
    def __init__(self, L: float = 0.0, W: float = 2.0):
        super().__init__()
        self.L = float(L)
        self.W = max(float(W), 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = F.relu(x - self.L)
        c = torch.clamp(u, max=self.W)
        return u - c + (c * c) * (1.0 / self.W)


class SmoothRectLU(nn.Module):
    """
    C1-Continuous Smooth RectLU (Fused Zero-Branching Identity):
    For u = ReLU(x - L) and c = clamp(u, max=W):
      f_S(x; L, W) = u - c + c^2 / (2W)
    Exact match to piecewise definition with minimal VRAM footprint.
    """
    def __init__(self, L: float = 0.0, W: float = 2.0):
        super().__init__()
        self.L = float(L)
        self.W = max(float(W), 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = F.relu(x - self.L)
        c = torch.clamp(u, max=self.W)
        return u - c + (c * c) * (0.5 / self.W)


class SmeLU(SmoothRectLU):
    """Symmetric Smooth ReLU (Shamir et al., 2020) centered on [-beta, +beta]."""
    def __init__(self, beta: float = 1.0):
        super().__init__(L=-float(beta), W=2.0 * float(beta))
        self.beta = float(beta)


class ASRectLU(nn.Module):
    """Adaptive Smooth RectLU (AS-RectLU) with trainable L and W."""
    def __init__(self, init_L: float = 0.0, init_W: float = 2.0):
        super().__init__()
        self.L = nn.Parameter(torch.tensor(float(init_L)))
        inv_sp = torch.log(torch.expm1(torch.tensor(max(float(init_W), 1e-4))))
        self.raw_W = nn.Parameter(inv_sp)

    @property
    def W(self) -> torch.Tensor:
        return F.softplus(self.raw_W) + 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.W
        u = F.relu(x - self.L)
        c = torch.clamp(u, max=W)
        return u - c + (c * c) / (2.0 * W)


class PerSampleSigmaASRectLU(nn.Module):
    """Per-Sample Root-Mean-Square Scale-Adaptive Smooth RectLU (sigma-AS-RectLU)."""
    def __init__(self, kappa: float = 0.8, symmetric: bool = True, eps: float = 1e-5, detach_scale: bool = True):
        super().__init__()
        self.kappa = float(kappa)
        self.symmetric = bool(symmetric)
        self.eps = float(eps)
        self.detach_scale = bool(detach_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(1, x.ndim))
        rms = torch.sqrt(torch.mean(x * x, dim=dims, keepdim=True) + self.eps)
        if self.detach_scale:
            rms = rms.detach()

        W = self.kappa * rms
        L = -0.5 * W if self.symmetric else 0.0
        u = F.relu(x - L)
        c = torch.clamp(u, max=W)
        return u - c + (c * c) / (2.0 * W)


class AffineFreeLNSmeLU(nn.Module):
    """Explicit Affine-Free Per-Sample LayerNorm Control followed by SmeLU(beta)."""
    def __init__(self, beta: float = 0.4, eps: float = 1e-5):
        super().__init__()
        self.smelu = SmeLU(beta=beta)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(1, x.ndim))
        mean = torch.mean(x, dim=dims, keepdim=True)
        std = torch.sqrt(torch.var(x, dim=dims, unbiased=False, keepdim=True) + self.eps)
        return self.smelu((x - mean) / std)


def get_activation(name: str, **kwargs) -> nn.Module:
    key = name.strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    if key == "relu":
        return nn.ReLU()
    elif key == "leakyrelu":
        return nn.LeakyReLU(negative_slope=kwargs.get("alpha", 0.01))
    elif key == "prelu":
        return nn.PReLU(init=kwargs.get("alpha", 0.25))
    elif key == "elu":
        return nn.ELU(alpha=kwargs.get("alpha", 1.0))
    elif key == "selu":
        return nn.SELU()
    elif key == "gelu":
        return nn.GELU()
    elif key in ("silu", "swish"):
        return nn.SiLU()
    elif key == "mish":
        return nn.Mish()
    elif key == "softplus":
        return nn.Softplus()
    elif key in ("rectlu", "rectlushifted", "d0rectlu"):
        return RectLUShifted(L=kwargs.get("L", 0.0), W=kwargs.get("W", 2.0))
    elif key in ("rectlucont", "rectlucontinuous", "c0rectlu"):
        return RectLUContinuous(L=kwargs.get("L", 0.0), W=kwargs.get("W", 2.0))
    elif key == "smelu":
        beta = kwargs.get("beta", kwargs.get("W", 2.0) / 2.0)
        return SmeLU(beta=beta)
    elif key == "smoothrectlu":
        return SmoothRectLU(L=kwargs.get("L", 0.0), W=kwargs.get("W", 2.0))
    elif key == "asrectlu":
        return ASRectLU(init_L=kwargs.get("L", 0.0), init_W=kwargs.get("W", 2.0))
    elif key in ("symmsigmaasrectlu", "sigmaasrectlu"):
        return PerSampleSigmaASRectLU(kappa=kwargs.get("kappa", 0.8), symmetric=True)
    elif key == "asymmsigmaasrectlu":
        return PerSampleSigmaASRectLU(kappa=kwargs.get("kappa", 0.5), symmetric=False)
    elif key in ("lnnoaffinesmelu", "lnsmelu"):
        return AffineFreeLNSmeLU(beta=kwargs.get("beta", 0.4))
    else:
        raise ValueError(f"Unknown activation configuration: {name}")