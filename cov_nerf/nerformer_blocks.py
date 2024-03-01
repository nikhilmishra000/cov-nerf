import torch
import torch.nn as nn

from layers.attention import BufferAttend1d
from layers.mlp import MLP
from layers.nonlinearity import LeakyReluArgs
from torch_utils import unflatten_dim


class RayTransformerBlock(nn.Module):
    def __init__(self, c: int, kdim: int, vdim: int, norm: bool = False):
        super().__init__()
        if norm:
            self.norm1 = nn.LayerNorm(c)
        else:
            self.norm1 = nn.Identity()
        self.attend_seq = BufferAttend1d(c, kdim, vdim)
        self.proj_seq = MLP(c + vdim, [c, c], nonlin_args=LeakyReluArgs())
        if norm:
            self.norm2 = nn.LayerNorm(c)
        else:
            self.norm2 = nn.Identity()
        self.out = MLP(c, [c, c], nonlin_args=LeakyReluArgs())

    def forward(self, x):
        # shape (n_rays, n_samples, c)
        assert x.ndim == 3
        m = torch.eye(x.size(-2), device=x.device, dtype=torch.bool).logical_not()
        xx = self.norm1(x)
        y = self.attend_seq(xx, xx, mask=m)
        x = x + self.proj_seq(torch.cat([x, y], dim=-1))
        x = self.out(self.norm2(x))
        return x


class NerformerBlock(nn.Module):
    def __init__(self, c: int, kdim: int, vdim: int, norm: bool = False):
        super().__init__()
        if norm:
            self.norm1 = nn.LayerNorm(c)
        else:
            self.norm1 = nn.Identity()
        self.attend_seq = BufferAttend1d(c, kdim, vdim)
        self.proj_seq = MLP(c + vdim, [c, c], nonlin_args=LeakyReluArgs())

        self.attend_cam = BufferAttend1d(c, kdim, vdim)
        self.proj_cam = MLP(c + vdim, [c, c], nonlin_args=LeakyReluArgs())

        if norm:
            self.norm2 = nn.LayerNorm(c)
        else:
            self.norm2 = nn.Identity()
        self.out = MLP(c, [c, c], nonlin_args=LeakyReluArgs())

    def forward(self, x):
        # shape (n_rays, n_samples, n_cams, c)
        assert x.ndim == 4

        # shape(n_cams * n_rays, n_samples, c)
        xx = x.moveaxis(2, 0).flatten(0, 1)
        m = torch.eye(xx.size(-2), device=x.device, dtype=torch.bool).logical_not()
        xx_normed = self.norm1(xx)
        yy = self.attend_seq(xx_normed, xx_normed, mask=m)
        xx = xx + self.proj_seq(torch.cat([xx, yy], dim=-1))
        x = unflatten_dim(xx, dim=0, shape=[x.size(2), x.size(0)]).moveaxis(0, 2)

        # shape(n_samples * n_rays,n_cams c)
        xx = x.moveaxis(1, 0).flatten(0, 1)
        m = torch.eye(xx.size(-2), device=x.device, dtype=torch.bool).logical_not()
        yy = self.attend_seq(xx, xx, mask=m)
        xx = xx + self.proj_cam(torch.cat([xx, yy], dim=-1))
        x = unflatten_dim(xx, dim=0, shape=[x.size(1), x.size(0)]).moveaxis(0, 1)

        x = self.out(self.norm2(x))
        return x


class CollapseCamsBlock(nn.Module):
    def __init__(self, c, kdim, vdim):
        super().__init__()
        self.attend_seq = BufferAttend1d(c, kdim, vdim)
        self.proj_seq = nn.Linear(vdim, c)

    def forward(self, x):
        # shape (n_rays, n_samples, c)
        assert x.ndim == 3
        m = torch.eye(x.size(-2), device=x.device, dtype=torch.bool).logical_not()
        y = self.attend_seq(x, x, mask=m)
        x = x + self.proj_seq(y)
        return x
