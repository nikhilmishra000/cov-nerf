from typing import Optional

import attr
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.conv_layer import ConvLayer, ConvLayerSpec, LayerOrder
from layers.nonlinearity import LeakyReluArgs, Nonlinearity, SigmoidArgs
from layers.normalization import GroupNormArgs
from layers.residual_block import ConvShortcut, Residual, ResidualMode, ShortcutStrategy


class SqueezeExcitation(nn.Module):
    def __init__(
        self, in_channels: int, squeeze_channels: int, conv_spec: ConvLayerSpec
    ):
        super().__init__()
        assert conv_spec.layer_order == LayerOrder.CONV_NORM_NONLIN
        conv_spec = conv_spec.as_ksize(1).as_no_normalization().as_bias(True)
        end_spec = attr.evolve(conv_spec, nonlin_args=SigmoidArgs())

        self._avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Conv, Nonlin, Conv, Sigmoid
        self._f = nn.Sequential(
            ConvLayer(in_channels, squeeze_channels, conv_spec),
            ConvLayer(squeeze_channels, in_channels, end_spec),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x * self._f(self._avg_pool(x))
        return out


class BottleneckTransform(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bottleneck_multiple: int,
        spec: ConvLayerSpec,
        se_ratio: Optional[float] = None,
    ):
        super().__init__()
        projective_spec = spec.as_ksize(1).as_stride(1).as_groups(1)
        assert spec.layer_order == LayerOrder.CONV_NORM_NONLIN
        bottleneck_channels = int(round(out_channels * bottleneck_multiple))
        se: nn.Module
        if se_ratio:
            se_channels = int(round(in_channels * se_ratio))
            se = SqueezeExcitation(bottleneck_channels, se_channels, projective_spec)
        else:
            se = nn.Identity()
        self._core = nn.Sequential(
            ConvLayer(in_channels, bottleneck_channels, projective_spec),
            ConvLayer(bottleneck_channels, bottleneck_channels, spec),
            se,
            ConvLayer(
                bottleneck_channels, out_channels, projective_spec.as_no_nonlinearity()
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._core(x)


def ResidualBottleneckBlock(
    in_channels: int,
    out_channels: int,
    bottleneck_multiple: int,
    spec,
    se_ratio: Optional[float] = None,
) -> nn.Module:
    """Residual bottleneck block: x + F(x), F = bottleneck transform."""
    if spec.layer_order != LayerOrder.CONV_NORM_NONLIN:
        raise ValueError(
            "ResidualBottleneckBlock only works with LayerOrder.CONV_NORM_NONLIN!"
        )

    shortcut = ConvShortcut(
        in_channels,
        out_channels,
        spec,
        ShortcutStrategy.CONV,
        identity_if_possible=True,
    )
    core = BottleneckTransform(
        in_channels, out_channels, bottleneck_multiple, spec, se_ratio
    )
    residual = Residual(core, shortcut, ResidualMode.BASIC_RESIDUAL)
    end = Nonlinearity(spec.nonlin_args)
    return nn.Sequential(residual, end)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        gn = lambda n: GroupNormArgs(num_per_group=n, affine=True)
        self.channels = c_lst = [32, 64, 128, 320, 768]
        self.strides = [2, 4, 8, 16, 32]
        conv_spec = ConvLayerSpec(
            layer_order=LayerOrder.CONV_NORM_NONLIN,
            nonlin_args=LeakyReluArgs(inplace=False),
            norm_args=gn(1),
        ).as_bias(False)

        self.pixel_means = nn.Parameter(
            torch.tensor((122.7717, 115.9465, 102.9801), dtype=torch.float32).view(
                1, 3, 1, 1
            ),
            requires_grad=False,
        )
        self.stem = ConvLayer(3, c_lst[0], spec=conv_spec.as_stride(2))
        self.up_blocks = nn.ModuleList(
            [
                ResidualBottleneckBlock(
                    in_channels=c_in,
                    out_channels=c_out,
                    bottleneck_multiple=1,
                    spec=conv_spec.as_stride(2),
                    se_ratio=1 / 4,
                )
                for c_in, c_out in zip(c_lst[:-1], c_lst[1:])
            ]
        )
        self.lat_blocks = nn.ModuleList(
            [
                ConvLayer(c, c, spec=conv_spec.as_ksize(1).as_no_nonlinearity())
                for c in c_lst
            ]
        )
        self.upsample_blocks = nn.ModuleList(
            [
                ConvLayer(c1, c2, conv_spec.as_ksize(1))
                for c1, c2 in zip(reversed(c_lst[1:]), reversed(c_lst[:-1]))
            ]
        )
        self.merge1 = nn.ModuleList(
            [ConvLayer(c, c, conv_spec.as_ksize(1)) for c in reversed(c_lst[:-1])]
        )
        self.downsample_blocks = nn.ModuleList(
            [
                ConvLayer(c2, c1, conv_spec.as_ksize(1).as_stride(2))
                for c1, c2 in zip(c_lst[1:], c_lst[:-1])
            ]
        )
        self.merge2 = nn.ModuleList(
            [ConvLayer(c, c, conv_spec.as_ksize(1)) for c in c_lst[1:]]
        )
        self.end = Nonlinearity(conv_spec.nonlin_args)

    def forward(self, rgbs: torch.Tensor):
        x = 255 * rgbs - self.pixel_means
        x = self.stem(x)
        blobs = [x]
        for block in self.up_blocks:
            x = block(x)
            blobs.append(x)

        x = self.lat_blocks[-1](blobs[-1])
        down_blobs = [self.end(x)]
        for up_block, blob, lat_block, merge_block in zip(
            self.upsample_blocks,
            reversed(blobs[:-1]),
            reversed(self.lat_blocks[:-1]),
            self.merge1,
        ):
            x_lat = lat_block(blob)
            x = x_lat + F.interpolate(
                up_block(x), mode="bilinear", size=x_lat.shape[-2:], align_corners=False
            )
            down_blobs.insert(0, merge_block(self.end(x)))

        up_blobs = [self.end(down_blobs[0])]
        x = down_blobs[0]
        for down_block, blob, merge_block in zip(
            self.downsample_blocks, down_blobs[1:], self.merge2
        ):
            x = down_block(x) + blob
            x = merge_block(self.end(x))
            up_blobs.append(self.end(x))

        return tuple(up_blobs)


class PositionalEncoding1d(nn.Module):
    def __init__(self, n_bases: int):
        super().__init__()
        self.n_bases = n_bases
        self.freq = nn.Parameter(
            torch.pi * 2 ** torch.arange(n_bases), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Fourier positional encoding."""
        x_scaled = torch.einsum("...i,i->...i", x.unsqueeze(-1), self.freq)
        return torch.cat([x_scaled.cos(), x_scaled.sin()], dim=-1)
