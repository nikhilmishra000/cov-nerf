import functools
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_blobs import ConvBlobSpec
from .conv_layer import ConvArgs2d, ConvLayer, ConvLayerSpec
from .nonlinearity import LeakyReluArgs
from .normalization import GroupNormArgs
from .residual_block import ResidualBlock


def resize(cardinality: int) -> callable:
    if cardinality == 2:
        return functools.partial(F.interpolate, mode="bilinear", align_corners=False)
    elif cardinality == 3:
        return functools.partial(F.interpolate, mode="trilinear", align_corners=False)
    else:
        raise NotImplementedError(
            "Resizing only support for tensors of cardinality 2 or 3."
        )


class UNetSimple(nn.Module):
    def __init__(
        self,
        in_channels: int,
        unet_spec: ConvBlobSpec,
        conv_spec: Optional[ConvLayerSpec] = None,
        lite: bool = False,
    ):
        """Construct simple UNet implementation. The forward pass takes a simple tensor as input, and returns all resolutions of the second UNet pass.

        The output tensor corresponding to the original input resolution can be accessed via `output.blobs[0]`.

        Parameters
        ----------
        in_channels: int
            Number of input channels.
        unet_spec: ConvBlobSpec
            Description of the feature maps.
        conv_spec: ConvLayerSpec
            Specification for the Conv Layers.
        lite: bool
            Whether to construct light weight UNet implementation by using ConvLayers instead of ResidualBlocks
        """
        super().__init__()
        if conv_spec is None:
            conv_spec = ConvLayerSpec(
                nonlin_args=LeakyReluArgs(),
                norm_args=GroupNormArgs(num_groups=1),
                conv_args=ConvArgs2d(),
            )

        assert conv_spec.conv_args.is_stride_1()

        self._out_spec = unet_spec
        self._cardinality = conv_spec.conv_args.cardinality
        self._resize = resize(self._cardinality)

        self._preprocess: nn.Module
        if in_channels != unet_spec.channels[0]:
            self._preprocess = ConvLayer(in_channels, unet_spec.channels[0], conv_spec)
        else:
            self._preprocess = nn.Identity()

        channels = unet_spec.channels

        strides = []
        for s1, s2 in zip(unet_spec.strides[:-1], unet_spec.strides[1:]):
            if isinstance(s1, int):
                assert isinstance(s2, int)
                strides.append(tuple([s2 // s1]))
            elif isinstance(s1, tuple):
                assert isinstance(s2, tuple)
                strides.append(tuple([ss2 // ss1 for ss1, ss2 in zip(s1, s2)]))
            else:
                raise NotImplementedError

        def _conv_block(c1: int, c2: int, spec: ConvLayerSpec) -> Any:
            return (
                ConvLayer(c1, c2, spec) if lite else ResidualBlock(c1, (c2, c2), spec)
            )

        self._firstpass = nn.ModuleList(
            [
                _conv_block(c1, c2, conv_spec.as_stride(s))
                for c1, c2, s in zip(channels[:-1], channels[1:], strides)
            ]
        )
        self._secondpass = nn.ModuleList(
            [_conv_block(c, c, conv_spec) for c in channels]
        )

        self._secondpass_upsample = nn.ModuleList(
            ConvLayer(c1, c2, conv_spec) for c1, c2 in zip(channels[1:], channels[:-1])
        )

    @property
    def output_spec(self):
        return self._out_spec

    def downward(self, xs: torch.Tensor) -> Sequence[torch.Tensor]:
        xs = self._preprocess(xs)
        first_blobs = [xs]
        x = xs

        for core in self._firstpass:
            x = core(x)
            first_blobs.append(x)
        return first_blobs

    def upward(self, blobs: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        x = blobs[-1]
        x = self._secondpass[-1](x)
        second_blobs = [x]
        for xx, core, core_up in zip(
            reversed(blobs[:-1]),
            reversed(self._secondpass[:-1]),
            reversed(self._secondpass_upsample),
        ):
            x = F.interpolate(core_up(x), size=xx.shape[-self._cardinality :])
            x = core(xx + x)
            second_blobs.insert(0, x)
        return second_blobs

    def forward(self, xs: torch.Tensor) -> Sequence[torch.Tensor]:
        """Apply Unet to input."""
        first_blobs = self.downward(xs)
        second_blobs = self.upward(first_blobs)
        return second_blobs
