import enum
from typing import List, Tuple

import torch
import torch.nn as nn

from .conv_layer import ConvLayer, ConvLayerSpec, LayerOrder, Nonlinearity


class ResidualMode(enum.Enum):
    """Control different types of residual connections."""

    NONE = 0
    """No residual connection."""

    BASIC_RESIDUAL = 1
    """Simple, Resnet-style residual connection."""

    GATED = 2
    """Gating operation, but no residual connection."""

    GATED_RESIDUAL = 3
    """Gating operation and residual connection."""


class ShortcutStrategy(enum.Enum):
    """Control how to perform spatial downsampling in shortcut connection."""

    CONV = 0
    """Conv with ksize=1 & appropriate stride."""

    MAX_POOL = 2
    """Conv with ksize=1 & stride=1, then max pool with appropriate stride"""

    AVG_POOL = 3
    """Conv with ksize=1 & stride=1, then avg pool with appropriate stride"""


def ConvShortcut(
    input_channels: int,
    output_channels: int,
    spec: ConvLayerSpec,
    shortcut_strategy: ShortcutStrategy,
    identity_if_possible: bool = True,
) -> nn.Module:
    """Create a shortcut connection for use in Resnet-like architectures.

    Parameters
    ----------
    input_channels : int
    output_channels : int
    spec : ConvLayerSpec
    shortcut_strategy : ShortcutStrategy
    identity_if_possible : bool
        When `input_channels == output_channels` and `stride == 1`, a shortcut is not strictly necessary (can be identity).
        If True (default), return an identity module if the above condition is met.
        Otherwise, always return the shortcut module that would have be returned when the condition is not met.

    Returns
    -------
    nn.Module
    """
    spec = spec.as_ksize(1).as_groups(1)
    if spec.layer_order == LayerOrder.CONV_NORM_NONLIN:
        # nonlinearity is applied after shortcut is added to the `core`
        spec = spec.as_no_nonlinearity()

    cores: List[nn.Module] = []
    if (
        input_channels == output_channels
        and spec.conv_args.is_stride_1()
        and identity_if_possible
    ):
        pass

    elif shortcut_strategy == ShortcutStrategy.CONV:
        cores.append(ConvLayer(input_channels, output_channels, spec))

    elif shortcut_strategy == ShortcutStrategy.MAX_POOL:
        cores.append(ConvLayer(input_channels, output_channels, spec.as_stride(1)))
        if not spec.conv_args.is_stride_1():
            # todo(nikhil): make a PoolLayer class or make it part of ConvLayer
            pool_cls = {2: nn.MaxPool2d, 3: nn.MaxPool3d}[spec.conv_args.cardinality]
            cores.append(pool_cls(spec.conv_args.stride, ceil_mode=True))

    elif shortcut_strategy == ShortcutStrategy.AVG_POOL:
        cores.append(ConvLayer(input_channels, output_channels, spec.as_stride(1)))
        if not spec.conv_args.is_stride_1():
            pool_cls = {2: nn.AvgPool2d, 3: nn.AvgPool3d}[spec.conv_args.cardinality]  # type: ignore
            cores.append(pool_cls(spec.conv_args.stride, ceil_mode=True))

    else:
        raise NotImplementedError(shortcut_strategy)

    return nn.Sequential(*cores)


class Residual(nn.Module):
    def __init__(
        self,
        core: nn.Module,
        shortcut: nn.Module,
        mode: ResidualMode = ResidualMode.BASIC_RESIDUAL,
    ):
        super().__init__()
        self._core = core
        self._shortcut = shortcut
        self._mode = mode

    def forward(self, x):
        xc = self._core(x)
        if self._mode == ResidualMode.NONE:
            out = xc
        elif self._mode == ResidualMode.GATED:
            xc1, xc2 = xc.chunk(2, dim=1)
            out = xc1 * xc2.sigmoid()
        elif self._mode == ResidualMode.BASIC_RESIDUAL:
            xs = self._shortcut(x)
            out = xc + xs
        elif self._mode == ResidualMode.GATED_RESIDUAL:
            xs = self._shortcut(x)
            xc1, xc2 = xc.chunk(2, dim=1)
            out = torch.addcmul(xs, xc1, xc2.sigmoid())
        else:
            raise NotImplementedError(self._mode)

        return out


# Note: there are many ways to implement "residual blocks for conv nets".
# `ResidualBlock()` is most similar to `tu2.ConvBlock()`, but other architecture variants are possible (and not unreasonable).
# If you desire something else:
#   Option 1: Add arguments to `ResidualBlock()`, if it can be done cleanly.
#   Option 2: Create a new helper function that, similarly to `ResidualBlock()`, constructs the `core` & `shortcut` that you want, and then calls `Residual()`.


def ResidualBlock(
    input_channels: int,
    channels: Tuple[int, ...],
    spec: ConvLayerSpec,
    mode: ResidualMode = ResidualMode.BASIC_RESIDUAL,
    shortcut_strategy: ShortcutStrategy = ShortcutStrategy.CONV,
) -> nn.Module:
    """Create a residual block for conv nets.

    Parameters
    ----------
    input_channels : int
    channels : Tuple[int, ...]
        The channel sizes for each layer in this block. See `ConvLayer.create_stack()`.
    spec : ConvLayerSpec
        All layers will use this spec, except for the stride.
        The first layer & the shortcut will use the specified stride, and all others will have stride 1.
    mode : ResidualMode
        Defaults to `ResidualMode.BASIC_RESIDUAL`.
    shortcut_strategy : ShortcutStrategy
        Defaults to `ShortcutStrategy.CONV`.

    Returns
    -------
    nn.Module
    """
    output_channels = channels[-1]
    if mode in [ResidualMode.GATED, ResidualMode.GATED_RESIDUAL]:
        channels = (*channels[:-1], 2 * channels[-1])
    end_spec = spec
    if spec.layer_order == LayerOrder.CONV_NORM_NONLIN:
        end_spec = end_spec.as_no_nonlinearity()
    specs: Tuple[ConvLayerSpec, ...]
    if len(channels) == 1:
        specs = (end_spec,)
    else:
        end_spec = end_spec.as_stride(1)
        specs = (spec, *(spec.as_stride(1) for _ in channels[1:-1]), end_spec)

    core = ConvLayer.create_stack(input_channels, channels, specs, as_list=False)

    if mode in [ResidualMode.BASIC_RESIDUAL, ResidualMode.GATED_RESIDUAL]:
        shortcut = ConvShortcut(
            input_channels,
            output_channels,
            spec=spec,
            shortcut_strategy=shortcut_strategy,
            identity_if_possible=True,
        )
    else:
        shortcut = nn.Identity()

    residual: nn.Module = Residual(core, shortcut, mode)  # type: ignore
    if spec.layer_order == LayerOrder.CONV_NORM_NONLIN:
        # We turned off the last Nonlinearity in `end_spec`. Here we perform the residual & then apply it.
        end = Nonlinearity(spec.nonlin_args)
        residual = nn.Sequential(residual, end)

    return residual
