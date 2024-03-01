import enum
from typing import Any, List, Optional, Sequence, Tuple, Union

import attr
import numpy as np
import torch
import torch.nn as nn

import np_utils as npu

from .nonlinearity import (
    IdentityNonlinArgs,
    LeakyReluArgs,
    NonlinArgs,
    Nonlinearity,
    ReluArgs,
    SigmoidArgs,
)
from .normalization import BatchNormArgs, IdentityNormArgs, Normalization, NormArgs


class LayerOrder(enum.Enum):
    """In which order the convolution, normalization, and nonlinearity are applied."""

    CONV_NORM_NONLIN = 0
    """Convolution, normalization, then nonlinearity. This is the standard most papers use."""

    NORM_NONLIN_CONV = 1
    """Normalization, nonlinearity, then convolution. A small set of papers use this, arguing it is better."""


class ConvArgs:
    def __init__(self):
        raise ValueError(
            "Do not instantiate ConvArgs() directly! Use ConvArgs2d() or ConvArgs3d()."
        )

    kernel_size: Tuple[int, ...] = NotImplemented
    """Size of the conv kernel being applied."""

    stride: Tuple[int, ...] = NotImplemented
    """Spacing between contiguous convolutional operations."""

    dilation: Tuple[int, ...] = NotImplemented
    """Spacing between elements of kernel when applied on the feature map."""

    groups: Optional[int] = NotImplemented
    """Number of groups in the convolution operation. As an example a 2d conv with in_channels C1 out_channels C2
    kernel_size 3 will have a tensor kernel of shape (C1, C2, 3, 3) when groups = 1. If groups = 2, there would be two
    sets of (C1 / 2, C2 / 2, 3, 3) kernels. If this is specified, group_width cannot be, and vica versa.
    """

    group_width: Optional[int] = NotImplemented
    """Number of channels per group in the convolution operation. The in_channel must be divisible by this value.
    The number of groups will be then `in_channel // group_width`."""

    cardinality: int = NotImplemented
    """Whether the feature map is 1d, 2d or 3d. Affects what type of convolution and normalization is used. Must be a
    value in {1, 2, 3}"""

    bias: bool = NotImplemented
    """Whether a bias is applied alongside the convolution kernel."""

    def validate_kernel_size(self, attribute, value):
        assert len(self.kernel_size) == self.cardinality
        assert all(isinstance(k, int) for k in self.kernel_size)
        assert all(
            k % 2 == 1 and k > 0 for k in self.kernel_size
        ), f"Kernel size must be positive & odd, got {self.kernel_size}"

    def validate_stride(self, attribute, value):
        assert all(isinstance(s, int) for s in self.stride)
        assert all(
            s > 0 for s in self.stride
        ), f"Stride must be positive, got {self.stride}"

    def validate_dilation(self, attribute, value):
        assert all(isinstance(d, int) for d in self.dilation)
        assert all(
            d > 0 for d in self.dilation
        ), f"Dilation must be positive, got {self.dilation}"

    def validate_groups(self, attribute, value):
        if self.groups is None:
            assert isinstance(self.group_width, int) and self.group_width > 0
        else:
            assert self.group_width is None
            assert isinstance(self.groups, int) and self.groups > 0

    def is_stride_1(self) -> bool:
        """Return if stride is equivalent to 1."""
        return all(s == 1 for s in self.stride)

    @property
    def padding(self) -> Tuple[int, ...]:
        """Calculate padding such that the output shape does not depend on the kernel size or dilation.

        This is called "SAME" padding in other DL frameworks.
        Reference: https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html
        """
        return tuple(
            int(np.ceil((k + (k - 1) * (d - 1) - s) / 2.0))
            for k, s, d in zip(self.kernel_size, self.stride, self.dilation)
        )


def _converter3(x) -> Tuple[int, int, int]:
    return tuple(npu.ensure_len(npu.list_if_not(x), 3))  # type: ignore


def _converter2(x) -> Tuple[int, int]:
    return tuple(npu.ensure_len(npu.list_if_not(x), 2))  # type: ignore


@attr.s(kw_only=True, frozen=True)
class ConvArgs3d(ConvArgs):
    cardinality: int = 3
    kernel_size: Tuple[int, int, int] = attr.ib(
        converter=_converter3,
        validator=ConvArgs.validate_kernel_size,
        default=(3, 3, 3),
    )
    stride: Tuple[int, int, int] = attr.ib(
        converter=_converter3, validator=ConvArgs.validate_stride, default=(1, 1, 1)
    )
    dilation: Tuple[int, int, int] = attr.ib(
        converter=_converter3, validator=ConvArgs.validate_dilation, default=(1, 1, 1)
    )
    groups: Optional[int] = attr.ib(default=1, validator=ConvArgs.validate_groups)
    group_width: Optional[int] = attr.ib(default=None)
    bias: bool = attr.ib(default=True)


@attr.s(kw_only=True, frozen=True)
class ConvArgs2d(ConvArgs):
    cardinality: int = 2
    kernel_size: Tuple[int, int] = attr.ib(
        converter=_converter2, validator=ConvArgs.validate_kernel_size, default=(3, 3)
    )
    stride: Tuple[int, int] = attr.ib(
        converter=_converter2, validator=ConvArgs.validate_stride, default=(1, 1)
    )
    dilation: Tuple[int, int] = attr.ib(
        converter=_converter2, validator=ConvArgs.validate_dilation, default=(1, 1)
    )
    groups: Optional[int] = attr.ib(default=1, validator=ConvArgs.validate_groups)
    group_width: Optional[int] = attr.ib(default=None)
    bias: bool = attr.ib(default=True)


def _conv_args2d_to_3d(conv_args: ConvArgs2d) -> ConvArgs3d:
    assert isinstance(conv_args, ConvArgs2d)

    def _extend(x: Tuple[int, int]) -> Tuple[int, int, int]:
        assert np.unique(x).size == 1
        return x + (x[0],)

    return ConvArgs3d(
        kernel_size=_extend(conv_args.kernel_size),
        stride=_extend(conv_args.stride),
        dilation=_extend(conv_args.dilation),
        groups=conv_args.groups,
        group_width=conv_args.group_width,
        bias=conv_args.bias,
    )


def _conv_args3d_to_2d(conv_args: ConvArgs3d) -> ConvArgs2d:
    assert isinstance(conv_args, ConvArgs3d)

    def _shrink(x: Tuple[int, int, int]) -> Tuple[int, int]:
        assert np.unique(x).size == 1
        return x[:-1]

    return ConvArgs2d(
        kernel_size=_shrink(conv_args.kernel_size),
        stride=_shrink(conv_args.stride),
        dilation=_shrink(conv_args.dilation),
        groups=conv_args.groups,
        group_width=conv_args.group_width,
        bias=conv_args.bias,
    )


@attr.s(kw_only=True, frozen=True)
class ConvLayerSpec:
    """Specify the configuration for the basic convolution unit a module."""

    layer_order: LayerOrder = attr.ib(
        default=LayerOrder.CONV_NORM_NONLIN,
    )
    """Order of operations."""

    conv_args: ConvArgs = attr.ib(factory=ConvArgs2d)
    """Arguments to convolution."""

    nonlin_args: NonlinArgs = attr.ib(
        factory=IdentityNonlinArgs,
    )
    """Arguments to nonlinearity."""

    norm_args: NormArgs = attr.ib(factory=IdentityNormArgs)
    """Arguments to normalization."""

    @conv_args.validator
    def validate_conv_args(self, attribute, value):
        assert isinstance(self.conv_args, ConvArgs2d) or isinstance(
            self.conv_args, ConvArgs3d
        )

    @norm_args.validator
    def norm_args_validator(self, attribute: Any, norm_args: NormArgs):
        if isinstance(norm_args, BatchNormArgs):
            # make sure cardinality of conv_args and norm_args are the same
            assert self.conv_args.cardinality == norm_args.cardinality

    def as_no_nonlinearity(self) -> "ConvLayerSpec":
        """Remove nonlinearity from ConvLayerSpec."""
        return attr.evolve(self, nonlin_args=IdentityNonlinArgs())

    def as_no_normalization(self) -> "ConvLayerSpec":
        """Remove normalization from ConvLayerSpec."""
        return attr.evolve(self, norm_args=IdentityNormArgs())

    def as_no_inplace(self) -> "ConvLayerSpec":
        """If NonLin is Relu or LeakyRelu, set inplace to False."""
        if isinstance(self.nonlin_args, ReluArgs) or isinstance(
            self.nonlin_args, LeakyReluArgs
        ):
            return attr.evolve(
                self, nonlin_args=attr.evolve(self.nonlin_args, inplace=False)
            )
        else:
            return self

    def as_layer_order(self, layer_order: LayerOrder):
        return attr.evolve(self, layer_order=layer_order)

    def as_begin_spec(self) -> "ConvLayerSpec":
        """Set the spec to be the proper format for the first layer."""
        if self.layer_order == LayerOrder.NORM_NONLIN_CONV:
            # must set norm and nonlin to identity so first operation is a conv
            return self.as_no_normalization().as_no_nonlinearity()
        elif self.layer_order == LayerOrder.CONV_NORM_NONLIN:
            # nothing needs to change, since first operation is a conv
            return self
        else:
            raise NotImplementedError

    def as_end_spec(self) -> "ConvLayerSpec":
        """Set the spec to be the proper format for the last layer (i.e. the prediction)."""
        if self.layer_order == LayerOrder.NORM_NONLIN_CONV:
            # nothing needs to change, since last operation is a conv
            return self
        elif self.layer_order == LayerOrder.CONV_NORM_NONLIN:
            # must set norm and nonlin to identity so last operation is a conv
            return self.as_no_normalization().as_no_nonlinearity()
        else:
            raise NotImplementedError

    def as_stride(self, stride: Union[int, Tuple[int, ...]]) -> "ConvLayerSpec":
        """Set stride to `stride` for ConvLayerSpec."""
        return attr.evolve(self, conv_args=attr.evolve(self.conv_args, stride=stride))

    def as_groups(self, groups: int) -> "ConvLayerSpec":
        return attr.evolve(
            self, conv_args=attr.evolve(self.conv_args, groups=groups, group_width=None)
        )

    def as_group_width(self, group_width: int) -> "ConvLayerSpec":
        return attr.evolve(
            self,
            conv_args=attr.evolve(self.conv_args, groups=None, group_width=group_width),
        )

    def as_ksize(self, ksize: int) -> "ConvLayerSpec":
        """Set kernel size to `ksize` for ConvLayerSpec."""
        return attr.evolve(
            self, conv_args=attr.evolve(self.conv_args, kernel_size=ksize)
        )

    def as_bias(self, bias: bool) -> "ConvLayerSpec":
        """Set bias to `bias` for ConvLayerSpec."""
        return attr.evolve(self, conv_args=attr.evolve(self.conv_args, bias=bias))

    def as_cardinality(self, cardinality: int):
        """Make the spec compatible with `cardinality`."""
        conv_args: ConvArgs
        if cardinality == 2 and not isinstance(self.conv_args, ConvArgs2d):
            assert isinstance(self.conv_args, ConvArgs3d)
            conv_args = _conv_args3d_to_2d(self.conv_args)
        elif cardinality == 3 and not isinstance(self.conv_args, ConvArgs3d):
            assert isinstance(self.conv_args, ConvArgs2d)
            conv_args = _conv_args2d_to_3d(self.conv_args)
        elif cardinality not in {2, 3}:
            raise NotImplementedError
        else:
            conv_args = self.conv_args
        norm_args: NormArgs
        if isinstance(self.norm_args, BatchNormArgs):
            norm_args = attr.evolve(self.norm_args, cardinality=cardinality)
        else:
            norm_args = self.norm_args
        return attr.evolve(self, conv_args=conv_args, norm_args=norm_args)


class ConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, spec: ConvLayerSpec):
        """Create a convolution layer.

        Parameters
        ----------
        in_channels: int
        out_channels: int
        spec: ConvLayerSpec
        """
        super().__init__()

        self._layer_order = spec.layer_order
        conv_args = spec.conv_args
        cardinality = spec.conv_args.cardinality
        if cardinality not in {2, 3}:
            raise ValueError(
                f"Expected cardinality to be 2 or 3 but got {cardinality}; we don't support 1D convolutions yet"
            )
        # mypy still complains...
        conv_class: nn.Module
        if cardinality == 2:
            conv_class = nn.Conv2d  # type: ignore
        elif cardinality == 3:
            conv_class = nn.Conv3d  # type: ignore
        else:
            raise NotImplementedError

        if isinstance(conv_args.groups, int):
            groups = conv_args.groups
        else:
            assert isinstance(conv_args.group_width, int)
            assert in_channels % conv_args.group_width == 0
            groups = in_channels // conv_args.group_width
            assert out_channels % groups == 0

        if spec.layer_order == LayerOrder.CONV_NORM_NONLIN and not isinstance(
            spec.norm_args, IdentityNormArgs
        ):
            # there is a norm right after the conv, thus we should not do a bias
            bias = False
        else:
            bias = conv_args.bias

        self._conv = conv_class(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=conv_args.padding,
            kernel_size=conv_args.kernel_size,
            dilation=conv_args.dilation,
            stride=conv_args.stride,
            groups=groups,
            bias=bias,
        )
        if spec.layer_order == LayerOrder.CONV_NORM_NONLIN:
            norm_channels = out_channels
        elif spec.layer_order == LayerOrder.NORM_NONLIN_CONV:
            norm_channels = in_channels
        else:
            raise NotImplementedError

        self._norm = Normalization(norm_channels, spec.norm_args)
        self._nonlin = Nonlinearity(spec.nonlin_args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution, nonlinearity, and normalization on input x.

        Parameters
        ----------
        x: torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        if self._layer_order == LayerOrder.CONV_NORM_NONLIN:
            out = self._nonlin(self._norm(self._conv(x)))
        elif self._layer_order == LayerOrder.NORM_NONLIN_CONV:
            out = self._conv(self._nonlin(self._norm(x)))
        else:
            raise NotImplementedError

        return out

    @classmethod
    def create_stack(
        cls,
        in_channels: int,
        channels: Sequence[int],
        specs: Sequence[ConvLayerSpec],
        as_list: bool = False,
    ) -> Union[nn.Sequential, List["ConvLayer"]]:
        """Create a stack of ConvLayers with the specified channels.

        Parameters
        ----------
        in_channels : int
        channels : Sequence[int]
            Layers will go `in_channels -> channels[0]`, `channels[0] -> channels[1]`, ... `channels[-2] -> channels[-1]`.
        specs : Sequence[ConvLayerSpec]
            Should have the same length as `channels`. Every spec should have the same cardinality.
        as_list : bool
            If True, returns a `List[ConvLayer]`.
            Otherwise (default), returns an `nn.Sequential` object.

        Returns
        -------
        Union[nn.Sequential, List["ConvLayer"]]
        """
        assert npu.all_same([s.conv_args.cardinality for s in specs])

        cores = []
        c_in = in_channels
        for c_out, spec in npu.zip_strict(channels, specs):
            cores.append(cls(c_in, c_out, spec))
            c_in = c_out

        if as_list:
            return cores
        else:
            return nn.Sequential(*cores)
