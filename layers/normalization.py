import enum
from typing import Any, Optional

import attr
import torch
import torch.nn as nn
from attr.validators import instance_of


class NormType(enum.Enum):
    IDENTITY = 0
    BATCH_NORM = 1
    SYNC_BATCH_NORM = 2
    GROUP_NORM = 3


@attr.s(kw_only=True, frozen=True)
class NormArgs:
    """Arguments required by a Normalization Layer."""


@attr.s(kw_only=True, frozen=True)
class IdentityNormArgs(NormArgs):
    """Placeholder class for no-operation norm (i.e. Identity)."""


@attr.s(kw_only=True, frozen=True)
class BatchNormArgs(NormArgs):
    """Arguments required by Batch Norm."""

    affine: bool = attr.ib(default=True, validator=instance_of(bool))
    """If true, an affine layer is applied after the normalization is completed."""

    eps: float = attr.ib(default=1e-5, validator=instance_of(float))
    """Added to standard deviation for numerical stability reasons."""

    momentum: float = attr.ib(default=0.01, validator=instance_of(float))
    """Momentum of running average statistics used during test time."""

    cardinality: int = attr.ib(default=2, validator=instance_of(int))
    """Whether the feature map is 1d, 2d or 3d. Affects what type of convolution and normalization is used. Must be a
    value in {1, 2, 3}"""


@attr.s(kw_only=True, frozen=True)
class SyncBatchNormArgs(BatchNormArgs):
    """Additional arguments required by Sync Batch Norm."""

    process_group: Optional[str] = attr.ib(default=None)
    """The process group the code running this is part of. Used to determine with which workers to sync the mini
    batch statistics with."""


@attr.s(kw_only=True, frozen=True)
class GroupNormArgs(NormArgs):
    """Arguments required by Group Norm."""

    affine: bool = attr.ib(default=True, validator=instance_of(bool))
    """If true, an affine layer is applied after the normalization is completed"""

    eps: float = attr.ib(default=1e-5, validator=instance_of(float))
    """Added to standard deviation for numerical stability reasons"""

    num_groups: Optional[int] = attr.ib(default=None)
    """If specified, num_per_group must be None. Specifies number of groups. The feature map num_channels must be
    divisble by this number."""

    num_per_group: Optional[int] = attr.ib(default=None)
    """If specified, num_groups must be None. Specifies number of channels per group. The feature map
    num_channels must be divisble by this number."""

    @num_groups.validator
    def num_groups_validator(self, attribute: Any, num_groups: Optional[int]) -> None:
        if isinstance(self.num_per_group, int):
            assert num_groups is None
            assert self.num_per_group > 0
        else:
            assert self.num_per_group is None
            assert isinstance(num_groups, int)
            assert num_groups > 0


def calculate_group_norm_groups(args: GroupNormArgs, in_channels: int) -> int:
    """Calculate number of groups for GroupNorm based on the specification in args and the input channels.

    Parameters
    ----------
    args: GroupNormArgs
        Arguments to Group Norm.
    in_channels
        Channels of input feature map.

    Returns
    -------
    int
        The number of groups for GroupNorm.
    """
    if args.num_groups is not None:
        assert in_channels % args.num_groups == 0, (in_channels, args.num_groups)
        return args.num_groups
    else:
        assert args.num_per_group is not None
        assert in_channels % args.num_per_group == 0, (in_channels, args.num_per_group)
        return in_channels // args.num_per_group


class Normalization(nn.Module):
    def __init__(self, norm_channels: int, norm_args: NormArgs):
        """Construct a Normalization layer as specified by the input arguments.

        Parameters
        ----------
        norm_channels: int
            Channels of the input feature map.
        norm_args: NormArgs
            Arguments to Normalization.
        """
        super().__init__()

        self._norm: nn.Module
        if isinstance(norm_args, IdentityNormArgs):
            self._norm = nn.Identity()
        elif isinstance(norm_args, GroupNormArgs):
            num_groups = calculate_group_norm_groups(norm_args, norm_channels)
            self._norm = nn.GroupNorm(
                num_groups=num_groups,
                num_channels=norm_channels,
                eps=norm_args.eps,
                affine=norm_args.affine,
            )
        elif isinstance(norm_args, BatchNormArgs):
            # mypy still complains...
            norm_class: nn.Module
            if norm_args.cardinality == 2:
                norm_class = nn.BatchNorm2d  # type: ignore
            elif norm_args.cardinality == 3:
                norm_class = nn.BatchNorm3d  # type: ignore
            else:
                raise NotImplementedError

            self._norm = norm_class(
                num_features=norm_channels,
                eps=norm_args.eps,
                momentum=norm_args.momentum,
                affine=norm_args.affine,
            )

        elif isinstance(norm_args, SyncBatchNormArgs):
            self._norm = nn.SyncBatchNorm(
                num_features=norm_channels,
                eps=norm_args.eps,
                momentum=norm_args.momentum,
                affine=norm_args.affine,
                process_group=norm_args.process_group,
            )
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalization on input x.

        Parameters
        ----------
        x: torch.Tensor
            Input feature map.

        Returns
        -------
        torch.Tensor
            Output feature map.
        """
        out = self._norm(x)
        return out
