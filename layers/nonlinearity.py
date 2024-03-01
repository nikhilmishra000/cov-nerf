from typing import Any

import attr
import torch
import torch.nn as nn
from attr.validators import instance_of


@attr.s(kw_only=True, frozen=True)
class NonlinArgs:
    """Arguments required by a a Nonlinearity."""


@attr.s(kw_only=True, frozen=True)
class IdentityNonlinArgs(NonlinArgs):
    """Placeholder class for no-operation nonlinearity (i.e. Identity)."""


@attr.s(kw_only=True, frozen=True)
class ReluArgs:
    """Arguments required by a ReLU."""

    inplace: bool = attr.ib(default=True, validator=instance_of(bool))
    """If True, the operation is done inplace."""


@attr.s(kw_only=True, frozen=True)
class LeakyReluArgs(NonlinArgs):
    """Arguments required by LeakyReLU."""

    negative_slope: float = attr.ib(default=0.2, validator=instance_of(float))
    """The multiple applied to input features < 0. A negative_slope of 0 is equivalent to RelU, of 1 is equivalent to
    Identity."""
    inplace: bool = attr.ib(default=True, validator=instance_of(bool))
    """If True, the operation is done inplace."""

    @negative_slope.validator
    def negative_slope_validator(self, attribute: Any, negative_slope: int) -> None:
        assert 0 <= negative_slope <= 1


@attr.s(kw_only=True, frozen=True)
class SigmoidArgs(NonlinArgs):
    """Arguments required for a Sigmoid."""


class Nonlinearity(nn.Module):
    def __init__(self, nonlin_args: NonlinArgs):
        """Construct a nonlinearity layer as specified by the arguments.

        Parameters
        ----------
        nonlin_args: NonlinArgs
            Arguments to nonlinearity
        """
        super().__init__()

        self._nonlin: nn.Module
        if isinstance(nonlin_args, IdentityNonlinArgs):
            self._nonlin = nn.Identity()
        elif isinstance(nonlin_args, ReluArgs):
            self._nonlin = nn.ReLU(inplace=nonlin_args.inplace)
        elif isinstance(nonlin_args, LeakyReluArgs):
            self._nonlin = nn.LeakyReLU(
                negative_slope=nonlin_args.negative_slope, inplace=nonlin_args.inplace
            )
        elif isinstance(nonlin_args, SigmoidArgs):
            self._nonlin = nn.Sigmoid()
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply nonlinearity on input x.

        Parameters
        ----------
        x: torch.Tensor
            Input feature map.

        Returns
        -------
        torch.Tensor
            Output feature map.
        """
        out = self._nonlin(x)
        return out
