from typing import List, Tuple

import torch
import torch.nn as nn

from layers.nonlinearity import NonlinArgs, Nonlinearity


class MLP(nn.Module):
    def __init__(
        self,
        in_channel: int,
        hidden_channels: Tuple[int, ...],
        nonlin_args: NonlinArgs,
        activate_final: bool = False,
    ):
        """Construct a multi layer perceptron, which comprises of alternating dense and nonlinearity layers.

        Parameters
        ----------
        in_channel: int
            Dimensionality of input vector.
        hidden_channels: Tuple[int, ...]
            Channel dimension of each hidden layer in the network.
        nonlin_args: NonlinArgs
            Specification of the nonlinearity layer.
        activate_final: bool
            If true, the last dense layer is followed by a nonlinearity.
        """
        super().__init__()
        channels = (in_channel, *hidden_channels)
        core: List[nn.Module] = []
        for i in range(len(channels) - 1):
            core.append(nn.Linear(channels[i], channels[i + 1]))
            if i + 2 < len(channels) or activate_final:
                core.append(Nonlinearity(nonlin_args=nonlin_args))
        self._core = nn.Sequential(*core)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._core(x)
