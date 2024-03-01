from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_utils import unflatten_dim


class BufferAttend1d(nn.Module):
    def __init__(
        self,
        in_channel: int,
        key_channel: int,
        val_channel: int,
        num_heads: int = 1,
        flatten_output: bool = True,
    ):
        """Construct 1D Transformer-style attention.

        Parameters
        ----------
        in_channel: int
            Channel dimension of input `x`.
        key_channel: int
            Dimension of the query and keys.
        val_channel: int
            Dimension of the values.
        num_heads: int
            The number of attention heads to use.
        flatten_output: bool
            If True (default), flatten the heads & channel dimensions of the output.
            See `forward()` for details.
        """
        super().__init__()
        self._key_channel = key_channel
        self._val_channel = val_channel
        self._num_heads = num_heads
        self._flatten_output = flatten_output

        self._key_fn = nn.Linear(in_channel, key_channel * num_heads)
        self._query_fn = nn.Linear(in_channel, key_channel * num_heads)
        self._value_fn = nn.Linear(in_channel, val_channel * num_heads)

    def forward(
        self, x: torch.Tensor, buffer: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):
        """Perform Transformer-style attention where x is the query and the buffer generates the values and keys.

        Parameters
        ----------
        x: torch.Tensor
            Shape: (..., q, d); used to generate the query.
        buffer: torch.Tensor
            Shape: (..., k, d); used to generate the keys and values.
        mask: Optional[torch.Tensor]
            Shape: (..., q, k); If present, used to mask out parts of the attention.

        Returns
        -------
        torch.Tensor
            A tensor of shape (..., q, *trailing_shape).
            If `flatten_output` is True, then trailing_shape is (num_heads * val_channel).
            Otherwise, it is (num_heads, val_channel).
        """
        if buffer is None:
            buffer = x

        # project to key/val channels
        query = self._query_fn(x).float()  # shape: (..., q, h * dk)
        keys = self._key_fn(buffer).float()  # shape: (..., k, h * dk)
        vals = self._value_fn(buffer).float()  # shape: (..., k, h * dv)

        # unflatten the heads dimension
        query = unflatten_dim(
            query, dim=-1, shape=[self._num_heads, self._key_channel]
        )  # shape: (..., q, h, dk)
        keys = unflatten_dim(
            keys, dim=-1, shape=[self._num_heads, self._key_channel]
        )  # shape: (..., k, h, dk)
        vals = unflatten_dim(
            vals, dim=-1, shape=[self._num_heads, self._val_channel]
        )  # shape: (..., k, h, dv)

        # perform transformer-style attention, shape: (..., q, k, h)
        logits = torch.einsum("...qhd, ...khd -> ...qkh", query, keys) / np.sqrt(
            self._key_channel
        )

        if mask is not None:
            logits = logits.masked_fill(~mask.unsqueeze(-1), -(2**12))

        # softmax over the `k` dimension
        probs = torch.softmax(logits, dim=-2)
        if mask is not None:
            # If there are no True entries in `mask` for a given slice, we would have `logits = [-2 ** 12, ..., -2 ** 12]`.
            # This would gives us `probs = [1/k, ..., 1/k]`, which gives an average over the values for the slice.
            # This is not what we want, so instead we manually set the `probs` to zero, which also makes `read` zero.

            # probs: shape (..., q, k, h), mask: shape (..., q, k)
            probs = probs.masked_fill(~mask.any(-1, keepdim=True).unsqueeze(-1), 0.0)

        read = torch.einsum("...qkh, ...khd -> ...qhd", probs, vals).type_as(
            x
        )  # shape: (..., q, h, dv)
        if self._flatten_output:
            read = read.flatten(-2, -1)
        return read
