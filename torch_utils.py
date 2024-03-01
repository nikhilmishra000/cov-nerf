from typing import Any, List, Optional, Union

import attr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import np_utils as npu

DeviceLike = str | torch.device


def to_np(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x

    elif torch.is_tensor(x):
        return x.detach().cpu().numpy()

    else:
        return np.asarray(x)


def to_torch(
    x: Any,
    device: DeviceLike | None = None,
    dtype: torch.dtype | None = None,
    recursive: bool = False,
    strict: bool = False,
    preserve_doubles: bool = False,
) -> Any:
    """Map input to torch.

    When `recursive == False`, the following conversions are applied.
      1. Tensors:
        - We call `.to(device, dtype)` using the specified values.
      2. Numpy arrays of a numeric type (booleans, integers, floating-point numbers):
        - Unless `preserve_doubles` is set, float64 is cast to float32 (it is rare for doubles to be used in torch).
        - Any signed integers dtypes (int8, int16, int32) are cast to int64 (which is also preferred in torch).
        - Convert the cast array to a tensor, then apply (1).
      3. Numpy arrays of non-numeric types (strings, objects):
        - These are ignored if `strict == False`, otherwise an error is raised.
      4. `DeviceTransferable` objects:
        - We call their `.to()` method, mimicking (1).
      5. Other types:
        - If `strict == True`, an error is raised.
        - If `strict == False`, we return the value as is.

    If `recursive == True`, then the input can be a nested structure (list/tuple/set/dict).
    We traverse the structure and apply the non-recursive conversion rules to each leaf.

    Parameters
    ----------
    x : Any
    device : Optional[DeviceLike]
        If supplied, all tensors will be put onto this device.
    dtype : Optional[torch.dtype]
        If supplied, all tensors will be casted to this dtype.
    recursive : bool
        If True, `x` can be a nested structure, and we will traverse it.
    strict : bool
        If True, and we encounter types that cannot be converted to tensor, an error will be raised.
    preserve_doubles : bool
        If False (default), float64 arrays will be cast to float32.
        If True, the original dtype will be preserved. This option may become the default in the future.

    Returns
    -------
    Any
        A tensor, or a structure identical to `x`, with the above conversions applied.
    Raises
    ------
    ValueError
        If `strict == True` and we cannot convert a value to a tensor.
    """
    if recursive:
        return npu.map_until(
            lambda z: to_torch(
                z, device=device, dtype=dtype, recursive=False, strict=strict
            ),
            x,
        )
    else:
        if torch.is_tensor(x):
            return x.to(device=device, dtype=dtype)

        elif any(
            isinstance(x, primitive_type) for primitive_type in (bool, int, float)
        ):
            return x

        else:
            x_array = to_np(x)
            x_dtype = x_array.dtype
            if np.issubdtype(x_dtype, np.number) or np.issubdtype(x_dtype, bool):
                if np.issubdtype(x_dtype, np.floating):
                    if x_dtype == np.float64 and not preserve_doubles:
                        x_array = x_array.astype(np.float32)

                elif np.issubdtype(x_dtype, np.signedinteger):
                    x_array = x_array.astype(np.int64)

                if x_array.size > 0:
                    return torch.from_numpy(np.ascontiguousarray(x_array)).to(
                        device=device, dtype=dtype
                    )
                else:
                    return torch.from_numpy(np.copy(x_array)).to(
                        device=device, dtype=dtype
                    )

            elif strict:
                raise ValueError(
                    "Input must be a numpy array of kind boolean, signed integer, unsigned integer, or floating-point!"
                )
            else:
                return x


class DeviceTransferableMixin:
    """Mixin class that provides implementation of the `DeviceTransferable` interface.

    Subclasses must be `attr` classes.
    """

    def to(
        self, device: Optional[DeviceLike] = None, dtype: Optional[torch.dtype] = None
    ) -> "DeviceTransferableMixin":
        """Transfer all `attr` attributes torch tensors to a certain device.

        Parameters
        ----------
        device : Type[Device]
            Device to transfer torch tensors to.

        Returns
        -------
        DeviceTransferableMixin
            The object with its torch objects in the desired device.
        """
        kvs = attr.asdict(self, recurse=False)
        kvs_updated = to_torch(
            kvs, device=device, dtype=dtype, recursive=True, strict=False
        )
        return attr.evolve(self, **kvs_updated)


@torch.jit.script_if_tracing
def expand_dim(
    x: torch.Tensor,
    dim: int,
    shape: Union[torch.Tensor, List[int]],
) -> torch.Tensor:
    """Expand a tensor to have extra dimensions, tiling the values without a copy.

    Parameters
    ----------
    x : torch.Tensor
    dim : int
        Insert the new dimensions at this index in `x.shape`.
    shape : Union[torch.Tensor, List[int]]
        The inserted dimensions will have this shape.

    Returns
    -------
    torch.Tensor
    """
    if dim < 0:
        dim = x.ndim + 1 + dim

    if isinstance(shape, torch.Tensor):
        expanded_shape = torch.Size([int(s.item()) for s in shape])
    else:
        expanded_shape = torch.Size(shape)

    single_shape = (
        x.shape[:dim] + torch.Size([1 for _ in expanded_shape]) + x.shape[dim:]
    )
    full_shape = x.shape[:dim] + expanded_shape + x.shape[dim:]
    return x.view(single_shape).expand(full_shape)


@torch.jit.script
def unflatten_dim(x: torch.Tensor, dim: int, shape: List[int]) -> torch.Tensor:
    """Unflatten a dimension of a tensor into the specified shape.

    Parameters
    ----------
    x : torch.Tensor
        Can have any shape/device/dtype.
    dim : int
        The dimension to unflatten.
    shape : List[int]
        The shape to unflatten `dim` into. `np.prod(shape)` should equal `x.size(dim)`.

    Returns
    -------
    torch.Tensor
        A tensor of the same dtype and device as `x`.
        The shape is the same except that `dim` has been expanded into `shape`.

    Example
    -------
    >>> x = torch.randn(4, 4, 4)
    >>> y = unflatten(x, dim=1, shape=[2, 2])
    >>> print(y.shape)
    (4, 2, 2, 4)
    """
    if dim < 0:
        dim = x.ndim + dim
    new_shape = x.shape[:dim] + torch.Size(shape) + x.shape[dim + 1 :]
    return x.reshape(new_shape)


def interp3d(x: torch.Tensor, p: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """Implement a non-batched version of `batch_interp3d()`.

    This function exists for legacy support, and might be removed in the near future.

    Parameters
    ----------
    x : torch.Tensor
        Shape (C, D, H, W), any dtype.
    p : torch.Tensor
        Shape (..., 3), dtype float32.
    mode : str
        One of {"bilinear", "nearest"}.

    Returns
    -------
    torch.Tensor
        Interpolated values from `x`, dtype `x.dtype` and shape(..., C)
    """
    return batch_interp3d(x[None], p[None], mode=mode)[0]


def batch_interp3d(
    x: torch.Tensor, p: torch.Tensor, mode: str = "bilinear"
) -> torch.Tensor:
    """Interpolate values from a 3D grid.

    Args
    ----
        x: shape(B, C, D, H, W), any dtype
        p: dtype float32, shape(B, ..., 3), format (d_idx, h_idx, w_idx)
           Note that this differs from `batch_interp2d()`, which uses the flipped format (w_idx, h_idx).
        mode: one of {"bilinear", "nearest"}

    Returns
    -------
        Interpolated values from `x`, dtype `x.dtype` and shape(B, ..., C)
    """
    if mode not in ("bilinear", "nearest"):
        raise ValueError(f"Unexpected interpolation mode: {mode}")

    if x.dtype not in (torch.float16, torch.float32) and mode != "nearest":
        raise ValueError(
            f'Must set mode="nearest" for non-floating dtypes (got dtype={x.dtype} and mode={mode}'
        )

    size = torch.tensor(x.shape[-3:], dtype=torch.float32, device=x.device)
    p_normed = (
        p.div(size.float().sub(1.0).mul(0.5)).sub(1.0).flip(-1)
    )  # same shape but [-1, 1] values

    x_interp = F.grid_sample(
        x.type_as(p_normed),
        p_normed.reshape(p_normed.size(0), -1, 1, 1, 3),
        mode=mode,
        align_corners=True,
    )  # shape(B, C, N, 1, 1)

    dims = [dim for dim in p.shape[:-1]] + [x.size(1)]
    return x_interp.permute(0, 2, 1, 3, 4).reshape(dims).type_as(x)


def batch_interp2d(
    x: torch.Tensor, px: torch.Tensor, mode: str = "bilinear"
) -> torch.Tensor:
    """Interpolate values from a 2D grid.

    This function acts like `F.grid_sample()`, except that the coordinates correspond to `x.shape` rather than being normalized to [-1, 1].

    Parameters
    ----------
    x : torch.Tensor
        Shape (B, C, H, W), any dtype.
    px : torch.Tensor
        Shape (B, ..., 2), dtype float32.
    mode : str
        One of {"bilinear", "nearest"}.

    Returns
    -------
    torch.Tensor
        Interpolated values from `x`, dtype `x.dtype` and shape(B, ..., C)
    """
    if mode not in ("bilinear", "nearest"):
        raise ValueError(f"Unexpected interpolation mode: {mode}")

    if x.dtype not in (torch.float16, torch.float32) and mode != "nearest":
        raise ValueError(
            f'Must set mode="nearest" for non-floating dtypes (got dtype={x.dtype} and mode={mode}'
        )

    im_size = torch.tensor(x.shape[-2:], dtype=torch.float32, device=x.device)
    px_normed = px.div(im_size.float().sub(1.0).flip(0).mul(0.5)).sub(
        1.0
    )  # same shape but [-1, 1] values

    x_interp = F.grid_sample(
        x.type_as(px_normed),
        px_normed.view(px_normed.size(0), -1, 1, 2),  # shape(B, ..., 2) -> (B, N, 1, 2)
        mode=mode,
        align_corners=True,
    )  # shape(1, C, N, 1)

    dims = [dim for dim in px.shape[:-1]] + [x.size(1)]
    return x_interp.permute(0, 2, 1, 3).reshape(dims).type_as(x)


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def BatchApply(module, dims=1, **kwargs):
    """Apply the module to input with extra leading dimensions.

    Args
    ----
        dims: The number of leading dimensions.
    """

    def apply_fn(*xs):
        leading_shapes = [x.shape[: dims + 1] for x in xs]
        assert npu.all_same(leading_shapes)
        flat_xs = [x.flatten(0, dims) for x in xs]
        flat_y = module(*flat_xs, **kwargs)
        return flat_y.reshape(*leading_shapes[0], *flat_y.shape[1:])

    return apply_fn
