from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import attr
from cached_property import cached_property

MaybeStride = Union[int, Tuple[int, ...]]

import np_utils as npu


def mapper(func, astype):
    def converter(x):
        return astype(func(xx) for xx in x)

    return converter


def _stride_converter(cardinality: int) -> Callable[[MaybeStride], Tuple[int, ...]]:
    def _converter(x: MaybeStride) -> Tuple[int, ...]:
        return tuple(npu.ensure_len(x, cardinality))

    return _converter


class ConvBlobSpec:
    strides: Tuple[Tuple[int, ...], ...] = NotImplemented
    """Each stride is always a tuple of length `cardinality`.

    They may be specified as integers, but will be expanded to a tuple on construction.
    Strides here are cumulative, and should be sorted in increasing order.

    Example
    -------

    A stride tuple of (2, 4, 12) indicates that the original input tensor was first subjected to a stride 2 convolution, then another stride 2 convolution, followed by a last stride 3 convolution.
    If you want to query the relative strides, use the `relative_strides` property of this class.

    strides = (2, 4, 12)
    relative_strides = (2, 2, 3)

    """

    channels: Tuple[int, ...] = NotImplemented
    """Channel sizes corresponding to each stride. Should have the same length as `strides`."""

    cardinality: int = NotImplemented
    """2 for conv2d, 3 for conv3d."""

    def __init__(self, *arg, **kwargs):
        raise ValueError(
            "Do not instantiate ConvBlobSpec() directly! Use ConvBlobSpec2d() or ConvBlobSpec3d()."
        )

    def __len__(self):
        return len(self.strides)

    def validate_strides(self, attribute, value):
        if len(self.strides) != len(self.channels):
            raise ValueError(
                f"Strides and channels must have the same length. Got {len(self.strides)} and {len(self.channels)}"
            )

        for s, s_prev in zip(self.strides[1:], self.strides[:-1]):
            all_nondecreasing = all([ss >= ss_prev for ss, ss_prev in zip(s, s_prev)])
            one_increasing = any([ss > ss_prev for ss, ss_prev in zip(s, s_prev)])
            if not all_nondecreasing or not one_increasing:
                raise ValueError(
                    f"Strides must be monotonically increasing! Got: {self.strides}"
                )

    @cached_property
    def _stride_to_channel(self) -> Dict[Tuple[int, ...], int]:
        return {s: c for s, c in npu.zip_strict(self.strides, self.channels)}

    def get_channel(self, stride: MaybeStride) -> int:
        """Get the channel corresponding to `stride`, or raise an error if it doesn't exist."""
        stride = _stride_converter(self.cardinality)(stride)
        if not stride in self._stride_to_channel:
            raise ValueError(
                f"Could not get channel for stride {stride}! The available strides are: {self.strides}"
            )

        return self._stride_to_channel[stride]

    def select(self, strides: Tuple[MaybeStride, ...]) -> "ConvBlobSpec":
        """Return a new `ConvBlobSpec` filtered to only include the specified `strides`. All must exist."""
        return attr.evolve(
            self,
            strides=strides,
            channels=tuple([self.get_channel(s) for s in strides]),
        )

    def clip(
        self,
        min_stride: Optional[MaybeStride] = None,
        max_stride: Optional[MaybeStride] = None,
    ) -> "ConvBlobSpec":
        if not min_stride is not None or max_stride is not None:
            raise ValueError(
                "At least one of (min_stride, max_stride) should not be None!"
            )

        start_idx = min_stride and self.strides.index(
            _stride_converter(self.cardinality)(min_stride)
        )
        end_idx = (
            max_stride
            and self.strides.index(_stride_converter(self.cardinality)(max_stride)) + 1
        )
        sli = slice(start_idx, end_idx)
        return attr.evolve(self, strides=self.strides[sli], channels=self.channels[sli])

    @cached_property
    def relative_strides(self) -> Tuple[Tuple[int, ...], ...]:
        """Find the relative stride between the blobs.

        Thus the first value is always 1 (relative stride of largest blob relative to itself is identity). The second
        value is the difference between self.strides[1] and self.strides[0].

        Example:
        strides = (2, (4, 8), (8, 8), 16)
        relative_strides = ((1, 1), (2, 4), (2, 1), (2, 2))

        Returns
        -------
        Tuple

        """
        strides: List[Tuple[int, ...]] = [tuple(1 for _ in range(self.cardinality))]
        for s1, s2 in zip(self.strides[:-1], self.strides[1:]):
            strides.append(tuple(ss2 // ss1 for ss1, ss2 in zip(s1, s2)))
        return tuple(strides)

    @cached_property
    def has_uniform_strides(self) -> bool:
        """Return True if all of the strides are uniform (are the same for each dimension).

        For example, (2, 2) is a uniform stride, but (2, 3) is not.
        """
        return all(npu.all_same(s) for s in self.strides)

    @cached_property
    def integer_strides(self) -> Tuple[int, ...]:
        """Convert `self.strides` to integers.

        If `not self.has_uniform_strides`, then this is not possible, and an error will be raised.
        """
        if not self.has_uniform_strides:
            raise ValueError(
                f"ConvBlobSpec does not have uniform strides! Strides are: {self.strides}"
            )
        return tuple(s[0] for s in self.strides)

    def to_json(self) -> Dict[str, Any]:
        return attr.asdict(self)

    @classmethod
    def from_json(cls, json: Dict[str, Any]) -> "ConvBlobSpec":
        return cls(**json)


@attr.s(frozen=True, kw_only=True)
class ConvBlobSpec2d(ConvBlobSpec):
    strides: Tuple[Tuple[int, int], ...] = attr.ib(
        converter=mapper(_stride_converter(2), astype=tuple),
        validator=ConvBlobSpec.validate_strides,
    )
    channels: Tuple[int, ...] = attr.ib(converter=mapper(int, astype=tuple))
    cardinality: int = 2


@attr.s(frozen=True, kw_only=True)
class ConvBlobSpec3d(ConvBlobSpec):
    strides: Tuple[Tuple[int, int, int], ...] = attr.ib(
        converter=mapper(_stride_converter(3), astype=tuple),
        validator=ConvBlobSpec.validate_strides,
    )
    channels: Tuple[int, ...] = attr.ib(converter=mapper(int, astype=tuple))
    cardinality: int = 3
