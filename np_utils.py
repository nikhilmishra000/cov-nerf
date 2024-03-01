import collections
import itertools
from typing import Any, Generator, Iterable, Tuple


def is_sequence(x):
    return isinstance(x, collections.abc.Sequence) and not isinstance(x, str)


def not_seq_or_dict(x):
    return not is_sequence(x) and not isinstance(x, dict)


def list_if_not(x):
    return list(x) if is_sequence(x) else [x]


def ensure_len(x, size):
    x = list_if_not(x)
    assert len(x) == size or len(x) == 1
    if len(x) == 1:
        x = x * size
    return x


def map_until(f, x, cond=None):
    cond = cond or not_seq_or_dict
    if cond(x):
        y = f(x)
    elif isinstance(x, dict):
        y = dict()
        for k, xx in x.items():
            y[k] = map_until(f, xx, cond)
    elif is_sequence(x):
        y = [map_until(f, xx, cond) for xx in x]
        y = type(x)(y)
    else:
        raise ValueError(type(x))
    return y


def all_same(xs):
    for x in xs:
        if x != xs[0]:
            return False
    return True


def zip_strict(*iterables: Iterable[Any]) -> Generator[Tuple[Any, ...], None, None]:
    """Like the built-in zip, but raise an error if the iterables do not all have the same lengths.

    Parameters
    ----------
    Iterable[Any]
        Iterables to be zipped.

    Yields
    ------
    Tuple[Any, ...]
        Zipped elems from the iterables.

    Raises
    ------
    ValueError
        If the iterables do not have the same length.
    """
    sentinel = object()
    for item in itertools.zip_longest(*iterables, fillvalue=sentinel):
        if any(x is sentinel for x in item):
            raise ValueError("iterables did not have the same lengths")

        yield item
