"""medio types for type annotations
Author: Jacob Reinhold <jcreinhold@gmail.com>
"""

from __future__ import annotations

__all__ = [
    "Bounds",
    "DataAffine",
    "Direction",
    "PathLike",
    "Shape",
    "ShapeLike",
    "SupportsArray",
    "TripletFloat",
]

import builtins
import os
import typing

import numpy.typing as npt

# https://www.python.org/dev/peps/pep-0519/#provide-specific-type-hinting-support
Bound = typing.Tuple[builtins.float, builtins.float]
Bounds = typing.Tuple[Bound, Bound, Bound]
DataAffine = typing.Tuple[npt.NDArray, npt.NDArray]
Direction2D = typing.Tuple[
    builtins.float, builtins.float, builtins.float, builtins.float
]
Direction3D = typing.Tuple[
    builtins.float,
    builtins.float,
    builtins.float,
    builtins.float,
    builtins.float,
    builtins.float,
    builtins.float,
    builtins.float,
    builtins.float,
]
Direction = typing.Union[Direction2D, Direction3D]
PathLike = typing.Union[builtins.str, os.PathLike]
Shape = typing.Tuple[builtins.int, ...]
ShapeLike = typing.Union[typing.SupportsIndex, typing.Sequence[typing.SupportsIndex]]
TripletFloat = typing.Tuple[builtins.float, builtins.float, builtins.float]


class SupportsArray(typing.Protocol):
    def __array__(self) -> npt.NDArray:
        ...
