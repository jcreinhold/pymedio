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

import numpy as np
import numpy.typing as npt

DType = typing.TypeVar("DType", bound=np.generic, covariant=True)
T = typing.TypeVar("T", bound=npt.NBitBase)

Float = typing.Union[np.floating[T], builtins.float]
Int = typing.Union[np.integer[T], builtins.int]

# https://www.python.org/dev/peps/pep-0519/#provide-specific-type-hinting-support
Bound = typing.Tuple[Float[T], Float[T]]
Bounds = typing.Tuple[Bound[T], Bound[T], Bound[T]]
DataAffine = typing.Tuple[npt.NDArray[DType], npt.NDArray[np.float64]]
Direction2D = typing.Tuple[Float[T], Float[T], Float[T], Float[T]]
Direction3D = typing.Tuple[Float[T], Float[T], Float[T], Float[T], Float[T], Float[T]]
Direction = typing.Union[Direction2D[T], Direction3D[T]]
PathLike = typing.Union[builtins.str, os.PathLike]
Shape = typing.Tuple[Int[T], ...]
ShapeLike = typing.Union[typing.SupportsIndex, typing.Sequence[typing.SupportsIndex]]
TripletFloat = typing.Tuple[Float[T], Float[T], Float[T]]


class SupportsArray(typing.Protocol):
    def __array__(self) -> npt.NDArray:
        ...
