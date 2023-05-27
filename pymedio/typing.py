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

import collections.abc
import os
import typing

import numpy as np
import numpy.typing as npt

DType = typing.TypeVar("DType", bound=np.generic, covariant=True)
T = typing.TypeVar("T", bound=npt.NBitBase)

Float = typing.Union[np.floating[T], float]
Int = typing.Union[np.integer[T], int]

# https://www.python.org/dev/peps/pep-0519/#provide-specific-type-hinting-support
Bound = tuple[Float[T], Float[T]]
Bounds = tuple[Bound[T], Bound[T], Bound[T]]
DataAffine = tuple[npt.NDArray[DType], npt.NDArray[np.float64]]
Direction2D = tuple[Float[T], Float[T], Float[T], Float[T]]
Direction3D = tuple[Float[T], Float[T], Float[T], Float[T], Float[T], Float[T]]
Direction = typing.Union[Direction2D[T], Direction3D[T]]
PathLike = typing.Union[str, os.PathLike]
Shape = tuple[Int[T], ...]
ShapeLike = typing.Union[
    typing.SupportsIndex, collections.abc.Sequence[typing.SupportsIndex]
]
TripletFloat = tuple[Float[T], Float[T], Float[T]]


class SupportsArray(typing.Protocol[DType]):
    def __array__(self) -> npt.NDArray[DType]:
        ...
