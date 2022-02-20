"""Base image class for image containers
Author: Jacob Reinhold <jcreinhold@gmail.com>
"""

from __future__ import annotations

__all__ = ["BasicImage"]

import builtins
import numbers
import typing
import warnings

import numpy as np
import numpy.typing as npt

import pymedio.typing as miot
import pymedio.utils as miou

_UfuncMethod = typing.Literal[
    "__call__", "reduce", "reduceat", "accumulate", "outer", "inner"
]
_Image = typing.TypeVar("_Image", bound="BasicImage")


class BasicImage(np.ndarray):

    _HANDLED_TYPES = (np.ndarray, numbers.Number)
    _affine: npt.NDArray

    def __new__(
        cls: typing.Type[_Image],
        data: npt.ArrayLike,
        affine: npt.NDArray | None = None,
    ) -> _Image:
        obj = cls._check_data(data).view(cls)
        obj._affine = cls._check_affine(affine)
        obj._affine.flags.writeable = False
        return obj

    def __array_finalize__(self, obj: builtins.object) -> None:
        if obj is None:
            return
        self._affine = self._check_affine(getattr(obj, "_affine", None))
        self._affine.flags.writeable = False

    @property
    def str_properties(self) -> typing.List[builtins.str]:
        return [
            f"shape: {self.shape}",
            f"spacing: {self.get_spacing_string()}",
            f"dtype: {self.dtype.name}",
        ]

    def __str__(self) -> builtins.str:
        properties = "; ".join(self.str_properties)
        string = f"{self.__class__.__name__}({properties})"
        return string

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: _UfuncMethod,
        *inputs: typing.Any,
        **kwargs: typing.Any,
    ) -> typing.Any:
        out = kwargs.get("out", ())
        for x in inputs + out:
            if not isinstance(x, self._HANDLED_TYPES + (self.__class__,)):
                return NotImplemented
        affine = self.affine
        is_cls = lambda y: isinstance(y, self.__class__)  # noqa: E731
        ufunc_args = tuple(x.__array__() if is_cls(x) else x for x in inputs)
        if out:
            kwargs["out"] = tuple(x.__array__() if is_cls(x) else x for x in out)
            if len(out) == 1 and is_cls(out[0]):
                affine = out[0].affine
        result = getattr(ufunc, method)(*ufunc_args, **kwargs)
        if isinstance(result, tuple):
            return tuple(self.__class__(x, affine) for x in result)
        elif method == "at":
            return None
        elif isinstance(result, np.ndarray):
            return typing.cast(_Image, self.__class__(result, affine))
        else:
            return result

    @property
    def affine(self) -> npt.NDArray:
        return self._affine

    @affine.setter
    def affine(self, new_affine: npt.NDArray) -> None:
        self._affine = self._check_affine(new_affine)

    @property
    def direction(self) -> miot.Direction:
        _, _, direction = miou.get_metadata_from_ras_affine(self.affine, lps=False)
        return direction

    @property
    def spacing(self) -> typing.Tuple[builtins.float, ...]:
        """Voxel spacing in mm."""
        _, spacing = miou.get_rotation_and_spacing_from_affine(self.affine)
        return tuple(spacing)

    @property
    def origin(self) -> typing.Tuple[builtins.float, ...]:
        """Center of first voxel in array, in mm."""
        return tuple(self.affine[:3, 3])

    @property
    def memory(self) -> builtins.int:
        """Number of Bytes that the tensor takes in the RAM."""
        mem: builtins.int = np.prod(self.shape) * self.itemsize
        return mem

    def get_spacing_string(self) -> builtins.str:
        strings = [f"{n:.2g}" for n in self.spacing]
        string = f'({", ".join(strings)})'
        return string

    @staticmethod
    def _check_data(data: npt.ArrayLike) -> npt.NDArray:
        data = np.asanyarray(data)
        if np.isnan(data).any():
            warnings.warn("NaNs found in data", RuntimeWarning)
        if any(d == 0 for d in data.shape):
            msg = f"Data must have all non-zero dimensions. Got {data.shape}."
            raise ValueError(msg)
        return data

    @staticmethod
    def _check_affine(affine: npt.NDArray | None) -> npt.NDArray:
        if affine is None:
            return np.eye(4, dtype=np.float64)
        if not isinstance(affine, np.ndarray):
            bad_type = type(affine)
            raise TypeError(f"Affine must be a NumPy array, not {bad_type}")
        if affine.shape != (4, 4):
            bad_shape = affine.shape
            raise ValueError(f"Affine shape must be (4, 4), not {bad_shape}")
        return miou.to_f64(affine)

    def to_npz(self, file: miot.PathLike | typing.BinaryIO) -> None:
        np.savez_compressed(file, data=self.__array__(), affine=self.affine)

    @classmethod
    def from_npz(
        cls, file: miot.PathLike | typing.BinaryIO, **np_load_kwargs
    ) -> _Image:
        _data = np.load(file, **np_load_kwargs)
        return cls(_data["data"], affine=_data["affine"])

    def torch_compatible(self) -> npt.NDArray:
        return miou.ensure_4d(miou.check_uint_to_int(self.__array__()))
