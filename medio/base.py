"""Base image class for image containers
Author: Jacob Reinhold <jcreinhold@gmail.com>
"""

from __future__ import annotations

__all__ = ["ImageBase"]

import builtins
import numbers
import typing
import warnings

import numpy as np
import numpy.lib.mixins
import numpy.typing as npt

import medio.typing as miot
import medio.utils as miou

UfuncMethod = typing.Literal[
    "__call__", "reduce", "reduceat", "accumulate", "outer", "inner"
]


class ImageBase(numpy.lib.mixins.NDArrayOperatorsMixin):

    _HANDLED_TYPES = (np.ndarray, numbers.Number)

    def __init__(
        self,
        data: npt.NDArray,
        affine: typing.Optional[npt.NDArray] = None,
    ):
        self._data = self._check_data(data)
        self._affine = self._check_affine(affine)

    @property
    def _repr_properties(self) -> typing.List[builtins.str]:
        return [
            f"shape: {self.shape}",
            f"spacing: {self.get_spacing_string()}",
            f"dtype: {self.data.dtype.name}",
        ]

    def __repr__(self) -> builtins.str:
        properties = "; ".join(self._repr_properties)
        string = f"{self.__class__.__name__}({properties})"
        return string

    def __array__(self, dtype: typing.Optional[npt.DTypeLike] = None) -> npt.NDArray:
        return np.asanyarray(self.data, dtype=dtype)

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: UfuncMethod,
        *inputs: typing.Any,
        **kwargs: typing.Any,
    ) -> typing.Any:
        """https://numpy.org/doc/stable/reference/generated/numpy.lib.mixins.NDArrayOperatorsMixin.html"""
        out = kwargs.get("out", ())
        for x in inputs + out:
            if not isinstance(x, self._HANDLED_TYPES + (self.__class__,)):
                return NotImplemented
        is_cls = lambda y: isinstance(y, self.__class__)
        ufunc_args = tuple(x.data if is_cls(x) else x for x in inputs)
        if out:
            kwargs["out"] = tuple(x.data if is_cls(x) else x for x in out)
        result = getattr(ufunc, method)(*ufunc_args, **kwargs)
        if isinstance(result, tuple):
            return tuple(self.__class__(x, self.affine) for x in result)
        elif method == "at":
            return None
        elif isinstance(result, np.ndarray):
            return self.__class__(result, self.affine)
        else:
            return result  # e.g., result of np.all

    @property
    def data(self) -> npt.NDArray:
        return self._data

    @data.setter
    def data(self, new_data: npt.NDArray) -> None:
        self._data = self._check_data(new_data)

    @property
    def affine(self) -> npt.NDArray:
        return self._affine

    @affine.setter
    def affine(self, new_affine: npt.NDArray) -> None:
        self._affine = self._check_affine(new_affine)

    @property
    def shape(self) -> miot.Shape:
        return self.data.shape

    @property
    def dtype(self) -> npt.DTypeLike:
        return self.data.dtype

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
    def itemsize(self) -> builtins.int:
        """Element size of the data type."""
        return self.data.itemsize

    @property
    def memory(self) -> builtins.float:
        """Number of Bytes that the tensor takes in the RAM."""
        mem: builtins.float = np.prod(self.shape) * self.itemsize
        return mem

    def get_spacing_string(self) -> builtins.str:
        strings = [f"{n:.2f}" for n in self.spacing]
        string = f'({", ".join(strings)})'
        return string

    @staticmethod
    def _check_data(data: npt.NDArray) -> npt.NDArray:
        if np.isnan(data).any():
            warnings.warn("NaNs found in data", RuntimeWarning)
        if any(d == 0 for d in data.shape):
            msg = f"Data must have all non-zero dimensions. Got {data.shape}."
            raise ValueError(msg)
        return data

    @staticmethod
    def _check_affine(affine: typing.Optional[npt.NDArray]) -> npt.NDArray:
        if affine is None:
            return np.eye(4)
        if not isinstance(affine, np.ndarray):
            bad_type = type(affine)
            raise TypeError(f"Affine must be a NumPy array, not {bad_type}")
        if affine.shape != (4, 4):
            bad_shape = affine.shape
            raise ValueError(f"Affine shape must be (4, 4), not {bad_shape}")
        return affine

    def to_npz(self, file: typing.Union[miot.PathLike, builtins.bytes]) -> None:
        np.savez_compressed(file, data=self.data, affine=self.affine)

    @classmethod
    def from_npz(
        cls: typing.Type[ImageBase], file: typing.Union[miot.PathLike, builtins.bytes]
    ) -> ImageBase:
        _data = np.load(file)
        return cls(data=_data["data"], affine=_data["affine"])

    def torch_compatible(self) -> npt.NDArray:
        return miou.ensure_4d(miou.check_uint_to_int(self.data))
