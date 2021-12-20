"""Base image class for image containers
Author: Jacob Reinhold <jcreinhold@gmail.com>
"""

from __future__ import annotations

__all__ = ["ImageBase"]

import builtins
import typing
import warnings

import numpy as np
import numpy.typing as npt

import medio.typing as miot


class ImageBase:
    def __init__(
        self,
        data: npt.NDArray,
        affine: typing.Optional[npt.NDArray] = None,
    ):
        self._data = self._parse_array(data)
        self._affine = self._parse_affine(affine)

    def __repr__(self) -> builtins.str:
        _properties = [
            f"shape: {self.shape}",
            f"dtype: {self.data.dtype.__name__}",  # type: ignore[attr-defined]
        ]
        properties = "; ".join(_properties)
        string = f"{self.__class__.__name__}({properties})"
        return string

    def __array__(self, dtype: typing.Optional[npt.DTypeLike] = None) -> npt.NDArray:
        return np.asanyarray(self.data, dtype=dtype)

    @property
    def data(self) -> npt.NDArray:
        return self._data

    @data.setter
    def data(self, new_data: npt.NDArray) -> None:
        self._data = self._parse_array(new_data)

    @property
    def affine(self) -> npt.NDArray:
        return self._affine

    @affine.setter
    def affine(self, new_affine: npt.NDArray) -> None:
        self._affine = self._parse_affine(new_affine)

    @property
    def shape(self) -> miot.Shape:
        return self.data.shape

    @property
    def dtype(self) -> npt.DTypeLike:
        return self.data.dtype

    @staticmethod
    def _parse_array(array: npt.NDArray) -> npt.NDArray:
        if np.isnan(array).any():
            warnings.warn("NaNs found in data", RuntimeWarning)
        if any(d == 0 for d in array.shape):
            msg = f"Data must have all non-zero dimensions. Got {array.shape}."
            raise ValueError(msg)
        return array

    @staticmethod
    def _parse_affine(affine: typing.Optional[npt.NDArray]) -> npt.NDArray:
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
