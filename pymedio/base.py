"""Base image class for image containers
Author: Jacob Reinhold <jcreinhold@gmail.com>
"""

from __future__ import annotations

__all__ = ["BasicImage"]

import builtins
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

    _affine: npt.NDArray

    def __new__(
        cls: typing.Type[_Image],
        data: npt.ArrayLike,
        affine: npt.NDArray | None = None,
        copy: builtins.bool = True,
    ) -> _Image:
        """CAUTION: if copy false, some np funcs unexpectedly mutate original array"""
        obj = cls._check_data(data, copy=copy).view(cls)
        obj._affine = cls._check_affine(affine)
        obj._affine.flags.writeable = False
        return obj

    def __array_finalize__(self, obj: builtins.object) -> None:
        if obj is None:
            return
        self._affine = self._check_affine(getattr(obj, "_affine", None))
        self._affine.flags.writeable = False

    @property
    def repr_properties(self) -> typing.List[builtins.str]:
        return [
            f"shape: {self.shape}",
            f"spacing: {self.get_spacing_string()}",
            f"dtype: {self.dtype.name}",
        ]

    def __repr__(self) -> builtins.str:
        if self.ndim == 0:
            return str(self.item())
        properties = "; ".join(self.repr_properties)
        string = f"{self.__class__.__name__}({properties})"
        return string

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: _UfuncMethod,
        *inputs: typing.Any,
        out: typing.Any = None,
        **kwargs: typing.Any,
    ) -> typing.Any:

        affine = self.affine

        ufunc_args = []
        for input_ in inputs:
            if isinstance(input_, BasicImage):
                ufunc_args.append(input_.view(np.ndarray))
            else:
                ufunc_args.append(input_)

        outputs = out
        if outputs:
            out_args = []
            for output in outputs:
                if isinstance(output, BasicImage):
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs["out"] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        results = super().__array_ufunc__(ufunc, method, *ufunc_args, **kwargs)

        if method == "at":
            if isinstance(inputs[0], BasicImage):
                inputs[0].affine = affine
            return

        if ufunc.nout == 1:
            results = (results,)

        results = tuple(
            (self.__class__(np.asarray(result), affine) if output is None else output)
            for result, output in zip(results, outputs)
        )

        return results[0] if len(results) == 1 else results

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
    def _check_data(data: npt.ArrayLike, copy: builtins.bool) -> npt.NDArray:
        data = np.array(data, copy=copy)
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
        np.savez_compressed(file, data=np.array(self), affine=self.affine)

    @classmethod
    def from_npz(
        cls, file: miot.PathLike | typing.BinaryIO, **np_load_kwargs
    ) -> _Image:
        _data = np.load(file, **np_load_kwargs)
        return cls(_data["data"], affine=_data["affine"])

    def torch_compatible(self) -> npt.NDArray:
        return miou.ensure_4d(miou.check_uint_to_int(np.array(self)))
