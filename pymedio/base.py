"""Base image class for image containers
Author: Jacob Reinhold <jcreinhold@gmail.com>
"""

from __future__ import annotations

__all__ = ["BasicImage"]

import builtins
import collections.abc
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


class BasicImage(npt.NDArray[miot.DType]):

    _affine: npt.NDArray

    def __new__(
        cls: typing.Type[_Image],
        data: npt.ArrayLike,
        affine: npt.NDArray | None = None,
        *,
        info: builtins.str | npt.NDArray[np.str_] | None = None,
        copy: builtins.bool = False,
    ) -> _Image:
        obj = cls._check_data(data, copy=copy).view(cls)
        obj._affine = cls._check_affine(affine)
        obj._affine.flags.writeable = False
        obj._info = cls._check_info(info)
        obj._info.flags.writeable = False
        return obj

    def __array_finalize__(self, obj: builtins.object) -> None:
        if obj is None:
            return
        self._affine = self._check_affine(getattr(obj, "_affine", None))
        self._affine.flags.writeable = False
        self._info = self._check_info(getattr(obj, "info", None))
        self._info.flags.writeable = False

    @property
    def repr_properties(self) -> builtins.list[builtins.str]:
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
        info = self.info

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
                inputs[0].info = info
            return

        if ufunc.nout == 1:
            results = (results,)

        results = tuple(
            (
                self.__class__(np.asarray(result), affine, info=info)
                if output is None
                else output
            )
            for result, output in zip(results, outputs)
        )

        return results[0] if len(results) == 1 else results

    @property
    def affine(self) -> npt.NDArray[np.float64]:
        return self._affine

    @affine.setter
    def affine(self, new_affine: npt.NDArray) -> None:
        self._affine = self._check_affine(new_affine)

    @property
    def info(self) -> npt.NDArray[np.str_]:
        return self._info

    @info.setter
    def info(self, new_info: builtins.str | npt.NDArray[np.str_] | None) -> None:
        self._info = self._check_info(new_info)

    @property
    def direction(self) -> miot.Direction:
        _, _, direction = miou.get_metadata_from_ras_affine(self.affine, lps=False)
        return direction

    @property
    def spacing(self) -> builtins.tuple[miot.Float, ...]:
        """Voxel spacing in mm."""
        _, spacing = miou.get_rotation_and_spacing_from_affine(self.affine)
        return tuple(spacing)

    @property
    def origin(self) -> builtins.tuple[miot.Float, ...]:
        """Center of first voxel in array, in mm."""
        return tuple(self.affine[:3, 3])

    @property
    def memory(self) -> miot.Int:
        """Number of bytes that the image array occupies in RAM"""
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
    def _check_affine(affine: npt.NDArray | None) -> npt.NDArray[np.float64]:
        if affine is None:
            return np.eye(4, dtype=np.float64)
        if not isinstance(affine, np.ndarray):
            bad_type = type(affine)
            raise TypeError(f"Affine must be a NumPy array, not {bad_type}")
        if affine.shape != (4, 4):
            bad_shape = affine.shape
            raise ValueError(f"Affine shape must be (4, 4), not {bad_shape}")
        return miou.to_f64(affine)

    @staticmethod
    def _check_info(
        info: builtins.str | npt.NDArray[np.str_] | None,
    ) -> npt.NDArray[np.str_]:
        if info is None:
            info = ""
        return np.asarray(info, dtype=np.str_)

    def to_npz(self, file: miot.PathLike | typing.BinaryIO) -> None:
        np.savez_compressed(
            file, data=np.asarray(self), affine=self.affine, info=self.info
        )

    @classmethod
    def from_npz(
        cls, file: miot.PathLike | typing.BinaryIO, **np_load_kwargs
    ) -> _Image:
        _data = np.load(file, **np_load_kwargs)
        return cls(_data["data"], affine=_data["affine"], info=_data["info"])

    def torch_compatible(self) -> npt.NDArray:
        return miou.ensure_4d(miou.check_uint_to_int(np.asarray(self)))

    def resample_image(self, shape: collections.abc.Sequence[builtins.int]) -> _Image:
        if self.ndim != len(shape):
            raise ValueError("length of 'shape' != number of dimensions in image.")
        if any(s <= 0 for s in shape):
            raise ValueError("All elements of 'shape' must be positive.")
        data = np.asarray(self)
        resolution_scale = np.empty(len(shape), dtype=np.float64)
        for i, (new_s, s) in enumerate(zip(shape, self.shape)):
            resolution_scale[i] = s / new_s

            def interpolate(fp: np.ndarray) -> np.ndarray:
                return np.interp(
                    np.linspace(0, s, dtype=self.dtype, endpoint=False, num=new_s),
                    np.arange(0, s, dtype=self.dtype),
                    fp,
                )

            data = np.apply_along_axis(interpolate, i, data)
        # https://math.stackexchange.com/q/1120209
        rzs = self.affine[:3, :3]  # rotation, scale (zoom), shear
        rotation, zs = np.linalg.qr(rzs)
        scale = zs.diagonal()
        shear = zs * (1.0 / scale)
        new_affine = np.zeros_like(self.affine)
        new_scale = scale * resolution_scale
        new_affine[:3, :3] = (rotation * new_scale) @ shear
        new_affine[:, 3] = self.affine[:, 3]
        return self.__class__(
            data.astype(self.dtype, copy=False), new_affine, info=self.info
        )
