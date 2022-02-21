"""Image class to hold a medical image
Taken from torchio and modified to use eager
load and use np.ndarray instead of torch.tensor
https://github.com/fepegar/torchio
Author: Jacob Reinhold <jcreinhold@gmail.com>
"""

from __future__ import annotations

__all__ = ["Image"]

import builtins
import typing
import zipfile

import numpy as np
import numpy.typing as npt

try:
    import nibabel as nib
    import SimpleITK as sitk
except ImportError as imp_exn:
    imp_exn_msg = f"NiBabel and SimpleITK must be installed to use {__name__}."
    raise ImportError(imp_exn_msg) from imp_exn

import pymedio.base as miob
import pymedio.dicom as miod
import pymedio.functional as miof
import pymedio.typing as miot


class Image(miob.BasicImage[typing.Any, miot.DType]):  # type: ignore[type-arg]
    @property
    def repr_properties(self) -> builtins.list[builtins.str]:
        props = super().repr_properties
        props += [f"orientation: {''.join(self.orientation)}+"]
        return props

    @property
    def orientation(self) -> builtins.tuple[builtins.str, builtins.str, builtins.str]:
        """Orientation codes."""
        codes: builtins.tuple[builtins.str, builtins.str, builtins.str]
        codes = nib.aff2axcodes(self.affine)
        return codes

    @property
    def bounds(self) -> npt.NDArray[np.float64]:
        """Position of centers of voxels in smallest and largest indices."""
        ini = 0, 0, 0
        fin: np.ndarray = np.asarray(self.shape) - 1
        point_ini = nib.affines.apply_affine(self.affine, ini)
        point_fin = nib.affines.apply_affine(self.affine, fin)
        return np.asarray((point_ini, point_fin))

    def axis_name_to_index(self, axis: builtins.str) -> builtins.int:
        # Top and bottom are used for the vertical 2D axis as the use of
        # Height vs Horizontal might be ambiguous
        if not isinstance(axis, str):
            raise ValueError("Axis must be a string")
        axis = axis[0].upper()

        if axis in "TB":  # Top, Bottom
            return -2
        else:
            try:
                index = self.orientation.index(axis)
            except ValueError:
                index = self.orientation.index(self.flip_axis(axis))
            # Return negative indices so that it does not matter whether we
            # refer to spatial dimensions or not
            index = -3 + index
            return index

    @staticmethod
    def flip_axis(axis: builtins.str) -> builtins.str:
        labels = "LRPAISTBDV"
        first = labels[::2]
        last = labels[1::2]
        flip_dict = {a: b for a, b in zip(first + last, last + first)}
        axis = axis[0].upper()
        flipped_axis = flip_dict.get(axis)
        if flipped_axis is None:
            values = ", ".join(labels)
            message = f"Axis not understood. Please use one of: {values}"
            raise ValueError(message)
        return flipped_axis

    def get_bounds(self) -> miot.Bounds:
        """Get minimum and maximum world coordinates occupied by the image."""
        first_index = 3 * (-0.5,)
        last_index: np.ndarray = np.asarray(self.shape) - 0.5
        first_point = nib.affines.apply_affine(self.affine, first_index)
        last_point = nib.affines.apply_affine(self.affine, last_index)
        array: np.ndarray = np.asarray((first_point, last_point))
        bounds_x, bounds_y, bounds_z = array.T.tolist()
        return bounds_x, bounds_y, bounds_z

    def save(self, path: miot.PathLike, *, squeeze: builtins.bool = True) -> None:
        miof.write_image(np.array(self), self.affine, path, squeeze=squeeze)

    def to_filename(self, path: miot.PathLike) -> None:
        self.save(path, squeeze=False)

    def get_center(self, lps: builtins.bool = False) -> miot.TripletFloat:
        """Get image center in RAS+ or LPS+ coordinates"""
        size: np.ndarray = np.asarray(self.shape)
        center_index = (size - 1) / 2
        r, a, s = nib.affines.apply_affine(self.affine, center_index)
        return (-r, -a, s) if lps else (r, a, s)

    @classmethod
    def from_path(
        cls: typing.Type[Image],
        path: miot.PathLike,
        *,
        dtype: typing.Type[miot.DType] | None = None,
        eager: builtins.bool = True,
    ) -> Image[miot.DType]:
        data, affine = miof.read_image(path, dtype=dtype, eager=eager)
        return cls(data=data, affine=affine)

    @classmethod
    def from_stream(
        cls: typing.Type[Image],
        data_stream: typing.IO,
        *,
        dtype: typing.Type[miot.DType] | None = None,
        gzipped: builtins.bool = False,
        image_class: miof.NibabelImageClass | None = None,
    ) -> Image[miot.DType]:
        data, affine = miof.read_image_from_stream(
            data_stream, dtype=dtype, gzipped=gzipped, image_class=image_class
        )
        return cls(data=data, affine=affine)

    @classmethod
    def from_zipped_stream(
        cls: typing.Type[Image],
        data_stream: typing.IO,
        *,
        dtype: typing.Type[miot.DType] | None = None,
        gzipped: builtins.bool = False,
        image_class: miof.NibabelImageClass | None = None,
        **zip_kwargs: typing.Any,
    ) -> Image[miot.DType]:
        with zipfile.ZipFile(data_stream, "r", **zip_kwargs) as zf:
            names = [name for name in zf.namelist() if not name.endswith("/")]
            if (n := len(names)) != 1:
                msg = f"{n} files in zipped archive. This constructor requires only 1."
                raise RuntimeError(msg)
            name = names[0]
            with zf.open(name, mode="r") as f:
                return cls.from_stream(
                    typing.cast(typing.BinaryIO, f),
                    dtype=dtype,
                    gzipped=gzipped,
                    image_class=image_class,
                )

    @classmethod
    def from_sitk(
        cls: typing.Type[Image],
        sitk_image: sitk.Image,
        *,
        dtype: typing.Type[miot.DType] | None = None,
    ) -> Image:
        data, affine = miof.sitk_to_array(sitk_image, dtype=dtype)
        return cls(data=data, affine=affine)

    @classmethod
    def from_dicom_image(
        cls: typing.Type[Image], dicom_image: miod.DICOMImage
    ) -> Image:
        return cls(data=dicom_image, affine=dicom_image.affine)

    @classmethod
    def from_dicom_zipped_stream(
        cls: typing.Type[Image],
        data_stream: typing.IO,
        *,
        max_nonuniformity: builtins.float = 5e-4,
        fail_outside_max_nonuniformity: builtins.bool = True,
        remove_anomalous_images: builtins.bool = True,
        encryption_key: builtins.bytes | builtins.str | None = None,
        rescale: builtins.bool | None = None,
        rescale_dtype: typing.Type[miot.DType] | None = None,
    ) -> Image[miot.DType]:
        dicom_image = miod.DICOMImage.from_zipped_stream(
            data_stream,
            max_nonuniformity=max_nonuniformity,
            fail_outside_max_nonuniformity=fail_outside_max_nonuniformity,
            remove_anomalous_images=remove_anomalous_images,
            encryption_key=encryption_key,
            rescale=rescale,
            rescale_dtype=rescale_dtype,
        )
        return cls.from_dicom_image(dicom_image)

    def to_sitk(self, **kwargs: builtins.bool) -> sitk.Image:
        """Get the image as an instance of :class:`sitk.Image`."""
        return miof.array_to_sitk(np.array(self), self.affine, **kwargs)

    def to_nibabel(self) -> nib.Nifti1Image:
        return nib.Nifti1Image(np.array(self), self.affine)
