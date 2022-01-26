"""DICOM-specific functions
inspired by dicom-numpy: https://github.com/innolitics/dicom-numpy
Author: Jacob Reinhold <jcreinhold@gmail.com>
"""

from __future__ import annotations

__all__ = [
    "DICOMDir",
    "DICOMImage",
]

import builtins
import dataclasses
import functools
import io
import logging
import math
import operator
import pathlib
import typing
import warnings
import zipfile

import numpy as np
import numpy.typing as npt

try:
    import pydicom
except (ModuleNotFoundError, ImportError) as imp_exn:
    msg = f"pydicom must be installed to use {__name__}."
    raise RuntimeError(msg) from imp_exn

import medio.base as miob
import medio.exceptions as mioe
import medio.typing as miot
import medio.utils as miou

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Cosines:
    row: npt.NDArray
    column: npt.NDArray
    slice: npt.NDArray

    @classmethod
    def from_orientation(
        cls: typing.Type[Cosines],
        image_orientation: typing.Sequence[builtins.float] | npt.NDArray,
    ) -> Cosines:
        row_cosine = np.asanyarray(image_orientation[:3], dtype=np.float64)
        column_cosine = np.asanyarray(image_orientation[3:], dtype=np.float64)
        slice_cosine = np.cross(row_cosine, column_cosine)
        cosines = cls(row_cosine, column_cosine, slice_cosine)
        cosines.validate()
        cosines.writable(False)
        return cosines

    def validate(self) -> None:
        dot_prod = float(np.dot(self.row, self.column).item())
        err_msg_dp = f"Non-orthogonal direction cosines: {self.row}, {self.column}"
        warn_msg_dp = f"Direction cosines aren't quite ortho: {self.row}, {self.column}"
        self._validate_value(dot_prod, err_msg_dp, warn_msg_dp, self._almost_zero)

        row_cosine_norm = float(np.linalg.norm(self.row).item())
        err_msg_rn = f"Row direction cosine's magnitude is not 1: {self.row}"
        warn_msg_rn = f"Row direction cosine's magnitude not quite 1: {self.row}"
        self._validate_value(row_cosine_norm, err_msg_rn, warn_msg_rn, self._almost_one)

        col_cosine_norm = float(np.linalg.norm(self.column).item())
        err_msg_cn = f"Column direction cosine's magnitude is not 1: {self.column}"
        warn_msg_cn = f"Column direction cosine's magnitude not quite 1: {self.column}"
        self._validate_value(col_cosine_norm, err_msg_cn, warn_msg_cn, self._almost_one)

    def writable(self, value: builtins.bool, /) -> None:
        self.row.flags.writeable = value
        self.column.flags.writeable = value
        self.slice.flags.writeable = value

    @staticmethod
    def _validate_value(
        value: builtins.float,
        err_msg: builtins.str,
        warn_msg: builtins.str,
        check_func: typing.Callable[..., builtins.bool],
    ) -> None:
        if not check_func(value, atol=1e-4):
            raise mioe.DicomImportException(value, err_msg)
        elif not check_func(value, atol=1e-8):
            warnings.warn(warn_msg)

    @staticmethod
    def _almost_zero(value: builtins.float, *, atol: builtins.float) -> builtins.bool:
        return math.isclose(value, 0.0, abs_tol=atol)

    @staticmethod
    def _almost_one(value: builtins.float, *, atol: builtins.float) -> builtins.bool:
        return math.isclose(value, 1.0, abs_tol=atol)


@dataclasses.dataclass(frozen=True)
class SortedSlices:
    slices: typing.Tuple[pydicom.Dataset, ...]
    indices: typing.Tuple[builtins.int, ...]
    positions: typing.Tuple[builtins.float, ...]
    cosines: Cosines

    def __len__(self) -> builtins.int:
        length = len(self.slices)
        assert length == len(self.indices)
        assert length == len(self.positions)
        return length

    @classmethod
    def from_datasets(
        cls: typing.Type[SortedSlices], slice_datasets: typing.Sequence[pydicom.Dataset]
    ) -> SortedSlices:
        """sort list of pydicom datasets into the correct order"""
        assert slice_datasets, "slice_datasets empty"
        image_orientation = np.asanyarray(
            slice_datasets[0].ImageOrientationPatient, dtype=np.float64
        )
        cosines = Cosines.from_orientation(image_orientation)
        cs = cosines.slice
        positions = [np.dot(cs, d.ImagePositionPatient).item() for d in slice_datasets]
        _sorted = typing.cast(
            typing.Iterable[typing.Tuple[builtins.int, builtins.float]],
            sorted(enumerate(positions), key=operator.itemgetter(1)),
        )
        sorted_indices, sorted_positions = miou.unzip(_sorted)
        sorted_slice_datasets = tuple(slice_datasets[i] for i in sorted_indices)
        return cls(
            slices=sorted_slice_datasets,
            indices=sorted_indices,
            positions=sorted_positions,
            cosines=cosines,
        )

    def check_nonuniformity(
        self,
        *,
        max_nonuniformity: builtins.float = 5e-4,
        fail_outside_max_nonuniformity: builtins.bool = True,
    ) -> None:
        if len(self) > 1:
            diffs = np.diff(self.positions)
            if not np.allclose(diffs, diffs[0], atol=0.0, rtol=max_nonuniformity):
                msg = f"The slice spacing is non-uniform. Slice spacings:\n{diffs}"
                if fail_outside_max_nonuniformity:
                    raise mioe.OutsideMaxNonUniformity(msg)
                else:
                    warnings.warn(msg)

    @functools.cached_property
    def patient_position(self) -> npt.NDArray:
        return np.asanyarray(self.slices[0].ImagePositionPatient, dtype=np.float64)

    @functools.cached_property
    def slice_spacing(self) -> builtins.float:
        spacing: builtins.float
        if len(self) > 1:
            slice_positions_diffs = np.diff(sorted(self.positions))
            spacing = float(np.median(slice_positions_diffs).item())
        elif len(self) == 1:
            spacing = float(getattr(self.slices[0], "SpacingBetweenSlices", 0))
        else:
            raise RuntimeError("slice_datasets must contain at least one dicom image")
        return spacing

    @functools.cached_property
    def affine(self) -> npt.NDArray:
        row_spacing, column_spacing = self.slices[0].PixelSpacing
        transform = np.identity(4, dtype=np.float64)
        slice_spacing = self.slice_spacing or 1.0
        transform[:3, 0] = self.cosines.row * column_spacing
        transform[:3, 1] = self.cosines.column * row_spacing
        transform[:3, 2] = self.cosines.slice * slice_spacing
        transform[:3, 3] = self.patient_position
        _flipxy_44 = miou.flipxy_44()
        transform_ras: npt.NDArray = np.dot(_flipxy_44, transform)
        return transform_ras


@dataclasses.dataclass(frozen=True)
class DICOMDir:
    slices: typing.Tuple[pydicom.Dataset, ...]
    positions: typing.Tuple[builtins.float, ...]
    spacing: builtins.float
    affine: npt.NDArray
    paths: typing.Tuple[miot.PathLike, ...] | None = None

    def __len__(self) -> builtins.int:
        length = len(self.slices)
        if self.paths is not None:
            assert length == len(list(self.paths))
        return length

    @classmethod
    def from_datasets(
        cls: typing.Type[DICOMDir],
        datasets: typing.Sequence[pydicom.Dataset],
        *,
        paths: typing.Sequence[miot.PathLike] | None = None,
        max_nonuniformity: builtins.float = 5e-4,
        fail_outside_max_nonuniformity: builtins.bool = True,
        remove_anomalous_images: builtins.bool = True,
    ) -> DICOMDir:
        if not datasets:
            msg = "Must provide at least one image DICOM dataset"
            raise mioe.DicomImportException(msg)
        sorted_slices = SortedSlices.from_datasets(datasets)
        sorted_slices.check_nonuniformity(
            max_nonuniformity=max_nonuniformity,
            fail_outside_max_nonuniformity=fail_outside_max_nonuniformity,
        )
        positions = tuple(sorted(sorted_slices.positions))
        dicom_dir = cls(
            slices=sorted_slices.slices,
            positions=positions,
            spacing=sorted_slices.slice_spacing,
            affine=sorted_slices.affine,
            paths=None if paths is None else tuple(paths),
        )
        if remove_anomalous_images:
            dicom_dir = dicom_dir.remove_anomalous_image_paths()
        dicom_dir.writable(False)
        return dicom_dir

    @classmethod
    def from_path(
        cls: typing.Type[DICOMDir],
        dicom_path: miot.PathLike | typing.Iterable[miot.PathLike],
        *,
        max_nonuniformity: builtins.float = 5e-4,
        fail_outside_max_nonuniformity: builtins.bool = True,
        remove_anomalous_images: builtins.bool = True,
        defer_size: builtins.str | builtins.int | None = "1 KB",
    ) -> DICOMDir:
        paths: typing.Tuple[miot.PathLike, ...]
        if (
            isinstance(dicom_path, (builtins.str, pathlib.Path))
            and (_dcm_dir := pathlib.Path(dicom_path)).is_dir()
        ):
            paths = tuple(sorted(_dcm_dir.glob("*.dcm")))
        elif (
            not isinstance(dicom_path, (builtins.str, pathlib.Path))
            and miou.is_iterable(dicom_path)
            and all(str(p).endswith(".dcm") for p in dicom_path)  # type: ignore[union-attr]  # noqa: E501
        ):
            paths = tuple(dicom_path)  # type: ignore[arg-type]
        else:
            raise ValueError("dicom_dir must be path to a dir. or a list of dcm paths")
        images = tuple(pydicom.dcmread(path, defer_size=defer_size) for path in paths)
        return cls.from_datasets(
            typing.cast(typing.List[pydicom.Dataset], images),
            paths=paths,
            max_nonuniformity=max_nonuniformity,
            fail_outside_max_nonuniformity=fail_outside_max_nonuniformity,
            remove_anomalous_images=remove_anomalous_images,
        )

    @classmethod
    def from_zipped_stream(
        cls: typing.Type[DICOMDir],
        data_stream: typing.BinaryIO,
        *,
        max_nonuniformity: builtins.float = 5e-4,
        fail_outside_max_nonuniformity: builtins.bool = True,
        remove_anomalous_images: builtins.bool = True,
        encryption_key: builtins.bytes | builtins.str | None = None,
        **zip_kwargs: typing.Any,
    ) -> DICOMDir:
        if encryption_key is not None:
            try:
                import cryptography.fernet as crypto
            except (ModuleNotFoundError, ImportError) as crypto_imp_exn:
                msg = "If encryption key provided, cryptography package required."
                raise RuntimeError(msg) from crypto_imp_exn
            fernet = crypto.Fernet(encryption_key)
            data_stream.seek(0)
            data_stream = io.BytesIO(fernet.decrypt(data_stream.read()))
        with zipfile.ZipFile(data_stream, mode="r", **zip_kwargs) as zf:
            datasets = cls.dicom_datasets_from_zip(zf)
        return cls.from_datasets(
            datasets,
            max_nonuniformity=max_nonuniformity,
            fail_outside_max_nonuniformity=fail_outside_max_nonuniformity,
            remove_anomalous_images=remove_anomalous_images,
        )

    @staticmethod
    def dicom_datasets_from_zip(
        zip_file: zipfile.ZipFile,
    ) -> typing.List[pydicom.Dataset]:
        datasets: typing.List[pydicom.Dataset] = []
        for name in zip_file.namelist():
            if name.endswith("/"):
                continue  # skip directories
            with zip_file.open(name, mode="r") as f:
                try:
                    datasets.append(pydicom.dcmread(f))  # type: ignore[arg-type]
                except pydicom.errors.InvalidDicomError as e:
                    msg = f"Skipping invalid DICOM file '{name}': {e}"
                    logger.info(msg)
        if not datasets:
            msg = "Zipfile does not contain any valid DICOM files"
            raise mioe.DicomImportException(msg)
        return datasets

    def remove_anomalous_image_paths(self) -> DICOMDir:
        orientations = [tuple(img.ImageOrientationPatient) for img in self.slices]
        unique_orientations, counts = np.unique(
            orientations, axis=0, return_counts=True
        )
        most_common_orientation = tuple(unique_orientations[np.argmax(counts)])
        paths = ([None] * len(self)) if (no_paths := self.paths is None) else self.paths
        out = [
            (img, pos, path)
            for img, pos, path, o in zip(
                self.slices, self.positions, paths, orientations
            )
            if o == most_common_orientation
        ]
        new_images, new_positions, new_image_paths = miou.unzip(out)
        return DICOMDir(
            slices=new_images,
            positions=new_positions,
            spacing=self.spacing,
            affine=self.affine,
            paths=None if no_paths else new_image_paths,
        )

    def validate(self) -> None:
        invariant_properties = frozenset(
            [
                "Modality",
                "SOPClassUID",
                "SeriesInstanceUID",
                "Rows",
                "Columns",
                "SamplesPerPixel",
                "PixelSpacing",
                "PixelRepresentation",
                "BitsAllocated",
            ]
        )
        for property_name in invariant_properties:
            self._slice_attribute_equal(property_name)
        self._slice_ndarray_attribute_almost_equal("ImageOrientationPatient", atol=1e-5)

    def writable(self, value: builtins.bool, /) -> None:
        self.affine.flags.writeable = value

    def _slice_attribute_equal(self, property_name: builtins.str) -> None:
        initial_value = getattr(self.slices[0], property_name, None)
        for dataset in self.slices[1:]:
            value = getattr(dataset, property_name, None)
            if value != initial_value:
                msg = "All slices must have the same value for "
                msg += f"'{property_name}': {value} != {initial_value}"
                raise mioe.DicomImportException(msg)

    def _slice_ndarray_attribute_almost_equal(
        self,
        property_name: builtins.str,
        *,
        atol: builtins.float,
    ) -> None:
        initial_value: npt._SupportsArray | None
        initial_value = getattr(self.slices[0], property_name, None)
        for dataset in self.slices[1:]:
            value: npt._SupportsArray | None
            value = getattr(dataset, property_name, None)
            if value is None or initial_value is None:
                msg = f"All slices must contain the attribute {property_name}"
                raise mioe.DicomImportException(msg)
            if not np.allclose(value, initial_value, atol=atol):
                msg = "All slices must have the same value for "
                msg += f"'{property_name}' within '{atol}': {value} != {initial_value}"
                raise mioe.DicomImportException(msg)

    @staticmethod
    def _is_dicomdir(dataset: pydicom.Dataset) -> builtins.bool:
        media_sop_class: builtins.str | None
        media_sop_class = getattr(dataset, "MediaStorageSOPClassUID", None)
        result: builtins.bool = media_sop_class == "1.2.840.10008.1.3.10"
        return result


class DICOMImage(miob.ImageBase):
    @classmethod
    def from_dicomdir(
        cls: typing.Type[DICOMImage],
        dicom_dir: DICOMDir,
        *,
        rescale: builtins.bool | None = None,
        rescale_dtype: npt.DTypeLike = np.float32,
        order: typing.Literal["F", "C"] | None = None,
    ) -> DICOMImage:
        data = cls._merge_slice_pixel_arrays(
            dicom_dir.slices, rescale=rescale, rescale_dtype=rescale_dtype, order=order
        )
        return cls(data=data, affine=dicom_dir.affine)

    @classmethod
    def from_path(
        cls: typing.Type[DICOMImage],
        dicom_path: miot.PathLike | typing.Iterable[miot.PathLike],
        *,
        max_nonuniformity: builtins.float = 5e-4,
        fail_outside_max_nonuniformity: builtins.bool = True,
        remove_anomalous_images: builtins.bool = True,
        rescale: builtins.bool | None = None,
        rescale_dtype: npt.DTypeLike = np.float32,
        order: typing.Literal["F", "C"] | None = None,
    ) -> DICOMImage:
        dicomdir = DICOMDir.from_path(
            dicom_path,
            max_nonuniformity=max_nonuniformity,
            fail_outside_max_nonuniformity=fail_outside_max_nonuniformity,
            remove_anomalous_images=remove_anomalous_images,
        )
        return cls.from_dicomdir(
            dicomdir, rescale=rescale, rescale_dtype=rescale_dtype, order=order
        )

    @classmethod
    def from_zipped_stream(
        cls: typing.Type[DICOMImage],
        data_stream: typing.BinaryIO,
        *,
        max_nonuniformity: builtins.float = 5e-4,
        fail_outside_max_nonuniformity: builtins.bool = True,
        remove_anomalous_images: builtins.bool = True,
        encryption_key: builtins.bytes | builtins.str | None = None,
        rescale: builtins.bool | None = None,
        rescale_dtype: npt.DTypeLike = np.float32,
        order: typing.Literal["F", "C"] | None = None,
        **zip_kwargs: typing.Any,
    ) -> DICOMImage:
        dicomdir = DICOMDir.from_zipped_stream(
            data_stream,
            max_nonuniformity=max_nonuniformity,
            fail_outside_max_nonuniformity=fail_outside_max_nonuniformity,
            remove_anomalous_images=remove_anomalous_images,
            encryption_key=encryption_key,
            **zip_kwargs,
        )
        return cls.from_dicomdir(
            dicomdir, rescale=rescale, rescale_dtype=rescale_dtype, order=order
        )

    @classmethod
    def _merge_slice_pixel_arrays(
        cls: typing.Type[DICOMImage],
        slices: typing.Sequence[pydicom.Dataset],
        *,
        rescale: builtins.bool | None = None,
        rescale_dtype: npt.DTypeLike = np.float32,
        order: typing.Literal["F", "C"] | None = None,
    ) -> npt.NDArray:
        if rescale is None:
            rescale = any(cls._requires_rescaling(d) for d in slices)

        first_dataset = slices[0]
        slice_dtype = first_dataset.pixel_array.dtype
        slice_shape = first_dataset.pixel_array.T.shape
        slice_order = "F" if first_dataset.pixel_array.T.flags.f_contiguous else "C"
        num_slices = len(slices)

        voxels_shape = slice_shape + (num_slices,)
        voxels_dtype = rescale_dtype if rescale else slice_dtype
        voxels = np.empty(
            voxels_shape,
            dtype=voxels_dtype,
            order=typing.cast(typing.Literal["F", "C"], slice_order),
        )

        for k, dataset in enumerate(slices):
            pixel_array = dataset.pixel_array.T.astype(voxels_dtype)
            if rescale:
                slope = float(getattr(dataset, "RescaleSlope", 1.0))
                intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
                if slope != 1.0:
                    pixel_array *= slope
                if intercept != 0.0:
                    pixel_array += intercept
            voxels[..., k] = pixel_array

        if order is not None:
            if order == "C":
                voxels = np.ascontiguousarray(voxels)
            elif order == "F":
                voxels = np.asfortranarray(voxels)
            else:
                msg = f"If order given, must be either 'F' or 'C'. Got {order}."
                raise ValueError(msg)

        assert voxels.dtype == voxels_dtype
        return voxels

    @staticmethod
    def _requires_rescaling(dataset: pydicom.Dataset) -> builtins.bool:
        return hasattr(dataset, "RescaleSlope") or hasattr(dataset, "RescaleIntercept")
