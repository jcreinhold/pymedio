"""DICOM-specific functions
inspired by dicom-numpy: https://github.com/innolitics/dicom-numpy
Author: Jacob Reinhold <jcreinhold@gmail.com>
"""

from __future__ import annotations

__all__ = [
    "DICOMDir",
    "DICOMImage",
    "gather_dicom",
]

import builtins
import collections.abc
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
except ImportError as imp_exn:
    imp_exn_msg = f"pydicom must be installed to use {__name__}."
    raise ImportError(imp_exn_msg) from imp_exn

import pymedio.base as miob
import pymedio.exceptions as mioe
import pymedio.typing as miot
import pymedio.utils as miou

logger = logging.getLogger(__name__)

ORIENTATION_ATOL = 1e-5

DType = typing.TypeVar("DType", bound=np.generic)
T = typing.TypeVar("T")
Datasets = typing.Iterable[pydicom.Dataset]


def _all_float_like(seq: collections.abc.Sequence[builtins.float]) -> builtins.bool:
    return all(isinstance(x, (float, int)) for x in seq)


def gather_dicom(
    dicom_path: miot.PathLike | typing.Iterable[miot.PathLike],
    *,
    defer_size: builtins.str | builtins.int | None = "1 KB",
    extension: builtins.str = ".dcm",
    return_paths: builtins.bool = False,
) -> Datasets | builtins.tuple[Datasets, builtins.tuple[miot.PathLike, ...]]:
    paths: builtins.tuple[miot.PathLike, ...]
    if (
        isinstance(dicom_path, (builtins.str, pathlib.Path))
        and (_dcm_dir := pathlib.Path(dicom_path)).is_dir()
    ):
        paths = tuple(sorted(_dcm_dir.glob(f"*{extension}")))
    elif (
        not isinstance(dicom_path, (builtins.str, pathlib.Path))
        and miou.is_iterable(dicom_path)
        and all(str(p).endswith(extension) for p in dicom_path)  # type: ignore[union-attr]  # noqa: E501
    ):
        paths = tuple(dicom_path)  # type: ignore[arg-type]
    else:
        raise ValueError("dicom_path must be path to a dir. or a list of dcm paths")
    datasets = (pydicom.dcmread(path, defer_size=defer_size) for path in paths)
    return (datasets, paths) if return_paths else datasets


@dataclasses.dataclass(frozen=True)
class Cosines:
    # dicom-numpy -> dicom_numpy/combine_slices.py
    # ITK -> Modules/IO/GDCM/src/itkGDCMImageIO.cxx
    row: npt.NDArray
    column: npt.NDArray
    slice: npt.NDArray

    def __repr__(self) -> builtins.str:
        return f"Cosines(row={self.row}, column={self.column}, slice={self.slice})"

    @classmethod
    def from_orientation(
        cls: typing.Type[Cosines],
        image_orientation: typing.Sequence[builtins.float] | npt.NDArray,
    ) -> Cosines:
        if isinstance(image_orientation, np.ndarray):
            if image_orientation.size != 6 or image_orientation.ndim != 1:
                raise ValueError("image_orientation must be seq. of 1 dim and len=6.")
        elif len(image_orientation) != 6 or not _all_float_like(image_orientation):
            raise ValueError("image_orientation must be seq. floats with len=6.")
        row_cosine = miou.to_f64(image_orientation[:3])
        column_cosine = miou.to_f64(image_orientation[3:])
        slice_cosine: np.ndarray = np.cross(row_cosine, column_cosine)
        cosines = cls(row_cosine, column_cosine, slice_cosine)
        cosines.writable(False)
        cosines.validate()
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
    slices: builtins.tuple[pydicom.Dataset, ...]
    indices: builtins.tuple[builtins.int, ...]  # mapping to get original slice order
    positions: builtins.tuple[builtins.float, ...]
    cosines: Cosines

    def __repr__(self) -> builtins.str:
        return f"SortedSlices(n_slices={len(self)}, cosines={self.cosines!r})"

    def __len__(self) -> builtins.int:
        _len = len(self.slices)
        exn_msg: builtins.list[builtins.str] = []
        if _len != (ind_len := len(self.indices)):
            exn_msg.append(f"num slices {_len} != num indices {ind_len}")
        if _len != (pos_len := len(self.positions)):
            exn_msg.append(f"num slices {_len} != num positions {pos_len}")
        if exn_msg:
            raise RuntimeError(" ".join(exn_msg))
        return _len

    @classmethod
    def from_datasets(
        cls, slice_datasets: typing.Sequence[pydicom.Dataset]
    ) -> SortedSlices:
        """sort list of pydicom datasets into the correct order"""
        if not slice_datasets:
            raise ValueError("slice_datasets empty")
        image_orientation = miou.to_f64(slice_datasets[0].ImageOrientationPatient)
        cosines = Cosines.from_orientation(image_orientation)
        ipps = (miou.to_f64(sd.ImagePositionPatient) for sd in slice_datasets)
        positions = (np.dot(cosines.slice, imp).item() for imp in ipps)
        _sorted = typing.cast(
            typing.Iterable[builtins.tuple[builtins.int, builtins.float]],
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
        missing_slices_cutoff: builtins.float = 1e-1,
    ) -> None:
        if len(self) > 1:
            diffs = np.diff(self.positions)
            warned = False
            if not np.allclose(diffs, diffs[0], atol=0.0, rtol=max_nonuniformity):
                msg = f"The slice spacing is non-uniform. Slice spacings:\n{diffs}"
                if fail_outside_max_nonuniformity:
                    raise mioe.OutsideMaxNonUniformity(msg)
                else:
                    warned = True
                    warnings.warn(msg)
            if not np.allclose(diffs, diffs[0], atol=0.0, rtol=missing_slices_cutoff):
                msg = "There appear to be missing slices."
                if fail_outside_max_nonuniformity:
                    raise mioe.MissingSlicesException(msg)
                elif not warned:
                    warnings.warn(msg)

    def remove_anomalous_slices(
        self,
        *,
        strict_unique_orientation: builtins.bool = True,
        unique_positions: builtins.bool = True,
    ) -> SortedSlices:
        to_float_tuple = lambda xs: tuple(float(x) for x in xs)  # noqa: E731
        orientations = [to_float_tuple(s.ImageOrientationPatient) for s in self.slices]
        if strict_unique_orientation:
            unq_oris: np.ndarray
            unq_oris, counts = np.unique(orientations, axis=0, return_counts=True)
            most_common_orientation = unq_oris[np.argmax(counts)]
        else:
            approx_unique_orientations = self._approx_unique(orientations)
            most_common_orientation = approx_unique_orientations[-1]
        seen_positions = set()
        out = []
        for _slice, idx, pos, o in self._zip_with(orientations):
            if np.allclose(o, most_common_orientation, atol=ORIENTATION_ATOL):
                if unique_positions and pos in seen_positions:
                    logger.debug(f"Slice at index {idx} has a non-unique position.")
                    continue
                out.append((_slice, idx, pos))
                seen_positions.add(pos)
            else:
                logger.debug(f"Slice at index {idx} has a different orientation.")
        new_slices, new_indices, new_positions = miou.unzip(out)
        if (n_removed := (len(self) - len(new_slices))) > 1:
            warnings.warn(f"{n_removed} anomalous images removed.")
        elif n_removed < 0:
            raise RuntimeError("Images added in remove image func. Report error.")
        new_image_orientation = miou.to_f64(most_common_orientation)
        new_cosines = Cosines.from_orientation(new_image_orientation)
        return SortedSlices(
            slices=new_slices,
            indices=new_indices,
            positions=new_positions,
            cosines=new_cosines,
        )

    @functools.cached_property
    def patient_position(self) -> npt.NDArray:
        return miou.to_f64(self.slices[0].ImagePositionPatient)

    @functools.cached_property
    def slice_spacing(self) -> builtins.float:
        spacing: builtins.float
        if len(self) > 1:
            slice_positions_diffs = np.diff(np.sort(self.positions))
            # avg. b/c that's what ITK seems to use, so use for consistency
            spacing = float(np.mean(slice_positions_diffs).item())
        elif len(self) == 1:
            spacing = float(getattr(self.slices[0], "SpacingBetweenSlices", 0))
        else:
            raise RuntimeError("slice_datasets must contain at least one dicom image")
        return spacing

    @functools.cached_property
    def affine(self) -> npt.NDArray:
        row_spacing, column_spacing = self.slices[0].PixelSpacing
        transform: np.ndarray = np.identity(4, dtype=np.float64)
        slice_spacing = self.slice_spacing or 1.0
        transform[:3, 0] = self.cosines.row * column_spacing
        transform[:3, 1] = self.cosines.column * row_spacing
        transform[:3, 2] = self.cosines.slice * slice_spacing
        transform[:3, 3] = self.patient_position
        transform_ras: npt.NDArray = np.dot(miou.flipxy_44(), transform)
        return transform_ras

    @staticmethod
    def _approx_unique(
        values: typing.Sequence[T],
        *,
        atol: builtins.float = ORIENTATION_ATOL,
    ) -> builtins.tuple[T, ...]:
        # TODO: improve computational efficiency
        # TODO: fix bad init -> bad result
        if not values:
            return tuple()
        approx_unique: builtins.dict[T, builtins.int] = dict()
        for val in values:
            min_dist = np.inf
            min_dist_val = None
            for target_val in approx_unique.keys():
                np_val = miou.to_f64(typing.cast(npt.ArrayLike, val))
                np_tgt_val = miou.to_f64(typing.cast(npt.ArrayLike, target_val))
                if np.allclose(np_val, np_tgt_val, atol=atol):
                    dist = np.linalg.norm(np_val - np_tgt_val).item()
                    if dist < min_dist:
                        min_dist = dist
                        min_dist_val = target_val
            if min_dist_val is None:
                approx_unique[val] = 1
            else:
                approx_unique[min_dist_val] += 1
        approx_unq_arrs, _ = miou.unzip(
            sorted(
                ((arr, count) for arr, count in approx_unique.items()),
                key=operator.itemgetter(1),
            )
        )
        return approx_unq_arrs

    def _zip_with(
        self, *args: typing.Iterable[typing.Any]
    ) -> typing.Iterable[builtins.tuple[typing.Any, ...]]:
        return zip(self.slices, self.indices, self.positions, *args)


@dataclasses.dataclass(frozen=True)
class DICOMDir:
    slices: builtins.tuple[pydicom.Dataset, ...]
    positions: builtins.tuple[builtins.float, ...]
    slice_spacing: builtins.float
    affine: npt.NDArray
    paths: builtins.tuple[miot.PathLike, ...] | None = None

    def __len__(self) -> builtins.int:
        _len = len(self.slices)
        if self.paths is not None and _len != (path_len := len(self.paths)):
            raise RuntimeError(f"num slices {_len} != num paths {path_len}")
        return _len

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
        if remove_anomalous_images:
            sorted_slices = sorted_slices.remove_anomalous_slices()
        sorted_slices.check_nonuniformity(
            max_nonuniformity=max_nonuniformity,
            fail_outside_max_nonuniformity=fail_outside_max_nonuniformity,
        )
        positions = tuple(sorted(sorted_slices.positions))
        idxs = sorted_slices.indices
        dicom_dir = cls(
            slices=sorted_slices.slices,
            positions=positions,
            slice_spacing=sorted_slices.slice_spacing,
            affine=sorted_slices.affine,
            paths=None if paths is None else tuple(paths[i] for i in idxs),
        )
        dicom_dir.writable(False)
        dicom_dir.validate()
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
        extension: builtins.str = ".dcm",
    ) -> DICOMDir:
        gathered = gather_dicom(
            dicom_path, defer_size=defer_size, extension=extension, return_paths=True
        )
        assert isinstance(gathered, tuple)
        images, paths = gathered  # unpack after type check for mypy
        return cls.from_datasets(
            tuple(images),
            paths=paths,
            max_nonuniformity=max_nonuniformity,
            fail_outside_max_nonuniformity=fail_outside_max_nonuniformity,
            remove_anomalous_images=remove_anomalous_images,
        )

    @classmethod
    def from_zipped_stream(
        cls: typing.Type[DICOMDir],
        data_stream: typing.IO,
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
    ) -> builtins.list[pydicom.Dataset]:
        datasets: builtins.list[pydicom.Dataset] = []
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

    def validate(self) -> None:
        invariant_properties = frozenset(
            (
                "BitsAllocated",
                "Columns",
                "Modality",
                "PixelRepresentation",
                "PixelSpacing",
                "Rows",
                "SamplesPerPixel",
                "SeriesInstanceUID",
                "SOPClassUID",
            )
        )
        for property_name in invariant_properties:
            self._slice_attribute_equal(property_name)
        self._slice_attribute_almost_equal(
            "ImageOrientationPatient", atol=ORIENTATION_ATOL
        )

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

    def _slice_attribute_almost_equal(
        self,
        property_name: builtins.str,
        *,
        atol: builtins.float,
    ) -> None:
        initial_value: miot.SupportsArray | None
        initial_value = getattr(self.slices[0], property_name, None)
        for dataset in self.slices[1:]:
            value: miot.SupportsArray | None
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


class DICOMImage(miob.BasicImage[typing.Any, miot.DType]):  # type: ignore[type-arg]
    @classmethod
    def from_dicomdir(
        cls: typing.Type[DICOMImage],
        dicom_dir: DICOMDir,
        *,
        rescale: builtins.bool | None = None,
        rescale_dtype: typing.Type[miot.DType] | None = None,
        order: typing.Literal["F", "C"] | None = None,
    ) -> DICOMImage[miot.DType]:
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
        rescale_dtype: typing.Type[miot.DType] | None = None,
        order: typing.Literal["F", "C"] | None = None,
        extension: builtins.str = ".dcm",
    ) -> DICOMImage[miot.DType]:
        dicomdir = DICOMDir.from_path(
            dicom_path,
            max_nonuniformity=max_nonuniformity,
            fail_outside_max_nonuniformity=fail_outside_max_nonuniformity,
            remove_anomalous_images=remove_anomalous_images,
            extension=extension,
        )
        return cls.from_dicomdir(
            dicomdir, rescale=rescale, rescale_dtype=rescale_dtype, order=order
        )

    @classmethod
    def from_zipped_stream(
        cls: typing.Type[DICOMImage],
        data_stream: typing.IO,
        *,
        max_nonuniformity: builtins.float = 5e-4,
        fail_outside_max_nonuniformity: builtins.bool = True,
        remove_anomalous_images: builtins.bool = True,
        encryption_key: builtins.bytes | builtins.str | None = None,
        rescale: builtins.bool | None = None,
        rescale_dtype: typing.Type[miot.DType] | None = None,
        order: typing.Literal["F", "C"] | None = None,
        **zip_kwargs: typing.Any,
    ) -> DICOMImage[miot.DType]:
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
        rescale_dtype: typing.Type[miot.DType] | None = None,
        order: typing.Literal["F", "C"] | None = None,
    ) -> npt.NDArray:
        if rescale is None:
            rescale = any(cls._requires_rescaling(d) for d in slices)
        if rescale and rescale_dtype is None:
            rescale_dtype = np.float32  # type: ignore[assignment]

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
            pixel_array = dataset.pixel_array.T.astype(voxels_dtype, copy=False)
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

        if voxels.dtype != voxels_dtype:
            exn_msg = f"voxels.dtype {voxels.dtype} != requested dtype {voxels_dtype}"
            raise RuntimeError(exn_msg)
        return voxels

    @staticmethod
    def _requires_rescaling(dataset: pydicom.Dataset) -> builtins.bool:
        return hasattr(dataset, "RescaleSlope") or hasattr(dataset, "RescaleIntercept")
