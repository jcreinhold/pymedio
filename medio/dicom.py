"""DICOM-specific functions
Author: Jacob Reinhold <jcreinhold@gmail.com>
"""

import builtins
import io
import logging
import math
import typing
import warnings
import zipfile

import numpy as np
import numpy.typing as npt
import pydicom

from medio.exceptions import DicomImportException

logger = logging.getLogger(__name__)


def combined_series_from_zipped_stream(
    data_stream: io.BytesIO,
    *,
    rescale: typing.Optional[builtins.bool] = None,
    rescale_dtype: npt.DTypeLike = np.float32,
    enforce_slice_spacing: builtins.bool = True,
) -> typing.Tuple[npt.NDArray, npt.NDArray]:
    with zipfile.ZipFile(data_stream, mode="r") as zf:
        datasets = dicom_datasets_from_zip(zf)
    voxels, ijk_to_xyz = combine_slices(
        datasets,
        rescale=rescale,
        rescale_dtype=rescale_dtype,
        enforce_slice_spacing=enforce_slice_spacing,
    )
    return voxels, ijk_to_xyz


def dicom_datasets_from_zip(zip_file: zipfile.ZipFile) -> typing.List[pydicom.Dataset]:
    datasets: typing.List[pydicom.Dataset] = []
    for name in zip_file.namelist():
        if name.endswith("/"):
            continue  # skip directories
        with zip_file.open(name, mode="r") as f:
            try:
                dataset = pydicom.dcmread(f)  # type: ignore[arg-type]
                datasets.append(dataset)
            except pydicom.errors.InvalidDicomError as e:
                msg = f"Skipping invalid DICOM file '{name}': {e}"
                logger.info(msg)

    if not datasets:
        msg = "Zipfile does not contain any valid DICOM files"
        raise DicomImportException(msg)

    return datasets


def combine_slices(
    datasets: typing.List[pydicom.Dataset],
    *,
    rescale: typing.Optional[builtins.bool] = None,
    rescale_dtype: npt.DTypeLike = np.float32,
    enforce_slice_spacing: builtins.bool = True,
) -> typing.Tuple[npt.NDArray, npt.NDArray]:
    """
    Given a list of pydicom datasets for an image series, stitch them together into a
    three-dimensional numpy array.  Also calculate a 4x4 affine transformation
    matrix that converts the ijk-pixel-indices into the xyz-coordinates in the
    DICOM patient's coordinate system.

    Returns a two-tuple containing the 3D-ndarray and the affine matrix.

    If `rescale` is set to `None` (the default), then the image array dtype
    will be preserved, unless any of the DICOM images contain either the
    `Rescale Slope
    <https://dicom.innolitics.com/ciods/ct-image/ct-image/00281053>`_ or the
    `Rescale Intercept <https://dicom.innolitics.com/ciods/ct-image/ct-image/00281052>`_
    attributes.  If either of these attributes are present, they will be
    applied to each slice individually.

    If `rescale` is `True` the voxels will be cast to `float32`, if set to
    `False`, the original dtype will be preserved even if DICOM rescaling information is present.

    If `enforce_slice_spacing` is set to `True`, `combine_slices` will raise a
    `DicomImportException` if there are missing slices detected in the
    datasets. If `enforce_slice_spacing` is set to `False`, missing slices will
    be ignored.

    The returned array has the column-major byte-order.

    Datasets produced by reading DICOMDIR files are ignored.

    This function requires that the datasets:

    - Be in same series (have the same
      `Series Instance UID <https://dicom.innolitics.com/ciods/ct-image/general-series/0020000e>`_,
      `Modality <https://dicom.innolitics.com/ciods/ct-image/general-series/00080060>`_,
      and `SOP Class UID <https://dicom.innolitics.com/ciods/ct-image/sop-common/00080016>`_).
    - The binary storage of each slice must be the same (have the same
      `Bits Allocated <https://dicom.innolitics.com/ciods/ct-image/image-pixel/00280100>`_ and
      `Pixel Representation <https://dicom.innolitics.com/ciods/ct-image/image-pixel/00280103>`_).
    - The image slice must approximately form a grid. This means there can not
      be any missing internal slices (missing slices on the ends of the dataset
      are not detected). This requirement is relaxed if `enforce_slice_spacing` is set to `False`.
    - Each slice must have the same
      `Rows <https://dicom.innolitics.com/ciods/ct-image/image-pixel/00280010>`_,
      `Columns <https://dicom.innolitics.com/ciods/ct-image/image-pixel/00280011>`_,
      `Samples Per Pixel <https://dicom.innolitics.com/ciods/ct-image/image-pixel/00280002>`_,
      `Pixel Spacing <https://dicom.innolitics.com/ciods/ct-image/image-plane/00280030>`_, and
      `Image Orientation (Patient) <https://dicom.innolitics.com/ciods/ct-image/image-plane/00200037>`_
      attribute values.
    - The direction cosines derived from the
      `Image Orientation (Patient) <https://dicom.innolitics.com/ciods/ct-image/image-plane/00200037>`_
      attribute must, within 1e-4, have a magnitude of 1.  The cosines must
      also be approximately perpendicular (their dot-product must be within
      1e-4 of 0).  Warnings are displayed if any of these approximations are
      below 1e-8, however, since we have seen real datasets with values up to
      1e-4, we let them pass.
    - The `Image Position (Patient) <https://dicom.innolitics.com/ciods/ct-image/image-plane/00200032>`_
      values must approximately form a line.

    If any of these conditions are not met, a `dicom_numpy.DicomImportException` is raised.
    """
    slice_datasets = [ds for ds in datasets if not _is_dicomdir(ds)]

    if len(slice_datasets) == 0:
        raise DicomImportException("Must provide at least one image DICOM dataset")

    _validate_slices_form_uniform_grid(
        slice_datasets, enforce_slice_spacing=enforce_slice_spacing
    )

    voxels = _merge_slice_pixel_arrays(
        slice_datasets, rescale=rescale, rescale_dtype=rescale_dtype
    )
    transform = _ijk_to_patient_xyz_transform_matrix(slice_datasets)

    return voxels, transform


def sort_by_slice_position(
    slice_datasets: typing.List[pydicom.Dataset],
) -> typing.List[pydicom.Dataset]:
    """
    Given a list of pydicom Datasets, return the datasets sorted in the image orientation direction.

    This does not require `pixel_array` to be present, and so may be used to associate instance Datasets
    with the voxels returned from `combine_slices.
    """
    slice_positions = _slice_positions(slice_datasets)
    return [
        d
        for (s, d) in sorted(
            zip(slice_positions, slice_datasets),
            key=lambda v: v[0],
        )
    ]


def _is_dicomdir(dataset: pydicom.Dataset) -> builtins.bool:
    media_sop_class: str = getattr(dataset, "MediaStorageSOPClassUID", None)
    result: builtins.bool = media_sop_class == "1.2.840.10008.1.3.10"
    return result


def _merge_slice_pixel_arrays(
    slice_datasets: typing.List[pydicom.Dataset],
    *,
    rescale: typing.Optional[builtins.bool] = None,
    rescale_dtype: npt.DTypeLike = np.float32,
) -> npt.NDArray:
    sorted_slice_datasets = sort_by_slice_position(slice_datasets)

    if rescale is None:
        rescale = any(_requires_rescaling(d) for d in sorted_slice_datasets)

    first_dataset = sorted_slice_datasets[0]
    slice_dtype = first_dataset.pixel_array.dtype
    slice_shape = first_dataset.pixel_array.T.shape
    num_slices = len(sorted_slice_datasets)

    voxels_shape = slice_shape + (num_slices,)
    voxels_dtype = rescale_dtype if rescale else slice_dtype
    voxels = np.empty(voxels_shape, dtype=voxels_dtype, order="F")

    for k, dataset in enumerate(sorted_slice_datasets):
        pixel_array = dataset.pixel_array.T
        if rescale:
            slope = float(getattr(dataset, "RescaleSlope", 1))
            intercept = float(getattr(dataset, "RescaleIntercept", 0))
            pixel_array = pixel_array.astype(np.float32) * slope + intercept
        voxels[..., k] = pixel_array

    return voxels


def _requires_rescaling(dataset: pydicom.Dataset) -> builtins.bool:
    return hasattr(dataset, "RescaleSlope") or hasattr(dataset, "RescaleIntercept")


def _ijk_to_patient_xyz_transform_matrix(
    slice_datasets: typing.List[pydicom.Dataset],
) -> npt.NDArray:
    first_dataset = sort_by_slice_position(slice_datasets)[0]
    image_orientation = first_dataset.ImageOrientationPatient
    row_cosine, column_cosine, slice_cosine = _extract_cosines(image_orientation)

    row_spacing, column_spacing = first_dataset.PixelSpacing
    slice_spacing = _slice_spacing(slice_datasets)

    transform = np.identity(4, dtype=np.float32)

    transform[:3, 0] = row_cosine * column_spacing
    transform[:3, 1] = column_cosine * row_spacing
    transform[:3, 2] = slice_cosine * slice_spacing

    transform[:3, 3] = first_dataset.ImagePositionPatient

    return transform


def _validate_slices_form_uniform_grid(
    slice_datasets: typing.List[pydicom.Dataset],
    enforce_slice_spacing: builtins.bool = True,
) -> None:
    """
    Perform various data checks to ensure that the list of slices form a
    evenly-spaced grid of data. Optionally, this can be slightly relaxed to
    allow for missing slices in the volume.

    Some of these checks are probably not required if the data follows the
    DICOM specification, however it seems pertinent to check anyway.
    """
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
        _slice_attribute_equal(slice_datasets, property_name)

    _validate_image_orientation(slice_datasets[0].ImageOrientationPatient)
    _slice_ndarray_attribute_almost_equal(
        slice_datasets, "ImageOrientationPatient", abs_tol=1e-5
    )

    if enforce_slice_spacing:
        slice_positions = _slice_positions(slice_datasets)
        _check_for_missing_slices(slice_positions)


def _validate_image_orientation(image_orientation: npt.NDArray) -> None:
    """
    Ensure that the image orientation is supported
    - The direction cosines have magnitudes of 1 (just in case)
    - The direction cosines are perpendicular
    """
    row_cosine, column_cosine, slice_cosine = _extract_cosines(image_orientation)

    dot_prod = np.dot(row_cosine, column_cosine)
    if not _almost_zero(dot_prod, abs_tol=1e-4):
        msg = f"Non-orthogonal direction cosines: {row_cosine}, {column_cosine}"
        raise DicomImportException(msg)
    elif not _almost_zero(dot_prod, abs_tol=1e-8):
        msg = f"Direction cosines aren't quite ortho.: {row_cosine}, {column_cosine}"
        warnings.warn(msg)

    row_cosine_norm = np.linalg.norm(row_cosine)
    if not _almost_one(row_cosine_norm, abs_tol=1e-4):
        msg = f"The row direction cosine's magnitude is not 1: {row_cosine}"
        raise DicomImportException(msg)
    elif not _almost_one(row_cosine_norm, abs_tol=1e-8):
        msg = f"The row direction cosine's magnitude is not quite 1: {row_cosine}"
        warnings.warn(msg)

    column_cosine_norm = np.linalg.norm(column_cosine)
    if not _almost_one(column_cosine_norm, abs_tol=1e-4):
        msg = f"The column direction cosine's magnitude is not 1: {column_cosine}"
        raise DicomImportException(msg)
    elif not _almost_one(column_cosine_norm, abs_tol=1e-8):
        msg = f"The column direction cosine's magnitude is not quite 1: {column_cosine}"
        warnings.warn(msg)


def _almost_zero(value: builtins.float, *, abs_tol: builtins.float) -> builtins.bool:
    return math.isclose(value, 0.0, abs_tol=abs_tol)


def _almost_one(value: builtins.float, *, abs_tol: builtins.float) -> builtins.bool:
    return math.isclose(value, 1.0, abs_tol=abs_tol)


def _extract_cosines(
    image_orientation: npt.NDArray,
) -> typing.Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    row_cosine = np.array(image_orientation[:3])
    column_cosine = np.array(image_orientation[3:])
    slice_cosine = np.cross(row_cosine, column_cosine)
    return row_cosine, column_cosine, slice_cosine


def _slice_attribute_equal(
    slice_datasets: typing.List[pydicom.Dataset], property_name: builtins.str
) -> None:
    initial_value = getattr(slice_datasets[0], property_name, None)
    for dataset in slice_datasets[1:]:
        value = getattr(dataset, property_name, None)
        if value != initial_value:
            msg = f'All slices must have the same value for "{property_name}": {value} != {initial_value}'
            raise DicomImportException(msg)


def _slice_ndarray_attribute_almost_equal(
    slice_datasets: typing.List[pydicom.Dataset],
    property_name: builtins.str,
    *,
    abs_tol: builtins.float,
) -> None:
    initial_value = getattr(slice_datasets[0], property_name, None)
    for dataset in slice_datasets[1:]:
        value = getattr(dataset, property_name, None)
        if not np.allclose(value, initial_value, atol=abs_tol):
            msg = (
                f'All slices must have the same value for "{property_name}" within "{abs_tol}": {value} != '
                f"{initial_value}"
            )
            raise DicomImportException(msg)


def _slice_positions(
    slice_datasets: typing.List[pydicom.Dataset],
) -> typing.List[builtins.float]:
    image_orientation = slice_datasets[0].ImageOrientationPatient
    row_cosine, column_cosine, slice_cosine = _extract_cosines(image_orientation)
    return [
        float(np.dot(slice_cosine, d.ImagePositionPatient).item())
        for d in slice_datasets
    ]


def _check_for_missing_slices(slice_positions: typing.List[builtins.float]) -> None:
    if len(slice_positions) > 1:
        slice_positions_diffs = np.diff(sorted(slice_positions))
        if not np.allclose(
            slice_positions_diffs, slice_positions_diffs[0], atol=0, rtol=1e-5
        ):
            # TODO: figure out how we should handle non-even slice spacing
            msg = f"The slice spacing is non-uniform. Slice spacings:\n{slice_positions_diffs}"
            logger.warning(msg)

        if not np.allclose(
            slice_positions_diffs, slice_positions_diffs[0], atol=0, rtol=1e-1
        ):
            raise DicomImportException("It appears there are missing slices")


def _slice_spacing(slice_datasets: typing.List[pydicom.Dataset]) -> npt.NDArray:
    spacing: npt.NDArray
    if len(slice_datasets) > 1:
        slice_positions = _slice_positions(slice_datasets)
        slice_positions_diffs = np.diff(sorted(slice_positions))
        spacing = np.median(slice_positions_diffs)
    elif len(slice_datasets) == 1:
        spacing = getattr(slice_datasets[0], "SpacingBetweenSlices", 0)
    else:
        raise RuntimeError("slice_datasets must contain at least one dicom image")
    return spacing
