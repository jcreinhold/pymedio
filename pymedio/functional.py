"""Functions for reading medical images
Taken from torchio and modified to output np.ndarray
https://github.com/fepegar/torchio
Author: Jacob Reinhold <jcreinhold@gmail.com>
"""

from __future__ import annotations

__all__ = [
    "read_affine",
    "read_image",
    "read_matrix",
    "write_image",
    "write_matrix",
]

import builtins
import gzip
import logging
import pathlib
import traceback
import typing
import warnings

import numpy as np
import numpy.typing as npt

try:
    import nibabel as nib
    import SimpleITK as sitk
except ImportError as imp_exn:
    imp_exn_msg = f"NiBabel and SimpleITK must be installed to use {__name__}."
    raise ImportError(imp_exn_msg) from imp_exn

import pymedio.typing as miot
import pymedio.utils as miou

# Image formats that are typically 2D
_2d_formats = [".jpg", ".jpeg", ".bmp", ".png", ".tif", ".tiff"]
IMAGE_2D_FORMATS = _2d_formats + [s.upper() for s in _2d_formats]

NibabelImageClass = typing.Type[
    typing.Union[
        nib.Nifti1Pair,
        nib.Nifti1Image,
        nib.Nifti2Pair,
        nib.Cifti2Image,
        nib.Nifti2Image,
        nib.Spm2AnalyzeImage,
        nib.Spm99AnalyzeImage,
        nib.AnalyzeImage,
        nib.Minc1Image,
        nib.Minc2Image,
        nib.MGHImage,
        nib.GiftiImage,
    ]
]

logger = logging.getLogger(__name__)


def read_image(
    path: miot.PathLike,
    *,
    dtype: typing.Type[miot.DType] | None = None,
    eager: builtins.bool = True,
) -> miot.DataAffine[miot.DType]:
    try:
        result = _read_sitk(path, dtype=dtype, copy=eager)
    except RuntimeError as exn1:  # try with NiBabel
        message = f"Error loading image with SimpleITK:\n{exn1}\n\nTrying NiBabel..."
        warnings.warn(message)
        try:
            result = _read_nibabel(path, dtype=dtype, mmap=not eager)
        except nib.loadsave.ImageFileError as exn2:
            message = (
                f"File '{path}' not understood."
                " Check supported formats by at"
                " https://simpleitk.readthedocs.io/en/master/IO.html#images"
                " and https://nipy.org/nibabel/api.html#file-formats"
            )
            raise RuntimeError(message) from exn2
    return result


def read_image_from_stream(
    stream: typing.IO,
    *,
    dtype: typing.Type[miot.DType] | None = None,
    gzipped: builtins.bool = False,
    image_class: typing.Optional[NibabelImageClass] = None,
) -> miot.DataAffine[miot.DType]:
    """https://mail.python.org/pipermail/neuroimaging/2017-February/001345.html"""
    _stream = gzip.GzipFile(fileobj=stream) if gzipped else stream
    fh = nib.FileHolder(fileobj=_stream)
    if image_class is None:
        for cls in nib.imageclasses.all_image_classes:
            if hasattr(cls, "from_file_map"):
                try:
                    img = cls.from_file_map({"header": fh, "image": fh}, mmap=False)
                    break
                except Exception:
                    logger.debug(traceback.format_exc())
        else:
            raise RuntimeError("Couldn't open data stream.")
    else:
        img = image_class.from_file_map({"header": fh, "image": fh}, mmap=False)
    data = img.get_fdata(dtype=dtype)
    if data.ndim == 5:
        data = data[..., 0, :]
        data = data.transpose(3, 0, 1, 2)
    affine = img.affine
    return data, affine


def _read_nibabel(
    path: miot.PathLike,
    *,
    dtype: typing.Type[miot.DType] | None = None,
    mmap: builtins.bool = False,
) -> miot.DataAffine[miot.DType]:
    img = nib.load(str(path), mmap=mmap)
    data = img.get_fdata(dtype=dtype)
    if data.ndim == 5:
        data = data[..., 0, :]
        data = data.transpose(3, 0, 1, 2)
    affine = miou.to_f64(img.affine)
    return data, affine


def _read_sitk(
    path: miot.PathLike,
    *,
    dtype: typing.Type[miot.DType] | None = None,
    copy: builtins.bool = True,
) -> miot.DataAffine[miot.DType]:
    if pathlib.Path(path).is_dir():  # assume DICOM
        image = _read_dicom_sitk(path)
    else:
        image = sitk.ReadImage(str(path))
    data, affine = sitk_to_array(image, dtype=dtype, copy=copy)
    return data, affine


def _read_dicom_sitk(directory: miot.PathLike) -> sitk.Image:
    directory = pathlib.Path(directory)
    if not directory.is_dir():  # unreachable if called from _read_sitk
        raise FileNotFoundError(f"Directory '{directory}' not found")
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(directory))
    if not dicom_names:
        message = f"The directory '{directory}' does not seem to contain DICOM files"
        raise FileNotFoundError(message)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image


def read_shape(path: miot.PathLike) -> miot.Shape:
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    num_channels = reader.GetNumberOfComponents()
    spatial_shape = reader.GetSize()
    num_dimensions = reader.GetDimension()
    if num_dimensions == 2:
        spatial_shape = *spatial_shape, 1
    elif num_dimensions == 4:  # assume bad NIfTI
        *spatial_shape, num_channels = spatial_shape
    sx, sy, sz = spatial_shape
    shape = (num_channels, sx, sy, sz)
    return shape


def read_affine(path: miot.PathLike) -> npt.NDArray[np.float64]:
    reader = get_reader(path)
    affine: npt.NDArray[np.float64] = get_ras_affine_from_sitk(reader)
    return affine


def get_reader(
    path: miot.PathLike, *, read: builtins.bool = True
) -> sitk.ImageFileReader:
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    if read:
        reader.ReadImageInformation()
    return reader


def write_image(
    array: npt.NDArray,
    affine: npt.NDArray,
    path: miot.PathLike,
    *,
    squeeze: builtins.bool = True,
    **write_sitk_kwargs: builtins.bool,
) -> None:
    if squeeze:
        array = array.squeeze()
    try:
        _write_sitk(array, affine, path, **write_sitk_kwargs)
    except RuntimeError:  # try with NiBabel
        _write_nibabel(array, affine, path)


def _write_nibabel(
    array: npt.NDArray,
    affine: npt.NDArray,
    path: miot.PathLike,
) -> None:
    """
    Expects a path with an extension that can be used by nibabel.save
    to write a NIfTI-1 image, such as '.nii.gz' or '.img'
    """
    num_components = array.shape[0]
    # NIfTI components must be at the end, in a 5D array
    if num_components == 1:
        array = array[0]
    elif array.ndim == 4:
        array = array.transpose((1, 2, 3, 0))
    suffix = pathlib.Path(str(path).replace(".gz", "")).suffix
    if ".nii" in suffix:
        img = nib.Nifti1Image(np.asanyarray(array), affine)
    elif ".hdr" in suffix or ".img" in suffix:
        img = nib.Nifti1Pair(np.asanyarray(array), affine)
    else:
        raise nib.loadsave.ImageFileError
    if num_components > 1:
        img.header.set_intent("vector")
    img.header["qform_code"] = 1
    img.header["sform_code"] = 0
    nib.save(img, str(path))


def _write_sitk(
    array: npt.NDArray,
    affine: npt.NDArray,
    path: miot.PathLike,
    *,
    use_compression: builtins.bool = True,
    is_multichannel: builtins.bool = False,
) -> None:
    path = pathlib.Path(path)
    if path.suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        msg = f"Casting to uint8 before saving to {path}"
        warnings.warn(msg, RuntimeWarning)
        array = array.astype(dtype=np.uint8, copy=False)
    image = array_to_sitk(array, affine, is_multichannel=is_multichannel)
    sitk.WriteImage(image, str(path), use_compression)


def read_matrix(path: miot.PathLike) -> npt.NDArray[np.float64]:
    """Read an affine transform and return array"""
    path = pathlib.Path(path)
    suffix = path.suffix
    if suffix in (".tfm", ".h5"):  # ITK
        array = _read_itk_matrix(path)
    elif suffix in (".txt", ".trsf"):  # NiftyReg, blockmatching
        array = _read_niftyreg_matrix(path)
    else:
        raise ValueError(f"Unknown suffix for transform file: '{suffix}'")
    return array


def write_matrix(matrix: npt.NDArray, path: miot.PathLike) -> None:
    """Write an affine transform."""
    path = pathlib.Path(path)
    suffix = path.suffix
    if suffix in (".tfm", ".h5"):  # ITK
        _write_itk_matrix(matrix, path)
    elif suffix in (".txt", ".trsf"):  # NiftyReg, blockmatching
        _write_niftyreg_matrix(matrix, path)


def _to_itk_convention(matrix: npt.NDArray) -> npt.NDArray[np.float64]:
    """RAS to LPS"""
    _flipxy_44 = miou.flipxy_44()
    matrix = np.dot(_flipxy_44, matrix)
    matrix = np.dot(matrix, _flipxy_44)
    matrix = np.linalg.inv(matrix)
    return matrix


def _from_itk_convention(matrix: npt.NDArray) -> npt.NDArray[np.float64]:
    """LPS to RAS"""
    _flipxy_44 = miou.flipxy_44()
    matrix = np.dot(matrix, _flipxy_44)
    matrix = np.dot(_flipxy_44, matrix)
    matrix = np.linalg.inv(matrix)
    return matrix


def _read_itk_matrix(path: miot.PathLike) -> npt.NDArray[np.float64]:
    """Read an affine transform in ITK's .tfm format"""
    transform = sitk.ReadTransform(str(path))
    parameters = transform.GetParameters()
    rotation_params = parameters[:9]
    rotation_matrix = miou.to_f64(rotation_params).reshape(3, 3)
    translation_params = parameters[9:]
    translation_vector = miou.to_f64(translation_params).reshape(3, 1)
    matrix: np.ndarray = np.hstack([rotation_matrix, translation_vector])
    homogeneous_matrix_lps: np.ndarray = np.vstack([matrix, [0.0, 0.0, 0.0, 1.0]])
    homogeneous_matrix_ras = _from_itk_convention(homogeneous_matrix_lps)
    return homogeneous_matrix_ras


def _write_itk_matrix(matrix: npt.NDArray, tfm_path: miot.PathLike) -> None:
    """The tfm file contains the matrix from floating to reference."""
    transform = _matrix_to_itk_transform(matrix)
    transform.WriteTransform(str(tfm_path))


def _matrix_to_itk_transform(
    matrix: npt.NDArray, *, dims: builtins.int = 3
) -> sitk.AffineTransform:
    matrix = _to_itk_convention(matrix)
    rotation = matrix[:dims, :dims].ravel().tolist()
    translation = matrix[:dims, 3].tolist()
    transform = sitk.AffineTransform(rotation, translation)
    return transform


def _read_niftyreg_matrix(path: miot.PathLike) -> npt.NDArray[np.float64]:
    """Read a NiftyReg matrix and return it as a torch.Tensor"""
    matrix: np.ndarray = np.loadtxt(path, dtype=np.float64)
    matrix = np.linalg.inv(matrix)
    return matrix


def _write_niftyreg_matrix(matrix: npt.NDArray, txt_path: miot.PathLike) -> None:
    """Write an affine transform in NiftyReg's .txt format (ref -> flo)"""
    matrix = np.linalg.inv(matrix)
    np.savetxt(txt_path, matrix, fmt="%.8f")


def array_to_sitk(
    array: npt.NDArray,
    affine: npt.NDArray,
    *,
    is_multichannel: builtins.bool = False,
) -> sitk.Image:
    """Create a SimpleITK image from an array and a 4x4 affine matrix."""
    ndim = array.ndim
    array = np.asanyarray(array)
    affine = miou.to_f64(affine)
    image = sitk.GetImageFromArray(array.transpose(), isVector=is_multichannel)
    is_2d = (ndim == 3 and is_multichannel) or (ndim == 2 and not is_multichannel)
    origin, spacing, direction = miou.get_metadata_from_ras_affine(
        affine,
        is_2d=is_2d,
    )
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    image.SetDirection(direction)
    num_spatial_dims = 2 if is_2d else 3
    offset = 1 if is_multichannel else 0
    if is_multichannel:
        num_components = array.shape[0]
        if (_n_comp := image.GetNumberOfComponentsPerPixel()) != num_components:
            msg = f"sitk components {_n_comp} != array components {num_components}"
            raise RuntimeError(msg)
    spatial_dims = array.shape[offset : offset + num_spatial_dims]
    if image.GetSize() != spatial_dims:
        raise RuntimeError(f"{image.GetSize()} != {spatial_dims}")
    return image


def sitk_to_array(
    image: sitk.Image,
    *,
    dtype: typing.Type[miot.DType] | None = None,
    copy: builtins.bool = True,
) -> miot.DataAffine[miot.DType]:
    arr = sitk.GetArrayFromImage(image) if copy else sitk.GetArrayViewFromImage(image)
    data: np.ndarray = np.asarray(arr, dtype=dtype).transpose()
    num_components = image.GetNumberOfComponentsPerPixel()
    input_spatial_dims = image.GetDimension()
    if input_spatial_dims == 5:  # probably a bad NIfTI (1, sx, sy, sz, c)
        # Try to fix it
        num_components = data.shape[-1]
        data = data[0]
        data = data.transpose(3, 0, 1, 2)
    if num_components > 1 and data.shape[0] != num_components:
        raise RuntimeError(f"{data.shape[0]} != {num_components}")
    affine: npt.NDArray[np.float64] = get_ras_affine_from_sitk(image)
    return data, affine


def get_ras_affine_from_sitk(
    sitk_object: sitk.Image | sitk.ImageFileReader,
    *,
    dtype: typing.Type[miot.DType] | None = None,
) -> npt.NDArray[miot.DType]:
    if dtype is None:
        dtype = np.float64  # type: ignore[assignment]
    spacing: np.ndarray = np.asarray(sitk_object.GetSpacing(), dtype=dtype)
    direction_lps: np.ndarray = np.asarray(sitk_object.GetDirection(), dtype=dtype)
    origin_lps: np.ndarray = np.asarray(sitk_object.GetOrigin(), dtype=dtype)
    direction_length = len(direction_lps)
    if direction_length == 9:
        rotation_lps = direction_lps.reshape(3, 3)
    elif direction_length == 4:  # ignore last dimension if 2D (1, W, H, 1)
        rotation_lps_2d = direction_lps.reshape(2, 2)
        rotation_lps = np.eye(3, dtype=dtype)
        rotation_lps[:2, :2] = rotation_lps_2d
        spacing = np.append(spacing, 1)
        origin_lps = np.append(origin_lps, 0)
    elif direction_length == 16:  # probably a bad NIfTI. Let's try to fix it
        rotation_lps = direction_lps.reshape(4, 4)[:3, :3]
        spacing = spacing[:-1]
        origin_lps = origin_lps[:-1]
    else:
        raise RuntimeError(f"Invalid direction length: {direction_length}")
    _flipxy_33 = miou.flipxy_33()
    rotation_ras = np.dot(_flipxy_33, rotation_lps)
    rotation_ras_zoom = rotation_ras * spacing
    translation_ras = np.dot(_flipxy_33, origin_lps)
    affine: np.ndarray = np.eye(4, dtype=dtype)
    affine[:3, :3] = rotation_ras_zoom
    affine[:3, 3] = translation_ras
    return affine
