"""Functions for reading medical images
Taken from torchio and modified to output np.ndarray
https://github.com/fepegar/torchio
Author: Jacob Reinhold <jcreinhold@gmail.com>
"""

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
except (ModuleNotFoundError, ImportError) as e:
    msg = f"nibabel and SimpleITK must be installed to use {__name__}."
    raise RuntimeError(msg) from e

import medio.typing as miot

# Matrices used to switch between LPS and RAS
FLIPXY_33 = np.diag([-1, -1, 1])
FLIPXY_44 = np.diag([-1, -1, 1, 1])

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
    path: miot.PathLike, *, dtype: npt.DTypeLike = np.float32
) -> miot.DataAffine:
    try:
        result = _read_sitk(path, dtype=dtype)
    except RuntimeError as e:  # try with NiBabel
        message = f"Error loading image with SimpleITK:\n{e}\n\nTrying NiBabel..."
        warnings.warn(message)
        try:
            result = _read_nibabel(path, dtype=dtype)
        except nib.loadsave.ImageFileError as e:
            message = (
                f"File '{path}' not understood."
                " Check supported formats by at"
                " https://simpleitk.readthedocs.io/en/master/IO.html#images"
                " and https://nipy.org/nibabel/api.html#file-formats"
            )
            raise RuntimeError(message) from e
    return result


def read_image_from_stream(
    stream: typing.BinaryIO,
    *,
    dtype: npt.DTypeLike = np.float32,
    gzipped: builtins.bool = False,
    image_class: typing.Optional[NibabelImageClass] = None,
) -> miot.DataAffine:
    """https://mail.python.org/pipermail/neuroimaging/2017-February/001345.html"""
    _stream = gzip.GzipFile(fileobj=stream) if gzipped else stream
    fh = nib.FileHolder(fileobj=_stream)
    if image_class is None:
        for cls in nib.imageclasses.all_image_classes:
            if hasattr(cls, "from_file_map"):
                try:
                    img = cls.from_file_map({"header": fh, "image": fh})
                    break
                except Exception:
                    logger.debug(traceback.format_exc())
        else:
            raise RuntimeError("Couldn't open data stream.")
    else:
        img = image_class.from_file_map({"header": fh, "image": fh})
    data = img.get_fdata(dtype=dtype)
    if data.ndim == 5:
        data = data[..., 0, :]
        data = data.transpose(3, 0, 1, 2)
    affine = img.affine
    return data, affine


def _read_nibabel(
    path: miot.PathLike, *, dtype: npt.DTypeLike = np.float32
) -> miot.DataAffine:
    img = nib.load(str(path))
    data = img.get_fdata(dtype=dtype)
    if data.ndim == 5:
        data = data[..., 0, :]
        data = data.transpose(3, 0, 1, 2)
    affine = img.affine
    return data, affine


def _read_sitk(
    path: miot.PathLike, *, dtype: npt.DTypeLike = np.float32
) -> miot.DataAffine:
    if pathlib.Path(path).is_dir():  # assume DICOM
        image = _read_dicom_sitk(path)
    else:
        image = sitk.ReadImage(str(path))
    data, affine = sitk_to_array(image, dtype=dtype)
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


def read_affine(path: miot.PathLike) -> npt.NDArray:
    reader = get_reader(path)
    affine = get_ras_affine_from_sitk(reader)
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
    squeeze: typing.Optional[builtins.bool] = None,
) -> None:
    try:
        _write_sitk(array, affine, path, squeeze=squeeze)
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
    assert array.ndim == 4
    num_components = array.shape[0]

    # NIfTI components must be at the end, in a 5D array
    if num_components == 1:
        array = array[0]
    else:
        array = array[np.newaxis].permute(2, 3, 4, 0, 1)
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
    squeeze: typing.Optional[builtins.bool] = None,
) -> None:
    path = pathlib.Path(path)
    if path.suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        warnings.warn(
            f"Casting to uint8 before saving to {path}",
            RuntimeWarning,
        )
        array = array.astype(np.uint8)
    if squeeze is None:
        force_3d = path.suffix not in IMAGE_2D_FORMATS
    else:
        force_3d = not squeeze
    image = array_to_sitk(array, affine, force_3d=force_3d)
    sitk.WriteImage(image, str(path), use_compression)


def read_matrix(path: miot.PathLike) -> npt.NDArray:
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


def _to_itk_convention(matrix: npt.NDArray) -> npt.NDArray:
    """RAS to LPS"""
    matrix = np.dot(FLIPXY_44, matrix)
    matrix = np.dot(matrix, FLIPXY_44)
    matrix = np.linalg.inv(matrix)
    return matrix


def _from_itk_convention(matrix: npt.NDArray) -> npt.NDArray:
    """LPS to RAS"""
    matrix = np.dot(matrix, FLIPXY_44)
    matrix = np.dot(FLIPXY_44, matrix)
    matrix = np.linalg.inv(matrix)
    return matrix


def _read_itk_matrix(path: miot.PathLike) -> npt.NDArray:
    """Read an affine transform in ITK's .tfm format"""
    transform = sitk.ReadTransform(str(path))
    parameters = transform.GetParameters()
    rotation_parameters = parameters[:9]
    rotation_matrix = np.array(rotation_parameters).reshape(3, 3)
    translation_parameters = parameters[9:]
    translation_vector = np.array(translation_parameters).reshape(3, 1)
    matrix = np.hstack([rotation_matrix, translation_vector])
    homogeneous_matrix_lps = np.vstack([matrix, [0, 0, 0, 1]])
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


def _read_niftyreg_matrix(trsf_path: miot.PathLike) -> npt.NDArray:
    """Read a NiftyReg matrix and return it as a torch.Tensor"""
    matrix: np.ndarray = np.loadtxt(trsf_path)
    matrix = np.linalg.inv(matrix)
    return matrix


def _write_niftyreg_matrix(matrix: npt.NDArray, txt_path: miot.PathLike) -> None:
    """Write an affine transform in NiftyReg's .txt format (ref -> flo)"""
    matrix = np.linalg.inv(matrix)
    np.savetxt(txt_path, matrix, fmt="%.8f")


def get_rotation_and_spacing_from_affine(
    affine: npt.NDArray,
) -> typing.Tuple[npt.NDArray, npt.NDArray]:
    # From https://github.com/nipy/nibabel/blob/master/nibabel/orientations.py
    rotation_zoom = affine[:3, :3]
    spacing = np.sqrt(np.sum(rotation_zoom * rotation_zoom, axis=0))
    rotation = rotation_zoom / spacing
    return rotation, spacing


def array_to_sitk(
    array: npt.NDArray,
    affine: npt.NDArray,
    *,
    force_3d: builtins.bool = False,
    force_4d: builtins.bool = False,
) -> sitk.Image:
    """Create a SimpleITK image from a tensor and a 4x4 affine matrix."""
    if array.ndim != 4:
        shape = tuple(array.shape)
        raise ValueError(f"Input must be 4D, but has shape {shape}")
    # Possibilities
    # (1, w, h, 1)
    # (c, w, h, 1)
    # (1, w, h, 1)
    # (c, w, h, d)
    array = np.asanyarray(array)
    affine = np.asanyarray(affine, dtype=np.float64)

    is_multichannel = array.shape[0] > 1 and not force_4d
    is_2d = array.shape[3] == 1 and not force_3d
    if is_2d:
        array = array[..., 0]
    if not is_multichannel and not force_4d:
        array = array[0]
    array = array.transpose()  # (W, H, D, C) or (W, H, D)
    image = sitk.GetImageFromArray(array, isVector=is_multichannel)

    origin, spacing, direction = get_sitk_metadata_from_ras_affine(
        affine,
        is_2d=is_2d,
    )
    image.SetOrigin(origin)  # should I add a 4th value if force_4d?
    image.SetSpacing(spacing)
    image.SetDirection(direction)

    if array.ndim == 4:
        assert image.GetNumberOfComponentsPerPixel() == array.shape[0]
    num_spatial_dims = 2 if is_2d else 3
    assert image.GetSize() == array.shape[1 : 1 + num_spatial_dims]

    return image


def sitk_to_array(
    image: sitk.Image, *, dtype: npt.DTypeLike = np.float32
) -> miot.DataAffine:
    array_view = sitk.GetArrayViewFromImage(image)
    data = np.asanyarray(array_view, dtype=dtype).transpose()
    num_components = image.GetNumberOfComponentsPerPixel()
    input_spatial_dims = image.GetDimension()
    if input_spatial_dims == 2:
        data = data[..., np.newaxis]
    elif input_spatial_dims == 4:  # probably a bad NIfTI (1, sx, sy, sz, c)
        # Try to fix it
        num_components = data.shape[-1]
        data = data[0]
        data = data.transpose(3, 0, 1, 2)
    assert data.shape[0] == num_components
    affine = get_ras_affine_from_sitk(image)
    return data, affine


def get_ras_affine_from_sitk(
    sitk_object: typing.Union[sitk.Image, sitk.ImageFileReader],
) -> npt.NDArray:
    spacing = np.asanyarray(sitk_object.GetSpacing())
    direction_lps = np.asanyarray(sitk_object.GetDirection())
    origin_lps = np.asanyarray(sitk_object.GetOrigin())
    direction_length = len(direction_lps)
    if direction_length == 9:
        rotation_lps = direction_lps.reshape(3, 3)
    elif direction_length == 4:  # ignore last dimension if 2D (1, W, H, 1)
        rotation_lps_2d = direction_lps.reshape(2, 2)
        rotation_lps = np.eye(3)
        rotation_lps[:2, :2] = rotation_lps_2d
        spacing = np.append(spacing, 1)
        origin_lps = np.append(origin_lps, 0)
    elif direction_length == 16:  # probably a bad NIfTI. Let's try to fix it
        rotation_lps = direction_lps.reshape(4, 4)[:3, :3]
        spacing = spacing[:-1]
        origin_lps = origin_lps[:-1]
    else:
        raise RuntimeError(f"Invalid direction length: {direction_length}")
    rotation_ras = np.dot(FLIPXY_33, rotation_lps)
    rotation_ras_zoom = rotation_ras * spacing
    translation_ras = np.dot(FLIPXY_33, origin_lps)
    affine = np.eye(4)
    affine[:3, :3] = rotation_ras_zoom
    affine[:3, 3] = translation_ras
    return affine


def get_sitk_metadata_from_ras_affine(
    affine: npt.NDArray,
    *,
    is_2d: builtins.bool = False,
    lps: builtins.bool = True,
) -> typing.Tuple[miot.TripletFloat, miot.TripletFloat, miot.Direction]:
    direction_ras, spacing_array = get_rotation_and_spacing_from_affine(affine)
    origin_ras = affine[:3, 3]
    origin_lps = np.dot(FLIPXY_33, origin_ras)
    direction_lps = np.dot(FLIPXY_33, direction_ras)
    if is_2d:  # ignore orientation if 2D (1, W, H, 1)
        direction_lps = np.diag((-1, -1)).astype(np.float64)
        direction_ras = np.diag((1, 1)).astype(np.float64)
    origin_array = origin_lps if lps else origin_ras
    direction_array = direction_lps if lps else direction_ras
    direction_array = direction_array.flatten()
    # The following are to comply with typing hints
    # (there must be prettier ways to do this)
    ox, oy, oz = origin_array
    sx, sy, sz = spacing_array
    direction: miot.Direction
    if is_2d:
        d1, d2, d3, d4 = direction_array
        direction = d1, d2, d3, d4
    else:
        d1, d2, d3, d4, d5, d6, d7, d8, d9 = direction_array
        direction = d1, d2, d3, d4, d5, d6, d7, d8, d9
    origin = ox, oy, oz
    spacing = sx, sy, sz
    return origin, spacing, direction


def ensure_4d(
    array: npt.NDArray, *, num_spatial_dims: typing.Optional[builtins.int] = None
) -> npt.NDArray:
    """for PyTorch"""
    num_dimensions = array.ndim
    if num_dimensions == 4:
        pass
    elif num_dimensions == 5:  # hope (W, H, D, 1, C)
        if array.shape[-2] == 1:
            array = array[..., 0, :]
            array = array.transpose((3, 0, 1, 2))
        else:
            raise ValueError("5D is not supported for shape[-2] > 1")
    elif num_dimensions == 2:  # assume 2D monochannel (W, H)
        array = array[np.newaxis, ..., np.newaxis]  # (1, W, H, 1)
    elif num_dimensions == 3:  # 2D multichannel or 3D monochannel?
        if num_spatial_dims == 2:
            array = array[..., np.newaxis]  # (C, W, H, 1)
        elif num_spatial_dims == 3:  # (W, H, D)
            array = array[np.newaxis]  # (1, W, H, D)
        else:  # try to guess
            shape = array.shape
            maybe_rgb = 3 in (shape[0], shape[-1])
            if maybe_rgb:
                if shape[-1] == 3:  # (W, H, 3)
                    array = array.transpose((2, 0, 1))  # (3, W, H)
                array = array[..., np.newaxis]  # (3, W, H, 1)
            else:  # (W, H, D)
                array = array[np.newaxis]  # (1, W, H, D)
    else:
        message = f"{num_dimensions}D images not supported yet."
        raise NotImplementedError(message)
    assert array.ndim == 4
    return array


def check_uint_to_int(array: npt.NDArray) -> npt.NDArray:
    """convert to int b/c PyTorch won't take uint16 nor uint32"""
    if array.dtype == np.uint16:
        return array.astype(np.int32)
    elif array.dtype == np.uint32:
        return array.astype(np.int64)
    else:
        return array
