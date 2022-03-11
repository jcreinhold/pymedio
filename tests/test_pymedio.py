"""Tests for `medio` package."""

from __future__ import annotations

import builtins
import gzip
import io
import pathlib
import zipfile

import cryptography.fernet as crypto
import nibabel as nib
import numpy as np
import pydicom
import pytest
import SimpleITK as sitk
from pydicom.data import get_testdata_file

import pymedio.dicom as miod
import pymedio.image as mioi

TEST_IMAGE_NAME = "CT_small.dcm"
TEST_IMAGE_SHAPE = (128, 128)
NUM_DUPLICATES = 5
NIFTI_IMAGE_SHAPE = (2, 2, 2)
DIST = 1e-05


@pytest.fixture(scope="session")
def dicom_image_path() -> pathlib.Path:
    path = get_testdata_file(TEST_IMAGE_NAME, read=False)
    assert isinstance(path, builtins.str)
    return pathlib.Path(path)


@pytest.fixture(scope="session")
def dicom_image_dir(
    tmp_path_factory: pytest.TempPathFactory, dicom_image_path: pathlib.Path
) -> pathlib.Path:
    dcm_dir = tmp_path_factory.mktemp("dicom").resolve(strict=True)
    path = dicom_image_path.resolve(strict=True)
    dcm = pydicom.dcmread(str(path))
    position = dcm.ImagePositionPatient
    for i in range(NUM_DUPLICATES):
        position[2] += DIST
        dcm.ImagePositionPatient = position
        pydicom.dcmwrite(dcm_dir / f"{i}.dcm", dcm)
    return dcm_dir


@pytest.fixture(scope="session")
def dicom_image(dicom_image_path: pathlib.Path) -> pydicom.Dataset:
    return pydicom.dcmread(dicom_image_path)


@pytest.fixture(scope="session")
def nifti() -> nib.Nifti1Image:
    data: np.ndarray = np.random.randn(*NIFTI_IMAGE_SHAPE).astype(np.float32)
    return nib.Nifti1Image(data, np.eye(4))


def _make_nifti_path(
    tmp_path_factory: pytest.TempPathFactory,
    nifti: nib.Nifti1Image,
    extension: builtins.str,
) -> pathlib.Path:
    dcm_dir = tmp_path_factory.mktemp(extension.replace(".", "")).resolve(strict=True)
    path = dcm_dir / ("test" + extension)
    nifti.to_filename(path)
    return path


@pytest.fixture(scope="session")
def nifti_gzipped_image_path(
    tmp_path_factory: pytest.TempPathFactory, nifti: nib.Nifti1Image
) -> pathlib.Path:
    return _make_nifti_path(tmp_path_factory, nifti, ".nii.gz")


@pytest.fixture(scope="session")
def nifti_image_path(
    tmp_path_factory: pytest.TempPathFactory, nifti: nib.Nifti1Image
) -> pathlib.Path:
    return _make_nifti_path(tmp_path_factory, nifti, ".nii")


@pytest.fixture(scope="session")
def zipped_dicom_path(
    tmp_path_factory: pytest.TempPathFactory, dicom_image: pydicom.Dataset
) -> pathlib.Path:
    zipped_path = tmp_path_factory.getbasetemp().resolve(strict=True) / "dcm.zip"
    with io.BytesIO() as buffer:
        pydicom.dcmwrite(buffer, dicom_image)
        buffer.seek(0)
        with zipfile.ZipFile(zipped_path, mode="w") as zf:
            zf.writestr("test", buffer.getvalue())
    return zipped_path


@pytest.fixture(scope="session")
def encryption_key() -> builtins.bytes:
    return crypto.Fernet.generate_key()


@pytest.fixture(scope="session")
def zipped_encrypted_dicom_path(
    tmp_path_factory: pytest.TempPathFactory,
    dicom_image: pydicom.Dataset,
    encryption_key: builtins.bytes,
) -> pathlib.Path:
    zipped_path = tmp_path_factory.getbasetemp().resolve(strict=True) / "dcm.zip"
    fernet = crypto.Fernet(encryption_key)
    with io.BytesIO() as zip_buffer:
        with zipfile.ZipFile(zip_buffer, mode="w") as zf:
            with io.BytesIO() as image_buffer:
                pydicom.dcmwrite(image_buffer, dicom_image)
                zf.writestr("test", image_buffer.getvalue())
        encrypted = fernet.encrypt(zip_buffer.getvalue())
    with open(zipped_path, "wb") as f:
        f.write(encrypted)
    return zipped_path


@pytest.fixture(scope="session")
def zipped_nifti_path(
    tmp_path_factory: pytest.TempPathFactory, nifti: nib.Nifti1Image
) -> pathlib.Path:
    zipped_path = tmp_path_factory.getbasetemp().resolve(strict=True) / "nii.zip"
    with io.BytesIO() as buffer:
        file_map = nib.Nifti1Image.make_file_map({"image": buffer, "header": buffer})
        nifti.to_file_map(file_map)
        with zipfile.ZipFile(zipped_path, mode="w") as zf:
            zf.writestr("test", buffer.getvalue())
    return zipped_path


@pytest.fixture(scope="session")
def zipped_nifti_gzipped_path(
    tmp_path_factory: pytest.TempPathFactory, nifti: nib.Nifti1Image
) -> pathlib.Path:
    zipped_path = tmp_path_factory.getbasetemp().resolve(strict=True) / "niigz.zip"
    with io.BytesIO() as buffer:
        file_map = nib.Nifti1Image.make_file_map({"image": buffer, "header": buffer})
        nifti.to_file_map(file_map)
        gz = gzip.compress(buffer.getvalue())
        with zipfile.ZipFile(zipped_path, mode="w") as zf:
            zf.writestr("test", gz)
    return zipped_path


@pytest.fixture
def image() -> mioi.Image:
    return mioi.Image(np.ones(NIFTI_IMAGE_SHAPE, dtype=np.float32))


def test_dicomdir_from_path(dicom_image_dir: pathlib.Path) -> None:
    dcmdir = miod.DICOMDir.from_path(dicom_image_dir)
    dcmdir.validate()
    assert len(dcmdir) == NUM_DUPLICATES


def test_dicomdir_from_dataset(dicom_image: pydicom.Dataset) -> None:
    datasets = [dicom_image]
    dcmdir = miod.DICOMDir.from_datasets(datasets)
    dcmdir.validate()
    assert len(dcmdir) == 1


def test_dicomdir_from_zipped_stream(zipped_dicom_path: pathlib.Path) -> None:
    with open(zipped_dicom_path, "rb") as f:
        dcmdir = miod.DICOMDir.from_zipped_stream(f)
    dcmdir.validate()
    assert len(dcmdir) == 1


def test_dicomimage_from_path(dicom_image_dir: pathlib.Path) -> None:
    image: miod.DICOMImage = miod.DICOMImage.from_path(dicom_image_dir)
    assert image.shape == (TEST_IMAGE_SHAPE + (NUM_DUPLICATES,))


def test_affine_in_image_vs_dicomimage(dicom_image_dir: pathlib.Path) -> None:
    dcm_image: miod.DICOMImage = miod.DICOMImage.from_path(dicom_image_dir)
    image: mioi.Image = mioi.Image.from_path(dicom_image_dir)
    assert dcm_image.shape == image.shape
    assert np.allclose(
        dcm_image.affine, image.affine
    ), f"\nDICOMImage:\n{dcm_image.affine}\nImage:\n{image.affine}"


def test_affine_vs_tio(dicom_image_dir: pathlib.Path) -> None:
    tio = pytest.importorskip("torchio")
    dcm_image: miod.DICOMImage = miod.DICOMImage.from_path(dicom_image_dir)
    image: mioi.Image = mioi.Image.from_path(dicom_image_dir)
    tio_image = tio.ScalarImage(dicom_image_dir)
    assert np.allclose(image.affine, tio_image.affine)
    assert np.allclose(dcm_image.affine, tio_image.affine)


def test_dicomimage_from_zipped_stream(zipped_dicom_path: pathlib.Path) -> None:
    with open(zipped_dicom_path, "rb") as f:
        image: miod.DICOMImage = miod.DICOMImage.from_zipped_stream(f)
    assert image.shape == (TEST_IMAGE_SHAPE + (1,))


def test_dicom_image_from_zipped_stream(zipped_dicom_path: pathlib.Path) -> None:
    with open(zipped_dicom_path, "rb") as f:
        image: mioi.Image = mioi.Image.from_dicom_zipped_stream(f)
    assert image.shape == (TEST_IMAGE_SHAPE + (1,))


def test_dicom_image_from_zipped_stream_encrypted(
    zipped_encrypted_dicom_path: pathlib.Path, encryption_key: builtins.bytes
) -> None:
    with open(zipped_encrypted_dicom_path, "rb") as f:
        image: mioi.Image = mioi.Image.from_dicom_zipped_stream(
            f, encryption_key=encryption_key
        )
    assert image.shape == (TEST_IMAGE_SHAPE + (1,))


def test_nifti_image_from_path(nifti_image_path: pathlib.Path) -> None:
    image: mioi.Image = mioi.Image.from_path(nifti_image_path)
    assert image.shape == NIFTI_IMAGE_SHAPE


def test_nifti_image_from_stream(nifti_image_path: pathlib.Path) -> None:
    with open(nifti_image_path, "rb") as f:
        image: mioi.Image = mioi.Image.from_stream(f)
    assert image.shape == NIFTI_IMAGE_SHAPE


def test_nifti_image_from_zipped_stream(zipped_nifti_path: pathlib.Path) -> None:
    with open(zipped_nifti_path, "rb") as f:
        image: mioi.Image = mioi.Image.from_zipped_stream(
            f, gzipped=False, image_class=nib.Nifti1Image
        )
    assert image.shape == NIFTI_IMAGE_SHAPE


def test_nifti_gzipped_image_from_zipped_stream(
    zipped_nifti_gzipped_path: pathlib.Path,
) -> None:
    with open(zipped_nifti_gzipped_path, "rb") as f:
        image: mioi.Image = mioi.Image.from_zipped_stream(
            f, gzipped=True, image_class=nib.Nifti1Image
        )
    assert image.shape == NIFTI_IMAGE_SHAPE


# flake8: noqa: E501
def test_numpy_ufuncs_on_dicom_image(dicom_image_dir: pathlib.Path) -> None:
    image: miod.DICOMImage = miod.DICOMImage.from_path(dicom_image_dir)
    image_arr: np.ndarray = np.array(image)
    assert image.shape == (TEST_IMAGE_SHAPE + (NUM_DUPLICATES,))
    image = image + 0.0
    image += 1.0
    image *= image
    assert isinstance(image, miod.DICOMImage)
    assert np.all(image == (image_arr + 1.0) ** 2)
    s = f"DICOMImage(shape: ({TEST_IMAGE_SHAPE[0]}, {TEST_IMAGE_SHAPE[1]}, {NUM_DUPLICATES}); spacing: (0.66, 0.66, {DIST}); dtype: float32)"
    assert repr(image) == s
    mask = image == 0.0
    assert isinstance(mask, miod.DICOMImage)
    subimage: miod.DICOMImage = image[:20, :20, :]
    assert isinstance(subimage, miod.DICOMImage)
    image = image.astype(np.float64)
    assert np.allclose(image, (image_arr + 1.0) ** 2)
    assert isinstance(image, miod.DICOMImage)
    s = f"DICOMImage(shape: ({TEST_IMAGE_SHAPE[0]}, {TEST_IMAGE_SHAPE[1]}, {NUM_DUPLICATES}); spacing: (0.66, 0.66, {DIST}); dtype: float64)"
    assert repr(image) == s


# flake8: noqa: E501
def test_numpy_ufuncs_on_image(image: mioi.Image) -> None:
    image += 1.0
    mask: mioi.Image = image > 1.0
    assert isinstance(mask, mioi.Image)
    assert np.all(mask)
    image *= image
    assert np.all(image == 4.0)
    assert isinstance(image, mioi.Image)
    image = image[0:2]
    assert isinstance(image, mioi.Image)
    image[0:2] += 1.0
    s = f"Image(shape: {NIFTI_IMAGE_SHAPE}; spacing: (1, 1, 1); dtype: float32; orientation: RAS+)"
    assert repr(image) == s
    _image: mioi.Image = np.squeeze(image).astype(np.float16)  # type: ignore[assignment]
    assert np.all(_image >= 4.0)
    s = f"Image(shape: {NIFTI_IMAGE_SHAPE}; spacing: (1, 1, 1); dtype: float16; orientation: RAS+)"
    assert repr(_image) == s
    image = image.ravel()
    assert np.all(image >= 4.0)
    assert isinstance(image, mioi.Image)
    image = image.view(mioi.Image)
    assert np.all(image >= 4.0)
    assert isinstance(image, mioi.Image)
    _image2 = image.view(np.ndarray)
    assert np.all(_image2 >= 4.0)
    assert isinstance(image, np.ndarray)
    assert isinstance(_image2, np.ndarray)


def test_convert_to_torch(image: mioi.Image) -> None:
    torch = pytest.importorskip("torch")
    torch.as_tensor(image.torch_compatible())


def test_resize(image: mioi.Image, dicom_image: pydicom.Dataset) -> None:
    new_shape = np.array(image.shape) * 2
    resized = image.resample_image(new_shape)  # type: ignore[arg-type]
    assert resized.shape == tuple(new_shape)
    orig_spacing: np.ndarray = np.array(image.spacing)
    new_spacing: np.ndarray = np.array(resized.spacing)
    assert np.allclose(orig_spacing, new_spacing * 2)
    assert resized.dtype == image.dtype

    datasets = [dicom_image]
    dcmdir = miod.DICOMDir.from_datasets(datasets)
    dcmimg: miod.DICOMImage = miod.DICOMImage.from_dicomdir(dcmdir)
    image = mioi.Image.from_dicom_image(dcmimg)
    new_shape = np.array(image.shape) * 2
    resized = image.resample_image(new_shape)  # type: ignore[arg-type]
    assert resized.shape == tuple(new_shape)
    orig_spacing = np.array(image.spacing)
    new_spacing = np.array(resized.spacing)
    assert np.allclose(orig_spacing, new_spacing * 2)
    assert resized.dtype == image.dtype


def test_save_image(
    tmp_path_factory: pytest.TempPathFactory, image: mioi.Image
) -> None:
    save_path = tmp_path_factory.mktemp("save").resolve(strict=True)
    image.save(save_path / "test.nii.gz")


def test_save_image_npz(image: mioi.Image) -> None:
    with io.BytesIO() as buffer:
        assert np.all(image == 1.0)
        image.to_npz(buffer)
        assert np.all(image == 1.0)
        buffer.seek(0)
        new_image: mioi.Image = mioi.Image.from_npz(buffer)
        assert np.all(image == new_image)


def test_to_sitk(image: mioi.Image) -> None:
    assert isinstance(image.to_sitk(), sitk.Image)


def test_to_nibabel(image: mioi.Image) -> None:
    assert isinstance(image.to_nibabel(), nib.Nifti1Image)
