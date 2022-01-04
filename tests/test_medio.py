"""Tests for `medio` package."""

from __future__ import annotations

import builtins
import gzip
import io
import os
import pathlib
import zipfile

import cryptography.fernet as crypto
import nibabel as nib
import numpy as np
import pydicom
import pytest
import SimpleITK as sitk
from pydicom.data import get_testdata_file

import medio.dicom as miod
import medio.image as mioi

try:
    import torch
except (ModuleNotFoundError, ImportError):
    torch = None  # type: ignore[assignment]

NUM_DUPLICATES = 2
TEST_IMAGE_NAME = "CT_small.dcm"
TEST_IMAGE_SHAPE = (128, 128)
NIFTI_IMAGE_SHAPE = (2, 2, 2)


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
    for i in range(NUM_DUPLICATES):
        os.symlink(dicom_image_path.resolve(strict=True), dcm_dir / f"{i}.dcm")
    return dcm_dir


@pytest.fixture(scope="session")
def dicom_image(dicom_image_path: pathlib.Path) -> pydicom.Dataset:
    return pydicom.dcmread(dicom_image_path)


@pytest.fixture(scope="session")
def nifti() -> nib.Nifti1Image:
    data = np.random.randn(*NIFTI_IMAGE_SHAPE).astype(np.float32)
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
    with io.BytesIO() as buffer:
        pydicom.dcmwrite(buffer, dicom_image)
        buffer.seek(0)
        with zipfile.ZipFile(zipped_path, mode="w") as zf:
            zf.writestr("test", fernet.encrypt(buffer.getvalue()))
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
    return mioi.Image(np.zeros(NIFTI_IMAGE_SHAPE, dtype=np.float32))


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
    image = miod.DICOMImage.from_path(dicom_image_dir)
    assert image.shape == (TEST_IMAGE_SHAPE + (NUM_DUPLICATES,))


def test_dicomimage_from_zipped_stream(zipped_dicom_path: pathlib.Path) -> None:
    with open(zipped_dicom_path, "rb") as f:
        image = miod.DICOMImage.from_zipped_stream(f)
    assert image.shape == (TEST_IMAGE_SHAPE + (1,))


def test_dicom_image_from_zipped_stream(zipped_dicom_path: pathlib.Path) -> None:
    with open(zipped_dicom_path, "rb") as f:
        image = mioi.Image.from_dicom_zipped_stream(f)
    assert image.shape == (TEST_IMAGE_SHAPE + (1,))


def test_dicom_image_from_zipped_stream_encrypted(
    zipped_encrypted_dicom_path: pathlib.Path, encryption_key: builtins.bytes
) -> None:
    with open(zipped_encrypted_dicom_path, "rb") as f:
        image = mioi.Image.from_dicom_zipped_stream(f, encryption_key=encryption_key)
    assert image.shape == (TEST_IMAGE_SHAPE + (1,))


def test_nifti_image_from_path(nifti_image_path: pathlib.Path) -> None:
    image = mioi.Image.from_path(nifti_image_path)
    assert image.shape == NIFTI_IMAGE_SHAPE


def test_nifti_image_from_stream(nifti_image_path: pathlib.Path) -> None:
    with open(nifti_image_path, "rb") as f:
        image = mioi.Image.from_stream(f)
    assert image.shape == NIFTI_IMAGE_SHAPE


def test_nifti_image_from_zipped_stream(zipped_nifti_path: pathlib.Path) -> None:
    with open(zipped_nifti_path, "rb") as f:
        image = mioi.Image.from_zipped_stream(
            f, gzipped=False, image_class=nib.Nifti1Image
        )
    assert image.shape == NIFTI_IMAGE_SHAPE


def test_nifti_gzipped_image_from_zipped_stream(
    zipped_nifti_gzipped_path: pathlib.Path,
) -> None:
    with open(zipped_nifti_gzipped_path, "rb") as f:
        image = mioi.Image.from_zipped_stream(
            f, gzipped=True, image_class=nib.Nifti1Image
        )
    assert image.shape == NIFTI_IMAGE_SHAPE


def test_numpy_ufuncs_on_dicom_image(dicom_image_dir: pathlib.Path) -> None:
    image = miod.DICOMImage.from_path(dicom_image_dir)
    assert image.shape == (TEST_IMAGE_SHAPE + (NUM_DUPLICATES,))
    image = image + 0.0
    image += 1.0
    image *= image
    assert isinstance(image, miod.DICOMImage)
    s = f"DICOMImage(shape: (128, 128, {NUM_DUPLICATES}); spacing: (0.66, 0.66, 0.00); dtype: float32)"
    assert str(image) == s
    mask = image == 0.0
    assert isinstance(mask, miod.DICOMImage)
    subimage = image[:20, :20, :]
    assert isinstance(subimage, miod.DICOMImage)
    image = image.astype(np.float16)
    assert isinstance(image, miod.DICOMImage)
    s = f"DICOMImage(shape: (128, 128, {NUM_DUPLICATES}); spacing: (0.66, 0.66, 0.00); dtype: float16)"
    assert str(image) == s


def test_numpy_ufuncs_on_image(image: mioi.Image) -> None:
    image += 1.0
    mask = image == 1.0
    assert isinstance(mask, mioi.Image)
    assert np.all(mask)
    image *= image
    assert np.all(image == 1.0)
    assert isinstance(image, mioi.Image)
    image = image[0:2]
    assert isinstance(image, mioi.Image)
    image[0:2] = 0.0
    s = f"Image(shape: {NIFTI_IMAGE_SHAPE}; spacing: (1.00, 1.00, 1.00); dtype: float32; orientation: RAS+)"
    assert str(image) == s
    _image = np.squeeze(image).astype(np.float16)
    s = f"Image(shape: {NIFTI_IMAGE_SHAPE}; spacing: (1.00, 1.00, 1.00); dtype: float16; orientation: RAS+)"
    assert str(_image) == s
    image = image.ravel()
    assert isinstance(image, mioi.Image)
    image = image.view(mioi.Image)
    assert isinstance(image, mioi.Image)
    _image2 = image.view(np.ndarray)
    assert isinstance(image, np.ndarray)


@pytest.mark.skipif(torch is None, reason="Requires torch")
def test_convert_to_torch(image: mioi.Image) -> None:
    torch.as_tensor(image.torch_compatible())


def test_save_image(
    tmp_path_factory: pytest.TempPathFactory, image: mioi.Image
) -> None:
    save_path = tmp_path_factory.mktemp("save").resolve(strict=True)
    image.save(save_path / "test.nii.gz")


def test_save_image_npz(image: mioi.Image) -> None:
    with io.BytesIO() as buffer:
        image.to_npz(buffer)
        buffer.seek(0)
        new_image: mioi.Image = mioi.Image.from_npz(buffer)
        assert np.all(image == new_image)


def test_to_sitk(image: mioi.Image) -> None:
    assert isinstance(image.to_sitk(), sitk.Image)


def test_to_nibabel(image: mioi.Image) -> None:
    assert isinstance(image.to_nibabel(), nib.Nifti1Image)
