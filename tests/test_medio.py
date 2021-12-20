#!/usr/bin/env python

"""Tests for `medio` package."""

import builtins
import gzip
import io
import os
import pathlib
import typing
import zipfile

import nibabel as nib
import numpy as np
import pydicom
import pytest
from pydicom.data import get_testdata_file

import medio.dicom as miod
import medio.image as mioi

NUM_DUPLICATES = 2
TEST_IMAGE_NAME = "CT_small.dcm"
TEST_IMAGE_SHAPE = (128, 128)
NIFTI_IMAGE_SHAPE = (1, 1, 1)


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
    data = np.random.randn(*NIFTI_IMAGE_SHAPE)
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
