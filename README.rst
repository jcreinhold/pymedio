=======
pymedio
=======

.. image:: https://img.shields.io/pypi/v/pymedio.svg
        :target: https://pypi.python.org/pypi/pymedio

.. image:: https://readthedocs.org/projects/pymedio/badge/?version=latest
        :target: https://pymedio.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

.. image:: https://img.shields.io/pypi/pyversions/pymedio
        :target: https://www.python.org/

Read arbitrary medical images in Python with various backends for scientific
computing.

Basically, this package is just a modified version of
`torchio <https://github.com/fepegar/torchio>`_ [1]_
and `dicom-numpy <https://github.com/innolitics/dicom-numpy>`_
which eagerly loads images (depending on the backend and settings) and returns
them as `NumPy <https://numpy.org/>`_ arrays (instead of PyTorch tensors, as
is the case in torchio).

There are also various factory functions to load and work with images from
memory buffers instead of from disk which is preferable in certain environments,
e.g., AWS Lambda.

The main motivation for yet another medical image reader is that I wanted the
flexibility of torchio in opening almost any medical image type, without requiring
PyTorch and without the automatic casting of images to arrays of 32-bit floating
point precision, which is an unnecessary and memory-intensive operation in many
scientific computing use cases.

* Free software: MIT license
* Documentation: https://pymedio.readthedocs.io.

Install
-------

The easiest way to install the package is through the following command::

    pip install pymedio

To install from the source directory, clone the repo and run::

    python setup.py install

To make the package as portable as possible, the package only comes with numpy installed by default.

If you want to load only DICOM images (using `pydicom <https://github.com/pydicom/pydicom>`_) install with::

    pip install "pymedio[dicom]"

If you want to load DICOM and non-DICOM images (using `SimpleITK <https://simpleitk.org/>`_ or
`nibabel <https://nipy.org/nibabel/>`_ as a backend), install with::

    pip install "pymedio[all]"

Basic Usage
-----------

Say you have a directory of DICOM images at the path ``dicom_dir``, and you installed the package with ``dicom``
extras, then you can open it with:

.. code-block:: python

    import pymedio.dicom as miod
    image = miod.DICOMImage.from_path("dicom_dir")

This uses `pydicom <https://github.com/pydicom/pydicom>`_ as a backend to open the image à la
`dicom-numpy <https://github.com/innolitics/dicom-numpy>`_.

If you have a NIfTI image at ``image.nii``, and you installed the package with ``all`` extras, you can open it with:

.. code-block:: python

    import pymedio.image as mioi
    image = mioi.Image.from_path("image.nii")

In either case, you can proceed to work with the image data like a normal numpy array, e.g.,

.. code-block:: python

    image += 1.0
    image *= image

Note that the image will have a ``affine`` attribute which stores the (affine) coordinate transformation
matrix to/from scanner coordinates/voxels (even after applying numpy functions/operations to the image).

You can convert an either image class to a torch tensor—if you have it installed—like:

.. code-block:: python

    import torch
    tensor = torch.as_tensor(image.torch_compatible())

If you want to save an image as a non-standard data type (e.g., a 16-bit floating point number), do so with:

.. code-block:: python

    # optionally convert the image to the desired type
    f16_image = image.astype(np.float16)
    f16_image.to_npz("image-f16.npz")

Then, in a more resource-constrained environment, you can install pymedio without any extras and run, e.g.,

.. code-block:: python

    from pymedio.base import ImageBase as Image
    f16_image = Image.from_npz("image-f16.npz")
    print(f16_image.affine)

To view the image in a standard reader following some processing (assuming ``all`` extras installed),
you can then do, e.g.,

.. code-block:: python

    import pymedio.image as mioi
    image = mioi.Image.from_npz("image-f16.npz")
    image.astype(np.float32).save("image.nii.gz")

References
----------

.. [1] Pérez-García, Fernando, Rachel Sparks, and Sebastien Ourselin. "TorchIO: a Python library for efficient loading,
       preprocessing, augmentation and patch-based sampling of medical images in deep learning." Computer Methods and
       Programs in Biomedicine (2021): 106236.
