=======
History
=======

0.2.11 (2022-03-25)
------------------

* Move non-uniformity check after removing anomalous slices in ``DICOMDir``.

0.2.10 (2022-03-15)
------------------

* Fix bug in ``info`` getter in ``BasicImage`` which retrieved ``affine`` instead of ``info``.

0.2.9 (2022-03-11)
------------------

* Add rudimentary ``resample_image`` to ``BasicImage``.
* Add optional ``info`` string to ``BasicImage``.
* Remove duplicate position images in ``remove_anomalous_image`` method (by default).

0.2.8 (2022-03-04)
------------------

* Bump minimum required numpy version to v1.22 since it's required for type annotations.

0.2.7 (2022-02-24)
------------------

* Fix bug in ``DICOMDir.from_path`` where extension not properly formatted with list of paths.
* Fix readthedocs build.

0.2.6 (2022-02-21)
------------------

* Fix mutation issue; appeared to be because took a view of SimpleITK image instead of a copy
* Only support Python 3.9 (and greater when SimpleITK does so). It takes too long to support
  old versions of Python, and the primary use case would be to support legacy systems,
  e.g., for industry.

0.2.5 (2022-02-20)
------------------

* Copy when creating new ``BasicImage`` by default, because (unexpectedly) unsafe otherwise.

0.2.4 (2022-02-20)
------------------

* Copy when saving image because mutation occuring that makes it unsafe otherwise

0.2.3 (2022-02-20)
------------------

* Revert to use ``.view(np.ndarray)`` to fix accidental mutation of original array

0.2.2 (2022-02-20)
------------------

* Change ``to_npz`` method in ``BasicImage`` to use ``__array__`` to fix bug when saving slice
* Fix various mypy issues in the BasicImage stubs

0.2.1 (2022-02-17)
------------------

* Change ``to_npz`` method in ``BasicImage`` to use ``.base`` to avoid new memory allocation
* Add pre-commit hooks and modify files according to them

0.2.0 (2022-01-27)
------------------

* Renamed import to pymedio from medio to avoid confusion with medio
* Renamed ImageBase to BasicImage because it can be used by itself
* Fixed another bug in affine matrix calculation in pymedio.dicom for consistency with ITK

0.1.9 (2022-01-26)
------------------

* Fixed bug in affine matrix calculation in medio.dicom; now returns same matrix as medio.image

0.1.8 (2022-01-11)
------------------

* Change order of encryption/zipping to reduce file size

0.1.7 (2022-01-04)
------------------

* Add option to decrypt a zipped file

0.1.6 (2021-12-30)
------------------

* Add order option to DICOMImage
* Add numpy load kwargs to ``from_npz`` class method in ImageBase.

0.1.5 (2021-12-29)
------------------

* Make pydicom optional

0.1.4 (2021-12-23)
------------------

* Fix bugs in save image

0.1.3 (2021-12-23)
------------------

* Lazy load DICOM files to reduce peak memory consumption
* Make DICOM and base image classes (more) immutable

0.1.2 (2021-12-22)
------------------

* Make Image classes proper subclasses of ndarray
* Add type hints to support normal numpy operations on Images

0.1.1 (2021-12-21)
------------------

* Avoid version collision

0.1.0 (2021-12-21)
------------------

* First release on PyPI.
