=======
History
=======

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
