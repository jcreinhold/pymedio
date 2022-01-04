=======
History
=======

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
