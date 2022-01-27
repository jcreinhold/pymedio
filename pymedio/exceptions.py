"""Project exceptions
Author: Jacob Reinhold <jcreinhold@gmail.com>
"""


class MedIOException(Exception):
    ...


class DicomImportException(MedIOException):
    ...


class OutsideMaxNonUniformity(MedIOException):
    ...


class MissingSlicesException(MedIOException):
    ...
