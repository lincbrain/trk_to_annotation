import sys
from enum import IntEnum
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import numpy as np


NATIVE_BYTEORDER = '<' if sys.byteorder == 'little' else '>'


class Whence(IntEnum):
    START = 0
    CURRENT = 1
    END = 2


class smart_open:
    """
    A smart file opener that can handle file paths and file-like objects.

    If a file path is provided, it opens the file and ensures it is closed
    after use. If a file-like object is provided, it uses it directly and
    restores its position after use.
    """

    def __init__(self, file: str | Path | BinaryIO, *args, **kwargs) -> None:
        # Offset to start reading from
        self._start_offset: int = kwargs.pop("offset", None)
        # Arguments for opening the file
        self._file = file
        self._args = args
        self._kwargs = kwargs
        self._mine = not hasattr(self._file, 'read')
        # Current position in the file-like object, to be restored on exit
        self._offset = 0
        if not self._mine:
            self._offset = self._file.tell()

    def __enter__(self) -> BinaryIO:
        if self._mine:
            self._fileobj = open(self._file, *self._args, **self._kwargs)
        else:
            self._fileobj = self._file
        if self._start_offset is not None:
            self._fileobj.seek(self._start_offset)
        return self._fileobj

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._mine:
            self._fileobj.close()
        else:
            self._fileobj.seek(self._offset)


def smart_read(
    file: BinaryIO, dtype: np.dtype, count: int | None = None
) -> np.ndarray | np.number:
    """Read one or more values from a binary file.

    Parameters
    ----------
    file : BinaryIO
        The binary file object to read from.
    dtype : np.dtype
        The data type of the values to read.
    count : int | None, default=None
        The number of values to read.
        If None, read a single value and return it.
        Otherwise, read 'count' values and return them in an array.

    Returns
    -------
    np.ndarray | np.number
        The read value(s). If count is None, returns a single value.
    """
    if isinstance(file, BytesIO):
        # Cannot use np.fromfile on a BytesIO
        file = file.read((count or 1) * np.dtype(dtype).itemsize)

    if isinstance(file, bytes):
        value = np.frombuffer(file, dtype=dtype, count=count or 1)
    else:
        value = np.fromfile(file, dtype=dtype, count=count or 1)

    if count is None:
        value = value[0]
    return value
