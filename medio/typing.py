# -*- coding: utf-8 -*-
"""
Author: Jacob Reinhold
Created on: 15 Dec 2021
"""

import builtins
import os
import typing

import numpy.typing as npt

# https://www.python.org/dev/peps/pep-0519/#provide-specific-type-hinting-support
PathLike = typing.Union[builtins.str, os.PathLike]
DataAffine = typing.Tuple[npt.NDArray, npt.NDArray]
Shape = typing.Tuple[builtins.int, builtins.int, builtins.int, builtins.int]
TripletFloat = typing.Tuple[builtins.float, builtins.float, builtins.float]
Direction2D = typing.Tuple[
    builtins.float, builtins.float, builtins.float, builtins.float
]
Direction3D = typing.Tuple[
    builtins.float,
    builtins.float,
    builtins.float,
    builtins.float,
    builtins.float,
    builtins.float,
    builtins.float,
    builtins.float,
    builtins.float,
]
Direction = typing.Union[Direction2D, Direction3D]
