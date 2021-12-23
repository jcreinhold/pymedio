"""miscellaneous project functions
Author: Jacob Reinhold <jcreinhold@gmail.com>
"""

__all__ = [
    "check_uint_to_int",
    "ensure_4d",
    "flipxy_33",
    "flipxy_44",
    "get_metadata_from_ras_affine",
    "get_rotation_and_spacing_from_affine",
    "is_iterable",
    "unzip",
]

import builtins
import typing

import numpy as np
import numpy.typing as npt

import medio.typing as miot


# Matrices used to switch between LPS and RAS
def flipxy_33() -> npt.NDArray:
    matrix: npt.NDArray = np.diag([-1, -1, 1])
    return matrix


def flipxy_44() -> npt.NDArray:
    matrix: npt.NDArray = np.diag([-1, -1, 1, 1])
    return matrix


def is_iterable(x: typing.Any) -> builtins.bool:
    try:
        _ = iter(x)
    except TypeError:
        return False
    else:
        return True


def unzip(
    zipped_list: typing.Iterable[typing.Tuple[typing.Any, ...]]
) -> typing.Iterable[typing.Tuple[typing.Any, ...]]:
    return zip(*zipped_list)


def get_rotation_and_spacing_from_affine(
    affine: npt.NDArray,
) -> typing.Tuple[npt.NDArray, npt.NDArray]:
    # From https://github.com/nipy/nibabel/blob/master/nibabel/orientations.py
    rotation_zoom = affine[:3, :3]
    spacing = np.sqrt(np.sum(rotation_zoom * rotation_zoom, axis=0))
    rotation = rotation_zoom / spacing
    return rotation, spacing


def get_metadata_from_ras_affine(
    affine: npt.NDArray,
    *,
    is_2d: builtins.bool = False,
    lps: builtins.bool = True,
) -> typing.Tuple[miot.TripletFloat, miot.TripletFloat, miot.Direction]:
    direction_ras, spacing_array = get_rotation_and_spacing_from_affine(affine)
    origin_ras = affine[:3, 3]
    _flipxy_33 = flipxy_33()
    origin_lps = np.dot(_flipxy_33, origin_ras)
    direction_lps = np.dot(_flipxy_33, direction_ras)
    if is_2d:  # ignore orientation if 2D (1, W, H, 1)
        direction_lps = np.diag((-1, -1)).astype(np.float64)
        direction_ras = np.diag((1, 1)).astype(np.float64)
    origin_array = origin_lps if lps else origin_ras
    direction_array = direction_lps if lps else direction_ras
    direction_array = direction_array.flatten()
    # The following are to comply with typing hints
    # (there must be prettier ways to do this)
    ox, oy, oz = origin_array
    sx, sy, sz = spacing_array
    direction: miot.Direction
    if is_2d:
        d1, d2, d3, d4 = direction_array
        direction = d1, d2, d3, d4
    else:
        d1, d2, d3, d4, d5, d6, d7, d8, d9 = direction_array
        direction = d1, d2, d3, d4, d5, d6, d7, d8, d9
    origin = ox, oy, oz
    spacing = sx, sy, sz
    return origin, spacing, direction


def ensure_4d(
    array: npt.NDArray, *, num_spatial_dims: typing.Optional[builtins.int] = None
) -> npt.NDArray:
    """for PyTorch"""
    num_dimensions = array.ndim
    if num_dimensions == 4:
        pass
    elif num_dimensions == 5:  # hope (W, H, D, 1, C)
        if array.shape[-2] == 1:
            array = array[..., 0, :]
            array = array.transpose((3, 0, 1, 2))
        else:
            raise ValueError("5D is not supported for shape[-2] > 1")
    elif num_dimensions == 2:  # assume 2D monochannel (W, H)
        array = array[np.newaxis, ..., np.newaxis]  # (1, W, H, 1)
    elif num_dimensions == 3:  # 2D multichannel or 3D monochannel?
        if num_spatial_dims == 2:
            array = array[..., np.newaxis]  # (C, W, H, 1)
        elif num_spatial_dims == 3:  # (W, H, D)
            array = array[np.newaxis]  # (1, W, H, D)
        else:  # try to guess
            shape = array.shape
            maybe_rgb = 3 in (shape[0], shape[-1])
            if maybe_rgb:
                if shape[-1] == 3:  # (W, H, 3)
                    array = array.transpose((2, 0, 1))  # (3, W, H)
                array = array[..., np.newaxis]  # (3, W, H, 1)
            else:  # (W, H, D)
                array = array[np.newaxis]  # (1, W, H, D)
    else:
        message = f"{num_dimensions}D images not supported yet."
        raise NotImplementedError(message)
    assert array.ndim == 4
    return array


def check_uint_to_int(array: npt.NDArray) -> npt.NDArray:
    """convert to int b/c PyTorch won't take uint16 nor uint32"""
    if array.dtype == np.uint16:
        return array.astype(np.int32)
    elif array.dtype == np.uint32:
        return array.astype(np.int64)
    else:
        return array
