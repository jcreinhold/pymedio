import builtins
import typing

import numpy as np
import numpy.typing as npt

import medio.typing as miot

_UfuncMethod = typing.Literal[
    "__call__", "reduce", "reduceat", "accumulate", "outer", "inner"
]
_Image = typing.TypeVar("_Image", bound="ImageBase")
_ArrayNumber_co = npt.NDArray[typing.Union[np.bool_, np.number[typing.Any]]]

class ImageBase(np.ndarray):
    _HANDLED_TYPES: typing.ClassVar[typing.Tuple[typing.Type, ...]]
    _affine: npt.NDArray
    def __new__(  # type: ignore[misc]
        cls, data: npt.ArrayLike, affine: typing.Optional[npt.NDArray] = ...
    ) -> _Image: ...
    def __array_finalize__(self, obj: _Image) -> None: ...
    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: _UfuncMethod,
        *inputs: typing.Any,
        **kwargs: typing.Any,
    ) -> typing.Any: ...
    @property
    def str_properties(self) -> typing.List[builtins.str]: ...
    @property
    def affine(self) -> npt.NDArray: ...
    @affine.setter
    def affine(self, new_affine: npt.NDArray) -> None: ...
    @property
    def direction(self) -> miot.Direction: ...
    @property
    def spacing(self) -> typing.Tuple[builtins.float, ...]: ...
    @property
    def origin(self) -> typing.Tuple[builtins.float, ...]: ...
    @property
    def memory(self) -> builtins.int: ...
    def get_spacing_string(self) -> builtins.str: ...
    def to_npz(self, file: typing.Union[miot.PathLike, builtins.bytes]) -> None: ...
    @classmethod
    def from_npz(cls, file: typing.Union[miot.PathLike, builtins.bytes]) -> _Image: ...
    def torch_compatible(self) -> npt.NDArray: ...
    def __getitem__(self: _Image, key: typing.Any) -> _Image: ...
    def __lt__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __le__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __gt__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __ge__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __abs__(self: _Image) -> _Image: ...
    def __invert__(self: _Image) -> _Image: ...
    def __neg__(self: _Image) -> _Image: ...
    def __matmul__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __rmatmul__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __mod__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __rmod__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __divmod__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __rdivmod__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __add__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __radd__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __iadd__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __sub__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __rsub__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __isub__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __mul__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __rmul__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __imul__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __floordiv__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __rfloordiv__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __ifloordiv__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __pow__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __rpow__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __ipow__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __truediv__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __rtruediv__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __itruediv__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __lshift__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __rlshift__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __ilshift__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __rshift__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __rrshift__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __irshift__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __and__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __rand__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __iand__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __xor__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __rxor__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __ixor__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __or__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __ror__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def __ior__(self: _Image, other: typing.Any) -> _Image: ...  # type: ignore[override]
    def byteswap(self: _Image, inplace: builtins.bool = ...) -> _Image: ...
    def squeeze(
        self: _Image,
        axis: typing.Union[
            typing.SupportsIndex, typing.Tuple[typing.SupportsIndex, ...]
        ] = ...,
    ) -> _Image: ...
    def swapaxes(
        self: _Image,
        axis1: typing.SupportsIndex,
        axis2: typing.SupportsIndex,
    ) -> _Image: ...
    @typing.overload
    def transpose(self: _Image, axes: npt._ShapeLike, /) -> _Image: ...
    @typing.overload
    def transpose(self: _Image, *axes: typing.SupportsIndex) -> _Image: ...
    def dot(self: _Image, b: npt.ArrayLike, out: None = ...) -> _Image: ...  # type: ignore[override]
    def astype(self: _Image, dtype: npt.DTypeLike, order: builtins.str = ..., casting: builtins.str = ..., subok: builtins.bool = ..., copy: builtins.bool = ...) -> _Image: ...  # type: ignore[override]
    def reshape(self: _Image, shape: npt._ShapeLike, order: builtins.str = ...) -> _Image: ...  # type: ignore[override]
    def view(self: _Image, dtype: npt.DTypeLike = ..., *args: typing.Any, **kwargs: typing.Any) -> _Image: ...  # type: ignore[override]
    def flatten(self: _Image, order: builtins.str = ...) -> _Image: ...  # type: ignore[override]
    def ravel(self: _Image, order: builtins.str = ...) -> _Image: ...  # type: ignore[override]
