import builtins
import typing


def is_iterable(x: typing.Any) -> builtins.bool:
    try:
        _ = iter(x)
    except TypeError:
        return False
    else:
        return True


def unzip(
    zipped_list: typing.Sequence[typing.Tuple[typing.Any, ...]]
) -> typing.List[typing.List[typing.Any]]:
    return list(zip(*zipped_list))
