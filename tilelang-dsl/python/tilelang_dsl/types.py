# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Public type markers for the TileLang DSL v1 surface."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


@dataclass(frozen=True)
class ScalarType:
    name: str

    def __repr__(self) -> str:
        return self.name


class TensorView:
    """Bare TensorView annotation marker for TileLang DSL v1."""


class PartitionTensorView:
    """Bare PartitionTensorView annotation marker for TileLang DSL v1."""


class Tile:
    """Bare Tile annotation marker for TileLang DSL v1."""


@dataclass(frozen=True)
class PointerType:
    element_dtype: ScalarType
    memory_space: "MemorySpace"

    def __repr__(self) -> str:
        return f"ptr({self.element_dtype!r}, {self.memory_space!r})"


@dataclass(frozen=True)
class VRegType:
    element_dtype: ScalarType
    lanes: int

    def __repr__(self) -> str:
        return f"vreg({self.element_dtype!r})"


@dataclass(frozen=True)
class MaskType:
    granularity: str

    def __repr__(self) -> str:
        return f"mask_{self.granularity}"


@dataclass(frozen=True)
class WildcardType:
    name: str

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class TypeVariable:
    name: str

    def __repr__(self) -> str:
        return f"TypeVar({self.name!r})"


class MemorySpace(str, Enum):
    GM = "gm"
    UB = "ub"


class BLayout(str, Enum):
    ROW_MAJOR = "row_major"
    COL_MAJOR = "col_major"


class SLayout(str, Enum):
    NONE_BOX = "none_box"


class PadValue(str, Enum):
    ZERO = "zero"


class Pipe(str, Enum):
    MTE1 = "PIPE_MTE1"
    MTE2 = "PIPE_MTE2"
    V = "PIPE_V"
    MTE3 = "PIPE_MTE3"
    ALL = "PIPE_ALL"


class Event(str, Enum):
    ID0 = "EVENT_ID0"
    ID1 = "EVENT_ID1"
    ID2 = "EVENT_ID2"
    ID3 = "EVENT_ID3"
    ID4 = "EVENT_ID4"
    ID5 = "EVENT_ID5"
    ID6 = "EVENT_ID6"
    ID7 = "EVENT_ID7"
    ID8 = "EVENT_ID8"
    ID9 = "EVENT_ID9"
    ID10 = "EVENT_ID10"
    ID11 = "EVENT_ID11"
    ID12 = "EVENT_ID12"
    ID13 = "EVENT_ID13"
    ID14 = "EVENT_ID14"
    ID15 = "EVENT_ID15"
    ID16 = "EVENT_ID16"
    ID17 = "EVENT_ID17"
    ID18 = "EVENT_ID18"
    ID19 = "EVENT_ID19"
    ID20 = "EVENT_ID20"
    ID21 = "EVENT_ID21"
    ID22 = "EVENT_ID22"
    ID23 = "EVENT_ID23"
    ID24 = "EVENT_ID24"
    ID25 = "EVENT_ID25"
    ID26 = "EVENT_ID26"
    ID27 = "EVENT_ID27"
    ID28 = "EVENT_ID28"
    ID29 = "EVENT_ID29"
    ID30 = "EVENT_ID30"
    ID31 = "EVENT_ID31"


class MaskPattern(str, Enum):
    ALL = "PAT_ALL"
    ALLF = "PAT_ALLF"
    EVEN = "PAT_EVEN"
    ODD = "PAT_ODD"
    VL16 = "PAT_VL16"
    VL32 = "PAT_VL32"


class PadMode(str, Enum):
    PadNull = "PadNull"
    PadFirstElem = "PadFirstElem"
    PadValue = "PadValue"


class PositionMode(str, Enum):
    LOWEST = "POS_LOWEST"


class OrderMode(str, Enum):
    ASC = "ORDER_ASC"


def _coerce_int_config_value(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"TileConfig field '{field_name}' must be an integer")
    return value


def _coerce_enum_config_value(
    value: Any,
    enum_type: type[Enum],
    field_name: str,
    default: Enum,
) -> Enum:
    if value is None:
        return default
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        for candidate in enum_type:
            if value in {candidate.name, candidate.value}:
                return candidate
    raise TypeError(f"TileConfig field '{field_name}' must be a {enum_type.__name__} or matching string")


@dataclass(frozen=True)
class TileConfig:
    fields: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "TileConfig":
        return cls(tuple(sorted(mapping.items())))

    def _field(self, *names: str) -> Any | None:
        values = dict(self.fields)
        for name in names:
            if name in values:
                return values[name]
        return None

    @property
    def b_layout(self) -> BLayout:
        return _coerce_enum_config_value(
            self._field("b_layout", "layout"),
            BLayout,
            "b_layout",
            BLayout.ROW_MAJOR,
        )

    @property
    def s_layout(self) -> SLayout:
        return _coerce_enum_config_value(
            self._field("s_layout", "slayout"),
            SLayout,
            "s_layout",
            SLayout.NONE_BOX,
        )

    @property
    def s_fractal_size(self) -> int:
        value = self._field("s_fractal_size", "fractal")
        if value is None:
            return 512
        return _coerce_int_config_value(value, "s_fractal_size")

    @property
    def pad_value(self) -> PadValue:
        return _coerce_enum_config_value(
            self._field("pad_value", "pad"),
            PadValue,
            "pad_value",
            PadValue.ZERO,
        )


@dataclass(frozen=True)
class TileLayoutDescriptor:
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    byte_strides: tuple[int, ...]
    offset: int = 0


@dataclass(frozen=True)
class TileSpecialization:
    shape: tuple[int, ...]
    memory_space: MemorySpace
    config: TileConfig | None = None
    valid_shape: tuple[int | None, ...] | None = None


i8 = ScalarType("i8")
i1 = ScalarType("i1")
i16 = ScalarType("i16")
i32 = ScalarType("i32")
i64 = ScalarType("i64")
f16 = ScalarType("f16")
bf16 = ScalarType("bf16")
f32 = ScalarType("f32")
PIPE = Pipe
EVENT = Event
PAT = MaskPattern
AnyFloat = WildcardType("AnyFloat")
AnyInt = WildcardType("AnyInt")
AnyType = WildcardType("AnyType")
AnyMask = WildcardType("AnyMask")
mask_b8 = MaskType("b8")
mask_b16 = MaskType("b16")
mask_b32 = MaskType("b32")


def TypeVar(name: str) -> TypeVariable:
    if not isinstance(name, str) or not name:
        raise TypeError("TypeVar name must be a non-empty string")
    return TypeVariable(name)


def ptr(dtype: ScalarType, memory_space: MemorySpace) -> PointerType:
    if not isinstance(dtype, ScalarType):
        raise TypeError("ptr() expects a TileLang scalar dtype")
    if not isinstance(memory_space, MemorySpace):
        raise TypeError("ptr() expects a TileLang MemorySpace")
    return PointerType(element_dtype=dtype, memory_space=memory_space)


def vreg(dtype: ScalarType) -> VRegType:
    if not isinstance(dtype, ScalarType):
        raise TypeError("vreg() expects a TileLang scalar dtype")
    return VRegType(element_dtype=dtype, lanes=get_lanes(dtype))


def bytewidth(dtype: ScalarType) -> int:
    if not isinstance(dtype, ScalarType):
        raise TypeError("bytewidth expects a TileLang scalar dtype")
    byte_widths = {
        "i8": 1,
        "i16": 2,
        "i32": 4,
        "f16": 2,
        "bf16": 2,
        "f32": 4,
    }
    width = byte_widths.get(dtype.name)
    if width is None:
        raise TypeError(f"dtype `{dtype.name}` is not supported by bytewidth")
    return width


def get_lanes(dtype: ScalarType) -> int:
    return 256 // bytewidth(dtype)


def elements_per_vreg(dtype: ScalarType) -> int:
    return get_lanes(dtype)


def constexpr(value: bool) -> bool:
    return value


def tile_strides(
    shape: tuple[int, ...],
    config: TileConfig | None = None,
) -> tuple[int, ...]:
    if not shape:
        return ()
    normalized = TileConfig() if config is None else config
    if normalized.b_layout == BLayout.COL_MAJOR and len(shape) == 2:
        return (1, shape[0])
    strides = [1]
    for dim in reversed(shape[1:]):
        strides.insert(0, strides[0] * dim)
    return tuple(strides)


def tile_byte_strides(
    shape: tuple[int, ...],
    dtype: ScalarType,
    config: TileConfig | None = None,
) -> tuple[int, ...]:
    element_bytes = bytewidth(dtype)
    return tuple(stride * element_bytes for stride in tile_strides(shape, config))


def tile_layout_descriptor(
    shape: tuple[int, ...],
    dtype: ScalarType,
    config: TileConfig | None = None,
    *,
    offset: int = 0,
) -> TileLayoutDescriptor:
    return TileLayoutDescriptor(
        shape=shape,
        strides=tile_strides(shape, config),
        byte_strides=tile_byte_strides(shape, dtype, config),
        offset=offset,
    )


__all__ = [
    "ScalarType",
    "WildcardType",
    "TypeVariable",
    "TypeVar",
    "TensorView",
    "PartitionTensorView",
    "Tile",
    "PointerType",
    "VRegType",
    "MaskType",
    "ptr",
    "vreg",
    "MemorySpace",
    "BLayout",
    "SLayout",
    "PadValue",
    "Pipe",
    "Event",
    "PIPE",
    "EVENT",
    "MaskPattern",
    "PAT",
    "PadMode",
    "PositionMode",
    "OrderMode",
    "TileConfig",
    "TileSpecialization",
    "i1",
    "i8",
    "i16",
    "i32",
    "i64",
    "f16",
    "bf16",
    "f32",
    "AnyFloat",
    "AnyInt",
    "AnyType",
    "AnyMask",
    "mask_b8",
    "mask_b16",
    "mask_b32",
    "constexpr",
    "bytewidth",
    "get_lanes",
    "elements_per_vreg",
]
