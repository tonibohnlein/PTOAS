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
import struct
from typing import Any, Mapping


@dataclass(frozen=True)
class ScalarType:
    name: str

    def __repr__(self) -> str:
        return self.name


_INTEGER_DTYPE_WIDTHS = {
    "i8": 8,
    "si8": 8,
    "ui8": 8,
    "i16": 16,
    "si16": 16,
    "ui16": 16,
    "i32": 32,
    "si32": 32,
    "ui32": 32,
    "i64": 64,
    "si64": 64,
    "ui64": 64,
}

_INTEGER_DTYPE_SIGNS = {
    "i8": "signless",
    "si8": "signed",
    "ui8": "unsigned",
    "i16": "signless",
    "si16": "signed",
    "ui16": "unsigned",
    "i32": "signless",
    "si32": "signed",
    "ui32": "unsigned",
    "i64": "signless",
    "si64": "signed",
    "ui64": "unsigned",
}

_FLOAT_DTYPE_WIDTHS = {
    "f16": 16,
    "bf16": 16,
    "f32": 32,
}

_DTYPE_BYTE_WIDTHS = {
    name: bits // 8 for name, bits in _INTEGER_DTYPE_WIDTHS.items()
}
_DTYPE_BYTE_WIDTHS.update({name: bits // 8 for name, bits in _FLOAT_DTYPE_WIDTHS.items()})


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
class VectorType:
    element_dtype: ScalarType
    shape: tuple[int, ...]

    def __repr__(self) -> str:
        return f"vector({self.element_dtype!r}, {self.shape!r})"


@dataclass(frozen=True)
class MaskType:
    granularity: str

    def __repr__(self) -> str:
        return f"mask_{self.granularity}"


@dataclass(frozen=True)
class AlignType:
    def __repr__(self) -> str:
        return "align"


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
    MAT = "mat"
    LEFT = "left"
    RIGHT = "right"
    ACC = "acc"
    BIAS = "bias"
    UB = "ub"


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


class BarrierType(str, Enum):
    VV_ALL = "VV_ALL"
    VST_VLD = "VST_VLD"
    VLD_VST = "VLD_VST"
    VST_VST = "VST_VST"
    VS_ALL = "VS_ALL"
    VST_LD = "VST_LD"
    VLD_ST = "VLD_ST"
    VST_ST = "VST_ST"
    SV_ALL = "SV_ALL"
    ST_VLD = "ST_VLD"
    LD_VST = "LD_VST"
    ST_VST = "ST_VST"


class MaskPattern(str, Enum):
    ALL = "PAT_ALL"
    ALLF = "PAT_ALLF"
    EVEN = "PAT_EVEN"
    ODD = "PAT_ODD"
    VL16 = "PAT_VL16"
    VL32 = "PAT_VL32"


class PredicateDist(str, Enum):
    NORM = "NORM"
    US = "US"
    DS = "DS"
    PK = "PK"


class VLoadDist(str, Enum):
    NORM = "NORM"
    BRC_B8 = "BRC_B8"
    BRC_B16 = "BRC_B16"
    BRC_B32 = "BRC_B32"
    US_B8 = "US_B8"
    US_B16 = "US_B16"
    DS_B8 = "DS_B8"
    DS_B16 = "DS_B16"
    UNPK_B8 = "UNPK_B8"
    UNPK_B16 = "UNPK_B16"
    UNPK_B32 = "UNPK_B32"
    BRC_BLK = "BRC_BLK"
    E2B_B16 = "E2B_B16"
    E2B_B32 = "E2B_B32"
    UNPK4 = "UNPK4"
    SPLT4CHN = "SPLT4CHN"
    SPLT2CHN_B8 = "SPLT2CHN_B8"
    SPLT2CHN_B16 = "SPLT2CHN_B16"


class VStoreDist(str, Enum):
    NORM_B8 = "NORM_B8"
    NORM_B16 = "NORM_B16"
    NORM_B32 = "NORM_B32"
    ONE_POINT_B8 = "1PT_B8"
    ONE_POINT_B16 = "1PT_B16"
    ONE_POINT_B32 = "1PT_B32"
    PK_B16 = "PK_B16"
    PK_B32 = "PK_B32"
    PK_B64 = "PK_B64"
    PK4_B32 = "PK4_B32"
    MRG4CHN_B8 = "MRG4CHN_B8"
    MRG2CHN_B8 = "MRG2CHN_B8"
    MRG2CHN_B16 = "MRG2CHN_B16"


class PredicatePart(str, Enum):
    LOWER = "LOWER"
    HIGHER = "HIGHER"


class CmpMode(str, Enum):
    EQ = "eq"
    NE = "ne"
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"


class PadMode(str, Enum):
    PadNull = "PadNull"
    PadFirstElem = "PadFirstElem"
    PadValue = "PadValue"


class BLayout(str, Enum):
    ROW_MAJOR = "row_major"
    COL_MAJOR = "col_major"


class SLayout(str, Enum):
    NONE_BOX = "none_box"
    ROW_MAJOR = "row_major"
    COL_MAJOR = "col_major"


def _float32_from_bits(bits: int) -> float:
    return struct.unpack(">f", bits.to_bytes(4, byteorder="big", signed=False))[0]


_FLOAT_DTYPE_MAX = {
    "f16": 65504.0,
    "bf16": _float32_from_bits(0x7F7F0000),
    "f32": _float32_from_bits(0x7F7FFFFF),
}
_FLOAT_DTYPE_MIN = {
    "f16": -65504.0,
    "bf16": _float32_from_bits(0xFF7F0000),
    "f32": _float32_from_bits(0xFF7FFFFF),
}


@dataclass(frozen=True)
class PadValue:
    """Tile pad descriptor matching the C++ PadValue design.

    Standard values occupy the low integer range:
    - NULL = 0
    - ZERO = 1
    - MAX = 2
    - MIN = 3

    Custom values use the C++ `CustomBase` convention and carry an f32 bit
    pattern authored through `custom_f32(...)`.
    """

    encoded: int
    _symbol_name: str | None = None
    _float32_bits: int | None = None

    CustomBase = 0x100000000
    _STANDARD_TEXT = {
        0: "null",
        1: "zero",
        2: "max",
        3: "min",
    }

    def __post_init__(self) -> None:
        if isinstance(self.encoded, bool) or not isinstance(self.encoded, int):
            raise TypeError("PadValue.encoded must be a uint64-compatible integer")
        if self.encoded < 0 or self.encoded >= (1 << 64):
            raise ValueError("PadValue.encoded must be in uint64 range")
        if self._float32_bits is not None and not (0 <= self._float32_bits < (1 << 32)):
            raise ValueError("PadValue custom float32 payload must be a 32-bit integer")

    @property
    def name(self) -> str:
        if self._symbol_name is not None:
            return self._symbol_name
        return "CUSTOM"

    @property
    def value(self) -> int:
        raise AttributeError(
            "PadValue.value is not available; use PadValue.encoded for host-side payload access "
            "or pad.eval(...) for scalar materialization"
        )

    @property
    def text(self) -> str:
        standard = self._STANDARD_TEXT.get(self.encoded)
        if standard is not None:
            return standard
        return f"0x{self.encoded:016X}"

    @property
    def is_custom(self) -> bool:
        return self._symbol_name is None and self.encoded >= self.CustomBase

    @property
    def float32_bits(self) -> int:
        if not self.is_custom:
            raise ValueError("only custom PadValue instances carry a float32 payload")
        if self._float32_bits is not None:
            return self._float32_bits
        return (self.encoded >> 32) & 0xFFFFFFFF

    def as_float32(self) -> float:
        return _float32_from_bits(self.float32_bits)

    def eval(self, dtype: ScalarType) -> int | float | None:
        if not isinstance(dtype, ScalarType):
            raise TypeError("PadValue.eval expects a TileLang scalar dtype")
        if self == PadValue.NULL:
            return None
        if self == PadValue.ZERO:
            return 0.0 if is_float_dtype(dtype) else 0
        if self == PadValue.MAX:
            if is_float_dtype(dtype):
                return _FLOAT_DTYPE_MAX[dtype.name]
            width = integer_bitwidth(dtype)
            signedness = integer_signedness(dtype)
            if width is None or signedness is None:
                raise TypeError(f"PadValue.MAX does not support dtype `{dtype.name}`")
            if signedness == "unsigned":
                return (1 << width) - 1
            return (1 << (width - 1)) - 1
        if self == PadValue.MIN:
            if is_float_dtype(dtype):
                return _FLOAT_DTYPE_MIN[dtype.name]
            width = integer_bitwidth(dtype)
            signedness = integer_signedness(dtype)
            if width is None or signedness is None:
                raise TypeError(f"PadValue.MIN does not support dtype `{dtype.name}`")
            if signedness == "unsigned":
                return 0
            return -(1 << (width - 1))
        if self.is_custom:
            if not is_float_dtype(dtype):
                raise TypeError(
                    "custom Tile pad_value currently only materializes for floating Tile element dtypes"
                )
            return self.as_float32()
        raise TypeError(f"unsupported PadValue payload {self!r}")

    @classmethod
    def from_uint64(cls, value: int) -> "PadValue":
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("PadValue.from_uint64 expects an integer")
        if value == 0:
            return cls.NULL
        if value == 1:
            return cls.ZERO
        if value == 2:
            return cls.MAX
        if value == 3:
            return cls.MIN
        if value < 0 or value >= (1 << 64):
            raise ValueError("PadValue.from_uint64 expects a uint64-compatible integer")
        return cls(value)

    @classmethod
    def custom_f32(cls, value: float | str | int) -> "PadValue":
        bits = cls._normalize_custom_f32_bits(value)
        encoded = cls.CustomBase | (bits << 32)
        return cls(encoded=encoded, _float32_bits=bits)

    @staticmethod
    def _normalize_custom_f32_bits(value: float | str | int) -> int:
        if isinstance(value, bool):
            raise TypeError("PadValue.custom_f32 does not accept bool")
        if isinstance(value, int):
            if value < 0 or value >= (1 << 32):
                raise ValueError("PadValue.custom_f32 integer payload must fit in 32 bits")
            return value
        if isinstance(value, str):
            text = value.strip()
            if text.lower().startswith("0x"):
                bits = int(text, 16)
                if bits < 0 or bits >= (1 << 32):
                    raise ValueError("PadValue.custom_f32 hex payload must fit in 32 bits")
                return bits
            value = float(text)
        packed = struct.pack(">f", float(value))
        return int.from_bytes(packed, byteorder="big", signed=False)

    def __repr__(self) -> str:
        if self == PadValue.NULL:
            return "PadValue.NULL"
        if self == PadValue.ZERO:
            return "PadValue.ZERO"
        if self == PadValue.MAX:
            return "PadValue.MAX"
        if self == PadValue.MIN:
            return "PadValue.MIN"
        return f"PadValue.custom_f32(0x{self.float32_bits:08X})"


PadValue.NULL = PadValue(0, "NULL")
PadValue.ZERO = PadValue(1, "ZERO")
PadValue.MAX = PadValue(2, "MAX")
PadValue.MIN = PadValue(3, "MIN")


class DeinterleaveDist(str, Enum):
    DINTLV = "DINTLV"
    BDINTLV = "BDINTLV"
    B8 = "DINTLV"
    B16 = "DINTLV"
    B32 = "DINTLV"
    BD = "BDINTLV"


class InterleaveDist(str, Enum):
    INTLV = "INTLV"
    B8 = "INTLV"
    B16 = "INTLV"
    B32 = "INTLV"


class PositionMode(str, Enum):
    LOWEST = "LOWEST"
    HIGHEST = "HIGHEST"


class OrderMode(str, Enum):
    ASC = "ASC"
    DESC = "DESC"


class VcvtRoundMode(str, Enum):
    R = "R"
    A = "A"
    F = "F"
    C = "C"
    Z = "Z"
    O = "O"


class VcvtSatMode(str, Enum):
    SAT = "SAT"
    NOSAT = "NOSAT"


class VcvtPartMode(str, Enum):
    EVEN = "EVEN"
    ODD = "ODD"
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class PostUpdateMode(str, Enum):
    POST_UPDATE = "POST_UPDATE"
    NO_POST_UPDATE = "NO_POST_UPDATE"


class FractalMode(str, Enum):
    ND2NZ = "nd2nz"
    DN2NZ = "dn2nz"
    NZ2ND = "nz2nd"
    NZ2DN = "nz2dn"
    NZ2NZ = "nz2nz"


@dataclass(frozen=True)
class TileConfig:
    fields: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "TileConfig":
        if not isinstance(mapping, Mapping):
            raise TypeError("TileConfig.from_mapping expects a mapping")
        normalized: dict[str, Any] = {}
        for key, value in mapping.items():
            canonical_key = cls._canonical_key(key)
            if canonical_key in normalized:
                raise ValueError(f"duplicate TileConfig field '{canonical_key}'")
            normalized[canonical_key] = cls._normalize_field_value(canonical_key, value)
        return cls(tuple(sorted(normalized.items())))

    @staticmethod
    def _canonical_key(key: Any) -> str:
        if not isinstance(key, str):
            raise TypeError("TileConfig field names must be strings")
        aliases = {
            "layout": "b_layout",
            "blayout": "b_layout",
            "b_layout": "b_layout",
            "slayout": "s_layout",
            "s_layout": "s_layout",
            "fractal": "s_fractal_size",
            "s_fractal_size": "s_fractal_size",
            "pad": "pad_value",
            "pad_value": "pad_value",
        }
        return aliases.get(key, key)

    @staticmethod
    def _normalize_field_value(key: str, value: Any) -> Any:
        if key == "b_layout":
            return TileConfig._normalize_b_layout(value)
        if key == "s_layout":
            return TileConfig._normalize_s_layout(value)
        if key == "s_fractal_size":
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("TileConfig.s_fractal_size must be an integer")
            return value
        if key == "pad_value":
            return TileConfig._normalize_pad_value(value)
        return value

    @staticmethod
    def _normalize_b_layout(value: Any) -> BLayout:
        if isinstance(value, BLayout):
            return value
        if isinstance(value, str):
            normalized = value.strip().upper().replace("-", "_")
            if normalized == "ROW_MAJOR":
                return BLayout.ROW_MAJOR
            if normalized == "COL_MAJOR":
                return BLayout.COL_MAJOR
        raise ValueError(f"unsupported TileConfig b_layout value {value!r}")

    @staticmethod
    def _normalize_s_layout(value: Any) -> SLayout:
        if isinstance(value, SLayout):
            return value
        if isinstance(value, str):
            normalized = value.strip().upper().replace("-", "_")
            if normalized == "NONE_BOX":
                return SLayout.NONE_BOX
            if normalized == "ROW_MAJOR":
                return SLayout.ROW_MAJOR
            if normalized == "COL_MAJOR":
                return SLayout.COL_MAJOR
        raise ValueError(f"unsupported TileConfig s_layout value {value!r}")

    @staticmethod
    def _normalize_pad_value(value: Any) -> PadValue:
        if isinstance(value, PadValue):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return PadValue.from_uint64(value)
        if isinstance(value, str):
            text = value.strip()
            if text.lower().startswith("0x"):
                return PadValue.from_uint64(int(text, 16))
            normalized = value.strip().upper().replace("-", "_")
            if normalized == "NULL":
                return PadValue.NULL
            if normalized == "ZERO":
                return PadValue.ZERO
            if normalized == "MAX":
                return PadValue.MAX
            if normalized == "MIN":
                return PadValue.MIN
        raise ValueError(f"unsupported TileConfig pad_value value {value!r}")

    @property
    def b_layout(self) -> BLayout:
        value = dict(self.fields).get("b_layout", BLayout.ROW_MAJOR)
        return self._normalize_b_layout(value)

    @property
    def s_layout(self) -> SLayout:
        value = dict(self.fields).get("s_layout", SLayout.NONE_BOX)
        return self._normalize_s_layout(value)

    @property
    def s_fractal_size(self) -> int:
        value = dict(self.fields).get("s_fractal_size", 512)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("TileConfig.s_fractal_size must be an integer")
        return value

    @property
    def pad_value(self) -> PadValue:
        value = dict(self.fields).get("pad_value", PadValue.NULL)
        return self._normalize_pad_value(value)

    @classmethod
    def for_memory_space(cls, memory_space: MemorySpace) -> "TileConfig":
        if not isinstance(memory_space, MemorySpace):
            raise TypeError("TileConfig.for_memory_space expects a TileLang MemorySpace")
        defaults: dict[str, Any]
        if memory_space in {MemorySpace.MAT, MemorySpace.LEFT}:
            defaults = {
                "b_layout": BLayout.COL_MAJOR,
                "s_layout": SLayout.ROW_MAJOR,
                "s_fractal_size": 512,
                "pad_value": PadValue.NULL,
            }
        elif memory_space == MemorySpace.RIGHT:
            defaults = {
                "b_layout": BLayout.ROW_MAJOR,
                "s_layout": SLayout.COL_MAJOR,
                "s_fractal_size": 512,
                "pad_value": PadValue.NULL,
            }
        elif memory_space == MemorySpace.ACC:
            defaults = {
                "b_layout": BLayout.COL_MAJOR,
                "s_layout": SLayout.ROW_MAJOR,
                "s_fractal_size": 1024,
                "pad_value": PadValue.NULL,
            }
        elif memory_space == MemorySpace.BIAS:
            defaults = {
                "b_layout": BLayout.ROW_MAJOR,
                "s_layout": SLayout.NONE_BOX,
                "s_fractal_size": 512,
                "pad_value": PadValue.NULL,
            }
        else:
            defaults = {
                "b_layout": BLayout.ROW_MAJOR,
                "s_layout": SLayout.NONE_BOX,
                "s_fractal_size": 512,
                "pad_value": PadValue.NULL,
            }
        return cls(tuple(sorted(defaults.items())))


@dataclass(frozen=True)
class TileSpecialization:
    shape: tuple[int, ...]
    memory_space: MemorySpace
    config: TileConfig | None = None
    valid_shape: tuple[int | None, ...] | None = None


i1 = ScalarType("i1")
i8 = ScalarType("i8")
si8 = ScalarType("si8")
ui8 = ScalarType("ui8")
i16 = ScalarType("i16")
si16 = ScalarType("si16")
ui16 = ScalarType("ui16")
i32 = ScalarType("i32")
si32 = ScalarType("si32")
ui32 = ScalarType("ui32")
i64 = ScalarType("i64")
si64 = ScalarType("si64")
ui64 = ScalarType("ui64")
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
align = AlignType()


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


def vector(dtype: ScalarType, shape: tuple[int, ...] | list[int] | int) -> VectorType:
    if not isinstance(dtype, ScalarType):
        raise TypeError("vector() expects a TileLang scalar dtype")
    if isinstance(shape, int) and not isinstance(shape, bool):
        normalized_shape = (shape,)
    elif isinstance(shape, (list, tuple)):
        normalized_shape = tuple(shape)
    else:
        raise TypeError("vector() expects a shape integer or a non-empty sequence of integers")
    if not normalized_shape:
        raise TypeError("vector() expects a non-empty shape")
    for dim in normalized_shape:
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise TypeError("vector() shape entries must be integers")
        if dim <= 0:
            raise TypeError("vector() shape entries must be positive")
    return VectorType(element_dtype=dtype, shape=normalized_shape)


def integer_bitwidth(dtype: ScalarType) -> int | None:
    if not isinstance(dtype, ScalarType):
        return None
    return _INTEGER_DTYPE_WIDTHS.get(dtype.name)


def integer_signedness(dtype: ScalarType) -> str | None:
    if not isinstance(dtype, ScalarType):
        return None
    return _INTEGER_DTYPE_SIGNS.get(dtype.name)


def is_integer_dtype(dtype: ScalarType) -> bool:
    return integer_bitwidth(dtype) is not None


def is_float_dtype(dtype: ScalarType) -> bool:
    return isinstance(dtype, ScalarType) and dtype.name in _FLOAT_DTYPE_WIDTHS


def bytewidth(dtype: ScalarType) -> int:
    if not isinstance(dtype, ScalarType):
        raise TypeError("bytewidth expects a TileLang scalar dtype")
    width = _DTYPE_BYTE_WIDTHS.get(dtype.name)
    if width is None:
        raise TypeError(f"dtype `{dtype.name}` is not supported by bytewidth")
    return width


def get_lanes(dtype: ScalarType) -> int:
    return 256 // bytewidth(dtype)


def elements_per_vreg(dtype: ScalarType) -> int:
    return get_lanes(dtype)


def constexpr(value: bool) -> bool:
    return value


def get_op_attr(name: str, default: Any = None) -> Any:
    if not isinstance(name, str) or not name:
        raise TypeError("get_op_attr expects a non-empty string attribute name")
    return default


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
    "VectorType",
    "MaskType",
    "ptr",
    "vreg",
    "vector",
    "MemorySpace",
    "Pipe",
    "Event",
    "PIPE",
    "EVENT",
    "MaskPattern",
    "PredicateDist",
    "VLoadDist",
    "VStoreDist",
    "PredicatePart",
    "CmpMode",
    "PAT",
    "BarrierType",
    "PadMode",
    "BLayout",
    "SLayout",
    "PadValue",
    "DeinterleaveDist",
    "InterleaveDist",
    "PositionMode",
    "OrderMode",
    "PostUpdateMode",
    "TileConfig",
    "TileSpecialization",
    "i1",
    "i8",
    "si8",
    "ui8",
    "i16",
    "si16",
    "ui16",
    "i32",
    "si32",
    "ui32",
    "i64",
    "si64",
    "ui64",
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
    "get_op_attr",
    "bytewidth",
    "get_lanes",
    "elements_per_vreg",
]
