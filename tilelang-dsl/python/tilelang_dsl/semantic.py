# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Semantic model for TileLang DSL descriptor lowering."""

from __future__ import annotations

import ast
import struct
from dataclasses import dataclass
from typing import Any

from .frontend_ast import (
    FrontendAssignStmt,
    FrontendAttributeExpr,
    FrontendBinaryExpr,
    FrontendCallExpr,
    FrontendConstantExpr,
    FrontendExprNode,
    FrontendExprStmt,
    FrontendForStmt,
    FrontendIfStmt,
    FrontendInlineProcNode,
    FrontendKernelNode,
    FrontendNameExpr,
    FrontendNameTarget,
    FrontendNoOpStmt,
    FrontendReturnStmt,
    FrontendSliceExpr,
    FrontendSourceLocation,
    FrontendStrictVecscopeStmt,
    FrontendStmtNode,
    FrontendSubscriptExpr,
    FrontendSymbolExpr,
    FrontendTargetNode,
    FrontendTupleExpr,
    FrontendTupleTarget,
    FrontendVecscopeStmt,
)
from .support_matrix import (
    DEFERRED_PTO_SURFACES,
    advanced_mode_message,
    deferred_surface_message,
    unsupported_feature_message,
)
from .types import (
    AlignType,
    BarrierType,
    BLayout,
    DeinterleaveDist,
    Event,
    InterleaveDist,
    MaskType,
    MaskPattern,
    MemorySpace,
    OrderMode,
    PadMode,
    PadValue,
    Pipe,
    PostUpdateMode,
    PositionMode,
    PointerType,
    ScalarType,
    SLayout,
    TileConfig,
    VcvtPartMode,
    VcvtRoundMode,
    VcvtSatMode,
    VRegType,
    bf16,
    bytewidth,
    f16,
    f32,
    i1,
    i8,
    i16,
    i32,
    i64,
    integer_bitwidth,
    integer_signedness,
    is_float_dtype,
    is_integer_dtype,
    si8,
    si16,
    si32,
    si64,
    align,
    ui8,
    ui16,
    ui32,
    ui64,
)


_DTYPE_SYMBOLS = {
    "i1": i1,
    "i8": i8,
    "si8": si8,
    "ui8": ui8,
    "i16": i16,
    "si16": si16,
    "ui16": ui16,
    "i32": i32,
    "si32": si32,
    "ui32": ui32,
    "i64": i64,
    "si64": si64,
    "ui64": ui64,
    "f16": f16,
    "bf16": bf16,
    "f32": f32,
}
_MASK_TYPE_SYMBOLS = {
    "mask_b8": MaskType("b8"),
    "mask_b16": MaskType("b16"),
    "mask_b32": MaskType("b32"),
}
_PATTERN_SYMBOLS = {pattern.name: pattern for pattern in MaskPattern}
_PIPE_SYMBOLS = {pipe.name: pipe for pipe in Pipe}
_EVENT_SYMBOLS = {event.name: event for event in Event}
_BARRIER_TYPE_SYMBOLS = {barrier_type.name: barrier_type for barrier_type in BarrierType}
_MEMORY_SPACE_SYMBOLS = {memory_space.name: memory_space for memory_space in MemorySpace}
_PAD_MODE_SYMBOLS = {pad_mode.name: pad_mode for pad_mode in PadMode}
_B_LAYOUT_SYMBOLS = {layout.name: layout for layout in BLayout}
_S_LAYOUT_SYMBOLS = {layout.name: layout for layout in SLayout}
_PAD_VALUE_SYMBOLS = {
    pad_value.name: pad_value
    for pad_value in (PadValue.NULL, PadValue.ZERO, PadValue.MAX, PadValue.MIN)
}
_DEINTERLEAVE_DIST_SYMBOLS = dict(DeinterleaveDist.__members__)
_INTERLEAVE_DIST_SYMBOLS = dict(InterleaveDist.__members__)
_POSITION_MODE_SYMBOLS = {position_mode.name: position_mode for position_mode in PositionMode}
_ORDER_MODE_SYMBOLS = {order_mode.name: order_mode for order_mode in OrderMode}
_VCVT_ROUND_MODE_SYMBOLS = {mode.name: mode for mode in VcvtRoundMode}
_VCVT_SAT_MODE_SYMBOLS = {mode.name: mode for mode in VcvtSatMode}
_VCVT_PART_MODE_SYMBOLS = {mode.name: mode for mode in VcvtPartMode}
_POST_UPDATE_MODE_SYMBOLS = {mode.name: mode for mode in PostUpdateMode}
_VCVT_ATTR_CONTRACTS: dict[tuple[str, str], tuple[bool, bool, bool]] = {
    # (src_kind, dst_kind): (requires_rnd, requires_sat, requires_part)
    ("f32", "f16"): (True, True, True),
    ("f32", "bf16"): (True, True, True),
    ("f32", "s16"): (True, True, True),
    ("f32", "s64"): (True, True, True),
    ("f32", "s32"): (True, True, False),
    ("f16", "f32"): (False, False, True),
    ("f16", "s32"): (False, False, True),
    ("f16", "s16"): (True, True, False),
    ("f16", "s8"): (True, True, True),
    ("f16", "u8"): (True, True, True),
    ("bf16", "f32"): (False, False, True),
    ("bf16", "s32"): (True, True, True),
    ("u8", "f16"): (False, False, True),
    ("u8", "u16"): (False, False, True),
    ("u8", "u32"): (False, False, True),
    ("s8", "f16"): (False, False, True),
    ("s8", "s16"): (False, False, True),
    ("s8", "s32"): (False, False, True),
    ("u16", "u8"): (False, True, True),
    ("u16", "u32"): (False, False, True),
    ("s16", "f16"): (True, False, False),
    ("s16", "f32"): (False, False, True),
    ("s16", "u32"): (False, False, True),
    ("s16", "s32"): (False, False, True),
    ("s16", "u8"): (False, True, True),
    ("u32", "u8"): (False, True, True),
    ("u32", "u16"): (False, True, True),
    ("u32", "s16"): (False, True, True),
    ("s32", "f32"): (True, False, False),
    ("s32", "u8"): (False, True, True),
    ("s32", "u16"): (False, True, True),
    ("s32", "s16"): (False, True, True),
    ("s32", "s64"): (False, False, True),
    ("s64", "f32"): (True, False, True),
    ("s64", "s32"): (False, True, True),
}


def _classify_vcvt_elem_kind(dtype: ScalarType) -> str | None:
    if dtype == f16:
        return "f16"
    if dtype == bf16:
        return "bf16"
    if dtype == f32:
        return "f32"
    if not is_integer_dtype(dtype):
        return None
    width = integer_bitwidth(dtype)
    sign = integer_signedness(dtype)
    is_unsigned = sign == "unsigned"
    if width == 8:
        return "u8" if is_unsigned else "s8"
    if width == 16:
        return "u16" if is_unsigned else "s16"
    if width == 32:
        return "u32" if is_unsigned else "s32"
    if width == 64:
        return None if is_unsigned else "s64"
    return None
_UNARY_VECTOR_OPS = {
    "vabs",
    "vrelu",
    "vexp",
    "vln",
    "vsqrt",
    "vrec",
    "vnot",
    "vcadd",
    "vcmax",
    "vbcnt",
    "vneg",
    "vcls",
    "vcmin",
    "vrsqrt",
    "vmov",
    "vsunpack",
    "vzunpack",
    "vusqz",
    "vsqz",
    "vexpdiff",
    "vtrc",
    "vcgadd",
    "vcgmax",
    "vcgmin",
    "vcpadd",
    "vsort32",
}
_BINARY_VECTOR_OPS = {
    "vadd",
    "vsub",
    "vmul",
    "vdiv",
    "vmax",
    "vmin",
    "vand",
    "vor",
    "vxor",
    "vaddrelu",
    "vaddreluconv",
    "vsubrelu",
    "vmulconv",
    "vshl",
    "vshr",
    "vprelu",
    "vpack",
    "vperm",
    "vmrgsort",
}
_VECTOR_SCALAR_OPS = {
    "vadds",
    "vsubs",
    "vmuls",
    "vdivs",
    "vmaxs",
    "vmins",
    "vlrelu",
    "vshls",
    "vshrs",
    "vands",
    "vors",
    "vxors",
}
_VECTOR_IMMEDIATE_OPS = {"vshift", "vslide"}
_TERNARY_VECTOR_OPS = {"vaxpy", "vmula"}
_MULTI_RESULT_VECTOR_OPS = {"vmull", "vldsx2", "vldus", "pstu"}
_BROADCAST_VECTOR_OPS = {"vbr", "vdup", "vci"}
_LOW_LEVEL_DMA_UNARY_CONFIG_OPS = {"set_mov_pad_val"}
_LOW_LEVEL_DMA_CONFIG_OPS = {
    "set_loop2_stride_outtoub",
    "set_loop1_stride_outtoub",
    "set_loop_size_outtoub",
    "set_loop2_stride_ubtoout",
    "set_loop1_stride_ubtoout",
    "set_loop_size_ubtoout",
}
_LOW_LEVEL_DMA_COPY_OPS = {
    "copy_gm_to_ubuf",
    "copy_ubuf_to_gm",
    "copy_ubuf_to_ubuf",
}
_MOV_PAD_SUPPORTED_SCALAR_DTYPES = frozenset(
    dtype.name for dtype in (i8, i16, i32, f16, bf16, f32)
)
_COMPARE_SELECT_OPS = {"vcmp", "vcmps", "vsel", "vselr", "vselrv2"}
_PREDICATE_MOVEMENT_OPS = {"pnot", "psel", "ppack", "punpack"}
_CARRY_OPS = {"vaddc", "vsubc", "vaddcs", "vsubcs"}
_REARRANGEMENT_OPS = {"vintlv", "vdintlv", "vintlvv2", "vdintlvv2"}
_UB_HELPER_OPS = {"vbitsort", "vmrgsort4"}
_ADVANCED_VECTOR_ACTIVITY_OPS = (
    _COMPARE_SELECT_OPS
    | _PREDICATE_MOVEMENT_OPS
    | _CARRY_OPS
    | _REARRANGEMENT_OPS
    | {"vcvt"}
)
_TENSORVIEW_RANK = 5


class SemanticType:
    """Base class for semantic value types."""


@dataclass(frozen=True)
class SemanticTensorViewType(SemanticType):
    element_dtype: ScalarType
    rank: int = _TENSORVIEW_RANK


@dataclass(frozen=True)
class SemanticPartitionTensorViewType(SemanticType):
    element_dtype: ScalarType
    rank: int = _TENSORVIEW_RANK


@dataclass(frozen=True)
class SemanticTensorSliceType(SemanticType):
    element_dtype: ScalarType
    rank: int
    extents: tuple[int | None, ...]
    physical_axes: tuple[int, ...]


@dataclass(frozen=True)
class SemanticTileType(SemanticType):
    element_dtype: ScalarType
    rank: int
    shape: tuple[int, ...] | None
    valid_shape: tuple[int | None, ...] | None
    memory_space: str | None
    config: TileConfig | None


@dataclass(frozen=True)
class SemanticTileConfigType(SemanticType):
    element_dtype: ScalarType | None = None


@dataclass(frozen=True)
class SemanticScalarType(SemanticType):
    dtype: ScalarType


@dataclass(frozen=True)
class SemanticPtrType(SemanticType):
    element_dtype: ScalarType
    memory_space: str


@dataclass(frozen=True)
class SemanticIndexType(SemanticType):
    pass


@dataclass(frozen=True)
class SemanticShapeType(SemanticType):
    rank: int


@dataclass(frozen=True)
class SemanticSliceType(SemanticType):
    pass


@dataclass(frozen=True)
class SemanticTupleType(SemanticType):
    elements: tuple[SemanticType, ...]


@dataclass(frozen=True)
class SemanticMetaType(SemanticType):
    kind: str


@dataclass(frozen=True)
class SemanticPadValueType(SemanticType):
    element_dtype: ScalarType | None = None


@dataclass(frozen=True)
class SemanticAlignType(SemanticType):
    pass


@dataclass(frozen=True)
class SemanticMaskType(SemanticType):
    granularity: str


@dataclass(frozen=True)
class SemanticVRegType(SemanticType):
    element_dtype: ScalarType
    lanes: int


_I32_TYPE = SemanticScalarType(dtype=i32)


@dataclass(frozen=True)
class SemanticBinding:
    name: str
    ssa_name: str
    type: SemanticType
    origin: str
    value: Any | None = None


@dataclass(frozen=True)
class SemanticTileBinding:
    name: str
    shape: tuple[int, ...]
    valid_shape: tuple[int | None, ...] | None
    memory_space: str
    config: Any


class SemanticExpr:
    """Base class for typed semantic expressions."""


@dataclass(frozen=True)
class SemanticBindingRef(SemanticExpr):
    binding: SemanticBinding
    type: SemanticType


@dataclass(frozen=True)
class SemanticLiteralExpr(SemanticExpr):
    value: Any
    type: SemanticType


@dataclass(frozen=True)
class SemanticSymbolExpr(SemanticExpr):
    namespace: str
    name: str
    value: Any
    type: SemanticType


@dataclass(frozen=True)
class SemanticSliceExpr(SemanticExpr):
    start: SemanticExpr | None
    stop: SemanticExpr | None
    step: SemanticExpr | None
    type: SemanticSliceType


@dataclass(frozen=True)
class SemanticTensorSliceAxis:
    start: SemanticExpr
    stop: SemanticExpr
    step: SemanticExpr
    extent: int | None


@dataclass(frozen=True)
class SemanticTupleExpr(SemanticExpr):
    elements: tuple[SemanticExpr, ...]
    type: SemanticTupleType


@dataclass(frozen=True)
class SemanticAttributeAccess(SemanticExpr):
    base: SemanticExpr
    attr: str
    type: SemanticType


@dataclass(frozen=True)
class SemanticSubscriptAccess(SemanticExpr):
    base: SemanticExpr
    index: SemanticExpr
    type: SemanticType


@dataclass(frozen=True)
class SemanticTensorSliceExpr(SemanticExpr):
    base: SemanticExpr
    slices: tuple[SemanticTensorSliceAxis, ...]
    type: SemanticTensorSliceType


@dataclass(frozen=True)
class SemanticBinaryExpr(SemanticExpr):
    lhs: SemanticExpr
    op: str
    rhs: SemanticExpr
    type: SemanticType


@dataclass(frozen=True)
class SemanticCallExpr(SemanticExpr):
    namespace: str | None
    name: str
    args: tuple[SemanticExpr, ...]
    type: SemanticType | None


class SemanticStmt:
    """Base class for semantic statements."""


@dataclass(frozen=True)
class SemanticAssignStmt(SemanticStmt):
    targets: tuple[SemanticBinding, ...]
    value: SemanticExpr
    annotation: Any | None = None


@dataclass(frozen=True)
class SemanticExprStmt(SemanticStmt):
    expr: SemanticExpr


@dataclass(frozen=True)
class SemanticDmaOptions:
    pad_mode: SemanticExpr | None = None
    pad_value: SemanticExpr | None = None
    left_padding: SemanticExpr | None = None
    right_padding: SemanticExpr | None = None
    init_out_buffer: SemanticExpr | None = None


@dataclass(frozen=True)
class SemanticDmaLoadStmt(SemanticStmt):
    src: SemanticTensorSliceExpr
    dst: SemanticExpr
    options: SemanticDmaOptions = SemanticDmaOptions()


@dataclass(frozen=True)
class SemanticDmaStoreStmt(SemanticStmt):
    src: SemanticExpr
    dst: SemanticTensorSliceExpr
    options: SemanticDmaOptions = SemanticDmaOptions()


@dataclass(frozen=True)
class SemanticVectorStoreStmt(SemanticStmt):
    value: SemanticExpr
    destination: SemanticExpr
    indices: tuple[SemanticExpr, ...]
    mask: SemanticExpr


@dataclass(frozen=True)
class SemanticVectorPairStoreStmt(SemanticStmt):
    low: SemanticExpr
    high: SemanticExpr
    destination: SemanticExpr
    indices: tuple[SemanticExpr, ...]
    dist: SemanticExpr
    mask: SemanticExpr


@dataclass(frozen=True)
class SemanticPredicateStoreStmt(SemanticStmt):
    value: SemanticExpr
    destination: SemanticExpr
    indices: tuple[SemanticExpr, ...]
    dist: SemanticExpr


@dataclass(frozen=True)
class SemanticAlignStoreStmt(SemanticStmt):
    op_name: str
    value: SemanticExpr
    destination: SemanticExpr
    indices: tuple[SemanticExpr, ...] = ()
    offset: SemanticExpr | None = None


@dataclass(frozen=True)
class SemanticScalarStoreStmt(SemanticStmt):
    value: SemanticExpr
    destination: SemanticExpr
    offset: SemanticExpr


@dataclass(frozen=True)
class SemanticVecscopeStmt(SemanticStmt):
    body: tuple[SemanticStmt, ...]


@dataclass(frozen=True)
class SemanticSetFlagStmt(SemanticStmt):
    src_pipe: str
    dst_pipe: str
    event: str


@dataclass(frozen=True)
class SemanticWaitFlagStmt(SemanticStmt):
    src_pipe: str
    dst_pipe: str
    event: str


@dataclass(frozen=True)
class SemanticPipeBarrierStmt(SemanticStmt):
    pipe: str


@dataclass(frozen=True)
class SemanticGetBufStmt(SemanticStmt):
    pipe: str
    buf_id: SemanticExpr
    mode: SemanticExpr


@dataclass(frozen=True)
class SemanticRlsBufStmt(SemanticStmt):
    pipe: str
    buf_id: SemanticExpr
    mode: SemanticExpr


@dataclass(frozen=True)
class SemanticMemBarStmt(SemanticStmt):
    barrier_type: str


@dataclass(frozen=True)
class SemanticSetCrossCoreStmt(SemanticStmt):
    core_id: SemanticExpr
    event_id: SemanticExpr


@dataclass(frozen=True)
class SemanticSetIntraBlockStmt(SemanticStmt):
    block_id: SemanticExpr
    event_id: SemanticExpr


@dataclass(frozen=True)
class SemanticSetIntraCoreStmt(SemanticStmt):
    config: SemanticExpr


@dataclass(frozen=True)
class SemanticWaitFlagDevStmt(SemanticStmt):
    core_id: SemanticExpr
    event_id: SemanticExpr


@dataclass(frozen=True)
class SemanticWaitIntraCoreStmt(SemanticStmt):
    block_id: SemanticExpr
    event_id: SemanticExpr


@dataclass(frozen=True)
class SemanticDmaConfigStmt(SemanticStmt):
    name: str
    first: SemanticExpr
    second: SemanticExpr


@dataclass(frozen=True)
class SemanticDmaUnaryConfigStmt(SemanticStmt):
    name: str
    value: SemanticExpr


@dataclass(frozen=True)
class SemanticLowLevelCopyStmt(SemanticStmt):
    name: str
    source: SemanticExpr
    destination: SemanticExpr
    operands: tuple[SemanticExpr, ...]


@dataclass(frozen=True)
class SemanticIfResult:
    result_binding: SemanticBinding
    then_binding: SemanticBinding
    else_binding: SemanticBinding


@dataclass(frozen=True)
class SemanticIfStmt(SemanticStmt):
    condition: SemanticExpr
    then_body: tuple[SemanticStmt, ...]
    else_body: tuple[SemanticStmt, ...]
    results: tuple[SemanticIfResult, ...]


@dataclass(frozen=True)
class SemanticReturnStmt(SemanticStmt):
    value: SemanticExpr | None


@dataclass(frozen=True)
class SemanticForStmt(SemanticStmt):
    induction_variable: SemanticBinding
    lower_bound: SemanticExpr
    upper_bound: SemanticExpr
    step: SemanticExpr
    body: tuple[SemanticStmt, ...]
    loop_carried: tuple[SemanticBinding, ...]


@dataclass(frozen=True)
class SemanticStrictVecscopeStmt(SemanticStmt):
    captures: tuple[SemanticExpr, ...]
    block_arguments: tuple[SemanticBinding, ...]
    body: tuple[SemanticStmt, ...]


@dataclass(frozen=True)
class SemanticParameter:
    binding: SemanticBinding

    @property
    def name(self) -> str:
        return self.binding.name

    @property
    def kind(self) -> str:
        return self.binding.origin

    @property
    def type(self) -> SemanticType:
        return self.binding.type

    @property
    def ssa_name(self) -> str:
        return self.binding.ssa_name


@dataclass(frozen=True)
class SemanticKernel:
    target: str
    op: str
    symbol_name: str
    verify_enabled: bool
    advanced_enabled: bool
    dtype_signature: tuple[Any, ...]
    parameters: tuple[SemanticParameter, ...]
    tile_bindings: tuple[SemanticTileBinding, ...]
    body: tuple[SemanticStmt, ...]
    inline_helpers: tuple["SemanticKernel", ...] = ()


class _SemanticAnalyzer:
    def __init__(self, node: FrontendKernelNode):
        self.node = node
        self._counter = 0
        self._disable_inference_depth = 0
        self._has_explicit_vecscope = self._contains_explicit_vecscope(node.body)
        self._tile_specializations = {
            spec.name: spec for spec in node.tile_specializations
        }
        self._hidden_parameters: list[SemanticParameter] = []
        self._inline_proc_nodes: dict[str, FrontendInlineProcNode] = {
            inline_proc.name: inline_proc for inline_proc in node.inline_procs
        }
        self._inline_proc_specializations: dict[tuple[str, tuple[SemanticType, ...]], SemanticKernel] = {}
        self._inline_proc_return_types: dict[tuple[str, tuple[SemanticType, ...]], SemanticType | None] = {}
        self._inline_proc_order: list[tuple[str, tuple[SemanticType, ...]]] = []
        self._inline_proc_active_stack: list[tuple[str, tuple[SemanticType, ...]]] = []

    def _expr_source_location(
        self,
        expr: FrontendExprNode | SemanticExpr,
    ) -> FrontendSourceLocation | None:
        return getattr(expr, "source_location", None)

    def _attach_expr_source_location(
        self,
        semantic_expr: SemanticExpr,
        frontend_expr: FrontendExprNode,
    ) -> SemanticExpr:
        source_location = self._expr_source_location(frontend_expr)
        if source_location is not None:
            object.__setattr__(semantic_expr, "source_location", source_location)
        return semantic_expr

    def _format_source_message(
        self,
        message: str,
        expr: FrontendExprNode | SemanticExpr | None = None,
    ) -> str:
        if expr is None:
            return message
        source_location = self._expr_source_location(expr)
        if source_location is None:
            return message
        return (
            f"{source_location.path}:{source_location.line}:{source_location.column}: "
            f"{message}"
        )

    def _raise_expr_type_error(
        self,
        message: str,
        expr: FrontendExprNode | SemanticExpr | None = None,
    ) -> None:
        raise TypeError(self._format_source_message(message, expr))

    def analyze(self) -> SemanticKernel:
        env: dict[str, SemanticBinding] = {}
        parameters = []
        for index, param in enumerate(self.node.parameters):
            binding = SemanticBinding(
                name=param.name,
                ssa_name=f"%arg{index}",
                type=self._parameter_type(param),
                origin=param.kind,
            )
            env[param.name] = binding
            parameters.append(SemanticParameter(binding=binding))
        body, _ = self._analyze_kernel_body(env)
        parameters.extend(self._hidden_parameters)
        tile_bindings = tuple(
            SemanticTileBinding(
                name=spec.name,
                shape=spec.shape,
                valid_shape=spec.valid_shape,
                memory_space=spec.memory_space,
                config=spec.config,
            )
            for spec in self.node.tile_specializations
        )
        return SemanticKernel(
            target=self.node.target,
            op=self.node.op,
            symbol_name=self.node.name,
            verify_enabled=self.node.verify_enabled,
            advanced_enabled=self.node.advanced_enabled,
            dtype_signature=self.node.dtype_signature,
            parameters=tuple(parameters),
            tile_bindings=tile_bindings,
            body=body,
            inline_helpers=tuple(
                self._inline_proc_specializations[key]
                for key in self._inline_proc_order
            ),
        )

    def _analyze_kernel_body(
        self,
        env: dict[str, SemanticBinding],
    ) -> tuple[tuple[SemanticStmt, ...], dict[str, SemanticBinding]]:
        return self._analyze_block(self.node.body, env, allow_outer_lookup=True)

    def _parameter_type(self, param: Any) -> SemanticType:
        if param.kind == "tensorview":
            return SemanticTensorViewType(
                element_dtype=param.dtype,
                rank=_TENSORVIEW_RANK,
            )
        if param.kind == "partition_tensor_view":
            return SemanticPartitionTensorViewType(
                element_dtype=param.dtype,
                rank=_TENSORVIEW_RANK,
            )
        if param.kind == "tile":
            spec = self._tile_specializations.get(param.name)
            rank = 2 if spec is None else len(spec.shape)
            shape = None if spec is None else spec.shape
            valid_shape = None if spec is None else (
                spec.shape if spec.valid_shape is None else spec.valid_shape
            )
            memory_space = None if spec is None else spec.memory_space
            return SemanticTileType(
                element_dtype=param.dtype,
                rank=rank,
                shape=shape,
                valid_shape=valid_shape,
                memory_space=memory_space,
                config=None if spec is None else (spec.config or TileConfig()),
            )
        if param.kind == "ptr":
            memory_space = param.annotation.memory_space.value
            return SemanticPtrType(
                element_dtype=param.dtype,
                memory_space=memory_space,
            )
        if param.kind == "mask":
            return SemanticMaskType(granularity=param.dtype.granularity)
        if param.kind == "scalar":
            return SemanticScalarType(dtype=param.dtype)
        raise ValueError(f"unsupported parameter kind {param.kind!r}")

    def _new_ssa_name(self, stem: str) -> str:
        name = f"%{stem}_{self._counter}"
        self._counter += 1
        return name

    def _tensor_shape_binding_name(self, tensor_name: str, axis: int) -> str:
        return f"__shape_{tensor_name}_{axis}"

    def _tensor_stride_binding_name(self, tensor_name: str, axis: int) -> str:
        return f"__stride_{tensor_name}_{axis}"

    def _tile_valid_shape_binding_name(self, tile_name: str, axis: int) -> str:
        return f"__valid_shape_{tile_name}_{axis}"

    def _ensure_hidden_parameter(
        self,
        hidden_name: str,
        origin: str,
    ) -> SemanticBinding:
        for parameter in self._hidden_parameters:
            if parameter.name == hidden_name:
                return parameter.binding
        binding = SemanticBinding(
            name=hidden_name,
            ssa_name=f"%arg{len(self.node.parameters) + len(self._hidden_parameters)}",
            type=SemanticIndexType(),
            origin=origin,
        )
        self._hidden_parameters.append(SemanticParameter(binding=binding))
        return binding

    def _ensure_tensor_shape_parameter(
        self,
        tensor_binding: SemanticBinding,
        axis: int,
    ) -> SemanticBinding:
        hidden_name = self._tensor_shape_binding_name(tensor_binding.name, axis)
        return self._ensure_hidden_parameter(hidden_name, "tensorview_shape")

    def _ensure_tensor_stride_parameter(
        self,
        tensor_binding: SemanticBinding,
        axis: int,
    ) -> SemanticBinding:
        hidden_name = self._tensor_stride_binding_name(tensor_binding.name, axis)
        return self._ensure_hidden_parameter(hidden_name, "tensorview_stride")

    def _ensure_tile_valid_shape_parameter(
        self,
        tile_binding: SemanticBinding,
        axis: int,
    ) -> SemanticBinding:
        hidden_name = self._tile_valid_shape_binding_name(tile_binding.name, axis)
        return self._ensure_hidden_parameter(hidden_name, "tile_valid_shape")

    def _make_binding(
        self,
        name: str,
        ty: SemanticType,
        origin: str,
        *,
        value: Any | None = None,
    ) -> SemanticBinding:
        stem = name if name.isidentifier() else "v"
        return SemanticBinding(
            name=name,
            ssa_name=self._new_ssa_name(stem),
            type=ty,
            origin=origin,
            value=value,
        )

    def _analyze_block(
        self,
        statements: tuple[FrontendStmtNode, ...],
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
        allow_inferred_vecscope: bool = True,
    ) -> tuple[tuple[SemanticStmt, ...], dict[str, SemanticBinding]]:
        current_env = dict(env)
        semantic_statements = []
        index = 0
        while index < len(statements):
            if self._should_infer_vecscope(
                statements[index],
                allow_inferred_vecscope=allow_inferred_vecscope,
            ):
                end = index + 1
                while end < len(statements) and self._should_infer_vecscope(
                    statements[end],
                    allow_inferred_vecscope=allow_inferred_vecscope,
                ):
                    end += 1
                run = statements[index:end]
                if self._run_contains_vector_op(run):
                    vecscope_stmt, current_env = self._analyze_inferred_vecscope(
                        run,
                        current_env,
                        allow_outer_lookup=allow_outer_lookup,
                    )
                    semantic_statements.append(
                        vecscope_stmt
                    )
                else:
                    for stmt in run:
                        emitted_stmts, current_env = self._analyze_stmt_or_inline(
                            stmt,
                            current_env,
                            allow_outer_lookup=allow_outer_lookup,
                        )
                        semantic_statements.extend(emitted_stmts)
                index = end
                continue

            emitted_stmts, current_env = self._analyze_stmt_or_inline(
                statements[index],
                current_env,
                allow_outer_lookup=allow_outer_lookup,
            )
            semantic_statements.extend(emitted_stmts)
            index += 1
        return tuple(semantic_statements), current_env

    def _analyze_stmt_or_inline(
        self,
        stmt: FrontendStmtNode,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> tuple[tuple[SemanticStmt, ...], dict[str, SemanticBinding]]:
        if isinstance(stmt, FrontendNoOpStmt):
            # Python `pass` lowers to a frontend no-op and does not materialize semantic IR.
            return tuple(), dict(env)
        if (
            isinstance(stmt, FrontendExprStmt)
            and isinstance(stmt.expr, FrontendConstantExpr)
            and isinstance(stmt.expr.value, str)
        ):
            # Treat Python docstring-style string expression statements as no-op.
            return tuple(), dict(env)
        if isinstance(stmt, FrontendIfStmt) and stmt.is_constexpr:
            return self._analyze_constexpr_if(
                stmt,
                env,
                allow_outer_lookup=allow_outer_lookup,
            )
        semantic_stmt, updated_env = self._analyze_stmt(
            stmt,
            env,
            allow_outer_lookup=allow_outer_lookup,
        )
        return (semantic_stmt,), updated_env

    def _wrap_kernel_body_in_inferred_vecscope(
        self,
        statements: tuple[SemanticStmt, ...],
    ) -> tuple[SemanticStmt, ...]:
        if not statements or not self._semantic_block_contains_vector_activity(statements):
            return statements

        body_end = len(statements)
        while body_end > 0 and isinstance(statements[body_end - 1], SemanticReturnStmt):
            body_end -= 1
        if body_end == 0:
            return statements

        wrapped_body = SemanticVecscopeStmt(body=statements[:body_end])
        return (wrapped_body, *statements[body_end:])

    def _should_infer_vecscope(
        self,
        stmt: FrontendStmtNode,
        *,
        allow_inferred_vecscope: bool,
    ) -> bool:
        if self._has_explicit_vecscope:
            return False
        if self._disable_inference_depth > 0:
            return False
        if not allow_inferred_vecscope:
            return False
        if isinstance(stmt, FrontendForStmt):
            return self._block_can_live_in_inferred_vecscope(stmt.body)
        name = self._frontend_vector_call_name(stmt)
        return name in (
            {
                "make_mask",
                "init_align",
                "vlds",
                "vldas",
                "vldus",
                "psts",
                "pstu",
                "vsst",
                "vsta",
                "vstas",
                "vstar",
                "vsts",
                "vstsx2",
                "vstus",
                "vstur",
            }
            | _UNARY_VECTOR_OPS
            | _BINARY_VECTOR_OPS
            | _VECTOR_SCALAR_OPS
            | _VECTOR_IMMEDIATE_OPS
            | _TERNARY_VECTOR_OPS
            | _MULTI_RESULT_VECTOR_OPS
            | _BROADCAST_VECTOR_OPS
            | _ADVANCED_VECTOR_ACTIVITY_OPS
        )

    def _block_can_live_in_inferred_vecscope(
        self,
        statements: tuple[FrontendStmtNode, ...],
    ) -> bool:
        saw_vector_activity = False
        for stmt in statements:
            if self._frontend_stmt_is_vecscope_boundary(stmt):
                return False
            if self._frontend_stmt_can_live_in_inferred_vecscope(stmt):
                saw_vector_activity = True
                continue
            if self._frontend_stmt_is_scalar_vecscope_stmt(stmt):
                continue
            return False
        return saw_vector_activity

    def _frontend_stmt_is_vecscope_boundary(self, stmt: FrontendStmtNode) -> bool:
        if isinstance(stmt, FrontendStrictVecscopeStmt):
            return True
        if isinstance(stmt, FrontendVecscopeStmt):
            return True
        if isinstance(stmt, FrontendIfStmt):
            return not stmt.is_constexpr
        return (
            isinstance(stmt, FrontendExprStmt)
            and (
                self._is_dma_call(stmt.expr)
                or self._is_sync_call(stmt.expr)
                or self._is_ub_helper_call(stmt.expr)
            )
        )

    def _constexpr_if_contains_vector_activity(self, stmt: FrontendIfStmt) -> bool:
        if not stmt.is_constexpr:
            return False
        return self._run_contains_vector_op(stmt.then_body) or self._run_contains_vector_op(stmt.else_body)

    def _frontend_stmt_can_live_in_inferred_vecscope(
        self,
        stmt: FrontendStmtNode,
    ) -> bool:
        if isinstance(stmt, FrontendForStmt):
            return self._block_can_live_in_inferred_vecscope(stmt.body)
        if isinstance(stmt, FrontendIfStmt):
            return self._constexpr_if_contains_vector_activity(stmt)
        return self._frontend_stmt_contains_vector_activity(stmt)

    def _frontend_stmt_is_scalar_vecscope_stmt(
        self,
        stmt: FrontendStmtNode,
    ) -> bool:
        return isinstance(stmt, FrontendNoOpStmt) or isinstance(stmt, FrontendAssignStmt) or (
            isinstance(stmt, FrontendExprStmt)
            and isinstance(stmt.expr, FrontendCallExpr)
            and stmt.expr.namespace == "pto"
            and stmt.expr.name == "store_scalar"
        ) or (
            isinstance(stmt, FrontendIfStmt) and stmt.is_constexpr
        )

    def _frontend_stmt_contains_vector_activity(self, stmt: FrontendStmtNode) -> bool:
        expr: FrontendExprNode | None = None
        if isinstance(stmt, FrontendAssignStmt):
            expr = stmt.value
        elif isinstance(stmt, FrontendExprStmt):
            expr = stmt.expr
        if not isinstance(expr, FrontendCallExpr):
            return False
        return (
            expr.namespace == "pto"
            and expr.name in (
                {
                    "make_mask",
                    "init_align",
                    "vlds",
                    "vldas",
                    "vldus",
                    "psts",
                    "pstu",
                    "vsst",
                    "vsta",
                    "vstas",
                    "vstar",
                    "vsts",
                    "vstsx2",
                    "vstus",
                    "vstur",
                }
                | _UNARY_VECTOR_OPS
                | _BINARY_VECTOR_OPS
                | _VECTOR_SCALAR_OPS
                | _VECTOR_IMMEDIATE_OPS
                | _TERNARY_VECTOR_OPS
                | _MULTI_RESULT_VECTOR_OPS
                | _BROADCAST_VECTOR_OPS
                | _ADVANCED_VECTOR_ACTIVITY_OPS
            )
        )

    def _run_contains_vector_op(self, statements: tuple[FrontendStmtNode, ...]) -> bool:
        for stmt in statements:
            if isinstance(stmt, FrontendForStmt) and self._block_can_live_in_inferred_vecscope(stmt.body):
                return True
            if isinstance(stmt, FrontendVecscopeStmt):
                if self._run_contains_vector_op(stmt.body):
                    return True
                continue
            if isinstance(stmt, FrontendIfStmt):
                if self._constexpr_if_contains_vector_activity(stmt):
                    return True
                continue
            name = self._frontend_vector_call_name(stmt)
            if name is None or name == "make_mask":
                continue
            return True
        return False

    def _frontend_vector_call_name(self, stmt: FrontendStmtNode) -> str | None:
        expr: FrontendExprNode | None = None
        if isinstance(stmt, FrontendAssignStmt):
            expr = stmt.value
        elif isinstance(stmt, FrontendExprStmt):
            expr = stmt.expr
        if (
            isinstance(expr, FrontendCallExpr)
            and expr.namespace == "pto"
        ):
            return expr.name
        return None

    def _analyze_inferred_vecscope(
        self,
        statements: tuple[FrontendStmtNode, ...],
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> tuple[SemanticVecscopeStmt, dict[str, SemanticBinding]]:
        self._disable_inference_depth += 1
        try:
            body, updated_env = self._analyze_block_without_inference(
                statements,
                env,
                allow_outer_lookup=allow_outer_lookup,
            )
        finally:
            self._disable_inference_depth -= 1
        return SemanticVecscopeStmt(body=body), updated_env

    def _analyze_block_without_inference(
        self,
        statements: tuple[FrontendStmtNode, ...],
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> tuple[tuple[SemanticStmt, ...], dict[str, SemanticBinding]]:
        current_env = dict(env)
        semantic_statements = []
        for stmt in statements:
            emitted_stmts, current_env = self._analyze_stmt_or_inline(
                stmt,
                current_env,
                allow_outer_lookup=allow_outer_lookup,
            )
            semantic_statements.extend(emitted_stmts)
        return tuple(semantic_statements), current_env

    def _semantic_block_contains_vector_activity(
        self,
        statements: tuple[SemanticStmt, ...],
    ) -> bool:
        for stmt in statements:
            if isinstance(stmt, SemanticVecscopeStmt):
                return True
            if isinstance(stmt, SemanticStrictVecscopeStmt):
                return True
            if isinstance(stmt, SemanticVectorStoreStmt):
                return True
            if isinstance(stmt, SemanticAssignStmt) and self._expr_contains_vector_activity(stmt.value):
                return True
            if isinstance(stmt, SemanticExprStmt) and self._expr_contains_vector_activity(stmt.expr):
                return True
            if isinstance(stmt, SemanticForStmt) and self._semantic_block_contains_vector_activity(stmt.body):
                return True
            if isinstance(stmt, SemanticIfStmt) and (
                self._semantic_block_contains_vector_activity(stmt.then_body)
                or self._semantic_block_contains_vector_activity(stmt.else_body)
            ):
                return True
        return False

    def _expr_contains_vector_activity(self, expr: SemanticExpr) -> bool:
        if isinstance(expr, SemanticCallExpr):
            if expr.namespace == "pto" and expr.name in (
                {"make_mask", "vlds"}
                | _UNARY_VECTOR_OPS
                | _BINARY_VECTOR_OPS
                | _VECTOR_SCALAR_OPS
                | _VECTOR_IMMEDIATE_OPS
                | _TERNARY_VECTOR_OPS
                | _MULTI_RESULT_VECTOR_OPS
                | _BROADCAST_VECTOR_OPS
                | _ADVANCED_VECTOR_ACTIVITY_OPS
            ):
                return True
            return any(self._expr_contains_vector_activity(arg) for arg in expr.args)
        if isinstance(expr, SemanticBinaryExpr):
            return self._expr_contains_vector_activity(expr.lhs) or self._expr_contains_vector_activity(expr.rhs)
        if isinstance(expr, SemanticTupleExpr):
            return any(self._expr_contains_vector_activity(element) for element in expr.elements)
        if isinstance(expr, SemanticAttributeAccess):
            return self._expr_contains_vector_activity(expr.base)
        if isinstance(expr, SemanticSubscriptAccess):
            return self._expr_contains_vector_activity(expr.base) or self._expr_contains_vector_activity(expr.index)
        if isinstance(expr, SemanticTensorSliceExpr):
            return self._expr_contains_vector_activity(expr.base)
        return False

    def _analyze_stmt(
        self,
        stmt: FrontendStmtNode,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> tuple[SemanticStmt, dict[str, SemanticBinding]]:
        if isinstance(stmt, FrontendAssignStmt):
            value = self._analyze_expr(stmt.value, env, allow_outer_lookup=allow_outer_lookup)
            updated_env = dict(env)
            targets = self._bind_assignment_target(
                stmt.target,
                value,
                updated_env,
                stmt.annotation,
            )
            return (
                SemanticAssignStmt(targets=targets, value=value, annotation=stmt.annotation),
                updated_env,
            )
        if isinstance(stmt, FrontendExprStmt):
            if self._is_dma_call(stmt.expr):
                return self._analyze_dma_stmt(stmt.expr, env, allow_outer_lookup=allow_outer_lookup)
            if self._is_sync_call(stmt.expr):
                return self._analyze_sync_stmt(stmt.expr, env, allow_outer_lookup=allow_outer_lookup)
            if self._is_low_level_dma_call(stmt.expr):
                return self._analyze_low_level_dma_stmt(
                    stmt.expr,
                    env,
                    allow_outer_lookup=allow_outer_lookup,
                )
            if self._is_vector_store_call(stmt.expr):
                return self._analyze_vector_store_stmt(stmt.expr, env, allow_outer_lookup=allow_outer_lookup)
            if self._is_scalar_store_call(stmt.expr):
                return self._analyze_scalar_store_stmt(stmt.expr, env, allow_outer_lookup=allow_outer_lookup)
            expr = self._analyze_expr(stmt.expr, env, allow_outer_lookup=allow_outer_lookup)
            return SemanticExprStmt(expr=expr), dict(env)
        if isinstance(stmt, FrontendReturnStmt):
            value = None
            if stmt.value is not None:
                value = self._analyze_expr(stmt.value, env, allow_outer_lookup=allow_outer_lookup)
            return SemanticReturnStmt(value=value), dict(env)
        if isinstance(stmt, FrontendForStmt):
            return self._analyze_for(stmt, env, allow_outer_lookup=allow_outer_lookup)
        if isinstance(stmt, FrontendIfStmt):
            return self._analyze_if(stmt, env, allow_outer_lookup=allow_outer_lookup)
        if isinstance(stmt, FrontendVecscopeStmt):
            return self._analyze_explicit_vecscope(stmt, env, allow_outer_lookup=allow_outer_lookup)
        if isinstance(stmt, FrontendStrictVecscopeStmt):
            return self._analyze_strict_vecscope(stmt, env)
        raise ValueError(f"unsupported frontend statement {type(stmt).__name__}")

    def _inline_proc_specialization_key(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> tuple[str, tuple[SemanticType, ...]]:
        return (name, tuple(arg.type for arg in args))

    def _inline_proc_symbol_name(
        self,
        name: str,
        index: int,
    ) -> str:
        sanitized = "".join(char if char.isalnum() else "_" for char in name)
        return f"__tl_inline_{sanitized}_{index}"

    def _collect_inline_helper_tile_bindings(
        self,
        parameters: tuple[SemanticParameter, ...],
    ) -> tuple[SemanticTileBinding, ...]:
        tile_bindings: list[SemanticTileBinding] = []
        for parameter in parameters:
            if not isinstance(parameter.type, SemanticTileType):
                continue
            if parameter.type.shape is None:
                continue
            tile_bindings.append(
                SemanticTileBinding(
                    name=parameter.name,
                    shape=parameter.type.shape,
                    valid_shape=parameter.type.valid_shape,
                    memory_space=parameter.type.memory_space or "ub",
                    config=parameter.type.config or TileConfig(),
                )
            )
        return tuple(tile_bindings)

    def _materialize_inline_proc_specialization(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticKernel:
        inline_proc_node = self._inline_proc_nodes.get(name)
        if inline_proc_node is None:
            raise TypeError(f"inline_proc `{name}` is not registered in the current TileLang module")

        key = self._inline_proc_specialization_key(name, args)
        existing = self._inline_proc_specializations.get(key)
        if existing is not None:
            return existing
        if key in self._inline_proc_active_stack:
            raise TypeError(
                f"recursive inline_proc call `{name}` is not supported in TileLang DSL v1"
            )

        if len(inline_proc_node.parameters) != len(args):
            raise TypeError(
                f"inline_proc `{name}` expects {len(inline_proc_node.parameters)} arguments in TileLang DSL v1"
            )

        helper_env: dict[str, SemanticBinding] = {}
        helper_parameters: list[SemanticParameter] = []
        for index, (param, arg_expr) in enumerate(zip(inline_proc_node.parameters, args)):
            binding = SemanticBinding(
                name=param.name,
                ssa_name=f"%arg{index}",
                type=arg_expr.type,
                origin="inline_param",
            )
            helper_env[param.name] = binding
            helper_parameters.append(SemanticParameter(binding=binding))

        saved_hidden_parameters = self._hidden_parameters
        self._hidden_parameters = []
        self._inline_proc_active_stack.append(key)
        try:
            body, _ = self._analyze_block(
                inline_proc_node.body,
                helper_env,
                allow_outer_lookup=False,
                allow_inferred_vecscope=True,
            )
        finally:
            self._inline_proc_active_stack.pop()
        helper_hidden_parameters = tuple(self._hidden_parameters)
        self._hidden_parameters = saved_hidden_parameters

        if helper_hidden_parameters:
            raise TypeError(
                f"inline_proc `{name}` currently does not support dynamic shape metadata captures in TileLang DSL v1"
            )

        return_type: SemanticType | None = None
        if body and isinstance(body[-1], SemanticReturnStmt):
            return_type = None if body[-1].value is None else body[-1].value.type

        helper_index = len(self._inline_proc_order)
        helper_kernel = SemanticKernel(
            target=self.node.target,
            op=self.node.op,
            symbol_name=self._inline_proc_symbol_name(name, helper_index),
            verify_enabled=False,
            advanced_enabled=self.node.advanced_enabled,
            dtype_signature=self.node.dtype_signature,
            parameters=tuple(helper_parameters),
            tile_bindings=self._collect_inline_helper_tile_bindings(tuple(helper_parameters)),
            body=body,
            inline_helpers=(),
        )
        self._inline_proc_specializations[key] = helper_kernel
        self._inline_proc_return_types[key] = return_type
        self._inline_proc_order.append(key)
        return helper_kernel

    def _analyze_inline_proc_call_expr(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        helper_kernel = self._materialize_inline_proc_specialization(name, args)
        key = self._inline_proc_specialization_key(name, args)
        return SemanticCallExpr(
            namespace=None,
            name=helper_kernel.symbol_name,
            args=args,
            type=self._inline_proc_return_types.get(key),
        )

    def _contains_explicit_vecscope(self, statements: tuple[FrontendStmtNode, ...]) -> bool:
        for stmt in statements:
            if isinstance(stmt, FrontendVecscopeStmt):
                return True
            if isinstance(stmt, FrontendForStmt):
                if self._contains_explicit_vecscope(stmt.body):
                    return True
                continue
            if isinstance(stmt, FrontendIfStmt):
                if self._contains_explicit_vecscope(stmt.then_body):
                    return True
                if self._contains_explicit_vecscope(stmt.else_body):
                    return True
                continue
            if isinstance(stmt, FrontendStrictVecscopeStmt):
                if self._contains_explicit_vecscope(stmt.body):
                    return True
        return False

    def _analyze_explicit_vecscope(
        self,
        stmt: FrontendVecscopeStmt,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> tuple[SemanticStmt, dict[str, SemanticBinding]]:
        body, updated_env = self._analyze_block(
            stmt.body,
            dict(env),
            allow_outer_lookup=allow_outer_lookup,
        )
        return SemanticVecscopeStmt(body=body), updated_env

    def _is_dma_call(self, expr: FrontendExprNode) -> bool:
        return (
            isinstance(expr, FrontendCallExpr)
            and expr.namespace == "pto"
            and expr.name in {"dma_load", "dma_store"}
        )

    def _is_vector_store_call(self, expr: FrontendExprNode) -> bool:
        return (
            isinstance(expr, FrontendCallExpr)
            and expr.namespace == "pto"
            and expr.name in {"psts", "vsst", "vsta", "vstas", "vstar", "vsts", "vstsx2"}
        )

    def _is_scalar_store_call(self, expr: FrontendExprNode) -> bool:
        return (
            isinstance(expr, FrontendCallExpr)
            and expr.namespace == "pto"
            and expr.name == "store_scalar"
        )

    def _is_sync_call(self, expr: FrontendExprNode) -> bool:
        return (
            isinstance(expr, FrontendCallExpr)
            and expr.namespace == "pto"
            and expr.name in {
                "set_flag",
                "wait_flag",
                "pipe_barrier",
                "barrier",
                "get_buf",
                "rls_buf",
                "mem_bar",
                "set_cross_core",
                "set_intra_block",
                "set_intra_core",
                "wait_flag_dev",
                "wait_intra_core",
            }
        )

    def _is_ub_helper_call(self, expr: FrontendExprNode) -> bool:
        return (
            isinstance(expr, FrontendCallExpr)
            and expr.namespace == "pto"
            and expr.name in _UB_HELPER_OPS
        )

    def _is_low_level_dma_call(self, expr: FrontendExprNode) -> bool:
        return (
            isinstance(expr, FrontendCallExpr)
            and expr.namespace == "pto"
            and expr.name in _LOW_LEVEL_DMA_UNARY_CONFIG_OPS | _LOW_LEVEL_DMA_CONFIG_OPS | _LOW_LEVEL_DMA_COPY_OPS
        )

    def _analyze_dma_stmt(
        self,
        expr: FrontendCallExpr,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> tuple[SemanticStmt, dict[str, SemanticBinding]]:
        args = tuple(
            self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
            for arg in expr.args
        )
        if expr.name == "dma_load":
            if len(args) != 2:
                raise TypeError("pto.dma_load expects exactly 2 positional arguments in TileLang DSL v1")
            src = self._require_tensor_slice(args[0], "pto.dma_load source")
            dst = self._require_tile_expr(args[1], "pto.dma_load destination")
            options = self._analyze_dma_options(
                expr.keywords,
                env,
                allow_outer_lookup=allow_outer_lookup,
                context="pto.dma_load",
            )
            self._validate_dma_load_profile(src, dst, options)
            return SemanticDmaLoadStmt(src=src, dst=dst, options=options), dict(env)
        if expr.name == "dma_store":
            if len(args) != 2:
                raise TypeError("pto.dma_store expects exactly 2 positional arguments in TileLang DSL v1")
            src = self._require_tile_expr(args[0], "pto.dma_store source")
            dst = self._require_tensor_slice(args[1], "pto.dma_store destination")
            options = self._analyze_dma_options(
                expr.keywords,
                env,
                allow_outer_lookup=allow_outer_lookup,
                context="pto.dma_store",
            )
            self._validate_dma_store_profile(src, dst, options)
            return SemanticDmaStoreStmt(src=src, dst=dst, options=options), dict(env)
        raise ValueError(f"unsupported DMA stmt pto.{expr.name}")

    def _analyze_dma_options(
        self,
        keywords: tuple[tuple[str, FrontendExprNode], ...],
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
        context: str,
    ) -> SemanticDmaOptions:
        analyzed: dict[str, SemanticExpr] = {}
        for name, keyword_expr in keywords:
            analyzed[name] = self._analyze_expr(
                keyword_expr,
                env,
                allow_outer_lookup=allow_outer_lookup,
            )

        pad_mode = analyzed.get("pad_mode")
        if pad_mode is not None:
            self._pad_mode_value(pad_mode, default=PadMode.PadNull)

        left_padding = analyzed.get("left_padding")
        if left_padding is not None:
            self._require_index_typed_expr(left_padding)

        right_padding = analyzed.get("right_padding")
        if right_padding is not None:
            self._require_index_typed_expr(right_padding)

        init_out_buffer = analyzed.get("init_out_buffer")
        if init_out_buffer is not None:
            self._require_i1_expr(init_out_buffer, f"{context} init_out_buffer")

        return SemanticDmaOptions(
            pad_mode=pad_mode,
            pad_value=analyzed.get("pad_value"),
            left_padding=left_padding,
            right_padding=right_padding,
            init_out_buffer=init_out_buffer,
        )

    def _analyze_vector_store_stmt(
        self,
        expr: FrontendCallExpr,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> tuple[SemanticStmt, dict[str, SemanticBinding]]:
        if expr.name == "psts":
            if len(expr.args) in {2, 3} and isinstance(expr.args[1], FrontendSubscriptExpr):
                raise TypeError(
                    "pto.psts does not support Tile element-indexing syntax in TileLang DSL v1; "
                    "use explicit pointer form `pto.psts(mask, buf, offset[, dist])`"
                )

            args = tuple(
                self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
                for arg in expr.args
            )
            dist_expr: SemanticExpr | None = None
            if len(args) == 3:
                value, destination, offset = args
                indices = (offset,)
            elif len(args) == 4:
                value, destination, offset, dist_expr = args
                indices = (offset,)
            else:
                raise TypeError(
                    "pto.psts expects 3 or 4 positional arguments in TileLang DSL v1: "
                    "`pto.psts(mask, buf, offset[, dist])`"
                )
            self._require_mask_expr(value, "pto.psts value")
            self._require_vector_pointer_expr(destination, "pto.psts destination")
            for index in indices:
                self._require_index_typed_expr(index)
            dist = self._normalize_predicate_store_dist(dist_expr, "pto.psts dist")
            return (
                SemanticPredicateStoreStmt(
                    value=value,
                    destination=destination,
                    indices=indices,
                    dist=dist,
                ),
                dict(env),
            )

        if expr.name in {"vsta", "vstas", "vstar"}:
            offset: SemanticExpr | None = None
            op_name = "vstas" if expr.name == "vsta" else expr.name
            if expr.name == "vsta":
                if len(expr.args) == 2:
                    value = self._analyze_expr(expr.args[0], env, allow_outer_lookup=allow_outer_lookup)
                    destination, indices = self._analyze_tile_vector_access(
                        expr.args[1],
                        env,
                        allow_outer_lookup=allow_outer_lookup,
                        context="pto.vsta destination",
                    )
                    offset = SemanticLiteralExpr(value=0, type=SemanticScalarType(dtype=i32))
                else:
                    args = tuple(
                        self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
                        for arg in expr.args
                    )
                    if len(args) != 3:
                        raise TypeError("pto.vsta expects 2 or 3 positional arguments in TileLang DSL v1")
                    value, destination, offset = args
                    indices = ()
            elif expr.name == "vstas":
                if len(expr.args) == 3 and isinstance(expr.args[1], FrontendSubscriptExpr):
                    value = self._analyze_expr(expr.args[0], env, allow_outer_lookup=allow_outer_lookup)
                    destination, indices = self._analyze_tile_vector_access(
                        expr.args[1],
                        env,
                        allow_outer_lookup=allow_outer_lookup,
                        context="pto.vstas destination",
                    )
                    offset = self._analyze_expr(expr.args[2], env, allow_outer_lookup=allow_outer_lookup)
                else:
                    args = tuple(
                        self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
                        for arg in expr.args
                    )
                    if len(args) != 3:
                        raise TypeError("pto.vstas expects exactly 3 positional arguments in TileLang DSL v1")
                    value, destination, offset = args
                    indices = ()
            else:
                if len(expr.args) == 2 and isinstance(expr.args[1], FrontendSubscriptExpr):
                    value = self._analyze_expr(expr.args[0], env, allow_outer_lookup=allow_outer_lookup)
                    destination, indices = self._analyze_tile_vector_access(
                        expr.args[1],
                        env,
                        allow_outer_lookup=allow_outer_lookup,
                        context="pto.vstar destination",
                    )
                else:
                    args = tuple(
                        self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
                        for arg in expr.args
                    )
                    if len(args) != 2:
                        raise TypeError("pto.vstar expects exactly 2 positional arguments in TileLang DSL v1")
                    value, destination = args
                    indices = ()
            self._require_align_expr(value, f"pto.{expr.name} value")
            self._require_vector_pointer_expr(destination, f"pto.{expr.name} destination")
            for index in indices:
                self._require_index_typed_expr(index)
            if offset is not None:
                self._require_i32_like_expr(offset, f"pto.{expr.name} offset")
            return (
                SemanticAlignStoreStmt(
                    op_name=op_name,
                    value=value,
                    destination=destination,
                    indices=indices,
                    offset=offset,
                ),
                dict(env),
            )

        if expr.name == "vsst":
            if len(expr.args) == 3:
                scalar = self._analyze_expr(expr.args[0], env, allow_outer_lookup=allow_outer_lookup)
                destination, indices = self._analyze_tile_vector_access(
                    expr.args[1],
                    env,
                    allow_outer_lookup=allow_outer_lookup,
                    context="pto.vsst destination",
                )
                mask = self._analyze_expr(expr.args[2], env, allow_outer_lookup=allow_outer_lookup)
            else:
                args = tuple(
                    self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
                    for arg in expr.args
                )
                if len(args) != 4:
                    raise TypeError("pto.vsst expects 3 or 4 positional arguments in TileLang DSL v1")
                scalar, destination, offset, mask = args
                indices = (offset,)
            scalar_type = self._require_scalar_expr(scalar, "pto.vsst scalar")
            self._require_vector_pointer_expr(destination, "pto.vsst destination")
            for index in indices:
                self._require_index_typed_expr(index)
            destination_dtype = destination.type.element_dtype
            if scalar_type.dtype != destination_dtype:
                raise TypeError("pto.vsst scalar dtype must match destination element dtype in TileLang DSL v1")
            value = SemanticCallExpr(
                namespace="pto",
                name="vbr",
                args=(scalar,),
                type=self._vreg_type_for_dtype(destination_dtype),
            )
            self._require_mask_for_vreg(mask, value.type, "pto.vsst")
            self._require_matching_vector_pointer(value.type, destination.type, "pto.vsst")
            return (
                SemanticVectorStoreStmt(
                    value=value,
                    destination=destination,
                    indices=indices,
                    mask=mask,
                ),
                dict(env),
            )

        if expr.name == "vsts":
            if len(expr.args) == 3:
                value = self._analyze_expr(expr.args[0], env, allow_outer_lookup=allow_outer_lookup)
                destination, indices = self._analyze_tile_vector_access(
                    expr.args[1],
                    env,
                    allow_outer_lookup=allow_outer_lookup,
                    context="pto.vsts destination",
                )
                mask = self._analyze_expr(expr.args[2], env, allow_outer_lookup=allow_outer_lookup)
            else:
                args = tuple(
                    self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
                    for arg in expr.args
                )
                if len(args) != 4:
                    raise TypeError("pto.vsts expects 3 or 4 positional arguments in TileLang DSL v1")
                value, destination, offset, mask = args
                indices = (offset,)
            self._require_vreg_expr(value, "pto.vsts value")
            self._require_vector_pointer_expr(destination, "pto.vsts destination")
            for index in indices:
                self._require_index_typed_expr(index)
            self._require_mask_for_vreg(mask, value.type, "pto.vsts")
            self._require_matching_vector_pointer(value.type, destination.type, "pto.vsts")
            return (
                SemanticVectorStoreStmt(
                    value=value,
                    destination=destination,
                    indices=indices,
                    mask=mask,
                ),
                dict(env),
            )

        if len(expr.args) == 5:
            low = self._analyze_expr(expr.args[0], env, allow_outer_lookup=allow_outer_lookup)
            high = self._analyze_expr(expr.args[1], env, allow_outer_lookup=allow_outer_lookup)
            destination, indices = self._analyze_tile_vector_access(
                expr.args[2],
                env,
                allow_outer_lookup=allow_outer_lookup,
                context="pto.vstsx2 destination",
            )
            dist = self._analyze_expr(expr.args[3], env, allow_outer_lookup=allow_outer_lookup)
            mask = self._analyze_expr(expr.args[4], env, allow_outer_lookup=allow_outer_lookup)
        else:
            args = tuple(
                self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
                for arg in expr.args
            )
            if len(args) != 6:
                raise TypeError("pto.vstsx2 expects 5 or 6 positional arguments in TileLang DSL v1")
            low, high, destination, offset, dist, mask = args
            indices = (offset,)
        low_type = self._require_vreg_expr(low, "pto.vstsx2 low")
        high_type = self._require_vreg_expr(high, "pto.vstsx2 high")
        if low_type != high_type:
            raise TypeError("pto.vstsx2 requires low/high vectors to use the same vector type")
        self._require_vector_pointer_expr(destination, "pto.vstsx2 destination")
        for index in indices:
            self._require_index_typed_expr(index)
        dist = self._normalize_vstsx2_dist(dist)
        self._require_mask_for_vreg(mask, low_type, "pto.vstsx2")
        self._require_matching_vector_pointer(low_type, destination.type, "pto.vstsx2")
        return (
            SemanticVectorPairStoreStmt(
                low=low,
                high=high,
                destination=destination,
                indices=indices,
                dist=dist,
                mask=mask,
            ),
            dict(env),
        )

    def _analyze_scalar_store_stmt(
        self,
        expr: FrontendCallExpr,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> tuple[SemanticStmt, dict[str, SemanticBinding]]:
        args = tuple(
            self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
            for arg in expr.args
        )
        if len(args) != 3:
            raise TypeError("pto.store_scalar expects exactly 3 positional arguments in TileLang DSL v1")
        if isinstance(args[0].type, SemanticPtrType):
            destination = self._require_pointer_expr(args[0], "pto.store_scalar destination")
            offset = args[1]
            value = args[2]
        else:
            value = args[0]
            destination = self._require_pointer_expr(args[1], "pto.store_scalar destination")
            offset = args[2]
        self._require_index_typed_expr(offset)
        value_type = self._require_scalar_expr(value, "pto.store_scalar value")
        if value_type.dtype != destination.type.element_dtype:
            raise TypeError("pto.store_scalar value dtype must match destination pointer element dtype")
        return (
            SemanticScalarStoreStmt(value=value, destination=destination, offset=offset),
            dict(env),
        )

    def _analyze_sync_stmt(
        self,
        expr: FrontendCallExpr,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> tuple[SemanticStmt, dict[str, SemanticBinding]]:
        args = tuple(
            self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
            for arg in expr.args
        )
        if expr.name in {"set_flag", "wait_flag"}:
            if len(args) != 3:
                raise TypeError(f"pto.{expr.name} expects exactly 3 positional arguments in TileLang DSL v1")
            src_pipe = self._require_sync_pipe(args[0], f"pto.{expr.name} source pipe")
            dst_pipe = self._require_sync_pipe(args[1], f"pto.{expr.name} destination pipe")
            event = self._require_sync_event(args[2], f"pto.{expr.name} event")
            if expr.name == "set_flag":
                return SemanticSetFlagStmt(src_pipe=src_pipe, dst_pipe=dst_pipe, event=event), dict(env)
            return SemanticWaitFlagStmt(src_pipe=src_pipe, dst_pipe=dst_pipe, event=event), dict(env)
        if expr.name in {"get_buf", "rls_buf"}:
            if len(args) not in {2, 3}:
                raise TypeError(f"pto.{expr.name} expects 2 or 3 positional arguments in TileLang DSL v1")
            pipe = self._require_sync_pipe(args[0], f"pto.{expr.name} pipe")
            self._require_i64_like_expr(args[1], f"pto.{expr.name} buf_id")
            mode = args[2] if len(args) == 3 else SemanticLiteralExpr(value=0, type=SemanticScalarType(dtype=i64))
            self._require_i64_like_expr(mode, f"pto.{expr.name} mode")
            if expr.name == "get_buf":
                return SemanticGetBufStmt(pipe=pipe, buf_id=args[1], mode=mode), dict(env)
            return SemanticRlsBufStmt(pipe=pipe, buf_id=args[1], mode=mode), dict(env)
        if expr.name == "mem_bar":
            if len(args) != 1:
                raise TypeError("pto.mem_bar expects exactly 1 positional argument in TileLang DSL v1")
            barrier_type = self._require_barrier_type(args[0], "pto.mem_bar barrier_type")
            return SemanticMemBarStmt(barrier_type=barrier_type), dict(env)
        if expr.name in {"set_cross_core", "set_intra_block", "wait_flag_dev", "wait_intra_core"}:
            if len(args) != 2:
                raise TypeError(f"pto.{expr.name} expects exactly 2 positional arguments in TileLang DSL v1")
            identifier = self._require_scalar_or_index_expr(args[0], f"pto.{expr.name} first operand")
            self._require_i64_like_expr(identifier, f"pto.{expr.name} first operand")
            event_id = self._normalize_event_id_expr(args[1], f"pto.{expr.name} event_id")
            if expr.name == "set_cross_core":
                return SemanticSetCrossCoreStmt(core_id=identifier, event_id=event_id), dict(env)
            if expr.name == "set_intra_block":
                return SemanticSetIntraBlockStmt(block_id=identifier, event_id=event_id), dict(env)
            if expr.name == "wait_flag_dev":
                return SemanticWaitFlagDevStmt(core_id=identifier, event_id=event_id), dict(env)
            return SemanticWaitIntraCoreStmt(block_id=identifier, event_id=event_id), dict(env)
        if expr.name == "set_intra_core":
            if len(args) != 1:
                raise TypeError("pto.set_intra_core expects exactly 1 positional argument in TileLang DSL v1")
            config = self._require_scalar_or_index_expr(args[0], "pto.set_intra_core config")
            self._require_i32_like_expr(config, "pto.set_intra_core config")
            return SemanticSetIntraCoreStmt(config=config), dict(env)
        if expr.name in {"pipe_barrier", "barrier"}:
            if len(args) != 1:
                raise TypeError(f"pto.{expr.name} expects exactly 1 positional argument in TileLang DSL v1")
            pipe = self._require_sync_pipe(args[0], f"pto.{expr.name} pipe")
            return SemanticPipeBarrierStmt(pipe=pipe), dict(env)
        raise ValueError(f"unsupported sync stmt pto.{expr.name}")

    def _analyze_low_level_dma_stmt(
        self,
        expr: FrontendCallExpr,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> tuple[SemanticStmt, dict[str, SemanticBinding]]:
        args = self._analyze_low_level_dma_operands(
            expr,
            env,
            allow_outer_lookup=allow_outer_lookup,
        )
        if expr.name in _LOW_LEVEL_DMA_UNARY_CONFIG_OPS:
            if len(args) != 1:
                raise TypeError(f"pto.{expr.name} expects exactly 1 positional argument in TileLang DSL")
            scalar = self._require_scalar_expr(args[0], f"pto.{expr.name} pad_value")
            if scalar.dtype.name not in _MOV_PAD_SUPPORTED_SCALAR_DTYPES:
                raise TypeError(
                    "pto.set_mov_pad_val pad_value must be one of i8, i16, i32, f16, bf16, or f32 in TileLang DSL v1"
                )
            return (
                SemanticDmaUnaryConfigStmt(
                    name=expr.name,
                    value=args[0],
                ),
                dict(env),
            )
        if expr.name in _LOW_LEVEL_DMA_CONFIG_OPS:
            if len(args) != 2:
                raise TypeError(f"pto.{expr.name} expects exactly 2 positional arguments in TileLang DSL")
            self._require_i64_like_expr(args[0], f"pto.{expr.name} first operand")
            self._require_i64_like_expr(args[1], f"pto.{expr.name} second operand")
            return (
                SemanticDmaConfigStmt(
                    name=expr.name,
                    first=args[0],
                    second=args[1],
                ),
                dict(env),
            )
        if expr.name == "copy_gm_to_ubuf":
            if len(args) != 11:
                raise TypeError("pto.copy_gm_to_ubuf expects exactly 11 positional arguments in TileLang DSL")
            source = self._require_pointer_expr(args[0], "pto.copy_gm_to_ubuf source", memory_space="gm")
            destination = self._require_pointer_expr(args[1], "pto.copy_gm_to_ubuf destination", memory_space="ub")
            for operand, label in zip(
                args[2:7] + args[8:],
                (
                    "sid",
                    "n_burst",
                    "len_burst",
                    "left_padding_count",
                    "right_padding_count",
                    "l2_cache_ctl",
                    "gm_stride",
                    "ub_stride",
                ),
            ):
                self._require_i64_like_expr(operand, f"pto.copy_gm_to_ubuf {label}")
            self._require_i1_expr(args[7], "pto.copy_gm_to_ubuf data_select_bit")
            return (
                SemanticLowLevelCopyStmt(
                    name=expr.name,
                    source=source,
                    destination=destination,
                    operands=args[2:],
                ),
                dict(env),
            )
        if expr.name == "copy_ubuf_to_gm":
            if len(args) != 8:
                raise TypeError("pto.copy_ubuf_to_gm expects exactly 8 positional arguments in TileLang DSL")
            source = self._require_pointer_expr(args[0], "pto.copy_ubuf_to_gm source", memory_space="ub")
            destination = self._require_pointer_expr(args[1], "pto.copy_ubuf_to_gm destination", memory_space="gm")
            for operand, label in zip(
                args[2:],
                (
                    "sid",
                    "n_burst",
                    "len_burst",
                    "reserved",
                    "burst_dst_stride",
                    "burst_src_stride",
                ),
            ):
                self._require_i64_like_expr(operand, f"pto.copy_ubuf_to_gm {label}")
            return (
                SemanticLowLevelCopyStmt(
                    name=expr.name,
                    source=source,
                    destination=destination,
                    operands=args[2:],
                ),
                dict(env),
            )
        if expr.name == "copy_ubuf_to_ubuf":
            if len(args) != 7:
                raise TypeError("pto.copy_ubuf_to_ubuf expects exactly 7 positional arguments in TileLang DSL")
            source = self._require_pointer_expr(args[0], "pto.copy_ubuf_to_ubuf source", memory_space="ub")
            destination = self._require_pointer_expr(args[1], "pto.copy_ubuf_to_ubuf destination", memory_space="ub")
            for operand, label in zip(
                args[2:],
                ("sid", "n_burst", "len_burst", "src_stride", "dst_stride"),
            ):
                self._require_i64_like_expr(operand, f"pto.copy_ubuf_to_ubuf {label}")
            return (
                SemanticLowLevelCopyStmt(
                    name=expr.name,
                    source=source,
                    destination=destination,
                    operands=args[2:],
                ),
                dict(env),
            )
        raise ValueError(f"unsupported low-level DMA stmt pto.{expr.name}")

    def _analyze_low_level_dma_operands(
        self,
        expr: FrontendCallExpr,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> tuple[SemanticExpr, ...]:
        if expr.args and expr.keywords:
            raise TypeError(
                f"pto.{expr.name} does not support mixing positional and keyword operands in TileLang DSL v1"
            )
        if not expr.keywords:
            return tuple(
                self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
                for arg in expr.args
            )

        analyzed_keywords: dict[str, SemanticExpr] = {
            name: self._analyze_expr(value, env, allow_outer_lookup=allow_outer_lookup)
            for name, value in expr.keywords
        }

        def index_literal(value: int) -> SemanticLiteralExpr:
            return SemanticLiteralExpr(value=value, type=SemanticIndexType())

        def bool_literal(value: bool) -> SemanticLiteralExpr:
            return SemanticLiteralExpr(value=value, type=SemanticScalarType(dtype=i1))

        if expr.name == "set_mov_pad_val":
            return (analyzed_keywords["pad_value"],)
        if expr.name in {
            "set_loop2_stride_outtoub",
            "set_loop1_stride_outtoub",
            "set_loop2_stride_ubtoout",
            "set_loop1_stride_ubtoout",
        }:
            return (
                analyzed_keywords["src_stride"],
                analyzed_keywords["dst_stride"],
            )
        if expr.name in {"set_loop_size_outtoub", "set_loop_size_ubtoout"}:
            return (
                analyzed_keywords["loop1"],
                analyzed_keywords["loop2"],
            )
        if expr.name == "copy_gm_to_ubuf":
            if "data_select_bit" in analyzed_keywords and "enable_ub_pad" in analyzed_keywords:
                raise TypeError(
                    "pto.copy_gm_to_ubuf keyword form accepts either `data_select_bit` or `enable_ub_pad`, not both"
                )
            return (
                analyzed_keywords["src"],
                analyzed_keywords["dst"],
                analyzed_keywords.get("sid", index_literal(0)),
                analyzed_keywords["n_burst"],
                analyzed_keywords["len_burst"],
                analyzed_keywords.get("left_padding_count", index_literal(0)),
                analyzed_keywords.get("right_padding_count", index_literal(0)),
                analyzed_keywords.get(
                    "data_select_bit",
                    analyzed_keywords.get("enable_ub_pad", bool_literal(False)),
                ),
                analyzed_keywords.get("l2_cache_ctl", index_literal(0)),
                analyzed_keywords["gm_stride"],
                analyzed_keywords["ub_stride"],
            )
        if expr.name == "copy_ubuf_to_gm":
            if "burst_dst_stride" in analyzed_keywords and "gm_stride" in analyzed_keywords:
                raise TypeError(
                    "pto.copy_ubuf_to_gm keyword form accepts either `burst_dst_stride` or `gm_stride`, not both"
                )
            if "burst_src_stride" in analyzed_keywords and "ub_stride" in analyzed_keywords:
                raise TypeError(
                    "pto.copy_ubuf_to_gm keyword form accepts either `burst_src_stride` or `ub_stride`, not both"
                )
            return (
                analyzed_keywords["src"],
                analyzed_keywords["dst"],
                analyzed_keywords.get("sid", index_literal(0)),
                analyzed_keywords["n_burst"],
                analyzed_keywords["len_burst"],
                analyzed_keywords.get("reserved", index_literal(0)),
                analyzed_keywords.get(
                    "burst_dst_stride",
                    analyzed_keywords["gm_stride"],
                ),
                analyzed_keywords.get(
                    "burst_src_stride",
                    analyzed_keywords["ub_stride"],
                ),
            )
        raise TypeError(
            f"pto.{expr.name} keyword form is not implemented in TileLang DSL v1"
        )

    def _require_tensor_slice(
        self,
        expr: SemanticExpr,
        context: str,
    ) -> SemanticTensorSliceExpr:
        if not isinstance(expr, SemanticTensorSliceExpr):
            raise TypeError(f"{context} must be a TensorView slice in TileLang DSL v1")
        return expr

    def _require_tile_expr(self, expr: SemanticExpr, context: str) -> SemanticExpr:
        if not isinstance(expr.type, SemanticTileType):
            raise TypeError(f"{context} must be a Tile value in TileLang DSL v1")
        if expr.type.rank != 2:
            raise TypeError(f"{context} currently only supports rank-2 Tile values in TileLang DSL v1")
        if expr.type.shape is None:
            raise TypeError(f"{context} requires a statically specialized Tile shape in TileLang DSL v1")
        if expr.type.memory_space != "ub":
            raise TypeError(f"{context} currently only supports MemorySpace.UB Tile values in TileLang DSL v1")
        return expr

    def _require_pointer_expr(
        self,
        expr: SemanticExpr,
        context: str,
        *,
        memory_space: str | None = None,
    ) -> SemanticExpr:
        if not isinstance(expr.type, SemanticPtrType):
            raise TypeError(f"{context} must be a pointer value in TileLang DSL")
        if memory_space is not None and expr.type.memory_space != memory_space:
            raise TypeError(f"{context} requires MemorySpace.{memory_space.upper()} pointers in TileLang DSL")
        return expr

    def _require_vector_pointer_expr(self, expr: SemanticExpr, context: str) -> SemanticExpr:
        if isinstance(expr.type, SemanticTileType):
            return self._require_tile_expr(expr, context)
        return self._require_pointer_expr(expr, context, memory_space="ub")

    def _validate_dma_common_types(
        self,
        tensor_slice_type: SemanticTensorSliceType,
        tile_type: SemanticTileType,
        op_name: str,
    ) -> None:
        if tensor_slice_type.rank != 2:
            raise TypeError(f"{op_name} currently only supports rank-2 TensorView slices in TileLang DSL v1")
        if tile_type.rank != 2 or tile_type.shape is None:
            raise TypeError(f"{op_name} requires a statically specialized rank-2 Tile in TileLang DSL v1")
        if tensor_slice_type.element_dtype != tile_type.element_dtype:
            raise TypeError(f"{op_name} requires matching TensorView/Tile element dtypes in TileLang DSL v1")

    def _validate_dma_load_profile(
        self,
        src: SemanticTensorSliceExpr,
        dst: SemanticExpr,
        options: SemanticDmaOptions,
    ) -> None:
        assert isinstance(dst.type, SemanticTileType)
        self._validate_dma_common_types(src.type, dst.type, "pto.dma_load")
        self._validate_dma_slice_profile(src, "pto.dma_load")

        pad_mode = self._pad_mode_value(options.pad_mode, default=PadMode.PadNull)
        left_padding = self._require_static_non_negative_index_value(
            options.left_padding,
            context="pto.dma_load left_padding",
            default=0,
        )
        right_padding = self._require_static_non_negative_index_value(
            options.right_padding,
            context="pto.dma_load right_padding",
            default=0,
        )
        self._require_static_bool_value(
            options.init_out_buffer,
            context="pto.dma_load init_out_buffer",
            default=False,
        )
        self._validate_dma_load_option_profile(options, pad_mode)

        valid_shape = self._resolved_tile_valid_shape(dst.type)
        expected_extents = (
            valid_shape[0],
            self._trimmed_tile_axis_extent(
                valid_shape[1],
                left_padding,
                right_padding,
                op_name="pto.dma_load",
                axis=1,
                window_label="destination Tile valid window",
            ),
        )
        self._validate_dma_extent_match(
            actual_extents=src.type.extents,
            expected_extents=expected_extents,
            op_name="pto.dma_load",
            actual_label="source slice",
            expected_label="destination Tile valid window",
            left_padding=left_padding,
            right_padding=right_padding,
        )

    def _validate_dma_store_profile(
        self,
        src: SemanticExpr,
        dst: SemanticTensorSliceExpr,
        options: SemanticDmaOptions,
    ) -> None:
        assert isinstance(src.type, SemanticTileType)
        self._validate_dma_common_types(dst.type, src.type, "pto.dma_store")
        self._validate_dma_slice_profile(dst, "pto.dma_store")

        pad_mode = self._pad_mode_value(options.pad_mode, default=PadMode.PadNull)
        left_padding = self._require_static_non_negative_index_value(
            options.left_padding,
            context="pto.dma_store left_padding",
            default=0,
        )
        right_padding = self._require_static_non_negative_index_value(
            options.right_padding,
            context="pto.dma_store right_padding",
            default=0,
        )
        self._validate_dma_store_option_profile(options, pad_mode)

        valid_shape = self._resolved_tile_valid_shape(src.type)
        expected_extents = (
            valid_shape[0],
            self._trimmed_tile_axis_extent(
                valid_shape[1],
                left_padding,
                right_padding,
                op_name="pto.dma_store",
                axis=1,
                window_label="source Tile interior window",
            ),
        )
        self._validate_dma_extent_match(
            actual_extents=dst.type.extents,
            expected_extents=expected_extents,
            op_name="pto.dma_store",
            actual_label="destination slice",
            expected_label="source Tile interior window",
            left_padding=left_padding,
            right_padding=right_padding,
        )

    def _validate_dma_slice_profile(
        self,
        tensor_slice: SemanticTensorSliceExpr,
        op_name: str,
    ) -> None:
        for axis, slice_axis in enumerate(tensor_slice.slices):
            step = self._static_index_value(slice_axis.step, default=1)
            if step is None:
                raise TypeError(
                    f"{op_name} stable frontend-only DMA profile requires a static positive "
                    f"slice step on axis {axis}"
                )
            if step <= 0:
                raise TypeError(
                    f"{op_name} stable frontend-only DMA profile requires a positive "
                    f"slice step on axis {axis}, got {step!r}"
                )
            if axis == 1 and step != 1:
                raise TypeError(
                    f"{op_name} stable frontend-only DMA profile only supports step == 1 "
                    "on TensorView slice axis 1"
                )

    def _validate_dma_load_option_profile(
        self,
        options: SemanticDmaOptions,
        pad_mode: PadMode,
    ) -> None:
        if pad_mode == PadMode.PadValue and options.pad_value is None:
            raise TypeError(
                "pto.dma_load stable frontend-only DMA profile requires `pad_value` when "
                "`pad_mode=PadMode.PadValue`"
            )
        if pad_mode != PadMode.PadValue and options.pad_value is not None:
            raise TypeError(
                "pto.dma_load stable frontend-only DMA profile only accepts `pad_value` "
                "when `pad_mode=PadMode.PadValue`"
            )

    def _validate_dma_store_option_profile(
        self,
        options: SemanticDmaOptions,
        pad_mode: PadMode,
    ) -> None:
        if options.pad_value is not None:
            raise TypeError(
                "pto.dma_store stable frontend-only DMA profile does not support `pad_value`; "
                "GM-side fill is unsupported"
            )
        if pad_mode != PadMode.PadNull:
            raise TypeError(
                "pto.dma_store stable frontend-only DMA profile only supports "
                "`pad_mode=PadMode.PadNull`; non-PadNull store padding would require GM-side fill"
            )

    def _resolved_tile_valid_shape(
        self,
        tile_type: SemanticTileType,
    ) -> tuple[int | None, ...]:
        assert tile_type.shape is not None
        return tile_type.shape if tile_type.valid_shape is None else tile_type.valid_shape

    def _trimmed_tile_axis_extent(
        self,
        base_extent: int | None,
        left_padding: int,
        right_padding: int,
        *,
        op_name: str,
        axis: int,
        window_label: str,
    ) -> int | None:
        if base_extent is None:
            return None
        trimmed_extent = base_extent - left_padding - right_padding
        if trimmed_extent <= 0:
            raise TypeError(
                f"{op_name} stable frontend-only DMA profile requires {window_label} axis {axis}="
                f"{base_extent!r} to remain positive after left_padding={left_padding} "
                f"and right_padding={right_padding}"
            )
        return trimmed_extent

    def _validate_dma_extent_match(
        self,
        *,
        actual_extents: tuple[int | None, ...],
        expected_extents: tuple[int | None, ...],
        op_name: str,
        actual_label: str,
        expected_label: str,
        left_padding: int,
        right_padding: int,
    ) -> None:
        for axis, (actual_extent, expected_extent) in enumerate(zip(actual_extents, expected_extents)):
            if actual_extent is None or expected_extent is None:
                continue
            if actual_extent != expected_extent:
                padding_suffix = ""
                if axis == 1 and (left_padding != 0 or right_padding != 0):
                    padding_suffix = (
                        f" after left_padding={left_padding} and right_padding={right_padding}"
                    )
                raise TypeError(
                    f"{op_name} stable frontend-only DMA profile requires {actual_label} extent "
                    f"axis {axis}={actual_extent!r} to match {expected_label} axis {axis}="
                    f"{expected_extent!r}{padding_suffix}"
                )

    def _bind_assignment_target(
        self,
        target: FrontendTargetNode,
        value: SemanticExpr,
        env: dict[str, SemanticBinding],
        annotation: Any | None,
    ) -> tuple[SemanticBinding, ...]:
        if isinstance(target, FrontendNameTarget):
            if isinstance(value.type, SemanticTupleType):
                raise ValueError("multi-result call assignment requires tuple binding in TileLang DSL v1")
            inferred_type: SemanticType = value.type
            if isinstance(value.type, SemanticTensorSliceType):
                # Tensor slicing materializes a logical partition descriptor value in IR.
                inferred_type = SemanticPartitionTensorViewType(
                    element_dtype=value.type.element_dtype,
                    rank=value.type.rank,
                )
            annotated_type = self._annotation_type(annotation, inferred_type, env)
            binding = self._make_binding(
                target.name,
                annotated_type if annotated_type is not None else inferred_type,
                "ssa",
                value=self._binding_value_for_expr(value),
            )
            env[target.name] = binding
            return (binding,)
        if isinstance(target, FrontendTupleTarget):
            if isinstance(value.type, SemanticTupleType):
                element_types = value.type.elements
            elif isinstance(value.type, SemanticShapeType):
                element_types = tuple(SemanticIndexType() for _ in range(value.type.rank))
            else:
                raise ValueError("tuple assignment expects a tuple-typed value")
            if annotation is not None:
                raise TypeError("annotated tuple assignment is not supported in TileLang DSL v1")
            if len(target.elements) != len(element_types):
                raise ValueError("tuple assignment arity must match the tuple value")
            tuple_values: tuple[SemanticExpr, ...]
            if isinstance(value, SemanticTupleExpr):
                tuple_values = value.elements
            elif isinstance(value, SemanticAttributeAccess) and isinstance(value.type, SemanticShapeType):
                if isinstance(value.base, SemanticBindingRef):
                    if isinstance(value.base.type, SemanticTileType) and value.attr == "valid_shape":
                        valid_shape = value.base.type.valid_shape
                        if valid_shape is not None:
                            for axis, dim in enumerate(valid_shape):
                                if dim is None:
                                    self._ensure_tile_valid_shape_parameter(value.base.binding, axis)
                tuple_values = tuple(
                    SemanticSubscriptAccess(
                        base=value,
                        index=SemanticLiteralExpr(value=axis, type=SemanticIndexType()),
                        type=SemanticIndexType(),
                    )
                    for axis in range(value.type.rank)
                )
            elif isinstance(value, SemanticCallExpr):
                tuple_values = value.args
            else:
                tuple_values = tuple(
                    SemanticLiteralExpr(value=None, type=element_type) for element_type in element_types
                )
            bindings = []
            for element, element_type, element_value in zip(target.elements, element_types, tuple_values):
                binding = self._make_binding(
                    element.name,
                    element_type,
                    "ssa",
                    value=self._binding_value_for_expr(element_value),
                )
                env[element.name] = binding
                bindings.append(binding)
            return tuple(bindings)
        raise ValueError(f"unsupported frontend assignment target {type(target).__name__}")

    def _binding_value_for_expr(self, expr: SemanticExpr) -> Any | None:
        return self._try_static_value(expr)

    def _annotation_type(
        self,
        annotation: Any | None,
        inferred_type: SemanticType | None,
        env: dict[str, SemanticBinding],
    ) -> SemanticType | None:
        if annotation is None:
            return inferred_type
        annotation_expr = self._analyze_annotation_expr(annotation, env)
        if isinstance(annotation_expr.type, SemanticMetaType):
            if annotation_expr.type.kind == "dtype" and isinstance(inferred_type, SemanticScalarType):
                dtype = self._require_dtype_symbol(annotation_expr, "annotated scalar type")
                if inferred_type.dtype != dtype:
                    raise TypeError(
                        f"annotated scalar type `{dtype!r}` does not match inferred {inferred_type.dtype!r}"
                    )
                return inferred_type
            if annotation_expr.type.kind == "ptr_type" and isinstance(inferred_type, SemanticPtrType):
                ptr_type = self._require_ptr_type_expr(annotation_expr, "annotated pointer type")
                if inferred_type.element_dtype != ptr_type.element_dtype:
                    raise TypeError(
                        f"annotated pointer type `{ptr_type!r}` does not match inferred pointer element type {inferred_type.element_dtype!r}"
                    )
                if inferred_type.memory_space != ptr_type.memory_space.value:
                    raise TypeError(
                        f"annotated pointer type `{ptr_type!r}` does not match inferred pointer memory space `{inferred_type.memory_space}`"
                    )
                return inferred_type
            if annotation_expr.type.kind == "vreg_type" and isinstance(inferred_type, SemanticVRegType):
                vreg_type = self._require_vreg_type_expr(annotation_expr, "annotated vector type")
                if inferred_type.element_dtype != vreg_type.element_dtype or inferred_type.lanes != vreg_type.lanes:
                    raise TypeError(
                        f"annotated vector type `{vreg_type!r}` does not match inferred !pto.vreg<{inferred_type.lanes}x{inferred_type.element_dtype.name}>"
                    )
                return inferred_type
            if annotation_expr.type.kind == "mask_type" and isinstance(inferred_type, SemanticMaskType):
                mask_type = self._require_mask_type_expr(annotation_expr, "annotated mask type")
                if inferred_type.granularity != mask_type.granularity:
                    raise TypeError(
                        f"annotated mask type `{mask_type!r}` does not match inferred !pto.mask<{inferred_type.granularity}>"
                    )
                return inferred_type
            if annotation_expr.type.kind == "align_type" and isinstance(inferred_type, SemanticAlignType):
                return inferred_type
            if (
                annotation_expr.type.kind == "partition_tensor_view_type"
                and isinstance(inferred_type, SemanticPartitionTensorViewType)
            ):
                return inferred_type
        raise TypeError("unsupported annotated assignment type in TileLang DSL v1")

    def _analyze_annotation_expr(
        self,
        annotation: ast.AST,
        env: dict[str, SemanticBinding],
    ) -> SemanticExpr:
        frontend_expr = self._build_frontend_annotation_expr(annotation)
        return self._analyze_expr(frontend_expr, env, allow_outer_lookup=True)

    def _build_frontend_annotation_expr(self, node: ast.AST) -> FrontendExprNode:
        if isinstance(node, ast.Name):
            return FrontendNameExpr(name=node.id)
        if isinstance(node, ast.Constant):
            return FrontendConstantExpr(value=node.value)
        if isinstance(node, ast.Attribute):
            path = self._annotation_attribute_path(node)
            if path is not None and path[0] in {"pto", "PAT", "PIPE", "EVENT"} and len(path) >= 2:
                return FrontendSymbolExpr(namespace=".".join(path[:-1]), name=path[-1])
            return FrontendAttributeExpr(
                base=self._build_frontend_annotation_expr(node.value),
                attr=node.attr,
            )
        if isinstance(node, ast.Call):
            if any(keyword.arg is None for keyword in node.keywords):
                raise TypeError("annotated assignment type does not support keyword unpacking in TileLang DSL v1")
            if node.keywords:
                raise TypeError("annotated assignment type does not support keyword arguments in TileLang DSL v1")
            if isinstance(node.func, ast.Name):
                return FrontendCallExpr(
                    namespace=None,
                    name=node.func.id,
                    args=tuple(self._build_frontend_annotation_expr(arg) for arg in node.args),
                    keywords=(),
                )
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                return FrontendCallExpr(
                    namespace=node.func.value.id,
                    name=node.func.attr,
                    args=tuple(self._build_frontend_annotation_expr(arg) for arg in node.args),
                    keywords=(),
                )
        raise TypeError("unsupported annotated assignment type in TileLang DSL v1")

    def _annotation_attribute_path(self, node: ast.AST) -> tuple[str, ...] | None:
        if isinstance(node, ast.Name):
            return (node.id,)
        if isinstance(node, ast.Attribute):
            base_path = self._annotation_attribute_path(node.value)
            if base_path is None:
                return None
            return base_path + (node.attr,)
        return None

    def _analyze_for(
        self,
        stmt: FrontendForStmt,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> tuple[SemanticStmt, dict[str, SemanticBinding]]:
        lower_bound = self._analyze_expr(stmt.lower_bound, env, allow_outer_lookup=allow_outer_lookup)
        upper_bound = self._analyze_expr(stmt.upper_bound, env, allow_outer_lookup=allow_outer_lookup)
        step = self._analyze_expr(stmt.step, env, allow_outer_lookup=allow_outer_lookup)
        for expr in (lower_bound, upper_bound, step):
            self._require_loop_bound_type(expr.type)

        body_env = dict(env)
        induction_variable = self._make_binding(stmt.target, SemanticIndexType(), "loop_iv")
        body_env[stmt.target] = induction_variable
        body, final_body_env = self._analyze_block(
            stmt.body,
            body_env,
            allow_outer_lookup=allow_outer_lookup,
        )

        updated_env = dict(env)
        loop_carried = []
        for name, outer_binding in env.items():
            final_binding = final_body_env.get(name)
            if final_binding is None or final_binding is outer_binding:
                continue
            merged_type = self._merge_loop_carried_types(outer_binding.type, final_binding.type)
            if merged_type is None:
                raise TypeError(
                    f"loop-carried binding '{name}' changes type from {outer_binding.type!r} to {final_binding.type!r}"
                )
            merged = self._make_binding(name, merged_type, "loop_result")
            updated_env[name] = merged
            loop_carried.append(merged)

        return (
            SemanticForStmt(
                induction_variable=induction_variable,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                step=step,
                body=body,
                loop_carried=tuple(loop_carried),
            ),
            updated_env,
        )

    def _analyze_if(
        self,
        stmt: FrontendIfStmt,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> tuple[SemanticStmt, dict[str, SemanticBinding]]:
        condition = self._analyze_expr(stmt.condition, env, allow_outer_lookup=allow_outer_lookup)
        self._require_condition_type(condition.type)
        if self._contains_meta_condition_operand(condition):
            raise TypeError(
                "if condition comparing meta values requires wrapping the condition with pto.constexpr(...) "
                "in TileLang DSL v1"
            )

        then_body, then_env = self._analyze_block(
            stmt.then_body,
            dict(env),
            allow_outer_lookup=allow_outer_lookup,
        )
        else_body, else_env = self._analyze_block(
            stmt.else_body,
            dict(env),
            allow_outer_lookup=allow_outer_lookup,
        )

        updated_env = dict(env)
        merged_results: list[SemanticIfResult] = []
        for name, outer_binding in env.items():
            then_binding = then_env.get(name, outer_binding)
            else_binding = else_env.get(name, outer_binding)
            if then_binding is outer_binding and else_binding is outer_binding:
                continue
            if then_binding.type != else_binding.type:
                raise TypeError(
                    f"if/else merge for '{name}' changes type between branches: "
                    f"{then_binding.type!r} vs {else_binding.type!r}"
                )
            merged_binding = self._make_binding(name, then_binding.type, "if_result")
            updated_env[name] = merged_binding
            merged_results.append(
                SemanticIfResult(
                    result_binding=merged_binding,
                    then_binding=then_binding,
                    else_binding=else_binding,
                )
            )

        return (
            SemanticIfStmt(
                condition=condition,
                then_body=then_body,
                else_body=else_body,
                results=tuple(merged_results),
            ),
            updated_env,
        )

    def _contains_meta_condition_operand(self, expr: SemanticExpr) -> bool:
        if isinstance(expr, SemanticBinaryExpr):
            if expr.op in {"eq", "ne"} and (
                isinstance(expr.lhs.type, SemanticMetaType) or isinstance(expr.rhs.type, SemanticMetaType)
            ):
                return True
            if expr.op in {"and", "or"}:
                return self._contains_meta_condition_operand(expr.lhs) or self._contains_meta_condition_operand(expr.rhs)
        return False

    def _analyze_constexpr_if(
        self,
        stmt: FrontendIfStmt,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> tuple[tuple[SemanticStmt, ...], dict[str, SemanticBinding]]:
        condition = self._analyze_expr(stmt.condition, env, allow_outer_lookup=allow_outer_lookup)
        self._require_condition_type(condition.type)
        static_value = self._require_constexpr_condition_bool(
            condition,
            context="if pto.constexpr(...) condition",
        )
        selected_body = stmt.then_body if static_value else stmt.else_body
        return self._analyze_block(
            selected_body,
            dict(env),
            allow_outer_lookup=allow_outer_lookup,
        )

    def _analyze_strict_vecscope(
        self,
        stmt: FrontendStrictVecscopeStmt,
        env: dict[str, SemanticBinding],
    ) -> tuple[SemanticStmt, dict[str, SemanticBinding]]:
        if not self.node.advanced_enabled:
            raise TypeError(advanced_mode_message("strict_vecscope"))
        if len(stmt.captures) != len(stmt.block_arguments):
            raise ValueError("strict_vecscope capture arity must match block arguments")

        captures = tuple(
            self._analyze_expr(expr, env, allow_outer_lookup=True)
            for expr in stmt.captures
        )
        scope_env: dict[str, SemanticBinding] = {}
        block_arguments = []
        for name, capture in zip(stmt.block_arguments, captures):
            if capture.type is None:
                raise TypeError(
                    f"strict_vecscope block argument '{name}' type could not be inferred"
                )
            block_binding = self._make_binding(name, capture.type, "strict_vecscope_arg")
            scope_env[name] = block_binding
            block_arguments.append(block_binding)
        self._disable_inference_depth += 1
        try:
            body, _ = self._analyze_block(
                stmt.body,
                scope_env,
                allow_outer_lookup=False,
                allow_inferred_vecscope=False,
            )
        finally:
            self._disable_inference_depth -= 1
        return (
            SemanticStrictVecscopeStmt(
                captures=captures,
                block_arguments=tuple(block_arguments),
                body=body,
            ),
            dict(env),
        )

    def _analyze_expr(
        self,
        expr: FrontendExprNode,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> SemanticExpr:
        if isinstance(expr, FrontendNameExpr):
            binding = env.get(expr.name)
            if binding is None:
                if allow_outer_lookup:
                    raise ValueError(f"unknown name '{expr.name}'")
                raise ValueError(
                    f"implicit capture of '{expr.name}' is not allowed in pto.strict_vecscope"
                )
            return self._attach_expr_source_location(
                SemanticBindingRef(binding=binding, type=binding.type),
                expr,
            )
        if isinstance(expr, FrontendConstantExpr):
            if isinstance(expr.value, bool):
                return self._attach_expr_source_location(
                    SemanticLiteralExpr(value=expr.value, type=SemanticScalarType(dtype=i1)),
                    expr,
                )
            if isinstance(expr.value, int):
                return self._attach_expr_source_location(
                    SemanticLiteralExpr(value=expr.value, type=SemanticIndexType()),
                    expr,
                )
            if isinstance(expr.value, float):
                return self._attach_expr_source_location(
                    SemanticLiteralExpr(
                        value=expr.value,
                        type=SemanticScalarType(dtype=f32),
                    ),
                    expr,
                )
            if isinstance(expr.value, str):
                return self._attach_expr_source_location(
                    SemanticLiteralExpr(
                        value=expr.value,
                        type=SemanticMetaType(kind="string"),
                    ),
                    expr,
                )
            if expr.value is None:
                return self._attach_expr_source_location(
                    SemanticLiteralExpr(value=None, type=SemanticIndexType()),
                    expr,
                )
            raise TypeError(f"unsupported constant {expr.value!r} in TileLang DSL v1")
        if isinstance(expr, FrontendSymbolExpr):
            return self._attach_expr_source_location(
                self._analyze_symbol_expr(expr),
                expr,
            )
        if isinstance(expr, FrontendSliceExpr):
            start = None if expr.start is None else self._analyze_expr(expr.start, env, allow_outer_lookup=allow_outer_lookup)
            stop = None if expr.stop is None else self._analyze_expr(expr.stop, env, allow_outer_lookup=allow_outer_lookup)
            step = None if expr.step is None else self._analyze_expr(expr.step, env, allow_outer_lookup=allow_outer_lookup)
            for item in (start, stop, step):
                if item is not None:
                    self._require_index_typed_expr(item)
            return self._attach_expr_source_location(
                SemanticSliceExpr(
                    start=start,
                    stop=stop,
                    step=step,
                    type=SemanticSliceType(),
                ),
                expr,
            )
        if isinstance(expr, FrontendTupleExpr):
            elements = tuple(
                self._analyze_expr(element, env, allow_outer_lookup=allow_outer_lookup)
                for element in expr.elements
            )
            return self._attach_expr_source_location(
                SemanticTupleExpr(
                    elements=elements,
                    type=SemanticTupleType(elements=tuple(element.type for element in elements)),
                ),
                expr,
            )
        if isinstance(expr, FrontendAttributeExpr):
            base = self._analyze_expr(expr.base, env, allow_outer_lookup=allow_outer_lookup)
            if expr.attr == "element_type":
                return self._attach_expr_source_location(self._element_type_expr(base), expr)
            if expr.attr == "rank":
                return self._attach_expr_source_location(self._rank_expr(base), expr)
            if expr.attr == "memory_space":
                return self._attach_expr_source_location(self._memory_space_expr(base), expr)
            if expr.attr == "pad_value" and isinstance(base.type, SemanticTileType):
                return self._attach_expr_source_location(self._tile_pad_value_expr(base), expr)
            if expr.attr == "config":
                return self._attach_expr_source_location(self._tile_config_expr(base), expr)
            if expr.attr == "valid_shape":
                return self._attach_expr_source_location(self._valid_shape_expr(base), expr)
            if expr.attr == "strides":
                return self._attach_expr_source_location(self._strides_expr(base), expr)
            if isinstance(base.type, SemanticTileConfigType):
                return self._attach_expr_source_location(self._tile_config_attr_expr(base, expr.attr), expr)
            attr_type = self._attribute_type(base, expr.attr)
            return self._attach_expr_source_location(
                SemanticAttributeAccess(base=base, attr=expr.attr, type=attr_type),
                expr,
            )
        if isinstance(expr, FrontendSubscriptExpr):
            base = self._analyze_expr(expr.base, env, allow_outer_lookup=allow_outer_lookup)
            index = self._analyze_expr(expr.index, env, allow_outer_lookup=allow_outer_lookup)
            result_type = self._subscript_type(base, index)
            if isinstance(result_type, SemanticTensorSliceType):
                slices = self._normalize_tensor_slice(index, base.type.rank)
                return self._attach_expr_source_location(
                    SemanticTensorSliceExpr(base=base, slices=slices, type=result_type),
                    expr,
                )
            return self._attach_expr_source_location(
                SemanticSubscriptAccess(base=base, index=index, type=result_type),
                expr,
            )
        if isinstance(expr, FrontendBinaryExpr):
            lhs = self._analyze_expr(expr.lhs, env, allow_outer_lookup=allow_outer_lookup)
            rhs = self._analyze_expr(expr.rhs, env, allow_outer_lookup=allow_outer_lookup)
            result_type = self._binary_type(lhs, rhs, expr.op)
            return self._attach_expr_source_location(
                SemanticBinaryExpr(lhs=lhs, op=expr.op, rhs=rhs, type=result_type),
                expr,
            )
        if isinstance(expr, FrontendCallExpr):
            if expr.namespace is None and expr.name in self._inline_proc_nodes:
                if expr.keywords:
                    raise TypeError(
                        f"inline_proc call `{expr.name}` reached semantic analysis with unresolved keywords in TileLang DSL v1"
                    )
                args = tuple(
                    self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
                    for arg in expr.args
                )
                return self._analyze_inline_proc_call_expr(expr.name, args)
            if expr.namespace is None and expr.name == "eval":
                if expr.keywords:
                    raise TypeError("method call `eval` does not support keyword arguments in TileLang DSL v1")
                if not expr.args:
                    raise TypeError("`eval()` expects a receiver in TileLang DSL v1")
                base = self._analyze_expr(expr.args[0], env, allow_outer_lookup=allow_outer_lookup)
                args = tuple(
                    self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
                    for arg in expr.args[1:]
                )
                return self._analyze_eval_method(base, args)
            if expr.namespace not in {None, "pto"} and expr.name == "eval":
                if expr.keywords:
                    raise TypeError("method call `eval` does not support keyword arguments in TileLang DSL v1")
                binding = env.get(expr.namespace)
                if binding is None:
                    if allow_outer_lookup:
                        raise ValueError(f"unknown name '{expr.namespace}'")
                    raise ValueError(
                        f"implicit capture of '{expr.namespace}' is not allowed in pto.strict_vecscope"
                    )
                base = SemanticBindingRef(binding=binding, type=binding.type)
                args = tuple(
                    self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
                    for arg in expr.args
                )
                return self._analyze_eval_method(base, args)
            if expr.namespace not in {None, "pto"} and expr.name == "as_ptr":
                if expr.keywords:
                    raise TypeError("method call `as_ptr` does not support keyword arguments in TileLang DSL v1")
                binding = env.get(expr.namespace)
                if binding is None:
                    if allow_outer_lookup:
                        raise ValueError(f"unknown name '{expr.namespace}'")
                    raise ValueError(
                        f"implicit capture of '{expr.namespace}' is not allowed in pto.strict_vecscope"
                    )
                base = SemanticBindingRef(binding=binding, type=binding.type)
                return self._analyze_as_ptr_method(base)
            if expr.namespace == "pto" and expr.name == "vlds" and len(expr.args) == 1:
                base, indices = self._analyze_tile_vector_access(
                    expr.args[0],
                    env,
                    allow_outer_lookup=allow_outer_lookup,
                    context="pto.vlds source",
                )
                return self._analyze_vlds((base, *indices))
            if (
                expr.namespace == "pto"
                and expr.name == "vldas"
                and len(expr.args) == 1
                and isinstance(expr.args[0], FrontendSubscriptExpr)
            ):
                base, indices = self._analyze_tile_vector_access(
                    expr.args[0],
                    env,
                    allow_outer_lookup=allow_outer_lookup,
                    context="pto.vldas source",
                )
                return self._analyze_vldas((base, *indices))
            if (
                expr.namespace == "pto"
                and expr.name == "vldus"
                and len(expr.args) == 2
                and isinstance(expr.args[0], FrontendSubscriptExpr)
            ):
                base, indices = self._analyze_tile_vector_access(
                    expr.args[0],
                    env,
                    allow_outer_lookup=allow_outer_lookup,
                    context="pto.vldus source",
                )
                align_expr = self._analyze_expr(expr.args[1], env, allow_outer_lookup=allow_outer_lookup)
                return self._analyze_vldus((base, *indices, align_expr))
            if expr.namespace == "pto" and expr.name == "vldsx2" and len(expr.args) == 2:
                base, indices = self._analyze_tile_vector_access(
                    expr.args[0],
                    env,
                    allow_outer_lookup=allow_outer_lookup,
                    context="pto.vldsx2 source",
                )
                dist = self._analyze_expr(expr.args[1], env, allow_outer_lookup=allow_outer_lookup)
                return self._analyze_vldsx2((base, *indices, dist))
            if expr.namespace == "pto" and expr.name == "vcvt":
                return self._analyze_vcvt_frontend_call(
                    expr,
                    env,
                    allow_outer_lookup=allow_outer_lookup,
                )
            if expr.keywords:
                raise TypeError(
                    f"call surface `{expr.namespace + '.' if expr.namespace else ''}{expr.name}` "
                    "carries keyword arguments, but semantic keyword handling is not implemented "
                    "in TileLang DSL v1 yet"
                )
            args = tuple(
                self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
                for arg in expr.args
            )
            return self._analyze_call_expr(expr.namespace, expr.name, args)
        raise ValueError(f"unsupported frontend expression {type(expr).__name__}")

    def _analyze_symbol_expr(self, expr: FrontendSymbolExpr) -> SemanticExpr:
        if expr.namespace == "pto":
            dtype = _DTYPE_SYMBOLS.get(expr.name)
            if dtype is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=dtype,
                    type=SemanticMetaType(kind="dtype"),
                )
            mask_type = _MASK_TYPE_SYMBOLS.get(expr.name)
            if mask_type is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=mask_type,
                    type=SemanticMetaType(kind="mask_type"),
                )
            if expr.name == "align":
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=align,
                    type=SemanticMetaType(kind="align_type"),
                )
            if expr.name == "PartitionTensorView":
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=expr.name,
                    type=SemanticMetaType(kind="partition_tensor_view_type"),
                )
        if expr.namespace in {"PAT", "pto.PAT", "pto.MaskPattern"}:
            pattern = _PATTERN_SYMBOLS.get(expr.name)
            if pattern is None and expr.name.startswith("PAT_"):
                canonical = expr.name[len("PAT_") :]
                pattern = _PATTERN_SYMBOLS.get(canonical)
            if pattern is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=pattern,
                    type=SemanticMetaType(kind="mask_pattern"),
                )
        if expr.namespace in {"PIPE", "pto.PIPE", "Pipe", "pto.Pipe"}:
            pipe = _PIPE_SYMBOLS.get(expr.name)
            if pipe is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=pipe,
                    type=SemanticMetaType(kind="pipe"),
                )
        if expr.namespace in {"EVENT", "pto.EVENT", "Event", "pto.Event"}:
            event = _EVENT_SYMBOLS.get(expr.name)
            if event is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=event,
                    type=SemanticMetaType(kind="event"),
                )
        if expr.namespace in {"BarrierType", "pto.BarrierType"}:
            barrier_type = _BARRIER_TYPE_SYMBOLS.get(expr.name)
            if barrier_type is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=barrier_type,
                    type=SemanticMetaType(kind="barrier_type"),
                )
        if expr.namespace in {"MemorySpace", "pto.MemorySpace"}:
            memory_space = _MEMORY_SPACE_SYMBOLS.get(expr.name)
            if memory_space is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=memory_space,
                    type=SemanticMetaType(kind="memory_space"),
                )
        if expr.namespace in {"PadMode", "pto.PadMode"}:
            pad_mode = _PAD_MODE_SYMBOLS.get(expr.name)
            if pad_mode is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=pad_mode,
                    type=SemanticMetaType(kind="pad_mode"),
                )
        if expr.namespace in {"BLayout", "pto.BLayout"}:
            b_layout = _B_LAYOUT_SYMBOLS.get(expr.name)
            if b_layout is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=b_layout,
                    type=SemanticMetaType(kind="b_layout"),
                )
        if expr.namespace in {"SLayout", "pto.SLayout"}:
            s_layout = _S_LAYOUT_SYMBOLS.get(expr.name)
            if s_layout is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=s_layout,
                    type=SemanticMetaType(kind="s_layout"),
                )
        if expr.namespace in {"PadValue", "pto.PadValue"}:
            pad_value = _PAD_VALUE_SYMBOLS.get(expr.name)
            if pad_value is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=pad_value,
                    type=SemanticPadValueType(),
                )
        if expr.namespace in {"DeinterleaveDist", "pto.DeinterleaveDist"}:
            dist = _DEINTERLEAVE_DIST_SYMBOLS.get(expr.name)
            if dist is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=dist,
                    type=SemanticMetaType(kind="deinterleave_dist"),
                )
        if expr.namespace in {"InterleaveDist", "pto.InterleaveDist"}:
            dist = _INTERLEAVE_DIST_SYMBOLS.get(expr.name)
            if dist is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=dist,
                    type=SemanticMetaType(kind="interleave_dist"),
                )
        if expr.namespace in {"PositionMode", "pto.PositionMode"}:
            position_mode = _POSITION_MODE_SYMBOLS.get(expr.name)
            if position_mode is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=position_mode,
                    type=SemanticMetaType(kind="position_mode"),
                )
        if expr.namespace in {"OrderMode", "pto.OrderMode"}:
            order_mode = _ORDER_MODE_SYMBOLS.get(expr.name)
            if order_mode is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=order_mode,
                    type=SemanticMetaType(kind="order_mode"),
                )
        if expr.namespace in {"VcvtRoundMode", "pto.VcvtRoundMode"}:
            round_mode = _VCVT_ROUND_MODE_SYMBOLS.get(expr.name)
            if round_mode is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=round_mode,
                    type=SemanticMetaType(kind="vcvt_round_mode"),
                )
        if expr.namespace in {"VcvtSatMode", "pto.VcvtSatMode"}:
            sat_mode = _VCVT_SAT_MODE_SYMBOLS.get(expr.name)
            if sat_mode is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=sat_mode,
                    type=SemanticMetaType(kind="vcvt_sat_mode"),
                )
        if expr.namespace in {"VcvtPartMode", "pto.VcvtPartMode"}:
            part_mode = _VCVT_PART_MODE_SYMBOLS.get(expr.name)
            if part_mode is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=part_mode,
                    type=SemanticMetaType(kind="vcvt_part_mode"),
                )
        if expr.namespace in {"PostUpdateMode", "pto.PostUpdateMode"}:
            post_update_mode = _POST_UPDATE_MODE_SYMBOLS.get(expr.name)
            if post_update_mode is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=post_update_mode,
                    type=SemanticMetaType(kind="post_update_mode"),
                )
        raise TypeError(
            f"symbol `{expr.namespace}.{expr.name}` is not supported in TileLang DSL v1"
        )

    def _attribute_type(self, base: SemanticExpr, attr: str) -> SemanticType:
        base_type = base.type
        if isinstance(base_type, (SemanticTensorViewType, SemanticPartitionTensorViewType)) and attr == "shape":
            return SemanticShapeType(rank=base_type.rank)
        if isinstance(base_type, (SemanticTensorViewType, SemanticPartitionTensorViewType)) and attr == "strides":
            return SemanticShapeType(rank=base_type.rank)
        if isinstance(base_type, SemanticTileType) and attr == "shape":
            return SemanticShapeType(rank=base_type.rank)
        if isinstance(base_type, (SemanticTensorViewType, SemanticPartitionTensorViewType, SemanticTileType)) and attr == "valid_shape":
            return SemanticShapeType(rank=base_type.rank)
        raise TypeError(f"unsupported attribute access '{attr}' in TileLang DSL v1")

    def _element_type_expr(self, base: SemanticExpr) -> SemanticExpr:
        base_type = base.type
        if isinstance(base_type, (SemanticTensorViewType, SemanticPartitionTensorViewType, SemanticTileType)):
            return SemanticSymbolExpr(
                namespace="pto",
                name=base_type.element_dtype.name,
                value=base_type.element_dtype,
                type=SemanticMetaType(kind="dtype"),
            )
        raise TypeError("unsupported attribute access 'element_type' in TileLang DSL v1")

    def _rank_expr(self, base: SemanticExpr) -> SemanticExpr:
        base_type = base.type
        if isinstance(base_type, (SemanticTensorViewType, SemanticPartitionTensorViewType, SemanticTileType)):
            return SemanticLiteralExpr(value=base_type.rank, type=SemanticIndexType())
        raise TypeError("unsupported attribute access 'rank' in TileLang DSL v1")

    def _memory_space_expr(self, base: SemanticExpr) -> SemanticExpr:
        base_type = base.type
        if isinstance(base_type, (SemanticTensorViewType, SemanticPartitionTensorViewType)):
            return SemanticSymbolExpr(
                namespace="pto",
                name=MemorySpace.GM.name,
                value=MemorySpace.GM,
                type=SemanticMetaType(kind="memory_space"),
            )
        if isinstance(base_type, SemanticTileType):
            memory_space = MemorySpace.UB if base_type.memory_space is None else MemorySpace(base_type.memory_space)
            return SemanticSymbolExpr(
                namespace="pto",
                name=memory_space.name,
                value=memory_space,
                type=SemanticMetaType(kind="memory_space"),
            )
        raise TypeError("unsupported attribute access 'memory_space' in TileLang DSL v1")

    def _tile_config_expr(self, base: SemanticExpr) -> SemanticExpr:
        base_type = base.type
        if isinstance(base_type, SemanticTileType):
            return SemanticLiteralExpr(
                value=base_type.config or TileConfig(),
                type=SemanticTileConfigType(element_dtype=base_type.element_dtype),
            )
        raise TypeError("unsupported attribute access 'config' in TileLang DSL v1")

    def _tile_pad_value_expr(self, base: SemanticExpr) -> SemanticExpr:
        base_type = base.type
        if not isinstance(base_type, SemanticTileType):
            raise TypeError("unsupported attribute access 'pad_value' in TileLang DSL v1")
        config = base_type.config or TileConfig()
        return SemanticSymbolExpr(
            namespace="pto",
            name=config.pad_value.name,
            value=config.pad_value,
            type=SemanticPadValueType(element_dtype=base_type.element_dtype),
        )

    def _pad_value_eval_expr(self, base: SemanticExpr) -> SemanticExpr:
        if not isinstance(base.type, SemanticPadValueType):
            raise TypeError("`eval()` expects a PadValue descriptor in TileLang DSL v1")
        if base.type.element_dtype is None:
            raise TypeError(
                "PadValue.eval() requires a Tile-bound or TileConfig-bound pad descriptor with an owning "
                "Tile element dtype in TileLang DSL v1"
            )
        pad_value = self._try_static_value(base)
        if not isinstance(pad_value, PadValue):
            raise TypeError("PadValue.eval() expects a statically known PadValue enum in TileLang DSL v1")
        pad_scalar = pad_value.materialize_scalar(base.type.element_dtype)
        if pad_scalar is None:
            raise TypeError(
                "PadValue.NULL.eval() is invalid in TileLang DSL v1; "
                "guard it with `pto.constexpr(tile.pad_value != pto.PadValue.NULL)` before calling `.eval()`"
            )
        return SemanticLiteralExpr(
            value=pad_scalar,
            type=SemanticScalarType(dtype=base.type.element_dtype),
        )

    def _analyze_eval_method(
        self,
        base: SemanticExpr,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if args:
            raise TypeError("`eval()` does not accept positional arguments in TileLang DSL v1")
        return self._pad_value_eval_expr(base)

    def _tile_config_attr_expr(self, base: SemanticExpr, attr: str) -> SemanticExpr:
        config = self._try_static_value(base)
        if not isinstance(config, TileConfig):
            raise TypeError("Tile config metadata must be statically known in TileLang DSL v1")
        if attr == "b_layout":
            return SemanticSymbolExpr(
                namespace="pto",
                name=config.b_layout.name,
                value=config.b_layout,
                type=SemanticMetaType(kind="b_layout"),
            )
        if attr == "s_layout":
            return SemanticSymbolExpr(
                namespace="pto",
                name=config.s_layout.name,
                value=config.s_layout,
                type=SemanticMetaType(kind="s_layout"),
            )
        if attr == "s_fractal_size":
            return SemanticLiteralExpr(
                value=config.s_fractal_size,
                type=SemanticScalarType(dtype=i32),
            )
        if attr == "pad_value":
            if not isinstance(base.type, SemanticTileConfigType):
                raise TypeError(
                    "TileConfig.pad_value expects a TileConfig value in TileLang DSL v1"
                )
            return SemanticSymbolExpr(
                namespace="pto",
                name=config.pad_value.name,
                value=config.pad_value,
                type=SemanticPadValueType(element_dtype=base.type.element_dtype),
            )
        raise TypeError(f"unsupported TileConfig attribute access '{attr}' in TileLang DSL v1")

    def _analyze_as_ptr_method(self, base: SemanticExpr) -> SemanticExpr:
        base_type = base.type
        if isinstance(base_type, (SemanticTensorViewType, SemanticPartitionTensorViewType)):
            return SemanticCallExpr(
                namespace="pto",
                name="tensor_view_as_ptr",
                args=(base,),
                type=SemanticPtrType(
                    element_dtype=base_type.element_dtype,
                    memory_space="gm",
                ),
            )
        if isinstance(base_type, SemanticTileType):
            return SemanticCallExpr(
                namespace="pto",
                name="tile_as_ptr",
                args=(base,),
                type=SemanticPtrType(
                    element_dtype=base_type.element_dtype,
                    memory_space=base_type.memory_space or "ub",
                ),
            )
        raise TypeError("`as_ptr()` expects a TensorView/PartitionTensorView or Tile value in TileLang DSL v1")

    def _valid_shape_expr(self, base: SemanticExpr) -> SemanticExpr:
        base_type = base.type
        if not isinstance(base_type, (SemanticTensorViewType, SemanticPartitionTensorViewType, SemanticTileType)):
            raise TypeError("unsupported attribute access 'valid_shape' in TileLang DSL v1")
        shape_access = SemanticAttributeAccess(
            base=base,
            attr="valid_shape",
            type=SemanticShapeType(rank=base_type.rank),
        )
        elements = []
        for axis in range(base_type.rank):
            if (
                isinstance(base, SemanticBindingRef)
                and isinstance(base.type, SemanticTileType)
                and base.type.valid_shape is not None
                and base.type.valid_shape[axis] is None
            ):
                self._ensure_tile_valid_shape_parameter(base.binding, axis)
            elements.append(
                SemanticSubscriptAccess(
                    base=shape_access,
                    index=SemanticLiteralExpr(value=axis, type=SemanticIndexType()),
                    type=SemanticIndexType(),
                )
            )
        return SemanticTupleExpr(
            elements=tuple(elements),
            type=SemanticTupleType(elements=tuple(SemanticIndexType() for _ in elements)),
        )

    def _strides_expr(self, base: SemanticExpr) -> SemanticExpr:
        base_type = base.type
        if not isinstance(base_type, (SemanticTensorViewType, SemanticPartitionTensorViewType)):
            raise TypeError("unsupported attribute access 'strides' in TileLang DSL v1")
        stride_access = SemanticAttributeAccess(
            base=base,
            attr="strides",
            type=SemanticShapeType(rank=base_type.rank),
        )
        elements = []
        for axis in range(base_type.rank):
            elements.append(
                SemanticSubscriptAccess(
                    base=stride_access,
                    index=SemanticLiteralExpr(value=axis, type=SemanticIndexType()),
                    type=SemanticIndexType(),
                )
            )
        return SemanticTupleExpr(
            elements=tuple(elements),
            type=SemanticTupleType(elements=tuple(SemanticIndexType() for _ in elements)),
        )

    def _subscript_type(self, base: SemanticExpr, index: SemanticExpr) -> SemanticType:
        if isinstance(base.type, SemanticShapeType):
            if not isinstance(index.type, SemanticIndexType):
                raise TypeError("shape subscript index must be an index value in TileLang DSL v1")
            if not isinstance(index, SemanticLiteralExpr) or not isinstance(index.value, int):
                raise TypeError(
                    "shape/stride/valid_shape subscript index must be an integer literal in TileLang DSL v1"
                )
            if index.value < 0 or index.value >= base.type.rank:
                raise TypeError(
                    f"shape subscript index {index.value} is out of bounds for rank {base.type.rank}"
                )
            return SemanticIndexType()
        if isinstance(base.type, SemanticTupleType):
            if not isinstance(index.type, SemanticIndexType):
                raise TypeError("tuple subscript index must be an index value in TileLang DSL v1")
            if not isinstance(base, SemanticTupleExpr):
                raise TypeError(
                    "tuple subscripting currently requires a shape-like tuple expression in TileLang DSL v1"
                )
            if not base.type.elements:
                raise TypeError("cannot subscript an empty tuple in TileLang DSL v1")
            if not isinstance(index, SemanticLiteralExpr) or not isinstance(index.value, int):
                raise TypeError("tuple subscript index must be an integer literal in TileLang DSL v1")

            if index.value < 0 or index.value >= len(base.type.elements):
                raise TypeError(
                    f"tuple subscript index {index.value} is out of bounds for tuple length {len(base.type.elements)}"
                )
            return base.type.elements[index.value]
        if isinstance(base.type, (SemanticTensorViewType, SemanticPartitionTensorViewType)):
            if not isinstance(index, SemanticTupleExpr):
                raise TypeError("TensorView slicing expects a tuple of slices in TileLang DSL v1")
            return self._tensor_slice_type(base.type, index)
        raise TypeError("unsupported subscript base in TileLang DSL v1")

    def _analyze_tile_vector_access(
        self,
        expr: FrontendExprNode,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
        context: str,
    ) -> tuple[SemanticExpr, tuple[SemanticExpr, ...]]:
        if not isinstance(expr, FrontendSubscriptExpr):
            raise TypeError(
                f"{context} expects Tile element-indexing syntax in TileLang DSL v1"
            )
        base = self._analyze_expr(expr.base, env, allow_outer_lookup=allow_outer_lookup)
        tile = self._require_tile_expr(base, context)
        indices = self._tile_vector_indices(
            expr.index,
            tile.type,
            env,
            allow_outer_lookup=allow_outer_lookup,
            context=context,
        )
        return base, indices

    def _tile_vector_indices(
        self,
        index_expr: FrontendExprNode,
        tile_type: SemanticTileType,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
        context: str,
    ) -> tuple[SemanticExpr, ...]:
        if tile_type.rank == 1:
            if not isinstance(index_expr, FrontendSliceExpr):
                raise TypeError(f"{context} expects Tile[start:] syntax for rank-1 Tile values")
            if index_expr.stop is not None:
                raise TypeError(f"{context} does not support explicit slice stop in TileLang DSL advanced mode")
            if index_expr.step is not None:
                raise TypeError(f"{context} does not support stepped Tile vector slices in TileLang DSL advanced mode")
            if index_expr.start is None:
                return (SemanticLiteralExpr(value=0, type=SemanticIndexType()),)
            start = self._analyze_expr(index_expr.start, env, allow_outer_lookup=allow_outer_lookup)
            self._require_index_typed_expr(start)
            return (start,)

        if tile_type.rank != 2 or tile_type.shape is None:
            raise TypeError(f"{context} currently only supports statically specialized rank-1 or rank-2 Tiles")
        if not isinstance(index_expr, FrontendTupleExpr) or len(index_expr.elements) != 2:
            raise TypeError(f"{context} expects Tile[row, col:] syntax for rank-2 Tile values")

        row_expr, col_expr = index_expr.elements
        if not isinstance(col_expr, FrontendSliceExpr):
            raise TypeError(f"{context} expects Tile[row, col:] syntax for rank-2 Tile values")
        if col_expr.stop is not None:
            raise TypeError(f"{context} does not support explicit slice stop in TileLang DSL advanced mode")
        if col_expr.step is not None:
            raise TypeError(f"{context} does not support stepped Tile vector slices in TileLang DSL advanced mode")

        row = self._analyze_expr(row_expr, env, allow_outer_lookup=allow_outer_lookup)
        self._require_index_typed_expr(row)
        if col_expr.start is None:
            col = SemanticLiteralExpr(value=0, type=SemanticIndexType())
        else:
            col = self._analyze_expr(col_expr.start, env, allow_outer_lookup=allow_outer_lookup)
            self._require_index_typed_expr(col)
        return (row, col)

    def _tensor_slice_type(
        self,
        tensor_type: SemanticTensorViewType | SemanticPartitionTensorViewType,
        index: SemanticTupleExpr,
    ) -> SemanticTensorSliceType:
        if not 1 <= len(index.elements) <= tensor_type.rank:
            raise TypeError(
                f"TensorView slice rank {len(index.elements)} must be between 1 and "
                f"{tensor_type.rank} in TileLang DSL v1"
            )
        axis_offset = tensor_type.rank - len(index.elements)
        extents = []
        for axis, element in enumerate(index.elements):
            if not isinstance(element, SemanticSliceExpr):
                raise TypeError(
                    f"TensorView slicing axis {axis} must use a Python slice in TileLang DSL v1"
                )
            self._require_optional_index_typed_expr(element.start)
            self._require_optional_index_typed_expr(element.stop)
            self._require_optional_index_typed_expr(element.step)

            if element.stop is None:
                raise TypeError("TensorView slicing requires explicit stop bounds in TileLang DSL v1")
            extents.append(self._normalized_tensor_slice_extent(element))
        return SemanticTensorSliceType(
            element_dtype=tensor_type.element_dtype,
            rank=len(index.elements),
            extents=tuple(extents),
            physical_axes=tuple(range(axis_offset, tensor_type.rank)),
        )

    def _normalize_tensor_slice(
        self,
        index: SemanticExpr,
        rank: int,
    ) -> tuple[SemanticTensorSliceAxis, ...]:
        if not isinstance(index, SemanticTupleExpr):
            raise TypeError("TensorView slicing expects a tuple index in TileLang DSL v1")
        if not 1 <= len(index.elements) <= rank:
            raise TypeError(
                f"TensorView slicing expects between 1 and {rank} slice elements in TileLang DSL v1"
            )
        slices = []
        for element in index.elements:
            if not isinstance(element, SemanticSliceExpr):
                raise TypeError("TensorView slicing only supports slice syntax in TileLang DSL v1")
            if element.stop is None:
                raise TypeError("TensorView slicing requires explicit stop bounds in TileLang DSL v1")
            start = self._normalize_optional_index_expr(element.start, default=0)
            stop = element.stop
            step = self._normalize_optional_index_expr(element.step, default=1)
            slices.append(
                SemanticTensorSliceAxis(
                    start=start,
                    stop=stop,
                    step=step,
                    extent=self._normalized_tensor_slice_extent(element),
                )
            )
        return tuple(slices)

    def _binary_type(
        self,
        lhs: SemanticExpr,
        rhs: SemanticExpr,
        op: str,
    ) -> SemanticType:
        if op in {"add", "sub", "mul", "mod", "floordiv", "bitand", "bitor", "bitxor", "lshift", "rshift"}:
            if isinstance(lhs.type, SemanticIndexType) and isinstance(rhs.type, SemanticIndexType):
                if op in {"add", "sub", "mul", "mod", "floordiv"}:
                    return SemanticIndexType()
            if isinstance(lhs.type, SemanticScalarType) and lhs.type == rhs.type:
                dtype = lhs.type.dtype
                if op in {"add", "sub", "mul"} and (is_integer_dtype(dtype) or is_float_dtype(dtype)):
                    return SemanticScalarType(dtype=dtype)
                if op in {"mod", "floordiv"} and is_integer_dtype(dtype):
                    return SemanticScalarType(dtype=dtype)
                if op in {"bitand", "bitor", "bitxor", "lshift", "rshift"} and is_integer_dtype(dtype):
                    return SemanticScalarType(dtype=dtype)
            raise TypeError(
                "binary expressions currently require matching index operands, "
                "or matching scalar operands (add/sub/mul for integer/float; "
                "mod/floordiv/bitwise/shift for integer)"
            )
        if op in {"eq", "ne"}:
            if isinstance(lhs.type, SemanticIndexType) and isinstance(rhs.type, SemanticIndexType):
                return SemanticScalarType(dtype=i1)
            if isinstance(lhs.type, SemanticScalarType) and lhs.type == rhs.type:
                return SemanticScalarType(dtype=i1)
            if isinstance(lhs.type, SemanticPadValueType) and isinstance(rhs.type, SemanticPadValueType):
                return SemanticScalarType(dtype=i1)
            if isinstance(lhs.type, SemanticMetaType) and lhs.type == rhs.type:
                return SemanticScalarType(dtype=i1)
            raise TypeError(
                "comparison expressions currently require matching scalar/meta types or index-typed operands"
            )
        if op in {"gt", "lt", "ge", "le"}:
            if isinstance(lhs.type, SemanticIndexType) and isinstance(rhs.type, SemanticIndexType):
                return SemanticScalarType(dtype=i1)
            if isinstance(lhs.type, SemanticScalarType) and lhs.type == rhs.type:
                return SemanticScalarType(dtype=i1)
            raise TypeError(
                "ordered comparison expressions currently require matching scalar types or index-typed operands"
            )
        if op in {"and", "or"}:
            self._require_condition_type(lhs.type)
            self._require_condition_type(rhs.type)
            return SemanticScalarType(dtype=i1)
        raise TypeError(f"unsupported binary operator '{op}' in TileLang DSL v1")

    def _analyze_call_expr(
        self,
        namespace: str | None,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if namespace is None and name == "range":
            return SemanticCallExpr(namespace=namespace, name=name, args=args, type=None)
        if namespace is None:
            raise TypeError(
                f"call surface `{name}` is not supported in TileLang DSL v1"
            )
        if namespace != "pto":
            raise TypeError(
                f"call surface `{namespace + '.' if namespace else ''}{name}` is not supported in TileLang DSL v1 yet"
            )
        if name in DEFERRED_PTO_SURFACES:
            raise TypeError(deferred_surface_message(name))
        if name in _DTYPE_SYMBOLS:
            return self._analyze_scalar_constructor(name, args)
        if name == "ptr":
            return self._analyze_ptr_type(args)
        if name == "vreg":
            return self._analyze_vreg_type(args)
        if name == "castptr":
            return self._analyze_castptr(args)
        if name == "addptr":
            return self._analyze_addptr(args)
        if name == "bytewidth":
            return self._analyze_bytewidth(args)
        if name in {"get_lanes", "elements_per_vreg"}:
            return self._analyze_get_lanes(args, call_name=name)
        if name == "constexpr":
            raise TypeError(
                "pto.constexpr(...) is only supported as an if-condition wrapper in TileLang DSL v1"
            )
        if name == "make_mask":
            return self._analyze_make_mask(args)
        if name in {
            "get_block_idx",
            "get_subblock_idx",
            "get_block_num",
            "get_subblock_num",
        }:
            return self._analyze_runtime_block_query(name, args)
        if name == "init_align":
            return self._analyze_init_align(args)
        if name == "vlds":
            return self._analyze_vlds(args)
        if name == "vldas":
            return self._analyze_vldas(args)
        if name == "vldus":
            return self._analyze_vldus(args)
        if name == "vldsx2":
            return self._analyze_vldsx2(args)
        if name == "pstu":
            return self._analyze_pstu(args)
        if name == "vstus":
            return self._analyze_vstus(args)
        if name == "vstur":
            return self._analyze_vstur(args)
        if name == "load_scalar":
            return self._analyze_load_scalar(args)
        if name in {"ppack", "punpack"}:
            return self._analyze_mask_part_op(name, args)
        if name in {"pnot", "psel"}:
            return self._analyze_mask_logic_op(name, args)
        if name in {"vcmp", "vcmps"}:
            return self._analyze_compare_op(name, args)
        if name in {"vsel", "vselr", "vselrv2"}:
            return self._analyze_select_op(name, args)
        if name in {"vaddc", "vsubc", "vaddcs", "vsubcs"}:
            return self._analyze_carry_op(name, args)
        if name in {"vintlv", "vdintlv", "vintlvv2", "vdintlvv2"}:
            return self._analyze_rearrangement_op(name, args)
        if name == "vcvt":
            return self._analyze_vcvt(args)
        if name == "vbitsort":
            return self._analyze_vbitsort(args)
        if name == "vmrgsort4":
            return self._analyze_vmrgsort4(args)
        if name in _BROADCAST_VECTOR_OPS:
            return self._analyze_broadcast_vector_op(name, args)
        if name in _MULTI_RESULT_VECTOR_OPS:
            return self._analyze_multi_result_vector_op(name, args)
        if name in _UNARY_VECTOR_OPS:
            return self._analyze_unary_vector_op(name, args)
        if name in _BINARY_VECTOR_OPS:
            return self._analyze_binary_vector_op(name, args)
        if name in _VECTOR_SCALAR_OPS:
            return self._analyze_vector_scalar_op(name, args)
        if name in _VECTOR_IMMEDIATE_OPS:
            return self._analyze_vector_immediate_op(name, args)
        if name in _TERNARY_VECTOR_OPS:
            return self._analyze_ternary_vector_op(name, args)
        raise TypeError(f"call surface `pto.{name}` is not supported in TileLang DSL v1 yet")

    def _analyze_make_mask(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if len(args) != 2:
            raise TypeError("pto.make_mask expects exactly 2 positional arguments in TileLang DSL v1")
        dtype_expr, value_expr = args
        dtype = self._require_dtype_symbol(dtype_expr, "pto.make_mask element type")
        if isinstance(value_expr, SemanticSymbolExpr) and value_expr.type.kind == "mask_pattern":
            return SemanticCallExpr(
                namespace="pto",
                name="make_mask",
                args=args,
                type=SemanticMaskType(granularity=self._mask_granularity_for_dtype(dtype)),
            )
        self._require_tail_remaining_expr(value_expr, "pto.make_mask tail remaining")
        return SemanticCallExpr(
            namespace="pto",
            name="make_mask",
            args=args,
            type=SemanticTupleType(
                elements=(
                    SemanticMaskType(granularity=self._mask_granularity_for_dtype(dtype)),
                    _I32_TYPE,
                )
            ),
        )

    def _analyze_scalar_constructor(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if len(args) != 1:
            raise TypeError(f"pto.{name} expects exactly 1 positional argument in TileLang DSL v1")

        target_dtype = _DTYPE_SYMBOLS[name]
        if (
            target_dtype.name in {"f16", "bf16", "f32"}
            and isinstance(args[0], SemanticLiteralExpr)
            and isinstance(args[0].type, SemanticMetaType)
            and args[0].type.kind == "string"
        ):
            parsed = self._parse_float_literal_string(args[0].value, target_dtype, f"pto.{name} value")
            return SemanticLiteralExpr(
                value=parsed,
                type=SemanticScalarType(dtype=target_dtype),
            )

        value = self._require_scalar_or_index_expr(args[0], f"pto.{name} value")

        if isinstance(value.type, SemanticScalarType) and value.type.dtype == target_dtype:
            return value

        if isinstance(value, SemanticLiteralExpr):
            literal_value = value.value
            if target_dtype == i1:
                if isinstance(literal_value, bool):
                    return SemanticLiteralExpr(value=literal_value, type=SemanticScalarType(dtype=i1))
                if isinstance(literal_value, int):
                    return SemanticLiteralExpr(value=bool(literal_value), type=SemanticScalarType(dtype=i1))
                if isinstance(literal_value, float):
                    return SemanticLiteralExpr(value=bool(literal_value), type=SemanticScalarType(dtype=i1))
            elif is_integer_dtype(target_dtype):
                if isinstance(literal_value, bool):
                    casted = int(literal_value)
                elif isinstance(literal_value, (int, float)):
                    casted = int(literal_value)
                else:
                    casted = None
                if casted is not None:
                    bits = integer_bitwidth(target_dtype)
                    signedness = integer_signedness(target_dtype)
                    assert bits is not None
                    if signedness == "unsigned":
                        min_value = 0
                        max_value = (1 << bits) - 1
                    else:
                        min_value = -(1 << (bits - 1))
                        max_value = (1 << (bits - 1)) - 1
                    if casted < min_value or casted > max_value:
                        raise TypeError(
                            f"pto.{name} value {casted} is out of range for {target_dtype.name} in TileLang DSL v1"
                        )
                    return SemanticLiteralExpr(value=casted, type=SemanticScalarType(dtype=target_dtype))
            else:
                if isinstance(literal_value, (bool, int, float)):
                    return SemanticLiteralExpr(
                        value=float(literal_value),
                        type=SemanticScalarType(dtype=target_dtype),
                    )

        return SemanticCallExpr(
            namespace="pto",
            name=name,
            args=(value,),
            type=SemanticScalarType(dtype=target_dtype),
        )

    def _parse_float_literal_string(
        self,
        literal: str,
        target_dtype: ScalarType,
        context: str,
    ) -> float:
        text = literal.strip().lower()
        if text in {"inf", "+inf", "infinity", "+infinity"}:
            return float("inf")
        if text in {"-inf", "-infinity"}:
            return float("-inf")
        if text in {"nan", "+nan", "-nan"}:
            return float("nan")

        if text.startswith("0x"):
            try:
                bit_pattern = int(text, 16)
            except ValueError as exc:
                raise TypeError(
                    f"{context} string literal {literal!r} is not a valid hex bit-pattern"
                ) from exc
            return self._float_from_bit_pattern(bit_pattern, target_dtype, context=context)

        try:
            return float(text)
        except ValueError as exc:
            raise TypeError(
                f"{context} string literal {literal!r} is not a valid float literal"
            ) from exc

    def _float_from_bit_pattern(
        self,
        bit_pattern: int,
        target_dtype: ScalarType,
        *,
        context: str,
    ) -> float:
        if target_dtype.name == "f16":
            if bit_pattern < 0 or bit_pattern > 0xFFFF:
                raise TypeError(f"{context} f16 bit-pattern must be in [0x0, 0xFFFF]")
            return float(struct.unpack(">e", struct.pack(">H", bit_pattern))[0])
        if target_dtype.name == "bf16":
            if bit_pattern < 0 or bit_pattern > 0xFFFF:
                raise TypeError(f"{context} bf16 bit-pattern must be in [0x0, 0xFFFF]")
            widened = bit_pattern << 16
            return float(struct.unpack(">f", struct.pack(">I", widened))[0])
        if target_dtype.name == "f32":
            if bit_pattern < 0 or bit_pattern > 0xFFFFFFFF:
                raise TypeError(f"{context} f32 bit-pattern must be in [0x0, 0xFFFFFFFF]")
            return float(struct.unpack(">f", struct.pack(">I", bit_pattern))[0])
        raise TypeError(f"{context} bit-pattern literals are not supported for dtype {target_dtype.name}")

    def _analyze_ptr_type(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if len(args) != 2:
            raise TypeError("pto.ptr expects exactly 2 positional arguments in TileLang DSL")
        dtype = self._require_dtype_symbol(args[0], "pto.ptr element type")
        memory_space = self._require_memory_space_symbol(args[1], "pto.ptr memory space")
        return SemanticLiteralExpr(
            value=PointerType(element_dtype=dtype, memory_space=memory_space),
            type=SemanticMetaType(kind="ptr_type"),
        )

    def _analyze_vreg_type(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if len(args) != 1:
            raise TypeError("pto.vreg expects exactly 1 positional argument in TileLang DSL v1")
        dtype = self._require_dtype_symbol(args[0], "pto.vreg element type")
        vreg_type = self._vreg_type_for_dtype(dtype)
        return SemanticLiteralExpr(
            value=VRegType(element_dtype=dtype, lanes=vreg_type.lanes),
            type=SemanticMetaType(kind="vreg_type"),
        )

    def _analyze_castptr(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if len(args) != 2:
            raise TypeError("pto.castptr expects exactly 2 positional arguments in TileLang DSL")
        value, target = args
        target_type = self._require_cast_target_type(target)
        if isinstance(target_type, SemanticPtrType):
            self._require_castptr_input(value, target_type)
        else:
            self._require_pointer_expr(value, "pto.castptr input")
        return SemanticCallExpr(namespace="pto", name="castptr", args=args, type=target_type)

    def _analyze_addptr(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if len(args) != 2:
            raise TypeError("pto.addptr expects exactly 2 positional arguments in TileLang DSL")
        pointer, offset = args
        ptr = self._require_pointer_expr(pointer, "pto.addptr pointer")
        self._require_index_typed_expr(offset)
        return SemanticCallExpr(namespace="pto", name="addptr", args=(ptr, offset), type=ptr.type)

    def _analyze_get_lanes(
        self,
        args: tuple[SemanticExpr, ...],
        *,
        call_name: str = "get_lanes",
    ) -> SemanticExpr:
        if len(args) != 1:
            raise TypeError(
                f"pto.{call_name} expects exactly 1 positional argument in TileLang DSL v1"
            )
        dtype = self._require_dtype_symbol(args[0], f"pto.{call_name} dtype")
        return SemanticLiteralExpr(value=self._vreg_type_for_dtype(dtype).lanes, type=SemanticIndexType())

    def _analyze_bytewidth(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if len(args) != 1:
            raise TypeError("pto.bytewidth expects exactly 1 positional argument in TileLang DSL v1")
        dtype = self._require_dtype_symbol(args[0], "pto.bytewidth dtype")
        return SemanticLiteralExpr(value=bytewidth(dtype), type=SemanticIndexType())

    def _analyze_init_align(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if args:
            raise TypeError("pto.init_align does not accept positional arguments in TileLang DSL v1")
        return SemanticCallExpr(namespace="pto", name="init_align", args=(), type=SemanticAlignType())

    def _analyze_vlds(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if len(args) < 2:
            raise TypeError("pto.vlds expects at least 2 positional arguments in TileLang DSL v1")
        source, *indices = args
        source_type = source.type
        if isinstance(source_type, SemanticTileType):
            source = self._require_tile_expr(source, "pto.vlds source")
        else:
            source = self._require_pointer_expr(source, "pto.vlds source", memory_space="ub")
        for index in indices:
            self._require_index_typed_expr(index)
        return SemanticCallExpr(
            namespace="pto",
            name="vlds",
            args=(source, *indices),
            type=self._vreg_type_for_dtype(source.type.element_dtype),
        )

    def _analyze_vldas(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if len(args) not in {1, 2, 3}:
            raise TypeError("pto.vldas expects 1 positional source or Tile[start:]/Tile[row, col:] in TileLang DSL v1")
        source, *indices = args
        source_type = source.type
        if isinstance(source_type, SemanticTileType):
            source = self._require_tile_expr(source, "pto.vldas source")
            for index in indices:
                self._require_index_typed_expr(index)
        else:
            if indices:
                raise TypeError("pto.vldas pointer syntax does not accept explicit indices in TileLang DSL v1")
            source = self._require_pointer_expr(source, "pto.vldas source", memory_space="ub")
        return SemanticCallExpr(
            namespace="pto",
            name="vldas",
            args=(source, *indices),
            type=SemanticAlignType(),
        )

    def _analyze_vldus(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if len(args) not in {2, 3, 4}:
            raise TypeError("pto.vldus expects (source, align) or Tile element-indexing syntax in TileLang DSL v1")
        source, *rest = args
        align_expr = rest[-1]
        index_args = rest[:-1]
        source_type = source.type
        if isinstance(source_type, SemanticTileType):
            source = self._require_tile_expr(source, "pto.vldus source")
            for index in index_args:
                self._require_index_typed_expr(index)
        else:
            if index_args:
                raise TypeError("pto.vldus pointer syntax does not accept explicit indices in TileLang DSL v1")
            source = self._require_pointer_expr(source, "pto.vldus source", memory_space="ub")
        self._require_align_expr(align_expr, "pto.vldus align")
        return SemanticCallExpr(
            namespace="pto",
            name="vldus",
            args=(source, *index_args, align_expr),
            type=SemanticTupleType(elements=(self._vreg_type_for_dtype(source.type.element_dtype), SemanticAlignType())),
        )

    def _analyze_vldsx2(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if len(args) not in {3, 4}:
            raise TypeError("pto.vldsx2 expects 3 or 4 positional arguments in TileLang DSL v1")
        source, *rest = args
        if len(rest) == 2:
            index_args = rest[:1]
            dist = rest[1]
        else:
            index_args = rest[:2]
            dist = rest[2]
        source_type = source.type
        if isinstance(source_type, SemanticTileType):
            source = self._require_tile_expr(source, "pto.vldsx2 source")
        else:
            source = self._require_pointer_expr(source, "pto.vldsx2 source", memory_space="ub")
        for index in index_args:
            self._require_index_typed_expr(index)
        dist = self._normalize_vldsx2_dist(dist)
        vreg_type = self._vreg_type_for_dtype(source.type.element_dtype)
        return SemanticCallExpr(
            namespace="pto",
            name="vldsx2",
            args=(source, *index_args, dist),
            type=SemanticTupleType(elements=(vreg_type, vreg_type)),
        )

    def _analyze_pstu(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if len(args) != 3:
            raise TypeError("pto.pstu expects exactly 3 positional arguments in TileLang DSL v1")
        align_expr, value, base = args
        self._require_align_expr(align_expr, "pto.pstu align_in")
        mask_type = self._require_mask_expr(value, "pto.pstu value")
        base = self._require_pointer_expr(base, "pto.pstu base", memory_space="ub")
        if mask_type.granularity == "b16":
            expected = ui16
        elif mask_type.granularity == "b32":
            expected = ui32
        else:
            raise TypeError("pto.pstu only supports !pto.mask<b16> and !pto.mask<b32> in TileLang DSL v1")
        if base.type.element_dtype != expected:
            raise TypeError(
                f"pto.pstu requires !pto.ptr<{expected.name}, ub> for mask granularity {mask_type.granularity}"
            )
        return SemanticCallExpr(
            namespace="pto",
            name="pstu",
            args=(align_expr, value, base),
            type=SemanticTupleType(elements=(SemanticAlignType(), base.type)),
        )

    def _analyze_vstus(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if len(args) != 4:
            raise TypeError("pto.vstus expects exactly 4 positional arguments in TileLang DSL v1")
        align_expr, offset, value, base = args
        self._require_align_expr(align_expr, "pto.vstus align_in")
        self._require_i32_like_expr(offset, "pto.vstus offset")
        self._require_vreg_expr(value, "pto.vstus value")
        base = self._require_pointer_expr(base, "pto.vstus base", memory_space="ub")
        return SemanticCallExpr(
            namespace="pto",
            name="vstus",
            args=(align_expr, offset, value, base),
            type=SemanticAlignType(),
        )

    def _analyze_vstur(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if len(args) not in {3, 4}:
            raise TypeError("pto.vstur expects 3 or 4 positional arguments in TileLang DSL v1")
        align_expr, value, base = args[:3]
        mode = self._normalize_post_update_mode(args[3] if len(args) == 4 else None, "pto.vstur mode")
        self._require_align_expr(align_expr, "pto.vstur align_in")
        self._require_vreg_expr(value, "pto.vstur value")
        base = self._require_pointer_expr(base, "pto.vstur base", memory_space="ub")
        return SemanticCallExpr(
            namespace="pto",
            name="vstur",
            args=(align_expr, value, base, mode),
            type=SemanticAlignType(),
        )

    def _analyze_load_scalar(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if len(args) == 2:
            destination_dtype = None
            pointer, offset = args
        elif len(args) == 3:
            destination_dtype = self._require_dtype_symbol(args[0], "pto.load_scalar result type")
            pointer, offset = args[1:]
        else:
            raise TypeError("pto.load_scalar expects 2 or 3 positional arguments in TileLang DSL v1")
        pointer = self._require_pointer_expr(pointer, "pto.load_scalar source")
        self._require_index_typed_expr(offset)
        if destination_dtype is not None and destination_dtype != pointer.type.element_dtype:
            raise TypeError("pto.load_scalar result type must match source pointer element dtype")
        return SemanticCallExpr(
            namespace="pto",
            name="load_scalar",
            args=(pointer, offset),
            type=SemanticScalarType(dtype=pointer.type.element_dtype),
        )

    def _analyze_runtime_block_query(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if args:
            raise TypeError(f"pto.{name} does not accept positional arguments in TileLang DSL v1")
        return SemanticCallExpr(
            namespace="pto",
            name=name,
            args=(),
            type=SemanticScalarType(dtype=i64),
        )

    def _analyze_broadcast_vector_op(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if name == "vbr":
            if len(args) != 1:
                raise TypeError("pto.vbr expects exactly 1 positional argument in TileLang DSL v1")
            value = args[0]
            vec_type = self._vreg_type_for_scalar_or_index(value, "pto.vbr value")
            return SemanticCallExpr(namespace="pto", name=name, args=args, type=vec_type)

        if name == "vdup":
            if len(args) not in {2, 3}:
                raise TypeError("pto.vdup expects 2 or 3 positional arguments in TileLang DSL v1")
            value = args[0]
            if isinstance(value.type, SemanticVRegType):
                vec_type = value.type
                mask = args[1]
                self._require_mask_for_vreg(mask, vec_type, "pto.vdup")
                position_arg = args[2] if len(args) == 3 else None
                position = self._normalize_position_mode(position_arg, "pto.vdup position")
                return SemanticCallExpr(
                    namespace="pto",
                    name=name,
                    args=(value, mask, position),
                    type=vec_type,
                )

            if len(args) == 3:
                raise TypeError(
                    "pto.vdup scalar input does not accept `position`; use `pto.vdup(input, mask)` "
                    "in TileLang DSL v1"
                )
            vec_type = self._vreg_type_for_scalar_or_index(value, "pto.vdup input")
            mask = args[1]
            self._require_mask_for_vreg(mask, vec_type, "pto.vdup")
            return SemanticCallExpr(
                namespace="pto",
                name=name,
                args=(value, mask),
                type=vec_type,
            )

        if name == "vci":
            if len(args) not in {1, 2}:
                raise TypeError("pto.vci expects 1 or 2 positional arguments in TileLang DSL v1")
            index = self._require_scalar_or_index_expr(args[0], "pto.vci index")
            index_dtype = i32 if isinstance(index.type, SemanticIndexType) else index.type.dtype
            if not (is_integer_dtype(index_dtype) and integer_bitwidth(index_dtype) in {8, 16, 32}):
                raise TypeError("pto.vci index only supports 8/16/32-bit integer dtypes in TileLang DSL v1")
            order_arg = args[1] if len(args) == 2 else None
            order = self._normalize_order_mode(order_arg, "pto.vci order")
            return SemanticCallExpr(
                namespace="pto",
                name=name,
                args=(index, order),
                type=self._vreg_type_for_dtype(index_dtype),
            )

        raise TypeError(f"call surface `pto.{name}` is not supported in TileLang DSL v1 yet")

    def _analyze_unary_vector_op(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if len(args) != 2:
            raise TypeError(f"pto.{name} expects exactly 2 positional arguments in TileLang DSL v1")
        value, mask = args
        vreg = self._require_vreg_expr(value, f"pto.{name} value")
        self._require_mask_for_vreg(mask, vreg, f"pto.{name}")
        self._validate_unary_dtype(name, vreg.element_dtype)
        return SemanticCallExpr(namespace="pto", name=name, args=args, type=vreg)

    def _analyze_binary_vector_op(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if len(args) != 3:
            raise TypeError(f"pto.{name} expects exactly 3 positional arguments in TileLang DSL v1")
        lhs_expr, rhs_expr, mask = args
        lhs = self._require_vreg_expr(lhs_expr, f"pto.{name} lhs")
        rhs = self._require_vreg_expr(rhs_expr, f"pto.{name} rhs")
        if name == "vperm":
            if not (is_integer_dtype(rhs.element_dtype) and integer_bitwidth(rhs.element_dtype) in {8, 16, 32}):
                raise TypeError("pto.vperm indices vector only supports integer vector dtypes in TileLang DSL v1")
            if lhs.lanes != rhs.lanes:
                raise TypeError("pto.vperm requires data/indices vectors to use the same lane width")
        elif lhs != rhs:
            raise TypeError(f"pto.{name} requires lhs/rhs vector types to match")
        self._require_mask_for_vreg(mask, lhs, f"pto.{name}")
        self._validate_binary_dtype(name, lhs.element_dtype)
        return SemanticCallExpr(namespace="pto", name=name, args=args, type=lhs)

    def _analyze_vector_scalar_op(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if len(args) != 3:
            raise TypeError(f"pto.{name} expects exactly 3 positional arguments in TileLang DSL v1")
        vector_expr, scalar_expr, mask = args
        vreg = self._require_vreg_expr(vector_expr, f"pto.{name} vector")
        scalar = self._require_scalar_expr(scalar_expr, f"pto.{name} scalar")
        if name in {"vshls", "vshrs"}:
            if scalar.dtype != i16:
                raise TypeError(f"pto.{name} scalar dtype must be i16")
        elif scalar.dtype != vreg.element_dtype:
            raise TypeError(f"pto.{name} scalar dtype must match vector element dtype")
        self._require_mask_for_vreg(mask, vreg, f"pto.{name}")
        self._validate_vector_scalar_dtype(name, vreg.element_dtype)
        return SemanticCallExpr(namespace="pto", name=name, args=args, type=vreg)

    def _analyze_vector_immediate_op(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if len(args) != 3:
            raise TypeError(f"pto.{name} expects exactly 3 positional arguments in TileLang DSL v1")
        vector = self._require_vreg_expr(args[0], f"pto.{name} vector")
        immediate = self._require_scalar_or_index_expr(args[1], f"pto.{name} immediate")
        if isinstance(immediate.type, SemanticScalarType) and not (
            is_integer_dtype(immediate.type.dtype) and integer_bitwidth(immediate.type.dtype) in {8, 16, 32}
        ):
            raise TypeError(f"pto.{name} immediate only supports 8/16/32-bit integer dtypes in TileLang DSL v1")
        self._require_mask_for_vreg(args[2], vector, f"pto.{name}")
        self._validate_vector_immediate_dtype(name, vector.element_dtype)
        return SemanticCallExpr(namespace="pto", name=name, args=args, type=vector)

    def _analyze_ternary_vector_op(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if len(args) != 4:
            raise TypeError(f"pto.{name} expects exactly 4 positional arguments in TileLang DSL v1")
        vec0 = self._require_vreg_expr(args[0], f"pto.{name} vec0")
        vec1 = self._require_vreg_expr(args[1], f"pto.{name} vec1")
        vec2 = self._require_vreg_expr(args[2], f"pto.{name} vec2")
        if not (vec0 == vec1 == vec2):
            raise TypeError(f"pto.{name} requires all vector operands to use the same vector type")
        self._require_mask_for_vreg(args[3], vec0, f"pto.{name}")
        self._validate_ternary_vector_dtype(name, vec0.element_dtype)
        return SemanticCallExpr(namespace="pto", name=name, args=args, type=vec0)

    def _analyze_multi_result_vector_op(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if name != "vmull":
            raise TypeError(f"call surface `pto.{name}` is not supported in TileLang DSL v1 yet")
        if len(args) != 3:
            raise TypeError("pto.vmull expects exactly 3 positional arguments in TileLang DSL")
        lhs = self._require_vreg_expr(args[0], "pto.vmull lhs")
        rhs = self._require_vreg_expr(args[1], "pto.vmull rhs")
        if lhs != rhs:
            raise TypeError("pto.vmull requires lhs/rhs vector types to match")
        self._require_mask_for_vreg(args[2], lhs, "pto.vmull")
        self._validate_multi_result_vector_dtype(name, lhs.element_dtype)
        return SemanticCallExpr(
            namespace="pto",
            name=name,
            args=args,
            type=SemanticTupleType(elements=(lhs, lhs)),
        )

    def _analyze_mask_part_op(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if len(args) != 2:
            raise TypeError(f"pto.{name} expects exactly 2 positional arguments in TileLang DSL")
        mask = self._require_mask_expr(args[0], f"pto.{name} mask")
        self._require_string_expr(args[1], f"pto.{name} part")
        return SemanticCallExpr(namespace="pto", name=name, args=args, type=mask)

    def _analyze_mask_logic_op(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if name == "pnot":
            if len(args) != 2:
                raise TypeError("pto.pnot expects exactly 2 positional arguments in TileLang DSL")
            value = self._require_mask_expr(args[0], "pto.pnot input")
            mask = self._require_mask_expr(args[1], "pto.pnot mask")
            self._require_matching_mask_types(value, mask, "pto.pnot")
            return SemanticCallExpr(namespace="pto", name=name, args=args, type=value)
        if len(args) != 3:
            raise TypeError("pto.psel expects exactly 3 positional arguments in TileLang DSL")
        src0 = self._require_mask_expr(args[0], "pto.psel src0")
        src1 = self._require_mask_expr(args[1], "pto.psel src1")
        mask = self._require_mask_expr(args[2], "pto.psel mask")
        self._require_matching_mask_types(src0, src1, "pto.psel")
        self._require_matching_mask_types(src0, mask, "pto.psel")
        return SemanticCallExpr(namespace="pto", name=name, args=args, type=src0)

    def _analyze_compare_op(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if name == "vcmp":
            if len(args) != 4:
                raise TypeError("pto.vcmp expects exactly 4 positional arguments in TileLang DSL")
            lhs = self._require_vreg_expr(args[0], "pto.vcmp lhs")
            rhs = self._require_vreg_expr(args[1], "pto.vcmp rhs")
            if lhs != rhs:
                raise TypeError("pto.vcmp requires lhs/rhs vector types to match")
            seed = self._require_mask_expr(args[2], "pto.vcmp seed mask")
            self._require_mask_for_vreg(args[2], lhs, "pto.vcmp")
            self._require_string_expr(args[3], "pto.vcmp compare mode")
            return SemanticCallExpr(
                namespace="pto",
                name=name,
                args=args,
                type=SemanticMaskType(granularity=seed.granularity),
            )

        if len(args) != 4:
            raise TypeError("pto.vcmps expects exactly 4 positional arguments in TileLang DSL")
        vector = self._require_vreg_expr(args[0], "pto.vcmps vector")
        scalar = self._require_scalar_expr(args[1], "pto.vcmps scalar")
        if scalar.dtype != vector.element_dtype:
            raise TypeError("pto.vcmps scalar dtype must match vector element dtype")
        seed = self._require_mask_expr(args[2], "pto.vcmps seed mask")
        self._require_mask_for_vreg(args[2], vector, "pto.vcmps")
        self._require_string_expr(args[3], "pto.vcmps compare mode")
        return SemanticCallExpr(
            namespace="pto",
            name=name,
            args=args,
            type=SemanticMaskType(granularity=seed.granularity),
        )

    def _analyze_select_op(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if name == "vsel":
            if len(args) != 3:
                raise TypeError("pto.vsel expects exactly 3 positional arguments in TileLang DSL")
            src0 = self._require_vreg_expr(args[0], "pto.vsel src0")
            src1 = self._require_vreg_expr(args[1], "pto.vsel src1")
            if src0 != src1:
                raise TypeError("pto.vsel requires src0/src1 vector types to match")
            self._require_mask_for_vreg(args[2], src0, "pto.vsel")
            return SemanticCallExpr(namespace="pto", name=name, args=args, type=src0)

        if len(args) != 2:
            raise TypeError(f"pto.{name} expects exactly 2 positional arguments in TileLang DSL")
        src0 = self._require_vreg_expr(args[0], f"pto.{name} src0")
        src1 = self._require_vreg_expr(args[1], f"pto.{name} src1")
        if src0 != src1:
            raise TypeError(f"pto.{name} requires src0/src1 vector types to match")
        return SemanticCallExpr(namespace="pto", name=name, args=args, type=src0)

    def _analyze_carry_op(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if name in {"vaddc", "vsubc"}:
            if len(args) != 3:
                raise TypeError(f"pto.{name} expects exactly 3 positional arguments in TileLang DSL")
            lhs = self._require_vreg_expr(args[0], f"pto.{name} lhs")
            rhs = self._require_vreg_expr(args[1], f"pto.{name} rhs")
            if lhs != rhs:
                raise TypeError(f"pto.{name} requires lhs/rhs vector types to match")
            self._require_mask_for_vreg(args[2], lhs, f"pto.{name}")
            carry_type = args[2].type
            return SemanticCallExpr(
                namespace="pto",
                name=name,
                args=args,
                type=SemanticTupleType(elements=(lhs, carry_type)),
            )

        if len(args) != 4:
            raise TypeError(f"pto.{name} expects exactly 4 positional arguments in TileLang DSL")
        lhs = self._require_vreg_expr(args[0], f"pto.{name} lhs")
        rhs = self._require_vreg_expr(args[1], f"pto.{name} rhs")
        if lhs != rhs:
            raise TypeError(f"pto.{name} requires lhs/rhs vector types to match")
        carry_in = self._require_mask_expr(args[2], f"pto.{name} carry_in")
        self._require_mask_for_vreg(args[3], lhs, f"pto.{name}")
        carry_mask = self._require_mask_expr(args[3], f"pto.{name} mask")
        self._require_matching_mask_types(carry_in, carry_mask, f"pto.{name}")
        return SemanticCallExpr(
            namespace="pto",
            name=name,
            args=args,
            type=SemanticTupleType(elements=(lhs, carry_in)),
        )

    def _analyze_rearrangement_op(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if name in {"vintlv", "vdintlv"}:
            if len(args) != 2:
                raise TypeError(f"pto.{name} expects exactly 2 positional arguments in TileLang DSL")
            lhs = self._require_vreg_expr(args[0], f"pto.{name} lhs")
            rhs = self._require_vreg_expr(args[1], f"pto.{name} rhs")
            if lhs != rhs:
                raise TypeError(f"pto.{name} requires lhs/rhs vector types to match")
            return SemanticCallExpr(
                namespace="pto",
                name=name,
                args=args,
                type=SemanticTupleType(elements=(lhs, lhs)),
            )

        if len(args) != 3:
            raise TypeError(f"pto.{name} expects exactly 3 positional arguments in TileLang DSL")
        lhs = self._require_vreg_expr(args[0], f"pto.{name} lhs")
        rhs = self._require_vreg_expr(args[1], f"pto.{name} rhs")
        if lhs != rhs:
            raise TypeError(f"pto.{name} requires lhs/rhs vector types to match")
        self._require_string_expr(args[2], f"pto.{name} part")
        return SemanticCallExpr(namespace="pto", name=name, args=args, type=lhs)

    def _missing_optional_meta_expr(self) -> SemanticLiteralExpr:
        return SemanticLiteralExpr(value=None, type=SemanticMetaType(kind="none"))

    def _analyze_vcvt_frontend_call(
        self,
        expr: FrontendCallExpr,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> SemanticExpr:
        if len(expr.args) != 3:
            raise TypeError(
                "pto.vcvt expects exactly 3 positional operands `(vec, to_type, mask)` "
                "before optional keyword attrs in TileLang DSL v1"
            )
        args = tuple(
            self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
            for arg in expr.args
        )
        analyzed_keywords = {
            name: self._analyze_expr(value, env, allow_outer_lookup=allow_outer_lookup)
            for name, value in expr.keywords
        }
        allowed_keywords = {"rnd", "sat", "part"}
        unexpected_keywords = sorted(set(analyzed_keywords) - allowed_keywords)
        if unexpected_keywords:
            keyword_text = ", ".join(unexpected_keywords)
            raise TypeError(
                "pto.vcvt only accepts keyword attrs `rnd`, `sat`, and `part`; "
                f"got unsupported keyword(s): {keyword_text}"
            )
        return self._analyze_vcvt(
            args,
            rnd=self._normalize_vcvt_round_mode(analyzed_keywords.get("rnd")),
            sat=self._normalize_vcvt_sat_mode(analyzed_keywords.get("sat")),
            part=self._normalize_vcvt_part_mode(analyzed_keywords.get("part")),
            rnd_explicit="rnd" in analyzed_keywords,
            sat_explicit="sat" in analyzed_keywords,
            part_explicit="part" in analyzed_keywords,
        )

    def _analyze_vcvt(
        self,
        args: tuple[SemanticExpr, ...],
        *,
        rnd: SemanticExpr | None = None,
        sat: SemanticExpr | None = None,
        part: SemanticExpr | None = None,
        rnd_explicit: bool = False,
        sat_explicit: bool = False,
        part_explicit: bool = False,
    ) -> SemanticExpr:
        if len(args) != 3:
            raise TypeError("pto.vcvt expects exactly 3 positional arguments in TileLang DSL")
        vector = self._require_vreg_expr(args[0], "pto.vcvt vector")
        target_dtype = self._require_dtype_symbol(args[1], "pto.vcvt to_type")
        self._require_mask_for_vreg(args[2], vector, "pto.vcvt")
        contract = self._lookup_vcvt_attr_contract(vector.element_dtype, target_dtype)
        if contract is not None:
            self._require_explicit_vcvt_attrs(
                src_dtype=vector.element_dtype,
                dst_dtype=target_dtype,
                rnd_required=contract[0],
                sat_required=contract[1],
                part_required=contract[2],
                rnd_explicit=rnd_explicit,
                sat_explicit=sat_explicit,
                part_explicit=part_explicit,
            )
        return SemanticCallExpr(
            namespace="pto",
            name="vcvt",
            args=(
                args[0],
                args[1],
                args[2],
                rnd if rnd is not None else self._missing_optional_meta_expr(),
                sat if sat is not None else self._missing_optional_meta_expr(),
                part if part is not None else self._missing_optional_meta_expr(),
            ),
            type=self._vreg_type_for_dtype(target_dtype),
        )

    def _analyze_vbitsort(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if len(args) != 4:
            raise TypeError("pto.vbitsort expects exactly 4 positional arguments in TileLang DSL v1")
        destination = self._require_pointer_expr(args[0], "pto.vbitsort destination", memory_space="ub")
        source = self._require_pointer_expr(args[1], "pto.vbitsort source", memory_space="ub")
        indices = self._require_pointer_expr(args[2], "pto.vbitsort indices", memory_space="ub")
        self._require_index_typed_expr(args[3])
        return SemanticCallExpr(
            namespace="pto",
            name="vbitsort",
            args=(destination, source, indices, args[3]),
            type=None,
        )

    def _analyze_vmrgsort4(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if len(args) != 7:
            raise TypeError("pto.vmrgsort4 expects exactly 7 positional arguments in TileLang DSL v1")
        destination = self._require_pointer_expr(args[0], "pto.vmrgsort4 destination", memory_space="ub")
        source0 = self._require_pointer_expr(args[1], "pto.vmrgsort4 src0", memory_space="ub")
        source1 = self._require_pointer_expr(args[2], "pto.vmrgsort4 src1", memory_space="ub")
        source2 = self._require_pointer_expr(args[3], "pto.vmrgsort4 src2", memory_space="ub")
        source3 = self._require_pointer_expr(args[4], "pto.vmrgsort4 src3", memory_space="ub")
        self._require_i64_like_expr(args[5], "pto.vmrgsort4 count")
        self._require_i64_like_expr(args[6], "pto.vmrgsort4 config")
        return SemanticCallExpr(
            namespace="pto",
            name="vmrgsort4",
            args=(destination, source0, source1, source2, source3, args[5], args[6]),
            type=None,
        )

    def _require_dtype_symbol(self, expr: SemanticExpr, context: str) -> ScalarType:
        if not (
            isinstance(expr, SemanticSymbolExpr)
            and expr.type.kind == "dtype"
            and isinstance(expr.value, ScalarType)
        ):
            if (
                isinstance(expr, SemanticBindingRef)
                and isinstance(expr.type, SemanticMetaType)
                and expr.type.kind == "dtype"
                and isinstance(expr.binding.value, ScalarType)
            ):
                return expr.binding.value
            raise TypeError(f"{context} must be a TileLang scalar dtype symbol in TileLang DSL v1")
        return expr.value

    def _require_memory_space_symbol(self, expr: SemanticExpr, context: str) -> MemorySpace:
        if (
            isinstance(expr, SemanticSymbolExpr)
            and expr.type.kind == "memory_space"
            and isinstance(expr.value, MemorySpace)
        ):
            return expr.value
        if (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "memory_space"
            and isinstance(expr.binding.value, MemorySpace)
        ):
            return expr.binding.value
        raise TypeError(f"{context} must be a TileLang MemorySpace symbol")

    def _require_ptr_type_expr(self, expr: SemanticExpr, context: str) -> PointerType:
        if (
            isinstance(expr, SemanticLiteralExpr)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "ptr_type"
            and isinstance(expr.value, PointerType)
        ):
            return expr.value
        if (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "ptr_type"
            and isinstance(expr.binding.value, PointerType)
        ):
            return expr.binding.value
        raise TypeError(f"{context} must be a pointer type constructed with pto.ptr(...)")

    def _require_vreg_type_expr(self, expr: SemanticExpr, context: str) -> VRegType:
        if (
            isinstance(expr, SemanticLiteralExpr)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "vreg_type"
            and isinstance(expr.value, VRegType)
        ):
            return expr.value
        if (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "vreg_type"
            and isinstance(expr.binding.value, VRegType)
        ):
            return expr.binding.value
        raise TypeError(f"{context} must be a vector type constructed with pto.vreg(...)")

    def _require_mask_type_expr(self, expr: SemanticExpr, context: str) -> MaskType:
        if (
            isinstance(expr, SemanticSymbolExpr)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "mask_type"
            and isinstance(expr.value, MaskType)
        ):
            return expr.value
        if (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "mask_type"
            and isinstance(expr.binding.value, MaskType)
        ):
            return expr.binding.value
        raise TypeError(f"{context} must be a mask type such as pto.mask_b32")

    def _require_cast_target_type(self, expr: SemanticExpr) -> SemanticType:
        if self._is_i64_dtype_expr(expr):
            return SemanticScalarType(dtype=i64)
        ptr_type = self._require_ptr_type_expr(expr, "pto.castptr target type")
        return SemanticPtrType(
            element_dtype=ptr_type.element_dtype,
            memory_space=ptr_type.memory_space.value,
        )

    def _require_castptr_input(self, expr: SemanticExpr, target_type: SemanticPtrType) -> None:
        if isinstance(expr.type, SemanticIndexType):
            return
        if isinstance(expr.type, SemanticScalarType) and expr.type.dtype == i64:
            return
        if isinstance(expr.type, SemanticPtrType):
            if expr.type.memory_space != target_type.memory_space:
                raise TypeError("pto.castptr pointer-to-pointer casts must stay within one PTO memory space")
            return
        raise TypeError("pto.castptr input must be an index/i64, pointer, or memref-backed address value")

    def _is_i64_dtype_expr(self, expr: SemanticExpr) -> bool:
        if isinstance(expr, SemanticSymbolExpr):
            return expr.type.kind == "dtype" and expr.value == i64
        if isinstance(expr, SemanticBindingRef):
            return (
                isinstance(expr.type, SemanticMetaType)
                and expr.type.kind == "dtype"
                and expr.binding.value == i64
            )
        return False

    def _require_vreg_expr(self, expr: SemanticExpr, context: str) -> SemanticVRegType:
        if not isinstance(expr.type, SemanticVRegType):
            raise TypeError(f"{context} must be a vector register value in TileLang DSL v1")
        return expr.type

    def _require_scalar_expr(self, expr: SemanticExpr, context: str) -> SemanticScalarType:
        if not isinstance(expr.type, SemanticScalarType):
            raise TypeError(f"{context} must be a scalar value in TileLang DSL v1")
        return expr.type

    def _require_scalar_or_index_expr(self, expr: SemanticExpr, context: str) -> SemanticExpr:
        if isinstance(expr.type, (SemanticScalarType, SemanticIndexType)):
            return expr
        raise TypeError(f"{context} must be a scalar or index value in TileLang DSL v1")

    def _vreg_type_for_scalar_or_index(self, expr: SemanticExpr, context: str) -> SemanticVRegType:
        value = self._require_scalar_or_index_expr(expr, context)
        if isinstance(value.type, SemanticScalarType):
            return self._vreg_type_for_dtype(value.type.dtype)
        return self._vreg_type_for_dtype(i32)

    def _normalize_position_mode(
        self,
        expr: SemanticExpr | None,
        context: str,
    ) -> SemanticExpr:
        if expr is None:
            return SemanticLiteralExpr(value=PositionMode.LOWEST.value, type=SemanticMetaType(kind="string"))
        if (
            isinstance(expr, SemanticSymbolExpr)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "position_mode"
            and isinstance(expr.value, PositionMode)
        ):
            return SemanticLiteralExpr(value=expr.value.value, type=SemanticMetaType(kind="string"))
        if (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "position_mode"
            and isinstance(expr.binding.value, PositionMode)
        ):
            return SemanticLiteralExpr(value=expr.binding.value.value, type=SemanticMetaType(kind="string"))
        position = self._require_string_expr(expr, context)
        if position == "POS_LOWEST":
            position = PositionMode.LOWEST.value
        if position not in {PositionMode.LOWEST.value, PositionMode.HIGHEST.value}:
            raise TypeError(
                "pto.vdup position must be `PositionMode.LOWEST` or `PositionMode.HIGHEST` in TileLang DSL v1"
            )
        return SemanticLiteralExpr(value=position, type=SemanticMetaType(kind="string"))

    def _normalize_order_mode(
        self,
        expr: SemanticExpr | None,
        context: str,
    ) -> SemanticExpr:
        if expr is None:
            return SemanticLiteralExpr(value=OrderMode.ASC.value, type=SemanticMetaType(kind="string"))
        if (
            isinstance(expr, SemanticSymbolExpr)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "order_mode"
            and isinstance(expr.value, OrderMode)
        ):
            return SemanticLiteralExpr(value=expr.value.value, type=SemanticMetaType(kind="string"))
        if (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "order_mode"
            and isinstance(expr.binding.value, OrderMode)
        ):
            return SemanticLiteralExpr(value=expr.binding.value.value, type=SemanticMetaType(kind="string"))
        order = self._require_string_expr(expr, context)
        if order != OrderMode.ASC.value:
            raise TypeError("pto.vci currently only supports order `OrderMode.ASC` in TileLang DSL v1")
        return SemanticLiteralExpr(value=order, type=SemanticMetaType(kind="string"))

    def _normalize_vcvt_round_mode(self, expr: SemanticExpr | None) -> SemanticExpr | None:
        if expr is None:
            return None
        if (
            isinstance(expr, SemanticSymbolExpr)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "vcvt_round_mode"
            and isinstance(expr.value, VcvtRoundMode)
        ):
            return SemanticLiteralExpr(value=expr.value.value, type=SemanticMetaType(kind="string"))
        if (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "vcvt_round_mode"
            and isinstance(expr.binding.value, VcvtRoundMode)
        ):
            return SemanticLiteralExpr(value=expr.binding.value.value, type=SemanticMetaType(kind="string"))
        round_mode = self._require_string_expr(expr, "pto.vcvt rnd")
        if round_mode not in {mode.value for mode in VcvtRoundMode}:
            raise TypeError(
                "pto.vcvt rnd must be a VcvtRoundMode enum such as "
                "`pto.VcvtRoundMode.R` or one of the canonical strings "
                '`"R"`, `"A"`, `"F"`, `"C"`, `"Z"`, `"O"` in TileLang DSL v1'
            )
        return SemanticLiteralExpr(value=round_mode, type=SemanticMetaType(kind="string"))

    def _normalize_vcvt_sat_mode(self, expr: SemanticExpr | None) -> SemanticExpr | None:
        if expr is None:
            return None
        if (
            isinstance(expr, SemanticSymbolExpr)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "vcvt_sat_mode"
            and isinstance(expr.value, VcvtSatMode)
        ):
            return SemanticLiteralExpr(value=expr.value.value, type=SemanticMetaType(kind="string"))
        if (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "vcvt_sat_mode"
            and isinstance(expr.binding.value, VcvtSatMode)
        ):
            return SemanticLiteralExpr(value=expr.binding.value.value, type=SemanticMetaType(kind="string"))
        sat_mode = self._require_string_expr(expr, "pto.vcvt sat")
        if sat_mode not in {mode.value for mode in VcvtSatMode}:
            raise TypeError(
                "pto.vcvt sat must be a VcvtSatMode enum such as "
                "`pto.VcvtSatMode.SAT` or `pto.VcvtSatMode.NOSAT`, or one of the "
                'canonical strings `"SAT"` / `"NOSAT"` in TileLang DSL v1'
            )
        return SemanticLiteralExpr(value=sat_mode, type=SemanticMetaType(kind="string"))

    def _normalize_vcvt_part_mode(self, expr: SemanticExpr | None) -> SemanticExpr | None:
        if expr is None:
            return None
        if (
            isinstance(expr, SemanticSymbolExpr)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "vcvt_part_mode"
            and isinstance(expr.value, VcvtPartMode)
        ):
            return SemanticLiteralExpr(value=expr.value.value, type=SemanticMetaType(kind="string"))
        if (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "vcvt_part_mode"
            and isinstance(expr.binding.value, VcvtPartMode)
        ):
            return SemanticLiteralExpr(value=expr.binding.value.value, type=SemanticMetaType(kind="string"))
        part_mode = self._require_string_expr(expr, "pto.vcvt part")
        if part_mode not in {mode.value for mode in VcvtPartMode}:
            raise TypeError(
                "pto.vcvt part must be a VcvtPartMode enum such as "
                "`pto.VcvtPartMode.EVEN` or `pto.VcvtPartMode.ODD`, or one of the "
                'canonical strings `"EVEN"` / `"ODD"` in TileLang DSL v1'
            )
        return SemanticLiteralExpr(value=part_mode, type=SemanticMetaType(kind="string"))

    def _lookup_vcvt_attr_contract(
        self, src_dtype: ScalarType, dst_dtype: ScalarType
    ) -> tuple[bool, bool, bool] | None:
        src_kind = _classify_vcvt_elem_kind(src_dtype)
        dst_kind = _classify_vcvt_elem_kind(dst_dtype)
        if src_kind is None or dst_kind is None:
            return None
        return _VCVT_ATTR_CONTRACTS.get((src_kind, dst_kind))

    def _require_explicit_vcvt_attrs(
        self,
        *,
        src_dtype: ScalarType,
        dst_dtype: ScalarType,
        rnd_required: bool,
        sat_required: bool,
        part_required: bool,
        rnd_explicit: bool,
        sat_explicit: bool,
        part_explicit: bool,
    ) -> None:
        pair = f"{src_dtype.name}->{dst_dtype.name}"

        def _check(attr_name: str, required: bool, explicit: bool) -> None:
            if required and not explicit:
                raise TypeError(
                    f"pto.vcvt {pair} requires explicit `{attr_name}=` in TileLang DSL v1"
                )
            if not required and explicit:
                raise TypeError(
                    f"pto.vcvt {pair} does not accept `{attr_name}=` for this type pair in TileLang DSL v1"
                )

        _check("rnd", rnd_required, rnd_explicit)
        _check("sat", sat_required, sat_explicit)
        _check("part", part_required, part_explicit)

    def _require_mask_expr(self, expr: SemanticExpr, context: str) -> SemanticMaskType:
        if not isinstance(expr.type, SemanticMaskType):
            raise TypeError(f"{context} must be a mask value in TileLang DSL")
        return expr.type

    def _require_align_expr(self, expr: SemanticExpr, context: str) -> None:
        if not isinstance(expr.type, SemanticAlignType):
            raise TypeError(f"{context} must be a pto.align value in TileLang DSL v1")

    def _require_matching_mask_types(
        self,
        lhs: SemanticMaskType,
        rhs: SemanticMaskType,
        context: str,
    ) -> None:
        if lhs != rhs:
            raise TypeError(f"{context} requires all mask operands to use the same mask granularity")

    def _require_string_expr(self, expr: SemanticExpr, context: str) -> str:
        if isinstance(expr, SemanticLiteralExpr) and isinstance(expr.type, SemanticMetaType) and expr.type.kind == "string":
            return expr.value
        if (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "string"
            and isinstance(expr.binding.value, str)
        ):
            return expr.binding.value
        raise TypeError(f"{context} must be a string literal in TileLang DSL")

    def _normalize_post_update_mode(
        self,
        expr: SemanticExpr | None,
        context: str,
    ) -> SemanticExpr:
        if expr is None:
            return SemanticLiteralExpr(value="NO_POST_UPDATE", type=SemanticMetaType(kind="string"))
        if (
            isinstance(expr, SemanticSymbolExpr)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "post_update_mode"
            and isinstance(expr.value, PostUpdateMode)
        ):
            return SemanticLiteralExpr(value=expr.value.value, type=SemanticMetaType(kind="string"))
        if (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "post_update_mode"
            and isinstance(expr.binding.value, PostUpdateMode)
        ):
            return SemanticLiteralExpr(value=expr.binding.value.value, type=SemanticMetaType(kind="string"))
        raise TypeError(
            "pto.vstur mode must be a PostUpdateMode enum such as "
            "`pto.PostUpdateMode.NO_POST_UPDATE` or `pto.PostUpdateMode.POST_UPDATE` in TileLang DSL v1"
        )

    def _normalize_predicate_store_dist(
        self,
        expr: SemanticExpr | None,
        context: str,
    ) -> SemanticExpr:
        if expr is None:
            return SemanticLiteralExpr(value="NORM", type=SemanticMetaType(kind="string"))
        dist = self._require_string_expr(expr, context)
        if dist not in {"NORM", "PK"}:
            raise TypeError("predicate store dist must be \"NORM\" or \"PK\" in TileLang DSL v1")
        return SemanticLiteralExpr(value=dist, type=SemanticMetaType(kind="string"))

    def _require_i1_expr(self, expr: SemanticExpr, context: str) -> None:
        scalar = self._require_scalar_expr(expr, context)
        if scalar.dtype != i1:
            raise TypeError(f"{context} must be an i1 value in TileLang DSL")

    def _require_i32_like_expr(self, expr: SemanticExpr, context: str) -> None:
        if isinstance(expr.type, SemanticIndexType):
            return
        scalar = self._require_scalar_expr(expr, context)
        if scalar.dtype != i32:
            raise TypeError(f"{context} must be an i32 or index value in TileLang DSL")

    def _require_i64_like_expr(self, expr: SemanticExpr, context: str) -> None:
        if isinstance(expr.type, SemanticIndexType):
            return
        scalar = self._require_scalar_expr(expr, context)
        if scalar.dtype != i64:
            raise TypeError(f"{context} must be an i64 or index value in TileLang DSL")

    def _require_tail_remaining_expr(self, expr: SemanticExpr, context: str) -> None:
        if isinstance(expr.type, SemanticIndexType):
            return
        if isinstance(expr.type, SemanticScalarType) and expr.type.dtype.name == "i32":
            return
        raise TypeError(f"{context} must be an i32 or index value in TileLang DSL v1")

    def _require_mask_for_vreg(
        self,
        mask_expr: SemanticExpr,
        vreg_type: SemanticVRegType,
        context: str,
    ) -> None:
        if not isinstance(mask_expr.type, SemanticMaskType):
            raise TypeError(f"{context} requires a mask operand in TileLang DSL v1")
        expected = self._mask_granularity_for_dtype(vreg_type.element_dtype)
        if mask_expr.type.granularity != expected:
            raise TypeError(
                f"{context} requires mask granularity {expected} for vector dtype {vreg_type.element_dtype!r}"
            )

    def _require_matching_vector_pointer(
        self,
        vreg_type: SemanticVRegType,
        pointer_type: SemanticType,
        context: str,
    ) -> None:
        if isinstance(pointer_type, SemanticTileType):
            if pointer_type.element_dtype != vreg_type.element_dtype:
                raise TypeError(f"{context} requires destination Tile dtype to match vector dtype")
            return
        if isinstance(pointer_type, SemanticPtrType):
            if pointer_type.memory_space != "ub":
                raise TypeError(f"{context} requires a UB pointer destination in TileLang DSL")
            if pointer_type.element_dtype != vreg_type.element_dtype:
                raise TypeError(f"{context} requires destination pointer dtype to match vector dtype")
            return
        raise TypeError(f"{context} requires a Tile or pointer destination in TileLang DSL")

    def _normalize_vldsx2_dist(self, expr: SemanticExpr) -> SemanticExpr:
        if (
            isinstance(expr, SemanticSymbolExpr)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "deinterleave_dist"
            and isinstance(expr.value, DeinterleaveDist)
        ):
            dist = expr.value.value
        elif (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "deinterleave_dist"
            and isinstance(expr.binding.value, DeinterleaveDist)
        ):
            dist = expr.binding.value.value
        else:
            dist = self._require_string_expr(expr, "pto.vldsx2 dist")
        legacy_map = {
            "DINTLV_B8": "DINTLV",
            "DINTLV_B16": "DINTLV",
            "DINTLV_B32": "DINTLV",
            "BD": "BDINTLV",
        }
        normalized = legacy_map.get(dist, dist)
        if normalized not in {"DINTLV", "BDINTLV"}:
            raise TypeError(
                "pto.vldsx2 dist must be one of \"DINTLV\" or \"BDINTLV\" in TileLang DSL v1"
            )
        return SemanticLiteralExpr(value=normalized, type=SemanticMetaType(kind="string"))

    def _normalize_vstsx2_dist(self, expr: SemanticExpr) -> SemanticExpr:
        if (
            isinstance(expr, SemanticSymbolExpr)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "interleave_dist"
            and isinstance(expr.value, InterleaveDist)
        ):
            dist = expr.value.value
        elif (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "interleave_dist"
            and isinstance(expr.binding.value, InterleaveDist)
        ):
            dist = expr.binding.value.value
        else:
            dist = self._require_string_expr(expr, "pto.vstsx2 dist")
        legacy_map = {
            "INTLV_B8": "INTLV",
            "INTLV_B16": "INTLV",
            "INTLV_B32": "INTLV",
        }
        normalized = legacy_map.get(dist, dist)
        if normalized != "INTLV":
            raise TypeError("pto.vstsx2 dist must be \"INTLV\" in TileLang DSL v1")
        return SemanticLiteralExpr(value=normalized, type=SemanticMetaType(kind="string"))

    def _mask_granularity_for_dtype(self, dtype: ScalarType) -> str:
        int_bits = integer_bitwidth(dtype)
        if dtype.name == "f32" or int_bits == 32:
            return "b32"
        if dtype.name in {"f16", "bf16"} or int_bits == 16:
            return "b16"
        if int_bits == 8:
            return "b8"
        raise TypeError(f"dtype `{dtype.name}` is not supported by make_mask/vector lowering in TileLang DSL v1")

    def _vreg_type_for_dtype(self, dtype: ScalarType) -> SemanticVRegType:
        width = bytewidth(dtype)
        if width not in {1, 2, 4}:
            raise TypeError(f"dtype `{dtype.name}` is not supported by vlds/vsts in TileLang DSL v1")
        return SemanticVRegType(element_dtype=dtype, lanes=256 // width)

    def _validate_unary_dtype(self, name: str, dtype: ScalarType) -> None:
        if name in {"vexp", "vln", "vsqrt", "vrec", "vrsqrt", "vexpdiff"} and dtype.name not in {"f16", "f32"}:
            raise TypeError(f"pto.{name} only supports f16/f32 in TileLang DSL v1")
        if name == "vrelu" and dtype.name not in {"f16", "f32"}:
            raise TypeError("pto.vrelu only supports f16/f32 in TileLang DSL v1")
        if name in {"vnot", "vbcnt", "vcls", "vsunpack", "vzunpack", "vusqz", "vsqz"} and not (
            is_integer_dtype(dtype) and integer_bitwidth(dtype) in {8, 16, 32}
        ):
            raise TypeError(f"pto.{name} only supports integer vector dtypes in TileLang DSL v1")
        if name in {"vabs", "vneg", "vmov", "vtrc", "vcadd", "vcmax", "vcmin"} and not (
            (is_integer_dtype(dtype) and integer_bitwidth(dtype) in {8, 16, 32}) or dtype.name in {"f16", "bf16", "f32"}
        ):
            raise TypeError(f"pto.{name} does not support this dtype in TileLang DSL v1")

    def _validate_binary_dtype(self, name: str, dtype: ScalarType) -> None:
        if name == "vdiv" and dtype.name not in {"f16", "f32"}:
            raise TypeError("pto.vdiv only supports f16/f32 in TileLang DSL v1")
        if name == "vprelu" and dtype.name not in {"f16", "f32"}:
            raise TypeError("pto.vprelu only supports f16/f32 in TileLang DSL v1")
        if name in {"vaddreluconv", "vmulconv"} and dtype.name not in {"f16", "bf16", "f32"}:
            raise TypeError(f"pto.{name} only supports f16/bf16/f32 in TileLang DSL v1")
        if name in {"vand", "vor", "vxor"} and not (
            is_integer_dtype(dtype) and integer_bitwidth(dtype) in {8, 16, 32}
        ):
            raise TypeError(f"pto.{name} only supports integer vector dtypes in TileLang DSL v1")
        if name in {"vshl", "vshr"} and not (
            is_integer_dtype(dtype) and integer_bitwidth(dtype) in {8, 16, 32}
        ):
            raise TypeError(f"pto.{name} only supports integer vector dtypes in TileLang DSL v1")
        if name == "vmul" and not (
            (is_integer_dtype(dtype) and integer_bitwidth(dtype) in {16, 32}) or dtype.name in {"f16", "f32"}
        ):
            raise TypeError("pto.vmul only supports 16/32-bit integer families and f16/f32 in TileLang DSL v1")
        if name == "vperm" and not (
            (is_integer_dtype(dtype) and integer_bitwidth(dtype) in {8, 16, 32}) or dtype.name in {"f16", "bf16", "f32"}
        ):
            raise TypeError("pto.vperm does not support this data vector dtype in TileLang DSL v1")
        if name in {"vadd", "vsub", "vmax", "vmin", "vaddrelu", "vsubrelu"} and not (
            (is_integer_dtype(dtype) and integer_bitwidth(dtype) in {8, 16, 32}) or dtype.name in {"f16", "bf16", "f32"}
        ):
            raise TypeError(f"pto.{name} does not support this dtype in TileLang DSL v1")
        if name in {"vpack", "vmrgsort"} and not (
            (is_integer_dtype(dtype) and integer_bitwidth(dtype) in {8, 16, 32}) or dtype.name in {"f16", "bf16", "f32"}
        ):
            raise TypeError(f"pto.{name} does not support this dtype in TileLang DSL v1")

    def _validate_vector_scalar_dtype(self, name: str, dtype: ScalarType) -> None:
        if name == "vdivs" and dtype.name not in {"f16", "f32"}:
            raise TypeError("pto.vdivs only supports f16/f32 in TileLang DSL v1")
        if name == "vlrelu" and dtype.name not in {"f16", "f32"}:
            raise TypeError("pto.vlrelu only supports f16/f32 in TileLang DSL v1")
        if name in {"vshls", "vshrs"} and not (
            is_integer_dtype(dtype) and integer_bitwidth(dtype) in {8, 16, 32}
        ):
            raise TypeError(f"pto.{name} only supports integer vector dtypes in TileLang DSL v1")
        if name in {"vands", "vors", "vxors"} and not (
            is_integer_dtype(dtype) and integer_bitwidth(dtype) in {8, 16, 32}
        ):
            raise TypeError(f"pto.{name} only supports integer vector dtypes in TileLang DSL v1")
        if name in {"vadds", "vsubs", "vmuls", "vmaxs", "vmins"} and not (
            (is_integer_dtype(dtype) and integer_bitwidth(dtype) in {8, 16, 32}) or dtype.name in {"f16", "bf16", "f32"}
        ):
            raise TypeError(f"pto.{name} does not support this dtype in TileLang DSL v1")

    def _validate_vector_immediate_dtype(self, name: str, dtype: ScalarType) -> None:
        if name in {"vshift", "vslide"} and not (
            (is_integer_dtype(dtype) and integer_bitwidth(dtype) in {8, 16, 32}) or dtype.name in {"f16", "bf16", "f32"}
        ):
            raise TypeError(f"pto.{name} does not support this vector dtype in TileLang DSL v1")

    def _validate_ternary_vector_dtype(self, name: str, dtype: ScalarType) -> None:
        if name == "vaxpy" and not (
            (is_integer_dtype(dtype) and integer_bitwidth(dtype) in {16, 32}) or dtype.name in {"f16", "f32"}
        ):
            raise TypeError("pto.vaxpy only supports 16/32-bit integer families and f16/f32 in TileLang DSL v1")
        if name == "vmula" and not (
            (is_integer_dtype(dtype) and integer_bitwidth(dtype) in {16, 32}) or dtype.name in {"f16", "f32"}
        ):
            raise TypeError("pto.vmula only supports 16/32-bit integer families and f16/f32 in TileLang DSL v1")

    def _validate_multi_result_vector_dtype(self, name: str, dtype: ScalarType) -> None:
        if name == "vmull" and not (is_integer_dtype(dtype) and integer_bitwidth(dtype) == 32):
            raise TypeError("pto.vmull only supports 32-bit integer vector families in TileLang DSL v1")

    def _require_sync_pipe(self, expr: SemanticExpr, context: str) -> str:
        if isinstance(expr, SemanticSymbolExpr) and expr.type.kind == "pipe":
            return expr.value.value
        if isinstance(expr, SemanticLiteralExpr) and isinstance(expr.type, SemanticMetaType) and expr.type.kind == "string":
            return expr.value
        raise TypeError(f"{context} must be a PIPE symbol or pipe string in TileLang DSL v1")

    def _require_sync_event(self, expr: SemanticExpr, context: str) -> str:
        if isinstance(expr, SemanticSymbolExpr) and expr.type.kind == "event":
            return expr.value.value
        if isinstance(expr, SemanticLiteralExpr) and isinstance(expr.type, SemanticMetaType) and expr.type.kind == "string":
            return expr.value
        raise TypeError(f"{context} must be an EVENT symbol or event string in TileLang DSL v1")

    def _require_barrier_type(self, expr: SemanticExpr, context: str) -> str:
        if isinstance(expr, SemanticSymbolExpr) and expr.type.kind == "barrier_type":
            return expr.value.value
        if isinstance(expr, SemanticBindingRef) and isinstance(expr.type, SemanticMetaType):
            if expr.type.kind == "barrier_type" and isinstance(expr.binding.value, BarrierType):
                return expr.binding.value.value
        if isinstance(expr, SemanticLiteralExpr) and isinstance(expr.type, SemanticMetaType) and expr.type.kind == "string":
            return expr.value
        raise TypeError(f"{context} must be a BarrierType symbol or string literal in TileLang DSL v1")

    def _normalize_event_id_expr(self, expr: SemanticExpr, context: str) -> SemanticExpr:
        if isinstance(expr, SemanticSymbolExpr) and expr.type.kind == "event" and isinstance(expr.value, Event):
            return SemanticLiteralExpr(
                value=int(expr.value.name[2:]),
                type=SemanticScalarType(dtype=i64),
            )
        if isinstance(expr, SemanticBindingRef) and isinstance(expr.type, SemanticMetaType):
            if expr.type.kind == "event" and isinstance(expr.binding.value, Event):
                return SemanticLiteralExpr(
                    value=int(expr.binding.value.name[2:]),
                    type=SemanticScalarType(dtype=i64),
                )
        self._require_i64_like_expr(expr, context)
        return expr

    def _pad_mode_value(
        self,
        expr: SemanticExpr | None,
        *,
        default: PadMode,
    ) -> PadMode:
        if expr is None:
            return default
        if isinstance(expr, SemanticSymbolExpr) and expr.type.kind == "pad_mode":
            return expr.value
        if (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "pad_mode"
            and isinstance(expr.binding.value, PadMode)
        ):
            return expr.binding.value
        raise TypeError("DMA pad_mode must be a PadMode symbol in TileLang DSL v1")

    def _require_loop_bound_type(self, ty: SemanticType) -> None:
        if isinstance(ty, (SemanticIndexType, SemanticScalarType)):
            return
        raise TypeError(f"loop bound must be scalar/index typed, got {ty!r}")

    def _require_condition_type(self, ty: SemanticType) -> None:
        if isinstance(ty, SemanticIndexType):
            return
        if isinstance(ty, SemanticScalarType):
            return
        raise TypeError(f"if condition must be scalar/index typed, got {ty!r}")

    def _merge_loop_carried_types(
        self,
        outer_type: SemanticType,
        final_type: SemanticType,
    ) -> SemanticType | None:
        if final_type == outer_type:
            return outer_type
        if (
            isinstance(outer_type, SemanticIndexType)
            and isinstance(final_type, SemanticScalarType)
            and final_type.dtype == i32
        ):
            return final_type
        if (
            isinstance(final_type, SemanticIndexType)
            and isinstance(outer_type, SemanticScalarType)
            and outer_type.dtype == i32
        ):
            return outer_type
        return None

    def _require_index_typed_expr(self, expr: SemanticExpr) -> None:
        if not isinstance(expr.type, SemanticIndexType):
            self._raise_expr_type_error(
                "slice bounds and vector offsets must be index-typed in TileLang DSL v1",
                expr,
            )

    def _try_static_dtype(self, expr: SemanticExpr) -> ScalarType | None:
        if (
            isinstance(expr, SemanticSymbolExpr)
            and expr.type.kind == "dtype"
            and isinstance(expr.value, ScalarType)
        ):
            return expr.value
        if (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticMetaType)
            and expr.type.kind == "dtype"
            and isinstance(expr.binding.value, ScalarType)
        ):
            return expr.binding.value
        return None

    def _try_static_subscript_value(self, expr: SemanticSubscriptAccess) -> Any | None:
        index_value = self._try_static_value(expr.index)
        if not isinstance(index_value, int):
            return None

        base = expr.base
        if isinstance(base, SemanticAttributeAccess) and isinstance(base.base, SemanticBindingRef):
            binding_ref = base.base
            binding_type = binding_ref.type
            if isinstance(binding_type, SemanticTileType):
                if base.attr == "shape" and binding_type.shape is not None:
                    if 0 <= index_value < len(binding_type.shape):
                        return binding_type.shape[index_value]
                if base.attr == "valid_shape" and binding_type.valid_shape is not None:
                    if 0 <= index_value < len(binding_type.valid_shape):
                        return binding_type.valid_shape[index_value]
                return None
            if isinstance(binding_type, (SemanticTensorViewType, SemanticPartitionTensorViewType)):
                return None

        base_value = self._try_static_value(base)
        if isinstance(base_value, (tuple, list)):
            if 0 <= index_value < len(base_value):
                return base_value[index_value]
            return None
        return None

    def _try_static_value(self, expr: SemanticExpr | None) -> Any | None:
        if expr is None:
            return None
        if isinstance(expr, SemanticSymbolExpr):
            return expr.value
        if isinstance(expr, SemanticLiteralExpr):
            return expr.value
        if isinstance(expr, SemanticBindingRef):
            return expr.binding.value
        if isinstance(expr, SemanticTupleExpr):
            elements = []
            for element in expr.elements:
                static_element = self._try_static_value(element)
                if static_element is None:
                    return None
                elements.append(static_element)
            return tuple(elements)
        if isinstance(expr, SemanticSubscriptAccess):
            return self._try_static_subscript_value(expr)
        if isinstance(expr, SemanticBinaryExpr):
            if expr.op in {"and", "or"}:
                lhs_bool = self._try_static_condition_bool(expr.lhs)
                rhs_bool = self._try_static_condition_bool(expr.rhs)
                if lhs_bool is None or rhs_bool is None:
                    return None
                if expr.op == "and":
                    return lhs_bool and rhs_bool
                return lhs_bool or rhs_bool
            lhs = self._try_static_value(expr.lhs)
            rhs = self._try_static_value(expr.rhs)
            if lhs is None or rhs is None:
                return None
            if expr.op == "add":
                if (
                    isinstance(lhs, (int, float))
                    and isinstance(rhs, (int, float))
                    and not isinstance(lhs, bool)
                    and not isinstance(rhs, bool)
                ):
                    return lhs + rhs
                return None
            if expr.op == "sub":
                if (
                    isinstance(lhs, (int, float))
                    and isinstance(rhs, (int, float))
                    and not isinstance(lhs, bool)
                    and not isinstance(rhs, bool)
                ):
                    return lhs - rhs
                return None
            if expr.op == "mul":
                if (
                    isinstance(lhs, (int, float))
                    and isinstance(rhs, (int, float))
                    and not isinstance(lhs, bool)
                    and not isinstance(rhs, bool)
                ):
                    return lhs * rhs
                return None
            if expr.op == "mod":
                if isinstance(lhs, int) and isinstance(rhs, int):
                    if rhs == 0:
                        return None
                    return lhs % rhs
                return None
            if expr.op == "floordiv":
                if isinstance(lhs, int) and isinstance(rhs, int):
                    if rhs == 0:
                        return None
                    return lhs // rhs
                return None
            if expr.op == "bitand":
                if isinstance(lhs, int) and isinstance(rhs, int):
                    if isinstance(lhs, bool) or isinstance(rhs, bool):
                        return None
                    return lhs & rhs
                return None
            if expr.op == "bitor":
                if isinstance(lhs, int) and isinstance(rhs, int):
                    if isinstance(lhs, bool) or isinstance(rhs, bool):
                        return None
                    return lhs | rhs
                return None
            if expr.op == "bitxor":
                if isinstance(lhs, int) and isinstance(rhs, int):
                    if isinstance(lhs, bool) or isinstance(rhs, bool):
                        return None
                    return lhs ^ rhs
                return None
            if expr.op == "lshift":
                if isinstance(lhs, int) and isinstance(rhs, int):
                    if isinstance(lhs, bool) or isinstance(rhs, bool) or rhs < 0:
                        return None
                    return lhs << rhs
                return None
            if expr.op == "rshift":
                if isinstance(lhs, int) and isinstance(rhs, int):
                    if isinstance(lhs, bool) or isinstance(rhs, bool) or rhs < 0:
                        return None
                    return lhs >> rhs
                return None
            if expr.op == "eq":
                return lhs == rhs
            if expr.op == "ne":
                return lhs != rhs
            if expr.op == "gt":
                try:
                    return lhs > rhs
                except TypeError:
                    return None
            if expr.op == "lt":
                try:
                    return lhs < rhs
                except TypeError:
                    return None
            if expr.op == "ge":
                try:
                    return lhs >= rhs
                except TypeError:
                    return None
            if expr.op == "le":
                try:
                    return lhs <= rhs
                except TypeError:
                    return None
            return None
        if isinstance(expr, SemanticCallExpr):
            if expr.namespace != "pto":
                return None
            if expr.name == "bytewidth":
                if len(expr.args) != 1:
                    return None
                dtype = self._try_static_dtype(expr.args[0])
                if dtype is None:
                    return None
                return bytewidth(dtype)
            if expr.name in {"get_lanes", "elements_per_vreg"}:
                if len(expr.args) != 1:
                    return None
                dtype = self._try_static_dtype(expr.args[0])
                if dtype is None:
                    return None
                return self._vreg_type_for_dtype(dtype).lanes
        return None

    def _try_static_condition_bool(self, expr: SemanticExpr | None) -> bool | None:
        value = self._try_static_value(expr)
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value != 0
        return None

    def _require_constexpr_condition_bool(
        self,
        expr: SemanticExpr,
        *,
        context: str,
    ) -> bool:
        value = self._try_static_condition_bool(expr)
        if value is None:
            raise TypeError(
                f"{context} must be a compile-time bool in TileLang DSL v1"
            )
        return value

    def _static_index_value(self, expr: SemanticExpr | None, *, default: int | None) -> int | None:
        if expr is None:
            return default
        value = self._try_static_value(expr)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        return None

    def _require_optional_index_typed_expr(self, expr: SemanticExpr | None) -> None:
        if expr is None:
            return
        self._require_index_typed_expr(expr)

    def _static_bool_value(self, expr: SemanticExpr | None, *, default: bool | None) -> bool | None:
        if expr is None:
            return default
        if isinstance(expr, SemanticLiteralExpr):
            if (
                isinstance(expr.type, SemanticScalarType)
                and expr.type.dtype == i1
                and isinstance(expr.value, bool)
            ):
                return expr.value
            return None
        if (
            isinstance(expr, SemanticBindingRef)
            and isinstance(expr.type, SemanticScalarType)
            and expr.type.dtype == i1
            and isinstance(expr.binding.value, bool)
        ):
            return expr.binding.value
        return None

    def _require_static_bool_value(
        self,
        expr: SemanticExpr | None,
        *,
        context: str,
        default: bool,
    ) -> bool:
        value = self._static_bool_value(expr, default=default)
        if value is None:
            raise TypeError(
                f"{context} must be a compile-time bool in the stable frontend-only DMA profile"
            )
        return value

    def _require_static_non_negative_index_value(
        self,
        expr: SemanticExpr | None,
        *,
        context: str,
        default: int,
    ) -> int:
        value = self._static_index_value(expr, default=default)
        if value is None:
            raise TypeError(
                f"{context} must be a static non-negative index in the stable frontend-only DMA profile"
            )
        if value < 0:
            raise TypeError(
                f"{context} must be a non-negative index in the stable frontend-only DMA profile"
            )
        return value

    def _normalize_optional_index_expr(
        self,
        expr: SemanticExpr | None,
        *,
        default: int,
    ) -> SemanticExpr:
        if expr is not None:
            return expr
        return SemanticLiteralExpr(value=default, type=SemanticIndexType())

    def _normalized_tensor_slice_extent(self, expr: SemanticSliceExpr) -> int | None:
        start = self._static_index_value(expr.start, default=0)
        stop = self._static_index_value(expr.stop, default=None)
        step = self._static_index_value(expr.step, default=1)
        if stop is None or start is None or step is None:
            return None
        if step <= 0:
            raise TypeError("TensorView slicing requires a positive static step in TileLang DSL v1")
        distance = stop - start
        if distance <= 0:
            raise TypeError("TensorView slicing requires positive extents in TileLang DSL v1")
        return (distance + step - 1) // step


def analyze_frontend_kernel(node: FrontendKernelNode) -> SemanticKernel:
    """Normalize descriptor-owned AST into a lowering semantic model."""

    return _SemanticAnalyzer(node).analyze()


__all__ = [
    "SemanticAssignStmt",
    "SemanticAttributeAccess",
    "SemanticBinaryExpr",
    "SemanticBinding",
    "SemanticBindingRef",
    "SemanticCallExpr",
    "SemanticDmaOptions",
    "SemanticDmaLoadStmt",
    "SemanticDmaStoreStmt",
    "SemanticExpr",
    "SemanticExprStmt",
    "SemanticForStmt",
    "SemanticGetBufStmt",
    "SemanticAlignStoreStmt",
    "SemanticAlignType",
    "SemanticIfResult",
    "SemanticIfStmt",
    "SemanticIndexType",
    "SemanticKernel",
    "SemanticLiteralExpr",
    "SemanticMemBarStmt",
    "SemanticMaskType",
    "SemanticPadValueType",
    "SemanticParameter",
    "SemanticPipeBarrierStmt",
    "SemanticPredicateStoreStmt",
    "SemanticRlsBufStmt",
    "SemanticReturnStmt",
    "SemanticScalarType",
    "SemanticSetCrossCoreStmt",
    "SemanticSetFlagStmt",
    "SemanticSetIntraBlockStmt",
    "SemanticSetIntraCoreStmt",
    "SemanticShapeType",
    "SemanticSliceExpr",
    "SemanticSliceType",
    "SemanticStmt",
    "SemanticVecscopeStmt",
    "SemanticStrictVecscopeStmt",
    "SemanticSubscriptAccess",
    "SemanticSymbolExpr",
    "SemanticTensorSliceAxis",
    "SemanticTensorSliceExpr",
    "SemanticTensorSliceType",
    "SemanticTensorViewType",
    "SemanticPartitionTensorViewType",
    "SemanticTileBinding",
    "SemanticTileConfigType",
    "SemanticTileType",
    "SemanticTupleExpr",
    "SemanticTupleType",
    "SemanticType",
    "SemanticVRegType",
    "SemanticVectorPairStoreStmt",
    "SemanticVectorStoreStmt",
    "SemanticWaitFlagDevStmt",
    "SemanticWaitFlagStmt",
    "SemanticWaitIntraCoreStmt",
    "analyze_frontend_kernel",
]
