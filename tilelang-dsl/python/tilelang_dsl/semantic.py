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
    FrontendReturnStmt,
    FrontendSliceExpr,
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
    BLayout,
    Event,
    MaskType,
    MaskPattern,
    MemorySpace,
    OrderMode,
    PadValue,
    PadMode,
    Pipe,
    PositionMode,
    PointerType,
    ScalarType,
    SLayout,
    TileConfig,
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
)


_DTYPE_SYMBOLS = {
    "i1": i1,
    "i8": i8,
    "i16": i16,
    "i32": i32,
    "i64": i64,
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
_MEMORY_SPACE_SYMBOLS = {memory_space.name: memory_space for memory_space in MemorySpace}
_B_LAYOUT_SYMBOLS = {layout.name: layout for layout in BLayout}
_S_LAYOUT_SYMBOLS = {layout.name: layout for layout in SLayout}
_PAD_VALUE_SYMBOLS = {pad_value.name: pad_value for pad_value in PadValue}
_PAD_MODE_SYMBOLS = {pad_mode.name: pad_mode for pad_mode in PadMode}
_POSITION_MODE_SYMBOLS = {position_mode.name: position_mode for position_mode in PositionMode}
_ORDER_MODE_SYMBOLS = {order_mode.name: order_mode for order_mode in OrderMode}
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
    "vbitsort",
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
_MULTI_RESULT_VECTOR_OPS = {"vmull"}
_BROADCAST_VECTOR_OPS = {"vbr", "vdup", "vci"}
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
_COMPARE_SELECT_OPS = {"vcmp", "vcmps", "vsel", "vselr", "vselrv2"}
_PREDICATE_MOVEMENT_OPS = {"pnot", "psel", "ppack", "punpack"}
_CARRY_OPS = {"vaddc", "vsubc", "vaddcs", "vsubcs"}
_REARRANGEMENT_OPS = {"vintlv", "vdintlv", "vintlvv2", "vdintlvv2"}
_ADVANCED_VECTOR_ACTIVITY_OPS = (
    _COMPARE_SELECT_OPS
    | _PREDICATE_MOVEMENT_OPS
    | _CARRY_OPS
    | _REARRANGEMENT_OPS
    | {"vcvt", "vmrgsort4"}
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
    config: TileConfig | None = None


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
    type: SemanticMetaType


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
class SemanticDmaConfigStmt(SemanticStmt):
    name: str
    first: SemanticExpr
    second: SemanticExpr


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
                config=None if spec is None else spec.config,
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
    ) -> tuple[tuple[SemanticStmt, ...], dict[str, SemanticBinding]]:
        current_env = dict(env)
        semantic_statements = []
        index = 0
        while index < len(statements):
            if self._should_infer_vecscope(statements[index], allow_outer_lookup=allow_outer_lookup):
                end = index + 1
                while end < len(statements) and self._should_infer_vecscope(
                    statements[end],
                    allow_outer_lookup=allow_outer_lookup,
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
        allow_outer_lookup: bool,
    ) -> bool:
        if self._has_explicit_vecscope:
            return False
        if self._disable_inference_depth > 0:
            return False
        if not allow_outer_lookup:
            return False
        if isinstance(stmt, FrontendForStmt):
            return self._block_can_live_in_inferred_vecscope(stmt.body)
        name = self._frontend_vector_call_name(stmt)
        return name in (
            {"make_mask", "vlds", "vsts"}
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
            and (self._is_dma_call(stmt.expr) or self._is_sync_call(stmt.expr))
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
        return isinstance(stmt, FrontendAssignStmt) or (
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
                {"make_mask", "vlds", "vsts"}
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
                    config=None,
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
            and expr.name == "vsts"
        )

    def _is_sync_call(self, expr: FrontendExprNode) -> bool:
        return (
            isinstance(expr, FrontendCallExpr)
            and expr.namespace == "pto"
            and expr.name in {"set_flag", "wait_flag", "pipe_barrier", "barrier"}
        )

    def _is_low_level_dma_call(self, expr: FrontendExprNode) -> bool:
        return (
            isinstance(expr, FrontendCallExpr)
            and expr.namespace == "pto"
            and expr.name in _LOW_LEVEL_DMA_CONFIG_OPS | _LOW_LEVEL_DMA_COPY_OPS
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
            annotated_type = self._annotation_type(annotation, value.type, env)
            binding = self._make_binding(
                target.name,
                annotated_type if annotated_type is not None else value.type,
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
            if path is not None and path[0] in {"pto", "PAT", "PIPE", "Pipe", "EVENT", "Event"} and len(path) >= 2:
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
        body, _ = self._analyze_block(
            stmt.body,
            scope_env,
            allow_outer_lookup=False,
        )
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
            return SemanticBindingRef(binding=binding, type=binding.type)
        if isinstance(expr, FrontendConstantExpr):
            if isinstance(expr.value, bool):
                return SemanticLiteralExpr(value=expr.value, type=SemanticScalarType(dtype=i1))
            if isinstance(expr.value, int):
                return SemanticLiteralExpr(value=expr.value, type=SemanticIndexType())
            if isinstance(expr.value, float):
                return SemanticLiteralExpr(
                    value=expr.value,
                    type=SemanticScalarType(dtype=f32),
                )
            if isinstance(expr.value, str):
                return SemanticLiteralExpr(
                    value=expr.value,
                    type=SemanticMetaType(kind="string"),
                )
            if expr.value is None:
                return SemanticLiteralExpr(value=None, type=SemanticIndexType())
            raise TypeError(f"unsupported constant {expr.value!r} in TileLang DSL v1")
        if isinstance(expr, FrontendSymbolExpr):
            return self._analyze_symbol_expr(expr)
        if isinstance(expr, FrontendSliceExpr):
            start = None if expr.start is None else self._analyze_expr(expr.start, env, allow_outer_lookup=allow_outer_lookup)
            stop = None if expr.stop is None else self._analyze_expr(expr.stop, env, allow_outer_lookup=allow_outer_lookup)
            step = None if expr.step is None else self._analyze_expr(expr.step, env, allow_outer_lookup=allow_outer_lookup)
            for item in (start, stop, step):
                if item is not None:
                    self._require_index_typed_expr(item)
            return SemanticSliceExpr(
                start=start,
                stop=stop,
                step=step,
                type=SemanticSliceType(),
            )
        if isinstance(expr, FrontendTupleExpr):
            elements = tuple(
                self._analyze_expr(element, env, allow_outer_lookup=allow_outer_lookup)
                for element in expr.elements
            )
            return SemanticTupleExpr(
                elements=elements,
                type=SemanticTupleType(elements=tuple(element.type for element in elements)),
            )
        if isinstance(expr, FrontendAttributeExpr):
            base = self._analyze_expr(expr.base, env, allow_outer_lookup=allow_outer_lookup)
            if expr.attr == "element_type":
                return self._element_type_expr(base)
            if expr.attr == "valid_shape":
                return self._valid_shape_expr(base)
            if expr.attr == "strides":
                return self._strides_expr(base)
            if expr.attr == "rank":
                return self._rank_expr(base)
            attr_type = self._attribute_type(base, expr.attr)
            return SemanticAttributeAccess(base=base, attr=expr.attr, type=attr_type)
        if isinstance(expr, FrontendSubscriptExpr):
            base = self._analyze_expr(expr.base, env, allow_outer_lookup=allow_outer_lookup)
            index = self._analyze_expr(expr.index, env, allow_outer_lookup=allow_outer_lookup)
            result_type = self._subscript_type(base, index)
            if isinstance(result_type, SemanticTensorSliceType):
                slices = self._normalize_tensor_slice(index, base.type.rank)
                return SemanticTensorSliceExpr(base=base, slices=slices, type=result_type)
            return SemanticSubscriptAccess(base=base, index=index, type=result_type)
        if isinstance(expr, FrontendBinaryExpr):
            lhs = self._analyze_expr(expr.lhs, env, allow_outer_lookup=allow_outer_lookup)
            rhs = self._analyze_expr(expr.rhs, env, allow_outer_lookup=allow_outer_lookup)
            result_type = self._binary_type(lhs, rhs, expr.op)
            return SemanticBinaryExpr(lhs=lhs, op=expr.op, rhs=rhs, type=result_type)
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
        if expr.namespace in {"PIPE", "Pipe", "pto.PIPE", "pto.Pipe"}:
            pipe = _PIPE_SYMBOLS.get(expr.name)
            if pipe is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=pipe,
                    type=SemanticMetaType(kind="pipe"),
                )
        if expr.namespace in {"EVENT", "Event", "pto.EVENT", "pto.Event"}:
            event = _EVENT_SYMBOLS.get(expr.name)
            if event is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=event,
                    type=SemanticMetaType(kind="event"),
                )
        if expr.namespace in {"pto.MemorySpace"}:
            memory_space = _MEMORY_SPACE_SYMBOLS.get(expr.name)
            if memory_space is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=memory_space,
                    type=SemanticMetaType(kind="memory_space"),
                )
        if expr.namespace in {"pto.BLayout"}:
            b_layout = _B_LAYOUT_SYMBOLS.get(expr.name)
            if b_layout is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=b_layout,
                    type=SemanticMetaType(kind="b_layout"),
                )
        if expr.namespace in {"pto.SLayout"}:
            s_layout = _S_LAYOUT_SYMBOLS.get(expr.name)
            if s_layout is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=s_layout,
                    type=SemanticMetaType(kind="s_layout"),
                )
        if expr.namespace in {"pto.PadValue"}:
            pad_value = _PAD_VALUE_SYMBOLS.get(expr.name)
            if pad_value is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=pad_value,
                    type=SemanticMetaType(kind="pad_value"),
                )
        if expr.namespace in {"pto.PadMode"}:
            pad_mode = _PAD_MODE_SYMBOLS.get(expr.name)
            if pad_mode is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=pad_mode,
                    type=SemanticMetaType(kind="pad_mode"),
                )
        if expr.namespace in {"pto.PositionMode"}:
            position_mode = _POSITION_MODE_SYMBOLS.get(expr.name)
            if position_mode is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=position_mode,
                    type=SemanticMetaType(kind="position_mode"),
                )
        if expr.namespace in {"pto.OrderMode"}:
            order_mode = _ORDER_MODE_SYMBOLS.get(expr.name)
            if order_mode is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=order_mode,
                    type=SemanticMetaType(kind="order_mode"),
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
        if isinstance(base_type, (SemanticTensorViewType, SemanticPartitionTensorViewType, SemanticTileType)) and attr == "rank":
            return SemanticIndexType()
        if isinstance(base_type, SemanticTileType) and attr == "memory_space":
            return SemanticMetaType(kind="memory_space")
        if isinstance(base_type, SemanticTileType) and attr == "config":
            return SemanticMetaType(kind="tile_config")
        if isinstance(base_type, SemanticMetaType) and base_type.kind == "tile_config":
            if attr == "b_layout":
                return SemanticMetaType(kind="b_layout")
            if attr == "s_layout":
                return SemanticMetaType(kind="s_layout")
            if attr == "s_fractal_size":
                return SemanticScalarType(dtype=i32)
            if attr == "pad_value":
                return SemanticMetaType(kind="pad_value")
        raise TypeError(f"unsupported attribute access '{attr}' in TileLang DSL v1")

    def _rank_expr(self, base: SemanticExpr) -> SemanticExpr:
        base_type = base.type
        if isinstance(base_type, (SemanticTensorViewType, SemanticPartitionTensorViewType, SemanticTileType)):
            return SemanticLiteralExpr(value=base_type.rank, type=SemanticIndexType())
        raise TypeError("unsupported attribute access 'rank' in TileLang DSL v1")

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

    def _static_shape_tuple_expr(self, values: tuple[int, ...]) -> SemanticTupleExpr:
        return SemanticTupleExpr(
            elements=tuple(
                SemanticLiteralExpr(value=value, type=SemanticIndexType())
                for value in values
            ),
            type=SemanticTupleType(elements=tuple(SemanticIndexType() for _ in values)),
        )

    def _subscript_type(self, base: SemanticExpr, index: SemanticExpr) -> SemanticType:
        if isinstance(base.type, SemanticShapeType):
            if not isinstance(index.type, SemanticIndexType):
                raise TypeError("shape subscript index must be an index value in TileLang DSL v1")
            if (
                isinstance(base, SemanticAttributeAccess)
                and isinstance(base.base, SemanticBindingRef)
                and isinstance(index, SemanticLiteralExpr)
                and isinstance(index.value, int)
            ):
                if index.value < 0 or index.value >= base.type.rank:
                    raise TypeError(
                        f"shape subscript index {index.value} is out of bounds for rank {base.type.rank}"
                    )
            return SemanticIndexType()
        if isinstance(base.type, SemanticTupleType):
            if not isinstance(index.type, SemanticIndexType):
                raise TypeError("tuple subscript index must be an index value in TileLang DSL v1")
            if isinstance(index, SemanticLiteralExpr) and isinstance(index.value, int):
                if index.value < 0 or index.value >= len(base.type.elements):
                    raise TypeError(
                        f"tuple subscript index {index.value} is out of bounds for arity {len(base.type.elements)}"
                    )
                return base.type.elements[index.value]
            raise TypeError("tuple subscript index must be a compile-time integer literal in TileLang DSL v1")
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
        if op in {"add", "sub", "mul", "floordiv"}:
            if isinstance(lhs.type, SemanticIndexType) and isinstance(rhs.type, SemanticIndexType):
                return SemanticIndexType()
            raise TypeError("binary expressions currently only support index-typed operands")
        if op in {"eq", "ne"}:
            if isinstance(lhs.type, SemanticIndexType) and isinstance(rhs.type, SemanticIndexType):
                return SemanticScalarType(dtype=i1)
            if isinstance(lhs.type, SemanticScalarType) and lhs.type == rhs.type:
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
        if name == "vlds":
            return self._analyze_vlds(args)
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
            elif target_dtype.name.startswith("i"):
                if isinstance(literal_value, bool):
                    casted = int(literal_value)
                elif isinstance(literal_value, (int, float)):
                    casted = int(literal_value)
                else:
                    casted = None
                if casted is not None:
                    bits = int(target_dtype.name[1:])
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
            if len(args) not in {1, 2}:
                raise TypeError("pto.vdup expects 1 or 2 positional arguments in TileLang DSL v1")
            value = args[0]
            if isinstance(value.type, SemanticVRegType):
                vec_type = value.type
            else:
                vec_type = self._vreg_type_for_scalar_or_index(value, "pto.vdup input")
            position_arg = args[1] if len(args) == 2 else None
            position = self._normalize_position_mode(position_arg, "pto.vdup position")
            return SemanticCallExpr(
                namespace="pto",
                name=name,
                args=(value, position),
                type=vec_type,
            )

        if name == "vci":
            if len(args) not in {1, 2}:
                raise TypeError("pto.vci expects 1 or 2 positional arguments in TileLang DSL v1")
            index = self._require_scalar_or_index_expr(args[0], "pto.vci index")
            index_dtype = i32 if isinstance(index.type, SemanticIndexType) else index.type.dtype
            if index_dtype.name not in {"i8", "i16", "i32"}:
                raise TypeError("pto.vci index only supports i8/i16/i32 in TileLang DSL v1")
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
            if rhs.element_dtype.name not in {"i8", "i16", "i32"}:
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
        if scalar.dtype != vreg.element_dtype:
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
        if isinstance(immediate.type, SemanticScalarType) and immediate.type.dtype.name not in {"i8", "i16", "i32"}:
            raise TypeError(f"pto.{name} immediate only supports i8/i16/i32 in TileLang DSL v1")
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

    def _analyze_vcvt(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if len(args) != 3:
            raise TypeError("pto.vcvt expects exactly 3 positional arguments in TileLang DSL")
        vector = self._require_vreg_expr(args[0], "pto.vcvt vector")
        target_dtype = self._require_dtype_symbol(args[1], "pto.vcvt to_type")
        self._require_mask_for_vreg(args[2], vector, "pto.vcvt")
        return SemanticCallExpr(
            namespace="pto",
            name="vcvt",
            args=args,
            type=self._vreg_type_for_dtype(target_dtype),
        )

    def _analyze_vmrgsort4(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if len(args) != 5:
            raise TypeError("pto.vmrgsort4 expects exactly 5 positional arguments in TileLang DSL")
        vec0 = self._require_vreg_expr(args[0], "pto.vmrgsort4 vec0")
        vec1 = self._require_vreg_expr(args[1], "pto.vmrgsort4 vec1")
        vec2 = self._require_vreg_expr(args[2], "pto.vmrgsort4 vec2")
        vec3 = self._require_vreg_expr(args[3], "pto.vmrgsort4 vec3")
        if not (vec0 == vec1 == vec2 == vec3):
            raise TypeError("pto.vmrgsort4 requires all vector operands to use the same vector type")
        self._require_mask_for_vreg(args[4], vec0, "pto.vmrgsort4")
        return SemanticCallExpr(namespace="pto", name="vmrgsort4", args=args, type=vec0)

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
        if position != PositionMode.LOWEST.value:
            raise TypeError(
                "pto.vdup currently only supports position `PositionMode.LOWEST` in TileLang DSL v1"
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

    def _require_mask_expr(self, expr: SemanticExpr, context: str) -> SemanticMaskType:
        if not isinstance(expr.type, SemanticMaskType):
            raise TypeError(f"{context} must be a mask value in TileLang DSL")
        return expr.type

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

    def _require_i1_expr(self, expr: SemanticExpr, context: str) -> None:
        scalar = self._require_scalar_expr(expr, context)
        if scalar.dtype != i1:
            raise TypeError(f"{context} must be an i1 value in TileLang DSL")

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

    def _mask_granularity_for_dtype(self, dtype: ScalarType) -> str:
        if dtype.name in {"f32", "i32"}:
            return "b32"
        if dtype.name in {"f16", "bf16", "i16"}:
            return "b16"
        if dtype.name == "i8":
            return "b8"
        raise TypeError(f"dtype `{dtype.name}` is not supported by make_mask/vector lowering in TileLang DSL v1")

    def _vreg_type_for_dtype(self, dtype: ScalarType) -> SemanticVRegType:
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
            raise TypeError(f"dtype `{dtype.name}` is not supported by vlds/vsts in TileLang DSL v1")
        return SemanticVRegType(element_dtype=dtype, lanes=256 // width)

    def _validate_unary_dtype(self, name: str, dtype: ScalarType) -> None:
        if name in {"vexp", "vln", "vsqrt", "vrec", "vrsqrt", "vexpdiff"} and dtype.name not in {"f16", "f32"}:
            raise TypeError(f"pto.{name} only supports f16/f32 in TileLang DSL v1")
        if name == "vrelu" and dtype.name not in {"f16", "f32"}:
            raise TypeError("pto.vrelu only supports f16/f32 in TileLang DSL v1")
        if name in {"vnot", "vbcnt", "vcls", "vsunpack", "vzunpack", "vusqz", "vsqz"} and dtype.name not in {"i8", "i16", "i32"}:
            raise TypeError(f"pto.{name} only supports integer vector dtypes in TileLang DSL v1")
        if name in {"vabs", "vneg", "vmov", "vtrc", "vbitsort", "vcadd", "vcmax", "vcmin"} and dtype.name not in {"i8", "i16", "i32", "f16", "bf16", "f32"}:
            raise TypeError(f"pto.{name} does not support this dtype in TileLang DSL v1")

    def _validate_binary_dtype(self, name: str, dtype: ScalarType) -> None:
        if name == "vdiv" and dtype.name not in {"f16", "f32"}:
            raise TypeError("pto.vdiv only supports f16/f32 in TileLang DSL v1")
        if name == "vprelu" and dtype.name not in {"f16", "f32"}:
            raise TypeError("pto.vprelu only supports f16/f32 in TileLang DSL v1")
        if name in {"vaddreluconv", "vmulconv"} and dtype.name not in {"f16", "bf16", "f32"}:
            raise TypeError(f"pto.{name} only supports f16/bf16/f32 in TileLang DSL v1")
        if name in {"vand", "vor", "vxor"} and dtype.name not in {"i8", "i16", "i32"}:
            raise TypeError(f"pto.{name} only supports integer vector dtypes in TileLang DSL v1")
        if name in {"vshl", "vshr"} and dtype.name not in {"i8", "i16", "i32"}:
            raise TypeError(f"pto.{name} only supports integer vector dtypes in TileLang DSL v1")
        if name == "vmul" and dtype.name not in {"i16", "i32", "f16", "f32"}:
            raise TypeError("pto.vmul only supports i16/i32/f16/f32 in TileLang DSL v1")
        if name == "vperm" and dtype.name not in {"i8", "i16", "i32", "f16", "bf16", "f32"}:
            raise TypeError("pto.vperm does not support this data vector dtype in TileLang DSL v1")
        if name in {"vadd", "vsub", "vmax", "vmin", "vaddrelu", "vsubrelu"} and dtype.name not in {"i8", "i16", "i32", "f16", "bf16", "f32"}:
            raise TypeError(f"pto.{name} does not support this dtype in TileLang DSL v1")
        if name in {"vpack", "vmrgsort"} and dtype.name not in {"i8", "i16", "i32", "f16", "bf16", "f32"}:
            raise TypeError(f"pto.{name} does not support this dtype in TileLang DSL v1")

    def _validate_vector_scalar_dtype(self, name: str, dtype: ScalarType) -> None:
        if name == "vdivs" and dtype.name not in {"f16", "f32"}:
            raise TypeError("pto.vdivs only supports f16/f32 in TileLang DSL v1")
        if name == "vlrelu" and dtype.name not in {"f16", "f32"}:
            raise TypeError("pto.vlrelu only supports f16/f32 in TileLang DSL v1")
        if name in {"vshls", "vshrs"} and dtype.name not in {"i8", "i16", "i32"}:
            raise TypeError(f"pto.{name} only supports integer vector dtypes in TileLang DSL v1")
        if name in {"vands", "vors", "vxors"} and dtype.name not in {"i8", "i16", "i32"}:
            raise TypeError(f"pto.{name} only supports integer vector dtypes in TileLang DSL v1")
        if name in {"vadds", "vsubs", "vmuls", "vmaxs", "vmins"} and dtype.name not in {"i8", "i16", "i32", "f16", "bf16", "f32"}:
            raise TypeError(f"pto.{name} does not support this dtype in TileLang DSL v1")

    def _validate_vector_immediate_dtype(self, name: str, dtype: ScalarType) -> None:
        if name in {"vshift", "vslide"} and dtype.name not in {"i8", "i16", "i32", "f16", "bf16", "f32"}:
            raise TypeError(f"pto.{name} does not support this vector dtype in TileLang DSL v1")

    def _validate_ternary_vector_dtype(self, name: str, dtype: ScalarType) -> None:
        if name == "vaxpy" and dtype.name not in {"i16", "i32", "f16", "f32"}:
            raise TypeError("pto.vaxpy only supports i16/i32/f16/f32 in TileLang DSL v1")
        if name == "vmula" and dtype.name not in {"i16", "i32", "f16", "f32"}:
            raise TypeError("pto.vmula only supports i16/i32/f16/f32 in TileLang DSL v1")

    def _validate_multi_result_vector_dtype(self, name: str, dtype: ScalarType) -> None:
        if name == "vmull" and dtype.name != "i32":
            raise TypeError("pto.vmull only supports i32 vectors in TileLang DSL v1")

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
            raise TypeError("slice bounds and vector offsets must be index-typed in TileLang DSL v1")

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
        if isinstance(expr, SemanticAttributeAccess):
            base_value = self._try_static_value(expr.base)
            if isinstance(base_value, TileConfig):
                if expr.attr == "b_layout":
                    return base_value.b_layout
                if expr.attr == "s_layout":
                    return base_value.s_layout
                if expr.attr == "s_fractal_size":
                    return base_value.s_fractal_size
                if expr.attr == "pad_value":
                    return base_value.pad_value
            if base_value is not None and hasattr(base_value, expr.attr):
                return getattr(base_value, expr.attr)
            if isinstance(expr.base.type, SemanticTileType):
                tile_type = expr.base.type
                config = TileConfig() if tile_type.config is None else tile_type.config
                if expr.attr == "shape":
                    return tile_type.shape
                if expr.attr == "valid_shape":
                    return self._resolved_tile_valid_shape(tile_type)
                if expr.attr == "rank":
                    return tile_type.rank
                if expr.attr == "memory_space":
                    return None if tile_type.memory_space is None else MemorySpace(tile_type.memory_space)
                if expr.attr == "config":
                    return config
            if isinstance(expr.base.type, (SemanticTensorViewType, SemanticPartitionTensorViewType)):
                if expr.attr == "rank":
                    return expr.base.type.rank
            return None
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
                if isinstance(lhs, int) and isinstance(rhs, int):
                    return lhs + rhs
                return None
            if expr.op == "sub":
                if isinstance(lhs, int) and isinstance(rhs, int):
                    return lhs - rhs
                return None
            if expr.op == "mul":
                if isinstance(lhs, int) and isinstance(rhs, int):
                    return lhs * rhs
                return None
            if expr.op == "floordiv":
                if isinstance(lhs, int) and isinstance(rhs, int):
                    if rhs == 0:
                        return None
                    return lhs // rhs
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
    "SemanticIfResult",
    "SemanticIfStmt",
    "SemanticIndexType",
    "SemanticKernel",
    "SemanticLiteralExpr",
    "SemanticMaskType",
    "SemanticParameter",
    "SemanticPipeBarrierStmt",
    "SemanticReturnStmt",
    "SemanticScalarType",
    "SemanticSetFlagStmt",
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
    "SemanticTileType",
    "SemanticTupleExpr",
    "SemanticTupleType",
    "SemanticType",
    "SemanticVRegType",
    "SemanticVectorStoreStmt",
    "SemanticWaitFlagStmt",
    "analyze_frontend_kernel",
]
