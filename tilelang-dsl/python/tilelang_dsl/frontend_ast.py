"""Frontend AST nodes for TileLang DSL descriptor materialization."""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from typing import Any

from .support_matrix import (
    ADVANCED_EXPR_PTO_CALLS,
    ADVANCED_TOPLEVEL_PTO_CALLS,
    ADVANCED_VECSCOPE_PTO_CALLS,
    DEFERRED_PTO_SURFACES,
    SUPPORTED_TOPLEVEL_PTO_CALLS,
    SUPPORTED_VECSCOPE_PTO_CALLS,
    advanced_mode_message,
    deferred_surface_message,
)


@dataclass(frozen=True)
class FrontendParameterNode:
    name: str
    kind: str
    annotation: Any
    dtype: Any


@dataclass(frozen=True)
class FrontendTileSpecializationNode:
    name: str
    shape: tuple[int, ...]
    memory_space: str
    config: Any
    valid_shape: tuple[int | None, ...] | None


class FrontendExprNode:
    """Base class for lowered frontend expressions."""


@dataclass(frozen=True)
class FrontendNameExpr(FrontendExprNode):
    name: str


@dataclass(frozen=True)
class FrontendConstantExpr(FrontendExprNode):
    value: Any


@dataclass(frozen=True)
class FrontendSymbolExpr(FrontendExprNode):
    namespace: str
    name: str


@dataclass(frozen=True)
class FrontendSliceExpr(FrontendExprNode):
    start: FrontendExprNode | None
    stop: FrontendExprNode | None
    step: FrontendExprNode | None


@dataclass(frozen=True)
class FrontendTupleExpr(FrontendExprNode):
    elements: tuple[FrontendExprNode, ...]


@dataclass(frozen=True)
class FrontendAttributeExpr(FrontendExprNode):
    base: FrontendExprNode
    attr: str


@dataclass(frozen=True)
class FrontendSubscriptExpr(FrontendExprNode):
    base: FrontendExprNode
    index: FrontendExprNode


@dataclass(frozen=True)
class FrontendBinaryExpr(FrontendExprNode):
    lhs: FrontendExprNode
    op: str
    rhs: FrontendExprNode


@dataclass(frozen=True)
class FrontendCallExpr(FrontendExprNode):
    namespace: str | None
    name: str
    args: tuple[FrontendExprNode, ...]
    keywords: tuple[tuple[str, FrontendExprNode], ...] = ()


class FrontendTargetNode:
    """Base class for assignment targets."""


@dataclass(frozen=True)
class FrontendNameTarget(FrontendTargetNode):
    name: str


@dataclass(frozen=True)
class FrontendTupleTarget(FrontendTargetNode):
    elements: tuple[FrontendNameTarget, ...]


class FrontendStmtNode:
    """Base class for lowered frontend statements."""


@dataclass(frozen=True)
class FrontendAssignStmt(FrontendStmtNode):
    target: FrontendTargetNode
    value: FrontendExprNode
    annotation: Any | None = None


@dataclass(frozen=True)
class FrontendExprStmt(FrontendStmtNode):
    expr: FrontendExprNode


@dataclass(frozen=True)
class FrontendReturnStmt(FrontendStmtNode):
    value: FrontendExprNode | None


@dataclass(frozen=True)
class FrontendForStmt(FrontendStmtNode):
    target: str
    lower_bound: FrontendExprNode
    upper_bound: FrontendExprNode
    step: FrontendExprNode
    body: tuple[FrontendStmtNode, ...]


@dataclass(frozen=True)
class FrontendIfStmt(FrontendStmtNode):
    condition: FrontendExprNode
    then_body: tuple[FrontendStmtNode, ...]
    else_body: tuple[FrontendStmtNode, ...]
    is_constexpr: bool = False


@dataclass(frozen=True)
class FrontendVecscopeStmt(FrontendStmtNode):
    body: tuple[FrontendStmtNode, ...]


@dataclass(frozen=True)
class FrontendStrictVecscopeStmt(FrontendStmtNode):
    captures: tuple[FrontendExprNode, ...]
    block_arguments: tuple[str, ...]
    body: tuple[FrontendStmtNode, ...]


@dataclass(frozen=True)
class FrontendInlineProcParameterNode:
    name: str
    annotation: Any
    default: FrontendExprNode | None


@dataclass(frozen=True)
class FrontendInlineProcNode:
    name: str
    parameters: tuple[FrontendInlineProcParameterNode, ...]
    body: tuple[FrontendStmtNode, ...]


@dataclass(frozen=True)
class FrontendKernelNode:
    target: str
    op: str
    name: str
    verify_enabled: bool
    advanced_enabled: bool
    dtype_signature: tuple[Any, ...]
    parameters: tuple[FrontendParameterNode, ...]
    tile_specializations: tuple[FrontendTileSpecializationNode, ...]
    body: tuple[FrontendStmtNode, ...]
    inline_procs: tuple[FrontendInlineProcNode, ...] = ()


@dataclass(frozen=True)
class _FrontendInlineProc:
    name: str
    source_info: Any
    signature: inspect.Signature


@dataclass(frozen=True)
class _FrontendBuildContext:
    source_info: Any
    templates: dict[str, dict[str, str]]
    selected_op: str | None
    advanced_enabled: bool
    inline_procs: dict[str, _FrontendInlineProc]
    active_inline_proc_stack: tuple[str, ...] = ()
    vecscope_depth: int = 0

    def error(self, node: ast.AST, message: str) -> Exception:
        if self.source_info is not None:
            return self.source_info.error(node, message)
        return ValueError(message)

    def nested_vecscope(self) -> "_FrontendBuildContext":
        return _FrontendBuildContext(
            source_info=self.source_info,
            templates=self.templates,
            selected_op=self.selected_op,
            advanced_enabled=self.advanced_enabled,
            inline_procs=self.inline_procs,
            active_inline_proc_stack=self.active_inline_proc_stack,
            vecscope_depth=self.vecscope_depth + 1,
        )

    def enter_inline_proc(self, name: str, source_info: Any) -> "_FrontendBuildContext":
        return _FrontendBuildContext(
            source_info=source_info,
            templates=self.templates,
            selected_op=self.selected_op,
            advanced_enabled=self.advanced_enabled,
            inline_procs=self.inline_procs,
            active_inline_proc_stack=(*self.active_inline_proc_stack, name),
            vecscope_depth=self.vecscope_depth,
        )


def _inline_proc_param_specs(inline_proc: _FrontendInlineProc) -> tuple[tuple[str, ast.expr | None], ...]:
    function_def = inline_proc.source_info.function_def
    params = function_def.args.args
    defaults = function_def.args.defaults
    first_default = len(params) - len(defaults)
    specs: list[tuple[str, ast.expr | None]] = []
    for index, param in enumerate(params):
        default_node: ast.expr | None = None
        if index >= first_default:
            default_node = defaults[index - first_default]
        specs.append((param.arg, default_node))
    return tuple(specs)


def _bind_inline_proc_call(
    node: ast.Call,
    inline_proc: _FrontendInlineProc,
    context: _FrontendBuildContext,
) -> tuple[FrontendExprNode, ...]:
    if any(keyword.arg is None for keyword in node.keywords):
        raise context.error(
            node,
            "keyword unpacking via `**` is not supported in TileLang DSL v1",
        )

    param_specs = _inline_proc_param_specs(inline_proc)
    param_names = tuple(param_name for param_name, _ in param_specs)
    bound: dict[str, FrontendExprNode] = {}

    if len(node.args) > len(param_specs):
        raise context.error(
            node,
            f"inline_proc `{inline_proc.name}` accepts at most {len(param_specs)} positional arguments in TileLang DSL v1",
        )

    for index, arg_node in enumerate(node.args):
        param_name = param_names[index]
        bound[param_name] = _build_expr(arg_node, context)

    seen_keywords: set[str] = set()
    for keyword in node.keywords:
        assert keyword.arg is not None
        if keyword.arg in seen_keywords:
            raise context.error(
                keyword.value,
                f"duplicate keyword `{keyword.arg}` for inline_proc `{inline_proc.name}` in TileLang DSL v1",
            )
        if keyword.arg not in param_names:
            raise context.error(
                keyword.value,
                f"inline_proc `{inline_proc.name}` does not define keyword `{keyword.arg}` in TileLang DSL v1",
            )
        if keyword.arg in bound:
            raise context.error(
                keyword.value,
                f"inline_proc `{inline_proc.name}` got multiple values for argument `{keyword.arg}` in TileLang DSL v1",
            )
        seen_keywords.add(keyword.arg)
        bound[keyword.arg] = _build_expr(keyword.value, context)

    ordered_args: list[FrontendExprNode] = []
    for param_name, default_node in param_specs:
        value = bound.get(param_name)
        if value is None:
            if default_node is None:
                raise context.error(
                    node,
                    f"inline_proc `{inline_proc.name}` is missing required argument `{param_name}` in TileLang DSL v1",
                )
            value = _build_expr(default_node, context)
        ordered_args.append(value)
    return tuple(ordered_args)


def _collect_name_reads(expr: FrontendExprNode) -> set[str]:
    if isinstance(expr, FrontendNameExpr):
        return {expr.name}
    if isinstance(expr, (FrontendConstantExpr, FrontendSymbolExpr)):
        return set()
    if isinstance(expr, FrontendSliceExpr):
        names: set[str] = set()
        if expr.start is not None:
            names |= _collect_name_reads(expr.start)
        if expr.stop is not None:
            names |= _collect_name_reads(expr.stop)
        if expr.step is not None:
            names |= _collect_name_reads(expr.step)
        return names
    if isinstance(expr, FrontendTupleExpr):
        names: set[str] = set()
        for element in expr.elements:
            names |= _collect_name_reads(element)
        return names
    if isinstance(expr, FrontendAttributeExpr):
        return _collect_name_reads(expr.base)
    if isinstance(expr, FrontendSubscriptExpr):
        return _collect_name_reads(expr.base) | _collect_name_reads(expr.index)
    if isinstance(expr, FrontendBinaryExpr):
        return _collect_name_reads(expr.lhs) | _collect_name_reads(expr.rhs)
    if isinstance(expr, FrontendCallExpr):
        names: set[str] = set()
        for arg in expr.args:
            names |= _collect_name_reads(arg)
        for _, keyword_value in expr.keywords:
            names |= _collect_name_reads(keyword_value)
        return names
    return set()


def _extract_target_names(target: FrontendTargetNode) -> set[str]:
    if isinstance(target, FrontendNameTarget):
        return {target.name}
    if isinstance(target, FrontendTupleTarget):
        return {element.name for element in target.elements}
    return set()

def _validate_inline_capture(
    stmt: FrontendStmtNode,
    param_names: set[str],
    assigned_names: set[str],
    *,
    context: _FrontendBuildContext,
) -> None:
    allowed = param_names | assigned_names
    if isinstance(stmt, FrontendAssignStmt):
        missing = _collect_name_reads(stmt.value) - allowed
        if missing:
            name = sorted(missing)[0]
            raise context.error(
                context.source_info.function_def,
                f"implicit capture of '{name}' is not allowed in inline_proc",
            )
        assigned_names |= _extract_target_names(stmt.target)
        return
    if isinstance(stmt, FrontendExprStmt):
        missing = _collect_name_reads(stmt.expr) - allowed
        if missing:
            name = sorted(missing)[0]
            raise context.error(
                context.source_info.function_def,
                f"implicit capture of '{name}' is not allowed in inline_proc",
            )
        return
    if isinstance(stmt, FrontendReturnStmt):
        if stmt.value is None:
            return
        missing = _collect_name_reads(stmt.value) - allowed
        if missing:
            name = sorted(missing)[0]
            raise context.error(
                context.source_info.function_def,
                f"implicit capture of '{name}' is not allowed in inline_proc",
            )
        return
    if isinstance(stmt, FrontendForStmt):
        header_reads = (
            _collect_name_reads(stmt.lower_bound)
            | _collect_name_reads(stmt.upper_bound)
            | _collect_name_reads(stmt.step)
        )
        missing = header_reads - allowed
        if missing:
            name = sorted(missing)[0]
            raise context.error(
                context.source_info.function_def,
                f"implicit capture of '{name}' is not allowed in inline_proc",
            )

        loop_assigned = set(assigned_names)
        loop_assigned.add(stmt.target)
        for child in stmt.body:
            _validate_inline_capture(child, param_names, loop_assigned, context=context)
        assigned_names.add(stmt.target)
        return
    if isinstance(stmt, FrontendIfStmt):
        missing = _collect_name_reads(stmt.condition) - allowed
        if missing:
            name = sorted(missing)[0]
            raise context.error(
                context.source_info.function_def,
                f"implicit capture of '{name}' is not allowed in inline_proc",
            )
        then_assigned = set(assigned_names)
        else_assigned = set(assigned_names)
        for child in stmt.then_body:
            _validate_inline_capture(child, param_names, then_assigned, context=context)
        for child in stmt.else_body:
            _validate_inline_capture(child, param_names, else_assigned, context=context)
        assigned_names |= then_assigned | else_assigned
        return
    if isinstance(stmt, FrontendVecscopeStmt):
        scope_assigned = set(assigned_names)
        for child in stmt.body:
            _validate_inline_capture(child, param_names, scope_assigned, context=context)
        assigned_names |= scope_assigned
        return
    if isinstance(stmt, FrontendStrictVecscopeStmt):
        captures_missing = set().union(*(_collect_name_reads(capture) for capture in stmt.captures)) - allowed
        if captures_missing:
            name = sorted(captures_missing)[0]
            raise context.error(
                context.source_info.function_def,
                f"implicit capture of '{name}' is not allowed in inline_proc",
            )
        scope_assigned = set(assigned_names) | set(stmt.block_arguments)
        for child in stmt.body:
            _validate_inline_capture(child, param_names, scope_assigned, context=context)
        assigned_names |= scope_assigned


def _collect_inline_proc_calls_expr(
    expr: FrontendExprNode,
    inline_proc_names: set[str],
    into: set[str],
) -> None:
    if isinstance(expr, FrontendCallExpr):
        if expr.namespace is None and expr.name in inline_proc_names:
            into.add(expr.name)
        for arg in expr.args:
            _collect_inline_proc_calls_expr(arg, inline_proc_names, into)
        for _, keyword_value in expr.keywords:
            _collect_inline_proc_calls_expr(keyword_value, inline_proc_names, into)
        return
    if isinstance(expr, FrontendBinaryExpr):
        _collect_inline_proc_calls_expr(expr.lhs, inline_proc_names, into)
        _collect_inline_proc_calls_expr(expr.rhs, inline_proc_names, into)
        return
    if isinstance(expr, FrontendTupleExpr):
        for element in expr.elements:
            _collect_inline_proc_calls_expr(element, inline_proc_names, into)
        return
    if isinstance(expr, FrontendSliceExpr):
        if expr.start is not None:
            _collect_inline_proc_calls_expr(expr.start, inline_proc_names, into)
        if expr.stop is not None:
            _collect_inline_proc_calls_expr(expr.stop, inline_proc_names, into)
        if expr.step is not None:
            _collect_inline_proc_calls_expr(expr.step, inline_proc_names, into)
        return
    if isinstance(expr, FrontendAttributeExpr):
        _collect_inline_proc_calls_expr(expr.base, inline_proc_names, into)
        return
    if isinstance(expr, FrontendSubscriptExpr):
        _collect_inline_proc_calls_expr(expr.base, inline_proc_names, into)
        _collect_inline_proc_calls_expr(expr.index, inline_proc_names, into)


def _collect_inline_proc_calls_stmt(
    stmt: FrontendStmtNode,
    inline_proc_names: set[str],
    into: set[str],
) -> None:
    if isinstance(stmt, FrontendAssignStmt):
        _collect_inline_proc_calls_expr(stmt.value, inline_proc_names, into)
        return
    if isinstance(stmt, FrontendExprStmt):
        _collect_inline_proc_calls_expr(stmt.expr, inline_proc_names, into)
        return
    if isinstance(stmt, FrontendReturnStmt):
        if stmt.value is not None:
            _collect_inline_proc_calls_expr(stmt.value, inline_proc_names, into)
        return
    if isinstance(stmt, FrontendForStmt):
        _collect_inline_proc_calls_expr(stmt.lower_bound, inline_proc_names, into)
        _collect_inline_proc_calls_expr(stmt.upper_bound, inline_proc_names, into)
        _collect_inline_proc_calls_expr(stmt.step, inline_proc_names, into)
        for child in stmt.body:
            _collect_inline_proc_calls_stmt(child, inline_proc_names, into)
        return
    if isinstance(stmt, FrontendIfStmt):
        _collect_inline_proc_calls_expr(stmt.condition, inline_proc_names, into)
        for child in stmt.then_body:
            _collect_inline_proc_calls_stmt(child, inline_proc_names, into)
        for child in stmt.else_body:
            _collect_inline_proc_calls_stmt(child, inline_proc_names, into)
        return
    if isinstance(stmt, FrontendVecscopeStmt):
        for child in stmt.body:
            _collect_inline_proc_calls_stmt(child, inline_proc_names, into)
        return
    if isinstance(stmt, FrontendStrictVecscopeStmt):
        for capture in stmt.captures:
            _collect_inline_proc_calls_expr(capture, inline_proc_names, into)
        for child in stmt.body:
            _collect_inline_proc_calls_stmt(child, inline_proc_names, into)


def _validate_inline_proc_call_graph(
    kernel_body: tuple[FrontendStmtNode, ...],
    inline_proc_nodes: tuple[FrontendInlineProcNode, ...],
    inline_proc_source_infos: dict[str, Any],
) -> None:
    inline_proc_names = {node.name for node in inline_proc_nodes}
    if not inline_proc_names:
        return

    edges: dict[str, set[str]] = {node.name: set() for node in inline_proc_nodes}
    for inline_proc_node in inline_proc_nodes:
        callees = edges[inline_proc_node.name]
        for stmt in inline_proc_node.body:
            _collect_inline_proc_calls_stmt(stmt, inline_proc_names, callees)

    root_callees: set[str] = set()
    for stmt in kernel_body:
        _collect_inline_proc_calls_stmt(stmt, inline_proc_names, root_callees)

    color: dict[str, int] = {}

    def dfs(name: str) -> None:
        state = color.get(name, 0)
        if state == 1:
            source_info = inline_proc_source_infos.get(name)
            if source_info is not None:
                raise source_info.error(
                    source_info.function_def,
                    f"recursive inline_proc call `{name}` is not supported in TileLang DSL v1",
                )
            raise ValueError(f"recursive inline_proc call `{name}` is not supported in TileLang DSL v1")
        if state == 2:
            return
        color[name] = 1
        for callee in edges.get(name, ()):
            dfs(callee)
        color[name] = 2

    for callee in sorted(root_callees):
        dfs(callee)


def _collect_reachable_inline_procs(
    kernel_body: tuple[FrontendStmtNode, ...],
    inline_proc_nodes: tuple[FrontendInlineProcNode, ...],
) -> set[str]:
    inline_proc_names = {node.name for node in inline_proc_nodes}
    if not inline_proc_names:
        return set()

    edges: dict[str, set[str]] = {node.name: set() for node in inline_proc_nodes}
    for inline_proc_node in inline_proc_nodes:
        for stmt in inline_proc_node.body:
            _collect_inline_proc_calls_stmt(stmt, inline_proc_names, edges[inline_proc_node.name])

    roots: set[str] = set()
    for stmt in kernel_body:
        _collect_inline_proc_calls_stmt(stmt, inline_proc_names, roots)

    reachable: set[str] = set()
    stack = list(roots)
    while stack:
        name = stack.pop()
        if name in reachable:
            continue
        reachable.add(name)
        stack.extend(edges.get(name, ()))
    return reachable


_BINARY_OP_NAMES = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.FloorDiv: "floordiv",
}
_COMPARE_OP_NAMES = {
    ast.Eq: "eq",
    ast.NotEq: "ne",
    ast.Gt: "gt",
    ast.Lt: "lt",
    ast.GtE: "ge",
    ast.LtE: "le",
}
_BOOL_OP_NAMES = {
    ast.And: "and",
    ast.Or: "or",
}

_DMA_CALL_KEYWORDS: dict[str, frozenset[str]] = {
    "set_loop2_stride_outtoub": frozenset({"src_stride", "dst_stride"}),
    "set_loop1_stride_outtoub": frozenset({"src_stride", "dst_stride"}),
    "set_loop_size_outtoub": frozenset({"loop1", "loop2"}),
    "set_loop2_stride_ubtoout": frozenset({"src_stride", "dst_stride"}),
    "set_loop1_stride_ubtoout": frozenset({"src_stride", "dst_stride"}),
    "set_loop_size_ubtoout": frozenset({"loop1", "loop2"}),
    "copy_gm_to_ubuf": frozenset(
        {
            "src",
            "dst",
            "sid",
            "n_burst",
            "len_burst",
            "left_padding_count",
            "right_padding_count",
            "data_select_bit",
            "enable_ub_pad",
            "l2_cache_ctl",
            "gm_stride",
            "ub_stride",
        }
    ),
    "copy_ubuf_to_gm": frozenset(
        {
            "src",
            "dst",
            "sid",
            "n_burst",
            "len_burst",
            "reserved",
            "burst_dst_stride",
            "burst_src_stride",
            "gm_stride",
            "ub_stride",
        }
    ),
}


def _attribute_path(node: ast.AST) -> tuple[str, ...] | None:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        base_path = _attribute_path(node.value)
        if base_path is None:
            return None
        return base_path + (node.attr,)
    return None


def _validate_resolved_template_op_surface(
    op_name: str,
    node: ast.AST,
    context: _FrontendBuildContext,
) -> None:
    if op_name in SUPPORTED_TOPLEVEL_PTO_CALLS:
        return
    if op_name in SUPPORTED_VECSCOPE_PTO_CALLS:
        return
    if op_name in ADVANCED_VECSCOPE_PTO_CALLS:
        if context.advanced_enabled:
            return
        raise context.error(
            node,
            advanced_mode_message(op_name),
        )
    if op_name in ADVANCED_EXPR_PTO_CALLS or op_name in ADVANCED_TOPLEVEL_PTO_CALLS:
        if context.advanced_enabled:
            return
        raise context.error(
            node,
            advanced_mode_message(op_name),
        )
    if op_name in DEFERRED_PTO_SURFACES:
        raise context.error(
            node,
            deferred_surface_message(op_name),
        )
    raise context.error(
        node,
        f"unsupported op surface `pto.{op_name}` in TileLang DSL v1",
    )


def _build_call_keywords(
    node: ast.Call,
    *,
    namespace: str | None,
    name: str,
    context: _FrontendBuildContext,
) -> tuple[tuple[str, FrontendExprNode], ...]:
    if not node.keywords:
        return ()

    for keyword in node.keywords:
        if keyword.arg is None:
            raise context.error(
                keyword.value,
                "keyword unpacking via `**` is not supported in TileLang DSL v1",
            )

    allowed_keywords = _DMA_CALL_KEYWORDS.get(name) if namespace == "pto" else None
    if allowed_keywords is None:
        call_name = f"{namespace + '.' if namespace else ''}{name}"
        raise context.error(
            node,
            f"`{call_name}` does not support keyword arguments in TileLang DSL v1; "
            "no public call surface currently accepts them",
        )

    seen: set[str] = set()
    built_keywords: list[tuple[str, FrontendExprNode]] = []
    for keyword in node.keywords:
        assert keyword.arg is not None
        if keyword.arg in seen:
            raise context.error(
                keyword.value,
                f"duplicate keyword `{keyword.arg}` for `pto.{name}` in TileLang DSL v1",
            )
        if keyword.arg not in allowed_keywords:
            raise context.error(
                keyword.value,
                f"unsupported keyword `{keyword.arg}` for `pto.{name}` in TileLang DSL v1",
            )
        seen.add(keyword.arg)
        built_keywords.append((keyword.arg, _build_expr(keyword.value, context)))
    return tuple(built_keywords)


def _build_expr(node: ast.AST, context: _FrontendBuildContext) -> FrontendExprNode:
    if isinstance(node, ast.Name):
        return FrontendNameExpr(name=node.id)
    if isinstance(node, ast.Constant):
        return FrontendConstantExpr(value=node.value)
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.UAdd):
            sign = 1
        elif isinstance(node.op, ast.USub):
            sign = -1
        else:
            raise context.error(
                node,
                f"unsupported unary operator `{type(node.op).__name__}` in TileLang DSL v1",
            )
        if not isinstance(node.operand, ast.Constant) or isinstance(node.operand.value, bool):
            raise context.error(
                node,
                "unary +/- currently only supports numeric literals in TileLang DSL v1",
            )
        literal = node.operand.value
        if not isinstance(literal, (int, float)):
            raise context.error(
                node,
                "unary +/- currently only supports numeric literals in TileLang DSL v1",
            )
        return FrontendConstantExpr(value=literal if sign > 0 else -literal)
    if isinstance(node, ast.Slice):
        start = None if node.lower is None else _build_expr(node.lower, context)
        stop = None if node.upper is None else _build_expr(node.upper, context)
        step = None if node.step is None else _build_expr(node.step, context)
        return FrontendSliceExpr(start=start, stop=stop, step=step)
    if isinstance(node, ast.Tuple):
        return FrontendTupleExpr(
            elements=tuple(_build_expr(elt, context) for elt in node.elts)
        )
    if isinstance(node, ast.Attribute):
        path = _attribute_path(node)
        if path is not None and path[0] in {"pto", "PAT", "PIPE", "Pipe", "EVENT", "Event"} and len(path) >= 2:
            return FrontendSymbolExpr(namespace=".".join(path[:-1]), name=path[-1])
        return FrontendAttributeExpr(base=_build_expr(node.value, context), attr=node.attr)
    if isinstance(node, ast.Subscript):
        return FrontendSubscriptExpr(
            base=_build_expr(node.value, context),
            index=_build_expr(node.slice, context),
        )
    if isinstance(node, ast.BinOp):
        op_name = _BINARY_OP_NAMES.get(type(node.op))
        if op_name is None:
            raise context.error(
                node,
                f"unsupported binary operator `{type(node.op).__name__}` in TileLang DSL v1",
            )
        return FrontendBinaryExpr(
            lhs=_build_expr(node.left, context),
            op=op_name,
            rhs=_build_expr(node.right, context),
        )
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise context.error(
                node,
                "chained comparisons are not supported in TileLang DSL v1",
            )
        op_name = _COMPARE_OP_NAMES.get(type(node.ops[0]))
        if op_name is None:
            raise context.error(
                node,
                f"unsupported comparison operator `{type(node.ops[0]).__name__}` in TileLang DSL v1",
            )
        return FrontendBinaryExpr(
            lhs=_build_expr(node.left, context),
            op=op_name,
            rhs=_build_expr(node.comparators[0], context),
        )
    if isinstance(node, ast.BoolOp):
        op_name = _BOOL_OP_NAMES.get(type(node.op))
        if op_name is None:
            raise context.error(
                node,
                f"unsupported boolean operator `{type(node.op).__name__}` in TileLang DSL v1",
            )
        if len(node.values) < 2:
            raise context.error(
                node,
                "boolean expressions must contain at least two operands in TileLang DSL v1",
            )
        expr = _build_expr(node.values[0], context)
        for value in node.values[1:]:
            expr = FrontendBinaryExpr(
                lhs=expr,
                op=op_name,
                rhs=_build_expr(value, context),
            )
        return expr
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in context.inline_procs:
            inline_proc = context.inline_procs[node.func.id]
            if node.func.id in context.active_inline_proc_stack:
                raise context.error(
                    node,
                    f"recursive inline_proc call `{node.func.id}` is not supported in TileLang DSL v1",
                )
            return FrontendCallExpr(
                namespace=None,
                name=node.func.id,
                args=_bind_inline_proc_call(node, inline_proc, context),
                keywords=(),
            )
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "pto"
            and node.func.attr == "tpl"
        ):
            if not node.args:
                raise context.error(
                    node,
                    "pto.tpl() requires a non-empty string literal slot name as the first argument",
                )
            slot_expr = node.args[0]
            if not (
                isinstance(slot_expr, ast.Constant)
                and isinstance(slot_expr.value, str)
                and slot_expr.value
            ):
                raise context.error(
                    slot_expr,
                    "pto.tpl() requires a non-empty string literal slot name",
                )
            slot_name = slot_expr.value
            slot_bindings = context.templates.get(slot_name)
            if slot_bindings is None:
                raise context.error(
                    slot_expr,
                    f"unknown template slot {slot_name!r} in TileLang DSL v1",
                )
            if context.selected_op is None:
                raise context.error(
                    node,
                    "pto.tpl() requires pto.select_kernel(...) to bind a concrete op before expansion",
                )
            resolved_op = slot_bindings.get(context.selected_op)
            if resolved_op is None:
                raise context.error(
                    slot_expr,
                    f"template slot {slot_name!r} does not define an implementation for "
                    f"selected op {context.selected_op!r}",
                )
            _validate_resolved_template_op_surface(resolved_op, node, context)
            return FrontendCallExpr(
                namespace="pto",
                name=resolved_op,
                args=tuple(_build_expr(arg, context) for arg in node.args[1:]),
                keywords=_build_call_keywords(
                    node,
                    namespace="pto",
                    name=resolved_op,
                    context=context,
                ),
            )
        if isinstance(node.func, ast.Name):
            return FrontendCallExpr(
                namespace=None,
                name=node.func.id,
                args=tuple(_build_expr(arg, context) for arg in node.args),
                keywords=_build_call_keywords(
                    node,
                    namespace=None,
                    name=node.func.id,
                    context=context,
                ),
            )
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            return FrontendCallExpr(
                namespace=node.func.value.id,
                name=node.func.attr,
                args=tuple(_build_expr(arg, context) for arg in node.args),
                keywords=_build_call_keywords(
                    node,
                    namespace=node.func.value.id,
                    name=node.func.attr,
                    context=context,
                ),
            )
    raise context.error(
        node,
        f"unsupported expression `{type(node).__name__}` in TileLang DSL v1",
    )


def _build_target(node: ast.AST, context: _FrontendBuildContext) -> FrontendTargetNode:
    if isinstance(node, ast.Name):
        return FrontendNameTarget(name=node.id)
    if isinstance(node, ast.Tuple):
        elements = []
        for elt in node.elts:
            if not isinstance(elt, ast.Name):
                raise context.error(elt, "tuple assignment only supports names in TileLang DSL v1")
            elements.append(FrontendNameTarget(name=elt.id))
        return FrontendTupleTarget(elements=tuple(elements))
    raise context.error(
        node,
        f"unsupported assignment target `{type(node).__name__}` in TileLang DSL v1",
    )


def _build_stmt_list(nodes: list[ast.stmt] | tuple[ast.stmt, ...], context: _FrontendBuildContext) -> tuple[FrontendStmtNode, ...]:
    return tuple(_build_stmt(node, context) for node in nodes)


def _build_stmt(node: ast.stmt, context: _FrontendBuildContext) -> FrontendStmtNode:
    if isinstance(node, ast.Assign):
        if len(node.targets) != 1:
            raise context.error(node, "multiple assignment targets are not supported in TileLang DSL v1")
        return FrontendAssignStmt(
            target=_build_target(node.targets[0], context),
            value=_build_expr(node.value, context),
        )
    if isinstance(node, ast.AnnAssign):
        if node.value is None:
            raise context.error(node, "annotation-only assignments are not supported in TileLang DSL v1")
        return FrontendAssignStmt(
            target=_build_target(node.target, context),
            value=_build_expr(node.value, context),
            annotation=node.annotation,
        )
    if isinstance(node, ast.Expr):
        return FrontendExprStmt(expr=_build_expr(node.value, context))
    if isinstance(node, ast.Return):
        value = None
        if node.value is not None:
            if not (isinstance(node.value, ast.Constant) and node.value.value is None):
                value = _build_expr(node.value, context)
        return FrontendReturnStmt(value=value)
    if isinstance(node, ast.For):
        if not isinstance(node.target, ast.Name):
            raise context.error(node.target, "for target must be a single name")
        if not isinstance(node.iter, ast.Call) or not isinstance(node.iter.func, ast.Name) or node.iter.func.id != "range":
            raise context.error(node.iter, "only Python range(lb, ub, step) loops are supported")
        if len(node.iter.args) != 3:
            raise context.error(node.iter, "range() expects exactly 3 arguments in TileLang DSL v1")
        return FrontendForStmt(
            target=node.target.id,
            lower_bound=_build_expr(node.iter.args[0], context),
            upper_bound=_build_expr(node.iter.args[1], context),
            step=_build_expr(node.iter.args[2], context),
            body=_build_stmt_list(node.body, context),
        )
    if isinstance(node, ast.If):
        is_constexpr = False
        condition_node: ast.AST = node.test
        if (
            isinstance(node.test, ast.Call)
            and isinstance(node.test.func, ast.Attribute)
            and isinstance(node.test.func.value, ast.Name)
            and node.test.func.value.id == "pto"
            and node.test.func.attr == "constexpr"
        ):
            if node.test.keywords:
                raise context.error(
                    node.test,
                    "pto.constexpr() does not support keyword arguments in TileLang DSL v1",
                )
            if len(node.test.args) != 1:
                raise context.error(
                    node.test,
                    "pto.constexpr() expects exactly 1 positional argument in TileLang DSL v1",
                )
            is_constexpr = True
            condition_node = node.test.args[0]
        return FrontendIfStmt(
            condition=_build_expr(condition_node, context),
            then_body=_build_stmt_list(node.body, context),
            else_body=_build_stmt_list(node.orelse, context),
            is_constexpr=is_constexpr,
        )
    if isinstance(node, ast.With):
        if len(node.items) != 1:
            raise context.error(node, "only a single with-item is supported in TileLang DSL v1")
        item = node.items[0]
        if not isinstance(item.context_expr, ast.Call):
            raise context.error(item.context_expr, "with context must be a call in TileLang DSL v1")
        if not (
            isinstance(item.context_expr.func, ast.Attribute)
            and isinstance(item.context_expr.func.value, ast.Name)
            and item.context_expr.func.value.id == "pto"
        ):
            raise context.error(
                item.context_expr,
                "only pto.vecscope/pto.strict_vecscope are supported in TileLang DSL v1",
            )
        with_name = item.context_expr.func.attr
        if with_name == "vecscope":
            if item.context_expr.args or item.context_expr.keywords:
                raise context.error(
                    item.context_expr,
                    "pto.vecscope() does not accept positional or keyword arguments in TileLang DSL v1",
                )
            if item.optional_vars is not None:
                raise context.error(item, "pto.vecscope() does not support `as` bindings in TileLang DSL v1")
            return FrontendVecscopeStmt(
                body=_build_stmt_list(node.body, context.nested_vecscope()),
            )
        if with_name != "strict_vecscope":
            raise context.error(
                item.context_expr,
                "only pto.vecscope/pto.strict_vecscope are supported in TileLang DSL v1",
            )
        if not context.advanced_enabled:
            raise context.error(
                item.context_expr,
                advanced_mode_message("strict_vecscope"),
            )
        if not isinstance(item.optional_vars, ast.Tuple):
            raise context.error(item, "pto.strict_vecscope requires tuple binding in 'as'")
        block_arguments = []
        for elt in item.optional_vars.elts:
            if not isinstance(elt, ast.Name):
                raise context.error(elt, "pto.strict_vecscope bindings must be names")
            block_arguments.append(elt.id)
        return FrontendStrictVecscopeStmt(
            captures=tuple(_build_expr(arg, context) for arg in item.context_expr.args),
            block_arguments=tuple(block_arguments),
            body=_build_stmt_list(node.body, context.nested_vecscope()),
        )
    raise context.error(
        node,
        f"unsupported statement `{type(node).__name__}` in TileLang DSL v1",
    )


def build_frontend_kernel_node(descriptor: Any) -> FrontendKernelNode:
    """Project the core-foundation descriptor into a lowering-owned AST."""

    parameters = tuple(
        FrontendParameterNode(
            name=param.name,
            kind=param.kind,
            annotation=param.annotation,
            dtype=param.dtype,
        )
        for param in descriptor.parameters
    )
    tile_specializations = tuple(
        FrontendTileSpecializationNode(
            name=name,
            shape=spec.shape,
            memory_space=spec.memory_space.value,
            config=spec.config,
            valid_shape=spec.valid_shape,
        )
        for name, spec in descriptor.specializations
    )
    source_info = descriptor._source_info
    sorted_inline_procs = tuple(sorted(descriptor.inline_procs.items(), key=lambda item: item[0]))
    context = _FrontendBuildContext(
        source_info=source_info,
        templates=descriptor.templates,
        selected_op=descriptor.selected_op,
        advanced_enabled=descriptor.advanced_enabled,
        inline_procs={
            name: _FrontendInlineProc(
                name=name,
                source_info=proc.source_info,
                signature=proc.signature,
            )
            for name, proc in sorted_inline_procs
        },
    )
    body = ()
    if source_info is not None:
        body = _build_stmt_list(source_info.function_def.body, context)

    inline_proc_descriptors = {name: descriptor for name, descriptor in sorted_inline_procs}
    inline_proc_names = set(inline_proc_descriptors)
    root_inline_calls: set[str] = set()
    for stmt in body:
        _collect_inline_proc_calls_stmt(stmt, inline_proc_names, root_inline_calls)

    inline_proc_nodes_by_name: dict[str, FrontendInlineProcNode] = {}
    inline_proc_source_infos: dict[str, Any] = {}
    pending = list(sorted(root_inline_calls))
    while pending:
        name = pending.pop()
        if name in inline_proc_nodes_by_name:
            continue
        inline_proc_descriptor = inline_proc_descriptors.get(name)
        if inline_proc_descriptor is None:
            continue
        inline_source = inline_proc_descriptor.source_info
        if inline_source is None:
            if source_info is not None:
                raise context.error(
                    source_info.function_def,
                    f"inline_proc `{name}` requires source-visible Python functions",
                )
            raise ValueError(
                f"inline_proc `{name}` requires source-visible Python functions"
            )
        inline_proc_source_infos[name] = inline_source
        helper_context = context.enter_inline_proc(name, inline_source)
        helper_body = _build_stmt_list(inline_source.function_def.body, helper_context)
        parameter_specs = _inline_proc_param_specs(
            _FrontendInlineProc(
                name=name,
                source_info=inline_source,
                signature=inline_proc_descriptor.signature,
            )
        )
        inline_proc_node = FrontendInlineProcNode(
            name=name,
            parameters=tuple(
                FrontendInlineProcParameterNode(
                    name=param_name,
                    annotation=arg.annotation,
                    default=None
                    if default_node is None
                    else _build_expr(default_node, helper_context),
                )
                for (param_name, default_node), arg in zip(parameter_specs, inline_source.function_def.args.args)
            ),
            body=helper_body,
        )
        inline_proc_nodes_by_name[name] = inline_proc_node
        nested_calls: set[str] = set()
        for stmt in helper_body:
            _collect_inline_proc_calls_stmt(stmt, inline_proc_names, nested_calls)
        for nested in sorted(nested_calls):
            if nested not in inline_proc_nodes_by_name:
                pending.append(nested)

    reachable_inline_proc_nodes = tuple(
        inline_proc_nodes_by_name[name]
        for name, _ in sorted_inline_procs
        if name in inline_proc_nodes_by_name
    )
    for inline_proc_node in reachable_inline_proc_nodes:
        source = inline_proc_source_infos[inline_proc_node.name]
        helper_context = context.enter_inline_proc(inline_proc_node.name, source)
        assigned_names: set[str] = set()
        param_names = {parameter.name for parameter in inline_proc_node.parameters}
        for stmt in inline_proc_node.body:
            _validate_inline_capture(
                stmt,
                param_names,
                assigned_names,
                context=helper_context,
            )

    _validate_inline_proc_call_graph(
        body,
        reachable_inline_proc_nodes,
        inline_proc_source_infos,
    )

    return FrontendKernelNode(
        target=descriptor.target,
        op=descriptor.op,
        name=descriptor.name,
        verify_enabled=descriptor.verify_enabled,
        advanced_enabled=descriptor.advanced_enabled,
        dtype_signature=descriptor.dtype_signature,
        parameters=parameters,
        tile_specializations=tile_specializations,
        body=body,
        inline_procs=reachable_inline_proc_nodes,
    )


__all__ = [
    "FrontendAssignStmt",
    "FrontendAttributeExpr",
    "FrontendBinaryExpr",
    "FrontendCallExpr",
    "FrontendConstantExpr",
    "FrontendExprNode",
    "FrontendExprStmt",
    "FrontendForStmt",
    "FrontendIfStmt",
    "FrontendInlineProcNode",
    "FrontendInlineProcParameterNode",
    "FrontendKernelNode",
    "FrontendNameExpr",
    "FrontendNameTarget",
    "FrontendParameterNode",
    "FrontendReturnStmt",
    "FrontendSliceExpr",
    "FrontendVecscopeStmt",
    "FrontendStrictVecscopeStmt",
    "FrontendStmtNode",
    "FrontendSubscriptExpr",
    "FrontendSymbolExpr",
    "FrontendTargetNode",
    "FrontendTileSpecializationNode",
    "FrontendTupleExpr",
    "FrontendTupleTarget",
    "build_frontend_kernel_node",
]
