# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0
# (the "License"). Please refer to the License for details. You may not use
# this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
# AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
# FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
# for the full text of the License.

"""Bounded, non-executing extraction of explicit main-driver tensor samples.

Only direct unconditional main statements are interpreted. Unsupported calls
or control poison the remaining segment rather than inventing argument values.
This is a separate collection population, not a replacement for static seeds.
"""

import ast
import math
import operator

DTYPES = {"float16", "float32", "float64", "bfloat16", "int8", "int16", "int32", "int64", "uint8", "bool"}
CONSTRUCTORS = {"empty", "zeros", "ones", "full", "rand", "randn"}
OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.FloorDiv: operator.floordiv}


def checked_scalar(value):
    if type(value) not in (int, float, bool) or not math.isfinite(value) or abs(value) > 2**31:
        raise ValueError("unsupported or oversized scalar")
    return value


def call_entry(call, names):
    target = call.func
    if isinstance(target, ast.Name) and target.id in names:
        return target.id
    if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
            and target.value.id in names and target.attr in {"lower", "compile", "specialize"}):
        return target.value.id
    return None


class SampleState:
    def __init__(self):
        self.values = {}
        self.torch_names = set()
        self.config_names = set()
        self.language_names = set()
        self.rebound = set()
        self.blocked = ""

    def expression(self, node, depth=0):
        if depth > 32:
            raise ValueError("expression nesting limit")
        if isinstance(node, ast.Constant):
            return checked_scalar(node.value)
        if isinstance(node, ast.Name) and node.id in self.values:
            return self.values[node.id]
        if isinstance(node, (ast.Tuple, ast.List)) and len(node.elts) <= 16:
            return [self.expression(item, depth + 1) for item in node.elts]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = checked_scalar(self.expression(node.operand, depth + 1))
            return checked_scalar(-value if isinstance(node.op, ast.USub) else value)
        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            left = checked_scalar(self.expression(node.left, depth + 1))
            right = checked_scalar(self.expression(node.right, depth + 1))
            return checked_scalar(OPERATORS[type(node.op)](left, right))
        if isinstance(node, ast.Call):
            return self.tensor(node, depth)
        raise ValueError("unresolved source expression")

    def torch_member(self, node):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in self.torch_names:
            return node.attr
        return ""

    def tensor(self, call, depth):
        kind = self.torch_member(call.func)
        if kind not in CONSTRUCTORS or any(keyword.arg != "dtype" for keyword in call.keywords):
            raise ValueError("unsupported tensor constructor or keywords")
        if len(call.keywords) != 1 or self.torch_member(call.keywords[0].value) not in DTYPES:
            raise ValueError("tensor sample requires explicit supported dtype")
        args = [self.expression(arg, depth + 1) for arg in call.args]
        if kind == "full":
            if len(args) != 2:
                raise ValueError("full requires explicit shape and fill")
            checked_scalar(args[1])
            shape = args[0]
        else:
            shape = args[0] if len(args) == 1 and isinstance(args[0], list) else args
        if not isinstance(shape, list) or not 1 <= len(shape) <= 8:
            raise ValueError("tensor rank outside collection bound")
        elements = 1
        for dimension in shape:
            if type(dimension) is not int or not 0 < dimension <= 2**24:
                raise ValueError("tensor extent outside collection bound")
            elements *= dimension
            if elements > 2**30:
                raise ValueError("tensor size outside collection bound")
        return {"shape": shape, "dtype": self.torch_member(call.keywords[0].value),
                "identity": f"{call.lineno}:{call.col_offset}"}

    def readonly_expression(self, node):
        if isinstance(node, ast.Constant):
            return True
        if isinstance(node, ast.Name):
            return node.id in self.values and self.values[node.id] != "config"
        if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp, ast.Tuple, ast.List)):
            return all(self.readonly_expression(child) for child in ast.iter_child_nodes(node)
                       if not isinstance(child, (ast.operator, ast.unaryop, ast.cmpop, ast.boolop, ast.expr_context)))
        if isinstance(node, ast.Call) and self.torch_member(node.func) in {"allclose", "equal"}:
            values = (*node.args, *(item.value for item in node.keywords))
            return all(self.readonly_expression(value) for value in values)
        return False

    def config_call(self, node):
        return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in self.config_names and not node.args and not node.keywords)

    def bind(self, targets, value):
        for target in targets:
            if not isinstance(target, ast.Name):
                raise ValueError("unsupported assignment target")
            self.torch_names.discard(target.id)
            self.config_names.discard(target.id)
            self.language_names.discard(target.id)
            self.rebound.add(target.id)
            self.values.pop(target.id, None)
            if value is not None:
                self.values[target.id] = value


def main_guard(node):
    return (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__"
            and len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.Eq)
            and len(node.test.comparators) == 1 and isinstance(node.test.comparators[0], ast.Constant)
            and node.test.comparators[0].value == "__main__" and not node.orelse)


def sample_call(call, entry, state):
    record = {"entry": entry, "line": call.lineno, "column": call.col_offset,
              "invocation": ast.unparse(call), "arguments": [], "keywords": {}, "config_override": None}
    try:
        if state.blocked:
            raise ValueError(state.blocked)
        if entry in state.rebound:
            raise ValueError("entry name was rebound")
        record["arguments"] = [state.expression(arg) for arg in call.args]
        for keyword in call.keywords:
            if keyword.arg is None or keyword.arg in record["keywords"]:
                raise ValueError("expanded or duplicate call keywords")
            if keyword.arg == "config":
                config_name = isinstance(keyword.value, ast.Name) and state.values.get(keyword.value.id) == "config"
                if not config_name and not state.config_call(keyword.value):
                    raise ValueError("unresolved or effectful config expression")
                record["config_override"] = ast.unparse(keyword.value)
            else:
                record["keywords"][keyword.arg] = state.expression(keyword.value)
        record["status"] = "source-samples"
    except (ValueError, ArithmeticError) as error:
        record.update(status="needs-driver-adapter", detail=str(error), arguments=[], keywords={})
    return record


def safe_definition(node, state):
    """Only inert defaults and trusted language typing/decorators are admitted."""
    if isinstance(node, ast.ClassDef):
        return False

    def annotation(value):
        if value is None or isinstance(value, ast.Constant):
            return True
        if isinstance(value, ast.Name):
            return value.id in {"int", "float", "bool", "str"} and value.id not in state.rebound
        if isinstance(value, (ast.List, ast.Tuple)):
            return all(annotation(item) for item in value.elts)
        if isinstance(value, ast.Subscript):
            return annotation(value.value) and annotation(value.slice)
        if isinstance(value, ast.Attribute):
            base = value
            while isinstance(base, ast.Attribute):
                base = base.value
            return isinstance(base, ast.Name) and base.id in state.language_names
        return False

    parameters = (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs,
                  *([node.args.vararg] if node.args.vararg else []),
                  *([node.args.kwarg] if node.args.kwarg else []))
    if not all(annotation(parameter.annotation) for parameter in parameters) or not annotation(node.returns):
        return False
    for value in (*node.args.defaults, *node.args.kw_defaults):
        if value is not None:
            try:
                state.expression(value)
            except (ValueError, ArithmeticError):
                return False
    for decorator in node.decorator_list:
        parts = []
        while isinstance(decorator, ast.Attribute):
            parts.append(decorator.attr)
            decorator = decorator.value
        if not (isinstance(decorator, ast.Name) and decorator.id in state.language_names
                and list(reversed(parts)) in (["jit"], ["jit", "incore"], ["jit", "host"],
                                             ["jit", "graph"], ["jit", "opaque"], ["jit", "inline"])):
            return False
    return True


def discover_driver_calls(text, names):
    tree = ast.parse(text)
    if sum(1 for _ in ast.walk(tree)) > 100000:
        raise ValueError("driver AST size limit")
    state = SampleState()
    records = []
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                name = alias.asname or alias.name
                state.torch_names.discard(name)
                state.config_names.discard(name)
                state.language_names.discard(name)
                state.values.pop(name, None)
                state.rebound.add(name)
                if alias.name == "torch":
                    state.torch_names.add(name)
                if alias.name == "pypto.language" and alias.asname:
                    state.language_names.add(name)
        elif isinstance(statement, ast.ImportFrom):
            for alias in statement.names:
                name = alias.asname or alias.name
                state.torch_names.discard(name)
                state.config_names.discard(name)
                state.language_names.discard(name)
                state.values.pop(name, None)
                state.rebound.add(name)
                if name == "*":
                    state.blocked = "wildcard import has unresolved bindings"
                if statement.module == "pypto.runtime" and alias.name == "RunConfig":
                    state.config_names.add(name)
                if statement.module == "pypto" and alias.name == "language":
                    state.language_names.add(name)
        elif isinstance(statement, ast.Assign):
            try:
                state.bind(statement.targets, state.expression(statement.value))
            except (ValueError, ArithmeticError):
                state.blocked = f"line {statement.lineno}: unresolved module initialization"
        elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not safe_definition(statement, state):
                state.blocked = f"line {statement.lineno}: unresolved definition-time effects"
            state.torch_names.discard(statement.name)
            state.config_names.discard(statement.name)
            state.language_names.discard(statement.name)
            state.values.pop(statement.name, None)
            state.rebound.discard(statement.name)
        elif main_guard(statement):
            for child in statement.body:
                process_statement(child, names, state, records)
        elif not (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)):
            state.blocked = f"line {statement.lineno}: unsupported module effects or control"
    return records


def process_statement(statement, names, state, records):
    value = statement.value if isinstance(statement, (ast.Assign, ast.Expr)) else None
    entry = call_entry(value, names) if isinstance(value, ast.Call) else None
    if entry:
        nested = [node for node in ast.walk(value) if node is not value
                  and isinstance(node, ast.Call) and call_entry(node, names)]
        if nested:
            state.blocked = "nested driver entry invocation"
            records.extend(sample_call(node, call_entry(node, names), state) for node in nested)
        record = sample_call(value, entry, state)
        records.append(record)
        if record["status"] != "source-samples":
            state.blocked = f"line {statement.lineno}: unresolved preceding call arguments"
        try:
            if isinstance(statement, ast.Assign):
                state.bind(statement.targets, None)
        except ValueError as error:
            state.blocked = str(error)
        return
    try:
        if isinstance(statement, ast.Assign):
            state.bind(statement.targets, "config" if state.config_call(value) else state.expression(value))
        elif isinstance(statement, ast.Assert) and state.readonly_expression(statement.test):
            # Message evaluation can execute arbitrary Python on failure; this
            # adapter extracts successful driver paths, not assertion outcomes.
            pass
        elif isinstance(value, ast.Constant) or (isinstance(value, ast.Call)
                                                and state.torch_member(value.func) == "manual_seed"
                                                and all(state.readonly_expression(arg) for arg in value.args)
                                                and not value.keywords):
            pass
        else:
            raise ValueError("unsupported driver statement or control")
    except (ValueError, ArithmeticError) as error:
        state.blocked = f"line {statement.lineno}: {error}"
        for node in ast.walk(statement):
            if isinstance(node, ast.Call) and (entry := call_entry(node, names)):
                records.append(sample_call(node, entry, state))
