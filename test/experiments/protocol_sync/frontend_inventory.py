# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0
# (the "License"). Please refer to the License for details. You may not use
# this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
# AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
# FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
# for the full text of the License.

"""Inventory source-backed frontend entry points without importing kernel code.

This is the static entry-point component of corpus collection, not a claim to
enumerate parameterized pytest cases. Nested definitions stay visible as
unresolved construction seeds instead of disappearing from the population.
"""

import ast
import hashlib
import subprocess
from pathlib import Path

from provenance import sha256


def qualified_name(node):
    """Read a dotted decorator spelling; never evaluate its arguments."""
    if isinstance(node, ast.Call):
        return qualified_name(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = qualified_name(node.value)
        return f"{base}.{node.attr}" if base else ""
    return ""


def entry_kind(node):
    for decorator in node.decorator_list:
        spelling = qualified_name(decorator)
        if spelling in {"pl.program", "program"}:
            return "program"
        if spelling in {"pl.jit", "jit", "pl.jit.host", "pl.jit.incore", "pl.jit.graph", "pl.jit.opaque"}:
            return "jit"
    return None


def discover_entries(text):
    """Record lexical ownership, including definitions requiring a factory."""
    entries = []

    def visit(node, owners):
        definition = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        if definition:
            kind = entry_kind(node)
            if kind:
                entries.append({"entry": ".".join((*owners, node.name)), "kind": kind,
                                "line": node.lineno, "top_level": not owners})
            owners = (*owners, node.name)
        for child in ast.iter_child_nodes(node):
            visit(child, owners)

    visit(ast.parse(text), ())
    return entries


def source_revision(root):
    """Only a clean, real source checkout can define the frozen population."""
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], check=True,
                              capture_output=True, text=True, timeout=30).stdout.strip()

    if git("status", "--porcelain", "--untracked-files=normal"):
        raise ValueError(f"frontend source checkout is not clean: {root}")
    return {"root": str(root), "commit": git("rev-parse", "HEAD"),
            "tree": git("rev-parse", "HEAD^{tree}"),
            "submodules": git("submodule", "status")}


def inventory_source(root, family):
    """Inventory examples/models; test adapters add their own parameterized seeds."""
    root = Path(root).resolve(strict=True)
    metadata = source_revision(root)
    records = []
    for directory in ("examples", "models"):
        for path in sorted((root / directory).rglob("*.py")):
            if not path.resolve(strict=True).is_relative_to(root):
                raise ValueError("frontend inventory source escapes its checkout")
            relative = path.relative_to(root).as_posix()
            text = path.read_text(encoding="utf-8")
            try:
                entries = discover_entries(text)
            except SyntaxError as error:
                entries = [{"entry": "", "kind": "syntax-error", "line": error.lineno,
                            "top_level": False, "error": str(error)}]
            for entry in entries:
                identity = f"{family}:{relative}:{entry['entry']}:{entry['line']}"
                records.append(dict(entry, case_id=hashlib.sha256(identity.encode()).hexdigest()[:20],
                                    family=family, source=relative, source_sha256=sha256(path),
                                    draft=family == "pypto-lib" and path.stem.endswith("_draft")))
    return metadata, records
