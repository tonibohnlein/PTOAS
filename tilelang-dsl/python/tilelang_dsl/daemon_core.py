# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Core logic for TileLang daemon instance caching.

This module provides:
- InstanceCache: In-process cache with FIFO eviction
- Single-flight: Deduplication for concurrent requests with same key
- KernelRegistry: Template scanning and registration
- instantiate(): The main interface for instance caching
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .stable_key import StableKey, compute_stable_key
from .kernel import (
    KernelRegistry,
    VKernelDescriptor,
    select_kernel,
    AnyType,
)
from .types import (
    MemorySpace,
    TileSpecialization,
    TileConfig,
)
import tilelang_dsl as pto


@dataclass
class InstanceCache:
    """In-process instance cache with FIFO eviction.

    Attributes:
        max_entries: Maximum number of cached instances
        cache: Dict mapping StableKey to MLIR text
        pending_requests: Dict for single-flight (in-progress requests)
        registry: KernelRegistry containing scanned templates
        stats: Statistics for hits/misses/evictions
    """
    max_entries: int = 1000
    cache: dict[StableKey, str] = field(default_factory=dict)
    pending_requests: dict[StableKey, asyncio.Future] = field(default_factory=dict)
    registry: KernelRegistry | None = None
    stats: dict[str, int] = field(default_factory=lambda: {
        "hits": 0,
        "misses": 0,
        "evictions": 0,
    })

    def scan_template_directory(self, template_dir: Path) -> None:
        """Scan template directory and register all @vkernel descriptors.

        Args:
            template_dir: Path to directory containing .py template files
        """
        self.registry = KernelRegistry()

        if not template_dir.is_dir():
            return

        for py_file in sorted(template_dir.glob("*.py")):
            descriptors = _import_and_find_descriptors(py_file)
            for desc in descriptors:
                self.registry.register(desc)

    async def instantiate(
        self,
        target: str,
        op: str,
        operand_specs: list[dict],
        context_attrs: dict[str, Any] | None = None,
    ) -> str:
        """Instantiate a template and return MLIR text.

        This is the main interface with caching and single-flight.

        Args:
            target: Target architecture ("a5", "a3")
            op: Operator name ("tadd", "tmul", etc.)
            operand_specs: List of operand spec dicts
            context_attrs: Additional context attributes

        Returns:
            Materialized MLIR text
        """
        # 1. Compute stable key
        key = compute_stable_key(target, op, operand_specs, context_attrs)

        # 2. Check cache
        if key in self.cache:
            self.stats["hits"] += 1
            return self.cache[key]

        # 3. Single-flight: check if same key is being instantiated
        if key in self.pending_requests:
            # Wait for the in-progress request to complete
            self.stats["hits"] += 1  # Count as hit since we reuse result
            return await self.pending_requests[key]

        # 4. Create future for single-flight
        future: asyncio.Future[str] = asyncio.Future()
        self.pending_requests[key] = future

        try:
            # 5. Instantiate (cache miss)
            self.stats["misses"] += 1
            mlir_text = await self._do_instantiate(
                target, op, operand_specs, context_attrs
            )

            # 6. Store in cache with FIFO eviction
            self._store_with_eviction(key, mlir_text)

            # 7. Complete future (notify all waiters)
            future.set_result(mlir_text)

            return mlir_text

        except Exception as e:
            # On error, notify all waiters with exception
            future.set_exception(e)
            raise

        finally:
            # 8. Clean up pending state
            if key in self.pending_requests:
                del self.pending_requests[key]

    async def _do_instantiate(
        self,
        target: str,
        op: str,
        operand_specs: list[dict],
        context_attrs: dict[str, Any] | None = None,
    ) -> str:
        """Actual instantiation logic (without caching).

        Args:
            target: Target architecture
            op: Operator name
            operand_specs: Operand specs
            context_attrs: Context attributes

        Returns:
            Materialized MLIR text
        """
        if self.registry is None:
            raise RuntimeError("InstanceCache not initialized: scan_template_directory() first")

        # 1. Build operand_types for select_kernel
        operand_types = tuple(
            _convert_dtype_str_to_scalar(spec.get("dtype"))
            for spec in operand_specs
        )

        # 2. Build context_attrs for constraint evaluation
        context = _build_positional_context_attrs(operand_specs)
        if context_attrs:
            context.update(context_attrs)

        # 3. Filter descriptors by operand schema
        filtered_descriptors = _filter_descriptors_by_operand_schema(
            list(self.registry),
            target=target,
            op_name=op,
            operand_specs=operand_specs,
        )

        if not filtered_descriptors:
            raise LookupError(
                f"No kernel found for target={target!r}, op={op!r}, "
                f"operand_types={operand_types!r}"
            )

        # 4. Create filtered registry and select kernel
        filtered_registry = KernelRegistry(tuple(filtered_descriptors))
        descriptor = select_kernel(
            target,
            op,
            operand_types,
            context_attrs=context,
            registry=filtered_registry,
            return_metadata=False,
        )

        # 5. Build TileSpecialization bindings
        tile_bindings = {}
        for param, spec in zip(descriptor.parameters, operand_specs):
            if param.kind == "tile":
                memory_space = spec.get("memory_space", "ub")
                if isinstance(memory_space, str):
                    memory_space = MemorySpace(memory_space.lower())

                config = spec.get("config")
                if config is not None and not isinstance(config, TileConfig):
                    config = TileConfig.from_mapping(config)

                tile_bindings[param.name] = TileSpecialization(
                    shape=tuple(spec.get("shape", [])),
                    memory_space=memory_space,
                    config=config,
                    valid_shape=tuple(spec.get("valid_shape", [])) if spec.get("valid_shape") else None,
                )

        # 6. Specialize descriptor
        specialized = descriptor.specialize(**tile_bindings)

        # 7. Materialize MLIR
        mlir_text = specialized.mlir_text()

        return mlir_text

    def _store_with_eviction(self, key: StableKey, mlir_text: str) -> None:
        """Store in cache with FIFO eviction.

        Args:
            key: StableKey
            mlir_text: MLIR text to cache
        """
        # Check if cache is full
        if len(self.cache) >= self.max_entries:
            # FIFO: remove the oldest entry (first inserted)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            self.stats["evictions"] += 1

        # Store new entry
        self.cache[key] = mlir_text

    def get_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        return {
            "total_entries": len(self.cache),
            "max_entries": self.max_entries,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "evictions": self.stats["evictions"],
            "hit_rate": (
                self.stats["hits"] / (self.stats["hits"] + self.stats["misses"])
                if (self.stats["hits"] + self.stats["misses"]) > 0
                else 0.0
            ),
        }

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.pending_requests.clear()


# ============================================================================
# Helper functions (adapted from expand_helper.py)
# ============================================================================

def _import_and_find_descriptors(py_file: Path) -> list[VKernelDescriptor]:
    """Import a .py file and find all @vkernel descriptors.

    Args:
        py_file: Path to Python file

    Returns:
        List of VKernelDescriptor objects
    """
    template_parent = py_file.parent.parent
    if str(template_parent) not in sys.path:
        sys.path.insert(0, str(template_parent))

    module_name = f"_tl_template_{py_file.stem}"
    spec = importlib.util.spec_from_file_location(module_name, str(py_file))
    if spec is None or spec.loader is None:
        return []

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        return []

    descriptors = []
    for attr_name in dir(module):
        obj = getattr(module, attr_name, None)
        if isinstance(obj, VKernelDescriptor):
            descriptors.append(obj)

    return descriptors


def _build_positional_context_attrs(operand_specs: list[dict]) -> dict[str, Any]:
    """Build positional context attrs for constraint evaluation.

    Args:
        operand_specs: List of operand spec dicts

    Returns:
        Dict of context attrs (arg0_shape, arg0_strides, etc.)
    """
    attrs: dict[str, Any] = {}
    for index, spec in enumerate(operand_specs):
        prefix = f"arg{index}"
        attrs[f"{prefix}_kind"] = spec.get("kind")
        dtype = spec.get("dtype")
        if isinstance(dtype, str):
            dtype = _convert_dtype_str_to_scalar(dtype)
        attrs[f"{prefix}_dtype"] = dtype

        if spec.get("kind") == "scalar":
            if "value" in spec:
                attrs[f"{prefix}_value"] = spec["value"]
            continue

        shape = spec.get("shape")
        if shape is not None:
            attrs[f"{prefix}_shape"] = tuple(shape)
            attrs[f"{prefix}_rank"] = len(shape)

        memory_space = spec.get("memory_space")
        if memory_space is not None:
            attrs[f"{prefix}_memory_space"] = memory_space

        if spec.get("kind") == "tile":
            valid_shape = spec.get("valid_shape")
            if valid_shape is not None:
                attrs[f"{prefix}_valid_shape"] = tuple(valid_shape)
            config = spec.get("config")
            if config is not None:
                attrs[f"{prefix}_config"] = config

        if spec.get("kind") == "view" and "strides" in spec:
            attrs[f"{prefix}_strides"] = tuple(spec["strides"])

    return attrs


def _operand_spec_matches_param_kind(param_kind: str, operand_kind: str) -> bool:
    """Check if operand kind matches parameter kind.

    Args:
        param_kind: Parameter kind from descriptor
        operand_kind: Operand kind from spec

    Returns:
        True if matches
    """
    if operand_kind == "tile":
        return param_kind == "tile"
    if operand_kind == "view":
        return param_kind in ("tensorview", "partition_tensor_view")
    if operand_kind == "scalar":
        return param_kind == "scalar"
    return False


def _convert_dtype_str_to_scalar(dtype_str: str) -> Any:
    """Convert dtype string to ScalarType object.
    
    Args:
        dtype_str: Dtype string (e.g. 'f32', 'i16')
    
    Returns:
        ScalarType object or original string if not found
    """
    # Map common dtype strings to ScalarType
    dtype_map = {
        'f16': pto.f16,
        'f32': pto.f32,
        'bf16': pto.bf16,
        'i8': pto.i8,
        'si8': pto.si8,
        'i16': pto.i16,
        'si16': pto.si16,
        'i32': pto.i32,
        'si32': pto.si32,
        'i64': pto.i64,
        'si64': pto.si64,
        'ui8': pto.ui8,
        'ui16': pto.ui16,
        'ui32': pto.ui32,
        'ui64': pto.ui64,
    }
    return dtype_map.get(dtype_str.lower(), dtype_str)


def _filter_descriptors_by_operand_schema(
    descriptors: list[VKernelDescriptor],
    *,
    target: str,
    op_name: str,
    operand_specs: list[dict],
) -> list[VKernelDescriptor]:
    """Filter descriptors by operand schema (kind + dtype).

    Args:
        descriptors: List of descriptors to filter
        target: Target architecture
        op_name: Operator name
        operand_specs: Operand specs

    Returns:
        Filtered list of descriptors
    """
    operand_types = tuple(
        _convert_dtype_str_to_scalar(spec.get("dtype"))
        for spec in operand_specs
    )
    filtered: list[VKernelDescriptor] = []

    for descriptor in descriptors:
        # Check target and op match
        if descriptor.target != target:
            continue
        if op_name not in descriptor.match_ops:
            continue

        # Try to bind dtype signature
        try:
            # Bind selected op first
            bound = descriptor._bind_selected_op(op_name)
            # Then try to match dtype signature
            matched = None
            for sig in bound.dtypes:
                if len(sig) != len(operand_types):
                    continue
                if all(_dtype_matches(sig[i], operand_types[i]) for i in range(len(sig))):
                    matched = sig
                    break
            if matched is None:
                continue
            
            # IMPORTANT: Use actual operand_types for binding, not signature
            # e.g., bind (f32, f32) instead of (AnyType, AnyType)
            bound = bound._bind_selected_dtype_signature(operand_types)

            # Check parameter kinds match operand kinds
            parameters = bound.parameters
            if len(parameters) != len(operand_specs):
                continue

            if all(
                _operand_spec_matches_param_kind(param.kind, spec.get("kind"))
                for param, spec in zip(parameters, operand_specs)
            ):
                filtered.append(bound)
        except Exception:
            continue

    return filtered


def _dtype_matches(dtype_sig: Any, dtype_spec: Any) -> bool:
    """Check if dtype from signature matches dtype from spec.

    Args:
        dtype_sig: Dtype from signature (ScalarType, AnyType, or type var)
        dtype_spec: Dtype from spec (ScalarType, string, or type)

    Returns:
        True if matches
    """
    # AnyType matches any dtype (wildcard)
    if dtype_sig == AnyType:
        return True
    
    # Normal matching logic
    if hasattr(dtype_sig, "name") and hasattr(dtype_spec, "name"):
        return dtype_sig.name.lower() == dtype_spec.name.lower()
    if hasattr(dtype_sig, "name"):
        return dtype_sig.name.lower() == str(dtype_spec).lower()
    return str(dtype_sig).lower() == str(dtype_spec).lower()
