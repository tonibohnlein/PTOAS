# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Acquire the ``ptoas._core`` native module.

Fast path: import the prebuilt abi3 ``_core`` shipped in the wheel. Fallback: on
ImportError (interpreter/ABI mismatch), compile a matching ``_core`` online from
the shipped sources under ``ptoas/_online/`` and load it from the build cache.

For the cache case the online ``_core`` lives outside the package dir, so its
``$ORIGIN`` rpath does NOT point at the shipped ``libPTOASCompiler`` DSO. We
therefore preload the DSO with ``RTLD_GLOBAL`` first so ``_core`` resolves the
compiler symbols regardless of where it was cached.
"""

import ctypes
import logging
from pathlib import Path
import threading
from typing import Optional

_log = logging.getLogger(__name__)

_QUALIFIED_MODULE = "ptoas._core"
_DSO_STEMS = ("libPTOASCompiler",)

_ensure_lock = threading.Lock()
_ensured_module = None


def _package_dir() -> Path:
    return Path(__file__).parent.resolve()


def _find_dso_in_dir(candidate: Path) -> Optional[Path]:
    """Return the first shipped DSO directly under ``candidate``, if any."""
    if not candidate.is_dir():
        return None
    for stem in _DSO_STEMS:
        for ext in (".so", ".dylib"):
            f = candidate / f"{stem}{ext}"
            if f.exists():
                return f
    return None


def _find_dso(pkg_dir: Path) -> Optional[Path]:
    candidates = [pkg_dir, pkg_dir / "mlir" / "_mlir_libs"]
    for cand in candidates:
        found = _find_dso_in_dir(cand)
        if found is not None:
            return found
    return None


def _load_shared_libs():
    """Preload the shipped, Python-independent DSO with RTLD_GLOBAL."""
    dso = _find_dso(_package_dir())
    if dso is None:
        _log.debug("libPTOASCompiler DSO not found for preload; relying on rpath")
        return
    try:
        ctypes.CDLL(str(dso), mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        _log.warning("Failed to preload %s: %s", dso, e)


def ensure_core():
    """Return the ``ptoas._core`` module, using the abi3 fast path when possible.

    The result is also registered as ``sys.modules['ptoas._core']`` so that
    ``from ptoas import _core`` works uniformly for both the prebuilt and the
    online-compiled module.
    """
    global _ensured_module
    if _ensured_module is not None:
        return _ensured_module
    with _ensure_lock:
        if _ensured_module is not None:
            return _ensured_module

        _load_shared_libs()
        try:
            import ptoas._core as core  # prebuilt abi3 fast path

            _ensured_module = core
            return core
        except ImportError as fast_path_error:
            _log.info(
                "Prebuilt ptoas._core unavailable for this interpreter (%s); "
                "falling back to online compilation.",
                fast_path_error,
            )

        from ._build_online import get_or_build_core

        _ensured_module = get_or_build_core()
        return _ensured_module
