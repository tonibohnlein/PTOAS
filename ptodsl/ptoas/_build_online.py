#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Online-compilation infrastructure for the ``ptoas._core`` pybind11 extension.

This module must NOT import ``ptoas._core`` (or anything that depends on it) so
that it stays importable when no matching ``_core`` binary exists yet: its whole
job is to produce one. The Python-version-independent ``libPTOASCompiler`` DSO is
shipped prebuilt inside the wheel; only the version-sensitive ``_core`` extension
is (re)compiled here against the shipped C-API-only source + header closure under
``ptoas/_online/``.
"""

import dataclasses
import fcntl
import importlib.machinery
import importlib.util
import logging
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
from typing import Callable, List, Optional, Tuple

_log = logging.getLogger(__name__)

_MODULE_NAME = "_core"
_QUALIFIED_MODULE = "ptoas._core"
_DSO_STEMS = ("libPTOASCompiler",)


def _package_dir() -> Path:
    """Directory of the installed (or editable) ``ptoas`` package."""
    return Path(__file__).parent.resolve()


def _lib_dir_contains_shipped_dso(candidate: Path) -> bool:
    """True if any shipped DSO (libPTOASCompiler.{so,dylib}) sits in candidate."""
    if not candidate.is_dir():
        return False
    for stem in _DSO_STEMS:
        for ext in (".so", ".dylib"):
            if (candidate / f"{stem}{ext}").exists():
                return True
    return False


def _find_shipped_lib_dir(pkg_dir: Path) -> Optional[Path]:
    """Locate the directory holding libPTOASCompiler.{so,dylib}.

    The DSO is installed alongside the prebuilt ``_core`` (package dir) with the
    MLIR shared libs; probe the package dir and the standard MLIR libs subdir.
    """
    candidates = [pkg_dir, pkg_dir / "mlir" / "_mlir_libs"]
    for cand in candidates:
        if _lib_dir_contains_shipped_dso(cand):
            return cand
    return None


class _CMakeContext:
    """Locate cmake and drive its configure/build/install phases."""

    @dataclasses.dataclass
    class CompileContext:
        src_dir: Path
        tmp_dir: Path
        install_prefix: Path
        cfg_cmd_ext: str = ""
        build_type: str = "Release"
        build_job_num: int = 32
        capture_output: bool = True

        def run_cmd(self, cmd: str):
            ret = subprocess.run(
                shlex.split(cmd),
                text=True,
                encoding="utf-8",
                capture_output=self.capture_output,
                check=not self.capture_output,
            )
            if ret.returncode != 0 and self.capture_output:
                _log.error("cmd: %s, ret: %s", cmd, ret.returncode)
                _log.error("stdout:\n%s", ret.stdout)
                _log.error("stderr:\n%s", ret.stderr)
            ret.check_returncode()

    def __init__(self):
        self.cmake = self._which_cmake()
        if self.cmake is None:
            raise RuntimeError(
                "Can not find cmake, please check your environment.\n"
                "Hint: install system-level cmake (e.g. `apt install cmake` or `yum install cmake`), "
                "the cmake pip package is NOT a valid substitute."
            )

    @staticmethod
    def _which_cmake() -> Optional[Path]:
        from ._which_cmake import which_cmake

        return which_cmake()

    def compile(self, ctx: "CompileContext") -> Path:
        build_dir = Path(ctx.tmp_dir, "build")
        if build_dir.exists():
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True)
        cmd = f"{self.cmake} -S {ctx.src_dir} -B {build_dir} -DCMAKE_BUILD_TYPE={ctx.build_type}"
        cmd += f" -DCMAKE_INSTALL_PREFIX={ctx.install_prefix} {ctx.cfg_cmd_ext}"
        ctx.run_cmd(cmd=cmd)
        cmd = f"{self.cmake} --build {build_dir}" + (f" -j {ctx.build_job_num}" if ctx.build_job_num else "")
        ctx.run_cmd(cmd=cmd)
        cmd = f"{self.cmake} --install {build_dir} --prefix {ctx.install_prefix}"
        ctx.run_cmd(cmd=cmd)
        return ctx.install_prefix


class _PythonContext:
    """Validate the target interpreter and locate its pybind11 cmake package."""

    _PYBIND11_MIN_VERSION = (2, 13, 6)

    def __init__(self):
        self.minor: int = 0
        self.pybind11_cmake_dir: Optional[Path] = None
        self._init_minor_version()
        self._init_development_component()
        self._init_pip_mod_pybind11()

    @staticmethod
    def _init_development_component():
        python_h = Path(sysconfig.get_path("include")) / "Python.h"
        if not python_h.exists():
            raise RuntimeError(
                f"Python development headers not found (expected {python_h}).\n"
                "Hint: install python3-dev (e.g. `apt install python3-dev` or `yum install python3-devel`)."
            )

    def _init_minor_version(self):
        minor = int(sys.version_info.minor)
        if minor < 9:
            raise RuntimeError(
                f"Python version 3.{minor} is not supported for online compilation, require >= 3.9.\n"
                "Hint: use a Python 3.9+ interpreter."
            )
        self.minor = minor

    def _init_pip_mod_pybind11(self):
        try:
            import pybind11
        except ImportError as e:
            raise RuntimeError(
                "pybind11 pip package not found.\nHint: install it with `pip install pybind11>=2.13.6`."
            ) from e
        ver_match = re.match(r"(\d+)\.(\d+)\.(\d+)", pybind11.__version__)
        if not ver_match:
            raise RuntimeError(
                f"Can't parse pybind11 version: {pybind11.__version__}.\n"
                "Hint: install it with `pip install pybind11>=2.13.6`."
            )
        current_ver = tuple(int(x) for x in ver_match.groups())
        if current_ver < self._PYBIND11_MIN_VERSION:
            raise RuntimeError(
                f"pybind11 version {pybind11.__version__} is too old, require >= 2.13.6.\n"
                "Hint: upgrade it with `pip install pybind11>=2.13.6`."
            )
        pybind11_dir = Path(pybind11.get_cmake_dir()).resolve()
        if not pybind11_dir or not pybind11_dir.exists():
            raise RuntimeError("pybind11 cmake dir empty.\nHint: install it with `pip install pybind11>=2.13.6`.")
        self.pybind11_cmake_dir = pybind11_dir


class BuildOnlineCoreManager:
    """Process-wide singleton that (re)builds and loads ``ptoas._core``.

    Fast path callers should try ``import ptoas._core`` first; this manager is
    the fallback taken on ImportError (interpreter/ABI mismatch against the
    prebuilt abi3 module).
    """

    _instances: dict = {}
    _new_lock: threading.Lock = threading.Lock()
    _compile_lock: threading.Lock = threading.Lock()
    _FLOCK_TIMEOUT: int = 300  # seconds a process waits for a peer's compile

    def __new__(cls):
        if cls not in cls._instances:
            with cls._new_lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__new__(cls)
        return cls._instances[cls]

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        from importlib import metadata

        self.pkg_dir: Path = _package_dir()
        try:
            self.version: str = metadata.version("ptoas")
        except Exception:
            self.version = ""
        self._online_dir: Path = self.pkg_dir / "_online"
        self._loaded_module = None
        self._cache_dir_value: Optional[Path] = None
        ver_info = sys.version_info
        self._lock_name: str = f".ptoas_core_build.cp{ver_info.major}{ver_info.minor}.lock"
        self._initialized = True

    # -- static helpers ---------------------------------------------------

    @staticmethod
    def _try_acquire_lock(lock_fd) -> bool:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False

    # -- public entry ------------------------------------------------------

    def get_or_build(self):
        """Return the loaded ``_core`` module, compiling it if necessary."""
        if self._loaded_module is not None:
            return self._loaded_module
        with self._compile_lock:
            if self._loaded_module is not None:
                return self._loaded_module
            self._ensure_core_locked()
        return self._loaded_module

    # -- core state machine ------------------------------------------------

    def _ensure_core_locked(self):
        found, so_path = self._find_core_so(cache_dir=self._cache_dir())
        if found:
            self._load_from_cache(so_path=so_path)
            return

        if not self._online_dir.exists():
            raise RuntimeError(
                "ptoas._core is unavailable for this interpreter and no online "
                f"sources were shipped (expected {self._online_dir}).\n"
                "Hint: install a wheel built with PTOAS_ENABLE_ONLINE_CORE_COMPILE=ON, "
                "or use the matching prebuilt interpreter."
            )

        # Cross-user cooperation: if someone is compiling into pkg_dir, wait.
        if self._wait_for_pkg_dir_compilation():
            return

        target_dir = self._cache_dir()
        lock_path = target_dir / self._lock_name
        lock_fd = None
        try:
            lock_fd = open(lock_path, "w")
            if self._try_acquire_lock(lock_fd):
                so_path = self._do_compile(target_dir=target_dir)
            else:
                so_path = self._wait_and_compile(lock_fd, target_dir, lock_path)

            if so_path is None:
                raise RuntimeError(
                    "Online compilation succeeded but _core not found in install output.\n"
                    f"Target directory: {target_dir}\n"
                    f"Searched suffixes: {importlib.machinery.EXTENSION_SUFFIXES}"
                )
            self._load_from_cache(so_path=so_path)
        finally:
            # Unlink while still holding the lock to avoid an inode-reuse race
            # (see the pypto original for the detailed reasoning).
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                pass
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                except OSError:
                    pass
                lock_fd.close()

    def _poll_until(
        self, predicate: Callable[[], bool], timeout: Optional[float] = None, interval: float = 1.0
    ) -> bool:
        if timeout is None:
            timeout = self._FLOCK_TIMEOUT
        deadline = time.monotonic() + timeout
        while True:
            if predicate():
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(interval)

    def _wait_for_pkg_dir_compilation(self) -> bool:
        lock_path = self.pkg_dir / self._lock_name
        if not os.access(self.pkg_dir, os.R_OK):
            return False
        if not lock_path.exists():
            return False
        _log.info("Detected compilation in progress at %s, waiting...", lock_path)
        if not self._poll_until(lambda: not lock_path.exists()):
            _log.warning("Timeout waiting for pkg_dir compilation")
            return False
        found, so_path = self._find_core_so()
        if found:
            self._load_from_cache(so_path=so_path)
            return True
        return False

    def _wait_and_compile(self, lock_fd, target_dir: Path, lock_path: Path) -> Optional[Path]:
        _log.info("Waiting for compilation by another process (lock: %s)...", lock_path)
        if self._poll_until(lambda: self._try_acquire_lock(lock_fd)):
            _log.info("Acquired compilation lock after waiting")
            found, so_path = self._find_core_so(cache_dir=target_dir)
            if found:
                return so_path
            return self._do_compile(target_dir=target_dir)
        _log.warning("Timeout waiting for compilation lock, will compile independently")
        return self._do_compile(target_dir=target_dir)

    def _do_compile(self, target_dir: Path) -> Optional[Path]:
        pyenv_ctx = _PythonContext()
        cmake_ctx = _CMakeContext()

        shipped_lib_dir = _find_shipped_lib_dir(self.pkg_dir)
        if shipped_lib_dir is None:
            raise RuntimeError(
                "Can not locate the shipped libPTOASCompiler DSO under "
                f"{self.pkg_dir}; the wheel appears to be incomplete."
            )

        with tempfile.TemporaryDirectory(prefix=f".ptoas_core_build.{os.getpid()}.") as tmp_dir:
            ext = f"-DPython3_EXECUTABLE={sys.executable}"
            ext += f" -DPython3_EXECUTABLE_VERSION=3.{pyenv_ctx.minor}"
            ext += f" -DPython3_MOD_PYBIND11_CMAKE_DIR={pyenv_ctx.pybind11_cmake_dir}"
            ext += f" -DPTOAS_SHIPPED_LIB_DIR={shipped_lib_dir}"
            compile_ctx = _CMakeContext.CompileContext(
                src_dir=self._online_dir,
                tmp_dir=Path(tmp_dir),
                install_prefix=target_dir,
                cfg_cmd_ext=ext,
            )
            cmake_ctx.compile(ctx=compile_ctx)

        _log.info("Compiled and installed _core to %s", target_dir)
        found, so_path = self._find_core_so(cache_dir=target_dir)
        return so_path if found else None

    # -- cache dir + discovery --------------------------------------------

    def _cache_dir(self) -> Path:
        if self._cache_dir_value is not None:
            return self._cache_dir_value
        self._cache_dir_value = self._compute_cache_dir()
        return self._cache_dir_value

    def _compute_cache_dir(self) -> Path:
        # Prefer the package dir when it belongs to us and is actually writable.
        try:
            if self.pkg_dir.stat().st_uid == os.getuid():
                test_file = self.pkg_dir / f".ptoas_writable_test.{os.getpid()}"
                try:
                    test_file.touch()
                    test_file.unlink()
                    return self.pkg_dir
                except OSError:
                    pass
        except OSError:
            pass

        cache_dir = None
        xdg_cache = os.environ.get("XDG_CACHE_HOME")
        if xdg_cache:
            if Path(xdg_cache).is_absolute():
                cache_dir = Path(xdg_cache)
            else:
                _log.warning("XDG_CACHE_HOME=%s is not absolute, fallback to ~/.cache", xdg_cache)
        cache_dir = cache_dir if cache_dir else Path.home() / ".cache"
        cache_dir = cache_dir / "cann" / "ptoas"
        if self.version:
            cache_dir = cache_dir / self.version
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _find_core_so(self, cache_dir: Optional[Path] = None) -> Tuple[bool, Optional[Path]]:
        search_dirs: List[Path] = [self.pkg_dir]
        if cache_dir is not None and cache_dir != self.pkg_dir:
            search_dirs.append(cache_dir)
        for d in search_dirs:
            for suffix in importlib.machinery.EXTENSION_SUFFIXES:
                so_path = d / f"{_MODULE_NAME}{suffix}"
                if so_path.exists():
                    _log.info("Found _core: %s", so_path)
                    return True, so_path
        return False, None

    def _load_from_cache(self, so_path: Path):
        if _QUALIFIED_MODULE in sys.modules:
            self._loaded_module = sys.modules[_QUALIFIED_MODULE]
            return

        spec = importlib.util.spec_from_file_location(_QUALIFIED_MODULE, so_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to create import spec for {so_path}")

        module = importlib.util.module_from_spec(spec)
        # Register BEFORE exec_module: pybind11 module init may re-enter import.
        sys.modules[_QUALIFIED_MODULE] = module
        import ptoas

        setattr(ptoas, _MODULE_NAME, module)

        spec.loader.exec_module(module)
        _log.info("Loaded _core from %s", so_path)
        self._loaded_module = module


def get_or_build_core():
    return BuildOnlineCoreManager().get_or_build()
