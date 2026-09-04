# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Standalone runtime recipe for the online-compiled `ptoas._core` fallback.
#
# This file is staged into the wheel at `ptoas/_online/CMakeLists.txt`. It is
# invoked at first import (by ptoas/_build_online.py) against the shipped,
# Python-version-independent DSO (libPTOASCompiler) that lives in the installed
# `ptoas/` package, to produce a `_core` extension matching the running
# interpreter's ABI.
#
# The Python driver passes these cache variables via -D:
#   Python3_EXECUTABLE            - the target interpreter
#   Python3_EXECUTABLE_VERSION    - "3.<minor>" for an EXACT find_package
#   Python3_MOD_PYBIND11_CMAKE_DIR - pybind11.get_cmake_dir() of that interpreter
#   PTOAS_SHIPPED_LIB_DIR         - dir holding libPTOASCompiler (the package dir)

cmake_minimum_required(VERSION 3.20)
project(ptoas_online_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

find_package(Python3 ${Python3_EXECUTABLE_VERSION} EXACT REQUIRED
  COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED
  PATHS ${Python3_MOD_PYBIND11_CMAKE_DIR} NO_DEFAULT_PATH)

pybind11_add_module(_core
  ${CMAKE_CURRENT_SOURCE_DIR}/src/PTOModule.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/NativeModuleOnline.cpp)

target_compile_definitions(_core PRIVATE
  PTOAS_ONLINE_BUILD=1
  MLIR_PYTHON_PACKAGE_PREFIX=ptoas.mlir.)

target_include_directories(_core PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link against the shipped, Python-independent DSO living in the package dir.
target_link_directories(_core PRIVATE ${PTOAS_SHIPPED_LIB_DIR})
target_link_libraries(_core PRIVATE PTOASCompiler)

# The cached `_core` is loaded from a directory that is NOT necessarily the
# package dir (see _build_online._compute_cache_dir). ctypes preloads the DSO
# with RTLD_GLOBAL before importing `_core`, so these rpaths are a best-effort
# fallback for the in-package cache case.
if(APPLE)
  set_target_properties(_core PROPERTIES INSTALL_RPATH
    "@loader_path;@loader_path/mlir/_mlir_libs")
else()
  set_target_properties(_core PROPERTIES INSTALL_RPATH
    "$ORIGIN;$ORIGIN/mlir/_mlir_libs")
endif()

install(TARGETS _core LIBRARY DESTINATION ".")
