# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Do not carry toolchain-injected paths, such as a Conda compiler RPATH, into
# installed Python modules. Distribution builds keep only package-relative
# RPATHs; editable builds add the external LLVM directory explicitly below.
set(CMAKE_INSTALL_REMOVE_ENVIRONMENT_RPATH ON)

set(PTOAS_RUNTIME_SOURCES
  ptoas.cpp
  ptoas_name_hints.cpp
  ptoas_cpp_rewrite.cpp
  ptoas_pipeline.cpp
  driver.cpp
  VPTOHostStubEmission.cpp
  ObjectEmission.cpp
)

add_library(PTOASVFSIMTSizePatcher STATIC
  VFSIMTSizePatcher.cpp
)
llvm_update_compile_flags(PTOASVFSIMTSizePatcher)
target_link_libraries(PTOASVFSIMTSizePatcher PUBLIC
  LLVMCore
  LLVMObject
  LLVMSupport
  MLIRSupport
)

set(PTOAS_WRAPPER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/ptoas_wrapper.py")
set(PTOAS_WRAPPER_BINARY "${CMAKE_CURRENT_BINARY_DIR}/ptoas")
set(PTOAS_INSTALL_WRAPPER_BINARY "${CMAKE_CURRENT_BINARY_DIR}/ptoas-install")

set(PTOAS_WRAPPER_PYTHON_ROOT_MODE "absolute")
set(PTOAS_WRAPPER_PYTHON_ROOT "${CMAKE_BINARY_DIR}/python")
configure_file(
  "${PTOAS_WRAPPER_SOURCE}"
  "${PTOAS_WRAPPER_BINARY}"
  @ONLY
)

# CMake install prefixes may be overridden at install time. Resolve the
# installed package relative to bin/ptoas instead of baking in the configure
# prefix.
set(PTOAS_WRAPPER_PYTHON_ROOT_MODE "wrapper-relative")
set(PTOAS_WRAPPER_PYTHON_ROOT "..")
configure_file(
  "${PTOAS_WRAPPER_SOURCE}"
  "${PTOAS_INSTALL_WRAPPER_BINARY}"
  @ONLY
)

foreach(_ptoas_wrapper IN ITEMS
    "${PTOAS_WRAPPER_BINARY}"
    "${PTOAS_INSTALL_WRAPPER_BINARY}")
  file(CHMOD "${_ptoas_wrapper}"
    PERMISSIONS
      OWNER_READ OWNER_WRITE OWNER_EXECUTE
      GROUP_READ GROUP_EXECUTE
      WORLD_READ WORLD_EXECUTE
  )
endforeach()

function(ptoas_configure_runtime_compile_target target_name)
  target_compile_definitions(${target_name} PRIVATE
    PTOAS_RELEASE_VERSION="${PTOAS_CLI_VERSION}"
    ${ARGN}
  )
  add_dependencies(${target_name}
    PTOOpsIncGen
  )
endfunction()

set(PTOAS_RUNTIME_LINK_LIBS
  PTOIR
  PTOTransforms
  PTOASVFSIMTSizePatcher
  ptobc_lib
  MLIRMlirOptMain
  MLIRBufferizationTransforms
  MLIRArithTransforms
  MLIRTensorTransforms
  MLIRFuncTransforms
  MLIRIR
  MLIRParser
  MLIRPass
  MLIREmitCDialect
  MLIRSCFDialect
  MLIRGPUDialect
  MLIRTransforms
  MLIRTargetCpp
  MLIRAffineDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRControlFlowDialect
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRFuncInlinerExtension
  MLIRLLVMIRTransforms
  MLIRSupport
  LLVMOption
  MLIRTransformUtils
)

# This aggregation root owns all compiler and PTO Python binding objects. The
# final shared library is declared earlier as PTOASCompiler; AddMLIR resolves
# this forward target relationship during generation.
add_mlir_library(PTOASCompilerImplementation
  OBJECT
  ENABLE_AGGREGATION
  EXCLUDE_FROM_LIBMLIR
  DISABLE_INSTALL
  PARTIAL_SOURCES_INTENDED

  ${PTOAS_RUNTIME_SOURCES}

  LINK_LIBS PUBLIC
  ${PTOAS_RUNTIME_LINK_LIBS}
)
ptoas_configure_runtime_compile_target(obj.PTOASCompilerImplementation)
target_compile_definitions(obj.PTOASCompilerImplementation PRIVATE
  "MLIR_PYTHON_PACKAGE_PREFIX=ptoas.mlir."
  PTOAS_COMPILER_BUILDING_LIBRARY=1
)
target_include_directories(PTOASCompilerImplementation PRIVATE
  "${CMAKE_SOURCE_DIR}/lib/Bindings/Python"
)

set_target_properties(PTOASCompiler PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
)

# Build the single PTOAS-owned Python extension as a thin entry point. Compiler,
# PTO dialect, and MLIR CAPI implementation all live in PTOASCompiler.
set(PTOAS_BUILD_PYTHON_PACKAGE_DIR "${CMAKE_BINARY_DIR}/python/ptoas")
add_mlir_python_extension(PTOASPythonCore _core
  INSTALL_COMPONENT PTOAS_Python
  INSTALL_DIR ptoas
  OUTPUT_DIRECTORY "${PTOAS_BUILD_PYTHON_PACKAGE_DIR}"
  SOURCES
    NativeModule.cpp
    "${CMAKE_SOURCE_DIR}/lib/Bindings/Python/PTOModule.cpp"
  LINK_LIBS
    PTOASCompiler
)
target_compile_definitions(PTOASPythonCore PRIVATE
  "MLIR_PYTHON_PACKAGE_PREFIX=ptoas.mlir."
)
# PTOModule.cpp now defines populatePTODialectBindings inside _core itself
# rather than importing it from the DSO, so mark the defining TU as the
# library-building side (dllexport on Windows; default visibility elsewhere).
set_property(SOURCE "${CMAKE_SOURCE_DIR}/lib/Bindings/Python/PTOModule.cpp"
  APPEND PROPERTY COMPILE_DEFINITIONS PTOAS_COMPILER_BUILDING_LIBRARY=1)
target_include_directories(PTOASPythonCore PRIVATE
  "${CMAKE_SOURCE_DIR}/lib/Bindings/Python"
)
mlir_python_setup_extension_rpath(PTOASPythonCore)

if(APPLE)
  set_property(TARGET PTOASPythonCore PROPERTY INSTALL_RPATH
    "@loader_path;@loader_path/mlir/_mlir_libs")
elseif(UNIX)
  set_property(TARGET PTOASPythonCore PROPERTY INSTALL_RPATH
    "$ORIGIN;$ORIGIN/mlir/_mlir_libs")
endif()
ptoas_append_editable_llvm_rpath(PTOASPythonCore)

declare_mlir_python_sources(PTOASRuntimePythonSources
  ROOT_DIR "${CMAKE_SOURCE_DIR}/ptodsl/ptoas"
  SOURCES_GLOB "*.py"
)
add_mlir_python_sources_target(PTOASRuntimePythonSourcesStage
  OUTPUT_DIRECTORY "${PTOAS_BUILD_PYTHON_PACKAGE_DIR}"
  SOURCES_TARGETS PTOASRuntimePythonSources
)

add_dependencies(PTOASRuntimePythonSourcesStage PTODSLPackageStage)

declare_mlir_python_sources(PTOASTileOpsPythonSources
  ROOT_DIR "${CMAKE_SOURCE_DIR}/lib/TileOps"
  SOURCES_GLOB "*.py"
)
add_mlir_python_sources_target(PTOASTileOpsPythonSourcesStage
  OUTPUT_DIRECTORY
  "${PTOAS_BUILD_PYTHON_PACKAGE_DIR}/_runtime/share/ptoas/TileOps"
  SOURCES_TARGETS PTOASTileOpsPythonSources
)

declare_mlir_python_sources(PTOASSoftOpsPythonSources
  ROOT_DIR "${CMAKE_SOURCE_DIR}/lib/SoftOps"
  SOURCES_GLOB "*.py"
)
add_mlir_python_sources_target(PTOASSoftOpsPythonSourcesStage
  OUTPUT_DIRECTORY
  "${PTOAS_BUILD_PYTHON_PACKAGE_DIR}/_runtime/share/ptoas/SoftOps"
  SOURCES_TARGETS PTOASSoftOpsPythonSources
)

add_custom_target(PTOASPythonPackage ALL)
add_dependencies(PTOASPythonPackage
  PTOASPythonCore
  PTOAS_Python
  PTOASRuntimePythonSourcesStage
  PTOASTileOpsPythonSourcesStage
  PTOASSoftOpsPythonSourcesStage
)
if(TARGET PTODSLPackage)
  add_dependencies(PTOASPythonPackage PTODSLPackage)
endif()

if(TARGET ptoas_runtime_deps)
  add_dependencies(ptoas_runtime_deps PTOASPythonPackage)
endif()

install(PROGRAMS "${PTOAS_INSTALL_WRAPPER_BINARY}"
  DESTINATION bin
  RENAME ptoas
  COMPONENT PTOAS_CompilerArchive
)

set(PTOAS_ARCHIVE_PYTHON_VERSION_FILE
    "${CMAKE_CURRENT_BINARY_DIR}/.ptoas-python-version")
file(WRITE "${PTOAS_ARCHIVE_PYTHON_VERSION_FILE}"
  "${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}\n")
install(FILES "${PTOAS_ARCHIVE_PYTHON_VERSION_FILE}"
  DESTINATION "."
  COMPONENT PTOAS_CompilerArchive
)

# The standalone compiler archive is staged by installing PTOAS_Python first
# and PTOAS_CompilerArchive second into the same prefix. Generate the deploy
# script so known native entry points come from CMake targets; the script scans
# only AddMLIRPython's generated extension directory for the remaining modules.
configure_file(
  "${CMAKE_SOURCE_DIR}/cmake/RelocatePTOASCompilerArchive.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/RelocatePTOASCompilerArchive.cmake.in"
  @ONLY
)
file(GENERATE
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/RelocatePTOASCompilerArchive.cmake"
  INPUT "${CMAKE_CURRENT_BINARY_DIR}/RelocatePTOASCompilerArchive.cmake.in"
)
install(SCRIPT
  "${CMAKE_CURRENT_BINARY_DIR}/RelocatePTOASCompilerArchive.cmake"
  COMPONENT PTOAS_CompilerArchive
)

install(
  DIRECTORY "${CMAKE_SOURCE_DIR}/lib/TileOps"
  DESTINATION "ptoas/_runtime/share/ptoas"
  COMPONENT PTOAS_Python
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

install(
  DIRECTORY "${CMAKE_SOURCE_DIR}/lib/SoftOps"
  DESTINATION "ptoas/_runtime/share/ptoas"
  COMPONENT PTOAS_Python
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

# ---------------------------------------------------------------------------
# Online-compilation fallback: ship the version-sensitive `_core` as source +
# a self-contained header closure + a standalone CMakeLists, so a matching
# extension can be compiled at first import against the shipped, Python-
# independent libPTOASCompiler DSO. The prebuilt abi3 `_core` above stays the
# fast path; this is only the fallback for non-matching interpreters.
# ---------------------------------------------------------------------------
option(PTOAS_ENABLE_ONLINE_CORE_COMPILE
  "Ship ptoas._core sources + header closure for runtime online compilation" OFF)

if(PTOAS_ENABLE_ONLINE_CORE_COMPILE)
  set(_ptoas_online_dst "ptoas/_online")

  # 1. The C-API-only dialect binding TU, its dialect-only module entry, and the
  #    shared PTOModule.h header. Both .cpp files `#include "PTOModule.h"`, so it
  #    is staged next to them (quote-include resolves within src/).
  install(FILES
      "${CMAKE_SOURCE_DIR}/lib/Bindings/Python/PTOModule.cpp"
      "${CMAKE_SOURCE_DIR}/lib/Bindings/Python/PTOModule.h"
      "${CMAKE_CURRENT_SOURCE_DIR}/NativeModuleOnline.cpp"
    DESTINATION "${_ptoas_online_dst}/src"
    COMPONENT PTOAS_Python)

  # 2. Standalone runtime build recipe (staged as ptoas/_online/CMakeLists.txt).
  install(FILES
      "${CMAKE_CURRENT_SOURCE_DIR}/online_template/CMakeLists.txt"
    DESTINATION "${_ptoas_online_dst}"
    COMPONENT PTOAS_Python)

  # 3. Project C-API headers.
  install(FILES
      "${CMAKE_SOURCE_DIR}/include/pto-c/Dialect/PTO.h"
      "${CMAKE_SOURCE_DIR}/include/pto-c/Dialect/PTOEnums.h"
    DESTINATION "${_ptoas_online_dst}/include/pto-c/Dialect"
    COMPONENT PTOAS_Python)

  # 4. Twine-free patched copy of PybindAdaptors.h (keeps the closure to
  #    mlir-c/* + pto-c/* + pybind11 + Python.h, no LLVM C++ ADT tree).
  install(FILES
      "${CMAKE_CURRENT_SOURCE_DIR}/online_include/mlir/Bindings/Python/PybindAdaptors.h"
    DESTINATION "${_ptoas_online_dst}/include/mlir/Bindings/Python"
    COMPONENT PTOAS_Python)

  # 5. mlir-c header closure + the generated mlir-config.h, resolved from the
  #    MLIR include dirs (source tree for mlir-c/*, build tree for
  #    mlir/Config/mlir-config.h). This list is the exact transitive closure of
  #    PTOModule.cpp under PTOAS_ONLINE_BUILD; adding an include there means
  #    extending this list.
  set(_ptoas_online_mlir_headers
      mlir-c/AffineExpr.h
      mlir-c/AffineMap.h
      mlir-c/Bindings/Python/Interop.h
      mlir-c/BuiltinAttributes.h
      mlir-c/BuiltinTypes.h
      mlir-c/Diagnostics.h
      mlir-c/ExecutionEngine.h
      mlir-c/IntegerSet.h
      mlir-c/IR.h
      mlir-c/Pass.h
      mlir-c/Rewrite.h
      mlir-c/Support.h
      mlir/Config/mlir-config.h)
  foreach(_hdr IN LISTS _ptoas_online_mlir_headers)
    set(_ptoas_found "")
    foreach(_inc IN LISTS MLIR_INCLUDE_DIRS)
      if(EXISTS "${_inc}/${_hdr}")
        set(_ptoas_found "${_inc}/${_hdr}")
        break()
      endif()
    endforeach()
    if(NOT _ptoas_found)
      message(FATAL_ERROR
        "PTOAS_ENABLE_ONLINE_CORE_COMPILE: required header '${_hdr}' not found "
        "in MLIR_INCLUDE_DIRS='${MLIR_INCLUDE_DIRS}'")
    endif()
    get_filename_component(_ptoas_hdr_subdir "${_hdr}" DIRECTORY)
    install(FILES "${_ptoas_found}"
      DESTINATION "${_ptoas_online_dst}/include/${_ptoas_hdr_subdir}"
      COMPONENT PTOAS_Python)
  endforeach()
endif()

