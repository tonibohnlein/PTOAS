# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.16)
# Invoke with -DCMAKE_CXX_COMPILER=bisheng after sourcing the device toolkit.
project(insertsync_device_adapters LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
foreach(required KERNEL_SRC PTO_ISA_ROOT BENCH_CASE)
  if(NOT DEFINED ${required})
    message(FATAL_ERROR "Missing -D${required}=...")
  endif()
endforeach()
set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
if(NOT ASCEND_HOME_PATH)
  message(FATAL_ERROR "Source the device toolkit's set_env.sh first")
endif()
if(BENCH_CASE STREQUAL "conv2d_interior")
  set(BENCH_DEFINE BENCH_CONV2D)
  set(BENCH_ARCH dav-c220-cube)
elseif(BENCH_CASE STREQUAL "flash_attention_cube")
  set(BENCH_DEFINE BENCH_FA)
  set(BENCH_ARCH dav-c220)
elseif(BENCH_CASE STREQUAL "gdn_wy")
  set(BENCH_DEFINE BENCH_GDN)
  set(BENCH_ARCH dav-c220)
elseif(BENCH_CASE STREQUAL "kda_wy")
  set(BENCH_DEFINE BENCH_KDA)
  set(BENCH_ARCH dav-c220)
else()
  message(FATAL_ERROR "Unknown BENCH_CASE: ${BENCH_CASE}")
endif()
configure_file("${KERNEL_SRC}" "${CMAKE_CURRENT_BINARY_DIR}/kernel.cpp" COPYONLY)
add_library(case_kernel SHARED launch.cpp)
target_compile_definitions(case_kernel PRIVATE ${BENCH_DEFINE} MEMORY_BASE)
target_include_directories(case_kernel PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/.."
  "${CMAKE_CURRENT_SOURCE_DIR}/../references" "${PTO_ISA_ROOT}/include"
  "${ASCEND_HOME_PATH}/include" "${ASCEND_HOME_PATH}/pkg_inc"
  "${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime")
target_compile_options(case_kernel PRIVATE -O2 -std=c++17 -xcce
  --cce-aicore-arch=${BENCH_ARCH} -Xhost-start -Xhost-end
  "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
  "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
  "SHELL:-mllvm -cce-aicore-record-overflow=true"
  "SHELL:-mllvm -cce-aicore-addr-transform"
  "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false")
target_link_options(case_kernel PRIVATE --cce-fatobj-link)
add_library(case_runner SHARED host.cpp)
target_include_directories(case_runner PRIVATE "${ASCEND_HOME_PATH}/include")
target_compile_options(case_runner PRIVATE -O2 -xc++)
target_link_directories(case_runner PRIVATE "${ASCEND_HOME_PATH}/lib64")
target_link_libraries(case_runner PRIVATE case_kernel ascendcl runtime)
