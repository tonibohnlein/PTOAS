# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
cmake_minimum_required(VERSION 3.16)
project(OAHSCoreTests LANGUAGES CXX)
enable_testing()
# Compile the same production core once for every standalone test executable.
add_library(oahs-core-under-test STATIC ../../lib/PTO/Transforms/OAHS/Plan.cpp)
target_include_directories(oahs-core-under-test PUBLIC ../../include)
target_compile_features(oahs-core-under-test PUBLIC cxx_std_17)
target_compile_options(oahs-core-under-test PRIVATE -Wall -Wextra -Werror -pedantic)
foreach(suite plan structured scaling analysis prefix)
  add_executable(oahs-${suite}-test ${suite}_test.cpp)
  target_link_libraries(oahs-${suite}-test PRIVATE oahs-core-under-test)
  target_compile_options(oahs-${suite}-test PRIVATE -Wall -Wextra -Werror -pedantic)
  add_test(NAME oahs_${suite} COMMAND oahs-${suite}-test)
endforeach()
