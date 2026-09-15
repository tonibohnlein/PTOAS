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
add_executable(oahs-plan-test plan_test.cpp ../../lib/PTO/Transforms/OAHS/Plan.cpp)
target_include_directories(oahs-plan-test PRIVATE ../../include)
target_compile_features(oahs-plan-test PRIVATE cxx_std_17)
target_compile_options(oahs-plan-test PRIVATE -Wall -Wextra -Werror -pedantic)
add_test(NAME oahs_plan COMMAND oahs-plan-test)
add_executable(oahs-structured-test structured_test.cpp ../../lib/PTO/Transforms/OAHS/Plan.cpp)
target_include_directories(oahs-structured-test PRIVATE ../../include)
target_compile_features(oahs-structured-test PRIVATE cxx_std_17)
target_compile_options(oahs-structured-test PRIVATE -Wall -Wextra -Werror -pedantic)
add_test(NAME oahs_structured COMMAND oahs-structured-test)
add_executable(oahs-scaling-test scaling_test.cpp ../../lib/PTO/Transforms/OAHS/Plan.cpp)
target_include_directories(oahs-scaling-test PRIVATE ../../include)
target_compile_features(oahs-scaling-test PRIVATE cxx_std_17)
target_compile_options(oahs-scaling-test PRIVATE -Wall -Wextra -Werror -pedantic)
add_test(NAME oahs_scaling COMMAND oahs-scaling-test)
