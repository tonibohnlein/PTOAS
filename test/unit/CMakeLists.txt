# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

add_executable(pto-schedule-graph-unit-test
  ScheduleGraphTest.cpp
)
target_include_directories(pto-schedule-graph-unit-test PRIVATE
  "${PROJECT_SOURCE_DIR}/include"
)
target_link_libraries(pto-schedule-graph-unit-test PRIVATE PTOTransforms)
add_test(NAME pto_schedule_graph_unit
  COMMAND pto-schedule-graph-unit-test
)
set_tests_properties(pto_schedule_graph_unit PROPERTIES LABELS "PTO;unit")

add_executable(pto-isa-duration-unit-test
  PTOISADurationTest.cpp
)
target_include_directories(pto-isa-duration-unit-test PRIVATE
  "${PROJECT_SOURCE_DIR}/include"
)
target_link_libraries(pto-isa-duration-unit-test PRIVATE PTOTransforms)
add_test(NAME pto_isa_duration_unit
  COMMAND pto-isa-duration-unit-test
)
set_tests_properties(pto_isa_duration_unit PROPERTIES LABELS "PTO;unit")
