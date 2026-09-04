// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTOAS_LIB_BINDINGS_PYTHON_PTOMODULE_H
#define PTOAS_LIB_BINDINGS_PYTHON_PTOMODULE_H

#ifdef PTOAS_ONLINE_BUILD
#  define PTOAS_COMPILER_EXPORT
#else
#  include "PTO/Compiler/CompilerApi.h"
#endif
#include "pybind11/pybind11.h"

namespace mlir::pto::python {

/// Adds PTO dialect types, attributes, enums, and registration helpers to the
/// project-owned ptoas._core extension module through PTOASCompiler.
PTOAS_COMPILER_EXPORT void
populatePTODialectBindings(pybind11::module_ &module);

} // namespace mlir::pto::python

#endif // PTOAS_LIB_BINDINGS_PYTHON_PTOMODULE_H
