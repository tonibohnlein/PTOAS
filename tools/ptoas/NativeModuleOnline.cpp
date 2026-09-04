// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Dialect-only `_core` entry point for the online-compiled fallback path.
//
// The prebuilt abi3 `_core` (tools/ptoas/NativeModule.cpp) additionally wires up
// the TileLib/SoftLib Python bridges and the `main` CLI entry. Those depend on
// the full C++ compiler tree (ptoas.h, TileLibService.h, mlir/CAPI/IR.h) and
// therefore cannot be compiled online with the shipped C-API-only header
// closure. When the online path is taken, only the PTO dialect bindings are
// exposed; the full-featured `main`/TileLib surface remains available through
// the prebuilt abi3 module.

#include "PTOModule.h"

#include "pybind11/pybind11.h"

namespace py = pybind11;

PYBIND11_MODULE(_core, module) {
  module.doc() = "PTOAS PTO dialect native bindings (online-compiled)";
  py::module_::import("ptoas.mlir.ir");
  mlir::pto::python::populatePTODialectBindings(module);
}
