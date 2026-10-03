// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Structured test output for independent finite checks of derived relations.
#include "PTO/Transforms/FrontierSynch/ArithmeticProgram.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
void dumpArithmeticJSON(func::FuncOp function, const fs::ArithmeticProgram& program)
{
    llvm::json::Array relations;
    for (auto [id, relation] : llvm::enumerate(program.primitives.relations)) {
        llvm::json::Array pieces;
        for (const auto& normalized : program.recognition.pieces) {
            if (normalized.relation != id) {
                continue;
            }
            llvm::json::Array rows, residues;
            for (auto residue : relation.pieces[normalized.piece].residues) {
                residues.push_back(residue);
            }
            for (const auto& row : normalized.rows) {
                llvm::json::Array coefficients;
                for (auto coefficient : row.coefficients) {
                    coefficients.push_back(coefficient);
                }
                rows.push_back(llvm::json::Object{{"coefficients", std::move(coefficients)},
                                                {"constant", row.constant}, {"equality", row.equality}});
            }
            pieces.push_back(llvm::json::Object{{"empty", normalized.empty}, {"residues", std::move(residues)},
                                               {"rows", std::move(rows)}});
        }
        relations.push_back(llvm::json::Object{
            {"kind", static_cast<unsigned>(relation.kind)},
            {"source", relation.sourceSite ? static_cast<int64_t>(*relation.sourceSite) : -1},
            {"target", relation.targetSite ? static_cast<int64_t>(*relation.targetSite) : -1},
            {"source_event", static_cast<unsigned>(relation.sourceEvent)},
            {"target_event", static_cast<unsigned>(relation.targetEvent)},
            {"source_dimensions", relation.sourceDimensions}, {"target_dimensions", relation.targetDimensions},
            {"space", relation.storageSpace ? static_cast<int64_t>(*relation.storageSpace) : -1},
            {"dimensions", relation.dimensions}, {"pieces", std::move(pieces)}});
    }
    llvm::json::Array sites, parameters;
    for (const auto& site : program.sites) {
        sites.push_back(llvm::json::Object{{"depth", site.loops.size()},
                                         {"pipe", static_cast<unsigned>(site.phase->kPipeValue)}});
    }
    for (auto parameter : program.parameters) {
        parameters.push_back(cast<BlockArgument>(parameter).getArgNumber());
    }
    llvm::json::Object document{{"function", function.getSymName()}, {"period", program.primitives.period},
                                {"sites", std::move(sites)}, {"parameters", std::move(parameters)},
                                {"relations", std::move(relations)}};
    llvm::outs() << "arithmetic-json " << llvm::json::Value(std::move(document)) << "\n";
}
