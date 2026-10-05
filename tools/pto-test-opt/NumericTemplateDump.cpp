// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Preserve original occurrence identity in late-expansion diagnostic output.
#include "PTO/Transforms/FrontierSynch/NumericTemplate.h"
#include "mlir/IR/AsmState.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
std::string name(Value value, AsmState& state)
{
    std::string text;
    llvm::raw_string_ostream stream(text);
    value.printAsOperand(stream, state);
    return text;
}
std::string expression(AffineExpr value)
{
    std::string text;
    llvm::raw_string_ostream stream(text);
    value.print(stream);
    return text;
}
llvm::json::Object range(const pto::SyncStorageCell& value)
{
    return llvm::json::Object{{"space", static_cast<unsigned>(value.space)},
                              {"begin", value.begin}, {"end", value.end}};
}
}
llvm::json::Object dumpNumericTemplate(const fs::NumericTemplate& result, AsmState& state)
{
    llvm::json::Array issues, payloads, atoms;
    for (const auto& diagnostic : result.result.diagnostics) {
        issues.push_back(llvm::json::Object{{"issue", fs::recognitionName(diagnostic.issue)},
            {"operation", diagnostic.anchor ? diagnostic.anchor->getName().getStringRef() : StringRef()}});
    }
    for (const auto& payload : result.payloads) {
        llvm::json::Array coordinates, effects;
        for (auto coordinate : payload.coordinates) {
            coordinates.push_back(llvm::json::Object{{"induction", name(coordinate.loop.getInductionVar(), state)},
                                                    {"value", coordinate.induction}});
        }
        for (const auto& effect : payload.effects) {
            llvm::json::Array maps, ranges, ids;
            for (const auto& map : effect.regions) {
                llvm::json::Array extents;
                for (auto extent : map.extents) {
                    extents.push_back(expression(extent));
                }
                auto argument = dyn_cast_or_null<BlockArgument>(map.base);
                maps.push_back(llvm::json::Object{
                    {"base_argument", argument ? static_cast<int64_t>(argument.getArgNumber()) : -1},
                    {"offset", expression(map.byteOffset)}, {"extents", std::move(extents)},
                    {"element_bytes", map.elementBytes}});
            }
            for (const auto& item : effect.ranges) {
                ranges.push_back(range(item));
            }
            for (auto atom : effect.atoms) {
                ids.push_back(atom);
            }
            effects.push_back(llvm::json::Object{{"source_effect", effect.sourceEffect},
                {"mode", effect.mode == pto::SyncAccessMode::Read ? "read" : "write"},
                {"maps", std::move(maps)}, {"ranges", std::move(ranges)}, {"atoms", std::move(ids)},
                {"discharge", static_cast<unsigned>(effect.discharge)}, {"outer_stride", effect.outerStride}});
        }
        payloads.push_back(llvm::json::Object{{"phase", payload.phase->GetIndex()},
            {"operation", payload.phase->elementOp->getName().getStringRef()},
            {"pipe", static_cast<unsigned>(payload.phase->kPipeValue)}, {"coordinates", std::move(coordinates)},
            {"effects", std::move(effects)}});
    }
    for (const auto& atom : result.atoms) {
        atoms.push_back(range(atom));
    }
    auto outer = result.outer;
    return llvm::json::Object{{"route", "numeric-template"}, {"state", fs::recognitionName(result.result.state)},
        {"issues", std::move(issues)}, {"lower", result.lower}, {"step", result.step},
        {"empty_invocation", result.emptyInvocation},
        {"upper", name(outer.getUpperBound(), state)}, {"counted_visits", result.countedVisits},
        {"counted_payloads", result.countedPayloads}, {"fragments", result.fragments},
        {"period", result.period}, {"refresh", result.refresh}, {"scope", "whole-function"},
        {"interfaces_ready", false}, {"payloads", std::move(payloads)}, {"atoms", std::move(atoms)}};
}
