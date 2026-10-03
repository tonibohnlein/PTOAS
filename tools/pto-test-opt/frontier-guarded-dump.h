// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Test/debug serialization only; does not evaluate production predicates.
#ifndef PTO_FRONTIER_GUARDED_DUMP_H
#define PTO_FRONTIER_GUARDED_DUMP_H
#include "PTO/Transforms/FrontierSynch/GuardedAnalysis.h"
#include "llvm/Support/JSON.h"
namespace frontier_test {
inline llvm::json::Array demands(mlir::ArrayRef<mlir::pto::frontiersynch::GuardedDemand> edges)
{
    llvm::json::Array result;
    for (const auto& edge : edges) {
        llvm::json::Array witnesses;
        for (const auto& witness : edge.witnesses) {
            witnesses.push_back(
                llvm::json::Object{
                    {"kind", static_cast<unsigned>(witness.storage.hazard)},
                    {"source", witness.storage.sourceFootprint},
                    {"consumer", witness.storage.consumerFootprint},
                    {"guard", witness.predicate}});
        }
        result.push_back(
            llvm::json::Object{
                {"source", edge.source},
                {"consumer", edge.consumer},
                {"guard", edge.predicate},
                {"witnesses", std::move(witnesses)}});
    }
    return result;
}
inline llvm::json::Object dump(const mlir::pto::frontiersynch::GuardedDemandAnalysis& analysis)
{
    namespace fs = mlir::pto::frontiersynch;
    llvm::json::Array nodes, pipes, phases, local, readiness;
    std::size_t atom = 0;
    for (const auto& node : analysis.predicates().nodes()) {
        llvm::json::Object entry{
            {"kind", static_cast<unsigned>(node.kind)}, {"first", node.first}, {"second", node.second}};
        if (node.kind == fs::PredicateKind::Atom) {
            entry["atom"] = atom++;
        }
        nodes.push_back(std::move(entry));
    }
    for (auto phase : analysis.phases()) {
        pipes.push_back(static_cast<unsigned>(phase->kPipeValue));
        phases.push_back(phase->GetIndex());
    }
    for (const auto& edge : analysis.localDemands()) {
        local.push_back(llvm::json::Object{{"demand", edge.demand}, {"guard", edge.nonadjacent}});
    }
    for (std::size_t source = 0; source < analysis.phases().size(); ++source) {
        llvm::json::Array row;
        for (std::size_t consumer = 0; consumer < analysis.phases().size(); ++consumer) {
            row.push_back(*analysis.completionBeforeStart(source, consumer));
        }
        readiness.push_back(std::move(row));
    }
    return llvm::json::Object{
        {"valid", true},
        {"nodes", std::move(nodes)},
        {"atoms", atom},
        {"occurrences", llvm::json::Array(analysis.occurrences())},
        {"pipes", std::move(pipes)},
        {"phases", std::move(phases)},
        {"generators", demands(analysis.generators())},
        {"retained", demands(analysis.retained())},
        {"local", std::move(local)},
        {"readiness", std::move(readiness)}};
}
} // namespace frontier_test
#endif
