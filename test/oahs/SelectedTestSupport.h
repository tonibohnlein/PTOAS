// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_OAHS_SELECTED_TEST_SUPPORT_H
#define PTO_OAHS_SELECTED_TEST_SUPPORT_H
#include "PTO/Transforms/OAHS/SelectedPlan.h"
#include "PTO/Transforms/OAHS/ObservedPrograms.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
namespace selected_test {
namespace o = mlir::pto::oahs;
inline void require(bool value, const std::string& message)
{
    if (!value) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}
inline o::Program base(unsigned cells, unsigned keys = 2)
{
    o::Program p;
    p.target.contract = "selected-test ordinary prefix core";
    for (unsigned i = 0; i < o::PipeCount; ++i) {
        p.target.supported[i] = p.target.barriers[i] = true;
        for (unsigned j = 0; j < o::PipeCount; ++j) {
            if (i != j) {
                for (unsigned k = 0; k < keys; ++k) {
                    p.target.keys[i][j].push_back(k);
                }
            }
        }
    }
    for (unsigned i = 0; i < cells; ++i) {
        o::Cell cell;
        cell.storage = o::Cell::Storage::CanonicalInterval;
        cell.addressSpace = "test";
        cell.coordinateSpace = "physical";
        cell.ranges = {{uint64_t(i) * 16, 16}};
        p.cells.push_back(cell);
    }
    return p;
}
inline o::Operation op(o::Pipe pipe, std::initializer_list<o::Access> effects)
{
    o::Operation operation;
    operation.pipe = pipe;
    operation.accesses = effects;
    operation.complete = true;
    return operation;
}
inline o::Region leaf(unsigned id) { return {o::Region::Operation, {}, id}; }
inline o::Region seq(std::initializer_list<o::Region> children) { return {o::Region::Sequence, children}; }
inline unsigned count(const o::SelectedPlan& plan, o::Command::Kind kind)
{
    return unsigned(std::count_if(plan.ledger.begin(), plan.ledger.end(), [&](const auto& endpoint) {
        return endpoint.command.kind == kind;
    }));
}
inline o::SelectedPlan accepted(const o::Program& p, const o::Commands& fixed = {})
{
    auto result = o::constructSelectedPlan(p, fixed);
    require(result.success, "constructor refused: " + result.reason + " at " + std::to_string(result.cut));
    const auto cold = o::checkCausalFrontier(p, result.commands);
    require(cold.accepted, "cold validation refused selected commands");
    require(cold.cuts.size() == result.certificate.cuts.size(), "cold certificate shape differs");
    for (std::size_t cut = 0; cut < cold.cuts.size(); ++cut) {
        const auto& actual = result.certificate.cuts[cut];
        const auto& expected = cold.cuts[cut];
        for (const auto& pair : {std::make_pair(actual.incoming, expected.incoming),
                                std::make_pair(actual.beforeIssue, expected.beforeIssue),
                                std::make_pair(actual.outgoing, expected.outgoing)}) {
            require(pair.first.reachable() == pair.second.reachable(), "cold reachability differs");
            if (pair.first.reachable()) {
                require(*pair.first.facts() == *pair.second.facts(), "cold selected-ledger facts differ");
            }
        }
    }
    require(result.work.selectedUpdates == result.updates.size(), "update accounting mismatch");
    return result;
}
} // namespace selected_test
#endif
