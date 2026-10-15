// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIER_SEQUENCE_INCOMING_QUERY_CHECKS_H
#define PTO_FRONTIER_SEQUENCE_INCOMING_QUERY_CHECKS_H
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/SequenceIncomingQuery.h"
namespace mlir::pto::test {
inline bool incomingQueryFailure(unsigned line)
{
    llvm::errs() << "incoming sequence query mismatch at line " << line << "\n";
    return false;
}
inline bool checkIncomingSequenceQueries(func::FuncOp function, const CompoundInstanceElement& phase)
{
    namespace fs = frontiersynch;
    using Kind = fs::PeriodicEventKind;
    Block inputs;
    // Independent truth table chooses the original occurrences before reading
    // their relation. Covers both event kinds and complete outer visit tuples.
    for (bool alias : {false, true}) {
        for (unsigned variant = 0; variant < 8; ++variant) {
            const unsigned mode = variant % 4;
            const uint64_t first = variant < 4 ? 0 : uint64_t(1) << 63;
            const uint64_t last = variant < 4 ? 1 : UINT64_MAX;
            for (Kind sourceKind : {Kind::Start, Kind::Completion}) {
                for (Kind targetKind : {Kind::Start, Kind::Completion}) {
                    auto e = std::make_shared<fs::RegionExpressions>();
                    auto bit = [&]() { return e->input(inputs.addArgument(
                        IntegerType::get(function.getContext(), 1), function.getLoc())); };
                    const auto left = bit(), right = bit(), unrelated = bit();
                    const std::array<fs::Expr, 4> table{bit(), bit(), bit(), bit()};
                    const auto active = mode == 0 ? e->land(left, right) : mode == 1 ? left :
                        mode == 2 ? unrelated : e->boolean(false);
                    fs::SequenceAnalysisState state(function, e);
                    state.children.resize(1);
                    auto& child = state.children.front(); child.anchors = {{&phase}, {&phase}};
                    auto& region = child.regional;
                    region.expressions = e; region.anchors = child.anchors;
                    region.capabilities.substitutionStableQueries = true;
                    region.occurrenceLoops.resize(2); region.outerLoops.resize(2, std::vector<scf::ForOp>(1));
                    region.presence = [e](fs::RegionalEvent) { return e->boolean(true); };
                    unsigned calls = 0;
                    bool valid = true;
                    region.reachability = [&](fs::RegionalEvent a, fs::RegionalEvent b) -> std::optional<fs::Expr> {
                        ++calls;
                        valid &= a.type == 0 && b.type == 1 && a.kind == sourceKind && b.kind == targetKind &&
                            a.visits.size() == 1 && b.visits.size() == 1;
                        if (!valid) { return std::nullopt; }
                        auto choose = [&](fs::Expr ordinal, fs::Expr visit, uint64_t shift) {
                            return e->eq(visit, e->add(ordinal, e->constant(shift)));
                        };
                        auto value = e->select(e->eq(a.ordinal, e->constant(last)),
                            e->select(e->eq(b.ordinal, e->constant(last)), table[3], table[2]),
                            e->select(e->eq(b.ordinal, e->constant(last)), table[1], table[0]));
                        return e->land(value, e->land(choose(a.ordinal, a.visits[0], 3),
                            choose(b.ordinal, b.visits[0], 5)));
                    };
                    auto endpoint = [&](unsigned type, fs::Expr condition, uint64_t shift) {
                        if (alias) {
                            auto yes = state.port(0, type, e->constant(last), {e->constant(shift + last)});
                            auto no = state.port(0, type, e->constant(first), {e->constant(shift + first)});
                            return state.selectPort(condition, yes, no);
                        }
                        // Raw selector-produced coordinates have no PortChoice
                        // metadata. They exercise specialization before the child.
                        auto ordinal = e->select(condition, e->constant(last), e->constant(first));
                        return state.port(0, type, ordinal, {e->add(ordinal, e->constant(shift))});
                    };
                    auto a = state.event(2 * endpoint(0, left, 3)), b = state.event(2 * endpoint(1, right, 5));
                    a.kind = sourceKind; b.kind = targetKind;
                    fs::SequenceIncomingQuery query(state);
                    auto answer = query.query(a, b, active);
                    const unsigned expectedCalls = mode == 3 ? 0 :
                        !alias || mode == 0 ? 1 : mode == 1 ? 2 : 4;
                    if (!answer || !valid || calls != expectedCalls ||
                        (mode == 0 && !alias && state.crossingQueryWork.incomingCoordinates != 4)) {
                        return incomingQueryFailure(__LINE__);
                    }
                    const auto size = e->size(), proofs = state.crossingQueryWork.incomingProofVisits;
                    if (query.query(a, b, active) != answer || calls != expectedCalls || e->size() != size ||
                        state.crossingQueryWork.incomingProofVisits != proofs) {
                        return incomingQueryFailure(__LINE__);
                    }
                    if (mode == 0) {
                        // Unselected relation inputs and the opposite selector
                        // can be unavailable: activity still masks them lazily.
                        fs::RegionExpressions::Transaction evaluation(*e);
                        fs::RegionExpressions::Substitution enabled({{left, e->boolean(true)},
                            {right, e->boolean(true)}, {table[3], e->boolean(true)}});
                        fs::RegionExpressions::Substitution disabled({{left, e->boolean(false)}});
                        if (e->constantValue(e->substitute(*answer, enabled)) != 1 ||
                            e->constantValue(e->substitute(*answer, disabled)) != 0) {
                            return incomingQueryFailure(__LINE__);
                        }
                    }
                    for (unsigned flags = 0; flags < 8; ++flags) {
                        for (unsigned valuation = 0; valuation < 16; ++valuation) {
                            const bool x = flags & 1, y = flags & 2, z = flags & 4;
                            const bool present = mode == 0 ? x && y : mode == 1 ? x : mode == 2 ? z : false;
                            std::vector<std::pair<fs::Expr, fs::Expr>> bindings{
                                {left, e->boolean(x)}, {right, e->boolean(y)}, {unrelated, e->boolean(z)}};
                            for (unsigned i = 0; i < 4; ++i) {
                                bindings.emplace_back(table[i], e->boolean(valuation & (1U << i)));
                            }
                            fs::RegionExpressions::Transaction evaluation(*e);
                            fs::RegionExpressions::Substitution substitution(bindings);
                            const auto actual = e->constantValue(e->substitute(*answer, substitution));
                            const bool expected = present && (valuation & (1U << (2 * unsigned(x) + unsigned(y))));
                            if (actual != unsigned(expected)) { return incomingQueryFailure(__LINE__); }
                        }
                    }
                }
            }
        }
    }
    {
        auto e = std::make_shared<fs::RegionExpressions>();
        auto active = e->input(inputs.addArgument(IntegerType::get(function.getContext(), 1), function.getLoc()));
        fs::SequenceAnalysisState state(function, e); state.children.resize(1);
        auto& child = state.children.front(); child.anchors = {{&phase}, {&phase}};
        auto& region = child.regional; region.expressions = e; region.anchors = child.anchors;
        region.occurrenceLoops.resize(2);
        region.presence = [e](fs::RegionalEvent) { return e->boolean(true); };
        const auto original = e->select(active, e->constant(1), e->constant(0));
        unsigned originalCalls = 0, speculativeCalls = 0;
        region.reachability = [&](fs::RegionalEvent a, fs::RegionalEvent) -> std::optional<fs::Expr> {
            if (a.ordinal != original) { ++speculativeCalls; return std::nullopt; }
            ++originalCalls; return e->boolean(true);
        };
        fs::SequenceIncomingQuery query(state);
        const auto a = state.port(0, 0, original), b = state.port(0, 1, e->constant(1));
        const auto answer = query.query(2 * a + 1, 2 * b, active);
        // A partial backend accepts its original symbolic request but refuses
        // constants. Missing qualification must avoid that speculative call.
        if (!answer || *answer != active || originalCalls != 1 || speculativeCalls ||
            state.crossingQueryWork.incomingCoordinates) { return incomingQueryFailure(__LINE__); }
    }
    for (unsigned count : {16U, 64U, 256U}) {
        auto e = std::make_shared<fs::RegionExpressions>();
        auto input = e->input(inputs.addArgument(IndexType::get(function.getContext()), function.getLoc()));
        fs::SequenceAnalysisState state(function, e); state.children.resize(1);
        auto& child = state.children.front(); child.anchors = {{&phase}, {&phase}};
        auto& region = child.regional; region.expressions = e; region.anchors = child.anchors;
        region.occurrenceLoops.resize(2);
        region.presence = [e](fs::RegionalEvent) { return e->boolean(true); };
        unsigned calls = 0;
        region.reachability = [&](fs::RegionalEvent a, fs::RegionalEvent b) -> std::optional<fs::Expr> {
            ++calls;
            if (e->constantValue(a.ordinal) != count / 2 || e->constantValue(b.ordinal) != count / 2) {
                return std::nullopt;
            }
            return e->boolean(true);
        };
        auto a = state.port(0, 0, e->constant(0)), b = state.port(0, 1, e->constant(0));
        for (unsigned i = 1; i < count; ++i) {
            const auto condition = e->eq(input, e->constant(i));
            a = state.selectPort(condition, state.port(0, 0, e->constant(i)), a);
            b = state.selectPort(condition, state.port(0, 1, e->constant(i)), b);
        }
        const auto active = e->eq(input, e->constant(count / 2));
        (void)e->boolean(false); (void)e->boolean(true);
        const auto size = e->size();
        fs::SequenceIncomingQuery query(state);
        auto answer = query.query(2 * a + 1, 2 * b, active);
        if (!answer || *answer != active || calls != 1 || e->size() != size ||
            state.crossingQueryWork.incomingArms != 2 * (count - count / 2)) {
            return incomingQueryFailure(__LINE__);
        }
        // An active unavailable original pair must refuse. Neither a previous
        // answer nor an inactive request grants availability in another domain.
        if (!query.query(2 * a + 1, 2 * b, e->boolean(false)) ||
            query.query(2 * a + 1, 2 * b, e->eq(input, e->constant(count - 1)))) {
            return incomingQueryFailure(__LINE__);
        }
    }
    llvm::outs() << "incoming query checks: consumed-domain truth table; exact residuals; 16/64/256 choices\n";
    return true;
}
} // namespace mlir::pto::test
#endif
