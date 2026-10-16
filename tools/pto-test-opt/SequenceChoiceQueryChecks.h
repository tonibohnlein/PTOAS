// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIER_SEQUENCE_CHOICE_QUERY_CHECKS_H
#define PTO_FRONTIER_SEQUENCE_CHOICE_QUERY_CHECKS_H
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/SequenceAnalysisInternal.h"
namespace mlir::pto::test {
inline bool correlatedQueryFailure(unsigned line)
{
    llvm::errs() << "correlated sequence query mismatch at line " << line << "\n";
    return false;
}
inline bool checkCorrelatedSequenceQueries(func::FuncOp function, const CompoundInstanceElement& phase)
{
    namespace fs = frontiersynch;
    using Kind = fs::PeriodicEventKind;
    Block inputs;
    for (bool framed : {false, true}) {
        for (bool shared : {false, true}) {
            for (unsigned missing = 0; missing < 3; ++missing) {
                for (Kind sourceKind : {Kind::Start, Kind::Completion}) {
                    for (Kind targetKind : {Kind::Start, Kind::Completion}) {
                        auto e = std::make_shared<fs::RegionExpressions>();
                        auto bit = [&]() {
                            return e->input(inputs.addArgument(
                                IntegerType::get(function.getContext(), 1), function.getLoc()));
                        };
                        const auto left = bit(), right = shared ? left : bit();
                        std::array<fs::Expr, 4> table{bit(), bit(), bit(), bit()};
                        fs::SequenceAnalysisState state(function, e);
                        state.children.resize(1);
                        auto& child = state.children.front();
                        child.anchors = {{&phase}, {&phase}};
                        auto& region = child.regional;
                        region.expressions = e; region.anchors = child.anchors;
                        region.occurrenceLoops.resize(2);
                        // The query oracle checks the retained coordinate tuple;
                        // no loop execution or endpoint availability is assumed.
                        if (framed) { region.outerLoops.resize(2, std::vector<scf::ForOp>(1)); }
                        region.presence = [e](fs::RegionalEvent) { return e->boolean(true); };
                        unsigned queries = 0;
                        bool originalFrame = true;
                        region.reachability = [&](fs::RegionalEvent a, fs::RegionalEvent b)
                            -> std::optional<fs::Expr> {
                            ++queries;
                            const auto x = e->constantValue(a.ordinal), y = e->constantValue(b.ordinal);
                            originalFrame &= a.type == 0 && b.type == 1 && a.kind == sourceKind &&
                                b.kind == targetKind && x && y && *x < 2 && *y < 2;
                            if (!originalFrame) { return std::nullopt; }
                            originalFrame &= a.visits.size() == unsigned(framed) &&
                                b.visits.size() == unsigned(framed);
                            if (framed) {
                                originalFrame &= e->constantValue(a.visits.front()) == 3 * *x + 1 &&
                                    e->constantValue(b.visits.front()) == 5 * *y + 2;
                            }
                            if (!originalFrame || (missing == 1 && x != y) || (missing == 2 && x == y)) {
                                return std::nullopt;
                            }
                            return table[2 * *x + *y];
                        };
                        std::array<uint32_t, 2> source, target;
                        for (unsigned arm = 0; arm < 2; ++arm) {
                            source[arm] = state.port(0, 0, e->constant(arm),
                                framed ? std::vector<fs::Expr>{e->constant(3 * arm + 1)} : std::vector<fs::Expr>{});
                            target[arm] = state.port(0, 1, e->constant(arm),
                                framed ? std::vector<fs::Expr>{e->constant(5 * arm + 2)} : std::vector<fs::Expr>{});
                        }
                        const auto a = state.selectPort(left, source[1], source[0]);
                        const auto b = state.selectPort(right, target[1], target[0]);
                        auto x = state.event(2 * a), y = state.event(2 * b);
                        x.kind = sourceKind; y.kind = targetKind;
                        auto answer = state.eventReachability(x, y);
                        const bool available = missing == 0 || (shared && missing == 1);
                        if (bool(answer) != available || !originalFrame || queries != (shared ? 2U : 4U) ||
                            state.crossingQueryWork.correlatedChoices != unsigned(shared)) {
                            return correlatedQueryFailure(__LINE__);
                        }
                        if (!available) { continue; }
                        const auto size = e->size(), cached = state.reachabilityCache.size();
                        if (state.eventReachability(x, y) != answer || e->size() != size ||
                            state.reachabilityCache.size() != cached || queries != (shared ? 2U : 4U)) {
                            return correlatedQueryFailure(__LINE__);
                        }
                        if (shared) {
                            // Leave every unselected leaf input unavailable.
                            // The original outer select must still mask it.
                            for (bool arm : {false, true}) {
                                fs::RegionExpressions::Substitution binding({
                                    {left, e->boolean(arm)}, {table[arm ? 3 : 0], e->boolean(true)}});
                                if (e->constantValue(e->substitute(*answer, binding)) != 1) {
                                    return correlatedQueryFailure(__LINE__);
                                }
                            }
                            const auto inputs = e->referencedInputs(*answer);
                            for (const auto& input : inputs) {
                                if (input.first == table[1] || input.first == table[2]) {
                                    return correlatedQueryFailure(__LINE__);
                                }
                            }
                        }
                        // Independent lazy truth table: select the original
                        // occurrences first, then read exactly that leaf pair.
                        for (unsigned valuation = 0; valuation < 16; ++valuation) {
                            for (unsigned flags = 0; flags < (shared ? 2U : 4U); ++flags) {
                                const unsigned ax = flags & 1, by = shared ? ax : (flags >> 1);
                                std::vector<std::pair<fs::Expr, fs::Expr>> values{{left, e->boolean(ax)}};
                                if (!shared) { values.emplace_back(right, e->boolean(by)); }
                                for (unsigned i = 0; i < table.size(); ++i) {
                                    values.emplace_back(table[i], e->boolean(valuation & (1U << i)));
                                }
                                fs::RegionExpressions::Substitution binding(values);
                                const auto expected = unsigned(bool(valuation & (1U << (2 * ax + by))));
                                if (e->constantValue(e->substitute(*answer, binding)) != expected) {
                                    return correlatedQueryFailure(__LINE__);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    for (unsigned count : {16U, 64U, 256U}) {
        auto e = std::make_shared<fs::RegionExpressions>();
        const auto choose = e->input(inputs.addArgument(IndexType::get(function.getContext()), function.getLoc()));
        fs::SequenceAnalysisState state(function, e);
        state.children.resize(1);
        auto& child = state.children.front(); child.anchors = {{&phase}, {&phase}};
        auto& region = child.regional;
        region.expressions = e; region.anchors = child.anchors; region.occurrenceLoops.resize(2);
        region.presence = [e](fs::RegionalEvent) { return e->boolean(true); };
        unsigned queries = 0;
        region.reachability = [&](fs::RegionalEvent a, fs::RegionalEvent b) -> std::optional<fs::Expr> {
            ++queries;
            const auto x = e->constantValue(a.ordinal), y = e->constantValue(b.ordinal);
            // Off-diagonal queries are deliberately unsupported. They cannot
            // be consumed by these identically selected original occurrences.
            if (!x || !y || x != y || *x >= count) { return std::nullopt; }
            return e->boolean(true);
        };
        auto a = state.port(0, 0, e->constant(0)), b = state.port(0, 1, e->constant(0));
        for (unsigned i = 1; i < count; ++i) {
            const auto condition = e->eq(choose, e->constant(i));
            a = state.selectPort(condition, state.port(0, 0, e->constant(i)), a);
            b = state.selectPort(condition, state.port(0, 1, e->constant(i)), b);
        }
        auto answer = state.eventReachability(state.event(2 * a + 1), state.event(2 * b));
        if (!answer || e->constantValue(*answer) != 1 || queries != count ||
            state.crossingQueryWork.correlatedChoices != count - 1 ||
            state.reachabilityCache.size() != 2 * count - 1) { return correlatedQueryFailure(__LINE__); }
        // The complete query is independent of the selector: no new required
        // input may survive merely because ordinal coordinates were folded.
        if (!e->referencedInputs(*answer).empty()) { return correlatedQueryFailure(__LINE__); }
    }
    {
        auto e = std::make_shared<fs::RegionExpressions>();
        const auto present = e->input(inputs.addArgument(
            IntegerType::get(function.getContext(), 1), function.getLoc()));
        fs::SequenceAnalysisState state(function, e);
        state.children.resize(1);
        auto& child = state.children.front(); child.anchors = {{&phase}};
        auto& region = child.regional;
        region.expressions = e; region.anchors = child.anchors; region.occurrenceLoops.resize(1);
        region.presence = [present](fs::RegionalEvent) { return present; };
        unsigned queries = 0;
        region.reachability = [&](fs::RegionalEvent, fs::RegionalEvent) -> std::optional<fs::Expr> {
            ++queries; return std::nullopt;
        };
        const auto port = state.port(0, 0, e->constant(0));
        for (Kind from : {Kind::Start, Kind::Completion}) {
            for (Kind to : {Kind::Start, Kind::Completion}) {
                auto source = state.event(2 * port), target = source;
                source.kind = from; target.kind = to;
                auto answer = state.eventReachability(source, target);
                if (!answer || queries) { return correlatedQueryFailure(__LINE__); }
                // Independent two-event graph: only I->C and reflexive paths.
                for (bool active : {false, true}) {
                    fs::RegionExpressions::Substitution binding({{present, e->boolean(active)}});
                    const bool expected = active && !(from == Kind::Completion && to == Kind::Start);
                    if (e->constantValue(e->substitute(*answer, binding)) != unsigned(expected)) {
                        return correlatedQueryFailure(__LINE__);
                    }
                }
            }
        }
    }
    llvm::outs() << "correlated-query checks: exact diagonal queries; linear 16/64/256-choice work\n";
    return true;
}
} // namespace mlir::pto::test
#endif
