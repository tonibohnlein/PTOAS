// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_NUMERICAL_REQUEST_ORACLE_H
#define PTO_NUMERICAL_REQUEST_ORACLE_H
#include "PTO/Transforms/InsertSync/SyncInput.h"
#include "PTO/Transforms/FrontierSynch/Shared/RegionalAnalysis.h"
#include "mlir/IR/Matchers.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>
#include <map>
#include <set>
namespace numerical_request_oracle {
using namespace mlir;
namespace fs = pto::frontiersynch;
// Compare original payload occurrences, not producer-specific type numbers.
// This fixture uses unconditional nested counted loops with constant bounds or
// its one upper-bound argument; unsupported source control fails explicitly.
class Oracle {
    using Key = std::pair<unsigned, std::vector<int64_t>>;
    using Point = std::map<Operation*, int64_t>;
    using Events = std::map<Key, fs::RegionalEvent>;
    struct Payload { SmallVector<scf::ForOp> loops; };
    struct Bounds { int64_t lower, upper, step; };
    func::FuncOp function;
    SmallVector<Payload> payloads;
    llvm::DenseMap<Operation*, unsigned> ids;
    uint64_t upper = 0;
    std::set<Key> present;
    std::optional<int64_t> scalar(Value value) const
    {
        APInt constant;
        if (matchPattern(value, m_ConstantInt(&constant)) && constant.isSignedIntN(64)) {
            return constant.getSExtValue();
        }
        auto callable = function;
        if (callable.getNumArguments() == 1 && value == callable.getArgument(0) && upper <= uint64_t(INT64_MAX)) {
            return static_cast<int64_t>(upper);
        }
        return {};
    }
    std::optional<Bounds> bounds(scf::ForOp loop) const
    {
        auto lower = scalar(loop.getLowerBound()), limit = scalar(loop.getUpperBound()), step = scalar(loop.getStep());
        if (!lower || !limit || !step || *step <= 0) { return {}; }
        return Bounds{*lower, *limit, *step};
    }
    static APInt wide(int64_t value) { return APInt(128, static_cast<uint64_t>(value), true); }
    static std::optional<int64_t> induction(const Bounds& limits, uint64_t ordinal)
    {
        auto value = wide(limits.lower) + APInt(128, ordinal) * wide(limits.step);
        return value.isSignedIntN(64) ? std::optional<int64_t>(value.getSExtValue()) : std::nullopt;
    }
    static std::optional<int64_t> trips(const Bounds& limits)
    {
        if (limits.upper <= limits.lower) { return 0; }
        auto span = wide(limits.upper) - wide(limits.lower);
        auto count = span.udiv(wide(limits.step)) + APInt(128, !span.urem(wide(limits.step)).isZero());
        return count.isSignedIntN(64) ? std::optional<int64_t>(count.getSExtValue()) : std::nullopt;
    }
    std::optional<uint64_t> evaluate(const fs::RegionalAnalysis& region, fs::RegionExpressions::Id root) const
    {
        auto& e = *region.expressions;
        SmallVector<std::pair<fs::RegionExpressions::Id, fs::RegionExpressions::Id>> bindings;
        for (auto [id, input] : e.referencedInputs(root)) {
            auto value = scalar(input);
            if (!value) { return {}; }
            bindings.push_back({id, e.constant(static_cast<uint64_t>(*value))});
        }
        fs::RegionExpressions::Substitution substitution(bindings);
        return e.constantValue(e.substitute(root, substitution));
    }
    std::optional<Key> key(const fs::RegionalAnalysis& region, const fs::RegionalEvent& event) const
    {
        if (event.type >= region.anchors.size() || region.occurrenceLoops.size() != region.anchors.size()) {
            return {};
        }
        const auto& anchor = region.anchors[event.type];
        auto id = anchor.phase ? ids.find(anchor.phase->elementOp) : ids.end();
        if (id == ids.end()) { return {}; }
        Point coordinates;
        auto add = [&](scf::ForOp loop, int64_t induction) {
            if (!loop) { return false; }
            auto [entry, added] = coordinates.emplace(loop.getOperation(), induction);
            return added || entry->second == induction;
        };
        for (auto coordinate : anchor.coordinates) {
            if (!add(coordinate.loop, coordinate.induction)) { return {}; }
        }
        auto ordinal = [&](scf::ForOp loop, fs::RegionExpressions::Id expression) {
            auto value = evaluate(region, expression);
            auto limits = bounds(loop);
            if (!value || !limits) { return false; }
            auto coordinate = induction(*limits, *value);
            return coordinate && add(loop, *coordinate);
        };
        auto leaf = region.occurrenceLoops[event.type];
        if (leaf ? !ordinal(leaf, event.ordinal) : evaluate(region, event.ordinal) != 0) { return {}; }
        if (region.outerLoops.empty()) { if (!event.visits.empty()) { return {}; } }
        else {
            if (region.outerLoops.size() != region.anchors.size() ||
                event.visits.size() != region.outerLoops[event.type].size()) { return {}; }
            if (!region.outerDivisors.empty() &&
                (region.outerDivisors.size() != region.anchors.size() ||
                 region.outerDivisors[event.type].size() != event.visits.size())) { return {}; }
            for (auto [i, loop] : llvm::enumerate(region.outerLoops[event.type])) {
                if ((!region.outerDivisors.empty() && region.outerDivisors[event.type][i] != 1) ||
                    !ordinal(loop, event.visits[i])) { return {}; }
            }
        }
        Key result{id->second, {}};
        for (auto loop : payloads[id->second].loops) {
            auto found = coordinates.find(loop.getOperation());
            if (found == coordinates.end()) { return {}; }
            result.second.push_back(found->second);
        }
        if (coordinates.size() != result.second.size()) { return {}; }
        return result;
    }
    bool inventory(const fs::RegionalAnalysis& region, Events& events)
    {
        if (region.occurrenceLoops.size() != region.anchors.size()) { return false; }
        for (uint32_t type = 0; type < region.anchors.size(); ++type) {
            SmallVector<scf::ForOp> frames;
            if (!region.outerLoops.empty()) {
                if (region.outerLoops.size() != region.anchors.size()) { return false; }
                llvm::append_range(frames, region.outerLoops[type]);
            }
            const auto outerCount = frames.size();
            if (region.occurrenceLoops[type]) { frames.push_back(region.occurrenceLoops[type]); }
            fs::RegionalEvent event{type, region.expressions->constant(0), fs::PeriodicEventKind::Start};
            event.visits.resize(outerCount);
            std::function<bool(unsigned)> enumerate = [&](unsigned position) {
                if (position == frames.size()) {
                    auto point = key(region, event);
                    auto expression = fs::regionalPresence(region, event);
                    auto active = expression ? evaluate(region, *expression) : std::nullopt;
                    if (!point || !active || *active != unsigned(present.count(*point))) { return false; }
                    auto [entry, added] = events.emplace(*point, event);
                    return added || !*active; // An active original occurrence has one exported representation.
                }
                auto limits = bounds(frames[position]);
                if (!limits) { return false; }
                const auto count = trips(*limits);
                if (!count) { return false; }
                // Include absent visits in zero/partial invocations, as well as
                // every actual visit. These are test valuations, not an admission limit.
                for (int64_t visit = 0; visit < std::max<int64_t>(*count, 3); ++visit) {
                    auto expression = region.expressions->constant(visit);
                    if (position < outerCount) { event.visits[position] = expression; }
                    else { event.ordinal = expression; }
                    if (!enumerate(position + 1)) { return false; }
                }
                return true;
            };
            if (!enumerate(0)) { return false; }
        }
        for (const auto& point : present) { if (!events.count(point)) { return false; } }
        return true;
    }
    using Selection = std::vector<std::pair<Key, unsigned>>;
    std::optional<Selection> selected(const fs::RegionalAnalysis& region,
        const std::vector<fs::RegionalSelector>& selectors) const
    {
        Selection result;
        for (const auto& selector : selectors) {
            auto active = evaluate(region, selector.present);
            if (!active || *active > 1) { return {}; }
            if (!*active) { continue; }
            auto point = key(region, selector.event);
            if (!point || !present.count(*point)) { return {}; }
            result.push_back({*point, unsigned(selector.event.kind)});
        }
        llvm::sort(result);
        return result;
    }
    bool selectors(const fs::RegionalAnalysis& a, const fs::RegionalStorageBoundary& x,
        const fs::RegionalAnalysis& b, const fs::RegionalStorageBoundary& y) const
    {
        auto same = [&](const auto& left, const auto& right) {
            auto p = selected(a, left), q = selected(b, right);
            return p && q && *p == *q;
        };
        if (!same(x.firstWriters, y.firstWriters) || !same(x.lastWriters, y.lastWriters)) { return false; }
        for (bool last : {false, true}) {
            const auto& left = last ? x.lastReaders : x.firstReaders;
            const auto& right = last ? y.lastReaders : y.firstReaders;
            std::set<uint32_t> pipes;
            for (const auto& entry : left) { pipes.insert(entry.first); }
            for (const auto& entry : right) { pipes.insert(entry.first); }
            const std::vector<fs::RegionalSelector> empty;
            for (auto pipe : pipes) {
                auto p = left.find(pipe), q = right.find(pipe);
                if (!same(p == left.end() ? empty : p->second, q == right.end() ? empty : q->second)) { return false; }
            }
        }
        return true;
    }
    bool storage(const fs::RegionalAnalysis& a, const fs::RegionalAnalysis& b) const
    {
        // Common interval refinement permits different exact cell partitions.
        for (const auto& x : a.storageBoundary) {
            auto cursor = x.cell.begin;
            while (cursor < x.cell.end) {
                const fs::RegionalStorageBoundary* match = nullptr;
                for (const auto& y : b.storageBoundary) {
                    if (pto::sameStorageDomain(x.cell, y.cell) && y.cell.begin <= cursor && cursor < y.cell.end) {
                        if (match) { return false; }
                        match = &y;
                    }
                }
                if (!match || !selectors(a, x, b, *match)) { return false; }
                cursor = std::min(x.cell.end, match->cell.end);
            }
        }
        return true;
    }
public:
    explicit Oracle(func::FuncOp function) : function(function) {}
    bool initialize(const pto::SyncInput& input, scf::ForOp root)
    {
        for (auto* phase : input.instructions()) {
            auto* operation = phase->elementOp;
            if (!root->isProperAncestor(operation)) { continue; }
            if (ids.count(operation)) { return false; }
            Payload payload;
            for (auto* parent = operation->getParentOp(); parent != function.getOperation();
                 parent = parent->getParentOp()) {
                if (!parent) { return false; }
                auto loop = dyn_cast<scf::ForOp>(parent);
                if (!loop) { return false; }
                payload.loops.push_back(loop);
            }
            std::reverse(payload.loops.begin(), payload.loops.end());
            ids[operation] = payloads.size(); payloads.push_back(std::move(payload));
        }
        return !payloads.empty();
    }
    bool compare(const fs::RegionalAnalysis& a, const fs::RegionalAnalysis& b, uint64_t bound)
    {
        upper = bound; present.clear();
        for (const auto& entry : llvm::enumerate(payloads)) {
            const auto& payload = entry.value();
            Key point{static_cast<unsigned>(entry.index()), {}};
            std::function<bool(unsigned)> enumerate = [&](unsigned position) {
                if (position == payload.loops.size()) { present.insert(point); return true; }
                auto limits = bounds(payload.loops[position]);
                if (!limits) { return false; }
                const auto count = trips(*limits);
                if (!count) { return false; }
                for (int64_t visit = 0; visit < *count; ++visit) {
                    auto value = induction(*limits, visit);
                    if (!value) { return false; }
                    point.second.push_back(*value);
                    if (!enumerate(position + 1)) { return false; }
                    point.second.pop_back();
                }
                return true;
            };
            if (!enumerate(0)) { return false; }
        }
        Events left, right;
        if (!inventory(a, left) || !inventory(b, right) || !storage(a, b) || !storage(b, a)) { return false; }
        std::set<Key> points;
        for (const auto& entry : left) { points.insert(entry.first); }
        for (const auto& entry : right) { points.insert(entry.first); }
        auto query = [&](const fs::RegionalAnalysis& region, const Events& events, const Key& from, const Key& to)
            -> std::optional<uint64_t> {
            auto x = events.find(from), y = events.find(to);
            if (x == events.end() || y == events.end()) { return 0; } // Unrepresented absent occurrence.
            auto source = x->second, target = y->second;
            source.kind = fs::PeriodicEventKind::Completion;
            auto answer = fs::regionalReachability(region, source, target);
            return answer ? evaluate(region, *answer) : std::nullopt;
        };
        for (const auto& from : points) {
            for (const auto& to : points) {
                auto p = query(a, left, from, to), q = query(b, right, from, to);
                if (!p || !q || *p != *q || ((!present.count(from) || !present.count(to)) && *p)) { return false; }
            }
        }
        return true;
    }
};
} // namespace numerical_request_oracle
#endif
