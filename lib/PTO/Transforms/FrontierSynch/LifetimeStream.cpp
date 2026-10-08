// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/LifetimeStream.h"
#include "PTO/Transforms/FrontierSynch/HardwareProtection.h"
#include <algorithm>
#include <map>
#include <set>
namespace mlir::pto::frontiersynch {
namespace {
using Id = RegionExpressions::Id;
struct Source {
    uint32_t age, site;
    bool operator<(const Source& other) const
    {
        return age != other.age ? age < other.age : site > other.site;
    }
};
std::vector<uint32_t> columns(const LifetimeStreamTemplate& t)
{
    auto result = t.pipes;
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}
bool validTemplate(const LifetimeStreamTemplate& t)
{
    if (t.pipes.empty() || t.pipes.size() > UINT32_MAX || t.span == UINT32_MAX ||
        t.maximumIterations > UINT64_MAX / t.pipes.size() ||
        (!t.operations.empty() && t.operations.size() != t.pipes.size())) { return false; }
    std::set<std::pair<uint32_t, uint32_t>> seen;
    for (const auto& a : t.accesses) {
        if (a.site >= t.pipes.size() || !seen.emplace(a.site, a.cell).second) { return false; }
    }
    for (const auto& edge : t.prerequisites) {
        if (edge.age > t.span || edge.source >= t.pipes.size() || edge.target >= t.pipes.size() ||
            (!edge.age && edge.source >= edge.target)) { return false; }
    }
    return true;
}
LifetimeStreamVisit emptyVisit(RegionExpressions& e, const LifetimeStreamTemplate& t, std::size_t k)
{
    LifetimeStreamVisit visit;
    visit.present.assign(t.pipes.size(), e.boolean(false));
    visit.ranks.assign(t.pipes.size(), e.constant(0));
    visit.completions.assign(t.pipes.size(), std::vector<Id>(k, e.constant(0)));
    visit.accesses.assign(t.accesses.size(), {e.boolean(false), e.boolean(false), e.boolean(true)});
    return visit;
}
bool validVisit(RegionExpressions& e, const LifetimeStreamVisit& visit,
                const LifetimeStreamTemplate& t, std::size_t k)
{
    auto valid = [&](Id value, bool boolean) { return value < e.size() && e.isBoolean(value) == boolean; };
    if (visit.present.size() != t.pipes.size() || visit.ranks.size() != t.pipes.size() ||
        visit.completions.size() != t.pipes.size() || visit.accesses.size() != t.accesses.size()) { return false; }
    for (std::size_t i = 0; i < t.pipes.size(); ++i) {
        if (!valid(visit.present[i], true) || !valid(visit.ranks[i], false) ||
            visit.completions[i].size() != k) { return false; }
        for (auto value : visit.completions[i]) { if (!valid(value, false)) { return false; } }
    }
    for (const auto& access : visit.accesses) {
        if (!valid(access.read, true) || !valid(access.write, true) ||
            !valid(access.noLaterWriter, true)) { return false; }
    }
    return true;
}
} // namespace
LifetimeStreamRegisters initializeLifetimeStream(RegionExpressions& e, const LifetimeStreamTemplate& t)
{
    if (!validTemplate(t)) { return {}; }
    const auto k = columns(t).size();
    LifetimeStreamRegisters registers;
    registers.frontier = initGuardedRankFrontier(e, static_cast<uint32_t>(k));
    registers.history.assign(t.span, emptyVisit(e, t, k));
    return registers;
}
LifetimeStreamTransition advanceLifetimeStream(RegionExpressions& e, const LifetimeStreamTemplate& t,
    const LifetimeStreamRegisters& registers, const LifetimeStreamInputs& input)
{
    LifetimeStreamTransition out;
    auto fail = [&](const char* message) { out.error = message; return out; };
    if (!validTemplate(t) || input.present.size() != t.pipes.size() ||
        input.read.size() != t.accesses.size() || input.write.size() != t.accesses.size() ||
        input.prerequisites.size() != t.prerequisites.size() || registers.history.size() != t.span) {
        return fail("invalid lifetime stream template or register dimensions");
    }
    for (const auto* values : {&input.present, &input.read, &input.write, &input.prerequisites}) {
        for (auto value : *values) {
            if (value >= e.size() || !e.isBoolean(value)) { return fail("invalid lifetime stream input predicate"); }
        }
    }
    const auto pipes = columns(t);
    for (const auto& visit : registers.history) {
        if (!validVisit(e, visit, t, pipes.size())) { return fail("invalid lifetime stream history"); }
    }
    out.next = registers;
    out.current = emptyVisit(e, t, pipes.size());
    out.current.present = input.present;
    std::map<uint32_t, std::vector<std::size_t>> cells;
    std::vector<std::vector<std::size_t>> accesses(t.pipes.size()), prerequisites(t.pipes.size());
    for (std::size_t i = 0; i < t.accesses.size(); ++i) {
        cells[t.accesses[i].cell].push_back(i); accesses[t.accesses[i].site].push_back(i);
    }
    for (std::size_t i = 0; i < t.prerequisites.size(); ++i) { prerequisites[t.prerequisites[i].target].push_back(i); }
    std::map<uint32_t, bool> plain;
    for (const auto& [cell, list] : cells) {
        std::set<uint64_t> operations;
        bool value = true;
        for (auto i : list) {
            if (!t.operations.empty() && t.operations[t.accesses[i].site]) {
                value &= operations.insert(t.operations[t.accesses[i].site]).second;
            }
        }
        plain[cell] = value;
    }
    auto visit = [&](uint32_t age) -> LifetimeStreamVisit& {
        return age ? out.next.history[age - 1] : out.current;
    };
    for (uint32_t target = 0; target < t.pipes.size(); ++target) {
        std::map<Source, Id> candidates, fixed;
        auto add = [&](auto& list, Source source, Id guard) {
            auto [found, inserted] = list.emplace(source, guard);
            if (!inserted) { found->second = e.lor(found->second, guard); }
        };
        for (auto b : accesses[target]) {
            const auto& descriptor = t.accesses[b];
            auto& current = out.current.accesses[b];
            current.read = e.land(input.present[target], input.read[b]);
            current.write = e.land(input.present[target], input.write[b]);
            for (uint32_t age = 0; age <= t.span; ++age) {
                for (auto a : cells[descriptor.cell]) {
                    const auto& source = t.accesses[a];
                    if (!age && source.site >= target) { continue; }
                    auto& previous = visit(age).accesses[a];
                    auto guard = e.lor(e.land(previous.write, e.lor(current.read, current.write)),
                                      e.land(previous.read, current.write));
                    if ((!age && !t.operations.empty() && t.operations[source.site] &&
                         t.operations[source.site] == t.operations[target]) ||
                        t.storageProtection.protectsScalar(t.pipes[source.site], t.pipes[target])) {
                        guard = e.boolean(false);
                    }
                    if ((!age || (source.invariantProtection && descriptor.invariantProtection)) &&
                        hardwareProtectsConflict(t.pipes[source.site], source.protectionGroup,
                                                 t.pipes[target], descriptor.protectionGroup)) {
                        guard = e.land(guard, e.lnot(e.land(previous.write, current.write)));
                    }
                    if (plain[descriptor.cell]) { guard = e.land(guard, previous.noLaterWriter); }
                    add(candidates, {age, source.site}, guard);
                    if (plain[descriptor.cell]) {
                        previous.noLaterWriter = e.land(previous.noLaterWriter, e.lnot(current.write));
                    }
                    if (out.accessPairs == UINT64_MAX) { return fail("lifetime stream pair counter overflow"); }
                    ++out.accessPairs;
                }
            }
        }
        for (auto id : prerequisites[target]) {
            const auto& edge = t.prerequisites[id];
            add(edge.native ? fixed : candidates, {edge.age, edge.source}, input.prerequisites[id]);
        }
        auto incoming = [&](const auto& list) {
            std::vector<GuardedRankIncoming> result;
            for (const auto& [source, predicate] : list) {
                const auto& snapshot = visit(source.age);
                const auto column = std::lower_bound(pipes.begin(), pipes.end(), t.pipes[source.site]) - pipes.begin();
                result.push_back({static_cast<uint32_t>(column), snapshot.ranks[source.site],
                    e.land(predicate, snapshot.present[source.site]), snapshot.completions[source.site]});
            }
            return result;
        };
        const auto column = std::lower_bound(pipes.begin(), pipes.end(), t.pipes[target]) - pipes.begin();
        auto ordinary = incoming(candidates), native = incoming(fixed);
        auto step = advanceGuardedRank(e, out.next.frontier, static_cast<uint32_t>(column),
                                      input.present[target], ordinary, native);
        if (!step.error.empty()) { out.error = step.error; return out; }
        out.current.ranks[target] = step.rank;
        out.current.completions[target] = std::move(step.completion);
        out.starts.push_back(std::move(step.start));
        std::size_t index = 0;
        for (const auto& [source, guard] : candidates) {
            const auto retained = step.retained[index++];
            if (e.constantValue(retained) != 0) {
                out.retained.push_back({{source.age, source.site, target, false}, retained});
            }
        }
    }
    if (t.span) {
        out.next.history.insert(out.next.history.begin(), out.current);
        out.next.history.pop_back();
    }
    out.error = e.constructionError();
    return out;
}
} // namespace mlir::pto::frontiersynch
