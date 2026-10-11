// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Sufficient allocation witnesses, including optional endpoint participation.
#include "PTO/Transforms/FrontierSynch/Structured/BoundedLifetimeAllocation.h"
#include "PTO/Transforms/FrontierSynch/Shared/HardwareProtection.h"
#include "PTO/Transforms/FrontierSynch/Shared/RegionalAllocation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
namespace fs = mlir::pto::frontiersynch;
namespace {
std::optional<int64_t> budget(mlir::DictionaryAttr certificate, unsigned groups = 1)
{
    if (!certificate) { return std::nullopt; }
    auto list = certificate.getAs<mlir::ArrayAttr>("groups");
    if (!list || list.size() != groups || !groups) { return std::nullopt; }
    auto first = mlir::dyn_cast<mlir::DictionaryAttr>(list[0]);
    auto value = first ? first.getAs<mlir::IntegerAttr>("budget") : mlir::IntegerAttr{};
    return value ? std::optional<int64_t>(value.getInt()) : std::nullopt;
}
fs::LifetimeWindowInput window(fs::RegionExpressions& e, unsigned slots)
{
    fs::LifetimeWindowInput result;
    result.sites = 2; result.span = slots;
    for (unsigned d = 0; d <= slots; ++d) {
        // One writer and one optionally participating reader of its bank.
        result.payloads.push_back({0, e.boolean(true)});
        result.payloads.push_back({1, e.boolean(true)});
        result.accesses.push_back({2 * d, d % slots, e.boolean(false), e.boolean(true), 0});
        result.accesses.push_back({2 * d + 1, d % slots, e.boolean(true), e.boolean(false), 0});
    }
    return result;
}
bool ordinary(mlir::func::FuncOp function)
{
    fs::RegionExpressions e;
    auto w = window(e, 2);
    std::vector<fs::GuardedRankEdge> demands{{0, 1, e.boolean(true)}};
    auto certify = [&](llvm::ArrayRef<uint8_t> always) {
        return fs::boundedLifetimeAllocationCertificate(function, e, w, always, demands, 5);
    };
    // An optional reader releases the bank to the mandatory producer two
    // iterations later. No intermediate optional execution may be assumed.
    if (budget(certify({1, 0})) != 2 || certify({0, 0})) { return false; }
    // A prerequisite between producers cannot invent an earlier return
    // from the optional reader before its bank is touched again.
    w.prerequisites.push_back({0, 2, e.boolean(true)});
    if (budget(certify({1, 0})) != 2) { return false; }
    auto malformed = w;
    malformed.accesses[0].payload = malformed.payloads.size();
    if (fs::boundedLifetimeAllocationCertificate(function, e, malformed, {1, 0}, demands, 0)) { return false; }
    malformed = w;
    malformed.operations = {1};
    return !fs::boundedLifetimeAllocationCertificate(function, e, malformed, {1, 0}, demands, 0);
}
bool palettes(mlir::func::FuncOp function)
{
    fs::RegionExpressions e;
    auto w = window(e, 1);
    // The producer contract permits at most one active partner on a pipe.
    // Alternative records of one source site share a single cyclic palette.
    std::vector<fs::GuardedRankEdge> demands{{0, 1, e.boolean(true)}, {0, 3, e.boolean(false)}};
    auto certificate = fs::boundedLifetimeAllocationCertificate(function, e, w, {1, 0}, demands, 0);
    if (budget(certificate) != 2) { return false; }
    auto group = mlir::cast<mlir::DictionaryAttr>(certificate.getAs<mlir::ArrayAttr>("groups")[0]);
    auto records = group.getAs<mlir::DenseI64ArrayAttr>("records");
    auto rules = group.getAs<mlir::ArrayAttr>("tuple_rules");
    if (!records || records.size() != 2 || records[0] != 0 || records[1] != 1 || !rules || rules.size() != 2) {
        return false;
    }
    auto rule = mlir::cast<mlir::DictionaryAttr>(rules[0]);
    if (rule.getAs<mlir::IntegerAttr>("coordinate_count").getInt() != 1) { return false; }
    // An empty demand set is a valid empty physical plan even without any
    // guaranteed execution; there is no event lane to justify.
    auto empty = fs::boundedLifetimeAllocationCertificate(function, e, w, {0, 0}, {}, 0);
    return empty && empty.getAs<mlir::ArrayAttr>("groups").empty();
}
bool sharedDirections(mlir::func::FuncOp function)
{
    fs::RegionExpressions e;
    auto w = window(e, 1);
    std::vector<fs::GuardedRankEdge> demands{{0, 1, e.boolean(true)}, {1, 2, e.boolean(true)}};
    auto certificate = fs::boundedLifetimeAllocationCertificate(function, e, w, {1, 1}, demands, 0);
    if (budget(certificate, 2) != 1) { return false; }
    auto groups = certificate.getAs<mlir::ArrayAttr>("groups");
    for (unsigned i = 0; i < 2; ++i) {
        auto group = mlir::cast<mlir::DictionaryAttr>(groups[i]);
        auto sources = group.getAs<mlir::DenseI64ArrayAttr>("sources");
        auto targets = group.getAs<mlir::DenseI64ArrayAttr>("targets");
        if (!sources || !targets || sources.size() != 1 || targets.size() != 1 ||
            sources[0] != i || targets[0] != 1 - i) { return false; }
    }
    return true;
}
bool constantDirections()
{
    fs::RegionalAnalysis region;
    region.expressions = std::make_shared<fs::RegionExpressions>();
    region.capabilities.exactQueries = true;
    auto& e = *region.expressions;
    fs::RegionalAllocationMember first;
    first.firstSource.ordinal = first.lastTarget.ordinal = e.constant(0);
    first.lastTarget.kind = fs::PeriodicEventKind::Completion;
    // Empty visits make these lifetimes compatible; direction must still be retained.
    first.active = e.boolean(false); first.sourcePipe = 0; first.targetPipe = 1;
    auto second = first; second.record = 1; second.sourcePipe = 1; second.targetPipe = 0;
    auto third = second; third.record = 2;
    fs::RegionalAllocationSummary input;
    input.groups.push_back({0, 1, 1, {first, second}});
    input.groups.push_back({1, 0, 1, {third}});
    auto output = fs::coalesceRegionalAllocation(region, input);
    if (!output || output->groups.size() != 2) { return false; }
    for (const auto& group : output->groups) {
        for (const auto& member : group.members) {
            if (member.sourcePipe != group.sourcePipe || member.targetPipe != group.targetPipe) { return false; }
        }
    }
    return output->groups[0].members.size() == 1 && output->groups[1].members.size() == 2;
}
bool protectedModes(mlir::func::FuncOp function)
{
    fs::RegionExpressions e;
    fs::LifetimeWindowInput w;
    w.sites = 3; w.span = 1;
    // Two mandatory accumulator phases on pipe 1 share a protected writer
    // group. Pipe 0 remains an ordinary conflicting access to their cell.
    for (unsigned i = 0; i < 6; ++i) {
        w.payloads.push_back({i % 3 ? 1U : 0U, e.boolean(true)});
        w.accesses.push_back({i, 0, e.boolean(true), e.boolean(true), i % 3 ? 1U : 0U});
    }
    std::vector<fs::GuardedRankEdge> demands{{0, 1, e.boolean(true)}};
    auto combined = fs::boundedLifetimeAllocationCertificate(function, e, w, {1, 1, 1}, demands, 0);
    auto split = w;
    split.accesses.clear();
    for (auto access : w.accesses) {
        access.write = e.boolean(false); split.accesses.push_back(access);
        access.read = e.boolean(false); access.write = e.boolean(true); split.accesses.push_back(access);
    }
    auto separated = fs::boundedLifetimeAllocationCertificate(function, e, split, {1, 1, 1}, demands, 0);
    return combined && combined == separated;
}
// Two optional endpoint sites are bracketed by guaranteed same-pipe
// occurrences. A mandatory writer refreshes each of two banks every two visits.
fs::LifetimeWindowInput guardedWindow(fs::RegionExpressions& e)
{
    fs::LifetimeWindowInput result;
    result.sites = 5; result.span = 2;
    for (unsigned d = 0; d <= result.span; ++d) {
        for (unsigned site = 0; site < result.sites; ++site) {
            result.payloads.push_back({site == 2 || site == 3 ? 1U : 0U, e.boolean(true)});
        }
        result.accesses.push_back({5 * d, d % 2, e.boolean(false), e.boolean(true), 0});
        result.accesses.push_back({5 * d + 1, d % 2, e.boolean(false), e.boolean(true), 0});
        result.accesses.push_back({5 * d + 2, d % 2, e.boolean(true), e.boolean(false), 0});
    }
    return result;
}
bool checkGuardedEnvelopes(fs::RegionExpressions& e, const fs::RegionalAllocationSummary& allocation,
    unsigned trips)
{
    if (allocation.groups.size() != 1) { return false; }
    const auto& group = allocation.groups.front();
    if (group.budget != 2 || group.members.size() != 1 || group.lanes.size() != 2) { return false; }
    const auto& member = group.members.front();
    if (member.record != 0 || member.singletonHandoff || !member.tupleRule ||
        member.tupleRule->coordinateCount != 1 || member.tupleRule->terms.size() != 1 ||
        member.firstSource.type != 0 || member.lastTarget.type != 3 ||
        member.firstSource.kind != fs::PeriodicEventKind::Start ||
        member.lastTarget.kind != fs::PeriodicEventKind::Completion ||
        e.constantValue(member.active) != std::optional<uint64_t>(trips != 0) ||
        e.constantValue(member.firstSource.ordinal) != 0 ||
        e.constantValue(member.lastTarget.ordinal) != (trips ? trips - 1 : 0)) { return false; }
    const auto& rule = *member.tupleRule;
    const auto& term = rule.terms.front();
    if (term.coordinate != 0 || term.stride != 1 || term.modulus != 2 || term.scale != 1) { return false; }
    // Independently enumerate the two endpoint-presence bits in each visit.
    // Neither optional site is an enclosing lifetime bound. On either native
    // pipe, (visit, site) order brackets every actually active notification.
    for (unsigned schedule = 0; schedule < (1U << (2 * trips)); ++schedule) {
        std::map<uint64_t, unsigned> prior;
        for (unsigned visit = 0; visit < trips; ++visit) {
            if (!((schedule >> (2 * visit)) & 1U) || !((schedule >> (2 * visit + 1)) & 1U)) { continue; }
            const auto lane = rule.base + (visit + term.phase) % term.modulus;
            if (lane >= group.lanes.size() || group.lanes[lane].firstSources.empty() ||
                group.lanes[lane].lastTargets.empty()) { return false; }
            for (const auto& first : group.lanes[lane].firstSources) {
                auto ordinal = e.constantValue(first.event.ordinal);
                if (!ordinal || first.event.type != 0 || *ordinal > visit ||
                    e.constantValue(first.present) != 1) { return false; }
            }
            for (const auto& last : group.lanes[lane].lastTargets) {
                auto ordinal = e.constantValue(last.event.ordinal);
                if (!ordinal || last.event.type != 3 || *ordinal < visit ||
                    e.constantValue(last.present) != 1) { return false; }
            }
            auto found = prior.find(lane);
            // Equal labels reuse only after the independent physical-bank
            // refresh, even if all intervening optional sites are absent.
            if (found != prior.end() && visit - found->second < 2) { return false; }
            prior[lane] = visit;
        }
    }
    return true;
}
bool regionalOrdinary(mlir::func::FuncOp function)
{
    fs::RegionExpressions e;
    auto w = guardedWindow(e);
    std::vector<fs::GuardedRankEdge> demands{{1, 2, e.boolean(true)}};
    const std::vector<uint8_t> always{1, 0, 0, 1, 1};
    if (budget(fs::boundedLifetimeAllocationCertificate(function, e, w, always, demands, 0)) != 2) {
        return false;
    }
    // Raw induction values with a non-unit step map to the same ordinal
    // lifetime family. These counts include zero, one and partial periods.
    for (unsigned trips = 0; trips <= 5; ++trips) {
        const unsigned lower = 11, step = 7, upper = lower + trips * step;
        const auto count = (upper - lower + step - 1) / step;
        auto allocation = fs::boundedLifetimeRegionalAllocation(e, e, w, always, demands, e.constant(count));
        if (!allocation || !checkGuardedEnvelopes(e, *allocation, trips)) { return false; }
    }
    // Original guarded demand expressions and the composing region's arena
    // are independent. Deliberately separate their numerical ID ranges.
    fs::RegionExpressions output;
    for (unsigned i = 0; i < e.size() + 16; ++i) { output.constant(1000 + i); }
    auto separate = fs::boundedLifetimeRegionalAllocation(e, output, w, always, demands, output.constant(5));
    if (!separate || !checkGuardedEnvelopes(output, *separate, 5)) { return false; }
    auto noLast = always;
    noLast[3] = 0;
    // The ordinary whole-loop proof can succeed while a composable enclosing
    // lifetime is unavailable. Preserve that distinction explicitly.
    if (!fs::boundedLifetimeAllocationCertificate(function, e, w, noLast, demands, 0) ||
        fs::boundedLifetimeRegionalAllocation(e, e, w, noLast, demands, e.constant(5))) { return false; }
    auto empty = fs::boundedLifetimeRegionalAllocation(e, e, w, {0, 0, 0, 0, 0}, {}, e.constant(0));
    if (!empty || !empty->groups.empty() ||
        fs::boundedLifetimeRegionalAllocation(e, e, w, always, demands, e.boolean(true))) { return false; }
    auto malformed = w;
    malformed.payloads.pop_back();
    return !fs::boundedLifetimeRegionalAllocation(e, e, malformed, always, demands, e.constant(5));
}
bool regionalDelayed()
{
    fs::RegionExpressions producer, output;
    auto w = guardedWindow(producer);
    std::vector<fs::GuardedRankEdge> demands{{1, 12, producer.boolean(true)}};
    for (unsigned trips = 0; trips <= 6; ++trips) {
        auto allocation = fs::boundedLifetimeRegionalAllocation(producer, output, w,
            {1, 0, 0, 1, 1}, demands, output.constant(trips));
        if (!allocation || allocation->groups.size() != 1) { return false; }
        const auto& member = allocation->groups.front().members.front();
        if (output.constantValue(member.active) != std::optional<uint64_t>(trips > 2) ||
            output.constantValue(member.lastTarget.ordinal) != (trips > 2 ? trips - 1 : 0)) { return false; }
        const auto& rule = *member.tupleRule;
        const auto& term = rule.terms.front();
        for (unsigned source = 0; source + 2 < trips; ++source) {
            const auto target = source + 2;
            const auto sourceLane = rule.base + (source + term.phase) % term.modulus;
            const auto targetLane = rule.base + (target - 2 + term.phase) % term.modulus;
            if (sourceLane != targetLane || sourceLane >= allocation->groups.front().budget) { return false; }
        }
    }
    return true;
}
bool regionalRollback()
{
    fs::RegionExpressions producer, output;
    auto w = guardedWindow(producer);
    std::vector<fs::GuardedRankEdge> demands{{0, 2, producer.boolean(true)}, {1, 3, producer.boolean(true)}};
    const auto trips = output.constant(7);
    const auto before = output.size();
    if (fs::boundedLifetimeRegionalAllocation(producer, output, w, {1, 0, 1, 0, 1}, demands, trips) ||
        output.size() != before || !output.constructionError().empty()) { return false; }
    if (!fs::boundedLifetimeRegionalAllocation(producer, output, w, {1, 0, 1, 1, 1}, demands, trips)) {
        return false;
    }
    output.add(fs::RegionExpressions::invalid, trips);
    const auto poisonedSize = output.size();
    return !output.constructionError().empty() &&
        !fs::boundedLifetimeRegionalAllocation(producer, output, w, {1, 0, 1, 1, 1}, demands, trips) &&
        output.size() == poisonedSize;
}
} // namespace
int runBoundedLifetimeAllocationChecks()
{
    mlir::MLIRContext context;
    context.loadDialect<mlir::func::FuncDialect>();
    mlir::Builder builder(&context);
    mlir::OwningOpRef<mlir::func::FuncOp> function(mlir::func::FuncOp::create(
        builder.getUnknownLoc(), "bounded_allocation_checks", builder.getFunctionType({}, {})));
    if (!ordinary(*function) || !palettes(*function) || !sharedDirections(*function) ||
        !constantDirections() || !protectedModes(*function) || !regionalOrdinary(*function) ||
        !regionalDelayed() || !regionalRollback()) {
        llvm::errs() << "bounded lifetime allocation certificate check failed\n";
        return 1;
    }
    llvm::outs() << "bounded lifetime allocation: optional endpoints, reuse gaps and mode merging passed\n";
    return 0;
}
