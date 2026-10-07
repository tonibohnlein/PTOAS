// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Command-order thresholds and a checked shared-cycle certificate adapter.
#include "PTO/Transforms/FrontierSynch/PeriodicSharedCertificate.h"
#include "PTO/Transforms/FrontierSynch/PeriodicSharedAllocation.h"
#include "PTO/IR/PTO.h"
#include <map>
#include <set>

namespace mlir::pto::frontiersynch {
namespace {
bool concretePipe(int64_t value)
{
    return value >= 0 && value <= static_cast<int64_t>(PIPE::PIPE_FIX) &&
           value != static_cast<int64_t>(PIPE::PIPE_ALL);
}
// The analysis index is owned by the caller and used only during encoding.
bool collect(const PeriodicAnalysis& analysis, std::vector<uint32_t>& records,
             std::vector<PeriodicRecord>& barriers, bool& exact)
{
    if (!analysis.error.empty() || analysis.payloads.size() >= UINT32_MAX) { return false; }
    std::map<uint32_t, std::vector<uint32_t>> pipes;
    for (uint32_t i = 0; i < analysis.payloads.size(); ++i) {
        if (!concretePipe(analysis.payloads[i].pipe)) { return false; }
        pipes[analysis.payloads[i].pipe].push_back(i);
    }
    std::vector<uint32_t> previous(analysis.payloads.size());
    for (const auto& pipe : pipes) {
        auto prior = pipe.second.back();
        for (auto type : pipe.second) { previous[type] = prior; prior = type; }
    }
    std::set<uint32_t> seen;
    for (auto id : analysis.retained) {
        if (id >= analysis.generators.size() || !seen.insert(id).second) { return false; }
        const auto& r = analysis.generators[id];
        if (r.source >= previous.size() || r.target >= previous.size() ||
            (!r.displacement && r.source >= r.target)) { return false; }
        if (analysis.payloads[r.source].pipe != analysis.payloads[r.target].pipe) {
            records.push_back(id);
            continue;
        }
        const auto prior = previous[r.target];
        const uint64_t wrap = prior >= r.target ? 1 : 0;
        if (r.displacement > wrap) {
            // This barrier is absent in early iterations. Treating its adjacent
            // predecessor edge as unconditional would invent startup order.
            exact = false;
        } else if (r.source != prior || r.displacement != wrap) {
            barriers.push_back({prior, r.target, wrap});
        }
    }
    return true;
}
bool reuseWeights(const PeriodicAnalysis& original, const PeriodicAnalysis& order,
                  const std::vector<uint32_t>& records, PeriodicReuseMatrix& weights)
{
    weights.assign(records.size(), std::vector<std::optional<uint64_t>>(records.size()));
    for (std::size_t i = 0; i < records.size(); ++i) {
        const auto& current = original.generators[records[i]];
        for (std::size_t j = 0; j < records.size(); ++j) {
            const auto& next = original.generators[records[j]];
            const bool same = original.payloads[current.target].pipe == original.payloads[next.source].pipe;
            auto query = order.eventThreshold(
                {current.target, same ? PeriodicEventKind::Start : PeriodicEventKind::Completion},
                {next.source, PeriodicEventKind::Start});
            if (query.error != PeriodicQueryError::None) { return false; }
            if (!query.displacement) { continue; }
            if (*query.displacement > UINT64_MAX - current.displacement) { return false; }
            weights[i][j] = current.displacement + *query.displacement;
        }
    }
    return true;
}
std::optional<int64_t> number(DictionaryAttr dictionary, StringRef key)
{
    auto attr = dictionary.getAs<IntegerAttr>(key);
    if (!attr || !attr.getType().isInteger(64)) { return std::nullopt; }
    return attr.getInt();
}
struct Entry {
    int64_t record = 0, source = 0, target = 0, begin = 0, count = 0, offset = 0;
};
std::optional<Entry> decodeEntry(Attribute raw, int64_t budget)
{
    auto d = dyn_cast<DictionaryAttr>(raw);
    if (!d) { return std::nullopt; }
    auto record = number(d, "record"), source = number(d, "source"), target = number(d, "target");
    auto begin = number(d, "lane_begin"), count = number(d, "lane_count"), offset = number(d, "offset");
    if (!record || !source || !target || !begin || !count || !offset || *record < 0 ||
        !concretePipe(*source) || !concretePipe(*target) || *source == *target || *begin < 0 ||
        *count <= 0 || *offset < 0 || *offset >= *count || *begin > budget || *count > budget - *begin) {
        return std::nullopt;
    }
    return Entry{*record, *source, *target, *begin, *count, *offset};
}
bool decodeEntries(ArrayAttr raw, int64_t budget, std::vector<Entry>& entries)
{
    std::set<int64_t> records;
    std::map<int64_t, int64_t> ranges;
    for (auto attr : raw) {
        auto entry = decodeEntry(attr, budget);
        if (!entry || !records.insert(entry->record).second) { return false; }
        auto range = ranges.emplace(entry->begin, entry->count);
        if (!range.second && range.first->second != entry->count) { return false; }
        entries.push_back(*entry);
    }
    int64_t end = 0;
    for (const auto& range : ranges) {
        if (range.first != end) { return false; }
        end += range.second; // decodeEntry bounded each range by budget <= INT64_MAX.
    }
    return end == budget;
}
} // namespace
DictionaryAttr encodePeriodicSharedAllocation(const PeriodicAnalysis& analysis, int64_t plan, MLIRContext* context)
{
    if (!context || plan < 0) { return {}; }
    bool exact = true;
    std::vector<uint32_t> records;
    std::vector<PeriodicRecord> barriers;
    if (!collect(analysis, records, barriers, exact)) { return {}; }
    PeriodicAnalysis reconstructed;
    const PeriodicAnalysis* order = &analysis;
    if (exact && !barriers.empty()) {
        auto native = analysis.nativePrerequisites;
        native.insert(native.end(), barriers.begin(), barriers.end());
        reconstructed = analyzePeriodicDemands(analysis.payloads, analysis.generators, native);
        if (!reconstructed.error.empty()) { return {}; }
        order = &reconstructed;
    }
    PeriodicReuseMatrix weights;
    if (!reuseWeights(analysis, *order, records, weights)) { return {}; }
    auto allocation = allocatePeriodicShared(weights);
    if (allocation.status != PeriodicSharedAllocationStatus::Success || allocation.budget > INT64_MAX) { return {}; }
    // A selected edge has shift d(current)+h, with h>=0. Along a lane chain,
    // source ordinals never decrease and each consumer is no later than the
    // next source. Between two active handoffs every intermediate endpoint
    // therefore belongs to the same prefix. Removed startup/terminal handoffs
    // cannot break reuse; forward event witnesses stay in that prefix too.
    // Fresh entry and paired commands give no preexisting notification, and
    // every emitted SET has its in-prefix WAIT, so exit is closed. Zero trips
    // execute no commands. These facts require an unconditional skeleton.
    Builder b(context);
    SmallVector<Attribute> entries;
    for (std::size_t i = 0; i < records.size(); ++i) {
        const auto& r = analysis.generators[records[i]];
        const auto& p = allocation.phases[i];
        entries.push_back(b.getDictionaryAttr({
            b.getNamedAttr("record", b.getI64IntegerAttr(records[i])),
            b.getNamedAttr("source", b.getI64IntegerAttr(analysis.payloads[r.source].pipe)),
            b.getNamedAttr("target", b.getI64IntegerAttr(analysis.payloads[r.target].pipe)),
            b.getNamedAttr("lane_begin", b.getI64IntegerAttr(static_cast<int64_t>(p.laneBegin))),
            b.getNamedAttr("lane_count", b.getI64IntegerAttr(static_cast<int64_t>(p.laneCount))),
            b.getNamedAttr("offset", b.getI64IntegerAttr(static_cast<int64_t>(p.offset)))}));
    }
    return b.getDictionaryAttr({b.getNamedAttr("version", b.getI64IntegerAttr(2)),
        b.getNamedAttr("plan", b.getI64IntegerAttr(plan)),
        b.getNamedAttr("strategy", b.getStringAttr("shared-cycle-cover")),
        b.getNamedAttr("order_exact", b.getBoolAttr(exact)),
        b.getNamedAttr("budget", b.getI64IntegerAttr(static_cast<int64_t>(allocation.budget))),
        b.getNamedAttr("entries", b.getArrayAttr(entries))});
}
FailureOr<PhysicalAllocationPlan> decodePeriodicSharedAllocation(
    func::FuncOp function, DictionaryAttr certificate, ArrayRef<int64_t> eligibleIds)
{
    if (!function || !certificate) { return failure(); }
    auto version = number(certificate, "version"), plan = number(certificate, "plan");
    auto budget = number(certificate, "budget");
    auto strategy = certificate.getAs<StringAttr>("strategy");
    auto exact = certificate.getAs<BoolAttr>("order_exact");
    auto raw = certificate.getAs<ArrayAttr>("entries");
    if (!version || *version != 2 || !plan || *plan < 0 || !budget || *budget < 0 || !strategy ||
        strategy.getValue() != "shared-cycle-cover" || !exact || !raw) {
        return function.emitError("malformed periodic shared-cycle certificate"), failure();
    }
    std::vector<Entry> entries;
    if (!decodeEntries(raw, *budget, entries)) {
        return function.emitError("invalid periodic shared-cycle entries or lane ranges"), failure();
    }
    std::set<int64_t> seenIds;
    for (auto id : eligibleIds) {
        if (id < 0 || id >= 6 || !seenIds.insert(id).second) {
            return function.emitError("periodic shared-cycle eligible IDs must be distinct values in 0..5"), failure();
        }
    }
    if (static_cast<uint64_t>(*budget) > eligibleIds.size()) {
        return function.emitError("minimum of supplied periodic cycle-cover graph needs ") << *budget
            << " shared IDs, only " << eligibleIds.size() << " available; order_exact=" << exact.getValue()
            << "; no universal fixed-plan or hardware minimum claim; scarcity repair not implemented yet", failure();
    }
    PhysicalAllocationPlan result;
    result.planId = *plan;
    for (const auto& entry : entries) {
        PhysicalRecordAllocation item{entry.record, static_cast<uint32_t>(entry.source),
            static_cast<uint32_t>(entry.target), 1, static_cast<uint64_t>(entry.offset), {}};
        auto ids = eligibleIds.slice(static_cast<std::size_t>(entry.begin), static_cast<std::size_t>(entry.count));
        item.ids.append(ids.begin(), ids.end());
        result.records.push_back(std::move(item));
    }
    return result;
}
} // namespace mlir::pto::frontiersynch
