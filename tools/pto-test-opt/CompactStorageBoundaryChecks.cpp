// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent finite occurrence DAG containment for qualified mixed composition.
#include "PTO/Transforms/FrontierSynch/CompactStorageBoundary.h"
#include "PTO/Transforms/FrontierSynch/HardwareProtection.h"
#include "llvm/Support/raw_ostream.h"
#include <set>
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
using Kind = fs::PeriodicEventKind;
struct Occurrence { uint32_t type; uint64_t visit; const pto::CompoundInstanceElement* phase; };
using Graph = std::vector<std::vector<bool>>;
Graph original(llvm::ArrayRef<Occurrence> word, const pto::SyncInput& input)
{
    Graph result(2 * word.size(), std::vector<bool>(2 * word.size(), false));
    const auto& model = input.accesses();
    for (std::size_t a = 0; a < word.size(); ++a) {
        result[2*a][2*a] = result[2*a+1][2*a+1] = result[2*a][2*a+1] = true;
        const auto p = static_cast<uint32_t>(word[a].phase->kPipeValue);
        for (std::size_t b = a + 1; b < word.size(); ++b) {
            const auto q = static_cast<uint32_t>(word[b].phase->kPipeValue);
            if (p == q) { result[2*a][2*b] = result[2*a+1][2*b+1] = true; }
            if (fs::ptoStorageProtection().protectsScalar(p, q)) { continue; }
            for (auto x : model.effectsFor(word[a].phase)) {
                for (auto y : model.effectsFor(word[b].phase)) {
                    if (model.mayConflict(x, y)) { result[2*a+1][2*b] = true; }
                }
            }
        }
    }
    // Independent concrete transitive closure, no compact extrema or class IDs.
    for (std::size_t k = 0; k < result.size(); ++k) {
        for (std::size_t a = 0; a < result.size(); ++a) {
            for (std::size_t b = 0; b < result.size(); ++b) {
                result[a][b] = result[a][b] || (result[a][k] && result[k][b]);
            }
        }
    }
    return result;
}
std::optional<bool> evaluate(const fs::RegionalOrderView& view, fs::RegionalEvent a, fs::RegionalEvent b,
                            fs::RegionExpressions::Id count, uint64_t trips)
{
    auto& arena = *view.context()->expressions();
    const auto query = fs::regionalReachability(view.regional(), a, b);
    if (!query) { return std::nullopt; }
    const std::pair<fs::RegionExpressions::Id, fs::RegionExpressions::Id> binding{count, arena.constant(trips)};
    fs::RegionExpressions::Substitution substitution(binding);
    const auto value = arena.constantValue(arena.substitute(*query, substitution));
    return value ? std::optional<bool>(*value != 0) : std::nullopt;
}
bool compare(const fs::CompactClassComposition& flat, const fs::CompactClassComposition& nested,
    llvm::ArrayRef<const pto::CompoundInstanceElement*> before,
    llvm::ArrayRef<const pto::CompoundInstanceElement*> body,
    llvm::ArrayRef<const pto::CompoundInstanceElement*> after, const pto::SyncInput& input,
    fs::RegionExpressions::Id count)
{
    const auto& bounds = flat.boundary->bounds();
    auto& arena = *bounds.context->expressions();
    for (uint64_t trips : {0, 1, 2, 4}) {
        std::vector<Occurrence> word;
        for (uint32_t site = 0; site < before.size(); ++site) { word.push_back({site, 0, before[site]}); }
        for (uint64_t visit = 0; visit < trips; ++visit) {
            for (uint32_t site = 0; site < body.size(); ++site) {
                word.push_back({static_cast<uint32_t>(before.size()) + site, visit, body[site]});
            }
        }
        for (uint32_t site = 0; site < after.size(); ++site) {
            word.push_back({static_cast<uint32_t>(before.size() + body.size()) + site, 0, after[site]});
        }
        const auto expected = original(word, input);
        for (std::size_t a = 0; a < word.size(); ++a) {
            for (std::size_t b = 0; b < word.size(); ++b) {
                fs::RegionalEvent source{word[a].type, arena.constant(word[a].visit), Kind::Completion};
                fs::RegionalEvent target{word[b].type, arena.constant(word[b].visit), Kind::Start};
                const auto lower = evaluate(*bounds.lower, source, target, count, trips);
                const auto upper = evaluate(*bounds.upper, source, target, count, trips);
                const auto other = evaluate(*nested.boundary->bounds().upper, source, target, count, trips);
                if (!lower || !upper || !other || (*lower && !expected[2*a+1][2*b]) ||
                    (expected[2*a+1][2*b] && !*upper) || *upper != *other) { return false; }
            }
        }
    }
    return true;
}
fs::CompactClasses finite(func::FuncOp function, const pto::SyncInput& input,
    llvm::ArrayRef<const pto::CompoundInstanceElement*> phases,
    const std::shared_ptr<fs::RegionExpressions>& arena, bool select, std::string& error)
{
    std::vector<fs::RequirementGroupId> groups(input.accesses().cells().size(), 1);
    auto frame = fs::captureFiniteRequirements(function, input, phases, arena, 42, groups, 2, error);
    if (!frame) { return {}; }
    fs::FiniteSelection selected;
    if (select && !groups.empty()) {
        std::vector<fs::FiniteReplacementRecord> records;
        std::set<uint64_t> seen;
        for (const auto& member : frame->original()->originalMemberships()) {
            if (member.group == 1 && seen.insert(member.record.originalRecord).second) {
                records.push_back({{43, member.record.originalRecord},
                    frame->analysis().scan.generators[member.record.originalRecord], arena->boolean(true)});
            }
        }
        selected = fs::applyCertifiedFiniteReplacement(frame, {}, 1, arena->boolean(true), records, error);
        if (!selected || selected->snapshot() == frame->original()) { return {}; }
    }
    auto result = fs::captureFiniteClassBoundary(frame, selected, error);
    if (result && selected && result->bounds().provenance != selected->snapshot()) { return {}; }
    return result;
}
} // namespace
int runCompactStorageBoundaryChecks(func::FuncOp function, const pto::SyncInput& input)
{
    if (!function->hasAttr("test.compact_boundary")) { return 0; }
    fs::PhaseIndex index;
    if (failed(index.build(function, input))) { return 1; }
    scf::ForOp loop;
    function.walk([&](scf::ForOp found) { loop = found; });
    if (!loop) { return 1; }
    auto arena = std::make_shared<fs::RegionExpressions>();
    const auto count = arena->input(loop.getUpperBound()), zero = arena->constant(0);
    const auto trips = arena->select(arena->slt(zero, count), count, zero);
    std::string error;
    auto domain = fs::captureCompactFixedBodyContext(loop, input, index, arena, trips, error);
    auto compact = fs::captureCompactClassBoundary(loop, index, domain, {}, error);
    if (!compact || !compact->compact() || !compact->compact()->mathematical) {
        llvm::errs() << "class capture: " << error << "\n"; return 1;
    }
    if (function->hasAttr("test.boundary_unavailable")) {
        auto declined = fs::composeCompactClassBoundaries(function, input, {compact});
        if (declined.error.empty() || !declined.mathematical || !declined.mathematical->original ||
            declined.mathematical->original->children.front().mathematicalOwner != compact) { return 1; }
        llvm::outs() << "compact class boundary unavailable preserved: " << function.getSymName() << "\n";
        return 0;
    }
    SmallVector<const pto::CompoundInstanceElement*> before, after;
    for (const auto* phase : input.instructions()) {
        if (loop->isProperAncestor(phase->elementOp)) { continue; }
        if (phase->elementOp->isBeforeInBlock(loop)) { before.push_back(phase); }
        else { after.push_back(phase); }
    }
    auto body = index.explicitSequence(*loop.getBody());
    if (failed(body) || before.empty() || after.empty()) { return 1; }
    auto prefix = finite(function, input, before, arena, true, error);
    auto suffix = finite(function, input, after, arena, false, error);
    if (!prefix || !suffix) { llvm::errs() << "finite class capture: " << error << "\n"; return 1; }
    auto flat = fs::composeCompactClassBoundaries(function, input, {prefix, compact, suffix});
    auto left = fs::composeCompactClassBoundaries(function, input, {prefix, compact});
    if (!flat.boundary || !left.boundary) {
        llvm::errs() << "class composition: " << flat.error << " / " << left.error << "\n"; return 1;
    }
    auto nested = fs::composeCompactClassBoundaries(function, input, {left.boundary, suffix});
    if (!nested.boundary || flat.boundary->nativeExports().capabilities.completeStorageModel ||
        nested.boundary->nativeExports().capabilities.exactSelectors ||
        flat.boundary->accesses().size() != prefix->accesses().size() + compact->accesses().size() +
            suffix->accesses().size() || !compare(flat, nested, before, *body, after, input, count)) {
        llvm::errs() << "class containment/hierarchy: " << nested.error << "\n"; return 1;
    }
    if (function->hasAttr("test.boundary_no_crossings") && !flat.crossings.upper.empty()) { return 1; }
    auto reversed = fs::composeCompactClassBoundaries(function, input, {suffix, compact, prefix});
    if (reversed.error.empty() || !reversed.mathematical || !reversed.mathematical->original) { return 1; }
    auto omitted = fs::composeCompactClassBoundaries(function, input, {prefix, suffix});
    if (omitted.error.empty() || !omitted.mathematical) { return 1; }
    llvm::outs() << "compact class boundary checks passed: " << function.getSymName() << "\n";
    return 0;
}
