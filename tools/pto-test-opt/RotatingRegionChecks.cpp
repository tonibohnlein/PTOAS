// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/FrontierAnalysis.h"
#include "PTO/Transforms/FrontierSynch/RotatingRegion.h"
#include "../../lib/PTO/Transforms/FrontierSynch/RepeatedRegionInternal.h"
#include "llvm/Support/raw_ostream.h"
#include <set>
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
using Id = fs::RegionExpressions::Id;
bool selectors(const fs::RegionalAnalysis& region, func::FuncOp function)
{
    auto& e = *region.expressions;
    for (uint64_t trips : {0, 1, 2, 7, 16}) {
        fs::RegionExpressions::Substitution values({
            {e.input(function.getArgument(0)), e.constant(trips)},
            {e.input(function.getArgument(1)), e.constant(3)}});
        for (uint64_t bank = 0; bank < 3; ++bank) {
            auto selected = region.storageSelectors({pto::AddressSpace::LEFT, {}, e.constant(bank * 512)});
            if (!selected) { return false; }
            auto check = [&](const auto& candidates, bool last) {
                unsigned count = 0;
                for (const auto& candidate : candidates) {
                    auto present = e.constantValue(e.substitute(candidate.present, values));
                    if (!present) { return false; }
                    if (!*present) { continue; }
                    if (candidate.event.visits.empty()) { return false; }
                    auto visit = e.constantValue(e.substitute(candidate.event.visits.front(), values));
                    const auto expected = last ? bank + 3 * ((trips - 1 - bank) / 3) : bank;
                    if (!visit || *visit != expected || candidate.event.type != 0) { return false; }
                    ++count;
                }
                return count == unsigned(bank < trips);
            };
            if (!check(selected->firstWriters, false) || !check(selected->lastWriters, true)) { return false; }
        }
    }
    return true;
}
fs::RotatingRegionInput normalized(const fs::RegionalAnalysis& child, uint64_t left, uint64_t right)
{
    fs::RotatingRegionInput input;
    input.child = child;
    for (const auto& cell : child.storageBoundary) {
        auto found = llvm::find_if(input.families, [&](const auto& f) {
            return f.firstBank.space == cell.cell.space && f.firstBank.base == cell.cell.base;
        });
        if (found == input.families.end()) {
            input.families.push_back({cell.cell});
            found = std::prev(input.families.end());
            found->banks = cell.cell.space == pto::AddressSpace::LEFT ? left :
                           cell.cell.space == pto::AddressSpace::RIGHT ? right : 1;
            found->stride = found->banks > 1 ? 1 : 0;
        } else {
            found->firstBank.begin = std::min(found->firstBank.begin, cell.cell.begin);
            found->firstBank.end = std::max(found->firstBank.end, cell.cell.end);
        }
    }
    for (auto& family : input.families) {
        family.bankStride = family.firstBank.end - family.firstBank.begin;
        for (const auto& access : child.accessBoundary) {
            const auto& effect = child.accessModel->effects()[access.effect];
            if (effect.memory->scope == family.firstBank.space) { family.effects.push_back(access.effect); }
        }
    }
    return input;
}
} // namespace
int runRotatingRegionChecks(func::FuncOp function)
{
    fs::FrontierAnalysis analysis(function);
    if (failed(analysis.initialize()) || !analysis.result()) { return 1; }
    fs::PhaseIndex index;
    if (failed(index.build(function, *analysis.input()))) { return 1; }
    const auto& program = *analysis.result();
    auto outer = llvm::find_if(program.nodes, [&](const auto& node) {
        return node.kind == fs::StructureKind::Loop && node.anchor->getParentOp() == function;
    });
    if (outer == program.nodes.end()) { return 1; }
    auto arena = std::make_shared<fs::RegionExpressions>();
    auto result = fs::recognizeRotatingRegion(function, *analysis.input(), program,
        outer - program.nodes.begin(), index, arena);
    if (!result.error.empty() || !result.state) {
        llvm::errs() << "rotating original-input recognition: " << result.error << "\n"; return 1;
    }
    std::set<uint64_t> distances;
    for (const auto& crossing : result.state->queryCrossings) { distances.insert(crossing.displacement); }
    if (!distances.count(3) || !distances.count(5) || result.state->phaseCount != 1 ||
        result.state->body.anchors.size() != (function->hasAttr("test.rotating_residual") ? 6U : 4U) ||
        !selectors(result.regional, function)) {
        llvm::errs() << "rotating independent-family selectors or distances lost\n"; return 1;
    }
    auto loop = cast<scf::ForOp>(outer->anchor);
    if (function->hasAttr("test.rotating_residual")) {
        bool owned = false, readOnly = false;
        if (!result.regional.symbolicStorage) { return 1; }
        for (const auto& family : result.regional.symbolicStorage->families) {
            if (family.space != pto::AddressSpace::GM) { continue; }
            owned |= family.kind == fs::RegionalStorageFamilyKind::VisitOwned && bool(family.owner);
            readOnly |= family.kind == fs::RegionalStorageFamilyKind::SharedReadOnly;
        }
        if (!owned || !readOnly) {
            llvm::errs() << "rotating original-input residual ownership/read-only exports missing\n"; return 1;
        }
        llvm::outs() << "rotating original-input residual storage: owned and read-only exports retained\n";
        return 0;
    }
    auto large = fs::repeatRotatingRegion(function, loop, normalized(result.state->body, 17, 19), result.state->trips);
    if (!large.error.empty() || !large.state || large.state->body.anchors.size() != 4 ||
        large.state->queryCrossings.size() != result.state->queryCrossings.size()) {
        llvm::errs() << "rotating family counts expanded canonical child or crossings: " << large.error << "\n";
        return 1;
    }
    const uint64_t hugeBanks = (uint64_t(1) << 40) + 1;
    auto hugeInput = normalized(result.state->body, hugeBanks, 19);
    for (auto& family : hugeInput.families) {
        if (family.firstBank.space == pto::AddressSpace::LEFT) { family.stride = 3; }
    }
    auto huge = fs::repeatRotatingRegion(function, loop, std::move(hugeInput), arena->constant(hugeBanks));
    const uint64_t visit = hugeBanks / 2, bank = (3 * visit) % hugeBanks;
    auto selectedHuge = huge.error.empty() ? huge.regional.storageSelectors(
        {pto::AddressSpace::LEFT, {}, arena->constant(bank * 512)}) : std::nullopt;
    if (!selectedHuge) { llvm::errs() << "large rotating selector: " << huge.error << "\n"; return 1; }
    fs::RegionExpressions::Substitution hugeValues({
        {arena->input(function.getArgument(0)), arena->constant(hugeBanks)},
        {arena->input(function.getArgument(1)), arena->constant(3)}});
    unsigned selectedCount = 0;
    for (const auto& selected : selectedHuge->firstWriters) {
        auto present = arena->constantValue(arena->substitute(selected.present, hugeValues));
        if (!present) { return 1; }
        if (*present) {
            if (selected.event.visits.empty() || arena->constantValue(selected.event.visits.front()) != visit) {
                llvm::errs() << "large rotating modular product overflowed\n"; return 1;
            }
            ++selectedCount;
        }
    }
    if (selectedCount != 1) { return 1; }
    auto residual = normalized(result.state->body, 3, 5);
    auto readOnly = llvm::find_if(residual.families, [](const auto& family) {
        return family.firstBank.space == pto::AddressSpace::MAT;
    });
    if (readOnly == residual.families.end() || readOnly->effects.size() < 2) { return 1; }
    residual.residualEffects.push_back(readOnly->effects.back());
    readOnly->effects.pop_back();
    auto withResidual = fs::repeatRotatingRegion(function, loop, std::move(residual), result.state->trips);
    if (!withResidual.error.empty() || !withResidual.regional.symbolicStorage) {
        llvm::errs() << "rotating shared residual read-only proof: " << withResidual.error << "\n"; return 1;
    }
    auto& e = *arena;
    for (uint64_t trips : {0, 1, 4}) {
        fs::RegionExpressions::Substitution values({
            {e.input(function.getArgument(0)), e.constant(trips)},
            {e.input(function.getArgument(1)), e.constant(2)}});
        auto selected = withResidual.regional.storageSelectors({pto::AddressSpace::MAT, {}, e.constant(0)});
        if (!selected) { return 1; }
        for (bool first : {false, true}) {
            unsigned count = 0;
            const auto& readers = first ? selected->firstReaders : selected->lastReaders;
            for (const auto& [pipe, candidates] : readers) {
                for (const auto& candidate : candidates) {
                    auto present = e.constantValue(e.substitute(candidate.present, values));
                    if (!present) { return 1; }
                    if (!*present) { continue; }
                    auto visit = e.constantValue(e.substitute(candidate.event.visits.front(), values));
                    if (!visit || *visit != (first ? 0 : trips - 1) || candidate.event.type != (first ? 0 : 1)) {
                        llvm::errs() << "rotating/residual read-only extremum differs\n"; return 1;
                    }
                    ++count;
                }
            }
            if (count != unsigned(trips != 0)) { return 1; }
        }
    }
    auto booleanTrips = fs::repeatRotatingRegion(function, loop, normalized(result.state->body, 3, 5),
        arena->boolean(true));
    auto wrongFunction = fs::repeatRotatingRegion({}, loop, normalized(result.state->body, 3, 5), result.state->trips);
    if (booleanTrips.error.empty() || wrongFunction.error.empty()) {
        llvm::errs() << "rotating malformed context or trip domain accepted\n"; return 1;
    }
    auto missing = normalized(result.state->body, 3, 5);
    auto cell = llvm::find_if(missing.child.storageBoundary, [](const auto& item) {
        return item.cell.space == pto::AddressSpace::LEFT;
    });
    if (cell == missing.child.storageBoundary.end() || cell->lastReaders.empty()) { return 1; }
    for (auto* side : {&cell->firstWriters, &cell->lastWriters}) {
        for (auto& writer : *side) { writer.present = arena->boolean(false); }
    }
    auto rejected = fs::repeatRotatingRegion(function, loop, std::move(missing), result.state->trips);
    if (rejected.error.find("refresh every accessed local cell") == std::string::npos) {
        llvm::errs() << "rotating missing local refresh was accepted: " << rejected.error << "\n"; return 1;
    }
    // A missing shared phase is not permission to discard an effectful leaf.
    // Insert an opaque call with no supplied effect interpretation, rebuild
    // structure, and require the generic leaf contract to reject it even when
    // the finite run provider has already failed and arithmetic is attempted.
    OpBuilder builder(result.state->body.anchors.front().phase->elementOp);
    auto opaque = builder.create<func::CallOp>(function.getLoc(), "opaque_external_effect", TypeRange{}, ValueRange{});
    auto altered = fs::recognizeProgram(function, *analysis.input());
    std::string rejection;
    if (succeeded(altered)) {
        auto candidate = llvm::find_if(altered->nodes, [&](const auto& node) {
            return node.kind == fs::StructureKind::Loop && node.anchor == loop;
        });
        if (candidate != altered->nodes.end()) {
            auto attempt = fs::recognizeRotatingRegion(function, *analysis.input(), *altered,
                candidate - altered->nodes.begin(), index, std::make_shared<fs::RegionExpressions>());
            rejection = attempt.error;
        }
    }
    opaque.erase();
    if (rejection.find("unmodeled-operation") == std::string::npos) {
        llvm::errs() << "rotating arithmetic fallback discarded an uninterpreted leaf: " << rejection << "\n";
        return 1;
    }
    llvm::outs() << "rotating compact child: independent weighted families, exact selectors, no joint expansion\n";
    return 0;
}
