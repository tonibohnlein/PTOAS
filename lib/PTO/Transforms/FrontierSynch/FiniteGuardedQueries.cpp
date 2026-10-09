// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "FiniteGuardedInternal.h"
#include "PTO/Transforms/FrontierSynch/ArithmeticStorageSelectors.h"
#include "../InsertSync/SyncRegionArithmetic.h"
#include "llvm/ADT/MapVector.h"
namespace mlir::pto::frontiersynch {
bool detail::preflightFiniteStoragePredicate(func::FuncOp function, const IntegerSystem& domain,
    ArrayRef<Value> parameters, bool physicalByte, uint64_t period, ArrayRef<uint64_t> residues,
    ArrayRef<AffineExpr> coordinates, std::string& error)
{
    Block inputs;
    RegionExpressions scratch;
    std::vector<RegionExpressions::Id> values;
    if (physicalByte) {
        auto byte = inputs.addArgument(IndexType::get(function.getContext()), function.getLoc());
        values.push_back(scratch.input(byte));
    }
    for (auto parameter : parameters) { values.push_back(scratch.input(parameter)); }
    auto result = coordinates.empty() ? scratch.integerPredicate(domain, values, period, residues) :
        scratch.integerMappedPredicate(domain, values, coordinates, period, residues, error)
            .value_or(RegionExpressions::invalid);
    const bool accepted = result != RegionExpressions::invalid && scratch.constructionError().empty();
    if (!accepted && error.empty()) { error = scratch.constructionError(); }
    return accepted;
}
void FiniteGuardedState::closeAndReduce()
{
    const auto count = effects.size();
    std::map<std::pair<uint32_t, uint32_t>, Expr> demands;
    auto add = [&](uint32_t source, uint32_t target, Expr guard) {
        auto [it, inserted] = demands.emplace(std::make_pair(source, target), guard);
        if (!inserted) {
            it->second = either(it->second, guard);
        }
    };
    std::map<uint32_t, std::vector<std::pair<uint32_t, CellAccess>>> accesses;
    for (uint32_t i = 0; i < count; ++i) {
        for (auto mode : effects[i].accesses) {
            accesses[mode.atom].push_back({i, mode});
        }
    }
    for (const auto& [cell, uses] : accesses) {
        // Protected writer pairs and simultaneous macro phases change the
        // lifetime-chain proof. Retain their exact raw pair generators; rank
        // reduction still applies, with the same O(W) candidate bound.
        bool plain = true;
        for (auto [i, x] : uses) {
            for (auto [j, y] : uses) {
                if (i == j) {
                    continue;
                }
                plain &= anchors[i].phase->elementOp != anchors[j].phase->elementOp &&
                         !ptoStorageProtection().protectsScalar(pipe(i), pipe(j)) &&
                         !hardwareProtectsConflict(pipe(i), x.protectionGroup, pipe(j), y.protectionGroup);
            }
        }
        for (std::size_t a = 0; a < uses.size(); ++a) {
            auto [i, x] = uses[a];
            auto noWriter = yes();
            for (std::size_t b = a + 1; b < uses.size(); ++b) {
                auto [j, y] = uses[b];
                ++cost.crossingCandidates;
                if (anchors[i].phase->elementOp != anchors[j].phase->elementOp && (x.write || y.write) &&
                    !ptoStorageProtection().protectsScalar(pipe(i), pipe(j)) &&
                    !hardwareProtectsConflict(pipe(i), x.protectionGroup, pipe(j), y.protectionGroup)) {
                    add(i, j, both(both(presence[i], presence[j]), noWriter));
                }
                if (plain && y.write) {
                    noWriter = both(noWriter, negate(presence[j]));
                }
            }
        }
    }
    for (auto edge : residual) {
        add(edge.source, edge.target, both(presence[edge.source], presence[edge.target]));
    }
    for (auto edge : guardedResidual) {
        add(edge.source, edge.target, both(edge.guard, both(presence[edge.source], presence[edge.target])));
    }
    std::vector<GuardedRankPayload> payloads;
    std::vector<GuardedRankEdge> generators, native;
    for (uint32_t i = 0; i < count; ++i) {
        payloads.push_back({pipe(i), presence[i]});
    }
    for (auto [pair, guard] : demands) {
        generators.push_back({pair.first, pair.second, guard});
    }
    for (auto edge : nativePrerequisites) {
        native.push_back({edge.source, edge.target, yes()});
    }
    for (auto edge : guardedNative) { native.push_back({edge.source, edge.target, edge.guard}); }
    rankIndex = reduceGuardedRanks(*arena, payloads, generators, native);
    retained.clear();
    for (auto edge : rankIndex.retained) {
        retained.push_back({edge.source, edge.target, edge.guard});
    }
}

void FiniteGuardedState::summarize(const SyncInput& input)
{
    auto selector = [&](uint32_t type, Expr present) {
        return RegionalSelector{{type, arena->constant(0), PeriodicEventKind::Start}, present};
    };
    std::map<uint32_t, std::vector<std::pair<uint32_t, CellAccess>>> cells;
    std::map<uint32_t, Expr> preceding, following;
    for (uint32_t i = 0; i < effects.size(); ++i) {
        for (auto mode : effects[i].accesses) {
            cells[mode.atom].push_back({i, mode});
        }
        auto [seen, added] = preceding.emplace(pipe(i), no());
        firstPayloads[pipe(i)].push_back(selector(i, both(presence[i], negate(seen->second))));
        seen->second = either(seen->second, presence[i]);
        ++cost.selectorComparisons;
    }
    for (std::size_t j = effects.size(); j; --j) {
        const auto i = static_cast<uint32_t>(j - 1);
        auto [seen, added] = following.emplace(pipe(i), no());
        lastPayloads[pipe(i)].push_back(selector(i, both(presence[i], negate(seen->second))));
        seen->second = either(seen->second, presence[i]);
        ++cost.selectorComparisons;
    }
    for (const auto& [cell, uses] : cells) {
        RegionalStorageBoundary boundary;
        boundary.cell = input.accesses().cells()[cell];
        for (auto [i, mode] : uses) {
            auto first = presence[i], last = presence[i];
            if (mode.write) {
                for (auto [j, other] : uses) {
                    if (!other.write || anchors[i].phase->elementOp == anchors[j].phase->elementOp) {
                        continue;
                    }
                    ++cost.selectorComparisons;
                    if (j < i) {
                        first = both(first, negate(presence[j]));
                    }
                    if (j > i) {
                        last = both(last, negate(presence[j]));
                    }
                }
                boundary.firstWriters.push_back(selector(i, first));
                boundary.lastWriters.push_back(selector(i, last));
            } else if (mode.read) {
                for (auto [j, other] : uses) {
                    ++cost.selectorComparisons;
                    bool supersedes = anchors[i].phase->elementOp != anchors[j].phase->elementOp &&
                                      (other.write || (other.read && pipe(i) == pipe(j)));
                    if (supersedes && j < i) {
                        first = both(first, negate(presence[j]));
                    }
                    if (supersedes && j > i) {
                        last = both(last, negate(presence[j]));
                    }
                }
                boundary.firstReaders[pipe(i)].push_back(selector(i, first));
                boundary.lastReaders[pipe(i)].push_back(selector(i, last));
            }
        }
        storageBoundary.push_back(std::move(boundary));
    }
}
namespace {
// A storage-only owner: selected boundaries borrow neither temporary producer
// relations nor the session. Every expression remains in the query arena.
struct ExpandedStorage {
    using Key = std::pair<AddressSpace, Value>;
    std::shared_ptr<const ArithmeticProgram> program;
    std::shared_ptr<FiniteGuardedState> state;
    std::shared_ptr<const SyncInput> input;
    ArithmeticStorageSelectors selectors;
    std::vector<RegionExpressions::Id> parameters;
    llvm::MapVector<Key, SmallVector<AffineExpr>> coordinateMaps;
    RegionExpressions::Id predicate(RegionExpressions& arena, const IntegerSystem& domain,
        ArrayRef<RegionExpressions::Id> values, ArrayRef<uint64_t> residues,
        std::optional<Key> family, std::string& error) const
    {
        if (!family) { return arena.integerPredicate(domain, values, selectors.period, residues); }
        auto found = coordinateMaps.find(*family);
        if (found == coordinateMaps.end()) {
            return arena.integerPredicate(domain, values, selectors.period, residues);
        }
        return arena.integerMappedPredicate(domain, values, found->second, selectors.period, residues, error)
            .value_or(RegionExpressions::invalid);
    }
    bool preflight(std::string& error, uint64_t* checkedPredicates)
    {
        auto check = [&](const IntegerSystem& domain, ArrayRef<uint64_t> residues, std::optional<Key> key) {
            if (checkedPredicates && *checkedPredicates != UINT64_MAX) { ++*checkedPredicates; }
            ArrayRef<AffineExpr> coordinates;
            if (key) {
                auto found = coordinateMaps.find(*key);
                if (found != coordinateMaps.end()) { coordinates = found->second; }
            }
            return detail::preflightFiniteStoragePredicate(program->context.function, domain,
                program->parameters, key.has_value(), selectors.period, residues, coordinates, error);
        };
        for (const auto& support : selectors.support) {
            std::vector<uint64_t> residues{support.byteResidue};
            llvm::append_range(residues, support.parameterResidues);
            if (!check(support.domain, residues, Key{support.space, support.base})) { return false; }
        }
        for (const auto& boundary : selectors.boundaries) {
            std::optional<Key> key;
            if (boundary.storageSpace) { key = Key{*boundary.storageSpace, boundary.storageBase}; }
            for (const auto& piece : boundary.selector.pieces) {
                auto residues = piece.inputResidues;
                llvm::append_range(residues, piece.parameterResidues);
                if (!check(piece.domain, residues, key)) { return false; }
            }
        }
        return true;
    }
    std::optional<std::vector<RegionalSelector>> select(const ArithmeticBoundarySelector& boundary,
        std::optional<RegionExpressions::Id> byte = {}) const
    {
        auto& arena = *state->arena;
        std::vector<RegionExpressions::Id> values;
        if (byte) { values.push_back(*byte); }
        llvm::append_range(values, parameters);
        std::map<std::size_t, RegionExpressions::Id> guards;
        for (const auto& piece : boundary.selector.pieces) {
            auto residues = piece.inputResidues;
            llvm::append_range(residues, piece.parameterResidues);
            std::optional<Key> key;
            if (byte) { key = Key{*boundary.storageSpace, boundary.storageBase}; }
            std::string error;
            auto present = predicate(arena, piece.domain, values, residues, key, error);
            if (present == RegionExpressions::invalid) { return std::nullopt; }
            auto [entry, inserted] = guards.emplace(piece.outputSite, present);
            if (!inserted) { entry->second = arena.lor(entry->second, present); }
        }
        std::vector<RegionalSelector> result;
        for (auto [site, present] : guards) {
            result.push_back({{static_cast<uint32_t>(site), arena.constant(0), PeriodicEventKind::Start}, present});
        }
        return result;
    }
    RegionExpressions::Id membership(RegionalByteAddress address) const
    {
        auto& arena = *state->arena;
        auto result = arena.boolean(false);
        std::vector<RegionExpressions::Id> values{address.offset};
        llvm::append_range(values, parameters);
        for (const auto& piece : selectors.support) {
            if (piece.space != address.space || piece.base != address.base) { continue; }
            std::vector<uint64_t> residues{piece.byteResidue};
            llvm::append_range(residues, piece.parameterResidues);
            std::string error;
            auto member = predicate(arena, piece.domain, values, residues, Key{piece.space, piece.base}, error);
            if (member == RegionExpressions::invalid) { return member; }
            result = arena.lor(result, member);
        }
        return result;
    }
};
} // namespace
FailureOr<RegionalAnalysis> expandedFiniteRegionalSelectors(const FiniteGuardedAnalysis& analysis,
    const RegionalAnalysis& queries, std::string& error, std::shared_ptr<const SyncInput> inputOwner,
    uint64_t* checkedPredicates)
{
    const auto program = analysis.expandedProgram;
    const auto state = analysis.state;
    const bool compatible = program && state && analysis.error.empty() && queries.capabilities.exactQueries &&
        queries.expressions == state->arena && queries.accessModel == state->accessModel &&
        queries.gmAliasPolicy == state->gmAliasPolicy && queries.anchors.size() == program->sites.size() &&
        (!inputOwner || &inputOwner->accesses() == state->accessModel);
    if (!compatible) { error = "finite storage exports require the original query/input context"; return failure(); }
    if (!program->extraction.dischargedEffects.empty()) {
        error = "finite storage exports for discharged effects not implemented yet"; return failure();
    }
    for (auto [id, site] : llvm::enumerate(program->sites)) {
        const auto& anchor = queries.anchors[id];
        const bool same = anchor.phase == site.phase && anchor.coordinates.size() == site.fixedCoordinates.size();
        if (!same) { error = "finite storage exports require original occurrence identities"; return failure(); }
        for (auto [coordinate, fixed] : llvm::zip(anchor.coordinates, site.fixedCoordinates)) {
            if (coordinate.loop != fixed.loop || coordinate.induction != fixed.induction) {
                error = "finite storage exports require original fixed coordinates"; return failure();
            }
        }
        for (auto effectID : state->accessModel->effectsFor(site.phase)) {
            const auto& effect = state->accessModel->effects()[effectID];
            const bool described = effect.rangesMaterialized || !effect.regions.empty();
            if (!described) {
                error = "finite storage export is missing a modeled access description"; return failure();
            }
        }
    }
    auto storage = std::make_shared<ExpandedStorage>();
    storage->program = state->expandedStorageProgram ? state->expandedStorageProgram : program;
    storage->state = state; storage->input = std::move(inputOwner);
    std::vector<uint32_t> pipes;
    for (const auto& site : program->sites) { pipes.push_back(static_cast<uint32_t>(site.phase->kPipeValue)); }
    storage->selectors = buildArithmeticStorageSelectors(*storage->program, pipes);
    if (!storage->selectors.error.empty()) { error = storage->selectors.error; return failure(); }
    for (const auto& boundary : storage->selectors.boundaries) {
        for (const auto& piece : boundary.selector.pieces) {
            const bool mapped = piece.outputSite < program->sites.size() && piece.outputs.empty();
            if (!mapped) { error = "finite storage selector has unmapped occurrence coordinates"; return failure(); }
        }
    }
    for (auto parameter : program->parameters) { storage->parameters.push_back(state->arena->input(parameter)); }
    SmallVector<AffineExpr> identity, symbols;
    auto function = program->context.function;
    auto* context = function.getContext();
    identity.push_back(getAffineDimExpr(0, context));
    for (unsigned i = 0; i < program->parameters.size(); ++i) {
        identity.push_back(getAffineDimExpr(i + 1, context));
        symbols.push_back(identity.back());
    }
    for (const auto& support : storage->selectors.support) {
        ExpandedStorage::Key key{support.space, support.base};
        auto translation = state->expandedStorageTranslations.find(key);
        if (translation == state->expandedStorageTranslations.end()) { continue; }
        auto [entry, inserted] = storage->coordinateMaps.try_emplace(key, identity);
        if (!inserted) { continue; }
        auto shifted = mlir::pto::detail::substitute(translation->second, {}, symbols);
        auto negative = mlir::pto::detail::checkedMul(shifted, getAffineConstantExpr(-1, context));
        entry->second[0] = mlir::pto::detail::checkedAdd(identity[0], negative);
        if (!entry->second[0]) {
            error = "finite physical coordinate adapter is not representable safely"; return failure();
        }
    }
    if (!storage->preflight(error, checkedPredicates)) { return failure(); }
    auto out = queries;
    auto certificate = std::make_shared<RegionalSymbolicStorageCertificate>();
    certificate->expressions = state->arena; certificate->accessModel = state->accessModel;
    certificate->gmAliasPolicy = state->gmAliasPolicy;
    using Key = std::pair<AddressSpace, Value>;
    llvm::MapVector<Key, std::vector<std::size_t>> families;
    for (uint32_t site = 0; site < program->sites.size(); ++site) {
        out.firstSitePayloads[site] = {};
        RegionalSelector occurrence{{site, state->arena->constant(0), PeriodicEventKind::Start}, state->presence[site]};
        for (auto id : state->accessModel->effectsFor(program->sites[site].phase)) {
            const auto& effect = state->accessModel->effects()[id];
            out.accessBoundary.push_back({id, occurrence, occurrence, false});
            if (!llvm::is_contained(out.symbolicStorageEffects, id)) { out.symbolicStorageEffects.push_back(id); }
            auto add = [&](Key key) {
                auto& effects = families[key];
                if (!llvm::is_contained(effects, id)) { effects.push_back(id); }
            };
            for (const auto& range : effect.ranges) { add({range.space, range.base}); }
            for (const auto& region : effect.regions) { add({effect.memory->scope, region.base}); }
        }
    }
    for (const auto& entry : families) {
        const auto key = entry.first;
        const auto& effects = entry.second;
        RegionalStorageFamily family;
        family.space = key.first; family.base = key.second; family.effects = effects;
        family.membership = [storage, key](RegionalByteAddress address) -> std::optional<RegionExpressions::Id> {
            auto& arena = *storage->state->arena;
            const bool valid = address.offset < arena.size() && !arena.isBoolean(address.offset);
            if (!valid) { return std::nullopt; }
            if (address.space != key.first) { return arena.boolean(false); }
            if (address.base != key.second) {
                SmallVector<SyncStorageCell> domains{{key.first, 0, 1, key.second},
                                                    {address.space, 0, 1, address.base}};
                if (!storageBasesAreComparable(domains, storage->state->gmAliasPolicy)) { return std::nullopt; }
                return arena.boolean(false);
            }
            auto present = storage->membership(address);
            return present == RegionExpressions::invalid ? std::nullopt :
                std::optional<RegionExpressions::Id>(present);
        };
        certificate->families.push_back(std::move(family));
    }
    out.symbolicStorage = certificate;
    out.storageSelectors = [storage](RegionalByteAddress address) -> std::optional<RegionalStorageSelectors> {
        auto& arena = *storage->state->arena;
        const bool valid = address.offset < arena.size() && !arena.isBoolean(address.offset);
        if (!valid) { return std::nullopt; }
        RegionalStorageSelectors result;
        for (const auto& boundary : storage->selectors.boundaries) {
            if (boundary.storageSpace != address.space) { continue; }
            if (boundary.storageBase != address.base) {
                SmallVector<SyncStorageCell> domains{{address.space, 0, 1, boundary.storageBase},
                                                    {address.space, 0, 1, address.base}};
                if (!storageBasesAreComparable(domains, storage->state->gmAliasPolicy)) { return std::nullopt; }
                continue;
            }
            auto selected = storage->select(boundary, address.offset);
            if (!selected) { return std::nullopt; }
            switch (boundary.kind) {
            case ArithmeticBoundaryKind::FirstWriter: llvm::append_range(result.firstWriters, *selected); break;
            case ArithmeticBoundaryKind::LastWriter: llvm::append_range(result.lastWriters, *selected); break;
            case ArithmeticBoundaryKind::FirstReaderBeforeWrite:
                llvm::append_range(result.firstReaders[*boundary.pipe], *selected); break;
            case ArithmeticBoundaryKind::LastReaderAfterWrite:
                llvm::append_range(result.lastReaders[*boundary.pipe], *selected); break;
            default: break;
            }
        }
        if (!arena.constructionError().empty()) { return std::nullopt; }
        return result;
    };
    for (const auto& boundary : storage->selectors.boundaries) {
        if (boundary.storageSpace) { continue; }
        auto selected = storage->select(boundary);
        if (!selected) { error = "finite native storage selector preparation failed"; return failure(); }
        switch (boundary.kind) {
        case ArithmeticBoundaryKind::FirstPayload:
            llvm::append_range(out.firstPayloads[*boundary.pipe], *selected); break;
        case ArithmeticBoundaryKind::LastPayload:
            llvm::append_range(out.lastPayloads[*boundary.pipe], *selected); break;
        case ArithmeticBoundaryKind::FirstSite:
            llvm::append_range(out.firstSitePayloads[*boundary.site], *selected); break;
        default: break;
        }
    }
    if (!state->arena->constructionError().empty()) { error = state->arena->constructionError(); return failure(); }
    out.capabilities.completeStorageModel = true;
    out.capabilities.exactSelectors = true;
    out.cost.expressionNodes = state->arena->size();
    return out;
}
} // namespace mlir::pto::frontiersynch
