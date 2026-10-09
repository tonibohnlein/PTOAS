// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Typed explicit/periodic adapters to the shared regional composition contract.
#include "PTO/Transforms/FrontierSynch/AnalysisCost.h"
#include "SequenceAnalysisInternal.h"
#include "PTO/Transforms/FrontierSynch/RegionalAllocation.h"
#include "llvm/ADT/MapVector.h"
namespace mlir::pto::frontiersynch {
namespace {
// Promote only a supplied complete uniform-atom certificate. Querying one byte
// of a family never establishes that its selected occurrences are uniform.
bool promoteUniformStorage(RegionalAnalysis& region)
{
    const auto& certificate = region.symbolicStorage;
    if (!certificate) { return true; }
    if (certificate->expressions != region.expressions || certificate->accessModel != region.accessModel ||
        certificate->gmAliasPolicy != region.gmAliasPolicy) { return false; }
    for (auto effect : certificate->uniformEffects) {
        if (!region.accessModel || effect >= region.accessModel->effects().size()) { return false; }
    }
    const bool needed = llvm::any_of(region.symbolicStorageEffects, [&](std::size_t effect) {
        return llvm::is_contained(certificate->uniformEffects, effect);
    });
    if (!needed) { return true; }
    llvm::append_range(region.storageBoundary, certificate->uniformBoundaries);
    llvm::erase_if(region.symbolicStorageEffects, [&](std::size_t effect) {
        return llvm::is_contained(certificate->uniformEffects, effect);
    });
    for (auto& access : region.accessBoundary) {
        if (llvm::is_contained(certificate->uniformEffects, access.effect)) { access.representedByCells = true; }
    }
    return true;
}
// A symbolic child may pass through finite composition if every cross-child
// storage pair involving it is provably conflict-free in the shared model.
// Its storage interface is retained for later, possibly conflicting consumers.
bool independentSymbolicStorage(ArrayRef<Child> children)
{
    for (std::size_t i = 0; i < children.size(); ++i) {
        const auto& region = children[i].regional;
        for (auto effect : region.symbolicStorageEffects) {
            if (!region.storageSelectors || !region.accessModel ||
                effect >= region.accessModel->effects().size()) { return false; }
            for (std::size_t j = 0; j < children.size(); ++j) {
                if (i == j) { continue; }
                const auto& other = children[j].regional;
                if (other.accessModel != region.accessModel) {
                    if (!other.anchors.empty() || !other.storageBoundary.empty()) { return false; }
                    continue;
                }
                for (const auto& anchor : other.anchors) {
                    for (auto target : region.accessModel->effectsFor(anchor.phase)) {
                        if (region.accessModel->mayConflict(effect, target)) { return false; }
                    }
                }
            }
        }
    }
    return true;
}
// A finite counterpart bounds the support of this crossing, not the whole
// symbolic family. Keep complete per-byte selectors and all symbolic exports.
// This optional byte adapter is charged and bounded like other finite exports.
bool projectFiniteCrossings(std::vector<Child>& children,
                           std::set<SequenceAnalysisState::StoragePair>& covered, RegionalCost& costs)
{
    for (const auto& child : children) {
        if (llvm::any_of(child.regional.storageBoundary, [](const auto& boundary) {
            return boundary.cell.begin > boundary.cell.end;
        })) { return false; }
    }
    std::vector<SmallVector<SyncStorageCell>> support(children.size());
    std::set<SequenceAnalysisState::StoragePair> pairs;
    for (uint32_t i = 0; i < children.size(); ++i) {
        const auto& source = children[i].regional;
        for (auto effect : source.symbolicStorageEffects) {
            if (!source.storageSelectors || !source.accessModel || !source.expressions ||
                effect >= source.accessModel->effects().size()) { return false; }
            const auto& modeled = source.accessModel->effects()[effect];
            if (!modeled.memory || modeled.regions.empty()) { return false; }
            for (uint32_t j = 0; j < children.size(); ++j) {
                if (i == j) { continue; }
                const auto& target = children[j].regional;
                if (target.accessModel != source.accessModel || target.expressions != source.expressions ||
                    target.gmAliasPolicy != source.gmAliasPolicy) { return false; }
                for (const auto& anchor : target.anchors) {
                    for (auto other : source.accessModel->effectsFor(anchor.phase)) {
                        if (!source.accessModel->mayConflict(effect, other)) { continue; }
                        const auto matches = [&](const auto& value) { return value.effect == other; };
                        if (!llvm::any_of(target.accessBoundary, matches) ||
                            llvm::any_of(target.accessBoundary, [&](const auto& value) {
                                return matches(value) && !value.representedByCells;
                            }) || llvm::any_of(target.deferredAccessBoundary, matches) ||
                            llvm::is_contained(target.symbolicStorageEffects, other)) { return false; }
                        bool found = false;
                        for (const auto& boundary : target.storageBoundary) {
                            const auto& cell = boundary.cell;
                            if (cell.space != modeled.memory->scope) { continue; }
                            SmallVector<SyncStorageCell> identities{cell};
                            for (const auto& region : modeled.regions) {
                                identities.push_back({cell.space, 0, 1, region.base});
                            }
                            if (!storageBasesAreComparable(identities, source.gmAliasPolicy)) { return false; }
                            if (!llvm::any_of(modeled.regions,
                                [&](const auto& region) { return region.base == cell.base; })) { continue; }
                            if (cell.begin > cell.end || cell.end > uint64_t(INT64_MAX) ||
                                cell.end - cell.begin > maxRegionalSlotVisits) { return false; }
                            found = true;
                            for (auto byte = cell.begin; byte < cell.end; ++byte) {
                                if (llvm::any_of(support[i], [&](const auto& old) {
                                    return sameStorageDomain(old, cell) && old.begin == byte;
                                })) { continue; }
                                if (support[i].size() == maxRegionalSlotVisits) { return false; }
                                support[i].push_back({cell.space, byte, byte + 1, cell.base});
                            }
                        }
                        if (!found) { return false; }
                        pairs.insert(i < j ? SequenceAnalysisState::StoragePair{i, effect, j, other} :
                                             SequenceAnalysisState::StoragePair{j, other, i, effect});
                    }
                }
            }
        }
    }
    std::vector<std::vector<RegionalStorageBoundary>> projected(children.size());
    for (uint32_t i = 0; i < children.size(); ++i) {
        const auto& source = children[i].regional;
        if (support[i].empty()) { continue; }
        for (const auto& original : source.storageBoundary) {
            std::set<uint64_t> cuts{original.cell.begin, original.cell.end};
            for (const auto& byte : support[i]) {
                if (!sameStorageDomain(byte, original.cell) || byte.begin < original.cell.begin ||
                    byte.end > original.cell.end) { continue; }
                cuts.insert(byte.begin); cuts.insert(byte.end);
            }
            for (auto it = cuts.begin(); std::next(it) != cuts.end(); ++it) {
                if (llvm::any_of(support[i], [&](const auto& byte) {
                    return sameStorageDomain(byte, original.cell) && byte.begin == *it;
                })) { continue; }
                auto piece = original;
                piece.cell.begin = *it; piece.cell.end = *std::next(it);
                projected[i].push_back(std::move(piece));
            }
        }
        for (const auto& byte : support[i]) {
            auto selected = source.storageSelectors({byte.space, byte.base, source.expressions->constant(byte.begin)});
            if (!selected) { return false; }
            projected[i].push_back({byte, std::move(selected->firstWriters), std::move(selected->lastWriters),
                std::move(selected->firstReaders), std::move(selected->lastReaders)});
        }
    }
    uint64_t addedBytes = 0;
    for (const auto& bytes : support) {
        accumulateCost(addedBytes, bytes.size());
    }
    // Publish only after every pair and every point query succeeds. A failed
    // attempt leaves original regional summaries available to relational routes.
    for (uint32_t i = 0; i < children.size(); ++i) {
        if (support[i].empty()) { continue; }
        children[i].regional.storageBoundary = std::move(projected[i]);
        children[i].regional.cost.cells = children[i].regional.storageBoundary.size();
        accumulateCost(children[i].regional.cost.boundaryBytes, support[i].size());
    }
    accumulateCost(costs.boundaryBytes, addedBytes);
    covered = std::move(pairs);
    return true;
}
std::optional<RegionalStorageSelectors> finiteStorageSelectors(
    const RegionalAnalysis& region, RegionalByteAddress address)
{
    if (region.storageSelectors) { return region.storageSelectors(address); }
    auto& e = *region.expressions;
    if (address.offset >= e.size() || e.isBoolean(address.offset) || !region.symbolicStorageEffects.empty() ||
        llvm::any_of(region.accessBoundary, [](const auto& access) { return !access.representedByCells; })) {
        return std::nullopt;
    }
    RegionalStorageSelectors out;
    auto append = [&](auto& destination, const auto& source, Expr guard) {
        for (auto value : source) {
            value.present = e.land(guard, value.present);
            destination.push_back(std::move(value));
        }
    };
    for (const auto& cell : region.storageBoundary) {
        if (cell.cell.space != address.space) { continue; }
        if (cell.cell.base != address.base) {
            SmallVector<SyncStorageCell> identities{cell.cell, {address.space, 0, 1, address.base}};
            if (!storageBasesAreComparable(identities, region.gmAliasPolicy)) { return std::nullopt; }
            continue;
        }
        auto active = e.land(e.le(e.constant(cell.cell.begin), address.offset),
                             e.lt(address.offset, e.constant(cell.cell.end)));
        append(out.firstWriters, cell.firstWriters, active);
        append(out.lastWriters, cell.lastWriters, active);
        for (const auto& [pipe, values] : cell.firstReaders) { append(out.firstReaders[pipe], values, active); }
        for (const auto& [pipe, values] : cell.lastReaders) { append(out.lastReaders[pipe], values, active); }
    }
    return out;
}
std::optional<RegionalStorageSelectors> sequenceStorageSelectors(const SequenceAnalysisState& state,
    ArrayRef<uint32_t> starts, RegionalByteAddress address)
{
    auto& e = state.expressions;
    RegionalStorageSelectors out;
    auto present = [&](const auto& values) {
        auto result = e.boolean(false);
        for (const auto& value : values) { result = e.lor(result, value.present); }
        return result;
    };
    auto append = [&](auto& destination, const auto& source, Expr guard, uint32_t start) {
        for (auto value : source) {
            value.event.type += start;
            value.present = e.land(guard, value.present);
            destination.push_back(std::move(value));
        }
    };
    auto mask = [&](auto& values, Expr guard) {
        for (auto& value : values) { value.present = e.land(guard, value.present); }
    };
    auto seenWriter = e.boolean(false);
    for (uint32_t child = 0; child < state.children.size(); ++child) {
        auto local = finiteStorageSelectors(state.children[child].regional, address);
        if (!local) { return std::nullopt; }
        const auto written = present(local->firstWriters);
        append(out.firstWriters, local->firstWriters, e.lnot(seenWriter), starts[child]);
        for (const auto& [pipe, readers] : local->firstReaders) {
            auto guard = e.lnot(e.lor(seenWriter, present(out.firstReaders[pipe])));
            append(out.firstReaders[pipe], readers, guard, starts[child]);
        }
        mask(out.lastWriters, e.lnot(written));
        append(out.lastWriters, local->lastWriters, e.boolean(true), starts[child]);
        for (auto& [pipe, readers] : out.lastReaders) {
            auto replaced = written;
            auto found = local->lastReaders.find(pipe);
            if (found != local->lastReaders.end()) { replaced = e.lor(replaced, present(found->second)); }
            mask(readers, e.lnot(replaced));
        }
        for (const auto& [pipe, readers] : local->lastReaders) {
            append(out.lastReaders[pipe], readers, e.boolean(true), starts[child]);
        }
        seenWriter = e.lor(seenWriter, written);
    }
    return out;
}
} // namespace
void SequenceAnalysisState::bindAdapters()
{
    for (uint32_t id = 0; id < children.size(); ++id) {
        auto& child = children[id];
        auto& out = child.regional;
        if (out.presence) { continue; }
        for (const auto& anchor : child.anchors) { arena->forbidRecomputation(anchor.phase->elementOp); }
        out.expressions = arena;
        out.anchors = child.anchors;
        out.occurrenceLoops.assign(child.anchors.size(), child.loop);
        out.capabilities = {true, true, true, true};
        out.gmAliasPolicy = input->memory().gmPolicy();
        out.accessModel = &input->accesses();
        for (uint32_t type = 0; type < child.anchors.size(); ++type) {
            auto present = expressions.lt(c(0), child.trips);
            RegionalSelector first{{type, c(0), PeriodicEventKind::Start}, present};
            // Each explicit/template type executes at ordinal zero whenever
            // this body is nonempty. Keep this per-type selector when a parent
            // adds repeat coordinates; a pipe-wide first is not sufficient for
            // an incoming scalar prerequisite targeting a later payload site.
            out.firstSitePayloads[type].push_back(first);
            RegionalSelector last{{type, expressions.sub(child.trips, c(1)), PeriodicEventKind::Start}, present};
            for (auto effect : input->accesses().effectsFor(child.anchors[type].phase)) {
                if (llvm::is_contained(child.dischargedEffects, effect)) {
                    out.deferredAccessBoundary.push_back({effect, first, last, false});
                    continue;
                }
                const auto& access = input->accesses().effects()[effect];
                out.accessBoundary.push_back({effect, first, last,
                    child.loop ? !access.regions.empty() : access.rangesMaterialized});
            }
        }
        out.cost = child.costs;
        out.cost.children = 1;
        out.cost.physicalFragments = child.patterns.size();
        out.presence = [this, id](RegionalEvent event) -> std::optional<Expr> {
            const auto& current = children[id];
            if (event.type >= current.anchors.size() || event.ordinal >= expressions.size() ||
                expressions.isBoolean(event.ordinal)) { return std::nullopt; }
            return expressions.lt(event.ordinal, current.trips);
        };
        out.reachability = [this, id](RegionalEvent a, RegionalEvent b) -> std::optional<Expr> {
            const auto& current = children[id];
            auto pa = current.regional.presence(a), pb = current.regional.presence(b);
            if (!pa || !pb) { return std::nullopt; }
            Expr reaches = no();
            if (current.loop) {
                auto threshold = current.periodic.eventThreshold({a.type, a.kind}, {b.type, b.kind});
                if (threshold.error != PeriodicQueryError::None) { return std::nullopt; }
                if (threshold.displacement) {
                    reaches = both(expressions.le(a.ordinal, b.ordinal),
                        expressions.le(c(*threshold.displacement), expressions.sub(b.ordinal, a.ordinal)));
                }
            } else {
                auto answer = explicitEventPrecedes(current.explicitAnalysis, {a.type, a.kind}, {b.type, b.kind});
                if (!answer) { return std::nullopt; }
                reaches = expressions.boolean(*answer);
            }
            return both(reaches, both(*pa, *pb));
        };
        out.prepare = [this, id]() -> FailureOr<std::unique_ptr<PreparedLogicalPlan>> {
            const auto& current = children[id];
            if (!current.loop) {
                auto prepared = prepareExplicitInsertion(function, current.explicitAnalysis, false);
                if (succeeded(prepared)) { (*prepared)->completeInvocation = false; }
                return prepared;
            }
            auto prepared = std::make_unique<PreparedLogicalPlan>(0);
            if (failed(prepareCountedEndpointCode(function, current.endpoints, *prepared))) { return failure(); }
            prepared->allocationPreparation = [owner = shared_from_this(), id](PreparedLogicalPlan& plan) {
                const auto& child = owner->children[id];
                plan.regionalAllocation = periodicRegionalAllocation(child.regional, child.periodic, child.trips);
            };
            return prepared;
        };
        auto convert = [&](Selected selected) {
            const auto& port = ports[selected.port];
            return RegionalSelector{{port.type, port.ordinal, PeriodicEventKind::Start, port.visits}, selected.present};
        };
        for (std::size_t cell = 0; cell < cells.size(); ++cell) {
            const auto& local = boundaries[id][cell];
            RegionalStorageBoundary value;
            value.cell = cells[cell];
            for (auto selected : local.firstWriters) { value.firstWriters.push_back(convert(selected)); }
            for (auto selected : local.lastWriters) { value.lastWriters.push_back(convert(selected)); }
            for (const auto& [pipe, selectors] : local.firstReaders) {
                for (auto selected : selectors) { value.firstReaders[pipe].push_back(convert(selected)); }
            }
            for (const auto& [pipe, selectors] : local.lastReaders) {
                for (auto selected : selectors) { value.lastReaders[pipe].push_back(convert(selected)); }
            }
            out.storageBoundary.push_back(std::move(value));
        }
        std::map<uint32_t, std::pair<uint32_t, uint32_t>> positions;
        for (uint32_t type = 0; type < child.anchors.size(); ++type) {
            auto pipe = static_cast<uint32_t>(child.anchors[type].phase->kPipeValue);
            auto [it, inserted] = positions.emplace(pipe, std::make_pair(type, type));
            it->second.second = type;
        }
        auto nonempty = expressions.lt(c(0), child.trips);
        for (auto [pipe, pair] : positions) {
            out.firstPayloads[pipe].push_back({{pair.first, c(0), PeriodicEventKind::Start}, nonempty});
            out.lastPayloads[pipe].push_back({{pair.second, expressions.sub(child.trips, c(1)),
                                              PeriodicEventKind::Start}, nonempty});
        }
    }
}
bool SequenceAnalysisState::importSummaries(bool requireEndpoints)
{
    for (auto& child : children) {
        if (!promoteUniformStorage(child.regional)) {
            return fail("symbolic storage certificate belongs to a different arena or modeled input");
        }
    }
    if (children.size() != 1 && !independentSymbolicStorage(children) &&
        !projectFiniteCrossings(children, finiteCrossingPairs, costs)) {
        return fail("symbolic storage crossings require a constructed uniform-atom or relational adapter");
    }
    std::map<AddressSpace, llvm::MapVector<Value, std::set<uint64_t>>> points;
    SmallVector<SyncStorageCell> identities;
    std::optional<GMAliasPolicy> gmPolicy;
    for (const auto& child : children) {
        const auto& out = child.regional;
        if (out.expressions != arena || !out.capabilities.completeStorageModel || !out.capabilities.exactQueries ||
            !out.capabilities.exactSelectors || !out.presence ||
            (requireEndpoints && (!out.capabilities.endpointRecipes || (!out.prepare && !out.prepareWithVisits))) ||
            !out.reachability || out.anchors.size() != out.occurrenceLoops.size() ||
            (!out.outerLoops.empty() && out.outerLoops.size() != out.anchors.size()) ||
            (!out.outerDivisors.empty() && (out.outerLoops.empty() ||
                out.outerDivisors.size() != out.anchors.size()))) {
            return fail("sequence child lacks a constructed exact regional interface");
        }
        for (std::size_t type = 0; type < out.outerLoops.size(); ++type) {
            if (!out.outerDivisors.empty() && (out.outerDivisors.size() != out.anchors.size() ||
                out.outerDivisors[type].size() != out.outerLoops[type].size() ||
                llvm::any_of(out.outerDivisors[type], [](uint64_t value) { return value == 0; }))) {
                return fail("regional coordinate divisors must be positive and match their frames");
            }
            Operation* parent = nullptr;
            for (auto loop : out.outerLoops[type]) {
                if (!loop || (parent && !parent->isProperAncestor(loop))) {
                    return fail("regional outer coordinates must name nested original loops");
                }
                parent = loop;
            }
            auto* payload = out.anchors[type].phase ? out.anchors[type].phase->elementOp : nullptr;
            if (parent && (!payload || !parent->isProperAncestor(payload) ||
                (out.occurrenceLoops[type] && !parent->isProperAncestor(out.occurrenceLoops[type])))) {
                return fail("regional outer coordinates do not enclose the payload and leaf loop");
            }
        }
        for (const auto& cell : out.storageBoundary) {
            if (cell.cell.begin > cell.cell.end) { return fail("regional storage boundary has an invalid range"); }
            if (cell.cell.begin == cell.cell.end) { continue; }
            identities.push_back(cell.cell);
            if (cell.cell.space == AddressSpace::GM) {
                if (gmPolicy && *gmPolicy != out.gmAliasPolicy) {
                    return fail("regional GM alias assumptions disagree");
                }
                gmPolicy = out.gmAliasPolicy;
            }
            points[cell.cell.space][cell.cell.base].insert(cell.cell.begin);
            points[cell.cell.space][cell.cell.base].insert(cell.cell.end);
        }
    }
    const SyncStorageEffects* commonModel = nullptr;
    bool unmodeledStorage = false;
    for (const auto& child : children) {
        if (!child.regional.accessModel) {
            unmodeledStorage |= llvm::any_of(child.regional.storageBoundary, [](const auto& boundary) {
                return boundary.cell.begin != boundary.cell.end;
            });
            continue;
        }
        if (commonModel && commonModel != child.regional.accessModel) {
            return fail("regional access models belong to different inputs");
        }
        commonModel = child.regional.accessModel;
    }
    if (commonModel && unmodeledStorage) {
        return fail("mixed regional storage needs shared effect selectors on every nonempty child");
    }
    if (!commonModel && !storageBasesAreComparable(identities, gmPolicy.value_or(GMAliasPolicy::MayAlias))) {
        return fail("regional storage relationships need the shared access model");
    }
    cells.clear(); ports.clear(); portIds.clear(); boundaries.clear();
    portChoices.clear(); selectorAlternatives.clear();
    nativeFirst.clear(); nativeLast.clear();
    nativeFirst.resize(children.size()); nativeLast.resize(children.size());
    for (const auto& [space, bases] : points) {
        for (const auto& [base, endpoints] : bases) {
            for (auto it = endpoints.begin(); it != endpoints.end() && std::next(it) != endpoints.end(); ++it) {
                SyncStorageCell cell{space, *it, *std::next(it), base};
                bool covered = false;
                for (const auto& child : children) {
                    for (const auto& local : child.regional.storageBoundary) {
                        covered |= sameStorageDomain(local.cell, cell) && local.cell.begin <= cell.begin &&
                                   local.cell.end >= cell.end;
                    }
                }
                if (covered) { cells.push_back(cell); }
            }
        }
    }
    boundaries.resize(children.size(), std::vector<CellBoundary>(cells.size()));
    for (uint32_t id = 0; id < children.size(); ++id) {
        auto& child = children[id];
        child.anchors = child.regional.anchors;
        if (auto first = child.regional.firstOrdinal;
            first && (*first >= expressions.size() || expressions.isBoolean(*first))) {
            return fail("regional first ordinal must be an integer expression in the common arena");
        }
        auto validSelector = [&](RegionalSelector selector) {
            return selector.event.type < child.anchors.size() &&
                   validRegionalEvent(child.regional, selector.event) && expressions.isBoolean(selector.present);
        };
        auto prune = [&](std::vector<RegionalSelector>& values) {
            if (llvm::any_of(values, [&](const auto& value) { return !validSelector(value); })) {
                return fail("regional selector has an invalid occurrence or predicate");
            }
            llvm::erase_if(values, [&](const auto& value) {
                return expressions.constantValue(value.present) == 0;
            });
            return true;
        };
        // Validate dead selectors too, then discard their interface entries.
        // They must not create ports in later native/query matrix assembly.
        for (auto& cell : child.regional.storageBoundary) {
            if (!prune(cell.firstWriters) || !prune(cell.lastWriters)) { return false; }
            for (auto& [pipe, values] : cell.firstReaders) { if (!prune(values)) { return false; } }
            for (auto& [pipe, values] : cell.lastReaders) { if (!prune(values)) { return false; } }
        }
        for (auto* side : {&child.regional.firstPayloads, &child.regional.lastPayloads}) {
            for (auto& [pipe, values] : *side) { if (!prune(values)) { return false; } }
        }
        for (auto& [type, values] : child.regional.firstSitePayloads) {
            if (type >= child.anchors.size() || llvm::any_of(values, [site = type](const auto& value) {
                    return value.event.type != site;
                })) {
                return fail("regional first-site selector has an inconsistent payload type");
            }
            if (!prune(values)) { return false; }
        }
        auto convert = [&](RegionalSelector selector, std::vector<Selected>& destination) {
            if (!validSelector(selector)) {
                fail("regional selector has an invalid occurrence or predicate"); return;
            }
            destination.push_back({port(id, selector.event), selector.present});
        };
        for (std::size_t cell = 0; cell < cells.size(); ++cell) {
            auto& out = boundaries[id][cell];
            for (const auto& local : child.regional.storageBoundary) {
                if (!sameStorageDomain(local.cell, cells[cell]) || local.cell.begin > cells[cell].begin ||
                    local.cell.end < cells[cell].end) { continue; }
                for (auto selected : local.firstWriters) { convert(selected, out.firstWriters); }
                for (auto selected : local.lastWriters) { convert(selected, out.lastWriters); }
                for (const auto& [pipe, values] : local.firstReaders) {
                    for (auto selected : values) { convert(selected, out.firstReaders[pipe]); }
                }
                for (const auto& [pipe, values] : local.lastReaders) {
                    for (auto selected : values) { convert(selected, out.lastReaders[pipe]); }
                }
            }
            normalizeSelectorAlternatives(out.firstWriters);
            normalizeSelectorAlternatives(out.lastWriters);
            for (auto& [pipe, values] : out.firstReaders) { normalizeSelectorAlternatives(values); }
            for (auto& [pipe, values] : out.lastReaders) { normalizeSelectorAlternatives(values); }
        }
        for (const auto& access : child.regional.accessBoundary) {
            if (!child.regional.accessModel || access.effect >= child.regional.accessModel->effects().size()) {
                return fail("regional access selector has no shared effect");
            }
            // Register a port only when a residual crossing actually uses it.
            // Fully geometric composition retains its compact boundary graph.
            if (!validSelector(access.first) || !validSelector(access.last)) {
                return fail("regional access selector has an invalid occurrence or predicate");
            }
        }
        auto importNative = [&](const auto& source, NativeSelectors& destination) {
            for (const auto& [pipe, values] : source) {
                auto& selected = destination[pipe];
                for (auto value : values) { convert(value, selected); }
                normalizeSelectorAlternatives(selected);
            }
        };
        importNative(child.regional.firstPayloads, nativeFirst[id]);
        importNative(child.regional.lastPayloads, nativeLast[id]);
    }
    return error.empty();
}
RegionalAnalysis sequenceRegionalResult(const SequenceAnalysis& analysis)
{
    RegionalAnalysis out;
    if (!analysis.state || !analysis.error.empty()) { return out; }
    if (analysis.state->relationalResult) { return *analysis.state->relationalResult; }
    auto owned = std::make_shared<SequenceAnalysis>(analysis);
    auto state = analysis.state;
    out.expressions = state->arena;
    out.capabilities = {true, true, true, true};
    out.cost = analysis.cost;
    auto coordinates = std::make_shared<std::vector<std::pair<uint32_t, uint32_t>>>();
    std::vector<uint32_t> starts;
    for (uint32_t child = 0; child < state->children.size(); ++child) {
        starts.push_back(out.anchors.size());
        const auto& region = state->children[child].regional;
        for (uint32_t type = 0; type < region.anchors.size(); ++type) {
            coordinates->push_back({child, type});
            out.anchors.push_back(region.anchors[type]);
            out.occurrenceLoops.push_back(region.occurrenceLoops[type]);
            out.outerLoops.push_back(region.outerLoops.empty() ? std::vector<scf::ForOp>{} : region.outerLoops[type]);
            out.outerDivisors.push_back(region.outerDivisors.empty() ?
                std::vector<uint64_t>(out.outerLoops.back().size(), 1) : region.outerDivisors[type]);
        }
    }
    for (uint32_t child = 0; child < state->children.size(); ++child) {
        const auto& regional = state->children[child].regional;
        for (const auto& [type, firsts] : regional.firstSitePayloads) {
            auto& exported = out.firstSitePayloads[type + starts[child]];
            for (auto first : firsts) {
                first.event.type += starts[child];
                exported.push_back(std::move(first));
            }
        }
        if (regional.accessModel) {
            out.accessModel = regional.accessModel;
            out.gmAliasPolicy = regional.gmAliasPolicy;
        }
        for (auto access : regional.deferredAccessBoundary) {
            access.first.event.type += starts[child];
            access.last.event.type += starts[child];
            out.deferredAccessBoundary.push_back(std::move(access));
        }
        for (auto access : regional.accessBoundary) {
            access.first.event.type += starts[child];
            access.last.event.type += starts[child];
            out.accessBoundary.push_back(std::move(access));
        }
    }
    out.presence = [state, coordinates](RegionalEvent event) -> std::optional<Expr> {
        if (event.type >= coordinates->size()) { return std::nullopt; }
        auto [child, type] = (*coordinates)[event.type];
        return regionalPresence(state->children[child].regional, {type, event.ordinal, event.kind, event.visits});
    };
    out.endpointEventGuard = [state, coordinates](RegionalEvent event) -> std::optional<Expr> {
        if (event.type >= coordinates->size()) { return std::nullopt; }
        auto [child, type] = (*coordinates)[event.type];
        const auto& region = state->children[child].regional;
        event.type = type;
        auto result = region.endpointEventGuard ? region.endpointEventGuard(event) :
                      std::optional<Expr>(state->expressions.boolean(true));
        if (!result) { return std::nullopt; }
        if (region.endpointSiteGuard) { result = state->expressions.land(*result, *region.endpointSiteGuard); }
        return result;
    };
    out.reachability = [owned, coordinates](RegionalEvent a, RegionalEvent b) -> std::optional<Expr> {
        if (a.type >= coordinates->size() || b.type >= coordinates->size()) { return std::nullopt; }
        auto [ac, at] = (*coordinates)[a.type]; auto [bc, bt] = (*coordinates)[b.type];
        return sequenceEventReachability(*owned,
            {ac, at, a.ordinal, a.kind, a.visits}, {bc, bt, b.ordinal, b.kind, b.visits});
    };
    out.referenceBefore = [state, coordinates](RegionalEvent a, RegionalEvent b) -> std::optional<Expr> {
        if (a.type >= coordinates->size() || b.type >= coordinates->size()) { return std::nullopt; }
        auto [ac, at] = (*coordinates)[a.type]; auto [bc, bt] = (*coordinates)[b.type];
        if (ac != bc) { return state->expressions.boolean(ac < bc); }
        return regionalReferenceBefore(state->children[ac].regional,
            {at, a.ordinal, a.kind, a.visits}, {bt, b.ordinal, b.kind, b.visits});
    };
    for (const auto& child : analysis.state->children) {
        out.capabilities.endpointRecipes &= child.regional.capabilities.endpointRecipes &&
            (bool(child.regional.prepare) || bool(child.regional.prepareWithVisits));
        out.capabilities.contextualGuards |= child.regional.capabilities.contextualGuards;
        if (llvm::any_of(child.regional.storageBoundary, [](const auto& boundary) {
                return boundary.cell.space == AddressSpace::GM && boundary.cell.begin != boundary.cell.end;
            })) {
            out.gmAliasPolicy = child.regional.gmAliasPolicy;
        }
    }
    if (out.capabilities.endpointRecipes) {
        out.prepare = [owned]() -> FailureOr<std::unique_ptr<PreparedLogicalPlan>> {
            auto result = prepareSequenceLogicalInsertion(*owned);
            if (succeeded(result)) { (*result)->completeInvocation = false; }
            return result;
        };
        out.prepareWithVisits = [owned](ArrayRef<scf::ForOp> enclosing) {
            return owned->state->prepare(enclosing);
        };
    }
    auto convert = [&](SequenceSelectedEvent selected) {
        const auto& occurrence = analysis.occurrences[selected.port];
        return RegionalSelector{{starts[occurrence.child] + occurrence.type,
            occurrence.ordinal, PeriodicEventKind::Start, occurrence.visits}, selected.present};
    };
    for (const auto& cell : analysis.storageBoundary) {
        RegionalStorageBoundary target; target.cell = cell.cell;
        for (auto value : cell.firstWriters) { target.firstWriters.push_back(convert(value)); }
        for (auto value : cell.lastWriters) { target.lastWriters.push_back(convert(value)); }
        for (const auto& [pipe, values] : cell.firstReaders) {
            for (auto value : values) { target.firstReaders[pipe].push_back(convert(value)); }
        }
        for (const auto& [pipe, values] : cell.lastReaders) {
            for (auto value : values) { target.lastReaders[pipe].push_back(convert(value)); }
        }
        out.storageBoundary.push_back(std::move(target));
    }
    for (const auto& [pipe, values] : analysis.firstPayloads) {
        for (auto value : values) { out.firstPayloads[pipe].push_back(convert(value)); }
    }
    for (const auto& [pipe, values] : analysis.lastPayloads) {
        for (auto value : values) { out.lastPayloads[pipe].push_back(convert(value)); }
    }
    if (state->numerical) {
        auto numerical = std::make_shared<RegionalNumericalInterface>();
        numerical->index = std::shared_ptr<const NumericalChainInterface>(state->numerical, &state->numerical->index);
        numerical->chainKeys = state->numericalChainKeys;
        for (const auto& port : state->ports) {
            for (auto kind : {PeriodicEventKind::Start, PeriodicEventKind::Completion}) {
                numerical->events.push_back({starts[port.child] + port.type, port.ordinal, kind, port.visits});
            }
        }
        numerical->query = [state, coordinates](RegionalEvent a, RegionalEvent b, NumericalChainQueryCost& cost)
            -> std::optional<bool> {
            if (a.type >= coordinates->size() || b.type >= coordinates->size()) { return std::nullopt; }
            auto [ac, at] = (*coordinates)[a.type]; auto [bc, bt] = (*coordinates)[b.type];
            auto answer = state->numericalReachability({ac, at, a.ordinal, a.kind, a.visits},
                {bc, bt, b.ordinal, b.kind, b.visits}, cost);
            auto value = answer ? state->expressions.constantValue(*answer) : std::nullopt;
            return value ? std::optional<bool>(*value != 0) : std::nullopt;
        };
        numerical->thresholds = [state, coordinates](RegionalEvent event, bool reverse, NumericalChainQueryCost& cost)
            -> std::optional<std::vector<uint32_t>> {
            if (event.type >= coordinates->size()) { return std::nullopt; }
            auto [child, type] = (*coordinates)[event.type];
            return state->numericalThresholds({child, type, event.ordinal, event.kind, event.visits}, reverse, cost);
        };
        out.numerical = std::move(numerical);
    }
    const bool hasSymbolic = llvm::any_of(state->children, [](const Child& child) {
        return child.regional.storageSelectors || !child.regional.symbolicStorageEffects.empty();
    });
    if (hasSymbolic) {
        out.storageSelectors = [state, starts](RegionalByteAddress address) {
            return sequenceStorageSelectors(*state, starts, address);
        };
        for (const auto& child : state->children) {
            for (auto effect : child.regional.symbolicStorageEffects) {
                if (!llvm::is_contained(out.symbolicStorageEffects, effect)) {
                    out.symbolicStorageEffects.push_back(effect);
                }
            }
        }
        // Physical membership survives sequence composition. Owner/selector
        // coordinates do not: retain the merged selector callback above, and
        // clear owner maps unless a producer supplies a composed certificate.
        auto certificate = std::make_shared<RegionalSymbolicStorageCertificate>();
        certificate->expressions = out.expressions;
        certificate->accessModel = out.accessModel;
        certificate->gmAliasPolicy = out.gmAliasPolicy;
        for (const auto& child : state->children) {
            if (!child.regional.symbolicStorage) { continue; }
            for (auto family : child.regional.symbolicStorage->families) {
                family.kind = RegionalStorageFamilyKind::General;
                family.owner = {};
                certificate->families.push_back(std::move(family));
            }
        }
        out.symbolicStorage = std::move(certificate);
    }
    if (state->children.size() == 1) {
        // A unary sequence changes neither identities nor the selected graph.
        out.numerical = state->children.front().regional.numerical;
        if (out.numerical) { out.reachability = state->children.front().regional.reachability; }
        out.storageSelectors = state->children.front().regional.storageSelectors;
        out.symbolicStorage = state->children.front().regional.symbolicStorage;
        out.arithmeticRelations = state->children.front().regional.arithmeticRelations;
        out.symbolicStorageEffects = state->children.front().regional.symbolicStorageEffects;
    }
    return out;
}
} // namespace mlir::pto::frontiersynch
