// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_ORIGINALOBLIGATIONADAPTER_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_ORIGINALOBLIGATIONADAPTER_H

#include "PTO/Transforms/FrontierSynch/ProgramAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include <set>

namespace mlir::pto::frontiersynch::detail {

// The base original representation uses NoControlId for the invocation scope.
// These obligations quantify over the complete original prefix ending just
// before their own consumer. Smaller use/placement intervals remain separate
// query answers; lexical child exits cannot truncate this obligation universe.
inline OriginalObligationKey obligationKey(
    const OriginalStructure& original, std::size_t target, std::size_t cell, OriginalObligationKey::Kind kind)
{
    OriginalObligationKey key;
    key.kind = kind;
    key.originalVersion = original.version;
    key.cell = cell;
    key.consumerOperation = target;
    key.consumerOriginal = original.operations[target].original;
    key.owner = original.body.originalOwner;
    key.start = {ObligationCut::Kind::OwnerEntry, key.owner};
    key.stop = {ObligationCut::Kind::PayloadBefore, target};
    key.includeStop = false;
    return key;
}

inline bool obligationHasRepetition(const Region& region)
{
    if (region.kind == Region::For || region.kind == Region::While) {
        return true;
    }
    for (const auto& child : region.children) {
        if (obligationHasRepetition(child)) {
            return true;
        }
    }
    return false;
}

inline std::unique_ptr<OriginalObligations> buildOriginalObligations(
    const OriginalStructure& original, const OriginalLifetimes& storage, const OriginalValueQueries& queries,
    const std::function<const std::vector<TypedOriginalRequirement>&(std::size_t)>& typedAt)
{
    using Kind = OriginalObligationKey::Kind;
    using Rep = OriginalObligationFamily::Representation;
    const bool repeated = obligationHasRepetition(original.body);

    // The fixed-use core labels a choice by its original site. Translate that
    // label to the defining SSA value identity, shared across all its if sites.
    // There is no claim that this value is observable at a proposed endpoint.
    auto guardIds = std::make_shared<std::map<std::size_t, std::size_t>>();
    std::map<FactoredGuardIdentity, std::size_t> values;
    llvm::DenseMap<mlir::Operation*, std::size_t> originalIds;
    for (std::size_t site = 0; site < original.originalSites.size(); ++site) {
        auto* op = original.originalSites[site];
        originalIds[op] = site;
        if (auto choice = dyn_cast<scf::IfOp>(op)) {
            const auto identity = queries.identity(choice.getCondition());
            auto inserted = values.try_emplace(
                FactoredGuardIdentity{
                    reinterpret_cast<std::uintptr_t>(identity.value.getAsOpaquePointer()), identity.occurrenceScope},
                site);
            (*guardIds)[site] = inserted.first->second;
        }
    }
    const auto marginalKeySupported = [&original](const OriginalObligationKey& key) {
        if (key.occurrences != OriginalObligationKey::Occurrences::OriginalControlPaths ||
            key.start.kind != ObligationCut::Kind::OwnerEntry || key.stop.kind != ObligationCut::Kind::PayloadBefore ||
            key.includeStop || key.originalVersion != original.version || key.occurrenceInterpretation != 0 ||
            key.owner != original.body.originalOwner || key.start.site != key.owner ||
            key.stop.site != key.consumerOperation || key.consumerOperation >= original.operations.size() ||
            original.operations[key.consumerOperation].original != key.consumerOriginal) {
            return false;
        }
        return true;
    };
    auto result = std::make_unique<OriginalObligations>(
        [values](FactoredGuardIdentity identity) {
            const auto found = values.find(identity);
            return found == values.end() ? NoControlId : found->second;
        },
        [&storage, marginalKeySupported](
            const OriginalObligationKey& key, ObligationOrigin source) -> std::optional<bool> {
            if (!marginalKeySupported(key)) {
                return std::nullopt;
            }
            return storage.mayOriginAt(
                key.consumerOperation, key.cell, source.operation, key.kind == Kind::WAR, source.incoming);
        },
        [&storage,
         marginalKeySupported](const OriginalObligationKey& key) -> std::optional<std::vector<ObligationOrigin>> {
            if (!marginalKeySupported(key)) {
                return std::nullopt;
            }
            if (!storage.hasMayOriginsAt(key.consumerOperation, key.cell, key.kind == Kind::WAR)) {
                return std::nullopt;
            }
            std::vector<ObligationOrigin> out;
            for (const auto& origin : storage.mayOriginsAt(key.consumerOperation, key.cell, key.kind == Kind::WAR)) {
                out.push_back({origin.operation, false});
            }
            if (storage.mayOriginAt(key.consumerOperation, key.cell, NoControlId, key.kind == Kind::WAR, true)
                    .value_or(true)) {
                out.push_back(ObligationOrigin::entry());
            }
            return out;
        },
        [&queries] { return queries.current(); });

    auto& predicates = result->formingConditions();
    std::vector<std::size_t> participation(original.originalSites.size(), ObligationConditions::yes);
    std::vector<bool> identified(original.originalSites.size(), !repeated);
    // On repeated programs the baseline has no occurrence-scoped guarded summary.
    // Keep the original-control marginal relation instead of reusing one Boolean
    // valuation for different visits. D4 will refine these same obligations.
    if (!repeated) {
        for (std::size_t site = 0; site < original.originalSites.size(); ++site) {
            auto* child = original.originalSites[site];
            for (auto* parent = child->getParentOp(); parent; child = parent, parent = parent->getParentOp()) {
                if (auto choice = dyn_cast<scf::IfOp>(parent)) {
                    const auto found = originalIds.find(parent);
                    if (found == originalIds.end() || !guardIds->count(found->second)) {
                        identified[site] = false;
                        continue;
                    }
                    auto guard = predicates.atom(guardIds->at(found->second));
                    if (child->getParentRegion() == &choice.getElseRegion()) {
                        guard = predicates.negate(guard);
                    }
                    participation[site] = predicates.choose(guard, participation[site], ObligationConditions::no);
                }
            }
        }
    }

    std::map<const FactoredUseArena*, bool> mappedArenaGuards;
    for (std::size_t target = 0; target < original.operations.size(); ++target) {
        std::set<std::pair<std::size_t, Kind>> roles;
        for (const auto& access : original.operations[target].accesses) {
            if (access.read) {
                roles.emplace(access.cell, Kind::RAW);
            }
            if (access.write) {
                roles.emplace(access.cell, Kind::WAR);
                roles.emplace(access.cell, Kind::WAW);
            }
        }
        for (const auto& [cell, kind] : roles) {
            OriginalObligationFamily family;
            family.key = obligationKey(original, target, cell, kind);
            const auto site = family.key.consumerOriginal;
            family.applicability = participation.at(site);
            family.consumerEffects = &storage.effectIncidences(target, cell, kind != Kind::RAW);
            const auto& expression = storage.factoredAt(target, cell);
            const auto* use = expression.site(target);
            auto mapped = mappedArenaGuards.try_emplace(expression.arena.get(), true);
            if (mapped.second) {
                for (const auto& node : expression.nodes()) {
                    if (node.kind == FactoredUseNode::Kind::Test && !values.count(node.guard)) {
                        mapped.first->second = false;
                        break;
                    }
                }
            }
            if (expression.complete && use && use->visited && mapped.first->second) {
                family.representation = Rep::Factored;
                family.key.occurrences = OriginalObligationKey::Occurrences::FixedUseProjection;
                family.key.owner = expression.frame.owner;
                family.key.occurrenceInterpretation = static_cast<std::size_t>(expression.frame.kind);
                family.key.start = {ObligationCut::Kind::OwnerEntry, expression.frame.owner};
                family.guardsIdentified = true;
                family.expression = &expression;
                family.sources = kind == Kind::WAR ? use->priorReaders : use->priorWriters;
                const auto hazard = kind == Kind::RAW ? FactoredUseNode::Hazard::RAW :
                                    kind == Kind::WAR ? FactoredUseNode::Hazard::WAR :
                                                        FactoredUseNode::Hazard::WAW;
                auto demand = use->demands[static_cast<std::size_t>(hazard)];
                if (demand && expression.nodes()[demand].kind == FactoredUseNode::Kind::Choose) {
                    demand = expression.nodes()[demand].left;
                }
                family.demandNode = demand ? demand : NoObligationIndex;
                family.applicability = result->importCondition(expression, use->applicability);
                if (!family.sources) {
                    continue;
                }
                family.unresolved =
                    "original scoped relation; incoming transport and endpoint qualification are separate";
            } else {
                family.representation = Rep::Marginal;
                if (!storage.hasMayOriginsAt(target, cell, kind == Kind::WAR).value_or(true)) {
                    continue;
                }
                family.unresolved = expression.reason + "; conservative original-control obligations retained";
            }
            result->add(std::move(family));
        }
    }

    // Control deadlines use original site IDs, including sites with no translated
    // payload phase. Multiple SSA producer phases are conjunctive members of one
    // typed family, with the existing unresolved incoming case retained.
    for (std::size_t site = 0; site < original.originalSites.size(); ++site) {
        // Preserve original prerequisite encounter order. Pointer identities are
        // equality keys only, never an ordering tie-break for stable family IDs.
        std::map<std::pair<std::size_t, std::size_t>, std::size_t> typedIndex;
        std::vector<OriginalObligationFamily> typed;
        for (const auto& request : typedAt(site)) {
            const auto value = reinterpret_cast<std::uintptr_t>(request.requiredValue.getAsOpaquePointer());
            const auto cause = static_cast<std::size_t>(request.cause);
            auto inserted = typedIndex.emplace(std::make_pair(value, cause), typed.size());
            if (inserted.second) {
                typed.emplace_back();
            }
            auto& family = typed[inserted.first->second];
            family.representation = Rep::Typed;
            family.key.kind = Kind::Typed;
            family.key.originalVersion = original.version;
            family.key.typedValue = value;
            family.key.typedCause = cause;
            family.key.consumerOriginal = site;
            family.key.owner = original.body.originalOwner;
            family.key.start = {ObligationCut::Kind::OwnerEntry, family.key.owner};
            family.key.stop = {ObligationCut::Kind::OriginalBefore, site};
            family.applicability = participation[site];
            family.guardsIdentified = identified[site];
            family.typedSources.push_back(
                request.incomingOrUnknownSource ? ObligationOrigin::entry() :
                                                  ObligationOrigin{request.completion.sourcePhase, false});
            family.unresolved = "original-value availability and occurrence qualification remain separate";
        }
        for (auto& family : typed) {
            std::sort(family.typedSources.begin(), family.typedSources.end());
            family.typedSources.erase(
                std::unique(family.typedSources.begin(), family.typedSources.end()), family.typedSources.end());
            result->add(std::move(family));
        }
    }
    result->freeze();
    return result;
}

} // namespace mlir::pto::frontiersynch::detail
#endif
