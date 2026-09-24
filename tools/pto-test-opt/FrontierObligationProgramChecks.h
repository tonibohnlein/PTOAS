// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#ifndef PTO_TEST_FRONTIER_OBLIGATION_PROGRAM_CHECKS_H
#define PTO_TEST_FRONTIER_OBLIGATION_PROGRAM_CHECKS_H
#include "PTO/Transforms/FrontierSynch/ProgramAnalysis.h"
#include <set>

// Native-only checks run before the legacy pair/corpus queries. They deliberately
// request origin output here; production formation must not request that output.
inline bool checkProgramObligations(const mlir::pto::frontiersynch::ProgramAnalysis& analysis)
{
    namespace fs = mlir::pto::frontiersynch;
    using Kind = fs::OriginalObligationKey::Kind;
    using Status = fs::ObligationMembership::Status;
    const auto& index = analysis.obligations();
    const auto& structure = analysis.structure();
    if (!index.complete() || analysis.lifetimes().stats().requirements != 0 || index.stats().enumeratedOrigins != 0 ||
        index.stats().membershipQueries != 0) {
        return false;
    }
    std::set<std::size_t> seen;
    for (std::size_t site = 0; site < structure.originalSites.size(); ++site) {
        for (auto id : analysis.obligationsAt(site)) {
            const auto* family = index.get(id);
            if (!family || family->key.consumerOriginal != site || !seen.insert(id.index).second) {
                return false;
            }
            const auto population = index.origins(id);
            if (!population.complete) {
                return false;
            }
            if (family->key.kind != Kind::Typed) {
                if (!family->consumerEffects || family->consumerEffects->empty() ||
                    family->key.consumerOperation >= structure.operations.size()) {
                    return false;
                }
            }
            for (const auto& member : population.members) {
                const auto witness = analysis.obligationWitness(member.id());
                if (witness.membership.status == Status::Invalid || witness.membership.status == Status::Excluded) {
                    return false;
                }
                if (member.source.incoming && witness.sourceOperation) {
                    return false;
                }
                if (family->key.kind == Kind::Typed) {
                    if (!witness.typedRequirement || witness.typedRequirement->deadlineOriginalSite != site) {
                        return false;
                    }
                    continue;
                }
                if (witness.physicalCell != &structure.cells[family->key.cell] ||
                    witness.consumerOperation != &structure.operations[family->key.consumerOperation] ||
                    witness.consumerEffects != family->consumerEffects) {
                    return false;
                }
                for (auto effect : *witness.consumerEffects) {
                    const auto& incidences = witness.consumerOperation->accesses;
                    if (effect >= incidences.size() || incidences[effect].cell != family->key.cell ||
                        (family->key.kind == Kind::RAW ? !incidences[effect].read : !incidences[effect].write)) {
                        return false;
                    }
                }
                if (!member.source.incoming) {
                    if (!witness.sourceOperation || !witness.sourceEffects || witness.sourceEffects->empty()) {
                        return false;
                    }
                    for (auto effect : *witness.sourceEffects) {
                        const auto& incidences = witness.sourceOperation->accesses;
                        if (effect >= incidences.size() || incidences[effect].cell != family->key.cell ||
                            (family->key.kind == Kind::WAR ? !incidences[effect].read : !incidences[effect].write)) {
                            return false;
                        }
                    }
                }
            }
        }
        // Every legacy typed/control prerequisite has an ID even at a control-only
        // site. Being an incoming or unresolved prerequisite is not an exclusion.
        for (const auto& request : analysis.typedRequirementsAt(site)) {
            bool found = false;
            for (auto id : analysis.obligationsAt(site)) {
                const auto* family = index.get(id);
                if (family->key.kind != Kind::Typed ||
                    family->key.typedCause != static_cast<std::size_t>(request.cause) ||
                    family->key.typedValue !=
                        reinterpret_cast<std::uintptr_t>(request.requiredValue.getAsOpaquePointer())) {
                    continue;
                }
                const auto origin = request.incomingOrUnknownSource ?
                                        fs::ObligationOrigin::entry() :
                                        fs::ObligationOrigin{request.completion.sourcePhase, false};
                // A syntactically contradictory original arm may be excluded by
                // its retained applicability, but its family identity still exists.
                found |= index.membership(id, origin).status != Status::Invalid;
            }
            if (!found) {
                return false;
            }
        }
    }
    for (std::size_t node = 0; node < index.predicates().size(); ++node) {
        const auto* predicate = index.predicates().get(node);
        if (predicate->kind == fs::ObligationConditions::Node::Kind::Atom &&
            !analysis.obligationGuardValue(predicate->left)) {
            return false;
        }
    }
    return seen.size() == index.stats().families && analysis.lifetimes().stats().requirements == 0;
}
#endif
