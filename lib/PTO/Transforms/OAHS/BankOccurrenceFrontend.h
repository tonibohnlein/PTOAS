// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_OAHS_BANK_OCCURRENCE_FRONTEND_H
#define PTO_OAHS_BANK_OCCURRENCE_FRONTEND_H
#include "PTO/Transforms/OAHS/ObservedPrograms.h"
#include <algorithm>
#include <map>
#include <set>
#include <tuple>

namespace mlir::pto::oahs {
namespace bank_occurrence_detail {
using Effect = std::tuple<unsigned, bool, bool, bool>;
inline std::set<Effect> effectSet(const std::vector<Access>& accesses)
{
    std::set<Effect> out;
    for (const auto& access : accesses) {
        out.emplace(access.cell, access.read, access.write, access.definiteWrite);
    }
    return out;
}

// The frontend supplies exact per-residue byte effects. Only varying effects
// are overlaid: an enclosing MAT selector cannot undo a child's L0 selection.
inline std::vector<Access> specialize(const std::vector<Access>& accesses,
    const CountedLoopRegion::PeriodicEffects& binding, unsigned residue)
{
    std::set<Effect> varying, common = effectSet(binding.residues.front());
    for (const auto& effects : binding.residues) {
        const auto current = effectSet(effects);
        varying.insert(current.begin(), current.end());
        for (auto it = common.begin(); it != common.end();) {
            if (!current.count(*it)) {
                it = common.erase(it);
            } else {
                ++it;
            }
        }
    }
    for (const auto& effect : common) {
        varying.erase(effect);
    }
    std::vector<Access> out;
    for (const auto& access : accesses) {
        if (!varying.count({access.cell, access.read, access.write, access.definiteWrite})) {
            out.push_back(access);
        }
    }
    for (const auto& access : binding.residues[residue]) {
        if (varying.count({access.cell, access.read, access.write, access.definiteWrite})) {
            out.push_back(access);
        }
    }
    return out;
}

inline bool validBoundary(const ObservedControl& q, const CountedLoopRegion& loop,
    const std::set<std::size_t>& members)
{
    const auto size = q.sites.size();
    if (loop.owner >= size || loop.header >= size || loop.bodyEntry >= size ||
        loop.continuation >= size || members.size() != loop.bodySites.size() ||
        !members.count(loop.bodyEntry) || members.count(loop.owner) ||
        members.count(loop.header) || members.count(loop.continuation) ||
        members.count(q.entry) || q.entry == loop.header) {
        return false;
    }
    if (q.sites[loop.owner].successors != std::vector<std::size_t>{loop.header} ||
        q.sites[loop.header].operation != NoControlId ||
        q.sites[loop.header].observation != NoControlId ||
        q.sites[loop.header].successors != std::vector<std::size_t>{loop.bodyEntry, loop.continuation}) {
        return false;
    }
    for (auto site : members) {
        if (site >= size) {
            return false;
        }
    }
    std::set<std::size_t> reachable;
    std::vector<std::size_t> todo{q.entry};
    while (!todo.empty()) {
        const auto at = todo.back();
        todo.pop_back();
        if (reachable.insert(at).second) {
            todo.insert(todo.end(), q.sites[at].successors.begin(), q.sites[at].successors.end());
        }
    }
    for (auto site : reachable) {
        for (auto next : q.sites[site].successors) {
            const bool leavesBody = members.count(site) && !members.count(next);
            const bool entersBody = !members.count(site) && members.count(next);
            const bool jumpsToHeader = !members.count(site) && next == loop.header;
            if (leavesBody && next != loop.header) {
                return false;
            }
            if (entersBody && site != loop.header) {
                return false;
            }
            if (jumpsToHeader && site != loop.owner) {
                return false;
            }
        }
    }
    return true;
}

inline bool expandChildBoundaries(ObservedLoop& child,
    const std::vector<std::map<std::size_t, std::size_t>>& copies)
{
    const auto original = loopEntryOccurrences(child);
    if (original.empty()) {
        return false;
    }
    std::vector<ObservedLoopOccurrence> occurrences;
    for (const auto& copy : copies) {
        for (const auto& occurrence : original) {
            auto mapped = occurrence;
            auto remap = [&](std::size_t& site) {
                const auto found = copy.find(site);
                if (found == copy.end()) {
                    return false;
                }
                site = found->second;
                return true;
            };
            if (!remap(mapped.entry) || !remap(mapped.exit) ||
                (mapped.bodyEntry != NoControlId && !remap(mapped.bodyEntry))) {
                return false;
            }
            for (auto* sites : {&mapped.sites, &mapped.exits}) {
                for (auto& site : *sites) {
                    if (!remap(site)) {
                        return false;
                    }
                }
            }
            occurrences.push_back(std::move(mapped));
        }
    }
    auto expand = [&](const std::vector<std::size_t>& sites) {
        std::vector<std::size_t> out;
        for (const auto& copy : copies) {
            for (auto site : sites) {
                const auto found = copy.find(site);
                if (found != copy.end()) {
                    out.push_back(found->second);
                }
            }
        }
        return out;
    };
    child.entries = expand(child.entries.empty() ? std::vector<std::size_t>{child.entry} : child.entries);
    child.exits = expand(child.exits.empty() ? std::vector<std::size_t>{child.exit} : child.exits);
    child.sites = expand(child.sites);
    child.entry = child.entries.front();
    child.exit = child.exits.front();
    child.occurrences = std::move(occurrences);
    child.bodyEntry = child.occurrences.front().bodyEntry;
    return true;
}
} // namespace bank_occurrence_detail

ObservedImport refineBankOccurrences(const Program& input, const CountedLoopRegion& loop)
{
    ObservedImport out;
    const auto valid = validateProgram(input);
    if (!valid.success) {
        out.reason = valid.reason;
        return out;
    }
    const std::set<std::size_t> members(loop.bodySites.begin(), loop.bodySites.end());
    const bool invalidOrbit = !input.observed || loop.period < 2 || !loop.decisions.empty() ||
        loop.effects.empty();
    const bool invalidBoundary = input.observed &&
        !bank_occurrence_detail::validBoundary(*input.observed, loop, members);
    if (invalidOrbit || invalidBoundary) {
        out.reason = "bank interface requires a qualified finite orbit and original region boundary";
        return out;
    }
    const auto& old = *input.observed;
    for (auto site : members) {
        const auto observation = old.sites[site].observation;
        if (observation == NoControlId) {
            continue;
        }
        for (const auto& atom : old.observations[observation].atoms) {
            if (atom.owner == loop.owner || atom.kind == ObservationAtom::LoopResidue) {
                // Existing child counted modes are allowed; a second bank-only
                // layer is not silently multiplied into a nested bank product.
                const auto& atoms = old.observations[observation].atoms;
                const bool counted = std::any_of(atoms.begin(), atoms.end(), [&](const auto& other) {
                    return other.owner == atom.owner && other.kind == ObservationAtom::LoopHasPrevious;
                });
                if (atom.owner == loop.owner || !counted) {
                    out.reason = "bank interface would duplicate an enclosing bank dimension";
                    return out;
                }
            }
        }
    }
    std::map<std::size_t, const CountedLoopRegion::PeriodicEffects*> effects;
    for (const auto& binding : loop.effects) {
        const bool unknownOperation = binding.operation >= input.operations.size();
        const bool invalidPeriod = binding.residues.size() != loop.period;
        const bool duplicateOperation = !effects.emplace(binding.operation, &binding).second;
        if (unknownOperation || invalidPeriod || duplicateOperation) {
            out.reason = "invalid bank effect binding";
            return out;
        }
        for (const auto& residue : binding.residues) {
            for (const auto& access : residue) {
                const bool invalidAccess = access.cell >= input.cells.size() || access.definiteWrite;
                if (invalidAccess) {
                    out.reason = "bank effect requires existing conservative storage cells";
                    return out;
                }
            }
        }
    }
    out.program = input;
    auto& q = *out.program.observed;
    const auto capacity = (q.sites.max_size() - q.sites.size()) / (loop.period - 1);
    const bool insufficientCapacity = members.size() + 1 > capacity;
    if (insufficientCapacity) {
        out.reason = "bank interface exceeds intrinsic site capacity";
        return out;
    }
    std::vector<std::map<std::size_t, std::size_t>> copies(loop.period);
    std::vector<std::size_t> headers;
    std::map<std::pair<std::size_t, unsigned>, std::size_t> phases, observations;
    for (unsigned residue = 0; residue < loop.period; ++residue) {
        headers.push_back(residue == 0 ? loop.header : q.sites.size());
        if (residue != 0) {
            q.sites.emplace_back();
            q.sites.back().context = old.sites[loop.header].context;
        }
        for (auto site : members) {
            copies[residue][site] = residue == 0 ? site : q.sites.size();
            auto node = old.sites[site];
            if (node.operation != NoControlId) {
                const auto original = input.operations[node.operation].original;
                const auto binding = effects.find(original);
                if (binding != effects.end()) {
                    const auto key = std::make_pair(node.operation, residue);
                    auto found = phases.find(key);
                    if (found == phases.end()) {
                        auto op = input.operations[node.operation];
                        op.accesses = bank_occurrence_detail::specialize(op.accesses, *binding->second, residue);
                        const bool unchanged = bank_occurrence_detail::effectSet(op.accesses) ==
                            bank_occurrence_detail::effectSet(input.operations[node.operation].accesses);
                        if (unchanged) {
                            found = phases.emplace(key, node.operation).first;
                        } else if (residue == 0) {
                            found = phases.emplace(key, node.operation).first;
                            out.program.operations[node.operation] = std::move(op);
                        } else {
                            found = phases.emplace(key, out.program.operations.size()).first;
                            out.program.operations.push_back(std::move(op));
                        }
                    }
                    node.operation = found->second;
                }
            }
            if (node.observation != NoControlId) {
                const auto key = std::make_pair(node.observation, residue);
                auto found = observations.find(key);
                if (found == observations.end()) {
                    auto observation = old.observations[node.observation];
                    observation.atoms.push_back({ObservationAtom::LoopResidue, loop.owner, loop.period, residue});
                    found = observations.emplace(key, q.observations.size()).first;
                    q.observations.push_back(std::move(observation));
                }
                node.observation = found->second;
            }
            if (residue == 0) {
                q.sites[site] = std::move(node);
            } else {
                q.sites.push_back(std::move(node));
            }
        }
    }
    for (unsigned residue = 0; residue < loop.period; ++residue) {
        q.sites[headers[residue]].successors = {copies[residue].at(loop.bodyEntry), loop.continuation};
        for (auto site : members) {
            auto& node = q.sites[copies[residue].at(site)];
            for (auto& target : node.successors) {
                target = target == loop.header ? headers[(residue + 1) % loop.period] : copies[residue].at(target);
            }
        }
    }
    q.sites[loop.owner].successors = {loop.atLeastOnce ? copies[0].at(loop.bodyEntry) : headers[0]};
    for (auto& child : q.loops) {
        const bool contained = members.count(child.entry) && members.count(child.exit);
        if (contained) {
            if (!bank_occurrence_detail::expandChildBoundaries(child, copies)) {
                out.reason = "incomplete child occurrence boundary correspondence";
                return out;
            }
        }
    }
    ObservedLoop region{loop.owner, loop.owner, loop.continuation, {}, copies[0].at(loop.bodyEntry), loop.atLeastOnce};
    for (const auto& copy : copies) {
        for (const auto& pair : copy) {
            region.sites.push_back(pair.second);
        }
    }
    region.sites.insert(region.sites.end(), headers.begin(), headers.end());
    q.loops.push_back(std::move(region));
    q.qualification += "; bank-occurrence-interface-v1";
    const auto checked = validateProgram(out.program);
    out.success = checked.success;
    out.reason = checked.reason;
    for (const auto& operation : out.program.operations) {
        out.originalPhases.push_back(operation.original);
    }
    return out;
}
} // namespace mlir::pto::oahs
#endif
