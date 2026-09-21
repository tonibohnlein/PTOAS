// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_OAHS_READER_VISIT_FRONTEND_H
#define PTO_OAHS_READER_VISIT_FRONTEND_H
#include "PTO/Transforms/OAHS/ObservedPrograms.h"
#include <algorithm>
#include <functional>
#include <map>
#include <set>
namespace mlir::pto::oahs {
ObservedImport refineReaderVisits(const Program& input, const ReaderVisitRegion& r)
{
    ObservedImport out;
    auto reject = [&](const char* reason) {
        out.reason = reason;
        return out;
    };
    const auto valid = validateProgram(input);
    if (!valid.success) {
        out.reason = valid.reason;
        return out;
    }
    if (!input.observed || r.firstConsumers.empty() || r.lastPublications.empty() || !r.step)
        return reject("joint reader visits need first/final endpoints and a positive step");
    const auto& old = *input.observed;
    const auto found =
        std::find_if(old.loops.begin(), old.loops.end(), [&](const auto& loop) { return loop.owner == r.owner; });
    if (found == old.loops.end() || r.owner >= old.sites.size() || !found->atLeastOnce || found->entry != r.owner ||
        found->exit >= old.sites.size() || found->bodyEntry == NoControlId || !found->firstVisitPrefix.empty() ||
        !found->entries.empty() || !found->exits.empty() || old.sites[r.owner].successors.size() != 1)
        return reject("joint reader visits need an unrefined nonempty owner");
    const auto header = old.sites[r.owner].successors.front();
    const auto body = found->bodyEntry;
    if (old.sites[header].successors != std::vector<std::size_t>{body, found->exit})
        return reject("joint reader visits need original counted control");
    std::set<std::size_t> members, active;
    std::function<bool(std::size_t)> walk = [&](std::size_t at) {
        if (at == header)
            return true;
        if (at == found->exit || at >= old.sites.size() || active.count(at))
            return false;
        if (members.count(at))
            return true;
        members.insert(at);
        active.insert(at);
        const auto& site = old.sites[at];
        if (site.successors.empty())
            return false;
        for (std::size_t i = 0; i < site.successors.size(); ++i) {
            const auto next = site.successors[i];
            const auto edgeOwner = site.backedgeOwners.empty() ? NoControlId : site.backedgeOwners[i];
            if ((edgeOwner != NoControlId && (edgeOwner != r.owner || next != header)) || !walk(next))
                return false;
        }
        active.erase(at);
        return true;
    };
    if (!walk(body))
        return reject("joint reader suffix needs an acyclic continuation to its header");
    for (const auto& loop : old.loops)
        if (loop.owner != r.owner && members.count(loop.entry))
            return reject("joint reader visits need a leaf body");
    for (std::size_t at = 0; at < old.sites.size(); ++at)
        for (auto next : old.sites[at].successors)
            if (members.count(next) && !members.count(at) && !(at == header && next == body))
                return reject("joint reader body has an external entry");

    const std::set<std::size_t> first(r.firstConsumers.begin(), r.firstConsumers.end());
    const std::set<std::size_t> last(r.lastPublications.begin(), r.lastPublications.end());
    std::set<std::size_t> pending = first;
    pending.insert(last.begin(), last.end());
    std::vector<std::size_t> prefix;
    auto at = body;
    // Keep the following anchor in the first prefix so the final copied payload
    // retains a straight post-access source position.
    while (!pending.empty() || prefix.empty()) {
        if (!members.count(at) || old.sites[at].successors.size() != 1)
            return reject("joint reader endpoints need an unconditional prefix");
        const auto observation = old.sites[at].observation;
        if (observation == NoControlId || !old.observations[observation].atoms.empty())
            return reject("joint reader prefix has an incompatible observation");
        prefix.push_back(at);
        pending.erase(at);
        at = old.sites[at].successors.front();
    }
    if (members.count(at))
        prefix.push_back(at);
    for (auto site : first) {
        if (last.count(site))
            return reject("first/final endpoints need separate word gaps");
        if (old.sites[site].operation == NoControlId)
            return reject("first consumer is not a payload");
    }
    for (auto site : first)
        pending.insert(site);
    pending.insert(last.begin(), last.end());
    for (auto site : pending) {
        const auto observation = old.sites[site].observation;
        if (std::count_if(
                old.sites.begin(), old.sites.end(), [&](const auto& s) { return s.observation == observation; }) != 1)
            return reject("joint reader endpoint is already shared");
    }

    // Validation above covers BOTH endpoint frontiers. The first-visit copy
    // only needs to reach its last acquisition and following source anchor;
    // later final-publication words can remain shared with interior visits.
    std::size_t firstEnd = 0;
    for (std::size_t i = 0; i < prefix.size(); ++i)
        if (first.count(prefix[i]))
            firstEnd = i + 1;
    prefix.resize(std::min(prefix.size(), firstEnd + 1));

    out.program = input;
    auto& q = *out.program.observed;
    std::map<std::pair<std::size_t, bool>, std::size_t> words;
    auto observe = [&](std::size_t original, std::size_t copy, bool isFirst, bool isFinal) {
        if (!first.count(original) && !last.count(original))
            return;
        const bool value = first.count(original) ? !isFirst : !isFinal;
        const auto key = std::make_pair(original, value);
        auto [where, inserted] = words.emplace(key, q.observations.size());
        if (inserted) {
            auto observation = old.observations[old.sites[original].observation];
            observation.atoms.push_back(
                {first.count(original) ? ObservationAtom::LoopHasPrevious : ObservationAtom::LoopHasNext, r.owner,
                 first.count(original) ? 1 : r.step, unsigned(value)});
            q.observations.push_back(std::move(observation));
        }
        q.sites[copy].observation = where->second;
    };
    std::vector<std::size_t> added, firstPrefix;
    if (r.singleVisit) {
        for (auto site : members) {
            observe(site, site, true, true);
            for (auto& next : q.sites[site].successors)
                if (next == header)
                    next = found->exit;
            q.sites[site].backedgeOwners.clear();
        }
        q.sites[r.owner].successors = {body};
        firstPrefix = prefix;
    } else {
        std::map<std::size_t, std::size_t> finalCopy, firstCopy;
        for (auto site : members) {
            finalCopy[site] = q.sites.size();
            added.push_back(q.sites.size());
            q.sites.push_back(old.sites[site]);
            observe(site, site, false, false);
            observe(site, finalCopy.at(site), false, true);
        }
        for (auto site : members) {
            auto& copy = q.sites[finalCopy.at(site)];
            for (auto& next : copy.successors)
                next = next == header ? found->exit : finalCopy.at(next);
            copy.backedgeOwners.clear();
        }
        // First and interior visits share the conditional suffix itself. The
        // final continuation needs distinct analytical sites to preserve exit
        // correlation, but ALL suffix payload and command identities stay shared.
        // There is no product of the suffix's original branch predicates.
        for (auto site : prefix) {
            firstCopy[site] = q.sites.size();
            added.push_back(q.sites.size());
            firstPrefix.push_back(q.sites.size());
            q.sites.push_back(old.sites[site]);
            q.sites.back().backedgeOwners.clear();
            observe(site, firstCopy.at(site), true, false);
        }
        for (auto site : prefix)
            for (auto& next : q.sites[firstCopy.at(site)].successors)
                if (firstCopy.count(next))
                    next = firstCopy.at(next);
        q.sites[r.owner].successors = {firstCopy.at(body)};
        q.sites[header].successors = {body, finalCopy.at(body)};
        q.sites[header].backedgeOwners.clear();
    }
    for (auto& loop : q.loops) {
        if (loop.owner != r.owner && std::find(loop.sites.begin(), loop.sites.end(), r.owner) == loop.sites.end())
            continue;
        loop.sites.insert(loop.sites.end(), added.begin(), added.end());
        if (loop.owner == r.owner) {
            if (std::find(loop.sites.begin(), loop.sites.end(), header) == loop.sites.end())
                loop.sites.push_back(header);
            loop.bodyEntry = firstPrefix.front();
            loop.firstVisitPrefix = firstPrefix;
            loop.lastVisitDistance = r.step;
        }
    }
    q.qualification += "; joint-reader-prefix-v1";
    for (std::size_t i = 0; i < input.operations.size(); ++i)
        out.originalPhases.push_back(i);
    const auto checked = validateProgram(out.program);
    out.success = checked.success;
    out.reason = checked.reason;
    return out;
}
} // namespace mlir::pto::oahs
#endif
