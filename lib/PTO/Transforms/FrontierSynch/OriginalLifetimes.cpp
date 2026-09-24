// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#include "PTO/Transforms/FrontierSynch/OriginalLifetimes.h"
#include "OriginalIntervals.h"
#include <algorithm>
#include <deque>
#include <limits>
#include <map>
#include <set>
#include <tuple>

namespace mlir::pto::frontiersynch {
namespace {
// The donor's four marginal provenance matrices are retained, with their rows
// indexed by original control site. Partial writes do not kill older origins.
struct Matrix {
    std::size_t width = 0;
    bool valid = true;
    std::vector<uint64_t> words;
    Matrix() = default;
    Matrix(std::size_t sites, std::size_t origins) : width(origins / 64 + (origins % 64 != 0))
    {
        if (width && sites > words.max_size() / width) {
            valid = false;
            return;
        }
        words.resize(sites * width);
    }
    std::vector<uint64_t> row(std::size_t site) const
    {
        const auto begin = words.begin() + site * width;
        return {begin, begin + width};
    }
    bool merge(std::size_t site, const std::vector<uint64_t>& row)
    {
        bool changed = false;
        for (std::size_t i = 0; i < width; ++i) {
            auto& word = words[site * width + i];
            const auto joined = word | row[i];
            changed |= joined != word;
            word = joined;
        }
        return changed;
    }
    bool test(std::size_t site, std::size_t bit) const { return (words[site * width + bit / 64] >> (bit % 64)) & 1; }
};
void collect(const Region& region, std::vector<std::size_t>& operations)
{
    if (region.kind == Region::Operation) {
        operations.push_back(region.operation);
    }
    for (const auto& child : region.children) {
        collect(child, operations);
    }
}
const Region* findOwner(const Region& region, std::size_t owner)
{
    if (region.originalOwner == owner) {
        return &region;
    }
    for (const auto& child : region.children) {
        if (const auto* found = findOwner(child, owner)) {
            return found;
        }
    }
    return nullptr;
}
void markRepeatingRegions(const Region& region, std::vector<bool>& repeated, bool repeating = false)
{
    repeating |= region.kind == Region::For || region.kind == Region::While;
    if (region.kind == Region::Operation && region.operation < repeated.size()) {
        repeated[region.operation] = repeating;
    }
    for (const auto& child : region.children) {
        markRepeatingRegions(child, repeated, repeating);
    }
}
void indexOwnerRegions(const Region& region, std::map<std::size_t, const Region*>& owners)
{
    if (region.kind != Region::Operation && region.originalOwner != NoControlId) {
        owners.emplace(region.originalOwner, &region);
    }
    for (const auto& child : region.children) {
        indexOwnerRegions(child, owners);
    }
}
} // namespace

struct OriginalLifetimes::Impl {
    const OriginalStructure& original;
    OriginalProgramVersion version;
    const OriginalValueQueries* values;
    detail::ControlGraph graph;
    std::vector<bool> reachable;
    std::vector<bool> repeatedOperation;
    std::vector<std::vector<std::size_t>> predecessors;
    struct CellInfo {
        std::vector<StorageOrigin> origins;
        std::vector<std::size_t> originAt;
        std::vector<Access> accessAt;
        Matrix previousWriters, previousReaders, nextWriters, nextReaders;
        std::vector<bool> noFullWriter;
    };
    std::vector<CellInfo> cells;
    FactoredUseService factoredUses;
    using EffectKey = std::tuple<std::size_t, std::size_t, bool>;
    std::map<EffectKey, std::vector<std::size_t>> effectsByRole;
    std::vector<std::vector<OriginalRequirement>> byDeadline;
    std::vector<std::vector<SourceSubscription>> bySource;
    mutable std::map<std::size_t, std::vector<bool>> ownerMembers;
    std::map<std::size_t, const Region*> ownerRegions;
    mutable std::map<OriginalInterval, OriginalMayAfter> continuations;
    mutable std::map<std::pair<OriginalInterval, bool>, PhysicalUseFrontier> intervalFrontiers;
    mutable OriginalLifetimeStats work;
    bool ready = false;
    bool requirementsBuilt = false;
    std::string error;

    explicit Impl(const OriginalStructure& value, const OriginalValueQueries* values)
        : original(value),
          version(value.version),
          values(values),
          graph(detail::buildControlGraph(value)),
          factoredUses(value, values)
    {
        if (!graph.valid) {
            error = graph.reason;
            return;
        }
        reachable = detail::reachableSites(graph);
        indexOwnerRegions(value.body, ownerRegions);
        repeatedOperation.resize(value.operations.size());
        markRepeatingRegions(value.body, repeatedOperation);
        predecessors.resize(graph.sites.size());
        byDeadline.resize(value.operations.size());
        bySource.resize(value.operations.size());
        work.originalSites = graph.sites.size();
        for (std::size_t site = 0; site < graph.sites.size(); ++site) {
            for (auto next : graph.sites[site].successors) {
                predecessors[next].push_back(site);
            }
        }
        cells.resize(value.cells.size());
        for (std::size_t operation = 0; operation < value.operations.size(); ++operation) {
            for (std::size_t incidence = 0; incidence < value.operations[operation].accesses.size(); ++incidence) {
                const auto& effect = value.operations[operation].accesses[incidence];
                if (effect.read) {
                    effectsByRole[{operation, effect.cell, false}].push_back(incidence);
                }
                if (effect.write) {
                    effectsByRole[{operation, effect.cell, true}].push_back(incidence);
                }
            }
        }
        for (std::size_t cell = 0; cell < cells.size(); ++cell) {
            auto& info = cells[cell];
            info.accessAt.resize(graph.sites.size());
            info.originAt.assign(graph.sites.size(), NoControlId);
            for (std::size_t site = 0; site < value.operations.size(); ++site) {
                if (!reachable[site]) {
                    continue;
                }
                auto& access = info.accessAt[site];
                access.cell = cell;
                for (const auto& effect : value.operations[site].accesses) {
                    if (effect.cell == cell) {
                        access.read |= effect.read;
                        access.write |= effect.write;
                        access.definiteWrite |= effect.write && effect.definiteWrite;
                    }
                }
                if (access.read || access.write) {
                    info.originAt[site] = info.origins.size();
                    info.origins.push_back({site, site});
                }
            }
            work.accessIncidences += info.origins.size();
            const auto sites = graph.sites.size(), count = info.origins.size();
            info.previousWriters = Matrix(sites, count);
            info.previousReaders = Matrix(sites, count);
            info.nextWriters = Matrix(sites, count);
            info.nextReaders = Matrix(sites, count);
            if (!info.previousWriters.valid || !info.previousReaders.valid || !info.nextWriters.valid ||
                !info.nextReaders.valid) {
                error = "original storage provenance exceeds representable matrix size";
                return;
            }
            work.storageWords += 4 * info.previousWriters.words.size();
            solve(info, false);
            solve(info, true);
            computeUninitialized(info);
        }
        ready = true;
    }
    void solve(CellInfo& info, bool backward)
    {
        auto& writers = backward ? info.nextWriters : info.previousWriters;
        auto& readers = backward ? info.nextReaders : info.previousReaders;
        std::deque<std::size_t> queue;
        std::vector<bool> queued = reachable;
        for (std::size_t site = 0; site < reachable.size(); ++site) {
            if (reachable[site]) {
                queue.push_back(site);
            }
        }
        while (!queue.empty()) {
            const auto site = queue.front();
            queue.pop_front();
            queued[site] = false;
            if (backward) {
                ++work.backwardEvaluations;
            } else {
                ++work.forwardEvaluations;
            }
            auto ws = writers.row(site), rs = readers.row(site);
            const auto& access = info.accessAt[site];
            if (access.write && access.definiteWrite) {
                std::fill(ws.begin(), ws.end(), 0);
                std::fill(rs.begin(), rs.end(), 0);
            }
            const auto bit = info.originAt[site];
            if (bit != NoControlId) {
                if (access.write) {
                    ws[bit / 64] |= uint64_t(1) << (bit % 64);
                }
                // RMW has already queried both old modes. After a definite
                // overwrite it is a new writer, not a pure reader of the old
                // generation. In reverse order its read precedes the overwrite, so
                // retain it. A possible/partial RMW must retain both modes.
                if (access.read && (backward || !access.definiteWrite)) {
                    rs[bit / 64] |= uint64_t(1) << (bit % 64);
                }
            }
            const auto& next = backward ? predecessors[site] : graph.sites[site].successors;
            for (auto target : next) {
                if (!reachable[target]) {
                    continue;
                }
                const bool changed = writers.merge(target, ws) | readers.merge(target, rs);
                if (changed && !queued[target]) {
                    queued[target] = true;
                    queue.push_back(target);
                }
            }
        }
    }
    void computeUninitialized(CellInfo& info)
    {
        info.noFullWriter.assign(graph.sites.size(), false);
        std::vector<std::size_t> pending{graph.entry};
        info.noFullWriter[graph.entry] = true;
        while (!pending.empty()) {
            const auto site = pending.back();
            pending.pop_back();
            if (info.accessAt[site].definiteWrite) {
                continue;
            }
            for (auto next : graph.sites[site].successors) {
                if (!info.noFullWriter[next]) {
                    info.noFullWriter[next] = true;
                    pending.push_back(next);
                }
            }
        }
    }
    std::vector<StorageOrigin> originsAt(const Matrix& matrix, const CellInfo& info, std::size_t site) const
    {
        std::vector<StorageOrigin> result;
        for (std::size_t bit = 0; bit < info.origins.size(); ++bit) {
            if (matrix.test(site, bit)) {
                result.push_back(info.origins[bit]);
            }
        }
        return result;
    }
    void ensureRequirements()
    {
        if (ready && !requirementsBuilt) {
            buildRequirements();
            requirementsBuilt = true;
        }
    }
    void buildRequirements()
    {
        for (std::size_t target = 0; target < original.operations.size(); ++target) {
            if (!reachable[target]) {
                continue;
            }
            std::set<std::pair<std::size_t, StorageRelationship::Kind>> demands;
            for (const auto& access : original.operations[target].accesses) {
                if (access.read) {
                    demands.emplace(access.cell, StorageRelationship::RAW);
                }
                if (access.write) {
                    demands.emplace(access.cell, StorageRelationship::WAR);
                    demands.emplace(access.cell, StorageRelationship::WAW);
                }
            }
            for (const auto& key : demands) {
                const auto [cell, kind] = key;
                const auto& info = cells[cell];
                const bool sourceWrites = kind != StorageRelationship::WAR;
                const bool targetWrites = kind != StorageRelationship::RAW;
                const auto& matrix = kind == StorageRelationship::WAR ? info.previousReaders : info.previousWriters;
                const auto& targetEffects = effectsByRole.at({target, cell, targetWrites});
                for (const auto& source : originsAt(matrix, info, target)) {
                    OriginalRequirement request;
                    request.relationship = {kind, cell, source, {target, target}};
                    request.source = {source.operation, SourceMilestone::After};
                    request.deadline = {target, SourceMilestone::Before};
                    request.sourceEffects = &effectsByRole.at({source.operation, cell, sourceWrites});
                    request.targetEffects = &targetEffects;
                    request.sourceEngine = original.operations[source.operation].instruction->kPipeValue;
                    request.targetEngine = original.operations[target].instruction->kPipeValue;
                    request.sourceGapExecutable = original.operations[source.operation].afterExecutable;
                    request.deadlineGapExecutable = original.operations[target].beforeExecutable;
                    // A marginal origin is not an occurrence or episode certificate.
                    const auto index = byDeadline[target].size();
                    const auto executableSource = original.operations[source.operation].enclosingAfter;
                    const bool validSource = executableSource < original.operations.size();
                    request.sourceSubscribed = validSource;
                    if (validSource) {
                        SourceSubscription subscription;
                        subscription.position = {executableSource, SourceMilestone::After};
                        subscription.sufficientPosition = request.source;
                        subscription.deadlineOperation = target;
                        subscription.cell = cell;
                        subscription.kind = kind;
                        subscription.requirementIndex = index;
                        subscription.executableInOriginalIR = true;
                        bySource[executableSource].push_back(std::move(subscription));
                    }
                    byDeadline[target].push_back(request);
                    ++work.requirements;
                }
            }
        }
    }
    bool mayExitWithoutAccess(std::size_t site, std::size_t cell) const
    {
        std::vector<bool> seen(graph.sites.size());
        std::vector<std::size_t> pending = graph.sites[site].successors;
        while (!pending.empty()) {
            const auto at = pending.back();
            pending.pop_back();
            if (seen[at]) {
                continue;
            }
            seen[at] = true;
            if (at == graph.exit) {
                return true;
            }
            const auto& access = cells[cell].accessAt[at];
            if (access.read || access.write) {
                continue;
            }
            pending.insert(pending.end(), graph.sites[at].successors.begin(), graph.sites[at].successors.end());
        }
        return false;
    }
    PhysicalUseFrontier frontier(const OriginalUseQuery& query, bool backward) const
    {
        PhysicalUseFrontier result;
        ++work.frontierQueries;
        if (query.version && query.version != version) {
            result.reason = "original-program version mismatch";
            return result;
        }
        const bool invalidCell = query.cell >= cells.size();
        const bool invalidOwner = query.owner != NoControlId && !ownerRegions.count(query.owner);
        if (invalidCell || invalidOwner) {
            result.reason = "unresolved original owner or cell";
            return result;
        }
        const auto owner = query.owner == NoControlId ? &original.body : ownerRegions.at(query.owner);
        auto membership = ownerMembers.find(query.owner);
        if (membership == ownerMembers.end()) {
            std::vector<std::size_t> operations;
            collect(*owner, operations);
            std::vector<bool> members(original.operations.size());
            for (auto operation : operations) {
                members[operation] = true;
            }
            membership = ownerMembers.emplace(query.owner, std::move(members)).first;
        }
        const auto& member = membership->second;
        if (query.starts.empty() || (!query.read && !query.write)) {
            result.reason = "missing continuation start or access class";
            return result;
        }
        for (auto site : query.starts) {
            const bool invalidSite = site >= graph.sites.size() || !reachable[site];
            const bool outsideOwner = site < original.operations.size() && !member[site];
            if (invalidSite || outsideOwner) {
                result.reason = "invalid original interval start";
                return result;
            }
        }
        for (auto site : query.stops) {
            const bool invalidSite = site >= graph.sites.size();
            const bool outsideOwner = site < original.operations.size() && !member[site];
            if (invalidSite || outsideOwner) {
                result.reason = "invalid original interval stop";
                return result;
            }
        }
        const std::set<std::size_t> stops(query.stops.begin(), query.stops.end());
        std::vector<bool> seen(graph.sites.size());
        std::vector<std::size_t> pending = query.starts;
        bool leftOwner = false;
        while (!pending.empty()) {
            const auto site = pending.back();
            pending.pop_back();
            if (seen[site]) {
                continue;
            }
            seen[site] = true;
            ++work.frontierSites;
            const bool outsideOwner = site < original.operations.size() && !member[site];
            if (outsideOwner) {
                leftOwner = true;
                result.reachesBoundary = true;
                continue;
            }
            if (stops.count(site) && !query.includeStops) {
                result.reachesBoundary = true;
                continue;
            }
            const auto& access = cells[query.cell].accessAt[site];
            if ((query.read && access.read) || (query.write && access.write)) {
                result.accesses.push_back({site, site});
                continue;
            }
            if (query.stopAtOtherAccess && (access.read || access.write)) {
                result.reachesBoundary = true;
                continue;
            }
            if (stops.count(site) || site == (backward ? graph.entry : graph.exit)) {
                result.reachesBoundary = true;
                continue;
            }
            const auto& next = backward ? predecessors[site] : graph.sites[site].successors;
            pending.insert(pending.end(), next.begin(), next.end());
        }
        result.status = leftOwner ? PhysicalUseFrontier::Status::Unknown :
                                    (result.accesses.empty() ? PhysicalUseFrontier::Status::NoHit :
                                                               PhysicalUseFrontier::Status::Present);
        if (leftOwner) {
            result.reason = "original use interval leaves its qualified owner";
        }
        std::sort(result.accesses.begin(), result.accesses.end(), [](const auto& a, const auto& b) {
            return a.site < b.site;
        });
        for (const auto& access : result.accesses) {
            const OriginalCut cut{access.operation, backward ? OriginalCut::After : OriginalCut::Before};
            if (!resolveOriginalCut(original, cut)) {
                result.status = PhysicalUseFrontier::Status::Unknown;
                result.reason = "physical frontier has no executable original cut";
                result.cuts.clear();
                return result;
            }
            result.cuts.push_back(cut);
        }
        return result;
    }
    OriginalIntervalResult prepare(OriginalIntervalRequest request) const
    {
        OriginalIntervalResult result;
        if (original.version != version) {
            result.reason = "original program changed; rebuild Phase A";
            return result;
        }
        if (request.selector.cell >= cells.size() ||
            (request.selector.engine && *request.selector.engine >= unsigned(PipelineType::PIPE_NUM)) ||
            (request.selector.physicalRelation != NoControlId &&
             request.selector.physicalRelation >= original.physicalAddresses.size())) {
            result.reason = "invalid original physical selector";
            return result;
        }
        return detail::prepareOriginalInterval(graph, version, std::move(request));
    }
    bool matches(std::size_t operation, const OriginalAccessSelector& selector) const
    {
        const auto& payload = original.operations[operation];
        if (selector.engine && unsigned(payload.instruction->kPipeValue) != *selector.engine) {
            return false;
        }
        for (const auto& effect : payload.accesses) {
            if (effect.cell == selector.cell &&
                (selector.physicalRelation == NoControlId || effect.physicalRelation == selector.physicalRelation) &&
                ((selector.read && effect.read) || (selector.write && effect.write))) {
                return true;
            }
        }
        return false;
    }
    PhysicalUseFrontier frontier(const OriginalInterval& interval, bool backward) const
    {
        PhysicalUseFrontier result;
        result.interval = interval;
        const auto prepared = prepare(interval.query);
        if (!prepared.valid || prepared.interval != interval) {
            result.reason = prepared.valid ? "inconsistent original interval owner" : prepared.reason;
            return result;
        }
        const auto key = std::make_pair(interval, backward);
        const auto prior = intervalFrontiers.find(key);
        if (prior != intervalFrontiers.end()) {
            ++work.intervalCacheHits;
            return prior->second;
        }
        const auto walk = detail::walkOriginalInterval(
            graph, interval, [&](std::size_t operation) { return matches(operation, interval.query.selector); },
            backward ? detail::IntervalSelection::Last : detail::IntervalSelection::First);
        work.frontierSites += walk.visitedSites;
        result.cases = walk.cases;
        result.reachesBoundary = walk.noHitPath;
        for (auto operation : walk.operations) {
            result.accesses.push_back({operation, operation});
        }
        if (!walk.complete || (walk.operations.empty() && walk.unresolvedOccurrence)) {
            result.reason = walk.complete ? "unqualified repeated stop interpretation" : walk.reason;
        } else {
            result.status =
                walk.operations.empty() ? PhysicalUseFrontier::Status::NoHit : PhysicalUseFrontier::Status::Present;
            for (auto operation : walk.operations) {
                const OriginalCut cut{operation, backward ? OriginalCut::After : OriginalCut::Before};
                if (!resolveOriginalCut(original, cut)) {
                    result.status = PhysicalUseFrontier::Status::Unknown;
                    result.reason = "physical frontier has no executable original cut";
                    result.cuts.clear();
                    break;
                }
                result.cuts.push_back(cut);
            }
        }
        return intervalFrontiers.emplace(key, std::move(result)).first->second;
    }
    OriginalMayAfter mayAfter(const OriginalInterval& interval) const
    {
        OriginalMayAfter result;
        result.interval = interval;
        const auto prepared = prepare(interval.query);
        if (!prepared.valid || prepared.interval != interval) {
            result.reason = prepared.valid ? "inconsistent original interval owner" : prepared.reason;
            return result;
        }
        const auto prior = continuations.find(interval);
        if (prior != continuations.end()) {
            ++work.intervalCacheHits;
            return prior->second;
        }
        const auto walk = detail::walkOriginalInterval(
            graph, interval, [&](std::size_t operation) { return matches(operation, interval.query.selector); });
        work.frontierSites += walk.visitedSites;
        result.cases = walk.cases;
        result.mayBypassStop = walk.cases.reachedOwnerExit;
        for (auto operation : walk.operations) {
            result.witnesses.push_back({operation, operation});
        }
        if (!walk.complete) {
            result.reason = walk.reason;
        } else if (!result.witnesses.empty()) {
            result.status = OriginalMayAfter::Status::May;
        } else if (walk.unresolvedOccurrence) {
            result.reason = "continuation has an unqualified repeated stop interpretation";
        } else {
            result.status = OriginalMayAfter::Status::NoHit;
        }
        return continuations.emplace(interval, std::move(result)).first->second;
    }
};

OriginalLifetimes::OriginalLifetimes(const OriginalStructure& original, const OriginalValueQueries* values)
    : impl(std::make_unique<Impl>(original, values))
{}
OriginalLifetimes::~OriginalLifetimes() = default;
bool OriginalLifetimes::complete() const { return impl->ready && impl->original.version == impl->version; }
const std::string& OriginalLifetimes::reason() const
{
    static const std::string changed = "original program changed; rebuild Phase A";
    return impl->original.version != impl->version ? changed : impl->error;
}
const OriginalLifetimeStats& OriginalLifetimes::stats() const { return impl->work; }
const FactoredUseResult& OriginalLifetimes::factored(std::size_t cell) const
{
    if (!complete()) {
        static const FactoredUseResult invalid;
        return invalid;
    }
    return impl->factoredUses.root(cell);
}
const FactoredUseResult& OriginalLifetimes::factoredAt(std::size_t operation, std::size_t cell) const
{
    if (!complete()) {
        static const FactoredUseResult invalid;
        return invalid;
    }
    return impl->factoredUses.at(operation, cell);
}
FactoredUseResult OriginalLifetimes::applyFactoredAt(
    std::size_t operation, std::size_t cell, FactoredUseInterface boundary) const
{
    if (!complete()) {
        static const FactoredUseResult invalid;
        return invalid;
    }
    return impl->factoredUses.applyAt(operation, cell, std::move(boundary));
}
const std::vector<OriginalRequirement>& OriginalLifetimes::requirementsAt(std::size_t operation) const
{
    impl->ensureRequirements();
    static const std::vector<OriginalRequirement> empty;
    return complete() && operation < impl->byDeadline.size() ? impl->byDeadline[operation] : empty;
}
const std::vector<SourceSubscription>& OriginalLifetimes::subscriptionsAt(std::size_t operation) const
{
    impl->ensureRequirements();
    static const std::vector<SourceSubscription> empty;
    return complete() && operation < impl->bySource.size() ? impl->bySource[operation] : empty;
}
std::optional<bool> OriginalLifetimes::mayOriginAt(
    std::size_t operation, std::size_t cell, std::size_t source, bool reader, bool incoming) const
{
    if (!impl->ready || operation >= impl->original.operations.size() || cell >= impl->cells.size()) {
        return std::nullopt;
    }
    if (!impl->reachable[operation]) {
        return false;
    }
    const auto& info = impl->cells[cell];
    if (incoming) {
        // The baseline has no supplied entry histories. Retain a possible
        // incoming reader/writer until a definite write cuts every such path.
        // This says nothing about acquired completion at that write.
        return info.noFullWriter[operation];
    }
    if (source >= impl->original.operations.size()) {
        return false;
    }
    const auto bit = info.originAt[source];
    if (bit == NoControlId) {
        return false;
    }
    const auto& matrix = reader ? info.previousReaders : info.previousWriters;
    return matrix.test(operation, bit);
}
std::optional<bool> OriginalLifetimes::hasMayOriginsAt(std::size_t operation, std::size_t cell, bool reader) const
{
    if (!impl->ready || operation >= impl->original.operations.size() || cell >= impl->cells.size()) {
        return std::nullopt;
    }
    if (!impl->reachable[operation]) {
        return false;
    }
    const auto& info = impl->cells[cell];
    if (info.noFullWriter[operation]) {
        return true;
    }
    const auto& matrix = reader ? info.previousReaders : info.previousWriters;
    for (std::size_t word = 0; word < matrix.width; ++word) {
        if (matrix.words[operation * matrix.width + word]) {
            return true;
        }
    }
    return false;
}
std::vector<StorageOrigin> OriginalLifetimes::mayOriginsAt(std::size_t operation, std::size_t cell, bool reader) const
{
    if (!impl->ready || operation >= impl->original.operations.size() || cell >= impl->cells.size() ||
        !impl->reachable[operation]) {
        return {};
    }
    const auto& info = impl->cells[cell];
    return impl->originsAt(reader ? info.previousReaders : info.previousWriters, info, operation);
}
const std::vector<std::size_t>& OriginalLifetimes::effectIncidences(
    std::size_t operation, std::size_t cell, bool write) const
{
    static const std::vector<std::size_t> empty;
    const auto found = impl->effectsByRole.find({operation, cell, write});
    return found == impl->effectsByRole.end() ? empty : found->second;
}
StorageLifecycle OriginalLifetimes::lifecycleAt(std::size_t operation, std::size_t cell) const
{
    StorageLifecycle result;
    result.cell = cell;
    if (!complete() || operation >= impl->original.operations.size() || cell >= impl->cells.size() ||
        !impl->reachable[operation]) {
        return result;
    }
    const auto& info = impl->cells[cell];
    result.reachable = true;
    result.access = {operation, operation};
    result.previousWriters = impl->originsAt(info.previousWriters, info, operation);
    result.previousReaders = impl->originsAt(info.previousReaders, info, operation);
    result.nextWriters = impl->originsAt(info.nextWriters, info, operation);
    result.nextReaders = impl->originsAt(info.nextReaders, info, operation);
    result.mayHaveNoPriorFullWrite = info.noFullWriter[operation];
    result.mayExitWithoutFurtherAccess = impl->mayExitWithoutAccess(operation, cell);
    return result;
}
PhysicalUseFrontier OriginalLifetimes::firstUse(const OriginalUseQuery& query) const
{
    if (!complete()) {
        return {};
    }
    return impl->frontier(query, false);
}
PhysicalUseFrontier OriginalLifetimes::lastUse(const OriginalUseQuery& query) const
{
    if (!complete()) {
        return {};
    }
    return impl->frontier(query, true);
}
PhysicalUseFrontier OriginalLifetimes::firstUse(const OriginalInterval& interval) const
{
    if (!complete()) {
        return {};
    }
    ++impl->work.frontierQueries;
    return impl->frontier(interval, false);
}
PhysicalUseFrontier OriginalLifetimes::lastUse(const OriginalInterval& interval) const
{
    if (!complete()) {
        return {};
    }
    ++impl->work.frontierQueries;
    return impl->frontier(interval, true);
}
OriginalSupportInterval OriginalLifetimes::supportBetween(
    std::size_t producer, std::size_t reuse, std::size_t cell) const
{
    OriginalIntervalRequest request;
    request.version = impl->version;
    request.selector.cell = cell;
    request.occurrence.source = producer;
    request.occurrence.target = reuse;
    request.start = {producer, OriginalCut::After};
    request.stop = {reuse, OriginalCut::Before};
    const auto prepared = prepareInterval(std::move(request));
    if (!prepared.valid) {
        OriginalSupportInterval result;
        result.reason = prepared.reason;
        return result;
    }
    return supportBetween(prepared.interval);
}
OriginalSupportInterval OriginalLifetimes::supportBetween(const OriginalInterval& interval) const
{
    OriginalSupportInterval result;
    result.interval = interval;
    const auto& query = interval.query;
    const auto producer = query.occurrence.source, reuse = query.occurrence.target;
    const auto cell = query.selector.cell;
    result.producer = producer;
    result.reuse = reuse;
    result.cell = cell;
    const auto prepared = prepareInterval(query);
    if (!prepared.valid || prepared.interval != interval) {
        result.reason = prepared.valid ? "inconsistent original support owner" : prepared.reason;
        return result;
    }
    if (producer >= impl->original.operations.size() || reuse >= impl->original.operations.size() ||
        query.start != OriginalCut{producer, OriginalCut::After} ||
        query.stop != OriginalCut{reuse, OriginalCut::Before} || query.includeStoppingAccess || !query.selector.read ||
        !query.selector.write || query.selector.engine || query.selector.physicalRelation != NoControlId ||
        !impl->cells[cell].accessAt[producer].write || !impl->cells[cell].accessAt[reuse].write) {
        result.reason = "support interval lacks complete producer/reuse write roles";
        return result;
    }
    impl->ensureRequirements();
    const auto& graph = impl->graph;
    const auto walk = detail::walkOriginalInterval(graph, interval, [](std::size_t) { return true; });
    if (!walk.complete) {
        result.reason = walk.reason;
        return result;
    }
    result.cases = walk.cases;
    result.reachesReuse = walk.cases.reachedStop;
    result.mayBypass = walk.cases.reachedOwnerExit;
    result.mayReenter = std::find(walk.operations.begin(), walk.operations.end(), producer) != walk.operations.end();
    std::set<std::size_t> affected{producer};
    for (auto site : walk.operations) {
        const auto& access = impl->cells[cell].accessAt[site];
        if (access.read) {
            result.readers.push_back({site, site});
        }
        result.mayReload |= access.write && site != producer;
        affected.insert(site);
    }
    if (result.reachesReuse) {
        affected.insert(reuse);
    }
    for (auto site : affected) {
        result.affectedRequirements.insert(
            result.affectedRequirements.end(), impl->byDeadline[site].begin(), impl->byDeadline[site].end());
    }
    if (!result.reachesReuse) {
        result.reason = "reuse is not reachable in this original continuation";
        return result;
    }
    result.complete = true;
    // Retain the independent incoming-path check. A local interval does not
    // prove that every target visit has this producer, nor invent an origin.
    std::vector<bool> bypassesProducer(graph.sites.size());
    std::vector<std::size_t> incoming{graph.entry};
    while (!incoming.empty()) {
        const auto site = incoming.back();
        incoming.pop_back();
        if (site == producer || bypassesProducer[site]) {
            continue;
        }
        bypassesProducer[site] = true;
        incoming.insert(incoming.end(), graph.sites[site].successors.begin(), graph.sites[site].successors.end());
    }
    result.stablePhysicalInterval =
        !result.mayReload && !result.mayReenter && !result.mayBypass && !bypassesProducer[reuse];
    result.generationEstablished = result.stablePhysicalInterval && !walk.unresolvedOccurrence &&
                                   query.occurrence.incomingInterface == NoControlId &&
                                   impl->cells[cell].accessAt[producer].definiteWrite;
    return result;
}
OriginalAccessSummary OriginalLifetimes::all(std::size_t owner, std::size_t cell) const
{
    OriginalAccessSummary result;
    if (!complete() || cell >= impl->cells.size()) {
        return result;
    }
    const auto* region = owner == NoControlId ? &impl->original.body : findOwner(impl->original.body, owner);
    if (!region) {
        return result;
    }
    result.complete = true;
    std::vector<std::size_t> operations;
    collect(*region, operations);
    for (auto operation : operations) {
        if (operation < impl->original.operations.size() && impl->reachable[operation]) {
            const auto& access = impl->cells[cell].accessAt[operation];
            if (access.read) {
                result.readers.push_back({operation, operation});
            }
            if (access.write) {
                result.writers.push_back({operation, operation});
            }
        }
    }
    return result;
}
OriginalIntervalResult OriginalLifetimes::prepareInterval(OriginalIntervalRequest request) const
{
    if (!complete()) {
        OriginalIntervalResult result;
        result.reason = "original analysis is incomplete or its snapshot changed";
        return result;
    }
    return impl->prepare(std::move(request));
}
OriginalMayAfter OriginalLifetimes::mayAfter(const OriginalInterval& interval) const
{
    if (!complete()) {
        OriginalMayAfter result;
        result.reason = "original analysis is incomplete or its snapshot changed";
        return result;
    }
    ++impl->work.frontierQueries;
    return impl->mayAfter(interval);
}
OriginalMayAfter OriginalLifetimes::mayAfter(const OriginalContinuationQuery& query) const
{
    OriginalIntervalRequest request;
    request.version = impl->version;
    request.selector.cell = query.cell;
    request.selector.read = query.read;
    request.selector.write = query.write;
    request.start = query.start;
    request.stop = query.stop.operation == NoControlId && query.stop.kind == OriginalCut::Kind::Payload ?
                       OriginalCut::scope(query.owner, OriginalCut::After) :
                       query.stop;
    request.includeStoppingAccess =
        request.stop.kind == OriginalCut::Kind::Payload && request.stop.side == OriginalCut::After;
    request.continuationOwner = query.owner;
    request.occurrence.source = query.start.operation;
    request.occurrence.target = query.stop.operation;
    const auto prepared = prepareInterval(std::move(request));
    if (!prepared.valid) {
        OriginalMayAfter result;
        result.reason = prepared.reason;
        return result;
    }
    // The legacy owner was a hard containment premise, not permission to widen.
    if (prepared.interval.owner != query.owner) {
        OriginalMayAfter result;
        result.reason = "continuation endpoint lies outside its original owner";
        return result;
    }
    return mayAfter(prepared.interval);
}
} // namespace mlir::pto::frontiersynch
