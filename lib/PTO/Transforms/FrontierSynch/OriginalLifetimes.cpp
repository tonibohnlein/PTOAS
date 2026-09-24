// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#include "PTO/Transforms/FrontierSynch/OriginalLifetimes.h"
#include "Control.h"
#include <algorithm>
#include <deque>
#include <map>
#include <set>

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
} // namespace

struct OriginalLifetimes::Impl {
    const OriginalStructure& original;
    detail::ControlGraph graph;
    std::vector<bool> reachable;
    std::vector<std::vector<std::size_t>> predecessors;
    struct CellInfo {
        std::vector<StorageOrigin> origins;
        std::vector<std::size_t> originAt;
        std::vector<Access> accessAt;
        Matrix previousWriters, previousReaders, nextWriters, nextReaders;
        std::vector<bool> noFullWriter;
    };
    std::vector<CellInfo> cells;
    std::vector<std::vector<OriginalRequirement>> byDeadline;
    std::vector<std::vector<SourceSubscription>> bySource;
    mutable OriginalLifetimeStats work;
    bool ready = false;
    std::string error;

    explicit Impl(const OriginalStructure& value) : original(value), graph(detail::buildControlGraph(value))
    {
        reachable = detail::reachableSites(graph);
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
        buildRequirements();
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
                if (access.read) {
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
    void buildRequirements()
    {
        for (std::size_t target = 0; target < original.operations.size(); ++target) {
            if (!reachable[target]) {
                continue;
            }
            for (const auto& access : original.operations[target].accesses) {
                const auto& info = cells[access.cell];
                auto append = [&](StorageRelationship::Kind kind, const Matrix& matrix) {
                    for (const auto& source : originsAt(matrix, info, target)) {
                        OriginalRequirement request;
                        request.relationship = {kind, access.cell, source, {target, target}};
                        request.source = {source.operation, SourceMilestone::After};
                        request.deadline = {target, SourceMilestone::Before};
                        request.sourceSubscribed = true;
                        // A marginal origin is not an occurrence or episode certificate.
                        bySource[source.operation].push_back({request.source, target, access.cell, kind});
                        byDeadline[target].push_back(request);
                        ++work.requirements;
                    }
                };
                if (access.read) {
                    append(StorageRelationship::RAW, info.previousWriters);
                }
                if (access.write) {
                    append(StorageRelationship::WAR, info.previousReaders);
                    append(StorageRelationship::WAW, info.previousWriters);
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
        if (query.cell >= cells.size() || (query.owner != NoControlId && !findOwner(original.body, query.owner))) {
            result.reason = "unresolved original owner or cell";
            return result;
        }
        if (query.starts.empty() || (!query.read && !query.write)) {
            result.reason = "missing continuation start or access class";
            return result;
        }
        for (auto site : query.starts) {
            if (site >= graph.sites.size() || !reachable[site]) {
                result.reason = "invalid original interval start";
                return result;
            }
        }
        for (auto site : query.stops) {
            if (site >= graph.sites.size()) {
                result.reason = "invalid original interval stop";
                return result;
            }
        }
        const std::set<std::size_t> stops(query.stops.begin(), query.stops.end());
        std::vector<bool> seen(graph.sites.size());
        std::vector<std::size_t> pending = query.starts;
        while (!pending.empty()) {
            const auto site = pending.back();
            pending.pop_back();
            if (seen[site]) {
                continue;
            }
            seen[site] = true;
            ++work.frontierSites;
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
        result.status =
            result.accesses.empty() ? PhysicalUseFrontier::Status::NoHit : PhysicalUseFrontier::Status::Present;
        std::sort(result.accesses.begin(), result.accesses.end(), [](const auto& a, const auto& b) {
            return a.site < b.site;
        });
        return result;
    }
};

OriginalLifetimes::OriginalLifetimes(const OriginalStructure& original) : impl(std::make_unique<Impl>(original)) {}
OriginalLifetimes::~OriginalLifetimes() = default;
bool OriginalLifetimes::complete() const { return impl->ready; }
const std::string& OriginalLifetimes::reason() const { return impl->error; }
const OriginalLifetimeStats& OriginalLifetimes::stats() const { return impl->work; }
const std::vector<OriginalRequirement>& OriginalLifetimes::requirementsAt(std::size_t operation) const
{
    static const std::vector<OriginalRequirement> empty;
    return impl->ready && operation < impl->byDeadline.size() ? impl->byDeadline[operation] : empty;
}
const std::vector<SourceSubscription>& OriginalLifetimes::subscriptionsAt(std::size_t operation) const
{
    static const std::vector<SourceSubscription> empty;
    return impl->ready && operation < impl->bySource.size() ? impl->bySource[operation] : empty;
}
StorageLifecycle OriginalLifetimes::lifecycleAt(std::size_t operation, std::size_t cell) const
{
    StorageLifecycle result;
    result.cell = cell;
    if (!impl->ready || operation >= impl->original.operations.size() || cell >= impl->cells.size() ||
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
    if (!impl->ready) {
        return {};
    }
    return impl->frontier(query, false);
}
PhysicalUseFrontier OriginalLifetimes::lastUse(const OriginalUseQuery& query) const
{
    if (!impl->ready) {
        return {};
    }
    return impl->frontier(query, true);
}
OriginalSupportInterval OriginalLifetimes::supportBetween(
    std::size_t producer, std::size_t reuse, std::size_t cell) const
{
    OriginalSupportInterval result;
    result.producer = producer;
    result.reuse = reuse;
    result.cell = cell;
    if (!impl->ready || cell >= impl->cells.size() || producer >= impl->original.operations.size() ||
        reuse >= impl->original.operations.size() || !impl->reachable[producer] || !impl->reachable[reuse] ||
        !impl->cells[cell].accessAt[producer].write || !impl->cells[cell].accessAt[reuse].write) {
        result.reason = "support interval lacks represented producer and reuse writes";
        return result;
    }
    const auto& graph = impl->graph;
    result.affectedRequirements.insert(
        result.affectedRequirements.end(), impl->byDeadline[producer].begin(), impl->byDeadline[producer].end());
    std::vector<bool> seen(graph.sites.size());
    std::vector<std::size_t> pending = graph.sites[producer].successors;
    while (!pending.empty()) {
        const auto site = pending.back();
        pending.pop_back();
        if (seen[site]) {
            continue;
        }
        seen[site] = true;
        if (site == reuse) {
            result.reachesReuse = true;
            result.affectedRequirements.insert(
                result.affectedRequirements.end(), impl->byDeadline[site].begin(), impl->byDeadline[site].end());
            continue;
        }
        if (site == graph.exit) {
            result.mayBypass = true;
            continue;
        }
        if (site == producer) {
            result.mayReenter = true;
        }
        if (site < impl->original.operations.size()) {
            const auto& access = impl->cells[cell].accessAt[site];
            if (access.read) {
                result.readers.push_back({site, site});
            }
            result.mayReload |= access.write && site != producer;
            result.affectedRequirements.insert(
                result.affectedRequirements.end(), impl->byDeadline[site].begin(), impl->byDeadline[site].end());
        }
        pending.insert(pending.end(), graph.sites[site].successors.begin(), graph.sites[site].successors.end());
    }
    if (!result.reachesReuse) {
        result.reason = "reuse is not reachable from this original producer";
        return result;
    }
    result.complete = true;
    std::vector<bool> bypassesProducer(graph.sites.size());
    std::vector<std::size_t> incoming{graph.entry};
    while (!incoming.empty()) {
        const auto site = incoming.back();
        incoming.pop_back();
        if (site == producer || bypassesProducer[site]) {
            continue;
        }
        bypassesProducer[site] = true;
        if (site == reuse) {
            break;
        }
        incoming.insert(incoming.end(), graph.sites[site].successors.begin(), graph.sites[site].successors.end());
    }
    result.generationEstablished = impl->cells[cell].accessAt[producer].definiteWrite && !result.mayReload &&
                                   !result.mayReenter && !result.mayBypass && !bypassesProducer[reuse];
    return result;
}
OriginalAccessSummary OriginalLifetimes::all(std::size_t owner, std::size_t cell) const
{
    OriginalAccessSummary result;
    if (!impl->ready || cell >= impl->cells.size()) {
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
} // namespace mlir::pto::frontiersynch
