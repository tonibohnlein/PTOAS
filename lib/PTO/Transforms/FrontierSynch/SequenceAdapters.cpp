// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Typed explicit/periodic adapters to the shared regional composition contract.
#include "SequenceAnalysisInternal.h"
namespace mlir::pto::frontiersynch {
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
                auto prepared = prepareExplicitInsertion(function, current.explicitAnalysis);
                if (succeeded(prepared)) { (*prepared)->completeInvocation = false; }
                return prepared;
            }
            auto prepared = std::make_unique<PreparedLogicalPlan>(0);
            if (failed(prepareCountedEndpointCode(function, current.endpoints, *prepared))) { return failure(); }
            return prepared;
        };
        auto convert = [&](Selected selected) {
            const auto& port = ports[selected.port];
            return RegionalSelector{{port.type, port.ordinal, PeriodicEventKind::Start}, selected.present};
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
bool SequenceAnalysisState::importSummaries()
{
    std::map<AddressSpace, std::set<uint64_t>> points;
    for (const auto& child : children) {
        const auto& out = child.regional;
        if (out.expressions != arena || !out.capabilities.exactEffects || !out.capabilities.exactQueries ||
            !out.capabilities.exactSelectors || !out.capabilities.endpointRecipes || !out.presence ||
            !out.reachability || !out.prepare || out.anchors.size() != out.occurrenceLoops.size()) {
            return fail("sequence child lacks a constructed exact regional interface");
        }
        for (const auto& cell : out.storageBoundary) {
            if (cell.cell.begin > cell.cell.end) { return fail("regional storage boundary has an invalid range"); }
            points[cell.cell.space].insert(cell.cell.begin);
            points[cell.cell.space].insert(cell.cell.end);
        }
    }
    cells.clear(); ports.clear(); portIds.clear(); boundaries.clear();
    for (const auto& [space, endpoints] : points) {
        for (auto it = endpoints.begin(); it != endpoints.end() && std::next(it) != endpoints.end(); ++it) {
            SyncStorageCell cell{space, *it, *std::next(it)};
            bool covered = false;
            for (const auto& child : children) {
                for (const auto& local : child.regional.storageBoundary) {
                    covered |= local.cell.space == space && local.cell.begin <= cell.begin &&
                               local.cell.end >= cell.end;
                }
            }
            if (covered) { cells.push_back(cell); }
        }
    }
    boundaries.resize(children.size(), std::vector<CellBoundary>(cells.size()));
    for (uint32_t id = 0; id < children.size(); ++id) {
        auto& child = children[id];
        child.anchors = child.regional.anchors;
        auto convert = [&](RegionalSelector selector, std::vector<Selected>& destination) {
            if (selector.event.type >= child.anchors.size() || selector.event.ordinal >= expressions.size() ||
                expressions.isBoolean(selector.event.ordinal) || !expressions.isBoolean(selector.present)) {
                fail("regional selector has an invalid occurrence or predicate"); return;
            }
            destination.push_back({port(id, selector.event.type, selector.event.ordinal), selector.present});
        };
        for (std::size_t cell = 0; cell < cells.size(); ++cell) {
            auto& out = boundaries[id][cell];
            for (const auto& local : child.regional.storageBoundary) {
                if (local.cell.space != cells[cell].space || local.cell.begin > cells[cell].begin ||
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
        }
        for (const auto* side : {&child.regional.firstPayloads, &child.regional.lastPayloads}) {
            for (const auto& [pipe, values] : *side) {
                std::vector<Selected> validated;
                for (auto selected : values) { convert(selected, validated); }
            }
        }
    }
    return error.empty();
}
RegionalAnalysis sequenceRegionalResult(const SequenceAnalysis& analysis)
{
    RegionalAnalysis out;
    if (!analysis.state || !analysis.error.empty()) { return out; }
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
        }
    }
    out.presence = [state, coordinates](RegionalEvent event) -> std::optional<Expr> {
        if (event.type >= coordinates->size()) { return std::nullopt; }
        auto [child, type] = (*coordinates)[event.type];
        return state->children[child].regional.presence({type, event.ordinal, event.kind});
    };
    out.reachability = [owned, coordinates](RegionalEvent a, RegionalEvent b) -> std::optional<Expr> {
        if (a.type >= coordinates->size() || b.type >= coordinates->size()) { return std::nullopt; }
        auto [ac, at] = (*coordinates)[a.type]; auto [bc, bt] = (*coordinates)[b.type];
        return sequenceEventReachability(*owned, {ac, at, a.ordinal, a.kind}, {bc, bt, b.ordinal, b.kind});
    };
    for (const auto& child : analysis.state->children) {
        out.capabilities.contextualGuards |= child.regional.capabilities.contextualGuards;
    }
    out.prepare = [owned]() -> FailureOr<std::unique_ptr<PreparedLogicalPlan>> {
        auto result = prepareSequenceInsertion(*owned);
        if (succeeded(result)) { (*result)->completeInvocation = false; }
        return result;
    };
    auto convert = [&](SequenceSelectedEvent selected) {
        const auto& occurrence = analysis.occurrences[selected.port];
        return RegionalSelector{{starts[occurrence.child] + occurrence.type,
            occurrence.ordinal, PeriodicEventKind::Start}, selected.present};
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
    return out;
}
} // namespace mlir::pto::frontiersynch
