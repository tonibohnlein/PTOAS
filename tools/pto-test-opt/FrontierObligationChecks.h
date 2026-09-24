// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#ifndef PTO_TEST_FRONTIER_OBLIGATION_CHECKS_H
#define PTO_TEST_FRONTIER_OBLIGATION_CHECKS_H
#include "PTO/Transforms/FrontierSynch/FactoredUse.h"
#include "PTO/Transforms/FrontierSynch/OriginalObligations.h"
#include <set>

namespace frontier_obligation_checks {
namespace fs = mlir::pto::frontiersynch;
using Kind = fs::OriginalObligationKey::Kind;
using Edge = std::tuple<std::size_t, std::size_t, Kind>;

inline fs::FactoredUseRegion operation(std::size_t id)
{
    fs::FactoredUseRegion r;
    r.kind = fs::FactoredUseRegion::Kind::Access;
    r.access = id;
    return r;
}
inline fs::FactoredUseRegion optional(std::size_t guard, std::size_t op)
{
    fs::FactoredUseRegion r;
    r.kind = fs::FactoredUseRegion::Kind::Choice;
    r.condition = guard;
    r.children = {operation(op), fs::FactoredUseRegion{}};
    return r;
}

inline bool check(bool partial, bool sameGuard)
{
    fs::FactoredUseProjection input;
    input.frame.cell = 0;
    const bool reads[] = {false, true, true, true, true, false};
    const bool writes[] = {true, false, true, false, false, true};
    for (std::size_t op = 0; op < 6; ++op) {
        auto access = std::make_shared<fs::FactoredUseAccess>();
        access->operation = op;
        access->read = reads[op];
        access->write = writes[op];
        access->definiteWrite = writes[op] && (op != 2 || !partial);
        if (reads[op]) {
            access->readIncidences = op == 3 ? std::vector<std::size_t>{0, 1} : std::vector<std::size_t>{0};
        }
        if (writes[op]) {
            access->writeIncidences = {0};
        }
        input.accesses.push_back(access);
    }
    const auto effects = input.accesses;
    fs::FactoredUseInterface boundary;
    boundary.arena = std::make_shared<fs::FactoredUseArena>(input.frame);
    boundary.incoming.writers =
        boundary.arena->incoming(0, fs::FactoredUseNode::Boundary::Entry, fs::FactoredUseNode::Role::Writer);
    const auto g = boundary.arena->test({10, fs::NoFactoredId});
    const auto h = boundary.arena->test({sameGuard ? 10UL : 11UL, fs::NoFactoredId});
    input.body.children = {operation(0), operation(1), optional(g, 2), operation(3), optional(h, 4), operation(5)};
    fs::FactoredUseBuilder transfer(std::move(input), boundary);
    const auto& graph = transfer.get();
    if (!graph.complete) {
        return false;
    }
    fs::OriginalObligations index([](fs::FactoredGuardIdentity guard) { return guard.value; });
    std::vector<fs::OriginalObligationFamilyId> ids;
    for (std::size_t node = 0; node < graph.nodes().size(); ++node) {
        const auto& d = graph.nodes()[node];
        if (d.kind != fs::FactoredUseNode::Kind::Demand) {
            continue;
        }
        fs::OriginalObligationFamily family;
        family.key.cell = 0;
        family.key.consumerOperation = d.operation;
        family.key.consumerOriginal = d.operation;
        family.key.kind = d.hazard == fs::FactoredUseNode::Hazard::RAW ? Kind::RAW :
                          d.hazard == fs::FactoredUseNode::Hazard::WAR ? Kind::WAR :
                                                                         Kind::WAW;
        family.key.occurrences = fs::OriginalObligationKey::Occurrences::FixedUseProjection;
        family.key.stop = {fs::ObligationCut::Kind::PayloadBefore, d.operation};
        family.representation = fs::OriginalObligationFamily::Representation::Factored;
        family.expression = &graph;
        family.sources = d.left;
        family.demandNode = node;
        family.applicability = index.importCondition(graph, graph.site(d.operation)->applicability);
        family.guardsIdentified = true;
        family.consumerEffects = family.key.kind == Kind::RAW ? &graph.site(d.operation)->access->readIncidences :
                                                                &graph.site(d.operation)->access->writeIncidences;
        ids.push_back(index.add(family));
    }
    index.freeze();
    if (!index.complete() || index.stats().enumeratedOrigins != 0) {
        return false;
    }
    for (bool g : {false, true}) {
        for (bool h : {false, true}) {
            const bool selectedH = sameGuard ? g : h;
            std::vector<std::size_t> trace{0, 1};
            if (g) {
                trace.push_back(2);
            }
            trace.push_back(3);
            if (selectedH) {
                trace.push_back(4);
            }
            trace.push_back(5);
            // Independent concrete provenance scan. It never reads a factored node.
            std::set<std::size_t> writers{fs::NoFactoredId}, readers;
            std::set<Edge> expected;
            for (auto op : trace) {
                bool read = false, write = false, full = false;
                read = effects[op]->read;
                write = effects[op]->write;
                full = effects[op]->definiteWrite;
                if (read) {
                    for (auto w : writers) {
                        expected.emplace(w, op, Kind::RAW);
                    }
                }
                if (write) {
                    for (auto w : writers) {
                        expected.emplace(w, op, Kind::WAW);
                    }
                    for (auto r : readers) {
                        expected.emplace(r, op, Kind::WAR);
                    }
                }
                if (full) {
                    writers = {op};
                    readers.clear();
                } else {
                    if (write) {
                        writers.insert(op);
                    }
                    if (read) {
                        readers.insert(op);
                    }
                }
            }
            std::set<Edge> actual;
            for (auto id : ids) {
                const auto* f = index.get(id);
                if (!f) {
                    return false;
                }
                if (f->key.consumerOperation == 3 && f->consumerEffects->size() != 2) {
                    return false;
                }
                auto enumeration = index.origins(id);
                if (!enumeration.complete) {
                    return false;
                }
                for (const auto& member : enumeration.members) {
                    const auto enabled = index.predicates().evaluate(
                        member.condition,
                        [&](std::size_t value) -> std::optional<bool> { return value == 10 ? g : selectedH; });
                    if (!enabled) {
                        return false;
                    }
                    if (*enabled) {
                        actual.emplace(
                            member.source.incoming ? fs::NoFactoredId : member.source.operation,
                            f->key.consumerOperation, f->key.kind);
                    }
                }
            }
            if (actual != expected) {
                return false;
            }
        }
    }
    return true;
}
inline bool run() { return check(false, false) && check(true, false) && check(false, true); }
} // namespace frontier_obligation_checks
#endif
