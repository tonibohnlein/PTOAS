// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SequenceAnalysisInternal.h"
#include "PTO/Transforms/FrontierSynch/HardwareProtection.h"
namespace mlir::pto::frontiersynch {
void SequenceAnalysisState::bridges()
{
    const SyncStorageEffects* model = nullptr;
    SmallVector<const CompoundInstanceElement*> phases;
    bool residual = false;
    for (const auto& child : children) {
        if (child.regional.accessModel) { model = child.regional.accessModel; }
        for (const auto& anchor : child.regional.anchors) { phases.push_back(anchor.phase); }
        for (const auto& access : child.regional.accessBoundary) { residual |= !access.representedByCells; }
    }
    residual |= model && model->hasUniformRelationships(phases);
    const auto protection = model ? invocationProtectionGroups(*model) : InvocationProtection{};
    auto storageCrossing = [&](uint32_t cell, Selected source, Selected target) {
        const auto& a = ports[source.port];
        const auto& b = ports[target.port];
        auto x = children[a.child].anchors[a.type].phase;
        auto y = children[b.child].anchors[b.type].phase;
        if (cells[cell].space == AddressSpace::ACC && hardwareProtectsConflict(
                pipe(source.port), protection.lookup(x), pipe(target.port), protection.lookup(y))) { return; }
        crossing(source, target);
    };
    if (residual) {
        for (uint32_t a = 0; a < children.size(); ++a) {
            for (uint32_t b = a + 1; b < children.size(); ++b) {
                const auto& left = children[a].regional;
                const auto& right = children[b].regional;
                if (!left.accessModel || !right.accessModel) { continue; }
                for (const auto& x : left.accessBoundary) {
                    for (const auto& y : right.accessBoundary) {
                        const auto& model = *left.accessModel;
                        bool conflict = model.uniformConflict(x.effect, y.effect);
                        if (!conflict && (!x.representedByCells || !y.representedByCells) &&
                            model.residualConflict(x.effect, y.effect)) {
                            if (left.occurrenceLoops[x.last.event.type] || right.occurrenceLoops[y.first.event.type]) {
                                fail("crossing symbolic access predicate needs an occurrence adapter"); return;
                            }
                            conflict = true;
                        }
                        if (conflict) {
                            crossing({port(a, x.last.event.type, x.last.event.ordinal), x.last.present},
                                     {port(b, y.first.event.type, y.first.event.ordinal), y.first.present});
                        }
                    }
                }
            }
        }
    }
    for (uint32_t cell = 0; cell < cells.size(); ++cell) {
        std::vector<Selected> writers;
        std::map<uint32_t, std::vector<Selected>> readers;
        for (uint32_t child = 0; child < children.size(); ++child) {
            const auto& summary = boundaries[child][cell];
            Expr overwritten = no();
            Expr anyReader = no();
            for (const auto& [p, oldReaders] : readers) {
                for (auto old : oldReaders) { anyReader = either(anyReader, old.present); }
            }
            for (auto first : summary.firstWriters) {
                overwritten = either(overwritten, first.present);
                for (auto old : writers) {
                    bool suppliedByReader = false;
                    for (const auto& [p, oldReaders] : readers) {
                        for (auto reader : oldReaders) {
                            ++costs.implicationChecks;
                            suppliedByReader |= expressions.implies(old.present, reader.present);
                        }
                    }
                    if (!suppliedByReader) {
                        old.present = both(old.present, negate(anyReader));
                        storageCrossing(cell, old, first);
                    }
                }
                for (const auto& [p, oldReaders] : readers) {
                    for (auto old : oldReaders) { storageCrossing(cell, old, first); }
                }
            }
            for (const auto& [p, firstReaders] : summary.firstReaders) {
                for (auto first : firstReaders) {
                    for (auto old : writers) { storageCrossing(cell, old, first); }
                }
            }
            for (auto& old : writers) { old.present = both(old.present, negate(overwritten)); }
            llvm::append_range(writers, summary.lastWriters);
            for (auto& [p, old] : readers) {
                Expr replaced = overwritten;
                auto found = summary.lastReaders.find(p);
                if (found != summary.lastReaders.end()) {
                    for (auto current : found->second) { replaced = either(replaced, current.present); }
                }
                for (auto& reader : old) { reader.present = both(reader.present, negate(replaced)); }
            }
            for (const auto& [p, current] : summary.lastReaders) { llvm::append_range(readers[p], current); }
        }
    }
}
void SequenceAnalysisState::canonicalizeCrossings()
{
    const auto size = ports.size();
    std::vector<std::vector<Expr>> original(size, std::vector<Expr>(size, no()));
    std::vector<std::vector<Expr>> equal(size, std::vector<Expr>(size, no()));
    std::vector<std::vector<Expr>> targetEquivalent(size, std::vector<Expr>(size, no()));
    for (const auto& edge : crossings) { original[edge.source][edge.target] = edge.guard; }
    for (uint32_t a = 0; a < size; ++a) {
        for (uint32_t b = 0; b < size; ++b) { equal[a][b] = same(a, b); }
    }
    // Boolean matrix products identify earlier records for the same actual
    // endpoint pair in O(P^3), without pairwise comparison of all r records.
    for (uint32_t s = 0; s < size; ++s) {
        for (uint32_t t = 0; t < size; ++t) {
            for (uint32_t v = 0; v < size; ++v) {
                targetEquivalent[s][t] = either(targetEquivalent[s][t], both(original[s][v], equal[t][v]));
            }
        }
    }
    for (auto& edge : crossings) {
        Expr earlier = no();
        for (uint32_t u = 0; u < edge.source; ++u) {
            earlier = either(earlier, both(equal[edge.source][u], targetEquivalent[u][edge.target]));
        }
        for (uint32_t v = 0; v < edge.target; ++v) {
            earlier = either(earlier, both(original[edge.source][v], equal[edge.target][v]));
        }
        edge.guard = both(edge.guard, negate(earlier));
    }
}
bool SequenceAnalysisState::valueBridges()
{
    for (uint32_t b = 0; b < children.size(); ++b) {
        for (uint32_t target = 0; target < children[b].anchors.size(); ++target) {
            for (const auto& edge : index.prerequisitesFor(children[b].anchors[target].phase->elementOp)) {
                for (uint32_t a = 0; a < b; ++a) {
                    for (uint32_t source = 0; source < children[a].anchors.size(); ++source) {
                        if (children[a].anchors[source].phase != edge.producer) { continue; }
                        auto sourceLoop = children[a].regional.occurrenceLoops[source];
                        auto targetLoop = children[b].regional.occurrenceLoops[target];
                        // Slices of one original loop have disjoint ordinals.
                        // SSA edges within its body connect the same iteration,
                        // and have already been handled by each local analysis.
                        if (sourceLoop && sourceLoop == targetLoop) { continue; }
                        if (sourceLoop) {
                            return fail("crossing value producer requires a last-occurrence selector");
                        }
                        const auto x = port(a, source, c(0));
                        const auto y = port(b, target, children[b].regional.firstOrdinal.value_or(c(0)));
                        const auto guard = both(present(x), present(y));
                        if (edge.native) { nativeValueCrossings.push_back({x, y, guard}); }
                        else { crossing({x, present(x)}, {y, present(y)}); }
                    }
                }
            }
        }
    }
    return error.empty();
}
bool SequenceAnalysisState::closure()
{
    const std::size_t size = 2 * ports.size();
    if (!error.empty() || (size && size > SIZE_MAX / size / sizeof(Expr))) {
        return fail("sequence boundary query matrix size overflow");
    }
    canonicalizeCrossings();
    graph.assign(size, std::vector<Expr>(size, no()));
    auto add = [&](std::size_t a, std::size_t b, Expr guard) { graph[a][b] = either(graph[a][b], guard); };
    for (uint32_t a = 0; a < ports.size(); ++a) {
        for (uint32_t b = 0; b < ports.size(); ++b) {
            const auto& x = ports[a];
            const auto& y = ports[b];
            if (x.child != y.child) { continue; }
            auto exists = both(present(a), present(b));
            for (unsigned ac = 0; ac < 2; ++ac) {
                for (unsigned bc = 0; bc < 2; ++bc) {
                    PeriodicEvent source{x.type, ac ? PeriodicEventKind::Completion : PeriodicEventKind::Start};
                    PeriodicEvent target{y.type, bc ? PeriodicEventKind::Completion : PeriodicEventKind::Start};
                    auto answer = children[x.child].regional.reachability(
                        {x.type, x.ordinal, source.kind}, {y.type, y.ordinal, target.kind});
                    if (!answer) { return fail("child all-event query unavailable"); }
                    Expr reachable = *answer;
                    add(2*a+ac, 2*b+bc, both(exists, reachable));
                }
            }
        }
    }
    const auto local = graph;
    std::map<uint32_t, std::vector<Selected>> preceding;
    for (uint32_t childId = 0; childId < children.size(); ++childId) {
        const auto& regional = children[childId].regional;
        for (const auto& [p, firsts] : regional.firstPayloads) {
            auto nonempty = no();
            for (auto selected : firsts) {
                nonempty = either(nonempty, selected.present);
                auto first = port(childId, selected.event.type, selected.event.ordinal);
                if (2*static_cast<std::size_t>(first) >= size) { return fail("missing native boundary port"); }
                for (auto old : preceding[p]) {
                    auto guard = both(old.present, selected.present);
                    add(2*old.port, 2*first, guard);
                    add(2*old.port+1, 2*first+1, guard);
                }
            }
            for (auto& old : preceding[p]) { old.present = both(old.present, negate(nonempty)); }
            auto found = regional.lastPayloads.find(p);
            if (found != regional.lastPayloads.end()) {
                for (auto selected : found->second) {
                    auto event = port(childId, selected.event.type, selected.event.ordinal);
                    preceding[p].push_back({event, selected.present});
                }
            }
        }
    }
    for (const auto& edge : nativeValueCrossings) { add(2*edge.source+1, 2*edge.target, edge.guard); }
    for (auto& edge : crossings) {
        auto native = no();
        for (const auto& fixed : nativeValueCrossings) {
            auto identical = both(same(fixed.source, edge.source), same(fixed.target, edge.target));
            native = either(native, both(fixed.guard, identical));
        }
        add(2*edge.source+1, 2*edge.target, edge.guard);
        edge.guard = both(edge.guard, negate(native));
    }
    // Each child query block is already closed. Every crossing advances the
    // child index, so propagate through the block DAG rather than repeatedly
    // closing symbolic reflexive aliases inside a child.
    auto links = std::move(graph);
    graph = local;
    std::map<uint32_t, std::vector<std::size_t>> childEvents;
    for (std::size_t event = 0; event < size; ++event) {
        childEvents[ports[event / 2].child].push_back(event);
    }
    for (std::size_t source = 0; source < size; ++source) {
        // Empty children have no ports. Visiting only populated blocks avoids
        // an extra H*P^2 factor when a caller supplies many empty regions.
        for (const auto& [child, events] : childEvents) {
            if (child <= ports[source / 2].child) { continue; }
            std::vector<Expr> entry(size, no());
            for (auto target : events) {
                for (std::size_t previous = 0; previous < size; ++previous) {
                    if (ports[previous / 2].child < child) {
                        entry[target] = either(entry[target], both(graph[source][previous], links[previous][target]));
                    }
                }
            }
            for (auto target : events) {
                for (auto first : events) {
                    graph[source][target] = either(graph[source][target], both(entry[first], local[first][target]));
                }
            }
        }
    }
    for (auto& edge : crossings) {
        Expr alternate = no();
        for (uint32_t z = 0; z < ports.size(); ++z) {
            for (unsigned completion = 0; completion < 2; ++completion) {
                auto distinct = negate(same(z, completion ? edge.source : edge.target));
                alternate = either(alternate, both(distinct,
                    both(graph[2*edge.source+1][2*z+completion], graph[2*z+completion][2*edge.target])));
            }
        }
        edge.guard = both(edge.guard, negate(alternate));
    }
    return true;
}

} // namespace mlir::pto::frontiersynch
