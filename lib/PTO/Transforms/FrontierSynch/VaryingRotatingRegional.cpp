// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Algebraic storage generators without expanding banks or loop iterations.
#include "PTO/Transforms/FrontierSynch/VaryingRotatingRegional.h"
#include "PTO/Transforms/FrontierSynch/RotatingAnalysis.h"
#include "CountedLoop.h"
#include "CircuitEndpoints.h"
#include "../InsertSync/SyncEffectRanges.h"
#include <map>
#include <tuple>
namespace mlir::pto::frontiersynch {
namespace {
using Id = RegionExpressions::Id;
using Occ = BoundaryOccurrence;
using Matrix = std::vector<std::vector<uint8_t>>;
struct Port {
    Occ occurrence;
    PeriodicEventKind kind;
};
using PortKey = std::tuple<uint32_t, uint64_t, bool, unsigned>;
PortKey key(Port p) { return {p.occurrence.type, p.occurrence.coordinate, p.occurrence.tail, unsigned(p.kind)}; }
Matrix product(const Matrix& a, const Matrix& b)
{
    Matrix out(a.size(), std::vector<uint8_t>(a.size()));
    for (std::size_t i = 0; i < a.size(); ++i) {
        for (std::size_t k = 0; k < a.size(); ++k) {
            if (!a[i][k]) {
                continue;
            }
            for (std::size_t j = 0; j < a.size(); ++j) {
                out[i][j] |= b[k][j];
            }
        }
    }
    return out;
}
struct State : std::enable_shared_from_this<State> {
    func::FuncOp function;
    VaryingRotatingRecognition recognized;
    RotatingAnalysis child;
    AffineRotatingVisits visits;
    std::shared_ptr<RegionExpressions> arena;
    RegionalAnalysis out;
    Id trips;
    std::vector<Port> ports;
    std::map<PortKey, std::size_t> portIds;
    std::vector<Matrix> startup, suffix;
    std::vector<std::vector<Matrix>> powers;
    using TransferKey = std::tuple<uint32_t, Id, unsigned, Id, Id>;
    std::map<TransferKey, std::vector<Id>> transferMemo;
    std::string error;
    RegionExpressions& e() { return *arena; }
    Id c(uint64_t x) { return e().constant(x); }
    Id mul(Id x, uint64_t n)
    {
        auto value = c(0);
        while (n) {
            if (n & 1) {
                value = e().add(value, x);
            }
            n >>= 1;
            if (n) {
                x = e().add(x, x);
            }
        }
        return value;
    }
    Id length(Id visit) { return e().add(mul(visit, visits.slope), c(visits.intercept)); }
    Id coordinate(Occ x, Id visit) { return x.tail ? e().sub(length(visit), c(x.coordinate)) : c(x.coordinate); }
    RegionalEvent event(Occ x, Id visit, PeriodicEventKind kind = PeriodicEventKind::Start)
    {
        return {x.type, coordinate(x, visit), kind, {visit}};
    }
    Id present(RegionalEvent x)
    {
        if (x.type >= child.phases.size() || x.visits.size() != 1 || x.ordinal >= e().size() ||
            x.visits[0] >= e().size() || e().isBoolean(x.ordinal) || e().isBoolean(x.visits[0]) ||
            (x.kind != PeriodicEventKind::Start && x.kind != PeriodicEventKind::Completion)) {
            return RegionExpressions::invalid;
        }
        return e().land(e().lt(x.visits[0], trips), e().lt(x.ordinal, length(x.visits[0])));
    }
    Id inside(RegionalEvent a, RegionalEvent b)
    {
        auto threshold = child.periodic.eventThreshold({a.type, a.kind}, {b.type, b.kind});
        if (threshold.error != PeriodicQueryError::None) {
            return RegionExpressions::invalid;
        }
        if (!threshold.displacement) {
            return e().boolean(false);
        }
        return e().land(
            e().le(a.ordinal, b.ordinal), e().le(c(*threshold.displacement), e().sub(b.ordinal, a.ordinal)));
    }
    RotatingBoundaryType type(uint64_t t)
    {
        // The recognizer proves actual lengths fit signed index. Representatives
        // used here are checked separately by analyzeAffineRotatingVisits.
        const auto n = visits.slope * t + visits.intercept;
        return n < visits.child.cutoff ? visits.child.select(n) : visits.child.types[n % visits.child.period];
    }
    void add(Occ x)
    {
        for (auto kind : {PeriodicEventKind::Start, PeriodicEventKind::Completion}) {
            Port p{x, kind};
            if (portIds.emplace(key(p), ports.size()).second) {
                ports.push_back(p);
            }
        }
    }
    void collect(const RotatingBoundaryType& t)
    {
        for (const auto& cell : t.cells) {
            if (cell.firstWriter) {
                add(*cell.firstWriter);
            }
            if (cell.lastWriter) {
                add(*cell.lastWriter);
            }
            for (auto [pipe, x] : cell.firstReaders) {
                add(x);
            }
            for (auto [pipe, x] : cell.lastReaders) {
                add(x);
            }
        }
        for (auto [pipe, x] : t.firstPayloads) {
            add(x);
        }
        for (auto [pipe, x] : t.lastPayloads) {
            add(x);
        }
    }
    bool nativeReaches(Port a, Port b, uint64_t n)
    {
        if (!n || (a.occurrence.tail && a.occurrence.coordinate > n) ||
            (b.occurrence.tail && b.occurrence.coordinate > n) || a.occurrence.at(n) >= n || b.occurrence.at(n) >= n) {
            return false;
        }
        auto threshold = child.periodic.eventThreshold({a.occurrence.type, a.kind}, {b.occurrence.type, b.kind});
        if (threshold.error != PeriodicQueryError::None) {
            error = "varying child query unavailable";
            return false;
        }
        return threshold.displacement && a.occurrence.at(n) <= b.occurrence.at(n) &&
               b.occurrence.at(n) - a.occurrence.at(n) >= *threshold.displacement;
    }
    Matrix transfer(uint64_t target)
    {
        auto left = type(target - 1), right = type(target);
        Matrix crossing(ports.size(), std::vector<uint8_t>(ports.size()));
        std::vector<std::pair<Port, Port>> edges;
        for (auto d : visits.crossingInto(target)) {
            edges.push_back({{d.source, PeriodicEventKind::Completion}, {d.target, PeriodicEventKind::Start}});
        }
        for (auto [pipe, last] : left.lastPayloads) {
            auto first = right.firstPayloads.find(pipe);
            if (first == right.firstPayloads.end()) {
                continue;
            }
            for (auto kind : {PeriodicEventKind::Start, PeriodicEventKind::Completion}) {
                edges.push_back({{last, kind}, {first->second, kind}});
            }
        }
        for (const auto& [a, b] : edges) {
            auto from = portIds.find(key(a)), to = portIds.find(key(b));
            if (from == portIds.end() || to == portIds.end()) {
                error = "varying crossing has no exported event port";
                return crossing;
            }
            crossing[from->second][to->second] = 1;
        }
        Matrix prefix(ports.size(), std::vector<uint8_t>(ports.size()));
        Matrix suffixPaths = prefix;
        for (std::size_t i = 0; i < ports.size(); ++i) {
            for (std::size_t j = 0; j < ports.size(); ++j) {
                prefix[i][j] = nativeReaches(ports[i], ports[j], left.representative);
                suffixPaths[i][j] = nativeReaches(ports[i], ports[j], right.representative);
            }
        }
        // Two Boolean products share paths among crossings: O(P^3), rather
        // than one P-by-P outer product for each of up to O(P^2) crossings.
        return product(product(prefix, crossing), suffixPaths);
    }
    bool build()
    {
        for (uint64_t t = 0; t <= visits.startup + visits.period; ++t) {
            collect(type(t));
        }
        for (uint64_t t = 1; t <= visits.startup; ++t) {
            startup.push_back(transfer(t));
        }
        for (uint64_t phase = 0; phase < visits.period; ++phase) {
            suffix.push_back(transfer(visits.startup + visits.period + phase + 1));
        }
        for (uint64_t phase = 0; phase < visits.period; ++phase) {
            Matrix cycle(ports.size(), std::vector<uint8_t>(ports.size()));
            for (std::size_t i = 0; i < ports.size(); ++i) {
                cycle[i][i] = 1;
            }
            for (uint64_t j = 0; j < visits.period; ++j) {
                cycle = product(cycle, suffix[(phase + j) % visits.period]);
            }
            std::vector<Matrix> column{std::move(cycle)};
            for (unsigned bit = 1; bit < 64; ++bit) {
                column.push_back(product(column.back(), column.back()));
            }
            powers.push_back(std::move(column));
        }
        return error.empty();
    }
    std::vector<Id> advance(const std::vector<Id>& values, const Matrix& matrix, Id enabled)
    {
        std::vector<Id> next(ports.size(), e().boolean(false));
        for (std::size_t j = 0; j < ports.size(); ++j) {
            for (std::size_t i = 0; i < ports.size(); ++i) {
                if (matrix[i][j]) {
                    next[j] = e().lor(next[j], values[i]);
                }
            }
            next[j] = e().select(enabled, next[j], values[j]);
        }
        return next;
    }
    std::optional<Id> query(RegionalEvent a, RegionalEvent b)
    {
        auto pa = present(a), pb = present(b);
        if (pa == RegionExpressions::invalid || pb == RegionExpressions::invalid) {
            return std::nullopt;
        }
        auto s = a.visits[0], t = b.visits[0];
        auto both = e().land(pa, pb);
        if (e().constantValue(both) == 0 || e().constantValue(e().lt(t, s)) == 1) {
            return e().boolean(false);
        }
        if (e().constantValue(e().eq(s, t)) == 1) {
            return e().land(both, inside(a, b));
        }
        auto direct = e().land(e().eq(s, t), inside(a, b));
        auto memoKey = TransferKey{a.type, a.ordinal, unsigned(a.kind), s, t};
        auto cached = transferMemo.find(memoKey);
        if (cached == transferMemo.end()) {
            std::vector<Id> values;
            for (auto port : ports) {
                auto target = event(port.occurrence, s, port.kind);
                values.push_back(e().land(present(target), inside(a, target)));
            }
            for (std::size_t j = 0; j < startup.size(); ++j) {
                values = advance(values, startup[j], e().land(e().le(s, c(j)), e().lt(c(j), t)));
            }
            auto begin = e().select(e().lt(s, c(visits.startup)), c(visits.startup), s);
            auto count = e().select(e().lt(begin, t), e().sub(t, begin), c(0));
            auto phase = e().rem(e().sub(begin, c(visits.startup)), c(visits.period));
            auto cycles = e().div(count, c(visits.period)), tail = e().rem(count, c(visits.period));
            std::vector<Id> final(ports.size(), e().boolean(false));
            for (uint64_t p = 0; p < visits.period; ++p) {
                auto row = values;
                for (unsigned bit = 0; bit < 64; ++bit) {
                    auto active = e().eq(e().rem(e().div(cycles, c(uint64_t(1) << bit)), c(2)), c(1));
                    row = advance(row, powers[p][bit], active);
                }
                for (uint64_t j = 0; j < visits.period; ++j) {
                    row = advance(row, suffix[(p + j) % visits.period], e().lt(c(j), tail));
                }
                for (std::size_t j = 0; j < ports.size(); ++j) {
                    final[j] = e().lor(final[j], e().land(e().eq(phase, c(p)), row[j]));
                }
            }
            cached = transferMemo.emplace(memoKey, std::move(final)).first;
        }
        const auto& final = cached->second;
        auto across = e().boolean(false);
        for (std::size_t j = 0; j < ports.size(); ++j) {
            auto source = event(ports[j].occurrence, t, ports[j].kind);
            across = e().lor(across, e().land(final[j], e().land(present(source), inside(source, b))));
        }
        return e().land(e().land(pa, pb), e().lor(direct, e().land(e().lt(s, t), across)));
    }
    Id before(RegionalEvent a, RegionalEvent b)
    {
        return e().lor(
            e().lt(a.visits[0], b.visits[0]),
            e().land(
                e().eq(a.visits[0], b.visits[0]),
                e().lor(
                    e().lt(a.ordinal, b.ordinal),
                    e().land(e().eq(a.ordinal, b.ordinal), e().boolean(a.type < b.type)))));
    }
    std::vector<RegionalSelector> firsts(std::vector<RegionalSelector> candidates)
    {
        auto original = candidates;
        for (auto& x : candidates) {
            for (const auto& y : original) {
                x.present = e().land(x.present, e().lnot(e().land(y.present, before(y.event, x.event))));
            }
        }
        return candidates;
    }
    RegionalSelector selector(Occ x, Id visit, Id enabled)
    {
        auto v = event(x, visit);
        return {v, e().land(enabled, present(v))};
    }
    bool selectors(const SyncInput& input)
    {
        std::vector<SyncStorageCell> cells;
        for (auto cell : visits.child.cells) {
            std::optional<SyncStorageCell> physical;
            for (std::size_t i = 0; i < child.fragments.size(); ++i) {
                auto f = child.fragments[i];
                if (f.family != cell.family || f.atom != cell.atom) {
                    continue;
                }
                const auto& access = recognized.child.accesses[i];
                const auto& effect = input.accesses().effects()[access.effect];
                auto ranges = mlir::pto::detail::physicalSlotRanges(input, *effect.memory);
                if (access.firstPhysicalSlot) {
                    auto range = *access.firstPhysicalSlot;
                    range.begin += cell.slot * access.physicalSlotStride;
                    range.end += cell.slot * access.physicalSlotStride;
                    physical = range;
                } else if (cell.slot < ranges.size()) {
                    physical = ranges[cell.slot];
                }
                if (!physical) {
                    error = "varying physical boundary cell unavailable";
                    return false;
                }
                physical->end = physical->begin + access.atom->second;
                physical->begin += access.atom->first;
                break;
            }
            if (!physical) {
                error = "varying boundary has no physical mapping";
                return false;
            }
            cells.push_back(*physical);
        }
        out.storageBoundary.resize(cells.size());
        for (std::size_t i = 0; i < cells.size(); ++i) {
            out.storageBoundary[i].cell = cells[i];
        }
        // All head extrema have appeared by the first long visit. Enumerating
        // this certified startup does not enumerate a runtime trip count.
        for (uint64_t v = 0; v <= visits.startup; ++v) {
            auto t = type(v);
            for (std::size_t i = 0; i < cells.size(); ++i) {
                auto& b = out.storageBoundary[i];
                if (t.cells[i].firstWriter) {
                    b.firstWriters.push_back(selector(*t.cells[i].firstWriter, c(v), e().boolean(true)));
                }
                for (auto [p, x] : t.cells[i].firstReaders) {
                    b.firstReaders[p].push_back(selector(x, c(v), e().boolean(true)));
                }
            }
            for (auto [p, x] : t.firstPayloads) {
                out.firstPayloads[p].push_back(selector(x, c(v), e().boolean(true)));
            }
        }
        auto final = e().sub(trips, c(1));
        for (uint64_t v = 0; v < visits.startup + visits.period; ++v) {
            auto t = type(v);
            auto active =
                v < visits.startup ?
                    e().eq(final, c(v)) :
                    e().land(
                        e().le(c(visits.startup), final),
                        e().eq(e().rem(e().sub(final, c(visits.startup)), c(visits.period)), c(v - visits.startup)));
            active = e().land(e().lt(c(0), trips), active);
            for (std::size_t i = 0; i < cells.size(); ++i) {
                auto& b = out.storageBoundary[i];
                if (t.cells[i].lastWriter) {
                    b.lastWriters.push_back(selector(*t.cells[i].lastWriter, final, active));
                }
                for (auto [p, x] : t.cells[i].lastReaders) {
                    b.lastReaders[p].push_back(selector(x, final, active));
                }
            }
            for (auto [p, x] : t.lastPayloads) {
                out.lastPayloads[p].push_back(selector(x, final, active));
            }
        }
        for (auto& b : out.storageBoundary) {
            b.firstWriters = firsts(std::move(b.firstWriters));
            for (auto& [p, readers] : b.firstReaders) {
                for (auto& reader : readers) {
                    for (const auto& writer : b.firstWriters) {
                        reader.present = e().land(
                            reader.present,
                            e().lnot(e().land(writer.present, e().lnot(before(reader.event, writer.event)))));
                    }
                }
                readers = firsts(std::move(readers));
            }
        }
        for (auto& [p, values] : out.firstPayloads) {
            values = firsts(std::move(values));
        }
        auto firstVisit = c(visits.intercept ? 0 : 1);
        for (uint32_t i = 0; i < child.phases.size(); ++i) {
            auto first = selector({i, 0, false}, firstVisit, e().boolean(true));
            auto last = selector({i, 1, true}, final, e().lt(c(0), trips));
            out.firstSitePayloads[i].push_back(first);
            for (auto effect : input.accesses().effectsFor(child.phases[i])) {
                out.accessBoundary.push_back({effect, first, last, true});
            }
        }
        return true;
    }
    FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepare();
};
FailureOr<std::unique_ptr<PreparedLogicalPlan>> State::prepare()
{
    CircuitEndpoints emit(function, e(), out.anchors);
    auto outer = recognized.outer, inner = recognized.inner;
    auto ordinal = [&](scf::ForOp loop) {
        auto domain = CountedLoop::get(loop);
        return e().div(e().sub(e().input(loop.getInductionVar()), e().input(loop.getLowerBound())), c(domain->step));
    };
    auto visit = ordinal(outer), i = ordinal(inner), n = length(visit);
    for (auto retained : child.periodic.retained) {
        auto record = child.periodic.generators[retained];
        auto d = c(record.displacement);
        auto publish = e().lt(d, e().sub(n, i)), consume = e().le(d, i);
        if (!emit.add(record.source, record.target, publish, consume, {visit, i}, {visit, e().sub(i, d)})) {
            error = "varying internal endpoint unavailable: " + e().lastEmissionError();
            return failure();
        }
    }
    auto add = [&](const std::vector<BoundaryDemand>& records, uint64_t targetVisit, bool periodic) {
        for (auto record : records) {
            auto predicate = [&](Id target) {
                if (!periodic) {
                    return e().eq(target, c(targetVisit));
                }
                return e().land(
                    e().lt(c(visits.startup), target),
                    e().eq(e().rem(e().sub(target, c(visits.startup)), c(visits.period)), c(targetVisit)));
            };
            auto publish = e().land(e().lt(c(1), e().sub(trips, visit)), predicate(e().add(visit, c(1))));
            auto consume = e().land(e().lt(c(0), visit), predicate(visit));
            publish = e().land(publish, e().eq(i, coordinate(record.source, visit)));
            consume = e().land(consume, e().eq(i, coordinate(record.target, visit)));
            if (!emit.add(
                    record.source.type, record.target.type, publish, consume, {visit, coordinate(record.source, visit)},
                    {e().sub(visit, c(1)), coordinate(record.source, e().sub(visit, c(1)))})) {
                return false;
            }
        }
        return true;
    };
    for (uint64_t v = 1; v < visits.startup; ++v) {
        if (!add(visits.startupCrossings[v], v, false)) {
            return failure();
        }
    }
    if (!add(visits.seam, visits.startup, false)) {
        return failure();
    }
    for (uint64_t p = 0; p < visits.period; ++p) {
        if (!add(visits.suffixCrossings[p], p, true)) {
            return failure();
        }
    }
    return emit.take();
}
} // namespace
FailureOr<RegionalAnalysis> varyingRotatingRegionalResult(
    func::FuncOp function, const VaryingRotatingRecognition& recognized, const PhaseIndex& index,
    const SyncInput& input, std::shared_ptr<RegionExpressions> expressions, std::string& error,
    const AffineRotatingVisits* certificate)
{
    if (!function || !expressions || !recognized.outer || !recognized.inner ||
        recognized.result.state != RecognitionState::Applicable ||
        recognized.outer->getParentOfType<func::FuncOp>() != function ||
        !recognized.outer->isProperAncestor(recognized.inner)) {
        error = "varying export requires a recognized original loop and expression arena";
        return failure();
    }
    if (certificate && (certificate->slope != recognized.slope || certificate->intercept != recognized.intercept ||
                        !certificate->period || !certificate->child.period ||
                        certificate->suffixCrossings.size() != certificate->period ||
                        certificate->startupCrossings.size() != certificate->startup ||
                        certificate->child.types.size() != certificate->child.period)) {
        error = "varying certificate does not match its recognized visit domain";
        return failure();
    }
    auto state = std::make_shared<State>();
    state->function = function;
    state->recognized = recognized;
    state->arena = std::move(expressions);
    state->visits = certificate ? *certificate : analyzeVaryingRotating(recognized, index, input);
    if (!state->visits.error.empty()) {
        error = state->visits.error;
        return failure();
    }
    auto inner = recognized.inner;
    auto phases = index.explicitSequence(*inner.getBody());
    auto domain = CountedLoop::get(recognized.outer);
    if (!domain || failed(phases)) {
        error = "varying counted domain unavailable";
        return failure();
    }
    state->child.loop = recognized.inner;
    state->child.phases = *phases;
    state->child.fragments = state->visits.child.fragments;
    state->child.periodic = state->visits.child.quotient;
    state->trips = domain->trips(state->e());
    auto& out = state->out;
    out.expressions = state->arena;
    out.accessModel = &input.accesses();
    out.gmAliasPolicy = input.memory().gmPolicy();
    for (auto* phase : state->child.phases) {
        auto* op = phase->elementOp;
        out.anchors.push_back({phase, {}, {op->getBlock(), op}, {op->getBlock(), op->getNextNode()}});
        out.occurrenceLoops.push_back(recognized.inner);
        out.outerLoops.push_back({recognized.outer});
    }
    if (!state->build() || !state->selectors(input)) {
        error = state->error;
        return failure();
    }
    // Closures own state. Keep them out of state's prototype to avoid cycles.
    auto result = out;
    result.presence = [state](RegionalEvent x) -> std::optional<Id> {
        auto value = state->present(x);
        return value == RegionExpressions::invalid ? std::nullopt : std::optional<Id>(value);
    };
    result.reachability = [state](RegionalEvent a, RegionalEvent b) { return state->query(a, b); };
    result.referenceBefore = [state](RegionalEvent a, RegionalEvent b) -> std::optional<Id> {
        if (state->present(a) == RegionExpressions::invalid || state->present(b) == RegionExpressions::invalid) {
            return std::nullopt;
        }
        return state->before(a, b);
    };
    result.prepare = [state]() { return state->prepare(); };
    result.prepareWithVisits =
        [state](ArrayRef<scf::ForOp> enclosing) -> FailureOr<std::unique_ptr<PreparedLogicalPlan>> {
        for (auto loop : enclosing) {
            if (!loop->isProperAncestor(state->recognized.outer)) {
                return failure();
            }
        }
        return state->prepare();
    };
    result.capabilities = {true, true, true, true, true};
    result.cost.repeatedRegions = 1;
    result.cost.cells = out.storageBoundary.size();
    result.cost.ports = state->ports.size();
    result.cost.phaseDescriptions = state->visits.startup + state->visits.period;
    result.cost.expressionNodes = state->e().size();
    return result;
}
} // namespace mlir::pto::frontiersynch
