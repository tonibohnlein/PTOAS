// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent concrete event graphs check every site pair around transport.
// Boundary extrema are checked by competing occurrence pairs, including all
// simultaneous envelopes of the same original macro invocation.
#include "PTO/Transforms/FrontierSynch/Structured/BoundedLifetimeRegional.h"
#include "PTO/Transforms/FrontierSynch/FrontierAnalysis.h"
#include "PTO/Transforms/FrontierSynch/Shared/HardwareProtection.h"
#include "PTO/Transforms/FrontierSynch/Shared/SequenceAnalysis.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/CountedLoop.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/RecognitionInternal.h"
#include "BoundedMandatoryTransportChecks.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
std::optional<APInt> scalar(Value value, scf::ForOp loop, uint64_t visit, uint64_t trips, unsigned pattern)
{
    APInt literal;
    if (matchPattern(value, m_ConstantInt(&literal))) { return literal; }
    if (value == loop.getInductionVar()) { return APInt(64, visit); }
    if (value == loop.getUpperBound()) { return APInt(64, trips); }
    auto* op = value.getDefiningOp();
    if (isa_and_nonnull<arith::CmpFOp>(op)) {
        return APInt(1, pattern == 0 ? 0 : pattern == 1 ? 1 : ((visit + pattern) & 1));
    }
    if (!op || op->getNumOperands() != 2) { return std::nullopt; }
    auto a = scalar(op->getOperand(0), loop, visit, trips, pattern);
    auto b = scalar(op->getOperand(1), loop, visit, trips, pattern);
    if (!a || !b) { return std::nullopt; }
    if (isa<arith::AddIOp>(op)) { return *a + *b; }
    if (isa<arith::SubIOp>(op)) { return *a - *b; }
    if (isa<arith::MulIOp>(op)) { return *a * *b; }
    if (auto comparison = dyn_cast<arith::CmpIOp>(op)) {
        return APInt(1, arith::applyCmpPredicate(comparison.getPredicate(), *a, *b));
    }
    if (isa<arith::AndIOp>(op)) { return *a & *b; }
    if (isa<arith::OrIOp>(op)) { return *a | *b; }
    return std::nullopt;
}
bool check(const fs::BoundedLifetimeDemandResult& demands, const fs::BoundedLifetimeRecognition& recognized,
    const fs::PhaseIndex& index, fs::RegionalAnalysis& queries, unsigned trips, unsigned pattern, uint64_t& count)
{
    const unsigned m = recognized.skeleton.phases.size(), payloads = m * trips;
    auto& e = *queries.expressions;
    auto loop = demands.loop;
    std::vector<bool> present(payloads);
    std::map<fs::RegionExpressions::Id, fs::RegionExpressions::Id> bindings;
    bindings.emplace(e.input(loop.getUpperBound()), e.constant(trips));
    std::vector<std::vector<unsigned>> graph(2 * payloads);
    std::map<uint32_t, unsigned> previous;
    DenseMap<const pto::CompoundInstanceElement*, unsigned> positions;
    SmallVector<const pto::CompoundInstanceElement*> phases;
    for (unsigned site = 0; site < m; ++site) {
        positions[recognized.skeleton.phases[site].phase] = site;
        phases.push_back(recognized.skeleton.phases[site].phase);
    }
    auto sameOccurrence = [&](unsigned a, unsigned b) {
        const auto& left = demands.anchors[a % m];
        const auto& right = demands.anchors[b % m];
        if (a / m != b / m || left.phase->elementOp != right.phase->elementOp ||
            left.coordinates.size() != right.coordinates.size()) { return false; }
        return llvm::all_of(llvm::zip(left.coordinates, right.coordinates), [](const auto& pair) {
            const auto& [x, y] = pair;
            return x.loop == y.loop && x.induction == y.induction;
        });
    };
    auto edge = [&](unsigned a, unsigned b) { graph[a].push_back(b); };
    for (unsigned visit = 0; visit < trips; ++visit) {
        for (unsigned site = 0; site < m; ++site) {
            const auto payload = visit * m + site;
            bool active = true;
            for (auto guard = recognized.skeleton.phases[site].guard; guard;
                guard = recognized.skeleton.guards[*guard].parent) {
                const auto& condition = recognized.skeleton.guards[*guard];
                auto value = scalar(condition.condition, loop, visit, trips, pattern);
                if (!value) { llvm::errs() << "bounded oracle cannot evaluate original integer guard\n"; return false; }
                active &= value->getBoolValue() == condition.takeThen;
            }
            present[payload] = active;
            const auto pipe = static_cast<uint32_t>(phases[site]->kPipeValue);
            if (active) {
                edge(2 * payload, 2 * payload + 1);
                if (auto found = previous.find(pipe); found != previous.end()) {
                    edge(2 * found->second, 2 * payload);
                    edge(2 * found->second + 1, 2 * payload + 1);
                }
                previous[pipe] = payload;
            }
            auto presence = fs::regionalPresence(queries, {site, e.constant(visit), fs::PeriodicEventKind::Start});
            if (!presence) { return false; }
            // Every opaque branch at this visit uses the chosen independent
            // oracle bit. Integer predicates are evaluated from the original IR.
            for (auto [id, input] : e.referencedInputs(*presence)) {
                if (!input.getType().isInteger(1)) { continue; }
                const bool bit = pattern == 0 ? 0 : pattern == 1 ? 1 : ((visit + pattern) & 1);
                auto inserted = bindings.emplace(id, e.boolean(bit));
                if (!inserted.second && inserted.first->second != e.boolean(bit)) { return false; }
            }
        }
    }
    auto prerequisites = index.mapPrerequisites(phases);
    if (!prerequisites.error.empty()) { return false; }
    for (unsigned visit = 0; visit < trips; ++visit) {
        for (const auto* list : {&prerequisites.demands, &prerequisites.native}) {
            for (auto requirement : *list) {
                auto a = visit * m + requirement.source, b = visit * m + requirement.target;
                if (present[a] && present[b]) { edge(2 * a + 1, 2 * b); }
            }
        }
    }
    struct Mode { bool read = false, write = false; };
    std::vector<std::map<unsigned, Mode>> cells(recognized.physical->atoms.size());
    for (unsigned visit = 0; visit < trips; ++visit) {
        for (const auto& projection : recognized.physical->accesses) {
            if (projection.phase != visit % recognized.physical->period) { continue; }
            const auto& access = recognized.skeleton.result.accesses[projection.access];
            const auto site = positions.lookup(demands.input->accesses().effects()[access.effect].phase);
            const auto payload = visit * m + site;
            if (!present[payload]) { continue; }
            auto& mode = cells[projection.atom][payload]; mode.read |= access.reads; mode.write |= access.writes;
        }
    }
    for (const auto& cell : cells) {
        for (auto a = cell.begin(); a != cell.end(); ++a) {
            for (auto b = std::next(a); b != cell.end(); ++b) {
                const auto p = static_cast<uint32_t>(phases[a->first % m]->kPipeValue);
                const auto q = static_cast<uint32_t>(phases[b->first % m]->kPipeValue);
                const bool simultaneous = sameOccurrence(a->first, b->first);
                if (simultaneous || fs::ptoStorageProtection().protectsScalar(p, q)) { continue; }
                if ((a->second.write && (b->second.read || b->second.write)) || (a->second.read && b->second.write)) {
                    edge(2 * a->first + 1, 2 * b->first);
                }
            }
        }
    }
    std::vector<llvm::BitVector> reach(2 * payloads, llvm::BitVector(2 * payloads));
    for (unsigned reverse = 0; reverse < graph.size(); ++reverse) {
        const auto source = graph.size() - 1 - reverse;
        if (present[source / 2]) { reach[source].set(source); }
        for (auto target : graph[source]) {
            if (target <= source) { return false; }
            reach[source] |= reach[target];
        }
    }
    // Selector recipes use symbolic tail coordinates. Bind their opaque
    // evaluations at the concrete visit, independently of the selector formula.
    auto bindCoordinate = [&](fs::RegionExpressions::Id ordinal) {
        fs::RegionExpressions::Substitution tripBinding({
            {e.input(loop.getUpperBound()), e.constant(trips)}});
        auto visit = e.constantValue(e.substitute(ordinal, tripBinding));
        if (!visit) { return false; }
        const bool bit = pattern == 0 ? false : pattern == 1 ? true : ((*visit + pattern) & 1);
        for (unsigned site = 0; site < m; ++site) {
            auto presence = fs::regionalPresence(queries, {site, ordinal, fs::PeriodicEventKind::Start});
            if (!presence) { return false; }
            for (auto [id, input] : e.referencedInputs(*presence)) {
                if (!input.getType().isInteger(1)) { continue; }
                auto inserted = bindings.emplace(id, e.boolean(bit));
                if (!inserted.second && inserted.first->second != e.boolean(bit)) { return false; }
            }
        }
        return true;
    };
    const auto horizon = recognized.refresh.span + 1;
    const auto n = fs::CountedLoop::get(loop)->trips(e);
    auto tail = e.select(e.lt(e.constant(horizon), n), e.sub(n, e.constant(horizon)), e.constant(0));
    if (!bindCoordinate(e.sub(n, e.constant(1)))) { return false; }
    for (uint64_t distance = 0; distance < horizon; ++distance) {
        if (!bindCoordinate(e.constant(distance)) || !bindCoordinate(e.add(tail, e.constant(distance)))) {
            return false;
        }
    }
    SmallVector<std::pair<fs::RegionExpressions::Id, fs::RegionExpressions::Id>> replacements;
    for (auto pair : bindings) { replacements.push_back(pair); }
    fs::RegionExpressions::Substitution substitution(replacements);
    auto evaluate = [&](fs::RegionExpressions::Id root) {
        if (pattern < 2) {
            auto constants = replacements;
            for (auto [id, input] : e.referencedInputs(root)) {
                if (input.getType().isInteger(1) && !bindings.count(id)) {
                    constants.emplace_back(id, e.boolean(pattern != 0));
                }
            }
            fs::RegionExpressions::Substitution specialized(constants);
            return e.constantValue(e.substitute(root, specialized));
        }
        return e.constantValue(e.substitute(root, substitution));
    };
    auto selected = [&](const std::vector<fs::RegionalSelector>& alternatives) {
        std::vector<unsigned> result;
        for (const auto& selector : alternatives) {
            auto active = evaluate(selector.present);
            if (!active) { result.push_back(UINT_MAX); continue; }
            if (!*active) { continue; }
            auto ordinal = evaluate(selector.event.ordinal);
            result.push_back(ordinal && *ordinal < trips ? *ordinal * m + selector.event.type : UINT_MAX);
        }
        llvm::sort(result);
        return result;
    };
    auto extrema = [&](const auto& uses, bool last, bool writer, std::optional<uint32_t> pipe,
                       const auto& originalPhases, unsigned sites) {
        std::vector<unsigned> result;
        for (const auto& candidate : uses) {
            const auto candidatePipe = static_cast<uint32_t>(originalPhases[candidate.first % sites]->kPipeValue);
            if (!(writer ? candidate.second.write :
                    candidate.second.read && !candidate.second.write && candidatePipe == *pipe)) { continue; }
            const bool blocked = llvm::any_of(uses, [&](const auto& other) {
                if (sameOccurrence(candidate.first, other.first) ||
                    (last ? other.first <= candidate.first : other.first >= candidate.first)) { return false; }
                const auto otherPipe = static_cast<uint32_t>(originalPhases[other.first % sites]->kPipeValue);
                return other.second.write || (!writer && other.second.read && otherPipe == *pipe);
            });
            if (!blocked) { result.push_back(candidate.first); }
        }
        llvm::sort(result);
        return result;
    };
    if (queries.capabilities.exactSelectors) {
        if (queries.storageBoundary.size() != cells.size()) { return false; }
        for (unsigned cell = 0; cell < cells.size(); ++cell) {
            std::vector<std::pair<unsigned, Mode>> uses(cells[cell].begin(), cells[cell].end());
            const auto& boundary = queries.storageBoundary[cell];
            if (selected(boundary.firstWriters) != extrema(uses, false, true, {}, phases, m) ||
                selected(boundary.lastWriters) != extrema(uses, true, true, {}, phases, m)) {
                llvm::errs() << "bounded writer selector mismatch cell=" << cell << " trips=" << trips
                             << " pattern=" << pattern << "\n";
                for (auto v : selected(boundary.firstWriters)) { llvm::errs() << " first=" << v; }
                for (auto v : extrema(uses, false, true, {}, phases, m)) { llvm::errs() << " expected-first=" << v; }
                for (auto v : selected(boundary.lastWriters)) { llvm::errs() << " last=" << v; }
                for (auto v : extrema(uses, true, true, {}, phases, m)) { llvm::errs() << " expected-last=" << v; }
                llvm::errs() << "\n"; return false;
            }
            for (bool last : {false, true}) {
                const auto& readers = last ? boundary.lastReaders : boundary.firstReaders;
                for (auto* phase : phases) {
                    const auto pipe = static_cast<uint32_t>(phase->kPipeValue);
                    auto found = readers.find(pipe);
                    const std::vector<fs::RegionalSelector> none;
                    if (selected(found == readers.end() ? none : found->second) !=
                        extrema(uses, last, false, pipe, phases, m)) {
                        llvm::errs() << "bounded reader selector mismatch cell=" << cell << "\n"; return false;
                    }
                }
            }
        }
        auto siteExtremum = [&](unsigned site, bool last) {
            std::vector<unsigned> expected;
            for (unsigned i = 0; i < trips; ++i) {
                const auto visit = last ? trips - 1 - i : i;
                if (present[visit * m + site]) { expected.push_back(visit * m + site); break; }
            }
            return expected;
        };
        for (const auto& [site, alternatives] : queries.firstSitePayloads) {
            if (selected(alternatives) != siteExtremum(site, false)) {
                llvm::errs() << "bounded site selector mismatch\n"; return false;
            }
        }
        std::map<std::size_t, std::pair<std::vector<unsigned>, std::vector<unsigned>>> deferred;
        for (const auto& boundary : queries.deferredAccessBoundary) {
            auto& actual = deferred[boundary.effect];
            llvm::append_range(actual.first, selected({boundary.first}));
            llvm::append_range(actual.second, selected({boundary.last}));
        }
        for (const auto& [effect, actual] : deferred) {
            const auto* phase = demands.input->accesses().effects()[effect].phase;
            auto found = positions.find(phase);
            if (found == positions.end() || actual.first != siteExtremum(found->second, false) ||
                actual.second != siteExtremum(found->second, true)) {
                llvm::errs() << "bounded deferred effect selector mismatch\n"; return false;
            }
        }
        for (bool last : {false, true}) {
            const auto& native = last ? queries.lastPayloads : queries.firstPayloads;
            for (const auto& [pipe, alternatives] : native) {
                std::vector<unsigned> expected;
                for (unsigned i = 0; i < payloads; ++i) {
                    const auto payload = last ? payloads - 1 - i : i;
                    if (present[payload] && static_cast<uint32_t>(phases[payload % m]->kPipeValue) == pipe) {
                        expected.push_back(payload); break;
                    }
                }
                if (selected(alternatives) != expected) {
                    llvm::errs() << "bounded native selector mismatch\n";
                    return false;
                }
            }
        }
    }
    for (unsigned site = 0; site < m; ++site) {
        for (unsigned visit = 0; visit <= trips; ++visit) {
            auto presence = fs::regionalPresence(queries, {site, e.constant(visit), fs::PeriodicEventKind::Start});
            auto actual = presence ? evaluate(*presence) : std::nullopt;
            if (!actual || (*actual != 0) != (visit < trips && present[visit * m + site])) {
                llvm::errs() << "bounded presence mismatch\n"; return false;
            }
        }
    }
    SmallVector<unsigned> visits;
    for (unsigned visit = 0; visit < trips; ++visit) {
        if (trips <= 3 || visit < 2 || visit + 3 >= trips) { visits.push_back(visit); }
    }
    for (auto av : visits) {
        for (auto bv : visits) {
            for (unsigned a = 0; a < 2 * m; ++a) {
                for (unsigned b = 0; b < 2 * m; ++b) {
                    // All type/event pairs remain covered; vary visit pairs
                    // without the full Cartesian product on this 60-site IR.
                    if (trips > 1 && (av != (a + b + pattern) % trips ||
                        bv != (a * 7 + b * 3 + pattern) % trips)) { continue; }
                    auto kind = [](unsigned value) {
                        return value & 1 ? fs::PeriodicEventKind::Completion : fs::PeriodicEventKind::Start;
                    };
                    auto actual = fs::regionalReachability(queries,
                        {a / 2, e.constant(av), kind(a)}, {b / 2, e.constant(bv), kind(b)});
                    if (!actual) { return false; }
                    auto value = e.constantValue(e.substitute(*actual, substitution));
                    const bool expected = reach[2 * (av * m) + a].test(2 * (bv * m) + b);
                    if (!value || (*value != 0) != expected) {
                        llvm::errs() << "bounded query oracle mismatch trips=" << trips << " pattern=" << pattern
                            << " source=" << av << ":" << a << " target=" << bv << ":" << b << "\n";
                        return false;
                    }
                    ++count;
                }
            }
        }
    }
    return true;
}
bool checkCoverageCorruption(const fs::BoundedLifetimeDemandResult& demands,
    const fs::BoundedLifetimeRecognition& isolated, unsigned cell, uint64_t horizon)
{
    auto rejects = [&](auto mutate) {
        auto changed = isolated;
        mutate(changed);
        return !fs::detail::physicalWriterCoverageApplies(changed, *demands.input, demands.loop, cell, horizon);
    };
    if (!rejects([](auto& v) { v.physical->coverageInput = nullptr; }) ||
        !rejects([](auto& v) { v.physical->coverageLoop = nullptr; }) ||
        !rejects([](auto& v) { ++v.physical->coveragePeriod; }) ||
        !rejects([](auto& v) { ++v.physical->writerCoverage.front().cell.begin; }) ||
        !rejects([](auto& v) { v.physical->writerCoverage.front().phase = v.physical->period; }) ||
        !rejects([](auto& v) { v.physical->writerCoverage.front().payloads.clear(); }) ||
        !rejects([](auto& v) { v.physical->writerCoverage.front().payloads.push_back(v.skeleton.phases.size()); }) ||
        !rejects([](auto& v) { v.physical->coveragePhases.clear(); }) ||
        !rejects([](auto& v) { v.physical->accesses.clear(); }) ||
        !rejects([](auto& v) { for (auto& access : v.skeleton.result.accesses) { access.writes = false; } })) {
        return false;
    }
    if (!isolated.physical->coverageGuards.empty() &&
        !rejects([](auto& v) { v.physical->coverageGuards.front().takeThen ^= true; })) { return false; }
    return true;
}
bool checkIncompleteCoverage(const fs::BoundedLifetimeDemandResult& demands,
    const fs::BoundedLifetimeRecognition& recognized)
{
    for (const auto& proof : recognized.physical->writerCoverage) {
        if (proof.payloads.empty() || !llvm::any_of(proof.payloads,
                [&](auto payload) { return payload != proof.payloads.front(); })) { continue; }
        auto incomplete = recognized;
        incomplete.physical->writerCoverage.assign(1, proof);
        auto& payloads = incomplete.physical->writerCoverage.front().payloads;
        const auto removed = payloads.front();
        llvm::erase_if(payloads, [&](auto payload) { return payload == removed; });
        if (fs::detail::physicalWriterCoverageApplies(incomplete, *demands.input, demands.loop,
                proof.atom, UINT64_MAX)) { return false; }
        llvm::outs() << "bounded writer coverage: missing complementary writer rejected\n";
        return true;
    }
    llvm::errs() << "bounded coverage fixture lacks a complementary writer witness\n";
    return false;
}
bool checkWriterCoverage(const fs::BoundedLifetimeDemandResult& demands,
    const fs::BoundedLifetimeRecognition& recognized)
{
    if (!recognized.physical || recognized.physical->writerCoverage.empty()) { return false; }
    auto selected = recognized.physical->writerCoverage.front();
    for (const auto& proof : recognized.physical->writerCoverage) {
        if (proof.tail > selected.tail) { selected = proof; }
    }
    if (!selected.tail) { llvm::errs() << "bounded fixture lacks an interior-tail witness\n"; return false; }
    auto isolated = recognized;
    isolated.physical->writerCoverage.assign(1, selected);
    const auto period = isolated.physical->period;
    if (selected.tail > UINT64_MAX - period) { return false; }
    const auto horizon = period + selected.tail;
    auto applies = [&](const auto& value, uint64_t length) {
        return fs::detail::physicalWriterCoverageApplies(value, *demands.input, demands.loop, selected.atom, length);
    };
    if (!applies(isolated, horizon) || applies(isolated, horizon - 1) ||
        !applies(isolated, UINT64_MAX) || !checkCoverageCorruption(demands, isolated, selected.atom, horizon) ||
        !checkIncompleteCoverage(demands, recognized)) {
        return false;
    }
    llvm::outs() << "bounded writer coverage: horizon boundaries and context isolation passed\n";
    return true;
}

std::optional<bool> evaluateMacroComposition(fs::RegionExpressions& e, fs::RegionExpressions::Id value,
    scf::ForOp loop, unsigned trips, bool enabled)
{
    SmallVector<std::pair<fs::RegionExpressions::Id, fs::RegionExpressions::Id>> bindings{
        {e.input(loop.getUpperBound()), e.constant(trips)}};
    for (auto [id, input] : e.referencedInputs(value)) {
        if (input.getType().isInteger(1)) { bindings.emplace_back(id, e.boolean(enabled)); }
    }
    fs::RegionExpressions::Substitution substitution(bindings);
    auto result = e.constantValue(e.substitute(value, substitution));
    if (!result || *result > 1) { return std::nullopt; }
    return *result != 0;
}
std::optional<bool> macroSitePresent(const fs::BoundedLifetimeRecognition& recognized, scf::ForOp loop,
    unsigned site, unsigned visit, unsigned trips, bool enabled)
{
    for (auto guard = recognized.skeleton.phases[site].guard; guard;
        guard = recognized.skeleton.guards[*guard].parent) {
        const auto& condition = recognized.skeleton.guards[*guard];
        // The external bit controls opaque floating predicates only. Integer
        // guards are original program arithmetic at this exact occurrence.
        auto value = scalar(condition.condition, loop, visit, trips, enabled ? 1 : 0);
        if (!value) { return std::nullopt; }
        if (condition.takeThen != value->getBoolValue()) { return false; }
    }
    return true;
}
std::vector<unsigned> macroCrossingSites(const fs::BoundedLifetimeDemandResult& demands,
    const fs::BoundedLifetimeRecognition& recognized, const fs::RegionalAnalysis& outside)
{
    std::vector<unsigned> result;
    for (unsigned site = 0; site < demands.anchors.size(); ++site) {
        const auto* phase = demands.anchors[site].phase;
        if (!phase->elementOp->getName().getStringRef().starts_with("pto.tscatter") &&
            !phase->elementOp->getName().getStringRef().starts_with("pto.tgather")) { continue; }
        for (const auto& projection : recognized.physical->accesses) {
            const auto& access = recognized.skeleton.result.accesses[projection.access];
            if (demands.input->accesses().effects()[access.effect].phase != phase) { continue; }
            const auto& cell = recognized.physical->atoms[projection.atom];
            const bool overlap = llvm::any_of(outside.storageBoundary, [&](const auto& boundary) {
                return pto::sameStorageDomain(cell, boundary.cell) && cell.begin < boundary.cell.end &&
                    boundary.cell.begin < cell.end && !boundary.firstWriters.empty();
            });
            if (overlap) { result.push_back(site); break; }
        }
    }
    return result;
}
bool checkMacroCrossings(fs::SequenceAnalysis& composed, const fs::BoundedLifetimeDemandResult& demands,
    const fs::BoundedLifetimeRecognition& recognized, ArrayRef<unsigned> sites, bool incoming, uint64_t& count)
{
    auto& e = *fs::sequenceExpressions(composed);
    for (unsigned trips : {0U, 1U, 2U, 5U}) {
        for (bool enabled : {false, true}) {
            for (unsigned site : sites) {
                const unsigned visit = incoming || !trips ? 0 : trips - 1;
                const auto ordinal = e.constant(visit);
                fs::SequenceEvent inner{1, site, ordinal, incoming ? fs::PeriodicEventKind::Start :
                    fs::PeriodicEventKind::Completion};
                fs::SequenceEvent outer{incoming ? 0U : 2U, 0, e.constant(0), incoming ?
                    fs::PeriodicEventKind::Completion : fs::PeriodicEventKind::Start};
                auto query = incoming ? fs::sequenceEventReachability(composed, outer, inner) :
                    fs::sequenceEventReachability(composed, inner, outer);
                if (!query) { return false; }
                auto value = evaluateMacroComposition(e, *query, demands.loop, trips, enabled);
                auto present = trips ? macroSitePresent(recognized, demands.loop, site, visit, trips, enabled) :
                    std::optional<bool>(false);
                if (!present) {
                    llvm::errs() << "macro parent oracle cannot evaluate original guard\n";
                    return false;
                }
                const bool expected = *present;
                if (!value || *value != expected) {
                    llvm::errs() << "macro parent crossing mismatch site=" << site << " trips=" << trips
                                 << " incoming=" << incoming << " enabled=" << enabled
                                 << " expected=" << expected << " actual=" << (value ? (*value ? 1 : 0) : -1)
                                 << "\n";
                    return false;
                }
                ++count;
            }
        }
    }
    return true;
}
bool checkMacroComposition(const fs::BoundedLifetimeDemandResult& demands,
    const fs::BoundedLifetimeRecognition& recognized, const fs::PhaseIndex& index, fs::RegionalAnalysis& queries)
{
    auto program = fs::recognizeProgram(demands.function, *demands.input, index);
    if (failed(program)) { return false; }
    std::vector<fs::RegionalAnalysis> outside;
    for (auto [id, node] : llvm::enumerate(program->nodes)) {
        if (node.kind != fs::StructureKind::ExplicitRun || !node.loops.empty() || !node.payloadCount) { continue; }
        auto analysis = fs::analyzeSequenceRegion(demands.function, *demands.input, *program, id,
            queries.expressions, {}, false);
        if (!analysis.error.empty()) { llvm::errs() << analysis.error << "\n"; return false; }
        outside.push_back(fs::sequenceRegionalResult(analysis));
    }
    if (outside.size() != 2 || outside[0].anchors.size() != 1 || outside[1].anchors.size() != 1) { return false; }
    auto incoming = macroCrossingSites(demands, recognized, outside[0]);
    auto outgoing = macroCrossingSites(demands, recognized, outside[1]);
    if (incoming.size() < 2 || outgoing.size() < 2) { return false; }
    auto composed = fs::composeRegionalSequence(demands.function, queries.expressions,
        {outside[0], queries, outside[1]}, false, false);
    if (!composed.error.empty()) { llvm::errs() << composed.error << "\n"; return false; }
    uint64_t count = 0;
    if (!checkMacroCrossings(composed, demands, recognized, incoming, true, count) ||
        !checkMacroCrossings(composed, demands, recognized, outgoing, false, count)) { return false; }
    llvm::outs() << "bounded macro composition comparisons=" << count << "\n";
    return true;
}
} // namespace
bool checkBoundedRegionalQueries(const fs::BoundedLifetimeDemandResult& demands,
    const fs::BoundedLifetimeRecognition& recognized, const fs::PhaseIndex& index, fs::RegionalAnalysis& queries)
{
    auto fail = [](StringRef stage) {
        llvm::errs() << "bounded query check failed: " << stage << "\n";
        return false;
    };
    if (!checkMandatoryTransport()) { return fail("mandatory transport independent controls"); }
    if (!checkWriterCoverage(demands, recognized)) { return fail("writer coverage independent controls"); }
    APInt lower, step;
    auto loop = demands.loop;
    if (!matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) || !lower.isZero() ||
        !matchPattern(loop.getStep(), m_ConstantInt(&step)) || !step.isOne()) {
        llvm::errs() << "bounded concrete fixture requires zero lower bound and unit step\n"; return false;
    }
    auto& e = *queries.expressions;
    fs::RegionalEvent source{0, e.constant(0), fs::PeriodicEventKind::Start};
    auto target = source;
    if (!fs::regionalReachability(queries, source, target)) { return fail("initial reflexive regional query"); }
    // Native same-pipe kinds must retain their small coordinate/presence
    // formula even for symbolic ordinals; no window-reachability substitution
    // is necessary. The full graph oracle below independently checks semantics.
    const auto first = e.input(loop.getInductionVar());
    const auto next = e.add(first, e.constant(1));
    const auto beforeNative = e.size();
    for (unsigned site = 0; site < demands.window.sites; ++site) {
        if (!demands.unconditional[site]) { continue; }
        for (auto kinds : {std::make_pair(fs::PeriodicEventKind::Start, fs::PeriodicEventKind::Start),
                           std::make_pair(fs::PeriodicEventKind::Start, fs::PeriodicEventKind::Completion),
                           std::make_pair(fs::PeriodicEventKind::Completion, fs::PeriodicEventKind::Completion)}) {
            auto native = fs::regionalReachability(queries, {site, first, kinds.first}, {site, next, kinds.second});
            if (!native) {
                llvm::errs() << "bounded native query unavailable site=" << site
                             << " kinds=" << unsigned(kinds.first) << ":" << unsigned(kinds.second) << "\n";
                return false;
            }
        }
    }
    if (e.size() - beforeNative > 16) {
        llvm::errs() << "bounded native query expanded its window: " << e.size() - beforeNative << " nodes\n";
        return false;
    }
    // The production Delta fixture's unconditional tadd and tstore are linked
    // through later native V completions and the final conversion's RAW edge.
    // The transport proof must cover every physical residue before replacing
    // this symbolic same-visit C->I query. The raw-conflict oracle below checks
    // actual reachability independently, including absent visits and branches.
    auto originalSite = [&](StringRef name) -> std::optional<uint32_t> {
        Operation* original = nullptr;
        for (auto& operation : *loop.getBody()) {
            if (operation.getName().getStringRef() != name) { continue; }
            if (original) { return std::nullopt; }
            original = &operation;
        }
        if (!original) { return std::nullopt; }
        std::optional<uint32_t> site;
        for (auto [id, anchor] : llvm::enumerate(demands.anchors)) {
            if (!anchor.phase || anchor.phase->elementOp != original) { continue; }
            if (site || id > UINT32_MAX) { return std::nullopt; }
            site = static_cast<uint32_t>(id);
        }
        return site;
    };
    // Resolve exact original operations through this export's phase inventory;
    // global, local and macro-envelope site numbers are not interchangeable.
    const auto producer = originalSite("pto.tadd"), consumer = originalSite("pto.tstore");
    if (!producer || !consumer || *producer >= demands.unconditional.size() ||
        *consumer >= demands.unconditional.size() || !demands.unconditional[*producer] ||
        !demands.unconditional[*consumer]) {
        return fail("production mandatory-query source/target fixture premises");
    }
    const auto beforeMandatory = e.size();
    const fs::RegionalEvent mandatorySource{*producer, first, fs::PeriodicEventKind::Completion};
    const fs::RegionalEvent mandatoryTarget{*consumer, first, fs::PeriodicEventKind::Start};
    auto mandatory = fs::regionalReachability(queries, mandatorySource, mandatoryTarget);
    if (!mandatory || e.size() - beforeMandatory > 16) {
        llvm::errs() << "bounded mandatory query expanded its window\n"; return false;
    }
    const auto retainedSize = e.size();
    for (unsigned retry = 0; retry < 32; ++retry) {
        if (fs::regionalReachability(queries, mandatorySource, mandatoryTarget) != mandatory ||
            e.size() != retainedSize) { return fail("mandatory query cache identity or construction growth"); }
    }
    for (uint64_t count : {0U, 1U, 2U, 5U}) {
        for (uint64_t visit = 0; visit <= count; ++visit) {
            fs::RegionExpressions::Substitution bindings({{first, e.constant(visit)},
                {e.input(loop.getUpperBound()), e.constant(count)}});
            const auto actual = e.constantValue(e.substitute(*mandatory, bindings));
            if (actual != uint64_t(visit < count)) {
                llvm::errs() << "bounded mandatory query evaluation trips=" << count << " visit=" << visit
                             << " actual=" << (actual ? std::to_string(*actual) : "unavailable")
                             << " expected=" << uint64_t(visit < count) << "\n";
                return false;
            }
        }
    }
    source.visits.push_back(e.constant(0));
    if (fs::regionalReachability(queries, source, target)) { return fail("unexpected extra-visit query accepted"); }
    source.visits.clear(); source.kind = static_cast<fs::PeriodicEventKind>(255);
    target.ordinal = e.constant(2 * recognized.refresh.span + 10);
    if (fs::regionalReachability(queries, source, target)) { return fail("invalid event-kind query accepted"); }
    // A read-only cell used only under an arbitrary branch has no bounded
    // fringe certificate. Refusing that stronger export must preserve queries.
    auto missing = recognized;
    bool injectedFailure = false;
    for (const auto& projection : recognized.physical->accesses) {
        const auto& access = recognized.skeleton.result.accesses[projection.access];
        const auto* phase = demands.input->accesses().effects()[access.effect].phase;
        auto site = llvm::find_if(recognized.skeleton.phases, [&](const auto& value) { return value.phase == phase; });
        if (site == recognized.skeleton.phases.end() || !site->guard || !access.reads || access.writes ||
            !isa_and_nonnull<arith::CmpFOp>(recognized.skeleton.guards[*site->guard].condition.getDefiningOp())) {
            continue;
        }
        auto extra = projection;
        extra.atom = missing.physical->atoms.size();
        missing.physical->atoms.push_back(recognized.physical->atoms[projection.atom]);
        missing.physical->accesses.push_back(extra);
        std::string error;
        llvm::errs() << "bounded negative selector begin\n";
        auto failedSelectors = fs::boundedLifetimeRegionalSelectors(demands, missing, queries, error);
        if (succeeded(failedSelectors) || error.empty() || !e.constructionError().empty()) {
            return fail("negative selector export did not preserve the precise refusal contract");
        }
        auto retained = fs::regionalReachability(queries,
            {0, e.constant(0), fs::PeriodicEventKind::Start}, {0, e.constant(0), fs::PeriodicEventKind::Completion});
        if (!retained) { return fail("negative selector export lost retained reachability"); }
        // Traverse every input, including abandoned proof expressions.
        SmallVector<fs::RegionExpressions::Id> roots;
        for (fs::RegionExpressions::Id root = 0; root < e.size(); ++root) { roots.push_back(root); }
        fs::RegionExpressions ownershipProbe;
        if (failed(ownershipProbe.import(e, roots))) { return fail("negative selector export left unowned inputs"); }
        llvm::errs() << "bounded negative selector checked\n";
        injectedFailure = true;
        break;
    }
    if (!injectedFailure) { llvm::errs() << "bounded failed-selector injection did not execute\n"; return false; }
    uint64_t count = 0;
    for (unsigned trips : {0u, 1u, 2u, 3u, static_cast<unsigned>(2 * recognized.refresh.span + 3),
            static_cast<unsigned>(2 * recognized.refresh.span + 4)}) {
        for (unsigned pattern = 0; pattern < 4; ++pattern) {
            if (!check(demands, recognized, index, queries, trips, pattern, count)) {
                llvm::errs() << "bounded independent graph check failed trips=" << trips
                             << " pattern=" << pattern << " completed-comparisons=" << count << "\n";
                return false;
            }
        }
    }
    llvm::outs() << "bounded-query independent event comparisons=" << count << "\n";
    return true;
}

bool checkBoundedMacroSelectors(const fs::BoundedLifetimeDemandResult& demands,
    const fs::BoundedLifetimeRecognition& recognized, const fs::PhaseIndex& index, fs::RegionalAnalysis& queries)
{
    if (auto expected = demands.function->getAttrOfType<BoolAttr>("test.coordinate_substitution_stable")) {
        if (queries.capabilities.substitutionStableQueries != expected.getValue()) {
            llvm::errs() << "bounded coordinate substitution qualification mismatch\n";
            return false;
        }
    }
    if (!queries.capabilities.exactSelectors || !recognized.physical) { return false; }
    bool simultaneousWriters = false, simultaneousReaderWriter = false;
    const auto& effects = demands.input->accesses().effects();
    for (const auto& a : recognized.physical->accesses) {
        for (const auto& b : recognized.physical->accesses) {
            if (a.atom != b.atom || a.phase != b.phase || a.access == b.access) { continue; }
            const auto& x = recognized.skeleton.result.accesses[a.access];
            const auto& y = recognized.skeleton.result.accesses[b.access];
            const auto* first = effects[x.effect].phase;
            const auto* second = effects[y.effect].phase;
            if (first == second || first->elementOp != second->elementOp) { continue; }
            simultaneousWriters |= x.writes && y.writes;
            simultaneousReaderWriter |= x.reads && !x.writes && y.writes;
        }
    }
    if (!simultaneousWriters ||
        (demands.function->hasAttr("test.bounded_macro_readers") && !simultaneousReaderWriter)) {
        llvm::errs() << "bounded macro fixture did not expose simultaneous boundary accesses\n";
        return false;
    }
    uint64_t comparisons = 0;
    for (unsigned trips : {0U, 1U, 2U, 5U}) {
        for (unsigned pattern = 0; pattern < 4; ++pattern) {
            if (!check(demands, recognized, index, queries, trips, pattern, comparisons)) { return false; }
        }
    }
    llvm::outs() << "bounded macro selectors: simultaneous writers=" << simultaneousWriters
                 << " reader-writer=" << simultaneousReaderWriter << " comparisons=" << comparisons << "\n";
    return checkMacroComposition(demands, recognized, index, queries);
}

namespace {
bool checkBoundedReadinessComposition(std::shared_ptr<fs::BoundedLifetimeDemandResult> demands,
    const fs::BoundedLifetimeRecognition& recognized, const fs::PhaseIndex& index)
{
    auto arena = std::make_shared<fs::RegionExpressions>();
    std::string error;
    auto queries = fs::boundedLifetimeRegionalQueries(demands, recognized, index, arena, error);
    if (failed(queries)) { llvm::errs() << error << "\n"; return false; }
    auto bounded = fs::boundedLifetimeRegionalSelectors(*demands, recognized, *queries, error);
    if (failed(bounded)) { llvm::errs() << error << "\n"; return false; }
    auto program = fs::recognizeProgram(demands->function, *demands->input, index);
    if (failed(program)) { return false; }
    std::vector<fs::RegionalAnalysis> outside;
    for (auto [id, node] : llvm::enumerate(program->nodes)) {
        if (node.kind != fs::StructureKind::ExplicitRun || !node.loops.empty() || !node.payloadCount) { continue; }
        auto child = fs::analyzeSequenceRegion(demands->function, *demands->input, *program, id, arena, {}, false);
        if (!child.error.empty()) { llvm::errs() << child.error << "\n"; return false; }
        outside.push_back(fs::sequenceRegionalResult(child));
    }
    if (outside.size() != 2 || outside[0].anchors.size() != 1 || outside[1].anchors.size() != 1) { return false; }
    auto composed = fs::composeRegionalSequence(demands->function, arena,
        {outside[0], *bounded, outside[1]}, true, false);
    if (!composed.error.empty()) { llvm::errs() << composed.error << "\n"; return false; }
    auto& e = *arena;
    // Independent facts for the fixture: seed reads a disjoint tile on S.
    // S native start/completion order reaches the first S write, but completion
    // does not gate its start. The SSA seed does gate the following V tadds.
    // The final S write conflicts with the suffix V tadds in the original tile.
    struct Expected { fs::RegionExpressions::Id query; bool whenPresent; };
    std::vector<Expected> expected;
    for (auto sourceKind : {fs::PeriodicEventKind::Start, fs::PeriodicEventKind::Completion}) {
        for (auto targetKind : {fs::PeriodicEventKind::Start, fs::PeriodicEventKind::Completion}) {
            for (unsigned site : {0U, 1U}) {
                auto answer = fs::sequenceEventReachability(composed,
                    {0, 0, e.constant(0), sourceKind}, {1, site, e.constant(0), targetKind});
                if (!answer) { return false; }
                expected.push_back({*answer, site == 1 || sourceKind == fs::PeriodicEventKind::Start ||
                    targetKind == fs::PeriodicEventKind::Completion});
            }
            auto exit = fs::sequenceEventReachability(composed,
                {1, 3, e.constant(0), sourceKind}, {2, 0, e.constant(0), targetKind});
            if (!exit) { return false; }
            expected.push_back({*exit, true});
        }
    }
    for (unsigned trips : {0U, 1U, 2U, 5U}) {
        for (auto answer : expected) {
            auto actual = evaluateMacroComposition(e, answer.query, demands->loop, trips, true);
            if (!actual || *actual != (trips && answer.whenPresent)) { return false; }
        }
    }
    llvm::outs() << "bounded-prerequisites: 48 composed event-kind/readiness checks passed\n";
    return true;
}
} // namespace

// The storage refresh certificate must not erase original SSA obligations at
// the loop boundary. Compare the reverse export with the forward edge inventory.
LogicalResult checkBoundedPrerequisiteScope(func::FuncOp function, pto::GMAliasPolicy policy)
{
    pto::SyncInput input(policy);
    fs::PhaseIndex index;
    if (failed(input.build(function)) || failed(index.build(function, input))) { return failure(); }
    auto loops = function.getOps<scf::ForOp>();
    if (loops.empty()) { return failure(); }
    auto loop = *loops.begin();
    for (const auto* phase : input.instructions()) {
        std::vector<fs::ValuePrerequisite> expected;
        function.walk([&](Operation* operation) {
            for (const auto& edge : index.prerequisitesFor(operation)) {
                if (edge.producer == phase) { expected.push_back(edge); }
            }
        });
        auto actual = index.prerequisitesFrom(phase);
        if (actual.size() != expected.size() || actual.data() != index.prerequisitesFrom(phase).data()) {
            return function.emitError("prerequisite reverse inventory was rebuilt or lost edges");
        }
        for (const auto& edge : expected) {
            if (llvm::none_of(actual, [&](const auto& candidate) {
                    return candidate.consumer == edge.consumer && candidate.native == edge.native &&
                        candidate.directSSA == edge.directSSA;
                })) { return function.emitError("prerequisite reverse inventory changed an edge"); }
        }
    }
    auto recognized = fs::recognizeBoundedLifetime(loop, index, input);
    if (function->hasAttr("test.carried_prerequisite") || function->hasAttr("test.escaping_prerequisite")) {
        if (recognized.skeleton.result.state == fs::RecognitionState::Applicable ||
            llvm::none_of(recognized.skeleton.result.diagnostics, [](const auto& diagnostic) {
                return diagnostic.issue == fs::RecognitionIssue::LoopCarriedState;
            })) {
            return function.emitError("carried prerequisite acquired an unsupported local span");
        }
        llvm::outs() << (function->hasAttr("test.escaping_prerequisite") ?
            "bounded-prerequisites: escaping result requires an exit occurrence map\n" :
            "bounded-prerequisites: carried obligation retained\n");
        return success();
    }
    if (recognized.skeleton.result.state != fs::RecognitionState::Applicable ||
        !recognized.prerequisites || recognized.prerequisites->incoming.empty() ||
        (recognized.prerequisites->local.native.empty() && recognized.prerequisites->local.demands.empty())) {
        return function.emitError("fixture lacks qualified local and incoming prerequisites");
    }
    auto arena = std::make_shared<fs::RegionExpressions>();
    std::string error;
    auto demands = fs::analyzeBoundedLifetimeRegion(function, loop, index, input, recognized, error, arena);
    if (failed(demands) || (*demands)->prerequisites != recognized.prerequisites) {
        return function.emitError("bounded construction did not retain prerequisite ownership: " + error);
    }
    if (!checkBoundedReadinessComposition(*demands, recognized, index)) {
        return function.emitError("bounded incoming/outgoing composition lost readiness");
    }
    const auto size = arena->size();
    auto foreign = recognized;
    auto proof = std::make_shared<fs::BoundedPrerequisiteCertificate>(*recognized.prerequisites);
    proof->input = nullptr; foreign.prerequisites = std::move(proof);
    if (succeeded(fs::analyzeBoundedLifetimeRegion(function, loop, index, input, foreign, error, arena)) ||
        error.find("local-prerequisite certificate") == std::string::npos || arena->size() != size) {
        return function.emitError("foreign bounded certificate was consumed or constructed expressions");
    }
    fs::PhaseIndex otherScope;
    if (failed(otherScope.build(function, input)) ||
        succeeded(fs::analyzeBoundedLifetimeRegion(function, loop, otherScope, input, recognized, error, arena)) ||
        error.find("local-prerequisite certificate") == std::string::npos || arena->size() != size) {
        return function.emitError("same-phase different prerequisite scope was consumed");
    }
    if (succeeded(fs::boundedLifetimeRegionalQueries(*demands, recognized, otherScope, arena, error)) ||
        error.find("local-prerequisite certificate") == std::string::npos || arena->size() != size) {
        return function.emitError("foreign prerequisite scope reached global-query construction");
    }
    if (succeeded(fs::prepareBoundedLifetimeLogicalResult(*demands, error, true)) ||
        error.find("whole invocation") == std::string::npos) {
        return function.emitError("regional refresh was promoted to whole-invocation coverage");
    }
    const auto previousScope = index.prerequisiteScope();
    if (failed(index.build(function, input)) || index.prerequisiteScope() == previousScope ||
        succeeded(fs::analyzeBoundedLifetimeRegion(function, loop, index, input, recognized, error, arena)) ||
        error.find("local-prerequisite certificate") == std::string::npos || arena->size() != size) {
        return function.emitError("rebuilt prerequisite scope accepted a stale certificate");
    }
    llvm::outs() << "bounded-prerequisites: local-map retained, incoming separate, foreign scope refused\n";
    return success();
}
