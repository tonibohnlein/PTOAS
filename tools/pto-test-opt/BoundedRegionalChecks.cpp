// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent concrete event graphs check every site pair around transport.
#include "PTO/Transforms/FrontierSynch/Structured/BoundedLifetimeRegional.h"
#include "PTO/Transforms/FrontierSynch/FrontierAnalysis.h"
#include "PTO/Transforms/FrontierSynch/Shared/HardwareProtection.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/CountedLoop.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/RecognitionInternal.h"
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
                const bool simultaneous = a->first / m == b->first / m &&
                    phases[a->first % m]->elementOp == phases[b->first % m]->elementOp;
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
    auto extrema = [](const auto& uses, bool last, bool writer, std::optional<uint32_t> pipe,
                      const auto& phases, unsigned m) {
        std::vector<unsigned> result;
        for (unsigned i = 0; i < uses.size(); ++i) {
            const auto& use = uses[last ? uses.size() - 1 - i : i];
            if (writer ? use.second.write : use.second.read && !use.second.write &&
                static_cast<uint32_t>(phases[use.first % m]->kPipeValue) == *pipe) {
                result.push_back(use.first); break;
            }
            if (!writer && use.second.write) { break; }
        }
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
} // namespace
bool checkBoundedRegionalQueries(const fs::BoundedLifetimeDemandResult& demands,
    const fs::BoundedLifetimeRecognition& recognized, const fs::PhaseIndex& index, fs::RegionalAnalysis& queries)
{
    if (!checkWriterCoverage(demands, recognized)) { return false; }
    APInt lower, step;
    auto loop = demands.loop;
    if (!matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) || !lower.isZero() ||
        !matchPattern(loop.getStep(), m_ConstantInt(&step)) || !step.isOne()) {
        llvm::errs() << "bounded concrete fixture requires zero lower bound and unit step\n"; return false;
    }
    auto& e = *queries.expressions;
    fs::RegionalEvent source{0, e.constant(0), fs::PeriodicEventKind::Start};
    auto target = source;
    if (!fs::regionalReachability(queries, source, target)) { return false; }
    source.visits.push_back(e.constant(0));
    if (fs::regionalReachability(queries, source, target)) { return false; }
    source.visits.clear(); source.kind = static_cast<fs::PeriodicEventKind>(255);
    target.ordinal = e.constant(2 * recognized.refresh.span + 10);
    if (fs::regionalReachability(queries, source, target)) { return false; }
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
        if (succeeded(failedSelectors) || error.empty() || !e.constructionError().empty()) { return false; }
        auto retained = fs::regionalReachability(queries,
            {0, e.constant(0), fs::PeriodicEventKind::Start}, {0, e.constant(0), fs::PeriodicEventKind::Completion});
        if (!retained) { return false; }
        // Traverse every input, including abandoned proof expressions.
        SmallVector<fs::RegionExpressions::Id> roots;
        for (fs::RegionExpressions::Id root = 0; root < e.size(); ++root) { roots.push_back(root); }
        fs::RegionExpressions ownershipProbe;
        if (failed(ownershipProbe.import(e, roots))) { return false; }
        llvm::errs() << "bounded negative selector checked\n";
        injectedFailure = true;
        break;
    }
    if (!injectedFailure) { llvm::errs() << "bounded failed-selector injection did not execute\n"; return false; }
    uint64_t count = 0;
    for (unsigned trips : {0u, 1u, 2u, 3u, static_cast<unsigned>(2 * recognized.refresh.span + 3),
            static_cast<unsigned>(2 * recognized.refresh.span + 4)}) {
        for (unsigned pattern = 0; pattern < 4; ++pattern) {
            if (!check(demands, recognized, index, queries, trips, pattern, count)) { return false; }
        }
    }
    llvm::outs() << "bounded-query independent event comparisons=" << count << "\n";
    return true;
}
