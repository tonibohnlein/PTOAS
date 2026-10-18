// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Entry/exit selector certificates are independent of global query transport.
// Macro envelopes share one occurrence boundary: none suppresses a sibling.
#include "../Shared/GuardedStorageScan.h"
#include "PTO/Transforms/FrontierSynch/Structured/BoundedLifetimeRegional.h"
#include "../Shared/CountedLoop.h"
#include "../Shared/RecognitionInternal.h"
#include "PTO/Transforms/FrontierSynch/Shared/ModularScalar.h"
#include "../../InsertSync/SyncScalarEvolution.h"
#include "mlir/IR/Matchers.h"
#include <set>
#include <tuple>
namespace mlir::pto::frontiersynch {
namespace {
using Id = RegionExpressions::Id;
struct Use {
    RegionalEvent event;
    Id read, write;
    uint32_t pipe;
};
class Selectors {
public:
    Selectors(
        const BoundedLifetimeDemandResult& demands, const BoundedLifetimeRecognition& recognized,
        RegionalAnalysis& result, std::string& error)
        : demands(demands),
          recognized(recognized),
          out(result),
          e(*result.expressions),
          error(error),
          horizon(recognized.refresh.span + 1),
          domain(*CountedLoop::get(demands.loop)),
          trips(domain.trips(e))
    {
        e.retainInputOwner(symbols);
        proofOrdinal = symbols->addArgument(IndexType::get(demands.function->getContext()), demands.loop->getLoc());
    }
    bool run()
    {
        if (!out.capabilities.exactQueries || !recognized.physical) {
            return fail("selectors need exact query context");
        }
        if (!nativeBoundaries() || !physicalBoundaries() || !deferredBoundaries()) {
            return false;
        }
        out.capabilities.completeStorageModel = true;
        out.capabilities.exactSelectors = true;
        out.capabilities.contextualGuards = true;
        out.cost.cells = out.storageBoundary.size();
        out.cost.expressionNodes = e.size();
        auto previousPresence = out.presence;
        auto previousQuery = out.reachability;
        auto owner = symbols;
        out.presence = [previousPresence, owner](RegionalEvent event) { return previousPresence(event); };
        out.reachability = [previousQuery, owner](RegionalEvent a, RegionalEvent b) { return previousQuery(a, b); };
        return e.constructionError().empty();
    }

private:
    const BoundedLifetimeDemandResult& demands;
    const BoundedLifetimeRecognition& recognized;
    RegionalAnalysis& out;
    RegionExpressions& e;
    std::string& error;
    uint64_t horizon;
    CountedLoop domain;
    Id trips;
    std::shared_ptr<Block> symbols = std::make_shared<Block>();
    std::map<Id, bool> emptiness;
    Value proofOrdinal;
    bool nativeBoundaries()
    {
        const auto m = demands.anchors.size();
        for (unsigned site = 0; site < m; ++site) {
            const auto pipe = static_cast<uint32_t>(demands.anchors[site].phase->kPipeValue);
            auto first = at(site, e.constant(0)), last = at(site, e.sub(trips, e.constant(1)));
            if (!first || !last) {
                return false;
            }
            // Every participating pipe has an unconditional occurrence each
            // visit (checked by the transport provider). Its boundary is local
            // to the first/last visit even when other sites are conditional.
            out.firstPayloads[pipe].push_back(*first);
            out.lastPayloads[pipe].push_back(*last);
        }
        trimNative(out.firstPayloads, false);
        trimNative(out.lastPayloads, true);
        return true;
    }
    bool physicalBoundaries()
    {
        for (unsigned cell = 0; cell < recognized.physical->atoms.size(); ++cell) {
            if (!storage(cell)) {
                return false;
            }
        }
        return true;
    }
    bool deferredBoundaries()
    {
        const auto m = demands.anchors.size();
        for (unsigned site = 0; site < m; ++site) {
            for (auto effect : demands.input->accesses().effectsFor(demands.anchors[site].phase)) {
                bool physical = false;
                for (const auto& projection : recognized.physical->accesses) {
                    const auto& access = recognized.skeleton.result.accesses[projection.access];
                    physical |= access.effect == effect || llvm::is_contained(access.effects, effect);
                }
                if (physical) {
                    continue;
                }
                const auto& modeled = demands.input->accesses().effects()[effect];
                if (!modeled.memory || modeled.memory->scope != AddressSpace::GM) {
                    return fail(
                        "nonphysical local effect " + std::to_string(effect) + " has no complete selector certificate");
                } // Complete byte atoms carry these crossings.
                // Keep a discharged effect's true occurrence extrema. A finite
                // fringe is accepted only after proving confinement for this
                // particular type, including every opaque predicate evaluation.
                auto head = siteFringe(site, false), tail = siteFringe(site, true);
                if (!confined(site, head, tail)) {
                    return false;
                }
                auto first = extrema(head, false), last = extrema(tail, true);
                for (auto firstAlternative : first) {
                    for (auto lastAlternative : last) {
                        auto a = firstAlternative;
                        auto b = lastAlternative;
                        auto present = e.land(a.present, b.present);
                        a.present = present;
                        b.present = present;
                        out.deferredAccessBoundary.push_back({effect, a, b, false});
                    }
                }
                out.firstSitePayloads[site] = first;
            }
        }
        return true;
    }
    bool fail(const std::string& text)
    {
        error = text;
        return false;
    }
    std::optional<RegionalSelector> at(unsigned site, Id ordinal)
    {
        RegionalEvent event{site, ordinal, PeriodicEventKind::Start};
        auto present = regionalPresence(out, event);
        if (!present) {
            fail("bounded selector has no exact occurrence predicate");
            return std::nullopt;
        }
        return RegionalSelector{event, *present};
    }
    Id tailBegin()
    {
        return e.select(e.lt(e.constant(horizon), trips), e.sub(trips, e.constant(horizon)), e.constant(0));
    }
    std::vector<RegionalSelector> siteFringe(unsigned site, bool tail)
    {
        std::vector<RegionalSelector> result;
        auto begin = tail ? tailBegin() : e.constant(0);
        for (uint64_t d = 0; d < horizon; ++d) {
            auto selector = at(site, e.add(begin, e.constant(d)));
            if (selector) {
                result.push_back(*selector);
            }
        }
        return result;
    }
    std::vector<RegionalSelector> extrema(const std::vector<RegionalSelector>& candidates, bool last)
    {
        std::vector<RegionalSelector> result;
        auto absent = e.boolean(true);
        for (unsigned index = 0; index < candidates.size(); ++index) {
            auto selected = candidates[last ? candidates.size() - 1 - index : index];
            auto present = selected.present;
            selected.present = e.land(absent, present);
            absent = e.land(absent, e.lnot(present));
            if (e.constantValue(selected.present) != 0) {
                result.push_back(selected);
            }
        }
        return result;
    }
    void trimNative(std::map<uint32_t, std::vector<RegionalSelector>>& selectors, bool last)
    {
        for (auto& [pipe, candidates] : selectors) {
            candidates = extrema(candidates, last);
        }
    }
    FailureOr<IntegerSystem> proofDomain(ArrayRef<std::pair<Id, Value>> referenced, Value ordinal)
    {
        std::vector<IntegerConstraint> rows;
        mlir::pto::detail::ScalarEvolution ranges(demands.function->getContext(), demands.loop);
        for (auto [column, reference] : llvm::enumerate(referenced)) {
            auto input = reference.second;
            if (input.getType().isInteger(1)) {
                continue;
            }
            auto range = input == ordinal ? std::optional<std::pair<int64_t, int64_t>>({0, domain.maximumOrdinal}) :
                                            ranges.signedRange(input);
            if (!range) {
                continue;
            }
            if (input != ordinal) {
                ModularScalar representation(e, demands.loop);
                range = representation.inputRange(input, *range);
                if (!range) {
                    continue;
                }
            }
            IntegerConstraint row;
            row.coefficients.resize(referenced.size());
            row.coefficients[column] = BoundInteger(1);
            row.bound = BoundInteger(range->second);
            rows.push_back(row);
            row.coefficients[column] = BoundInteger(-1);
            row.bound = -BoundInteger(range->first);
            rows.push_back(row);
        }
        return IntegerSystem::create(referenced.size(), rows);
    }
    bool empty(Id predicate, Value ordinal)
    {
        if (auto value = e.constantValue(predicate)) {
            return *value == 0;
        }
        if (auto found = emptiness.find(predicate); found != emptiness.end()) {
            return found->second;
        }
        // Preserve shared Boolean structure before requesting a distributed
        // integer relation. This sufficient proof treats arithmetic atoms as
        // opaque; an inconclusive result leaves exact integer lowering available.
        ++out.cost.implicationChecks;
        if (e.implies(predicate, e.boolean(false))) {
            emptiness.emplace(predicate, true);
            return true;
        }
        auto referenced = e.referencedInputs(predicate);
        SmallVector<Id> variables;
        for (auto [id, input] : referenced) { (void)input; variables.push_back(id); }
        auto context = proofDomain(referenced, ordinal);
        if (failed(context)) {
            return false;
        }
        std::string diagnostic;
        RegionExpressions::RelationCost cost;
        auto pieces = e.integerRelation(predicate, variables, *context, diagnostic, &cost);
        ++out.cost.implicationChecks;
        if (failed(pieces)) {
            return false;
        }
        auto result = llvm::all_of(*pieces, [](const IntegerSystem& piece) { return piece.isEmpty(); });
        emptiness.emplace(predicate, result);
        return result;
    }
    bool confined(unsigned site, const std::vector<RegionalSelector>& head, const std::vector<RegionalSelector>& tail)
    {
        auto ordinal = proofOrdinal;
        auto any = at(site, e.input(ordinal));
        if (!any) {
            return false;
        }
        for (const auto* fringe : {&head, &tail}) {
            auto some = e.boolean(false);
            for (auto selector : *fringe) {
                some = e.lor(some, selector.present);
            }
            if (!empty(e.land(any->present, e.lnot(some)), ordinal)) {
                return fail("effect occurrence extrema are not certified within the bounded entry/exit fringe");
            }
        }
        return true;
    }
    std::vector<Use> uses(unsigned cell, Id begin)
    {
        std::vector<Use> result;
        std::map<unsigned, std::pair<Id, Id>> patterns;
        const auto& physical = *recognized.physical;
        for (uint64_t distance = 0; distance < horizon; ++distance) {
            auto ordinal = e.add(begin, e.constant(distance));
            patterns.clear();
            for (const auto& projection : physical.accesses) {
                if (projection.atom != cell) {
                    continue;
                }
                const auto& access = recognized.skeleton.result.accesses[projection.access];
                const auto* phase = demands.input->accesses().effects()[access.effect].phase;
                auto found = llvm::find_if(demands.anchors, [&](const auto& anchor) { return anchor.phase == phase; });
                if (found == demands.anchors.end()) {
                    continue;
                }
                auto site = found - demands.anchors.begin();
                auto selected = at(site, ordinal);
                if (!selected) {
                    continue;
                }
                auto active = e.land(
                    selected->present, e.eq(e.rem(ordinal, e.constant(physical.period)), e.constant(projection.phase)));
                auto inserted = patterns.try_emplace(site, e.boolean(false), e.boolean(false));
                if (access.reads) {
                    inserted.first->second.first = e.lor(inserted.first->second.first, active);
                }
                if (access.writes) {
                    inserted.first->second.second = e.lor(inserted.first->second.second, active);
                }
            }
            for (auto [site, modes] : patterns) {
                result.push_back(
                    {{site, ordinal, PeriodicEventKind::Start},
                     modes.first,
                     modes.second,
                     static_cast<uint32_t>(demands.anchors[site].phase->kPipeValue)});
            }
        }
        return result;
    }
    bool coverage(unsigned cell, std::optional<uint32_t> readerPipe)
    {
        // A full window contains the producer-certified writer phase before
        // its interior tail. Keep original guards on the actual fringe events.
        if (!readerPipe && detail::physicalWriterCoverageApplies(
                recognized, *demands.input, demands.loop, cell, horizon)) {
            return true;
        }
        // A complete unconditional physical orbit proves coverage directly.
        // Every visit in the quantified full window exists; a window at least
        // one period long includes every residue, independently of its start.
        const auto& physical = *recognized.physical;
        if (horizon >= physical.period) {
            bool covered = false;
            for (const auto& projection : physical.accesses) {
                if (projection.atom != cell) {
                    continue;
                }
                const auto& access = recognized.skeleton.result.accesses[projection.access];
                const auto* phase = demands.input->accesses().effects()[access.effect].phase;
                auto found = llvm::find_if(demands.anchors, [&](const auto& anchor) { return anchor.phase == phase; });
                if (found == demands.anchors.end()) {
                    continue;
                }
                const auto site = found - demands.anchors.begin();
                covered |= demands.unconditional[site] &&
                           (readerPipe ? access.reads && static_cast<uint32_t>(phase->kPipeValue) == *readerPipe :
                                         access.writes);
            }
            if (covered) {
                return true;
            }
        }
        // A uniform refresh/use proof, independently checked for every absolute
        // physical phase. Source guards remain arbitrary distinct evaluations.
        auto begin = e.input(proofOrdinal);
        auto choices = uses(cell, begin);
        auto some = e.boolean(false);
        for (auto use : choices) {
            if (!readerPipe || use.pipe == *readerPipe) {
                some = e.lor(some, readerPipe ? use.read : use.write);
            }
        }
        // Quantify the actual window start, not just one representative of an
        // address phase. Guards can depend on the absolute iteration ordinal.
        auto longEnough = e.le(e.add(begin, e.constant(horizon)), trips);
        if (!empty(e.land(longEnough, e.lnot(some)), proofOrdinal)) {
            return fail(
                readerPipe ? "read-only atom lacks bounded per-pipe use coverage" :
                             "writable atom lacks bounded entry/exit refresh coverage");
        }
        return true;
    }
    using OccurrenceKey = std::tuple<Operation*, Id, std::vector<Id>,
                                    std::vector<std::pair<Operation*, int64_t>>>;
    std::vector<std::vector<Use>> occurrenceGroups(ArrayRef<Use> fringe)
    {
        std::map<OccurrenceKey, std::size_t> positions;
        std::vector<std::vector<Use>> groups;
        for (const auto& use : fringe) {
            const auto& anchor = demands.anchors[use.event.type];
            std::vector<std::pair<Operation*, int64_t>> fixed;
            for (auto coordinate : anchor.coordinates) {
                fixed.emplace_back(coordinate.loop.getOperation(), coordinate.induction);
            }
            OccurrenceKey key{anchor.phase->elementOp, use.event.ordinal, use.event.visits, std::move(fixed)};
            auto inserted = positions.try_emplace(std::move(key), groups.size());
            if (inserted.second) { groups.emplace_back(); }
            groups[inserted.first->second].push_back(use);
        }
        return groups;
    }
    void selectBoundary(unsigned cell, bool tail, RegionalStorageBoundary& boundary)
    {
        auto fringe = uses(cell, tail ? tailBegin() : e.constant(0));
        auto groups = occurrenceGroups(fringe);
        detail::GuardedBoundaryScan scan(e);
        for (std::size_t index = 0; index < groups.size(); ++index) {
            scan.group(groups[tail ? groups.size() - 1 - index : index],
                [](const Use& use) { return detail::GuardedStorageMode{use.pipe, use.read, use.write}; },
                [&](const Use& use, Id writer, Id reader) {
                    if (e.constantValue(writer) != 0) {
                        (tail ? boundary.lastWriters : boundary.firstWriters).push_back({use.event, writer});
                    }
                    if (e.constantValue(reader) != 0) {
                        (tail ? boundary.lastReaders : boundary.firstReaders)[use.pipe].push_back({use.event, reader});
                    }
                });
        }
    }
    bool storage(unsigned cell)
    {
        const auto& atom = recognized.physical->atoms[cell];
        bool writable = false;
        std::set<uint32_t> readers;
        for (const auto& projection : recognized.physical->accesses) {
            if (projection.atom != cell) {
                continue;
            }
            const auto& access = recognized.skeleton.result.accesses[projection.access];
            writable |= access.writes;
            if (access.reads) {
                readers.insert(
                    static_cast<uint32_t>(demands.input->accesses().effects()[access.effect].phase->kPipeValue));
            }
        }
        if (writable && !coverage(cell, {})) {
            return false;
        }
        if (!writable) {
            for (auto pipe : readers) {
                if (!coverage(cell, pipe)) {
                    return false;
                }
            }
        }
        RegionalStorageBoundary boundary;
        boundary.cell = atom;
        selectBoundary(cell, false, boundary);
        selectBoundary(cell, true, boundary);
        out.storageBoundary.push_back(std::move(boundary));
        return true;
    }
};
} // namespace
FailureOr<RegionalAnalysis> boundedLifetimeRegionalSelectors(
    const BoundedLifetimeDemandResult& demands, const BoundedLifetimeRecognition& recognized,
    const RegionalAnalysis& queries, std::string& error)
{
    if (!queries.expressions || !queries.expressions->constructionError().empty() || !demands.input ||
        !demands.prerequisites || demands.prerequisites != recognized.prerequisites ||
        queries.accessModel != &demands.input->accesses() ||
        queries.gmAliasPolicy != demands.input->memory().gmPolicy() ||
        recognized.skeleton.phases.size() != demands.anchors.size() ||
        queries.anchors.size() != demands.anchors.size() || !CountedLoop::get(demands.loop)) {
        error = "bounded selectors belong to an incompatible input context";
        return failure();
    }
    for (auto [actual, expected] : llvm::zip(queries.anchors, demands.anchors)) {
        if (actual.phase != expected.phase) {
            error = "bounded selector occurrence identities differ";
            return failure();
        }
    }
    auto out = queries;
    Selectors exporter(demands, recognized, out, error);
    if (!exporter.run()) {
        return failure();
    }
    return out;
}
} // namespace mlir::pto::frontiersynch
