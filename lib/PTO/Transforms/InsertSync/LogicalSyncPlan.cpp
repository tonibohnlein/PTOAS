// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/InsertSync/LogicalSyncPlan.h"
#include "PTO/Transforms/InsertSync/SyncPhysicalFacts.h"
#include "PTO/Transforms/InsertSync/SyncGlobalOccurrences.h"
#include "PTO/Transforms/InsertSync/MemoryDependentAnalyzer.h"
#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"
#include "PTO/Transforms/InsertSync/SyncEffectCoverage.h"
#include "PTO/Transforms/InsertSync/SyncPayloadSnapshot.h"
#include "mlir/Analysis/FlatLinearValueConstraints.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Verifier.h"
#include <map>
#include <set>
#include <cstdlib>
#include <limits>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::logical_sync;
using namespace mlir::presburger;
namespace {
using Lane = PipelineType;
using Domain = std::pair<Lane, Lane>;
using Requirement = OrderingRequirement;
struct Stream {
    Domain pipes;
    Relation matching;
    unsigned source;
    int key = -1;
};
struct Barrier {
    unsigned point;
    Lane pipe;
    PresburgerSet domain;
    std::optional<Relation> logical;
};
class Constructor {
    func::FuncOp function;
    SyncPayloadSnapshot payload;
    llvm::function_ref<void(func::FuncOp)> emissionMutation;
    testing::RequirementObserver observeRequirements;
    InsertSyncGMAliasMode gm;
    SyncIRs ir;
    Buffer2MemInfoMap buffers;
    MemoryDependentAnalyzer memory;
    RelationQueries queries;
    SyncOccurrences facts;
    std::vector<const CompoundInstanceElement*> phases;
    std::vector<Lane> lanes;
    std::vector<Requirement> requirements;
    std::vector<Stream> streams;
    std::vector<Barrier> barriers;
    std::map<std::tuple<unsigned, unsigned, bool>, Relation> orderCache;
    std::map<std::pair<Lane, bool>, Relation> laneCache;
    std::optional<Relation> globalCache;
    std::optional<Relation> issueCache;
    std::optional<CompletionQueries> completionOrder;
    ConstructionResult result;
    unsigned exitPoint = 0;
    // Requirements are immutable after discovery. Receipts certify all of a
    // target's requirements for this selected logical plan version. Adding a
    // barrier preserves existing receipts; deletion needs a fresh full proof.
    uint64_t planVersion = 0, receiptVersion = 0;
    std::set<unsigned> provedTargets;

    bool fail(ConstructionResult::Status status, StringRef reason)
    {
        result.status = status;
        result.reason = reason.str();
        return false;
    }
    bool expect(QueryStatus status, StringRef reason)
    {
        if (status == QueryStatus::Proved)
            return true;
        return fail(
            status == QueryStatus::BudgetExhausted ? ConstructionResult::AnalysisLimit :
            status == QueryStatus::Unsupported     ? ConstructionResult::Unsupported :
                                                     ConstructionResult::Unproved,
            reason);
    }
    bool queryFailed(QueryStatus status, StringRef reason)
    {
        if (status == QueryStatus::Proved || status == QueryStatus::NotEstablished)
            return false;
        expect(status, reason);
        return true;
    }
    Relation empty() const
    {
        return Relation::getEmpty(
            PresburgerSpace::getRelationSpace(facts.dimensions(), facts.dimensions(), facts.parameters.size()));
    }
    std::optional<Relation> take(RelationResult answer)
    {
        if (!answer) {
            fail(
                answer.status == QueryStatus::BudgetExhausted ? ConstructionResult::AnalysisLimit :
                answer.status == QueryStatus::Unsupported     ? ConstructionResult::Unsupported :
                                                                ConstructionResult::Unproved,
                answer.reason);
            return {};
        }
        return std::move(answer.relation);
    }
    std::optional<Relation> compose(const Relation& a, const Relation& b) { return take(queries.compose(a, b)); }
    std::optional<Relation> order(unsigned p, unsigned q, bool inclusive = false)
    {
        auto key = std::make_tuple(p, q, inclusive);
        if (auto found = orderCache.find(key); found != orderCache.end())
            return found->second;
        auto value = take(facts.ordered(p, q, inclusive));
        if (value)
            value = take(queries.normalize(*value));
        if (value)
            orderCache.emplace(key, *value);
        return value;
    }
    std::optional<Relation> laneOrder(Lane pipe, bool inclusive)
    {
        auto key = std::make_pair(pipe, inclusive);
        if (auto found = laneCache.find(key); found != laneCache.end())
            return found->second;
        auto value = empty();
        for (unsigned p = 0; p < lanes.size(); ++p)
            if (lanes[p] == pipe)
                for (unsigned q = 0; q < lanes.size(); ++q)
                    if (lanes[q] == pipe) {
                        auto edge = order(p, q, inclusive);
                        if (!edge)
                            return {};
                        value.unionInPlace(*edge);
                    }
        laneCache.emplace(key, value);
        return value;
    }
    Relation identity() const
    {
        auto value = empty();
        for (unsigned p = 0; p < lanes.size(); ++p)
            value.unionInPlace(facts.identity(p));
        return value;
    }
    std::optional<Relation> issueOrder()
    {
        if (issueCache)
            return issueCache;
        auto value = empty();
        std::set<Lane> pipes(lanes.begin(), lanes.end());
        for (auto pipe : pipes) {
            auto lane = laneOrder(pipe, true);
            if (!lane)
                return {};
            value.unionInPlace(*lane);
        }
        issueCache = value;
        return value;
    }
    std::optional<Relation> globalOrder()
    {
        if (globalCache)
            return globalCache;
        auto value = empty();
        for (unsigned p = 0; p < lanes.size(); ++p)
            for (unsigned q = 0; q < lanes.size(); ++q) {
                auto edge = order(p, q, true);
                if (!edge)
                    return {};
                value.unionInPlace(*edge);
            }
        globalCache = value;
        return value;
    }
    std::optional<CompletionQueries> completion(Relation handoffs)
    {
        if (!completionOrder) {
            auto issue = issueOrder(), global = globalOrder();
            if (!issue || !global)
                return {};
            completionOrder.emplace(empty(), *issue, *global);
        }
        return completionOrder->withHandoffs(std::move(handoffs));
    }
    bool discover()
    {
        if (!supportsLogicalSyncTranslation(function))
            return fail(ConstructionResult::Unsupported, "memory loop forwarding requires qualified translation");
        PTOIRTranslator translator(ir, memory, buffers, function, SyncAnalysisMode::NORMALSYNC);
        translator.Build();
        auto coverage = inspectInsertSyncEffectCoverage(function, ir, false);
        if (failed(coverage))
            return fail(ConstructionResult::InternalError, "effect inspection failed");
        if (!*coverage)
            return fail(ConstructionResult::Unsupported, "unqualified physical effects");
        auto physical = importSyncPhysicalFacts(function, ir, queries.remainingWork());
        queries.spend(physical.work);
        if (physical.status != SyncPhysicalFacts::Status::Complete)
            return fail(
                physical.status == SyncPhysicalFacts::Status::InternalError ? ConstructionResult::InternalError :
                physical.status == SyncPhysicalFacts::Status::AnalysisLimit ? ConstructionResult::AnalysisLimit :
                                                                                    ConstructionResult::Unsupported,
                physical.reason);
        // Construction does not consume the exact-slice lifecycle projection:
        // partially overlapping translated accesses must retain their conflicts.
        phases = physical.phases;
        SmallVector<Operation*> points;
        for (auto* phase : phases) {
            if (!phase || !phase->elementOp || phase->macroOpInstanceId >= 0)
                return fail(ConstructionResult::Unsupported, "multi-phase operation occurrence summary");
            points.push_back(phase->elementOp);
            lanes.push_back(phase->kPipeValue);
        }
        if (physical.lifetimeScope != function.getOperation())
            return fail(ConstructionResult::Unsupported, "physical section exit lowering not established");
        if (!llvm::hasSingleElement(function.getBody()) ||
            !isa<func::ReturnOp>(function.getBody().front().getTerminator()))
            return fail(ConstructionResult::Unsupported, "logical function exit shape");
        exitPoint = points.size();
        points.push_back(function.getBody().front().getTerminator());
        lanes.push_back(physical.cube ? Lane::PIPE_MTE1 : Lane::PIPE_V);
        facts = SyncOccurrences::build(function, points);
        if (!queries.spend(facts.work))
            return fail(ConstructionResult::AnalysisLimit, "occurrence import work budget");
        if (!facts.complete)
            return fail(
                facts.limitExceeded ? ConstructionResult::AnalysisLimit : ConstructionResult::Unsupported,
                facts.reason);
        auto add = [&](unsigned p, unsigned q, const BaseMemInfo* a, const BaseMemInfo* b,
                       Requirement::Kind kind) -> bool {
            if (a && b && !logicalSyncMayAlias(a, b, function, gm))
                return true;
            if (a && b && disjointInsertSyncGlobalOccurrences(a, points[p], b, points[q], function))
                return true;
            auto relation = order(p, q);
            if (!relation)
                return false;
            if (!relation->isIntegerEmpty())
                requirements.push_back({kind, p, q, a, b, std::move(*relation)});
            return true;
        };
        for (unsigned p = 0; p < phases.size(); ++p)
            for (unsigned q = 0; q < phases.size(); ++q) {
                auto* a = phases[p];
                auto* b = phases[q];
                for (auto* x : a->defVec)
                    for (auto* y : b->useVec)
                        if (!add(p, q, x, y, Requirement::RAW))
                            return false;
                for (auto* x : a->useVec)
                    for (auto* y : b->defVec)
                        if (!add(p, q, x, y, Requirement::WAR))
                            return false;
                for (auto* x : a->defVec)
                    for (auto* y : b->defVec)
                        if (!add(p, q, x, y, Requirement::WAW))
                            return false;
                if (lanes[p] != lanes[q])
                    for (auto* x : a->useVec)
                        for (auto* y : b->useVec)
                            if (x->scope == AddressSpace::ACC && y->scope == AddressSpace::ACC &&
                                !add(p, q, x, y, Requirement::AccResource))
                                return false;
            }
        for (unsigned p = 0; p < phases.size(); ++p)
            if (!add(p, exitPoint, nullptr, nullptr, Requirement::Exit))
                return false;
        result.requirements = requirements.size();
        if (observeRequirements)
            observeRequirements(facts, phases, requirements);
        return true;
    }
    QueryStatus functional(const Relation& handoff)
    {
        auto inverse = handoff;
        inverse.inverse();
        auto consumers = queries.compose(inverse, handoff);
        if (!consumers)
            return consumers.status;
        auto producers = queries.compose(handoff, inverse);
        if (!producers)
            return producers.status;
        auto diagonal = identity();
        auto status = queries.contains(diagonal, *consumers.relation);
        return status == QueryStatus::Proved ? queries.contains(diagonal, *producers.relation) : status;
    }
    bool constructHandoffs()
    {
        std::map<Domain, Relation> missing;
        std::set<std::pair<unsigned, unsigned>> seen;
        for (const auto& r : requirements)
            if (lanes[r.source] != lanes[r.target]) {
                // Multiple immutable access obligations can name the same phase
                // occurrence relation; construction needs that relation only once.
                if (!seen.emplace(r.source, r.target).second)
                    continue;
                auto key = Domain{lanes[r.source], lanes[r.target]};
                auto [it, inserted] = missing.try_emplace(key, empty());
                (void)inserted;
                it->second.unionInPlace(r.occurrences);
            }
        for (const auto& [pipes, needed] : missing) {
            auto before = laneOrder(pipes.first, false), through = laneOrder(pipes.first, true);
            auto destinations = laneOrder(pipes.second, false);
            if (!before || !through || !destinations)
                return false;
            auto latest = take(queries.latestSources(needed, *before));
            if (!latest)
                return false;
            auto staircase = take(queries.staircase(*latest, *through, *destinations));
            if (!staircase)
                return false;
            if (!expect(functional(*staircase), "handoff participation is not one-to-one"))
                return false;
            // Static publication families are logical streams, not numeric keys.
            // Alternatives for first/empty/later use remain in the same family.
            for (unsigned p = 0; p < phases.size(); ++p)
                if (lanes[p] == pipes.first) {
                    auto family = take(queries.normalize(staircase->intersectDomain(facts.domain(p).getRangeSet())));
                    if (!family)
                        return false;
                    if (!family->isIntegerEmpty())
                        streams.push_back({pipes, std::move(*family), p, -1});
                }
        }
        return true;
    }
    std::optional<Relation> primitive(
        std::optional<unsigned> omittedBarrier = {}, std::optional<unsigned> omittedStream = {})
    {
        auto supply = empty();
        for (unsigned i = 0; i < streams.size(); ++i)
            if (omittedStream != i) {
                auto& stream = streams[i];
                supply.unionInPlace(stream.matching);
            }
        for (unsigned i = 0; i < barriers.size(); ++i)
            if (omittedBarrier != i) {
                auto& barrier = barriers[i];
                if (barrier.logical) {
                    supply.unionInPlace(*barrier.logical);
                    continue;
                }
                // In a clear original block interval, the previous lane operation
                // is the exact predecessor in this invocation. Reuse the imported
                // IV identities rather than rediscovering it through every pair.
                auto* anchor = facts.points[barrier.point].operation;
                for (auto* op = anchor->getPrevNode(); op; op = op->getPrevNode()) {
                    if (op->getNumRegions())
                        break;
                    auto found = facts.ids.find(op);
                    if (found == facts.ids.end() || lanes[found->second] != barrier.pipe)
                        continue;
                    auto edge = order(found->second, barrier.point);
                    if (!edge)
                        return {};
                    IntegerRelation invocation(empty().getSpace());
                    for (unsigned iv = 1; iv < facts.dimensions(); ++iv) {
                        SmallVector<int64_t> row(invocation.getNumCols());
                        row[iv] = 1;
                        row[facts.dimensions() + iv] = -1;
                        invocation.addEquality(row);
                    }
                    barrier.logical =
                        take(queries.normalize(edge->intersect(Relation(invocation)).intersectRange(barrier.domain)));
                    if (!barrier.logical)
                        return {};
                    break;
                }
                if (barrier.logical) {
                    supply.unionInPlace(*barrier.logical);
                    continue;
                }
                auto before = laneOrder(barrier.pipe, false);
                if (!before)
                    return {};
                auto candidates = empty();
                for (unsigned source = 0; source < lanes.size(); ++source)
                    if (lanes[source] == barrier.pipe) {
                        auto edge = order(source, barrier.point);
                        if (!edge)
                            return {};
                        candidates.unionInPlace(*edge);
                    }
                auto cut = take(queries.normalize(candidates.intersectRange(barrier.domain)));
                if (!cut)
                    return {};
                auto edge = take(queries.latestSources(*cut, *before));
                if (!edge)
                    return {};
                barrier.logical = *edge;
                supply.unionInPlace(*edge);
            }
        return supply;
    }
    Relation requirementsAt(unsigned point) const
    {
        auto value = empty();
        std::set<unsigned> seen;
        for (const auto& r : requirements)
            if (r.target == point && seen.insert(r.source).second)
                value.unionInPlace(r.occurrences);
        return value;
    }
    QueryStatus proveRequirements(CompletionQueries& completed)
    {
        for (unsigned p = 0; p < lanes.size(); ++p) {
            auto status = completed.prove(requirementsAt(p), queries, 2);
            if (status != QueryStatus::Proved)
                return status;
        }
        return QueryStatus::Proved;
    }
    bool repairBarriers()
    {
        auto supply = primitive();
        if (!supply)
            return false;
        auto completed = completion(*supply);
        if (!completed)
            return false;
        for (unsigned p = 0; p < lanes.size(); ++p) {
            auto required = requirementsAt(p);
            auto status = completed->prove(required, queries, 2);
            if (std::getenv("PTOAS_LOGICAL_TRACE"))
                llvm::errs() << "logical phase " << p << " work " << queries.work() << " status " << unsigned(status)
                             << " required pieces " << required.getNumDisjuncts() << " logical edges "
                             << supply->getNumDisjuncts() << " known pieces " << completed->supply().getNumDisjuncts()
                             << "\n";
            if (status == QueryStatus::Proved) {
                provedTargets.insert(p);
                continue;
            }
            if (queryFailed(
                    status, "combined completion query unavailable at phase " + std::to_string(p) + "; streams=" +
                                std::to_string(streams.size()) + "; barriers=" + std::to_string(barriers.size())))
                return false;
            auto missing = take(queries.subtract(required, completed->supply()));
            if (!missing)
                return false;
            // Cross-lane construction must already supply its own guarantees.
            for (const auto& r : requirements)
                if (r.target == p && lanes[r.source] != lanes[p])
                    if (!expect(
                            queries.contains(completed->supply(), r.occurrences),
                            "cross-lane occurrence requirement remains unordered"))
                        return false;
            auto domain = take(queries.normalize(missing->getRangeSet()));
            if (!domain)
                return false;
            auto ambient = facts.domain(p).getRangeSet();
            auto covers = queries.contains(*domain, ambient);
            if (queryFailed(covers, "barrier execution domain query"))
                return false;
            barriers.push_back({p, lanes[p], covers == QueryStatus::Proved ? ambient : PresburgerSet(*domain), {}});
            auto laneBefore = laneOrder(lanes[p], false);
            if (!laneBefore)
                return false;
            auto barrierSupply = take(queries.normalize(laneBefore->intersectRange(barriers.back().domain)));
            if (!barrierSupply)
                return false;
            // The new barrier supplies the exact missing subset; the prior
            // proved supply covers its complement. This is a constructive
            // proof, not a receipt inferred from merely selecting a barrier.
            if (!expect(queries.contains(*barrierSupply, *missing), "new barrier does not supply missing requirements"))
                return false;
            ++planVersion;
            receiptVersion = planVersion;
            provedTargets.insert(p);
            supply = primitive();
            if (!supply)
                return false;
            if (!barriers.back().logical)
                return fail(ConstructionResult::InternalError, "new barrier has no logical completion cut");
            if (!expect(completed->addHandoffs(*barriers.back().logical, queries), "new barrier completion update"))
                return false;
        }
        // Feedback is part of initial construction. Candidate barriers never
        // supply the proof of their own deletion.
        for (unsigned i = 0; i < barriers.size();) {
            auto remaining = primitive(i);
            if (!remaining)
                return false;
            auto trial = completion(*remaining);
            if (!trial)
                return false;
            // First challenge the demand at this cut. A failed local query
            // keeps the barrier; a successful one is followed by every other
            // requirement before deletion can be accepted.
            auto status = trial->prove(requirementsAt(barriers[i].point), queries, 2);
            if (status == QueryStatus::Proved)
                status = proveRequirements(*trial);
            if (status == QueryStatus::Proved) {
                barriers.erase(barriers.begin() + i);
                ++planVersion;
                receiptVersion = planVersion;
                provedTargets.clear();
                for (unsigned p = 0; p < lanes.size(); ++p)
                    provedTargets.insert(p);
            } else if (queryFailed(status, "barrier deletion completion query"))
                return false;
            else
                ++i;
        }
        return true;
    }
    // Lowering and independent reconstruction are provided below. They consume
    // this plan and never ask ordinary InsertSync to manufacture a seed.
    bool realize();
    bool reconstruct();
    QueryStatus reuseSafe(const Relation& matching, Lane pipe, CompletionQueries& completed)
    {
        auto before = laneOrder(pipe, false);
        if (!before)
            return result.status == ConstructionResult::AnalysisLimit ? QueryStatus::BudgetExhausted :
                                                                        QueryStatus::Unsupported;
        auto domain = matching.getDomainSet();
        auto following = before->intersectDomain(domain).intersectRange(domain);
        auto next = queries.firstTargets(following, *before);
        if (!next)
            return next.status;
        auto inverse = matching;
        inverse.inverse();
        auto required = queries.compose(inverse, *next.relation);
        if (!required)
            return required.status;
        return completed.prove(*required.relation, queries, 2);
    }

public:
    Constructor(
        func::FuncOp f, InsertSyncGMAliasMode gm, uint64_t budget, llvm::function_ref<void(func::FuncOp)> mutate = {},
        testing::RequirementObserver observe = {})
        : function(f), payload(f), emissionMutation(mutate), observeRequirements(observe), gm(gm), queries(budget)
    {}
    ConstructionResult run()
    {
        StringRef stage = "discovery";
        auto run = [&](StringRef name, auto method) {
            stage = name;
            auto start = queries.work();
            bool ok = (this->*method)();
            if (std::getenv("PTOAS_LOGICAL_TRACE"))
                llvm::errs() << "logical stage " << name << " work " << queries.work() - start << " complete " << ok
                             << "\n";
            if (!ok)
                result.reason += " (stage work " + std::to_string(queries.work() - start) + ")";
            return ok;
        };
        if (run("discovery", &Constructor::discover) && run("handoffs", &Constructor::constructHandoffs) &&
            run("barriers", &Constructor::repairBarriers) && run("realization", &Constructor::realize)) {
            result.status = ConstructionResult::Applied;
            result.reason = "constructed and reconstructed occurrence handoffs";
            result.handoffs = streams.size();
            result.barriers = barriers.size();
        } else
            result.reason = stage.str() + ": " + result.reason;
        result.work = queries.work();
        return result;
    }
};
} // namespace

namespace {
// These are lowering conditions, not program patterns. A proposed mathematical
// domain must equal their interpretation at the actual insertion point. No
// guessed trip distance or unavailable future condition is emitted.
struct BoundaryTest {
    enum Kind { First, Last, Nonempty, Predicate, BooleanParameter, ConstantBound, DifferenceBound,
                ParameterBound, ParameterResidue } kind;
    unsigned loop;
    bool truth;
    int64_t bound = 0;
    arith::CmpIPredicate comparison = arith::CmpIPredicate::eq;
    bool fromUpper = false;
    int64_t modulus = 1;
};
using Clause = SmallVector<BoundaryTest, 2>;
using Guard = SmallVector<Clause, 2>;
// Short-circuit DNF can duplicate a suffix at each failed literal. Account for
// that actual tree before allocation/emission, rather than depending on the
// later occurrence importer to discover an already expanded function.
bool chargeGuardEmission(const Guard& guard, uint64_t& remaining)
{
    constexpr uint64_t operationsPerLiteral = 12; // includes scalar ops and both region yields
    constexpr unsigned maxDepth = 64;
    uint64_t nodes = 0;
    unsigned depth = 0;
    for (const auto& clause : guard) {
        if (clause.size() > maxDepth - depth) return false;
        depth += clause.size();
    }
    const bool shortCircuit = llvm::any_of(guard, [](const Clause& clause) {
        return llvm::any_of(clause, [](const BoundaryTest& test) {
            return test.kind == BoundaryTest::ParameterResidue;
        });
    });
    if (shortCircuit) {
        for (const auto& clause : llvm::reverse(guard)) {
            uint64_t suffix = nodes;
            nodes = 1; // the action after a successful clause
            for (unsigned i = 0; i < clause.size(); ++i) {
                if (suffix > remaining || nodes > remaining - suffix ||
                    operationsPerLiteral > remaining - suffix - nodes) return false;
                nodes += suffix + operationsPerLiteral;
            }
        }
    } else {
        nodes = 4; // the action and optional guarding if/yield
        for (const auto& clause : guard)
            nodes += operationsPerLiteral * (clause.size() + 1);
    }
    if (nodes > remaining) return false;
    remaining -= nodes;
    return true;
}
struct Endpoint {
    unsigned point;
    bool publication;
    unsigned stream;
    Guard guard;
};
class BoundaryLowering {
    const SyncOccurrences& facts;
    RelationQueries& queries;
    DominanceInfo dominance;
    QueryStatus outcome = QueryStatus::Unsupported;
    bool queryFailed(QueryStatus status)
    {
        if (status == QueryStatus::Proved || status == QueryStatus::NotEstablished)
            return false;
        outcome = status;
        return true;
    }
    std::optional<Relation> condition(unsigned point, BoundaryTest test)
    {
        if (test.kind == BoundaryTest::ConstantBound || test.kind == BoundaryTest::ParameterBound) {
            auto domain = facts.domain(point);
            auto restricted = Relation::getEmpty(domain.getSpace());
            for (auto piece : domain.getAllDisjuncts()) {
                auto type = test.comparison == arith::CmpIPredicate::eq  ? BoundType::EQ :
                            test.comparison == arith::CmpIPredicate::sle ? BoundType::UB :
                                                                           BoundType::LB;
                piece.addBound(type, test.kind == BoundaryTest::ConstantBound ? test.loop + 1 :
                                   facts.dimensions() + test.loop, test.bound);
                restricted.unionInPlace(Relation(piece));
            }
            return restricted;
        }
        if (test.kind == BoundaryTest::Predicate) {
            auto positive = facts.predicateDomain(test.loop, point);
            if (test.truth)
                return positive;
            auto negative = queries.subtract(facts.domain(point), positive);
            if (!negative)
                outcome = negative.status;
            return negative ? std::move(negative.relation) : std::nullopt;
        }
        auto* context = facts.points[point].operation->getContext();
        AffineExpr expression;
        if (test.kind == BoundaryTest::ParameterResidue)
            expression = getAffineSymbolExpr(test.loop, context) % test.modulus - test.bound;
        else if (test.kind == BoundaryTest::BooleanParameter)
            expression = getAffineSymbolExpr(test.loop, context) - 1;
        else {
            auto loop = facts.loopDomains[test.loop];
            auto iv = getAffineDimExpr(test.loop, context);
            expression = test.kind == BoundaryTest::DifferenceBound ?
                             (test.fromUpper ? loop.upper - iv : iv - loop.lower) :
                         test.kind == BoundaryTest::First ? iv - loop.lower :
                         test.kind == BoundaryTest::Last  ? iv + loop.step - loop.upper :
                                                            loop.upper - loop.lower - 1;
        }
        SmallVector<AffineExpr> dims, symbols;
        for (unsigned i = 0; i < facts.loops.size(); ++i)
            dims.push_back(getAffineDimExpr(i + 1, context));
        for (unsigned i = 0; i < facts.parameters.size(); ++i)
            symbols.push_back(getAffineSymbolExpr(i, context));
        expression = expression.replaceDimsAndSymbols(dims, symbols);
        auto build = [&](AffineExpr expr, bool equality) -> std::optional<Relation> {
            auto set = IntegerSet::get(facts.dimensions(), symbols.size(), {expr}, {equality});
            FlatLinearConstraints flat(facts.dimensions(), symbols.size());
            std::vector<SmallVector<int64_t, 8>> rows;
            if (failed(getFlattenedAffineExprs(set, &rows, &flat)))
                return {};
            SmallVector<llvm::DynamicAPInt> row;
            for (int64_t coefficient : rows.front()) row.emplace_back(coefficient);
            if (test.kind == BoundaryTest::DifferenceBound)
                row.back() += test.comparison == arith::CmpIPredicate::sle ?
                                  llvm::DynamicAPInt(test.bound) : -llvm::DynamicAPInt(test.bound);
            if (equality) flat.addEquality(row);
            else flat.addInequality(row);
            return facts.domain(point).intersect(Relation(flat));
        };
        if (test.kind == BoundaryTest::DifferenceBound && test.comparison == arith::CmpIPredicate::sle)
            expression = -expression;
        auto positive = build(expression,
            test.kind == BoundaryTest::First || test.kind == BoundaryTest::BooleanParameter ||
            test.kind == BoundaryTest::ParameterResidue ||
            (test.kind == BoundaryTest::DifferenceBound && test.comparison == arith::CmpIPredicate::eq));
        if (!positive || test.truth)
            return positive;
        auto negative = queries.subtract(facts.domain(point), *positive);
        if (!negative)
            outcome = negative.status;
        return negative ? std::move(negative.relation) : std::nullopt;
    }

public:
    BoundaryLowering(const SyncOccurrences& f, RelationQueries& q) : facts(f), queries(q) {}
    QueryStatus status() const { return outcome; }
    std::optional<Guard> prepare(unsigned point, const PresburgerSet& wanted)
    {
        outcome = QueryStatus::Unsupported;
        Relation target = wanted;
        auto ambient = facts.domain(point);
        auto status = queries.contains(target, ambient);
        if (status == QueryStatus::Proved)
            return Guard{Clause{}};
        if (queryFailed(status))
            return {};
        struct Candidate {
            BoundaryTest test;
            Relation domain;
        };
        std::vector<Candidate> candidates;
        Operation* anchor = facts.points[point].operation;
        for (unsigned i = 0; i < facts.loops.size(); ++i) {
            auto loop = facts.loops[i];
            if (!dominance.dominates(loop.getLowerBound(), anchor) ||
                !dominance.dominates(loop.getUpperBound(), anchor))
                continue;
            for (auto kind : {BoundaryTest::First, BoundaryTest::Last, BoundaryTest::Nonempty}) {
                if (kind != BoundaryTest::Nonempty && !loop->isProperAncestor(anchor))
                    continue;
                for (bool truth : {true, false}) {
                    BoundaryTest test{kind, i, truth};
                    auto domain = condition(point, test);
                    if (!domain)
                        return {};
                    candidates.push_back({test, std::move(*domain)});
                }
            }
        }
        for (unsigned i = 0; i < facts.predicates.size(); ++i) {
            if (!dominance.dominates(facts.predicates[i].value, anchor))
                continue;
            for (bool truth : {true, false}) {
                BoundaryTest test{BoundaryTest::Predicate, i, truth};
                auto domain = condition(point, test);
                if (!domain)
                    return {};
                candidates.push_back({test, std::move(*domain)});
            }
        }
        // A normalized branch can be an equivalent spelling of an already
        // available Boolean parameter even when that spelling is computed
        // later. Reuse the original parameter binding and prove the domain;
        // never hoist the later expression or assume its value is available.
        for (unsigned i = 0; i < facts.parameters.size(); ++i) {
            Value value = facts.parameters[i];
            if (!value.getType().isInteger(1) || !dominance.dominates(value, anchor))
                continue;
            for (bool truth : {true, false}) {
                BoundaryTest test{BoundaryTest::BooleanParameter, i, truth};
                auto domain = condition(point, test);
                if (!domain)
                    return {};
                candidates.push_back({test, std::move(*domain)});
            }
        }
        if (candidates.size() > 32)
            return {};
        Guard guard;
        Relation remaining = target;
        auto admit = [&](const Relation& domain, Clause clause) -> bool {
            if (domain.isIntegerEmpty())
                return true;
            // A candidate already covered by prior clauses cannot contribute.
            // Test the intersection directly rather than subtracting and then
            // proving that the difference still contains the entire remainder.
            auto contribution = queries.normalize(domain.intersect(remaining));
            if (!contribution) { outcome = contribution.status; return false; }
            if (contribution.relation->isIntegerEmpty()) return true;
            auto included = queries.contains(target, domain);
            if (queryFailed(included))
                return false;
            if (included != QueryStatus::Proved)
                return true;
            auto rest = queries.subtract(remaining, domain);
            if (!rest) {
                outcome = rest.status;
                return false;
            }
            guard.push_back(std::move(clause));
            remaining = std::move(*rest.relation);
            return true;
        };
        unsigned coveredCandidates = 0;
        auto cover = [&]() -> std::optional<bool> {
            for (unsigned i = coveredCandidates; i < candidates.size(); ++i) {
                const auto& a = candidates[i];
                if (!admit(a.domain, Clause{a.test}))
                    return std::nullopt;
                if (remaining.isIntegerEmpty())
                    return true;
            }
            for (unsigned i = 0; i < candidates.size(); ++i)
                for (unsigned j = std::max(i + 1, coveredCandidates); j < candidates.size(); ++j) {
                    auto domain = candidates[i].domain.intersect(candidates[j].domain);
                    if (!admit(domain, Clause{candidates[i].test, candidates[j].test}))
                        return std::nullopt;
                    if (remaining.isIntegerEmpty())
                        return true;
                }
            // Target membership is immutable and accepted clauses only shrink
            // remaining. Previously tried singles/pairs never become useful
            // when new proposal forms are appended.
            coveredCandidates = candidates.size();
            return false;
        };
        auto covered = cover();
        if (!covered)
            return {};
        if (*covered)
            return guard;
        // A first use carried from a different branch can occur at an interior
        // iteration. Recover simple constant bounds from the actual required
        // domain, rather than enumerating special first/second-use recipes.
        // Emission is only cmp(iv, constant), with no new overflow-prone math.
        auto normalized = queries.normalize(remaining);
        if (!normalized) {
            outcome = normalized.status;
            return {};
        }
        std::set<std::tuple<unsigned, arith::CmpIPredicate, int64_t>> bounds;
        for (const auto& piece : normalized.relation->getAllDisjuncts()) {
            Simplex simplex(piece);
            if (simplex.isEmpty())
                continue;
            for (unsigned i = 0; i < facts.loops.size(); ++i) {
                auto loop = facts.loops[i];
                if (!loop->isProperAncestor(anchor))
                    continue;
                SmallVector<llvm::DynamicAPInt> objective(piece.getNumCols(), llvm::DynamicAPInt(0));
                objective[i + 1] = 1;
                std::optional<int64_t> lower;
                for (auto direction : {Simplex::Direction::Down, Simplex::Direction::Up}) {
                    uint64_t cost =
                        uint64_t(piece.getNumEqualities() + piece.getNumInequalities() + 1) * piece.getNumCols();
                    if (!queries.spend(cost)) {
                        outcome = QueryStatus::BudgetExhausted;
                        return {};
                    }
                    auto optimum = simplex.computeOptimum(direction, objective);
                    if (!optimum.isBounded())
                        continue;
                    // Rational extrema only propose constants. The existing
                    // INTEGER domain-equality query must prove the resulting
                    // guards, including parity, gaps and all parameter values.
                    auto bound = direction == Simplex::Direction::Down ? presburger::ceil(*optimum) :
                                                                         presburger::floor(*optimum);
                    if (bound < std::numeric_limits<int64_t>::min() || bound > std::numeric_limits<int64_t>::max())
                        continue;
                    int64_t value = int64_t(bound);
                    if (auto integer = dyn_cast<IntegerType>(loop.getInductionVar().getType()))
                        if (!llvm::isIntN(integer.getWidth(), value))
                            continue;
                    auto comparison =
                        direction == Simplex::Direction::Down ? arith::CmpIPredicate::sge : arith::CmpIPredicate::sle;
                    bounds.emplace(i, comparison, value);
                    if (direction == Simplex::Direction::Down)
                        lower = value;
                    else if (lower == value)
                        bounds.emplace(i, arith::CmpIPredicate::eq, value);
                }
            }
        }
        for (auto [loop, comparison, bound] : bounds) {
            BoundaryTest test{BoundaryTest::ConstantBound, loop, true, bound, comparison};
            auto domain = condition(point, test);
            if (!domain)
                return {};
            if (candidates.size() == 32)
                break;
            candidates.push_back({test, std::move(*domain)});
        }
        covered = cover();
        if (covered && *covered)
            return guard;
        if (!covered)
            return {};
        // General differences of available loop bounds and IVs. Thresholds
        // come from the requested integer domain, not from a slot count or a
        // kernel-specific last-use recipe. Prove arithmetic on the FULL anchor
        // domain before considering any threshold's true subset.
        auto* context = anchor->getContext();
        const unsigned n = facts.dimensions(), ns = facts.parameters.size();
        for (unsigned i = 0; i < facts.loops.size(); ++i) {
            auto loop = facts.loops[i];
            if (!loop->isProperAncestor(anchor))
                continue;
            for (bool fromUpper : {false, true}) {
                SmallVector<AffineExpr> dims, syms;
                for (unsigned d = 0; d < facts.loops.size(); ++d)
                    dims.push_back(getAffineDimExpr(d + 1, context));
                for (unsigned s = 0; s < ns; ++s)
                    syms.push_back(getAffineSymbolExpr(s, context));
                auto ld = facts.loopDomains[i];
                auto iv = getAffineDimExpr(i, context);
                auto expression = (fromUpper ? ld.upper - iv : iv - ld.lower).replaceDimsAndSymbols(dims, syms);
                auto set = IntegerSet::get(n + 1, ns, {expression - getAffineDimExpr(n, context)}, {true});
                FlatLinearConstraints flat(n + 1, ns);
                std::vector<SmallVector<int64_t, 8>> rows;
                if (failed(getFlattenedAffineExprs(set, &rows, &flat)))
                    continue;
                flat.addEquality(rows.front());
                auto lift = [&](Relation domain) {
                    domain.insertVarInPlace(VarKind::Range, n);
                    return domain.intersect(Relation(flat));
                };
                auto full = takeRange(lift(ambient), n);
                if (!full) {
                    if (outcome == QueryStatus::BudgetExhausted) return {};
                    continue;
                }
                unsigned width = isa<IndexType>(loop.getInductionVar().getType()) ? 64 :
                                     cast<IntegerType>(loop.getInductionVar().getType()).getWidth();
                if (!llvm::isIntN(width, full->first) || !llvm::isIntN(width, full->second))
                    continue;
                auto wantedRange = lift(remaining);
                auto extrema = takeRange(wantedRange, n);
                if (!extrema) {
                    if (outcome == QueryStatus::BudgetExhausted) return {};
                    continue;
                }
                // Relaxed floor witnesses can put the rational endpoint at an
                // integer hole. Skip only proved-empty levels, with a fixed
                // bound; exact guard-domain cover remains the acceptance gate.
                for (auto comparison : {arith::CmpIPredicate::sge, arith::CmpIPredicate::sle}) {
                    int64_t bound = comparison == arith::CmpIPredicate::sge ? extrema->first : extrema->second;
                    for (unsigned step = 0; step < 8; ++step) {
                        auto slice = Relation::getEmpty(wantedRange.getSpace());
                        for (auto piece : wantedRange.getAllDisjuncts()) {
                            piece.addBound(BoundType::EQ, n, bound);
                            slice.unionInPlace(Relation(piece));
                        }
                        auto present = queries.normalize(slice);
                        if (!present) { outcome = present.status; return {}; }
                        if (!present.relation->isIntegerEmpty()) break;
                        if ((comparison == arith::CmpIPredicate::sge && bound == INT64_MAX) ||
                            (comparison == arith::CmpIPredicate::sle && bound == INT64_MIN)) break;
                        bound += comparison == arith::CmpIPredicate::sge ? 1 : -1;
                    }
                    if (!llvm::isIntN(width, bound)) continue;
                    BoundaryTest test{BoundaryTest::DifferenceBound, i, true, bound, comparison, fromUpper};
                    auto domain = condition(point, test);
                    if (!domain) return {};
                    if (!admit(*domain, Clause{test})) return {};
                    if (remaining.isIntegerEmpty()) return guard;
                    // Reuse already qualified predicate alternatives where a
                    // difference is needed only on part of the original path.
                    for (const auto& candidate : candidates) {
                        if (!admit(domain->intersect(candidate.domain), Clause{test, candidate.test})) return {};
                        if (remaining.isIntegerEmpty()) return guard;
                    }
                }
            }
        }
        // At an enclosing boundary, a final participating lane can depend on
        // a parameter's residue. Recover only small divisors actually present
        // in the selected relation. Neither loop counts nor buffer depth are
        // consulted. A nonnegative bound is evaluated FIRST, so signed
        // remainder is mathematical modulo wherever its definition executes.
        std::set<int64_t> moduli;
        for (const auto& piece : remaining.getAllDisjuncts()) {
            auto divisions = piece.getLocalReprs();
            for (auto denominator : divisions.getDenoms())
                if (denominator >= 2 && denominator <= 16)
                    moduli.insert(int64_t(denominator));
        }
        for (unsigned s = 0; s < facts.parameters.size(); ++s) {
            Value value = facts.parameters[s];
            if (!dominance.dominates(value, anchor) || value.getType().isInteger(1)) continue;
            auto range = takeRange(remaining, n + s);
            if (!range) {
                if (outcome == QueryStatus::BudgetExhausted) return {};
                continue;
            }
            int64_t lower = range->first;
            for (unsigned step = 0; step < 8 && lower < INT64_MAX; ++step) {
                auto slice = Relation::getEmpty(remaining.getSpace());
                for (auto piece : remaining.getAllDisjuncts()) {
                    piece.addBound(BoundType::EQ, n + s, lower);
                    slice.unionInPlace(Relation(piece));
                }
                auto check = queries.normalize(slice);
                if (!check) { outcome = check.status; return {}; }
                if (!check.relation->isIntegerEmpty()) break;
                ++lower;
            }
            if (lower < 0) continue;
            unsigned width = isa<IndexType>(value.getType()) ? 64 : cast<IntegerType>(value.getType()).getWidth();
            if (!llvm::isIntN(width, lower)) continue;
            BoundaryTest bound{BoundaryTest::ParameterBound, s, true, lower, arith::CmpIPredicate::sge};
            auto nonnegative = condition(point, bound);
            if (!nonnegative) return {};
            if (!admit(*nonnegative, Clause{bound})) return {};
            if (remaining.isIntegerEmpty()) return guard;
            for (int64_t modulus : moduli) {
                if (!llvm::isIntN(width, modulus)) continue;
                for (int64_t residue = 0; residue < modulus; ++residue) {
                    BoundaryTest test{BoundaryTest::ParameterResidue, s, true, residue,
                                      arith::CmpIPredicate::eq, false, modulus};
                    auto domain = condition(point, test);
                    if (!domain || !admit(domain->intersect(*nonnegative), Clause{bound, test})) return {};
                    if (remaining.isIntegerEmpty()) return guard;
                }
            }
        }
        return {};
    }
    std::optional<std::pair<int64_t, int64_t>> takeRange(const Relation& relation, unsigned coordinate)
    {
        std::optional<std::pair<int64_t, int64_t>> range;
        for (const auto& piece : relation.getAllDisjuncts()) {
            if (!queries.spend(uint64_t(piece.getNumConstraints() + 1) * piece.getNumCols() * 2)) {
                outcome = QueryStatus::BudgetExhausted; return {};
            }
            Simplex simplex(piece);
            if (simplex.isEmpty()) continue;
            SmallVector<llvm::DynamicAPInt> objective(piece.getNumCols());
            objective[coordinate] = 1;
            auto lo = simplex.computeOptimum(Simplex::Direction::Down, objective);
            auto hi = simplex.computeOptimum(Simplex::Direction::Up, objective);
            if (!lo.isBounded() || !hi.isBounded()) return {};
            auto low = presburger::ceil(*lo), high = presburger::floor(*hi);
            if (low < INT64_MIN || high > INT64_MAX || low > high) return {};
            int64_t a = int64_t(low), b = int64_t(high);
            range = range ? std::make_pair(std::min(range->first, a), std::max(range->second, b)) : std::make_pair(a, b);
        }
        return range;
    }
    Value emitGuard(OpBuilder& builder, Location location, const Guard& guard) const
    {
        if (guard.size() == 1 && guard.front().empty())
            return {};
        Value disjunction;
        for (const auto& clause : guard) {
            Value conjunction;
            for (const auto& test : clause) {
                Value value;
                if (test.kind == BoundaryTest::Predicate)
                    value = facts.predicates[test.loop].value;
                else if (test.kind == BoundaryTest::BooleanParameter)
                    value = facts.parameters[test.loop];
                else if (test.kind == BoundaryTest::ParameterBound || test.kind == BoundaryTest::ParameterResidue) {
                    value = facts.parameters[test.loop];
                    if (test.kind == BoundaryTest::ParameterResidue) {
                        auto divisor = builder.create<arith::ConstantOp>(location, builder.getIntegerAttr(value.getType(), test.modulus));
                        value = builder.create<arith::RemSIOp>(location, value, divisor);
                    }
                    auto bound = builder.create<arith::ConstantOp>(location, builder.getIntegerAttr(value.getType(), test.bound));
                    value = builder.create<arith::CmpIOp>(location, test.comparison, value, bound);
                }
                else if (test.kind == BoundaryTest::ConstantBound || test.kind == BoundaryTest::DifferenceBound) {
                    auto loop = facts.loops[test.loop];
                    Value iv = loop.getInductionVar();
                    if (test.kind == BoundaryTest::DifferenceBound)
                        iv = test.fromUpper ? builder.create<arith::SubIOp>(location, loop.getUpperBound(), iv) :
                                              builder.create<arith::SubIOp>(location, iv, loop.getLowerBound());
                    auto constant =
                        builder.create<arith::ConstantOp>(location, builder.getIntegerAttr(iv.getType(), test.bound));
                    value = builder.create<arith::CmpIOp>(location, test.comparison, iv, constant);
                } else {
                    auto loop = facts.loops[test.loop];
                    if (test.kind == BoundaryTest::Nonempty)
                        value = builder.create<arith::CmpIOp>(
                            location, arith::CmpIPredicate::slt, loop.getLowerBound(), loop.getUpperBound());
                    else if (test.kind == BoundaryTest::First)
                        value = builder.create<arith::CmpIOp>(
                            location, arith::CmpIPredicate::eq, loop.getInductionVar(), loop.getLowerBound());
                    else {
                        // The occurrence importer proves this induction increment
                        // representable in the actual induction type.
                        auto next = builder.create<arith::AddIOp>(location, loop.getInductionVar(), loop.getStep());
                        value = builder.create<arith::CmpIOp>(
                            location, arith::CmpIPredicate::sge, next, loop.getUpperBound());
                    }
                }
                if (!test.truth) {
                    auto one = builder.create<arith::ConstantIntOp>(location, 1, 1);
                    value = builder.create<arith::XOrIOp>(location, value, one);
                }
                conjunction = conjunction ? builder.create<arith::AndIOp>(location, conjunction, value) : value;
            }
            disjunction = disjunction ? builder.create<arith::OrIOp>(location, disjunction, conjunction) : conjunction;
        }
        return disjunction;
    }
};
} // namespace

bool Constructor::realize()
{
    BoundaryLowering lowering(facts, queries);
    std::vector<Endpoint> endpoints;
    std::vector<Guard> barrierGuards;
    // Participation and predicate legality precede assignment, including entry,
    // first use, skipped/empty readers and final-use occurrences.
    for (unsigned i = 0; i < streams.size(); ++i) {
        auto& stream = streams[i];
        if (std::getenv("PTOAS_LOGICAL_TRACE"))
            llvm::errs() << "logical prepare publication " << stream.source << " work " << queries.work() << "\n";
        auto publication = lowering.prepare(stream.source, stream.matching.getDomainSet());
        if (!publication) {
            if (std::getenv("PTOAS_LOGICAL_TRACE")) {
                llvm::errs() << "unlowered publication at " << stream.source << "\n";
                stream.matching.getDomainSet().print(llvm::errs());
            }
            return expect(lowering.status(), "publication domain has no qualified boundary lowering");
        }
        endpoints.push_back({stream.source, true, i, std::move(*publication)});
        for (unsigned p = 0; p < lanes.size(); ++p) {
            auto domain = stream.matching.getRangeSet().intersect(facts.domain(p).getRangeSet());
            if (domain.isIntegerEmpty())
                continue;
            if (std::getenv("PTOAS_LOGICAL_TRACE"))
                llvm::errs() << "logical prepare acquisition " << p << " work " << queries.work() << "\n";
            auto acquisition = lowering.prepare(p, domain);
            if (!acquisition) {
                if (std::getenv("PTOAS_LOGICAL_TRACE")) {
                    llvm::errs() << "unlowered acquisition at " << p << " from " << stream.source << "\n";
                    domain.print(llvm::errs());
                }
                return expect(lowering.status(), "acquisition domain has no qualified boundary lowering");
            }
            endpoints.push_back({p, false, i, std::move(*acquisition)});
        }
    }
    for (const auto& barrier : barriers) {
        auto guard = lowering.prepare(barrier.point, barrier.domain);
        if (!guard)
            return expect(lowering.status(), "barrier domain has no qualified boundary lowering");
        barrierGuards.push_back(std::move(*guard));
    }
    uint64_t emissionAllowance = 4096;
    for (const auto& endpoint : endpoints)
        if (!chargeGuardEmission(endpoint.guard, emissionAllowance))
            return fail(ConstructionResult::AnalysisLimit, "guard emission size or depth limit");
    for (const auto& guard : barrierGuards)
        if (!chargeGuardEmission(guard, emissionAllowance))
            return fail(ConstructionResult::AnalysisLimit, "guard emission size or depth limit");
    auto supply = primitive();
    if (!supply)
        return false;
    auto issued = issueOrder();
    if (!issued)
        return false;
    // Preparing legal endpoint guards changes no selected logical action. Reuse
    // the established requirement receipt; actual emitted IR is still freshly
    // reconstructed and proved below, including its keys and token ownership.
    if (receiptVersion != planVersion || provedTargets.size() != lanes.size())
        return fail(ConstructionResult::InternalError, "logical requirement receipt is incomplete or stale");
    // A single assignment interface sees every selected stream. Sharing is
    // permitted only with occurrence-specific consumption-before-rearm proof;
    // assignment does not alter publication or acquisition boundaries.
    std::map<std::tuple<Lane, Lane, unsigned>, Relation> assignments;
    // All assignment trials query the same logical completion plan. Key
    // proposals change the reuse requirement, not the payload handoffs.
    auto assignmentCompletion = completion(*supply);
    if (!assignmentCompletion)
        return false;
    for (auto& stream : streams) {
        bool assigned = false;
        for (unsigned k = 0; k < 8; ++k) {
            auto key = std::make_tuple(stream.pipes.first, stream.pipes.second, k);
            auto matching = stream.matching;
            if (auto found = assignments.find(key); found != assignments.end())
                matching.unionInPlace(found->second);
            auto participation = functional(matching);
            if (queryFailed(participation, "assignment participation query"))
                return false;
            if (participation != QueryStatus::Proved)
                continue;
            auto reuse = reuseSafe(matching, stream.pipes.first, *assignmentCompletion);
            if (queryFailed(reuse, "assignment consumption-before-rearm query"))
                return false;
            if (reuse != QueryStatus::Proved)
                continue;
            assignments.insert_or_assign(key, std::move(matching));
            stream.key = k;
            assigned = true;
            break;
        }
        if (!assigned)
            return fail(
                ConstructionResult::AllocationFailure,
                "no proved assignment within the event domain; boundaries retained");
    }
    auto at = [&](unsigned p, bool after, const Guard& guard, auto action) {
        auto* anchor = facts.points[p].operation;
        OpBuilder builder(anchor);
        if (after)
            builder.setInsertionPointAfter(anchor);
        bool shortCircuit = llvm::any_of(guard, [](const Clause& clause) {
            return llvm::any_of(clause, [](const BoundaryTest& test) {
                return test.kind == BoundaryTest::ParameterResidue;
            });
        });
        if (shortCircuit) {
            // An ordered DNF: failed clauses try the next one; the first true
            // clause emits exactly one action. Remainders execute only beneath
            // their nonnegative bound, which fresh definition-domain import
            // independently qualifies. No result-bearing scalar if is needed.
            std::function<void(OpBuilder&, unsigned, unsigned)> emit;
            emit = [&](OpBuilder& b, unsigned clause, unsigned literal) {
                if (clause == guard.size()) return;
                if (literal == guard[clause].size()) { action(b, anchor->getLoc()); return; }
                auto condition = lowering.emitGuard(b, anchor->getLoc(), Guard{Clause{guard[clause][literal]}});
                auto branch = b.create<scf::IfOp>(anchor->getLoc(), condition, true);
                OpBuilder yes = OpBuilder::atBlockBegin(&branch.getThenRegion().front());
                emit(yes, clause, literal + 1);
                OpBuilder no = OpBuilder::atBlockBegin(&branch.getElseRegion().front());
                emit(no, clause + 1, 0);
            };
            emit(builder, 0, 0);
            return;
        }
        Value condition = lowering.emitGuard(builder, anchor->getLoc(), guard);
        if (condition) {
            auto branch = builder.create<scf::IfOp>(anchor->getLoc(), condition, false);
            builder.setInsertionPointToStart(&branch.getThenRegion().front());
        }
        action(builder, anchor->getLoc());
    };
    for (const auto& endpoint : endpoints) {
        auto& stream = streams[endpoint.stream];
        at(endpoint.point, endpoint.publication, endpoint.guard, [&](OpBuilder& builder, Location location) {
            auto source = PipeAttr::get(function.getContext(), static_cast<PIPE>(stream.pipes.first));
            auto target = PipeAttr::get(function.getContext(), static_cast<PIPE>(stream.pipes.second));
            auto key = EventAttr::get(function.getContext(), static_cast<EVENT>(stream.key));
            if (endpoint.publication)
                builder.create<SetFlagOp>(location, source, target, key);
            else
                builder.create<WaitFlagOp>(location, source, target, key);
        });
    }
    for (unsigned i = 0; i < barriers.size(); ++i) {
        auto& barrier = barriers[i];
        at(barrier.point, false, barrierGuards[i], [&](OpBuilder& builder, Location location) {
            builder.create<BarrierOp>(location, PipeAttr::get(function.getContext(), static_cast<PIPE>(barrier.pipe)));
        });
    }
    if (emissionMutation)
        emissionMutation(function);
    if (failed(verify(function)))
        return fail(ConstructionResult::InternalError, "malformed logical emission");
    return reconstruct();
}

bool Constructor::reconstruct()
{
    if (!payload.preserved(function, [](Operation* op) {
            // The constructor owns only events, barriers and their scalar
            // guards. Extra allocations/views/resources are payload changes,
            // even when the physical-effect translator would not count them.
            return isa<
                SetFlagOp, WaitFlagOp, BarrierOp, scf::IfOp, scf::YieldOp, arith::ConstantOp, arith::CmpIOp,
                arith::AddIOp, arith::SubIOp, arith::RemSIOp, arith::AndIOp, arith::OrIOp, arith::XOrIOp>(op);
        }))
        return fail(
            ConstructionResult::InternalError, "emission changed original payload, control or allocation contract");
    // Re-translate actual payload effects after emission. Insertion metadata and
    // the selected plan's claims are not semantic input to this reconstruction.
    SyncIRs rebuiltIR;
    Buffer2MemInfoMap rebuiltBuffers;
    PTOIRTranslator translator(rebuiltIR, memory, rebuiltBuffers, function, SyncAnalysisMode::NORMALSYNC);
    translator.Build();
    std::map<Operation*, const CompoundInstanceElement*> rebuiltPhases;
    for (const auto& element : rebuiltIR)
        if (auto* phase = dyn_cast<CompoundInstanceElement>(element.get())) {
            if (!rebuiltPhases.emplace(phase->elementOp, phase).second)
                return fail(ConstructionResult::Unsupported, "reconstructed multi-phase operation");
        }
    if (rebuiltPhases.size() != phases.size())
        return fail(ConstructionResult::InternalError, "emission changed represented physical phases");
    auto sameEffects = [](const auto& a, const auto& b) {
        if (a.size() != b.size())
            return false;
        for (unsigned i = 0; i < a.size(); ++i)
            if (!(*a[i] == *b[i]))
                return false;
        return true;
    };
    for (auto* old : phases) {
        auto found = rebuiltPhases.find(old->elementOp);
        if (found == rebuiltPhases.end() || old->kPipeValue != found->second->kPipeValue ||
            !sameEffects(old->useVec, found->second->useVec) || !sameEffects(old->defVec, found->second->defVec))
            return fail(ConstructionResult::InternalError, "emission changed physical access contract");
    }
    SmallVector<Operation*> points;
    for (const auto& point : facts.points)
        points.push_back(point.operation);
    using Key = std::tuple<Lane, Lane, unsigned>;
    struct Events {
        SmallVector<unsigned> publications, acquisitions;
    };
    std::map<Key, Events> groups;
    SmallVector<std::pair<unsigned, Lane>> actualBarriers;
    function.walk([&](Operation* op) {
        if (auto set = dyn_cast<SetFlagOp>(op)) {
            auto key =
                Key{static_cast<Lane>(set.getSrcPipe().getPipe()), static_cast<Lane>(set.getDstPipe().getPipe()),
                    unsigned(set.getEventId().getEvent())};
            groups[key].publications.push_back(points.size());
            points.push_back(op);
        } else if (auto wait = dyn_cast<WaitFlagOp>(op)) {
            auto key =
                Key{static_cast<Lane>(wait.getSrcPipe().getPipe()), static_cast<Lane>(wait.getDstPipe().getPipe()),
                    unsigned(wait.getEventId().getEvent())};
            groups[key].acquisitions.push_back(points.size());
            points.push_back(op);
        } else if (auto barrier = dyn_cast<BarrierOp>(op)) {
            actualBarriers.push_back({points.size(), static_cast<Lane>(barrier.getPipe().getPipe())});
            points.push_back(op);
        }
    });
    auto actual = SyncOccurrences::build(function, points);
    if (!queries.spend(actual.work))
        return fail(ConstructionResult::AnalysisLimit, "emitted occurrence import work budget");
    if (!actual.complete)
        return fail(
            actual.limitExceeded ? ConstructionResult::AnalysisLimit : ConstructionResult::Unproved,
            "emitted occurrence import: " + actual.reason);
    if (actual.loops != facts.loops || actual.parameters != facts.parameters)
        return fail(ConstructionResult::Unproved, "emitted occurrence bindings differ from input");
    for (unsigned p = 0; p < facts.points.size(); ++p)
        if (!expect(
                queries.contains(actual.domain(p), facts.domain(p)),
                "emitted payload domain missing input occurrences") ||
            !expect(
                queries.contains(facts.domain(p), actual.domain(p)), "emitted payload domain adds input occurrences"))
            return false;
    std::map<std::pair<unsigned, unsigned>, Relation> actualOrder;
    auto before = [&](unsigned a, unsigned b) -> std::optional<Relation> {
        auto key = std::make_pair(a, b);
        if (auto found = actualOrder.find(key); found != actualOrder.end())
            return found->second;
        auto relation = take(actual.ordered(a, b));
        if (!relation)
            return {};
        auto normalized = take(queries.normalize(*relation));
        if (normalized)
            actualOrder.emplace(key, *normalized);
        return normalized;
    };
    auto supply = empty();
    std::vector<std::pair<Key, Relation>> actualMatching;
    for (const auto& [key, events] : groups) {
        auto [sourceLane, targetLane, keyNumber] = key;
        if (keyNumber >= 8 || sourceLane == targetLane)
            return fail(ConstructionResult::InternalError, "invalid emitted event domain");
        auto preceding = empty(), pubOrder = empty();
        auto pubDomain = PresburgerSet::getEmpty(actual.domain(0).getRangeSet().getSpace());
        auto waitDomain = pubDomain;
        for (unsigned p : events.publications) {
            pubDomain.unionInPlace(actual.domain(p).getRangeSet());
            for (unsigned w : events.acquisitions) {
                auto edge = before(p, w);
                if (!edge)
                    return false;
                preceding.unionInPlace(*edge);
            }
            for (unsigned q : events.publications) {
                auto edge = before(p, q);
                if (!edge)
                    return false;
                pubOrder.unionInPlace(*edge);
            }
        }
        for (unsigned w : events.acquisitions)
            waitDomain.unionInPlace(actual.domain(w).getRangeSet());
        auto matched = take(queries.latestSources(preceding, pubOrder));
        if (!matched)
            return false;
        auto inverse = *matched;
        inverse.inverse();
        auto consumers = compose(inverse, *matched), producers = compose(*matched, inverse);
        if (!consumers || !producers)
            return false;
        auto diagonal = empty();
        for (unsigned p : events.publications)
            diagonal.unionInPlace(actual.identity(p));
        for (unsigned w : events.acquisitions)
            diagonal.unionInPlace(actual.identity(w));
        if (!expect(queries.contains(diagonal, *consumers), "emitted publication consumed multiple times") ||
            !expect(queries.contains(diagonal, *producers), "emitted acquisition has multiple publications") ||
            !expect(queries.contains(matched->getDomainSet(), pubDomain), "emitted publication lacks acquisition") ||
            !expect(queries.contains(matched->getRangeSet(), waitDomain), "emitted acquisition lacks publication"))
            return false;
        auto prefix = empty(), suffix = empty();
        for (unsigned p = 0; p < lanes.size(); ++p) {
            if (lanes[p] == sourceLane)
                for (unsigned publication : events.publications) {
                    auto edge = before(p, publication);
                    if (!edge)
                        return false;
                    prefix.unionInPlace(*edge);
                }
            if (lanes[p] == targetLane)
                for (unsigned acquisition : events.acquisitions) {
                    auto edge = before(acquisition, p);
                    if (!edge)
                        return false;
                    suffix.unionInPlace(*edge);
                }
        }
        // Recover the actual captured source prefix and blocked destination cut.
        // Equality here checks that realization did not recapture independent
        // work or move a wait before an earlier destination operation.
        auto sourceBefore = laneOrder(sourceLane, false), targetBefore = laneOrder(targetLane, false);
        if (!sourceBefore || !targetBefore)
            return false;
        auto lastSource = take(queries.latestSources(prefix, *sourceBefore));
        if (!lastSource)
            return false;
        auto firstTarget = take(queries.firstTargets(suffix, *targetBefore));
        if (!firstTarget)
            return false;
        auto a = compose(*lastSource, *matched);
        if (!a)
            return false;
        auto logical = compose(*a, *firstTarget);
        if (!logical)
            return false;
        auto intended = empty();
        for (const auto& stream : streams)
            if (stream.pipes == Domain{sourceLane, targetLane} && stream.key == int(keyNumber))
                intended.unionInPlace(stream.matching);
        if (!expect(queries.contains(intended, *logical), "realization broadened selected handoff boundary") ||
            !expect(queries.contains(*logical, intended), "realization lost selected handoff boundary"))
            return false;
        supply.unionInPlace(*logical);
        actualMatching.push_back({key, std::move(*logical)});
    }
    if (actualBarriers.size() != barriers.size())
        return fail(ConstructionResult::InternalError, "emission changed selected barrier inventory");
    std::set<unsigned> checkedBarriers;
    for (auto [barrier, pipe] : actualBarriers) {
        auto prefix = empty(), suffix = empty();
        for (unsigned p = 0; p < lanes.size(); ++p)
            if (lanes[p] == pipe) {
                auto a = before(p, barrier), b = before(barrier, p);
                if (!a || !b)
                    return false;
                prefix.unionInPlace(*a);
                suffix.unionInPlace(*b);
            }
        auto beforeLane = laneOrder(pipe, false);
        if (!beforeLane)
            return false;
        auto last = take(queries.latestSources(prefix, *beforeLane));
        if (!last)
            return false;
        auto first = take(queries.firstTargets(suffix, *beforeLane));
        if (!first)
            return false;
        auto actualPrefix = compose(prefix, *first);
        if (!actualPrefix)
            return false;
        bool matchedBarrier = false;
        for (unsigned i = 0; i < barriers.size(); ++i) {
            if (checkedBarriers.count(i) || barriers[i].pipe != pipe)
                continue;
            auto domain = queries.contains(first->getRangeSet(), barriers[i].domain);
            if (queryFailed(domain, "reconstructed barrier domain query"))
                return false;
            if (domain != QueryStatus::Proved)
                continue;
            auto exact = queries.contains(barriers[i].domain, first->getRangeSet());
            if (queryFailed(exact, "reconstructed barrier domain query"))
                return false;
            if (exact != QueryStatus::Proved)
                continue;
            auto intendedPrefix = beforeLane->intersectRange(barriers[i].domain);
            if (!expect(
                    queries.contains(intendedPrefix, *actualPrefix), "realization broadened selected barrier cut") ||
                !expect(queries.contains(*actualPrefix, intendedPrefix), "realization lost selected barrier cut"))
                return false;
            checkedBarriers.insert(i);
            matchedBarrier = true;
            break;
        }
        if (!matchedBarrier)
            return fail(ConstructionResult::Unproved, "realization changed selected barrier domain");
        auto cut = compose(*last, *first);
        if (!cut)
            return false;
        supply.unionInPlace(*cut);
    }
    auto completed = completion(supply);
    if (!completed)
        return false;
    if (!expect(proveRequirements(*completed), "reconstructed ordering or exit retirement unavailable"))
        return false;
    for (const auto& [key, matching] : actualMatching)
        if (!expect(
                reuseSafe(matching, std::get<0>(key), *completed),
                "reconstructed consumption-before-rearm unavailable"))
            return false;
    return true;
}

namespace {
ConstructionResult construct(
    func::FuncOp function, InsertSyncGMAliasMode gm, uint64_t budget,
    llvm::function_ref<void(func::FuncOp)> mutate = {}, testing::RequirementObserver observe = {})
{
    OwningOpRef<ModuleOp> stage(ModuleOp::create(function.getLoc()));
    SmallVector<ModuleOp> ancestors;
    for (Operation* op = function->getParentOp(); op; op = op->getParentOp())
        if (auto module = dyn_cast<ModuleOp>(op))
            ancestors.push_back(module);
    ModuleOp parent = *stage;
    for (auto module : llvm::reverse(ancestors)) {
        auto child = ModuleOp::create(module.getLoc());
        child->setAttrs(module->getAttrs());
        parent.getBody()->push_back(child);
        parent = child;
    }
    IRMapping mapping;
    auto working = cast<func::FuncOp>(function->clone(mapping));
    parent.getBody()->push_back(working);
    auto result = Constructor(working, gm, budget, mutate, observe).run();
    if (result.status == ConstructionResult::Applied)
        function.getBody().takeBody(working.getBody());
    return result;
}
} // namespace

ConstructionResult mlir::pto::logical_sync::constructLogicalSync(
    func::FuncOp function, InsertSyncGMAliasMode gm, bool useMmad, uint64_t budget)
{
    (void)useMmad; // Qualified intrinsic ACC discharge is a later native milestone.
    return construct(function, gm, budget);
}

ConstructionResult mlir::pto::logical_sync::testing::constructWithEmissionMutation(
    func::FuncOp function, InsertSyncGMAliasMode gm, uint64_t budget, llvm::function_ref<void(func::FuncOp)> mutate,
    testing::RequirementObserver observe)
{
    return construct(function, gm, budget, mutate, observe);
}

bool mlir::pto::logical_sync::testing::guardEmissionFits(
    ArrayRef<unsigned> clauseSizes, bool shortCircuit, uint64_t allowance)
{
    Guard guard;
    for (unsigned size : clauseSizes) {
        if (size > 65 || guard.size() >= 65) return false;
        guard.emplace_back(size, BoundaryTest{shortCircuit ? BoundaryTest::ParameterResidue : BoundaryTest::First,
                                              0, true});
    }
    return chargeGuardEmission(guard, allowance);
}
