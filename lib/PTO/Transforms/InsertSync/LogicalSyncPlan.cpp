// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
// See LICENSE for details. Provided AS IS, WITHOUT WARRANTIES OF ANY KIND.
#include "PTO/Transforms/InsertSync/LogicalSyncPlan.h"
#include "PTO/Transforms/InsertSync/LifecycleSynthesis.h"
#include "PTO/Transforms/InsertSync/MemoryDependentAnalyzer.h"
#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"
#include "PTO/Transforms/InsertSync/SyncEffectCoverage.h"
#include "PTO/Transforms/InsertSync/SyncPayloadSnapshot.h"
#include "mlir/Analysis/FlatLinearValueConstraints.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Verifier.h"
#include <map>
#include <set>
#include <cstdlib>

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
        memory.setGMContract(function, gm);
        PTOIRTranslator translator(ir, memory, buffers, function, SyncAnalysisMode::NORMALSYNC);
        translator.enableGenerationFlow();
        translator.Build();
        auto coverage = inspectInsertSyncEffectCoverage(function, ir, false);
        if (failed(coverage))
            return fail(ConstructionResult::InternalError, "effect inspection failed");
        if (!*coverage)
            return fail(ConstructionResult::Unsupported, "unqualified physical effects");
        insert_sync_frontier::Budget importBudget;
        auto importAllowance = queries.remainingWork();
        importBudget.left = importAllowance;
        auto physical = buildInsertSyncLifecycleStructure(function, ir, importBudget, false);
        queries.spend(importAllowance - importBudget.left);
        if (physical.status != StorageFrontierSnapshot::Status::Complete)
            return fail(
                physical.status == StorageFrontierSnapshot::Status::InternalError ? ConstructionResult::InternalError :
                physical.status == StorageFrontierSnapshot::Status::AnalysisLimit ? ConstructionResult::AnalysisLimit :
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
            if (a && b && !memory.MemAlias(a, b))
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
            completed = completion(*supply);
            if (!completed)
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
    QueryStatus reuseSafe(const Relation& matching, Lane pipe, const Relation& supply)
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
        auto completed = completion(supply);
        if (!completed)
            return result.status == ConstructionResult::AnalysisLimit ? QueryStatus::BudgetExhausted :
                                                                        QueryStatus::Unsupported;
        return completed->prove(*required.relation, queries, 2);
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
    enum Kind { First, Last, Nonempty, Predicate, BooleanParameter } kind;
    unsigned loop;
    bool truth;
};
using Clause = SmallVector<BoundaryTest, 2>;
using Guard = SmallVector<Clause, 2>;
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
        if (test.kind == BoundaryTest::BooleanParameter)
            expression = getAffineSymbolExpr(test.loop, context) - 1;
        else {
            auto loop = facts.loopDomains[test.loop];
            auto iv = getAffineDimExpr(test.loop, context);
            expression = test.kind == BoundaryTest::First ? iv - loop.lower :
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
            if (equality)
                flat.addEquality(rows.front());
            else
                flat.addInequality(rows.front());
            return facts.domain(point).intersect(Relation(flat));
        };
        auto positive =
            build(expression, test.kind == BoundaryTest::First || test.kind == BoundaryTest::BooleanParameter);
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
            auto unchanged = queries.contains(*rest.relation, remaining);
            if (queryFailed(unchanged))
                return false;
            if (unchanged == QueryStatus::Proved)
                return true;
            guard.push_back(std::move(clause));
            remaining = std::move(*rest.relation);
            return true;
        };
        for (const auto& a : candidates) {
            if (!admit(a.domain, Clause{a.test}))
                return {};
            if (remaining.isIntegerEmpty())
                return guard;
        }
        for (unsigned i = 0; i < candidates.size(); ++i)
            for (unsigned j = i + 1; j < candidates.size(); ++j) {
                auto domain = candidates[i].domain.intersect(candidates[j].domain);
                if (!admit(domain, Clause{candidates[i].test, candidates[j].test}))
                    return {};
                if (remaining.isIntegerEmpty())
                    return guard;
            }
        return {};
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
                else {
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
            auto acquisition = lowering.prepare(p, domain);
            if (!acquisition)
                return expect(lowering.status(), "acquisition domain has no qualified boundary lowering");
            endpoints.push_back({p, false, i, std::move(*acquisition)});
        }
    }
    for (const auto& barrier : barriers) {
        auto guard = lowering.prepare(barrier.point, barrier.domain);
        if (!guard)
            return expect(lowering.status(), "barrier domain has no qualified boundary lowering");
        barrierGuards.push_back(std::move(*guard));
    }
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
            auto reuse = reuseSafe(matching, stream.pipes.first, *supply);
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
                arith::AddIOp, arith::AndIOp, arith::OrIOp, arith::XOrIOp>(op);
        }))
        return fail(
            ConstructionResult::InternalError, "emission changed original payload, control or allocation contract");
    // Re-translate actual payload effects after emission. Insertion metadata and
    // the selected plan's claims are not semantic input to this reconstruction.
    SyncIRs rebuiltIR;
    Buffer2MemInfoMap rebuiltBuffers;
    PTOIRTranslator translator(rebuiltIR, memory, rebuiltBuffers, function, SyncAnalysisMode::NORMALSYNC);
    translator.enableGenerationFlow();
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
                reuseSafe(matching, std::get<0>(key), supply), "reconstructed consumption-before-rearm unavailable"))
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
