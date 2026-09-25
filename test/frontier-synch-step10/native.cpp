// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "PTO/Transforms/FrontierSynch/ProgramAnalysis.h"
#include "PTO/Transforms/FrontierSynch/ExactFrontiers.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <functional>
#include <iostream>
#include <map>
#include <set>
using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::frontiersynch;
using BoundaryStatus = OriginalBoundaryResult::Status;
using ValueStatus = OriginalValueQualification::Status;

// Verified, actual MLIR control/scalar programs with explicitly declared exact
// translated-effect fixtures. No arith opcode is classified as a memory access
// by production code. These tests exercise PUBLIC ProgramAnalysis; they are not
// substitutes for the full PTO importer/development-kernel gate.
struct Effect { std::size_t cell; bool read, write, full; PipelineType engine; };
struct Fixture {
    OwningOpRef<ModuleOp> module;
    OriginalStructure original;
    SyncInput input;
    const OriginalStructure* analyzed = nullptr;
    const OriginalStructure& structure() const { return analyzed ? *analyzed : original; }
    func::FuncOp function() const { return structure().function; }
    std::unique_ptr<ProgramAnalysis> analyze()
    {
        auto result = std::make_unique<ProgramAnalysis>(input, std::move(original));
        analyzed = &result->structure();
        return result;
    }
    std::map<std::string, mlir::Operation*> named;
    std::map<std::string, std::size_t> payloads;
    std::map<mlir::Operation*, std::size_t> sites;
    std::map<std::string, Effect> effects;
    std::vector<std::unique_ptr<CompoundInstanceElement>> phases;
    Fixture(MLIRContext& context, StringRef text, std::map<std::string, Effect> effects)
        : effects(std::move(effects))
    {
        module = parseSourceString<ModuleOp>(text, &context);
        assert(module && succeeded(verify(*module)));
        original.function = *module->getOps<func::FuncOp>().begin();
        original.function.walk<WalkOrder::PreOrder>([&](mlir::Operation* op) {
            if (op != original.function.getOperation()) {
                sites[op] = original.originalSites.size();
                original.originalSites.push_back(op);
            }
            if (auto loc = dyn_cast<NameLoc>(op->getLoc())) {
                named[loc.getName().getValue().str()] = op;
            }
        });
        original.cells.resize(2);
        original.body = block(original.function.getBody().front());
    }
    frontiersynch::Region block(Block& block)
    {
        frontiersynch::Region result;
        result.kind = frontiersynch::Region::Sequence;
        for (auto& op : block) {
            auto name = dyn_cast<NameLoc>(op.getLoc());
            const auto effect = name ? effects.find(name.getName().getValue().str()) : effects.end();
            if (effect != effects.end()) {
                const auto id = original.operations.size();
                const auto& e = effect->second;
                auto phase = std::make_unique<CompoundInstanceElement>(id, SmallVector<const BaseMemInfo*>{},
                    SmallVector<const BaseMemInfo*>{}, e.engine, op.getName());
                phase->elementOp = &op;
                original.operations.push_back({phase.get(), sites.at(&op),
                    {{e.cell, e.read, e.write, e.full}}, true, true, id});
                phases.push_back(std::move(phase));
                payloads[effect->first] = id;
                frontiersynch::Region child;
                child.kind = frontiersynch::Region::Operation;
                child.operation = id;
                result.children.push_back(std::move(child));
            } else if (auto choice = dyn_cast<scf::IfOp>(&op)) {
                frontiersynch::Region child;
                child.kind = frontiersynch::Region::Choice;
                child.originalOwner = sites.at(&op);
                for (auto& arm : op.getRegions()) {
                    child.children.push_back(arm.empty() ? frontiersynch::Region{} : this->block(arm.front()));
                }
                result.children.push_back(std::move(child));
            } else if (auto loop = dyn_cast<scf::ForOp>(&op)) {
                frontiersynch::Region child;
                child.kind = frontiersynch::Region::For;
                child.originalOwner = sites.at(&op);
                child.zeroTripPossible = true;
                child.qualifiedCounted = true; // arithmetic is independently rechecked by the service
                child.children.push_back(this->block(*loop.getBody()));
                result.children.push_back(std::move(child));
            }
        }
        return result;
    }
    mlir::Operation* op(const char* name) const { return named.at(name); }
    std::size_t payload(const char* name) const { return payloads.at(name); }
    std::string text() const
    {
        std::string out; llvm::raw_string_ostream stream(out); module.get().print(stream); stream.flush(); return out;
    }
    OriginalInterval interval(ProgramAnalysis& analysis, PipelineType engine, bool read = true, bool write = false,
                              bool includeTarget = false) const
    {
        OriginalIntervalRequest q;
        q.version = structure().version;
        q.start = {payload("W"), OriginalCut::After};
        q.stop = {payload("W2"), OriginalCut::Before};
        q.includeStoppingAccess = includeTarget;
        q.selector.cell = 0; q.selector.read = read; q.selector.write = write;
        q.selector.engine = unsigned(engine);
        q.occurrence.stopVisit = OriginalOccurrenceContext::StopVisit::FirstReach;
        q.occurrence.source = payload("W"); q.occurrence.target = payload("W2");
        const auto answer = analysis.prepareInterval(q);
        assert(answer.valid);
        return answer.interval;
    }
};
using Environment = DenseMap<Value, uint64_t>;
static unsigned width(Value value)
{
    return value.getType().isIndex() ? 64 : cast<IntegerType>(value.getType()).getWidth();
}
static std::optional<uint64_t> scalar(Value value, const Environment& environment)
{
    const auto found = environment.find(value);
    if (found != environment.end()) return found->second;
    if (auto c = value.getDefiningOp<arith::ConstantOp>()) {
        auto attr = dyn_cast<IntegerAttr>(c.getValue());
        return attr ? std::optional<uint64_t>(attr.getValue().getZExtValue()) : std::nullopt;
    }
    auto* op = value.getDefiningOp();
    if (!op || op->getNumOperands() != 2) return {};
    const auto a = scalar(op->getOperand(0), environment), b = scalar(op->getOperand(1), environment);
    if (!a || !b) return {};
    if (isa<arith::AndIOp>(op)) return *a & *b;
    if (isa<arith::OrIOp>(op)) return *a | *b;
    if (isa<arith::XOrIOp>(op)) return *a ^ *b;
    if (isa<arith::AddIOp>(op)) return (*a + *b) & value_arithmetic::Integer{width(value), true}.mask();
    if (isa<arith::RemUIOp>(op)) return *b ? std::optional<uint64_t>(*a % *b) : std::nullopt;
    if (auto cmp = dyn_cast<arith::CmpIOp>(op)) {
        const llvm::APInt x(width(cmp.getLhs()), *a), y(width(cmp.getRhs()), *b);
        using P = arith::CmpIPredicate;
        switch (cmp.getPredicate()) {
        case P::eq: return x == y; case P::ne: return x != y;
        case P::slt: return x.slt(y); case P::sle: return x.sle(y);
        case P::sgt: return x.sgt(y); case P::sge: return x.sge(y);
        case P::ult: return x.ult(y); case P::ule: return x.ule(y);
        case P::ugt: return x.ugt(y); case P::uge: return x.uge(y);
        }
    }
    return {};
}
static bool predicate(ProgramAnalysis& analysis, const Fixture& fixture, std::size_t root,
                      const Environment& environment)
{
    std::map<std::size_t, bool> memo;
    std::function<bool(std::size_t)> visit = [&](std::size_t id) -> bool {
        auto found = memo.find(id);
        if (found != memo.end()) return found->second;
        const auto node = analysis.predicate(id);
        bool out = false;
        switch (node.kind) {
        case ParticipationExpression::False: break;
        case ParticipationExpression::True: out = true; break;
        case ParticipationExpression::And: { auto a = visit(node.left), b = visit(node.right); out = a && b; break; }
        case ParticipationExpression::Or: { auto a = visit(node.left), b = visit(node.right); out = a || b; break; }
        case ParticipationExpression::Not: out = !visit(node.left); break;
        case ParticipationExpression::Atom: {
            const auto& atom = node.atom;
            if (atom.kind == ObservationAtom::OriginalBoolean) {
                auto branch = cast<scf::IfOp>(fixture.structure().originalSites.at(atom.owner));
                const auto value = scalar(branch.getCondition(), environment); assert(value);
                out = *value == atom.value;
            } else if (atom.kind == ObservationAtom::IntervalNonEmpty || atom.kind == ObservationAtom::IntervalFirst ||
                       atom.kind == ObservationAtom::IntervalLast) {
                const auto* record = analysis.intervalParticipation(atom.parameter); assert(record);
                const auto value = evaluateIntervalParticipation(*record, atom.kind, [&](Value v) { return scalar(v, environment); });
                assert(value); out = unsigned(*value) == atom.value;
            } else {
                auto loop = cast<scf::ForOp>(fixture.structure().originalSites.at(atom.owner));
                auto lo = scalar(loop.getLowerBound(), environment), hi = scalar(loop.getUpperBound(), environment);
                auto step = scalar(loop.getStep(), environment);
                assert(lo && hi && step);
                const value_arithmetic::Integer type{width(loop.getInductionVar()),
                    loop->hasAttr("unsignedCmp") || loop->hasAttr("unsigned_cmp")};
                if (atom.kind == ObservationAtom::LoopNonEmpty) {
                    out = unsigned(type.less(*lo, *hi)) == atom.value;
                } else {
                    auto iv = scalar(loop.getInductionVar(), environment); assert(iv);
                    const auto value = atom.kind == ObservationAtom::LoopHasPrevious ?
                        value_arithmetic::hasPrevious(type, *lo, *hi, *step, *iv, atom.parameter) :
                        value_arithmetic::hasNext(type, *lo, *hi, *step, *iv, atom.parameter);
                    assert(value); out = unsigned(*value) == atom.value;
                }
            }
            break;
        }
        default: assert(false && "invalid production predicate");
        }
        memo.emplace(id, out); return out;
    };
    return visit(root);
}
using Occurrence = std::pair<std::size_t, uint64_t>;
static std::vector<Occurrence> endpointVisits(ProgramAnalysis& analysis, const Fixture& fixture, std::size_t frontier,
                                             const Environment& environment, uint64_t coordinate)
{
    std::vector<Occurrence> out;
    std::function<void(std::size_t)> walk = [&](std::size_t id) {
        const auto node = analysis.readerFrontier(id);
        switch (node.kind) {
        case GuardedReadFrontier::Empty: return;
        case GuardedReadFrontier::Access: out.emplace_back(node.operation, coordinate); return;
        case GuardedReadFrontier::Union: walk(node.left); walk(node.right); return;
        case GuardedReadFrontier::Guard: if (predicate(analysis, fixture, node.predicate, environment)) walk(node.left); return;
        default: assert(false && "invalid production frontier");
        }
    };
    walk(frontier); return out;
}
static void optionalReaders(MLIRContext& context)
{
    Fixture f(context, R"mlir(module {
      func.func @optional(%g: i1, %h: i1) {
        %z = arith.constant 0 : index
        %o = arith.constant 1 : index
        %w = arith.addi %z, %o : index loc("W")
        scf.if %g { %a = arith.addi %z, %o : index loc("A") }
        scf.if %h { %b = arith.addi %z, %o : index loc("B") }
        %c = arith.addi %z, %o : index loc("C")
        %u = arith.addi %z, %o : index loc("U")
        %w2 = arith.addi %z, %o : index loc("W2")
        return
      }
    })mlir", {{"W", {0,false,true,true,PipelineType::PIPE_MTE2}}, {"W2", {0,false,true,true,PipelineType::PIPE_MTE2}},
               {"A", {0,true,false,false,PipelineType::PIPE_V}}, {"B", {0,true,false,false,PipelineType::PIPE_V}},
               {"C", {0,true,false,false,PipelineType::PIPE_MTE3}}, {"U", {1,true,false,false,PipelineType::PIPE_V}}});
    const auto ir = f.text();
    auto owned = f.analyze();
    auto& analysis = *owned; assert(analysis.complete());
    const auto obligations = analysis.obligationsAt(f.sites.at(f.op("W2")));
    auto q = f.interval(analysis, PipelineType::PIPE_V);
    const auto first = analysis.firstConflict(q, true), last = analysis.lastRelevantUse(q, true);
    assert(first.status == BoundaryStatus::Exact && last.status == BoundaryStatus::Exact);
    const auto third = analysis.lastRelevantUse(f.interval(analysis, PipelineType::PIPE_MTE3), true);
    assert(third.status == BoundaryStatus::Exact && third.cuts.size() == 1 && third.cuts[0].operation == f.payload("C"));
    for (bool g : {false,true}) for (bool h : {false,true}) {
        Environment env;
        env[f.function().getArgument(0)] = g; env[f.function().getArgument(1)] = h;
        std::vector<Occurrence> concrete;
        if (g) concrete.emplace_back(f.payload("A"), 0);
        if (h) concrete.emplace_back(f.payload("B"), 0);
        const auto a = endpointVisits(analysis, f, first.frontier, env, 0);
        const auto b = endpointVisits(analysis, f, last.frontier, env, 0);
        assert(a.size() == unsigned(!concrete.empty()) && b.size() == unsigned(!concrete.empty()));
        assert(predicate(analysis, f, first.noHit, env) == concrete.empty());
        if (!concrete.empty()) assert(a[0] == concrete.front() && b[0] == concrete.back());
    }
    auto writes = f.interval(analysis, PipelineType::PIPE_MTE2, false, true);
    assert(analysis.firstConflict(writes).status == BoundaryStatus::NoHit);
    writes.query.includeStoppingAccess = true;
    const auto conflict = analysis.firstConflict(writes);
    assert(conflict.status == BoundaryStatus::Exact && conflict.cuts.size() == 1 &&
           conflict.cuts[0] == (OriginalCut{f.payload("W2"), OriginalCut::Before}));
    assert(analysis.endpointCandidates(first, SourceMilestone::After).empty());
    assert(!analysis.endpointCandidates(first, SourceMilestone::Before).empty());
    const auto stats = analysis.exactFrontierStats();
    analysis.firstConflict(q, true); analysis.lastRelevantUse(q, true);
    assert(stats.summaryQueries == analysis.exactFrontierStats().summaryQueries);
    auto stale = q; stale.query.version = OriginalProgramVersion::fresh();
    assert(analysis.firstConflict(stale).status == BoundaryStatus::Unknown);
    assert(analysis.obligationsAt(f.sites.at(f.op("W2"))) == obligations && f.text() == ir);
}
static void intervalLoop(MLIRContext& context)
{
    Fixture f(context, R"mlir(module {
      func.func @interval(%n: index, %lo: index, %hi: index, %enabled: i1) {
        %z = arith.constant 0 : index
        %o = arith.constant 1 : index
        %w = arith.addi %z, %o : index loc("W")
        scf.for %i = %z to %n step %o {
          %l = arith.cmpi sge, %i, %lo : index
          %h = arith.cmpi slt, %i, %hi : index
          %both = arith.andi %l, %h : i1
          %condition = arith.andi %both, %enabled : i1
          scf.if %condition { %r = arith.addi %z, %o : index loc("R") }
        } loc("loop")
        %u = arith.addi %z, %o : index loc("U")
        %w2 = arith.addi %z, %o : index loc("W2")
        return
      }
    })mlir", {{"W", {0,false,true,true,PipelineType::PIPE_MTE2}}, {"W2", {0,false,true,true,PipelineType::PIPE_MTE2}},
               {"R", {0,true,false,false,PipelineType::PIPE_V}}, {"U", {1,true,false,false,PipelineType::PIPE_V}}});
    const auto ir = f.text();
    auto owned = f.analyze();
    auto& analysis = *owned; assert(analysis.complete());
    const auto q = f.interval(analysis, PipelineType::PIPE_V);
    const auto first = analysis.firstConflict(q, true), last = analysis.lastRelevantUse(q, true);
    assert(first.status == BoundaryStatus::Exact && last.status == BoundaryStatus::Exact);
    assert(first.endpointQualification.available() && last.endpointQualification.available());
    auto loop = cast<scf::ForOp>(f.op("loop"));
    for (uint64_t n = 0; n < 10; ++n) for (int64_t lo : {-2,0,2,8,11})
      for (int64_t hi : {0,1,3,7,12}) for (bool enabled : {false,true}) {
        Environment env;
        env[f.function().getArgument(0)] = n;
        env[f.function().getArgument(1)] = uint64_t(lo);
        env[f.function().getArgument(2)] = uint64_t(hi);
        env[f.function().getArgument(3)] = enabled;
        std::vector<Occurrence> concrete, starts, ends;
        for (uint64_t j = 0; j < n; ++j) {
            env[loop.getInductionVar()] = j;
            if (enabled && lo <= int64_t(j) && int64_t(j) < hi) concrete.emplace_back(f.payload("R"), j);
            auto a = endpointVisits(analysis, f, first.frontier, env, j);
            auto b = endpointVisits(analysis, f, last.frontier, env, j);
            starts.insert(starts.end(), a.begin(), a.end()); ends.insert(ends.end(), b.begin(), b.end());
        }
        assert(starts.size() == unsigned(!concrete.empty()) && ends.size() == unsigned(!concrete.empty()));
        assert(predicate(analysis, f, first.noHit, env) == concrete.empty());
        if (!concrete.empty()) assert(starts[0] == concrete.front() && ends[0] == concrete.back());
    }
    assert(analysis.exactFrontierStats().intervalRecipes > 0 && f.text() == ir);
}
static void unavailable(MLIRContext& context)
{
    Fixture f(context, R"mlir(module {
      func.func @later(%x: index) {
        %z = arith.constant 0 : index
        %o = arith.constant 1 : index
        %w = arith.addi %z, %o : index loc("W")
        %a = arith.addi %z, %o : index loc("A")
        %g = arith.cmpi sgt, %x, %z : index
        scf.if %g { %b = arith.addi %z, %o : index loc("B") }
        %u = arith.addi %z, %o : index loc("U")
        %w2 = arith.addi %z, %o : index loc("W2")
        return
      }
    })mlir", {{"W", {0,false,true,true,PipelineType::PIPE_MTE2}}, {"W2", {0,false,true,true,PipelineType::PIPE_MTE2}},
               {"A", {0,true,false,false,PipelineType::PIPE_V}}, {"B", {0,true,false,false,PipelineType::PIPE_V}},
               {"U", {1,true,false,false,PipelineType::PIPE_V}}});
    auto owned = f.analyze();
    auto& analysis = *owned;
    const auto q = f.interval(analysis, PipelineType::PIPE_V);
    const auto first = analysis.firstConflict(q, true), last = analysis.lastRelevantUse(q, true);
    assert(first.status == BoundaryStatus::Exact && first.cuts.size() == 1 && first.cuts[0].operation == f.payload("A"));
    assert(last.status == BoundaryStatus::Unknown && last.endpointQualification.status == ValueStatus::NotObservableHere);
    assert(last.mayAccesses && !last.mayAccesses->empty());
    assert(analysis.exactFrontiers(q, true).status == OriginalExactFrontiers::Status::Exact);
}
static void conservativeCases(MLIRContext& context)
{
    Fixture f(context, R"mlir(module {
      func.func @varying(%n: index) {
        %z = arith.constant 0 : index
        %o = arith.constant 1 : index
        %two = arith.constant 2 : index
        %w = arith.addi %z, %o : index loc("W")
        scf.for %i = %z to %n step %o {
          %residue = arith.remui %i, %two : index
          %g = arith.cmpi eq, %residue, %z : index
          scf.if %g { %r = arith.addi %z, %o : index loc("R") }
        }
        %w2 = arith.addi %z, %o : index loc("W2")
        return
      }
    })mlir", {{"W", {0,false,true,true,PipelineType::PIPE_MTE2}}, {"W2", {0,false,true,true,PipelineType::PIPE_MTE2}},
               {"R", {0,true,false,false,PipelineType::PIPE_V}}});
    auto owned = f.analyze();
    auto& analysis = *owned;
    const auto obligations = analysis.obligationsAt(f.sites.at(f.op("W2")));
    const auto q = f.interval(analysis, PipelineType::PIPE_V);
    const auto last = analysis.lastRelevantUse(q, true);
    assert(last.status == BoundaryStatus::Unknown && last.mayAccesses && !last.mayAccesses->empty());
    assert(analysis.obligationsAt(f.sites.at(f.op("W2"))) == obligations);
    Fixture split(context, R"mlir(module {
      func.func @split() {
        %z = arith.constant 0 : index
        %o = arith.constant 1 : index
        %w = arith.addi %z, %o : index loc("W")
        %a = arith.addi %z, %o : index loc("A")
        %x = arith.addi %z, %o : index loc("X")
        %b = arith.addi %z, %o : index loc("B")
        %w2 = arith.addi %z, %o : index loc("W2")
        return
      }
    })mlir", {{"W", {0,false,true,true,PipelineType::PIPE_MTE2}}, {"W2", {0,false,true,true,PipelineType::PIPE_MTE2}},
               {"A", {0,true,false,false,PipelineType::PIPE_V}}, {"B", {0,true,false,false,PipelineType::PIPE_V}},
               {"X", {0,false,true,false,PipelineType::PIPE_MTE2}}});
    auto splitOwned = split.analyze();
    auto& splitAnalysis = *splitOwned;
    assert(splitAnalysis.firstConflict(split.interval(splitAnalysis, PipelineType::PIPE_V), true).status == BoundaryStatus::Unknown);
    assert(splitAnalysis.firstConflict(split.interval(splitAnalysis, PipelineType::PIPE_V), false).status == BoundaryStatus::Exact);
    const auto partial = splitAnalysis.firstConflict(split.interval(splitAnalysis, PipelineType::PIPE_MTE2, false, true));
    assert(partial.status == BoundaryStatus::Unknown && partial.mayAccesses && !partial.mayAccesses->empty());
}
static void countedAndSingleton(MLIRContext& context)
{
    Fixture f(context, R"mlir(module {
      func.func @counted(%n: index, %g: i1) {
        %z = arith.constant 0 : index
        %o = arith.constant 1 : index
        %w = arith.addi %z, %o : index loc("W")
        scf.for %i = %z to %n step %o {
          scf.if %g { %r = arith.addi %z, %o : index loc("R") }
        } loc("loop")
        %w2 = arith.addi %z, %o : index loc("W2")
        return
      }
    })mlir", {{"W", {0,false,true,true,PipelineType::PIPE_MTE2}}, {"W2", {0,false,true,true,PipelineType::PIPE_MTE2}},
               {"R", {0,true,false,false,PipelineType::PIPE_V}}});
    auto owned = f.analyze(); auto& analysis = *owned;
    const auto q = f.interval(analysis, PipelineType::PIPE_V);
    const auto first = analysis.firstConflict(q, true), last = analysis.lastRelevantUse(q, true);
    assert(first.status == BoundaryStatus::Exact && last.status == BoundaryStatus::Exact);
    auto loop = cast<scf::ForOp>(f.op("loop"));
    for (uint64_t n : {0,1,2,9}) for (bool g : {false,true}) {
        Environment env;
        env[f.function().getArgument(0)] = n;
        env[f.function().getArgument(1)] = g;
        std::vector<Occurrence> starts, ends, concrete;
        for (uint64_t j = 0; j < n; ++j) {
            env[loop.getInductionVar()] = j;
            if (g) concrete.emplace_back(f.payload("R"), j);
            auto a = endpointVisits(analysis, f, first.frontier, env, j);
            auto b = endpointVisits(analysis, f, last.frontier, env, j);
            starts.insert(starts.end(), a.begin(), a.end());
            ends.insert(ends.end(), b.begin(), b.end());
        }
        assert(starts.size() == unsigned(!concrete.empty()) && ends.size() == unsigned(!concrete.empty()));
        assert(predicate(analysis, f, first.noHit, env) == concrete.empty());
        if (!concrete.empty()) assert(starts.front() == concrete.front() && ends.front() == concrete.back());
    }
    assert(analysis.exactFrontierStats().intervalRecipes == 0); // D3, not I.2 guessing

    // Equality is the singleton interval [k,k+1), only after proving +1 safe.
    for (const auto* k : {"2", "9223372036854775807"}) {
        std::string source = R"mlir(module {
          func.func @singleton(%n: index) {
            %z = arith.constant 0 : index
            %o = arith.constant 1 : index
            %k = arith.constant K : index
            %w = arith.addi %z, %o : index loc("W")
            scf.for %i = %z to %n step %o {
              %g = arith.cmpi eq, %i, %k : index
              scf.if %g { %r = arith.addi %z, %o : index loc("R") }
            } loc("loop")
            %w2 = arith.addi %z, %o : index loc("W2")
            return
          }
        })mlir";
        source.replace(source.find("constant K"), 10, std::string("constant ") + k);
        Fixture singleton(context, source,
            {{"W", {0,false,true,true,PipelineType::PIPE_MTE2}}, {"W2", {0,false,true,true,PipelineType::PIPE_MTE2}},
             {"R", {0,true,false,false,PipelineType::PIPE_V}}});
        auto singleOwned = singleton.analyze(); auto& single = *singleOwned;
        const auto domain = singleton.interval(single, PipelineType::PIPE_V);
        const auto start = single.firstConflict(domain, true), end = single.lastRelevantUse(domain, true);
        if (std::string(k) != "2") {
            assert(start.status == BoundaryStatus::Unknown && end.status == BoundaryStatus::Unknown);
            assert(start.reason.find("+1") != std::string::npos);
            continue;
        }
        assert(start.status == BoundaryStatus::Exact && end.status == BoundaryStatus::Exact);
        auto repeat = cast<scf::ForOp>(singleton.op("loop"));
        for (uint64_t n = 0; n < 7; ++n) {
            Environment env; env[singleton.function().getArgument(0)] = n;
            std::vector<Occurrence> a, b;
            for (uint64_t j = 0; j < n; ++j) {
                env[repeat.getInductionVar()] = j;
                auto x = endpointVisits(single, singleton, start.frontier, env, j);
                auto y = endpointVisits(single, singleton, end.frontier, env, j);
                a.insert(a.end(), x.begin(), x.end()); b.insert(b.end(), y.begin(), y.end());
            }
            assert(a.size() == unsigned(n > 2) && b.size() == unsigned(n > 2));
            if (n > 2) assert(a.front().second == 2 && b.front().second == 2);
        }
    }
}
static void completionAndLegalCuts(MLIRContext& context)
{
    Fixture f(context, R"mlir(module {
      func.func @completion(%x: index) {
        %z = arith.constant 0 : index
        %o = arith.constant 1 : index
        %w = arith.addi %z, %o : index loc("W")
        %g = arith.cmpi sgt, %x, %z : index loc("G")
        %a = arith.addi %z, %o : index loc("A")
        scf.if %g { %b = arith.addi %z, %o : index loc("B") }
        %w2 = arith.addi %z, %o : index loc("W2")
        return
      }
    })mlir", {{"W", {0,false,true,true,PipelineType::PIPE_MTE2}}, {"W2", {0,false,true,true,PipelineType::PIPE_MTE2}},
               {"G", {1,false,true,true,PipelineType::PIPE_MTE3}}, {"A", {0,true,false,false,PipelineType::PIPE_V}},
               {"B", {0,true,false,false,PipelineType::PIPE_V}}});
    auto owned = f.analyze(); auto& analysis = *owned;
    const auto domain = f.interval(analysis, PipelineType::PIPE_V);
    const auto last = analysis.lastRelevantUse(domain, true);
    // The earlier A source can obtain g through an independent prerequisite.
    // The B endpoint's lexical condition, by contrast, cannot circularly enable
    // g. The aggregate can be unresolved, but must retain A's named prerequisite.
    assert(last.status == BoundaryStatus::Unknown);
    assert(!last.endpointQualification.prerequisites.empty());
    const auto raw = analysis.exactFrontiers(domain, true);
    std::function<void(std::size_t, std::size_t)> check = [&](std::size_t root, std::size_t guard) {
        const auto node = analysis.readerFrontier(root);
        if (node.kind == GuardedReadFrontier::Union) { check(node.left, guard); check(node.right, guard); }
        else if (node.kind == GuardedReadFrontier::Guard) check(node.left, node.predicate);
        else if (node.kind == GuardedReadFrontier::Access && node.operation == f.payload("A")) {
            const auto qualified = analysis.guardQualificationAt(guard, node.operation, SourceMilestone::After);
            assert(qualified.status == ValueStatus::NeedsCompletion && qualified.prerequisites.size() == 1);
            assert(qualified.prerequisites.front().sourcePhase == f.payload("G"));
            assert(qualified.prerequisites.front().applicability.empty());
        }
    };
    check(raw.last, 1);

    Fixture phases(context, R"mlir(module {
      func.func @cuts() {
        %z = arith.constant 0 : index
        %o = arith.constant 1 : index
        %w = arith.addi %z, %o : index loc("W")
        %a = arith.addi %z, %o : index loc("A")
        %w2 = arith.addi %z, %o : index loc("W2")
        return
      }
    })mlir", {{"W", {0,false,true,true,PipelineType::PIPE_MTE2}}, {"W2", {0,false,true,true,PipelineType::PIPE_MTE2}},
               {"A", {0,true,false,false,PipelineType::PIPE_V}}});
    // This fixture withholds the internal after gap; it does not fabricate a
    // lowering that exposes a partial instruction's completion.
    phases.original.operations[phases.payload("A")].afterExecutable = false;
    auto phasesOwned = phases.analyze(); auto& cuts = *phasesOwned;
    const auto interval = phases.interval(cuts, PipelineType::PIPE_V);
    assert(cuts.firstConflict(interval, true).status == BoundaryStatus::Exact);
    assert(cuts.lastRelevantUse(interval, true).status == BoundaryStatus::Unknown);
}

int main()
{
    MLIRContext context;
    context.loadDialect<arith::ArithDialect, scf::SCFDialect, func::FuncDialect>();
    optionalReaders(context); intervalLoop(context); unavailable(context); conservativeCases(context);
    countedAndSingleton(context); completionAndLegalCuts(context);
    std::cout << "PASS: native ProgramAnalysis exact frontiers, interval participation, independent availability and retained obligations\n";
}
