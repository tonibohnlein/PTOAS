// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Evaluation-aware cut specialization never replays opaque predicate loads.
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/IterationPredicates.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/OwningOpRef.h"
using namespace mlir;
namespace fs = pto::frontiersynch;
namespace {
struct Fixture {
    explicit Fixture(MLIRContext* context, unsigned trips)
        : module(ModuleOp::create(UnknownLoc::get(context)))
    {
        OpBuilder b(context); auto loc = b.getUnknownLoc();
        b.setInsertionPointToEnd(module->getBody());
        auto type = MemRefType::get({2}, b.getI32Type());
        function = b.create<func::FuncOp>(loc, "evaluations", b.getFunctionType({type}, {}));
        b.setInsertionPointToStart(function.addEntryBlock());
        auto lower = b.create<arith::ConstantIndexOp>(loc, 11);
        auto upper = b.create<arith::ConstantIndexOp>(loc, 11 + 3 * trips);
        auto step = b.create<arith::ConstantIndexOp>(loc, 3);
        auto zero = b.create<arith::ConstantIndexOp>(loc, 0);
        auto one = b.create<arith::ConstantIndexOp>(loc, 1);
        auto integerZero = b.create<arith::ConstantIntOp>(loc, 0, 32);
        loop = b.create<scf::ForOp>(loc, lower, upper, step);
        b.setInsertionPointToStart(loop.getBody());
        early = b.create<arith::ConstantIndexOp>(loc, 42);
        auto first = b.create<memref::LoadOp>(loc, function.getArgument(0), ValueRange{zero});
        p = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, first, integerZero);
        auto branch = b.create<scf::IfOp>(loc, p, true);
        b.setInsertionPointToStart(branch.thenBlock());
        thenCut = b.create<arith::ConstantIndexOp>(loc, 43);
        b.setInsertionPointToStart(branch.elseBlock());
        elseCut = b.create<arith::ConstantIndexOp>(loc, 44);
        b.setInsertionPointAfter(branch);
        auto second = b.create<memref::LoadOp>(loc, function.getArgument(0), ValueRange{one});
        r = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, second, integerZero);
        late = loop.getBody()->getTerminator();
        b.setInsertionPointAfter(loop); b.create<func::ReturnOp>(loc);
    }
    OwningOpRef<ModuleOp> module;
    func::FuncOp function;
    scf::ForOp loop;
    Value p, r;
    Operation *early, *thenCut, *elseCut, *late;
};
std::string render(Operation* op)
{
    std::string text; llvm::raw_string_ostream out(text); op->print(out); return text;
}
struct Preparation {
    Block detached;
    fs::RegionExpressions::CutEmission context;
};
bool attempt(fs::IterationPredicates& predicates, fs::RegionExpressions& e,
             fs::RegionExpressions::Id root, Operation* cut, bool expected,
             std::optional<uint64_t> constant = {}, Preparation* retained = nullptr)
{
    Preparation fresh;
    auto& detached = retained ? retained->detached : fresh.detached;
    auto& context = retained ? retained->context : fresh.context;
    OpBuilder b(cut->getContext()); b.setInsertionPointToEnd(&detached);
    auto selected = predicates.prepare(root, b, cut, context);
    const bool wrongConstant = succeeded(selected) && constant && e.constantValue(*selected) != constant;
    if (wrongConstant) { return false; }
    auto value = succeeded(selected) ? e.emitContextual(*selected, b, cut, context) : FailureOr<Value>(failure());
    const bool available = succeeded(value);
    if (available != expected) { return false; }
    // Loads may be reused at a legal cut, but never cloned into preparation.
    for (auto& op : detached) { if (isa<memref::LoadOp>(op)) { return false; } }
    return true;
}
bool provenance(Fixture& f)
{
    fs::RegionExpressions source;
    fs::RegionExpressions::Id p, r, past, ordinal;
    {
        fs::IterationPredicates local(f.loop, source, {});
        ordinal = fs::CountedLoop::get(f.loop)->ordinal(source);
        p = local.at(f.p, ordinal);
        r = local.at(f.r, ordinal);
        past = local.at(f.p, source.sub(ordinal, source.constant(1)));
    }
    // The arena keeps symbols and provenance after the producer is destroyed.
    const auto recipe = source.evaluation(p);
    const auto earlier = source.evaluation(past);
    if (!recipe || !earlier || recipe->original != f.p || recipe->loop != f.loop ||
        recipe->ordinal != ordinal || earlier->ordinal == ordinal || p == past) { return false; }
    fs::RegionExpressions::Substitution shift({{ordinal, source.constant(0)}});
    if (source.substitute(p, shift) != p || source.evaluation(p)->ordinal != ordinal) { return false; }
    auto placeholder = source.referencedInputs(p).front().second;
    fs::RegionExpressions destination;
    // Pre-intern the Input: import must not assume ordinal IDs precede it.
    auto targetInput = destination.input(placeholder);
    for (unsigned i = 0; i < 13; ++i) { destination.constant(100 + i); }
    auto imported = destination.import(source, {p, past});
    if (failed(imported) || (*imported)[0] != targetInput) { return false; }
    auto mapped = destination.evaluation(targetInput);
    auto importedOrdinal = destination.import(source, {ordinal});
    if (!mapped || failed(importedOrdinal) || mapped->ordinal != (*importedOrdinal)[0] ||
        mapped->ordinal == ordinal || mapped->original != f.p || mapped->loop != f.loop) { return false; }
    if (destination.evaluatedInput(placeholder, f.p, f.loop, mapped->ordinal) != targetInput) { return false; }
    {
        Block detached;
        OpBuilder builder(f.function.getContext()); builder.setInsertionPointToEnd(&detached);
        auto folded = builder.create<arith::ConstantIntOp>(f.loop.getLoc(), 1, 1);
        fs::RegionExpressions::Transaction rejectedConstant(destination);
        if (destination.evaluatedInput(folded, f.p, f.loop, mapped->ordinal) != destination.invalid) { return false; }
    }
    const auto count = destination.size();
    {
        fs::RegionExpressions::Transaction outer(destination);
        auto symbols = std::make_shared<Block>(); destination.retainInputOwner(symbols);
        auto value = symbols->addArgument(f.p.getType(), f.loop.getLoc());
        auto fresh = destination.evaluatedInput(value, f.r, f.loop, destination.constant(42));
        if (!destination.evaluation(fresh)) { return false; }
        {
            fs::RegionExpressions::Transaction inner(destination);
            if (destination.evaluatedInput(placeholder, f.r, f.loop, mapped->ordinal) != destination.invalid) {
                return false;
            }
        }
        if (!destination.constructionError().empty() || !destination.evaluation(fresh)) { return false; }
    }
    if (destination.size() != count || destination.evaluation(count)) { return false; }
    // A failed import may have registered an existing Input before discovering
    // another conflicting record. Both metadata and expression IDs roll back.
    fs::RegionExpressions rejected;
    auto existing = rejected.input(placeholder);
    auto zero = rejected.constant(0);
    auto other = source.referencedInputs(r).front().second;
    rejected.evaluatedInput(other, f.p, f.loop, zero);
    const auto before = rejected.size();
    auto failedImport = rejected.import(source, {source.land(p, r)});
    if (succeeded(failedImport) || rejected.size() != before || rejected.evaluation(existing) ||
        !rejected.constructionError().empty()) { return false; }
    // Individually acyclic registries can become cyclic when the same Input
    // has provenance only in the other arena. Reject the combined cycle.
    auto symbols = std::make_shared<Block>();
    auto x = symbols->addArgument(f.p.getType(), f.loop.getLoc());
    auto y = symbols->addArgument(f.p.getType(), f.loop.getLoc());
    fs::RegionExpressions left, right;
    left.retainInputOwner(symbols); right.retainInputOwner(symbols);
    auto xp = left.input(x);
    auto yo = left.select(xp, left.constant(0), left.constant(1));
    auto yp = left.evaluatedInput(y, f.r, f.loop, yo);
    auto yq = right.input(y);
    auto xo = right.select(yq, right.constant(0), right.constant(1));
    auto xq = right.evaluatedInput(x, f.p, f.loop, xo);
    const auto rightSize = right.size();
    return failed(right.import(left, {yp})) && right.size() == rightSize &&
        !right.evaluation(yq) && right.evaluation(xq) && right.constructionError().empty();
}
bool check(MLIRContext* context, unsigned trips)
{
    Fixture f(context, trips);
    if (failed(verify(*f.module))) { return false; }
    if (!provenance(f)) { return false; }
    const auto before = render(f.module->getOperation());
    fs::RegionExpressions e;
    fs::IterationPredicates predicates(f.loop, e, {});
    auto ordinal = fs::CountedLoop::get(f.loop)->ordinal(e);
    auto p = predicates.at(f.p, ordinal), r = predicates.at(f.r, ordinal);
    auto past = predicates.at(f.p, e.sub(ordinal, e.constant(1)));
    auto future = predicates.at(f.r, e.add(ordinal, e.constant(1)));
    const bool bounded = predicates.boundedLookback(e.land(p, future), ordinal, 1) == 1 &&
        !predicates.boundedLookback(future, ordinal, 0);
    if (!bounded) { return false; }
    auto thenRoot = e.select(p, e.boolean(true), future);
    auto elseRoot = e.select(p, past, e.boolean(false));
    Fixture foreign(context, trips);
    if (!attempt(predicates, e, e.boolean(true), foreign.thenCut, false)) { return false; }
    Preparation retained;
    const bool retainedReady = attempt(predicates, e, future, f.late, false, {}, &retained) &&
        attempt(predicates, e, r, f.late, true, {}, &retained) &&
        attempt(predicates, e, future, f.late, false, {}, &retained);
    if (!retainedReady) { return false; }
    for (unsigned retry = 0; retry < 2; ++retry) {
        const bool ready = attempt(predicates, e, thenRoot, f.thenCut, true, 1) &&
            attempt(predicates, e, elseRoot, f.elseCut, true, 0) &&
            attempt(predicates, e, p, f.thenCut, true, 1) &&
            attempt(predicates, e, p, f.elseCut, true, 0) &&
            attempt(predicates, e, p, f.early, false) &&
            attempt(predicates, e, r, f.thenCut, false) &&
            attempt(predicates, e, r, f.late, true) &&
            attempt(predicates, e, past, f.thenCut, false) &&
            attempt(predicates, e, future, f.late, false);
        if (!ready) { return false; }
    }
    return before == render(f.module->getOperation());
}
} // namespace
bool runIterationPredicateChecks(MLIRContext* context)
{
    context->loadDialect<func::FuncDialect, scf::SCFDialect, arith::ArithDialect, memref::MemRefDialect>();
    for (unsigned trips : {0U, 1U, 3U}) { if (!check(context, trips)) { return false; } }
    llvm::outs() << "iteration predicate checks passed\n";
    return true;
}
