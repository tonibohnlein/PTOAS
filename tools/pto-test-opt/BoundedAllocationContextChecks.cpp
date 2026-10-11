// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent legal-ordinal evaluation checks contextual inactivity and cache scope.
#include "../../lib/PTO/Transforms/FrontierSynch/Structured/BoundedLifetimeAllocationContext.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Structured/BoundedLifetimeInactiveRecords.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/IterationPredicates.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/CountedLoop.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
namespace fs = pto::frontiersynch;
namespace {
struct Fixture {
    explicit Fixture(MLIRContext& context) : module(ModuleOp::create(UnknownLoc::get(&context))) {}
    OwningOpRef<ModuleOp> module;
    std::shared_ptr<pto::SyncInput> input = std::make_shared<pto::SyncInput>();
    std::shared_ptr<fs::BoundedLifetimeDemandResult> owner;
    Value parameter, opaque, lower;
    fs::RegionExpressions::Id first, future, outside;
};
bool makeLoop(Fixture& f, int64_t lower, int64_t step, unsigned trips)
{
    OpBuilder b(f.module->getContext());
    b.setInsertionPointToEnd(f.module->getBody());
    auto function = b.create<func::FuncOp>(b.getUnknownLoc(), "context",
        b.getFunctionType({b.getIndexType(), b.getI1Type()}, {}));
    auto* entry = function.addEntryBlock();
    b.setInsertionPointToStart(entry);
    f.parameter = entry->getArgument(0); f.opaque = entry->getArgument(1);
    f.lower = b.create<arith::ConstantIndexOp>(b.getUnknownLoc(), lower);
    auto upper = b.create<arith::ConstantIndexOp>(b.getUnknownLoc(), lower + step * trips);
    auto stride = b.create<arith::ConstantIndexOp>(b.getUnknownLoc(), step);
    auto loop = b.create<scf::ForOp>(b.getUnknownLoc(), f.lower, upper, stride);
    b.setInsertionPointToStart(loop.getBody());
    auto first = b.create<arith::CmpIOp>(b.getUnknownLoc(), arith::CmpIPredicate::eq,
        loop.getInductionVar(), f.lower);
    auto outside = b.create<arith::CmpIOp>(b.getUnknownLoc(), arith::CmpIPredicate::eq,
        loop.getInductionVar(), upper);
    b.setInsertionPointAfter(loop); b.create<func::ReturnOp>(b.getUnknownLoc());
    if (failed(f.input->build(function))) { return false; }
    f.owner = std::make_shared<fs::BoundedLifetimeDemandResult>(std::make_shared<fs::RegionExpressions>());
    f.owner->function = function; f.owner->loop = loop; f.owner->input = f.input.get();
    f.owner->anchors.resize(2); f.owner->unconditional = {1, 1};
    auto& e = f.owner->expressions;
    f.owner->base = e.input(f.owner->symbols.addArgument(b.getIndexType(), b.getUnknownLoc()));
    f.owner->predicates = std::make_unique<fs::IterationPredicates>(loop, e,
        llvm::ArrayRef<const pto::CompoundInstanceElement*>{});
    f.first = f.owner->predicates->at(first, f.owner->base);
    f.future = f.owner->predicates->at(first, e.add(f.owner->base, e.constant(1)));
    f.outside = f.owner->predicates->at(outside, f.owner->base);
    f.owner->predicateBindings = {{f.first, {first, 0}}, {f.future, {first, 1}}, {f.outside, {outside, 0}}};
    return true;
}
void makeWindow(Fixture& f)
{
    auto& e = f.owner->expressions;
    const auto base = f.owner->base;
    auto trips = fs::CountedLoop::get(f.owner->loop)->trips(e);
    auto& w = f.owner->window; w.sites = 2; w.span = 1;
    for (unsigned d = 0; d <= 1; ++d) {
        auto active = e.land(e.lt(base, trips), e.lt(e.constant(d), e.sub(trips, base)));
        w.payloads.push_back({0, active}); w.payloads.push_back({1, active});
    }
    auto rank = e.add(e.select(f.first, e.constant(1), e.constant(0)),
        e.select(e.lnot(f.first), e.constant(1), e.constant(0)));
    for (auto guard : {f.outside, f.future, e.eq(rank, e.constant(2)), f.first, e.lnot(f.first),
        e.input(f.opaque), e.eq(e.input(f.parameter), e.constant(0)), e.land(e.input(f.opaque), f.future)}) {
        f.owner->analysis.sourceDemands.push_back({0, 1, guard});
    }
}
bool checkExpected(llvm::ArrayRef<uint8_t> proofs, unsigned trips)
{
    std::vector<uint8_t> expected{1, 1, 1, uint8_t(trips == 0), uint8_t(trips < 2),
        uint8_t(trips == 0), uint8_t(trips == 0), 1};
    return proofs == llvm::ArrayRef<uint8_t>(expected);
}
bool concrete(Fixture& f, llvm::ArrayRef<uint8_t> proofs, unsigned trips)
{
    auto& e = f.owner->expressions;
    for (unsigned ordinal = 0; ordinal < trips; ++ordinal) {
        for (int64_t parameter : {-1, 0, 1}) {
            for (bool opaque : {false, true}) {
                fs::RegionExpressions::Substitution values({{f.owner->base, e.constant(ordinal)},
                    {e.input(f.parameter), e.constant(parameter)}, {e.input(f.opaque), e.boolean(opaque)}});
                for (auto [record, edge] : llvm::enumerate(f.owner->analysis.sourceDemands)) {
                    auto value = e.constantValue(e.substitute(edge.guard, values));
                    if (!value || (proofs[record] && *value)) { return false; }
                }
            }
        }
    }
    return true;
}
bool mutations(Fixture& f, const std::shared_ptr<fs::detail::BoundedLifetimeAllocationContext>& proof)
{
    auto& e = f.owner->expressions;
    auto query = [&]() { return proof->inactive(e, f.owner->window, f.owner->unconditional,
                                               f.owner->analysis.sourceDemands); };
    auto present = f.owner->window.payloads.front().present;
    f.owner->window.payloads.front().present = e.boolean(true);
    const bool windowRejected = !query();
    f.owner->window.payloads.front().present = present;
    auto binding = f.owner->predicateBindings.front();
    f.owner->predicateBindings.front().first = e.boolean(true);
    const bool bindingRejected = !query();
    f.owner->predicateBindings.front() = binding;
    auto old = f.lower.getDefiningOp()->getAttr("value");
    f.lower.getDefiningOp()->setAttr("value", IntegerAttr::get(IndexType::get(f.module->getContext()), 123));
    const bool boundRejected = !query();
    f.lower.getDefiningOp()->setAttr("value", old);
    auto guard = f.owner->analysis.sourceDemands.front().guard;
    f.owner->analysis.sourceDemands.front().guard = e.boolean(true);
    const bool recordRejected = !query();
    f.owner->analysis.sourceDemands.front().guard = guard;
    return windowRejected && bindingRejected && boundRejected && recordRejected && query().has_value();
}
bool scope(Fixture& f, const std::shared_ptr<fs::detail::BoundedLifetimeAllocationContext>& proof)
{
    auto& e = f.owner->expressions;
    fs::RegionExpressions foreign;
    if (proof->inactive(foreign, f.owner->window, f.owner->unconditional,
            f.owner->analysis.sourceDemands)) { return false; }
    auto records = f.owner->analysis.sourceDemands;
    auto window = f.owner->window;
    auto flags = f.owner->unconditional;
    if (proof->inactive(e, f.owner->window, f.owner->unconditional, records) ||
        proof->inactive(e, window, f.owner->unconditional, f.owner->analysis.sourceDemands) ||
        proof->inactive(e, f.owner->window, flags, f.owner->analysis.sourceDemands)) { return false; }
    auto foreignInput = std::make_shared<pto::SyncInput>();
    return mutations(f, proof) && !fs::detail::BoundedLifetimeAllocationContext::create(f.owner, foreignInput) &&
        proof->proofBuilds() == 1;
}
bool startup(Fixture& f, unsigned trips)
{
    auto& e = f.owner->expressions;
    auto count = fs::CountedLoop::get(f.owner->loop)->trips(e);
    // WAIT at consumer-distance uses the same source ordinal as SET. Startup
    // suppresses negative source ordinals; unsigned legal presence excludes them.
    auto legal = e.lt(f.owner->base, count);
    for (unsigned consumer = 0; consumer <= trips; ++consumer) {
        for (unsigned distance = 0; distance <= 1; ++distance) {
            int64_t source = int64_t(consumer) - distance;
            fs::RegionExpressions::Substitution values({{f.owner->base, e.constant(source)}});
            auto present = e.constantValue(e.substitute(legal, values));
            if (!present || bool(*present) != (source >= 0 && source < trips)) { return false; }
            if (consumer >= distance && consumer < trips && !*present) { return false; }
        }
    }
    return true;
}
bool check(MLIRContext& context, int64_t lower, int64_t step, unsigned trips)
{
    Fixture f(context);
    if (!makeLoop(f, lower, step, trips)) { return false; }
    makeWindow(f);
    auto& e = f.owner->expressions;
    fs::detail::BoundedInactiveRecordProof raw(e, f.owner->window, f.owner->unconditional);
    if (raw.proves(f.outside)) { return false; }
    auto proof = fs::detail::BoundedLifetimeAllocationContext::create(f.owner, f.input);
    if (!proof) { return false; }
    auto size = e.size();
    auto flags = proof->inactive(e, f.owner->window, f.owner->unconditional, f.owner->analysis.sourceDemands);
    if (!flags || e.size() != size || !e.constructionError().empty() || !checkExpected(*flags, trips)) {
        return false;
    }
    return concrete(f, *flags, trips) && startup(f, trips) && scope(f, proof);
}
} // namespace
bool runBoundedAllocationContextChecks()
{
    MLIRContext context;
    context.loadDialect<func::FuncDialect, scf::SCFDialect, arith::ArithDialect>();
    for (int64_t lower : {-11, 0, 11}) {
        for (int64_t step : {1, 7}) {
            for (unsigned trips = 0; trips <= 5; ++trips) {
                if (!check(context, lower, step, trips)) {
                    llvm::errs() << "bounded invocation context failed lower=" << lower << " step=" << step
                                 << " trips=" << trips << '\n';
                    return false;
                }
            }
        }
    }
    return true;
}
