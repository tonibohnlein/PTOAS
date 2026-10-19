// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Uniform inactive recipes preserve original records and detached ownership.
#include "../../lib/PTO/Transforms/FrontierSynch/Structured/BoundedLifetimeInactiveRecords.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/IterationPredicates.h"
#include "PTO/Transforms/FrontierSynch/Structured/BoundedLifetimeInsertion.h"
#include "mlir/IR/OwningOpRef.h"
using namespace mlir;
namespace fs = pto::frontiersynch;
namespace {
bool proofChecks(MLIRContext& context)
{
    Block symbols; auto loc = UnknownLoc::get(&context);
    auto n = symbols.addArgument(IndexType::get(&context), loc);
    auto p = symbols.addArgument(IntegerType::get(&context, 1), loc);
    auto q = symbols.addArgument(IntegerType::get(&context, 1), loc);
    fs::RegionExpressions e;
    const auto p0 = e.lt(e.constant(0), e.input(n));
    const auto p1 = e.land(p0, e.lt(e.constant(1), e.input(n)));
    auto rank = [&](auto x) { return e.select(x, e.constant(1), e.constant(0)); };
    const auto dead = e.select(e.lt(rank(p0), rank(p1)), e.input(p), e.boolean(false));
    fs::LifetimeWindowInput w; w.sites = 2; w.span = 1;
    w.payloads = {{0, p0}, {1, p0}, {0, p1}, {1, p1}};
    std::vector<fs::GuardedRankEdge> records{{0, 1, e.input(p)}, {0, 2, dead}, {0, 1, e.input(q)}};
    const auto size = e.size();
    auto flags = fs::detail::boundedInactiveEmissionRecipes(e, w, {1, 1}, records);
    const bool correct = flags && *flags == std::vector<uint8_t>({0, 1, 0}) && e.size() == size;
    if (!correct) { return false; }
    {
        fs::RegionExpressions fresh;
        const auto first = fresh.input(p);
        const auto second = fresh.land(first, fresh.input(q));
        auto schema = w;
        schema.payloads = {{0, first}, {1, first}, {0, second}, {1, second}};
        const auto originalSize = fresh.size();
        const fs::GuardedRankEdge invalid{0, 1, static_cast<fs::RegionExpressions::Id>(originalSize)};
        if (fs::detail::boundedInactiveEmissionRecipes(fresh, schema, {1, 1}, {invalid})) { return false; }
        fs::detail::BoundedInactiveEmissionContext proof(fresh, schema, {1, 1});
        // Pattern construction makes this ID Boolean, but it was never an
        // original demand predicate and must not acquire valid provenance.
        if (!proof.valid() || fresh.size() <= originalSize || !fresh.isBoolean(invalid.guard) ||
            proof.proves(invalid)) { return false; }
    }
    {
        fs::detail::BoundedInactiveEmissionContext proof(e, w, {1, 1});
        if (!proof.valid() || proof.schemaPayloadVisits() != w.payloads.size()) { return false; }
        for (unsigned repeat = 0; repeat < 64; ++repeat) {
            // Model expressions appended by endpoint emission between lazy
            // proofs. Neither their IDs nor their meaning may be rolled back.
            const auto emitted = e.add(e.input(n), e.constant(100 + repeat));
            const auto retainedSize = e.size();
            for (unsigned record = 0; record < records.size(); ++record) {
                auto flag = proof.proves(records[record]);
                if (!flag || *flag != bool((*flags)[record]) || e.size() != retainedSize) { return false; }
            }
            fs::RegionExpressions::Substitution bind({{e.input(n), e.constant(7)}});
            if (e.constantValue(e.substitute(emitted, bind)) != 107 + repeat) { return false; }
        }
        if (proof.schemaPayloadVisits() != w.payloads.size()) { return false; }
        auto malformed = records.front(); malformed.guard = e.input(n);
        if (proof.proves(malformed) || proof.proves(records[1]) != true) { return false; }
    }
    for (uint64_t trips : {uint64_t(0), uint64_t(1), uint64_t(2), uint64_t(3), UINT64_MAX - 1, UINT64_MAX}) {
        for (unsigned bits = 0; bits < 4; ++bits) {
            fs::RegionExpressions::Substitution values({{e.input(n), e.constant(trips)},
                {e.input(p), e.boolean(bits & 1)}, {e.input(q), e.boolean(bits & 2)}});
            const bool inactive = e.constantValue(e.substitute(dead, values)) == 0;
            if (!inactive) { return false; }
        }
    }
    auto broken = w; broken.span = UINT64_MAX;
    if (fs::detail::boundedInactiveEmissionRecipes(e, broken, {1, 1}, records)) { return false; }
    broken = w; broken.payloads[0].present = e.invalid;
    if (fs::detail::boundedInactiveEmissionRecipes(e, broken, {1, 1}, records)) { return false; }
    if (fs::detail::boundedInactiveEmissionRecipes(e, w, {1}, records)) { return false; }
    records[1].guard = e.input(n);
    if (fs::detail::boundedInactiveEmissionRecipes(e, w, {1, 1}, records)) { return false; }
    // Independent presence bits do not justify dropping their ordering guard.
    w.payloads[0].present = e.input(p); w.payloads[2].present = e.input(q);
    records[1].guard = e.lt(rank(e.input(p)), rank(e.input(q)));
    flags = fs::detail::boundedInactiveEmissionRecipes(e, w, {1, 1}, records);
    return flags && !(*flags)[1];
}
bool recordChecks(MLIRContext& context, bool crossPipe)
{
    OpBuilder b(&context); auto loc = b.getUnknownLoc();
    OwningOpRef<ModuleOp> module(ModuleOp::create(loc));
    b.setInsertionPointToEnd(module->getBody());
    b.create<func::FuncOp>(loc, "opaque", b.getFunctionType({}, {b.getI1Type()}));
    auto f = b.create<func::FuncOp>(loc, "records", b.getFunctionType({b.getI1Type()}, {}));
    b.setInsertionPointToStart(f.addEntryBlock());
    auto zero = b.create<arith::ConstantIndexOp>(loc, 0);
    auto three = b.create<arith::ConstantIndexOp>(loc, 3);
    auto one = b.create<arith::ConstantIndexOp>(loc, 1);
    auto loop = b.create<scf::ForOp>(loc, zero, three, one);
    b.setInsertionPointToStart(loop.getBody());
    auto source = b.create<arith::ConstantIndexOp>(loc, 12);
    auto target = b.create<arith::ConstantIndexOp>(loc, 13);
    auto opaque = b.create<func::CallOp>(loc, "opaque", TypeRange{b.getI1Type()}, ValueRange{}).getResult(0);
    b.setInsertionPointAfter(loop); b.create<func::ReturnOp>(loc);
    auto owner = std::make_shared<fs::BoundedLifetimeDemandResult>(std::make_shared<fs::RegionExpressions>());
    owner->function = f; owner->loop = loop;
    auto& e = owner->expressions;
    owner->base = e.input(owner->symbols.addArgument(b.getIndexType(), loc));
    owner->ordinal = fs::CountedLoop::get(loop)->ordinal(e);
    owner->predicates = std::make_unique<fs::IterationPredicates>(loop, e,
        llvm::ArrayRef<const pto::CompoundInstanceElement*>{});
    const auto p0 = e.lt(owner->base, e.constant(3));
    const auto p1 = e.land(p0, e.lt(e.constant(1), e.sub(e.constant(3), owner->base)));
    auto rank = [&](auto x) { return e.select(x, e.constant(1), e.constant(0)); };
    auto future = owner->predicates->at(opaque, e.add(owner->base, e.constant(1)));
    owner->predicateBindings.push_back({future, {opaque, 1}});
    auto dead = e.select(e.lt(rank(p0), rank(p1)), future, e.boolean(false));
    auto& w = owner->window; w.sites = 2; w.span = 1;
    w.payloads = {{0, p0}, {1, p0}, {0, p1}, {1, p1}};
    owner->unconditional = {1, 1};
    owner->analysis.sourceDemands = {{0, 1, p0}, {0, crossPipe ? 3U : 2U, dead}, {0, 1, p0}};
    auto wrapped = w;
    wrapped.payloads[2].present = e.lt(e.add(owner->base, e.constant(1)), e.constant(3));
    auto wrappingGuard = e.lt(rank(p0), rank(wrapped.payloads[2].present));
    std::vector<fs::GuardedRankEdge> wrapping{{0, 2, wrappingGuard}};
    auto wrappingFlags = fs::detail::boundedInactiveEmissionRecipes(e, wrapped, {1, 1}, wrapping);
    if (!wrappingFlags || (*wrappingFlags)[0]) { return false; }
    fs::RegionExpressions::Substitution maximum({{owner->base, e.constant(UINT64_MAX)}});
    const bool wraps = e.constantValue(e.substitute(wrappingGuard, maximum)) == 1;
    if (!wraps) { return false; }
    pto::CompoundInstanceElement a(0, {}, {}, pto::PipelineType::PIPE_MTE2, source->getName());
    pto::CompoundInstanceElement c(1, {}, {}, pto::PipelineType::PIPE_V, target->getName());
    a.elementOp = source; c.elementOp = target;
    owner->anchors = {{&a, {}, {source->getBlock(), source}, {source->getBlock(), source->getNextNode()}},
        {&c, {}, {target->getBlock(), target}, {target->getBlock(), target->getNextNode()}}};
    std::string error;
    for (unsigned retry = 0; retry < 2; ++retry) {
        auto plan = fs::prepareBoundedLifetimeLogicalResult(owner, error, false);
        const bool prepared = succeeded(plan) && (*plan)->families.size() == 3;
        if (!prepared) {
            llvm::errs() << "record preparation: " << error << " success=" << succeeded(plan)
                         << " families=" << (succeeded(plan) ? (*plan)->families.size() : 0) << "\n";
            return false;
        }
        for (unsigned record = 0; record < 3; ++record) {
            const bool originalRecord = (*plan)->families[record].members.front().record == record;
            if (!originalRecord) { return false; }
        }
        unsigned deadEndpoints = 0;
        for (auto& endpoint : (*plan)->endpoints) {
            if (endpoint.records != SmallVector<uint32_t>{1}) { continue; }
            auto constant = endpoint.guard.getDefiningOp<arith::ConstantIntOp>();
            const bool falseGuard = constant && constant.value() == 0;
            if (!falseGuard) { return false; }
            ++deadEndpoints;
            if (crossPipe) {
                const bool identity = endpoint.identity && endpoint.memberCoordinates.size() == 1;
                if (!identity) { return false; }
                auto member = endpoint.memberCoordinates.front().getDefiningOp<arith::ConstantIndexOp>();
                const bool originalMember = member && member.value() == 1;
                if (!originalMember) { return false; }
            }
        }
        const bool stable = deadEndpoints == (crossPipe ? 2U : 1U) &&
            owner->analysis.sourceDemands[1].guard == dead;
        if (!stable) { return false; }
    }
    return true;
}
} // namespace
bool runBoundedEmissionRecipeChecks()
{
    MLIRContext context; context.disableMultithreading();
    context.loadDialect<func::FuncDialect, scf::SCFDialect, arith::ArithDialect>();
    const bool proof = proofChecks(context);
    if (!proof) {
        llvm::errs() << "inactive proof check failed\n";
        return false;
    }
    const bool local = recordChecks(context, false);
    if (!local) {
        llvm::errs() << "same-pipe recipe check failed\n";
        return false;
    }
    const bool crossing = recordChecks(context, true);
    if (!crossing) {
        llvm::errs() << "cross-pipe recipe check failed\n";
        return false;
    }
    llvm::outs() << "bounded emission recipe checks passed\n";
    return true;
}
