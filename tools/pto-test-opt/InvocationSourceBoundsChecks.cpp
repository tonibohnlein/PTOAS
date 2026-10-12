// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Check exact small counts and conservative signed-range bounds without enumerating large loops.
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/InvocationSourceBounds.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
namespace fs = pto::frontiersynch;
namespace {
struct Fixture {
    OwningOpRef<ModuleOp> module;
    func::FuncOp function;
    OpBuilder builder;
    std::vector<fs::PrefixSource> sources;
    explicit Fixture(MLIRContext& context)
        : module(ModuleOp::create(UnknownLoc::get(&context))), builder(&context) {
        builder.setInsertionPointToEnd(module->getBody());
        auto type = builder.getFunctionType({builder.getI32Type(), builder.getI32Type(),
                                            builder.getI1Type(), builder.getIndexType()}, {});
        function = builder.create<func::FuncOp>(builder.getUnknownLoc(), "counts", type);
        builder.setInsertionPointToStart(function.addEntryBlock());
    }
    Value constant(int64_t n) { return builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), n); }
    Operation* source(unsigned pipe) {
        auto value = constant(123); auto* operation = value.getDefiningOp();
        sources.push_back({operation, pipe, {}}); return operation;
    }
    void counted(int64_t lower, int64_t upper, int64_t step, unsigned pipe) {
        auto l = constant(lower), u = constant(upper), s = constant(step);
        auto loop = builder.create<scf::ForOp>(builder.getUnknownLoc(), l, u, s);
        OpBuilder::InsertionGuard restore(builder); builder.setInsertionPointToStart(loop.getBody());
        source(pipe);
    }
    void finish() { builder.create<func::ReturnOp>(builder.getUnknownLoc()); }
};
bool check(Fixture& f, const std::map<uint32_t,uint64_t>& expected) {
    if (failed(verify(*f.module))) { return false; }
    std::string error;
    auto result = fs::inferInvocationSourceBounds(f.function, f.sources, error);
    const bool valid = succeeded(result) && *result == expected && error.empty();
    if (!valid) {
        llvm::errs() << "source-bound mismatch: " << error << "\n";
    }
    return valid;
}
bool constants(MLIRContext& context) {
    for (int64_t lower : {-5, 0, 7}) {
        for (int64_t upper : {-5, 0, 7}) {
            for (int64_t step : {1, 2, 5}) {
                Fixture f(context); f.counted(lower, upper, step, 0); f.finish();
                uint64_t count = 0;
                for (int64_t i = lower; i < upper; i += step) { ++count; }
                if (!check(f, {{0,count}})) { return false; }
            }
        }
    }
    Fixture empty(context);
    auto zero = empty.constant(0), one = empty.constant(1);
    auto outer = empty.builder.create<scf::ForOp>(empty.builder.getUnknownLoc(), zero, zero, one);
    {
        OpBuilder::InsertionGuard restore(empty.builder);
        empty.builder.setInsertionPointToStart(outer.getBody());
        auto unknown = empty.builder.create<scf::ForOp>(empty.builder.getUnknownLoc(), zero,
                                                       empty.function.getArgument(3), one);
        empty.builder.setInsertionPointToStart(unknown.getBody()); empty.source(0);
    }
    empty.finish(); return check(empty, {{0,0}});
}
bool structure(MLIRContext& context) {
    Fixture f(context); auto zero = f.constant(0), two = f.constant(2), one = f.constant(1);
    auto outer = f.builder.create<scf::ForOp>(f.builder.getUnknownLoc(), zero, two, one);
    {
        OpBuilder::InsertionGuard restore(f.builder); f.builder.setInsertionPointToStart(outer.getBody());
        f.counted(0,3,1,0); // six original actions on pipe0.
        auto* action = f.source(1); f.sources.push_back({action,2,{}}); // one operation, two phases.
    }
    auto branch = f.builder.create<scf::IfOp>(f.builder.getUnknownLoc(), f.function.getArgument(2), true);
    {
        OpBuilder::InsertionGuard restore(f.builder); f.builder.setInsertionPointToStart(branch.thenBlock());
        f.counted(0,2,1,0); f.source(1);
        f.builder.setInsertionPointToStart(branch.elseBlock()); f.counted(0,3,1,0); f.source(2);
    }
    f.counted(0,2,1,0); f.finish();
    // Independent trace arithmetic: 2*3 + max(2,3) + 2;
    // pipes1/2 each have two outer occurrences and one possible branch occurrence.
    return check(f, {{0,11},{1,3},{2,3}});
}
bool dynamicRanges(MLIRContext& context) {
    for (bool difference : {false,true}) {
        Fixture f(context); auto& b = f.builder; auto loc = b.getUnknownLoc();
        auto a = b.create<arith::ExtSIOp>(loc,b.getI64Type(),f.function.getArgument(0));
        auto z = b.create<arith::ExtSIOp>(loc,b.getI64Type(),f.function.getArgument(1));
        Value lower, upper;
        if (difference) {
            lower = f.constant(0);
            auto delta = b.create<arith::SubIOp>(loc,z,a);
            upper = b.create<arith::IndexCastOp>(loc,b.getIndexType(),delta);
        } else {
            lower = b.create<arith::IndexCastOp>(loc,b.getIndexType(),a);
            upper = b.create<arith::IndexCastOp>(loc,b.getIndexType(),z);
        }
        auto one = f.constant(1);
        auto loop = b.create<scf::ForOp>(loc,lower,upper,one);
        { OpBuilder::InsertionGuard restore(b); b.setInsertionPointToStart(loop.getBody()); f.source(0); }
        f.finish();
        if (!check(f, {{0,UINT64_C(4294967295)}})) { return false; }
    }
    return true;
}
bool boundaries(MLIRContext& context) {
    Fixture edge(context); edge.counted(0,INT64_MAX-1,1,0); edge.finish();
    if (!check(edge, {{0,uint64_t(INT64_MAX)-1}})) { return false; }
    Fixture wide(context); wide.counted(INT64_MIN,INT64_MAX,4,0); wide.finish();
    if (!check(wide, {{0,UINT64_C(4611686018427387904)}})) { return false; }
    Fixture tooLarge(context); tooLarge.counted(0,INT64_MAX,1,0); tooLarge.finish();
    std::string error;
    const bool invalid = succeeded(fs::inferInvocationSourceBounds(tooLarge.function,tooLarge.sources,error)) ||
        error.empty();
    if (invalid) {
        return false;
    }
    Fixture product(context);
    auto start = product.constant(0), end = product.constant(INT64_MAX-1), step = product.constant(1);
    auto loop = product.builder.create<scf::ForOp>(product.builder.getUnknownLoc(),start,end,step);
    {
        OpBuilder::InsertionGuard restore(product.builder);
        product.builder.setInsertionPointToStart(loop.getBody()); product.counted(0,2,1,0);
    }
    product.finish();
    if (succeeded(fs::inferInvocationSourceBounds(product.function,product.sources,error))) { return false; }
    auto duplicate = edge.sources; duplicate.push_back(duplicate.front());
    if (succeeded(fs::inferInvocationSourceBounds(edge.function,duplicate,error))) { return false; }
    if (succeeded(fs::inferInvocationSourceBounds(edge.function,wide.sources,error))) { return false; }
    Fixture noSources(context); auto zero = noSources.constant(0), one = noSources.constant(1);
    noSources.builder.create<scf::ForOp>(noSources.builder.getUnknownLoc(),zero,
                                       noSources.function.getArgument(3),one);
    noSources.finish(); return check(noSources, {});
}
} // namespace
bool runInvocationSourceBoundsChecks() {
    MLIRContext context; context.disableMultithreading();
    context.loadDialect<func::FuncDialect,scf::SCFDialect,arith::ArithDialect>();
    const bool passed = constants(context) && structure(context) && dynamicRanges(context) && boundaries(context);
    if (!passed) { return false; }
    llvm::outs() << "invocation source bounds checks passed\n";
    return true;
}
