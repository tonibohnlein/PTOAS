// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Exercise the common inserter without recognition or numerical-loop recipes.
#include "SyncLogicalInsertionChecks.h"
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/FrontierSynch/LogicalInsertion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
std::string render(func::FuncOp function)
{
    std::string text;
    llvm::raw_string_ostream out(text);
    function.print(out);
    return text;
}
bool rejectedUnchanged(func::FuncOp function, fs::PreparedLogicalPlan& plan)
{
    auto before = render(function);
    ScopedDiagnosticHandler silence(function.getContext(), [](Diagnostic&) { return success(); });
    return failed(fs::insertLogicalSynchronization(function, plan)) && render(function) == before;
}
} // namespace
LogicalResult runPreparedInsertionChecks(func::FuncOp function)
{
    SmallVector<Operation*> cuts;
    function.walk([&](Operation* op) {
        if (op->hasAttr("test.cut")) {
            cuts.push_back(op);
        }
    });
    if (cuts.size() != 3 || function.getNumArguments() != 3) {
        return function.emitError("prepared insertion fixture requires three cuts and arguments");
    }
    using Kind = fs::LogicalCommandKind;
    const auto mte2 = static_cast<uint32_t>(pto::PIPE::PIPE_MTE2);
    const auto vector = static_cast<uint32_t>(pto::PIPE::PIPE_V);
    const auto mte3 = static_cast<uint32_t>(pto::PIPE::PIPE_MTE3);
    auto guard = function.getArgument(0), other = function.getArgument(1), ordinal = function.getArgument(2);
    // Exercise both an unavailable value and a detached preparation dependency
    // whose definition would be inserted after its use. Neither may mutate IR.
    fs::PreparedLogicalPlan bad(7);
    bad.endpoints.push_back({cuts[0], Kind::Set, mte2, vector, 0, guard, cuts[0]->getResult(0)});
    if (!rejectedUnchanged(function, bad)) {
        return function.emitError("unavailable identity was accepted or rejection changed IR");
    }
    bad.endpoints = {{cuts[0], Kind::Set, mte2, vector, 0, guard, ordinal, {Value()}}};
    if (!rejectedUnchanged(function, bad)) {
        return function.emitError("null family member was accepted or rejection changed IR");
    }
    bad.endpoints = {{cuts[0], Kind::Set, mte2, vector, 0, guard, ordinal},
                     {cuts[1], Kind::Wait, mte2, vector, 0, guard, ordinal}};
    fs::EndpointFamily malformed;
    malformed.members.push_back({0, 0, 1, {}, {}});
    bad.families.push_back(malformed);
    if (!rejectedUnchanged(function, bad)) {
        return function.emitError("invalid family cuts were accepted or rejection changed IR");
    }
    bad.families.clear();
    bad.endpoints.clear();
    OpBuilder builder(function.getContext());
    builder.setInsertionPointToEnd(&bad.addPreparation(cuts[2]));
    auto late = builder.create<arith::ConstantIndexOp>(function.getLoc(), 1);
    builder.setInsertionPointToEnd(&bad.addPreparation(cuts[0]));
    builder.create<arith::AddIOp>(function.getLoc(), late, ordinal);
    if (!rejectedUnchanged(function, bad)) {
        return function.emitError("non-dominating preparation was accepted or rejection changed IR");
    }
    builder.setInsertionPoint(function.getBody().front().getTerminator());
    auto isolated = builder.create<func::FuncOp>(function.getLoc(), "isolated_test", builder.getFunctionType({}, {}));
    builder.setInsertionPointToStart(isolated.addEntryBlock());
    auto terminator = builder.create<func::ReturnOp>(function.getLoc());
    fs::PreparedLogicalPlan capture(7);
    capture.endpoints.push_back({terminator, Kind::Set, mte2, vector, 0, guard, ordinal});
    const bool rejected = rejectedUnchanged(function, capture);
    isolated.erase();
    if (!rejected) {
        return function.emitError("capture across isolation was accepted or rejection changed IR");
    }
    fs::PreparedLogicalPlan plan(7);
    builder.setInsertionPointToEnd(&plan.addPreparation(cuts[0]));
    auto zero = builder.create<arith::ConstantIndexOp>(function.getLoc(), 0);
    auto identity = builder.create<arith::AddIOp>(function.getLoc(), ordinal, zero);
    auto sharedGuard = builder.create<arith::AndIOp>(function.getLoc(), guard, guard);
    builder.setInsertionPointToEnd(&plan.addPreparation(cuts[0]));
    auto duplicateZero = builder.create<arith::ConstantIndexOp>(function.getLoc(), 0);
    auto duplicateIdentity = builder.create<arith::AddIOp>(function.getLoc(), ordinal, duplicateZero);
    auto duplicateGuard = builder.create<arith::AndIOp>(function.getLoc(), guard, guard);
    builder.setInsertionPointToEnd(&plan.addPreparation(cuts[1]));
    auto laterUse = builder.create<arith::AddIOp>(function.getLoc(), duplicateIdentity, duplicateZero);
    // Deliberately unordered input; the inserter must group by actual cut and
    // emit SET, one coalesced local barrier, WAIT regardless of record order.
    plan.endpoints = {
        {cuts[1], Kind::Wait, mte2, vector, 0, duplicateGuard, duplicateIdentity, {duplicateZero}},
        {cuts[2], Kind::Wait, vector, mte3, 1, other, identity},
        {cuts[1], Kind::Barrier, vector, vector, 2, guard, {}},
        {cuts[0], Kind::Set, mte2, vector, 0, duplicateGuard, duplicateIdentity, {duplicateZero}},
        {cuts[1], Kind::Set, vector, mte3, 1, other, identity},
        {cuts[1], Kind::Barrier, vector, vector, 3, other, {}}};
    if (failed(fs::insertLogicalSynchronization(function, plan)) || failed(verify(function))) {
        return failure();
    }
    if (plan.endpoints[0].guard != sharedGuard || plan.endpoints[0].identity != identity ||
        plan.endpoints[0].memberCoordinates.front() != zero ||
        laterUse.getLhs() != identity || laterUse.getRhs() != zero) {
        return function.emitError("same-cut preparation sharing lost a raw endpoint or later SSA use");
    }
    fs::PreparedLogicalPlan duplicate(7);
    if (!rejectedUnchanged(function, duplicate)) {
        return function.emitError("duplicate namespace was accepted or rejection changed IR");
    }
    fs::PreparedLogicalPlan independent(8);
    auto* last = function.getBody().front().getTerminator();
    independent.endpoints = {{last, Kind::Set, mte2, vector, 0, guard, ordinal},
                            {last, Kind::Wait, mte2, vector, 0, guard, ordinal}};
    if (failed(fs::insertLogicalSynchronization(function, independent)) || failed(verify(function))) {
        return failure();
    }
    llvm::outs() << "prepared-insertion: unavailable operands and namespace collision rejected without mutation\n";
    return success();
}
