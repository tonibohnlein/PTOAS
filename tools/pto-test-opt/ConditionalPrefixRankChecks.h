// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_CONDITIONAL_PREFIX_RANK_CHECKS_H
#define PTO_CONDITIONAL_PREFIX_RANK_CHECKS_H
// Included in the independent prefix interpreter: callbacks are qualified even
// when their runtime activity is false; only emitted rank evaluation is guarded.
bool conditionalPrefixRanks(MLIRContext& context)
{
    auto fail = [](llvm::StringRef stage) {
        llvm::errs() << "conditional prefix ranks failed: " << stage << "\n";
        return false;
    };
    OpBuilder builder(&context);
    const auto loc = builder.getUnknownLoc();
    OwningOpRef<ModuleOp> module(ModuleOp::create(loc));
    builder.setInsertionPointToEnd(module->getBody());
    auto payload = builder.create<func::FuncOp>(loc, "payload",
        builder.getFunctionType({builder.getIndexType()}, {}));
    payload.setPrivate();
    auto function = builder.create<func::FuncOp>(loc, "conditional_rank", builder.getFunctionType(
        {builder.getIndexType(), builder.getIndexType(), builder.getI1Type(), builder.getIndexType()}, {}));
    builder.setInsertionPointToStart(function.addEntryBlock());
    auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    auto one = builder.create<arith::ConstantIndexOp>(loc, 1);
    auto two = builder.create<arith::ConstantIndexOp>(loc, 2);
    auto doubled = builder.create<arith::AddIOp>(loc, function.getArgument(0), function.getArgument(0));
    auto upper = builder.create<arith::AddIOp>(loc, doubled, one);
    auto outer = builder.create<scf::ForOp>(loc, zero, function.getArgument(1), one);
    builder.setInsertionPointToStart(outer.getBody());
    auto visits = builder.create<scf::ForOp>(loc, one, upper, two);
    builder.setInsertionPointToStart(visits.getBody());
    auto source = builder.create<func::CallOp>(loc, "payload", TypeRange{}, ValueRange{zero});
    builder.setInsertionPointAfter(visits);
    auto target = builder.create<func::CallOp>(loc, "payload", TypeRange{}, ValueRange{two});
    builder.setInsertionPointAfter(outer);
    builder.create<func::ReturnOp>(loc);
    if (failed(verify(*module))) { return fail("input verification"); }
    const auto original = render(module->getOperation());
    pto::CompoundInstanceElement phase(0, SmallVector<const pto::BaseMemInfo*>{},
        SmallVector<const pto::BaseMemInfo*>{}, static_cast<pto::PipelineType>(P), source->getName());
    phase.elementOp = source;
    fs::InvocationPrefixCollection collection(function);
    auto& recipe = collection.recipe();
    recipe.privateScalarControl = true; recipe.planId = 19; recipe.sourceBounds = {{P, 6}};
    recipe.slots.push_back({builder.getIndexType(), builder.getIndexAttr(0)});
    std::string error;
    if (failed(collection.source(&phase,
        [](OpBuilder&, const IRMapping&, llvm::MutableArrayRef<Value> state, Value rank, std::string&) {
            state[0] = rank;
            return success();
        }, error))) { return fail("publication registration"); }
    auto arena = std::make_shared<fs::RegionExpressions>();
    const auto trips = arena->input(function.getArgument(0));
    const auto safe = arena->lt(trips, arena->constant(100));
    const auto active = arena->land(arena->input(function.getArgument(2)), safe);
    auto activity = fs::preparePrefixExpression(arena, active, target, collection, "rank-activity");
    auto selection = fs::preparePrefixExpression(arena, safe, target, collection, "rank-selection");
    struct Control { unsigned mode = 0, calls = 0; };
    auto control = std::make_shared<Control>();
    const Value originalTrips = function.getArgument(0), denominator = function.getArgument(3);
    auto rank = [selection, control, loc, originalTrips, denominator](OpBuilder& b, const IRMapping& mapping,
        llvm::ArrayRef<Value> state, std::string& diagnostic) -> FailureOr<Value> {
        ++control->calls;
        auto selected = selection(b, mapping, state, diagnostic);
        if (failed(selected)) { return failure(); }
        // Dynamic division is valid owned scalar callback work, while the
        // expression arena deliberately admits only constant positive divisors.
        auto trips = mapping.lookup(originalTrips);
        auto quotient = b.create<arith::DivUIOp>(loc, trips, mapping.lookup(denominator));
        quotient->setAttr("test.rank_work", b.getUnitAttr());
        auto equal = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, quotient, trips);
        auto predicate = b.create<arith::AndIOp>(loc, *selected, equal);
        if (control->mode == 1) { diagnostic = "unavailable rank control"; return failure(); }
        if (control->mode == 2) { return *selected; } // Wrong type must still refuse.
        auto absent = b.create<arith::ConstantIndexOp>(loc, 0);
        return b.create<arith::SelectOp>(loc, predicate, state[0], absent).getResult();
    };
    // The second activity callback runs outside the first rank branch. Both
    // rank callbacks share an expression session but must own separate blocks.
    for (unsigned duplicate = 0; duplicate < 2; ++duplicate) {
        if (failed(fs::appendPrefixIncoming(collection, target, {P, Q}, activity, rank, error))) {
            return fail("incoming registration");
        }
    }
    fs::PrivateStateError diagnostic;
    auto prepared = fs::materializeInvocationPrefix(function, recipe, diagnostic);
    if (failed(prepared) || control->calls != 2 || failed(verify(*(*prepared)->module))) {
        llvm::errs() << "rank materialization: " << diagnostic.message << "; callbacks=" << control->calls << "\n";
        return fail("initial materialization");
    }
    const auto emitted = render((*prepared)->function);
    unsigned staticDivisions = 0;
    (*prepared)->function.walk([&](arith::DivUIOp op) {
        if (!op->hasAttr("test.rank_work")) { return; }
        ++staticDivisions;
        if (!op->getParentOfType<scf::IfOp>()) { staticDivisions += 100; }
    });
    if (staticDivisions != 2) {
        llvm::errs() << "rank guarded division count=" << staticDivisions << "\n";
        return fail("static guarded division ownership");
    }
    for (unsigned outerTrips : {0U, 1U, 2U}) {
        for (unsigned sourceTrips : {0U, 1U, 3U}) {
            for (bool enabled : {false, true}) {
                Execution execution;
                auto output = (*prepared)->function;
                execution.values[output.getArgument(0)] = sourceTrips;
                execution.values[output.getArgument(1)] = outerTrips;
                execution.values[output.getArgument(2)] = enabled;
                execution.values[output.getArgument(3)] = enabled ? 1 : 0;
                execution.block(output.front());
                std::vector<unsigned> roles;
                for (unsigned visit = 0; visit < outerTrips; ++visit) {
                    roles.insert(roles.end(), sourceTrips, 0); roles.push_back(2);
                }
                auto expected = native(roles);
                if (enabled && sourceTrips) {
                    for (unsigned visit = 0; visit < outerTrips; ++visit) {
                        const auto targetIndex = (visit + 1) * (sourceTrips + 1) - 1;
                        edge(expected, 2 * (targetIndex - 1) + 1, 2 * targetIndex);
                    }
                }
                close(expected);
                if (!execution.valid || execution.rankDivisions != (enabled ? 2 * outerTrips : 0) ||
                    !verifyTrace(execution.events, roles, &expected, outerTrips * sourceTrips)) {
                    llvm::errs() << "rank execution: outer=" << outerTrips << ", sources=" << sourceTrips
                        << ", enabled=" << enabled << ", valid=" << execution.valid
                        << ", divisions=" << execution.rankDivisions << "\n";
                    return fail("independent execution and command graph");
                }
            }
        }
    }
    // A dynamic runtime-false argument does not excuse an unavailable compile
    // callback. Failed attempts must discard branch SSA and permit an exact retry.
    for (unsigned mode : {1U, 2U}) {
        control->mode = mode; control->calls = 0;
        auto refused = fs::materializeInvocationPrefix(function, recipe, diagnostic);
        if (succeeded(refused) || control->calls != 1 || original != render(module->getOperation())) {
            return fail("callback refusal atomicity");
        }
        control->mode = 0; control->calls = 0;
        auto retry = fs::materializeInvocationPrefix(function, recipe, diagnostic);
        if (failed(retry) || control->calls != 2 || render((*retry)->function) != emitted ||
            failed(verify(*(*retry)->module))) { return fail("callback retry equivalence"); }
    }
    // Preserve the original constant paths, including the structural evidence
    // seen by the separate full-prefix lifetime qualifier.
    for (bool enabled : {false, true}) {
        fs::InvocationPrefixCollection fixed(function);
        auto& specification = fixed.recipe();
        specification.privateScalarControl = true; specification.planId = 19;
        specification.sourceBounds = recipe.sourceBounds;
        specification.slots = recipe.slots; specification.sources = recipe.sources;
        unsigned calls = 0;
        auto activity = [enabled](OpBuilder& b, const IRMapping&, llvm::ArrayRef<Value>, std::string&)
            -> FailureOr<Value> { return b.create<arith::ConstantIntOp>(b.getUnknownLoc(), enabled, 1).getResult(); };
        auto selected = [&calls](OpBuilder&, const IRMapping&, llvm::ArrayRef<Value> state, std::string&)
            -> FailureOr<Value> { ++calls; return state[0]; };
        if (failed(fs::appendPrefixIncoming(fixed, target, {P, Q}, activity, selected, error))) {
            return fail("constant incoming registration");
        }
        auto constant = fs::materializeInvocationPrefix(function, specification, diagnostic);
        if (failed(constant) || calls != unsigned(enabled)) { return fail("constant callback qualification"); }
        bool changedEvidence = false;
        (*constant)->function.walk([&](scf::ForOp loop) {
            changedEvidence |= loop->hasAttr("pto.prefix_full_acquisition") &&
                !loop->hasAttr("pto.prefix_terminal_acquisition");
        });
        if (changedEvidence) { return fail("constant structural evidence"); }
    }
    llvm::outs() << "prefix rank guards: inactive undefined work skipped, exact ranks, callback/cache isolation\n";
    return original == render(module->getOperation()) || fail("original IR preservation");
}
#endif
