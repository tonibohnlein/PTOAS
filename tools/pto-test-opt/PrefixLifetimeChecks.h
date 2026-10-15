// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_PREFIX_LIFETIME_CHECKS_H
#define PTO_PREFIX_LIFETIME_CHECKS_H
// Included in the independent prefix interpreter. Notification occupancy is
// retained until an ALL completion observes the matching WAIT, not merely its
// scalar issue. Thus textual WAIT-before-SET alone cannot authorize reuse.
bool prefixTokens(
    func::FuncOp logical, func::FuncOp physical, unsigned trips, unsigned bits,
    const std::map<std::pair<unsigned, unsigned>, unsigned>& widths, ArrayRef<int64_t> ids)
{
    Execution expected, actual;
    expected.run(logical, trips, bits);
    actual.run(physical, trips, bits);
    if (!expected.valid || !actual.valid || expected.events.size() != actual.events.size()) {
        return false;
    }
    using Key = std::tuple<unsigned, unsigned, int64_t>;
    std::map<Key, bool> occupied;
    // Repeated enclosing invocations share physical flags; only actual barriers
    // and matching waits retire occupancy in this conservative event oracle.
    for (unsigned invocation = 0; invocation < 2; ++invocation) {
        for (auto [wanted, emitted] : llvm::zip(expected.events, actual.events)) {
            if (wanted.kind != emitted.kind || wanted.role != emitted.role || wanted.source != emitted.source ||
                wanted.target != emitted.target) {
                return false;
            }
            if (wanted.kind == 3 && wanted.source == unsigned(pto::PIPE::PIPE_ALL)) {
                for (auto it = occupied.begin(); it != occupied.end();) {
                    if (it->second) {
                        it = occupied.erase(it);
                    } else {
                        ++it;
                    }
                }
            } else if (wanted.kind == 1 || wanted.kind == 2) {
                auto width = widths.at({wanted.source, wanted.target});
                if (!width || wanted.ordinal <= 0 || emitted.ordinal != ids[(wanted.ordinal - 1) % width]) {
                    return false;
                }
                Key key{emitted.source, emitted.target, emitted.ordinal};
                if (wanted.kind == 1) {
                    if (!occupied.emplace(key, false).second) {
                        return false;
                    }
                } else {
                    auto found = occupied.find(key);
                    if (found == occupied.end() || found->second) {
                        return false;
                    }
                    found->second = true;
                }
            }
        }
        if (!occupied.empty()) {
            return false;
        }
    }
    return true;
}
bool closePrefix(func::FuncOp function)
{
    OpBuilder b(function.front().getTerminator());
    b.create<pto::BarrierOp>(function.getLoc(), pto::PipeAttr::get(function.getContext(), pto::PIPE::PIPE_ALL));
    std::string error;
    return succeeded(fs::exportPrefixLifetimes(function, error));
}
bool freshPrefixLifetime(MLIRContext& context, bool loops)
{
    Fixture fixture(context, loops);
    fs::PrivateStateError error;
    auto prepared = fs::materializeInvocationPrefix(fixture.function, recipe(fixture), error);
    if (failed(prepared) || !closePrefix((*prepared)->function)) {
        return relationalPrefixFailure(__LINE__);
    }
    OwningOpRef<func::FuncOp> logical(cast<func::FuncOp>((*prepared)->function->clone()));
    const auto before = render((*prepared)->function);
    {
        ScopedDiagnosticHandler silence(&context, [](Diagnostic&) { return success(); });
        if (succeeded(fs::allocatePhysicalEventIds((*prepared)->function, {2, 5})) ||
            render((*prepared)->function) != before) {
            return relationalPrefixFailure(__LINE__);
        }
    }
    if (failed(fs::allocatePhysicalEventIds((*prepared)->function, {0, 2, 5}))) {
        return relationalPrefixFailure(__LINE__);
    }
    for (unsigned trips = 0; trips <= 3; ++trips) {
        for (unsigned bits = 0; bits < (1U << (4 * trips)); ++bits) {
            if (!prefixTokens(
                    *logical, (*prepared)->function, trips, bits, {{{P, Q}, 3}, {{P, S}, 3}, {{R, Q}, 3}}, {0, 2, 5})) {
                return relationalPrefixFailure(__LINE__);
            }
        }
    }
    // Roundtrip preserves the certificate and exact binding digest.
    auto saved = parseSourceString<ModuleOp>(
        "module { func.func private @payload(index)\n" + render(*logical) + "\n}", &context);
    auto savedFunction = saved ? saved->lookupSymbol<func::FuncOp>("prefix") : func::FuncOp{};
    if (!savedFunction || failed(fs::allocatePhysicalEventIds(savedFunction, {0, 2, 5}))) {
        return relationalPrefixFailure(__LINE__);
    }
    // A selector change can leave the old counts unchanged. It must not reuse
    // the producer's sealed body certificate.
    OwningOpRef<func::FuncOp> changed(cast<func::FuncOp>((*logical)->clone()));
    pto::LogicalWaitOp wait;
    changed->walk([&](pto::LogicalWaitOp op) {
        if (!wait) {
            wait = op;
        }
    });
    OpBuilder b(wait);
    auto interval = cast<scf::ForOp>(wait->getParentOp());
    auto end = interval.getUpperBound().getDefiningOp<arith::AddIOp>();
    end->setOperand(0, b.create<arith::ConstantIndexOp>(wait.getLoc(), 0));
    const auto changedBefore = render(*changed);
    ScopedDiagnosticHandler silence(&context, [](Diagnostic&) { return success(); });
    return failed(fs::allocatePhysicalEventIds(*changed, {0, 2, 5})) && render(*changed) == changedBefore;
}
bool epochPrefixLifetime(MLIRContext& context, unsigned unsupported = 0, uint64_t sourceBound = 12)
{
    Fixture fixture(context);
    fixture.function.getBody().getBlocks().clear();
    OpBuilder b(&context);
    b.setInsertionPointToStart(fixture.function.addEntryBlock());
    auto loc = fixture.function.getLoc();
    auto zero = b.create<arith::ConstantIndexOp>(loc, 0), one = b.create<arith::ConstantIndexOp>(loc, 1);
    auto loop = b.create<scf::ForOp>(loc, zero, fixture.function.getArgument(0), one);
    fs::InvocationPrefixRecipe spec;
    spec.planId = 31;
    spec.privateScalarControl = true;
    spec.sourceBounds = {{P, sourceBound}};
    spec.directions = {{P, Q}};
    spec.slots = {{b.getIndexType(), b.getIndexAttr(0)}};
    {
        OpBuilder::InsertionGuard restore(b);
        b.setInsertionPointToStart(loop.getBody());
        auto flag = [&](unsigned offset) -> Value {
            Value value = fixture.function.getArgument(9 + offset);
            for (int visit = 1; visit >= 0; --visit) {
                auto position = b.create<arith::ConstantIndexOp>(loc, visit);
                auto active = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, loop.getInductionVar(), position);
                value =
                    b.create<arith::SelectOp>(loc, active, fixture.function.getArgument(1 + visit * 4 + offset), value);
            }
            return value;
        };
        auto active = flag(0), extra = flag(1), consume = flag(2);
        auto branch = b.create<scf::IfOp>(loc, active, false);
        {
            OpBuilder::InsertionGuard arm(b);
            b.setInsertionPointToStart(branch.thenBlock());
            auto payload = [&](unsigned role) {
                auto tag = b.create<arith::ConstantIndexOp>(loc, role);
                return b.create<func::CallOp>(loc, "payload", TypeRange{}, ValueRange{tag});
            };
            auto optional = b.create<scf::IfOp>(loc, extra, false);
            {
                OpBuilder::InsertionGuard optionalArm(b);
                b.setInsertionPointToStart(optional.thenBlock());
                if (unsupported == 3) {
                    auto inner = b.create<scf::ForOp>(loc, zero, fixture.function.getArgument(0), one);
                    OpBuilder::InsertionGuard nested(b);
                    b.setInsertionPointToStart(inner.getBody());
                    spec.sources.push_back({payload(0), P, {}});
                } else {
                    spec.sources.push_back({payload(0), P, {}});
                }
            }
            spec.sources.push_back(
                {payload(0), P,
                 [](OpBuilder&, const IRMapping&, llvm::MutableArrayRef<Value> state, Value rank, std::string&) {
                     state[0] = rank;
                     return success();
                 }});
            auto incoming = [unsupported](
                                OpBuilder& builder, const IRMapping&, ArrayRef<Value> state,
                                std::string&) -> FailureOr<Value> {
                if (unsupported == 2) {
                    auto condition = builder.create<arith::ConstantIntOp>(builder.getUnknownLoc(), 1, 1);
                    auto unrelated = builder.create<scf::IfOp>(
                        builder.getUnknownLoc(), TypeRange{state[0].getType()}, condition, true);
                    for (auto* arm : {unrelated.thenBlock(), unrelated.elseBlock()}) {
                        OpBuilder::InsertionGuard restore(builder);
                        builder.setInsertionPointToEnd(arm);
                        builder.create<scf::YieldOp>(builder.getUnknownLoc(), state[0]);
                    }
                    return unrelated.getResult(0);
                }
                return state[0];
            };
            auto optionalConsumer = b.create<scf::IfOp>(loc, consume, unsupported == 1);
            {
                OpBuilder::InsertionGuard optionalArm(b);
                b.setInsertionPointToStart(optionalConsumer.thenBlock());
                spec.consumers.push_back({payload(2), {P, Q}, incoming});
            }
            if (unsupported == 1) {
                OpBuilder::InsertionGuard elseArm(b);
                b.setInsertionPointToStart(optionalConsumer.elseBlock());
                auto skipped = payload(4);
                spec.actions.push_back(
                    {skipped, fs::PrivateStateCut::Before,
                     [](OpBuilder& builder, const IRMapping&, llvm::MutableArrayRef<Value> state, std::string&) {
                         state[0] = builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), 0);
                         return success();
                     }});
            }
            spec.consumers.push_back({payload(3), {P, Q}, incoming});
        }
        b.create<pto::BarrierOp>(loc, pto::PipeAttr::get(&context, pto::PIPE::PIPE_ALL));
    }
    b.create<func::ReturnOp>(loc);
    fs::PrivateStateError error;
    auto prepared = fs::materializeInvocationPrefix(fixture.function, spec, error);
    if (failed(prepared) || !closePrefix((*prepared)->function)) {
        return relationalPrefixFailure(__LINE__);
    }
    OwningOpRef<func::FuncOp> logical(cast<func::FuncOp>((*prepared)->function->clone()));
    auto lifetime = (*prepared)->function->getAttrOfType<DictionaryAttr>("pto.prefix_lifetimes");
    auto descriptor = cast<DictionaryAttr>(lifetime.getAs<ArrayAttr>("directions")[0]);
    if (descriptor.getAs<StringAttr>("strategy").getValue() != (unsupported ? "fresh-invocation" : "drained-epochs")) {
        return false;
    }
    if (unsupported) {
        const auto before = render((*prepared)->function);
        ScopedDiagnosticHandler silence(&context, [](Diagnostic&) { return success(); });
        return failed(fs::allocatePhysicalEventIds((*prepared)->function, {2, 5})) &&
               render((*prepared)->function) == before;
    }
    if (failed(fs::allocatePhysicalEventIds((*prepared)->function, {2, 5}))) {
        return relationalPrefixFailure(__LINE__);
    }
    unsigned executions = 0;
    for (unsigned trips = 0; trips <= 3; ++trips) {
        for (unsigned bits = 0; bits < (1U << (4 * trips)); ++bits) {
            if (!prefixTokens(*logical, (*prepared)->function, trips, bits, {{{P, Q}, 2}}, {2, 5})) {
                return relationalPrefixFailure(__LINE__);
            }
            ++executions;
        }
    }
    // Removing the already-existing epoch barrier leaves valid logical matching
    // but disables modulo reuse. Re-export falls back to the fresh bound.
    OwningOpRef<func::FuncOp> unclosed(cast<func::FuncOp>((*logical)->clone()));
    SmallVector<Operation*> erase;
    unclosed->walk([&](pto::BarrierOp barrier) {
        if (barrier->getParentOfType<scf::ForOp>()) {
            erase.push_back(barrier);
        }
    });
    for (auto* op : erase) {
        op->erase();
    }
    std::string obligation;
    if (failed(fs::exportPrefixLifetimes(*unclosed, obligation))) {
        return relationalPrefixFailure(__LINE__);
    }
    const auto before = render(*unclosed);
    ScopedDiagnosticHandler silence(&context, [](Diagnostic&) { return success(); });
    if (succeeded(fs::allocatePhysicalEventIds(*unclosed, {2, 5})) || render(*unclosed) != before) {
        return relationalPrefixFailure(__LINE__);
    }
    llvm::outs() << "prefix lifetime checks: fresh and drained epochs; executions=" << executions << "\n";
    return true;
}
bool emptyPrefixLifetime(MLIRContext& context)
{
    Fixture fixture(context);
    fs::InvocationPrefixRecipe spec;
    spec.planId = 44;
    spec.privateScalarControl = true;
    spec.sourceBounds = {{P, 0}};
    spec.directions = {{P, Q}};
    fs::PrivateStateError error;
    auto prepared = fs::materializeInvocationPrefix(fixture.function, spec, error);
    if (failed(prepared) || !closePrefix((*prepared)->function) ||
        failed(fs::allocatePhysicalEventIds((*prepared)->function, {}))) {
        return false;
    }
    bool flags = false;
    (*prepared)->function.walk([&](Operation* op) {
        flags |=
            isa<pto::LogicalSetOp, pto::LogicalWaitOp, pto::SetFlagOp, pto::WaitFlagOp, pto::SetFlagDynOp,
                pto::WaitFlagDynOp>(op);
    });
    return !flags && succeeded(verify((*prepared)->function));
}
bool prefixLifetimeChecks(MLIRContext& context)
{
    return emptyPrefixLifetime(context) && freshPrefixLifetime(context, false) && freshPrefixLifetime(context, true) &&
           epochPrefixLifetime(context) && epochPrefixLifetime(context, 1) && epochPrefixLifetime(context, 2) &&
           epochPrefixLifetime(context, 3) && epochPrefixLifetime(context, 0, 120) &&
           epochPrefixLifetime(context, 0, 120000);
}
#endif
