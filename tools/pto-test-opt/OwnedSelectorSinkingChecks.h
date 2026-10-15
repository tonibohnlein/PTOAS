// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent unfolding of frozen family contributions, before and after SSA.
#ifndef PTO_OWNED_SELECTOR_SINKING_CHECKS_H
#define PTO_OWNED_SELECTOR_SINKING_CHECKS_H
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/OwnedSelectorSinking.h"
namespace {
bool checkOwnedSelectorBoundaries(mlir::MLIRContext* context)
{
    using namespace mlir;
    namespace detail = mlir::pto::frontiersynch::detail;
    OpBuilder builder(context);
    auto location = builder.getUnknownLoc();
    OwningOpRef<func::FuncOp> function(func::FuncOp::create(location, "selector_boundaries",
        builder.getFunctionType({builder.getI64Type(), builder.getI64Type(), builder.getI1Type()}, {})));
    auto* body = function->addEntryBlock();
    builder.setInsertionPointToEnd(body);
    auto original = builder.create<arith::DivUIOp>(location, body->getArgument(0), body->getArgument(1));
    auto* firstCut = builder.create<arith::ConstantIndexOp>(location, 23).getOperation();
    auto* lastCut = builder.create<func::ReturnOp>(location).getOperation();
    Block code;
    builder.setInsertionPointToEnd(&code);
    auto exclusive = builder.create<arith::DivUIOp>(location, body->getArgument(0), body->getArgument(1));
    auto shared = builder.create<arith::AddIOp>(location, body->getArgument(0), body->getArgument(1));
    auto evidence = builder.create<arith::DivUIOp>(location, body->getArgument(0), body->getArgument(1));
    auto differentCut = builder.create<arith::SubIOp>(location, body->getArgument(0), body->getArgument(1));
    auto guard = builder.create<arith::CmpIOp>(location, arith::CmpIPredicate::eq,
        body->getArgument(0), body->getArgument(1));
    auto split = builder.create<arith::AddUIExtendedOp>(location, body->getArgument(0), body->getArgument(1));
    auto together = builder.create<arith::AddUIExtendedOp>(location, body->getArgument(0), body->getArgument(1));
    auto dead = builder.create<arith::ConstantIntOp>(location, 37, 64);
    detail::OwnedSelectorSinking sinking;
    sinking.retain(code, firstCut);
    sinking.preserve(evidence);
    body->getOperations().splice(firstCut->getIterator(), code.getOperations());
    builder.setInsertionPoint(firstCut);
    auto a = builder.create<scf::IfOp>(location, body->getArgument(2), false);
    auto b = builder.create<scf::IfOp>(location, guard, false);
    builder.setInsertionPoint(lastCut);
    auto c = builder.create<scf::IfOp>(location, body->getArgument(2), false);
    sinking.guard(a, firstCut); sinking.guard(b, firstCut); sinking.guard(c, lastCut);
    builder.setInsertionPointToStart(a.thenBlock());
    builder.create<arith::AddIOp>(location, exclusive, original);
    builder.create<arith::AddIOp>(location, evidence, original);
    builder.create<arith::AddIOp>(location, shared, split.getResult(0));
    builder.create<arith::AddIOp>(location, together.getResult(0), original);
    builder.create<arith::OrIOp>(location, together.getResult(1), body->getArgument(2));
    builder.setInsertionPointToStart(b.thenBlock());
    builder.create<arith::AddIOp>(location, shared, original);
    builder.create<arith::OrIOp>(location, split.getResult(1), body->getArgument(2));
    builder.setInsertionPointToStart(c.thenBlock());
    builder.create<arith::AddIOp>(location, differentCut, original);
    if (sinking.run(false) != 2 || exclusive->getBlock() != a.thenBlock() || together->getBlock() != a.thenBlock() ||
        original->getBlock() != body || evidence->getBlock() != body || shared->getBlock() != body ||
        differentCut->getBlock() != body ||
        guard->getBlock() != body || split->getBlock() != body || dead->getBlock() != body) { return false; }
    return succeeded(verify(*function));
}
bool checkIdenticalSelectorGuardBoundaries(mlir::MLIRContext* context)
{
    using namespace mlir;
    namespace detail = mlir::pto::frontiersynch::detail;
    for (unsigned mode = 0; mode < 5; ++mode) {
        OpBuilder builder(context);
        auto location = builder.getUnknownLoc();
        OwningOpRef<func::FuncOp> function(func::FuncOp::create(location, "shared_guard_boundaries",
            builder.getFunctionType({builder.getI64Type(), builder.getI64Type(),
                                     builder.getI1Type(), builder.getI1Type()}, {})));
        auto* body = function->addEntryBlock();
        builder.setInsertionPointToEnd(body);
        auto* cut = builder.create<func::ReturnOp>(location).getOperation();
        Block code;
        builder.setInsertionPointToEnd(&code);
        auto quotient = builder.create<arith::DivUIOp>(location, body->getArgument(0), body->getArgument(1));
        detail::OwnedSelectorSinking sinking;
        sinking.retain(code, cut);
        body->getOperations().splice(cut->getIterator(), code.getOperations());
        builder.setInsertionPoint(cut);
        auto first = builder.create<scf::IfOp>(location, body->getArgument(2), false);
        scf::IfOp barrier;
        if (mode == 1) {
            barrier = builder.create<scf::IfOp>(location, body->getArgument(2), false);
            OpBuilder::InsertionGuard restore(builder);
            builder.setInsertionPointToStart(barrier.thenBlock());
            builder.create<pto::BarrierOp>(location, pto::PipeAttr::get(context, pto::PIPE::PIPE_MTE1));
        }
        auto last = builder.create<scf::IfOp>(location, body->getArgument(mode == 2 ? 3 : 2), false);
        if (mode == 3) { last->setAttr("test.distinct", builder.getUnitAttr()); }
        sinking.guard(first, cut); sinking.guard(last, mode == 4 ? first.getOperation() : cut);
        for (auto entry : llvm::enumerate(SmallVector<scf::IfOp>{first, last})) {
            auto branch = entry.value();
            builder.setInsertionPointToStart(branch.thenBlock());
            auto use = builder.create<arith::AddIOp>(location, quotient, body->getArgument(0));
            use->setAttr("test.order", builder.getI64IntegerAttr(entry.index()));
        }
        const auto moved = sinking.run(false);
        if (moved != (mode ? 0U : 1U) || sinking.mergedGuards() != (mode ? 0U : 1U) ||
            quotient->getBlock() != (mode ? body : first.thenBlock()) || failed(verify(*function))) { return false; }
        SmallVector<unsigned> order;
        unsigned branches = 0;
        function->walk([&](Operation* operation) {
            branches += isa<scf::IfOp>(operation);
            if (auto tag = operation->getAttrOfType<IntegerAttr>("test.order")) { order.push_back(tag.getInt()); }
            if (isa<pto::BarrierOp>(operation)) { order.push_back(7); }
        });
        const SmallVector<unsigned> expected = mode == 1 ? SmallVector<unsigned>{0, 7, 1} :
            SmallVector<unsigned>{0, 1};
        if (order != expected || branches != (mode == 1 ? 3U : mode ? 2U : 1U)) { return false; }
    }
    return true;
}
bool checkSelectorGuardPartitionCost(mlir::MLIRContext* context)
{
    using namespace mlir;
    namespace detail = mlir::pto::frontiersynch::detail;
    // Removing one consumer can split a complete guard set and add scalar
    // control. This is why input-only dominance cannot replace the full count.
    for (bool subset : {false, true}) {
        OpBuilder builder(context); auto loc = builder.getUnknownLoc();
        OwningOpRef<func::FuncOp> function(func::FuncOp::create(loc, "guard_partition_cost",
            builder.getFunctionType({builder.getI64Type(), builder.getI64Type(), builder.getI1Type(),
                                     builder.getI1Type(), builder.getI1Type()}, {})));
        auto* body = function->addEntryBlock(); builder.setInsertionPointToEnd(body);
        auto* cut = builder.create<func::ReturnOp>(loc).getOperation();
        Block code; builder.setInsertionPointToEnd(&code);
        auto a = builder.create<arith::DivUIOp>(loc, body->getArgument(0), body->getArgument(1));
        auto b = builder.create<arith::AddIOp>(loc, a, body->getArgument(0));
        detail::OwnedSelectorSinking sinking; sinking.retain(code, cut);
        body->getOperations().splice(cut->getIterator(), code.getOperations());
        for (unsigned i = 0; i < 3; ++i) {
            builder.setInsertionPoint(cut);
            auto branch = builder.create<scf::IfOp>(loc, body->getArgument(i + 2), false);
            sinking.guard(branch, cut); builder.setInsertionPointToStart(branch.thenBlock());
            builder.create<arith::AddIOp>(loc, a, body->getArgument(0));
            if (!subset || i != 2) { builder.create<arith::AddIOp>(loc, b, body->getArgument(0)); }
        }
        unsigned before = 0, after = 0;
        function->walk([&](Operation* op) { before += op->getName().getDialectNamespace() == "arith"; });
        sinking.run();
        function->walk([&](Operation* op) { after += op->getName().getDialectNamespace() == "arith"; });
        if (failed(verify(*function)) || after - before != (subset ? 5U : 4U) ||
            sinking.constructedScalars() != after - before) { return false; }
    }
    return true;
}
bool checkSharedLiteralProvenance(mlir::MLIRContext* context)
{
    using namespace mlir;
    namespace detail = mlir::pto::frontiersynch::detail;
    OpBuilder builder(context); auto location = builder.getUnknownLoc();
    OwningOpRef<func::FuncOp> function(func::FuncOp::create(location, "shared_literal_provenance",
        builder.getFunctionType({builder.getIndexType(), builder.getIndexType(),
                                 builder.getI1Type(), builder.getI1Type()}, {})));
    auto* body = function->addEntryBlock(); builder.setInsertionPointToEnd(body);
    auto* cut = builder.create<func::ReturnOp>(location).getOperation();
    Block code; builder.setInsertionPointToEnd(&code);
    auto ordinal = builder.create<arith::ConstantIndexOp>(location, 0);
    auto member = builder.create<arith::ConstantIndexOp>(location, 7);
    auto coordinate = builder.create<arith::DivUIOp>(location, body->getArgument(0), body->getArgument(1));
    detail::OwnedSelectorSinking sinking; sinking.retain(code, cut);
    body->getOperations().splice(cut->getIterator(), code.getOperations());
    SmallVector<Operation*> commands;
    for (unsigned i = 0; i < 2; ++i) {
        builder.setInsertionPoint(cut);
        auto branch = builder.create<scf::IfOp>(location, body->getArgument(i + 2), false);
        sinking.guard(branch, cut); builder.setInsertionPointToStart(branch.thenBlock());
        OperationState state(location, "pto.logical_set");
        state.addOperands({ordinal.getResult(), member.getResult(), coordinate.getResult()});
        state.addAttribute("src_pipe", pto::PipeAttr::get(context, pto::PIPE::PIPE_MTE2));
        state.addAttribute("dst_pipe", pto::PipeAttr::get(context, pto::PIPE::PIPE_MTE1));
        state.addAttribute("plan_id", builder.getI64IntegerAttr(0));
        state.addAttribute("record_id", builder.getI64IntegerAttr(7));
        state.addAttribute("pto.endpoint_piece", builder.getI64IntegerAttr(i));
        commands.push_back(builder.create(state));
    }
    if (sinking.run() != 1 || sinking.constructedScalars() != 2 || failed(verify(*function)) ||
        ordinal->getBlock() != body || member->getBlock() != body || coordinate->getBlock() == body) {
        return false;
    }
    SmallVector<Operation*> observed;
    function->walk([&](pto::LogicalSetOp command) { observed.push_back(command.getOperation()); });
    if (observed != commands) { return false; }
    for (auto entry : llvm::enumerate(commands)) {
        auto* command = entry.value();
        APInt zero, record;
        auto branch = dyn_cast<scf::IfOp>(command->getParentOp());
        if (!branch || branch.getCondition() != body->getArgument(entry.index() + 2) ||
            !matchPattern(command->getOperand(0), m_ConstantInt(&zero)) || !zero.isZero() ||
            !matchPattern(command->getOperand(1), m_ConstantInt(&record)) || record != APInt(64, 7) ||
            command->getAttrOfType<IntegerAttr>("pto.endpoint_piece").getInt() !=
                static_cast<int64_t>(entry.index())) { return false; }
    }
    return true;
}
bool checkDifferingSelectorGroups(mlir::MLIRContext* context)
{
    using namespace mlir;
    namespace detail = mlir::pto::frontiersynch::detail;
    if (!checkSharedLiteralProvenance(context)) { return false; }
    for (unsigned mode = 0; mode < 6; ++mode) {
        OpBuilder builder(context); auto location = builder.getUnknownLoc();
        OwningOpRef<func::FuncOp> function(func::FuncOp::create(location, "different_guard_groups",
            builder.getFunctionType({builder.getI64Type(), builder.getI64Type(),
                                     builder.getI1Type(), builder.getI1Type()}, {})));
        auto* body = function->addEntryBlock(); builder.setInsertionPointToEnd(body);
        auto* cut = builder.create<arith::ConstantIndexOp>(location, 37).getOperation();
        auto* last = builder.create<func::ReturnOp>(location).getOperation();
        Block code; builder.setInsertionPointToEnd(&code);
        auto quotient = builder.create<arith::DivUIOp>(location, body->getArgument(0), body->getArgument(1));
        auto multi = builder.create<arith::AddUIExtendedOp>(location, quotient, body->getArgument(0));
        auto dependent = builder.create<arith::AddIOp>(location, multi.getResult(0), body->getArgument(0));
        Value condition = body->getArgument(3);
        if (mode == 1) { builder.create<arith::DivUIOp>(location, body->getArgument(0), quotient); }
        if (mode == 2) {
            condition = builder.create<arith::CmpIOp>(location, arith::CmpIPredicate::eq,
                quotient, body->getArgument(0));
        }
        detail::OwnedSelectorSinking sinking; sinking.retain(code, cut);
        if (mode == 5) { sinking.preserve(quotient); }
        body->getOperations().splice(cut->getIterator(), code.getOperations());
        builder.setInsertionPoint(cut);
        if (mode == 3) { builder.create<arith::AddIOp>(location, quotient, body->getArgument(0)); }
        auto a = builder.create<scf::IfOp>(location, body->getArgument(2), false);
        if (mode == 4) { builder.setInsertionPoint(last); }
        auto b = builder.create<scf::IfOp>(location, condition, false);
        sinking.guard(a, cut); sinking.guard(b, mode == 4 ? last : cut);
        builder.setInsertionPointToStart(a.thenBlock());
        auto first = builder.create<arith::AddIOp>(location, dependent, body->getArgument(0));
        first->setAttr("test.value", builder.getI64IntegerAttr(0));
        builder.setInsertionPointToStart(b.thenBlock());
        auto second = builder.create<arith::AddIOp>(location, multi.getResult(0), body->getArgument(0));
        second->setAttr("test.value", builder.getI64IntegerAttr(1));
        builder.create<arith::OrIOp>(location, multi.getResult(1), body->getArgument(3));
        sinking.run();
        if (failed(verify(*function)) || (mode ? quotient->getBlock() != body : quotient->getBlock() == body)) {
            return false;
        }
        if (!mode && sinking.constructedScalars() != 4) { return false; }
        for (unsigned valuation = 0; valuation < 4; ++valuation) {
            const bool x = valuation & 1, y = valuation & 2;
            DenseMap<Value, APInt> values;
            values[body->getArgument(0)] = APInt(64, 17);
            values[body->getArgument(1)] = APInt(64, !mode && !x && !y ? 0 : 3);
            values[body->getArgument(2)] = APInt(1, x); values[body->getArgument(3)] = APInt(1, y);
            SmallVector<uint64_t> observed;
            std::function<bool(Block&)> execute = [&](Block& block) {
                for (auto& operation : block) {
                    if (auto add = dyn_cast<arith::AddUIExtendedOp>(operation)) {
                        bool overflow = false;
                        values[add.getResult(0)] =
                            values.lookup(add.getLhs()).uadd_ov(values.lookup(add.getRhs()), overflow);
                        values[add.getResult(1)] = APInt(1, overflow);
                    } else if (operation.getName().getDialectNamespace() == "arith") {
                        if (!evaluateDeferredScalar(operation, values)) { return false; }
                    } else if (auto branch = dyn_cast<scf::IfOp>(operation)) {
                        auto& region = values.lookup(branch.getCondition()).isZero() ?
                            branch.getElseRegion() : branch.getThenRegion();
                        if (!region.empty()) {
                            if (!execute(region.front())) { return false; }
                            auto yield = cast<scf::YieldOp>(region.front().getTerminator());
                            for (auto [result, value] : llvm::zip(branch.getResults(), yield.getOperands())) {
                                values[result] = values.lookup(value);
                            }
                        }
                    } else if (!isa<scf::YieldOp, func::ReturnOp>(operation)) { return false; }
                    if (operation.hasAttr("test.value")) {
                        observed.push_back(values.lookup(operation.getResult(0)).getZExtValue());
                    }
                }
                return true;
            };
            SmallVector<uint64_t> expected;
            if (x) { expected.push_back(56); }
            if (mode != 2 && y) { expected.push_back(39); }
            if (!execute(*body) || observed != expected) { return false; }
        }
    }
    return true;
}
bool checkOwnedSelectorRoutes(mlir::MLIRContext* context, bool differing = false)
{
    using namespace mlir;
    namespace fs = mlir::pto::frontiersynch;
    using Opcodes = std::map<std::string, unsigned>;
    std::map<std::tuple<unsigned, bool, unsigned>, std::pair<Opcodes, unsigned>> sizes;
    for (unsigned route = 0; route < 4; ++route) {
        for (unsigned scenario = 0; scenario < 12; ++scenario) {
            const unsigned recordCount = scenario / 4 == 0 ? 1 : scenario / 4 == 1 ? 2 : 4;
            const bool renamed = (scenario / 2) % 2, nested = scenario % 2;
            auto fail = [&](llvm::StringRef stage) {
                llvm::errs() << "selector route=" << route << " records=" << recordCount
                    << " renamed=" << renamed << " nested=" << nested << " differing=" << differing
                    << " failed: " << stage << "\n";
                return false;
            };
            OpBuilder builder(context);
            auto location = builder.getUnknownLoc();
            OwningOpRef<ModuleOp> module(ModuleOp::create(location));
            auto function = func::FuncOp::create(location, "conditional_selector",
                builder.getFunctionType({builder.getIndexType(), builder.getIndexType(),
                                         builder.getI1Type(), builder.getI1Type(), builder.getI1Type()}, {}));
            module->push_back(function);
            auto* body = function.addEntryBlock();
            builder.setInsertionPointToEnd(body);
            auto* block = body;
            if (nested) {
                auto outer = builder.create<scf::IfOp>(location, body->getArgument(3), false);
                builder.create<func::ReturnOp>(location);
                block = outer.thenBlock();
                builder.setInsertionPointToStart(block);
            }
            auto* source = builder.create<arith::ConstantIndexOp>(location, 19).getOperation();
            auto* target = nested ? block->getTerminator() : builder.create<func::ReturnOp>(location).getOperation();
            auto arena = std::make_shared<fs::RegionExpressions>();
            // Owned mathematical recipes require a positive constant divisor.
            // Dynamic zero-divisor inactivity is covered by the direct owned
            // preparation checks above; these routes check guarded evaluation,
            // serialization and complete construction with an admitted recipe.
            const auto quotient = arena->div(arena->input(body->getArgument(0)), arena->constant(3));
            SmallVector<fs::ContextualEndpointFamily> endpoints;
            SmallVector<int64_t> records, eligible;
            fs::PreparedLogicalPlan plan(0);
            for (unsigned index = 0; index <= recordCount; ++index) { eligible.push_back(index); }
            for (unsigned index = 0; index < recordCount; ++index) {
                const uint32_t record = renamed ? 1000 + 37 * index : index;
                records.push_back(record);
                for (bool set : {true, false}) {
                    fs::ContextualEndpointFamily endpoint;
                    endpoint.piece.cut = {block, set ? source : target};
                    endpoint.piece.kind = set ? fs::EndpointKind::Set : fs::EndpointKind::Wait;
                    endpoint.piece.sourcePipe = 1; endpoint.piece.targetPipe = 2;
                    endpoint.piece.namespaceId = renamed ? 1000 : 0;
                    endpoint.piece.order = endpoints.size();
                    endpoint.piece.members.push_back({record, {}});
                    endpoint.activity = arena->input(body->getArgument(differing && index % 2 ? 4 : 2));
                    endpoint.ordinal = quotient;
                    endpoints.push_back(endpoint);
                }
                fs::EndpointFamily family;
                family.id = record; family.sourcePipe = 1; family.targetPipe = 2;
                family.sourceCut = {block, source}; family.targetCut = {block, target};
                family.members.push_back({record, 0, 1, {}, {}}); plan.families.push_back(family);
            }
            auto recipe = fs::buildContextualFamilyEndpointRecipe(arena, endpoints);
            if (failed(recipe)) { return fail("recipe preparation"); }
            plan.familyRecipe = *recipe;
            auto direction = builder.getDictionaryAttr({
                builder.getNamedAttr("source", builder.getI64IntegerAttr(1)),
                builder.getNamedAttr("target", builder.getI64IntegerAttr(2)),
                builder.getNamedAttr("budget", builder.getI64IntegerAttr(recordCount + 1)),
                builder.getNamedAttr("records", builder.getDenseI64ArrayAttr(records))});
            plan.allocationCertificate = builder.getDictionaryAttr({
                builder.getNamedAttr("version", builder.getI64IntegerAttr(2)),
                builder.getNamedAttr("plan", builder.getI64IntegerAttr(0)),
                builder.getNamedAttr("directions", builder.getArrayAttr({direction}))});
            std::optional<uint64_t> predicted;
            if (route == 1) {
                if (failed(fs::planPhysicalFamilyOutput(function, plan, eligible))) { return fail("physical plan"); }
                predicted = fs::physicalFamilyConstructionOperations(plan);
                if (!predicted) { return fail("complete construction count"); }
            }
            plan.deferredOutput = route == 2;
            if (failed(fs::insertLogicalSynchronization(function, plan))) { return fail("materialization"); }
            if (predicted && *predicted != plan.familyWork.preparationScalars + plan.familyWork.compactionScalars +
                plan.familyWork.mappingScalars + plan.familyWork.barrierScalars) {
                llvm::errs() << "predicted=" << *predicted << " preparation=" << plan.familyWork.preparationScalars
                    << " compaction=" << plan.familyWork.compactionScalars
                    << " mapping=" << plan.familyWork.mappingScalars
                    << " barriers=" << plan.familyWork.barrierScalars << "\n";
                return fail("construction parity");
            }
            if (route >= 2) {
                std::string text;
                llvm::raw_string_ostream output(text); module->print(output);
                auto parsed = parseSourceString<ModuleOp>(output.str(), context);
                if (!parsed) { return fail("saved parse"); }
                module = std::move(parsed); function = *module->getOps<func::FuncOp>().begin();
                if (failed(fs::allocatePhysicalEventIds(function, eligible))) { return fail("saved allocation"); }
                // Deferred allocation commits a cloned body. Reacquire its
                // arguments before checking exact guard identity or execution.
                body = &function.front();
            }
            if (failed(verify(function))) { return fail("output verification"); }
            unsigned divisions = 0;
            bool conditional = true;
            function.walk([&](arith::DivUIOp op) {
                ++divisions;
                auto branch = op->getParentOfType<scf::IfOp>();
                // Ordinary split allocation removes temporary cut metadata.
                // The exact command activity still identifies the guarded scope.
                if (differing && recordCount > 1) {
                    auto joined = branch ? branch.getCondition().getDefiningOp<arith::OrIOp>() : arith::OrIOp{};
                    conditional &= joined &&
                        ((joined.getLhs() == body->getArgument(2) && joined.getRhs() == body->getArgument(4)) ||
                         (joined.getLhs() == body->getArgument(4) && joined.getRhs() == body->getArgument(2)));
                } else { conditional &= branch && branch.getCondition() == body->getArgument(2); }
            });
            if (divisions != 2 || !conditional) {
                llvm::errs() << "division sites=" << divisions << " conditional=" << conditional << "\n";
                return fail("static divisions");
            }
            unsigned constants = 0, branches = 0;
            Opcodes arithmetic;
            function.walk([&](Operation* operation) {
                if (isa<arith::ConstantOp>(operation)) { ++constants; }
                else if (operation->getName().getDialectNamespace() == "arith") {
                    ++arithmetic[operation->getName().getStringRef().str()];
                }
                branches += isa<scf::IfOp>(operation);
            });
            const unsigned expectedBranches = differing && recordCount > 1 ? recordCount * 2 + 2 : 2;
            if (branches != expectedBranches + unsigned(nested)) {
                llvm::errs() << "branches=" << branches << " expected=" << expectedBranches + unsigned(nested) << "\n";
                return fail("static branches");
            }
            // A member literal can share the divisor constant before renaming.
            // Require the same dispatch/arithmetic schedule, not identical
            // literal pooling. Complete construction parity above still counts
            // every constructed constant separately for each mapping request.
            const auto shape = std::make_pair(arithmetic, branches);
            auto [entry, inserted] = sizes.emplace(std::make_tuple(route, nested, recordCount), shape);
            if (!inserted && entry->second != shape) {
                llvm::errs() << "renamed constants=" << constants << " branches=" << branches << "\n";
                for (const auto& [opcode, count] : arithmetic) {
                    auto previous = entry->second.first.find(opcode);
                    llvm::errs() << opcode << " count=" << count << " previous="
                        << (previous == entry->second.first.end() ? 0U : previous->second) << "\n";
                }
                return fail("renaming dispatch invariance");
            }
            for (unsigned valuation = 0; valuation < 8; ++valuation) {
                const bool active = valuation & 1, outer = valuation & 2, second = valuation & 4;
                const bool any = active || (differing && recordCount > 1 && second);
                {
                    DenseMap<Value, APInt> values;
                    values[body->getArgument(0)] = APInt(64, 17);
                    values[body->getArgument(1)] = APInt(64, any ? 3 : 0);
                    values[body->getArgument(2)] = APInt(1, active);
                    values[body->getArgument(3)] = APInt(1, outer);
                    values[body->getArgument(4)] = APInt(1, second);
                    SmallVector<std::tuple<bool, uint64_t, uint64_t>> commands;
                    unsigned evaluated = 0;
                    std::function<bool(Block&)> execute = [&](Block& current) {
                        for (auto& operation : current) {
                            if (auto division = dyn_cast<arith::DivUIOp>(operation)) {
                                if (values.lookup(division.getRhs()).isZero()) { return false; }
                                ++evaluated;
                            }
                            if (operation.getName().getDialectNamespace() == "arith") {
                                if (!evaluateDeferredScalar(operation, values)) { return false; }
                            } else if (auto branch = dyn_cast<scf::IfOp>(operation)) {
                                auto& region = values.lookup(branch.getCondition()).isZero() ?
                                    branch.getElseRegion() : branch.getThenRegion();
                                if (!region.empty()) {
                                    if (!execute(region.front())) { return false; }
                                    auto yielded = cast<scf::YieldOp>(region.front().getTerminator());
                                    for (auto entry : llvm::zip(branch.getResults(), yielded.getOperands())) {
                                        values[std::get<0>(entry)] = values.lookup(std::get<1>(entry));
                                    }
                                }
                            } else if (isa<pto::LogicalSetOp, pto::LogicalWaitOp,
                                           pto::SetFlagDynOp, pto::WaitFlagDynOp>(operation)) {
                                const bool logical = isa<pto::LogicalSetOp, pto::LogicalWaitOp>(operation);
                                commands.push_back({isa<pto::LogicalSetOp, pto::SetFlagDynOp>(operation),
                                    values.lookup(operation.getOperand(0)).getZExtValue(), logical ?
                                    values.lookup(operation.getOperand(1)).getZExtValue() : UINT64_MAX});
                            } else if (!isa<scf::YieldOp, func::ReturnOp>(operation)) { return false; }
                        }
                        return true;
                    };
                    SmallVector<std::tuple<bool, uint64_t, uint64_t>> expected;
                    const bool enabled = any && (!nested || outer);
                    if (enabled) {
                        for (bool set : {true, false}) {
                            for (unsigned index = 0; index < recordCount; ++index) {
                                if (!(differing && index % 2 ? second : active)) { continue; }
                                expected.push_back({set, route ? (recordCount * 5 + index) % (recordCount + 1) : 5U,
                                    route ? UINT64_MAX : static_cast<uint64_t>(records[index])});
                            }
                        }
                    }
                    if (!execute(*body) || commands != expected || evaluated != (enabled ? 2U : 0U)) {
                        llvm::errs() << "valuation=" << valuation << " commands=" << commands.size()
                            << " expected=" << expected.size() << " divisions=" << evaluated << "\n";
                        return fail("independent command trace");
                    }
                }
            }
        }
    }
    return true;
}
} // namespace
#endif
