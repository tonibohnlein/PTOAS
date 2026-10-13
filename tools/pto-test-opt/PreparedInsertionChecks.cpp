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
#include "PTO/Transforms/FrontierSynch/Shared/LogicalInsertion.h"
#include "PTO/Transforms/FrontierSynch/Shared/ExecutionContexts.h"
#include "PTO/Transforms/FrontierSynch/Shared/RegionExpressions.h"
#include "PTO/Transforms/FrontierSynch/Shared/GuardedPublication.h"
#include "PTO/IR/PTOSyncCapabilities.h"
#include "PTO/Transforms/FrontierSynch/Finite/FiniteAllocation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include "EndpointOrderChecks.h"
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
bool publicationRecipeChecks(MLIRContext* context)
{
    OpBuilder builder(context);
    auto location = builder.getUnknownLoc();
    OwningOpRef<ModuleOp> module = ModuleOp::create(location);
    builder.setInsertionPointToStart(module->getBody());
    auto function = builder.create<func::FuncOp>(location, "publication_recipe",
        builder.getFunctionType({builder.getIndexType()}, {}));
    builder.setInsertionPointToStart(function.addEntryBlock());
    auto one = builder.create<arith::ConstantIndexOp>(location, 1);
    auto source = builder.create<arith::AddIOp>(location, function.getArgument(0), one);
    auto condition = builder.create<arith::CmpIOp>(location, arith::CmpIPredicate::eq, source, one);
    auto consumer = builder.create<arith::AddIOp>(location, source, one);
    auto terminal = builder.create<func::ReturnOp>(location);
    fs::RegionExpressions arena, foreign;
    // Model an original value that must not be replayed before its producer.
    arena.forbidRecomputation(source);
    auto guard = arena.input(condition);
    auto sourceCut = fs::TemplateEndpointCut{source->getBlock(), source};
    auto consumerCut = fs::TemplateEndpointCut{consumer->getBlock(), consumer};
    std::map<Operation*, fs::RegionExpressions::CutEmission> preparation;
    auto available = [&](fs::RegionExpressions::Id expression, Operation* cut) {
        return success(succeeded(arena.prepareContextual(expression, cut, preparation[cut])));
    };
    const auto original = render(function);
    std::string error;
    auto recipe = fs::preparePublicationRecipe(function, arena, sourceCut, consumerCut,
        guard, guard, guard, available, error);
    if (failed(recipe) || render(function) != original) { return false; }
    for (const auto& entry : preparation) {
        if (!entry.second.values.empty() || !entry.second.inputs.empty()) { return false; }
    }
    fs::PreparedLogicalPlan plan(0);
    std::map<Operation*, fs::RegionExpressions::CutEmission> contexts;
    auto emit = [&](fs::RegionExpressions::Id expression, Operation* cut) {
        builder.setInsertionPointToEnd(&plan.addPreparation(cut));
        return arena.emitContextual(expression, builder, cut, contexts[cut]);
    };
    if (succeeded(fs::materializePublicationRecipe(**recipe, foreign, emit, condition, error)) ||
        !plan.preparation.empty()) { return false; }
    auto publication = fs::materializePublicationRecipe(**recipe, arena, emit, condition, error);
    if (failed(publication) || publication->cleanupCut != terminal.getOperation() ||
        !publication->targetChoices || render(function) != original) { return false; }
    auto direct = fs::preparePublicationRecipe(function, arena, sourceCut, consumerCut,
        arena.boolean(true), arena.boolean(true), arena.boolean(true), available, error);
    if (failed(direct)) { return false; }
    auto ordinary = fs::materializePublicationRecipe(**direct, arena, emit, condition, error);
    return succeeded(ordinary) && !ordinary->cleanupCut && !ordinary->targetChoices;
}
bool cleanupSelectionChecks(MLIRContext* context)
{
    context->getOrLoadDialect<scf::SCFDialect>();
    for (int64_t lower : {-3, 0, 2}) {
        for (int64_t step : {1, 2}) {
            for (int64_t count : {0, 1, 4}) {
                OpBuilder builder(context);
                auto location = builder.getUnknownLoc();
                OwningOpRef<ModuleOp> module = ModuleOp::create(location);
                builder.setInsertionPointToStart(module->getBody());
                auto type = builder.getFunctionType({builder.getIndexType(), builder.getI1Type()}, {});
                auto function = builder.create<func::FuncOp>(location, "cleanup_selection", type);
                builder.setInsertionPointToStart(function.addEntryBlock());
                auto lo = builder.create<arith::ConstantIndexOp>(location, lower);
                auto hi = builder.create<arith::ConstantIndexOp>(location, lower + count * step);
                auto stride = builder.create<arith::ConstantIndexOp>(location, step);
                auto loop = builder.create<scf::ForOp>(location, lo, hi, stride);
                auto terminal = builder.create<func::ReturnOp>(location).getOperation();
                builder.setInsertionPoint(loop.getBody()->getTerminator());
                auto difference = builder.create<arith::SubIOp>(location, loop.getInductionVar(), lo);
                auto visit = builder.create<arith::DivUIOp>(location, difference, stride);
                auto first = builder.create<arith::CmpIOp>(location, arith::CmpIPredicate::eq, visit, stride);
                auto cut = builder.create<arith::AddIOp>(location, lo, stride).getOperation();
                fs::RegionExpressions arena;
                auto offset = arena.sub(arena.input(loop.getInductionVar()), arena.input(lo));
                auto ordinal = arena.div(offset, arena.constant(step));
                auto selected = arena.eq(ordinal, arena.constant(1));
                auto residual = arena.input(function.getArgument(1));
                std::vector<std::pair<fs::RegionExpressions::Id, bool>> predicates = {
                    {arena.eq(arena.input(loop.getInductionVar()), arena.constant(lower)), true},
                    {selected, true}, {arena.land(selected, residual), true},
                    {arena.select(residual, selected, arena.boolean(false)), true},
                    {arena.lt(ordinal, arena.constant(2)), false},
                    {arena.eq(arena.div(offset, arena.constant(2 * step)), arena.constant(1)), false},
                    {arena.eq(ordinal, arena.add(ordinal, arena.constant(1))), false},
                    {arena.boolean(false), true}, {arena.boolean(true), false},
                    {arena.eq(ordinal, arena.input(function.getArgument(0))), true},
                    {arena.input(first), true},
                    {arena.lor(selected, arena.eq(ordinal, arena.constant(2))), false}};
                SmallVector<fs::TemplateEndpointCut> cuts{{cut->getBlock(), cut}, {terminal->getBlock(), terminal}};
                const auto original = render(function);
                for (auto [predicate, expected] : predicates) {
                    const bool planned = fs::qualifyTerminalCleanupSelection(cuts, arena, predicate);
                    if (planned != expected || render(function) != original) { return false; }
                    Block detached;
                    builder.setInsertionPointToEnd(&detached);
                    llvm::DenseMap<fs::RegionExpressions::Id, Value> values;
                    auto emitted = arena.emit(predicate, builder, cut, values);
                    if (failed(emitted) || bool(fs::qualifyTerminalCleanupCuts(cuts, *emitted)) != expected) {
                        return false;
                    }
                }
                if (fs::qualifyTerminalCleanupSelection(cuts, arena, fs::RegionExpressions::invalid) ||
                    fs::qualifyTerminalCleanupSelection(cuts, arena, arena.constant(0)) ||
                    fs::qualifyTerminalCleanupSelection({}, arena, selected)) { return false; }
                // Original then-arm facts still constrain a predicate that is
                // itself true; an else arm does not imply the equality.
                builder.setInsertionPoint(cut);
                auto branch = builder.create<scf::IfOp>(location, first, true);
                for (bool thenArm : {true, false}) {
                    auto* branchCut = thenArm ? branch.thenBlock()->getTerminator() :
                                                branch.elseBlock()->getTerminator();
                    cuts.front() = {branchCut->getBlock(), branchCut};
                    if (fs::qualifyTerminalCleanupSelection(cuts, arena, arena.boolean(true)) != thenArm) {
                        return false;
                    }
                }
                // One fixed coordinate in the inner loop is insufficient when
                // the selected value varies with an enclosing invocation.
                builder.setInsertionPoint(cut);
                auto inner = builder.create<scf::ForOp>(location, lo, hi, stride);
                builder.setInsertionPoint(inner.getBody()->getTerminator());
                auto nested = builder.create<arith::AddIOp>(location, lo, stride).getOperation();
                auto dependent = arena.eq(arena.input(inner.getInductionVar()), arena.input(loop.getInductionVar()));
                auto both = arena.land(arena.eq(arena.input(inner.getInductionVar()), arena.constant(lower)),
                    arena.eq(arena.input(loop.getInductionVar()), arena.constant(lower)));
                cuts.front() = {nested->getBlock(), nested};
                if (fs::qualifyTerminalCleanupSelection(cuts, arena, dependent) ||
                    !fs::qualifyTerminalCleanupSelection(cuts, arena, both)) { return false; }
            }
        }
    }
    return true;
}
bool cleanupPredicateChecks(MLIRContext* context)
{
    OpBuilder builder(context);
    auto location = builder.getUnknownLoc();
    OwningOpRef<ModuleOp> module = ModuleOp::create(location);
    builder.setInsertionPointToStart(module->getBody());
    auto type = builder.getFunctionType({builder.getI1Type()}, {});
    auto function = builder.create<func::FuncOp>(location, "cleanup_predicate", type);
    builder.setInsertionPointToStart(function.addEntryBlock());
    auto zero = builder.create<arith::ConstantIndexOp>(location, 0);
    auto source = builder.create<arith::AddIOp>(location, zero, zero).getOperation();
    auto target = builder.create<arith::AddIOp>(location, zero, zero).getOperation();
    auto terminal = builder.create<func::ReturnOp>(location).getOperation();
    fs::PreparedLogicalPlan plan(0);
    plan.independentPieces = true;
    plan.groupedFamilies = true;
    builder.setInsertionPointToEnd(&plan.addPreparation(source));
    auto yes = builder.create<arith::ConstantIntOp>(location, 1, 1);
    Value first = builder.create<arith::AndIOp>(location, function.getArgument(0), yes);
    Value duplicate = builder.create<arith::AndIOp>(location, function.getArgument(0), yes);
    auto unused = builder.create<arith::XOrIOp>(location, duplicate, yes);
    fs::EndpointFamily family;
    family.sourcePipe = 4; family.targetPipe = 1;
    family.sourceCut = {source->getBlock(), source};
    family.targetCut = {target->getBlock(), target};
    family.members.push_back({0, 0, 1, {}, {}});
    family.targetChoices = fs::qualifyTerminalCleanupCuts(
        {family.targetCut, {terminal->getBlock(), terminal}}, duplicate);
    if (!family.targetChoices) { return false; }
    plan.families.push_back(std::move(family));
    using Kind = fs::LogicalCommandKind;
    plan.endpoints = {{source, Kind::Set, 4, 1, 0, yes, zero, {zero}, {0}, 0},
                      {target, Kind::Wait, 4, 1, 0, duplicate, zero, {zero}, {0}, 1},
                      {terminal, Kind::Wait, 4, 1, 0, unused, zero, {zero}, {0}, 2}};
    if (failed(fs::insertLogicalSynchronization(function, plan)) || failed(verify(*module)) ||
        plan.families.front().targetChoices->consumerPredicate() != first) { return false; }
    for (int64_t active : {0, 1}) {
        function->setAttr("test.trace_arguments", builder.getDenseI64ArrayAttr({active}));
        auto trace = traceStructuredLogicalInsertion(function, {});
        auto* events = trace.getArray("events");
        if (trace.getString("error") != "" || !events || events->size() != 2 ||
            (*events)[0].getAsObject()->getString("kind") != "set" ||
            (*events)[1].getAsObject()->getString("kind") != "wait") { return false; }
    }
    return true;
}
bool coalescingChecks(MLIRContext* context)
{
    context->getOrLoadDialect<scf::SCFDialect>();
    // Different coordinates are exclusive; repeated coordinates and commands
    // interleaved in another direction must retain their multiplicity/order.
    for (unsigned mode = 0; mode < 7; ++mode) {
        for (unsigned count : {8U, 128U}) {
            OpBuilder builder(context);
            auto location = builder.getUnknownLoc();
            OwningOpRef<ModuleOp> module = ModuleOp::create(location);
            builder.setInsertionPointToStart(module->getBody());
            auto type = builder.getFunctionType({builder.getI1Type()}, {});
            auto function = builder.create<func::FuncOp>(location, "alternatives", type);
            function->setAttr("test.trace_arguments", builder.getDenseI64ArrayAttr({1}));
            builder.setInsertionPointToStart(function.addEntryBlock());
            auto zero = builder.create<arith::ConstantIndexOp>(location, 0);
            auto one = builder.create<arith::ConstantIndexOp>(location, 1);
            auto end = builder.create<arith::ConstantIndexOp>(location, count);
            auto loop = builder.create<scf::ForOp>(location, zero, end, one);
            builder.create<func::ReturnOp>(location);
            builder.setInsertionPoint(loop.getBody()->getTerminator());
            auto cut = builder.create<arith::AddIOp>(location, zero, one).getOperation();
            fs::PreparedLogicalPlan plan(0);
            plan.independentPieces = true;
            plan.groupedFamilies = true;
            builder.setInsertionPointToEnd(&plan.addPreparation(cut));
            for (unsigned record = 0; record < count; ++record) {
                const auto coordinate = mode == 1 ? 0 : mode == 6 ?
                    (record / 4) * 2 + (record % 4 == 1 || record % 4 == 2) : record;
                const bool alternating = mode == 2 || mode >= 5;
                const uint32_t target = alternating && record % 2 ? 3 : 0;
                const auto name = alternating ? record % 2 : 0;
                auto label = builder.create<arith::ConstantIndexOp>(location, record);
                auto value = builder.create<arith::ConstantIndexOp>(location, coordinate);
                Value guard = builder.create<arith::CmpIOp>(location, arith::CmpIPredicate::eq,
                                                           loop.getInductionVar(), value);
                if (mode >= 3) {
                    Value predicate = function.getArgument(0);
                    if (mode == 4 && record % 2) {
                        auto yes = builder.create<arith::ConstantIntOp>(location, 1, 1);
                        predicate = builder.create<arith::XOrIOp>(location, predicate, yes);
                    }
                    guard = builder.create<arith::AndIOp>(location, guard, predicate);
                }
                fs::EndpointFamily family;
                family.id = record; family.sourcePipe = 4; family.targetPipe = target;
                family.sourceCut = {cut->getBlock(), cut}; family.targetCut = family.sourceCut;
                family.members.push_back({record, 0, 0, {{loop, coordinate}}, {{loop, coordinate}}});
                plan.families.push_back(std::move(family));
                for (auto kind : {fs::LogicalCommandKind::Set, fs::LogicalCommandKind::Wait}) {
                    plan.endpoints.push_back({cut, kind, 4, target, name, guard, zero, {label},
                                              {record}, static_cast<int64_t>(plan.endpoints.size())});
                    fs::retainEndpointCoordinates(plan.endpoints.back(), {{loop, coordinate}});
                    if (mode == 5) {
                        // The same proof may arrive as a producer-owned residue
                        // domain; no emitted Boolean expression is decoded.
                        plan.endpoints.back().domain = {{{loop.getInductionVar(), count, coordinate}}};
                    }
                }
            }
            if (failed(fs::insertLogicalSynchronization(function, plan)) || failed(verify(*module))) { return false; }
            unsigned sets = 0, waits = 0;
            function.walk([&](pto::LogicalSetOp) { ++sets; });
            function.walk([&](pto::LogicalWaitOp) { ++waits; });
            const auto expected = mode == 0 || mode == 3 ? 1 : (mode == 2 || mode == 4 || mode == 5) ? 2 : count;
            if ((mode != 6 && (sets != expected || waits != expected)) ||
                (mode == 6 && (sets < 3 || waits < 3 || sets > count || waits > count)) ||
                plan.families.size() != count) { return false; }
            unsigned coverage = 0;
            for (const auto& endpoint : plan.endpoints) { coverage += endpoint.records.size(); }
            if (coverage != 2 * count) { return false; }
            auto trace = traceStructuredLogicalInsertion(function, {});
            auto* events = trace.getArray("events");
            const unsigned commands = mode == 4 ? count : 2 * count;
            if (trace.getString("error") != "" || !events || events->size() != commands) { return false; }
            for (unsigned i = 0; i < commands; ++i) {
                auto* event = (*events)[i].getAsObject();
                const bool publish = mode == 1 ? i < count : mode == 6 ? i % 4 < 2 : i % 2 == 0;
                constexpr unsigned crossingOrder[] = {0, 3, 0, 3, 1, 2, 1, 2};
                const auto record = mode == 1 ? i % count : mode == 6 ? (i / 8) * 4 + crossingOrder[i % 8] :
                    (mode == 4 ? 2 : 1) * (i / 2);
                auto* selected = event ? event->getArray("members") : nullptr;
                if (!selected || selected->size() != 1 || (*selected)[0].getAsInteger() != record ||
                    event->getString("kind") != (publish ? "set" : "wait")) { return false; }
            }
            if (mode == 3) {
                function->setAttr("test.trace_arguments", builder.getDenseI64ArrayAttr({0}));
                auto inactive = traceStructuredLogicalInsertion(function, {});
                if (inactive.getString("error") != "" || !inactive.getArray("events")->empty()) { return false; }
            }
            loop.setUpperBound(zero);
            auto empty = traceStructuredLogicalInsertion(function, {});
            if (empty.getString("error") != "" || !empty.getArray("events")->empty()) { return false; }
        }
    }
    return true;
}
LogicalResult sectionContextChecks(func::FuncOp function)
{
    const auto before = render(function);
    const bool functionGuard = pto::needsA5NoSplitVectorGuard(function);
    unsigned projected = 0;
    auto prepare = [&](func::FuncOp projectedFunction)
        -> FailureOr<std::unique_ptr<fs::PreparedLogicalPlan>> {
        ++projected;
        if (fs::hasPhysicalSections(projectedFunction)) { return failure(); }
        unsigned branches = 0;
        Value shared;
        bool same = true;
        projectedFunction.walk([&](scf::IfOp branch) {
            ++branches;
            same &= !shared || shared == branch.getCondition();
            shared = branch.getCondition();
        });
        if (branches != 2 || !same || !shared) { return failure(); }
        auto plan = std::make_unique<fs::PreparedLogicalPlan>(0);
        fs::ExplicitAnalysis empty;
        plan->allocationCertificate = fs::explicitAllocationCertificate(empty, 0, function.getContext());
        return plan;
    };
    {
        ScopedDiagnosticHandler silence(function.getContext(), [](Diagnostic&) { return success(); });
        auto reject = [](func::FuncOp) -> FailureOr<std::unique_ptr<fs::PreparedLogicalPlan>> {
            return failure();
        };
        if (succeeded(fs::insertContextSynchronization(function, reject)) || render(function) != before) {
            return function.emitError("failed section preparation changed the original function");
        }
    }
    if (failed(fs::insertContextSynchronization(function, prepare)) || projected != 1 ||
        pto::needsA5NoSplitVectorGuard(function) != functionGuard || failed(verify(function))) {
        return function.emitError("section projection lost implicit participation or SSA visibility");
    }
    unsigned branches = 0;
    bool doubleGuard = false;
    function.walk([&](pto::SectionVectorOp section) {
        doubleGuard |= pto::needsA5NoSplitVectorGuard(section);
        branches += llvm::count_if(section.getBody().front(), [](Operation& op) { return isa<scf::IfOp>(op); });
    });
    if (branches != 2 || doubleGuard) { return function.emitError("incorrect section predicate materialization"); }
    auto bundle = function->getAttr(fs::ContextPlansAttr);
    function->setAttr(fs::ContextPlansAttr, DictionaryAttr::get(function.getContext()));
    auto invalid = render(function);
    {
        ScopedDiagnosticHandler silence(function.getContext(), [](Diagnostic&) { return success(); });
        if (succeeded(fs::allocateContextSynchronization(function, {0,1,2,3,4,5})) ||
            render(function) != invalid) { return failure(); }
    }
    function->setAttr(fs::ContextPlansAttr, bundle);
    if (failed(fs::allocateContextSynchronization(function, {0,1,2,3,4,5}))) { return failure(); }
    llvm::outs() << "section-context: predicates preserved; failed transactions unchanged\n";
    return success();
}
} // namespace
LogicalResult runPreparedInsertionChecks(func::FuncOp function)
{
    if (!endpointOrderGraphChecks() || !endpointOrderFamilyChecks(function.getContext())) {
        return function.emitError("early family refinement changed cycle membership or unfolded order");
    }
    if (!publicationRecipeChecks(function.getContext())) {
        return function.emitError("publication recipe changed availability or cleanup");
    }
    if (!cleanupSelectionChecks(function.getContext())) {
        return function.emitError("cleanup selection changed across scalar lowering");
    }
    if (!cleanupPredicateChecks(function.getContext())) {
        return function.emitError("preparation deduplication lost terminal cleanup predicate");
    }
    if (!coalescingChecks(function.getContext())) { return function.emitError("endpoint coalescing changed coverage"); }
    if (fs::hasPhysicalSections(function)) { return sectionContextChecks(function); }
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
    const auto scalar = static_cast<uint32_t>(pto::PIPE::PIPE_S);
    bad.endpoints = {{cuts[0], Kind::Barrier, scalar, scalar, 0, guard, {}}};
    if (!rejectedUnchanged(function, bad)) {
        return function.emitError("forbidden scalar barrier was accepted or rejection changed IR");
    }
    bad.endpoints.clear();
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
