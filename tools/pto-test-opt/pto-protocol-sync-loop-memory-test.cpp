// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- pto-protocol-sync-loop-memory-test.cpp - Dynamic hazard oracle -------===//
// Expand small programs into actual phase occurrences and compare every byte-
// overlap hazard against instantiated canonical requirements. The reference
// uses fixture byte sets, not production atoms or outstanding-access transfer.

#include "PTO/Transforms/InsertSync/LegacySyncIRAdapter.h"
#include "PTO/Transforms/ProtocolSync/DirectRepair.h"
#include "PTO/Transforms/ProtocolSync/LocalMemoryAnalysis.h"
#include "PTO/Transforms/ProtocolSync/LoopFrontierRepair.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

bool checkLoopFrontierInterleavings(const StructuredSyncIR& schedule, unsigned trips);

namespace {

constexpr StringLiteral kPrelude = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @loop_oracle(%n: index, %condition: i1,
                        %in: !pto.partition_tensor_view<16x16xf16>,
                        %out: !pto.partition_tensor_view<16x16xf16>)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %c0 = arith.constant 0 : i64
    %c256 = arith.constant 256 : i64
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %a = pto.alloc_tile addr = %c0 : !pto.tile_buf<vec, 16x16xf16>
    %b = pto.alloc_tile addr = %c256 : !pto.tile_buf<vec, 16x16xf16>
)mlir";
constexpr StringLiteral kLoop = "scf.for %i = %zero to %n step %one {\n";

std::string effect(unsigned mode, StringRef buffer)
{
    const std::string operand = buffer.str() + " : !pto.tile_buf<vec, 16x16xf16>";
    if (mode == 0) {
        return "pto.tload ins(%in : !pto.partition_tensor_view<16x16xf16>) outs(" + operand + ")\n";
    }
    if (mode == 1) {
        return "pto.tstore ins(" + operand + ") outs(%out : !pto.partition_tensor_view<16x16xf16>)\n";
    }
    return "pto.tabs ins(" + operand + ") outs(" + operand + ")\n";
}

std::string program(unsigned encoding)
{
    std::string text = kPrelude.str() + effect(encoding % 3, "%a") + kLoop.str();
    encoding /= 3;
    for (StringRef buffer : {StringRef("%a"), StringRef("%b"), StringRef("%a")}) {
        text += effect(encoding % 3, buffer);
        encoding /= 3;
    }
    return text + "}\n" + effect(1, "%a") + effect(0, "%b") + "return\n}\n}\n";
}

bool check(bool condition, StringRef message)
{
    if (!condition) {
        llvm::errs() << "FAIL: " << message << '\n';
    }
    return condition;
}

struct Fixture {
    OwningOpRef<ModuleOp> module;
    func::FuncOp function;
    StructuredSyncIR schedule;
    LegacySyncSnapshot snapshot;
    SyncSemanticContext context;

    explicit Fixture(OwningOpRef<ModuleOp> module)
        : module(std::move(module)), function(*this->module->getOps<func::FuncOp>().begin()), schedule(function)
    {}

    bool build()
    {
        LegacySyncIRAdapter adapter;
        if (failed(adapter.buildSnapshot(function, snapshot))) {
            return false;
        }
        context = adapter.buildSemanticContext(snapshot);
        return succeeded(StructuredSyncIRBuilder(context).build(function, schedule));
    }
};

struct Instance {
    const SyncPhase* phase = nullptr;
    bool body = false;
    unsigned iteration = 0;
};

SmallVector<Instance> expandTrace(const Fixture& fixture, unsigned trips)
{
    SmallVector<Instance> trace;
    func::FuncOp function = fixture.function;
    const auto append = [&](Operation& operation, bool body, unsigned iteration) {
        for (const SyncPhase& phase : fixture.schedule.getPhases()) {
            if (phase.operation == &operation) {
                trace.push_back({&phase, body, iteration});
            }
        }
    };
    for (Operation& operation : function.getBody().front()) {
        if (auto loop = dyn_cast<scf::ForOp>(&operation)) {
            for (unsigned iteration = 0; iteration < trips; ++iteration) {
                for (Operation& nested : *loop.getBody()) {
                    append(nested, true, iteration);
                }
            }
        } else {
            append(operation, false, 0);
        }
    }
    return trace;
}

bool matchesRelation(const Instance& source, const Instance& target, SyncIterationRelationKind kind, unsigned trips)
{
    switch (kind) {
        case SyncIterationRelationKind::SameIteration:
            return source.body == target.body && source.iteration == target.iteration;
        case SyncIterationRelationKind::LoopCarried:
            return source.body && target.body && target.iteration == source.iteration + 1;
        case SyncIterationRelationKind::LoopEntry:
            return !source.body && target.body;
        case SyncIterationRelationKind::LoopExit:
            return source.body && !target.body;
        case SyncIterationRelationKind::LoopBypass:
            return trips == 0 && !source.body && !target.body;
        default:
            return false;
    }
}

bool overlaps(const SyncAccess& first, const SyncAccess& second)
{
    // The fixture names encode independent byte sets A=[0,512), B=[256,768).
    // Read the fixture constants, not recoverLocalAccessRegion or its atoms.
    const auto begin = [](const SyncAccess& access) {
        auto allocation = access.value.getDefiningOp<AllocTileOp>();
        auto address = allocation.getAddr().getDefiningOp<arith::ConstantOp>();
        return cast<IntegerAttr>(address.getValue()).getInt();
    };
    const std::int64_t a = begin(first);
    const std::int64_t b = begin(second);
    return a < b + 512 && b < a + 512;
}

bool oracle(
    const Fixture& fixture, const SyncLocalMemoryAnalysis& analysis, unsigned trips,
    std::optional<SyncIterationRelationKind> removed = std::nullopt)
{
    const SmallVector<Instance> trace = expandTrace(fixture, trips);
    SmallVector<llvm::BitVector> reach(trace.size(), llvm::BitVector(trace.size()));
    for (const SyncResidualObligation& requirement : analysis.requirements) {
        if (removed == requirement.iteration.kind) {
            continue;
        }
        for (unsigned source = 0; source < trace.size(); ++source) {
            for (unsigned target = source + 1; target < trace.size(); ++target) {
                const bool endpoints =
                    trace[source].phase->id == requirement.source && trace[target].phase->id == requirement.target;
                if (endpoints && matchesRelation(trace[source], trace[target], requirement.iteration.kind, trips)) {
                    reach[source].set(target);
                }
            }
        }
    }
    for (unsigned source = trace.size(); source > 0; --source) {
        for (int next : reach[source - 1].set_bits()) {
            reach[source - 1] |= reach[next];
        }
    }
    for (unsigned source = 0; source < trace.size(); ++source) {
        for (unsigned target = source + 1; target < trace.size(); ++target) {
            for (SyncAccessId a : trace[source].phase->accesses) {
                const SyncAccess& first = *fixture.schedule.findAccess(a);
                if (first.storage.space != AddressSpace::VEC) {
                    continue;
                }
                for (SyncAccessId b : trace[target].phase->accesses) {
                    const SyncAccess& second = *fixture.schedule.findAccess(b);
                    const bool localHazard =
                        second.storage.space == AddressSpace::VEC &&
                        (first.mode != SyncAccessMode::Read || second.mode != SyncAccessMode::Read);
                    const bool uncovered = localHazard && overlaps(first, second) && !reach[source].test(target);
                    if (uncovered) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

bool testOracle(MLIRContext& context)
{
    for (unsigned encoding = 0; encoding < 81; ++encoding) {
        auto module = parseSourceString<ModuleOp>(program(encoding), &context);
        if (!module || failed(verify(*module))) {
            return false;
        }
        Fixture fixture(std::move(module));
        if (!fixture.build()) {
            return false;
        }
        SyncLocalFlowOptions options;
        options.analyzeSingleLoop = true;
        auto analysis = analyzeLocalMemory(fixture.schedule, options);
        const bool complete = succeeded(analysis) && analysis->loopStatus == SyncLocalLoopStatus::Complete &&
                              analysis->coveredAccesses.none() && !analysis->boundary.empty();
        if (!check(complete, "single-loop diagnostic must complete without admitting emission")) {
            return false;
        }
        for (const SyncResidualObligation& requirement : analysis->requirements) {
            const bool carried = requirement.iteration.kind == SyncIterationRelationKind::LoopCarried;
            const bool boundary = requirement.iteration.kind != SyncIterationRelationKind::SameIteration;
            if (!check(
                    (!carried || requirement.iteration.distance == 1) &&
                        (!boundary || requirement.iteration.carrier == analysis->loopCarrier),
                    "dynamic relations must retain distance and carrier")) {
                return false;
            }
        }
        for (unsigned trips : {0, 1, 2, 3, 4, 7, 8, 11}) {
            if (!oracle(fixture, *analysis, trips)) {
                llvm::errs() << "FAIL: uncovered dynamic hazard encoding=" << encoding << " trips=" << trips << '\n';
                return false;
            }
        }
    }
    llvm::outs() << "protocol-sync loop occurrence oracle: 81 programs, 648 traces pass\n";
    return true;
}

bool testMutationsAndLimits(MLIRContext& context)
{
    auto module = parseSourceString<ModuleOp>(program(3), &context);
    if (!module) {
        return false;
    }
    Fixture fixture(std::move(module));
    if (!fixture.build()) {
        return false;
    }
    SyncLocalFlowOptions options;
    options.analyzeSingleLoop = true;
    auto analysis = analyzeLocalMemory(fixture.schedule, options);
    if (failed(analysis)) {
        return false;
    }
    for (auto [kind, trips] :
         {std::pair{SyncIterationRelationKind::LoopEntry, 1u}, std::pair{SyncIterationRelationKind::LoopExit, 1u},
          std::pair{SyncIterationRelationKind::LoopCarried, 3u},
          std::pair{SyncIterationRelationKind::LoopBypass, 0u}}) {
        if (!check(!oracle(fixture, *analysis, trips, kind), "removing a loop boundary/backedge must lose coverage")) {
            return false;
        }
    }
    options.maximumExpandedEntries = 0;
    auto limited = analyzeLocalMemory(fixture.schedule, options);
    if (!check(
            succeeded(limited) && limited->loopStatus == SyncLocalLoopStatus::LimitExceeded &&
                limited->requirements.empty() && limited->coveredAccesses.none(),
            "loop analysis limit must not expose partial coverage")) {
        return false;
    }
    auto production = analyzeLocalMemory(fixture.schedule);
    if (!check(
            succeeded(production) && production->loopStatus == SyncLocalLoopStatus::NotRequested &&
                production->coveredAccesses.none(),
            "production loop protection must remain unchanged")) {
        return false;
    }
    llvm::outs() << "protocol-sync loop boundary mutations and limits: pass\n";
    return true;
}

bool testSelfPhaseAndBoundaries(MLIRContext& context)
{
    const std::string bodies[] = {
        effect(0, "%a"),
        "scf.if %condition {\n" + effect(0, "%a") + "}\n",
        "scf.for %j = %zero to %n step %one {\n" + effect(0, "%a") + "}\n",
        "%addr = arith.index_cast %i : index to i64\n"
        "%dynamic = pto.alloc_tile addr = %addr : !pto.tile_buf<vec, 16x16xf16>\n" +
            effect(0, "%dynamic"),
    };
    for (auto [index, body] : llvm::enumerate(bodies)) {
        auto module = parseSourceString<ModuleOp>(kPrelude.str() + kLoop.str() + body + "}\nreturn\n}\n}\n", &context);
        if (!module) {
            return false;
        }
        Fixture fixture(std::move(module));
        if (!fixture.build()) {
            return false;
        }
        SyncLocalFlowOptions options;
        options.analyzeSingleLoop = true;
        auto analysis = analyzeLocalMemory(fixture.schedule, options);
        if (failed(analysis)) {
            return false;
        }
        if (index != 0) {
            if (!check(
                    analysis->loopStatus == SyncLocalLoopStatus::Unsupported && analysis->requirements.empty() &&
                        analysis->coveredAccesses.none(),
                    "choice, nesting and dynamic range remain unsupported")) {
                return false;
            }
            continue;
        }
        const bool selfBackedge = llvm::any_of(analysis->requirements, [](const SyncResidualObligation& requirement) {
            return requirement.source == requirement.target && requirement.sourceAccess == requirement.targetAccess &&
                   requirement.iteration.kind == SyncIterationRelationKind::LoopCarried &&
                   requirement.iteration.distance == 1;
        });
        if (!check(
                selfBackedge && oracle(fixture, *analysis, 11) &&
                    !oracle(fixture, *analysis, 2, SyncIterationRelationKind::LoopCarried),
                "same static phase must retain a next-iteration WAW obligation")) {
            return false;
        }
    }
    llvm::outs() << "protocol-sync loop self-phase and unsupported boundaries: pass\n";
    return true;
}

OwningOpRef<ModuleOp> repairLoop(MLIRContext& context, StringRef body)
{
    auto module =
        parseSourceString<ModuleOp>(kPrelude.str() + kLoop.str() + body.str() + "}\nreturn\n}\n}\n", &context);
    if (!module || failed(verify(*module))) {
        return {};
    }
    Fixture fixture(std::move(module));
    if (!fixture.build()) {
        return {};
    }
    auto plan = buildLoopFrontierRepairPlan(fixture.schedule);
    const bool ready = succeeded(plan) && plan->status == SyncLoopFrontierStatus::Ready;
    if (!ready) {
        return {};
    }
    IRMapping mapping;
    OwningOpRef<ModuleOp> clone = cast<ModuleOp>(fixture.module->getOperation()->clone(mapping));
    auto function = cast<func::FuncOp>(mapping.lookup(fixture.function.getOperation()));
    const bool emitted =
        succeeded(materializeLoopFrontierRepair(function, mapping, *plan)) && succeeded(verify(*clone));
    if (!emitted) {
        return {};
    }
    return clone;
}

bool concreteCycle(OwningOpRef<ModuleOp> module)
{
    if (!module || failed(verify(*module))) {
        return false;
    }
    Fixture fixture(std::move(module));
    return fixture.build() && succeeded(verifyConcreteLoopFrontierRepair(fixture.schedule));
}

bool interleavings(OwningOpRef<ModuleOp> module)
{
    Fixture fixture(std::move(module));
    if (!fixture.build()) {
        return false;
    }
    for (unsigned trips : {0, 1, 2, 3, 4, 7, 8, 11}) {
        if (!checkLoopFrontierInterleavings(fixture.schedule, trips)) {
            llvm::errs() << "FAIL: unsafe FIFO interleaving or oracle limit at trips=" << trips << '\n';
            return false;
        }
    }
    return true;
}

SmallVector<Operation*> syncOperations(ModuleOp module)
{
    SmallVector<Operation*> operations;
    module.walk([&](Operation* operation) {
        if (isa<SetFlagOp, WaitFlagOp, BarrierOp>(operation)) {
            operations.push_back(operation);
        }
    });
    return operations;
}

bool testConcreteCycles(MLIRContext& context)
{
    for (unsigned encoding = 0; encoding < 27; ++encoding) {
        const std::string body =
            effect(encoding % 3, "%a") + effect((encoding / 3) % 3, "%b") + effect((encoding / 9) % 3, "%a");
        auto module = repairLoop(context, body);
        if (!check(module && concreteCycle(cast<ModuleOp>(module->clone())), "concrete phase cycle must verify")) {
            return false;
        }
        if (!check(interleavings(cast<ModuleOp>(module->clone())), "all enabled FIFO interleavings must be safe")) {
            return false;
        }
        const std::size_t count = syncOperations(*module).size();
        for (std::size_t index = 0; index < count; ++index) {
            OwningOpRef<ModuleOp> mutated = cast<ModuleOp>(module->clone());
            syncOperations(*mutated)[index]->erase();
            if (!check(!concreteCycle(std::move(mutated)), "deleting any cycle action must reject")) {
                return false;
            }
        }
    }
    if (!check(concreteCycle(repairLoop(context, effect(0, "%a"))), "single-pipe self recurrence must verify")) {
        return false;
    }
    llvm::outs() << "protocol-sync concrete loop cycles: 27 topologies and action deletions pass\n";
    return true;
}

bool testCyclePlacementAndResources(MLIRContext& context)
{
    const std::string body = effect(0, "%a") + effect(2, "%a") + effect(0, "%b") + effect(2, "%b") + effect(1, "%a");
    auto module = repairLoop(context, body);
    if (!module || !concreteCycle(cast<ModuleOp>(module->clone()))) {
        return false;
    }
    for (unsigned mutation = 0; mutation < 5; ++mutation) {
        OwningOpRef<ModuleOp> mutated = cast<ModuleOp>(module->clone());
        auto function = *mutated->getOps<func::FuncOp>().begin();
        auto loop = *function.getOps<scf::ForOp>().begin();
        auto set = *loop.getOps<SetFlagOp>().begin();
        auto wait = *loop.getOps<WaitFlagOp>().begin();
        if (mutation == 0) {
            set->moveBefore(set->getPrevNode());
        } else if (mutation == 1) {
            wait->moveAfter(wait->getNextNode());
        } else if (mutation == 2) {
            set.setEventIdAttr(EventAttr::get(&context, EVENT::EVENT_ID7));
        } else if (mutation == 3) {
            OpBuilder builder(set);
            auto choice = builder.create<scf::IfOp>(set.getLoc(), function.getArgument(1), false);
            set->moveBefore(choice.thenBlock()->getTerminator());
        } else {
            OpBuilder builder(set);
            builder.setInsertionPointAfter(set);
            builder.insert(set->clone());
            Fixture witness(cast<ModuleOp>(mutated->clone()));
            if (!check(
                    witness.build() && !checkLoopFrontierInterleavings(witness.schedule, 3),
                    "independent execution must detect a live-key rearm")) {
                return false;
            }
        }
        if (!check(!concreteCycle(std::move(mutated)), "moved, mismatched or guarded cycle action must reject")) {
            return false;
        }
    }
    auto original = parseSourceString<ModuleOp>(kPrelude.str() + kLoop.str() + body + "}\nreturn\n}\n}\n", &context);
    if (!original) {
        return false;
    }
    Fixture fixture(std::move(original));
    if (!fixture.build()) {
        return false;
    }
    SyncEventReservation reservation{PIPE::PIPE_MTE2, PIPE::PIPE_V, {0, 2, 4}};
    auto holes = buildLoopFrontierRepairPlan(fixture.schedule, {reservation});
    if (!check(
            succeeded(holes) && holes->status == SyncLoopFrontierStatus::Ready, "non-contiguous IDs must allocate")) {
        return false;
    }
    for (const auto& edge : holes->edges) {
        if (edge.sourcePipe == reservation.source && edge.targetPipe == reservation.target &&
            !check(
                edge.eventId && !llvm::is_contained(reservation.eventIds, *edge.eventId),
                "reserved ID was allocated")) {
            return false;
        }
    }
    const auto target = ProtocolSyncTarget::resolve(fixture.function);
    reservation.eventIds.assign(target.getCompilerEventIds().begin(), target.getCompilerEventIds().end());
    auto exhausted = buildLoopFrontierRepairPlan(fixture.schedule, {reservation});
    if (!check(
            succeeded(exhausted) && exhausted->status == SyncLoopFrontierStatus::ResourceInfeasible &&
                exhausted->edges.empty(),
            "event exhaustion must not expose a partial recipe")) {
        return false;
    }
    auto prefixModule = parseSourceString<ModuleOp>(program(3), &context);
    if (!prefixModule) {
        return false;
    }
    Fixture prefix(std::move(prefixModule));
    if (!prefix.build()) {
        return false;
    }
    auto unsupported = buildLoopFrontierRepairPlan(prefix.schedule);
    if (!check(
            succeeded(unsupported) && unsupported->status == SyncLoopFrontierStatus::Unsupported &&
                unsupported->edges.empty(),
            "prefix effects need a separate boundary composition proof")) {
        return false;
    }
    llvm::outs() << "protocol-sync loop placement mutations and event reservations: pass\n";
    return true;
}

} // namespace

int main()
{
    DialectRegistry registry;
    registry.insert<PTODialect, arith::ArithDialect, func::FuncDialect, scf::SCFDialect>();
    MLIRContext context(registry);
    context.disableMultithreading();
    return testOracle(context) && testMutationsAndLimits(context) && testSelfPhaseAndBoundaries(context) &&
                   testConcreteCycles(context) && testCyclePlacementAndResources(context) ?
               0 :
               1;
}
