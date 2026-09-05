// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- ProtocolSyncStructuredTest.cpp - Independent paths and occurrences ---===//
#include "PTO/Transforms/InsertSync/LegacySyncIRAdapter.h"
#include "PTO/Transforms/ProtocolSync/MixedProtocolPlan.h"
#include "PTO/Transforms/ProtocolSync/ConcreteSyncVerifier.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <string>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

bool checkStructuredFrontierInterleavings(const StructuredSyncIR&, unsigned, std::uint64_t, bool* = nullptr);

namespace {
const std::string kLoad =
    "pto.tload ins(%in : !pto.partition_tensor_view<16x16xf16>) outs(%a : !pto.tile_buf<vec, 16x16xf16>)\n";
const std::string kCompute =
    "pto.tabs ins(%a : !pto.tile_buf<vec, 16x16xf16>) outs(%b : !pto.tile_buf<vec, 16x16xf16>)\n";
const std::string kStore =
    "pto.tstore ins(%b : !pto.tile_buf<vec, 16x16xf16>) outs(%out : !pto.partition_tensor_view<16x16xf16>)\n";
const std::string kPrelude = R"mlir(
module attributes {pto.target_arch = "a3"} {
 func.func @structured(%n: index, %condition: i1,
   %in: !pto.partition_tensor_view<16x16xf16>, %out: !pto.partition_tensor_view<16x16xf16>)
   attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
   %c0 = arith.constant 0 : i64
   %c256 = arith.constant 256 : i64
   %zero = arith.constant 0 : index
   %one = arith.constant 1 : index
   %a = pto.alloc_tile addr = %c0 : !pto.tile_buf<vec, 16x16xf16>
   %b = pto.alloc_tile addr = %c256 : !pto.tile_buf<vec, 16x16xf16>
)mlir";

struct Fixture {
    OwningOpRef<ModuleOp> module;
    func::FuncOp function;
    StructuredSyncIR schedule;
    LegacySyncSnapshot snapshot;
    SyncSemanticContext semantics;
    explicit Fixture(OwningOpRef<ModuleOp> input)
        : module(std::move(input)), function(*module->getOps<func::FuncOp>().begin()), schedule(function)
    {}
    bool build()
    {
        LegacySyncIRAdapter adapter;
        if (failed(adapter.buildSnapshot(function, snapshot))) {
            return false;
        }
        semantics = adapter.buildSemanticContext(snapshot);
        return succeeded(StructuredSyncIRBuilder(semantics).build(function, schedule)) &&
               schedule.getFailures().empty();
    }
};
struct Instance {
    SyncPhaseId phase;
    bool body;
    unsigned iteration;
};
bool relationMatches(const Instance& a, const Instance& b, const SyncIterationRelation& relation)
{
    switch (relation.kind) {
        case SyncIterationRelationKind::SameIteration:
            return a.body == b.body && a.iteration == b.iteration;
        case SyncIterationRelationKind::LoopEntry:
            return !a.body && b.body;
        case SyncIterationRelationKind::LoopExit:
            return a.body && !b.body;
        case SyncIterationRelationKind::LoopCarriedAny:
            return a.body && b.body && a.iteration < b.iteration;
        default:
            return false;
    }
}
SmallVector<Instance, 32> trace(const StructuredSyncIR& schedule, unsigned trips, std::uint64_t choices, bool& valid)
{
    valid = true;
    SmallVector<Instance, 32> result;
    unsigned decision = 0;
    const auto visit = [&](const auto& self, Block& block, bool body, unsigned iteration) -> void {
        for (Operation& op : block) {
            if (!valid) {
                return;
            }
            if (auto choice = dyn_cast<scf::IfOp>(op)) {
                if (decision >= 64) {
                    valid = false;
                    return;
                }
                const bool takeThen = (choices & (std::uint64_t{1} << decision++)) != 0;
                Region& arm = takeThen ? choice.getThenRegion() : choice.getElseRegion();
                if (!arm.empty()) {
                    self(self, arm.front(), body, iteration);
                }
            } else if (auto loop = dyn_cast<scf::ForOp>(op)) {
                for (unsigned i = 0; i < trips; ++i) {
                    self(self, *loop.getBody(), true, i);
                }
            } else {
                for (const auto& phase : schedule.getPhases()) {
                    if (phase.operation == &op) {
                        result.push_back({phase.id, body, iteration});
                    }
                }
            }
        }
    };
    visit(visit, schedule.getFunction().getBody().front(), false, 0);
    return result;
}
bool requirementsCover(
    const StructuredSyncIR& schedule, const SyncLocalMemoryAnalysis& local, unsigned trips, std::uint64_t choices)
{
    bool valid = false;
    const auto instances = trace(schedule, trips, choices, valid);
    if (!valid) {
        return false;
    }
    for (unsigned i = 0; i < instances.size(); ++i) {
        const auto& a = instances[i];
        for (unsigned j = i + 1; j < instances.size(); ++j) {
            const auto& b = instances[j];
            for (auto x : schedule.findPhase(a.phase)->accesses) {
                const auto* source = schedule.findAccess(x);
                if (source->storage.space != AddressSpace::VEC) {
                    continue;
                }
                for (auto y : schedule.findPhase(b.phase)->accesses) {
                    const auto* target = schedule.findAccess(y);
                    if (target->storage.space != AddressSpace::VEC ||
                        (source->mode == SyncAccessMode::Read && target->mode == SyncAccessMode::Read)) {
                        continue;
                    }
                    // Fixture allocations [0,512) and [256,768) always overlap.
                    const bool covered = llvm::any_of(local.requirements, [&](const auto& obligation) {
                        return obligation.sourceAccess == x && obligation.targetAccess == y &&
                               !obligation.atoms.empty() && relationMatches(a, b, obligation.iteration);
                    });
                    if (!covered) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}
bool runPaths(const Fixture& original, const SyncLocalMemoryAnalysis& local, Fixture& emitted, bool recurring)
{
    for (unsigned trips : {0, 1, 2, 3, 4, 7, 8, 11}) {
        const unsigned paths = recurring && trips <= 3 ? 1u << trips : 2;
        for (unsigned path = 0; path < paths + 2; ++path) {
            const std::uint64_t choices = path < paths  ? path :
                                          path == paths ? 0xaaaaaaaaaaaaaaaaULL :
                                                          0xffffffffffffffffULL;
            const bool invalidPath = !requirementsCover(original.schedule, local, trips, choices) ||
                                     !checkStructuredFrontierInterleavings(emitted.schedule, trips, choices);
            if (invalidPath) {
                llvm::errs() << "FAIL structured path trips=" << trips << " choices=" << choices << '\n';
                return false;
            }
        }
    }
    return true;
}
bool rejectMutations(const Fixture& emitted)
{
    // Distinguish a concrete unsafe execution from oracle budget exhaustion.
    for (unsigned mutationKind = 0; mutationKind < 4; ++mutationKind) {
        OwningOpRef<ModuleOp> mutation = cast<ModuleOp>(emitted.module.get()->clone());
        auto function = *mutation->getOps<func::FuncOp>().begin();
        WaitFlagOp selected;
        function.walk([&](WaitFlagOp wait) {
            const bool reverse = wait.getDstPipe().getPipe() == PIPE::PIPE_V;
            const bool branch = wait->getParentOfType<scf::IfOp>() != nullptr;
            const bool matchesMutation = !selected && (mutationKind == 0 || reverse) && (mutationKind < 2 || branch);
            if (matchesMutation) {
                selected = wait;
            }
        });
        if (!selected) {
            continue; // The vector-only arm has no branch-local event.
        }
        if (mutationKind == 3) {
            // Move the acquisition onto the untaken path: it cannot consume
            // the signal emitted by the other arm.
            auto branch = selected->getParentOfType<scf::IfOp>();
            if (branch.getElseRegion().empty()) {
                return false;
            }
            selected->moveBefore(branch.getElseRegion().front().getTerminator());
        } else if (mutationKind == 2) {
            // A common acquisition is invalid when the signal can be skipped.
            selected->moveAfter(selected->getParentOfType<scf::IfOp>());
        } else {
            selected->erase();
        }
        Fixture unsafe(std::move(mutation));
        const bool acceptedMutation = !unsafe.build() || succeeded(verifyFreshConcreteSyncSemantics(unsafe.function));
        if (acceptedMutation) {
            return false;
        }
        bool witness = false;
        const std::uint64_t choices = mutationKind == 2 ? 0 : 0xffffffffffffffffULL;
        const bool safe = checkStructuredFrontierInterleavings(unsafe.schedule, 3, choices, &witness);
        if (safe || !witness) {
            llvm::errs() << "FAIL structured mutation=" << mutationKind << " witness=" << witness << '\n';
            return false;
        }
    }
    return true;
}
bool runCase(MLIRContext& context, unsigned shape, StringRef alias)
{
    const bool recurring = shape >= 3;
    const std::string thenArm = shape % 3 == 0 ? kCompute : kStore;
    const std::string elseArm = shape % 3 == 2 ? kCompute : "";
    const std::string choice = "scf.if %condition {\n" + thenArm + "} else {\n" + elseArm + "}\n";
    const std::string middle =
        recurring ? "scf.for %i = %zero to %n step %one {\n" + choice + kCompute + "}\n" : choice;
    auto module =
        parseSourceString<ModuleOp>(kPrelude + kLoad + middle + kCompute + kStore + "return\n}\n}\n", &context);
    if (!module) {
        return false;
    }
    Fixture fixture(std::move(module));
    fixture.function->setAttr("pto.gm_alias", StringAttr::get(&context, alias));
    if (!fixture.build()) {
        return false;
    }
    SyncLocalFlowOptions options;
    options.analyzeStructured = true;
    auto local = analyzeLocalMemory(fixture.schedule, options);
    const bool invalidAnalysis = failed(local) || local->structuredStatus != SyncLocalStructuredStatus::Complete ||
                                 local->coveredAccesses.any() || local->regionSummaries.empty();
    if (invalidAnalysis) {
        llvm::errs() << "FAIL structured analysis\n";
        return false;
    }
    for (const auto& state : local->states) {
        if (!state.mayHaveLiveIn || !state.mustDefinitions.empty()) {
            return false;
        }
    }
    options.maximumExpandedEntries = 0;
    auto limited = analyzeLocalMemory(fixture.schedule, options);
    const bool invalidLimit = failed(limited) ||
                              limited->structuredStatus != SyncLocalStructuredStatus::LimitExceeded ||
                              !limited->requirements.empty() || !limited->states.empty();
    if (invalidLimit) {
        return false;
    }
    auto stages = analyzePipelineStages(fixture.schedule);
    if (failed(stages)) {
        return false;
    }
    auto timelines = analyzeStorageTimelines(fixture.schedule, *stages);
    auto channels = analyzeChannels(fixture.schedule, *stages, timelines);
    auto plan = buildMixedProtocolPlan(fixture.schedule, *stages, timelines, channels, false);
    const bool invalidPlan = failed(plan) || !plan->structuredFrontier ||
                             failed(verifyMixedProtocolPlan(fixture.schedule, *stages, timelines, channels, *plan));
    if (invalidPlan) {
        llvm::errs() << "FAIL structured native shape=" << shape << " alias=" << alias << '\n';
        if (succeeded(plan)) {
            printMixedProtocolPlan(fixture.function, *plan, llvm::errs());
        }
        return false;
    }
    auto bad = *plan;
    bad.structuredFrontier->requirements.front().atoms.clear();
    if (succeeded(verifyMixedProtocolPlan(fixture.schedule, *stages, timelines, channels, bad))) {
        return false;
    }
    bad = *plan;
    bad.selectedWorld.acknowledgedPhases.pop_back();
    if (succeeded(verifyMixedProtocolPlan(fixture.schedule, *stages, timelines, channels, bad))) {
        return false;
    }
    IRMapping mapping;
    OwningOpRef<ModuleOp> clone = cast<ModuleOp>(fixture.module->getOperation()->clone(mapping));
    auto function = cast<func::FuncOp>(mapping.lookup(fixture.function.getOperation()));
    if (failed(materializeStructuredFrontier(function, mapping, *plan->structuredFrontier))) {
        llvm::errs() << "FAIL structured materialization\n";
        return false;
    }
    Fixture emitted(std::move(clone));
    if (!emitted.build()) {
        return false;
    }
    StringRef failedStage;
    if (failed(verifyConcreteSyncSemantics(emitted.semantics, emitted.function, nullptr, &failedStage))) {
        llvm::errs() << "FAIL structured verification stage=" << failedStage << '\n';
        emitted.function.print(llvm::errs());
        return false;
    }
    if (!runPaths(fixture, *local, emitted, recurring)) {
        return false;
    }
    return rejectMutations(emitted);
}
bool rejectPublication(MLIRContext& context, StringRef alias, bool recurring)
{
    const std::string reload =
        "pto.tload ins(%out : !pto.partition_tensor_view<16x16xf16>) outs(%a : !pto.tile_buf<vec, 16x16xf16>)\n";
    const std::string choice = "scf.if %condition {\n" + kStore + reload + "}\n";
    const std::string middle = recurring ? "scf.for %i = %zero to %n step %one {\n" + choice + "}\n" : choice;
    auto module = parseSourceString<ModuleOp>(kPrelude + kLoad + kCompute + middle + "return\n}\n}\n", &context);
    if (!module) {
        return false;
    }
    Fixture fixture(std::move(module));
    fixture.function->setAttr("pto.gm_alias", StringAttr::get(&context, alias));
    if (!fixture.build()) {
        return false;
    }
    auto stages = analyzePipelineStages(fixture.schedule);
    if (failed(stages)) {
        return false;
    }
    auto timelines = analyzeStorageTimelines(fixture.schedule, *stages);
    auto channels = analyzeChannels(fixture.schedule, *stages, timelines);
    auto plan = buildMixedProtocolPlan(fixture.schedule, *stages, timelines, channels, false);
    const bool acceptedPublication = failed(plan) || plan->isComplete();
    if (acceptedPublication) {
        return false;
    }
    // Even a structurally balanced emitted recipe cannot manufacture GM
    // publication. Verify fresh IR, not just the planner's rejection.
    auto recipe = buildStructuredFrontierPlan(fixture.schedule);
    const bool missingRecipe = failed(recipe) || !*recipe;
    if (missingRecipe) {
        return false;
    }
    IRMapping mapping;
    OwningOpRef<ModuleOp> clone = cast<ModuleOp>(fixture.module->getOperation()->clone(mapping));
    auto function = cast<func::FuncOp>(mapping.lookup(fixture.function.getOperation()));
    return succeeded(materializeStructuredFrontier(function, mapping, **recipe)) &&
           failed(verifyFreshConcreteSyncSemantics(function));
}
} // namespace

bool runStructuredFrontierTests(MLIRContext& context)
{
    for (unsigned shape = 0; shape < 6; ++shape) {
        for (StringRef alias : {"may-alias", "assume-disjoint-arguments"}) {
            if (!runCase(context, shape, alias)) {
                llvm::errs() << "FAIL structured fixture shape=" << shape << " alias=" << alias << '\n';
                return false;
            }
        }
    }
    for (StringRef alias : {"may-alias", "assume-disjoint-arguments"}) {
        const bool invalidPublicationCheck =
            !rejectPublication(context, alias, false) || !rejectPublication(context, alias, true);
        if (invalidPublicationCheck) {
            llvm::errs() << "FAIL structured GM publication negative\n";
            return false;
        }
    }
    llvm::outs() << "protocol-sync structured choices and loops: 12 native worlds, paths and mutations pass\n";
    return true;
}
