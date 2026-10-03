// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Read-only inspection of shared phases, storage facts and MLIR control queries.
// Successful analysis does not establish certified synchronization emission.
#include "PTO/Transforms/FrontierSynch/StorageAnalysis.h"
#include "PTO/Transforms/FrontierSynch/PhaseIndex.h"
#include "PTO/Transforms/FrontierSynch/DemandAnalysis.h"
#include "PTO/Transforms/InsertSync/SyncInput.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include "frontier-guarded-dump.h"
#include <string>

int runAdjacentLocalUpperChecks(mlir::MLIRContext& context);
int runGeneralCompositionChecks(mlir::MLIRContext& context);
int runPeriodicNestedQueryChecks(mlir::MLIRContext& context);
int runSyncTargetPreflightChecks(llvm::StringRef path, mlir::MLIRContext& context);
int runGMAliasChecks(llvm::StringRef path, mlir::MLIRContext& context);
int runOneWayRepairChecks(llvm::StringRef path, mlir::MLIRContext& context);
int runOneWayPhysicalChecks(llvm::StringRef path, mlir::MLIRContext& context);
int runSymbolicAllocationChecks(mlir::MLIRContext& context);
int runStructuredCounterChecks(mlir::MLIRContext& context);
int runPeriodicAllocationChecks(llvm::StringRef path, mlir::MLIRContext& context);
int runSingleStreamChecks(llvm::StringRef path, mlir::MLIRContext& context);
int runFinitePhysicalChecks(llvm::StringRef path, mlir::MLIRContext& context);
int runGeneralCountedChecks(llvm::StringRef path, mlir::MLIRContext& context);
int runCountedReaderChecks(llvm::StringRef path, mlir::MLIRContext& context);
int runFixedBodyUpperChecks(llvm::StringRef path, mlir::MLIRContext& context);
int runStationaryCellChecks(llvm::StringRef path, mlir::MLIRContext& context);
int runRegionalRequestChecks(llvm::StringRef path, mlir::MLIRContext& context);
int runAtomicStoreChecks(llvm::StringRef path, mlir::MLIRContext& context);
int runStructuredJSON(llvm::StringRef path, mlir::MLIRContext& context);
namespace {
namespace fs = mlir::pto::frontiersynch;
enum class Mode { Inspect, Storage, Structure, Demands, Guarded, Verify, ExpectFailure };
std::string printIR(mlir::Operation* operation)
{
    std::string result;
    llvm::raw_string_ostream stream(result);
    operation->print(stream);
    return result;
}

bool checkBoundPartition(const fs::StorageAnalysis& storage)
{
    const auto& shared = storage.bounds();
    for (const auto& memory : shared.memory()) {
        if (memory.bounds != mlir::pto::SyncBoundStatus::ConstantPhysical) {
            if (!memory.boundCells.empty()) { return false; }
            continue;
        }
        if (memory.boundingAddresses.size() != 1 || memory.boundCells.empty()) { return false; }
        auto begin = memory.boundingAddresses.front();
        auto end = begin + memory.boundingSize;
        auto cursor = begin;
        for (auto id : memory.boundCells) {
            if (id >= shared.cells().size()) { return false; }
            const auto& cell = shared.cells()[id];
            if (cell.addressSpace != memory.addressSpace || cell.begin != cursor || cell.end <= cell.begin ||
                cell.end > end) { return false; }
            cursor = cell.end;
        }
        if (cursor != end) { return false; }
        // Independent point-membership oracle at all emitted boundaries: no
        // byte enumeration and no use of the production sweep state.
        for (auto [id, cell] : llvm::enumerate(shared.cells())) {
            if (cell.addressSpace != memory.addressSpace) { continue; }
            bool expected = begin <= cell.begin && cell.end <= end;
            if (llvm::is_contained(memory.boundCells, id) != expected) { return false; }
        }
    }
    return true;
}

void dumpStorage(const fs::StorageAnalysis& storage)
{
    for (auto [id, footprint] : llvm::enumerate(storage.footprints())) {
        const auto& memory = *footprint.memory;
        llvm::outs() << "  footprint=" << id << " bytes=" << memory.allocateSize
                     << " physical=" << memory.hasKnownPhysicalAddresses
                     << " unknown-range=" << memory.aliasesUnknownRange << " addresses=[";
        llvm::interleaveComma(memory.baseAddresses, llvm::outs());
        llvm::outs() << "]\n";
        for (const auto& access : footprint.accesses) {
            llvm::outs() << "  access footprint=" << id << " phase=" << access.phase->GetIndex()
                         << " read=" << access.read << " write=" << access.write << "\n";
        }
    }
    for (const auto& alias : storage.aliases()) {
        llvm::outs() << "  alias footprints=" << alias.first << ", " << alias.second << "\n";
    }
    for (auto [id, cell] : llvm::enumerate(storage.bounds().cells())) {
        llvm::outs() << "  bound-cell=" << id << " space=" << cell.addressSpace
                     << " interval=[" << cell.begin << "," << cell.end << ") coverage=may-access\n";
    }
    for (const auto& memory : storage.bounds().memory()) {
        llvm::outs() << "  bound-memory=" << memory.id << " status=" << static_cast<unsigned>(memory.bounds)
                     << " cells=[";
        llvm::interleaveComma(memory.boundCells, llvm::outs());
        llvm::outs() << "]\n";
    }
}

void dumpSequence(const fs::PhaseIndex& index, mlir::Region& region)
{
    if (!region.hasOneBlock()) {
        llvm::outs() << "structured\n";
        return;
    }
    auto sequence = index.explicitSequence(region.front());
    if (mlir::failed(sequence)) {
        llvm::outs() << "structured\n";
        return;
    }
    llvm::outs() << "[";
    llvm::interleaveComma(*sequence, llvm::outs(), [](const auto* phase) { llvm::outs() << phase->GetIndex(); });
    llvm::outs() << "]\n";
}

void dumpArguments(const fs::PhaseIndex& index, mlir::Region& region)
{
    if (region.empty() || region.front().empty()) {
        return;
    }
    mlir::Operation* entry = &region.front().front();
    for (auto argument : region.front().getArguments()) {
        llvm::outs() << "    region=" << region.getRegionNumber() << " argument=" << argument.getArgNumber()
                     << " at-entry=" << index.valueAvailable(argument, entry, fs::Boundary::Before)
                     << " before-owner=" << index.valueAvailable(argument, region.getParentOp(), fs::Boundary::Before)
                     << " after-owner=" << index.valueAvailable(argument, region.getParentOp(), fs::Boundary::After)
                     << "\n";
    }
}

void dumpControl(const fs::PhaseIndex& index, mlir::RegionBranchOpInterface control)
{
    llvm::outs() << "  control=" << control->getName() << " regions=" << control->getNumRegions();
    if (auto loop = mlir::dyn_cast<mlir::LoopLikeOpInterface>(control.getOperation())) {
        llvm::outs() << " loop-regions=" << loop.getLoopRegions().size()
                     << " carried=" << loop.getRegionIterArgs().size();
    }
    llvm::outs() << "\n";
    auto dumpSuccessors = [control](mlir::RegionBranchPoint from) mutable {
        llvm::SmallVector<mlir::RegionSuccessor> successors;
        control.getSuccessorRegions(from, successors);
        for (const auto& successor : successors) {
            llvm::outs() << "    from=";
            if (from.isParent()) {
                llvm::outs() << "entry";
            } else {
                llvm::outs() << from.getRegionOrNull()->getRegionNumber();
            }
            llvm::outs() << " to=";
            if (successor.isParent()) {
                llvm::outs() << "exit";
            } else {
                llvm::outs() << successor.getSuccessor()->getRegionNumber();
            }
            llvm::outs() << " forwarded=" << successor.getSuccessorInputs().size() << "\n";
        }
    };
    dumpSuccessors(mlir::RegionBranchPoint::parent());
    for (auto& region : control->getRegions()) {
        dumpSuccessors(mlir::RegionBranchPoint(region));
        llvm::outs() << "    region=" << region.getRegionNumber() << " sequence=";
        dumpSequence(index, region);
        dumpArguments(index, region);
    }
}

void dumpAvailability(const fs::PhaseIndex& index, mlir::func::FuncOp function)
{
    llvm::SmallVector<mlir::Operation*> anchors, definitions;
    function.walk([&](mlir::Operation* operation) {
        if (operation->hasAttr("sync.test.anchor")) {
            anchors.push_back(operation);
        }
        if (operation->getNumResults() == 1 && operation->hasAttr("sync.test.value")) {
            definitions.push_back(operation);
        }
    });
    for (auto* anchor : anchors) {
        for (auto* definition : definitions) {
            llvm::outs() << "  available value=" << definition->getAttr("sync.test.value")
                         << " anchor=" << anchor->getAttr("sync.test.anchor")
                         << " before=" << index.valueAvailable(definition->getResult(0), anchor, fs::Boundary::Before)
                         << " after=" << index.valueAvailable(definition->getResult(0), anchor, fs::Boundary::After)
                         << "\n";
        }
    }
    for (auto [position, first] : llvm::enumerate(anchors)) {
        for (auto* second : llvm::ArrayRef(anchors).drop_front(position + 1)) {
            llvm::outs() << "  sites=" << first->getAttr("sync.test.anchor") << ","
                         << second->getAttr("sync.test.anchor")
                         << " exclusive-in-common-invocation=" << mlir::insideMutuallyExclusiveRegions(first, second)
                         << "\n";
        }
    }
}

mlir::LogicalResult dumpStructure(mlir::func::FuncOp function, const mlir::pto::SyncInput& input)
{
    fs::PhaseIndex index;
    if (mlir::failed(index.build(function, input))) {
        return mlir::failure();
    }
    llvm::outs() << "  root-sequence=";
    dumpSequence(index, function.getBody());
    function.walk([&](mlir::RegionBranchOpInterface control) { dumpControl(index, control); });
    for (const auto* phase : input.instructions()) {
        llvm::outs() << "  phase=" << phase->GetIndex() << " anchor-phases=" << index.phasesFor(phase->elementOp).size()
                     << " context=[";
        llvm::interleaveComma(index.controlPath(*phase), llvm::outs(), [](mlir::Region* region) {
            llvm::outs() << region->getParentOp()->getName() << ":" << region->getRegionNumber();
        });
        llvm::outs() << "]\n";
    }
    dumpAvailability(index, function);
    return mlir::success();
}

mlir::LogicalResult dumpDemands(
    mlir::func::FuncOp function, const mlir::pto::SyncInput& input, const fs::StorageAnalysis& storage)
{
    fs::PhaseIndex index;
    if (mlir::failed(index.build(function, input))) {
        return mlir::failure();
    }
    auto sequence = index.explicitSequence(function.getBody().front());
    if (mlir::failed(sequence)) {
        return function.emitError("explicit demand inspection requires a straight-line single-phase sequence");
    }
    fs::LifetimeAnalysis lifetimes;
    fs::RankReduction reduction;
    if (mlir::failed(lifetimes.build(*sequence, storage)) ||
        mlir::failed(reduction.build(*sequence, lifetimes.generators()))) {
        return function.emitError("invalid explicit demand analysis input");
    }
    // Independent legacy entrypoint comparison: normalization must never
    // remove an authoritative conservative alias requirement or alter F*.
    fs::LifetimeAnalysis legacy;
    fs::RankReduction legacyReduction;
    if (mlir::failed(legacy.build(*sequence, storage.footprints(), storage.aliases())) ||
        mlir::failed(legacyReduction.build(*sequence, legacy.generators())) ||
        legacy.generators().size() != lifetimes.generators().size() ||
        legacyReduction.retained() != reduction.retained()) { return mlir::failure(); }
    for (auto [id, edge] : llvm::enumerate(lifetimes.generators())) {
        const auto& original = legacy.generators()[id];
        if (edge.source != original.source || edge.consumer != original.consumer ||
            edge.witnesses.size() != original.witnesses.size()) { return mlir::failure(); }
        for (auto [index, witness] : llvm::enumerate(edge.witnesses)) {
            const auto& old = original.witnesses[index];
            if (witness.hazard != old.hazard || witness.sourceFootprint != old.sourceFootprint ||
                witness.consumerFootprint != old.consumerFootprint || witness.previousWriter != old.previousWriter ||
                witness.sourceMemory != storage.footprints()[witness.sourceFootprint].sharedMemory ||
                witness.consumerMemory != storage.footprints()[witness.consumerFootprint].sharedMemory ||
                !witness.sourceMemory || !witness.consumerMemory) { return mlir::failure(); }
        }
    }
    llvm::outs() << "  modeled-alias-conflicts generators=" << lifetimes.generators().size()
                 << " retained=" << reduction.retained().size() << " bypasses=" << lifetimes.bypasses().size()
                 << " nonadjacent-local=" << reduction.nonadjacentLocal().size() << " pipes=[";
    llvm::interleaveComma(
        reduction.pipes(), llvm::outs(), [](auto pipe) { llvm::outs() << static_cast<unsigned>(pipe); });
    llvm::outs() << "]\n";
    for (auto [site, summary] : llvm::enumerate(reduction.summaries())) {
        llvm::outs() << "  site=" << site << " phase=" << summary.phase->GetIndex() << " rank=" << summary.rank
                     << " S=[";
        llvm::interleaveComma(summary.S, llvm::outs());
        llvm::outs() << "] T=[";
        llvm::interleaveComma(summary.T, llvm::outs());
        llvm::outs() << "]\n";
    }
    for (auto [id, edge] : llvm::enumerate(lifetimes.generators())) {
        llvm::outs() << "  demand=" << edge.source << "->" << edge.consumer
                     << " retained=" << llvm::is_contained(reduction.retained(), id) << " witnesses=";
        for (const auto& witness : edge.witnesses) {
            llvm::outs() << " " << static_cast<unsigned>(witness.hazard) << ":" << witness.sourceFootprint << ","
                         << witness.consumerFootprint;
            if (witness.sourceMemory && witness.consumerMemory) {
                llvm::outs() << "{bound-memory=" << *witness.sourceMemory << "," << *witness.consumerMemory << "}";
            }
        }
        llvm::outs() << "\n";
    }
    return mlir::success();
}

mlir::LogicalResult analyze(mlir::func::FuncOp function, Mode mode)
{
    mlir::pto::SyncInput input;
    if (mlir::failed(input.build(function))) {
        return mlir::failure();
    }
    if (function.isDeclaration()) {
        llvm::outs() << function.getName() << ": phases=0 declaration-skipped\n";
        return mlir::success();
    }
    fs::StorageAnalysis storage(input);
    if (!checkBoundPartition(storage)) { return function.emitError("invalid shared bound-cell partition"); }
    llvm::outs() << function.getName() << ": phases=" << input.instructions().size()
                 << " footprints=" << storage.footprints().size() << " alias-pairs=" << storage.aliases().size()
                 << " alias-queries=" << storage.aliasQueries() << "\n";
    if (mode == Mode::Storage) {
        dumpStorage(storage);
    }
    for (const auto* phase : input.instructions()) {
        llvm::outs() << "  phase=" << phase->GetIndex() << " op=" << phase->opName.getStringRef()
                     << " pipe=" << static_cast<unsigned>(phase->kPipeValue) << " reads=" << phase->useVec.size()
                     << " writes=" << phase->defVec.size() << "\n";
    }
    if (mode == Mode::Structure) {
        return dumpStructure(function, input);
    }
    if (mode == Mode::Guarded) {
        fs::GuardedDemandAnalysis analysis;
        if (mlir::failed(analysis.build(function, input, storage))) {
            return mlir::failure();
        }
        llvm::outs() << "  guarded=" << llvm::json::Value(frontier_test::dump(analysis)) << "\n";
        return mlir::success();
    }
    if (mode == Mode::Demands) {
        return dumpDemands(function, input, storage);
    }
    return mlir::success();
}

mlir::LogicalResult runFile(llvm::StringRef path, mlir::MLIRContext& context, Mode mode)
{
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(path, &context);
    if (!module || mlir::failed(mlir::verify(*module))) {
        return mlir::failure();
    }
    if (mode == Mode::Verify) {
        std::size_t localSync = 0, crossSync = 0;
        module->walk([&](mlir::Operation* operation) {
            const auto name = operation->getName().getStringRef();
            localSync += name == "pto.set_flag_dyn" || name == "pto.wait_flag_dyn";
            crossSync += name == "pto.sync.set" || name == "pto.sync.wait";
        });
        llvm::outs() << "verified: local-sync=" << localSync << " cross-sync=" << crossSync << "\n";
        return mlir::success();
    }
    const auto before = printIR(module->getOperation());
    for (auto function : module->getOps<mlir::func::FuncOp>()) {
        if (mode == Mode::ExpectFailure) {
            mlir::pto::SyncInput input;
            const bool translated = mlir::succeeded(input.build(function));
            const bool empty = input.ir().empty() && input.instructions().empty() && input.buffers().empty();
            if (translated || !empty) {
                return function.emitError("failed translation exposed partial shared records");
            }
            continue;
        }
        if (mlir::failed(analyze(function, mode))) {
            return mlir::failure();
        }
    }
    if (before != printIR(module->getOperation())) {
        return module->emitError("shared synchronization inspection changed the original IR");
    }
    if (mode == Mode::ExpectFailure) {
        llvm::outs() << "translation-failed; shared-records-empty; original-ir-unchanged\n";
    } else {
        llvm::outs() << "original-ir-unchanged; construction-not-run\n";
    }
    return mlir::success();
}

mlir::FailureOr<Mode> parseMode(llvm::StringRef option)
{
    if (option == "--verify-only") {
        return Mode::Verify;
    }
    if (option == "--expect-translation-failure") {
        return Mode::ExpectFailure;
    }
    if (option == "--dump-storage") {
        return Mode::Storage;
    }
    if (option == "--dump-structure") {
        return Mode::Structure;
    }
    if (option == "--dump-guarded-demands") {
        return Mode::Guarded;
    }
    if (option == "--dump-demands") {
        return Mode::Demands;
    }
    return mlir::failure();
}
} // namespace

int main(int argc, char** argv)
{
    bool preflight = argc == 3 && llvm::StringRef(argv[1]) == "--sync-preflight-check";
    bool gmAlias = argc == 3 && llvm::StringRef(argv[1]) == "--gm-alias-check";
    bool oneWayRepair = argc == 3 && llvm::StringRef(argv[1]) == "--one-way-repair-check";
    bool oneWayPhysical = argc == 3 && llvm::StringRef(argv[1]) == "--one-way-physical-check";
    bool adjacentLocalUpper = argc == 3 && llvm::StringRef(argv[1]) == "--adjacent-local-upper-check";
    bool generalComposition = argc == 3 && llvm::StringRef(argv[1]) == "--general-composition-check";
    bool periodicNestedQueries = argc == 3 && llvm::StringRef(argv[1]) == "--periodic-nested-query-check";
    bool symbolicAllocation = argc == 3 && llvm::StringRef(argv[1]) == "--symbolic-allocation-check";
    bool structuredCounters = argc == 3 && llvm::StringRef(argv[1]) == "--structured-counter-check";
    bool periodicAllocation = argc == 3 && llvm::StringRef(argv[1]) == "--periodic-allocation-check";
    bool singleStream = argc == 3 && llvm::StringRef(argv[1]) == "--single-stream-check";
    bool finitePhysical = argc == 3 && llvm::StringRef(argv[1]) == "--finite-physical-check";
    bool generalCounted = argc == 3 && llvm::StringRef(argv[1]) == "--general-counted-check";
    bool counted = argc == 3 && llvm::StringRef(argv[1]) == "--counted-reader-check";
    bool upper = argc == 3 && llvm::StringRef(argv[1]) == "--fixed-body-upper-check";
    bool stationary = argc == 3 && llvm::StringRef(argv[1]) == "--stationary-cell-check";
    bool regional = argc == 3 && llvm::StringRef(argv[1]) == "--regional-request-check";
    bool atomicStore = argc == 3 && llvm::StringRef(argv[1]) == "--atomic-store-check";
    bool structured = argc == 3 && llvm::StringRef(argv[1]) == "--structured-json";
    bool regularMode = argc == 3 && !structured &&
                       !preflight && !gmAlias && !atomicStore && !regional &&
                       !stationary && !counted &&
                       !generalCounted && !finitePhysical && !upper && !singleStream && !periodicAllocation &&
                       !oneWayRepair && !oneWayPhysical && !structuredCounters &&
                       !symbolicAllocation && !adjacentLocalUpper && !generalComposition && !periodicNestedQueries;
    auto mode = regularMode ? parseMode(argv[1]) : mlir::FailureOr<Mode>(Mode::Inspect);
    if ((argc != 2 && argc != 3) || mlir::failed(mode)) {
        llvm::errs() << "usage: pto-frontier-analysis-test "
                        "[--verify-only|--expect-translation-failure|--dump-storage|--dump-structure|--dump-demands|--"
                        "dump-guarded-demands] "
                        "input.pto\n";
        return 1;
    }
    mlir::DialectRegistry dialects;
    dialects.insert<
        mlir::pto::PTODialect, mlir::func::FuncDialect, mlir::arith::ArithDialect, mlir::scf::SCFDialect,
        mlir::cf::ControlFlowDialect, mlir::affine::AffineDialect, mlir::memref::MemRefDialect>();
    mlir::MLIRContext context(dialects);
    context.disableMultithreading();
    if (oneWayRepair) { return runOneWayRepairChecks(argv[2], context); }
    if (oneWayPhysical) { return runOneWayPhysicalChecks(argv[2], context); }
    if (adjacentLocalUpper) { return runAdjacentLocalUpperChecks(context); }
    if (generalComposition) { return runGeneralCompositionChecks(context); }
    if (periodicNestedQueries) { return runPeriodicNestedQueryChecks(context); }
    if (symbolicAllocation) { return runSymbolicAllocationChecks(context); }
    if (structuredCounters) { return runStructuredCounterChecks(context); }
    if (periodicAllocation) { return runPeriodicAllocationChecks(argv[2], context); }
    if (singleStream) { return runSingleStreamChecks(argv[2], context); }
    if (finitePhysical) { return runFinitePhysicalChecks(argv[2], context); }
    if (generalCounted) { return runGeneralCountedChecks(argv[2], context); }
    if (counted) { return runCountedReaderChecks(argv[2], context); }
    if (upper) { return runFixedBodyUpperChecks(argv[2], context); }
    if (stationary) { return runStationaryCellChecks(argv[2], context); }
    if (regional) { return runRegionalRequestChecks(argv[2], context); }
    if (atomicStore) {
        return runAtomicStoreChecks(argv[2], context);
    }
    if (gmAlias) {
        return runGMAliasChecks(argv[2], context);
    }
    if (preflight) {
        return runSyncTargetPreflightChecks(argv[2], context);
    }
    if (structured) {
        return runStructuredJSON(argv[2], context);
    }
    return mlir::failed(runFile(argv[argc == 3 ? 2 : 1], context, *mode));
}
