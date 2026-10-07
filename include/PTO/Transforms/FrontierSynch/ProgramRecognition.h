// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Read-only structural recognition. All IR and Step 0 links are borrowed;
// function and SyncInput must outlive the result and remain unchanged.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_PROGRAMRECOGNITION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_PROGRAMRECOGNITION_H
#include "PTO/Transforms/FrontierSynch/ArithmeticProgram.h"
#include "PTO/IR/PTOSyncCapabilities.h"
#include "PTO/Transforms/FrontierSynch/NumericTemplateAnalysis.h"
#include "PTO/Transforms/FrontierSynch/NumericTemplateEndpoints.h"
#include "PTO/Transforms/FrontierSynch/PeriodicAllocation.h"
namespace mlir::pto::frontiersynch {
enum class StructureKind { Sequence, ExplicitRun, Loop, Conditional, Unsupported, Section };
struct StructureNode {
    StructureKind kind = StructureKind::Sequence;
    Operation* anchor = nullptr;
    Region* region = nullptr; // Original sequence/arm/body; never cloned.
    std::optional<std::size_t> parent;
    SmallVector<std::size_t> children;
    SmallVector<Operation*> operations; // Adjacent original leaves of an explicit run.
    SmallVector<std::size_t> payloads;
    std::size_t payloadCount = 0; // Includes descendants; counts static phases.
    SmallVector<scf::ForOp> loops; // Enclosing coordinates, outer to inner.
    std::optional<std::size_t> guard;
    bool unsupportedContext = false;
    SyncPhysicalCore executionCore() const { return recoverSyncPhysicalCore(anchor); }
    std::optional<RecognitionResult> explicitResult;
    std::optional<RecognitionResult> rotatingResult;
    std::optional<GuardedRecognition> finiteGuardedResult;
    std::optional<GuardedRecognition> guardedRotatingResult;
    std::optional<NumericTemplate> numericTemplate;
    std::optional<PeriodicAnalysis> periodicAnalysis;
    std::optional<NumericTemplateEndpoints> logicalEndpoints;
    std::optional<PeriodicAllocation> periodicAllocation;
};
struct ProgramPayload {
    const CompoundInstanceElement* phase = nullptr;
    std::size_t node = 0;
    SmallVector<std::size_t> effects; // Indices in input.accesses().effects().
};
struct ProgramGuard {
    std::optional<std::size_t> parent;
    scf::IfOp branch;
    bool takeThen = true;
    bool availableBeforeBranch = false;
    // Availability before each loop enclosing this branch (not its descendants).
    SmallVector<scf::ForOp> loops;
    SmallVector<bool> availableBeforeLoops;
};
struct ProgramRecognition {
    SmallVector<StructureNode, 0> nodes;
    // Stable identities in worklist order, never execution/reference ranks.
    SmallVector<ProgramPayload> payloads;
    SmallVector<ProgramGuard> guards;
    // Filled on demand by FrontierAnalysis::recognizeArithmetic. The producer
    // accepts a whole function only; subtrees are not independent invocations.
    std::optional<ArithmeticProgram> arithmetic;
};
// Visits all regions without unrolling, fuses adjacent leaves, and runs the
// structural checks independently. Arithmetic extraction is requested separately.
// Applicable certifies the input contract only. Backend fields remain absent
// until an explicit FrontierAnalysis::analyzeNumericCandidates request.
FailureOr<ProgramRecognition> recognizeProgram(func::FuncOp function, const SyncInput& input);
StringRef structureName(StructureKind kind);
} // namespace mlir::pto::frontiersynch
#endif
