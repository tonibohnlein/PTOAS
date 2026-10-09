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
#include "PTO/Transforms/FrontierSynch/VaryingRotatingRecognition.h"
#include "PTO/IR/PTOSyncCapabilities.h"
#include "PTO/Transforms/FrontierSynch/NumericTemplateAnalysis.h"
#include "PTO/Transforms/FrontierSynch/NumericTemplateEndpoints.h"
#include "PTO/Transforms/FrontierSynch/PeriodicAllocation.h"
namespace mlir::pto::frontiersynch {
struct BoundedLifetimeDemandResult;
enum class StructureKind { Sequence, ExplicitRun, Loop, Conditional, Unsupported, Section };
enum class ContractClass {
    Finite, FiniteGuarded, Periodic, GuardedPeriodic, BoundedLifetime,
    VaryingPeriodic, NumericTemplate, Differences, Octagons, BoundedCoefficients,
    Sequence, Repetition, FiniteOverlay, FiniteVisitTypes
};
enum class ContractStatus { Established, Violated, Unproved, NotEvaluated };
enum class ContractImplementation { NotRequested, Available, Unavailable };
enum class ContractDiagnosticKind { CriterionViolation, UnmetObligation, ProducerLimit, AdapterGap };
struct ContractObligation {
    std::string name;
    ContractStatus status = ContractStatus::NotEvaluated;
};
// Exact diagnostic multiplicity with the first source witness. Membership
// does not depend on retaining a separate copy for every failing row/piece.
struct ArithmeticContractDiagnostic {
    ArithmeticIssue issue;
    bool outsideClass = false;
    std::size_t relation = 0;
    std::size_t piece = 0;
    uint64_t count = 0;
};
SmallVector<ArithmeticContractDiagnostic> summarizeArithmeticDiagnostics(ArrayRef<ArithmeticDiagnostic> diagnostics);
struct ProgramContractCandidate {
    ContractClass kind = ContractClass::Finite;
    std::optional<std::size_t> node; // Absent for whole-function arithmetic profiles.
    ContractStatus membership = ContractStatus::NotEvaluated;
    SmallVector<RecognitionDiagnostic> diagnostics;
    SmallVector<ArithmeticContractDiagnostic> arithmeticDiagnostics;
    std::optional<ArithmeticLimits> arithmeticProfile;
    SmallVector<ContractObligation> obligations;
    // These describe observed pipeline stages only. Their failure never changes
    // membership, and absent observations never mean a violated class contract.
    ContractImplementation demands = ContractImplementation::NotRequested;
    ContractImplementation endpointRecipes = ContractImplementation::NotRequested;
    ContractImplementation allocation = ContractImplementation::NotRequested;
    std::string implementationError;
};
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
    std::optional<BoundedLifetimeRecognition> boundedLifetime;
    // Analysis cache owns a separate arena: failed parent-region transactions
    // must not invalidate successful source-window demand circuits.
    mutable bool boundedDemandAttempted = false;
    mutable std::shared_ptr<BoundedLifetimeDemandResult> boundedDemands;
    mutable std::string boundedDemandError, boundedExportError;
    std::optional<VaryingRotatingRecognition> varyingRotating;
    std::optional<AffineRotatingVisits> varyingDemands;
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
    // Immutable session configuration for all specialized composition views.
    // Direct library clients retain the historical regional defaults.
    SmallVector<ArithmeticLimits> regionalArithmeticProfiles{{8, 8, 1, 4096}, {8, 8, 2, 4096}};
    SmallVector<StructureNode, 0> nodes;
    // Stable identities in worklist order, never execution/reference ranks.
    SmallVector<ProgramPayload> payloads;
    SmallVector<ProgramGuard> guards;
    // Filled on demand by FrontierAnalysis::recognizeArithmetic. The producer
    // accepts a whole function only; subtrees are not independent invocations.
    std::optional<ArithmeticProgram> arithmetic;
    // Recognition-only snapshots for every requested fixed arithmetic profile.
    // Preserve earlier successes even when a later profile or backend fails.
    SmallVector<ProgramContractCandidate, 0> arithmeticContracts;
    // Original whole-function sequence evidence, recorded before endpoint
    // preparation. Generic selected/bounding compositions cannot supply it.
    std::optional<ProgramContractCandidate> sequenceContract;
    SmallVector<ProgramContractCandidate, 0> finiteVisitContracts;
    SmallVector<ProgramContractCandidate, 0> contractAudit;
};
// Rebuild only from existing recognition/backend observations; no extraction,
// demands, regional preparation, emission or allocation is invoked here.
void refreshProgramContractAudit(ProgramRecognition& program);
void recordArithmeticContractAttempt(ProgramRecognition& program, const ArithmeticLimits& limits,
                                     const ArithmeticProgram& arithmetic);
void recordArithmeticContractAttempt(ProgramRecognition& program, const ArithmeticLimits& limits,
                                     const ArithmeticProgram& arithmetic, std::optional<std::size_t> node);
// Update only endpoint availability on an existing sequence snapshot. An empty
// error means successful detached preparation; membership is never changed.
void recordSequenceEndpointAttempt(ProgramRecognition& program, StringRef error);
ContractDiagnosticKind contractDiagnosticKind(const RecognitionDiagnostic& diagnostic);
StringRef contractName(ContractClass kind);
StringRef contractName(ContractStatus status);
StringRef contractName(ContractImplementation status);
StringRef contractName(ContractDiagnosticKind kind);
// Visits all regions without unrolling, fuses adjacent leaves, and runs the
// structural checks independently. Arithmetic extraction is requested separately.
// Applicable certifies the input contract only. Backend fields remain absent
// until an explicit FrontierAnalysis::analyzeNumericCandidates request.
FailureOr<ProgramRecognition> recognizeProgram(func::FuncOp function, const SyncInput& input);
// Reuses the session index; the compatibility overload constructs its own.
FailureOr<ProgramRecognition> recognizeProgram(func::FuncOp function, const SyncInput& input,
                                               const PhaseIndex& index);
StringRef structureName(StructureKind kind);
} // namespace mlir::pto::frontiersynch
#endif
