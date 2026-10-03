// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Exact occurrence relations from the original structured MLIR control tree.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_STRUCTUREDINPUTADAPTER_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_STRUCTUREDINPUTADAPTER_H
#include "PTO/Transforms/FrontierSynch/PhaseIndex.h"
#include "PTO/Transforms/FrontierSynch/SymbolicDemandAnalysis.h"
#include <functional>
#include <string>

namespace mlir::pto::frontiersynch {
enum class StructuredImportStatus {
    Success,
    InvalidInput,
    UnsupportedControl,
    UnsupportedArithmetic,
    UnsupportedGuard,
    UnsupportedEffects,
    MissingQualification,
    RepresentationLimit
};
struct StructuredImportIssue {
    StructuredImportStatus status = StructuredImportStatus::Success;
    Operation* operation = nullptr;
    Value value;
    std::string reason;
};
struct StructuredSite {
    const CompoundInstanceElement* phase = nullptr;
    Operation* anchor = nullptr;
    SmallVector<Operation*> loops;
    SmallVector<Value> inductionVariables;
    // Original outer-to-inner regions, including the function body. The source
    // remains the sole control tree. localPhase is supplied for multi-phase ops.
    SmallVector<Region*> regions;
    std::size_t localPhase = 0;
};
struct ExactStructuredEffects {
    SymbolicPrimitive context;       // Unit -> Unit, admitted parameter valuations.
    SymbolicPrimitive reads, writes; // Complete Occurrence -> Cell relations.
    SymbolicPrimitive extras;        // Forward Occurrence -> Occurrence prerequisites.
    std::optional<SymbolicPrimitive> generators;
};
using ExactStructuredEffectsProvider =
    std::function<FailureOr<ExactStructuredEffects>(SymbolicSchemaHandle, ArrayRef<StructuredSite>)>;
// Invocation-owned validated metadata, borrowing unchanged MLIR and SyncInput.
// Construction checks ownership/coordinate identities once; it certifies neither
// arithmetic nor effects. Reuse after source mutation requires a new context.
class StructuredImportContext {
public:
    static FailureOr<std::shared_ptr<const StructuredImportContext>> create(
        func::FuncOp function, const SyncInput& input, SymbolicSchemaHandle schema,
        std::shared_ptr<const PhaseIndex> phases = {});
    bool matches(func::FuncOp function, const SyncInput& input, SymbolicSchemaHandle schema) const;
    const PhaseIndex& phases() const { return *index; }
    const DenseMap<const CompoundInstanceElement*, std::size_t>& tags() const { return phaseTags; }
    std::size_t phaseCount(Operation* scope) const { return scopeCounts.lookup(scope); }
private:
    StructuredImportContext() = default;
    func::FuncOp function;
    const SyncInput* input = nullptr;
    SymbolicSchemaHandle schema;
    std::shared_ptr<const PhaseIndex> index;
    DenseMap<const CompoundInstanceElement*, std::size_t> phaseTags;
    DenseMap<Operation*, std::size_t> scopeCounts;
};
struct StructuredImportOptions {
    RequirementModel model = RequirementModel::ExactPhysical;
    // Optional original subtree; ancestor guards/loop coordinates stay active.
    Operation* scope = nullptr;
    // Optional invocation schema preserves shared site IDs across scoped results.
    SymbolicSchemaHandle schema;
    std::shared_ptr<const StructuredImportContext> context;
    SmallVector<Value> parameters;
    SmallVector<Type> cellAxes;
    std::uint64_t physicalPartition = 0;
    // Must return exactly the anchor's shared phases in their semantic order.
    std::function<FailureOr<SmallVector<const CompoundInstanceElement*>>(
        Operation*, ArrayRef<const CompoundInstanceElement*>)>
        phaseOrder;
    // Non-function-argument symbols require an invocation-immutable result
    // certificate; availability and lack of enclosing loop are also checked.
    std::function<bool(Value, SymbolicSchemaHandle, const SymbolicPrimitive&)> immutableParameter;
    // Certifies mathematical interpretation on the validated admitted context.
    // Required for translated arithmetic/casts and loop progression, including
    // its terminating increment. Index width is NOT assumed to be 64 bits.
    // Intentional modular source semantics require a different exact adapter.
    std::function<bool(Operation*, SymbolicSchemaHandle, const SymbolicPrimitive&)> mathematicalArithmetic;
};
class StructuredInput;
using StructuredInputHandle = std::shared_ptr<const StructuredInput>;
struct StructuredImportResult {
    StructuredInputHandle input;
    StructuredImportIssue issue;
    bool succeeded() const { return bool(input); }
};
class StructuredInput {
public:
    SymbolicSchemaHandle schema() const { return owner; }
    const SymbolicInputs& relations() const { return inputs; }
    ArrayRef<StructuredSite> sites() const { return phaseSites; }

private:
    friend class StructuredInputAdapter;
    SymbolicSchemaHandle owner;
    SymbolicInputs inputs;
    SmallVector<StructuredSite> phaseSites;
};
class StructuredInputAdapter {
public:
    // Atomic, solver-free construction. SCF/affine positive constant-step loops
    // and deterministic affine/Boolean conditions are supported. Exact effects,
    // common partition and native pipe-chain semantics are supplied premises.
    // Certificates qualify source semantics; they never assert solver success.
    // Provider relation ownership/roles are checked BEFORE qualifications run.
    // Callbacks and option arrays are not retained. MLIR/SyncInput/context and
    // their phase records must remain alive and unchanged for the result's life.
    // Success supplies an exact artifact, not backend or endpoint availability.
    static StructuredImportResult build(
        func::FuncOp function, const SyncInput& input, const StructuredImportOptions& options,
        const ExactStructuredEffectsProvider& effects);
};
} // namespace mlir::pto::frontiersynch
#endif
