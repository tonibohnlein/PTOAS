// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Check the existing translator's own output without imposing ProtocolSync's
// narrower operation catalogue. Every claimed payload effect must appear in
// the correct read/write set; metadata is accounted for separately.
#include "PTO/Transforms/InsertSync/SyncEffectCoverage.h"
#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Threading.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;

namespace {
bool memoryType(Type type)
{
    return isa<TileBufType, MultiTileBufType, PtrType, TensorViewType, PartitionTensorViewType, BaseMemRefType>(type);
}

using Operations = llvm::DenseMap<Operation*, SmallVector<const CompoundInstanceElement*>>;

bool represented(const Operations& operations, Operation* op, Value value, bool write)
{
    auto found = operations.find(op);
    if (found == operations.end()) {
        return false;
    }
    for (const auto* phase : found->second) {
        const auto& accesses = write ? phase->defVec : phase->useVec;
        if (llvm::any_of(accesses, [&](const BaseMemInfo* info) { return info && info->baseBuffer == value; })) {
            return true;
        }
    }
    return false;
}

LogicalResult requireEffect(const Operations& operations, Operation* op, Value value, bool write)
{
    if (!value || !represented(operations, op, value, write)) {
        return op->emitError("InsertSync unsupported effect: ")
               << (write ? "write" : "read") << " has no translated storage/pipeline summary";
    }
    return success();
}

LogicalResult helper(const Operations& operations, func::CallOp call)
{
    auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(call, call.getCalleeAttr());
    auto effects = callee ? callee->getAttrOfType<ArrayAttr>("pto.tileop.effects") : ArrayAttr();
    if (!effects && callee && !callee.isDeclaration()) {
        // A visible, entirely pure helper needs no asynchronous payload
        // summary. Unknown calls and pipe operations are not pure by fiat.
        bool pure = true;
        callee.getBody().walk([&](Operation* nested) {
            if (!isa<func::ReturnOp, scf::YieldOp>(nested) &&
                (isa<OpPipeInterface>(nested) || !isMemoryEffectFree(nested))) {
                pure = false;
            }
        });
        if (pure) {
            return success();
        }
    }
    if (!effects || effects.size() != call.getNumOperands()) {
        return call.emitError("InsertSync unsupported helper: complete pto.tileop.effects contract required");
    }
    for (auto [operand, attribute] : llvm::zip(call.getOperands(), effects)) {
        auto text = dyn_cast<StringAttr>(attribute);
        if (!text || (text.getValue() != "none" && text.getValue() != "read" && text.getValue() != "write" &&
                      text.getValue() != "readwrite")) {
            return call.emitError("InsertSync unsupported helper: invalid operand effect");
        }
        if (!memoryType(operand.getType())) {
            if (text.getValue() != "none") {
                return call.emitError("InsertSync unsupported helper: scalar effect requires a resource summary");
            }
            continue;
        }
        if ((text.getValue() == "read" || text.getValue() == "readwrite") &&
            failed(requireEffect(operations, call, operand, false))) {
            return failure();
        }
        if ((text.getValue() == "write" || text.getValue() == "readwrite") &&
            failed(requireEffect(operations, call, operand, true))) {
            return failure();
        }
    }
    return success();
}

// Descriptor updates execute in scalar control and do not retire payload
// accesses. Full allocation bounds remain conservative across these versions.
// Until scalar asynchronous producers have an explicit descriptor dependency,
// do not erase their participation by calling the update pure.
bool scalarDescriptorInput(Value input)
{
    if (!input) return false;
    SmallVector<Value> pending{input};
    llvm::DenseSet<Value> visited;
    while (!pending.empty()) {
        Value value = pending.pop_back_val();
        if (!visited.insert(value).second) {
            continue;
        }
        if (visited.size() > 256) {
            return false;
        }
        if (auto argument = dyn_cast<BlockArgument>(value)) {
            if (isa<func::FuncOp>(argument.getOwner()->getParentOp())) {
                continue;
            }
            auto loop = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
            if (loop && argument == loop.getInductionVar()) {
                continue;
            }
            return false;
        }
        Operation* op = value.getDefiningOp();
        if (!op || isa<OpPipeInterface>(op) || !isMemoryEffectFree(op) || op->getNumRegions()) {
            return false;
        }
        llvm::append_range(pending, op->getOperands());
    }
    return true;
}

} // namespace

bool mlir::pto::isInsertSyncScalarPrerequisite(Value value)
{
    return scalarDescriptorInput(value);
}

namespace {
LogicalResult checkOperation(const Operations& operations, Operation* op)
{
    if (auto call = dyn_cast<func::CallOp>(op)) {
        return helper(operations, call);
    }
    if (auto descriptor = dyn_cast<SetValidShapeOp>(op)) {
        if (!scalarDescriptorInput(descriptor.getValidRow()) || !scalarDescriptorInput(descriptor.getValidCol())) {
            return op->emitError(
                "InsertSync unsupported descriptor prerequisite: asynchronous or untracked scalar source");
        }
        return success();
    }
    if (isa<GetValidShapeOp>(op)) {
        return success();
    }
    if (auto macro = getSyncMacroModel(op)) {
        for (const auto& phase : macro->phases) {
            for (Value value : phase.defValues) {
                if (failed(requireEffect(operations, op, value, true))) {
                    return failure();
                }
            }
            for (Value value : phase.useValues) {
                if (failed(requireEffect(operations, op, value, false))) {
                    return failure();
                }
            }
        }
        return success();
    }
    // Physical sections introduce execution context, not another effect.
    // The walk still checks every child independently.
    if (isa<AllocTileOp, AllocMultiTileOp, DeclareTileOp, DeclareGlobalOp,
            SectionCubeOp, SectionVectorOp>(op) ||
        op->hasTrait<OpTrait::IsTerminator>() || op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
        return success();
    }
    if (auto interface = dyn_cast<MemoryEffectOpInterface>(op)) {
        SmallVector<MemoryEffects::EffectInstance> effects;
        interface.getEffects(effects);
        for (const auto& effect : effects) {
            // Some operation interfaces list by-value scalar inputs as reads.
            // Constants and scalar SSA computations are not memory accesses.
            // Do not use this rule for untracked asynchronous scalar results.
            if (isa<MemoryEffects::Read>(effect.getEffect()) && effect.getValue() &&
                !memoryType(effect.getValue().getType()) && scalarDescriptorInput(effect.getValue())) {
                continue;
            }
            if (!isa<MemoryEffects::Read, MemoryEffects::Write>(effect.getEffect()) || !effect.getValue() ||
                !memoryType(effect.getValue().getType())) {
                return op->emitError("InsertSync unsupported effect: resource or memory operand is not modeled");
            }
            if (failed(
                    requireEffect(operations, op, effect.getValue(), isa<MemoryEffects::Write>(effect.getEffect())))) {
                return failure();
            }
        }
        return success();
    }
    if (isMemoryEffectFree(op)) {
        return success();
    }
    return op->emitError("InsertSync unsupported effect: missing semantic summary");
}
} // namespace

LogicalResult mlir::pto::checkInsertSyncEffectCoverage(func::FuncOp function, const SyncIRs& syncIR)
{
    Operations operations;
    for (const auto& element : syncIR) {
        if (auto* phase = dyn_cast<CompoundInstanceElement>(element.get())) {
            operations[phase->elementOp].push_back(phase);
        }
    }
    auto result = function.getBody().walk([&](Operation* op) {
        return failed(checkOperation(operations, op)) ? WalkResult::interrupt() : WalkResult::advance();
    });
    return failure(result.wasInterrupted());
}

namespace {
// Validate only authored contracts here, not the existence of a new summary.
// Missing contracts remain coverage gaps. A present malformed contract must
// never become a promise used by the translator or a reason to bypass checks.
LogicalResult validateExplicitEffectContracts(func::FuncOp function)
{
    auto result = function.getBody().walk([&](func::CallOp call) {
        auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(call, call.getCalleeAttr());
        if (!callee) {
            return WalkResult::advance();
        }
        Attribute raw = callee->getAttr("pto.tileop.effects");
        if (!raw) {
            return WalkResult::advance();
        }
        auto effects = dyn_cast<ArrayAttr>(raw);
        if (!effects || effects.size() != call.getNumOperands()) {
            call.emitError("InsertSync invalid helper contract: operand effect arity/type mismatch");
            return WalkResult::interrupt();
        }
        for (Attribute attribute : effects) {
            auto text = dyn_cast<StringAttr>(attribute);
            if (!text || (text.getValue() != "none" && text.getValue() != "read" &&
                          text.getValue() != "write" && text.getValue() != "readwrite")) {
                call.emitError("InsertSync unsupported helper: invalid operand effect");
                return WalkResult::interrupt();
            }
        }
        return WalkResult::advance();
    });
    return failure(result.wasInterrupted());
}
} // namespace

FailureOr<bool> mlir::pto::inspectInsertSyncEffectCoverage(
    func::FuncOp function, const SyncIRs& syncIR, bool strict)
{
    if (failed(validateExplicitEffectContracts(function))) {
        return failure();
    }
    if (strict) {
        if (failed(checkInsertSyncEffectCoverage(function, syncIR))) {
            return failure();
        }
        return true;
    }

    // Scope the handler to this read-only completeness query. It does not wrap
    // translation, planning, allocation, code generation, or their errors.
    // Preserve source locations and the original reason. No exception/assertion
    // is caught, and no missing operation is classified as pure.
    SmallVector<std::pair<Location, std::string>> gaps;
    LogicalResult checked = success();
    {
        // The diagnostic engine is shared by parallel function passes. Only
        // intercept this query's thread; another function's errors must keep
        // reaching its own handler (including hard contract diagnostics).
        const uint64_t checkingThread = llvm::get_threadid();
        ScopedDiagnosticHandler handler(function.getContext(), [&](Diagnostic& diagnostic) {
            if (llvm::get_threadid() != checkingThread ||
                diagnostic.getSeverity() != DiagnosticSeverity::Error) {
                return failure();
            }
            std::string message;
            llvm::raw_string_ostream stream(message);
            diagnostic.print(stream);
            stream.flush();
            gaps.emplace_back(diagnostic.getLocation(), std::move(message));
            return success();
        });
        checked = checkInsertSyncEffectCoverage(function, syncIR);
    }
    for (const auto& gap : gaps) {
        emitRemark(gap.first) << "InsertSync coverage gap (legacy translation retained, not verified): " << gap.second;
    }
    if (failed(checked) && gaps.empty()) {
        // An unclassified checker failure is not the expected missing-summary
        // outcome and is not silently converted to compatibility success.
        function.emitError("InsertSync coverage query failed without a diagnostic");
        return failure();
    }
    return succeeded(checked) && gaps.empty();
}
