// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Allocation-only lowering of a certified logical plan. No scarcity repair.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_PHYSICALALLOCATION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_PHYSICALALLOCATION_H
#include "PTO/Transforms/FrontierSynch/Structured/PeriodicAllocation.h"
#include "PTO/Transforms/FrontierSynch/Structured/NumericTemplate.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseMap.h"
#include <optional>
namespace mlir::pto::frontiersynch {
inline constexpr llvm::StringLiteral CyclicAllocationAttr = "pto.cyclic_allocation";
// Certificate of a producer-proved uniform cyclic assignment. This is an
// internal pass interface, not a proof checker for arbitrary annotated IR.
// Guards, logical matching, occurrence ordinals and payload order must remain
// unchanged between production and consumption of this certificate.
DictionaryAttr encodeCyclicAllocation(const PeriodicAllocation& allocation, int64_t planId, MLIRContext* context);
// A source-tuple rule indexes one certified palette without changing matching.
// Coordinate zero is the logical source identity; subsequent coordinates are
// source enclosing visits or producer-certified derived selectors, excluding
// the original-record selector operand. A derived selector must evaluate
// identically at both endpoints and cannot stand in for a regional visit.
struct PhysicalCoordinateTerm {
    uint64_t coordinate = 0, stride = 0, phase = 0, modulus = 1, scale = 1;
};
struct PhysicalTupleRule {
    uint64_t coordinateCount = 1, base = 0;
    std::vector<PhysicalCoordinateTerm> terms;
};
// offset = base + sum scale * ((stride * coordinate + phase) mod modulus).
// Checks serialization bounds and that every possible offset fits the palette.
bool validPhysicalTupleRule(const PhysicalTupleRule& rule, uint64_t budget);
struct PhysicalRecordAllocation {
    int64_t record = 0;
    uint32_t sourcePipe = 0;
    uint32_t targetPipe = 0;
    uint64_t stride = 0;
    uint64_t phase = 0;
    // Ordered certified slot-to-ID assignment. Nonconflicting slots may
    // repeat a physical ID; cyclic and tuple offsets index the full table.
    SmallVector<int64_t> ids;
    std::optional<PhysicalTupleRule> tupleRule;
};
struct PhysicalAllocationPlan {
    int64_t planId = 0;
    std::vector<PhysicalRecordAllocation> records;
};
// Common hardware/input admission before either physical-output route.
LogicalResult preflightPhysicalOutput(func::FuncOp function, DictionaryAttr certificate);
// Validate grouped command provenance using the same family/cut/tuple checks as
// pure allocation. Singleton records are returned without rewriting identities.
FailureOr<llvm::DenseMap<Operation*, int64_t>> resolveExecutedFamilyCommands(
    func::FuncOp function, const PhysicalAllocationPlan& records);
// Both retained and serialized plans use this pure mapping planner. The
// supplied allocation is already certified; fitting only re-expresses it.
class PhysicalEndpointMap;
FailureOr<std::shared_ptr<const PhysicalEndpointMap>> planPhysicalEndpointMap(
    Operation* cut, ArrayRef<PhysicalRecordAllocation> members, ArrayRef<SmallVector<TemplateCoordinate>> points,
    uint64_t tupleArity, bool nested, DominanceInfo& dominance);
bool physicalMapRequiresMember(const PhysicalEndpointMap& mapping);
uint64_t physicalMapTupleArity(const PhysicalEndpointMap& mapping);
// Sufficient equality of complete total maps, independent of record labels.
bool equivalentPhysicalMaps(const PhysicalEndpointMap& a, const PhysicalEndpointMap& b);
std::optional<uint64_t> physicalMapCost(const PhysicalEndpointMap& mapping);
// Actual scalar construction before optional CSE/hoisting, using the emission
// schedule with symbolic tokens. Includes constants, selectors and tuple dispatch;
// constructs no SSA and excludes already prepared matching/guard inputs.
std::optional<uint64_t> physicalMapConstructionOperations(const PhysicalEndpointMap& mapping);
enum class PhysicalMapComparisonKind { Unknown, ExactFootprint, InputDominance };
// Identical live input masks cancel frozen upstream preparation; different
// selectors/tuple dispatch use exact shared-schedule counts. A residual dominance
// proof allows removing inputs/selectors without counting exclusive savings.
// The caller must preserve the physical-site/coalescing partition. Strictly
// smaller construction then proves a complete improvement at that frozen site.
// Unknown is not a cost or permission to replace an incumbent; no SSA is built.
std::optional<std::pair<uint64_t, uint64_t>> comparePhysicalMapConstruction(
    const PhysicalEndpointMap& incumbent, const PhysicalEndpointMap& candidate,
    PhysicalMapComparisonKind* kind = nullptr);
// Lowering is infallible for a validated immutable map and preflighted bindings.
// Matching coordinates exclude the provenance label: source ordinal first,
// followed by enclosing invocation coordinates. Lower only inside exact activity.
Value emitPhysicalEndpointMap(
    OpBuilder& builder, Location location, const PhysicalEndpointMap& mapping, ArrayRef<Value> tuple, Value member);
void emitPhysicalCommand(
    OpBuilder& builder, Location location, bool publish, uint32_t sourcePipe, uint32_t targetPipe, Value id);
FailureOr<PhysicalAllocationPlan> decodePhysicalAllocation(func::FuncOp function, ArrayRef<int64_t> eligibleIds);
FailureOr<PhysicalAllocationPlan> decodePhysicalAllocationCertificate(
    func::FuncOp function, DictionaryAttr certificate, ArrayRef<int64_t> eligibleIds);
FailureOr<PhysicalAllocationPlan> decodeCyclicAllocation(func::FuncOp function, ArrayRef<int64_t> eligibleIds);
// Decode retained alternative evidence without temporarily changing function attributes.
FailureOr<PhysicalAllocationPlan> decodeCyclicAllocationCertificate(
    func::FuncOp function, DictionaryAttr certificate, ArrayRef<int64_t> eligibleIds);
// Success lowers all logical commands; failure leaves the function unchanged.
// The caller supplies an eligible subset of native IDs 0..7. Default pools
// retain 0..5; explicitly including 6/7 is experimental. Every
// numeric ID holds at most one notification in its (source, target) direction.
// Numeric labels in different directions have independent notification state.
LogicalResult allocatePhysicalEventIds(func::FuncOp function, ArrayRef<int64_t> eligibleIds);
} // namespace mlir::pto::frontiersynch
#endif
