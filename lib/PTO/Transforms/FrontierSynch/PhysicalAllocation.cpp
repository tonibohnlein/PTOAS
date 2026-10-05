// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Preflight logical families, preserve certified member-to-phase maps, and lower
// commands in place. Allocation does not inspect or reconstruct endpoint guards.
#include "PTO/Transforms/FrontierSynch/PhysicalAllocation.h"
#include "PTO/Transforms/FrontierSynch/FamilyExpressions.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/CSE.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <tuple>
#include <optional>
namespace mlir::pto::frontiersynch {
namespace {
struct Family {
    SmallVector<const PhysicalRecordAllocation*> members;
    std::optional<int64_t> sourceCut, targetCut;
    SmallVector<int64_t> originalRecords;
    int64_t sourcePipe = 0, targetPipe = 0;
    uint64_t displacement = 0;
};
using Records = llvm::DenseMap<int64_t, const PhysicalRecordAllocation*>;
using Families = std::map<int64_t, Family>;
struct Endpoint {
    Operation* operation = nullptr;
    const PhysicalRecordAllocation* allocation = nullptr;
    SmallVector<uint64_t> phases;
    SmallVector<uint64_t> labels;
    std::unique_ptr<Block> phaseCode;
    Operation* phaseBefore = nullptr;
    Value directPhase;
};
std::optional<int64_t> number(DictionaryAttr dictionary, StringRef name)
{
    auto attr = dictionary.getAs<IntegerAttr>(name);
    if (!attr || !attr.getValue().isSignedIntN(64)) {
        return std::nullopt;
    }
    return attr.getInt();
}
LogicalResult readFamily(func::FuncOp function, DictionaryAttr item, const Records& records,
                         Families& families, llvm::DenseSet<int64_t>& seen)
{
    auto id = number(item, "id"), source = number(item, "source_pipe"), target = number(item, "target_pipe");
    auto sourceCut = number(item, "source_cut"), targetCut = number(item, "target_cut");
    auto displacement = number(item, "displacement");
    auto local = item.getAs<BoolAttr>("local");
    auto members = item.getAs<ArrayAttr>("members");
    if (!id || *id < 0 || !source || !target || !sourceCut || !targetCut || *targetCut < 0 ||
        !displacement || !local || local.getValue() != (*source == *target) || (!local.getValue() && *sourceCut < 0) ||
        !members || members.empty() || families.count(*id)) {
        return function.emitError("malformed or duplicate endpoint-family allocation identity");
    }
    Family family;
    family.sourceCut = sourceCut;
    family.targetCut = targetCut;
    family.sourcePipe = *source;
    family.targetPipe = *target;
    family.displacement = static_cast<uint64_t>(*displacement);
    for (auto attr : members) {
        auto member = dyn_cast<DictionaryAttr>(attr);
        auto record = member ? number(member, "record") : std::nullopt;
        const auto* allocation = record ? records.lookup(*record) : nullptr;
        if (!record || *record < 0 || !seen.insert(*record).second ||
            (local.getValue() ? allocation != nullptr : allocation == nullptr)) {
            return function.emitError("endpoint-family member is missing, repeated, or absent from its allocation");
        }
        family.originalRecords.push_back(*record);
        if (!allocation) {
            continue;
        }
        if (allocation->sourcePipe != *source || allocation->targetPipe != *target ||
            (!family.members.empty() && (allocation->stride != family.members.front()->stride ||
                                        allocation->ids != family.members.front()->ids))) {
            return function.emitError("endpoint-family members have incompatible cyclic allocations");
        }
        family.members.push_back(allocation);
    }
    // Keep local family identities occupied, but no notification can name one.
    families.emplace(*id, std::move(family));
    return success();
}
FailureOr<Families> readFamilies(func::FuncOp function, const PhysicalAllocationPlan& plan, const Records& records)
{
    Families result;
    auto attribute = function->getAttr("pto.endpoint_families");
    auto metadata = dyn_cast_or_null<DictionaryAttr>(attribute);
    if (attribute && !metadata) {
        return function.emitError("malformed endpoint-family metadata"), failure();
    }
    auto version = metadata ? number(metadata, "version") : std::optional<int64_t>(1);
    auto planId = metadata ? number(metadata, "plan") : std::optional<int64_t>(plan.planId);
    if (!version || (*version != 1 && *version != 2 && *version != 3) || !planId || *planId != plan.planId) {
        return function.emitError("unsupported endpoint-family metadata version or plan identity"), failure();
    }
    if (*version == 1) {
        // Version 1 stored provenance for per-record logical commands; adapt
        // those commands as singleton families without parsing their guards.
        for (const auto& record : plan.records) {
            result.emplace(record.record, Family{{&record}, {}, {}});
        }
        return result;
    }
    auto families = metadata.getAs<ArrayAttr>("families");
    if (!families) {
        return function.emitError("endpoint-family metadata is missing families"), failure();
    }
    llvm::DenseSet<int64_t> seen;
    for (auto attr : families) {
        auto item = dyn_cast<DictionaryAttr>(attr);
        if (!item) {
            return function.emitError("malformed endpoint-family entry"), failure();
        }
        if (failed(readFamily(function, item, records, result, seen))) {
            return failure();
        }
    }
    for (const auto& record : plan.records) {
        if (!seen.contains(record.record)) {
            return function.emitError("allocation record is missing from the endpoint-family partition"), failure();
        }
    }
    return result;
}
bool validMember(Operation* op, const Family& family)
{
    if (op->getNumOperands() < 1 || op->getNumOperands() > 2 || !op->getOperand(0).getType().isIndex()) {
        return false;
    }
    if (op->getNumOperands() == 1) {
        return family.members.size() == 1;
    }
    auto member = op->getOperand(1);
    if (!member.getType().isIndex()) {
        return false;
    }
    APInt value;
    if (matchPattern(member, m_ConstantInt(&value))) {
        return !value.isNegative() && value.getActiveBits() <= 64 && value.getZExtValue() < family.members.size();
    }
    return family.sourceCut.has_value(); // Dynamic members require the grouped-family producer's certificate.
}
struct Piece {
    int64_t family = 0, kind = 0, cut = 0;
    int64_t sourcePipe = 0, targetPipe = 0;
    SmallVector<int64_t> records;
};
using Pieces = std::map<int64_t, Piece>;
using Namespace = std::tuple<int64_t, int64_t, uint64_t>;
using SideCounts = llvm::DenseMap<int64_t, std::array<unsigned, 3>>;
FailureOr<Pieces> readPieces(func::FuncOp function, const Families& families)
{
    std::map<int64_t, const Family*> owners;
    std::map<Namespace, int64_t> namespaces;
    for (const auto& entry : families) {
        const auto& family = entry.second;
        Namespace key{family.sourcePipe, family.targetPipe, family.displacement};
        for (auto record : family.originalRecords) {
            owners.emplace(record, &family);
            auto [found, added] = namespaces.emplace(key, record);
            if (!added) {
                found->second = std::min(found->second, record);
            }
        }
    }
    auto metadata = function->getAttrOfType<DictionaryAttr>("pto.endpoint_families");
    auto pieces = metadata.getAs<ArrayAttr>("pieces");
    if (!pieces) {
        return function.emitError("endpoint-family metadata is missing executable pieces"), failure();
    }
    Pieces result;
    SideCounts counts;
    for (auto attr : pieces) {
        auto item = dyn_cast<DictionaryAttr>(attr);
        if (!item) {
            return function.emitError("malformed endpoint-family piece"), failure();
        }
        auto id = number(item, "id"), family = number(item, "family"), kind = number(item, "kind");
        auto cut = number(item, "cut"), source = number(item, "source_pipe"), target = number(item, "target_pipe");
        auto displacement = number(item, "displacement");
        auto records = item.getAs<DenseI64ArrayAttr>("records");
        if (!id || *id < 0 || !family || !kind || *kind < 0 || *kind > 2 || !cut || *cut < 0 ||
            !source || !target || !displacement || !records || records.empty() || result.count(*id)) {
            return function.emitError("malformed or duplicate endpoint-family piece identity"), failure();
        }
        Namespace key{*source, *target, static_cast<uint64_t>(*displacement)};
        auto name = namespaces.find(key);
        if (name == namespaces.end() || name->second != *family) {
            return function.emitError("endpoint piece has an inconsistent family namespace"), failure();
        }
        Piece piece{*family, *kind, *cut, *source, *target, {}};
        for (auto record : records.asArrayRef()) {
            auto owner = owners.find(record);
            if (owner == owners.end()) {
                return function.emitError("endpoint piece names an unknown original record"), failure();
            }
            const auto& provenance = *owner->second;
            const bool local = provenance.sourcePipe == provenance.targetPipe;
            auto expectedCut = *kind == 0 ? provenance.sourceCut : provenance.targetCut;
            if (Namespace{provenance.sourcePipe, provenance.targetPipe, provenance.displacement} != key ||
                local != (*kind == 1) || expectedCut != cut || ++counts[record][*kind] != 1) {
                return function.emitError("endpoint piece has repeated or inconsistent record coverage"), failure();
            }
            piece.records.push_back(record);
        }
        result.emplace(*id, std::move(piece));
    }
    for (const auto& entry : owners) {
        const bool local = entry.second->sourcePipe == entry.second->targetPipe;
        const std::array<unsigned, 3> expected = local ? std::array<unsigned, 3>{0, 1, 0} :
            std::array<unsigned, 3>{1, 0, 1};
        if (counts.lookup(entry.first) != expected) {
            return function.emitError("endpoint pieces do not cover every original record endpoint"), failure();
        }
    }
    return result;
}
bool validRecordMember(Operation* op, const Piece& piece)
{
    if (op->getNumOperands() != 2 || !op->getOperand(0).getType().isIndex() ||
        !op->getOperand(1).getType().isIndex()) {
        return false;
    }
    APInt value;
    if (!matchPattern(op->getOperand(1), m_ConstantInt(&value))) {
        return true; // Its exact label domain is certified by the producer.
    }
    return !value.isNegative() && value.isSignedIntN(64) &&
        llvm::is_contained(piece.records, value.getSExtValue());
}
struct CoordinateProvenance {
    llvm::DenseMap<int64_t, scf::ForOp> loops;
    std::map<int64_t, DictionaryAttr> members;
};
FailureOr<CoordinateProvenance> readCoordinates(func::FuncOp function)
{
    CoordinateProvenance result;
    auto walked = function.walk([&](scf::ForOp loop) -> WalkResult {
        auto attr = loop->getAttr("pto.family_loop");
        if (!attr) {
            return WalkResult::advance();
        }
        auto id = dyn_cast<IntegerAttr>(attr);
        if (!id || !id.getValue().isSignedIntN(64) || id.getInt() < 0 ||
            !result.loops.try_emplace(id.getInt(), loop).second) {
            loop.emitError("invalid or repeated endpoint-family loop identity");
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
    });
    if (walked.wasInterrupted()) {
        return failure();
    }
    auto metadata = function->getAttrOfType<DictionaryAttr>("pto.endpoint_families");
    for (auto attr : metadata.getAs<ArrayAttr>("families")) {
        auto family = cast<DictionaryAttr>(attr);
        for (auto memberAttr : family.getAs<ArrayAttr>("members")) {
            auto member = cast<DictionaryAttr>(memberAttr);
            result.members.emplace(*number(member, "record"), member);
        }
    }
    return result;
}
LogicalResult prepareCoordinatePhase(Endpoint& endpoint, const CoordinateProvenance& provenance,
                                     DominanceInfo& dominance, bool publish)
{
    SmallVector<SmallVector<TemplateCoordinate>> points;
    bool anyCoordinates = false;
    Operation* before = endpoint.operation;
    auto branch = dyn_cast<scf::IfOp>(before->getParentOp());
    bool outside = static_cast<bool>(branch);
    for (auto label : endpoint.labels) {
        auto member = provenance.members.find(static_cast<int64_t>(label));
        if (member == provenance.members.end()) {
            return before->emitError("endpoint-family phase references an unknown original record");
        }
        auto tuple = member->second.getAs<ArrayAttr>(publish ? "source" : "target");
        if (!tuple) {
            return before->emitError("endpoint-family member has malformed coordinate provenance");
        }
        SmallVector<TemplateCoordinate> point;
        for (auto attr : tuple) {
            auto coordinate = dyn_cast<DenseI64ArrayAttr>(attr);
            if (!coordinate || coordinate.size() != 2 || coordinate[0] < 0) {
                return before->emitError("endpoint-family coordinate requires a loop identity and induction value");
            }
            auto loop = provenance.loops.lookup(coordinate[0]);
            if (!loop || !dominance.properlyDominates(loop.getInductionVar(), before)) {
                return before->emitError("endpoint-family coordinate is unavailable at its command cut");
            }
            outside &= branch && dominance.properlyDominates(loop.getInductionVar(), branch.getOperation());
            point.push_back({loop, coordinate[1]});
        }
        anyCoordinates |= !point.empty();
        points.push_back(std::move(point));
    }
    if (!anyCoordinates) {
        // Old saved v3 fixtures can certify label selections without coordinate
        // provenance. They use the same exact label-to-phase expression adapter.
        return success();
    }
    auto block = std::make_unique<Block>();
    OpBuilder builder(before->getContext());
    builder.setInsertionPointToEnd(block.get());
    SmallVector<int64_t> phases;
    for (auto phase : endpoint.phases) {
        phases.push_back(static_cast<int64_t>(phase));
    }
    auto result = emitFamilyExpressions(builder, before->getLoc(), points, phases, endpoint.allocation->ids.size());
    if (!result.error.empty()) {
        return before->emitError("invalid endpoint-family phase coordinates: ") << result.error;
    }
    endpoint.phaseBefore = outside ? branch.getOperation() : before;
    endpoint.directPhase = result.member;
    endpoint.phaseCode = std::move(block);
    return success();
}
FailureOr<SmallVector<Endpoint>> preflightPieces(func::FuncOp function, const PhysicalAllocationPlan& plan,
                                               const Records& records, const Families& families)
{
    auto pieces = readPieces(function, families);
    if (failed(pieces)) {
        return failure();
    }
    auto coordinates = readCoordinates(function);
    if (failed(coordinates)) {
        return failure();
    }
    DominanceInfo dominance(function);
    SmallVector<Endpoint> endpoints;
    llvm::DenseSet<int64_t> seen;
    auto walked = function.walk([&](Operation* op) -> WalkResult {
        const bool publish = isa<LogicalSetOp>(op), consume = isa<LogicalWaitOp>(op);
        if (!publish && !consume) {
            if (isa<SetFlagOp, WaitFlagOp, SetFlagDynOp, WaitFlagDynOp, RecordEventOp, WaitEventOp>(op)) {
                op->emitError("physical allocation cannot share its supplied IDs with existing synchronization");
                return WalkResult::interrupt();
            }
            return WalkResult::advance();
        }
        auto pieceId = op->getAttrOfType<IntegerAttr>("pto.endpoint_piece");
        auto found = pieceId ? pieces->find(pieceId.getInt()) : pieces->end();
        if (found == pieces->end() || !seen.insert(found->first).second) {
            op->emitError("logical endpoint has a missing or repeated executable piece");
            return WalkResult::interrupt();
        }
        const auto& piece = found->second;
        auto id = op->getAttrOfType<IntegerAttr>("plan_id"), family = op->getAttrOfType<IntegerAttr>("record_id");
        auto cut = op->getParentOp()->getAttrOfType<IntegerAttr>("pto.endpoint_cut");
        auto source = op->getAttrOfType<PipeAttr>("src_pipe"), target = op->getAttrOfType<PipeAttr>("dst_pipe");
        if (!id || id.getInt() != plan.planId || !family || family.getInt() != piece.family ||
            piece.kind != (publish ? 0 : 2) || !cut || cut.getInt() != piece.cut ||
            !source || static_cast<int64_t>(source.getPipe()) != piece.sourcePipe ||
            !target || static_cast<int64_t>(target.getPipe()) != piece.targetPipe || !validRecordMember(op, piece)) {
            op->emitError("logical endpoint disagrees with its executable piece certificate");
            return WalkResult::interrupt();
        }
        Endpoint endpoint{op, records.lookup(piece.records.front()), {}, {}};
        if (!endpoint.allocation) {
            op->emitError("executable notification piece has no physical allocation");
            return WalkResult::interrupt();
        }
        for (auto record : piece.records) {
            const auto* member = records.lookup(record);
            if (!member || member->stride != endpoint.allocation->stride || member->ids != endpoint.allocation->ids) {
                op->emitError("executable piece has inconsistent member allocations");
                return WalkResult::interrupt();
            }
            endpoint.phases.push_back(member->phase % member->ids.size());
            endpoint.labels.push_back(static_cast<uint64_t>(record));
        }
        if (failed(prepareCoordinatePhase(endpoint, *coordinates, dominance, publish))) {
            return WalkResult::interrupt();
        }
        endpoints.push_back(std::move(endpoint));
        return WalkResult::advance();
    });
    if (walked.wasInterrupted()) {
        return failure();
    }
    for (const auto& entry : *pieces) {
        if (entry.second.kind != 1 && !seen.contains(entry.first)) {
            return function.emitError("executable piece has no logical endpoint"), failure();
        }
    }
    return endpoints;
}
FailureOr<SmallVector<Endpoint>> preflight(func::FuncOp function, const PhysicalAllocationPlan& plan)
{
    Records records;
    llvm::DenseMap<int64_t, std::pair<unsigned, unsigned>> counts;
    for (const auto& record : plan.records) {
        records[record.record] = &record;
    }
    auto families = readFamilies(function, plan, records);
    if (failed(families)) {
        return failure();
    }
    auto metadata = function->getAttrOfType<DictionaryAttr>("pto.endpoint_families");
    if (metadata && number(metadata, "version") == 3) {
        return preflightPieces(function, plan, records, *families);
    }
    SmallVector<Endpoint> endpoints;
    auto walked = function.walk([&](Operation* op) -> WalkResult {
        const bool publish = isa<LogicalSetOp>(op), consume = isa<LogicalWaitOp>(op);
        if (!publish && !consume) {
            if (isa<SetFlagOp, WaitFlagOp, SetFlagDynOp, WaitFlagDynOp, RecordEventOp, WaitEventOp>(op)) {
                op->emitError("physical allocation cannot share its supplied IDs with existing synchronization");
                return WalkResult::interrupt();
            }
            return WalkResult::advance();
        }
        auto id = op->getAttrOfType<IntegerAttr>("plan_id");
        auto record = op->getAttrOfType<IntegerAttr>("record_id");
        auto source = op->getAttrOfType<PipeAttr>("src_pipe"), target = op->getAttrOfType<PipeAttr>("dst_pipe");
        auto found = record ? families->find(record.getInt()) : families->end();
        if (!id || id.getInt() != plan.planId || found == families->end() || found->second.members.empty()) {
            op->emitError("logical endpoint does not match its cyclic allocation certificate");
            return WalkResult::interrupt();
        }
        const auto& family = found->second;
        const auto* allocation = family.members.front();
        if (!source || !target || static_cast<uint32_t>(source.getPipe()) != allocation->sourcePipe ||
            static_cast<uint32_t>(target.getPipe()) != allocation->targetPipe || !validMember(op, family)) {
            op->emitError("logical endpoint has an invalid family member or pipe assignment");
            return WalkResult::interrupt();
        }
        auto expectedCut = publish ? family.sourceCut : family.targetCut;
        auto cut = op->getParentOp()->getAttrOfType<IntegerAttr>("pto.endpoint_cut");
        if (expectedCut && (!cut || cut.getInt() != *expectedCut)) {
            op->emitError("logical endpoint does not match its certified family cut");
            return WalkResult::interrupt();
        }
        Endpoint endpoint{op, allocation, {}};
        for (const auto* member : family.members) {
            auto& count = counts[member->record];
            unsigned& occurrences = publish ? count.first : count.second;
            if (++occurrences != 1) {
                op->emitError("cyclic allocation expects one static SET and WAIT per family member");
                return WalkResult::interrupt();
            }
            endpoint.phases.push_back(member->phase % member->ids.size());
            endpoint.labels.push_back(endpoint.labels.size());
        }
        endpoints.push_back(std::move(endpoint));
        return WalkResult::advance();
    });
    if (walked.wasInterrupted()) {
        return failure();
    }
    for (const auto& record : plan.records) {
        if (counts.lookup(record.record) != std::make_pair(1U, 1U)) {
            return function.emitError("cyclic allocation record has missing logical endpoints"), failure();
        }
    }
    return endpoints;
}
Value memberPhase(OpBuilder& builder, const Endpoint& endpoint)
{
    auto location = endpoint.operation->getLoc();
    auto number = [&](uint64_t value) -> Value {
        return builder.create<arith::ConstantIndexOp>(location, static_cast<int64_t>(value));
    };
    const auto& phases = endpoint.phases;
    if (phases.size() == 1) {
        return number(phases.front());
    }
    Value member = endpoint.operation->getOperand(1);
    const uint64_t capacity = endpoint.allocation->ids.size();
    uint64_t coefficient = 0, intercept = 0;
    bool affine = false;
    // Label differences need not be invertible modulo the capacity. Enumerate
    // its small finite coefficient set and verify every original record label.
    for (uint64_t candidate = 0; candidate < capacity && !affine; ++candidate) {
        const auto offset = (phases.front() + capacity - (endpoint.labels.front() % capacity) * candidate % capacity)
            % capacity;
        bool valid = true;
        for (std::size_t i = 0; i < phases.size(); ++i) {
            valid &= phases[i] == (offset + (endpoint.labels[i] % capacity) * candidate) % capacity;
        }
        if (valid) {
            coefficient = candidate;
            intercept = offset;
            affine = true;
        }
    }
    if (affine) {
        if (!coefficient) {
            return number(intercept);
        }
        Value modulus = number(capacity);
        Value term = builder.create<arith::RemUIOp>(location, member, modulus);
        if (coefficient == 1 && !intercept) {
            return term;
        }
        if (coefficient != 1) {
            term = builder.create<arith::MulIOp>(location, term, number(coefficient));
        }
        if (intercept) {
            term = builder.create<arith::AddIOp>(location, number(intercept), term);
        }
        return builder.create<arith::RemUIOp>(location, term, modulus);
    }
    Value result = number(phases.front());
    // An irregular finite phase map remains one command with a shared decision
    // expression. It is never expanded back into independently guarded commands.
    for (std::size_t i = 1; i < phases.size(); ++i) {
        if (phases[i] != phases.front()) {
            auto selected = builder.create<arith::CmpIOp>(location, arith::CmpIPredicate::eq,
                                                          member, number(endpoint.labels[i]));
            result = builder.create<arith::SelectOp>(location, selected, number(phases[i]), result);
        }
    }
    return result;
}
Value physicalId(OpBuilder& builder, Location location, Value ordinal,
                 const PhysicalRecordAllocation& record, Value phaseOffset)
{
    auto constant = [&](uint64_t value) -> Value {
        return builder.create<arith::ConstantIndexOp>(location, static_cast<int64_t>(value));
    };
    const uint64_t capacity = record.ids.size();
    const uint64_t stride = record.stride % capacity;
    // memberPhase and the singleton constant are already in [0, capacity).
    // A zero cyclic stride needs no ordinal arithmetic or second remainder.
    Value phase = phaseOffset ? phaseOffset : constant(record.phase % capacity);
    if (stride) {
        Value modulus = constant(capacity);
        Value term = builder.create<arith::RemUIOp>(location, ordinal, modulus);
        if (stride != 1) {
            term = builder.create<arith::MulIOp>(location, term, constant(stride));
        }
        // Reduce before multiplying; native capacity bounds these intermediates.
        phase = builder.create<arith::AddIOp>(location, term, phase);
        phase = builder.create<arith::RemUIOp>(location, phase, modulus);
    }
    bool contiguous = true;
    for (std::size_t i = 0; i < record.ids.size(); ++i) {
        contiguous &= record.ids[i] == record.ids.front() + static_cast<int64_t>(i);
    }
    if (contiguous) {
        if (record.ids.front()) {
            return builder.create<arith::AddIOp>(location, phase, constant(record.ids.front()));
        }
        return phase;
    }
    Value id = constant(record.ids.front());
    for (std::size_t i = 1; i < record.ids.size(); ++i) {
        Value selected = builder.create<arith::CmpIOp>(location, arith::CmpIPredicate::eq, phase, constant(i));
        id = builder.create<arith::SelectOp>(location, selected, constant(record.ids[i]), id);
    }
    return id;
}
void emitAllocatedCommand(OpBuilder& builder, Operation* op, const PhysicalRecordAllocation& record,
                          Value eventId)
{
    const bool publish = isa<LogicalSetOp>(op);
    auto source = op->getAttrOfType<PipeAttr>("src_pipe"), target = op->getAttrOfType<PipeAttr>("dst_pipe");
    if (record.ids.size() == 1) {
        auto id = EventAttr::get(op->getContext(), *symbolizeEVENT(static_cast<uint32_t>(record.ids.front())));
        if (publish) {
            builder.create<SetFlagOp>(op->getLoc(), source, target, id);
        } else {
            builder.create<WaitFlagOp>(op->getLoc(), source, target, id);
        }
    } else {
        if (publish) {
            builder.create<SetFlagDynOp>(op->getLoc(), source, target, eventId);
        } else {
            builder.create<WaitFlagDynOp>(op->getLoc(), source, target, eventId);
        }
    }
}
} // namespace
LogicalResult allocatePhysicalEventIds(func::FuncOp function, ArrayRef<int64_t> eligibleIds)
{
    if (!function) {
        return failure();
    }
    auto plan = decodeCyclicAllocation(function, eligibleIds);
    if (failed(plan)) {
        return failure();
    }
    auto endpoints = preflight(function, *plan);
    if (failed(endpoints)) {
        return failure();
    }
    const auto width = DataLayout::closest(function).getTypeSizeInBits(IndexType::get(function.getContext()));
    if (width.isScalable() || width.getFixedValue() != 64) {
        return function.emitError("physical allocation requires 64-bit occurrence indices");
    }
    DominanceInfo dominance(function);
    for (const auto& endpoint : *endpoints) {
        OpBuilder builder(endpoint.operation);
        auto branch = dyn_cast<scf::IfOp>(endpoint.operation->getParentOp());
        if (branch && branch->hasAttr("pto.endpoint_cut") &&
            llvm::all_of(endpoint.operation->getOperands(), [&](Value operand) {
                return dominance.properlyDominates(operand, branch.getOperation());
            })) {
            builder.setInsertionPoint(branch);
        }
        Value eventId;
        if (endpoint.allocation->ids.size() != 1) {
            Value phase;
            if (endpoint.directPhase) {
                auto* before = endpoint.phaseBefore;
                before->getBlock()->getOperations().splice(before->getIterator(), endpoint.phaseCode->getOperations());
                phase = endpoint.directPhase;
                // The ordinal may be defined inside its guard even when the
                // coordinate phase can safely be shared outside that guard.
                if (before == endpoint.operation) {
                    builder.setInsertionPoint(endpoint.operation);
                }
            } else {
                phase = memberPhase(builder, endpoint);
            }
            eventId = physicalId(builder, endpoint.operation->getLoc(), endpoint.operation->getOperand(0),
                                 *endpoint.allocation, phase);
        }
        // All member-phase arithmetic is total and may be shared at the cut.
        // Command execution remains inside the original presence guard.
        builder.setInsertionPoint(endpoint.operation);
        emitAllocatedCommand(builder, endpoint.operation, *endpoint.allocation, eventId);
        endpoint.operation->erase();
    }
    function.walk([](Operation* op) {
        op->removeAttr("pto.endpoint_cut");
        op->removeAttr("pto.family_loop");
        op->removeAttr("pto.endpoint_piece");
    });
    function->removeAttr("pto.endpoint_families");
    function->removeAttr(CyclicAllocationAttr);
    IRRewriter rewriter(function.getContext());
    eliminateCommonSubExpressions(rewriter, dominance, function);
    return success();
}
} // namespace mlir::pto::frontiersynch
namespace mlir::pto {
#define GEN_PASS_DEF_PTOFRONTIERALLOCATE
#include "PTO/Transforms/Passes.h.inc"
namespace {
class PTOFrontierAllocatePass : public impl::PTOFrontierAllocateBase<PTOFrontierAllocatePass> {
public:
    using Base = impl::PTOFrontierAllocateBase<PTOFrontierAllocatePass>;
    PTOFrontierAllocatePass() = default;
    explicit PTOFrontierAllocatePass(const PTOFrontierAllocateOptions& options) : Base(options) {}
    void runOnOperation() override
    {
        if (failed(frontiersynch::allocatePhysicalEventIds(getOperation(), eligibleIds))) {
            signalPassFailure();
        }
    }
};
} // namespace
std::unique_ptr<Pass> createPTOFrontierAllocatePass()
{
    return std::make_unique<PTOFrontierAllocatePass>();
}
std::unique_ptr<Pass> createPTOFrontierAllocatePass(const PTOFrontierAllocateOptions& options)
{
    return std::make_unique<PTOFrontierAllocatePass>(options);
}
} // namespace mlir::pto
