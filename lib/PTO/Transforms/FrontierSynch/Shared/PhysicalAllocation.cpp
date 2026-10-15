// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Preflight logical families, preserve certified member-to-phase maps, and lower
// commands in place. Allocation does not inspect or reconstruct endpoint guards.
#include "PTO/Transforms/FrontierSynch/Shared/PhysicalAllocation.h"
#include "PhysicalMapInputs.h"
#include "PTO/Transforms/FrontierSynch/Shared/PhysicalExecutedCounters.h"
#include "PTO/Transforms/FrontierSynch/Shared/ExecutionContexts.h"
#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include "PTO/Transforms/FrontierSynch/Finite/FiniteAllocation.h"
#include "PTO/Transforms/FrontierSynch/Shared/FamilyExpressions.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/InsertSync/SyncInput.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <tuple>
#include <optional>
#include <set>
#include <unordered_map>
namespace mlir::pto::frontiersynch {
bool validPhysicalTupleRule(const PhysicalTupleRule& rule, uint64_t budget)
{
    if (!budget || !rule.coordinateCount || rule.coordinateCount > INT64_MAX ||
        rule.base > INT64_MAX || rule.base >= budget) { return false; }
    uint64_t maximum = rule.base;
    for (const auto& term : rule.terms) {
        if (term.coordinate >= rule.coordinateCount || term.stride > INT64_MAX ||
            term.phase > INT64_MAX || !term.modulus || term.modulus > INT64_MAX ||
            term.scale > INT64_MAX) { return false; }
        const auto residue = term.modulus - 1;
        if (residue && term.scale > (budget - 1 - maximum) / residue) { return false; }
        maximum += term.scale * residue;
    }
    return true;
}
namespace {
struct Family {
    SmallVector<const PhysicalRecordAllocation*> members;
    std::optional<int64_t> sourceCut, targetCut;
    SmallVector<int64_t> originalRecords;
    int64_t sourcePipe = 0, targetPipe = 0;
    uint64_t displacement = 0;
    SmallVector<int64_t> sourceChoices, targetChoices; // Invocation loop ID, then exact alternative cut IDs.
};
using Records = llvm::DenseMap<int64_t, const PhysicalRecordAllocation*>;
using Families = std::map<int64_t, Family>;
struct ComposedTupleMapping {
    PhysicalTupleRule rule;
    Value base;
    SmallVector<Value> phases;
};
// Validated high-level physical map. Planning and candidate scoring retain
// fitted family selectors; only the selected map creates scalar operations.
struct PhysicalIdRecipe {
    enum class Kind { Phase, Record, Id, Tuple } kind = Kind::Phase;
    std::shared_ptr<const FamilyExpressionRecipe> selector;
    SmallVector<std::shared_ptr<const FamilyExpressionRecipe>> phases;
    PhysicalTupleRule tuple;
    SmallVector<int64_t> palette;
};
struct Endpoint {
    Operation* operation = nullptr;
    const PhysicalRecordAllocation* allocation = nullptr;
    SmallVector<uint64_t> phases;
    SmallVector<uint64_t> labels;
    std::unique_ptr<PhysicalIdRecipe> mapping;
    std::unique_ptr<Block> phaseCode;
    Operation* phaseBefore = nullptr;
    Value directPhase; // Original record label when tupleMembers selects mixed palettes; phase otherwise.
    SmallVector<const PhysicalRecordAllocation*> tupleMembers;
    Value directId; // Coordinate-to-ID composition, independent of original record numbering.
    // Shared dynamic tuple skeleton with coordinate-selected base and phases.
    std::unique_ptr<ComposedTupleMapping> composedTuple;
};
} // namespace
class PhysicalEndpointMap {
public:
    std::vector<PhysicalRecordAllocation> records;
    Endpoint endpoint;
    uint64_t arity = 0;
};
namespace {
std::optional<int64_t> constantTupleId(const PhysicalRecordAllocation& record);
std::optional<uint64_t> mappingCost(ArrayRef<Endpoint> endpoints);

std::vector<uint64_t> physicalMappingKey(const PhysicalRecordAllocation& member)
{
    if (auto fixed = constantTupleId(member)) { return {0, static_cast<uint64_t>(*fixed)}; }
    PhysicalTupleRule flat{1, 0, {{0, member.stride, member.phase, member.ids.size(), 1}}};
    const auto& rule = member.tupleRule ? *member.tupleRule : flat;
    uint64_t base = rule.base;
    std::map<std::array<uint64_t, 4>, uint64_t> terms;
    for (const auto& term : rule.terms) {
        if (!term.scale || term.modulus == 1) { continue; }
        const auto stride = term.stride % term.modulus, phase = term.phase % term.modulus;
        // These supplied rules have passed the palette-range check. Folding
        // constants and adding equal nonnegative terms cannot exceed its bound.
        if (!stride) { base += term.scale * phase; }
        else { terms[{term.coordinate, stride, phase, term.modulus}] += term.scale; }
    }
    std::vector<uint64_t> key{1, base, member.ids.size()};
    for (auto id : member.ids) { key.push_back(static_cast<uint64_t>(id)); }
    for (const auto& [term, scale] : terms) {
        key.insert(key.end(), term.begin(), term.end());
        key.push_back(scale);
    }
    return key;
}

SmallVector<int64_t> constantMemberIds(const Endpoint& endpoint)
{
    SmallVector<int64_t> result;
    for (const auto* member : endpoint.tupleMembers) {
        auto id = constantTupleId(*member);
        if (!id) { return {}; }
        result.push_back(*id);
    }
    return result;
}

struct CommonTupleMapping {
    PhysicalTupleRule dynamic;
    SmallVector<int64_t> bases;
    SmallVector<SmallVector<int64_t>> phases;
    bool varyingPhases = false;
};
// Factor a supplied assignment, never infer reuse from the resulting formula.
// The input rules have already passed the complete palette-range certificate.
std::optional<CommonTupleMapping> commonTupleMapping(ArrayRef<const PhysicalRecordAllocation*> members)
{
    if (members.empty() || !members.front()->tupleRule) { return std::nullopt; }
    CommonTupleMapping result;
    result.dynamic.coordinateCount = members.front()->tupleRule->coordinateCount;
    using Term = std::array<uint64_t, 5>;
    std::vector<Term> common;
    for (const auto* member : members) {
        if (!member->tupleRule || member->ids != members.front()->ids ||
            member->tupleRule->coordinateCount != result.dynamic.coordinateCount) { return std::nullopt; }
        uint64_t base = member->tupleRule->base;
        std::vector<Term> terms;
        for (const auto& term : member->tupleRule->terms) {
            if (!term.scale || term.modulus == 1) { continue; }
            const auto stride = term.stride % term.modulus, phase = term.phase % term.modulus;
            if (!stride) { base += term.scale * phase; }
            // Sort by the common skeleton before phase, so independently
            // ordered terms still pair exactly. Equal skeletons are additive.
            else { terms.push_back({term.coordinate, stride, term.modulus, term.scale, phase}); }
        }
        llvm::sort(terms);
        if (result.bases.empty()) {
            common = terms;
            result.phases.resize(terms.size());
        }
        if (terms.size() != common.size()) { return std::nullopt; }
        for (std::size_t i = 0; i < terms.size(); ++i) {
            if (!std::equal(terms[i].begin(), terms[i].begin() + 4, common[i].begin())) { return std::nullopt; }
            result.phases[i].push_back(static_cast<int64_t>(terms[i][4]));
            result.varyingPhases |= terms[i][4] != common[i][4];
        }
        result.bases.push_back(static_cast<int64_t>(base));
    }
    if (common.empty()) { return std::nullopt; } // Constant IDs already have a stronger composition path.
    for (const auto& term : common) {
        result.dynamic.terms.push_back({term[0], term[1], term[4], term[2], term[3]});
    }
    return result;
}

Value selectPhysicalPalette(OpBuilder& builder, Location location, Value index, ArrayRef<int64_t> palette)
{
    if (palette.size() > 1 && llvm::all_of(llvm::enumerate(palette), [&](auto entry) {
            return entry.value() == palette.front() + static_cast<int64_t>(entry.index());
        })) {
        if (!palette.front()) { return index; }
        Value base = builder.create<arith::ConstantIndexOp>(location, palette.front());
        return builder.create<arith::AddIOp>(location, index, base);
    }
    Value result = builder.create<arith::ConstantIndexOp>(location, palette.front());
    for (std::size_t i = 1; i < palette.size(); ++i) {
        Value position = builder.create<arith::ConstantIndexOp>(location, static_cast<int64_t>(i));
        Value selected = builder.create<arith::CmpIOp>(location, arith::CmpIPredicate::eq, index, position);
        Value id = builder.create<arith::ConstantIndexOp>(location, palette[i]);
        result = builder.create<arith::SelectOp>(location, selected, id, result);
    }
    return result;
}
bool commonPalette(ArrayRef<const PhysicalRecordAllocation*> members)
{
    return !members.empty() && llvm::all_of(members, [&](const PhysicalRecordAllocation* member) {
        return !member->tupleRule && member->stride == members.front()->stride &&
               member->ids == members.front()->ids;
    });
}
std::optional<int64_t> number(DictionaryAttr dictionary, StringRef name)
{
    auto attr = dictionary.getAs<IntegerAttr>(name);
    if (!attr || !attr.getValue().isSignedIntN(64)) {
        return std::nullopt;
    }
    return attr.getInt();
}
bool readChoices(DictionaryAttr item, StringRef name, int64_t primary, SmallVectorImpl<int64_t>& output)
{
    auto raw = item.get(name);
    if (!raw) { return true; }
    auto array = dyn_cast<DenseI64ArrayAttr>(raw);
    if (!array) { return false; }
    if (array.empty()) { return true; }
    if (array.size() < 3 || array[0] < -1 ||
        (array[0] == -1 && (name != "target_choices" || array.size() != 3)) || array[1] != primary) {
        return false;
    }
    std::set<int64_t> cuts;
    for (auto cut : array.asArrayRef().drop_front()) {
        if (cut < 0 || !cuts.insert(cut).second) { return false; }
    }
    output.append(array.asArrayRef().begin(), array.asArrayRef().end());
    return true;
}
LogicalResult readFamily(func::FuncOp function, DictionaryAttr item, const Records& records,
                         Families& families, llvm::DenseSet<int64_t>& seen, bool nested)
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
    if (!readChoices(item, "source_choices", *sourceCut, family.sourceChoices) ||
        !readChoices(item, "target_choices", *targetCut, family.targetChoices) ||
        (local.getValue() && (!family.sourceChoices.empty() ||
            (!family.targetChoices.empty() && family.targetChoices.front() == -1)))) {
        return function.emitError("malformed endpoint alternative cut partition");
    }
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
        if (allocation->sourcePipe != *source || allocation->targetPipe != *target || allocation->ids.empty() ||
            (!nested && allocation->tupleRule &&
             (allocation->tupleRule->coordinateCount != 1 ||
              !validPhysicalTupleRule(*allocation->tupleRule, allocation->ids.size())))) {
            return function.emitError("endpoint-family member has invalid pipes, palette, or source-tuple rule");
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
    if (!version || (*version != 1 && *version != 2 && *version != 3 && *version != 4) ||
        !planId || *planId != plan.planId) {
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
        if (failed(readFamily(function, item, records, result, seen, *version == 4))) {
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
bool validSourceTuple(Operation* op, const Family& family)
{
    if (family.members.size() != 1 || op->getNumOperands() == 0 ||
        !llvm::all_of(op->getOperandTypes(), [](Type type) { return type.isIndex(); })) { return false; }
    const auto& allocation = *family.members.front();
    APInt identity;
    return !allocation.ids.empty() && allocation.stride % allocation.ids.size() == 0 &&
        matchPattern(op->getOperand(0), m_ConstantInt(&identity)) && identity.isZero();
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
using SideCuts = std::map<std::pair<int64_t, int64_t>, std::set<int64_t>>;
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
    SideCuts coverage;
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
            const auto& choices = *kind == 0 ? provenance.sourceChoices : provenance.targetChoices;
            auto expectedCut = *kind == 0 ? provenance.sourceCut : provenance.targetCut;
            const bool validCut = choices.empty() ? expectedCut == cut :
                llvm::is_contained(ArrayRef<int64_t>(choices).drop_front(), *cut);
            if (Namespace{provenance.sourcePipe, provenance.targetPipe, provenance.displacement} != key ||
                local != (*kind == 1) || !validCut || !coverage[{record, *kind}].insert(*cut).second) {
                return function.emitError("endpoint piece has repeated or inconsistent record coverage"), failure();
            }
            piece.records.push_back(record);
        }
        result.emplace(*id, std::move(piece));
    }
    for (const auto& entry : owners) {
        const bool local = entry.second->sourcePipe == entry.second->targetPipe;
        for (int64_t kind = 0; kind < 3; ++kind) {
            const auto& choices = kind == 0 ? entry.second->sourceChoices : entry.second->targetChoices;
            const bool required = local ? kind == 1 : kind != 1;
            const auto expected = required ? (choices.empty() ? 1U : choices.size() - 1) : 0U;
            if (coverage[{entry.first, kind}].size() != expected) {
                return function.emitError("endpoint pieces do not cover every original record endpoint"), failure();
            }
        }
    }
    return result;
}
bool validRecordMember(Operation* op, const Piece& piece, bool nested)
{
    if ((nested ? op->getNumOperands() < 2 : op->getNumOperands() != 2) ||
        !llvm::all_of(op->getOperandTypes(), [](Type type) { return type.isIndex(); })) {
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
    // Serialized nonnegative IDs can equal LLVM DenseMap sentinel values.
    // Preserve the metadata namespace and reject unknown IDs with ordinary lookup.
    std::map<int64_t, scf::ForOp> loops;
    std::map<int64_t, DictionaryAttr> members;
};
FailureOr<CoordinateProvenance> readCoordinates(func::FuncOp function)
{
    CoordinateProvenance result;
    auto walked = function.walk([&](scf::ForOp loop) -> WalkResult {
        if (!belongsToActiveContext(function, loop)) { return WalkResult::advance(); }
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
bool distinctCoordinateMembers(const Endpoint& endpoint, const CoordinateProvenance& provenance, bool publish)
{
    llvm::DenseSet<Attribute> seen;
    SmallVector<int64_t> axes;
    for (auto label : endpoint.labels) {
        auto found = provenance.members.find(static_cast<int64_t>(label));
        if (found == provenance.members.end()) { return false; }
        auto tuple = found->second.getAs<ArrayAttr>(publish ? "source" : "target");
        if (!tuple || tuple.empty() || !seen.insert(tuple).second) { return false; }
        SmallVector<int64_t> current;
        for (auto attr : tuple) {
            auto coordinate = dyn_cast<DenseI64ArrayAttr>(attr);
            if (!coordinate || coordinate.size() != 2) { return false; }
            current.push_back(coordinate[0]);
        }
        if (axes.empty()) { axes = current; }
        if (axes != current) { return false; }
    }
    return !seen.empty();
}
LogicalResult planPointMapping(Endpoint& endpoint, ArrayRef<SmallVector<TemplateCoordinate>> points);
LogicalResult planCoordinatePhase(
    Endpoint& endpoint, const CoordinateProvenance& provenance, DominanceInfo& dominance, bool publish)
{
    SmallVector<SmallVector<TemplateCoordinate>> points;
    Operation* before = endpoint.operation;
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
            const auto found = provenance.loops.find(coordinate[0]);
            auto loop = found == provenance.loops.end() ? scf::ForOp{} : found->second;
            if (!loop || !dominance.properlyDominates(loop.getInductionVar(), before)) {
                return before->emitError("endpoint-family coordinate is unavailable at its command cut");
            }
            point.push_back({loop, coordinate[1]});
        }
        points.push_back(std::move(point));
    }
    return planPointMapping(endpoint, points);
}
LogicalResult planPointMapping(Endpoint& endpoint, ArrayRef<SmallVector<TemplateCoordinate>> points)
{
    if (!endpoint.tupleMembers.empty()) {
        const auto key = physicalMappingKey(*endpoint.tupleMembers.front());
        if (llvm::all_of(endpoint.tupleMembers, [&](const auto* member) {
                return physicalMappingKey(*member) == key;
            })) {
            // This whole family has one map. Its arbitrary record selector is
            // irrelevant to physical IDs, including for nested matching tuples.
            endpoint.tupleMembers.resize(1);
            return success();
        }
    }
    if (llvm::all_of(points, [](const auto& point) { return point.empty(); })) {
        // Old saved v3 fixtures can certify label selections without coordinate
        // provenance. They use the same exact label-to-phase expression adapter.
        return success();
    }
    auto mapping = std::make_unique<PhysicalIdRecipe>();
    SmallVector<int64_t> values;
    const bool generic = !endpoint.tupleMembers.empty();
    for (auto value : generic ? endpoint.labels : endpoint.phases) {
        values.push_back(static_cast<int64_t>(value));
    }
    auto ids = constantMemberIds(endpoint);
    auto commonTuple = ids.empty() ? commonTupleMapping(endpoint.tupleMembers) : std::nullopt;
    SmallVector<int64_t> palette = ids;
    llvm::sort(palette);
    palette.erase(std::unique(palette.begin(), palette.end()), palette.end());
    const bool directId = !ids.empty();
    if (directId) {
        // Compose with the final assignment before fitting coordinates. Record
        // labels are arbitrary provenance, not a runtime lookup-table index.
        // Palette ranks also support noncontiguous eligible hardware IDs.
        values.clear();
        for (auto id : ids) {
            values.push_back(std::lower_bound(palette.begin(), palette.end(), id) - palette.begin());
        }
    }
    if (commonTuple) { values = commonTuple->bases; }
    const uint64_t modulus = directId ? palette.size() : commonTuple ?
        static_cast<uint64_t>(*llvm::max_element(commonTuple->bases)) + 1 :
        generic ? 0 : endpoint.allocation->ids.size();
    auto result = buildFamilyExpressions(points, values, modulus, FamilyExpressionOutputs::Member);
    if (!result.error.empty()) {
        return endpoint.operation->emitError("invalid endpoint-family phase coordinates: ") << result.error;
    }
    mapping->selector = std::move(result.recipe);
    if (directId) {
        mapping->kind = PhysicalIdRecipe::Kind::Id;
        mapping->palette = std::move(palette);
    } else if (commonTuple) {
        mapping->kind = PhysicalIdRecipe::Kind::Tuple;
        mapping->tuple = std::move(commonTuple->dynamic);
        if (commonTuple->varyingPhases) {
            for (std::size_t i = 0; i < commonTuple->phases.size(); ++i) {
                auto phase = buildFamilyExpressions(
                    points, commonTuple->phases[i], mapping->tuple.terms[i].modulus, FamilyExpressionOutputs::Member);
                if (!phase.error.empty()) {
                    return endpoint.operation->emitError("invalid endpoint-family tuple phase: ") << phase.error;
                }
                mapping->phases.push_back(std::move(phase.recipe));
            }
        }
    } else {
        mapping->kind = generic ? PhysicalIdRecipe::Kind::Record : PhysicalIdRecipe::Kind::Phase;
    }
    endpoint.mapping = std::move(mapping);
    if (commonTuple && commonTuple->varyingPhases) {
        Endpoint incumbent{endpoint.operation, endpoint.allocation, {}, {}};
        incumbent.labels = endpoint.labels;
        incumbent.tupleMembers = endpoint.tupleMembers;
        auto candidateCost = mappingCost(ArrayRef<Endpoint>(&endpoint, 1));
        auto incumbentCost = mappingCost(ArrayRef<Endpoint>(&incumbent, 1));
        if (!candidateCost || !incumbentCost || *candidateCost >= *incumbentCost) {
            endpoint.mapping.reset();
        }
    }
    return success();
}
FailureOr<SmallVector<Endpoint>> preflightPieces(func::FuncOp function, const PhysicalAllocationPlan& plan,
                                               const Records& records, const Families& families)
{
    auto pieces = readPieces(function, families);
    if (failed(pieces)) {
        return failure();
    }
    const bool nested = number(function->getAttrOfType<DictionaryAttr>("pto.endpoint_families"), "version") == 4;
    auto coordinates = readCoordinates(function);
    if (failed(coordinates)) { return failure(); }
    std::map<std::pair<int64_t, int64_t>, SmallVector<Operation*>> actualChoices;
    std::map<int64_t, unsigned> coordinateCounts;
    DominanceInfo dominance(function);
    SmallVector<Endpoint> endpoints;
    llvm::DenseSet<int64_t> seen;
    auto walked = function.walk([&](Operation* op) -> WalkResult {
        if (!belongsToActiveContext(function, op)) { return WalkResult::advance(); }
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
            !target || static_cast<int64_t>(target.getPipe()) != piece.targetPipe ||
            !validRecordMember(op, piece, nested)) {
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
            if (!member || member->ids.empty() || (!nested && member->tupleRule &&
                (member->tupleRule->coordinateCount != 1 ||
                 !validPhysicalTupleRule(*member->tupleRule, member->ids.size())))) {
                op->emitError("executable piece has inconsistent member allocations");
                return WalkResult::interrupt();
            }
            if (nested) {
                const auto count = op->getNumOperands() - 1;
                auto [entry, added] = coordinateCounts.emplace(record, count);
                if ((!added && entry->second != count) ||
                    (member->tupleRule ? member->tupleRule->coordinateCount != count ||
                        !validPhysicalTupleRule(*member->tupleRule, member->ids.size()) :
                        member->stride % member->ids.size() != 0)) {
                    op->emitError("nested allocation requires a certified rule with matching source-tuple arity");
                    return WalkResult::interrupt();
                }
            }
            actualChoices[{record, piece.kind}].push_back(op);
            endpoint.tupleMembers.push_back(member);
            endpoint.phases.push_back(member->phase % member->ids.size());
            endpoint.labels.push_back(static_cast<uint64_t>(record));
        }
        if (!nested && commonPalette(endpoint.tupleMembers)) { endpoint.tupleMembers.clear(); }
        const bool composedCoordinates = nested &&
            (!constantMemberIds(endpoint).empty() || commonTupleMapping(endpoint.tupleMembers)) &&
            distinctCoordinateMembers(endpoint, *coordinates, publish);
        if ((!nested || composedCoordinates) &&
            failed(planCoordinatePhase(endpoint, *coordinates, dominance, publish))) {
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
    for (const auto& [id, family] : families) {
        (void)id;
        if (family.sourcePipe == family.targetPipe) { continue; }
        for (int64_t kind : {0, 2}) {
            const auto& choices = kind == 0 ? family.sourceChoices : family.targetChoices;
            if (choices.empty()) { continue; }
            const auto found = coordinates->loops.find(choices.front());
            auto loop = found == coordinates->loops.end() ? scf::ForOp{} : found->second;
            for (auto record : family.originalRecords) {
                const auto& commands = actualChoices[{record, kind}];
                if (choices.front() == -1) {
                    const auto& publications = actualChoices[{record, 0}];
                    auto publication = publications.size() == 1 ?
                        dyn_cast_or_null<scf::IfOp>(publications.front()->getParentOp()) : scf::IfOp{};
                    auto* terminal = function.front().getTerminator();
                    if (!publication || !qualifyTerminalCleanupCuts(
                        {{publication->getBlock(), publication}, {terminal->getBlock(), terminal}},
                        publication.getCondition())) {
                        return function.emitError("terminal cleanup requires one selected publication per invocation"),
                            failure();
                    }
                }
                if (!(choices.front() == -1 ? validateTerminalCleanupCommands(function, commands) :
                      validateEndpointCommandChoices(loop, commands))) {
                    return function.emitError("endpoint alternatives do not partition original branch execution"),
                        failure();
                }
            }
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
    if (metadata && (number(metadata, "version") == 3 || number(metadata, "version") == 4)) {
        return preflightPieces(function, plan, records, *families);
    }
    auto certificate = function->getAttrOfType<DictionaryAttr>(CyclicAllocationAttr);
    auto strategy = certificate ? certificate.getAs<StringAttr>("strategy") : StringAttr{};
    const bool sourceTuples = strategy && strategy.getValue() == "dedicated-families";
    SmallVector<Endpoint> endpoints;
    auto walked = function.walk([&](Operation* op) -> WalkResult {
        if (!belongsToActiveContext(function, op)) { return WalkResult::advance(); }
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
            static_cast<uint32_t>(target.getPipe()) != allocation->targetPipe ||
            !(sourceTuples ? validSourceTuple(op, family) : validMember(op, family))) {
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
            if (member->ids.empty() || (member->tupleRule &&
                (member->tupleRule->coordinateCount != 1 ||
                 !validPhysicalTupleRule(*member->tupleRule, member->ids.size())))) {
                op->emitError("flat logical endpoint requires a one-coordinate allocation rule");
                return WalkResult::interrupt();
            }
            auto& count = counts[member->record];
            unsigned& occurrences = publish ? count.first : count.second;
            if (++occurrences != 1) {
                op->emitError("cyclic allocation expects one static SET and WAIT per family member");
                return WalkResult::interrupt();
            }
            endpoint.phases.push_back(member->phase % member->ids.size());
            endpoint.labels.push_back(endpoint.labels.size());
            endpoint.tupleMembers.push_back(member);
        }
        if (commonPalette(endpoint.tupleMembers)) { endpoint.tupleMembers.clear(); }
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
Value memberPhase(OpBuilder& builder, const Endpoint& endpoint, Value member)
{
    auto location = endpoint.operation->getLoc();
    auto number = [&](uint64_t value) -> Value {
        return builder.create<arith::ConstantIndexOp>(location, static_cast<int64_t>(value));
    };
    const auto& phases = endpoint.phases;
    if (phases.size() == 1) {
        return number(phases.front());
    }
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
    return selectPhysicalPalette(builder, location, phase, record.ids);
}
std::optional<int64_t> constantTupleId(const PhysicalRecordAllocation& record)
{
    if (!record.tupleRule) {
        return record.stride % record.ids.size() == 0 ?
            std::optional<int64_t>(record.ids[record.phase % record.ids.size()]) : std::nullopt;
    }
    uint64_t offset = record.tupleRule->base;
    for (const auto& term : record.tupleRule->terms) {
        if (!term.scale) { continue; }
        if (term.stride % term.modulus) { return std::nullopt; }
        offset += term.scale * (term.phase % term.modulus);
    }
    return record.ids[offset];
}
Value tuplePhysicalId(OpBuilder& builder, const Endpoint& endpoint, ArrayRef<Value> coordinates, Value member)
{
    auto* op = endpoint.operation;
    auto location = op->getLoc();
    std::map<uint64_t, Value> constants;
    auto constant = [&](uint64_t value) -> Value {
        auto [entry, added] = constants.try_emplace(value);
        if (added) { entry->second = builder.create<arith::ConstantIndexOp>(location, static_cast<int64_t>(value)); }
        return entry->second;
    };
    using TermKey = std::tuple<uint64_t, uint64_t, uint64_t, uint64_t>;
    std::map<TermKey, Value> terms;
    std::map<std::vector<uint64_t>, Value> mappings;
    auto tupleOffset = [&](const PhysicalTupleRule& rule, Value offset, ArrayRef<Value> phases = {}) {
        for (auto [index, term] : llvm::enumerate(rule.terms)) {
            if (!term.scale || term.modulus == 1) {
                continue;
            }
            const auto stride = term.stride % term.modulus, phase = term.phase % term.modulus;
            if (!phases.empty()) {
                // Selected phases are exact on the supplied member domain.
                // Reduce the source coordinate first, as for static phases.
                Value value = phases[index];
                if (stride) {
                    auto coordinate = coordinates[term.coordinate];
                    auto modulus = constant(term.modulus);
                    Value product = builder.create<arith::RemUIOp>(location, coordinate, modulus);
                    if (stride != 1) {
                        product = builder.create<arith::MulIOp>(location, product, constant(stride));
                    }
                    value = builder.create<arith::AddIOp>(location, product, value);
                    value = builder.create<arith::RemUIOp>(location, value, modulus);
                }
                if (term.scale != 1) {
                    value = builder.create<arith::MulIOp>(location, value, constant(term.scale));
                }
                offset = builder.create<arith::AddIOp>(location, offset, value);
                continue;
            }
            const TermKey key{term.coordinate, stride, phase, term.modulus};
            auto [entry, added] = terms.try_emplace(key);
            if (added) {
                Value value = constant(phase);
                if (stride) {
                    auto coordinate = coordinates[term.coordinate];
                    auto modulus = constant(term.modulus);
                    auto residue = builder.create<arith::RemUIOp>(location, coordinate, modulus);
                    Value product = residue;
                    if (stride != 1) {
                        product = builder.create<arith::MulIOp>(location, product, constant(stride));
                    }
                    if (phase) {
                        product = builder.create<arith::AddIOp>(location, product, constant(phase));
                    }
                    value = builder.create<arith::RemUIOp>(location, product, modulus);
                }
                entry->second = value;
            }
            Value value = entry->second;
            if (term.scale != 1) {
                value = builder.create<arith::MulIOp>(location, value, constant(term.scale));
            }
            offset = builder.create<arith::AddIOp>(location, offset, value);
        }
        return offset;
    };
    // Compose source selection with the common nonconstant skeleton before
    // lowering a dispatcher. Exact bases/phases use original coordinates.
    if (endpoint.composedTuple) {
        auto offset = tupleOffset(endpoint.composedTuple->rule, endpoint.composedTuple->base,
                                  endpoint.composedTuple->phases);
        return selectPhysicalPalette(builder, location, offset, endpoint.allocation->ids);
    }
    auto memberId = [&](const PhysicalRecordAllocation& member) -> Value {
        if (auto fixed = constantTupleId(member)) { return constant(*fixed); }
        // Flat grouped families keep their original source ordinal in operand
        // zero. The same bounded arithmetic serves nested tuple families.
        PhysicalTupleRule flatRule{1, 0, {{0, member.stride, member.phase, member.ids.size(), 1}}};
        const auto& rule = member.tupleRule ? *member.tupleRule : flatRule;
        auto [cached, inserted] = mappings.try_emplace(physicalMappingKey(member));
        if (!inserted) { return cached->second; }
        // Checked term bounds keep offsets within the palette. Products use
        // reduced residues, so large original coordinates cannot wrap.
        Value offset = tupleOffset(rule, constant(rule.base));
        cached->second = selectPhysicalPalette(builder, location, offset, member.ids);
        return cached->second;
    };
    Value fallback = memberId(*endpoint.tupleMembers.front());
    Value result = fallback;
    for (std::size_t i = 1; i < endpoint.tupleMembers.size(); ++i) {
        Value selectedId = memberId(*endpoint.tupleMembers[i]);
        if (selectedId == fallback) { continue; }
        Value label = endpoint.directPhase ? endpoint.directPhase : member;
        auto selected = builder.create<arith::CmpIOp>(location, arith::CmpIPredicate::eq,
            label, constant(endpoint.labels[i]));
        result = builder.create<arith::SelectOp>(location, selected, selectedId, result);
    }
    return result;
}
// Score complete live map recipes without constructing scratch SSA. The
// deterministic estimate charges selects and non-power-of-two remainders more
// heavily; it is not an exact instruction count or a hardware latency model.
std::optional<uint64_t> mappingCost(ArrayRef<Endpoint> endpoints)
{
    uint64_t cost = 0;
    bool valid = true;
    auto add = [&](uint64_t amount) {
        if (amount > uint64_t(INT64_MAX) - cost) {
            valid = false;
            return;
        }
        cost += amount;
    };
    auto rem = [](uint64_t modulus) { return modulus && !(modulus & (modulus - 1)) ? 1U : 8U; };
    auto palette = [&](ArrayRef<int64_t> ids) {
        if (ids.size() <= 1) {
            return;
        }
        const bool contiguous = llvm::all_of(llvm::enumerate(ids), [&](auto entry) {
            return entry.value() == ids.front() + static_cast<int64_t>(entry.index());
        });
        add(contiguous ? uint64_t(ids.front() != 0) : 4 * (ids.size() - 1));
    };
    auto selector = [&](const FamilyExpressionRecipe& recipe) {
        auto estimate = estimateFamilyExpressions(recipe);
        if (!estimate) {
            valid = false;
            return;
        }
        add(estimate->weighted);
    };
    auto tuple = [&](const PhysicalTupleRule& rule, bool varying) {
        std::set<std::array<uint64_t, 4>> terms;
        for (const auto& term : rule.terms) {
            if (!term.scale || term.modulus == 1) {
                continue;
            }
            const auto stride = term.stride % term.modulus, phase = term.phase % term.modulus;
            if (varying || terms.insert({term.coordinate, stride, phase, term.modulus}).second) {
                if (stride) {
                    add(2 * rem(term.modulus) + uint64_t(stride != 1) + uint64_t(varying || phase != 0));
                }
            }
            add(uint64_t(term.scale != 1) + 1);
        }
    };
    for (const auto& endpoint : endpoints) {
        const auto* mapping = endpoint.mapping.get();
        if (mapping) {
            std::set<std::vector<uint64_t>> selected;
            auto unique = [&](const FamilyExpressionRecipe& recipe) {
                SmallVector<uint64_t> key;
                appendFamilyExpressionKey(recipe, key);
                if (selected.insert(std::vector<uint64_t>(key.begin(), key.end())).second) {
                    selector(recipe);
                }
            };
            unique(*mapping->selector);
            for (const auto& phase : mapping->phases) {
                unique(*phase);
            }
            if (mapping->kind == PhysicalIdRecipe::Kind::Id) {
                palette(mapping->palette);
                continue;
            }
            if (mapping->kind == PhysicalIdRecipe::Kind::Tuple) {
                tuple(mapping->tuple, !mapping->phases.empty());
                palette(endpoint.allocation->ids);
                continue;
            }
        }
        if (!endpoint.tupleMembers.empty()) {
            std::set<std::vector<uint64_t>> maps;
            const auto fallback = physicalMappingKey(*endpoint.tupleMembers.front());
            for (std::size_t i = 0; i < endpoint.tupleMembers.size(); ++i) {
                const auto& member = *endpoint.tupleMembers[i];
                const auto key = physicalMappingKey(member);
                if (i && key != fallback) {
                    add(4);
                }
                if (!maps.insert(key).second || constantTupleId(member)) {
                    continue;
                }
                PhysicalTupleRule flat{1, 0, {{0, member.stride, member.phase, member.ids.size(), 1}}};
                tuple(member.tupleRule ? *member.tupleRule : flat, false);
                palette(member.ids);
            }
            continue;
        }
        const auto& record = *endpoint.allocation;
        if (record.ids.size() <= 1) {
            continue;
        }
        if (!mapping) {
            // A saved family without coordinate evidence keeps its certified
            // finite member dispatcher. Equal phases need no selection work.
            for (auto phase : ArrayRef<uint64_t>(endpoint.phases).drop_front()) {
                if (phase != endpoint.phases.front()) {
                    add(4);
                }
            }
        }
        const auto stride = record.stride % record.ids.size();
        if (stride) {
            add(2 * rem(record.ids.size()) + uint64_t(stride != 1) + 1);
        }
        palette(record.ids);
    }
    return valid ? std::optional(cost) : std::nullopt;
}
void emitMapSelectors(OpBuilder& builder, Location location, Endpoint& endpoint, const PhysicalIdRecipe& mapping)
{
    std::map<std::vector<uint64_t>, Value> values;
    auto emit = [&](const FamilyExpressionRecipe& recipe) {
        SmallVector<uint64_t> key;
        appendFamilyExpressionKey(recipe, key);
        auto [entry, added] = values.try_emplace(std::vector<uint64_t>(key.begin(), key.end()));
        if (added) {
            auto selected = emitFamilyExpressions(builder, location, recipe);
            assert(selected.error.empty() && selected.member && "validated family map must lower");
            entry->second = selected.member;
        }
        return entry->second;
    };
    const auto selected = emit(*mapping.selector);
    if (mapping.kind == PhysicalIdRecipe::Kind::Id) {
        endpoint.directId = selectPhysicalPalette(builder, location, selected, mapping.palette);
    } else if (mapping.kind == PhysicalIdRecipe::Kind::Tuple) {
        endpoint.composedTuple = std::make_unique<ComposedTupleMapping>();
        endpoint.composedTuple->rule = mapping.tuple;
        endpoint.composedTuple->base = selected;
        for (const auto& phase : mapping.phases) {
            endpoint.composedTuple->phases.push_back(emit(*phase));
        }
    } else {
        endpoint.directPhase = selected;
    }
}
Value emitEndpointId(OpBuilder& builder, const Endpoint& endpoint, ArrayRef<Value> tuple, Value member)
{
    if (endpoint.directId) {
        return endpoint.directId;
    }
    if (!endpoint.tupleMembers.empty()) {
        return tuplePhysicalId(builder, endpoint, tuple, member);
    }
    const auto phase = endpoint.directPhase ? endpoint.directPhase : memberPhase(builder, endpoint, member);
    return physicalId(builder, endpoint.operation->getLoc(), tuple.front(), *endpoint.allocation, phase);
}
LogicalResult materializePhysicalMapping(Endpoint& endpoint, DominanceInfo& dominance)
{
    if (!endpoint.mapping) {
        return success();
    }
    auto* before = endpoint.operation;
    auto block = std::make_unique<Block>();
    OpBuilder builder(before->getContext());
    builder.setInsertionPointToEnd(block.get());
    const auto& mapping = *endpoint.mapping;
    emitMapSelectors(builder, before->getLoc(), endpoint, mapping);
    endpoint.phaseBefore = before;
    auto branch = dyn_cast<scf::IfOp>(before->getParentOp());
    const auto estimate = estimateFamilyExpressions(*mapping.selector);
    // Only standalone small selectors can move to a dominating command guard.
    // Tuple/ID maps remain conditional; complete cheap expressions still share
    // through CheapIdSharing after the whole selected map is lowered.
    const bool cheap =
        estimate && estimate->cheap && mapping.phases.empty() &&
        (mapping.kind == PhysicalIdRecipe::Kind::Phase || mapping.kind == PhysicalIdRecipe::Kind::Record);
    const bool available =
        branch && llvm::all_of(*block, [&](Operation& operation) {
            return llvm::all_of(operation.getOperands(), [&](Value operand) {
                return operand.getParentBlock() == block.get() || dominance.properlyDominates(operand, branch);
            });
        });
    if (cheap && available) {
        endpoint.phaseBefore = branch;
    }
    endpoint.phaseCode = std::move(block);
    return success();
}
bool candidateFits(DictionaryAttr candidate, std::size_t capacity)
{
    auto directions = candidate.getAs<ArrayAttr>("directions");
    if (!directions) { return false; }
    return llvm::all_of(directions, [&](Attribute raw) {
        auto direction = dyn_cast<DictionaryAttr>(raw);
        auto budget = direction ? direction.getAs<IntegerAttr>("budget") : IntegerAttr{};
        return budget && budget.getType().isInteger(64) && budget.getInt() >= 0 &&
            static_cast<uint64_t>(budget.getInt()) <= capacity;
    });
}
// Share the complete small ID calculation, including tuple rules and palette
// selection. Only generated, total integer arithmetic crosses a command guard.
// Original payloads and expensive selectors remain at their original positions.
class CheapIdSharing {
public:
    CheapIdSharing(const llvm::DenseSet<Operation*>& original, DominanceInfo& dominance)
        : original(original), dominance(dominance) {}
    Value share(Value id, Operation* command)
    {
        auto branch = dyn_cast<scf::IfOp>(command->getParentOp());
        if (!branch || !branch->hasAttr("pto.endpoint_cut")) { return id; }
        SmallVector<Operation*> ordered;
        llvm::DenseSet<Operation*> seen;
        unsigned cost = 0;
        std::function<bool(Value)> collect = [&](Value value) {
            if (dominance.properlyDominates(value, branch)) { return true; }
            auto* op = value.getDefiningOp();
            if (!op || original.contains(op) || !branch->isProperAncestor(op)) { return false; }
            if (!seen.insert(op).second) { return true; }
            if (seen.size() > 64 || (!isa<arith::ConstantOp>(op) && ++cost > 8)) { return false; }
            if (!isa<arith::ConstantOp, arith::AddIOp, arith::SubIOp, arith::MulIOp,
                     arith::RemUIOp, arith::CmpIOp, arith::SelectOp, arith::IndexCastOp>(op)) { return false; }
            if (auto rem = dyn_cast<arith::RemUIOp>(op)) {
                APInt modulus;
                if (!matchPattern(rem.getRhs(), m_ConstantInt(&modulus)) || !modulus.isPowerOf2()) { return false; }
            }
            for (auto operand : op->getOperands()) { if (!collect(operand)) { return false; } }
            ordered.push_back(op);
            return true;
        };
        if (!collect(id) || ordered.empty()) { return id; }
        OpBuilder builder(branch);
        IRMapping mapping;
        auto& buckets = cuts[branch->getBlock()];
        for (auto* operation : ordered) {
            auto* copy = builder.clone(*operation, mapping);
            auto hash = static_cast<std::size_t>(OperationEquivalence::computeHash(copy,
                OperationEquivalence::directHashValue, OperationEquivalence::ignoreHashValue,
                OperationEquivalence::IgnoreLocations));
            auto& candidates = buckets[hash];
            auto found = llvm::find_if(candidates, [&](Operation* previous) {
                return dominance.properlyDominates(previous, branch) &&
                    OperationEquivalence::isEquivalentTo(previous, copy, OperationEquivalence::IgnoreLocations);
            });
            if (found == candidates.end()) { candidates.push_back(copy); }
            else {
                for (auto [old, shared] : llvm::zip(operation->getResults(), (*found)->getResults())) {
                    mapping.map(old, shared);
                }
                copy->erase();
            }
        }
        auto result = mapping.lookupOrDefault(id);
        // These operations have no users except generated ID arithmetic until
        // the allocated command is emitted below. Remove unused originals now.
        for (auto* operation : llvm::reverse(ordered)) {
            if (operation->use_empty()) { operation->erase(); }
        }
        return result;
    }
private:
    const llvm::DenseSet<Operation*>& original;
    DominanceInfo& dominance;
    DenseMap<Block*, DenseMap<std::size_t, SmallVector<Operation*>>> cuts;
};
void compactGeneratedArithmetic(func::FuncOp function, const llvm::DenseSet<Operation*>& original,
                                DominanceInfo& dominance)
{
    // Allocation owns only its new arithmetic. Whole-function CSE also deletes
    // unused reads and merges original payloads, invalidating occurrence identities.
    // Each block has its own hash directory; no branch or invocation is crossed.
    // Expected work is linear in visited syntax and generated expression size,
    // plus equivalence checks within hash-collision buckets; no trip expansion.
    // Original pointer membership is tested without dereferencing erased commands.
    function.walk([&](Operation* owner) {
        for (auto& region : owner->getRegions()) {
            for (auto& block : region) {
                std::unordered_map<std::size_t, SmallVector<Operation*>> known;
                for (auto& operation : llvm::make_early_inc_range(block)) {
                    auto* op = &operation;
                    if (original.contains(op) || op->getNumRegions() || !op->getNumResults() ||
                        op->getName().getDialectNamespace() != "arith" || !isMemoryEffectFree(op) ||
                        llvm::any_of(op->getUsers(), [&](Operation* user) { return original.contains(user); })) {
                        continue;
                    }
                    const auto hash = static_cast<std::size_t>(OperationEquivalence::computeHash(op,
                        OperationEquivalence::directHashValue, OperationEquivalence::ignoreHashValue,
                        OperationEquivalence::IgnoreLocations));
                    auto& candidates = known[hash];
                    auto found = llvm::find_if(candidates, [&](Operation* previous) {
                        return dominance.properlyDominates(previous, op) && OperationEquivalence::isEquivalentTo(
                            previous, op, OperationEquivalence::IgnoreLocations);
                    });
                    if (found == candidates.end()) { candidates.push_back(op); }
                    else {
                        op->replaceAllUsesWith((*found)->getResults());
                        op->erase();
                    }
                }
            }
        }
    });
}
} // namespace
bool physicalCandidateFits(DictionaryAttr certificate, std::size_t capacity)
{
    return candidateFits(certificate, capacity);
}
LogicalResult preflightPhysicalOutput(func::FuncOp function, DictionaryAttr certificate)
{
    if (!function) {
        return failure();
    }
    const auto width = DataLayout::closest(function).getTypeSizeInBits(IndexType::get(function.getContext()));
    if (width.isScalable() || width.getFixedValue() != 64) {
        return function.emitError("physical allocation requires 64-bit occurrence indices");
    }
    bool hidden = false, manual = false;
    function.walk([&](Operation* op) {
        if (!belongsToActiveContext(function, op)) {
            return;
        }
        auto model = getSyncMacroModel(op);
        hidden |= model && !model->hiddenEvents.empty();
        manual |= isa<SetFlagOp, WaitFlagOp, SetFlagDynOp, WaitFlagDynOp, RecordEventOp, WaitEventOp>(op);
    });
    if (manual) {
        return function.emitError("physical allocation cannot share its supplied IDs with existing synchronization");
    }
    auto kind = certificate ? certificate.getAs<StringAttr>("kind") : StringAttr{};
    if (hidden && (!kind || kind.getValue() != "finite" || !certificate.getAs<UnitAttr>("macro_reservations"))) {
        return function.emitError("macro hidden-event allocation certificate not implemented for this route");
    }
    return success();
}
FailureOr<std::shared_ptr<const PhysicalEndpointMap>> planPhysicalEndpointMap(
    Operation* cut, ArrayRef<PhysicalRecordAllocation> members, ArrayRef<SmallVector<TemplateCoordinate>> points,
    uint64_t tupleArity, bool nested, DominanceInfo& dominance)
{
    if (!cut || members.empty() || !tupleArity || points.size() != members.size()) {
        return failure();
    }
    auto result = std::make_shared<PhysicalEndpointMap>();
    result->records.assign(members.begin(), members.end());
    result->arity = tupleArity;
    auto& endpoint = result->endpoint;
    endpoint.operation = cut;
    endpoint.allocation = &result->records.front();
    std::set<int64_t> labels;
    for (const auto& record : result->records) {
        std::set<int64_t> palette;
        for (auto id : record.ids) {
            if (id < 0 || id > 5 || !palette.insert(id).second) {
                return failure();
            }
        }
        if (record.sourcePipe == record.targetPipe || record.sourcePipe > static_cast<uint32_t>(PIPE::PIPE_FIX) ||
            record.targetPipe > static_cast<uint32_t>(PIPE::PIPE_FIX) ||
            record.sourcePipe == static_cast<uint32_t>(PIPE::PIPE_ALL) ||
            record.targetPipe == static_cast<uint32_t>(PIPE::PIPE_ALL)) {
            return failure();
        }
        if (record.record < 0 || !labels.insert(record.record).second || record.ids.empty() ||
            record.sourcePipe != members.front().sourcePipe || record.targetPipe != members.front().targetPipe ||
            (record.tupleRule ? record.tupleRule->coordinateCount != tupleArity ||
                                    !validPhysicalTupleRule(*record.tupleRule, record.ids.size()) :
                                nested && record.stride % record.ids.size() != 0)) {
            return failure();
        }
        endpoint.labels.push_back(record.record);
        endpoint.phases.push_back(record.phase % record.ids.size());
        endpoint.tupleMembers.push_back(&record);
    }
    if (!nested && commonPalette(endpoint.tupleMembers)) {
        endpoint.tupleMembers.clear();
    }
    bool distinct = true;
    std::set<std::vector<std::pair<Operation*, int64_t>>> seen;
    SmallVector<Operation*> axes;
    for (const auto& point : points) {
        std::vector<std::pair<Operation*, int64_t>> key;
        SmallVector<Operation*> current;
        for (auto coordinate : point) {
            auto loop = coordinate.loop;
            if (!loop || !dominance.properlyDominates(loop.getInductionVar(), cut)) {
                return failure();
            }
            current.push_back(loop.getOperation());
            key.push_back({loop.getOperation(), coordinate.induction});
        }
        if (axes.empty()) {
            axes = current;
        }
        distinct &= !point.empty() && axes == current && seen.insert(key).second;
    }
    if ((!nested || distinct) && failed(planPointMapping(endpoint, points))) {
        return failure();
    }
    // Coordinate-free identical maps still discard the arbitrary record label.
    if (!endpoint.mapping && !endpoint.tupleMembers.empty()) {
        const auto first = physicalMappingKey(*endpoint.tupleMembers.front());
        if (llvm::all_of(
                endpoint.tupleMembers, [&](const auto* record) { return physicalMappingKey(*record) == first; })) {
            endpoint.tupleMembers.resize(1);
        }
    }
    return std::shared_ptr<const PhysicalEndpointMap>(std::move(result));
}
uint64_t physicalMapTupleArity(const PhysicalEndpointMap& mapping) { return mapping.arity; }
bool physicalMapRequiresMember(const PhysicalEndpointMap& mapping)
{
    const auto& endpoint = mapping.endpoint;
    if (endpoint.mapping) {
        return false;
    }
    return endpoint.tupleMembers.empty() ? endpoint.phases.size() > 1 : endpoint.tupleMembers.size() > 1;
}
bool physicalMapNeedsCoordinate(const PhysicalEndpointMap& mapping, uint64_t coordinate)
{
    if (coordinate >= mapping.arity) {
        return false;
    }
    const auto& endpoint = mapping.endpoint;
    auto needs = [&](const PhysicalTupleRule& rule) {
        return llvm::any_of(rule.terms, [&](const auto& term) {
            return term.coordinate == coordinate && term.scale && term.modulus > 1 && term.stride % term.modulus;
        });
    };
    if (endpoint.mapping) {
        const auto& recipe = *endpoint.mapping;
        if (recipe.kind == PhysicalIdRecipe::Kind::Id) {
            return false;
        }
        if (recipe.kind == PhysicalIdRecipe::Kind::Tuple) {
            return needs(recipe.tuple);
        }
        if (recipe.kind == PhysicalIdRecipe::Kind::Phase) {
            return coordinate == 0 && endpoint.allocation->stride % endpoint.allocation->ids.size();
        }
    }
    if (endpoint.tupleMembers.empty()) {
        return coordinate == 0 && endpoint.allocation->stride % endpoint.allocation->ids.size();
    }
    return llvm::any_of(endpoint.tupleMembers, [&](const auto* record) {
        return record->tupleRule ? needs(*record->tupleRule) : coordinate == 0 && record->stride % record->ids.size();
    });
}
bool equivalentPhysicalMaps(const PhysicalEndpointMap& a, const PhysicalEndpointMap& b)
{
    if (&a == &b) {
        return true;
    }
    if (a.arity != b.arity || physicalMapRequiresMember(a) || physicalMapRequiresMember(b)) {
        return false;
    }
    auto key = [](const PhysicalEndpointMap& map) {
        const auto& endpoint = map.endpoint;
        SmallVector<uint64_t> result{map.arity, endpoint.allocation->sourcePipe, endpoint.allocation->targetPipe};
        const auto* recipe = endpoint.mapping.get();
        result.push_back(bool(recipe));
        if (!recipe) {
            if (endpoint.tupleMembers.empty()) {
                auto record = *endpoint.allocation;
                record.phase = endpoint.phases.front();
                auto identity = physicalMappingKey(record);
                result.append(identity.begin(), identity.end());
            } else {
                auto identity = physicalMappingKey(*endpoint.tupleMembers.front());
                result.append(identity.begin(), identity.end());
            }
            return result;
        }
        result.push_back(uint64_t(recipe->kind));
        appendFamilyExpressionKey(*recipe->selector, result);
        result.push_back(recipe->phases.size());
        for (const auto& phase : recipe->phases) {
            appendFamilyExpressionKey(*phase, result);
        }
        result.append({recipe->tuple.coordinateCount, recipe->tuple.base, recipe->tuple.terms.size()});
        for (const auto& term : recipe->tuple.terms) {
            result.append({term.coordinate, term.stride, term.phase, term.modulus, term.scale});
        }
        const auto& palette = recipe->kind == PhysicalIdRecipe::Kind::Id ? recipe->palette : endpoint.allocation->ids;
        result.push_back(palette.size());
        for (auto id : palette) {
            result.push_back(id);
        }
        if (recipe->kind == PhysicalIdRecipe::Kind::Phase) {
            result.push_back(endpoint.allocation->stride % palette.size());
        }
        if (recipe->kind == PhysicalIdRecipe::Kind::Record) {
            result.push_back(endpoint.tupleMembers.size());
            for (auto [index, member] : llvm::enumerate(endpoint.tupleMembers)) {
                auto identity = physicalMappingKey(*member);
                result.append({endpoint.labels[index], identity.size()});
                result.append(identity.begin(), identity.end());
            }
        }
        return result;
    };
    return key(a) == key(b);
}
std::optional<uint64_t> physicalMapCost(const PhysicalEndpointMap& mapping)
{
    return mappingCost(ArrayRef<Endpoint>(&mapping.endpoint, 1));
}
Value emitPhysicalEndpointMap(
    OpBuilder& builder, Location location, const PhysicalEndpointMap& mapping, ArrayRef<Value> tuple, Value member)
{
    assert(tuple.size() == mapping.arity && "physical matching tuple changed after preflight");
    assert((!physicalMapRequiresMember(mapping) || member) && "physical record selector is required");
    assert(llvm::all_of(llvm::enumerate(tuple), [&](auto input) {
        return !physicalMapNeedsCoordinate(mapping, input.index()) || input.value();
    }) && "physical map coordinate is required");
    Endpoint endpoint;
    endpoint.operation = mapping.endpoint.operation;
    endpoint.allocation = mapping.endpoint.allocation;
    endpoint.phases = mapping.endpoint.phases;
    endpoint.labels = mapping.endpoint.labels;
    endpoint.tupleMembers = mapping.endpoint.tupleMembers;
    if (mapping.endpoint.mapping) {
        emitMapSelectors(builder, location, endpoint, *mapping.endpoint.mapping);
    }
    return emitEndpointId(builder, endpoint, tuple, member);
}
void emitPhysicalCommand(
    OpBuilder& builder, Location location, bool publish, uint32_t sourcePipe, uint32_t targetPipe, Value id)
{
    auto source = PipeAttr::get(builder.getContext(), static_cast<PIPE>(sourcePipe));
    auto target = PipeAttr::get(builder.getContext(), static_cast<PIPE>(targetPipe));
    APInt number;
    if (matchPattern(id, m_ConstantInt(&number))) {
        auto event =
            EventAttr::get(builder.getContext(), *symbolizeEVENT(static_cast<uint32_t>(number.getZExtValue())));
        if (publish) {
            builder.create<SetFlagOp>(location, source, target, event);
        } else {
            builder.create<WaitFlagOp>(location, source, target, event);
        }
    } else {
        if (publish) {
            builder.create<SetFlagDynOp>(location, source, target, id);
        } else {
            builder.create<WaitFlagDynOp>(location, source, target, id);
        }
    }
}
LogicalResult allocatePhysicalEventIds(func::FuncOp function, ArrayRef<int64_t> eligibleIds)
{
    if (!function) {
        return failure();
    }
    if (function->hasAttr("pto.prefix_protocol")) {
        return function.emitError("invocation prefix physical allocation requires a missing lifetime export");
    }
    auto cyclic = function->getAttrOfType<DictionaryAttr>(CyclicAllocationAttr);
    auto strategy = cyclic ? cyclic.getAs<StringAttr>("strategy") : StringAttr{};
    if (strategy && strategy.getValue() == "executed-family-counters") {
        return allocateExecutedFamilyCounters(function, cyclic, eligibleIds);
    }
    auto certificate = function->getAttrOfType<DictionaryAttr>(FiniteAllocationAttr);
    if (!certificate) {
        certificate = cyclic;
    }
    if (failed(preflightPhysicalOutput(function, certificate))) {
        return failure();
    }
    if (auto families = function->getAttrOfType<DictionaryAttr>("pto.endpoint_families")) {
        if (auto version = families.get("protocol_version")) {
            auto value = dyn_cast<IntegerAttr>(version);
            if (!value || !value.getType().isInteger(64) || value.getInt() < 0 || value.getInt() > 1) {
                return function.emitError("unsupported normalized synchronization protocol version");
            }
        }
    }
    auto plan = decodePhysicalAllocation(function, eligibleIds);
    if (failed(plan)) {
        return failure();
    }
    auto endpoints = preflight(function, *plan);
    if (failed(endpoints)) {
        return failure();
    }
    auto cyclicEvidence = function->getAttrOfType<DictionaryAttr>(CyclicAllocationAttr);
    auto alternativeEvidence =
        cyclicEvidence ? cyclicEvidence.getAs<DictionaryAttr>("regular_candidate") : DictionaryAttr{};
    std::optional<uint64_t> originalCost, alternativeCost;
    bool choseAlternative = false;
    if (alternativeEvidence && candidateFits(alternativeEvidence, eligibleIds.size())) {
        ScopedDiagnosticHandler optionalDiagnostics(function.getContext(), [](Diagnostic&) { return success(); });
        auto alternative = decodeCyclicAllocationCertificate(function, alternativeEvidence, eligibleIds);
        if (succeeded(alternative) && alternative->planId == plan->planId) {
            auto alternativeEndpoints = preflight(function, *alternative);
            if (succeeded(alternativeEndpoints)) {
                originalCost = mappingCost(*endpoints);
                alternativeCost = mappingCost(*alternativeEndpoints);
                if (alternativeCost && originalCost && *alternativeCost < *originalCost) {
                    choseAlternative = true;
                    // Endpoints borrow candidate storage; moving the vector preserves it.
                    *endpoints = std::move(*alternativeEndpoints);
                    *plan = std::move(*alternative);
                }
            }
        }
    }
    DictionaryAttr mappingChoice;
    if (originalCost && alternativeCost) {
        Builder metadata(function.getContext());
        mappingChoice = metadata.getDictionaryAttr(
            {metadata.getNamedAttr("strategy", metadata.getStringAttr(choseAlternative ? "regular" : "cycle-cover")),
             metadata.getNamedAttr("cost_model", metadata.getStringAttr("structural-map-v1")),
             metadata.getNamedAttr("cycle_cost", metadata.getI64IntegerAttr(*originalCost)),
             metadata.getNamedAttr("regular_cost", metadata.getI64IntegerAttr(*alternativeCost))});
    }
    DominanceInfo dominance(function);
    for (auto& endpoint : *endpoints) {
        if (failed(materializePhysicalMapping(endpoint, dominance))) {
            return failure();
        }
    }
    if (mappingChoice) {
        function->setAttr("pto.physical_mapping_choice", mappingChoice);
    }
    llvm::DenseSet<Operation*> original;
    function.walk([&](Operation* operation) { original.insert(operation); });
    CheapIdSharing sharing(original, dominance);
    for (const auto& endpoint : *endpoints) {
        OpBuilder builder(endpoint.operation);
        // Keep allocation arithmetic conditional on the command. Dominance
        // permits hoisting, but does not make speculative dispatch profitable.
        if (endpoint.phaseCode) {
            auto* before = endpoint.phaseBefore;
            before->getBlock()->getOperations().splice(before->getIterator(), endpoint.phaseCode->getOperations());
            if (before == endpoint.operation) { builder.setInsertionPoint(endpoint.operation); }
        }
        SmallVector<Value> tuple{endpoint.operation->getOperand(0)};
        if (endpoint.operation->getNumOperands() > 2) {
            tuple.append(endpoint.operation->operand_begin() + 2, endpoint.operation->operand_end());
        }
        auto member = endpoint.operation->getNumOperands() > 1 ? endpoint.operation->getOperand(1) : Value{};
        auto eventId = emitEndpointId(builder, endpoint, tuple, member);
        eventId = sharing.share(eventId, endpoint.operation);
        builder.setInsertionPoint(endpoint.operation);
        auto source = endpoint.operation->getAttrOfType<PipeAttr>("src_pipe");
        auto target = endpoint.operation->getAttrOfType<PipeAttr>("dst_pipe");
        emitPhysicalCommand(
            builder, endpoint.operation->getLoc(), isa<LogicalSetOp>(endpoint.operation),
            static_cast<uint32_t>(source.getPipe()), static_cast<uint32_t>(target.getPipe()), eventId);
        original.erase(endpoint.operation);
        endpoint.operation->erase();
    }
    function.walk([&](Operation* op) {
        if (!belongsToActiveContext(function, op)) { return; }
        op->removeAttr("pto.endpoint_cut");
        op->removeAttr("pto.family_loop");
        op->removeAttr("pto.endpoint_piece");
    });
    function->removeAttr("pto.endpoint_families");
    function->removeAttr(CyclicAllocationAttr);
    function->removeAttr(FiniteAllocationAttr);
    if (!endpoints->empty() && !function->hasAttr(ActiveContextAttr)) {
        compactGeneratedArithmetic(function, original, dominance);
    }
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
        auto function = getOperation();
        if (function.isDeclaration()) { return; }
        if (function->hasAttr(frontiersynch::ContextPlansAttr)) {
            if (failed(frontiersynch::allocateContextSynchronization(function, eligibleIds))) { signalPassFailure(); }
            return;
        }
        if (hasManualOnCoreSynchronization(function)) {
            auto logical = function.walk([](Operation* op) {
                return isa<LogicalSetOp, LogicalWaitOp>(op)
                    ? WalkResult::interrupt() : WalkResult::advance();
            });
            // Analysis left a manual function untouched. A mixed logical and
            // physical plan must still pass the allocator's rejection checks.
            if (!logical.wasInterrupted()) { return; }
        }
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
