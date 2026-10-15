// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// A saved numerical family is executable logical mathematics, not a callback or
// a placeholder physical assignment. Decode and qualify it before emitting SSA.
#include "PTO/Transforms/FrontierSynch/Shared/DeferredFamily.h"
#include "PTO/Transforms/FrontierSynch/Shared/PhysicalAllocation.h"
#include "PTO/Transforms/FrontierSynch/Shared/ExecutionContexts.h"
#include "PTO/Transforms/FrontierSynch/Finite/FiniteAllocation.h"
#include "PTO/IR/PTO.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"
#include <array>
#include <set>
namespace mlir::pto::frontiersynch {
namespace {
constexpr StringLiteral recipeAttr = "pto.deferred_family";
constexpr StringLiteral loopAttr = "pto.deferred_family_loop";
constexpr StringLiteral cutAttr = "pto.deferred_family_cut";
bool fields(DictionaryAttr data, ArrayRef<StringRef> names)
{
    return data && data.size() == names.size() && llvm::all_of(names, [&](StringRef name) { return data.get(name); });
}
std::string metadataSeal(DictionaryAttr data)
{
    std::string text;
    llvm::raw_string_ostream stream(text);
    data.print(stream);
    llvm::SHA256 hash;
    hash.update(text);
    return llvm::toHex(hash.final());
}
bool boundedLoop(scf::ForOp loop)
{
    APInt lower, step, upper;
    if (!loop || !loop.getInductionVar().getType().isIndex() ||
        !matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) || !lower.isSignedIntN(64) || lower.isNegative() ||
        !matchPattern(loop.getStep(), m_ConstantInt(&step)) || !step.isSignedIntN(64) || !step.isStrictlyPositive()) {
        return false;
    }
    // Include the final increment: an ordinal bound alone does not exclude wrap.
    auto maximum = APInt(128, INT64_MAX);
    if (matchPattern(loop.getUpperBound(), m_ConstantInt(&upper))) {
        if (!upper.isSignedIntN(64)) { return false; }
        maximum = upper.sextOrTrunc(128);
    }
    auto start = lower.sextOrTrunc(128), stride = step.sextOrTrunc(128);
    if (maximum.sle(start)) { return true; }
    auto trips = (maximum - start - 1).udiv(stride) + 1;
    return (start + trips * stride).isSignedIntN(64);
}
std::optional<int64_t> integer(DictionaryAttr data, StringRef name)
{
    auto value = data ? data.getAs<IntegerAttr>(name) : IntegerAttr{};
    if (!value || !value.getType().isInteger(64)) { return std::nullopt; }
    return value.getInt();
}
bool nonnegative(std::optional<int64_t> value, uint64_t limit = INT64_MAX)
{
    return value && *value >= 0 && static_cast<uint64_t>(*value) <= limit;
}
bool sameCoordinates(ArrayRef<TemplateCoordinate> a, ArrayRef<TemplateCoordinate> b)
{
    return a.size() == b.size() && llvm::all_of(llvm::zip(a, b), [](auto pair) {
        return std::get<0>(pair).loop == std::get<1>(pair).loop &&
            std::get<0>(pair).induction == std::get<1>(pair).induction;
    });
}
LogicalResult validate(func::FuncOp function, const PreparedLogicalPlan& plan, const NumericalFamilyRecipe& recipe)
{
    if (function->hasAttr(ActiveContextAttr) || hasPhysicalSections(function)) {
        return function.emitError("deferred numerical families require one unprojected function context");
    }
    const auto width = DataLayout::closest(function).getTypeSizeInBits(IndexType::get(function.getContext()));
    if (width.isScalable() || width.getFixedValue() != 64 || plan.planId < 0 || recipe.pieces.empty() ||
        plan.families.empty() || !plan.allocationCertificate) {
        return function.emitError(
            "deferred numerical families require 64-bit indices, nonempty provenance and allocation evidence");
    }
    if (recipe.outer && recipe.outer->getParentOfType<func::FuncOp>() != function) { return failure(); }
    if (recipe.outer && !boundedLoop(recipe.outer)) {
        return function.emitError("deferred numerical recipe requires constant nonnegative lower, positive step "
                                  "and a nonwrapping final increment");
    }
    struct Member { const EndpointFamily* family; const EndpointFamilyMember* member; unsigned seen = 0; };
    std::map<uint32_t, Member> members;
    std::set<uint32_t> families;
    for (const auto& family : plan.families) {
        if (!families.insert(family.id).second || family.members.empty() ||
            family.sourceChoices || family.targetChoices ||
            family.displacement > INT64_MAX || family.sourceOrder > INT64_MAX || family.targetOrder > INT64_MAX ||
            family.sourcePipe > static_cast<uint32_t>(PIPE::PIPE_FIX) ||
            family.targetPipe > static_cast<uint32_t>(PIPE::PIPE_FIX) ||
            family.sourcePipe == static_cast<uint32_t>(PIPE::PIPE_ALL) ||
            family.targetPipe == static_cast<uint32_t>(PIPE::PIPE_ALL) ||
            family.local != (family.sourcePipe == family.targetPipe)) {
            return function.emitError(
                "deferred numerical family has invalid pipes, identity or unsupported cleanup choices");
        }
        for (const auto& member : family.members) {
            if (!members.emplace(member.record, Member{&family, &member}).second) {
                return function.emitError("deferred numerical family repeats a provenance record");
            }
        }
    }
    DominanceInfo dominance(function);
    std::set<std::pair<Operation*, std::size_t>> order;
    for (const auto& piece : recipe.pieces) {
        auto* cut = piece.cut.before;
        const bool local = piece.kind == EndpointKind::Barrier, publish = piece.kind == EndpointKind::Set;
        if (!cut || cut->getParentOfType<func::FuncOp>() != function || piece.cut.block != cut->getBlock() ||
            piece.members.empty() || piece.displacement > INT64_MAX || piece.order > INT64_MAX ||
            (!recipe.outer && piece.displacement) ||
            (recipe.outer && !recipe.outer->isProperAncestor(cut)) ||
            !order.emplace(cut, piece.order).second) {
            return function.emitError("deferred numerical piece has invalid cut, order or counted-loop binding");
        }
        for (const auto& point : piece.members) {
            auto found = members.find(point.record);
            if (found == members.end()) { return function.emitError("deferred piece references unknown record"); }
            auto& entry = found->second;
            const auto& family = *entry.family;
            const auto expected = publish ? family.sourceCut : family.targetCut;
            const auto& coordinates = publish ? entry.member->sourceCoordinates : entry.member->targetCoordinates;
            const unsigned bit = publish ? 1U : local ? 2U : 4U;
            if ((entry.seen & bit) || local != family.local || piece.sourcePipe != family.sourcePipe ||
                piece.targetPipe != family.targetPipe || piece.displacement != family.displacement ||
                expected.before != cut || !sameCoordinates(coordinates, point.coordinates)) {
                return function.emitError("deferred piece disagrees with its matching provenance");
            }
            entry.seen |= bit;
            for (const auto& coordinate : point.coordinates) {
                auto loop = coordinate.loop;
                if (!boundedLoop(loop) ||
                    !dominance.properlyDominates(loop.getInductionVar(), cut)) {
                    return function.emitError(
                        "deferred coordinate must have nonwrapping counted bounds and dominate its original cut");
                }
            }
        }
    }
    for (const auto& [record, entry] : members) {
        (void)record;
        if (entry.seen != (entry.family->local ? 2U : 5U)) {
            return function.emitError("deferred family has missing publication or acquisition pieces");
        }
    }
    return success();
}
struct Encoder {
    Builder builder;
    std::map<Operation*, int64_t> loops, cuts;
    explicit Encoder(MLIRContext* context) : builder(context) {}
    NamedAttribute number(StringRef name, int64_t value)
    {
        return builder.getNamedAttr(name, builder.getI64IntegerAttr(value));
    }
    int64_t loop(scf::ForOp value)
    {
        if (!value) { return -1; }
        return loops.emplace(value.getOperation(), loops.size()).first->second;
    }
    ArrayAttr coordinates(ArrayRef<TemplateCoordinate> values)
    {
        SmallVector<Attribute> result;
        for (const auto& value : values) {
            result.push_back(builder.getDenseI64ArrayAttr({loop(value.loop), value.induction}));
        }
        return builder.getArrayAttr(result);
    }
    DictionaryAttr piece(const EndpointPiece& value)
    {
        SmallVector<Attribute> members;
        for (const auto& member : value.members) {
            members.push_back(builder.getDictionaryAttr({number("record", member.record),
                builder.getNamedAttr("coordinates", coordinates(member.coordinates))}));
        }
        return builder.getDictionaryAttr({number("namespace", value.namespaceId), number("source", value.sourcePipe),
            number("target", value.targetPipe), number("kind", static_cast<int64_t>(value.kind)),
            number("displacement", value.displacement), number("order", value.order),
            builder.getNamedAttr("members", builder.getArrayAttr(members))});
    }
    DictionaryAttr family(const EndpointFamily& value)
    {
        SmallVector<Attribute> members;
        for (const auto& member : value.members) {
            members.push_back(builder.getDictionaryAttr({number("record", member.record),
                number("source", member.source), number("target", member.target),
                builder.getNamedAttr("source_coordinates", coordinates(member.sourceCoordinates)),
                builder.getNamedAttr("target_coordinates", coordinates(member.targetCoordinates))}));
        }
        return builder.getDictionaryAttr({number("id", value.id), number("source", value.sourcePipe),
            number("target", value.targetPipe), number("displacement", value.displacement),
            number("source_order", value.sourceOrder), number("target_order", value.targetOrder),
            number("source_cut", value.local ? -1 : cuts.at(value.sourceCut.before)),
            number("target_cut", cuts.at(value.targetCut.before)),
            builder.getNamedAttr("local", builder.getBoolAttr(value.local)),
            builder.getNamedAttr("members", builder.getArrayAttr(members))});
    }
};
struct Decoder {
    func::FuncOp function;
    std::map<int64_t, scf::ForOp> loops;
    std::map<int64_t, Operation*> cuts;
    FailureOr<SmallVector<TemplateCoordinate>> coordinates(Attribute raw)
    {
        auto values = dyn_cast_or_null<ArrayAttr>(raw);
        if (!values) { return failure(); }
        SmallVector<TemplateCoordinate> result;
        std::set<int64_t> seen;
        for (auto value : values) {
            auto pair = dyn_cast<DenseI64ArrayAttr>(value);
            if (!pair || pair.size() != 2 || !loops.count(pair[0]) || !seen.insert(pair[0]).second) {
                return failure();
            }
            result.push_back({loops.at(pair[0]), pair[1]});
        }
        return result;
    }
    FailureOr<EndpointPiece> piece(DictionaryAttr data, Operation* cut)
    {
        const auto space = integer(data, "namespace"), source = integer(data, "source");
        const auto target = integer(data, "target");
        const auto kind = integer(data, "kind"), displacement = integer(data, "displacement");
        const auto order = integer(data, "order");
        auto members = data ? data.getAs<ArrayAttr>("members") : ArrayAttr{};
        if (!fields(data, {"namespace", "source", "target", "kind", "displacement", "order", "members"}) ||
            !nonnegative(space, UINT32_MAX) || !nonnegative(source, UINT32_MAX) || !nonnegative(target, UINT32_MAX) ||
            !nonnegative(kind, 2) || !nonnegative(displacement) || !nonnegative(order) || !members) {
            return failure();
        }
        EndpointPiece result;
        result.namespaceId = *space; result.sourcePipe = *source; result.targetPipe = *target;
        result.kind = static_cast<EndpointKind>(*kind); result.displacement = *displacement; result.order = *order;
        result.cut = {cut->getBlock(), cut};
        for (auto raw : members) {
            auto member = dyn_cast<DictionaryAttr>(raw);
            const auto record = integer(member, "record");
            if (!fields(member, {"record", "coordinates"}) || !nonnegative(record, UINT32_MAX)) { return failure(); }
            auto point = coordinates(member.get("coordinates"));
            if (failed(point)) { return failure(); }
            result.members.push_back({static_cast<uint32_t>(*record), std::move(*point)});
        }
        return result;
    }
    FailureOr<EndpointFamily> family(DictionaryAttr data)
    {
        auto id = integer(data, "id"), source = integer(data, "source"), target = integer(data, "target");
        auto displacement = integer(data, "displacement"), sourceOrder = integer(data, "source_order"),
            targetOrder = integer(data, "target_order"), sourceCut = integer(data, "source_cut");
        const auto targetCut = integer(data, "target_cut");
        auto local = data ? data.getAs<BoolAttr>("local") : BoolAttr{};
        auto members = data ? data.getAs<ArrayAttr>("members") : ArrayAttr{};
        if (!fields(data, {"id", "source", "target", "displacement", "source_order", "target_order",
                           "source_cut", "target_cut", "local", "members"}) ||
            !nonnegative(id, UINT32_MAX) || !nonnegative(source, UINT32_MAX) || !nonnegative(target, UINT32_MAX) ||
            !nonnegative(displacement) || !nonnegative(sourceOrder) || !nonnegative(targetOrder) ||
            !sourceCut || !targetCut || !local || !members || !cuts.count(*targetCut) ||
            (local.getValue() ? *sourceCut != -1 : !cuts.count(*sourceCut))) { return failure(); }
        EndpointFamily result;
        result.id = *id; result.sourcePipe = *source; result.targetPipe = *target;
        result.displacement = *displacement; result.local = local.getValue();
        result.sourceOrder = *sourceOrder; result.targetOrder = *targetOrder;
        if (!result.local) { result.sourceCut = {cuts.at(*sourceCut)->getBlock(), cuts.at(*sourceCut)}; }
        result.targetCut = {cuts.at(*targetCut)->getBlock(), cuts.at(*targetCut)};
        for (auto raw : members) {
            auto member = dyn_cast<DictionaryAttr>(raw);
            auto record = integer(member, "record"), from = integer(member, "source"), to = integer(member, "target");
            if (!fields(member, {"record", "source", "target", "source_coordinates", "target_coordinates"}) ||
                !nonnegative(record, UINT32_MAX) || !nonnegative(from, UINT32_MAX) || !nonnegative(to, UINT32_MAX)) {
                return failure();
            }
            auto sourcePoint = coordinates(member.get("source_coordinates"));
            auto targetPoint = coordinates(member.get("target_coordinates"));
            if (failed(sourcePoint) || failed(targetPoint)) { return failure(); }
            result.members.push_back({static_cast<uint32_t>(*record), static_cast<uint32_t>(*from),
                static_cast<uint32_t>(*to), std::move(*sourcePoint), std::move(*targetPoint)});
        }
        return result;
    }
};
} // namespace
#include "DeferredContextualFamily.h"
LogicalResult insertDeferredFamilyOutput(func::FuncOp function, PreparedLogicalPlan& plan)
{
    if (!function) { return failure(); }
    if (llvm::any_of(plan.families, [](const EndpointFamily& family) {
        return family.local || bool(family.sourceChoices) || bool(family.targetChoices);
    })) { return insertContextualDeferredFamily(function, plan); }
    auto recipe = numericalFamilyRecipe(plan);
    if (failed(recipe)) {
        return insertContextualDeferredFamily(function, plan);
    }
    // SCF may omit a yield and its attributes when printing. The owned
    // contextual directory supports the same numerical leaf without losing cuts.
    if (llvm::any_of(recipe->pieces, [](const EndpointPiece& piece) {
        return isa<scf::YieldOp>(piece.cut.before);
    })) { return insertContextualDeferredFamily(function, plan); }
    if (function->hasAttr(recipeAttr) || function->hasAttr("pto.endpoint_families") ||
        function->hasAttr(CyclicAllocationAttr) || function->hasAttr(FiniteAllocationAttr) ||
        failed(validate(function, plan, *recipe))) { return failure(); }
    bool occupied = false;
    function.walk([&](Operation* op) {
        occupied |= op->hasAttr(cutAttr) || op->hasAttr(loopAttr) ||
            isa<LogicalFamilyOp, LogicalSetOp, LogicalWaitOp>(op);
    });
    if (occupied) { return function.emitError("deferred-output cannot replace existing logical synchronization"); }
    Encoder encoder(function.getContext());
    const auto outer = encoder.loop(recipe->outer);
    std::map<Operation*, SmallVector<Attribute>> pieces;
    for (const auto& piece : recipe->pieces) {
        encoder.cuts.emplace(piece.cut.before, encoder.cuts.size());
        pieces[piece.cut.before].push_back(encoder.piece(piece));
    }
    SmallVector<Attribute> families;
    for (const auto& family : plan.families) { families.push_back(encoder.family(family)); }
    auto metadata = encoder.builder.getDictionaryAttr({
        encoder.number("version", 1), encoder.number("plan", plan.planId),
        encoder.number("outer", outer), encoder.number("cut_count", pieces.size()),
        encoder.builder.getNamedAttr("complete", encoder.builder.getBoolAttr(plan.completeInvocation)),
        encoder.builder.getNamedAttr("families", encoder.builder.getArrayAttr(families)),
        encoder.builder.getNamedAttr("allocation", plan.allocationCertificate)});
    const auto protocol = encoder.builder.getStringAttr(metadataSeal(metadata));
    // All qualification precedes mutation. Each marker owns one original cut;
    // its SSA axes make dominance and binding changes visible to verification.
    for (const auto& [operation, id] : encoder.loops) {
        operation->setAttr(loopAttr, encoder.builder.getI64IntegerAttr(id));
    }
    std::map<int64_t, scf::ForOp> orderedLoops;
    for (const auto& [operation, id] : encoder.loops) { orderedLoops.emplace(id, cast<scf::ForOp>(operation)); }
    for (const auto& [cut, values] : pieces) {
        SmallVector<Value> axes;
        SmallVector<int64_t> ids;
        for (auto [id, loop] : orderedLoops) {
            if (loop->isProperAncestor(cut)) { axes.push_back(loop.getInductionVar()); ids.push_back(id); }
        }
        OpBuilder builder(cut);
        OperationState state(cut->getLoc(), LogicalFamilyOp::getOperationName());
        state.addOperands(axes);
        state.addAttribute("plan_id", builder.getI64IntegerAttr(plan.planId));
        state.addAttribute("cut_id", builder.getI64IntegerAttr(encoder.cuts.at(cut)));
        state.addAttribute("axis_ids", builder.getDenseI64ArrayAttr(ids));
        state.addAttribute("pieces", builder.getArrayAttr(values));
        builder.create(state);
        cut->setAttr(cutAttr, builder.getDictionaryAttr({
            encoder.number("id", encoder.cuts.at(cut)),
            builder.getNamedAttr("protocol", protocol),
            builder.getNamedAttr("pieces", builder.getArrayAttr(values)),
            builder.getNamedAttr("axis_ids", builder.getDenseI64ArrayAttr(ids))}));
    }
    function->setAttr(recipeAttr, metadata);
    if (plan.recognitionReport) { function->setAttr("pto.frontier_recognition", plan.recognitionReport); }
    return success();
}
FailureOr<std::unique_ptr<PreparedLogicalPlan>> readDeferredFamilyOutput(func::FuncOp function)
{
    if (!function) { return failure(); }
    auto data = function->getAttrOfType<DictionaryAttr>(recipeAttr);
    if (integer(data, "version") == 2 || integer(data, "version") == 3 || integer(data, "version") == 4) {
        return readContextualDeferredFamily(function, data);
    }
    const auto version = integer(data, "version"), planId = integer(data, "plan"), outer = integer(data, "outer"),
        cutCount = integer(data, "cut_count");
    auto complete = data ? data.getAs<BoolAttr>("complete") : BoolAttr{};
    auto families = data ? data.getAs<ArrayAttr>("families") : ArrayAttr{};
    auto allocation = data ? data.getAs<DictionaryAttr>("allocation") : DictionaryAttr{};
    if (!fields(data, {"version", "plan", "outer", "cut_count", "complete", "families", "allocation"}) ||
        version != 1 || !nonnegative(planId) || !outer || *outer < -1 || !nonnegative(cutCount) ||
        !complete || !families || !allocation) {
        function.emitError("malformed or unsupported deferred numerical family metadata"); return failure();
    }
    Decoder decoder{function, {}, {}};
    bool invalid = false;
    std::set<int64_t> anchors;
    std::map<int64_t, Operation*> markers;
    const auto protocol = StringAttr::get(function.getContext(), metadataSeal(data));
    function.walk([&](Operation* op) {
        if (auto raw = op->getAttr(cutAttr)) {
            auto anchor = dyn_cast<DictionaryAttr>(raw);
            auto id = integer(anchor, "id");
            auto* previous = op->getBlock() ? op->getPrevNode() : nullptr;
            if (!fields(anchor, {"id", "pieces", "axis_ids", "protocol"}) || !nonnegative(id) ||
                !anchors.insert(*id).second ||
                anchor.getAs<StringAttr>("protocol") != protocol || !previous || !isa<LogicalFamilyOp>(previous) ||
                previous->getAttr("cut_id") != anchor.get("id") ||
                previous->getAttr("pieces") != anchor.get("pieces") ||
                previous->getAttr("axis_ids") != anchor.get("axis_ids")) { invalid = true; }
        }
        if (auto raw = op->getAttr(loopAttr)) {
            auto id = dyn_cast<IntegerAttr>(raw); auto loop = dyn_cast<scf::ForOp>(op);
            if (!id || !id.getType().isInteger(64) || id.getInt() < 0 || !loop ||
                !decoder.loops.emplace(id.getInt(), loop).second) { invalid = true; }
        }
        if (auto marker = dyn_cast<LogicalFamilyOp>(op)) {
            auto id = marker->getAttrOfType<IntegerAttr>("cut_id");
            auto owner = marker->getAttrOfType<IntegerAttr>("plan_id");
            if (!fields(op->getAttrDictionary(), {"plan_id", "cut_id", "axis_ids", "pieces"}) ||
                !id || !owner || !id.getType().isInteger(64) || !owner.getType().isInteger(64) ||
                owner.getInt() != *planId || id.getInt() < 0 ||
                !markers.emplace(id.getInt(), op).second || failed(mlir::verify(marker.getOperation()))) {
                invalid = true;
            }
            auto* next = op->getNextNode();
            if (!next || !next->hasAttr(cutAttr)) { invalid = true; }
            else if (id && id.getType().isInteger(64)) { decoder.cuts.emplace(id.getInt(), next); }
        } else if (isa<LogicalSetOp, LogicalWaitOp>(op)) { invalid = true; }
    });
    if (invalid || anchors.size() != decoder.cuts.size() || decoder.cuts.size() != static_cast<uint64_t>(*cutCount) ||
        (*outer != -1 && !decoder.loops.count(*outer))) {
        function.emitError("deferred numerical family has missing or repeated loop/cut bindings"); return failure();
    }
    NumericalFamilyRecipe recipe;
    if (*outer != -1) { recipe.outer = decoder.loops.at(*outer); }
    DominanceInfo dominance(function);
    for (const auto& [id, cut] : markers) {
        auto axes = cut->getAttrOfType<DenseI64ArrayAttr>("axis_ids");
        std::set<int64_t> seen;
        for (auto [axis, value] : llvm::zip(axes.asArrayRef(), cut->getOperands())) {
            if (!decoder.loops.count(axis) || !seen.insert(axis).second ||
                decoder.loops.at(axis).getInductionVar() != value || !dominance.properlyDominates(value, cut)) {
                cut->emitError("deferred numerical family has invalid or non-dominating axis binding");
                return failure();
            }
        }
        for (auto raw : cut->getAttrOfType<ArrayAttr>("pieces")) {
            auto piece = decoder.piece(dyn_cast<DictionaryAttr>(raw), decoder.cuts.at(id));
            if (failed(piece)) { cut->emitError("malformed deferred numerical piece"); return failure(); }
            for (const auto& member : piece->members) {
                for (const auto& coordinate : member.coordinates) {
                    auto axis = coordinate.loop->getAttrOfType<IntegerAttr>(loopAttr);
                    if (!seen.count(axis.getInt())) {
                        cut->emitError("deferred piece omits its SSA axis binding"); return failure();
                    }
                }
            }
            if (*outer != -1 && !seen.count(*outer)) { return failure(); }
            recipe.pieces.push_back(std::move(*piece));
        }
    }
    auto plan = std::make_unique<PreparedLogicalPlan>(*planId);
    plan->completeInvocation = complete.getValue();
    plan->groupedFamilies = plan->independentPieces = true;
    plan->allocationCertificate = allocation;
    for (auto raw : families) {
        auto family = decoder.family(dyn_cast<DictionaryAttr>(raw));
        if (failed(family)) { function.emitError("malformed deferred family provenance"); return failure(); }
        plan->families.push_back(std::move(*family));
    }
    if (failed(validate(function, *plan, recipe))) { return failure(); }
    auto restored = restoreNumericalFamilyRecipe(recipe);
    if (failed(restored)) { function.emitError("invalid deferred numerical selector domain"); return failure(); }
    plan->familyRecipe = *restored;
    if (failed(normalizeLogicalFamilies(*plan))) { return failure(); }
    return plan;
}
LogicalResult allocateDeferredFamilyOutput(func::FuncOp function, ArrayRef<int64_t> eligibleIds)
{
    if (!function) { return failure(); }
    // Decode and lower in an isolated clone. A malformed artifact, unavailable
    // mapping or failed insertion cannot consume the saved protocol.
    OwningOpRef<ModuleOp> staging = ModuleOp::create(function.getLoc());
    if (auto parent = function->getParentOfType<ModuleOp>()) {
        (*staging)->setAttrs(parent->getAttrs());
        // Original payload calls/globals keep their symbol context while the
        // target alone is changed and committed. No sibling mutation escapes.
        for (auto& sibling : *parent.getBody()) {
            if (&sibling != function.getOperation()) { staging->getBody()->push_back(sibling.clone()); }
        }
    }
    auto clone = cast<func::FuncOp>(function->clone());
    staging->getBody()->push_back(clone);
    auto plan = readDeferredFamilyOutput(clone);
    if (failed(plan)) { return failure(); }
    // The decoded plan names the original payload anchors, so marker removal
    // precedes ordinary insertion and cannot leave logical artifacts behind.
    SmallVector<Operation*> markers;
    clone.walk([&](LogicalFamilyOp op) { markers.push_back(op); });
    for (auto* marker : markers) { marker->erase(); }
    clone.walk([&](Operation* op) { op->removeAttr(loopAttr); op->removeAttr(cutAttr); op->removeAttr(sourceAttr); });
    clone->removeAttr(recipeAttr);
    if (failed(planPhysicalFamilyOutput(clone, **plan, eligibleIds)) ||
        failed(insertLogicalSynchronization(clone, **plan))) { return failure(); }
    if (failed(verify(clone))) { return failure(); }
    function->setAttrs(clone->getAttrs());
    function.getBody().takeBody(clone.getBody());
    return success();
}
} // namespace mlir::pto::frontiersynch
