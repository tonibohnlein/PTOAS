// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Preserve original occurrence identity in late-expansion diagnostic output.
#include "PTO/Transforms/FrontierSynch/Structured/NumericTemplateAnalysis.h"
#include "PTO/Transforms/FrontierSynch/ProgramRecognition.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Structured/NumericTemplatePlan.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Structured/NumericTemplateRecurringGlobal.h"
#include "PTO/Transforms/FrontierSynch/Structured/NumericTemplateInsertion.h"
#include "PTO/Transforms/FrontierSynch/Structured/NumericTemplateRegional.h"
#include "AlignedModularCellOracle.h"
#include "NumericAtomReferenceChecks.h"
#include "mlir/IR/AsmState.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
std::string name(Value value, AsmState& state)
{
    std::string text;
    llvm::raw_string_ostream stream(text);
    value.printAsOperand(stream, state);
    return text;
}
std::string expression(AffineExpr value)
{
    std::string text;
    llvm::raw_string_ostream stream(text);
    value.print(stream);
    return text;
}
llvm::json::Object range(const pto::SyncStorageCell& value)
{
    return llvm::json::Object{{"space", static_cast<unsigned>(value.space)},
                              {"begin", value.begin}, {"end", value.end}};
}
}
llvm::json::Object dumpNumericTemplate(const fs::NumericTemplate& result, AsmState& state)
{
    llvm::json::Array issues, payloads, atoms;
    for (const auto& diagnostic : result.result.diagnostics) {
        issues.push_back(llvm::json::Object{{"issue", fs::recognitionName(diagnostic.issue)},
            {"category", diagnostic.outsideClass ? "class-mismatch" : "unmet-obligation"},
            {"operation", diagnostic.anchor ? diagnostic.anchor->getName().getStringRef() : StringRef()}});
    }
    auto occurrences = fs::numericTemplateOccurrences(result);
    std::size_t payloadIndex = 0;
    for (const auto& payload : result.payloads) {
        llvm::json::Array coordinates, effects;
        for (auto coordinate : payload.coordinates) {
            coordinates.push_back(llvm::json::Object{{"induction", name(coordinate.loop.getInductionVar(), state)},
                                                    {"value", coordinate.induction}});
        }
        for (const auto& effect : payload.effects) {
            llvm::json::Array maps, ranges, ids;
            for (const auto& map : effect.regions) {
                llvm::json::Array extents, invariantSymbols;
                for (Value symbol : map.invariantSymbols) {
                    invariantSymbols.push_back(name(symbol, state));
                }
                for (auto extent : map.extents) {
                    extents.push_back(expression(extent));
                }
                auto argument = dyn_cast_or_null<BlockArgument>(map.base);
                maps.push_back(llvm::json::Object{
                    {"base_argument", argument ? static_cast<int64_t>(argument.getArgNumber()) : -1},
                    {"offset", expression(map.byteOffset)}, {"extents", std::move(extents)},
                    {"invariant_symbols", std::move(invariantSymbols)},
                    {"element_bytes", map.elementBytes}});
            }
            for (const auto& item : effect.ranges) {
                ranges.push_back(range(item));
            }
            if (effect.validAtomReferences()) {
                for (auto atom : effect.atomReferences()) { ids.push_back(atom); }
            }
            effects.push_back(llvm::json::Object{{"source_effect", effect.sourceEffect},
                {"mode", effect.mode == pto::SyncAccessMode::Read ? "read" : "write"},
                {"valid_atom_references", effect.validAtomReferences()},
                {"maps", std::move(maps)}, {"ranges", std::move(ranges)}, {"atoms", std::move(ids)},
                {"discharge", static_cast<unsigned>(effect.discharge)}, {"outer_stride", effect.outerStride}});
        }
        llvm::json::Array protection;
        if (succeeded(occurrences)) {
            for (const auto& access : (*occurrences)[payloadIndex].accesses) {
                if (access.write && access.protectionGroup != 0) {
                    protection.push_back(llvm::json::Array{access.atom, access.protectionGroup});
                }
            }
        }
        ++payloadIndex;
        payloads.push_back(llvm::json::Object{{"hardware_protection", std::move(protection)},
            {"phase", payload.phase->GetIndex()},
            {"operation", payload.phase->elementOp->getName().getStringRef()},
            {"pipe", static_cast<unsigned>(payload.phase->kPipeValue)}, {"coordinates", std::move(coordinates)},
            {"effects", std::move(effects)}});
    }
    for (const auto& atom : result.atoms) {
        atoms.push_back(range(atom));
    }
    auto outer = result.outer;
    return llvm::json::Object{{"route", "numeric-template"}, {"state", fs::recognitionName(result.result.state)},
        {"issues", std::move(issues)}, {"lower", result.lower}, {"step", result.step},
        {"empty_invocation", result.emptyInvocation},
        {"upper", name(outer.getUpperBound(), state)}, {"counted_visits", result.countedVisits},
        {"counted_payloads", result.countedPayloads}, {"fragments", result.fragments},
        {"estimated_attempted_fragments", result.estimatedAttemptedFragments},
        {"atom_reference_requests", result.atomReferenceCost.requests},
        {"atom_reference_hits", result.atomReferenceCost.hits},
        {"atom_reference_expansions", result.atomReferenceCost.expansions},
        {"atom_reference_expanded_ids", result.atomReferenceCost.expandedIds},
        {"period", result.period}, {"refresh", result.refresh}, {"scope", "whole-function"},
        {"interfaces_ready", false}, {"payloads", std::move(payloads)}, {"atoms", std::move(atoms)}};
}

LogicalResult checkNumericPreflight(func::FuncOp function, const pto::SyncInput& input,
    const fs::ProgramRecognition& program)
{
    if (!numericAtomReferenceChecks(function.getContext())) {
        return function.emitError("numeric atom reference cache ownership/protection checks failed");
    }
    fs::PhaseIndex index;
    if (failed(index.build(function, input))) { return failure(); }
    for (const auto& node : program.nodes) {
        if (!node.numericTemplate) { continue; }
        const auto plan = fs::preflightNumericTemplate(cast<scf::ForOp>(node.anchor), index, input);
        const bool pure = plan.form.payloads.empty() && plan.form.atoms.empty() && !plan.form.fragments &&
            plan.form.nativePrerequisites.empty() && plan.form.valueDemands.empty() &&
            llvm::all_of(plan.payloads, [](const auto& payload) { return payload.effects.empty(); });
        if (!pure) { return function.emitError("numeric preflight constructed effects or native requirements"); }
        pto::SyncInput foreign;
        fs::PhaseIndex foreignIndex;
        auto refusedInput = fs::materializeNumericTemplate(plan, index, foreign);
        auto refusedIndex = fs::materializeNumericTemplate(plan, foreignIndex, input);
        const bool rejected = refusedInput.result.state != fs::RecognitionState::Applicable &&
            refusedIndex.result.state != fs::RecognitionState::Applicable;
        if (!rejected) { return function.emitError("numeric preflight accepted a foreign context"); }
        auto actual = fs::materializeNumericTemplate(plan, index, input);
        const auto& expected = *node.numericTemplate;
        bool same = actual.result.state == expected.result.state &&
            actual.payloads.size() == expected.payloads.size() &&
            actual.countedVisits == expected.countedVisits && actual.countedPayloads == expected.countedPayloads &&
            actual.fragments == expected.fragments && actual.period == expected.period &&
            actual.refresh == expected.refresh && actual.emptyInvocation == expected.emptyInvocation;
        for (auto [a, b] : llvm::zip(actual.payloads, expected.payloads)) {
            same &= a.phase == b.phase && a.coordinates.size() == b.coordinates.size() &&
                a.effects.size() == b.effects.size() && a.invocationProtection == b.invocationProtection;
            for (auto [x, y] : llvm::zip(a.coordinates, b.coordinates)) {
                same &= x.loop == y.loop && x.induction == y.induction;
            }
        }
        if (!same) { return function.emitError("numeric retained preflight changed original mapping or construction"); }
    }
    return success();
}

namespace {
bool recurringBudget(func::FuncOp function, const pto::SyncInput& input, const fs::NumericTemplate& form)
{
    fs::PhaseIndex limitIndex;
    if (failed(limitIndex.build(function, input)) || !form.fragments) { return false; }
    auto limits = form.limits;
    limits.fragments = form.fragments - 1;
    const auto limited = fs::recognizeNumericTemplate(form.outer, limitIndex, input, limits);
    return limited.result.state != fs::RecognitionState::Applicable && limited.estimatedAttemptedFragments &&
        !limited.fragments && limited.payloads.empty() && limited.atoms.empty() &&
        limited.atomReferenceCost.requests && limited.atomReferenceCost.expansions;
}
bool privateBoundSnapshot(const fs::NumericTemplate& form)
{
    auto loop = form.outer;
    const auto upper = loop.getUpperBound();
    loop->setOperand(1, loop.getLowerBound());
    const bool refused = !form.recurringGlobal->applies(form) && failed(fs::numericTemplateOccurrences(form));
    loop->setOperand(1, upper);
    return refused && form.recurringGlobal->applies(form);
}
bool privateWithinVisitRefusal(const fs::NumericTemplate& form)
{
    const auto& group = form.recurringGlobal->privateGroups.front();
    if (group.accesses.empty()) { return false; }
    fs::detail::AlignedModularMap map;
    map.base = group.base; map.origin = group.origin; map.width = group.width;
    map.alignmentBits = group.alignmentBits; map.byteStride = group.byteStride;
    map.ordinalStride = group.ordinalStride; map.access = group.accesses.front();
    auto zeroOrigin = map;
    zeroOrigin.origin = Value{};
    const SmallVector<std::size_t> singlePayload{0};
    const SmallVector<uint8_t> singleWrite{1};
    if (!fs::detail::provePrivateAlignedMaps(form.outer, {zeroOrigin}, singlePayload, singleWrite, UINT32_MAX)) {
        return false;
    }
    SmallVector<fs::detail::AlignedModularMap> maps{map, map};
    const SmallVector<std::size_t> payloads{0, 1};
    const SmallVector<uint8_t> writes{0, 1};
    if (fs::detail::provePrivateAlignedMaps(form.outer, maps, payloads, writes, UINT32_MAX)) { return false; }
    for (auto& empty : maps) { empty.access.intervals.clear(); }
    if (!fs::detail::provePrivateAlignedMaps(form.outer, maps, payloads, writes, UINT32_MAX)) { return false; }
    maps.front().width = 0;
    return !fs::detail::provePrivateAlignedMaps(form.outer, maps, payloads, writes, UINT32_MAX);
}
bool recurringMembership(const fs::NumericTemplate& form)
{
    const auto owner = form.recurringGlobal;
    const auto& record = owner->effects.front();
    auto rejected = [&](auto mutate) {
        auto changed = form;
        mutate(changed);
        return !owner->applies(changed) && failed(fs::numericTemplateOccurrences(changed));
    };
    return rejected([&](auto& x) { x.payloads[record.payload].effects.push_back(
            x.payloads[record.payload].effects[record.effect]); }) &&
        rejected([&](auto& x) { x.payloads[record.payload].effects.erase(
            x.payloads[record.payload].effects.begin() + record.effect); }) &&
        rejected([&](auto& x) { ++x.payloads[record.payload].effects[record.effect].sourceEffect; }) &&
        rejected([&](auto& x) { x.payloads[record.payload].effects[record.effect].mode =
            x.payloads[record.payload].effects[record.effect].mode == pto::SyncAccessMode::Read ?
                pto::SyncAccessMode::Write : pto::SyncAccessMode::Read; }) &&
        rejected([&](auto& x) { x.payloads[record.payload].effects[record.effect].ranges.push_back(
            pto::SyncStorageCell{pto::AddressSpace::GM, 0, 1}); }) &&
        rejected([&](auto& x) { x.payloads[record.payload].effects[record.effect].sharedAtoms =
            std::make_shared<const std::vector<std::size_t>>(std::vector<std::size_t>{0}); }) &&
        rejected([&](auto& x) { auto& e = x.payloads[record.payload].effects[record.effect];
            e.sharedAtoms = std::make_shared<const std::vector<std::size_t>>(); e.atoms.push_back(0); });
}
bool recurringExportRefusal(func::FuncOp function, const pto::SyncInput& input, const fs::NumericTemplate& form)
{
    const auto owner = form.recurringGlobal;
    std::string error;
    auto arena = std::make_shared<fs::RegionExpressions>();
    fs::NumericBodyMathematics body{form, {}, {}, {}, {}};
    return failed(fs::exportNumericBody(function, input, body, arena, {}, error)) &&
        error == "relative recurring GM storage has no regional selector adapter" &&
        form.recurringGlobal == owner && owner->applies(form);
}
bool independentPlans(const fs::PreparedLogicalPlan& first, const fs::PreparedLogicalPlan& second)
{
    DenseSet<Operation*> operations;
    DenseSet<Block*> blocks;
    for (const auto& stage : first.preparation) {
        blocks.insert(stage.code.get());
        stage.code->walk([&](Operation* op) {
            operations.insert(op);
            for (auto& region : op->getRegions()) { for (auto& block : region) { blocks.insert(&block); } }
        });
    }
    bool independent = true;
    for (const auto& stage : second.preparation) {
        independent &= !blocks.contains(stage.code.get());
        stage.code->walk([&](Operation* op) {
            independent &= !operations.contains(op);
            for (auto value : op->getOperands()) {
                if (auto argument = dyn_cast<BlockArgument>(value)) {
                    independent &= !blocks.contains(argument.getOwner());
                } else { independent &= !operations.contains(value.getDefiningOp()); }
            }
        });
    }
    return independent;
}
std::string renderPreparation(const fs::PreparedLogicalPlan& plan)
{
    std::string text;
    llvm::raw_string_ostream stream(text);
    for (const auto& stage : plan.preparation) { stage.code->print(stream); }
    return text;
}
bool recurringFreshPlans(func::FuncOp function, const fs::ProgramRecognition& program)
{
    auto first = fs::prepareNumericTemplateInsertion(function, program);
    auto second = fs::prepareNumericTemplateInsertion(function, program);
    if (failed(first) || failed(second) || !(*first)->allocationCertificate || !(*second)->allocationCertificate ||
        !(*first)->familyRecipe || !(*second)->familyRecipe ||
        !(*first)->preparation.empty() || !(*second)->preparation.empty() ||
        failed(fs::materializeLogicalFamilies(**first)) || failed(fs::materializeLogicalFamilies(**second)) ||
        (*first)->familyRecipe || (*second)->familyRecipe ||
        (*first)->preparation.empty() || (*second)->preparation.empty() || !independentPlans(**first, **second)) {
        return false;
    }
    const auto before = renderPreparation(**second);
    first->reset();
    return before == renderPreparation(**second);
}
} // namespace
LogicalResult checkRecurringNumericExports(func::FuncOp function, const pto::SyncInput& input,
    const fs::ProgramRecognition& program)
{
    if (!function->hasAttr("test.recurring_global_numeric")) { return success(); }
    if (!alignedCellOracle()) { return function.emitError("aligned original-byte oracle failed"); }
    const fs::NumericTemplate* form = nullptr;
    for (const auto& node : program.nodes) {
        if (node.numericTemplate && node.numericTemplate->recurringGlobal) {
            if (form) { return function.emitError("expected one recurring numerical owner"); }
            form = &*node.numericTemplate;
        }
    }
    if (!form || !form->recurringGlobal->applies(*form) || !form->estimatedAttemptedFragments) {
        return function.emitError("missing sealed recurring numerical proof");
    }
    if (function->hasAttr("test.private_global_numeric")) {
        const auto& owner = *form->recurringGlobal;
        if (owner.privateGroups.size() != 1 || !owner.groups.empty() ||
            llvm::any_of(owner.effects, [](const auto& effect) {
                return effect.kind != fs::detail::RelativeGlobalKind::PrivateTranslation || !effect.atoms.empty();
            })) { return function.emitError("expected the original private-translation certificate"); }
        if (!privateBoundSnapshot(*form) || !privateWithinVisitRefusal(*form)) {
            return function.emitError("private bound snapshot or same-segment writer refusal failed");
        }
    }
    if (!recurringBudget(function, input, *form) || !recurringMembership(*form) ||
        !recurringExportRefusal(function, input, *form) || !recurringFreshPlans(function, program) ||
        !form->recurringGlobal->applies(*form)) {
        return function.emitError("recurring budget, sealed membership or export ownership check failed");
    }
    llvm::errs() << "recurring-numeric: sealed membership fresh plans retained math unavailable selectors\n";
    return success();
}
