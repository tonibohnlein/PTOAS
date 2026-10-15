// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent unfolding of frozen family contributions, before and after SSA.
#ifndef PTO_FAMILY_PROTOCOL_CHECKS_H
#define PTO_FAMILY_PROTOCOL_CHECKS_H
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/APInt.h"
#include "PTO/Transforms/FrontierSynch/Shared/PhysicalAllocation.h"
#include "mlir/IR/Verifier.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
namespace {
namespace protocol_fs = mlir::pto::frontiersynch;
bool evaluateProtocolBlock(mlir::Block& block, mlir::DenseMap<mlir::Value, uint64_t>& values)
{
    using namespace mlir;
    for (auto& operation : block) {
        SmallVector<uint64_t> args;
        for (auto operand : operation.getOperands()) {
            auto found = values.find(operand);
            if (found == values.end()) {
                return false;
            }
            args.push_back(found->second);
        }
        uint64_t result;
        if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
            result = cast<IntegerAttr>(constant.getValue()).getValue().getZExtValue();
        } else if (isa<arith::AddIOp>(operation)) {
            result = args[0] + args[1];
        } else if (isa<arith::SubIOp>(operation)) {
            result = args[0] - args[1];
        } else if (isa<arith::MulIOp>(operation)) {
            result = args[0] * args[1];
        } else if (isa<arith::DivUIOp>(operation)) {
            if (!args[1]) {
                return false;
            }
            result = args[0] / args[1];
        } else if (isa<arith::RemUIOp>(operation)) {
            if (!args[1]) {
                return false;
            }
            result = args[0] % args[1];
        } else if (isa<arith::AndIOp>(operation)) {
            result = args[0] & args[1];
        } else if (isa<arith::OrIOp>(operation)) {
            result = args[0] | args[1];
        } else if (isa<arith::XOrIOp>(operation)) {
            result = args[0] ^ args[1];
        } else if (isa<arith::SelectOp>(operation)) {
            result = args[0] ? args[1] : args[2];
        } else if (isa<arith::IndexCastOp>(operation)) {
            result = args[0];
        } else if (auto compare = dyn_cast<arith::CmpIOp>(operation)) {
            const llvm::APInt a(64, args[0]), b(64, args[1]);
            switch (compare.getPredicate()) {
                case arith::CmpIPredicate::eq:
                    result = a == b;
                    break;
                case arith::CmpIPredicate::ne:
                    result = a != b;
                    break;
                case arith::CmpIPredicate::ult:
                    result = a.ult(b);
                    break;
                case arith::CmpIPredicate::ule:
                    result = a.ule(b);
                    break;
                case arith::CmpIPredicate::uge:
                    result = a.uge(b);
                    break;
                case arith::CmpIPredicate::ugt:
                    result = a.ugt(b);
                    break;
                case arith::CmpIPredicate::slt:
                    result = a.slt(b);
                    break;
                case arith::CmpIPredicate::sle:
                    result = a.sle(b);
                    break;
                case arith::CmpIPredicate::sge:
                    result = a.sge(b);
                    break;
                case arith::CmpIPredicate::sgt:
                    result = a.sgt(b);
                    break;
            }
        } else {
            return false;
        }
        if (operation.getNumResults() != 1) {
            return false;
        }
        values[operation.getResult(0)] = result;
    }
    return true;
}
bool checkFamilyProtocol(mlir::MLIRContext* context, unsigned count, unsigned mode, bool physical = false)
{
    using namespace mlir;
    namespace fs = protocol_fs;
    OpBuilder builder(context);
    const auto location = builder.getUnknownLoc();
    OwningOpRef<ModuleOp> module(ModuleOp::create(location));
    auto function =
        func::FuncOp::create(location, "family_protocol", builder.getFunctionType({builder.getI1Type()}, {}));
    module->push_back(function);
    auto* entry = function.addEntryBlock();
    builder.setInsertionPointToEnd(entry);
    auto zero = builder.create<arith::ConstantIndexOp>(location, 0);
    auto one = builder.create<arith::ConstantIndexOp>(location, 1);
    auto bound = builder.create<arith::ConstantIndexOp>(location, count);
    auto fixedLabel = builder.create<arith::ConstantIndexOp>(location, 100000);
    auto yes = builder.create<arith::ConstantIntOp>(location, 1, 1);
    auto loop = builder.create<scf::ForOp>(location, zero, bound, one);
    builder.create<func::ReturnOp>(location);
    builder.setInsertionPointToStart(loop.getBody());
    auto* sourceCut = builder.create<arith::ConstantIndexOp>(location, 7).getOperation();
    auto* targetCut = builder.create<arith::ConstantIndexOp>(location, 8).getOperation();
    auto arena = std::make_shared<fs::RegionExpressions>();
    const auto coordinate = arena->input(loop.getInductionVar()), gate = arena->input(entry->getArgument(0));
    fs::PreparedLogicalPlan plan(0);
    plan.nestedIdentities = mode == 4;
    plan.independentPieces = true;
    plan.groupedFamilies = true;
    auto family = [&](uint32_t record, std::optional<unsigned> point) {
        fs::EndpointFamily f;
        f.id = record;
        f.sourcePipe = 1;
        f.targetPipe = 2;
        f.sourceCut = {loop.getBody(), sourceCut};
        f.targetCut = {loop.getBody(), targetCut};
        fs::EndpointFamilyMember member{record, 0, 1, {}, {}};
        if (point) {
            member.sourceCoordinates.push_back({loop, *point});
            member.targetCoordinates = member.sourceCoordinates;
        }
        f.members.push_back(std::move(member));
        plan.families.push_back(std::move(f));
    };
    family(100000, std::nullopt);
    for (auto kind : {fs::LogicalCommandKind::Set, fs::LogicalCommandKind::Wait}) {
        fs::PreparedLogicalEndpoint endpoint{
            kind == fs::LogicalCommandKind::Set ? sourceCut : targetCut, kind, 1, 2, 10, yes, zero};
        endpoint.memberCoordinates.push_back(fixedLabel);
        endpoint.records.push_back(100000);
        endpoint.piece = plan.endpoints.size();
        plan.endpoints.push_back(std::move(endpoint));
    }
    std::vector<uint32_t> labels;
    for (unsigned i = 0; i < count; ++i) {
        const auto label = uint32_t(10 + (37 * i) % count);
        const auto point = mode == 2 ? i % 2 : i;
        labels.push_back(label);
        family(label, point);
        SmallVector<fs::ContextualEndpointFamily> endpoints;
        for (auto kind : {fs::EndpointKind::Set, fs::EndpointKind::Wait}) {
            fs::ContextualEndpointFamily endpoint;
            endpoint.piece.namespaceId = 0;
            endpoint.piece.sourcePipe = 1;
            endpoint.piece.targetPipe = 2;
            endpoint.piece.kind = kind;
            endpoint.piece.cut = {loop.getBody(), kind == fs::EndpointKind::Set ? sourceCut : targetCut};
            endpoint.piece.members.push_back({label, {{loop, point}}});
            endpoint.activity = arena->eq(coordinate, arena->constant(point));
            if (mode == 4) {
                endpoint.activity = arena->land(endpoint.activity, gate);
            }
            endpoint.ordinal = arena->constant(0);
            endpoints.push_back(std::move(endpoint));
        }
        fs::ContextualFamilyExpressions expressions(arena);
        auto leaf = fs::buildMixedFamilyEndpointRecipe(
            expressions, endpoints, mode == 4 ? ArrayRef<uint8_t>({0, 0}) : ArrayRef<uint8_t>({1, 1}), {});
        if (failed(leaf)) {
            return false;
        }
        if (mode == 3) {
            leaf = fs::maskFamilyActivity(*leaf, arena, i < count / 2 ? gate : arena->lnot(gate));
            if (failed(leaf)) {
                return false;
            }
        }
        if (mode == 4) {
            leaf = fs::prefixFamilyInvocation(*leaf, arena, coordinate);
            if (failed(leaf)) {
                return false;
            }
            leaf = fs::prefixFamilyInvocation(*leaf, arena, arena->constant(7));
            if (failed(leaf)) {
                return false;
            }
        }
        const std::size_t position = mode == 1 && i >= count / 2 ? 2 : 1;
        auto appended = fs::appendFamilyEndpointRecipe(plan.familyRecipe, *leaf, 0, position);
        if (failed(appended)) {
            return false;
        }
        plan.familyRecipe = *appended;
    }
    const auto originalRecipe = plan.familyRecipe;
    if (failed(fs::normalizeLogicalFamilies(plan)) || plan.preparation.size() || plan.endpoints.size() != 2 ||
        !plan.familyProtocol || plan.familyWork.materializations || plan.familyWork.normalizationInputs != 2 * count) {
        return false;
    }
    if (mode == 0 && plan.familyWork.normalizedSites != 2) {
        return false;
    }
    const auto frozen = plan.familyRecipe;
    const auto key = fs::normalizedFamilyProtocolKey(*frozen);
    if (key.empty() || failed(fs::normalizeLogicalFamilies(plan)) || plan.familyRecipe != frozen) {
        return false;
    }
    auto arities = fs::familyMatchingArities(*frozen);
    if (failed(arities) || arities->size() != count) {
        return false;
    }
    for (const auto& item : *arities) {
        if (item.second != (mode == 4 ? 3 : 1)) {
            return false;
        }
    }
    // Frozen owner and structural metadata cannot be silently replaced under
    // allocation evidence. Restore each mutation and retain the incumbent.
    plan.endpoints.front().record = 99;
    if (succeeded(fs::normalizeLogicalFamilies(plan))) {
        return false;
    }
    plan.endpoints.front().record = 10;
    plan.completeInvocation = true;
    if (succeeded(fs::normalizeLogicalFamilies(plan))) {
        return false;
    }
    plan.completeInvocation = false;
    plan.familyRecipe = originalRecipe;
    if (succeeded(fs::normalizeLogicalFamilies(plan))) {
        return false;
    }
    plan.familyRecipe = frozen;
    if (failed(fs::normalizeLogicalFamilies(plan))) {
        return false;
    }
    if (physical && mode == 0) {
        SmallVector<fs::PhysicalRecordAllocation> maps, renamed, changed;
        SmallVector<SmallVector<fs::TemplateCoordinate>> points;
        for (unsigned i = 0; i < count; ++i) {
            maps.push_back({int64_t(labels[i]), 1, 2, 0, i % 2, {3, 5}, std::nullopt});
            renamed.push_back(maps.back());
            renamed.back().record = 500 + i;
            changed.push_back(maps.back());
            changed.back().ids = {5, 3};
            points.push_back({{loop, i}});
        }
        DominanceInfo dominance(function);
        auto a = fs::planPhysicalEndpointMap(sourceCut, maps, points, 1, false, dominance);
        auto b = fs::planPhysicalEndpointMap(sourceCut, renamed, points, 1, false, dominance);
        auto c = fs::planPhysicalEndpointMap(sourceCut, changed, points, 1, false, dominance);
        if (failed(a) || failed(b) || failed(c) || fs::physicalMapRequiresMember(**a) ||
            !fs::equivalentPhysicalMaps(**a, **b) || fs::equivalentPhysicalMaps(**a, **c)) {
            return false;
        }
        // These are supplied assignment expressions, not a reuse oracle. Exact
        // finite point evaluation checks re-expression and provenance independence.
        for (unsigned i = 0; i < count; ++i) {
            Block code;
            builder.setInsertionPointToEnd(&code);
            auto id = fs::emitPhysicalEndpointMap(builder, location, **a, {zero}, {});
            DenseMap<Value, uint64_t> values;
            values[zero] = 0;
            values[loop.getInductionVar()] = i;
            if (!evaluateProtocolBlock(code, values) || values.lookup(id) != (i % 2 ? 5 : 3)) {
                return false;
            }
            if (std::distance(code.begin(), code.end()) > 16) {
                return false;
            }
        }
    }
    if (physical && mode == 4) {
        DominanceInfo dominance(function);
        const fs::PhysicalRecordAllocation record{
            999998, 1, 2, 0, 0, {3, 5}, fs::PhysicalTupleRule{3, 0, {{2, 1, 0, 2, 1}}}};
        const SmallVector<fs::PhysicalRecordAllocation> records{record};
        const SmallVector<SmallVector<fs::TemplateCoordinate>> points{{}};
        auto mapping = fs::planPhysicalEndpointMap(sourceCut, records, points, 3, true, dominance);
        if (failed(mapping)) {
            return false;
        }
        for (unsigned i = 0; i < count; ++i) {
            Block code;
            builder.setInsertionPointToEnd(&code);
            auto id = fs::emitPhysicalEndpointMap(builder, location, **mapping, {{}, {}, loop.getInductionVar()}, {});
            DenseMap<Value, uint64_t> values;
            values[loop.getInductionVar()] = i;
            if (!evaluateProtocolBlock(code, values) || values.lookup(id) != (i % 2 ? 5 : 3)) {
                return false;
            }
        }
    }
    if (physical && mode == 4) {
        SmallVector<fs::PhysicalRecordAllocation> records;
        SmallVector<SmallVector<fs::TemplateCoordinate>> points;
        for (unsigned i = 0; i < count; ++i) {
            records.push_back(
                {int64_t(labels[i]),
                 1,
                 2,
                 0,
                 0,
                 {0, 1, 3, 5},
                 fs::PhysicalTupleRule{3, 0, {{0, 0, i % 2, 2, 1}, {2, 1, i % 2, 2, 2}}}});
            points.push_back({{loop, i}});
        }
        DominanceInfo dominance(function);
        auto mapping = fs::planPhysicalEndpointMap(sourceCut, records, points, 3, true, dominance);
        if (failed(mapping) || (count >= 16 && fs::physicalMapRequiresMember(**mapping))) {
            return false;
        }
        const uint64_t palette[] = {0, 1, 3, 5};
        for (unsigned i = 0; i < count; ++i) {
            const auto visit = (i * 7 + 3) % 9;
            Block code;
            builder.setInsertionPointToEnd(&code);
            Value selected = fs::physicalMapRequiresMember(**mapping) ?
                                 builder.create<arith::ConstantIndexOp>(location, labels[i]) :
                                 Value{};
            Value coordinate = builder.create<arith::ConstantIndexOp>(location, visit);
            auto id = fs::emitPhysicalEndpointMap(builder, location, **mapping, {{}, {}, coordinate}, selected);
            DenseMap<Value, uint64_t> values;
            values[loop.getInductionVar()] = i;
            const auto expected = palette[i % 2 + 2 * ((visit + i % 2) % 2)];
            if (!evaluateProtocolBlock(code, values) || values.lookup(id) != expected ||
                (count >= 16 && std::distance(code.begin(), code.end()) > 32)) {
                return false;
            }
        }
    }
    if (physical) {
        auto group = [&](ArrayRef<int64_t> records, ArrayRef<int64_t> conflicts) {
            SmallVector<int64_t> zeros(records.size(), 0);
            return builder.getDictionaryAttr(
                {builder.getNamedAttr("source", builder.getI64IntegerAttr(1)),
                 builder.getNamedAttr("target", builder.getI64IntegerAttr(2)),
                 builder.getNamedAttr("budget", builder.getI64IntegerAttr(1)),
                 builder.getNamedAttr("records", builder.getDenseI64ArrayAttr(records)),
                 builder.getNamedAttr("strides", builder.getDenseI64ArrayAttr(zeros)),
                 builder.getNamedAttr("phases", builder.getDenseI64ArrayAttr(zeros)),
                 builder.getNamedAttr("conflicts", builder.getDenseI64ArrayAttr(conflicts))});
        };
        SmallVector<int64_t> records(labels.begin(), labels.end());
        plan.allocationCertificate = builder.getDictionaryAttr(
            {builder.getNamedAttr("version", builder.getI64IntegerAttr(2)),
             builder.getNamedAttr("plan", builder.getI64IntegerAttr(0)),
             builder.getNamedAttr("strategy", builder.getStringAttr("regional-palettes")),
             builder.getNamedAttr("groups", builder.getArrayAttr({group({100000}, {}), group(records, {0})}))});
        // An insufficient pool leaves both the immutable protocol and source IR
        // available for retry. The two coactive groups have disjoint palettes.
        NamedAttrList metadata(plan.allocationCertificate);
        auto regular = builder.getDictionaryAttr(
            {builder.getNamedAttr("version", builder.getI64IntegerAttr(2)),
             builder.getNamedAttr("plan", builder.getI64IntegerAttr(0)),
             builder.getNamedAttr(
                 "directions", builder.getArrayAttr({builder.getDictionaryAttr(
                                   {builder.getNamedAttr("source", builder.getI64IntegerAttr(1)),
                                    builder.getNamedAttr("target", builder.getI64IntegerAttr(2)),
                                    builder.getNamedAttr("budget", builder.getI64IntegerAttr(mode == 1 ? 1 : 3)),
                                    builder.getNamedAttr("records", builder.getDenseI64ArrayAttr(records))})}))});
        metadata.set("regular_candidate", regular);
        plan.allocationCertificate = metadata.getDictionary(context);
        ScopedDiagnosticHandler diagnostics(context, [](Diagnostic&) { return success(); });
        auto layout = parseAttribute("#dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>", context);
        if (!layout) {
            return false;
        }
        (*module)->setAttr("dlti.dl_spec", layout);
        if (succeeded(fs::planPhysicalFamilyOutput(function, plan, {3, 5})) || plan.physicalOutput) {
            return false;
        }
        (*module)->removeAttr("dlti.dl_spec");
        if (succeeded(fs::planPhysicalFamilyOutput(function, plan, {3})) || plan.physicalOutput ||
            plan.familyRecipe != frozen || !plan.preparation.empty()) {
            return false;
        }
        if (failed(fs::planPhysicalFamilyOutput(function, plan, {3, 5})) || !fs::validPhysicalFamilyOutput(plan) ||
            !plan.preparation.empty()) {
            return false;
        }
        auto certificate = plan.allocationCertificate;
        plan.allocationCertificate = {};
        if (fs::validPhysicalFamilyOutput(plan)) {
            return false;
        }
        plan.allocationCertificate = certificate;
        if (succeeded(fs::planPhysicalFamilyOutput(function, plan, {0, 1}))) {
            return false;
        }
    }
    arena.reset(); // The frozen artifact retains every expression owner.
    if (failed(fs::materializeLogicalFamilies(plan)) || plan.familyRecipe || plan.familyProtocol != frozen ||
        plan.familyWork.materializations != 1) {
        return false;
    }
    if (physical) {
        DominanceInfo dominance(function);
        const fs::PhysicalRecordAllocation foreignRecord{999999, 1, 2, 0, 0, {0}, std::nullopt};
        const SmallVector<fs::PhysicalRecordAllocation> records{foreignRecord};
        const SmallVector<SmallVector<fs::TemplateCoordinate>> points{{}};
        auto foreign = fs::planPhysicalEndpointMap(sourceCut, records, points, 1, false, dominance);
        if (failed(foreign)) {
            return false;
        }
        auto& endpoint = plan.endpoints.front();
        const auto owned = endpoint.physicalMapping;
        endpoint.physicalMapping = *foreign;
        if (fs::validPhysicalFamilyEndpoint(plan, endpoint)) {
            return false;
        }
        endpoint.physicalMapping = owned;
        if (!fs::validPhysicalFamilyEndpoint(plan, endpoint)) {
            return false;
        }
        plan.completeInvocation = true;
        if (fs::validFamilyEndpointProtocol(plan)) {
            return false;
        }
        plan.completeInvocation = false;
    }
    if (physical && mode == 4) {
        for (const auto& endpoint : plan.endpoints) {
            if (endpoint.identity ||
                llvm::any_of(endpoint.memberCoordinates, [](Value value) { return bool(value); })) {
                return false;
            }
        }
    }
    const auto operations = plan.preparation.size();
    if (failed(fs::materializeLogicalFamilies(plan)) || plan.preparation.size() != operations ||
        plan.familyWork.materializations != 1) {
        return false;
    }
    for (bool enabled : {false, true}) {
        for (unsigned iteration = 0; iteration < count; ++iteration) {
            DenseMap<Value, uint64_t> values;
            values[loop.getInductionVar()] = iteration;
            values[entry->getArgument(0)] = enabled;
            function->walk([&](arith::ConstantOp op) {
                if (auto integer = dyn_cast<IntegerAttr>(op.getValue())) {
                    values[op.getResult()] = integer.getValue().getZExtValue();
                }
            });
            for (const auto& stage : plan.preparation) {
                if (!evaluateProtocolBlock(*stage.code, values)) {
                    return false;
                }
            }
            std::vector<uint64_t> sets, waits, expectedSets{100000}, expectedWaits;
            if (mode != 1 || iteration < count / 2) {
                for (unsigned i = 0; i < count; ++i) {
                    if ((mode == 2 ? i % 2 : i) == iteration && (mode != 4 || enabled) &&
                        (mode != 3 || (i < count / 2 ? enabled : !enabled))) {
                        expectedWaits.push_back(labels[i]);
                    }
                }
                expectedWaits.push_back(100000);
            } else {
                expectedWaits = {100000, labels[iteration]};
            }
            for (unsigned i = 0; i < count; ++i) {
                if ((mode == 2 ? i % 2 : i) == iteration && (mode != 4 || enabled) &&
                    (mode != 3 || (i < count / 2 ? enabled : !enabled))) {
                    expectedSets.push_back(labels[i]);
                }
            }
            for (const auto& endpoint : plan.endpoints) {
                if (!values.lookup(endpoint.guard)) {
                    continue;
                }
                uint64_t selected;
                if (physical) {
                    Block code;
                    builder.setInsertionPointToEnd(&code);
                    SmallVector<Value> tuple{endpoint.identity};
                    tuple.append(endpoint.memberCoordinates.begin() + 1, endpoint.memberCoordinates.end());
                    auto id = fs::emitPhysicalEndpointMap(
                        builder, location, *endpoint.physicalMapping, tuple, endpoint.memberCoordinates.front());
                    if (!id || !evaluateProtocolBlock(code, values)) {
                        return false;
                    }
                    selected = values.lookup(id);
                    for (auto& operation : code) {
                        for (auto value : operation.getResults()) {
                            values.erase(value);
                        }
                    }
                } else {
                    auto label = values.find(endpoint.memberCoordinates.front());
                    if (label == values.end()) {
                        return false;
                    }
                    selected = label->second;
                }
                (endpoint.kind == fs::LogicalCommandKind::Set ? sets : waits).push_back(selected);
            }
            if (physical) {
                for (auto* expected : {&expectedSets, &expectedWaits}) {
                    for (auto& label : *expected) {
                        label = label == 100000 ? 3 : 5;
                    }
                }
            }
            if (sets != expectedSets || waits != expectedWaits) {
                return false;
            }
        }
    }
    if (physical) {
        if (failed(fs::insertLogicalSynchronization(function, plan)) || failed(verify(function))) {
            return false;
        }
        bool logical = false;
        function->walk([&](Operation* operation) {
            logical |= operation->getName().getStringRef() == "pto.logical_set" ||
                       operation->getName().getStringRef() == "pto.logical_wait";
        });
        if (logical || function->hasAttr("pto.endpoint_families")) {
            return false;
        }
        if (mode == 4) {
            unsigned sets = 0, waits = 0;
            function.walk([&](Operation* operation) {
                auto name = operation->getName().getStringRef();
                sets += name == "pto.set_flag" || name == "pto.set_flag_dyn";
                waits += name == "pto.wait_flag" || name == "pto.wait_flag_dyn";
            });
            if (sets != 2 || waits != 2) {
                return false;
            }
        }
    }
    return true;
}
bool checkPhysicalMacroAdmission(mlir::MLIRContext* context)
{
    using namespace mlir;
    auto module = parseSourceString<ModuleOp>(
        R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @hidden(%mem: !pto.partition_tensor_view<1x1x1x64x32xf16>,
                   %idx: !pto.partition_tensor_view<1x1x1x1x32xi32>,
                   %dst: !pto.tile_buf<loc=mat, dtype=f16, rows=32, cols=32, v_row=32, v_col=32,
                       blayout=col_major, slayout=row_major, fractal=512, pad=0>) {
    pto.mgather ins(%mem, %idx : !pto.partition_tensor_view<1x1x1x64x32xf16>,
        !pto.partition_tensor_view<1x1x1x1x32xi32>)
      outs(%dst : !pto.tile_buf<loc=mat, dtype=f16, rows=32, cols=32, v_row=32, v_col=32,
          blayout=col_major, slayout=row_major, fractal=512, pad=0>) {coalesce = #pto<coalesce row>}
    return
  }
})mlir",
        context);
    if (!module) {
        return false;
    }
    auto function = *module->getOps<func::FuncOp>().begin();
    OpBuilder builder(context);
    auto certificate = builder.getDictionaryAttr({});
    ScopedDiagnosticHandler diagnostics(context, [](Diagnostic&) { return success(); });
    if (succeeded(protocol_fs::preflightPhysicalOutput(function, certificate))) {
        return false;
    }
    certificate = builder.getDictionaryAttr(
        {builder.getNamedAttr("kind", builder.getStringAttr("finite")),
         builder.getNamedAttr("macro_reservations", builder.getUnitAttr())});
    return succeeded(protocol_fs::preflightPhysicalOutput(function, certificate));
}
bool checkProtocolSelectionLifetime(mlir::MLIRContext* context)
{
    using namespace mlir;
    namespace fs = protocol_fs;
    OpBuilder builder(context);
    auto location = builder.getUnknownLoc();
    OwningOpRef<func::FuncOp> function(func::FuncOp::create(
        location, "protocol_selection_owner", builder.getFunctionType({}, {})));
    auto* body = function->addEntryBlock();
    builder.setInsertionPointToEnd(body);
    auto* cut = builder.create<arith::ConstantIndexOp>(location, 0).getOperation();
    auto* terminal = builder.create<func::ReturnOp>(location).getOperation();
    std::weak_ptr<const fs::RegionExpressions> proofOwner;
    std::shared_ptr<const fs::FamilyEndpointRecipe> cached;
    {
        auto arena = std::make_shared<fs::RegionExpressions>();
        fs::ContextualEndpointFamily endpoint;
        endpoint.piece.cut = {body, cut};
        endpoint.piece.kind = fs::EndpointKind::Barrier;
        endpoint.piece.sourcePipe = endpoint.piece.targetPipe = 1;
        endpoint.piece.members.push_back({0, {}});
        endpoint.activity = arena->boolean(true);
        auto recipe = fs::buildContextualFamilyEndpointRecipe(arena, {endpoint});
        if (failed(recipe)) { return false; }
        // The pending proof deliberately belongs to a different arena from
        // the deferred command. Only the frozen evidence inventory owns it.
        auto proof = std::make_shared<fs::RegionExpressions>();
        proofOwner = proof;
        auto choice = fs::qualifyTerminalCleanupCuts(
            {{body, cut}, {body, terminal}}, proof, proof->boolean(true));
        if (!choice) { return false; }
        fs::PreparedLogicalPlan plan(0);
        plan.familyRecipe = *recipe;
        fs::EndpointFamily family;
        family.id = 0;
        family.targetChoices = choice;
        plan.families.push_back(std::move(family));
        if (failed(fs::normalizeLogicalFamilies(plan))) { return false; }
        cached = plan.familyProtocol;
    }
    if (proofOwner.expired()) { return false; }
    cached.reset();
    return proofOwner.expired();
}
bool checkFamilyProtocols(mlir::MLIRContext* context)
{
    if (!checkPhysicalMacroAdmission(context) || !checkProtocolSelectionLifetime(context)) {
        return false;
    }
    for (unsigned count : {1U, 4U, 16U, 64U}) {
        for (unsigned mode = 0; mode < 5; ++mode) {
            if (!checkFamilyProtocol(context, count, mode) ||
                (mode != 2 && !checkFamilyProtocol(context, count, mode, true))) {
                llvm::errs() << "family protocol unfolding mismatch " << count << "," << mode << "\n";
                return false;
            }
        }
    }
    llvm::outs()
        << "normalized family protocol: 36 independent logical/combined order/ownership/scaling cases passed\n";
    return true;
}
} // namespace
#endif
