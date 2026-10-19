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
bool checkFamilyProtocol(mlir::MLIRContext* context, unsigned count, unsigned mode,
                         bool physical = false, bool batched = false)
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
    SmallVector<fs::ContextualEndpointFamily> endpoints;
    for (unsigned i = 0; i < count; ++i) {
        const auto label = uint32_t(10 + (37 * i) % count);
        const auto point = mode == 2 ? i % 2 : i;
        labels.push_back(label);
        family(label, point);
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
        // Two already-grouped numerical leaves exercise normalization of
        // pieces with many members, rather than only singleton input leaves.
        if (batched && (i + 1) % (count / 2)) { continue; }
        fs::ContextualFamilyExpressions expressions(arena);
        const SmallVector<uint8_t> numerical(endpoints.size(), mode != 4);
        auto leaf = fs::buildMixedFamilyEndpointRecipe(
            expressions, endpoints, numerical, {});
        endpoints.clear();
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
        !plan.familyProtocol || plan.familyWork.materializations ||
        plan.familyWork.normalizationInputs != (batched ? 4 : 2 * count)) {
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
    auto* publication = builder.create<arith::ConstantIndexOp>(location, 0).getOperation();
    auto* cut = builder.create<arith::ConstantIndexOp>(location, 1).getOperation();
    auto* terminal = builder.create<func::ReturnOp>(location).getOperation();
    std::weak_ptr<const fs::RegionExpressions> proofOwner;
    std::shared_ptr<const fs::FamilyEndpointRecipe> cached;
    {
        auto arena = std::make_shared<fs::RegionExpressions>();
        SmallVector<fs::ContextualEndpointFamily> endpoints;
        for (auto* before : {publication, cut, terminal}) {
            fs::ContextualEndpointFamily endpoint;
            endpoint.piece.cut = {body, before};
            endpoint.piece.kind = before == publication ? fs::EndpointKind::Set : fs::EndpointKind::Wait;
            endpoint.piece.sourcePipe = 1; endpoint.piece.targetPipe = 2;
            endpoint.piece.members.push_back({0, {}});
            endpoint.activity = arena->boolean(before != terminal);
            endpoint.ordinal = arena->constant(0);
            endpoints.push_back(std::move(endpoint));
        }
        auto recipe = fs::buildContextualFamilyEndpointRecipe(arena, endpoints);
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
        // The directory owns the actual singleton handoff, including the
        // inactive complementary cleanup side. Its proof arena is independent.
        family.sourcePipe = 1; family.targetPipe = 2;
        family.sourceCut = {body, publication}; family.targetCut = {body, cut};
        family.members.push_back({0, 0, 1, {}, {}});
        family.targetChoices = choice;
        plan.families.push_back(std::move(family));
        if (failed(fs::normalizeLogicalFamilies(plan))) { return false; }
        cached = plan.familyProtocol;
    }
    if (proofOwner.expired()) { return false; }
    cached.reset();
    return proofOwner.expired();
}
bool checkCountedPhysicalFamily(protocol_fs::PreparedLogicalPlan& plan, mlir::OpBuilder& builder)
{
    using namespace mlir;
    namespace fs = protocol_fs;
    const auto predicted = fs::physicalFamilyConstructionOperations(plan);
    if (!predicted || predicted != fs::physicalFamilyConstructionOperations(plan) ||
        !plan.preparation.empty() || !plan.endpoints.empty() ||
        failed(fs::materializeLogicalFamilies(plan))) { return false; }
    uint64_t constructed = 0;
    for (const auto& stage : plan.preparation) { constructed += stage.code->getOperations().size(); }
    for (const auto& endpoint : plan.endpoints) {
        if (!endpoint.physicalMapping) { continue; }
        Block code;
        builder.setInsertionPointToEnd(&code);
        SmallVector<Value> tuple{endpoint.identity};
        tuple.append(endpoint.memberCoordinates.begin() + 1, endpoint.memberCoordinates.end());
        (void)fs::emitPhysicalEndpointMap(builder, endpoint.before->getLoc(), *endpoint.physicalMapping,
            tuple, endpoint.memberCoordinates.front());
        constructed += code.getOperations().size();
    }
    return constructed == *predicted && !fs::physicalFamilyConstructionOperations(plan);
}
bool checkMatchingCoordinateRecipes(mlir::MLIRContext* context)
{
    using namespace mlir;
    namespace fs = protocol_fs;
    OpBuilder b(context);
    auto loc = b.getUnknownLoc();
    OwningOpRef<ModuleOp> module(ModuleOp::create(loc));
    auto f = func::FuncOp::create(loc, "matching_coordinates", b.getFunctionType({b.getIndexType()}, {}));
    module->push_back(f);
    auto* body = f.addEntryBlock();
    b.setInsertionPointToEnd(body);
    auto one = b.create<arith::ConstantIndexOp>(loc, 1);
    auto two = b.create<arith::ConstantIndexOp>(loc, 2);
    auto loop = b.create<scf::ForOp>(loc, one, body->getArgument(0), two);
    b.create<func::ReturnOp>(loc);
    b.setInsertionPointToStart(loop.getBody());
    auto* setCut = b.create<arith::ConstantIndexOp>(loc, 7).getOperation();
    auto* waitCut = b.create<arith::ConstantIndexOp>(loc, 8).getOperation();
    auto arena = std::make_shared<fs::RegionExpressions>();
    auto visit = arena->input(loop.getInductionVar());
    auto limit = arena->input(body->getArgument(0));
    auto live = arena->add(arena->add(visit, visit), arena->constant(1));
    auto dead = visit;
    for (unsigned i = 0; i < 32; ++i) { dead = arena->add(arena->add(arena->add(dead, dead), dead), limit); }
    SmallVector<fs::ContextualEndpointFamily> endpoints;
    for (auto kind : {fs::EndpointKind::Set, fs::EndpointKind::Wait}) {
        fs::ContextualEndpointFamily e;
        e.piece.kind = kind; e.piece.sourcePipe = 1; e.piece.targetPipe = 2;
        e.piece.cut = {loop.getBody(), kind == fs::EndpointKind::Set ? setCut : waitCut};
        e.piece.members.push_back({0, {}});
        e.activity = arena->lt(visit, limit); e.ordinal = arena->constant(0);
        // The acquisition recomputes the source tuple through a different
        // expression, rather than borrowing the publication's prepared SSA.
        e.matchingCoordinates = {dead, kind == fs::EndpointKind::Set ? live :
            arena->add(visit, arena->add(visit, arena->constant(1)))};
        endpoints.push_back(e);
    }
    auto recipe = fs::buildContextualFamilyEndpointRecipe(arena, endpoints);
    if (failed(recipe)) { return false; }
    recipe = fs::prefixFamilyInvocation(*recipe, arena, visit);
    if (failed(recipe)) { return false; }
    recipe = fs::maskFamilyActivity(*recipe, arena, arena->lnot(arena->eq(visit, arena->constant(5))));
    if (failed(recipe)) { return false; }
    recipe = fs::prefixFamilyInvocation(*recipe, arena, arena->constant(7));
    if (failed(recipe)) { return false; }
    auto makePlan = [&]() {
        auto plan = std::make_unique<fs::PreparedLogicalPlan>(0);
        plan->nestedIdentities = plan->groupedFamilies = plan->independentPieces = true;
        plan->familyRecipe = *recipe;
        fs::EndpointFamily family;
        family.sourcePipe = 1; family.targetPipe = 2;
        family.sourceCut = {loop.getBody(), setCut}; family.targetCut = {loop.getBody(), waitCut};
        family.members.push_back({0, 0, 1, {}, {}});
        plan->families.push_back(family);
        return plan;
    };
    auto logical = makePlan(), retry = makePlan(), physical = makePlan();
    if (failed(fs::normalizeLogicalFamilies(*physical))) { return false; }
    const auto frozen = physical->familyRecipe;
    const auto key = fs::normalizedFamilyProtocolKey(*frozen);
    auto changed = endpoints;
    changed.front().matchingCoordinates.back() = arena->add(live, arena->constant(1));
    auto changedRecipe = fs::buildContextualFamilyEndpointRecipe(arena, changed);
    auto originalRecipe = fs::buildContextualFamilyEndpointRecipe(arena, endpoints);
    if (failed(changedRecipe) || failed(originalRecipe)) { return false; }
    fs::PreparedLogicalPlan originalKey(0), changedKey(0);
    originalKey.familyRecipe = *originalRecipe; changedKey.familyRecipe = *changedRecipe;
    if (failed(fs::normalizeLogicalFamilies(originalKey)) || failed(fs::normalizeLogicalFamilies(changedKey)) ||
        fs::normalizedFamilyProtocolKey(*originalKey.familyRecipe) ==
            fs::normalizedFamilyProtocolKey(*changedKey.familyRecipe)) { return false; }
    physical->familyRecipe = *changedRecipe;
    if (succeeded(fs::normalizeLogicalFamilies(*physical))) { return false; }
    physical->familyRecipe = frozen;
    if (fs::normalizedFamilyProtocolKey(*physical->familyRecipe) != key) { return false; }
    auto rule = b.getDictionaryAttr({b.getNamedAttr("coordinate_count", b.getI64IntegerAttr(5)),
        b.getNamedAttr("base", b.getI64IntegerAttr(0)),
        b.getNamedAttr("terms", b.getArrayAttr({b.getDenseI64ArrayAttr({4, 1, 0, 3, 1})}))});
    auto group = b.getDictionaryAttr({b.getNamedAttr("source", b.getI64IntegerAttr(1)),
        b.getNamedAttr("target", b.getI64IntegerAttr(2)), b.getNamedAttr("budget", b.getI64IntegerAttr(3)),
        b.getNamedAttr("records", b.getDenseI64ArrayAttr({0})), b.getNamedAttr("strides", b.getDenseI64ArrayAttr({0})),
        b.getNamedAttr("phases", b.getDenseI64ArrayAttr({0})), b.getNamedAttr("conflicts", b.getDenseI64ArrayAttr({})),
        b.getNamedAttr("tuple_rules", b.getArrayAttr({rule}))});
    // Supplied mapping evidence tests lowering only; it is not a reuse proof.
    physical->allocationCertificate = b.getDictionaryAttr({b.getNamedAttr("version", b.getI64IntegerAttr(2)),
        b.getNamedAttr("plan", b.getI64IntegerAttr(0)),
        b.getNamedAttr("strategy", b.getStringAttr("regional-palettes")),
        b.getNamedAttr("groups", b.getArrayAttr({group}))});
    if (failed(fs::planPhysicalFamilyOutput(f, *physical, {0, 1, 2})) || !physical->preparation.empty() ||
        failed(fs::materializeLogicalFamilies(*logical)) || failed(fs::materializeLogicalFamilies(*retry)) ||
        !checkCountedPhysicalFamily(*physical, b)) { return false; }
    if (logical->endpoints.size() != 2 || physical->endpoints.size() != 2 ||
        logical->endpoints[0].memberCoordinates.back() == retry->endpoints[0].memberCoordinates.back()) {
        return false;
    }
    std::size_t logicalOps = 0, physicalOps = 0;
    for (const auto& stage : logical->preparation) { logicalOps += stage.code->getOperations().size(); }
    for (const auto& stage : physical->preparation) { physicalOps += stage.code->getOperations().size(); }
    if (logicalOps < physicalOps + 32) { return false; }
    for (uint64_t n : {0U, 1U, 4U, 9U}) {
        for (uint64_t iv = 1; iv < n; iv += 2) {
            uint64_t expectedDead = iv;
            for (unsigned i = 0; i < 32; ++i) { expectedDead = 3 * expectedDead + n; }
            SmallVector<SmallVector<uint64_t>> tuples;
            for (const auto* plan : {logical.get(), physical.get()}) {
                DenseMap<Value, uint64_t> values;
                values[loop.getInductionVar()] = iv; values[body->getArgument(0)] = n;
                for (const auto& stage : plan->preparation) {
                    if (!evaluateProtocolBlock(*stage.code, values)) { return false; }
                }
                for (const auto& endpoint : plan->endpoints) {
                    if (bool(values.lookup(endpoint.guard)) != (iv != 5)) { return false; }
                    if (plan == logical.get()) {
                        SmallVector<uint64_t> tuple;
                        for (auto value : endpoint.memberCoordinates) { tuple.push_back(values.lookup(value)); }
                        if (tuple.size() != 5 || tuple[0] != 0 || tuple[1] != 7 || tuple[2] != iv ||
                            tuple[3] != expectedDead || tuple[4] != 2 * iv + 1) {
                            return false;
                        }
                        tuples.push_back(tuple);
                    } else {
                        const auto& tuple = endpoint.memberCoordinates;
                        if (endpoint.identity || tuple.size() != 5 || tuple[1] || tuple[2] || tuple[3] || !tuple[4]) {
                            return false;
                        }
                        Block code;
                        b.setInsertionPointToEnd(&code);
                        auto id = fs::emitPhysicalEndpointMap(b, loc, *endpoint.physicalMapping,
                            {{}, {}, {}, {}, tuple[4]}, tuple[0]);
                        if (!evaluateProtocolBlock(code, values) || values.lookup(id) != (2 * iv + 1) % 3) {
                            return false;
                        }
                    }
                }
            }
            if (tuples.size() != 2 || tuples[0] != tuples[1]) { return false; }
        }
    }
    changed = endpoints; changed.back().matchingCoordinates.pop_back();
    auto mismatched = fs::buildContextualFamilyEndpointRecipe(arena, changed);
    if (failed(mismatched) || succeeded(fs::familyMatchingArities(**mismatched))) { return false; }
    fs::ContextualFamilyExpressions expressions(arena);
    if (succeeded(fs::buildMixedFamilyEndpointRecipe(expressions, endpoints, {1, 1}, {}))) { return false; }
    b.setInsertionPoint(loop.getBody()->getTerminator());
    auto late = b.create<func::CallOp>(loc, "unavailable", TypeRange{b.getIndexType()}, ValueRange{});
    changed = endpoints; changed.front().matchingCoordinates = {arena->input(late.getResult(0))};
    return failed(fs::buildContextualFamilyEndpointRecipe(arena, changed));
}
bool checkCoalescingConstructionComparison(mlir::MLIRContext* context)
{
    using namespace mlir;
    namespace fs = protocol_fs;
    OpBuilder b(context);
    auto loc = b.getUnknownLoc();
    OwningOpRef<func::FuncOp> f(func::FuncOp::create(
        loc, "coalescing_cost", b.getFunctionType({b.getIndexType(), b.getI1Type()}, {})));
    auto* body = f->addEntryBlock();
    b.setInsertionPointToEnd(body);
    auto* source = b.create<arith::ConstantIndexOp>(loc, 7).getOperation();
    auto* target = b.create<func::ReturnOp>(loc).getOperation();
    fs::PreparedLogicalPlan plan(0);
    auto arena = std::make_shared<fs::RegionExpressions>();
    SmallVector<fs::ContextualEndpointFamily> endpoints;
    for (uint32_t record : {0U, 1U}) {
        fs::EndpointFamily family;
        family.id = record; family.sourcePipe = 1; family.targetPipe = 2;
        family.sourceCut = {body, source}; family.targetCut = {body, target};
        family.members.push_back({record, 0, 1, {}, {}});
        plan.families.push_back(family);
        for (bool set : {true, false}) {
            fs::ContextualEndpointFamily endpoint;
            endpoint.piece.cut = {body, set ? source : target};
            endpoint.piece.kind = set ? fs::EndpointKind::Set : fs::EndpointKind::Wait;
            // Both records share pipes/displacement and therefore namespace 0;
            // their member labels still identify distinct original records.
            endpoint.piece.namespaceId = 0;
            endpoint.piece.sourcePipe = 1; endpoint.piece.targetPipe = 2;
            endpoint.piece.members.push_back({record, {}});
            endpoint.activity = record ? arena->input(body->getArgument(1)) : arena->boolean(true);
            endpoint.ordinal = arena->input(body->getArgument(0));
            endpoints.push_back(std::move(endpoint));
        }
    }
    auto recipe = fs::buildContextualFamilyEndpointRecipe(arena, endpoints);
    if (failed(recipe)) { return false; }
    plan.familyRecipe = *recipe;
    auto certificate = [&](int64_t budget) {
        auto direction = b.getDictionaryAttr({b.getNamedAttr("source", b.getI64IntegerAttr(1)),
            b.getNamedAttr("target", b.getI64IntegerAttr(2)), b.getNamedAttr("budget", b.getI64IntegerAttr(budget)),
            b.getNamedAttr("records", b.getDenseI64ArrayAttr({0, 1}))});
        return b.getDictionaryAttr({b.getNamedAttr("version", b.getI64IntegerAttr(2)),
            b.getNamedAttr("plan", b.getI64IntegerAttr(0)),
            b.getNamedAttr("directions", b.getArrayAttr({direction}))});
    };
    NamedAttrList evidence(certificate(3));
    evidence.set("regular_candidate", certificate(2));
    plan.allocationCertificate = evidence.getDictionary(context);
    return succeeded(fs::planPhysicalFamilyOutput(*f, plan, {0, 2, 5})) &&
        plan.mappingWork.candidates == 1 && plan.mappingWork.completeFootprints == 1 &&
        !plan.mappingWork.unknownCoalescing && !plan.mappingWork.unavailablePreparation &&
        !plan.mappingWork.exactFootprints && bool(fs::physicalFamilyConstructionOperations(plan));
}
bool checkMapConstructionFootprints(mlir::MLIRContext* context)
{
    using namespace mlir;
    namespace fs = protocol_fs;
    OpBuilder b(context);
    auto loc = b.getUnknownLoc();
    OwningOpRef<func::FuncOp> f(func::FuncOp::create(
        loc, "map_footprints", b.getFunctionType({b.getIndexType(), b.getIndexType()}, {})));
    auto* body = f->addEntryBlock();
    b.setInsertionPointToEnd(body);
    auto zero = b.create<arith::ConstantIndexOp>(loc, 0);
    auto one = b.create<arith::ConstantIndexOp>(loc, 1);
    auto limit = b.create<arith::ConstantIndexOp>(loc, 16);
    auto loop = b.create<scf::ForOp>(loc, zero, limit, one);
    b.create<func::ReturnOp>(loc);
    auto* cut = loop.getBody()->getTerminator();
    DominanceInfo dominance(*f);
    // Different selectors, composed tuple phases and residual record dispatch
    // all run through the same scalar schedule as production emission. Counts
    // include new constants even if equal original constants dominate the cut.
    for (unsigned mode = 0; mode < 4; ++mode) {
        SmallVector<std::shared_ptr<const fs::PhysicalEndpointMap>> candidates;
        for (unsigned alternative = 0; alternative < 2; ++alternative) {
            SmallVector<fs::PhysicalRecordAllocation> records;
            SmallVector<SmallVector<fs::TemplateCoordinate>> points;
            for (unsigned i = 0; i < 16; ++i) {
                const unsigned phase = alternative ? (i == 3 || i == 7) : i % 2;
                fs::PhysicalRecordAllocation record{int64_t(10 + 7 * i), 1, 2, 0, phase, {0, 1}};
                if (mode) {
                    const uint64_t modulus = mode == 1 ? 2 : 3;
                    if (mode == 2) { record.ids = {0, 2, 5}; }
                    if (mode == 3) { record.ids = {0, 1, 0}; }
                    record.tupleRule = fs::PhysicalTupleRule{
                        2, 0, {{1, mode == 1 ? 1U : 1U + i % 2, phase, modulus, 1}}};
                }
                records.push_back(std::move(record));
                points.push_back({{loop, i}});
            }
            auto mapping = fs::planPhysicalEndpointMap(cut, records, points, 2, true, dominance);
            if (failed(mapping)) { return false; }
            candidates.push_back(*mapping);
            const auto expectedWork = fs::physicalMapConstructionOperations(**mapping);
            if (!expectedWork) { return false; }
            for (unsigned repeat = 0; repeat < 2; ++repeat) {
                Block code;
                b.setInsertionPointToEnd(&code);
                auto selected = fs::emitPhysicalEndpointMap(b, loc, **mapping,
                    {body->getArgument(0), body->getArgument(1)}, {});
                if (static_cast<uint64_t>(std::distance(code.begin(), code.end())) != *expectedWork) {
                    return false;
                }
                for (unsigned i = 0; i < 16; ++i) {
                    for (uint64_t coordinate : {0U, 1U, 2U, 5U, 19U}) {
                        DenseMap<Value, uint64_t> values;
                        values[body->getArgument(0)] = 0;
                        values[body->getArgument(1)] = coordinate;
                        values[loop.getInductionVar()] = i;
                        const auto& record = records[i];
                        const uint64_t index = !mode ? record.phase :
                            (coordinate * record.tupleRule->terms[0].stride +
                             record.tupleRule->terms[0].phase) % record.ids.size();
                        if (!evaluateProtocolBlock(code, values) ||
                            values.lookup(selected) != static_cast<uint64_t>(record.ids[index])) { return false; }
                    }
                }
            }
        }
        fs::PhysicalMapComparisonKind kind;
        auto compared = fs::comparePhysicalMapConstruction(*candidates[0], *candidates[1], &kind);
        if (!compared || kind != fs::PhysicalMapComparisonKind::ExactFootprint ||
            compared->first != *fs::physicalMapConstructionOperations(*candidates[0]) ||
            compared->second != *fs::physicalMapConstructionOperations(*candidates[1])) { return false; }
    }
    return true;
}
bool checkPhysicalConstructionComparison(mlir::MLIRContext* context)
{
    using namespace mlir;
    namespace fs = protocol_fs;
    OpBuilder b(context);
    auto loc = b.getUnknownLoc();
    OwningOpRef<func::FuncOp> f(func::FuncOp::create(
        loc, "mapping_construction", b.getFunctionType({b.getIndexType()}, {})));
    auto* body = f->addEntryBlock();
    b.setInsertionPointToEnd(body);
    auto* cut = b.create<func::ReturnOp>(loc).getOperation();
    DominanceInfo dominance(*f);
    auto plan = [&](uint64_t stride, ArrayRef<int64_t> ids, bool nested = false, uint64_t phase = 0) {
        fs::PhysicalRecordAllocation record{0, 1, 2, stride, phase, {}};
        record.ids.append(ids.begin(), ids.end());
        return fs::planPhysicalEndpointMap(cut, {record}, {{}}, 1, nested, dominance);
    };
    // The table is indexed by certified slot, not by distinct ID. Nonconstant
    // repeated tables detect accidental deduplication or a changed period.
    for (auto ids : {SmallVector<int64_t>{0}, SmallVector<int64_t>{0, 1},
                     SmallVector<int64_t>{2, 3}, SmallVector<int64_t>{0, 2, 5},
                     SmallVector<int64_t>{6, 7}, SmallVector<int64_t>{0, 1, 2, 3, 4, 5, 6},
                     SmallVector<int64_t>{0, 1, 2, 3, 4, 5, 6, 7},
                     SmallVector<int64_t>{0, 0}, SmallVector<int64_t>{0, 1, 0},
                     SmallVector<int64_t>{5, 1, 5, 1}, SmallVector<int64_t>{0, 1, 0, 2, 0, 1}}) {
        for (uint64_t stride : {0U, 1U, 2U, 7U}) {
            for (uint64_t phase : {0U, 1U, 5U}) {
                auto mapping = plan(stride, ids, false, phase);
                if (failed(mapping)) { return false; }
                auto cost = fs::comparePhysicalMapConstruction(**mapping, **mapping);
                if (!cost || cost->first != cost->second) { return false; }
                Block code;
                b.setInsertionPointToEnd(&code);
                auto id = fs::emitPhysicalEndpointMap(b, loc, **mapping, {body->getArgument(0)}, {});
                if (static_cast<uint64_t>(std::distance(code.begin(), code.end())) != cost->first) { return false; }
                for (uint64_t ordinal : {0U, 1U, 2U, 5U, 19U}) {
                    DenseMap<Value, uint64_t> values;
                    values[body->getArgument(0)] = ordinal;
                    const auto expected = static_cast<uint64_t>(ids[(stride * ordinal + phase) % ids.size()]);
                    if (!evaluateProtocolBlock(code, values) || values.lookup(id) != expected) { return false; }
                }
            }
        }
    }
    {
        ScopedDiagnosticHandler diagnostics(context, [](Diagnostic&) { return success(); });
        for (auto ids : {SmallVector<int64_t>{}, SmallVector<int64_t>{-1}, SmallVector<int64_t>{8},
                         SmallVector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0}}) {
            if (succeeded(plan(1, ids))) { return false; }
        }
    }
    auto expensive = plan(1, {0, 2, 5}), cheap = plan(1, {0, 1}), fixed = plan(0, {0, 1});
    auto nested = plan(0, {3}, true);
    if (failed(expensive) || failed(cheap) || failed(fixed) || failed(nested)) { return false; }
    auto comparison = fs::comparePhysicalMapConstruction(**expensive, **cheap);
    auto reducedInputs = fs::comparePhysicalMapConstruction(**expensive, **fixed);
    auto constant = fs::comparePhysicalMapConstruction(**nested, **nested);
    if (!comparison || comparison->second >= comparison->first || !constant || constant->first != 1 ||
        !reducedInputs || reducedInputs->second >= reducedInputs->first ||
        fs::comparePhysicalMapConstruction(**fixed, **expensive)) { return false; }
    fs::PhysicalRecordAllocation tuple{0, 1, 2, 0, 0, {0, 1}};
    tuple.tupleRule = fs::PhysicalTupleRule{1, 0, {{0, 1, 0, 2, 1}}};
    auto supported = fs::planPhysicalEndpointMap(cut, {tuple}, {{}}, 1, true, dominance);
    fs::PhysicalMapComparisonKind kind;
    auto tupleComparison = succeeded(supported) ?
        fs::comparePhysicalMapConstruction(**supported, **supported, &kind) : std::nullopt;
    return tupleComparison && kind == fs::PhysicalMapComparisonKind::ExactFootprint &&
        checkMapConstructionFootprints(context) && checkCoalescingConstructionComparison(context);
}
bool checkCertifiedConstructionChoice(mlir::MLIRContext* context)
{
    using namespace mlir;
    namespace fs = protocol_fs;
    // Both supplied certificates are valid for this one sequential handoff.
    // A noncontiguous three-ID incumbent needs dispatch; the two-ID candidate
    // has identical live matching inputs and needs fewer constructed operations.
    // Modes cover a cheaper candidate, a cost tie, and fewer live inputs with
    // strictly cheaper map construction. No upstream savings are assumed.
    for (unsigned variant = 0; variant < 24; ++variant) {
        const unsigned mode = (variant / 2) % 4;
        const bool combined = variant % 2;
        const bool inventory = variant >= 8;
        const bool deferred = variant >= 16 && !combined;
        const bool retainTupleIncumbent = mode == 3 && !combined && !deferred;
        auto fail = [&](llvm::StringRef stage) {
            llvm::errs() << "certified construction variant=" << variant << " mode=" << mode
                << " combined=" << combined << " inventory=" << inventory << " deferred=" << deferred
                << " failed: " << stage << "\n";
            return false;
        };
        const SmallVector<int64_t> eligible =
            mode == 1 ? SmallVector<int64_t>{0, 1, 2} : SmallVector<int64_t>{0, 1, 3};
        const uint64_t expectedBudget = mode == 2 ? 1 : 2;
        OpBuilder b(context);
        auto loc = b.getUnknownLoc();
        OwningOpRef<ModuleOp> module(ModuleOp::create(loc));
        auto f = func::FuncOp::create(loc, "certified_construction_choice", b.getFunctionType({b.getIndexType()}, {}));
        module->push_back(f);
        auto* body = f.addEntryBlock();
        b.setInsertionPointToEnd(body);
        auto* setCut = b.create<arith::ConstantIndexOp>(loc, 7).getOperation();
        auto* waitCut = b.create<func::ReturnOp>(loc).getOperation();
        auto arena = std::make_shared<fs::RegionExpressions>();
        SmallVector<fs::ContextualEndpointFamily> endpoints;
        for (bool set : {true, false}) {
            fs::ContextualEndpointFamily e;
            e.piece.cut = {body, set ? setCut : waitCut};
            e.piece.kind = set ? fs::EndpointKind::Set : fs::EndpointKind::Wait;
            e.piece.sourcePipe = 1; e.piece.targetPipe = 2;
            e.piece.members.push_back({0, {}});
            e.activity = arena->boolean(true); e.ordinal = arena->input(body->getArgument(0));
            if (mode >= 2) {
                auto expensive = e.ordinal;
                for (unsigned depth = 0; depth < 32; ++depth) {
                    expensive = arena->div(arena->add(expensive, arena->constant(depth + 17)), arena->constant(3));
                }
                if (mode == 2) { e.ordinal = expensive; }
                else { e.matchingCoordinates.push_back(expensive); }
            }
            endpoints.push_back(e);
        }
        SmallVector<fs::ContextualSharedExpression> shared;
        if (mode >= 2) {
            shared.push_back({mode == 2 ? endpoints.front().ordinal : endpoints.front().matchingCoordinates.front(),
                setCut});
        }
        auto recipe = fs::buildContextualFamilyEndpointRecipe(arena, endpoints, shared);
        if (failed(recipe)) { return fail("recipe preparation"); }
        fs::PreparedLogicalPlan plan(0);
        // Mode 3 carries the ordinal plus one explicit matching coordinate.
        // Declare the tuple/independent-piece contract used by both endpoints.
        plan.nestedIdentities = plan.independentPieces = mode == 3;
        plan.familyRecipe = *recipe;
        fs::EndpointFamily family;
        family.sourcePipe = 1; family.targetPipe = 2;
        family.sourceCut = {body, setCut}; family.targetCut = {body, waitCut};
        family.members.push_back({0, 0, 1, {}, {}});
        plan.families.push_back(family);
        auto certificate = [&](int64_t budget) {
            auto direction = b.getDictionaryAttr({b.getNamedAttr("source", b.getI64IntegerAttr(1)),
                b.getNamedAttr("target", b.getI64IntegerAttr(2)), b.getNamedAttr("budget", b.getI64IntegerAttr(budget)),
                b.getNamedAttr("records", b.getDenseI64ArrayAttr({0}))});
            return b.getDictionaryAttr({b.getNamedAttr("version", b.getI64IntegerAttr(2)),
                b.getNamedAttr("plan", b.getI64IntegerAttr(0)),
                b.getNamedAttr("directions", b.getArrayAttr({direction}))});
        };
        NamedAttrList evidence(certificate(mode == 1 ? 2 : 3));
        evidence.set("regular_candidate", certificate(mode == 0 ? 2 : mode == 1 ? 3 : 1));
        if (mode == 3) {
            auto tupleCertificate = [&](int64_t coordinate) {
                auto rule = b.getDictionaryAttr({b.getNamedAttr("coordinate_count", b.getI64IntegerAttr(2)),
                    b.getNamedAttr("base", b.getI64IntegerAttr(0)),
                    b.getNamedAttr("terms", b.getArrayAttr({b.getDenseI64ArrayAttr({coordinate, 1, 0, 2, 1})}))});
                auto group = b.getDictionaryAttr({b.getNamedAttr("source", b.getI64IntegerAttr(1)),
                    b.getNamedAttr("target", b.getI64IntegerAttr(2)), b.getNamedAttr("budget", b.getI64IntegerAttr(2)),
                    b.getNamedAttr("records", b.getDenseI64ArrayAttr({0})),
                    b.getNamedAttr("strides", b.getDenseI64ArrayAttr({0})),
                    b.getNamedAttr("phases", b.getDenseI64ArrayAttr({0})),
                    b.getNamedAttr("conflicts", b.getDenseI64ArrayAttr({})),
                    b.getNamedAttr("tuple_rules", b.getArrayAttr({rule}))});
                return b.getDictionaryAttr({b.getNamedAttr("version", b.getI64IntegerAttr(2)),
                    b.getNamedAttr("plan", b.getI64IntegerAttr(0)),
                    // Match the real regional export's routing discriminator.
                    b.getNamedAttr("kind", b.getStringAttr("finite")),
                    b.getNamedAttr("strategy", b.getStringAttr("regional-palettes")),
                    b.getNamedAttr("groups", b.getArrayAttr({group}))});
            };
            // Sequential one-handoff protocol admits both maps. The candidate
            // swaps the live coordinate; only the full root schedule sees its
            // saving because the map suffix has exactly the same operations.
            evidence = NamedAttrList(tupleCertificate(1));
            evidence.set("regular_candidate", tupleCertificate(0));
        }
        if (inventory) {
            auto candidate = cast<DictionaryAttr>(evidence.get("regular_candidate"));
            NamedAttrList foreign(candidate);
            foreign.set("plan", b.getI64IntegerAttr(1));
            evidence.erase("regular_candidate");
            auto laterIncumbent = evidence.getDictionary(context);
            // Duplicate evidence is enumerated once. A foreign plan's otherwise
            // valid certificate cannot replace this protocol's incumbent.
            evidence.set("certified_candidates", b.getArrayAttr(
                {candidate, candidate, laterIncumbent, foreign.getDictionary(context)}));
        }
        plan.allocationCertificate = evidence.getDictionary(context);
        if (inventory && combined) {
            NamedAttrList malformed(plan.allocationCertificate);
            malformed.set("certified_candidates", b.getI64IntegerAttr(0));
            plan.allocationCertificate = malformed.getDictionary(context);
            ScopedDiagnosticHandler diagnostics(context, [](Diagnostic&) { return success(); });
            if (succeeded(fs::planPhysicalFamilyOutput(f, plan, eligible)) || plan.physicalOutput ||
                !plan.preparation.empty() || !plan.endpoints.empty()) { return fail("malformed inventory atomicity"); }
            plan.allocationCertificate = evidence.getDictionary(context);
        }
        if (combined && failed(fs::planPhysicalFamilyOutput(f, plan, eligible))) { return fail("physical plan"); }
        if (combined && (plan.mappingWork.candidates != (inventory ? 2U : 1U) ||
            plan.mappingWork.completeFootprints != (inventory ? 2U : 1U) ||
            plan.mappingWork.unknownInputs || plan.mappingWork.unknownCoalescing ||
            plan.mappingWork.unavailablePreparation)) { return fail("complete footprint inventory"); }
        if (combined && !checkCountedPhysicalFamily(plan, b)) { return fail("construction count parity"); }
        plan.deferredOutput = deferred;
        if (failed(fs::insertLogicalSynchronization(f, plan))) { return fail("logical materialization"); }
        if (!combined) {
            // Round-trip the complete evidence and protocol: no borrowed callback
            // or mutable plan survives into the reproducible split artifact.
            std::string text;
            llvm::raw_string_ostream output(text);
            module->print(output);
            auto saved = parseSourceString<ModuleOp>(output.str(), context);
            if (!saved) { return fail("saved parse"); }
            module = std::move(saved);
            f = *module->getOps<func::FuncOp>().begin();
            if (failed(fs::allocatePhysicalEventIds(f, eligible))) { return fail("saved allocation"); }
            auto choice = f->getAttrOfType<DictionaryAttr>("pto.physical_mapping_choice");
            const auto expectedStrategy = mode == 1 ? "cycle-cover" : inventory ? "certified-candidate" : "regular";
            // Eager split IR has already paid for both tuple inputs. Without
            // the owned recipe, changing live input masks has unknown cost.
            // A later identical incumbent can produce a tie receipt, but must
            // not make the unsupported candidate win.
            if (retainTupleIncumbent) {
                if (!inventory && choice) { return fail("unknown tuple comparison retained incumbent"); }
                if (inventory) {
                    auto strategy = choice ? choice.getAs<StringAttr>("strategy") : StringAttr{};
                    auto selected = choice ? choice.getAs<BoolAttr>("candidate_selected") : BoolAttr{};
                    auto index = choice ? choice.getAs<IntegerAttr>("candidate_index") : IntegerAttr{};
                    auto before = choice ? choice.getAs<IntegerAttr>("compared_incumbent_cost") : IntegerAttr{};
                    auto after = choice ? choice.getAs<IntegerAttr>("candidate_cost") : IntegerAttr{};
                    if (!strategy || strategy.getValue() != "cycle-cover" || !selected || selected.getValue() ||
                        !index || index.getInt() != 1 || !before || !after || before.getInt() != after.getInt()) {
                        return fail("identical tuple incumbent tie receipt");
                    }
                }
            } else if (!deferred &&
                (!choice || choice.getAs<StringAttr>("strategy").getValue() != expectedStrategy)) {
                return fail("selected strategy");
            }
            if (!deferred && inventory && mode == 0) {
                // The first candidate wins; the later original mapping loses.
                // Its reversed comparison must not replace the winning receipt.
                auto before = choice.getAs<IntegerAttr>("compared_incumbent_cost");
                auto after = choice.getAs<IntegerAttr>("candidate_cost");
                auto index = choice.getAs<IntegerAttr>("candidate_index");
                auto selected = choice.getAs<BoolAttr>("candidate_selected");
                if (!before || !after || !index || !selected || before.getInt() <= after.getInt() ||
                    index.getInt() != 0 || !selected.getValue()) { return fail("winning comparison receipt"); }
            }
        }
        if (failed(verify(f))) { return fail("output verification"); }
        if (mode >= 2 && combined) {
            // The selected constant map needs no ordinal. Its producer must
            // remain unmaterialized, including the expensive dead expression.
            if ((mode == 2 && !f.getArgument(0).use_empty()) ||
                f.walk([](arith::DivUIOp) { return WalkResult::interrupt(); }).wasInterrupted()) {
                return fail("unselected input liveness");
            }
        }
        // Evaluate the emitted arithmetic and command trace independently.
        for (uint64_t ordinal : {0U, 1U, 2U, 3U, 17U}) {
            uint64_t selectedCoordinate = ordinal;
            if (retainTupleIncumbent) {
                for (unsigned depth = 0; depth < 32; ++depth) {
                    selectedCoordinate = (selectedCoordinate + depth + 17) / 3;
                }
            }
            DenseMap<Value, uint64_t> values;
            values[f.getArgument(0)] = ordinal;
            SmallVector<std::pair<bool, uint64_t>> trace;
            std::function<bool(Block&)> evaluate = [&](Block& block) {
                for (auto& op : block) {
                    if (op.getName().getDialectNamespace() == "arith") {
                        Block code;
                        code.push_back(op.clone());
                        if (!evaluateProtocolBlock(code, values)) { return false; }
                        values[op.getResult(0)] = values.lookup(code.front().getResult(0));
                        values.erase(code.front().getResult(0));
                    } else if (auto branch = dyn_cast<scf::IfOp>(op)) {
                        auto& region = values.lookup(branch.getCondition()) ?
                            branch.getThenRegion() : branch.getElseRegion();
                        if (!region.empty() && !evaluate(region.front())) { return false; }
                    } else if (isa<pto::SetFlagOp, pto::WaitFlagOp, pto::SetFlagDynOp, pto::WaitFlagDynOp>(op)) {
                        auto fixed = op.getAttrOfType<pto::EventAttr>("event_id");
                        if (!fixed && (op.getNumOperands() != 1 || !values.count(op.getOperand(0)))) { return false; }
                        const uint64_t id = fixed ? static_cast<uint64_t>(fixed.getEvent()) :
                            values.lookup(op.getOperand(0));
                        trace.push_back({isa<pto::SetFlagOp, pto::SetFlagDynOp>(op), id});
                    } else if (!isa<func::ReturnOp, scf::YieldOp>(op)) { return false; }
                }
                return true;
            };
            if (!evaluate(f.front()) || trace != SmallVector<std::pair<bool, uint64_t>>{
                    {true, static_cast<uint64_t>(eligible[selectedCoordinate % expectedBudget])},
                    {false, static_cast<uint64_t>(eligible[selectedCoordinate % expectedBudget])}}) {
                llvm::errs() << "ordinal=" << ordinal << " commands=" << trace.size() << "\n";
                return fail("independent command trace");
            }
        }
    }
    return true;
}
bool checkFamilyProtocols(mlir::MLIRContext* context)
{
    auto check = [](bool passed, llvm::StringRef stage) {
        if (!passed) { llvm::errs() << "family protocol check failed: " << stage << "\n"; }
        return passed;
    };
    if (!check(checkPhysicalMacroAdmission(context), "physical macro admission") ||
        !check(checkProtocolSelectionLifetime(context), "selection lifetime") ||
        !check(checkMatchingCoordinateRecipes(context), "matching coordinates") ||
        !check(checkPhysicalConstructionComparison(context), "physical construction comparison") ||
        !check(checkCertifiedConstructionChoice(context), "certified construction choice")) {
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
    for (unsigned members : {128U, 512U, 2048U}) {
        if (!checkFamilyProtocol(context, 2 * members, 0, false, true)) {
            llvm::errs() << "grouped family normalization mismatch " << members << "\n";
            return false;
        }
    }
    llvm::outs()
        << "normalized family protocol: 36 independent logical/combined order/ownership/scaling cases passed\n";
    return true;
}
} // namespace
#endif
