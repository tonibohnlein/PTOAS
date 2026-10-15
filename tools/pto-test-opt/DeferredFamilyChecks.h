// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_DEFERRED_FAMILY_CHECKS_H
#define PTO_DEFERRED_FAMILY_CHECKS_H
#include "PTO/Transforms/FrontierSynch/Shared/DeferredFamily.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"
namespace {
std::string deferredText(mlir::Operation* operation)
{
    std::string text;
    llvm::raw_string_ostream out(text); operation->print(out); return text;
}
// Test-only resealing isolates semantic checks from stale-directory detection.
void resealDeferredContext(mlir::func::FuncOp function, mlir::DictionaryAttr metadata)
{
    using namespace mlir;
    NamedAttrList fields(metadata); fields.erase("protocol");
    auto unsignedMetadata = fields.getDictionary(function.getContext());
    std::string text;
    llvm::raw_string_ostream stream(text); unsignedMetadata.print(stream); stream.flush();
    llvm::SHA256 hash; hash.update(text);
    fields.set("protocol", StringAttr::get(function.getContext(), llvm::toHex(hash.final())));
    function->setAttr("pto.deferred_family", fields.getDictionary(function.getContext()));
}
bool evaluateDeferredScalar(mlir::Operation& operation, mlir::DenseMap<mlir::Value, llvm::APInt>& values)
{
    using namespace mlir;
    SmallVector<APInt> args;
    for (auto value : operation.getOperands()) {
        auto found = values.find(value);
        if (found == values.end()) { return false; }
        args.push_back(found->second);
    }
    if (operation.getNumResults() != 1) { return false; }
    APInt result;
    auto type = operation.getResult(0).getType();
    const unsigned width = type.isIndex() ? 64 : cast<IntegerType>(type).getWidth();
    if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
        result = cast<IntegerAttr>(constant.getValue()).getValue();
    } else if (isa<arith::IndexCastOp, arith::ExtSIOp>(operation)) { result = args[0].sextOrTrunc(width); }
    else if (isa<arith::IndexCastUIOp, arith::ExtUIOp, arith::TruncIOp>(operation)) {
        result = args[0].zextOrTrunc(width);
    }
    else if (isa<arith::AddIOp>(operation)) { result = args[0] + args[1]; }
    else if (isa<arith::SubIOp>(operation)) { result = args[0] - args[1]; }
    else if (isa<arith::MulIOp>(operation)) { result = args[0] * args[1]; }
    else if (isa<arith::AndIOp>(operation)) { result = args[0] & args[1]; }
    else if (isa<arith::OrIOp>(operation)) { result = args[0] | args[1]; }
    else if (isa<arith::XOrIOp>(operation)) { result = args[0] ^ args[1]; }
    else if (isa<arith::SelectOp>(operation)) { result = args[0].isZero() ? args[2] : args[1]; }
    else if (isa<arith::DivUIOp, arith::RemUIOp>(operation)) {
        if (args[1].isZero()) { return false; }
        result = isa<arith::DivUIOp>(operation) ? args[0].udiv(args[1]) : args[0].urem(args[1]);
    } else if (isa<arith::FloorDivSIOp, arith::CeilDivSIOp>(operation)) {
        if (!args[1].isStrictlyPositive()) { return false; }
        result = args[0].sdiv(args[1]);
        if (!args[0].srem(args[1]).isZero()) {
            if (isa<arith::FloorDivSIOp>(operation) && args[0].isNegative()) { --result; }
            if (isa<arith::CeilDivSIOp>(operation) && !args[0].isNegative()) { ++result; }
        }
    } else if (auto compare = dyn_cast<arith::CmpIOp>(operation)) {
        bool value = false;
        switch (compare.getPredicate()) {
        case arith::CmpIPredicate::eq: value = args[0] == args[1]; break;
        case arith::CmpIPredicate::ne: value = args[0] != args[1]; break;
        case arith::CmpIPredicate::ult: value = args[0].ult(args[1]); break;
        case arith::CmpIPredicate::ule: value = args[0].ule(args[1]); break;
        case arith::CmpIPredicate::ugt: value = args[0].ugt(args[1]); break;
        case arith::CmpIPredicate::uge: value = args[0].uge(args[1]); break;
        case arith::CmpIPredicate::slt: value = args[0].slt(args[1]); break;
        case arith::CmpIPredicate::sle: value = args[0].sle(args[1]); break;
        case arith::CmpIPredicate::sgt: value = args[0].sgt(args[1]); break;
        case arith::CmpIPredicate::sge: value = args[0].sge(args[1]); break;
        }
        result = APInt(1, value);
    } else { return false; }
    values[operation.getResult(0)] = result;
    return true;
}
bool deferredFixtureFailure(llvm::StringRef stage, unsigned line)
{
    llvm::errs() << "deferred family fixture failed: " << stage << " (line " << line << ")\n";
    return false;
}
bool checkDeferredExpressionArchive(mlir::MLIRContext* context)
{
    using namespace mlir;
    namespace fs = mlir::pto::frontiersynch;
    llvm::StringRef stage = "expression archive preparation";
    auto fail = [&](unsigned line) { return deferredFixtureFailure(stage, line); };
    auto module = parseSourceString<ModuleOp>(
        "module { func.func @archive(%x: index, %gate: i1) { return } }", context);
    if (!module) { return fail(__LINE__); }
    auto function = module->lookupSymbol<func::FuncOp>("archive");
    auto* cut = function.front().getTerminator();
    fs::RegionExpressions source, restored;
    auto x = source.input(function.getArgument(0)), gate = source.input(function.getArgument(1));
    auto zero = source.constant(0), three = source.constant(3);
    auto quotient = source.div(x, three), remainder = source.rem(x, three);
    auto system = fs::IntegerSystem::create(1, {{{fs::BoundInteger(1)}, fs::BoundInteger(2)},
        {{fs::BoundInteger(-1)}, fs::BoundInteger(2)}});
    if (failed(system)) { return fail(__LINE__); }
    std::string diagnostic;
    auto mapped = source.integerMappedPredicate(*system, {x}, {getAffineDimExpr(0, context).floorDiv(3)},
        1, {0}, diagnostic);
    if (!mapped) { return fail(__LINE__); }
    auto witness = source.integerWitness({{fs::BoundInteger(1)}, fs::BoundInteger(-2)}, fs::BoundInteger(3),
        {x}, 1, {0}, 0);
    SmallVector<fs::RegionExpressions::Id> roots{source.slt(x, zero), source.sle(x, source.constant(INT64_MAX)),
        source.sub(zero, x), quotient, remainder, source.select(gate, quotient, remainder), *mapped, witness};
    Builder b(context);
    auto graph = source.archive(roots, context, [&](Value value) -> Attribute {
        auto argument = dyn_cast<BlockArgument>(value);
        return argument && argument.getOwner() == &function.front() ?
            b.getI64IntegerAttr(argument.getArgNumber()) : Attribute{};
    });
    if (failed(graph)) { return fail(__LINE__); }
    auto decodeValue = [&](Attribute raw) -> Value {
        auto index = dyn_cast<IntegerAttr>(raw);
        if (!index || !index.getType().isInteger(64) || index.getInt() < 0 || index.getInt() >= 2) { return {}; }
        return function.getArgument(index.getInt());
    };
    auto decoded = restored.restoreArchive(*graph, decodeValue);
    if (failed(decoded) || decoded->size() != roots.size()) { return fail(__LINE__); }
    stage = "expression archive independent scalar equivalence";
    const uint64_t samples[] = {0, 1, 2, 3, 6, 7, uint64_t(-1), uint64_t(-2), uint64_t(-7),
        uint64_t(INT64_MIN), uint64_t(INT64_MAX), UINT64_MAX};
    for (uint64_t sample : samples) {
        for (bool predicate : {false, true}) {
            APInt signedInput(128, sample, true);
            APInt floor = signedInput.sdiv(APInt(128, 3));
            if (signedInput.isNegative() && !signedInput.srem(APInt(128, 3)).isZero()) { --floor; }
            auto numerator = signedInput - APInt(128, 2);
            auto expectedWitness = numerator.sdiv(APInt(128, 3));
            if (numerator.isNegative() && !numerator.srem(APInt(128, 3)).isZero()) { --expectedWitness; }
            SmallVector<uint64_t> expected{APInt(64, sample).isNegative(),
                APInt(64, sample).sle(APInt(64, INT64_MAX)), uint64_t(0) - sample,
                sample / 3, sample % 3, predicate ? sample / 3 : sample % 3,
                floor.sge(APInt(128, uint64_t(-2), true)) && floor.sle(APInt(128, 2)),
                expectedWitness.trunc(64).getZExtValue()};
            for (unsigned version = 0; version < 2; ++version) {
                auto& arena = version ? restored : source;
                auto selected = version ? ArrayRef<fs::RegionExpressions::Id>(*decoded) :
                    ArrayRef<fs::RegionExpressions::Id>(roots);
                Block code;
                OpBuilder builder(context); builder.setInsertionPointToEnd(&code);
                DenseMap<fs::RegionExpressions::Id, Value> memo;
                SmallVector<Value> outputs;
                for (auto root : selected) {
                    auto output = arena.emit(root, builder, cut, memo);
                    if (failed(output)) {
                        llvm::errs() << "archive emission: " << arena.error() << "\n";
                        return fail(__LINE__);
                    }
                    outputs.push_back(*output);
                }
                DenseMap<Value, APInt> values;
                values[function.getArgument(0)] = APInt(64, sample);
                values[function.getArgument(1)] = APInt(1, predicate);
                for (auto& op : code) { if (!evaluateDeferredScalar(op, values)) { return fail(__LINE__); } }
                for (auto [output, value] : llvm::zip(outputs, expected)) {
                    if (values.lookup(output).getZExtValue() != value) {
                        llvm::errs() << "archive sample=" << sample << " predicate=" << predicate
                            << " version=" << version << " expected=" << value
                            << " actual=" << values.lookup(output).getZExtValue() << "\n";
                        return fail(__LINE__);
                    }
                }
            }
        }
    }
    // Untrusted archives cannot establish undefined division, forward edges,
    // unknown opcodes or wrong result types. Every failure rolls back IDs/errors.
    stage = "expression archive malformed refusal";
    for (unsigned mode = 0; mode < 7; ++mode) {
        NamedAttrList data(*graph);
        auto array = cast<ArrayAttr>(data.get("nodes"));
        SmallVector<Attribute> nodes(array.begin(), array.end());
        if (mode == 0) { data.set("version", b.getI64IntegerAttr(99)); }
        else {
            bool changed = false;
            for (auto& raw : nodes) {
                NamedAttrList node(cast<DictionaryAttr>(raw));
                auto opcode = cast<IntegerAttr>(node.get("opcode")).getInt();
                NamedAttrList payload(cast<DictionaryAttr>(node.get("payload")));
                if (mode == 1 && opcode == 0 && payload.get("literal") == b.getI64IntegerAttr(3)) {
                    payload.set("literal", b.getI64IntegerAttr(0)); node.set("payload", payload.getDictionary(context));
                } else if (mode == 2 && opcode == 4) {
                    node.set("operands", b.getDenseI64ArrayAttr({static_cast<int64_t>(nodes.size()), 0}));
                } else if (mode == 3 && opcode == 4) { node.set("opcode", b.getI64IntegerAttr(99)); }
                else if (mode == 4 && opcode == 4) { node.set("boolean", b.getBoolAttr(true)); }
                else if ((mode == 5 || mode == 6) && opcode == 15 &&
                         cast<BoolAttr>(payload.get("predicate")).getValue()) {
                    if (mode == 5) {
                        payload.set("dimensions", b.getI64IntegerAttr(UINT_MAX));
                        payload.set("empty", b.getBoolAttr(true));
                    } else {
                        payload.set("rows",
                            b.getArrayAttr({b.getArrayAttr({b.getArrayAttr({}), b.getStringAttr("0")})}));
                    }
                    node.set("payload", payload.getDictionary(context));
                } else { continue; }
                raw = node.getDictionary(context); changed = true; break;
            }
            if (!changed) { return fail(__LINE__); }
            data.set("nodes", b.getArrayAttr(nodes));
        }
        fs::RegionExpressions destination;
        destination.constant(123);
        const auto before = destination.size();
        if (succeeded(destination.restoreArchive(data.getDictionary(context), decodeValue)) ||
            destination.size() != before || !destination.error().empty()) { return fail(__LINE__); }
    }
    return true;
}
bool checkDeferredFamilies(mlir::MLIRContext* context)
{
    using namespace mlir;
    namespace fs = mlir::pto::frontiersynch;
    llvm::StringRef stage = "expression archive prerequisite";
    auto fail = [&](unsigned line) { return deferredFixtureFailure(stage, line); };
    if (!checkDeferredExpressionArchive(context)) { return fail(__LINE__); }
    constexpr const char* source = R"mlir(
      module attributes {pto.target_arch = "a3"} {
        func.func @deferred(%n: index) {
          %one = arith.constant 1 : index
          %two = arith.constant 2 : index
          scf.for %i = %one to %n step %two {
            %source = arith.addi %i, %one : index
            %target = arith.addi %i, %two : index
          }
          return
        }
      }
    )mlir";
    auto makePlan = [&](func::FuncOp function, int64_t firstVisit = 1)
        -> FailureOr<std::unique_ptr<fs::PreparedLogicalPlan>> {
        auto loop = *function.getOps<scf::ForOp>().begin();
        auto* sourceCut = &loop.getBody()->front();
        auto* targetCut = sourceCut->getNextNode();
        fs::NumericalFamilyRecipe recipe;
        recipe.outer = loop;
        fs::EndpointPiece first, last;
        first.kind = fs::EndpointKind::Set; last.kind = fs::EndpointKind::Wait;
        first.cut = {sourceCut->getBlock(), sourceCut}; last.cut = {targetCut->getBlock(), targetCut};
        first.sourcePipe = last.sourcePipe = 1; first.targetPipe = last.targetPipe = 2;
        auto plan = std::make_unique<fs::PreparedLogicalPlan>(0);
        plan->groupedFamilies = plan->independentPieces = true;
        fs::EndpointFamily family;
        family.id = 0; family.sourcePipe = 1; family.targetPipe = 2;
        family.sourceCut = first.cut; family.targetCut = last.cut;
        const uint32_t labels[] = {2, 0, 3, 1};
        for (unsigned i = 0; i < 4; ++i) {
            SmallVector<fs::TemplateCoordinate> point{{loop, firstVisit + 2 * i}};
            first.members.push_back({labels[i], point}); last.members.push_back({labels[i], point});
            family.members.push_back({labels[i], 0, 1, point, point});
        }
        plan->families.push_back(std::move(family));
        recipe.pieces = {first, last};
        auto restored = fs::restoreNumericalFamilyRecipe(recipe);
        if (failed(restored)) { return failure(); }
        plan->familyRecipe = *restored;
        Builder b(context);
        auto group = b.getDictionaryAttr({b.getNamedAttr("source", b.getI64IntegerAttr(1)),
            b.getNamedAttr("target", b.getI64IntegerAttr(2)), b.getNamedAttr("budget", b.getI64IntegerAttr(1)),
            b.getNamedAttr("records", b.getDenseI64ArrayAttr({0, 1, 2, 3})),
            b.getNamedAttr("strides", b.getDenseI64ArrayAttr({0, 0, 0, 0})),
            b.getNamedAttr("phases", b.getDenseI64ArrayAttr({0, 0, 0, 0})),
            b.getNamedAttr("conflicts", b.getDenseI64ArrayAttr({}))});
        // Supplied placement tests codec/lowering, not allocation optimality.
        plan->allocationCertificate = b.getDictionaryAttr({b.getNamedAttr("version", b.getI64IntegerAttr(2)),
            b.getNamedAttr("plan", b.getI64IntegerAttr(0)),
            b.getNamedAttr("strategy", b.getStringAttr("regional-palettes")),
            b.getNamedAttr("groups", b.getArrayAttr({group}))});
        return plan;
    };
    auto countArithmetic = [](func::FuncOp function) {
        unsigned count = 0;
        function.walk([&](Operation* op) { count += op->getName().getDialectNamespace() == "arith"; });
        return count;
    };
    stage = "numerical route preparation";
    std::string saved, combined;
    unsigned physicalOps = 0, logicalOps = 0;
    for (unsigned mode = 0; mode < 3; ++mode) {
        auto module = parseSourceString<ModuleOp>(source, context);
        if (!module) { return fail(__LINE__); }
        auto function = module->lookupSymbol<func::FuncOp>("deferred");
        auto plan = makePlan(function);
        if (failed(plan)) { return fail(__LINE__); }
        const auto originalOps = countArithmetic(function);
        if (mode == 0) {
            (*plan)->deferredOutput = true;
        } else if (mode == 1 && failed(fs::planPhysicalFamilyOutput(function, **plan, {3}))) { return fail(__LINE__); }
        if (failed(fs::insertLogicalSynchronization(function, **plan)) || failed(verify(function))) {
            return fail(__LINE__);
        }
        if (mode == 0) {
            if (countArithmetic(function) != originalOps || (*plan)->familyWork.materializations ||
                !(*plan)->preparation.empty()) { return fail(__LINE__); }
            saved = deferredText(module->getOperation());
        } else if (mode == 1) {
            combined = deferredText(function); physicalOps = countArithmetic(function);
        } else { logicalOps = countArithmetic(function); }
    }
    stage = "numerical construction liveness";
    if (physicalOps >= logicalOps || saved.empty()) { return fail(__LINE__); }
    // Every producer owner and original IR has died before parsing this artifact.
    stage = "numerical saved allocation equivalence";
    auto restored = parseSourceString<ModuleOp>(saved, context);
    if (!restored) { return fail(__LINE__); }
    auto function = restored->lookupSymbol<func::FuncOp>("deferred");
    if (failed(fs::allocatePhysicalEventIds(function, {3})) || failed(verify(function)) ||
        countArithmetic(function) != physicalOps || deferredText(function) != combined) { return fail(__LINE__); }
    auto trace = [&](func::FuncOp function, uint64_t bound,
                     SmallVector<std::pair<bool, uint64_t>>& actual) {
        DenseMap<Value, APInt> values;
        values[function.getArgument(0)] = APInt(64, bound);
        std::function<bool(Block&)> execute = [&](Block& block) {
            for (auto& op : block) {
                if (op.getName().getDialectNamespace() == "arith") {
                    if (!evaluateDeferredScalar(op, values)) { return fail(__LINE__); }
                } else if (auto loop = dyn_cast<scf::ForOp>(op)) {
                    if (!values.count(loop.getLowerBound()) || !values.count(loop.getUpperBound()) ||
                        !values.count(loop.getStep()) || values.lookup(loop.getStep()).isZero()) {
                        return fail(__LINE__);
                    }
                    for (uint64_t i = values.lookup(loop.getLowerBound()).getZExtValue();
                         i < values.lookup(loop.getUpperBound()).getZExtValue();
                         i += values.lookup(loop.getStep()).getZExtValue()) {
                        values[loop.getInductionVar()] = APInt(64, i);
                        if (!execute(*loop.getBody())) { return fail(__LINE__); }
                    }
                } else if (auto branch = dyn_cast<scf::IfOp>(op)) {
                    if (!values.count(branch.getCondition())) { return fail(__LINE__); }
                    auto& region = !values.lookup(branch.getCondition()).isZero() ?
                        branch.getThenRegion() : branch.getElseRegion();
                    if (region.empty()) {
                        if (branch.getNumResults()) { return fail(__LINE__); }
                        continue;
                    }
                    if (!execute(region.front())) { return fail(__LINE__); }
                    auto yielded = dyn_cast<scf::YieldOp>(region.front().getTerminator());
                    if (!yielded || yielded.getNumOperands() != branch.getNumResults()) { return fail(__LINE__); }
                    for (auto [result, operand] : llvm::zip(branch.getResults(), yielded.getOperands())) {
                        auto found = values.find(operand);
                        if (found == values.end()) { return fail(__LINE__); }
                        const APInt value = found->second;
                        values[result] = value;
                    }
                } else if (isa<pto::SetFlagOp, pto::WaitFlagOp, pto::SetFlagDynOp, pto::WaitFlagDynOp>(op)) {
                    auto fixed = op.getAttrOfType<pto::EventAttr>("event_id");
                    if (!fixed && (op.getNumOperands() != 1 || !values.count(op.getOperand(0)))) {
                        return fail(__LINE__);
                    }
                    auto id = fixed ? static_cast<uint64_t>(fixed.getEvent()) :
                        values.lookup(op.getOperand(0)).getZExtValue();
                    actual.push_back({isa<pto::SetFlagOp, pto::SetFlagDynOp>(op), id});
                } else if (!isa<func::ReturnOp, scf::YieldOp>(op)) { return fail(__LINE__); }
            }
            return true;
        };
        return execute(function.front());
    };
    stage = "numerical independent command trace";
    // Independent finite execution includes zero/partial invocations, skipped
    // visits and out-of-domain visits. Record labels never select more commands.
    for (uint64_t bound : {0U, 1U, 2U, 5U, 8U, 11U}) {
        SmallVector<std::pair<bool, uint64_t>> actual, expected;
        for (uint64_t i = 1; i < bound && i <= 7; i += 2) {
            expected.push_back({true, 3}); expected.push_back({false, 3});
        }
        if (!trace(function, bound, actual) || actual != expected) { return fail(__LINE__); }
    }
    // Two coactive pieces at each cut retain their order and distinct source
    // occurrence across displacement. Independent IDs permit simultaneous live
    // publications; the next visit acquires those exact source occurrences.
    stage = "displaced route preparation";
    std::string displacedSaved, displacedCombined;
    for (unsigned mode = 0; mode < 2; ++mode) {
        auto module = parseSourceString<ModuleOp>(source, context);
        if (!module) { return fail(__LINE__); }
        auto f = module->lookupSymbol<func::FuncOp>("deferred");
        auto loop = *f.getOps<scf::ForOp>().begin();
        auto* sourceCut = &loop.getBody()->front();
        auto* targetCut = loop.getBody()->getTerminator();
        auto plan = std::make_unique<fs::PreparedLogicalPlan>(0);
        plan->groupedFamilies = plan->independentPieces = true;
        fs::NumericalFamilyRecipe recipe;
        recipe.outer = loop;
        for (uint32_t record = 0; record < 2; ++record) {
            fs::EndpointPiece first, last;
            first.kind = fs::EndpointKind::Set; last.kind = fs::EndpointKind::Wait;
            first.cut = {sourceCut->getBlock(), sourceCut}; last.cut = {targetCut->getBlock(), targetCut};
            first.sourcePipe = last.sourcePipe = 1; first.targetPipe = last.targetPipe = 2;
            first.displacement = last.displacement = 1; first.order = last.order = record;
            SmallVector<fs::TemplateCoordinate> from{{loop, 1}}, to{{loop, 3}};
            first.members.push_back({record, from}); last.members.push_back({record, to});
            recipe.pieces.push_back(first); recipe.pieces.push_back(last);
            fs::EndpointFamily family;
            family.id = record; family.sourcePipe = 1; family.targetPipe = 2; family.displacement = 1;
            family.sourceOrder = family.targetOrder = record;
            family.sourceCut = first.cut; family.targetCut = last.cut;
            family.members.push_back({record, 0, 1, from, to});
            plan->families.push_back(std::move(family));
        }
        auto restored = fs::restoreNumericalFamilyRecipe(recipe);
        if (failed(restored)) { return fail(__LINE__); }
        plan->familyRecipe = *restored;
        Builder b(context);
        auto group = b.getDictionaryAttr({b.getNamedAttr("source", b.getI64IntegerAttr(1)),
            b.getNamedAttr("target", b.getI64IntegerAttr(2)), b.getNamedAttr("budget", b.getI64IntegerAttr(2)),
            b.getNamedAttr("records", b.getDenseI64ArrayAttr({0, 1})),
            b.getNamedAttr("strides", b.getDenseI64ArrayAttr({0, 0})),
            b.getNamedAttr("phases", b.getDenseI64ArrayAttr({0, 1})),
            b.getNamedAttr("conflicts", b.getDenseI64ArrayAttr({}))});
        plan->allocationCertificate = b.getDictionaryAttr({b.getNamedAttr("version", b.getI64IntegerAttr(2)),
            b.getNamedAttr("plan", b.getI64IntegerAttr(0)),
            b.getNamedAttr("strategy", b.getStringAttr("regional-palettes")),
            b.getNamedAttr("groups", b.getArrayAttr({group}))});
        if (mode == 0) { plan->deferredOutput = true; }
        else if (failed(fs::planPhysicalFamilyOutput(f, *plan, {3, 4}))) { return fail(__LINE__); }
        if (failed(fs::insertLogicalSynchronization(f, *plan))) { return fail(__LINE__); }
        if (mode == 0) {
            // Numerical leaves with an implicit terminator cut use the owned
            // v2 directory; ordinary numerical cases above retain v1 output.
            auto metadata = f->getAttrOfType<DictionaryAttr>("pto.deferred_family");
            if (!metadata || metadata.getAs<IntegerAttr>("version").getInt() != 2) { return fail(__LINE__); }
            displacedSaved = deferredText(module->getOperation());
        } else { displacedCombined = deferredText(f); }
    }
    stage = "displaced saved allocation equivalence";
    auto displaced = parseSourceString<ModuleOp>(displacedSaved, context);
    if (!displaced) { return fail(__LINE__); }
    auto displacedFunction = displaced->lookupSymbol<func::FuncOp>("deferred");
    if (failed(fs::allocatePhysicalEventIds(displacedFunction, {3, 4})) ||
        deferredText(displacedFunction) != displacedCombined) { return fail(__LINE__); }
    stage = "displaced independent command trace";
    for (uint64_t bound : {0U, 1U, 2U, 5U, 8U, 11U}) {
        SmallVector<std::pair<bool, uint64_t>> actual, expected;
        if (bound > 3) { expected = {{true, 3}, {true, 4}, {false, 3}, {false, 4}}; }
        if (!trace(displacedFunction, bound, actual) || actual != expected) { return fail(__LINE__); }
    }
    stage = "displaced order corruption refusal";
    {
        auto broken = parseSourceString<ModuleOp>(displacedSaved, context);
        if (!broken) { return fail(__LINE__); }
        auto f = broken->lookupSymbol<func::FuncOp>("deferred");
        pto::LogicalFamilyOp marker;
        f.walk([&](pto::LogicalFamilyOp op) { if (!marker) { marker = op; } });
        auto pieces = marker.getPieces();
        if (pieces.size() != 2) { return fail(__LINE__); }
        Builder b(context);
        auto metadata = f->getAttrOfType<DictionaryAttr>("pto.deferred_family");
        auto rawNodes = metadata.getAs<ArrayAttr>("nodes");
        SmallVector<Attribute> nodes(rawNodes.begin(), rawNodes.end());
        for (auto [i, reference] : llvm::enumerate(pieces)) {
            auto entry = cast<DictionaryAttr>(reference);
            const auto nodeId = entry.getAs<IntegerAttr>("node").getInt();
            const auto pieceId = entry.getAs<IntegerAttr>("piece").getInt();
            NamedAttrList node(cast<DictionaryAttr>(nodes[nodeId]));
            auto rawPieces = cast<ArrayAttr>(node.get("pieces"));
            SmallVector<Attribute> retained(rawPieces.begin(), rawPieces.end());
            NamedAttrList wrapper(cast<DictionaryAttr>(retained[pieceId]));
            NamedAttrList piece(cast<DictionaryAttr>(wrapper.get("piece")));
            piece.set("order", b.getI64IntegerAttr(1 - i));
            wrapper.set("piece", piece.getDictionary(context));
            retained[pieceId] = wrapper.getDictionary(context);
            node.set("pieces", b.getArrayAttr(retained)); nodes[nodeId] = node.getDictionary(context);
        }
        NamedAttrList changed(metadata); changed.set("nodes", b.getArrayAttr(nodes));
        f->setAttr("pto.deferred_family", changed.getDictionary(context));
        const auto before = deferredText(f);
        ScopedDiagnosticHandler diagnostics(context, [](Diagnostic&) { return success(); });
        if (succeeded(fs::allocatePhysicalEventIds(f, {3, 4})) || deferredText(f) != before) { return fail(__LINE__); }
    }
    stage = "numerical malformed refusal";
    for (unsigned mode = 0; mode < 14; ++mode) {
        auto broken = parseSourceString<ModuleOp>(saved, context);
        if (!broken) { return fail(__LINE__); }
        auto f = broken->lookupSymbol<func::FuncOp>("deferred");
        SmallVector<pto::LogicalFamilyOp> markers;
        f.walk([&](pto::LogicalFamilyOp op) { markers.push_back(op); });
        if (markers.size() != 2) { return fail(__LINE__); }
        Builder b(context);
        if (mode == 0) {
            NamedAttrList data(f->getAttrOfType<DictionaryAttr>("pto.deferred_family"));
            data.set("version", b.getI64IntegerAttr(99));
            f->setAttr("pto.deferred_family", data.getDictionary(context));
        } else if (mode == 1) { markers.back()->erase(); }
        else if (mode == 2) { markers.front()->setAttr("axis_ids", b.getDenseI64ArrayAttr({99})); }
        else if (mode == 3) { markers.front()->setOperand(0, f.getArgument(0)); }
        else if (mode == 4) {
            auto loop = *f.getOps<scf::ForOp>().begin();
            markers.front()->moveBefore(loop);
        } else if (mode == 5) {
            auto pieces = markers.front()->getAttrOfType<ArrayAttr>("pieces");
            NamedAttrList piece(cast<DictionaryAttr>(pieces[0]));
            piece.set("displacement", b.getI64IntegerAttr(1));
            markers.front()->setAttr("pieces", b.getArrayAttr({piece.getDictionary(context)}));
        }
        else if (mode == 6) { markers.front()->moveBefore(markers.back()); }
        else if (mode == 7) { markers.front()->setAttr("unsupported", b.getUnitAttr()); }
        else if (mode == 8) { f->removeAttr("pto.deferred_family"); }
        else if (mode == 9) {
            NamedAttrList data(f->getAttrOfType<DictionaryAttr>("pto.deferred_family"));
            data.set("complete", b.getBoolAttr(true)); f->setAttr("pto.deferred_family", data.getDictionary(context));
        } else if (mode == 10) {
            NamedAttrList data(f->getAttrOfType<DictionaryAttr>("pto.deferred_family"));
            auto families = cast<ArrayAttr>(data.get("families"));
            NamedAttrList family(cast<DictionaryAttr>(families[0]));
            family.set("source_order", b.getI64IntegerAttr(99));
            data.set("families", b.getArrayAttr({family.getDictionary(context)}));
            f->setAttr("pto.deferred_family", data.getDictionary(context));
        } else if (mode == 11) {
            OpBuilder builder(markers.front());
            auto badAxis = builder.create<arith::ConstantIntOp>(markers.front().getLoc(), 0, 64);
            markers.front()->setOperand(0, badAxis);
        } else {
            markers.front()->setAttr(mode == 12 ? "plan_id" : "cut_id",
                IntegerAttr::get(IntegerType::get(context, 128), APInt::getSignedMaxValue(128)));
        }
        const auto before = deferredText(f);
        ScopedDiagnosticHandler diagnostics(context, [](Diagnostic&) { return success(); });
        if (succeeded(fs::allocatePhysicalEventIds(f, {3})) || deferredText(f) != before) { return fail(__LINE__); }
    }
    stage = "numerical final increment refusal";
    {
        auto wrapping = parseSourceString<ModuleOp>(source, context);
        if (!wrapping) { return fail(__LINE__); }
        auto f = wrapping->lookupSymbol<func::FuncOp>("deferred");
        auto loop = *f.getOps<scf::ForOp>().begin();
        OpBuilder builder(loop);
        auto zero = builder.create<arith::ConstantIndexOp>(loop.getLoc(), 0);
        loop.getLowerBoundMutable().assign(zero);
        // Keep member points on this loop's even lattice. Its unbounded upper
        // still permits a final increment past INT64_MAX; alignment must not
        // prevent the intended serialization no-wrap qualification from running.
        auto plan = makePlan(f, 0);
        if (failed(plan)) { return fail(__LINE__); }
        (*plan)->deferredOutput = true;
        const auto before = deferredText(f);
        std::string refusal;
        ScopedDiagnosticHandler diagnostics(context, [&](Diagnostic& diagnostic) {
            llvm::raw_string_ostream stream(refusal); diagnostic.print(stream);
            return success();
        });
        if (succeeded(fs::insertLogicalSynchronization(f, **plan)) || deferredText(f) != before ||
            refusal.find("nonwrapping final increment") == std::string::npos) {
            return fail(__LINE__);
        }
    }
    // Contextual leaves survive owner destruction and parsing, together with
    // composition offsets, two invocation coordinates and an exact activity mask.
    // A supplied tuple placement isolates codec/liveness from reuse certification.
    stage = "contextual route preparation";
    std::string contextualSaved, contextualCombined;
    unsigned contextualPhysicalOps = 0, contextualLogicalOps = 0;
    for (unsigned mode = 0; mode < 3; ++mode) {
        auto module = parseSourceString<ModuleOp>(source, context);
        if (!module) { return fail(__LINE__); }
        auto f = module->lookupSymbol<func::FuncOp>("deferred");
        auto loop = *f.getOps<scf::ForOp>().begin();
        auto* setCut = &loop.getBody()->front(); auto* waitCut = loop.getBody()->getTerminator();
        auto arena = std::make_shared<fs::RegionExpressions>();
        // Both a cut and a forbidden-recomputation source bind the implicit
        // zero-result yield; its operation attributes disappear under custom printing.
        arena->forbidRecomputation(waitCut);
        auto visit = arena->input(loop.getInductionVar()), limit = arena->input(f.getArgument(0));
        auto domain = fs::IntegerSystem::create(2,
            {{{fs::BoundInteger(1), fs::BoundInteger(-1)}, fs::BoundInteger(-1)}});
        if (failed(domain)) { return fail(__LINE__); }
        auto activity = arena->integerPredicate(*domain, {visit, limit}, 1, {0, 0});
        auto live = arena->integerWitness({{fs::BoundInteger(2)}, fs::BoundInteger(1)}, fs::BoundInteger(1),
            {visit}, 1, {0}, 0);
        auto dead = visit;
        for (unsigned i = 0; i < 32; ++i) { dead = arena->add(arena->add(arena->add(dead, dead), dead), limit); }
        auto plan = std::make_unique<fs::PreparedLogicalPlan>(0);
        plan->nestedIdentities = plan->groupedFamilies = plan->independentPieces = true;
        for (uint32_t record = 0; record < 2; ++record) {
            SmallVector<fs::ContextualEndpointFamily> endpoints;
            for (auto kind : {fs::EndpointKind::Set, fs::EndpointKind::Wait}) {
                fs::ContextualEndpointFamily endpoint;
                endpoint.piece.kind = kind; endpoint.piece.sourcePipe = 1; endpoint.piece.targetPipe = 2;
                endpoint.piece.order = record;
                endpoint.piece.cut = {loop.getBody(), kind == fs::EndpointKind::Set ? setCut : waitCut};
                endpoint.piece.members.push_back({0, {}});
                endpoint.activity = activity; endpoint.ordinal = arena->constant(0);
                endpoint.matchingCoordinates = {dead, live};
                endpoints.push_back(endpoint);
            }
            auto leaf = fs::buildContextualFamilyEndpointRecipe(arena, endpoints, {{activity, setCut}});
            if (failed(leaf)) { return fail(__LINE__); }
            auto joined = fs::appendFamilyEndpointRecipe(plan->familyRecipe, *leaf, record, 0);
            if (failed(joined)) { return fail(__LINE__); }
            plan->familyRecipe = *joined;
            fs::EndpointFamily family;
            family.id = record; family.sourcePipe = 1; family.targetPipe = 2;
            family.sourceOrder = family.targetOrder = record;
            family.sourceCut = {loop.getBody(), setCut}; family.targetCut = {loop.getBody(), waitCut};
            family.members.push_back({record, 0, 1, {}, {}});
            plan->families.push_back(family);
        }
        auto wrapped = fs::prefixFamilyInvocation(plan->familyRecipe, arena, visit);
        if (failed(wrapped)) { return fail(__LINE__); }
        wrapped = fs::maskFamilyActivity(*wrapped, arena, arena->lnot(arena->eq(visit, arena->constant(5))));
        if (failed(wrapped)) { return fail(__LINE__); }
        wrapped = fs::prefixFamilyInvocation(*wrapped, arena, arena->constant(7));
        if (failed(wrapped)) { return fail(__LINE__); }
        plan->familyRecipe = *wrapped;
        Builder b(context);
        SmallVector<Attribute> rules;
        for (int64_t base : {0, 3}) {
            rules.push_back(b.getDictionaryAttr({b.getNamedAttr("coordinate_count", b.getI64IntegerAttr(5)),
                b.getNamedAttr("base", b.getI64IntegerAttr(base)),
                b.getNamedAttr("terms", b.getArrayAttr({b.getDenseI64ArrayAttr({4, 1, 0, 3, 1})}))}));
        }
        auto group = b.getDictionaryAttr({b.getNamedAttr("source", b.getI64IntegerAttr(1)),
            b.getNamedAttr("target", b.getI64IntegerAttr(2)), b.getNamedAttr("budget", b.getI64IntegerAttr(6)),
            b.getNamedAttr("records", b.getDenseI64ArrayAttr({0, 1})),
            b.getNamedAttr("strides", b.getDenseI64ArrayAttr({0, 0})),
            b.getNamedAttr("phases", b.getDenseI64ArrayAttr({0, 3})),
            b.getNamedAttr("conflicts", b.getDenseI64ArrayAttr({})),
            b.getNamedAttr("tuple_rules", b.getArrayAttr(rules))});
        plan->allocationCertificate = b.getDictionaryAttr({b.getNamedAttr("version", b.getI64IntegerAttr(2)),
            b.getNamedAttr("plan", b.getI64IntegerAttr(0)),
            b.getNamedAttr("strategy", b.getStringAttr("regional-palettes")),
            b.getNamedAttr("groups", b.getArrayAttr({group}))});
        auto originalOps = countArithmetic(f);
        if (mode == 0) { plan->deferredOutput = true; }
        else if (mode == 1 && failed(fs::planPhysicalFamilyOutput(f, *plan, {0, 1, 2, 3, 4, 5}))) {
            return fail(__LINE__);
        }
        if (failed(fs::insertLogicalSynchronization(f, *plan)) || failed(verify(f))) { return fail(__LINE__); }
        if (mode == 0) {
            if (countArithmetic(f) != originalOps || plan->familyWork.materializations) { return fail(__LINE__); }
            auto data = f->getAttrOfType<DictionaryAttr>("pto.deferred_family");
            if (!data || data.getAs<IntegerAttr>("version").getInt() != 2 ||
                data.getAs<ArrayAttr>("arenas").size() != 1) {
                return fail(__LINE__);
            }
            contextualSaved = deferredText(module->getOperation());
        } else if (mode == 1) {
            contextualCombined = deferredText(f); contextualPhysicalOps = countArithmetic(f);
        } else { contextualLogicalOps = countArithmetic(f); }
    }
    stage = "contextual saved allocation and liveness equivalence";
    auto contextual = parseSourceString<ModuleOp>(contextualSaved, context);
    if (!contextual) { return fail(__LINE__); }
    auto contextualFunction = contextual->lookupSymbol<func::FuncOp>("deferred");
    if (contextualLogicalOps < contextualPhysicalOps + 32 ||
        failed(fs::allocatePhysicalEventIds(contextualFunction, {0, 1, 2, 3, 4, 5})) ||
        failed(verify(contextualFunction)) || deferredText(contextualFunction) != contextualCombined) {
            return fail(__LINE__);
        }
    stage = "contextual independent command trace";
    for (uint64_t bound : {0U, 1U, 2U, 5U, 8U, 11U}) {
        SmallVector<std::pair<bool, uint64_t>> actual, expected;
        for (uint64_t visit = 1; visit < bound; visit += 2) {
            if (visit == 5) { continue; }
            auto id = (2 * visit + 1) % 3;
            expected.append({{true, id}, {true, id + 3}, {false, id}, {false, id + 3}});
        }
        if (!trace(contextualFunction, bound, actual) || actual != expected) { return fail(__LINE__); }
    }
    stage = "contextual malformed refusal";
    for (unsigned mode = 0; mode < 13; ++mode) {
        auto broken = parseSourceString<ModuleOp>(contextualSaved, context);
        if (!broken) { return fail(__LINE__); }
        auto f = broken->lookupSymbol<func::FuncOp>("deferred");
        Builder b(context);
        auto original = f->getAttrOfType<DictionaryAttr>("pto.deferred_family");
        NamedAttrList data(original);
        auto bindings = original.getAs<ArrayAttr>("sources");
        if (!bindings || bindings.empty()) { return fail(__LINE__); }
        SmallVector<Attribute> sources(bindings.begin(), bindings.end());
        if (mode == 0) {
            sources.pop_back(); data.set("sources", b.getArrayAttr(sources));
        } else if (mode == 1 || mode == 4) {
            NamedAttrList binding(cast<DictionaryAttr>(sources[0]));
            if (mode == 1) { binding.set("path", b.getDenseI64ArrayAttr({INT64_MAX, 0, 0})); }
            else { binding.set("name", b.getStringAttr("arith.invalid")); }
            sources[0] = binding.getDictionary(context);
            data.set("sources", b.getArrayAttr(sources));
        } else if (mode == 2 || mode == 3) {
            data.set(mode == 2 ? "arenas" : "nodes", b.getArrayAttr({}));
        } else if (mode == 5) {
            if (sources.size() < 2) { return fail(__LINE__); }
            sources[1] = sources[0]; data.set("sources", b.getArrayAttr(sources));
        } else if (mode == 6) {
            // Moving the marker away from its implicit original terminator must
            // not retarget the archived endpoint to another cut in the same loop.
            auto loop = *f.getOps<scf::ForOp>().begin();
            auto* marker = loop.getBody()->getTerminator()->getPrevNode();
            if (!isa<pto::LogicalFamilyOp>(marker)) { return fail(__LINE__); }
            marker->moveBefore(&loop.getBody()->front());
        } else if (mode == 7) {
            // Obsolete per-operation bindings cannot become an alternate decoder route.
            f->setAttr("pto.deferred_family_source", b.getDictionaryAttr({}));
        } else {
            auto rawCuts = original.getAs<ArrayAttr>("cuts");
            SmallVector<Attribute> cuts(rawCuts.begin(), rawCuts.end());
            auto found = llvm::find_if(cuts, [](Attribute raw) {
                return cast<DictionaryAttr>(raw).getAs<StringAttr>("name").getValue() == "scf.yield";
            });
            if (found == cuts.end()) { return fail(__LINE__); }
            NamedAttrList cut(cast<DictionaryAttr>(*found));
            if (mode == 8) { cut.set("path", b.getDenseI64ArrayAttr({0, INT64_MAX, 0})); }
            else if (mode == 9) { cut.set("path", b.getDenseI64ArrayAttr({0, 0})); }
            else if (mode == 10) { cut.set("axis_ids", b.getDenseI64ArrayAttr({INT64_MAX})); }
            else if (mode == 11) { cuts.push_back(*found); }
            else { cut.set("name", b.getStringAttr("func.return")); }
            if (mode != 11) { *found = cut.getDictionary(context); }
            data.set("cuts", b.getArrayAttr(cuts));
        }
        if (mode < 4) { f->setAttr("pto.deferred_family", data.getDictionary(context)); }
        else { resealDeferredContext(f, data.getDictionary(context)); }
        auto before = deferredText(f);
        ScopedDiagnosticHandler diagnostics(context, [](Diagnostic&) { return success(); });
        if (succeeded(fs::allocatePhysicalEventIds(f, {0, 1, 2, 3, 4, 5})) || deferredText(f) != before) {
            return fail(__LINE__);
        }
    }
    stage = "saved module symbol ownership";
    // A saved original payload call must retain the module's symbol context
    // while the allocator stages and verifies only the target function.
    {
        auto module = parseSourceString<ModuleOp>(source, context);
        if (!module) { return fail(__LINE__); }
        auto f = module->lookupSymbol<func::FuncOp>("deferred");
        OpBuilder builder(context);
        auto callee = func::FuncOp::create(f.getLoc(), "payload", builder.getFunctionType({}, {}));
        callee.setPrivate(); module->push_back(callee);
        builder.setInsertionPoint(f.front().getTerminator());
        builder.create<func::CallOp>(f.getLoc(), "payload", TypeRange{}, ValueRange{});
        auto plan = makePlan(f);
        if (failed(plan)) { return fail(__LINE__); }
        (*plan)->deferredOutput = true;
        if (failed(fs::insertLogicalSynchronization(f, **plan))) { return fail(__LINE__); }
        auto parsed = parseSourceString<ModuleOp>(deferredText(module->getOperation()), context);
        if (!parsed) { return fail(__LINE__); }
        auto target = parsed->lookupSymbol<func::FuncOp>("deferred");
        if (failed(fs::allocatePhysicalEventIds(target, {3})) || failed(verify(*parsed)) ||
            !parsed->lookupSymbol<func::FuncOp>("payload")) { return fail(__LINE__); }
    }
    stage = "unsupported concrete plan atomicity";
    // Unsupported concrete plans must fail before changing source IR.
    auto unsupported = parseSourceString<ModuleOp>(source, context);
    if (!unsupported) { return fail(__LINE__); }
    auto f = unsupported->lookupSymbol<func::FuncOp>("deferred");
    auto plan = makePlan(f);
    if (failed(plan) || failed(fs::materializeLogicalFamilies(**plan))) { return fail(__LINE__); }
    (*plan)->deferredOutput = true;
    const auto before = deferredText(f);
    ScopedDiagnosticHandler diagnostics(context, [](Diagnostic&) { return success(); });
    if (succeeded(fs::insertLogicalSynchronization(f, **plan)) || deferredText(f) != before) { return fail(__LINE__); }
    llvm::outs() << "deferred numerical families: roundtrip live-control exact traces corruption refusal checked\n";
    return true;
}
} // namespace
#endif
