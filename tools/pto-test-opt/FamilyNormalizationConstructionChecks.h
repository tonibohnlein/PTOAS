// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent operation-equivalence and actual-construction parity controls.
#ifndef PTO_FAMILY_NORMALIZATION_CONSTRUCTION_CHECKS_H
#define PTO_FAMILY_NORMALIZATION_CONSTRUCTION_CHECKS_H
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/ScalarConstruction.h"
#include "mlir/IR/OperationSupport.h"
namespace {
bool checkScalarConstructionTrace(mlir::MLIRContext* context)
{
    using namespace mlir;
    namespace detail = mlir::pto::frontiersynch::detail;
    OpBuilder builder(context);
    auto loc = builder.getUnknownLoc();
    OwningOpRef<func::FuncOp> function(func::FuncOp::create(loc, "scalar_trace",
        builder.getFunctionType({builder.getI64Type(), builder.getI64Type()}, {})));
    auto* body = function->addEntryBlock();
    builder.setInsertionPointToEnd(body);
    auto plain = builder.create<arith::AddIOp>(loc, body->getArgument(0), body->getArgument(1));
    auto flagged = builder.create<arith::AddIOp>(loc, body->getArgument(0), body->getArgument(1));
    flagged.setOverflowFlags(arith::IntegerOverflowFlags::nsw);
    auto attributed = builder.create<arith::AddIOp>(loc, body->getArgument(0), body->getArgument(1));
    attributed->setAttr("test.tag", builder.getUnitAttr());
    auto multi = builder.create<arith::AddUIExtendedOp>(loc, body->getArgument(0), body->getArgument(1));
    auto* cut = builder.create<func::ReturnOp>(loc).getOperation();
    Block code;
    builder.setInsertionPointToEnd(&code);
    detail::ScalarConstruction actual(builder, loc), counted;
    actual.enableTrace(context); counted.enableTrace(context);
    auto schedule = [&](detail::ScalarConstruction& sink) {
        sink.setInsertionCut(cut);
        auto a = sink.input(body->getArgument(0)), b = sink.input(body->getArgument(1));
        auto sum = sink.binary<arith::AddIOp>(a, b);
        (void)sink.binary<arith::AddIOp>(b, a);
        (void)sink.replay(*plain, {a, b});
        (void)sink.replay(*flagged, {a, b});
        (void)sink.replay(*attributed, {a, b});
        auto wide = sink.replay(*multi, {a, b});
        auto reversed = sink.replay(*multi, {b, a});
        (void)sink.binary<arith::AddIOp>(wide[0], reversed[0]);
        (void)sink.binary<arith::OrIOp>(wide[1], reversed[1]);
        (void)sink.integer(APInt(32, 7));
        (void)sink.integer(APInt(64, 7));
        sink.setInsertionCut(plain.getOperation());
        (void)sink.binary<arith::AddIOp>(a, b);
        return sum;
    };
    (void)schedule(actual); (void)schedule(counted);
    if (!actual.validTrace() || !counted.validTrace() || actual.operations() != counted.operations() ||
        actual.trace().size() != code.getOperations().size()) { return false; }
    auto realAliases = detail::deduplicateScalarInstructions(actual.trace());
    auto tokenAliases = detail::deduplicateScalarInstructions(counted.trace());
    if (realAliases.redundant != tokenAliases.redundant || realAliases.redundant.size() != 3) { return false; }
    SmallVector<Operation*> operations;
    for (auto& operation : code) { operations.push_back(&operation); }
    for (std::size_t i = 0; i < operations.size(); ++i) {
        for (std::size_t j = 0; j < operations.size(); ++j) {
            // LLVM's native operation comparator supplies the independent
            // attribute/property/type and commutative operand oracle.
            const bool equivalent = actual.trace()[i].cut == actual.trace()[j].cut &&
                OperationEquivalence::isEquivalentTo(operations[i], operations[j],
                    OperationEquivalence::IgnoreLocations);
            auto a = actual.trace()[i], b = actual.trace()[j];
            if (a.commutative) { llvm::sort(a.operands); }
            if (b.commutative) { llvm::sort(b.operands); }
            if (equivalent != (a.cut == b.cut && detail::sameScalarInstruction(a, b))) { return false; }
        }
    }
    detail::ScalarConstruction unsupported;
    unsupported.enableTrace(context); unsupported.setInsertionCut(cut);
    (void)unsupported.replay(*cut, {});
    if (unsupported.validTrace()) { return false; }
    // Multi-result correspondence preserves result number instead of aliasing
    // the carry with the sum. The identical instruction at another cut stays.
    const auto& first = counted.trace()[5]; const auto& second = counted.trace()[6];
    return tokenAliases.canonical(second.results[0]) == first.results[0] &&
        tokenAliases.canonical(second.results[1]) == first.results[1] &&
        first.results[0] != first.results[1] &&
        tokenAliases.canonical(counted.trace().back().results[0]) == counted.trace().back().results[0];
}
bool checkNormalizedConstructionParity(mlir::MLIRContext* context)
{
    using namespace mlir;
    namespace fs = mlir::pto::frontiersynch;
    // Same protocol, two already supplied certified maps. Equal phases permit
    // residue compaction; different phases preserve both commands. No candidate
    // proof is constructed by the comparison or this supplied-mapping fixture.
    for (unsigned mode = 0; mode < 4; ++mode) {
        const bool shared = mode % 2, finite = mode >= 2;
        auto fail = [&](llvm::StringRef stage) {
            llvm::errs() << "normalization construction mode=" << mode << " failed: " << stage << "\n";
            return false;
        };
        OpBuilder builder(context);
        auto loc = builder.getUnknownLoc();
        OwningOpRef<func::FuncOp> function(func::FuncOp::create(loc, "normalization_count",
            builder.getFunctionType({builder.getIndexType(), builder.getI1Type()}, {})));
        auto* body = function->addEntryBlock();
        builder.setInsertionPointToEnd(body);
        scf::ForOp loop;
        if (finite) {
            auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
            auto one = builder.create<arith::ConstantIndexOp>(loc, 1);
            auto two = builder.create<arith::ConstantIndexOp>(loc, 2);
            loop = builder.create<scf::ForOp>(loc, zero, two, one);
            builder.create<func::ReturnOp>(loc);
            builder.setInsertionPointToStart(loop.getBody());
        }
        auto* source = builder.create<arith::ConstantIndexOp>(loc, 11).getOperation();
        auto* target = finite ? loop.getBody()->getTerminator() :
            builder.create<func::ReturnOp>(loc).getOperation();
        auto* endpointBlock = source->getBlock();
        auto arena = std::make_shared<fs::RegionExpressions>();
        const auto guard = arena->input(body->getArgument(1));
        const auto ordinal = arena->input(body->getArgument(0));
        const auto tag = arena->select(guard, arena->constant(0), arena->constant(1));
        fs::PreparedLogicalPlan plan(0);
        plan.nestedIdentities = finite;
        SmallVector<fs::ContextualEndpointFamily> endpoints;
        for (uint32_t record : {0U, 1U, 2U, 3U}) {
            const bool local = record >= 2;
            fs::EndpointFamily family;
            family.id = record; family.sourcePipe = 1; family.targetPipe = local ? 1 : 2;
            family.local = local;
            family.sourceCut = {endpointBlock, source}; family.targetCut = {endpointBlock, target};
            SmallVector<fs::TemplateCoordinate> coordinates;
            if (finite && !local) { coordinates.push_back({loop, static_cast<int64_t>(record)}); }
            family.members.push_back({record, 0, 1, coordinates, coordinates}); plan.families.push_back(family);
            for (bool set : {true, false}) {
                if (local && set) { continue; }
                fs::ContextualEndpointFamily endpoint;
                endpoint.piece.cut = {endpointBlock, set ? source : target};
                endpoint.piece.kind = local ? fs::EndpointKind::Barrier :
                    set ? fs::EndpointKind::Set : fs::EndpointKind::Wait;
                // A shared pipe/displacement namespace uses its smallest
                // original record; both local records retain separate pieces.
                endpoint.piece.namespaceId = local ? 2 : 0;
                endpoint.piece.sourcePipe = 1; endpoint.piece.targetPipe = family.targetPipe;
                endpoint.piece.order = endpoints.size(); endpoint.piece.members.push_back({record, coordinates});
                endpoint.activity = local ? arena->boolean(true) : finite ?
                    arena->land(arena->eq(arena->input(loop.getInductionVar()), arena->constant(record)), guard) :
                    record ? arena->lnot(guard) : guard;
                endpoint.ordinal = ordinal;
                if (!local && !finite) {
                    endpoint.residueCoordinate = tag; endpoint.modulus = 2; endpoint.residue = record;
                }
                endpoints.push_back(std::move(endpoint));
            }
        }
        auto recipe = fs::buildContextualFamilyEndpointRecipe(arena, endpoints);
        if (failed(recipe)) { return fail("recipe preparation"); }
        plan.familyRecipe = *recipe;
        auto group = builder.getDictionaryAttr({
            builder.getNamedAttr("source", builder.getI64IntegerAttr(1)),
            builder.getNamedAttr("target", builder.getI64IntegerAttr(2)),
            builder.getNamedAttr("budget", builder.getI64IntegerAttr(2)),
            builder.getNamedAttr("records", builder.getDenseI64ArrayAttr({0, 1})),
            builder.getNamedAttr("strides", builder.getDenseI64ArrayAttr({1, 1})),
            builder.getNamedAttr("phases", builder.getDenseI64ArrayAttr({0, shared ? 0 : 1})),
            builder.getNamedAttr("conflicts", builder.getDenseI64ArrayAttr({}))});
        if (finite) {
            SmallVector<Attribute> rules;
            for (unsigned record : {0U, 1U}) {
                rules.push_back(builder.getDictionaryAttr({
                    builder.getNamedAttr("coordinate_count", builder.getI64IntegerAttr(1)),
                    builder.getNamedAttr("base", builder.getI64IntegerAttr(0)),
                    builder.getNamedAttr("terms", builder.getArrayAttr({builder.getDenseI64ArrayAttr(
                        {0, 1, shared ? 0 : record, 2, 1})}))}));
            }
            NamedAttrList mapped(group);
            mapped.set("tuple_rules", builder.getArrayAttr(rules));
            group = mapped.getDictionary(context);
        }
        plan.allocationCertificate = builder.getDictionaryAttr({
            builder.getNamedAttr("version", builder.getI64IntegerAttr(2)),
            builder.getNamedAttr("plan", builder.getI64IntegerAttr(0)),
            builder.getNamedAttr("strategy", builder.getStringAttr("regional-palettes")),
            builder.getNamedAttr("groups", builder.getArrayAttr({group}))});
        if (failed(fs::planPhysicalFamilyOutput(*function, plan, {0, 1}))) { return fail("physical plan"); }
        const auto predicted = fs::physicalFamilyConstructionOperations(plan);
        if (!predicted || !plan.preparation.empty() || !plan.endpoints.empty()) { return fail("count without SSA"); }
        if (failed(fs::insertLogicalSynchronization(*function, plan)) || failed(verify(*function))) {
            return fail("materialization and verification");
        }
        const auto& work = plan.familyWork;
        if (*predicted != work.preparationScalars + work.compactionScalars + work.mappingScalars +
            work.barrierScalars || work.barrierScalars != 1 || bool(work.compactionScalars) != shared) {
            llvm::errs() << "predicted=" << *predicted << " preparation=" << work.preparationScalars
                << " compaction=" << work.compactionScalars << " mapping=" << work.mappingScalars
                << " barriers=" << work.barrierScalars << "\n";
            return fail("construction parity");
        }
        unsigned sets = 0, waits = 0;
        function->walk([&](Operation* operation) {
            sets += isa<pto::SetFlagOp, pto::SetFlagDynOp>(operation);
            waits += isa<pto::WaitFlagOp, pto::WaitFlagDynOp>(operation);
        });
        if (sets != (shared ? 1U : 2U) || waits != sets) {
            llvm::errs() << "SET sites=" << sets << " WAIT sites=" << waits << "\n";
            return fail("static commands");
        }
        for (uint64_t visit : {0U, 1U, 3U, 8U}) {
            for (bool chooseFirst : {false, true}) {
                DenseMap<Value, APInt> values;
                values[body->getArgument(0)] = APInt(64, visit);
                values[body->getArgument(1)] = APInt(1, chooseFirst);
                SmallVector<std::pair<unsigned, uint64_t>> commands;
                std::function<bool(Block&)> execute = [&](Block& block) {
                    for (auto& operation : block) {
                        if (operation.getName().getDialectNamespace() == "arith") {
                            if (!evaluateDeferredScalar(operation, values)) { return false; }
                        } else if (auto countedLoop = dyn_cast<scf::ForOp>(operation)) {
                            if (!values.count(countedLoop.getLowerBound()) ||
                                !values.count(countedLoop.getUpperBound()) ||
                                !values.count(countedLoop.getStep())) { return false; }
                            const auto low = values.lookup(countedLoop.getLowerBound()).getZExtValue();
                            const auto high = values.lookup(countedLoop.getUpperBound()).getZExtValue();
                            const auto step = values.lookup(countedLoop.getStep()).getZExtValue();
                            if (!step) { return false; }
                            for (auto iv = low; iv < high; iv += step) {
                                values[countedLoop.getInductionVar()] = APInt(64, iv);
                                if (!execute(*countedLoop.getBody())) { return false; }
                            }
                        } else if (auto branch = dyn_cast<scf::IfOp>(operation)) {
                            if (!values.count(branch.getCondition())) { return false; }
                            auto& region = values.lookup(branch.getCondition()).isZero() ?
                                branch.getElseRegion() : branch.getThenRegion();
                            if (region.empty()) {
                                if (branch.getNumResults()) { return false; }
                                continue;
                            }
                            if (!execute(region.front())) { return false; }
                            auto yielded = dyn_cast<scf::YieldOp>(region.front().getTerminator());
                            if (!yielded || yielded.getNumOperands() != branch.getNumResults()) { return false; }
                            for (auto pair : llvm::zip(branch.getResults(), yielded.getOperands())) {
                                auto found = values.find(std::get<1>(pair));
                                if (found == values.end()) { return false; }
                                const APInt result = found->second;
                                values[std::get<0>(pair)] = result;
                            }
                        } else if (isa<pto::SetFlagOp, pto::WaitFlagOp,
                                       pto::SetFlagDynOp, pto::WaitFlagDynOp>(operation)) {
                            const auto fixed = operation.getAttrOfType<pto::EventAttr>("event_id");
                            if (!fixed && !values.count(operation.getOperand(0))) { return false; }
                            const auto id = fixed ? static_cast<uint64_t>(fixed.getEvent()) :
                                values.lookup(operation.getOperand(0)).getZExtValue();
                            commands.push_back({isa<pto::SetFlagOp, pto::SetFlagDynOp>(operation) ? 0U : 2U, id});
                        } else if (isa<pto::BarrierOp>(operation)) { commands.push_back({1U, 0U}); }
                        else if (!isa<func::ReturnOp, scf::YieldOp>(operation)) { return false; }
                    }
                    return true;
                };
                SmallVector<std::pair<unsigned, uint64_t>> expected;
                for (unsigned iv = 0; iv < (finite ? 2U : 1U); ++iv) {
                    const auto phase = shared ? 0 : finite ? iv : chooseFirst ? 0 : 1;
                    const auto id = (visit + phase) % 2;
                    if (!finite || chooseFirst) { expected.push_back({0, id}); }
                    expected.push_back({1, 0});
                    if (!finite || chooseFirst) { expected.push_back({2, id}); }
                }
                if (!execute(*body) || commands != expected) {
                    llvm::errs() << "visit=" << visit << " chooseFirst=" << chooseFirst
                        << " commands=" << commands.size() << " expected=" << expected.size() << "\n";
                    return fail("independent command trace");
                }
            }
        }
    }
    return true;
}
} // namespace
#endif
