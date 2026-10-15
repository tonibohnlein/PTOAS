// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_DEFERRED_LOCAL_CHECKS_H
#define PTO_DEFERRED_LOCAL_CHECKS_H
namespace {
bool checkDeferredLocalFamilies(mlir::MLIRContext* context)
{
    using namespace mlir;
    namespace fs = mlir::pto::frontiersynch;
    constexpr StringLiteral source = R"mlir(
      module {
        func.func @local(%n: index) {
          %zero = arith.constant 0 : index
          %one = arith.constant 1 : index
          scf.for %i = %zero to %n step %one {
            scf.for %j = %zero to %one step %one {
              %a = arith.addi %i, %j : index
            }
            %b = arith.addi %i, %one : index
          }
          return
        }
      })mlir";
    for (bool numerical : {false, true}) {
        std::string saved, combined;
        for (bool deferred : {true, false}) {
            auto module = parseSourceString<ModuleOp>(source, context);
            if (!module) { return false; }
            auto function = module->lookupSymbol<func::FuncOp>("local");
            auto outer = *function.getOps<scf::ForOp>().begin();
            auto inner = *outer.getBody()->getOps<scf::ForOp>().begin();
            auto* before = inner.getBody()->getTerminator();
            auto* after = outer.getBody()->getTerminator();
            fs::PreparedLogicalPlan plan(0);
            plan.groupedFamilies = plan.independentPieces = true;
            fs::EndpointFamily family;
            family.id = 0; family.local = true;
            family.sourcePipe = family.targetPipe = static_cast<uint32_t>(pto::PIPE::PIPE_M);
            family.sourceCut = {before->getBlock(), before};
            family.targetCut = {after->getBlock(), after};
            family.members.push_back({0, 0, 1, {{inner, 0}}, {{outer, 0}}});
            plan.families.push_back(family);
            fs::EndpointPiece piece;
            piece.kind = fs::EndpointKind::Barrier;
            piece.sourcePipe = family.sourcePipe; piece.targetPipe = family.targetPipe;
            piece.cut = family.targetCut; piece.members.push_back({0, {{outer, 0}}});
            if (numerical) {
                fs::NumericalFamilyRecipe recipe;
                recipe.outer = outer; recipe.pieces.push_back(piece);
                auto owned = fs::restoreNumericalFamilyRecipe(recipe);
                if (failed(owned)) { return false; }
                plan.familyRecipe = *owned;
            } else {
                auto arena = std::make_shared<fs::RegionExpressions>();
                fs::ContextualEndpointFamily endpoint;
                endpoint.piece = piece;
                endpoint.activity = arena->eq(arena->input(outer.getInductionVar()), arena->constant(0));
                auto owned = fs::buildContextualFamilyEndpointRecipe(arena, {endpoint}, {});
                if (failed(owned)) { return false; }
                plan.familyRecipe = *owned;
            }
            Builder builder(context);
            // Local barriers consume no event IDs. This fixture supplies an empty
            // allocation, and makes no claim about a new reuse certificate.
            plan.allocationCertificate = builder.getDictionaryAttr({
                builder.getNamedAttr("version", builder.getI64IntegerAttr(2)),
                builder.getNamedAttr("plan", builder.getI64IntegerAttr(0)),
                builder.getNamedAttr("strategy", builder.getStringAttr("regional-palettes")),
                builder.getNamedAttr("groups", builder.getArrayAttr({}))});
            plan.deferredOutput = deferred;
            if (!deferred && failed(fs::planPhysicalFamilyOutput(function, plan, {3}))) { return false; }
            if (failed(fs::insertLogicalSynchronization(function, plan)) || failed(verify(function))) { return false; }
            if (deferred) { saved = deferredText(module->getOperation()); }
            else { combined = deferredText(function); }
        }
        auto parsed = parseSourceString<ModuleOp>(saved, context);
        if (!parsed) { return false; }
        auto function = parsed->lookupSymbol<func::FuncOp>("local");
        auto metadata = function->getAttrOfType<DictionaryAttr>("pto.deferred_family");
        if (!metadata || metadata.getAs<IntegerAttr>("version").getInt() != 2 ||
            metadata.getAs<ArrayAttr>("local_sources").size() != 1) { return false; }
        auto outer = *function.getOps<scf::ForOp>().begin();
        auto inner = *outer.getBody()->getOps<scf::ForOp>().begin();
        {
            auto decoded = fs::readDeferredFamilyOutput(function);
            if (failed(decoded) || (*decoded)->families.size() != 1 ||
                (*decoded)->families[0].sourceCut.before != inner.getBody()->getTerminator() ||
                (*decoded)->families[0].targetCut.before != outer.getBody()->getTerminator() ||
                (*decoded)->families[0].members[0].sourceCoordinates[0].loop != inner) { return false; }
        }
        if (failed(fs::allocatePhysicalEventIds(function, {3})) || failed(verify(function)) ||
            deferredText(function) != combined) { return false; }
        outer = *function.getOps<scf::ForOp>().begin();
        for (unsigned count : {0U, 1U, 3U}) {
            DenseMap<Value, APInt> values;
            values[function.getArgument(0)] = APInt(64, count);
            SmallVector<uint64_t> barriers;
            std::function<bool(Block&)> execute = [&](Block& block) {
                for (auto& op : block) {
                    if (auto loop = dyn_cast<scf::ForOp>(op)) {
                        const auto lower = values.lookup(loop.getLowerBound()).getZExtValue();
                        const auto upper = values.lookup(loop.getUpperBound()).getZExtValue();
                        const auto step = values.lookup(loop.getStep()).getZExtValue();
                        if (!step || !loop.getInitArgs().empty()) { return false; }
                        for (auto i = lower; i < upper; i += step) {
                            values[loop.getInductionVar()] = APInt(64, i);
                            if (!execute(*loop.getBody())) { return false; }
                        }
                    } else if (auto branch = dyn_cast<scf::IfOp>(op)) {
                        auto& region = values.lookup(branch.getCondition()).isZero() ?
                            branch.getElseRegion() : branch.getThenRegion();
                        if (!region.empty() && !execute(region.front())) { return false; }
                    } else if (auto barrier = dyn_cast<pto::BarrierOp>(op)) {
                        if (barrier->getAttrOfType<pto::PipeAttr>("pipe").getPipe() != pto::PIPE::PIPE_M) {
                            return false;
                        }
                        barriers.push_back(values.lookup(outer.getInductionVar()).getZExtValue());
                    } else if (!isa<scf::YieldOp, func::ReturnOp>(op) && !evaluateDeferredScalar(op, values)) {
                        return false; // Includes every unexpected SET, WAIT or terminal drain.
                    }
                }
                return true;
            };
            const SmallVector<uint64_t> expected = count ? SmallVector<uint64_t>{0} : SmallVector<uint64_t>{};
            if (!execute(function.front()) || barriers != expected) { return false; }
        }
        for (unsigned corruption = 0; corruption < 8; ++corruption) {
            auto broken = parseSourceString<ModuleOp>(saved, context);
            if (!broken) { return false; }
            auto target = broken->lookupSymbol<func::FuncOp>("local");
            auto original = target->getAttrOfType<DictionaryAttr>("pto.deferred_family");
            auto binding = cast<DenseI64ArrayAttr>(original.getAs<ArrayAttr>("local_sources")[0]);
            Builder builder(context);
            SmallVector<Attribute> bindings;
            if (corruption == 1) {
                bindings.push_back(binding); bindings.push_back(builder.getDenseI64ArrayAttr({99, binding[1]}));
            }
            else if (corruption == 2) { bindings.push_back(builder.getDenseI64ArrayAttr({binding[0], INT64_MAX})); }
            else if (corruption == 3) { bindings.append({binding, binding}); }
            else if (corruption == 4) { bindings.push_back(builder.getDenseI64ArrayAttr({binding[0]})); }
            NamedAttrList changed(original);
            if (corruption == 7) {
                // A valid original operation in the wrong frame must still fail
                // source-coordinate scope validation after decoding.
                auto sources = original.getAs<ArrayAttr>("sources");
                SmallVector<Attribute> relocated(sources.begin(), sources.end());
                auto cut = cast<DictionaryAttr>(original.getAs<ArrayAttr>("cuts")[0]);
                relocated.push_back(builder.getDictionaryAttr({
                    builder.getNamedAttr("path", cut.get("path")),
                    builder.getNamedAttr("name", cut.get("name"))}));
                bindings.push_back(builder.getDenseI64ArrayAttr({binding[0], int64_t(sources.size())}));
                changed.set("sources", builder.getArrayAttr(relocated));
                changed.set("source_count", builder.getI64IntegerAttr(relocated.size()));
            }
            if (corruption == 5) { changed.erase("local_sources"); }
            else {
                if (corruption == 6) { bindings.push_back(builder.getStringAttr("invalid")); }
                changed.set("local_sources", builder.getArrayAttr(bindings));
            }
            resealDeferredContext(target, changed.getDictionary(context));
            const auto before = deferredText(target);
            ScopedDiagnosticHandler diagnostics(context, [](Diagnostic&) { return success(); });
            if (succeeded(fs::allocatePhysicalEventIds(target, {3})) || deferredText(target) != before) {
                return false;
            }
        }
    }
    return true;
}
} // namespace
#endif
