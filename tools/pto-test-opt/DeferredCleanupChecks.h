// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_DEFERRED_CLEANUP_CHECKS_H
#define PTO_DEFERRED_CLEANUP_CHECKS_H
#include "PTO/Transforms/FrontierSynch/Finite/FiniteGuardedAnalysis.h"
#include "PTO/Transforms/FrontierSynch/Shared/SequenceAnalysis.h"
#include "llvm/Support/SHA256.h"
#include "llvm/ADT/StringExtras.h"
namespace {
namespace cleanup_fs = mlir::pto::frontiersynch;
std::string deferredCleanupSource(unsigned variant)
{
    std::string text = R"mlir(
      !mat = !pto.tile_buf<mat, 16x16xf16, blayout=col_major, slayout=row_major>
      !left = !pto.tile_buf<left, 16x16xf16, slayout=row_major>
      !right = !pto.tile_buf<right, 16x16xf16, slayout=col_major>
      !acc = !pto.tile_buf<acc, 16x16xf32, blayout=col_major, slayout=row_major, fractal=1024>
      module attributes {pto.target_arch = "a3"} {
        func.func @cleanup(%g: i1, %h: i1, %n: index) {
          %base = arith.constant 0 : i64
          %other = arith.constant 512 : i64
          %accaddr = arith.constant 2048 : i64
          %zero = arith.constant 0 : index
          %one = arith.constant 1 : index
          %mat = pto.alloc_tile addr = %base : !mat
          %left = pto.alloc_tile addr = %base : !left
          %second = pto.alloc_tile addr = %other : !left
          %right = pto.alloc_tile addr = %base : !right
          %acc = pto.alloc_tile addr = %base : !acc
          %aux = pto.alloc_tile addr = %accaddr : !acc
          pto.textract ins(%mat, %zero, %zero : !mat, index, index) outs(%left : !left) {test.label = "L"}
    )mlir";
    if (variant == 1) {
        text += R"mlir(
          scf.for %i = %zero to %n step %one {
            pto.tmatmul ins(%second, %right : !left, !right) outs(%aux : !acc) {test.label = "U"}
          }
        )mlir";
    }
    text += R"mlir(
          %choice = scf.if %g -> i1 { scf.yield %h : i1 } else { scf.yield %g : i1 }
    )mlir";
    if (variant == 2) {
        text += "scf.for %outer = %zero to %n step %one { scf.for %inner = %zero to %n step %one {";
    }
    text += R"mlir(
          scf.if %choice {
            pto.tmatmul ins(%left, %right : !left, !right) outs(%acc : !acc) {test.label = "C"}
          }
    )mlir";
    if (variant == 2) { text += "} }"; }
    return text + "return } }";
}
mlir::FailureOr<std::unique_ptr<cleanup_fs::PreparedLogicalPlan>> prepareDeferredCleanup(
    mlir::func::FuncOp function, unsigned variant)
{
    using namespace mlir;
    pto::SyncInput input(pto::GMAliasPolicy::MayNotAlias);
    if (failed(input.build(function))) { return failure(); }
    if (variant) {
        auto program = cleanup_fs::recognizeProgram(function, input);
        if (failed(program)) { return failure(); }
        auto analysis = cleanup_fs::analyzeSequence(function, input, *program);
        return cleanup_fs::prepareSequenceInsertion(analysis);
    }
    cleanup_fs::PhaseIndex index;
    if (failed(index.build(function, input))) { return failure(); }
    SmallVector<Operation*> roots;
    for (auto& op : function.front()) { roots.push_back(&op); }
    auto analysis = cleanup_fs::analyzeFiniteGuarded(function, roots, index, input);
    return cleanup_fs::prepareFiniteGuardedInsertion(analysis);
}
struct DeferredCleanupTrace {
    mlir::DenseMap<mlir::Value, llvm::APInt> values;
    std::vector<std::string> events;
    unsigned barriers = 0;
    mlir::FailureOr<mlir::SmallVector<llvm::APInt>> block(mlir::Block& body)
    {
        using namespace mlir;
        for (auto& op : body) {
            if (isa<scf::YieldOp, func::ReturnOp>(op)) {
                SmallVector<APInt> result;
                for (auto value : op.getOperands()) {
                    if (!values.count(value)) { return failure(); }
                    result.push_back(values.lookup(value));
                }
                return result;
            }
            if (auto loop = dyn_cast<scf::ForOp>(op)) {
                if (!loop.getInitArgs().empty()) { return failure(); }
                const auto lower = values.lookup(loop.getLowerBound()).getSExtValue();
                const auto upper = values.lookup(loop.getUpperBound()).getSExtValue();
                const auto step = values.lookup(loop.getStep()).getSExtValue();
                if (step <= 0) { return failure(); }
                for (auto i = lower; i < upper; i += step) {
                    values[loop.getInductionVar()] = APInt(64, i);
                    if (failed(block(*loop.getBody()))) { return failure(); }
                }
            } else if (auto branch = dyn_cast<scf::IfOp>(op)) {
                if (!values.count(branch.getCondition())) { return failure(); }
                auto* selected = values.lookup(branch.getCondition()).isZero() ?
                    branch.elseBlock() : branch.thenBlock();
                if (!selected) { continue; }
                auto results = block(*selected);
                if (failed(results) || results->size() != branch.getNumResults()) { return failure(); }
                for (auto [value, bits] : llvm::zip(branch.getResults(), *results)) { values[value] = bits; }
            } else if (isa<pto::SetFlagOp, pto::WaitFlagOp, pto::SetFlagDynOp, pto::WaitFlagDynOp>(op)) {
                auto fixed = op.getAttrOfType<pto::EventAttr>("event_id");
                if (!fixed && (op.getNumOperands() != 1 || !values.count(op.getOperand(0)))) { return failure(); }
                const auto id = fixed ? static_cast<uint64_t>(fixed.getEvent()) :
                    values.lookup(op.getOperand(0)).getZExtValue();
                if (id != 3) { return failure(); }
                events.push_back(isa<pto::SetFlagOp, pto::SetFlagDynOp>(op) ? "SET" : "WAIT");
            } else if (auto barrier = dyn_cast<pto::BarrierOp>(op)) {
                auto pipe = barrier->getAttrOfType<pto::PipeAttr>("pipe");
                if (!pipe) { return failure(); }
                if (pipe.getPipe() == pto::PIPE::PIPE_ALL) { ++barriers; }
                else if (pipe.getPipe() == pto::PIPE::PIPE_M) { events.push_back("BARRIER_M"); }
                else { return failure(); }
            } else if (auto label = op.getAttrOfType<StringAttr>("test.label")) {
                events.push_back(label.getValue().str());
            } else if (op.getName().getStringRef().starts_with("pto.")) {
                if (!isa<pto::AllocTileOp>(op)) { return failure(); }
            } else if (!evaluateDeferredScalar(op, values)) { return failure(); }
        }
        return failure();
    }
};
bool traceDeferredCleanup(mlir::func::FuncOp function, unsigned variant)
{
    using namespace mlir;
    for (unsigned g : {0U, 1U}) {
        for (unsigned h : {0U, 1U}) {
            for (unsigned n : {0U, 1U, 3U}) {
                DeferredCleanupTrace trace;
                trace.values[function.getArgument(0)] = APInt(1, g);
                trace.values[function.getArgument(1)] = APInt(1, h);
                trace.values[function.getArgument(2)] = APInt(64, n);
                const unsigned consumers = variant == 2 ? n * n : 1;
                std::vector<std::string> expected{"L", "SET"};
                // The repeated matmul writes the same accumulator. Preserve
                // its required same-pipe ordering between successive payloads.
                auto appendRepeated = [&](unsigned count, llvm::StringRef label) {
                    for (unsigned i = 0; i < count; ++i) {
                        if (i) { expected.push_back("BARRIER_M"); }
                        expected.push_back(label.str());
                    }
                };
                if (variant == 1) { appendRepeated(n, "U"); }
                if (g && h && consumers) {
                    expected.push_back("WAIT"); appendRepeated(consumers, "C");
                } else { expected.push_back("WAIT"); }
                if (failed(trace.block(function.front())) || trace.events != expected || trace.barriers > 1) {
                    llvm::errs() << "deferred cleanup trace variant=" << variant << " g=" << g << " h=" << h
                        << " n=" << n << " full barriers=" << trace.barriers << "\nactual:";
                    for (const auto& event : trace.events) { llvm::errs() << " " << event; }
                    llvm::errs() << "\nexpected:";
                    for (const auto& event : expected) { llvm::errs() << " " << event; }
                    llvm::errs() << "\n";
                    return false;
                }
            }
        }
    }
    return true;
}
// A retained singleton witness is necessary, but cannot qualify a different
// emitted WAIT guard. Exercise the real producer and owned recipe restore API;
// the invalid case changes only activity at the nested consumer's original cut.
bool checkPendingCleanupActualGuard(mlir::MLIRContext* context)
{
    using namespace mlir;
    auto fail = [](unsigned line) {
        llvm::errs() << "checkPendingCleanupActualGuard failed at source line " << line << "\n";
        return false;
    };
    for (bool corrupt : {false, true}) {
        auto module = parseSourceString<ModuleOp>(deferredCleanupSource(2), context);
        if (!module) { return fail(__LINE__); }
        auto function = module->lookupSymbol<func::FuncOp>("cleanup");
        auto prepared = prepareDeferredCleanup(function, 2);
        if (failed(prepared)) { return fail(__LINE__); }
        auto& plan = **prepared;
        std::shared_ptr<const cleanup_fs::EndpointCutChoices> witness;
        Operation* consumer = nullptr;
        for (const auto& family : plan.families) {
            if (!family.targetChoices || !family.targetChoices->pendingMaterialization()) { continue; }
            if (witness) { return fail(__LINE__); }
            witness = family.targetChoices; consumer = family.targetCut.before;
        }
        if (!witness || !witness->validPendingSelection() || !consumer) { return fail(__LINE__); }
        auto nodes = cleanup_fs::deferredFamilyRecipe(plan);
        if (failed(nodes)) { return fail(__LINE__); }
        unsigned changed = 0;
        if (corrupt) {
            for (auto& node : *nodes) {
                if (node.kind == cleanup_fs::DeferredFamilyNode::Kind::Activity) {
                    auto found = node.contextCuts.find(consumer);
                    if (found != node.contextCuts.end()) { found->second = node.arena->boolean(true); }
                }
                if (node.kind != cleanup_fs::DeferredFamilyNode::Kind::Leaf) { continue; }
                for (unsigned i = 0; i < node.bindings.size(); ++i) {
                    auto& binding = node.bindings[i];
                    if (binding.piece.kind != cleanup_fs::EndpointKind::Wait ||
                        binding.piece.cut.before != consumer) { continue; }
                    if (!node.arena || i >= node.numerical.size() || node.numerical[i]) { return fail(__LINE__); }
                    binding.activity = node.arena->boolean(true);
                    ++changed;
                }
            }
            if (!changed) { return fail(__LINE__); }
        }
        auto restored = cleanup_fs::restoreDeferredFamilyRecipe(*nodes);
        if (failed(restored)) { return fail(__LINE__); }
        plan.familyRecipe = *restored; plan.familyProtocol.reset();
        // Freeze the replacement recipe normally, retaining the original valid
        // pending witness. No stale protocol-key failure may mask the guard check.
        if (failed(cleanup_fs::planPhysicalFamilyOutput(function, plan, {3})) ||
            !cleanup_fs::validFamilyEndpointProtocol(plan) || !witness->validPendingSelection()) {
            return fail(__LINE__);
        }
        const auto before = deferredText(function);
        const auto predicted = cleanup_fs::physicalFamilyConstructionOperations(plan);
        if (bool(predicted) == corrupt || !plan.preparation.empty() || !plan.endpoints.empty() ||
            plan.familyWork.materializations || deferredText(function) != before) { return fail(__LINE__); }
        if (!corrupt) {
            if (failed(cleanup_fs::insertLogicalSynchronization(function, plan)) ||
                !traceDeferredCleanup(function, 2)) { return fail(__LINE__); }
            const auto& work = plan.familyWork;
            if (*predicted != work.preparationScalars + work.compactionScalars + work.mappingScalars +
                work.barrierScalars) { return fail(__LINE__); }
            continue;
        }
        const auto recipe = plan.familyRecipe;
        const auto protocol = plan.familyProtocol;
        ScopedDiagnosticHandler diagnostics(context, [](Diagnostic&) { return success(); });
        if (succeeded(cleanup_fs::materializeLogicalFamilies(plan)) || deferredText(function) != before ||
            plan.familyRecipe != recipe || plan.familyProtocol != protocol || !plan.preparation.empty() ||
            !plan.endpoints.empty() || plan.familyWork.materializations || !witness->validPendingSelection()) {
            return fail(__LINE__);
        }
        unsigned retained = 0;
        for (const auto& family : plan.families) { retained += family.targetChoices == witness; }
        if (retained != 1) { return fail(__LINE__); }
    }
    return true;
}
bool checkDeferredCleanup(mlir::MLIRContext* context)
{
    using namespace mlir;
    auto fail = [](unsigned line) {
        llvm::errs() << "checkDeferredCleanup failed at source line " << line << "\n";
        return false;
    };
    if (!checkPendingCleanupActualGuard(context)) { return fail(__LINE__); }
    for (unsigned variant : {0U, 1U, 2U}) {
        std::string saved, combined;
        for (unsigned route : {0U, 1U, 2U}) {
            auto module = parseSourceString<ModuleOp>(deferredCleanupSource(variant), context);
            if (!module) { return fail(__LINE__); }
            auto function = module->lookupSymbol<func::FuncOp>("cleanup");
            auto plan = prepareDeferredCleanup(function, variant);
            if (failed(plan) || !(*plan)->allocationCertificate || !(*plan)->familyRecipe ||
                !(*plan)->preparation.empty() || (*plan)->familyWork.materializations) { return fail(__LINE__); }
            unsigned choices = 0;
            for (const auto& family : (*plan)->families) {
                if (family.targetChoices && family.targetChoices->pendingSelection()) { ++choices; }
            }
            if (choices != 1) { return fail(__LINE__); }
            std::optional<uint64_t> predicted;
            if (route == 1) {
                if (failed(cleanup_fs::planPhysicalFamilyOutput(function, **plan, {3}))) { return fail(__LINE__); }
                predicted = cleanup_fs::physicalFamilyConstructionOperations(**plan);
                if (!predicted || !(*plan)->preparation.empty() || (*plan)->familyWork.materializations) {
                    return fail(__LINE__);
                }
            }
            (*plan)->deferredOutput = route == 2;
            if (failed(cleanup_fs::insertLogicalSynchronization(function, **plan))) { return fail(__LINE__); }
            if (predicted) {
                const auto& work = (*plan)->familyWork;
                if (*predicted != work.preparationScalars + work.compactionScalars + work.mappingScalars +
                    work.barrierScalars) { return fail(__LINE__); }
            }
            if (route == 2) {
                if ((*plan)->familyWork.materializations) { return fail(__LINE__); }
                auto metadata = function->getAttrOfType<DictionaryAttr>("pto.deferred_family");
                if (!metadata || metadata.getAs<IntegerAttr>("version").getInt() != 3) { return fail(__LINE__); }
                saved = deferredText(module->getOperation());
            } else {
                if (route == 0 && failed(cleanup_fs::allocatePhysicalEventIds(function, {3}))) {
                    return fail(__LINE__);
                }
                if (failed(verify(function)) || !traceDeferredCleanup(function, variant)) { return fail(__LINE__); }
                if (route == 1) { combined = deferredText(function); }
            }
        } // Every mathematical producer and original plan has been destroyed.
        auto replay = parseSourceString<ModuleOp>(saved, context);
        if (!replay) { return fail(__LINE__); }
        auto function = replay->lookupSymbol<func::FuncOp>("cleanup");
        if (failed(cleanup_fs::allocatePhysicalEventIds(function, {3})) || failed(verify(function)) ||
            deferredText(function) != combined || !traceDeferredCleanup(function, variant)) { return fail(__LINE__); }
        for (unsigned corruption = 0; corruption < (variant == 2 ? 10U : 9U); ++corruption) {
            auto broken = parseSourceString<ModuleOp>(saved, context);
            if (!broken) { return fail(__LINE__); }
            auto target = broken->lookupSymbol<func::FuncOp>("cleanup");
            Builder builder(context);
            auto original = target->getAttrOfType<DictionaryAttr>("pto.deferred_family");
            auto entries = original.getAs<ArrayAttr>("cleanup_choices");
            if (!entries || entries.size() != 1) { return fail(__LINE__); }
            NamedAttrList choice(cast<DictionaryAttr>(entries[0])), metadata(original);
            auto cuts = cast<DenseI64ArrayAttr>(choice.get("cuts"));
            if (corruption == 0) { choice.set("cuts", builder.getDenseI64ArrayAttr({cuts[1], cuts[0]})); }
            else if (corruption == 1) { choice.set("cuts", builder.getDenseI64ArrayAttr({cuts[0], cuts[0]})); }
            else if (corruption == 2) { choice.set("cuts", builder.getDenseI64ArrayAttr({cuts[0]})); }
            else if (corruption == 3) { choice.set("predicate", builder.getI64IntegerAttr(INT64_MAX)); }
            else if (corruption == 4) { choice.set("arena", builder.getI64IntegerAttr(-1)); }
            else if (corruption == 5) { choice.set("predicate", builder.getI32IntegerAttr(0)); }
            else if (corruption == 7 || corruption == 9) {
                const auto arenaId = cast<IntegerAttr>(choice.get("arena")).getInt();
                auto arena = cast<DictionaryAttr>(original.getAs<ArrayAttr>("arenas")[arenaId]);
                auto graph = arena.getAs<DictionaryAttr>("dag");
                auto roots = graph.getAs<DenseI64ArrayAttr>("roots");
                auto nodes = graph.getAs<ArrayAttr>("nodes");
                bool replaced = false;
                for (auto [index, root] : llvm::enumerate(roots.asArrayRef())) {
                    auto node = cast<DictionaryAttr>(nodes[root]);
                    const bool boolean = node.getAs<BoolAttr>("boolean").getValue();
                    const bool always = boolean && node.getAs<IntegerAttr>("opcode").getInt() == 0 &&
                        node.getAs<DictionaryAttr>("payload").getAs<IntegerAttr>("literal").getInt() == 1;
                    if (corruption == 7 ? !boolean : always) {
                        choice.set("predicate", builder.getI64IntegerAttr(index)); replaced = true; break;
                    }
                }
                if (!replaced) { return fail(__LINE__); }
            } else if (corruption == 8) {
                auto sources = original.getAs<ArrayAttr>("sources");
                if (!sources || sources.empty()) { return fail(__LINE__); }
                SmallVector<Attribute> shortened(sources.begin(), sources.end());
                shortened.pop_back(); metadata.set("sources", builder.getArrayAttr(shortened));
            }
            SmallVector<Attribute> changed{choice.getDictionary(context)};
            if (corruption == 6) { changed.push_back(changed.front()); }
            metadata.set("cleanup_choices", builder.getArrayAttr(changed));
            resealDeferredContext(target, metadata.getDictionary(context));
            const auto before = deferredText(target);
            ScopedDiagnosticHandler diagnostics(context, [](Diagnostic&) { return success(); });
            if (succeeded(cleanup_fs::allocatePhysicalEventIds(target, {3})) || deferredText(target) != before) {
                return fail(__LINE__);
            }
        }
    }
    return true;
}
} // namespace
#endif
