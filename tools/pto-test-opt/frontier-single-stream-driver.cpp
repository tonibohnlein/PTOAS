// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Actual production placement/physical admission and mutation rollback.
#include "../../lib/PTO/Transforms/FrontierSynch/DirectEmissionInternal.h"
#include "../../lib/PTO/Transforms/FrontierSynch/ExplicitPhysicalEmission.h"
#include "../../lib/PTO/Transforms/FrontierSynch/SingleStreamLoop.h"
#include "../../lib/PTO/Transforms/FrontierSynch/RegionalRequests.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DenseMap.h"
#include <functional>
namespace {
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
std::string render(Operation* operation)
{
    std::string text;
    llvm::raw_string_ostream output(text);
    operation->print(output);
    return text;
}
bool repeatedCoordinates(fs::DirectEmissionResult& result)
{
    scf::ForOp loop;
    result.pending->walk([&](scf::ForOp operation) { loop = operation; });
    if (!loop) { return false; }
    const auto& plan = *result.selected->boundaryLoop;
    const auto count = result.selected->sites.size();
    const auto ready = static_cast<int64_t>(plan.first() * count + plan.firstCompute());
    const auto release = static_cast<int64_t>(plan.last() * count + plan.first());
    bool valid = true;
    result.pending->walk([&](Operation* operation) {
        if (!isa<pto::LogicalSetOp, pto::LogicalWaitOp>(operation)) { return; }
        const auto key = operation->getAttrOfType<IntegerAttr>("key").getInt();
        if (key != ready && key != release) { valid &= operation->getNumOperands() == 0; return; }
        if (operation->getNumOperands() != 1) { valid = false; return; }
        auto coordinate = operation->getOperand(0);
        if (key == ready || isa<pto::LogicalSetOp>(operation)) {
            valid &= coordinate == loop.getInductionVar(); return;
        }
        auto previous = coordinate.getDefiningOp<arith::SubIOp>();
        auto one = previous ? previous.getRhs().getDefiningOp<arith::ConstantOp>() : arith::ConstantOp{};
        auto literal = one ? dyn_cast<IntegerAttr>(one.getValue()) : IntegerAttr{};
        valid &= previous && previous.getLhs() == loop.getInductionVar() && literal && literal.getInt() == 1;
    });
    return valid;
}
bool frameCoordinates(fs::DirectEmissionResult& result, ArrayRef<scf::ForOp> frames)
{
    const auto& plan = *result.selected->boundaryLoop;
    if (!plan.repeatedPair()) { return true; }
    const auto count = result.selected->sites.size();
    const auto ready = static_cast<int64_t>(plan.first() * count + plan.firstCompute());
    const auto release = static_cast<int64_t>(plan.last() * count + plan.first());
    bool valid = true;
    result.pending->walk([&](Operation* operation) {
        if (!isa<pto::LogicalSetOp, pto::LogicalWaitOp>(operation)) { return; }
        const auto key = operation->getAttrOfType<IntegerAttr>("key").getInt();
        if (key != ready && key != release) { valid &= operation->getNumOperands() == 0; return; }
        if (operation->getNumOperands() != frames.size()) { valid = false; return; }
        if (key == ready || isa<pto::LogicalSetOp>(operation)) {
            for (auto [coordinate, frame] : llvm::zip(operation->getOperands(), frames)) {
                auto originalFrame = frame;
                valid &= coordinate == originalFrame.getInductionVar();
            }
            return;
        }
        // Independent tuple predecessor checks: each carry depth, including
        // inner-to-outer re-entry. This test does not consult quotient distances.
        for (unsigned carry = 0; carry < frames.size(); ++carry) {
            DenseMap<Value, int64_t> values;
            SmallVector<int64_t> bounds;
            for (unsigned axis = 0; axis < frames.size(); ++axis) {
                auto frame = frames[axis];
                auto constant = frame.getUpperBound().getDefiningOp<arith::ConstantOp>();
                bounds.push_back(constant ? cast<IntegerAttr>(constant.getValue()).getInt() : 3);
                values[frame.getInductionVar()] = axis == carry ? 1 : 0;
                values[frame.getUpperBound()] = bounds.back();
            }
            if (bounds[carry] < 2) { continue; }
            std::function<std::optional<int64_t>(Value)> evaluate = [&](Value value) -> std::optional<int64_t> {
                if (auto found = values.find(value); found != values.end()) { return found->second; }
                if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
                    return cast<IntegerAttr>(constant.getValue()).getInt();
                }
                auto* definition = value.getDefiningOp();
                if (!definition) { return std::nullopt; }
                SmallVector<int64_t> arguments;
                for (auto operand : definition->getOperands()) {
                    auto argument = evaluate(operand);
                    if (!argument) { return std::nullopt; } arguments.push_back(*argument);
                }
                if (isa<arith::SubIOp>(definition)) { return arguments[0] - arguments[1]; }
                if (isa<arith::AndIOp>(definition)) { return arguments[0] && arguments[1]; }
                if (isa<arith::SelectOp>(definition)) { return arguments[0] ? arguments[1] : arguments[2]; }
                if (auto comparison = dyn_cast<arith::CmpIOp>(definition)) {
                    if (comparison.getPredicate() == arith::CmpIPredicate::eq) {
                        return arguments[0] == arguments[1];
                    }
                    if (comparison.getPredicate() == arith::CmpIPredicate::sgt) {
                        return arguments[0] > arguments[1];
                    }
                }
                return std::nullopt;
            };
            for (unsigned axis = 0; axis < frames.size(); ++axis) {
                auto actual = evaluate(operation->getOperand(axis));
                const int64_t expected = axis > carry ? bounds[axis] - 1 : 0;
                valid &= actual && *actual == expected;
            }
        }
    });
    return valid;
}
bool rowExpandSharedWitnesses(func::FuncOp function, const pto::SyncInput& input)
{
    pto::TRowExpandMulOp operation;
    function.walk([&](pto::TRowExpandMulOp candidate) { operation = candidate; });
    if (!operation) { return true; }
    const auto records = input.target().phases();
    if (records.size() != 3 || records[1].phase->elementOp != operation) { return false; }
    const auto* source = records.front().phase;
    const auto* body = records[1].phase;
    for (Value value : {operation.getSrc0(), operation.getSrc1()}) {
        SmallVector<const pto::BaseMemInfo*> reads;
        for (const auto* memory : body->useVec) {
            if (memory->baseBuffer == value) { reads.push_back(memory); }
        }
        pto::DepBaseMemInfoPairVec pairs;
        if (reads.empty() || !input.memory().DepBetween(source->defVec, reads, pairs) ||
            !llvm::all_of(pairs, [](const auto& pair) {
                return pair.first->rootBuffer != pair.second->rootBuffer;
            })) { return false; }
    }
    const auto scratch = operation.getTmp();
    auto hasScratch = [&](ArrayRef<const pto::BaseMemInfo*> memories) {
        return llvm::any_of(memories, [&](const auto* memory) { return memory->baseBuffer == scratch; });
    };
    pto::SyncInput unspecified;
    return hasScratch(body->useVec) && hasScratch(body->defVec) &&
        succeeded(unspecified.build(function));
}
bool legacyElseRejected(func::FuncOp function, const pto::SyncInput& original, MLIRContext& context)
{
    auto sourceModule = function->getParentOfType<ModuleOp>();
    if (!sourceModule) { return false; }
    OwningOpRef<ModuleOp> container(cast<ModuleOp>(sourceModule->clone()));
    auto variant = container->lookupSymbol<func::FuncOp>(function.getSymName());
    if (!variant || container->getOperation()->getAttrDictionary() != sourceModule->getAttrDictionary()) {
        return false;
    }
    scf::IfOp common;
    for (auto choice : variant.front().getOps<scf::IfOp>()) {
        if (!choice.getThenRegion().front().getOps<scf::ForOp>().empty()) { common = choice; }
    }
    if (!common) { return false; }
    auto& region = common.getElseRegion();
    if (region.empty()) { region.push_back(new Block()); }
    OpBuilder builder(&context);
    builder.setInsertionPointToStart(&region.front());
    builder.create<arith::ConstantIndexOp>(common.getLoc(), 0);
    if (!isa<scf::YieldOp>(region.front().back())) {
        builder.setInsertionPointToEnd(&region.front());
        builder.create<scf::YieldOp>(common.getLoc());
    }
    if (failed(verify(variant))) { return false; }
    pto::SyncInput changed;
    // Bind the SAME normative profile and original module/architecture. Only
    // original else participation changes; no missing-target false positive.
    if (failed(changed.build(variant)) ||
        changed.target().architecture() != original.target().architecture() ||
        changed.target().phases().size() != original.target().phases().size()) { return false; }
    const auto oldPhases = original.target().phases(), newPhases = changed.target().phases();
    for (auto [before, after] : llvm::zip(oldPhases, newPhases)) {
        if (before.context.core != after.context.core) { return false; }
    }
    if (
        !changed.target().participation().owner || changed.target().participation().thenOnly() ||
        changed.target().originalBarrierChain().barrier) { return false; }
    auto trace = fs::TraceDemandAnalysis::build(variant, changed);
    if (failed(trace)) { return false; }
    fs::CostLedger costs(false);
    std::string reason;
    auto child = fs::SingleStreamLoop::buildRepeatedRegion(variant, changed, *trace, costs, reason);
    auto invocation = fs::SingleStreamLoop::build(variant, changed, *trace, costs, reason);
    return failed(child) && failed(invocation);
}
bool sharedGuardDagChecks(func::FuncOp function)
{
    auto module = function->getParentOfType<ModuleOp>();
    if (!module) { return false; }
    OwningOpRef<ModuleOp> clone(cast<ModuleOp>(module->clone()));
    auto variant = clone->lookupSymbol<func::FuncOp>(function.getSymName());
    scf::ForOp inner;
    scf::IfOp exit;
    variant.walk<WalkOrder::PreOrder>([&](scf::ForOp loop) { inner = loop; });
    variant.walk([&](pto::TStoreOp store) { exit = store->getParentOfType<scf::IfOp>(); });
    if (!inner || !exit) { return false; }
    auto frames = pto::recoverSyncLoopFrames(variant, inner);
    if (failed(frames)) { return false; }
    unsigned initial = 0;
    if (!pto::syncFramesNonempty(exit.getCondition(), *frames, &initial)) { return false; }
    OpBuilder builder(exit);
    Value condition = exit.getCondition();
    constexpr unsigned depth = 64;
    for (unsigned level = 0; level < depth; ++level) {
        condition = builder.create<arith::AndIOp>(exit.getLoc(), condition, condition);
    }
    unsigned visited = 0;
    if (!pto::syncFramesNonempty(condition, *frames, &visited) || visited != initial + depth) { return false; }
    auto trueValue = builder.create<arith::ConstantIntOp>(exit.getLoc(), 1, 1);
    auto falseValue = builder.create<arith::ConstantIntOp>(exit.getLoc(), 0, 1);
    auto withTrue = builder.create<arith::AndIOp>(exit.getLoc(), condition, trueValue);
    auto withFalse = builder.create<arith::AndIOp>(exit.getLoc(), condition, falseValue);
    auto first = cast<scf::ForOp>(frames->front());
    auto unknown = builder.create<arith::CmpIOp>(exit.getLoc(), arith::CmpIPredicate::eq,
                                               first.getUpperBound(), first.getUpperBound());
    auto withUnknown = builder.create<arith::AndIOp>(exit.getLoc(), condition, unknown);
    if (!pto::syncFramesNonempty(withTrue, *frames, &visited) || visited != initial + depth + 2 ||
        pto::syncFramesNonempty(withFalse, *frames) || pto::syncFramesNonempty(withUnknown, *frames)) {
        return false;
    }
    exit.getConditionMutable().set(withTrue);
    if (failed(verify(*clone))) { return false; }
    pto::SyncInput input;
    if (failed(input.build(variant))) { return false; }
    auto trace = fs::TraceDemandAnalysis::build(variant, input);
    if (failed(trace)) { return false; }
    fs::CostLedger costs(false);
    const auto before = render(variant);
    auto pending = fs::emitDirectDemands(variant, input, *trace, costs);
    return pending.emitted && succeeded(fs::emitExplicitPhysical(input, *trace, pending, costs)) &&
           render(variant) == before;
}
int orderedFrameChecks(func::FuncOp function, const pto::SyncInput& input,
    const fs::TraceDemandAnalysis& trace, fs::CostLedger& costs)
{
    auto fail = [](StringRef reason) { llvm::errs() << reason << '\n'; return 1; };
    scf::ForOp outer;
    function.walk<WalkOrder::PreOrder>([&](scf::ForOp loop) { if (!outer) { outer = loop; } });
    if (!outer) { return fail("original outer frame missing"); }
    if (!sharedGuardDagChecks(function)) { return fail("shared guard DAG work/semantic bound failed"); }
    const auto original = render(function);
    fs::RegionalRequests requests(function, input, trace, costs);
    auto ports = fs::AnalysisNeeds::modeledCovers();
    ports.interfaces |= fs::interfaceBit(fs::DemandInterface::RegionBoundaryPorts);
    auto child = requests.request(outer, requests.contextFor(outer), fs::RegionalMode::Modeled, ports);
    auto cached = requests.request(outer, requests.contextFor(outer), fs::RegionalMode::Modeled, ports);
    if (!child.analysis || child.analysis != cached.analysis || !child.analysis->boundaryLoop ||
        !child.analysis->boundaryLoop->regionOnly || child.analysis->boundaryLoop->frames.size() < 2 ||
        child.analysis->boundaryLoop->frames.front() != outer) {
        llvm::errs() << "child outcome=" << child.outcome << " obligation=" << child.obligation
                     << " same-cache=" << (child.analysis == cached.analysis) << '\n';
        if (child.analysis && child.analysis->boundaryLoop) {
            llvm::errs() << "child frames=" << child.analysis->boundaryLoop->frames.size()
                         << " region-only=" << child.analysis->boundaryLoop->regionOnly << '\n';
        }
        return fail("proper cached frame child contract missing");
    }
    auto inner = child.analysis->boundaryLoop->loop;
    if (inner == outer) { return fail("frame child lost descendant ownership"); }
    auto visit = requests.request(inner, requests.contextFor(inner), fs::RegionalMode::Modeled, ports);
    if (visit.analysis || visit.obligation.empty() || StringRef(visit.obligation).contains("capacity")) {
        return fail("inner visit falsely advertised whole ancestor frame domain");
    }
    auto parent = requests.request(function, requests.context(), fs::RegionalMode::Modeled,
                                   fs::AnalysisNeeds::modeledCovers());
    if (!parent.analysis || !parent.analysis->boundaryLoop || parent.analysis->regionalChildren.size() != 1 ||
        parent.analysis->regionalChildren.front() != child.analysis ||
        parent.analysis->boundaryLoop->loopChild != child.analysis->boundaryLoop) {
        return fail("parent did not consume identical cached frame child");
    }
    auto strict = fs::AnalysisNeeds::minimumExact();
    strict.interfaces |= fs::interfaceBit(fs::DemandInterface::RegionBoundaryPorts);
    if (requests.request(outer, requests.contextFor(outer), fs::RegionalMode::Modeled, strict).analysis) {
        return fail("strict original minimum incorrectly admitted frame child");
    }
    for (unsigned mutation = 0; mutation < 7; ++mutation) {
        auto pending = fs::emitDirectDemands(function, input, trace, costs);
        if (!pending.emitted || !pending.selected || !pending.selected->boundaryLoop ||
            pending.selected->boundaryLoop->frames.size() < 2 || !pending.singleStreamPending) {
            llvm::errs() << pending.reason << '\n'; return fail("frame logical emission missing");
        }
        SmallVector<scf::ForOp> frames;
        pto::LogicalWaitOp wait;
        pto::LogicalSetOp publication;
        pto::BarrierOp barrier;
        pending.pending->walk<WalkOrder::PreOrder>([&](Operation* operation) {
            if (auto loop = dyn_cast<scf::ForOp>(operation)) { frames.push_back(loop); }
            if (auto endpoint = dyn_cast<pto::LogicalWaitOp>(operation)) { if (!wait) { wait = endpoint; } }
            if (auto endpoint = dyn_cast<pto::LogicalSetOp>(operation)) {
                if (!publication) { publication = endpoint; }
            }
            if (auto local = dyn_cast<pto::BarrierOp>(operation)) {
                if (local.getPipe().getPipe() == pto::PIPE::PIPE_V && !barrier) { barrier = local; }
            }
        });
        if (frames.size() < 2 || !wait || !publication || !barrier) {
            return fail("frame pending templates incomplete");
        }
        if (!frameCoordinates(pending, frames)) { return fail("original tuple generation matching failed"); }
        auto originalBound = outer.getUpperBound();
        if (mutation == 1) { wait.erase(); }
        if (mutation == 2) { frames.back()->setOperand(1, frames.front().getUpperBound()); }
        if (mutation == 3) {
            if (pending.selected->boundaryLoop->repeatedPair()) {
                if (publication->getNumOperands() != frames.size()) {
                    return fail("repeated frame publication identity missing");
                }
                publication->setOperand(frames.size() - 1, frames.front().getInductionVar());
            } else { publication->moveBefore(frames.front()); }
        }
        if (mutation == 4) {
            OpBuilder duplicate(publication); duplicate.setInsertionPointAfter(publication);
            duplicate.insert(publication->clone());
        }
        if (mutation == 5) { barrier.erase(); }
        if (mutation == 6) { outer->setOperand(1, function.getArgument(2)); }
        if (failed(verify(pending.pending.get()))) { return fail("frame mutation invalid original IR"); }
        const bool physical = succeeded(fs::emitExplicitPhysical(input, trace, pending, costs));
        if (mutation == 6) { outer->setOperand(1, originalBound); }
        if (physical != (mutation == 0) || render(function) != original) {
            llvm::errs() << "frame mutation=" << mutation << " physical=" << physical << '\n';
            return fail("frame physical transaction/seal mismatch");
        }
    }
    llvm::outs() << "verified cached original loop frames and seven pending/source admission cases\n";
    return 0;
}
} // namespace
int runSingleStreamChecks(llvm::StringRef path, mlir::MLIRContext& context)
{
    auto module = parseSourceFile<ModuleOp>(path, &context);
    if (!module) { return 1; }
    auto function = *module->getOps<func::FuncOp>().begin();
    const auto before = render(function);
    pto::SyncInput input;
    if (failed(input.build(function)) ||
        !rowExpandSharedWitnesses(function, input)) { return 1; }
    auto trace = fs::TraceDemandAnalysis::build(function, input);
    if (failed(trace) || input.slotSelections().size() < 2) { return 1; }
    fs::CostLedger costs(true);
    bool nested = false;
    function.walk([&](scf::ForOp loop) { nested |= bool(loop->getParentOfType<scf::ForOp>()); });
    if (nested) { return orderedFrameChecks(function, input, *trace, costs); }
    const bool repeated = input.target().originalBarrierChain().barrier != nullptr;
    const bool prefix = input.target().originalBarrierChain().prefix != nullptr;
    const bool conditional = input.target().participation().owner != nullptr;
    if (repeated && conditional && !legacyElseRejected(function, input, context)) {
        llvm::errs() << "nonempty scalar else incorrectly admitted legacy protocol\n"; return 1;
    }
    const bool computed = input.target().participation().witness.origin ==
        pto::SyncActivationOrigin::ComparisonResult;
    const unsigned mutationCount = computed ? (!repeated ? 18U : 15U) :
        conditional ? 12U : prefix ? 10U : repeated ? 9U : 5U;
    for (unsigned mutation = 0; mutation < mutationCount; ++mutation) {
        if ((mutation == 9 && !prefix) || (!repeated && mutation >= 5 && mutation <= 8)) { continue; }
        auto result = fs::emitDirectDemands(function, input, *trace, costs);
        if (!result.emitted || result.selected->kind != fs::SelectedAnalysis::Kind::BoundaryLoop ||
            !result.singleStreamPending || failed(verify(result.pending.get()))) {
            llvm::errs() << result.reason << '\n'; return 1;
        }
        if (repeated && !repeatedCoordinates(result)) {
            llvm::errs() << "repeated logical source generation mismatch\n"; return 1;
        }
        pto::LogicalWaitOp wait;
        result.pending->walk([&](pto::LogicalWaitOp operation) { if (!wait) { wait = operation; } });
        if (!wait) { return 1; }
        if (mutation == 1) { wait.erase(); }
        if (mutation == 2) {
            auto guard = wait->getParentOfType<scf::IfOp>();
            auto compare = guard.getCondition().getDefiningOp<arith::CmpIOp>();
            if (!compare) { return 1; }
            compare->setAttr("predicate", arith::CmpIPredicateAttr::get(&context, arith::CmpIPredicate::ne));
        }
        if (mutation == 3) {
            auto terminal = *result.pending->getBody().front().getOps<func::ReturnOp>().begin();
            auto barrier = dyn_cast<pto::BarrierOp>(terminal->getPrevNode());
            if (!barrier) { return 1; }
            barrier.erase();
        }
        if (mutation == 4) {
            auto guard = wait->getParentOfType<scf::IfOp>();
            wait->moveBefore(guard);
        }
        if (mutation >= 5 && mutation <= 7) {
            pto::BarrierOp original;
            result.pending->walk([&](pto::BarrierOp operation) {
                if (operation.getPipe().getPipe() == pto::PIPE::PIPE_MTE2) { original = operation; }
            });
            if (!original) { return 1; }
            if (mutation == 5) { original.erase(); }
            if (mutation == 6) {
                Operation* producer = nullptr;
                result.pending->walk([&](pto::TLoadOp operation) { producer = operation; });
                if (!producer) { return 1; }
                original->moveAfter(producer);
            }
            if (mutation == 7) { original->setAttr("pipe", pto::PipeAttr::get(&context, pto::PIPE::PIPE_V)); }
        }
        if (mutation == 8) { wait->setOperands(ValueRange{}); }
        if (mutation == 9) {
            auto initial = *result.pending->front().getOps<pto::TLoadOp>().begin();
            initial->setOperand(0, result.pending->getArgument(1));
        }
        if (mutation >= 12 && mutation <= 14) {
            scf::ForOp loop;
            result.pending->walk([&](scf::ForOp operation) { loop = operation; });
            auto outer = loop ? loop->getParentOfType<scf::IfOp>() : scf::IfOp{};
            auto comparison = outer ? outer.getCondition().getDefiningOp<arith::CmpIOp>() : arith::CmpIOp{};
            if (!comparison) { return 1; }
            const bool argumentFirst = input.target().participation().witness.argumentFirst;
            if (mutation == 12) {
                comparison->setAttr("predicate", arith::CmpIPredicateAttr::get(&context, arith::CmpIPredicate::ne));
            } else if (mutation == 13) {
                comparison->setOperand(argumentFirst ? 0 : 1, result.pending->getArgument(2));
            } else {
                auto* literal = comparison->getOperand(argumentFirst ? 1 : 0).getDefiningOp();
                if (!literal) { return 1; }
                literal->setAttr("value", OpBuilder(&context).getIndexAttr(1));
            }
        }
        if (mutation == 10 || mutation == 11) {
            scf::ForOp loop;
            result.pending->walk([&](scf::ForOp operation) { loop = operation; });
            auto outer = loop ? loop->getParentOfType<scf::IfOp>() : scf::IfOp{};
            if (!outer) { return 1; }
            if (mutation == 10) {
                OpBuilder builder(outer);
                auto condition = builder.create<arith::ConstantOp>(outer.getLoc(), builder.getBoolAttr(false));
                outer->setOperand(0, condition);
            } else {
                pto::LogicalWaitOp final;
                result.pending->walk([&](pto::LogicalWaitOp operation) {
                    if (!operation->getNumOperands()) { final = operation; }
                });
                if (!final) { return 1; }
                final->moveBefore(outer);
            }
        }
        if (mutation >= 15) {
            pto::LogicalSetOp publication;
            result.pending->walk([&](pto::LogicalSetOp operation) { if (!publication) { publication = operation; } });
            scf::ForOp loop;
            result.pending->walk([&](scf::ForOp operation) { loop = operation; });
            auto common = loop ? loop->getParentOfType<scf::IfOp>() : scf::IfOp{};
            if (!publication || !common) { return 1; }
            if (mutation == 15) { publication->moveBefore(common); }
            else if (mutation == 16) { OpBuilder(publication).clone(*publication); }
            else {
                pto::TXorOp consumer;
                loop->walk([&](pto::TXorOp operation) { if (!consumer) { consumer = operation; } });
                if (!consumer) { return 1; }
                publication->moveAfter(consumer);
            }
            // These are valid SSA mutations: a dominating unguarded SET or
            // surplus/late publication must fail matching/cut certification.
            if (failed(verify(result.pending.get()))) { return 1; }
        }
        auto status = fs::emitExplicitPhysical(input, *trace, result, costs);
        if ((mutation == 0) != succeeded(status) || render(function) != before) {
            llvm::errs() << result.reason << '\n'; return 1;
        }
        if (mutation && result.reason.find("changed after placement") == std::string::npos) { return 1; }
    }
    fs::RegionalRequests requests(function, input, *trace, costs);
    auto needs = fs::AnalysisNeeds::modeledCovers();
    SmallVector<scf::ForOp> sequence;
    function.walk([&](scf::ForOp owner) { sequence.push_back(owner); });
    if (!repeated && sequence.size() == 2) {
        SmallVector<fs::SelectedAnalysisHandle> children;
        auto ports = fs::AnalysisNeeds::modeledCovers();
        ports.interfaces |= fs::interfaceBit(fs::DemandInterface::RegionBoundaryPorts);
        for (auto owner : sequence) {
            auto child = requests.request(owner, requests.contextFor(owner), fs::RegionalMode::Modeled, ports);
            auto cached = requests.request(owner, requests.contextFor(owner), fs::RegionalMode::Modeled, ports);
            if (!child.analysis || cached.analysis != child.analysis || !child.analysis->boundaryLoop ||
                child.analysis->boundaryLoop->loop != owner ||
                child.analysis->boundaryLoop->bound != owner.getUpperBound() ||
                !child.analysis->boundaryLoop->regionOnly ||
                child.analysis->sites.size() != child.analysis->boundaryLoop->bodies.size()) { return 1; }
            const auto& body = *child.analysis->boundaryLoop;
            auto origin = input.target().participation().armFor(owner);
            if (origin && (body.armRegion != origin->region || body.activationOutcome != origin->outcome)) { return 1; }
            auto closed = body.portQuery(false, fs::PeriodicEventKind::Start, false, fs::PeriodicEventKind::Start);
            if (failed(closed) || closed->bound != owner.getUpperBound() || closed->armRegion != body.armRegion ||
                closed->activationOutcome != body.activationOutcome) { return 1; }
            for (int64_t trips : {0, 1, 2, 5}) {
                for (unsigned from = 0; from < 4; ++from) {
                    for (unsigned to = 0; to < 4; ++to) {
                        auto query = body.portQuery(from >= 2,
                            from % 2 ? fs::PeriodicEventKind::Completion : fs::PeriodicEventKind::Start,
                            to >= 2, to % 2 ? fs::PeriodicEventKind::Completion : fs::PeriodicEventKind::Start);
                        if (failed(query)) { return 1; }
                        if (!trips) { continue; }
                        const auto size = static_cast<int64_t>(body.bodies.size());
                        const auto source = from >= 2 ? trips * size - 1 : 0;
                        const auto target = to >= 2 ? trips * size - 1 : 0;
                        // Independent total RMW chain with paired native chains.
                        const bool expected = target > source ||
                            (target == source && !(from % 2 && !(to % 2)));
                        const bool actual = query->reachable &&
                            llvm::DynamicAPInt(query->coefficient) * llvm::DynamicAPInt(trips) +
                                llvm::DynamicAPInt(query->constant) >= 0;
                        if (actual != expected) { return 1; }
                    }
                }
            }
            auto strict = fs::AnalysisNeeds::minimumExact(); strict.interfaces = ports.interfaces;
            auto rejected = requests.request(owner, requests.contextFor(owner), fs::RegionalMode::Modeled, strict);
            if (rejected.analysis) { return 1; }
            children.push_back(child.analysis);
        }
        auto parent = requests.request(function, requests.context(), fs::RegionalMode::Modeled, needs);
        if (!parent.analysis || parent.analysis->regionalChildren != children ||
            parent.analysis->boundaryLoop->children[0] != children[0]->boundaryLoop ||
            parent.analysis->boundaryLoop->children[1] != children[1]->boundaryLoop || !requests.cacheHits()) {
            return 1;
        }
        for (unsigned mutation = 0; mutation < 3; ++mutation) {
            auto result = fs::emitDirectDemands(function, input, *trace, costs);
            if (!result.emitted) { return 1; }
            SmallVector<scf::ForOp> loops;
            result.pending->walk([&](scf::ForOp loop) { loops.push_back(loop); });
            if (mutation == 0) { loops[1]->setOperand(1, loops[0].getUpperBound()); }
            if (mutation == 1) {
                pto::BarrierOp local;
                loops[1]->walk([&](pto::BarrierOp barrier) { local = barrier; });
                if (!local) { return 1; } local.erase();
            }
            if (mutation == 2) {
                pto::LogicalSetOp publication;
                result.pending->walk([&](pto::LogicalSetOp operation) {
                    if (operation->getNumOperands()) { publication = operation; }
                });
                if (!publication) { return 1; } publication->setOperands(ValueRange{});
            }
            if (failed(verify(result.pending.get())) ||
                succeeded(fs::emitExplicitPhysical(input, *trace, result, costs)) ||
                render(function) != before) { return 1; }
        }
    }
    if (repeated) {
        auto* owner = input.target().originalBarrierChain().loop;
        needs.interfaces |= fs::interfaceBit(fs::DemandInterface::RegionBoundaryPorts);
        auto child = requests.request(owner, requests.contextFor(owner), fs::RegionalMode::Modeled, needs);
        if (!child.analysis || !child.analysis->boundaryLoop || !child.analysis->boundaryLoop->regionOnly ||
            child.analysis->sites != child.analysis->boundaryLoop->bodies) { return 1; }
        const auto first = child.analysis->boundaryLoop->firstPort();
        const auto last = child.analysis->boundaryLoop->lastPort();
        if (first.site != child.analysis->sites.front() || first.coefficient || first.offset ||
            last.site != child.analysis->sites.back() || last.coefficient != 1 || last.offset != -1 ||
            !first.requiresPositiveTrips || !last.requiresPositiveTrips ||
            first.activation != input.target().participation().condition ||
            last.activation != first.activation ||
            first.participation != input.target().participation().owner ||
            last.participation != first.participation) { return 1; }
        auto loop = cast<scf::ForOp>(owner);
        auto query = child.analysis->boundaryLoop->portQuery(false, fs::PeriodicEventKind::Start,
                                                           false, fs::PeriodicEventKind::Completion);
        if (failed(query) || !query->reachable || query->bound != loop.getUpperBound() ||
            query->activation != child.analysis->boundaryLoop->activation || query->coefficient || query->constant) {
            return 1;
        }
        const auto& body = *child.analysis->boundaryLoop;
        for (int64_t trips : {0, 1, 2, 5}) {
            for (unsigned from = 0; from < 4; ++from) {
                for (unsigned to = 0; to < 4; ++to) {
                    auto reached = body.portQuery(from >= 2,
                        from % 2 ? fs::PeriodicEventKind::Completion : fs::PeriodicEventKind::Start,
                        to >= 2, to % 2 ? fs::PeriodicEventKind::Completion : fs::PeriodicEventKind::Start);
                    if (failed(reached)) { return 1; }
                    if (!trips) { continue; }
                    const auto size = static_cast<int64_t>(body.bodies.size());
                    const auto source = from >= 2 ? trips * size - 1 : 0;
                    const auto target = to >= 2 ? trips * size - 1 : 0;
                    const bool expected = target > source ||
                        (target == source && !(from % 2 && !(to % 2)));
                    const bool actual = reached->reachable &&
                        llvm::DynamicAPInt(reached->coefficient) * llvm::DynamicAPInt(trips) +
                            llvm::DynamicAPInt(reached->constant) >= 0;
                    if (actual != expected) { return 1; }
                }
            }
        }
        if (conditional && !llvm::is_contained(child.analysis->regionalContext->entryRegions,
                &cast<scf::IfOp>(first.participation).getThenRegion())) { return 1; }
        auto parent = requests.request(function, requests.context(), fs::RegionalMode::Modeled,
                                       fs::AnalysisNeeds::modeledCovers());
        if (!parent.analysis || !parent.analysis->boundaryLoop || parent.analysis->regionalChildren.size() != 1 ||
            parent.analysis->regionalChildren.front() != child.analysis ||
            parent.analysis->boundaryLoop->loopChild != child.analysis->boundaryLoop || !requests.cacheHits()) {
            return 1;
        }
        auto strict = fs::AnalysisNeeds::minimumExact();
        strict.interfaces |= fs::interfaceBit(fs::DemandInterface::RegionBoundaryPorts);
        auto rejected = requests.request(owner, requests.contextFor(owner), fs::RegionalMode::Modeled, strict);
        if (rejected.analysis || render(function) != before) { return 1; }
    }
    needs = fs::AnalysisNeeds::modeledCovers();
    needs.interfaces |= fs::interfaceBit(fs::DemandInterface::CellBoundaries);
    auto unavailable = requests.request(function, requests.context(), fs::RegionalMode::Modeled, needs);
    if (unavailable.analysis || render(function) != before) { return 1; }
    needs = fs::AnalysisNeeds::minimumExact();
    SmallVector<fs::AnalysisAttempt> attempts;
    std::string reason;
    auto selected = fs::selectAnalysis(function, input, *trace, attempts, reason, costs, needs);
    if (selected || render(function) != before) { return 1; }
    llvm::outs() << "verified structural modeled covers, original rollback "
                    "and source/generation/physical protocol mutations\n";
    return 0;
}
