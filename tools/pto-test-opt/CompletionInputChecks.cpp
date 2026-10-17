// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent overlap and scalar-first-use checks for the completion adapter.
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/CompletionStorage.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/CompletionReadiness.h"
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/FrontierSynch/Shared/CompletionRegion.h"
#include "PTO/Transforms/FrontierSynch/Shared/LogicalInsertion.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/CompletionInlineChild.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>
#include <random>
using namespace mlir;
namespace fs = pto::frontiersynch;
namespace {
bool intersects(const fs::CompletionFacts& a, const fs::CompletionFacts& b)
{
    for (std::size_t i = 0; i < a.size(); ++i) { if (a[i] && b[i]) { return true; } }
    return false;
}
bool storageChecks(MLIRContext* context)
{
    using Memory = pto::BaseMemInfo;
    using Space = pto::AddressSpace;
    std::mt19937 random(5702);
    OperationName name("pto.texpands", context);
    for (unsigned trial = 0; trial < 1000; ++trial) {
        std::vector<std::unique_ptr<Memory>> memory;
        std::vector<std::unique_ptr<pto::CompoundInstanceElement>> owners;
        SmallVector<const pto::CompoundInstanceElement*> phases;
        for (unsigned i = 0; i < 20; ++i) {
            memory.push_back(std::make_unique<Memory>(Value{}, Value{}, i % 2 ? Space::VEC : Space::MAT,
                SmallVector<uint64_t>{random() % 128, random() % 128}, 1 + random() % 8, true));
            owners.push_back(std::make_unique<pto::CompoundInstanceElement>(i,
                SmallVector<const Memory*>{memory.back().get()}, SmallVector<const Memory*>{},
                pto::PipelineType::PIPE_V, name));
            phases.push_back(owners.back().get());
        }
        fs::CompletionStorage storage(phases);
        for (unsigned a = 0; a < 20; ++a) {
            for (unsigned b = 0; b < 20; ++b) {
                bool overlap = false;
                for (auto x : memory[a]->baseAddresses) {
                    for (auto y : memory[b]->baseAddresses) {
                        overlap |= memory[a]->scope == memory[b]->scope &&
                            x < y + memory[b]->allocateSize && y < x + memory[a]->allocateSize;
                    }
                }
                const bool joined = intersects(storage.effects({phases[a]}), storage.effects({phases[b]}));
                if ((overlap && !joined) || (memory[a]->scope != memory[b]->scope && joined)) { return false; }
            }
        }
    }
    Memory a({}, {}, Space::VEC, {0}, 16, true), b({}, {}, Space::VEC, {32}, 16, true);
    Memory adjacent({}, {}, Space::VEC, {16}, 16, true);
    Memory unknown({}, {}, Space::VEC, {}, 0, false);
    Memory lost({}, {}, Space::VEC, {0}, 16, true, true);
    Memory overflow({}, {}, Space::VEC, {std::numeric_limits<uint64_t>::max()}, 16, true);
    Memory zero({}, {}, Space::VEC, {0}, 0, true);
    Memory other({}, {}, Space::MAT, {0}, 16, true);
    Memory any({}, {}, Space::Zero, {}, 0, false);
    std::vector<std::unique_ptr<pto::CompoundInstanceElement>> owners;
    SmallVector<const pto::CompoundInstanceElement*> phases;
    for (const Memory* item : {&a, &b, &unknown, &lost, &overflow, &zero, &other, &any}) {
        owners.push_back(std::make_unique<pto::CompoundInstanceElement>(owners.size(),
            SmallVector<const Memory*>{item}, SmallVector<const Memory*>{}, pto::PipelineType::PIPE_V, name));
        phases.push_back(owners.back().get());
    }
    fs::CompletionStorage storage(phases);
    const auto left = storage.effects({phases[0]}), right = storage.effects({phases[1]});
    if (intersects(left, right) || storage.size() != 3) { return false; }
    for (unsigned i = 2; i < 6; ++i) {
        auto facts = storage.effects({phases[i]});
        if (!intersects(left, facts) || !intersects(right, facts) ||
            intersects(storage.effects({phases[6]}), facts)) { return false; }
    }
    auto all = storage.effects({phases[7]});
    if (!llvm::all_of(all, [](uint8_t mode) { return mode == 2; })) { return false; }
    owners.push_back(std::make_unique<pto::CompoundInstanceElement>(8,
        SmallVector<const Memory*>{&adjacent}, SmallVector<const Memory*>{}, pto::PipelineType::PIPE_V, name));
    phases.push_back(owners.back().get());
    fs::CompletionStorage connected(phases);
    if (connected.size() != 2 || !intersects(connected.effects({phases[0]}),
                                           connected.effects({phases[1]}))) { return false; }
    fs::CompletionStorage empty({});
    auto lifted = empty.liftCoarse({{3}, {1}, {2}, false});
    if (empty.size() != 1 || lifted.all != fs::CompletionFacts{3} || lifted.exposed != fs::CompletionFacts{1} ||
        lifted.suffix != fs::CompletionFacts{2} || lifted.transparent) { return false; }
    auto collapsed = fs::collapseCompletionSummary({{1, 2}, {0, 1}, {2, 0}, true});
    return collapsed.all == fs::CompletionFacts{3} && collapsed.exposed == fs::CompletionFacts{1} &&
        collapsed.suffix == fs::CompletionFacts{2} && collapsed.transparent;
}
LogicalResult readinessChecks(func::FuncOp function)
{
    auto input = std::make_shared<pto::SyncInput>();
    if (failed(input->build(function, pto::SyncInstructionView::PipeEnvelopes))) { return failure(); }
    unsigned asynchronous = 0;
    for (auto& record : input->ir()) {
        auto* phase = dyn_cast<pto::CompoundInstanceElement>(record.get());
        if (!phase || !phase->elementOp->hasAttr("test.async")) { continue; }
        // A test-owned imported phase contract; tgetval's real PIPE_S semantics
        // are not changed in production. Exercise the complete readiness adapter.
        phase->kPipeValue = pto::PipelineType::PIPE_V;
        ++asynchronous;
    }
    auto index = std::make_shared<fs::PhaseIndex>();
    if (!asynchronous || failed(index->build(function, *input))) { return failure(); }
    SmallVector<Operation*> roots;
    for (auto& op : function.front().without_terminator()) {
        if (!op.hasAttr("test.external")) { roots.push_back(&op); }
    }
    auto result = fs::CompletionReadiness::qualify(roots, *index);
    if (function->hasAttr("test.readiness_refuse")) {
        if (succeeded(result)) { return function.emitError("unsupported readiness was accepted"); }
    } else {
        if (failed(result) || result->size() != 1) { return function.emitError("first-use readiness missing"); }
        for (auto* root : roots) {
            if (result->protects(root) != root->hasAttr("test.first_use")) {
                return function.emitError("completion is after the actual scalar use");
            }
        }
    }
    if (succeeded(result)) {
        std::string error;
        auto source = fs::CompletionRegion::captureSource(function, input, *index, error);
        auto owner = fs::CompletionRegion::buildWithChildren(function, input, *index, {}, error, source);
        if (!owner || owner->constructionWork().readinessCuts != 1) {
            return function.emitError("production readiness graph failed: ") << error;
        }
        Operation* first = nullptr;
        for (auto* root : roots) { if (root->hasAttr("test.first_use")) { first = root; } }
        // An otherwise empty exact child containing scalar arithmetic must not
        // hide the parent's mandatory completion cut.
        auto finiteSource = fs::finiteFunctionInput(function, input, index);
        auto slice = fs::finiteRegionInput(finiteSource, {first});
        auto child = fs::CompletionInlineChild::qualify(fs::analyzeFinite(slice, fs::FiniteStrategy::Concrete));
        if (!child || fs::CompletionRegion::buildWithChildren(function, input, *index, {child}, error, source)) {
            return function.emitError("atomic child hid scalar readiness");
        }
        fs::PreparedLogicalPlan plan(0);
        plan.completionOwner = owner; plan.completeInvocation = true;
        if (failed(fs::insertLogicalSynchronization(function, plan))) { return failure(); }
        bool protectedUse = false;
        function.walk([&](Operation* op) {
            if (!op->hasAttr("test.first_use")) { return; }
            auto barrier = dyn_cast_or_null<pto::BarrierOp>(op->getPrevNode());
            protectedUse = barrier && barrier.getPipe().getPipe() == pto::PIPE::PIPE_ALL;
        });
        if (!protectedUse) { return function.emitError("first scalar helper executed before completion"); }
    }
    llvm::outs() << "completion readiness: " << function.getSymName() << " checked\n";
    return success();
}
} // namespace
LogicalResult runCompletionInputChecks(func::FuncOp function)
{
    if (function->hasAttr("test.storage_classes")) {
        if (!storageChecks(function.getContext())) { return function.emitError("storage class oracle failed"); }
        llvm::outs() << "completion storage: 400000 overlap checks and wildcard/call contracts passed\n";
        return success();
    }
    return readinessChecks(function);
}
