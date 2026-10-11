// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Validate regional adapter boundaries independently of backend algorithms.
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/RegionalCapabilitiesInternal.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/RegionalRelationsInternal.h"
#include "PTO/Transforms/FrontierSynch/Shared/RegionalNumericalInterface.h"
#include "PTO/Transforms/FrontierSynch/Shared/RegionalRelations.h"
#include "PTO/Transforms/FrontierSynch/Shared/LogicalInsertion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/ScopeExit.h"
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
bool contexts()
{
    pto::SyncInput input, otherInput;
    fs::RegionalAnalysis region;
    region.expressions = std::make_shared<fs::RegionExpressions>();
    region.accessModel = &input.accesses();
    region.gmAliasPolicy = input.memory().gmPolicy();
    auto expected = fs::regionalContext(region);
    if (!fs::matchesRegionalContext(region, expected)) {
        return false;
    }
    auto otherArena = std::make_shared<fs::RegionExpressions>();
    // Equal numeric IDs cannot establish provenance; context identity can.
    if (region.expressions->constant(0) != otherArena->constant(0)) {
        return false;
    }
    expected.expressions = otherArena.get();
    if (fs::matchesRegionalContext(region, expected)) {
        return false;
    }
    expected = fs::regionalContext(region);
    expected.accessModel = &otherInput.accesses();
    if (fs::matchesRegionalContext(region, expected)) {
        return false;
    }
    expected = fs::regionalContext(region);
    expected.gmAliasPolicy = region.gmAliasPolicy == pto::GMAliasPolicy::MayAlias ? pto::GMAliasPolicy::MayNotAlias :
                                                                                    pto::GMAliasPolicy::MayAlias;
    return !fs::matchesRegionalContext(region, expected);
}
bool predicates()
{
    fs::RegionalAnalysis region;
    region.expressions = std::make_shared<fs::RegionExpressions>();
    auto& arena = *region.expressions;
    const auto zero = arena.constant(0), truth = arena.boolean(true);
    const fs::RegionalEvent event{42, zero, fs::PeriodicEventKind::Start};
    // Flat query-only producers intentionally own their type directory.
    region.presence = [truth](auto) { return std::optional(truth); };
    region.reachability = [truth](auto, auto) { return std::optional(truth); };
    region.referenceBefore = region.reachability;
    if (!fs::validRegionalOccurrenceSchema(region) || fs::regionalPresence(region, event) != truth ||
        fs::regionalReachability(region, event, event) != truth) {
        return false;
    }
    for (std::optional<fs::RegionExpressions::Id> result :
         {std::optional<fs::RegionExpressions::Id>{}, std::optional(zero),
          std::optional(fs::RegionExpressions::invalid)}) {
        region.presence = [result](auto) { return result; };
        region.reachability = [result](auto, auto) { return result; };
        region.referenceBefore = region.reachability;
        if (fs::regionalPresence(region, event) || fs::regionalReachability(region, event, event) ||
            fs::regionalReferenceBefore(region, event, event)) {
            return false;
        }
    }
    return true;
}
bool frames(MLIRContext& context)
{
    context.getOrLoadDialect<func::FuncDialect>();
    context.getOrLoadDialect<arith::ArithDialect>();
    context.getOrLoadDialect<scf::SCFDialect>();
    auto function = func::FuncOp::create(
        UnknownLoc::get(&context), "frames", FunctionType::get(&context, {IndexType::get(&context)}, {}));
    auto* body = function.addEntryBlock();
    auto cleanup = llvm::make_scope_exit([&] { function.erase(); });
    OpBuilder builder(body, body->begin());
    auto zeroOp = builder.create<arith::ConstantIndexOp>(function.getLoc(), 0);
    auto oneOp = builder.create<arith::ConstantIndexOp>(function.getLoc(), 1);
    auto loop = builder.create<scf::ForOp>(function.getLoc(), zeroOp, function.getArgument(0), oneOp);
    fs::RegionalAnalysis region;
    region.expressions = std::make_shared<fs::RegionExpressions>();
    const auto zero = region.expressions->constant(0), truth = region.expressions->boolean(true);
    pto::CompoundInstanceElement v(0, {}, {}, pto::PipelineType::PIPE_V, zeroOp->getName());
    pto::CompoundInstanceElement s(1, {}, {}, pto::PipelineType::PIPE_S, zeroOp->getName());
    v.elementOp = s.elementOp = zeroOp;
    region.anchors.resize(2);
    region.anchors[0].phase = &v;
    region.anchors[1].phase = &s;
    region.occurrenceLoops = {{}, {}};
    region.outerLoops = {{loop}, {loop}};
    region.outerDivisors = {{1}, {2}};
    unsigned calls = 0;
    region.presence = [&](auto) {
        ++calls;
        return std::optional(truth);
    };
    fs::RegionalEvent event{0, zero, fs::PeriodicEventKind::Start, {zero}};
    if (!fs::validRegionalOccurrenceSchema(region) || fs::regionalPresence(region, event) != truth) {
        return false;
    }
    const auto initialCalls = calls;
    auto malformed = event;
    malformed.kind = static_cast<fs::PeriodicEventKind>(99);
    if (fs::regionalPresence(region, malformed)) {
        return false;
    }
    malformed = event;
    malformed.ordinal = truth;
    if (fs::regionalPresence(region, malformed)) {
        return false;
    }
    malformed = event;
    malformed.visits = {fs::RegionExpressions::invalid};
    if (fs::regionalPresence(region, malformed)) {
        return false;
    }
    malformed = event;
    malformed.type = 2;
    if (fs::regionalPresence(region, malformed) || calls != initialCalls) {
        return false;
    }
    auto invalid = region;
    invalid.outerLoops.pop_back();
    if (fs::validRegionalOccurrenceSchema(invalid)) {
        return false;
    }
    invalid = region;
    invalid.outerDivisors[0].clear();
    if (fs::validRegionalOccurrenceSchema(invalid) || fs::regionalPresence(invalid, event)) {
        return false;
    }
    invalid = region;
    invalid.outerDivisors[0][0] = 0;
    if (fs::validRegionalOccurrenceSchema(invalid) || fs::regionalPresence(invalid, event)) {
        return false;
    }
    invalid = region;
    invalid.outerLoops[0][0] = {};
    return !fs::validRegionalOccurrenceSchema(invalid) && calls == initialCalls;
}
bool adapterPreflight(MLIRContext& context)
{
    auto function = func::FuncOp::create(UnknownLoc::get(&context), "preflight", FunctionType::get(&context, {}, {}));
    auto* body = function.addEntryBlock();
    auto cleanup = llvm::make_scope_exit([&] { function.erase(); });
    OpBuilder builder(body, body->begin());
    auto op = builder.create<arith::ConstantIndexOp>(function.getLoc(), 0);
    pto::CompoundInstanceElement phase(0, {}, {}, pto::PipelineType::PIPE_V, op->getName());
    phase.elementOp = op;
    pto::SyncInput input;
    fs::PhaseIndex index;
    if (failed(index.build(function, {&phase}))) {
        return false;
    }
    fs::RegionalAnalysis region;
    region.expressions = std::make_shared<fs::RegionExpressions>();
    auto arena = region.expressions;
    const auto truth = arena->boolean(true);
    region.accessModel = &input.accesses();
    region.gmAliasPolicy = input.memory().gmPolicy();
    region.capabilities = {true, true, true, false};
    region.anchors.push_back({&phase, {}, {body, op}, {body, op->getNextNode()}});
    region.occurrenceLoops = {{}};
    unsigned calls = 0;
    region.presence = [&](auto) {
        ++calls;
        return std::optional(truth);
    };
    region.referenceBefore = [&](auto, auto) {
        ++calls;
        return std::optional(arena->boolean(false));
    };
    region.reachability = [&](fs::RegionalEvent a, fs::RegionalEvent b) {
        ++calls;
        return std::optional(arena->boolean(a.kind == b.kind || a.kind == fs::PeriodicEventKind::Start));
    };
    const auto nodes = arena->size();
    std::string error;
    auto invalid = region;
    invalid.expressions.reset();
    if (succeeded(fs::requestCallbackRegionalRelations(invalid, function, input, index, error))) {
        return false;
    }
    invalid = region;
    invalid.outerLoops = {{}};
    invalid.outerDivisors = {{1}};
    if (succeeded(fs::requestCallbackRegionalRelations(invalid, function, input, index, error))) {
        return false;
    }
    invalid = region;
    invalid.occurrenceLoops.clear();
    if (succeeded(fs::requestCallbackRegionalRelations(invalid, function, input, index, error)) || calls != 0 ||
        arena->size() != nodes) {
        return false;
    }
    // A failed stronger request must not poison the unchanged result's retry.
    auto request = fs::requestCallbackRegionalRelations(region, function, input, index, error);
    if (failed(request) || calls == 0) {
        return false;
    }
    auto data = (*request)->lower({}, error);
    return succeeded(data) && data->sites.size() == 1 && data->occurrences.size() == 1;
}
bool invalidation()
{
    fs::RegionalAnalysis region;
    region.expressions = std::make_shared<fs::RegionExpressions>();
    const auto truth = region.expressions->boolean(true);
    region.numerical = std::make_shared<fs::RegionalNumericalInterface>();
    region.arithmeticRelations = std::make_shared<fs::ArithmeticRegionalRelations>();
    region.relations = std::make_shared<fs::RegionalRelations>();
    region.symbolicStorage = std::make_shared<fs::RegionalSymbolicStorageCertificate>();
    region.anchors.resize(2);
    region.occurrenceLoops.resize(2);
    region.symbolicStorageEffects = {3, 5};
    region.accessBoundary.resize(2);
    region.deferredAccessBoundary.resize(1);
    const auto first = region.expressions->constant(7);
    region.firstOrdinal = first;
    region.capabilities = {true, true, true, true};
    region.presence = [truth](auto) { return std::optional(truth); };
    region.reachability = [truth](auto, auto) { return std::optional(truth); };
    region.referenceBefore = region.reachability;
    region.endpointEventGuard = region.presence;
    region.storageSelectors = [](auto) { return std::optional(fs::RegionalStorageSelectors{}); };
    auto owner = std::make_shared<int>(17);
    region.prepare = [owner]() -> FailureOr<std::unique_ptr<fs::PreparedLogicalPlan>> {
        return std::make_unique<fs::PreparedLogicalPlan>(*owner);
    };
    const auto retained = std::make_shared<const fs::RegionalAnalysis>(region);
    fs::invalidateDerivedRegionalCertificates(region);
    if (region.numerical || region.arithmeticRelations || region.relations || region.symbolicStorage ||
        !retained->numerical || !retained->arithmeticRelations || !retained->relations || !retained->symbolicStorage) {
        return false;
    }
    const fs::RegionalEvent event{0, region.expressions->constant(0), fs::PeriodicEventKind::Start};
    return region.anchors.size() == 2 && region.occurrenceLoops.size() == 2 &&
           region.symbolicStorageEffects == std::vector<std::size_t>{3, 5} && region.accessBoundary.size() == 2 &&
           region.deferredAccessBoundary.size() == 1 && region.firstOrdinal == first &&
           region.capabilities.endpointRecipes && region.presence(event) == truth &&
           region.reachability(event, event) == truth && region.referenceBefore(event, event) == truth &&
           region.endpointEventGuard(event) == truth &&
           region.storageSelectors({pto::AddressSpace::VEC, {}, event.ordinal}) && succeeded(region.prepare());
}
} // namespace
bool runRegionalCapabilityChecks(MLIRContext& context)
{
    return contexts() && predicates() && frames(context) && adapterPreflight(context) && invalidation();
}
