// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/PeriodicEventAssignment.h"
#include "../../lib/PTO/Transforms/FrontierSynch/ExplicitPhysicalEmission.h"
#include "../../lib/PTO/Transforms/FrontierSynch/DirectEmissionInternal.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <functional>
#include <memory>
#include <limits>
namespace {
using namespace mlir;
namespace fs = pto::frontiersynch;
using llvm::DynamicAPInt;
using Graph = SmallVector<llvm::BitVector>;
using Pipe = pto::PipelineType;
std::string render(func::FuncOp function)
{
    std::string text;
    llvm::raw_string_ostream stream(text);
    function->print(stream);
    return text;
}
bool productionChecks(llvm::StringRef path, MLIRContext& context)
{
    auto module = parseSourceFile<ModuleOp>(path, &context);
    if (!module) { return false; }
    for (auto function : module->getOps<func::FuncOp>()) {
        if (!function->hasAttr("test.periodic_physical")) { continue; }
        pto::SyncInput input;
        if (failed(input.build(function))) { return false; }
        auto trace = fs::TraceDemandAnalysis::build(function, input);
        if (failed(trace)) { return false; }
        fs::CostLedger costs;
        const auto original = render(function);
        auto plan = fs::emitDirectDemands(function, input, *trace, costs);
        if (!plan.emitted || plan.selected->kind != fs::SelectedAnalysis::Kind::Periodic ||
            failed(fs::emitExplicitPhysical(input, *trace, plan, costs))) { return false; }
        if (render(function) != original || plan.privateSelectors) { return false; }
        auto missing = fs::emitDirectDemands(function, input, *trace, costs);
        pto::LogicalWaitOp acquisition;
        missing.pending->walk([&](pto::LogicalWaitOp wait) { acquisition = wait; });
        if (!acquisition) { return false; }
        acquisition.erase();
        if (succeeded(fs::emitExplicitPhysical(input, *trace, missing, costs)) || render(function) != original) {
            return false;
        }
    }
    return true;
}
bool ordinalLoweringChecks(MLIRContext& context)
{
    context.getOrLoadDialect<func::FuncDialect>();
    context.getOrLoadDialect<arith::ArithDialect>();
    // Evaluate the actual production scalar IR, including noncontiguous IDs
    // and signed Index spans too wide for a signed i64 subtraction.
    fs::PeriodicEventAssignment assignment;
    assignment.status = fs::PeriodicAssignmentStatus::Certified;
    assignment.eligibleIds = {1, 3, 5};
    assignment.phases.resize(2);
    struct Sample { int64_t lower, source, step; };
    const Sample samples[] = {{0, 0, 1}, {-5, 7, 2}, {-100, 61, 7},
        {std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max(), 1}};
    for (const auto& sample : samples) {
        for (std::size_t phase = 0; phase < assignment.phases.size(); ++phase) {
            OpBuilder builder(&context);
            auto module = ModuleOp::create(builder.getUnknownLoc());
            OwningOpRef<ModuleOp> owned(module);
            builder.setInsertionPointToEnd(module.getBody());
            auto function = builder.create<func::FuncOp>(module.getLoc(), "ordinal",
                builder.getFunctionType({}, {builder.getIndexType()}));
            builder.setInsertionPointToStart(function.addEntryBlock());
            Value lower = builder.create<arith::ConstantIndexOp>(module.getLoc(), sample.lower);
            Value source = builder.create<arith::ConstantIndexOp>(module.getLoc(), sample.source);
            auto id = fs::lowerPeriodicPhysicalId(builder, module.getLoc(), source, lower, sample.step,
                                                 assignment, phase);
            builder.create<func::ReturnOp>(module.getLoc(), id);
            PassManager manager(&context);
            manager.addPass(createCanonicalizerPass());
            if (failed(manager.run(module))) { return false; }
            APInt actual;
            auto terminal = cast<func::ReturnOp>(function.getBody().front().getTerminator());
            if (!matchPattern(terminal.getOperand(0), m_ConstantInt(&actual))) { return false; }
            const auto distance = DynamicAPInt(sample.source) - DynamicAPInt(sample.lower);
            const auto ordinal = (distance / DynamicAPInt(sample.step)) * 2 + DynamicAPInt(static_cast<int64_t>(phase));
            auto slot = static_cast<int64_t>(ordinal % DynamicAPInt(3));
            if (actual.getSExtValue() != assignment.eligibleIds[slot]) { return false; }
        }
    }
    return true;
}
struct Pair {
    std::size_t source, target, phase, period;
    std::size_t publication = 0, acquisition = 0;
};
struct Event { unsigned kind; Pipe pipe; std::size_t node; };
void transitiveClosure(Graph& graph)
{
    for (std::size_t via = 0; via < graph.size(); ++via) {
        for (auto& row : graph) { if (row.test(via)) { row |= graph[via]; } }
    }
}
// Test-only unfolding follows payload/command contracts directly; no quotient
// threshold, rank summary or production allocator supplies a reference edge.
Graph commandGraph(ArrayRef<const pto::CompoundInstanceElement*> sites,
    std::size_t length, SmallVectorImpl<Pair>& pairs)
{
    SmallVector<Event> events;
    std::size_t nodes = 0;
    for (std::size_t occurrence = 0; occurrence < length; ++occurrence) {
        const auto pipe = sites[occurrence % sites.size()]->kPipeValue;
        for (auto& pair : pairs) {
            if (pair.target == occurrence) { pair.acquisition = nodes; events.push_back({2, pipe, nodes++}); }
        }
        events.push_back({0, pipe, nodes});
        nodes += 2;
        for (auto& pair : pairs) {
            if (pair.source == occurrence) { pair.publication = nodes; events.push_back({1, pipe, nodes++}); }
        }
    }
    Graph result(nodes, llvm::BitVector(nodes));
    for (auto [index, event] : llvm::enumerate(events)) {
        if (event.kind == 0) { result[event.node].set(event.node + 1); }
        for (std::size_t prior = 0; prior < index; ++prior) {
            const auto& previous = events[prior];
            if (previous.pipe != event.pipe) { continue; }
            const auto completion = previous.node + (previous.kind == 0 ? 1 : 0);
            if (previous.kind == 0) { result[previous.node].set(event.node); }
            if (event.kind == 1 || previous.kind == 2) { result[completion].set(event.node); }
            if (previous.kind == 0 && event.kind == 0) { result[completion].set(event.node + 1); }
        }
    }
    for (const auto& pair : pairs) { result[pair.publication].set(pair.acquisition); }
    transitiveClosure(result);
    return result;
}
unsigned colorBudget(ArrayRef<Pair> pairs, const Graph& graph)
{
    SmallVector<unsigned> colors(pairs.size());
    for (unsigned capacity = 0; capacity <= pairs.size(); ++capacity) {
        std::function<bool(std::size_t, unsigned)> visit;
        visit = [&](std::size_t current, unsigned used) {
            if (current == pairs.size()) { return true; }
            for (unsigned color = 0; color < std::min(capacity, used + 1); ++color) {
                bool compatible = true;
                for (std::size_t earlier = 0; earlier < current; ++earlier) {
                    if (colors[earlier] == color &&
                        !graph[pairs[earlier].acquisition].test(pairs[current].publication)) {
                        compatible = false;
                    }
                }
                if (!compatible) { continue; }
                colors[current] = color;
                if (visit(current + 1, std::max(used, color + 1))) { return true; }
            }
            return false;
        };
        if (visit(0, 0)) { return capacity; }
    }
    return pairs.size();
}
SmallVector<std::size_t> directedRecords(const fs::PeriodicDemandReduction& reduction, Pipe source)
{
    SmallVector<std::size_t> result;
    for (auto id : reduction.retained()) {
        if (reduction.sites()[reduction.generators()[id].edge.source]->kPipeValue == source) {
            result.push_back(id);
        }
    }
    return result;
}
bool finiteOracle(const fs::PeriodicDemandReduction& reduction, std::size_t length, Pipe source)
{
    const auto width = reduction.sites().size();
    SmallVector<Pair> all;
    for (std::size_t occurrence = 0; occurrence < length; ++occurrence) {
        for (const auto& record : reduction.generators()) {
            const auto& edge = record.edge;
            if (edge.source != occurrence % width || edge.distance > 16) { continue; }
            const auto target = (occurrence / width + static_cast<std::size_t>(static_cast<int64_t>(edge.distance))) *
                                width + edge.consumer;
            if (target < length) { all.push_back({occurrence, target, edge.source, occurrence / width}); }
        }
    }
    auto graph = commandGraph(reduction.sites(), length, all);
    SmallVector<Pair> pool;
    for (const auto& pair : all) {
        if (reduction.sites()[pair.phase]->kPipeValue == source) { pool.push_back(pair); }
    }
    const auto required = colorBudget(pool, graph);
    const auto records = directedRecords(reduction, source);
    for (unsigned capacity = 0; capacity <= 6; ++capacity) {
        SmallVector<unsigned> eligible;
        for (unsigned i = 0; i < capacity; ++i) { eligible.push_back(2 * i + 2); }
        const auto assigned = fs::assignPeriodicEvents(reduction, records, eligible, DynamicAPInt(length));
        if (!assigned.required || *assigned.required != DynamicAPInt(required) ||
            !assigned.handoffCount || *assigned.handoffCount != DynamicAPInt(pool.size()) ||
            (assigned.status == fs::PeriodicAssignmentStatus::Certified) != (required <= capacity)) { return false; }
        if (required > capacity) { continue; }
        SmallVector<unsigned> ids;
        for (const auto& pair : pool) {
            auto phase = llvm::find_if(assigned.phases, [&](const auto& edge) { return edge.source == pair.phase; });
            if (phase == assigned.phases.end()) { return false; }
            const auto index = static_cast<std::size_t>(phase - assigned.phases.begin());
            auto id = assigned.sourceId(index, DynamicAPInt(pair.period));
            auto acquired = assigned.consumerId(index, DynamicAPInt(pair.target / width));
            if (failed(id) || failed(acquired) || *id != *acquired || !llvm::is_contained(eligible, *id)) {
                return false;
            }
            for (std::size_t earlier = 0; earlier < ids.size(); ++earlier) {
                if (ids[earlier] == *id && !graph[pool[earlier].acquisition].test(pair.publication)) { return false; }
            }
            ids.push_back(*id);
        }
    }
    return true;
}
bool rotatingChecks(ArrayRef<const pto::CompoundInstanceElement*> sites, unsigned slots)
{
    SmallVector<fs::PeriodicPrerequisite> records;
    for (std::size_t site = 0; site < sites.size(); site += 2) {
        records.push_back({site, site + 1, DynamicAPInt(0)});
        records.push_back({site + 1, site, DynamicAPInt(slots)});
    }
    fs::PeriodicDemandReduction reduction;
    if (failed(reduction.build(sites, records)) || reduction.retained().size() != records.size()) { return false; }
    for (std::size_t length = 0; length <= 16; ++length) {
        if (!finiteOracle(reduction, length, Pipe::PIPE_MTE2) || !finiteOracle(reduction, length, Pipe::PIPE_V)) {
            return false;
        }
    }
    auto pool = directedRecords(reduction, Pipe::PIPE_MTE2);
    // Mathematical test IDs; this vector does not model target capacity.
    const auto uniform = fs::assignPeriodicEvents(reduction, pool,
        {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24});
    if (uniform.status != fs::PeriodicAssignmentStatus::Certified || !uniform.uniformRequired) { return false; }
    if (sites.size() == 2 && *uniform.uniformRequired != DynamicAPInt(slots)) { return false; }
    if (slots != 2 || sites.size() != 2) { return true; }
    const auto holes = fs::assignPeriodicEvents(reduction, pool, {2, 4});
    const auto reverse = directedRecords(reduction, Pipe::PIPE_V);
    const auto shortReverse = fs::assignPeriodicEvents(reduction, reverse, {2, 4}, DynamicAPInt(6));
    const auto longerReverse = fs::assignPeriodicEvents(reduction, reverse, {2, 4}, DynamicAPInt(8));
    if (!shortReverse.required || *shortReverse.required != 1 ||
        !longerReverse.required || *longerReverse.required != 2) { return false; }
    DynamicAPInt huge(1);
    for (unsigned bit = 0; bit < 200; ++bit) { huge *= 2; }
    auto first = holes.sourceId(0, huge), second = holes.sourceId(0, huge + 1);
    auto matched = holes.consumerId(0, huge);
    const auto released = fs::assignPeriodicEvents(reduction, reverse, {2, 4});
    auto reverseSource = released.sourceId(0, huge), reverseConsumer = released.consumerId(0, huge + 2);
    return succeeded(first) && succeeded(second) && succeeded(matched) &&
           *first == 2 && *second == 4 && *matched == *first && failed(holes.sourceId(0, DynamicAPInt(-1))) &&
           succeeded(reverseSource) && succeeded(reverseConsumer) && *reverseSource == *reverseConsumer &&
           failed(released.consumerId(0, DynamicAPInt(1)));
}
bool encodedDistanceChecks(ArrayRef<const pto::CompoundInstanceElement*> sites)
{
    DynamicAPInt distance(1);
    for (unsigned bit = 0; bit < 100; ++bit) { distance *= 2; }
    fs::PeriodicDemandReduction reduction;
    if (failed(reduction.build(sites.take_front(2), {{0, 1, DynamicAPInt(0)}, {1, 0, distance}}))) {
        return false;
    }
    const auto ready = fs::assignPeriodicEvents(reduction, directedRecords(reduction, Pipe::PIPE_MTE2), {2, 4});
    const auto release = fs::assignPeriodicEvents(reduction, directedRecords(reduction, Pipe::PIPE_V), {},
        DynamicAPInt(16));
    return ready.status == fs::PeriodicAssignmentStatus::Counterexample && ready.uniformRequired &&
           *ready.uniformRequired == distance && release.status == fs::PeriodicAssignmentStatus::Certified &&
           release.required && *release.required == 0 && reduction.vertexCount() == 4;
}
bool boundaryChecks(ArrayRef<const pto::CompoundInstanceElement*> sites)
{
    fs::PeriodicDemandReduction oneWay;
    if (failed(oneWay.build(sites.take_front(2), {{0, 1, DynamicAPInt(0)}}))) { return false; }
    const auto uniform = fs::assignPeriodicEvents(oneWay, {0}, {2, 4});
    if (uniform.status != fs::PeriodicAssignmentStatus::Counterexample || uniform.uniformRequired) { return false; }
    for (std::size_t length = 0; length <= 16; ++length) {
        if (!finiteOracle(oneWay, length, Pipe::PIPE_MTE2)) { return false; }
    }
    if (fs::assignPeriodicEvents(oneWay, {0}, {2, 2}).status != fs::PeriodicAssignmentStatus::Unsupported ||
        fs::assignPeriodicEvents(oneWay, {0, 0}, {2}).status != fs::PeriodicAssignmentStatus::Unsupported ||
        fs::assignPeriodicEvents(oneWay, {1}, {2}).status != fs::PeriodicAssignmentStatus::Unsupported ||
        fs::assignPeriodicEvents(oneWay, {0}, {2}, DynamicAPInt(-1)).status !=
            fs::PeriodicAssignmentStatus::Unsupported) { return false; }
    fs::PeriodicDemandReduction multiple;
    if (failed(multiple.build(sites, {{0, 1, DynamicAPInt(0)}, {2, 3, DynamicAPInt(0)},
                                      {0, 3, DynamicAPInt(0)}}))) { return false; }
    if (fs::assignPeriodicEvents(multiple, {0}, {2}).status != fs::PeriodicAssignmentStatus::Unsupported) {
        return false;
    }
    for (auto id : llvm::seq<std::size_t>(0, multiple.generators().size())) {
        if (!llvm::is_contained(multiple.retained(), id) &&
            fs::assignPeriodicEvents(multiple, {id}, {2}).status != fs::PeriodicAssignmentStatus::Unsupported) {
            return false;
        }
    }
    fs::PeriodicDemandReduction local;
    if (failed(local.build(sites, {{0, 2, DynamicAPInt(0)}}))) { return false; }
    return fs::assignPeriodicEvents(local, {0}, {2}).status == fs::PeriodicAssignmentStatus::Unsupported &&
           fs::assignPeriodicEvents(oneWay, {}, {}, DynamicAPInt(0)).status == fs::PeriodicAssignmentStatus::Certified;
}
} // namespace
int runPeriodicAllocationChecks(llvm::StringRef path, mlir::MLIRContext& context)
{
    if (!productionChecks(path, context)) {
        llvm::errs() << "periodic physical production/rollback check failed\n";
        return 1;
    }
    if (!ordinalLoweringChecks(context)) {
        llvm::errs() << "periodic physical ordinal lowering oracle failed\n";
        return 1;
    }
    SmallVector<std::unique_ptr<pto::CompoundInstanceElement>> owned;
    SmallVector<const pto::CompoundInstanceElement*> sites;
    for (unsigned site = 0; site < 4; ++site) {
        owned.push_back(std::make_unique<pto::CompoundInstanceElement>(site,
            SmallVector<const pto::BaseMemInfo*>{}, SmallVector<const pto::BaseMemInfo*>{},
            site % 2 ? Pipe::PIPE_V : Pipe::PIPE_MTE2, OperationName("test.phase", &context)));
        sites.push_back(owned.back().get());
    }
    for (unsigned slots : {1U, 2U, 3U, 5U}) {
        if (!rotatingChecks(ArrayRef<const pto::CompoundInstanceElement*>(sites).take_front(2), slots) ||
            !rotatingChecks(sites, slots)) { llvm::errs() << "periodic allocation rotating oracle failed\n"; return 1; }
    }
    if (!boundaryChecks(sites) || !encodedDistanceChecks(sites)) {
        llvm::errs() << "periodic allocation boundary oracle failed\n";
        return 1;
    }
    llvm::outs() << "periodic allocation independent command/rearm/color oracles passed\n";
    return 0;
}
