// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent coloring/matching/rearm checks and real pending-plan mutations.
#include "../../lib/PTO/Transforms/FrontierSynch/ExplicitPhysicalEmission.h"
#include "../../lib/PTO/Transforms/FrontierSynch/DirectEmissionInternal.h"
#include "PTO/Transforms/FrontierSynch/FiniteEventAssignment.h"
#include "PTO/Transforms/FrontierSynch/FrontierSynch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <chrono>
#include <map>
namespace {
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
using Graph = SmallVector<llvm::BitVector>;
void close(Graph& graph)
{
    for (std::size_t k = 0; k < graph.size(); ++k) {
        for (std::size_t a = 0; a < graph.size(); ++a) { if (graph[a].test(k)) { graph[a] |= graph[k]; } }
    }
}
bool compatible(ArrayRef<fs::FiniteHandoff> handoffs, const Graph& reach, unsigned a, unsigned b)
{
    return reach[handoffs[a].acquisition].test(handoffs[b].publication) ||
           reach[handoffs[b].acquisition].test(handoffs[a].publication);
}
unsigned independentBudget(ArrayRef<fs::FiniteHandoff> handoffs, const Graph& reach)
{
    if (handoffs.empty()) { return 0; }
    SmallVector<unsigned> colors(handoffs.size());
    for (unsigned count = 1; count <= handoffs.size(); ++count) {
        std::function<bool(unsigned)> assign = [&](unsigned node) {
            if (node == handoffs.size()) { return true; }
            for (unsigned id = 0; id < count; ++id) {
                bool valid = true;
                for (unsigned prior = 0; prior < node; ++prior) {
                    if (colors[prior] == id && !compatible(handoffs, reach, prior, node)) { valid = false; }
                }
                if (valid) { colors[node] = id; if (assign(node + 1)) { return true; } }
            }
            return false;
        };
        if (assign(0)) { return count; }
    }
    return handoffs.size();
}
bool finiteOracles()
{
    for (unsigned count : {0U, 1U, 2U, 6U, 7U}) {
        for (bool chained : {false, true}) {
            Graph reach(2 * count, llvm::BitVector(2 * count));
            SmallVector<fs::FiniteHandoff> handoffs;
            for (unsigned i = 0; i < count; ++i) {
                handoffs.push_back({2 * i, 2 * i + 1}); reach[2 * i].set(2 * i + 1);
                if (chained && i + 1 < count) { reach[2 * i + 1].set(2 * i + 2); }
            }
            close(reach);
            unsigned required = independentBudget(handoffs, reach);
            for (unsigned capacity : {0U, 1U, 6U, 7U}) {
                SmallVector<unsigned> eligible;
                for (unsigned id = 0; id < capacity; ++id) { eligible.push_back(2 * id + 1); }
                auto result = fs::assignFiniteEvents(handoffs, eligible, reach);
                if (result.required != required || result.certified != (required <= capacity)) { return false; }
                if (!result.certified) { continue; }
                for (unsigned i = 0; i < count; ++i) {
                    if (!llvm::is_contained(eligible, result.ids[i])) { return false; }
                    for (unsigned j = 0; j < i; ++j) {
                        if (result.ids[i] == result.ids[j] && !compatible(handoffs, reach, i, j)) { return false; }
                    }
                }
            }
        }
    }
    // Mixed and diamond rearm orders exercise nontrivial chain covers.
    for (const auto& order : SmallVector<SmallVector<std::pair<unsigned, unsigned>>>{
             {{0, 1}, {0, 2}, {1, 3}, {2, 3}}, {{0, 2}, {1, 2}, {2, 4}, {3, 4}}}) {
        Graph reach(10, llvm::BitVector(10));
        SmallVector<fs::FiniteHandoff> handoffs;
        for (unsigned i = 0; i < 5; ++i) {
            handoffs.push_back({2 * i, 2 * i + 1});
            reach[2 * i].set(2 * i + 1);
        }
        for (auto [a, b] : order) { reach[2 * a + 1].set(2 * b); }
        close(reach);
        unsigned required = independentBudget(handoffs, reach);
        for (unsigned capacity = 0; capacity <= 5; ++capacity) {
            SmallVector<unsigned> eligible;
            for (unsigned id = 0; id < capacity; ++id) { eligible.push_back(2 * id + 1); }
            auto result = fs::assignFiniteEvents(handoffs, eligible, reach);
            if (result.required != required || result.certified != (required <= capacity)) { return false; }
            if (!result.certified) { continue; }
            for (unsigned i = 0; i < 5; ++i) {
                for (unsigned j = 0; j < i; ++j) {
                    if (result.ids[i] == result.ids[j] && !compatible(handoffs, reach, i, j)) { return false; }
                }
            }
        }
    }
    Graph ragged{llvm::BitVector(2), llvm::BitVector(0)};
    ragged[0].set(1);
    if (fs::assignFiniteEvents({}, {0}, ragged).certified) { return false; }
    Graph bad(2, llvm::BitVector(2));
    SmallVector<fs::FiniteHandoff> unmatched{{0, 1}};
    return !fs::assignFiniteEvents(unmatched, {0}, bad).certified;
}
// Supplied native/model premises only: repeated abstract occurrences borrow
// shared phase pipes, never qualify repeated production payload completion.
bool summaryOracles(ArrayRef<fs::StructuredSite> sites)
{
    if (sites.size() < 2) { return false; }
    auto sourcePipe = sites[0].phase->kPipeValue, targetPipe = sites[1].phase->kPipeValue;
    for (unsigned count : {0U, 1U, 2U, 6U, 7U}) {
        for (unsigned distance : {1U, 2U, 8U}) {
            SmallVector<const pto::CompoundInstanceElement*> phases;
            SmallVector<fs::Demand> demands;
            for (unsigned i = 0; i < count; ++i) {
                phases.push_back(sites[0].phase); phases.push_back(sites[1].phase);
                demands.push_back({2 * i, 2 * i + 1, {}});
                if (i + distance < count) { demands.push_back({2 * i + 1, 2 * (i + distance), {}}); }
            }
            fs::RankReduction reduction;
            if (failed(reduction.build(phases, demands))) { return false; }
            struct Event { unsigned kind; pto::PipelineType pipe; std::size_t node; };
            SmallVector<Event> events;
            std::size_t nodes = 0;
            SmallVector<std::size_t> payloadStarts(phases.size());
            SmallVector<std::size_t> retained;
            retained.append(reduction.retained().begin(), reduction.retained().end());
            SmallVector<fs::FiniteHandoff> endpoints(retained.size());
            for (std::size_t site = 0; site < phases.size(); ++site) {
                for (auto [pair, id] : llvm::enumerate(retained)) {
                    if (demands[id].consumer == site) {
                        endpoints[pair].acquisition = nodes;
                        events.push_back({2, phases[site]->kPipeValue, nodes++});
                    }
                }
                payloadStarts[site] = nodes;
                events.push_back({0, phases[site]->kPipeValue, nodes});
                nodes += 2;
                for (auto [pair, id] : llvm::enumerate(retained)) {
                    if (demands[id].source == site) {
                        endpoints[pair].publication = nodes;
                        events.push_back({1, phases[site]->kPipeValue, nodes++});
                    }
                }
            }
            Graph reach(nodes, llvm::BitVector(nodes));
            for (auto [i, event] : llvm::enumerate(events)) {
                if (!event.kind) { reach[event.node].set(event.node + 1); }
                for (std::size_t j = 0; j < i; ++j) {
                    if (events[j].pipe != event.pipe) { continue; }
                    const auto& prior = events[j];
                    auto completion = prior.kind ? prior.node : prior.node + 1;
                    if (!prior.kind) { reach[prior.node].set(event.node); }
                    if (event.kind == 1) { reach[completion].set(event.node); }
                    if (prior.kind == 2) { reach[completion].set(event.node); }
                    if (!prior.kind && !event.kind) { reach[completion].set(event.node + 1); }
                }
            }
            for (auto handoff : endpoints) { reach[handoff.publication].set(handoff.acquisition); }
            close(reach);
            SmallVector<fs::OrderedHandoffSummary> rows;
            SmallVector<fs::FiniteHandoff> pool;

            for (auto [pair, id] : llvm::enumerate(retained)) {
                const auto& demand = demands[id];
                if (phases[demand.source]->kPipeValue != sourcePipe ||
                    phases[demand.consumer]->kPipeValue != targetPipe) { continue; }
                const auto& source = reduction.summaries()[demand.source];
                const auto& consumer = reduction.summaries()[demand.consumer];
                rows.push_back({source.rank, consumer.rank, source.S[consumer.pipe]});
                pool.push_back(endpoints[pair]);
            }
            auto agrees = [&](const Graph& graph) {
                for (std::size_t a = 0; a < phases.size(); ++a) {
                    const auto& source = reduction.summaries()[a];
                    for (std::size_t b = 0; b < phases.size(); ++b) {
                        bool expected = reduction.summaries()[b].S[source.pipe] >= source.rank;
                        if (graph[payloadStarts[a] + 1].test(payloadStarts[b]) != expected) { return false; }
                    }
                }
                return true;
            };
            if (!agrees(reach)) { return false; }
            if (distance == 8 && count >= 2) {
                Graph falselyGated = reach;
                falselyGated[pool[0].publication].set(payloadStarts[2]);
                close(falselyGated);
                if (agrees(falselyGated)) { return false; }
            }
            auto budget = independentBudget(pool, reach);
            if (distance == 8 && budget != count) { return false; }
            if (count && distance < count && budget != distance) { return false; }
            for (unsigned capacity : {0U, 1U, 6U, 7U}) {
                SmallVector<unsigned> eligible;
                for (unsigned id = 0; id < capacity; ++id) { eligible.push_back(2 * id + 1); }
                auto assigned = fs::assignOrderedFiniteEvents(rows, eligible);
                if (assigned.required != budget || assigned.certified != (budget <= capacity)) { return false; }
                for (std::size_t i = 0; i < pool.size(); ++i) {
                    std::size_t first = pool.size();
                    for (std::size_t j = i + 1; j < pool.size(); ++j) {
                        bool rearm = reach[pool[i].acquisition].test(pool[j].publication);
                        if (rearm != (rows[j].sourceTargetPrefix >= rows[i].targetRank)) { return false; }
                        if (rearm && first == pool.size()) { first = j; }
                    }
                    if (assigned.thresholds[i] != first) { return false; }
                    if (!assigned.certified) { continue; }
                    if (assigned.ids[i] != eligible[i % budget]) { return false; }
                    for (std::size_t j = 0; j < i; ++j) {
                        if (assigned.ids[i] == assigned.ids[j] && !compatible(pool, reach, i, j)) { return false; }
                    }
                }
            }
        }
    }
    return !fs::assignOrderedFiniteEvents({{1, 1, 0}, {1, 2, 0}}, {0}).certified &&
           !fs::assignOrderedFiniteEvents({{1, 1, 1}}, {0}).certified;
}
std::string render(Operation* op)
{
    std::string text; llvm::raw_string_ostream stream(text); op->print(stream); return text;
}
bool physicalTrace(func::FuncOp function)
{
    std::map<std::tuple<pto::PIPE, pto::PIPE, pto::EVENT>, bool> live;
    unsigned sets = 0, waits = 0;
    bool valid = true;
    function.walk([&](Operation* op) {
        if (auto set = dyn_cast<pto::SetFlagOp>(op)) {
            auto key = std::make_tuple(set.getSrcPipe().getPipe(), set.getDstPipe().getPipe(),
                                       set.getEventId().getEvent());
            if (live[key] || static_cast<unsigned>(set.getEventId().getEvent()) >= 6) { valid = false; }
            live[key] = true; ++sets;
        } else if (auto wait = dyn_cast<pto::WaitFlagOp>(op)) {
            auto key = std::make_tuple(wait.getSrcPipe().getPipe(), wait.getDstPipe().getPipe(),
                                       wait.getEventId().getEvent());
            if (!live[key]) { valid = false; }
            live[key] = false; ++waits;
        } else if (isa<pto::LogicalSetOp, pto::LogicalWaitOp>(op)) { valid = false; }
    });
    return valid && sets == 2 && waits == 2 && llvm::all_of(live, [](const auto& entry) { return !entry.second; });
}
// Independently close the exported completion-only command basis and compare
// payload prerequisites with the admitted mathematical summaries. Multi-payload
// SET-gating counterexamples are covered by summaryOracles above.
bool setObservation(func::FuncOp function, const fs::SelectedAnalysis& selected)
{
    auto certificate = function->getAttrOfType<DictionaryAttr>("pto.frontier.physical");
    if (!certificate) { return false; }
    auto nodes = certificate.getAs<IntegerAttr>("nodes");
    auto edges = certificate.getAs<ArrayAttr>("causal_edges");
    auto matching = certificate.getAs<ArrayAttr>("matching");
    if (!nodes || nodes.getInt() <= 0 || !edges || !matching || matching.empty()) { return false; }
    Graph basis(nodes.getInt(), llvm::BitVector(nodes.getInt()));
    for (auto edge : edges) {
        auto pair = dyn_cast<ArrayAttr>(edge);
        if (!pair || pair.size() != 2) { return false; }
        auto a = dyn_cast<IntegerAttr>(pair[0]), b = dyn_cast<IntegerAttr>(pair[1]);
        if (!a || !b || a.getInt() < 0 || b.getInt() < 0 ||
            a.getInt() >= nodes.getInt() || b.getInt() >= nodes.getInt()) { return false; }
        basis[a.getInt()].set(b.getInt());
    }
    auto valid = [&](Graph graph) {
        close(graph);
        for (auto attr : matching) {
            auto match = dyn_cast<DictionaryAttr>(attr);
            if (!match) { return false; }
            auto source = match.getAs<IntegerAttr>("source_site");
            auto publication = match.getAs<IntegerAttr>("publication_event");
            if (!source || !publication) { return false; }
            auto completion = 2 * source.getInt() + 1;
            auto start = publication.getInt();
            if (completion < 0 || start < 0 || completion >= nodes.getInt() || start >= nodes.getInt() ||
                graph[start].test(2 * source.getInt()) || !graph[completion].test(start)) { return false; }
        }
        for (std::size_t a = 0; a < selected.sites.size(); ++a) {
            const auto& source = selected.explicitReduction.summaries()[a];
            for (std::size_t b = 0; b < selected.sites.size(); ++b) {
                bool expected = selected.explicitReduction.summaries()[b].S[source.pipe] >= source.rank;
                if (graph[2 * selected.sites[a] + 1].test(2 * selected.sites[b]) != expected) { return false; }
            }
        }
        return true;
    };
    if (!valid(basis)) { return false; }
    auto first = cast<DictionaryAttr>(matching[0]);
    auto source = first.getAs<IntegerAttr>("source_site").getInt();
    auto publication = first.getAs<IntegerAttr>("publication_event").getInt();
    basis[publication].set(2 * source);
    return !valid(basis);
}
bool sharedReservations(MLIRContext& context)
{
    // Real translated communication macro, not a fabricated pool or event list.
    auto module = parseSourceString<ModuleOp>(R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @hidden(%dst: !pto.partition_tensor_view<128xf32>,
                    %src: !pto.partition_tensor_view<128xf32>)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %ping = pto.alloc_tile : !pto.tile_buf<vec, 1x128xf32>
    %pong = pto.alloc_tile : !pto.tile_buf<vec, 1x128xf32>
    pto.comm.tget(%dst, %src, buf(%ping, %pong) :
      !pto.partition_tensor_view<128xf32>, !pto.partition_tensor_view<128xf32>,
      !pto.tile_buf<vec, 1x128xf32>, !pto.tile_buf<vec, 1x128xf32>)
    return
  }
})mlir", &context);
    if (!module) { return false; }
    pto::SyncInput input;
    auto source = module->lookupSymbol<func::FuncOp>("hidden");
    if (failed(input.build(source)) || input.target().phases().size() != 2) { return false; }
    std::string reason;
    for (auto direction : {std::make_pair(pto::PIPE::PIPE_MTE2, pto::PIPE::PIPE_MTE3),
                           std::make_pair(pto::PIPE::PIPE_MTE3, pto::PIPE::PIPE_MTE2)}) {
        auto pool = input.target().eventPool(pto::SyncPhysicalCore::AIV,
                                            direction.first, direction.second, reason);
        if (failed(pool) || pool->eligibleIds != SmallVector<unsigned>({2, 3, 4, 5}) ||
            pool->reservedIds != SmallVector<unsigned>({6, 7, 0, 1})) { return false; }
    }
    auto independent = input.target().eventPool(pto::SyncPhysicalCore::AIV,
        pto::PIPE::PIPE_MTE2, pto::PIPE::PIPE_V, reason);
    return succeeded(independent) && independent->eligibleIds.size() == 6;
}
bool deletedSourceRejected(func::FuncOp original)
{
    auto module = original->getParentOfType<ModuleOp>();
    OwningOpRef<ModuleOp> owned(cast<ModuleOp>(module->clone()));
    auto source = owned->lookupSymbol<func::FuncOp>(original.getSymName());
    if (!source) { return false; }
    pto::SyncInput input;
    if (failed(input.build(source))) { return false; }
    auto trace = fs::TraceDemandAnalysis::build(source, input);
    if (failed(trace)) { return false; }
    fs::CostLedger costs(false);
    auto plan = fs::emitDirectDemands(source, input, *trace, costs);
    if (!plan.emitted || !plan.pending) { return false; }
    source.erase();
    // The owner stays alive. Physical rejection must precede access to any
    // analyzer/target anchor borrowed from the deleted source function.
    return failed(fs::emitExplicitPhysical(input, *trace, plan, costs));
}
bool kernel(func::FuncOp function)
{
    pto::SyncInput input;
    if (failed(input.build(function))) { return false; }
    auto trace = fs::TraceDemandAnalysis::build(function, input);
    if (failed(trace) || trace->sites().size() != 3) { return false; }
    auto started = std::chrono::steady_clock::now();
    if (!summaryOracles(trace->sites())) { return false; }
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started).count();
    llvm::errs() << "saved-summary independent command/color oracles: " << elapsed << " ms\n";
    auto before = render(function);
    fs::CostLedger costs(true);
    auto makePlan = [&]() { return fs::emitDirectDemands(function, input, *trace, costs); };
    auto good = makePlan();
    if (failed(fs::emitExplicitPhysical(input, *trace, good, costs)) || !physicalTrace(good.pending.get()) ||
        !setObservation(good.pending.get(), *good.selected)) {
        llvm::errs() << good.reason << '\n'; return false;
    }
    // A physical acquisition without its matching directed publication fails
    // the independent event-state checker, including terminal consumption.
    pto::WaitFlagOp physicalWait;
    good.pending->walk([&](pto::WaitFlagOp wait) { if (!physicalWait) { physicalWait = wait; } });
    physicalWait->setAttr("event_id", pto::EventAttr::get(function.getContext(), static_cast<pto::EVENT>(1)));
    if (physicalTrace(good.pending.get())) { return false; }
    for (unsigned mutation : {0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U}) {
        auto plan = makePlan();
        pto::LogicalSetOp set;
        pto::LogicalWaitOp wait;
        plan.pending->walk([&](pto::LogicalSetOp op) { if (!set) { set = op; } });
        plan.pending->walk([&](pto::LogicalWaitOp op) { if (!wait) { wait = op; } });
        if (!set || !wait) { return false; }
        if (mutation == 0) { wait.erase(); }
        else if (mutation == 1) {
            Operation* consumer = nullptr;
            for (auto& op : plan.pending->getBody().front()) {
                auto site = op.getAttrOfType<IntegerAttr>("pto.frontier.site");
                if (site && site.getInt() == 1) { consumer = &op; }
            }
            if (!consumer) { return false; }
            set->moveAfter(consumer);
        } else if (mutation == 2) { set.erase(); wait.erase(); }
        else if (mutation == 7) {
            Operation* payload = nullptr;
            for (auto& op : plan.pending->getBody().front()) {
                if (isa<pto::TAbsOp>(op)) { payload = &op; break; }
            }
            if (!payload || payload->getNumOperands() != 2) { return false; }
            // Same type, dominating SSA: changing a modeled input must not keep
            // the original analysis/target certificate attached to the clone.
            payload->setOperand(0, payload->getOperand(1));
        } else if (mutation == 8) {
            arith::ConstantOp address;
            plan.pending->walk([&](arith::ConstantOp op) {
                auto value = dyn_cast<IntegerAttr>(op.getValue());
                if (value && value.getInt() == 8192) { address = op; }
            });
            if (!address) { return false; }
            address.setValueAttr(IntegerAttr::get(address.getType(), 16384));
        } else if (mutation == 9) {
            pto::TStoreOp store;
            plan.pending->walk([&](pto::TStoreOp operation) { store = operation; });
            if (!store) { return false; }
            // ODS inherent property changes the destination from write-only to
            // RMW. The immutable effects/reduction cannot certify this clone.
            store.setAtomicTypeAttr(pto::AtomicTypeAttr::get(function.getContext(), pto::AtomicType::AtomicAdd));
        } else {
            auto drain = *plan.pending->getBody().front().getOps<pto::BarrierOp>().begin();
            OpBuilder builder(drain);
            auto loc = drain.getLoc();
            if (mutation == 3) {
                auto event = pto::EventAttr::get(function.getContext(), static_cast<pto::EVENT>(0));
                builder.create<pto::SetFlagOp>(loc, set.getSrcPipe(), set.getDstPipe(), event);
                builder.create<pto::WaitFlagOp>(loc, set.getSrcPipe(), set.getDstPipe(), event);
            } else if (mutation == 4) { drain.erase(); }
            else if (mutation == 5) {
                function.getContext()->getOrLoadDialect<scf::SCFDialect>();
                auto condition = builder.create<arith::ConstantIntOp>(loc, 1, 1);
                auto branch = builder.create<scf::IfOp>(loc, TypeRange{}, condition.getResult(), false);
                OpBuilder nested = OpBuilder::atBlockBegin(branch.thenBlock());
                nested.create<pto::BarrierOp>(loc, pto::PipeAttr::get(function.getContext(), pto::PIPE::PIPE_V));
            } else {
                Operation* payload = nullptr;
                for (auto& op : plan.pending->getBody().front()) {
                    if (isa<pto::OpPipeInterface>(op)) { payload = &op; break; }
                }
                if (!payload) { return false; }
                auto* injected = payload->clone();
                injected->removeAttr("pto.frontier.site");
                builder.insert(injected);
            }
        }
        if ((mutation == 3 || mutation == 5 || mutation == 6 || mutation == 7 || mutation == 8 || mutation == 9) &&
            failed(verify(plan.pending.get()))) {
            return false;
        }
        if (succeeded(fs::emitExplicitPhysical(input, *trace, plan, costs)) || render(function) != before) {
            return false;
        }
        plan.emitted = false;
        plan.pending = nullptr;
        auto report = fs::costReport(function, input, *trace, plan, false);
        fs::finishCostReport(report, costs, plan);
        auto* stages = report.getObject("stages");
        if (report.getString("analysis") != "selected-minimum" ||
            report.getString("requirements") != "shared-modeled" || !stages ||
            stages->getObject("effect_recovery")->getString("status") == "failed" ||
            stages->getObject("allocation")->getString("status") != "failed") { return false; }
    }
    // Original target context is borrowed separately from captured effects.
    // A module-only change cannot be detected by printing the function alone.
    auto contextPlan = makePlan();
    auto* owner = function->getParentOp();
    auto originalAttributes = owner->getAttrDictionary();
    owner->setAttr("pto.target_arch", StringAttr::get(function.getContext(), "a5"));
    bool rejectedContext = failed(fs::emitExplicitPhysical(input, *trace, contextPlan, costs));
    owner->setAttrs(originalAttributes);
    if (!rejectedContext || render(function) != before) { return false; }
    auto sourcePlan = makePlan();
    auto originalFunctionAttributes = function->getAttrDictionary();
    function->setAttr("b1.original_mutation", UnitAttr::get(function.getContext()));
    bool rejectedSource = failed(fs::emitExplicitPhysical(input, *trace, sourcePlan, costs));
    function->setAttrs(originalFunctionAttributes);
    return rejectedSource && render(function) == before && deletedSourceRejected(function);
}
bool oneWayKernel(func::FuncOp function)
{
    pto::SyncInput input;
    if (failed(input.build(function)) ||
        input.target().finiteDrains().size() != 7) {
        return false;
    }
    auto trace = fs::TraceDemandAnalysis::build(function, input);
    if (failed(trace) || trace->sites().size() != 16) { return false; }
    fs::CostLedger costs(false);
    auto before = render(function);
    auto makePlan = [&]() { return fs::emitDirectDemands(function, input, *trace, costs); };
    auto fixed = makePlan();
    if (!fixed.emitted || succeeded(fs::emitExplicitPhysical(input, *trace, fixed, costs)) ||
        fixed.reason.find("requires 8 IDs; available 6") == std::string::npos || render(function) != before) {
        return false;
    }
    auto good = makePlan();
    if (!good.emitted || failed(fs::emitExplicitPhysical(input, *trace, good, costs, true)) ||
        good.sets != 6 || good.waits != 6 || render(function) != before) {
        llvm::errs() << good.reason << '\n';
        return false;
    }
    for (unsigned mutation = 0; mutation < 4; ++mutation) {
        auto plan = makePlan();
        if (!plan.emitted || !plan.pending || plan.sets != 8 || plan.waits != 8) { return false; }
        auto& body = plan.pending->getBody().front();
        if (mutation == 0) { auto wait = *body.getOps<pto::LogicalWaitOp>().begin(); wait.erase(); }
        else if (mutation == 1) { auto barrier = *body.getOps<pto::BarrierOp>().begin(); barrier.erase(); }
        else if (mutation == 2) {
            auto set = *body.getOps<pto::LogicalSetOp>().begin();
            set->moveBefore(*body.getOps<pto::TLoadOp>().begin());
        } else {
            SmallVector<pto::TXorOp> payloads;
            for (auto payload : body.getOps<pto::TXorOp>()) { payloads.push_back(payload); }
            if (payloads.size() < 2) { return false; }
            payloads[0]->setOperand(0, payloads[1]->getOperand(0));
        }
        if (failed(verify(*plan.pending)) || succeeded(fs::emitExplicitPhysical(input, *trace, plan, costs, true)) ||
            render(function) != before) {
            return false;
        }
    }
    auto sourcePlan = makePlan();
    auto drain = *function.getBody().front().getOps<pto::BarrierOp>().begin();
    auto pipe = drain.getPipe();
    drain->setAttr("pipe", pto::PipeAttr::get(function.getContext(), pto::PIPE::PIPE_V));
    bool rejected = succeeded(verify(function)) &&
        failed(fs::emitExplicitPhysical(input, *trace, sourcePlan, costs, true));
    drain->setAttr("pipe", pipe);
    return rejected && render(function) == before;
}
} // namespace
int runOneWayPhysicalChecks(llvm::StringRef path, mlir::MLIRContext& context)
{
    auto module = parseSourceFile<ModuleOp>(path, &context);
    if (!module || !llvm::hasSingleElement(module->getOps<func::FuncOp>()) ||
        !oneWayKernel(*module->getOps<func::FuncOp>().begin())) {
        return 1;
    }
    llvm::outs() << "one-way production: fixed capacity refused, six-ID repair, "
                 << "four valid pending mutations and original drain mutation rejected transactionally\n";
    return 0;
}
int runFinitePhysicalChecks(llvm::StringRef path, mlir::MLIRContext& context)
{
    auto module = parseSourceFile<ModuleOp>(path, &context);
    if (!module || !finiteOracles() || !sharedReservations(context)) {
        return 1;
    }
    unsigned count = 0;
    for (auto function : module->getOps<func::FuncOp>()) { if (!kernel(function)) { return 1; } ++count; }
    if (count != 1) { return 1; }
    llvm::outs() << "verified explicit physical matching, causal budgets 0/1/6/7, "
                 << "shared hidden reservations and abstract eligible-ID exclusions, "
                 << "SET observation and thirteen mutations; shared input consumed without native admission\n";
    return 0;
}
