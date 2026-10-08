// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/RepeatedPhases.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/OwningOpRef.h"
#include <limits>
#include <set>
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
using Matrix = std::vector<std::vector<bool>>;
void close(Matrix& graph)
{
    for (std::size_t k = 0; k < graph.size(); ++k) {
        for (std::size_t i = 0; i < graph.size(); ++i) {
            for (std::size_t j = 0; j < graph.size(); ++j) {
                graph[i][j] = graph[i][j] || (graph[i][k] && graph[k][j]);
            }
        }
    }
}
unsigned mask(unsigned phase, unsigned profile)
{
    if (profile == 1) { return phase % 2 ? 2 : 1; }
    if (profile == 2) { return phase == 0 ? 0 : 3; }
    return profile == 3 ? 0 : 3;
}
bool selected(fs::RegionExpressions& e, const std::vector<fs::RegionalSelector>& values,
              unsigned q, std::optional<unsigned> expected,
              fs::RegionExpressions::Substitution* bindings = nullptr)
{
    std::optional<unsigned> found;
    for (const auto& value : values) {
        auto enabled = e.constantValue(bindings ? e.substitute(value.present, *bindings) : value.present);
        if (!enabled || *enabled > 1) { return false; }
        if (!*enabled) { continue; }
        if (found || value.event.visits.size() != 1) { return false; }
        auto coordinate = value.event.visits.front();
        auto n = e.constantValue(bindings ? e.substitute(coordinate, *bindings) : coordinate);
        if (!n) { return false; }
        found = 2 * (*n * q + value.event.type / 2) + value.event.type % 2;
    }
    return found == expected;
}
bool check(func::FuncOp function, Operation* anchor, scf::ForOp loop,
           unsigned q, unsigned trips, unsigned begin, unsigned profile, uint64_t& checked,
           bool singleton = false, bool symbolicTail = false, bool runtimeCoordinates = false)
{
    auto arena = std::make_shared<fs::RegionExpressions>();
    auto& e = *arena;
    auto zero = e.constant(0);
    std::vector<pto::CompoundInstanceElement> payloads;
    for (auto pipe : {pto::PipelineType::PIPE_MTE1, pto::PipelineType::PIPE_M}) {
        payloads.emplace_back(payloads.size(), SmallVector<const pto::BaseMemInfo*>{},
            SmallVector<const pto::BaseMemInfo*>{}, pipe, anchor->getName());
        payloads.back().elementOp = anchor;
    }
    std::vector<fs::RegionalAnalysis> phases;
    for (unsigned j = 0; j < q; ++j) {
        unsigned active = mask(j, profile);
        fs::RegionalAnalysis view;
        view.expressions = arena; view.capabilities = {true, true, true, false};
        for (unsigned type = 0; type < 2; ++type) {
            view.anchors.push_back({&payloads[type], {}, {anchor->getBlock(), anchor},
                {anchor->getBlock(), anchor->getNextNode()}});
            view.occurrenceLoops.push_back({});
            fs::RegionalSelector selector{{type, zero, fs::PeriodicEventKind::Start}, e.boolean(active & (1U << type))};
            auto pipe = static_cast<uint32_t>(payloads[type].kPipeValue);
            view.firstPayloads[pipe].push_back(selector); view.lastPayloads[pipe].push_back(selector);
            view.firstSitePayloads[type].push_back(selector);
        }
        view.presence = [arena, active, zero](fs::RegionalEvent event) -> std::optional<fs::RegionExpressions::Id> {
            if (event.type >= 2 || !event.visits.empty()) { return std::nullopt; }
            return arena->land(arena->boolean(active & (1U << event.type)), arena->eq(event.ordinal, zero));
        };
        view.reachability = [arena, present = view.presence](fs::RegionalEvent a, fs::RegionalEvent b)
            -> std::optional<fs::RegionExpressions::Id> {
            auto pa = present(a), pb = present(b);
            if (!pa || !pb) { return std::nullopt; }
            bool edge = a.type < b.type || (a.type == b.type &&
                (a.kind == fs::PeriodicEventKind::Start || b.kind == fs::PeriodicEventKind::Completion));
            return arena->land(arena->land(*pa, *pb), arena->boolean(edge));
        };
        fs::RegionalStorageBoundary cell;
        cell.cell = {pto::AddressSpace::LEFT, 8 * (j % 2), 8 * (j % 2 + 1)};
        cell.firstWriters.push_back({{0, zero, fs::PeriodicEventKind::Start}, e.boolean(active & 1)});
        cell.lastWriters = cell.firstWriters;
        auto readerPipe = static_cast<uint32_t>(payloads[1].kPipeValue);
        cell.firstReaders[readerPipe].push_back({{1, zero, fs::PeriodicEventKind::Start}, e.boolean(active == 2)});
        cell.lastReaders[readerPipe].push_back({{1, zero, fs::PeriodicEventKind::Start}, e.boolean(active & 2)});
        view.storageBoundary.push_back(cell);
        phases.push_back(std::move(view));
    }
    auto endExpression = symbolicTail ? e.input(loop.getInductionVar()) : e.constant(trips);
    auto beginExpression = symbolicTail ? e.select(e.lt(endExpression, e.constant(1)), zero,
        e.sub(endExpression, e.constant(1))) : e.constant(begin);
    fs::RegionExpressions::Substitution bindings({{endExpression, e.constant(trips)}});
    auto repeated = fs::repeatPhasedRegions(function, loop, std::move(phases), endExpression, {}, beginExpression,
        singleton ? std::optional<uint64_t>(1) : std::nullopt);
    if (!repeated.error.empty()) { llvm::errs() << repeated.error << "\n"; return false; }
    if (runtimeCoordinates) {
        if (!repeated.regional.endpointEventGuard) { return false; }
        for (int64_t lower : {int64_t(-9), int64_t(5), std::numeric_limits<int64_t>::min()}) {
            for (unsigned visit = 0; visit < 6; ++visit) {
                fs::RegionExpressions::Substitution coordinates({
                    {e.input(loop.getLowerBound()), e.constant(static_cast<uint64_t>(lower))},
                    {e.input(loop.getInductionVar()), e.constant(static_cast<uint64_t>(lower + 3 * visit))}});
                for (unsigned type = 0; type < 2 * q; ++type) {
                    const auto guard = repeated.regional.endpointEventGuard(
                        {type, zero, fs::PeriodicEventKind::Start, {e.constant(visit / q)}});
                    if (!guard || e.constantValue(e.substitute(*guard, coordinates)) != (visit % q == type / 2)) {
                        return false;
                    }
                }
            }
        }
    }
    if (repeated.regional.capabilities.endpointRecipes || repeated.regional.prepare ||
        repeated.regional.prepareWithVisits) { return false; }
    auto active = [&](unsigned occurrence) {
        return occurrence / 2 >= begin && (mask((occurrence / 2) % q, profile) & (1U << (occurrence % 2)));
    };
    const unsigned count = 2 * trips;
    Matrix graph(2 * count, std::vector<bool>(2 * count));
    for (unsigned i = 0; i < count; ++i) {
        if (!active(i)) { continue; }
        graph[2*i][2*i] = graph[2*i+1][2*i+1] = graph[2*i][2*i+1] = true;
        for (unsigned j = i+1; j < count; ++j) {
            if (!active(j)) { continue; }
            if (i%2 == j%2) { graph[2*i][2*j] = graph[2*i+1][2*j+1] = true; }
            if (((i/2)%q)%2 == ((j/2)%q)%2 && (i%2 == 0 || j%2 == 0)) { graph[2*i+1][2*j] = true; }
        }
    }
    close(graph);
    auto event = [&](unsigned vertex) {
        unsigned occurrence = vertex/2, visit = occurrence/2;
        return fs::RegionalEvent{2*(visit%q)+occurrence%2, zero, vertex%2 ? fs::PeriodicEventKind::Completion :
            fs::PeriodicEventKind::Start, {e.constant(visit/q)}};
    };
    for (unsigned i = 0; i < 2*count; ++i) {
        for (unsigned j = 0; j < 2*count; ++j) {
            auto value = fs::regionalReachability(repeated.regional, event(i), event(j));
            if (!value || e.constantValue(e.substitute(*value, bindings)) != uint64_t(graph[i][j])) {
                llvm::errs() << "phase mismatch q=" << q << " T="
                             << trips << " begin=" << begin << " profile=" << profile << " " << i << "," << j << "\n";
                return false;
            }
            ++checked;
        }
    }
    for (const auto& cell : repeated.regional.storageBoundary) {
        std::optional<unsigned> firstWriter, lastWriter, firstReader, lastReader;
        for (unsigned i = 0; i < count; ++i) {
            if (!active(i) || 8*(((i/2)%q)%2) != cell.cell.begin) { continue; }
            if (i%2 == 0) {
                if (!firstWriter) { firstWriter = i; }
                lastWriter = i; lastReader.reset();
            } else {
                if (!firstWriter && !firstReader) { firstReader = i; }
                lastReader = i;
            }
        }
        auto pipe = static_cast<uint32_t>(payloads[1].kPipeValue);
        auto readers = [&](const auto& table) -> std::vector<fs::RegionalSelector> {
            auto found = table.find(pipe);
            return found == table.end() ? std::vector<fs::RegionalSelector>{} : found->second;
        };
        if (!selected(e, cell.firstWriters, q, firstWriter, &bindings) ||
            !selected(e, cell.lastWriters, q, lastWriter, &bindings) ||
            !selected(e, readers(cell.firstReaders), q, firstReader, &bindings) ||
            !selected(e, readers(cell.lastReaders), q, lastReader, &bindings)) {
            llvm::errs() << "phase selector mismatch q=" << q << " T="
                             << trips << " begin=" << begin << " profile=" << profile << "\n";
            return false;
        }
    }
    for (unsigned type = 0; type < 2*q; ++type) {
        std::optional<unsigned> expected;
        for (unsigned i = 0; i < count; ++i) {
            if (active(i) && 2*((i/2)%q)+i%2 == type) { expected = i; break; }
        }
        if (!selected(e, repeated.regional.firstSitePayloads.at(type), q, expected, &bindings)) { return false; }
    }
    return e.constructionError().empty();
}
} // namespace
bool runRepeatedPhaseChecks(func::FuncOp function)
{
    Operation* anchor = nullptr;
    function.walk([&](Operation* op) { if (op->hasAttr("test.nested")) { anchor = op; } });
    auto loop = anchor ? anchor->getParentOfType<scf::ForOp>() : scf::ForOp();
    if (!loop) { return false; }
    uint64_t checked = 0;
    for (unsigned q : {2U, 3U}) {
        for (unsigned trips : std::set<unsigned>{0, 1, q-1, q, q+1}) {
            for (unsigned profile = 0; profile < 4; ++profile) {
                const auto begin = trips ? trips - 1 : 0;
                if (!check(function, anchor, loop, q, trips, begin, profile, checked, true, true)) { return false; }
            }
            for (unsigned begin = 0; begin <= trips+1; ++begin) {
                for (unsigned profile = 0; profile < 4; ++profile) {
                    if (!check(function, anchor, loop, q, trips, begin, profile, checked)) { return false; }
                    const auto end = std::min(begin + 1, trips);
                    if (!check(function, anchor, loop, q, end, begin, profile, checked, true)) { return false; }
                }
            }
        }
    }
    // Keep the original expression fixture immutable. The cloned inner loop
    // uses a runtime signed lower bound and stride three; synthetic phase
    // queries above remain independently checked against their unfolded graph.
    OwningOpRef<func::FuncOp> copy(function.clone());
    Operation* copiedAnchor = nullptr;
    copy->walk([&](Operation* op) { if (op->hasAttr("test.nested")) { copiedAnchor = op; } });
    auto copiedLoop = copiedAnchor ? copiedAnchor->getParentOfType<scf::ForOp>() : scf::ForOp();
    if (!copiedLoop || copy->getNumArguments() == 0 || !copy->getArgument(0).getType().isIndex()) { return false; }
    OpBuilder builder(copiedLoop);
    auto stride = builder.create<arith::ConstantIndexOp>(copiedLoop.getLoc(), 3);
    copiedLoop->setOperand(0, copy->getArgument(0));
    copiedLoop->setOperand(2, stride);
    for (unsigned q : {2U, 3U}) {
        if (!check(*copy, copiedAnchor, copiedLoop, q, 4, 1, 0, checked, false, false, true)) { return false; }
    }
    llvm::errs() << "repeated phase oracle: " << checked << " event queries plus storage selectors\n";
    return true;
}
