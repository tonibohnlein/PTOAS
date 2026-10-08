// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/FiniteVisitDemands.h"
#include "PTO/Transforms/FrontierSynch/FiniteVisitRecognition.h"
#include "PTO/Transforms/InsertSync/SyncInput.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <array>
#include <set>

using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
using Matrix = std::vector<std::vector<bool>>;
using Edge = std::pair<unsigned, unsigned>;
constexpr unsigned siteCount = 3;
constexpr unsigned typeCount = 3;
constexpr std::array<std::array<bool, siteCount>, typeCount> writes{{
    {{false, true, false}}, {{true, false, true}}, {{false, false, true}}
}};
constexpr std::array<pto::PipelineType, siteCount> pipes{{
    pto::PipelineType::PIPE_MTE1, pto::PipelineType::PIPE_M, pto::PipelineType::PIPE_V
}};
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
// This oracle starts with every raw ordered storage conflict, not boundary
// selectors or the constructor's crossing candidates.
Matrix unfolded(llvm::ArrayRef<unsigned> word, bool storage,
                llvm::ArrayRef<fs::FiniteVisitPrerequisite> extra = {})
{
    const unsigned count = siteCount * word.size();
    Matrix graph(2 * count, std::vector<bool>(2 * count));
    for (unsigned a = 0; a < count; ++a) {
        graph[2*a][2*a] = graph[2*a+1][2*a+1] = graph[2*a][2*a+1] = true;
        for (unsigned b = a+1; b < count; ++b) {
            if (a % siteCount == b % siteCount) {
                graph[2*a][2*b] = graph[2*a+1][2*b+1] = true;
            }
            if (storage && (writes[word[a/siteCount]][a%siteCount] ||
                            writes[word[b/siteCount]][b%siteCount])) {
                graph[2*a+1][2*b] = true;
            }
        }
    }
    for (unsigned visit = 1; visit < word.size(); ++visit) {
        for (const auto& edge : extra) {
            if ((!storage && !edge.native) || edge.sourceType != word[visit-1] ||
                edge.targetType != word[visit]) { continue; }
            graph[2*siteCount*(visit-1) + 2*edge.source.type + 1][2*siteCount*visit + 2*edge.target.type] = true;
        }
    }
    close(graph);
    return graph;
}
std::set<Edge> covers(const Matrix& required, const Matrix& native)
{
    std::set<Edge> result;
    for (unsigned a = 1; a < required.size(); a += 2) {
        for (unsigned b = 0; b < required.size(); b += 2) {
            if (!required[a][b] || native[a][b]) { continue; }
            bool intermediate = false;
            for (unsigned k = 0; k < required.size(); ++k) {
                if (k != a && k != b && required[a][k] && required[k][b]) { intermediate = true; break; }
            }
            if (!intermediate) { result.emplace(a, b); }
        }
    }
    return result;
}
fs::RegionalAnalysis makeType(std::shared_ptr<fs::RegionExpressions> arena,
                              const std::vector<pto::CompoundInstanceElement>& phases, unsigned type)
{
    auto& e = *arena;
    const auto zero = e.constant(0), yes = e.boolean(true);
    fs::RegionalAnalysis region;
    region.expressions = arena;
    region.capabilities = {true, true, true, false};
    for (unsigned site = 0; site < siteCount; ++site) {
        auto* anchor = phases[site].elementOp;
        region.anchors.push_back({&phases[site], {}, {anchor->getBlock(), anchor},
                                  {anchor->getBlock(), anchor->getNextNode()}});
        region.occurrenceLoops.push_back({});
        fs::RegionalSelector selector{{site, zero, fs::PeriodicEventKind::Start}, yes};
        const auto pipe = static_cast<uint32_t>(pipes[site]);
        region.firstPayloads[pipe] = region.lastPayloads[pipe] = {selector};
        region.firstSitePayloads[site] = {selector};
    }
    region.presence = [arena, zero](fs::RegionalEvent event) -> std::optional<fs::RegionExpressions::Id> {
        if (event.type >= siteCount || !event.visits.empty()) { return std::nullopt; }
        return arena->eq(event.ordinal, zero);
    };
    const auto local = unfolded({type}, true);
    region.reachability = [arena, local, presence = region.presence](fs::RegionalEvent a, fs::RegionalEvent b)
        -> std::optional<fs::RegionExpressions::Id> {
        auto pa = presence(a), pb = presence(b);
        if (!pa || !pb) { return std::nullopt; }
        const auto i = 2*a.type + unsigned(a.kind == fs::PeriodicEventKind::Completion);
        const auto j = 2*b.type + unsigned(b.kind == fs::PeriodicEventKind::Completion);
        return arena->land(arena->land(*pa, *pb), arena->boolean(local[i][j]));
    };
    fs::RegionalStorageBoundary cell;
    cell.cell = {pto::AddressSpace::LEFT, 0, 8};
    unsigned first = siteCount, last = 0;
    for (unsigned site = 0; site < siteCount; ++site) {
        if (writes[type][site]) { first = std::min(first, site); last = site; }
    }
    cell.firstWriters = {{{first, zero, fs::PeriodicEventKind::Start}, yes}};
    cell.lastWriters = {{{last, zero, fs::PeriodicEventKind::Start}, yes}};
    for (unsigned site = 0; site < siteCount; ++site) {
        if (writes[type][site]) { continue; }
        const auto pipe = static_cast<uint32_t>(pipes[site]);
        fs::RegionalSelector selector{{site, zero, fs::PeriodicEventKind::Start}, yes};
        if (site < first) { cell.firstReaders[pipe] = {selector}; }
        if (site > last) { cell.lastReaders[pipe] = {selector}; }
    }
    region.storageBoundary.push_back(std::move(cell));
    return region;
}
bool checkWord(const fs::FiniteVisitDemands& library, llvm::ArrayRef<unsigned> word,
               uint64_t& pairsChecked)
{
    auto& e = *library.original->types.front().body.expressions;
    const auto& extra = library.original->prerequisites;
    const auto expected = unfolded(word, true, extra), native = unfolded(word, false, extra);
    Matrix actual(expected.size(), std::vector<bool>(expected.size()));
    std::set<Edge> demands;
    for (unsigned visit = 0; visit < word.size(); ++visit) {
        const auto local = unfolded({word[visit]}, true), localNative = unfolded({word[visit]}, false);
        const auto offset = 2 * siteCount * visit;
        for (unsigned a = 0; a < local.size(); ++a) {
            for (unsigned b = 0; b < local.size(); ++b) { actual[offset+a][offset+b] = local[a][b]; }
        }
        for (const auto& edge : covers(local, localNative)) {
            demands.emplace(offset + edge.first, offset + edge.second);
        }
        if (!visit) { continue; }
        const auto& boundary = library.boundaries[word[visit-1] * typeCount + word[visit]];
        if (boundary.sourceType != word[visit-1] || boundary.targetType != word[visit]) { return false; }
        for (bool isNative : {false, true}) {
            const auto& crossings = isNative ? boundary.native : boundary.demands;
            for (const auto& crossing : crossings) {
                auto enabled = e.constantValue(crossing.guard);
                if (!enabled) { return false; }
                if (!*enabled) { continue; }
                if (crossing.native != isNative || crossing.displacement != 1 ||
                    crossing.source.type >= siteCount || crossing.target.type >= siteCount ||
                    !crossing.source.visits.empty() || !crossing.target.visits.empty() ||
                    e.constantValue(crossing.source.ordinal) != 0 ||
                    e.constantValue(crossing.target.ordinal) != 0) { return false; }
                const auto a = offset - 2*siteCount + 2*crossing.source.type +
                               unsigned(crossing.source.kind == fs::PeriodicEventKind::Completion);
                const auto b = offset + 2*crossing.target.type +
                               unsigned(crossing.target.kind == fs::PeriodicEventKind::Completion);
                actual[a][b] = true;
                if (!isNative) {
                    if (crossing.source.kind != fs::PeriodicEventKind::Completion ||
                        crossing.target.kind != fs::PeriodicEventKind::Start) { return false; }
                    demands.emplace(a, b);
                }
            }
        }
    }
    if (demands != covers(expected, native)) {
        llvm::errs() << "finite visit retained demand set differs from independent full-graph covers\n";
        return false;
    }
    close(actual);
    for (unsigned a = 0; a < expected.size(); ++a) {
        for (unsigned b = 0; b < expected.size(); ++b) {
            if (actual[a][b] != expected[a][b]) {
                llvm::errs() << "finite visit closure differs at " << a << "," << b << "\n";
                return false;
            }
            ++pairsChecked;
        }
    }
    return true;
}
bool rejectedInputs(func::FuncOp function, const fs::FiniteVisitInput& good)
{
    auto missingRefresh = good;
    auto& cell = missingRefresh.types.back().body.storageBoundary.front();
    cell.firstWriters.clear();
    cell.lastWriters.clear();
    auto refresh = fs::buildFiniteVisitDemands(function, std::move(missingRefresh));
    if (refresh.error.empty() || !refresh.boundaries.empty() || !refresh.original) {
        llvm::errs() << "finite visit accepted a type missing common persistent refresh\n";
        return false;
    }
    auto missingPipe = good;
    const auto pipe = static_cast<uint32_t>(pipes.back());
    missingPipe.types.back().body.firstPayloads.erase(pipe);
    missingPipe.types.back().body.lastPayloads.erase(pipe);
    auto absent = fs::buildFiniteVisitDemands(function, std::move(missingPipe));
    if (absent.error.empty() || !absent.boundaries.empty() || !absent.original) {
        llvm::errs() << "finite visit accepted a type missing common pipe selectors\n";
        return false;
    }
    auto conditionalRefresh = good;
    auto arena = good.types.front().body.expressions;
    const auto predicate = arena->input(function.getArgument(0));
    conditionalRefresh.types.back().body.storageBoundary.front().firstWriters.front().present = predicate;
    conditionalRefresh.types.back().body.storageBoundary.front().lastWriters.front().present = predicate;
    auto conditional = fs::buildFiniteVisitDemands(function, std::move(conditionalRefresh));
    if (conditional.error.empty()) {
        llvm::errs() << "finite visit accepted refresh available only under an unproved predicate\n";
        return false;
    }
    auto unavailable = good;
    unavailable.types.back().body.reachability = [](fs::RegionalEvent, fs::RegionalEvent)
        -> std::optional<fs::RegionExpressions::Id> { return std::nullopt; };
    auto partial = fs::buildFiniteVisitDemands(function, std::move(unavailable));
    if (partial.error.empty() || !partial.boundaries.empty() || !partial.cost.pairReductions) {
        llvm::errs() << "finite visit failed pair retained partial products or did not exercise prior pairs\n";
        return false;
    }
    auto invalid = good;
    invalid.prerequisites.push_back({typeCount, 0, {}, {}, arena->boolean(true), false});
    return !fs::buildFiniteVisitDemands(function, std::move(invalid)).error.empty();
}
bool uniformlyOptional(func::FuncOp function, const fs::FiniteVisitInput& good)
{
    auto input = good;
    auto arena = input.types.front().body.expressions;
    const auto enabled = arena->input(function.getArgument(0));
    for (auto& type : input.types) {
        const auto presence = type.body.presence;
        const auto reaches = type.body.reachability;
        type.body.presence = [arena, enabled, presence](fs::RegionalEvent event)
            -> std::optional<fs::RegionExpressions::Id> {
            auto result = presence(event);
            return result ? std::optional<fs::RegionExpressions::Id>(arena->land(enabled, *result)) : std::nullopt;
        };
        type.body.reachability = [arena, enabled, reaches](fs::RegionalEvent a, fs::RegionalEvent b)
            -> std::optional<fs::RegionExpressions::Id> {
            auto result = reaches(a, b);
            return result ? std::optional<fs::RegionExpressions::Id>(arena->land(enabled, *result)) : std::nullopt;
        };
    }
    auto library = fs::buildFiniteVisitDemands(function, std::move(input));
    if (!library.error.empty()) { llvm::errs() << library.error << "\n"; return false; }
    // At enabled=true the same independent active-visit oracle applies. At
    // enabled=false every type is empty, so no boundary edge may be present.
    for (auto& boundary : library.boundaries) {
        for (auto* records : {&boundary.demands, &boundary.native}) {
            for (auto& edge : *records) {
                auto active = arena->constantUnder(enabled, edge.guard);
                auto absent = arena->constantUnder(arena->lnot(enabled), edge.guard);
                if (!active || absent != 0) { return false; }
                edge.guard = arena->boolean(*active != 0);
            }
        }
    }
    uint64_t checked = 0;
    return checkWord(library, {0, 2, 1, 0}, checked);
}
bool additionalCases(func::FuncOp function, const fs::FiniteVisitInput& good)
{
    auto arena = good.types.front().body.expressions;
    const auto zero = arena->constant(0), yes = arena->boolean(true);
    for (bool native : {false, true}) {
        auto input = good;
        input.prerequisites.push_back({0, 0, {2, zero, fs::PeriodicEventKind::Completion},
            {0, zero, fs::PeriodicEventKind::Start}, yes, native});
        auto result = fs::buildFiniteVisitDemands(function, std::move(input));
        uint64_t count = 0;
        if (!result.error.empty() || !checkWord(result, {0, 0, 1, 0}, count) ||
            !checkWord(result, {0, 0, 0}, count)) {
            llvm::errs() << "finite visit additional prerequisite mismatch\n";
            return false;
        }
    }
    // Identical bytes, independently partitioned by each child. Boundary
    // qualification must reconcile atoms before testing common refresh.
    auto partitioned = good;
    for (unsigned type = 1; type < typeCount; ++type) {
        auto cell = partitioned.types[type].body.storageBoundary.front();
        partitioned.types[type].body.storageBoundary.clear();
        const unsigned split = type == 1 ? 3 : 5;
        cell.cell.end = split;
        partitioned.types[type].body.storageBoundary.push_back(cell);
        cell.cell.begin = split;
        cell.cell.end = 8;
        partitioned.types[type].body.storageBoundary.push_back(cell);
    }
    auto result = fs::buildFiniteVisitDemands(function, std::move(partitioned));
    uint64_t count = 0;
    if (!result.error.empty() || !checkWord(result, {2, 0, 1, 2}, count)) {
        llvm::errs() << "finite visit physical repartition mismatch: " << result.error << "\n";
        return false;
    }
    auto empty = good;
    empty.context = arena->boolean(false);
    auto inactive = fs::buildFiniteVisitDemands(function, std::move(empty));
    if (!inactive.error.empty()) { return false; }
    for (const auto& boundary : inactive.boundaries) {
        for (const auto* records : {&boundary.demands, &boundary.native}) {
            for (const auto& edge : *records) {
                if (arena->constantValue(edge.guard) != 0) { return false; }
            }
        }
    }
    return true;
}
bool originalIR(MLIRContext* context, unsigned mode)
{
    std::string source = R"mlir(module attributes {pto.target_arch = "a3"} {
      func.func @selected(%p: !pto.ptr<f32, gm>, %n: index, %cut: index) {
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %value = arith.constant 1.0 : f32
        scf.for %visit = %zero to %n step %one {
          %choose = arith.cmpi slt, %visit, %cut : index
          scf.if %choose {
            pto.store %value, %p[%zero] : !pto.ptr<f32, gm>, f32
            %read = pto.load %p[%zero] : !pto.ptr<f32, gm> -> f32
          } else {
            %read = pto.load %p[%zero] : !pto.ptr<f32, gm> -> f32
            pto.store %value, %p[%zero] : !pto.ptr<f32, gm>, f32
          }
        }
        return
      }
    })mlir";
    const std::string store = "pto.store %value, %p[%zero] : !pto.ptr<f32, gm>, f32";
    if (mode == 1) {
        const auto position = source.rfind(store);
        if (position == std::string::npos) { return false; }
        source.erase(position, store.size());
    } else if (mode == 2) {
        const auto position = source.find("%p[%zero]");
        if (position == std::string::npos) { return false; }
        source.replace(position, std::string("%p[%zero]").size(), "%p[%visit]");
    }
    auto module = parseSourceString<ModuleOp>(source, context);
    if (!module) { return false; }
    auto function = module->lookupSymbol<func::FuncOp>("selected");
    pto::SyncInput input(pto::GMAliasPolicy::MayNotAlias);
    if (failed(input.build(function))) { return false; }
    auto program = fs::recognizeProgram(function, input);
    if (failed(program)) { return false; }
    for (std::size_t node = 0; node < program->nodes.size(); ++node) {
        if (program->nodes[node].kind != fs::StructureKind::Loop) { continue; }
        auto result = fs::analyzeFiniteVisitLoop(function, input, *program, node);
        const auto& contract = result.recognition.contract;
        if (mode) {
            return (!result.demands || !result.demands->error.empty()) &&
                   contract.membership != fs::ContractStatus::Established;
        }
        if (!result.recognition.error.empty() || !result.demands || !result.demands->error.empty() ||
            result.recognition.alternatives.size() != 2 || result.children.size() != 2 ||
            result.demands->cost.pairReductions != 4 ||
            contract.membership != fs::ContractStatus::Established ||
            contract.demands != fs::ContractImplementation::Available ||
            contract.endpointRecipes == fs::ContractImplementation::Available ||
            contract.allocation == fs::ContractImplementation::Available) {
            llvm::errs() << "finite visit original-IR qualification failed: " << result.recognition.error << "\n";
            if (result.demands) { llvm::errs() << result.demands->error << "\n"; }
            return false;
        }
        for (const auto& alternative : result.recognition.alternatives) {
            if (alternative.selection.empty()) { return false; }
        }
        return true;
    }
    return false;
}
} // namespace
bool runFiniteVisitChecks(MLIRContext* context)
{
    constexpr const char* source = R"mlir(module {
      func.func @finite_visits(%enabled: i1) {
        %a = arith.constant 0 : index
        %b = arith.constant 1 : index
        %c = arith.constant 2 : index
        return
      }
    })mlir";
    auto module = parseSourceString<ModuleOp>(source, context);
    if (!module) { return false; }
    auto function = module->lookupSymbol<func::FuncOp>("finite_visits");
    auto arena = std::make_shared<fs::RegionExpressions>();
    std::vector<pto::CompoundInstanceElement> phases;
    phases.reserve(siteCount);
    for (auto& operation : function.front()) {
        if (phases.size() == siteCount) { break; }
        phases.emplace_back(phases.size(), SmallVector<const pto::BaseMemInfo*>{},
            SmallVector<const pto::BaseMemInfo*>{}, pipes[phases.size()], operation.getName());
        phases.back().elementOp = &operation;
    }
    if (phases.size() != siteCount) { return false; }
    fs::FiniteVisitInput input;
    for (unsigned type = 0; type < typeCount; ++type) {
        input.types.push_back({makeType(arena, phases, type), {}});
    }
    auto library = fs::buildFiniteVisitDemands(function, input);
    if (!library.error.empty()) { llvm::errs() << library.error << "\n"; return false; }
    if (!library.original || library.boundaries.size() != typeCount*typeCount ||
        library.cost.pairReductions != typeCount*typeCount || library.persistentCells.size() != 1 ||
        library.pipes.size() != siteCount) { return false; }
    const auto nodes = arena->size();
    uint64_t pairsChecked = 0, wordsChecked = 0;
    unsigned words = 1;
    for (unsigned length = 0; length <= 4; ++length) {
        for (unsigned code = 0; code < words; ++code) {
            std::vector<unsigned> word(length);
            unsigned remaining = code;
            for (auto& type : word) { type = remaining % typeCount; remaining /= typeCount; }
            if (!checkWord(library, word, pairsChecked)) {
                llvm::errs() << "finite visit word length=" << length << " code=" << code << "\n";
                return false;
            }
            ++wordsChecked;
        }
        words *= typeCount;
    }
    // Arbitrarily many word selections only reuse the precomputed h² products;
    // no new expression or analysis is needed to select their recipes.
    if (arena->size() != nodes || library.cost.pairReductions != typeCount*typeCount ||
        !rejectedInputs(function, input) || !additionalCases(function, input) ||
        !uniformlyOptional(function, input)) { return false; }
    for (unsigned mode = 0; mode < 3; ++mode) {
        if (!originalIR(context, mode)) {
            llvm::errs() << "finite visit original-IR mode " << mode << " failed\n";
            return false;
        }
    }
    llvm::outs() << "finite visit checked " << wordsChecked << " words and " << pairsChecked
                 << " event pairs with " << library.cost.pairReductions << " pair reductions\n";
    return true;
}
