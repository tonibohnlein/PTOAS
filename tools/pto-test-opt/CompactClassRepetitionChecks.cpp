// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent concrete nested occurrence oracle; production never unfolds trips.
#include "PTO/Transforms/FrontierSynch/CompactClassRepetition.h"
#include "PTO/Transforms/FrontierSynch/HardwareProtection.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
using Expr = fs::RegionExpressions::Id;
using Kind = fs::PeriodicEventKind;
using Graph = std::vector<std::vector<bool>>;
struct Executed { fs::RegionalEvent event; const pto::CompoundInstanceElement* phase; };
std::optional<int64_t> scalar(Value value, const DenseMap<Value, int64_t>& values)
{
    if (auto found = values.find(value); found != values.end()) { return found->second; }
    APInt constant;
    if (matchPattern(value, m_ConstantInt(&constant)) && constant.isSignedIntN(64)) { return constant.getSExtValue(); }
    if (auto add = value.getDefiningOp<arith::AddIOp>()) {
        auto a = scalar(add.getLhs(), values), b = scalar(add.getRhs(), values);
        if (a && b) { return *a + *b; }
    }
    return std::nullopt;
}
bool execute(Block& block, const fs::PhaseIndex& index, const fs::RegionalAnalysis& region,
    DenseMap<Value, int64_t>& values, DenseMap<Operation*, uint64_t>& visits, std::vector<Executed>& word)
{
    auto& arena = *region.expressions;
    for (Operation& operation : block) {
        if (auto loop = dyn_cast<scf::ForOp>(operation)) {
            const auto lower = scalar(loop.getLowerBound(), values), upper = scalar(loop.getUpperBound(), values),
                       step = scalar(loop.getStep(), values);
            if (!lower || !upper || !step || *step <= 0) { return false; }
            uint64_t ordinal = 0;
            for (int64_t iv = *lower; iv < *upper; iv += *step, ++ordinal) {
                values[loop.getInductionVar()] = iv; visits[loop] = ordinal;
                if (!execute(*loop.getBody(), index, region, values, visits, word)) { return false; }
            }
            values.erase(loop.getInductionVar()); visits.erase(loop);
            continue;
        }
        if (auto branch = dyn_cast<scf::IfOp>(operation)) {
            const auto condition = scalar(branch.getCondition(), values);
            if (!condition) { return false; }
            auto& arm = *condition ? branch.getThenRegion() : branch.getElseRegion();
            if (!arm.empty() && !execute(arm.front(), index, region, values, visits, word)) { return false; }
            continue;
        }
        for (auto* phase : index.phasesFor(&operation)) {
            uint32_t type = 0;
            while (type < region.anchors.size() && region.anchors[type].phase != phase) { ++type; }
            if (type == region.anchors.size()) { return false; }
            fs::RegionalEvent event{type, arena.constant(0), Kind::Start};
            if (auto loop = region.occurrenceLoops[type]) { event.ordinal = arena.constant(visits.lookup(loop)); }
            if (!region.outerLoops.empty()) {
                for (auto loop : region.outerLoops[type]) {
                    event.visits.push_back(arena.constant(visits.lookup(loop)));
                }
            }
            word.push_back({std::move(event), phase});
        }
    }
    return true;
}
Graph original(llvm::ArrayRef<Executed> word, const pto::SyncInput& input)
{
    Graph graph(2 * word.size(), std::vector<bool>(2 * word.size(), false));
    const auto& model = input.accesses();
    for (std::size_t a = 0; a < word.size(); ++a) {
        graph[2*a][2*a] = graph[2*a+1][2*a+1] = graph[2*a][2*a+1] = true;
        const auto p = static_cast<uint32_t>(word[a].phase->kPipeValue);
        for (std::size_t b = a + 1; b < word.size(); ++b) {
            const auto q = static_cast<uint32_t>(word[b].phase->kPipeValue);
            if (p == q) { graph[2*a][2*b] = graph[2*a+1][2*b+1] = true; }
            if (fs::ptoStorageProtection().protectsScalar(p, q)) { continue; }
            for (auto x : model.effectsFor(word[a].phase)) {
                for (auto y : model.effectsFor(word[b].phase)) {
                    if (model.mayConflict(x, y)) { graph[2*a+1][2*b] = true; }
                }
            }
        }
    }
    for (std::size_t k = 0; k < graph.size(); ++k) {
        for (std::size_t a = 0; a < graph.size(); ++a) {
            for (std::size_t b = 0; b < graph.size(); ++b) {
                graph[a][b] = graph[a][b] || (graph[a][k] && graph[k][b]);
            }
        }
    }
    return graph;
}
std::optional<bool> query(const fs::RegionalOrderView& view, fs::RegionalEvent a, fs::RegionalEvent b,
                          const DenseMap<Value, int64_t>& values)
{
    auto& arena = *view.context()->expressions();
    auto answer = fs::regionalReachability(view.regional(), a, b);
    if (!answer) { return std::nullopt; }
    SmallVector<std::pair<Expr, Expr>> bindings;
    for (const auto& [id, value] : arena.referencedInputs(*answer)) {
        auto number = scalar(value, values);
        if (!number) { return std::nullopt; }
        bindings.push_back({id, value.getType().isInteger(1) ? arena.boolean(*number != 0) :
                                                          arena.constant(static_cast<uint64_t>(*number))});
    }
    fs::RegionExpressions::Substitution substitution(bindings);
    const auto result = arena.constantValue(arena.substitute(*answer, substitution));
    return result ? std::optional<bool>(*result != 0) : std::nullopt;
}
bool verify(func::FuncOp function, scf::ForOp outer, const pto::SyncInput& input,
            const fs::PhaseIndex& index, const fs::CompactClassRepetition& result, uint64_t& cases)
{
    const auto& bounds = result.boundary->bounds();
    const auto& region = bounds.upper->regional();
    const bool guarded = llvm::any_of(function.getArgumentTypes(), [](Type type) { return type.isInteger(1); });
    for (int64_t flag = 0; flag < (guarded ? 2 : 1); ++flag) {
        for (int64_t outerCount : {0, 1, 2}) {
            for (int64_t innerCount : {0, 1, 3}) {
                DenseMap<Value, int64_t> values;
                for (auto argument : function.getArguments()) {
                    values[argument] = argument.getType().isInteger(1) ? flag : innerCount;
                }
                values[function.getArgument(0)] = outerCount;
                DenseMap<Operation*, uint64_t> visits;
                std::vector<Executed> word;
                // Enumerate only this test's small outer prefix, with original loop
                // indices mapped independently into the exported tuple metadata.
                for (int64_t visit = 0; visit < outerCount; ++visit) {
                    values[outer.getInductionVar()] = visit; visits[outer] = visit;
                    if (!execute(*outer.getBody(), index, region, values, visits, word)) { return false; }
                }
                values.erase(outer.getInductionVar());
                const auto expected = original(word, input);
                for (std::size_t a = 0; a < word.size(); ++a) {
                    for (std::size_t b = 0; b < word.size(); ++b) {
                        auto source = word[a].event, target = word[b].event;
                        source.kind = Kind::Completion; target.kind = Kind::Start;
                        const auto upper = query(*bounds.upper, source, target, values);
                        const auto lower = query(*bounds.lower, source, target, values);
                        if (!upper || !lower || (*lower && !expected[2*a+1][2*b]) ||
                            (expected[2*a+1][2*b] && !*upper)) { return false; }
                    }
                }
                // The unrelated final V payload need not finish before the next
                // invocation's first scalar reader: the actual last writer releases it.
                if (function->hasAttr("test.repeat_early") && outerCount == 2 && innerCount > 0) {
                    const auto half = word.size() / 2;
                    auto source = word[half - 1].event, target = word[half].event;
                    source.kind = Kind::Completion; target.kind = Kind::Start;
                    auto early = query(*bounds.upper, source, target, values);
                    if (!early || *early) { return false; }
                }
                ++cases;
            }
        }
    }
    return true;
}
} // namespace
int runCompactClassRepetitionChecks(func::FuncOp function, const pto::SyncInput& input)
{
    if (!function->hasAttr("test.class_repeat")) { return 0; }
    scf::ForOp outer;
    for (auto loop : function.getOps<scf::ForOp>()) { if (outer) { return 1; } outer = loop; }
    if (!outer) { return 1; }
    fs::PhaseIndex index;
    if (failed(index.build(function, input))) { return 1; }
    auto arena = std::make_shared<fs::RegionExpressions>();
    auto result = fs::analyzeCompactClassRepetition(function, outer, input, arena);
    if (function->hasAttr("test.repeat_reject")) {
        if (result.error.find("uniform") == std::string::npos || !result.original || result.boundary) {
            llvm::errs() << "expected relative-body preservation: " << result.error << "\n"; return 1;
        }
        return 0;
    }
    uint64_t cases = 0;
    if (!result.error.empty() || !result.boundary || !result.mathematical ||
        result.boundary->invocationBlock() != outer->getBlock() ||
        result.boundary->nativeExports().capabilities.completeStorageModel ||
        !verify(function, outer, input, index, result, cases)) {
        llvm::errs() << "compact class repetition failed: " << function.getSymName() << ": " << result.error << "\n";
        return 1;
    }
    auto foreign = function->getParentOfType<ModuleOp>().lookupSymbol<func::FuncOp>("scope_other");
    std::string error;
    std::vector<fs::RequirementGroupId> groups(input.accesses().cells().size(), 1);
    if (!foreign || fs::captureFiniteRequirementsInBlock(function, foreign.front(), input, {}, arena,
        100, groups, 2, error) || error.empty()) { return 1; }
    // A root composition cannot silently reinterpret a relative body frame.
    const auto wrongScope = fs::composeCompactClassBoundaries(function, input, {result.original});
    if (wrongScope.error.empty() || wrongScope.original.empty()) { return 1; }
    llvm::outs() << "compact class repetition passed: " << function.getSymName() << " cases=" << cases << "\n";
    return 0;
}
