// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/LifetimeStream.h"
#include "PTO/Transforms/FrontierSynch/BoundedLifetime.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/raw_ostream.h"
#include <set>
namespace fs = mlir::pto::frontiersynch;
namespace {
using Id = fs::RegionExpressions::Id;
using Pair = std::pair<uint32_t, uint32_t>;
fs::LifetimeStreamTemplate specification(unsigned mode, uint64_t count)
{
    fs::LifetimeStreamTemplate t;
    t.span = mode == 4 ? 7 : 2;
    t.maximumIterations = count;
    t.pipes = {0, 1, 0};
    t.accesses = {{0, 0, mode == 2 || mode == 3 ? 1U : 0U, mode == 3},
                  {1, 0, 0, false}, {2, 0, mode == 2 || mode == 3 ? 1U : 0U, mode == 3}};
    if (mode == 1) { t.pipes = {0, 0, 0}; t.storageProtection = {0}; }
    if (mode == 4) { t.operations = {1, 1, 0}; }
    t.prerequisites = {{1, 2, 0, false}, {0, 0, 2, true}};
    return t;
}
fs::LifetimeStreamInputs inputs(fs::RegionExpressions& e, uint64_t visit)
{
    fs::LifetimeStreamInputs in;
    in.present = {e.boolean(true), e.boolean(visit % 3 != 1), e.boolean(true)};
    in.read = {e.boolean(false), e.boolean(true), e.boolean(visit % 2 != 0)};
    // Every two visits refresh the cell, but intervening writers are conditional.
    in.write = {e.boolean(visit % 2 == 0), e.boolean(false), e.boolean(visit % 3 == 0)};
    in.prerequisites = {e.boolean(visit % 2 != 0), e.boolean(visit % 3 == 0)};
    return in;
}
template <typename F> bool eachRegister(fs::LifetimeStreamRegisters& registers, F change)
{
    auto row = [&](auto& values) {
        for (auto& value : values) { if (!change(value)) { return false; } }
        return true;
    };
    if (!row(registers.frontier.counters)) { return false; }
    for (auto* rows : {&registers.frontier.starts, &registers.frontier.completions}) {
        for (auto& values : *rows) { if (!row(values)) { return false; } }
    }
    for (auto& visit : registers.history) {
        if (!row(visit.present) || !row(visit.ranks)) { return false; }
        for (auto& values : visit.completions) { if (!row(values)) { return false; } }
        for (auto& access : visit.accesses) {
            if (!change(access.read) || !change(access.write) || !change(access.noLaterWriter)) { return false; }
        }
    }
    return true;
}
bool scenario(unsigned mode, bool symbolic)
{
    mlir::MLIRContext context;
    mlir::Block symbols;
    constexpr uint32_t count = 8, sites = 3;
    const auto t = specification(mode, count);
    fs::RegionExpressions oracle;
    fs::LifetimeWindowInput full;
    full.sites = sites; full.span = count - 1; full.storageProtection = t.storageProtection;
    for (uint32_t i = 0; i < count; ++i) {
        const auto in = inputs(oracle, i);
        for (uint32_t site = 0; site < sites; ++site) {
            full.payloads.push_back({t.pipes[site], in.present[site]});
            if (!t.operations.empty()) { full.operations.push_back(t.operations[site] ? 1 + i : 0); }
        }
        for (std::size_t a = 0; a < t.accesses.size(); ++a) {
            const auto& access = t.accesses[a];
            auto group = access.protectionGroup;
            if (group && !access.invariantProtection) { group += i; }
            full.accesses.push_back({i * sites + access.site, access.cell, in.read[a], in.write[a], group});
        }
        for (std::size_t p = 0; p < t.prerequisites.size(); ++p) {
            const auto& edge = t.prerequisites[p];
            if (i < edge.age) { continue; }
            auto& edges = edge.native ? full.nativePrerequisites : full.prerequisites;
            edges.push_back({(i - edge.age) * sites + edge.source, i * sites + edge.target, in.prerequisites[p]});
        }
    }
    const auto expected = fs::analyzeLifetimeWindow(oracle, full);
    if (!expected.error.empty()) { return false; }
    std::set<Pair> demands;
    for (const auto& edge : expected.window.retained) {
        if (oracle.constantValue(edge.guard) == 1) { demands.emplace(edge.source, edge.target); }
    }
    auto arena = std::make_unique<fs::RegionExpressions>();
    auto registers = fs::initializeLifetimeStream(*arena, t);
    for (uint32_t i = 0; i < count; ++i) {
        auto in = inputs(*arena, i);
        std::vector<std::pair<Id, Id>> bindings;
        auto symbolicInput = [&](Id& value) {
            auto type = mlir::IntegerType::get(&context, arena->isBoolean(value) ? 1 : 64);
            auto argument = symbols.addArgument(type, mlir::UnknownLoc::get(&context));
            auto input = arena->input(argument);
            bindings.emplace_back(input, value);
            value = input;
            return true;
        };
        if (symbolic) {
            eachRegister(registers, symbolicInput);
            for (auto* row : {&in.present, &in.read, &in.write, &in.prerequisites}) {
                for (auto& value : *row) { symbolicInput(value); }
            }
        }
        auto step = fs::advanceLifetimeStream(*arena, t, registers, in);
        if (!step.error.empty()) { llvm::errs() << step.error << "\n"; return false; }
        if (symbolic) {
            fs::RegionExpressions::Substitution substitution(bindings);
            auto evaluate = [&](Id& value) {
                value = arena->substitute(value, substitution);
                return bool(arena->constantValue(value));
            };
            if (!eachRegister(step.next, evaluate)) { return false; }
            for (auto& edge : step.retained) { if (!evaluate(edge.guard)) { return false; } }
            for (auto& rank : step.current.ranks) { if (!evaluate(rank)) { return false; } }
            for (auto* rows : {&step.starts, &step.current.completions}) {
                for (auto& row : *rows) {
                    for (auto& value : row) { if (!evaluate(value)) { return false; } }
                }
            }
        }
        std::set<Pair> actual, selected;
        for (const auto& edge : step.retained) {
            if (arena->constantValue(edge.guard) == 0) { continue; }
            if (arena->constantValue(edge.guard) != 1 || edge.edge.age > i) { return false; }
            actual.emplace((i - edge.edge.age) * sites + edge.edge.source, i * sites + edge.edge.target);
        }
        for (auto edge : demands) { if (edge.second / sites == i) { selected.insert(edge); } }
        if (actual != selected) {
            llvm::errs() << "stream demand mismatch mode=" << mode << " visit=" << i << "\n";
            return false;
        }
        for (uint32_t site = 0; site < sites; ++site) {
            if (arena->constantValue(step.current.ranks[site]) !=
                oracle.constantValue(expected.window.ranks[i * sites + site])) {
                return false;
            }
            for (std::size_t k = 0; k < step.starts[site].size(); ++k) {
                if (arena->constantValue(step.starts[site][k]) !=
                        oracle.constantValue(expected.window.starts[i * sites + site][k]) ||
                    arena->constantValue(step.current.completions[site][k]) !=
                        oracle.constantValue(expected.window.completions[i * sites + site][k])) { return false; }
            }
        }
        auto fresh = std::make_unique<fs::RegionExpressions>();
        registers = std::move(step.next);
        if (!eachRegister(registers, [&](Id& value) {
            auto number = arena->constantValue(value);
            if (!number) { return false; }
            value = arena->isBoolean(value) ? fresh->boolean(*number != 0) : fresh->constant(*number);
            return true;
        })) { return false; }
        arena = std::move(fresh);
    }
    return true;
}
bool boundedState()
{
    auto t = specification(0, 256);
    auto arena = std::make_unique<fs::RegionExpressions>();
    auto registers = fs::initializeLifetimeStream(*arena, t);
    std::size_t slots = 0;
    eachRegister(registers, [&](Id&) { ++slots; return true; });
    for (uint64_t i = 0; i < t.maximumIterations; ++i) {
        auto step = fs::advanceLifetimeStream(*arena, t, registers, inputs(*arena, i));
        if (!step.error.empty()) { return false; }
        auto fresh = std::make_unique<fs::RegionExpressions>();
        registers = std::move(step.next);
        std::size_t current = 0;
        if (!eachRegister(registers, [&](Id& value) {
            ++current;
            auto number = arena->constantValue(value);
            if (!number) { return false; }
            value = arena->isBoolean(value) ? fresh->boolean(*number != 0) : fresh->constant(*number);
            return true;
        }) || current != slots || fresh->size() > 2 * slots + 8) { return false; }
        arena = std::move(fresh);
    }
    t.maximumIterations = UINT64_MAX;
    if (fs::advanceLifetimeStream(*arena, t, registers, inputs(*arena, 0)).error.empty()) { return false; }
    t = specification(0, 0);
    auto initial = fs::initializeLifetimeStream(*arena, t);
    for (auto rank : initial.frontier.counters) { if (arena->constantValue(rank) != 0) { return false; } }
    return true;
}
} // namespace
int runLifetimeStreamChecks()
{
    for (unsigned mode = 0; mode < 5; ++mode) {
        if (!scenario(mode, false) || !scenario(mode, true)) {
            llvm::errs() << "lifetime streaming scenario " << mode << " failed\n";
            return 1;
        }
    }
    if (!boundedState()) { llvm::errs() << "lifetime streaming bounded state failed\n"; return 1; }
    llvm::outs() << "lifetime stream: full-window profiles and bounded register state passed\n";
    return 0;
}
