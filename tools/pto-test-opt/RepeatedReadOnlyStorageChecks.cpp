// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Symbolic reader lifting preserves inner ownership and rejects every write alias.
#include "PTO/Transforms/FrontierSynch/RepeatedStorage.h"
#include "PTO/Transforms/FrontierSynch/RepeatedPhases.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/STLExtras.h"
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
constexpr const char* source = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @readers(%p: !pto.ptr<f32, gm>, %q: !pto.ptr<f32, gm>, %n: index, %m: index) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %value = arith.constant 1.0 : f32
    scf.for %t = %zero to %n step %one {
      scf.for %i = %zero to %m step %one {
        %read = pto.load %p[%i] : !pto.ptr<f32, gm> -> f32
      }
      pto.store %value, %q[%zero] : !pto.ptr<f32, gm>, f32
    }
    return
  }
})mlir";
fs::RegionalAnalysis makeBody(const pto::SyncInput& input, func::FuncOp function,
                             scf::ForOp outer, scf::ForOp inner)
{
    fs::RegionalAnalysis body;
    body.expressions = std::make_shared<fs::RegionExpressions>();
    body.accessModel = &input.accesses();
    body.gmAliasPolicy = input.memory().gmPolicy();
    body.capabilities = {true, true, true, false};
    auto arena = body.expressions;
    auto& e = *arena;
    const auto zero = e.constant(0), count = e.input(function.getArgument(3));
    for (const auto* phase : input.instructions()) {
        if (!outer->isProperAncestor(phase->elementOp) || input.accesses().effectsFor(phase).empty()) { continue; }
        const auto type = static_cast<uint32_t>(body.anchors.size());
        auto* op = phase->elementOp;
        const bool nested = inner->isProperAncestor(op);
        body.anchors.push_back({phase, {}, {op->getBlock(), op}, {op->getBlock(), op->getNextNode()}});
        body.occurrenceLoops.push_back(nested ? inner : scf::ForOp());
        fs::RegionalSelector first{{type, zero, fs::PeriodicEventKind::Start},
            nested ? e.lt(zero, count) : e.boolean(true)};
        auto last = first;
        if (nested) { last.event.ordinal = e.sub(count, e.constant(1)); }
        const auto pipe = static_cast<uint32_t>(phase->kPipeValue);
        auto initial = first;
        for (const auto& previous : body.firstPayloads[pipe]) {
            initial.present = e.land(initial.present, e.lnot(previous.present));
        }
        body.firstPayloads[pipe].push_back(initial);
        for (auto& previous : body.lastPayloads[pipe]) {
            previous.present = e.land(previous.present, e.lnot(last.present));
        }
        body.lastPayloads[pipe].push_back(last);
        for (auto id : input.accesses().effectsFor(phase)) {
            body.accessBoundary.push_back({id, first, last, false});
            if (input.accesses().effects()[id].mode == pto::SyncAccessMode::Read &&
                input.accesses().effects()[id].memory->scope == pto::AddressSpace::GM) {
                body.symbolicStorageEffects.push_back(id);
            }
        }
    }
    body.presence = [arena, count](fs::RegionalEvent event) -> std::optional<fs::RegionExpressions::Id> {
        if (event.type > 1 || !event.visits.empty()) { return std::nullopt; }
        return event.type == 0 ? arena->lt(event.ordinal, count) : arena->eq(event.ordinal, arena->constant(0));
    };
    body.reachability = [arena, present = body.presence](fs::RegionalEvent a, fs::RegionalEvent b)
        -> std::optional<fs::RegionExpressions::Id> {
        auto pa = present(a), pb = present(b);
        if (!pa || !pb) { return std::nullopt; }
        auto within = arena->lor(arena->lt(a.ordinal, b.ordinal), arena->land(arena->eq(a.ordinal, b.ordinal),
            arena->boolean(a.kind == fs::PeriodicEventKind::Start || b.kind == fs::PeriodicEventKind::Completion)));
        auto order = a.type == b.type ? within : arena->boolean(a.type < b.type);
        return arena->land(order, arena->land(*pa, *pb));
    };
    // A byte belongs to exactly one inner reader. The outer lifting must retain
    // this ordinal, including holes/out-of-range absence and unavailable bases.
    if (body.anchors.empty()) { return body; }
    const auto readerPipe = static_cast<uint32_t>(body.anchors.front().phase->kPipeValue);
    body.storageSelectors = [arena, count, readerPipe, base = function.getArgument(0)](fs::RegionalByteAddress address)
        -> std::optional<fs::RegionalStorageSelectors> {
        if (address.base != base || address.space != pto::AddressSpace::GM) { return std::nullopt; }
        const auto ordinal = arena->div(address.offset, arena->constant(4));
        fs::RegionalSelector selected{{0, ordinal, fs::PeriodicEventKind::Start}, arena->lt(ordinal, count)};
        fs::RegionalStorageSelectors result;
        result.firstReaders[readerPipe].push_back(selected);
        result.lastReaders[readerPipe].push_back(selected);
        return result;
    };
    return body;
}
bool checkSelectors(const fs::RegionalAnalysis& repeated, func::FuncOp function, uint64_t trips)
{
    auto& e = *repeated.expressions;
    if (!trips) {
        auto empty = repeated.storageSelectors({pto::AddressSpace::GM, function.getArgument(0), e.constant(0)});
        return empty && empty->firstWriters.empty() && empty->lastWriters.empty() &&
            empty->firstReaders.empty() && empty->lastReaders.empty();
    }
    for (uint64_t inner : {0U, 1U, 5U}) {
        fs::RegionExpressions::Substitution bindings({{e.input(function.getArgument(3)), e.constant(inner)}});
        for (uint64_t ordinal = 0; ordinal <= inner; ++ordinal) {
            auto selectors = repeated.storageSelectors({pto::AddressSpace::GM, function.getArgument(0),
                e.constant(ordinal * 4 + 1)});
            if (!selectors || !selectors->firstWriters.empty() || !selectors->lastWriters.empty()) { return false; }
            for (bool last : {false, true}) {
                auto& readers = last ? selectors->lastReaders : selectors->firstReaders;
                if (readers.size() != 1 || readers.begin()->second.size() != 1) { return false; }
                const auto& selected = readers.begin()->second.front();
                if (e.constantValue(e.substitute(selected.present, bindings)) != uint64_t(trips && ordinal < inner) ||
                    selected.event.visits.size() != 1 || e.constantValue(selected.event.ordinal) != ordinal) {
                    return false;
                }
                if (trips && e.constantValue(selected.event.visits.front()) != (last ? trips - 1 : 0)) { return false; }
            }
        }
    }
    return !repeated.storageSelectors({pto::AddressSpace::GM, function.getArgument(1), e.constant(0)});
}
bool scenario(MLIRContext* context, unsigned kind)
{
    std::string text(source);
    if (kind == 1 || kind == 2 || kind >= 8) {
        auto position = text.find("%q[%zero]");
        text.replace(position, std::string("%q[%zero]").size(), "%p[%zero]");
    }
    if (kind >= 8) {
        auto position = text.find("%p[%i]");
        text.replace(position, std::string("%p[%i]").size(), "%p[%zero]");
    }
    if (kind == 3) {
        auto position = text.find("%p[%i]");
        text.replace(position, std::string("%p[%i]").size(), "%p[%t]");
    }
    auto module = parseSourceString<ModuleOp>(text, context);
    if (!module) { return false; }
    auto function = module->lookupSymbol<func::FuncOp>("readers");
    SmallVector<scf::ForOp> loops;
    function.walk<WalkOrder::PreOrder>([&](scf::ForOp loop) { loops.push_back(loop); });
    if (loops.size() != 2) { return false; }
    pto::SyncInput input(kind == 4 ? pto::GMAliasPolicy::MayAlias : pto::GMAliasPolicy::MayNotAlias);
    if (failed(input.build(function))) { return false; }
    auto body = makeBody(input, function, loops[0], loops[1]);
    if (body.anchors.size() != 2 || body.symbolicStorageEffects.size() != 1) { return false; }
    if (kind == 2) {
        llvm::erase_if(body.accessBoundary, [&](const auto& access) {
            return input.accesses().effects()[access.effect].mode == pto::SyncAccessMode::Write;
        });
    }
    if (kind == 5) { body.symbolicStorageEffects.push_back(input.accesses().effects().size()); }
    if (kind == 6) { body.storageSelectors = {}; }
    if (kind == 7 || kind >= 8) {
        for (const auto& access : body.accessBoundary) {
            if (input.accesses().effects()[access.effect].mode == pto::SyncAccessMode::Write) {
                body.symbolicStorageEffects.push_back(access.effect);
            }
        }
    }
    if (kind >= 8) {
        auto arena = body.expressions;
        const auto pipe = static_cast<uint32_t>(body.anchors[0].phase->kPipeValue);
        body.storageSelectors = [arena, pipe, base = function.getArgument(0), count = function.getArgument(3), kind]
            (fs::RegionalByteAddress address) -> std::optional<fs::RegionalStorageSelectors> {
            if (kind == 9 || address.base != base) { return std::nullopt; }
            const auto active = arena->lt(address.offset, arena->constant(4));
            fs::RegionalStorageSelectors result;
            fs::RegionalSelector writer{{1, arena->constant(0), fs::PeriodicEventKind::Start}, active};
            result.firstWriters.push_back(writer);
            result.lastWriters.push_back(writer);
            fs::RegionalSelector reader{{0, arena->constant(0), fs::PeriodicEventKind::Start},
                arena->land(active, arena->lt(arena->constant(0), arena->input(count)))};
            result.firstReaders[pipe].push_back(reader);
            return result;
        };
    }
    if (kind == 0 || kind == 8) {
        // Phased interval export needs separate clipping of the symbolic callback
        // and finite atoms. A nonzero begin must not silently retain visit zero.
        auto phased = fs::repeatPhasedRegions(function, loops[0], {body}, body.expressions->constant(5),
            {}, body.expressions->constant(2));
        if (phased.error.find("symbolic") == std::string::npos) { return false; }
    }
    for (uint64_t trips : {0U, 1U, 3U}) {
        auto repeated = fs::repeatInvariantRegion(function, loops[0], body, body.expressions->constant(trips));
        const bool emptyAdmitted = !trips && (kind == 1 || kind == 2 || kind == 4 || kind >= 7);
        if (kind == 8 || emptyAdmitted) {
            if (!repeated.error.empty()) { return false; }
            if (!trips) {
                if (!checkSelectors(repeated.regional, function, trips)) { return false; }
                continue;
            }
            if (repeated.regional.cost.cells != body.cost.cells + 4 ||
                repeated.regional.cost.boundaryBytes != body.cost.boundaryBytes + 4) { return false; }
            auto selected = repeated.regional.storageSelectors({pto::AddressSpace::GM, function.getArgument(0),
                body.expressions->constant(1)});
            if (!selected || selected->firstWriters.size() != 1 || selected->lastWriters.size() != 1 ||
                !selected->lastReaders.empty() ||
                body.expressions->constantValue(selected->firstWriters[0].event.visits.front()) != 0 ||
                body.expressions->constantValue(selected->lastWriters[0].event.visits.front()) != trips - 1) {
                return false;
            }
        } else if (kind) {
            if (repeated.error.empty()) { return false; }
        } else if (!repeated.error.empty() || repeated.regional.symbolicStorageEffects != body.symbolicStorageEffects ||
                   repeated.regional.accessBoundary.size() != body.accessBoundary.size() ||
                   !checkSelectors(repeated.regional, function, trips)) {
            llvm::errs() << "read-only symbolic lifting failed: " << repeated.error << "\n";
            return false;
        }
    }
    return true;
}
// A finite GM scalar reader and an ordinary MTE3 writer through a potentially
// aliasing pointer. Native pipe order cannot establish writer->next-reader.
bool residualCounterexample(MLIRContext* context, unsigned mode)
{
    constexpr const char* program = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @readers(%p: !pto.ptr<f32, gm>, %q: !pto.ptr<f32, gm>, %n: index, %m: index) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %sixteen = arith.constant 16 : index
    %addr = arith.constant 0 : i64
    %tile = pto.alloc_tile addr = %addr valid_row = %one valid_col = %sixteen :
      !pto.tile_buf<vec, 1x16xf32, valid=?x?>
    %view = pto.make_tensor_view %q, shape = [%one, %sixteen], strides = [%sixteen, %one]
      {layout = #pto.layout<nd>} : !pto.tensor_view<?x?xf32>
    scf.for %t = %zero to %n step %one {
      scf.for %i = %zero to %m step %one {
        %read = pto.load %p[%zero] : !pto.ptr<f32, gm> -> f32
      }
      %part = pto.partition_view %view, offsets = [%zero, %m], sizes = [%one, %sixteen] :
        !pto.tensor_view<?x?xf32>
      pto.tstore ins(%tile : !pto.tile_buf<vec, 1x16xf32, valid=?x?>)
        outs(%part : !pto.partition_tensor_view<1x16xf32>) {layout = #pto.layout<nd>}
    }
    return
  }
})mlir";
    auto module = parseSourceString<ModuleOp>(program, context);
    if (!module) { return false; }
    auto function = module->lookupSymbol<func::FuncOp>("readers");
    SmallVector<scf::ForOp> loops;
    function.walk<WalkOrder::PreOrder>([&](scf::ForOp loop) { loops.push_back(loop); });
    if (loops.size() != 2) { return false; }
    pto::SyncInput input(pto::GMAliasPolicy::MayAlias);
    if (failed(input.build(function))) { return false; }
    auto body = makeBody(input, function, loops[0], loops[1]);
    if (body.anchors.size() != 2 || body.symbolicStorageEffects.size() != 1) { return false; }
    if (mode) {
        for (auto it = body.accessBoundary.begin(); it != body.accessBoundary.end();) {
            const auto& effect = input.accesses().effects()[it->effect];
            if (effect.mode != pto::SyncAccessMode::Write) { ++it; continue; }
            if (mode == 2) { body.deferredAccessBoundary.push_back(*it); }
            it = body.accessBoundary.erase(it);
        }
    }
    auto arena = body.expressions;
    const auto count = arena->input(function.getArgument(3));
    const auto pipe = static_cast<uint32_t>(body.anchors[0].phase->kPipeValue);
    body.storageSelectors = [arena, count, pipe, base = function.getArgument(0)](fs::RegionalByteAddress address)
        -> std::optional<fs::RegionalStorageSelectors> {
        if (address.base != base) { return std::nullopt; }
        const auto present = arena->land(arena->lt(address.offset, arena->constant(4)),
            arena->lt(arena->constant(0), count));
        fs::RegionalStorageSelectors selected;
        selected.firstReaders[pipe].push_back({{0, arena->constant(0), fs::PeriodicEventKind::Start}, present});
        selected.lastReaders[pipe].push_back(
            {{0, arena->sub(count, arena->constant(1)), fs::PeriodicEventKind::Start}, present});
        return selected;
    };
    auto repeated = fs::repeatInvariantRegion(function, loops[0], body, arena->constant(2));
    if (mode == 1) { return !repeated.error.empty(); }
    if (!repeated.error.empty()) {
        llvm::errs() << "residual symbolic repetition: " << repeated.error << "\n";
        return false;
    }
    auto reaches = fs::regionalReachability(repeated.regional,
        {1, arena->constant(0), fs::PeriodicEventKind::Completion, {arena->constant(0)}},
        {0, arena->constant(0), fs::PeriodicEventKind::Start, {arena->constant(1)}});
    fs::RegionExpressions::Substitution bindings({{count, arena->constant(1)}});
    return reaches && arena->constantValue(arena->substitute(*reaches, bindings)) == 1;
}
} // namespace
bool runRepeatedReadOnlyStorageChecks(MLIRContext* context)
{
    for (unsigned kind = 0; kind < 10; ++kind) {
        if (!scenario(context, kind)) {
            llvm::errs() << "read-only symbolic repetition scenario " << kind << " failed\n";
            return false;
        }
    }
    for (unsigned mode = 0; mode < 3; ++mode) {
        if (!residualCounterexample(context, mode)) {
            llvm::errs() << "finite symbolic residual-writer scenario " << mode << " failed\n";
            return false;
        }
    }
    llvm::outs() << "read-only symbolic repetition selectors and rejection checks passed\n";
    return true;
}
