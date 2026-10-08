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
#include "PTO/Transforms/FrontierSynch/SequenceAnalysis.h"
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
bool checkPhasedSelectors(const fs::RegionalAnalysis& body, func::FuncOp function, scf::ForOp loop, bool written)
{
    auto& e = *body.expressions;
    using Occurrence = std::pair<uint64_t, uint64_t>; // Original outer visit, inner ordinal.
    for (uint64_t q : {1U, 2U, 3U}) {
        for (uint64_t begin = 0; begin <= q + 1; ++begin) {
            for (uint64_t end = 0; end <= 2 * q + 2; ++end) {
                auto repeated = fs::repeatPhasedRegions(function, loop,
                    std::vector<fs::RegionalAnalysis>(q, body), e.constant(end), {}, e.constant(begin));
                if (!repeated.error.empty() || !repeated.regional.storageSelectors ||
                    repeated.regional.symbolicStorageEffects != body.symbolicStorageEffects) {
                    llvm::errs() << "phased symbolic export: " << repeated.error << "\n";
                    return false;
                }
                const auto& out = repeated.regional;
                if (written && begin < end && out.storageBoundary.empty()) { return false; }
                for (uint64_t inner : {0U, 1U, 4U}) {
                    fs::RegionExpressions::Substitution bindings({
                        {e.input(function.getArgument(3)), e.constant(inner)}});
                    for (uint64_t byte : {0U, 1U, 3U, 4U, 17U}) {
                        auto selectors = out.storageSelectors({pto::AddressSpace::GM, function.getArgument(0),
                                                               e.constant(byte)});
                        if (!selectors) { return false; }
                        // Enumerate original visit occurrences independently of
                        // the period coordinates and the production folds.
                        std::vector<Occurrence> writers, readers;
                        for (uint64_t visit = begin; visit < end; ++visit) {
                            if (written ? byte < 4 && inner > 0 : byte / 4 < inner) {
                                readers.emplace_back(visit, written ? 0 : byte / 4);
                            }
                            if (written && byte < 4) { writers.emplace_back(visit, 0); }
                        }
                        auto check = [&](const std::vector<fs::RegionalSelector>& values,
                                         std::optional<Occurrence> expected, unsigned type) {
                            std::optional<Occurrence> found;
                            for (const auto& value : values) {
                                auto active = e.constantValue(e.substitute(value.present, bindings));
                                if (!active) { return false; }
                                if (!*active) { continue; }
                                if (found || value.event.visits.size() != 1 || value.event.type % 2 != type) {
                                    return false;
                                }
                                auto period = e.constantValue(e.substitute(value.event.visits.front(), bindings));
                                auto ordinal = e.constantValue(e.substitute(value.event.ordinal, bindings));
                                if (!period || !ordinal) { return false; }
                                found = Occurrence{*period * q + value.event.type / 2, *ordinal};
                            }
                            return found == expected;
                        };
                        auto first = [](const auto& values) -> std::optional<Occurrence> {
                            return values.empty() ? std::nullopt : std::optional<Occurrence>(values.front());
                        };
                        auto last = [](const auto& values) -> std::optional<Occurrence> {
                            return values.empty() ? std::nullopt : std::optional<Occurrence>(values.back());
                        };
                        auto prefix = first(readers), suffix = last(readers);
                        // A visit's reader precedes its writer. Its prefix
                        // survives the first writer; its suffix does not.
                        if (prefix && !writers.empty() && prefix->first > writers.front().first) { prefix.reset(); }
                        if (suffix && !writers.empty() && suffix->first <= writers.back().first) { suffix.reset(); }
                        const auto pipe = static_cast<uint32_t>(body.anchors[0].phase->kPipeValue);
                        if (!check(selectors->firstWriters, first(writers), 1) ||
                            !check(selectors->lastWriters, last(writers), 1) ||
                            !check(selectors->firstReaders[pipe], prefix, 0) ||
                            !check(selectors->lastReaders[pipe], suffix, 0)) { return false; }
                        if (written && byte < 4 && begin < end) {
                            bool found = false;
                            for (const auto& cell : out.storageBoundary) {
                                if (cell.cell.base != function.getArgument(0) ||
                                    cell.cell.begin > byte || cell.cell.end <= byte) { continue; }
                                found = true;
                                if (!check(cell.firstWriters, first(writers), 1) ||
                                    !check(cell.lastWriters, last(writers), 1)) { return false; }
                            }
                            if (!found) { return false; }
                        }
                    }
                }
            }
        }
    }
    return true;
}
bool distinctPhaseScenario(MLIRContext* context)
{
    auto fail = [](StringRef message) {
        llvm::errs() << "distinct phases: " << message << "\n";
        return false;
    };
    const char* text = R"mlir(module attributes {pto.target_arch = "a3"} {
      func.func @alternating(%p: !pto.ptr<f32, gm>, %unused: !pto.ptr<f32, gm>, %n: index, %m: index) {
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %two = arith.constant 2 : index
        %value = arith.constant 1.0 : f32
        scf.for %t = %zero to %n step %one {
          %phase = arith.remui %t, %two : index
          %reader = arith.cmpi eq, %phase, %zero : index
          scf.if %reader {
            scf.for %i = %zero to %m step %one {
              %read = pto.load %p[%zero] : !pto.ptr<f32, gm> -> f32
            }
          } else {
            pto.store %value, %p[%zero] : !pto.ptr<f32, gm>, f32
          }
        }
        return
      }
    })mlir";
    auto module = parseSourceString<ModuleOp>(text, context);
    if (!module) { return fail("fixture parse"); }
    auto function = *module->getOps<func::FuncOp>().begin();
    SmallVector<scf::ForOp> loops;
    function.walk<WalkOrder::PreOrder>([&](scf::ForOp loop) { loops.push_back(loop); });
    pto::SyncInput input(pto::GMAliasPolicy::MayNotAlias);
    if (loops.size() != 2 || failed(input.build(function))) { return fail("shared input or loop count"); }
    auto complete = makeBody(input, function, loops[0], loops[1]);
    if (complete.anchors.size() != 2) { return fail("payload count"); }
    if (complete.anchors[0].phase->elementOp->getName().getStringRef() != "pto.load" ||
        complete.anchors[1].phase->elementOp->getName().getStringRef() != "pto.store") {
        return fail("fixture payload order");
    }
    auto arena = complete.expressions;
    auto& e = *arena;
    const auto count = e.input(function.getArgument(3)), zero = e.constant(0);
    std::vector<fs::RegionalAnalysis> phases;
    for (unsigned which = 0; which < 2; ++which) {
        fs::RegionalAnalysis phase;
        phase.expressions = arena; phase.accessModel = &input.accesses();
        phase.gmAliasPolicy = input.memory().gmPolicy(); phase.capabilities = {true, true, true, false};
        phase.anchors.push_back(complete.anchors[which]);
        phase.occurrenceLoops.push_back(complete.occurrenceLoops[which]);
        const auto pipe = static_cast<uint32_t>(phase.anchors[0].phase->kPipeValue);
        fs::RegionalSelector first{{0, zero, fs::PeriodicEventKind::Start},
            which ? e.boolean(true) : e.lt(zero, count)};
        auto last = first;
        if (!which) { last.event.ordinal = e.sub(count, e.constant(1)); }
        phase.firstPayloads[pipe] = {first}; phase.lastPayloads[pipe] = {last};
        phase.firstSitePayloads[0] = {first};
        for (auto effect : input.accesses().effectsFor(phase.anchors[0].phase)) {
            phase.accessBoundary.push_back({effect, first, last, false});
            phase.symbolicStorageEffects.push_back(effect);
        }
        phase.presence = [arena, count, which](fs::RegionalEvent event) -> std::optional<fs::RegionExpressions::Id> {
            if (event.type || !event.visits.empty()) { return std::nullopt; }
            return which ? arena->eq(event.ordinal, arena->constant(0)) : arena->lt(event.ordinal, count);
        };
        phase.reachability = [original = complete.reachability, which](fs::RegionalEvent a, fs::RegionalEvent b) {
            if (a.type || b.type) { return std::optional<fs::RegionExpressions::Id>{}; }
            a.type = which; b.type = which;
            return original(a, b);
        };
        phase.storageSelectors = [arena, first, last, pipe, which, base = function.getArgument(0)]
            (fs::RegionalByteAddress address) -> std::optional<fs::RegionalStorageSelectors> {
            if (address.space != pto::AddressSpace::GM || address.base != base) { return std::nullopt; }
            auto a = first, b = last;
            const auto inside = arena->lt(address.offset, arena->constant(4));
            a.present = arena->land(a.present, inside); b.present = arena->land(b.present, inside);
            fs::RegionalStorageSelectors result;
            if (which) { result.firstWriters = {a}; result.lastWriters = {b}; }
            else { result.firstReaders[pipe] = {a}; result.lastReaders[pipe] = {b}; }
            return result;
        };
        phases.push_back(std::move(phase));
    }
    for (uint64_t begin = 0; begin < 4; ++begin) {
        for (uint64_t end = begin; end < 7; ++end) {
            auto repeated = fs::repeatPhasedRegions(function, loops[0], phases, e.constant(end), {}, e.constant(begin));
            if (!repeated.error.empty()) { llvm::errs() << repeated.error << "\n"; return false; }
            auto values = repeated.regional.storageSelectors({pto::AddressSpace::GM, function.getArgument(0), zero});
            if (!values) { return fail("byte callback unavailable"); }
            std::optional<uint64_t> first, last;
            for (uint64_t visit = begin; visit < end; ++visit) {
                if (visit % 2) { if (!first) { first = visit; } last = visit; }
            }
            auto check = [&](const auto& selectors, std::optional<uint64_t> expected) {
                std::optional<uint64_t> found;
                for (const auto& selector : selectors) {
                    auto active = e.constantValue(selector.present);
                    if (!active) { return false; }
                    if (!*active) { continue; }
                    if (found || selector.event.visits.size() != 1) { return false; }
                    auto period = e.constantValue(selector.event.visits.front());
                    if (!period) { return false; }
                    found = 2 * *period + selector.event.type;
                }
                return found == expected;
            };
            if (!check(values->firstWriters, first) || !check(values->lastWriters, last)) {
                llvm::errs() << "distinct phases interval [" << begin << "," << end << ")\n";
                return fail("first/last writer mismatch");
            }
            if (begin <= 1 && end > 2) {
                // These actual load/store payloads share PIPE_S. Their
                // protected memory hazard needs native issue order, not an
                // additional completion-to-issue synchronization edge.
                auto reaches = fs::regionalReachability(repeated.regional,
                    {1, zero, fs::PeriodicEventKind::Start, {zero}},
                    {0, zero, fs::PeriodicEventKind::Start, {e.constant(1)}});
                fs::RegionExpressions::Substitution bindings({{count, e.constant(1)}});
                if (!reaches || e.constantValue(e.substitute(*reaches, bindings)) != 1) {
                    return fail("native issue-order wrap unavailable");
                }
            }
        }
    }
    auto writerExtrema = phases[1].accessBoundary;
    phases[1].accessBoundary.clear();
    auto missing = fs::repeatPhasedRegions(function, loops[0], phases, e.constant(3));
    if (missing.error.empty()) { return fail("missing writer extrema admitted"); }
    phases[1].deferredAccessBoundary = std::move(writerExtrema);
    phases[1].symbolicStorageEffects.clear(); phases[1].storageSelectors = {};
    auto deferred = fs::repeatPhasedRegions(function, loops[0], phases, e.constant(3));
    if (deferred.error.find("no exported or deferred effect extrema") == std::string::npos) {
        return fail("unconsumed deferred sibling authorized coverage");
    }
    return true;
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
        if (!checkPhasedSelectors(body, function, loops[0], kind == 8)) { return false; }
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
bool symbolicSequence(MLIRContext* context, unsigned mode)
{
    // Ordinary scalar memory instructions provide the shared modeled ranges.
    // This checks the region/storage API independently of scalar protection.
    std::string source = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @siblings(%p: !pto.ptr<f32, gm>, %q: !pto.ptr<f32, gm>, %n: index) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %value = arith.constant 1.0 : f32
    scf.for %i = %zero to %n step %one {
      pto.store %value, %p[%i] : !pto.ptr<f32, gm>, f32
    }
    scf.for %j = %zero to %n step %one {
      %read = pto.load %q[%j] : !pto.ptr<f32, gm> -> f32
    }
    return
  }
})mlir";
    if (mode != 0) {
        auto position = source.find("%q[%j]");
        source.replace(position, 6, "%p[%j]");
    }
    if (mode == 1) {
        auto position = source.find("pto.store %value, %p[%i] : !pto.ptr<f32, gm>, f32");
        source.replace(position, std::string("pto.store %value, %p[%i] : !pto.ptr<f32, gm>, f32").size(),
            "%first = pto.load %p[%i] : !pto.ptr<f32, gm> -> f32");
    }
    if (mode == 3) {
        for (const char* index : {"%p[%i]", "%p[%j]"}) {
            auto position = source.find(index);
            source.replace(position, std::string(index).size(), "%p[%zero]");
        }
    }
    auto module = parseSourceString<ModuleOp>(source, context);
    if (!module) { return false; }
    auto function = *module->getOps<func::FuncOp>().begin();
    pto::SyncInput input(pto::GMAliasPolicy::MayNotAlias);
    if (failed(input.build(function))) { return false; }
    auto arena = std::make_shared<fs::RegionExpressions>();
    auto& e = *arena;
    auto count = e.input(function.getArgument(2));
    std::vector<fs::RegionalAnalysis> children;
    for (const auto* phase : input.instructions()) {
        const auto ids = input.accesses().effectsFor(phase);
        if (ids.empty()) { continue; }
        auto loop = phase->elementOp->getParentOfType<scf::ForOp>();
        if (!loop || ids.size() != 1) { return false; }
        const auto effect = ids.front();
        const bool writer = input.accesses().effects()[effect].mode == pto::SyncAccessMode::Write;
        const auto pipe = static_cast<uint32_t>(phase->kPipeValue);
        fs::RegionalAnalysis child;
        child.expressions = arena; child.accessModel = &input.accesses();
        child.gmAliasPolicy = pto::GMAliasPolicy::MayNotAlias;
        child.capabilities = {true, true, true, false};
        auto* op = phase->elementOp;
        child.anchors.push_back({phase, {}, {op->getBlock(), op}, {op->getBlock(), op->getNextNode()}});
        child.occurrenceLoops.push_back(loop);
        fs::RegionalSelector first{{0, e.constant(0), fs::PeriodicEventKind::Start}, e.lt(e.constant(0), count)};
        auto last = first; last.event.ordinal = e.sub(count, e.constant(1));
        child.firstPayloads[pipe] = {first}; child.lastPayloads[pipe] = {last};
        child.accessBoundary.push_back({effect, first, last, false});
        child.symbolicStorageEffects = {effect};
        child.presence = [arena, count](fs::RegionalEvent event) -> std::optional<fs::RegionExpressions::Id> {
            if (event.type || !event.visits.empty()) { return std::nullopt; }
            return arena->lt(event.ordinal, count);
        };
        child.reachability = [arena, count](fs::RegionalEvent a, fs::RegionalEvent b)
            -> std::optional<fs::RegionExpressions::Id> {
            if (a.type || b.type || !a.visits.empty() || !b.visits.empty()) { return std::nullopt; }
            auto native = a.kind == fs::PeriodicEventKind::Completion && b.kind == fs::PeriodicEventKind::Start ?
                arena->boolean(false) : arena->le(a.ordinal, b.ordinal);
            return arena->land(native, arena->land(arena->lt(a.ordinal, count), arena->lt(b.ordinal, count)));
        };
        const auto base = mode == 0 && !writer ? function.getArgument(1) : function.getArgument(0);
        child.storageSelectors = [arena, count, pipe, writer, base, mode](fs::RegionalByteAddress address)
            -> std::optional<fs::RegionalStorageSelectors> {
            fs::RegionalStorageSelectors out;
            if (address.space != pto::AddressSpace::GM || address.base != base) { return out; }
            auto ordinal = arena->div(address.offset, arena->constant(4));
            fs::RegionalSelector value{{0, ordinal, fs::PeriodicEventKind::Start}, arena->lt(ordinal, count)};
            auto lastValue = value;
            if (mode == 3) {
                value.event.ordinal = arena->constant(0);
                value.present = arena->land(arena->lt(address.offset, arena->constant(4)),
                    arena->lt(arena->constant(0), count));
                lastValue = value;
                lastValue.event.ordinal = arena->sub(count, arena->constant(1));
            }
            if (writer) { out.firstWriters = {value}; out.lastWriters = {lastValue}; }
            else { out.firstReaders[pipe] = {value}; out.lastReaders[pipe] = {lastValue}; }
            return out;
        };
        if (mode == 3) {
            // Fixed four-byte accesses have a single uniform selector atom;
            // its proof uses the exact operand map, not per-byte sampling.
            auto certificate = std::make_shared<fs::RegionalSymbolicStorageCertificate>();
            certificate->expressions = arena; certificate->accessModel = child.accessModel;
            certificate->gmAliasPolicy = child.gmAliasPolicy; certificate->uniformEffects = {effect};
            fs::RegionalStorageBoundary atom;
            atom.cell = {pto::AddressSpace::GM, 0, 4, base};
            if (writer) { atom.firstWriters = {first}; atom.lastWriters = {last}; }
            else { atom.firstReaders[pipe] = {first}; atom.lastReaders[pipe] = {last}; }
            certificate->uniformBoundaries.push_back(std::move(atom));
            child.symbolicStorage = std::move(certificate);
        }
        children.push_back(std::move(child));
    }
    if (children.size() != 2) { return false; }
    auto composed = fs::composeRegionalSequence(function, arena, children, false, false);
    if (mode == 2) { return !composed.error.empty(); } // Truly parameterized crossing needs a relation adapter.
    if (!composed.error.empty()) { llvm::errs() << composed.error; return false; }
    auto out = fs::sequenceRegionalResult(composed);
    if (!out.storageSelectors || out.symbolicStorageEffects.size() != (mode == 3 ? 0 : 2) ||
        out.storageBoundary.size() != (mode == 3 ? 1 : 0)) { return false; }
    if (mode == 3) {
        for (uint64_t trips : {0U, 1U, 5U}) {
            fs::RegionExpressions::Substitution bind({{count, e.constant(trips)}});
            for (uint64_t byte : {0U, 3U, 4U}) {
                auto selected = out.storageSelectors({pto::AddressSpace::GM, function.getArgument(0),
                    e.constant(byte)});
                if (!selected) { return false; }
                unsigned active = 0;
                for (const auto& value : selected->lastWriters) {
                    auto present = e.constantValue(e.substitute(value.present, bind));
                    if (!present) { return false; }
                    if (*present) {
                        ++active;
                        if (value.event.type != 0 ||
                            e.constantValue(e.substitute(value.event.ordinal, bind)) != trips - 1) { return false; }
                    }
                }
                if (active != unsigned(trips && byte < 4)) { return false; }
            }
        }
        return true;
    }
    for (uint64_t trips : {0U, 1U, 5U}) {
        fs::RegionExpressions::Substitution bind({{count, e.constant(trips)}});
        for (uint64_t ordinal = 0; ordinal <= trips; ++ordinal) {
            auto selected = out.storageSelectors({pto::AddressSpace::GM, function.getArgument(0),
                e.constant(4*ordinal)});
            if (!selected) { return false; }
            auto check = [&](const std::vector<fs::RegionalSelector>& values, uint32_t type) {
                unsigned active = 0;
                for (const auto& value : values) {
                    auto present = e.constantValue(e.substitute(value.present, bind));
                    if (!present) { return false; }
                    if (*present) {
                        ++active;
                        if (value.event.type != type ||
                            e.constantValue(value.event.ordinal) != ordinal) { return false; }
                    }
                }
                return active == unsigned(ordinal < trips);
            };
            if (mode == 0) {
                if (!check(selected->firstWriters, 0) || !check(selected->lastWriters, 0)) { return false; }
            } else {
                if (selected->firstReaders.size() != 1 || selected->lastReaders.size() != 1 ||
                    !check(selected->firstReaders.begin()->second, 0) ||
                    !check(selected->lastReaders.begin()->second, 1)) { return false; }
            }
        }
    }
    return true;
}
} // namespace
bool runRepeatedReadOnlyStorageChecks(MLIRContext* context)
{
    for (unsigned mode = 0; mode < 4; ++mode) {
        if (!symbolicSequence(context, mode)) {
            llvm::errs() << "symbolic sibling storage scenario " << mode << " failed\n";
            return false;
        }
    }
    if (!distinctPhaseScenario(context)) {
        llvm::errs() << "distinct phased symbolic storage failed\n";
        return false;
    }
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
