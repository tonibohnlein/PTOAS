// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Byte-level owner oracle with overlapping read/write footprints and holes.
#include "PTO/Transforms/FrontierSynch/RepeatedStorage.h"
#include "PTO/Transforms/FrontierSynch/RepeatedPhases.h"
#include "../../lib/PTO/Transforms/FrontierSynch/DisjointTranslations.h"
#include "../../lib/PTO/Transforms/FrontierSynch/PhaseLogicalView.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
constexpr const char* source = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @storage(%p: !pto.ptr<f32, gm>, %q: !pto.ptr<f32, gm>, %n: index) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    %value = arith.constant 1.0 : f32
    scf.for %t = %zero to %n step %one {
      %offset = arith.muli %t, %four overflow<nsw> : index
      %old = pto.load %p[%offset] : !pto.ptr<f32, gm> -> f32
      pto.store %value, %p[%offset] : !pto.ptr<f32, gm>, f32
      %new = pto.load %p[%offset] : !pto.ptr<f32, gm> -> f32
      %shared = pto.load %q[%zero] : !pto.ptr<f32, gm> -> f32
    }
    return
  }
})mlir";
fs::RegionalAnalysis bodyFor(const pto::SyncInput& input, scf::ForOp loop)
{
    fs::RegionalAnalysis body;
    body.expressions = std::make_shared<fs::RegionExpressions>();
    body.accessModel = &input.accesses(); body.gmAliasPolicy = input.memory().gmPolicy();
    body.capabilities = {true, true, true, false};
    auto& e = *body.expressions;
    const auto zero = e.constant(0), yes = e.boolean(true);
    for (const auto* phase : input.instructions()) {
        if (!loop->isProperAncestor(phase->elementOp) || input.accesses().effectsFor(phase).empty()) { continue; }
        const auto type = static_cast<uint32_t>(body.anchors.size());
        auto* op = phase->elementOp;
        body.anchors.push_back({phase, {}, {op->getBlock(), op}, {op->getBlock(), op->getNextNode()}});
        body.occurrenceLoops.push_back({});
        fs::RegionalSelector selected{{type, zero, fs::PeriodicEventKind::Start}, yes};
        auto pipe = static_cast<uint32_t>(phase->kPipeValue);
        if (!body.firstPayloads.count(pipe)) { body.firstPayloads[pipe].push_back(selected); }
        body.lastPayloads[pipe] = {selected};
        for (auto effect : input.accesses().effectsFor(phase)) {
            body.accessBoundary.push_back({effect, selected, selected, false});
        }
    }
    const auto count = body.anchors.size();
    body.presence = [arena = body.expressions, count](fs::RegionalEvent event)
        -> std::optional<fs::RegionExpressions::Id> {
        if (event.type >= count || !event.visits.empty()) { return std::nullopt; }
        return arena->eq(event.ordinal, arena->constant(0));
    };
    body.reachability = [arena = body.expressions, present = body.presence](fs::RegionalEvent a, fs::RegionalEvent b)
        -> std::optional<fs::RegionExpressions::Id> {
        auto pa = present(a), pb = present(b);
        if (!pa || !pb) { return std::nullopt; }
        auto order = arena->boolean(a.type < b.type || (a.type == b.type &&
            (a.kind == fs::PeriodicEventKind::Start || b.kind == fs::PeriodicEventKind::Completion)));
        return arena->land(order, arena->land(*pa, *pb));
    };
    return body;
}
bool selected(ArrayRef<fs::RegionalSelector> candidates, fs::RegionExpressions& e,
              std::optional<std::pair<uint32_t, uint64_t>> expected)
{
    unsigned found = 0;
    for (const auto& candidate : candidates) {
        auto present = e.constantValue(candidate.present);
        if (!present) { return false; }
        if (!*present) { continue; }
        if (!expected || candidate.event.type != expected->first || candidate.event.visits.size() != 1 ||
            e.constantValue(candidate.event.visits.front()) != expected->second) { return false; }
        ++found;
    }
    return found == unsigned(expected.has_value());
}
bool readers(const std::map<uint32_t, std::vector<fs::RegionalSelector>>& candidates, fs::RegionExpressions& e,
             std::optional<std::pair<uint32_t, uint64_t>> expected)
{
    std::vector<fs::RegionalSelector> all;
    for (const auto& [pipe, values] : candidates) { all.insert(all.end(), values.begin(), values.end()); }
    return selected(all, e, expected);
}
bool check(const fs::RegionalAnalysis& body, scf::ForOp loop, func::FuncOp function, uint64_t trips)
{
    auto& e = *body.expressions;
    auto certificate = fs::recognizeRepeatedStorage(body, loop, e.constant(trips));
    if (!certificate.storage) { llvm::errs() << certificate.error << "\n"; return false; }
    auto repeated = fs::repeatEvolvingRegion(function, certificate.storage);
    if (!repeated.error.empty() || repeated.regional.symbolicStorageEffects.size() != 4 ||
        repeated.regional.accessBoundary.size() != 4) { return false; }
    for (uint64_t byte = 0; byte < 16 * (trips + 1); ++byte) {
        fs::RegionalByteAddress address{pto::AddressSpace::GM, function.getArgument(0), e.constant(byte)};
        auto result = repeated.regional.storageSelectors(address);
        auto owner = certificate.storage->owner(0, address);
        if (!result || !owner || e.constantValue(owner->present) != uint64_t(byte / 16 < trips)) { return false; }
        const bool active = byte / 16 < trips && byte % 16 < 4;
        auto expected = [&](uint32_t type) -> std::optional<std::pair<uint32_t, uint64_t>> {
            if (!active) { return std::nullopt; }
            return std::make_pair(type, byte / 16);
        };
        if (!selected(result->firstWriters, e, expected(1)) || !selected(result->lastWriters, e, expected(1)) ||
            !readers(result->firstReaders, e, expected(0)) || !readers(result->lastReaders, e, expected(2))) {
            return false;
        }
    }
    auto shared = repeated.regional.storageSelectors({pto::AddressSpace::GM, function.getArgument(1), e.constant(1)});
    const auto first = trips ?
        std::optional<std::pair<uint32_t, uint64_t>>(std::make_pair(3U, uint64_t(0))) : std::nullopt;
    const auto last = trips ?
        std::optional<std::pair<uint32_t, uint64_t>>(std::make_pair(3U, trips - 1)) : std::nullopt;
    if (!shared || !selected(shared->firstWriters, e, {}) || !selected(shared->lastWriters, e, {}) ||
        !readers(shared->firstReaders, e, first) || !readers(shared->lastReaders, e, last)) { return false; }
    auto nested = fs::recognizeRepeatedStorage(repeated.regional, loop, e.constant(trips));
    return !nested.storage && !nested.error.empty();
}
// The same translated map must reject signed origins, admit bounded symbolic
// natural origins, and reject origin evaluation that could overflow uint64.
bool originDomain(MLIRContext* context, unsigned kind)
{
    std::string text(source);
    const auto signature = text.find("%n: index)");
    text.replace(signature, std::string("%n: index)").size(), "%n: index, %s: index)");
    const auto loopStart = text.find("    scf.for");
    std::string definition = "    %limit = arith.constant 1024 : index\n";
    if (kind == 1) { definition += "    %origin = arith.minui %s, %limit : index\n"; }
    if (kind == 2) { definition += "    %origin = arith.maxsi %s, %zero : index\n"; }
    if (kind == 3) { definition += "    %origin = arith.divsi %s, %four : index\n"; }
    if (kind == 4) { definition += "    %origin = arith.remsi %s, %four : index\n"; }
    if (kind == 5) {
        definition += "    %natural = arith.maxsi %s, %zero : index\n"
            "    %overflow = arith.muli %natural, %four overflow<nsw> : index\n"
            "    %origin = arith.divui %overflow, %four : index\n";
    }
    if (kind == 6) { definition += "    %origin = arith.divui %s, %zero : index\n"; }
    if (kind == 7) {
        definition += "    %negative = arith.constant -4 : index\n"
            "    %origin = arith.divsi %s, %negative : index\n";
    }
    text.insert(loopStart, definition);
    const auto offsetEnd = text.find("\n", text.find("%offset ="));
    text.insert(offsetEnd, "\n      %shifted = arith.addi %offset, " +
        std::string(kind ? "%origin" : "%s") + " overflow<nsw> : index");
    for (auto position = text.find("%p[%offset]"); position != std::string::npos;
         position = text.find("%p[%offset]", position)) {
        text.replace(position, std::string("%p[%offset]").size(), "%p[%shifted]");
    }
    auto module = parseSourceString<ModuleOp>(text, context);
    if (!module) { return false; }
    auto function = module->lookupSymbol<func::FuncOp>("storage");
    scf::ForOp loop;
    Value origin;
    function.walk([&](scf::ForOp found) { loop = found; });
    function.walk([&](Operation* operation) {
        if (operation->getName().getStringRef() == "arith.minui") { origin = operation->getResult(0); }
    });
    pto::SyncInput input(pto::GMAliasPolicy::MayNotAlias);
    if (failed(input.build(function))) { return false; }
    auto body = bodyFor(input, loop);
    auto& e = *body.expressions;
    const auto trips = e.input(function.getArgument(2));
    auto certificate = fs::recognizeRepeatedStorage(body, loop, trips);
    // kind zero includes s=-2: 4*s+16*t reaches byte8 at t=1. Such signed
    // origins cannot use unsigned owner inversion and must fail admission.
    if (kind != 1) { return !certificate.storage; }
    if (!certificate.storage || !origin) { return false; }
    fs::RegionalByteAddress address{pto::AddressSpace::GM, function.getArgument(0), e.constant(45)};
    auto owner = certificate.storage->owner(0, address);
    auto selectors = certificate.storage->selectors(address);
    if (!owner || !selectors) { return false; }
    fs::RegionExpressions::Substitution bindings({{e.input(origin), e.constant(7)}, {trips, e.constant(3)}});
    if (e.constantValue(e.substitute(owner->present, bindings)) != 1 ||
        e.constantValue(e.substitute(owner->visit, bindings)) != 1 ||
        e.constantValue(e.substitute(owner->localByte, bindings)) != 1) { return false; }
    for (auto& candidate : selectors->firstWriters) {
        candidate.present = e.substitute(candidate.present, bindings);
        for (auto& visit : candidate.event.visits) { visit = e.substitute(visit, bindings); }
    }
    return selected(selectors->firstWriters, e, std::make_pair(1U, uint64_t(1)));
}
// Exercise actual dependency checks, lazy refusals, protected caller coordinates
// and placeholders retained beyond a failed export's wrapper lifetime.
bool logicalPhaseInputs(MLIRContext* context)
{
    auto module = parseSourceString<ModuleOp>(R"mlir(module {
      func.func @phase(%n: index) {
        %z = arith.constant 0 : index
        %one = arith.constant 1 : index
        %two = arith.constant 2 : index
        scf.for %i = %z to %n step %one {
          %quotient = arith.divui %i, %two : index
          %guard = arith.cmpi ult, %i, %two : index
        }
        return
      }
    })mlir", context);
    if (!module) { return false; }
    auto function = module->lookupSymbol<func::FuncOp>("phase");
    scf::ForOp loop;
    Value quotient, guard;
    function.walk([&](scf::ForOp found) { loop = found; });
    function.walk([&](arith::DivUIOp op) { quotient = op.getResult(); });
    function.walk([&](arith::CmpIOp op) { guard = op.getResult(); });
    auto index = std::make_shared<fs::PhaseIndex>();
    if (failed(index->build(function, ArrayRef<const pto::CompoundInstanceElement*>{}))) { return false; }
    auto arena = std::make_shared<fs::RegionExpressions>();
    auto& e = *arena;
    auto iv = e.input(loop.getInductionVar());
    auto unsupported = e.input(quotient), boolean = e.input(guard);
    fs::RegionalAnalysis body;
    body.expressions = arena;
    body.presence = [arena, iv](fs::RegionalEvent event) {
        return std::optional<fs::RegionExpressions::Id>(arena->eq(event.ordinal, iv));
    };
    body.endpointEventGuard = body.presence;
    body.storageSelectors = [arena, iv](fs::RegionalByteAddress address) {
        fs::RegionalStorageSelectors selectors;
        selectors.firstWriters.push_back({{0, arena->add(address.offset, iv)}, arena->boolean(true)});
        return std::optional<fs::RegionalStorageSelectors>(std::move(selectors));
    };
    // An unsupported scalar in the arena is irrelevant until a logical root
    // actually uses it. Caller iv must remain the caller's coordinate.
    auto view = body;
    if (!fs::detail::bindLogicalPhase(view, index, loop, 1, 2)) { return false; }
    auto result = view.presence({0, iv});
    if (!result) { return false; }
    fs::RegionExpressions::Substitution zero({{iv, e.constant(0)}}), one({{iv, e.constant(1)}});
    if (e.constantValue(e.substitute(*result, zero)) != 0 ||
        e.constantValue(e.substitute(*result, one)) != 1) { return false; }
    auto endpoint = view.endpointEventGuard({0, unsupported});
    auto storage = view.storageSelectors({pto::AddressSpace::GM, {}, unsupported});
    fs::RegionExpressions::Substitution caller({{unsupported, e.constant(5)}});
    if (!endpoint || e.constantValue(e.substitute(*endpoint, caller)) != 0 ||
        !storage || storage->firstWriters.size() != 1 ||
        e.constantValue(e.substitute(storage->firstWriters.front().event.ordinal, caller)) != 6) { return false; }
    for (auto unavailable : {unsupported, boolean}) {
        auto eager = body;
        eager.firstPayloads[0].push_back({{0, unavailable}, e.boolean(true)});
        if (fs::detail::bindLogicalPhase(eager, index, loop, 0, 2)) { return false; }
        auto lazy = body;
        lazy.presence = [unavailable](fs::RegionalEvent) {
            return std::optional<fs::RegionExpressions::Id>(unavailable);
        };
        lazy.endpointEventGuard = lazy.presence;
        lazy.storageSelectors = [unavailable, arena](fs::RegionalByteAddress) {
            fs::RegionalStorageSelectors selectors;
            selectors.firstWriters.push_back({{0, unavailable}, arena->boolean(true)});
            return std::optional<fs::RegionalStorageSelectors>(std::move(selectors));
        };
        if (!fs::detail::bindLogicalPhase(lazy, index, loop, 0, 2) || lazy.presence({0, e.constant(0)}) ||
            lazy.endpointEventGuard({0, e.constant(0)}) ||
            lazy.storageSelectors({pto::AddressSpace::GM, {}, e.constant(0)})) { return false; }
    }
    std::optional<fs::RegionExpressions::Id> cached;
    auto memoized = body;
    memoized.presence = [&cached, arena](fs::RegionalEvent event) {
        cached = arena->eq(event.ordinal, arena->constant(7)); return cached;
    };
    {
        auto transient = memoized;
        if (!fs::detail::bindLogicalPhase(transient, index, loop, 0, 2) || !transient.presence({0, iv})) {
            return false;
        }
    }
    if (!cached) { return false; }
    const auto inputs = e.referencedInputs(*cached);
    return inputs.size() == 1 && isa<BlockArgument>(inputs.front().second) &&
        inputs.front().second != loop.getInductionVar();
}
// Check the owner against the original integer address formula, including
// quotient boundaries, scaled subtraction spelling, holes and nonunit visits.
bool quotientOrigins(MLIRContext* context)
{
    for (unsigned width : {32U, 64U}) {
        for (unsigned step : {1U, 3U}) {
            for (unsigned divisor : {1U, 4U, 7U}) {
                for (unsigned kind = 0; kind < 3; ++kind) {
                    std::string text(source);
                    const auto integerType = width == 32 ? "i32" : "i64";
                    const uint64_t limit = width == 32 ? 1024 : (UINT64_MAX - 512) / 4;
                    text.replace(text.find("%n: index)"), 10,
                        "%n: index, %s: " + std::string(integerType) + ")");
                    text.insert(text.find("  func.func"), "  // Owner oracle\n");
                    text.insert(text.find("    scf.for"),
                        "    %limit = arith.constant " + std::to_string(limit) + " : " + integerType + "\n"
                        "    %bounded = arith.minui %s, %limit : " + integerType + "\n"
                        "    %bounded_index = arith.index_cast %bounded : " + integerType + " to index\n"
                        "    %divisor = arith.constant " + std::to_string(divisor) + " : index\n"
                        "    %step = arith.constant " + std::to_string(step) + " : index\n"
                        "    %lower = arith.constant 2 : index\n");
                    text.replace(text.find("%zero to %n step %one"), 21, "%lower to %n step %step");
                    text.insert(text.find("      %offset ="),
                        "      %quotient = arith.divui %bounded_index, %divisor : index\n"
                        "      %product = arith.muli %quotient, %divisor overflow<nsw> : index\n"
                        "      %remainder = arith.subi %bounded_index, %product : index\n"
                        "      %modulo = arith.remui %bounded_index, %divisor : index\n");
                    const char* origin = kind == 0 ? "%quotient" : kind == 1 ? "%modulo" : "%remainder";
                    text.insert(text.find("\n", text.find("%offset =")),
                        "\n      %shifted = arith.addi %offset, " + std::string(origin) + " overflow<nsw> : index");
                    for (auto pos = text.find("%p[%offset]"); pos != std::string::npos;
                         pos = text.find("%p[%offset]")) {
                        text.replace(pos, 11, "%p[%shifted]");
                    }
                    text.replace(text.find("pto.target_arch = \"a3\""), 22,
                        "pto.target_arch = \"a3\", dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, " +
                        std::to_string(width) + " : i32>>");
                    auto module = parseSourceString<ModuleOp>(text, context);
                    if (!module) { return false; }
                    auto function = module->lookupSymbol<func::FuncOp>("storage");
                    scf::ForOp loop;
                    Value bounded;
                    function.walk([&](scf::ForOp found) { loop = found; });
                    function.walk([&](arith::MinUIOp op) { bounded = op.getResult(); });
                    pto::SyncInput input(pto::GMAliasPolicy::MayNotAlias);
                    if (failed(input.build(function))) { return false; }
                    auto body = bodyFor(input, loop);
                    auto& e = *body.expressions;
                    for (uint64_t trips : {0U, 1U, 3U}) {
                        auto certificate = fs::recognizeRepeatedStorage(body, loop, e.constant(trips));
                        if (!certificate.storage) {
                            llvm::errs() << "quotient origin: " << certificate.error << "\n";
                            return false;
                        }
                        for (uint64_t value : std::array<uint64_t, 5>{0, divisor - 1, divisor, divisor + 1, limit}) {
                            const uint64_t localStart = 32 % (16 * step);
                            const uint64_t originValue = kind == 0 ? value / divisor : value % divisor;
                            const uint64_t start = 32 - localStart + 4 * originValue;
                            fs::RegionExpressions::Substitution binding({{e.input(bounded), e.constant(value)}});
                            for (uint64_t delta = 0; delta < 16 * step * (trips + 1); ++delta) {
                                const auto byte = start + delta;
                                auto address = fs::RegionalByteAddress{pto::AddressSpace::GM,
                                    function.getArgument(0), e.constant(byte)};
                                auto owner = certificate.storage->owner(0, address);
                                auto selectors = certificate.storage->selectors(address);
                                if (!owner || !selectors) { return false; }
                                auto eval = [&](fs::RegionExpressions::Id root) {
                                    return e.constantValue(e.substitute(root, binding));
                                };
                                const bool reserved = delta / (16 * step) < trips;
                                if (eval(owner->present) != uint64_t(reserved)) { return false; }
                                if (reserved && (eval(owner->visit) != delta / (16 * step) ||
                                    eval(owner->localByte) != delta % (16 * step))) { return false; }
                                for (auto& selector : selectors->firstWriters) {
                                    selector.present = e.substitute(selector.present, binding);
                                    for (auto& visit : selector.event.visits) { visit = e.substitute(visit, binding); }
                                }
                                const auto local = delta % (16 * step);
                                const bool touched = reserved && local >= localStart && local < localStart + 4;
                                const auto expected = touched ? std::optional<std::pair<uint32_t, uint64_t>>(
                                    std::make_pair(1U, delta / (16 * step))) : std::nullopt;
                                if (!selected(selectors->firstWriters, e, expected)) { return false; }
                            }
                        }
                    }
                }
            }
        }
    }
    return true;
}
// Independently inspect the ownership export for corpus qr_proj_seed's inner
// 16x128 tiles. This invokes the nested adapter even when the dispatcher can
// choose a numerical template for this particular constant inner count.
bool projectedCorpus(MLIRContext* context)
{
    constexpr const char* corpus = R"mlir(module attributes {pto.backend = "emitc", pto.kernel_kind =
#pto.kernel_kind<vector>, pto.target_arch = "a3"} {
  func.func @qr_proj_seed(%arg0: !pto.ptr<f32, gm>, %arg1: index, %arg2: index) attributes {pto.kernel_kind =
#pto.kernel_kind<vector>} {
    %c0_i64 = arith.constant 0 : i64
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %c128 = arith.constant 128 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = pto.make_tensor_view %arg0, shape = [%arg2, %c1024], strides = [%c1024, %c1] {layout = #pto.layout<nd>} :
!pto.tensor_view<?x?xf32>
    %1 = arith.divsi %arg2, %c16 : index
    scf.for %arg3 = %c0 to %1 step %c1 {
      %2 = arith.muli %arg3, %c16 : index
      scf.for %arg4 = %c0 to %c8 step %c1 {
        %3 = arith.muli %arg4, %c128 : index
        %4 = pto.alloc_tile addr = %c0_i64 valid_row = %c16 valid_col = %c128 : !pto.tile_buf<vec, 16x128xf32,
valid=?x?>
        pto.texpands ins(%cst : f32) outs(%4 : !pto.tile_buf<vec, 16x128xf32, valid=?x?>)
        %5 = arith.maxsi %2, %c0 : index
        %6 = arith.maxsi %3, %c0 : index
        %7 = pto.partition_view %0, offsets = [%5, %6], sizes = [%c16, %c128] : !pto.tensor_view<?x?xf32>
        pto.tstore ins(%4 : !pto.tile_buf<vec, 16x128xf32, valid=?x?>) outs(%7 :
!pto.partition_tensor_view<16x128xf32>) {layout = #pto.layout<nd>}
      }
    }
    return
  }
}
)mlir";
    auto module = parseSourceString<ModuleOp>(corpus, context);
    if (!module) { return false; }
    auto function = module->lookupSymbol<func::FuncOp>("qr_proj_seed");
    SmallVector<scf::ForOp> loops;
    function.walk<WalkOrder::PreOrder>([&](scf::ForOp loop) { loops.push_back(loop); });
    if (loops.size() != 2) { return false; }
    pto::SyncInput input(pto::GMAliasPolicy::MayNotAlias);
    if (failed(input.build(function))) { return false; }
    auto body = bodyFor(input, loops[0]);
    auto& e = *body.expressions;
    body.occurrenceLoops.assign(body.anchors.size(), loops[1]);
    body.presence = [arena = body.expressions](fs::RegionalEvent event) -> std::optional<fs::RegionExpressions::Id> {
        if (event.type >= 2 || !event.visits.empty()) { return std::nullopt; }
        return arena->lt(event.ordinal, arena->constant(8));
    };
    for (auto& access : body.accessBoundary) {
        access.last.event.ordinal = e.constant(7);
        access.representedByCells = input.accesses().effects()[access.effect].memory->scope != pto::AddressSpace::GM;
    }
    for (uint64_t trips : {0U, 1U, 3U}) {
        auto certificate = fs::recognizeRepeatedStorage(body, loops[0], e.constant(trips));
        if (!certificate.storage) { llvm::errs() << certificate.error << "\n"; return false; }
        for (uint64_t visit = 0; visit <= trips; ++visit) {
            for (uint64_t row : {0U, 7U, 15U}) {
                for (uint64_t column = 0; column < 8; ++column) {
                    for (uint64_t byte : {0U, 511U}) {
                        auto selectors = certificate.storage->selectors({pto::AddressSpace::GM,
                            function.getArgument(0), e.constant(visit * 65536 + row * 4096 + column * 512 + byte)});
                        if (!selectors) { return false; }
                        for (auto* candidates : {&selectors->firstWriters, &selectors->lastWriters}) {
                            unsigned active = 0;
                            for (const auto& value : *candidates) {
                                auto present = e.constantValue(value.present);
                                if (!present) { return false; }
                                if (!*present) { continue; }
                                ++active;
                                if (value.event.type != 1 || value.event.visits.size() != 1 ||
                                    e.constantValue(value.event.visits[0]) != visit ||
                                    e.constantValue(value.event.ordinal) != column) { return false; }
                            }
                            if (active != unsigned(visit < trips)) { return false; }
                        }
                    }
                }
            }
        }
    }
    auto malformed = body; malformed.expressions.reset();
    if (fs::recognizeRepeatedStorage(malformed, loops[0], e.constant(1)).storage) { return false; }
    malformed = body; malformed.occurrenceLoops.clear();
    return !fs::recognizeRepeatedStorage(malformed, loops[0], e.constant(1)).storage;
}
// Symbolic variant of the corpus tiled-output ownership pattern: preserve a
// clipped inner count instead of enumerating its possible 1024 iterations.
bool symbolicInnerOwner(MLIRContext* context)
{
    constexpr const char* text = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @symbolic_owned(%p: !pto.ptr<f32, gm>, %n: index, %trips: index) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %limit = arith.constant 1024 : index
    %count = arith.minui %n, %limit : index
    %value = arith.constant 1.0 : f32
    scf.for %t = %zero to %trips step %one {
      %base = arith.muli %t, %limit overflow<nsw> : index
      scf.for %i = %zero to %count step %one {
        %offset = arith.addi %base, %i overflow<nsw> : index
        pto.store %value, %p[%offset] : !pto.ptr<f32, gm>, f32
      }
    }
    return
  }
})mlir";
    auto module = parseSourceString<ModuleOp>(text, context);
    if (!module) { return false; }
    auto function = module->lookupSymbol<func::FuncOp>("symbolic_owned");
    SmallVector<scf::ForOp> loops;
    function.walk<WalkOrder::PreOrder>([&](scf::ForOp loop) { loops.push_back(loop); });
    if (loops.size() != 2) { return false; }
    pto::SyncInput input(pto::GMAliasPolicy::MayNotAlias);
    if (failed(input.build(function))) { return false; }
    auto body = bodyFor(input, loops[0]);
    if (body.anchors.size() != 1) { return false; }
    auto& e = *body.expressions;
    auto count = e.input(loops[1].getUpperBound());
    body.occurrenceLoops[0] = loops[1];
    body.presence = [arena = body.expressions, count](fs::RegionalEvent event)
        -> std::optional<fs::RegionExpressions::Id> {
        if (event.type || !event.visits.empty()) { return std::nullopt; }
        return arena->lt(event.ordinal, count);
    };
    for (auto& access : body.accessBoundary) {
        access.last.event.ordinal = e.sub(count, e.constant(1));
        access.first.present = access.last.present = e.lt(e.constant(0), count);
    }
    auto result = fs::recognizeRepeatedStorage(body, loops[0], e.constant(2));
    if (!result.storage) { llvm::errs() << result.error << "\n"; return false; }
    auto certificate = result.storage->certificate();
    if (!certificate || certificate->families.size() != 1) { return false; }
    const auto& family = certificate->families.front();
    if (!family.owner || !family.membership || family.kind != fs::RegionalStorageFamilyKind::VisitOwned) {
        return false;
    }
    for (uint64_t n : {0U, 1U, 7U, 1024U}) {
        for (uint64_t byte : {0U, 3U, 4U, 27U, 28U, 4095U, 4096U, 4100U, 8191U, 8192U}) {
            fs::RegionalByteAddress address{pto::AddressSpace::GM, function.getArgument(0), e.constant(byte)};
            auto selectors = result.storage->selectors(address);
            auto owner = family.owner(address);
            auto membership = family.membership(address);
            if (!selectors || !owner || !membership) { return false; }
            fs::RegionExpressions::Substitution binding({{count, e.constant(n)}});
            auto evaluated = [&](fs::RegionExpressions::Id id) { return e.constantValue(e.substitute(id, binding)); };
            const bool expected = byte < 8192 && byte % 4096 < 4 * n;
            if (evaluated(*membership) != uint64_t(expected) ||
                evaluated(owner->present) != uint64_t(byte < 8192)) { return false; }
            for (auto* candidates : {&selectors->firstWriters, &selectors->lastWriters}) {
                unsigned present = 0;
                for (const auto& candidate : *candidates) {
                    auto active = evaluated(candidate.present);
                    if (!active) { return false; }
                    if (!*active) { continue; }
                    ++present;
                    if (candidate.event.visits.size() != 1 ||
                        evaluated(candidate.event.visits[0]) != byte / 4096 ||
                        evaluated(candidate.event.ordinal) != (byte % 4096) / 4) { return false; }
                }
                if (present != unsigned(expected)) { return false; }
            }
        }
    }
    return true;
}
// Individually owned alternatives can still alias different visits. Exercise
// the joint proof with equal owners, a one-visit shift, and unequal strides.
bool alternativeOwners(MLIRContext* context)
{
    for (unsigned kind = 0; kind < 3; ++kind) {
        std::string text = R"mlir(module attributes {pto.target_arch = "a3"} {
          func.func @alternatives(%p: !pto.ptr<f32, gm>, %n: index) {
            %z = arith.constant 0 : index
            %one = arith.constant 1 : index
            %four = arith.constant 4 : index
            %two = arith.constant 2 : index
            %v = arith.constant 1.0 : f32
            scf.for %t = %z to %n step %one {
              %a = arith.muli %t, %four overflow<nsw> : index
              SECOND
              pto.store %v, %p[%a] : !pto.ptr<f32, gm>, f32
              pto.store %v, %p[%b] : !pto.ptr<f32, gm>, f32
            }
            return
          }
        })mlir";
        const std::string second = kind == 0 ? "%b = arith.addi %a, %z overflow<nsw> : index" :
            kind == 1 ? "%b = arith.addi %a, %four overflow<nsw> : index" :
                        "%b = arith.muli %t, %two overflow<nsw> : index";
        text.replace(text.find("SECOND"), 6, second);
        auto module = parseSourceString<ModuleOp>(text, context);
        if (!module) { return false; }
        auto function = module->lookupSymbol<func::FuncOp>("alternatives");
        scf::ForOp loop;
        function.walk([&](scf::ForOp found) { loop = found; });
        pto::SyncInput input(pto::GMAliasPolicy::MayNotAlias);
        if (failed(input.build(function))) { return false; }
        auto full = bodyFor(input, loop);
        if (full.anchors.size() != 2) { return false; }
        std::vector<fs::RegionalAnalysis> types;
        for (unsigned i = 0; i < 2; ++i) {
            auto body = full;
            body.anchors = {full.anchors[i]}; body.occurrenceLoops = {{}};
            body.accessBoundary.clear(); body.firstPayloads.clear(); body.lastPayloads.clear();
            for (auto access : full.accessBoundary) {
                if (access.first.event.type != i) { continue; }
                access.first.event.type = access.last.event.type = 0;
                body.accessBoundary.push_back(access);
                auto pipe = static_cast<uint32_t>(body.anchors.front().phase->kPipeValue);
                body.firstPayloads[pipe] = {access.first}; body.lastPayloads[pipe] = {access.last};
            }
            // Both separate proofs must succeed, including the unsafe unions.
            if (!fs::recognizeRepeatedStorage(body, loop, full.expressions->constant(4)).storage) { return false; }
            types.push_back(std::move(body));
        }
        auto joint = fs::recognizeRepeatedStorageTypes(types, loop, full.expressions->constant(4));
        if (joint.error.empty() != (kind == 0)) { llvm::errs() << joint.error << "\n"; return false; }
    }
    return true;
}
// One finite reader and one omitted shared reader overlap on the same pipe.
// Their union must export exactly one extremum, also across partial periods.
bool phasedReadOnlyOverlap(MLIRContext* context)
{
    constexpr const char* text = R"mlir(module attributes {pto.target_arch = "a3"} {
      func.func @readers(%p: !pto.ptr<f32, gm>, %n: index) {
        %z = arith.constant 0 : index
        %one = arith.constant 1 : index
        scf.for %t = %z to %n step %one {
          %a = pto.load %p[%z] : !pto.ptr<f32, gm> -> f32
          %b = pto.load %p[%z] : !pto.ptr<f32, gm> -> f32
        }
        return
      }
    })mlir";
    auto module = parseSourceString<ModuleOp>(text, context);
    if (!module) { return false; }
    auto function = module->lookupSymbol<func::FuncOp>("readers");
    scf::ForOp loop;
    function.walk([&](scf::ForOp found) { loop = found; });
    pto::SyncInput input(pto::GMAliasPolicy::MayNotAlias);
    if (failed(input.build(function))) { return false; }
    auto body = bodyFor(input, loop);
    if (body.anchors.size() != 2 || body.accessBoundary.size() != 2) {
        llvm::errs() << "readonly phase test shape: " << body.anchors.size() << "/" << body.accessBoundary.size() << "\n";
        return false;
    }
    auto& e = *body.expressions;
    const auto pipe = static_cast<uint32_t>(body.anchors.front().phase->kPipeValue);
    body.accessBoundary.front().representedByCells = true;
    fs::RegionalStorageBoundary cell;
    cell.cell = {pto::AddressSpace::GM, 0, 4, function.getArgument(0)};
    cell.firstReaders[pipe] = {body.accessBoundary.front().first};
    cell.lastReaders[pipe] = {body.accessBoundary.front().last};
    body.storageBoundary.push_back(cell);
    for (uint64_t trips : {0U, 1U, 2U, 3U, 5U}) {
        for (uint64_t begin = 0; begin <= trips; ++begin) {
            std::vector<fs::RegionalAnalysis> phases{body, body};
            auto proof = std::make_shared<fs::RepeatedStorageTypesResult>(
                fs::recognizeRepeatedStorageTypes(phases, loop, e.constant(trips)));
            if (!proof->error.empty()) { llvm::errs() << proof->error << "\n"; return false; }
            auto repeated = fs::repeatPhasedRegions(function, loop, phases, e.constant(trips), {},
                e.constant(begin), std::nullopt, proof);
            if (!repeated.error.empty() || !repeated.regional.storageSelectors) {
                llvm::errs() << repeated.error << "\n"; return false;
            }
            for (uint64_t byte : {0U, 3U, 4U}) {
                auto result = repeated.regional.storageSelectors(
                    {pto::AddressSpace::GM, function.getArgument(0), e.constant(byte)});
                auto expected = [&](bool first) -> std::optional<std::pair<uint32_t, uint64_t>> {
                    if (byte >= 4 || begin == trips) { return std::nullopt; }
                    const auto visit = first ? begin : trips - 1;
                    return std::make_pair(uint32_t(2 * (visit % 2) + (first ? 0 : 1)), visit / 2);
                };
                if (!result || !readers(result->firstReaders, e, expected(true)) ||
                    !readers(result->lastReaders, e, expected(false))) {
                    llvm::errs() << "readonly phase selectors: trips=" << trips << " begin=" << begin
                                 << " byte=" << byte << " query=" << bool(result) << "\n";
                    if (result) {
                        for (auto* side : {&result->firstReaders, &result->lastReaders}) {
                            for (const auto& [pipe, values] : *side) {
                                for (const auto& value : values) {
                                    llvm::errs() << " type=" << value.event.type << " present="
                                        << e.constantValue(value.present).value_or(99) << " visits="
                                        << value.event.visits.size();
                                    for (auto visit : value.event.visits) {
                                        llvm::errs() << ":" << e.constantValue(visit).value_or(99);
                                    }
                                    llvm::errs() << "\n";
                                }
                            }
                        }
                    }
                    return false;
                }
            }
            // Equal type counts do not authorize a certificate for reordered
            // anchor identities or a different access/alias context.
            // An empty domain can discard every omitted family and needs no
            // ownership projection. Identity binding applies to retained proof.
            for (unsigned mismatch = 0; proof->storage.front() && mismatch < 3; ++mismatch) {
                auto invalid = phases;
                if (mismatch == 0) { std::swap(invalid[0].anchors[0], invalid[0].anchors[1]); }
                if (mismatch == 1) { invalid[0].accessModel = nullptr; }
                if (mismatch == 2) { invalid[0].gmAliasPolicy = pto::GMAliasPolicy::MayAlias; }
                auto rejected = fs::repeatPhasedRegions(function, loop, std::move(invalid),
                    e.constant(trips), {}, e.constant(begin), std::nullopt, proof);
                if (rejected.error.empty()) {
                    llvm::errs() << "readonly phase accepted mismatch=" << mismatch << " trips=" << trips << "\n";
                    return false;
                }
            }
        }
    }
    return true;
}
bool residualProjection(MLIRContext* context)
{
    for (bool mixed : {false, true}) {
        std::string text(source);
        auto shared = text.find("      %shared =");
        text.replace(shared, text.find('\n', shared) - shared,
            "      pto.store %value, %q[%zero] : !pto.ptr<f32, gm>, f32");
        if (!mixed) {
            auto first = text.find("      %old =");
            text.erase(first, text.find("      pto.store %value, %q", first) - first);
        }
        auto module = parseSourceString<ModuleOp>(text, context);
        if (!module) { return false; }
        auto function = module->lookupSymbol<func::FuncOp>("storage");
        scf::ForOp loop;
        function.walk([&](scf::ForOp found) { loop = found; });
        pto::SyncInput input(pto::GMAliasPolicy::MayNotAlias);
        if (failed(input.build(function))) { return false; }
        auto body = bodyFor(input, loop);
        auto result = fs::recognizeRepeatedStorageTypes({body, body}, loop, body.expressions->constant(4));
        if (result.error.find("unclassified residual") == std::string::npos) { return false; }
    }
    return true;
}
// input_rmsnorm's 16 rows of 512 bf16 columns in a 7168-column tensor.
// Check the exact translated union against pairwise concrete intervals, including
// a fifteenth visit that collides with the next row and partial-column strides.
bool corpusColumns()
{
    SmallVector<pto::SyncStorageCell> ranges;
    for (uint64_t row = 0; row < 16; ++row) {
        ranges.push_back({pto::AddressSpace::GM, row * 14336, row * 14336 + 1024});
    }
    for (uint64_t stride : {512U, 1024U, 2048U}) {
        for (uint64_t trips = 0; trips <= 16; ++trips) {
            bool disjoint = true;
            for (uint64_t a = 0; a < trips; ++a) {
                for (uint64_t b = 0; b < a; ++b) {
                    for (const auto& x : ranges) {
                        for (const auto& y : ranges) {
                            disjoint &= x.begin + a * stride >= y.end + b * stride ||
                                y.begin + b * stride >= x.end + a * stride;
                        }
                    }
                }
            }
            if (fs::detail::disjointTranslations(ranges, APInt(128, stride), trips) != disjoint) { return false; }
        }
    }
    return true;
}
} // namespace
bool runRepeatedStorageChecks(MLIRContext* context)
{
    if (!logicalPhaseInputs(context)) { llvm::errs() << "logical phase input checks failed\n"; return false; }
    if (!quotientOrigins(context)) { llvm::errs() << "quotient origin checks failed\n"; return false; }
    if (!residualProjection(context)) { llvm::errs() << "residual projection check failed\n"; return false; }
    if (!alternativeOwners(context)) { llvm::errs() << "alternative owner check failed\n"; return false; }
    if (!phasedReadOnlyOverlap(context)) { llvm::errs() << "phased read-only overlap check failed\n"; return false; }
    if (!symbolicInnerOwner(context)) { llvm::errs() << "symbolic inner owner check failed\n"; return false; }
    if (!corpusColumns()) { llvm::errs() << "corpus column union check failed\n"; return false; }
    if (!projectedCorpus(context)) { llvm::errs() << "corpus nested owner check failed\n"; return false; }
    auto module = parseSourceString<ModuleOp>(source, context);
    if (!module) { return false; }
    auto function = module->lookupSymbol<func::FuncOp>("storage");
    scf::ForOp loop;
    function.walk([&](scf::ForOp found) { loop = found; });
    pto::SyncInput input(pto::GMAliasPolicy::MayNotAlias);
    if (failed(input.build(function))) { return false; }
    auto body = bodyFor(input, loop);
    if (body.anchors.size() != 4 || body.accessBoundary.size() != 4) { return false; }
    for (uint64_t trips = 0; trips < 4; ++trips) {
        if (!check(body, loop, function, trips)) { return false; }
    }
    // A separately proved persistent bank profile must be compared against
    // the entire owned reservation, while projected selectors exclude it.
    const auto persistentEffect = body.accessBoundary.back().effect;
    fs::RepeatedPersistentStorage persistent{{pto::AddressSpace::GM, 0, 64, function.getArgument(1)},
        {persistentEffect}, true};
    auto projected = fs::recognizeRepeatedStorageWithPersistent(body, loop,
        body.expressions->constant(3), {persistent});
    if (!projected.storage) { llvm::errs() << projected.error << "\n"; return false; }
    auto present = projected.storage->projectedSelectors({pto::AddressSpace::GM, function.getArgument(0),
        body.expressions->constant(0)});
    auto absent = projected.storage->projectedSelectors({pto::AddressSpace::GM, function.getArgument(1),
        body.expressions->constant(0)});
    if (!present || !absent || present->firstWriters.empty() || !absent->firstReaders.empty() ||
        !absent->lastReaders.empty()) { return false; }
    auto aliasesPersistent = body;
    aliasesPersistent.gmAliasPolicy = pto::GMAliasPolicy::MayAlias;
    if (fs::recognizeRepeatedStorageWithPersistent(aliasesPersistent, loop,
        body.expressions->constant(3), {persistent}).storage) { return false; }
    // Full profile reservations, not the first canonical bank, govern safety.
    persistent.reservation.base = function.getArgument(0);
    persistent.reservation.begin = 32; persistent.reservation.end = 68;
    if (fs::recognizeRepeatedStorageWithPersistent(body, loop,
        body.expressions->constant(4), {persistent}).storage) { return false; }
    auto missing = body; missing.accessBoundary.erase(missing.accessBoundary.begin());
    if (fs::recognizeRepeatedStorage(missing, loop, body.expressions->constant(2)).storage) { return false; }
    auto aliases = body; aliases.gmAliasPolicy = pto::GMAliasPolicy::MayAlias;
    if (fs::recognizeRepeatedStorage(aliases, loop, body.expressions->constant(2)).storage) { return false; }
    for (unsigned kind = 0; kind < 8; ++kind) {
        if (!originDomain(context, kind)) { return false; }
    }
    llvm::outs() << "repeated storage byte owners and boundary selectors passed\n";
    return true;
}
