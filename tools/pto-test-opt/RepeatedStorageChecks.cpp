// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Byte-level owner oracle with overlapping read/write footprints and holes.
#include "PTO/Transforms/FrontierSynch/RepeatedStorage.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
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
} // namespace
bool runRepeatedStorageChecks(MLIRContext* context)
{
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
    auto missing = body; missing.accessBoundary.erase(missing.accessBoundary.begin());
    if (fs::recognizeRepeatedStorage(missing, loop, body.expressions->constant(2)).storage) { return false; }
    auto aliases = body; aliases.gmAliasPolicy = pto::GMAliasPolicy::MayAlias;
    if (fs::recognizeRepeatedStorage(aliases, loop, body.expressions->constant(2)).storage) { return false; }
    for (unsigned kind = 0; kind < 3; ++kind) {
        if (!originDomain(context, kind)) { return false; }
    }
    llvm::outs() << "repeated storage byte owners and boundary selectors passed\n";
    return true;
}
