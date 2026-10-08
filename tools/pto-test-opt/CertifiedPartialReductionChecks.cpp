// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/CertifiedPartialReduction.h"
#include "PTO/Transforms/InsertSync/SyncInput.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
using Id = fs::RegionExpressions::Id;
using Matrix = std::vector<std::vector<bool>>;
constexpr unsigned Predicates = 3;
constexpr unsigned Valuations = 1U << Predicates;
// The oracle follows graph arcs from every vertex; it does not use the reducer's
// path certificates or any production reachability/reduction implementation.
Matrix closure(Matrix direct)
{
    Matrix result(direct.size(), std::vector<bool>(direct.size()));
    for (std::size_t source = 0; source < direct.size(); ++source) {
        std::vector<std::size_t> pending;
        if (direct[source][source]) { pending.push_back(source); result[source][source] = true; }
        for (std::size_t next = 0; next < pending.size(); ++next) {
            for (std::size_t target = 0; target < direct.size(); ++target) {
                if (direct[pending[next]][target] && !result[source][target]) {
                    result[source][target] = true;
                    pending.push_back(target);
                }
            }
        }
    }
    return result;
}
struct Fixture {
    Fixture(MLIRContext& context, const pto::SyncInput& input, uint32_t sites, uint32_t trips = 1)
        : input(input), sites(sites), trips(trips), arena(std::make_shared<fs::RegionExpressions>())
    {
        conditions = {arena->boolean(false), arena->boolean(true)};
        for (unsigned i = 0; i < Predicates; ++i) {
            auto value = parameters.addArgument(IntegerType::get(&context, 1), UnknownLoc::get(&context));
            predicates.push_back(arena->input(value));
            conditions.push_back(predicates.back());
            conditions.push_back(arena->lnot(predicates.back()));
        }
        for (uint32_t type = 0; type < sites; ++type) { pipes.push_back(type); }
        presence.assign(sites * trips, conditions[1]);
    }
    fs::RegionalEvent event(uint32_t occurrence, fs::PeriodicEventKind kind) const
    {
        return {occurrence % sites, arena->constant(occurrence / sites), kind};
    }
    void edge(uint32_t source, uint32_t target, unsigned guard = 1)
    {
        generators.push_back({{19, 100 + generators.size()}, event(source, fs::PeriodicEventKind::Completion),
            event(target, fs::PeriodicEventKind::Start), conditions[guard]});
    }
    uint64_t value(Id expression, unsigned mask) const
    {
        std::vector<std::pair<Id, Id>> bindings;
        for (unsigned i = 0; i < predicates.size(); ++i) {
            bindings.push_back({predicates[i], arena->boolean((mask & (1U << i)) != 0)});
        }
        fs::RegionExpressions::Substitution substitution(bindings);
        return arena->constantValue(arena->substitute(expression, substitution)).value_or(UINT64_MAX);
    }
    uint32_t occurrence(const fs::RegionalEvent& event, unsigned mask) const
    {
        return static_cast<uint32_t>(value(event.ordinal, mask)) * sites + event.type;
    }
    Matrix graph(unsigned mask, llvm::ArrayRef<Id> retained, bool nativeOnly = false) const;
    bool build(bool unknown = false, bool unknownOrder = false);
    std::vector<fs::PartialReductionAttempt> attempts() const;
    bool check(const fs::CertifiedPartialReduction& result, uint64_t& valuations) const;
    const pto::SyncInput& input;
    uint32_t sites, trips;
    Block parameters;
    std::shared_ptr<fs::RegionExpressions> arena;
    std::vector<Id> predicates, conditions, presence;
    std::vector<uint32_t> pipes;
    std::vector<fs::PartialReductionGenerator> generators;
    std::vector<std::pair<uint32_t, uint32_t>> fixed;
    fs::OrderContext context;
    fs::PartialGraphSnapshot snapshot;
};
Matrix Fixture::graph(unsigned mask, llvm::ArrayRef<Id> retained, bool nativeOnly) const
{
    const auto count = sites * trips;
    Matrix direct(2 * count, std::vector<bool>(2 * count));
    for (uint32_t a = 0; a < count; ++a) {
        if (value(presence[a], mask) != 1) { continue; }
        direct[2*a][2*a] = direct[2*a+1][2*a+1] = direct[2*a][2*a+1] = true;
        for (uint32_t b = a + 1; b < count; ++b) {
            if (value(presence[b], mask) == 1 && pipes[a % sites] == pipes[b % sites]) {
                direct[2*a][2*b] = direct[2*a+1][2*b+1] = true;
            }
        }
    }
    for (auto [a, b] : fixed) {
        if (value(presence[a], mask) == 1 && value(presence[b], mask) == 1) { direct[2*a+1][2*b] = true; }
    }
    if (!nativeOnly) {
        for (std::size_t i = 0; i < generators.size(); ++i) {
            const auto& edge = generators[i];
            auto a = occurrence(edge.source, mask), b = occurrence(edge.target, mask);
            auto active = retained.empty() ? edge.active : retained[i];
            if (a < count && b < count && value(active, mask) == 1 && value(presence[a], mask) == 1 &&
                value(presence[b], mask) == 1) { direct[2*a+1][2*b] = true; }
        }
    }
    return closure(std::move(direct));
}
using CircuitMatrix = std::vector<std::vector<Id>>;
CircuitMatrix table(const Fixture& fixture, bool nativeOnly)
{
    auto& e = *fixture.arena;
    const auto count = 2 * fixture.sites * fixture.trips;
    CircuitMatrix result(count, std::vector<Id>(count, e.boolean(false)));
    for (unsigned mask = 0; mask < Valuations; ++mask) {
        auto guard = e.boolean(true);
        for (unsigned i = 0; i < Predicates; ++i) {
            auto term = fixture.predicates[i];
            guard = e.land(guard, mask & (1U << i) ? term : e.lnot(term));
        }
        auto reachable = fixture.graph(mask, {}, nativeOnly);
        for (uint32_t a = 0; a < count; ++a) {
            for (uint32_t b = 0; b < count; ++b) {
                if (reachable[a][b]) { result[a][b] = e.lor(result[a][b], guard); }
            }
        }
    }
    return result;
}
fs::RegionalOrderQueries query(const Fixture& fixture, CircuitMatrix table, bool unknown)
{
    auto arena = fixture.arena;
    const auto sites = fixture.sites, trips = fixture.trips;
    fs::RegionalOrderQueries queries;
    queries.reachability = [arena, sites, trips, table = std::move(table), unknown]
        (fs::RegionalEvent a, fs::RegionalEvent b) -> std::optional<Id> {
        if (unknown || a.type >= sites || b.type >= sites) { return std::nullopt; }
        auto result = arena->boolean(false);
        for (uint32_t i = 0; i < trips; ++i) {
            for (uint32_t j = 0; j < trips; ++j) {
                auto ai = 2 * (i * sites + a.type) + (a.kind == fs::PeriodicEventKind::Completion);
                auto bi = 2 * (j * sites + b.type) + (b.kind == fs::PeriodicEventKind::Completion);
                auto match = arena->land(arena->eq(a.ordinal, arena->constant(i)),
                                         arena->eq(b.ordinal, arena->constant(j)));
                result = arena->lor(result, arena->land(match, table[ai][bi]));
            }
        }
        return result;
    };
    return queries;
}
bool Fixture::build(bool unknown, bool unknownOrder)
{
    fs::RegionalAnalysis domain;
    domain.expressions = arena;
    domain.accessModel = &input.accesses();
    domain.gmAliasPolicy = input.memory().gmPolicy();
    domain.presence = [e = arena, sites = sites, trips = trips, present = presence]
        (fs::RegionalEvent event) -> std::optional<Id> {
        if (event.type >= sites) { return std::nullopt; }
        auto result = e->boolean(false);
        for (uint32_t i = 0; i < trips; ++i) {
            result = e->lor(result, e->land(e->eq(event.ordinal, e->constant(i)), present[i * sites + event.type]));
        }
        return result;
    };
    if (unknownOrder) {
        domain.referenceBefore = [](fs::RegionalEvent, fs::RegionalEvent) -> std::optional<Id> { return std::nullopt; };
    }
    std::string error;
    auto captured = fs::captureRegionalOrderContext(input, domain, error);
    if (failed(captured)) { llvm::errs() << error << "\n"; return false; }
    context = *captured;
    auto actual = fs::makeRegionalOrderView(context, query(*this, table(*this, false), unknown), error);
    auto native = fs::makeRegionalOrderView(context, query(*this, table(*this, true), false), error);
    if (failed(actual) || failed(native)) { llvm::errs() << error << "\n"; return false; }
    std::vector<fs::RequirementMembership> memberships;
    for (const auto& edge : generators) { memberships.push_back({3, edge.record, edge.active}); }
    auto provenance = fs::createRequirementProvenance(context, {{3, fs::RequirementScope::Internal}},
                                                      std::move(memberships), error);
    if (!provenance) { llvm::errs() << error << "\n"; return false; }
    snapshot = std::make_shared<const fs::CertifiedPartialGraph>(fs::CertifiedPartialGraph{
        *actual, *native, generators, std::move(provenance)});
    return true;
}
std::vector<fs::PartialReductionAttempt> Fixture::attempts() const
{
    std::vector<fs::PartialReductionAttempt> result;
    for (std::size_t i = 0; i < generators.size(); ++i) {
        result.push_back({snapshot, i, conditions[1], std::nullopt});
        for (uint32_t w = 0; w < sites * trips; ++w) {
            // Include endpoints deliberately: strict order must reject them.
            for (auto kind : {fs::PeriodicEventKind::Start, fs::PeriodicEventKind::Completion}) {
                result.push_back({snapshot, i, conditions[1], event(w, kind)});
            }
        }
    }
    return result;
}
bool Fixture::check(const fs::CertifiedPartialReduction& result, uint64_t& valuations) const
{
    if (!result.error.empty() || result.graph != snapshot || result.retained.size() != generators.size() ||
        result.removed.size() != generators.size()) { llvm::errs() << result.error << "\n"; return false; }
    if (result.graph->provenance != snapshot->provenance) { return false; }
    for (unsigned mask = 0; mask < Valuations; ++mask) {
        if (graph(mask, {}) != graph(mask, result.retained)) { return false; }
        for (std::size_t i = 0; i < generators.size(); ++i) {
            const auto active = value(generators[i].active, mask);
            const auto retained = value(result.retained[i], mask), removed = value(result.removed[i], mask);
            if (retained > 1 || removed > 1 || (retained && removed) || (retained | removed) != active) {
                return false;
            }
            if (!(result.graph->generators[i].record == generators[i].record) ||
                result.graph->generators[i].active != generators[i].active) { return false; }
        }
        ++valuations;
    }
    return true;
}
bool exhaustive(MLIRContext& context, const pto::SyncInput& input, uint64_t& valuations)
{
    for (unsigned edges = 0; edges < 64; ++edges) {
        for (unsigned mode = 0; mode < 4; ++mode) {
            Fixture fixture(context, input, 4);
            if (mode & 1U) { fixture.pipes = {0, 1, 0, 1}; }
            if (mode & 2U) { fixture.presence[1] = fixture.conditions[4]; }
            unsigned index = 0;
            for (uint32_t a = 0; a < 4; ++a) {
                for (uint32_t b = a + 1; b < 4; ++b) {
                    if (edges & (1U << index)) { fixture.edge(a, b, mode == 0 ? 1 : 2 + index % 6); }
                    ++index;
                }
            }
            if (!fixture.build()) { return false; }
            auto attempts = fixture.attempts();
            auto result = fs::reduceCertifiedPartial(fixture.snapshot, attempts);
            if (!fixture.check(result, valuations) || result.cost.attempts != attempts.size()) { return false; }
        }
    }
    return true;
}
bool cornerCases(MLIRContext& context, const pto::SyncInput& input, uint64_t& valuations)
{
    Fixture shortcut(context, input, 4);
    for (uint32_t a = 0; a < 4; ++a) { for (uint32_t b = a + 1; b < 4; ++b) { shortcut.edge(a, b); } }
    if (!shortcut.build()) { return false; }
    auto shortcuts = fs::reduceCertifiedPartial(shortcut.snapshot, shortcut.attempts());
    if (!shortcut.check(shortcuts, valuations) || shortcut.value(shortcuts.removed[1], 0) != 1 ||
        shortcut.value(shortcuts.removed[2], 0) != 1) { return false; }
    Fixture duplicate(context, input, 2);
    duplicate.edge(0, 1, 2); duplicate.edge(0, 1, 3); duplicate.edge(0, 1, 4);
    if (!duplicate.build()) { return false; }
    auto aliases = fs::reduceCertifiedPartial(duplicate.snapshot, duplicate.attempts());
    if (!duplicate.check(aliases, valuations)) { return false; }
    for (auto removed : aliases.removed) { if (duplicate.arena->constantValue(removed) != 0) { return false; } }
    Fixture native(context, input, 2);
    native.edge(0, 1); native.fixed.push_back({0, 1});
    if (!native.build()) { return false; }
    auto fixed = fs::reduceCertifiedPartial(native.snapshot, native.attempts());
    if (!native.check(fixed, valuations) || native.value(fixed.removed[0], 0) != 1) { return false; }
    Fixture unknown(context, input, 3);
    unknown.edge(0, 1); unknown.edge(1, 2); unknown.edge(0, 2);
    if (!unknown.build(true)) { return false; }
    auto unavailable = fs::reduceCertifiedPartial(unknown.snapshot, unknown.attempts());
    return unknown.check(unavailable, valuations) && unknown.arena->constantValue(unavailable.removed[2]) == 0;
}
bool guardedTemplates(MLIRContext& context, const pto::SyncInput& input, uint64_t& valuations)
{
    Fixture exclusive(context, input, 3);
    exclusive.edge(0, 1, 2); exclusive.edge(1, 2, 3); exclusive.edge(0, 2);
    if (!exclusive.build()) { return false; }
    auto separate = fs::reduceCertifiedPartial(exclusive.snapshot, exclusive.attempts());
    if (!exclusive.check(separate, valuations)) { return false; }
    for (unsigned mask = 0; mask < Valuations; ++mask) {
        if (exclusive.value(separate.removed[2], mask) != 0) { return false; }
    }
    Fixture absent(context, input, 3);
    absent.presence[1] = absent.conditions[4];
    absent.edge(0, 1); absent.edge(1, 2); absent.edge(0, 2);
    if (!absent.build()) { return false; }
    auto guarded = fs::reduceCertifiedPartial(absent.snapshot, absent.attempts());
    if (!absent.check(guarded, valuations)) { return false; }
    for (unsigned mask = 0; mask < Valuations; ++mask) {
        if (absent.value(guarded.removed[2], mask) != ((mask >> 1) & 1U)) { return false; }
    }
    Fixture readers(context, input, 4);
    readers.pipes = {0, 1, 1, 0};
    readers.edge(0, 1); readers.edge(0, 2); readers.edge(1, 3); readers.edge(2, 3); readers.edge(0, 3);
    if (!readers.build()) { return false; }
    auto reuse = fs::reduceCertifiedPartial(readers.snapshot, readers.attempts());
    return readers.check(reuse, valuations) && readers.value(reuse.removed[2], 0) == 1 &&
        readers.value(reuse.removed[4], 0) == 1;
}
bool mappedAliases(MLIRContext& context, const pto::SyncInput& input, uint64_t& valuations)
{
    Fixture alias(context, input, 2, 2);
    alias.edge(0, 1, 2); alias.edge(2, 3, 3); alias.edge(0, 1);
    auto ordinal = alias.arena->select(alias.predicates[0], alias.arena->constant(0), alias.arena->constant(1));
    alias.generators[2].source.ordinal = ordinal;
    alias.generators[2].target.ordinal = ordinal;
    if (!alias.build()) { return false; }
    auto result = fs::reduceCertifiedPartial(alias.snapshot, alias.attempts());
    if (!alias.check(result, valuations)) { return false; }
    for (auto removed : result.removed) {
        for (unsigned mask = 0; mask < Valuations; ++mask) {
            if (alias.value(removed, mask) != 0) { return false; }
        }
    }
    Fixture visit(context, input, 2, 2);
    for (auto& present : visit.presence) { present = visit.conditions[2]; }
    visit.edge(0, 1); visit.edge(1, 2); visit.edge(0, 2);
    if (!visit.build()) { return false; }
    auto cross = fs::reduceCertifiedPartial(visit.snapshot, visit.attempts());
    if (!visit.check(cross, valuations)) { return false; }
    for (unsigned mask = 0; mask < Valuations; ++mask) {
        if (visit.value(cross.removed[2], mask) != (mask & 1U)) { return false; }
    }
    Fixture order(context, input, 3);
    order.edge(0, 1); order.edge(1, 2); order.edge(0, 2);
    if (!order.build(false, true)) { return false; }
    auto unknown = fs::reduceCertifiedPartial(order.snapshot, order.attempts());
    return order.check(unknown, valuations) && order.arena->constantValue(unknown.removed[2]) == 0;
}
bool shiftedWitness(MLIRContext& context, const pto::SyncInput& input, uint64_t& valuations)
{
    Fixture fixture(context, input, 2, 3);
    auto& e = *fixture.arena;
    auto ordinal = e.select(fixture.predicates[0], e.constant(0), e.constant(1));
    auto next = e.add(ordinal, e.constant(1));
    // The producer explicitly certifies nonwrapping forward shifts. A real
    // compact producer supplies its own uniform query; this finite oracle
    // independently checks both shifted visits and a truncated final visit.
    auto forward = e.lt(ordinal, next);
    fixture.presence[4] = fixture.conditions[4];
    fixture.edge(0, 1); fixture.edge(1, 2); fixture.edge(0, 2);
    for (auto& edge : fixture.generators) { edge.source.ordinal = ordinal; }
    fixture.generators[0].target.ordinal = ordinal;
    for (std::size_t i = 1; i < fixture.generators.size(); ++i) {
        fixture.generators[i].target.ordinal = next;
        fixture.generators[i].active = forward;
    }
    if (!fixture.build()) { return false; }
    auto witness = fixture.event(1, fs::PeriodicEventKind::Start);
    witness.ordinal = ordinal;
    auto result = fs::reduceCertifiedPartial(fixture.snapshot,
        {{fixture.snapshot, 2, fixture.conditions[1], witness}});
    if (!fixture.check(result, valuations)) { return false; }
    for (unsigned mask = 0; mask < Valuations; ++mask) {
        const bool targetExists = (mask & 1U) || (mask & 2U);
        if (fixture.value(result.removed[2], mask) != targetExists) { return false; }
    }
    return true;
}
bool rejected(const fs::CertifiedPartialReduction& result)
{
    return !result.error.empty() && result.retained.empty() && result.removed.empty();
}
bool validationAndRetry(MLIRContext& context, const pto::SyncInput& input, uint64_t& valuations)
{
    Fixture fixture(context, input, 3);
    fixture.edge(0, 1, 2); fixture.edge(1, 2, 4); fixture.edge(0, 2);
    if (!fixture.build()) { return false; }
    auto cache = std::make_shared<std::vector<Id>>();
    auto base = fixture.snapshot->actualPaths.regional().reachability;
    fs::RegionalOrderQueries queries;
    queries.reachability = [base, cache](fs::RegionalEvent a, fs::RegionalEvent b) -> std::optional<Id> {
        auto answer = base(a, b);
        if (answer) { cache->push_back(*answer); }
        return answer;
    };
    std::string error;
    auto view = fs::makeRegionalOrderView(fixture.context, std::move(queries), error);
    if (failed(view)) { return false; }
    auto original = fixture.snapshot;
    fixture.snapshot = std::make_shared<const fs::CertifiedPartialGraph>(fs::CertifiedPartialGraph{
        *view, original->nativePaths, original->generators, original->provenance});
    auto attempts = fixture.attempts();
    attempts.push_back({original, 0, fixture.conditions[1], std::nullopt});
    if (!rejected(fs::reduceCertifiedPartial(fixture.snapshot, attempts)) || cache->empty()) { return false; }
    for (auto id : *cache) { if (id >= fixture.arena->size() || !fixture.arena->isBoolean(id)) { return false; } }
    attempts.pop_back();
    if (!fixture.check(fs::reduceCertifiedPartial(fixture.snapshot, attempts), valuations)) { return false; }
    auto invalid = attempts.front();
    invalid.generator = fixture.generators.size();
    if (!rejected(fs::reduceCertifiedPartial(fixture.snapshot, {invalid}))) { return false; }
    invalid = attempts.front(); invalid.guard = fixture.arena->constant(1);
    if (!rejected(fs::reduceCertifiedPartial(fixture.snapshot, {invalid}))) { return false; }
    invalid = attempts.front(); invalid.intermediate = fixture.event(1, fs::PeriodicEventKind::Start);
    invalid.intermediate->ordinal = fixture.conditions[1];
    if (!rejected(fs::reduceCertifiedPartial(fixture.snapshot, {invalid}))) { return false; }
    auto changed = std::make_shared<fs::CertifiedPartialGraph>(*fixture.snapshot);
    changed->generators[1].record = changed->generators[0].record;
    if (!rejected(fs::reduceCertifiedPartial(changed, {}))) { return false; }
    changed = std::make_shared<fs::CertifiedPartialGraph>(*fixture.snapshot);
    std::swap(changed->generators[2].source, changed->generators[2].target);
    changed->generators[2].source.kind = fs::PeriodicEventKind::Completion;
    changed->generators[2].target.kind = fs::PeriodicEventKind::Start;
    if (!rejected(fs::reduceCertifiedPartial(changed, {}))) { return false; }
    Fixture other(context, input, 3);
    if (!other.build()) { return false; }
    changed = std::make_shared<fs::CertifiedPartialGraph>(*fixture.snapshot);
    changed->nativePaths = other.snapshot->nativePaths;
    if (!rejected(fs::reduceCertifiedPartial(changed, {}))) { return false; }
    changed = std::make_shared<fs::CertifiedPartialGraph>(*fixture.snapshot);
    changed->provenance = other.snapshot->provenance;
    if (!rejected(fs::reduceCertifiedPartial(changed, {}))) { return false; }
    fs::RegionalOrderQueries malformed;
    malformed.reachability = [e = fixture.arena](fs::RegionalEvent, fs::RegionalEvent) -> std::optional<Id> {
        return e->constant(1); // Integer answers are not Boolean proof predicates.
    };
    auto badView = fs::makeRegionalOrderView(fixture.context, std::move(malformed), error);
    if (failed(badView)) { return false; }
    changed = std::make_shared<fs::CertifiedPartialGraph>(*fixture.snapshot);
    changed->nativePaths = *badView;
    if (!rejected(fs::reduceCertifiedPartial(changed, {{changed, 2, fixture.conditions[1], std::nullopt}}))) {
        return false;
    }
    return rejected(fs::reduceCertifiedPartial({}, {}));
}
} // namespace
int runCertifiedPartialReductionChecks()
{
    MLIRContext context;
    context.disableMultithreading();
    context.loadDialect<func::FuncDialect>();
    Builder builder(&context);
    OwningOpRef<func::FuncOp> function(func::FuncOp::create(
        builder.getUnknownLoc(), "partial_reduction_checks", builder.getFunctionType({}, {})));
    pto::SyncInput input;
    if (failed(input.build(*function))) { return 1; }
    uint64_t valuations = 0;
    if (!exhaustive(context, input, valuations) || !cornerCases(context, input, valuations) ||
        !guardedTemplates(context, input, valuations) || !mappedAliases(context, input, valuations) ||
        !shiftedWitness(context, input, valuations) || !validationAndRetry(context, input, valuations)) {
        llvm::errs() << "certified partial reduction graph oracle failed\n";
        return 1;
    }
    llvm::outs() << "certified partial reduction: " << valuations << " independent graph valuations passed\n";
    return 0;
}
