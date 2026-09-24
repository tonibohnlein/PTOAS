// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#include "PTO/Transforms/FrontierSynch/OriginalObligations.h"
#include <cstdlib>
#include <iostream>
#include <set>
#include <new>

namespace fs = mlir::pto::frontiersynch;
#define CHECK(x)                                                            \
    do {                                                                    \
        if (!(x)) {                                                         \
            std::cerr << "check failed: " #x << " at " << __LINE__ << '\n'; \
            std::abort();                                                   \
        }                                                                   \
    } while (false)
using K = fs::OriginalObligationKey::Kind;
using N = fs::FactoredUseNode;
using S = fs::ObligationMembership::Status;

struct Expression {
    fs::FactoredUseResult graph;
    std::size_t incoming;
    std::vector<std::vector<std::size_t>> readIncidences;
    Expression()
    {
        graph.cell = 0;
        graph.complete = true;
        graph.arena = std::make_shared<fs::FactoredUseArena>(fs::FactoredUseFrame{});
        incoming = graph.arena->incoming(0, N::Boundary::Entry, N::Role::Writer);
    }
    std::size_t access(std::size_t op)
    {
        auto effect = std::make_shared<fs::FactoredUseAccess>();
        effect->operation = op;
        return graph.arena->origin(effect);
    }
    std::size_t both(std::size_t a, std::size_t b) { return graph.arena->both(a, b, N::Sort::Origins); }
    std::size_t choose(std::size_t guard, std::size_t a, std::size_t b)
    {
        return graph.arena->choose(graph.arena->test({guard, fs::NoFactoredId}), a, b, N::Sort::Origins);
    }
    fs::OriginalObligationFamily family(std::size_t sources, std::size_t target, K kind = K::RAW)
    {
        auto effect = std::make_shared<fs::FactoredUseAccess>();
        effect->operation = target;
        const auto hazard = kind == K::RAW ? N::Hazard::RAW : kind == K::WAR ? N::Hazard::WAR : N::Hazard::WAW;
        const auto demand = graph.arena->demand(sources, effect, hazard);
        fs::OriginalObligationFamily f;
        f.key.cell = 0;
        f.key.kind = kind;
        f.key.consumerOriginal = target;
        f.key.consumerOperation = target;
        f.key.owner = 100;
        f.key.occurrences = fs::OriginalObligationKey::Occurrences::FixedUseProjection;
        f.key.start = {fs::ObligationCut::Kind::OwnerEntry, 100};
        f.key.stop = {fs::ObligationCut::Kind::PayloadBefore, target};
        f.representation = fs::OriginalObligationFamily::Representation::Factored;
        f.expression = &graph;
        f.sources = sources;
        f.demandNode = demand ? demand : fs::NoObligationIndex;
        f.guardsIdentified = true;
        return f;
    }
};

bool enabled(
    const fs::OriginalObligations& index, fs::OriginalObligationFamilyId id, fs::ObligationOrigin source,
    const std::map<std::size_t, bool>& values)
{
    const auto member = index.membership(id, source);
    CHECK(member.status != S::Invalid);
    const auto result = index.predicates().evaluate(member.condition, [&](std::size_t g) -> std::optional<bool> {
        const auto found = values.find(g);
        return found == values.end() ? std::nullopt : std::optional<bool>(found->second);
    });
    CHECK(result.has_value());
    return *result;
}

void conditionalWriterAndStableIdentity()
{
    Expression e;
    const auto w0 = e.access(0), w1 = e.access(2);
    const auto selected = e.choose(10, w1, w0);
    auto reader = e.family(selected, 3);
    // The final full write changes outgoing provenance, not the earlier demand.
    e.graph.finalWriters = e.access(5);
    auto lastWrite = e.family(selected, 5, K::WAW);
    e.readIncidences.resize(6);
    e.readIncidences[3] = {0, 2};
    reader.consumerEffects = &e.readIncidences[3];
    fs::OriginalObligations index([](fs::FactoredGuardIdentity g) { return g.value; }, {}, {});
    const auto id = index.add(reader);
    CHECK(index.add(reader) == id);
    const auto second = index.add(lastWrite);
    index.freeze();
    CHECK(index.complete() && index.stats().families == 2);
    // Descriptors are merely references; duplication creates no residual units.
    struct Descriptor {
        fs::OriginalObligationId obligation;
        std::size_t sourceCut;
    };
    const auto obligation = index.membership(id, {0, false}).id();
    const std::vector<Descriptor> descriptors{{obligation, 0}, {obligation, 1}, {obligation, 1}};
    for (const auto& d : descriptors) {
        CHECK(d.obligation == obligation);
    }
    CHECK(index.atOriginalSite(3).size() == 1);
    CHECK(index.membership(obligation).id() == obligation);
    CHECK(index.membership(id, {2, false}).id() != obligation);
    CHECK(index.get(id)->consumerEffects->size() == 2);
    CHECK(index.get(id)->key.sourceRole() == fs::OriginalObligationKey::Role::Writer);
    for (bool g : {false, true}) {
        // Independent concrete scan of W0; [g:W1]; B; W2.
        std::size_t lastWriter = 0;
        if (g) {
            lastWriter = 2;
        }
        const auto expectedAtB = lastWriter;
        const auto expectedAtW2 = lastWriter;
        lastWriter = 5;
        CHECK(lastWriter == 5);
        for (std::size_t candidate : {0, 2, 3, 5}) {
            CHECK(enabled(index, id, {candidate, false}, {{10, g}}) == (candidate == expectedAtB));
            CHECK(enabled(index, second, {candidate, false}, {{10, g}}) == (candidate == expectedAtW2));
        }
        CHECK(!enabled(index, id, fs::ObligationOrigin::entry(), {{10, g}}));
    }
    CHECK(index.stats().enumeratedOrigins == 0);
    const auto listed = index.origins(id);
    CHECK(listed.complete && listed.members.size() == 2);
    CHECK(index.witness(id, {0, false}).status == S::Guarded);
    CHECK(index.stats().witnessesRequested == 1);
    CHECK(index.add(reader).universe == nullptr); // frozen
    fs::OriginalObligations other({}, {}, {});
    other.freeze();
    CHECK(other.membership(id, {0, false}).status == S::Invalid);
}

void sharedGuardAndIncoming()
{
    Expression e;
    const auto w = e.access(1);
    const auto first = e.choose(10, w, e.incoming);
    // Two original if sites use the SAME defining Boolean value. On g, the
    // second branch reaches w; on !g it reaches the explicit incoming writer.
    const auto again = e.choose(11, first, e.incoming);
    auto read = e.family(again, 2);
    auto write = e.family(0, 3, K::WAR);
    write.incomingReaderSurvival = first;
    fs::OriginalObligations index([](fs::FactoredGuardIdentity) { return 7; }, {}, {});
    const auto r = index.add(read), u = index.add(write);
    index.freeze();
    CHECK(index.complete());
    for (bool g : {false, true}) {
        CHECK(enabled(index, r, {1, false}, {{7, g}}) == g);
        CHECK(enabled(index, r, fs::ObligationOrigin::entry(), {{7, g}}) == !g);
        CHECK(enabled(index, u, fs::ObligationOrigin::entry(), {{7, g}}) == !g);
        CHECK(!enabled(index, u, {1, false}, {{7, g}}));
    }
    CHECK(index.membership(r, fs::ObligationOrigin::entry()).status == S::Conservative);
    CHECK(index.membership(r, {12, true}).status == S::Invalid);
    CHECK(!index.predicates().evaluate(index.membership(r, {1, false}).condition, {}));
    const auto incoming = index.origins(u);
    CHECK(incoming.complete && incoming.members.size() == 1);
}

void weakWritesAndRMW()
{
    Expression e;
    const auto oldWriter = e.access(0), oldReader = e.access(1), partial = e.access(2);
    const auto retained = e.both(oldWriter, partial);
    auto raw = e.family(retained, 3, K::RAW);
    auto waw = e.family(retained, 3, K::WAW);
    auto war = e.family(oldReader, 3, K::WAR);
    // This operation is RMW; its new access is never an incoming self-demand.
    e.graph.finalWriters = e.access(3);
    fs::OriginalObligations index({}, {}, {});
    const auto a = index.add(raw), b = index.add(waw), c = index.add(war);
    index.freeze();
    CHECK(index.complete());
    CHECK(enabled(index, a, {0, false}, {}) && enabled(index, a, {2, false}, {}));
    CHECK(enabled(index, b, {0, false}, {}) && enabled(index, c, {1, false}, {}));
    for (auto id : {a, b, c}) {
        CHECK(!enabled(index, id, {3, false}, {}));
    }
}

void scopeIdentityAndUnresolvedMarginals()
{
    std::size_t queries = 0, enumerations = 0;
    fs::OriginalObligations index(
        {},
        [&](const fs::OriginalObligationKey& k, fs::ObligationOrigin o) -> std::optional<bool> {
            ++queries;
            CHECK(k.owner == 100 || k.owner == 101);
            if (o.incoming) {
                return true;
            }
            if (o.operation == 0) {
                return true;
            }
            if (o.operation == 1) {
                return false;
            }
            return std::nullopt;
        },
        [&](const fs::OriginalObligationKey&) {
            ++enumerations;
            return std::vector<fs::ObligationOrigin>{{0, false}, {2, false}, fs::ObligationOrigin::entry()};
        });
    fs::OriginalObligationFamily f;
    f.key.cell = 0;
    f.key.consumerOriginal = 4;
    f.key.consumerOperation = 4;
    f.key.owner = 100;
    const auto a = index.add(f);
    f.key.owner = 101;
    const auto b = index.add(f);
    f.key.includeStop = true;
    const auto c = index.add(f);
    f.key.stop = {fs::ObligationCut::Kind::OwnerExit, 101};
    const auto d = index.add(f);
    f.key.originalVersion = fs::OriginalProgramVersion::fresh();
    const auto versioned = index.add(f);
    CHECK(a != b && b != c && c != d && d != versioned);
    index.freeze();
    CHECK(index.complete());
    CHECK(queries == 0 && enumerations == 0);
    CHECK(index.membership(a, {0, false}).status == S::Conservative);
    CHECK(queries == 1 && enumerations == 0);
    CHECK(index.membership(a, {1, false}).status == S::Excluded);
    CHECK(index.membership(a, {2, false}).status == S::Conservative);
    CHECK(index.origins(a).members.size() == 3 && enumerations == 1);
    fs::OriginalObligations missing({}, {}, {});
    const auto m = missing.add(f);
    missing.freeze();
    CHECK(missing.membership(m, {0, false}).status == S::Conservative);
    CHECK(!missing.origins(m).complete); // no false empty population
}

void typedDeadlinesAndImpossibleParticipation()
{
    fs::OriginalObligations index({}, {}, {});
    auto& c = index.formingConditions();
    const auto g = c.atom(2);
    fs::OriginalObligationFamily f;
    f.key.kind = K::Typed;
    f.key.consumerOriginal = 11;
    f.key.typedValue = 42;
    f.key.typedCause = 1;
    f.representation = fs::OriginalObligationFamily::Representation::Typed;
    f.typedSources = {{3, false}, {4, false}, fs::ObligationOrigin::entry()};
    const auto typed = index.add(f);
    f.key.consumerOriginal = 12;
    f.applicability = c.both(g, c.negate(g));
    const auto never = index.add(f);
    index.freeze();
    CHECK(index.complete());
    CHECK(index.atOriginalSite(11).size() == 1 && index.atOperation(11).empty());
    const auto members = index.origins(typed);
    CHECK(members.complete && members.members.size() == 3);
    CHECK(members.members[0].id() != members.members[1].id());
    CHECK(index.membership(never, {3, false}).status == S::Excluded);
    CHECK(index.membership(typed, {6, false}).status == S::Excluded);
}

void independentReaders(std::size_t count, bool enumerateValuations)
{
    Expression e;
    std::size_t root = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const auto reader = e.access(i);
        root = e.both(root, e.choose(i, reader, 0));
    }
    auto family = e.family(root, count, K::WAR);
    fs::OriginalObligations index({}, {}, {});
    const auto id = index.add(family);
    index.freeze();
    CHECK(index.complete());
    CHECK(index.stats().families == 1 && index.stats().membershipQueries == 0);
    CHECK(e.graph.nodes().size() <= 4 * count + 4);
    if (enumerateValuations) {
        for (std::size_t mask = 0; mask < (std::size_t(1) << count); ++mask) {
            std::map<std::size_t, bool> values;
            std::set<std::size_t> concreteReaders;
            for (std::size_t i = 0; i < count; ++i) {
                values[i] = bool(mask & (std::size_t(1) << i));
                if (values[i]) {
                    concreteReaders.insert(i);
                }
            }
            for (std::size_t i = 0; i < count; ++i) {
                CHECK(enabled(index, id, {i, false}, values) == bool(concreteReaders.count(i)));
            }
        }
    } else {
        const auto a = index.membership(id, {0, false});
        const auto b = index.witness(id, {count - 1, false});
        CHECK(a.status == S::Guarded && b.status == S::Guarded);
        CHECK(index.predicates().evaluate(a.condition, [](std::size_t) { return std::optional<bool>(true); }) == true);
        CHECK(index.predicates().size() < 8 * count + 10);
        CHECK(index.stats().originNodesInspected < 5 * e.graph.nodes().size());
    }
    CHECK(index.stats().enumeratedOrigins == 0);
    const auto first = index.membership(id, {0, false}).id();
    const auto last = index.membership(id, {count - 1, false}).id();
    CHECK(first != last); // Independent readers MUST NOT become one residual unit.
    CHECK(index.membership(first).id() == first);
    CHECK(index.witness(last).id() == last);
}

void malformedAndConflictingRegistration()
{
    Expression e;
    auto f = e.family(e.access(0), 1);
    fs::OriginalObligations index({}, {}, {});
    CHECK(index.add(f).universe);
    f.sources = 1; // inconsistent with the original demand node
    CHECK(!index.add(f).universe);
    index.freeze();
    CHECK(!index.complete());
    Expression bad;
    auto invalid = bad.family(bad.access(0), 1);
    invalid.sources = 99;
    fs::OriginalObligations badIndex({}, {}, {});
    CHECK(!badIndex.add(invalid).universe);
    badIndex.freeze();
    CHECK(!badIndex.complete());
}

void unresolvedEnumeration()
{
    fs::OriginalObligations index(
        {}, [](const auto&, fs::ObligationOrigin) -> std::optional<bool> { return std::nullopt; },
        [](const auto&) -> std::optional<std::vector<fs::ObligationOrigin>> { return std::nullopt; });
    fs::OriginalObligationFamily family;
    const auto id = index.add(family);
    index.freeze();
    CHECK(index.membership(id, {0, false}).status == S::Conservative);
    CHECK(!index.origins(id).complete);
}

void nestedConditionsAndStaleIds()
{
    Expression e;
    const auto source = e.access(0);
    const auto nested = e.choose(10, e.choose(11, source, 0), 0);
    fs::OriginalObligations index([](fs::FactoredGuardIdentity g) { return g.value; });
    const auto id = index.add(e.family(nested, 1));
    index.freeze();
    const auto member = index.membership(id, {0, false});
    unsigned innerVisits = 0;
    const auto bypass = index.predicates().evaluate(member.condition, [&](std::size_t g) -> std::optional<bool> {
        if (g == 10) {
            return false;
        }
        ++innerVisits;
        return std::nullopt;
    });
    CHECK(bypass == false && innerVisits == 0);
    fs::OriginalObligations applicabilityIndex([](fs::FactoredGuardIdentity g) { return g.value; });
    auto family = e.family(e.choose(11, source, 0), 2);
    family.applicability = applicabilityIndex.formingConditions().atom(10);
    const auto conditional = applicabilityIndex.add(family);
    applicabilityIndex.freeze();
    innerVisits = 0;
    const auto conditionalMember = applicabilityIndex.membership(conditional, {0, false});
    CHECK(
        applicabilityIndex.predicates().evaluate(
            conditionalMember.condition, [&](std::size_t g) -> std::optional<bool> {
                if (g == 10) {
                    return false;
                }
                ++innerVisits;
                return std::nullopt;
            }) == false);
    CHECK(innerVisits == 0);
    bool current = true;
    fs::OriginalObligations versioned({}, {}, {}, [&] { return current; });
    const auto currentId = versioned.add(e.family(source, 1));
    versioned.freeze();
    CHECK(!versioned.atOperation(1).empty());
    current = false;
    CHECK(versioned.atOperation(1).empty());
    CHECK(versioned.atOriginalSite(1).empty());
    CHECK(versioned.membership(currentId, {0, false}).status == S::Invalid);
    alignas(fs::OriginalObligations) unsigned char buffer[sizeof(fs::OriginalObligations)];
    auto* first = new (buffer) fs::OriginalObligations;
    const auto old = first->add(e.family(source, 1));
    first->freeze();
    first->~OriginalObligations();
    auto* second = new (buffer) fs::OriginalObligations;
    second->add(e.family(source, 1));
    second->freeze();
    CHECK(second->membership(old, {0, false}).status == S::Invalid);
    second->~OriginalObligations();
}

int main()
{
    unresolvedEnumeration();
    nestedConditionsAndStaleIds();
    conditionalWriterAndStableIdentity();
    sharedGuardAndIncoming();
    weakWritesAndRMW();
    scopeIdentityAndUnresolvedMarginals();
    typedDeadlinesAndImpossibleParticipation();
    independentReaders(8, true);
    independentReaders(10000, false);
    malformedAndConflictingRegistration();
    std::cout << "PASS: 8 obligation-model groups; 256 small valuations; 10000-reader shared-DAG stress\n";
}
