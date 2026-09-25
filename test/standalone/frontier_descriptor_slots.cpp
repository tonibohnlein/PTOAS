// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#include "PTO/Transforms/FrontierSynch/OriginalDescriptorSlots.h"
#include <cstdlib>
#include <iostream>
#include <set>

using namespace mlir::pto::frontiersynch;
using Slots = OriginalDescriptorSlots;
using Form = DescriptorBoundary::Form;
using Direction = DescriptorBoundary::Direction;
using Status = DescriptorObservation::Status;
static std::size_t checks = 0;
#define CHECK(condition) do { ++checks; if (!(condition)) { \
    std::cerr << __FILE__ << ':' << __LINE__ << ": " #condition "\n"; std::exit(1); } } while (false)

struct Fixture {
    Slots slots;
    OriginalProgramVersion version = OriginalProgramVersion::fresh();
    DescriptorBoundary boundary(Direction direction, bool cover, std::size_t op = 1)
    {
        DescriptorBoundary b;
        b.direction = direction;
        b.form = cover ? Form::Covering : Form::Exact;
        b.multiplicity = cover ? DescriptorBoundary::Multiplicity::OncePerInterval :
                                 DescriptorBoundary::Multiplicity::ExactlyParticipating;
        b.interval.query.version = version;
        b.interval.query.selector.cell = 7;
        b.interval.query.start = {0, OriginalCut::Before};
        b.interval.query.stop = {100, OriginalCut::Before};
        b.interval.query.occurrence.stopVisit = OriginalOccurrenceContext::StopVisit::FirstReach;
        b.intervalQualified = b.referenceCoverage = true;
        DescriptorEndpoint e;
        e.cut = {op, direction == Direction::Source ? OriginalCut::After : OriginalCut::Before};
        e.legal = true;
        e.observation.status = Status::Available;
        b.endpoints.push_back(e);
        if (cover) {
            b.mayExecuteWithoutAccess = true;
            b.extraWork.push_back(b.interval);
            b.noHit = DescriptorPredicate{DescriptorPredicate::Domain::Obligation, 17, 99};
            b.continuationCases.bypass = true;
        }
        return b;
    }
    DescriptorLeafFacts facts(std::size_t obligation = 1)
    {
        DescriptorLeafFacts f;
        f.obligations.push_back({DescriptorFactRef::Kind::Obligation, 42, obligation});
        f.matchingQualified = true;
        f.occurrenceMatching.push_back({DescriptorFactRef::Kind::Occurrence, 42, obligation});
        return f;
    }
    Slots::Id add(DescriptorLeafFacts f, DescriptorBoundary es, DescriptorBoundary cs,
                  DescriptorBoundary et, DescriptorBoundary ct)
    {
        const auto s = slots.boundary(es), sc = slots.boundary(cs), t = slots.boundary(et), tc = slots.boundary(ct);
        return slots.leaf(std::move(f), {s, sc, t, tc});
    }
    Slots::Id leaf(std::size_t id = 1, bool badSource = false, bool badTarget = false)
    {
        auto es = boundary(Direction::Source, false, id), cs = boundary(Direction::Source, true, id + 1);
        auto et = boundary(Direction::Target, false, id + 10), ct = boundary(Direction::Target, true, id + 9);
        if (badSource) { es.endpoints[0].observation.status = Status::NotObservableHere; }
        if (badTarget) { et.endpoints[0].observation.status = Status::NotObservableHere; }
        return add(facts(id), es, cs, et, ct);
    }
};
static Slots::Id slot(const std::vector<Slots::SlotRoot>& roots, unsigned s)
{
    for (const auto& root : roots) { if (root.aliases & (1u << s)) { return root.root; } }
    CHECK(false); return 0;
}
static std::set<std::size_t> obligations(const Slots& slots, Slots::Id root, std::uint64_t valuation)
{
    std::vector<Slots::Id> todo{root};
    std::set<std::size_t> result;
    while (!todo.empty()) {
        const auto id = todo.back(); todo.pop_back();
        const auto& node = slots.node(id);
        if (node.kind == Slots::Kind::Leaf) {
            for (const auto& ref : slots.facts(node.facts).obligations) { result.insert(ref.index); }
        } else if (node.kind == Slots::Kind::Both) {
            todo.push_back(node.left); todo.push_back(node.right);
        } else if (node.kind == Slots::Kind::Choose) {
            CHECK(node.condition.root < 64);
            todo.push_back(valuation & (std::uint64_t(1) << node.condition.root) ? node.left : node.right);
        }
    }
    return result;
}
static void directionalSlots()
{
    for (unsigned bad = 0; bad < 4; ++bad) {
        Fixture f;
        const auto input = f.leaf(1, bad & 1, bad & 2);
        const auto roots = f.slots.form(input);
        CHECK(roots.size() == (bad == 3 ? 1 : bad ? 2 : 3));
        for (const auto& entry : roots) { CHECK(entry.fullyDescribed); }
        for (unsigned s = 0; s < 3; ++s) {
            const auto r = slot(roots, s);
            const auto& node = f.slots.node(r);
            CHECK(f.slots.getBoundary(node.source).form == ((bad & 1) || s == 1 ? Form::Covering : Form::Exact));
            CHECK(f.slots.getBoundary(node.target).form == ((bad & 2) || s == 2 ? Form::Covering : Form::Exact));
            CHECK(f.slots.facts(node.facts).obligations == f.facts().obligations);
            CHECK(f.slots.leafComponentsQualified(r));
        }
        // The unused exact query remains available, with its own obstruction.
        if (bad & 1) { CHECK(!f.slots.getBoundary(0).qualified()); }
        CHECK(f.slots.freeze());
    }
}
static void choicesDoNotCross()
{
    Fixture f;
    const auto a = f.leaf(11, false, true), b = f.leaf(22, true, false);
    DescriptorPredicate g{DescriptorPredicate::Domain::Obligation, 123, 0};
    const auto input = f.slots.choose(g, a, b);
    const auto roots = f.slots.form(input);
    CHECK(roots.size() <= 3);
    for (unsigned s = 0; s < 3; ++s) {
        const auto root = slot(roots, s);
        CHECK(f.slots.node(root).kind == Slots::Kind::Choose);
        CHECK(f.slots.node(root).condition == g);
        CHECK(obligations(f.slots, root, 1) == std::set<std::size_t>{11});
        CHECK(obligations(f.slots, root, 0) == std::set<std::size_t>{22});
    }
    const auto& preferred = f.slots.node(slot(roots, 0));
    const auto& aNode = f.slots.node(preferred.left);
    const auto& bNode = f.slots.node(preferred.right);
    CHECK(f.slots.getBoundary(aNode.source).form == Form::Exact);
    CHECK(f.slots.getBoundary(aNode.target).form == Form::Covering);
    CHECK(f.slots.getBoundary(bNode.source).form == Form::Covering);
    CHECK(f.slots.getBoundary(bNode.target).form == Form::Exact);
    CHECK(f.slots.freeze());
}
static void independentReaders()
{
    Fixture f;
    Slots::Id root = 0;
    constexpr unsigned readers = 10;
    for (unsigned i = 0; i < readers; ++i) {
        auto one = f.slots.choose({DescriptorPredicate::Domain::Obligation, 7, i}, f.leaf(i + 1, i & 1, i & 2), 0);
        root = f.slots.both(root, one);
    }
    const auto roots = f.slots.form(root);
    CHECK(roots.size() <= 3);
    for (unsigned mask = 0; mask < (1u << readers); ++mask) {
        std::set<std::size_t> expected;
        for (unsigned i = 0; i < readers; ++i) { if (mask & (1u << i)) { expected.insert(i + 1); } }
        for (unsigned s = 0; s < 3; ++s) { CHECK(obligations(f.slots, slot(roots, s), mask) == expected); }
    }
    CHECK(f.slots.freeze());
    const auto& work = f.slots.stats();
    CHECK(work.formationVisits == work.inputs);
    CHECK(work.nodes <= 3 * work.inputs);
}
static void unavailableAndMissing()
{
    Fixture f;
    auto es = f.boundary(Direction::Source, false), cs = f.boundary(Direction::Source, true);
    auto et = f.boundary(Direction::Target, false), ct = f.boundary(Direction::Target, true);
    es.endpoints[0].observation.status = Status::NotObservableHere;
    cs.form = Form::Unresolved;
    cs.unresolved = {"source cover needs step 11"};
    const auto roots = f.slots.form(f.add(f.facts(), es, cs, et, ct));
    CHECK(roots.size() == 3);
    for (const auto& entry : roots) { CHECK(!entry.fullyDescribed); }
    const auto& preferred = f.slots.node(slot(roots, 0));
    CHECK(!f.slots.leafComponentsQualified(slot(roots, 0)));
    CHECK(f.slots.getBoundary(preferred.target).qualified());
    CHECK(f.slots.getBoundary(preferred.source).endpoints[0].observation.status == Status::NotObservableHere);
    for (unsigned s = 0; s < 3; ++s) { CHECK(obligations(f.slots, slot(roots, s), 0) == std::set<std::size_t>{1}); }
    CHECK(f.slots.freeze());
}
static void unresolvedArmRetainsCoverage()
{
    Fixture f;
    const auto known = f.leaf(11);
    auto es = f.boundary(Direction::Source, false), cs = f.boundary(Direction::Source, true);
    auto et = f.boundary(Direction::Target, false), ct = f.boundary(Direction::Target, true);
    es.form = cs.form = Form::Unresolved;
    es.unresolved = cs.unresolved = {"source boundary premise missing"};
    const auto unknown = f.add(f.facts(22), es, cs, et, ct);
    const auto input = f.slots.choose({DescriptorPredicate::Domain::Obligation, 7, 0}, known, unknown);
    const auto roots = f.slots.form(input);
    for (const auto& entry : roots) {
        CHECK(!entry.fullyDescribed);
        CHECK(obligations(f.slots, entry.root, 1) == std::set<std::size_t>{11});
        CHECK(obligations(f.slots, entry.root, 0) == std::set<std::size_t>{22});
    }
    CHECK(f.slots.freeze());
}
static void supportAndExtraWork()
{
    Fixture f;
    auto es = f.boundary(Direction::Source, false), cs = f.boundary(Direction::Source, true);
    auto et = f.boundary(Direction::Target, false), ct = f.boundary(Direction::Target, true);
    const DescriptorFactRef prerequisite{DescriptorFactRef::Kind::CompletionPrerequisite, 19, 3};
    es.endpoints[0].observation.status = Status::NeedsCompletion;
    es.endpoints[0].observation.prerequisites = {prerequisite};
    es.endpoints[0].observation.independentlyDischargeable = true;
    auto facts = f.facts();
    facts.support.push_back({{DescriptorFactRef::Kind::SupportRole, 19, 9}, prerequisite, -3, {}});
    facts.observations = {prerequisite};
    const auto roots = f.slots.form(f.add(facts, es, cs, et, ct));
    const auto& preferred = f.slots.node(slot(roots, 0));
    CHECK(f.slots.getBoundary(preferred.source).form == Form::Exact);
    CHECK(f.slots.facts(preferred.facts).support == facts.support);
    CHECK(f.slots.facts(preferred.facts).observations == facts.observations);
    const auto& cover = f.slots.getBoundary(f.slots.node(slot(roots, 1)).source);
    CHECK(cover.extraWork == cs.extraWork);
    CHECK(cover.mayExecuteWithoutAccess);
    CHECK(cover.noHit == cs.noHit);
    CHECK(cover.continuationCases.bypass);
    CHECK(f.slots.getBoundary(preferred.source).endpoints[0].observation.prerequisites[0] == prerequisite);
    // No selected completion is created by retaining this prerequisite.
    CHECK(f.slots.freeze());

    Fixture blocked;
    es.interval.query.version = cs.interval.query.version = et.interval.query.version = ct.interval.query.version = blocked.version;
    es.endpoints[0].observation.independentlyDischargeable = false;
    const auto fallback = blocked.slots.form(blocked.add(facts, es, cs, et, ct));
    CHECK(blocked.slots.getBoundary(blocked.slots.node(slot(fallback, 0)).source).form == Form::Covering);
    CHECK(blocked.slots.freeze());
}
static void missingNoHitAndIdentity()
{
    Fixture f;
    auto es = f.boundary(Direction::Source, false), cs = f.boundary(Direction::Source, true);
    auto et = f.boundary(Direction::Target, false), ct = f.boundary(Direction::Target, true);
    es.form = cs.form = et.form = ct.form = Form::NoHit;
    const auto a = f.add(f.facts(55), es, cs, et, ct);
    CHECK(f.add(f.facts(55), es, cs, et, ct) == a);
    const auto roots = f.slots.form(a);
    for (unsigned s = 0; s < 3; ++s) {
        CHECK(!f.slots.leafComponentsQualified(slot(roots, s)));
        CHECK(obligations(f.slots, slot(roots, s), 0) == std::set<std::size_t>{55});
    }
    for (const auto& entry : roots) { CHECK(!entry.fullyDescribed); }
    // A second witness/obligation is not erased by coincident cuts.
    const auto b = f.add(f.facts(56), es, cs, et, ct);
    CHECK(a != b);
    const auto both = f.slots.form(f.slots.both(a, b));
    CHECK(obligations(f.slots, slot(both, 0), 0) == (std::set<std::size_t>{55, 56}));
    CHECK(f.slots.freeze());
}
static void freezeAndInputValidation()
{
    Fixture f;
    const auto a = f.leaf();
    const auto roots = f.slots.form(a);
    CHECK(f.slots.freeze());
    const auto oldNodes = f.slots.stats().nodes, oldVisits = f.slots.stats().formationVisits;
    // Simulate repeated unsuccessful binding probes: the returned repertoire
    // exposes const reads, and even an illicit formation call is refused.
    for (unsigned i = 0; i < 100; ++i) {
        CHECK(f.slots.form(a).empty());
        CHECK(f.slots.both(a, a) == NoControlId);
        CHECK(f.slots.boundary(f.boundary(Direction::Source, true)) == NoControlId);
        CHECK(obligations(f.slots, slot(roots, 0), 0) == std::set<std::size_t>{1});
    }
    CHECK(f.slots.complete());
    CHECK(f.slots.stats().nodes == oldNodes);
    CHECK(f.slots.stats().formationVisits == oldVisits);
    Fixture bad;
    auto wrong = bad.boundary(Direction::Target, true);
    CHECK(bad.add(bad.facts(), wrong, wrong, wrong, wrong) == NoControlId);
    CHECK(!bad.slots.freeze());
    CHECK(!bad.slots.reason().empty());
    Slots empty;
    const auto noDemand = empty.form(0);
    CHECK(noDemand.size() == 1);
    CHECK(!noDemand.front().fullyDescribed);
    CHECK(empty.freeze());
}
static void largeSharedFormation()
{
    Fixture f;
    Slots::Id root = 0;
    constexpr unsigned readers = 4096;
    for (unsigned i = 0; i < readers; ++i) {
        auto optional = f.slots.choose({DescriptorPredicate::Domain::Obligation, 7, i}, f.leaf(i + 1), 0);
        root = f.slots.both(root, optional);
    }
    const auto roots = f.slots.form(root);
    CHECK(roots.size() == 3);
    CHECK(f.slots.freeze());
    CHECK(f.slots.stats().formationVisits == f.slots.stats().inputs);
    CHECK(f.slots.stats().inputs <= 3 * readers + 1);
    CHECK(f.slots.stats().nodes <= 9 * readers + 1);
    CHECK(f.slots.stats().facts == readers);
    std::cout << "4096 optional readers: " << f.slots.stats().inputs << " input nodes, "
              << f.slots.stats().nodes << " descriptor nodes, 3 roots; no valuations enumerated\n";
}
int main()
{
    directionalSlots(); choicesDoNotCross(); independentReaders(); unavailableAndMissing();
    unresolvedArmRetainsCoverage(); supportAndExtraWork(); missingNoHitAndIdentity();
    freezeAndInputValidation(); largeSharedFormation();
    std::cout << "descriptor-slot checks passed: " << checks << '\n';
}
