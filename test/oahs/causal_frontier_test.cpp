// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/OAHS/CausalFrontier.h"
#include "PTO/Transforms/OAHS/Analysis.h"
#include "PTO/Transforms/OAHS/ObservedPrograms.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>

namespace o = mlir::pto::oahs;
namespace {
unsigned checks = 0;
void check(bool value, unsigned line)
{
    ++checks;
    if (!value) {
        std::cerr << "causal frontier check " << line << '\n';
        std::abort();
    }
}
#define CHECK(x) check(bool(x), __LINE__)
const auto P = o::Pipe::S, Q = o::Pipe::V, R = o::Pipe::M;
o::Program base(std::size_t n, unsigned cells = 2)
{
    o::Program p;
    p.target.contract = "ordinary issue-only test contract";
    p.cells.resize(cells);
    p.operations.resize(n);
    for (auto& op : p.operations) {
        op.complete = true;
        op.pipe = P;
    }
    for (unsigned i = 0; i < o::PipeCount; ++i) {
        p.target.supported[i] = p.target.barriers[i] = true;
        for (unsigned j = 0; j < o::PipeCount; ++j)
            if (i != j)
                p.target.keys[i][j] = {0, 1};
    }
    return p;
}
o::Command pub(o::Pipe a, o::Pipe b, unsigned key = 0) { return {o::Command::Publish, a, b, key}; }
o::Command wait(o::Pipe a, o::Pipe b, unsigned key = 0) { return {o::Command::Acquire, a, b, key}; }
o::Command fence(o::Pipe a) { return {o::Command::Barrier, a, a, 0}; }
o::FrontierState apply(const o::CausalFrontier& f, const o::FrontierState& s, o::Command c, o::Cut cut = 0)
{
    auto step = f.command(s, c, {cut, 0});
    CHECK(step.applied);
    return step.state;
}
o::FrontierState issue(const o::CausalFrontier& f, const o::FrontierState& s, unsigned op)
{
    auto step = f.issue(s, op);
    CHECK(step.applied);
    return step.state;
}
void primitives()
{
    auto p = base(3);
    p.operations[0].accesses = {{0, false, true}};
    p.operations[1] = p.operations[0];
    p.operations[2].pipe = Q;
    p.operations[2].accesses = {{0, true, false}};
    o::CausalFrontier f(p);
    CHECK(f.complete());
    auto s = issue(f, f.initial(), 0);
    auto saved = s;
    s = apply(f, s, pub(P, Q));
    auto bad = f.issue(s, 1);
    CHECK(!bad.applied && bad.failure == o::FrontierFailure::Payload && bad.state == s);
    CHECK(bad.residuals.size() == 1 && bad.residuals[0].sourceWrite);
    // SET observes completion but must NOT fence later source payload issue.
    s = apply(f, s, fence(P));
    s = issue(f, s, 1);
    s = apply(f, s, wait(P, Q));
    CHECK(!f.issue(s, 2).applied); // old snapshot does not cover the fresh write
    CHECK(saved != s && !f.issue(saved, 2).applied);
    // A new, actual prefix can cover the new generation on another key.
    s = apply(f, s, pub(P, Q, 1));
    s = apply(f, s, wait(P, Q, 1));
    s = issue(f, s, 2);
    CHECK(f.exit(s).applied);
    // A fence completes work but leaves notification occupancy intact.
    auto full = apply(f, f.initial(), pub(P, Q));
    full = apply(f, full, fence(P));
    CHECK(f.exit(full).failure == o::FrontierFailure::UnconsumedAtExit);
    CHECK(f.command(full, pub(P, Q), {0, 0}).failure == o::FrontierFailure::PublicationNotEmpty);
    CHECK(f.command(f.initial(), wait(P, Q), {0, 0}).failure == o::FrontierFailure::AcquisitionNotFull);
}
void generations()
{
    auto p = base(3);
    p.operations[0].accesses = p.operations[2].accesses = {{0, false, true}};
    p.operations[1].pipe = Q;
    p.operations[1].accesses = {{0, true, false}};
    o::CausalFrontier f(p);
    auto s = issue(f, f.initial(), 0);
    s = apply(f, s, pub(P, Q));
    s = apply(f, s, wait(P, Q));
    s = apply(f, s, pub(Q, P)); // acknowledges consumption before Q's read
    s = issue(f, s, 1);
    s = apply(f, s, wait(Q, P));
    auto fail = f.issue(s, 2);
    CHECK(!fail.applied && fail.residuals.size() == 1 && !fail.residuals[0].sourceWrite);
    s = apply(f, s, pub(P, Q)); // rearm is proved, but read release is not
    s = apply(f, s, wait(P, Q));
    CHECK(f.command(s, pub(P, Q), {0, 0}).failure == o::FrontierFailure::ConsumptionNotEstablished);
    auto fenced = apply(f, s, fence(P));
    CHECK(f.command(fenced, pub(P, Q), {0, 0}).failure == o::FrontierFailure::ConsumptionNotEstablished);
    s = apply(f, s, pub(Q, P)); // actual new return carries the read and new D
    s = apply(f, s, wait(Q, P));
    s = issue(f, s, 2);
    s = apply(f, s, pub(P, Q));
    CHECK(f.command(s, wait(P, Q), {0, 0}).applied);
}
void sharingAndReaders()
{
    auto p = base(3);
    p.operations[0].accesses = {{0, true, false}, {1, false, true}};
    p.operations[1].pipe = Q; // a payload-free relay is also a valid engine
    p.operations[2].pipe = R;
    p.operations[2].accesses = {{0, false, true}, {1, true, false}};
    o::CausalFrontier f(p);
    auto s = issue(f, f.initial(), 0);
    CHECK(f.issue(s, 2).residuals.size() == 2);
    for (auto c : {pub(P, Q), wait(P, Q), pub(Q, R), wait(Q, R)})
        s = apply(f, s, c);
    CHECK(f.issue(s, 2).applied); // one actual movement receipt covers both roles
    p.operations[0].accesses = p.operations[1].accesses = {{0, true, false}};
    p.operations[2].accesses = {{0, false, true, true}};
    o::CausalFrontier readers(p);
    s = issue(readers, issue(readers, readers.initial(), 0), 1);
    CHECK(readers.issue(s, 2).residuals.size() == 2); // full overwrite does not release either reader
    s = apply(readers, apply(readers, s, pub(P, R)), wait(P, R));
    auto residual = readers.issue(s, 2).residuals;
    CHECK(residual.size() == 1 && residual[0].source == Q);
    s = apply(readers, apply(readers, s, pub(Q, R)), wait(Q, R));
    CHECK(readers.issue(s, 2).applied);
}
void joins()
{
    auto p = base(2);
    p.operations[0].accesses = {{0, false, true}};
    p.operations[1].pipe = Q;
    p.operations[1].accesses = {{0, true, false}};
    o::CausalFrontier f(p);
    auto s = issue(f, f.initial(), 0);
    auto left = apply(f, s, pub(P, Q), 0), right = apply(f, s, pub(P, Q), 1);
    auto joined = f.join(left, right);
    CHECK(joined.applied);
    std::size_t key = 0;
    for (; key < f.keys().size(); ++key)
        if (f.keys()[key].source == P && f.keys()[key].observer == Q)
            break;
    CHECK(joined.state.facts()->events[key].publishers.size() == 2);
    auto acquired = apply(f, joined.state, wait(P, Q));
    CHECK(f.issue(acquired, 1).applied);
    joined = f.join(left, f.initial());
    CHECK(joined.state.facts()->events[key].occupancy == 3);
    const auto publication = 2 * o::PipeCount + key;
    for (const auto& row : joined.state.facts()->reach)
        CHECK(!o::frontierContains(row, publication));
    for (const auto& entry : joined.state.facts()->history.present())
        CHECK(!o::frontierContains(entry.second, publication));
    CHECK(f.command(joined.state, wait(P, Q), {0, 0}).failure == o::FrontierFailure::AcquisitionNotFull);
    CHECK(f.command(joined.state, pub(P, Q), {0, 0}).failure == o::FrontierFailure::PublicationNotEmpty);
    CHECK(f.join({}, left).state == left); // unreachable is not fresh empty
    CHECK(f.join({}, {}).applied && !f.join({}, {}).state.reachable());
    CHECK(f.join(f.join(left, right).state, s).state == f.join(left, f.join(right, s).state).state);
    CHECK(f.join(left, right).state == f.join(right, left).state);
    o::CausalFrontier other(p);
    CHECK(!f.join(left, other.initial()).applied);
    CHECK(!f.issue(other.initial(), 0).applied);
    p.operations[0].accesses.clear(); // immutable original program
    CHECK(f.issue(f.initial(), 0).state.facts()->history.find(1) != nullptr);
}
void closedJoins()
{
    auto p = base(2);
    const auto observer = o::Pipe::MTE1;
    p.operations[0].accesses = {{0, false, true}};
    p.operations[1].pipe = observer;
    p.operations[1].accesses = {{0, true, false}};
    o::CausalFrontier f(p);
    auto source = issue(f, f.initial(), 0);
    auto through = [&](o::Pipe relay) {
        auto s = source;
        for (auto c : {pub(P, relay), wait(P, relay), pub(relay, observer), wait(relay, observer)})
            s = apply(f, s, c);
        return s;
    };
    // The same completion consequence survives distinct witnesses in each arm.
    auto common = f.join(through(Q), through(R)).state;
    CHECK(f.issue(common, 1).applied);
    auto left = apply(f, apply(f, source, pub(P, Q)), wait(P, Q));
    auto right = apply(f, apply(f, source, pub(Q, observer)), wait(Q, observer));
    // Intersect closed branch relations; union then closure would splice a
    // P->Q->observer path that occurs in neither arm.
    CHECK(!f.issue(f.join(left, right).state, 1).applied);
}
o::Region leaf(unsigned i) { return {o::Region::Operation, {}, i}; }
o::Region seq(std::initializer_list<o::Region> c) { return {o::Region::Sequence, c}; }
void structured()
{
    auto p = base(2);
    p.operations[0].accesses = {{0, false, true}};
    p.operations[1].pipe = Q;
    p.operations[1].accesses = {{0, true, false}};
    p.body = seq({{o::Region::For, {leaf(0)}, 0, true}, leaf(1)});
    o::Commands commands(3);
    commands[0] = {fence(P)};
    auto bad = o::checkCausalFrontier(p, commands);
    CHECK(bad.complete && !bad.accepted && bad.cuts.empty() && bad.failure == o::FrontierFailure::Payload);
    commands[1] = {pub(P, Q), wait(P, Q)};
    auto good = o::checkCausalFrontier(p, commands);
    CHECK(good.accepted && good.cuts.size() == 3);
    // Waiting outside a possibly empty publishing loop is not a seeded wait.
    commands[0] = {fence(P), pub(P, Q)};
    commands[1] = {wait(P, Q)};
    CHECK(!o::checkCausalFrontier(p, commands).accepted);
    // The before region of while executes even when after executes zero times.
    p.body = seq({{o::Region::While, {leaf(0), seq({})}}, leaf(1)});
    commands[0] = {fence(P)};
    commands[1].clear();
    CHECK(!o::checkCausalFrontier(p, commands).accepted);
    commands[1] = {pub(P, Q), wait(P, Q)};
    CHECK(o::checkCausalFrontier(p, commands).accepted);
    // Whole loop with a real control-only body-exit cut and no entry key reset.
    o::ObservedControl control;
    control.qualification = "test original counted-loop control";
    control.entry = 0;
    control.exit = 4;
    control.sites.resize(5);
    for (unsigned i = 0; i < 5; ++i) {
        control.observations.push_back({i, {}, true});
        control.sites[i].observation = i;
    }
    control.sites[0].successors = {1, 4};
    control.sites[1].operation = 0;
    control.sites[1].successors = {2};
    control.sites[2].operation = 1;
    control.sites[2].successors = {3};
    control.sites[3].successors = {0};
    p.observed = control;
    commands.assign(5, {});
    commands[2] = {pub(P, Q), wait(P, Q)};
    commands[3] = {pub(Q, P), wait(Q, P)};
    good = o::checkCausalFrontier(p, commands);
    CHECK(good.accepted && good.siteEvaluations > 5);
    commands[3].clear();
    CHECK(!o::checkCausalFrontier(p, commands).accepted);
    p.observed->sites[2].observation = 1;
    CHECK(!o::checkCausalFrontier(p, commands).complete); // original word uniformity
}
void nativeAccumulatorOrdering()
{
    auto p = base(4);
    p.target = mlir::pto::a3SyncProfile(mlir::pto::SyncCore::Cube);
    auto &acc = p.cells[0];
    acc.storage = o::Cell::Storage::CanonicalInterval;
    acc.coordinateSpace = "physical-local";
    acc.ranges = {{0, 131072}};
    acc.domain = o::Cell::Domain::Accumulator;
    p.nativeAccumulatorClasses = 1;
    p.operations[0].pipe = p.operations[1].pipe = o::Pipe::M;
    p.operations[0].accesses = {{0, false, true, false, 0}, {1, true, false}};
    p.operations[1].accesses = {{0, true, true, false, 0}, {1, true, false}};
    p.operations[1].nativeMmadAccumulate = true;
    p.operations[2].pipe = o::Pipe::FIX;
    p.operations[2].accesses = {{0, true, false}};
    p.operations[3].pipe = o::Pipe::MTE1;
    p.operations[3].accesses = {{1, false, true}};
    o::CausalFrontier f(p);
    auto first = issue(f, f.initial(), 0);
    auto second = issue(f, first, 1);
    CHECK(!f.inspect(second, 2).applied); // ACC is not globally complete.
    CHECK(!f.inspect(second, 3).applied); // Operand readers are still pending.
    auto moved = apply(f, second, pub(o::Pipe::M, o::Pipe::FIX));
    moved = apply(f, moved, wait(o::Pipe::M, o::Pipe::FIX));
    CHECK(f.inspect(moved, 2).applied);
    CHECK(!f.inspect(moved, 3).applied);
    o::Commands commands(5);
    const auto report = o::analyze(p, commands);
    CHECK(report.complete);
    CHECK(std::none_of(report.residuals.begin(), report.residuals.end(), [](const auto &r) {
        return r.demand.consumer == 1;
    }));
    CHECK(!report.verified());
    auto initializing = p;
    initializing.operations[1].nativeMmadAccumulate = false;
    o::CausalFrontier fresh(initializing);
    CHECK(!fresh.inspect(issue(fresh, fresh.initial(), 0), 1).applied);
    p.operations[0].accesses[0].nativeAccumulatorClass = o::NoControlId;
    o::CausalFrontier conservative(p);
    CHECK(!conservative.inspect(issue(conservative, conservative.initial(), 0), 1).applied);
    p.operations[0].accesses[0].nativeAccumulatorClass = 0;
    p.cells[0].storage = o::Cell::Storage::OverlapWitness;
    CHECK(!o::CausalFrontier(p).complete());
}
void scopedAccumulatorHistories()
{
    auto p = base(3, 1);
    p.target = mlir::pto::a3SyncProfile(mlir::pto::SyncCore::Cube);
    auto& cell = p.cells[0];
    cell.storage = o::Cell::Storage::CanonicalInterval;
    cell.domain = o::Cell::Domain::Accumulator;
    cell.coordinateSpace = "physical-local";
    cell.ranges = {{0, 131072}};
    p.nativeAccumulatorClasses = 2;
    for (auto& operation : p.operations) {
        operation.pipe = o::Pipe::M;
        operation.accesses = {{0, true, true, false, 0}};
        operation.nativeMmadAccumulate = true;
    }
    p.operations[0].nativeMmadAccumulate = false;
    p.operations[0].accesses[0].read = false;
    o::Commands commands(4);
    CHECK(o::checkCausalFrontier(p, commands).accepted);
    CHECK(o::analyze(p, commands).verified());
    for (auto oldClass : {std::size_t(1), o::NoControlId}) {
        p.operations[0].accesses[0].nativeAccumulatorClass = oldClass;
        o::CausalFrontier frontier(p);
        const auto old = issue(frontier, frontier.initial(), 0);
        CHECK(!frontier.inspect(old, 1).applied);
        CHECK(!o::checkCausalFrontier(p, commands).accepted);
        const auto pending = o::analyze(p, commands);
        CHECK(std::any_of(pending.residuals.begin(), pending.residuals.end(), [](const auto& r) {
            return r.demand.producer == 0 && r.demand.consumer == 2;
        })); // later compatible issuance never hides the old incompatible access
        for (const auto& order : {std::vector<std::size_t>{0, 1}, {1, 0}}) {
            const auto hypothesis = frontier.assumePreviousAccesses(frontier.initial(), order);
            CHECK(hypothesis.applied && !frontier.inspect(hypothesis.state, 2).applied);
        }
        const auto compatible = issue(frontier, frontier.initial(), 1);
        const auto joined = frontier.join(old, compatible);
        CHECK(joined.applied && !frontier.inspect(joined.state, 2).applied);
        const auto skipped = frontier.join(old, frontier.initial());
        CHECK(skipped.applied && !frontier.inspect(skipped.state, 2).applied);
        const auto repaired = apply(frontier, old, fence(o::Pipe::M));
        const auto accumulated = issue(frontier, repaired, 1);
        CHECK(frontier.inspect(accumulated, 2).applied);
        commands[1] = {fence(o::Pipe::M)};
        CHECK(o::checkCausalFrontier(p, commands).accepted);
        CHECK(o::analyze(p, commands).verified());
        commands[1].clear();
    }
    p.operations[0].accesses[0].nativeAccumulatorClass = 0;
    for (bool reverse : {false, true}) {
        auto mixed = p;
        mixed.operations[1].accesses.push_back({0, true, true, false, 1});
        if (reverse) {
            std::reverse(mixed.operations[1].accesses.begin(), mixed.operations[1].accesses.end());
        }
        CHECK(!o::checkCausalFrontier(mixed, commands).accepted);
        CHECK(!o::analyze(mixed, commands).verified());
    }
    auto branch = p;
    branch.operations[1].accesses[0].nativeAccumulatorClass = 1;
    o::Region a, b, consumer, choice;
    a.kind = b.kind = consumer.kind = o::Region::Operation;
    a.operation = 0; b.operation = 1; consumer.operation = 2;
    choice.kind = o::Region::Choice;
    choice.children = {a, b};
    branch.body.children = {choice, consumer};
    commands.assign(o::commandCutCount(branch), {});
    CHECK(!o::checkCausalFrontier(branch, commands).accepted);
    CHECK(!o::analyze(branch, commands).verified());
    auto periodic = p;
    periodic.operations.resize(2);
    periodic.operations[0].nativeMmadAccumulate = true;
    periodic.cells.push_back(periodic.cells[0]);
    periodic.cells[1].ranges = {{131072, 131072}};
    const auto unchanged = o::makePeriodicLoop(periodic, 2, {{0, 0, {0, 0}}, {1, 0, {0, 0}}});
    CHECK(unchanged.success);
    CHECK(std::all_of(unchanged.program.operations.begin(), unchanged.program.operations.end(), [](const auto& op) {
        return op.accesses[0].nativeAccumulatorClass == 0;
    }));
    CHECK(o::analyze(unchanged.program, o::Commands(o::commandCutCount(unchanged.program))).verified());
    const auto moved = o::makePeriodicLoop(periodic, 2, {{0, 0, {0, 1}}, {1, 0, {0, 1}}});
    CHECK(moved.success);
    for (const auto& op : moved.program.operations) {
        const auto& access = op.accesses[0];
        CHECK(access.nativeAccumulatorClass == (access.cell == 0 ? 0 : o::NoControlId));
    }
    const auto movedReport = o::analyze(moved.program, o::Commands(o::commandCutCount(moved.program)));
    CHECK(movedReport.complete && !movedReport.verified());
    CHECK(std::all_of(movedReport.residuals.begin(), movedReport.residuals.end(), [](const auto& r) {
        return r.demand.cell == 1;
    }));
    for (unsigned invalid = 0; invalid < 7; ++invalid) {
        auto bad = p;
        if (invalid == 0) { bad.operations[0].accesses[0].nativeAccumulatorClass = 2; }
        if (invalid == 1) { bad.operations[0].pipe = o::Pipe::FIX; }
        if (invalid == 2) { bad.cells[0].domain = o::Cell::Domain::General; }
        if (invalid == 3) { bad.cells[0].storage = o::Cell::Storage::OverlapWitness; }
        if (invalid == 4) { bad.cells[0].exclusive = true; }
        if (invalid == 5) { bad.nativeAccumulatorClasses = o::NoControlId; }
        if (invalid == 6) { bad.target.contract = "another target"; }
        CHECK(!o::validateProgram(bad).success);
        CHECK(!o::CausalFrontier(bad).complete());
    }
}
void qualifications()
{
    auto p = base(1);
    p.operations[0].accesses = {{0, true, false}};
    auto unavailable = p;
    unavailable.operations[0].complete = false;
    CHECK(!o::CausalFrontier(unavailable).complete());
    unavailable = p;
    unavailable.target.synchronous[0] = true;
    CHECK(o::CausalFrontier(unavailable).complete());
    unavailable = p;
    unavailable.cells[0].exclusive = true;
    CHECK(!o::CausalFrontier(unavailable).complete());
    unavailable = p;
    unavailable.operations[0].resources = {{"queue", true, false}};
    CHECK(!o::CausalFrontier(unavailable).complete());
    unavailable = p;
    unavailable.invocation.retirement = o::Program::InvocationContract::DrainAllAtReturn;
    CHECK(!o::CausalFrontier(unavailable).complete());
    p.target.barrierAll = true;
    o::CausalFrontier f(p);
    CHECK(f.complete()); // availability is not execution of an unsupported ALL
    CHECK(f.command(f.initial(), {}, {0, 0}).failure == o::FrontierFailure::UnsupportedContract);
    CHECK(!o::checkCausalFrontier(p, {{{}}, {}}).complete);
    p.reservations.push_back({P, Q, 0, true});
    o::CausalFrontier reserved(p);
    CHECK(reserved.complete());
    CHECK(!reserved.command(reserved.initial(), pub(P, Q), {0, 0}).applied);
    CHECK(reserved.command(reserved.initial(), pub(P, Q, 1), {0, 0}).applied);
    CHECK(!reserved.issue(reserved.initial(), 100).applied);
    CHECK(!reserved.command(reserved.initial(), fence(P), {100, 0}).applied);
}
} // namespace
int main()
{
    primitives();
    generations();
    sharingAndReaders();
    joins();
    closedJoins();
    structured();
    qualifications();
    nativeAccumulatorOrdering();
    scopedAccumulatorHistories();
    std::cout << "causal frontier: " << checks << " assertions passed\n";
}
