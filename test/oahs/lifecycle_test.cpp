// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/OAHS/StorageFrontiers.h"
#include "PTO/Transforms/OAHS/StorageWitnesses.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
namespace o = mlir::pto::oahs;
namespace {
unsigned checks = 0;
void check(bool value, unsigned line)
{
    ++checks;
    if (!value) {
        std::cerr << "lifecycle check " << line << '\n';
        std::abort();
    }
}
#define CHECK(x) check(bool(x), __LINE__)
o::Program base(unsigned n)
{
    o::Program p;
    p.target.contract = "test ordinary complete footprints";
    for (unsigned i = 0; i < o::PipeCount; ++i) {
        p.target.supported[i] = p.target.barriers[i] = true;
        for (unsigned j = 0; j < o::PipeCount; ++j)
            if (i != j)
                p.target.keys[i][j] = {0, 1};
    }
    p.operations.resize(n);
    for (unsigned i = 0; i < n; ++i) {
        p.operations[i].complete = true;
        p.operations[i].pipe = o::Pipe(i % 3);
    }
    return p;
}
o::FootprintGroup footprint(unsigned op, uint64_t begin, uint64_t size, bool write, std::string space = "physical")
{
    o::FootprintGroup g;
    g.description.addressSpace = "local";
    g.description.coordinateSpace = std::move(space);
    g.description.ranges = {{begin, size}};
    g.description.storageOrigins = {op};
    g.uses.push_back({op, !write, write});
    return g;
}
bool conflicts(const o::Program& p, unsigned a, unsigned b)
{
    for (const auto& x : p.operations[a].accesses)
        for (const auto& y : p.operations[b].accesses)
            if (x.cell == y.cell && (x.write || y.write || p.cells[x.cell].exclusive))
                return true;
    return false;
}
o::Region leaf(unsigned i) { return {o::Region::Operation, {}, i}; }
o::Region seq(std::initializer_list<o::Region> c) { return {o::Region::Sequence, c}; }
bool contains(const std::vector<o::StorageOrigin>& a, unsigned op)
{
    return std::any_of(a.begin(), a.end(), [&](const auto& x) { return x.operation == op; });
}
void canonicalStorage()
{
    auto p = base(4);
    std::vector<o::FootprintGroup> g = {
        footprint(0, 0, 16, true), footprint(1, 8, 16, false), footprint(2, 32, 8, true),
        footprint(3, 0, 0, false, "")};
    g.back().description.unknownRange = true;
    const auto count = o::appendCanonicalStorage(p, g, [](auto a, auto b) { return a == 3 || b == 3; });
    CHECK(count.canonicalCells == 4);
    CHECK(count.pairCells == 2);
    CHECK(conflicts(p, 0, 1));
    CHECK(!conflicts(p, 0, 2) && !conflicts(p, 1, 2));
    CHECK(conflicts(p, 0, 3) && conflicts(p, 2, 3));
    bool sharedRoots = false;
    for (const auto& cell : p.cells)
        if (cell.storage == o::Cell::Storage::CanonicalInterval && cell.ranges[0].first == 8)
            sharedRoots = cell.storageOrigins == std::vector<std::size_t>({0, 1});
    CHECK(sharedRoots);
    CHECK(o::validateProgram(p).success);
    for (const auto& op : p.operations)
        for (const auto& a : op.accesses)
            CHECK(!a.definiteWrite);
    // Equal offsets in different root-relative spaces are not equal addresses.
    p = base(2);
    g = {footprint(0, 0, 16, true, "root0"), footprint(1, 0, 16, false, "root1")};
    o::appendCanonicalStorage(p, g, [](auto, auto) { return false; });
    CHECK(!conflicts(p, 0, 1));
    // A genuine conservative alias across coordinate spaces must still survive.
    p = base(2);
    o::appendCanonicalStorage(p, g, [](auto, auto) { return true; });
    CHECK(conflicts(p, 0, 1));
    // Overflow does not become a zero-sized or disjoint canonical atom.
    p = base(2);
    g[0].description.ranges = {{std::numeric_limits<uint64_t>::max() - 2, 8}};
    auto c = o::appendCanonicalStorage(p, g, [](auto, auto) { return true; });
    CHECK(c.canonicalCells == 1 && c.pairCells == 1 && conflicts(p, 0, 1));
    p.cells[0].storage = o::Cell::Storage::CanonicalInterval;
    CHECK(!o::validateProgram(p).success);
}
void differentialFootprints()
{
    std::mt19937 random(1701);
    for (unsigned trial = 0; trial < 300; ++trial) {
        std::vector<o::FootprintGroup> g;
        for (unsigned i = 0; i < 12; ++i) {
            const auto begin = random() % 64;
            const auto size = 1 + random() % 20;
            const bool write = random() % 2;
            auto f = footprint(i, begin, size, write);
            if (random() % 5 == 0) {
                f.description.unknownRange = true;
                f.description.coordinateSpace.clear();
            }
            f.description.exclusive = random() % 7 == 0;
            if (random() % 4 == 0)
                f.description.addressSpace = "other-domain";
            g.push_back(f);
        }
        auto alias = [&](auto a, auto b) {
            const auto &x = g[a].description, &y = g[b].description;
            if (x.addressSpace != y.addressSpace)
                return false;
            if (x.unknownRange || y.unknownRange)
                return true;
            const auto [xb, xs] = x.ranges[0];
            const auto [yb, ys] = y.ranges[0];
            return xb < yb + ys && yb < xb + xs;
        };
        auto old = base(12), now = base(12);
        o::appendStorageWitnesses(old, g, alias);
        o::appendCanonicalStorage(now, g, alias);
        CHECK(o::validateProgram(now).success);
        for (unsigned a = 0; a < 12; ++a)
            for (unsigned b = 0; b < 12; ++b)
                CHECK(conflicts(old, a, b) == conflicts(now, a, b));
    }
}
void lifecycle()
{
    auto p = base(3);
    p.cells.resize(1);
    p.cells[0].storage = o::Cell::Storage::CanonicalInterval;
    p.cells[0].coordinateSpace = "physical";
    p.cells[0].ranges = {{0, 16}};
    p.operations[0].accesses = {{0, false, true, true}};
    p.operations[1].accesses = {{0, true, false}};
    p.operations[2].accesses = {{0, false, true, true}};
    o::StorageFrontierAnalysis a(p);
    CHECK(a.complete());
    auto l = a.lifecycleAt(1, 0);
    CHECK(contains(l.previousWriters, 0) && contains(l.nextWriters, 2));
    CHECK(!l.mayHaveNoPriorFullWrite && !l.mayExitWithoutFurtherAccess);
    auto r = a.relationshipsAt(1);
    CHECK(r.size() == 1);
    auto provenance = a.describeRequirement(r[0]);
    CHECK(a.classifyRequirement(r[0]) == provenance.reasons);
    CHECK(provenance.reasons & o::KnownReadiness);
    CHECK(provenance.reasons & o::AdditionalOverlap);
    CHECK(provenance.occurrence.kind == o::OccurrenceQualification::SingleVisit);
    CHECK(a.lifecycleAt(2, 0).mayExitWithoutFurtherAccess);
    for (const auto& reuse : a.relationshipsAt(2))
        CHECK(a.describeRequirement(reuse).reasons & o::KnownReuse);
    // The same descriptor without a full-write certificate is only an overlap.
    p.operations[0].accesses[0].definiteWrite = false;
    o::StorageFrontierAnalysis partial(p);
    CHECK(!(partial.describeRequirement(partial.relationshipsAt(1)[0]).reasons & o::KnownReadiness));
    CHECK(partial.lifecycleAt(1, 0).mayHaveNoPriorFullWrite);
    // A may-write between a full definition and a read invalidates readiness.
    auto modified = p;
    modified.operations[0].accesses[0].definiteWrite = true;
    modified.operations[1].accesses = {{0, false, true, false}};
    modified.operations[2].accesses = {{0, true, false}};
    o::StorageFrontierAnalysis intervening(modified);
    for (const auto& edge : intervening.relationshipsAt(2))
        CHECK(!(intervening.describeRequirement(edge).reasons & o::KnownReadiness));
    p.operations[0].accesses[0].definiteWrite = true;
    // A branch-local writer cannot initialize its bypass path.
    p.body = seq({{o::Region::Choice, {leaf(0), seq({})}}, leaf(1), leaf(2)});
    o::StorageFrontierAnalysis branch(p);
    CHECK(branch.lifecycleAt(1, 0).mayHaveNoPriorFullWrite);
    CHECK(!(branch.classifyRequirement(branch.relationshipsAt(1)[0]) & o::KnownReadiness));
    // Read-only child sees both the outside writer and outside overwrite.
    p.body = seq({leaf(0), {o::Region::For, {leaf(1)}, 0, true}, leaf(2)});
    o::StorageFrontierAnalysis loop(p);
    l = loop.lifecycleAt(1, 0);
    CHECK(l.enclosingLoops.size() == 1);
    CHECK(contains(l.previousWriters, 0) && contains(l.nextWriters, 2));
    CHECK(!l.mayExitWithoutFurtherAccess);
    auto q = loop.describeRequirement(loop.relationshipsAt(1)[0]);
    CHECK(q.occurrence.kind == o::OccurrenceQualification::Unknown);
    CHECK(!(q.reasons & o::KnownReadiness));
    // While-before remains mandatory; after may execute zero times.
    p.body = {o::Region::While, {seq({leaf(0), leaf(1)}), leaf(2)}};
    o::StorageFrontierAnalysis w(p);
    CHECK(w.lifecycleAt(1, 0).mayExitWithoutFurtherAccess);
    CHECK(!w.lifecycleAt(2, 0).mayExitWithoutFurtherAccess);
    CHECK(contains(w.lifecycleAt(2, 0).nextWriters, 0));
    // A strong overwrite changes provenance, not original completion demands.
    p.body = {};
    p.operations[1].accesses = {{0, false, true, true}};
    p.operations[2].accesses = {{0, true, false}};
    auto unsynchronized = o::analyze(p);
    CHECK(std::any_of(unsynchronized.residuals.begin(), unsynchronized.residuals.end(), [](const auto& x) {
        return x.demand.producer == 0 && x.demand.consumer == 2;
    }));
}
} // namespace
int main()
{
    canonicalStorage();
    differentialFootprints();
    lifecycle();
    std::cout << "OAHS canonical storage/lifecycle: " << checks << " checks passed\n";
}
