// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent finite native+modeled-hazard oracle for invocation-local queries.
#include "../../lib/PTO/Transforms/FrontierSynch/DirectEmissionInternal.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
namespace {
using namespace mlir;
namespace fs = pto::frontiersynch;
using Int = llvm::DynamicAPInt;
using Role = fs::SignedAxisRole;
using Kind = fs::PeriodicEventKind;
using Tuple = fs::SymbolicTuple;
using Pipe = pto::PipelineType;
using Graph = SmallVector<SmallVector<bool>>;
struct Occurrence { unsigned site; SmallVector<Int> coordinates; };
fs::SignedTerm term(Role role, unsigned index, int sign) { return {{role, index}, sign}; }
fs::SignedAtom atom(std::initializer_list<fs::SignedTerm> terms, int64_t bound)
{
    return {SmallVector<fs::SignedTerm, 2>(terms), Int(bound)};
}
FailureOr<fs::SignedInputs> inputs(fs::SignedSpaceHandle space, unsigned depth)
{
    fs::SignedPiece admitted;
    admitted.residues.assign(2, Int(0));
    admitted.atoms = {atom({term(Role::Parameter, 0, -1)}, 0),
        atom({term(Role::Parameter, 1, -1)}, 0), atom({term(Role::Parameter, 1, 1)}, 1)};
    SmallVector<fs::SignedPiece> present;
    for (unsigned site = 0; site < 2; ++site) {
        fs::SignedPiece occurrence;
        occurrence.range.site = site;
        occurrence.residues.assign(depth + 2, Int(0));
        for (unsigned axis = 0; axis < depth; ++axis) {
            occurrence.atoms.push_back(atom({term(Role::Range, axis, -1)}, 0));
            if (axis + 1 != depth) {
                occurrence.atoms.push_back(atom({term(Role::Range, axis, 1)}, 1));
            }
        }
        occurrence.atoms.push_back(atom({term(Role::Range, 0, 1), term(Role::Parameter, 1, -1)}, 0));
        occurrence.atoms.push_back(atom({term(Role::Range, depth - 1, 1), term(Role::Parameter, 0, -1)}, -1));
        present.push_back(std::move(occurrence));
    }
    auto context = fs::SignedRelation::import(space, Tuple::Unit, Tuple::Unit, {admitted});
    auto presence = fs::SignedRelation::import(space, Tuple::Unit, Tuple::Occurrence, present);
    if (!context.succeeded() || !presence.succeeded()) { return failure(); }
    fs::SignedInputs result;
    result.context = context.value;
    result.present = presence.value;
    return result;
}
void close(Graph& graph)
{
    for (std::size_t middle = 0; middle < graph.size(); ++middle) {
        for (std::size_t source = 0; source < graph.size(); ++source) {
            for (std::size_t target = 0; target < graph.size(); ++target) {
                graph[source][target] = graph[source][target] ||
                    (graph[source][middle] && graph[middle][target]);
            }
        }
    }
}
bool sameInvocation(const Occurrence& a, const Occurrence& b)
{
    return ArrayRef<Int>(a.coordinates).drop_back() == ArrayRef<Int>(b.coordinates).drop_back();
}
bool present(const Occurrence& occurrence, unsigned n, unsigned guard)
{
    return occurrence.coordinates.front() <= Int(guard) && occurrence.coordinates.back() < Int(n);
}
bool check(const fs::SelectedAnalysis& selected, unsigned depth, unsigned n, unsigned guard)
{
    // Include absent endpoints (one-past-trip and inactive ancestor visits).
    SmallVector<Occurrence> occurrences;
    for (unsigned task = 0; task < 2; ++task) {
        for (unsigned batch = 0; batch < (depth == 3 ? 2U : 1U); ++batch) {
            for (unsigned iteration = 0; iteration <= n; ++iteration) {
                for (unsigned site = 0; site < 2; ++site) {
                    SmallVector<Int> coordinates{Int(task)};
                    if (depth == 3) { coordinates.push_back(Int(batch)); }
                    coordinates.push_back(Int(iteration));
                    occurrences.push_back({site, std::move(coordinates)});
                }
            }
        }
    }
    auto count = 2 * occurrences.size();
    Graph native(count, SmallVector<bool>(count, false));
    Graph requirements(count, SmallVector<bool>(count, false));
    // Build all raw forward cross-pipe hazardous pairs, rather than taking the
    // periodic reduction's generators or thresholds as the oracle.
    for (std::size_t a = 0; a < occurrences.size(); ++a) {
        for (std::size_t b = 0; b < occurrences.size(); ++b) {
            const auto& source = occurrences[a];
            const auto& target = occurrences[b];
            if (!present(source, n, guard) || !present(target, n, guard) || !sameInvocation(source, target)) {
                continue;
            }
            bool same = source.coordinates.back() == target.coordinates.back() && source.site == target.site;
            bool forward = source.coordinates.back() < target.coordinates.back() ||
                (source.coordinates.back() == target.coordinates.back() && source.site < target.site);
            if (source.site == target.site && (same || forward)) {
                native[2 * a][2 * b] = true;
                native[2 * a + 1][2 * b + 1] = true;
                native[2 * a][2 * b + 1] = true;
            }
            if (source.site != target.site && forward) { requirements[2 * a + 1][2 * b] = true; }
        }
    }
    Graph reach = native;
    for (std::size_t a = 0; a < count; ++a) {
        for (std::size_t b = 0; b < count; ++b) { reach[a][b] = reach[a][b] || requirements[a][b]; }
    }
    close(reach);
    for (std::size_t a = 0; a < count; ++a) {
        for (std::size_t b = 0; b < count; ++b) {
            bool cover = requirements[a][b] && !native[a][b];
            for (std::size_t middle = 0; cover && middle < count; ++middle) {
                if (middle != a && middle != b && reach[a][middle] && reach[middle][b]) { cover = false; }
            }
            fs::SignedPoint source{{occurrences[a / 2].site, a % 2 ? Kind::Completion : Kind::Start},
                occurrences[a / 2].coordinates};
            fs::SignedPoint target{{occurrences[b / 2].site, b % 2 ? Kind::Completion : Kind::Start},
                occurrences[b / 2].coordinates};
            auto actual = selected.reachability->contains(source, target, {Int(n), Int(guard)});
            auto minimum = selected.minimum->contains(source, target, {Int(n), Int(guard)});
            if (!actual.succeeded() || !minimum.succeeded() || actual.value != reach[a][b] || minimum.value != cover) {
                llvm::errs() << "nested periodic mismatch depth=" << depth << " n=" << n
                    << " guard=" << guard << " source=" << a << " target=" << b << '\n';
                return false;
            }
        }
    }
    return true;
}
bool checkPadding(const fs::SelectedAnalysis& selected)
{
    auto exported = selected.minimum->toSymbolic();
    if (failed(exported)) { return false; }
    // Active depth two in a depth-three global schema: padding is a fixed zero
    // axis, never a new free ancestor coordinate introduced by query lifting.
    SmallVector<Int> point{Int(0), Int(0), Int(0), Int(0), Int(1),
        Int(1), Int(0), Int(0), Int(0), Int(0), Int(1), Int(0)};
    if (!exported->relation().containsPoint(point)) { return false; }
    point[3] = Int(1);
    if (exported->relation().containsPoint(point)) { return false; }
    point[3] = Int(0); point[8] = Int(1);
    return !exported->relation().containsPoint(point);
}
} // namespace
int runPeriodicNestedQueryChecks(mlir::MLIRContext& context)
{
    auto module = parseSourceString<ModuleOp>(R"mlir(
      module { func.func @nested(%n: index, %g: index) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        scf.for %task = %c0 to %c2 step %c1 {
          scf.for %batch = %c0 to %c2 step %c1 {
            scf.for %i = %c0 to %n step %c1 { }
          }
        }
        return
      } })mlir", &context);
    if (!module) { return 1; }
    auto function = *module->getOps<func::FuncOp>().begin();
    SmallVector<scf::ForOp> loops;
    function.walk([&](scf::ForOp loop) { loops.push_back(loop); });
    if (loops.size() != 3) { return 1; }
    for (unsigned depth : {2U, 3U}) {
        SmallVector<std::unique_ptr<pto::CompoundInstanceElement>> phases;
        SmallVector<fs::SymbolicSite> sites;
        SmallVector<Value> coordinates{loops[0].getInductionVar()};
        if (depth == 3) { coordinates.push_back(loops[1].getInductionVar()); }
        coordinates.push_back(loops[2].getInductionVar());
        for (unsigned site = 0; site < 3; ++site) {
            phases.push_back(std::make_unique<pto::CompoundInstanceElement>(site,
                SmallVector<const pto::BaseMemInfo*>{}, SmallVector<const pto::BaseMemInfo*>{},
                site == 1 ? Pipe::PIPE_V : Pipe::PIPE_MTE2, OperationName("test.phase", &context)));
            auto active = coordinates;
            if (depth == 2 && site == 2) { active.insert(active.begin() + 1, loops[1].getInductionVar()); }
            sites.push_back({phases.back().get(), std::move(active)});
        }
        auto schema = fs::SymbolicSchema::create(sites,
            {function.getArgument(0), function.getArgument(1)}, {}, 0);
        if (failed(schema)) { return 1; }
        auto space = fs::SignedSpace::create(*schema, Int(1));
        if (!space.succeeded()) { return 1; }
        auto source = inputs(space.value, depth);
        if (failed(source)) { return 1; }
        fs::SelectedAnalysis selected;
        selected.kind = fs::SelectedAnalysis::Kind::Periodic;
        selected.sites = {0, 1}; selected.loop = loops[2];
        if (failed(selected.periodic.build({phases[0].get(), phases[1].get()},
                {{0, 1, Int(0)}, {1, 0, Int(1)}}))) { return 1; }
        std::string reason;
        if (failed(fs::liftPeriodicQueries(selected, *source, reason))) {
            llvm::errs() << reason << '\n'; return 1;
        }
        for (unsigned n : {0U, 1U, 2U, 3U}) {
            for (unsigned guard : {0U, 1U}) {
                if (!check(selected, depth, n, guard)) { return 1; }
            }
        }
        if (depth == 2 && !checkPadding(selected)) { return 1; }
    }
    llvm::outs() << "nested periodic queries: 16 independent finite closure/cover contexts passed; "
                    "ancestor separation, absent endpoints and global padding passed\n";
    return 0;
}
