// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent finite event closure oracle for delayed local strengthening.
// The symbolic fixture has unbounded n and an immutable B-presence parameter.
#include "PTO/Transforms/FrontierSynch/SignedDemandAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
using R = fs::SignedRelationHandle;
using Pipe = pto::PipelineType;
using Matrix = SmallVector<SmallVector<bool>>;
struct Inputs { R context, native, minimum; };
struct Occurrence { std::size_t site; unsigned iteration; };
fs::SignedTerm term(Role role, unsigned index, int sign) { return {{role, index}, sign}; }
fs::SignedAtom atom(std::initializer_list<fs::SignedTerm> terms, int64_t bound)
{
    return {SmallVector<fs::SignedTerm, 2>(terms), Int(bound)};
}
SmallVector<fs::SignedAtom> contextAtoms()
{
    return {atom({term(Role::Parameter, 0, -1)}, 0),
        atom({term(Role::Parameter, 1, -1)}, 0), atom({term(Role::Parameter, 1, 1)}, 1)};
}
fs::SignedPiece pair(std::size_t a, Kind first, std::size_t b, Kind second, int64_t distance)
{
    fs::SignedPiece result;
    result.domain = {a, first}; result.range = {b, second};
    result.residues = {Int(0), Int(0), Int(0), Int(0)};
    result.atoms = contextAtoms();
    for (Role role : {Role::Domain, Role::Range}) {
        result.atoms.push_back(atom({term(role, 0, -1)}, 0));
        result.atoms.push_back(atom({term(role, 0, 1), term(Role::Parameter, 0, -1)}, -1));
    }
    if (a == 1 || b == 1) { result.atoms.push_back(atom({term(Role::Parameter, 1, -1)}, -1)); }
    result.atoms.push_back(atom({term(Role::Domain, 0, 1), term(Role::Range, 0, -1)}, distance));
    return result;
}
FailureOr<R> import(fs::SignedSpaceHandle space, Tuple domain, Tuple range, ArrayRef<fs::SignedPiece> pieces)
{
    auto result = fs::SignedRelation::import(space, domain, range, pieces);
    if (!result.succeeded()) { return failure(); }
    return result.value;
}
FailureOr<Inputs> inputs(fs::SignedSpaceHandle space, bool includeLocal)
{
    fs::SignedPiece context;
    context.residues = {Int(0), Int(0)};
    context.atoms = contextAtoms();
    SmallVector<fs::SignedPiece> native, minimum;
    for (std::size_t a = 0; a < 4; ++a) {
        for (std::size_t b = 0; b < 4; ++b) {
            if ((a == 3) != (b == 3)) { continue; }
            for (auto kinds : {std::pair{Kind::Start, Kind::Start},
                               std::pair{Kind::Completion, Kind::Completion},
                               std::pair{Kind::Start, Kind::Completion}}) {
                native.push_back(pair(a, kinds.first, b, kinds.second, a <= b ? 0 : -1));
            }
        }
    }
    for (auto sites : {std::pair<std::size_t, std::size_t>{0, 2},
                       std::pair<std::size_t, std::size_t>{2, 3}}) {
        if (!includeLocal && sites.first == 0) { continue; }
        auto edge = pair(sites.first, Kind::Completion, sites.second, Kind::Start, 0);
        edge.atoms.push_back(atom({term(Role::Range, 0, 1), term(Role::Domain, 0, -1)}, 0));
        minimum.push_back(edge);
    }
    auto admitted = import(space, Tuple::Unit, Tuple::Unit, {context});
    auto order = import(space, Tuple::Event, Tuple::Event, native);
    auto covers = import(space, Tuple::Event, Tuple::Event, minimum);
    if (failed(admitted) || failed(order) || failed(covers)) { return failure(); }
    return Inputs{*admitted, *order, *covers};
}
Matrix emptyMatrix(std::size_t count) { return Matrix(count, SmallVector<bool>(count, false)); }
void closure(Matrix& edges)
{
    for (std::size_t middle = 0; middle < edges.size(); ++middle) {
        for (std::size_t a = 0; a < edges.size(); ++a) {
            for (std::size_t b = 0; b < edges.size(); ++b) {
                edges[a][b] = edges[a][b] || (edges[a][middle] && edges[middle][b]);
            }
        }
    }
}
Matrix covers(const Matrix& reachable, const Matrix& native)
{
    Matrix result = emptyMatrix(reachable.size());
    for (std::size_t a = 0; a < reachable.size(); ++a) {
        for (std::size_t b = 0; b < reachable.size(); ++b) {
            if (a == b || !reachable[a][b] || native[a][b]) { continue; }
            bool intermediate = false;
            for (std::size_t c = 0; c < reachable.size(); ++c) {
                if (c != a && c != b && reachable[a][c] && reachable[c][b]) { intermediate = true; }
            }
            result[a][b] = !intermediate;
        }
    }
    return result;
}
struct Execution {
    SmallVector<Occurrence> occurrences;
    Matrix native, oldReach, upperReach, minimum;
};
Execution execute(unsigned n, bool presentB, bool includeLocal)
{
    Execution result;
    for (unsigned iteration = 0; iteration < n; ++iteration) {
        for (std::size_t site = 0; site < 4; ++site) {
            if (site != 1 || presentB) { result.occurrences.push_back({site, iteration}); }
        }
    }
    const auto count = result.occurrences.size();
    result.native = emptyMatrix(2 * count);
    for (std::size_t a = 0; a < count; ++a) {
        for (std::size_t b = a; b < count; ++b) {
            if ((result.occurrences[a].site == 3) != (result.occurrences[b].site == 3)) { continue; }
            result.native[2 * a][2 * b] = true;
            result.native[2 * a + 1][2 * b + 1] = true;
            result.native[2 * a][2 * b + 1] = true;
        }
    }
    result.oldReach = result.native;
    result.upperReach = result.native;
    for (std::size_t consumer = 0; consumer < count; ++consumer) {
        auto target = result.occurrences[consumer];
        for (std::size_t source = 0; source < consumer; ++source) {
            auto producer = result.occurrences[source];
            if (producer.iteration != target.iteration) { continue; }
            bool local = includeLocal && producer.site == 0 && target.site == 2;
            bool cross = producer.site == 2 && target.site == 3;
            if (local || cross) {
                result.oldReach[2 * source + 1][2 * consumer] = true;
                result.upperReach[2 * source + 1][2 * consumer] = true;
            }
        }
        if (includeLocal && target.site == 2) {
            // Walk the actual finite pipe sequence, including guard absence.
            std::size_t previous = consumer;
            while (previous != 0) {
                --previous;
                if (result.occurrences[previous].site != 3) {
                    result.upperReach[2 * previous + 1][2 * consumer] = true;
                    break;
                }
            }
        }
    }
    closure(result.oldReach); closure(result.upperReach);
    result.minimum = covers(result.upperReach, result.native);
    return result;
}
fs::SignedPoint point(const Occurrence& occurrence, bool completion)
{
    return {{occurrence.site, completion ? Kind::Completion : Kind::Start}, {Int(occurrence.iteration)}};
}
bool checkRelations(fs::SignedAnalysisHandle upper, const Execution& execution, unsigned n, bool presentB)
{
    for (std::size_t a = 0; a < execution.upperReach.size(); ++a) {
        for (std::size_t b = 0; b < execution.upperReach.size(); ++b) {
            auto source = point(execution.occurrences[a / 2], a % 2 != 0);
            auto target = point(execution.occurrences[b / 2], b % 2 != 0);
            auto reach = upper->reachability()->contains(source, target, {Int(n), Int(presentB)});
            auto cover = upper->minimum()->contains(source, target, {Int(n), Int(presentB)});
            if (!reach.succeeded() || !cover.succeeded() || reach.value != execution.upperReach[a][b] ||
                cover.value != execution.minimum[a][b] || (execution.oldReach[a][b] && !reach.value)) { return false; }
        }
    }
    return true;
}
bool checkSelector(fs::SignedSelectorHandle selector, const Occurrence& occurrence, bool incoming,
                   const Execution& execution, std::size_t position, Pipe other, unsigned n, bool presentB)
{
    fs::SymbolicEvent event{occurrence.site, {Int(occurrence.iteration)}, incoming ? Kind::Start : Kind::Completion};
    auto selected = selector->evaluate(event, {Int(n), Int(presentB)});
    if (!selected.succeeded()) { return false; }
    std::optional<fs::SymbolicEvent> expected;
    for (std::size_t partner = 0; partner < execution.occurrences.size(); ++partner) {
        auto candidate = execution.occurrences[partner];
        Pipe pipe = candidate.site == 3 ? Pipe::PIPE_V : Pipe::PIPE_MTE2;
        bool cover = incoming ? execution.minimum[2 * partner + 1][2 * position] :
                               execution.minimum[2 * position + 1][2 * partner];
        if (cover && pipe == other) {
            if (expected) { return false; }
            expected = fs::SymbolicEvent{candidate.site, {Int(candidate.iteration)},
                incoming ? Kind::Completion : Kind::Start};
        }
    }
    if (selected.value.has_value() != expected.has_value()) { return false; }
    return !expected || (selected.value->site == expected->site && selected.value->kind == expected->kind &&
        selected.value->coordinates == expected->coordinates);
}
bool checkExecution(fs::SignedAnalysisHandle upper, unsigned n, bool presentB, bool includeLocal)
{
    auto execution = execute(n, presentB, includeLocal);
    if (!checkRelations(upper, execution, n, presentB)) { return false; }
    for (std::size_t position = 0; position < execution.occurrences.size(); ++position) {
        auto occurrence = execution.occurrences[position];
        Pipe current = occurrence.site == 3 ? Pipe::PIPE_V : Pipe::PIPE_MTE2;
        for (Pipe other : {Pipe::PIPE_MTE2, Pipe::PIPE_V}) {
            auto outgoing = upper->outgoing(current, other), incoming = upper->incoming(other, current);
            if (!outgoing.succeeded() || !incoming.succeeded() ||
                !checkSelector(outgoing.value, occurrence, false, execution, position, other, n, presentB) ||
                !checkSelector(incoming.value, occurrence, true, execution, position, other, n, presentB)) {
                return false;
            }
        }
    }
    // Query absent endpoints too, including empty invocations and skipped B.
    for (std::size_t source = 0; source < 4; ++source) {
        auto local = upper->minimum()->contains({{source, Kind::Completion}, {Int(0)}},
            {{2U, Kind::Start}, {Int(0)}}, {Int(n), Int(presentB)});
        bool expected = n != 0 && includeLocal && source == (presentB ? 1U : 0U);
        if (!local.succeeded() || local.value != expected) { return false; }
    }
    return true;
}
} // namespace
int runAdjacentLocalUpperChecks(mlir::MLIRContext& context)
{
    auto module = parseSourceString<ModuleOp>(
        "module { func.func @schema(%a: index, %b: index, %c: index, %d: index, %n: index, %g: index) { return } }",
        &context);
    if (!module) { return 1; }
    auto function = *module->getOps<func::FuncOp>().begin();
    SmallVector<std::unique_ptr<pto::CompoundInstanceElement>> phases;
    SmallVector<fs::SymbolicSite> sites;
    for (std::size_t site = 0; site < 4; ++site) {
        phases.push_back(std::make_unique<pto::CompoundInstanceElement>(site,
            SmallVector<const pto::BaseMemInfo*>{}, SmallVector<const pto::BaseMemInfo*>{},
            site == 3 ? Pipe::PIPE_V : Pipe::PIPE_MTE2, OperationName("test.phase", &context)));
        sites.push_back({phases.back().get(), {function.getArgument(site)}});
    }
    auto schema = fs::SymbolicSchema::create(sites, {function.getArgument(4), function.getArgument(5)}, {}, 0);
    if (failed(schema)) { return 1; }
    auto space = fs::SignedSpace::create(*schema, Int(1));
    if (!space.succeeded()) { return 1; }
    for (bool includeLocal : {false, true}) {
        auto selected = inputs(space.value, includeLocal);
        if (failed(selected)) { return 1; }
        auto upper = fs::SignedDemandAnalysis::adjacentLocalUpper(space.value,
            selected->context, selected->native, selected->minimum);
        if (!upper.succeeded() || upper.value->native() != selected->native ||
            upper.value->context() != selected->context) { return 1; }
        for (unsigned n : {0, 1, 2, 4}) {
            for (bool presentB : {false, true}) {
                if (!checkExecution(upper.value, n, presentB, includeLocal)) {
                    llvm::errs() << "adjacent local upper finite oracle failed: n=" << n
                                 << " B=" << presentB << " local=" << includeLocal << "\n";
                    return 1;
                }
            }
        }
    }
    llvm::outs() << "adjacent local upper finite closure, cover, and selector oracles passed\n";
    return 0;
}
