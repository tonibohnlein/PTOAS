// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent generic relation and finite graph oracles for optimized composition.
#include "../../lib/PTO/Transforms/FrontierSynch/GeneralQueries.h"
#include "../../lib/PTO/Transforms/FrontierSynch/SymbolicComposition.h"
#include "PTO/Transforms/InsertSync/SyncInput.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <string>
#include <memory>
#include <utility>
namespace {
using namespace mlir;
namespace fs = pto::frontiersynch;
using R = presburger::PresburgerRelation;
using I = presburger::IntegerRelation;
using Int = llvm::DynamicAPInt;
using Tuple = fs::SymbolicTuple;
using Graph = SmallVector<SmallVector<unsigned char>>;
const char* fixture = R"mlir(
!vec = !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=128, v_row=16, v_col=128,
                    blayout=row_major, slayout=none_box, fractal=512, pad=0>
!gm = !pto.partition_tensor_view<16x128xi32>
module {
  func.func @compose(%src: !gm, %dst: !gm, %n: index) {
    %a = arith.constant 0 : i64
    %b = arith.constant 8192 : i64
    %c = arith.constant 16384 : i64
    %x = pto.alloc_tile addr = %a : !vec
    %y = pto.alloc_tile addr = %b : !vec
    %z = pto.alloc_tile addr = %c : !vec
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %middle = arith.subi %n, %one : index
    scf.for %i = %zero to %n step %two {
      pto.tload ins(%src : !gm) outs(%x : !vec)
    }
    scf.for %j = %zero to %middle step %two {
      pto.txor ins(%x, %x, %z : !vec, !vec, !vec) outs(%y : !vec)
    }
    scf.for %k = %zero to %n step %two {
      pto.tstore ins(%y : !vec) outs(%dst : !gm)
    }
    return
  }
})mlir";
R genericCompose(R first, const R& second)
{
    first.compose(second);
    return first;
}
FailureOr<fs::ExactStructuredEffects> effects(fs::SymbolicSchemaHandle schema, bool empty)
{
    auto contextSpace = *schema->space(Tuple::Unit, Tuple::Unit);
    I admitted(contextSpace);
    admitted.addBound(presburger::BoundType::LB, 0, Int(0));
    admitted.addBound(presburger::BoundType::UB, 0, Int(2));
    auto context = fs::SymbolicPrimitive::import(schema, Tuple::Unit, Tuple::Unit,
                                               schema->parameters(), R(admitted));
    auto access = fs::SymbolicPrimitive::import(schema, Tuple::Occurrence, Tuple::Cell,
        schema->parameters(), R::getEmpty(*schema->space(Tuple::Occurrence, Tuple::Cell)));
    auto space = *schema->space(Tuple::Occurrence, Tuple::Occurrence);
    R generators = R::getEmpty(space);
    if (!empty) {
        for (auto pair : {std::pair<unsigned, unsigned>{0, 1}, {1, 2}, {0, 2}}) {
            I piece(space);
            unsigned width = piece.getNumDomainVars();
            piece.addBound(presburger::BoundType::EQ, 0, Int(pair.first));
            piece.addBound(presburger::BoundType::EQ, width, Int(pair.second));
            SmallVector<Int> row(piece.getNumCols(), Int(0));
            row[1] = Int(1);
            row[width + 1] = Int(-1);
            piece.addEquality(row);
            generators.unionInPlace(piece);
        }
    }
    auto required = fs::SymbolicPrimitive::import(schema, Tuple::Occurrence, Tuple::Occurrence,
        schema->parameters(), generators);
    auto extras = fs::SymbolicPrimitive::import(schema, Tuple::Occurrence, Tuple::Occurrence,
        schema->parameters(), R::getEmpty(space));
    if (failed(context) || failed(access) || failed(required) || failed(extras)) { return failure(); }
    return fs::ExactStructuredEffects{*context, *access, *access, *extras, *required};
}
FailureOr<R> materialize(fs::SymbolicEvaluator& evaluator, const fs::SymbolicRoot& root)
{
    auto relation = evaluator.materialize(root);
    if (failed(relation)) { return failure(); }
    return relation->relation();
}
R siteSet(fs::SymbolicSchemaHandle schema, unsigned site)
{
    auto space = *schema->space(Tuple::Unit, Tuple::Event);
    I piece(space);
    piece.addBound(presburger::BoundType::EQ, 0, Int(site));
    return R(piece);
}
void close(Graph& graph)
{
    for (std::size_t k = 0; k < graph.size(); ++k) {
        for (std::size_t a = 0; a < graph.size(); ++a) {
            for (std::size_t b = 0; b < graph.size(); ++b) { graph[a][b] |= graph[a][k] && graph[k][b]; }
        }
    }
}
struct Event { unsigned site, coordinate, kind; };
bool contains(const R& relation, Event source, Event target, unsigned parameter)
{
    return relation.containsPoint(SmallVector<Int>{Int(source.site), Int(source.coordinate), Int(source.kind),
        Int(target.site), Int(target.coordinate), Int(target.kind), Int(parameter)});
}
// Exhaustive equality on this driver's admitted finite contexts and event
// coordinates, with absent/off-lattice tuples included. This is a bounded
// regression oracle, not an unbounded relation equivalence certificate.
bool boundedEqual(const R& actual, const R& expected)
{
    for (unsigned n = 0; n <= 2; ++n) {
        for (unsigned a = 0; a < 3; ++a) {
            for (unsigned b = 0; b < 3; ++b) {
                for (unsigned ac = 0; ac <= 2; ++ac) {
                    for (unsigned bc = 0; bc <= 2; ++bc) {
                        for (unsigned ak : {0U, 1U}) {
                            for (unsigned bk : {0U, 1U}) {
                                bool observed = contains(actual, {a, ac, ak}, {b, bc, bk}, n);
                                bool wanted = contains(expected, {a, ac, ak}, {b, bc, bk}, n);
                                if (observed != wanted) {
                                    llvm::errs() << "bounded mismatch n=" << n << " source=(" << a << ","
                                                 << ac << "," << ak << ") target=(" << b << "," << bc << ","
                                                 << bk << ") actual=" << observed << " expected=" << wanted << "\n";
                                    return false;
                                }
                            }
                        }
                    }
                }
            }
        }
        for (Event invalid : {Event{3, 0, 0}, Event{0, 0, 2}}) {
            if (contains(actual, invalid, {0, 0, 0}, n) ||
                contains(actual, {0, 0, 0}, invalid, n)) { return false; }
        }
    }
    return true;
}
bool finiteOracle(const fs::GeneralQueries& queries, const R& native, const R& generators)
{
    for (unsigned n = 0; n <= 2; ++n) {
        SmallVector<Event> events;
        for (unsigned site = 0; site < 3; ++site) {
            unsigned bound = site == 1 ? (n ? n - 1 : 0) : n;
            for (unsigned coordinate = 0; coordinate < bound; coordinate += 2) {
                events.push_back({site, coordinate, 0}); events.push_back({site, coordinate, 1});
            }
        }
        Graph baseline(events.size(), SmallVector<unsigned char>(events.size()));
        auto required = baseline;
        for (std::size_t a = 0; a < events.size(); ++a) {
            for (std::size_t b = 0; b < events.size(); ++b) {
                baseline[a][b] = contains(native, events[a], events[b], n);
                required[a][b] = baseline[a][b] || contains(generators, events[a], events[b], n);
            }
        }
        close(baseline); close(required);
        for (std::size_t a = 0; a < events.size(); ++a) {
            for (std::size_t b = 0; b < events.size(); ++b) {
                bool covered = required[a][b] && !baseline[a][b] && a != b;
                for (std::size_t k = 0; k < events.size(); ++k) {
                    if (k != a && k != b && required[a][k] && required[k][b]) { covered = false; }
                }
                if (contains(queries.reachability, events[a], events[b], n) != bool(required[a][b]) ||
                    contains(queries.minimum, events[a], events[b], n) != covered) { return false; }
            }
        }
        // Absent and off-lattice endpoints must not enter exported queries.
        if (contains(queries.reachability, {1, n, 0}, {0, 0, 1}, n) ||
            contains(queries.reachability, {0, 1, 0}, {0, 1, 1}, n)) { return false; }
    }
    return true;
}
bool joins(fs::SymbolicSchemaHandle schema)
{
    auto space = *schema->space(Tuple::Event, Tuple::Event);
    auto make = [&](bool fixedSite, bool fixedKind) {
        I piece(space);
        piece.addBound(presburger::BoundType::EQ, 0, Int(0));
        if (fixedSite) { piece.addBound(presburger::BoundType::EQ, 3, Int(1)); }
        else {
            piece.addBound(presburger::BoundType::LB, 3, Int(0));
            piece.addBound(presburger::BoundType::UB, 3, Int(2));
        }
        piece.addBound(presburger::BoundType::EQ, 2, Int(0));
        if (fixedKind) { piece.addBound(presburger::BoundType::EQ, 5, Int(1)); }
        else {
            piece.addBound(presburger::BoundType::LB, 5, Int(0));
            piece.addBound(presburger::BoundType::UB, 5, Int(1));
        }
        return R(piece);
    };
    for (auto fixed : {std::pair<bool, bool>{true, true}, {false, true}, {true, false}}) {
        auto left = make(fixed.first, fixed.second);
        for (unsigned kind : {0U, 1U}) {
            // Left range/site 1 matches right source/site 1; opposite kinds do not.
            I piece(space);
            piece.addBound(presburger::BoundType::EQ, 0, Int(1));
            piece.addBound(presburger::BoundType::EQ, 3, Int(2));
            piece.addBound(presburger::BoundType::EQ, 2, Int(kind));
            piece.addBound(presburger::BoundType::EQ, 5, Int(0));
            R right(piece);
            auto actual = fs::composeGeneralEventRelations(left, right, schema);
            if (!actual.isEqual(genericCompose(left, right))) { return false; }
        }
    }
    return true;
}
bool typedJoins(fs::SymbolicSchemaHandle original, MLIRContext& context)
{
    auto created = fs::SymbolicSchema::create(original->sites(), original->parameters(),
        {IndexType::get(&context)}, original->partition());
    if (failed(created)) { return false; }
    auto schema = *created;
    struct Shape { Tuple source, middle, target; };
    for (auto shape : {Shape{Tuple::Occurrence, Tuple::Occurrence, Tuple::Event},
                       Shape{Tuple::Unit, Tuple::Occurrence, Tuple::Event},
                       Shape{Tuple::Event, Tuple::Event, Tuple::Unit},
                       Shape{Tuple::Event, Tuple::Cell, Tuple::Event},
                       Shape{Tuple::Event, Tuple::Unit, Tuple::Event}}) {
        for (bool bound : {false, true}) {
            auto firstSpace = *schema->space(shape.source, shape.middle, bound);
            auto secondSpace = *schema->space(shape.middle, shape.target, bound);
            for (bool literal : {false, true}) {
                I first(firstSpace), second(secondSpace);
                unsigned offset = first.getNumDomainVars();
                unsigned middleWidth = second.getNumDomainVars();
                bool tagged = shape.middle == Tuple::Occurrence || shape.middle == Tuple::Event;
                if (tagged) {
                    if (literal) { first.addBound(presburger::BoundType::EQ, offset, Int(0)); }
                    else {
                        first.addBound(presburger::BoundType::LB, offset, Int(0));
                        first.addBound(presburger::BoundType::UB, offset, Int(2));
                    }
                    second.addBound(presburger::BoundType::EQ, 0, Int(0));
                    if (shape.middle == Tuple::Event) {
                        first.addBound(presburger::BoundType::EQ, offset + middleWidth - 1, Int(1));
                        second.addBound(presburger::BoundType::EQ, middleWidth - 1, Int(1));
                    }
                }
                // A genuine existential div local survives the middle join:
                // the middle coordinate is even, independently of its tag.
                if (middleWidth) {
                    unsigned coordinate = tagged ? 1 : 0;
                    first.appendVar(presburger::VarKind::Local);
                    SmallVector<Int> row(first.getNumCols(), Int(0));
                    row[offset + coordinate] = Int(1);
                    row[first.getNumVars() - 1] = Int(-2);
                    first.addEquality(row);
                    if (shape.source == Tuple::Occurrence || shape.source == Tuple::Event) {
                        SmallVector<Int> tied(first.getNumCols(), Int(0));
                        tied[offset + coordinate] = Int(1);
                        tied[1] = Int(-1);
                        first.addEquality(tied);
                    }
                    second.addBound(presburger::BoundType::LB, coordinate, Int(-3));
                    second.addBound(presburger::BoundType::UB, coordinate, Int(3));
                    if (shape.target == Tuple::Occurrence || shape.target == Tuple::Event) {
                        SmallVector<Int> tied(second.getNumCols(), Int(0));
                        tied[coordinate] = Int(1);
                        tied[middleWidth + 1] = Int(-1);
                        second.addEquality(tied);
                    }
                }
                if (!bound) {
                    first.addBound(presburger::BoundType::LB, first.getNumDimVars(), Int(0));
                    first.addBound(presburger::BoundType::UB, first.getNumDimVars(), Int(4));
                    second.addBound(presburger::BoundType::LB, second.getNumDimVars(), Int(2));
                    second.addBound(presburger::BoundType::UB, second.getNumDimVars(), Int(6));
                }
                R left(first), right(second);
                auto actual = fs::composeSymbolicRelations(left, right, schema, shape.source,
                    shape.middle, shape.target, bound);
                if (!actual.isEqual(genericCompose(left, right))) { return false; }
                auto noLeft = R::getEmpty(firstSpace);
                auto noRight = R::getEmpty(secondSpace);
                if (!fs::composeSymbolicRelations(noLeft, right, schema, shape.source,
                        shape.middle, shape.target, bound).isIntegerEmpty() ||
                    !fs::composeSymbolicRelations(left, noRight, schema, shape.source,
                        shape.middle, shape.target, bound).isIntegerEmpty()) { return false; }
            }
        }
    }
    return true;
}
bool normalizationOracle()
{
    // Independently enumerate the integer formula, including a parity local
    // that cannot be removed by unit substitution. External axes stay intact.
    for (int sign : {-1, 1}) {
        I piece(presburger::PresburgerSpace::getRelationSpace(1, 1, 1, 3));
        piece.addEquality({Int(1), Int(0), Int(0), Int(-sign), Int(0), Int(0), Int(0)});
        piece.addEquality({Int(0), Int(0), Int(1), Int(1), Int(-1), Int(0), Int(0)});
        piece.addEquality({Int(0), Int(-1), Int(0), Int(0), Int(0), Int(2), Int(0)});
        piece.addInequality({Int(0), Int(0), Int(0), Int(0), Int(1), Int(0), Int(-3)});
        piece.addInequality({Int(0), Int(1), Int(0), Int(-2), Int(0), Int(0), Int(5)});
        R original(piece), normalized = fs::normalizeSymbolicRelation(original);
        if (normalized.getNumDomainVars() != 1 || normalized.getNumRangeVars() != 1 ||
            normalized.getNumSymbolVars() != 1) { return false; }
        for (int x = -3; x <= 3; ++x) {
            for (int y = -4; y <= 6; ++y) {
                for (int parameter = -1; parameter <= 2; ++parameter) {
                    SmallVector<Int> point{Int(x), Int(y), Int(parameter)};
                    bool expected = sign * x + parameter >= 3 && y - 2 * sign * x + 5 >= 0 && y % 2 == 0;
                    if (original.containsPoint(point) != expected || normalized.containsPoint(point) != expected) {
                        llvm::errs() << "unit-local normalization changed integer membership\n";
                        return false;
                    }
                }
            }
        }
        SmallVector<Int> contradiction(piece.getNumCols(), Int(0));
        contradiction.back() = Int(1);
        piece.addEquality(contradiction);
        if (!fs::normalizeSymbolicRelation(R(piece)).isIntegerEmpty()) { return false; }
    }
    return true;
}
bool subtractionOracle(fs::SymbolicSchemaHandle schema)
{
    // Integer-empty RHS with a rational witness q=-1/2 must remove nothing.
    // Keep a separate represented LHS local to cover local merging as well.
    for (bool bound : {false, true}) {
        auto space = *schema->space(Tuple::Event, Tuple::Event, bound);
        I left(space);
        SmallVector<Int> point{Int(1), Int(0), Int(1), Int(2), Int(0), Int(0)};
        if (!bound) { point.push_back(Int(2)); }
        for (unsigned axis = 0; axis < point.size(); ++axis) {
            left.addBound(presburger::BoundType::EQ, axis, point[axis]);
        }
        left.appendVar(presburger::VarKind::Local);
        SmallVector<Int> definition(left.getNumCols(), Int(0));
        definition[1] = Int(1); definition[left.getNumVars() - 1] = Int(-2);
        left.addEquality(definition);
        I right = left;
        right.appendVar(presburger::VarKind::Local);
        SmallVector<Int> odd(right.getNumCols(), Int(0));
        odd[right.getNumVars() - 1] = Int(2); odd.back() = Int(1);
        right.addInequality(odd);
        for (auto& coefficient : odd) { coefficient = -coefficient; }
        right.addInequality(odd);
        R lhs(left), rhs(right);
        auto result = fs::subtractSymbolicRelations(lhs, rhs, schema, Tuple::Event, Tuple::Event, bound);
        if (!rhs.isIntegerEmpty() || !lhs.containsPoint(point) || !result.containsPoint(point)) {
            llvm::errs() << "tiny integer-empty parity RHS removed nonempty LHS bound=" << bound << "\n";
            return false;
        }
    }
    for (bool bound : {false, true}) {
        for (bool literal : {false, true}) {
            for (bool matching : {false, true}) {
                auto space = *schema->space(Tuple::Event, Tuple::Event, bound);
                I left(space), right(space);
                if (literal) { left.addBound(presburger::BoundType::EQ, 0, Int(0)); }
                else {
                    left.addBound(presburger::BoundType::LB, 0, Int(0));
                    left.addBound(presburger::BoundType::UB, 0, Int(1));
                }
                left.addBound(presburger::BoundType::EQ, 2, Int(1));
                left.addBound(presburger::BoundType::EQ, 3, Int(2));
                left.addBound(presburger::BoundType::EQ, 5, Int(0));
                left.addBound(presburger::BoundType::LB, 4, Int(-2));
                left.addBound(presburger::BoundType::UB, 4, Int(2));
                right.addBound(presburger::BoundType::EQ, 0, Int(matching ? 0 : 1));
                right.addBound(presburger::BoundType::EQ, 2, Int(1));
                right.addBound(presburger::BoundType::EQ, 3, Int(2));
                right.addBound(presburger::BoundType::EQ, 5, Int(matching ? 0 : 1));
                right.appendVar(presburger::VarKind::Local);
                SmallVector<Int> parity(right.getNumCols(), Int(0));
                parity[1] = Int(1); parity[right.getNumVars() - 1] = Int(-2);
                right.addEquality(parity);
                // Arbitrary existential RHS: exists q, 0<=q<=target coordinate.
                // This requires exact integer elimination, not projection.
                right.appendVar(presburger::VarKind::Local);
                unsigned local = right.getNumVars() - 1;
                right.addBound(presburger::BoundType::LB, local, Int(0));
                SmallVector<Int> row(right.getNumCols(), Int(0));
                row[4] = Int(1); row[local] = Int(-1);
                right.addInequality(row);
                if (!bound) {
                    left.addBound(presburger::BoundType::LB, 6, Int(0));
                    left.addBound(presburger::BoundType::UB, 6, Int(2));
                    right.addBound(presburger::BoundType::EQ, 6, Int(1));
                }
                R a(left), b(right);
                auto raw = a.subtract(b.hasOnlyDivLocals() ? b : b.computeReprWithOnlyDivLocals());
                auto actual = fs::subtractSymbolicRelations(a, b, schema, Tuple::Event, Tuple::Event, bound);
                if (!actual.isEqual(raw)) { return false; }
                if (!bound && literal && matching) {
                    for (int coordinate : {-2, 0, 2}) {
                        SmallVector<Int> point{Int(0), Int(0), Int(1), Int(2), Int(coordinate), Int(0), Int(1)};
                        bool expected = coordinate < 0;
                        if (actual.containsPoint(point) != expected || raw.containsPoint(point) != expected) {
                            llvm::errs() << "tiny raw subtraction integer-QE mismatch coordinate=" << coordinate
                                         << " actual=" << actual.containsPoint(point)
                                         << " generic=" << raw.containsPoint(point) << " expected=" << expected << "\n";
                            return false;
                        }
                    }
                }
            }
        }
    }
    for (auto shape : {std::pair<Tuple, Tuple>{Tuple::Unit, Tuple::Event},
                       {Tuple::Cell, Tuple::Unit}}) {
        auto space = *schema->space(shape.first, shape.second);
        R a{I(space)};
        I piece(space);
        piece.addBound(presburger::BoundType::EQ, piece.getNumDimVars(), Int(1));
        R b(piece);
        auto actual = fs::subtractSymbolicRelations(a, b, schema, shape.first, shape.second);
        if (!actual.isEqual(a.subtract(b))) { return false; }
    }
    return true;
}
bool scenario(MLIRContext& context, bool empty, bool onePipe, bool sourcesSharePipe = false)
{
    std::string source(fixture);
    if (onePipe || sourcesSharePipe) {
        for (auto operation : {std::string("pto.txor ins(%x, %x, %z : !vec, !vec, !vec) outs(%y : !vec)"),
                               std::string("pto.tstore ins(%y : !vec) outs(%dst : !gm)")}) {
            if (!onePipe && operation.find("pto.txor") == 0) { continue; }
            auto position = source.find(operation);
            if (position == std::string::npos) { return false; }
            source.replace(position, operation.size(), "pto.tload ins(%src : !gm) outs(%x : !vec)");
        }
    }
    auto module = parseSourceString<ModuleOp>(source, &context);
    if (!module) { return false; }
    auto function = *module->getOps<func::FuncOp>().begin();
    pto::SyncInput input;
    if (failed(input.build(function)) || input.instructions().size() != 3) { return false; }
    fs::StructuredImportOptions options;
    options.parameters = {function.getArgument(2)};
    options.mathematicalArithmetic = [](Operation*, fs::SymbolicSchemaHandle, const fs::SymbolicPrimitive&) {
        return true; // Owned fixture has n in [0,2] and fixed positive progression.
    };
    auto imported = fs::StructuredInputAdapter::build(function, input, options,
        [&](fs::SymbolicSchemaHandle schema, ArrayRef<fs::StructuredSite>) { return effects(schema, empty); });
    if (!imported.succeeded()) { llvm::errs() << imported.issue.reason << "\n"; return false; }
    auto schema = imported.input->schema();
    if (!empty && !onePipe && !sourcesSharePipe) {
        llvm::errs() << "composition: tiny raw subtraction before full scenario\n";
        if (!subtractionOracle(schema)) { return false; }
    }
    auto analysis = fs::SymbolicDemandAnalysis::build(schema, imported.input->relations());
    if (failed(analysis)) { return false; }
    auto evaluator = fs::SymbolicEvaluator::uniform(*analysis);
    if (failed(evaluator)) { return false; }
    auto native = materialize(*evaluator, (*analysis)->native());
    auto identity = materialize(*evaluator, (*analysis)->identity());
    auto generators = materialize(*evaluator, (*analysis)->generators());
    llvm::errs() << "composition: uniform evaluator reachability\n";
    auto reach = materialize(*evaluator, (*analysis)->reachability());
    auto present = materialize(*evaluator, (*analysis)->presence());
    if (failed(native) || failed(identity) || failed(generators) || failed(reach) || failed(present)) { return false; }
    // Enumerate this bounded fixture independently: only coordinate zero can
    // execute for n in [0,2]. This avoids expanding a generic uniform H square.
    auto eventSpace = *schema->space(Tuple::Event, Tuple::Event);
    R goldenReach = R::getEmpty(eventSpace), covers = R::getEmpty(eventSpace);
    auto addPoint = [&](R& relation, Event a, Event b, unsigned n) {
        I piece(eventSpace);
        SmallVector<unsigned> point{a.site, a.coordinate, a.kind, b.site, b.coordinate, b.kind, n};
        for (unsigned column = 0; column < point.size(); ++column) {
            piece.addBound(presburger::BoundType::EQ, column, Int(point[column]));
        }
        relation.unionInPlace(piece);
    };
    for (unsigned n = 0; n <= 2; ++n) {
        SmallVector<Event> events;
        for (unsigned site = 0; site < 3; ++site) {
            if (n > (site == 1 ? 1U : 0U)) {
                events.push_back({site, 0, 0}); events.push_back({site, 0, 1});
            }
        }
        Graph baseline(events.size(), SmallVector<unsigned char>(events.size()));
        auto required = baseline;
        for (std::size_t a = 0; a < events.size(); ++a) {
            for (std::size_t b = 0; b < events.size(); ++b) {
                baseline[a][b] = contains(*native, events[a], events[b], n);
                required[a][b] = baseline[a][b] || contains(*generators, events[a], events[b], n);
            }
        }
        close(baseline); close(required);
        for (std::size_t a = 0; a < events.size(); ++a) {
            for (std::size_t b = 0; b < events.size(); ++b) {
                if (required[a][b]) { addPoint(goldenReach, events[a], events[b], n); }
                bool cover = required[a][b] && !baseline[a][b] && a != b;
                for (std::size_t k = 0; k < events.size(); ++k) {
                    if (k != a && k != b && required[a][k] && required[k][b]) { cover = false; }
                }
                if (cover) { addPoint(covers, events[a], events[b], n); }
            }
        }
    }
    llvm::errs() << "composition: finite golden comparison\n";
    if (!boundedEqual(*reach, goldenReach)) { return false; }
    SmallVector<fs::SelectedAnalysisHandle> children;
    for (unsigned site = 0; site < 3; ++site) {
        auto domain = siteSet(schema, site).getRangeSet();
        auto restrict = [&](const R& relation) { return relation.intersectDomain(domain).intersectRange(domain); };
        auto child = std::make_shared<fs::SelectedAnalysis>();
        child->kind = fs::SelectedAnalysis::Kind::General;
        child->sites = {site};
        child->general = std::make_shared<fs::GeneralQueries>(schema, imported.input->relations().context.relation(),
            present->intersectRange(domain), restrict(*native), restrict(covers), restrict(goldenReach));
        child->contract = fs::generalAnalysisContract();
        children.push_back(child);
    }
    fs::CostLedger costs;
    std::string reason;
    llvm::errs() << "composition: balanced production merge\n";
    auto result = fs::composeGeneralRegions(function, imported.input, children, costs, reason);
    if (failed(result)) { llvm::errs() << reason << "\n"; return false; }
    llvm::errs() << "composition: merged reachability bounded comparison\n";
    if (!boundedEqual((*result)->general->reachability, goldenReach)) { return false; }
    llvm::errs() << "composition: merged minimum bounded comparison\n";
    if (!boundedEqual((*result)->general->minimum, covers)) { return false; }
    llvm::errs() << "composition: merged independent closure/cover comparison\n";
    if (!finiteOracle(*(*result)->general, *native, *generators)) { return false; }
    if (!empty && !onePipe && !sourcesSharePipe) {
        llvm::errs() << "composition: tiny raw uniform typed joins\n";
        if (!joins(schema) || !typedJoins(schema, context)) { return false; }
    }
    auto wrong = std::make_shared<fs::SelectedAnalysis>(*children.front());
    auto previous = wrong->general;
    I foreignContext(*schema->space(Tuple::Unit, Tuple::Unit));
    foreignContext.addBound(presburger::BoundType::EQ, 0, Int(1));
    wrong->general = std::make_shared<fs::GeneralQueries>(schema, R(foreignContext), previous->present,
        previous->native, previous->minimum, previous->reachability);
    children.front() = wrong;
    return failed(fs::composeGeneralRegions(function, imported.input, children, costs, reason)) &&
        reason == "general child query uses a different original schema/context";
}
} // namespace
int runGeneralCompositionChecks(mlir::MLIRContext& context)
{
    if (!normalizationOracle()) { return 1; }
    for (bool empty : {false, true}) {
        for (bool onePipe : {false, true}) {
            if (!scenario(context, empty, onePipe)) {
                llvm::errs() << "general composition uniform/finite oracle failed: empty=" << empty
                             << " onePipe=" << onePipe << "\n";
                return 1;
            }
        }
    }
    if (!scenario(context, false, false, true)) {
        llvm::errs() << "general composition shared-source-pipe oracle failed\n";
        return 1;
    }
    llvm::outs() << "general composition: raw uniform joins and finite closure/cover oracles passed\n";
    return 0;
}
