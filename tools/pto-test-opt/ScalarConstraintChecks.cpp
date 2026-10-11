// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Check path isolation and machine-semantic range proofs independently of
// recognizer selection, including the literal prefill wrapping counterexample.
#include "../../lib/PTO/Transforms/InsertSync/SyncScalarEvolution.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/CountedLoop.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Arithmetic/ArithmeticProgramInternal.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>
using namespace mlir;
bool runMachinePlaceholderChecks(MLIRContext* context);
namespace {
struct Case {
    const char* name;
    std::string body;
    int64_t lower, upper;
    unsigned bits = 64;
    enum Expression { Any, X, XPlusOne, Opaque, Scaled } expression = Any;
};
std::string probe(StringRef value = "%x")
{
    return "%probe = arith.addi " + value.str() + ", %zero {test.probe} : index\n";
}
bool check(MLIRContext* context, const Case& test)
{
    std::string text;
    llvm::raw_string_ostream out(text);
    out << "module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, " << test.bits
        << " : i32>>} { func.func @case(%x: index, %a: i32, %b: i32) {\n"
        << "%zero = arith.constant 0 : index\n%one = arith.constant 1 : index\n"
        << "%three = arith.constant 3 : index\n%four = arith.constant 4 : index\n"
        << "%hundred = arith.constant 100 : index\n"
        << "%min = arith.constant " << (test.bits == 64 ? INT64_MIN : INT32_MIN) << " : index\n"
        << "%max = arith.constant " << (test.bits == 64 ? INT64_MAX : INT32_MAX) << " : index\n"
        << test.body << "\nreturn\n}}";
    auto module = parseSourceString<ModuleOp>(text, context);
    if (!module || failed(verify(*module))) {
        return false;
    }
    Operation* anchor = nullptr;
    module->walk([&](Operation* operation) {
        if (operation->hasAttr("test.probe")) {
            anchor = operation;
        }
    });
    if (!anchor) {
        return false;
    }
    const auto before = [&]() {
        std::string source;
        llvm::raw_string_ostream stream(source);
        module->print(stream);
        return source;
    }();
    auto function = module->lookupSymbol<func::FuncOp>("case");
    pto::detail::ScalarEvolution evolution(context, anchor);
    auto input = anchor->getOperand(0);
    auto expectedRange = std::make_pair(test.lower, test.upper);
    auto symbol = [&](Value value) { return getAffineSymbolExpr(value == function.getArgument(0) ? 0 : 1, context); };
    // Exercise both query orders and memoized repeats: range-only queries must
    // never contaminate the expression cache with their placeholder symbols.
    for (unsigned repeat = 0; repeat < 2; ++repeat) {
        pto::detail::ScalarEvolution expressionFirst(context, anchor);
        auto firstExpression = expressionFirst.value(input, symbol);
        auto firstRange = expressionFirst.signedRange(input);
        auto range = evolution.signedRange(input);
        auto expression = evolution.value(input, symbol);
        if (firstExpression != expression || firstRange != range) {
            return false;
        }
        AffineExpr expected;
        auto x = getAffineSymbolExpr(0, context), opaque = getAffineSymbolExpr(1, context);
        switch (test.expression) {
            case Case::Any:
                break;
            case Case::X:
                expected = x;
                break;
            case Case::XPlusOne:
                expected = x + 1;
                break;
            case Case::Opaque:
                expected = opaque;
                break;
            case Case::Scaled:
                expected = opaque * 4;
                break;
        }
        if (range != expectedRange || (expected && expression != expected)) {
            llvm::errs() << test.name << ": unexpected range/expression ";
            if (range) {
                llvm::errs() << range->first << "," << range->second;
            }
            if (expression) {
                llvm::errs() << " " << expression;
            }
            llvm::errs() << "\n";
            return false;
        }
    }
    std::string after;
    llvm::raw_string_ostream stream(after);
    module->print(stream);
    return before == after;
}
bool interleavedContexts(MLIRContext* context)
{
    auto module = parseSourceString<ModuleOp>(
        R"(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
  func.func @contexts(%x: index) {
    %zero = arith.constant 0 : index
    %condition = arith.cmpi sgt, %x, %zero : index
    scf.if %condition {
      %then = arith.addi %x, %zero {test.probe} : index
    } else {
      %else = arith.addi %x, %zero {test.probe} : index
    }
    %root = arith.addi %x, %zero {test.probe} : index
    return
  }
})",
        context);
    if (!module || failed(verify(*module))) {
        return false;
    }
    SmallVector<Operation*> anchors;
    module->walk([&](Operation* operation) {
        if (operation->hasAttr("test.probe")) {
            anchors.push_back(operation);
        }
    });
    if (anchors.size() != 3) {
        return false;
    }
    pto::detail::ScalarEvolution thenContext(context, anchors[0]), elseContext(context, anchors[1]),
        rootContext(context, anchors[2]);
    auto x = anchors[0]->getOperand(0);
    auto thenRange = std::make_pair(int64_t{1}, int64_t{INT32_MAX});
    auto elseRange = std::make_pair(int64_t{INT32_MIN}, int64_t{0});
    auto rootRange = std::make_pair(int64_t{INT32_MIN}, int64_t{INT32_MAX});
    return thenContext.signedRange(x) == thenRange && rootContext.signedRange(x) == rootRange &&
           elseContext.signedRange(x) == elseRange && thenContext.signedRange(x) == thenRange &&
           elseContext.signedRange(x) == elseRange && rootContext.signedRange(x) == rootRange;
}
bool controlProofContext(MLIRContext* context)
{
    auto module = parseSourceString<ModuleOp>(
        R"(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64 : i32>>} {
  func.func @control() {
    %lower = arith.constant -4 : index
    %upper = arith.constant 8 : index
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    scf.for %i = %lower to %upper step %one {
      %quotient = arith.divui %i, %four : index
      %nonnegative = arith.cmpi sge, %i, %zero : index
      %positive = arith.cmpi sgt, %quotient, %zero : index
      scf.if %nonnegative {
        scf.if %positive {
          %probe = arith.addi %quotient, %zero {test.probe} : index
        }
      }
    }
    return
  }
})",
        context);
    if (!module || failed(verify(*module))) {
        return false;
    }
    auto function = module->lookupSymbol<func::FuncOp>("control");
    Operation* anchor = nullptr;
    scf::ForOp loop;
    module->walk([&](Operation* operation) {
        if (operation->hasAttr("test.probe")) {
            anchor = operation;
        }
        if (auto found = dyn_cast<scf::ForOp>(operation)) {
            loop = found;
        }
    });
    if (!anchor || !loop) {
        return false;
    }
    namespace fs = pto::frontiersynch;
    fs::PhaseIndex index;
    if (failed(index.build(function, ArrayRef<const pto::CompoundInstanceElement*>{}))) {
        return false;
    }
    pto::CompoundInstanceElement phase(0, {}, {}, pto::PipelineType::PIPE_S, anchor->getName());
    phase.elementOp = anchor;
    fs::ArithmeticSite site{&phase, {loop}, {}};
    fs::ArithmeticProgram output;
    output.context.function = function;
    output.context.root = function;
    fs::ArithmeticLimits limits;
    fs::detail::ProgramBuilder builder{output, limits, context, index, DenseMap<Value, unsigned>(), {}};
    auto quotient = anchor->getOperand(0);
    auto comparison = cast<scf::IfOp>(anchor->getParentOp()).getCondition();
    auto expected = getAffineDimExpr(0, context).floorDiv(4);
    // Geometry is exact on executed nonnegative visits. The presence formula
    // itself must not borrow that assumption: its separate machine recipe
    // retains unsigned division on negative visits, without a free parameter.
    auto geometry = builder.value(quotient, site, 0);
    auto definition = builder.value(quotient, site, 0, fs::detail::ProgramBuilder::ScalarProofContext::Definition);
    return builder.prepareValue(quotient, site) && geometry == expected && definition && definition != geometry &&
           builder.prepareGuard(comparison, site) && !builder.machineBindings.empty() && output.parameters.empty();
}
bool failedPrimitiveStopsWork(MLIRContext* context)
{
    auto module = parseSourceString<ModuleOp>("module { func.func @failed() { return } }", context);
    if (!module || failed(verify(*module))) { return false; }
    auto function = module->lookupSymbol<func::FuncOp>("failed");
    namespace fs = pto::frontiersynch;
    fs::PhaseIndex index;
    if (failed(index.build(function, ArrayRef<const pto::CompoundInstanceElement*>{}))) { return false; }
    fs::ArithmeticProgram output;
    output.context = {function, function};
    const fs::ArithmeticLimits limits{8, 8, 2, 8};
    fs::detail::ProgramBuilder builder{output, limits, context, index, DenseMap<Value, unsigned>(), {}};
    auto relation = builder.relation(fs::PrimitiveKind::Context, 0);
    builder.emit(relation, {});
    if (relation.pieces.size() != 1) { return false; }
    // A machine placeholder cannot use this period-two representation. This
    // is an actual primitive failure, not an injected diagnostic or timer.
    builder.emit(relation, {getAffineSymbolExpr(limits.dimensions, context)});
    if (output.extraction.state == fs::RecognitionState::Applicable ||
        output.extraction.diagnostics.size() != 1 || output.machinePrimitiveConversions) { return false; }
    const auto pieces = relation.pieces.size();
    builder.emit(relation, {});
    SmallVector<SmallVector<AffineExpr>> recipes;
    builder.emitForSites(relation, {}, {}, &recipes);
    if (relation.pieces.size() != pieces || !recipes.empty() ||
        output.extraction.diagnostics.size() != 1 || output.machinePrimitiveConversions) { return false; }
    // Failure belongs to this producer attempt. A fresh context can still
    // construct the supported representation without inheriting that failure.
    fs::ArithmeticProgram fresh;
    fresh.context = {function, function};
    fs::detail::ProgramBuilder retry{fresh, limits, context, index, DenseMap<Value, unsigned>(), {}};
    auto accepted = retry.relation(fs::PrimitiveKind::Context, 0);
    retry.emit(accepted, {});
    return fresh.extraction.state == fs::RecognitionState::Applicable && accepted.pieces.size() == 1;
}
std::optional<pto::frontiersynch::PrimitiveRelation> configuredPrimitive(
    pto::frontiersynch::detail::ProgramBuilder& builder, MLIRContext* context, bool machine, int64_t scale)
{
    namespace fs = pto::frontiersynch;
    auto relation = builder.relation(fs::PrimitiveKind::Context, 2);
    if (machine) {
        auto system = fs::IntegerSystem::create(2, {},
            {{{fs::BoundInteger(1), fs::BoundInteger(0)}, fs::BoundInteger(0), fs::BoundInteger(2)},
             {{fs::BoundInteger(0), fs::BoundInteger(1)}, fs::BoundInteger(0), fs::BoundInteger(2)}});
        if (failed(system)) { return std::nullopt; }
        fs::ResiduePiece piece;
        piece.residues = {0, 0};
        piece.machineSystem = std::make_shared<const fs::IntegerSystem>(std::move(*system));
        relation.pieces.push_back(std::move(piece));
    } else {
        auto a = getAffineDimExpr(0, context), b = getAffineDimExpr(1, context);
        builder.emit(relation, {a * (3 * scale) + b * (2 * scale)});
    }
    return relation;
}
bool sameMachineSystem(const std::shared_ptr<const pto::frontiersynch::IntegerSystem>& a,
                       const std::shared_ptr<const pto::frontiersynch::IntegerSystem>& b)
{
    if (!a || !b) { return !a && !b; }
    if (a->dimensions() != b->dimensions() || a->constraints().size() != b->constraints().size() ||
        a->congruences().size() != b->congruences().size()) { return false; }
    for (std::size_t i = 0; i < a->constraints().size(); ++i) {
        if (a->constraints()[i].coefficients != b->constraints()[i].coefficients ||
            a->constraints()[i].bound != b->constraints()[i].bound) { return false; }
    }
    for (std::size_t i = 0; i < a->congruences().size(); ++i) {
        const auto& x = a->congruences()[i];
        const auto& y = b->congruences()[i];
        if (x.coefficients != y.coefficients || x.residue != y.residue || x.modulus != y.modulus) { return false; }
    }
    return true;
}
bool sameRecognition(const pto::frontiersynch::ArithmeticRecognition& a,
                     const pto::frontiersynch::ArithmeticRecognition& b)
{
    if (a.state != b.state || a.arithmeticClass != b.arithmeticClass ||
        a.observedDimensions != b.observedDimensions || a.observedCoefficient != b.observedCoefficient ||
        !a.diagnostics.empty() || !b.diagnostics.empty() || a.pieces.size() != b.pieces.size()) { return false; }
    for (std::size_t i = 0; i < a.pieces.size(); ++i) {
        const auto& x = a.pieces[i];
        const auto& y = b.pieces[i];
        if (x.relation != y.relation || x.piece != y.piece || x.empty != y.empty ||
            x.rows.size() != y.rows.size() || !sameMachineSystem(x.machineSystem, y.machineSystem)) { return false; }
        for (std::size_t j = 0; j < x.rows.size(); ++j) {
            if (x.rows[j].coefficients != y.rows[j].coefficients || x.rows[j].constant != y.rows[j].constant ||
                x.rows[j].equality != y.rows[j].equality) { return false; }
        }
    }
    return true;
}
bool checkConfiguredPrimitive(MLIRContext* context, func::FuncOp function,
    const pto::frontiersynch::PhaseIndex& index, pto::frontiersynch::ArithmeticLimits limits,
    bool machine, int64_t scale, std::optional<pto::frontiersynch::ArithmeticIssue> issue)
{
    namespace fs = pto::frontiersynch;
    fs::ArithmeticProgram output;
    output.context = {function, function};
    fs::detail::ProgramBuilder builder{output, limits, context, index, DenseMap<Value, unsigned>(), {}};
    auto relation = configuredPrimitive(builder, context, machine, scale);
    if (!relation || builder.appendPrimitive(std::move(*relation)) != !issue.has_value()) { return false; }
    if (!issue) {
        for (auto kind : {fs::PrimitiveKind::Occurrences, fs::PrimitiveKind::Order, fs::PrimitiveKind::Native,
                         fs::PrimitiveKind::Reads, fs::PrimitiveKind::Writes, fs::PrimitiveKind::Prerequisites}) {
            if (!builder.appendPrimitive(builder.relation(kind, 0))) { return false; }
        }
        fs::finishArithmeticRecognition(output.primitives, limits, output.recognition);
        return output.recognition.state == fs::RecognitionState::Applicable &&
            sameRecognition(output.recognition, fs::recognizeArithmetic(output.primitives, limits));
    }
    if (output.recognition.state == fs::RecognitionState::Applicable || !output.recognition.pieces.empty() ||
        output.recognition.diagnostics.size() != 1 || output.recognition.diagnostics.front().issue != *issue) {
        return false;
    }
    const auto relations = output.primitives.relations.size();
    auto later = builder.relation(fs::PrimitiveKind::Context, 0);
    builder.emit(later, {});
    SmallVector<SmallVector<AffineExpr>> recipes;
    builder.emitForSites(later, {}, {}, &recipes);
    return later.pieces.empty() && recipes.empty() && !builder.appendPrimitive(std::move(later)) &&
        output.primitives.relations.size() == relations && !output.machinePrimitiveConversions;
}
bool configuredPrimitiveStopsWork(MLIRContext* context)
{
    auto module = parseSourceString<ModuleOp>("module { func.func @configured() { return } }", context);
    if (!module || failed(verify(*module))) { return false; }
    auto function = module->lookupSymbol<func::FuncOp>("configured");
    namespace fs = pto::frontiersynch;
    fs::PhaseIndex index;
    if (failed(index.build(function, ArrayRef<const pto::CompoundInstanceElement*>{}))) { return false; }
    return checkConfiguredPrimitive(context, function, index, {8, 3, 1, 3}, true, 1,
               fs::ArithmeticIssue::DimensionLimit) &&
        checkConfiguredPrimitive(context, function, index, {8, 4, 1, 2}, false, 1,
               fs::ArithmeticIssue::CoefficientLimit) &&
        checkConfiguredPrimitive(context, function, index, {8, 4, 1, 3}, true, 1, std::nullopt) &&
        checkConfiguredPrimitive(context, function, index, {8, 4, 1, 3}, false, 1, std::nullopt) &&
        checkConfiguredPrimitive(context, function, index, {8, 4, 1, 3}, false, 2, std::nullopt);
}
bool correlatedCountedDomain(MLIRContext* context, int64_t offset, int64_t stride, bool swapped, unsigned bits = 64)
{
    namespace fs = pto::frontiersynch;
    const auto operands = swapped ? "%delta, %lower" : "%lower, %delta";
    std::string source = "module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, " +
        std::to_string(bits) + " : i32>>} { func.func @counted(%lower: index, %foreign: index) { "
        "%delta = arith.constant " + std::to_string(offset) + " : index\n"
        "%step = arith.constant " + std::to_string(stride) + " : index\n"
        "%upper = arith.addi " + operands + " : index\n"
        "scf.for %iv = %lower to %upper step %step {} return } }";
    auto module = parseSourceString<ModuleOp>(source, context);
    if (!module || failed(verify(*module))) { return false; }
    auto function = module->lookupSymbol<func::FuncOp>("counted");
    scf::ForOp loop;
    function.walk([&](scf::ForOp value) { loop = value; });
    auto domain = fs::CountedLoop::get(loop);
    if (bits != 64) { return !domain; }
    if (offset < 0) { return static_cast<bool>(domain) == (stride > 1); }
    if (!domain) { return false; }
    const uint64_t maximum = offset ? (static_cast<uint64_t>(offset) - 1) / stride + 1 : 0;
    if (domain->maximumOrdinal != (maximum ? maximum - 1 : 0)) { return false; }
    fs::RegionExpressions arena;
    const auto trips = domain->trips(arena);
    const auto lowerInput = arena.input(loop.getLowerBound()), upperInput = arena.input(loop.getUpperBound());
    for (int64_t lower : {INT64_MIN, INT64_MIN + 1, int64_t{-11}, int64_t{-1}, int64_t{0},
                         int64_t{1}, INT64_MAX - 8, INT64_MAX - 1, INT64_MAX}) {
        APInt low(64, static_cast<uint64_t>(lower)), high = low + APInt(64, static_cast<uint64_t>(offset));
        APInt distance = high.sgt(low) ? high.sext(128) - low.sext(128) : APInt(128, 0);
        APInt expected = distance.isZero() ? distance : (distance - 1).udiv(APInt(128, stride)) + 1;
        fs::RegionExpressions::Substitution binding({{lowerInput, arena.constant(low.getZExtValue())},
                                                    {upperInput, arena.constant(high.getZExtValue())}});
        if (arena.constantValue(arena.substitute(trips, binding)) != expected.getZExtValue() ||
            expected.getZExtValue() > maximum) { return false; }
    }
    // A certified selected view must use its supplied constants, not the
    // original correlated expression; unrelated upper operands remain unknown.
    auto selected = fs::CountedLoop::get(loop, 0, 100);
    if (!selected || selected->maximumOrdinal != 99 / static_cast<uint64_t>(stride)) { return false; }
    auto lowerOnly = fs::CountedLoop::get(loop, 0);
    auto upperOnly = fs::CountedLoop::get(loop, std::nullopt, 0);
    if (!lowerOnly || static_cast<bool>(upperOnly) != (stride > 1)) { return false; }
    auto sum = loop.getUpperBound().getDefiningOp<arith::AddIOp>();
    sum->setOperand(swapped ? 1 : 0, function.getArgument(1));
    return static_cast<bool>(fs::CountedLoop::get(loop)) == (stride > 1);
}
bool correlatedCountedDomains(MLIRContext* context)
{
    if (!correlatedCountedDomain(context, 8, 1, false, 32)) { return false; }
    for (int64_t offset : {int64_t{-1}, int64_t{0}, int64_t{1}, int64_t{8}, INT64_MAX}) {
        for (int64_t step : {int64_t{1}, int64_t{3}, INT64_MAX}) {
            if (!correlatedCountedDomain(context, offset, step, false) ||
                !correlatedCountedDomain(context, offset, step, true)) { return false; }
        }
    }
    return true;
}

struct ModularBandCase {
    int64_t coefficientA, coefficientB, offsetA, offsetB;
    int64_t strideA, strideB, sizeA, sizeB;
    unsigned bytes;
    bool disjoint, floorOrigin = false;
    int64_t rowStride = 0, rows = 1;
};
bool modularBandCase(MLIRContext* context, Value base, Value iv, const ModularBandCase& test,
                     uint64_t& comparisons)
{
    auto symbol = getAffineSymbolExpr(0, context), dimension = getAffineDimExpr(0, context);
    if (test.floorOrigin) { symbol = symbol.floorDiv(3); }
    auto region = [&](int64_t coefficient, int64_t offset, int64_t stride, int64_t size) {
        pto::SyncAccessRegion value;
        value.base = base;
        value.byteOffset = symbol * coefficient + offset + dimension * stride +
            getAffineDimExpr(1, context) * test.rowStride;
        value.extents = {getAffineConstantExpr(size, context), getAffineConstantExpr(test.rows, context)};
        value.symbols = {iv};
        value.elementBytes = test.bytes;
        return value;
    };
    auto a = region(test.coefficientA, test.offsetA, test.strideA, test.sizeA);
    auto b = region(test.coefficientB, test.offsetB, test.strideB, test.sizeB);
    if (pto::regionsProvablyDisjoint(a, b) != test.disjoint ||
        pto::regionsProvablyDisjoint(b, a) != test.disjoint) { return false; }
    auto footprint = [&](int64_t original, int64_t coefficient, int64_t offset, int64_t stride, int64_t size) {
        const int64_t row = test.floorOrigin ? original / 3 - (original % 3 < 0) : original;
        std::set<uint64_t> bytes;
        for (int64_t rowIndex = 0; rowIndex < test.rows; ++rowIndex) {
            for (int64_t coordinate = 0; coordinate < size; ++coordinate) {
                const uint64_t start = static_cast<uint64_t>(row) * static_cast<uint64_t>(coefficient) +
                    static_cast<uint64_t>(offset) + static_cast<uint64_t>(coordinate) * static_cast<uint64_t>(stride) +
                    static_cast<uint64_t>(rowIndex) * static_cast<uint64_t>(test.rowStride);
                for (unsigned byte = 0; byte < test.bytes; ++byte) { bytes.insert(start + byte); }
            }
        }
        return bytes;
    };
    for (int64_t x : {INT64_MIN, INT64_MIN + 1, int64_t{-3}, int64_t{-1}, int64_t{0}, int64_t{1}, INT64_MAX}) {
        auto left = footprint(x, test.coefficientA, test.offsetA, test.strideA, test.sizeA);
        for (int64_t y : {INT64_MIN, INT64_MIN + 1, int64_t{-3}, int64_t{-1}, int64_t{0}, int64_t{1}, INT64_MAX}) {
            auto right = footprint(y, test.coefficientB, test.offsetB, test.strideB, test.sizeB);
            for (auto byte : left) {
                ++comparisons;
                if (test.disjoint && right.count(byte)) { return false; }
            }
        }
    }
    return true;
}
bool modularBandAliases(MLIRContext* context, bool sameBase, bool overlapping, pto::GMAliasPolicy policy)
{
    const std::string tile = "!pto.tile_buf<vec, 1x64xbf16, valid=1x64>";
    std::string source = "module attributes {pto.target_arch = \"a3\"} {func.func @bands(";
    source += "%a: !pto.ptr<bf16, gm>, %b: !pto.ptr<bf16, gm>) {\n";
    source += "%zero = arith.constant 0 : index\n%one = arith.constant 1 : index\n";
    source += "%eight = arith.constant 8 : index\n%width = arith.constant 128 : index\n";
    source += "%height = arith.constant 1024 : index\n%size = arith.constant 64 : index\n";
    source += "%shift = arith.constant " + std::to_string(overlapping ? 32 : 64) + " : index\n";
    source += "%address = arith.constant 0 : i64\n%tile = pto.alloc_tile addr = %address : " + tile + "\n";
    source += "%x = pto.make_tensor_view %a, shape = [%height, %width], strides = [%width, %one] "
              ": !pto.tensor_view<?x?xbf16>\n";
    source += "%y = pto.make_tensor_view " + std::string(sameBase ? "%a" : "%b") +
              ", shape = [%height, %width], strides = [%width, %one] : !pto.tensor_view<?x?xbf16>\n";
    source += "scf.for %i = %zero to %eight step %one {\n";
    source += "%u = pto.partition_view %x, offsets = [%i, %zero], sizes = [%one, %size] "
              ": !pto.tensor_view<?x?xbf16>\n";
    source += "%v = pto.partition_view %y, offsets = [%i, %shift], sizes = [%one, %size] "
              ": !pto.tensor_view<?x?xbf16>\n";
    for (const auto* view : {"%u", "%v"}) {
        source += "pto.tstore ins(%tile : " + tile + ") outs(" + view +
                  " : !pto.partition_tensor_view<1x64xbf16>)\n";
    }
    source += "} return }}";
    auto module = parseSourceString<ModuleOp>(source, context);
    if (!module || failed(verify(*module))) { return false; }
    pto::SyncInput input(policy);
    if (failed(input.build(module->lookupSymbol<func::FuncOp>("bands")))) { return false; }
    SmallVector<std::size_t> globals;
    for (auto [id, effect] : llvm::enumerate(input.accesses().effects())) {
        if (effect.memory->scope == pto::AddressSpace::GM && effect.mode == pto::SyncAccessMode::Write) {
            globals.push_back(id);
        }
    }
    const bool expected = sameBase ? overlapping : policy == pto::GMAliasPolicy::MayAlias;
    return globals.size() == 2 && input.accesses().mayConflict(globals[0], globals[1]) == expected &&
        input.accesses().mayConflict(globals[1], globals[0]) == expected;
}
bool modularBandChecks(MLIRContext* context)
{
    auto module = parseSourceString<ModuleOp>(R"mlir(module {
      func.func @bands(%a: !pto.ptr<i8, gm>, %b: !pto.ptr<i8, gm>) {
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        scf.for %i = %zero to %one step %one { }
        return
      }
    })mlir", context);
    if (!module || failed(verify(*module))) { return false; }
    auto function = module->lookupSymbol<func::FuncOp>("bands");
    scf::ForOp loop;
    function.walk([&](scf::ForOp value) { loop = value; });
    const SmallVector<ModularBandCase> cases{
        {256, 256, 0, 128, 2, 2, 64, 64, 2, true},
        {256, 512, 0, 128, 2, 2, 64, 64, 2, true},
        {256, -256, 0, 128, 2, 2, 64, 64, 2, true},
        {256, 256, 0, 126, 2, 2, 64, 64, 2, false},
        {256, 256, 0, 128, 4, 2, 32, 64, 2, true},
        {16, 16, -2, 2, 1, 1, 4, 4, 1, true},
        {16, 16, -2, 1, 1, 1, 4, 4, 1, false},
        {16, 16, 0, 8, 1, 1, 16, 4, 1, false},
        {255, 256, 0, 128, 2, 2, 64, 64, 2, false},
        {256, 256, INT64_MIN, 128, 2, 2, 64, 64, 2, false},
        {256, 256, 0, 128, -2, 2, 64, 64, 2, false},
        {256, 256, 0, 128, 2, 2, 64, 64, 2, true, true},
        {1, 1, 0, 128, 1, 1, 4, 4, 1, false},
        {256, 256, 0, 128, 2, 2, 64, 64, 2, true, false, 256, 8},
        {256, 256, 0, 128, 2, 2, 64, 64, 2, true, false, 512, 8},
        {256, 256, 0, 128, 2, 2, 64, 64, 2, true, false, 0, 8},
        {256, 256, 0, 128, 2, 2, 64, 64, 2, false, false, 257, 8},
        {256, 256, 0, 126, 2, 2, 64, 64, 2, false, false, 256, 8},
        {256, 256, -128, 0, 2, 2, 64, 64, 2, true, false, 256, 8},
        {256, 256, 0, 128, 2, 2, 128, 64, 2, false, false, 256, 8}};
    uint64_t comparisons = 0;
    for (const auto& test : cases) {
        if (!modularBandCase(context, function.getArgument(0), loop.getInductionVar(), test, comparisons)) {
            llvm::errs() << "modular-band independent-visit case failed\n"; return false;
        }
    }
    auto symbol = getAffineSymbolExpr(0, context), dimension = getAffineDimExpr(0, context);
    pto::SyncAccessRegion a{function.getArgument(0), symbol * 256 + dimension,
                           {getAffineConstantExpr(4, context)}, {loop.getInductionVar()}, 1, {}};
    auto b = a;
    b.byteOffset = b.byteOffset + 128;
    b.base = function.getArgument(1);
    if (pto::regionsProvablyDisjoint(a, b)) { return false; }
    b.base = {};
    if (pto::regionsProvablyDisjoint(a, b)) { return false; }
    b.base = a.base;
    b.extents = {symbol};
    if (pto::regionsProvablyDisjoint(a, b)) { return false; }
    b.extents = {getAffineConstantExpr(0, context)};
    if (!pto::regionsProvablyDisjoint(a, b)) { return false; }
    b.extents = a.extents;
    auto layout = parseAttribute("#dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>", context);
    if (!layout) { return false; }
    (*module)->setAttr("dlti.dl_spec", layout);
    if (pto::regionsProvablyDisjoint(a, b)) { return false; }
    (*module)->removeAttr("dlti.dl_spec");
    for (auto policy : {pto::GMAliasPolicy::MayNotAlias, pto::GMAliasPolicy::MayAlias}) {
        for (bool sameBase : {false, true}) {
            for (bool overlapping : {false, true}) {
                if (!modularBandAliases(context, sameBase, overlapping, policy)) { return false; }
            }
        }
    }
    llvm::outs() << "modular bands: " << comparisons
                 << " independent original byte checks and eight alias cases passed\n";
    return true;
}
} // namespace
bool runScalarConstraintChecks(MLIRContext* context)
{
    std::vector<Case> cases;
    for (unsigned bits : {32U, 64U}) {
        int64_t low = bits == 64 ? INT64_MIN : INT32_MIN;
        int64_t high = bits == 64 ? INT64_MAX : INT32_MAX;
        cases.push_back({"unsigned quotient range", "%q = arith.divui %x, %four : index\n" + probe("%q"),
            0, bits == 64 ? (int64_t(1) << 62) - 1 : (int64_t(1) << 30) - 1, bits, Case::Opaque});
        cases.push_back(
            {"then", "%g = arith.cmpi sgt, %x, %zero : index\nscf.if %g {" + probe() + "}", 1, high, bits, Case::X});
        cases.push_back(
            {"else", "%g = arith.cmpi sgt, %x, %zero : index\nscf.if %g {} else {" + probe() + "}", low, 0, bits});
        cases.push_back(
            {"swapped", "%g = arith.cmpi slt, %zero, %x : index\nscf.if %g {" + probe() + "}", 1, high, bits});
        cases.push_back(
            {"sibling", "%g = arith.cmpi sgt, %x, %zero : index\nscf.if %g {} else {" + probe() + "}", low, 0, bits});
        cases.push_back(
            {"post-merge", "%g = arith.cmpi sgt, %x, %zero : index\nscf.if %g {}\n" + probe(), low, high, bits});
        cases.push_back(
            {"unsigned-low", "%g = arith.cmpi ult, %x, %hundred : index\nscf.if %g {" + probe() + "}", 0, 99, bits});
        cases.push_back(
            {"unsigned-cross-sign", "%g = arith.cmpi uge, %x, %one : index\nscf.if %g {" + probe() + "}", low, high,
             bits});
        cases.push_back(
            {"unsigned-negative-half", "%g = arith.cmpi uge, %x, %min : index\nscf.if %g {" + probe() + "}", low, -1,
             bits});
        cases.push_back(
            {"unsigned-positive-half", "%g = arith.cmpi ult, %x, %min : index\nscf.if %g {" + probe() + "}", 0, high,
             bits});
        cases.push_back(
            {"signed-empty", "%g = arith.cmpi slt, %x, %min : index\nscf.if %g {" + probe() + "}", low, high, bits});
        cases.push_back(
            {"unsigned-empty",
             "%g = arith.cmpi ugt, %x, %max : index\n%h = arith.cmpi ult, %x, %zero : index\nscf.if %h {" + probe() +
                 "}",
             low, high, bits});
        std::string comparisons = "%g = arith.cmpi sge, %x, %zero : index\n%h = arith.cmpi slt, %x, %hundred : index\n";
        cases.push_back(
            {"and-true", comparisons + "%both = arith.andi %g, %h : i1\nscf.if %both {" + probe() + "}", 0, 99, bits});
        cases.push_back(
            {"and-false", comparisons + "%both = arith.andi %g, %h : i1\nscf.if %both {} else {" + probe() + "}", low,
             high, bits});
        cases.push_back(
            {"or-true", comparisons + "%either = arith.ori %g, %h : i1\nscf.if %either {" + probe() + "}", low, high,
             bits});
        cases.push_back(
            {"or-false",
             "%g = arith.cmpi slt, %x, %zero : index\n%h = arith.cmpi sge, %x, %hundred : index\n%either = arith.ori "
             "%g, %h : i1\nscf.if %either {} else {" +
                 probe() + "}",
             0, 99, bits});
        cases.push_back(
            {"contradictory",
             comparisons + "scf.if %g {scf.if %h {} else {\n%j = arith.cmpi slt, %x, %zero : index\nscf.if %j {" +
                 probe() + "}}}",
             low, high, bits});
        cases.push_back(
            {"result-does-not-prove-operation",
             "%y = arith.addi %x, %one : index\n%g = arith.cmpi sgt, %y, %zero : index\nscf.if %g {" + probe("%y") +
                 "}",
             1, high, bits, Case::Opaque});
        cases.push_back(
            {"bounded-operand-proves-operation",
             comparisons + "%both = arith.andi %g, %h : i1\n%y = arith.addi %x, %one : index\nscf.if %both {" +
                 probe("%y") + "}",
             1, 100, bits, Case::XPlusOne});
        cases.push_back(
            {"nested-loop",
             comparisons +
                 "%both = arith.andi %g, %h : i1\nscf.if %both {scf.for %i = %zero to %four step %one {\n%y = "
                 "arith.addi %x, %one : index\n" +
                 probe("%y") + "}}",
             1, 100, bits, Case::XPlusOne});
    }
    // Loaded i32 endpoints in a 64-bit index layout: only an enclosing
    // positivity guard can turn unsigned chunk division into affine division.
    std::string chunks =
        "%start = arith.index_cast %a : i32 to index\n%end = arith.index_cast %b : i32 to index\n%length = arith.subi "
        "%end, %start : index\n%sum = arith.addi %length, %three : index\n%count = arith.divui %sum, %four : "
        "index\n%positive = arith.cmpi sgt, %length, %zero : index\n";
    std::string loop =
        "scf.for %i = %zero to %count step %one {\n%scaled = arith.muli %i, %four : index\n" + probe("%scaled") + "}\n";
    cases.push_back({"guarded-chunks", chunks + "scf.if %positive {" + loop + "}", 0, 4294967292LL, 64, Case::Scaled});
    cases.push_back(
        {"literal-prefill-sibling", chunks + "scf.if %positive {}\n" + loop, INT64_MIN, INT64_MAX, 64, Case::Opaque});
    for (const auto& test : cases) {
        if (!check(context, test)) {
            return false;
        }
    }
    if (!interleavedContexts(context) || !controlProofContext(context) || !failedPrimitiveStopsWork(context) ||
        !configuredPrimitiveStopsWork(context) || !correlatedCountedDomains(context) || !modularBandChecks(context) ||
        !runMachinePlaceholderChecks(context)) {
        return false;
    }
    // Independent bit-vector witness: start=0, end=-5, iv=2^61 is an
    // executed visit under literal unsigned ceil-div, and 4*iv wraps signed.
    APInt length(64, static_cast<uint64_t>(-5LL));
    APInt count = (length + 3).udiv(APInt(64, 4));
    APInt ordinal(64, uint64_t{1} << 61);
    APInt scaled = ordinal * 4;
    if (!ordinal.ult(count) || scaled.getSExtValue() != INT64_MIN ||
        (length - scaled).getSExtValue() != INT64_MAX - 4) {
        return false;
    }
    llvm::outs() << "scalar constraints: " << cases.size()
                 << " occurrence-local checks and literal wrapping witness passed\n";
    return true;
}
