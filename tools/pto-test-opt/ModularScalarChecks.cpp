// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Compare normalized source arithmetic with APInt machine semantics. Small
// widths are exhaustive; the original prefill counterexample is checked at 64.
#include "PTO/Transforms/FrontierSynch/Shared/ModularScalar.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Arithmetic/ArithmeticProgramInternal.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
bool runWidth(MLIRContext* context, unsigned bits, uint64_t& count)
{
    const auto type = "i" + std::to_string(bits);
    std::string source = "module { func.func @f(%x: " + type + ", %i: " + type + ") {\n"
        "%three = arith.constant 3 : " + type + "\n%four = arith.constant 4 : " + type + "\n"
        "%sum = arith.addi %x, %three : " + type + "\n"
        "%n = arith.divui %sum, %four : " + type + "\n"
        "%scaled = arith.muli %i, %four : " + type + "\n"
        "%tail = arith.subi %x, %scaled : " + type + "\n"
        "%signed = arith.divsi %x, %three : " + type + "\n"
        "%rem = arith.remsi %x, %three : " + type + "\nreturn\n}}";
    auto module = parseSourceString<ModuleOp>(source, context);
    if (!module) { return false; }
    auto function = module->lookupSymbol<func::FuncOp>("f");
    SmallVector<Value> values;
    function.walk([&](Operation* op) {
        if (isa<arith::DivUIOp, arith::SubIOp, arith::DivSIOp, arith::RemSIOp>(op)) {
            values.push_back(op->getResult(0));
        }
    });
    const uint64_t limit = bits <= 8 ? uint64_t(1) << bits : 0;
    SmallVector<uint64_t> probes;
    if (limit) {
        for (uint64_t x = 0; x < limit; ++x) { probes.push_back(x); }
    } else {
        probes = {0, 1, 3, 4, UINT64_MAX, UINT64_MAX - 4, uint64_t(1) << (bits - 3),
                  uint64_t(1) << (bits - 1), (uint64_t(1) << (bits - 1)) - 1};
    }
    for (auto x : probes) {
        for (auto i : probes) {
            fs::RegionExpressions e;
            fs::ModularScalar scalar(e, function);
            auto binding = [&](Value v) -> std::optional<fs::RegionExpressions::Id> {
                if (v == function.getArgument(0)) { return e.constant(APInt(bits, x).getZExtValue()); }
                if (v == function.getArgument(1)) { return e.constant(APInt(bits, i).getZExtValue()); }
                return std::nullopt;
            };
            APInt a(bits, x), b(bits, i), three(bits, 3), four(bits, 4);
            SmallVector<APInt> expected{(a + three).udiv(four), a - b * four, a.sdiv(three), a.srem(three)};
            for (auto [v, want] : llvm::zip(values, expected)) {
                auto actual = scalar.value(v, binding);
                if (!actual || e.constantValue(*actual) != want.getZExtValue()) {
                    llvm::errs() << "modular scalar mismatch width=" << bits << " x=" << x << " i=" << i << "\n";
                    return false;
                }
                ++count;
            }
        }
    }
    return true;
}
bool casts(MLIRContext* context, uint64_t& count)
{
    auto module = parseSourceString<ModuleOp>(R"mlir(module attributes {
      dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
      func.func @f(%x: i8, %index: index) {
        %sign = arith.extsi %x : i8 to i32
        %zero = arith.extui %x : i8 to i32
        %signedIndex = arith.index_cast %x : i8 to index
        %unsignedIndex = arith.index_castui %x : i8 to index
        %wide = arith.index_cast %index : index to i64
        %small = arith.trunci %wide : i64 to i8
        return
      }
    })mlir", context);
    if (!module) { return false; }
    auto function = module->lookupSymbol<func::FuncOp>("f");
    SmallVector<Value> values;
    function.walk([&](Operation* op) {
        if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::IndexCastOp, arith::IndexCastUIOp, arith::TruncIOp>(op)) {
            values.push_back(op->getResult(0));
        }
    });
    for (uint64_t x = 0; x < 256; ++x) {
        for (uint64_t index : {uint64_t(0), uint64_t(1), uint64_t(INT32_MAX),
                              uint64_t(1) << 31, uint64_t(UINT32_MAX)}) {
            fs::RegionExpressions e;
            fs::ModularScalar scalar(e, function);
            auto binding = [&](Value value) -> std::optional<fs::RegionExpressions::Id> {
                if (value == function.getArgument(0)) { return e.constant(x); }
                if (value == function.getArgument(1)) { return e.constant(index); }
                return std::nullopt;
            };
            APInt a(8, x), b(32, index);
            SmallVector<APInt> expected{a.sext(32), a.zext(32), a.sext(32), a.zext(32), b.sext(64), b.trunc(8)};
            for (auto [value, wanted] : llvm::zip(values, expected)) {
                auto actual = scalar.value(value, binding);
                if (!actual || e.constantValue(*actual) != wanted.getZExtValue()) { return false; }
                ++count;
            }
        }
    }
    return true;
}
bool symbolic(MLIRContext* context, uint64_t& count)
{
    auto module = parseSourceString<ModuleOp>(R"mlir(module {
      func.func @f(%x: i64, %i: i64) {
        %four = arith.constant 4 : i64
        %three = arith.constant 3 : i64
        %sum = arith.addi %x, %three : i64
        %n = arith.divui %sum, %four : i64
        %scaled = arith.muli %i, %four : i64
        %tail = arith.subi %x, %scaled : i64
        %positive = arith.cmpi sgt, %tail, %three : i64
        return
      }
    })mlir", context);
    if (!module) { return false; }
    auto function = module->lookupSymbol<func::FuncOp>("f");
    fs::RegionExpressions e;
    fs::ModularScalar scalar(e, function);
    auto x = e.input(function.getArgument(0)), i = e.input(function.getArgument(1));
    auto binding = [&](Value v) -> std::optional<fs::RegionExpressions::Id> {
        if (v == function.getArgument(0)) { return x; }
        if (v == function.getArgument(1)) { return i; }
        return std::nullopt;
    };
    Value comparison;
    function.walk([&](arith::CmpIOp op) { comparison = op.getResult(); });
    auto predicate = scalar.value(comparison, binding);
    auto universe = fs::IntegerSystem::create(2, {});
    if (!predicate || failed(universe)) { return false; }
    std::string error;
    auto relation = e.integerRelation(*predicate, {x, i}, *universe, error);
    if (failed(relation)) { llvm::errs() << "modular symbolic lowering: " << error << "\n"; return false; }
    std::vector<fs::IntegerConstraint> rows(4);
    for (auto& row : rows) { row.coefficients.resize(2); }
    rows[0].coefficients[0] = fs::BoundInteger(1); rows[0].bound = fs::BoundInteger(int64_t(1) << 32);
    rows[1].coefficients[0] = fs::BoundInteger(-1); rows[1].bound = fs::BoundInteger(int64_t(1) << 32);
    rows[2].coefficients[1] = fs::BoundInteger(-1); rows[2].bound = fs::BoundInteger(0);
    rows[3].coefficients[1] = fs::BoundInteger(1); rows[3].bound = fs::BoundInteger((int64_t(1) << 62) - 1);
    auto boundedDomain = fs::IntegerSystem::create(2, rows);
    if (failed(boundedDomain)) { return false; }
    auto fused = e.integerRelation(*predicate, {x, i}, *boundedDomain, error);
    if (failed(fused)) { return false; }
    for (int64_t length : {int64_t(-5), int64_t(-1), int64_t(0), int64_t(4), INT64_MIN, INT64_MAX}) {
        for (int64_t ordinal : {int64_t(0), int64_t(1), int64_t(3), int64_t(1) << 61, INT64_MAX}) {
            for (bool bounded : {false, true}) {
            if (bounded && (length < -(int64_t(1) << 32) || length > (int64_t(1) << 32) ||
                           ordinal < 0 || ordinal >= (int64_t(1) << 62))) { continue; }
            bool actual = false;
            for (const auto& piece : (bounded ? *fused : *relation)) {
                bool present = true;
                for (const auto& row : piece.constraints()) {
                    auto value = row.coefficients[0] * fs::BoundInteger(length) +
                        row.coefficients[1] * fs::BoundInteger(ordinal);
                    present &= value <= row.bound;
                }
                for (const auto& row : piece.congruences()) {
                    auto value = row.coefficients[0] * fs::BoundInteger(length) +
                        row.coefficients[1] * fs::BoundInteger(ordinal);
                    present &= mod(value, row.modulus) == mod(row.residue, row.modulus);
                }
                actual |= present;
            }
            const auto tail = APInt(64, length) - APInt(64, ordinal) * APInt(64, 4);
            if (actual != tail.sgt(APInt(64, 3))) {
                llvm::errs() << "modular symbolic mismatch " << length << ", " << ordinal << "\n";
                return false;
            }
            ++count;
            }
        }
    }
    return true;
}
bool inputRanges(MLIRContext* context)
{
    auto module = parseSourceString<ModuleOp>(R"mlir(module attributes {
      dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
      func.func @f(%integer: i32, %index: index) { return }
    })mlir", context);
    if (!module) { return false; }
    auto function = module->lookupSymbol<func::FuncOp>("f");
    fs::RegionExpressions e;
    fs::ModularScalar scalar(e, function);
    const auto signedRange = std::make_pair(int64_t(INT32_MIN), int64_t(INT32_MAX));
    if (scalar.inputRange(function.getArgument(0), signedRange) !=
            std::make_pair(int64_t(0), int64_t(UINT32_MAX)) ||
        scalar.inputRange(function.getArgument(1), signedRange) != signedRange) { return false; }
    if (scalar.inputRange(function.getArgument(0), {-5, -1}) !=
            std::make_pair(int64_t(UINT32_MAX) - 4, int64_t(UINT32_MAX)) ||
        scalar.inputRange(function.getArgument(0), {-5, 5}) !=
            std::make_pair(int64_t(0), int64_t(UINT32_MAX)) ||
        scalar.inputRange(function.getArgument(0), {0, 5}) !=
            std::make_pair(int64_t(0), int64_t(5))) { return false; }
    auto variable = e.input(function.getArgument(1));
    std::vector<fs::IntegerConstraint> rows(2);
    rows[0].coefficients = {fs::BoundInteger(1)}; rows[0].bound = fs::BoundInteger(INT32_MAX);
    rows[1].coefficients = {fs::BoundInteger(-1)}; rows[1].bound = -fs::BoundInteger(INT32_MIN);
    auto domain = fs::IntegerSystem::create(1, rows);
    std::string error;
    if (failed(domain)) { return false; }
    auto relation = e.integerRelation(e.slt(variable, e.constant(0)), {variable}, *domain, error);
    if (failed(relation) || relation->empty()) { return false; }
    bool containsNegative = false;
    for (const auto& piece : *relation) {
        containsNegative |= llvm::all_of(piece.constraints(), [](const auto& row) {
            return -row.coefficients[0] <= row.bound;
        });
    }
    return containsNegative;
}
bool fixedVisits(MLIRContext* context)
{
    auto module = parseSourceString<ModuleOp>(R"mlir(module {
      func.func @f() {
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %three = arith.constant 3 : index
        %max = arith.constant 9223372036854775807 : index
        scf.for %i = %zero to %three step %one {
          %wrapped = arith.muli %i, %max : index
        }
        return
      }
    })mlir", context);
    if (!module) { return false; }
    auto function = module->lookupSymbol<func::FuncOp>("f");
    scf::ForOp loop;
    Value wrapped;
    function.walk([&](scf::ForOp op) { loop = op; });
    loop.walk([&](arith::MulIOp op) { wrapped = op.getResult(); });
    fs::PhaseIndex index;
    if (failed(index.build(function, ArrayRef<const pto::CompoundInstanceElement*>{}))) { return false; }
    fs::ArithmeticProgram program;
    program.context.function = function; program.context.root = loop;
    fs::ArithmeticLimits limits{8, 8, 1, 4096};
    fs::detail::ProgramBuilder builder{program, limits, context, index, DenseMap<Value, unsigned>(), {}};
    fs::ArithmeticSite first, second;
    first.fixedCoordinates.push_back({loop, 0}); second.fixedCoordinates.push_back({loop, 2});
    auto a = builder.machineValue(wrapped, first, 0), b = builder.machineValue(wrapped, second, 0);
    if (!a || !b || a == b || builder.machineValue(wrapped, first, 0) != a) { return false; }
    fs::PrimitiveRelation positive, negative;
    if (!builder.emitMachine(positive, {a}) || !builder.emitMachine(negative, {b}) ||
        positive.pieces.empty() || !negative.pieces.empty()) {
        llvm::errs() << "fixed-visit machine recipe mismatch\n"; return false;
    }
    fs::PrimitiveRelation reused;
    if (!builder.emitMachine(reused, {a}) || reused.pieces.size() != positive.pieces.size() ||
        reused.pieces.front().machineSystem != positive.pieces.front().machineSystem ||
        program.machinePrimitiveCacheHits != 1 || program.machinePrimitiveConversions != 2) { return false; }
    return true;
}
bool coupledEntryDefinitions(MLIRContext* context)
{
    auto module = parseSourceString<ModuleOp>(R"mlir(module {
      func.func @f(%start: i8, %end: i8, %keep: index) {
        %s = arith.index_cast %start : i8 to index
        %e = arith.index_cast %end : i8 to index
        %length = arith.subi %e, %s : index
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %three = arith.constant 3 : index
        %four = arith.constant 4 : index
        %sum = arith.addi %length, %three : index
        %count = arith.divui %sum, %four : index
        scf.for %i = %zero to %count step %one {
          %scaled = arith.muli %i, %four : index
          %tail = arith.subi %length, %scaled : index
        }
        return
      }
    })mlir", context);
    if (!module) { return false; }
    auto function = module->lookupSymbol<func::FuncOp>("f");
    scf::ForOp loop;
    Value length, count, tail, start, end;
    function.walk([&](scf::ForOp op) { loop = op; });
    function.walk([&](arith::IndexCastOp op) {
        (op.getIn() == function.getArgument(0) ? start : end) = op.getOut();
    });
    function.walk([&](arith::SubIOp op) {
        (op->getParentOp() == function.getOperation() ? length : tail) = op.getResult();
    });
    function.walk([&](arith::DivUIOp op) { count = op.getResult(); });
    fs::PhaseIndex index;
    if (failed(index.build(function, ArrayRef<const pto::CompoundInstanceElement*>{}))) { return false; }
    fs::ArithmeticProgram program;
    program.context.function = function; program.context.root = loop;
    fs::ArithmeticLimits limits{8, 8, 1, 4096};
    fs::detail::ProgramBuilder builder{program, limits, context, index, DenseMap<Value, unsigned>(), {}};
    fs::ArithmeticSite entry;
    if (!builder.prepareValue(function.getArgument(2), entry) ||
        !builder.prepareValue(count, entry) || program.parameters.size() != 3 ||
        program.parameters.front() != function.getArgument(2) ||
        builder.parameterIds.count(length) || builder.parameterIds.count(count) ||
        !builder.parameterIds.count(start) || !builder.parameterIds.count(end)) { return false; }
    const auto parameterIds = builder.parameterIds;
    // Opposite sides of the 64-bit wrap boundary must share exactly the same
    // entry leaves, including after speculative affine extraction fails.
    for (int64_t visit : {int64_t(0), int64_t(1) << 61}) {
        fs::ArithmeticSite site;
        site.fixedCoordinates.push_back({loop, visit});
        if (!builder.prepareValue(tail, site) || !(builder.parameterIds == parameterIds)) { return false; }
        auto recipe = builder.machineValue(tail, site, 0);
        auto relation = builder.relation(fs::PrimitiveKind::Occurrences, 0);
        if (!recipe || !builder.emitMachine(relation, {recipe})) { return false; }
        SmallVector<fs::BoundInteger> point(program.parameters.size(), fs::BoundInteger(0));
        point[builder.parameterIds.lookup(end)] = fs::BoundInteger(-5);
        const bool contains = llvm::any_of(relation.pieces, [&](const auto& piece) {
            if (!piece.machineSystem) { return false; }
            auto dot = [&](const auto& coefficients) {
                fs::BoundInteger value(0);
                for (auto [id, coefficient] : llvm::enumerate(coefficients)) { value += coefficient * point[id]; }
                return value;
            };
            return llvm::all_of(piece.machineSystem->constraints(), [&](const auto& row) {
                return dot(row.coefficients) <= row.bound;
            }) && llvm::all_of(piece.machineSystem->congruences(), [&](const auto& row) {
                return mod(dot(row.coefficients) - row.residue, row.modulus) == fs::BoundInteger(0);
            });
        });
        if (contains != (visit != 0)) { return false; }
    }
    return builder.prepareValue(count, entry) && builder.parameterIds == parameterIds;
}
bool wideEntryFallback(MLIRContext* context, bool useArgument = false)
{
    auto module = parseSourceString<ModuleOp>(R"mlir(
    !tile = !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=8,
      blayout=row_major, slayout=none_box, fractal=512, pad=0>
    module {
      func.func @f(%a: index, %b: index, %c: index, %d: index, %e: index,
                   %f: index, %g: index, %h: index, %j: index) {
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %base = arith.constant 0 : i64
        %tile = pto.alloc_tile addr = %base : !tile
        %s1 = arith.addi %a, %b : index
        %s2 = arith.addi %s1, %c : index
        %s3 = arith.addi %s2, %d : index
        %s4 = arith.addi %s3, %e : index
        %s5 = arith.addi %s4, %f : index
        %s6 = arith.addi %s5, %g : index
        %s7 = arith.addi %s6, %h : index
        %count = arith.addi %s7, %j : index
        scf.for %i = %zero to %count step %one {
          %value = pto.tgetval ins(%tile, %zero : !tile, index) outs : f32
        }
        return
      }
    })mlir", context);
    if (!module) { return false; }
    auto function = module->lookupSymbol<func::FuncOp>("f");
    scf::ForOp loop;
    function.walk([&](scf::ForOp op) { loop = op; });
    if (useArgument) {
        OpBuilder builder(loop.getBody(), loop.getBody()->begin());
        builder.create<arith::CmpIOp>(loop.getLoc(), arith::CmpIPredicate::slt,
            loop.getInductionVar(), function.getArgument(0));
        // Make the argument relevant to actual control, rather than dead metadata.
        auto condition = cast<arith::CmpIOp>(loop.getBody()->front());
        auto branch = builder.create<scf::IfOp>(loop.getLoc(), condition.getResult(), false);
        auto* payload = condition->getNextNode()->getNextNode();
        payload->moveBefore(branch.thenBlock()->getTerminator());
    }
    pto::SyncInput input;
    fs::PhaseIndex index;
    if (failed(input.build(function, pto::SyncInstructionView::PipeEnvelopes)) ||
        failed(index.build(function, input))) { return false; }
    auto program = fs::recognizeArithmeticProgram({function, loop}, index, input, input.accesses(), {8, 8, 1, 4096});
    const bool ready = program.extraction.state == fs::RecognitionState::Applicable &&
        program.recognition.state == fs::RecognitionState::Applicable &&
        program.parameters.size() == (useArgument ? 2U : 1U) &&
        llvm::is_contained(program.parameters, loop.getUpperBound()) &&
        (!useArgument || llvm::is_contained(program.parameters, function.getArgument(0)));
    if (!ready) { return false; }
    // Check the selected occurrence domain independently of scalar extraction.
    // Negative counts include wrapped entry sums; the actual SSA count is signed.
    for (int64_t count : {INT64_MIN, int64_t(-5), int64_t(0), int64_t(1), int64_t(4), INT64_MAX}) {
        for (int64_t visit : {int64_t(-1), int64_t(0), int64_t(1), int64_t(3)}) {
            bool present = false;
            for (const auto& piece : program.recognition.pieces) {
                const auto& relation = program.primitives.relations[piece.relation];
                if (relation.kind != fs::PrimitiveKind::Occurrences || piece.empty) { continue; }
                std::vector<fs::BoundInteger> point{fs::BoundInteger(visit)};
                for (auto parameter : program.parameters) {
                    point.emplace_back(parameter == loop.getUpperBound() ? count : 2);
                }
                present |= llvm::all_of(piece.rows, [&](const auto& row) {
                    if (row.coefficients.size() != point.size()) { return false; }
                    fs::BoundInteger value(row.constant);
                    for (auto [coefficient, coordinate] : llvm::zip(row.coefficients, point)) {
                        value += fs::BoundInteger(coefficient) * coordinate;
                    }
                    return row.equality ? value == fs::BoundInteger(0) : value >= fs::BoundInteger(0);
                });
            }
            const bool expected = visit >= 0 && visit < count && (!useArgument || visit < 2);
            if (present != expected) { return false; }
        }
    }
    return true;
}
} // namespace
bool runMachineCertificateChecks();
bool runModularScalarChecks(MLIRContext* context)
{
    uint64_t count = 0;
    if (!runMachineCertificateChecks()) { return false; }
    for (unsigned width : {4, 5, 8, 32, 64}) {
        if (!runWidth(context, width, count)) { return false; }
    }
    if (!inputRanges(context)) { return false; }
    if (!casts(context, count)) { return false; }
    if (!fixedVisits(context)) { return false; }
    if (!coupledEntryDefinitions(context)) {
        llvm::errs() << "coupled entry definition regression\n"; return false;
    }
    if (!wideEntryFallback(context) || !wideEntryFallback(context, true)) {
        llvm::errs() << "wide entry fallback regression\n"; return false;
    }
    if (!symbolic(context, count)) { return false; }
    llvm::outs() << "modular scalar: " << count << " APInt comparisons passed\n";
    return true;
}
