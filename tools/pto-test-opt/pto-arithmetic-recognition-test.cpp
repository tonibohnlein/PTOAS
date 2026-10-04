// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Exercise supplied-primitive recognition independently of IR effect recovery.
#include "PTO/Transforms/FrontierSynch/ArithmeticRecognition.h"
#include "PTO/Transforms/FrontierSynch/ArithmeticProgram.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
fs::ArithmeticPrimitives bundle(AffineExpr expression, bool equality = false)
{
    fs::ArithmeticPrimitives input;
    input.pipeCount = 3;
    input.parameters.push_back("N");
    for (unsigned role = 0; role < 7; ++role) {
        fs::PrimitiveRelation relation;
        relation.kind = static_cast<fs::PrimitiveKind>(role);
        relation.dimensions = 3;
        relation.coordinates = {{"x", fs::CoordinateKind::Occurrence}, {"y", fs::CoordinateKind::Occurrence},
                                {"z", fs::CoordinateKind::Auxiliary}, {"N", fs::CoordinateKind::Parameter}};
        // Empty unions deliberately exercise present-but-empty roles.
        if (relation.kind == fs::PrimitiveKind::Reads) {
            relation.pieces.push_back({IntegerSet::get(3, 1, {expression}, {equality}), {0, 0, 0, 0}});
        }
        input.relations.push_back(std::move(relation));
    }
    return input;
}
void dump(StringRef name, const fs::ArithmeticPrimitives& input, fs::ArithmeticLimits limits = {3, 4, 1, 7})
{
    const auto result = fs::recognizeArithmetic(input, limits);
    llvm::outs() << name << ": " << fs::recognitionName(result.state)
                 << " class=" << fs::recognitionName(result.arithmeticClass)
                 << " dimensions=" << result.observedDimensions
                 << " coefficient=" << result.observedCoefficient << "\n";
    for (const auto& diagnostic : result.diagnostics) {
        llvm::outs() << "  issue " << fs::recognitionName(diagnostic.issue) << "\n";
    }
    for (const auto& piece : result.pieces) {
        llvm::outs() << "  piece empty=" << piece.empty << " rows=" << piece.rows.size() << "\n";
        for (const auto& row : piece.rows) {
            llvm::outs() << "  row ";
            llvm::interleaveComma(row.coefficients, llvm::outs());
            llvm::outs() << " constant=" << row.constant << " equality=" << row.equality << "\n";
        }
    }
}
// Check row normalization against integer evaluation, independently of its
// gcd/floor implementation, for both equalities and inequalities.
bool checkIntegerRows(MLIRContext& context)
{
    auto x = getAffineDimExpr(0, &context), y = getAffineDimExpr(1, &context);
    unsigned checks = 0;
    for (int a = -2; a <= 2; ++a) {
        for (int b = -2; b <= 2; ++b) {
            for (int c = -3; c <= 3; ++c) {
                for (bool equality : {false, true}) {
                    auto result = fs::recognizeArithmetic(bundle(a*x + b*y + c, equality), {3, 4, 1, 7});
                    const bool valid = result.state == fs::RecognitionState::Applicable && result.pieces.size() == 1;
                    if (!valid) {
                        return false;
                    }
                    const auto& piece = result.pieces.front();
                    for (int xv = -3; xv <= 3; ++xv) {
                        for (int yv = -3; yv <= 3; ++yv) {
                            int original = a*xv + b*yv + c;
                            bool actual = !piece.empty;
                            for (const auto& row : piece.rows) {
                                auto value = row.coefficients[0]*xv + row.coefficients[1]*yv + row.constant;
                                actual &= row.equality ? value == 0 : value >= 0;
                            }
                            if (actual != (equality ? original == 0 : original >= 0)) {
                                return false;
                            }
                            ++checks;
                        }
                    }
                }
            }
        }
    }
    llvm::outs() << "integer-row-equivalence: " << checks << " passed\n";
    return true;
}

}
int main()
{
    MLIRContext context;
    context.disableMultithreading();
    if (!checkIntegerRows(context)) {
        return 1;
    }
    auto x = getAffineDimExpr(0, &context), y = getAffineDimExpr(1, &context), z = getAffineDimExpr(2, &context);
    auto n = getAffineSymbolExpr(0, &context);
    dump("difference", bundle(n - x));
    dump("octagon", bundle(x + y - 5));
    dump("three-coordinates", bundle(n - x - y));
    dump("bounded", bundle(x - 7*y - 3*z - 4));
    dump("positive-round", bundle(3 - 2*x));
    dump("negative-round", bundle(-3 - 2*x));
    dump("impossible-equality", bundle(2*x - 1, true));
    dump("cancellation", bundle(7*x - 7*x + y));
    dump("constant-false", bundle(getAffineConstantExpr(-1, &context)));
    dump("constant-true", bundle(getAffineConstantExpr(0, &context)));
    dump("too-large", bundle(x - 8*y));
    dump("division", bundle(x.floorDiv(2)));
    dump("modulo", bundle(x % 2));
    dump("nonlinear", bundle(x*n));
    auto oversized = bundle(x);
    dump("dimension-limit", oversized, {3, 3, 1, 7});
    oversized.pipeCount = 4;
    dump("pipe-limit", oversized);
    auto missing = bundle(x);
    missing.relations.pop_back();
    dump("missing-role", missing);
    dump("missing-primitives", {});
    auto residues = bundle(x - y);
    residues.period = 2;
    dump("period-mismatch", residues);
    dump("residue-valid", residues, {3, 4, 2, 7});
    residues.relations[4].pieces[0].residues[0] = 2;
    dump("residue-invalid", residues, {3, 4, 2, 7});
    auto parameters = bundle(x);
    parameters.parameters.push_back("hidden");
    dump("missing-parameter", parameters);
    auto overflow = bundle(x);
    // Build the expression without constant folding away intermediate products.
    auto scale = getAffineConstantExpr(INT64_MAX, &context);
    overflow.relations[4].pieces[0].system = IntegerSet::get(3, 1, {(x*scale + y)*2}, {false});
    dump("coefficient-overflow", overflow);
    dump("minimum-coefficient", bundle(x * INT64_MIN + INT64_MIN));
    auto duplicate = bundle(x);
    duplicate.relations[4].coordinates[1].name = "x";
    dump("duplicate-coordinate", duplicate);
    auto incompleteResidues = bundle(x);
    incompleteResidues.relations[4].pieces[0].residues.pop_back();
    dump("missing-residue", incompleteResidues);
    auto unionPieces = bundle(x - y);
    unionPieces.relations[4].pieces.push_back({IntegerSet::get(3, 1, {x + y - 5}, {false}), {0, 0, 0, 0}});
    dump("union-class", unionPieces);
    context.getOrLoadDialect<func::FuncDialect>();
    OpBuilder builder(&context);
    OwningOpRef<func::FuncOp> function(
        func::FuncOp::create(builder.getUnknownLoc(), "empty", builder.getFunctionType({}, {})));
    builder.setInsertionPointToStart(function->addEntryBlock());
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    pto::SyncInput shared;
    fs::PhaseIndex index;
    if (failed(shared.build(*function)) || failed(index.build(*function, shared))) {
        return 1;
    }
    const auto& effects = shared.accesses();
    for (auto limits : {fs::ArithmeticLimits{8, 8, INT64_MAX, 8}, fs::ArithmeticLimits{8, 100, 2, 8}}) {
        auto program = fs::recognizeArithmeticProgram(*function, index, shared, effects, limits);
        llvm::outs() << "producer-limit: " << fs::recognitionName(program.extraction.state)
                     << " exports=" << program.primitives.relations.size() << "\n";
    }
    auto metadata = bundle(x);
    metadata.relations[4].sourceDimensions = 4;
    dump("invalid-tags", metadata);
    return 0;
}
