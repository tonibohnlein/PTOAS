// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// General integer reuse: quantified pairs, residue transitions and union coverage.
#include "PTO/Transforms/FrontierSynch/GeneralArithmeticAllocation.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/raw_ostream.h"
namespace fs = mlir::pto::frontiersynch;
namespace {
fs::IntegerConstraint row(std::initializer_list<int64_t> values, int64_t bound)
{
    fs::IntegerConstraint result;
    for (auto value : values) { result.coefficients.push_back(fs::BoundInteger(value)); }
    result.bound = fs::BoundInteger(bound);
    return result;
}
fs::IntegerCongruence parity(unsigned coordinate, unsigned value)
{
    fs::IntegerCongruence result;
    result.coefficients.assign(3, fs::BoundInteger(0));
    result.coefficients[coordinate] = fs::BoundInteger(1);
    result.modulus = fs::BoundInteger(2); result.residue = fs::BoundInteger(value);
    return result;
}
fs::ArithmeticRelationKey demand(uint64_t residue)
{
    return {{0, fs::ArithmeticEvent::Completion, {residue}},
            {1, fs::ArithmeticEvent::Start, {residue}}, {0}};
}
fs::ArithmeticRelationKey reuse(uint64_t earlier, uint64_t later)
{
    return {{1, fs::ArithmeticEvent::Completion, {earlier}},
            {0, fs::ArithmeticEvent::Start, {later}}, {0}};
}
bool parameterContexts(mlir::MLIRContext& context)
{
    fs::GeneralArithmeticDemandAnalysis analysis;
    analysis.exactMinimum = true; analysis.parameterCount = 1;
    for (int64_t value : {0, 1}) {
        auto point = fs::IntegerSystem::create(3, {row({1, 0, 0}, value), row({-1, 0, 0}, -value),
            row({0, 1, 0}, value), row({0, -1, 0}, -value),
            row({0, 0, 1}, value), row({0, 0, -1}, -value)});
        if (mlir::failed(point)) { return false; }
        analysis.minimumDemands[demand(0)].push_back(*point);
    }
    // Each parameter context has just one occurrence. Pairing instances from
    // different contexts would incorrectly demand a reuse edge.
    return bool(fs::generalArithmeticAllocationCertificate(analysis, {0, 1}, {{{0, 1}, 0}}, 0, &context));
}
bool singleton(mlir::MLIRContext& context)
{
    fs::GeneralArithmeticDemandAnalysis analysis;
    analysis.exactMinimum = true;
    fs::ArithmeticRelationKey edge{{0, fs::ArithmeticEvent::Completion, {}},
                                  {1, fs::ArithmeticEvent::Start, {}}, {}};
    analysis.minimumDemands[edge] = {fs::IntegerSystem{}};
    // With no source coordinates there is only one dynamic handoff. No
    // completion-to-next-start witness is needed for its dedicated ID.
    return bool(fs::generalArithmeticAllocationCertificate(analysis, {0, 1}, {{{0, 1}, 0}}, 0, &context));
}
bool nestedCoordinates(mlir::MLIRContext& context)
{
    fs::GeneralArithmeticDemandAnalysis analysis;
    analysis.exactMinimum = true;
    fs::ArithmeticRelationKey edge{{0, fs::ArithmeticEvent::Completion, {0, 0}},
                                  {1, fs::ArithmeticEvent::Start, {0, 0}}, {}};
    fs::ArithmeticRelationKey order{{1, fs::ArithmeticEvent::Completion, {0, 0}},
                                   {0, fs::ArithmeticEvent::Start, {0, 0}}, {}};
    auto domain = fs::IntegerSystem::create(4,
        {row({1, 0, -1, 0}, 0), row({-1, 0, 1, 0}, 0), row({0, 1, 0, -1}, 0),
         row({0, -1, 0, 1}, 0), row({-1, 0, 0, 0}, 0), row({1, 0, 0, 0}, 1),
         row({0, -1, 0, 0}, 0), row({0, 1, 0, 0}, 1)});
    auto outer = fs::IntegerSystem::create(4, {row({1, 0, -1, 0}, -1)});
    auto inner = fs::IntegerSystem::create(4,
        {row({1, 0, -1, 0}, 0), row({-1, 0, 1, 0}, 0), row({0, 1, 0, -1}, -1)});
    if (mlir::failed(domain) || mlir::failed(outer) || mlir::failed(inner)) { return false; }
    analysis.minimumDemands[edge] = {*domain};
    analysis.requiredOrder[order] = {*outer, *inner};
    auto certify = [&]() {
        return fs::generalArithmeticAllocationCertificate(analysis, {0, 1}, {{{0, 1}, 0}}, 0, &context);
    };
    if (!certify()) { return false; }
    analysis.requiredOrder[order] = {*inner};
    return !certify();
}
bool check(mlir::MLIRContext& context)
{
    fs::GeneralArithmeticDemandAnalysis analysis;
    analysis.period = 2; analysis.parameterCount = 1; analysis.exactMinimum = true;
    // Quotient coordinates (x,y,p): y=2*x, 0<=x<=p<=10^9.
    // This is deliberately outside the difference-bound subclass.
    auto edge = fs::IntegerSystem::create(3, {row({-2, 1, 0}, 0), row({2, -1, 0}, 0),
        row({-1, 0, 0}, 0), row({1, 0, -1}, 0), row({0, 0, 1}, 1000000000)});
    if (mlir::failed(edge)) { return false; }
    for (uint64_t a = 0; a < 2; ++a) {
        analysis.minimumDemands[demand(a)] = {*edge};
        for (uint64_t b = 0; b < 2; ++b) {
            // Query coordinates (earlier y,later x,p); y is even after
            // projecting away the earlier x. Neither parity piece alone
            // covers all later x values; the complete union is necessary.
            for (unsigned residue = 0; residue < 2; ++residue) {
                auto ordered = fs::IntegerSystem::create(3,
                    {row({1, -2, 0}, a < b ? 0 : -2), row({-1, 0, 0}, 0),
                     row({0, 1, -1}, 0), row({0, 0, 1}, 1000000000)},
                    {parity(0, 0), parity(1, residue)});
                if (mlir::failed(ordered)) { return false; }
                analysis.requiredOrder[reuse(a, b)].push_back(*ordered);
            }
        }
    }
    const std::map<std::pair<std::size_t, std::size_t>, int64_t> records{{{0, 1}, 7}};
    auto certify = [&](const auto& value) {
        return fs::generalArithmeticAllocationCertificate(value, {0, 1}, records, 3, &context);
    };
    auto certificate = certify(analysis);
    if (!certificate || certificate.getAs<mlir::StringAttr>("strategy").getValue() != "dedicated-families") {
        return false;
    }
    auto directions = certificate.getAs<mlir::ArrayAttr>("directions");
    if (!directions || directions.size() != 1) { return false; }
    auto direction = mlir::dyn_cast<mlir::DictionaryAttr>(directions[0]);
    if (!direction || direction.getAs<mlir::IntegerAttr>("budget").getInt() != 1) {
        return false;
    }
    auto encodedRecords = direction.getAs<mlir::DenseI64ArrayAttr>("records");
    if (!encodedRecords || encodedRecords.size() != 1 || encodedRecords[0] != 7) { return false; }
    auto missingResidue = analysis;
    missingResidue.requiredOrder.erase(reuse(0, 1));
    if (certify(missingResidue)) { return false; }
    auto missingParity = analysis;
    missingParity.requiredOrder[reuse(1, 0)].pop_back();
    if (certify(missingParity)) { return false; }
    auto shortWitness = analysis;
    auto prefix = fs::IntegerSystem::create(3, {row({0, 1, 0}, 7)});
    if (mlir::failed(prefix)) { return false; }
    for (auto& [key, pieces] : shortWitness.requiredOrder) {
        for (auto& piece : pieces) {
            auto restricted = piece.intersect(*prefix);
            if (mlir::failed(restricted)) { return false; }
            piece = std::move(*restricted);
        }
    }
    if (certify(shortWitness)) { return false; }
    auto malformed = analysis;
    malformed.minimumDemands[demand(0)] = {fs::IntegerSystem{}};
    return !certify(malformed);
}
} // namespace
int runGeneralArithmeticAllocationChecks()
{
    mlir::MLIRContext context;
    if (!check(context) || !parameterContexts(context) || !singleton(context) || !nestedCoordinates(context)) {
        llvm::errs() << "general arithmetic allocation certificate check failed\n";
        return 1;
    }
    llvm::outs() << "general arithmetic allocation: exact unions, residues and all source pairs passed\n";
    return 0;
}
