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
        return fs::encodeGeneralArithmeticAllocationCertificate(
            fs::buildArithmeticHandoffAllocation(analysis, {0, 1}, 1), {0, 1}, {{{0, 1}, 0}}, 0, &context);
    };
    if (!certify()) { return false; }
    analysis.requiredOrder[order] = {*inner};
    return !certify();
}
bool activeSuccessors(uint64_t period)
{
    fs::GeneralArithmeticDemandAnalysis analysis;
    analysis.exactMinimum = true; analysis.period = period;
    fs::ArithmeticRelationKey edge{{0, fs::ArithmeticEvent::Completion, {0}},
                                  {1, fs::ArithmeticEvent::Start, {0}}, {}};
    fs::ArithmeticRelationKey ordered{{1, fs::ArithmeticEvent::Completion, {0}},
                                     {0, fs::ArithmeticEvent::Start, {0}}, {}};
    // Active sources 0,2,6,8,10: ordinal modulo two is always zero. Two
    // executed-handoff lanes are sufficient; one is not. Numeric gaps and
    // the missing source 4 must not be interpreted as executed publications.
    for (int64_t coordinate : {0, 2, 6, 8, 10}) {
        auto quotient = coordinate / static_cast<int64_t>(period);
        auto point = fs::IntegerSystem::create(2,
            {row({1, 0}, quotient), row({-1, 0}, -quotient),
             row({0, 1}, quotient), row({0, -1}, -quotient)});
        if (mlir::failed(point)) { return false; }
        edge.source.residues = {coordinate % period}; edge.target.residues = edge.source.residues;
        analysis.minimumDemands[edge].push_back(*point);
    }
    for (uint64_t from = 0; from < period; ++from) {
        for (uint64_t to = 0; to < period; ++to) {
            auto bound = floorDiv(fs::BoundInteger(to) - fs::BoundInteger(from) - 4, fs::BoundInteger(period));
            auto constraint = row({1, -1}, 0); constraint.bound = bound;
            auto reuse = fs::IntegerSystem::create(2, {constraint});
            if (mlir::failed(reuse)) { return false; }
            ordered.source.residues = {from}; ordered.target.residues = {to};
            analysis.requiredOrder[ordered] = {*reuse};
        }
    }
    auto one = fs::buildArithmeticHandoffAllocation(analysis, {0, 1}, 1);
    auto two = fs::buildArithmeticHandoffAllocation(analysis, {0, 1}, 2);
    if (one.error.empty() || !two.error.empty() || two.families.size() != 1 || two.families[0].width != 2) {
        return false;
    }
    auto first = fs::evaluateGeneralArithmeticSelector(two.families[0].firstSource, period, {}, {});
    auto last = fs::evaluateGeneralArithmeticSelector(two.families[0].lastTarget, period, {}, {});
    if (mlir::failed(first) || mlir::failed(last) || !*first || !*last ||
        (**first).coordinates != std::vector<fs::BoundInteger>{fs::BoundInteger(0)} ||
        (**last).coordinates != std::vector<fs::BoundInteger>{fs::BoundInteger(10)}) { return false; }
    // Even a palette wider than the finite domain cannot justify two counters
    // if their source and consumer execution sequences disagree.
    analysis.minimumDemands.clear(); analysis.requiredOrder.clear();
    for (int64_t coordinate : {0, 2}) {
        auto source = coordinate / static_cast<int64_t>(period);
        auto target = (2 - coordinate) / static_cast<int64_t>(period);
        auto point = fs::IntegerSystem::create(2,
            {row({1, 0}, source), row({-1, 0}, -source), row({0, 1}, target), row({0, -1}, -target)});
        if (mlir::failed(point)) { return false; }
        edge.source.residues = {coordinate % period}; edge.target.residues = {(2 - coordinate) % period};
        analysis.minimumDemands[edge].push_back(*point);
    }
    return !fs::buildArithmeticHandoffAllocation(analysis, {0, 1}, 6).error.empty();
}
bool differenceBoundSuccessors(mlir::MLIRContext& context)
{
    fs::ArithmeticDemandAnalysis analysis; analysis.exactMinimum = true;
    fs::ArithmeticRelationKey edge{{0, fs::ArithmeticEvent::Completion, {0}},
                                  {1, fs::ArithmeticEvent::Start, {0}}, {}};
    fs::ArithmeticRelationKey order{{1, fs::ArithmeticEvent::Completion, {0}},
                                   {0, fs::ArithmeticEvent::Start, {0}}, {}};
    for (int64_t value : {0, 2, 6, 8, 10}) {
        auto point = fs::DifferenceBoundSystem::create(2,
            {{1, 0, fs::BoundInteger(value)}, {0, 1, fs::BoundInteger(-value)},
             {2, 0, fs::BoundInteger(value)}, {0, 2, fs::BoundInteger(-value)}});
        if (mlir::failed(point)) { return false; }
        analysis.minimumDemands[edge].push_back(*point);
    }
    auto reuse = fs::DifferenceBoundSystem::create(2, {{1, 2, fs::BoundInteger(-4)}});
    if (mlir::failed(reuse)) { return false; }
    analysis.requiredOrder[order] = {*reuse};
    auto proof = fs::buildArithmeticHandoffAllocation(analysis, {0, 1});
    if (!proof.error.empty() || proof.families.size() != 1 || proof.families.front().width != 2) { return false; }
    auto encoded = fs::encodeGeneralArithmeticAllocationCertificate(proof, {0, 1}, {{{0, 1}, 0}}, 0, &context);
    return encoded && encoded.getAs<mlir::StringAttr>("strategy").getValue() == "executed-family-counters";
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
        return fs::encodeGeneralArithmeticAllocationCertificate(
            fs::buildArithmeticHandoffAllocation(value, {0, 1}, 1), {0, 1}, records, 3, &context);
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
    if (!check(context) || !parameterContexts(context) || !singleton(context) || !nestedCoordinates(context) ||
        !activeSuccessors(1) || !activeSuccessors(3) || !differenceBoundSuccessors(context)) {
        llvm::errs() << "general arithmetic allocation certificate check failed\n";
        return 1;
    }
    llvm::outs() << "general arithmetic allocation: exact unions, residues and all source pairs passed\n";
    return 0;
}
