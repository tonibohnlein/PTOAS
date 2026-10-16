// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// General integer reuse: quantified pairs, residue transitions and union coverage.
#include "PTO/Transforms/FrontierSynch/Arithmetic/GeneralArithmeticAllocation.h"
#include "PTO/Transforms/FrontierSynch/Shared/AllocationProfile.h"
#include "../../lib/PTO/Transforms/FrontierSynch/AnalysisSessionInternal.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Arithmetic/ArithmeticHandoffProof.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "UnaryHandoffExtremaChecks.h"
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
    const auto coordinate = fs::buildArithmeticCoordinateHandoffAllocation(analysis, {0, 1}, 1);
    return coordinate.error.empty() && coordinate.families.size() == 1 &&
        coordinate.families.front().lifetime.width == 1 &&
        bool(fs::generalArithmeticAllocationCertificate(analysis, {0, 1}, {{{0, 1}, 0}}, 0, &context));
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
    // The same insufficient one-ID obligation is observed with profiling on
    // and off. A malformed required relation exercises construction failure,
    // which must remain distinct from a constructed but uncovered relation.
    auto unavailable = analysis;
    unavailable.requiredOrder.begin()->second = {fs::IntegerSystem{}};
    for (bool malformed : {false, true}) {
        const auto& input = malformed ? unavailable : analysis;
        const auto baseline = fs::buildArithmeticHandoffAllocation(input, {0, 1}, 1);
        const std::string outcome = malformed ? "construction-unavailable" : "reuse-uncovered-within-bound";
        if (baseline.error.find("stage one-id-difference, outcome " + outcome) == std::string::npos) {
            return false;
        }
        for (bool enabled : {false, true}) {
            auto ledger = std::make_shared<fs::AnalysisLedger>();
            ledger->profiling = enabled;
            fs::ArithmeticHandoffAllocation observed;
            {
                fs::AnalysisProfileScope request(ledger, "allocation", 0, "single-id-refusal-profile");
                fs::AllocationProfile profile(ledger);
                observed = fs::buildArithmeticHandoffAllocation(input, {0, 1}, 1, &profile);
            }
            if (observed.error != baseline.error || !observed.families.empty() ||
                observed.cost.pieceJoins != baseline.cost.pieceJoins ||
                observed.cost.projections != baseline.cost.projections ||
                observed.cost.relationCompositions != baseline.cost.relationCompositions ||
                observed.cost.differences != baseline.cost.differences) { return false; }
            unsigned refusals = 0;
            for (const auto& stage : ledger->stages) {
                if (stage.method == "arithmetic-handoff-refusal:one-id-difference:" + outcome) { ++refusals; }
            }
            if (enabled ? refusals != 1 : !ledger->stages.empty()) { return false; }
        }
    }
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
bool coordinateMapping(uint64_t period, bool negative, bool missingReuse)
{
    fs::GeneralArithmeticDemandAnalysis analysis;
    analysis.exactMinimum = true; analysis.period = period;
    const std::vector<int64_t> visits = negative ? std::vector<int64_t>{-2, 0, 2, 3, 6, 8, 10} :
                                                 std::vector<int64_t>{0, 2, 3, 6, 8, 10};
    auto quotient = [&](int64_t value) {
        return static_cast<int64_t>(floorDiv(fs::BoundInteger(value), fs::BoundInteger(period)));
    };
    auto residue = [&](int64_t value) {
        return static_cast<uint64_t>(value - quotient(value) * static_cast<int64_t>(period));
    };
    auto point = [&](int64_t from, int64_t to) {
        return fs::IntegerSystem::create(2, {row({1, 0}, quotient(from)), row({-1, 0}, -quotient(from)),
            row({0, 1}, quotient(to)), row({0, -1}, -quotient(to))});
    };
    for (auto source : visits) {
        // The WAIT's own IV is different: its ID must use the inverse map.
        auto instance = point(source, source + 11);
        if (mlir::failed(instance)) { return false; }
        analysis.minimumDemands[{{0, fs::ArithmeticEvent::Completion, {residue(source)}},
            {1, fs::ArithmeticEvent::Start, {residue(source + 11)}}, {}}].push_back(*instance);
        for (auto later : visits) {
            if (later - source < 4 || (missingReuse && source == 2 && later == 6)) { continue; }
            auto ordered = point(source + 11, later);
            if (mlir::failed(ordered)) { return false; }
            analysis.requiredOrder[{{1, fs::ArithmeticEvent::Completion, {residue(source + 11)}},
                {0, fs::ArithmeticEvent::Start, {residue(later)}}, {}}].push_back(*ordered);
        }
    }
    // The width-one obligations are identical. The coordinate construction
    // obtains earlier-source support from inverse(F*) rather than joining its
    // projection again. Preserve refusal while reducing exact join work.
    const auto full = fs::buildArithmeticHandoffAllocation(analysis, {0, 1}, 1);
    const auto factored = fs::buildArithmeticCoordinateHandoffAllocation(analysis, {0, 1}, 1);
    if (full.error.empty() != factored.error.empty() ||
        (period == 1 && factored.cost.pieceJoins >= full.cost.pieceJoins)) { return false; }
    auto proof = fs::buildArithmeticCoordinateHandoffAllocation(analysis, {0, 1}, 4);
    if (negative || missingReuse) { return !proof.error.empty(); }
    if (!proof.error.empty() || proof.families.size() != 1) { return false; }
    const auto& family = proof.families.front();
    if (family.sourceCoordinate != std::optional<unsigned>(0) || family.lifetime.width != 4) { return false; }
    // Actual matching-coordinate transport and physical reuse are checked by
    // coordinateModulo in PhysicalExecutedCounterChecks, through both emitters.
    auto last = fs::evaluateGeneralArithmeticSelector(family.lifetime.lastTarget, period, {}, {});
    return mlir::succeeded(last) && *last &&
        (**last).coordinates == std::vector<fs::BoundInteger>{fs::BoundInteger(21)};
}
bool coordinateContexts(bool missingReuse)
{
    fs::GeneralArithmeticDemandAnalysis analysis;
    analysis.exactMinimum = true; analysis.period = 2; analysis.parameterCount = 1;
    auto point = [](int64_t from, int64_t to, int64_t parameter) {
        return fs::IntegerSystem::create(3, {row({1, 0, 0}, from / 2), row({-1, 0, 0}, -from / 2),
            row({0, 1, 0}, to / 2), row({0, -1, 0}, -to / 2),
            row({0, 0, 1}, parameter / 2), row({0, 0, -1}, -parameter / 2)});
    };
    for (int64_t parameter : {0, 1, 4}) {
        const auto base = 10 * parameter;
        for (int64_t offset : {0, 1, 3, 4}) {
            const auto source = base + offset, target = source + 11;
            const auto p = static_cast<uint64_t>(parameter % 2);
            auto instance = point(source, target, parameter);
            if (mlir::failed(instance)) { return false; }
            analysis.minimumDemands[{{0, fs::ArithmeticEvent::Completion, {uint64_t(source % 2)}},
                {1, fs::ArithmeticEvent::Start, {uint64_t(target % 2)}}, {p}}].push_back(*instance);
            for (int64_t next : {0, 1, 3, 4}) {
                if (next - offset < 2 || (missingReuse && parameter == 4 && offset == 0 && next == 4)) { continue; }
                const auto later = base + next;
                auto ordered = point(target, later, parameter);
                if (mlir::failed(ordered)) { return false; }
                analysis.requiredOrder[{{1, fs::ArithmeticEvent::Completion, {uint64_t(target % 2)}},
                    {0, fs::ArithmeticEvent::Start, {uint64_t(later % 2)}}, {p}}].push_back(*ordered);
            }
        }
    }
    auto ledger = std::make_shared<fs::AnalysisLedger>();
    ledger->profiling = true;
    fs::ArithmeticCoordinateHandoffAllocation proof;
    {
        fs::AnalysisProfileScope request(ledger, "allocation", 0, "coordinate-context-test");
        fs::AllocationProfile profile(ledger);
        proof = fs::buildArithmeticCoordinateHandoffAllocation(analysis, {0, 1}, 2, &profile);
    }
    uint64_t exclusive = 0;
    unsigned buckets = 0;
    for (const auto& stage : ledger->stages) {
        exclusive += stage.exclusiveNanoseconds;
        if (stage.aggregateCalls) { ++buckets; }
    }
    if (ledger->stages.empty() || exclusive != ledger->stages.front().elapsedNanoseconds || buckets < 3) {
        llvm::errs() << "coordinate profiling: exclusive=" << exclusive << " buckets=" << buckets << "\n";
        return false;
    }
    if (missingReuse) { return !proof.error.empty(); }
    if (!proof.error.empty() || proof.families.size() != 1 || proof.families[0].lifetime.width != 2) {
        llvm::errs() << "coordinate contexts: " << proof.error << " families=" << proof.families.size() << "\n";
        return false;
    }
    const auto& family = proof.families[0].lifetime;
    for (int64_t parameter : {0, 1, 4, 7}) {
        const auto first = fs::evaluateGeneralArithmeticSelector(
            family.firstSource, 2, {}, {fs::BoundInteger(parameter)});
        const auto last = fs::evaluateGeneralArithmeticSelector(
            family.lastTarget, 2, {}, {fs::BoundInteger(parameter)});
        if (mlir::failed(first) || mlir::failed(last)) {
            llvm::errs() << "coordinate extrema evaluation failed: " << parameter << "\n";
            return false;
        }
        if (parameter == 7) { if (*first || *last) { return false; } continue; }
        if (!*first || !*last ||
            (**first).coordinates != std::vector<fs::BoundInteger>{fs::BoundInteger(10 * parameter)} ||
            (**last).coordinates != std::vector<fs::BoundInteger>{fs::BoundInteger(10 * parameter + 15)}) {
            llvm::errs() << "coordinate extrema mismatch: " << parameter << "\n";
            return false;
        }
    }
    return true;
}
bool boundedConstructionProfile()
{
    fs::GeneralArithmeticDemandAnalysis analysis;
    analysis.exactMinimum = true;
    std::vector<uint32_t> pipes;
    for (unsigned family = 0; family < 32; ++family) {
        const auto source = pipes.size();
        pipes.push_back(0); pipes.push_back(1);
        analysis.minimumDemands[{{source, fs::ArithmeticEvent::Completion, {}},
            {source + 1, fs::ArithmeticEvent::Start, {}}, {}}] = {fs::IntegerSystem{}};
    }
    for (bool coordinate : {false, true}) { for (bool enabled : {false, true}) {
        auto ledger = std::make_shared<fs::AnalysisLedger>();
        ledger->profiling = enabled;
        {
            fs::AnalysisProfileScope request(ledger, "allocation", 0, "bounded-coordinate-profile");
            fs::AllocationProfile profile(ledger);
            if (coordinate) {
                auto proof = fs::buildArithmeticCoordinateHandoffAllocation(analysis, pipes, 1, &profile);
                if (!proof.error.empty() || proof.families.size() != 32 ||
                    llvm::any_of(proof.families, [](const auto& family) { return family.lifetime.width != 1; })) {
                    return false;
                }
            } else {
                auto proof = fs::buildArithmeticHandoffAllocation(analysis, pipes, 1, &profile);
                if (!proof.error.empty() || proof.families.size() != 32 ||
                    llvm::any_of(proof.families, [](const auto& family) { return family.width != 1; })) {
                    return false;
                }
            }
        }
        unsigned workRecords = 0;
        for (const auto& stage : ledger->stages) {
            if (stage.method == "arithmetic-handoff-construction") { ++workRecords; }
        }
        if ((!enabled && !ledger->stages.empty()) ||
            (enabled && (workRecords != 1 || ledger->stages.size() > fs::AllocationProfile::Count + 2))) {
            return false;
        }
    } }
    return true;
}
bool coupledCompositionCoverage()
{
    using Relation = fs::GeneralArithmeticRelation;
    const fs::ArithmeticRelationKey ak{{1, fs::ArithmeticEvent::Completion, {0}},
        {0, fs::ArithmeticEvent::Start, {0, 0}}, {0}};
    const fs::ArithmeticRelationKey bk{ak.target, {2, fs::ArithmeticEvent::Start, {0}}, {0}};
    const fs::ArithmeticRelationKey ck{ak.source, bk.target, {0}};
    // A(x,z0,z1,p): z0=x+p, z1=x-p, even z0, -1<=x,p<=2.
    auto congruence = parity(1, 0);
    congruence.coefficients.push_back(fs::BoundInteger(0));
    auto a = fs::IntegerSystem::create(4, {
        row({-1, 1, 0, -1}, 0), row({1, -1, 0, 1}, 0),
        row({-1, 0, 1, 1}, 0), row({1, 0, -1, -1}, 0),
        row({1, 0, 0, 0}, 2), row({-1, 0, 0, 0}, 1),
        row({0, 0, 0, 1}, 2), row({0, 0, 0, -1}, 1)}, {congruence});
    if (mlir::failed(a)) { return false; }
    for (bool extra : {false, true}) {
        Relation left{{ak, {*a}}}, right, required;
        // B(z0,z1,y,p): y=z0+2*z1+offset. The second disjunct
        // has a distinct uncovered output for each admitted witness.
        for (int64_t offset = 0; offset <= int64_t(extra); ++offset) {
            auto b = fs::IntegerSystem::create(4, {
                row({-1, -2, 1, 0}, offset), row({1, 2, -1, 0}, -offset)});
            if (mlir::failed(b)) { return false; }
            right[bk].push_back(*b);
        }
        for (bool positive : {false, true}) {
            auto c = fs::IntegerSystem::create(3, {
                row({-3, 1, 1}, 0), row({3, -1, -1}, 0),
                positive ? row({-1, 0, 0}, -1) : row({1, 0, 0}, 0)});
            if (mlir::failed(c)) { return false; }
            required[ck].push_back(*c);
        }
        fs::ArithmeticAnalysisCost projectedCost, liftedCost;
        auto projected = fs::detail::materializeHandoffComposition(left, right, 1, projectedCost);
        auto lifted = fs::detail::handoffCompositionCovered(left, right, required, 1, liftedCost);
        if (mlir::failed(projected) || mlir::failed(lifted)) { return false; }
        bool reference = true, finite = true;
        for (const auto& [key, pieces] : *projected) {
            auto missing = fs::subtractIntegerUnions(3, pieces, required.at(key));
            if (mlir::failed(missing)) { return false; }
            reference &= missing->empty();
        }
        for (int x = -1; x <= 2; ++x) { for (int p = -1; p <= 2; ++p) {
            for (int z0 = -3; z0 <= 4; ++z0) { for (int z1 = -3; z1 <= 4; ++z1) {
                for (int y = -5; y <= 8; ++y) {
                    const bool joins = z0 == x + p && z1 == x - p && z0 % 2 == 0 &&
                        (y == z0 + 2 * z1 || (extra && y == z0 + 2 * z1 + 1));
                    if (joins && y != 3 * x - p) { finite = false; }
                }
            } }
        } }
        if (*lifted != finite || *lifted != reference || *lifted == extra ||
            liftedCost.projections || liftedCost.relationCompositions) { return false; }
    }
    return true;
}
bool liftedCompositionCoverage()
{
    using Relation = fs::GeneralArithmeticRelation;
    const std::vector<fs::IntegerConstraint> box{
        row({1, 0, 0}, 2), row({-1, 0, 0}, 1), row({0, 1, 0}, 2), row({0, -1, 0}, 1),
        row({0, 0, 1}, 2), row({0, 0, -1}, 1)};
    for (unsigned mode = 0; mode < 11; ++mode) {
        fs::ArithmeticRelationKey leftKey{{1, fs::ArithmeticEvent::Completion, {0}},
            {0, fs::ArithmeticEvent::Start, {0}}, {0}};
        fs::ArithmeticRelationKey rightKey{{0, fs::ArithmeticEvent::Start, {0}},
            {2, fs::ArithmeticEvent::Start, {0}}, {0}};
        fs::ArithmeticRelationKey coverKey{leftKey.source, rightKey.target, {0}};
        if (mode == 3) { rightKey.source.event = fs::ArithmeticEvent::Completion; }
        if (mode == 4) { rightKey.source.residues[0] = 1; }
        if (mode == 5) { rightKey.parameterResidues[0] = 1; }
        if (mode == 7) { coverKey.parameterResidues[0] = 1; }
        auto leftRows = box;
        if (mode == 9) { leftRows.push_back(row({0, 0, 1}, -2)); }
        auto left = fs::IntegerSystem::create(3, leftRows, {parity(2, 0)});
        auto right = fs::IntegerSystem::create(3, box, {parity(2, mode == 6 ? 1 : 0)});
        if (mlir::failed(left) || mlir::failed(right)) { return false; }
        Relation a{{leftKey, {*left}}}, b{{rightKey, {*right}}}, required;
        if (mode != 2) {
            for (unsigned half = 0; half < (mode == 1 ? 1u : 2u); ++half) {
                auto rows = box;
                rows.push_back(half ? row({-1, 0, 0}, -1) : row({1, 0, 0}, 0));
                auto piece = fs::IntegerSystem::create(3, rows, {parity(2, mode == 8 ? 1 : 0)});
                if (mlir::failed(piece)) { return false; }
                required[coverKey].push_back(*piece);
            }
        }
        if (mode == 10) {
            // Multiple disjuncts and intermediate witnesses must all satisfy
            // coverage; no one required disjunct covers the whole join.
            a[leftKey].push_back(*left);
            b[rightKey].push_back(*right);
        }
        fs::ArithmeticAnalysisCost projectedCost, liftedCost;
        const auto projected = fs::detail::materializeHandoffComposition(a, b, 1, projectedCost);
        const auto lifted = fs::detail::handoffCompositionCovered(a, b, required, 1, liftedCost);
        if (mlir::failed(projected) || mlir::failed(lifted)) { return false; }
        bool reference = true;
        for (const auto& [key, pieces] : *projected) {
            const auto found = required.find(key);
            const std::vector<fs::IntegerSystem> empty;
            auto missing = fs::subtractIntegerUnions(3, pieces, found == required.end() ? empty : found->second);
            if (mlir::failed(missing)) { return false; }
            reference &= missing->empty();
        }
        // Independent exhaustive interpretation of these finite formulas.
        bool finite = true;
        for (int x = -1; x <= 2; ++x) { for (int z = -1; z <= 2; ++z) {
            for (int y = -1; y <= 2; ++y) { for (int p = -1; p <= 2; ++p) {
                const bool joins = mode != 3 && mode != 4 && mode != 5 && mode != 6 && mode != 9 && p % 2 == 0;
                const bool covered = mode != 2 && mode != 7 && mode != 8 && (mode != 1 || x <= 0);
                if (joins && !covered) { finite = false; }
            } }
        } }
        if (*lifted != reference || *lifted != finite || liftedCost.projections ||
            liftedCost.relationCompositions || liftedCost.pieceJoins > projectedCost.pieceJoins ||
            (mode == 0 && !projectedCost.projections)) { return false; }
    }
    return true;
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
    auto checked = [](llvm::StringRef name, bool valid) {
        if (!valid) { llvm::errs() << name << " failed\n"; }
        return valid;
    };
    if (!checked("check(context)", check(context)) ||
        !checked("unaryHandoffExtremaChecks", unaryHandoffExtremaChecks::run()) ||
        !checked("parameterContexts(context)", parameterContexts(context)) ||
        !checked("singleton(context)", singleton(context)) ||
        !checked("nestedCoordinates(context)", nestedCoordinates(context)) ||
        !checked("activeSuccessors(1)", activeSuccessors(1)) ||
        !checked("activeSuccessors(3)", activeSuccessors(3)) ||
        !checked("differenceBoundSuccessors(context)", differenceBoundSuccessors(context)) ||
        !checked("coordinateMapping(1, false, false)", coordinateMapping(1, false, false)) ||
        !checked("coordinateMapping(3, false, false)", coordinateMapping(3, false, false)) ||
        !checked("coordinateMapping(5, false, false)", coordinateMapping(5, false, false)) ||
        !checked("coordinateMapping(3, true, false)", coordinateMapping(3, true, false)) ||
        !checked("coordinateMapping(3, false, true)", coordinateMapping(3, false, true)) ||
        !checked("coordinateContexts(false)", coordinateContexts(false)) ||
        !checked("coordinateContexts(true)", coordinateContexts(true)) ||
        !checked("boundedConstructionProfile()", boundedConstructionProfile()) ||
        !checked("liftedCompositionCoverage()", liftedCompositionCoverage()) ||
        !checked("coupledCompositionCoverage()", coupledCompositionCoverage())) {
        llvm::errs() << "general arithmetic allocation certificate check failed\n";
        return 1;
    }
    llvm::outs() << "general arithmetic allocation: exact unions, residues and all source pairs passed\n";
    return 0;
}
