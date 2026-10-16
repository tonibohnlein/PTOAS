// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TEST_UNARYHANDOFFEXTREMACHECKS_H
#define PTO_TEST_UNARYHANDOFFEXTREMACHECKS_H
#include "../../lib/PTO/Transforms/FrontierSynch/Arithmetic/ArithmeticUnaryExtrema.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Arithmetic/ArithmeticHandoffProof.h"
#include "PTO/Transforms/FrontierSynch/AnalysisProfile.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
namespace unaryHandoffExtremaChecks {
namespace fs = mlir::pto::frontiersynch;
using Integer = fs::BoundInteger;
inline fs::IntegerConstraint row(std::initializer_list<int64_t> values, int64_t bound)
{
    fs::IntegerConstraint result;
    for (auto value : values) { result.coefficients.emplace_back(value); }
    result.bound = Integer(bound);
    return result;
}
inline fs::IntegerCongruence congruence(std::initializer_list<int64_t> values, int64_t residue, int64_t modulus)
{
    fs::IntegerCongruence result;
    for (auto value : values) { result.coefficients.emplace_back(value); }
    result.residue = Integer(residue); result.modulus = Integer(modulus);
    return result;
}
inline bool contains(const fs::IntegerSystem& system, llvm::ArrayRef<Integer> point)
{
    if (system.isKnownEmpty()) { return false; }
    for (const auto& atom : system.constraints()) {
        Integer value(0);
        for (unsigned i = 0; i < point.size(); ++i) { value += point[i] * atom.coefficients[i]; }
        if (value > atom.bound) { return false; }
    }
    for (const auto& atom : system.congruences()) {
        Integer value(0);
        for (unsigned i = 0; i < point.size(); ++i) { value += point[i] * atom.coefficients[i]; }
        if ((value - atom.residue) % atom.modulus != Integer(0)) { return false; }
    }
    return true;
}
inline bool compareFinite(const fs::GeneralArithmeticRelation& support, uint64_t period, bool baseline)
{
    for (bool first : {false, true}) {
        fs::ArithmeticAnalysisCost cost;
        auto fast = fs::detail::unaryHandoffExtremum(support, period, 2, first, cost);
        if (!fast || cost.integerDifference.requests != cost.differences) { return false; }
        std::optional<fs::GeneralArithmeticEndpointSelector> reference;
        if (baseline) {
            fs::ArithmeticAnalysisCost oldCost;
            auto old = fs::detail::materializeHandoffExtremum(support, period, 2, first, oldCost, false);
            if (mlir::failed(old)) { return false; }
            reference.emplace(std::move(*old));
        }
        for (int64_t a = -2; a <= 2; ++a) {
            for (int64_t b = -2; b <= 2; ++b) {
                for (uint64_t parameterResidue = 0; parameterResidue < period; ++parameterResidue) {
                    std::optional<Integer> expected;
                    // Every fixture below bounds all feasible quotients in
                    // [-12,12] for these parameter valuations. Enumerate the
                    // ORIGINAL support, not candidate domains or witnesses.
                    for (const auto& [key, pieces] : support) {
                        if (key.parameterResidues != std::vector<uint64_t>{parameterResidue, 0}) { continue; }
                        for (int64_t q = -12; q <= 12; ++q) {
                            const Integer point[]{Integer(q), Integer(a), Integer(b)};
                            if (!llvm::any_of(pieces, [&](const auto& piece) { return contains(piece, point); })) {
                                continue;
                            }
                            auto value = Integer(static_cast<int64_t>(period)) * Integer(q) +
                                Integer(static_cast<int64_t>(key.source.residues.front()));
                            if (!expected || (first ? value < *expected : value > *expected)) { expected = value; }
                        }
                    }
                    const Integer parameters[]{Integer(static_cast<int64_t>(period)) * Integer(a) +
                        Integer(static_cast<int64_t>(parameterResidue)),
                        Integer(static_cast<int64_t>(period)) * Integer(b)};
                    auto check = [&](const fs::GeneralArithmeticEndpointSelector& selector) {
                        auto selected = fs::evaluateGeneralArithmeticSelector(selector, period, {}, parameters);
                        if (mlir::failed(selected) || bool(*selected) != bool(expected)) { return false; }
                        if (expected && ((**selected).site != 3 ||
                            (**selected).coordinates != std::vector<Integer>{*expected})) { return false; }
                        // Stable first match must not hide an overlapping
                        // wrong witness: every enabled piece agrees exactly.
                        for (const auto& piece : selector.pieces) {
                            auto single = selector; single.pieces = {piece};
                            auto value = fs::evaluateGeneralArithmeticSelector(single, period, {}, parameters);
                            if (mlir::failed(value) || (*value && (!expected ||
                                (**value).coordinates != std::vector<Integer>{*expected}))) { return false; }
                        }
                        return true;
                    };
                    if (!check(*fast) || (reference && !check(*reference))) { return false; }
                }
            }
        }
    }
    return true;
}
inline bool fixture(unsigned mode, uint64_t period)
{
    fs::GeneralArithmeticRelation support;
    for (uint64_t residue : {uint64_t(0), period - 1}) {
        fs::ArithmeticRelationKey key{{3, fs::ArithmeticEvent::Start, {residue}}, {}, {0, 0}};
        std::vector<fs::IntegerConstraint> rows;
        std::vector<fs::IntegerCongruence> congruences;
        switch (mode) {
        case 0:
            // Several active lower rows; no opposite affine strip or unary
            // interval. Cooper witnesses must include the least feasible one.
            rows = {row({-1, 1, 0}, 0), row({-1, 0, -1}, 0), row({1, -1, -1}, 5)};
            congruences = {congruence({2, 1, 0}, 1, 3), congruence({1, 0, 1}, 0, 2)};
            break;
        case 1:
            // Five-value unary interval is smaller than its congruence period.
            rows = {row({-1, 0, 0}, 2), row({1, 0, 0}, 2)};
            congruences = {congruence({1, 1, 0}, 0, 17)};
            break;
        case 2:
            // Narrow nonunit affine strip, with coupled noncoprime residues.
            rows = {row({2, -1, 0}, 1), row({-2, 1, 0}, 1)};
            congruences = {congruence({2, 0, 1}, 0, 4), congruence({3, 1, 0}, 0, 6)};
            break;
        case 3:
            // Equality has an integral witness only for even parameter a.
            rows = {row({2, -1, 0}, 0), row({-2, 1, 0}, 0)};
            congruences = {congruence({1, 0, 1}, 0, 2)};
            break;
        default:
            rows = {row({-1, 0, 0}, 4), row({1, 0, 0}, -1), row({0, 1, 0}, 1)};
            break;
        }
        auto system = fs::IntegerSystem::create(3, rows, congruences);
        if (mlir::failed(system)) { return false; }
        support[key].push_back(*system);
        if (mode == 4) {
            auto later = fs::IntegerSystem::create(3,
                {row({-1, 0, 0}, -1), row({1, 0, 0}, 3), row({0, -1, 0}, 0)});
            if (mlir::failed(later)) { return false; }
            support[key].push_back(*later); support[key].push_back(*system);
            if (period > 1) {
                key.parameterResidues[0] = 1;
                support[key] = {*later};
            }
        }
    }
    return compareFinite(support, period, mode == 4 && period == 1);
}
inline bool residual()
{
    fs::ArithmeticRelationKey key{{3, fs::ArithmeticEvent::Start, {0}}, {}, {0, 0}};
    auto upper = fs::IntegerSystem::create(3, {row({1, 0, 0}, 4)});
    auto point = fs::IntegerSystem::create(3, {row({1, 0, 0}, 1), row({-1, 0, 0}, -1)});
    if (mlir::failed(upper) || mlir::failed(point)) { return false; }
    fs::ArithmeticAnalysisCost cost;
    fs::GeneralArithmeticRelation support{{key, {*upper}}};
    if (fs::detail::unaryHandoffExtremum(support, 1, 2, true, cost) || cost.projections || cost.differences) {
        return false;
    }
    for (bool enabled : {false, true}) {
        auto noMinimum = fs::detail::materializeHandoffExtremum(support, 1, 2, true, cost, enabled);
        if (mlir::failed(noMinimum) || !noMinimum->pieces.empty()) { return false; }
    }
    support[key] = {*point};
    key.source.site = 4; support[key] = {*point};
    if (fs::detail::unaryHandoffExtremum(support, 1, 2, true, cost)) { return false; }
    for (bool enabled : {false, true}) {
        if (mlir::succeeded(fs::detail::materializeHandoffExtremum(support, 1, 2, true, cost, enabled))) {
            return false;
        }
    }
    support.clear(); key.source.residues = {0, 0};
    auto nested = fs::IntegerSystem::create(4,
        {row({1, 0, 0, 0}, 0), row({-1, 0, 0, 0}, 0), row({0, 1, 0, 0}, 1), row({0, -1, 0, 0}, -1)});
    if (mlir::failed(nested)) { return false; }
    support[key] = {*nested};
    if (fs::detail::unaryHandoffExtremum(support, 1, 2, true, cost)) { return false; }
    for (bool enabled : {false, true}) {
        auto result = fs::detail::materializeHandoffExtremum(support, 1, 2, true, cost, enabled);
        if (mlir::failed(result)) { return false; }
        auto value = fs::evaluateGeneralArithmeticSelector(*result, 1, {}, {Integer(0), Integer(0)});
        if (mlir::failed(value) || !*value || (**value).coordinates != std::vector<Integer>{Integer(0), Integer(1)}) {
            return false;
        }
    }
    return true;
}
inline bool parameterFree()
{
    fs::GeneralArithmeticRelation support;
    fs::ArithmeticRelationKey key{{3, fs::ArithmeticEvent::Start, {2}}, {}, {}};
    auto negative = fs::IntegerSystem::create(1, {row({1}, -2), row({-1}, 2)});
    auto positive = fs::IntegerSystem::create(1, {row({1}, 1), row({-1}, -1)});
    if (mlir::failed(negative) || mlir::failed(positive)) { return false; }
    support[key] = {*negative}; key.source.residues = {0}; support[key] = {*positive};
    for (bool first : {false, true}) {
        fs::ArithmeticAnalysisCost cost;
        auto selected = fs::detail::unaryHandoffExtremum(support, 3, 0, first, cost);
        if (!selected) { return false; }
        auto value = fs::evaluateGeneralArithmeticSelector(*selected, 3, {}, {});
        if (mlir::failed(value) || !*value ||
            (**value).coordinates != std::vector<Integer>{Integer(first ? -4 : 3)}) { return false; }
    }
    fs::ArithmeticAnalysisCost cost;
    auto empty = fs::detail::unaryHandoffExtremum({}, 3, 0, true, cost);
    fs::detail::unaryExtrema::Work accumulated;
    auto repeated = fs::detail::unaryHandoffExtremum({}, 3, 0, true, cost, nullptr, &accumulated);
    auto unsupported = fs::detail::unaryHandoffExtremum(support, 3, 1, true, cost, nullptr, &accumulated);
    return empty && empty->pieces.empty() && repeated && !unsupported &&
        accumulated.requests == 2 && accumulated.qualified == 1 && accumulated.selected == 1;
}
inline bool profile()
{
    fs::ArithmeticRelationKey key{{3, fs::ArithmeticEvent::Start, {0}}, {}, {}};
    auto interval = fs::IntegerSystem::create(1, {row({-1}, 2), row({1}, 3)});
    if (mlir::failed(interval)) { return false; }
    for (bool enabled : {false, true}) {
        auto ledger = std::make_shared<fs::AnalysisLedger>(); ledger->profiling = enabled;
        fs::ArithmeticAnalysisCost cost;
        cost.integerDifference.requests = 7; // Earlier unrelated work is not attributed to this attempt.
        {
            fs::AnalysisProfileScope request(ledger, "allocation", 0, "unary-extrema-profile");
            fs::AllocationProfile profile(ledger);
            auto result = fs::detail::unaryHandoffExtremum({{key, {*interval}}}, 1, 0, true, cost, &profile);
            if (!result || result->pieces.size() != 1) { return false; }
        }
        if (cost.projections != 1 || cost.differences != 1 || cost.integerDifference.requests != 8) { return false; }
        unsigned records = 0;
        for (const auto& record : ledger->stages) {
            if (record.method != "arithmetic-unary-extrema") { continue; }
            ++records;
            std::map<std::string, uint64_t> work(record.work.begin(), record.work.end());
            if (work["qualified"] != 1 || work["selected"] != 1 || work["witness-requests"] != 1 ||
                work["candidate-witnesses"] != 1 || work["parameter-differences"] != 1 ||
                work["selected-pieces"] != 1) { return false; }
        }
        if (enabled ? records != 1 : !ledger->stages.empty()) { return false; }
    }
    return true;
}
inline bool scaling()
{
    for (unsigned count : {4U, 8U, 16U}) {
        fs::GeneralArithmeticRelation support;
        fs::ArithmeticRelationKey key{{3, fs::ArithmeticEvent::Start, {0}}, {}, {}};
        for (unsigned i = 0; i < count; ++i) {
            const auto coordinate = -int64_t(count) + 3 * int64_t(i);
            auto point = fs::IntegerSystem::create(1, {row({1}, coordinate), row({-1}, -coordinate)});
            if (mlir::failed(point)) { return false; }
            support[key].push_back(std::move(*point));
        }
        for (bool first : {false, true}) {
            fs::ArithmeticAnalysisCost fastCost, oldCost;
            auto fast = fs::detail::materializeHandoffExtremum(support, 1, 0, first, fastCost, true);
            auto old = fs::detail::materializeHandoffExtremum(support, 1, 0, first, oldCost, false);
            if (mlir::failed(fast) || mlir::failed(old) || fastCost.pieceJoins >= oldCost.pieceJoins ||
                fastCost.projections >= oldCost.projections || fastCost.integerDifference.requests != count) {
                return false;
            }
            const auto expected = -int64_t(count) + (first ? 0 : 3 * (int64_t(count) - 1));
            for (const auto* selector : {&*fast, &*old}) {
                auto value = fs::evaluateGeneralArithmeticSelector(*selector, 1, {}, {});
                if (mlir::failed(value) || !*value ||
                    (**value).coordinates != std::vector<Integer>{Integer(expected)}) { return false; }
            }
        }
    }
    return true;
}
inline bool run()
{
    for (unsigned mode = 0; mode < 5; ++mode) {
        if (!fixture(mode, 1)) { llvm::errs() << "unary extrema finite oracle mode " << mode << "\n"; return false; }
    }
    return fixture(2, 3) && fixture(4, 3) && fixture(4, 5) && residual() && parameterFree() && profile() && scaling();
}
} // namespace unaryHandoffExtremaChecks
#endif
