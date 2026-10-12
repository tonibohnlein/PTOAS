// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Infer conservative action counts from original signed SCF domains using wide arithmetic.
#include "InvocationSourceBounds.h"
#include "CountedLoop.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"
#include <algorithm>
#include <limits>
#include <set>
namespace mlir::pto::frontiersynch {
namespace {
using Counts = std::map<uint32_t, uint64_t>;
constexpr uint64_t maximumCount = uint64_t(INT64_MAX) - 1;
std::optional<APInt> loopBound(scf::ForOp loop)
{
    auto domain = CountedLoop::get(loop);
    APInt step;
    const auto bits = DataLayout::closest(loop).getTypeSizeInBits(loop.getInductionVar().getType());
    const bool supported = loop.getInductionVar().getType().isIndex() && !bits.isScalable() &&
        bits.getFixedValue() == 64 && matchPattern(loop.getStep(), m_ConstantInt(&step)) &&
        step.isSignedIntN(64) && step.getSExtValue() > 0;
    if (!supported) { return std::nullopt; }
    const bool empty = loop.getLowerBound() == loop.getUpperBound();
    if (empty) { return APInt(128, 0); }
    mlir::pto::detail::ScalarEvolution ranges(loop.getContext(), loop);
    const auto lower = ranges.signedRange(loop.getLowerBound());
    const auto upper = ranges.signedRange(loop.getUpperBound());
    std::optional<APInt> result;
    if (lower && upper) {
        const auto distance = APInt(128, upper->second, true) - APInt(128, lower->first, true);
        result = distance.isStrictlyPositive() ?
            (distance - 1).udiv(step.sextOrTrunc(128)) + 1 : APInt(128, 0);
    }
    // The correlated-domain certificate can improve independent endpoint ranges.
    // maximumOrdinal=0 can mean empty or one visit: +1 is a safe upper bound.
    if (domain) {
        APInt correlated(128, domain->maximumOrdinal); correlated += 1;
        if (!result || correlated.ult(*result)) { result = correlated; }
    }
    return result;
}
class Counter {
public:
    Counter(func::FuncOp original, ArrayRef<PrefixSource> sources, std::string& error)
        : original(original), sources(sources), error(error) {}
    FailureOr<Counts> run()
    {
        const bool validFunction = original && original.getBody().hasOneBlock() && succeeded(verify(original));
        if (!validFunction) { return fail("source bounds require one original function body block"); }
        std::set<std::pair<Operation*, uint32_t>> seen;
        for (const auto& source : sources) {
            const bool valid = source.original && original->isProperAncestor(source.original) &&
                !source.original->hasTrait<OpTrait::IsTerminator>() &&
                seen.emplace(source.original, source.pipe).second;
            if (!valid) { return fail("source bounds have invalid or duplicate original action provenance"); }
            at[source.original].push_back(source.pipe);
        }
        auto result = block(original.front());
        if (failed(result)) { return failure(); }
        for (const auto& source : sources) { result->try_emplace(source.pipe, 0); }
        return result;
    }
private:
    FailureOr<Counts> fail(StringRef message) { error = message.str(); return failure(); }
    bool add(Counts& total, const Counts& extra)
    {
        for (auto [pipe, count] : extra) {
            auto& current = total[pipe];
            if (count > maximumCount - current) {
                error = "invocation source count and exclusive sentinel are not representable";
                return false;
            }
            current += count;
        }
        return true;
    }
    FailureOr<Counts> branch(scf::IfOp conditional)
    {
        auto thenCounts = block(conditional.getThenRegion().front());
        if (failed(thenCounts)) { return failure(); }
        if (!conditional.getElseRegion().empty()) {
            auto elseCounts = block(conditional.getElseRegion().front());
            if (failed(elseCounts)) { return failure(); }
            for (auto [pipe, count] : *elseCounts) { (*thenCounts)[pipe] = std::max((*thenCounts)[pipe], count); }
        }
        return thenCounts;
    }
    FailureOr<Counts> loop(scf::ForOp counted)
    {
        auto bound = loopBound(counted);
        if (bound && bound->isZero()) { return Counts{}; }
        auto body = block(*counted.getBody());
        const bool terminal = failed(body) || body->empty();
        if (terminal) { return body; }
        if (!bound) { return fail("source-bearing loop has no certified original visit bound"); }
        for (auto& item : *body) {
            const auto product = *bound * APInt(128, item.second);
            if (product.ugt(APInt(128, maximumCount))) {
                return fail("nested source count and exclusive sentinel are not representable");
            }
            item.second = product.getZExtValue();
        }
        return body;
    }
    FailureOr<Counts> block(Block& source)
    {
        Counts total;
        for (auto& operation : source) {
            FailureOr<Counts> nested = Counts{};
            if (auto counted = dyn_cast<scf::ForOp>(operation)) { nested = loop(counted); }
            else if (auto conditional = dyn_cast<scf::IfOp>(operation)) { nested = branch(conditional); }
            else if (operation.getNumRegions() || operation.getNumSuccessors()) {
                return fail("source bounds require supported original structured control");
            }
            const bool invalid = failed(nested) || !add(total, *nested);
            if (invalid) { return failure(); }
            auto found = at.find(&operation);
            if (found != at.end()) {
                for (auto pipe : found->second) { if (!add(total, Counts{{pipe, 1}})) { return failure(); } }
            }
        }
        return total;
    }
    func::FuncOp original;
    ArrayRef<PrefixSource> sources;
    std::string& error;
    std::map<Operation*, SmallVector<uint32_t>> at;
};
} // namespace
FailureOr<std::map<uint32_t, uint64_t>> inferInvocationSourceBounds(
    func::FuncOp original, llvm::ArrayRef<PrefixSource> sources, std::string& error)
{
    error.clear();
    return Counter(original, sources, error).run();
}
} // namespace mlir::pto::frontiersynch
