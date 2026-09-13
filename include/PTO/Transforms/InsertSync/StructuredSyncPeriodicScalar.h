// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_INSERTSYNC_STRUCTUREDSYNCPERIODICSCALAR_H
#define PTO_TRANSFORMS_INSERTSYNC_STRUCTUREDSYNCPERIODICSCALAR_H

#include "PTO/Transforms/InsertSync/StructuredSyncOrdinal.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseMap.h"
#include <algorithm>
#include <set>

namespace mlir::pto::structured_sync {
// Shared exact interpretation of a qualified scalar fragment in iteration
// ordinals. Callers choose optional work bounds; failure loses precision only.
class PeriodicScalar {
    static std::optional<int64_t> literal(Value value)
    {
        IntegerAttr attr;
        if (!value || !matchPattern(value, m_Constant(&attr)) || !attr.getValue().isSignedIntN(64))
            return {};
        if (value.getType().isInteger(1))
            return int64_t(attr.getValue().getZExtValue());
        return attr.getValue().getSExtValue();
    }
    Value iv;
    LoopOrdinal induction;
    uint64_t offset = 0;
    bool initial = false, split = false;
    uint64_t lastOrdinal = 0;
    llvm::DenseMap<Value, std::optional<uint64_t>> periods;
    struct Affine {
        int64_t coefficient, constant;
    };
    llvm::DenseMap<Value, std::optional<Affine>> affineCache;

    std::optional<Affine> affine(Value value)
    {
        auto found = affineCache.find(value);
        if (found != affineCache.end())
            return found->second;
        auto compute = [&]() -> std::optional<Affine> {
            if (!isa<IndexType>(value.getType()))
                return {};
            if (value == iv)
                return Affine{induction.step, induction.lower};
            if (auto c = literal(value))
                return Affine{0, *c};
            auto* op = value.getDefiningOp();
            if (!op || op->getNumOperands() != 2 || !isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(op))
                return {};
            auto x = affine(op->getOperand(0)), y = affine(op->getOperand(1));
            if (!x || !y)
                return {};
            std::optional<int64_t> a, b;
            if (isa<arith::MulIOp>(op)) {
                if (x->coefficient && y->coefficient)
                    return {};
                if (y->coefficient)
                    std::swap(x, y);
                a = ordinalMultiply(x->coefficient, y->constant);
                b = ordinalMultiply(x->constant, y->constant);
            } else {
                if (isa<arith::SubIOp>(op)) {
                    auto ya = ordinalNegate(y->coefficient), yb = ordinalNegate(y->constant);
                    if (!ya || !yb)
                        return {};
                    y = Affine{*ya, *yb};
                }
                a = ordinalAdd(x->coefficient, y->coefficient);
                b = ordinalAdd(x->constant, y->constant);
            }
            if (!a || !b || lastOrdinal > uint64_t(INT64_MAX))
                return {};
            auto product = ordinalMultiply(*a, int64_t(lastOrdinal));
            if (!product || !ordinalAdd(*product, *b))
                return {};
            return Affine{*a, *b};
        };
        auto result = compute();
        affineCache[value] = result;
        return result;
    }
    // Return the value at ordinal zero of an eq/ne that flips permanently
    // after ordinal zero. Difference is used only after width/range checks.
    std::optional<bool> firstTest(Value value)
    {
        auto cmp = value.getDefiningOp<arith::CmpIOp>();
        if (!cmp || (cmp.getPredicate() != arith::CmpIPredicate::eq && cmp.getPredicate() != arith::CmpIPredicate::ne))
            return {};
        auto x = affine(cmp.getLhs()), y = affine(cmp.getRhs());
        if (!x || !y)
            return {};
        auto ya = ordinalNegate(y->coefficient), yb = ordinalNegate(y->constant);
        if (!ya || !yb)
            return {};
        auto a = ordinalAdd(x->coefficient, *ya), b = ordinalAdd(x->constant, *yb);
        if (!a || !b || !*a || *b)
            return {};
        return cmp.getPredicate() == arith::CmpIPredicate::eq;
    }
    std::optional<OrdinalResidue> cycle(Value value) const
    {
        auto* op = value.getDefiningOp();
        if (!op)
            return {};
        std::optional<int64_t> modulus;
        if (isa<arith::RemSIOp, arith::RemUIOp>(op) && op->getOperand(0) == iv)
            modulus = literal(op->getOperand(1));
        if (auto mask = dyn_cast<arith::AndIOp>(op)) {
            Value other = mask.getLhs() == iv ? mask.getRhs() : (mask.getRhs() == iv ? mask.getLhs() : Value());
            if (other) {
                auto bits = literal(other);
                if (!bits || *bits < 0 || *bits == INT64_MAX)
                    return {};
                const uint64_t m = uint64_t(*bits) + 1;
                if (m & (m - 1))
                    return {};
                modulus = int64_t(m);
            }
        }
        if (!modulus || *modulus <= 0)
            return {};
        return OrdinalResidue::get(induction, uint64_t(*modulus), offset);
    }
    void cut(const OrdinalResidue& c, uint64_t raw)
    {
        if (auto r = c.preimage(raw)) {
            cuts.insert(*r);
            if (*r < UINT64_MAX)
                cuts.insert(*r + 1);
        }
    }

public:
    std::set<uint64_t> cuts{0};
    PeriodicScalar(
        Value originalIV, LoopOrdinal loop = {}, uint64_t start = 0, bool first = false, bool startup = false,
        std::optional<int64_t> upper = {})
        : iv(originalIV), induction(loop), offset(start), initial(first), split(startup)
    {
        auto count = induction.count(upper.value_or(INT64_MAX));
        lastOrdinal = count && *count ? *count - 1 : 0;
    }
    bool hasInitialization(Value value)
    {
        if (firstTest(value))
            return true;
        auto* op = value.getDefiningOp();
        if (!op || !value.getType().isInteger(1) || !isa<arith::CmpIOp, arith::AndIOp, arith::OrIOp, arith::XOrIOp>(op))
            return false;
        for (Value operand : op->getOperands())
            if (hasInitialization(operand))
                return true;
        return false;
    }
    std::optional<uint64_t> selectorPeriod(Value value) const
    {
        if (auto c = cycle(value))
            return c->period;
        return {};
    }
    std::optional<uint64_t> period(Value value)
    {
        auto found = periods.find(value);
        if (found != periods.end())
            return found->second;
        auto result = computePeriod(value);
        periods[value] = result;
        return result;
    }
    std::optional<uint64_t> computePeriod(Value value)
    {
        if (literal(value))
            return 1;
        if (firstTest(value))
            return split ? std::optional<uint64_t>(1) : std::nullopt;
        if (auto c = cycle(value))
            return initial ? uint64_t(1) : c->period;
        auto* op = value.getDefiningOp();
        if (!op)
            return {};
        if (!isa<arith::CmpIOp>(op) &&
            !(value.getType().isInteger(1) && isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp>(op)))
            return {};
        uint64_t p = 1;
        for (Value operand : op->getOperands()) {
            auto q = period(operand);
            if (!q || (p != 1 && *q != 1 && p != *q))
                return {};
            p = std::max(p, *q);
        }
        if (auto cmp = dyn_cast<arith::CmpIOp>(op)) {
            // The loop below partitions a residue VALUE against a literal.
            // Equal effective periods do not make two residue values constant
            // on those intervals: for i=6*k, (i%4)!=(i%12) is false at k=0
            // and true at k=1, although both periods are 2. With no cuts, the
            // old importer could erase the entire executing true arm.
            // Boolean operands already carry their own truth-change cuts;
            // period-one operands are constant in this represented phase.
            // Other value/value comparisons need a separate exact partition.
            if (p > 1 && !cmp.getLhs().getType().isInteger(1) && !literal(cmp.getLhs()) && !literal(cmp.getRhs()))
                return {};
            for (unsigned side = 0; side < 2; ++side) {
                auto c = cycle(op->getOperand(side));
                auto bound = literal(op->getOperand(1 - side));
                if (!c || !bound || initial)
                    continue;
                bool equality =
                    cmp.getPredicate() == arith::CmpIPredicate::eq || cmp.getPredicate() == arith::CmpIPredicate::ne;
                // General strided threshold permutations are not flattened to
                // intervals. Equality has one modular preimage; unit stride
                // also admits threshold and wrap cuts. Other shapes decline.
                if (!equality && induction.step != 1 && c->period != 1)
                    return {};
                cut(*c, 0);
                if (*bound >= 0) {
                    cut(*c, uint64_t(*bound));
                    if (*bound < INT64_MAX)
                        cut(*c, uint64_t(*bound) + 1);
                }
            }
        }
        return p;
    }
    std::optional<int64_t> evaluate(Value value, uint64_t residue)
    {
        llvm::DenseMap<Value, std::optional<int64_t>> cache;
        return eval(value, residue, cache);
    }

private:
    std::optional<int64_t> eval(Value value, uint64_t residue, llvm::DenseMap<Value, std::optional<int64_t>>& cache)
    {
        auto found = cache.find(value);
        if (found != cache.end())
            return found->second;
        auto compute = [&]() -> std::optional<int64_t> {
            if (auto c = literal(value))
                return c;
            if (auto first = firstTest(value)) {
                if (!split)
                    return {};
                return initial ? *first : !*first;
            }
            if (auto c = cycle(value))
                return int64_t(c->at(initial ? 0 : residue));
            auto* op = value.getDefiningOp();
            if (!op || op->getNumOperands() != 2)
                return {};
            auto a = eval(op->getOperand(0), residue, cache), b = eval(op->getOperand(1), residue, cache);
            if (!a || !b)
                return {};
            if (auto cmp = dyn_cast<arith::CmpIOp>(op)) {
                OrdinalCompare kind;
                switch (cmp.getPredicate()) {
                    case arith::CmpIPredicate::eq:
                        kind = OrdinalCompare::EQ;
                        break;
                    case arith::CmpIPredicate::ne:
                        kind = OrdinalCompare::NE;
                        break;
                    case arith::CmpIPredicate::slt:
                        kind = OrdinalCompare::SLT;
                        break;
                    case arith::CmpIPredicate::sle:
                        kind = OrdinalCompare::SLE;
                        break;
                    case arith::CmpIPredicate::sgt:
                        kind = OrdinalCompare::SGT;
                        break;
                    case arith::CmpIPredicate::sge:
                        kind = OrdinalCompare::SGE;
                        break;
                    case arith::CmpIPredicate::ult:
                        kind = OrdinalCompare::ULT;
                        break;
                    case arith::CmpIPredicate::ule:
                        kind = OrdinalCompare::ULE;
                        break;
                    case arith::CmpIPredicate::ugt:
                        kind = OrdinalCompare::UGT;
                        break;
                    case arith::CmpIPredicate::uge:
                        kind = OrdinalCompare::UGE;
                        break;
                    default:
                        return {};
                }
                return ordinalCompare(kind, *a, *b, cmp.getLhs().getType().isInteger(1));
            }
            if (!value.getType().isInteger(1))
                return {};
            if (isa<arith::AndIOp>(op))
                return bool(*a) && bool(*b);
            if (isa<arith::OrIOp>(op))
                return bool(*a) || bool(*b);
            if (isa<arith::XOrIOp>(op))
                return bool(*a) != bool(*b);
            return {};
        };
        auto answer = compute();
        cache[value] = answer;
        return answer;
    }
};
} // namespace mlir::pto::structured_sync
#endif
