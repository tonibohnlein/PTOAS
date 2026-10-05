// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Preserve exact finite domains as boxes and member maps as verified arithmetic
// or shared decision expressions. No generated guard is inspected or parsed.
#include "PTO/Transforms/FrontierSynch/FamilyExpressions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <tuple>
namespace mlir::pto::frontiersynch {
namespace {
using Tuple = SmallVector<int64_t>;
struct Axis { Value induction; int64_t lower = 0; int64_t step = 1; bool normalized = true; };
struct Box { Tuple low, high; };
struct Domain { SmallVector<Axis> axes; SmallVector<Tuple> points; };
std::optional<int64_t> constant(Value value)
{
    APInt number;
    if (!matchPattern(value, m_ConstantInt(&number)) || !number.isSignedIntN(64)) {
        return std::nullopt;
    }
    return number.getSExtValue();
}
APInt wide(int64_t value) { return APInt(128, static_cast<uint64_t>(value), true); }
std::string validate(ArrayRef<SmallVector<TemplateCoordinate>> members, Domain& domain)
{
    if (members.size() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return "family member count exceeds index representation";
    }
    if (members.empty()) {
        return {};
    }
    std::set<Operation*> loops;
    for (auto coordinate : members.front()) {
        auto loop = coordinate.loop;
        if (!loop || !loops.insert(loop.getOperation()).second) {
            return "family coordinates require distinct loops";
        }
        auto lower = constant(loop.getLowerBound()), step = constant(loop.getStep());
        auto type = loop.getInductionVar().getType();
        auto integer = dyn_cast<IntegerType>(type);
        if ((step && *step <= 0) || (!type.isIndex() && (!integer || integer.getWidth() > 64))) {
            return "family coordinates require positive steps with at most 64-bit indices";
        }
        // Numerical expansion may evaluate nested bounds from enclosing indices.
        // Retain raw induction coordinates when no common literal lattice exists.
        const bool normalized = lower && step;
        domain.axes.push_back({loop.getInductionVar(), normalized ? *lower : 0,
                               normalized ? *step : 1, normalized});
    }
    std::set<Tuple> seen;
    for (const auto& member : members) {
        if (member.size() != domain.axes.size()) {
            return "family members have different coordinate dimensions";
        }
        Tuple point;
        for (std::size_t axis = 0; axis < member.size(); ++axis) {
            if (member[axis].loop != members.front()[axis].loop) {
                return "family members have different loop schemas";
            }
            const auto& spec = domain.axes[axis];
            auto type = dyn_cast<IntegerType>(spec.induction.getType());
            APInt offset = wide(member[axis].induction) - wide(spec.lower);
            if ((type && !wide(member[axis].induction).isSignedIntN(type.getWidth())) ||
                (spec.normalized && offset.isNegative()) ||
                !offset.srem(wide(spec.step)).isZero() || !offset.sdiv(wide(spec.step)).isSignedIntN(64)) {
                return "family member is outside the representable iteration lattice";
            }
            point.push_back(offset.sdiv(wide(spec.step)).getSExtValue());
        }
        if (!seen.insert(point).second) {
            return "family member coordinates must be unique";
        }
        domain.points.push_back(std::move(point));
    }
    return {};
}
SmallVector<Box> boxes(const Domain& domain)
{
    SmallVector<Box> result;
    for (const auto& point : domain.points) {
        result.push_back({point, point});
    }
    for (std::size_t axis = 0; axis < domain.axes.size(); ++axis) {
        llvm::sort(result, [axis](const Box& a, const Box& b) {
            for (std::size_t j = 0; j < a.low.size(); ++j) {
                if (j != axis && std::tie(a.low[j], a.high[j]) != std::tie(b.low[j], b.high[j])) {
                    return std::tie(a.low[j], a.high[j]) < std::tie(b.low[j], b.high[j]);
                }
            }
            return a.low[axis] < b.low[axis];
        });
        SmallVector<Box> merged;
        for (const auto& box : result) {
            bool adjacent = !merged.empty() && merged.back().high[axis] < std::numeric_limits<int64_t>::max() &&
                merged.back().high[axis] + 1 == box.low[axis];
            for (std::size_t j = 0; adjacent && j < domain.axes.size(); ++j) {
                adjacent &= j == axis || (merged.back().low[j] == box.low[j] && merged.back().high[j] == box.high[j]);
            }
            if (adjacent) {
                merged.back().high[axis] = box.high[axis];
            } else {
                merged.push_back(box);
            }
        }
        result = std::move(merged);
    }
    return result;
}
bool fit(const Domain& domain, Tuple& coefficients, int64_t& intercept)
{
    coefficients.assign(domain.axes.size(), 0);
    for (std::size_t axis = 0; axis < domain.axes.size(); ++axis) {
        std::map<Tuple, std::pair<int64_t, int64_t>> representatives;
        std::optional<int64_t> coefficient;
        for (std::size_t member = 0; member < domain.points.size(); ++member) {
            Tuple key = domain.points[member];
            const auto position = key[axis];
            key.erase(key.begin() + axis);
            auto [entry, inserted] = representatives.try_emplace(key, position, static_cast<int64_t>(member));
            if (inserted) {
                continue;
            }
            auto delta = wide(position) - wide(entry->second.first);
            auto value = wide(static_cast<int64_t>(member)) - wide(entry->second.second);
            if (delta.isZero() || !value.srem(delta).isZero() || !value.sdiv(delta).isSignedIntN(64)) {
                return false;
            }
            auto candidate = value.sdiv(delta).getSExtValue();
            if (coefficient && *coefficient != candidate) {
                return false;
            }
            coefficient = candidate;
        }
        coefficients[axis] = coefficient.value_or(0);
    }
    APInt base = wide(0);
    for (std::size_t axis = 0; axis < coefficients.size(); ++axis) {
        // Products fit in signed 128 bits; refuse an overflowing accumulated offset.
        bool overflow = false;
        base = base.ssub_ov(wide(domain.points.front()[axis]) * wide(coefficients[axis]), overflow);
        if (overflow) {
            return false;
        }
    }
    if (!base.isSignedIntN(64)) {
        return false;
    }
    intercept = base.getSExtValue();
    for (std::size_t member = 0; member < domain.points.size(); ++member) {
        APInt value = base;
        for (std::size_t axis = 0; axis < coefficients.size(); ++axis) {
            bool overflow = false;
            auto term = wide(domain.points[member][axis]) * wide(coefficients[axis]);
            value = value.sadd_ov(term, overflow);
            if (overflow || !term.isSignedIntN(64) || !value.isSignedIntN(64)) {
                return false;
            }
        }
        if (value != wide(static_cast<int64_t>(member))) {
            return false;
        }
    }
    return true;
}
class Emitter {
public:
    Emitter(OpBuilder& builder, Location location) : builder(builder), location(location) {}
    Value number(int64_t value)
    {
        auto found = numbers.find(value);
        if (found != numbers.end()) {
            return found->second;
        }
        return numbers.emplace(value, builder.create<arith::ConstantIndexOp>(location, value)).first->second;
    }
    Value boolean(bool value) { return builder.create<arith::ConstantIntOp>(location, value, 1); }
    void normalize(ArrayRef<Axis> axes)
    {
        for (const auto& axis : axes) {
            Value induction = axis.induction;
            if (!induction.getType().isIndex()) {
                induction = builder.create<arith::IndexCastOp>(location, builder.getIndexType(), induction);
            }
            if (axis.normalized) {
                Value value = builder.create<arith::SubIOp>(location, induction, number(axis.lower));
                indices.push_back(builder.create<arith::DivUIOp>(location, value, number(axis.step)));
            } else {
                indices.push_back(induction);
            }
        }
    }
    Value test(std::size_t axis, arith::CmpIPredicate predicate, int64_t bound)
    {
        auto key = std::make_tuple(axis, predicate, bound);
        auto found = comparisons.find(key);
        if (found != comparisons.end()) {
            return found->second;
        }
        Value result = builder.create<arith::CmpIOp>(location, predicate, indices[axis], number(bound));
        comparisons.emplace(key, result);
        return result;
    }
    Value guard(const Box& box)
    {
        auto key = std::make_pair(box.low, box.high);
        auto found = guards.find(key);
        if (found != guards.end()) {
            return found->second;
        }
        Value result;
        for (std::size_t axis = 0; axis < indices.size(); ++axis) {
            Value condition;
            if (box.low[axis] == box.high[axis]) {
                condition = test(axis, arith::CmpIPredicate::eq, box.low[axis]);
            } else {
                auto low = test(axis, arith::CmpIPredicate::sge, box.low[axis]);
                auto high = test(axis, arith::CmpIPredicate::sle, box.high[axis]);
                condition = builder.create<arith::AndIOp>(location, low, high);
            }
            if (result) {
                result = builder.create<arith::AndIOp>(location, result, condition);
            } else {
                result = condition;
            }
        }
        result = result ? result : boolean(true);
        guards.emplace(std::move(key), result);
        return result;
    }
    Value affine(ArrayRef<int64_t> coefficients, int64_t intercept)
    {
        Value result = number(intercept);
        for (std::size_t axis = 0; axis < indices.size(); ++axis) {
            if (coefficients[axis]) {
                Value term = builder.create<arith::MulIOp>(location, indices[axis], number(coefficients[axis]));
                result = builder.create<arith::AddIOp>(location, result, term);
            }
        }
        return result;
    }
    Value decision(ArrayRef<Tuple> points)
    {
        Value result = number(0);
        for (std::size_t member = 1; member < points.size(); ++member) {
            result = builder.create<arith::SelectOp>(location, guard({points[member], points[member]}),
                number(static_cast<int64_t>(member)), result);
        }
        return result;
    }
    Value presence(ArrayRef<Box> boxes)
    {
        Value result;
        for (const auto& box : boxes) {
            Value condition = guard(box);
            if (result) {
                result = builder.create<arith::OrIOp>(location, result, condition);
            } else {
                result = condition;
            }
        }
        return result ? result : boolean(false);
    }
private:
    OpBuilder& builder;
    Location location;
    SmallVector<Value> indices;
    std::map<int64_t, Value> numbers;
    std::map<std::pair<Tuple, Tuple>, Value> guards;
    std::map<std::tuple<std::size_t, arith::CmpIPredicate, int64_t>, Value> comparisons;
};
} // namespace
FamilyExpressions emitFamilyExpressions(OpBuilder& builder, Location location,
    ArrayRef<SmallVector<TemplateCoordinate>> members)
{
    Domain domain;
    if (auto error = validate(members, domain); !error.empty()) {
        return {std::move(error), {}, {}};
    }
    Tuple coefficients;
    int64_t intercept = 0;
    const bool affine = !domain.points.empty() && fit(domain, coefficients, intercept);
    auto regions = boxes(domain);
    Emitter emitter(builder, location);
    emitter.normalize(domain.axes);
    Value present = emitter.presence(regions);
    Value member = affine ? emitter.affine(coefficients, intercept) : emitter.decision(domain.points);
    return {{}, present, member};
}
} // namespace mlir::pto::frontiersynch
