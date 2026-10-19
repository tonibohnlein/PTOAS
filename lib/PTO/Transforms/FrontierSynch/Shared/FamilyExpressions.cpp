// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Preserve exact finite domains as boxes and member maps as verified arithmetic
// or shared decision expressions. No generated guard is inspected or parsed.
#include "PTO/Transforms/FrontierSynch/Shared/EventIdPool.h"
#include "PTO/Transforms/FrontierSynch/Shared/FamilyExpressions.h"
#include "ScalarConstruction.h"
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
struct Domain { SmallVector<Axis> axes; SmallVector<Tuple> points; SmallVector<int64_t> labels; };
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
            auto [entry, inserted] = representatives.try_emplace(key, position, domain.labels[member]);
            if (inserted) {
                continue;
            }
            auto delta = wide(position) - wide(entry->second.first);
            auto value = wide(domain.labels[member]) - wide(entry->second.second);
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
    APInt base = wide(domain.labels.front());
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
        if (value != wide(domain.labels[member])) {
            return false;
        }
    }
    return true;
}
int64_t residue(int64_t value, uint64_t modulus)
{
    const auto divisor = static_cast<int64_t>(modulus);
    const auto remainder = value % divisor;
    return remainder < 0 ? remainder + divisor : remainder;
}
bool fitModular(const Domain& domain, uint64_t modulus, Tuple& coefficients, int64_t& intercept)
{
    coefficients.assign(domain.axes.size(), 0);
    for (std::size_t axis = 0; axis < domain.axes.size(); ++axis) {
        SmallVector<bool> valid(modulus, true);
        std::map<Tuple, std::pair<int64_t, int64_t>> representatives;
        for (std::size_t member = 0; member < domain.points.size(); ++member) {
            Tuple key = domain.points[member];
            const auto position = residue(key[axis], modulus);
            key.erase(key.begin() + axis);
            auto [entry, inserted] = representatives.try_emplace(key, position, domain.labels[member]);
            if (inserted) {
                continue;
            }
            const auto delta = residue(position - entry->second.first, modulus);
            for (uint64_t coefficient = 0; coefficient < modulus; ++coefficient) {
                const auto predicted = residue(entry->second.second +
                    delta * static_cast<int64_t>(coefficient), modulus);
                valid[coefficient] = valid[coefficient] && predicted == domain.labels[member];
            }
        }
        const auto found = std::find(valid.begin(), valid.end(), true);
        if (found == valid.end()) {
            return false;
        }
        coefficients[axis] = static_cast<int64_t>(found - valid.begin());
    }
    intercept = domain.labels.front();
    for (std::size_t axis = 0; axis < coefficients.size(); ++axis) {
        intercept = residue(intercept - residue(domain.points.front()[axis], modulus) * coefficients[axis], modulus);
    }
    // Separately fitting axis differences can leave unconstrained directions;
    // accept only after checking the resulting expression at every exact point.
    for (std::size_t member = 0; member < domain.points.size(); ++member) {
        auto value = intercept;
        for (std::size_t axis = 0; axis < coefficients.size(); ++axis) {
            value = residue(value + residue(domain.points[member][axis], modulus) * coefficients[axis], modulus);
        }
        if (value != domain.labels[member]) {
            return false;
        }
    }
    return true;
}
class Emitter {
public:
    Emitter(detail::ScalarConstruction& construction, ArrayRef<Axis> axes)
        : construction(construction), axes(axes), indices(axes.size()) {}
    detail::ScalarToken number(int64_t value)
    {
        auto found = numbers.find(value);
        if (found != numbers.end()) {
            return found->second;
        }
        return numbers.emplace(value, construction.number(value)).first->second;
    }
    detail::ScalarToken boolean(bool value) { return construction.boolean(value); }
    detail::ScalarToken coordinate(std::size_t index)
    {
        if (indices[index]) { return indices[index]; }
        const auto& axis = axes[index];
        detail::ScalarToken induction = construction.input(axis.induction);
        if (!axis.induction.getType().isIndex()) {
            induction = construction.indexCast(induction);
        }
        if (axis.normalized) {
            detail::ScalarToken value = construction.binary<arith::SubIOp>(induction, number(axis.lower));
            induction = construction.binary<arith::DivUIOp>(value, number(axis.step));
        }
        indices[index] = induction;
        return induction;
    }
    detail::ScalarToken test(std::size_t axis, arith::CmpIPredicate predicate, int64_t bound)
    {
        auto key = std::make_tuple(axis, predicate, bound);
        auto found = comparisons.find(key);
        if (found != comparisons.end()) {
            return found->second;
        }
        detail::ScalarToken result = construction.compare(predicate, coordinate(axis), number(bound));
        comparisons.emplace(key, result);
        return result;
    }
    detail::ScalarToken guard(const Box& box)
    {
        auto key = std::make_pair(box.low, box.high);
        auto found = guards.find(key);
        if (found != guards.end()) {
            return found->second;
        }
        detail::ScalarToken result;
        for (std::size_t axis = 0; axis < indices.size(); ++axis) {
            detail::ScalarToken condition;
            if (box.low[axis] == box.high[axis]) {
                condition = test(axis, arith::CmpIPredicate::eq, box.low[axis]);
            } else {
                auto low = test(axis, arith::CmpIPredicate::sge, box.low[axis]);
                auto high = test(axis, arith::CmpIPredicate::sle, box.high[axis]);
                condition = construction.binary<arith::AndIOp>(low, high);
            }
            if (result) {
                result = construction.binary<arith::AndIOp>(result, condition);
            } else {
                result = condition;
            }
        }
        result = result ? result : boolean(true);
        guards.emplace(std::move(key), result);
        return result;
    }
    detail::ScalarToken affine(ArrayRef<int64_t> coefficients, int64_t intercept)
    {
        detail::ScalarToken result = number(intercept);
        for (std::size_t axis = 0; axis < indices.size(); ++axis) {
            if (coefficients[axis]) {
                auto term = construction.binary<arith::MulIOp>(coordinate(axis), number(coefficients[axis]));
                result = construction.binary<arith::AddIOp>(result, term);
            }
        }
        return result;
    }
    detail::ScalarToken modular(ArrayRef<int64_t> coefficients, int64_t intercept, uint64_t modulus)
    {
        detail::ScalarToken divisor;
        detail::ScalarToken result = number(intercept);
        for (std::size_t axis = 0; axis < indices.size(); ++axis) {
            if (!coefficients[axis]) {
                continue;
            }
            if (!divisor) { divisor = number(static_cast<int64_t>(modulus)); }
            detail::ScalarToken reduced;
            if (axes[axis].normalized) {
                reduced = construction.binary<arith::RemUIOp>(coordinate(axis), divisor);
            } else {
                detail::ScalarToken remainder = construction.binary<arith::RemSIOp>(coordinate(axis), divisor);
                detail::ScalarToken negative = construction.compare(arith::CmpIPredicate::slt, remainder, number(0));
                detail::ScalarToken positive = construction.binary<arith::AddIOp>(remainder, divisor);
                reduced = construction.select(negative, positive, remainder);
            }
            // Reduce coordinates before multiplying. With modulus at most eight,
            // every product and partial sum is bounded independently of indices.
            detail::ScalarToken term = construction.binary<arith::MulIOp>(reduced, number(coefficients[axis]));
            detail::ScalarToken sum = construction.binary<arith::AddIOp>(result, term);
            result = construction.binary<arith::RemUIOp>(sum, divisor);
        }
        return result;
    }
    detail::ScalarToken decision(int64_t fallback, const std::map<int64_t, SmallVector<Box>>& alternatives)
    {
        detail::ScalarToken result = number(fallback);
        for (const auto& alternative : alternatives) {
            result = construction.select(presence(alternative.second),
                number(alternative.first), result);
        }
        return result;
    }
    detail::ScalarToken presence(ArrayRef<Box> boxes)
    {
        detail::ScalarToken result;
        for (const auto& box : boxes) {
            detail::ScalarToken condition = guard(box);
            if (result) {
                result = construction.binary<arith::OrIOp>(result, condition);
            } else {
                result = condition;
            }
        }
        return result ? result : boolean(false);
    }
private:
    detail::ScalarConstruction& construction;
    ArrayRef<Axis> axes;
    SmallVector<detail::ScalarToken> indices;
    std::map<int64_t, detail::ScalarToken> numbers;
    std::map<std::pair<Tuple, Tuple>, detail::ScalarToken> guards;
    std::map<std::tuple<std::size_t, arith::CmpIPredicate, int64_t>, detail::ScalarToken> comparisons;
};
} // namespace
class FamilyExpressionRecipe {
public:
    SmallVector<Axis> axes;
    int64_t fallback = 0;
    SmallVector<Box> regions;
    Tuple coefficients;
    int64_t intercept = 0;
    uint64_t modulus = 0;
    bool affine = false;
    bool includePresence = true;
    std::map<int64_t, SmallVector<Box>> alternatives;
};
FamilyExpressionPlan buildFamilyExpressions(ArrayRef<SmallVector<TemplateCoordinate>> members,
    ArrayRef<int64_t> labels, uint64_t modulus, FamilyExpressionOutputs outputs)
{
    constexpr uint64_t maximumModulus = NativeEventIdCount;
    if (modulus > maximumModulus) {
        return {"family modular labels require a modulus of at most eight", {}};
    }
    if (!labels.empty() && labels.size() != members.size()) {
        return {"family member and label counts differ", {}};
    }
    auto recipe = std::make_shared<FamilyExpressionRecipe>();
    Domain domain;
    if (auto error = validate(members, domain); !error.empty()) { return {std::move(error), {}}; }
    for (std::size_t member = 0; member < members.size(); ++member) {
        const auto label = labels.empty() ? static_cast<int64_t>(member) : labels[member];
        domain.labels.push_back(modulus ? residue(label, modulus) : label);
    }
    recipe->modulus = modulus;
    recipe->affine = !domain.points.empty() &&
        (modulus ? fitModular(domain, modulus, recipe->coefficients, recipe->intercept) :
                   fit(domain, recipe->coefficients, recipe->intercept));
    recipe->includePresence = outputs == FamilyExpressionOutputs::PresenceAndMember;
    if (recipe->includePresence) { recipe->regions = boxes(domain); }
    if (!recipe->affine) {
        std::map<int64_t, Domain> alternatives;
        const auto fallback = domain.labels.empty() ? 0 : domain.labels.front();
        for (std::size_t member = 0; member < domain.points.size(); ++member) {
            const auto label = domain.labels[member];
            if (label == fallback) { continue; }
            auto [entry, inserted] = alternatives.try_emplace(label);
            if (inserted) { entry->second.axes = domain.axes; }
            entry->second.points.push_back(domain.points[member]);
        }
        // Equal outcomes share exact disjoint boxes; no guard-equivalence
        // search is needed. The default remains total outside the domain.
        for (const auto& alternative : alternatives) {
            recipe->alternatives.emplace(alternative.first, boxes(alternative.second));
        }
    }
    recipe->axes = std::move(domain.axes);
    recipe->fallback = domain.labels.empty() ? 0 : domain.labels.front();
    return {{}, std::move(recipe)};
}
void appendFamilyExpressionKey(const FamilyExpressionRecipe& recipe, SmallVectorImpl<uint64_t>& key)
{
    key.push_back(recipe.axes.size());
    for (const auto& axis : recipe.axes) {
        key.push_back(reinterpret_cast<uintptr_t>(axis.induction.getAsOpaquePointer()));
        key.push_back(static_cast<uint64_t>(axis.lower));
        key.push_back(static_cast<uint64_t>(axis.step));
        key.push_back(axis.normalized);
    }
    key.push_back(static_cast<uint64_t>(recipe.fallback));
    key.push_back(recipe.modulus);
    key.push_back(recipe.affine);
    key.push_back(recipe.includePresence);
    key.push_back(static_cast<uint64_t>(recipe.intercept));
    key.push_back(recipe.coefficients.size());
    for (auto coefficient : recipe.coefficients) {
        key.push_back(static_cast<uint64_t>(coefficient));
    }
    auto boxes = [&](ArrayRef<Box> values) {
        key.push_back(values.size());
        for (const auto& box : values) {
            key.push_back(box.low.size());
            for (auto value : box.low) {
                key.push_back(static_cast<uint64_t>(value));
            }
            for (auto value : box.high) {
                key.push_back(static_cast<uint64_t>(value));
            }
        }
    };
    boxes(recipe.regions);
    key.push_back(recipe.alternatives.size());
    for (const auto& [label, regions] : recipe.alternatives) {
        key.push_back(static_cast<uint64_t>(label));
        boxes(regions);
    }
}
std::optional<FamilyExpressionEstimate> estimateFamilyExpressions(const FamilyExpressionRecipe& recipe)
{
    FamilyExpressionEstimate result;
    bool valid = true;
    auto add = [&](uint64_t operations, uint64_t weighted) {
        if (operations > UINT64_MAX - result.operations || weighted > uint64_t(INT64_MAX) - result.weighted) {
            valid = false;
            return;
        }
        result.operations += operations;
        result.weighted += weighted;
    };
    auto remainder = [&](uint64_t modulus) {
        const bool powerOfTwo = modulus && !(modulus & (modulus - 1));
        add(1, powerOfTwo ? 1 : 8);
        result.cheap &= powerOfTwo;
    };
    std::set<std::size_t> coordinates;
    auto coordinate = [&](std::size_t index) {
        if (!coordinates.insert(index).second) {
            return;
        }
        const auto& axis = recipe.axes[index];
        if (!axis.induction.getType().isIndex()) {
            add(1, 1);
        }
        if (axis.normalized) {
            add(2, 2);
            result.cheap = false;
        }
    };
    std::set<std::tuple<std::size_t, arith::CmpIPredicate, int64_t>> comparisons;
    std::set<std::pair<Tuple, Tuple>> guards;
    auto presence = [&](ArrayRef<Box> boxes) {
        for (const auto& box : boxes) {
            if (!guards.emplace(box.low, box.high).second) {
                continue;
            }
            for (std::size_t axis = 0; axis < recipe.axes.size(); ++axis) {
                coordinate(axis);
                const bool point = box.low[axis] == box.high[axis];
                if (comparisons
                        .emplace(axis, point ? arith::CmpIPredicate::eq : arith::CmpIPredicate::sge, box.low[axis])
                        .second) {
                    add(1, 1);
                }
                if (!point) {
                    if (comparisons.emplace(axis, arith::CmpIPredicate::sle, box.high[axis]).second) {
                        add(1, 1);
                    }
                    add(1, 1);
                }
            }
            if (!recipe.axes.empty()) {
                add(recipe.axes.size() - 1, recipe.axes.size() - 1);
            }
        }
        if (!boxes.empty()) {
            add(boxes.size() - 1, boxes.size() - 1);
        }
    };
    if (recipe.includePresence) {
        presence(recipe.regions);
    }
    if (recipe.affine) {
        for (std::size_t axis = 0; axis < recipe.axes.size(); ++axis) {
            if (!recipe.coefficients[axis]) {
                continue;
            }
            coordinate(axis);
            if (!recipe.modulus) {
                add(2, 2);
                continue;
            }
            if (recipe.axes[axis].normalized) {
                remainder(recipe.modulus);
            } else {
                add(4, 6);
                result.cheap = false;
            }
            add(2, 2);
            remainder(recipe.modulus);
        }
    } else {
        for (const auto& [label, boxes] : recipe.alternatives) {
            (void)label;
            presence(boxes);
            add(1, 3);
        }
    }
    result.cheap &= result.operations <= 8;
    return valid ? std::optional(result) : std::nullopt;
}
detail::FamilyScalarTokens detail::constructFamilyExpressions(ScalarConstruction& construction,
    const FamilyExpressionRecipe& recipe)
{
    Emitter emitter(construction, recipe.axes);
    auto present = recipe.includePresence ? emitter.presence(recipe.regions) : ScalarToken{};
    auto member = recipe.affine ?
        (recipe.modulus ? emitter.modular(recipe.coefficients, recipe.intercept, recipe.modulus) :
                         emitter.affine(recipe.coefficients, recipe.intercept)) :
        emitter.decision(recipe.fallback, recipe.alternatives);
    return {present, member};
}
detail::ScalarToken detail::constructFamilyPresence(ScalarConstruction& construction,
    const FamilyExpressionRecipe& recipe)
{
    if (!recipe.includePresence) { return {}; }
    Emitter emitter(construction, recipe.axes);
    return emitter.presence(recipe.regions);
}
Value emitFamilyPresence(OpBuilder& builder, Location location, const FamilyExpressionRecipe& recipe)
{
    detail::ScalarConstruction construction(builder, location);
    return detail::constructFamilyPresence(construction, recipe).value;
}
FamilyExpressions emitFamilyExpressions(OpBuilder& builder, Location location, const FamilyExpressionRecipe& recipe)
{
    detail::ScalarConstruction construction(builder, location);
    auto result = detail::constructFamilyExpressions(construction, recipe);
    return {{}, result.present.value, result.member.value};
}
FamilyExpressions emitFamilyExpressions(OpBuilder& builder, Location location,
    ArrayRef<SmallVector<TemplateCoordinate>> members, ArrayRef<int64_t> labels, uint64_t modulus,
    FamilyExpressionOutputs outputs)
{
    auto plan = buildFamilyExpressions(members, labels, modulus, outputs);
    if (!plan.error.empty()) { return {std::move(plan.error), {}, {}}; }
    return emitFamilyExpressions(builder, location, *plan.recipe);
}
} // namespace mlir::pto::frontiersynch
