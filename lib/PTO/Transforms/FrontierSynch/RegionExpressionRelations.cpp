// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/AnalysisCost.h"
#include "RegionIntegerExpressionsInternal.h"
#include "mlir/IR/BuiltinTypes.h"
#include <map>
#include <numeric>
#include <set>
namespace mlir::pto::frontiersynch {
namespace {
using Integer = BoundInteger;
struct Linear {
    std::map<unsigned, Integer> coefficients;
    Integer constant{0};
};
Linear addLinear(Linear a, const Linear& b, const Integer& scale = Integer(1))
{
    a.constant += scale * b.constant;
    for (const auto& [column, coefficient] : b.coefficients) {
        a.coefficients[column] += scale * coefficient;
        if (a.coefficients[column] == 0) { a.coefficients.erase(column); }
    }
    return a;
}
Linear scaled(Linear a, const Integer& scale)
{
    a.constant *= scale;
    for (auto& [column, coefficient] : a.coefficients) { coefficient *= scale; }
    return a;
}
Linear literal(const Integer& value) { Linear result; result.constant = value; return result; }
struct Atom {
    Linear expression;
    Integer bound{0};
    // Zero denotes <=bound; positive denotes ==bound modulo modulus.
    Integer modulus{0};
};
using Clause = std::vector<Atom>;
using Formula = std::vector<Clause>;
struct ValueForm {
    Linear expression;
    Integer low{0}, high{0};
};
Integer unsignedInteger(uint64_t value)
{
    return Integer(static_cast<int64_t>(value >> 1)) * Integer(2) + Integer(static_cast<int64_t>(value & 1));
}
Integer bitsLimit() { return Integer(int64_t(1) << 62) * Integer(4); }
// Reject inconsistent alternatives before distributing further definitions.
// This is sound interval propagation, not a complete integer-feasibility test.
// Keeping the undetermined clauses preserves the exact relation.
bool simplify(Clause& clause)
{
    std::map<std::map<unsigned, Integer>, Integer> tight;
    Clause congruences;
    for (const auto& atom : clause) {
        const auto bound = atom.bound - atom.expression.constant;
        if (atom.expression.coefficients.empty()) {
            if (atom.modulus == 0 ? bound < 0 : mod(bound, atom.modulus) != 0) { return false; }
            continue;
        }
        if (atom.modulus != 0) { congruences.push_back(atom); continue; }
        auto inserted = tight.emplace(atom.expression.coefficients, bound);
        if (!inserted.second) { inserted.first->second = std::min(inserted.first->second, bound); }
    }
    using Bounds = std::pair<std::optional<Integer>, std::optional<Integer>>;
    std::map<unsigned, Bounds> ranges;
    for (const auto& [coefficients, bound] : tight) {
        auto opposite = coefficients;
        for (auto& [column, coefficient] : opposite) { (void)column; coefficient = -coefficient; }
        auto found = tight.find(opposite);
        if (found != tight.end() && bound + found->second < 0) { return false; }
        for (const auto& [column, coefficient] : coefficients) { (void)coefficient; ranges.try_emplace(column); }
    }
    for (unsigned pass = 0; pass <= ranges.size(); ++pass) {
        bool changed = false;
        for (const auto& [coefficients, bound] : tight) {
            for (const auto& [selected, factor] : coefficients) {
                Integer other(0);
                bool known = true;
                for (const auto& [column, coefficient] : coefficients) {
                    if (column == selected) { continue; }
                    const auto& range = ranges[column];
                    const auto& value = coefficient > 0 ? range.first : range.second;
                    if (!value) { known = false; break; }
                    other += coefficient * *value;
                }
                if (!known) { continue; }
                auto& range = ranges[selected];
                if (factor > 0) {
                    auto high = floorDiv(bound - other, factor);
                    if (!range.second || high < *range.second) { range.second = high; changed = true; }
                } else {
                    auto low = -floorDiv(bound - other, -factor);
                    if (!range.first || low > *range.first) { range.first = low; changed = true; }
                }
                if (range.first && range.second && *range.first > *range.second) { return false; }
            }
        }
        if (!changed) { break; }
    }
    clause = std::move(congruences);
    for (const auto& [coefficients, bound] : tight) {
        clause.push_back({Linear{coefficients, Integer(0)}, bound});
    }
    return true;
}
Formula conjoin(const Formula& a, const Formula& b)
{
    Formula result;
    for (const auto& x : a) {
        for (const auto& y : b) {
            auto both = x;
            both.insert(both.end(), y.begin(), y.end());
            if (simplify(both)) { result.push_back(std::move(both)); }
        }
    }
    return result;
}
void append(Formula& to, Formula from)
{
    to.insert(to.end(), std::make_move_iterator(from.begin()), std::make_move_iterator(from.end()));
}
} // namespace
// Definitions are shared across all Boolean alternatives. Base expressions are
// total bit-vector functions, so constructing an unselected branch is sound.
class RegionExpressions::RelationBuilder {
public:
    RelationBuilder(const RegionExpressions& arena, ArrayRef<Id> variables, const IntegerSystem& domain,
                    std::string& diagnostic, RelationCost* cost)
        : arena(arena), variables(variables), domain(domain), diagnostic(diagnostic), cost(cost),
          columns(variables.size()), modulus(bitsLimit()), sign(modulus / Integer(2)) {}
    FailureOr<std::vector<IntegerSystem>> run(Id predicate);
private:
    const RegionExpressions& arena;
    ArrayRef<Id> variables;
    const IntegerSystem& domain;
    std::string& diagnostic;
    RelationCost* cost;
    unsigned columns;
    Integer modulus, sign;
    Formula definitions{{}};
    std::map<Id, ValueForm> values, signedValues, inputs;
    std::map<unsigned, std::pair<Integer, Integer>> columnBounds;
    std::map<std::pair<Id, bool>, Formula> predicates;
    bool failed = false;
    void reject(StringRef message)
    {
        if (!failed) { diagnostic = message.str(); }
        failed = true;
    }
    Linear column()
    {
        Linear result;
        if (columns == UINT_MAX) { reject("regional relation dimension exceeds representation"); return result; }
        result.coefficients.emplace(columns++, Integer(1));
        return result;
    }
    Formula conjunction(const Formula& a, const Formula& b)
    {
        if (!a.empty() && b.size() > UINT64_MAX / a.size()) {
            reject("regional relation formula size exceeds representation"); return {};
        }
        auto result = conjoin(a, b);
        if (cost) {
            accumulateCost(cost->formulaProducts, a.size() * b.size());
            cost->peakClauses = std::max<uint64_t>(cost->peakClauses, result.size());
        }
        return result;
    }
    void require(Formula formula) { definitions = conjunction(definitions, formula); }
    Formula equality(const Linear& a, const Linear& b)
    {
        auto difference = addLinear(a, b, Integer(-1));
        return {{Atom{difference, Integer(0)}, Atom{scaled(difference, Integer(-1)), Integer(0)}}};
    }
    ValueForm bounded(Linear expression, Integer low, Integer high)
    {
        if (expression.coefficients.size() == 1 && expression.coefficients.begin()->second == 1) {
            auto bounds = std::make_pair(low - expression.constant, high - expression.constant);
            auto inserted = columnBounds.emplace(expression.coefficients.begin()->first, bounds);
            if (!inserted.second) {
                inserted.first->second.first = std::max(inserted.first->second.first, bounds.first);
                inserted.first->second.second = std::min(inserted.first->second.second, bounds.second);
            }
        }
        require({{Atom{expression, high}, Atom{scaled(expression, Integer(-1)), -low}}});
        return {std::move(expression), std::move(low), std::move(high)};
    }
    ValueForm refine(ValueForm form)
    {
        Integer low = form.expression.constant, high = form.expression.constant;
        for (const auto& [column, coefficient] : form.expression.coefficients) {
            auto bounds = columnBounds.find(column);
            if (bounds == columnBounds.end()) { return form; }
            low += coefficient * (coefficient > 0 ? bounds->second.first : bounds->second.second);
            high += coefficient * (coefficient > 0 ? bounds->second.second : bounds->second.first);
        }
        form.low = std::max(form.low, low); form.high = std::min(form.high, high);
        if (form.low == form.high) { form.expression = literal(form.low); }
        return form;
    }
    ValueForm wrap(ValueForm original)
    {
        original = refine(std::move(original));
        if (original.low >= 0 && original.high < modulus) { return original; }
        const auto first = floorDiv(original.low, modulus), last = floorDiv(original.high, modulus);
        if (first == last) {
            original.expression.constant -= first * modulus;
            original.low -= first * modulus; original.high -= first * modulus;
            return original;
        }
        auto result = bounded(column(), Integer(0), modulus - 1);
        if (last - first <= 1) {
            auto one = original.expression; one.constant -= first * modulus;
            auto two = original.expression; two.constant -= last * modulus;
            auto alternatives = equality(result.expression, one);
            append(alternatives, equality(result.expression, two));
            require(std::move(alternatives));
        } else {
            auto turns = quotient(original, modulus);
            require(equality(result.expression, addLinear(original.expression, turns.expression, -modulus)));
        }
        return result;
    }
    ValueForm signedValue(ValueForm bits)
    {
        bits = refine(std::move(bits));
        if (bits.high < sign) { return bits; }
        if (bits.low >= sign) {
            bits.expression.constant -= modulus;
            bits.low -= modulus; bits.high -= modulus;
            return bits;
        }
        auto result = bounded(column(), -sign, sign - 1);
        auto alternatives = equality(result.expression, bits.expression);
        auto negative = bits.expression; negative.constant -= modulus;
        append(alternatives, equality(result.expression, negative));
        require(std::move(alternatives));
        return result;
    }
    ValueForm signedExpression(Id id)
    {
        auto cached = signedValues.find(id);
        if (cached != signedValues.end()) { return cached->second; }
        auto input = inputs.find(id);
        auto result = input == inputs.end() ? signedValue(value(id)) : input->second;
        signedValues.emplace(id, result);
        return result;
    }
    ValueForm quotient(ValueForm value, const Integer& divisor)
    {
        value = refine(std::move(value));
        if (divisor == 1) { return value; }
        if (divisor <= 0) { reject("regional relation divisor must be positive"); return {}; }
        const auto low = floorDiv(value.low, divisor), high = floorDiv(value.high, divisor);
        if (low == high) { return {literal(low), low, high}; }
        auto result = bounded(column(), low, high);
        auto difference = addLinear(scaled(result.expression, divisor), value.expression, Integer(-1));
        require({{Atom{difference, Integer(0)}, Atom{scaled(difference, Integer(-1)), divisor - 1}}});
        return result;
    }
    Formula comparison(ValueForm a, ValueForm b, bool strict)
    {
        if (a.high < b.low || (!strict && a.high <= b.low)) { return {{}}; }
        if (a.low > b.high || (strict && a.low >= b.high)) { return {}; }
        auto difference = addLinear(a.expression, b.expression, Integer(-1));
        Integer low = difference.constant, high = difference.constant;
        bool known = true;
        for (const auto& [column, coefficient] : difference.coefficients) {
            auto bounds = columnBounds.find(column);
            if (bounds == columnBounds.end()) { known = false; break; }
            low += coefficient * (coefficient > 0 ? bounds->second.first : bounds->second.second);
            high += coefficient * (coefficient > 0 ? bounds->second.second : bounds->second.first);
        }
        const Integer limit(strict ? -1 : 0);
        if (known && high <= limit) { return {{}}; }
        if (known && low > limit) { return {}; }
        return {{Atom{difference, limit}}};
    }
    Formula atom(Atom condition, bool truth)
    {
        if (truth) { return {{std::move(condition)}}; }
        if (condition.modulus == 0) {
            return {{Atom{scaled(condition.expression, Integer(-1)), -condition.bound - 1}}};
        }
        // Complement a congruence without enumerating its residue classes.
        auto remainder = bounded(column(), Integer(0), condition.modulus - 1);
        require({{Atom{addLinear(remainder.expression, condition.expression, Integer(-1)),
                       Integer(0), condition.modulus}}});
        const auto target = mod(condition.bound, condition.modulus);
        return {{Atom{remainder.expression, target - 1}},
                {Atom{scaled(remainder.expression, Integer(-1)), -target - 1}}};
    }
    std::vector<ValueForm> integerInputs(const IntegerRecipe& recipe)
    {
        std::vector<ValueForm> result;
        for (auto input : recipe.inputs) {
            result.push_back(quotient(signedExpression(input), Integer(recipe.period)));
        }
        return result;
    }
    ValueForm affine(const IntegerAffine& expression, ArrayRef<ValueForm> inputs)
    {
        ValueForm result{literal(expression.constant), expression.constant, expression.constant};
        for (auto [coefficient, input] : llvm::zip(expression.coefficients, inputs)) {
            result.expression = addLinear(result.expression, input.expression, coefficient);
            result.low += coefficient * (coefficient >= 0 ? input.low : input.high);
            result.high += coefficient * (coefficient >= 0 ? input.high : input.low);
        }
        return result;
    }
    Formula integerPredicate(const IntegerRecipe& recipe, bool truth)
    {
        if (!recipe.coordinates.empty()) {
            reject("relational lowering for mapped integer predicates not implemented yet"); return {};
        }
        auto inputs = integerInputs(recipe);
        Clause conditions;
        for (auto [input, residue] : llvm::zip(recipe.inputs, recipe.residues)) {
            conditions.push_back({signedExpression(input).expression, Integer(residue), Integer(recipe.period)});
        }
        for (const auto& row : recipe.system.constraints()) {
            conditions.push_back({affine({row.coefficients, Integer(0)}, inputs).expression, row.bound});
        }
        for (const auto& row : recipe.system.congruences()) {
            conditions.push_back({affine({row.coefficients, Integer(0)}, inputs).expression,
                                  row.residue, row.modulus});
        }
        if (truth) { return {std::move(conditions)}; }
        Formula result;
        for (auto& condition : conditions) { append(result, atom(std::move(condition), false)); }
        return result;
    }
    ValueForm value(Id id);
    Formula predicate(Id id, bool truth);
};
ValueForm RegionExpressions::RelationBuilder::value(Id id)
{
    auto found = values.find(id);
    if (found != values.end()) { return found->second; }
    if (failed || id >= arena.nodes.size()) { reject("invalid regional relation expression"); return {}; }
    if (cost) { ++cost->gates; }
    const auto& node = arena.nodes[id];
    ValueForm result;
    if (node.kind == Kind::Constant) {
        const Integer number = unsignedInteger(node.literal);
        result = {literal(number), number, number};
    } else if (node.kind == Kind::Input) {
        auto input = inputs.find(id);
        if (input == inputs.end()) { reject("regional relation has an unbound input"); return {}; }
        result = node.boolean ? input->second : wrap(input->second);
    } else if (node.boolean) {
        result = bounded(column(), Integer(0), Integer(1));
        auto yes = conjunction(predicate(id, true), equality(result.expression, literal(Integer(1))));
        auto no = conjunction(predicate(id, false), equality(result.expression, literal(Integer(0))));
        append(yes, std::move(no)); require(std::move(yes));
    } else if (node.kind == Kind::Add || node.kind == Kind::Sub) {
        auto a = value(node.a), b = value(node.b);
        const bool plus = node.kind == Kind::Add;
        result = wrap({addLinear(a.expression, b.expression, Integer(plus ? 1 : -1)),
                       plus ? a.low + b.low : a.low - b.high,
                       plus ? a.high + b.high : a.high - b.low});
    } else if (node.kind == Kind::Div || node.kind == Kind::Rem) {
        auto denominator = arena.constantValue(node.b);
        if (!denominator || !*denominator) {
            reject("regional relation requires a constant positive divisor"); return {};
        }
        auto source = value(node.a);
        result = quotient(source, unsignedInteger(*denominator));
        if (node.kind == Kind::Rem) {
            result = {addLinear(source.expression, result.expression, -unsignedInteger(*denominator)),
                      Integer(0), unsignedInteger(*denominator) - 1};
        }
    } else if (node.kind == Kind::Select) {
        auto condition = predicate(node.a, true);
        if (condition.empty()) { result = value(node.c); }
        else if (condition.size() == 1 && condition.front().empty()) { result = value(node.b); }
        else {
            auto yes = value(node.b), no = value(node.c);
            auto low = std::min(yes.low, no.low), high = std::max(yes.high, no.high);
            const auto& test = arena.nodes[node.a];
            // The quotient's min-plus circuit must keep its certified small
            // distance range; union bounds would grow through every minimum.
            const bool minimum = (test.kind == Kind::Lt || test.kind == Kind::Le) &&
                test.a == node.b && test.b == node.c;
            auto yesWins = minimum ? comparison(yes, no, false) : Formula{};
            auto noWins = minimum ? comparison(no, yes, false) : Formula{};
            if (minimum) { high = std::min(yes.high, no.high); }
            if (minimum && yesWins.size() == 1 && yesWins.front().empty()) { result = yes; }
            else if (minimum && noWins.size() == 1 && noWins.front().empty()) { result = no; }
            else if (low == high) { result = {literal(low), low, high}; }
            else {
                result = bounded(column(), low, high);
                auto yesBranch = conjunction(condition, equality(result.expression, yes.expression));
                auto noBranch = conjunction(predicate(node.a, false), equality(result.expression, no.expression));
                append(yesBranch, std::move(noBranch)); require(std::move(yesBranch));
            }
        }
    } else if (node.kind == Kind::Integer && node.integer) {
        const auto& recipe = *node.integer;
        auto inputs = integerInputs(recipe);
        auto divided = quotient(affine(recipe.numerator, inputs), recipe.denominator);
        const auto offset = literal(Integer(recipe.outputResidue));
        result = wrap({addLinear(scaled(divided.expression, Integer(recipe.period)), offset),
                       divided.low * Integer(recipe.period) + Integer(recipe.outputResidue),
                       divided.high * Integer(recipe.period) + Integer(recipe.outputResidue)});
    } else {
        reject("unsupported regional relation value expression"); return {};
    }
    values.emplace(id, result);
    return result;
}
Formula RegionExpressions::RelationBuilder::predicate(Id id, bool truth)
{
    const auto key = std::make_pair(id, truth);
    auto found = predicates.find(key);
    if (found != predicates.end()) { return found->second; }
    if (failed || id >= arena.nodes.size() || !arena.nodes[id].boolean) {
        reject("regional relation requires a Boolean predicate"); return {};
    }
    if (cost) { ++cost->gates; }
    const auto& node = arena.nodes[id];
    Formula result;
    if (node.kind == Kind::Constant) {
        if (bool(node.literal) == truth) { result.emplace_back(); }
    } else if (node.kind == Kind::Input) {
        result = equality(value(id).expression, literal(Integer(truth ? 1 : 0)));
    } else if (node.kind == Kind::Not) {
        result = predicate(node.a, !truth);
    } else if (node.kind == Kind::And || node.kind == Kind::Or) {
        auto a = predicate(node.a, truth);
        const bool intersect = (node.kind == Kind::And) == truth;
        if (intersect && a.empty()) { result = {}; }
        else if (!intersect && a.size() == 1 && a.front().empty()) { result = {{}}; }
        else {
            auto b = predicate(node.b, truth);
            if (intersect) { result = conjunction(a, b); }
            else { result = std::move(a); append(result, std::move(b)); }
        }
    } else if (node.kind == Kind::Select) {
        auto condition = predicate(node.a, true);
        if (condition.empty()) { result = predicate(node.c, truth); }
        else if (condition.size() == 1 && condition.front().empty()) { result = predicate(node.b, truth); }
        else {
            result = conjunction(condition, predicate(node.b, truth));
            append(result, conjunction(predicate(node.a, false), predicate(node.c, truth)));
        }
    } else if (node.kind == Kind::Integer && node.integer) {
        result = integerPredicate(*node.integer, truth);
    } else {
        const bool isSigned = node.kind == Kind::SLt || node.kind == Kind::SLe;
        auto a = isSigned ? signedExpression(node.a) : value(node.a);
        auto b = isSigned ? signedExpression(node.b) : value(node.b);
        if (node.kind == Kind::Eq) {
            result = truth ? equality(a.expression, b.expression) : comparison(a, b, true);
            if (!truth) { append(result, comparison(b, a, true)); }
        } else if (node.kind == Kind::Lt || node.kind == Kind::SLt ||
                   node.kind == Kind::Le || node.kind == Kind::SLe) {
            const bool strict = node.kind == Kind::Lt || node.kind == Kind::SLt;
            result = truth ? comparison(a, b, strict) : comparison(b, a, !strict);
        } else { reject("unsupported regional relation Boolean expression"); }
    }
    predicates.emplace(key, result);
    return result;
}
FailureOr<std::vector<IntegerSystem>> RegionExpressions::RelationBuilder::run(Id root)
{
    diagnostic.clear();
    if (domain.dimensions() != variables.size() || variables.size() > UINT_MAX) {
        diagnostic = "regional relation domain dimensions differ"; return failure();
    }
    if (root >= arena.nodes.size() || !arena.nodes[root].boolean) {
        diagnostic = "regional relation requires a Boolean predicate"; return failure();
    }
    for (auto [id, input] : arena.referencedInputs(root)) {
        (void)input;
        if (!llvm::is_contained(variables, id)) {
            diagnostic = "regional relation has an unbound input"; return failure();
        }
    }
    if (cost) { *cost = {}; }
    for (unsigned i = 0; i < variables.size(); ++i) {
        const auto id = variables[i];
        if (id >= arena.nodes.size() || arena.nodes[id].kind != Kind::Input || inputs.count(id)) {
            diagnostic = "regional relation variables must be distinct Input expressions"; return failure();
        }
        const bool boolean = arena.nodes[id].boolean;
        Integer low = boolean ? Integer(0) : -sign, high = boolean ? Integer(1) : sign - 1;
        if (auto type = dyn_cast<IntegerType>(arena.nodes[id].value.getType())) {
            if (type.getWidth() < 64) {
                low = Integer(0);
                high = unsignedInteger((uint64_t(1) << type.getWidth()) - 1);
            }
        }
        for (const auto& row : domain.constraints()) {
            if (row.coefficients[i] == 0) { continue; }
            bool single = true;
            for (unsigned j = 0; j < variables.size(); ++j) {
                if (j != i && row.coefficients[j] != 0) { single = false; break; }
            }
            if (!single) { continue; }
            const auto coefficient = row.coefficients[i];
            if (coefficient > 0) { high = std::min(high, floorDiv(row.bound, coefficient)); }
            else { low = std::max(low, -floorDiv(row.bound, -coefficient)); }
        }
        Linear external; external.coefficients.emplace(i, Integer(1));
        auto signedInput = bounded(external, low, high);
        inputs.emplace(id, std::move(signedInput));
    }
    // Presence/domain conjunctions often prove nonnegative counted coordinates.
    // Propagate only unavoidable signed comparisons between original columns
    // and constants; no fact is taken from one arm of a disjunction.
    struct BoundEdge { Id a, b; bool strict; };
    std::vector<BoundEdge> edges;
    std::set<std::pair<Id, bool>> observed;
    std::function<void(Id, bool)> observe = [&](Id id, bool truth) {
        if (id >= arena.nodes.size() || !observed.insert({id, truth}).second) { return; }
        const auto& node = arena.nodes[id];
        if (node.kind == Kind::Not) { observe(node.a, !truth); }
        else if ((node.kind == Kind::And && truth) || (node.kind == Kind::Or && !truth)) {
            observe(node.a, truth); observe(node.b, truth);
        } else if (node.kind == Kind::SLt || node.kind == Kind::SLe) {
            if (truth) { edges.push_back({node.a, node.b, node.kind == Kind::SLt}); }
            else { edges.push_back({node.b, node.a, node.kind != Kind::SLt}); }
        } else if (node.kind == Kind::Eq && truth) {
            edges.push_back({node.a, node.b, false}); edges.push_back({node.b, node.a, false});
        }
    };
    observe(root, true);
    auto bound = [&](Id id) -> std::optional<std::pair<Integer, Integer>> {
        auto found = inputs.find(id);
        if (found != inputs.end()) { return std::make_pair(found->second.low, found->second.high); }
        auto constant = arena.constantValue(id);
        if (!constant) { return std::nullopt; }
        auto number = unsignedInteger(*constant);
        if (number >= sign) { number -= modulus; }
        return std::make_pair(number, number);
    };
    for (unsigned pass = 0; pass <= variables.size(); ++pass) {
        bool changed = false;
        for (const auto& edge : edges) {
            auto a = bound(edge.a), b = bound(edge.b);
            if (!a || !b) { continue; }
            auto left = inputs.find(edge.a), right = inputs.find(edge.b);
            const Integer gap(edge.strict ? 1 : 0);
            if (left != inputs.end() && b->second - gap < left->second.high) {
                left->second.high = b->second - gap; changed = true;
            }
            if (right != inputs.end() && a->first + gap > right->second.low) {
                right->second.low = a->first + gap; changed = true;
            }
        }
        if (!changed) { break; }
    }
    for (const auto& [id, input] : inputs) {
        (void)id;
        if (input.low > input.high) { return std::vector<IntegerSystem>{}; }
        bounded(input.expression, input.low, input.high);
    }
    auto condition = predicate(root, true);
    if (failed) { return failure(); }
    auto formulas = conjunction(definitions, condition);
    if (failed) { return failure(); }
    std::vector<unsigned> keep(variables.size());
    std::iota(keep.begin(), keep.end(), 0);
    std::vector<unsigned> domainMap(variables.size());
    std::iota(domainMap.begin(), domainMap.end(), 0);
    auto liftedDomain = domain.remap(columns, domainMap);
    if (mlir::failed(liftedDomain)) { diagnostic = "regional relation domain remap failed"; return failure(); }
    std::vector<IntegerSystem> result;
    for (const auto& clause : formulas) {
        std::vector<IntegerConstraint> rows;
        std::vector<IntegerCongruence> congruences;
        for (const auto& atom : clause) {
            std::vector<Integer> coefficients(columns);
            for (const auto& [column, coefficient] : atom.expression.coefficients) {
                coefficients[column] = coefficient;
            }
            if (atom.modulus == 0) { rows.push_back({std::move(coefficients), atom.bound - atom.expression.constant}); }
            else {
                congruences.push_back({std::move(coefficients), atom.bound - atom.expression.constant, atom.modulus});
            }
        }
        auto system = IntegerSystem::create(columns, rows, congruences);
        if (mlir::failed(system)) { diagnostic = "regional relation constraints failed"; return failure(); }
        auto restricted = system->intersect(*liftedDomain);
        if (mlir::failed(restricted)) { diagnostic = "regional relation intersection failed"; return failure(); }
        if (restricted->isKnownEmpty()) { continue; }
        if (cost) { ++cost->projections; }
        auto pieces = restricted->project(keep);
        if (mlir::failed(pieces)) { diagnostic = "regional relation projection unavailable"; return failure(); }
        for (auto& piece : *pieces) {
            if (!piece.isKnownEmpty() && !llvm::is_contained(result, piece)) { result.push_back(std::move(piece)); }
        }
    }
    if (cost) { cost->pieces = result.size(); }
    return result;
}
FailureOr<std::vector<IntegerSystem>> RegionExpressions::integerRelation(Id predicate,
    ArrayRef<Id> variables, const IntegerSystem& domain, std::string& diagnostic, RelationCost* cost) const
{
    return RelationBuilder(*this, variables, domain, diagnostic, cost).run(predicate);
}
} // namespace mlir::pto::frontiersynch
