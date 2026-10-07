// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Intern expressions independently of source cuts; emit shared arithmetic only
// after checking original-IR availability at each detached preparation cut.
#include "PTO/Transforms/FrontierSynch/RegionExpressions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
namespace mlir::pto::frontiersynch {
RegionExpressions::Transaction::Transaction(RegionExpressions& arena)
    : arena(arena), count(arena.nodes.size()), constructionMessage(arena.constructionMessage),
      emissionMessage(arena.emissionMessage) {}
RegionExpressions::Transaction::~Transaction()
{
    if (committed) { return; }
    while (arena.nodes.size() > count) {
        arena.interned.erase(arena.nodes.back());
        arena.nodes.pop_back();
    }
    arena.constructionMessage = std::move(constructionMessage);
    arena.emissionMessage = std::move(emissionMessage);
    // Forbidden recomputation is monotone source safety information. Keep it,
    // together with dominance of the unchanged original IR, across attempts.
}
bool RegionExpressions::Node::operator==(const Node& other) const
{
    return kind == other.kind && boolean == other.boolean && a == other.a && b == other.b &&
        c == other.c && literal == other.literal && value == other.value && equalInteger(*this, other);
}
std::size_t RegionExpressions::Hash::operator()(const Node& node) const
{
    return llvm::hash_combine(static_cast<unsigned>(node.kind), node.boolean, node.a, node.b,
                              node.c, node.literal, node.value.getAsOpaquePointer(), hashInteger(node));
}
RegionExpressions::Id RegionExpressions::reject(const char* message)
{
    if (constructionMessage.empty()) {
        constructionMessage = message;
    }
    return invalid;
}
RegionExpressions::Id RegionExpressions::intern(Node node)
{
    auto found = interned.find(node);
    if (found != interned.end()) {
        return found->second;
    }
    if (nodes.size() >= invalid - 1) {
        return reject("regional expression count exceeds representation");
    }
    Id id = static_cast<Id>(nodes.size());
    nodes.push_back(node);
    interned.emplace(node, id);
    return id;
}
RegionExpressions::Id RegionExpressions::constant(uint64_t value)
{
    Node node;
    node.literal = value;
    return intern(node);
}
RegionExpressions::Id RegionExpressions::boolean(bool value)
{
    Node node;
    node.literal = value;
    node.boolean = true;
    return intern(node);
}
RegionExpressions::Id RegionExpressions::input(Value value)
{
    if (!value) {
        return reject("regional expression has an absent SSA input");
    }
    auto integer = dyn_cast<IntegerType>(value.getType());
    if (!value.getType().isIndex() && (!integer || integer.getWidth() > 64)) {
        return reject("regional expression input requires index or at most 64-bit integer");
    }
    APInt number;
    if (matchPattern(value, m_ConstantInt(&number)) && number.getBitWidth() <= 64) {
        return value.getType().isInteger(1) ? boolean(!number.isZero()) : constant(number.getZExtValue());
    }
    Node node;
    node.kind = Kind::Input;
    node.boolean = value.getType().isInteger(1);
    node.value = value;
    return intern(node);
}
std::optional<uint64_t> RegionExpressions::constantValue(Id expression) const
{
    if (!valid(expression) || nodes[expression].kind != Kind::Constant) {
        return std::nullopt;
    }
    return nodes[expression].literal;
}
std::optional<uint64_t> RegionExpressions::constantUnder(Id premise, Id expression)
{
    if (!isBoolean(premise) || !valid(expression)) { return std::nullopt; }
    std::vector<std::optional<uint64_t>> values(static_cast<std::size_t>(expression) + 1);
    for (Id i = 0; i <= expression; ++i) {
        const auto node = nodes[i]; // Boolean negations below may grow nodes.
        auto& value = values[i];
        if (node.kind == Kind::Constant) { value = node.literal; }
        else if (node.kind == Kind::Select) {
            if (values[node.a]) { value = *values[node.a] ? values[node.b] : values[node.c]; }
            else if (values[node.b] && values[node.b] == values[node.c]) { value = values[node.b]; }
        } else if (node.kind == Kind::Not && values[node.a]) { value = !*values[node.a]; }
        else if (node.a != invalid && node.b != invalid && values[node.a] && values[node.b]) {
            value = fold(node.kind, *values[node.a], *values[node.b]);
        }
        if (!value && node.boolean) {
            if (refutesNegation(premise, i)) { value = 1; }
            else if (refutesNegation(premise, lnot(i))) { value = 0; }
        }
    }
    return values[expression];
}
bool RegionExpressions::isBoolean(Id expression) const
{
    return valid(expression) && nodes[expression].boolean;
}
RegionExpressions::Truth RegionExpressions::evaluateBoolean(
    const Node& node, const llvm::DenseMap<Id, Truth>& values) const
{
    auto known = [](bool value) { return value ? Truth::True : Truth::False; };
    Truth a = node.a == invalid ? Truth::Unknown : values.lookup(node.a);
    Truth b = node.b == invalid ? Truth::Unknown : values.lookup(node.b);
    switch (node.kind) {
        case Kind::Constant: return known(node.literal != 0);
        case Kind::Not: return a == Truth::Unknown ? a : known(a == Truth::False);
        case Kind::And:
            if (a == Truth::False || b == Truth::False) { return Truth::False; }
            return a == Truth::True && b == Truth::True ? Truth::True : Truth::Unknown;
        case Kind::Or:
            if (a == Truth::True || b == Truth::True) { return Truth::True; }
            return a == Truth::False && b == Truth::False ? Truth::False : Truth::Unknown;
        case Kind::Eq:
            return a != Truth::Unknown && b != Truth::Unknown ? known(a == b) : Truth::Unknown;
        case Kind::Select: {
            Truth c = values.lookup(node.c);
            if (a != Truth::Unknown) { return a == Truth::True ? b : c; }
            return b == c ? b : Truth::Unknown;
        }
        default: return Truth::Unknown;
    }
}
bool RegionExpressions::refutesNegation(Id premise, Id consequence) const
{
    // These bindings are necessary if premise AND NOT consequence holds.
    // Propagate only forced facts, never choose a Boolean valuation. Keeping
    // comparisons as independent atoms enlarges the possible valuation set.
    llvm::DenseMap<Id, Truth> bindings;
    SmallVector<std::pair<Id, Truth>> pending{{premise, Truth::True}, {consequence, Truth::False}};
    while (!pending.empty()) {
        const auto [id, required] = pending.pop_back_val();
        auto [entry, added] = bindings.try_emplace(id, required);
        if (!added) {
            if (entry->second != required) { return true; }
            continue;
        }
        const Node& node = nodes[id];
        if (node.kind == Kind::Not) {
            pending.push_back({node.a, required == Truth::True ? Truth::False : Truth::True});
        } else if ((node.kind == Kind::And && required == Truth::True) ||
                   (node.kind == Kind::Or && required == Truth::False)) {
            pending.push_back({node.a, required});
            pending.push_back({node.b, required});
        } else if (node.kind == Kind::Select) {
            // A Boolean select whose opposite arm cannot produce the required
            // truth value forces its condition and selected arm. Keep the
            // select in emitted code: replacing it by AND/OR would lose poison
            // masking for predicates replayed from inactive branches.
            const auto yes = constantValue(node.b), no = constantValue(node.c);
            const bool truth = required == Truth::True;
            if (no && (*no != 0) != truth) {
                pending.push_back({node.a, Truth::True});
                pending.push_back({node.b, required});
            }
            if (yes && (*yes != 0) != truth) {
                pending.push_back({node.a, Truth::False});
                pending.push_back({node.c, required});
            }
        }
    }
    // Only Boolean ancestors of the two formulas can contribute to this
    // proof. Other region queries share the arena but are irrelevant here.
    // A postorder walk visits shared gates once; comparisons remain atoms.
    SmallVector<std::pair<Id, bool>> walk{{premise, false}, {consequence, false}};
    SmallVector<Id> order;
    llvm::DenseSet<Id> seen;
    while (!walk.empty()) {
        const auto [id, expanded] = walk.pop_back_val();
        if (expanded) { order.push_back(id); continue; }
        if (!seen.insert(id).second) { continue; }
        walk.push_back({id, true});
        const auto& node = nodes[id];
        for (Id operand : {node.a, node.b, node.c}) {
            if (isBoolean(operand)) { walk.push_back({operand, false}); }
        }
    }
    llvm::DenseMap<Id, Truth> values;
    for (Id id : order) {
        const auto evaluated = evaluateBoolean(nodes[id], values);
        const auto required = bindings.lookup(id);
        if (required != Truth::Unknown && evaluated != Truth::Unknown && required != evaluated) {
            return true;
        }
        values[id] = required == Truth::Unknown ? evaluated : required;
    }
    return false;
}
bool RegionExpressions::implies(Id premise, Id consequence) const
{
    if (!isBoolean(premise) || !isBoolean(consequence)) { return false; }
    SmallVector<Id> pending{consequence};
    llvm::DenseSet<Id> seen;
    while (!pending.empty()) {
        Id next = pending.pop_back_val();
        if (!seen.insert(next).second) { continue; }
        if (nodes[next].kind == Kind::And) {
            pending.push_back(nodes[next].a);
            pending.push_back(nodes[next].b);
        } else if (!refutesNegation(premise, next)) {
            return false;
        }
    }
    return true;
}
SmallVector<RegionExpressions::Id> RegionExpressions::commonBooleanConjuncts(ArrayRef<Id> predicates) const
{
    SmallVector<Id> common;
    bool initialized = false;
    for (Id predicate : predicates) {
        if (!isBoolean(predicate)) { return {}; }
        if (constantValue(predicate) == 0) { continue; }
        llvm::DenseSet<Id> conjuncts, seen;
        SmallVector<Id> pending{predicate};
        while (!pending.empty()) {
            const auto id = pending.pop_back_val();
            if (!seen.insert(id).second) { continue; }
            const auto& node = nodes[id];
            if (node.kind == Kind::And) {
                pending.push_back(node.a); pending.push_back(node.b);
            } else if (constantValue(id) != 1) {
                conjuncts.insert(id);
            }
        }
        if (!initialized) {
            common.assign(conjuncts.begin(), conjuncts.end());
            initialized = true;
        } else {
            llvm::erase_if(common, [&](Id id) { return !conjuncts.count(id); });
        }
        if (common.empty()) { break; }
    }
    llvm::sort(common);
    return common;
}
std::optional<uint64_t> RegionExpressions::fold(Kind kind, uint64_t a, uint64_t b) const
{
    // Unsigned wrap is deliberate: these are the same bits emitted by index
    // arithmetic on the producer's certified 64-bit index layout.
    switch (kind) {
        case Kind::Add: return a + b;
        case Kind::Sub: return a - b;
        case Kind::Div: return b ? std::optional<uint64_t>(a / b) : std::nullopt;
        case Kind::Rem: return b ? std::optional<uint64_t>(a % b) : std::nullopt;
        case Kind::Lt: return a < b;
        case Kind::Le: return a <= b;
        case Kind::Eq: return a == b;
        case Kind::SLt: return APInt(64, a).slt(APInt(64, b));
        case Kind::SLe: return APInt(64, a).sle(APInt(64, b));
        case Kind::And: return a & b;
        case Kind::Or: return a | b;
        default: return std::nullopt;
    }
}
RegionExpressions::Id RegionExpressions::absorbBoolean(Kind kind, Id a, Id b) const
{
    // Ordered unsigned thresholds on one expression absorb each other. The
    // returned atom already exists; inspecting two comparison nodes is O(1).
    struct Threshold { Id operand; uint64_t bound; bool lower; };
    auto threshold = [&](Id id) -> std::optional<Threshold> {
        const Node& node = nodes[id];
        if (node.kind != Kind::Lt && node.kind != Kind::Le) { return std::nullopt; }
        if (auto left = constantValue(node.a)) {
            if (node.kind == Kind::Lt && *left == UINT64_MAX) { return std::nullopt; }
            return Threshold{node.b, *left + (node.kind == Kind::Lt), true};
        }
        if (auto right = constantValue(node.b)) {
            if (node.kind == Kind::Lt && *right == 0) { return std::nullopt; }
            return Threshold{node.a, *right - (node.kind == Kind::Lt), false};
        }
        return std::nullopt;
    };
    auto left = threshold(a), right = threshold(b);
    if (left && right && left->operand == right->operand && left->lower == right->lower) {
        const bool chooseLarger = (kind == Kind::And) == left->lower;
        return (chooseLarger ? left->bound >= right->bound : left->bound <= right->bound) ? a : b;
    }
    Kind opposite = kind == Kind::And ? Kind::Or : Kind::And;
    // Inspect only immediate children: absorption and repeated-factor removal
    // add constant work per gate, regardless of the circuit's depth or sharing.
    for (Id nested : {a, b}) {
        const Node& child = nodes[nested];
        Id other = nested == a ? b : a;
        if (child.a != other && child.b != other) { continue; }
        if (child.kind == opposite) { return other; }
        if (child.kind == kind) { return nested; }
    }
    return invalid;
}
RegionExpressions::Id RegionExpressions::binary(Kind kind, Id a, Id b)
{
    if (!valid(a) || !valid(b)) {
        return reject("regional expression references an invalid operand");
    }
    bool logic = kind == Kind::And || kind == Kind::Or;
    bool comparison = kind == Kind::Lt || kind == Kind::Le || kind == Kind::Eq ||
        kind == Kind::SLt || kind == Kind::SLe;
    if (isBoolean(a) != isBoolean(b) || (kind != Kind::Eq && isBoolean(a) != logic)) {
        return reject("regional expression operand types disagree");
    }
    auto av = constantValue(a), bv = constantValue(b);
    if ((kind == Kind::Div || kind == Kind::Rem) && (!bv || *bv == 0)) {
        return reject("regional division requires a positive constant divisor");
    }
    if (av && bv) {
        auto result = fold(kind, *av, *bv);
        if (result) {
            return logic || comparison ? boolean(*result != 0) : constant(*result);
        }
    }
    if (a == b) {
        if (kind == Kind::Eq || kind == Kind::Le || kind == Kind::SLe) { return boolean(true); }
        if (kind == Kind::Lt || kind == Kind::SLt) { return boolean(false); }
        if (kind == Kind::Sub) { return constant(0); }
        if (logic) { return a; }
    }
    // These are unsigned identities. Signed loop-bound comparisons retain
    // their separate semantics, including for negative source indices.
    if (kind == Kind::Le && (av == 0 || bv == UINT64_MAX)) { return boolean(true); }
    if (kind == Kind::Lt && (bv == 0 || av == UINT64_MAX)) { return boolean(false); }
    if (kind == Kind::Le && av && *av != 0) { return lt(constant(*av - 1), b); }
    if ((kind == Kind::Lt || kind == Kind::Le) && nodes[a].kind == Kind::Sub && nodes[a].a == b) {
        auto amount = constantValue(nodes[a].b);
        // For a positive unsigned c, x-c wraps precisely when x<c.
        if (amount && *amount != 0) { return le(nodes[a].b, b); }
    }
    if (kind == Kind::Eq) {
        if (bv == 0 && nodes[a].kind == Kind::Sub) { return eq(nodes[a].a, nodes[a].b); }
        if (av == 0 && nodes[b].kind == Kind::Sub) { return eq(nodes[b].a, nodes[b].b); }
    }
    if (logic && ((nodes[a].kind == Kind::Not && nodes[a].a == b) ||
                  (nodes[b].kind == Kind::Not && nodes[b].a == a))) {
        return boolean(kind == Kind::Or);
    }
    if (logic) {
        Id simplified = absorbBoolean(kind, a, b);
        if (simplified != invalid) { return simplified; }
    }
    if (kind == Kind::And && (av || bv)) { return av ? (*av ? b : a) : (*bv ? a : b); }
    if (kind == Kind::Or && (av || bv)) { return av ? (*av ? a : b) : (*bv ? b : a); }
    if ((kind == Kind::Add || kind == Kind::Sub) && bv == 0) { return a; }
    if (kind == Kind::Add && av == 0) { return b; }
    if (kind == Kind::Div && bv == 1) { return a; }
    if (kind == Kind::Div && bv && nodes[a].kind == Kind::Select) {
        const auto choice = nodes[a];
        const auto comparison = nodes[choice.a];
        if (comparison.kind == Kind::Lt && comparison.a == choice.b && comparison.b == choice.c) {
            // A literal cap below the divisor bounds both outcomes of min.
            auto yes = constantValue(choice.b), no = constantValue(choice.c);
            if ((yes && *yes < *bv) || (no && *no < *bv)) { return constant(0); }
        }
    }
    if (kind == Kind::Rem && bv == 1) { return constant(0); }
    if ((kind == Kind::Add || kind == Kind::Eq || logic) && b < a) {
        std::swap(a, b);
    }
    Node node;
    node.kind = kind;
    node.boolean = logic || comparison;
    node.a = a;
    node.b = b;
    return intern(node);
}
RegionExpressions::Id RegionExpressions::minimum(Id a, Id b)
{
    if (!valid(a) || !valid(b) || isBoolean(a) || isBoolean(b)) {
        return reject("regional minimum requires index operands");
    }
    if (a == b) { return a; }
    auto av = constantValue(a), bv = constantValue(b);
    if (av && bv) { return constant(std::min(*av, *bv)); }
    if (av == 0 || bv == 0) { return constant(0); }
    // Clamp two literal alternatives without distributing over a general DAG.
    for (unsigned swap = 0; swap < 2; ++swap) {
        Id fixed = swap ? b : a, choice = swap ? a : b;
        auto value = constantValue(fixed);
        const Node node = nodes[choice];
        if (!value || node.kind != Kind::Select) { continue; }
        auto yes = constantValue(node.b), no = constantValue(node.c);
        if (yes && no) {
            return select(node.a, constant(std::min(*value, *yes)), constant(std::min(*value, *no)));
        }
    }
    const Node left = nodes[a], right = nodes[b];
    if (left.kind == Kind::Select && right.kind == Kind::Select && left.b == right.b && left.c == right.c) {
        auto yes = constantValue(left.b), no = constantValue(left.c);
        if (yes && no) {
            auto guard = *yes < *no ? lor(left.a, right.a) : land(left.a, right.a);
            return select(guard, left.b, left.c);
        }
    }
    // With a shared literal infinity, the cheaper active edge always wins.
    // Keep the inactive edge's choice intact instead of comparing two guarded
    // distances. Inspect only these two nodes; do not distribute larger DAGs.
    if (left.kind == Kind::Select && right.kind == Kind::Select && left.c == right.c) {
        const auto x = constantValue(left.b), y = constantValue(right.b), cap = constantValue(left.c);
        if (x && y && cap && *x <= *cap && *y <= *cap) {
            if (*x < *y) { return select(left.a, left.b, b); }
            if (*y < *x) { return select(right.a, right.b, a); }
        }
    }
    if (b < a) { std::swap(a, b); }
    return select(lt(a, b), a, b);
}
RegionExpressions::Id RegionExpressions::boundedMinPlus(Id current, Id a, Id b, uint64_t cap)
{
    if (!valid(current) || !valid(a) || !valid(b) || isBoolean(current) ||
        isBoolean(a) || isBoolean(b) || cap > UINT64_MAX / 2) {
        return reject("regional bounded sum requires bounded index operands");
    }
    const auto absent = constant(cap);
    if (a == absent || b == absent) { return current; }
    auto av = constantValue(a), bv = constantValue(b);
    if (av && bv) {
        if (*av > cap || *bv > cap) { return reject("regional bounded sum exceeds its operand bound"); }
        return minimum(current, constant(std::min(*av + *bv, cap)));
    }
    auto zeroChoice = [&](Id value) -> std::optional<Id> {
        if (constantValue(value) == 0) { return boolean(true); }
        const auto node = nodes[value];
        if (node.kind == Kind::Select && constantValue(node.b) == 0 && node.c == absent) { return node.a; }
        return std::nullopt;
    };
    auto boundedMinimum = [&](Id x, Id y) {
        // Both arguments are within [0,cap]. A disabled zero-cost edge is
        // infinity and cannot improve the other argument. Keep its select:
        // inactive predicates must not lose their original poison masking.
        if (auto guard = zeroChoice(x)) { return select(*guard, constant(0), y); }
        if (auto guard = zeroChoice(y)) { return select(*guard, constant(0), x); }
        return minimum(x, y);
    };
    if (auto guard = zeroChoice(a)) { return select(*guard, boundedMinimum(current, b), current); }
    if (auto guard = zeroChoice(b)) { return select(*guard, boundedMinimum(current, a), current); }
    struct Choice { Id guard; uint64_t value; };
    auto choice = [&](Id value) -> std::optional<Choice> {
        if (auto literal = constantValue(value)) {
            if (*literal <= cap) { return Choice{boolean(true), *literal}; }
            return std::nullopt;
        }
        const Node node = nodes[value];
        if (node.kind != Kind::Select) { return std::nullopt; }
        auto finite = constantValue(node.b), fallback = constantValue(node.c);
        if (finite && *finite <= cap && fallback == cap) { return Choice{node.a, *finite}; }
        return std::nullopt;
    };
    auto left = choice(a), right = choice(b);
    if (left && right) {
        auto candidate = select(land(left->guard, right->guard),
            constant(std::min(left->value + right->value, cap)), absent);
        return boundedMinimum(current, candidate);
    }
    // current<=cap already saturates the candidate: do not build a second
    // minimum that would duplicate work at every closure relaxation.
    return minimum(add(a, b), current);
}
RegionExpressions::Id RegionExpressions::add(Id a, Id b) { return binary(Kind::Add, a, b); }
RegionExpressions::Id RegionExpressions::sub(Id a, Id b) { return binary(Kind::Sub, a, b); }
RegionExpressions::Id RegionExpressions::div(Id a, Id b) { return binary(Kind::Div, a, b); }
RegionExpressions::Id RegionExpressions::rem(Id a, Id b) { return binary(Kind::Rem, a, b); }
RegionExpressions::Id RegionExpressions::lt(Id a, Id b) { return binary(Kind::Lt, a, b); }
RegionExpressions::Id RegionExpressions::le(Id a, Id b) { return binary(Kind::Le, a, b); }
RegionExpressions::Id RegionExpressions::eq(Id a, Id b) { return binary(Kind::Eq, a, b); }
RegionExpressions::Id RegionExpressions::slt(Id a, Id b) { return binary(Kind::SLt, a, b); }
RegionExpressions::Id RegionExpressions::sle(Id a, Id b) { return binary(Kind::SLe, a, b); }
RegionExpressions::Id RegionExpressions::land(Id a, Id b) { return binary(Kind::And, a, b); }
RegionExpressions::Id RegionExpressions::lor(Id a, Id b) { return binary(Kind::Or, a, b); }
RegionExpressions::Id RegionExpressions::lnot(Id a)
{
    if (!isBoolean(a)) { return reject("regional negation requires a Boolean operand"); }
    if (auto value = constantValue(a)) { return boolean(!*value); }
    if (nodes[a].kind == Kind::Not) { return nodes[a].a; }
    Node node;
    node.kind = Kind::Not;
    node.boolean = true;
    node.a = a;
    return intern(node);
}
RegionExpressions::Id RegionExpressions::select(Id condition, Id yes, Id no)
{
    if (!isBoolean(condition) || !valid(yes) || !valid(no) || isBoolean(yes) != isBoolean(no)) {
        return reject("regional selection requires Boolean condition and equally typed values");
    }
    if (yes == no) { return yes; }
    if (auto value = constantValue(condition)) { return *value ? yes : no; }
    if (isBoolean(yes)) {
        const auto y = constantValue(yes), n = constantValue(no);
        if (y && n) { return *y ? condition : lnot(condition); }
    }
    // Selecting the same condition twice cannot visit the opposite inner arm.
    // Preserve the outer select so inactive poison remains masked.
    if (nodes[yes].kind == Kind::Select && nodes[yes].a == condition) { yes = nodes[yes].b; }
    if (nodes[no].kind == Kind::Select && nodes[no].a == condition) { no = nodes[no].c; }
    if (yes == no) { return yes; }
    Node node;
    node.kind = Kind::Select;
    node.boolean = isBoolean(yes);
    node.a = condition;
    node.b = yes;
    node.c = no;
    return intern(node);
}
bool RegionExpressions::prepareEmission(Id expression, OpBuilder& builder, Operation* cut,
    const llvm::DenseMap<Id, Value>& memo, SmallVectorImpl<Id>& order)
{
    auto* block = builder.getInsertionBlock();
    auto function = cut ? cut->getParentOfType<func::FuncOp>() : func::FuncOp();
    if (!constructionMessage.empty() || !valid(expression) || !block || block->getParent() || !function) {
        emissionMessage = "regional expression emission requires a valid root, original cut and detached block";
        return false;
    }
    if (dominanceRoot != function.getOperation()) {
        dominanceRoot = function.getOperation();
        dominance = std::make_unique<DominanceInfo>(dominanceRoot);
    }
    SmallVector<Id> pending{expression};
    llvm::DenseSet<Id> seen;
    while (!pending.empty()) {
        Id id = pending.pop_back_val();
        if (memo.count(id) || !seen.insert(id).second) { continue; }
        const Node& node = nodes[id];
        if (node.kind == Kind::Input) {
            Value original = node.value;
            auto* region = original.getParentRegion();
            auto* owner = region ? region->getParentOp() : nullptr;
            if (!owner || (owner != function && !function->isProperAncestor(owner)) ||
                !dominance->properlyDominates(node.value, cut)) {
                emissionMessage = "regional expression input is unavailable at endpoint cut";
                return false;
            }
        }
        appendOperands(node, pending);
        order.push_back(id);
    }
    // Interned parents always follow their operands; sorting avoids recursive
    // traversal and stack growth for long closure circuits.
    llvm::sort(order);
    return true;
}
Value RegionExpressions::emitNode(const Node& node, OpBuilder& builder, Location location,
    const llvm::DenseMap<Id, Value>& memo) const
{
    // DenseMap reserves UINT32_MAX; absent operands must never be looked up.
    Value a = node.a == invalid ? Value() : memo.lookup(node.a);
    Value b = node.b == invalid ? Value() : memo.lookup(node.b);
    Value c = node.c == invalid ? Value() : memo.lookup(node.c);
    switch (node.kind) {
        case Kind::Integer: return emitInteger(node, builder, location, memo);
        case Kind::Constant:
            if (node.boolean) { return builder.create<arith::ConstantIntOp>(location, node.literal, 1); }
            return builder.create<arith::ConstantOp>(location, builder.getIndexType(),
                IntegerAttr::get(builder.getIndexType(), APInt(64, node.literal)));
        case Kind::Input:
            if (node.value.getType().isIndex() || node.boolean) { return node.value; }
            return builder.create<arith::IndexCastUIOp>(location, builder.getIndexType(), node.value);
        case Kind::Add: return builder.create<arith::AddIOp>(location, a, b);
        case Kind::Sub: return builder.create<arith::SubIOp>(location, a, b);
        case Kind::Div: return builder.create<arith::DivUIOp>(location, a, b);
        case Kind::Rem: return builder.create<arith::RemUIOp>(location, a, b);
        case Kind::Lt: return builder.create<arith::CmpIOp>(location, arith::CmpIPredicate::ult, a, b);
        case Kind::Le: return builder.create<arith::CmpIOp>(location, arith::CmpIPredicate::ule, a, b);
        case Kind::Eq: return builder.create<arith::CmpIOp>(location, arith::CmpIPredicate::eq, a, b);
        case Kind::SLt: return builder.create<arith::CmpIOp>(location, arith::CmpIPredicate::slt, a, b);
        case Kind::SLe: return builder.create<arith::CmpIOp>(location, arith::CmpIPredicate::sle, a, b);
        case Kind::And: return builder.create<arith::AndIOp>(location, a, b);
        case Kind::Or: return builder.create<arith::OrIOp>(location, a, b);
        case Kind::Not:
            return builder.create<arith::XOrIOp>(location, a, builder.create<arith::ConstantIntOp>(location, 1, 1));
        case Kind::Select: return builder.create<arith::SelectOp>(location, a, b, c);
        default: return {};
    }
}
FailureOr<Value> RegionExpressions::emit(Id expression, OpBuilder& builder, Operation* cut,
    llvm::DenseMap<Id, Value>& memo)
{
    emissionMessage.clear();
    SmallVector<Id> order;
    if (!prepareEmission(expression, builder, cut, memo, order)) { return failure(); }
    for (Id id : order) {
        memo[id] = emitNode(nodes[id], builder, cut->getLoc(), memo);
    }
    return memo.lookup(expression);
}
SmallVector<std::pair<RegionExpressions::Id, Value>> RegionExpressions::referencedInputs(Id root) const
{
    SmallVector<std::pair<Id, Value>> result;
    SmallVector<Id> pending{root};
    llvm::DenseSet<Id> seen;
    while (!pending.empty()) {
        auto id = pending.pop_back_val();
        if (!valid(id) || !seen.insert(id).second) { continue; }
        if (nodes[id].kind == Kind::Input) { result.push_back({id,nodes[id].value}); }
        appendOperands(nodes[id],pending);
    }
    llvm::sort(result,[](const auto& a,const auto& b) { return a.first < b.first; });
    return result;
}
} // namespace mlir::pto::frontiersynch
