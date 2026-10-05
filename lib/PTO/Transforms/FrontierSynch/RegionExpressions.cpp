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
bool RegionExpressions::Node::operator==(const Node& other) const
{
    return kind == other.kind && boolean == other.boolean && a == other.a && b == other.b &&
        c == other.c && literal == other.literal && value == other.value;
}
std::size_t RegionExpressions::Hash::operator()(const Node& node) const
{
    return llvm::hash_combine(static_cast<unsigned>(node.kind), node.boolean, node.a, node.b,
                              node.c, node.literal, node.value.getAsOpaquePointer());
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
bool RegionExpressions::isBoolean(Id expression) const
{
    return valid(expression) && nodes[expression].boolean;
}
RegionExpressions::Truth RegionExpressions::evaluateBoolean(const Node& node, ArrayRef<Truth> values) const
{
    auto known = [](bool value) { return value ? Truth::True : Truth::False; };
    Truth a = node.a == invalid ? Truth::Unknown : values[node.a];
    Truth b = node.b == invalid ? Truth::Unknown : values[node.b];
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
            Truth c = values[node.c];
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
    const auto count = static_cast<std::size_t>(std::max(premise, consequence)) + 1;
    SmallVector<Truth> bindings(count, Truth::Unknown);
    SmallVector<std::pair<Id, Truth>> pending{{premise, Truth::True}, {consequence, Truth::False}};
    while (!pending.empty()) {
        const auto [id, required] = pending.pop_back_val();
        if (bindings[id] != Truth::Unknown) {
            if (bindings[id] != required) { return true; }
            continue;
        }
        bindings[id] = required;
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
    SmallVector<Truth> values(count, Truth::Unknown);
    for (std::size_t id = 0; id < count; ++id) {
        if (!nodes[id].boolean) { continue; }
        const auto evaluated = evaluateBoolean(nodes[id], values);
        const auto required = bindings[id];
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
        for (Id operand : {node.a, node.b, node.c}) {
            if (operand != invalid) { pending.push_back(operand); }
        }
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
} // namespace mlir::pto::frontiersynch
