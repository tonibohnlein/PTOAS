// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Structured must-state analysis. Branch transactions retain only changed
// descriptors; repeated accesses do not rescan users or block prefixes.
#include "PTO/Transforms/InsertSync/SyncDescriptorState.h"
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/InsertSync/SyncScalarFacts.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/MathExtras.h"
#include <map>
namespace mlir::pto {
namespace {
using Shape = SyncDescriptorShape;
using State = DenseMap<Value, Shape>;
bool constantInt(Value value, Operation* access, int64_t& result)
{
    APInt integer;
    if (!value || !matchPattern(value, m_ConstantInt(&integer)) || !integer.isSignedIntN(64)) {
        return false;
    }
    if (value.getType().isIndex()) {
        auto bits = DataLayout::closest(access).getTypeSizeInBits(value.getType());
        if (bits.isScalable() || !bits.getFixedValue() || bits.getFixedValue() > 64 ||
            !integer.isSignedIntN(bits.getFixedValue())) {
            return false;
        }
    }
    result = integer.getSExtValue();
    return true;
}
SyncDescriptorDimension dimension(Value value, Operation* access, const SyncScalarFacts& scalars)
{
    if (!value) {
        return {};
    }
    DenseMap<Value, unsigned> symbols;
    auto expression = scalars.value(value, access, [&](Value original) {
        auto entry = symbols.try_emplace(original, symbols.size());
        return getAffineSymbolExpr(entry.first->second, access->getContext());
    });
    if (auto folded = dyn_cast_or_null<AffineConstantExpr>(expression)) {
        return {folded.getValue(), {}};
    }
    return {std::nullopt, value};
}
Shape meet(const Shape& a, const Shape& b)
{
    Shape result;
    for (unsigned i = 0; i < 2; ++i) {
        if (a[i] == b[i]) {
            result[i] = a[i];
        }
    }
    return result;
}
bool definedWithin(Value value, Operation* owner)
{
    if (!value) {
        return false;
    }
    auto* definition = value.getDefiningOp();
    if (!definition) {
        definition = cast<BlockArgument>(value).getOwner()->getParentOp();
    }
    return definition == owner || owner->isProperAncestor(definition);
}
Shape acrossVisit(Shape shape, Operation* loop)
{
    for (auto& dim : shape) {
        if (definedWithin(dim.value, loop)) {
            dim = {};
        }
    }
    return shape;
}
Shape initial(Value value, Operation* access, const SyncScalarFacts& scalars)
{
    auto type = dyn_cast<TileBufType>(value.getType());
    Shape result;
    if (!type || type.getValidShape().size() != 2) {
        return result;
    }
    Value row, col;
    if (auto alloc = value.getDefiningOp<AllocTileOp>()) {
        row = alloc.getValidRow();
        col = alloc.getValidCol();
    } else if (auto view = value.getDefiningOp<SubViewOp>()) {
        row = view.getValidRow();
        col = view.getValidCol();
    } else if (auto get = value.getDefiningOp<MultiTileGetOp>()) {
        if (auto alloc = get.getSource().getDefiningOp<AllocMultiTileOp>()) {
            row = alloc.getValidRow();
            col = alloc.getValidCol();
        }
    }
    for (unsigned i = 0; i < 2; ++i) {
        auto size = type.getValidShape()[i];
        result[i] = size >= 0 ? SyncDescriptorDimension{size, {}} : dimension(i == 0 ? row : col, access, scalars);
    }
    return result;
}
} // namespace
struct SyncDescriptorState::Impl {
    State state;
    DenseMap<Operation*, State> facts;
    DenseMap<Operation*, SmallVector<Value>> extra;
    std::shared_ptr<const SyncScalarFacts> scalars;
    DenseSet<Value> dynamicDescriptors;
    std::map<Operation*, SmallVector<Value>> loopInputs;
    struct LoopMemo {
        SmallVector<Shape> entry;
        SmallVector<Shape> exit;
    };
    DenseMap<Operation*, LoopMemo> loopMemo;
    std::map<Operation*, State> loopHeaders;
    // Each frame stores the old value on the first write, not a whole state copy.
    SmallVector<DenseMap<Value, std::optional<Shape>>> frames;
    mutable SyncDescriptorWork counters;
    Shape get(Value value) const
    {
        auto found = state.find(value);
        return found == state.end() ? Shape{} : found->second;
    }
    void set(Value value, Shape shape)
    {
        if (!frames.empty()) {
            auto found = state.find(value);
            frames.back().try_emplace(value, found == state.end() ? std::nullopt : std::optional<Shape>(found->second));
        }
        state[value] = std::move(shape);
    }
    State rollback()
    {
        auto old = std::move(frames.back());
        frames.pop_back();
        State changed;
        for (const auto& entry : old) {
            changed[entry.first] = get(entry.first);
            if (entry.second) {
                state[entry.first] = *entry.second;
            } else {
                state.erase(entry.first);
            }
        }
        return changed;
    }
    void invalidate()
    {
        for (auto descriptor : dynamicDescriptors) {
            if (state.count(descriptor)) {
                set(descriptor, {});
            }
        }
    }
    void snapshot(Operation& op)
    {
        auto& at = facts[&op];
        at.clear();
        for (auto value : op.getOperands()) {
            if (isa<TileBufType>(value.getType())) {
                at[value] = get(value);
            }
        }
        auto found = extra.find(&op);
        if (found != extra.end()) {
            for (auto value : found->second) {
                at[value] = get(value);
            }
        }
    }
    State trial(Region& region)
    {
        frames.emplace_back();
        walk(region);
        return rollback();
    }
    void conditional(scf::IfOp branch)
    {
        auto yes = trial(branch.getThenRegion());
        auto no = trial(branch.getElseRegion());
        DenseSet<Value> keys;
        for (const auto& e : yes) {
            keys.insert(e.first);
        }
        for (const auto& e : no) {
            keys.insert(e.first);
        }
        for (auto key : keys) {
            if (definedWithin(key, branch)) {
                continue;
            }
            auto y = yes.find(key), n = no.find(key);
            set(key, meet(y == yes.end() ? get(key) : y->second, n == no.end() ? get(key) : n->second));
        }
    }
    void loop(scf::ForOp loop)
    {
        auto& inputs = loopInputs[loop];
        SmallVector<Shape> entry;
        for (auto value : inputs) {
            entry.push_back(get(value));
        }
        auto memo = loopMemo.find(loop);
        if (memo != loopMemo.end() && memo->second.entry == entry) {
            ++counters.reusedLoops;
            // The lexical subtree's snapshots still describe its last evaluated
            // entry. Reuse only that most recent entry, including unknown facts.
            for (unsigned i = 0; i < inputs.size(); ++i) {
                set(inputs[i], memo->second.exit[i]);
            }
            return;
        }
        loopBody(loop);
        LoopMemo result;
        result.entry = std::move(entry);
        for (auto value : inputs) {
            result.exit.push_back(get(value));
        }
        loopMemo[loop] = std::move(result);
    }
    void loopBody(scf::ForOp loop)
    {
        int64_t lower = 0, upper = 0, step = 0;
        const bool constant = constantInt(loop.getLowerBound(), loop, lower) &&
                              constantInt(loop.getUpperBound(), loop, upper) &&
                              constantInt(loop.getStep(), loop, step) && step > 0;
        const bool zero = constant && lower >= upper;
        int64_t next = 0;
        const bool one = constant && !zero && !llvm::AddOverflow(lower, step, next) && next >= upper;
        if (zero) {
            // Index unreachable accesses conservatively, without publishing their state.
            (void)trial(loop.getRegion());
            return;
        }
        if (one) {
            auto exit = trial(loop.getRegion());
            for (const auto& e : exit) {
                if (!definedWithin(e.first, loop)) {
                    set(e.first, acrossVisit(e.second, loop));
                }
            }
            return;
        }
        // Headers are monotone across enclosing-loop rewalks too. Each lexical
        // descriptor dimension can lose its known header fact only once; a fresh
        // outer traversal must not restart nested convergence from initial facts.
        // Every extra body pass is charged to such a loss, bounding source visits
        // by source size times the number of indexed header dimensions.
        // Strong updates in the body can still establish precise point facts.
        State entry, back;
        auto& header = loopHeaders[loop];
        for (auto& e : header) {
            entry[e.first] = get(e.first);
            auto joined = meet(e.second, entry[e.first]);
            if (joined != e.second) {
                ++counters.headerFactLosses;
            }
            e.second = joined;
        }
        bool changed = true;
        while (changed) {
            ++counters.fixedPointPasses;
            frames.emplace_back();
            for (const auto& e : header) {
                set(e.first, e.second);
            }
            walk(loop.getRegion());
            back = rollback();
            changed = false;
            for (const auto& e : back) {
                if (definedWithin(e.first, loop)) {
                    continue;
                }
                auto start = entry.try_emplace(e.first, get(e.first));
                auto joined = meet(start.first->second, acrossVisit(e.second, loop));
                auto old = header.find(e.first);
                const auto previous = old == header.end() ? start.first->second : old->second;
                joined = meet(previous, joined);
                if (previous != joined) {
                    changed = true;
                    ++counters.headerFactLosses;
                }
                header[e.first] = joined;
            }
        }
        for (const auto& e : header) {
            // Unknown count includes the identity (zero visits). A known positive
            // loop may export its final strong update, but no loop-local SSA value.
            set(e.first, constant ? acrossVisit(back.lookup(e.first), loop) : e.second);
        }
    }
    void operation(Operation& op)
    {
        ++counters.sourceVisits;
        snapshot(op);
        if (auto update = dyn_cast<SetValidShapeOp>(op)) {
            ++counters.updates;
            set(update.getSource(),
                {dimension(update.getValidRow(), &op, *scalars), dimension(update.getValidCol(), &op, *scalars)});
        } else if (auto branch = dyn_cast<scf::IfOp>(op)) {
            conditional(branch);
        } else if (auto repeat = dyn_cast<scf::ForOp>(op)) {
            loop(repeat);
        } else if (isa<SectionCubeOp, SectionVectorOp>(op)) {
            for (auto& region : op.getRegions()) {
                walk(region);
            }
        } else if (op.getNumRegions()) {
            invalidate();
            for (auto& region : op.getRegions()) {
                (void)trial(region);
            }
            invalidate();
        } else if (isa<func::CallOp>(op)) {
            // An opaque callee can mutate a passed descriptor. No interprocedural
            // metadata guarantee is inferred from ordinary buffer effects.
            invalidate();
        }
        for (auto value : op.getResults()) {
            if (isa<TileBufType>(value.getType())) {
                set(value, initial(value, &op, *scalars));
            }
        }
    }
    void walk(Region& region)
    {
        if (region.empty()) {
            return;
        }
        if (!region.hasOneBlock()) {
            invalidate();
            for (auto& block : region) {
                frames.emplace_back();
                for (auto& op : block) {
                    operation(op);
                }
                (void)rollback();
            }
            invalidate();
            return;
        }
        for (auto& op : region.front()) {
            operation(op);
        }
    }
};
SyncDescriptorState::SyncDescriptorState(
    func::FuncOp function, ArrayRef<Query> queries, std::shared_ptr<const SyncScalarFacts> scalars)
    : impl(std::make_unique<Impl>())
{
    impl->scalars = scalars ? std::move(scalars) : std::make_shared<SyncScalarFacts>();
    if (!function || function.isDeclaration()) {
        return;
    }
    function.walk([&](Operation* op) {
        ++impl->counters.indexedOperations;
        for (auto value : op->getResults()) {
            if (auto type = dyn_cast<TileBufType>(value.getType()); type && type.hasDynamicValid()) {
                impl->dynamicDescriptors.insert(value);
            }
        }
        for (auto value : op->getOperands()) {
            if (auto type = dyn_cast<TileBufType>(value.getType()); type && type.hasDynamicValid()) {
                impl->dynamicDescriptors.insert(value);
            }
        }
    });
    for (const auto& query : queries) {
        if (query.first && query.second && isa<TileBufType>(query.second.getType())) {
            impl->extra[query.first].push_back(query.second);
        }
    }
    // Index the external descriptor dependencies of each loop once. This is
    // bounded by operand count times nesting depth, not by runtime trip counts.
    // Include translator-only base-buffer queries in the same context key.
    DenseMap<Operation*, DenseSet<Value>> dependencies;
    function.walk([&](Operation* op) {
        ++impl->counters.indexedOperations;
        SmallVector<Value> values(op->getOperands());
        if (isa<func::CallOp>(op) ||
            (op->getNumRegions() && !isa<scf::ForOp, scf::IfOp, SectionCubeOp, SectionVectorOp>(op))) {
            values.append(impl->dynamicDescriptors.begin(), impl->dynamicDescriptors.end());
        }
        auto extra = impl->extra.find(op);
        if (extra != impl->extra.end()) {
            values.append(extra->second);
        }
        for (auto value : values) {
            if (!isa<TileBufType>(value.getType())) {
                continue;
            }
            for (auto* parent = op->getParentOp(); parent != function && parent; parent = parent->getParentOp()) {
                if (isa<scf::ForOp>(parent) && !definedWithin(value, parent)) {
                    if (dependencies[parent].insert(value).second) {
                        impl->loopInputs[parent].push_back(value);
                    }
                }
            }
        }
    });
    for (auto arg : function.getArguments()) {
        if (isa<TileBufType>(arg.getType())) {
            impl->state[arg] = initial(arg, function, *impl->scalars);
        }
    }
    impl->walk(function.getBody());
}
SyncDescriptorState::~SyncDescriptorState() = default;
std::optional<SyncDescriptorShape> SyncDescriptorState::before(Value descriptor, Operation* access) const
{
    ++impl->counters.lookups;
    auto at = impl->facts.find(access);
    if (at == impl->facts.end()) {
        return std::nullopt;
    }
    auto found = at->second.find(descriptor);
    return found == at->second.end() ? std::nullopt : std::optional<Shape>(found->second);
}
std::optional<std::array<int64_t, 2>> SyncDescriptorState::constantShape(Value descriptor, Operation* access) const
{
    auto shape = before(descriptor, access);
    if (!shape || !(*shape)[0].constant || !(*shape)[1].constant) {
        return std::nullopt;
    }
    return std::array<int64_t, 2>{*(*shape)[0].constant, *(*shape)[1].constant};
}
SyncDescriptorWork SyncDescriptorState::work() const { return impl->counters; }
} // namespace mlir::pto
