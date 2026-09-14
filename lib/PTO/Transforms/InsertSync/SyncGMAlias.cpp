// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Preserve all possible argument roots through views and structured forwarding.
// Opaque producers, integer arithmetic and exhausted traces cannot prove noalias.
#include "PTO/Transforms/InsertSync/SyncGMAlias.h"
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include <functional>
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <vector>

using namespace mlir;
using namespace mlir::pto;

namespace {
constexpr unsigned kMaximumRootValues = 256;

bool appendLoopSources(scf::ForOp loop, unsigned index, SmallVectorImpl<Value>& pending)
{
    if (index >= loop.getInitArgs().size() || index >= loop.getYieldedValues().size()) {
        return false;
    }
    pending.push_back(loop.getInitArgs()[index]);
    pending.push_back(loop.getYieldedValues()[index]);
    return true;
}

bool appendSources(Value value, func::FuncOp function, SmallVectorImpl<Value>& pending, SmallVectorImpl<Value>& roots)
{
    if (auto argument = dyn_cast<BlockArgument>(value)) {
        if (argument.getOwner() == &function.getBody().front()) {
            if (!isa<PtrType, TensorViewType, PartitionTensorViewType, TileBufType, MultiTileBufType,
                     BaseMemRefType>(value.getType())) {
                return false;
            }
            roots.push_back(value);
            return true;
        }
        Operation* owner = argument.getOwner()->getParentOp();
        unsigned index = argument.getArgNumber();
        if (auto loop = dyn_cast_or_null<scf::WhileOp>(owner)) {
            if (loop.getBefore().empty() || loop.getAfter().empty())
                return false;
            if (argument.getOwner() == &loop.getBefore().front()) {
                auto yield = dyn_cast<scf::YieldOp>(loop.getAfter().front().getTerminator());
                if (!yield || index >= loop.getInits().size() || index >= yield.getResults().size())
                    return false;
                pending.push_back(loop.getInits()[index]);
                pending.push_back(yield.getResults()[index]);
                return true;
            }
            auto condition = dyn_cast<scf::ConditionOp>(loop.getBefore().front().getTerminator());
            if (argument.getOwner() != &loop.getAfter().front() || !condition || index >= condition.getArgs().size())
                return false;
            pending.push_back(condition.getArgs()[index]);
            return true;
        }
        auto loop = dyn_cast_or_null<scf::ForOp>(owner);
        if (!loop || index == 0 || argument.getOwner() != loop.getBody())
            return false;
        return appendLoopSources(loop, index - 1, pending);
    }
    Operation* op = value.getDefiningOp();
    if (!op) {
        return false;
    }
    if (isa<AllocTileOp, AllocMultiTileOp, DeclareTileOp, DeclareGlobalOp>(op)) {
        roots.push_back(value);
        return true;
    }
    if (auto cast = dyn_cast<CastPtrOp>(op)) {
        if (!isa<PtrType>(cast.getInput().getType())) {
            return false;
        }
        pending.push_back(cast.getInput());
        return true;
    }
    if (isa<AddPtrOp, MakeTensorViewOp, PartitionViewOp, SubViewOp, TReshapeOp, BitcastOp, MultiTileGetOp>(op)) {
        pending.push_back(op->getOperand(0));
        return true;
    }
    if (auto select = dyn_cast<arith::SelectOp>(op)) {
        pending.push_back(select.getTrueValue());
        pending.push_back(select.getFalseValue());
        return true;
    }
    unsigned index = cast<OpResult>(value).getResultNumber();
    if (auto loop = dyn_cast<scf::ForOp>(op)) {
        return appendLoopSources(loop, index, pending);
    }
    if (auto loop = dyn_cast<scf::WhileOp>(op)) {
        if (loop.getBefore().empty())
            return false;
        auto condition = dyn_cast<scf::ConditionOp>(loop.getBefore().front().getTerminator());
        if (!condition || index >= condition.getArgs().size())
            return false;
        // Results are the condition's forwarded values even when the AFTER
        // region never executes. Its signature may differ from the inits.
        pending.push_back(condition.getArgs()[index]);
        return true;
    }
    if (auto choice = dyn_cast<scf::IfOp>(op)) {
        if (choice.getThenRegion().empty() || choice.getElseRegion().empty()) {
            return false;
        }
        for (Region* region : {&choice.getThenRegion(), &choice.getElseRegion()}) {
            auto yield = dyn_cast<scf::YieldOp>(region->front().getTerminator());
            if (!yield || index >= yield.getResults().size()) {
                return false;
            }
            pending.push_back(yield.getResults()[index]);
        }
        return true;
    }
    return false;
}
} // namespace

std::optional<InsertSyncGMRange> mlir::pto::traceInsertSyncGMRange(func::FuncOp function, Value value)
{
    constexpr uint64_t limit = INT64_MAX;
    auto add = [&](uint64_t a, uint64_t b, uint64_t& out) {
        if (b > limit - a) return false;
        out = a + b;
        return true;
    };
    auto mul = [&](uint64_t a, uint64_t b, uint64_t& out) {
        if (a && b > limit / a) return false;
        out = a * b;
        return true;
    };
    auto constant = [](Value v, uint64_t& out) {
        APInt integer;
        if (!matchPattern(v, m_ConstantInt(&integer)) || integer.isNegative() || integer.getActiveBits() > 63)
            return false;
        out = integer.getZExtValue();
        return true;
    };
    struct Descriptor {
        Value root;
        uint64_t offset = 0, elementBytes = 0;
        SmallVector<uint64_t> shape, strides;
    };
    std::function<std::optional<Descriptor>(Value, unsigned)> trace;
    trace = [&](Value v, unsigned depth) -> std::optional<Descriptor> {
        if (!v || depth > 32) return {};
        if (auto argument = dyn_cast<BlockArgument>(v)) {
            auto ptr = dyn_cast<PtrType>(v.getType());
            if (!ptr || argument.getOwner() != &function.getBody().front()) return {};
            if (!isa<IntegerType, FloatType>(ptr.getElementType())) return {};
            unsigned bits = ptr.getElementType().getIntOrFloatBitWidth();
            if (!bits || bits % 8) return {};
            return Descriptor{v, 0, bits / 8, {}, {}};
        }
        if (auto cast = v.getDefiningOp<CastPtrOp>()) {
            auto from = dyn_cast<PtrType>(cast.getInput().getType());
            auto to = dyn_cast<PtrType>(v.getType());
            if (!from || !to || from.getElementType() != to.getElementType()) return {};
            return trace(cast.getInput(), depth + 1);
        }
        if (auto pointer = v.getDefiningOp<AddPtrOp>()) {
            auto d = trace(pointer.getPtr(), depth + 1);
            uint64_t offset, bytes;
            if (!d || !constant(pointer.getOffset(), offset) ||
                !mul(offset, d->elementBytes, bytes) || !add(d->offset, bytes, d->offset)) return {};
            return d;
        }
        if (auto view = v.getDefiningOp<MakeTensorViewOp>()) {
            auto type = dyn_cast<TensorViewType>(v.getType());
            if (!type || (type.getLayoutAttr() && type.getLayoutAttr().getLayout() != Layout::ND) ||
                (view.getLayoutAttr() && view.getLayoutAttr().getLayout() != Layout::ND)) return {};
            auto d = trace(view.getPtr(), depth + 1);
            if (!d || view.getShape().size() != view.getStrides().size() || view.getShape().size() > 8) return {};
            for (auto [size, stride] : llvm::zip(view.getShape(), view.getStrides())) {
                uint64_t n, s;
                if (!constant(size, n) || !n || !constant(stride, s) || !s) return {};
                d->shape.push_back(n);
                d->strides.push_back(s);
            }
            return d;
        }
        if (auto part = v.getDefiningOp<PartitionViewOp>()) {
            auto type = cast<PartitionTensorViewType>(v.getType());
            if (type.getLayoutAttr() && type.getLayoutAttr().getLayout() != Layout::ND) return {};
            auto d = trace(part.getSource(), depth + 1);
            if (!d || part.getOffsets().size() != d->shape.size() || part.getSizes().size() != d->shape.size()) return {};
            for (unsigned i = 0; i < d->shape.size(); ++i) {
                uint64_t offset, size, end, elements, bytes;
                if (!constant(part.getOffsets()[i], offset) || !constant(part.getSizes()[i], size) || !size ||
                    !add(offset, size, end) || end > d->shape[i] ||
                    !mul(offset, d->strides[i], elements) || !mul(elements, d->elementBytes, bytes) ||
                    !add(d->offset, bytes, d->offset)) return {};
                d->shape[i] = size;
            }
            return d;
        }
        return {};
    };
    if (!function || function.getBody().empty()) return {};
    auto d = trace(value, 0);
    if (!d || d->shape.empty()) return {};
    uint64_t elements = 1;
    for (unsigned i = d->shape.size(); i-- > 0;) {
        if (d->shape[i] != 1 && d->strides[i] != elements) return {};
        if (!mul(elements, d->shape[i], elements)) return {};
    }
    uint64_t bytes, upper;
    if (!mul(elements, d->elementBytes, bytes) || !add(d->offset, bytes, upper)) return {};
    return InsertSyncGMRange{d->root, d->offset, upper};
}

FailureOr<InsertSyncGMAliasMode> mlir::pto::resolveInsertSyncGMAlias(func::FuncOp function, StringRef overrideMode)
{
    // Distinct SSA arguments are not a noalias promise. Callers may supply one
    // explicitly; it is usable only after a complete argument-root proof.
    StringRef mode = "may-alias";
    if (Attribute attribute = function->getAttr("pto.gm_alias")) {
        auto text = dyn_cast<StringAttr>(attribute);
        if (!text || (text.getValue() != "may-alias" && text.getValue() != "assume-disjoint-arguments")) {
            function.emitError("invalid pto.gm_alias contract");
            return failure();
        }
        mode = text.getValue();
    }
    if (!overrideMode.empty()) {
        mode = overrideMode;
    }
    if (mode != "may-alias" && mode != "assume-disjoint-arguments") {
        function.emitError("InsertSync GM mode must be may-alias or assume-disjoint-arguments");
        return failure();
    }
    return mode == "may-alias" ? InsertSyncGMAliasMode::MayAlias : InsertSyncGMAliasMode::DisjointArguments;
}

InsertSyncMemoryOrigins mlir::pto::traceInsertSyncMemoryOrigins(func::FuncOp function, Value value)
{
    InsertSyncMemoryOrigins result;
    if (!function || function.getBody().empty() || !value)
        return result;
    constexpr uint64_t maxWork = 1u << 18;
    auto charge = [&](uint64_t amount) {
        if (amount > maxWork - result.work)
            return false;
        result.work += amount;
        return true;
    };
    struct Fact {
        SmallVector<unsigned, 2> sources;
        llvm::DenseSet<Value> origins;
        bool unknown = false, unknownRange = false;
    };
    llvm::DenseMap<Value, unsigned> ids;
    SmallVector<Value> values{value};
    llvm::DenseSet<Value> seenRoots;
    ids[value] = 0;
    std::vector<Fact> facts;
    auto finish = [&](bool exhausted) {
        // Every queued value is reachable from the query by a conservative
        // source edge. Retain roots when discovery itself hits the cap; a
        // fixed point is needed only to decide whether the set is complete.
        result.hasUnknown = exhausted || facts.empty() || facts[0].unknown || result.values.empty();
        result.complete = !result.hasUnknown;
        result.preservesRootRange = result.complete && !facts[0].unknownRange;
        return result;
    };
    for (unsigned index = 0; index < values.size(); ++index) {
        if (!charge(1))
            return finish(true);
        SmallVector<Value> sources, roots;
        Fact fact;
        fact.unknown = !appendSources(values[index], function, sources, roots);
        // The native subview verifier does not prove offset+extent is within
        // the backing allocation. Preserve provenance but widen its footprint
        // before extracting any use, including loop-before effects.
        fact.unknownRange = fact.unknown || isa_and_nonnull<SubViewOp>(values[index].getDefiningOp());
        if (!charge(roots.size()))
            return finish(true);
        for (Value root : roots) {
            fact.origins.insert(root);
            if (seenRoots.insert(root).second) {
                if (!charge(1))
                    return finish(true);
                result.values.push_back(root);
            }
        }
        for (Value source : sources) {
            if (!source || !charge(1))
                return finish(true);
            auto found = ids.find(source);
            if (found == ids.end()) {
                if (values.size() == kMaximumRootValues)
                    return finish(true);
                unsigned id = values.size();
                ids[source] = id;
                values.push_back(source);
                fact.sources.push_back(id);
            } else
                fact.sources.push_back(found->second);
        }
        facts.push_back(std::move(fact));
    }
    // Finite monotone origin propagation. Unknown is independent of the
    // retained origins; unresolved closed cycles are marked unknown after the
    // first fixed point, then propagated to their users.
    for (unsigned phase = 0; phase < 2; ++phase) {
        bool changed;
        do {
            changed = false;
            for (auto& fact : facts)
                for (unsigned source : fact.sources) {
                    const auto& from = facts[source];
                    if (!charge(1 + from.origins.size()))
                        return finish(true);
                    if (from.unknown && !fact.unknown)
                        changed = fact.unknown = true;
                    if (from.unknownRange && !fact.unknownRange)
                        changed = fact.unknownRange = true;
                    // A self edge cannot add origins and must not mutate a
                    // DenseSet while iterating over it.
                    if (&fact != &from)
                        for (Value root : from.origins)
                            changed |= fact.origins.insert(root).second;
                }
        } while (changed);
        if (!phase)
            for (auto& fact : facts)
                fact.unknown |= fact.origins.empty();
    }
    // Stable source order makes both cell partitions and reports repeatable.
    return finish(false);
}

InsertSyncGMRoots mlir::pto::traceInsertSyncGMRoots(func::FuncOp function, Value value)
{
    auto origins = traceInsertSyncMemoryOrigins(function, value);
    InsertSyncGMRoots result;
    result.complete = origins.complete;
    for (Value root : origins.values) {
        auto argument = dyn_cast<BlockArgument>(root);
        if (!argument || argument.getOwner() != &function.getBody().front() ||
            !isa<PtrType, TensorViewType, PartitionTensorViewType>(root.getType()))
            result.complete = false;
        else
            result.arguments.push_back(root);
    }
    return result;
}

bool mlir::pto::disjointInsertSyncGMRoots(func::FuncOp function, Value first, Value second)
{
    auto left = traceInsertSyncGMRoots(function, first);
    auto right = traceInsertSyncGMRoots(function, second);
    return left.complete && right.complete &&
           llvm::none_of(left.arguments, [&](Value root) { return llvm::is_contained(right.arguments, root); });
}
