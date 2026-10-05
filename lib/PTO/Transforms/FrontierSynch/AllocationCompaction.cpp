// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Recombine enumerated endpoint guards into exact boxes with checked modular
// ID formulas. The point set is authoritative: no sampling or widening occurs.
#include "PTO/Transforms/FrontierSynch/PhysicalAllocation.h"
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/CSE.h"
#include "llvm/ADT/MapVector.h"
#include <algorithm>
#include <limits>
namespace mlir::pto::frontiersynch {
namespace {
using Tuple = SmallVector<uint64_t>;
struct Point {
    scf::IfOp branch;
    Operation* command = nullptr;
    const PhysicalRecordAllocation* record = nullptr;
    SmallVector<Value> axes, residual;
    Tuple position;
};
struct Box { Tuple low, high; };
struct Family {
    SmallVector<Point*> points;
    Tuple coefficients;
    SmallVector<Box> boxes;
};
std::optional<int64_t> integer(Value value)
{
    APInt result;
    if (!matchPattern(value, m_ConstantInt(&result)) || !result.isSignedIntN(64)) {
        return std::nullopt;
    }
    return result.getSExtValue();
}
bool coordinate(Value value, Point& point)
{
    auto cmp = value.getDefiningOp<arith::CmpIOp>();
    if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::eq) {
        return false;
    }
    auto arg = dyn_cast<BlockArgument>(cmp.getLhs());
    auto loop = arg ? dyn_cast_or_null<scf::ForOp>(arg.getOwner()->getParentOp()) : scf::ForOp();
    auto expected = integer(cmp.getRhs());
    if (!loop || loop.getInductionVar() != arg || !expected) {
        return false;
    }
    auto lower = integer(loop.getLowerBound()), step = integer(loop.getStep());
    if (!lower || !step || *step <= 0 || *expected < *lower || llvm::is_contained(point.axes, arg)) {
        return false;
    }
    uint64_t offset = static_cast<uint64_t>(*expected) - static_cast<uint64_t>(*lower);
    if (offset % *step || offset / *step > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return false;
    }
    point.axes.push_back(arg);
    point.position.push_back(offset / *step);
    return true;
}
void guard(Value value, Point& point)
{
    if (auto both = value.getDefiningOp<arith::AndIOp>()) {
        guard(both.getLhs(), point);
        guard(both.getRhs(), point);
    } else if (!coordinate(value, point)) {
        point.residual.push_back(value);
    }
}
bool sameFamily(const Point& a, const Point& b)
{
    const auto& x = *a.record;
    const auto& y = *b.record;
    return a.command->getName() == b.command->getName() && a.axes == b.axes && a.residual == b.residual &&
        a.command->getOperand(0) == b.command->getOperand(0) && x.ids == y.ids &&
        x.sourcePipe == y.sourcePipe && x.targetPipe == y.targetPipe && x.stride == y.stride;
}
bool disjoint(const Point& a, const Point& b)
{
    for (std::size_t i = 0; i < a.axes.size(); ++i) {
        for (std::size_t j = 0; j < b.axes.size(); ++j) {
            if (a.axes[i] == b.axes[j] && a.position[i] != b.position[j]) {
                return true;
            }
        }
    }
    return false;
}
bool fit(Family& family)
{
    const auto& base = *family.points.front();
    const auto modulus = base.record->ids.size();
    family.coefficients.assign(base.axes.size(), 0);
    for (std::size_t axis = 0; axis < base.axes.size(); ++axis) {
        for (uint64_t coefficient = 0; coefficient < modulus; ++coefficient) {
            bool valid = true;
            for (const auto* a : family.points) {
                for (const auto* b : family.points) {
                    bool sameOthers = true;
                    for (std::size_t j = 0; j < base.axes.size(); ++j) {
                        sameOthers &= j == axis || a->position[j] == b->position[j];
                    }
                    auto delta = (a->position[axis] % modulus + modulus - b->position[axis] % modulus) % modulus;
                    if (sameOthers && (b->record->phase % modulus + delta * coefficient) % modulus !=
                        a->record->phase % modulus) {
                        valid = false;
                    }
                }
            }
            if (valid) {
                family.coefficients[axis] = coefficient;
                break;
            }
        }
    }
    for (const auto* point : family.points) {
        auto phase = base.record->phase % modulus;
        for (std::size_t j = 0; j < base.axes.size(); ++j) {
            auto delta = (point->position[j] % modulus + modulus - base.position[j] % modulus) % modulus;
            phase = (phase + delta * family.coefficients[j]) % modulus;
        }
        if (phase != point->record->phase % modulus) {
            return false;
        }
    }
    return true;
}
void boxes(Family& family)
{
    for (const auto* point : family.points) {
        family.boxes.push_back({point->position, point->position});
    }
    for (std::size_t axis = 0; axis < family.coefficients.size(); ++axis) {
        auto less = [axis](const Box& a, const Box& b) {
            for (std::size_t j = 0; j < a.low.size(); ++j) {
                if (j != axis && std::tie(a.low[j], a.high[j]) != std::tie(b.low[j], b.high[j])) {
                    return std::tie(a.low[j], a.high[j]) < std::tie(b.low[j], b.high[j]);
                }
            }
            return a.low[axis] < b.low[axis];
        };
        llvm::sort(family.boxes, less);
        SmallVector<Box> merged;
        for (const auto& box : family.boxes) {
            bool adjacent = !merged.empty() && merged.back().high[axis] + 1 == box.low[axis];
            for (std::size_t j = 0; adjacent && j < box.low.size(); ++j) {
                adjacent &= j == axis || (merged.back().low[j] == box.low[j] && merged.back().high[j] == box.high[j]);
            }
            if (adjacent) {
                merged.back().high[axis] = box.high[axis];
            } else {
                merged.push_back(box);
            }
        }
        family.boxes = std::move(merged);
    }
}
// Preserve the order of every pair that can execute together. Mutually exclusive
// coordinate points may change static order. A cycle declines the whole cut.
SmallVector<unsigned> order(ArrayRef<Point> points, ArrayRef<unsigned> owners, unsigned count)
{
    std::vector<std::vector<bool>> edges(count, std::vector<bool>(count, false));
    SmallVector<unsigned> incoming(count, 0), result;
    for (std::size_t i = 0; i < points.size(); ++i) {
        for (std::size_t j = i + 1; j < points.size(); ++j) {
            auto a = owners[i], b = owners[j];
            if (a != b && !disjoint(points[i], points[j]) && !edges[a][b]) {
                edges[a][b] = true;
                ++incoming[b];
            }
        }
    }
    for (unsigned i = 0; i < count; ++i) {
        if (!incoming[i]) {
            result.push_back(i);
        }
    }
    for (std::size_t next = 0; next < result.size(); ++next) {
        for (unsigned j = 0; j < count; ++j) {
            if (edges[result[next]][j] && !--incoming[j]) {
                result.push_back(j);
            }
        }
    }
    return result;
}
Value number(OpBuilder& b, Location loc, uint64_t value)
{
    return b.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(value));
}
SmallVector<Value> coordinates(OpBuilder& b, const Point& point)
{
    SmallVector<Value> result;
    auto loc = point.command->getLoc();
    for (auto axis : point.axes) {
        auto loop = cast<scf::ForOp>(cast<BlockArgument>(axis).getOwner()->getParentOp());
        Value offset = b.create<arith::SubIOp>(loc, axis, loop.getLowerBound());
        Value index = b.create<arith::DivUIOp>(loc, offset, loop.getStep());
        if (!index.getType().isIndex()) {
            index = b.create<arith::IndexCastUIOp>(loc, b.getIndexType(), index);
        }
        result.push_back(index);
    }
    return result;
}
bool fullUpper(Value axis, uint64_t high)
{
    auto loop = cast<scf::ForOp>(cast<BlockArgument>(axis).getOwner()->getParentOp());
    auto low = integer(loop.getLowerBound()), upper = integer(loop.getUpperBound()), step = integer(loop.getStep());
    if (!low || !upper || !step || *step <= 0 || *upper <= *low) {
        return false;
    }
    uint64_t span = static_cast<uint64_t>(*upper) - static_cast<uint64_t>(*low);
    return high == (span - 1) / *step;
}
Value boxGuard(OpBuilder& b, const Point& base, ArrayRef<Value> indices, const Box& box)
{
    auto loc = base.command->getLoc();
    Value condition = b.create<arith::ConstantIntOp>(loc, 1, 1);
    for (std::size_t j = 0; j < indices.size(); ++j) {
        if (box.low[j]) {
            auto low = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge, indices[j], number(b, loc, box.low[j]));
            condition = b.create<arith::AndIOp>(loc, condition, low);
        }
        if (!fullUpper(base.axes[j], box.high[j])) {
            auto high = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ule,
                                               indices[j], number(b, loc, box.high[j]));
            condition = b.create<arith::AndIOp>(loc, condition, high);
        }
    }
    return condition;
}
void emitFamily(OpBuilder& b, const Family& family)
{
    const auto& base = *family.points.front();
    auto loc = base.command->getLoc();
    auto indices = coordinates(b, base);
    auto modulus = base.record->ids.size();
    uint64_t intercept = base.record->phase % modulus;
    for (std::size_t j = 0; j < indices.size(); ++j) {
        auto offset = base.position[j] % modulus * family.coefficients[j] % modulus;
        intercept = (intercept + modulus - offset) % modulus;
    }
    Value phase = number(b, loc, intercept);
    for (std::size_t j = 0; j < indices.size(); ++j) {
        if (family.coefficients[j]) {
            Value term = b.create<arith::RemUIOp>(loc, indices[j], number(b, loc, modulus));
            term = b.create<arith::MulIOp>(loc, term, number(b, loc, family.coefficients[j]));
            phase = b.create<arith::AddIOp>(loc, phase, term);
            phase = b.create<arith::RemUIOp>(loc, phase, number(b, loc, modulus));
        }
    }
    Value condition = b.create<arith::ConstantIntOp>(loc, 0, 1);
    for (const auto& box : family.boxes) {
        condition = b.create<arith::OrIOp>(loc, condition, boxGuard(b, base, indices, box));
    }
    for (auto residual : base.residual) {
        condition = b.create<arith::AndIOp>(loc, condition, residual);
    }
    auto branch = b.create<scf::IfOp>(loc, condition, false);
    OpBuilder::InsertionGuard restore(b);
    b.setInsertionPointToStart(branch.thenBlock());
    emitAllocatedCommand(b, base.command, *base.record, phase);
}
bool compactCut(SmallVector<Point>& points, SmallVectorImpl<Operation*>& erased)
{
    // Do not cross an unrepresented command (notably a local barrier).
    for (auto* op = points.front().branch.getOperation(); op != points.back().branch.getOperation();
         op = op->getNextNode()) {
        if (!op || (!llvm::any_of(points, [op](const Point& p) { return p.branch == op; }) &&
                    op->getName().getDialectNamespace() != "arith")) {
            return false;
        }
    }
    std::vector<Family> families;
    SmallVector<unsigned> owners;
    for (auto& point : points) {
        unsigned selected = 0;
        while (selected < families.size() && !sameFamily(*families[selected].points.front(), point)) {
            ++selected;
        }
        if (selected == families.size()) {
            families.push_back({});
        }
        for (auto* prior : families[selected].points) {
            if (!disjoint(*prior, point)) {
                return false; // Never collapse two simultaneously executed commands.
            }
        }
        families[selected].points.push_back(&point);
        owners.push_back(selected);
    }
    std::size_t count = 0;
    for (auto& family : families) {
        if (!fit(family)) {
            return false;
        }
        boxes(family);
        count += family.boxes.size();
    }
    if (count >= points.size()) {
        return false;
    }
    auto ordered = order(points, owners, families.size());
    if (ordered.size() != families.size()) {
        return false;
    }
    OpBuilder builder(points.back().branch);
    for (auto index : ordered) {
        emitFamily(builder, families[index]);
    }
    for (auto& point : points) {
        erased.push_back(point.command);
        point.branch.erase();
    }
    return true;
}
} // namespace
SmallVector<Operation*> compactAllocatedEndpoints(func::FuncOp function, const PhysicalAllocationPlan& plan)
{
    // Share arithmetic before comparing residual guards and source ordinals.
    IRRewriter rewriter(function.getContext());
    DominanceInfo dominance(function);
    eliminateCommonSubExpressions(rewriter, dominance, function);
    llvm::DenseMap<int64_t, const PhysicalRecordAllocation*> records;
    for (const auto& record : plan.records) {
        records[record.record] = &record;
    }
    llvm::MapVector<std::pair<Block*, int64_t>, SmallVector<Point>> cuts;
    function.walk([&](scf::IfOp branch) {
        auto cut = branch->getAttrOfType<IntegerAttr>("pto.endpoint_cut");
        if (!cut || !branch.getElseRegion().empty() || branch.thenBlock()->getOperations().size() != 2) {
            return;
        }
        auto* command = &branch.thenBlock()->front();
        if (!isa<LogicalSetOp, LogicalWaitOp>(command)) {
            return;
        }
        auto* record = records.lookup(command->getAttrOfType<IntegerAttr>("record_id").getInt());
        Point point{branch, command, record, {}, {}, {}};
        guard(branch.getCondition(), point);
        cuts[{branch->getBlock(), cut.getInt()}].push_back(std::move(point));
    });
    SmallVector<Operation*> erased;
    for (auto& entry : cuts) {
        compactCut(entry.second, erased);
    }
    return erased;
}
} // namespace mlir::pto::frontiersynch
