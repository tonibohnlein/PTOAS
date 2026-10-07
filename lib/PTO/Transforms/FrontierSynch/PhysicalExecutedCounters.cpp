// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/PhysicalExecutedCounters.h"
#include "PTO/Transforms/FrontierSynch/ExecutionContexts.h"
#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include <map>
#include <set>
namespace mlir::pto::frontiersynch {
namespace {
struct Family {
    int64_t source = 0, target = 0;
    unsigned width = 0, counter = 0;
    SmallVector<int64_t> ids;
    bool publication = false, consumption = false;
};
using Families = std::map<int64_t, Family>;
std::optional<int64_t> number(DictionaryAttr item, StringRef name)
{
    auto value = item ? item.getAs<IntegerAttr>(name) : IntegerAttr{};
    if (!value || !value.getType().isInteger(64)) { return std::nullopt; }
    return value.getInt();
}
bool pipe(int64_t value)
{
    return value >= 0 && value <= static_cast<int64_t>(PIPE::PIPE_FIX) && value != static_cast<int64_t>(PIPE::PIPE_ALL);
}
FailureOr<Families> decode(func::FuncOp function, DictionaryAttr certificate, ArrayRef<int64_t> eligible)
{
    auto strategy = certificate ? certificate.getAs<StringAttr>("strategy") : StringAttr{};
    auto scope = certificate ? certificate.getAs<StringAttr>("scope") : StringAttr{};
    auto records = certificate ? certificate.getAs<ArrayAttr>("families") : ArrayAttr{};
    auto budget = number(certificate, "total_budget");
    if (!strategy || strategy.getValue() != "executed-family-counters" || !scope || scope.getValue() != "function" ||
        number(certificate, "version") != 2 || !number(certificate, "plan") || *number(certificate, "plan") < 0 ||
        !records || !budget || *budget < 0 || !function.getBody().hasOneBlock() ||
        function->hasAttr(ActiveContextAttr) || hasPhysicalSections(function)) {
        function.emitError("executed arithmetic allocation requires a closed whole-function certificate");
        return failure();
    }
    std::set<int64_t> available;
    for (auto id : eligible) {
        if (id < 0 || id >= 6 || !available.insert(id).second) {
            return function.emitError("executed allocation requires distinct eligible IDs in 0..5"), failure();
        }
    }
    function.walk([&](Operation* op) {
        if (auto model = getSyncMacroModel(op)) {
            for (const auto& hidden : model->hiddenEvents) {
                for (auto id : hidden.eventIds) { available.erase(id); }
            }
        }
    });
    Families result;
    unsigned counters = 0;
    uint64_t total = 0;
    for (auto attribute : records) {
        auto item = dyn_cast<DictionaryAttr>(attribute);
        auto record = number(item, "record"), source = number(item, "source"), target = number(item, "target");
        auto width = number(item, "width");
        if (!record || *record < 0 || !source || !target || !pipe(*source) || !pipe(*target) || *source == *target ||
            !width || *width < 1 || *width > 6 || result.count(*record) ||
            total > INT64_MAX - static_cast<uint64_t>(*width)) {
            return function.emitError("malformed executed arithmetic family allocation"), failure();
        }
        if (available.size() < static_cast<uint64_t>(*width)) {
            return function.emitError("supplied IDs cannot realize the sufficient executed-family palettes"), failure();
        }
        Family family; family.source = *source; family.target = *target; family.width = *width;
        if (family.width > 1) { family.counter = counters; counters += 2; }
        for (unsigned i = 0; i < family.width; ++i) {
            family.ids.push_back(*available.begin()); available.erase(available.begin());
        }
        total += family.width; result.emplace(*record, std::move(family));
    }
    if (total != static_cast<uint64_t>(*budget)) {
        return function.emitError("executed family budget disagrees with its records"), failure();
    }
    return result;
}
LogicalResult preflight(func::FuncOp function, int64_t plan, Families& families)
{
    auto walked = function.walk([&](Operation* op) -> WalkResult {
        if (op == function.getOperation()) { return WalkResult::advance(); }
        const bool publish = isa<LogicalSetOp>(op), consume = isa<LogicalWaitOp>(op);
        if (publish || consume) {
            auto record = op->getAttrOfType<IntegerAttr>("record_id");
            auto identity = op->getAttrOfType<IntegerAttr>("plan_id");
            auto source = op->getAttrOfType<PipeAttr>("src_pipe"), target = op->getAttrOfType<PipeAttr>("dst_pipe");
            auto found = record ? families.find(record.getInt()) : families.end();
            if (found == families.end() || !identity || identity.getInt() != plan || !source || !target ||
                static_cast<int64_t>(source.getPipe()) != found->second.source ||
                static_cast<int64_t>(target.getPipe()) != found->second.target) {
                op->emitError("logical command disagrees with executed-family certificate");
                return WalkResult::interrupt();
            }
            (publish ? found->second.publication : found->second.consumption) = true;
        } else if (isa<SetFlagOp, WaitFlagOp, SetFlagDynOp, WaitFlagDynOp, RecordEventOp, WaitEventOp>(op)) {
            op->emitError("executed allocation cannot share IDs with existing synchronization");
            return WalkResult::interrupt();
        }
        if (op->getNumRegions() && !isa<scf::ForOp, scf::IfOp>(op)) {
            op->emitError("executed allocation does not support this region control flow");
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
    });
    if (walked.wasInterrupted()) { return failure(); }
    for (const auto& [record, family] : families) {
        if (!family.publication || !family.consumption) {
            return function.emitError("executed family lacks a publication or consumption command");
        }
    }
    return success();
}
class Lowering {
public:
    explicit Lowering(const Families& families) : families(families) {}
    void block(Block& block, SmallVector<Value>& counters);
private:
    const Families& families;
    void command(Operation* op, SmallVector<Value>& counters);
    void loop(scf::ForOp op, SmallVector<Value>& counters);
    void branch(scf::IfOp op, SmallVector<Value>& counters);
};
void Lowering::command(Operation* op, SmallVector<Value>& counters)
{
    const auto& family = families.at(op->getAttrOfType<IntegerAttr>("record_id").getInt());
    OpBuilder b(op); auto loc = op->getLoc();
    auto source = op->getAttrOfType<PipeAttr>("src_pipe"), target = op->getAttrOfType<PipeAttr>("dst_pipe");
    const bool publish = isa<LogicalSetOp>(op);
    if (family.width == 1) {
        auto id = EventAttr::get(op->getContext(), *symbolizeEVENT(static_cast<uint32_t>(family.ids.front())));
        if (publish) { b.create<SetFlagOp>(loc, source, target, id); }
        else { b.create<WaitFlagOp>(loc, source, target, id); }
    } else {
        const unsigned index = family.counter + (publish ? 0 : 1);
        Value phase = counters[index];
        auto constant = [&](int64_t value) -> Value { return b.create<arith::ConstantIndexOp>(loc, value); };
        Value id = constant(family.ids.front());
        for (unsigned i = 1; i < family.width; ++i) {
            auto match = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, phase, constant(i));
            id = b.create<arith::SelectOp>(loc, match, constant(family.ids[i]), id);
        }
        if (publish) { b.create<SetFlagDynOp>(loc, source, target, id); }
        else { b.create<WaitFlagDynOp>(loc, source, target, id); }
        auto next = b.create<arith::AddIOp>(loc, phase, constant(1));
        auto wrap = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, next, constant(family.width));
        counters[index] = b.create<arith::SelectOp>(loc, wrap, constant(0), next);
    }
    op->erase();
}
void appendYield(Block& body, ArrayRef<Value> counters)
{
    auto previous = cast<scf::YieldOp>(body.getTerminator());
    SmallVector<Value> values(previous.getOperands()); llvm::append_range(values, counters);
    OpBuilder b(previous); b.create<scf::YieldOp>(previous.getLoc(), values); previous.erase();
}
void copyAttributes(Operation* before, Operation* after)
{
    for (auto attribute : before->getAttrs()) {
        auto name = attribute.getName().getValue();
        if (name == "operandSegmentSizes" || name == "resultSegmentSizes" ||
            name == "operand_segment_sizes" || name == "result_segment_sizes") { continue; }
        after->setAttr(attribute.getName(), attribute.getValue());
    }
}
void Lowering::loop(scf::ForOp op, SmallVector<Value>& counters)
{
    OpBuilder b(op);
    SmallVector<Value> initial(op.getInitArgs()); llvm::append_range(initial, counters);
    auto next = b.create<scf::ForOp>(op.getLoc(), op.getLowerBound(), op.getUpperBound(), op.getStep(), initial);
    copyAttributes(op, next);
    auto* body = next.getBody();
    if (!body->empty()) { body->getTerminator()->erase(); }
    for (auto [before, after] : llvm::zip(op.getBody()->getArguments(), body->getArguments())) {
        before.replaceAllUsesWith(after);
    }
    body->getOperations().splice(body->end(), op.getBody()->getOperations());
    SmallVector<Value> inside(next.getRegionIterArgs().drop_front(op.getNumResults()));
    block(*body, inside); appendYield(*body, inside);
    for (auto [before, after] : llvm::zip(op.getResults(), next.getResults())) { before.replaceAllUsesWith(after); }
    counters.assign(next.getResults().begin() + op.getNumResults(), next.getResults().end());
    op.erase();
}
void Lowering::branch(scf::IfOp op, SmallVector<Value>& counters)
{
    OpBuilder b(op);
    SmallVector<Type> types(op.getResultTypes());
    for (auto value : counters) { types.push_back(value.getType()); }
    auto next = b.create<scf::IfOp>(op.getLoc(), types, op.getCondition(), true);
    copyAttributes(op, next);
    next.getThenRegion().takeBody(op.getThenRegion());
    if (!op.getElseRegion().empty()) { next.getElseRegion().takeBody(op.getElseRegion()); }
    if (next.elseBlock()->empty()) {
        b.setInsertionPointToEnd(next.elseBlock()); b.create<scf::YieldOp>(op.getLoc());
    }
    for (auto* body : {next.thenBlock(), next.elseBlock()}) {
        auto inside = counters; block(*body, inside); appendYield(*body, inside);
    }
    for (auto [before, after] : llvm::zip(op.getResults(), next.getResults())) { before.replaceAllUsesWith(after); }
    counters.assign(next.getResults().begin() + op.getNumResults(), next.getResults().end());
    op.erase();
}
void Lowering::block(Block& body, SmallVector<Value>& counters)
{
    for (auto& operation : llvm::make_early_inc_range(body)) {
        auto* op = &operation;
        if (isa<LogicalSetOp, LogicalWaitOp>(op)) { command(op, counters); }
        else if (auto counted = dyn_cast<scf::ForOp>(op)) { loop(counted, counters); }
        else if (auto conditional = dyn_cast<scf::IfOp>(op)) { branch(conditional, counters); }
    }
}
} // namespace
LogicalResult allocateExecutedFamilyCounters(func::FuncOp function,
    DictionaryAttr certificate, ArrayRef<int64_t> eligibleIds)
{
    if (!function) { return failure(); }
    auto width = DataLayout::closest(function).getTypeSizeInBits(IndexType::get(function.getContext()));
    if (width.isScalable() || width.getFixedValue() != 64) {
        return function.emitError("executed counter allocation requires 64-bit occurrence indices");
    }
    auto families = decode(function, certificate, eligibleIds);
    if (failed(families) || failed(preflight(function, *number(certificate, "plan"), *families))) { return failure(); }
    // ExecutionContexts' FunctionCopy is private. Preserve its symbol-context
    // transaction contract here: the original body is replaced only on success.
    OwningOpRef<ModuleOp> module(ModuleOp::create(function.getLoc()));
    if (auto parent = function->getParentOfType<ModuleOp>()) {
        (*module)->setAttrs(parent->getAttrs());
        for (auto sibling : parent.getOps<func::FuncOp>()) {
            if (sibling != function) { module->push_back(sibling.clone()); }
        }
    }
    auto copy = function.clone(); module->push_back(copy);
    SmallVector<Value> counters;
    OpBuilder b(&copy.front(), copy.front().begin());
    for (const auto& [record, family] : *families) {
        if (family.width > 1) {
            counters.push_back(b.create<arith::ConstantIndexOp>(copy.getLoc(), 0));
            counters.push_back(b.create<arith::ConstantIndexOp>(copy.getLoc(), 0));
        }
    }
    Lowering lowering(*families); lowering.block(copy.front(), counters);
    copy->removeAttr(CyclicAllocationAttr); copy->removeAttr("pto.endpoint_families");
    copy.walk([&](Operation* op) {
        op->removeAttr("pto.endpoint_cut"); op->removeAttr("pto.endpoint_piece"); op->removeAttr("pto.family_loop");
    });
    if (failed(verify(copy))) { return function.emitError("executed counter lowering produced invalid structured IR"); }
    function->setAttrs(copy->getAttrs()); function.getBody().takeBody(copy.getBody());
    return success();
}
} // namespace mlir::pto::frontiersynch
