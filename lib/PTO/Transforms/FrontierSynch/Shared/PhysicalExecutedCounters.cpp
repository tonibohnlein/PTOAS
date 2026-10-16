// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/Shared/PhysicalExecutedCounters.h"
#include "PTO/Transforms/FrontierSynch/Shared/FunctionStaging.h"
#include "PTO/Transforms/FrontierSynch/Shared/ExecutionContexts.h"
#include "PTO/Transforms/FrontierSynch/Shared/FamilyInsertion.h"
#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include <map>
#include <set>
#include <tuple>
namespace mlir::pto::frontiersynch {
namespace {
struct Family {
    int64_t source = 0, target = 0;
    unsigned width = 0, counter = 0;
    SmallVector<int64_t> ids;
    bool publication = false, consumption = false;
};
using Families = std::map<int64_t, Family>;
} // namespace
struct ExecutedFamilyOutput {
    Families families;
    DictionaryAttr certificate;
    std::shared_ptr<const FamilyEndpointRecipe> protocol;
    std::map<std::tuple<Operation*, LogicalCommandKind, uint32_t>, std::pair<uint32_t, uint32_t>> bindings;
};
namespace {
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
    std::set<int64_t> eligibleSet;
    for (auto id : eligible) {
        if (id < 0 || id >= 6 || !eligibleSet.insert(id).second) {
            return function.emitError("executed allocation requires distinct eligible IDs in 0..5"), failure();
        }
    }
    using Direction = std::pair<uint32_t, uint32_t>;
    std::map<Direction, std::set<int64_t>> reservations, availableByDirection;
    function.walk([&](Operation* op) {
        if (auto model = getSyncMacroModel(op)) {
            for (const auto& hidden : model->hiddenEvents) {
                auto& ids = reservations[
                    {static_cast<uint32_t>(hidden.srcPipe), static_cast<uint32_t>(hidden.dstPipe)}];
                ids.insert(hidden.eventIds.begin(), hidden.eventIds.end());
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
        Direction direction{static_cast<uint32_t>(*source), static_cast<uint32_t>(*target)};
        auto entry = availableByDirection.emplace(direction, eligibleSet);
        auto& available = entry.first->second;
        if (entry.second) {
            for (auto id : reservations[direction]) { available.erase(id); }
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
using Commands = llvm::DenseMap<Operation*, int64_t>;
FailureOr<Commands> preflight(func::FuncOp function, int64_t plan, Families& families)
{
    if (function->hasAttr("pto.deferred_family")) {
        return function.emitError("executed allocation requires deferred families to be decoded first"), failure();
    }
    Commands commands;
    if (function->hasAttr("pto.endpoint_families")) {
        PhysicalAllocationPlan records; records.planId = plan;
        for (const auto& [record, family] : families) {
            PhysicalRecordAllocation item;
            item.record = record; item.sourcePipe = family.source; item.targetPipe = family.target;
            // Only validate provenance here. Counter execution owns the full
            // palette; this constant map does not stand in for its assignment.
            item.ids.push_back(family.ids.front()); records.records.push_back(std::move(item));
        }
        auto resolved = resolveExecutedFamilyCommands(function, records);
        if (failed(resolved)) { return failure(); }
        commands = std::move(*resolved);
    }
    auto walked = function.walk([&](Operation* op) -> WalkResult {
        if (op == function.getOperation()) { return WalkResult::advance(); }
        const bool publish = isa<LogicalSetOp>(op), consume = isa<LogicalWaitOp>(op);
        if (publish || consume) {
            auto record = op->getAttrOfType<IntegerAttr>("record_id");
            auto identity = op->getAttrOfType<IntegerAttr>("plan_id");
            auto source = op->getAttrOfType<PipeAttr>("src_pipe"), target = op->getAttrOfType<PipeAttr>("dst_pipe");
            auto resolved = commands.find(op);
            auto original = resolved != commands.end() ? std::optional<int64_t>(resolved->second) :
                (record && record.getType().isInteger(64) ? std::optional<int64_t>(record.getInt()) : std::nullopt);
            auto found = original ? families.find(*original) : families.end();
            if (found == families.end() || !identity || !identity.getType().isInteger(64) ||
                identity.getInt() != plan || !source || !target ||
                static_cast<int64_t>(source.getPipe()) != found->second.source ||
                static_cast<int64_t>(target.getPipe()) != found->second.target) {
                op->emitError("logical command disagrees with executed-family certificate");
                return WalkResult::interrupt();
            }
            commands.try_emplace(op, *original);
            (publish ? found->second.publication : found->second.consumption) = true;
        } else if (isa<LogicalFamilyOp, SetFlagOp, WaitFlagOp, SetFlagDynOp,
                       WaitFlagDynOp, RecordEventOp, WaitEventOp>(op)) {
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
            return function.emitError("executed family lacks a publication or consumption command"), failure();
        }
    }
    return commands;
}
class Lowering {
public:
    Lowering(const Families& families, const Commands& commands) : families(families), commands(commands) {}
    Lowering(const Families& families, const Commands& commands, PreparedLogicalPlan& plan)
        : families(families), commands(commands)
    {
        for (auto& endpoint : plan.endpoints) {
            actions[endpoint.before].push_back(&endpoint);
            bindings.push_back(&endpoint.guard);
            for (auto& box : endpoint.domain) {
                for (auto& constraint : box) { bindings.push_back(&constraint.coordinate); }
            }
        }
    }
    void block(Block& block, SmallVector<Value>& counters);
private:
    const Families& families;
    const Commands& commands;
    std::map<Operation*, SmallVector<const PreparedLogicalEndpoint*>> actions;
    // Update retained handles while old values still exist. No pointer identity
    // from erased SCF operations is kept as a later lookup key.
    SmallVector<Value*> bindings;
    void replace(Value before, Value after)
    {
        for (auto* value : bindings) { if (*value == before) { *value = after; } }
        before.replaceAllUsesWith(after);
    }
    void action(const PreparedLogicalEndpoint& endpoint, SmallVector<Value>& counters);
    void event(OpBuilder& builder, Location location, const Family& family, bool publish,
        SmallVector<Value>& counters);
    void command(Operation* op, SmallVector<Value>& counters);
    void loop(scf::ForOp op, SmallVector<Value>& counters);
    void branch(scf::IfOp op, SmallVector<Value>& counters);
};
void Lowering::command(Operation* op, SmallVector<Value>& counters)
{
    const auto& family = families.at(commands.lookup(op));
    OpBuilder b(op); auto loc = op->getLoc();
    event(b, loc, family, isa<LogicalSetOp>(op), counters);
    op->erase();
}
void Lowering::event(OpBuilder& b, Location loc, const Family& family, bool publish, SmallVector<Value>& counters)
{
    auto source = PipeAttr::get(b.getContext(), static_cast<PIPE>(family.source));
    auto target = PipeAttr::get(b.getContext(), static_cast<PIPE>(family.target));
    if (family.width == 1) {
        auto id = EventAttr::get(b.getContext(), *symbolizeEVENT(static_cast<uint32_t>(family.ids.front())));
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
}
void Lowering::action(const PreparedLogicalEndpoint& endpoint, SmallVector<Value>& counters)
{
    OpBuilder b(endpoint.before); auto loc = endpoint.before->getLoc();
    if (endpoint.kind == LogicalCommandKind::Barrier) {
        auto branch = b.create<scf::IfOp>(loc, endpoint.guard, false);
        b.setInsertionPointToStart(branch.thenBlock());
        b.create<BarrierOp>(loc, PipeAttr::get(b.getContext(), static_cast<PIPE>(endpoint.sourcePipe)));
        return;
    }
    SmallVector<Type> types;
    for (auto value : counters) { types.push_back(value.getType()); }
    auto branch = b.create<scf::IfOp>(loc, types, endpoint.guard, true);
    auto inside = counters;
    if (!branch.thenBlock()->empty()) { branch.thenBlock()->getTerminator()->erase(); }
    b.setInsertionPointToEnd(branch.thenBlock());
    event(b, loc, families.at(endpoint.records.front()), endpoint.kind == LogicalCommandKind::Set, inside);
    b.create<scf::YieldOp>(loc, inside);
    if (!branch.elseBlock()->empty()) { branch.elseBlock()->getTerminator()->erase(); }
    b.setInsertionPointToEnd(branch.elseBlock()); b.create<scf::YieldOp>(loc, counters);
    counters.assign(branch.getResults().begin(), branch.getResults().end());
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
        replace(before, after);
    }
    body->getOperations().splice(body->end(), op.getBody()->getOperations());
    SmallVector<Value> inside(next.getRegionIterArgs().drop_front(op.getNumResults()));
    block(*body, inside); appendYield(*body, inside);
    for (auto [before, after] : llvm::zip(op.getResults(), next.getResults())) { replace(before, after); }
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
    for (auto [before, after] : llvm::zip(op.getResults(), next.getResults())) { replace(before, after); }
    counters.assign(next.getResults().begin() + op.getNumResults(), next.getResults().end());
    op.erase();
}
void Lowering::block(Block& body, SmallVector<Value>& counters)
{
    for (auto& operation : llvm::make_early_inc_range(body)) {
        auto* op = &operation;
        auto before = actions.find(op);
        if (before != actions.end()) {
            OpBuilder builder(op);
            std::map<uint32_t, PreparedLogicalEndpoint> barriers;
            for (const auto* endpoint : before->second) {
                if (endpoint->kind != LogicalCommandKind::Barrier) { continue; }
                auto [entry, added] = barriers.emplace(endpoint->sourcePipe, *endpoint);
                auto guard = endpoint->guard;
                entry->second.guard = added ? guard :
                    Value(builder.create<arith::OrIOp>(op->getLoc(), entry->second.guard, guard));
            }
            for (const auto* endpoint : before->second) {
                if (endpoint->kind == LogicalCommandKind::Set) { action(*endpoint, counters); }
            }
            for (const auto& [pipe, endpoint] : barriers) { action(endpoint, counters); }
            for (const auto* endpoint : before->second) {
                if (endpoint->kind == LogicalCommandKind::Wait) { action(*endpoint, counters); }
            }
        }
        if (isa<LogicalSetOp, LogicalWaitOp>(op)) { command(op, counters); }
        else if (auto counted = dyn_cast<scf::ForOp>(op)) { loop(counted, counters); }
        else if (auto conditional = dyn_cast<scf::IfOp>(op)) { branch(conditional, counters); }
    }
}
} // namespace
LogicalResult planExecutedFamilyOutput(func::FuncOp function, PreparedLogicalPlan& plan, ArrayRef<int64_t> eligibleIds)
{
    if (plan.executedOutput || plan.physicalOutput || !plan.familyProtocol || !plan.familyRecipe ||
        !plan.endpoints.empty() || plan.privateState || plan.completionOwner ||
        !plan.independentPieces || !plan.completeInvocation) { return failure(); }
    const auto width = DataLayout::closest(function).getTypeSizeInBits(IndexType::get(function.getContext()));
    if (width.isScalable() || width.getFixedValue() != 64) { return failure(); }
    auto families = decode(function, plan.allocationCertificate, eligibleIds);
    if (failed(families) || number(plan.allocationCertificate, "plan") != plan.planId) { return failure(); }
    auto owner = std::make_shared<ExecutedFamilyOutput>();
    std::set<uint32_t> records;
    for (const auto& family : plan.families) {
        if (family.members.size() != 1 || family.displacement || family.sourceChoices || family.targetChoices ||
            !family.members.front().sourceCoordinates.empty() || !family.members.front().targetCoordinates.empty()) {
            return function.emitError("executed family output requires singleton original-cut arithmetic families");
        }
        const auto record = family.members.front().record;
        if (!records.insert(record).second) { return failure(); }
        if (!family.local) {
            auto found = families->find(record);
            if (found == families->end() || found->second.source != family.sourcePipe ||
                found->second.target != family.targetPipe) { return failure(); }
        }
        auto add = [&](Operation* cut, LogicalCommandKind kind) {
            return cut && function->isProperAncestor(cut) &&
                owner->bindings.emplace(std::make_tuple(cut, kind, record),
                    std::make_pair(family.sourcePipe, family.targetPipe)).second;
        };
        if (family.local ? !add(family.targetCut.before, LogicalCommandKind::Barrier) :
            (!add(family.sourceCut.before, LogicalCommandKind::Set) ||
             !add(family.targetCut.before, LogicalCommandKind::Wait))) { return failure(); }
    }
    for (const auto& [record, family] : *families) {
        if (!records.count(record)) { return failure(); }
    }
    auto walked = function.walk([&](Operation* op) -> WalkResult {
        if (op == function.getOperation()) { return WalkResult::advance(); }
        if (isa<LogicalSetOp, LogicalWaitOp, LogicalFamilyOp, SetFlagOp, WaitFlagOp,
                SetFlagDynOp, WaitFlagDynOp, RecordEventOp, WaitEventOp>(op) ||
            (op->getNumRegions() && !isa<scf::ForOp, scf::IfOp>(op))) {
            op->emitError("executed family output requires unchanged structured control without existing events");
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
    });
    if (walked.wasInterrupted()) { return failure(); }
    owner->families = std::move(*families); owner->certificate = plan.allocationCertificate;
    owner->protocol = plan.familyProtocol; plan.executedOutput = std::move(owner);
    return success();
}
bool validExecutedFamilyOutput(const PreparedLogicalPlan& plan)
{
    return !plan.executedOutput || (!plan.physicalOutput &&
        plan.executedOutput->certificate == plan.allocationCertificate &&
        plan.executedOutput->protocol == plan.familyProtocol);
}
bool validExecutedFamilyEndpoint(const PreparedLogicalPlan& plan, const PreparedLogicalEndpoint& endpoint)
{
    if (!plan.executedOutput || endpoint.physicalMapping || endpoint.records.size() != 1) { return false; }
    auto found = plan.executedOutput->bindings.find({endpoint.before, endpoint.kind, endpoint.records.front()});
    return found != plan.executedOutput->bindings.end() &&
        found->second == std::make_pair(endpoint.sourcePipe, endpoint.targetPipe);
}
std::optional<uint64_t> executedFamilyCounterCount(const PreparedLogicalPlan& plan)
{
    if (!plan.executedOutput || !validExecutedFamilyOutput(plan)) { return std::nullopt; }
    uint64_t count = 0;
    for (const auto& [record, family] : plan.executedOutput->families) { count += family.width > 1 ? 2 : 0; }
    return count;
}
void emitExecutedFamilyOutput(func::FuncOp function, PreparedLogicalPlan& plan)
{
    const auto& families = plan.executedOutput->families;
    SmallVector<Value> counters;
    OpBuilder b(&function.front(), function.front().begin());
    for (const auto& [record, family] : families) {
        if (family.width > 1) {
            counters.push_back(b.create<arith::ConstantIndexOp>(function.getLoc(), 0));
            counters.push_back(b.create<arith::ConstantIndexOp>(function.getLoc(), 0));
        }
    }
    Commands commands;
    Lowering lowering(families, commands, plan); lowering.block(function.front(), counters);
}
LogicalResult allocateExecutedFamilyCounters(func::FuncOp function,
    DictionaryAttr certificate, ArrayRef<int64_t> eligibleIds)
{
    if (!function) { return failure(); }
    auto width = DataLayout::closest(function).getTypeSizeInBits(IndexType::get(function.getContext()));
    if (width.isScalable() || width.getFixedValue() != 64) {
        return function.emitError("executed counter allocation requires 64-bit occurrence indices");
    }
    auto families = decode(function, certificate, eligibleIds);
    if (failed(families)) { return failure(); }
    auto commands = preflight(function, *number(certificate, "plan"), *families);
    if (failed(commands)) { return failure(); }
    FunctionStaging staging(function);
    if (!staging.function) { return failure(); }
    auto copy = staging.function;
    const auto& mapping = staging.mapping;
    Commands copiedCommands;
    for (const auto& [operation, record] : *commands) { copiedCommands[mapping.lookup(operation)] = record; }
    SmallVector<Value> counters;
    OpBuilder b(&copy.front(), copy.front().begin());
    for (const auto& [record, family] : *families) {
        if (family.width > 1) {
            counters.push_back(b.create<arith::ConstantIndexOp>(copy.getLoc(), 0));
            counters.push_back(b.create<arith::ConstantIndexOp>(copy.getLoc(), 0));
        }
    }
    Lowering lowering(*families, copiedCommands); lowering.block(copy.front(), counters);
    copy->removeAttr(CyclicAllocationAttr); copy->removeAttr("pto.endpoint_families");
    copy.walk([&](Operation* op) {
        op->removeAttr("pto.endpoint_cut"); op->removeAttr("pto.endpoint_piece"); op->removeAttr("pto.family_loop");
    });
    if (failed(verify(copy))) { return function.emitError("executed counter lowering produced invalid structured IR"); }
    staging.commit(function);
    return success();
}
} // namespace mlir::pto::frontiersynch
