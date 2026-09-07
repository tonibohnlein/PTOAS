// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Adapt ProtocolSync's issued/completed scoreboard, not its planner or admission
// model. Action causality separately proves wait consumption before key reuse.
#include "PTO/Transforms/InsertSync/SyncAudit.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOMultiBuffer.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/InsertSync/SyncAuditPaths.h"
#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include <limits>
#include <map>
#include <tuple>

using namespace mlir;
using namespace mlir::pto;

namespace {
using Key = std::tuple<PIPE, PIPE, unsigned>;
struct Access {
    Operation* op;
    uint64_t begin;
    uint64_t end;
    bool write;
    unsigned action;
};
struct Lane {
    llvm::BitVector issued;
    llvm::BitVector completed;
    llvm::BitVector executed;
};
struct Token {
    llvm::BitVector completed;
    llvm::BitVector executed;
};

bool ordinaryPipe(PIPE pipe) { return pipe == PIPE::PIPE_MTE2 || pipe == PIPE::PIPE_V || pipe == PIPE::PIPE_MTE3; }

bool footprint(Value value, uint64_t& begin, uint64_t& end)
{
    auto alloc = value.getDefiningOp<AllocTileOp>();
    auto type = dyn_cast<TileBufType>(value.getType());
    auto space = type ? dyn_cast_or_null<AddressSpaceAttr>(type.getMemorySpace()) : AddressSpaceAttr();
    IntegerAttr address;
    if (!alloc || !type || !space || space.getAddressSpace() != AddressSpace::VEC ||
        !matchPattern(alloc.getAddr(), m_Constant(&address)) || address.getValue().isNegative() ||
        address.getValue().getActiveBits() > 64 ||
        type.getBLayoutValueI32() != static_cast<int32_t>(BLayout::RowMajor) ||
        type.getSLayoutValueI32() != static_cast<int32_t>(SLayout::NoneBox) ||
        type.getCompactModeI32() == static_cast<int32_t>(CompactMode::RowPlusOne)) {
        return false;
    }
    uint64_t size = getPTOStorageElemByteSize(type.getElementType());
    if (!size) {
        return false;
    }
    for (int64_t dimension : type.getShape()) {
        if (dimension <= 0 || size > std::numeric_limits<uint64_t>::max() / dimension) {
            return false;
        }
        size *= dimension;
    }
    begin = address.getValue().getZExtValue();
    if (size > std::numeric_limits<uint64_t>::max() - begin) {
        return false;
    }
    end = begin + size;
    return true;
}

class Scoreboard {
public:
    explicit Scoreboard(unsigned count) : count(count) {}

    InsertSyncAuditResult run(ArrayRef<Operation*> path, Operation* exit)
    {
        for (Operation* op : path) {
            if (auto set = dyn_cast<SetFlagOp>(op)) {
                ++remainingSignals[{
                    set.getSrcPipe().getPipe(), set.getDstPipe().getPipe(),
                    static_cast<unsigned>(set.getEventId().getEvent())}];
            }
        }
        for (Operation* operation : path) {
            Operation& op = *operation;
            ++action;
            if (auto set = dyn_cast<SetFlagOp>(op)) {
                if (!event(
                        op, set.getSrcPipe().getPipe(), set.getDstPipe().getPipe(),
                        static_cast<unsigned>(set.getEventId().getEvent()), true)) {
                    return result;
                }
            } else if (auto wait = dyn_cast<WaitFlagOp>(op)) {
                if (!event(
                        op, wait.getSrcPipe().getPipe(), wait.getDstPipe().getPipe(),
                        static_cast<unsigned>(wait.getEventId().getEvent()), false)) {
                    return result;
                }
            } else if (auto barrier = dyn_cast<BarrierOp>(op)) {
                if (!barrierAction(barrier)) {
                    return result;
                }
            } else if (auto physical = dyn_cast<OpPipeInterface>(op)) {
                if (!issue(op, physical.getPipe())) {
                    return result;
                }
            } else if (!isa<AllocTileOp, func::ReturnOp>(op) && !isMemoryEffectFree(&op)) {
                return {InsertSyncAuditStatus::Unsupported, nullptr, &op, "unmodeled operation effect"};
            }
        }
        if (!tokens.empty()) {
            return {InsertSyncAuditStatus::InvalidToken, nullptr, exit, "unconsumed event at return"};
        }
        if (!accesses.empty() && !drained) {
            return {InsertSyncAuditStatus::Uncovered, accesses.back().op, exit, "outstanding work at return"};
        }
        return {
            InsertSyncAuditStatus::VerifiedLocal, nullptr, nullptr,
            "local conservative bounds and static event lifetimes"};
    }

private:
    Lane& lane(PIPE pipe)
    {
        auto [entry, inserted] = lanes.try_emplace(pipe);
        if (inserted) {
            entry->second = {llvm::BitVector(count), llvm::BitVector(count), llvm::BitVector(count)};
        }
        return entry->second;
    }

    bool fail(InsertSyncAuditStatus status, Operation* source, Operation* target, StringRef reason)
    {
        result = {status, source, target, reason.str()};
        return false;
    }

    bool issue(Operation& op, PIPE pipe)
    {
        if (getSyncMacroModel(&op)) {
            return fail(InsertSyncAuditStatus::Unsupported, nullptr, &op,
                        "multi-phase macro or hidden events require a dedicated audit summary");
        }
        if (!ordinaryPipe(pipe) || drained) {
            return fail(InsertSyncAuditStatus::Unsupported, nullptr, &op, "unsupported pipeline or nonterminal drain");
        }
        auto effects = dyn_cast<MemoryEffectOpInterface>(op);
        if (!effects) {
            return fail(InsertSyncAuditStatus::Unsupported, nullptr, &op, "missing memory-effect interface");
        }
        SmallVector<MemoryEffects::EffectInstance> instances;
        effects.getEffects(instances);
        Lane& current = lane(pipe);
        current.executed.set(action);
        for (auto effect : instances) {
            if (!isa<MemoryEffects::Read, MemoryEffects::Write>(effect.getEffect())) {
                return fail(InsertSyncAuditStatus::Unsupported, nullptr, &op, "non-memory resource effect");
            }
            Value value = effect.getValue();
            // A local-only certificate excludes GM publication. GM inputs/outputs of
            // ordinary tload/tstore are explicitly outside its coverage statement.
            if (value && isa<PtrType, TensorViewType, PartitionTensorViewType>(value.getType())) {
                continue;
            }
            uint64_t begin = 0, end = 0;
            if (!value || !footprint(value, begin, end)) {
                return fail(InsertSyncAuditStatus::Unsupported, nullptr, &op, "unsupported local footprint");
            }
            bool write = isa<MemoryEffects::Write>(effect.getEffect());
            for (const Access& previous : accesses) {
                if (previous.action != action && (write || previous.write) && begin < previous.end &&
                    previous.begin < end && !current.completed.test(previous.action)) {
                    return fail(
                        InsertSyncAuditStatus::Uncovered, previous.op, &op, "overlapping local bounds lack completion");
                }
            }
            accesses.push_back({&op, begin, end, write, action});
        }
        current.issued.set(action);
        return true;
    }

    bool event(Operation& op, PIPE source, PIPE target, unsigned id, bool set)
    {
        if (!ordinaryPipe(source) || !ordinaryPipe(target) || source == target || id >= 8) {
            return fail(InsertSyncAuditStatus::Unsupported, nullptr, &op, "unqualified event domain");
        }
        Key key{source, target, id};
        Lane& current = lane(set ? source : target);
        if (set) {
            --remainingSignals[key];
            auto consumed = lastWait.find(key);
            if (tokens.count(key) || (consumed != lastWait.end() && !current.executed.test(consumed->second))) {
                return fail(
                    InsertSyncAuditStatus::InvalidToken, nullptr, &op, "event consumption before rearm is unproved");
            }
            current.executed.set(action);
            auto completion = current.completed;
            completion |= current.issued;
            tokens.emplace(key, Token{std::move(completion), current.executed});
        } else {
            auto found = tokens.find(key);
            if (found == tokens.end()) {
                if (remainingSignals[key] == 0) {
                    return fail(
                        InsertSyncAuditStatus::InvalidToken, nullptr, &op, "wait has no matching signal on this path");
                }
                return fail(
                    InsertSyncAuditStatus::Unsupported, nullptr, &op,
                    "wait has no preceding signal in supported order");
            }
            current.completed |= found->second.completed;
            current.executed |= found->second.executed;
            current.executed.set(action);
            tokens.erase(found);
            lastWait[key] = action;
        }
        return true;
    }

    bool barrierAction(BarrierOp barrier)
    {
        PIPE pipe = barrier.getPipe().getPipe();
        if (pipe == PIPE::PIPE_ALL) {
            // Only terminal drain: do not invent cross-lane supply from body ALL.
            drained = true;
            return true;
        }
        if (!ordinaryPipe(pipe)) {
            return fail(InsertSyncAuditStatus::Unsupported, nullptr, barrier, "unqualified named barrier");
        }
        Lane& current = lane(pipe);
        current.completed |= current.issued;
        current.executed.set(action);
        return true;
    }

    unsigned count;
    unsigned action = 0;
    bool drained = false;
    std::map<PIPE, Lane> lanes;
    std::map<Key, Token> tokens;
    std::map<Key, unsigned> lastWait;
    std::map<Key, unsigned> remainingSignals;
    SmallVector<Access> accesses;
    InsertSyncAuditResult result;
};
} // namespace

InsertSyncAuditResult mlir::pto::auditInsertSyncLocal(func::FuncOp function)
{
    auto module = function->getParentOfType<ModuleOp>();
    auto arch = module ? module->getAttrOfType<StringAttr>("pto.target_arch") : StringAttr();
    auto core = function->getAttrOfType<FunctionKernelKindAttr>("pto.kernel_kind");
    if (!arch || (arch.getValue() != "a2" && arch.getValue() != "a3") || !core ||
        core.getKernelKind() != FunctionKernelKind::Vector || !llvm::hasSingleElement(function.getBody())) {
        return {
            InsertSyncAuditStatus::Unsupported, nullptr, function,
            "requires explicit A2/A3 vector context and one block"};
    }
    auto paths = buildInsertSyncAuditPaths(function);
    if (!paths.unsupported.empty()) {
        return {InsertSyncAuditStatus::Unsupported, nullptr, paths.witness, paths.unsupported};
    }
    if (paths.paths.size() != paths.pathFeasible.size()) {
        return {InsertSyncAuditStatus::Unsupported, nullptr, function, "audit path metadata mismatch"};
    }
    InsertSyncAuditResult unresolved;
    bool hasUnresolved = false;
    for (auto [path, feasible] : llvm::zip(paths.paths, paths.pathFeasible)) {
        auto result = Scoreboard(path.size() + 1).run(path, function.getBody().front().getTerminator());
        if (result.status == InsertSyncAuditStatus::VerifiedLocal) {
            continue;
        }
        if (feasible && result.status != InsertSyncAuditStatus::Unsupported) {
            return result;
        }
        // A counterexample along an uninterpreted predicate assignment is
        // not yet an executable counterexample. Keep searching exact paths.
        if (!feasible && result.status != InsertSyncAuditStatus::Unsupported) {
            result.status = InsertSyncAuditStatus::Unsupported;
            result.reason = "abstract control path lacks a feasibility witness: " + result.reason;
        }
        if (!hasUnresolved) {
            unresolved = std::move(result);
            hasUnresolved = true;
        }
    }
    if (hasUnresolved) {
        return unresolved;
    }
    return {
        InsertSyncAuditStatus::VerifiedLocal, nullptr, nullptr,
        "all enumerated finite paths: local bounds and event generations; GM excluded"};
}

StringRef mlir::pto::stringifyInsertSyncAuditStatus(InsertSyncAuditStatus status)
{
    switch (status) {
        case InsertSyncAuditStatus::VerifiedLocal:
            return "verified-local";
        case InsertSyncAuditStatus::Uncovered:
            return "uncovered";
        case InsertSyncAuditStatus::InvalidToken:
            return "invalid-token";
        case InsertSyncAuditStatus::Unsupported:
            return "unsupported";
    }
    llvm_unreachable("invalid InsertSync audit status");
}
