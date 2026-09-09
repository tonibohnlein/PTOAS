// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
// See LICENSE for details. Provided AS IS, WITHOUT WARRANTIES OF ANY KIND.
#ifndef PTO_TRANSFORMS_INSERTSYNC_SYNCPAYLOADSNAPSHOT_H
#define PTO_TRANSFORMS_INSERTSYNC_SYNCPAYLOADSNAPSHOT_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <vector>

namespace mlir::pto {
// Synchronization construction may add operations, but must preserve the
// scheduled program it was given. Effect equality alone cannot detect changed
// rounding, scalar operands, block ownership or independent payload reordering.
class SyncPayloadSnapshot {
    struct Original {
        Operation* operation;
        Block* block;
        DictionaryAttr attributes;
        SmallVector<Value> operands;
        SmallVector<Type> results;
        SmallVector<Block*> successors;
    };
    struct OriginalBlock {
        Block* block;
        Region* region;
        SmallVector<Value> arguments;
        SmallVector<Type> types;
    };
    std::vector<Original> operations;
    std::vector<OriginalBlock> blocks;

public:
    explicit SyncPayloadSnapshot(func::FuncOp function)
    {
        function.walk<WalkOrder::PreOrder>([&](Operation* op) {
            operations.push_back(
                {op, op->getBlock(), op->getAttrDictionary(), SmallVector<Value>(op->getOperands()),
                 SmallVector<Type>(op->getResultTypes()), SmallVector<Block*>(op->getSuccessors())});
            for (Region& region : op->getRegions())
                for (Block& block : region)
                    blocks.push_back(
                        {&block, &region, SmallVector<Value>(block.getArguments()),
                         SmallVector<Type>(block.getArgumentTypes())});
        });
    }
    bool preserved(func::FuncOp function, llvm::function_ref<bool(Operation*)> allowedAddition) const
    {
        llvm::SmallPtrSet<Operation*, 32> original;
        for (const auto& record : operations)
            original.insert(record.operation);
        std::vector<Operation*> now;
        bool additionsValid = true;
        function.walk<WalkOrder::PreOrder>([&](Operation* op) {
            if (original.contains(op))
                now.push_back(op);
            else if (!allowedAddition(op))
                additionsValid = false;
        });
        if (!additionsValid || now.size() != operations.size())
            return false;
        for (unsigned i = 0; i < now.size(); ++i) {
            const auto& record = operations[i];
            auto* op = now[i];
            if (op != record.operation || op->getBlock() != record.block ||
                op->getAttrDictionary() != record.attributes || !llvm::equal(op->getOperands(), record.operands) ||
                !llvm::equal(op->getResultTypes(), record.results) ||
                !llvm::equal(op->getSuccessors(), record.successors))
                return false;
        }
        // Original operations still exist, so their original regions/blocks
        // must also exist before these identities can be queried.
        llvm::SmallPtrSet<Block*, 32> currentBlocks;
        function.walk([&](Operation* op) {
            for (Region& region : op->getRegions())
                for (Block& block : region)
                    currentBlocks.insert(&block);
        });
        for (const auto& record : blocks)
            if (!currentBlocks.contains(record.block) || record.block->getParent() != record.region ||
                !llvm::equal(record.block->getArguments(), record.arguments) ||
                !llvm::equal(record.block->getArgumentTypes(), record.types))
                return false;
        return true;
    }
};
} // namespace mlir::pto
#endif
