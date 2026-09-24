// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_SYNCTILEDESCRIPTORSTATE_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_SYNCTILEDESCRIPTORSTATE_H

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/FrontierSynch/SyncSlotMapping.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <array>
#include <optional>

namespace mlir::pto {
// Effective descriptor dimensions at original uses. Descriptor state is distinct
// from storage aliasing: two views can share bytes but own separate metadata.
// Branches retain equal reaching values; loops havoc only mutated descriptors,
// then local definitions/updates can establish dimensions inside the body.
// Unknown control or forwarding never manufactures a known dynamic dimension.
class SyncTileDescriptorState {
public:
  using Dimensions = std::array<uint64_t, 2>;
  using Shape = std::optional<Dimensions>;

  explicit SyncTileDescriptorState(func::FuncOp function) {
    State state;
    // Entry descriptors must participate in branch joins just like local ones.
    // Otherwise a conditional mutation of an argument disappears at the join.
    for (Value argument : function.getArguments()) {
      if (isa<TileBufType>(argument.getType())) {
        state[argument] = initial(argument);
      }
    }
    visit(function.getBody(), state);
  }

  Shape at(Value value, mlir::Operation *operation) const {
    auto found = uses.find(operation);
    if (found == uses.end()) {
      return {};
    }
    auto shape = found->second.find(value);
    return shape == found->second.end() ? Shape{} : shape->second;
  }

private:
  using State = llvm::DenseMap<Value, Shape>;
  llvm::DenseMap<mlir::Operation *, State> uses;
  SyncSlotMapping::ConstantCache constants;

  Shape initial(Value value, Value row = {}, Value col = {}) {
    auto type = dyn_cast<TileBufType>(value.getType());
    if (!type || type.getShape().size() != 2 || type.getValidShape().size() != 2) {
      return {};
    }
    Dimensions result{};
    for (unsigned dim = 0; dim < 2; ++dim) {
      const Value explicitSize = dim ? col : row;
      const auto size = explicitSize ? SyncSlotMapping::evaluateConstant(explicitSize, constants) :
          (type.getValidShape()[dim] >= 0 ? ShapeScalar(type.getValidShape()[dim]) : std::nullopt);
      if (!size || type.getShape()[dim] < 0 || *size > uint64_t(type.getShape()[dim])) {
        return {};
      }
      result[dim] = *size;
    }
    return result;
  }
  using ShapeScalar = std::optional<uint64_t>;

  Shape lookup(Value value, const State &state) {
    auto found = state.find(value);
    return found == state.end() ? initial(value) : found->second;
  }

  void invalidateMutations(mlir::Operation &operation, State &state) {
    operation.walk([&state](SetValidShapeOp update) { state[update.getSource()] = {}; });
  }

  void visit(mlir::Region &region, State &state) {
    if (!region.hasOneBlock()) {
      for (Block &block : region) {
        for (mlir::Operation &operation : block) {
          invalidateMutations(operation, state);
        }
      }
      return;
    }
    for (mlir::Operation &operation : region.front()) {
      for (Value operand : operation.getOperands()) {
        if (isa<TileBufType>(operand.getType())) {
          uses[&operation][operand] = lookup(operand, state);
        }
      }
      if (auto alloc = dyn_cast<AllocTileOp>(operation)) {
        state[alloc.getResult()] = initial(alloc.getResult(), alloc.getValidRow(), alloc.getValidCol());
      } else if (auto view = dyn_cast<SubViewOp>(operation)) {
        state[view.getResult()] = initial(view.getResult(), view.getValidRow(), view.getValidCol());
      } else if (auto update = dyn_cast<SetValidShapeOp>(operation)) {
        state[update.getSource()] = initial(update.getSource(), update.getValidRow(), update.getValidCol());
      } else if (auto choice = dyn_cast<scf::IfOp>(operation)) {
        State left = state, right = state;
        visit(choice.getThenRegion(), left);
        visit(choice.getElseRegion(), right);
        for (auto &entry : state) {
          entry.second = lookup(entry.first, left) == lookup(entry.first, right) ?
              lookup(entry.first, left) : Shape{};
        }
      } else if (operation.getNumRegions()) {
        invalidateMutations(operation, state);
        State body = state;
        for (mlir::Region &child : operation.getRegions()) {
          visit(child, body);
        }
      }
      for (Value result : operation.getResults()) {
        if (isa<TileBufType>(result.getType()) && !state.count(result)) {
          state[result] = initial(result);
        }
      }
    }
  }
};
} // namespace mlir::pto
#endif
