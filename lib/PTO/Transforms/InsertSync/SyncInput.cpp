// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/InsertSync/SyncInput.h"
#include "PTO/Transforms/InsertSync/SyncStorageBounds.h"
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"

namespace mlir::pto {
SyncInput::SyncInput(GMAliasPolicy policy)
    : storageBoundsInfo(std::make_unique<SyncStorageBounds>()), aliases(policy) {}
SyncInput::~SyncInput() = default;
const SyncStorageBounds& SyncInput::storageBounds() const { return *storageBoundsInfo; }
LogicalResult SyncInput::build(func::FuncOp function) {
  slotRecords.clear();
  slotIdentities.clear();
  storageBoundsInfo->clear();
  targetInfo.reset();
  phases.clear();
  nodes.clear();
  storage.clear();
  if (function.isDeclaration()) {
    targetInfo.build(function, {}, aliases);
    return success();
  }
  PTOIRTranslator translator(nodes, aliases, storage, function, SyncAnalysisMode::NORMALSYNC);
  if (failed(translator.Build())) {
    return failure();
  }
  for (const auto &node : nodes) {
    if (const auto *instruction = dyn_cast<CompoundInstanceElement>(node.get())) {
      phases.push_back(instruction);
    }
  }
  targetInfo.build(function, phases, aliases);
  storageBoundsInfo->capture(phases);
  captureSlotSelections(function);
  unsigned phaseIndex = 0;
  for (const auto &node : nodes) {
    if (auto *phase = dyn_cast<CompoundInstanceElement>(node.get())) {
      auto core = targetInfo.phases()[phaseIndex++].context.core;
      phase->compoundCoreType = core == SyncPhysicalCore::AIC ? TCoreType::CUBE :
                                core == SyncPhysicalCore::AIV ? TCoreType::VECTOR : TCoreType::CUBE_OR_VECTOR;
    }
  }
  return success();
}
const SyncSlotSelection *SyncInput::slotSelection(Value value) const {
  auto found = slotIdentities.find(value);
  return found == slotIdentities.end() ? nullptr : &slotRecords[found->second];
}
void SyncInput::captureSlotSelections(func::FuncOp function) {
  function.walk([&](MultiTileGetOp op) {
    SyncSlotSelection record;
    record.selected = op.getResult();
    record.slot = op.getSlot();
    record.source = targetInfo.sourceIdentity(op).value_or(-1);
    record.slotSource = targetInfo.sourceIdentity(record.slot.getDefiningOp()).value_or(-1);
    record.slots = op.getSource().getType().getCount();
    auto found = storage.find(op.getResult());
    if (found != storage.end() && found->second.size() == 1) {
      const auto &memory = *found->second.front();
      if (memory.hasKnownPhysicalAddresses && !memory.aliasesUnknownRange &&
          memory.baseAddresses.size() == record.slots) {
        record.addresses = memory.baseAddresses;
        record.boundingSize = memory.allocateSize;
      }
    }
    auto remainder = record.slot.getDefiningOp<arith::RemUIOp>();
    auto loop = op->getParentOfType<scf::ForOp>();
    IntegerAttr modulus, lower, step;
    if (loop && remainder && remainder.getLhs() == loop.getInductionVar() &&
        matchPattern(remainder.getRhs(), m_Constant(&modulus)) &&
        modulus.getValue().isStrictlyPositive() && modulus.getValue().getLimitedValue() == record.slots &&
        matchPattern(loop.getLowerBound(), m_Constant(&lower)) && lower.getValue().isZero() &&
        matchPattern(loop.getStep(), m_Constant(&step)) && step.getValue().isOne()) {
      record.loop = loop;
      record.loopSource = targetInfo.sourceIdentity(loop).value_or(-1);
      record.unitOrdinalModulo = true;
    }
    slotIdentities[record.selected] = slotRecords.size();
    slotRecords.push_back(std::move(record));
  });
}
ArrayAttr SyncInput::slotAttribute(MLIRContext *context) const {
  Builder builder(context);
  SmallVector<Attribute> selections;
  for (const auto &record : slotRecords) {
    SmallVector<Attribute> addresses;
    for (auto address : record.addresses) {
      addresses.push_back(builder.getIntegerAttr(builder.getI64Type(), APInt(64, address)));
    }
    selections.push_back(builder.getDictionaryAttr({
        builder.getNamedAttr("source", builder.getI64IntegerAttr(record.source)),
        builder.getNamedAttr("slot_source", builder.getI64IntegerAttr(record.slotSource)),
        builder.getNamedAttr("loop_source", builder.getI64IntegerAttr(record.loopSource)),
        builder.getNamedAttr("slots", builder.getI64IntegerAttr(record.slots)),
        builder.getNamedAttr("addresses", builder.getArrayAttr(addresses)),
        builder.getNamedAttr("bounding_size",
                             builder.getIntegerAttr(builder.getI64Type(), APInt(64, record.boundingSize))),
        builder.getNamedAttr("unit_ordinal_modulo", builder.getBoolAttr(record.unitOrdinalModulo))}));
  }
  return builder.getArrayAttr(selections);
}
} // namespace mlir::pto
