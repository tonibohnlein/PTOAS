// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp. Source contract: pto-isa 0c112d61f41342bd0867ce1080c29f1590d72484,
// npu/a2a3/{TPush,TPop,TFree}.hpp and PTOToEmitC::buildTPipeToken.
// Tile-entry TPOP loads through MTE2 and returns GM credit there. Tile TFREE is
// a no-op on this path; it must not be modeled as a release of the local tile.
std::optional<mlir::pto::SyncProtocolModel>
mlir::pto::getSyncProtocolModel(Operation *op) {
  SyncProtocolModel model;
  if (auto collective = dyn_cast<SyncAllOp>(op)) {
    model.kind = SyncProtocolModel::Collective;
    model.participants = collective.getCoreType().getValue();
    if (getTargetArch(op) != PTOArch::A3) {
      model.gap = "collective requires the qualified A3 lowering";
      return model;
    }
    if (collective.getMode().getValue() != SyncAllMode::Hard) {
      model.gap = "soft collective requires scratch, visibility, and private-event contracts";
      return model;
    }
    if (collective.getGmWorkspace() || collective.getUsedCores()) {
      model.gap = "hard collective must not have workspace or participant-count operands";
      return model;
    }
    // pto-isa 0c112d61, a2a3/SyncAll.hpp::SYNCALL_IMPL and
    // common/type.hpp. Hard SYNCALL has no payload/scratch byte accesses or
    // directional local keys. Preserve the intrinsic's ALL and cross-core
    // handshake; do not turn its device flags into local event reservations.
    model.contract = "a3-hard-collective-v1/pto-isa-0c112d61";
    model.localDrainBefore = true;
    switch (*model.participants) {
    case SyncCoreType::AIVOnly:
      model.crossCoreFlagBase = 14;
      model.crossCoreFlagCount = 1;
      break;
    case SyncCoreType::AICOnly:
      model.crossCoreFlagBase = 11;
      model.crossCoreFlagCount = 1;
      break;
    case SyncCoreType::Mix:
      model.crossCoreFlagBase = 11;
      model.crossCoreFlagCount = 3;
      break;
    }
    return model;
  }
  InitializeL2G2LPipeOp init;
  if (auto value = dyn_cast<InitializeL2G2LPipeOp>(op)) {
    init = value;
    model.handle = value.getPipe();
  } else if (auto value = dyn_cast<TPushOp>(op)) {
    model.kind = SyncProtocolModel::Send;
    model.handle = value.getPipeHandle();
    model.tile = value.getTile();
    model.pipeline = value.getPipe();
  } else if (auto value = dyn_cast<TPopOp>(op)) {
    model.kind = SyncProtocolModel::Receive;
    model.handle = value.getPipeHandle();
    model.tile = value.getTile();
    model.pipeline = value.getPipe();
  } else if (auto value = dyn_cast<TFreeOp>(op)) {
    model.kind = SyncProtocolModel::Release;
    model.handle = value.getPipeHandle();
    if (value.getEntry()) {
      model.gap = "global-entry release requires its ownership boundary contract";
      return model;
    }
  } else {
    return std::nullopt;
  }
  if (!init) init = model.handle.getDefiningOp<InitializeL2G2LPipeOp>();
  if (!init || getTargetArch(op) != PTOArch::A3) {
    model.gap = "queue requires the qualified A3 GM-backed tile-entry lowering";
    return model;
  }
  model.contract = "a3-gm-tile-fifo-v1/pto-isa-0c112d61";
  if (init.getAccPushEpilogueAttr()) {
    model.gap = "queue epilogue requires an additional phase/quantization contract";
    return model;
  }
  // The EmitC path leaves EN_UNIT_FLAG at its false template default.
  auto flags = init.getFlagBaseAttr();
  const auto direction = init.getDirMask();
  model.localSlots = init.getLocalSlotNumAttr()
      ? unsigned(init.getLocalSlotNumAttr().getInt()) : unsigned(init.getSlotNum());
  if (!flags || flags.getInt() < 0 || direction < 1 || direction > 3 ||
      init.getSlotNum() <= 0 || init.getSlotSize() <= 0 ||
      !model.localSlots || model.localSlots > unsigned(init.getSlotNum())) {
    model.gap = "queue initialization must be resolved and validated before synchronization";
    return model;
  }
  model.crossCoreFlagBase = unsigned(flags.getInt());
  model.crossCoreFlagCount = direction == 3 ? 4 : 2;
  if (model.crossCoreFlagBase + model.crossCoreFlagCount > 16) {
    model.gap = "queue cross-core flag reservation exceeds the source profile";
    return model;
  }
  // A handle's entry family must be uniform. Global entries use a different
  // TALLOC/commit/release protocol; never infer tile semantics from one user.
  for (Operation *user : model.handle.getUsers()) {
    Value tile;
    bool qualifiedSplit = true;
    if (auto push = dyn_cast<TPushOp>(user)) {
      tile = push.getTile();
      qualifiedSplit = push.getSplit() <= 2 && !push.getAivSubblockid();
    } else if (auto pop = dyn_cast<TPopOp>(user)) {
      tile = pop.getTile();
      qualifiedSplit = pop.getSplit() <= 2 && !pop.getAivSubblockid();
    }
    else if (auto release = dyn_cast<TFreeOp>(user)) {
      if (!release.getEntry()) continue;
    }
    if (!qualifiedSplit) {
      model.gap = "queue split/subblock overload lacks a source-qualified contract";
      return model;
    }
    if (!tile || !isa<TileBufType>(tile.getType())) {
      model.gap = "queue has global-entry or unrepresented protocol users";
      return model;
    }
  }
  model.globalStorage = init.getGmAddr();
  if (model.kind == SyncProtocolModel::Initialize ||
      model.kind == SyncProtocolModel::Release)
    return model;
  auto type = dyn_cast<TileBufType>(model.tile.getType());
  if (!type) {
    model.gap = "queue requires a tile-entry byte-effect contract";
    return model;
  }
  // TILE_NO_SPLIT has no sub-AIV offset and uses a dense GM tile rectangle.
  // Retain a whole-slot may-footprint (no definite-write assertion). Mutable
  // valid dimensions obey the original tile bounds, as for local footprints.
  const bool unsplit = isa<TPushOp>(op) ? cast<TPushOp>(op).getSplit() == 0
                                      : cast<TPopOp>(op).getSplit() == 0;
  const auto space = cast<AddressSpaceAttr>(type.getMemorySpace()).getAddressSpace();
  if (direction == 3 && init.getSlotNum() == 2 && unsplit &&
      space == AddressSpace::VEC && init.getNosplitAttr() &&
      init.getNosplitAttr().getValue()) {
    const auto bits = type.getElementType().getIntOrFloatBitWidth();
    uint64_t bytes = bits / 8;
    bool fits = bits && bits % 8 == 0 && type.getShape().size() == 2;
    for (auto dimension : type.getShape()) {
      if (dimension <= 0 || uint64_t(dimension) > uint64_t(init.getSlotSize()) / std::max(uint64_t(1), bytes)) {
        fits = false;
        break;
      }
      bytes *= uint64_t(dimension);
    }
    if (fits && bytes <= uint64_t(init.getSlotSize())) {
      model.globalSlots = 2;
      model.globalSlotBytes = uint64_t(init.getSlotSize());
    }
  }
  if (model.kind == SyncProtocolModel::Send) {
    if (model.pipeline != PIPE::PIPE_FIX && model.pipeline != PIPE::PIPE_MTE3) {
      model.gap = "queue producer has no qualified physical pipeline";
      return model;
    }
    model.reads.push_back(model.tile);
    model.writes.push_back(model.globalStorage);
  } else {
    // EmitC maps the first local operand to C2V (VEC) and, for BOTH,
    // the peer operand to V2C (MAT). Absent operands lower to address zero.
    auto space = cast<AddressSpaceAttr>(type.getMemorySpace()).getAddressSpace();
    model.localBase = direction == 3 && space == AddressSpace::MAT
                          ? init.getPeerLocalAddr() : init.getLocalAddr();
    if (model.pipeline != PIPE::PIPE_MTE2) {
      model.gap = "queue consumer has no qualified local slot/pipeline binding";
      return model;
    }
    model.reads.push_back(model.globalStorage);
    model.writes.push_back(model.tile);
  }
  return model;
}
