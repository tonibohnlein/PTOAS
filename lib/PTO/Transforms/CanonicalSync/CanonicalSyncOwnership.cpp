// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "CanonicalSyncInternal.h"

#include "PTO/Transforms/SlotAffineAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::canonical_sync_detail;

namespace {

constexpr std::uint32_t kReadyReleaseDepth = 2;
constexpr std::uint32_t kReadyReleaseReuseDistance = 2;
constexpr std::uint32_t kReadyReleaseWitnessHorizon = 4;

bool isExactTwoSlotInterval(ArrayRef<CanonicalByteInterval> intervals) {
  if (intervals.size() != kReadyReleaseDepth || intervals[0].size == 0 ||
      intervals[0].size != intervals[1].size) {
    return false;
  }
  const std::optional<std::uint64_t> firstEnd = intervals[0].end();
  const std::optional<std::uint64_t> secondEnd = intervals[1].end();
  return firstEnd && secondEnd &&
         (*firstEnd <= intervals[1].begin || *secondEnd <= intervals[0].begin);
}

bool sameIntervals(ArrayRef<CanonicalByteInterval> first,
                   ArrayRef<CanonicalByteInterval> second) {
  if (first.size() != second.size()) {
    return false;
  }
  for (size_t index = 0; index < first.size(); ++index) {
    if (first[index].begin != second[index].begin ||
        first[index].size != second[index].size) {
      return false;
    }
  }
  return true;
}

bool hasRepeatingAncestor(scf::ForOp loop) {
  for (Operation *parent = loop->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isa<scf::ForOp, scf::WhileOp>(parent)) {
      return true;
    }
  }
  return false;
}

std::optional<CanonicalStorageGeneration>
buildGeneration(const CanonicalSyncProgram &program,
                const CanonicalSyncTarget &target, Value storage,
                ArrayRef<CanonicalAccessId> accessIds) {
  auto type = dyn_cast<MultiTileBufType>(storage.getType());
  if (!type || type.getCount() != kReadyReleaseDepth ||
      accessIds.size() != 2) {
    return std::nullopt;
  }

  const CanonicalAccess *producerAccess = nullptr;
  const CanonicalAccess *consumerAccess = nullptr;
  for (CanonicalAccessId id : accessIds) {
    const CanonicalAccess &access = program.getAccess(id);
    if (access.mode == CanonicalAccessMode::Write && !producerAccess) {
      producerAccess = &access;
    } else if (access.mode == CanonicalAccessMode::Read && !consumerAccess) {
      consumerAccess = &access;
    } else {
      return std::nullopt;
    }
  }
  if (!producerAccess || !consumerAccess || !producerAccess->physical ||
      !consumerAccess->physical || producerAccess->unknownRange ||
      consumerAccess->unknownRange ||
      !isExactTwoSlotInterval(producerAccess->intervals) ||
      !sameIntervals(producerAccess->intervals, consumerAccess->intervals)) {
    return std::nullopt;
  }

  const CanonicalPhase &producer = program.getPhase(producerAccess->phase);
  const CanonicalPhase &consumer = program.getPhase(consumerAccess->phase);
  const bool exactControl = producer.controlPath.empty() &&
                            consumer.controlPath.empty() &&
                            producer.loopPath.size() == 1 &&
                            producer.loopPath == consumer.loopPath;
  if (!exactControl || producer.resource.core != consumer.resource.core ||
      producer.resource == consumer.resource ||
      producer.operation->getBlock() != consumer.operation->getBlock() ||
      !producer.operation->isBeforeInBlock(consumer.operation) ||
      !target.supportsEvent(producer.resource, consumer.resource) ||
      !target.supportsEvent(consumer.resource, producer.resource)) {
    return std::nullopt;
  }

  const CanonicalRegionId loopId = producer.loopPath.front();
  if (loopId >= program.getRegions().size()) {
    return std::nullopt;
  }
  auto loop = dyn_cast_or_null<scf::ForOp>(program.getRegion(loopId).operation);
  const bool exactLoop = loop && !hasRepeatingAncestor(loop) &&
                         producer.operation->getBlock() == loop.getBody();
  if (!exactLoop || !producerAccess->slotExpression ||
      producerAccess->slotExpression != consumerAccess->slotExpression) {
    return std::nullopt;
  }
  const std::optional<std::uint32_t> slotOffset =
      matchUnitStrideModuloSlot(producerAccess->slotExpression, loop,
                                kReadyReleaseDepth);
  if (!slotOffset) {
    return std::nullopt;
  }

  CanonicalStorageGeneration generation;
  generation.storage = storage;
  generation.space = producerAccess->space;
  generation.loop = loopId;
  generation.producer = producer.resource;
  generation.consumer = consumer.resource;
  generation.producerAccess = producerAccess->id;
  generation.consumerAccess = consumerAccess->id;
  generation.writeAcquire = {producer.operation,
                             CanonicalProgramPointPosition::Before};
  generation.ready = {producer.operation,
                      CanonicalProgramPointPosition::After};
  generation.readAcquire = {consumer.operation,
                            CanonicalProgramPointPosition::Before};
  generation.lastUse = {consumer.operation,
                        CanonicalProgramPointPosition::After};
  generation.slotExpression = producerAccess->slotExpression;
  generation.staticDepth = kReadyReleaseDepth;
  generation.slotOffset = *slotOffset;
  generation.reuseDistance = kReadyReleaseReuseDistance;
  generation.witnessHorizon = kReadyReleaseWitnessHorizon;
  return generation;
}

bool hasDistance(const CanonicalDemand &demand, CanonicalRegionId loop,
                 CanonicalIterationRelation relation) {
  return llvm::any_of(demand.iterationDistance,
                      [&](const CanonicalLoopDistance &distance) {
                        return distance.loop == loop &&
                               distance.relation == relation;
                      });
}

bool matchesCause(const CanonicalDemandCause &cause,
                  CanonicalAccessId source, CanonicalAccessId target) {
  return cause.sourceAccess == source && cause.targetAccess == target;
}

} // namespace

LogicalResult
mlir::pto::canonical_sync_detail::deriveCanonicalStorageGenerations(
    CanonicalSyncProgram &program, const CanonicalSyncTarget &target,
    CanonicalOwnershipPlanning ownershipPlanning) {
  if (ownershipPlanning == CanonicalOwnershipPlanning::Disabled) {
    return success();
  }

  DenseMap<Value, SmallVector<CanonicalAccessId, 4>> accessesByRoot;
  for (const CanonicalAccess &access : program.getAccesses()) {
    const bool local = access.aliasRoot && !access.unknownSpace &&
                       access.space != AddressSpace::GM &&
                       access.space != AddressSpace::Zero;
    if (local && isa<MultiTileBufType>(access.aliasRoot.getType())) {
      accessesByRoot[access.aliasRoot].push_back(access.id);
    }
  }
  for (const auto &entry : accessesByRoot) {
    std::optional<CanonicalStorageGeneration> generation =
        buildGeneration(program, target, entry.first, entry.second);
    if (generation) {
      program.appendStorageGeneration(std::move(*generation));
    }
  }
  if (CanonicalSyncStatistics *statistics = program.getStatistics()) {
    statistics->storageGenerations = program.getStorageGenerations().size();
  }
  return success();
}

void mlir::pto::canonical_sync_detail::attachCanonicalOwnershipDemandEdges(
    CanonicalSyncProgram &program) {
  CanonicalSyncStatistics *statistics = program.getStatistics();
  for (const CanonicalStorageGeneration &generation :
       program.getStorageGenerations()) {
    CanonicalOwnershipChannel channel;
    channel.generation = generation.id;
    channel.storage = generation.storage;
    channel.space = generation.space;
    channel.loop = generation.loop;
    channel.producer = generation.producer;
    channel.consumer = generation.consumer;
    channel.staticDepth = generation.staticDepth;
    channel.slotTracked = true;

    for (const CanonicalDemand &demand : program.getDemands()) {
      for (const CanonicalDemandCause &cause : demand.causes) {
        const bool ready = demand.kind == CanonicalDemandKind::Raw &&
                           hasDistance(demand, generation.loop,
                                       CanonicalIterationRelation::Same) &&
                           matchesCause(cause, generation.producerAccess,
                                        generation.consumerAccess);
        const bool release = demand.kind == CanonicalDemandKind::War &&
                             hasDistance(
                                 demand, generation.loop,
                                 CanonicalIterationRelation::AnyPositive) &&
                             matchesCause(cause, generation.consumerAccess,
                                          generation.producerAccess);
        if (ready) {
          channel.readyEdges.push_back(
              {demand.id, cause.sourceAccess, cause.targetAccess});
        }
        if (release) {
          channel.releaseEdges.push_back(
              {demand.id, cause.sourceAccess, cause.targetAccess});
        }
      }
    }
    channel.readyRelease2Eligible = !channel.readyEdges.empty() &&
                                    !channel.releaseEdges.empty();
    if (!channel.readyRelease2Eligible) {
      continue;
    }
    if (statistics) {
      ++statistics->ownershipChannels;
      statistics->ownershipReadyEdges += channel.readyEdges.size();
      statistics->ownershipReleaseEdges += channel.releaseEdges.size();
      ++statistics->depthTwoOwnershipChannels;
      ++statistics->slotTrackedOwnershipChannels;
    }
    program.appendOwnershipChannel(std::move(channel));
  }
}

bool mlir::pto::canonical_sync_detail::canonicalOwnershipChannelCoversDemand(
    const CanonicalSyncProgram &program,
    const CanonicalOwnershipChannel &channel, const CanonicalDemand &demand) {
  if (!channel.readyRelease2Eligible ||
      demand.requirement != CanonicalRequirement::Completion ||
      demand.causes.empty()) {
    return false;
  }
  const CanonicalStorageGeneration &generation =
      program.getStorageGeneration(channel.generation);
  const bool positive = hasDistance(demand, generation.loop,
                                    CanonicalIterationRelation::AnyPositive);
  const bool same = hasDistance(demand, generation.loop,
                                CanonicalIterationRelation::Same);
  return llvm::all_of(demand.causes, [&](const CanonicalDemandCause &cause) {
    const bool ready = demand.kind == CanonicalDemandKind::Raw &&
                       (same || positive) &&
                       matchesCause(cause, generation.producerAccess,
                                    generation.consumerAccess);
    const bool release = demand.kind == CanonicalDemandKind::War && positive &&
                         matchesCause(cause, generation.consumerAccess,
                                      generation.producerAccess);
    const bool overwrite = demand.kind == CanonicalDemandKind::Waw &&
                           positive &&
                           matchesCause(cause, generation.producerAccess,
                                        generation.producerAccess);
    return ready || release || overwrite;
  });
}
