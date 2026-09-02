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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::canonical_sync_detail;

namespace {

struct OwnershipKey {
  Value storage;
  AddressSpace space = AddressSpace::Zero;
  CanonicalRegionId loop = kInvalidCanonicalSyncId;
  CanonicalPhysicalResource producer;
  CanonicalPhysicalResource consumer;
  SmallVector<CanonicalControlAtom, 2> guard;
};

bool sameKey(const OwnershipKey &left, const OwnershipKey &right) {
  return left.storage == right.storage && left.space == right.space &&
         left.loop == right.loop && left.producer == right.producer &&
         left.consumer == right.consumer && left.guard == right.guard;
}

std::uint64_t hashKey(const OwnershipKey &key) {
  llvm::hash_code hash = llvm::hash_combine(
      key.storage, static_cast<unsigned>(key.space), key.loop,
      static_cast<unsigned>(key.producer.core),
      static_cast<unsigned>(key.producer.pipe),
      static_cast<unsigned>(key.consumer.core),
      static_cast<unsigned>(key.consumer.pipe));
  for (const CanonicalControlAtom &atom : key.guard) {
    hash = llvm::hash_combine(hash, atom.choice, atom.arm);
  }
  return static_cast<std::uint64_t>(static_cast<std::size_t>(hash));
}

bool isLocalStorage(const CanonicalAccess &access) {
  return !access.unknownSpace && access.space != AddressSpace::GM &&
         access.space != AddressSpace::Zero && access.aliasRoot;
}

std::optional<CanonicalRegionId>
getCarryingLoop(const CanonicalDemand &demand) {
  std::optional<CanonicalRegionId> carryingLoop;
  for (const CanonicalLoopDistance &distance : demand.iterationDistance) {
    if (distance.relation != CanonicalIterationRelation::AnyPositive) {
      continue;
    }
    if (carryingLoop) {
      return std::nullopt;
    }
    carryingLoop = distance.loop;
  }
  return carryingLoop;
}

bool isSameIterationAt(const CanonicalDemand &demand,
                       CanonicalRegionId loop) {
  bool containsLoop = false;
  for (const CanonicalLoopDistance &distance : demand.iterationDistance) {
    if (distance.relation != CanonicalIterationRelation::Same) {
      return false;
    }
    containsLoop |= distance.loop == loop;
  }
  return containsLoop;
}

std::uint32_t getStaticDepth(Value storage) {
  if (auto multi = dyn_cast<MultiTileBufType>(storage.getType())) {
    return multi.getCount();
  }
  return 1;
}

bool containsEdge(ArrayRef<CanonicalOwnershipEdge> edges,
                  const CanonicalOwnershipEdge &candidate) {
  return llvm::any_of(edges, [&](const CanonicalOwnershipEdge &edge) {
    return edge.demand == candidate.demand &&
           edge.sourceAccess == candidate.sourceAccess &&
           edge.targetAccess == candidate.targetAccess;
  });
}

class OwnershipChannelIndex {
public:
  CanonicalOwnershipChannel &findOrCreate(const OwnershipKey &key) {
    const std::uint64_t hash = hashKey(key);
    for (unsigned index : buckets[hash]) {
      if (sameKey(keys[index], key)) {
        return channels[index];
      }
    }
    const unsigned index = channels.size();
    buckets[hash].push_back(index);
    keys.push_back(key);
    CanonicalOwnershipChannel channel;
    channel.storage = key.storage;
    channel.space = key.space;
    channel.loop = key.loop;
    channel.producer = key.producer;
    channel.consumer = key.consumer;
    channel.guard = key.guard;
    channel.staticDepth = getStaticDepth(key.storage);
    channel.slotTracked = true;
    channels.push_back(std::move(channel));
    return channels.back();
  }

  CanonicalOwnershipChannel *find(const OwnershipKey &key) {
    auto found = buckets.find(hashKey(key));
    if (found == buckets.end()) {
      return nullptr;
    }
    for (unsigned index : found->second) {
      if (sameKey(keys[index], key)) {
        return &channels[index];
      }
    }
    return nullptr;
  }

  SmallVector<CanonicalOwnershipChannel, 0> takeChannels() {
    return std::move(channels);
  }

private:
  llvm::DenseMap<std::uint64_t, SmallVector<unsigned, 1>> buckets;
  SmallVector<OwnershipKey, 0> keys;
  SmallVector<CanonicalOwnershipChannel, 0> channels;
};

std::optional<OwnershipKey>
buildReleaseKey(const CanonicalSyncProgram &program,
                const CanonicalDemand &demand,
                const CanonicalDemandCause &cause) {
  if (demand.kind != CanonicalDemandKind::War ||
      demand.requirement != CanonicalRequirement::Completion ||
      cause.sourceAccess == kInvalidCanonicalSyncId ||
      cause.targetAccess == kInvalidCanonicalSyncId ||
      demand.sourceGuard != demand.targetGuard) {
    return std::nullopt;
  }
  std::optional<CanonicalRegionId> carryingLoop = getCarryingLoop(demand);
  if (!carryingLoop) {
    return std::nullopt;
  }
  const CanonicalAccess &source = program.getAccess(cause.sourceAccess);
  const CanonicalAccess &target = program.getAccess(cause.targetAccess);
  const bool matchingLocalStorage =
      isLocalStorage(source) && isLocalStorage(target) &&
      source.space == target.space && source.aliasRoot == target.aliasRoot;
  if (!matchingLocalStorage) {
    return std::nullopt;
  }
  const CanonicalPhysicalResource consumer =
      program.getPhase(source.phase).resource;
  const CanonicalPhysicalResource producer =
      program.getPhase(target.phase).resource;
  if (producer.core != consumer.core || producer == consumer) {
    return std::nullopt;
  }
  return OwnershipKey{target.aliasRoot, target.space, *carryingLoop, producer,
                      consumer, demand.sourceGuard};
}

void collectReleaseEdges(const CanonicalSyncProgram &program,
                         OwnershipChannelIndex &index) {
  for (const CanonicalDemand &demand : program.getDemands()) {
    for (const CanonicalDemandCause &cause : demand.causes) {
      std::optional<OwnershipKey> key =
          buildReleaseKey(program, demand, cause);
      if (!key) {
        continue;
      }
      CanonicalOwnershipChannel &channel = index.findOrCreate(*key);
      CanonicalOwnershipEdge edge{demand.id, cause.sourceAccess,
                                  cause.targetAccess};
      if (!containsEdge(channel.releaseEdges, edge)) {
        channel.releaseEdges.push_back(edge);
      }
      const CanonicalAccess &source = program.getAccess(cause.sourceAccess);
      const CanonicalAccess &target = program.getAccess(cause.targetAccess);
      channel.slotTracked &= source.slotExpression && target.slotExpression;
    }
  }
}

void collectReadyEdges(const CanonicalSyncProgram &program,
                       OwnershipChannelIndex &index) {
  for (const CanonicalDemand &demand : program.getDemands()) {
    if (demand.kind != CanonicalDemandKind::Raw ||
        demand.requirement != CanonicalRequirement::Completion ||
        demand.sourceGuard != demand.targetGuard) {
      continue;
    }
    for (const CanonicalDemandCause &cause : demand.causes) {
      if (cause.sourceAccess == kInvalidCanonicalSyncId ||
          cause.targetAccess == kInvalidCanonicalSyncId) {
        continue;
      }
      const CanonicalAccess &source = program.getAccess(cause.sourceAccess);
      const CanonicalAccess &target = program.getAccess(cause.targetAccess);
      const bool matchingLocalStorage =
          isLocalStorage(source) && isLocalStorage(target) &&
          source.space == target.space && source.aliasRoot == target.aliasRoot;
      if (!matchingLocalStorage) {
        continue;
      }
      const CanonicalPhysicalResource producer =
          program.getPhase(source.phase).resource;
      const CanonicalPhysicalResource consumer =
          program.getPhase(target.phase).resource;
      if (producer.core != consumer.core || producer == consumer) {
        continue;
      }
      for (const CanonicalLoopDistance &distance :
           demand.iterationDistance) {
        if (!isSameIterationAt(demand, distance.loop)) {
          continue;
        }
        OwnershipKey key{source.aliasRoot, source.space, distance.loop,
                         producer, consumer, demand.sourceGuard};
        CanonicalOwnershipChannel *channel = index.find(key);
        if (!channel) {
          continue;
        }
        CanonicalOwnershipEdge edge{demand.id, cause.sourceAccess,
                                    cause.targetAccess};
        if (!containsEdge(channel->readyEdges, edge)) {
          channel->readyEdges.push_back(edge);
        }
        channel->slotTracked &=
            source.slotExpression && target.slotExpression;
      }
    }
  }
}

} // namespace

void mlir::pto::canonical_sync_detail::deriveCanonicalOwnershipChannels(
    CanonicalSyncProgram &program) {
  OwnershipChannelIndex index;
  collectReleaseEdges(program, index);
  collectReadyEdges(program, index);

  CanonicalSyncStatistics *statistics = program.getStatistics();
  for (CanonicalOwnershipChannel &channel : index.takeChannels()) {
    const bool completeChannel =
        !channel.readyEdges.empty() && !channel.releaseEdges.empty();
    if (!completeChannel) {
      continue;
    }
    if (statistics) {
      ++statistics->ownershipChannels;
      statistics->ownershipReadyEdges += channel.readyEdges.size();
      statistics->ownershipReleaseEdges += channel.releaseEdges.size();
      statistics->depthTwoOwnershipChannels += channel.staticDepth == 2;
      statistics->slotTrackedOwnershipChannels += channel.slotTracked;
    }
    program.appendOwnershipChannel(std::move(channel));
  }
}
