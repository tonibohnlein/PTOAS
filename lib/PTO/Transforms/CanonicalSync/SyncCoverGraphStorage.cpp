// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverGraph.h"

#include "SyncCoverGraphInternal.h"

#include <algorithm>

using namespace mlir::pto;
using namespace mlir::pto::sync_cover_detail;

bool mlir::pto::syncCoverStorageModeReads(SyncCoverStorageAccessMode mode) {
  return mode == SyncCoverStorageAccessMode::Read ||
         mode == SyncCoverStorageAccessMode::ReadWrite;
}

bool mlir::pto::syncCoverStorageModeWrites(SyncCoverStorageAccessMode mode) {
  return mode == SyncCoverStorageAccessMode::Write ||
         mode == SyncCoverStorageAccessMode::ReadWrite;
}

SyncCoverGraphResult
SyncCoverGraph::addStorageDomain(SyncCoverStorageDomainRole role,
                                 std::uint32_t addressSpace) {
  if (!canMutateStructure()) {
    return {SyncCoverGraphError::StructureFrozen, storageDomains_.size()};
  }
  const SyncCoverStorageDomainId id = storageDomains_.size();
  storageDomains_.push_back({id, role, addressSpace});
  return {SyncCoverGraphError::None, id};
}

SyncCoverGraphResult SyncCoverGraph::addStorageAccess(
    SyncCoverNodeId node, SyncCoverStorageDomainId domain,
    SyncCoverStorageAccessFamilyId family, SyncCoverStorageInterval extent,
    SyncCoverStorageAccessMode mode, std::optional<unsigned> addressOrdinal,
    bool exactPhysical, SyncCoverStorageAccessPath path) {
  if (!canMutateStructure()) {
    return {SyncCoverGraphError::StructureFrozen, storageAccesses_.size()};
  }
  if (node >= nodes_.size()) {
    return {SyncCoverGraphError::InvalidNode, storageAccesses_.size()};
  }
  if (domain >= storageDomains_.size()) {
    return {SyncCoverGraphError::InvalidStorageDomain, storageAccesses_.size()};
  }
  const bool invalidAccess = extent.begin >= extent.end ||
                             !isValidAccessMode(mode) ||
                             !isValidAccessPath(path);
  if (invalidAccess) {
    return {SyncCoverGraphError::InvalidStorageAccess, storageAccesses_.size()};
  }
  const StorageAccessKey key{node,       domain,         family, extent.begin,
                             extent.end, addressOrdinal, path};
  auto existing = storageAccessIds_.find(key);
  if (existing != storageAccessIds_.end()) {
    SyncCoverStorageAccess &access = storageAccesses_[existing->second];
    const unsigned merged =
        static_cast<unsigned>(access.mode) | static_cast<unsigned>(mode);
    access.mode = static_cast<SyncCoverStorageAccessMode>(merged);
    access.exactPhysical &= exactPhysical;
    return {SyncCoverGraphError::None, access.id};
  }
  const SyncCoverStorageAccessId id = storageAccesses_.size();
  storageAccesses_.push_back({id, node, domain, family, extent, mode,
                              addressOrdinal, exactPhysical, path});
  storageAccessIds_.emplace(key, id);
  return {SyncCoverGraphError::None, id};
}

SyncCoverGraphResult
SyncCoverGraph::addStorageWitness(SyncCoverStorageAccessId sourceAccess,
                                  SyncCoverStorageAccessId targetAccess) {
  if (!canMutateStructure()) {
    return {SyncCoverGraphError::StructureFrozen, storageWitnesses_.size()};
  }
  const bool invalidAccess = sourceAccess >= storageAccesses_.size() ||
                             targetAccess >= storageAccesses_.size();
  if (invalidAccess) {
    return {SyncCoverGraphError::InvalidStorageAccess,
            storageWitnesses_.size()};
  }
  const SyncCoverStorageAccess &source = storageAccesses_[sourceAccess];
  const SyncCoverStorageAccess &target = storageAccesses_[targetAccess];
  if (source.domain != target.domain) {
    return {SyncCoverGraphError::InvalidStorageWitness,
            storageWitnesses_.size()};
  }
  const SyncCoverStorageInterval overlap{
      std::max(source.extent.begin, target.extent.begin),
      std::min(source.extent.end, target.extent.end)};
  if (overlap.begin >= overlap.end) {
    return {SyncCoverGraphError::InvalidStorageWitness,
            storageWitnesses_.size()};
  }
  const auto key = std::make_pair(sourceAccess, targetAccess);
  auto existing = storageWitnessIds_.find(key);
  if (existing != storageWitnessIds_.end()) {
    return {SyncCoverGraphError::None, existing->second};
  }
  const SyncCoverStorageWitnessId id = storageWitnesses_.size();
  storageWitnesses_.push_back({id, sourceAccess, targetAccess, overlap});
  storageWitnessIds_.emplace(key, id);
  return {SyncCoverGraphError::None, id};
}

SyncCoverGraphResult SyncCoverGraph::addTargetCompletionCertificate(
    SyncCoverTargetCompletionKind kind, SyncCoverNodeId completionNode,
    SyncCoverNodeId target, std::uint32_t sourceResource,
    std::uint32_t targetResource,
    std::vector<SyncCoverStorageDomainId> storageDomains,
    std::vector<SyncCoverDemandId> demands) {
  if (!canMutateStructure()) {
    return {SyncCoverGraphError::StructureFrozen,
            targetCompletionCertificates_.size()};
  }
  std::sort(storageDomains.begin(), storageDomains.end());
  storageDomains.erase(
      std::unique(storageDomains.begin(), storageDomains.end()),
      storageDomains.end());
  std::sort(demands.begin(), demands.end());
  demands.erase(std::unique(demands.begin(), demands.end()), demands.end());
  const SyncCoverTargetCompletionCertificateId id =
      targetCompletionCertificates_.size();
  targetCompletionCertificates_.push_back(
      {id, kind, completionNode, target, sourceResource, targetResource,
       std::move(storageDomains), std::move(demands)});
  const SyncCoverGraphResult validated = validateTargetCompletionCertificates();
  if (!validated) {
    targetCompletionCertificates_.pop_back();
    return {validated.error, id};
  }
  return {SyncCoverGraphError::None, id};
}

SyncCoverGraphResult SyncCoverGraph::addBasicOwnershipCertificate(
    SyncCoverBasicOwnershipCertificate certificate) {
  std::vector<SyncCoverBasicOwnershipCertificate> certificates;
  certificates.push_back(std::move(certificate));
  return addBasicOwnershipCertificates(std::move(certificates));
}

SyncCoverGraphResult SyncCoverGraph::addBasicOwnershipCertificates(
    std::vector<SyncCoverBasicOwnershipCertificate> certificates) {
  if (certificates.empty()) {
    return {SyncCoverGraphError::None, std::nullopt};
  }
  if (!canMutateStructure()) {
    return {SyncCoverGraphError::StructureFrozen,
            basicOwnershipCertificates_.size()};
  }
  const SyncCoverBasicOwnershipCertificateId firstId =
      basicOwnershipCertificates_.size();
  for (SyncCoverBasicOwnershipCertificate &certificate : certificates) {
    certificate.id = basicOwnershipCertificates_.size();
    for (SyncCoverBasicOwnershipLane &lane : certificate.lanes) {
      for (SyncCoverBasicOwnershipSlot &slot : lane.slots) {
        std::sort(slot.accesses.begin(), slot.accesses.end());
        slot.accesses.erase(
            std::unique(slot.accesses.begin(), slot.accesses.end()),
            slot.accesses.end());
      }
    }
    for (SyncCoverBasicOwnershipPath &path : certificate.paths) {
      for (SyncCoverBasicOwnershipUse &use : path.uses) {
        std::sort(use.producers.begin(), use.producers.end());
        use.producers.erase(
            std::unique(use.producers.begin(), use.producers.end()),
            use.producers.end());
        std::sort(use.consumers.begin(), use.consumers.end());
        use.consumers.erase(
            std::unique(use.consumers.begin(), use.consumers.end()),
            use.consumers.end());
      }
    }
    std::sort(certificate.initialProducers.begin(),
              certificate.initialProducers.end());
    certificate.initialProducers.erase(
        std::unique(certificate.initialProducers.begin(),
                    certificate.initialProducers.end()),
        certificate.initialProducers.end());
    std::sort(certificate.initiallyFreeLanes.begin(),
              certificate.initiallyFreeLanes.end());
    certificate.initiallyFreeLanes.erase(
        std::unique(certificate.initiallyFreeLanes.begin(),
                    certificate.initiallyFreeLanes.end()),
        certificate.initiallyFreeLanes.end());
    basicOwnershipCertificates_.push_back(std::move(certificate));
  }
  const SyncCoverGraphResult validated = validateBasicOwnershipCertificates();
  if (!validated) {
    basicOwnershipCertificates_.resize(firstId);
    return {validated.error, firstId};
  }
  return {SyncCoverGraphError::None, firstId};
}

bool SyncCoverGraph::hasTargetCompletionCertificate(
    SyncCoverTargetCompletionKind kind, SyncCoverNodeId completionNode,
    SyncCoverNodeId target, std::uint32_t sourceResource,
    std::uint32_t targetResource, SyncCoverDemandId demand) const {
  return std::any_of(
      targetCompletionCertificates_.begin(),
      targetCompletionCertificates_.end(),
      [&](const SyncCoverTargetCompletionCertificate &certificate) {
        return certificate.kind == kind &&
               certificate.completionNode == completionNode &&
               certificate.target == target &&
               certificate.sourceResource == sourceResource &&
               certificate.targetResource == targetResource &&
               std::binary_search(certificate.demands.begin(),
                                  certificate.demands.end(), demand);
      });
}
