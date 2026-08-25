// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/CanonicalSync.h"

#include <iostream>
#include <string_view>
#include <vector>

using namespace mlir::pto;

namespace {

bool check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "CanonicalSyncCoveringAllocationTest failure: " << message
              << '\n';
  }
  return condition;
}

CanonicalSyncCoveringShadowSnapshot makeValidSnapshot() {
  CanonicalSyncCoveringShadowSnapshot snapshot;
  snapshot.selectionAttempted = true;
  snapshot.selectionError = SyncCoverSelectionError::None;
  snapshot.selectedMechanisms = 2;
  snapshot.resourceDomainCount = 1;
  snapshot.selectedProviders = {
      {0, {CanonicalSelectionMechanismKind::EventBundle, 0}},
      {2, {CanonicalSelectionMechanismKind::EventBundle, 1}}};
  snapshot.resourceDomainDetails = {
      {0, SyncCoverResourceKind::EventId, 1, 2, 0, 2, {1}}};
  snapshot.selectedResourceUses = {
      {0,
       {CanonicalSelectionMechanismKind::EventBundle, 0},
       0,
       0,
       SyncCoverResourceKind::EventId,
       1,
       2,
       0,
       0,
       0,
       1,
       {0, 1},
       0},
      {2,
       {CanonicalSelectionMechanismKind::EventBundle, 1},
       0,
       0,
       SyncCoverResourceKind::EventId,
       1,
       2,
       0,
       0,
       0,
       1,
       {2, 3},
       0}};
  snapshot.selectedAllocations = {
      {0,
       {CanonicalSelectionMechanismKind::EventBundle, 0},
       0,
       0,
       SyncCoverResourceKind::EventId,
       1,
       2,
       {0}},
      {2,
       {CanonicalSelectionMechanismKind::EventBundle, 1},
       0,
       0,
       SyncCoverResourceKind::EventId,
       1,
       2,
       {0}}};
  snapshot.selectedResources.resourceFeasible = true;
  SyncCoverDomainFeasibility feasibility;
  feasibility.domain = 0;
  feasibility.required = 1;
  feasibility.available = 1;
  feasibility.allocations = {{{0, 0, 1}, {0}}, {{2, 0, 1}, {0}}};
  snapshot.selectedResources.domains.push_back(std::move(feasibility));
  return snapshot;
}

bool hasError(const CanonicalSyncCoveringShadowSnapshot &snapshot,
              CanonicalSyncCoveringAllocationError expected) {
  return validateCanonicalSyncCoveringAllocation(snapshot).error == expected;
}

bool testValidAndExactAllocation() {
  bool passed = check(static_cast<bool>(
                          validateCanonicalSyncCoveringAllocation(
                              makeValidSnapshot())),
                      "non-overlapping uses may reuse one physical ID");

  CanonicalSyncCoveringShadowSnapshot mismatch = makeValidSnapshot();
  mismatch.selectedAllocations[1].ids = {1};
  passed &= check(hasError(mismatch,
                           CanonicalSyncCoveringAllocationError::InvalidAllocation),
                  "translated IDs must equal the final solver allocation");
  return passed;
}

bool testOwnerSetValidation() {
  CanonicalSyncCoveringShadowSnapshot missing = makeValidSnapshot();
  missing.selectedAllocations.pop_back();
  bool passed = check(
      hasError(missing,
               CanonicalSyncCoveringAllocationError::InvalidAllocation),
      "every selected resource use requires an allocation");

  CanonicalSyncCoveringShadowSnapshot duplicate = makeValidSnapshot();
  duplicate.selectedAllocations.push_back(duplicate.selectedAllocations.back());
  passed &= check(
      hasError(duplicate,
               CanonicalSyncCoveringAllocationError::InvalidAllocation),
      "duplicate allocation owners fail closed");

  CanonicalSyncCoveringShadowSnapshot stale = makeValidSnapshot();
  stale.selectedAllocations[1].mechanism = 7;
  passed &= check(
      hasError(stale,
               CanonicalSyncCoveringAllocationError::InvalidAllocation),
      "allocations for unselected mechanisms fail closed");
  return passed;
}

bool testPhysicalIdValidation() {
  CanonicalSyncCoveringShadowSnapshot reserved = makeValidSnapshot();
  reserved.resourceDomainDetails[0].reservedIds = {0};
  bool passed = check(
      hasError(reserved,
               CanonicalSyncCoveringAllocationError::InvalidAllocation),
      "reserved physical IDs cannot be assigned");

  CanonicalSyncCoveringShadowSnapshot conflict = makeValidSnapshot();
  conflict.resourceDomainDetails[0].reservedIds.clear();
  conflict.selectedResources.domains[0].available = 2;
  conflict.selectedResources.domains[0].required = 2;
  conflict.selectedResourceUses[1].lifetime = {1, 3};
  passed &= check(
      hasError(conflict,
               CanonicalSyncCoveringAllocationError::ConflictingAssignment),
      "overlapping closed lifetimes cannot share one physical ID");

  CanonicalSyncCoveringShadowSnapshot outOfRange = makeValidSnapshot();
  outOfRange.resourceDomainDetails[0].reservedIds.clear();
  outOfRange.selectedAllocations[0].ids = {2};
  outOfRange.selectedResources.domains[0].allocations[0].ids = {2};
  passed &= check(
      hasError(outOfRange,
               CanonicalSyncCoveringAllocationError::InvalidAllocation),
      "physical IDs outside the domain budget fail closed");

  CanonicalSyncCoveringShadowSnapshot duplicateLane = makeValidSnapshot();
  duplicateLane.resourceDomainDetails[0].reservedIds.clear();
  duplicateLane.selectedResourceUses[0].width = 2;
  duplicateLane.selectedAllocations[0].ids = {0, 0};
  duplicateLane.selectedResources.domains[0].allocations[0].owner.width = 2;
  duplicateLane.selectedResources.domains[0].allocations[0].ids = {0, 0};
  passed &= check(
      hasError(duplicateLane,
               CanonicalSyncCoveringAllocationError::InvalidAllocation),
      "one width-N use cannot repeat a physical ID");
  return passed;
}

bool testDomainAndKindValidation() {
  CanonicalSyncCoveringShadowSnapshot pressure = makeValidSnapshot();
  pressure.selectedResources.domains[0].required = 2;
  bool passed = check(
      hasError(pressure,
               CanonicalSyncCoveringAllocationError::InvalidPressure),
      "stored feasibility pressure must match selected lifetimes");

  CanonicalSyncCoveringShadowSnapshot token = makeValidSnapshot();
  token.resourceDomainDetails[0].kind = SyncCoverResourceKind::BufferToken;
  token.resourceDomainDetails[0].poolIdentity = 9;
  token.selectedResourceUses[0].kind = SyncCoverResourceKind::BufferToken;
  token.selectedResourceUses[0].poolIdentity = 9;
  token.selectedResourceUses[0].eventIndex.reset();
  token.selectedAllocations[0].kind = SyncCoverResourceKind::BufferToken;
  passed &= check(
      hasError(token,
               CanonicalSyncCoveringAllocationError::UnsupportedResourceKind),
      "BufferToken emission is rejected in version one");
  return passed;
}

bool testLiveLifetimeAuthentication() {
  const CanonicalSyncCoveringSelectedResourceUse selected =
      makeValidSnapshot().selectedResourceUses.front();
  SyncCoverResourceUse live;
  live.domain = selected.domain;
  live.scope = selected.scope;
  live.distance = selected.distance;
  live.width = selected.width;
  bool passed = check(
      canonicalSyncCoveringResourceUseMatches(selected, live, {0, 1}),
      "stored use agrees with its live mechanism lifetime");
  passed &= check(
      !canonicalSyncCoveringResourceUseMatches(selected, live, {0, 0}),
      "shortened stored lifetimes cannot authenticate");
  return passed;
}

} // namespace

int main() {
  const bool passed = testValidAndExactAllocation() &&
                      testOwnerSetValidation() && testPhysicalIdValidation() &&
                      testDomainAndKindValidation() &&
                      testLiveLifetimeAuthentication();
  return passed ? 0 : 1;
}
