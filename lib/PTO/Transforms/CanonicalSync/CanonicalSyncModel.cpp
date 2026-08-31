// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/CanonicalSyncModel.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <tuple>

using namespace mlir;
using namespace mlir::pto;

bool CanonicalPhysicalResource::operator<(
    const CanonicalPhysicalResource &other) const {
  return std::tie(core, pipe) < std::tie(other.core, other.pipe);
}

std::optional<std::uint64_t> CanonicalByteInterval::end() const {
  const std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  if (size > maximum - begin) {
    return std::nullopt;
  }
  return begin + size;
}

template <typename Record>
static std::uint32_t appendRecord(llvm::SmallVectorImpl<Record> &records,
                                  Record record, bool frozen) {
  const bool idExhausted = records.size() >= kInvalidCanonicalSyncId;
  if (frozen || idExhausted) {
    llvm_unreachable("cannot append to a frozen or exhausted sync model");
  }
  const auto id = static_cast<std::uint32_t>(records.size());
  record.id = id;
  records.push_back(std::move(record));
  return id;
}

CanonicalRegionId CanonicalSyncProgram::appendRegion(CanonicalRegion region) {
  return appendRecord(regions, std::move(region), graphFrozen);
}

CanonicalPhaseId CanonicalSyncProgram::appendPhase(CanonicalPhase phase) {
  return appendRecord(phases, std::move(phase), graphFrozen);
}

CanonicalAccessId CanonicalSyncProgram::appendAccess(CanonicalAccess access) {
  return appendRecord(accesses, std::move(access), graphFrozen);
}

CanonicalDemandId CanonicalSyncProgram::appendDemand(CanonicalDemand demand) {
  return appendRecord(demands, std::move(demand), graphFrozen);
}

void CanonicalSyncProgram::appendDemandCause(CanonicalDemandId demand,
                                             CanonicalDemandCause cause) {
  if (graphFrozen || demand >= demands.size()) {
    llvm_unreachable("cannot extend an invalid or frozen demand");
  }
  demands[demand].causes.push_back(std::move(cause));
}

CanonicalMechanismId
CanonicalSyncProgram::appendMechanism(CanonicalMechanism mechanism) {
  return appendRecord(mechanisms, std::move(mechanism), frozen);
}

void CanonicalSyncProgram::setMechanismEventId(CanonicalMechanismId mechanism,
                                               unsigned eventId) {
  if (frozen || mechanism >= mechanisms.size()) {
    llvm_unreachable("cannot allocate an invalid or frozen mechanism");
  }
  mechanisms[mechanism].eventId = eventId;
}

void CanonicalSyncProgram::setDirectMechanism(CanonicalDemandId demand,
                                              CanonicalMechanismId mechanism) {
  const bool invalidDemand = demand >= demands.size();
  const bool invalidMechanism = mechanism >= mechanisms.size();
  if (frozen || invalidDemand || invalidMechanism) {
    llvm_unreachable("cannot map an invalid or frozen direct mechanism");
  }
  if (directMechanisms.empty()) {
    directMechanisms.assign(demands.size(), kInvalidCanonicalSyncId);
  }
  directMechanisms[demand] = mechanism;
}

void CanonicalSyncProgram::appendCoverageWorld(CanonicalCoverageWorld world) {
  if (frozen) {
    llvm_unreachable("cannot append to a frozen sync model");
  }
  coverageWorlds.push_back(std::move(world));
}

static bool validControlPath(llvm::ArrayRef<CanonicalControlAtom> path,
                             size_t regionCount) {
  return llvm::all_of(path, [regionCount](const CanonicalControlAtom &atom) {
    return atom.choice < regionCount;
  });
}

LogicalResult CanonicalSyncProgram::freezeGraph() {
  if (graphFrozen) {
    return success();
  }
  const auto fail = [this](llvm::Twine message) {
    function.emitError("invalid canonical synchronization model: ") << message;
    return failure();
  };
  for (const CanonicalRegion &region : regions) {
    if (region.id != 0 && region.parent >= regions.size()) {
      return fail("region has an invalid parent");
    }
  }
  for (const CanonicalPhase &phase : phases) {
    const bool invalidRegion = phase.region >= regions.size();
    const bool invalidControl =
        !validControlPath(phase.controlPath, regions.size());
    if (!phase.operation || invalidRegion || invalidControl) {
      return fail("phase has an invalid operation, region, or control path");
    }
  }
  for (const CanonicalAccess &access : accesses) {
    if (access.phase >= phases.size()) {
      return fail("access has an invalid phase");
    }
    for (const CanonicalByteInterval &interval : access.intervals) {
      if (!interval.end()) {
        return fail("access byte interval overflows uint64_t");
      }
    }
  }
  for (const CanonicalDemand &demand : demands) {
    const bool validTarget = demand.kind == CanonicalDemandKind::ExitCompletion
                                 ? demand.target == kInvalidCanonicalSyncId
                                 : demand.target < phases.size();
    const bool validSource = demand.source < phases.size();
    const bool validOwner = demand.owner < regions.size();
    const bool validGuard = validControlPath(demand.guard, regions.size());
    if (!validSource || !validTarget || !validOwner || !validGuard) {
      return fail("demand has an invalid endpoint, owner, or guard");
    }
  }
  graphFrozen = true;
  return success();
}

LogicalResult CanonicalSyncProgram::freeze() {
  if (frozen) {
    return success();
  }
  if (failed(freezeGraph())) {
    return failure();
  }
  const auto fail = [this](llvm::Twine message) {
    function.emitError("invalid canonical synchronization plan: ") << message;
    return failure();
  };
  for (const CanonicalMechanism &mechanism : mechanisms) {
    const bool tail = mechanism.kind == CanonicalMechanismKind::TailBarrier;
    const bool validCuts =
        tail ? mechanism.sourceCut == kInvalidCanonicalSyncId &&
                   mechanism.targetCut == kInvalidCanonicalSyncId
             : mechanism.sourceCut < phases.size() &&
                   mechanism.targetCut < phases.size();
    if (!validCuts || mechanism.actionRegion >= regions.size()) {
      return fail("mechanism has an invalid cut or action region");
    }
  }
  for (const CanonicalCoverageWorld &world : coverageWorlds) {
    if (llvm::any_of(world.mechanisms,
                     [this](CanonicalMechanismId id) {
                       return id >= mechanisms.size();
                     }) ||
        llvm::any_of(world.covered, [this](CanonicalDemandId id) {
          return id >= demands.size();
        })) {
      return fail("coverage world references an invalid ID");
    }
  }
  const bool wrongDirectCount = directMechanisms.size() != demands.size();
  if (wrongDirectCount ||
      llvm::any_of(directMechanisms, [this](CanonicalMechanismId id) {
        return id >= mechanisms.size();
      })) {
    return fail("every demand must name a valid direct mechanism");
  }
  frozen = true;
  return success();
}

const CanonicalRegion &
CanonicalSyncProgram::getRegion(CanonicalRegionId id) const {
  return regions[id];
}

const CanonicalPhase &
CanonicalSyncProgram::getPhase(CanonicalPhaseId id) const {
  return phases[id];
}

const CanonicalAccess &
CanonicalSyncProgram::getAccess(CanonicalAccessId id) const {
  return accesses[id];
}

const CanonicalDemand &
CanonicalSyncProgram::getDemand(CanonicalDemandId id) const {
  return demands[id];
}

const CanonicalMechanism &
CanonicalSyncProgram::getMechanism(CanonicalMechanismId id) const {
  return mechanisms[id];
}

StringRef mlir::pto::stringifyCanonicalCore(CanonicalCore core) {
  return core == CanonicalCore::AIC ? "AIC" : "AIV";
}

StringRef mlir::pto::stringifyCanonicalRegionKind(CanonicalRegionKind kind) {
  switch (kind) {
  case CanonicalRegionKind::Function:
    return "function";
  case CanonicalRegionKind::Sequence:
    return "sequence";
  case CanonicalRegionKind::Choice:
    return "choice";
  case CanonicalRegionKind::Loop:
    return "loop";
  case CanonicalRegionKind::Transparent:
    return "transparent";
  }
  llvm_unreachable("unknown canonical region kind");
}

StringRef mlir::pto::stringifyCanonicalAccessMode(CanonicalAccessMode mode) {
  switch (mode) {
  case CanonicalAccessMode::Read:
    return "R";
  case CanonicalAccessMode::Write:
    return "W";
  case CanonicalAccessMode::ReadWrite:
    return "RW";
  }
  llvm_unreachable("unknown canonical access mode");
}

StringRef mlir::pto::stringifyCanonicalDemandKind(CanonicalDemandKind kind) {
  switch (kind) {
  case CanonicalDemandKind::Raw:
    return "RAW";
  case CanonicalDemandKind::War:
    return "WAR";
  case CanonicalDemandKind::Waw:
    return "WAW";
  case CanonicalDemandKind::HardwareAccReadConflict:
    return "ACC_RAR";
  case CanonicalDemandKind::SsaCompletion:
    return "SSA";
  case CanonicalDemandKind::ExitCompletion:
    return "EXIT";
  case CanonicalDemandKind::Visibility:
    return "VISIBILITY";
  }
  llvm_unreachable("unknown canonical demand kind");
}

StringRef
mlir::pto::stringifyCanonicalMechanismKind(CanonicalMechanismKind kind) {
  switch (kind) {
  case CanonicalMechanismKind::IntrinsicOrder:
    return "intrinsic";
  case CanonicalMechanismKind::PipeBarrier:
    return "barrier";
  case CanonicalMechanismKind::Event:
    return "event";
  case CanonicalMechanismKind::FixedFence:
    return "fixed-fence";
  case CanonicalMechanismKind::TailBarrier:
    return "tail";
  }
  llvm_unreachable("unknown canonical mechanism kind");
}
