// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/CanonicalSyncAnalysis.h"

#include "PTO/Transforms/InsertSync/MemoryDependentAnalyzer.h"
#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>
#include <map>
#include <set>
#include <tuple>

namespace mlir::pto {
namespace {

using HazardKey = std::tuple<SyncCoverNodeId, SyncCoverNodeId,
                             SyncCoverDemandKind, AddressSpace>;
using HazardKeySet = std::set<HazardKey>;

bool contains(const HazardKeySet &keys, const HazardKey &key) {
  return keys.find(key) != keys.end();
}

StringRef hazardName(SyncCoverDemandKind kind) {
  switch (kind) {
  case SyncCoverDemandKind::SSA:
    return "SSA";
  case SyncCoverDemandKind::MemoryRAW:
    return "RAW";
  case SyncCoverDemandKind::MemoryWAR:
    return "WAR";
  case SyncCoverDemandKind::MemoryWAW:
    return "WAW";
  case SyncCoverDemandKind::HardwareAccRAR:
    return "ACC_RAR";
  }
  return "unknown";
}

std::string keyText(const HazardKey &key) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "source=" << std::get<0>(key) << " target=" << std::get<1>(key)
         << " kind=" << hazardName(std::get<2>(key))
         << " space=" << static_cast<unsigned>(std::get<3>(key));
  return text;
}

std::string canonicalRangeText(const SyncCoverStorageAccess &access) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  if (access.extent.begin == 0 &&
      access.extent.end == std::numeric_limits<std::uint64_t>::max()) {
    stream << "unknown";
  } else {
    stream << '[' << access.extent.begin << ',' << access.extent.end << ')';
  }
  return text;
}

std::string insertSyncRangeText(const BaseMemInfo *access) {
  if (!access || access->aliasesUnknownRange || access->allocateSize == 0) {
    return "unknown";
  }
  std::string text;
  llvm::raw_string_ostream stream(text);
  const auto printRange = [&](std::uint64_t begin) {
    stream << '[' << begin << ',';
    if (begin >
        std::numeric_limits<std::uint64_t>::max() - access->allocateSize) {
      stream << "overflow";
    } else {
      stream << begin + access->allocateSize;
    }
    stream << ')';
  };
  if (access->rootRelativeOffset) {
    printRange(*access->rootRelativeOffset);
    return text;
  }
  stream << '{';
  llvm::interleaveComma(access->baseAddresses, stream,
                        [&](std::uint64_t base) { printRange(base); });
  stream << '}';
  return text;
}

bool modesMatch(SyncCoverDemandKind kind, const SyncCoverStorageAccess &source,
                const SyncCoverStorageAccess &target) {
  switch (kind) {
  case SyncCoverDemandKind::MemoryRAW:
    return syncCoverStorageModeWrites(source.mode) &&
           syncCoverStorageModeReads(target.mode);
  case SyncCoverDemandKind::MemoryWAR:
    return syncCoverStorageModeReads(source.mode) &&
           syncCoverStorageModeWrites(target.mode);
  case SyncCoverDemandKind::MemoryWAW:
    return syncCoverStorageModeWrites(source.mode) &&
           syncCoverStorageModeWrites(target.mode);
  case SyncCoverDemandKind::HardwareAccRAR:
    return syncCoverStorageModeReads(source.mode) &&
           syncCoverStorageModeReads(target.mode);
  case SyncCoverDemandKind::SSA:
    return false;
  }
  return false;
}

struct Ledger {
  HazardKeySet keys;
  std::map<HazardKey, std::set<std::string>> details;

  void add(const HazardKey &key, std::string detail) {
    keys.insert(key);
    details[key].insert(std::move(detail));
  }

  std::string dump() const {
    std::string text;
    llvm::raw_string_ostream stream(text);
    for (const HazardKey &key : keys) {
      stream << keyText(key);
      auto position = details.find(key);
      if (position != details.end()) {
        stream << " ranges=";
        llvm::interleaveComma(position->second, stream);
      }
      stream << '\n';
    }
    return text;
  }
};

std::optional<SyncCoverNodeId>
findNode(const CanonicalSyncProgram &program,
         const CompoundInstanceElement &compound) {
  for (auto [nodeId, binding] : llvm::enumerate(program.getNodeBindings())) {
    if (binding.operation == compound.elementOp &&
        binding.macroPhase == compound.macroOpInstanceId) {
      return nodeId;
    }
  }
  return std::nullopt;
}

void addInsertSyncHazards(
    Ledger &ledger, MemoryDependentAnalyzer &analyzer,
    const CompoundInstanceElement &source,
    const CompoundInstanceElement &target, SyncCoverNodeId sourceNode,
    SyncCoverNodeId targetNode, SyncCoverDemandKind kind,
    const SmallVector<const BaseMemInfo *> &sourceAccesses,
    const SmallVector<const BaseMemInfo *> &targetAccesses) {
  DepBaseMemInfoPairVec pairs;
  if (!analyzer.DepBetween(targetAccesses, sourceAccesses, pairs)) {
    return;
  }
  for (const auto &[targetAccess, sourceAccess] : pairs) {
    if (!sourceAccess || !targetAccess) {
      continue;
    }
    if (kind == SyncCoverDemandKind::HardwareAccRAR &&
        (source.kPipeValue == target.kPipeValue ||
         sourceAccess->scope != AddressSpace::ACC)) {
      continue;
    }
    const HazardKey key{sourceNode, targetNode, kind, sourceAccess->scope};
    ledger.add(key, "source=" + insertSyncRangeText(sourceAccess) +
                        " target=" + insertSyncRangeText(targetAccess));
  }
}

} // namespace

FailureOr<CanonicalSyncHazardParityReport>
compareCanonicalSyncRawHazardsWithInsertSync(
    const CanonicalSyncProgram &program, std::size_t maximumPairInspections) {
  CanonicalSyncHazardParityReport report;
  const SyncCoverGraph &graph = program.getGraph();

  Ledger canonical;
  for (const SyncCoverDemand &demand : graph.getDemands()) {
    if (demand.distance != 0) {
      continue;
    }
    for (SyncCoverDemandKind kind : demand.provenanceKinds) {
      if (kind == SyncCoverDemandKind::SSA) {
        continue;
      }
      for (SyncCoverStorageWitnessId witnessId : demand.storageWitnesses) {
        const SyncCoverStorageWitness &witness =
            graph.getStorageWitnesses()[witnessId];
        const SyncCoverStorageAccess &source =
            graph.getStorageAccesses()[witness.sourceAccess];
        const SyncCoverStorageAccess &target =
            graph.getStorageAccesses()[witness.targetAccess];
        if (!modesMatch(kind, source, target)) {
          continue;
        }
        const HazardKey key{demand.source, demand.target, kind,
                            program.getStorageSpaces()[source.domain]};
        canonical.add(key, "source=" + canonicalRangeText(source) +
                               " target=" + canonicalRangeText(target));
      }
    }
  }
  report.canonicalRawHazards = canonical.dump();

  const bool hasStructuredControl =
      graph.getScopes().size() != 1 || !graph.getControls().empty();
  if (hasStructuredControl) {
    report.incompleteReason =
        "structured control requires the explicit-unroll parity oracle";
    return report;
  }

  SyncIRs syncIR;
  Buffer2MemInfoMap bufferMap;
  OperationMemInfoStorage operationMemInfos;
  PTOIRTranslatorOptions translatorOptions;
  translatorOptions.preciseGmRanges = true;
  translatorOptions.includeExtendedEffects = true;
  translatorOptions.failOnUnmodeledEffects = true;
  PTOIRTranslator translator(syncIR, bufferMap, operationMemInfos,
                             program.getFunction(), translatorOptions);
  if (failed(translator.Build())) {
    return failure();
  }

  std::vector<const CompoundInstanceElement *> compounds;
  for (const std::unique_ptr<InstanceElement> &element : syncIR) {
    if (const auto *compound =
            dyn_cast<CompoundInstanceElement>(element.get())) {
      compounds.push_back(compound);
    }
  }

  Ledger insertSync;
  MemoryDependentAnalyzer analyzer;
  for (std::size_t targetIndex = 0; targetIndex < compounds.size();
       ++targetIndex) {
    const CompoundInstanceElement &target = *compounds[targetIndex];
    const std::optional<SyncCoverNodeId> targetNode = findNode(program, target);
    if (!targetNode) {
      return failure();
    }
    for (std::size_t sourceIndex = 0; sourceIndex < targetIndex;
         ++sourceIndex) {
      if (report.pairInspections == maximumPairInspections) {
        report.incompleteReason = "InsertSync parity pair budget exhausted";
        report.insertSyncRawHazards = insertSync.dump();
        return report;
      }
      ++report.pairInspections;
      const CompoundInstanceElement &source = *compounds[sourceIndex];
      const std::optional<SyncCoverNodeId> sourceNode =
          findNode(program, source);
      if (!sourceNode) {
        return failure();
      }
      addInsertSyncHazards(insertSync, analyzer, source, target, *sourceNode,
                           *targetNode, SyncCoverDemandKind::MemoryRAW,
                           source.defVec, target.useVec);
      addInsertSyncHazards(insertSync, analyzer, source, target, *sourceNode,
                           *targetNode, SyncCoverDemandKind::MemoryWAR,
                           source.useVec, target.defVec);
      addInsertSyncHazards(insertSync, analyzer, source, target, *sourceNode,
                           *targetNode, SyncCoverDemandKind::MemoryWAW,
                           source.defVec, target.defVec);
      addInsertSyncHazards(insertSync, analyzer, source, target, *sourceNode,
                           *targetNode, SyncCoverDemandKind::HardwareAccRAR,
                           source.useVec, target.useVec);
    }
  }
  report.insertSyncRawHazards = insertSync.dump();

  for (const HazardKey &key : canonical.keys) {
    if (!contains(insertSync.keys, key)) {
      report.canonicalOnly.push_back(keyText(key));
    }
  }
  for (const HazardKey &key : insertSync.keys) {
    if (!contains(canonical.keys, key)) {
      report.insertSyncOnly.push_back(keyText(key));
    }
  }
  report.complete = true;
  return report;
}

} // namespace mlir::pto
