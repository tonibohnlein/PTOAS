// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- InsertSyncDebug.cpp - Debug printing for PTO InsertSync ------------===//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/InsertSync/InsertSyncDebug.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Matchers.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"

#include <algorithm>
#include <map>
#include <mutex>
#include <set>
#include <string>

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr unsigned kDebugDumpIndentSpaces = 2;

llvm::cl::opt<unsigned> insertSyncDebugLevelOpt(
    "pto-insert-sync-debug",
    llvm::cl::desc("Debug verbosity for PTOInsertSync: "
                   "0=off, 1=phase, 2=syncir, 3=trace"),
    llvm::cl::init(0));

llvm::cl::opt<std::string> insertSyncSummaryPathOpt(
    "pto-insert-sync-summary",
    llvm::cl::desc("Write final PTOInsertSync statistics as JSON Lines"),
    llvm::cl::value_desc("path"), llvm::cl::init(""));

llvm::cl::opt<std::string> insertSyncScheduleGraphPathOpt(
    "pto-insert-sync-schedule-graph",
    llvm::cl::desc("Write the analyzed PTOInsertSync schedule graph as JSON "
                   "Lines"),
    llvm::cl::value_desc("path"), llvm::cl::init(""));

} // namespace

unsigned mlir::pto::getInsertSyncDebugLevel() { return insertSyncDebugLevelOpt; }

bool mlir::pto::isInsertSyncSummaryEnabled() {
  return !insertSyncSummaryPathOpt.empty();
}

bool mlir::pto::isInsertSyncScheduleGraphEnabled() {
  return !insertSyncScheduleGraphPathOpt.empty();
}

bool mlir::pto::isInsertSyncDebugEnabled(InsertSyncDebugLevel minLevel) {
  return getInsertSyncDebugLevel() >= static_cast<unsigned>(minLevel);
}

static llvm::StringRef getPipelineName(PipelineType pipe) {
  switch (pipe) {
  case PipelineType::PIPE_S:
    return "PIPE_S";
  case PipelineType::PIPE_V:
    return "PIPE_V";
  case PipelineType::PIPE_M:
    return "PIPE_M";
  case PipelineType::PIPE_MTE1:
    return "PIPE_MTE1";
  case PipelineType::PIPE_MTE2:
    return "PIPE_MTE2";
  case PipelineType::PIPE_MTE3:
    return "PIPE_MTE3";
  case PipelineType::PIPE_ALL:
    return "PIPE_ALL";
  case PipelineType::PIPE_MTE4:
    return "PIPE_MTE4";
  case PipelineType::PIPE_MTE5:
    return "PIPE_MTE5";
  case PipelineType::PIPE_V2:
    return "PIPE_V2";
  case PipelineType::PIPE_FIX:
    return "PIPE_FIX";
  case PipelineType::VIRTUAL_PIPE_MTE2_L1A:
    return "VIRTUAL_PIPE_MTE2_L1A";
  case PipelineType::VIRTUAL_PIPE_MTE2_L1B:
    return "VIRTUAL_PIPE_MTE2_L1B";
  case PipelineType::PIPE_NUM:
    return "PIPE_NUM";
  case PipelineType::PIPE_UNASSIGNED:
    return "PIPE_UNASSIGNED";
  }
  return "PIPE_UNKNOWN";
}

static llvm::StringRef getBranchKindName(KindOfBranch kind) {
  switch (kind) {
  case KindOfBranch::IF_BEGIN:
    return "IF_BEGIN";
  case KindOfBranch::ELSE_BEGIN:
    return "ELSE_BEGIN";
  case KindOfBranch::IF_END:
    return "IF_END";
  }
  return "BRANCH_UNKNOWN";
}

static llvm::StringRef getLoopKindName(KindOfLoop kind) {
  switch (kind) {
  case KindOfLoop::LOOP_BEGIN:
    return "LOOP_BEGIN";
  case KindOfLoop::LOOP_END:
    return "LOOP_END";
  }
  return "LOOP_UNKNOWN";
}

static llvm::StringRef getMemScopeName(pto::AddressSpace scope) {
  switch (scope) {
  case pto::AddressSpace::Zero:
    return "Zero";
  case pto::AddressSpace::GM:
    return "GM";
  case pto::AddressSpace::VEC:
    return "VEC";
  case pto::AddressSpace::MAT:
    return "MAT";
  case pto::AddressSpace::ACC:
    return "ACC";
  case pto::AddressSpace::LEFT:
    return "LEFT";
  case pto::AddressSpace::RIGHT:
    return "RIGHT";
  case pto::AddressSpace::BIAS:
    return "BIAS";
  case pto::AddressSpace::SCALING:
    return "SCALING";
  }
  return "SCOPE_UNKNOWN";
}

static std::string printValue(Value value, AsmState *state) {
  if (!value)
    return "";
  std::string storage;
  llvm::raw_string_ostream os(storage);
  if (state)
    value.printAsOperand(os, *state);
  else
    os << "<value>";
  return os.str();
}

template <typename Printable>
static std::string printObject(const Printable &object) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  object.print(os);
  return os.str();
}

static llvm::json::Object encodeMemInfo(const BaseMemInfo *info,
                                        AsmState *state) {
  llvm::json::Object object;
  if (!info) {
    object["null"] = true;
    return object;
  }

  object["base"] = printValue(info->baseBuffer, state);
  object["root"] = printValue(info->rootBuffer, state);
  object["scope"] = getMemScopeName(info->scope).str();
  object["allocate_size_bytes"] = static_cast<int64_t>(info->allocateSize);
  object["known_physical_addresses"] = info->hasKnownPhysicalAddresses;
  object["aliases_unknown_range"] = info->aliasesUnknownRange;
  llvm::json::Array addresses;
  for (uint64_t address : info->baseAddresses)
    addresses.push_back(static_cast<int64_t>(address));
  object["base_addresses"] = std::move(addresses);
  return object;
}

static llvm::json::Array encodeMemInfoList(
    const SmallVector<const BaseMemInfo *> &infos, AsmState *state) {
  llvm::json::Array array;
  for (const BaseMemInfo *info : infos)
    array.push_back(encodeMemInfo(info, state));
  return array;
}

static std::optional<int64_t> getStaticTripCount(Operation *op) {
  auto forOp = dyn_cast_or_null<scf::ForOp>(op);
  if (!forOp)
    return std::nullopt;
  std::optional<int64_t> lower = getConstantIntValue(forOp.getLowerBound());
  std::optional<int64_t> upper = getConstantIntValue(forOp.getUpperBound());
  std::optional<int64_t> step = getConstantIntValue(forOp.getStep());
  if (!lower || !upper || !step || *step <= 0)
    return std::nullopt;
  if (*upper <= *lower)
    return 0;
  return (*upper - *lower + *step - 1) / *step;
}

static llvm::json::Object encodeOperation(Operation *op) {
  llvm::json::Object object;
  if (!op)
    return object;

  object["name"] = op->getName().getStringRef().str();
  const std::string location = printObject(op->getLoc());
  object["location"] = location;
  // PyPTO can wrap source locations in a diagnostic-only NameLoc of the form
  //   loc("pypto.access.N"("file.py":line:column))
  // where N is the same preorder coordinate exported in raw DSA candidate
  // records. Keep the parsed integer beside the lossless location string so
  // downstream schedule analysis does not depend on MLIR's pretty-printer.
  constexpr llvm::StringLiteral accessPrefix = "pypto.access.";
  llvm::StringRef locationRef(location);
  const size_t prefixPosition = locationRef.find(accessPrefix);
  if (prefixPosition != llvm::StringRef::npos) {
    llvm::StringRef suffix = locationRef.drop_front(prefixPosition + accessPrefix.size());
    const size_t end = suffix.find_first_not_of("0123456789");
    llvm::StringRef digits = suffix.take_front(end);
    uint64_t accessOrder = 0;
    if (!digits.empty() && !digits.getAsInteger(10, accessOrder))
      object["pypto_access_order"] = static_cast<int64_t>(accessOrder);
  }

  llvm::json::Array operands;
  for (Type type : op->getOperandTypes())
    operands.push_back(printObject(type));
  object["operand_types"] = std::move(operands);

  llvm::json::Array results;
  for (Type type : op->getResultTypes())
    results.push_back(printObject(type));
  object["result_types"] = std::move(results);

  llvm::json::Object attributes;
  for (NamedAttribute attribute : op->getAttrs())
    attributes[attribute.getName().strref().str()] =
        printObject(attribute.getValue());
  object["attributes"] = std::move(attributes);
  return object;
}

static llvm::json::Array encodeUnsignedStack(ArrayRef<unsigned> values) {
  llvm::json::Array array;
  for (unsigned value : values)
    array.push_back(static_cast<int64_t>(value));
  return array;
}

static llvm::json::Array encodeValues(ArrayRef<Value> values, AsmState *state) {
  llvm::json::Array array;
  for (Value value : values)
    array.push_back(printValue(value, state));
  return array;
}

static bool appendJsonLine(llvm::StringRef path, llvm::json::Object record,
                           std::mutex &outputMutex,
                           std::string &initializedPath,
                           llvm::StringRef description) {
  std::lock_guard<std::mutex> lock(outputMutex);
  const bool append = initializedPath == path;
  std::error_code error;
  llvm::sys::fs::OpenFlags flags = llvm::sys::fs::OF_Text;
  if (append)
    flags |= llvm::sys::fs::OF_Append;
  llvm::raw_fd_ostream output(path, error, flags);
  if (error) {
    llvm::errs() << "error: cannot write " << description << " '" << path
                 << "': " << error.message() << "\n";
    return false;
  }
  output << llvm::json::Value(std::move(record)) << "\n";
  initializedPath = path.str();
  return true;
}

static void dumpEventIds(llvm::raw_ostream &os,
                         const SmallVector<int> &eventIds) {
  os << "[";
  for (size_t i = 0; i < eventIds.size(); ++i) {
    os << eventIds[i];
    if (i + 1 != eventIds.size())
      os << ",";
  }
  os << "]";
}

static void dumpSyncOp(llvm::raw_ostream &os, const SyncOperation *op,
                       bool showUselessSync) {
  if (!op)
    return;
  if (op->uselessSync && !showUselessSync)
    return;

  os << SyncOperation::TypeName(op->GetType());
  os << " <" << getPipelineName(op->GetSrcPipe()) << " -> "
     << getPipelineName(op->GetDstPipe()) << ">";
  os << " idx=" << op->GetSyncIndex();

  if (op->GetForEndIndex().has_value())
    os << " forEnd=" << op->GetForEndIndex().value();

  if (op->eventIdNum != 1)
    os << " eventIdNum=" << op->eventIdNum;

  if (op->isCompensation)
    os << " compensation";
  if (op->uselessSync)
    os << " useless";

  if (!op->eventIds.empty()) {
    os << " eventIds=";
    dumpEventIds(os, op->eventIds);
  }
}

static void dumpMemInfoList(llvm::raw_ostream &os, llvm::StringRef tag,
                            const SmallVector<const BaseMemInfo *> &list,
                            mlir::AsmState *state) {
  os << tag << "=[";
  for (size_t i = 0; i < list.size(); ++i) {
    const BaseMemInfo *info = list[i];
    if (!info) {
      os << "<null>";
    } else if (info->rootBuffer) {
      if (state) {
        info->rootBuffer.printAsOperand(os, *state);
      } else {
        os << "<value>";
      }
      os << "(" << getMemScopeName(info->scope) << ")";
    } else {
      os << "<null-root>";
    }
    if (i + 1 != list.size())
      os << ", ";
  }
  os << "]";
}

static void dumpSyncIR(llvm::raw_ostream &os, const SyncIRs &syncIR,
                       Operation *opForPrinting, InsertSyncDumpOptions options,
                       bool showMemInfo) {
  std::optional<mlir::AsmState> state;
  if (showMemInfo && opForPrinting)
    state.emplace(opForPrinting);

  int indent = 0;
  auto indentBy = [&](int extra = 0) {
    return static_cast<unsigned>(std::max(0, indent) * 2 + extra);
  };

  for (const auto &e : syncIR) {
    if (!e)
      continue;

    if (auto *loop = dyn_cast<LoopInstanceElement>(e.get())) {
      if (loop->getLoopKind() == KindOfLoop::LOOP_END)
        indent = std::max(0, indent - 1);
    }
    if (auto *branch = dyn_cast<BranchInstanceElement>(e.get())) {
      if (branch->getBranchKind() == KindOfBranch::IF_END ||
          branch->getBranchKind() == KindOfBranch::ELSE_BEGIN)
        indent = std::max(0, indent - 1);
    }

    os.indent(indentBy());
    os << llvm::formatv("[{0,4}] ", e->GetIndex());

    switch (e->GetKind()) {
    case InstanceElement::KindTy::COMPOUND: {
      auto *comp = cast<CompoundInstanceElement>(e.get());
      os << "COMPOUND " << comp->opName.getStringRef() << " ["
         << getPipelineName(comp->kPipeValue) << "]";
      os << "\n";
      if (showMemInfo) {
        os.indent(indentBy(kDebugDumpIndentSpaces));
        dumpMemInfoList(os, "def", comp->defVec, state ? &*state : nullptr);
        os << "\n";
        os.indent(indentBy(kDebugDumpIndentSpaces));
        dumpMemInfoList(os, "use", comp->useVec, state ? &*state : nullptr);
        os << "\n";
      }
      break;
    }
    case InstanceElement::KindTy::LOOP: {
      auto *loop = cast<LoopInstanceElement>(e.get());
      os << "LOOP " << getLoopKindName(loop->getLoopKind())
         << " (begin=" << loop->beginId << ", end=" << loop->endId << ")\n";
      break;
    }
    case InstanceElement::KindTy::BRANCH: {
      auto *branch = cast<BranchInstanceElement>(e.get());
      os << "BRANCH " << getBranchKindName(branch->getBranchKind())
         << " (begin=" << branch->beginId << ", branch=" << branch->branchId
         << ", end=" << branch->endId << ")\n";
      break;
    }
    case InstanceElement::KindTy::PLACE_HOLDER: {
      auto *ph = cast<PlaceHolderInstanceElement>(e.get());
      os << "PLACE_HOLDER (parentScopeId=" << ph->parentScopeId;
      if (ph->isVirtualElse)
        os << ", virtualElse";
      os << ")\n";
      break;
    }
    }

    auto dumpOps = [&](llvm::StringRef prefix, const SyncOps &ops) {
      for (const auto *op : ops) {
        if (!op)
          continue;
        if (op->uselessSync && !options.showUselessSync)
          continue;
        os.indent(indentBy(kDebugDumpIndentSpaces));
        os << prefix << ": ";
        dumpSyncOp(os, op, options.showUselessSync);
        os << "\n";
      }
    };

    dumpOps("PRE ", e->pipeBefore);
    dumpOps("POST", e->pipeAfter);

    if (auto *loop = dyn_cast<LoopInstanceElement>(e.get())) {
      if (loop->getLoopKind() == KindOfLoop::LOOP_BEGIN)
        indent += 1;
    }
    if (auto *branch = dyn_cast<BranchInstanceElement>(e.get())) {
      if (branch->getBranchKind() == KindOfBranch::IF_BEGIN ||
          branch->getBranchKind() == KindOfBranch::ELSE_BEGIN)
        indent += 1;
    }
  }
}

void mlir::pto::dumpInsertSyncPhase(llvm::StringRef phase, const SyncIRs &syncIR,
                                   const SyncOperations &syncOperations,
                                   Operation *opForPrinting,
                                   llvm::raw_ostream &os) {
  const unsigned level = getInsertSyncDebugLevel();
  if (level < static_cast<unsigned>(InsertSyncDebugLevel::Phase))
    return;

  unsigned activeOps = 0;
  unsigned setCnt = 0, waitCnt = 0, barrierCnt = 0;
  unsigned blockSetCnt = 0, blockWaitCnt = 0, blockAllCnt = 0;
  for (const auto &group : syncOperations) {
    for (const auto &op : group) {
      if (!op)
        continue;
      if (op->uselessSync)
        continue;
      activeOps++;
      switch (op->GetType()) {
      case SyncOperation::TYPE::SET_EVENT:
        setCnt++;
        break;
      case SyncOperation::TYPE::WAIT_EVENT:
        waitCnt++;
        break;
      case SyncOperation::TYPE::PIPE_BARRIER:
      case SyncOperation::TYPE::PIPE_BARRIER_CUBE:
      case SyncOperation::TYPE::PIPE_BARRIER_VECTOR:
        barrierCnt++;
        break;
      case SyncOperation::TYPE::SYNC_BLOCK_SET:
        blockSetCnt++;
        break;
      case SyncOperation::TYPE::SYNC_BLOCK_WAIT:
        blockWaitCnt++;
        break;
      case SyncOperation::TYPE::SYNC_BLOCK_ALL:
        blockAllCnt++;
        break;
      }
    }
  }

  os << "\n// === [PTOInsertSync Debug] " << phase << " === //\n";
  os << llvm::formatv("// nodes={0}, syncGroups={1}, activeOps={2} "
                      "(set={3}, wait={4}, barrier={5}, blockSet={6}, "
                      "blockWait={7}, blockAll={8})\n",
                      syncIR.size(), syncOperations.size(), activeOps, setCnt,
                      waitCnt, barrierCnt, blockSetCnt, blockWaitCnt,
                      blockAllCnt);

  if (level < static_cast<unsigned>(InsertSyncDebugLevel::SyncIR)) {
    os << "// ========================================= //\n";
    return;
  }

  InsertSyncDumpOptions options;
  const bool showMemInfo =
      level >= static_cast<unsigned>(InsertSyncDebugLevel::Trace);
  options.showMemInfo = showMemInfo;
  options.showUselessSync = showMemInfo;

  dumpSyncIR(os, syncIR, opForPrinting, options, showMemInfo);
  os << "// ========================================= //\n";
}

bool mlir::pto::writeInsertSyncSummary(
    llvm::StringRef status, const SyncIRs &syncIR,
    const SyncOperations &syncOperations, Operation *opForPrinting) {
  if (!isInsertSyncSummaryEnabled())
    return true;

  unsigned activeGroups = 0;
  unsigned activeOps = 0;
  unsigned setCnt = 0;
  unsigned waitCnt = 0;
  unsigned barrierCnt = 0;
  unsigned blockSetCnt = 0;
  unsigned blockWaitCnt = 0;
  unsigned blockAllCnt = 0;
  unsigned pipeAllGroups = 0;
  unsigned loopCarriedGroups = 0;
  unsigned multiEventGroups = 0;
  unsigned eventSyncGroups = 0;
  unsigned compensationOps = 0;
  unsigned requestedEventSlots = 0;
  std::map<std::string, unsigned> pipePairGroups;
  std::map<std::string, std::set<int>> pipePairEventIds;

  for (const auto &group : syncOperations) {
    const SyncOperation *representative = nullptr;
    bool groupPipeAll = false;
    bool groupLoopCarried = false;
    bool groupUsesEvent = false;
    unsigned groupEventIdSlots = 0;
    std::set<int> groupEventIds;
    for (const auto &ownedOp : group) {
      const SyncOperation *op = ownedOp.get();
      if (!op || op->uselessSync)
        continue;
      if (!representative)
        representative = op;
      activeOps++;
      groupPipeAll |= op->GetSrcPipe() == PipelineType::PIPE_ALL ||
                      op->GetDstPipe() == PipelineType::PIPE_ALL;
      groupLoopCarried |= op->GetForEndIndex().has_value();
      if (op->isCompensation)
        compensationOps++;
      for (int eventId : op->eventIds)
        groupEventIds.insert(eventId);
      switch (op->GetType()) {
      case SyncOperation::TYPE::SET_EVENT:
        setCnt++;
        groupUsesEvent = true;
        groupEventIdSlots =
            std::max(groupEventIdSlots, static_cast<unsigned>(op->eventIdNum));
        break;
      case SyncOperation::TYPE::WAIT_EVENT:
        waitCnt++;
        groupUsesEvent = true;
        groupEventIdSlots =
            std::max(groupEventIdSlots, static_cast<unsigned>(op->eventIdNum));
        break;
      case SyncOperation::TYPE::PIPE_BARRIER:
      case SyncOperation::TYPE::PIPE_BARRIER_CUBE:
      case SyncOperation::TYPE::PIPE_BARRIER_VECTOR:
        barrierCnt++;
        break;
      case SyncOperation::TYPE::SYNC_BLOCK_SET:
        blockSetCnt++;
        break;
      case SyncOperation::TYPE::SYNC_BLOCK_WAIT:
        blockWaitCnt++;
        break;
      case SyncOperation::TYPE::SYNC_BLOCK_ALL:
        blockAllCnt++;
        break;
      }
    }
    if (!representative)
      continue;
    activeGroups++;
    pipeAllGroups += groupPipeAll ? 1 : 0;
    loopCarriedGroups += groupLoopCarried ? 1 : 0;
    multiEventGroups += groupUsesEvent && groupEventIdSlots > 1 ? 1 : 0;
    eventSyncGroups += groupUsesEvent ? 1 : 0;
    if (groupUsesEvent)
      requestedEventSlots += groupEventIdSlots;
    const std::string pair =
        (getPipelineName(representative->GetSrcPipe()) + "->" +
         getPipelineName(representative->GetDstPipe()))
            .str();
    pipePairGroups[pair]++;
    if (groupUsesEvent)
      pipePairEventIds[pair].insert(groupEventIds.begin(), groupEventIds.end());
  }

  llvm::json::Object pipePairs;
  for (const auto &[pair, count] : pipePairGroups)
    pipePairs[pair] = static_cast<int64_t>(count);
  llvm::json::Object pipePairIds;
  for (const auto &[pair, eventIds] : pipePairEventIds)
    pipePairIds[pair] = static_cast<int64_t>(eventIds.size());

  std::string functionName = "<unknown>";
  if (opForPrinting) {
    if (auto name = opForPrinting->getAttrOfType<StringAttr>("sym_name"))
      functionName = name.str();
  }

  llvm::json::Object summary;
  summary["schema_version"] = 1;
  summary["function"] = functionName;
  summary["status"] = status.str();
  summary["sync_ir_nodes"] = static_cast<int64_t>(syncIR.size());
  summary["active_sync_groups"] = static_cast<int64_t>(activeGroups);
  summary["active_sync_operations"] = static_cast<int64_t>(activeOps);
  summary["set_operations"] = static_cast<int64_t>(setCnt);
  summary["wait_operations"] = static_cast<int64_t>(waitCnt);
  summary["barrier_operations"] = static_cast<int64_t>(barrierCnt);
  summary["block_set_operations"] = static_cast<int64_t>(blockSetCnt);
  summary["block_wait_operations"] = static_cast<int64_t>(blockWaitCnt);
  summary["block_all_operations"] = static_cast<int64_t>(blockAllCnt);
  summary["pipe_all_groups"] = static_cast<int64_t>(pipeAllGroups);
  summary["loop_carried_groups"] = static_cast<int64_t>(loopCarriedGroups);
  summary["multi_event_groups"] = static_cast<int64_t>(multiEventGroups);
  summary["event_sync_groups"] = static_cast<int64_t>(eventSyncGroups);
  summary["compensation_operations"] = static_cast<int64_t>(compensationOps);
  summary["requested_event_slots"] =
      static_cast<int64_t>(requestedEventSlots);
  summary["pipe_pair_groups"] = std::move(pipePairs);
  summary["pipe_pair_event_ids"] = std::move(pipePairIds);

  static std::mutex outputMutex;
  static std::string initializedPath;
  std::lock_guard<std::mutex> lock(outputMutex);
  const bool append = initializedPath == insertSyncSummaryPathOpt;
  std::error_code error;
  llvm::sys::fs::OpenFlags flags = llvm::sys::fs::OF_Text;
  if (append)
    flags |= llvm::sys::fs::OF_Append;
  llvm::raw_fd_ostream output(insertSyncSummaryPathOpt, error, flags);
  if (error) {
    llvm::errs() << "error: cannot write PTOInsertSync summary '"
                 << insertSyncSummaryPathOpt << "': " << error.message()
                 << "\n";
    return false;
  }
  output << llvm::json::Value(std::move(summary)) << "\n";
  initializedPath = insertSyncSummaryPathOpt;
  return true;
}

bool mlir::pto::writeInsertSyncScheduleGraph(
    llvm::StringRef status, const SyncIRs &syncIR,
    const SyncOperations &syncOperations, Operation *opForPrinting) {
  if (!isInsertSyncScheduleGraphEnabled())
    return true;

  std::optional<AsmState> state;
  if (opForPrinting)
    state.emplace(opForPrinting);
  AsmState *asmState = state ? &*state : nullptr;

  std::string functionName = "<unknown>";
  if (opForPrinting) {
    if (auto name = opForPrinting->getAttrOfType<StringAttr>("sym_name"))
      functionName = name.str();
  }

  llvm::json::Array nodes;
  llvm::json::Array streamEdges;
  SmallVector<unsigned> loopStack;
  SmallVector<unsigned> branchStack;
  std::map<PipelineType, unsigned> previousByPipe;

  for (const auto &ownedElement : syncIR) {
    const InstanceElement *element = ownedElement.get();
    if (!element)
      continue;

    if (auto *loop = dyn_cast<LoopInstanceElement>(element)) {
      if (loop->getLoopKind() == KindOfLoop::LOOP_END && !loopStack.empty())
        loopStack.pop_back();
    }
    if (auto *branch = dyn_cast<BranchInstanceElement>(element)) {
      if (branch->getBranchKind() == KindOfBranch::IF_END &&
          !branchStack.empty())
        branchStack.pop_back();
    }

    llvm::json::Object node;
    node["id"] = static_cast<int64_t>(element->GetIndex());
    node["loop_stack"] = encodeUnsignedStack(loopStack);
    node["branch_stack"] = encodeUnsignedStack(branchStack);

    switch (element->GetKind()) {
    case InstanceElement::KindTy::COMPOUND: {
      auto *compound = cast<CompoundInstanceElement>(element);
      node["kind"] = "operation";
      node["op_name"] = compound->opName.getStringRef().str();
      node["pipe"] = getPipelineName(compound->kPipeValue).str();
      node["macro_phase"] = static_cast<int64_t>(compound->macroOpInstanceId);
      node["defs"] = encodeMemInfoList(compound->defVec, asmState);
      node["uses"] = encodeMemInfoList(compound->useVec, asmState);
      node["operation"] = encodeOperation(compound->elementOp);

      auto previous = previousByPipe.find(compound->kPipeValue);
      if (previous != previousByPipe.end()) {
        llvm::json::Object edge;
        edge["source"] = static_cast<int64_t>(previous->second);
        edge["target"] = static_cast<int64_t>(element->GetIndex());
        edge["pipe"] = getPipelineName(compound->kPipeValue).str();
        streamEdges.push_back(std::move(edge));
      }
      previousByPipe[compound->kPipeValue] = element->GetIndex();
      break;
    }
    case InstanceElement::KindTy::LOOP: {
      auto *loop = cast<LoopInstanceElement>(element);
      node["kind"] = "loop";
      node["loop_kind"] = getLoopKindName(loop->getLoopKind()).str();
      node["begin"] = static_cast<int64_t>(loop->beginId);
      node["end"] = static_cast<int64_t>(loop->endId);
      node["operation"] = encodeOperation(loop->elementOp);
      if (std::optional<int64_t> tripCount =
              getStaticTripCount(loop->elementOp))
        node["static_trip_count"] = *tripCount;
      else
        node["static_trip_count"] = nullptr;
      break;
    }
    case InstanceElement::KindTy::BRANCH: {
      auto *branch = cast<BranchInstanceElement>(element);
      node["kind"] = "branch";
      node["branch_kind"] = getBranchKindName(branch->getBranchKind()).str();
      node["begin"] = static_cast<int64_t>(branch->beginId);
      node["branch"] = static_cast<int64_t>(branch->branchId);
      node["end"] = static_cast<int64_t>(branch->endId);
      node["operation"] = encodeOperation(branch->elementOp);
      break;
    }
    case InstanceElement::KindTy::PLACE_HOLDER: {
      auto *placeholder = cast<PlaceHolderInstanceElement>(element);
      node["kind"] = "placeholder";
      node["parent_scope"] =
          static_cast<int64_t>(placeholder->parentScopeId);
      node["virtual_else"] = placeholder->isVirtualElse;
      break;
    }
    }
    nodes.push_back(std::move(node));

    if (auto *loop = dyn_cast<LoopInstanceElement>(element)) {
      if (loop->getLoopKind() == KindOfLoop::LOOP_BEGIN)
        loopStack.push_back(loop->GetIndex());
    }
    if (auto *branch = dyn_cast<BranchInstanceElement>(element)) {
      if (branch->getBranchKind() == KindOfBranch::IF_BEGIN)
        branchStack.push_back(branch->GetIndex());
      else if (branch->getBranchKind() == KindOfBranch::ELSE_BEGIN) {
        if (!branchStack.empty())
          branchStack.pop_back();
        branchStack.push_back(branch->GetIndex());
      }
    }
  }

  llvm::json::Array syncEdges;
  llvm::json::Array syncGroups;
  for (size_t groupIndex = 0; groupIndex < syncOperations.size();
       ++groupIndex) {
    const auto &group = syncOperations[groupIndex];
    SmallVector<unsigned> sources;
    SmallVector<unsigned> targets;
    SmallVector<Value> roots;
    llvm::json::Array operations;
    const SyncOperation *representative = nullptr;
    bool loopCarried = false;

    for (const auto &ownedSync : group) {
      const SyncOperation *sync = ownedSync.get();
      if (!sync || sync->uselessSync)
        continue;
      if (!representative)
        representative = sync;
      loopCarried |= sync->GetForEndIndex().has_value();
      for (Value root : sync->depRootBuffers) {
        if (!llvm::is_contained(roots, root))
          roots.push_back(root);
      }

      llvm::json::Object operation;
      operation["type"] = SyncOperation::TypeName(sync->GetType());
      operation["node"] = static_cast<int64_t>(sync->GetSyncIRIndex());
      operation["dependency_node"] =
          static_cast<int64_t>(sync->GetDepSyncIRIndex());
      operation["compensation"] = sync->isCompensation;
      operation["event_slots"] = static_cast<int64_t>(sync->eventIdNum);
      llvm::json::Array eventIds;
      for (int eventId : sync->eventIds)
        eventIds.push_back(static_cast<int64_t>(eventId));
      operation["event_ids"] = std::move(eventIds);
      if (sync->GetForEndIndex())
        operation["loop_end"] =
            static_cast<int64_t>(*sync->GetForEndIndex());
      else
        operation["loop_end"] = nullptr;
      operations.push_back(std::move(operation));

      if (sync->isSyncSetType()) {
        if (!llvm::is_contained(sources, sync->GetSyncIRIndex()))
          sources.push_back(sync->GetSyncIRIndex());
      } else if (sync->isSyncWaitType()) {
        if (!llvm::is_contained(targets, sync->GetSyncIRIndex()))
          targets.push_back(sync->GetSyncIRIndex());
      } else if (sync->isBarrierType()) {
        if (!llvm::is_contained(sources, sync->GetDepSyncIRIndex()))
          sources.push_back(sync->GetDepSyncIRIndex());
        if (!llvm::is_contained(targets, sync->GetSyncIRIndex()))
          targets.push_back(sync->GetSyncIRIndex());
      }
    }

    if (!representative)
      continue;

    llvm::json::Object groupObject;
    groupObject["id"] = static_cast<int64_t>(groupIndex);
    groupObject["sync_index"] =
        static_cast<int64_t>(representative->GetSyncIndex());
    groupObject["src_pipe"] =
        getPipelineName(representative->GetSrcPipe()).str();
    groupObject["dst_pipe"] =
        getPipelineName(representative->GetDstPipe()).str();
    groupObject["loop_carried"] = loopCarried;
    groupObject["root_buffers"] = encodeValues(roots, asmState);
    groupObject["operations"] = std::move(operations);
    syncGroups.push_back(std::move(groupObject));

    for (unsigned source : sources) {
      for (unsigned target : targets) {
        llvm::json::Object edge;
        edge["source"] = static_cast<int64_t>(source);
        edge["target"] = static_cast<int64_t>(target);
        edge["group"] = static_cast<int64_t>(groupIndex);
        edge["src_pipe"] =
            getPipelineName(representative->GetSrcPipe()).str();
        edge["dst_pipe"] =
            getPipelineName(representative->GetDstPipe()).str();
        edge["loop_carried"] = loopCarried;
        edge["root_buffers"] = encodeValues(roots, asmState);
        syncEdges.push_back(std::move(edge));
      }
    }
  }

  llvm::json::Object record;
  record["schema_version"] = 1;
  record["function"] = functionName;
  record["status"] = status.str();
  record["node_count"] = static_cast<int64_t>(syncIR.size());
  record["duration_model"] = "unestimated";
  record["nodes"] = std::move(nodes);
  record["stream_edges"] = std::move(streamEdges);
  record["sync_groups"] = std::move(syncGroups);
  record["sync_edges"] = std::move(syncEdges);

  static std::mutex outputMutex;
  static std::string initializedPath;
  return appendJsonLine(insertSyncScheduleGraphPathOpt, std::move(record),
                        outputMutex, initializedPath,
                        "PTOInsertSync schedule graph");
}
