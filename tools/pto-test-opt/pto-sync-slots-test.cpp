// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"
#include "PTO/Transforms/InsertSync/SyncPhysicalFacts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>
#include <set>

using namespace mlir;
using namespace mlir::pto;
using llvm::json::Array;
using llvm::json::Object;
int main(int argc, char **argv) {
  if (argc != 2) return 2;
  DialectRegistry registry;
  registry.insert<PTODialect, func::FuncDialect, scf::SCFDialect, arith::ArithDialect>();
  MLIRContext context(registry, MLIRContext::Threading::DISABLED);
  auto module = parseSourceFile<ModuleOp>(argv[1], &context);
  if (!module) return 2;
  Array functions;
  unsigned checks = 0;
  bool passed = true;
  auto check = [&](bool value) { ++checks; passed &= value; };
  // Large disjoint/touching populations must not pay for a Cartesian product.
  // These synthetic maps satisfy the same nonoverlapping, aligned interval
  // contract as qualified native maps, without needing thousands of IR ops.
  for (unsigned count : {32u, 256u, 1024u}) {
    SyncPhysicalSlotMapping left{{}, {}, 32, AddressSpace::VEC};
    SyncPhysicalSlotMapping right{{}, {}, 32, AddressSpace::VEC};
    for (unsigned i = 0; i < count; ++i) {
      left.bases.push_back(uint64_t(i) * 128);
      right.bases.push_back(uint64_t(i) * 128);
    }
    uint64_t charged = 0;
    auto overlapping = overlappingSyncPhysicalSlots(left, right, [&](uint64_t n) {
      charged += n; return true;
    });
    check(bool(overlapping) && overlapping->size() == count);
    if (overlapping)
      for (auto [i, j] : *overlapping) check(i == j);
    if (count >= 256) check(charged < uint64_t(count) * count);
    const uint64_t overlapWork = charged;
    for (uint64_t &base : right.bases) base += left.bytes;
    charged = 0;
    auto touching = overlappingSyncPhysicalSlots(left, right, [&](uint64_t n) {
      charged += n; return true;
    });
    check(bool(touching) && touching->empty());
    check(charged + count == overlapWork);
    right.scope = AddressSpace::MAT;
    charged = 0;
    auto separateSpace = overlappingSyncPhysicalSlots(left, right, [&](uint64_t n) {
      charged += n; return true;
    });
    check(bool(separateSpace) && separateSpace->empty() && charged == 1);
  }
  for (auto function : module->getOps<func::FuncOp>()) {
    if (function.isDeclaration()) continue;
    SyncIRs ir;
    Buffer2MemInfoMap buffers;
    MemoryDependentAnalyzer memory;
    PTOIRTranslator translator(ir, memory, buffers, function, SyncAnalysisMode::NORMALSYNC);
    translator.Build();
    auto physical = importSyncPhysicalFacts(function, ir, 1000000);
    Array records, layouts, overlaps;
    SmallVector<SyncPhysicalSlotMapping> mappings;
    SmallVector<unsigned> recordIds;
    function.walk([&](Operation *op) {
      if (auto multi = dyn_cast<AllocMultiTileOp>(op)) {
        auto layout = getPTOStaticMultiTileSlotLayout(multi.getResult().getType().getSlotType());
        Object row{{"qualified", succeeded(layout)}};
        if (succeeded(layout)) {
          row["bytes"] = int64_t(layout->footprintBytes);
          row["alignment"] = int64_t(layout->alignmentBytes);
          row["stride"] = int64_t(layout->strideBytes);
        }
        layouts.push_back(std::move(row));
      }
      for (Value value : op->getResults()) {
        auto found = buffers.find(value);
        if (found == buffers.end()) continue;
        for (const auto &info : found->second) {
          auto mapping = qualifySyncPhysicalSlots(info.get(), buffers);
          Object record{{"operation", op->getName().getStringRef()}, {"qualified", bool(mapping)},
            {"qualification_work", int64_t(estimateSyncPhysicalSlotQualificationWork(info.get(), buffers))}};
          if (mapping) {
            record["selector"] = bool(mapping->selector);
            record["bytes"] = int64_t(mapping->bytes);
            record["scope"] = int64_t(mapping->scope);
            Array addresses;
            for (uint64_t base : mapping->bases) addresses.push_back(std::to_string(base));
            record["bases"] = std::move(addresses);
            recordIds.push_back(records.size()); mappings.push_back(*mapping);
            auto wrong = info->clone();
            wrong->aliasesUnknownRange = true;
            check(!qualifySyncPhysicalSlots(wrong.get(), buffers));
            wrong = info->clone(); wrong->baseAddresses[0] = std::numeric_limits<uint64_t>::max();
            check(!qualifySyncPhysicalSlots(wrong.get(), buffers));
            if (mapping->selector) {
              wrong = info->clone(); ++wrong->allocateSize;
              check(!qualifySyncPhysicalSlots(wrong.get(), buffers));
              wrong = info->clone(); ++wrong->baseAddresses[0];
              check(!qualifySyncPhysicalSlots(wrong.get(), buffers));
            }
          }
          records.push_back(std::move(record));
        }
      }
    });
    for (unsigned a = 0; a < mappings.size(); ++a) for (unsigned b = 0; b < mappings.size(); ++b) {
      const auto &left = mappings[a], &right = mappings[b];
      uint64_t charged = 0;
      auto actual = overlappingSyncPhysicalSlots(left, right, [&](uint64_t n) { charged += n; return true; });
      check(bool(actual));
      if (!actual) continue;
      std::vector<std::pair<unsigned,unsigned>> expected;
      // Intentionally independent Cartesian oracle for the native sweep.
      if (left.scope == right.scope)
        for (unsigned i = 0; i < left.bases.size(); ++i)
          for (unsigned j = 0; j < right.bases.size(); ++j)
            if (left.bases[i] < right.bases[j] + right.bytes && right.bases[j] < left.bases[i] + left.bytes)
              expected.emplace_back(i,j);
      std::set<std::pair<unsigned,unsigned>> actualSet(actual->begin(), actual->end());
      check(actualSet.size() == actual->size());
      check(actualSet == std::set<std::pair<unsigned,unsigned>>(expected.begin(), expected.end()));
      check(!overlappingSyncPhysicalSlots(left, right, [](uint64_t) { return false; }));
      uint64_t remaining = charged;
      auto exact = overlappingSyncPhysicalSlots(left, right, [&](uint64_t n) {
        if (n > remaining) return false;
        remaining -= n; return true;
      });
      check(bool(exact) && *exact == *actual && !remaining);
      if (charged) {
        remaining = charged - 1;
        check(!overlappingSyncPhysicalSlots(left, right, [&](uint64_t n) {
          if (n > remaining) return false;
          remaining -= n; return true;
        }));
      }
      Array pairs;
      for (auto [i,j] : actualSet) pairs.push_back(Array{int64_t(i),int64_t(j)});
      overlaps.push_back(Object{{"left",int64_t(recordIds[a])},{"right",int64_t(recordIds[b])},{"pairs",std::move(pairs)}});
    }
    functions.push_back(Object{{"function",function.getSymName()},
      {"physical_complete",physical.status == SyncPhysicalFacts::Status::Complete}, {"reason",physical.reason},
      {"records",std::move(records)},{"layouts",std::move(layouts)},{"overlaps",std::move(overlaps)}});
  }
  llvm::outs() << llvm::formatv("{0:2}\n", llvm::json::Value(Object{{"passed",passed},{"checks",int64_t(checks)},
                                                                   {"functions",std::move(functions)}}));
  return passed ? 0 : 1;
}
