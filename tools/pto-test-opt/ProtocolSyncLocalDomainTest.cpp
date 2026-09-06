// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

// Independent enumeration of complete-box layouts and physical storage domains.
#include "PTO/Transforms/InsertSync/LegacySyncIRAdapter.h"
#include "PTO/Transforms/ProtocolSync/LocalMemoryAnalysis.h"
#include "PTO/Transforms/ProtocolSync/MixedProtocolPlan.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <set>
#include <string>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {
bool check(bool condition, StringRef message)
{
    if (!condition) {
        llvm::errs() << "FAIL local domain: " << message << '\n';
    }
    return condition;
}

bool layoutOracle(MLIRContext& context)
{
    for (const char* space : {"mat", "left", "right", "acc"}) {
        for (unsigned layout = 0; layout < 3; ++layout) {
            for (unsigned width : {1U, 2U, 4U}) {
                for (unsigned fractal : {512U, 1024U}) {
                    for (unsigned compact : {0U, 1U}) {
                        const unsigned boxRows = fractal == 1024 || layout != 1 ? 16 : 32 / width;
                        const unsigned boxCols = fractal == 1024 || layout == 1 ? 16 : 32 / width;
                        const unsigned rows = 2 * boxRows;
                        const unsigned cols = 3 * boxCols;
                        const std::string type = "!pto.tile_buf<loc=" + std::string(space) +
                                                 (width == 1 ? ", dtype=i" : ", dtype=f") + std::to_string(8 * width) +
                                                 ", rows=" + std::to_string(rows) + ", cols=" + std::to_string(cols) +
                                                 ", v_row=" + std::to_string(rows) + ", v_col=" + std::to_string(cols) +
                                                 ", blayout=" + (layout == 0 ? "col_major" : "row_major") +
                                                 ", slayout=" + (layout == 1 ? "col_major" : "row_major") +
                                                 ", fractal=" + std::to_string(fractal) +
                                                 ", pad=0, compact=" + std::to_string(compact) + ">";
                        const std::string source =
                            "module { func.func @geometry() { %base = arith.constant 8192 : i64\n"
                            "%a = pto.alloc_tile addr = %base : " +
                            type + "\nreturn } }";
                        auto module = parseSourceString<ModuleOp>(source, &context);
                        if (!module) {
                            return false;
                        }
                        auto function = *module->getOps<func::FuncOp>().begin();
                        auto allocation = *function.getOps<AllocTileOp>().begin();
                        SyncAccess access;
                        access.value = allocation.getResult();
                        access.storage.space =
                            cast<AddressSpaceAttr>(allocation.getResult().getType().getMemorySpace()).getAddressSpace();
                        const auto region = recoverLocalAccessRegion(access);
                        if (fractal == 1024 && width != 4) {
                            if (!check(region.precision == SyncRegionPrecision::Unknown, "unqualified C-fractal")) {
                                return false;
                            }
                            continue;
                        }
                        std::set<std::uint64_t> bytes;
                        for (unsigned r = 0; r < rows; ++r) {
                            for (unsigned c = 0; c < cols; ++c) {
                                const unsigned box =
                                    layout == 0 ? (c / boxCols) * 2 + r / boxRows : (r / boxRows) * 3 + c / boxCols;
                                const unsigned within = layout == 1 ? (c % boxCols) * boxRows + r % boxRows :
                                                                      (r % boxRows) * boxCols + c % boxCols;
                                for (unsigned byte = 0; byte < width; ++byte) {
                                    bytes.insert(8192 + (box * boxRows * boxCols + within) * width + byte);
                                }
                            }
                        }
                        const bool equal = region.precision == SyncRegionPrecision::Conservative &&
                                           region.space == access.storage.space &&
                                           region.interval.begin == *bytes.begin() &&
                                           region.interval.size == bytes.size() &&
                                           *bytes.rbegin() + 1 == region.interval.begin + region.interval.size;
                        if (!check(equal, "complete-box byte-set reconstruction")) {
                            return false;
                        }
                    }
                }
            }
        }
    }
    return true;
}

bool unknownGeometryOracle(MLIRContext& context)
{
    // Some cases deliberately violate operation-level shape constraints: the
    // footprint helper must independently refuse an unqualified byte bound.
    for (unsigned variant = 0; variant < 8; ++variant) {
        const std::string rows = variant == 0 ? "17" : "32";
        const std::string validRows = variant == 1 ? "16" : variant == 2 ? "?" : rows;
        const std::string compact = variant == 1 || variant == 2 ? "1" : variant == 3 ? "2" : "0";
        const std::string type =
            "!pto.tile_buf<loc=mat, dtype=f16, rows=" + rows + ", cols=32, v_row=" + validRows +
            ", v_col=32, blayout=col_major, slayout=" + (variant == 4 ? "col_major" : "row_major") +
            ", fractal=" + (variant == 5 ? "32" : "512") + ", pad=0, compact=" + compact + ">";
        const std::string address = variant == 6 ? "%arg" : "%base";
        const std::string source = "module { func.func @unknown(%arg: i64) { %base = arith.constant 8192 : i64\n"
                                   "%a = pto.alloc_tile addr = " +
                                   address + " : " + type + "\nreturn } }";
        auto module = parseSourceString<ModuleOp>(source, ParserConfig(&context, false));
        if (!module) {
            return false;
        }
        auto function = *module->getOps<func::FuncOp>().begin();
        auto allocation = *function.getOps<AllocTileOp>().begin();
        SyncAccess access;
        access.value = allocation.getResult();
        access.storage.space = variant == 7 ? AddressSpace::LEFT : AddressSpace::MAT;
        const auto region = recoverLocalAccessRegion(access);
        if (!check(region.precision == SyncRegionPrecision::Unknown, "unqualified geometry remains unknown")) {
            return false;
        }
    }
    return true;
}

constexpr StringLiteral kCube = R"mlir(
module attributes {pto.target_arch = "a3"} {
 func.func @domains(%in: !pto.partition_tensor_view<16x16xf16>, %out: !pto.partition_tensor_view<16x16xf32>)
 attributes {pto.kernel_kind = #pto.kernel_kind<cube>, pto.gm_alias = "assume-disjoint-arguments"} {
  %zero = arith.constant 0 : i64
  %a = pto.alloc_tile addr = %zero : !pto.tile_buf<mat, 16x16xf16>
  %b = pto.alloc_tile addr = %zero : !pto.tile_buf<mat, 16x16xf16>
  %left = pto.alloc_tile addr = %zero : !pto.tile_buf<left, 16x16xf16>
  %right = pto.alloc_tile addr = %zero : !pto.tile_buf<right, 16x16xf16>
  %acc = pto.alloc_tile addr = %zero : !pto.tile_buf<acc, 16x16xf32>
  pto.tload ins(%in : !pto.partition_tensor_view<16x16xf16>) outs(%a : !pto.tile_buf<mat, 16x16xf16>)
  pto.tmov ins(%a : !pto.tile_buf<mat, 16x16xf16>) outs(%left : !pto.tile_buf<left, 16x16xf16>)
  pto.tload ins(%in : !pto.partition_tensor_view<16x16xf16>) outs(%b : !pto.tile_buf<mat, 16x16xf16>)
  pto.tmov ins(%b : !pto.tile_buf<mat, 16x16xf16>) outs(%right : !pto.tile_buf<right, 16x16xf16>)
  pto.tmatmul ins(%left, %right : !pto.tile_buf<left, 16x16xf16>, !pto.tile_buf<right, 16x16xf16>)
    outs(%acc : !pto.tile_buf<acc, 16x16xf32>)
  pto.tmatmul.acc ins(%acc, %left, %right : !pto.tile_buf<acc, 16x16xf32>,
    !pto.tile_buf<left, 16x16xf16>, !pto.tile_buf<right, 16x16xf16>) outs(%acc : !pto.tile_buf<acc, 16x16xf32>)
  pto.tstore ins(%acc : !pto.tile_buf<acc, 16x16xf32>) outs(%out : !pto.partition_tensor_view<16x16xf32>)
  return
 }
})mlir";

bool domainOracle(MLIRContext& context)
{
    auto module = parseSourceString<ModuleOp>(kCube, &context);
    if (!module) {
        return false;
    }
    auto function = *module->getOps<func::FuncOp>().begin();
    LegacySyncIRAdapter adapter;
    LegacySyncSnapshot snapshot;
    if (failed(adapter.buildSnapshot(function, snapshot))) {
        return false;
    }
    StructuredSyncIR schedule(function);
    if (failed(StructuredSyncIRBuilder(adapter.buildSemanticContext(snapshot)).build(function, schedule))) {
        return false;
    }
    auto memory = analyzeLocalMemory(schedule);
    if (!check(
            succeeded(memory) && memory->boundary.empty() && memory->atoms.size() == 4, "four separate cube spaces")) {
        return false;
    }
    for (const auto& atom : memory->atoms) {
        std::set<SyncStorageFamilyId> families;
        for (auto id : atom.accesses) {
            const auto& access = *schedule.findAccess(id);
            families.insert(access.family);
            const bool correctSpace = access.storage.space == atom.space && atom.core == SyncPhysicalCore::Cube;
            const bool accProtected = atom.space != AddressSpace::ACC || !memory->coveredAccesses.test(id);
            if (!check(correctSpace && accProtected, "domain identity and effect completeness are separate")) {
                return false;
            }
        }
        const bool aliasLost = atom.space == AddressSpace::MAT && families.size() != 2;
        if (!check(!aliasLost, "same-space physical alias retained")) {
            return false;
        }
    }
    auto stages = analyzePipelineStages(schedule);
    if (failed(stages)) {
        return false;
    }
    auto timelines = analyzeStorageTimelines(schedule, *stages);
    auto channels = analyzeChannels(schedule, *stages, timelines);
    auto result = interpretSelectedWorld(schedule, *stages, timelines, channels, {});
    if (failed(result)) {
        return false;
    }
    const bool accUnresolved = llvm::any_of(
        result->obligations, [](const auto& obligation) { return obligation.kind == SyncObligationKind::AccConflict; });
    return check(accUnresolved, "spatial recovery must not erase ACC protection");
}

bool ownerOracle(MLIRContext& context)
{
    for (unsigned variant = 0; variant < 3; ++variant) {
        const std::string kind = variant == 0 ? "vector" : "cube";
        const std::string attributes =
            variant == 1 ? "" : " attributes {pto.kernel_kind = #pto.kernel_kind<" + kind + ">}";
        const std::string source =
            "module { func.func @owner(%in: !pto.partition_tensor_view<16x16xf16>)" + attributes + " {\n" +
            (variant == 2 ? "pto.section.cube {\n" : "") +
            "%base = arith.constant 0 : i64\n"
            "%a = pto.alloc_tile addr = %base : !pto.tile_buf<mat, 16x16xf16>\n"
            "pto.tload ins(%in : !pto.partition_tensor_view<16x16xf16>) outs(%a : !pto.tile_buf<mat, 16x16xf16>)\n" +
            (variant == 2 ? "}\n" : "") + "return } }";
        auto module = parseSourceString<ModuleOp>(source, ParserConfig(&context, false));
        if (!module) {
            return false;
        }
        auto function = *module->getOps<func::FuncOp>().begin();
        LegacySyncIRAdapter adapter;
        LegacySyncSnapshot snapshot;
        if (failed(adapter.buildSnapshot(function, snapshot))) {
            continue;
        }
        StructuredSyncIR schedule(function);
        if (failed(StructuredSyncIRBuilder(adapter.buildSemanticContext(snapshot)).build(function, schedule))) {
            continue;
        }
        auto memory = analyzeLocalMemory(schedule);
        const bool unsupported = failed(memory) || (!memory->boundary.empty() && memory->coveredAccesses.none());
        if (!check(unsupported, "unknown, mismatched or section-scoped owner must not be certified")) {
            return false;
        }
    }
    return true;
}
} // namespace

bool testProtocolSyncLocalDomains(MLIRContext& context)
{
    const bool valid =
        layoutOracle(context) && unknownGeometryOracle(context) && domainOracle(context) && ownerOracle(context);
    if (!valid) {
        return false;
    }
    llvm::outs() << "protocol-sync local domains: layout byte sets, separate spaces and retained ACC effects pass\n";
    return true;
}
