// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Sufficient allocation witnesses, including optional endpoint participation.
#include "PTO/Transforms/FrontierSynch/BoundedLifetimeAllocation.h"
#include "PTO/Transforms/FrontierSynch/HardwareProtection.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/Support/raw_ostream.h"
namespace fs = mlir::pto::frontiersynch;
namespace {
std::optional<int64_t> budget(mlir::DictionaryAttr certificate, unsigned groups = 1)
{
    if (!certificate) { return std::nullopt; }
    auto list = certificate.getAs<mlir::ArrayAttr>("groups");
    if (!list || list.size() != groups || !groups) { return std::nullopt; }
    auto first = mlir::dyn_cast<mlir::DictionaryAttr>(list[0]);
    auto value = first ? first.getAs<mlir::IntegerAttr>("budget") : mlir::IntegerAttr{};
    return value ? std::optional<int64_t>(value.getInt()) : std::nullopt;
}
fs::LifetimeWindowInput window(fs::RegionExpressions& e, unsigned slots)
{
    fs::LifetimeWindowInput result;
    result.sites = 2; result.span = slots;
    for (unsigned d = 0; d <= slots; ++d) {
        // One writer and one optionally participating reader of its bank.
        result.payloads.push_back({0, e.boolean(true)});
        result.payloads.push_back({1, e.boolean(true)});
        result.accesses.push_back({2 * d, d % slots, e.boolean(false), e.boolean(true), 0});
        result.accesses.push_back({2 * d + 1, d % slots, e.boolean(true), e.boolean(false), 0});
    }
    return result;
}
bool ordinary(mlir::func::FuncOp function)
{
    fs::RegionExpressions e;
    auto w = window(e, 2);
    std::vector<fs::GuardedRankEdge> demands{{0, 1, e.boolean(true)}};
    auto certify = [&](llvm::ArrayRef<uint8_t> always) {
        return fs::boundedLifetimeAllocationCertificate(function, e, w, always, demands, 5);
    };
    // An optional reader releases the bank to the mandatory producer two
    // iterations later. No intermediate optional execution may be assumed.
    if (budget(certify({1, 0})) != 2 || certify({0, 0})) { return false; }
    // A prerequisite between producers cannot invent an earlier return
    // from the optional reader before its bank is touched again.
    w.prerequisites.push_back({0, 2, e.boolean(true)});
    if (budget(certify({1, 0})) != 2) { return false; }
    auto malformed = w;
    malformed.accesses[0].payload = malformed.payloads.size();
    if (fs::boundedLifetimeAllocationCertificate(function, e, malformed, {1, 0}, demands, 0)) { return false; }
    malformed = w;
    malformed.operations = {1};
    return !fs::boundedLifetimeAllocationCertificate(function, e, malformed, {1, 0}, demands, 0);
}
bool palettes(mlir::func::FuncOp function)
{
    fs::RegionExpressions e;
    auto w = window(e, 1);
    // The producer contract permits at most one active partner on a pipe.
    // Alternative records of one source site share a single cyclic palette.
    std::vector<fs::GuardedRankEdge> demands{{0, 1, e.boolean(true)}, {0, 3, e.boolean(false)}};
    auto certificate = fs::boundedLifetimeAllocationCertificate(function, e, w, {1, 0}, demands, 0);
    if (budget(certificate) != 2) { return false; }
    auto group = mlir::cast<mlir::DictionaryAttr>(certificate.getAs<mlir::ArrayAttr>("groups")[0]);
    auto records = group.getAs<mlir::DenseI64ArrayAttr>("records");
    auto rules = group.getAs<mlir::ArrayAttr>("tuple_rules");
    if (!records || records.size() != 2 || records[0] != 0 || records[1] != 1 || !rules || rules.size() != 2) {
        return false;
    }
    auto rule = mlir::cast<mlir::DictionaryAttr>(rules[0]);
    if (rule.getAs<mlir::IntegerAttr>("coordinate_count").getInt() != 1) { return false; }
    // An empty demand set is a valid empty physical plan even without any
    // guaranteed execution; there is no event lane to justify.
    auto empty = fs::boundedLifetimeAllocationCertificate(function, e, w, {0, 0}, {}, 0);
    return empty && empty.getAs<mlir::ArrayAttr>("groups").empty();
}
bool protectedModes(mlir::func::FuncOp function)
{
    fs::RegionExpressions e;
    fs::LifetimeWindowInput w;
    w.sites = 3; w.span = 1;
    // Two mandatory accumulator phases on pipe 1 share a protected writer
    // group. Pipe 0 remains an ordinary conflicting access to their cell.
    for (unsigned i = 0; i < 6; ++i) {
        w.payloads.push_back({i % 3 ? 1U : 0U, e.boolean(true)});
        w.accesses.push_back({i, 0, e.boolean(true), e.boolean(true), i % 3 ? 1U : 0U});
    }
    std::vector<fs::GuardedRankEdge> demands{{0, 1, e.boolean(true)}};
    auto combined = fs::boundedLifetimeAllocationCertificate(function, e, w, {1, 1, 1}, demands, 0);
    auto split = w;
    split.accesses.clear();
    for (auto access : w.accesses) {
        access.write = e.boolean(false); split.accesses.push_back(access);
        access.read = e.boolean(false); access.write = e.boolean(true); split.accesses.push_back(access);
    }
    auto separated = fs::boundedLifetimeAllocationCertificate(function, e, split, {1, 1, 1}, demands, 0);
    return combined && combined == separated;
}
} // namespace
int runBoundedLifetimeAllocationChecks()
{
    mlir::MLIRContext context;
    context.loadDialect<mlir::func::FuncDialect>();
    mlir::Builder builder(&context);
    mlir::OwningOpRef<mlir::func::FuncOp> function(mlir::func::FuncOp::create(
        builder.getUnknownLoc(), "bounded_allocation_checks", builder.getFunctionType({}, {})));
    if (!ordinary(*function) || !palettes(*function) || !protectedModes(*function)) {
        llvm::errs() << "bounded lifetime allocation certificate check failed\n";
        return 1;
    }
    llvm::outs() << "bounded lifetime allocation: optional endpoints, reuse gaps and mode merging passed\n";
    return 0;
}
