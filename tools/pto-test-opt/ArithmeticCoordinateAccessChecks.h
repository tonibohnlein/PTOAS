// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_ARITHMETIC_COORDINATE_ACCESS_CHECKS_H
#define PTO_ARITHMETIC_COORDINATE_ACCESS_CHECKS_H
#include "PTO/Transforms/FrontierSynch/Shared/PhaseIndex.h"
#include "PTO/Transforms/FrontierSynch/FrontierAnalysis.h"
namespace {
// Requests are isolated by immutable session owner, including after reset.
// Both sessions analyze the same original region; only allocation differs.
bool arithmeticCoordinatePolicyIsolation(func::FuncOp function)
{
    using Policy = fs::ArithmeticRegionalAllocationPolicy;
    fs::FrontierAnalysis historical(function), candidate(function, Policy::CoordinateCandidate);
    auto initialize = [&](fs::FrontierAnalysis& session) {
        const fs::ArithmeticLimits profile{2, 4, 4, 4096};
        return succeeded(session.initialize()) &&
            succeeded(session.configureArithmeticProfiles({profile}, {profile}));
    };
    if (!initialize(historical) || !initialize(candidate)) { return false; }
    auto loop = *function.getOps<scf::ForOp>().begin();
    auto request = [&](fs::FrontierAnalysis& session) {
        const auto* program = session.result();
        auto node = llvm::find_if(program->nodes, [&](const auto& item) { return item.anchor == loop; });
        if (node == program->nodes.end()) { return fs::AnalysisOutcome{}; }
        fs::AnalysisRequest needed;
        needed.region = static_cast<std::size_t>(node - program->nodes.begin());
        needed.needs.queries = needed.needs.selectors = true;
        return session.analyzeArithmeticRegional(needed);
    };
    auto first = request(historical), second = request(candidate);
    auto retry = request(candidate);
    if (first.status != fs::AnalysisStatus::Ready || second.status != fs::AnalysisStatus::Ready ||
        !first.regionalExports || !second.regionalExports || first.mathematical == second.mathematical ||
        first.regionalExports == second.regionalExports || retry.mathematical != second.mathematical ||
        retry.regionalExports != second.regionalExports ||
        historical.arithmeticAllocationPolicy() != Policy::SingleId ||
        candidate.arithmeticAllocationPolicy() != Policy::CoordinateCandidate ||
        succeeded(historical.prepareLogical(second)) || succeeded(candidate.prepareLogical(first))) { return false; }
    auto checkAllocation = [&](fs::PreparedLogicalPlan& plan, bool wider) {
        fs::prepareAllocationSupport(plan);
        if (!wider) { return !plan.regionalAllocation; }
        return plan.regionalAllocation && plan.regionalAllocation->groups.size() == 2 &&
            llvm::all_of(plan.regionalAllocation->groups, [](const auto& group) { return group.budget == 3; });
    };
    for (bool wider : {false, true}) {
        auto& session = wider ? candidate : historical;
        auto& answer = wider ? second : first;
        auto plan = session.prepareLogical(answer);
        auto exported = answer.regionalExports->prepare();
        if (failed(plan) || failed(exported) || !checkAllocation(**plan, wider) ||
            !checkAllocation(**exported, wider)) { return false; }
    }
    candidate.invalidate();
    // An answer-owned callback keeps its policy after the session cache dies.
    auto retained = second.regionalExports->prepare();
    if (failed(retained) || !checkAllocation(**retained, true) ||
        candidate.arithmeticAllocationPolicy() != Policy::CoordinateCandidate || !initialize(candidate)) {
        return false;
    }
    auto fresh = request(candidate);
    if (fresh.status != fs::AnalysisStatus::Ready || fresh.mathematical == second.mathematical ||
        fresh.regionalExports == second.regionalExports) { return false; }
    auto plan = candidate.prepareLogical(fresh);
    return succeeded(plan) && checkAllocation(**plan, true);
}
// Real storage extraction supplies every relation. With odd IVs and four
// slots, the read of slot (i+2)%4 pairs with the preceding writer at i-2;
// the next writer of that slot is at i+2. Both directions therefore match
// the next-visit command graph, independently specified by the trace oracle.
bool arithmeticCoordinateAccessIntegration(MLIRContext& context)
{
    constexpr const char* ir = R"mlir(
      !mat = !pto.tile_buf<mat, 16x16xf16, blayout=col_major, slayout=row_major>
      !left = !pto.tile_buf<left, 16x16xf16, slayout=row_major>
      !right = !pto.tile_buf<right, 16x16xf16, slayout=col_major>
      !acc = !pto.tile_buf<acc, 16x16xf32, blayout=col_major, slayout=row_major, fractal=1024>
      !ring = !pto.multi_tile_buf<!pto.tile_buf<left, 16x16xf16, slayout=row_major>, count=4>
      module attributes {pto.target_arch = "a3"} {
        func.func @coordinate_access() -> index {
          %base = arith.constant 0 : i64
          %other = arith.constant 2048 : i64
          %zero = arith.constant 0 : index
          %two = arith.constant 2 : index
          %three = arith.constant 3 : index
          %four = arith.constant 4 : index
          %limit = arith.constant 16 : index
          %accbytes = arith.constant 1024 : index
          %accbase = arith.constant 16384 : index
          %mat = pto.alloc_tile addr = %base : !mat
          %right = pto.alloc_tile addr = %base : !right
          %ring0 = pto.alloc_multi_tile addr = %base : !ring
          %ring1 = pto.alloc_multi_tile addr = %other : !ring
          scf.for %i = %three to %limit step %two {
            %slot = arith.remui %i, %four : index
            %next = arith.addi %i, %two overflow<nsw> : index
            %readslot = arith.remui %next, %four : index
            %write = pto.multi_tile_get %ring0[%slot] : !ring -> !left
            %read = pto.multi_tile_get %ring0[%readslot] : !ring -> !left
            %offset = arith.muli %i, %accbytes overflow<nsw> : index
            %address = arith.index_cast %offset : index to i64
            %acc = pto.alloc_tile addr = %address : !acc
            pto.textract ins(%mat, %zero, %zero : !mat, index, index) outs(%write : !left)
            pto.tmatmul ins(%read, %right : !left, !right) outs(%acc : !acc)
          }
          scf.for %j = %three to %limit step %two {
            %slot = arith.remui %j, %four : index
            %next = arith.addi %j, %two overflow<nsw> : index
            %readslot = arith.remui %next, %four : index
            %write = pto.multi_tile_get %ring1[%slot] : !ring -> !left
            %read = pto.multi_tile_get %ring1[%readslot] : !ring -> !left
            %offset = arith.muli %j, %accbytes overflow<nsw> : index
            %shifted = arith.addi %offset, %accbase overflow<nsw> : index
            %address = arith.index_cast %shifted : index to i64
            %acc = pto.alloc_tile addr = %address : !acc
            pto.textract ins(%mat, %zero, %zero : !mat, index, index) outs(%write : !left)
            pto.tmatmul ins(%read, %right : !left, !right) outs(%acc : !acc)
          }
          return %three : index
        }
      }
    )mlir";
    auto module = parseSourceString<ModuleOp>(ir, &context);
    if (!module) { return false; }
    auto function = module->lookupSymbol<func::FuncOp>("coordinate_access");
    const auto original = printed(function);
    if (!arithmeticCoordinatePolicyIsolation(function) || printed(function) != original) { return false; }
    pto::SyncInput input;
    fs::PhaseIndex index;
    if (failed(input.build(function, pto::SyncInstructionView::PipeEnvelopes)) ||
        failed(index.build(function, input))) { return false; }
    auto arena = std::make_shared<fs::RegionExpressions>();
    std::vector<fs::RegionalAnalysis> children;
    std::string error;
    for (auto loop : function.getOps<scf::ForOp>()) {
        const fs::ArithmeticRegionContext scope{function, loop};
        auto program = fs::recognizeArithmeticProgram(scope, index, input, input.accesses(), {2, 4, 4, 4096});
        if (program.extraction.state != fs::RecognitionState::Applicable ||
            program.recognition.state != fs::RecognitionState::Applicable || program.sites.size() != 2) {
            llvm::errs() << "coordinate access recognition failed\n"; return false;
        }
        auto mathematics = fs::analyzeArithmeticRegionDemands(scope, index, input, &program, error);
        if (!mathematics || !mathematics->analysis.exactMinimum || !mathematics->analysis.error.empty()) {
            llvm::errs() << "coordinate access demands: " << error << '\n'; return false;
        }
        auto child = fs::exportArithmeticRegion(*mathematics, arena, true, error, {},
            fs::ArithmeticRegionalAllocationPolicy::CoordinateCandidate);
        if (failed(child)) { llvm::errs() << "coordinate access export: " << error << '\n'; return false; }
        auto prepared = child->prepare();
        if (failed(prepared)) { return false; }
        fs::prepareAllocationSupport(**prepared);
        if (!(*prepared)->regionalAllocation || (*prepared)->regionalAllocation->groups.size() != 2) { return false; }
        for (const auto& group : (*prepared)->regionalAllocation->groups) {
            if (group.budget != 3 || group.members.size() != 1 || !group.lanes.empty()) { return false; }
            const auto& rule = group.members.front().tupleRule;
            if (!rule || rule->coordinateCount != 2 || rule->terms.size() != 1 ||
                rule->terms.front().coordinate != 1 || rule->terms.front().modulus != 3) { return false; }
        }
        children.push_back(std::move(*child));
    }
    if (children.size() != 2) { return false; }
    auto sequence = fs::composeRegionalSequence(function, arena, std::move(children), false, true, {},
        fs::ArithmeticRegionalAllocationPolicy::CoordinateCandidate);
    if (!sequence.error.empty()) {
        llvm::errs() << "coordinate access parent: " << sequence.error << '\n'; return false;
    }
    auto prepared = fs::prepareSequenceInsertion(sequence);
    if (failed(prepared) || !(*prepared)->allocationCertificate || !(*prepared)->regionalAllocation ||
        (*prepared)->regionalAllocation->groups.size() != 4 || printed(function) != original ||
        failed(fs::insertLogicalSynchronization(function, **prepared)) ||
        failed(fs::allocatePhysicalEventIds(function, {0, 1, 2, 3, 4, 5})) || failed(verify(function))) {
        return false;
    }
    // The independent scalar oracle needs no tile values. Remove exactly the
    // fixture's side-effect-only payloads and their metadata from a private
    // clone after verification; keep every synchronization command and cut.
    OwningOpRef<ModuleOp> execution = cast<ModuleOp>(module->clone());
    auto executable = execution->lookupSymbol<func::FuncOp>("coordinate_access");
    SmallVector<Operation*> payloads;
    unsigned accesses = 0;
    executable.walk([&](Operation* op) {
        auto name = op->getName().getStringRef();
        if (name == "pto.textract" || name == "pto.tmatmul") { ++accesses; payloads.push_back(op); }
        else if (name == "pto.multi_tile_get" || name == "pto.alloc_multi_tile" || name == "pto.alloc_tile") {
            payloads.push_back(op);
        }
    });
    if (!hasOnlyTerminalInvocationDrain(executable) || accesses != 4) { return false; }
    for (auto* op : llvm::reverse(payloads)) {
        if (!op->use_empty()) { return false; }
        op->erase();
    }
    Values values; Trace trace; DirectedTrace directed;
    auto result = execute(executable.front(), values, trace, &directed);
    const auto p = static_cast<uint32_t>(pto::PipelineType::PIPE_MTE1);
    const auto q = static_cast<uint32_t>(pto::PipelineType::PIPE_M);
    return succeeded(result) && result->size() == 1 && result->front() == 3 &&
        integrationCommandOracle(directed, 16, p, q);
}
} // namespace
#endif
