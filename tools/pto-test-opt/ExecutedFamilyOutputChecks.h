// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent direct/split execution with live structured result bindings.
#ifndef PTO_TEST_EXECUTED_FAMILY_OUTPUT_CHECKS_H
#define PTO_TEST_EXECUTED_FAMILY_OUTPUT_CHECKS_H
namespace {
bool structuredExecutedOutput(MLIRContext& context, bool combined)
{
    constexpr const char* input = R"mlir(
      module {
        func.func @structured(%m: index, %n: index) -> index {
          %zero = arith.constant 0 : index
          %one = arith.constant 1 : index
          %three = arith.constant 3 : index
          %sum = scf.for %outer = %zero to %m step %one iter_args(%carry = %zero) -> index {
            %inner = scf.for %i = %zero to %n step %one iter_args(%acc = %carry) -> index {
              %residue = arith.remui %i, %three : index
              %active = arith.cmpi ne, %residue, %one : index
              %selected:2 = scf.if %active -> (i1, index) {
                scf.yield %active, %i : i1, index
              } else {
                scf.yield %active, %zero : i1, index
              }
              %chosen = scf.if %selected#0 -> i1 {
                scf.yield %selected#0 : i1
              } else {
                scf.yield %selected#0 : i1
              }
              %source = arith.addi %i, %zero {test.source} : index
              %target = arith.addi %acc, %selected#1 {test.target} : index
              scf.yield %target : index
            }
            scf.yield %inner : index
          }
          return %sum : index
        }
      }
    )mlir";
    auto module = parseSourceString<ModuleOp>(input, &context);
    if (!module) { return false; }
    auto function = module->lookupSymbol<func::FuncOp>("structured");
    Operation* source = nullptr; Operation* target = nullptr;
    SmallVector<scf::IfOp> branches; SmallVector<scf::ForOp> loops;
    function.walk([&](Operation* op) {
        if (op->hasAttr("test.source")) { source = op; }
        if (op->hasAttr("test.target")) { target = op; }
        if (auto branch = dyn_cast<scf::IfOp>(op)) { branches.push_back(branch); }
        if (auto loop = dyn_cast<scf::ForOp>(op)) { loops.push_back(loop); }
    });
    if (!source || !target || branches.size() != 2 || loops.size() != 2) { return false; }
    auto arena = std::make_shared<fs::RegionExpressions>();
    fs::PreparedLogicalPlan plan(0);
    plan.completeInvocation = plan.nestedIdentities = plan.groupedFamilies = plan.independentPieces = true;
    std::vector<fs::ContextualEndpointFamily> endpoints;
    for (uint32_t record = 0; record < 3; ++record) {
        const bool local = record != 0;
        fs::EndpointFamily family;
        family.id = record; family.local = local;
        family.sourcePipe = static_cast<uint32_t>(local ? pto::PIPE::PIPE_V : pto::PIPE::PIPE_MTE2);
        family.targetPipe = static_cast<uint32_t>(local ? pto::PIPE::PIPE_V : pto::PIPE::PIPE_MTE1);
        family.sourceCut = family.targetCut = {target->getBlock(), target};
        family.members.push_back({record, 0, 1, {}, {}});
        plan.families.push_back(family);
        for (auto kind : {fs::EndpointKind::Set, fs::EndpointKind::Barrier, fs::EndpointKind::Wait}) {
            if (local != (kind == fs::EndpointKind::Barrier)) { continue; }
            fs::ContextualEndpointFamily endpoint;
            endpoint.piece.kind = kind; endpoint.piece.sourcePipe = family.sourcePipe;
            endpoint.piece.targetPipe = family.targetPipe; endpoint.piece.namespaceId = local ? 1 : 0;
            endpoint.piece.cut = family.targetCut; endpoint.piece.order = endpoints.size();
            endpoint.piece.members.push_back({record, {}});
            // A raw original SCF result deliberately avoids an arithmetic
            // wrapper: the retained guard handle itself must follow replacement.
            endpoint.activity = arena->input(branches.back().getResult(0));
            endpoint.ordinal = arena->constant(0);
            if (!local) {
                for (auto loop : loops) {
                    endpoint.matchingCoordinates.push_back(arena->input(loop.getInductionVar()));
                }
            }
            endpoints.push_back(std::move(endpoint));
        }
    }
    auto recipe = fs::buildContextualFamilyEndpointRecipe(arena, endpoints);
    if (failed(recipe)) { return false; }
    plan.familyRecipe = std::move(*recipe); plan.allocationCertificate = certificate(&context, 2);
    if (combined && failed(fs::planPhysicalFamilyOutput(function, plan, {1, 3}))) { return false; }
    if (failed(fs::insertLogicalSynchronization(function, plan))) { return false; }
    if (!combined && failed(fs::allocateExecutedFamilyCounters(function, certificate(&context, 2), {1, 3}))) {
        return false;
    }
    unsigned barriers = 0;
    function.walk([&](pto::BarrierOp) { ++barriers; });
    if (barriers != 2 || failed(verify(function))) { return false; }
    for (uint64_t outer : {0U, 1U, 2U}) {
        for (uint64_t bound : {0U, 1U, 2U, 5U, 8U}) {
            Values values; values[function.getArgument(0)] = outer; values[function.getArgument(1)] = bound;
            Trace actual, expected; uint64_t sum = 0, count = 0;
            for (uint64_t o = 0; o < outer; ++o) {
                for (uint64_t i = 0; i < bound; ++i) {
                    if (i % 3 == 1) { continue; }
                    const auto id = count++ % 2 ? 3 : 1;
                    expected.emplace_back(true, id); expected.emplace_back(false, id); sum += i;
                }
            }
            auto result = execute(function.front(), values, actual);
            if (failed(result) || result->size() != 1 || result->front() != sum || actual != expected) { return false; }
        }
    }
    return true;
}
} // namespace
#endif
