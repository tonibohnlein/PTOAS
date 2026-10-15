// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// A structural, exhaustive partition of original branch cuts in one invocation.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_ENDPOINTCUTCHOICES_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_ENDPOINTCUTCHOICES_H
#include "PTO/Transforms/FrontierSynch/Structured/NumericTemplateEndpoints.h"
#include <memory>
namespace mlir::pto::frontiersynch {
class RegionExpressions;
class EndpointCutChoices {
public:
    scf::ForOp loop() const { return invocation; }
    ArrayRef<TemplateEndpointCut> cuts() const { return alternatives; }
    Value consumerPredicate() const { return terminalPredicate; }
    bool pendingMaterialization() const { return bool(terminalArena); }
    bool validPendingSelection() const;
    // Attempt-local identity of retained selection evidence. A fresh choice
    // wrapper around the same arena predicate must not force allocation again.
    // The containing protocol retains this object and therefore the arena.
    void appendPendingSelectionKey(SmallVectorImpl<uint64_t>& key) const
    {
        key.push_back(reinterpret_cast<uintptr_t>(terminalArena.get()));
        key.push_back(terminalExpression);
    }
private:
    EndpointCutChoices(scf::ForOp loop, ArrayRef<TemplateEndpointCut> cuts, Value predicate = {})
        : invocation(loop), alternatives(cuts.begin(), cuts.end()), terminalPredicate(predicate) {}
    scf::ForOp invocation;
    std::vector<TemplateEndpointCut> alternatives;
    Value terminalPredicate;
    std::shared_ptr<const RegionExpressions> terminalArena;
    uint32_t terminalExpression = 0;
    friend std::shared_ptr<const EndpointCutChoices> qualifyTerminalCleanupCuts(
        ArrayRef<TemplateEndpointCut>, std::shared_ptr<const RegionExpressions>, uint32_t);
    friend std::shared_ptr<const EndpointCutChoices> qualifyTerminalCleanupCuts(
        ArrayRef<TemplateEndpointCut>, Value);
    friend std::shared_ptr<const EndpointCutChoices> qualifyEndpointCutChoices(
        scf::ForOp, ArrayRef<TemplateEndpointCut>);
};
// Exactly one supplied cut occurs per invocation, proved only from original
// scf.if arm structure. Independent sequential choices and missing arms fail.
// No payload/predicate replay or branch valuation enumeration. IR is borrowed
// unchanged until insertion. Work is polynomial in the explicit path metadata.
std::shared_ptr<const EndpointCutChoices> qualifyEndpointCutChoices(
    scf::ForOp loop, ArrayRef<TemplateEndpointCut> cuts);
// Finite source publication: consumer WAIT or terminal cleanup WAIT. A null
// loop identifies this kind. The producer certifies complementary activity;
// this qualifier verifies original cuts. A loop-contained consumer requires a
// predicate selecting at most one original coordinate in every enclosing loop.
// Selected coordinates must be invariant across all those loops; the physical
// allocator rechecks the same property on the actual command guard.
std::shared_ptr<const EndpointCutChoices> qualifyTerminalCleanupCuts(
    ArrayRef<TemplateEndpointCut> cuts, Value consumerPredicate = {});
// Retain the same singleton proof without generated SSA. The owning arena and
// original cuts survive composition; final lowering rechecks the actual guard.
std::shared_ptr<const EndpointCutChoices> qualifyTerminalCleanupCuts(
    ArrayRef<TemplateEndpointCut> cuts, std::shared_ptr<const RegionExpressions> arena,
    uint32_t consumerPredicate);
// The same singleton-cut proof before scalar materialization. The exact arena
// predicate is interpreted with original branch facts; necessary exclusion
// domains are insufficient. This proves multiplicity, not value availability.
// The arena and original invocation must remain unchanged until lowering.
bool qualifyTerminalCleanupSelection(ArrayRef<TemplateEndpointCut> cuts,
    const RegionExpressions& arena, uint32_t consumerPredicate);
bool validateTerminalCleanupCommands(func::FuncOp function, ArrayRef<Operation*> commands);
// Revalidate serialized alternatives against actual command ancestors. Only
// each command's immediate generated endpoint guard wrapper is omitted; every
// more distant branch still must partition the invocation exhaustively.
bool validateEndpointCommandChoices(scf::ForOp loop, ArrayRef<Operation*> commands);
} // namespace mlir::pto::frontiersynch
#endif
