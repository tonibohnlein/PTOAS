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
#include "PTO/Transforms/FrontierSynch/NumericTemplateEndpoints.h"
#include <memory>
namespace mlir::pto::frontiersynch {
class EndpointCutChoices {
public:
    scf::ForOp loop() const { return invocation; }
    ArrayRef<TemplateEndpointCut> cuts() const { return alternatives; }
private:
    EndpointCutChoices(scf::ForOp loop, ArrayRef<TemplateEndpointCut> cuts)
        : invocation(loop), alternatives(cuts.begin(), cuts.end()) {}
    scf::ForOp invocation;
    std::vector<TemplateEndpointCut> alternatives;
    friend std::shared_ptr<const EndpointCutChoices> qualifyTerminalCleanupCuts(
        ArrayRef<TemplateEndpointCut>);
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
// this qualifier verifies original cuts and excludes repeated occurrences.
std::shared_ptr<const EndpointCutChoices> qualifyTerminalCleanupCuts(ArrayRef<TemplateEndpointCut> cuts);
bool validateTerminalCleanupCommands(func::FuncOp function, ArrayRef<Operation*> commands);
// Revalidate serialized alternatives against actual command ancestors. Only
// each command's immediate generated endpoint guard wrapper is omitted; every
// more distant branch still must partition the invocation exhaustively.
bool validateEndpointCommandChoices(scf::ForOp loop, ArrayRef<Operation*> commands);
} // namespace mlir::pto::frontiersynch
#endif
