// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- ResidualDump.cpp - Deterministic selected-world diagnostics ------===//

#include "PTO/Transforms/ProtocolSync/ResidualObligation.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

void printPhaseId(SyncPhaseId phase, llvm::raw_ostream& output)
{
    if (phase == kInvalidSyncId) {
        output << "unknown";
    } else {
        output << '#' << phase;
    }
}

void printIteration(const SyncIterationRelation& relation, llvm::raw_ostream& output)
{
    output << stringifySyncIterationRelationKind(relation.kind) << " distance=" << relation.distance;
    if (relation.carrier != kInvalidSyncId) {
        output << " carrier=#" << relation.carrier;
    }
}

} // namespace

void mlir::pto::protocol_sync::printResidualObligations(
    func::FuncOp function, const SyncSelectedWorld& world, const SyncInterpretationResult& result,
    llvm::raw_ostream& output)
{
    output << "PROTOCOL-SYNC selected-world function=@" << function.getSymName()
           << " status=" << (result.isComplete() ? "complete" : "residual") << " protocols=" << world.protocols.size()
           << " completions=" << world.completions.size() << " visibility=" << world.visibility.size()
           << " exit-completions=" << world.exitCompletedPhases.size() << " obligations=" << result.obligations.size()
           << '\n';
    for (const SyncSelectedProtocol& protocol : world.protocols) {
        output << "  selected-protocol kind=" << stringifySyncSelectedProtocolKind(protocol.kind) << " channel=#"
               << protocol.channel << " generation=#" << protocol.generation << " capacity=" << protocol.capacity
               << '\n';
    }
    for (const SyncSelectedCompletion& completion : world.completions) {
        output << "  selected-completion source=#" << completion.source << " target=#" << completion.target
               << " control=" << stringifySyncControlRelation(completion.control) << " iteration=";
        printIteration(completion.iteration, output);
        output << '\n';
    }
    for (const SyncSelectedVisibility& visibility : world.visibility) {
        output << "  selected-visibility source=#" << visibility.source << " target=#" << visibility.target
               << " control=" << stringifySyncControlRelation(visibility.control) << " iteration=";
        printIteration(visibility.iteration, output);
        output << '\n';
    }
    for (const SyncResidualObligation& obligation : result.obligations) {
        output << "  obligation #" << obligation.id << " kind=" << stringifySyncObligationKind(obligation.kind)
               << " source=";
        printPhaseId(obligation.source, output);
        output << " target=";
        printPhaseId(obligation.target, output);
        output << " generation=";
        if (obligation.generation) {
            output << '#' << *obligation.generation;
        } else {
            output << "none";
        }
        output << " channel=";
        if (obligation.channel) {
            output << '#' << *obligation.channel;
        } else {
            output << "none";
        }
        output << " control=" << stringifySyncControlRelation(obligation.control) << " iteration=";
        printIteration(obligation.iteration, output);
        output << " detail=\"" << obligation.detail << "\"\n";
    }
    output << "PROTOCOL-SYNC selected-world-end function=@" << function.getSymName() << '\n';
}

StringRef mlir::pto::protocol_sync::stringifySyncObligationKind(SyncObligationKind kind)
{
    switch (kind) {
        case SyncObligationKind::Completion:
            return "completion";
        case SyncObligationKind::Reclamation:
            return "reclamation";
        case SyncObligationKind::SSACompletion:
            return "ssa-completion";
        case SyncObligationKind::OrderedMemory:
            return "ordered-memory";
        case SyncObligationKind::AccConflict:
            return "acc-conflict";
        case SyncObligationKind::Visibility:
            return "visibility";
        case SyncObligationKind::ExitCompletion:
            return "exit-completion";
        case SyncObligationKind::UnknownAlias:
            return "unknown-alias";
    }
    return "unknown-alias";
}

StringRef mlir::pto::protocol_sync::stringifySyncControlRelation(SyncControlRelation relation)
{
    switch (relation) {
        case SyncControlRelation::MustExecute:
            return "must-execute";
        case SyncControlRelation::SameGuard:
            return "same-guard";
        case SyncControlRelation::Unknown:
            return "unknown";
    }
    return "unknown";
}

StringRef mlir::pto::protocol_sync::stringifySyncIterationRelationKind(SyncIterationRelationKind kind)
{
    switch (kind) {
        case SyncIterationRelationKind::SameIteration:
            return "same-iteration";
        case SyncIterationRelationKind::LoopCarried:
            return "loop-carried";
        case SyncIterationRelationKind::Unknown:
            return "unknown";
    }
    return "unknown";
}

StringRef mlir::pto::protocol_sync::stringifySyncSelectedProtocolKind(SyncSelectedProtocolKind kind)
{
    switch (kind) {
        case SyncSelectedProtocolKind::OneShotPublish:
            return "one-shot-publish";
        case SyncSelectedProtocolKind::ReadyRelease:
            return "ready-release";
    }
    return "one-shot-publish";
}
