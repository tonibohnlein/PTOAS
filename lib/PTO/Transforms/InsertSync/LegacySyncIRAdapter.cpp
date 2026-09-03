// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- LegacySyncIRAdapter.cpp - ProtocolSync shadow comparison ---------===//

#include "PTO/Transforms/InsertSync/LegacySyncIRAdapter.h"

#include "PTO/Transforms/InsertSync/MemoryDependentAnalyzer.h"
#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"
#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include "PTO/Transforms/SlotAffineAnalysis.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;
using namespace llvm;

namespace {

enum class ShadowTokenKind {
    Phase,
    LoopBegin,
    LoopEnd,
    IfBegin,
    ElseBegin,
    IfEnd,
    Placeholder,
};

struct ShadowToken {
    ShadowTokenKind kind;
    Operation* operation = nullptr;
};

std::optional<PIPE> convertLegacyPipe(PipelineType pipe)
{
    switch (pipe) {
        case PipelineType::PIPE_S:
            return PIPE::PIPE_S;
        case PipelineType::PIPE_V:
            return PIPE::PIPE_V;
        case PipelineType::PIPE_M:
            return PIPE::PIPE_M;
        case PipelineType::PIPE_MTE1:
            return PIPE::PIPE_MTE1;
        case PipelineType::PIPE_MTE2:
            return PIPE::PIPE_MTE2;
        case PipelineType::PIPE_MTE3:
            return PIPE::PIPE_MTE3;
        case PipelineType::PIPE_FIX:
            return PIPE::PIPE_FIX;
        default:
            return std::nullopt;
    }
}

SyncPhysicalCore convertLegacyCore(TCoreType core)
{
    if (core == TCoreType::CUBE) {
        return SyncPhysicalCore::Cube;
    }
    if (core == TCoreType::VECTOR) {
        return SyncPhysicalCore::Vector;
    }
    return SyncPhysicalCore::Unknown;
}

SyncPhysicalCore getLegacyCoreForPipe(PIPE pipe)
{
    return pipe == PIPE::PIPE_M || pipe == PIPE::PIPE_MTE1 ? SyncPhysicalCore::Cube : SyncPhysicalCore::Vector;
}

void appendLegacyTokens(const LegacySyncSnapshot& legacy, SmallVectorImpl<ShadowToken>& tokens)
{
    for (const std::unique_ptr<InstanceElement>& element : legacy.syncIR) {
        switch (element->GetKind()) {
            case InstanceElement::KindTy::COMPOUND:
                tokens.push_back({ShadowTokenKind::Phase, element->elementOp});
                break;
            case InstanceElement::KindTy::LOOP: {
                auto* loop = dyn_cast<LoopInstanceElement>(element.get());
                tokens.push_back(
                    {loop->getLoopKind() == KindOfLoop::LOOP_BEGIN ? ShadowTokenKind::LoopBegin :
                                                                     ShadowTokenKind::LoopEnd,
                     element->elementOp});
                break;
            }
            case InstanceElement::KindTy::BRANCH: {
                auto* branch = dyn_cast<BranchInstanceElement>(element.get());
                ShadowTokenKind kind = ShadowTokenKind::IfEnd;
                const KindOfBranch branchKind = branch->getBranchKind();
                if (branchKind == KindOfBranch::IF_BEGIN) {
                    kind = ShadowTokenKind::IfBegin;
                } else if (branchKind == KindOfBranch::ELSE_BEGIN) {
                    kind = ShadowTokenKind::ElseBegin;
                }
                tokens.push_back({kind, element->elementOp});
                break;
            }
            case InstanceElement::KindTy::PLACE_HOLDER:
                tokens.push_back({ShadowTokenKind::Placeholder, element->elementOp});
                break;
        }
    }
}

void appendScheduleRegionTokens(
    const StructuredSyncIR& schedule, SyncRegionId id, SmallVectorImpl<ShadowToken>& tokens);

void appendChoiceTokens(
    const StructuredSyncIR& schedule, const SyncRegion& choice, SmallVectorImpl<ShadowToken>& tokens)
{
    tokens.push_back({ShadowTokenKind::IfBegin, choice.operation});
    for (const SyncRegionElement& element : choice.elements) {
        const SyncRegion* alternative = schedule.findRegion(element.child);
        if (!alternative) {
            return;
        }
        if (alternative->arm == 1) {
            tokens.push_back({ShadowTokenKind::ElseBegin, choice.operation});
        }
        appendScheduleRegionTokens(schedule, alternative->id, tokens);
        tokens.push_back({ShadowTokenKind::Placeholder, choice.operation});
    }
    tokens.push_back({ShadowTokenKind::IfEnd, choice.operation});
}

void appendScheduleRegionTokens(const StructuredSyncIR& schedule, SyncRegionId id, SmallVectorImpl<ShadowToken>& tokens)
{
    const SyncRegion* region = schedule.findRegion(id);
    if (!region) {
        return;
    }
    if (region->kind == SyncRegionKind::Loop) {
        tokens.push_back({ShadowTokenKind::LoopBegin, region->operation});
    } else if (region->kind == SyncRegionKind::Choice) {
        appendChoiceTokens(schedule, *region, tokens);
        return;
    }
    for (const SyncRegionElement& element : region->elements) {
        if (element.kind == SyncRegionElement::Kind::Phase) {
            const SyncPhase* phase = schedule.findPhase(element.phase);
            if (!phase) {
                return;
            }
            tokens.push_back({ShadowTokenKind::Phase, phase->operation});
        } else {
            appendScheduleRegionTokens(schedule, element.child, tokens);
        }
    }
    if (region->kind == SyncRegionKind::Loop) {
        tokens.push_back({ShadowTokenKind::LoopEnd, region->operation});
    }
}

bool sameIntervals(const BaseMemInfo& legacy, const SyncStorageProvenance& storage)
{
    const bool intervalCountsMatch = legacy.baseAddresses.size() == storage.intervals.size();
    if (!intervalCountsMatch) {
        return false;
    }
    for (auto [address, interval] : llvm::zip(legacy.baseAddresses, storage.intervals)) {
        if (address != interval.begin || legacy.allocateSize != interval.size) {
            return false;
        }
    }
    return true;
}

bool sameStorage(const BaseMemInfo& legacy, const SyncStorageProvenance& storage)
{
    const bool legacyUnknownRange =
        legacy.aliasesUnknownRange || legacy.baseAddresses.empty() || legacy.allocateSize == 0;
    return legacy.baseBuffer == storage.value && legacy.rootBuffer == storage.root && legacy.scope == storage.space &&
           legacy.hasKnownPhysicalAddresses == storage.physical &&
           legacy.aliasesUnknownRange == storage.aliasesUnknownRange && legacyUnknownRange == storage.unknownRange &&
           sameIntervals(legacy, storage);
}

bool sameSlot(const BaseMemInfo& legacy, const SyncAccess& access)
{
    Value selector = findMultiTileSlotExpr(legacy.baseBuffer);
    if (!selector) {
        return !access.slot;
    }
    std::optional<std::uint32_t> depth = getSyncSlotDepth(legacy.baseBuffer);
    const bool sameSelector = access.slot && access.slot->selector == selector;
    const bool sameDepth = access.slot && access.slot->depth == depth.value_or(0);
    return sameSelector && sameDepth;
}

void addMismatch(LegacySyncParityResult& result, StringRef category, std::uint32_t index, StringRef detail)
{
    result.mismatches.push_back({category.str(), index, detail.str()});
}

void compareStructure(
    const LegacySyncSnapshot& legacy, const StructuredSyncIR& schedule, LegacySyncParityResult& result)
{
    SmallVector<ShadowToken, 32> legacyTokens;
    SmallVector<ShadowToken, 32> scheduleTokens;
    appendLegacyTokens(legacy, legacyTokens);
    if (!schedule.getRegions().empty()) {
        appendScheduleRegionTokens(schedule, 0, scheduleTokens);
    }
    const bool tokenCountsMatch = legacyTokens.size() == scheduleTokens.size();
    if (!tokenCountsMatch) {
        addMismatch(result, "structure", 0, "token counts differ");
        return;
    }
    for (auto [index, pair] : llvm::enumerate(llvm::zip(legacyTokens, scheduleTokens))) {
        const auto& [legacyToken, scheduleToken] = pair;
        if (legacyToken.kind != scheduleToken.kind ||
            (legacyToken.kind == ShadowTokenKind::Phase && legacyToken.operation != scheduleToken.operation)) {
            addMismatch(result, "structure", index, "token sequence differs");
        }
    }
}

void compareAccesses(
    const CompoundInstanceElement& legacy, const StructuredSyncIR& schedule, const SyncPhase& phase,
    LegacySyncParityResult& result)
{
    SmallVector<const SyncAccess*, 4> reads;
    SmallVector<const SyncAccess*, 4> writes;
    for (SyncAccessId id : phase.accesses) {
        const SyncAccess* access = schedule.findAccess(id);
        if (!access) {
            addMismatch(result, "memory", phase.id, "invalid access id");
            return;
        }
        if (access->mode == SyncAccessMode::Read) {
            reads.push_back(access);
        } else if (access->mode == SyncAccessMode::Write) {
            writes.push_back(access);
        }
    }
    const bool readCountsMatch = reads.size() == legacy.useVec.size();
    const bool writeCountsMatch = writes.size() == legacy.defVec.size();
    if (!readCountsMatch || !writeCountsMatch) {
        addMismatch(result, "memory", phase.id, "definition/use counts differ");
        return;
    }
    for (auto [expected, actual] : llvm::zip(legacy.useVec, reads)) {
        const bool storageMatches = sameStorage(*expected, actual->storage);
        const bool slotMatches = sameSlot(*expected, *actual);
        if (!storageMatches || !slotMatches) {
            addMismatch(result, "memory", phase.id, "read provenance or slot differs");
        }
    }
    for (auto [expected, actual] : llvm::zip(legacy.defVec, writes)) {
        const bool storageMatches = sameStorage(*expected, actual->storage);
        const bool slotMatches = sameSlot(*expected, *actual);
        if (!storageMatches || !slotMatches) {
            addMismatch(result, "memory", phase.id, "write provenance or slot differs");
        }
    }
}

void comparePhases(const LegacySyncSnapshot& legacy, const StructuredSyncIR& schedule, LegacySyncParityResult& result)
{
    SmallVector<const CompoundInstanceElement*, 32> compounds;
    for (const std::unique_ptr<InstanceElement>& element : legacy.syncIR) {
        if (auto* compound = dyn_cast<CompoundInstanceElement>(element.get())) {
            compounds.push_back(compound);
        }
    }
    const bool phaseCountsMatch = compounds.size() == schedule.getPhases().size();
    if (!phaseCountsMatch) {
        addMismatch(result, "phase", 0, "phase counts differ");
        return;
    }
    for (auto [index, pair] : llvm::enumerate(llvm::zip(compounds, schedule.getPhases()))) {
        const auto& [legacyPhase, phase] = pair;
        std::optional<PIPE> pipe = convertLegacyPipe(legacyPhase->kPipeValue);
        std::optional<unsigned> macroPhase =
            legacyPhase->macroOpInstanceId < 0 ?
                std::nullopt :
                std::optional<unsigned>(static_cast<unsigned>(legacyPhase->macroOpInstanceId));
        const SyncCompletionKind completion =
            macroPhase ? SyncCompletionKind::MacroInternal : SyncCompletionKind::PhaseEnd;
        const SyncPhysicalCore legacyCore = convertLegacyCore(legacyPhase->compoundCoreType);
        const SyncPhysicalCore expectedLegacyCore = pipe ? getLegacyCoreForPipe(*pipe) : SyncPhysicalCore::Unknown;
        if (!pipe || *pipe != phase.pipe || legacyPhase->elementOp != phase.operation ||
            macroPhase != phase.macroPhase || legacyCore != expectedLegacyCore || completion != phase.completion) {
            addMismatch(result, "phase", index, "operation, core, pipeline, macro phase, or completion differs");
            continue;
        }
        compareAccesses(*legacyPhase, schedule, phase, result);
    }
}

void compareSemanticMetadata(const StructuredSyncIR& schedule, LegacySyncParityResult& result)
{
    for (auto [index, summary] : llvm::enumerate(schedule.getSummaries())) {
        std::optional<SyncQueueSemantics> queue = getSyncQueueSemantics(summary.operation);
        const bool queuePresenceMatches = queue.has_value() == summary.queue.has_value();
        if (!queuePresenceMatches) {
            addMismatch(result, "queue", index, "queue classification differs");
        } else if (queue && summary.queue) {
            const bool sameRole = queue->role == summary.queue->role;
            const bool sameHandle = queue->handle == summary.queue->handle;
            const bool sameDepth = queue->depth == summary.queue->depth;
            if (!sameRole || !sameHandle || !sameDepth) {
                addMismatch(result, "queue", index, "queue role, handle, or depth differs");
            }
        }

        std::optional<SyncMacroModel> model = getSyncMacroModel(summary.operation);
        if (!model) {
            if (!summary.eventReservations.empty()) {
                addMismatch(result, "reservation", index, "non-macro operation reserves hidden events");
            }
            continue;
        }
        const bool reservationCountsMatch = model->hiddenEvents.size() == summary.eventReservations.size();
        if (!reservationCountsMatch) {
            addMismatch(result, "reservation", index, "hidden event reservation counts differ");
            continue;
        }
        for (auto [legacyReservation, reservation] : llvm::zip(model->hiddenEvents, summary.eventReservations)) {
            std::optional<PIPE> source = convertLegacyPipe(legacyReservation.srcPipe);
            std::optional<PIPE> target = convertLegacyPipe(legacyReservation.dstPipe);
            const bool sameDirection =
                source && target && *source == reservation.source && *target == reservation.target;
            const bool sameEventIds = legacyReservation.eventIds == reservation.eventIds;
            if (!sameDirection || !sameEventIds) {
                addMismatch(result, "reservation", index, "hidden event reservation differs");
            }
        }
    }
}

} // namespace

LogicalResult LegacySyncIRAdapter::buildSnapshot(func::FuncOp function, LegacySyncSnapshot& snapshot) const
{
    const bool snapshotEmpty = snapshot.syncIR.empty() && snapshot.storage.empty();
    if (!snapshotEmpty) {
        return failure();
    }
    MemoryDependentAnalyzer analyzer;
    PTOIRTranslator translator(snapshot.syncIR, analyzer, snapshot.storage, function, SyncAnalysisMode::NORMALSYNC);
    translator.Build();
    return success();
}

SyncSemanticContext LegacySyncIRAdapter::buildSemanticContext(const LegacySyncSnapshot& snapshot) const
{
    SyncSemanticContext context;
    for (const auto& [value, infos] : snapshot.storage) {
        for (const std::unique_ptr<BaseMemInfo>& info : infos) {
            if (!info) {
                continue;
            }
            SyncStorageProvenance provenance;
            provenance.value = info->baseBuffer;
            provenance.root = info->rootBuffer;
            provenance.space = info->scope;
            provenance.physical = info->hasKnownPhysicalAddresses;
            provenance.aliasesUnknownRange = info->aliasesUnknownRange;
            provenance.unknownRange =
                info->aliasesUnknownRange || info->baseAddresses.empty() || info->allocateSize == 0;
            for (std::uint64_t address : info->baseAddresses) {
                provenance.intervals.push_back({address, info->allocateSize});
            }
            context.addStorage(value, std::move(provenance));
        }
    }
    return context;
}

LegacySyncParityResult LegacySyncIRAdapter::compare(
    const LegacySyncSnapshot& legacy, const StructuredSyncIR& schedule) const
{
    LegacySyncParityResult result;
    compareStructure(legacy, schedule, result);
    comparePhases(legacy, schedule, result);
    compareSemanticMetadata(schedule, result);
    return result;
}

void protocol_sync::printLegacySyncParity(
    func::FuncOp function, const LegacySyncParityResult& result, raw_ostream& output)
{
    output << "PROTOCOL-SYNC legacy-parity function=@" << function.getSymName()
           << " status=" << (result.matches() ? "match" : "mismatch") << " mismatches=" << result.mismatches.size()
           << '\n';
    for (const LegacySyncParityMismatch& mismatch : result.mismatches) {
        output << "  parity-mismatch category=" << mismatch.category << " index=" << mismatch.index << " detail=\""
               << mismatch.detail << "\"\n";
    }
}
