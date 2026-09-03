// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- SyncSemantics.cpp - Shared ProtocolSync semantic records ---------===//

#include "PTO/Transforms/ProtocolSync/SyncSemantics.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

void SyncSemanticContext::addStorage(Value value, SyncStorageProvenance provenance)
{
    storage[value].push_back(std::move(provenance));
}

ArrayRef<SyncStorageProvenance> SyncSemanticContext::lookupStorage(Value value) const
{
    auto found = storage.find(value);
    return found == storage.end() ? ArrayRef<SyncStorageProvenance>() : ArrayRef<SyncStorageProvenance>(found->second);
}

StringRef mlir::pto::protocol_sync::stringifySyncPhysicalCore(SyncPhysicalCore core)
{
    switch (core) {
        case SyncPhysicalCore::Cube:
            return "cube";
        case SyncPhysicalCore::Vector:
            return "vector";
        case SyncPhysicalCore::Unknown:
            return "unknown";
    }
    return "unknown";
}

StringRef mlir::pto::protocol_sync::stringifySyncAccessMode(SyncAccessMode mode)
{
    switch (mode) {
        case SyncAccessMode::Read:
            return "read";
        case SyncAccessMode::Write:
            return "write";
        case SyncAccessMode::ReadWrite:
            return "read-write";
        case SyncAccessMode::Ordered:
            return "ordered";
    }
    return "unknown";
}

StringRef mlir::pto::protocol_sync::stringifySyncVisibility(SyncVisibilityClass visibility)
{
    switch (visibility) {
        case SyncVisibilityClass::Local:
            return "local";
        case SyncVisibilityClass::Global:
            return "global";
        case SyncVisibilityClass::Unknown:
            return "unknown";
    }
    return "unknown";
}

StringRef mlir::pto::protocol_sync::stringifySyncCompletionKind(SyncCompletionKind completion)
{
    switch (completion) {
        case SyncCompletionKind::PhaseEnd:
            return "phase-end";
        case SyncCompletionKind::MacroInternal:
            return "macro-internal";
        case SyncCompletionKind::Unknown:
            return "unknown";
    }
    return "unknown";
}

StringRef mlir::pto::protocol_sync::stringifySyncQueueRole(SyncQueueRole role)
{
    switch (role) {
        case SyncQueueRole::Initialize:
            return "initialize";
        case SyncQueueRole::ProducerAcquire:
            return "producer-acquire";
        case SyncQueueRole::ProducerPublish:
            return "producer-publish";
        case SyncQueueRole::ConsumerAcquire:
            return "consumer-acquire";
        case SyncQueueRole::ConsumerRelease:
            return "consumer-release";
    }
    return "unknown";
}

StringRef mlir::pto::protocol_sync::stringifySyncSummaryProvider(SyncSummaryProvider provider)
{
    switch (provider) {
        case SyncSummaryProvider::FixedSynchronization:
            return "fixed-sync";
        case SyncSummaryProvider::Queue:
            return "queue";
        case SyncSummaryProvider::Macro:
            return "macro";
        case SyncSummaryProvider::Helper:
            return "helper";
        case SyncSummaryProvider::Pipeline:
            return "pipeline";
        case SyncSummaryProvider::Structural:
            return "structural";
        case SyncSummaryProvider::None:
            return "none";
    }
    return "none";
}

StringRef mlir::pto::protocol_sync::stringifySyncFailureReason(SyncFailureReason reason)
{
    switch (reason) {
        case SyncFailureReason::None:
            return "none";
        case SyncFailureReason::MissingPipeline:
            return "missing-pipeline";
        case SyncFailureReason::MissingPhysicalCore:
            return "missing-physical-core";
        case SyncFailureReason::MissingStorageProvenance:
            return "missing-storage-provenance";
        case SyncFailureReason::UnsupportedEffectfulOperation:
            return "unsupported-effectful-operation";
        case SyncFailureReason::UnsupportedRegion:
            return "unsupported-region";
        case SyncFailureReason::LegacyStructureMismatch:
            return "legacy-structure-mismatch";
        case SyncFailureReason::LegacyPhaseMismatch:
            return "legacy-phase-mismatch";
        case SyncFailureReason::InternalInvariant:
            return "internal-invariant";
    }
    return "internal-invariant";
}
