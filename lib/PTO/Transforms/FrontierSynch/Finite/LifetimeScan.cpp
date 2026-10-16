// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Linear lifetime scan; completion chains compress readers on the same pipe.
#include "PTO/Transforms/FrontierSynch/Finite/LifetimeScanInput.h"
#include "PTO/Transforms/FrontierSynch/Shared/HardwareProtection.h"
#include <limits>
#include <optional>
#include <unordered_map>
namespace mlir::pto::frontiersynch {
namespace {
struct Writer {
    uint32_t payload;
    uint32_t pipe;
    uint64_t protectionGroup;
};
struct CellState {
    std::optional<Writer> writer;
    std::unordered_map<uint32_t, uint32_t> readers;
};
class Scanner {
public:
    Scanner(StorageProtectionPolicy policy, bool witnesses, StorageScanWork* counters)
        : protection(policy), retainWitnesses(witnesses), work(counters)
    {}
    StorageScanResult output;
    bool run(const StorageScanInput& input, llvm::ArrayRef<StorageGenerator> prerequisites)
    {
        for (std::size_t id = 0; id < input.size; ++id) {
            const auto occurrence = input.occurrence(id);
            if (!positions.emplace(occurrence.payload, positions.size()).second) {
                output.error = "duplicate payload identity";
                return false;
            }
            std::unordered_map<uint32_t, CellAccess> modes;
            const bool complete = input.accesses(id, [&](const CellAccess& access) {
                if (work) {
                    count(work->incidenceVisits);
                }
                if (!access.read && !access.write) {
                    output.error = "empty access mode";
                    return false;
                }
                auto inserted = modes.emplace(access.atom, access);
                auto& mode = inserted.first->second;
                if (mode.protectionGroup != access.protectionGroup) {
                    output.error = "inconsistent access protection";
                    return false;
                }
                mode.read |= access.read;
                mode.write |= access.write;
                return true;
            });
            if (!complete) {
                if (output.error.empty()) {
                    output.error = "invalid access input";
                }
                return false;
            }
            for (const auto& [atom, access] : modes) {
                if (work) {
                    count(work->consolidatedVisits);
                }
                if (access.protectionGroup != 0 &&
                    (!access.write || !validGroup(access.protectionGroup, occurrence.pipe))) {
                    output.error = "protection requires writers on one pipe";
                    return false;
                }
                if (!visit(occurrence, atom, access)) {
                    return false;
                }
            }
        }
        for (const auto& edge : prerequisites) {
            auto from = positions.find(edge.source), to = positions.find(edge.target);
            if (from == positions.end() || to == positions.end() || from->second >= to->second) {
                output.error = "nonforward or absent prerequisite endpoint";
                return false;
            }
            if (!emit(edge.source, edge.target, 0, StorageHazard::Supplied)) {
                return false;
            }
        }
        return true;
    }

private:
    StorageProtectionPolicy protection;
    bool retainWitnesses;
    StorageScanWork* work;
    void count(uint64_t& counter)
    {
        if (counter == UINT64_MAX) {
            work->counterOverflow = true;
        } else {
            ++counter;
        }
    }
    std::unordered_map<uint32_t, std::size_t> positions;
    std::unordered_map<uint32_t, CellState> cells;
    std::unordered_map<uint64_t, uint32_t> emitted;
    std::unordered_map<uint64_t, uint32_t> groupPipes;
    bool validGroup(uint64_t group, uint32_t pipe) { return groupPipes.emplace(group, pipe).first->second == pipe; }
    bool emit(uint32_t source, uint32_t target, uint32_t atom, StorageHazard hazard)
    {
        const uint64_t key = (uint64_t(source) << 32) | target;
        auto found = emitted.find(key);
        if (found == emitted.end()) {
            if (output.generators.size() >= std::numeric_limits<uint32_t>::max()) {
                output.error = "generator identity overflow";
                return false;
            }
            const auto id = static_cast<uint32_t>(output.generators.size());
            found = emitted.emplace(key, id).first;
            output.generators.push_back({source, target});
        }
        if (work) {
            count(work->hazardWitnesses);
        }
        if (retainWitnesses) {
            output.witnesses.push_back({found->second, atom, hazard});
            if (work) {
                count(work->retainedWitnesses);
            }
        }
        return true;
    }
    bool visit(const StorageOccurrenceMetadata& occurrence, uint32_t atom, const CellAccess& access)
    {
        auto& state = cells[atom];
        // A protected same-pipe writer chain uses native C/C order before a
        // retained software edge and I/I order after it. An entirely protected
        // chain has protected endpoints by group membership. Thus the sparse
        // scan remains complete without adding fictitious C/I hardware edges.
        if (state.writer &&
            (protection.protectsScalar(state.writer->pipe, occurrence.pipe) ||
             hardwareProtectsConflict(
                 state.writer->pipe, state.writer->protectionGroup, occurrence.pipe, access.protectionGroup))) {
            output.protectedHazards += unsigned(access.read) + unsigned(access.write);
        } else if (state.writer) {
            if (access.read && !emit(state.writer->payload, occurrence.payload, atom, StorageHazard::RAW)) {
                return false;
            }
            if (access.write && !emit(state.writer->payload, occurrence.payload, atom, StorageHazard::WAW)) {
                return false;
            }
        }
        if (access.write) {
            for (const auto& reader : state.readers) {
                if (protection.protectsScalar(reader.first, occurrence.pipe)) {
                    ++output.protectedHazards;
                    continue;
                }
                if (!emit(reader.second, occurrence.payload, atom, StorageHazard::WAR)) {
                    return false;
                }
            }
            // Discard bucket capacity too: a past wide lifetime must not make
            // every subsequent overwrite clear its historical pipe count.
            std::unordered_map<uint32_t, uint32_t>().swap(state.readers);
            state.writer = Writer{occurrence.payload, occurrence.pipe, access.protectionGroup};
        } else {
            state.readers[occurrence.pipe] = occurrence.payload;
        }
        return true;
    }
};
} // namespace
StorageScanResult scanStorageLifetimes(
    const StorageScanInput& input, llvm::ArrayRef<StorageGenerator> prerequisites, StorageProtectionPolicy protection,
    bool retainWitnesses, StorageScanWork* work)
{
    if (work) {
        *work = {};
    }
    Scanner scanner(protection, retainWitnesses, work);
    if (!scanner.run(input, prerequisites)) {
        StorageScanResult failure;
        failure.error = scanner.output.error;
        return failure;
    }
    return std::move(scanner.output);
}
StorageScanResult scanStorageLifetimes(
    llvm::ArrayRef<ExplicitEffects> occurrences, llvm::ArrayRef<StorageGenerator> prerequisites,
    StorageProtectionPolicy protection, StorageScanWork* work)
{
    auto metadata = [&](std::size_t id) {
        return StorageOccurrenceMetadata{occurrences[id].payload, occurrences[id].pipe};
    };
    auto accesses = [&](std::size_t id, StorageAccessConsumer consume) {
        for (const auto& access : occurrences[id].accesses) {
            if (!consume(access)) {
                return false;
            }
        }
        return true;
    };
    return scanStorageLifetimes(StorageScanInput{occurrences.size(), metadata, accesses},
                                prerequisites, protection, true, work);
}
} // namespace mlir::pto::frontiersynch
