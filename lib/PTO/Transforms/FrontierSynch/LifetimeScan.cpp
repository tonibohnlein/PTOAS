// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Linear lifetime scan; completion chains compress readers on the same pipe.
#include "PTO/Transforms/FrontierSynch/LifetimeScan.h"
#include "PTO/Transforms/FrontierSynch/HardwareProtection.h"
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
    StorageScanResult output;
    bool run(llvm::ArrayRef<ExplicitEffects> input, llvm::ArrayRef<StorageGenerator> prerequisites)
    {
        for (const auto& occurrence : input) {
            if (!positions.emplace(occurrence.payload, positions.size()).second) {
                output.error = "duplicate payload identity";
                return false;
            }
            std::unordered_map<uint32_t, CellAccess> modes;
            for (const auto& access : occurrence.accesses) {
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
            }
            for (const auto& [atom, access] : modes) {
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
    std::unordered_map<uint32_t, std::size_t> positions;
    std::unordered_map<uint32_t, CellState> cells;
    std::unordered_map<uint64_t, uint32_t> emitted;
    std::unordered_map<uint64_t, uint32_t> groupPipes;
    bool validGroup(uint64_t group, uint32_t pipe)
    {
        return groupPipes.emplace(group, pipe).first->second == pipe;
    }
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
        output.witnesses.push_back({found->second, atom, hazard});
        return true;
    }
    bool visit(const ExplicitEffects& occurrence, uint32_t atom, const CellAccess& access)
    {
        auto& state = cells[atom];
        // A protected same-pipe writer chain uses native C/C order before a
        // retained software edge and I/I order after it. An entirely protected
        // chain has protected endpoints by group membership. Thus the sparse
        // scan remains complete without adding fictitious C/I hardware edges.
        if (state.writer && hardwareProtectsConflict(state.writer->pipe, state.writer->protectionGroup,
                                                    occurrence.pipe, access.protectionGroup)) {
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
StorageScanResult scanStorageLifetimes(llvm::ArrayRef<ExplicitEffects> occurrences,
                                      llvm::ArrayRef<StorageGenerator> prerequisites)
{
    Scanner scanner;
    if (!scanner.run(occurrences, prerequisites)) {
        StorageScanResult failure;
        failure.error = scanner.output.error;
        return failure;
    }
    return std::move(scanner.output);
}
} // namespace mlir::pto::frontiersynch
