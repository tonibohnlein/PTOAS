// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SelectedInternal.h"

namespace mlir::pto::oahs::selected {
bool Constructor::fixedBoundaryPacket(const SourceGapQualification& facts, Group& group)
{
    // One designated source-local acknowledgment, followed by the ordinary
    // matched transfer. No candidate-ledger solve or provisional reservation.
    if (facts.gap.left != NoAnalysisId) { return false; }
    const auto& keys = frontier.keys();
    for (Id forward = 0; forward < keys.size(); ++forward) {
        if (!sourceKeyNeighbors(facts, forward)) { continue; }
        // A unique old WAIT has its own earlier prescribed helper source.
        // Do not move that publication here merely to fit a source-local proof.
        bool joined = true;
        for (const auto& pair : control.correspondence(facts.gap.cut, current).pairs) {
            if (!control.reachable[pair.first]) { continue; }
            joined &= cache.cuts[pair.first].incoming.consumptions[forward].size() > 1;
        }
        if (!joined) { continue; }
        const auto& a = keys[forward];
        bool empty = true, missingConsumption = false;
        for (const auto& prefix : facts.prefixes) {
            empty &= prefix.facts()->events[forward].occupancy == 1;
            State before; before.causal = prefix;
            missingConsumption |= !canPublish(before, forward);
        }
        if (!empty || !missingConsumption) { continue; }
        for (Id reverse = 0; reverse < keys.size(); ++reverse) {
            const auto& b = keys[reverse];
            const bool eligible = b.source == a.observer && b.observer == a.source &&
                helperFreeKey(reverse) && ledger.eventUses(b).empty();
            if (!eligible) { continue; }
            const Command publish{Command::Publish, b.source, b.observer, b.key};
            const Command acquire{Command::Acquire, b.source, b.observer, b.key};
            const Command send{Command::Publish, a.source, a.observer, a.key};
            bool executable = true;
            for (const auto& prefix : facts.prefixes) {
                auto state = prefix;
                Id offset = 0;
                for (const auto& command : {publish, acquire, send}) {
                    ++result.work.repairSourceCommands;
                    const auto step = frontier.command(state, command, {facts.gap.cut, offset++});
                    if (!step.applied) { executable = false; break; }
                    state = step.state;
                }
                if (!executable) { break; }
            }
            if (!executable) { continue; }
            const auto request = result.decisions.size();
            const OrderedPacket endpoints{
                {facts.gap.cut, publish, EndpointPurpose::ConsumptionAcknowledgment,
                    request, NoAnalysisId, facts.gap},
                {facts.gap.cut, acquire, EndpointPurpose::ConsumptionAcknowledgment,
                    request, NoAnalysisId, facts.gap},
                {facts.gap.cut, send, EndpointPurpose::Completion, request, NoAnalysisId, facts.gap},
                {current, {Command::Acquire, a.source, a.observer, a.key}, EndpointPurpose::Completion, request}};
            auto packet = prepareOwnedPacket(endpoints);
            const bool supported = packet && !packet->restoredEndpoints && preservePublications(*packet);
            if (!supported) { continue; }
            packet->qualified = true;
            group.packet = std::move(*packet);
            group.forwardKey = forward;
            group.repairKey = reverse;
            return true;
        }
    }
    return false;
}
} // namespace mlir::pto::oahs::selected
