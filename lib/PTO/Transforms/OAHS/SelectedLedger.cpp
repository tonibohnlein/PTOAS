// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SelectedInternal.h"
#include <algorithm>
#include <iterator>

namespace mlir::pto::oahs::selected {

Id accessClass(const FrontierRequirement& r)
{
    return (Id(r.cell) * PipeCount + unsigned(r.source)) * 2 + Id(r.sourceWrite);
}
bool identical(const Command& a, const Command& b)
{
    return a.kind == b.kind && a.source == b.source && a.observer == b.observer && a.key == b.key;
}
std::vector<Id> unionIds(const std::vector<Id>& a, const std::vector<Id>& b)
{
    std::vector<Id> out;
    std::set_union(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(out));
    return out;
}
Ledger::Ledger(const Program& p, const std::vector<Cut>& canonicalWords)
    : program(p), canonicalCut(canonicalWords), words(commandCutCount(p))
{
}
// Same value as canonicalCommandCut, read from the control's one-pass memo.
// Cuts beyond the table, and programs without a refined control, are their own
// canonical word, exactly as the program-level query defines.
Cut Ledger::canonical(Cut cut) const
{
    return cut < canonicalCut.size() ? canonicalCut[cut] : cut;
}
bool Ledger::initialize(const Commands& fixed, std::string& reason)
{
    if (fixed.empty()) {
        return true;
    }
    if (fixed.size() != words.size()) {
        reason = "fixed command cuts do not match the original control";
        return false;
    }
    for (Cut cut = 0; cut < fixed.size(); ++cut) {
        if (!legalCommandCut(program, cut) && !fixed[cut].empty()) {
            reason = "fixed endpoint has no legal original cut";
            return false;
        }
        const auto leader = canonical(cut);
        if (fixed[cut].size() != fixed[leader].size() ||
            !std::equal(fixed[cut].begin(), fixed[cut].end(), fixed[leader].begin(), identical)) {
            reason = "fixed words disagree at one original observation";
            return false;
        }
        if (leader == cut) {
            for (const auto& command : fixed[cut]) {
                append(cut, command, EndpointPurpose::Fixed);
            }
        }
    }
    changed.clear();
    return true;
}
Id Ledger::insert(Cut cut, Id offset, Command command, EndpointPurpose purpose, Id request, Id ack)
{
    cut = canonical(cut);
    const auto id = endpoints.size();
    endpoints.push_back({id, cut, command, purpose, request, ack});
    recordEvent(endpoints.back());
    words[cut].insert(words[cut].begin() + offset, id);
    ++revision;
    changed.push_back(cut);
    return id;
}
Id Ledger::append(Cut cut, Command command, EndpointPurpose purpose, Id request, Id ack)
{
    const auto leader = canonical(cut);
    return insert(leader, words[leader].size(), command, purpose, request, ack);
}
WordGap Ledger::tail(Cut cut) const
{
    const auto& ids = word(cut);
    return {canonical(cut), ids.empty() ? NoAnalysisId : ids.back(), NoAnalysisId};
}
std::optional<WordGap> Ledger::gapAfter(Id predecessor) const
{
    if (predecessor >= endpoints.size() || !active(predecessor)) {
        return {};
    }
    const auto cut = endpoints[predecessor].cut;
    const auto& ids = word(cut);
    const auto found = std::find(ids.begin(), ids.end(), predecessor);
    if (found == ids.end()) {
        return {};
    }
    const auto next = std::next(found);
    return WordGap{cut, predecessor, next == ids.end() ? NoAnalysisId : *next};
}
std::map<Id, WordGap> Ledger::gapsAfter(const std::vector<Id>& anchors) const
{
    std::map<Cut, std::set<Id>> byWord;
    for (auto id : anchors) {
        const bool valid = id < endpoints.size() && active(id);
        if (valid) { byWord[endpoints[id].cut].insert(id); }
    }
    std::map<Id, WordGap> result;
    for (const auto& [cut, requested] : byWord) {
        const auto& ids = word(cut);
        for (Id offset = 0; offset < ids.size(); ++offset) {
            if (requested.count(ids[offset])) {
                result.emplace(ids[offset], WordGap{cut, ids[offset],
                    offset + 1 == ids.size() ? NoAnalysisId : ids[offset + 1]});
            }
        }
    }
    return result;
}
PreparedPacket Ledger::preparePacket(const OrderedPacket& packet) const
{
    PreparedPacket out;
    out.owner = this;
    out.version = revision;
    out.firstEndpoint = endpoints.size();
    if (packet.size() > endpoints.max_size() - endpoints.size()) {
        out.error = "packet endpoint population exceeds representable storage";
        return out;
    }
    std::map<Cut, std::map<Id, Id>> positions;
    std::set<Id> restoring;
    for (const auto& item : packet) {
        if (!legalCommandCut(program, item.cut)) {
            out.error = "packet has no legal original command word";
            return out;
        }
        const auto cut = canonical(item.cut);
        const auto gap = item.gap.value_or(tail(cut));
        auto offset = word(cut).size();
        if (gap.right != NoAnalysisId) {
            auto at = positions.find(cut);
            if (at == positions.end()) {
                auto& index = positions[cut];
                const auto& ids = word(cut);
                for (Id indexOffset = 0; indexOffset < ids.size(); ++indexOffset) {
                    index.emplace(ids[indexOffset], indexOffset);
                }
                at = positions.find(cut);
            }
            const auto right = at->second.find(gap.right);
            offset = right == at->second.end() ? NoAnalysisId : right->second;
        }
        if (canonical(gap.cut) != cut || offset == NoAnalysisId ||
            gap.left != (offset == 0 ? NoAnalysisId : word(cut)[offset - 1])) {
            out.error = "packet gap neighbors are no longer adjacent in their original word";
            return out;
        }
        auto acknowledgment = item.acknowledges;
        if (item.acknowledgesPacket != NoAnalysisId) {
            if (acknowledgment != NoAnalysisId || item.acknowledgesPacket >= out.ordered.size()) {
                out.error = "packet acknowledgment does not name an earlier endpoint";
                return out;
            }
            acknowledgment = out.ordered[item.acknowledgesPacket];
        } else if (acknowledgment != NoAnalysisId &&
                   (acknowledgment >= endpoints.size() || !active(acknowledgment))) {
            out.error = "packet acknowledgment names an inactive endpoint";
            return out;
        }
        auto id = item.restore;
        if (id != NoAnalysisId) {
            const bool invalidRestore = id >= endpoints.size() || active(id) || !restoring.insert(id).second;
            if (invalidRestore) {
                out.error = "packet restoration does not name a distinct inactive endpoint";
                return out;
            }
            const auto& original = endpoints[id];
            const bool relocation = item.relocate && original.command.kind == Command::Acquire &&
                original.purpose == EndpointPurpose::ConsumptionAcknowledgment;
            const bool changedIdentity = (original.cut != cut && !relocation) ||
                (item.relocate && !relocation) || !identical(original.command, item.command) ||
                original.purpose != item.purpose || original.request != item.request ||
                original.acknowledges != acknowledgment;
            if (changedIdentity) {
                out.error = "packet restoration changes original endpoint identity or provenance";
                return out;
            }
            if (relocation) {
                auto placed = original;
                if (placed.originalCut == NoAnalysisId) { placed.originalCut = original.cut; }
                placed.cut = cut;
                out.relocated.emplace(id, std::move(placed));
            }
            out.restored.push_back(id);
        } else {
            if (item.relocate) { out.error = "relocation requires an inactive acknowledgment identity"; return out; }
            id = out.firstEndpoint + out.endpoints.size();
            out.endpoints.push_back({id, cut, item.command, item.purpose, item.request, acknowledgment});
        }
        out.ordered.push_back(id);
        out.insertions[cut][offset].push_back(id);
    }
    out.ready = true;
    return out;
}
namespace {
// Merge in one pass rather than shifting the original word for every gap.
std::vector<Id> insertedWord(const std::vector<Id>& original,
                            const std::map<Id, std::vector<Id>>& insertions)
{
    std::vector<Id> out;
    Id first = 0;
    for (const auto& [offset, ids] : insertions) {
        out.insert(out.end(), original.begin() + first, original.begin() + offset);
        out.insert(out.end(), ids.begin(), ids.end());
        first = offset;
    }
    out.insert(out.end(), original.begin() + first, original.end());
    return out;
}
} // namespace
std::vector<Id> Ledger::appendPacket(const PreparedPacket& packet)
{
    if (!packet.ready || packet.owner != this || packet.version != revision ||
        packet.firstEndpoint != endpoints.size()) {
        return {};
    }
    for (const auto& endpoint : packet.endpoints) {
        recordEvent(endpoint);
    }
    for (const auto& [id, endpoint] : packet.relocated) { endpoints[id] = endpoint; }
    for (auto id : packet.restored) {
        setDormant(id, false);
    }
    endpoints.insert(endpoints.end(), packet.endpoints.begin(), packet.endpoints.end());
    for (const auto& [cut, insertions] : packet.insertions) {
        auto& word = words[cut];
        if (insertions.size() == 1 && insertions.begin()->first == word.size()) {
            // Ordinary append packets retain amortized append cost.
            const auto& added = insertions.begin()->second;
            word.insert(word.end(), added.begin(), added.end());
        } else {
            word = insertedWord(word, insertions);
        }
        changed.push_back(cut);
    }
    revision += packet.ordered.size();
    return packet.ordered;
}
std::optional<Commands> Ledger::withPacket(const PreparedPacket& packet) const
{
    if (!packet.ready || packet.owner != this || packet.version != revision ||
        packet.firstEndpoint != endpoints.size()) {
        return {};
    }
    Commands out(words.size());
    for (Cut cut = 0; cut < words.size(); ++cut) {
        const auto edits = packet.insertions.find(canonical(cut));
        const auto ids = edits == packet.insertions.end() ? word(cut) : insertedWord(word(cut), edits->second);
        for (auto id : ids) {
            out[cut].push_back(id < endpoints.size() ? endpoints[id].command :
                              packet.endpoints[id - packet.firstEndpoint].command);
        }
    }
    return out;
}
std::optional<PacketView> Ledger::packetView(const PreparedPacket& packet) const
{
    if (!packet.ready || packet.owner != this || packet.version != revision ||
        packet.firstEndpoint != endpoints.size()) { return {}; }
    PacketView view;
    view.ledger = this;
    view.packet = &packet;
    view.canonical = &canonicalCut;
    for (const auto& entry : packet.insertions) {
        view.words.emplace(entry.first, insertedWord(word(entry.first), entry.second));
    }
    return view;
}
const std::vector<Id>& PacketView::word(Cut site) const
{
    const auto found = words.find((*canonical)[site]);
    return found == words.end() ? ledger->word(site) : found->second;
}
const SelectedEndpoint& PacketView::endpoint(Id id) const
{
    const auto moved = packet->relocated.find(id);
    if (moved != packet->relocated.end()) { return moved->second; }
    return id < packet->firstEndpoint ? ledger->endpoint(id) : packet->endpoints[id - packet->firstEndpoint];
}
uint64_t PacketView::version() const { return packet->version + packet->ordered.size(); }
void Ledger::recordEvent(const SelectedEndpoint& endpoint)
{
    const auto& command = endpoint.command;
    if (command.kind == Command::Publish || command.kind == Command::Acquire) {
        byEvent[{command.source, command.observer, command.key}].push_back(endpoint.id);
    }
}
const std::vector<Id>& Ledger::eventUses(const EventIdentity& identity) const
{
    static const std::vector<Id> empty;
    const auto found = byEvent.find({identity.source, identity.observer, identity.key});
    return found == byEvent.end() ? empty : found->second;
}
bool Ledger::hasDormantUses(const EventIdentity& identity) const
{
    const auto found = dormantEvents.find({identity.source, identity.observer, identity.key});
    return found != dormantEvents.end() && found->second != 0;
}
void Ledger::setDormant(Id id, bool dormant)
{
    const auto& command = endpoints.at(id).command;
    if (dormant) {
        removed.insert(id);
    } else {
        removed.erase(id);
    }
    if (command.kind == Command::Publish || command.kind == Command::Acquire) {
        auto& count = dormantEvents[{command.source, command.observer, command.key}];
        if (dormant) {
            ++count;
        } else {
            --count;
        }
    }
}
std::optional<PacketEndpoint> Ledger::restoration(Id id, const WordGap& gap) const
{
    const bool unavailable = id >= endpoints.size() || active(id);
    if (unavailable) {
        return {};
    }
    const auto& e = endpoints[id];
    return PacketEndpoint{e.cut, e.command, e.purpose, e.request, e.acknowledges, gap, NoAnalysisId, id};
}
std::optional<PacketEndpoint> Ledger::relocateAcknowledgment(Id id, const WordGap& gap) const
{
    auto item = restoration(id, gap);
    if (!item || item->command.kind != Command::Acquire ||
        item->purpose != EndpointPurpose::ConsumptionAcknowledgment) { return {}; }
    item->cut = gap.cut;
    item->relocate = true;
    return item;
}
void Ledger::erase(Id id)
{
    if (!active(id)) {
        return;
    }
    const auto cut = endpoints.at(id).cut;
    auto& word = words[cut];
    word.erase(std::find(word.begin(), word.end(), id));
    setDormant(id, true);
    changed.push_back(cut);
    ++revision;
}
const std::vector<Id>& Ledger::word(Cut cut) const
{
    static const std::vector<Id> empty;
    if (cut >= words.size()) {
        return empty;
    }
    return words[canonical(cut)];
}
const SelectedEndpoint& Ledger::endpoint(Id id) const { return endpoints.at(id); }
Commands Ledger::commands() const
{
    Commands out(words.size());
    for (Cut cut = 0; cut < out.size(); ++cut) {
        for (auto id : word(cut)) {
            out[cut].push_back(endpoints[id].command);
        }
    }
    return out;
}

} // namespace mlir::pto::oahs::selected
