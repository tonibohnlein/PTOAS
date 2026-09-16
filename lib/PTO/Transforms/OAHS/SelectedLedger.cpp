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
Ledger::Ledger(const Program& p) : program(p), words(commandCutCount(p)) {}
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
        const auto leader = canonicalCommandCut(program, cut);
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
    cut = canonicalCommandCut(program, cut);
    const auto id = endpoints.size();
    endpoints.push_back({id, cut, command, purpose, request, ack});
    words[cut].insert(words[cut].begin() + offset, id);
    ++revision;
    changed.push_back(cut);
    return id;
}
Id Ledger::append(Cut cut, Command command, EndpointPurpose purpose, Id request, Id ack)
{
    const auto leader = canonicalCommandCut(program, cut);
    return insert(leader, words[leader].size(), command, purpose, request, ack);
}
Id Ledger::after(Id predecessor, Command command, EndpointPurpose purpose, Id request, Id ack)
{
    const auto cut = endpoints.at(predecessor).cut;
    const auto& ids = words[cut];
    const auto found = std::find(ids.begin(), ids.end(), predecessor);
    return insert(cut, Id(found - ids.begin()) + 1, command, purpose, request, ack);
}
const std::vector<Id>& Ledger::word(Cut cut) const
{
    static const std::vector<Id> empty;
    if (cut >= words.size()) {
        return empty;
    }
    return words[canonicalCommandCut(program, cut)];
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
