// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Fixture assertions only. The actual effects and words come from ordinary
// native import and selected construction, with no supplied candidate commands.
void diagnoseSlots(mlir::pto::oahs::NativeAnalysis input)
{
    namespace o = mlir::pto::oahs;
    if (!input.program.staticFifoSlots)
        std::abort();
    auto baseline = baselineWords(input.program);
    auto plan = o::constructSelectedPlan(input.program);
    auto check = o::checkCausalFrontier(input.program, plan.commands);
    llvm::errs() << "NATIVE_RESULT accepted=" << check.accepted << " split=" << plan.work.splitRelays
                 << " trials=" << plan.work.relayTrials << " trial_sites=" << plan.work.relayTrialSites << "\n";
    if (!plan.success || !check.accepted || plan.work.splitRelays < 2)
        std::abort();
    auto metadata = diagnosticModel(input);
    diagnosticPlan("native_relay", metadata, plan);
    std::map<std::string, o::Commands> mutations;
    auto same = [](const o::Command& a, const o::Command& b) {
        return a.kind == b.kind && a.source == b.source && a.observer == b.observer && a.key == b.key;
    };
    auto decisionFor = [&](unsigned operation) -> const o::SelectedDecision& {
        auto found = std::find_if(plan.decisions.begin(), plan.decisions.end(), [&](const auto& d) {
            return input.program.observed->sites[d.consumer].operation == operation && d.source == o::Pipe::FIX &&
                   d.observer == o::Pipe::MTE2;
        });
        if (found == plan.decisions.end() || found->endpoints.size() != 4)
            std::abort();
        return *found;
    };
    auto endpoint = [&](std::size_t id) -> const o::SelectedEndpoint& {
        auto found = std::find_if(plan.ledger.begin(), plan.ledger.end(), [&](const auto& e) { return e.id == id; });
        if (found == plan.ledger.end())
            std::abort();
        return *found;
    };
    auto remove = [&](o::Commands& words, const o::SelectedEndpoint& e) {
        auto& word = words[e.cut];
        auto at = std::find_if(word.begin(), word.end(), [&](const auto& c) { return same(c, e.command); });
        if (at == word.end())
            std::abort();
        word.erase(at);
    };
    const auto& early = decisionFor(23);
    const auto& later = decisionFor(28);
    if (endpoint(early.endpoints[0]).command.observer != o::Pipe::MTE1 ||
        endpoint(later.endpoints[0]).command.observer != o::Pipe::M ||
        endpoint(later.endpoints[1]).cut == later.consumer)
        std::abort();
    for (unsigned leg = 0; leg < 2; ++leg) {
        auto words = plan.commands;
        remove(words, endpoint(early.endpoints[2 * leg]));
        remove(words, endpoint(early.endpoints[2 * leg + 1]));
        mutations[leg ? "missing_second_leg" : "missing_first_leg"] = std::move(words);
    }
    auto missing = plan.commands;
    for (auto id : later.endpoints)
        remove(missing, endpoint(id));
    mutations["missing_later_receipt"] = std::move(missing);
    auto late = plan.commands;
    for (unsigned i : {1u, 2u})
        remove(late, endpoint(later.endpoints[i]));
    const auto& wait = endpoint(later.endpoints[3]);
    auto& word = late[later.consumer];
    auto at = std::find_if(word.begin(), word.end(), [&](const auto& c) { return same(c, wait.command); });
    if (at == word.end())
        std::abort();
    word.insert(at, {endpoint(later.endpoints[1]).command, endpoint(later.endpoints[2]).command});
    mutations["late_forwarding"] = std::move(late);
    for (const auto& [name, words] : mutations) {
        auto checked = o::checkCausalFrontier(input.program, words);
        llvm::errs() << "MUTATION " << name << " accepted=" << checked.accepted << " reason=" << checked.reason << "\n";
        if (checked.accepted != (name == "late_forwarding"))
            std::abort();
    }
    finiteRelay(input.program, baseline, plan.commands, mutations);
}
