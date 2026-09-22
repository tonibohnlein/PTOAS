// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Bounded, path-specific open-fragment certificate. Entry A/T/S/D ports are
// independent symbolic inputs. Payload issues/finishes have distinct symbols;
// commands propagate dependencies without inventing a payload completion.
// Compare all payload observations and exit ports, including outward event
// publications and consumption representatives. Token legality is separate.
#include "SelectedInternal.h"

namespace mlir::pto::oahs::selected {
namespace {
using Signature = std::set<Id>;
void include(Signature& target, const Signature& source)
{
    target.insert(source.begin(), source.end());
}
bool subset(const Signature& left, const Signature& right)
{
    return std::includes(right.begin(), right.end(), left.begin(), left.end());
}
struct Ports {
    std::vector<Signature> values;
    Id keys;
    explicit Ports(Id count) : values(2 * PipeCount + 2 * count), keys(count)
    {
        for (Id i = 0; i < values.size(); ++i) {
            values[i].insert(i);
        }
    }
    void command(const Command& command, Id key)
    {
        if (command.kind == Command::BarrierAll) {
            Signature done;
            for (Id i = 0; i < 2 * PipeCount; ++i) {
                include(done, values[i]);
            }
            for (Id i = 0; i < 2 * PipeCount; ++i) {
                values[i] = done;
            }
            return;
        }
        const auto pipe = Id(command.kind == Command::Acquire ? command.observer : command.source);
        auto done = values[pipe];
        const auto input = command.kind == Command::Acquire ? 2 * PipeCount + key : PipeCount + pipe;
        include(done, values[input]);
        include(values[PipeCount + pipe], done);
        if (command.kind == Command::Publish) {
            values[2 * PipeCount + key] = std::move(done);
        } else {
            values[pipe] = done;
            if (command.kind == Command::Acquire) {
                values[2 * PipeCount + keys + key] = std::move(done);
            }
        }
    }
    Signature payload(Pipe pipe, Cut site)
    {
        auto& issued = values[Id(pipe)];
        issued.insert(values.size() + 2 * site);
        auto done = issued;
        done.insert(values.size() + 2 * site + 1);
        include(values[PipeCount + Id(pipe)], done);
        return done;
    }
    bool containedIn(const Ports& other) const
    {
        for (Id i = 0; i < values.size(); ++i) {
            if (!subset(values[i], other.values[i])) {
                return false;
            }
        }
        return true;
    }
};
} // namespace

bool certifyPublicationOrder(const Program& program, const Control& control, const Ledger& ledger,
    const std::vector<EventIdentity>& identities, Id publication, Cut target, Id offset,
    std::vector<Cut>& crossed)
{
    crossed.clear();
    const bool valid = control.complete && publication < ledger.records().size() &&
        ledger.active(publication) && target < control.graph.sites.size();
    if (!valid) {
        return false;
    }
    const auto endpoint = ledger.endpoint(publication);
    const auto destination = control.canonicalCut[target];
    const auto& correspondence = control.correspondence(destination, endpoint.cut);
    const bool ordinary = endpoint.command.kind == Command::Publish && endpoint.cut != destination &&
        offset <= ledger.word(destination).size() && correspondence.qualified;
    if (!ordinary) {
        return false;
    }
    std::map<std::tuple<Pipe, Pipe, unsigned>, Id> keys;
    for (Id i = 0; i < identities.size(); ++i) {
        const auto& key = identities[i];
        keys.emplace(std::make_tuple(key.source, key.observer, key.key), i);
    }
    auto index = [&](const Command& command) {
        const auto found = keys.find({command.source, command.observer, command.key});
        return found == keys.end() ? NoAnalysisId : found->second;
    };
    struct Path {
        Cut at, source;
        Ports old, next;
        std::set<Cut> visited;
    };
    const auto publicationKey = index(endpoint.command);
    if (publicationKey == NoAnalysisId) {
        return false;
    }
    std::vector<Path> todo;
    std::set<Cut> sources;
    for (const auto& pair : correspondence.pairs) {
        if (!sources.insert(pair.first).second) {
            continue;
        }
        if (!control.sourceCut(pair.first, endpoint.command.source)) {
            return false;
        }
        Path entry{pair.first, pair.first, Ports(keys.size()), Ports(keys.size()), {}};
        entry.next.command(endpoint.command, publicationKey);
        todo.push_back(std::move(entry));
    }
    std::set<Cut> sites;
    unsigned completed = 0, steps = 0;
    // Conservative compile-time bounds, not a limit on admitted executions.
    // Each accepted fragment has no backedge and at most 64 original paths.
    while (!todo.empty()) {
        auto path = std::move(todo.back());
        todo.pop_back();
        if (++steps > 8192 || !path.visited.insert(path.at).second) {
            return false;
        }
        sites.insert(control.canonicalCut[path.at]);
        const auto& word = ledger.word(path.at);
        bool finished = false;
        for (Id i = path.at == path.source ? offset : 0; i < word.size(); ++i) {
            const auto& command = ledger.endpoint(word[i]).command;
            if (word[i] == publication) {
                path.old.command(command, index(command));
                finished = true;
                break;
            }
            const bool event = command.kind == Command::Publish || command.kind == Command::Acquire;
            const auto key = index(command);
            if (event && (key == NoAnalysisId || key == publicationKey)) {
                return false;
            }
            path.old.command(command, index(command));
            path.next.command(command, index(command));
        }
        if (finished) {
            const bool paired = std::binary_search(correspondence.pairs.begin(), correspondence.pairs.end(),
                                                   std::make_pair(path.source, path.at));
            if (++completed > 64 || !paired || !path.next.containedIn(path.old)) {
                return false;
            }
            continue;
        }
        const auto operation = control.graph.operations[path.at];
        if (operation != NoAnalysisId) {
            const auto pipe = program.operations[operation].pipe;
            const auto oldDone = path.old.payload(pipe, path.at);
            const auto newDone = path.next.payload(pipe, path.at);
            const bool included = subset(newDone, oldDone) &&
                subset(path.next.values[Id(pipe)], path.old.values[Id(pipe)]);
            if (!included) {
                return false;
            }
        }
        const auto& node = control.graph.sites[path.at];
        const bool backedge = std::any_of(node.backedgeOwners.begin(), node.backedgeOwners.end(),
            [](Id owner) { return owner != NoAnalysisId; });
        const bool unsupported = node.successors.empty() || backedge;
        if (unsupported) {
            return false;
        }
        for (auto next : node.successors) {
            auto child = path;
            child.at = next;
            todo.push_back(std::move(child));
        }
    }
    crossed.assign(sites.begin(), sites.end());
    return completed != 0;
}
} // namespace mlir::pto::oahs::selected
