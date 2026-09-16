// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/OAHS/CausalFrontier.h"
#include <algorithm>
#include <iterator>
#include <limits>

namespace mlir::pto::oahs {
namespace {
FrontierBits bits(std::size_t n) { return FrontierBits(n / 64 + bool(n % 64)); }
void set(FrontierBits& b, std::size_t i) { b[i / 64] |= uint64_t(1) << (i % 64); }
void clear(FrontierBits& b, std::size_t i) { b[i / 64] &= ~(uint64_t(1) << (i % 64)); }
void unite(FrontierBits& a, const FrontierBits& b)
{
    for (std::size_t i = 0; i < a.size(); ++i)
        a[i] |= b[i];
}
void intersect(FrontierBits& a, const FrontierBits& b)
{
    for (std::size_t i = 0; i < a.size(); ++i)
        a[i] &= b[i];
}
bool sameKey(const EventIdentity& a, const EventIdentity& b)
{
    return a.source == b.source && a.observer == b.observer && a.key == b.key;
}
FrontierStep reject(const FrontierState& s, FrontierFailure failure, std::string reason)
{
    FrontierStep r;
    r.state = s;
    r.failure = failure;
    r.reason = std::move(reason);
    return r;
}
FrontierStep accepted(const FrontierState& s)
{
    FrontierStep r;
    r.applied = true;
    r.state = s;
    return r;
}
} // namespace

bool frontierContains(const FrontierBits& b, std::size_t i)
{
    return i / 64 < b.size() && (b[i / 64] & (uint64_t(1) << (i % 64)));
}

namespace detail {
struct CausalFrontierModel {
    const Program program;
    bool valid = false;
    std::string reason;
    std::vector<EventIdentity> keys;
    explicit CausalFrontierModel(Program p) : program(std::move(p))
    {
        auto validation = validateProgram(program);
        if (!validation.success) {
            reason = validation.reason;
            return;
        }
        if (program.finalBlocks) {
            reason = "causal frontier phase adapter is not qualified";
            return;
        }
        if (program.invocation.retirement != Program::InvocationContract::NoRetirement &&
            (program.invocation.retirement != Program::InvocationContract::DrainAllAtReturn ||
             !program.target.barrierAll)) {
            reason = "unavailable invocation retirement contract";
            return;
        }
        for (const auto& c : program.cells)
            if (c.exclusive) {
                reason = "causal frontier exclusive-resource adapter is not qualified";
                return;
            }
        for (const auto& op : program.operations)
            if (op.finalBlock || !op.resources.empty() || !op.visibility.empty() || !op.internalTransfers.empty() ||
                !op.authoredEvents.empty()) {
                reason = "causal frontier typed-effect adapter is not qualified";
                return;
            }
        for (unsigned a = 0; a < PipeCount; ++a)
            for (unsigned b = 0; b < PipeCount; ++b) {
                if (a == b || !program.target.supported[a] || !program.target.supported[b])
                    continue;
                auto numbers = program.target.keys[a][b];
                std::sort(numbers.begin(), numbers.end());
                numbers.erase(std::unique(numbers.begin(), numbers.end()), numbers.end());
                for (auto k : numbers)
                    keys.push_back({Pipe(a), Pipe(b), k, false});
            }
        for (const auto& reserved : program.reservations) {
            const auto& pool = program.target.keys;
            if (unsigned(reserved.source) >= PipeCount || unsigned(reserved.observer) >= PipeCount ||
                reserved.source == reserved.observer || !program.target.supported[unsigned(reserved.source)] ||
                !program.target.supported[unsigned(reserved.observer)]) {
                reason = "invalid causal frontier key reservation";
                return;
            }
            const auto& numbers = pool[unsigned(reserved.source)][unsigned(reserved.observer)];
            if (std::find(numbers.begin(), numbers.end(), reserved.key) == numbers.end()) {
                reason = "unavailable causal frontier key reservation";
                return;
            }
        }
        keys.erase(
            std::remove_if(
                keys.begin(), keys.end(),
                [&](const auto& key) {
                    return std::any_of(program.reservations.begin(), program.reservations.end(), [&](const auto& r) {
                        return sameKey(key, r);
                    });
                }),
            keys.end());
        // Representability checks, not analysis-work or candidate budgets.
        if (program.cells.size() > std::vector<std::optional<FrontierBits>>().max_size() / (2 * PipeCount) ||
            keys.size() > (std::vector<FrontierBits>().max_size() - 2 * PipeCount - 3) / 2) {
            reason = "causal frontier dimensions are not representable";
            return;
        }
        valid = true;
    }
    std::size_t ports() const { return 2 * PipeCount + 2 * keys.size(); }
    std::size_t publication(std::size_t e) const { return 2 * PipeCount + e; }
    std::size_t consumption(std::size_t e) const { return 2 * PipeCount + keys.size() + e; }
};
struct CausalFrontierState {
    std::shared_ptr<const CausalFrontierModel> model;
    FrontierFacts facts;
};
} // namespace detail

const FrontierFacts* FrontierState::facts() const { return data ? &data->facts : nullptr; }
bool FrontierState::operator==(const FrontierState& b) const
{
    return data == b.data || (data && b.data && data->model == b.data->model && data->facts == b.data->facts);
}
CausalFrontier::CausalFrontier(Program p) : model(std::make_shared<detail::CausalFrontierModel>(std::move(p))) {}
bool CausalFrontier::complete() const { return model->valid; }
const std::string& CausalFrontier::reason() const { return model->reason; }
const std::vector<EventIdentity>& CausalFrontier::keys() const { return model->keys; }
FrontierState CausalFrontier::initial() const
{
    FrontierState out;
    if (!complete())
        return out;
    auto s = std::make_shared<detail::CausalFrontierState>();
    s->model = model;
    const auto n = model->ports();
    auto active = bits(n);
    for (std::size_t i = 0; i < n; ++i)
        if (i < 2 * PipeCount || i >= 2 * PipeCount + keys().size())
            set(active, i);
    s->facts.reach.assign(n, bits(n));
    for (std::size_t i = 0; i < n; ++i)
        if (frontierContains(active, i))
            s->facts.reach[i] = active;
    s->facts.history.resize(model->program.cells.size() * PipeCount * 2);
    s->facts.events.resize(keys().size());
    out.data = std::move(s);
    return out;
}
FrontierStep CausalFrontier::checkState(const FrontierState& s) const
{
    if (!complete())
        return reject(s, FrontierFailure::UnsupportedContract, reason());
    if (s.data && s.data->model != model)
        return reject(s, FrontierFailure::InvalidInput, "frontier snapshot belongs to a different original program");
    return accepted(s);
}
FrontierStep CausalFrontier::join(const FrontierState& a, const FrontierState& b) const
{
    auto check = checkState(a);
    if (!check.applied)
        return check;
    check = checkState(b);
    if (!check.applied)
        return reject(a, check.failure, check.reason);
    if (!a.reachable())
        return accepted(b);
    if (!b.reachable() || a == b)
        return accepted(a);
    auto data = std::make_shared<detail::CausalFrontierState>(*a.data);
    auto& f = data->facts;
    const auto& other = b.data->facts;
    f.terminalRetired = f.terminalRetired && other.terminalRetired;
    f.mayBeRetired = f.mayBeRetired || other.mayBeRetired;
    // Each predecessor is already transitively closed. Intersecting closed
    // relations preserves shared consequences without splicing branch paths.
    for (std::size_t i = 0; i < f.reach.size(); ++i)
        intersect(f.reach[i], other.reach[i]);
    for (std::size_t i = 0; i < f.history.size(); ++i) {
        if (!f.history[i])
            f.history[i] = other.history[i];
        else if (other.history[i])
            intersect(*f.history[i], *other.history[i]);
    }
    for (std::size_t e = 0; e < keys().size(); ++e) {
        auto& event = f.events[e];
        event.occupancy |= other.events[e].occupancy;
        std::vector<FrontierBinding> bindings;
        std::set_union(
            event.publishers.begin(), event.publishers.end(), other.events[e].publishers.begin(),
            other.events[e].publishers.end(), std::back_inserter(bindings));
        event.publishers = std::move(bindings);
        if (event.occupancy != 2) {
            const auto port = model->publication(e);
            f.reach[port] = bits(model->ports());
            for (auto& row : f.reach)
                clear(row, port);
            for (auto& h : f.history)
                if (h)
                    clear(*h, port);
        }
    }
    FrontierState out;
    out.data = std::move(data);
    return accepted(out);
}
FrontierStep CausalFrontier::inspect(const FrontierState& s, std::size_t operation) const
{
    auto checked = checkState(s);
    if (!checked.applied)
        return checked;
    if (operation >= model->program.operations.size())
        return reject(s, FrontierFailure::InvalidInput, "invalid original physical phase");
    if (!s.reachable())
        return accepted(s);
    if (s.data->facts.mayBeRetired) {
        return reject(s, FrontierFailure::InvalidInput, "payload follows terminal retirement");
    }
    const auto& op = model->program.operations[operation];
    auto failed = reject(s, FrontierFailure::Payload, "unresolved original byte completion");
    // Merge duplicate effects before querying so every RMW role participates
    // in one residual and one update, independent of access-list ordering.
    std::vector<std::pair<bool, bool>> roles(model->program.cells.size());
    for (const auto& a : op.accesses) {
        roles[a.cell].first |= a.read;
        roles[a.cell].second |= a.write;
    }
    for (unsigned cell = 0; cell < roles.size(); ++cell) {
        const auto [read, write] = roles[cell];
        if (!read && !write)
            continue;
        for (unsigned source = 0; source < PipeCount; ++source)
            for (unsigned mode = 0; mode < 2; ++mode) {
                const auto& h = s.data->facts.history[(cell * PipeCount + source) * 2 + mode];
                if ((write || (read && mode)) && h && !frontierContains(*h, unsigned(op.pipe)))
                    failed.residuals.push_back({cell, Pipe(source), bool(mode), operation, read, write});
            }
    }
    if (!failed.residuals.empty())
        return failed;
    return accepted(s);
}
FrontierStep CausalFrontier::issue(const FrontierState& s, std::size_t operation) const
{
    auto checked = inspect(s, operation);
    if (!checked.applied || !s.reachable()) {
        return checked;
    }
    return extend(s, model->program.operations[operation].pipe, nullptr, operation, NoAnalysisId, {});
}
FrontierStep CausalFrontier::pendingIssue(const FrontierState& s, std::size_t operation) const
{
    auto checked = inspect(s, operation);
    if ((!checked.applied && checked.failure != FrontierFailure::Payload) || !s.reachable()) {
        return checked;
    }
    return extend(s, model->program.operations[operation].pipe, nullptr, operation, NoAnalysisId, {});
}
FrontierStep CausalFrontier::assumePreviousAccesses(
    const FrontierState& s, const std::vector<std::size_t>& operations) const
{
    auto checked = checkState(s);
    if (!checked.applied) {
        return checked;
    }
    for (auto operation : operations) {
        if (operation >= model->program.operations.size()) {
            return reject(s, FrontierFailure::InvalidInput, "invalid loop access hypothesis");
        }
    }
    if (!s.reachable()) {
        return accepted(s);
    }
    if (s.data->facts.mayBeRetired) {
        return reject(s, FrontierFailure::InvalidInput, "loop follows terminal retirement");
    }
    auto data = std::make_shared<detail::CausalFrontierState>(*s.data);
    for (auto operation : operations) {
        const auto& op = model->program.operations[operation];
        const auto sourcePrefix = PipeCount + unsigned(op.pipe);
        // A first-entry prefix can alias the fresh launch root. That equality
        // is not evidence that a possible previous body access has completed.
        // Forget its outgoing consequences before adding the may-history;
        // otherwise a later unrelated SET would transport invented credit.
        data->facts.reach[sourcePrefix] = bits(model->ports());
        set(data->facts.reach[sourcePrefix], sourcePrefix);
        auto history = bits(model->ports());
        set(history, sourcePrefix);
        if (model->program.target.synchronous[unsigned(op.pipe)]) {
            set(history, unsigned(op.pipe));
        }
        for (const auto& access : op.accesses) {
            const auto index = (std::size_t(access.cell) * PipeCount + unsigned(op.pipe)) * 2;
            if (access.read) {
                data->facts.history[index] = history;
            }
            if (access.write) {
                data->facts.history[index + 1] = history;
            }
        }
    }
    FrontierState out;
    out.data = std::move(data);
    return accepted(out);
}
FrontierStep CausalFrontier::command(const FrontierState& s, const Command& c, FrontierBinding binding) const
{
    auto checked = checkState(s);
    if (!checked.applied)
        return checked;
    if (!legalCommandCut(model->program, binding.cut) || binding.command == NoAnalysisId)
        return reject(s, FrontierFailure::InvalidInput, "command requires an original legal cut and endpoint identity");
    if (s.reachable() && s.data->facts.mayBeRetired) {
        return reject(s, FrontierFailure::InvalidInput, "command follows terminal retirement");
    }
    if (c.kind == Command::BarrierAll) {
        if (!model->program.target.barrierAll || binding.cut != invocationExitCut(model->program)) {
            return reject(s, FrontierFailure::UnsupportedContract, "only declared terminal ALL is supported");
        }
        if (!s.reachable()) {
            return accepted(s);
        }
        auto data = std::make_shared<detail::CausalFrontierState>(*s.data);
        data->facts.terminalRetired = true;
        data->facts.mayBeRetired = true;
        // This marker is checked only at the root exit. It neither consumes a
        // token nor grants interior memory or rearming credit.
        FrontierState out;
        out.data = std::move(data);
        return accepted(out);
    }
    if (c.kind != Command::Barrier && c.kind != Command::Publish && c.kind != Command::Acquire)
        return reject(s, FrontierFailure::InvalidInput, "invalid causal frontier command");
    std::size_t key = NoAnalysisId;
    if (c.kind == Command::Barrier) {
        if (unsigned(c.source) >= PipeCount || !model->program.target.supported[unsigned(c.source)] ||
            !model->program.target.barriers[unsigned(c.source)])
            return reject(s, FrontierFailure::InvalidInput, "unavailable named prefix fence");
    } else {
        EventIdentity identity{c.source, c.observer, c.key, false};
        const auto at = std::find_if(keys().begin(), keys().end(), [&](const auto& e) { return sameKey(e, identity); });
        if (at == keys().end())
            return reject(s, FrontierFailure::InvalidInput, "unavailable or reserved directional event key");
        key = std::size_t(at - keys().begin());
    }
    if (!s.reachable())
        return accepted(s);
    if (c.kind == Command::Publish && s.data->facts.events[key].occupancy != 1)
        return reject(s, FrontierFailure::PublicationNotEmpty, "publication requires must-empty balance");
    if (c.kind == Command::Acquire &&
        (s.data->facts.events[key].occupancy != 2 || s.data->facts.events[key].publishers.empty()))
        return reject(s, FrontierFailure::AcquisitionNotFull, "acquisition requires must-full matched publication");
    return extend(s, c.kind == Command::Acquire ? c.observer : c.source, &c, NoAnalysisId, key, binding);
}
FrontierStep CausalFrontier::exit(const FrontierState& s) const
{
    auto checked = checkState(s);
    if (!checked.applied || !s.reachable())
        return checked;
    for (const auto& event : s.data->facts.events)
        if (event.occupancy != 1)
            return reject(s, FrontierFailure::UnconsumedAtExit, "invocation may leave an unconsumed publication");
    if (model->program.invocation.retirement == Program::InvocationContract::DrainAllAtReturn &&
        !s.data->facts.terminalRetired) {
        return reject(s, FrontierFailure::MissingRetirement, "missing declared invocation retirement");
    }
    return accepted(s);
}

FrontierStep CausalFrontier::extend(
    const FrontierState& s, Pipe pipe, const Command* command, std::size_t operation, std::size_t key,
    FrontierBinding binding) const
{
    const bool publish = command && command->kind == Command::Publish;
    const bool acquire = command && command->kind == Command::Acquire;
    const bool fence = command && command->kind == Command::Barrier;
    const auto n = model->ports(), issue = n, finish = n + 1, aggregate = n + 2;
    const std::size_t gate = unsigned(pipe), prefix = PipeCount + unsigned(pipe);
    auto rows = s.data->facts.reach;
    rows.resize(n + 3);
    for (auto& row : rows)
        row.resize(bits(n + 3).size());
    // The retained relation is already transitively closed and the fresh
    // vertices have no edge back into it, so closure only adds, to each old
    // row, the fresh vertices it reaches through gate, prefix, or publication:
    // gate -> issue -> finish -> aggregate, prefix -> aggregate, and the
    // command-specific prefix -> finish (SET, fence) or S[e] -> finish (WAIT).
    for (auto v : {issue, finish, aggregate})
        set(rows[v], v);
    set(rows[issue], finish);
    set(rows[issue], aggregate);
    set(rows[finish], aggregate);
    const bool prefixFinish = publish || fence;
    for (std::size_t r = 0; r < n; ++r) {
        auto& row = rows[r];
        const bool toIssue = frontierContains(row, gate);
        const bool toFinish = toIssue || (prefixFinish && frontierContains(row, prefix)) ||
                              (acquire && frontierContains(row, model->publication(key)));
        if (toIssue)
            set(row, issue);
        if (toFinish)
            set(row, finish);
        if (toFinish || frontierContains(row, prefix))
            set(row, aggregate);
    }
    // Test the path that REALLY exists. Adding a desired rearm edge here would
    // turn the checker into its own oracle and acknowledge stale generations.
    if (publish && !frontierContains(rows[model->consumption(key)], finish))
        return reject(
            s, FrontierFailure::ConsumptionNotEstablished, "latest consumption does not precede this publication");
    auto data = std::make_shared<detail::CausalFrontierState>();
    data->model = model;
    auto& out = data->facts;
    out.events = s.data->facts.events;
    std::vector<std::size_t> mapping(n);
    for (std::size_t i = 0; i < n; ++i)
        mapping[i] = i;
    const bool synchronousPayload = !command && model->program.target.synchronous[unsigned(pipe)];
    mapping[gate] = acquire || fence || synchronousPayload ? finish : issue;
    mapping[prefix] = aggregate;
    if (publish) {
        mapping[model->publication(key)] = finish;
        out.events[key] = {2, {binding}};
    } else if (acquire) {
        mapping[model->consumption(key)] = finish;
        out.events[key] = {1, {}};
    }
    for (std::size_t e = 0; e < keys().size(); ++e)
        if (out.events[e].occupancy != 2)
            mapping[model->publication(e)] = NoAnalysisId;
    auto project = [&](const FrontierBits& reached) {
        auto b = bits(n);
        for (std::size_t i = 0; i < n; ++i)
            if (mapping[i] != NoAnalysisId && frontierContains(reached, mapping[i]))
                set(b, i);
        return b;
    };
    out.reach.assign(n, bits(n));
    for (std::size_t i = 0; i < n; ++i)
        if (mapping[i] != NoAnalysisId)
            out.reach[i] = project(rows[mapping[i]]);
    out.history = s.data->facts.history;
    for (auto& h : out.history)
        if (h) {
            auto image = bits(n + 3);
            for (std::size_t i = 0; i < n; ++i)
                if (frontierContains(*h, i))
                    unite(image, rows[i]);
            h = project(image);
        }
    if (!command) {
        auto latest = bits(n);
        set(latest, prefix);
        if (synchronousPayload) {
            set(latest, gate);
        }
        for (const auto& a : model->program.operations[operation].accesses) {
            const auto index = (std::size_t(a.cell) * PipeCount + unsigned(pipe)) * 2;
            if (a.read)
                out.history[index] = latest;
            if (a.write)
                out.history[index + 1] = latest;
        }
    }
    FrontierState result;
    result.data = std::move(data);
    return accepted(result);
}
} // namespace mlir::pto::oahs
