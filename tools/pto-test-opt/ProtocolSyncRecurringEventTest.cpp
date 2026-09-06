// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

// Independent finite binary-token execution, not production graph closure.
#include "PTO/Transforms/ProtocolSync/RecurringEventLifetime.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <utility>
#include <vector>

using namespace mlir::pto::protocol_sync;

namespace {

using Channel = SyncRecurringEventChannel;
using Command = std::pair<unsigned, bool>;                // channel, set versus wait
using State = std::pair<std::vector<unsigned>, unsigned>; // lane cursors, tokens

bool tokenOracle(const std::vector<Channel>& channels, unsigned trips)
{
    std::map<unsigned, std::map<unsigned, Command>> body;
    for (unsigned id = 0; id < channels.size(); ++id) {
        const auto& channel = channels[id];
        body[channel.setLane][channel.setOrder] = {id, true};
        body[channel.waitLane][channel.waitOrder] = {id, false};
    }
    std::vector<std::vector<Command>> queues;
    for (const auto& lane : body) {
        std::vector<Command> commands;
        for (unsigned id = 0; id < channels.size(); ++id) {
            if (channels[id].primed && channels[id].setLane == lane.first) {
                commands.push_back({id, true});
            }
        }
        for (unsigned iteration = 0; iteration < trips; ++iteration) {
            for (const auto& entry : lane.second) {
                commands.push_back(entry.second);
            }
        }
        for (unsigned id = 0; id < channels.size(); ++id) {
            if (channels[id].drained && channels[id].waitLane == lane.first) {
                commands.push_back({id, false});
            }
        }
        queues.push_back(std::move(commands));
    }
    std::vector<State> pending{{std::vector<unsigned>(queues.size()), 0}};
    std::set<State> visited;
    while (!pending.empty()) {
        State state = std::move(pending.back());
        pending.pop_back();
        if (!visited.insert(state).second) {
            continue;
        }
        bool finished = true;
        bool progress = false;
        for (unsigned lane = 0; lane < queues.size(); ++lane) {
            if (state.first[lane] == queues[lane].size()) {
                continue;
            }
            finished = false;
            const auto command = queues[lane][state.first[lane]];
            const unsigned mask = 1U << command.first;
            const bool full = (state.second & mask) != 0;
            if (command.second && full) {
                return false; // Illegal rearming, not a blocking set.
            }
            if (!command.second && !full) {
                continue;
            }
            State next = state;
            ++next.first[lane];
            next.second ^= mask;
            pending.push_back(std::move(next));
            progress = true;
        }
        const bool deadlockedOrLeaked = (!finished && !progress) || (finished && state.second != 0);
        if (deadlockedOrLeaked) {
            return false;
        }
    }
    return true;
}

bool check(bool result, const char* detail)
{
    if (!result) {
        llvm::errs() << "FAIL recurring events: " << detail << '\n';
    }
    return result;
}

bool testForkJoin()
{
    // MTE2=0, V=1, MTE3=2. Actual hazard handoffs only; no load/load edge.
    // readyA, readyB, readyC, releaseA, releaseB.
    std::vector<Channel> channels{
        {0, 1, 1, 0, 0, false, false},
        {0, 1, 3, 1, 0, false, false},
        {1, 2, 2, 0, 0, false, false},
        {2, 0, 1, 0, 1, true, true},
        {1, 0, 3, 2, 1, true, true}};
    const auto proof = proveRecurringEventLifetimes(channels);
    if (!check(proof.status == SyncRecurringEventStatus::Proven, "selective fork/join induction")) {
        return false;
    }
    for (unsigned trips : {0U, 1U, 2U, 3U, 4U, 7U}) {
        if (!check(tokenOracle(channels, trips), "fork/join finite execution")) {
            return false;
        }
    }
    auto broken = channels;
    broken[3].primed = false;
    if (!check(
            proveRecurringEventLifetimes(broken).status == SyncRecurringEventStatus::InvalidContract,
            "missing prime")) {
        return false;
    }
    broken = channels;
    broken[4].drained = false;
    if (!check(
            proveRecurringEventLifetimes(broken).status == SyncRecurringEventStatus::InvalidContract,
            "missing drain")) {
        return false;
    }
    broken = channels;
    broken[0].setOrder = broken[1].setOrder;
    if (!check(
            proveRecurringEventLifetimes(broken).status == SyncRecurringEventStatus::InvalidContract,
            "ambiguous event position")) {
        return false;
    }
    return true;
}

bool testEnumerated()
{
    unsigned admitted = 0;
    unsigned rejected = 0;
    // All orders/directions/distances of two two-lane channels. The oracle
    // executes literal primes/body/drains and explores every enabled action.
    for (unsigned directions = 0; directions < 4; ++directions) {
        for (unsigned distances = 0; distances < 4; ++distances) {
            std::array<unsigned, 4> order{0, 1, 2, 3};
            do {
                std::vector<Channel> channels;
                for (unsigned id = 0; id < 2; ++id) {
                    const unsigned source = (directions >> id) & 1U;
                    const unsigned distance = (distances >> id) & 1U;
                    channels.push_back(
                        {source, 1 - source, order[2 * id], order[2 * id + 1], distance, distance == 1, distance == 1});
                }
                const bool proven = proveRecurringEventLifetimes(channels).status == SyncRecurringEventStatus::Proven;
                bool safe = true;
                for (unsigned trips = 0; trips <= 3; ++trips) {
                    safe &= tokenOracle(channels, trips);
                }
                if (!check(!proven || safe, "accepted induction has token counterexample")) {
                    return false;
                }
                admitted += proven;
                rejected += !proven;
            } while (std::next_permutation(order.begin(), order.end()));
        }
    }
    return check(admitted != 0 && rejected != 0 && admitted + rejected == 384, "enumeration populations");
}

bool testNegativeProofs()
{
    const std::vector<Channel> forward{{0, 1, 0, 0, 0, false, false}};
    const std::vector<Channel> carried{{0, 1, 0, 0, 1, true, true}};
    const std::vector<Channel> deadlock{{0, 1, 1, 0, 0, false, false}, {1, 0, 1, 0, 0, false, false}};
    // A consumed carried credit reaches its rearm only one iteration too late.
    // A reachability check that forgets iteration distance would accept this.
    const std::vector<Channel> tooFar{{0, 1, 1, 0, 1, true, true}, {1, 0, 1, 0, 1, true, true}};
    const bool noAck = proveRecurringEventLifetimes(forward).status == SyncRecurringEventStatus::UnprovedRearm;
    const bool initialRearm = proveRecurringEventLifetimes(carried).status == SyncRecurringEventStatus::UnprovedRearm;
    const bool cycle = proveRecurringEventLifetimes(deadlock).status == SyncRecurringEventStatus::ZeroDistanceCycle;
    const bool wrongDistance = proveRecurringEventLifetimes(tooFar).status == SyncRecurringEventStatus::UnprovedRearm;
    return check(noAck && !tokenOracle(forward, 2), "forward rearm witness") &&
           check(initialRearm && !tokenOracle(carried, 1), "initial-credit rearm witness") &&
           check(cycle && !tokenOracle(deadlock, 1), "zero-distance deadlock witness") &&
           check(wrongDistance && !tokenOracle(tooFar, 2), "wrong-distance rearm witness");
}

bool testContractBounds()
{
    std::vector<Channel> channels;
    for (unsigned pair = 0; pair < 32; ++pair) {
        const unsigned source = 2 * pair;
        channels.push_back({source, source + 1, 1, 0, 0, false, false});
        channels.push_back({source + 1, source, 1, 0, 1, true, true});
    }
    const bool maximum = proveRecurringEventLifetimes(channels).status == SyncRecurringEventStatus::Proven;
    channels.push_back(channels.front());
    const bool bounded = proveRecurringEventLifetimes(channels).status == SyncRecurringEventStatus::AnalysisLimit;
    channels.resize(1);
    channels[0].distance = 2;
    const bool distance = proveRecurringEventLifetimes(channels).status == SyncRecurringEventStatus::InvalidContract;
    channels[0].distance = 0;
    channels[0].waitLane = channels[0].setLane;
    const bool lane = proveRecurringEventLifetimes(channels).status == SyncRecurringEventStatus::InvalidContract;
    return check(maximum && bounded && distance && lane, "contract and immutable budget boundaries");
}

} // namespace

bool testProtocolSyncRecurringEvents()
{
    const bool passed = testContractBounds() && testForkJoin() && testEnumerated() && testNegativeProofs();
    if (passed) {
        llvm::outs() << "protocol-sync recurring event induction: fork/join and 384 token encodings pass\n";
    }
    return passed;
}
