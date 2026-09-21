// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_OAHS_FIFO_SLOT_FRONTEND_H
#define PTO_OAHS_FIFO_SLOT_FRONTEND_H
#include "PTO/Transforms/OAHS/ObservedPrograms.h"
#include <algorithm>
#include <array>
#include <limits>
#include <map>

namespace mlir::pto::oahs {
ObservedImport refineStaticSlots(const Program& input, const AlternatingSlotRegion& region)
{
    ObservedImport result;
    auto refuse = [&](const char* why) {
        result.reason = why;
        return result;
    };
    if (!validateProgram(input).success || !input.observed || input.alternatingSlots || input.staticFifoSlots ||
        region.slots != 2 || !region.slotBytes || region.slotBytes > std::numeric_limits<uint64_t>::max() / 2 ||
        region.cell >= input.cells.size() || input.cells[region.cell].exclusive || region.reads.empty() ||
        region.writes.empty())
        return refuse("unsupported static FIFO slot interface");
    std::vector<unsigned> role(input.operations.size()), slots(input.operations.size());
    for (unsigned kind : {1u, 2u}) {
        for (auto op : kind == 1 ? region.reads : region.writes) {
            if (op >= role.size() || role[op])
                return refuse("invalid static slot binding");
            unsigned effect = 0;
            for (auto access : input.operations[op].accesses)
                if (access.cell == region.cell)
                    effect |= unsigned(access.read) | (unsigned(access.write) << 1);
            if (effect != kind)
                return refuse("static slot binding differs from original access");
            role[op] = kind;
        }
    }
    for (std::size_t op = 0; op < role.size(); ++op)
        if (!role[op])
            for (auto access : input.operations[op].accesses)
                if (access.cell == region.cell)
                    return refuse("unrepresented static FIFO access");
    const auto& graph = *input.observed;
    // Each monotone fact has only two bits. Keep the two cursors independent;
    // do not materialize a product with one another or with loop observations.
    for (unsigned kind : {1u, 2u}) {
        std::vector<unsigned> masks(graph.sites.size());
        std::vector<Cut> work{graph.entry};
        masks[graph.entry] = 1;
        for (std::size_t i = 0; i < work.size(); ++i) {
            const auto at = work[i], op = graph.sites[at].operation;
            unsigned next = masks[at];
            if (op != NoControlId && role[op] == kind) {
                slots[op] |= next;
                next = ((next & 1) << 1) | ((next & 2) >> 1);
            }
            for (auto to : graph.sites[at].successors) {
                const auto joined = masks[to] | next;
                if (joined != masks[to]) {
                    masks[to] = joined;
                    work.push_back(to);
                }
            }
        }
    }
    for (std::size_t op = 0; op < role.size(); ++op)
        if (role[op] && slots[op] != 1 && slots[op] != 2)
            return refuse("static FIFO access has ambiguous or unreachable cursor");
    result.program = input;
    auto& p = result.program;
    const auto first = p.cells.size();
    p.staticFifoSlots = Program::StaticFifoSlots{{unsigned(first), unsigned(first + 1)}, region.reads, region.writes};
    for (unsigned slot = 0; slot < 2; ++slot) {
        auto cell = input.cells[region.cell];
        cell.storage = Cell::Storage::CanonicalInterval;
        cell.unknownRange = false;
        cell.coordinateSpace += "; qualified-static-FIFO-root:" + std::to_string(region.cell);
        cell.ranges = {{slot * region.slotBytes, region.slotBytes}};
        cell.provenance = "lowering-qualified static FIFO cursor";
        p.cells.push_back(std::move(cell));
    }
    for (std::size_t op = 0; op < role.size(); ++op)
        if (role[op])
            for (auto& access : p.operations[op].accesses)
                if (access.cell == region.cell) {
                    access.cell = first + unsigned(slots[op] == 2);
                    access.definiteWrite = false;
                }
    auto valid = validateProgram(p);
    result.success = valid.success;
    result.reason = valid.reason;
    return result;
}
ObservedImport refineAlternatingSlots(const Program& input, const AlternatingSlotRegion& region)
{
    ObservedImport result;
    auto refuse = [&](const char* why) {
        result.reason = why;
        return result;
    };
    if (!validateProgram(input).success || !input.observed || input.alternatingSlots || input.staticFifoSlots || region.slots != 2 ||
        !region.slotBytes || region.slotBytes > std::numeric_limits<uint64_t>::max() / 2 ||
        region.cell >= input.cells.size() || input.cells[region.cell].exclusive || region.reads.empty() ||
        region.writes.empty())
        return refuse("unsupported alternating slot interface");
    const auto& old = *input.observed;
    std::vector<unsigned> role(input.operations.size());
    Pipe reader = Pipe::Count, writer = Pipe::Count;
    for (unsigned kind : {1u, 2u}) {
        auto& pipe = kind == 1 ? reader : writer;
        for (auto op : kind == 1 ? region.reads : region.writes) {
            if (op >= role.size() || role[op])
                return refuse("invalid slot operation binding");
            const auto& operation = input.operations[op];
            if (pipe != Pipe::Count && pipe != operation.pipe)
                return refuse("mixed slot pipeline");
            pipe = operation.pipe;
            unsigned effect = 0;
            for (auto a : operation.accesses)
                if (a.cell == region.cell)
                    effect |= unsigned(a.read) | (unsigned(a.write) << 1);
            if (effect != kind)
                return refuse("slot binding differs from original access");
            role[op] = kind;
        }
    }
    if (reader == writer)
        return refuse("slot interface needs distinct pipelines");
    for (std::size_t op = 0; op < role.size(); ++op)
        if (!role[op])
            for (auto a : input.operations[op].accesses)
                if (a.cell == region.cell)
                    return refuse("unrepresented FIFO root access");
    // One stage per original site proves complete participation. Unlike a
    // counter product this declines joins with unequal push/pop populations.
    std::vector<int> stage(old.sites.size(), -1);
    std::vector<Cut> work{old.entry};
    stage[old.entry] = 0;
    for (std::size_t i = 0; i < work.size(); ++i) {
        auto at = work[i];
        auto op = old.sites[at].operation;
        int next = stage[at];
        if (op != NoControlId && role[op]) {
            if (role[op] != unsigned(next + 1))
                return refuse("nonalternating FIFO participation");
            next = 1 - next;
        }
        if (at == old.exit && next)
            return refuse("incomplete FIFO episode at exit");
        for (auto to : old.sites[at].successors) {
            if (stage[to] < 0) {
                stage[to] = next;
                work.push_back(to);
            } else if (stage[to] != next)
                return refuse("unequal FIFO participation at join");
        }
    }
    result.program = input;
    auto& p = result.program;
    auto& q = *p.observed;
    Program::AlternatingSlots interface;
    interface.reader = reader;
    interface.writer = writer;
    for (unsigned slot = 0; slot < 2; ++slot) {
        interface.cells.push_back(p.cells.size());
        auto cell = input.cells[region.cell];
        cell.storage = Cell::Storage::CanonicalInterval;
        cell.unknownRange = false;
        cell.coordinateSpace += "; qualified-FIFO-root:" + std::to_string(region.cell);
        cell.ranges = {{slot * region.slotBytes, region.slotBytes}};
        cell.provenance = "lowering-qualified participating FIFO slot";
        p.cells.push_back(std::move(cell));
    }
    // At most one additional site per original site, independent of trip
    // counts and the number of guards. Invocation exit joins both phases.
    std::vector<std::array<Cut, 2>> copies(old.sites.size(), {NoControlId, NoControlId});
    std::vector<std::pair<Cut, unsigned>> pending;
    auto locate = [&](Cut at, unsigned phase) {
        if (at == old.exit)
            phase = 0;
        auto& id = copies[at][phase];
        if (id != NoControlId)
            return id;
        id = phase ? q.sites.size() : at;
        if (phase)
            q.sites.push_back(old.sites[at]);
        pending.emplace_back(at, phase);
        return id;
    };
    q.entry = locate(old.entry, 0);
    std::map<std::pair<std::size_t, unsigned>, std::size_t> phases;
    for (std::size_t i = 0; i < pending.size(); ++i) {
        auto [at, phase] = pending[i];
        auto id = copies[at][phase];
        auto op = old.sites[at].operation;
        if (op != NoControlId && role[op]) {
            auto key = std::make_pair(op, phase);
            auto found = phases.find(key);
            if (found == phases.end()) {
                auto copy = input.operations[op];
                for (auto& a : copy.accesses)
                    if (a.cell == region.cell) {
                        a.cell = interface.cells[phase];
                        a.definiteWrite = false;
                    }
                auto number = phase ? p.operations.size() : op;
                if (phase)
                    p.operations.push_back(copy);
                else
                    p.operations[op] = copy;
                found = phases.emplace(key, number).first;
            }
            q.sites[id].operation = found->second;
            if (role[op] == 1)
                interface.reads.push_back(id);
        }
        unsigned nextPhase = phase ^ (op != NoControlId && role[op] == 2);
        std::vector<Cut> next;
        for (auto to : old.sites[at].successors)
            next.push_back(locate(to, nextPhase));
        q.sites[id].successors = std::move(next);
    }
    // This interface owns the sole hidden dimension. No old counted/recurring
    // placement descriptor may be carried through with incomplete boundaries.
    q.loops.clear();
    q.qualification += "; alternating-two-slot-FIFO-v1";
    p.alternatingSlots = std::move(interface);
    const auto valid = validateProgram(p);
    result.success = valid.success;
    result.reason = valid.reason;
    return result;
}
} // namespace mlir::pto::oahs
#endif
