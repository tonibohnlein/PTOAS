// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Algebraic storage generators without expanding banks or loop iterations.
#include "PTO/Transforms/FrontierSynch/RotatingBoundary.h"
#include "PeriodicAnalysisInternal.h"
#include "PTO/Transforms/FrontierSynch/HardwareProtection.h"
#include "llvm/ADT/APInt.h"
#include <algorithm>
#include <numeric>
#include <set>
#include <tuple>
namespace mlir::pto::frontiersynch {
namespace {
using Occ = BoundaryOccurrence;
uint64_t inverse(uint64_t value, uint64_t modulus)
{
    uint64_t a = modulus, b = value;
    llvm::APInt x(256, 0), y(256, 1);
    while (b) {
        auto next = x - llvm::APInt(256, a / b) * y;
        auto remainder = a % b;
        a = b;
        b = remainder;
        x = y;
        y = next;
    }
    auto answer = x.srem(llvm::APInt(256, modulus));
    if (answer.isNegative()) {
        answer += llvm::APInt(256, modulus);
    }
    return answer.getZExtValue();
}
std::optional<std::pair<uint64_t, uint64_t>> progression(const RotatingFragment& f, uint64_t slot)
{
    const auto stride = f.stride % f.slots, offset = f.offset % f.slots;
    const auto divisor = std::gcd(stride, f.slots), period = f.slots / divisor;
    const auto delta = slot >= offset ? slot - offset : f.slots - (offset - slot);
    if (delta % divisor) {
        return std::nullopt;
    }
    const auto first = period == 1 ?
                           0 :
                           (llvm::APInt(128, delta / divisor) * llvm::APInt(128, inverse(stride / divisor, period)))
                               .urem(llvm::APInt(128, period))
                               .getZExtValue();
    return std::make_pair(first, period);
}
bool before(Occ a, Occ b, uint64_t length)
{
    return std::make_pair(a.at(length), a.type) < std::make_pair(b.at(length), b.type);
}
void choose(std::optional<Occ>& old, Occ next, uint64_t length, bool first)
{
    if (!old || (first ? before(next, *old, length) : before(*old, next, length))) {
        old = next;
    }
}
void choose(std::map<uint32_t, Occ>& values, uint32_t pipe, Occ next, uint64_t length, bool first)
{
    auto found = values.find(pipe);
    if (found == values.end()) {
        values.emplace(pipe, next);
    } else if (first ? before(next, found->second, length) : before(found->second, next, length)) {
        found->second = next;
    }
}
std::optional<bool> reaches(
    const PeriodicAnalysis& q, Occ a, PeriodicEventKind ak, Occ b, PeriodicEventKind bk, uint64_t length)
{
    auto threshold = q.eventThreshold({a.type, ak}, {b.type, bk});
    if (threshold.error != PeriodicQueryError::None) {
        return std::nullopt;
    }
    return threshold.displacement && a.at(length) <= b.at(length) &&
           b.at(length) - a.at(length) >= *threshold.displacement;
}
void indexBoundary(RotatingBoundaryType& type, const PeriodicAnalysis& quotient)
{
    std::vector<std::pair<Occ, PeriodicEventKind>> events;
    std::map<std::pair<uint32_t, PeriodicEventKind>, std::vector<uint32_t>> groups;
    auto add = [&](Occ occurrence) {
        for (auto kind : {PeriodicEventKind::Start, PeriodicEventKind::Completion}) {
            if (events.size() >= UINT32_MAX) {
                type.index.error = "boundary event identity overflow";
                return;
            }
            auto key = std::make_tuple(occurrence.type, occurrence.at(type.representative), kind);
            if (type.eventIds.emplace(key, events.size()).second) {
                groups[{quotient.payloads[occurrence.type].pipe, kind}].push_back(events.size());
                events.emplace_back(occurrence, kind);
            }
        }
    };
    for (auto port : type.relationshipPorts) {
        add(port);
    }
    for (const auto& cell : type.cells) {
        if (cell.firstWriter) {
            add(*cell.firstWriter);
        }
        if (cell.lastWriter) {
            add(*cell.lastWriter);
        }
        for (const auto& [pipe, value] : cell.firstReaders) {
            add(value);
        }
        for (const auto& [pipe, value] : cell.lastReaders) {
            add(value);
        }
    }
    for (const auto& [pipe, value] : type.firstPayloads) {
        add(value);
    }
    for (const auto& [pipe, value] : type.lastPayloads) {
        add(value);
    }
    if (!type.index.error.empty()) {
        return;
    }
    std::vector<std::vector<uint32_t>> chains;
    for (auto& [key, list] : groups) {
        std::sort(list.begin(), list.end(), [&](uint32_t a, uint32_t b) {
            return before(events[a].first, events[b].first, type.representative);
        });
        chains.push_back(std::move(list));
    }
    type.index = buildNumericalChainInterface(std::move(chains), [&](uint32_t a, uint32_t b) {
        return reaches(
            quotient, events[a].first, events[a].second, events[b].first, events[b].second, type.representative);
    });
}
} // namespace
RotatingBoundaryType RotatingBoundaryCertificate::select(uint64_t length) const
{
    RotatingBoundaryType result;
    result.representative = length;
    result.cells.resize(cells.size());
    if (!length || !error.empty()) {
        return result;
    }
    for (uint32_t type = 0; type < quotient.payloads.size(); ++type) {
        auto pipe = quotient.payloads[type].pipe;
        result.firstPayloads.emplace(pipe, Occ{type, 0, false});
        result.lastPayloads[pipe] = {type, 1, true};
    }
    for (std::size_t cell = 0; cell < cells.size(); ++cell) {
        auto& out = result.cells[cell];
        struct Use {
            Occ first, last;
            bool read, write;
        };
        std::map<std::pair<uint32_t, uint64_t>, Use> uses;
        for (const auto& f : fragments) {
            if (f.family != cells[cell].family || f.atom != cells[cell].atom) {
                continue;
            }
            auto visit = progression(f, cells[cell].slot);
            if (!visit || visit->first >= length) {
                continue;
            }
            const auto last = length - 1 - ((length - 1 - visit->first) % visit->second);
            auto key = std::make_pair(f.payload, visit->first);
            auto [it, added] = uses.emplace(
                key, Use{{f.payload, visit->first, false}, {f.payload, length - last, true}, false, false});
            it->second.read |= f.read;
            it->second.write |= f.write;
        }
        for (const auto& [key, use] : uses) {
            if (use.write) {
                choose(out.firstWriter, use.first, length, true);
                choose(out.lastWriter, use.last, length, false);
            }
        }
        for (const auto& [key, use] : uses) {
            if (!use.read || use.write) {
                continue;
            }
            auto pipe = quotient.payloads[use.first.type].pipe;
            if (!out.firstWriter || before(use.first, *out.firstWriter, length)) {
                choose(out.firstReaders, pipe, use.first, length, true);
            }
            if (!out.lastWriter || before(*out.lastWriter, use.last, length)) {
                choose(out.lastReaders, pipe, use.last, length, false);
            }
        }
    }
    std::set<std::pair<uint32_t, bool>> relationshipSites;
    for (const auto& [source, target] : uniformCrossings) {
        if (relationshipSites.emplace(source, true).second) {
            result.relationshipPorts.push_back({source, 1, true});
        }
        if (relationshipSites.emplace(target, false).second) {
            result.relationshipPorts.push_back({target, 0, false});
        }
    }
    indexBoundary(result, quotient);
    return result;
}
std::optional<std::vector<BoundaryDemand>> RotatingBoundaryCertificate::crossings(
    const RotatingBoundaryType& left, const RotatingBoundaryType& right) const
{
    if (!left.representative || !right.representative) {
        return std::vector<BoundaryDemand>{};
    }
    if (left.cells.size() != cells.size() || right.cells.size() != cells.size()) {
        return std::nullopt;
    }
    struct Edge {
        Occ a, b;
        PeriodicEventKind ak, bk;
        bool native;
    };
    std::vector<Edge> edges;
    std::set<std::tuple<uint32_t, uint64_t, uint32_t, uint64_t, unsigned, unsigned>> seen;
    auto add = [&](Occ a, Occ b, PeriodicEventKind ak, PeriodicEventKind bk, bool native) {
        if (!native &&
            storageProtection.protectsScalar(quotient.payloads[a.type].pipe, quotient.payloads[b.type].pipe)) {
            return;
        }
        auto key = std::make_tuple(
            a.type, a.at(left.representative), b.type, b.at(right.representative), unsigned(ak), unsigned(bk));
        if (seen.insert(key).second) {
            edges.push_back({a, b, ak, bk, native});
        }
    };
    const auto start = PeriodicEventKind::Start, finish = PeriodicEventKind::Completion;
    for (const auto& [pipe, last] : left.lastPayloads) {
        auto first = right.firstPayloads.find(pipe);
        if (first == right.firstPayloads.end()) {
            continue;
        }
        add(last, first->second, start, start, true);
        add(last, first->second, finish, finish, true);
    }
    for (const auto& [source, target] : uniformCrossings) {
        add({source, 1, true}, {target, 0, false}, finish, start, false);
    }
    for (std::size_t i = 0; i < cells.size(); ++i) {
        const auto& a = left.cells[i];
        const auto& b = right.cells[i];
        if (a.lastWriter) {
            if (b.firstWriter) {
                // Check this cell's writer witnesses before endpoint deduplication:
                // another cell may impose the same edge without protection.
                auto groups = [&](Occ occurrence, uint64_t length) {
                    std::vector<uint64_t> values;
                    for (const auto& fragment : fragments) {
                        if (!fragment.write || fragment.payload != occurrence.type ||
                            fragment.family != cells[i].family || fragment.atom != cells[i].atom) {
                            continue;
                        }
                        auto visits = progression(fragment, cells[i].slot);
                        const auto ordinal = occurrence.at(length);
                        if (visits && ordinal >= visits->first &&
                            (ordinal - visits->first) % visits->second == 0) {
                            values.push_back(fragment.protectionGroup);
                        }
                    }
                    return values;
                };
                const auto sources = groups(*a.lastWriter, left.representative);
                const auto targets = groups(*b.firstWriter, right.representative);
                bool protectedPair = !sources.empty() && !targets.empty();
                for (auto source : sources) {
                    for (auto target : targets) {
                        protectedPair &= (source & invocationProtectionBit) &&
                            hardwareProtectsConflict(quotient.payloads[a.lastWriter->type].pipe, source,
                                                     quotient.payloads[b.firstWriter->type].pipe, target);
                    }
                }
                if (!protectedPair) {
                    add(*a.lastWriter, *b.firstWriter, finish, start, false);
                }
            }
            for (const auto& [pipe, read] : b.firstReaders) {
                add(*a.lastWriter, read, finish, start, false);
            }
        }
        if (b.firstWriter) {
            for (const auto& [pipe, read] : a.lastReaders) {
                add(read, *b.firstWriter, finish, start, false);
            }
        }
    }
    std::vector<NumericalCrossing> indexed;
    for (const auto& edge : edges) {
        auto source = left.eventIds.find({edge.a.type, edge.a.at(left.representative), edge.ak});
        auto target = right.eventIds.find({edge.b.type, edge.b.at(right.representative), edge.bk});
        if (source == left.eventIds.end() || target == right.eventIds.end()) {
            return std::nullopt;
        }
        indexed.push_back({source->second, target->second});
    }
    auto retained = reduceNumericalCrossings(left.index, right.index, indexed);
    std::vector<BoundaryDemand> result;
    if (!retained) {
        return std::nullopt;
    }
    for (std::size_t i = 0; i < edges.size(); ++i) {
        if (!edges[i].native && (*retained)[i]) {
            result.push_back({edges[i].a, edges[i].b});
        }
    }
    return result;
}
RotatingBoundaryCertificate buildRotatingBoundaryCertificate(
    llvm::ArrayRef<PeriodicPayload> payloads, llvm::ArrayRef<RotatingFragment> fragments,
    llvm::ArrayRef<RotatingBoundaryCell> cells, llvm::ArrayRef<PeriodicRecord> prerequisites, uint64_t maximumTypes,
    StorageProtectionPolicy protection)
{
    for (const auto& fragment : fragments) {
        if (fragment.protectionGroup) {
            RotatingBoundaryCertificate failure;
            failure.error = "protected boundary needs a cross-visit protection adapter";
            return failure;
        }
    }
    auto extracted = extractRotatingGenerators(payloads, fragments, protection);
    if (!extracted.error.empty()) {
        RotatingBoundaryCertificate failure;
        failure.error = extracted.error;
        return failure;
    }
    extracted.generators.insert(extracted.generators.end(), prerequisites.begin(), prerequisites.end());
    return buildRotatingBoundaryCertificate(
        analyzePeriodicDemands(payloads, extracted.generators), fragments, cells, {}, maximumTypes, protection);
}
RotatingBoundaryCertificate buildRotatingBoundaryCertificate(
    PeriodicAnalysis quotient, llvm::ArrayRef<RotatingFragment> fragments,
    llvm::ArrayRef<RotatingBoundaryCell> cells,
    llvm::ArrayRef<std::pair<uint32_t, uint32_t>> uniformCrossings,
    uint64_t maximumTypes, StorageProtectionPolicy protection)
{
    RotatingBoundaryCertificate out;
    out.storageProtection = protection;
    out.quotient = std::move(quotient);
    const auto& payloads = out.quotient.payloads;
    auto fail = [&](const char* message) {
        out.error = message;
        out.types.clear();
        return out;
    };
    if (!out.quotient.error.empty() || payloads.empty()) {
        return fail("invalid rotating boundary input");
    }
    for (const auto& [source, target] : uniformCrossings) {
        if (source >= payloads.size() || target >= payloads.size()) {
            return fail("boundary relationship outside child namespace");
        }
    }
    out.uniformCrossings.assign(uniformCrossings.begin(), uniformCrossings.end());
    std::map<uint32_t, std::pair<uint64_t, uint64_t>> families;
    for (const auto& f : fragments) {
        if (f.payload >= payloads.size() || !f.slots || (!f.read && !f.write)) {
            return fail("boundary fragment outside child namespace");
        }
        const auto shape = std::make_pair(f.slots, f.stride % f.slots);
        auto family = families.emplace(f.family, shape);
        if (!family.second && family.first->second != shape) {
            return fail("inconsistent rotating boundary family slots or stride");
        }
        if (f.protectionGroup && !f.write) {
            return fail("rotating boundary protection requires writer effects");
        }
        const auto period = f.slots / std::gcd(f.slots, f.stride % f.slots);
        out.refresh = std::max(out.refresh, period);
        if (!periodic::multiply(out.period / std::gcd(out.period, period), period, out.period)) {
            return fail("boundary selector period overflow");
        }
        std::set<uint64_t> slots;
        for (const auto& cell : cells) {
            if (cell.family == f.family && cell.atom == f.atom) {
                if (cell.slot >= f.slots) {
                    return fail("boundary cell slot outside family");
                }
                slots.insert(cell.slot);
            }
        }
        // Require the explicitly declared complete family. Enumeration belongs
        // to the caller, so a large encoded bank count is never expanded here.
        if (slots.size() != f.slots) {
            return fail("boundary cell list does not cover family");
        }
    }
    uint64_t maximum = 0;
    for (const auto& row : out.quotient.frontiers) {
        for (const auto& distance : row.distances) {
            if (distance) {
                maximum = std::max(maximum, periodic::threshold(*distance, row.count, row.count));
            }
        }
    }
    if (!periodic::multiply(out.refresh, 2, out.cutoff) || !periodic::add(out.cutoff, maximum, out.cutoff) ||
        !periodic::add(out.cutoff, 1, out.cutoff)) {
        return fail("boundary cutoff overflow");
    }
    out.fragments.assign(fragments.begin(), fragments.end());
    out.cells.assign(cells.begin(), cells.end());
    if (out.period > maximumTypes || out.period > out.types.max_size()) {
        return fail("boundary residue expansion exceeds representation");
    }
    for (uint64_t residue = 0; residue < out.period; ++residue) {
        const auto current = out.cutoff % out.period;
        const auto shift = residue >= current ? residue - current : out.period - (current - residue);
        uint64_t representative;
        if (!periodic::add(out.cutoff, shift, representative)) {
            return fail("boundary representative overflow");
        }
        out.types.push_back(out.select(representative));
        if (!out.types.back().index.error.empty()) {
            return fail("boundary chain queries unavailable");
        }
    }
    return out;
}
AffineRotatingVisits analyzeAffineRotatingVisits(
    RotatingBoundaryCertificate child, uint64_t slope, uint64_t intercept, uint64_t maximumStartup)
{
    AffineRotatingVisits out;
    out.slope = slope;
    out.intercept = intercept;
    out.child = std::move(child);
    if (!out.child.error.empty()) {
        out.error = out.child.error;
        return out;
    }
    if (!slope) {
        out.error = "affine visits require a positive slope";
        return out;
    }
    auto delta = out.child.cutoff > intercept ? out.child.cutoff - intercept : 0;
    out.startup = delta / slope + (delta % slope != 0);
    out.period = out.child.period / std::gcd(slope, out.child.period);
    if (out.startup > maximumStartup) {
        out.error = "affine startup expansion exceeds representation";
        return out;
    }
    auto type = [&](uint64_t visit) -> std::optional<RotatingBoundaryType> {
        uint64_t length;
        if (!periodic::multiply(slope, visit, length) || !periodic::add(length, intercept, length)) {
            return std::nullopt;
        }
        return length < out.child.cutoff ? out.child.select(length) : out.child.types[length % out.child.period];
    };
    auto bridge = [&](uint64_t visit) -> std::optional<std::vector<BoundaryDemand>> {
        if (!visit) {
            return std::vector<BoundaryDemand>{};
        }
        auto a = type(visit - 1), b = type(visit);
        if (!a || !b) {
            return std::nullopt;
        }
        return out.child.crossings(*a, *b);
    };
    for (uint64_t visit = 0; visit < out.startup; ++visit) {
        auto result = bridge(visit);
        if (!result) {
            out.error = "affine startup length overflow";
            return out;
        }
        out.startupCrossings.push_back(std::move(*result));
    }
    auto seam = bridge(out.startup);
    if (!seam) {
        out.error = "affine seam length overflow";
        return out;
    }
    out.seam = std::move(*seam);
    for (uint64_t phase = 0; phase < out.period; ++phase) {
        uint64_t visit;
        if (!periodic::add(out.startup, out.period, visit) || !periodic::add(visit, phase, visit)) {
            out.error = "affine suffix visit overflow";
            return out;
        }
        auto result = bridge(visit);
        if (!result) {
            out.error = "affine suffix length overflow";
            return out;
        }
        out.suffixCrossings.push_back(std::move(*result));
    }
    return out;
}
const std::vector<BoundaryDemand>& AffineRotatingVisits::crossingInto(uint64_t visit) const
{
    static const std::vector<BoundaryDemand> empty;
    if (!error.empty() || !visit) {
        return empty;
    }
    if (visit < startup) {
        return startupCrossings[visit];
    }
    if (visit == startup) {
        return seam;
    }
    return suffixCrossings[(visit - startup) % period];
}
} // namespace mlir::pto::frontiersynch
