// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Shared numerical transfer construction; allocation never owns a second emitter.
#include "VaryingBoundaryQuery.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include <map>
#include <tuple>
namespace mlir::pto::frontiersynch {
RotatingBoundaryType selectVaryingBoundaryType(const AffineRotatingVisits& visits, uint64_t visit)
{
    const auto n = llvm::APInt(128, visits.slope) * llvm::APInt(128, visit) + llvm::APInt(128, visits.intercept);
    return n.ult(llvm::APInt(128, visits.child.cutoff)) ? visits.child.select(n.getZExtValue()) :
        visits.child.types[n.urem(llvm::APInt(128, visits.child.period)).getZExtValue()];
}
namespace {
using Matrix = VaryingBoundaryMatrix;
using Port = VaryingBoundaryPort;
using Occ = BoundaryOccurrence;
using PortKey = std::tuple<uint32_t, uint64_t, bool, unsigned>;
PortKey key(Port p) { return {p.occurrence.type, p.occurrence.coordinate, p.occurrence.tail, unsigned(p.kind)}; }
Matrix product(const Matrix& a, const Matrix& b)
{
    Matrix out(a.size(), std::vector<uint8_t>(a.size()));
    for (std::size_t i = 0; i < a.size(); ++i) {
        for (std::size_t k = 0; k < a.size(); ++k) {
            if (!a[i][k]) {
                continue;
            }
            for (std::size_t j = 0; j < a.size(); ++j) {
                out[i][j] |= b[k][j];
            }
        }
    }
    return out;
}
} // namespace
std::vector<std::vector<VaryingBoundaryMatrix>> buildVaryingBoundaryPowers(
    ArrayRef<Matrix> suffix, uint64_t& rotationProducts, uint64_t& squaringProducts)
{
    rotationProducts = squaringProducts = 0;
    if (suffix.empty()) { return {}; }
    Matrix identity(suffix.front().size(), std::vector<uint8_t>(suffix.front().size()));
    for (std::size_t i = 0; i < identity.size(); ++i) { identity[i][i] = 1; }
    std::vector<Matrix> prefixes{identity}, suffixes(suffix.size() + 1);
    for (const auto& matrix : suffix) {
        prefixes.push_back(product(prefixes.back(), matrix));
        ++rotationProducts;
    }
    suffixes.back() = std::move(identity);
    for (std::size_t phase = suffix.size(); phase > 0; --phase) {
        suffixes[phase - 1] = product(suffix[phase - 1], suffixes[phase]);
        ++rotationProducts;
    }
    std::vector<std::vector<Matrix>> powers;
    for (std::size_t phase = 0; phase < suffix.size(); ++phase) {
        // Order matters: S_phase ... S_last S_0 ... S_(phase-1).
        std::vector<Matrix> column{product(suffixes[phase], prefixes[phase])};
        ++rotationProducts;
        while (column.size() < 64) {
            auto next = product(column.back(), column.back());
            ++squaringProducts;
            if (next == column.back()) { break; }
            column.push_back(std::move(next));
        }
        powers.push_back(std::move(column));
    }
    return powers;
}
bool sameVaryingBoundaryQuotient(const PeriodicAnalysis& a, const PeriodicAnalysis& b)
{
    auto sameRecords = [](const auto& left, const auto& right) {
        return left.size() == right.size() && llvm::all_of(llvm::zip(left, right), [](const auto& pair) {
            const auto& x = std::get<0>(pair); const auto& y = std::get<1>(pair);
            return std::tie(x.source, x.target, x.displacement) == std::tie(y.source, y.target, y.displacement);
        });
    };
    if (!a.error.empty() || !b.error.empty() || a.payloads.size() != b.payloads.size() ||
        a.frontiers.size() != b.frontiers.size() || a.retained != b.retained ||
        a.sourceRows != b.sourceRows || a.localRanks != b.localRanks || a.pipeRows != b.pipeRows ||
        !sameRecords(a.generators, b.generators) || !sameRecords(a.nativePrerequisites, b.nativePrerequisites)) {
        return false;
    }
    for (std::size_t i = 0; i < a.payloads.size(); ++i) {
        if (a.payloads[i].pipe != b.payloads[i].pipe) { return false; }
    }
    for (std::size_t i = 0; i < a.frontiers.size(); ++i) {
        const auto& x = a.frontiers[i]; const auto& y = b.frontiers[i];
        if (x.pipe != y.pipe || x.count != y.count || x.distances != y.distances) { return false; }
    }
    return true;
}
namespace {
struct FrameBuilder {
    VaryingBoundaryFrame& frame;
    const AffineRotatingVisits& original;
    std::string& error;
    std::map<PortKey, std::size_t> portIds;
    const AffineRotatingVisits& visits() const { return frame.visits(); }
    RotatingBoundaryType type(uint64_t t) { return selectVaryingBoundaryType(visits(), t); }
    void add(Occ x)
    {
        for (auto kind : {PeriodicEventKind::Start, PeriodicEventKind::Completion}) {
            Port p{x, kind};
            if (frame.ports.size() >= UINT32_MAX) { error = "varying frame port identity overflow"; return; }
            if (portIds.emplace(key(p), frame.ports.size()).second) {
                frame.ports.push_back(p);
            }
        }
    }
    void collect(const RotatingBoundaryType& t)
    {
        for (auto occurrence : t.relationshipPorts) {
            add(occurrence);
        }
        for (const auto& cell : t.cells) {
            if (cell.firstWriter) {
                add(*cell.firstWriter);
            }
            if (cell.lastWriter) {
                add(*cell.lastWriter);
            }
            for (auto [pipe, x] : cell.firstReaders) {
                add(x);
            }
            for (auto [pipe, x] : cell.lastReaders) {
                add(x);
            }
        }
        for (auto [pipe, x] : t.firstPayloads) {
            add(x);
        }
        for (auto [pipe, x] : t.lastPayloads) {
            add(x);
        }
        for (const auto& [lane, selected] : t.lanes) {
            if (selected.firstPublication) { add(*selected.firstPublication); }
            if (selected.lastConsumption) { add(*selected.lastConsumption); }
        }
    }
    bool nativeReaches(Port a, Port b, uint64_t n)
    {
        if (!n || (a.occurrence.tail && a.occurrence.coordinate > n) ||
            (b.occurrence.tail && b.occurrence.coordinate > n) || a.occurrence.at(n) >= n || b.occurrence.at(n) >= n) {
            return false;
        }
        auto threshold = visits().child.quotient.eventThreshold(
            {a.occurrence.type, a.kind}, {b.occurrence.type, b.kind});
        if (threshold.error != PeriodicQueryError::None) {
            error = "varying child query unavailable";
            return false;
        }
        return threshold.displacement && a.occurrence.at(n) <= b.occurrence.at(n) &&
               b.occurrence.at(n) - a.occurrence.at(n) >= *threshold.displacement;
    }
    Matrix transfer(uint64_t target)
    {
        auto left = type(target - 1), right = type(target);
        Matrix crossing(frame.ports.size(), std::vector<uint8_t>(frame.ports.size()));
        std::vector<std::pair<Port, Port>> edges;
        for (auto d : original.crossingInto(target)) {
            edges.push_back({{d.source, PeriodicEventKind::Completion}, {d.target, PeriodicEventKind::Start}});
        }
        for (auto [pipe, last] : left.lastPayloads) {
            auto first = right.firstPayloads.find(pipe);
            if (first == right.firstPayloads.end()) {
                continue;
            }
            for (auto kind : {PeriodicEventKind::Start, PeriodicEventKind::Completion}) {
                edges.push_back({{last, kind}, {first->second, kind}});
            }
        }
        for (const auto& [a, b] : edges) {
            auto from = portIds.find(key(a)), to = portIds.find(key(b));
            if (from == portIds.end() || to == portIds.end()) {
                error = "varying crossing has no exported event port";
                return crossing;
            }
            crossing[from->second][to->second] = 1;
        }
        Matrix prefix(frame.ports.size(), std::vector<uint8_t>(frame.ports.size()));
        Matrix suffixPaths = prefix;
        for (std::size_t i = 0; i < frame.ports.size(); ++i) {
            for (std::size_t j = 0; j < frame.ports.size(); ++j) {
                prefix[i][j] = nativeReaches(frame.ports[i], frame.ports[j], left.representative);
                suffixPaths[i][j] = nativeReaches(frame.ports[i], frame.ports[j], right.representative);
            }
        }
        // Two Boolean products share paths among crossings: O(P^3), rather
        // than one P-by-P outer product for each of up to O(P^2) crossings.
        return product(product(prefix, crossing), suffixPaths);
    }
    bool build()
    {
        for (uint64_t t = 0; t <= visits().startup + visits().period; ++t) {
            collect(type(t));
            if (!error.empty()) { return false; }
        }
        if (!frame.ports.empty() &&
            frame.ports.size() > std::vector<uint8_t>().max_size() / frame.ports.size()) {
            error = "varying frame matrix size overflow"; return false;
        }
        for (uint64_t t = 1; t <= visits().startup; ++t) {
            frame.startup.push_back(transfer(t));
            if (!error.empty()) { return false; }
        }
        for (uint64_t phase = 0; phase < visits().period; ++phase) {
            frame.suffix.push_back(transfer(visits().startup + visits().period + phase + 1));
            if (!error.empty()) { return false; }
        }
        frame.powers = buildVaryingBoundaryPowers(frame.suffix, frame.rotationProducts, frame.squaringProducts);
        return error.empty();
    }
};
} // namespace
std::shared_ptr<const VaryingBoundaryFrame> VaryingBoundaryFrame::build(
    std::shared_ptr<const AffineRotatingVisits> original,
    std::shared_ptr<const AffineRotatingVisits> attachments, std::string& error)
{
    error.clear();
    const bool valid = original && attachments && original->error.empty() && attachments->error.empty() &&
        original->slope && original->period && attachments->period && attachments->child.period &&
        attachments->child.types.size() == attachments->child.period &&
        original->slope == attachments->slope && original->intercept == attachments->intercept &&
        attachments->startup >= original->startup && attachments->period % original->period == 0 &&
        original->startupCrossings.size() == original->startup &&
        original->suffixCrossings.size() == original->period &&
        sameVaryingBoundaryQuotient(original->child.quotient, attachments->child.quotient);
    if (!valid || attachments->period > (UINT64_MAX - 1) / 2 ||
        attachments->startup > UINT64_MAX - 2 * attachments->period - 1) {
        error = "varying proof frame does not refine the original mathematical domain"; return {};
    }
    // Private construction guarantees that consumers never see a null domain.
    auto frame = std::shared_ptr<VaryingBoundaryFrame>(new VaryingBoundaryFrame(std::move(attachments)));
    FrameBuilder builder{*frame, *original, error, {}};
    if (!builder.build()) { return {}; }
    return frame;
}
} // namespace mlir::pto::frontiersynch
