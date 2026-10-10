// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// A finite address schedule is not a loop unroll. It proves refresh over
// common physical bytes even when allocation families overlap or use different
// selector strides. Branch values remain independent at different visits.
#include "RecognitionInternal.h"
#include "PhysicalRefreshLimits.h"
#include "../InsertSync/SyncEffectRanges.h"
#include "../InsertSync/SyncScalarEvolution.h"
#include "CountedLoop.h"
#include "llvm/ADT/STLExtras.h"
#include <map>
#include <numeric>
#include <tuple>
namespace mlir::pto::frontiersynch::detail {
namespace {
// Match the existing numerical fragment representation limit. Exceeding it
// leaves a construction obligation, not an exclusion from bounded lifetimes.
constexpr uint64_t fragmentLimit = physicalRefreshFragmentLimit;
struct Projection {
    std::size_t access;
    uint64_t phase;
    SyncStorageCell range;
};
std::optional<SyncStorageCell> physicalRange(const RotatingAccess& access, uint64_t slot, const SyncInput& input)
{
    if (!access.atom || access.effect >= input.accesses().effects().size()) {
        return std::nullopt;
    }
    const auto& effect = input.accesses().effects()[access.effect];
    if (!effect.memory || effect.memory->scope == AddressSpace::GM || effect.memory->scope == AddressSpace::Zero) {
        return std::nullopt;
    }
    SyncStorageCell bank;
    if (access.firstPhysicalSlot) {
        bank = *access.firstPhysicalSlot;
        auto displacement = APInt(128, slot) * APInt(128, access.physicalSlotStride);
        auto begin = APInt(128, bank.begin) + displacement;
        auto end = APInt(128, bank.end) + displacement;
        if (!begin.isIntN(64) || !end.isIntN(64)) {
            return std::nullopt;
        }
        bank.begin = begin.getZExtValue();
        bank.end = end.getZExtValue();
    } else {
        // The shared descriptor preserves slot order; sorting here would change
        // the meaning of an explicitly addressed multi-buffer selector.
        auto slots = mlir::pto::detail::physicalSlotRanges(input, *effect.memory);
        if (slots.size() != access.slots || slot >= slots.size()) {
            return std::nullopt;
        }
        bank = slots[slot];
    }
    if (bank.base || bank.end < bank.begin || access.atom->first >= access.atom->second ||
        access.atom->second > bank.end - bank.begin) {
        return std::nullopt;
    }
    return SyncStorageCell{bank.space, bank.begin + access.atom->first, bank.begin + access.atom->second};
}
// These facts apply only when ordinal+tail<trips. They certify refresh;
// they never replace a guard in the demand circuits or occurrence domains.
struct InteriorFacts {
    DenseMap<Value, bool> values;
    uint64_t tail = 0;
};
InteriorFacts interiorFacts(const GuardedRecognition& skeleton, scf::ForOp loop)
{
    InteriorFacts result;
    auto domain = CountedLoop::get(loop);
    if (!domain) {
        return result;
    }
    mlir::pto::detail::ScalarEvolution evolution(loop.getContext(), loop);
    const auto iv = getAffineDimExpr(0, loop.getContext());
    for (const auto& guard : skeleton.guards) {
        if (result.values.count(guard.condition)) {
            continue;
        }
        auto compare = guard.condition.getDefiningOp<arith::CmpIOp>();
        if (!compare) {
            continue;
        }
        auto predicate = compare.getPredicate();
        Value candidate;
        if (compare.getRhs() == loop.getUpperBound()) {
            candidate = compare.getLhs();
        } else if (compare.getLhs() == loop.getUpperBound()) {
            candidate = compare.getRhs();
            switch (predicate) {
                case arith::CmpIPredicate::sgt:
                    predicate = arith::CmpIPredicate::slt;
                    break;
                case arith::CmpIPredicate::sle:
                    predicate = arith::CmpIPredicate::sge;
                    break;
                case arith::CmpIPredicate::ugt:
                    predicate = arith::CmpIPredicate::ult;
                    break;
                case arith::CmpIPredicate::ule:
                    predicate = arith::CmpIPredicate::uge;
                    break;
                default:
                    continue;
            }
        } else {
            continue;
        }
        const bool positive = predicate == arith::CmpIPredicate::slt || predicate == arith::CmpIPredicate::ult;
        const bool negative = predicate == arith::CmpIPredicate::sge || predicate == arith::CmpIPredicate::uge;
        if (!positive && !negative) {
            continue;
        }
        auto expression = evolution.value(
            candidate, [&](Value value) -> AffineExpr { return value == loop.getInductionVar() ? iv : AffineExpr{}; });
        int64_t delta = 0;
        if (expression != iv) {
            auto add = dyn_cast_or_null<AffineBinaryOpExpr>(expression);
            auto constant = add ? dyn_cast<AffineConstantExpr>(add.getRHS()) : AffineConstantExpr{};
            if (!add || add.getKind() != AffineExprKind::Add || add.getLHS() != iv || !constant ||
                constant.getValue() < 0) {
                continue;
            }
            delta = constant.getValue();
        }
        // Admission has proved each machine intermediate nonwrapping. The
        // original rotating-domain lower is nonnegative, so executed IV and
        // upper are nonnegative and unsigned comparison has the same meaning.
        const auto stride = static_cast<uint64_t>(domain->step);
        const auto distance = static_cast<uint64_t>(delta);
        const auto tail = distance / stride + (distance % stride != 0);
        result.values[guard.condition] = positive;
        result.tail = std::max(result.tail, tail);
    }
    return result;
}
class PhysicalRefreshBuilder {
public:
    PhysicalRefreshBuilder(const GuardedRecognition& skeleton, const SyncInput& input)
        : skeleton(skeleton), input(input)
    {}
    std::optional<PhysicalRefreshDescription> build(scf::ForOp loop, RefreshCertificate& certificate)
    {
        if (!period() || !project() || !partition() || !refresh(loop, certificate)) {
            certificate.error = error;
            return std::nullopt;
        }
        return std::move(result);
    }

private:
    const GuardedRecognition& skeleton;
    const SyncInput& input;
    PhysicalRefreshDescription result;
    std::vector<Projection> projections;
    std::map<AddressSpace, SmallVector<uint64_t>> endpoints;
    std::map<std::tuple<AddressSpace, uint64_t, uint64_t>, uint32_t> atoms;
    std::map<std::pair<uint64_t, uint32_t>, SmallVector<std::size_t>> writers;
    std::vector<uint8_t> writable;
    std::string error;
    bool fail(StringRef message)
    {
        error = message.str();
        return false;
    }
    bool period()
    {
        for (const auto& access : skeleton.result.accesses) {
            if (!access.slots || access.parameterOffset || !access.atom) {
                return fail("physical refresh requires complete numerical address maps");
            }
            const auto period = access.slots / std::gcd(access.stride, access.slots);
            const auto factor = period / std::gcd(result.period, period);
            if (factor > fragmentLimit / result.period) {
                return fail("physical address-period construction exceeds numerical fragment representation");
            }
            result.period *= factor;
        }
        if (skeleton.result.accesses.size() > fragmentLimit / result.period) {
            return fail("physical address projection exceeds numerical fragment representation");
        }
        return true;
    }
    bool project()
    {
        for (uint64_t phase = 0; phase < result.period; ++phase) {
            for (auto [id, access] : llvm::enumerate(skeleton.result.accesses)) {
                auto slot = (APInt(128, phase) * APInt(128, access.stride) + APInt(128, access.offset))
                                .urem(APInt(128, access.slots))
                                .getZExtValue();
                auto range = physicalRange(access, slot, input);
                if (!range) {
                    return fail("physical refresh projection lacks an exact absolute local range");
                }
                endpoints[range->space].push_back(range->begin);
                endpoints[range->space].push_back(range->end);
                projections.push_back({id, phase, *range});
            }
        }
        return true;
    }
    bool partition()
    {
        DenseMap<const CompoundInstanceElement*, std::size_t> positions;
        for (auto [id, site] : llvm::enumerate(skeleton.phases)) {
            positions[site.phase] = id;
        }
        for (auto& [space, points] : endpoints) {
            llvm::sort(points);
            points.erase(std::unique(points.begin(), points.end()), points.end());
        }
        for (const auto& projection : projections) {
            const auto& access = skeleton.result.accesses[projection.access];
            auto found = positions.find(input.accesses().effects()[access.effect].phase);
            if (found == positions.end()) {
                return fail("physical refresh access has no potential payload");
            }
            const auto& points = endpoints[projection.range.space];
            auto first = llvm::lower_bound(points, projection.range.begin);
            auto last = llvm::lower_bound(points, projection.range.end);
            for (auto it = first; it != last; ++it) {
                if (result.accesses.size() == fragmentLimit) {
                    return fail("physical atom projection exceeds numerical fragment representation");
                }
                auto [atom, inserted] =
                    atoms.emplace(std::make_tuple(projection.range.space, *it, *std::next(it)), atoms.size());
                if (inserted) {
                    result.atoms.push_back({projection.range.space, *it, *std::next(it)});
                    writable.push_back(0);
                }
                const auto cell = atom->second;
                result.accesses.push_back({projection.access, projection.phase, cell});
                if (access.writes) {
                    writable[cell] = 1;
                    writers[{projection.phase, cell}].push_back(found->second);
                }
            }
        }
        return true;
    }
    bool refresh(scf::ForOp loop, RefreshCertificate& certificate)
    {
        std::vector<uint8_t> refreshed(result.atoms.size());
        for (const auto& [key, payloads] : writers) {
            if (coversEveryVisit(skeleton, payloads)) {
                refreshed[key.second] = 1;
            }
        }
        bool incomplete = false;
        for (std::size_t cell = 0; cell < writable.size(); ++cell) {
            incomplete |= writable[cell] && !refreshed[cell];
        }
        uint64_t tail = 0;
        if (incomplete) {
            auto facts = interiorFacts(skeleton, loop);
            tail = facts.tail;
            for (const auto& [key, payloads] : writers) {
                if (coversEveryVisitUnderFacts(skeleton, payloads, facts.values)) {
                    refreshed[key.second] = 1;
                }
            }
            for (std::size_t cell = 0; cell < writable.size(); ++cell) {
                if (writable[cell] && !refreshed[cell]) {
                    const auto& atom = result.atoms[cell];
                    return fail(
                        "a writable physical atom [" + std::to_string(atom.begin) + "," + std::to_string(atom.end) +
                        ") in space " + std::to_string(static_cast<unsigned>(atom.space)) +
                        " has no collectively complete refresh within the address period");
                }
            }
        }
        if (tail > fragmentLimit - result.period) {
            return fail("physical refresh exit-suffix window exceeds numerical fragment representation");
        }
        certificate.span = result.period + tail;
        if (!physicalRefreshWindowFits(
                certificate.span, skeleton.phases.size(), skeleton.guards.size(), result.accesses.size())) {
            return fail("physical refresh window exceeds numerical fragment representation");
        }
        // Address modulo stays P; the inclusive reducer window has
        // P+tail+1 visits, with original guards and no imaginary exit writer.
        return true;
    }
};
} // namespace
std::optional<PhysicalRefreshDescription> certifyPhysicalRefresh(
    const GuardedRecognition& skeleton, const SyncInput& input, scf::ForOp loop, RefreshCertificate& certificate)
{
    return PhysicalRefreshBuilder(skeleton, input).build(loop, certificate);
}
} // namespace mlir::pto::frontiersynch::detail
