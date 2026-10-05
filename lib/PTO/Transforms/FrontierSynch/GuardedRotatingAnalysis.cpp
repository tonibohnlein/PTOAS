// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/GuardedRotatingAnalysis.h"
#include "PTO/Transforms/FrontierSynch/HardwareProtection.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/ADT/APInt.h"
#include <map>
#include <numeric>
#include <tuple>
namespace mlir::pto::frontiersynch {
namespace {
using Expr = RegionExpressions::Id;
uint64_t inverse(uint64_t value, uint64_t modulus)
{
    uint64_t oldRemainder = modulus, remainder = value;
    llvm::APInt oldCoefficient(256, 0), coefficient(256, 1);
    while (remainder) {
        const auto quotient = oldRemainder / remainder;
        const auto nextRemainder = oldRemainder % remainder;
        const auto nextCoefficient = oldCoefficient - llvm::APInt(256, quotient) * coefficient;
        oldRemainder = remainder;
        remainder = nextRemainder;
        oldCoefficient = coefficient;
        coefficient = nextCoefficient;
    }
    auto result = oldCoefficient.srem(llvm::APInt(256, modulus));
    if (result.isNegative()) { result += llvm::APInt(256, modulus); }
    return result.getZExtValue();
}
struct Fragment {
    uint32_t payload = 0, family = 0, atom = 0;
    uint64_t slots = 0, divisor = 0, refresh = 0, inverseStride = 0;
    Expr offset = RegionExpressions::invalid, active = RegionExpressions::invalid;
    Expr read = RegionExpressions::invalid, write = RegionExpressions::invalid;
    bool reads = false, writes = false;
};
class Extractor {
public:
    GuardedRotatingAnalysis result;
    Extractor(scf::ForOp loop, const SyncInput& input, const GuardedRecognition& recognized)
        : input(input), recognized(recognized)
    {
        result.loop = loop;
        result.expressions = std::make_shared<RegionExpressions>();
    }
    void run()
    {
        if (!result.loop || recognized.result.state != RecognitionState::Applicable ||
            recognized.phases.size() > UINT32_MAX / 2 || recognized.result.accesses.size() > UINT32_MAX) {
            fail("guarded rotation requires a certified bounded potential skeleton"); return;
        }
        const auto bits = DataLayout::closest(result.loop).getTypeSizeInBits(IndexType::get(result.loop.getContext()));
        if (bits.isScalable() || bits.getFixedValue() != 64) {
            fail("guarded rotation requires a 64-bit index representation"); return;
        }
        if (!prepareGuards() || !prepareFragments()) { return; }
        if (hasPotentialProtection()) { fail("guarded conditional accumulator protection is not supported"); return; }
        normalize();
        for (std::size_t target = 0; target < fragments.size(); ++target) {
            neighbors(target, false);
            neighbors(target, true);
        }
        if (!dag().constructionError().empty()) { fail(dag().constructionError()); return; }
        result.periodic = analyzeGuardedPeriodicQuotient(result.expressions, result.payloads, result.generators);
        if (!result.periodic.error.empty()) { fail(result.periodic.error); }
    }
private:
    const SyncInput& input;
    const GuardedRecognition& recognized;
    std::vector<Expr> guards;
    std::vector<Fragment> fragments;
    RegionExpressions& dag() { return *result.expressions; }
    Expr c(uint64_t value) { return dag().constant(value); }
    Expr yes() { return dag().boolean(true); }
    Expr no() { return dag().boolean(false); }
    void fail(const std::string& error) { result.error = error; result.generators.clear(); }
    Expr modAdd(Expr a, Expr b, uint64_t modulus)
    {
        // Both operands are residues and modulus <= INT64_MAX: the sum fits.
        return dag().rem(dag().add(a, b), c(modulus));
    }
    Expr modMultiply(Expr value, uint64_t factor, uint64_t modulus)
    {
        Expr result = c(0);
        factor %= modulus;
        while (factor) {
            if (factor & 1) { result = modAdd(result, value, modulus); }
            factor >>= 1;
            if (factor) { value = modAdd(value, value, modulus); }
        }
        return result;
    }
    Expr signedRemainder(Expr value, uint64_t modulus)
    {
        auto negative = dag().slt(value, c(0));
        auto magnitude = dag().select(negative, dag().sub(c(0), value), value);
        auto remainder = dag().rem(magnitude, c(modulus));
        auto negated = dag().select(dag().eq(remainder, c(0)), c(0), dag().sub(c(modulus), remainder));
        return dag().select(negative, negated, remainder);
    }
    Expr offset(AffineExpr expression, ArrayRef<Value> parameters, uint64_t modulus)
    {
        if (!expression || !modulus || modulus > INT64_MAX) { return RegionExpressions::invalid; }
        if (auto constant = dyn_cast<AffineConstantExpr>(expression)) {
            auto residue = constant.getValue() % static_cast<int64_t>(modulus);
            return c(residue < 0 ? residue + static_cast<int64_t>(modulus) : residue);
        }
        if (auto symbol = dyn_cast<AffineSymbolExpr>(expression)) {
            if (symbol.getPosition() >= parameters.size() ||
                !isa<IndexType>(parameters[symbol.getPosition()].getType())) {
                return RegionExpressions::invalid;
            }
            return signedRemainder(dag().input(parameters[symbol.getPosition()]), modulus);
        }
        auto binary = dyn_cast<AffineBinaryOpExpr>(expression);
        if (!binary) { return RegionExpressions::invalid; }
        if (expression.getKind() == AffineExprKind::Add) {
            auto a = offset(binary.getLHS(), parameters, modulus), b = offset(binary.getRHS(), parameters, modulus);
            if (a == RegionExpressions::invalid || b == RegionExpressions::invalid) {
                return RegionExpressions::invalid;
            }
            return modAdd(a, b, modulus);
        }
        auto number = dyn_cast<AffineConstantExpr>(binary.getRHS());
        if (!number) { return RegionExpressions::invalid; }
        const auto coefficient = number.getValue();
        if (expression.getKind() == AffineExprKind::Mul) {
            auto value = offset(binary.getLHS(), parameters, modulus);
            if (value == RegionExpressions::invalid) { return value; }
            auto residue = coefficient % static_cast<int64_t>(modulus);
            return modMultiply(value, residue < 0 ? residue + static_cast<int64_t>(modulus) : residue, modulus);
        }
        if (coefficient <= 0) { return RegionExpressions::invalid; }
        const auto divisor = static_cast<uint64_t>(coefficient);
        if (expression.getKind() == AffineExprKind::Mod) {
            auto value = offset(binary.getLHS(), parameters, divisor);
            return value == RegionExpressions::invalid ? value : dag().rem(value, c(modulus));
        }
        if ((expression.getKind() != AffineExprKind::FloorDiv && expression.getKind() != AffineExprKind::CeilDiv) ||
            divisor > static_cast<uint64_t>(INT64_MAX) / modulus) { return RegionExpressions::invalid; }
        auto value = offset(binary.getLHS(), parameters, divisor * modulus);
        if (value == RegionExpressions::invalid) { return value; }
        auto quotient = dag().div(value, c(divisor));
        if (expression.getKind() == AffineExprKind::CeilDiv) {
            quotient = dag().add(quotient, dag().select(dag().eq(dag().rem(value, c(divisor)), c(0)), c(0), c(1)));
        }
        return dag().rem(quotient, c(modulus));
    }
    bool prepareGuards()
    {
        for (const auto& guard : recognized.guards) {
            if (!guard.condition || !guard.condition.getType().isInteger(1) ||
                (guard.parent && *guard.parent >= guards.size())) {
                fail("invalid guarded rotating presence predicate"); return false;
            }
            auto value = dag().input(guard.condition);
            if (!guard.takeThen) { value = dag().lnot(value); }
            // Presence follows the enclosing branch. Emission still requires
            // all predicate inputs to be available or safely replayable.
            guards.push_back(guard.parent ? dag().select(guards[*guard.parent], value, no()) : value);
        }
        for (const auto& item : recognized.phases) {
            if (!item.phase || !item.phase->elementOp || (item.guard && *item.guard >= guards.size())) {
                fail("invalid guarded rotating phase"); return false;
            }
            result.phases.push_back(item.phase);
            result.payloads.push_back({static_cast<uint32_t>(item.phase->kPipeValue),
                                       item.guard ? guards[*item.guard] : yes()});
            dag().forbidRecomputation(item.phase->elementOp);
        }
        return true;
    }
    bool prepareFragments()
    {
        DenseMap<const CompoundInstanceElement*, uint32_t> positions;
        DenseMap<Value, uint32_t> families;
        struct Family { uint64_t slots, stride, divisor, refresh, inverseStride; };
        std::map<uint32_t, Family> descriptions;
        std::map<std::tuple<uint32_t, uint64_t, uint64_t>, uint32_t> atoms;
        for (uint32_t i = 0; i < result.phases.size(); ++i) { positions[result.phases[i]] = i; }
        for (const auto& access : recognized.result.accesses) {
            if (!access.atom || !access.slots || access.slots > INT64_MAX ||
                access.effect >= input.accesses().effects().size() ||
                (access.guard && *access.guard >= guards.size())) {
                fail("guarded rotation requires exact fragments and representable constant slot counts"); return false;
            }
            const auto& effect = input.accesses().effects()[access.effect];
            if (!effect.memory) { fail("guarded rotating effect has no storage identity"); return false; }
            auto position = positions.find(effect.phase);
            if (position == positions.end()) { fail("guarded fragment has no potential payload"); return false; }
            const auto family = families.try_emplace(access.family, families.size()).first->second;
            const auto stride = access.stride % access.slots;
            auto description = descriptions.find(family);
            if (description == descriptions.end()) {
                const auto divisor = std::gcd(stride, access.slots);
                const auto refresh = access.slots / divisor;
                description = descriptions.emplace(family, Family{access.slots, stride, divisor, refresh,
                    refresh == 1 ? 0 : inverse(stride / divisor, refresh)}).first;
            }
            if (description->second.slots != access.slots || description->second.stride != stride) {
                fail("guarded rotating family requires one common stride and modulus"); return false;
            }
            auto atom = atoms.emplace(std::make_tuple(family, access.atom->first, access.atom->second),
                                      atoms.size()).first->second;
            Fragment fragment;
            fragment.payload = position->second; fragment.family = family; fragment.atom = atom;
            fragment.slots = access.slots; fragment.divisor = description->second.divisor;
            fragment.refresh = description->second.refresh;
            fragment.inverseStride = description->second.inverseStride;
            fragment.offset = access.parameterOffset ? offset(access.parameterOffset, access.parameters, access.slots) :
                                                       c(access.offset % access.slots);
            if (fragment.offset == RegionExpressions::invalid) {
                fail("guarded rotating offset has unsupported or unrepresentable exact modular arithmetic");
                return false;
            }
            fragment.active = result.payloads[fragment.payload].presence;
            if (access.guard) { fragment.active = dag().land(fragment.active, guards[*access.guard]); }
            // Replayed scalar arithmetic may produce poison on an inactive
            // original arm (for example, an overflow-flagged addition). Mask
            // that offset before equality and neighbor circuits consume it;
            // Boolean AND with an inactive guard would still propagate poison.
            fragment.offset = dag().select(fragment.active, fragment.offset, c(0));
            fragment.reads = access.reads; fragment.writes = access.writes;
            fragments.push_back(fragment);
            result.refreshBound = std::max(result.refreshBound, fragment.refresh);
        }
        return true;
    }
    bool hasPotentialProtection()
    {
        std::vector<ExplicitEffects> candidates(result.phases.size());
        for (uint32_t i = 0; i < result.phases.size(); ++i) {
            auto& candidate = candidates[i];
            candidate.payload = i; candidate.pipe = result.payloads[i].pipe;
            bool read = false, write = false;
            for (auto id : input.accesses().effectsFor(result.phases[i])) {
                const auto& effect = input.accesses().effects()[id];
                if (effect.memory && effect.memory->scope == AddressSpace::ACC) {
                    read |= effect.mode == SyncAccessMode::Read;
                    write |= effect.mode == SyncAccessMode::Write;
                }
            }
            if (write) { candidate.accesses.push_back({0, read, true}); }
        }
        // Test potential writer pairs through the common target rule. Collapsing
        // accumulator atoms and skipping intermediate sites overapproximates
        // possible protection; rejecting it is conservative. In particular an
        // inactive intervening site cannot hide a newly protected pair. Check
        // both orders and repeated sites, since pairs may cross iteration ends.
        for (std::size_t a = 0; a < candidates.size(); ++a) {
            if (candidates[a].accesses.empty()) { continue; }
            for (std::size_t b = 0; b < candidates.size(); ++b) {
                if (candidates[b].accesses.empty()) { continue; }
                auto first = candidates[a], second = candidates[b];
                second.payload = static_cast<uint32_t>(candidates.size() + b);
                HardwareProtectionBuilder protection;
                protection.observe(result.phases[a]->elementOp, first, {0});
                protection.observe(result.phases[b]->elementOp, second, {0});
                if (hardwareProtectsConflict(first.pipe, first.accesses.front().protectionGroup,
                                            second.pipe, second.accesses.front().protectionGroup)) {
                    return true;
                }
            }
        }
        return false;
    }
    bool sameCellFamily(const Fragment& a, const Fragment& b) const
    {
        return a.family == b.family && a.atom == b.atom;
    }
    void normalize()
    {
        for (std::size_t i = 0; i < fragments.size(); ++i) {
            auto& fragment = fragments[i];
            auto earlier = no(), reads = no(), writes = no();
            for (std::size_t j = 0; j < fragments.size(); ++j) {
                const auto& other = fragments[j];
                if (fragment.payload != other.payload || !sameCellFamily(fragment, other)) { continue; }
                auto matches = dag().land(other.active, dag().eq(fragment.offset, other.offset));
                if (j < i) { earlier = dag().lor(earlier, matches); }
                if (other.reads) { reads = dag().lor(reads, matches); }
                if (other.writes) { writes = dag().lor(writes, matches); }
            }
            auto representative = dag().land(fragment.active, dag().lnot(earlier));
            fragment.read = dag().land(representative, reads);
            fragment.write = dag().land(representative, writes);
        }
    }
    Expr selfRefreshWitness(std::size_t targetId)
    {
        const auto& target = fragments[targetId];
        auto witness = no();
        for (const auto& reader : fragments) {
            if (reader.payload == target.payload || !sameCellFamily(reader, target)) { continue; }
            auto readOnly = dag().land(reader.read, dag().lnot(reader.write));
            auto sameOrbit = target.divisor == 1 ? yes() : dag().eq(
                dag().rem(reader.offset, c(target.divisor)), dag().rem(target.offset, c(target.divisor)));
            witness = dag().lor(witness, dag().land(readOnly, sameOrbit));
        }
        return witness;
    }
    void neighbors(std::size_t targetId, bool following)
    {
        const auto& target = fragments[targetId];
        struct Candidate { uint32_t writer; Expr distance, active; };
        std::vector<Candidate> candidates;
        auto bestPresent = no(), bestDistance = c(0), bestPosition = c(0), bestId = c(0);
        for (uint32_t id = 0; id < fragments.size(); ++id) {
            const auto& writer = fragments[id];
            if (!sameCellFamily(target, writer)) { continue; }
            auto difference = following ? dag().sub(dag().add(target.offset, c(target.slots)), writer.offset) :
                                          dag().sub(dag().add(writer.offset, c(target.slots)), target.offset);
            difference = dag().rem(difference, c(target.slots));
            auto compatible = id == targetId ? yes() : dag().eq(dag().rem(difference, c(target.divisor)), c(0));
            auto distance = id == targetId ? c(0) :
                modMultiply(dag().div(difference, c(target.divisor)), target.inverseStride, target.refresh);
            const bool zeroForbidden = following ? writer.payload <= target.payload : writer.payload >= target.payload;
            if (zeroForbidden) { distance = dag().select(dag().eq(distance, c(0)), c(target.refresh), distance); }
            auto active = dag().land(writer.write, compatible);
            auto betterPosition = following ? dag().lt(c(writer.payload), bestPosition) :
                                              dag().lt(bestPosition, c(writer.payload));
            auto better = dag().lor(dag().lnot(bestPresent), dag().lor(dag().lt(distance, bestDistance),
                dag().land(dag().eq(distance, bestDistance), betterPosition)));
            better = dag().land(active, better);
            bestDistance = dag().select(better, distance, bestDistance);
            bestPosition = dag().select(better, c(writer.payload), bestPosition);
            bestId = dag().select(better, c(id), bestId);
            bestPresent = dag().lor(bestPresent, active);
            candidates.push_back({id, distance, active});
        }
        auto targetActive = following ? dag().land(target.read, dag().lnot(target.write)) :
                                       dag().lor(target.read, target.write);
        for (const auto& candidate : candidates) {
            auto active = dag().land(candidate.active,
                dag().land(targetActive, dag().land(bestPresent, dag().eq(bestId, c(candidate.writer)))));
            // A self predecessor at refresh R certifies that no compatible
            // writer intervenes. A guaranteed read-only payload in this orbit
            // occurs strictly inside those two writer visits. Its RAW and WAR
            // records therefore form an alternative path, for every offset.
            // Both records connect distinct sites, so deleting self records
            // simultaneously cannot invalidate their replacement witnesses.
            if (!following && candidate.writer == targetId) {
                active = dag().land(active, dag().lnot(selfRefreshWitness(targetId)));
            }
            const auto source = following ? target.payload : fragments[candidate.writer].payload;
            const auto destination = following ? fragments[candidate.writer].payload : target.payload;
            if (dag().constantValue(active) == std::optional<uint64_t>(0)) { continue; }
            result.generators.push_back({source, destination, candidate.distance, active, target.refresh});
        }
    }
};
} // namespace
GuardedRotatingAnalysis analyzeGuardedRotating(scf::ForOp loop, const SyncInput& input,
                                               const GuardedRecognition& recognition)
{
    Extractor extractor(loop, input, recognition);
    extractor.run();
    return std::move(extractor.result);
}
} // namespace mlir::pto::frontiersynch
