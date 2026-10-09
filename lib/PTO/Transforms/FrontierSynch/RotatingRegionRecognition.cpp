// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// q=1 syntax check over shared effects. No instruction-specific footprint rules.
#include "PTO/Transforms/FrontierSynch/RotatingRegion.h"
#include "PTO/Transforms/FrontierSynch/ArithmeticRegional.h"
#include "PTO/Transforms/FrontierSynch/NumericTemplateRegional.h"
#include "PTO/Transforms/FrontierSynch/SequenceAnalysis.h"
#include "PhaseNormalization.h"
#include "CountedLoop.h"
#include "RecognitionInternal.h"
#include "../InsertSync/SyncEffectRanges.h"
#include "mlir/IR/Matchers.h"
namespace mlir::pto::frontiersynch {
namespace {
using Id = RegionExpressions::Id;
std::optional<uint64_t> positiveConstant(Value value, bool zero = false)
{
    APInt number;
    if (!matchPattern(value, m_ConstantInt(&number)) || !number.isSignedIntN(64) ||
        number.isNegative() || (!zero && number.isZero())) { return std::nullopt; }
    return number.getZExtValue();
}
class Producer {
public:
    Producer(func::FuncOp function, const SyncInput& input, const ProgramRecognition& program,
             std::size_t node, const PhaseIndex& index, std::shared_ptr<RegionExpressions> arena)
        : function(function), input(input), program(program), node(node), index(index), arena(std::move(arena)),
          loop(dyn_cast_or_null<scf::ForOp>(program.nodes[node].anchor)),
          normalizer(loop, index, *this->arena) {}
    RepeatedRegionAnalysis run()
    {
        if (!loop || program.nodes[node].children.size() != 1 || index.hasRelevantCarriedState(loop)) {
            return fail("rotating compact child requires one original body and no evolving carried state");
        }
        auto domain = CountedLoop::get(loop);
        auto lower = positiveConstant(loop.getLowerBound(), true);
        if (!domain || !lower) { return fail("rotating bank ordinal origin is not numerical"); }
        origin = *lower;
        bool invariant = true, nested = false;
        loop.getBody()->walk([&](Operation* operation) {
            if (auto inner = dyn_cast<scf::ForOp>(operation)) {
                nested = true;
                invariant &= normalizer.independent(inner.getLowerBound()) &&
                    normalizer.independent(inner.getUpperBound()) && normalizer.independent(inner.getStep());
            } else if (auto branch = dyn_cast<scf::IfOp>(operation)) {
                invariant &= normalizer.independent(branch.getCondition());
            }
        });
        if (!nested || !invariant) {
            return fail("rotating compact child control and nested domains are not invariant after bank renaming");
        }
        SmallVector<const CompoundInstanceElement*> phases;
        for (auto* phase : input.instructions()) {
            if (phase->elementOp && loop->isProperAncestor(phase->elementOp)) { phases.push_back(phase); }
        }
        RecognitionResult recognized;
        detail::inspectRotatingPhases(loop, phases, input, input.accesses(), recognized, index);
        if (recognized.state != RecognitionState::Applicable) {
            return fail("rotating compact child bank geometry or numerical slot formula is unproved");
        }
        bool rotating = false;
        for (const auto& access : recognized.accesses) {
            if (!access.atom || access.parameterOffset) {
                return fail("rotating compact child requires numerical local footprint cells");
            }
            auto found = familyIndices.find(access.family);
            if (found == familyIndices.end()) {
                auto geometry = family(access);
                if (!geometry) { return fail(error); }
                found = familyIndices.try_emplace(access.family, specification.families.size()).first;
                specification.families.push_back(std::move(*geometry));
            }
            auto& target = specification.families[found->second];
            if (target.banks != access.slots || target.stride != access.stride || target.offset != access.offset) {
                return fail("one rotating family accesses different banks within the same visit");
            }
            rotating |= target.banks > 1;
            for (auto effect : access.effects) {
                if (!llvm::is_contained(target.effects, effect)) { target.effects.push_back(effect); }
                const auto& record = input.accesses().effects()[effect];
                if (record.selection) { geometryValues[record.selection->selector] = target.offset; }
            }
        }
        if (!rotating) { return fail("no rotating compact-child family"); }
        // Discharge inside a child is not an inter-visit independence proof.
        // Retain these original IDs and let shared owner/read-only geometry
        // validate their maps against all banks before omitting any crossings.
        specification.residualEffects.assign(recognized.dischargedEffects.begin(), recognized.dischargedEffects.end());
        auto child = canonical(program.nodes[node].children.front());
        if (failed(child)) { return fail(error); }
        specification.child = std::move(*child);
        return repeatRotatingRegion(function, loop, std::move(specification), domain->trips(*arena));
    }
private:
    func::FuncOp function;
    const SyncInput& input;
    const ProgramRecognition& program;
    std::size_t node;
    const PhaseIndex& index;
    std::shared_ptr<RegionExpressions> arena;
    scf::ForOp loop;
    PhaseNormalization normalizer;
    uint64_t origin = 0;
    DenseMap<Value, uint64_t> geometryValues;
    DenseMap<Value, std::size_t> familyIndices;
    RotatingRegionInput specification;
    std::string error;
    RepeatedRegionAnalysis fail(StringRef reason)
    {
        RepeatedRegionAnalysis result; result.error = reason.str(); return result;
    }
    std::optional<int64_t> geometry(Value value)
    {
        if (auto found = geometryValues.find(value); found != geometryValues.end()) { return found->second; }
        if (value == loop.getInductionVar()) { return static_cast<int64_t>(origin); }
        if (normalizer.independent(value)) { return std::nullopt; }
        auto fixed = normalizer.atPhase(value, 0, PhaseNormalization::modulus(value).value_or(1));
        if (!fixed) { return std::nullopt; }
        auto literal = arena->constantValue(*fixed);
        return literal ? std::optional<int64_t>(APInt(64, *literal).getSExtValue()) : std::nullopt;
    }
    std::optional<RotatingRegionFamily> family(const RotatingAccess& access)
    {
        RotatingRegionFamily result;
        result.banks = access.slots; result.stride = access.stride; result.offset = access.offset;
        if (access.firstPhysicalSlot) {
            result.firstBank = *access.firstPhysicalSlot;
            result.bankStride = access.physicalSlotStride;
            return result;
        }
        auto buffers = input.buffers().find(access.family);
        if (buffers == input.buffers().end() || buffers->second.size() != 1) {
            error = "rotating family lacks a unique shared physical descriptor"; return std::nullopt;
        }
        const auto& memory = *buffers->second.front();
        auto alloc = access.family.getDefiningOp<AllocMultiTileOp>();
        auto base = alloc && alloc.getAddr() ? positiveConstant(alloc.getAddr(), true) : std::nullopt;
        if (base && !alloc->hasAttr("pto.multi_buffer_addrs") && memory.allocateSize) {
            if (*base > INT64_MAX || memory.allocateSize > uint64_t(INT64_MAX) - *base) {
                error = "rotating family base extent exceeds coordinate representation"; return std::nullopt;
            }
            result.firstBank = {memory.scope, *base, *base + memory.allocateSize};
            result.bankStride = memory.allocateSize;
            return result;
        }
        auto ranges = mlir::pto::detail::physicalSlotRanges(input, memory);
        if (ranges.size() != access.slots || ranges.empty()) {
            error = "rotating family requires an affine constant-stride physical bank map"; return std::nullopt;
        }
        result.firstBank = ranges.front();
        result.bankStride = ranges.size() == 1 ? ranges.front().end-ranges.front().begin :
                                               ranges[1].begin-ranges[0].begin;
        for (std::size_t i = 0; i < ranges.size(); ++i) {
            if (!sameStorageDomain(ranges[i], result.firstBank) ||
                APInt(128, ranges[i].begin) != APInt(128, result.firstBank.begin) +
                    APInt(128, result.bankStride)*APInt(128, i) ||
                ranges[i].end-ranges[i].begin != result.firstBank.end-result.firstBank.begin) {
                error = "rotating family physical bank map is not affine and injective"; return std::nullopt;
            }
        }
        return result;
    }
    FailureOr<RegionalAnalysis> canonical(std::size_t id)
    {
        const auto& child = program.nodes[id];
        if (child.kind == StructureKind::Sequence || child.kind == StructureKind::Section) {
            std::vector<RegionalAnalysis> parts;
            for (auto nested : child.children) {
                if (program.nodes[nested].payloads.empty() && program.nodes[nested].children.empty()) { continue; }
                auto view = canonical(nested);
                if (failed(view)) { return failure(); }
                parts.push_back(std::move(*view));
            }
            SmallVector<scf::ForOp> enclosing(child.loops.begin(), child.loops.end());
            if (!llvm::is_contained(enclosing, loop)) { enclosing.push_back(loop); }
            auto composed = composeRegionalSequenceWithin(function, arena, std::move(parts), true, true, enclosing);
            if (!composed.error.empty()) { error = composed.error; return failure(); }
            return sequenceRegionalResult(composed);
        }
        if (child.kind == StructureKind::ExplicitRun) {
            SmallVector<scf::ForOp> enclosing(child.loops.begin(), child.loops.end());
            if (!llvm::is_contained(enclosing, loop)) { enclosing.push_back(loop); }
            auto finite = specializedExplicitRunRegional(function, loop, child.operations, index, input,
                arena, enclosing, [&](Value value) { return geometry(value); }, error);
            if (succeeded(finite)) { return finite; }
            // The finite run adapter cannot export every symbolic GM origin.
            // Reuse the same arithmetic leaf provider as compact children,
            // retaining original operations and composing their exact exports.
            std::vector<RegionalAnalysis> parts;
            for (auto* operation : child.operations) {
                if (index.phasesFor(operation).empty()) {
                    RecognitionResult leaf;
                    detail::inspectLeaf(*operation, index, leaf);
                    if (leaf.state != RecognitionState::Applicable) {
                        error = "rotating explicit phase has an uninterpreted leaf";
                        for (const auto& diagnostic : leaf.diagnostics) {
                            error += "; " + recognitionName(diagnostic.issue).str();
                        }
                        return failure();
                    }
                    continue;
                }
                error.clear();
                auto leaf = analyzeArithmeticRegionWithProfiles({function, operation}, index, input, arena, error,
                    program.regionalArithmeticProfiles,
                    [&](Value value) -> std::optional<Id> {
                        auto fixed = geometry(value);
                        return fixed ? arena->constant(static_cast<uint64_t>(*fixed)) : arena->input(value);
                    });
                if (failed(leaf)) { return failure(); }
                parts.push_back(std::move(*leaf));
            }
            auto composed = composeRegionalSequenceWithin(function, arena, std::move(parts), true, true, enclosing);
            if (!composed.error.empty()) { error = composed.error; return failure(); }
            return sequenceRegionalResult(composed);
        }
        if (child.kind == StructureKind::Loop || child.kind == StructureKind::Conditional) {
            return analyzeArithmeticRegionWithProfiles({function, child.anchor}, index, input, arena, error,
                program.regionalArithmeticProfiles,
                [&](Value value) -> std::optional<Id> {
                    auto fixed = geometry(value);
                    if (fixed) { return arena->constant(static_cast<uint64_t>(*fixed)); }
                    return arena->input(value);
                });
        }
        error = "rotating canonical child has no regional provider"; return failure();
    }
};
} // namespace
RepeatedRegionAnalysis recognizeRotatingRegion(func::FuncOp function, const SyncInput& input,
    const ProgramRecognition& program, std::size_t node, const PhaseIndex& index,
    std::shared_ptr<RegionExpressions> expressions)
{
    if (node >= program.nodes.size() || !expressions) {
        RepeatedRegionAnalysis result; result.error = "invalid rotating region request"; return result;
    }
    return Producer(function, input, program, node, index, std::move(expressions)).run();
}
} // namespace mlir::pto::frontiersynch
