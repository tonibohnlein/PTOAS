// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Expand actual retained records for the independent small-instance oracle.
#ifndef PTO_TEST_COMPACTREGIONDEMANDS_H
#define PTO_TEST_COMPACTREGIONDEMANDS_H
#include "CompactRegionCoordinates.h"
#include "PTO/Transforms/FrontierSynch/Arithmetic/ArithmeticRegional.h"
#include "llvm/Support/JSON.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Finite/FiniteGuardedInternal.h"
namespace compact_contract {
class Demands {
public:
    const fs::RegionalAnalysis& root;
    ArrayRef<const pto::CompoundInstanceElement*> phases;
    const std::vector<Key>& original;
    std::string error;
    bool selected(
        const fs::RegionalAnalysis& region, fs::RegionalEvent a, fs::RegionalEvent b, const Coordinates& bindings,
        const fs::Child* child = nullptr)
    {
        if (auto repeat = region.retainedRepetition.lock()) {
            if (a.visits.empty() || b.visits.empty()) {
                return fail("missing repeated coordinate");
            }
            auto from = number(region, a.visits.front()), to = number(region, b.visits.front());
            if (!from || !to) {
                return fail("nonconcrete repeated coordinate");
            }
            a.visits.erase(a.visits.begin());
            b.visits.erase(b.visits.begin());
            if (*from == *to) {
                return selected(repeat->body, a, b, bindings);
            }
            if (*to < *from) {
                return false;
            }
            for (const auto& record : repeat->crossings) {
                if (!record.native && record.displacement == *to - *from &&
                    match(repeat->body, record.source, a, bindings) &&
                    match(repeat->body, record.target, b, bindings) && truth(repeat->body, record.guard, bindings)) {
                    return true;
                }
            }
            return false;
        }
        if (auto sequence = region.retainedSequence.lock()) {
            auto identify = [&](uint32_t type) -> std::pair<std::size_t, uint32_t> {
                for (std::size_t i = 0; i < sequence->children.size(); ++i) {
                    const auto size = sequence->children[i].regional.anchors.size();
                    if (type < size) {
                        return {i, type};
                    }
                    type -= size;
                }
                return {sequence->children.size(), 0};
            };
            auto x = identify(a.type), y = identify(b.type);
            if (x.first == sequence->children.size() || y.first == sequence->children.size()) {
                return fail("sequence type outside retained children");
            }
            a.type = x.second;
            b.type = y.second;
            for (const auto& record : sequence->crossings) {
                if (record.source >= sequence->ports.size() || record.target >= sequence->ports.size()) {
                    return fail("invalid retained crossing port");
                }
                const auto& from = sequence->ports[record.source];
                const auto& to = sequence->ports[record.target];
                if (from.child == x.first && to.child == y.first &&
                    match(sequence->children[x.first].regional, from.event(), a, bindings) &&
                    match(sequence->children[y.first].regional, to.event(), b, bindings) &&
                    truth(region, record.guard, bindings)) {
                    return true;
                }
            }
            // The numerical body's synthetic one-child sequence stores every
            // reduced internal edge as a crossing. Its child supplies only
            // cuts and numerical queries, and deliberately has no demands.
            if (region.construction && region.construction->representation == "finite-numerical-body") {
                return false;
            }
            return x.first == y.first &&
                   selected(sequence->children[x.first].regional, a, b, bindings, &sequence->children[x.first]);
        }
        auto owner = region.retainedMathematics.lock();
        if (!owner && child) { owner = child->prefixMathematics; }
        const auto* finite = child && child->finite ?
                                 child->finite.get() :
                                 (owner && owner->finiteDemands ? owner->finiteDemands.get() : nullptr);
        if (finite && finite->concrete()) {
            const auto& retained = finite->concrete()->reduction.retained;
            return llvm::any_of(
                retained, [&](const auto& record) { return record.source == a.type && record.target == b.type; });
        }
        const auto* guarded = finite ? finite->guarded() :
            (owner && owner->expandedFiniteDemands ? owner->expandedFiniteDemands.get() : nullptr);
        if (guarded && guarded->state) {
            const auto& state = *guarded->state;
            if (state.arena != region.expressions) {
                return fail("guarded record arena mismatch");
            }
            return llvm::any_of(state.retained, [&](const auto& record) {
                return record.source == a.type && record.target == b.type && truth(region, record.guard, bindings);
            });
        }
        if (owner && owner->arithmeticRegionalDemands) {
            const auto& arithmetic = *owner->arithmeticRegionalDemands;
            // Empty is established from the retained demand inventory, never
            // inferred from an unavailable relation or failed query export.
            if (arithmetic.differenceDemands) {
                if (arithmetic.differenceDemands->minimumDemands.empty()) { return false; }
            } else if (arithmetic.analysis.hasRelations(fs::ArithmeticRelationView::Demands) &&
                       arithmetic.analysis.minimumDemands.empty()) { return false; }
            return fail("nonempty arithmetic record expansion unavailable in this test adapter");
        }
        if (child && !child->periodic.payloads.empty()) {
            auto from = number(region, a.ordinal), to = number(region, b.ordinal);
            if (!from || !to || *to < *from) {
                return false;
            }
            return llvm::any_of(child->periodic.retained, [&](uint32_t index) {
                if (index >= child->periodic.generators.size()) {
                    return fail("invalid periodic record index");
                }
                const auto& record = child->periodic.generators[index];
                return record.source == a.type && record.target == b.type && record.displacement == *to - *from;
            });
        }
        if (region.anchors.size() == 1 && region.occurrenceLoops.size() == 1 && !region.occurrenceLoops[0]) {
            return false;
        }
        return fail(
            "retained leaf demand inventory unavailable: anchors=" + std::to_string(region.anchors.size()) +
            " finite=" + std::to_string(finite != nullptr) + " child=" + std::to_string(child != nullptr));
    }
    llvm::json::Array expand(const std::vector<std::optional<fs::RegionalEvent>>& events)
    {
        llvm::json::Array result;
        for (std::size_t a = 0; a < events.size(); ++a) {
            if (!events[a]) {
                continue;
            }
            Coordinates bindings;
            auto loops = enclosing(phases[original[a].first]->elementOp);
            if (loops.size() != original[a].second.size()) {
                fail("original occurrence coordinate arity mismatch");
                return result;
            }
            for (std::size_t i = 0; i < loops.size(); ++i) {
                bindings[loops[i]] = original[a].second[i];
            }
            for (std::size_t b = a + 1; b < events.size(); ++b) {
                if (events[b] && selected(root, *events[a], *events[b], bindings)) {
                    result.push_back(llvm::json::Array{a, b});
                }
            }
        }
        return result;
    }

private:
    bool fail(llvm::StringRef reason)
    {
        if (error.empty()) {
            error = reason.str();
        }
        return false;
    }
    std::optional<uint64_t> evaluate(
        const fs::RegionalAnalysis& region, fs::RegionExpressions::Id id, const Coordinates& bindings)
    {
        auto& e = *region.expressions;
        SmallVector<std::pair<fs::RegionExpressions::Id, fs::RegionExpressions::Id>> values;
        for (auto entry : bindings) {
            if (auto loop = dyn_cast<scf::ForOp>(entry.first)) {
                values.emplace_back(e.input(loop.getInductionVar()), e.constant(entry.second));
            }
        }
        fs::RegionExpressions::Substitution substitution(values);
        return number(region, e.substitute(id, substitution));
    }
    bool truth(const fs::RegionalAnalysis& region, fs::RegionExpressions::Id guard, const Coordinates& bindings)
    {
        auto value = evaluate(region, guard, bindings);
        if (!value) {
            return fail("retained record predicate is not concrete under original visit");
        }
        return *value != 0;
    }
    bool match(
        const fs::RegionalAnalysis& region, const fs::RegionalEvent& record, const fs::RegionalEvent& actual,
        const Coordinates& bindings)
    {
        if (record.type != actual.type || record.visits.size() != actual.visits.size()) {
            return false;
        }
        auto equal = [&](auto a, auto b) {
            auto x = evaluate(region, a, bindings), y = number(region, b);
            if (!x || !y) {
                return fail("retained record coordinate is not concrete");
            }
            return *x == *y;
        };
        if (!equal(record.ordinal, actual.ordinal)) {
            return false;
        }
        for (std::size_t i = 0; i < record.visits.size(); ++i) {
            if (!equal(record.visits[i], actual.visits[i])) {
                return false;
            }
        }
        return true;
    }
};
} // namespace compact_contract
#endif
