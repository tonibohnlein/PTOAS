// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_EXACTFRONTIERCORE_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_EXACTFRONTIERCORE_H

#include "PTO/Transforms/FrontierSynch/OriginalProgramPoints.h"
#include "PTO/Transforms/FrontierSynch/OriginalValueArithmetic.h"
#include <algorithm>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace mlir::pto::frontiersynch::exact_frontier {

// D3's reference-position algebra, independent of selected synchronization.
// IDs 0/1 denote false/true; frontier 0 is empty. The client supplies the shared
// predicate/frontier arena. None of these procedures enumerates valuations.
struct Summary {
    bool complete = true;
    std::size_t nonempty = 0, first = 0, last = 0;
    std::string reason;
};
inline Summary unknown(std::string reason) { return {false, 0, 0, 0, std::move(reason)}; }
struct Algebra {
    std::function<std::size_t(std::size_t, std::size_t)> conjunction, disjunction, unite, guard;
    std::function<std::size_t(std::size_t)> negate;
};
inline Summary sequence(const Algebra& a, const std::vector<Summary>& parts)
{
    std::vector<std::size_t> suffix(parts.size() + 1, 0);
    for (auto i = parts.size(); i-- > 0;) {
        if (!parts[i].complete) {
            return parts[i];
        }
        suffix[i] = a.disjunction(parts[i].nonempty, suffix[i + 1]);
    }
    Summary out;
    std::size_t prefix = 0;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        out.first = a.unite(out.first, a.guard(parts[i].first, a.negate(prefix)));
        out.last = a.unite(out.last, a.guard(parts[i].last, a.negate(suffix[i + 1])));
        prefix = a.disjunction(prefix, parts[i].nonempty);
    }
    out.nonempty = prefix;
    return out;
}
inline Summary choice(const Algebra& a, std::size_t condition, const Summary& yes, const Summary& no)
{
    // A proved constant choice has no unexecuted-arm arithmetic/qualification.
    if (condition == 0) {
        return no;
    }
    if (condition == 1) {
        return yes;
    }
    if (!yes.complete || !no.complete) {
        return !yes.complete ? yes : no;
    }
    const auto negative = a.negate(condition);
    return {true,
            a.disjunction(a.conjunction(condition, yes.nonempty), a.conjunction(negative, no.nonempty)),
            a.unite(a.guard(yes.first, condition), a.guard(no.first, negative)),
            a.unite(a.guard(yes.last, condition), a.guard(no.last, negative)), {}};
}
inline Summary counted(
    const Algebra& a, const Summary& body, std::size_t visits, std::size_t firstVisit,
    std::size_t lastVisit, bool invariantParticipation)
{
    if (visits == 0) {
        return {}; // No body visit, including no speculative body arithmetic.
    }
    if (!body.complete) {
        return body;
    }
    if (body.nonempty == 0) {
        return {};
    }
    if (!invariantParticipation) {
        return unknown("counted participation varies without an admitted interval rule");
    }
    return {true, a.conjunction(visits, body.nonempty),
            a.guard(body.first, firstVisit), a.guard(body.last, lastVisit), {}};
}

// A compact I.2 recipe over immutable original-value references. A successor
// bound is permitted only after the adapter has independently proved +1 safe.
// Evaluation checks again; it never wraps a bound to fabricate an empty domain.
// Normalize comparisons without enumerating Boolean cases. The native adapter
// first checks that the MLIR comparison uses the loop's integer interpretation.
enum class Comparison { Equal, NotEqual, Less, LessEqual, Greater, GreaterEqual };
inline Comparison normalizeComparison(Comparison p, bool swapped, bool positive)
{
    if (swapped) {
        switch (p) {
        case Comparison::Less: p = Comparison::Greater; break;
        case Comparison::LessEqual: p = Comparison::GreaterEqual; break;
        case Comparison::Greater: p = Comparison::Less; break;
        case Comparison::GreaterEqual: p = Comparison::LessEqual; break;
        default: break;
        }
    }
    if (!positive) {
        switch (p) {
        case Comparison::Equal: return Comparison::NotEqual;
        case Comparison::NotEqual: return Comparison::Equal;
        case Comparison::Less: return Comparison::GreaterEqual;
        case Comparison::LessEqual: return Comparison::Greater;
        case Comparison::Greater: return Comparison::LessEqual;
        case Comparison::GreaterEqual: return Comparison::Less;
        }
    }
    return p;
}
struct Bound {
    std::size_t reference = NoControlId;
    bool successor = false;
    bool operator<(const Bound& other) const
    {
        return std::tie(reference, successor) < std::tie(other.reference, other.successor);
    }
};
struct IntervalRecipe {
    value_arithmetic::Integer integer;
    std::vector<Bound> lower, upper; // includes the original loop's LB and UB
};
struct Intersection {
    bool valid = false, nonempty = false;
    uint64_t lower = 0, upper = 0;
    // No last coordinate is computed by intersection(). This permits a caller
    // to test nonemptiness without evaluating h-1 on a nonparticipating path.
};
template <class Lookup>
Intersection intersection(const IntervalRecipe& recipe, Lookup lookup)
{
    Intersection out;
    const auto type = recipe.integer;
    if (!type.valid() || recipe.lower.empty() || recipe.upper.empty()) {
        return out;
    }
    auto bound = [&](Bound b) -> std::optional<uint64_t> {
        auto value = lookup(b.reference);
        if (!value || !type.fits(*value)) {
            return {};
        }
        return b.successor ? value_arithmetic::checked(value_arithmetic::Binary::Add, type, *value, 1) : value;
    };
    auto fold = [&](const std::vector<Bound>& bounds, bool minimum) -> std::optional<uint64_t> {
        auto value = bound(bounds.front());
        for (std::size_t i = 1; value && i < bounds.size(); ++i) {
            auto next = bound(bounds[i]);
            if (!next) {
                return {};
            }
            value = value_arithmetic::checked(
                minimum ? value_arithmetic::Binary::Minimum : value_arithmetic::Binary::Maximum,
                type, *value, *next);
        }
        return value;
    };
    auto lo = fold(recipe.lower, false), hi = fold(recipe.upper, true);
    if (!lo || !hi) {
        return out;
    }
    out.valid = true;
    out.lower = *lo;
    out.upper = *hi;
    out.nonempty = type.less(*lo, *hi);
    return out;
}
inline std::optional<uint64_t> coordinate(
    value_arithmetic::Integer type, const Intersection& domain, bool last)
{
    if (!domain.valid || !domain.nonempty || !type.fits(domain.lower) || !type.fits(domain.upper) ||
        !type.less(domain.lower, domain.upper)) {
        return {}; // The empty arm evaluates no subtraction.
    }
    return last ? value_arithmetic::checked(value_arithmetic::Binary::Subtract, type, domain.upper, 1) :
                  std::optional<uint64_t>(domain.lower);
}

// A cut-delimited contiguous structural corridor. Only sequence wrappers are
// transparent. Complete choices/repeats stay in parts and undergo the same
// structural query. A cut inside an unfinished child cannot be widened to the
// child's entry/exit. Transport across such a split needs a separate premise.
class SliceIndex {
public:
    struct Slice {
        bool complete = false;
        const std::vector<const Region*>* parts = nullptr;
        std::size_t begin = 0, end = 0;
        std::string reason;
    };
    explicit SliceIndex(const Region& root)
    {
        addContext(root, OriginalCut::scope(NoControlId, OriginalCut::Before),
                   OriginalCut::scope(NoControlId, OriginalCut::After), false);
    }
    Slice slice(const OriginalIntervalRequest& q) const
    {
        Slice out;
        if (q.occurrence.stopVisit == OriginalOccurrenceContext::StopVisit::AfterBackedge) {
            out.reason = "exact frontier needs qualified re-entry transport for this backedge interval";
            return out;
        }
        const auto start = gaps.find(q.start), actualStop = gaps.find(q.stop);
        if (start == gaps.end() || actualStop == gaps.end()) {
            out.reason = "exact frontier has no represented structural cut";
            return out;
        }
        auto stopCut = q.stop;
        if (stopCut.kind == OriginalCut::Kind::Payload) {
            stopCut.side = q.includeStoppingAccess ? OriginalCut::After : OriginalCut::Before;
        } else if (q.includeStoppingAccess) {
            out.reason = "structured stopping cut has no payload access";
            return out;
        }
        const auto stop = gaps.find(stopCut);
        if (stop == gaps.end()) {
            out.reason = "stopping payload has no structural access boundary";
            return out;
        }
        // A gap can have aliases at an empty child. Pick a common sequence
        // context, never concatenate paths from separate alternative arms.
        for (const auto& a : start->second) {
            for (const auto& b : stop->second) {
                if (a.context != b.context) {
                    continue;
                }
                const auto& context = contexts[a.context];
                if (context.repeated && q.occurrence.stopVisit != OriginalOccurrenceContext::StopVisit::FirstReach) {
                    out.reason = "repeated-body corridor lacks its first-reach occurrence interpretation";
                    continue;
                }
                const bool sameAfter = q.start == q.stop && q.start.side == OriginalCut::After;
                if (a.position > b.position && !sameAfter) {
                    continue;
                }
                out.complete = true;
                out.parts = &context.parts;
                out.begin = a.position;
                out.end = sameAfter ? a.position : b.position;
                out.reason.clear();
                return out;
            }
        }
        if (out.reason.empty()) {
            out.reason = "cuts split a structured child; missing qualified child-boundary transport";
        }
        return out;
    }
    std::size_t contextCount() const { return contexts.size(); }

private:
    struct Context { std::vector<const Region*> parts; bool repeated = false; };
    struct Gap { std::size_t context, position; };
    void addGap(OriginalCut cut, std::size_t context, std::size_t position)
    {
        gaps[cut].push_back({context, position});
    }
    void flatten(const Region& region, std::size_t context)
    {
        if (region.kind == Region::Sequence && region.originalOwner == NoControlId) {
            for (const auto& child : region.children) {
                flatten(child, context);
            }
            return;
        }
        const auto position = contexts[context].parts.size();
        contexts[context].parts.push_back(&region);
        if (region.kind == Region::Operation) {
            addGap({region.operation, OriginalCut::Before}, context, position);
            addGap({region.operation, OriginalCut::After}, context, position + 1);
        } else if (region.originalOwner != NoControlId) {
            addGap(OriginalCut::scope(region.originalOwner, OriginalCut::Before), context, position);
            addGap(OriginalCut::scope(region.originalOwner, OriginalCut::After), context, position + 1);
        }
    }
    void addContext(const Region& region, OriginalCut before, OriginalCut after, bool repeated)
    {
        const auto context = contexts.size();
        contexts.push_back({{}, repeated});
        if (region.kind == Region::Sequence) {
            for (const auto& child : region.children) { flatten(child, context); }
        } else {
            flatten(region, context);
        }
        addGap(before, context, 0);
        addGap(after, context, contexts[context].parts.size());
        // Copy pointers before recursion: contexts can reallocate.
        const auto parts = contexts[context].parts;
        for (const auto* part : parts) {
            if (part->kind == Region::Operation || part->originalOwner == NoControlId) {
                continue;
            }
            if (part->kind == Region::Sequence) {
                addContext(*part, OriginalCut::scope(part->originalOwner, OriginalCut::Before),
                           OriginalCut::scope(part->originalOwner, OriginalCut::After), repeated);
                continue;
            }
            for (std::size_t i = 0; i < part->children.size(); ++i) {
                addContext(part->children[i],
                           OriginalCut::childBoundary(part->originalOwner, i, OriginalCut::Before),
                           OriginalCut::childBoundary(part->originalOwner, i, OriginalCut::After),
                           repeated || part->kind == Region::For || part->kind == Region::While);
            }
        }
    }
    std::vector<Context> contexts;
    std::map<OriginalCut, std::vector<Gap>> gaps;
};

} // namespace mlir::pto::frontiersynch::exact_frontier
#endif
