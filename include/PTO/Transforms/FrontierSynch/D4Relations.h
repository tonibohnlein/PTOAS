// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_D4RELATIONS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_D4RELATIONS_H

#include "PTO/Transforms/FrontierSynch/FactoredUse.h"

namespace mlir::pto::frontiersynch {

// The D1/D2 producer owns these qualifications. D4 may link qualified first/last
// *uses* of a physical cell across children; it cannot derive a coordinate map
// from a scalar reset, a modulus, a may-role answer or a child invocation number.
// `coordinate` is the producer's immutable occurrence descriptor. It is not a
// runtime counter. Equal sites with unequal descriptors remain different roles.
struct D4UseRole {
    std::size_t site = NoControlId, coordinate = NoControlId;
    std::size_t owner = NoControlId, cell = NoControlId;
    OriginalCut cut;
    bool operator==(const D4UseRole& other) const
    {
        return std::tie(site, coordinate, owner, cell, cut) ==
               std::tie(other.site, other.coordinate, other.owner, other.cell, other.cut);
    }
};
enum class D4CorrespondenceRule { Unqualified, D1, D2 };
// An already established D1/D2 relation, kept independently of a boundary
// profile. Optional distances and NoFactoredId domains remain unresolved;
// composition never makes them exact. D2 distances apply INSIDE the qualified
// child domain, not across a reset/re-entry boundary. Those links use role refs.
struct D4ExistingCorrespondence {
    D4CorrespondenceRule rule = D4CorrespondenceRule::Unqualified;
    OriginalInterval interpretation;
    std::size_t source = NoControlId, target = NoControlId, selectorOwner = NoControlId;
    // Boundary qualification is part of the retained premise, not mutable
    // profile decoration. D4 checks these exact role/condition references.
    D4UseRole firstUse, lastUse;
    std::size_t nonempty = NoFactoredId;
    std::optional<D4UseRole> enclosingUse;
    std::optional<std::size_t> previousDistance, nextDistance;
    std::size_t previousDomain = NoFactoredId, nextDomain = NoFactoredId;
    std::size_t initialDomain = NoFactoredId, finalDomain = NoFactoredId;
    bool qualified = false;
};
struct D4ChildUse {
    using Rule = D4CorrespondenceRule;
    Rule rule = Rule::Unqualified;
    std::shared_ptr<const D4ExistingCorrespondence> correspondence;
    enum class Kind { NoUse, CompleteUses, ReadFragment } kind = Kind::NoUse;
    // Exact original interpretation of the supplied child relation, including
    // its enclosing owner/horizon. A local relation with another owner is NOT
    // an enclosing relation merely because the same bank name occurs there.
    OriginalInterval interpretation;
    std::size_t child = NoControlId;
    D4UseRole first, last;
    // For ReadFragment, the qualified enclosing writer/use identity must agree
    // on both sides. A genuine intervening overwrite requires another use.
    D4UseRole enclosingUse;
    std::size_t nonempty = 0;
    // Qualified by D1/D2/D3 and the original interval/physical projection.
    // NoUse needs a proved All exclusion, not a failed positive query.
    bool occurrenceQualified = false, noUseProved = false;
    std::string missingPremise;
};

// A separate typed DAG of occurrence references. Its conditions are shared
// original FactoredUseArena expressions, with exactly the same guard identity.
// It contains no selected notification, completion or protocol state.
struct D4UseExpression {
    enum class Kind { Incoming, Use, Choose, Unknown, NoUse } kind = Kind::Incoming;
    D4UseRole use;
    std::size_t condition = 1, yes = 0, no = 0;
    std::size_t child = NoControlId;
};
struct D4UseLink {
    std::size_t child = NoControlId;
    D4UseRole endpoint;
    // Backward links name successor uses; forward links name predecessors.
    std::size_t otherUse = 0, applicability = 0;
    bool backward = false;
};
struct D4RelationComposition {
    OriginalInterval interpretation;
    std::shared_ptr<const FactoredUseArena> conditions;
    // Complete composition of the ORIGINAL relation. Availability of a newly
    // composed first/last predicate at its endpoint is a separate qualification.
    bool complete = false;
    // Incoming means the DECLARED owning input/continuation, not fresh history.
    std::vector<D4UseExpression> expressions;
    std::vector<D4UseLink> links;
    // Keep the original within-child maps/domains, not only the new links.
    std::vector<D4ChildUse> children;
    std::size_t lastUse = 0, nextUse = 0;
    std::vector<std::pair<std::size_t, std::string>> unresolved;
};

// Compose profiles for ONE physical-use sequence. Independent cells/selectors
// are independent calls, not a joint period or valuation product. The child
// first/last/domain qualification is an explicit input premise of D4 (3.6/I.1).
// All endpoints and domains are retained by reference; no constant distance is
// inferred for an edge crossing a child boundary.
inline D4RelationComposition composeD4Relations(
    const OriginalInterval& interpretation, std::shared_ptr<const FactoredUseArena> conditions,
    const std::vector<D4ChildUse>& children)
{
    D4RelationComposition out;
    out.interpretation = interpretation;
    out.children = children;
    out.conditions = std::move(conditions);
    out.expressions.push_back({}); // Owning incoming previous-use interface.
    out.expressions.push_back({}); // Owning following next-use interface.
    out.nextUse = 1;
    if (!out.conditions || !interpretation.query.version ||
        out.conditions->frame().snapshot != interpretation.query.version ||
        out.conditions->frame().owner != interpretation.owner ||
        out.conditions->frame().cell != interpretation.query.selector.cell) {
        out.unresolved.push_back({NoControlId, "D4 original owner, cell, snapshot or predicate arena differs"});
        return out;
    }
    auto roleValid = [&](const D4UseRole& use) {
        return use.site != NoControlId && use.coordinate != NoControlId && use.owner == interpretation.owner &&
               use.cell == interpretation.query.selector.cell;
    };
    auto append = [&](D4UseExpression value) {
        const auto id = out.expressions.size();
        out.expressions.push_back(std::move(value));
        return id;
    };
    auto choose = [&](std::size_t condition, std::size_t yes, std::size_t no) {
        if (condition == 0 || yes == no) {
            return no;
        }
        if (condition == 1) {
            return yes;
        }
        D4UseExpression node;
        node.kind = D4UseExpression::Kind::Choose;
        node.condition = condition;
        node.yes = yes;
        node.no = no;
        return append(node);
    };
    std::vector<bool> admitted(children.size(), false);
    out.complete = true;
    for (std::size_t i = 0; i < children.size(); ++i) {
        const auto& child = children[i];
        std::string missing = child.missingPremise;
        if (!(child.interpretation == interpretation)) {
            missing = "D4 child needs an already qualified relation in the same owning continuation";
        } else if (child.kind == D4ChildUse::Kind::NoUse) {
            if (child.noUseProved) {
                admitted[i] = true;
            } else {
                missing = "D4 no-use child lacks a proved All exclusion";
            }
        } else if (!child.correspondence || !child.correspondence->qualified ||
                   child.correspondence->rule != child.rule ||
                   !(child.correspondence->interpretation == interpretation)) {
            if (missing.empty()) {
                missing = "D4 requires a retained, qualified child relation in this exact original interpretation";
            }
        } else if (!(child.correspondence->firstUse == child.first) ||
                   !(child.correspondence->lastUse == child.last) ||
                   child.correspondence->nonempty != child.nonempty) {
            missing = "D4 child boundaries/participation differ from the retained qualified relation";
        } else if (child.rule == D4ChildUse::Rule::D2 &&
                   child.correspondence->source == child.correspondence->target &&
                   ((child.correspondence->previousDistance && *child.correspondence->previousDistance == 0) ||
                    (child.correspondence->nextDistance && *child.correspondence->nextDistance == 0))) {
            missing = "D4 child relation has an invalid same-occurrence self edge";
        } else if (!child.occurrenceQualified || child.rule == D4ChildUse::Rule::Unqualified) {
            if (missing.empty()) {
                missing = "D4 requires an already qualified child D1/D2 occurrence relation";
            }
        } else if (!roleValid(child.first) || !roleValid(child.last)) {
            missing = "D4 first/last use lacks its original physical owner or occurrence descriptor";
        } else if (!out.conditions->hasSort(child.nonempty, FactoredUseNode::Sort::Condition) ||
                   (*out.conditions)[child.nonempty].containsUnresolved) {
            missing = "D4 child participation is unresolved in this original guard interpretation";
        } else if (child.kind == D4ChildUse::Kind::ReadFragment) {
            missing = "D4 read fragments require composeD4ReadFragments, not a complete-use transition";
        } else {
            admitted[i] = true;
        }
        if (!admitted[i]) {
            out.complete = false;
            out.unresolved.push_back({child.child, std::move(missing)});
        }
    }
    auto direction = [&](bool backward) {
        std::size_t state = backward ? 1 : 0;
        for (std::size_t j = 0; j < children.size(); ++j) {
            const auto i = backward ? children.size() - j - 1 : j;
            const auto& child = children[i];
            if (!admitted[i]) {
                // Retain the incoming relation as a may dependency, but an
                // unknown child must block an exact bypass to a later use.
                D4UseExpression unknown;
                unknown.kind = D4UseExpression::Kind::Unknown;
                unknown.child = child.child;
                unknown.yes = state;
                state = append(unknown);
                continue;
            }
            if (child.kind == D4ChildUse::Kind::NoUse || child.nonempty == 0) {
                continue;
            }
            out.links.push_back(
                {child.child, backward ? child.last : child.first, state, child.nonempty, backward});
            D4UseExpression use;
            use.kind = D4UseExpression::Kind::Use;
            use.use = backward ? child.first : child.last;
            state = choose(child.nonempty, append(use), state);
        }
        return state;
    };
    out.lastUse = direction(false);
    out.nextUse = direction(true);
    return out;
}

// D3's sequence equations, used by D4 when a child boundary cuts an unfinished
// read episode. This returns first/last READER references in the enclosing use;
// it does not advance the physical predecessor sequence or supply completion.
struct D4ReadComposition {
    OriginalInterval interpretation;
    D4UseRole enclosingUse;
    D4RelationComposition boundaries;
    bool complete = false;
    std::size_t nonempty = 0, first = 0, last = 0;
    std::shared_ptr<FactoredUseArena> conditions;
};
inline D4ReadComposition composeD4ReadFragments(
    const OriginalInterval& interpretation, std::shared_ptr<FactoredUseArena> conditions,
    const D4UseRole& enclosingUse, const std::vector<D4ChildUse>& children)
{
    D4ReadComposition out;
    out.interpretation = interpretation;
    out.enclosingUse = enclosingUse;
    out.conditions = conditions;
    auto profiles = children;
    const bool namedIncoming = enclosingUse.site == NoControlId &&
                               interpretation.query.occurrence.incomingInterface != NoControlId &&
                               enclosingUse.coordinate == interpretation.query.occurrence.incomingInterface;
    const bool validEnclosing = (enclosingUse.site != NoControlId || namedIncoming) &&
                                enclosingUse.coordinate != NoControlId && enclosingUse.owner == interpretation.owner &&
                                enclosingUse.cell == interpretation.query.selector.cell;
    for (auto& child : profiles) {
        if (child.kind == D4ChildUse::Kind::NoUse) {
            continue;
        }
        if (!validEnclosing || child.kind != D4ChildUse::Kind::ReadFragment ||
            !(child.enclosingUse == enclosingUse) || !child.correspondence ||
            !child.correspondence->enclosingUse || !(*child.correspondence->enclosingUse == enclosingUse)) {
            child.occurrenceQualified = false;
            child.missingPremise = "D4 read fragment changed its enclosing use (possible intervening overwrite)";
        }
        // Only the first/last algebra is reused; these are not exported as
        // complete-use links. The original profiles remain unchanged.
        child.kind = D4ChildUse::Kind::CompleteUses;
    }
    out.boundaries = composeD4Relations(interpretation, conditions, profiles);
    out.complete = out.boundaries.complete && validEnclosing;
    if (!validEnclosing) {
        out.boundaries.complete = false;
        out.boundaries.unresolved.push_back({NoControlId, "D4 unfinished use has no qualified enclosing identity"});
    }
    out.first = out.boundaries.nextUse;
    out.last = out.boundaries.lastUse;
    out.boundaries.links.clear();
    out.boundaries.children = children; // Preserve ReadFragment, never export a completed-use relabeling.
    // A no-reader path has NO frontier, unlike a complete-use sequence whose
    // no-use path carries its incoming physical predecessor onward.
    out.boundaries.expressions[0].kind = D4UseExpression::Kind::NoUse;
    out.boundaries.expressions[1].kind = D4UseExpression::Kind::NoUse;
    if (out.complete) {
        for (const auto& child : children) {
            if (child.kind != D4ChildUse::Kind::NoUse) {
                out.nonempty = conditions->choose(child.nonempty, 1, out.nonempty, FactoredUseNode::Sort::Condition);
            }
        }
    } else {
        // Unknown participation is not the no-reader condition.
        out.nonempty = conditions ? conditions->unresolved(interpretation.owner, FactoredUseNode::Sort::Condition) :
                                    NoFactoredId;
    }
    return out;
}
} // namespace mlir::pto::frontiersynch
#endif
