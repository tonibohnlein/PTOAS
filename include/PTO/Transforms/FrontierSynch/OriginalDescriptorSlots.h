// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_ORIGINALDESCRIPTORSLOTS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_ORIGINALDESCRIPTORSLOTS_H

#include "PTO/Transforms/FrontierSynch/OriginalProgramPoints.h"
#include <algorithm>
#include <array>
#include <map>
#include <cassert>

namespace mlir::pto::frontiersynch {

// These references describe ORIGINAL facts. None names a selected ledger,
// physical event, acquired receipt, or a certificate for an executable packet.
// The containing request repertoire owns/borrows their immutable source arenas.
struct DescriptorPredicate {
    enum class Domain { False, True, Obligation, FactoredUse, Reader, OriginalValue } domain = Domain::True;
    std::uintptr_t arena = 0;
    std::size_t root = 0;
    auto key() const { return std::tie(domain, arena, root); }
    bool operator==(const DescriptorPredicate& other) const { return key() == other.key(); }
    bool operator<(const DescriptorPredicate& other) const { return key() < other.key(); }
};
struct DescriptorFactRef {
    enum class Kind { Obligation, Occurrence, CompletionPrerequisite, SupportRole, BoundaryQuery } kind = Kind::Obligation;
    std::uintptr_t arena = 0;
    std::size_t index = NoControlId;
    OriginalProgramVersion universe = {};
    auto key() const { return std::tie(kind, arena, index, universe); }
    bool operator==(const DescriptorFactRef& other) const { return key() == other.key(); }
    bool operator<(const DescriptorFactRef& other) const { return key() < other.key(); }
};
struct DescriptorSupportLink {
    DescriptorFactRef role;
    DescriptorFactRef justification;
    std::int64_t occurrenceShift = 0;
    DescriptorPredicate applicability;
    auto key() const { return std::tie(role, justification, occurrenceShift, applicability); }
    bool operator==(const DescriptorSupportLink& other) const { return key() == other.key(); }
    bool operator<(const DescriptorSupportLink& other) const { return key() < other.key(); }
};
struct DescriptorObservation {
    enum class Status { Available, NeedsCompletion, NotObservableHere, Unresolved } status = Status::Unresolved;
    DescriptorPredicate predicate;
    // Completion prerequisites must be independently dischargeable; naming one
    // is not credit. The adapter retains the actual typed prerequisite records.
    std::vector<DescriptorFactRef> prerequisites;
    bool independentlyDischargeable = false;
    std::vector<std::string> unresolved;
    bool qualified() const
    {
        return status == Status::Available ||
               (status == Status::NeedsCompletion && independentlyDischargeable && !prerequisites.empty());
    }
    auto key() const
    {
        return std::tie(status, predicate, prerequisites, independentlyDischargeable, unresolved);
    }
    bool operator==(const DescriptorObservation& other) const { return key() == other.key(); }
    bool operator<(const DescriptorObservation& other) const { return key() < other.key(); }
};
struct DescriptorEndpoint {
    OriginalCut cut;
    DescriptorPredicate condition;
    DescriptorObservation observation;
    bool legal = false;
    auto key() const { return std::tie(cut, condition, observation, legal); }
    bool operator==(const DescriptorEndpoint& other) const { return key() == other.key(); }
    bool operator<(const DescriptorEndpoint& other) const { return key() < other.key(); }
};
struct DescriptorBoundary {
    enum class Form { Unresolved, NoHit, Exact, Covering } form = Form::Unresolved;
    enum class Direction { Source, Target } direction = Direction::Source;
    enum class Multiplicity { Unresolved, ExactlyParticipating, OncePerInterval } multiplicity = Multiplicity::Unresolved;
    // Fixed interval and selector, including original version and continuation.
    OriginalInterval interval;
    bool intervalQualified = false;
    std::vector<DescriptorEndpoint> endpoints;
    bool referenceCoverage = false;
    bool mayExecuteWithoutAccess = false;
    std::optional<DescriptorPredicate> noHit;
    OriginalContinuationCases continuationCases;
    // Work is represented by original intervals, not by an inferred latency.
    std::vector<OriginalInterval> extraWork;
    std::vector<DescriptorFactRef> witnesses;
    std::vector<std::string> unresolved;
    bool qualified() const
    {
        return (form == Form::Exact || form == Form::Covering) && intervalQualified && bool(interval.query.version) && referenceCoverage &&
               multiplicity != Multiplicity::Unresolved && !endpoints.empty() && unresolved.empty() &&
               std::all_of(endpoints.begin(), endpoints.end(), [](const DescriptorEndpoint& e) {
                   return e.legal && e.observation.predicate == e.condition && e.observation.qualified();
               });
    }
    auto key() const
    {
        return std::tie(form, direction, multiplicity, interval, intervalQualified, endpoints, referenceCoverage,
                        mayExecuteWithoutAccess, noHit, continuationCases.incoming, continuationCases.childEntry,
                        continuationCases.bypass, continuationCases.backedge, continuationCases.reachedStop,
                        continuationCases.reachedOwnerExit, extraWork, witnesses, unresolved);
    }
    bool operator==(const DescriptorBoundary& other) const { return key() == other.key(); }
    bool operator<(const DescriptorBoundary& other) const { return key() < other.key(); }
};
struct DescriptorLeafFacts {
    // References into OriginalRequests, not new obligation IDs. A family-wide
    // unresolved leaf can retain an unexpanded source relation behind one ref.
    std::vector<DescriptorFactRef> obligations;
    DescriptorPredicate applicability;
    unsigned sourceEngine = 0, targetEngine = 0;
    std::vector<DescriptorFactRef> occurrenceMatching, observations;
    std::vector<DescriptorSupportLink> support;
    bool matchingQualified = false;
    std::vector<std::string> unresolved;
    auto key() const
    {
        return std::tie(obligations, applicability, sourceEngine, targetEngine, occurrenceMatching, support,
                        observations, matchingQualified, unresolved);
    }
    bool operator==(const DescriptorLeafFacts& other) const { return key() == other.key(); }
    bool operator<(const DescriptorLeafFacts& other) const { return key() < other.key(); }
};

// Draft v0.44 I.4: one algebra, three pointwise slots. Both is conjunction;
// Choose is one shared original alternative. Neither operation distributes
// over the other. In particular, m optional readers do not form 2^m cases.
class OriginalDescriptorSlots {
public:
    using Id = std::size_t;
    enum class Slot : unsigned { Preferred, SourceCover, TargetCover };
    enum class Kind { Empty, Leaf, Both, Choose };
    struct Answers {
        Id exactSource, coveringSource, exactTarget, coveringTarget;
        auto key() const { return std::tie(exactSource, coveringSource, exactTarget, coveringTarget); }
        bool operator<(const Answers& other) const { return key() < other.key(); }
    };
    struct Node {
        Kind kind = Kind::Empty;
        Id left = 0, right = 0;
        DescriptorPredicate condition;
        // Leaf fields: facts is shared between its three descriptors; source
        // and target index independent boundary answers, never a menu product.
        Id facts = NoControlId, source = NoControlId, target = NoControlId;
        auto key() const { return std::tie(kind, left, right, condition, facts, source, target); }
        bool operator<(const Node& other) const { return key() < other.key(); }
    };
    struct SlotRoot {
        Slot first = Slot::Preferred;
        Id root = 0;
        // A duplicate slot is an alias, not another probe or another obligation.
        unsigned aliases = 0;
        // Unresolved slots remain in the analysis inventory, but are not
        // recipes to probe. This is original-data completeness, not legality
        // of an emitted packet or availability of selected proof contexts.
        bool fullyDescribed = false;
    };
    struct Stats {
        std::size_t inputs = 0, boundaries = 0, facts = 0, nodes = 0;
        std::size_t formationVisits = 0, roots = 0;
    };
    OriginalDescriptorSlots() { inputs.push_back({}); nodes.push_back({}); described.push_back(true); }
    Id boundary(DescriptorBoundary b)
    {
        if (!forming()) { return NoControlId; }
        const auto it = boundaryIndex.find(b);
        if (it != boundaryIndex.end()) { return it->second; }
        const Id id = boundaries.size();
        boundaryIndex.emplace(b, id);
        boundaries.push_back(std::move(b));
        return id;
    }
    Id leaf(DescriptorLeafFacts facts, Answers answers)
    {
        if (!forming()) { return NoControlId; }
        if (!validateBoundary(answers.exactSource, DescriptorBoundary::Direction::Source, false) ||
            !validateBoundary(answers.coveringSource, DescriptorBoundary::Direction::Source, true) ||
            !validateBoundary(answers.exactTarget, DescriptorBoundary::Direction::Target, false) ||
            !validateBoundary(answers.coveringTarget, DescriptorBoundary::Direction::Target, true) ||
            facts.obligations.empty()) {
            error = "invalid directional answer or missing original obligation";
            return NoControlId;
        }
        auto found = factIndex.find(facts);
        Id f = factsTable.size();
        if (found != factIndex.end()) {
            f = found->second;
        } else {
            factIndex.emplace(facts, f);
            factsTable.push_back(std::move(facts));
        }
        Input input;
        input.kind = Kind::Leaf;
        input.facts = f;
        input.answers = answers;
        return internInput(input);
    }
    Id both(Id a, Id b)
    {
        if (!requireInputs(a, b)) { return NoControlId; }
        if (!a || a == b) { return b; }
        if (!b) { return a; }
        if (b < a) { std::swap(a, b); }
        Input input;
        input.kind = Kind::Both; input.left = a; input.right = b;
        return internInput(input);
    }
    Id choose(DescriptorPredicate condition, Id a, Id b)
    {
        if (!requireInputs(a, b)) { return NoControlId; }
        if (condition.domain == DescriptorPredicate::Domain::True || a == b) { return a; }
        if (condition.domain == DescriptorPredicate::Domain::False) { return b; }
        Input input;
        input.kind = Kind::Choose; input.condition = condition; input.left = a; input.right = b;
        return internInput(input);
    }
    // These are preparation queries. freeze() removes the formation API from
    // the construction path; failed bindings cannot reopen it or add roots.
    std::vector<SlotRoot> form(Id root)
    {
        if (!forming()) { return {}; }
        if (root >= inputs.size()) {
            error = "unknown descriptor input root";
            return {};
        }
        // Topological formation visits every new shared input once across ALL
        // groups. The public builder requires children to precede their parent.
        while (formed.size() < inputs.size()) {
            const Id id = formed.size();
            const auto in = inputs[id];
            std::array<Id, 3> result{};
            ++work.formationVisits;
            if (in.kind == Kind::Leaf) {
                const auto a = in.answers;
                const Id source = preferred(a.exactSource, a.coveringSource);
                const Id target = preferred(a.exactTarget, a.coveringTarget);
                const std::array<std::pair<Id, Id>, 3> pairs{{
                    {source, target}, {a.coveringSource, target}, {source, a.coveringTarget}}};
                for (unsigned s = 0; s < 3; ++s) {
                    Node n; n.kind = Kind::Leaf; n.facts = in.facts;
                    n.source = pairs[s].first; n.target = pairs[s].second;
                    result[s] = internNode(n);
                }
            } else if (in.kind != Kind::Empty) {
                for (unsigned s = 0; s < 3; ++s) {
                    result[s] = combine(in.kind, in.condition, formed[in.left][s], formed[in.right][s]);
                }
            }
            formed.push_back(result);
        }
        std::vector<SlotRoot> unique;
        for (unsigned s = 0; s < 3; ++s) {
            const Id r = formed[root][s];
            auto it = std::find_if(unique.begin(), unique.end(), [r](const SlotRoot& entry) { return entry.root == r; });
            if (it == unique.end()) {
                unique.push_back({static_cast<Slot>(s), r, 1u << s, r != 0 && described[r]});
            } else {
                it->aliases |= 1u << s;
            }
        }
        work.roots += unique.size();
        return unique;
    }
    bool freeze()
    {
        if (!forming()) { return false; }
        // An unformed input would be a late-discovery hole in the repertoire.
        if (formed.size() != inputs.size()) {
            error = "unformed descriptor input at freeze";
            return false;
        }
        frozen = true;
        work.inputs = inputs.size(); work.boundaries = boundaries.size();
        work.facts = factsTable.size(); work.nodes = nodes.size();
        return true;
    }
    bool complete() const { return frozen && error.empty(); }
    const std::string& reason() const { return error; }
    const Stats& stats() const { return work; }
    const Node& node(Id id) const { return nodes.at(id); }
    const DescriptorBoundary& getBoundary(Id id) const { return boundaries.at(id); }
    const DescriptorLeafFacts& facts(Id id) const { return factsTable.at(id); }
    // This is only a structural readiness test, NOT a packet certificate.
    bool leafComponentsQualified(Id id) const
    {
        const auto& n = node(id);
        return n.kind == Kind::Leaf && facts(n.facts).matchingQualified && facts(n.facts).unresolved.empty() &&
               getBoundary(n.source).qualified() && getBoundary(n.target).qualified();
    }

private:
    struct Input {
        Kind kind = Kind::Empty;
        Id left = 0, right = 0, facts = NoControlId;
        DescriptorPredicate condition;
        Answers answers{NoControlId, NoControlId, NoControlId, NoControlId};
        auto key() const { return std::tie(kind, left, right, facts, condition, answers); }
        bool operator<(const Input& other) const { return key() < other.key(); }
    };
    bool forming() const { return !frozen && error.empty(); }
    bool requireInputs(Id a, Id b)
    {
        if (!forming()) { return false; }
        if (a >= inputs.size() || b >= inputs.size()) {
            error = "unknown descriptor child";
            return false;
        }
        return true;
    }
    bool validateBoundary(Id id, DescriptorBoundary::Direction direction, bool cover) const
    {
        if (id >= boundaries.size()) { return false; }
        const auto& b = boundaries[id];
        return b.direction == direction && !(b.form == DescriptorBoundary::Form::Exact && cover) &&
               !(b.form == DescriptorBoundary::Form::Covering && !cover);
    }
    Id preferred(Id exact, Id cover) const
    {
        if (boundaries[exact].qualified()) { return exact; }
        if (boundaries[cover].qualified()) { return cover; }
        // Keep the exact obstruction, and retain the independent cover in its
        // own declared slot. No missing answer is converted into an empty arm.
        return exact;
    }
    Id internInput(const Input& input)
    {
        const auto found = inputIndex.find(input);
        if (found != inputIndex.end()) { return found->second; }
        const Id id = inputs.size(); inputIndex.emplace(input, id); inputs.push_back(input); return id;
    }
    Id internNode(const Node& node)
    {
        const auto found = nodeIndex.find(node);
        if (found != nodeIndex.end()) { return found->second; }
        const Id id = nodes.size();
        nodeIndex.emplace(node, id);
        nodes.push_back(node);
        bool complete = true;
        if (node.kind == Kind::Leaf) {
            const auto& f = factsTable[node.facts];
            complete = f.matchingQualified && f.unresolved.empty() &&
                       boundaries[node.source].qualified() && boundaries[node.target].qualified();
        } else if (node.kind == Kind::Both || node.kind == Kind::Choose) {
            complete = described[node.left] && described[node.right];
        }
        described.push_back(complete);
        return id;
    }
    Id combine(Kind kind, DescriptorPredicate condition, Id a, Id b)
    {
        if (kind == Kind::Both) {
            if (!a || a == b) { return b; }
            if (!b) { return a; }
            if (b < a) { std::swap(a, b); }
        } else {
            if (a == b) { return a; }
            if (condition.domain == DescriptorPredicate::Domain::True) { return a; }
            if (condition.domain == DescriptorPredicate::Domain::False) { return b; }
        }
        Node n; n.kind = kind; n.condition = condition; n.left = a; n.right = b;
        return internNode(n);
    }
    bool frozen = false;
    std::string error;
    Stats work;
    std::vector<DescriptorBoundary> boundaries;
    std::map<DescriptorBoundary, Id> boundaryIndex;
    std::vector<DescriptorLeafFacts> factsTable;
    std::map<DescriptorLeafFacts, Id> factIndex;
    std::vector<Input> inputs;
    std::map<Input, Id> inputIndex;
    std::vector<Node> nodes;
    std::vector<bool> described;
    std::map<Node, Id> nodeIndex;
    std::vector<std::array<Id, 3>> formed;
};
} // namespace mlir::pto::frontiersynch
#endif
