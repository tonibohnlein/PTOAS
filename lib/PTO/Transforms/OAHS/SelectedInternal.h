// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_OAHS_SELECTED_INTERNAL_H
#define PTO_OAHS_SELECTED_INTERNAL_H

#include "Control.h"
#include "PTO/Transforms/OAHS/SelectedPlan.h"
#include <map>
#include <set>

namespace mlir::pto::oahs::selected {

using Id = std::size_t;
struct Component {
    std::vector<Id> sites, order, entries;
    bool cyclic = false;
};
struct Control {
    detail::ControlGraph graph;
    std::vector<std::vector<Id>> predecessors;
    // Construction-only, zero-or-more loop summary traversal. Acceptance uses
    // graph's original edges, not these exit-summary edges or hypotheses.
    std::vector<std::vector<Id>> constructionEdges, headerAccesses;
    std::vector<Component> components;
    std::vector<Id> component, position, frame;
    std::vector<bool> reachable;
    bool complete = false, validInput = false;
    std::string reason;
    explicit Control(const Program&);
    bool straight(Id, Id) const;
    Cut after(Id) const;
};

class Ledger {
public:
    explicit Ledger(const Program&);
    bool initialize(const Commands&, std::string&);
    Id append(Cut, Command, EndpointPurpose, Id request = NoAnalysisId, Id ack = NoAnalysisId);
    Id after(Id, Command, EndpointPurpose, Id request, Id ack);
    const std::vector<Id>& word(Cut) const;
    const SelectedEndpoint& endpoint(Id) const;
    Commands commands() const;
    uint64_t version() const { return revision; }
    const std::vector<SelectedEndpoint>& records() const { return endpoints; }
    const std::vector<Cut>& changes() const { return changed; }
    void clearChanges() { changed.clear(); }

private:
    const Program& program;
    uint64_t revision = 0;
    std::vector<std::vector<Id>> words;
    std::vector<SelectedEndpoint> endpoints;
    std::vector<Cut> changed;
    Id insert(Cut, Id, Command, EndpointPurpose, Id, Id);
};

struct State {
    FrontierState causal;
    // Last original static origin, or unknown after an unqualified recurrence
    // or an incompatible choice. These names never become runtime predicates.
    std::vector<Id> latest;
    std::vector<std::vector<Id>> consumptions;
};
struct Checkpoint {
    State incoming, before, outgoing;
};
struct Replay {
    uint64_t version = 0, evaluations = 0;
    bool success = true;
    Cut failureCut = NoAnalysisId;
    std::string reason;
    std::vector<Checkpoint> cuts;
    std::map<Id, State> afterEndpoint;
    // Leading components whose cuts hold their actual fixed point rather than
    // the construction-only hypothesis traversal of the active component. Only
    // those may be reused by a later replay; everything else is recomputed.
    std::size_t fixedComponents = 0, reusedComponents = 0;
};
struct RecurringRequirement {
    unsigned cell = 0;
    Pipe source = Pipe::S, observer = Pipe::S;
    std::vector<Cut> publications, acquisitions;
    Id owner = NoAnalysisId;
};
// A storage/control qualifier: it returns requirements and original frontiers,
// not commands or physical key choices. Empty means ordinary F1--F8 applies.
std::vector<RecurringRequirement> qualifyCyclicFrontiers(const Program&, const Control&);

struct Group {
    Pipe source = Pipe::S;
    Cut publication = NoAnalysisId;
    std::vector<FrontierRequirement> requirements;
    std::set<Id> coverage;
    bool common = false;
};

class Constructor {
    friend struct ReplayTestAccess;
public:
    explicit Constructor(const Program&);
    SelectedPlan run(const Commands&);

private:
    const Program& program;
    CausalFrontier frontier;
    Control control;
    StorageFrontierAnalysis storage;
    Ledger ledger;
    SelectedPlan result;
    Replay cache;
    std::vector<bool> finalized;
    Id activeComponent = 0, activeOffset = 0;
    Cut current = NoAnalysisId;
    // Roles are stable within a selected loop component. A reverse acknowledgment
    // key must not be borrowed as another recurring channel's forward key.
    std::map<std::pair<Pipe, Pipe>, std::pair<Id, Id>> closedBindings;
    std::set<Id> closedKeys, recurringKeys;

    bool fail(SelectedFailure, std::string, Cut = NoAnalysisId);
    State initial() const;
    bool join(State&, const State&);
    bool word(State&, Cut, Replay&);
    bool payload(State&, Cut, Replay&, bool pending = false);
    bool contextualReplay();
    bool fixedComponent(Id, const std::vector<State>&, Replay&, std::vector<State>&);
    bool partialComponent(Id, const std::vector<State>&, Replay&);
    bool replay();
    bool advance();
    void refreshSources();
    bool update();
    State& currentState();
    std::vector<FrontierRequirement> residual() const;
    std::map<Id, unsigned> reasons(Cut) const;
    std::vector<Group> groups(const std::vector<FrontierRequirement>&, RequirementStage);
    Group sourceGroup(Pipe, const std::vector<FrontierRequirement>&,
                      const std::vector<FrontierRequirement>&);
    std::set<Id> coverage(Cut, Pipe, const std::vector<FrontierRequirement>&) const;
    bool consume();
    bool bind(Group&, RequirementStage);
    bool edge(Pipe, Pipe, Cut&, bool, SelectedDecision&);
    bool acknowledgment(Pipe, Pipe, Cut&, Id&, SelectedDecision&);
    Id reusable(Pipe, Pipe, const State&);
    bool canPublish(const State&, Id) const;
    bool clearInterval(Id, Cut, Cut) const;
    std::vector<Pipe> route(Pipe, Pipe) const;
    bool recurring(const std::vector<RecurringRequirement>&);
    bool finish();
};

Id accessClass(const FrontierRequirement&);
bool identical(const Command&, const Command&);
std::vector<Id> unionIds(const std::vector<Id>&, const std::vector<Id>&);

} // namespace mlir::pto::oahs::selected
#endif
