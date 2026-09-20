// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SelectedInternal.h"
#include <algorithm>

namespace mlir::pto::oahs::selected {
namespace {
bool properSubset(const std::set<Id>& a, const std::set<Id>& b)
{
    return a.size() < b.size() && std::includes(b.begin(), b.end(), a.begin(), a.end());
}
} // namespace
std::map<Id, unsigned> Constructor::reasons(Cut site) const
{
    return requirements.reasons(site);
}
std::set<Id> Constructor::coverage(
    Cut cut, Pipe source, const std::vector<FrontierRequirement>& requirements, bool atStart) const
{
    std::set<Id> out;
    const auto& atSource = atStart ? cache.cuts[cut].incoming : cache.cuts[cut].before;
    if (!atSource.causal.reachable()) {
        return out;
    }
    const auto& history = atSource.causal.facts()->history;
    for (const auto& r : requirements) {
        const auto index = accessClass(r);
        const auto* reached = history.find(index);
        if (freshBetween(cut, current, index) && reached &&
            frontierContains(*reached, PipeCount + unsigned(source))) {
            out.insert(index);
        }
    }
    return out;
}
bool Constructor::freshBetween(Cut source, Cut target, Id access) const
{
    if (source == target) return true;
    if (!control.straight(source, target)) return false;
    return !control.lookahead.hasIssueBetween(
        control.frame[source], access, control.position[source], control.position[target]);
}
bool Constructor::sourceFrontier(
    Pipe source, const std::vector<FrontierRequirement>& required, Group& group,
    const std::vector<FrontierRequirement>& all, const std::set<Id>* promotion) const
{
    // This extension is intentionally acyclic: static predecessor identities
    // are not a bank-generation correspondence across an unqualified loop.
    if (required.empty() || control.components[activeComponent].cyclic ||
        control.canonicalCut[current] != current) return false;
    auto uniqueWord = [&](Cut cut) {
        if (cut >= control.canonicalCut.size() || control.canonicalCut[cut] != cut) return false;
        const auto& occurrences = control.wordOccurrences[cut];
        return std::count_if(occurrences.begin(), occurrences.end(),
                            [&](Id site) { return control.reachable[site]; }) == 1;
    };
    if (!uniqueWord(current)) return false;
    std::set<Id> needed;
    for (const auto& requirement : required) needed.insert(accessClass(requirement));
    std::map<Id, const SelectedSource*> availableSources;
    for (const auto& handle : result.sources) {
        if (handle.pipe == source && handle.version == cache.version && handle.snapshot.reachable())
            availableSources.emplace(handle.origin, &handle);
    }
    std::set<Cut> publications;
    std::set<Id> regenerated;
    std::vector<bool> seen(control.graph.sites.size());
    auto todo = control.predecessors[current];
    while (!todo.empty()) {
        const auto site = todo.back(); todo.pop_back();
        if (!control.reachable[site] || seen[site]) continue;
        seen[site] = true;
        const auto component = control.component[site];
        if (component == NoAnalysisId || control.components[component].cyclic) return false;
        const auto operation = control.graph.operations[site];
        bool touchesRequiredClass = false;
        if (operation != NoAnalysisId) {
            const auto& op = program.operations[operation];
            for (const auto& access : op.accesses) {
                const auto base = (Id(access.cell) * PipeCount + unsigned(op.pipe)) * 2;
                touchesRequiredClass |= (access.read && needed.count(base)) ||
                                        (access.write && needed.count(base + 1));
            }
        }
        if (touchesRequiredClass) {
            const auto found = availableSources.find(site);
            if (found == availableSources.end()) return false;
            const auto& handle = *found->second;
            if (handle.cut == current || !uniqueWord(handle.cut)) return false;
            const auto sourceComponent = control.component[handle.cut];
            if (sourceComponent == NoAnalysisId || control.components[sourceComponent].cyclic) return false;
            const auto& snapshot = cache.cuts[handle.cut].before.causal;
            if (!snapshot.reachable()) return false;
            // Do not assume that an absent class on one arm means an optional
            // corresponding producer. That case needs a separate qualifier.
            for (auto access : needed) {
                const auto* history = snapshot.facts()->history.find(access);
                if (!history || !frontierContains(*history, PipeCount + unsigned(source))) return false;
            }
            publications.insert(handle.cut);
            continue;
        }
        // These issues lie after a selected alternative publication on at
        // least one path. Exclude their classes from additional credit, even
        // if another arm's source snapshot contains an older occurrence.
        if (operation != NoAnalysisId) {
            const auto& op = program.operations[operation];
            for (const auto& access : op.accesses) {
                const auto base = (Id(access.cell) * PipeCount + unsigned(op.pipe)) * 2;
                if (access.read) regenerated.insert(base);
                if (access.write) regenerated.insert(base + 1);
            }
        }
        if (site == control.graph.entry || control.predecessors[site].empty()) return false;
        const auto& before = control.predecessors[site];
        todo.insert(todo.end(), before.begin(), before.end());
    }
    std::vector<Cut> cuts(publications.begin(), publications.end());
    if (!control.lookahead.balancedTransfer(cuts, current, control.graph.entry, control.graph.exit)) return false;
    auto covered = needed;
    for (const auto& r : all) {
        const auto access = accessClass(r);
        if (regenerated.count(access)) continue;
        if (std::all_of(cuts.begin(), cuts.end(), [&](Cut cut) {
                const auto* history = cache.cuts[cut].before.causal.facts()->history.find(access);
                return history && frontierContains(*history, PipeCount + unsigned(source));
            })) covered.insert(access);
    }
    if (promotion && std::none_of(promotion->begin(), promotion->end(),
            [&](Id access) { return covered.count(access); })) return false;
    // No edit/retry: only offer this vocabulary when an unused physical key has
    // its complete source-time certificate at ALL alternative publications.
    const auto observer = program.operations[control.graph.operations[current]].pipe;
    Id selected = NoAnalysisId;
    for (Id key = 0; key < frontier.keys().size(); ++key) {
        const auto& identity = frontier.keys()[key];
        if (identity.source != source || identity.observer != observer || closedKeys.count(key)) continue;
        const bool used = std::any_of(ledger.records().begin(), ledger.records().end(), [&](const auto& endpoint) {
            const auto& command = endpoint.command;
            return (command.kind == Command::Publish || command.kind == Command::Acquire) &&
                command.source == source && command.observer == observer && command.key == identity.key;
        });
        if (used || !std::all_of(cuts.begin(), cuts.end(), [&](Cut cut) {
                return canPublish(cache.cuts[cut].before, key);
            })) continue;
        selected = key;
        break;
    }
    if (selected == NoAnalysisId) return false;
    group.publications = std::move(cuts);
    group.publication = *std::min_element(group.publications.begin(), group.publications.end(),
        [&](Cut a, Cut b) { return control.position[a] < control.position[b]; });
    group.forwardKey = selected;
    group.version = cache.version;
    group.common = false;
    group.coverage = std::move(covered); // all paths, source-time credit, no regenerated class
    return true;
}
bool Constructor::loopEntryFrontier(
    Pipe source, const std::vector<FrontierRequirement>& required, Group& group,
    const std::vector<FrontierRequirement>& all, const std::set<Id>* promotion)
{
    if (!program.observed || required.empty()) return false;
    std::set<Id> needed;
    for (const auto& r : required) needed.insert(accessClass(r));
    const auto observer = program.operations[control.graph.operations[current]].pipe;
    for (const auto& loop : control.loopEntries) {
        std::vector<Cut> acquisitions{loop.entry};
        for (auto cut : loop.firstInputConsumers)
            if (program.operations[control.graph.operations[cut]].pipe == observer)
                acquisitions.push_back(cut);
        for (auto acquisition : acquisitions) {
            const bool atEntry = acquisition == loop.entry;
            const auto first = atEntry ? loop.firstConsumers[unsigned(observer)]
                                       : std::vector<Cut>{acquisition};
            // SCC construction may visit a later consumer before a qualified
            // first use. Select the receipt at that actual first deadline,
            // or at entry only when every first observer payload requires it.
            const bool eligible = !first.empty() &&
                std::find(loop.sites.begin(), loop.sites.end(), current) != loop.sites.end() &&
                control.canonicalCut[acquisition] == acquisition;
            if (!eligible) continue;
            if (!std::all_of(first.begin(), first.end(), [&](Cut cut) {
                const auto& op = program.operations[control.graph.operations[cut]];
                return std::all_of(required.begin(), required.end(), [&](const auto& r) {
                    return std::any_of(op.accesses.begin(), op.accesses.end(), [&](const auto& a) {
                        return a.cell == r.cell && (a.write || (a.read && r.sourceWrite));
                    });
                });
            })) continue;
            // Moving a wait ahead of Q's first payload can still order another
            // engine through an earlier Q publication. Consult actual selected
            // words, not just payload order. Existing entry commands precede the
            // appended acquisition; existing deadline commands follow it only if
            // we hoist, so the latter must also be checked.
            const bool communicates = atEntry && std::any_of(
                loop.crossedWords[unsigned(observer)].begin(),
                loop.crossedWords[unsigned(observer)].end(), [&](Cut cut) {
                    return std::any_of(ledger.word(cut).begin(), ledger.word(cut).end(), [&](Id id) {
                        const auto& command = ledger.endpoint(id).command;
                        return command.kind == Command::BarrierAll ||
                            (command.kind == Command::Publish && command.source == observer);
                    });
                });
            if (atEntry && communicates) continue;
            // No relevant source-class occurrence may be refreshed inside the
            // region. Readiness/release belongs to this bank, not to a maximum
            // operation number or to every operation on its source engine.
            if (std::any_of(needed.begin(), needed.end(), [&](Id access) {
                return loop.issuedClasses.count(access) != 0;
            })) continue;
            const SelectedSource* selected = nullptr;
            for (const auto& handle : result.sources) {
                if (handle.pipe != source || handle.version != cache.version || !handle.snapshot.reachable() ||
                    !control.straight(handle.cut, loop.entry)) continue;
                bool covered = true;
                const auto& snapshot = cache.cuts[handle.cut].before.causal;
                if (!snapshot.reachable()) continue;
                for (auto access : needed) {
                    const auto* history = snapshot.facts()->history.find(access);
                    covered &= freshBetween(handle.cut, loop.entry, access) && history &&
                        frontierContains(*history, PipeCount + unsigned(source));
                }
                if (covered && (!selected || control.position[handle.cut] < control.position[selected->cut]))
                    selected = &handle;
            }
            // With no saved source in this invocation, a source-inactive region
            // can establish its incoming completion at entry. All alternative first
            // observer payloads must need these same classes: a branch containing
            // unrelated observer work is not a reason to advance its deadline.
            const bool regional = atEntry && !selected && !loop.issuedPipes.count(source) &&
                std::none_of(loop.sites.begin(), loop.sites.end(), [&](Cut cut) {
                    return std::any_of(ledger.word(cut).begin(), ledger.word(cut).end(), [&](Id id) {
                        const auto& c = ledger.endpoint(id).command;
                        return c.kind == Command::BarrierAll || c.source == source ||
                            ((c.kind == Command::Publish || c.kind == Command::Acquire) && c.observer == source);
                    });
                });
            if (!selected && !regional) continue;
            const auto publication = selected ? selected->cut : loop.entry;
            if (!regional && !control.lookahead.balancedTransfer({publication}, acquisition,
                                           control.graph.entry, control.graph.exit)) continue;
            // Additional credit comes from the actual publication checkpoint,
            // never the consumer checkpoint. Invariance inside the region and
            // freshness on the incoming corridor preserve occurrence identity.
            auto covered = needed;
            const auto& snapshot = cache.cuts[publication].before.causal;
            if (snapshot.reachable()) for (const auto& r : all) {
                const auto access = accessClass(r);
                const auto* history = snapshot.facts()->history.find(access);
                if (!loop.issuedClasses.count(access) &&
                    freshBetween(publication, loop.entry, access) && history &&
                    frontierContains(*history, PipeCount + unsigned(source))) covered.insert(access);
            }
            // An overlap-only promotion without Known credit cannot succeed.
            // Reject it before key selection and the whole-program trial solve.
            if (promotion && std::none_of(promotion->begin(), promotion->end(),
                    [&](Id access) { return covered.count(access); })) continue;
            auto unused = [&](Pipe a, Pipe b) {
                for (Id key = 0; key < frontier.keys().size(); ++key) {
                    const auto& e = frontier.keys()[key];
                    if (e.source != a || e.observer != b || closedKeys.count(key) || recurringKeys.count(key)) continue;
                    if (std::none_of(ledger.records().begin(), ledger.records().end(), [&](const auto& r) {
                        if (!ledger.active(r.id)) return false;
                        const auto& c = r.command;
                        return (c.kind == Command::Publish || c.kind == Command::Acquire) &&
                            c.source == a && c.observer == b && c.key == e.key;
                    })) return key;
                }
                // Entry protocols may finish before a sibling scope. Reuse
                // their physical key only with actual empty/consumed credit at
                // the new publication, and no old use in this reader region.
                // The existing all-path trial below still checks every event
                // generation; lexical scope exit alone grants no ownership.
                const auto at = a == source ? publication : acquisition;
                for (auto key : entryProtocolKeys) {
                    const auto& e = frontier.keys()[key];
                    if (e.source != a || e.observer != b ||
                        !canPublish(cache.cuts[at].before, key)) continue;
                    const bool inRegion = std::any_of(ledger.records().begin(), ledger.records().end(),
                        [&](const auto& endpoint) {
                            const auto& c = endpoint.command;
                            return ledger.active(endpoint.id) &&
                                (c.kind == Command::Publish || c.kind == Command::Acquire) &&
                                c.source == a && c.observer == b && c.key == e.key &&
                                std::find(loop.sites.begin(), loop.sites.end(), endpoint.cut) != loop.sites.end();
                        });
                    if (!inRegion) return key;
                }
                return NoAnalysisId;
            };
            const auto forward = unused(source, observer);
            if (forward == NoAnalysisId) continue;
            const bool repeats = control.components[control.component[publication]].cyclic;
            Id reverse = NoAnalysisId;
            auto commands = ledger.commands();
            commands[publication].push_back({Command::Publish, source, observer, frontier.keys()[forward].key});
            commands[acquisition].push_back({Command::Acquire, source, observer, frontier.keys()[forward].key});
            auto trial = analyze(program, commands, {false});
            result.work.loopEntryAnalysisSites += trial.stats.siteEvaluations;
            const bool needsConsumption = std::any_of(trial.protocol.begin(), trial.protocol.end(), [&](const auto& r) {
                return r.kind == ProtocolObligation::ConsumptionNotEstablished &&
                    r.event.source == source && r.event.observer == observer &&
                    r.event.key == frontier.keys()[forward].key;
            });
            // Existing causal paths get the first opportunity to prove reuse. A
            // return is justified by this key's missing consumption certificate,
            // not merely by being textually inside a repeated component.
            if (trial.complete && trial.diagnostics.empty() && repeats && needsConsumption) {
                reverse = unused(observer, source);
                if (reverse == NoAnalysisId) continue;
                commands[acquisition].push_back({Command::Publish, observer, source, frontier.keys()[reverse].key});
                commands[acquisition].push_back({Command::Acquire, observer, source, frontier.keys()[reverse].key});
                trial = analyze(program, commands, {false});
                result.work.loopEntryAnalysisSites += trial.stats.siteEvaluations;
            }
            if (!trial.complete || !trial.diagnostics.empty() || !trial.protocol.empty()) continue;
            group.publication = publication;
            group.publications = {publication};
            group.entryAcquisition = acquisition;
            group.entryReturnKey = reverse;
            group.entryRepeats = repeats;
            group.forwardKey = forward;
            group.version = ledger.version();
            group.coverage = std::move(covered);
            return true;
        }
    }
    return false;
}
Group Constructor::sourceGroup(
    Pipe source, const std::vector<FrontierRequirement>& required,
    const std::vector<FrontierRequirement>& all, const std::set<Id>* promotion)
{
    Group group;
    group.source = source;
    group.requirements = required;
    std::set<Id> needed;
    for (const auto& requirement : required) needed.insert(accessClass(requirement));
    const SelectedSource* selected = nullptr;
    for (const auto& handle : result.sources) {
        if (handle.pipe != source || handle.version != cache.version || !handle.snapshot.reachable() ||
            !control.straight(handle.cut, current)) continue;
        // A shared word also executes on its later analytical occurrences.
        // A source available only on the first-prefix corridor cannot publish
        // once for an unconditional acquisition on all those visits.
        const auto& occurrences = control.wordOccurrences[control.canonicalCut[current]];
        const auto& publications = control.wordOccurrences[control.canonicalCut[handle.cut]];
        if (control.firstPrefixWords.count(control.canonicalCut[current]) &&
            std::any_of(occurrences.begin(), occurrences.end(), [&](Cut cut) {
                return control.reachable[cut] &&
                    std::none_of(publications.begin(), publications.end(), [&](Cut publication) {
                        return control.reachable[publication] && control.straight(publication, cut);
                    });
            })) continue;
        const auto covered = coverage(handle.cut, source, required);
        if (!std::includes(covered.begin(), covered.end(), needed.begin(), needed.end())) continue;
        if (!selected || control.position[handle.cut] < control.position[selected->cut]) selected = &handle;
    }
    // A view of the existing deadline-indexed storage frontiers supplies the
    // useful original publication boundary. Actual coverage is queried at the
    // exact post-origin gap; immutable provenance supplies no causal credit.
    const auto observer = program.operations[control.graph.operations[current]].pipe;
    if (options.sourceGaps && selected && std::any_of(requirements.at(current).begin(),
            requirements.at(current).end(), [&](const auto& fact) {
                return fact.publication == selected->cut && fact.source == source;
            })) {
        const auto covered = coverage(selected->cut, source, all, true);
        const auto key = virginAtStart(selected->cut, source, observer);
        if (key != NoAnalysisId && std::includes(covered.begin(), covered.end(), needed.begin(), needed.end())) {
            group.publication = selected->cut;
            group.atWordStart = true;
            group.forwardKey = key;
            group.version = cache.version;
            group.coverage = covered;
            return group;
        }
    }
    const bool comparable = selected != nullptr;
    if (!comparable && sourceFrontier(source, required, group, all, promotion)) return group;
    if (!comparable && loopEntryFrontier(source, required, group, all, promotion)) return group;
    group.publication = comparable ? selected->cut : current;
    group.common = !comparable;
    group.coverage = coverage(group.publication, source, all);
    return group;
}
std::vector<Group> Constructor::groups(
    const std::vector<FrontierRequirement>& all, RequirementStage stage)
{
    const auto labels = reasons(current);
    const auto observer = program.operations[control.graph.operations[current]].pipe;
    std::map<Pipe, std::vector<FrontierRequirement>> sources, overlapSources;
    std::set<Id> known;
    for (const auto& r : all) {
        const auto found = labels.find(accessClass(r));
        const auto flags = found == labels.end() ? unsigned(AdditionalOverlap) : found->second;
        if (r.source == observer) continue;
        if (stage == RequirementStage::Known && !(flags & (KnownReadiness | KnownReuse))) {
            overlapSources[r.source].push_back(r);
            continue;
        }
        sources[r.source].push_back(r);
        known.insert(accessClass(r));
    }
    std::vector<Group> pending, ordered;
    for (const auto& source : sources) {
        pending.push_back(sourceGroup(source.first, source.second, all));
    }
    // A required overlap transfer may already carry a known reuse/readiness
    // prerequisite through selected handoffs. Let that actual provider compete
    // now instead of first installing a duplicate direct return. Keep an
    // existing known source's early prefix: do not widen it with its later
    // overlap demands. No future transfer is credited by this query.
    if (!known.empty()) for (const auto& source : overlapSources) {
        if (sources.count(source.first)) continue;
        auto group = sourceGroup(source.first, source.second, all, &known);
        if (std::any_of(known.begin(), known.end(), [&](Id access) {
                return group.coverage.count(access) != 0;
            })) pending.push_back(std::move(group));
    }
    while (!pending.empty()) {
        Id selected = NoAnalysisId;
        for (Id i = 0; i < pending.size(); ++i) {
            bool maximal = true;
            for (const auto& other : pending) {
                maximal &= !properSubset(pending[i].coverage, other.coverage);
            }
            if (!maximal) {
                continue;
            }
            if (selected == NoAnalysisId) {
                selected = i;
                continue;
            }
            const auto a = pending[i].publication, b = pending[selected].publication;
            const bool commonFrame = control.straight(a, b) || control.straight(b, a);
            if ((commonFrame && control.position[a] > control.position[b]) ||
                ((!commonFrame || a == b) && std::make_pair(a, pending[i].source) <
                                                std::make_pair(b, pending[selected].source))) {
                selected = i;
            }
        }
        const auto baseline = selected;
        // Audit the exact candidate population before proposing a new policy.
        // Equal sizes are not equal certified residual sets.
        for (Id i = 0; i < pending.size(); ++i)
            if (i != selected && !pending[selected].coverage.empty() &&
                pending[i].coverage == pending[selected].coverage)
                ++result.work.equalCoveragePairs;
        if (options.equalCoverageBinding && std::any_of(pending.begin(),pending.end(),[&](const auto& g) {
                return &g != &pending[baseline] && !g.coverage.empty() && g.coverage == pending[baseline].coverage;
            })) {
            ++result.work.bindingProbes;
            // Retain the original winner if already helper-free. Probe only
            // equal SETS in this same priority population; do not mutate state.
            if (helperFreeBinding(pending[baseline], observer) == NoAnalysisId) {
                Id best = NoAnalysisId, binding = NoAnalysisId;
                for (Id i = 0; i < pending.size(); ++i) {
                    if (i == baseline || pending[i].coverage != pending[baseline].coverage) continue;
                    ++result.work.bindingProbes;
                    const auto key = helperFreeBinding(pending[i], observer);
                    if (key == NoAnalysisId) continue; // Unknown, not Impossible
                    bool better = best == NoAnalysisId;
                    if (!better) {
                        const auto a = pending[i].publication, b = pending[best].publication;
                        const bool frame = control.straight(a,b) || control.straight(b,a);
                        better = (frame && control.position[a] > control.position[b]) ||
                            ((!frame || a == b) && std::make_pair(a,pending[i].source) <
                                                      std::make_pair(b,pending[best].source));
                    }
                    if (better) { best = i; binding = key; }
                }
                if (best != NoAnalysisId) {
                    selected = best;
                    pending[best].forwardKey = binding;
                    pending[best].version = ledger.version();
                    pending[best].bindingCertified = true;
                    ++result.work.bindingChoices;
                }
            }
        }
        ordered.push_back(std::move(pending[selected]));
        break; // The next actual receipt invalidates the remaining ranking.
    }
    return ordered;
}
bool Constructor::consume()
{
    const auto operation = control.graph.operations[current];
    if (operation == NoAnalysisId || !currentState().causal.reachable()) {
        return true;
    }
    const auto observer = program.operations[operation].pipe;
    auto crossCount = [&](const std::vector<FrontierRequirement>& values) {
        return std::count_if(values.begin(), values.end(),
            [&](const auto& r) { return r.source != observer; });
    };
    for (auto stage : {RequirementStage::Known, RequirementStage::Overlap}) {
        while (true) {
            const auto before = residual();
            auto requests = groups(before, stage);
            if (requests.empty()) break;
            if (!bind(requests.front(), stage)) return false;
            // A real acquisition can change which remaining prefix is best.
            // Re-form groups from the new frontier; never retain a stale source
            // requirement merely because it was in an earlier candidate list.
            // No payload was issued, so strict residual decrease is the finite
            // progress measure, not an iteration limit or a retry budget.
            if (crossCount(residual()) >= crossCount(before)) {
                return fail(SelectedFailure::MissingParticipation,
                    "selected transfer did not reduce the cross-engine residual", current);
            }
        }
    }
    const auto missing = residual();
    for (const auto& r : missing) {
        if (r.source != observer) {
            return fail(SelectedFailure::MissingParticipation,
                "cross-engine requirement survived its selected transfer", current);
        }
    }
    if (!missing.empty()) {
        if (!program.target.barriers[unsigned(observer)]) {
            return fail(SelectedFailure::UnsupportedContract,
                "no qualified named fence for the remaining same-engine requirement", current);
        }
        const auto version = ledger.version();
        result.fences.push_back({current, observer, version, missing});
        // Repair this occurrence at its actual deadline. A future analytical
        // copy may receive completion from transfers selected before it is
        // reached. Ledger canonicalization still emits one command when
        // several analytical sites genuinely share the same authored word.
        ledger.append(current, {Command::Barrier, observer, Pipe::S, 0},
                      EndpointPurpose::LocalFence);
        if (!update()) {
            return false;
        }
    }
    const auto checked = frontier.inspect(currentState().causal, operation);
    if (!checked.applied) {
        return fail(SelectedFailure::FinalValidation, checked.reason, current);
    }
    return true;
}
} // namespace mlir::pto::oahs::selected
