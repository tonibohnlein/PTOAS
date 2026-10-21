// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Probe the production-selected mathematics; reference effects live in Python.
#include "CompactRegionCoordinates.h"
#include "CompactRegionDemands.h"
#include "PTO/Transforms/FrontierSynch/Arithmetic/ArithmeticRegional.h"
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/FrontierSynch/FrontierAnalysis.h"
#include "PTO/Transforms/FrontierSynch/Shared/RetainedInventory.h"
#include "PTO/Transforms/FrontierSynch/Shared/PhysicalAllocation.h"
#include "PTO/Transforms/FrontierSynch/Shared/FamilyInsertion.h"
#include "PTO/Transforms/FrontierSynch/Shared/RegionalAllocation.h"
#include "PTO/Transforms/FrontierSynch/Shared/RegionalLaneExports.h"
#include "../../lib/PTO/Transforms/FrontierSynch/WholeInvocationIdentity.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/CompletionInlineChild.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/raw_ostream.h"
#include <set>
mlir::LogicalResult runGroupedCompletionParent(mlir::func::FuncOp function,
    std::shared_ptr<const mlir::pto::frontiersynch::MathematicalResult> math);
namespace {
using namespace compact_contract;
llvm::json::Array originalStructure(const fs::FrontierAnalysis& session)
{
    llvm::json::Array rows;
    const auto* program = session.result();
    if (!program) { return rows; }
    for (std::size_t id = 0; id < program->nodes.size(); ++id) {
        const auto& node = program->nodes[id];
        std::string location;
        if (node.anchor) { llvm::raw_string_ostream(location) << node.anchor->getLoc(); }
        rows.push_back(llvm::json::Object{{"region", id}, {"location", location},
            {"enclosing_coordinates", node.loops.size()}, {"static_payloads", node.payloadCount}});
    }
    return rows;
}
llvm::json::Object arithmeticSummary(const fs::MathematicalResult& math)
{
    auto owner = math.arithmeticRegionalDemands;
    if (!owner) {
        return {};
    }
    const auto& form = owner->program;
    uint64_t pieces = 0;
    if (owner->differenceDemands) {
        for (const auto& entry : owner->differenceDemands->minimumDemands) {
            pieces += entry.second.size();
        }
    } else {
        for (const auto& entry : owner->analysis.minimumDemands) {
            pieces += entry.second.size();
        }
    }
    return llvm::json::Object{
        {"sites", form.sites.size()},
        {"parameters", form.parameters.size()},
        {"primitive_relations", form.primitives.relations.size()},
        {"retained_pieces", pieces},
        {"class", fs::recognitionName(form.recognition.arithmeticClass)},
        {"observed_dimensions", form.recognition.observedDimensions},
        {"observed_coefficient", form.recognition.observedCoefficient},
        {"difference_bound_reducer", bool(owner->differenceDemands)}};
}
llvm::json::Object expandedPointChecks(const fs::MathematicalResult& math)
{
    const auto analysis = math.expandedFiniteDemands;
    if (!analysis || !analysis->expandedProgram || !analysis->state) {
        return llvm::json::Object{{"error", "expanded owner missing"}};
    }
    const auto program = analysis->expandedProgram;
    if (program->sites.empty() || program->sites.front().fixedCoordinates.empty()) {
        return llvm::json::Object{{"error", "fixed occurrence missing"}};
    }
    auto& state = *analysis->state;
    auto valid = state.prepare({}, program.get());
    auto invalid = *program;
    invalid.sites.front().fixedCoordinates.clear();
    const bool missing = failed(state.prepare({}, &invalid));
    invalid = *program;
    invalid.sites.front().fixedCoordinates.push_back(invalid.sites.front().fixedCoordinates.front());
    const bool duplicate = failed(state.prepare({}, &invalid));
    invalid = *program;
    invalid.sites.front().fixedCoordinates.front().induction = -1;
    const bool negative = failed(state.prepare({}, &invalid));
    auto loop = program->sites.front().fixedCoordinates.front().loop;
    const auto local = state.arena->eq(state.arena->input(loop.getInductionVar()), state.arena->constant(0));
    fs::RegionalDemandFilter changing = [local](const auto&, const auto&) {
        return std::optional<fs::RegionExpressions::Id>(local);
    };
    const bool history = state.retained.empty() || failed(state.prepare(changing, program.get()));
    auto again = state.prepare({}, program.get());
    const auto no = state.arena->boolean(false);
    fs::RegionalDemandFilter filter = [no](const auto&, const auto&) {
        return std::optional<fs::RegionExpressions::Id>(no);
    };
    auto empty = state.prepare(filter, program.get());
    return llvm::json::Object{
        {"valid", succeeded(valid)},
        {"missing_frame_refused", missing},
        {"duplicate_frame_refused", duplicate},
        {"negative_point_refused", negative},
        {"changing_activity_refused", history},
        {"retained_owner_reusable", succeeded(again)},
        {"false_filter_empty", succeeded(empty) && (*empty)->families.empty()},
        {"families", succeeded(valid) ? (*valid)->families.size() : 0}};
}
std::vector<uint64_t> constructionSnapshot(const fs::FrontierAnalysis& session)
{
    const auto& c = session.constructionCounts();
    return {c.structuralIndices,         c.mathematicalAttempts,      c.requestWalks,
            c.exhaustedWalkHits,         c.explicitReductions,        c.rotatingReductions,
            c.guardedRotatingReductions, c.varyingBoundaryReductions, c.varyingQueryBuilds,
            c.varyingSelectorBuilds,     c.boundedQueryBuilds,        c.boundedSelectorBuilds,
            c.arithmeticQueryBuilds,     c.arithmeticSelectorBuilds,  c.expandedSelectorBuilds,
            c.expandedSelectorChecks,    c.logicalPreparations,       c.allocationExports};
}
std::vector<uint64_t> retainedSnapshot(const fs::MathematicalResult& initial)
{
    auto* math = &initial;
    while (math->identityComposition) {
        math = math->identityComposition->child.get();
    }
    std::vector<const fs::RegionalAnalysis*> pending;
    std::vector<uint64_t> result;
    std::set<const void*> seen;
    auto sequence = [&](const fs::SequenceAnalysisState& state) {
        result.insert(result.end(), {state.expressions.size(), state.reachabilityCache.size()});
        for (const auto& child : state.children) {
            pending.push_back(&child.regional);
        }
    };
    if (math->sequenceDemands && math->sequenceDemands->state) {
        sequence(*math->sequenceDemands->state);
    } else if (math->regionalDemands) {
        pending.push_back(&*math->regionalDemands);
    }
    while (!pending.empty()) {
        auto* region = pending.back();
        pending.pop_back();
        result.push_back(region->expressions ? region->expressions->size() : 0);
        if (auto state = region->retainedRepetition.lock()) {
            if (!seen.insert(state.get()).second) {
                continue;
            }
            result.insert(
                result.end(), {state->queryMemo.size(), state->bodyPortMemo.size(), state->distanceMemo.size()});
            pending.push_back(&state->body);
        } else if (auto state = region->retainedSequence.lock()) {
            if (seen.insert(state.get()).second) {
                sequence(*state);
            }
        }
    }
    return result;
}
llvm::json::Value certificateJSON(Attribute attribute)
{
    if (auto dictionary = dyn_cast<DictionaryAttr>(attribute)) {
        llvm::json::Object result;
        for (auto entry : dictionary) { result[entry.getName().str()] = certificateJSON(entry.getValue()); }
        return result;
    }
    if (auto array = dyn_cast<ArrayAttr>(attribute)) {
        llvm::json::Array result;
        for (auto entry : array) { result.push_back(certificateJSON(entry)); }
        return result;
    }
    if (auto array = dyn_cast<DenseI64ArrayAttr>(attribute)) {
        llvm::json::Array result;
        for (auto entry : array.asArrayRef()) { result.push_back(entry); }
        return result;
    }
    if (auto integer = dyn_cast<IntegerAttr>(attribute)) { return integer.getInt(); }
    if (auto string = dyn_cast<StringAttr>(attribute)) { return string.getValue().str(); }
    std::string text;
    llvm::raw_string_ostream(text) << attribute;
    return text;
}
std::vector<fs::RegionalAllocationLane> suppliedLanes(const fs::PreparedLogicalPlan& plan)
{
    std::vector<fs::RegionalAllocationLane> lanes;
    for (auto group : plan.regionalAllocation->groups) {
        fs::exportConstantRegionalLanes(group);
        if (group.lanes.empty()) {
            fs::RegionalAllocationLane envelope;
            for (const auto& member : group.members) {
                envelope.firstSources.push_back({member.firstSource, member.active});
                envelope.lastTargets.push_back({member.lastTarget, member.active});
            }
            group.lanes.assign(group.budget, envelope);
        }
        lanes.insert(lanes.end(), group.lanes.begin(), group.lanes.end());
    }
    return lanes;
}
// Explicit proof inventory. False means unproved, never "this lane is active".
llvm::json::Array laneActivityProbe(func::FuncOp function, fs::SequenceAnalysis& sequence,
    const fs::PreparedLogicalPlan& plan)
{
    llvm::json::Array rows;
    if (!function->hasAttr("test.contract_lane_activity") || !plan.regionalAllocation) { return rows; }
    auto& e = *fs::sequenceRegionalResult(sequence).expressions;
    auto lanes = suppliedLanes(plan);
    for (auto [index, lane] : llvm::enumerate(lanes)) {
        auto present = e.boolean(false);
        for (const auto& source : lane.firstSources) { present = e.lor(present, source.present); }
        rows.push_back(llvm::json::Object{{"lane", index},
            {"proved_inactive", e.impliesByIntervals(present, e.boolean(false))}});
    }
    return rows;
}
// Explicit diagnostic only: construct the existing sufficient query for a
// named pair of supplied lanes. It never changes evidence or selects a plan.
llvm::json::Array lanePairProbe(func::FuncOp function, fs::SequenceAnalysis& sequence,
    const fs::PreparedLogicalPlan& plan)
{
    auto pair = function->getAttrOfType<DenseI64ArrayAttr>("test.contract_lane_pair");
    llvm::json::Array rows;
    if (!pair || pair.size() != 2 || !plan.regionalAllocation) { return rows; }
    auto lanes = suppliedLanes(plan);
    if (pair[0] < 0 || pair[1] < 0 || uint64_t(pair[0]) >= lanes.size() ||
        uint64_t(pair[1]) >= lanes.size()) { return rows; }
    const auto& region = fs::sequenceRegionalResult(sequence);
    auto& e = *region.expressions;
    auto text = [&](fs::RegionExpressions::Id id) {
        std::string result;
        llvm::raw_string_ostream out(result);
        e.printExpression(id, out);
        return result;
    };
    auto inactive = [&](const auto& selectors) {
        auto active = e.boolean(false);
        for (const auto& selector : selectors) { active = e.lor(active, selector.present); }
        return e.impliesByIntervals(active, e.boolean(false));
    };
    const bool lastInactive = inactive(lanes[pair[0]].lastTargets);
    const bool firstInactive = inactive(lanes[pair[1]].firstSources);
    for (const auto& last : lanes[pair[0]].lastTargets) {
        for (const auto& first : lanes[pair[1]].firstSources) {
            auto active = e.land(last.present, first.present);
            auto query = fs::regionalHandoffReuse(region, last.event, first.event);
            fs::RegionExpressions::ThresholdProofCost intervalCost;
            const bool interval = query && e.impliesByIntervals(active, *query, &intervalCost);
            rows.push_back(llvm::json::Object{{"consumer_type", last.event.type},
                {"producer_type", first.event.type}, {"active", text(active)},
                {"query_available", bool(query)}, {"query", query ? text(*query) : "unavailable"},
                {"implies", query && e.implies(active, *query)},
                {"inactive", e.constantUnder(active, active) == 0},
                {"constant_under", query && e.constantUnder(active, *query) == 1},
                {"threshold_implies", query && e.inactiveByThresholds(e.land(active, e.lnot(*query)))},
                {"threshold_inactive", e.inactiveByThresholds(active)},
                {"interval_implies", interval}, {"interval_cells", intervalCost.cells},
                {"interval_dag_visits", intervalCost.dagVisits},
                {"last_lane_inactive", lastInactive}, {"first_lane_inactive", firstInactive}});
        }
    }
    return rows;
}
// Unlike retained inventory, these explicit probes construct endpoints and
// validate an existing six-ID certificate. They never publish a child import.
void preparedCompletionProbe(func::FuncOp function, fs::SequenceAnalysis& sequence,
    fs::PreparedLogicalPlan& plan, llvm::json::Object& details)
{
    details["lane_pair_probe"] = lanePairProbe(function, sequence, plan);
    details["lane_activity_probe"] = laneActivityProbe(function, sequence, plan);
    details["allocation_certificate"] = bool(plan.allocationCertificate);
    if (plan.allocationCertificate) {
        std::string assembly;
        llvm::raw_string_ostream(assembly) << plan.allocationCertificate;
        details["certificate_assembly"] = assembly;
        details["certificate"] = certificateJSON(plan.allocationCertificate);
    }
    details["private_state"] = bool(plan.privateState);
    details["prepared_complete_invocation"] = plan.completeInvocation;
    details["families"] = plan.families.size();
    details["endpoints"] = plan.endpoints.size();
    details["materialized"] = succeeded(fs::materializeLogicalFamilies(plan));
    details["materialized_endpoints"] = plan.endpoints.size();
    unsigned choices = 0, outside = 0, multiple = 0;
    auto loop = dyn_cast_or_null<scf::ForOp>(sequence.state->constructionAnchor);
    for (const auto& family : plan.families) {
        choices += bool(family.sourceChoices) || bool(family.targetChoices);
    }
    for (const auto& endpoint : plan.endpoints) {
        outside += !loop || !loop->isProperAncestor(endpoint.before);
        multiple += endpoint.records.size() != 1;
    }
    details["choice_families"] = choices;
    details["outside_endpoints"] = outside;
    details["nonsingleton_endpoints"] = multiple;
    llvm::json::Array shapes;
    for (const auto& endpoint : plan.endpoints) {
        if (endpoint.kind == fs::LogicalCommandKind::Barrier) { continue; }
        llvm::json::Array records;
        for (auto record : endpoint.records) { records.push_back(record); }
        shapes.push_back(llvm::json::Object{{"records", std::move(records)},
            {"arity", endpoint.memberCoordinates.size()}, {"kind", static_cast<unsigned>(endpoint.kind)}});
    }
    details["endpoint_shapes"] = std::move(shapes);
    auto assignment = fs::decodePhysicalAllocationCertificate(function,
        plan.allocationCertificate, fs::defaultEventIds());
    details["physical_certificate_decoded"] = succeeded(assignment);
    llvm::json::Array physical;
    if (succeeded(assignment)) {
        for (const auto& record : assignment->records) {
            llvm::json::Array ids, terms;
            for (auto id : record.ids) { ids.push_back(id); }
            llvm::json::Object row{{"record", record.record}, {"source", record.sourcePipe},
                {"target", record.targetPipe}, {"ids", std::move(ids)},
                {"stride", record.stride}, {"phase", record.phase}};
            if (record.tupleRule) {
                const auto& rule = *record.tupleRule;
                for (const auto& term : rule.terms) {
                    terms.push_back(llvm::json::Array{term.coordinate, term.stride,
                        term.phase, term.modulus, term.scale});
                }
                row["tuple"] = llvm::json::Object{{"arity", rule.coordinateCount},
                    {"base", rule.base}, {"terms", std::move(terms)}};
            }
            physical.push_back(std::move(row));
        }
    }
    details["physical_assignment"] = std::move(physical);
}
FailureOr<std::unique_ptr<fs::PreparedLogicalPlan>> boundCompletionProbe(fs::SequenceAnalysis& sequence)
{
    auto prepared = fs::prepareSequenceLogicalInsertion(sequence, sequence.state->requiredOuterLoops);
    if (succeeded(prepared)) {
        fs::prepareAllocationSupport(**prepared);
        if (!(*prepared)->allocationCertificate) {
            (*prepared)->allocationCertificate = fs::regionalAllocationCertificate(
                fs::sequenceRegionalResult(sequence), **prepared);
        }
    }
    return prepared;
}
llvm::json::Object completionPreparationProbe(func::FuncOp function,
    const fs::MathematicalResult& mathematics)
{
    llvm::json::Object details{{"backend", mathematics.backend},
        {"completion_export", bool(mathematics.completionChild)},
        {"inventory_before_preparation", fs::retainedAnalysisInventory(mathematics)}};
    auto sequence = mathematics.sequenceDemands;
    if (!sequence || !sequence->state) { return details; }
    details["required_outer_loops"] = sequence->state->requiredOuterLoops.size();
    if (function->hasAttr("test.contract_completion_default")) {
        auto prepared = fs::prepareSequenceInsertion(*sequence);
        details["default_frame_prepared"] = succeeded(prepared);
    }
    details["complete_invocation"] = sequence->state->completeInvocation;
    auto prepared = fs::prepareSequenceLogicalInsertion(*sequence);
    details["unbound_preparation_error"] = sequence->insertionError;
    if (function->hasAttr("test.contract_completion_bound")) { prepared = boundCompletionProbe(*sequence); }
    details["preparation_error"] = sequence->insertionError;
    details["prepared"] = succeeded(prepared);
    if (succeeded(prepared)) { preparedCompletionProbe(function, *sequence, **prepared, details); }
    if (function->hasAttr("test.contract_completion_bound") && mathematics.completionChild) {
        auto child = mathematics.completionChild();
        details["closed_child_qualified"] = bool(child);
        if (child) {
            details["child_pending_payloads"] = llvm::any_of(child->summary().suffix,
                [](uint8_t mode) { return mode != 0; });
            details["child_transparent"] = child->summary().transparent;
            details["child_pending_notification_completion"] = child->returnsPendingEvents();
        }
    }
    return details;
}
struct Probe {
    const fs::RegionalAnalysis& region;
    ArrayRef<const pto::CompoundInstanceElement*> phases;
    std::vector<Key> supplied;
    std::vector<std::optional<fs::RegionalEvent>> events;
    std::string error;
    bool resolve(ArrayAttr descriptions)
    {
        for (auto attribute : descriptions) {
            auto raw = dyn_cast<DenseI64ArrayAttr>(attribute);
            if (!raw || raw.empty() || raw[0] < 0 || static_cast<uint64_t>(raw[0]) >= phases.size()) {
                return false;
            }
            Key requested{static_cast<uint64_t>(raw[0]), {}};
            auto loops = enclosing(phases[requested.first]->elementOp);
            if (static_cast<uint64_t>(raw.size()) != loops.size() + 1) {
                return false;
            }
            Coordinates coordinates;
            for (std::size_t i = 0; i < loops.size(); ++i) {
                if (raw[i + 1] < 0) {
                    return false;
                }
                coordinates[loops[i]] = static_cast<uint64_t>(raw[i + 1]);
                requested.second.push_back(static_cast<uint64_t>(raw[i + 1]));
            }
            std::optional<fs::RegionalEvent> match;
            for (uint32_t type = 0; type < region.anchors.size(); ++type) {
                if (region.anchors[type].phase != phases[requested.first]) {
                    continue;
                }
                auto event = candidate(region, type, coordinates);
                if (!event) {
                    error = "original occurrence frame unavailable in test consumer";
                    return false;
                }
                auto original = key(region, *event, phases);
                if (!original) {
                    error = "original occurrence coordinate decoding unavailable in test consumer";
                    return false;
                }
                if (*original != requested) { continue; }
                auto present = fs::regionalPresence(region, *event);
                if (!present || !number(region, *present)) {
                    error = "nonconstant occurrence presence";
                    return false;
                }
                if (number(region, *present) == 0) {
                    continue;
                }
                if (match) {
                    error = "multiple present representations of one occurrence";
                    return false;
                }
                match = *event;
            }
            supplied.push_back(std::move(requested));
            events.push_back(std::move(match));
        }
        return true;
    }
    std::optional<llvm::json::Array> selectors(ArrayRef<fs::RegionalSelector> values)
    {
        llvm::json::Array result;
        for (const auto& value : values) {
            auto active = number(region, value.present);
            if (!active) {
                return {};
            }
            if (!*active) {
                continue;
            }
            auto occurrence = key(region, value.event, phases);
            if (!occurrence) {
                return {};
            }
            auto found = llvm::find(supplied, *occurrence);
            if (found == supplied.end()) {
                return {};
            }
            result.push_back(static_cast<uint64_t>(found - supplied.begin()));
        }
        return result;
    }
    llvm::json::Object boundaries(ArrayAttr cells)
    {
        llvm::json::Array rows;
        for (auto attribute : cells) {
            auto raw = dyn_cast<DenseI64ArrayAttr>(attribute);
            if (!raw || raw.size() != 3 || raw[0] < 1 ||
                raw[0] > static_cast<int64_t>(pto::AddressSpace::VEC) || raw[2] < 0) {
                return llvm::json::Object{{"error", "bad cell"}};
            }
            Value base;
            if (raw[1] >= 0) {
                auto function = phases.front()->elementOp->getParentOfType<func::FuncOp>();
                if (static_cast<uint64_t>(raw[1]) >= function.getNumArguments()) {
                    return llvm::json::Object{{"error", "bad base"}};
                }
                base = function.getArgument(raw[1]);
            }
            fs::RegionalByteAddress address{
                static_cast<pto::AddressSpace>(raw[0]), base,
                region.expressions->constant(static_cast<uint64_t>(raw[2]))};
            auto selected = fs::regionalStorageSelectors(region, address);
            if (!selected) {
                return llvm::json::Object{
                    {"error", "storage selector unavailable"}, {"cell", llvm::json::Array{raw[0], raw[1], raw[2]}}};
            }
            auto first = selectors(selected->firstWriters), last = selectors(selected->lastWriters);
            if (!first || !last) {
                return llvm::json::Object{{"error", "nonconcrete writer selector"}};
            }
            llvm::json::Object row{{"first_writers", std::move(*first)}, {"last_writers", std::move(*last)}};
            for (auto named :
                 {std::make_pair("first_readers", &selected->firstReaders),
                  std::make_pair("last_readers", &selected->lastReaders)}) {
                llvm::json::Object readers;
                for (const auto& entry : *named.second) {
                    auto values = selectors(entry.second);
                    if (!values) {
                        return llvm::json::Object{{"error", "nonconcrete reader selector"}};
                    }
                    readers[std::to_string(entry.first)] = std::move(*values);
                }
                row[named.first] = std::move(readers);
            }
            rows.push_back(std::move(row));
        }
        return llvm::json::Object{{"error", ""}, {"rows", std::move(rows)}};
    }
    llvm::json::Array queries()
    {
        llvm::json::Array rows;
        for (std::size_t a = 0; a < 2 * events.size(); ++a) {
            std::string row;
            for (std::size_t b = 0; b < 2 * events.size(); ++b) {
                if (!events[a / 2] || !events[b / 2]) {
                    row += '0';
                    continue;
                }
                auto source = *events[a / 2], target = *events[b / 2];
                source.kind = static_cast<fs::PeriodicEventKind>(a % 2);
                target.kind = static_cast<fs::PeriodicEventKind>(b % 2);
                auto query = fs::regionalReachability(region, source, target);
                auto answer = query ? number(region, *query) : std::nullopt;
                row += answer ? (*answer ? '1' : '0') : '?';
            }
            rows.push_back(std::move(row));
        }
        return rows;
    }
};
} // namespace
int runCompactRegionContract(const char* filename)
{
    using namespace compact_contract;
    DialectRegistry dialects;
    dialects.insert<pto::PTODialect, func::FuncDialect, arith::ArithDialect, scf::SCFDialect>();
    MLIRContext context(dialects);
    context.disableMultithreading();
    auto module = parseSourceFile<ModuleOp>(filename, &context);
    if (!module) {
        return 1;
    }
    for (auto function : module->getOps<func::FuncOp>()) {
        fs::FrontierAnalysis session(function);
        auto policy = pto::GMAliasPolicy::MayNotAlias;
        if (auto alias = function->getAttrOfType<StringAttr>("test.contract_gm_alias")) {
            if (alias.getValue() != "may-alias" && alias.getValue() != "may-not-alias") {
                return 1;
            }
            policy = alias.getValue() == "may-alias" ?
                pto::GMAliasPolicy::MayAlias : pto::GMAliasPolicy::MayNotAlias;
        }
        if (failed(session.initialize(policy))) {
            return 1;
        }
        fs::AnalysisRequest request;
        request.needs.synchronization = true;
        request.mode = fs::AnalysisMode::Fallback;
        // Separate diagnostic requests never upgrade the ordinary production
        // inventory merely to fill a report. The caller pins each request.
        const bool mathOnly = function->hasAttr("test.contract_math_only");
        if (mathOnly) {
            request.needs.synchronization = false;
            request.mode = fs::AnalysisMode::MinimumExact;
        }
        if (auto region = function->getAttrOfType<IntegerAttr>("test.contract_region")) {
            if (region.getInt() < 0) { return 1; }
            request.region = static_cast<std::size_t>(region.getInt());
        }
        if (auto selected = function->getAttrOfType<IntegerAttr>("test.contract_loop")) {
            SmallVector<scf::ForOp> loops;
            function.walk<WalkOrder::PreOrder>([&](scf::ForOp loop) { loops.push_back(loop); });
            if (selected.getInt() < 0 || static_cast<uint64_t>(selected.getInt()) >= loops.size()) { return 1; }
            const auto* structure = session.result();
            auto found = llvm::find_if(structure->nodes, [&](const auto& node) {
                return node.kind == fs::StructureKind::Loop && node.anchor == loops[selected.getInt()];
            });
            if (found == structure->nodes.end()) { return 1; }
            request.region = std::distance(structure->nodes.begin(), found);
        }
        if (auto needs = function->getAttrOfType<IntegerAttr>("test.contract_exports")) {
            if (needs.getInt() < 0 || needs.getInt() > 7) { return 1; }
            request.needs.queries = needs.getInt() & 1;
            request.needs.selectors = needs.getInt() & 2;
            request.needs.synchronization = needs.getInt() & 4;
        }
        // Explicit research comparisons use the existing named requests. They
        // do not change production selection or silently try another strategy.
        auto strategy = function->getAttrOfType<StringAttr>("test.contract_strategy");
        fs::AnalysisOutcome answer;
        if (!strategy || strategy.getValue() == "production") {
            answer = session.analyze(request);
        } else if (strategy.getValue() == "arithmetic") {
            answer = session.analyzeArithmeticRegional(request);
        } else if (strategy.getValue() == "numerical") {
            answer = session.analyzeNumericalRegion(request);
        } else if (strategy.getValue() == "expanded") {
            answer = session.analyzeFiniteExpansion(request);
        } else {
            function.emitError("unknown compact-class test strategy");
            return 1;
        }
        if (answer.status != fs::AnalysisStatus::Ready || !answer.mathematical) {
            llvm::json::Array obligations;
            for (const auto& item : answer.obligations) {
                obligations.push_back(item.diagnostic);
            }
            llvm::outs() << llvm::json::Value(
                                llvm::json::Object{
                                    {"error", answer.completion ?
                                                  "selected completion owner has no exact query export" :
                                                  "production exact export not ready"},
                                    {"status", static_cast<unsigned>(answer.status)},
                                    {"backend", answer.mathematical ? answer.mathematical->backend : "unavailable"},
                                    {"retained_mathematics", bool(answer.mathematical)},
                                    {"structure", mathOnly ? originalStructure(session) : llvm::json::Array{}},
                                    {"obligations", std::move(obligations)}})
                         << "\n";
            return 0;
        }
        if (mathOnly && function->hasAttr("test.contract_completion_parent")) {
            if (failed(runGroupedCompletionParent(function, answer.mathematical))) { return 1; }
            function->getParentOp()->print(llvm::outs());
            llvm::outs() << "\n";
            return 0;
        }
        if (mathOnly && function->hasAttr("test.contract_point_checks")) {
            llvm::outs() << llvm::json::Value(expandedPointChecks(*answer.mathematical)) << "\n";
            return 0;
        }
        if (mathOnly && function->hasAttr("test.contract_emit")) {
            auto prepared = session.prepareLogical(answer);
            if (failed(prepared) || failed(fs::insertLogicalSynchronization(function, **prepared))) { return 1; }
            function->getParentOp()->print(llvm::outs());
            llvm::outs() << "\n";
            return 0;
        }
        if (mathOnly && function->hasAttr("test.contract_completion_emit")) {
            auto sequence = answer.mathematical->sequenceDemands;
            if (!sequence || !sequence->state) { return 1; }
            auto prepared = fs::prepareSequenceLogicalInsertion(*sequence, sequence->state->requiredOuterLoops);
            if (failed(prepared)) { return 1; }
            auto invocation = std::make_shared<fs::RegionExpressions>();
            for (auto loop : sequence->state->requiredOuterLoops) {
                auto recipe = fs::prefixFamilyInvocation((*prepared)->familyRecipe, invocation,
                    invocation->input(loop.getInductionVar()));
                if (failed(recipe)) { return 1; }
                (*prepared)->familyRecipe = std::move(*recipe);
            }
            (*prepared)->nestedIdentities = true;
            if (failed(fs::insertLogicalSynchronization(function, **prepared))) { return 1; }
            function->getParentOp()->print(llvm::outs());
            llvm::outs() << "\n";
            return 0;
        }
        if (mathOnly && function->hasAttr("test.contract_completion_probe")) {
            llvm::outs() << llvm::json::Value(completionPreparationProbe(function, *answer.mathematical)) << "\n";
            return 0;
        }
        if (mathOnly) {
            llvm::outs() << llvm::json::Value(llvm::json::Object{
                {"error", ""}, {"request", "mathematics-only-diagnostic"},
                {"region", request.region}, {"backend", answer.mathematical->backend},
                {"arithmetic", arithmeticSummary(*answer.mathematical)},
                {"new_construction", llvm::json::Array{session.constructionCounts().mathematicalAttempts,
                    session.constructionCounts().arithmeticQueryBuilds,
                    session.constructionCounts().arithmeticSelectorBuilds}},
                {"structure", originalStructure(session)},
                {"inventory", fs::retainedAnalysisInventory(*answer.mathematical)}}) << "\n";
            return 0;
        }
        const auto counts = constructionSnapshot(session);
        const auto sizes = retainedSnapshot(*answer.mathematical);
        auto inventory = fs::retainedAnalysisInventory(*answer.mathematical);
        auto again = fs::retainedAnalysisInventory(*answer.mathematical);
        std::string first, second;
        llvm::raw_string_ostream(first) << llvm::json::Value(std::move(inventory));
        llvm::raw_string_ostream(second) << llvm::json::Value(std::move(again));
        if (first != second || counts != constructionSnapshot(session) ||
            sizes != retainedSnapshot(*answer.mathematical)) {
            return 1;
        }
        auto math = answer.mathematical;
        while (math->identityComposition) {
            math = math->identityComposition->child;
        }
        fs::RegionalAnalysis region;
        if (answer.regionalExports) {
            region = *answer.regionalExports;
        } else if (math->sequenceDemands) {
            region = fs::sequenceRegionalResult(*math->sequenceDemands);
        } else if (math->finiteDemands) {
            region = fs::finiteRegionalResult(math->finiteDemands);
        } else if (math->regionalDemands) {
            region = *math->regionalDemands;
        } else {
            llvm::outs() << "{\"error\":\"selected export unavailable\"}\n";
            return 0;
        }
        if (region.retainedMathematics.expired()) { region.retainedMathematics = math; }
        auto occurrences = function->getAttrOfType<ArrayAttr>("test.contract_occurrences");
        auto cells = function->getAttrOfType<ArrayAttr>("test.contract_cells");
        if (!occurrences || !cells || !region.expressions) {
            return 1;
        }
        Probe probe{region, session.input()->instructions(), {}, {}, {}};
        if (!probe.resolve(occurrences)) {
            llvm::outs() << llvm::json::Value(llvm::json::Object{{"error", probe.error}}) << "\n";
            return 0;
        }
        llvm::json::Object decodeChecks;
        if (function->hasAttr("test.contract_decode_checks")) {
            // A minimal quotient-frame view deliberately unsupported by this
            // consumer. No query callback is needed: decoding must refuse first.
            fs::RegionalAnalysis unsupported;
            unsupported.expressions = region.expressions;
            unsupported.anchors = region.anchors;
            unsupported.occurrenceLoops.resize(region.anchors.size());
            unsupported.outerLoops.resize(region.anchors.size());
            unsupported.outerDivisors.resize(region.anchors.size());
            bool changed = false;
            for (std::size_t type = 0; type < region.anchors.size(); ++type) {
                const auto loops = enclosing(region.anchors[type].phase->elementOp);
                unsupported.outerLoops[type].assign(loops.begin(), loops.end());
                unsupported.outerDivisors[type].assign(unsupported.outerLoops[type].size(), 2);
                changed |= !unsupported.outerLoops[type].empty();
            }
            if (!changed) { return 1; }
            Probe incomplete{unsupported, session.input()->instructions(), {}, {}, {}};
            const bool rejected = !incomplete.resolve(occurrences) &&
                incomplete.error == "original occurrence coordinate decoding unavailable in test consumer";
            if (!rejected) { return 1; }
            decodeChecks["unsupported_frame_unavailable"] = true;
        }
        llvm::json::Array present;
        for (const auto& event : probe.events) {
            present.push_back(event.has_value());
        }
        auto boundary = probe.boundaries(cells);
        Demands demands{region, session.input()->instructions(), probe.supplied, {}};
        auto records=demands.expand(probe.events);
        llvm::outs() << llvm::json::Value(
                            llvm::json::Object{
                                {"error", ""},
                                {"backend", answer.mathematical->backend},
                                {"arithmetic", arithmeticSummary(*math)},
                                {"inventory_read_only", true},
                                {"decode_checks", std::move(decodeChecks)},
                                {"selector_source",
                                 region.storageSelectors ? "provider-callback" : "shared-cell-coverage"},
                                {"present", std::move(present)},
                                {"event_rows", probe.queries()}, {"retained_demands", std::move(records)},
                                      {"retained_demand_error", demands.error},
                                {"boundaries", std::move(boundary)}})
                     << "\n";
    }
    return 0;
}
