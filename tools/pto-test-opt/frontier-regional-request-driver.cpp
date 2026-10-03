// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Calls the actual production regional service; no target qualification bypass.
#include "../../lib/PTO/Transforms/FrontierSynch/RegionalRequests.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/ScopeExit.h"
#include <set>
namespace {
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
bool failRegional(unsigned line)
{
    llvm::errs() << "regional check failed at line " << line << "\n";
    return false;
}
// Test-only all-event closure oracle for the small explicit executions below.
// It does not use a production reducer or reconstruct a symbolic trace circuit.
std::set<std::pair<std::size_t, std::size_t>> unfoldedCovers(
    const fs::TraceDemandAnalysis& source, ArrayRef<std::size_t> word)
{
    const auto count = word.size();
    SmallVector<SmallVector<bool>> reach(2 * count, SmallVector<bool>(2 * count, false));
    SmallVector<std::pair<std::size_t, std::size_t>> demands;
    DenseMap<pto::PipelineType, std::size_t> previous;
    auto conflicts = source.conflicts();
    for (std::size_t b = 0; b < count; ++b) {
        reach[2 * b][2 * b + 1] = true;
        auto pipe = source.sites()[word[b]].phase->kPipeValue;
        auto found = previous.find(pipe);
        if (found != previous.end()) {
            reach[2 * found->second][2 * b] = true;
            reach[2 * found->second + 1][2 * b + 1] = true;
        }
        previous[pipe] = b;
        for (std::size_t a = 0; a < b; ++a) {
            if (llvm::is_contained(conflicts[word[b]], word[a])) {
                reach[2 * a + 1][2 * b] = true;
                demands.emplace_back(a, b);
            }
        }
    }
    for (std::size_t middle = 0; middle < reach.size(); ++middle) {
        for (std::size_t a = 0; a < reach.size(); ++a) {
            for (std::size_t b = 0; b < reach.size(); ++b) {
                reach[a][b] = reach[a][b] || (reach[a][middle] && reach[middle][b]);
            }
        }
    }
    std::set<std::pair<std::size_t, std::size_t>> covers;
    for (auto [a, b] : demands) {
        bool redundant = false;
        for (std::size_t middle = 0; middle < reach.size(); ++middle) {
            redundant |= reach[2 * a + 1][middle] && reach[middle][2 * b];
        }
        if (!redundant) { covers.emplace(a, b); }
    }
    return covers;
}
bool checkPreparation(func::FuncOp function)
{
    pto::SyncInput input;
    if (failed(input.build(function))) { return failRegional(__LINE__); }
    auto trace = fs::TraceDemandAnalysis::build(function, input);
    if (failed(trace)) { return failRegional(__LINE__); }
    fs::CostLedger costs(true);
    fs::RegionalRequests requests(function, input, *trace, costs);
    auto needs = fs::AnalysisNeeds::modeledCovers();
    std::string before;
    llvm::raw_string_ostream original(before);
    function.print(original);
    Operation* owner = function;
    bool late = function.getName() == "late_outcome";
    if (!late) {
        function.walk([&](scf::IfOp branch) { owner = branch; });
    }
    auto context = requests.contextFor(owner);
    auto mathematical = requests.request(owner, context, fs::RegionalMode::Modeled, needs);
    if (!mathematical.analysis || mathematical.analysis->kind != fs::SelectedAnalysis::Kind::Guarded) {
        return failRegional(__LINE__);
    }
    auto matching = requests.request(owner, context, fs::RegionalMode::Modeled, needs,
        fs::RegionalRepresentation::NativeSummaries, fs::RegionalPreparation::SelectorMatching);
    if (late) {
        if (matching.analysis || matching.obligation.find("unavailable outcome") == std::string::npos) {
            return failRegional(__LINE__);
        }
    } else if (!matching.analysis || matching.analysis == mathematical.analysis) { return failRegional(__LINE__); }
    auto hits = requests.cacheHits();
    auto attempts = requests.attempts().size();
    auto repeated = requests.request(owner, context, fs::RegionalMode::Modeled, needs,
        fs::RegionalRepresentation::NativeSummaries, fs::RegionalPreparation::SelectorMatching);
    if (repeated.analysis != matching.analysis || repeated.obligation != matching.obligation ||
        requests.cacheHits() != hits + 1 || requests.attempts().size() != attempts) { return failRegional(__LINE__); }
    if (!late) {
        auto history = requests.attempts();
        auto first = llvm::find_if(history, [](const auto& attempt) {
            return attempt.route == "regional-difference-bounds";
        });
        auto guarded = llvm::find_if(history, [](const auto& attempt) {
            return attempt.route == "regional-finite-guarded" && attempt.outcome == "ready";
        });
        if (first == history.end() || guarded == history.end() || first >= guarded) { return failRegional(__LINE__); }
        auto root = requests.request(function, requests.context(), fs::RegionalMode::Modeled, needs,
            fs::RegionalRepresentation::NativeSummaries, fs::RegionalPreparation::SelectorMatching);
        if (root.analysis) { return failRegional(__LINE__); }
        auto index = std::make_shared<fs::PhaseIndex>();
        if (failed(index->build(function, input))) { return failRegional(__LINE__); }
        fs::StorageAnalysis storage(input, matching.analysis->guarded.phases());
        fs::GuardedDemandAnalysis invalid;
        SmallVector<const pto::CompoundInstanceElement*> wrong(matching.analysis->guarded.phases());
        std::reverse(wrong.begin(), wrong.end());
        if (succeeded(invalid.build(owner, *index, wrong, storage))) { return failRegional(__LINE__); }
    }
    std::string after;
    llvm::raw_string_ostream output(after);
    function.print(output);
    if (before != after) { return failRegional(__LINE__); }
    return true;
}
bool checkLocalPreparation(func::FuncOp function)
{
    pto::SyncInput input;
    if (failed(input.build(function))) { return failRegional(__LINE__); }
    auto trace = fs::TraceDemandAnalysis::build(function, input);
    if (failed(trace)) { return failRegional(__LINE__); }
    std::string before;
    llvm::raw_string_ostream original(before);
    function.print(original);
    fs::CostLedger costs(true);
    fs::RegionalRequests requests(function, input, *trace, costs);
    auto needs = fs::AnalysisNeeds::modeledCovers();
    auto mathematical = requests.request(function, requests.context(), fs::RegionalMode::Modeled, needs);
    if (!mathematical.analysis || mathematical.analysis->kind != fs::SelectedAnalysis::Kind::Guarded ||
        mathematical.analysis->guarded.localDemands().empty()) { return failRegional(__LINE__); }
    auto matching = requests.request(function, requests.context(), fs::RegionalMode::Modeled, needs,
        fs::RegionalRepresentation::NativeSummaries, fs::RegionalPreparation::SelectorMatching);
    bool exclusive = function.getName() == "guard_local_exclusive";
    if (exclusive ? !matching.analysis :
        (matching.analysis || matching.obligation.find("local-adjacency") == std::string::npos)) {
        return failRegional(__LINE__);
    }
    auto hits = requests.cacheHits();
    auto repeated = requests.request(function, requests.context(), fs::RegionalMode::Modeled, needs,
        fs::RegionalRepresentation::NativeSummaries, fs::RegionalPreparation::SelectorMatching);
    if (requests.cacheHits() != hits + 1 || repeated.analysis != matching.analysis ||
        repeated.obligation != matching.obligation) { return failRegional(__LINE__); }
    std::string after;
    llvm::raw_string_ostream output(after);
    function.print(output);
    return before == after || failRegional(__LINE__);
}
bool check(func::FuncOp function)
{
    pto::SyncInput input;
    if (failed(input.build(function))) { return failRegional(__LINE__); }
    auto trace = fs::TraceDemandAnalysis::build(function, input);
    if (failed(trace) || trace->conflictsPrepared()) { return failRegional(__LINE__); }
    std::string original;
    llvm::raw_string_ostream output(original);
    function.print(output);
    fs::CostLedger costs(true);
    fs::RegionalRequests requests(function, input, *trace, costs);
    auto needs = fs::AnalysisNeeds::modeledCovers();
    auto root = requests.request(function, requests.context(), fs::RegionalMode::Modeled, needs);
    scf::ForOp loop;
    function.walk([&](scf::ForOp candidate) { loop = candidate; });
    if (requests.signatureMembershipQueries() > 4 * trace->sites().size()) { return failRegional(__LINE__); }
    if (!loop || root.children.empty()) { return failRegional(__LINE__); }
    if (function.getName() == "prologue_loop_epilogue" &&
        (!root.analysis || root.analysis->route != "structured-composition" ||
         root.analysis->regionalChildren.size() != 3)) {
        llvm::errs() << "root obligation: " << root.obligation << "\n";
        return failRegional(__LINE__);
    }
    if (function.getName() == "unsupported_sibling" && root.analysis) { return failRegional(__LINE__); }
    auto quotient = requests.request(loop, requests.context(), fs::RegionalMode::Modeled, needs);
    if (!quotient.analysis || quotient.analysis->kind != fs::SelectedAnalysis::Kind::Periodic ||
        quotient.analysis->sites.size() != 2 || quotient.analysis->periodic.retained().empty()) {
            return failRegional(__LINE__);
        }
    auto threshold = quotient.analysis->periodic.threshold(0, 1, fs::PeriodicEventKind::Start);
    if (failed(threshold) || !*threshold || **threshold != 0) { return failRegional(__LINE__); }
    auto hits = requests.cacheHits();
    auto attempts = requests.attempts().size();
    if (requests.request(loop, requests.context(), fs::RegionalMode::Modeled, needs).analysis != quotient.analysis ||
        requests.request(function, requests.context(), fs::RegionalMode::Modeled, needs).outcome != root.outcome ||
        requests.cacheHits() != hits + 2 || requests.attempts().size() != attempts) { return failRegional(__LINE__); }
    // A different configured route list must not reuse an economical cache
    // entry, including its proper-child analysis and failed capability request.
    auto extended = requests.request(loop, requests.context(), fs::RegionalMode::Modeled, needs,
        fs::RegionalRepresentation::NativeSummaries, fs::RegionalPreparation::Mathematical,
        fs::RegionalRoutePolicy::GeneralExtension);
    if (!extended.analysis || extended.analysis == quotient.analysis) { return failRegional(__LINE__); }
    hits = requests.cacheHits();
    attempts = requests.attempts().size();
    auto extensionHit = requests.request(loop, requests.context(), fs::RegionalMode::Modeled, needs,
        fs::RegionalRepresentation::NativeSummaries, fs::RegionalPreparation::Mathematical,
        fs::RegionalRoutePolicy::GeneralExtension);
    if (extensionHit.analysis != extended.analysis || requests.cacheHits() != hits + 1 ||
        requests.attempts().size() != attempts) { return failRegional(__LINE__); }
    auto boundaries = needs;
    boundaries.interfaces |= fs::interfaceBit(fs::DemandInterface::CellBoundaries);
    auto absent = requests.request(loop, requests.context(), fs::RegionalMode::Modeled, boundaries);
    if (absent.analysis || absent.obligation.find("cell boundaries") == std::string::npos) {
        return failRegional(__LINE__);
    }
    hits = requests.cacheHits();
    requests.request(loop, requests.context(), fs::RegionalMode::Modeled, boundaries);
    if (requests.cacheHits() != hits + 1) { return failRegional(__LINE__); }
    auto extendedFailure = requests.request(loop, requests.context(), fs::RegionalMode::Modeled, boundaries,
        fs::RegionalRepresentation::NativeSummaries, fs::RegionalPreparation::Mathematical,
        fs::RegionalRoutePolicy::GeneralExtension);
    if (extendedFailure.analysis || extendedFailure.obligation.find("cell boundaries") == std::string::npos) {
        return failRegional(__LINE__);
    }
    hits = requests.cacheHits();
    attempts = requests.attempts().size();
    auto repeatedFailure = requests.request(loop, requests.context(), fs::RegionalMode::Modeled, boundaries,
        fs::RegionalRepresentation::NativeSummaries, fs::RegionalPreparation::Mathematical,
        fs::RegionalRoutePolicy::GeneralExtension);
    if (repeatedFailure.obligation != extendedFailure.obligation || requests.cacheHits() != hits + 1 ||
        requests.attempts().size() != attempts) { return failRegional(__LINE__); }
    auto separate = requests.contextFor(loop);
    hits = requests.cacheHits();
    auto scoped = requests.request(loop, separate, fs::RegionalMode::Modeled, needs);
    if (!scoped.analysis || scoped.analysis->requestContext != separate ||
        quotient.analysis->requestContext != requests.context() ||
        scoped.analysis == quotient.analysis || requests.cacheHits() != hits + 1) { return failRegional(__LINE__); }
    fs::SelectedAnalysisHandle retained;
    {
        fs::RegionalRequests temporary(function, input, *trace, costs);
        retained = temporary.request(loop, temporary.contextFor(loop), fs::RegionalMode::Modeled, needs).analysis;
    }
    if (!retained || !retained->regionalContext || !retained->requestContext ||
        retained->regionalContext->scope != loop ||
        !llvm::is_contained(retained->regionalContext->entryRegions, loop->getParentRegion()) ||
        !retained->regionalContext->controlValues) { return failRegional(__LINE__); }
    auto exact = requests.request(loop, separate, fs::RegionalMode::MinimumExact, needs);
    if (exact.analysis || exact.obligation.find("shared modeled effects") == std::string::npos) {
        return failRegional(__LINE__);
    }
    auto arithmetic = requests.request(loop, separate, fs::RegionalMode::Modeled, needs,
                                      fs::RegionalRepresentation::ArithmeticRelations);
    if (!arithmetic.analysis || (arithmetic.analysis->kind != fs::SelectedAnalysis::Kind::Signed &&
         arithmetic.analysis->kind != fs::SelectedAnalysis::Kind::Periodic) ||
        !arithmetic.analysis->reachability || !arithmetic.analysis->minimum) { return failRegional(__LINE__); }
    if (function.getName() == "prologue_loop_epilogue" &&
        (arithmetic.analysis->kind != fs::SelectedAnalysis::Kind::Periodic ||
         arithmetic.analysis->route != "regional-periodic-quotient")) { return failRegional(__LINE__); }
    // Same arity is insufficient: a reordered schema changes site identities.
    auto sourceSchema = arithmetic.analysis->structured->schema();
    SmallVector<fs::SymbolicSite> reversed(sourceSchema->sites().rbegin(), sourceSchema->sites().rend());
    auto foreign = fs::SymbolicSchema::create(reversed, sourceSchema->parameters(), sourceSchema->cellTypes(),
                                            sourceSchema->partition());
    if (failed(foreign)) { return failRegional(__LINE__); }
    auto foreignSpace = fs::SignedSpace::create(*foreign, llvm::DynamicAPInt(1));
    auto nonunit = fs::SignedSpace::create(sourceSchema, llvm::DynamicAPInt(2));
    if (!foreignSpace.succeeded() || !nonunit.succeeded()) { return failRegional(__LINE__); }
    std::string wrongSpace;
    if (succeeded(fs::specializePrimitives(arithmetic.analysis->structured, fs::ArithmeticClass::Octagons,
                                         wrongSpace, foreignSpace.value)) ||
        succeeded(fs::specializePrimitives(arithmetic.analysis->structured, fs::ArithmeticClass::Octagons,
                                         wrongSpace, nonunit.value))) { return failRegional(__LINE__); }
    auto arithmeticHits = requests.cacheHits();
    auto again = requests.request(loop, separate, fs::RegionalMode::Modeled, needs,
                                  fs::RegionalRepresentation::ArithmeticRelations);
    if (again.analysis != arithmetic.analysis || requests.cacheHits() != arithmeticHits + 1) {
        return failRegional(__LINE__);
    }
    if (function.getName() == "guarded_child") {
        fs::SignedPoint source{{1, fs::PeriodicEventKind::Completion}, {llvm::DynamicAPInt(0)}};
        fs::SignedPoint target{{2, fs::PeriodicEventKind::Start}, {llvm::DynamicAPInt(0)}};
        auto skipped = arithmetic.analysis->minimum->contains(
            source, target, {llvm::DynamicAPInt(1), llvm::DynamicAPInt(0)});
        if (!skipped.succeeded() || skipped.value) { return failRegional(__LINE__); }
    }
    if (function.getName() == "prologue_loop_epilogue") {
        for (int trips : {0, 1, 3}) {
            SmallVector<std::size_t> word{0};
            SmallVector<fs::SignedPoint> occurrences{{{0, std::nullopt}, {}}};
            for (int iteration = 0; iteration < trips; ++iteration) {
                for (std::size_t site : {1U, 2U}) {
                    word.push_back(site);
                    occurrences.push_back({{site, std::nullopt}, {llvm::DynamicAPInt(iteration)}});
                }
            }
            word.push_back(3);
            occurrences.push_back({{3, std::nullopt}, {}});
            auto covers = unfoldedCovers(*trace, word);
            for (std::size_t a = 0; a < word.size(); ++a) {
                for (std::size_t b = 0; b < word.size(); ++b) {
                    auto source = occurrences[a], target = occurrences[b];
                    source.tag.kind = fs::PeriodicEventKind::Completion;
                    target.tag.kind = fs::PeriodicEventKind::Start;
                    auto actual = root.analysis->minimum->contains(source, target, {llvm::DynamicAPInt(trips)});
                    bool expected = covers.count({a, b}) != 0;
                    if (!actual.succeeded() || actual.value != expected) {
                        llvm::errs() << "query disagreement trips=" << trips << " source=" << a << " target=" << b
                                     << " status=" << static_cast<unsigned>(actual.status) << " actual=" << actual.value
                                     << " expected=" << expected << "\n";
                        return failRegional(__LINE__);
                    }
                }
            }
        }
    }
    std::string after;
    llvm::raw_string_ostream finalOutput(after);
    function.print(finalOutput);
    if (after != original) { return failRegional(__LINE__); }
    // The real production selector must recognize a periodic child BEFORE a
    // rejected parent arithmetic import, even when a sibling has nonlinear control.
    SmallVector<fs::AnalysisAttempt> history;
    std::string reason;
    auto selected = fs::selectAnalysis(function, input, *trace, history, reason, costs);
    auto child = llvm::find_if(history, [](const auto& attempt) {
        return attempt.route == "regional-periodic-quotient" && attempt.outcome == "ready";
    });
    if (child == history.end()) { return failRegional(__LINE__); }
    if (selected && selected->regionalChildren.empty()) { return failRegional(__LINE__); }
    if (function.getName() == "prologue_loop_epilogue") {
        if (!selected || selected->route != "structured-composition" || selected->endpoints.empty()) {
            return failRegional(__LINE__);
        }
        auto executable = needs;
        executable.interfaces |= fs::interfaceBit(fs::DemandInterface::ExecutableEndpoints);
        fs::RegionalRequests rejectedRequests(function, input, *trace, costs);
        auto rejected = rejectedRequests.request(function, rejectedRequests.context(), fs::RegionalMode::Modeled,
                                                 executable);
        if (rejected.analysis) { return failRegional(__LINE__); }
        for (const auto& attempt : rejectedRequests.attempts()) {
            if (attempt.region == &function.getBody() && attempt.outcome == "ready" &&
                (attempt.route == "structured-composition" || attempt.route == "regional-sequence-composition")) {
                return failRegional(__LINE__);
            }
        }
        IRMapping mapping;
        fs::DirectEmissionResult staged;
        staged.pending = cast<func::FuncOp>(function->clone(mapping));
        if (failed(fs::emitPreparedEndpoints(mapping, *selected, staged)) || !staged.sets || !staged.waits) {
            return failRegional(__LINE__);
        }
        // Exercise production bitmap preservation for a genuine whole-loop
        // quotient request, with original scalar endpoint qualification enabled.
        auto onlyLoop = cast<func::FuncOp>(function->clone());
        onlyLoop.setName("regional_threshold_case");
        function->getBlock()->push_back(onlyLoop);
        auto cleanup = llvm::make_scope_exit([&]() { onlyLoop.erase(); });
        SmallVector<Operation*> remove;
        for (auto& operation : onlyLoop.getBody().front()) {
            if (isa<func::CallOp>(operation)) { remove.push_back(&operation); }
        }
        for (auto* operation : remove) { operation->erase(); }
        pto::SyncInput onlyInput;
        if (failed(onlyInput.build(onlyLoop))) { return failRegional(__LINE__); }
        auto onlyTrace = fs::TraceDemandAnalysis::build(onlyLoop, onlyInput);
        if (failed(onlyTrace)) { return failRegional(__LINE__); }
        auto thresholds = needs;
        thresholds.interfaces |= fs::interfaceBit(fs::DemandInterface::PeriodicThresholds);
        SmallVector<fs::AnalysisAttempt> onlyHistory;
        auto accepted = fs::selectAnalysis(onlyLoop, onlyInput, *onlyTrace, onlyHistory, reason, costs, thresholds);
        if (!accepted || !accepted->contract.accepts(thresholds, reason)) { return failRegional(__LINE__); }
    }
    return true;
}
} // namespace
int runRegionalRequestChecks(llvm::StringRef path, mlir::MLIRContext& context)
{
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(path, &context);
    if (!module) { return 1; }
    unsigned tested = 0;
    for (auto function : module->getOps<mlir::func::FuncOp>()) {
        if (!function.isExternal()) {
            auto accepted = function.getName().starts_with("guard_local_") ? checkLocalPreparation(function) :
                function.getName() == "opaque_guarded_child" || function.getName() == "late_outcome" ?
                checkPreparation(function) : check(function);
            if (!accepted) { llvm::errs() << "regional check failed: " << function.getName() << "\n"; return 1; }
            ++tested;
        }
    }
    if (tested != 7) { return 1; }
    llvm::outs() << "verified regional requests: child quotient, sibling independence, "
                    "cache/context/needs and rollback\n";
    return 0;
}
