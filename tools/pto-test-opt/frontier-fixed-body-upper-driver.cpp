// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent finite event-graph oracle; production import never unfolds trips.
#include "../../lib/PTO/Transforms/FrontierSynch/RegionalRequests.h"
#include "../../lib/PTO/Transforms/FrontierSynch/FixedBodyUpper.h"
#include "PTO/Transforms/InsertSync/SyncStorageBounds.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
namespace {
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
using Graph = SmallVector<SmallVector<unsigned char>>;
void close(Graph& graph)
{
    for (std::size_t k = 0; k < graph.size(); ++k) {
        for (std::size_t a = 0; a < graph.size(); ++a) {
            for (std::size_t b = 0; b < graph.size(); ++b) {
                graph[a][b] |= graph[a][k] && graph[k][b];
            }
        }
    }
}
bool oracle(const fs::FixedBodyUpper& upper, const pto::SyncInput& input, unsigned trips)
{
    auto phases = upper.selected().sites();
    auto count = phases.size() * trips;
    Graph required(2 * count, SmallVector<unsigned char>(2 * count));
    DenseMap<pto::PipelineType, std::size_t> last;
    for (std::size_t a = 0; a < count; ++a) {
        required[2 * a][2 * a + 1] = 1;
        auto pipe = phases[a % phases.size()]->kPipeValue;
        auto found = last.find(pipe);
        if (found != last.end()) {
            required[2 * found->second][2 * a] = 1;
            required[2 * found->second + 1][2 * a + 1] = 1;
        }
        last[pipe] = a;
    }
    auto selected = required;
    auto conflict = [&](const auto* a, const auto* b) {
        pto::DepBaseMemInfoPairVec witnesses;
        return input.memory().DepBetween(a->defVec, b->useVec, witnesses) ||
               input.memory().DepBetween(a->useVec, b->defVec, witnesses) ||
               input.memory().DepBetween(a->defVec, b->defVec, witnesses);
    };
    for (std::size_t a = 0; a < count; ++a) {
        for (std::size_t b = a + 1; b < count; ++b) {
            if (conflict(phases[a % phases.size()], phases[b % phases.size()])) {
                required[2 * a + 1][2 * b] = 1;
            }
        }
    }
    for (auto id : upper.selected().retained()) {
        const auto& edge = upper.selected().generators()[id].edge;
        if (edge.distance > llvm::DynamicAPInt(trips)) {
            continue;
        }
        unsigned distance = 0;
        while (llvm::DynamicAPInt(distance) < edge.distance) {
            ++distance;
        }
        for (unsigned i = 0; i + distance < trips; ++i) {
            selected[2 * (i * phases.size() + edge.source) + 1][2 * ((i + distance) * phases.size() + edge.consumer)] =
                1;
        }
    }
    close(required);
    close(selected);
    int64_t excess = 0, selectedPairs = 0;
    for (std::size_t a = 0; a < count; ++a) {
        for (std::size_t b = 0; b < count; ++b) {
            bool need = required[2 * a + 1][2 * b], got = selected[2 * a + 1][2 * b];
            if (need && !got) {
                return false;
            }
            selectedPairs += got;
            excess += got && !need;
        }
    }
    auto bound = upper.excessBound(llvm::DynamicAPInt(trips));
    return bound == llvm::DynamicAPInt(selectedPairs) && bound >= llvm::DynamicAPInt(excess);
}
bool check(func::FuncOp function)
{
    pto::SyncInput input;
    if (failed(input.build(function))) {
        return false;
    }
    if (function.getName() == "partial_rmw") {
        bool checked = false;
        function.walk([&](pto::TAxpyOp axpy) {
            SmallVector<MemoryEffects::EffectInstance> effects;
            axpy.getEffects(effects);
            unsigned reads = 0, writes = 0;
            for (const auto& effect : effects) {
                if (effect.getValue() == axpy.getScalar()) {
                    return;
                }
                reads += isa<MemoryEffects::Read>(effect.getEffect());
                writes += isa<MemoryEffects::Write>(effect.getEffect());
            }
            checked = reads == 2 && writes == 1;
        });
        if (!checked) {
            return false;
        }
    }
    auto trace = fs::TraceDemandAnalysis::build(function, input);
    if (failed(trace)) {
        return false;
    }
    scf::ForOp loop;
    function.walk([&](scf::ForOp op) { loop = op; });
    SmallVector<const pto::CompoundInstanceElement*> phases;
    for (auto& site : trace->sites()) {
        if (loop->isProperAncestor(site.phase->elementOp)) {
            phases.push_back(site.phase);
        }
    }
    fs::CostLedger costs(true);
    std::string reason;
    auto upper = fs::FixedBodyUpper::build(input, phases, costs, reason);
    if (function.getName() == "changing") {
        if (succeeded(upper)) {
            // Concrete addresses: dynamic base i and fixed base 8192, size
            // 8192. They are disjoint in row zero and overlap in row one.
            // Every concrete forward WAW must be contained in the upper graph.
            for (unsigned a = 0; a < 6; ++a) {
                for (unsigned b = a + 1; b < 6; ++b) {
                    uint64_t first = a % 2 == 0 ? a / 2 : 8192;
                    uint64_t second = b % 2 == 0 ? b / 2 : 8192;
                    if (first >= second + 8192 || second >= first + 8192) {
                        continue;
                    }
                    auto reaches = (*upper)->selected().reaches(
                        a % 2, b % 2, fs::PeriodicEventKind::Start, llvm::DynamicAPInt(b / 2 - a / 2));
                    if (failed(reaches) || !*reaches) {
                        return false;
                    }
                }
            }
        }
        fs::RegionalRequests requests(function, input, *trace, costs);
        auto result = requests.request(
            loop, requests.contextFor(loop), fs::RegionalMode::Modeled, fs::AnalysisNeeds::defaultPolicy(),
            fs::RegionalRepresentation::ArithmeticRelations, fs::RegionalPreparation::SelectorMatching);
        if (failed(upper)) {
            // A typed occurrence-geometry obligation is an honest rejection;
            // no narrower same-iteration exclusion may certify upper results.
            return reason.find("occurrence-invariant") != std::string::npos &&
                   (!result.analysis || !result.analysis->upper);
        }
        if (result.analysis && result.analysis->upper) {
            for (unsigned trips : {0U, 1U, 3U}) {
                if (!oracle(**upper, input, trips)) {
                    return false;
                }
            }
            return result.analysis->contract.closure == fs::SelectedClosure::SoundUpper;
        }
        // Scalar/geometry lowering qualification is separate from the sound
        // no-kill origin graph; retain the explicit route obligation.
        return !result.obligation.empty() || (result.analysis && !result.analysis->upper);
    }
    if (failed(upper)) {
        llvm::errs() << reason << '\n';
        return false;
    }
    if ((*upper)->consolidated().size() > phases.size() * (*upper)->selected().pipes().size()) {
        return false;
    }
    for (unsigned trips : {0U, 1U, 2U, 4U}) {
        if (!oracle(**upper, input, trips)) {
            llvm::errs() << "upper finite oracle failed at " << trips << " trips\n";
            return false;
        }
    }
    if (function.getName() == "relay") {
        // A,X,R,B: reducing A->R->B first must avoid strengthening X->B.
        auto extra = (*upper)->selected().reaches(1, 3, fs::PeriodicEventKind::Start, llvm::DynamicAPInt(0));
        if (failed(extra) || *extra) {
            return false;
        }
    }
    auto contract = fs::FixedBodyUpper::contract();
    if (contract.accepts(fs::AnalysisNeeds::modeledCovers(), reason) ||
        !contract.accepts(fs::AnalysisNeeds::defaultPolicy(), reason) ||
        contract.accepts(fs::AnalysisNeeds::minimumExact(), reason)) {
        return false;
    }
    fs::RegionalRequests requests(function, input, *trace, costs);
    auto upperNeeds = fs::AnalysisNeeds::defaultPolicy();
    auto strict = fs::AnalysisNeeds::modeledCovers();
    auto context = requests.contextFor(loop);
    auto first = requests.request(loop, context, fs::RegionalMode::Modeled, upperNeeds);
    auto second = requests.request(loop, context, fs::RegionalMode::Modeled, strict);
    if (!first.analysis || (second.analysis && second.analysis->contract.closure == fs::SelectedClosure::SoundUpper)) {
        return false;
    }
    auto repeated = requests.request(loop, context, fs::RegionalMode::Modeled, upperNeeds);
    if (repeated.analysis != first.analysis) {
        return false;
    }
    if (function.getName() == "local_upper") {
        auto strictMatching = requests.request(
            loop, context, fs::RegionalMode::Modeled, strict, fs::RegionalRepresentation::NativeSummaries,
            fs::RegionalPreparation::SelectorMatching);
        if (strictMatching.analysis) {
            return false;
        }
        auto matching = requests.request(
            loop, context, fs::RegionalMode::Modeled, upperNeeds, fs::RegionalRepresentation::NativeSummaries,
            fs::RegionalPreparation::SelectorMatching);
        if (!matching.analysis || !matching.analysis->upper ||
            matching.analysis->contract.closure != fs::SelectedClosure::SoundUpper) {
            return false;
        }
    }
    if (function.getName() == "local_upper") {
        auto emitted = fs::emitDirectDemands(function, input, *trace, costs);
        // Production must realize the ready upper plan before attempting the
        // optional unrestricted symbolic composition extension.
        if (llvm::any_of(emitted.attempts, [](const auto& attempt) {
                return attempt.route == "presburger-region-composition";
            })) {
            return false;
        }
        if (!emitted.emitted || !emitted.selected || !emitted.selected->upper ||
            emitted.logicalContract.closure != fs::SelectedClosure::SoundUpper) {
            llvm::errs() << "upper logical emission: " << emitted.reason << '\n';
            return false;
        }
        auto evidence = dyn_cast<DictionaryAttr>(fs::retainSelectedAnalysis(function, *trace, emitted));
        if (!evidence || evidence.getAs<StringAttr>("demand_name").getValue() != "F_hat") {
            return false;
        }
    }
    if (function.getName() == "parent_upper") {
        auto needs = upperNeeds;
        needs.interfaces |= fs::interfaceBit(fs::DemandInterface::RegionQueries);
        auto parent = requests.request(
            function, requests.context(), fs::RegionalMode::Modeled, needs, fs::RegionalRepresentation::NativeSummaries,
            fs::RegionalPreparation::SelectorMatching);
        if (!parent.analysis || parent.analysis->contract.closure != fs::SelectedClosure::SoundUpper ||
            !parent.analysis->reachability) {
            return false;
        }
        auto strictParent = requests.request(
            function, requests.context(), fs::RegionalMode::Modeled, strict,
            fs::RegionalRepresentation::NativeSummaries, fs::RegionalPreparation::SelectorMatching);
        if (strictParent.analysis && strictParent.analysis->contract.closure == fs::SelectedClosure::SoundUpper) {
            return false;
        }
        for (unsigned trips : {1U, 3U}) {
            auto entry = parent.analysis->reachability->contains(
                {{0, fs::PeriodicEventKind::Completion}, {}},
                {{1, fs::PeriodicEventKind::Start}, {llvm::DynamicAPInt(0)}}, {llvm::DynamicAPInt(trips)});
            auto exit = parent.analysis->reachability->contains(
                {{3, fs::PeriodicEventKind::Completion}, {llvm::DynamicAPInt(trips - 1)}},
                {{4, fs::PeriodicEventKind::Start}, {}}, {llvm::DynamicAPInt(trips)});
            if (!entry.succeeded() || !entry.value || !exit.succeeded() || !exit.value) {
                return false;
            }
        }
        auto child = requests.request(
            loop, context, fs::RegionalMode::Modeled, needs, fs::RegionalRepresentation::NativeSummaries,
            fs::RegionalPreparation::SelectorMatching);
        if (!child.analysis || !child.analysis->upper) {
            return false;
        }
        fs::DirectEmissionResult retained;
        retained.selected = child.analysis;
        auto encoded = dyn_cast<DictionaryAttr>(fs::retainSelectedAnalysis(function, *trace, retained));
        if (!encoded || encoded.getAs<StringAttr>("demand_name").getValue() != "F_hat") {
            return false;
        }
        auto terms = encoded.getAs<ArrayAttr>("excess_terms");
        if (!terms || terms.empty()) {
            return false;
        }
        for (auto [id, term] : llvm::enumerate(child.analysis->upper->excessTerms())) {
            auto actual = dyn_cast<DictionaryAttr>(terms[id]);
            if (!actual ||
                actual.getAs<IntegerAttr>("consumer").getInt() !=
                    static_cast<int64_t>(child.analysis->sites[term.consumer]) ||
                actual.getAs<IntegerAttr>("pipe").getInt() !=
                    static_cast<int64_t>(child.analysis->periodic.pipes()[term.pipe])) {
                return false;
            }
        }
        auto clone = OwningOpRef<func::FuncOp>(cast<func::FuncOp>(function->clone()));
        clone.get()->setAttr("pto.frontier.analysis", encoded);
        std::string text;
        llvm::raw_string_ostream output(text);
        output << "module {\n";
        clone.get().print(output);
        output << "\n}\n";
        auto parsed = parseSourceString<ModuleOp>(output.str(), function.getContext());
        if (!parsed || (*parsed->getOps<func::FuncOp>().begin())->getAttr("pto.frontier.analysis") != encoded) {
            return false;
        }
        retained.selected = parent.analysis;
        auto parentEvidence = dyn_cast<DictionaryAttr>(fs::retainSelectedAnalysis(function, *trace, retained));
        if (!parentEvidence || parentEvidence.getAs<StringAttr>("selected_closure").getValue() != "sound-upper" ||
            !parentEvidence.getAs<StringAttr>("excess_formula")) {
            return false;
        }
    }
    if (function.getName() == "partial_rmw") {
        unsigned hazards = 0;
        for (const auto& origin : (*upper)->distances()) {
            if (origin.maximum) {
                return false;
            }
            hazards |= 1U << static_cast<unsigned>(origin.hazard);
        }
        if (hazards != 7) {
            return false;
        }
    }
    if (function.getName() == "local_upper") {
        // The unsupported request follows identical exact routes. Allowing
        // upper must not add effects/backend work before capability rejection.
        auto unavailable = strict;
        unavailable.interfaces |= fs::interfaceBit(fs::DemandInterface::CellBoundaries);
        fs::CostLedger exactCosts(true), upperCosts(true);
        fs::RegionalRequests exactRequests(function, input, *trace, exactCosts);
        fs::RegionalRequests upperRequests(function, input, *trace, upperCosts);
        auto rejectedExact =
            exactRequests.request(loop, exactRequests.contextFor(loop), fs::RegionalMode::Modeled, unavailable);
        unavailable.allowSoundUpper = true;
        auto rejectedUpper =
            upperRequests.request(loop, upperRequests.contextFor(loop), fs::RegionalMode::Modeled, unavailable);
        if (rejectedExact.analysis || rejectedUpper.analysis ||
            exactCosts.stage(fs::CostStage::Effects).invocations !=
                upperCosts.stage(fs::CostStage::Effects).invocations ||
            exactCosts.stage(fs::CostStage::Backend).invocations !=
                upperCosts.stage(fs::CostStage::Backend).invocations) {
            return false;
        }
    }
    auto mathematical = upperNeeds;
    mathematical.interfaces |= fs::interfaceBit(fs::DemandInterface::RegionQueries);
    auto queried = requests.request(
        loop, context, fs::RegionalMode::Modeled, mathematical, fs::RegionalRepresentation::ArithmeticRelations);
    if (!queried.analysis || !queried.analysis->reachability) {
        llvm::errs() << "upper query obligation: " << queried.obligation << '\n';
        return false;
    }
    return true;
}
} // namespace
int runFixedBodyUpperChecks(llvm::StringRef path, mlir::MLIRContext& context)
{
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(path, &context);
    if (!module) {
        return 1;
    }
    for (auto function : module->getOps<mlir::func::FuncOp>()) {
        if (!check(function)) {
            llvm::errs() << "upper check failed: " << function.getName() << '\n';
            return 1;
        }
    }
    llvm::outs() << "verified sound upper coverage, finite excess certificate, delayed locals and request isolation\n";
    return 0;
}
