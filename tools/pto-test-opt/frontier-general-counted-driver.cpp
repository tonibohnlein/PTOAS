// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Independent finite closure/cover oracle. No numeric expansion in production.
#include "../../lib/PTO/Transforms/FrontierSynch/RegionalRequests.h"
#include "../../lib/PTO/Transforms/FrontierSynch/GeneralQueries.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <cstdlib>
namespace {
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
using Int = llvm::DynamicAPInt;
using Graph = SmallVector<SmallVector<unsigned char>>;
struct Occurrence { std::size_t site; SmallVector<Int> coordinates; };
void close(Graph& graph)
{
    for (std::size_t k = 0; k < graph.size(); ++k) {
        for (std::size_t a = 0; a < graph.size(); ++a) {
            for (std::size_t b = 0; b < graph.size(); ++b) { graph[a][b] |= graph[a][k] && graph[k][b]; }
        }
    }
}
SmallVector<Int> tuple(const Occurrence& occurrence, unsigned depth, unsigned kind)
{
    SmallVector<Int> values{Int(static_cast<int64_t>(occurrence.site))};
    llvm::append_range(values, occurrence.coordinates);
    while (values.size() <= depth) { values.push_back(Int(0)); }
    values.push_back(Int(kind)); return values;
}
bool contains(const presburger::PresburgerRelation& relation, const Occurrence& a, const Occurrence& b,
              unsigned depth, unsigned s, unsigned t, int n, int c)
{
    auto point = tuple(a, depth, s);
    llvm::append_range(point, tuple(b, depth, t));
    point.push_back(Int(n)); point.push_back(Int(c));
    return relation.containsPoint(point);
}
bool oracle(const fs::SelectedAnalysis& selected, const pto::SyncInput& input, scf::ForOp outer, int n, int c)
{
    auto query = selected.general;
    SmallVector<Occurrence> word;
    SmallVector<std::size_t> loopSites;
    for (auto id : selected.sites) {
        auto* anchor = query->schema->sites()[id].phase->elementOp;
        if (outer->isProperAncestor(anchor)) { loopSites.push_back(id); }
    }
    if (loopSites.size() != 2) { return false; }
    for (auto id : selected.sites) {
        if (id == loopSites.front()) {
            for (int i = 0; i < std::min(n, 4); ++i) {
                word.push_back({loopSites[0], {Int(i)}});
                for (int j = 0; j < 2 * i + c; ++j) {
                    word.push_back({loopSites[1], {Int(i), Int(j)}});
                }
            }
        } else if (id != loopSites.back()) { word.push_back({id, {}}); }
    }
    Graph native(2 * word.size(), SmallVector<unsigned char>(2 * word.size()));
    for (std::size_t a = 0; a < word.size(); ++a) {
        native[2 * a][2 * a + 1] = 1;
        for (std::size_t b = a + 1; b < word.size(); ++b) {
            if (query->schema->sites()[word[a].site].phase->kPipeValue ==
                query->schema->sites()[word[b].site].phase->kPipeValue) {
                native[2 * a][2 * b] = 1; native[2 * a + 1][2 * b + 1] = 1;
            }
        }
    }
    auto required = native;
    for (std::size_t a = 0; a < word.size(); ++a) {
        for (std::size_t b = a + 1; b < word.size(); ++b) {
            auto* first = query->schema->sites()[word[a].site].phase;
            auto* second = query->schema->sites()[word[b].site].phase;
            pto::DepBaseMemInfoPairVec witnesses;
            if (input.memory().DepBetween(first->defVec, second->useVec, witnesses) ||
                input.memory().DepBetween(first->useVec, second->defVec, witnesses) ||
                input.memory().DepBetween(first->defVec, second->defVec, witnesses)) {
                required[2 * a + 1][2 * b] = 1;
            }
        }
    }
    close(native); close(required);
    auto depth = query->schema->coordinateDepth();
    for (std::size_t a = 0; a < word.size(); ++a) {
        for (std::size_t b = 0; b < word.size(); ++b) {
            for (unsigned s : {0U, 1U}) {
                for (unsigned t : {0U, 1U}) {
                    bool expected = (a == b && s == t) || required[2 * a + s][2 * b + t];
                    if (contains(query->reachability, word[a], word[b], depth, s, t, n, c) != expected) {
                        llvm::errs() << "reach mismatch " << n << ',' << c << ':' << a << ',' << b << '\n';
                        return false;
                    }
                }
            }
            bool cover = required[2 * a + 1][2 * b] && !native[2 * a + 1][2 * b];
            for (std::size_t k = 0; k < required.size(); ++k) {
                if (k != 2 * a + 1 && k != 2 * b && required[2 * a + 1][k] && required[k][2 * b]) { cover = false; }
            }
            if (contains(query->minimum, word[a], word[b], depth, 1, 0, n, c) != cover) {
                llvm::errs() << "cover mismatch " << n << ',' << c << ':' << a << ',' << b << '\n'; return false;
            }
        }
    }
    if (selected.generalEndpoints) {
        for (const auto& endpoint : selected.generalEndpoints->endpoints) {
            for (const auto& occurrence : word) {
                auto pipe = endpoint.outgoing ? endpoint.source : endpoint.target;
                if (query->schema->sites()[occurrence.site].phase->kPipeValue != pipe) { continue; }
                auto point = tuple(occurrence, depth, endpoint.outgoing ? 1 : 0);
                point.push_back(Int(n)); point.push_back(Int(c));
                auto value = endpoint.function.valueAt(point);
                unsigned partners = 0;
                for (const auto& other : word) {
                    auto otherPipe = query->schema->sites()[other.site].phase->kPipeValue;
                    if (otherPipe != (endpoint.outgoing ? endpoint.target : endpoint.source)) { continue; }
                    auto a = endpoint.outgoing ? occurrence : other, b = endpoint.outgoing ? other : occurrence;
                    if (!contains(query->minimum, a, b, depth, 1, 0, n, c)) { continue; }
                    ++partners;
                    if (!value || *value != tuple(other, depth, endpoint.outgoing ? 0 : 1)) { return false; }
                }
                if (partners > 1 || bool(value) != (partners == 1)) { return false; }
            }
        }
    }
    return true;
}
bool repeated(func::FuncOp function, fs::RegionalRequests& requests, scf::ForOp outer)
{
    auto needs = fs::AnalysisNeeds::modeledCovers();
    needs.interfaces |= fs::interfaceBit(fs::DemandInterface::RegionQueries);
    auto result = requests.request(outer, requests.contextFor(outer), fs::RegionalMode::Modeled, needs,
                                  fs::RegionalRepresentation::PresburgerRelations);
    for (const auto& attempt : requests.attempts()) {
        if (attempt.route == "presburger-region-composition" && attempt.outcome == "ready") { return false; }
    }
    if (!result.analysis || !result.analysis->minimum || !result.analysis->reachability ||
        result.analysis->sites.size() != 2) { return false; }
    fs::SignedPoint reader{{result.analysis->sites[1], fs::PeriodicEventKind::Completion}, {Int(0)}};
    fs::SignedPoint writer{{result.analysis->sites[0], fs::PeriodicEventKind::Start}, {Int(1)}};
    auto reach = result.analysis->reachability->contains(reader, writer, {Int(2), Int(0)});
    auto cover = result.analysis->minimum->contains(reader, writer, {Int(2), Int(0)});
    return reach.succeeded() && reach.value && cover.succeeded() && cover.value;
}
bool cutOrder(const fs::SelectedAnalysis& selected, IRMapping& mapping)
{
    if (selected.sites.size() != 3 || !selected.general) {
        llvm::errs() << "cut-order requires three original sites and general queries\n"; return false;
    }
    auto schema = selected.general->schema;
    auto producer = selected.sites[0], source = selected.sites[1], consumer = selected.sites[2];
    // Both prerequisites must actually be retained for this same consumer.
    SmallVector<Int> local{Int(static_cast<int64_t>(source)), Int(1),
                           Int(static_cast<int64_t>(consumer)), Int(0)};
    auto cross = local; cross[0] = Int(static_cast<int64_t>(producer));
    if (!selected.general->minimum.containsPoint(local) || !selected.general->minimum.containsPoint(cross)) {
        llvm::errs() << "cut-order retained prerequisites absent\n"; return false;
    }
    auto* start = mapping.lookup(schema->sites()[producer].phase->elementOp);
    auto* end = mapping.lookup(schema->sites()[consumer].phase->elementOp);
    auto* intermediate = mapping.lookup(schema->sites()[source].phase->elementOp);
    unsigned category = 0;
    bool barrier = false, wait = false, valid = true;
    for (auto* operation = start->getNextNode(); operation && operation != end; operation = operation->getNextNode()) {
        if (operation == intermediate) { category = 0; continue; }
        operation->walk([&](Operation* command) {
            unsigned next = 0;
            if (isa<pto::BarrierOp>(command)) { next = 1; barrier = true; }
            else if (isa<pto::LogicalWaitOp>(command)) { next = 2; wait = true; }
            else if (!isa<pto::LogicalSetOp>(command)) { return; }
            if (next < category) { valid = false; }
            category = next;
        });
    }
    if (!valid || !barrier || !wait) {
        llvm::errs() << "cut-order invalid=" << !valid << " barrier=" << barrier << " wait=" << wait << '\n';
        end->getParentOp()->print(llvm::errs());
    }
    return valid && barrier && wait;
}
bool retainedRelation(ArrayAttr encoded, const presburger::PresburgerRelation& source)
{
    if (!encoded || encoded.size() != source.getNumDisjuncts()) { return false; }
    for (auto [index, poly] : llvm::enumerate(source.getAllDisjuncts())) {
        auto piece = dyn_cast<DictionaryAttr>(encoded[index]);
        if (!piece) { return false; }
        const std::pair<const char*, unsigned> dimensions[] = {
            {"domain", poly.getNumDomainVars()}, {"range", poly.getNumRangeVars()},
            {"symbols", poly.getNumSymbolVars()}, {"locals", poly.getNumLocalVars()}};
        for (auto [name, count] : dimensions) {
            auto dimension = piece.getAs<IntegerAttr>(name);
            if (!dimension || dimension.getInt() != count) { return false; }
        }
        for (bool equality : {true, false}) {
            auto rows = piece.getAs<ArrayAttr>(equality ? "equalities" : "inequalities");
            unsigned count = equality ? poly.getNumEqualities() : poly.getNumInequalities();
            if (!rows || rows.size() != count) { return false; }
            for (unsigned row = 0; row < count; ++row) {
                auto values = dyn_cast<ArrayAttr>(rows[row]);
                auto expected = equality ? poly.getEquality(row) : poly.getInequality(row);
                if (!values || values.size() != expected.size()) { return false; }
                for (auto [column, coefficient] : llvm::enumerate(expected)) {
                    auto value = dyn_cast<StringAttr>(values[column]);
                    std::string decimal; llvm::raw_string_ostream text(decimal); text << coefficient;
                    if (!value || value.getValue() != text.str()) { return false; }
                }
            }
        }
    }
    return true;
}
bool retainedEvidence(func::FuncOp function, const fs::TraceDemandAnalysis& trace,
                      fs::DirectEmissionResult& emission, func::FuncOp clone)
{
    // Call the production builder directly with no dump/report request. Check
    // coefficient identity, not only the shape or presence of the certificate.
    auto encoded = dyn_cast<DictionaryAttr>(fs::retainSelectedAnalysis(function, trace, emission));
    if (!encoded || !emission.selected || !emission.selected->general) { return false; }
    const auto& query = *emission.selected->general;
    const std::pair<const char*, const presburger::PresburgerRelation*> relations[] = {
        {"context", &query.context}, {"presence", &query.present}, {"minimum", &query.minimum},
        {"native", &query.native}, {"reachability", &query.reachability}};
    for (auto [name, relation] : relations) {
        if (!retainedRelation(encoded.getAs<ArrayAttr>(name), *relation)) { return false; }
    }
    auto depths = encoded.getAs<ArrayAttr>("coordinate_depths");
    auto parameters = encoded.getAs<ArrayAttr>("parameters");
    if (!depths || depths.size() != query.schema->sites().size() ||
        !parameters || parameters.size() != query.schema->parameters().size()) { return false; }
    for (auto [id, site] : llvm::enumerate(query.schema->sites())) {
        auto depth = dyn_cast<IntegerAttr>(depths[id]);
        if (!depth || depth.getInt() != static_cast<int64_t>(site.coordinates.size())) { return false; }
    }
    for (auto [id, value] : llvm::enumerate(query.schema->parameters())) {
        auto argument = dyn_cast<BlockArgument>(value);
        auto identity = dyn_cast<IntegerAttr>(parameters[id]);
        if (!argument || argument.getOwner() != &function.front() || !identity ||
            identity.getInt() != argument.getArgNumber()) { return false; }
    }
    clone->setAttr("pto.frontier.analysis", encoded);
    std::string printed; llvm::raw_string_ostream text(printed);
    text << "module {\n"; clone.print(text); text << "\n}\n";
    auto reparsed = parseSourceString<ModuleOp>(text.str(), function.getContext());
    if (!reparsed) { return false; }
    auto parsed = *reparsed->getOps<func::FuncOp>().begin();
    if (parsed->getAttr("pto.frontier.analysis") != encoded) { return false; }
    for (auto value : query.schema->parameters()) {
        auto argument = cast<BlockArgument>(value);
        if (argument.getArgNumber() >= parsed.getNumArguments() ||
            parsed.getArgument(argument.getArgNumber()).getType() != argument.getType()) { return false; }
    }
    return true;
}
bool mixedPolicy(func::FuncOp function, const pto::SyncInput& input, const fs::TraceDemandAnalysis& trace)
{
    fs::CostLedger costs(true);
    SmallVector<fs::AnalysisAttempt> attempts;
    std::string reason;
    auto selected = fs::selectAnalysis(function, input, trace, attempts, reason, costs);
    if (!selected || selected->kind != fs::SelectedAnalysis::Kind::General ||
        selected->contract.closure != fs::SelectedClosure::SoundUpper) {
        llvm::errs() << "mixed policy selection: " << reason << '\n';
        return false;
    }
    auto ready = llvm::find_if(attempts, [](const auto& attempt) {
        return attempt.route == "presburger-region-composition" && attempt.outcome == "ready";
    });
    if (ready == attempts.end() || selected->regionalChildren.size() != 2 ||
        !llvm::any_of(selected->regionalChildren, [](const auto& child) { return bool(child->upper); })) {
        return false;
    }
    // A single known writer-to-first-reader occurrence must survive the merge.
    auto depth = selected->general->schema->coordinateDepth();
    Occurrence writer{0, {Int(0)}}, reader{1, {Int(0), Int(0)}};
    if (!contains(selected->general->minimum, writer, reader, depth, 1, 0, 2, 1)) { return false; }
    fs::RegionalRequests requests(function, input, trace, costs);
    auto strict = requests.request(function, requests.context(), fs::RegionalMode::Modeled,
        fs::AnalysisNeeds::modeledCovers(), fs::RegionalRepresentation::NativeSummaries,
        fs::RegionalPreparation::SelectorMatching, fs::RegionalRoutePolicy::GeneralExtension);
    if (strict.analysis || strict.obligation.empty()) { return false; }
    IRMapping mapping;
    OwningOpRef<func::FuncOp> clone(cast<func::FuncOp>(function->clone(mapping)));
    fs::DirectEmissionResult emitted;
    if (failed(fs::emitGeneralEndpoints(mapping, *selected, emitted)) || failed(verify(*clone))) {
        llvm::errs() << "mixed logical emission: " << emitted.reason << '\n';
        return false;
    }
    emitted.selected = selected;
    return retainedEvidence(function, trace, emitted, *clone);
}
bool check(func::FuncOp function)
{
    auto begin = std::chrono::steady_clock::now();
    auto stage = [&](llvm::StringRef label) {
        auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        llvm::errs() << function.getName() << ": " << label << " at " << elapsed << "s\n";
    };
    pto::SyncInput input;
    if (failed(input.build(function))) { return false; }
    auto trace = fs::TraceDemandAnalysis::build(function, input);
    if (failed(trace)) { return false; }
    scf::ForOp outer;
    for (auto loop : function.getOps<scf::ForOp>()) { outer = loop; }
    bool ordering = function.getName() == "cut_order";
    if (!outer && !ordering) { return false; }
    std::string before; llvm::raw_string_ostream original(before); function.print(original);
    fs::CostLedger costs(true);
    if (function.getName() == "mixed_policy") {
        bool valid = mixedPolicy(function, input, *trace);
        std::string after; llvm::raw_string_ostream unchanged(after); function.print(unchanged);
        return valid && before == after;
    }
    fs::RegionalRequests requests(function, input, *trace, costs);
    if (function.getName() == "repeated") {
        bool valid = repeated(function, requests, outer);
        std::string after; llvm::raw_string_ostream unchanged(after); function.print(unchanged);
        return valid && before == after;
    }
    auto needs = fs::AnalysisNeeds::modeledCovers();
    needs.interfaces |= fs::interfaceBit(fs::DemandInterface::RegionQueries);
    auto result = requests.request(function, requests.context(), fs::RegionalMode::Modeled, needs,
        fs::RegionalRepresentation::PresburgerRelations, fs::RegionalPreparation::SelectorMatching);
    if (!result.analysis || result.analysis->kind != fs::SelectedAnalysis::Kind::General ||
        !result.analysis->generalEndpoints || result.analysis->contract.effects != fs::EffectDomain::SharedModeled) {
        for (const auto& attempt : requests.attempts()) {
            llvm::errs() << attempt.route << ':' << attempt.obligation << '\n';
        }
        return false;
    }
    stage("request and selector qualification complete");
    for (auto kind : {fs::CostStage::Backend, fs::CostStage::Merge, fs::CostStage::Selectors}) {
        llvm::errs() << function.getName() << ": exclusive stage " << static_cast<unsigned>(kind)
                     << " = " << costs.stage(kind).nanoseconds / 1.0e9 << "s\n";
    }
    if (function.getName() == "parent" && result.analysis->regionalChildren.empty()) { return false; }
    if (!ordering) {
        for (int n : {0, 1, 2, 4}) {
            for (int c : {-3, -1, 0, 2}) { if (!oracle(*result.analysis, input, outer, n, c)) { return false; } }
        }
        auto child = requests.request(outer, requests.contextFor(outer), fs::RegionalMode::Modeled, needs,
            fs::RegionalRepresentation::PresburgerRelations, fs::RegionalPreparation::SelectorMatching);
        if (!child.analysis || !child.analysis->generalEndpoints) { return false; }
        Occurrence absent{child.analysis->sites.front(), {Int(0)}};
        auto depth = child.analysis->general->schema->coordinateDepth();
        if (contains(child.analysis->general->reachability, absent, absent, depth, 0, 0, 0, -3)) { return false; }
        for (const auto& endpoint : child.analysis->generalEndpoints->endpoints) {
            auto point = tuple(absent, depth, endpoint.outgoing ? 1 : 0);
            point.push_back(Int(0)); point.push_back(Int(-3));
            if (endpoint.function.valueAt(point)) { return false; }
        }
    }
    stage("finite oracle complete");
    // Exercise the genuine compiler endpoint helper on a pending clone. This
    // certifies only logical IR verification, never native or physical hardware.
    IRMapping mapping;
    OwningOpRef<func::FuncOp> clone(cast<func::FuncOp>(function->clone(mapping)));
    fs::DirectEmissionResult emission;
    if (failed(fs::emitGeneralEndpoints(mapping, *result.analysis, emission)) || failed(verify(*clone))) {
        llvm::errs() << emission.reason << '\n'; return false;
    }
    emission.selected = result.analysis;
    if (!retainedEvidence(function, *trace, emission, *clone)) { return false; }
    stage("logical emission and diagnostic-off evidence roundtrip complete");
    if (ordering && !cutOrder(*result.analysis, mapping)) { return false; }
    std::string after; llvm::raw_string_ostream unchanged(after); function.print(unchanged);
    return before == after;
}
} // namespace
int runGeneralCountedChecks(llvm::StringRef path, mlir::MLIRContext& context)
{
    auto module = parseSourceFile<ModuleOp>(path, &context);
    if (!module) { return 1; }
    unsigned count = 0;
    const char* selectedFunction = std::getenv("PTO_GENERAL_TEST_FUNCTION");
    for (auto function : module->getOps<func::FuncOp>()) {
        if (selectedFunction && function.getName() != selectedFunction) { continue; }
        if (!check(function)) {
            llvm::errs() << "general counted check failed: " << function.getName() << '\n'; return 1;
        }
        ++count;
    }
    if (count != (selectedFunction ? 1U : 4U)) { return 1; }
    llvm::outs() << "verified general affine counted queries, negative rows, uniform selectors and parent crossings\n";
    return 0;
}
