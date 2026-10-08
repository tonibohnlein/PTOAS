// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Check canonical occurrence ownership independently of selected query exports.
#include "PTO/Transforms/FrontierSynch/BoundingRegionalAnalysis.h"
#include "PTO/Transforms/FrontierSynch/RegionalNumericalInterface.h"
#include "PTO/Transforms/FrontierSynch/RequirementProvenance.h"
#include "PTO/Transforms/InsertSync/SyncInput.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
using Expr = fs::RegionExpressions::Id;
using Event = fs::RegionalEvent;
fs::RegionalAnalysis domainFor(func::FuncOp function, const pto::SyncInput& input)
{
    fs::RegionalAnalysis domain;
    domain.expressions = std::make_shared<fs::RegionExpressions>();
    domain.accessModel = &input.accesses();
    domain.gmAliasPolicy = input.memory().gmPolicy();
    SmallVector<scf::ForOp> loops;
    function.walk([&loops](scf::ForOp loop) { loops.push_back(loop); });
    if (loops.size() < 2) { return domain; }
    // Query-only sites need no endpoint phase/cut; their domain is still nested.
    domain.anchors.resize(2);
    domain.occurrenceLoops.assign(2, loops[1]);
    domain.outerLoops.assign(2, {loops[0]});
    domain.outerDivisors.assign(2, {2});
    auto arena = domain.expressions;
    const auto predicate = arena->input(function.getArgument(1));
    domain.firstOrdinal = arena->constant(3);
    domain.endpointSiteGuard = predicate;
    domain.presence = [arena, predicate](Event event) -> std::optional<Expr> {
        if (event.type >= 2 || event.visits.size() != 1) { return std::nullopt; }
        auto inSlice = arena->land(arena->le(arena->constant(3), event.ordinal),
                                  arena->lt(event.ordinal, arena->constant(5)));
        auto present = arena->land(inSlice, arena->lt(event.visits[0], arena->constant(3)));
        return event.type == 0 ? arena->land(predicate, present) : present;
    };
    domain.referenceBefore = [arena](Event a, Event b) -> std::optional<Expr> {
        if (a.visits.size() != 1 || b.visits.size() != 1) { return std::nullopt; }
        auto local = arena->lor(arena->lt(a.ordinal, b.ordinal),
            arena->land(arena->eq(a.ordinal, b.ordinal), arena->boolean(a.type > b.type)));
        return arena->lor(arena->lt(a.visits[0], b.visits[0]),
            arena->land(arena->eq(a.visits[0], b.visits[0]), local));
    };
    domain.endpointEventGuard = [arena](Event event) -> std::optional<Expr> {
        if (event.visits.size() != 1) { return std::nullopt; }
        return arena->eq(arena->rem(event.visits[0], arena->constant(2)), arena->constant(1));
    };
    return domain;
}
fs::RegionalOrderQueries queriesFor(fs::OrderContext context, bool crossing)
{
    fs::RegionalOrderQueries queries;
    queries.reachability = [context, crossing](Event a, Event b) -> std::optional<Expr> {
        auto arena = context->expressions();
        if (!arena->constantValue(a.ordinal) || !arena->constantValue(b.ordinal)) { return std::nullopt; }
        auto pa = fs::regionalPresence(context->domain(), a), pb = fs::regionalPresence(context->domain(), b);
        if (!pa || !pb) { return std::nullopt; }
        auto present = arena->land(*pa, *pb);
        auto after = fs::regionalReferenceBefore(context->domain(), b, a);
        if (!after) { return std::nullopt; }
        bool kinds = a.kind != fs::PeriodicEventKind::Completion || b.kind != fs::PeriodicEventKind::Start;
        bool native = a.type == b.type && kinds;
        bool added = crossing && a.type == 1 && b.type == 0;
        return arena->land(present, arena->land(arena->lnot(*after), arena->boolean(native || added)));
    };
    return queries;
}
bool checkViews(func::FuncOp function, const pto::SyncInput& input)
{
    auto domain = domainFor(function, input);
    if (domain.anchors.size() != 2) { return false; }
    auto calls = std::make_shared<unsigned>(0);
    domain.reachability = [calls](Event, Event) -> std::optional<Expr> { ++*calls; return std::nullopt; };
    domain.prepare = [calls]() -> FailureOr<std::unique_ptr<fs::PreparedLogicalPlan>> {
        ++*calls; return failure();
    };
    domain.prepareWithVisits = [calls](ArrayRef<scf::ForOp>) -> FailureOr<std::unique_ptr<fs::PreparedLogicalPlan>> {
        ++*calls; return failure();
    };
    domain.prepareFiltered = [calls](const fs::RegionalDemandFilter&)
        -> FailureOr<std::unique_ptr<fs::PreparedLogicalPlan>> { ++*calls; return failure(); };
    domain.capabilities.exactQueries = domain.capabilities.endpointRecipes = true;
    domain.numerical = std::make_shared<fs::RegionalNumericalInterface>();
    domain.cost.numericalMerges = 7;
    std::string error;
    auto captured = fs::captureRegionalOrderContext(input, domain, error);
    if (failed(captured)) { return false; }
    const auto& clean = (*captured)->domain();
    if (clean.reachability || clean.numerical || clean.prepare || clean.prepareWithVisits || clean.prepareFiltered ||
        clean.capabilities.exactQueries || clean.capabilities.endpointRecipes || clean.cost.numericalMerges) {
        return false;
    }
    // Mutating the caller's metadata cannot change the captured occurrence frame.
    domain.outerDivisors[0][0] = 17;
    domain.presence = {};
    auto lower = fs::makeRegionalOrderView(*captured, queriesFor(*captured, false), error);
    auto upper = fs::makeRegionalOrderView(*captured, queriesFor(*captured, true), error);
    if (failed(lower) || failed(upper)) { return false; }
    auto arena = (*captured)->expressions();
    const auto predicate = arena->input(function.getArgument(1));
    Event source{1, arena->constant(3), fs::PeriodicEventKind::Completion, {arena->constant(1)}};
    Event target{0, source.ordinal, fs::PeriodicEventKind::Start, source.visits};
    for (const auto* view : {&*lower, &*upper}) {
        const auto& region = view->regional();
        if (region.outerDivisors[0][0] != 2 || region.firstOrdinal != source.ordinal || region.numerical ||
            region.prepare || region.prepareWithVisits || region.prepareFiltered ||
            region.capabilities.endpointRecipes || !region.capabilities.exactQueries ||
            fs::regionalPresence(region, target) != predicate ||
            region.endpointEventGuard(target) != arena->boolean(true) ||
            fs::regionalReferenceBefore(region, source, target) != arena->boolean(true)) { return false; }
    }
    if (fs::regionalReachability(lower->regional(), source, target) != arena->boolean(false) ||
        fs::regionalReachability(upper->regional(), source, target) != predicate) { return false; }
    target.ordinal = arena->input(function.getArgument(0));
    if (fs::regionalReachability(upper->regional(), source, target)) { return false; }
    return *calls == 0;
}
bool checkBindings(func::FuncOp function, const pto::SyncInput& input)
{
    auto domain = domainFor(function, input);
    if (domain.anchors.size() != 2) { return false; }
    std::string error;
    auto context = fs::captureRegionalOrderContext(input, domain, error);
    if (failed(context)) { return false; }
    fs::BoundingRegionalResult missing;
    missing.context = *context;
    missing.guarantee = fs::InputOrderGuarantee::InputOrderEquivalent;
    missing.reduction = fs::ReductionQuality::Partial;
    const auto yes = domain.expressions->boolean(true);
    auto ownership = fs::createRequirementProvenance(*context, {{1}}, {{1, {3, 9}, yes}}, error);
    missing.provenance = ownership;
    if (!ownership || failed(fs::validateBoundingRegionalResult(missing, error)) ||
        missing.provenance != ownership || missing.lower || missing.upper) { return false; }
    auto foreign = fs::captureRegionalOrderContext(input, domain, error);
    if (failed(foreign) || *foreign == *context) { return false; }
    missing.provenance = fs::createRequirementProvenance(*foreign, {{1}}, {{1, {3, 9}, yes}}, error);
    if (!missing.provenance || succeeded(fs::validateBoundingRegionalResult(missing, error))) { return false; }
    missing.provenance = ownership;
    auto view = fs::makeRegionalOrderView(*foreign, queriesFor(*foreign, false), error);
    if (failed(view)) { return false; }
    missing.lower = *view;
    if (succeeded(fs::validateBoundingRegionalResult(missing, error))) { return false; }
    missing.lower.reset();
    missing.reduction = static_cast<fs::ReductionQuality>(255);
    if (succeeded(fs::validateBoundingRegionalResult(missing, error)) ||
        succeeded(fs::makeRegionalOrderView(*context, {}, error)) ||
        succeeded(fs::makeRegionalOrderView({}, queriesFor(*context, false), error))) { return false; }
    auto invalid = domain;
    invalid.accessModel = nullptr;
    if (succeeded(fs::captureRegionalOrderContext(input, invalid, error))) { return false; }
    invalid = domain;
    invalid.gmAliasPolicy = domain.gmAliasPolicy == pto::GMAliasPolicy::MayAlias ?
        pto::GMAliasPolicy::MayNotAlias : pto::GMAliasPolicy::MayAlias;
    if (succeeded(fs::captureRegionalOrderContext(input, invalid, error))) { return false; }
    invalid = domain;
    invalid.outerDivisors[0][0] = 0;
    if (succeeded(fs::captureRegionalOrderContext(input, invalid, error))) { return false; }
    invalid = domain;
    invalid.firstOrdinal = domain.expressions->boolean(true);
    if (succeeded(fs::captureRegionalOrderContext(input, invalid, error))) { return false; }
    // A successful retry must not retain a failed attempt's diagnostic.
    auto retried = fs::captureRegionalOrderContext(input, domain, error);
    if (failed(retried) || !error.empty()) { return false; }
    error = "old query error";
    auto query = fs::makeRegionalOrderView(*context, queriesFor(*context, false), error);
    if (failed(query) || !error.empty()) { return false; }
    missing.reduction = fs::ReductionQuality::Generators;
    error = "old validation error";
    return succeeded(fs::validateBoundingRegionalResult(missing, error)) && error.empty();
}
bool checkFailedArena(const pto::SyncInput& input)
{
    fs::RegionalAnalysis domain;
    domain.expressions = std::make_shared<fs::RegionExpressions>();
    domain.accessModel = &input.accesses();
    domain.gmAliasPolicy = input.memory().gmPolicy();
    std::string error;
    auto context = fs::captureRegionalOrderContext(input, domain, error);
    if (failed(context)) { return false; }
    auto arena = domain.expressions;
    auto zero = arena->constant(0);
    (void)arena->div(zero, zero);
    if (arena->constructionError().empty() ||
        succeeded(fs::captureRegionalOrderContext(input, domain, error))) { return false; }
    return failed(fs::makeRegionalOrderView(*context, queriesFor(*context, false), error));
}
bool checkNumerical(func::FuncOp function, const pto::SyncInput& input)
{
    auto domain = domainFor(function, input);
    std::string error;
    auto context = fs::captureRegionalOrderContext(input, domain, error);
    if (failed(context)) { return false; }
    auto numerical = std::make_shared<fs::RegionalNumericalInterface>();
    auto index = std::make_shared<fs::NumericalChainInterface>(
        fs::buildNumericalChainInterface({{0}}, [](uint32_t, uint32_t) { return std::optional<bool>(true); }));
    auto arena = domain.expressions;
    numerical->index = index;
    numerical->events = {{1, arena->constant(3), fs::PeriodicEventKind::Start, {arena->constant(1)}}};
    numerical->chainKeys = {{0, fs::PeriodicEventKind::Start}};
    numerical->query = [](Event, Event, fs::NumericalChainQueryCost&) { return std::optional<bool>(true); };
    numerical->thresholds = [](Event, bool reverse, fs::NumericalChainQueryCost&) {
        return std::optional<std::vector<uint32_t>>({{reverse ? 1U : 0U}});
    };
    auto queries = queriesFor(*context, false);
    queries.numerical = numerical;
    auto view = fs::makeRegionalOrderView(*context, queries, error);
    if (failed(view) || view->regional().numerical != numerical) { return false; }
    // Do not mutate an index retained by a published view. Each malformed
    // candidate owns a fresh copy, as real selected-order adapters must do.
    for (unsigned defect = 0; defect < 6; ++defect) {
        auto bad = std::make_shared<fs::NumericalChainInterface>(*index);
        switch (defect) {
            case 0: bad->forward.clear(); break;
            case 1: bad->reverse[0].clear(); break;
            case 2: bad->chain[0] = 1; break;
            case 3: bad->rank[0] = 1; break;
            case 4: bad->chains[0].push_back(0); break;
            default: bad->forward[0][0] = 2; break;
        }
        auto candidate = std::make_shared<fs::RegionalNumericalInterface>(*numerical);
        candidate->index = bad;
        queries.numerical = candidate;
        if (succeeded(fs::makeRegionalOrderView(*context, queries, error))) { return false; }
    }
    return true;
}
} // namespace
int runBoundingRegionalChecks(func::FuncOp function, const pto::SyncInput& input)
{
    if (function.getNumArguments() < 2 || !function.getArgument(0).getType().isIndex() ||
        !function.getArgument(1).getType().isInteger(1)) { return 1; }
    if (!checkViews(function, input)) { llvm::errs() << "bounding regional views failed\n"; return 1; }
    if (!checkBindings(function, input)) { llvm::errs() << "bounding regional context checks failed\n"; return 1; }
    if (!checkFailedArena(input)) { llvm::errs() << "bounding regional invalid arena check failed\n"; return 1; }
    if (!checkNumerical(function, input)) { llvm::errs() << "bounding numerical structure check failed\n"; return 1; }
    llvm::outs() << "bounding regional contract checks passed\n";
    return 0;
}
