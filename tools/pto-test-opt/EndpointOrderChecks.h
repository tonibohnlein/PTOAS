// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent closure and unfolded-order checks for early command families.
#ifndef PTO_TEST_ENDPOINTORDER_CHECKS_H
#define PTO_TEST_ENDPOINTORDER_CHECKS_H
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/EndpointOrder.h"
#include "PTO/Transforms/FrontierSynch/Shared/EndpointPieces.h"
#include "PTO/Transforms/FrontierSynch/Shared/FamilyInsertion.h"
#include "PTO/Transforms/FrontierSynch/Shared/FamilyExpressions.h"
bool runFamilyMaterializationChecks(mlir::MLIRContext* context);
bool runMixedFamilyChecks(mlir::MLIRContext* context);
bool runAllocationPreparationChecks();
namespace {
bool familyRequestedOutputChecks(mlir::scf::ForOp loop)
{
    using namespace mlir;
    namespace fs = mlir::pto::frontiersynch;
    SmallVector<SmallVector<fs::TemplateCoordinate>> points{{{loop, 0}}, {{loop, 2}}, {{loop, 4}}};
    SmallVector<int64_t> labels{7, 7, 7};
    auto plan = fs::buildFamilyExpressions(points, labels, 0, fs::FamilyExpressionOutputs::Member);
    if (!plan.error.empty() || !plan.recipe) { return false; }
    Block code;
    OpBuilder builder(loop.getContext());
    builder.setInsertionPointToEnd(&code);
    auto output = fs::emitFamilyExpressions(builder, loop.getLoc(), *plan.recipe);
    // A constant selected value needs neither coordinate arithmetic nor the
    // sparse-domain guard. Check construction, before any canonicalizer/DCE.
    if (!output.error.empty() || output.present || code.getOperations().size() != 1) { return false; }
    auto constant = output.member.getDefiningOp<arith::ConstantIndexOp>();
    if (!constant || constant.value() != 7) { return false; }
    auto modular = fs::emitFamilyExpressions(builder, loop.getLoc(), points, labels, 2,
                                             fs::FamilyExpressionOutputs::Member);
    auto residue = modular.member.getDefiningOp<arith::ConstantIndexOp>();
    if (!modular.error.empty() || modular.present || !residue || residue.value() != 1 ||
        code.getOperations().size() != 2) { return false; }
    points.push_back(points.front()); labels.push_back(7);
    auto invalid = fs::emitFamilyExpressions(builder, loop.getLoc(), points, labels, 0,
                                             fs::FamilyExpressionOutputs::Member);
    if (invalid.error.empty() || code.getOperations().size() != 2) { return false; }
    points.pop_back(); labels.pop_back();
    auto command = fs::emitFamilyExpressions(builder, loop.getLoc(), points, labels);
    return command.error.empty() && command.present && command.member && code.getOperations().size() > 1;
}
bool contextualEndpointRecipeChecks(const mlir::pto::frontiersynch::NumericTemplateEndpoints& input)
{
    using namespace mlir;
    namespace fs = mlir::pto::frontiersynch;
    auto arena = std::make_shared<fs::RegionExpressions>();
    fs::ContextualEndpointFamily family;
    family.piece.cut = input.groups.front().cut;
    family.piece.sourcePipe = 1; family.piece.targetPipe = 2;
    family.piece.namespaceId = 7;
    family.piece.members.push_back({7, {}});
    family.activity = arena->boolean(true);
    family.ordinal = arena->constant(13);
    family.residueCoordinate = arena->constant(5);
    family.modulus = 3; family.residue = 2;
    if (succeeded(fs::buildContextualFamilyEndpointRecipe(arena, {family}, {{family.ordinal, nullptr}}))) {
        return false;
    }
    fs::ContextualFamilyExpressions planning(arena);
    auto first = planning.prepare(family.ordinal, family.piece.cut.before);
    auto again = planning.prepare(family.ordinal, family.piece.cut.before);
    if (failed(first) || failed(again) || *first != *again) { return false; }
    auto leaf = fs::buildContextualFamilyEndpointRecipe(
        planning, {family}, {{family.ordinal, family.piece.cut.before}});
    if (failed(leaf)) { return false; }
    // Reusing the availability cache without the first build's shared roots
    // must produce a self-contained recipe, not inherit a missing binding.
    auto independent = fs::buildContextualFamilyEndpointRecipe(planning, {family});
    if (failed(independent)) { return false; }
    fs::PreparedLogicalPlan retry(0);
    retry.familyRecipe = *independent;
    if (failed(fs::materializeLogicalFamilies(retry)) || retry.endpoints.size() != 1) { return false; }
    fs::PreparedLogicalPlan plan(0);
    plan.familyRecipe = *leaf;
    if (!plan.endpoints.empty() || !plan.preparation.empty() ||
        failed(fs::materializeLogicalFamilies(plan)) || plan.endpoints.size() != 1) { return false; }
    const auto& endpoint = plan.endpoints.front();
    auto ordinal = endpoint.identity.getDefiningOp<arith::ConstantIndexOp>();
    auto member = endpoint.memberCoordinates.front().getDefiningOp<arith::ConstantIndexOp>();
    if (!ordinal || ordinal.value() != 13 || !member || member.value() != 7 || endpoint.record != 7 ||
        endpoint.domain.size() != 1 || endpoint.domain.front().size() != 1 ||
        endpoint.domain.front().front().modulus != 3 || endpoint.domain.front().front().value != 2) { return false; }
    auto invalid = family;
    invalid.residue = 3;
    if (succeeded(fs::buildContextualFamilyEndpointRecipe(arena, {invalid}))) { return false; }
    invalid = family; invalid.activity = family.ordinal;
    if (succeeded(fs::buildContextualFamilyEndpointRecipe(arena, {invalid}))) { return false; }
    invalid = family; invalid.piece.members.push_back({8, {}});
    if (succeeded(fs::buildContextualFamilyEndpointRecipe(arena, {invalid}))) { return false; }
    // Replayability revoked after planning must fail atomically during final
    // lowering, preserving the still-owned recipe and original payload IR.
    auto* cut = family.piece.cut.before;
    family.ordinal = arena->input(cut->getResult(0));
    auto replay = fs::buildContextualFamilyEndpointRecipe(arena, {family});
    if (failed(replay)) { return false; }
    arena->forbidRecomputation(cut);
    fs::PreparedLogicalPlan rejected(0);
    rejected.familyRecipe = *replay;
    return failed(fs::materializeLogicalFamilies(rejected)) && rejected.preparation.empty() &&
        rejected.endpoints.empty() && bool(rejected.familyRecipe) &&
        failed(fs::buildContextualFamilyEndpointRecipe(arena, {family}));
}
bool endpointOrderGraphChecks()
{
    namespace fs = mlir::pto::frontiersynch;
    // Exhaust every four-node directed graph, including self edges. Reachable
    // diagonals independently identify cycle membership, not blocked descendants.
    for (unsigned mask = 0; mask < 65536; ++mask) {
        bool reaches[4][4] = {};
        std::vector<std::set<std::size_t>> edges(4);
        for (unsigned a = 0; a < 4; ++a) {
            for (unsigned b = 0; b < 4; ++b) {
                if (mask & (1U << (4 * a + b))) { edges[a].insert(b); reaches[a][b] = true; }
            }
        }
        for (unsigned k = 0; k < 4; ++k) {
            for (unsigned a = 0; a < 4; ++a) {
                for (unsigned b = 0; b < 4; ++b) { reaches[a][b] |= reaches[a][k] && reaches[k][b]; }
            }
        }
        auto result = fs::orderEndpointGroups(edges);
        for (unsigned a = 0; a < 4; ++a) {
            if (bool(result.cyclic.count(a)) != reaches[a][a]) { return false; }
        }
        if (result.cyclic.empty()) {
            if (result.positions.size() != 4) { return false; }
            std::vector<std::size_t> rank(4);
            for (unsigned i = 0; i < 4; ++i) { rank[result.positions[i]] = i; }
            for (unsigned a = 0; a < 4; ++a) {
                for (auto b : edges[a]) { if (rank[a] >= rank[b]) { return false; } }
            }
        }
    }
    // A long descendant chain exercises iterative traversal and must remain
    // intact even though topological elimination cannot remove any vertex.
    std::vector<std::set<std::size_t>> chain(8192);
    for (std::size_t i = 1; i < chain.size(); ++i) { chain[i - 1].insert(i); }
    chain[1].insert(0);
    auto result = fs::orderEndpointGroups(chain);
    return result.cyclic == std::set<std::size_t>{0, 1};
}
bool endpointInvocationRecipeChecks(const mlir::pto::frontiersynch::NumericTemplateEndpoints& input,
                                    std::shared_ptr<const mlir::pto::frontiersynch::FamilyEndpointRecipe> leaf,
                                    const std::vector<mlir::pto::frontiersynch::EndpointFamily>& families)
{
    using namespace mlir;
    namespace fs = mlir::pto::frontiersynch;
    auto arena = std::make_shared<fs::RegionExpressions>();
    auto inner = fs::prefixFamilyInvocation(leaf, arena, arena->constant(13));
    if (failed(inner)) { return false; }
    auto outer = fs::prefixFamilyInvocation(*inner, arena, arena->constant(29));
    if (failed(outer)) { return false; }
    fs::PreparedLogicalPlan plan(0);
    plan.familyRecipe = *outer;
    plan.families = families;
    auto shapes = fs::familyMatchingArities(*plan.familyRecipe);
    if (failed(shapes)) { return false; }
    auto view = fs::takeRegionalAllocationView(plan, true);
    if (!view || !plan.preparation.empty() || !plan.endpoints.empty() ||
        failed(fs::materializeLogicalFamilies(plan))) { return false; }
    for (const auto& endpoint : plan.endpoints) {
        if (endpoint.kind == fs::LogicalCommandKind::Barrier) {
            if (!endpoint.memberCoordinates.empty()) { return false; }
            continue;
        }
        if (endpoint.memberCoordinates.size() != 3) { return false; }
        for (auto record : endpoint.records) {
            if (shapes->at(record) != endpoint.memberCoordinates.size()) { return false; }
        }
        for (auto [index, expected] : {std::pair<unsigned, int64_t>{1, 29}, {2, 13}}) {
            auto constant = endpoint.memberCoordinates[index].getDefiningOp<arith::ConstantIndexOp>();
            if (!constant || constant.value() != expected) { return false; }
        }
    }
    auto* cut = input.groups.front().cut.before;
    auto unavailable = std::make_shared<fs::RegionExpressions>();
    unavailable->forbidRecomputation(cut);
    auto future = unavailable->input(cut->getResult(0));
    if (succeeded(fs::prefixFamilyInvocation(leaf, unavailable, future)) ||
        succeeded(fs::prefixFamilyInvocation(leaf, arena, arena->boolean(true)))) { return false; }
    // Invalidate an originally replayable input after planning. Lowering must
    // discard its temporary arithmetic and preserve the incumbent plan.
    auto late = std::make_shared<fs::RegionExpressions>();
    auto replay = fs::prefixFamilyInvocation(leaf, late, late->input(cut->getResult(0)));
    if (failed(replay)) { return false; }
    late->forbidRecomputation(cut);
    fs::PreparedLogicalPlan rejected(0);
    rejected.familyRecipe = *replay;
    rejected.families = families;
    if (succeeded(fs::materializeLogicalFamilies(rejected)) || !rejected.preparation.empty() ||
        !rejected.endpoints.empty() || !rejected.familyRecipe) { return false; }
    return true;
}
// Compare delayed sequence assembly with an independently materialized leaf.
// Concrete commands between children must keep their exact inventory position.
bool endpointRecipeCompositionChecks(const mlir::pto::frontiersynch::NumericTemplateEndpoints& input,
                                     std::shared_ptr<const mlir::pto::frontiersynch::FamilyEndpointRecipe> leaf,
                                     const std::vector<mlir::pto::frontiersynch::EndpointFamily>& families)
{
    namespace fs = mlir::pto::frontiersynch;
    fs::PreparedLogicalPlan reference(0), composed(0);
    reference.familyRecipe = leaf;
    if (failed(fs::materializeLogicalFamilies(reference))) { return false; }
    auto first = fs::appendFamilyEndpointRecipe({}, leaf, 10, 1);
    if (failed(first)) { return false; }
    auto child = fs::appendFamilyEndpointRecipe(*first, leaf, 100, 2);
    auto prefix = fs::appendFamilyEndpointRecipe({}, leaf, 3000, 0);
    if (failed(child) || failed(prefix)) { return false; }
    auto root = fs::appendFamilyEndpointRecipe(*prefix, *child, 1000, 1);
    if (failed(root)) { return false; }
    composed.familyRecipe = *root;
    for (uint32_t base : {3000U, 1010U, 1100U}) {
        for (auto family : families) {
            family.id += base;
            for (auto& member : family.members) { member.record += base; }
            composed.families.push_back(std::move(family));
        }
    }
    bool allocationSawRecipe = false;
    composed.allocationPreparation = [&](fs::PreparedLogicalPlan& view) {
        allocationSawRecipe = view.familyRecipe && view.endpoints.empty() && view.preparation.empty();
    };
    auto view = fs::takeRegionalAllocationView(composed, true);
    if (!view) { return false; }
    fs::prepareAllocationSupport(*view);
    if (!allocationSawRecipe || !composed.preparation.empty()) { return false; }
    auto cut = input.groups.front().cut.before;
    for (uint32_t record : {900000U, 900001U, 900002U}) {
        fs::PreparedLogicalEndpoint endpoint{cut, fs::LogicalCommandKind::Barrier, 1, 1, record, {}, {}};
        endpoint.records.push_back(record);
        composed.endpoints.push_back(std::move(endpoint));
    }
    // Invalid placement and namespace overflow fail before any scalar lowering.
    auto bad = fs::appendFamilyEndpointRecipe({}, leaf, 0, 4);
    fs::PreparedLogicalPlan invalid(0);
    if (failed(bad)) { return false; }
    invalid.familyRecipe = *bad;
    if (succeeded(fs::materializeLogicalFamilies(invalid)) || !invalid.preparation.empty() ||
        !invalid.endpoints.empty() ||
        succeeded(fs::appendFamilyEndpointRecipe({}, leaf, UINT32_MAX, 0)) ||
        succeeded(fs::appendFamilyEndpointRecipe({}, *child, 0, SIZE_MAX))) { return false; }
    auto reversed = fs::appendFamilyEndpointRecipe(*child, leaf, 200, 0);
    if (failed(reversed)) { return false; }
    invalid.familyRecipe = *reversed;
    invalid.endpoints.resize(2);
    if (succeeded(fs::materializeLogicalFamilies(invalid)) || !invalid.preparation.empty() ||
        invalid.endpoints.size() != 2) { return false; }
    invalid.familyRecipe = *root;
    invalid.endpoints.resize(3);
    if (succeeded(fs::materializeLogicalFamilies(invalid)) || !invalid.preparation.empty() ||
        invalid.endpoints.size() != 3) { return false; }
    if (failed(fs::materializeLogicalFamilies(composed)) || composed.familyRecipe ||
        composed.endpoints.size() != reference.endpoints.size() * 3 + 3) { return false; }
    std::size_t position = 0;
    auto checkLeaf = [&](uint32_t base) {
        for (const auto& original : reference.endpoints) {
            const auto& actual = composed.endpoints[position++];
            auto records = original.records;
            for (auto& record : records) { record += base; }
            if (actual.before != original.before || actual.kind != original.kind || actual.records != records ||
                actual.record != original.record + 1010 || actual.piece != int64_t(position - 1)) {
                return false;
            }
        }
        return true;
    };
    if (!checkLeaf(3000) || composed.endpoints[position++].record != 900000 ||
        composed.endpoints[position++].record != 900001 || !checkLeaf(1010) ||
        composed.endpoints[position++].record != 900002 || !checkLeaf(1100)) { return false; }
    return position == composed.endpoints.size();
}
// Masks must share original-cut values without referencing a later leaf's
// detached guard from an earlier shared block. Check dependency order directly.
bool endpointActivityRecipeChecks(const mlir::pto::frontiersynch::NumericTemplateEndpoints& input,
                                 std::shared_ptr<const mlir::pto::frontiersynch::FamilyEndpointRecipe> leaf,
                                 const std::vector<mlir::pto::frontiersynch::EndpointFamily>& families)
{
    namespace fs = mlir::pto::frontiersynch;
    auto arena = std::make_shared<fs::RegionExpressions>();
    auto masked = fs::maskFamilyActivity(leaf, arena, arena->boolean(false));
    if (failed(masked) || succeeded(fs::maskFamilyActivity(leaf, arena, arena->constant(7)))) { return false; }
    auto prefixed = fs::prefixFamilyInvocation(*masked, arena, arena->constant(7));
    if (failed(prefixed)) { return false; }
    auto joined = fs::appendFamilyEndpointRecipe(*prefixed, *masked, 1000, 0);
    if (failed(joined)) { return false; }
    fs::PreparedLogicalPlan plan(0);
    plan.familyRecipe = *joined;
    for (uint32_t base : {0U, 1000U}) {
        for (auto family : families) {
            family.id += base;
            for (auto& member : family.members) { member.record += base; }
            plan.families.push_back(std::move(family));
        }
    }
    auto shapes = fs::familyMatchingArities(*plan.familyRecipe);
    if (failed(shapes) || !plan.preparation.empty() || !plan.endpoints.empty()) { return false; }
    // Same record under different invocation depths is an invalid matching
    // shape; do not pick one endpoint side's arity and silently accept it.
    auto conflict = fs::appendFamilyEndpointRecipe(*prefixed, *masked, 0, 0);
    if (failed(conflict) || succeeded(fs::familyMatchingArities(**conflict))) { return false; }
    if (failed(fs::materializeLogicalFamilies(plan))) { return false; }
    llvm::DenseSet<mlir::Value> available;
    for (const auto& stage : plan.preparation) {
        for (auto& op : *stage.code) {
            for (auto value : op.getOperands()) {
                auto* definition = value.getDefiningOp();
                if (definition && !definition->getParentOp() && !available.contains(value)) { return false; }
            }
            for (auto value : op.getResults()) { available.insert(value); }
        }
    }
    for (const auto& endpoint : plan.endpoints) {
        auto selected = endpoint.guard.getDefiningOp<mlir::arith::SelectOp>();
        if (!selected) { return false; }
        auto condition = selected.getCondition().getDefiningOp<mlir::arith::ConstantOp>();
        if (!condition || !llvm::cast<mlir::IntegerAttr>(condition.getValue()).getValue().isZero()) { return false; }
        if (endpoint.memberCoordinates.size() != (endpoint.records.front() < 1000 ? 2U : 1U)) { return false; }
        for (auto record : endpoint.records) {
            if (shapes->at(record) != endpoint.memberCoordinates.size()) { return false; }
        }
    }
    auto* cut = input.groups.front().cut.before;
    auto late = std::make_shared<fs::RegionExpressions>();
    auto guard = late->eq(late->input(cut->getResult(0)), late->constant(0));
    auto replay = fs::maskFamilyActivity(leaf, late, guard);
    if (failed(replay)) { return false; }
    late->forbidRecomputation(cut);
    fs::PreparedLogicalPlan rejected(0);
    rejected.familyRecipe = *replay;
    rejected.families = families;
    return failed(fs::materializeLogicalFamilies(rejected)) && rejected.preparation.empty() &&
        rejected.endpoints.empty() && bool(rejected.familyRecipe);
}
bool endpointActivityBarrierChecks(mlir::pto::frontiersynch::NumericTemplateEndpoints input)
{
    namespace fs = mlir::pto::frontiersynch;
    input.outer = {};
    const auto pipe = static_cast<uint32_t>(input.anchors[0].phase->kPipeValue);
    input.logical.recipes = {{0, 0, 0, pipe, 0, fs::EndpointKind::Barrier}};
    auto leaf = fs::buildFiniteFamilyEndpointRecipe(input);
    if (failed(leaf)) { return false; }
    auto arena = std::make_shared<fs::RegionExpressions>();
    auto masked = fs::maskFamilyActivity(*leaf, arena, arena->boolean(false));
    if (failed(masked)) { return false; }
    auto outer = fs::prefixFamilyInvocation(*masked, arena, arena->constant(9));
    if (failed(outer)) { return false; }
    fs::PreparedLogicalPlan plan(0);
    plan.familyRecipe = *outer;
    fs::EndpointFamily family;
    family.sourcePipe = pipe; family.targetPipe = pipe; family.local = true;
    family.members.push_back({0, 0, 0, {}, {}});
    plan.families.push_back(std::move(family));
    if (failed(fs::materializeLogicalFamilies(plan)) || plan.endpoints.size() != 1) { return false; }
    const auto& endpoint = plan.endpoints.front();
    auto selected = endpoint.guard.getDefiningOp<mlir::arith::SelectOp>();
    if (!selected) { return false; }
    auto condition = selected.getCondition().getDefiningOp<mlir::arith::ConstantOp>();
    return condition && llvm::cast<mlir::IntegerAttr>(condition.getValue()).getValue().isZero() &&
        !endpoint.identity && endpoint.memberCoordinates.empty();
}
// Finite families share the ordering/selector compiler, but have no runtime
// outer ordinal. Include holes, simultaneous points and arbitrary record labels.
bool finiteEndpointRecipeChecks(mlir::pto::frontiersynch::NumericTemplateEndpoints input)
{
    namespace fs = mlir::pto::frontiersynch;
    if (succeeded(fs::buildFiniteFamilyEndpointRecipe(input))) { return false; }
    input.outer = {};
    if (succeeded(fs::buildFiniteFamilyEndpointRecipe(input))) { return false; }
    for (auto& recipe : input.logical.recipes) { recipe.displacement = 0; }
    for (auto& anchor : input.anchors) {
        for (auto& coordinate : anchor.coordinates) { coordinate.induction = 2 * coordinate.induction + 1; }
    }
    auto recipe = fs::buildFiniteFamilyEndpointRecipe(input);
    auto pieces = fs::buildEndpointPieces(input);
    if (failed(recipe) || !pieces.error.empty()) { return false; }
    fs::PreparedLogicalPlan plan(0);
    plan.familyRecipe = *recipe;
    if (!plan.preparation.empty() || !plan.endpoints.empty() ||
        failed(fs::materializeLogicalFamilies(plan)) || plan.endpoints.size() != pieces.pieces.size()) { return false; }
    for (const auto& endpoint : plan.endpoints) {
        auto ordinal = endpoint.identity.getDefiningOp<mlir::arith::ConstantOp>();
        if (!ordinal || !llvm::cast<mlir::IntegerAttr>(ordinal.getValue()).getValue().isZero()) { return false; }
        auto found = llvm::find_if(pieces.pieces, [&](const auto& piece) {
            return piece.kind == (endpoint.kind == fs::LogicalCommandKind::Set ?
                fs::EndpointKind::Set : fs::EndpointKind::Wait) &&
                piece.members.size() == endpoint.records.size() &&
                llvm::all_of(llvm::zip(piece.members, endpoint.records), [](auto pair) {
                    return std::get<0>(pair).record == std::get<1>(pair);
                });
        });
        if (found == pieces.pieces.end()) { return false; }
    }
    auto invalid = input;
    invalid.groups.clear();
    if (succeeded(fs::buildFiniteFamilyEndpointRecipe(invalid))) { return false; }
    invalid = input;
    invalid.logical.recipes.clear();
    return failed(fs::buildFiniteFamilyEndpointRecipe(invalid));
}
bool endpointOrderFamilyChecks(mlir::MLIRContext* context)
{
    using namespace mlir;
    namespace fs = mlir::pto::frontiersynch;
    context->getOrLoadDialect<scf::SCFDialect>();
    OpBuilder builder(context);
    auto loc = builder.getUnknownLoc();
    OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
    builder.setInsertionPointToStart(module->getBody());
    auto function = builder.create<func::FuncOp>(loc, "endpoint_order", builder.getFunctionType({}, {}));
    builder.setInsertionPointToStart(function.addEntryBlock());
    auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    auto one = builder.create<arith::ConstantIndexOp>(loc, 1);
    auto end = builder.create<arith::ConstantIndexOp>(loc, 16);
    auto loop = builder.create<scf::ForOp>(loc, zero, end, one);
    builder.create<func::ReturnOp>(loc);
    if (!familyRequestedOutputChecks(loop) || !runFamilyMaterializationChecks(context) ||
        !runMixedFamilyChecks(context) || !runAllocationPreparationChecks()) { return false; }
    builder.setInsertionPoint(loop.getBody()->getTerminator());
    auto cut = builder.create<arith::AddIOp>(loc, zero, one).getOperation();
    pto::CompoundInstanceElement source(0, {}, {}, pto::PipelineType::PIPE_MTE2, cut->getName());
    pto::CompoundInstanceElement target(1, {}, {}, pto::PipelineType::PIPE_MTE1, cut->getName());
    // Different displacements give three family namespaces. At visit zero:
    // A then B then C; at visit one: B then A then C. C is never in the cycle.
    for (unsigned count : {2U, 4U, 16U}) {
        for (bool rename : {false, true}) {
            fs::NumericTemplateEndpoints input;
            input.outer = loop;
            fs::TemplateEndpointCut position{cut->getBlock(), cut};
            input.groups.push_back({position, {}});
            auto append = [&](unsigned index, unsigned coordinate, unsigned displacement) {
                uint32_t record = rename ? 100 + (count + 4 - index) * 7 : index;
                uint32_t type = input.anchors.size();
                input.anchors.push_back({&source, {{loop, coordinate}}, position, position});
                input.anchors.push_back({&target, {{loop, coordinate}}, position, position});
                for (auto kind : {fs::EndpointKind::Set, fs::EndpointKind::Wait}) {
                    auto pipe = kind == fs::EndpointKind::Set ? 4U : 3U;
                    input.groups.front().recipes.push_back(input.logical.recipes.size());
                    input.logical.recipes.push_back({record, type, type + 1, pipe, displacement, kind});
                }
            };
            append(0, 0, 0); append(1, 0, 1); append(2, 1, 1); append(3, 1, 0);
            for (unsigned i = 0; i < count; ++i) { append(4 + i, i, 2); }
            auto families = fs::buildEndpointFamilies(input);
            auto pieces = fs::buildEndpointPieces(input);
            auto recipe = fs::buildFamilyEndpointRecipe(input);
            if (!families.error.empty() || !pieces.error.empty() || failed(recipe) ||
                families.families.size() != 5 || pieces.pieces.size() != 10 ||
                fs::familyEndpointPieceCount(**recipe) != 10 ||
                !endpointRecipeCompositionChecks(input, *recipe, families.families) ||
                !endpointInvocationRecipeChecks(input, *recipe, families.families) ||
                !finiteEndpointRecipeChecks(input) ||
                !endpointActivityRecipeChecks(input, *recipe, families.families) ||
                !contextualEndpointRecipeChecks(input) ||
                !endpointActivityBarrierChecks(input)) { return false; }
            auto downstream = llvm::find_if(families.families, [](const auto& f) { return f.displacement == 2; });
            if (downstream == families.families.end() || downstream->members.size() != count) { return false; }
            llvm::sort(pieces.pieces, [](const auto& a, const auto& b) { return a.order < b.order; });
            for (unsigned coordinate = 0; coordinate < count; ++coordinate) {
                for (uint64_t trips : {0U, 1U, 2U, 5U}) {
                    for (uint64_t ordinal = 0; ordinal < trips; ++ordinal) {
                        std::vector<std::pair<fs::EndpointKind, uint32_t>> expected, actual;
                        for (auto kind : {fs::EndpointKind::Set, fs::EndpointKind::Wait}) {
                            for (uint32_t i = 0; i < input.logical.recipes.size(); ++i) {
                                const auto& r = input.logical.recipes[i];
                                if (r.kind == kind && input.anchors[r.source].coordinates[0].induction == coordinate &&
                                    input.logical.evaluate(i, trips, ordinal).active) {
                                    expected.emplace_back(kind, r.record);
                                }
                            }
                        }
                        for (const auto& piece : pieces.pieces) {
                            const bool active = piece.kind == fs::EndpointKind::Set ?
                                piece.displacement < trips && ordinal < trips - piece.displacement :
                                ordinal >= piece.displacement;
                            if (!active) { continue; }
                            for (const auto& member : piece.members) {
                                if (member.coordinates[0].induction == coordinate) {
                                    actual.emplace_back(piece.kind, member.record);
                                }
                            }
                        }
                        if (actual != expected) { return false; }
                    }
                }
            }
        }
    }
    return succeeded(verify(*module));
}
} // namespace
#endif
