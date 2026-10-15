// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_RELATIONAL_PREFIX_CHECKS_H
#define PTO_RELATIONAL_PREFIX_CHECKS_H
// Included within the independent prefix trace harness's anonymous namespace.
bool relationalPrefixFailure(unsigned line)
{
    llvm::errs() << "relational prefix check failed at line " << line << "\n";
    return false;
}
bool flatPrefixBatches(MLIRContext& context)
{
    Fixture f(context, true);
    Operation* source = nullptr;
    Operation* consumer = nullptr;
    f.function.walk([&](func::CallOp call) {
        auto role = call->getAttrOfType<IntegerAttr>("test.role").getInt();
        if (role == 0 && !source) {
            source = call;
        }
        if (role == 2 && !consumer) {
            consumer = call;
        }
    });
    pto::CompoundInstanceElement phase(
        0, SmallVector<const pto::BaseMemInfo*>{}, SmallVector<const pto::BaseMemInfo*>{},
        static_cast<pto::PipelineType>(P), source->getName());
    phase.elementOp = source;
    fs::RegionExpressions arena;
    auto value = arena.input(f.function.getArgument(0));
    for (unsigned i = 0; i < 512; ++i) {
        value = arena.div(arena.add(value, arena.constant(i + 1)), arena.constant(3));
    }
    SmallVector<fs::RegionExpressions::Id> roots;
    for (unsigned i = 0; i < 1024; ++i) {
        roots.push_back(arena.eq(value, arena.constant(i)));
    }
    uint64_t unionNodes = 0, singletonNodes = 0;
    auto inputs = arena.referencedInputs(roots, &unionNodes);
    for (auto root : roots) {
        uint64_t nodes = 0;
        arena.referencedInputs(ArrayRef<fs::RegionExpressions::Id>(root), &nodes);
        singletonNodes += nodes;
    }
    if (inputs.size() != 1 || unionNodes > arena.size() || unionNodes * 100 >= singletonNodes) {
        return relationalPrefixFailure(__LINE__);
    }
    fs::InvocationPrefixRecipe saved;
    unsigned visited = 0;
    {
        fs::InvocationPrefixCollection collection(f.function);
        std::string error;
        for (unsigned i = 0; i < 16384; ++i) {
            if (failed(collection.source(
                    &phase,
                    [&visited, i](OpBuilder&, const IRMapping&, MutableArrayRef<Value>, Value, std::string&) {
                        if (visited != i) {
                            return failure();
                        }
                        ++visited;
                        return success();
                    },
                    error))) {
                return relationalPrefixFailure(__LINE__);
            }
            if (failed(collection.consumer(
                    consumer, {P, Q},
                    [](OpBuilder& b, const IRMapping&, ArrayRef<Value>, std::string&) -> FailureOr<Value> {
                        return b.create<arith::ConstantIndexOp>(b.getUnknownLoc(), 1).getResult();
                    },
                    error))) {
                return relationalPrefixFailure(__LINE__);
            }
        }
        saved = collection.recipe();
    }
    Block scratch;
    OpBuilder b(&context);
    b.setInsertionPointToEnd(&scratch);
    IRMapping mapping;
    std::string error;
    SmallVector<Value> state;
    auto rank = b.create<arith::ConstantIndexOp>(b.getUnknownLoc(), 1);
    if (failed(saved.sources.front().snapshot(b, mapping, state, rank, error)) || visited != 16384) {
        return relationalPrefixFailure(__LINE__);
    }
    auto incoming = saved.consumers.front().incoming(b, mapping, state, error);
    return succeeded(incoming) && saved.sources.size() == 1 && saved.consumers.size() == 1;
}
// Build F* by independently unfolding five optional payloads and testing event
// closure. Only the supplied exact finite relation is consumed by the adapter.
bool relationalPrefixCases(MLIRContext& context)
{
    OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(&context)));
    OpBuilder b(&context);
    auto loc = b.getUnknownLoc();
    b.setInsertionPointToEnd(module->getBody());
    auto payload = b.create<func::FuncOp>(loc, "payload", b.getFunctionType({b.getIndexType()}, {}));
    payload.setPrivate();
    SmallVector<Type> types(13, b.getI1Type());
    types[0] = b.getIndexType();
    auto function = b.create<func::FuncOp>(loc, "relational", b.getFunctionType(types, {}));
    b.setInsertionPointToStart(function.addEntryBlock());
    const std::array<unsigned, 5> roles{0, 0, 1, 2, 3};
    std::vector<pto::CompoundInstanceElement> phases;
    phases.reserve(roles.size());
    auto input = std::make_shared<pto::SyncInput>();
    auto arena = std::make_shared<fs::RegionExpressions>();
    fs::RegionalAnalysis region;
    region.expressions = arena;
    region.accessModel = &input->accesses();
    region.gmAliasPolicy = input->memory().gmPolicy();
    auto relation = std::make_shared<fs::RegionalRelations>();
    region.relations = relation;
    auto& data = relation->data;
    data.input = input.get();
    data.context = {function, function};
    data.completeRequiredOrder = true;
    data.analysis.parameterCount = roles.size();
    data.analysis.pipeCount = 16;
    for (unsigned i = 0; i < roles.size(); ++i) {
        auto flag = function.getArgument(i + 1);
        data.parameters.push_back(arena->input(flag));
        data.parameterValues.push_back(flag);
        auto branch = b.create<scf::IfOp>(loc, flag, false);
        Operation* op;
        {
            OpBuilder::InsertionGuard restore(b);
            b.setInsertionPointToStart(branch.thenBlock());
            auto tag = b.create<arith::ConstantIndexOp>(loc, roles[i]);
            op = b.create<func::CallOp>(loc, "payload", TypeRange{}, ValueRange{tag});
        }
        phases.emplace_back(
            i, SmallVector<const pto::BaseMemInfo*>{}, SmallVector<const pto::BaseMemInfo*>{},
            static_cast<pto::PipelineType>(pipe(roles[i])), op->getName());
        phases.back().elementOp = op;
        fs::TemplateEndpointAnchor anchor{
            &phases.back(), {}, {op->getBlock(), op}, {op->getBlock(), op->getNextNode()}};
        region.anchors.push_back(anchor);
        region.occurrenceLoops.push_back({});
        data.sites.push_back({&phases.back(), {}, {}, {}});
        fs::RegionalAnalysis child;
        child.expressions = arena;
        child.accessModel = region.accessModel;
        child.gmAliasPolicy = region.gmAliasPolicy;
        child.anchors.push_back(anchor);
        child.occurrenceLoops.push_back({});
        auto contribution = std::make_shared<fs::RegionalPrefixContribution>();
        contribution->input = input;
        contribution->expressions = arena;
        contribution->function = function;
        contribution->accessModel = child.accessModel;
        contribution->aliasPolicy = child.gmAliasPolicy;
        contribution->matchesOccurrenceFrame = [anchor, arena](const fs::RegionalAnalysis& current) {
            return current.expressions == arena && current.anchors.size() == 1 &&
                   current.anchors[0].phase == anchor.phase;
        };
        contribution->collect = [phase = &phases.back()](
                                    fs::InvocationPrefixCollection& collection, std::string& error) {
            return collection.source(phase, {}, error);
        };
        child.invocationPrefix = contribution;
        relation->children.push_back(std::move(child));
    }
    b.create<func::ReturnOp>(loc);
    region.presence =
        [arena, parameters = data.parameters](fs::RegionalEvent event) -> std::optional<fs::RegionExpressions::Id> {
        return arena->land(arena->eq(event.ordinal, arena->constant(0)), parameters[event.type]);
    };
    std::array<Graph, 32> expected;
    std::array<std::vector<unsigned>, 32> activeRoles;
    for (unsigned bits = 0; bits < 32; ++bits) {
        std::vector<unsigned> active;
        for (unsigned i = 0; i < roles.size(); ++i) {
            if ((bits >> i) & 1) {
                active.push_back(i);
                activeRoles[bits].push_back(roles[i]);
            }
        }
        auto graph = native(activeRoles[bits]);
        for (unsigned a = 0; a < active.size(); ++a) {
            for (unsigned c = a + 1; c < active.size(); ++c) {
                auto source = active[a], target = active[c];
                if ((target == 3 && source < 3) || (target == 4 && source == 0)) {
                    edge(graph, 2 * a + 1, 2 * c);
                }
            }
        }
        close(graph);
        expected[bits] = graph;
        std::vector<fs::IntegerConstraint> constraints;
        for (unsigned i = 0; i < roles.size(); ++i) {
            fs::IntegerConstraint upper, lower;
            upper.coefficients.resize(roles.size());
            lower.coefficients.resize(roles.size());
            upper.coefficients[i] = fs::BoundInteger(1);
            lower.coefficients[i] = fs::BoundInteger(-1);
            upper.bound = fs::BoundInteger((bits >> i) & 1);
            lower.bound = -upper.bound;
            constraints.push_back(upper);
            constraints.push_back(lower);
        }
        auto domain = fs::IntegerSystem::create(roles.size(), constraints);
        if (failed(domain)) {
            return relationalPrefixFailure(__LINE__);
        }
        for (unsigned a = 0; a < active.size(); ++a) {
            for (unsigned c = a + 1; c < active.size(); ++c) {
                const auto from = 2 * a + 1, to = 2 * c;
                if (!((graph[from] >> to) & 1) || pipe(roles[active[a]]) == pipe(roles[active[c]])) {
                    continue;
                }
                bool cover = true;
                for (unsigned mid = 0; mid < graph.size(); ++mid) {
                    if (mid != from && mid != to && ((graph[from] >> mid) & 1) && ((graph[mid] >> to) & 1)) {
                        cover = false;
                    }
                }
                if (cover) {
                    fs::ArithmeticRelationKey key{
                        {active[a], fs::ArithmeticEvent::Completion, {}},
                        {active[c], fs::ArithmeticEvent::Start, {}},
                        std::vector<uint64_t>(roles.size(), 0)};
                    relation->newCrossings[key].push_back(*domain);
                    relation->newCrossings[key].push_back(
                        *domain); // Overlapping pieces must not duplicate acquisitions.
                }
            }
        }
    }
    auto auxiliary = std::make_shared<Block>();
    std::weak_ptr<Block> auxiliaryLifetime = auxiliary;
    relation->auxiliaryOwners.push_back(auxiliary);
    region.invocationPrefix = fs::relationalPrefixContribution(region, input);
    relation->auxiliaryOwners.clear();
    auxiliary.reset();
    if (!region.invocationPrefix || auxiliaryLifetime.expired()) {
        return relationalPrefixFailure(__LINE__);
    }
    std::string diagnostic;
    auto invalid = region;
    invalid.relations = std::make_shared<fs::RegionalRelations>(*relation);
    fs::InvocationPrefixCollection stale(function, input);
    if (succeeded(fs::collectRegionalInvocationPrefix(invalid, stale, diagnostic))) {
        return relationalPrefixFailure(__LINE__);
    }
    fs::InvocationPrefixCollection collection(function, input);
    if (failed(fs::collectRegionalInvocationPrefix(region, collection, diagnostic))) {
        llvm::errs() << "relational prefix collect: " << diagnostic << "\n";
        return relationalPrefixFailure(__LINE__);
    }
    auto& spec = collection.recipe();
    spec.planId = 19;
    spec.privateScalarControl = true;
    spec.sourceBounds = {{P, 2}, {R, 1}, {Q, 2}};
    fs::PrivateStateError error;
    auto staged = fs::materializeInvocationPrefix(function, spec, error);
    if (failed(staged)) {
        llvm::errs() << "relational prefix materialize: " << error.message << "\n";
        return relationalPrefixFailure(__LINE__);
    }
    for (unsigned bits = 0; bits < 32; ++bits) {
        Execution execution;
        execution.run((*staged)->function, 0, bits);
        const auto publications = ((bits >> 0) & 1) + ((bits >> 1) & 1) + ((bits >> 2) & 1);
        if (!execution.valid || !verifyTrace(execution.events, activeRoles[bits], &expected[bits], publications)) {
            llvm::errs() << "relational prefix trace bits=" << bits << "\n";
            return relationalPrefixFailure(__LINE__);
        }
    }
    // Parameters that are original payload results must never be replayed at
    // earlier consumers. Failure leaves the original function untouched.
    b.setInsertionPoint(&function.front().back());
    auto opaque = b.create<func::FuncOp>(loc, "opaque", b.getFunctionType({}, {b.getI1Type()}));
    opaque.setPrivate();
    opaque->moveBefore(function);
    b.setInsertionPoint(&function.front().back());
    auto unavailable = b.create<func::CallOp>(loc, "opaque", TypeRange{b.getI1Type()}, ValueRange{}).getResult(0);
    auto late = std::make_shared<fs::RegionalRelations>(*relation);
    late->data.parameters[0] = arena->input(unavailable);
    invalid = region;
    invalid.relations = late;
    invalid.invocationPrefix = fs::relationalPrefixContribution(invalid, input);
    fs::InvocationPrefixCollection lateCollection(function, input);
    if (!invalid.invocationPrefix || failed(fs::collectRegionalInvocationPrefix(invalid, lateCollection, diagnostic))) {
        return relationalPrefixFailure(__LINE__);
    }
    lateCollection.recipe().privateScalarControl = true;
    lateCollection.recipe().sourceBounds = spec.sourceBounds;
    const auto beforeFailure = render(function);
    if (succeeded(fs::materializeInvocationPrefix(function, lateCollection.recipe(), error)) ||
        render(function) != beforeFailure) {
        return relationalPrefixFailure(__LINE__);
    }
    // A source/target on one nonscalar pipe requires a guarded barrier rather
    // than a consuming event. Preserve the other directions' ownership.
    phases[2].kPipeValue = static_cast<pto::PipelineType>(Q);
    invalid = region;
    invalid.invocationPrefix = fs::relationalPrefixContribution(invalid, input);
    fs::InvocationPrefixCollection samePipe(function, input);
    if (!invalid.invocationPrefix || failed(fs::collectRegionalInvocationPrefix(invalid, samePipe, diagnostic))) {
        return relationalPrefixFailure(__LINE__);
    }
    samePipe.recipe().privateScalarControl = true;
    samePipe.recipe().sourceBounds = {{P, 2}, {Q, 3}};
    auto same = fs::materializeInvocationPrefix(function, samePipe.recipe(), error);
    unsigned barriers = 0;
    if (succeeded(same)) {
        (*same)->function.walk([&](pto::BarrierOp) { ++barriers; });
    }
    phases[2].kPipeValue = static_cast<pto::PipelineType>(R);
    if (failed(same) || !barriers) {
        return relationalPrefixFailure(__LINE__);
    }
    auto wrong = std::make_shared<fs::RegionalRelations>(*relation);
    auto key = wrong->newCrossings.begin()->first;
    auto pieces = wrong->newCrossings.begin()->second;
    wrong->newCrossings.erase(wrong->newCrossings.begin());
    key.source.residues.push_back(0);
    wrong->newCrossings.emplace(key, pieces);
    invalid = region;
    invalid.relations = wrong;
    invalid.invocationPrefix = fs::relationalPrefixContribution(invalid, input);
    fs::InvocationPrefixCollection malformed(function, input);
    if (!invalid.invocationPrefix || succeeded(fs::collectRegionalInvocationPrefix(invalid, malformed, diagnostic))) {
        return relationalPrefixFailure(__LINE__);
    }
    b.setInsertionPoint(&function.front().back());
    auto one = b.create<arith::ConstantIndexOp>(loc, 1);
    auto loop = b.create<scf::ForOp>(loc, function.getArgument(0), function.getArgument(0), one);
    phases[0].elementOp->moveBefore(&loop.getBody()->front());
    invalid = region;
    invalid.anchors[0].before = {phases[0].elementOp->getBlock(), phases[0].elementOp};
    invalid.anchors[0].after = {phases[0].elementOp->getBlock(), phases[0].elementOp->getNextNode()};
    return !fs::relationalPrefixContribution(invalid, input);
}

// Exercise real effect recovery, leaf mathematics, relational composition,
// sequence export and the root coordinator without synthetic child callbacks.
bool productionRelationalPrefix(MLIRContext& context)
{
    auto module = parseSourceString<ModuleOp>(
        R"mlir(
!tile = !pto.tile_buf<mat, 16x16xf16, blayout=col_major, slayout=row_major>
!left = !pto.tile_buf<left, 16x16xf16, slayout=row_major>
module attributes {pto.target_arch = "a3"} {
  func.func @composed(%a: !pto.ptr<f16, gm>, %g: i1) {
    %base = arith.constant 0 : i64
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %n = arith.constant 16 : index
    %view = pto.make_tensor_view %a, shape = [%n, %n], strides = [%n, %one]
      {layout = #pto.layout<nd>} : !pto.tensor_view<?x?xf16>
    %part = pto.partition_view %view, offsets = [%zero, %zero], sizes = [%n, %n] : !pto.tensor_view<?x?xf16>
    %tile = pto.alloc_tile addr = %base : !tile
    %left = pto.alloc_tile addr = %base : !left
    scf.if %g {
    pto.tload ins(%part : !pto.partition_tensor_view<16x16xf16>) outs(%tile : !tile)
    pto.textract ins(%tile, %zero, %zero : !tile, index, index) outs(%left : !left)
    pto.tload ins(%part : !pto.partition_tensor_view<16x16xf16>) outs(%tile : !tile)
    }
    return
  }
})mlir",
        &context);
    if (!module) {
        return relationalPrefixFailure(__LINE__);
    }
    auto function = *module->getOps<func::FuncOp>().begin();
    auto input = std::make_shared<pto::SyncInput>();
    fs::PhaseIndex index;
    if (failed(input->build(function)) || failed(index.build(function, *input)) || input->instructions().size() != 3) {
        return relationalPrefixFailure(__LINE__);
    }
    auto arena = std::make_shared<fs::RegionExpressions>();
    std::vector<fs::RegionalAnalysis> children;
    std::vector<std::shared_ptr<const fs::MathematicalResult>> owners;
    for (unsigned first : {0U, 2U}) {
        const auto phases = input->instructions().slice(first, first == 0 ? 2 : 1);
        SmallVector<Operation*> roots;
        for (auto* phase : phases) {
            roots.push_back(phase->elementOp);
        }
        auto finite = fs::analyzeFiniteGuarded(function, roots, index, *input, arena);
        auto exact = std::make_shared<fs::ExplicitAnalysis>(fs::analyzeExplicit(phases, index, *input));
        if (!finite.error.empty() || !exact->error.empty() || (first == 0 && exact->reduction.retained.empty())) {
            return relationalPrefixFailure(__LINE__);
        }
        auto regional = fs::finiteGuardedRegionalResult(finite);
        auto retained = std::make_shared<fs::MathematicalResult>();
        retained->input = input;
        retained->explicitDemands = exact;
        retained->backend = "explicit";
        regional.retainedMathematics = retained;
        owners.push_back(retained);
        children.push_back(regional);
    }
    std::string error;
    auto composed = fs::composeSymbolicRegionalSequence(children, function, *input, index, error);
    if (failed(composed) || !composed->relations || composed->relations->newCrossings.empty()) {
        llvm::errs() << "production relational composition: " << error << "\n";
        return relationalPrefixFailure(__LINE__);
    }
    // The relation need not carry internal F*: actual child mathematics owns
    // that handoff. Retain exact required closure and certified new crossings.
    auto carrier = std::make_shared<fs::RegionalRelations>(*composed->relations);
    carrier->data.analysis.minimumDemands = carrier->newCrossings;
    carrier->data.analysis.exactMinimum = false;
    composed->relations = std::move(carrier);
    auto state = std::make_shared<fs::SequenceAnalysisState>(function, arena);
    state->input = input.get();
    state->relationalResult = *composed;
    for (unsigned i = 0; i < children.size(); ++i) {
        fs::Child child;
        child.regional = children[i];
        child.anchors = children[i].anchors;
        child.prefixMathematics = owners[i];
        state->children.push_back(std::move(child));
    }
    fs::SequenceAnalysis sequence;
    sequence.state = state;
    auto regional = fs::sequenceRegionalResult(sequence);
    if (!regional.invocationPrefix) {
        return relationalPrefixFailure(__LINE__);
    }
    auto branch = *function.getOps<scf::IfOp>().begin();
    const auto activity = arena->input(function.getArgument(1));
    auto masked = fs::guardOriginalArm(regional, activity, branch.getThenRegion());
    auto arbitrary = fs::guardOriginalArm(regional, arena->lnot(activity), branch.getThenRegion());
    if (!masked.invocationPrefix || arbitrary.invocationPrefix || masked.relations) {
        return relationalPrefixFailure(__LINE__);
    }
    auto original = render(function);
    fs::InvocationPrefixCollection collection(function, input);
    if (failed(fs::collectRegionalInvocationPrefix(masked, collection, error))) {
        llvm::errs() << "production relational collection: " << error << "\n";
        return relationalPrefixFailure(__LINE__);
    }
    auto& spec = collection.recipe();
    spec.privateScalarControl = true;
    for (auto* phase : input->instructions()) {
        ++spec.sourceBounds[static_cast<uint32_t>(phase->kPipeValue)];
    }
    fs::PrivateStateError diagnostic;
    auto staged = fs::materializeInvocationPrefix(function, spec, diagnostic);
    if (failed(staged)) {
        llvm::errs() << "production relational materialization: " << diagnostic.message << "\n";
        return relationalPrefixFailure(__LINE__);
    }
    unsigned sets = 0, waits = 0;
    (*staged)->function.walk([&](Operation* op) {
        sets += op->getName().getStringRef() == "pto.logical_set";
        waits += op->getName().getStringRef() == "pto.logical_wait";
    });
    auto missing = std::make_shared<fs::SequenceAnalysisState>(function, arena);
    missing->input = input.get();
    missing->relationalResult = *composed;
    missing->children = state->children;
    missing->children[0].prefixMathematics.reset();
    fs::SequenceAnalysis missingAnalysis;
    missingAnalysis.state = missing;
    auto missingRegion = fs::sequenceRegionalResult(missingAnalysis);
    fs::InvocationPrefixCollection refused(function, input);
    if (missingRegion.invocationPrefix &&
        succeeded(fs::collectRegionalInvocationPrefix(missingRegion, refused, error))) {
        return relationalPrefixFailure(__LINE__);
    }
    return sets == 3 && waits == 4 && render(function) == original && succeeded(verify((*staged)->function));
}

#endif
