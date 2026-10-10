// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "../../lib/PTO/Transforms/FrontierSynch/RegionalRelationsInternal.h"
#include "../../lib/PTO/Transforms/FrontierSynch/SequenceAnalysisInternal.h"
#include "../../lib/PTO/Transforms/FrontierSynch/ScalarPrerequisiteMapping.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/ScopeExit.h"
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
using I = fs::BoundInteger;
bool contains(const fs::IntegerSystem& system, ArrayRef<int64_t> point)
{
    if (system.dimensions() != point.size()) { return false; }
    auto dot = [&](const auto& coefficients) {
        I result(0);
        for (unsigned i = 0; i < point.size(); ++i) { result += coefficients[i] * I(point[i]); }
        return result;
    };
    for (const auto& row : system.constraints()) { if (dot(row.coefficients) > row.bound) { return false; } }
    for (const auto& row : system.congruences()) {
        if ((dot(row.coefficients) - row.residue) % row.modulus != I(0)) { return false; }
    }
    return true;
}
bool quotientCoordinates(MLIRContext& context)
{
    Block parameters;
    for (unsigned i = 0; i < 3; ++i) { parameters.addArgument(IndexType::get(&context), UnknownLoc::get(&context)); }
    auto arena = std::make_shared<fs::RegionExpressions>();
    fs::RegionalRelationData data;
    data.analysis.period = 3; data.analysis.parameterCount = 2;
    data.selectors.period = 3; data.selectors.parameterCount = 2;
    data.parameterValues = {parameters.getArgument(1), parameters.getArgument(0)};
    data.parameters = {arena->input(parameters.getArgument(1)), arena->input(parameters.getArgument(0))};
    auto system = fs::IntegerSystem::create(4,
        {{{I(1), I(-1), I(0), I(0)}, I(2)}, {{I(0), I(0), I(-1), I(0)}, I(3)}},
        {{{I(2), I(0), I(1), I(0)}, I(1), I(4)}});
    if (failed(system)) { return false; }
    fs::ArithmeticRelationKey key{{0, fs::ArithmeticEvent::Completion, {1}},
                                  {1, fs::ArithmeticEvent::Start, {2}}, {0, 1}};
    data.analysis.requiredOrder[key] = {*system};
    fs::ArithmeticBoundarySelector boundary;
    boundary.kind = fs::ArithmeticBoundaryKind::FirstWriter; boundary.storageSpace = pto::AddressSpace::VEC;
    boundary.selector.inputDimensions = 1; boundary.selector.parameterCount = 2;
    auto domain = fs::IntegerSystem::create(3, {{{I(-1), I(0), I(0)}, I(0)},
                                              {{I(1), I(0), I(0)}, I(8)}});
    if (failed(domain)) { return false; }
    boundary.selector.pieces.push_back({*domain, {1}, {0, 1}, 0,
        {{{{I(2), I(1), I(0)}, I(3)}, I(1), 2}}});
    data.selectors.boundaries.push_back(boundary);
    SmallVector<Value> values{parameters.getArgument(0), parameters.getArgument(2), parameters.getArgument(1)};
    std::vector<fs::RegionExpressions::Id> bindings;
    for (auto value : values) { bindings.push_back(arena->input(value)); }
    std::string error;
    if (failed(fs::normalizeRegionalRelationData(data, values, bindings, error)) || data.analysis.period != 1 ||
        data.analysis.requiredOrder.size() != 1) { llvm::errs() << error << '\n'; return false; }
    const auto& actual = data.analysis.requiredOrder.begin()->second.front();
    for (int64_t x = -8; x <= 8; ++x) {
        for (int64_t y = -8; y <= 8; ++y) {
            for (int64_t p = -3; p <= 3; ++p) {
                for (int64_t q = -3; q <= 3; ++q) {
                    const bool onLattice = (x - 1) % 3 == 0 && (y - 2) % 3 == 0 && p % 3 == 0 && (q - 1) % 3 == 0;
                    const bool expected = onLattice &&
                        contains(*system, {(x - 1) / 3, (y - 2) / 3, p / 3, (q - 1) / 3});
                    if (contains(actual, {x, y, q, 7, p}) != expected) { return false; }
                }
            }
        }
    }
    const auto& selected = data.selectors.boundaries.front().selector;
    for (int64_t byte = -2; byte < 30; ++byte) {
        auto value = fs::evaluateGeneralArithmeticSelector(selected, 1, {I(byte)}, {I(1), I(7), I(0)});
        const bool present = byte >= 1 && byte <= 25 && (byte - 1) % 3 == 0;
        if (failed(value) || value->has_value() != present) { return false; }
        if (present && (*value)->coordinates != std::vector<I>{I(2 * byte + 9)}) { return false; }
    }
    return true;
}
fs::RegionalRelationData leaf(const pto::SyncInput& input, func::FuncOp function,
    const pto::CompoundInstanceElement* phase, bool writes)
{
    fs::RegionalRelationData data;
    data.input = &input; data.context = {function, function}; data.sites.push_back({phase, {}, {}});
    data.analysis.period = 1; data.analysis.parameterCount = 1; data.analysis.pipeCount = 1;
    data.analysis.exactMinimum = true; data.completeRequiredOrder = true;
    data.selectors.period = 1; data.selectors.parameterCount = 1;
    data.parameterValues.push_back(function.getArgument(0)); data.parameters = {0};
    auto present = fs::IntegerSystem::create(1, {{{I(-1)}, I(-1)}});
    data.occurrences.push_back({0, {}, {0}, *present});
    for (auto a : {fs::ArithmeticEvent::Start, fs::ArithmeticEvent::Completion}) {
        for (auto b : {fs::ArithmeticEvent::Start, fs::ArithmeticEvent::Completion}) {
            if (a == fs::ArithmeticEvent::Completion && b == fs::ArithmeticEvent::Start) { continue; }
            fs::ArithmeticRelationKey key{{0, a, {}}, {0, b, {}}, {0}};
            data.analysis.nativeOrder[key] = {*present};
            if (a != b) { data.analysis.requiredOrder[key] = {*present}; }
        }
    }
    auto bytes = fs::IntegerSystem::create(2,
        {{{I(-1), I(0)}, I(0)}, {{I(1), I(-1)}, I(-1)}});
    data.selectors.support.push_back({pto::AddressSpace::VEC, {}, 0, {0}, *bytes});
    for (bool first : {true, false}) {
        fs::ArithmeticBoundarySelector selector;
        using Boundary = fs::ArithmeticBoundaryKind;
        selector.kind = writes ? (first ? Boundary::FirstWriter : Boundary::LastWriter)
            : (first ? Boundary::FirstReaderBeforeWrite : Boundary::LastReaderAfterWrite);
        selector.storageSpace = pto::AddressSpace::VEC;
        if (!writes) { selector.pipe = static_cast<uint32_t>(phase->kPipeValue); }
        selector.selector.inputDimensions = 1; selector.selector.parameterCount = 1;
        selector.selector.pieces.push_back({*bytes, {0}, {0}, 0, {}});
        data.selectors.boundaries.push_back(std::move(selector));
    }
    return data;
}
bool association(MLIRContext& context)
{
    context.getOrLoadDialect<func::FuncDialect>();
    auto function = func::FuncOp::create(UnknownLoc::get(&context), "relations",
        FunctionType::get(&context, {IndexType::get(&context)}, {}));
    function.addEntryBlock();
    auto cleanup = llvm::make_scope_exit([&] { function.erase(); });
    pto::SyncInput input;
    pto::CompoundInstanceElement a(0, {}, {}, pto::PipelineType::PIPE_MTE2, OperationName("func.call", &context));
    pto::CompoundInstanceElement b(1, {}, {}, pto::PipelineType::PIPE_V, OperationName("func.call", &context));
    pto::CompoundInstanceElement c(2, {}, {}, pto::PipelineType::PIPE_MTE3, OperationName("func.call", &context));
    OpBuilder builder(&function.front(), function.front().begin());
    a.elementOp = builder.create<func::CallOp>(function.getLoc(), "a", TypeRange{}, ValueRange{});
    b.elementOp = builder.create<func::CallOp>(function.getLoc(), "b", TypeRange{}, ValueRange{});
    c.elementOp = builder.create<func::CallOp>(function.getLoc(), "c", TypeRange{}, ValueRange{});
    auto left = leaf(input, function, &a, true);
    auto middle = leaf(input, function, &b, false), right = leaf(input, function, &c, true);
    std::string error;
    auto lm = fs::composeRegionalRelationData(left, middle, {function, function}, error);
    auto mr = fs::composeRegionalRelationData(middle, right, {function, function}, error);
    if (failed(lm) || failed(mr)) { llvm::errs() << error << '\n'; return false; }
    auto lmr = fs::composeRegionalRelationData(*lm, right, {function, function}, error);
    auto lmrOther = fs::composeRegionalRelationData(left, *mr, {function, function}, error);
    if (failed(lmr) || failed(lmrOther)) { llvm::errs() << error << '\n'; return false; }
    for (int64_t n : {0, 1, 3, 1000000000}) {
        std::set<std::pair<unsigned, unsigned>> expected;
        if (n > 0) { expected = {{0, 1}, {1, 2}}; }
        for (const auto* data : {&*lmr, &*lmrOther}) {
            std::set<std::pair<unsigned, unsigned>> found;
            for (const auto& [key, pieces] : data->analysis.minimumDemands) {
                if (llvm::any_of(pieces, [&](const auto& piece) { return contains(piece, {n}); })) {
                    found.insert({key.source.site, key.target.site});
                }
            }
            if (found != expected) { return false; }
        }
    }
    return true;
}

bool retainedResult(MLIRContext& context)
{
    context.getOrLoadDialect<func::FuncDialect>();
    auto function = func::FuncOp::create(UnknownLoc::get(&context), "retained",
        FunctionType::get(&context, {IndexType::get(&context)}, {}));
    auto* body = function.addEntryBlock();
    auto cleanup = llvm::make_scope_exit([&] { function.erase(); });
    OpBuilder builder(&context); builder.setInsertionPointToStart(body);
    auto first = builder.create<func::CallOp>(function.getLoc(), "first", TypeRange{}, ValueRange{});
    auto second = builder.create<func::CallOp>(function.getLoc(), "second", TypeRange{}, ValueRange{});
    builder.create<func::ReturnOp>(function.getLoc());
    pto::CompoundInstanceElement a(0, {}, {}, pto::PipelineType::PIPE_MTE2, first->getName());
    pto::CompoundInstanceElement b(1, {}, {}, pto::PipelineType::PIPE_V, second->getName());
    a.elementOp = first; b.elementOp = second;
    pto::SyncInput input;
    auto arena = std::make_shared<fs::RegionExpressions>();
    auto n = arena->input(function.getArgument(0));
    auto left = leaf(input, function, &a, true), right = leaf(input, function, &b, false);
    left.parameters = {n}; right.parameters = {n};
    std::string error;
    auto composed = fs::composeRegionalRelationData(left, right, {function, function}, error);
    if (failed(composed)) { return false; }
    auto carrier = std::make_shared<fs::RegionalRelations>(); carrier->data = *composed;
    carrier->newCrossings = composed->analysis.minimumDemands;
    for (auto* phase : {&a, &b}) {
        fs::RegionalAnalysis child; child.expressions = arena; child.accessModel = &input.accesses();
        child.gmAliasPolicy = input.memory().gmPolicy(); child.capabilities = {true, true, true, false};
        child.occurrenceLoops.push_back({}); child.outerLoops.push_back({});
        auto* op = phase->elementOp;
        child.anchors.push_back({phase, {}, {body, op}, {body, op->getNextNode()}});
        carrier->children.push_back(std::move(child));
    }
    auto exported = fs::exportRegionalRelationData(carrier, arena, error);
    if (failed(exported) || exported->relations != carrier || exported->capabilities.endpointRecipes ||
        !exported->prepare || succeeded(exported->prepare())) { return false; }
    fs::ProgramRecognition program;
    program.nodes.emplace_back(); program.nodes.front().anchor = function;
    program.nodes.front().region = &function.getBody();
    auto state = std::make_shared<fs::SequenceAnalysisState>(function, input, program);
    state->relationalResult = *exported; state->error = "missing endpoint adapter";
    fs::SequenceAnalysis analysis; analysis.state = state; analysis.insertionError = state->error;
    fs::recordSequenceContractAttempt(program, input, analysis);
    if (!program.sequenceContract || program.sequenceContract->membership != fs::ContractStatus::Established ||
        program.sequenceContract->demands != fs::ContractImplementation::Available ||
        program.sequenceContract->implementationError != analysis.insertionError) { return false; }
    return true;
}
bool scalarCrossings(MLIRContext& context)
{
    context.getOrLoadDialect<arith::ArithDialect>();
    auto function = func::FuncOp::create(UnknownLoc::get(&context), "scalar_relations",
        FunctionType::get(&context, {IndexType::get(&context)}, {}));
    function.addEntryBlock();
    auto cleanup = llvm::make_scope_exit([&] { function.erase(); });
    OpBuilder builder(&function.front(), function.front().begin());
    auto aOp = builder.create<arith::ConstantIndexOp>(function.getLoc(), 1);
    auto bOp = builder.create<arith::AddIOp>(function.getLoc(), aOp, function.getArgument(0));
    auto cOp = builder.create<arith::AddIOp>(function.getLoc(), bOp, aOp);
    pto::SyncInput input;
    pto::CompoundInstanceElement a(0, {}, {}, pto::PipelineType::PIPE_MTE2, aOp->getName());
    pto::CompoundInstanceElement b(1, {}, {}, pto::PipelineType::PIPE_V, bOp->getName());
    pto::CompoundInstanceElement c(2, {}, {}, pto::PipelineType::PIPE_MTE3, cOp->getName());
    a.elementOp = aOp; b.elementOp = bOp; c.elementOp = cOp;
    auto make = [&](const auto* phase) {
        auto data = leaf(input, function, phase, false);
        data.selectors = {}; data.selectors.period = 1; data.selectors.parameterCount = 1;
        return data;
    };
    auto left = make(&a), right = make(&c);
    for (bool native : {false, true}) {
        right.incomingPrerequisites = {{&a, cOp, native, true}};
        std::string error;
        auto combined = fs::composeRegionalRelationData(left, right, {function, function}, error);
        const bool resolved = succeeded(combined) && combined->incomingPrerequisites.empty();
        if (!resolved) {
            llvm::errs() << error << '\n'; return false;
        }
        for (int64_t n : {0, 1, 3, 1000000000}) {
            fs::ArithmeticRelationKey key{{0, fs::ArithmeticEvent::Completion, {}},
                {1, fs::ArithmeticEvent::Start, {}}, {0}};
            auto has = [&](const auto& relation) {
                auto found = relation.find(key);
                return found != relation.end() && llvm::any_of(found->second,
                    [&](const auto& piece) { return contains(piece, {n}); });
            };
            const bool correct = has(combined->analysis.minimumDemands) == (n > 0 && !native) &&
                has(combined->analysis.nativeOrder) == (n > 0 && native) &&
                has(combined->analysis.requiredOrder) == (n > 0);
            if (!correct) { return false; }
        }
    }
    // Retain an absent producer through an intermediate merge, then resolve
    // its exact original consumer after the producer enters the parent.
    auto middle = make(&b);
    right.incomingPrerequisites = {{&a, cOp, false, true}};
    std::string error;
    auto suffix = fs::composeRegionalRelationData(middle, right, {function, function}, error);
    const bool retained = succeeded(suffix) && suffix->incomingPrerequisites.size() == 1;
    if (!retained) { return false; }
    auto complete = fs::composeRegionalRelationData(left, *suffix, {function, function}, error);
    const bool resolved = succeeded(complete) && complete->incomingPrerequisites.empty();
    if (!resolved) { return false; }
    fs::ArithmeticRelationKey expected{{0, fs::ArithmeticEvent::Completion, {}},
        {2, fs::ArithmeticEvent::Start, {}}, {0}};
    const bool minimum = complete->analysis.minimumDemands.size() == 1 &&
        complete->analysis.minimumDemands.find(expected) != complete->analysis.minimumDemands.end();
    if (!minimum) { return false; }
    right.incomingPrerequisites.front().directSSA = false;
    if (succeeded(fs::composeRegionalRelationData(left, right, {function, function}, error))) { return false; }
    left.incomingPrerequisites = {{&c, aOp, false, true}};
    right.incomingPrerequisites.clear();
    if (succeeded(fs::composeRegionalRelationData(left, right, {function, function}, error))) { return false; }
    return true;
}
bool sequentialAdmission(MLIRContext& context)
{
    context.getOrLoadDialect<scf::SCFDialect>();
    auto function = func::FuncOp::create(UnknownLoc::get(&context), "invocations",
        FunctionType::get(&context, {IndexType::get(&context)}, {}));
    function.addEntryBlock();
    auto cleanup = llvm::make_scope_exit([&] { function.erase(); });
    OpBuilder builder(&function.front(), function.front().begin());
    auto zero = builder.create<arith::ConstantIndexOp>(function.getLoc(), 0);
    auto step = builder.create<arith::ConstantIndexOp>(function.getLoc(), 3);
    auto loop = builder.create<scf::ForOp>(function.getLoc(), zero, function.getArgument(0), step);
    builder.setInsertionPointToStart(loop.getBody());
    auto aOp = builder.create<arith::AddIOp>(function.getLoc(), loop.getInductionVar(), step);
    auto bOp = builder.create<arith::AddIOp>(function.getLoc(), aOp, step);
    pto::CompoundInstanceElement a(0, {}, {}, pto::PipelineType::PIPE_MTE2, aOp->getName());
    pto::CompoundInstanceElement b(1, {}, {}, pto::PipelineType::PIPE_V, bOp->getName());
    a.elementOp = aOp; b.elementOp = bOp;
    pto::SyncInput input;
    auto left = leaf(input, function, &a, false), right = leaf(input, function, &b, false);
    for (auto* data : {&left, &right}) {
        data->selectors = {}; data->selectors.period = 1; data->selectors.parameterCount = 1;
        data->enclosing = {loop}; data->parameterValues = {loop.getInductionVar()};
    }
    right.incomingPrerequisites = {{&a, bOp, true, true}};
    std::string error;
    auto bound = fs::composeRegionalRelationData(left, right, {function, loop}, error);
    const bool boundAccepted = succeeded(bound);
    if (!boundAccepted) {
        llvm::errs() << error << '\n';
        return false;
    }
    auto reversed = fs::composeRegionalRelationData(right, left, {function, loop}, error);
    const bool reversedRejected = failed(reversed) && error.find("sequential invocation") != std::string::npos;
    if (!reversedRejected) { return false; }
    auto savedAnchor = a.elementOp;
    a.elementOp = loop;
    auto overlap = fs::composeRegionalRelationData(left, right, {function, loop}, error);
    a.elementOp = savedAnchor;
    const bool overlapRejected = failed(overlap) && error.find("sequential invocation") != std::string::npos;
    if (!overlapRejected) { return false; }
    builder.setInsertionPointAfter(bOp);
    auto one = builder.create<arith::ConstantIndexOp>(function.getLoc(), 1);
    auto inner = builder.create<scf::ForOp>(function.getLoc(), zero, loop.getInductionVar(), one);
    builder.setInsertionPointToStart(inner.getBody());
    auto nestedOp = builder.create<arith::AddIOp>(function.getLoc(), aOp, inner.getInductionVar());
    b.elementOp = nestedOp;
    auto nested = right;
    nested.sites.front().loops = {inner};
    nested.incomingPrerequisites = {{&a, nestedOp, true, true}};
    nested.occurrences.front().residues = {0};
    nested.occurrences.front().system = *fs::IntegerSystem::create(2,
        {{{I(-1), I(0)}, I(0)}, {{I(1), I(-1)}, I(-1)}});
    auto diagonal = fs::IntegerSystem::create(3,
        {{{I(-1), I(0), I(0)}, I(0)}, {{I(1), I(0), I(-1)}, I(-1)},
         {{I(1), I(-1), I(0)}, I(0)}, {{I(-1), I(1), I(0)}, I(0)}});
    for (auto* relation : {&nested.analysis.nativeOrder, &nested.analysis.requiredOrder}) {
        fs::GeneralArithmeticRelation mapped;
        for (const auto& [key, pieces] : *relation) {
            auto newKey = key; newKey.source.residues = {0}; newKey.target.residues = {0};
            mapped[newKey] = {*diagonal};
        }
        *relation = std::move(mapped);
    }
    auto prefix = fs::composeRegionalRelationData(left, nested, {function, loop}, error);
    b.elementOp = bOp;
    const bool prefixAccepted = succeeded(prefix) && prefix->analysis.minimumDemands.empty();
    if (!prefixAccepted) { return false; }
    fs::ArithmeticRelationKey edge{{0, fs::ArithmeticEvent::Completion, {}},
        {1, fs::ArithmeticEvent::Start, {0}}, {0}};
    auto found = prefix->analysis.nativeOrder.find(edge);
    if (found == prefix->analysis.nativeOrder.end()) { return false; }
    for (int64_t n : {0, 1, 3, 9}) {
        for (int64_t j = -1; j <= n; ++j) {
            const bool present = llvm::any_of(found->second,
                [&](const auto& piece) { return contains(piece, {j, n}); });
            if (present != (n > 0 && j >= 0 && j < n)) { return false; }
        }
    }
    // Independently enumerating the same original loop is not sequential
    // composition. Its crossings require a fiberwise repetition adapter.
    for (auto* data : {&left, &right}) {
        data->enclosing.clear(); data->sites.front().loops = {loop};
        auto domain = fs::IntegerSystem::create(2, {});
        data->occurrences.front().residues = {0}; data->occurrences.front().system = *domain;
        for (auto* relation : {&data->analysis.nativeOrder, &data->analysis.requiredOrder}) {
            fs::GeneralArithmeticRelation mapped;
            for (const auto& [key, pieces] : *relation) {
                auto newKey = key; newKey.source.residues = {0}; newKey.target.residues = {0};
                mapped[newKey] = {*fs::IntegerSystem::create(3, {})};
            }
            *relation = std::move(mapped);
        }
    }
    auto dynamic = fs::composeRegionalRelationData(left, right, {function, function}, error);
    const bool dynamicRejected = failed(dynamic) && error.find("sequential invocation") != std::string::npos;
    if (!dynamicRejected) { return false; }
    // Opposite conditional arms do not share a sequential invocation scope.
    builder.setInsertionPointAfter(loop);
    auto condition = builder.create<arith::ConstantIntOp>(function.getLoc(), 1, 1);
    auto branch = builder.create<scf::IfOp>(function.getLoc(), condition, true);
    builder.setInsertionPointToStart(&branch.getThenRegion().front());
    a.elementOp = builder.create<arith::ConstantIndexOp>(function.getLoc(), 1);
    builder.setInsertionPointToStart(&branch.getElseRegion().front());
    b.elementOp = builder.create<arith::ConstantIndexOp>(function.getLoc(), 2);
    left = leaf(input, function, &a, false); right = leaf(input, function, &b, false);
    auto arms = fs::composeRegionalRelationData(left, right, {function, function}, error);
    return failed(arms) && error.find("sequential invocation") != std::string::npos;
}
bool fixedPrerequisiteMappings(MLIRContext& context)
{
    auto function = func::FuncOp::create(UnknownLoc::get(&context), "fixed_prerequisites",
        FunctionType::get(&context, {}, {}));
    function.addEntryBlock();
    auto cleanup = llvm::make_scope_exit([&] { function.erase(); });
    OpBuilder builder(&function.front(), function.front().begin());
    auto zero = builder.create<arith::ConstantIndexOp>(function.getLoc(), 0);
    auto one = builder.create<arith::ConstantIndexOp>(function.getLoc(), 1);
    auto two = builder.create<arith::ConstantIndexOp>(function.getLoc(), 2);
    auto outside = builder.create<arith::AddIOp>(function.getLoc(), zero, one);
    auto loop = builder.create<scf::ForOp>(function.getLoc(), zero, two, one);
    builder.setInsertionPointToStart(loop.getBody());
    auto producer = builder.create<arith::AddIOp>(function.getLoc(), outside, loop.getInductionVar());
    auto consumer = builder.create<arith::AddIOp>(function.getLoc(), producer, one);
    pto::CompoundInstanceElement a(0, {}, {}, pto::PipelineType::PIPE_S, producer->getName());
    pto::CompoundInstanceElement b(1, {}, {}, pto::PipelineType::PIPE_S, consumer->getName());
    pto::CompoundInstanceElement outer(2, {}, {}, pto::PipelineType::PIPE_S, outside->getName());
    a.elementOp = producer; b.elementOp = consumer; outer.elementOp = outside;
    DominanceInfo dominance(function);
    fs::ValuePrerequisite edge{&a, consumer, true, true};
    for (int64_t source : {0, 1}) {
        for (int64_t target : {0, 1}) {
            fs::ArithmeticSite from{&a, {}, {}, {{loop, source}}};
            fs::ArithmeticSite to{&b, {}, {}, {{loop, target}}};
            const bool mapped = fs::detail::directPrerequisiteMapping(edge, from, to, dominance);
            if (mapped != (source == target)) {
                return false;
            }
            from.fixedCoordinates.clear(); from.loops = {loop};
            if (fs::detail::directPrerequisiteMapping(edge, from, to, dominance)) { return false; }
            from.loops.clear(); from.fixedCoordinates = {{loop, source}};
            to.fixedCoordinates.clear(); to.loops = {loop};
            if (fs::detail::directPrerequisiteMapping(edge, from, to, dominance)) { return false; }
        }
    }
    fs::ArithmeticSite broadcast{&outer, {}, {}, {}};
    fs::ArithmeticSite receiver{&b, {}, {}, {{loop, 1}}};
    fs::ValuePrerequisite incoming{&outer, consumer, true, true};
    const bool broadcasts = fs::detail::directPrerequisiteMapping(incoming, broadcast, receiver, dominance);
    if (!broadcasts) { return false; }
    builder.setInsertionPointAfter(loop);
    auto sibling = builder.create<scf::ForOp>(function.getLoc(), zero, two, one);
    fs::ArithmeticSite from{&a, {}, {}, {{loop, 0}}};
    receiver.fixedCoordinates = {{sibling, 0}};
    return !fs::detail::directPrerequisiteMapping(edge, from, receiver, dominance);
}
bool nativeScalarClosure()
{
    // Independently close a finite all-event graph. Native scalar edges are
    // mixed with intrinsic I->C and FIFO order, including a two-edge chain.
    using E = fs::ArithmeticEvent;
    fs::GeneralArithmeticRelation input;
    auto domain = fs::IntegerSystem::create(0, {});
    if (failed(domain)) { return false; }
    bool expected[6][6] = {};
    auto edge = [&](unsigned a, unsigned b) {
        fs::ArithmeticRelationKey key{{a/2, a%2 ? E::Completion : E::Start, {}},
            {b/2, b%2 ? E::Completion : E::Start, {}}, {}};
        input[key] = {*domain}; expected[a][b] = true;
    };
    for (unsigned site = 0; site < 3; ++site) {
        edge(2*site, 2*site); edge(2*site+1, 2*site+1); edge(2*site, 2*site+1);
    }
    edge(0, 2); edge(1, 3); edge(0, 3); // Two scalar events on the same pipe.
    edge(1, 2); edge(3, 4); // Synchronous scalar completion prerequisites.
    for (unsigned k = 0; k < 6; ++k) {
        for (unsigned a = 0; a < 6; ++a) {
            for (unsigned b = 0; b < 6; ++b) { expected[a][b] |= expected[a][k] && expected[k][b]; }
        }
    }
    fs::ArithmeticAnalysisCost cost;
    std::string error;
    auto actual = fs::completeGeneralArithmeticNativeOrder(std::move(input), 2, 0, cost, error);
    if (failed(actual)) { return false; }
    for (unsigned a = 0; a < 6; ++a) {
        for (unsigned b = 0; b < 6; ++b) {
            fs::ArithmeticRelationKey key{{a/2, a%2 ? E::Completion : E::Start, {}},
                {b/2, b%2 ? E::Completion : E::Start, {}}, {}};
            auto found = actual->find(key);
            const bool present = found != actual->end() && llvm::any_of(found->second,
                [&](const auto& piece) { return contains(piece, {}); });
            if (present != expected[a][b]) { return false; }
        }
    }
    return cost.relationCompositions > 0;
}
bool retainedInputs(MLIRContext& context)
{
    std::weak_ptr<Block> weak;
    auto arena = std::make_shared<fs::RegionExpressions>();
    fs::RegionExpressions::Id input = 0;
    {
        auto owner = std::make_shared<Block>(); weak = owner;
        auto value = owner->addArgument(IndexType::get(&context), UnknownLoc::get(&context));
        arena->retainInputOwner(owner); input = arena->input(value);
        auto domain = fs::IntegerSystem::create(0, {});
        std::string error;
        // A failed export must not destroy SSA inputs retained by query DAGs.
        if (succeeded(arena->integerRelation(arena->eq(input, arena->constant(1)), {}, *domain, error))) {
            return false;
        }
    }
    auto inputs = arena->referencedInputs(input);
    if (weak.expired() || inputs.size() != 1 || !inputs.front().second.getType().isIndex()) { return false; }
    arena.reset();
    return weak.expired();
}
} // namespace
int runRegionalRelationChecks()
{
    MLIRContext context;
    if (!quotientCoordinates(context) || !association(context) ||
        !retainedResult(context) || !retainedInputs(context) ||
        !scalarCrossings(context) || !sequentialAdmission(context) ||
        !fixedPrerequisiteMappings(context) || !nativeScalarClosure()) {
        llvm::errs() << "regional relation normalization/composition oracle failed\n"; return 1;
    }
    llvm::outs() << "regional relations: raw coordinates, parameter permutations, "
                 << "symbolic bytes and associative covers passed\n";
    return 0;
}
