// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Real arithmetic export and parent allocation, checked by an independent command graph.
#ifndef PTO_ARITHMETIC_COORDINATE_INTEGRATION_CHECKS_H
#define PTO_ARITHMETIC_COORDINATE_INTEGRATION_CHECKS_H
#include "PTO/Transforms/FrontierSynch/Arithmetic/ArithmeticRegional.h"
#include "PTO/Transforms/FrontierSynch/Shared/EndpointCutChoices.h"
#include "PTO/Transforms/FrontierSynch/Shared/FamilyInsertion.h"
#include "PTO/Transforms/FrontierSynch/Shared/RegionalAllocation.h"
#include "PTO/Transforms/FrontierSynch/Shared/SequenceAnalysis.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Verifier.h"
namespace {
using IntegrationInteger = fs::BoundInteger;
bool scalarFreeCoordinatePlan(const fs::PreparedLogicalPlan& plan)
{
    return !plan.familyWork.materializations && plan.preparation.empty() && plan.endpoints.empty();
}
FailureOr<std::shared_ptr<const fs::FamilyEndpointRecipe>> integrationOneSidedRecipe(
    const fs::EndpointFamily& family, scf::ForOp loop, bool publication)
{
    auto arena = std::make_shared<fs::RegionExpressions>();
    fs::ContextualEndpointFamily endpoint;
    endpoint.piece.namespaceId = family.id;
    endpoint.piece.sourcePipe = family.sourcePipe; endpoint.piece.targetPipe = family.targetPipe;
    endpoint.piece.kind = publication ? fs::EndpointKind::Set : fs::EndpointKind::Wait;
    endpoint.piece.cut = publication ? family.sourceCut : family.targetCut;
    endpoint.piece.members.push_back({family.members.front().record, {}});
    endpoint.activity = arena->boolean(true); endpoint.ordinal = arena->constant(0);
    endpoint.matchingCoordinates.push_back(arena->input(loop.getInductionVar()));
    return fs::buildContextualFamilyEndpointRecipe(arena, {endpoint});
}
fs::IntegerConstraint integrationRow(std::initializer_list<int64_t> values, int64_t bound)
{
    fs::IntegerConstraint row;
    for (auto value : values) { row.coefficients.emplace_back(value); }
    row.bound = IntegrationInteger(bound);
    return row;
}
// The supplied mathematical fixture has two native pipes and both handoffs
// A(i)->B(i+2), B(i)->A(i+2). Its completion/start closure is strict next-visit
// order across pipes and two-visit order on one pipe. The command oracle below
// reconstructs this causality without consulting these relations.
bool integrationMathematics(fs::ArithmeticRegionalRelations& retained, MLIRContext& context,
    func::FuncOp function, scf::ForOp loop, const pto::SyncInput& input,
    const pto::CompoundInstanceElement* a, const pto::CompoundInstanceElement* b)
{
    auto& program = retained.program;
    retained.input = &input;
    program.context = {function, loop};
    program.sites = {{a, {loop}, {}}, {b, {loop}, {}}};
    program.parameters = {function.getArgument(0)};
    program.primitives.period = 2; program.primitives.pipeCount = 2;
    program.primitives.parameters = {"limit"};
    auto primitive = [&](fs::PrimitiveKind kind, std::optional<unsigned> source,
        std::optional<unsigned> target, unsigned residue, const fs::IntegerSystem* system) {
        fs::PrimitiveRelation relation;
        relation.kind = kind; relation.sourceSite = source; relation.targetSite = target;
        relation.sourceDimensions = source ? 1 : 0; relation.targetDimensions = target ? 1 : 0;
        relation.dimensions = relation.sourceDimensions + relation.targetDimensions;
        for (unsigned i = 0; i < relation.dimensions; ++i) {
            relation.coordinates.push_back({"i" + std::to_string(i), fs::CoordinateKind::Occurrence});
        }
        relation.coordinates.push_back({"limit", fs::CoordinateKind::Parameter});
        if (system) {
            SmallVector<AffineExpr> constraints;
            SmallVector<bool> equalities;
            for (const auto& row : system->constraints()) {
                auto expression = getAffineConstantExpr(static_cast<int64_t>(row.bound), &context);
                for (unsigned i = 0; i < row.coefficients.size(); ++i) {
                    auto variable = i < relation.dimensions ? getAffineDimExpr(i, &context) :
                        getAffineSymbolExpr(i - relation.dimensions, &context);
                    expression = expression - variable * static_cast<int64_t>(row.coefficients[i]);
                }
                constraints.push_back(expression); equalities.push_back(false);
            }
            SmallVector<uint64_t> residues(relation.dimensions, 1);
            residues.push_back(residue);
            relation.pieces.push_back({IntegerSet::get(relation.dimensions, 1, constraints, equalities), residues});
        }
        program.primitives.relations.push_back(std::move(relation));
    };
    for (auto kind : {fs::PrimitiveKind::Reads, fs::PrimitiveKind::Writes,
                     fs::PrimitiveKind::Native, fs::PrimitiveKind::Prerequisites}) {
        primitive(kind, {}, {}, 0, nullptr);
    }
    retained.analysis.period = 2; retained.analysis.parameterCount = 1;
    retained.analysis.pipeCount = 2; retained.analysis.exactMinimum = true;
    for (unsigned residue : {0U, 1U}) {
        auto entry = fs::IntegerSystem::create(1,
            {integrationRow({-1}, residue ? -1 : -2), integrationRow({1}, residue ? 7 : 8)});
        auto occurrence = fs::IntegerSystem::create(2,
            {integrationRow({-1, 0}, -1), integrationRow({1, -1}, -1)});
        if (failed(entry) || failed(occurrence)) { return false; }
        primitive(fs::PrimitiveKind::Context, {}, {}, residue, &*entry);
        for (unsigned source = 0; source < 2; ++source) {
            primitive(fs::PrimitiveKind::Occurrences, source, {}, residue, &*occurrence);
            for (unsigned target = 0; target < 2; ++target) {
                auto order = fs::IntegerSystem::create(3,
                    {integrationRow({-1, 0, 0}, -1), integrationRow({0, 1, -1}, -1),
                     integrationRow({1, -1, 0}, source < target ? 0 : -1)});
                if (failed(order)) { return false; }
                primitive(fs::PrimitiveKind::Order, source, target, residue, &*order);
                for (auto from : {fs::ArithmeticEvent::Start, fs::ArithmeticEvent::Completion}) {
                    for (auto to : {fs::ArithmeticEvent::Start, fs::ArithmeticEvent::Completion}) {
                        const auto distance = source != target ? 1 :
                            from == fs::ArithmeticEvent::Completion && to == fs::ArithmeticEvent::Start ? 2 : 0;
                        auto reaches = fs::IntegerSystem::create(3,
                            {integrationRow({-1, 0, 0}, -1), integrationRow({0, 1, -1}, -1),
                             integrationRow({1, -1, 0}, -distance),
                             integrationRow({0, 0, -1}, residue ? -1 : -2),
                             integrationRow({0, 0, 1}, residue ? 7 : 8)});
                        if (failed(reaches)) { return false; }
                        retained.analysis.requiredOrder.edit()[{{source, from, {1}}, {target, to, {1}}, {residue}}] =
                            {*reaches};
                    }
                }
            }
            auto demand = fs::IntegerSystem::create(3,
                {integrationRow({-1, 0, 0}, -1), integrationRow({0, 1, -1}, -1),
                 integrationRow({1, -1, 0}, -1), integrationRow({-1, 1, 0}, 1),
                 integrationRow({0, 0, -1}, residue ? -1 : -2),
                 integrationRow({0, 0, 1}, residue ? 7 : 8)});
            if (failed(demand)) { return false; }
            retained.analysis.minimumDemands.edit()[{{source, fs::ArithmeticEvent::Completion, {1}},
                {1 - source, fs::ArithmeticEvent::Start, {1}}, {residue}}] = {*demand};
        }
    }
    program.recognition = fs::recognizeArithmetic(program.primitives, {2, 3, 2, 16});
    return program.recognition.state == fs::RecognitionState::Applicable;
}
bool integrationCommandOracle(const DirectedTrace& actual, int64_t limit, uint32_t p, uint32_t q)
{
    using Key = std::tuple<unsigned, uint32_t, int64_t>;
    struct Pair { std::optional<unsigned> set, wait; };
    std::map<Key, Pair> pairs;
    unsigned position = 0;
    auto command = [&](bool set, unsigned stage, uint32_t source, uint32_t target, int64_t coordinate) {
        if (position >= actual.size()) { return false; }
        const auto& [isSet, from, to, id] = actual[position];
        if (isSet != set || from != source || to != target || id > 5) { return false; }
        auto& pair = pairs[{stage, source, coordinate}];
        auto& side = set ? pair.set : pair.wait;
        if (side) { return false; }
        side = position++;
        return true;
    };
    for (unsigned stage = 0; stage < 2; ++stage) {
        for (int64_t i = 3; i < limit; i += 2) {
            for (auto direction : {std::make_pair(p, q), std::make_pair(q, p)}) {
                if (i > 3 && !command(false, stage, direction.second, direction.first, i - 2)) { return false; }
                if (i + 2 < limit && !command(true, stage, direction.first, direction.second, i)) { return false; }
            }
        }
    }
    if (position != actual.size()) { return false; }
    std::vector<std::vector<bool>> graph(actual.size(), std::vector<bool>(actual.size()));
    std::map<uint32_t, unsigned> previous;
    for (unsigned i = 0; i < actual.size(); ++i) {
        auto [set, source, target, id] = actual[i];
        const auto pipe = set ? source : target;
        if (auto found = previous.find(pipe); found != previous.end()) { graph[found->second][i] = true; }
        previous[pipe] = i;
    }
    for (const auto& [key, pair] : pairs) {
        if (!pair.set || !pair.wait || std::get<3>(actual[*pair.set]) != std::get<3>(actual[*pair.wait])) {
            return false;
        }
        graph[*pair.set][*pair.wait] = true; // Matching only; no textual cross-pipe edge.
    }
    for (unsigned k = 0; k < actual.size(); ++k) {
        for (unsigned i = 0; i < actual.size(); ++i) {
            for (unsigned j = 0; j < actual.size(); ++j) { graph[i][j] = graph[i][j] || (graph[i][k] && graph[k][j]); }
        }
    }
    for (unsigned i = 0; i < actual.size(); ++i) { if (graph[i][i]) { return false; } }
    for (auto first = pairs.begin(); first != pairs.end(); ++first) {
        for (auto next = std::next(first); next != pairs.end(); ++next) {
            if (std::get<1>(first->first) != std::get<1>(next->first) ||
                std::get<3>(actual[*first->second.set]) != std::get<3>(actual[*next->second.set])) { continue; }
            if (!graph[*first->second.wait][*next->second.set] &&
                !graph[*next->second.wait][*first->second.set]) { return false; }
        }
    }
    return true;
}
// Whole-invocation completion follows every modeled command. Qualify its
// original terminal position before omitting it from the flags-only graph;
// an internal drain would change reuse premises and must fail this check.
bool hasOnlyTerminalInvocationDrain(func::FuncOp function)
{
    unsigned drains = 0;
    bool valid = true;
    function.walk([&](pto::BarrierOp barrier) {
        ++drains;
        valid &= barrier.getPipe() == pto::PipeAttr::get(function.getContext(), pto::PIPE::PIPE_ALL) &&
            barrier->getBlock() == &function.front() && isa_and_nonnull<func::ReturnOp>(barrier->getNextNode());
    });
    return valid && drains == 1;
}
bool arithmeticCoordinateIntegration(MLIRContext& context)
{
    constexpr const char* ir = R"mlir(
      module attributes {pto.target_arch = "a3"} {
        func.func @coordinate_parent(%limit: index) -> index {
          %three = arith.constant 3 : index
          %two = arith.constant 2 : index
          scf.for %i = %three to %limit step %two {
            %a = arith.addi %i, %two : index
            %b = arith.addi %i, %three : index
          }
          scf.for %j = %three to %limit step %two {
            %a = arith.addi %j, %two : index
            %b = arith.addi %j, %three : index
          }
          return %three : index
        }
      }
    )mlir";
    auto module = parseSourceString<ModuleOp>(ir, &context);
    if (!module) { return false; }
    auto function = module->lookupSymbol<func::FuncOp>("coordinate_parent");
    auto original = printed(function);
    SmallVector<scf::ForOp> loops;
    function.walk([&](scf::ForOp loop) { loops.push_back(loop); });
    pto::SyncInput input;
    auto arena = std::make_shared<fs::RegionExpressions>();
    std::vector<std::unique_ptr<pto::CompoundInstanceElement>> phases;
    std::vector<fs::RegionalAnalysis> children;
    std::string error;
    const auto p = static_cast<uint32_t>(pto::PipelineType::PIPE_MTE2);
    const auto q = static_cast<uint32_t>(pto::PipelineType::PIPE_MTE1);
    for (auto loop : loops) {
        auto* first = &loop.getBody()->front();
        for (auto* op : {first, first->getNextNode()}) {
            phases.push_back(std::make_unique<pto::CompoundInstanceElement>(phases.size(),
                SmallVector<const pto::BaseMemInfo*>{}, SmallVector<const pto::BaseMemInfo*>{},
                op == first ? pto::PipelineType::PIPE_MTE2 : pto::PipelineType::PIPE_MTE1, op->getName()));
            phases.back()->elementOp = op;
        }
        fs::ArithmeticRegionalRelations mathematics;
        if (!integrationMathematics(mathematics, context, function, loop, input,
                phases[phases.size() - 2].get(), phases.back().get())) { return false; }
        // The historical default must not silently run the wider coordinate search.
        auto historical = fs::exportArithmeticRegion(mathematics, arena, true, error);
        if (failed(historical)) { return false; }
        auto unavailable = historical->prepare();
        if (failed(unavailable)) { return false; }
        fs::prepareAllocationSupport(**unavailable);
        if ((*unavailable)->regionalAllocation) { return false; }
        auto child = fs::exportArithmeticRegion(mathematics, arena, true, error, {},
            fs::ArithmeticRegionalAllocationPolicy::CoordinateCandidate);
        if (failed(child)) { llvm::errs() << "coordinate child export: " << error << '\n'; return false; }
        auto* target = first->getNextNode();
        auto singleton = arena->eq(arena->input(loop.getInductionVar()), arena->constant(3));
        auto choices = fs::qualifyTerminalCleanupCuts({{target->getBlock(), target},
            {&function.front(), function.front().getTerminator()}}, arena, singleton);
        if (!choices) { return false; }
        for (bool publication : {false, true}) {
            auto bad = child->prepare();
            if (failed(bad) || (*bad)->families.empty()) { return false; }
            auto& family = (*bad)->families.front();
            if (publication) { family.sourceChoices = choices; }
            else { family.targetChoices = choices; }
            fs::prepareAllocationSupport(**bad);
            if ((*bad)->regionalAllocation) { return false; }
        }
        for (unsigned malformed = 0; malformed < 3; ++malformed) {
            auto bad = child->prepare();
            if (failed(bad) || (*bad)->families.empty() || !scalarFreeCoordinatePlan(**bad)) { return false; }
            auto recipe = malformed < 2 ?
                integrationOneSidedRecipe((*bad)->families.front(), loop, malformed == 0) :
                fs::prefixFamilyInvocation((*bad)->familyRecipe, arena, arena->constant(0));
            if (failed(recipe)) { return false; }
            auto shapes = fs::familyMatchingShapes(**recipe);
            if (failed(shapes) || shapes->empty()) { return false; }
            if (malformed < 2) {
                const auto& shape = shapes->begin()->second;
                if (shape.coordinateCount != 2 || shape.hasSet != (malformed == 0) ||
                    shape.hasWait != (malformed == 1) || shape.hasLocal) { return false; }
            } else {
                for (const auto& [record, shape] : *shapes) {
                    if (shape.coordinateCount != 3 || !shape.hasSet || !shape.hasWait || shape.hasLocal) {
                        return false;
                    }
                }
            }
            (*bad)->familyRecipe = std::move(*recipe);
            fs::prepareAllocationSupport(**bad);
            if ((*bad)->regionalAllocation || !scalarFreeCoordinatePlan(**bad)) { return false; }
        }
        {
            auto bad = child->prepare();
            if (failed(bad) || (*bad)->families.empty() || !scalarFreeCoordinatePlan(**bad)) { return false; }
            const auto& family = (*bad)->families.front();
            fs::PreparedLogicalEndpoint concrete;
            concrete.before = family.targetCut.before; concrete.kind = fs::LogicalCommandKind::Barrier;
            concrete.record = family.id; concrete.records.push_back(family.members.front().record);
            (*bad)->endpoints.push_back(std::move(concrete));
            // An owned recipe does not authorize ignoring additional concrete
            // contributions to the same record. Preserve the full exact path.
            fs::prepareAllocationSupport(**bad);
            if ((*bad)->regionalAllocation || (*bad)->familyWork.materializations ||
                !(*bad)->preparation.empty() || (*bad)->endpoints.size() != 1) { return false; }
        }
        auto good = child->prepare();
        if (failed(good) || !scalarFreeCoordinatePlan(**good)) { return false; }
        fs::prepareAllocationSupport(**good);
        if (!(*good)->regionalAllocation || (*good)->regionalAllocation->groups.size() != 2 ||
            !scalarFreeCoordinatePlan(**good)) { return false; }
        for (const auto& group : (*good)->regionalAllocation->groups) {
            if (group.budget != 3 || group.members.size() != 1 || !group.lanes.empty()) { return false; }
            const auto& rule = group.members.front().tupleRule;
            if (!rule || rule->coordinateCount != 2 || rule->terms.size() != 1 ||
                rule->terms.front().coordinate != 1 || rule->terms.front().modulus != 3) { return false; }
        }
        children.push_back(std::move(*child));
    }
    auto sequence = fs::composeRegionalSequence(function, arena, std::move(children), false, true, {},
        fs::ArithmeticRegionalAllocationPolicy::CoordinateCandidate);
    if (!sequence.error.empty()) {
        llvm::errs() << "coordinate parent: " << sequence.error << '\n'; return false;
    }
    auto prepared = fs::prepareSequenceInsertion(sequence);
    if (failed(prepared) || !(*prepared)->allocationCertificate || !(*prepared)->regionalAllocation ||
        (*prepared)->regionalAllocation->groups.size() != 4 || printed(function) != original) { return false; }
    std::set<uint32_t> records;
    for (const auto& group : (*prepared)->regionalAllocation->groups) {
        for (const auto& member : group.members) {
            if (member.firstSource.type / 2 != member.record / 2 ||
                member.lastTarget.type / 2 != member.record / 2) { return false; }
            records.insert(member.record);
        }
    }
    if (records != std::set<uint32_t>{0, 1, 2, 3} ||
        failed(fs::insertLogicalSynchronization(function, **prepared)) ||
        failed(fs::allocatePhysicalEventIds(function, {0, 1, 2, 3, 4, 5})) || failed(verify(function))) {
        return false;
    }
    if (!hasOnlyTerminalInvocationDrain(function)) { return false; }
    for (int64_t limit = 3; limit <= 16; ++limit) {
        Values values; values[function.getArgument(0)] = limit;
        Trace trace; DirectedTrace directed;
        auto result = execute(function.front(), values, trace, &directed);
        if (failed(result) || result->size() != 1 || result->front() != 3 ||
            !integrationCommandOracle(directed, limit, p, q)) { return false; }
        if (limit == 16) {
            auto unsafe = directed;
            for (auto& command : unsafe) { std::get<3>(command) = 0; }
            if (integrationCommandOracle(unsafe, limit, p, q)) { return false; }
        }
    }
    return true;
}
} // namespace
#endif
