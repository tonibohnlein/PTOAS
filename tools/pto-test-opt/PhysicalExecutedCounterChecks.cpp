// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent scalar execution of transformed structured counter IR.
#include "PTO/Transforms/FrontierSynch/Shared/PhysicalExecutedCounters.h"
#include "PTO/Transforms/FrontierSynch/Arithmetic/ArithmeticInsertion.h"
#include "PTO/Transforms/FrontierSynch/Arithmetic/ArithmeticHandoffAllocation.h"
#include "PTO/Transforms/FrontierSynch/Shared/PhysicalAllocation.h"
#include "PTO/Transforms/FrontierSynch/Shared/FamilyInsertion.h"
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
constexpr const char* program = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @counter(%n: index) -> index {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %three = arith.constant 3 : index
    %sum = scf.for %outer = %zero to %two step %one iter_args(%carry = %zero) -> index {
      %inner = scf.for %i = %zero to %n step %one iter_args(%acc = %carry) -> index {
        %residue = arith.remui %i, %three : index
        %active = arith.cmpi ne, %residue, %one : index
        %value = scf.if %active -> index {
          pto.logical_set [<PIPE_MTE2>, <PIPE_MTE1>] plan 0 record 0 ordinal %zero members(%outer, %i)
          scf.yield %i : index
        } else {
          scf.yield %zero : index
        }
        scf.if %active {
          pto.logical_wait [<PIPE_MTE2>, <PIPE_MTE1>] plan 0 record 0 ordinal %zero members(%outer, %i)
        }
        %next = arith.addi %acc, %value : index
        scf.yield %next : index
      }
      scf.yield %inner : index
    }
    return %sum : index
  }
}
)mlir";
using Trace = std::vector<std::pair<bool, uint64_t>>;
using DirectedTrace = std::vector<std::tuple<bool, uint32_t, uint32_t, uint64_t>>;
using Values = llvm::DenseMap<Value, uint64_t>;
FailureOr<SmallVector<uint64_t>> execute(Block& body, Values& values, Trace& trace, DirectedTrace* directed = nullptr)
{
    auto read = [&](Value value) { return values.lookup(value); };
    for (auto& operation : body) {
        auto* op = &operation;
        if (isa<scf::YieldOp, func::ReturnOp>(op)) {
            SmallVector<uint64_t> result;
            for (auto operand : op->getOperands()) { result.push_back(read(operand)); }
            return result;
        }
        if (auto loop = dyn_cast<scf::ForOp>(op)) {
            SmallVector<uint64_t> state;
            for (auto initial : loop.getInitArgs()) { state.push_back(read(initial)); }
            for (auto i = static_cast<int64_t>(read(loop.getLowerBound()));
                 i < static_cast<int64_t>(read(loop.getUpperBound()));
                 i += static_cast<int64_t>(read(loop.getStep()))) {
                values[loop.getInductionVar()] = i;
                for (auto [argument, value] : llvm::zip(loop.getRegionIterArgs(), state)) { values[argument] = value; }
                auto next = execute(*loop.getBody(), values, trace, directed);
                if (failed(next)) { return failure(); }
                state = std::move(*next);
            }
            for (auto [result, value] : llvm::zip(loop.getResults(), state)) { values[result] = value; }
            continue;
        }
        if (auto branch = dyn_cast<scf::IfOp>(op)) {
            Block* selected = read(branch.getCondition()) ? branch.thenBlock() : branch.elseBlock();
            if (!selected) { continue; }
            auto result = execute(*selected, values, trace, directed);
            if (failed(result)) { return failure(); }
            for (auto [output, value] : llvm::zip(branch.getResults(), *result)) { values[output] = value; }
            continue;
        }
        if (isa<pto::LogicalSetOp, pto::LogicalWaitOp>(op)) {
            trace.emplace_back(isa<pto::LogicalSetOp>(op), read(op->getOperands().back())); continue;
        }
        if (isa<pto::SetFlagDynOp, pto::WaitFlagDynOp>(op)) {
            if (directed) {
                auto source = op->getAttrOfType<pto::PipeAttr>("src_pipe");
                auto target = op->getAttrOfType<pto::PipeAttr>("dst_pipe");
                if (!source || !target) { return failure(); }
                directed->emplace_back(isa<pto::SetFlagDynOp>(op), static_cast<uint32_t>(source.getPipe()),
                    static_cast<uint32_t>(target.getPipe()), read(op->getOperand(0)));
            }
            trace.emplace_back(isa<pto::SetFlagDynOp>(op), read(op->getOperand(0))); continue;
        }
        if (isa<pto::BarrierOp>(op)) { continue; }
        APInt constant;
        uint64_t result = 0;
        if (matchPattern(op, m_ConstantInt(&constant))) { result = constant.getSExtValue(); }
        else if (isa<arith::AddIOp>(op)) { result = read(op->getOperand(0)) + read(op->getOperand(1)); }
        else if (isa<arith::RemUIOp>(op)) { result = read(op->getOperand(0)) % read(op->getOperand(1)); }
        else if (isa<arith::SubIOp>(op)) { result = read(op->getOperand(0)) - read(op->getOperand(1)); }
        else if (isa<arith::MulIOp>(op)) { result = read(op->getOperand(0)) * read(op->getOperand(1)); }
        else if (isa<arith::FloorDivSIOp>(op)) {
            auto a = static_cast<int64_t>(read(op->getOperand(0)));
            auto b = static_cast<int64_t>(read(op->getOperand(1)));
            result = a / b - (a % b < 0 ? 1 : 0);
        } else if (isa<arith::IndexCastOp, arith::ExtUIOp>(op)) { result = read(op->getOperand(0)); }
        else if (isa<arith::AndIOp>(op)) { result = read(op->getOperand(0)) & read(op->getOperand(1)); }
        else if (isa<arith::OrIOp>(op)) { result = read(op->getOperand(0)) | read(op->getOperand(1)); }
        else if (auto select = dyn_cast<arith::SelectOp>(op)) {
            result = read(select.getCondition()) ? read(select.getTrueValue()) : read(select.getFalseValue());
        } else if (auto comparison = dyn_cast<arith::CmpIOp>(op)) {
            auto a = read(comparison.getLhs()), b = read(comparison.getRhs());
            if (comparison.getPredicate() == arith::CmpIPredicate::eq) { result = a == b; }
            else if (comparison.getPredicate() == arith::CmpIPredicate::ne) { result = a != b; }
            else if (comparison.getPredicate() == arith::CmpIPredicate::ult) { result = a < b; }
            else if (comparison.getPredicate() == arith::CmpIPredicate::sle) {
                result = static_cast<int64_t>(a) <= static_cast<int64_t>(b);
            } else { return failure(); }
        } else { return failure(); }
        if (op->getNumResults() != 1) { return failure(); }
        values[op->getResult(0)] = result;
    }
    return failure();
}
DictionaryAttr certificate(MLIRContext* context, unsigned width)
{
    Builder b(context);
    auto family = b.getDictionaryAttr({b.getNamedAttr("record", b.getI64IntegerAttr(0)),
        b.getNamedAttr("source", b.getI64IntegerAttr(static_cast<int64_t>(pto::PIPE::PIPE_MTE2))),
        b.getNamedAttr("target", b.getI64IntegerAttr(static_cast<int64_t>(pto::PIPE::PIPE_MTE1))),
        b.getNamedAttr("width", b.getI64IntegerAttr(width))});
    return b.getDictionaryAttr({b.getNamedAttr("version", b.getI64IntegerAttr(2)),
        b.getNamedAttr("plan", b.getI64IntegerAttr(0)), b.getNamedAttr("scope", b.getStringAttr("function")),
        b.getNamedAttr("strategy", b.getStringAttr("executed-family-counters")),
        b.getNamedAttr("families", b.getArrayAttr({family})),
        b.getNamedAttr("total_budget", b.getI64IntegerAttr(width))});
}
std::string printed(func::FuncOp function)
{
    std::string result; llvm::raw_string_ostream stream(result); function.print(stream); return result;
}
bool periodThree(MLIRContext& context)
{
    constexpr const char* input = R"mlir(
      module {
        func.func @period3() -> index {
          %zero = arith.constant 0 : index
          %one = arith.constant 1 : index
          %eleven = arith.constant 11 : index
          %sum = scf.for %i = %zero to %eleven step %one iter_args(%acc = %zero) -> index {
            %source = arith.addi %i, %zero : index
            %target = arith.addi %acc, %source : index
            scf.yield %target : index
          }
          return %sum : index
        }
      }
    )mlir";
    auto module = parseSourceString<ModuleOp>(input, &context);
    if (!module) { return false; }
    auto function = module->lookupSymbol<func::FuncOp>("period3");
    scf::ForOp loop; function.walk([&](scf::ForOp found) { loop = found; });
    auto* source = &loop.getBody()->front(); auto* target = source->getNextNode();
    // Supplied certified sites intentionally bypass the bounded recognizer;
    // this tests the generic arithmetic emitter's residue interface directly.
    pto::CompoundInstanceElement a(0, {}, {}, pto::PipelineType::PIPE_MTE2, source->getName());
    pto::CompoundInstanceElement b(1, {}, {}, pto::PipelineType::PIPE_MTE1, target->getName());
    a.elementOp = source; b.elementOp = target;
    fs::ArithmeticProgram program; program.context = {function, function.getOperation()};
    program.sites.push_back({&a, {loop}, {}}); program.sites.push_back({&b, {loop}, {}});
    fs::GeneralArithmeticDemandAnalysis analysis; analysis.exactMinimum = true; analysis.period = 3;
    for (uint64_t residue : {0U, 2U}) {
        fs::ArithmeticRelationKey key{{0, fs::ArithmeticEvent::Completion, {residue}},
                                     {1, fs::ArithmeticEvent::Start, {residue}}, {}};
        auto relation = fs::IntegerSystem::create(2,
            {{{fs::BoundInteger(1), fs::BoundInteger(-1)}, fs::BoundInteger(0)},
             {{fs::BoundInteger(-1), fs::BoundInteger(1)}, fs::BoundInteger(0)},
             {{fs::BoundInteger(-1), fs::BoundInteger(0)}, fs::BoundInteger(0)},
             {{fs::BoundInteger(1), fs::BoundInteger(0)}, fs::BoundInteger((10 - residue) / 3)}});
        if (failed(relation)) { return false; }
        analysis.minimumDemands[key] = {*relation};
    }
    std::string error;
    fs::ArithmeticRecordMap records;
    const auto original = printed(function);
    auto prepared = fs::prepareGeneralArithmeticLogicalInsertion(function, program, analysis, error, records);
    auto regional = fs::prepareGeneralArithmeticRegionalInsertion(function, program, analysis, error);
    fs::ArithmeticRecordMap repeatedRecords;
    auto repeated = fs::prepareGeneralArithmeticLogicalInsertion(function, program, analysis, error, repeatedRecords);
    if (failed(prepared) || failed(regional) || failed(repeated) || records != repeatedRecords ||
        records.size() != 1 || records.begin()->first != std::make_pair(std::size_t(0), std::size_t(1)) ||
        records.begin()->second != 0 || !(*prepared)->familyRecipe || !(*repeated)->familyRecipe ||
        (*prepared)->familyProtocol || (*prepared)->familyWork.materializations ||
        (*repeated)->familyWork.materializations || !(*prepared)->completeInvocation ||
        !(*prepared)->preparation.empty() || !(*prepared)->endpoints.empty() ||
        !(*regional)->familyRecipe || (*regional)->completeInvocation || !(*regional)->endpoints.empty() ||
        !(*regional)->preparation.empty() || (*regional)->familyWork.materializations ||
        failed(fs::materializeLogicalFamilies(**regional)) || (*regional)->endpoints.size() != 2 ||
        failed(fs::materializeLogicalFamilies(**prepared)) ||
        failed(fs::materializeLogicalFamilies(**repeated)) || printed(function) != original) { return false; }
    auto identity = fs::prepareGeneralArithmeticRegionalInsertion(function, program, analysis, error);
    if (failed(identity) || !(*identity)->familyRecipe || !(*identity)->preparation.empty()) { return false; }
    (*identity)->completeInvocation = true;
    if (failed(fs::materializeLogicalFamilies(**identity)) ||
        !fs::validFamilyEndpointProtocol(**identity) || (*identity)->familyWork.materializations != 1 ||
        failed(fs::materializeLogicalFamilies(**identity)) || (*identity)->familyWork.materializations != 1) {
        return false;
    }
    if ((*prepared)->endpoints.size() != 2 || (*repeated)->endpoints.size() != 2 ||
        (*prepared)->endpoints.front().guard == (*repeated)->endpoints.front().guard) { return false; }
    for (unsigned i = 0; i < 2; ++i) {
        const auto& root = (*prepared)->endpoints[i];
        const auto& child = (*regional)->endpoints[i];
        if (root.before != child.before || root.kind != child.kind || child.records.size() != 1 ||
            root.record != child.record || root.records != child.records ||
            root.sourcePipe != child.sourcePipe || root.targetPipe != child.targetPipe ||
            root.memberCoordinates.size() != 2 || child.memberCoordinates.size() != 2) { return false; }
    }
    if (failed(fs::insertLogicalSynchronization(function, **regional))) { return false; }
    Values values; Trace actual, expected;
    for (uint64_t i = 0; i < 11; ++i) {
        if (i % 3 != 1) { expected.emplace_back(true, i); expected.emplace_back(false, i); }
    }
    auto result = execute(function.front(), values, actual);
    return succeeded(result) && result->size() == 1 && result->front() == 55 && actual == expected;
}
bool signedOverlappingSelectors(MLIRContext& context, unsigned route = 0, bool difference = false)
{
    constexpr const char* input = R"mlir(
      module {
        func.func @negative() -> index {
          %low = arith.constant -5 : index
          %high = arith.constant 6 : index
          %step = arith.constant 2 : index
          scf.for %i = %low to %high step %step {
            %source = arith.addi %i, %step : index
            %target = arith.addi %source, %step : index
          }
          return %low : index
        }
      }
    )mlir";
    auto module = parseSourceString<ModuleOp>(input, &context);
    if (!module) { return false; }
    auto function = module->lookupSymbol<func::FuncOp>("negative");
    scf::ForOp loop; function.walk([&](scf::ForOp found) { loop = found; });
    auto* source = &loop.getBody()->front(); auto* target = source->getNextNode();
    pto::CompoundInstanceElement a(0, {}, {}, pto::PipelineType::PIPE_MTE2, source->getName());
    pto::CompoundInstanceElement b(1, {}, {}, pto::PipelineType::PIPE_MTE1, target->getName());
    a.elementOp = source; b.elementOp = target;
    fs::ArithmeticProgram program; program.context = {function, function.getOperation()};
    program.sites.push_back({&a, {loop}, {}}); program.sites.push_back({&b, {loop}, {}});
    fs::GeneralArithmeticDemandAnalysis analysis; analysis.exactMinimum = true; analysis.period = 2;
    auto row = [](int64_t x, int64_t y, int64_t bound) {
        return fs::IntegerConstraint{{fs::BoundInteger(x), fs::BoundInteger(y)}, fs::BoundInteger(bound)};
    };
    // Two overlapping exact pieces select target=source+2. Original odd
    // coordinates include negatives, whose period quotient requires floor.
    auto first = fs::IntegerSystem::create(2, {row(1, -1, -1), row(-1, 1, 1), row(-1, 0, 3), row(1, 0, 0)});
    auto second = fs::IntegerSystem::create(2, {row(1, -1, -1), row(-1, 1, 1), row(-1, 0, 1), row(1, 0, 1)});
    if (failed(first) || failed(second)) { return false; }
    analysis.minimumDemands[{{0, fs::ArithmeticEvent::Completion, {1}},
        {1, fs::ArithmeticEvent::Start, {1}}, {}}] = {*first, *second};
    auto selectors = fs::buildGeneralArithmeticSelectors(analysis, {0, 1});
    if (!selectors.error.empty() || selectors.inverse.size() != 1 || selectors.inverse.front().pieces.size() < 2) {
        return false;
    }
    // Independently select the shared source -1 at target 1 from each piece.
    for (const auto& piece : selectors.inverse.front().pieces) {
        auto single = selectors.inverse.front(); single.pieces = {piece};
        auto selected = fs::evaluateGeneralArithmeticSelector(single, 2, {fs::BoundInteger(1)}, {});
        if (failed(selected) || !*selected ||
            (**selected).coordinates != std::vector<fs::BoundInteger>{fs::BoundInteger(-1)}) {
            return false;
        }
    }
    std::string error;
    fs::ArithmeticRecordMap records;
    fs::ArithmeticDemandAnalysis dbm;
    dbm.exactMinimum = true; dbm.period = 2;
    auto dbmFirst = fs::DifferenceBoundSystem::create(2, {{1, 2, fs::BoundInteger(-1)},
        {2, 1, fs::BoundInteger(1)}, {0, 1, fs::BoundInteger(3)}, {1, 0, fs::BoundInteger(0)}});
    auto dbmSecond = fs::DifferenceBoundSystem::create(2, {{1, 2, fs::BoundInteger(-1)},
        {2, 1, fs::BoundInteger(1)}, {0, 1, fs::BoundInteger(1)}, {1, 0, fs::BoundInteger(1)}});
    if (failed(dbmFirst) || failed(dbmSecond)) { return false; }
    dbm.minimumDemands[{{0, fs::ArithmeticEvent::Completion, {1}},
        {1, fs::ArithmeticEvent::Start, {1}}, {}}] = {*dbmFirst, *dbmSecond};
    auto prepared = difference ? fs::prepareArithmeticLogicalInsertion(function, program, dbm, error, records) :
        fs::prepareGeneralArithmeticLogicalInsertion(function, program, analysis, error, records);
    if (succeeded(prepared) && (!(*prepared)->familyRecipe || !(*prepared)->preparation.empty() ||
        (*prepared)->familyWork.materializations)) { return false; }
    if (failed(prepared)) { return false; }
    if (route == 2) {
        (*prepared)->allocationCertificate = certificate(&context, 2);
        if (failed(fs::planPhysicalFamilyOutput(function, **prepared, {1, 3}))) { return false; }
    }
    if (failed(fs::insertLogicalSynchronization(function, **prepared))) { return false; }
    if (route == 1 && failed(fs::allocateExecutedFamilyCounters(function, certificate(&context, 2), {1, 3}))) {
        return false;
    }
    Values values; Trace actual, expected;
    uint64_t published = 0, consumed = 0;
    for (int64_t i = -5; i < 6; i += 2) {
        if (i <= 3) { expected.emplace_back(true, route ? (published++ % 2 ? 3 : 1) : static_cast<uint64_t>(i)); }
        if (i >= -3) {
            expected.emplace_back(false, route ? (consumed++ % 2 ? 3 : 1) : static_cast<uint64_t>(i - 2));
        }
    }
    auto result = execute(function.front(), values, actual);
    return succeeded(result) && result->size() == 1 && result->front() == static_cast<uint64_t>(-5) &&
        actual == expected;
}
FailureOr<std::unique_ptr<fs::PreparedLogicalPlan>> simultaneousRecordPlan(
    const fs::ArithmeticProgram& program, scf::ForOp loop, fs::ArithmeticRecordMap& records)
{
    // Two coactive targets on one pipe violate exact-F* functionality. This
    // fixture supplies common endpoint recipes directly, without exclusion
    // evidence, to check simultaneous command lowering and counter ownership.
    auto plan = std::make_unique<fs::PreparedLogicalPlan>(0);
    plan->completeInvocation = plan->groupedFamilies = plan->independentPieces = true;
    auto arena = std::make_shared<fs::RegionExpressions>();
    auto induction = arena->input(loop.getInductionVar());
    auto active = arena->lt(induction, arena->constant(6));
    SmallVector<fs::ContextualEndpointFamily> endpoints;
    const auto source = static_cast<uint32_t>(program.sites[0].phase->kPipeValue);
    for (uint32_t record = 0; record < 2; ++record) {
        const auto targetSite = 2 * record + 1;
        const auto target = static_cast<uint32_t>(program.sites[targetSite].phase->kPipeValue);
        auto* publication = program.sites[0].phase->elementOp->getNextNode();
        auto* acquisition = program.sites[targetSite].phase->elementOp;
        fs::EndpointFamily family;
        family.id = record; family.sourcePipe = source; family.targetPipe = target;
        family.sourceCut = {publication->getBlock(), publication};
        family.targetCut = {acquisition->getBlock(), acquisition};
        family.members.push_back({record, 0, targetSite, {}, {}});
        plan->families.push_back(std::move(family));
        records[{0, targetSite}] = record;
        for (bool forward : {true, false}) {
            auto* cut = forward ? publication : acquisition;
            fs::ContextualEndpointFamily endpoint;
            endpoint.piece.namespaceId = 0;
            endpoint.piece.sourcePipe = source; endpoint.piece.targetPipe = target;
            endpoint.piece.kind = forward ? fs::EndpointKind::Set : fs::EndpointKind::Wait;
            endpoint.piece.cut = {cut->getBlock(), cut};
            endpoint.piece.order = endpoints.size();
            endpoint.piece.members.push_back({record, {}});
            endpoint.activity = active; endpoint.ordinal = arena->constant(0);
            endpoint.matchingCoordinates.push_back(induction);
            endpoints.push_back(std::move(endpoint));
        }
    }
    auto recipe = fs::buildContextualFamilyEndpointRecipe(arena, endpoints);
    if (failed(recipe)) { return failure(); }
    plan->familyRecipe = std::move(*recipe); plan->nestedIdentities = true;
    return plan;
}
bool multipleRootRecords(MLIRContext& context, bool combined, uint64_t& arithmetic,
    bool simultaneous = false, bool saved = false, bool difference = false)
{
    auto reject = [&](StringRef stage) {
        llvm::errs() << "multiple records stage=" << stage << " simultaneous=" << simultaneous
                     << " combined=" << combined << " saved=" << saved << " difference=" << difference << '\n';
        return false;
    };
    constexpr const char* input = R"mlir(
      module {
        func.func @two_records(%n: index) -> index {
          %zero = arith.constant 0 : index
          %one = arith.constant 1 : index
          %six = arith.constant 6 : index
          scf.for %i = %zero to %n step %one {
            %source0 = arith.addi %i, %zero : index
            %target0 = arith.addi %source0, %one : index
            %source1 = arith.addi %i, %one : index
            %target1 = arith.addi %source1, %one : index
          }
          return %zero : index
        }
      }
    )mlir";
    auto module = parseSourceString<ModuleOp>(input, &context);
    if (!module) { return false; }
    auto function = module->lookupSymbol<func::FuncOp>("two_records");
    scf::ForOp loop; function.walk([&](scf::ForOp found) { loop = found; });
    SmallVector<Operation*> sites;
    for (auto& op : *loop.getBody()) {
        if (!isa<scf::YieldOp>(op)) { sites.push_back(&op); }
    }
    if (sites.size() != 4) { return false; }
    std::vector<std::unique_ptr<pto::CompoundInstanceElement>> phases;
    fs::ArithmeticProgram program; program.context = {function, function.getOperation()};
    for (unsigned i = 0; i < sites.size(); ++i) {
        phases.push_back(std::make_unique<pto::CompoundInstanceElement>(i,
            SmallVector<const pto::BaseMemInfo*>{}, SmallVector<const pto::BaseMemInfo*>{},
            i % 2 ? pto::PipelineType::PIPE_MTE1 : pto::PipelineType::PIPE_MTE2, sites[i]->getName()));
        phases.back()->elementOp = sites[i];
        program.sites.push_back({phases.back().get(), {loop}, {}});
    }
    fs::GeneralArithmeticDemandAnalysis analysis; analysis.exactMinimum = true; analysis.period = 1;
    auto relation = fs::IntegerSystem::create(2,
        {{{fs::BoundInteger(1), fs::BoundInteger(-1)}, fs::BoundInteger(0)},
         {{fs::BoundInteger(-1), fs::BoundInteger(1)}, fs::BoundInteger(0)},
         {{fs::BoundInteger(-1), fs::BoundInteger(0)}, fs::BoundInteger(0)},
         {{fs::BoundInteger(1), fs::BoundInteger(0)}, fs::BoundInteger(5)}});
    if (failed(relation)) { return false; }
    for (unsigned source : {0U, 2U}) {
        analysis.minimumDemands[{{source, fs::ArithmeticEvent::Completion, {0}},
            {source + 1, fs::ArithmeticEvent::Start, {0}}, {}}] = {*relation};
    }
    std::string error;
    fs::ArithmeticRecordMap records;
    fs::ArithmeticDemandAnalysis dbm; dbm.exactMinimum = true; dbm.period = 1;
    auto dbmRelation = fs::DifferenceBoundSystem::create(2, {{1, 2, fs::BoundInteger(0)},
        {2, 1, fs::BoundInteger(0)}, {0, 1, fs::BoundInteger(0)}, {1, 0, fs::BoundInteger(5)}});
    if (failed(dbmRelation)) { return false; }
    for (unsigned source : {0U, 2U}) {
        dbm.minimumDemands[{{source, fs::ArithmeticEvent::Completion, {0}},
            {source + 1, fs::ArithmeticEvent::Start, {0}}, {}}] = {*dbmRelation};
    }
    auto prepared = simultaneous ? simultaneousRecordPlan(program, loop, records) :
        difference ? fs::prepareArithmeticLogicalInsertion(function, program, dbm, error, records) :
                     fs::prepareGeneralArithmeticLogicalInsertion(function, program, analysis, error, records);
    if (failed(prepared) || records.size() != 2 || !(*prepared)->familyRecipe ||
        (*prepared)->familyWork.materializations || !(*prepared)->preparation.empty()) { return reject("recipe"); }
    // Supplied width-two certificates exercise original-record counter ownership;
    // this lowering fixture is not an allocation theorem for arbitrary protocols.
    Builder builder(&context);
    auto base = certificate(&context, 2);
    auto family0 = cast<DictionaryAttr>(base.getAs<ArrayAttr>("families")[0]);
    NamedAttrList family1(family0);
    family1.set("record", builder.getI64IntegerAttr(1));
    NamedAttrList allocation(base);
    allocation.set("families", builder.getArrayAttr({family0, family1.getDictionary(&context)}));
    allocation.set("total_budget", builder.getI64IntegerAttr(4));
    auto evidence = allocation.getDictionary(&context);
    OwningOpRef<ModuleOp> replay;
    if (saved) {
        (*prepared)->allocationCertificate = evidence;
        (*prepared)->deferredOutput = true;
        if (failed(fs::insertLogicalSynchronization(function, **prepared))) { return false; }
        unsigned markers = 0;
        function.walk([&](pto::LogicalFamilyOp) { ++markers; });
        if (!markers || (*prepared)->familyWork.materializations) { return false; }
        std::string serialized;
        llvm::raw_string_ostream stream(serialized); module->print(stream); stream.flush();
        (*prepared).reset(); // The parsed artifact cannot borrow this producer.
        replay = parseSourceString<ModuleOp>(serialized, &context);
        if (!replay) { return false; }
        function = replay->lookupSymbol<func::FuncOp>("two_records");
        if (!function || failed(fs::allocatePhysicalEventIds(function, {0, 1, 2, 3}))) { return false; }
        bool unresolved = false;
        function.walk([&](Operation* op) {
            unresolved |= isa<pto::LogicalFamilyOp, pto::LogicalSetOp, pto::LogicalWaitOp>(op);
        });
        if (unresolved) { return false; }
    } else if (combined) {
        (*prepared)->allocationCertificate = evidence;
        if (failed(fs::planPhysicalFamilyOutput(function, **prepared, {0, 1, 2, 3})) ||
            !(*prepared)->executedOutput || !(*prepared)->preparation.empty() ||
            fs::executedFamilyCounterCount(**prepared) != 4 ||
            failed(fs::materializeLogicalFamilies(**prepared))) { return false; }
        for (const auto& endpoint : (*prepared)->endpoints) {
            if (endpoint.identity ||
                llvm::any_of(endpoint.memberCoordinates, [](Value value) { return bool(value); })) {
                return false;
            }
        }
        if (failed(fs::insertLogicalSynchronization(function, **prepared))) { return false; }
    } else {
        if (failed(fs::insertLogicalSynchronization(function, **prepared))) { return reject("logical insertion"); }
        std::map<int64_t, unsigned> commands;
        bool valid = true;
        function.walk([&](Operation* op) {
            if (!isa<pto::LogicalSetOp, pto::LogicalWaitOp>(op)) { return; }
            APInt member;
            if (op->getAttrOfType<IntegerAttr>("record_id").getInt() != 0 || op->getNumOperands() < 2 ||
                !matchPattern(op->getOperand(1), m_ConstantInt(&member))) { valid = false; return; }
            ++commands[member.getSExtValue()];
        });
        if (!valid || commands != std::map<int64_t, unsigned>{{0, 2}, {1, 2}}) {
            llvm::errs() << "command labels valid=" << valid;
            for (auto [member, count] : commands) { llvm::errs() << " " << member << ':' << count; }
            llvm::errs() << '\n'; return reject("command inventory");
        }
        for (unsigned corruption = 0; corruption < 5; ++corruption) {
            OwningOpRef<func::FuncOp> copy(function.clone());
            Operation* first = nullptr;
            copy->walk([&](pto::LogicalSetOp op) {
                if (first) { return; }
                if (corruption == 3) {
                    APInt ordinal, member;
                    if (op->getNumOperands() < 2 || !matchPattern(op->getOperand(0), m_ConstantInt(&ordinal)) ||
                        !matchPattern(op->getOperand(1), m_ConstantInt(&member)) || ordinal == member) { return; }
                }
                first = op;
            });
            if (!first) { return reject("corruption target"); }
            if (corruption == 0) { first->setAttr("record_id", builder.getI64IntegerAttr(1)); }
            if (corruption == 1) { first->setAttr("pto.endpoint_piece", builder.getI64IntegerAttr(99)); }
            if (corruption == 2) { first->setOperand(1, first->getOperand(2)); }
            if (corruption == 3) { first->setOperand(1, first->getOperand(0));
                first->setOperand(0, first->getOperand(2)); }
            if (corruption == 4) {
                first->setAttr("plan_id", builder.getIntegerAttr(builder.getIntegerType(128), APInt(128, 0)));
            }
            const auto before = printed(*copy);
            ScopedDiagnosticHandler diagnostics(&context, [](Diagnostic&) { return success(); });
            if (succeeded(fs::allocateExecutedFamilyCounters(*copy, evidence, {0, 1, 2, 3})) ||
                printed(*copy) != before) {
                llvm::errs() << "corruption=" << corruption << '\n'; return reject("atomic refusal");
            }
        }
        if (failed(fs::allocateExecutedFamilyCounters(function, evidence, {0, 1, 2, 3}))) {
            return reject("counter allocation");
        }
    }
    arithmetic = 0;
    function.walk([&](Operation* op) { arithmetic += op->getName().getDialectNamespace() == "arith"; });
    for (uint64_t bound : {0U, 1U, 2U, 5U, 6U}) {
        Values values; values[function.getArgument(0)] = bound; Trace actual, expected;
        for (uint64_t i = 0; i < bound; ++i) {
            expected.emplace_back(true, i % 2);
            if (simultaneous) { expected.emplace_back(true, 2 + i % 2); }
            expected.emplace_back(false, i % 2);
            if (!simultaneous) { expected.emplace_back(true, 2 + i % 2); }
            expected.emplace_back(false, 2 + i % 2);
        }
        auto result = execute(function.front(), values, actual);
        if (failed(result) || result->size() != 1 || result->front() != 0 || actual != expected) {
            llvm::errs() << "bound=" << bound << " commands=" << actual.size() << '/' << expected.size() << '\n';
            return reject("independent trace");
        }
    }
    return true;
}
bool check(MLIRContext& context)
{
    auto module = parseSourceString<ModuleOp>(program, &context);
    if (!module) { return false; }
    auto function = module->lookupSymbol<func::FuncOp>("counter");
    const auto original = printed(function);
    for (bool metadata : {false, true}) {
        OwningOpRef<func::FuncOp> copy(function.clone());
        OpBuilder builder(copy->front().getTerminator());
        if (metadata) { (*copy)->setAttr("pto.deferred_family", builder.getDictionaryAttr({})); }
        else {
            OperationState state(copy->getLoc(), pto::LogicalFamilyOp::getOperationName());
            state.addAttribute("plan_id", builder.getI64IntegerAttr(0));
            state.addAttribute("cut_id", builder.getI64IntegerAttr(0));
            state.addAttribute("axis_ids", builder.getDenseI64ArrayAttr({}));
            state.addAttribute("pieces", builder.getArrayAttr({builder.getDictionaryAttr({})}));
            builder.create(state);
        }
        const auto before = printed(*copy);
        ScopedDiagnosticHandler diagnostics(&context, [](Diagnostic&) { return success(); });
        if (succeeded(fs::allocateExecutedFamilyCounters(*copy, certificate(&context, 2), {1, 3})) ||
            printed(*copy) != before) { return false; }
    }
    {
        ScopedDiagnosticHandler diagnostics(&context, [](Diagnostic&) { return success(); });
        if (succeeded(fs::allocateExecutedFamilyCounters(function, certificate(&context, 2), {3})) ||
            printed(function) != original) { return false; }
    }
    if (failed(fs::allocateExecutedFamilyCounters(function, certificate(&context, 2), {1, 3}))) { return false; }
    for (uint64_t trips = 0; trips <= 12; ++trips) {
        Values values; values[function.getArgument(0)] = trips; Trace actual, expected;
        uint64_t sum = 0, count = 0;
        for (unsigned outer = 0; outer < 2; ++outer) {
            for (uint64_t i = 0; i < trips; ++i) {
                if (i % 3 == 1) { continue; }
                auto id = count++ % 2 ? 3 : 1;
                expected.emplace_back(true, id); expected.emplace_back(false, id); sum += i;
            }
        }
        auto result = execute(function.front(), values, actual);
        if (failed(result) || result->size() != 1 || result->front() != sum || actual != expected) { return false; }
    }
    return true;
}
bool coordinateModulo(MLIRContext& context)
{
    constexpr const char* input = R"mlir(
      module attributes {pto.target_arch = "a3"} {
        func.func @delayed() -> index {
          %zero = arith.constant 0 : index
          %three = arith.constant 3 : index
          %two = arith.constant 2 : index
          %sixteen = arith.constant 16 : index
          %sum = scf.for %i = %three to %sixteen step %two iter_args(%acc = %zero) -> index {
            %source = arith.addi %i, %zero : index
            %target = arith.addi %acc, %source : index
            scf.yield %target : index
          }
          return %sum : index
        }
      }
    )mlir";
    auto module = parseSourceString<ModuleOp>(input, &context);
    if (!module) { return false; }
    auto function = module->lookupSymbol<func::FuncOp>("delayed");
    scf::ForOp loop; function.walk([&](scf::ForOp found) { loop = found; });
    auto* source = &loop.getBody()->front(); auto* target = source->getNextNode();
    pto::CompoundInstanceElement a(0, {}, {}, pto::PipelineType::PIPE_MTE2, source->getName());
    pto::CompoundInstanceElement b(1, {}, {}, pto::PipelineType::PIPE_MTE1, target->getName());
    a.elementOp = source; b.elementOp = target;
    fs::ArithmeticProgram program; program.context = {function, function.getOperation()};
    program.sites.push_back({&a, {loop}, {}}); program.sites.push_back({&b, {loop}, {}});
    fs::GeneralArithmeticDemandAnalysis analysis; analysis.exactMinimum = true; analysis.period = 2;
    auto row = [](int64_t x, int64_t y, int64_t bound) {
        return fs::IntegerConstraint{{fs::BoundInteger(x), fs::BoundInteger(y)}, fs::BoundInteger(bound)};
    };
    // Sources 3,5,...,13 match consumers 5,7,...,15. A later source can reuse
    // only after that consumer, so same-iteration textual SET/WAIT is insufficient.
    auto demands = fs::IntegerSystem::create(2, {row(1, -1, -1), row(-1, 1, 1),
        row(-1, 0, -1), row(1, 0, 6)});
    auto required = fs::IntegerSystem::create(2, {row(1, -1, -1)});
    if (failed(demands) || failed(required)) { return false; }
    analysis.minimumDemands[{{0, fs::ArithmeticEvent::Completion, {1}},
        {1, fs::ArithmeticEvent::Start, {1}}, {}}] = {*demands};
    analysis.requiredOrder[{{1, fs::ArithmeticEvent::Completion, {1}},
        {0, fs::ArithmeticEvent::Start, {1}}, {}}] = {*required};
    const auto p = static_cast<uint32_t>(pto::PIPE::PIPE_MTE2), q = static_cast<uint32_t>(pto::PIPE::PIPE_MTE1);
    auto proof = fs::buildArithmeticCoordinateHandoffAllocation(analysis, {p, q});
    if (!proof.error.empty() || proof.families.size() != 1 || proof.families[0].lifetime.width != 3 ||
        proof.families[0].sourceCoordinate != std::optional<unsigned>(0)) { return false; }
    std::string error;
    auto prepared = fs::prepareGeneralArithmeticRegionalInsertion(function, program, analysis, error);
    if (failed(prepared)) { return false; }
    // Serialize the proved assignment into the ordinary regional consumer.
    Builder builder(&context);
    auto tuple = builder.getDictionaryAttr({builder.getNamedAttr("coordinate_count", builder.getI64IntegerAttr(2)),
        builder.getNamedAttr("base", builder.getI64IntegerAttr(0)),
        builder.getNamedAttr("terms", builder.getArrayAttr({builder.getDenseI64ArrayAttr({1, 1, 0, 3, 1})}))});
    auto group = builder.getDictionaryAttr({builder.getNamedAttr("source", builder.getI64IntegerAttr(p)),
        builder.getNamedAttr("target", builder.getI64IntegerAttr(q)),
        builder.getNamedAttr("budget", builder.getI64IntegerAttr(3)),
        builder.getNamedAttr("records", builder.getDenseI64ArrayAttr({0})),
        builder.getNamedAttr("strides", builder.getDenseI64ArrayAttr({0})),
        builder.getNamedAttr("phases", builder.getDenseI64ArrayAttr({0})),
        builder.getNamedAttr("conflicts", builder.getDenseI64ArrayAttr({})),
        builder.getNamedAttr("tuple_rules", builder.getArrayAttr({tuple}))});
    (*prepared)->allocationCertificate = builder.getDictionaryAttr({
        builder.getNamedAttr("version", builder.getI64IntegerAttr(2)),
        builder.getNamedAttr("kind", builder.getStringAttr("finite")),
        builder.getNamedAttr("plan", builder.getI64IntegerAttr(0)),
        builder.getNamedAttr("strategy", builder.getStringAttr("regional-palettes")),
        builder.getNamedAttr("macro_reservations", builder.getUnitAttr()),
        builder.getNamedAttr("groups", builder.getArrayAttr({group}))});
    if (failed(fs::insertLogicalSynchronization(function, **prepared)) ||
        failed(fs::allocatePhysicalEventIds(function, {0, 2, 5}))) { return false; }
    Values values; Trace actual, expected;
    const uint64_t palette[] = {0, 2, 5};
    for (uint64_t i = 3; i < 16; i += 2) {
        if (i <= 13) { expected.emplace_back(true, palette[i % 3]); }
        if (i >= 5) { expected.emplace_back(false, palette[(i - 2) % 3]); }
    }
    auto result = execute(function.front(), values, actual);
    // Independently check that every same-ID reuse waits for the previous
    // consumer iteration; a SET followed textually by WAIT does not suffice.
    for (uint64_t first = 3; first <= 13; first += 2) {
        for (uint64_t next = first + 2; next <= 13; next += 2) {
            if (first % 3 == next % 3 && first + 2 >= next) { return false; }
        }
    }
    return succeeded(result) && actual == expected;
}
} // namespace
#include "ArithmeticCoordinateIntegrationChecks.h"
#include "ArithmeticCoordinateAccessChecks.h"
#include "ExecutedFamilyOutputChecks.h"
#include "DifferenceArithmeticRecipeChecks.h"
int runPhysicalExecutedCounterChecks()
{
    MLIRContext context;
    context.loadDialect<pto::PTODialect, func::FuncDialect, arith::ArithDialect, scf::SCFDialect>();
    uint64_t splitArithmetic = 0, combinedArithmetic = 0, simultaneousArithmetic = 0, savedArithmetic = 0;
    auto checked = [](bool passed, StringRef name) {
        if (!passed) { llvm::errs() << "executed-counter subcheck failed: " << name << "\n"; }
        return passed;
    };
    if (!checked(differenceArithmeticRecipeChecks(context), "differenceArithmeticRecipeChecks") ||
        !checked(differenceCertifiedAllocation(context, 0), "differenceCertifiedAllocation") ||
        !checked(differenceCertifiedAllocation(context, 1), "differenceCertifiedAllocation") ||
        !checked(differenceCertifiedAllocation(context, 2), "differenceCertifiedAllocation") ||
        !checked(signedOverlappingSelectors(context, 0, true), "signedOverlappingSelectors") ||
        !checked(signedOverlappingSelectors(context, 1, true), "signedOverlappingSelectors") ||
        !checked(signedOverlappingSelectors(context, 2, true), "signedOverlappingSelectors") ||
        !checked(multipleRootRecords(context, false, splitArithmetic, false, false, true),
            "multipleRootRecords difference split") ||
        !checked(multipleRootRecords(context, true, combinedArithmetic, false, false, true),
            "multipleRootRecords difference combined") ||
        combinedArithmetic >= splitArithmetic ||
        !checked(multipleRootRecords(context, false, savedArithmetic, false, true, true),
            "multipleRootRecords difference saved") ||
        savedArithmetic != combinedArithmetic ||
        !checked(check(context), "check") ||
        !checked(periodThree(context), "periodThree") ||
        !checked(signedOverlappingSelectors(context), "signedOverlappingSelectors") ||
        !checked(signedOverlappingSelectors(context, 1), "signedOverlappingSelectors") ||
        !checked(signedOverlappingSelectors(context, 2), "signedOverlappingSelectors") ||
        !checked(multipleRootRecords(context, false, splitArithmetic), "multipleRootRecords general split") ||
        !checked(multipleRootRecords(context, true, combinedArithmetic),
            "multipleRootRecords general combined") || combinedArithmetic >= splitArithmetic ||
        !checked(multipleRootRecords(context, false, savedArithmetic, false, true),
            "multipleRootRecords general saved") || savedArithmetic != combinedArithmetic ||
        !checked(multipleRootRecords(context, false, simultaneousArithmetic, true),
            "multipleRootRecords simultaneous split") ||
        !checked(multipleRootRecords(context, true, simultaneousArithmetic, true),
            "multipleRootRecords simultaneous combined") ||
        !checked(structuredExecutedOutput(context, false), "structuredExecutedOutput") ||
        !checked(structuredExecutedOutput(context, true), "structuredExecutedOutput") ||
        !checked(coordinateModulo(context), "coordinateModulo") ||
        !checked(arithmeticCoordinateIntegration(context), "arithmeticCoordinateIntegration") ||
        !checked(arithmeticCoordinateAccessIntegration(context), "arithmeticCoordinateAccessIntegration")) {
        llvm::errs() << "executed arithmetic counter checks failed\n"; return 1;
    }
    llvm::outs() << "executed arithmetic counters: guards, nested loops, results, zero trips and rollback passed\n";
    return 0;
}
