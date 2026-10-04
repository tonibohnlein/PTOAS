// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Stable structural JSON for coverage audits; no event ordering is inferred here.
#include "PTO/Transforms/FrontierSynch/ProgramRecognition.h"
#include "mlir/IR/AsmState.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
llvm::json::Object dumpPeriodicAnalysis(const fs::PeriodicAnalysis& result);
llvm::json::Object dumpNumericTemplateEndpoints(const fs::NumericTemplateEndpoints& result, AsmState& state);
llvm::json::Object dumpNumericTemplate(const fs::NumericTemplate& result, AsmState& state);
namespace {
std::string valueName(Value value, AsmState& state) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    value.printAsOperand(stream, state);
    return text;
}
std::string location(Operation* op) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    if (op) {
        op->getLoc().print(stream);
    }
    return text;
}
llvm::json::Array ids(ArrayRef<std::size_t> values) {
    llvm::json::Array array;
    for (auto value : values) {
        array.push_back(value);
    }
    return array;
}
llvm::json::Object attempt(StringRef route, const fs::RecognitionResult& result) {
    llvm::json::Array issues;
    for (const auto& issue : result.diagnostics) {
        issues.push_back(llvm::json::Object{{"issue", fs::recognitionName(issue.issue)},
            {"operation", issue.anchor ? issue.anchor->getName().getStringRef() : StringRef()},
            {"location", location(issue.anchor)}});
    }
    return llvm::json::Object{{"route", route}, {"state", fs::recognitionName(result.state)},
                              {"issues", std::move(issues)}};
}
}
LogicalResult dumpProgramRecognition(func::FuncOp function, const pto::SyncInput& input,
                                     const fs::ProgramRecognition& program) {
    const auto* result = &program;
    AsmState state(function);
    llvm::json::Array nodes, payloads, guards;
    for (auto [id, node] : llvm::enumerate(result->nodes)) {
        llvm::json::Array attempts, loops;
        for (auto loop : node.loops) {
            loops.push_back(valueName(loop.getInductionVar(), state));
        }
        if (node.explicitResult) {
            attempts.push_back(attempt("explicit", *node.explicitResult));
        }
        if (node.rotatingResult) {
            attempts.push_back(attempt("rotating", *node.rotatingResult));
        }
        if (node.finiteGuardedResult) {
            auto item = attempt("finite-guarded", node.finiteGuardedResult->result);
            item["entry_guards_available"] = node.finiteGuardedResult->entryGuardsAvailable;
            attempts.push_back(std::move(item));
        }
        if (node.guardedRotatingResult) {
            auto item = attempt("guarded-rotating", node.guardedRotatingResult->result);
            item["entry_guards_available"] = node.guardedRotatingResult->entryGuardsAvailable;
            item["entry_expression_count"] = node.guardedRotatingResult->entryExpressions.size();
            attempts.push_back(std::move(item));
        }
        if (node.numericTemplate) {
            auto candidate = dumpNumericTemplate(*node.numericTemplate, state);
            if (node.periodicAnalysis) {
                candidate["analysis"] = dumpPeriodicAnalysis(*node.periodicAnalysis);
            }
            if (node.logicalEndpoints) {
                candidate["logical_endpoints"] = dumpNumericTemplateEndpoints(*node.logicalEndpoints, state);
            }
            attempts.push_back(std::move(candidate));
        }
        llvm::json::Object object{{"id", id}, {"kind", fs::structureName(node.kind)},
            {"parent", node.parent ? static_cast<int64_t>(*node.parent) : -1},
            {"children", ids(node.children)}, {"payloads", ids(node.payloads)},
            {"payload_count", node.payloadCount},
            {"guard", node.guard ? static_cast<int64_t>(*node.guard) : -1},
            {"loops", std::move(loops)}, {"unsupported_context", node.unsupportedContext},
            {"location", location(node.anchor)}, {"attempts", std::move(attempts)}};
        if (auto loop = dyn_cast_or_null<scf::ForOp>(node.anchor); loop && node.kind == fs::StructureKind::Loop) {
            object["induction"] = valueName(loop.getInductionVar(), state);
            object["lower"] = valueName(loop.getLowerBound(), state);
            object["upper"] = valueName(loop.getUpperBound(), state);
            object["step"] = valueName(loop.getStep(), state);
            object["iter_args"] = loop.getNumRegionIterArgs();
        }
        if (node.region && isa<scf::IfOp>(node.region->getParentOp())) {
            object["arm"] = node.region->getRegionNumber() == 0 ? "then" : "else";
        }
        nodes.push_back(std::move(object));
    }
    for (auto [id, payload] : llvm::enumerate(result->payloads)) {
        llvm::json::Array effects;
        for (auto effectId : payload.effects) {
            const auto& effect = input.accesses().effects()[effectId];
            StringRef precision = effect.precision == pto::SyncAccessPrecision::Exact ? "exact" :
                (effect.precision == pto::SyncAccessPrecision::UpperBound ? "upper" : "unknown");
            effects.push_back(llvm::json::Object{{"id", effectId}, {"precision", precision},
                {"mode", effect.mode == pto::SyncAccessMode::Read ? "read" : "write"},
                {"space", static_cast<unsigned>(effect.memory->scope)},
                {"has_descriptor", effect.descriptorRegion.has_value()},
                {"has_access_map", !effect.regions.empty()}, {"concrete_exact_ranges", effect.exactRanges}});
        }
        payloads.push_back(llvm::json::Object{{"id", id}, {"node", payload.node},
            {"operation", payload.phase->opName.getStringRef()},
            {"location", location(payload.phase->elementOp)},
            {"pipe", static_cast<unsigned>(payload.phase->kPipeValue)}, {"effects", std::move(effects)}});
    }
    for (auto [id, guard] : llvm::enumerate(result->guards)) {
        auto branch = guard.branch;
        llvm::json::Array availability, loops;
        for (auto loop : guard.loops) {
            loops.push_back(valueName(loop.getInductionVar(), state));
        }
        for (bool available : guard.availableBeforeLoops) {
            availability.push_back(available);
        }
        guards.push_back(llvm::json::Object{{"id", id},
            {"parent", guard.parent ? static_cast<int64_t>(*guard.parent) : -1},
            {"condition", valueName(branch.getCondition(), state)}, {"then", guard.takeThen},
            {"available_before_branch", guard.availableBeforeBranch},
            {"available_before_loops", std::move(availability)}, {"loops", std::move(loops)}});
    }
    llvm::json::Object arithmetic{{"scope", "whole-function"}, {"state", "not-run"}};
    if (result->arithmetic) {
        const auto& program = *result->arithmetic;
        arithmetic = attempt("arithmetic", program.extraction);
        arithmetic["scope"] = "whole-function";
        if (program.extraction.state == fs::RecognitionState::Applicable) {
            arithmetic["state"] = fs::recognitionName(program.recognition.state);
            arithmetic["class"] = fs::recognitionName(program.recognition.arithmeticClass);
            llvm::json::Array issues;
            for (const auto& issue : program.recognition.diagnostics) {
                issues.push_back(llvm::json::Object{{"issue", fs::recognitionName(issue.issue)},
                    {"relation", issue.relation}, {"piece", issue.piece}});
            }
            arithmetic["issues"] = std::move(issues);
        }
    }
    llvm::json::Object document{{"function", function.getSymName()}, {"analysis_ready", false},
        {"nodes", std::move(nodes)}, {"payloads", std::move(payloads)}, {"guards", std::move(guards)},
        {"arithmetic", std::move(arithmetic)}};
    llvm::outs() << llvm::json::Value(std::move(document)) << "\n";
    return success();
}
