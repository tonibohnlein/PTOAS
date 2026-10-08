// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Connect source structure to recognizers without flattening nested control.
#include "PTO/Transforms/FrontierSynch/ProgramRecognition.h"
#include "ArithmeticRows.h"
#include "PTO/Transforms/FrontierSynch/FiniteVisitRecognition.h"
#include "llvm/ADT/DenseSet.h"
#include <map>
namespace mlir::pto::frontiersynch {
namespace {
class StructureBuilder {
public:
    StructureBuilder(ProgramRecognition& result, const PhaseIndex& index, const SyncInput& input)
        : result(result), index(index), input(input) {}

    std::size_t add(StructureKind kind, Operation* anchor, Region* region,
                    std::optional<std::size_t> parent) {
        StructureNode node;
        node.kind = kind;
        node.anchor = anchor;
        node.region = region;
        node.parent = parent;
        if (parent) {
            node.loops = result.nodes[*parent].loops;
            node.guard = result.nodes[*parent].guard;
            node.unsupportedContext = result.nodes[*parent].unsupportedContext;
        }
        auto id = result.nodes.size();
        result.nodes.push_back(std::move(node));
        if (parent) {
            result.nodes[*parent].children.push_back(id);
        }
        return id;
    }

    void sequence(std::size_t id) {
        Region* region = result.nodes[id].region;
        const bool multipleBlocks = region->getBlocks().size() > 1;
        if (multipleBlocks) {
            result.nodes[id].kind = StructureKind::Unsupported;
            result.nodes[id].unsupportedContext = true;
        }
        for (Block& block : *region) {
            std::optional<std::size_t> run;
            for (Operation& op : block) {
                if (op.hasTrait<OpTrait::IsTerminator>()) {
                    continue;
                }
                if (!op.getNumRegions()) {
                    if (!run) {
                        run = add(StructureKind::ExplicitRun, &op, nullptr, id);
                    }
                    result.nodes[*run].operations.push_back(&op);
                    continue;
                }
                run.reset();
                auto kind = isa<scf::ForOp>(op) ? StructureKind::Loop :
                    (isa<scf::IfOp>(op) ? StructureKind::Conditional :
                     (isa<SectionCubeOp, SectionVectorOp>(op) ? StructureKind::Section : StructureKind::Unsupported));
                auto child = add(kind, &op, nullptr, id);
                result.nodes[child].unsupportedContext |= kind == StructureKind::Unsupported;
                // Retain payload-bearing structured anchors too; their route
                // remains subject to the existing multi-phase/control checks.
            }
        }
    }

    void control(std::size_t id) {
        Operation* op = result.nodes[id].anchor;
        for (Region& region : op->getRegions()) {
            auto child = add(StructureKind::Sequence, op, &region, id);
            if (auto loop = dyn_cast<scf::ForOp>(op)) {
                result.nodes[child].loops.push_back(loop);
            }
            if (auto branch = dyn_cast<scf::IfOp>(op)) {
                ProgramGuard guard;
                guard.parent = result.nodes[id].guard;
                guard.branch = branch;
                guard.takeThen = &region == &branch.getThenRegion();
                guard.availableBeforeBranch = index.valueAvailable(branch.getCondition(), op, Boundary::Before);
                guard.loops = result.nodes[id].loops;
                for (auto loop : guard.loops) {
                    guard.availableBeforeLoops.push_back(
                        index.valueAvailable(branch.getCondition(), loop, Boundary::Before));
                }
                result.nodes[child].guard = result.guards.size();
                result.guards.push_back(std::move(guard));
            }
        }
    }

    LogicalResult payloads() {
        DenseSet<const CompoundInstanceElement*> seen;
        for (std::size_t id = 0; id < result.nodes.size(); ++id) {
            auto& node = result.nodes[id];
            SmallVector<Operation*> anchors = node.operations;
            if (!node.region && node.kind != StructureKind::ExplicitRun && node.anchor) {
                anchors.push_back(node.anchor);
            }
            for (auto* anchor : anchors) {
                for (auto* phase : index.phasesFor(anchor)) {
                    if (!seen.insert(phase).second) {
                        return failure();
                    }
                    ProgramPayload payload;
                    payload.phase = phase;
                    payload.node = id;
                    auto effects = input.accesses().effectsFor(phase);
                    payload.effects.append(effects.begin(), effects.end());
                    node.payloads.push_back(result.payloads.size());
                    result.payloads.push_back(std::move(payload));
                }
            }
        }
        return success(seen.size() == input.instructions().size());
    }
private:
    ProgramRecognition& result;
    const PhaseIndex& index;
    const SyncInput& input;
};
} // namespace

ContractDiagnosticKind contractDiagnosticKind(const RecognitionDiagnostic& diagnostic)
{
    switch (diagnostic.issue) {
    case RecognitionIssue::TemplateExpansionLimit:
    case RecognitionIssue::ArithmeticConfiguration:
    case RecognitionIssue::ArithmeticPeriod: // Current IR producer accepts P<=2 only.
        return ContractDiagnosticKind::ProducerLimit;
    case RecognitionIssue::MultiplePhases:
    case RecognitionIssue::UnsupportedView:
    case RecognitionIssue::TemplateContext:
    case RecognitionIssue::IndexArithmetic:
    case RecognitionIssue::SlotExpression:
    case RecognitionIssue::LoopDomain:
    case RecognitionIssue::LoopCarriedState:
        return ContractDiagnosticKind::AdapterGap;
    default:
        return diagnostic.outsideClass ? ContractDiagnosticKind::CriterionViolation :
                                         ContractDiagnosticKind::UnmetObligation;
    }
}
namespace {
ContractStatus membership(const RecognitionResult& result)
{
    if (result.state == RecognitionState::Applicable) { return ContractStatus::Established; }
    // An adapter/work-limit failure proves no exclusion from the mathematical
    // class. A genuine violated criterion applies only to this candidate.
    for (const auto& diagnostic : result.diagnostics) {
        if (contractDiagnosticKind(diagnostic) == ContractDiagnosticKind::CriterionViolation) {
            return ContractStatus::Violated;
        }
    }
    return ContractStatus::Unproved;
}
ContractImplementation observed(StringRef error)
{
    return error.empty() ? ContractImplementation::Available : ContractImplementation::Unavailable;
}
void appendContract(ProgramRecognition& program, std::size_t node, ContractClass kind,
                    const RecognitionResult* recognition)
{
    ProgramContractCandidate candidate;
    candidate.kind = kind; candidate.node = node;
    if (recognition) {
        candidate.membership = membership(*recognition);
        candidate.diagnostics = recognition->diagnostics;
    } else {
        candidate.obligations.push_back({"recognizer-in-original-control-context", ContractStatus::NotEvaluated});
    }
    const auto& source = program.nodes[node];
    if (kind == ContractClass::VaryingPeriodic && source.varyingDemands) {
        candidate.demands = observed(source.varyingDemands->error);
        candidate.implementationError = source.varyingDemands->error;
    }
    if (kind == ContractClass::NumericTemplate) {
        if (source.periodicAnalysis) {
            candidate.demands = observed(source.periodicAnalysis->error);
            candidate.implementationError = source.periodicAnalysis->error;
        }
        if (source.logicalEndpoints) {
            candidate.endpointRecipes = observed(source.logicalEndpoints->logical.error);
            if (!source.logicalEndpoints->logical.error.empty()) {
                candidate.implementationError = source.logicalEndpoints->logical.error;
            }
        }
        if (source.periodicAllocation) {
            candidate.allocation = observed(source.periodicAllocation->error);
            if (!source.periodicAllocation->error.empty()) {
                candidate.implementationError = source.periodicAllocation->error;
            }
        }
    }
    program.contractAudit.push_back(std::move(candidate));
}
bool establishedChild(const StructureNode& node)
{
    const auto yes = [](const RecognitionResult& result) { return membership(result) == ContractStatus::Established; };
    return (node.explicitResult && yes(*node.explicitResult)) ||
        (node.finiteGuardedResult && yes(node.finiteGuardedResult->result)) ||
        (node.rotatingResult && yes(*node.rotatingResult)) ||
        (node.guardedRotatingResult && yes(node.guardedRotatingResult->result)) ||
        (node.boundedLifetime && yes(node.boundedLifetime->skeleton.result)) ||
        (node.varyingRotating && yes(node.varyingRotating->result)) ||
        (node.numericTemplate && yes(node.numericTemplate->result));
}
void appendComposition(ProgramRecognition& program, std::size_t id, ContractClass kind)
{
    if (id == 0 && kind == ContractClass::Sequence && program.sequenceContract) {
        program.contractAudit.push_back(*program.sequenceContract);
        return;
    }
    const auto& node = program.nodes[id];
    ProgramContractCandidate candidate;
    candidate.kind = kind; candidate.node = id; candidate.membership = ContractStatus::Unproved;
    candidate.obligations.push_back({"original-structured-occurrence-tree", node.unsupportedContext ?
        ContractStatus::Unproved : ContractStatus::Established});
    for (auto child : node.children) {
        candidate.obligations.push_back({"child-" + std::to_string(child) + "-exact-contract",
            establishedChild(program.nodes[child]) ? ContractStatus::Established : ContractStatus::Unproved});
    }
    candidate.obligations.push_back({"shared-context-exact-child-queries", ContractStatus::NotEvaluated});
    candidate.obligations.push_back({"complete-storage-boundary-selectors", ContractStatus::NotEvaluated});
    candidate.obligations.push_back({"native-first-last-and-supplied-prerequisite-coverage",
        ContractStatus::NotEvaluated});
    if (kind == ContractClass::Repetition) {
        candidate.obligations.push_back({"uniform-period-or-certified-phase-context", ContractStatus::NotEvaluated});
        candidate.obligations.push_back({"all-cross-visit-storage-and-prerequisite-coverage",
            ContractStatus::NotEvaluated});
    } else if (kind == ContractClass::FiniteOverlay) {
        candidate.obligations.push_back({"finite-overlay-endpoints-in-base-occurrence-domain",
            ContractStatus::NotEvaluated});
    }
    program.contractAudit.push_back(std::move(candidate));
}
} // namespace
SmallVector<ArithmeticContractDiagnostic> summarizeArithmeticDiagnostics(ArrayRef<ArithmeticDiagnostic> diagnostics)
{
    SmallVector<ArithmeticContractDiagnostic> result;
    std::map<std::pair<ArithmeticIssue, bool>, std::size_t> positions;
    for (const auto& diagnostic : diagnostics) {
        auto [found, inserted] = positions.try_emplace(
            std::make_pair(diagnostic.issue, diagnostic.outsideClass), result.size());
        if (inserted) {
            result.push_back({diagnostic.issue, diagnostic.outsideClass, diagnostic.relation, diagnostic.piece, 0});
        }
        // A count is bounded by the input ArrayRef's addressable element count.
        ++result[found->second].count;
    }
    return result;
}
void recordArithmeticContractAttempt(ProgramRecognition& program, const ArithmeticLimits& limits,
                                     const ArithmeticProgram& arithmetic)
{
    const auto diagnostics = summarizeArithmeticDiagnostics(arithmetic.recognition.diagnostics);
    for (auto kind : {ContractClass::Differences, ContractClass::Octagons, ContractClass::BoundedCoefficients}) {
        ProgramContractCandidate candidate;
        candidate.kind = kind; candidate.arithmeticProfile = limits;
        candidate.diagnostics = arithmetic.extraction.diagnostics;
        candidate.membership = membership(arithmetic.extraction);
        if (arithmetic.extraction.state == RecognitionState::Applicable) {
            const auto& recognition = arithmetic.recognition;
            candidate.arithmeticDiagnostics = diagnostics;
            candidate.membership = ContractStatus::Unproved;
            if (recognition.state == RecognitionState::Applicable) {
                // Backend selection may force the general integer importer for
                // existential locals even when every normalized row is a DBM
                // row. Class membership follows the rows, not that selection.
                auto rowClass = ArithmeticClass::Differences;
                for (const auto& piece : recognition.pieces) {
                    if (piece.empty) { continue; }
                    for (const auto& row : piece.rows) { rowClass = std::max(rowClass, detail::classifyRow(row)); }
                }
                const bool qualifies = kind == ContractClass::BoundedCoefficients ||
                    rowClass == ArithmeticClass::Differences ||
                    (kind == ContractClass::Octagons && rowClass == ArithmeticClass::Octagons);
                candidate.membership = qualifies ? ContractStatus::Established : ContractStatus::Violated;
                candidate.obligations.push_back({"normalized-coefficient-row-form", candidate.membership});
            } else if (llvm::any_of(recognition.diagnostics, [](const auto& diagnostic) {
                return diagnostic.outsideClass && diagnostic.issue != ArithmeticIssue::InvalidConfiguration;
            })) {
                candidate.membership = ContractStatus::Violated;
            }
        }
        program.arithmeticContracts.push_back(std::move(candidate));
    }
}
void recordSequenceEndpointAttempt(ProgramRecognition& program, StringRef error)
{
    if (!program.sequenceContract) { return; }
    auto& candidate = *program.sequenceContract;
    candidate.endpointRecipes = error.empty() ? ContractImplementation::Available : ContractImplementation::Unavailable;
    candidate.implementationError = error.str();
    refreshProgramContractAudit(program);
}
void refreshProgramContractAudit(ProgramRecognition& program)
{
    program.contractAudit.clear();
    for (auto [id, node] : llvm::enumerate(program.nodes)) {
        if (node.kind == StructureKind::ExplicitRun) {
            appendContract(program, id, ContractClass::Finite, node.explicitResult ? &*node.explicitResult : nullptr);
        } else if (node.kind == StructureKind::Conditional) {
            appendContract(program, id, ContractClass::FiniteGuarded,
                node.finiteGuardedResult ? &node.finiteGuardedResult->result : nullptr);
        } else if (node.kind == StructureKind::Sequence) {
            appendContract(program, id, ContractClass::FiniteGuarded,
                node.finiteGuardedResult ? &node.finiteGuardedResult->result : nullptr);
            appendComposition(program, id, ContractClass::Sequence);
            if (node.children.size() > 1) { appendComposition(program, id, ContractClass::FiniteOverlay); }
        } else if (node.kind == StructureKind::Loop) {
            appendContract(program, id, ContractClass::Periodic, node.rotatingResult ? &*node.rotatingResult : nullptr);
            appendContract(program, id, ContractClass::GuardedPeriodic,
                node.guardedRotatingResult ? &node.guardedRotatingResult->result : nullptr);
            appendContract(program, id, ContractClass::BoundedLifetime,
                node.boundedLifetime ? &node.boundedLifetime->skeleton.result : nullptr);
            appendContract(program, id, ContractClass::VaryingPeriodic,
                node.varyingRotating ? &node.varyingRotating->result : nullptr);
            appendContract(program, id, ContractClass::NumericTemplate,
                node.numericTemplate ? &node.numericTemplate->result : nullptr);
            appendComposition(program, id, ContractClass::Repetition);
        }
    }
    llvm::append_range(program.contractAudit, program.finiteVisitContracts);
    llvm::append_range(program.contractAudit, program.arithmeticContracts);
    if (program.arithmeticContracts.empty()) {
        for (auto kind : {ContractClass::Differences, ContractClass::Octagons, ContractClass::BoundedCoefficients}) {
            ProgramContractCandidate candidate; candidate.kind = kind;
            candidate.obligations.push_back({"fixed-profile-complete-primitive-recognition",
                ContractStatus::NotEvaluated});
            program.contractAudit.push_back(std::move(candidate));
        }
    }
}

FailureOr<ProgramRecognition> recognizeProgram(func::FuncOp function, const SyncInput& input) {
    PhaseIndex index;
    if (failed(index.build(function, input))) {
        return failure();
    }
    ProgramRecognition result;
    StructureBuilder builder(result, index, input);
    builder.add(StructureKind::Sequence, function, &function.getBody(), std::nullopt);
    // Worklist order is deterministic; parent/children preserve reference order.
    // No recursive host traversal or cloning of counted iterations is needed.
    for (std::size_t id = 0; id < result.nodes.size(); ++id) {
        if (result.nodes[id].region) {
            builder.sequence(id);
        } else if (result.nodes[id].kind != StructureKind::ExplicitRun) {
            builder.control(id);
        }
    }
    if (failed(builder.payloads())) {
        function.emitError("structural recognition did not preserve every shared payload exactly once");
        return failure();
    }
    for (auto& node : result.nodes) {
        node.payloadCount = node.payloads.size();
    }
    for (std::size_t id = result.nodes.size(); id > 0; --id) {
        const auto& node = result.nodes[id - 1];
        if (node.parent) {
            result.nodes[*node.parent].payloadCount += node.payloadCount;
        }
    }
    for (auto& node : result.nodes) {
        if (node.unsupportedContext) {
            continue;
        }
        if (node.kind == StructureKind::ExplicitRun) {
            node.explicitResult = recognizeExplicitRun(node.operations, index, input.accesses());
        } else if (node.kind == StructureKind::Sequence) {
            node.finiteGuardedResult = recognizeFiniteGuarded(*node.region, index, input.accesses());
        } else if (node.kind == StructureKind::Conditional) {
            Operation* root = node.anchor;
            node.finiteGuardedResult = recognizeFiniteGuarded(ArrayRef<Operation*>(&root, 1), index, input.accesses());
        } else if (node.kind == StructureKind::Loop) {
            auto loop = cast<scf::ForOp>(node.anchor);
            node.boundedLifetime = recognizeBoundedLifetime(loop, index, input);
            node.varyingRotating = recognizeVaryingRotating(loop, index, input);
            node.rotatingResult = recognizeRotating(loop, index, input, input.accesses());
            node.guardedRotatingResult = recognizeGuardedRotating(loop, index, input, input.accesses());
        }
    }
    // Charged numerical inner expansion is a late compact-route candidate;
    // it never changes the original region tree or the explicit-run route.
    for (auto& node : result.nodes) {
        if (!node.unsupportedContext && node.kind == StructureKind::Loop && node.payloadCount) {
            node.numericTemplate = recognizeNumericTemplate(cast<scf::ForOp>(node.anchor), index, input);
        }
    }
    for (std::size_t id = 0; id < result.nodes.size(); ++id) {
        if (result.nodes[id].kind == StructureKind::Loop) {
            auto finite = recognizeFiniteVisitLoop(function, input, result, id, index);
            result.finiteVisitContracts.push_back(std::move(finite.contract));
        }
    }
    refreshProgramContractAudit(result);
    return result;
}
StringRef contractName(ContractClass kind)
{
    switch (kind) {
    case ContractClass::Finite: return "finite-occurrences";
    case ContractClass::FiniteGuarded: return "finite-guarded-occurrences";
    case ContractClass::Periodic: return "periodic-storage";
    case ContractClass::GuardedPeriodic: return "invariant-guarded-periodic-storage";
    case ContractClass::BoundedLifetime: return "bounded-lifetime";
    case ContractClass::VaryingPeriodic: return "affine-varying-periodic-visits";
    case ContractClass::NumericTemplate: return "charged-numeric-template";
    case ContractClass::Differences: return "arithmetic-differences";
    case ContractClass::Octagons: return "arithmetic-octagons";
    case ContractClass::BoundedCoefficients: return "arithmetic-bounded-coefficients";
    case ContractClass::Sequence: return "regional-sequence";
    case ContractClass::Repetition: return "regional-repetition";
    case ContractClass::FiniteOverlay: return "finite-overlay";
    case ContractClass::FiniteVisitTypes: return "finite-visit-types";
    }
    return "invalid";
}
StringRef contractName(ContractStatus status)
{
    switch (status) {
    case ContractStatus::Established: return "established";
    case ContractStatus::Violated: return "violated";
    case ContractStatus::Unproved: return "unproved";
    case ContractStatus::NotEvaluated: return "not-evaluated";
    }
    return "invalid";
}
StringRef contractName(ContractImplementation status)
{
    switch (status) {
    case ContractImplementation::NotRequested: return "not-requested";
    case ContractImplementation::Available: return "available";
    case ContractImplementation::Unavailable: return "unavailable";
    }
    return "invalid";
}
StringRef contractName(ContractDiagnosticKind kind)
{
    switch (kind) {
    case ContractDiagnosticKind::CriterionViolation: return "candidate-criterion-violation";
    case ContractDiagnosticKind::UnmetObligation: return "unmet-obligation";
    case ContractDiagnosticKind::ProducerLimit: return "producer-limit";
    case ContractDiagnosticKind::AdapterGap: return "adapter-gap";
    }
    return "invalid";
}
StringRef structureName(StructureKind kind) {
    switch (kind) {
    case StructureKind::Sequence: return "sequence";
    case StructureKind::ExplicitRun: return "explicit-run";
    case StructureKind::Loop: return "loop";
    case StructureKind::Conditional: return "conditional";
    case StructureKind::Section: return "section";
    case StructureKind::Unsupported: return "unsupported";
    default: return "invalid";
    }
}
} // namespace mlir::pto::frontiersynch
