// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent test oracle: compare the recognized tree and borrowed identities
// against original MLIR regions, operation order, and structural ancestry.
#include "PTO/Transforms/FrontierSynch/ProgramRecognition.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>

using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
struct SourceContext {
    SmallVector<scf::ForOp> loops;
    SmallVector<Region*> branches; // Innermost first, directly from source ancestry.
    bool unsupported = false;
};

SourceContext contextAt(Region* region, func::FuncOp function)
{
    SourceContext context;
    for (; region; region = region->getParentRegion()) {
        context.unsupported |= region->getBlocks().size() > 1;
        if (region == &function.getBody()) {
            break;
        }
        Operation* owner = region->getParentOp();
        if (auto loop = dyn_cast<scf::ForOp>(owner)) {
            context.loops.push_back(loop);
        } else if (isa<scf::IfOp>(owner)) {
            context.branches.push_back(region);
        } else if (!isa<pto::SectionCubeOp, pto::SectionVectorOp>(owner)) {
            context.unsupported = true;
        }
    }
    std::reverse(context.loops.begin(), context.loops.end());
    return context;
}

fs::StructureKind controlKind(Operation* operation)
{
    if (isa<scf::ForOp>(operation)) {
        return fs::StructureKind::Loop;
    }
    if (isa<scf::IfOp>(operation)) {
        return fs::StructureKind::Conditional;
    }
    if (isa<pto::SectionCubeOp, pto::SectionVectorOp>(operation)) {
        return fs::StructureKind::Section;
    }
    return fs::StructureKind::Unsupported;
}

bool auditContracts()
{
    const SmallVector<fs::ArithmeticDiagnostic> repeatedDiagnostics{
        {fs::ArithmeticIssue::CoefficientLimit, true, 7, 3},
        {fs::ArithmeticIssue::CoefficientLimit, true, 9, 4},
        {fs::ArithmeticIssue::CoefficientLimit, false, 11, 5},
        {fs::ArithmeticIssue::InvalidConfiguration, false, 2, 1},
        {fs::ArithmeticIssue::CoefficientLimit, true, 12, 6}};
    const auto summaries = fs::summarizeArithmeticDiagnostics(repeatedDiagnostics);
    if (summaries.size() != 3 || summaries[0].count != 3 || summaries[0].relation != 7 ||
        summaries[0].piece != 3 || !summaries[0].outsideClass || summaries[1].count != 1 ||
        summaries[1].outsideClass || summaries[2].issue != fs::ArithmeticIssue::InvalidConfiguration ||
        !fs::summarizeArithmeticDiagnostics({}).empty()) { return false; }
    fs::ProgramRecognition countedProgram;
    fs::ArithmeticProgram rejectedArithmetic;
    rejectedArithmetic.recognition.state = fs::RecognitionState::NotApplicable;
    rejectedArithmetic.recognition.diagnostics = repeatedDiagnostics;
    fs::recordArithmeticContractAttempt(countedProgram, {8, 8, 2, 8}, rejectedArithmetic);
    fs::refreshProgramContractAudit(countedProgram);
    if (countedProgram.contractAudit.size() != 3 ||
        llvm::any_of(countedProgram.contractAudit, [](const auto& candidate) {
            return candidate.membership != fs::ContractStatus::Violated ||
                candidate.arithmeticDiagnostics.size() != 3 ||
                candidate.arithmeticDiagnostics[0].count != 3 || candidate.arithmeticDiagnostics[0].relation != 7;
        })) { return false; }
    fs::ProgramRecognition program;
    fs::StructureNode root; root.kind = fs::StructureKind::Sequence; root.children.push_back(1);
    fs::StructureNode child; child.kind = fs::StructureKind::Loop;
    child.numericTemplate.emplace(); // Established input contract, unavailable demand backend.
    child.periodicAnalysis.emplace(); child.periodicAnalysis->error = "synthetic backend gap";
    program.nodes.push_back(std::move(root)); program.nodes.push_back(std::move(child));
    fs::refreshProgramContractAudit(program);
    bool sawNumeric = false, sawComposition = false;
    for (const auto& candidate : program.contractAudit) {
        if (candidate.kind == fs::ContractClass::NumericTemplate && candidate.node == 1) {
            sawNumeric = candidate.membership == fs::ContractStatus::Established &&
                candidate.demands == fs::ContractImplementation::Unavailable &&
                candidate.implementationError == "synthetic backend gap";
        }
        if (candidate.kind == fs::ContractClass::Sequence && candidate.node == 0) {
            sawComposition = candidate.membership == fs::ContractStatus::Unproved &&
                llvm::any_of(candidate.obligations, [](const auto& obligation) {
                    return obligation.name == "child-1-exact-contract" &&
                        obligation.status == fs::ContractStatus::Established;
                }) && llvm::any_of(candidate.obligations, [](const auto& obligation) {
                    return obligation.name == "complete-storage-boundary-selectors" &&
                        obligation.status == fs::ContractStatus::NotEvaluated;
                });
        }
    }
    if (!sawNumeric || !sawComposition) { return false; }
    const fs::RecognitionDiagnostic limit{fs::RecognitionIssue::TemplateExpansionLimit, true, nullptr};
    const fs::RecognitionDiagnostic adapter{fs::RecognitionIssue::UnsupportedView, true, nullptr};
    if (fs::contractDiagnosticKind(limit) != fs::ContractDiagnosticKind::ProducerLimit ||
        fs::contractDiagnosticKind(adapter) != fs::ContractDiagnosticKind::AdapterGap) { return false; }
    auto& recognition = program.nodes[1].numericTemplate->result;
    recognition.state = fs::RecognitionState::NotApplicable;
    recognition.diagnostics = {limit, adapter};
    fs::refreshProgramContractAudit(program);
    for (const auto& candidate : program.contractAudit) {
        if (candidate.kind == fs::ContractClass::NumericTemplate &&
            candidate.membership != fs::ContractStatus::Unproved) { return false; }
    }
    recognition.diagnostics.push_back({fs::RecognitionIssue::CommonStride, true, nullptr});
    fs::refreshProgramContractAudit(program);
    for (const auto& candidate : program.contractAudit) {
        if (candidate.kind == fs::ContractClass::NumericTemplate &&
            candidate.membership != fs::ContractStatus::Unproved) { return false; }
    }
    // Exact normalized DBM rows with existential coordinates may still select
    // the general importer. That backend choice must not reject DBM membership.
    fs::ArithmeticProgram arithmetic;
    arithmetic.recognition.arithmeticClass = fs::ArithmeticClass::BoundedCoefficients;
    fs::NormalizedPiece piece{0, 0, false, {}}; piece.rows.push_back({{1, -1}, 0, false});
    arithmetic.recognition.pieces.push_back(std::move(piece));
    fs::recordArithmeticContractAttempt(program, {8, 8, 1, 8}, arithmetic);
    arithmetic.extraction.note(fs::RecognitionIssue::UnsupportedView, nullptr);
    fs::recordArithmeticContractAttempt(program, {8, 8, 2, 8}, arithmetic);
    fs::refreshProgramContractAudit(program);
    unsigned established = 0, unproved = 0;
    for (const auto& candidate : program.contractAudit) {
        if (!candidate.arithmeticProfile) { continue; }
        if (candidate.arithmeticProfile->period == 1 && candidate.membership == fs::ContractStatus::Established) {
            ++established;
        } else if (candidate.arithmeticProfile->period == 2 && candidate.membership == fs::ContractStatus::Unproved) {
            ++unproved;
        } else { return false; }
    }
    return established == 3 && unproved == 3;
}

class StructureVerifier {
public:
    StructureVerifier(func::FuncOp function, const pto::SyncInput& input,
                      const fs::ProgramRecognition& program)
        : function(function), input(input), program(program) {}

    LogicalResult run()
    {
        const bool invalidRoot = program.nodes.empty() || program.nodes[0].parent ||
            program.nodes[0].region != &function.getBody();
        if (invalidRoot || failed(index.build(function, input))) {
            return reject("missing function root or invalid phase index");
        }
        SmallVector<std::size_t> pending{0};
        DenseSet<std::size_t> reached;
        while (!pending.empty()) {
            auto id = pending.pop_back_val();
            const bool invalidId = id >= program.nodes.size() || !reached.insert(id).second;
            if (invalidId || failed(checkNode(id))) {
                return reject("invalid or repeated structure node");
            }
            llvm::append_range(pending, program.nodes[id].children);
        }
        const bool missingRecords = reached.size() != program.nodes.size() ||
            payloads.size() != program.payloads.size() || phases.size() != input.instructions().size() ||
            effects.size() != input.accesses().effects().size();
        if (missingRecords) {
            return reject("unreachable nodes or missing shared records");
        }
        return checkGuards();
    }

private:
    LogicalResult reject(StringRef message)
    {
        return function.emitError("program structure verification: ") << message;
    }

    bool matchesGuards(std::optional<std::size_t> guard, ArrayRef<Region*> branches) const
    {
        for (Region* region : branches) {
            if (!guard || *guard >= program.guards.size()) {
                return false;
            }
            const auto& record = program.guards[*guard];
            auto branch = dyn_cast<scf::IfOp>(region->getParentOp());
            const bool takeThen = &branch.getThenRegion() == region;
            if (record.branch != branch || record.takeThen != takeThen) {
                return false;
            }
            guard = record.parent;
        }
        return !guard;
    }

    LogicalResult checkNode(std::size_t id)
    {
        const auto& node = program.nodes[id];
        if (!node.anchor) {
            return reject("node has no source anchor");
        }
        auto context = contextAt(node.region ? node.region : node.anchor->getParentRegion(), function);
        bool unsupported = context.unsupported || (!node.region && node.kind == fs::StructureKind::Unsupported);
        if (node.loops != context.loops || node.unsupportedContext != unsupported ||
            !matchesGuards(node.guard, context.branches)) {
            return reject("node context disagrees with source ancestry");
        }
        if (unsupported && (node.explicitResult || node.rotatingResult || node.finiteGuardedResult ||
                            node.guardedRotatingResult)) {
            return reject("recognition escaped an unsupported context");
        }
        for (auto child : node.children) {
            const bool wrongParent = child >= program.nodes.size() || program.nodes[child].parent != id;
            if (wrongParent) {
                return reject("child does not identify its owning parent");
            }
        }
        if (failed(checkChildren(node))) {
            return failure();
        }
        return checkPayloads(id);
    }

    LogicalResult checkChildren(const fs::StructureNode& node)
    {
        if (node.region) {
            return checkSequence(node);
        }
        if (node.kind == fs::StructureKind::ExplicitRun) {
            const bool invalidRun = !node.children.empty() || node.operations.empty() ||
                node.anchor != node.operations.front();
            if (invalidRun) {
                return reject("explicit run has children or the wrong source anchor");
            }
            return success();
        }
        const bool invalidControl = !node.anchor->getNumRegions() || node.kind != controlKind(node.anchor) ||
            !node.operations.empty() || node.children.size() != node.anchor->getNumRegions();
        if (invalidControl) {
            return reject("control node does not preserve its source regions");
        }
        for (auto [number, child] : llvm::enumerate(node.children)) {
            const auto& body = program.nodes[child];
            Region* expectedRegion = &node.anchor->getRegion(number);
            if (body.region != expectedRegion || body.anchor != node.anchor) {
                return reject("control body or branch arms were reordered");
            }
        }
        return success();
    }

    LogicalResult checkSequence(const fs::StructureNode& node)
    {
        const auto expectedKind = node.region->getBlocks().size() > 1 ?
            fs::StructureKind::Unsupported : fs::StructureKind::Sequence;
        const bool invalidSequence = node.kind != expectedKind || !node.operations.empty() ||
            !node.payloads.empty() || node.anchor != node.region->getParentOp();
        if (invalidSequence) {
            return reject("sequence is not the original source region");
        }
        SmallVector<Operation*> actual;
        const fs::StructureNode* previous = nullptr;
        for (auto child : node.children) {
            const auto& part = program.nodes[child];
            if (part.region) {
                return reject("region appears directly inside a sequence");
            }
            if (part.kind == fs::StructureKind::ExplicitRun) {
                if (part.operations.empty()) {
                    return reject("empty explicit run");
                }
                Operation* last = nullptr;
                for (auto* operation : part.operations) {
                    const bool invalidLeaf = !operation || operation->getNumRegions() ||
                        (last && last->getNextNode() != operation);
                    if (invalidLeaf) {
                        return reject("explicit run is not contiguous source leaves");
                    }
                    last = operation;
                }
                if (previous && previous->kind == fs::StructureKind::ExplicitRun &&
                    previous->operations.back()->getNextNode() == part.operations.front()) {
                    return reject("adjacent explicit runs were not combined");
                }
                llvm::append_range(actual, part.operations);
            } else {
                actual.push_back(part.anchor);
            }
            previous = &part;
        }
        SmallVector<Operation*> expected;
        for (Block& block : *node.region) {
            for (Operation& operation : block) {
                if (!operation.hasTrait<OpTrait::IsTerminator>()) {
                    expected.push_back(&operation);
                }
            }
        }
        return actual == expected ? success() : reject("sequence order differs from original MLIR");
    }

    LogicalResult checkPayloads(std::size_t id)
    {
        const auto& node = program.nodes[id];
        SmallVector<const pto::CompoundInstanceElement*> expected;
        SmallVector<Operation*> anchors = node.operations;
        if (!node.region && node.kind != fs::StructureKind::ExplicitRun) {
            anchors.push_back(node.anchor);
        }
        for (auto* anchor : anchors) {
            llvm::append_range(expected, index.phasesFor(anchor));
        }
        const bool wrongPhaseCount = expected.size() != node.payloads.size();
        if (wrongPhaseCount) {
            return reject("node lost or gained shared phases");
        }
        for (auto [number, payloadId] : llvm::enumerate(node.payloads)) {
            const bool invalidPayload = payloadId >= program.payloads.size() || !payloads.insert(payloadId).second;
            if (invalidPayload) {
                return reject("invalid or multiply owned payload");
            }
            const auto& payload = program.payloads[payloadId];
            if (payload.node != id || payload.phase != expected[number] || !phases.insert(payload.phase).second ||
                ArrayRef<std::size_t>(payload.effects) != input.accesses().effectsFor(payload.phase)) {
                return reject("payload phase or effect identities differ from shared input");
            }
            for (auto effect : payload.effects) {
                const bool wrongOwner = effect >= input.accesses().effects().size() ||
                    !effects.insert(effect).second || input.accesses().effects()[effect].phase != payload.phase;
                if (wrongOwner) {
                    return reject("effect has an invalid phase owner");
                }
            }
        }
        std::size_t count = 0;
        for (const auto* phase : input.instructions()) {
            Operation* anchor = phase->elementOp;
            bool contains = node.region ? node.region->isAncestor(anchor->getParentRegion()) :
                (node.kind == fs::StructureKind::ExplicitRun ? llvm::is_contained(node.operations, anchor) :
                 node.anchor == anchor || node.anchor->isProperAncestor(anchor));
            count += contains;
        }
        return count == node.payloadCount ? success() : reject("subtree phase count differs from MLIR");
    }

    LogicalResult checkGuards()
    {
        DenseSet<Region*> arms;
        std::size_t expected = 0;
        function.walk([&](scf::IfOp branch) { expected += branch->getNumRegions(); });
        for (const auto& guard : program.guards) {
            auto branch = guard.branch;
            if (!branch || !function->isProperAncestor(branch)) {
                return reject("guard branch is outside the function");
            }
            auto context = contextAt(branch->getParentRegion(), function);
            Region* arm = guard.takeThen ? &branch.getThenRegion() : &branch.getElseRegion();
            const bool available = index.valueAvailable(branch.getCondition(), branch, fs::Boundary::Before);
            if (!arms.insert(arm).second || guard.loops != context.loops ||
                !matchesGuards(guard.parent, context.branches) ||
                guard.availableBeforeLoops.size() != context.loops.size() ||
                guard.availableBeforeBranch != available) {
                return reject("guard identity, ancestry, or availability differs from MLIR");
            }
            for (auto [number, loop] : llvm::enumerate(context.loops)) {
                if (guard.availableBeforeLoops[number] !=
                    index.valueAvailable(branch.getCondition(), loop, fs::Boundary::Before)) {
                    return reject("guard loop availability differs from dominance");
                }
            }
        }
        return arms.size() == expected ? success() : reject("missing source branch guards");
    }

    func::FuncOp function;
    const pto::SyncInput& input;
    const fs::ProgramRecognition& program;
    fs::PhaseIndex index;
    DenseSet<std::size_t> payloads, effects;
    DenseSet<const pto::CompoundInstanceElement*> phases;
};
} // namespace

LogicalResult verifyProgramStructure(func::FuncOp function, const pto::SyncInput& input,
                                     const fs::ProgramRecognition& program)
{
    if (!auditContracts()) { return function.emitError("independent contract audit checks failed"); }
    return StructureVerifier(function, input, program).run();
}
