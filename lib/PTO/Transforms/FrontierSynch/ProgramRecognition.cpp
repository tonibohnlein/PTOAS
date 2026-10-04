// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Connect source structure to recognizers without flattening nested control.
#include "PTO/Transforms/FrontierSynch/ProgramRecognition.h"
#include "llvm/ADT/DenseSet.h"
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
                    (isa<scf::IfOp>(op) ? StructureKind::Conditional : StructureKind::Unsupported);
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

FailureOr<ProgramRecognition> recognizeProgram(func::FuncOp function, const SyncInput& input,
                                               const ArithmeticLimits& limits) {
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
        } else if (node.kind == StructureKind::Loop) {
            auto loop = cast<scf::ForOp>(node.anchor);
            node.rotatingResult = recognizeRotating(loop, index, input, input.accesses());
            node.guardedRotatingResult = recognizeGuardedRotating(loop, index, input, input.accesses());
        }
    }
    if (!function.isDeclaration()) {
        result.arithmetic = recognizeArithmeticProgram(function, index, input, input.accesses(), limits);
    }
    // Charged numerical inner expansion is a late compact-route candidate;
    // it never changes the original region tree or the explicit-run route.
    for (auto& node : result.nodes) {
        if (!node.unsupportedContext && node.kind == StructureKind::Loop && node.payloadCount) {
            node.numericTemplate = recognizeNumericTemplate(cast<scf::ForOp>(node.anchor), index, input);
        }
    }
    return result;
}
StringRef structureName(StructureKind kind) {
    switch (kind) {
    case StructureKind::Sequence: return "sequence";
    case StructureKind::ExplicitRun: return "explicit-run";
    case StructureKind::Loop: return "loop";
    case StructureKind::Conditional: return "conditional";
    case StructureKind::Unsupported: return "unsupported";
    default: return "invalid";
    }
}
} // namespace mlir::pto::frontiersynch
