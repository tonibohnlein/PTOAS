// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
// Native integration tests: real PTO input goes through the shared SyncInput,
// physical importer, OriginalLifetimes and ProgramAnalysis. No test attribute
// declares write coverage and no fixture mutates the imported effect records.
#include "PTO/Transforms/FrontierSynch/ProgramAnalysis.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <set>

namespace {
namespace fs = mlir::pto::frontiersynch;
using Node = fs::FactoredUseNode;
using Set = std::set<std::size_t>;

struct Interpretation {
    const fs::FactoredUseResult& uses;
    std::map<fs::FactoredGuardIdentity, bool> values;
    bool valid = true;
    bool condition(std::size_t id)
    {
        const auto& node = uses.nodes().at(id);
        if (node.kind == Node::Kind::Empty) {
            return false;
        }
        if (node.kind == Node::Kind::True) {
            return true;
        }
        if (node.kind == Node::Kind::Test) {
            const auto found = values.find(node.guard);
            if (found != values.end()) {
                return found->second;
            }
        } else if (node.kind == Node::Kind::Choose) {
            return condition(condition(node.condition) ? node.left : node.right);
        }
        valid = false;
        return false;
    }
    Set origins(std::size_t id)
    {
        const auto& node = uses.nodes().at(id);
        switch (node.kind) {
            case Node::Kind::Empty:
                return {};
            case Node::Kind::Incoming:
                return {fs::NoControlId};
            case Node::Kind::Access:
                return {node.operation};
            case Node::Kind::Choose:
                return origins(condition(node.condition) ? node.left : node.right);
            case Node::Kind::Both: {
                auto left = origins(node.left);
                const auto right = origins(node.right);
                left.insert(right.begin(), right.end());
                return left;
            }
            default:
                valid = false;
                return {};
        }
    }
};

std::vector<std::size_t> named(const fs::OriginalStructure& original, llvm::StringRef name)
{
    std::vector<std::size_t> result;
    for (std::size_t id = 0; id < original.operations.size(); ++id) {
        const auto* phase = original.operations[id].instruction;
        if (phase && phase->elementOp->getName().getStringRef() == name) {
            result.push_back(id);
        }
    }
    return result;
}
std::size_t commonCell(const fs::OriginalStructure& original, std::size_t writer, std::size_t reader)
{
    for (const auto& write : original.operations[writer].accesses) {
        if (!write.write) {
            continue;
        }
        for (const auto& read : original.operations[reader].accesses) {
            if (read.read && read.cell == write.cell && !original.cells[read.cell].unknownRange) {
                return read.cell;
            }
        }
    }
    return fs::NoControlId;
}

bool checkImportedTyped(mlir::MLIRContext& context, bool twoPhases = false)
{
    auto module = mlir::parseSourceString<mlir::ModuleOp>(
        R"mlir(
!tile = !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16, v_row=16, v_col=16,
    blayout=row_major, slayout=none_box, fractal=512, pad=0>
module attributes {pto.target_arch = "a3"} {
  func.func @typed_import() {
    %addr = arith.constant 0 : i64
    %idx = arith.constant 0 : index
    %zero = arith.constant 0 : i32
    %tile = pto.alloc_tile addr = %addr : !tile
    %value = pto.tgetval ins(%tile, %idx : !tile, index) outs : i32
    %condition = arith.cmpi sgt, %value, %zero : i32
    scf.if %condition { scf.yield }
    return
  }
}
)mlir",
        &context);
    if (!module || mlir::failed(mlir::verify(*module))) {
        return false;
    }
    auto function = *module->getOps<mlir::func::FuncOp>().begin();
    mlir::pto::SyncInput input;
    fs::OriginalStructure original;
    if (mlir::failed(input.build(function)) || mlir::failed(fs::importOriginalStructure(function, input, original))) {
        return false;
    }
    std::unique_ptr<mlir::pto::CompoundInstanceElement> additionalPhase;
    if (twoPhases) {
        if (original.operations.size() != 1) {
            return false;
        }
        const auto* first = original.operations[0].instruction;
        additionalPhase = std::make_unique<mlir::pto::CompoundInstanceElement>(
            1, first->defVec, first->useVec, mlir::pto::PipelineType::PIPE_V, first->elementOp->getName());
        additionalPhase->elementOp = first->elementOp;
        auto second = original.operations[0];
        second.instruction = additionalPhase.get();
        second.beforeExecutable = false;
        second.enclosingAfter = 1;
        original.operations[0].afterExecutable = false;
        original.operations[0].enclosingAfter = 1;
        original.operations.push_back(second);
        std::function<void(fs::Region&)> split = [&](fs::Region& region) {
            if (region.kind == fs::Region::Operation && region.operation == 0) {
                const auto firstRegion = region;
                auto secondRegion = region;
                secondRegion.operation = 1;
                region = fs::Region{};
                region.children = {firstRegion, secondRegion};
                return;
            }
            for (auto& child : region.children) {
                split(child);
            }
        };
        split(original.body);
    }
    fs::ProgramAnalysis analysis(input, std::move(original));
    std::size_t comparison = fs::NoControlId, branch = fs::NoControlId, producer = fs::NoControlId;
    const auto& structure = analysis.structure();
    for (std::size_t site = 0; site < structure.originalSites.size(); ++site) {
        if (mlir::isa<mlir::arith::CmpIOp>(structure.originalSites[site])) {
            comparison = site;
        }
        if (mlir::isa<mlir::scf::IfOp>(structure.originalSites[site])) {
            branch = site;
        }
    }
    for (std::size_t phase = 0; phase < structure.operations.size(); ++phase) {
        if (mlir::isa<mlir::pto::TGetValOp>(structure.operations[phase].instruction->elementOp)) {
            producer = phase;
        }
    }
    if (!analysis.complete() || comparison == fs::NoControlId || branch == fs::NoControlId ||
        producer == fs::NoControlId) {
        return false;
    }
    const auto& requirements = analysis.typedRequirementsAt(comparison);
    const auto count = twoPhases ? 2u : 1u;
    if (requirements.size() != count || !analysis.typedRequirementsAt(branch).empty()) {
        return false;
    }
    const auto& subscriptions = analysis.typedSubscriptionsAt(producer);
    if (subscriptions.size() != count) {
        return false;
    }
    std::set<std::size_t> origins;
    for (const auto& request : requirements) {
        if (request.deadlineOriginalSite != comparison || request.source.operation != producer ||
            !request.sourceGapExecutable ||
            request.availability != fs::OriginalValueQualification::Status::NeedsCompletion) {
            return false;
        }
        origins.insert(request.completion.sourcePhase);
    }
    const auto& index = analysis.obligations();
    std::set<std::size_t> witnessed;
    for (auto id : analysis.obligationsAt(comparison)) {
        for (const auto& member : index.origins(id).members) {
            auto witness = analysis.obligationWitness(member.id());
            if (!witness.typedRequirement || member.source.incoming ||
                witness.typedRequirement->completion.sourcePhase != member.source.operation) {
                return false;
            }
            witnessed.insert(member.source.operation);
        }
    }
    return origins.size() == count && witnessed == origins;
}

bool checkPublicViews(const fs::ProgramAnalysis& analysis)
{
    const auto& original = analysis.structure();
    std::size_t views = 0;
    for (std::size_t op = 0; op < original.operations.size(); ++op) {
        for (const auto& effect : original.operations[op].accesses) {
            if (!effect.read && !effect.write) {
                continue;
            }
            const auto& uses = analysis.originalUsesAt(op, effect.cell);
            const auto* site = uses.site(op);
            if (!site || !site->visited || !uses.arena) {
                return false;
            }
            for (auto hazard : {Node::Hazard::RAW, Node::Hazard::WAR, Node::Hazard::WAW}) {
                if (hazard == Node::Hazard::RAW ? !effect.read : !effect.write) {
                    continue;
                }
                const auto view = uses.requirement(op, hazard);
                if (!view.valid || view.targetEffects().empty()) {
                    return false;
                }
                for (auto incidence : view.targetEffects()) {
                    const auto& witness = original.operations[op].accesses.at(incidence);
                    if (witness.cell != effect.cell || !witness.memory ||
                        (hazard == Node::Hazard::RAW ? !witness.read : !witness.write)) {
                        return false;
                    }
                }
                ++views;
            }
        }
        const auto& requests = analysis.requirementsAt(op);
        for (std::size_t index = 0; index < requests.size(); ++index) {
            const auto& request = requests[index];
            const auto& answer = analysis.interpretAt(op, index);
            if (answer.factoredUse != &analysis.originalUsesAt(op, request.relationship.cell) ||
                !answer.factoredDemand.valid || answer.factoredDemand.target->operation != op) {
                return false;
            }
        }
    }
    return views != 0;
}

bool conditional(const fs::ProgramAnalysis& analysis, mlir::func::FuncOp function)
{
    const auto& original = analysis.structure();
    const auto writes = named(original, "pto.tload"), reads = named(original, "pto.tstore");
    if (writes.size() != 2 || reads.size() != 2) {
        return false;
    }
    const auto cell = commonCell(original, writes[0], reads[0]);
    if (cell == fs::NoControlId) {
        return false;
    }
    const auto& uses = analysis.originalUsesAt(reads[1], cell);
    if (!uses.complete) {
        return false;
    }
    const fs::FactoredGuardIdentity g{
        reinterpret_cast<std::uintptr_t>(function.getArgument(0).getAsOpaquePointer()), fs::NoControlId};
    std::size_t tests = 0;
    for (const auto& node : uses.nodes()) {
        if (node.kind == Node::Kind::Test) {
            if (!(node.guard == g)) {
                return false;
            }
            ++tests;
        }
    }
    if (tests != 1) {
        return false; // Two scf.if sites, one original Boolean.
    }
    bool replacement = false;
    for (const auto& effect : original.operations[writes[1]].accesses) {
        replacement |= effect.cell == cell && effect.write && effect.definiteWrite;
    }
    for (bool value : {false, true}) {
        Interpretation evaluated{uses, {{g, value}}};
        const auto before = evaluated.origins(uses.site(writes[1])->priorWriters);
        auto expected = before;
        if (value) {
            if (replacement) {
                expected.clear();
            }
            expected.insert(writes[1]);
        }
        if (evaluated.origins(uses.site(reads[1])->priorWriters) != expected ||
            evaluated.condition(uses.site(reads[0])->applicability) != value || !evaluated.valid) {
            return false;
        }
        // Before the first read on g, the preceding conditional producer is active;
        // on !g that read must contribute no obligation at all.
        const auto view = uses.requirement(reads[0], Node::Hazard::RAW);
        if (!view.valid || evaluated.condition(view.applicability) != value) {
            return false;
        }
    }
    // A caller may supply fresh histories explicitly; the default was not fresh.
    fs::FactoredUseInterface entry;
    entry.arena = uses.arena;
    const auto fresh = analysis.applyOriginalUsesAt(reads[1], cell, entry);
    if (!fresh.complete || fresh.arena != uses.arena ||
        !uses.nodes()[uses.site(writes[0])->priorWriters].containsIncoming ||
        fresh.site(writes[0])->priorWriters != 0 || fresh.site(writes[0])->priorReaders != 0) {
        return false;
    }
    return true;
}

bool counted(const fs::ProgramAnalysis& analysis, mlir::func::FuncOp function)
{
    const auto& original = analysis.structure();
    const auto writes = named(original, "pto.tload"), reads = named(original, "pto.tstore");
    if (writes.size() != 2 || reads.size() != 3) {
        return false;
    }
    const auto cell = commonCell(original, writes[0], reads[0]);
    if (cell == fs::NoControlId) {
        return false;
    }
    const auto& root = analysis.lifetimes().factored(cell);
    const auto& body = analysis.originalUsesAt(reads[0], cell);
    if (root.complete || root.repeatedInterfaces.empty() || !root.site(writes[0]) || !body.complete ||
        body.frame.kind != fs::FactoredUseFrame::Kind::ForBody ||
        !body.nodes()[body.site(reads[0])->priorWriters].containsIncoming ||
        !body.nodes()[body.site(reads.back())->nextWriters].containsIncoming) {
        return false;
    }
    std::size_t invariant = 0, varying = 0;
    const auto g = reinterpret_cast<std::uintptr_t>(function.getArgument(0).getAsOpaquePointer());
    for (const auto& node : body.nodes()) {
        if (node.kind != Node::Kind::Test) {
            continue;
        }
        if (node.guard.value == g && node.guard.scope == fs::NoControlId) {
            ++invariant;
        } else if (node.guard.value != g && node.guard.scope == body.frame.owner) {
            ++varying;
        } else {
            return false;
        }
    }
    return invariant == 1 && varying == 1;
}

bool readModifyWrite(const fs::ProgramAnalysis& analysis)
{
    const auto& original = analysis.structure();
    const auto writes = named(original, "pto.tload"), reads = named(original, "pto.tstore");
    const auto updates = named(original, "pto.tmuls");
    if (writes.size() != 2 || reads.size() != 2 || updates.size() != 1) {
        return false;
    }
    const auto cell = commonCell(original, writes[0], reads[0]);
    if (cell == fs::NoControlId) {
        return false;
    }
    const auto& uses = analysis.originalUsesAt(updates[0], cell);
    const auto* site = uses.site(updates[0]);
    if (!uses.complete || !site || !site->access->read || !site->access->write) {
        return false;
    }
    const auto raw = uses.requirement(updates[0], Node::Hazard::RAW);
    const auto waw = uses.requirement(updates[0], Node::Hazard::WAW);
    const auto war = uses.requirement(updates[0], Node::Hazard::WAR);
    if (!raw.valid || !waw.valid || !war.valid || raw.sources != waw.sources || raw.sources != site->priorWriters ||
        war.sources != site->priorReaders) {
        return false;
    }
    Interpretation evaluated{uses, {}};
    if (evaluated.origins(raw.sources).count(updates[0]) || evaluated.origins(war.sources).count(updates[0]) ||
        !evaluated.origins(war.sources).count(reads[0])) {
        return false;
    }
    const auto subsequent = evaluated.origins(uses.site(reads[1])->priorWriters);
    if (!subsequent.count(updates[0])) {
        return false;
    }
    if (site->access->definiteWrite ? subsequent.size() != 1 : !subsequent.count(writes[0])) {
        return false;
    }
    return evaluated.valid;
}

bool whileVisits(const fs::ProgramAnalysis& analysis)
{
    const auto& original = analysis.structure();
    const auto writes = named(original, "pto.tload"), reads = named(original, "pto.tstore");
    if (writes.size() != 2 || reads.size() != 1) {
        return false;
    }
    const auto cell = commonCell(original, writes[0], reads[0]);
    if (cell == fs::NoControlId) {
        return false;
    }
    const auto& before = analysis.originalUsesAt(reads[0], cell);
    const auto& after = analysis.originalUsesAt(writes[1], cell);
    if (!before.complete || !after.complete || before.arena == after.arena ||
        before.frame.kind != fs::FactoredUseFrame::Kind::WhileBefore ||
        after.frame.kind != fs::FactoredUseFrame::Kind::WhileAfter || before.frame.owner != after.frame.owner ||
        !before.nodes()[before.site(reads[0])->priorWriters].containsIncoming ||
        !after.nodes()[after.site(writes[1])->priorReaders].containsIncoming) {
        return false;
    }
    fs::FactoredUseInterface wrongVisit;
    wrongVisit.arena = before.arena;
    return !analysis.applyOriginalUsesAt(writes[1], cell, wrongVisit).complete;
}
} // namespace

bool checkFactoredNativeProvenance()
{
    mlir::DialectRegistry dialects;
    dialects.insert<mlir::pto::PTODialect, mlir::func::FuncDialect, mlir::arith::ArithDialect, mlir::scf::SCFDialect>();
    mlir::MLIRContext context(dialects);
    context.disableMultithreading();
    const char* source = R"mlir(
!tile = !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=128, v_row=16, v_col=128,
    blayout=row_major, slayout=none_box, fractal=512, pad=0>
!view = !pto.partition_tensor_view<16x128xf32>
module attributes {pto.target_arch = "a2a3"} {
  func.func @conditional(%g: i1, %src: !view, %dst: !view)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %addr = arith.constant 0 : i64
    %x = pto.alloc_tile addr = %addr : !tile
    pto.tload ins(%src : !view) outs(%x : !tile)
    scf.if %g {
      pto.tload ins(%src : !view) outs(%x : !tile)
    }
    scf.if %g {
      pto.tstore ins(%x : !tile) outs(%dst : !view)
    }
    pto.tstore ins(%x : !tile) outs(%dst : !view)
    return
  }
  func.func @counted(%g: i1, %src: !view, %dst: !view)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %addr = arith.constant 0 : i64
    %x = pto.alloc_tile addr = %addr : !tile
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    pto.tload ins(%src : !view) outs(%x : !tile)
    scf.for %i = %c0 to %c4 step %c1 {
      pto.tstore ins(%x : !tile) outs(%dst : !view)
      scf.if %g { pto.tload ins(%src : !view) outs(%x : !tile) }
      scf.if %g { pto.tstore ins(%x : !tile) outs(%dst : !view) }
      %h = arith.cmpi eq, %i, %c0 : index
      scf.if %h { pto.tstore ins(%x : !tile) outs(%dst : !view) }
    }
    return
  }
  func.func @rmw(%src: !view, %dst: !view)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %addr = arith.constant 0 : i64
    %one = arith.constant 1.0 : f32
    %x = pto.alloc_tile addr = %addr : !tile
    pto.tload ins(%src : !view) outs(%x : !tile)
    pto.tstore ins(%x : !tile) outs(%dst : !view)
    pto.tmuls ins(%x, %one : !tile, f32) outs(%x : !tile)
    pto.tstore ins(%x : !tile) outs(%dst : !view)
    pto.tload ins(%src : !view) outs(%x : !tile)
    return
  }
  func.func @while_visits(%src: !view, %dst: !view)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %addr = arith.constant 0 : i64
    %x = pto.alloc_tile addr = %addr : !tile
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    pto.tload ins(%src : !view) outs(%x : !tile)
    %final = scf.while (%i = %c0) : (index) -> index {
      pto.tstore ins(%x : !tile) outs(%dst : !view)
      %more = arith.cmpi slt, %i, %c4 : index
      scf.condition(%more) %i : index
    } do {
    ^bb0(%i: index):
      pto.tload ins(%src : !view) outs(%x : !tile)
      %next = arith.addi %i, %c1 : index
      scf.yield %next : index
    }
    return
  }
}
)mlir";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
    if (!module || mlir::failed(mlir::verify(*module))) {
        return false;
    }
    std::string originalIR;
    llvm::raw_string_ostream before(originalIR);
    module->print(before);
    for (auto function : module->getOps<mlir::func::FuncOp>()) {
        mlir::pto::SyncInput input;
        fs::OriginalStructure original;
        if (mlir::failed(input.build(function)) ||
            mlir::failed(fs::importOriginalStructure(function, input, original))) {
            return false;
        }
        {
            fs::OriginalLifetimes lifetimes(original);
            const auto& uses = lifetimes.factored(0);
            fs::FactoredUseInterface retained;
            retained.arena = uses.arena;
            original.version = fs::OriginalProgramVersion::fresh();
            if (lifetimes.complete() || lifetimes.factored(0).complete || lifetimes.factoredAt(0, 0).complete ||
                lifetimes.applyFactoredAt(0, 0, retained).complete) {
                return false;
            }
            if (fs::FactoredProvenance(original, 0, {}, retained).get().complete) {
                return false;
            }
        }
        fs::ProgramAnalysis analysis(input, std::move(original));
        bool passed = analysis.complete() && checkPublicViews(analysis);
        if (passed && function.getName() == "conditional") {
            passed = conditional(analysis, function);
        }
        if (passed && function.getName() == "counted") {
            passed = counted(analysis, function);
        }
        if (passed && function.getName() == "rmw") {
            passed = readModifyWrite(analysis);
        }
        if (passed && function.getName() == "while_visits") {
            passed = whileVisits(analysis);
        }
        if (!passed) {
            llvm::errs() << "native factored provenance fixture failed: " << function.getName() << '\n';
            return false;
        }
    }
    std::string afterIR;
    llvm::raw_string_ostream after(afterIR);
    module->print(after);
    return originalIR == afterIR && checkImportedTyped(context) && checkImportedTyped(context, true);
}
