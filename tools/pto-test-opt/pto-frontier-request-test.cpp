// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
// Native integration of production obligations and descriptor formation. Phase
// effects and directional query answers are semantic fixtures, not native PTO
// effect/cover recognition or a selected-event certificate.
#include "PTO/Transforms/FrontierSynch/ProgramAnalysis.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <iostream>
#include <set>
using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::frontiersynch;
static void check(bool b, int line = __builtin_LINE())
{
    if (!b) { std::cerr << "request assertion failed at " << line << '\n'; std::exit(1); }
}
static frontiersynch::Region operation(std::size_t id)
{
    frontiersynch::Region r; r.kind = frontiersynch::Region::Operation; r.operation = id; return r;
}
static OriginalStructure fixture(func::FuncOp function, std::vector<std::unique_ptr<CompoundInstanceElement>>& instructions)
{
    OriginalStructure p; p.function = function; p.cells.resize(1); p.operations.resize(5); p.originalSites.resize(6);
    function.walk([&](mlir::Operation* op) {
        if (isa<scf::IfOp>(op)) { p.originalSites[5] = op; }
        if (auto location = dyn_cast<NameLoc>(op->getLoc())) {
            for (unsigned i = 0; i < 5; ++i) {
                if (location.getName().getValue() == "p" + std::to_string(i)) { p.originalSites[i] = op; }
            }
        }
    });
    for (unsigned i = 0; i < 5; ++i) {
        check(p.originalSites[i]);
        auto pipe = i == 2 ? PipelineType::PIPE_V : i == 3 ? PipelineType::PIPE_MTE3 : PipelineType::PIPE_MTE2;
        instructions.push_back(std::make_unique<CompoundInstanceElement>(
            i, SmallVector<const BaseMemInfo*>{}, SmallVector<const BaseMemInfo*>{}, pipe, p.originalSites[i]->getName()));
        instructions.back()->elementOp = p.originalSites[i];
        auto& op = p.operations[i]; op.instruction = instructions.back().get(); op.original = i;
        op.beforeExecutable = op.afterExecutable = true; op.enclosingAfter = i;
        op.accesses.push_back({0, i == 2 || i == 3, i != 2 && i != 3, i != 2 && i != 3});
    }
    frontiersynch::Region choice; choice.kind = frontiersynch::Region::Choice; choice.originalOwner = 5;
    choice.children = {operation(1), frontiersynch::Region{}};
    p.body.children = {operation(0), choice, operation(2), operation(3), operation(4)};
    return p;
}
static DescriptorBoundary suppliedBoundary(
    const ProgramAnalysis& analysis, const OriginalRequestReference& ref, bool source, bool covering)
{
    const auto* family = analysis.obligations().get(ref.family); check(family);
    DescriptorBoundary b;
    b.direction = source ? DescriptorBoundary::Direction::Source : DescriptorBoundary::Direction::Target;
    b.form = covering ? DescriptorBoundary::Form::Covering : DescriptorBoundary::Form::Exact;
    b.interval.query.version = family->key.originalVersion; b.interval.owner = family->key.owner;
    b.interval.query.selector.cell = family->key.cell;
    b.interval.query.occurrence.source = ref.origin ? ref.origin->operation : NoControlId;
    b.interval.query.occurrence.target = family->key.consumerOperation;
    b.interval.query.start = {ref.origin ? ref.origin->operation : NoControlId, OriginalCut::After};
    b.interval.query.stop = {family->key.consumerOperation, OriginalCut::Before};
    b.intervalQualified = b.referenceCoverage = true;
    b.multiplicity = covering ? DescriptorBoundary::Multiplicity::OncePerInterval :
                               DescriptorBoundary::Multiplicity::ExactlyParticipating;
    DescriptorEndpoint e;
    e.cut = source ? b.interval.query.start : b.interval.query.stop;
    e.condition = ref.applicability; e.observation.predicate = ref.applicability;
    e.observation.status = DescriptorObservation::Status::Available; e.legal = true;
    // These are supplied boundary answers for the algebra integration test.
    // Force independent preferred forms on the two alternative writer arms.
    if (!covering && ref.origin && ((source && ref.origin->operation == 0) ||
                                   (!source && ref.origin->operation == 1))) {
        e.observation.status = DescriptorObservation::Status::NotObservableHere;
    }
    b.endpoints.push_back(e);
    if (!ref.origin || ref.origin->incoming) {
        b.form = DescriptorBoundary::Form::Unresolved; b.unresolved = {"incoming interface not supplied"};
    }
    return b;
}
int main()
{
    MLIRContext context; context.loadDialect<arith::ArithDialect, scf::SCFDialect, func::FuncDialect>();
    auto module = parseSourceString<ModuleOp>(R"mlir(
module {
  func.func @test(%g: i1) {
    %p0 = arith.constant 0 : i32 loc("p0")
    scf.if %g { %p1 = arith.constant 1 : i32 loc("p1") }
    %p2 = arith.constant 2 : i32 loc("p2")
    %p3 = arith.constant 3 : i32 loc("p3")
    %p4 = arith.constant 4 : i32 loc("p4")
    return
  }
}
)mlir", &context);
    check(module && succeeded(verify(*module)));
    auto text = [&]() { std::string s; llvm::raw_string_ostream out(s); module->print(out); return out.str(); };
    const auto before = text();
    SyncInput input;
    std::vector<std::unique_ptr<CompoundInstanceElement>> instructions;
    std::size_t calls = 0;
    OriginalRequestBoundaryProvider provider = [&](const ProgramAnalysis& analysis, const OriginalRequestReference& ref) {
        ++calls;
        OriginalRequestAnswers a;
        a.exactSource = suppliedBoundary(analysis, ref, true, false);
        a.coveringSource = suppliedBoundary(analysis, ref, true, true);
        a.exactTarget = suppliedBoundary(analysis, ref, false, false);
        a.coveringTarget = suppliedBoundary(analysis, ref, false, true);
        a.fixedVisit.exact = ref.origin && !ref.origin->incoming;
        return a;
    };
    ProgramAnalysis analysis(input, fixture(module->lookupSymbol<func::FuncOp>("test"), instructions), provider);
    check(analysis.complete()); check(analysis.requests());
    const auto& requests = *analysis.requests();
    check(requests.complete()); check(!requests.groups().empty());
    check(calls == requests.providerQueries());
    check(analysis.lifetimes().stats().requirements == 0); // No compatibility-pair expansion.
    std::set<std::size_t> representedFamilies;
    std::set<unsigned> readerEngines;
    DescriptorFactRef retained{};
    const auto initialFamilies = analysis.obligations().stats().families;
    for (const auto& group : requests.groups()) {
        check(group.declaredSlots.size() <= 3);
        check(group.descriptors.size() <= group.declaredSlots.size());
        for (const auto& slot : group.descriptors) { check(slot.fullyDescribed); }
        for (auto family : group.families) {
            check(analysis.obligations().get(family)); representedFamilies.insert(family.index);
        }
        if (group.deadline == 4 && group.sourceRole == OriginalObligationKey::Role::Reader && group.sourceEngine) {
            readerEngines.insert(*group.sourceEngine);
        }
        for (const auto& slot : group.declaredSlots) {
            std::set<std::size_t> seen;
            std::vector<std::size_t> todo{slot.root};
            while (!todo.empty()) {
                const auto id = todo.back(); todo.pop_back();
                if (!seen.insert(id).second) { continue; }
                const auto& n = requests.descriptors().node(id);
                if (n.kind == OriginalDescriptorSlots::Kind::Both || n.kind == OriginalDescriptorSlots::Kind::Choose) {
                    todo.push_back(n.left); todo.push_back(n.right);
                } else if (n.kind == OriginalDescriptorSlots::Kind::Leaf) {
                    for (auto ref : requests.descriptors().facts(n.facts).obligations) {
                        const auto* r = requests.reference(ref); check(r);
                        retained = ref;
                        if (auto obligation = r->obligation()) {
                            const auto member = analysis.obligations().membership(*obligation);
                            check(member.status != ObligationMembership::Status::Invalid);
                            check(member.id() == *obligation);
                        }
                    }
                }
            }
        }
    }
    // Every registered family with a nonempty represented source expression
    // remains in the group inventory, including unresolved incoming cases.
    for (std::size_t site = 0; site < analysis.structure().originalSites.size(); ++site) {
        for (auto id : analysis.obligationsAt(site)) {
            const auto population = analysis.obligations().origins(id);
            if (!population.complete || !population.members.empty()) { check(representedFamilies.count(id.index)); }
        }
    }
    check(readerEngines.count(unsigned(PipelineType::PIPE_V)));
    check(readerEngines.count(unsigned(PipelineType::PIPE_MTE3)));
    check(analysis.obligations().stats().families == initialFamilies);
    const auto preparedCalls = calls;
    for (unsigned i = 0; i < 8; ++i) { check(analysis.requests()->groups().size() == requests.groups().size()); }
    check(calls == preparedCalls); check(text() == before);
    // The real default adapter also runs during ProgramAnalysis preparation.
    // It must expose a supported immediate fixed visit, while retaining the
    // independently queried covers and explicit unresolved incoming slots.
    std::vector<std::unique_ptr<CompoundInstanceElement>> defaultInstructions;
    ProgramAnalysis direct(input, fixture(module->lookupSymbol<func::FuncOp>("test"), defaultInstructions));
    check(direct.complete());
    std::size_t fullyDescribed = 0, unresolved = 0;
    for (const auto& group : direct.requests()->groups()) {
        fullyDescribed += group.descriptors.size();
        for (const auto& slot : group.declaredSlots) { unresolved += !slot.fullyDescribed; }
    }
    check(fullyDescribed > 0); check(unresolved > 0);
    bool sourceCover = false, targetCover = false;
    const auto& slots = direct.requests()->descriptors();
    for (std::size_t i = 0; i < slots.stats().nodes; ++i) {
        const auto& node = slots.node(i);
        if (node.kind != OriginalDescriptorSlots::Kind::Leaf) { continue; }
        sourceCover |= slots.getBoundary(node.source).form == DescriptorBoundary::Form::Covering;
        targetCover |= slots.getBoundary(node.target).form == DescriptorBoundary::Form::Covering;
    }
    check(sourceCover && targetCover);
    auto foreign = retained;
    foreign.universe = OriginalProgramVersion::fresh();
    check(!requests.reference(foreign));
    check(direct.lifetimes().stats().requirements == 0); check(text() == before);
    // Original snapshot changes invalidate descriptors and borrowed IDs, too.
    ++const_cast<OriginalStructure&>(analysis.structure()).version.revision;
    check(!requests.complete()); check(requests.groups().empty()); check(!requests.reference(retained));
    std::cout << "PASS: request groups, immutable IDs, independent engines, prederived slots, stale snapshots\n";
}
