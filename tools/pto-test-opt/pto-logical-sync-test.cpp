// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/IR/PTO.h"
#include "LogicalSyncTestJson.h"
#include "PTO/Transforms/InsertSync/LogicalSyncPlan.h"
#include "PTO/Transforms/InsertSync/MemoryDependentAnalyzer.h"
#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"
#include "PTO/Transforms/InsertSync/SyncPhysicalFacts.h"
#include "PTO/Transforms/InsertSync/SyncGlobalOccurrences.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::logical_sync;
static std::string print(Operation* op)
{
    std::string text;
    llvm::raw_string_ostream stream(text);
    op->print(stream);
    return text;
}
int main(int argc, char** argv)
{
    if (argc != 3 && argc != 4)
        return 2;
    DialectRegistry registry;
    registry.insert<PTODialect, func::FuncDialect, scf::SCFDialect, arith::ArithDialect>();
    MLIRContext context(registry, MLIRContext::Threading::DISABLED);
    auto module = parseSourceFile<ModuleOp>(argv[1], &context);
    if (!module || !llvm::hasSingleElement(module->getOps<func::FuncOp>()))
        return 2;
    // Portable source fixtures select A3. Explicit targets remain evidence
    // for the physical-context qualification tests.
    auto arch = module->getOperation()->getAttrOfType<StringAttr>("pto.target_arch");
    if (!arch || arch.getValue() == "a2a3")
        module->getOperation()->setAttr("pto.target_arch", StringAttr::get(&context, "a3"));
    auto function = *module->getOps<func::FuncOp>().begin();
    StringRef mutation(argv[2]);
    if (mutation == "guard-growth") {
        SmallVector<unsigned> exponential(32, 2), linear(64, 1);
        bool passed = testing::guardEmissionFits({2}, true, 4096) &&
                      testing::guardEmissionFits({64}, false, 4096) &&
                      testing::guardEmissionFits(linear, true, 4096) &&
                      !testing::guardEmissionFits(exponential, true, 4096) &&
                      !testing::guardEmissionFits({65}, true, 4096) &&
                      !testing::guardEmissionFits({2}, true, 0);
        llvm::outs() << llvm::json::Value(llvm::json::Object{{"passed", passed}, {"checks", 6}}) << "\n";
        return passed ? 0 : 1;
    }
    if (mutation == "access-queries") {
        SyncIRs ir;
        Buffer2MemInfoMap buffers;
        MemoryDependentAnalyzer memory;
        PTOIRTranslator translator(ir, memory, buffers, function, SyncAnalysisMode::NORMALSYNC);
        translator.Build();
        struct Access { const BaseMemInfo *info; Operation *op; bool write; };
        SmallVector<Access> accesses;
        for (const auto &element : ir)
            if (auto *phase = dyn_cast<CompoundInstanceElement>(element.get())) {
                for (auto *info : phase->useVec)
                    if (info->scope == AddressSpace::GM) accesses.push_back({info, phase->elementOp, false});
                for (auto *info : phase->defVec)
                    if (info->scope == AddressSpace::GM) accesses.push_back({info, phase->elementOp, true});
            }
        llvm::json::Array answers;
        for (unsigned i = 0; i < accesses.size(); ++i) for (unsigned j = 0; j < accesses.size(); ++j) {
            auto a = accesses[i], b = accesses[j];
            answers.push_back(llvm::json::Object{
                {"source", i}, {"target", j}, {"source_write", a.write}, {"target_write", b.write},
                {"may_alias", logicalSyncMayAlias(a.info, b.info, function, InsertSyncGMAliasMode::MayAlias)},
                {"disjoint_arguments_alias", logicalSyncMayAlias(a.info, b.info, function, InsertSyncGMAliasMode::DisjointArguments)},
                {"distinct_occurrences_disjoint", disjointInsertSyncGlobalOccurrences(a.info, a.op, b.info, b.op, function)}});
        }
        llvm::outs() << llvm::json::Value(llvm::json::Object{{"accesses", accesses.size()}, {"queries", std::move(answers)}}) << "\n";
        return 0;
    }
    bool invoked = false, changed = false, exportComplete = true, countsPreserved = true;
    llvm::json::Array requirements, points, orders, accesses, physicalPhases;
    uint64_t budget = kDefaultLogicalSyncWorkBudget;
    if (argc == 4 && StringRef(argv[3]).getAsInteger(10, budget))
        return 2;
    auto original = print(function);
    auto result = testing::constructWithEmissionMutation(
        function, InsertSyncGMAliasMode::DisjointArguments, budget,
        [&](func::FuncOp candidate) {
            invoked = true;
            if (mutation == "none" || mutation == "retirement")
                return;
            auto counts = [](func::FuncOp f) {
                unsigned sets = 0, waits = 0;
                f.walk([&](Operation* op) {
                    sets += isa<SetFlagOp>(op);
                    waits += isa<WaitFlagOp>(op);
                });
                return std::make_pair(sets, waits);
            };
            auto beforeCounts = counts(candidate);
            if (mutation == "erase-retirement" || mutation == "wrong-retirement-pipe" ||
                mutation == "early-retirement" || mutation == "hint-retirement" ||
                mutation == "unconsumed-before-retirement") {
                auto terminal = dyn_cast_or_null<BarrierOp>(candidate.getBody().front().getTerminator()->getPrevNode());
                if (terminal && terminal.getPipe().getPipe() == PIPE::PIPE_ALL) {
                    if (mutation == "erase-retirement")
                        terminal.erase();
                    else if (mutation == "wrong-retirement-pipe")
                        terminal.setPipeAttr(PipeAttr::get(&context, PIPE::PIPE_MTE1));
                    else if (mutation == "early-retirement")
                        terminal->moveBefore(&candidate.getBody().front().front());
                    else if (mutation == "hint-retirement") {
                        terminal->setAttr("pto.auto_sync_tail_barrier", UnitAttr::get(&context));
                        terminal->setAttr("pto.auto_sync_tail_hint", StringAttr::get(&context, "setwait_mte3_to_s_event0"));
                    } else {
                        OpBuilder builder(terminal);
                        builder.create<SetFlagOp>(terminal.getLoc(), PipeAttr::get(&context, PIPE::PIPE_MTE2),
                            PipeAttr::get(&context, PIPE::PIPE_V), EventAttr::get(&context, EVENT::EVENT_ID7));
                    }
                    changed = true;
                }
            } else if (mutation == "erase-wait") {
                WaitFlagOp selected;
                candidate.walk([&](WaitFlagOp wait) {
                    if (!selected)
                        selected = wait;
                });
                if (selected) {
                    selected.erase();
                    changed = true;
                }
            } else if (mutation == "widen-barrier") {
                scf::IfOp selected;
                candidate.walk([&](BarrierOp barrier) {
                    auto branch = dyn_cast<scf::IfOp>(barrier->getParentOp());
                    if (!selected && branch)
                        selected = branch;
                });
                if (selected) {
                    OpBuilder builder(selected);
                    auto value = builder.create<arith::ConstantIntOp>(selected.getLoc(), 1, 1);
                    selected.getConditionMutable().assign(value);
                    changed = true;
                }
            } else if (mutation == "change-residue" || mutation == "unguard-remainder") {
                arith::RemSIOp selected;
                candidate.walk([&](arith::RemSIOp op) {
                    if (!selected) selected = op;
                });
                if (selected && mutation == "change-residue") {
                    OpBuilder builder(selected);
                    auto one = builder.create<arith::ConstantOp>(selected.getLoc(),
                        builder.getIntegerAttr(selected.getType(), 1));
                    selected.setOperand(1, one);
                    changed = true;
                } else if (selected) {
                    auto guard = selected->getParentOfType<scf::IfOp>();
                    if (guard) {
                        OpBuilder builder(guard);
                        auto yes = builder.create<arith::ConstantIntOp>(guard.getLoc(), 1, 1);
                        guard.getConditionMutable().assign(yes);
                        changed = true;
                    }
                }
            } else if (mutation == "change-slot-distance") {
                arith::CmpIOp selected;
                candidate.walk([&](arith::CmpIOp op) {
                    if (!selected && op.getLhs().getDefiningOp<arith::SubIOp>()) selected = op;
                });
                if (selected) {
                    selected.setPredicate(arith::CmpIPredicate::eq);
                    changed = true;
                }
            } else if (mutation == "swap-loads") {
                SmallVector<Operation*> loads;
                candidate.walk([&](TLoadOp load) { loads.push_back(load); });
                if (loads.size() >= 2 && loads[0]->getBlock() == loads[1]->getBlock()) {
                    loads[1]->moveBefore(loads[0]);
                    changed = true;
                }
            } else if (mutation == "change-rounding") {
                TCvtOp selected;
                candidate.walk([&](TCvtOp convert) {
                    if (!selected)
                        selected = convert;
                });
                if (selected) {
                    selected.setRmodeAttr(RoundModeAttr::get(&context, RoundMode::CAST_RINT));
                    changed = true;
                }
            } else if (mutation == "add-allocation") {
                AllocTileOp selected;
                candidate.walk([&](AllocTileOp op) {
                    if (!selected)
                        selected = op;
                });
                if (selected) {
                    OpBuilder builder(selected);
                    builder.insert(selected->clone());
                    changed = true;
                }
            } else if (mutation == "swap-wait-keys") {
                WaitFlagOp first, second;
                candidate.walk([&](WaitFlagOp wait) {
                    if (wait.getSrcPipe().getPipe() != PIPE::PIPE_MTE2 || wait.getDstPipe().getPipe() != PIPE::PIPE_V)
                        return;
                    if (!first)
                        first = wait;
                    else if (!second && first.getEventId() != wait.getEventId())
                        second = wait;
                });
                if (first && second) {
                    auto id = first.getEventId();
                    first.setEventIdAttr(second.getEventId());
                    second.setEventIdAttr(id);
                    changed = true;
                }
            }
            countsPreserved = beforeCounts == counts(candidate);
        },
        [&](const SyncOccurrences& facts, ArrayRef<const CompoundInstanceElement*> phases,
            ArrayRef<OrderingRequirement> required) {
            if (mutation != "facts" && mutation != "retirement")
                return;
            llvm::DenseMap<const BaseMemInfo*, unsigned> accessIds;
            llvm::DenseMap<Value, unsigned> roots;
            auto access = [&](const BaseMemInfo* value) -> llvm::json::Value {
                if (!value)
                    return nullptr;
                auto [entry, inserted] = accessIds.try_emplace(value, accessIds.size());
                if (!inserted)
                    return entry->second;
                unsigned id = entry->second;
                auto [root, newRoot] = roots.try_emplace(value->rootBuffer, roots.size());
                (void)newRoot;
                llvm::json::Array addresses;
                for (uint64_t address : value->baseAddresses)
                    addresses.push_back(std::to_string(address));
                accesses.push_back(
                    llvm::json::Object{
                        {"id", id},
                        {"root", root->second},
                        {"scope", stringifyAddressSpace(value->scope)},
                        {"base_addresses", std::move(addresses)},
                        {"allocation_bytes", std::to_string(value->allocateSize)},
                        {"physical", value->hasKnownPhysicalAddresses},
                        {"unknown_range", value->aliasesUnknownRange},
                        {"definite", false}});
                return id;
            };
            for (unsigned p = 0; p < phases.size(); ++p) {
                llvm::json::Array reads, writes;
                for (auto* value : phases[p]->useVec)
                    reads.push_back(access(value));
                for (auto* value : phases[p]->defVec)
                    writes.push_back(access(value));
                physicalPhases.push_back(
                    llvm::json::Object{
                        {"id", p},
                        {"op", phases[p]->elementOp->getName().getStringRef()},
                        {"pipe", unsigned(phases[p]->kPipeValue)},
                        {"reads", std::move(reads)},
                        {"writes", std::move(writes)}});
            }
            for (unsigned p = 0; p < facts.points.size(); ++p) {
                points.push_back(testing::encode(facts.domain(p)));
                for (unsigned q = 0; q < facts.points.size(); ++q) {
                    auto relation = facts.ordered(p, q);
                    if (relation)
                        orders.push_back(
                            llvm::json::Object{
                                {"source", p}, {"target", q}, {"relation", testing::encode(*relation.relation)}});
                    else
                        exportComplete = false;
                }
            }
            static constexpr const char* kinds[] = {"RAW", "WAR", "WAW", "ACC-resource", "retirement"};
            for (const auto& r : required)
                requirements.push_back(
                    llvm::json::Object{
                        {"id", requirements.size()},
                        {"kind", kinds[r.kind]},
                        {"source", r.source},
                        {"target", r.target},
                        {"property", r.kind == OrderingRequirement::Retirement ? "completed-before-kernel-retirement" :
                                                                               "source-completion-before-target"},
                        {"source_access", access(r.sourceAccess)},
                        {"target_access", access(r.targetAccess)},
                        {"occurrences", testing::encode(r.occurrences)}});
        });
    static constexpr const char *statuses[] = {
        "applied", "unsupported", "analysis-limit", "unproved", "allocation-failure", "internal-error"};
    bool applied = result.status == ConstructionResult::Applied;
    bool preserved = original == print(function);
    llvm::outs() << llvm::json::Value(
                        llvm::json::Object{
                            {"mutation", mutation},
                            {"status", statuses[result.status]},
                            {"invoked", invoked},
                            {"changed", changed},
                            {"applied", applied},
                            {"counts_preserved", countsPreserved},
                            {"original_preserved", preserved},
                            {"reason", result.reason},
                            {"work", int64_t(result.work)},
                            {"emitted_ir", mutation == "retirement" && applied ? print(function) : ""},
                            {"export_complete", exportComplete},
                            {"accesses", std::move(accesses)},
                            {"phases", std::move(physicalPhases)},
                            {"requirements", std::move(requirements)},
                            {"points", std::move(points)},
                            {"orders", std::move(orders)}})
                 << "\n";
    if (mutation == "none" || mutation == "facts" || mutation == "retirement")
        return invoked && applied && exportComplete ? 0 : 1;
    if (!(invoked && changed && !applied && preserved))
        return 1;
    if (mutation == "swap-loads" || mutation == "change-rounding" || mutation == "add-allocation")
        return StringRef(result.reason).contains("original payload") ? 0 : 1;
    if (mutation == "swap-wait-keys")
        return countsPreserved ? 0 : 1;
    if (mutation == "change-residue" || mutation == "unguard-remainder" || mutation == "change-slot-distance")
        return countsPreserved ? 0 : 1;
    if (mutation == "erase-wait")
        return StringRef(result.reason).contains("emitted publication") ||
                       StringRef(result.reason).contains("emitted acquisition") ?
                   0 :
                   1;
    if (mutation == "erase-retirement" || mutation == "wrong-retirement-pipe" ||
        mutation == "early-retirement" || mutation == "hint-retirement")
        return StringRef(result.reason).contains("retirement drain") ? 0 : 1;
    if (mutation == "unconsumed-before-retirement")
        return StringRef(result.reason).contains("emitted publication") ||
                       StringRef(result.reason).contains("emitted acquisition") ? 0 : 1;
    if (mutation == "widen-barrier")
        return StringRef(result.reason).contains("barrier domain") ? 0 : 1;
    return 2;
}
