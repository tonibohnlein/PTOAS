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
#include "PTO/Transforms/InsertSync/SyncAddressAnalysis.h"
#include "PTO/Transforms/InsertSync/SyncPhysicalFacts.h"
#include "PTO/Transforms/InsertSync/SyncGlobalOccurrences.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/FileSystem.h"
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
    if (argc < 3 || argc > 5)
        return 2;
    DialectRegistry registry;
    registry.insert<PTODialect, func::FuncDialect, scf::SCFDialect, arith::ArithDialect, DLTIDialect>();
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
    if (mutation == "address-values") {
        SyncAddressEvaluator evaluator(function);
        llvm::json::Array values;
        function.walk([&](Operation *op) {
            if (op->getDialect() != context.getLoadedDialect<arith::ArithDialect>()) return;
            for (Value value : op->getResults()) {
                auto evaluated = evaluator.evaluate(value);
                llvm::json::Object row{{"operation", op->getName().getStringRef()}, {"known", bool(evaluated)}};
                if (evaluated) {
                    llvm::SmallString<64> signedText, unsignedText;
                    evaluated->toString(signedText, 10, true);
                    evaluated->toString(unsignedText, 10, false);
                    row["bits"] = evaluated->getBitWidth();
                    row["signed"] = signedText.str().str();
                    row["unsigned"] = unsignedText.str().str();
                }
                values.push_back(std::move(row));
            }
        });
        auto visits = evaluator.visits();
        function.walk([&](Operation *op) {
            if (op->getDialect() == context.getLoadedDialect<arith::ArithDialect>())
                for (Value value : op->getResults()) (void)evaluator.evaluate(value);
        });
        llvm::outs() << llvm::json::Value(llvm::json::Object{
            {"mode", "address-values"}, {"values", std::move(values)},
            {"visits", visits}, {"repeated_visits", evaluator.visits()}}) << "\n";
        return 0;
    }
    if (mutation == "physical-addresses") {
        auto admission = qualifySyncPhysicalAddresses(function);
        const char *status = admission.status == SyncAddressAdmission::Rejected ? "reject" :
                             admission.status == SyncAddressAdmission::Conservative ? "conservative" : "safe";
        llvm::json::Object output{{"mode", "physical-addresses"}, {"admission", status},
                                  {"reason", admission.reason}, {"translated", false}};
        // Refused geometry must never reach the translator's internal
        // invariants, including in this read-only test entry point.
        if (admission.status == SyncAddressAdmission::Rejected) {
            llvm::outs() << llvm::json::Value(std::move(output)) << "\n";
            return 0;
        }
        SyncAddressEvaluator evaluator(function);
        SmallVector<Value> addresses;
        function.walk([&](Operation *op) {
            if (auto allocation = dyn_cast<AllocTileOp>(op)) {
                if (auto address = allocation.getAddr()) addresses.push_back(address);
            } else if (auto allocation = dyn_cast<AllocMultiTileOp>(op)) {
                if (auto address = allocation.getAddr()) addresses.push_back(address);
            }
        });
        for (Value address : addresses) (void)evaluator.evaluate(address);
        auto visits = evaluator.visits();
        for (Value address : addresses) (void)evaluator.evaluate(address);
        output["address_evaluation_requests"] = addresses.size();
        output["address_evaluation_visits"] = visits;
        output["address_evaluation_repeated_visits"] = evaluator.visits();
        SyncIRs ir;
        Buffer2MemInfoMap buffers;
        MemoryDependentAnalyzer memory;
        PTOIRTranslator translator(ir, memory, buffers, function, SyncAnalysisMode::NORMALSYNC);
        if (failed(translator.Build())) {
            output["reason"] = "physical translation failed";
            llvm::outs() << llvm::json::Value(std::move(output)) << "\n";
            return 1;
        }
        struct Access { const BaseMemInfo *info; unsigned phase; bool write; };
        SmallVector<Access> accesses;
        unsigned phaseId = 0;
        for (const auto &element : ir)
            if (auto *phase = dyn_cast<CompoundInstanceElement>(element.get())) {
                for (auto *info : phase->useVec) accesses.push_back({info, phaseId, false});
                for (auto *info : phase->defVec) accesses.push_back({info, phaseId, true});
                ++phaseId;
            }
        llvm::DenseMap<Value, unsigned> roots;
        llvm::json::Array facts, aliases;
        for (unsigned id = 0; id < accesses.size(); ++id) {
            auto access = accesses[id];
            auto *info = access.info;
            auto root = roots.try_emplace(info->rootBuffer, roots.size()).first->second;
            llvm::json::Array bases;
            for (auto address : info->baseAddresses) bases.push_back(std::to_string(address));
            facts.push_back(llvm::json::Object{
                {"id", id}, {"phase", access.phase}, {"write", access.write}, {"root", root},
                {"scope", static_cast<unsigned>(info->scope)}, {"base_addresses", std::move(bases)},
                {"allocation_bytes", std::to_string(info->allocateSize)},
                {"physical", info->hasKnownPhysicalAddresses}, {"unknown_range", info->aliasesUnknownRange}});
        }
        for (unsigned i = 0; i < accesses.size(); ++i)
            for (unsigned j = 0; j < accesses.size(); ++j)
                aliases.push_back(llvm::json::Object{
                    {"source", i}, {"target", j},
                    {"legacy", memory.MemAlias(accesses[i].info, accesses[j].info)},
                    {"logical", logicalSyncMayAlias(accesses[i].info, accesses[j].info, function,
                                                    InsertSyncGMAliasMode::MayAlias)}});
        output["translated"] = true;
        output["accesses"] = std::move(facts);
        output["aliases"] = std::move(aliases);
        llvm::outs() << llvm::json::Value(std::move(output)) << "\n";
        return 0;
    }
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
        if (failed(translator.Build()))
            return 1;
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
    bool originalUsesPreserved = true, discoveryWritten = false, discoveryFailed = false;
    bool boundaryConditionsInvoked = false, boundaryConditionsPassed = false;
    bool physicalSlotTablesPreserved = true, originalPayloadUsesPreserved = true;
    unsigned selectorPermutationCount = 0;
    const bool selectorPermutation = mutation == "slot-selector-common-permutation";
    const bool slotMutation = mutation == "slot-guard-selector-binding" ||
                              mutation == "slot-guard-known-parameter" ||
                              mutation == "slot-guard-unbound-parameter";
    llvm::DenseMap<Operation*, SmallVector<Value>> originalOperands;
    SmallVector<Operation*> originalPayload;
    SmallVector<Value> originalParameters;
    llvm::json::Array requirements, points, orders, accesses, physicalPhases;
    uint64_t budget = kDefaultLogicalSyncWorkBudget;
    std::string discoveryPath;
    bool explicitBudget = false;
    for (int i = 3; i < argc; ++i) {
        StringRef argument(argv[i]);
        if (argument.consume_front("--discovery-output=")) {
            if (argument.empty() || !discoveryPath.empty()) return 2;
            discoveryPath = argument.str();
        } else {
            if (explicitBudget || argument.getAsInteger(10, budget)) return 2;
            explicitBudget = true;
        }
    }
    auto original = print(function);
    auto result = testing::constructWithEmissionMutation(
        function, InsertSyncGMAliasMode::DisjointArguments, budget,
        [&](func::FuncOp candidate) {
            invoked = true;
            if (mutation == "none" || mutation == "retirement" || mutation == "boundary-conditions")
                return;
            auto counts = [](func::FuncOp f) {
                unsigned sets = 0, waits = 0, barriers = 0;
                f.walk([&](Operation* op) {
                    sets += isa<SetFlagOp>(op);
                    waits += isa<WaitFlagOp>(op);
                    barriers += isa<BarrierOp>(op);
                });
                return std::make_tuple(sets, waits, barriers);
            };
            auto beforeCounts = counts(candidate);
            if (selectorPermutation) {
                // Apply the same bijection to BOTH original D2 selectors.
                // Pairwise equality is invariant under s -> 1-s, although each
                // selector now denotes the other physical slot. The original
                // descriptor operand contract must reject this before the
                // later fresh selector-value check; never bypass that check
                // merely to claim coverage of a later rejection stage.
                SmallVector<MultiTileGetOp> selections;
                SmallVector<std::pair<Operation*, DictionaryAttr>> tables;
                bool qualified = true;
                candidate.walk([&](AllocMultiTileOp alloc) {
                    tables.push_back({alloc, alloc->getAttrDictionary()});
                });
                candidate.walk([&](MultiTileGetOp get) {
                    if (!originalOperands.contains(get)) return;
                    auto rem = get.getSlot().getDefiningOp<arith::RemUIOp>();
                    APInt divisor;
                    if (get.getSource().getType().getCount() != 2 || !rem ||
                        !originalOperands.contains(rem) ||
                        !matchPattern(rem.getRhs(), m_ConstantInt(&divisor)) || divisor != 2) {
                        qualified = false;
                        return;
                    }
                    selections.push_back(get);
                });
                if (qualified && selections.size() == 2) {
                    for (auto get : selections) {
                        OpBuilder builder(get);
                        auto one = builder.create<arith::ConstantOp>(get.getLoc(),
                            builder.getIntegerAttr(get.getSlot().getType(), 1));
                        auto permuted = builder.create<arith::SubIOp>(get.getLoc(), one, get.getSlot());
                        get.getSlotMutable().assign(permuted);
                        ++selectorPermutationCount;
                    }
                    changed = true;
                }
                for (const auto &[op, attrs] : tables)
                    physicalSlotTablesPreserved &= op->getAttrDictionary() == attrs;
                for (Operation* op : originalPayload)
                    originalPayloadUsesPreserved &= llvm::equal(op->getOperands(), originalOperands.at(op));
                for (const auto &[op, operands] : originalOperands)
                    originalUsesPreserved &= llvm::equal(op->getOperands(), operands);
            } else if (slotMutation) {
                DominanceInfo dominance(candidate);
                candidate.walk([&](arith::CmpIOp compare) {
                    if (changed || originalOperands.contains(compare)) return;
                    for (OpOperand &operand : compare->getOpOperands()) {
                        auto iv = dyn_cast<BlockArgument>(operand.get());
                        if (!iv || !isa<scf::ForOp>(iv.getOwner()->getParentOp())) continue;
                        Value replacement;
                        if (mutation == "slot-guard-selector-binding") {
                            candidate.walk([&](arith::RemUIOp remainder) {
                                if (!replacement && originalOperands.contains(remainder) &&
                                    remainder.getLhs() == iv && dominance.dominates(remainder.getResult(), compare))
                                    replacement = remainder.getResult();
                            });
                        } else {
                            bool known = mutation == "slot-guard-known-parameter";
                            for (Value argument : candidate.getArguments())
                                if (!replacement && argument.getType() == iv.getType() &&
                                    llvm::is_contained(originalParameters, argument) == known)
                                    replacement = argument;
                        }
                        if (replacement && replacement != iv && dominance.dominates(replacement, compare)) {
                            operand.set(replacement);
                            changed = true;
                            break;
                        }
                    }
                });
                for (const auto &[op, operands] : originalOperands)
                    originalUsesPreserved &= llvm::equal(op->getOperands(), operands);
            } else if (mutation == "publication-before-source" || mutation == "acquisition-after-consumer" ||
                mutation == "publication-after-independent-load" || mutation == "unrepresented-event-lane") {
                SetFlagOp publication;
                WaitFlagOp acquisition;
                candidate.walk([&](SetFlagOp set) {
                    if (!publication && set.getSrcPipe().getPipe() == PIPE::PIPE_MTE2 &&
                        set.getDstPipe().getPipe() == PIPE::PIPE_V) publication = set;
                });
                if (publication)
                    candidate.walk([&](WaitFlagOp wait) {
                        if (!acquisition && wait.getSrcPipe() == publication.getSrcPipe() &&
                            wait.getDstPipe() == publication.getDstPipe() &&
                            wait.getEventId() == publication.getEventId()) acquisition = wait;
                    });
                if (publication && mutation == "publication-before-source") {
                    for (Operation* op = publication->getPrevNode(); op; op = op->getPrevNode()) {
                        if (op->getNumRegions()) break;
                        if (isa<TLoadOp>(op)) {
                            publication->moveBefore(op);
                            changed = true;
                            break;
                        }
                    }
                } else if (publication && mutation == "publication-after-independent-load") {
                    for (Operation* op = publication->getNextNode(); op; op = op->getNextNode()) {
                        if (op->getNumRegions()) break;
                        if (isa<TLoadOp>(op)) {
                            publication->moveAfter(op);
                            changed = true;
                            break;
                        }
                    }
                } else if (acquisition && mutation == "acquisition-after-consumer") {
                    for (Operation* op = acquisition->getNextNode(); op; op = op->getNextNode()) {
                        if (op->getNumRegions()) break;
                        if (isa<TAbsOp>(op)) {
                            acquisition->moveAfter(op);
                            changed = true;
                            break;
                        }
                    }
                } else if (publication && acquisition && mutation == "unrepresented-event-lane") {
                    // Change both endpoints, keeping the token balanced while
                    // introducing a lane absent from these vector fixtures.
                    publication.setDstPipeAttr(PipeAttr::get(&context, PIPE::PIPE_MTE1));
                    acquisition.setDstPipeAttr(PipeAttr::get(&context, PIPE::PIPE_MTE1));
                    changed = true;
                }
            } else if (mutation == "barrier-after-consumer" || mutation == "unrepresented-barrier-lane") {
                BarrierOp selected;
                candidate.walk([&](BarrierOp barrier) {
                    if (!selected && barrier.getPipe().getPipe() != PIPE::PIPE_ALL) selected = barrier;
                });
                if (selected && mutation == "unrepresented-barrier-lane") {
                    selected.setPipeAttr(PipeAttr::get(&context, PIPE::PIPE_MTE1));
                    changed = true;
                } else if (selected) {
                    for (Operation* op = selected->getNextNode(); op; op = op->getNextNode()) {
                        if (op->getNumRegions()) break;
                        if (isa<TAbsOp>(op)) {
                            selected->moveAfter(op); changed = true; break;
                        }
                    }
                }
            } else if (mutation == "erase-retirement" || mutation == "wrong-retirement-pipe" ||
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
            if (mutation == "boundary-conditions") {
                boundaryConditionsInvoked = true;
                boundaryConditionsPassed = testing::checkBoundaryConditions(facts, kDefaultLogicalSyncWorkBudget);
            }
            if ((slotMutation || selectorPermutation) && !phases.empty()) {
                auto candidate = phases.front()->elementOp->getParentOfType<func::FuncOp>();
                candidate.walk([&](Operation *op) {
                    originalOperands.try_emplace(op, op->getOperands().begin(), op->getOperands().end());
                });
                for (const auto* phase : phases)
                    originalPayload.push_back(phase->elementOp);
                originalParameters.assign(facts.parameters.begin(), facts.parameters.end());
            }
            if (mutation != "facts" && mutation != "retirement" && discoveryPath.empty())
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
            if (!discoveryPath.empty()) {
                std::error_code error;
                auto partial = discoveryPath + ".partial";
                llvm::raw_fd_ostream stream(partial, error, llvm::sys::fs::OF_Text);
                if (error) { discoveryFailed = true; return; }
                stream << llvm::json::Value(llvm::json::Object{
                    {"discovery_only", true}, {"export_complete", exportComplete},
                    {"accesses", llvm::json::Array(accesses)}, {"phases", llvm::json::Array(physicalPhases)},
                    {"requirements", llvm::json::Array(requirements)}, {"points", llvm::json::Array(points)},
                    {"orders", llvm::json::Array(orders)}}) << "\n";
                stream.close();
                if (stream.has_error() || llvm::sys::fs::rename(partial, discoveryPath))
                    discoveryFailed = true;
                else discoveryWritten = true;
            }
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
                            {"original_scalar_uses_preserved", originalUsesPreserved},
                            {"selector_permutation_count", selectorPermutationCount},
                            {"physical_slot_tables_preserved", physicalSlotTablesPreserved},
                            {"original_payload_uses_preserved", originalPayloadUsesPreserved},
                            {"original_parameter_count", originalParameters.size()},
                            {"discovery_written", discoveryWritten},
                            {"boundary_conditions_invoked", boundaryConditionsInvoked},
                            {"boundary_conditions_passed", boundaryConditionsPassed},
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
    if (discoveryFailed) return 2;
    if (mutation == "boundary-conditions")
        return boundaryConditionsInvoked && boundaryConditionsPassed && invoked && applied ? 0 : 1;
    if (mutation == "none" || mutation == "facts" || mutation == "retirement")
        return invoked && applied && exportComplete ? 0 : 1;
    if (!(invoked && changed && !applied && preserved))
        return 1;
    if (selectorPermutation)
        return selectorPermutationCount == 2 && countsPreserved && physicalSlotTablesPreserved &&
                       originalPayloadUsesPreserved && !originalUsesPreserved &&
                       StringRef(result.reason).contains("original payload") ? 0 : 1;
    if (slotMutation)
        return countsPreserved && originalUsesPreserved && !StringRef(result.reason).contains("original payload") ? 0 : 1;
    if (mutation == "swap-loads" || mutation == "change-rounding" || mutation == "add-allocation")
        return StringRef(result.reason).contains("original payload") ? 0 : 1;
    if (mutation == "publication-before-source" || mutation == "acquisition-after-consumer" ||
        mutation == "publication-after-independent-load")
        return countsPreserved && StringRef(result.reason).contains("handoff boundary") ? 0 : 1;
    if (mutation == "unrepresented-event-lane")
        return countsPreserved && StringRef(result.reason).contains("event domain") ? 0 : 1;
    if (mutation == "barrier-after-consumer" || mutation == "unrepresented-barrier-lane")
        return countsPreserved && StringRef(result.reason).contains("barrier") ? 0 : 1;
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
