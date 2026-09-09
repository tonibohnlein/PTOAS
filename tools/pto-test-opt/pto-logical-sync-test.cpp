// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
// See LICENSE for details. Provided AS IS, WITHOUT WARRANTIES OF ANY KIND.
#include "PTO/IR/PTO.h"
#include "LogicalSyncTestJson.h"
#include "PTO/Transforms/InsertSync/LogicalSyncPlan.h"
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
    // The native test target is A3, just as --pto-arch=a3 selects it in the
    // compiler. This qualifies a portable a2a3 module without editing payloads.
    module->getOperation()->setAttr("pto.target_arch", StringAttr::get(&context, "a3"));
    auto function = *module->getOps<func::FuncOp>().begin();
    StringRef mutation(argv[2]);
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
            if (mutation == "none")
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
            if (mutation == "erase-wait") {
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
            if (mutation != "facts")
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
            static constexpr const char* kinds[] = {"RAW", "WAR", "WAW", "ACC-resource", "exit"};
            for (const auto& r : required)
                requirements.push_back(
                    llvm::json::Object{
                        {"id", requirements.size()},
                        {"kind", kinds[r.kind]},
                        {"source", r.source},
                        {"target", r.target},
                        {"property", "source-completion-before-target"},
                        {"source_access", access(r.sourceAccess)},
                        {"target_access", access(r.targetAccess)},
                        {"occurrences", testing::encode(r.occurrences)}});
        });
    bool applied = result.status == ConstructionResult::Applied;
    bool preserved = original == print(function);
    llvm::outs() << llvm::json::Value(
                        llvm::json::Object{
                            {"mutation", mutation},
                            {"invoked", invoked},
                            {"changed", changed},
                            {"applied", applied},
                            {"counts_preserved", countsPreserved},
                            {"original_preserved", preserved},
                            {"reason", result.reason},
                            {"work", int64_t(result.work)},
                            {"export_complete", exportComplete},
                            {"accesses", std::move(accesses)},
                            {"phases", std::move(physicalPhases)},
                            {"requirements", std::move(requirements)},
                            {"points", std::move(points)},
                            {"orders", std::move(orders)}})
                 << "\n";
    if (mutation == "none" || mutation == "facts")
        return invoked && applied && exportComplete ? 0 : 1;
    if (!(invoked && changed && !applied && preserved))
        return 1;
    if (mutation == "swap-loads" || mutation == "change-rounding" || mutation == "add-allocation")
        return StringRef(result.reason).contains("original payload") ? 0 : 1;
    if (mutation == "swap-wait-keys")
        return countsPreserved ? 0 : 1;
    if (mutation == "erase-wait")
        return StringRef(result.reason).contains("emitted publication") ||
                       StringRef(result.reason).contains("emitted acquisition") ?
                   0 :
                   1;
    if (mutation == "widen-barrier")
        return StringRef(result.reason).contains("barrier domain") ? 0 : 1;
    return 2;
}
