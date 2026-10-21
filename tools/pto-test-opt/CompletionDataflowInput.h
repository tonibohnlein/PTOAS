// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Test-only static dataflow input: original SCF plus existing completion classes.
#ifndef PTO_TEST_COMPLETIONDATAFLOWINPUT_H
#define PTO_TEST_COMPLETIONDATAFLOWINPUT_H
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/CompletionStorage.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "SourcePipeServiceChecks.h"
#include "CurrentCutNativeOrder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"
#include <chrono>
namespace completion_dataflow {
using namespace mlir;
namespace fs = pto::frontiersynch;
inline uint64_t line(Operation* operation)
{
    auto location = dyn_cast<FileLineColLoc>(operation->getLoc());
    return location ? location.getLine() : 0;
}
inline llvm::json::Object reservation(const pto::BaseMemInfo& memory, bool write)
{
    llvm::json::Array bases;
    for (auto address : memory.baseAddresses) {
        bases.push_back(address);
    }
    return llvm::json::Object{
        {"space", static_cast<unsigned>(memory.scope)},
        {"bases", std::move(bases)},
        {"bytes", memory.allocateSize},
        {"write", write},
        {"known", memory.hasKnownPhysicalAddresses && !memory.aliasesUnknownRange}};
}
class Input {
public:
    explicit Input(ArrayRef<const pto::CompoundInstanceElement*> instructions)
        : phases(instructions), storage(instructions)
    {
        classSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
    }
    llvm::json::Object exportFunction(func::FuncOp function)
    {
        const auto start = std::chrono::steady_clock::now();
        llvm::json::Array sites;
        std::string architecture;
        for (Operation* op = function; op; op = op->getParentOp()) {
            if (auto attr = op->getAttrOfType<StringAttr>("pto.target_arch")) {
                architecture = attr.str();
                break;
            }
        }
        auto core = pto::recoverSyncPhysicalCore(function);
        for (const auto* phase : phases) {
            llvm::json::Array facts, effects;
            auto summary = storage.effects({phase});
            for (uint8_t mode : summary) {
                facts.push_back(mode);
            }
            for (const auto* memory : phase->useVec) {
                effects.push_back(reservation(*memory, false));
            }
            for (const auto* memory : phase->defVec) {
                effects.push_back(reservation(*memory, true));
            }
            const auto order = current_cut::nativeOrder(*phase, architecture, core);
            sites.push_back(
                llvm::json::Object{
                    {"line", line(phase->elementOp)},
                    {"operation", phase->opName.getStringRef()},
                    {"pipe", static_cast<unsigned>(phase->kPipeValue)},
                    {"native_prefix_order", order == current_cut::NativeOrder::Serialized  ? "serialized" :
                                            order == current_cut::NativeOrder::Overlapping ? "overlapping" :
                                                                                             "unavailable"},
                    {"facts", std::move(facts)},
                    {"effects", std::move(effects)}});
        }
        llvm::json::Array tree;
        if (!llvm::hasSingleElement(function.getBody())) {
            error = "research input requires one structured function block";
        } else {
            tree = block(function.front());
        }
        llvm::json::Object result{
            {"classes", storage.size()},
            {"sites", std::move(sites)},
            {"tree", std::move(tree)},
            {"error", error},
            {"guarantee", "static-may-storage-facts-only"},
            {"class_preparation_seconds", classSeconds},
            {"descriptor_tree_seconds",
             std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()}};
        if (error.empty()) {
            result["projected"] = exportSourcePipeServices(result);
        }
        return result;
    }

private:
    ArrayRef<const pto::CompoundInstanceElement*> phases;
    std::chrono::steady_clock::time_point began = std::chrono::steady_clock::now();
    fs::CompletionStorage storage;
    double classSeconds = 0;
    std::string error;
    llvm::json::Array block(Block& source)
    {
        llvm::json::Array result;
        for (auto& operation : source) {
            if (auto loop = dyn_cast<scf::ForOp>(operation)) {
                result.push_back(
                    llvm::json::Object{{"kind", "loop"}, {"line", line(loop)}, {"body", block(*loop.getBody())}});
            } else if (auto branch = dyn_cast<scf::IfOp>(operation)) {
                llvm::json::Array otherwise;
                if (!branch.getElseRegion().empty()) {
                    otherwise = block(branch.getElseRegion().front());
                }
                result.push_back(
                    llvm::json::Object{
                        {"kind", "if"},
                        {"line", line(branch)},
                        {"then", block(branch.getThenRegion().front())},
                        {"else", std::move(otherwise)}});
            } else {
                for (auto [id, phase] : llvm::enumerate(phases)) {
                    if (phase->elementOp == &operation) {
                        result.push_back(llvm::json::Object{{"kind", "payload"}, {"site", id}});
                    }
                }
                if (operation.getNumRegions()) {
                    error = "unsupported structured control in research input";
                }
            }
        }
        return result;
    }
};
} // namespace completion_dataflow
#endif
