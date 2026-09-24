// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#ifndef PTO_TEST_FRONTIERWRITECOVERAGECHECKS_H
#define PTO_TEST_FRONTIERWRITECOVERAGECHECKS_H
#include "PTO/Transforms/FrontierSynch/ProgramAnalysis.h"
#include "llvm/Support/raw_ostream.h"
#include <set>
#include <string>
#include <utility>

// Test-only assertions attached to real PTO operations. Untagged programs keep
// the old runner behavior. Explicit expected source tags inspect all applicable
// public requirements, rather than just checking that analysis ran.
inline mlir::LogicalResult checkWriteCoverageExpectations(const mlir::pto::frontiersynch::ProgramAnalysis& analysis)
{
    namespace fs = mlir::pto::frontiersynch;
    bool tested = false;
    const auto& original = analysis.structure();
    std::set<std::string> tags;
    for (const auto& phase : original.operations) {
        auto* op = phase.instruction->elementOp;
        if (auto tag = op->getAttrOfType<mlir::StringAttr>("test.tag")) {
            if (!tags.insert(tag.getValue().str()).second) {
                return op->emitError("duplicate test.tag");
            }
        }
    }
    for (std::size_t target = 0; target < original.operations.size(); ++target) {
        const auto& phase = original.operations[target];
        auto* op = phase.instruction->elementOp;
        if (auto expected = op->getAttrOfType<mlir::BoolAttr>("test.strong")) {
            tested = true;
            bool hasWrite = false, hasStrong = false;
            for (const auto& access : phase.accesses) {
                hasWrite |= access.write;
                hasStrong |= access.write && access.definiteWrite;
                if (access.definiteWrite &&
                    (original.cells[access.cell].storage != fs::Cell::Storage::CanonicalInterval || !access.memory)) {
                    return op->emitError("strong update lacks an exact cell or translated witness");
                }
            }
            if (!hasWrite || hasStrong != expected.getValue()) {
                return op->emitError("unexpected full-cell overwrite qualification");
            }
        }
        if (auto expected = op->getAttrOfType<mlir::StringAttr>("test.coverage")) {
            tested = true;
            bool found = false;
            for (const auto& access : phase.accesses) {
                if (access.write) {
                    found = true;
                    if (expected.getValue() != access.coverage.reason()) {
                        return op->emitError("unexpected coverage status: ") << access.coverage.reason();
                    }
                }
            }
            if (!found) {
                return op->emitError("expected write effects were dropped");
            }
        }
        if (auto expected = op->getAttrOfType<mlir::ArrayAttr>("test.next_readers")) {
            tested = true;
            std::set<std::string> wanted, actual;
            for (auto attribute : expected) {
                auto tag = mlir::dyn_cast<mlir::StringAttr>(attribute);
                if (!tag || !tags.count(tag.getValue().str())) {
                    return op->emitError("invalid expected next-reader tag");
                }
                wanted.insert(tag.getValue().str());
            }
            for (const auto& access : phase.accesses) {
                if (!access.write) {
                    continue;
                }
                for (auto reader : analysis.lifetimes().lifecycleAt(target, access.cell).nextReaders) {
                    auto* source = original.operations.at(reader.operation).instruction->elementOp;
                    auto tag = source->getAttrOfType<mlir::StringAttr>("test.tag");
                    if (!tag) {
                        return op->emitError("untagged next reader");
                    }
                    actual.insert(tag.getValue().str());
                }
            }
            if (wanted != actual) {
                return op->emitError("unexpected backward next-reader set");
            }
        }
        for (const auto& kind :
             {std::make_pair("test.raw", fs::StorageRelationship::RAW),
              std::make_pair("test.war", fs::StorageRelationship::WAR),
              std::make_pair("test.waw", fs::StorageRelationship::WAW)}) {
            auto expected = op->getAttrOfType<mlir::ArrayAttr>(kind.first);
            if (!expected) {
                continue;
            }
            tested = true;
            std::set<std::string> wanted, actual;
            for (auto attribute : expected) {
                auto tag = mlir::dyn_cast<mlir::StringAttr>(attribute);
                if (!tag || !tags.count(tag.getValue().str())) {
                    return op->emitError("invalid expected source tag");
                }
                wanted.insert(tag.getValue().str());
            }
            for (const auto& request : analysis.requirementsAt(target)) {
                if (request.relationship.kind != kind.second) {
                    continue;
                }
                auto* source = original.operations.at(request.source.operation).instruction->elementOp;
                auto tag = source->getAttrOfType<mlir::StringAttr>("test.tag");
                if (!tag) {
                    return op->emitError("untagged source in an asserted requirement set");
                }
                actual.insert(tag.getValue().str());
            }
            if (wanted != actual) {
                return op->emitError("unexpected source set for ") << kind.first;
            }
        }
    }
    if (tested) {
        auto function = original.function;
        llvm::outs() << function.getName() << ": write-coverage-expectations-passed\n";
    }
    return mlir::success();
}
#endif
