// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Ordered residual candidates without traversing read-only/read-only products.
#ifndef PTO_FRONTIERSYNCH_WRITER_PAIR_INDEX_H
#define PTO_FRONTIERSYNCH_WRITER_PAIR_INDEX_H
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <algorithm>
#include <vector>
namespace mlir::pto::frontiersynch {
class WriterPairIndex {
public:
    // Only a certified read-only entry may be omitted. Unknown roles and
    // read/modify/write entries must be classified as potential writers.
    WriterPairIndex(
        std::size_t size, llvm::function_ref<bool(std::size_t)> readOnly,
        llvm::function_ref<bool(std::size_t)> needsPair = nullptr)
        : onlyRead(size), relevant(size)
    {
        for (std::size_t i = 0; i < size; ++i) {
            onlyRead[i] = readOnly(i);
            relevant[i] = !needsPair || needsPair(i);
            if (!onlyRead[i]) {
                writers.push_back(i);
            }
            if (relevant[i]) {
                participants.push_back(i);
                if (!onlyRead[i]) {
                    participatingWriters.push_back(i);
                }
            }
        }
    }
    bool hasWriters() const { return !writers.empty(); }
    // Exactly the original lexicographic pair order with read/read pairs
    // and irrelevant/irrelevant pairs removed. False stops the traversal without visiting later candidates.
    template <class Visitor>
    bool pairs(Visitor visit) const
    {
        if (participants.empty()) {
            return true;
        }
        for (std::size_t i = 0; i < onlyRead.size(); ++i) {
            const auto* candidates = partners(i, *this);
            if (candidates) {
                for (auto j = std::upper_bound(candidates->begin(), candidates->end(), i); j != candidates->end();
                     ++j) {
                    if (!visit(i, *j)) {
                        return false;
                    }
                }
            } else {
                for (std::size_t j = i + 1; j < onlyRead.size(); ++j) {
                    if (!visit(i, j)) {
                        return false;
                    }
                }
            }
        }
        return true;
    }
    template <class Visitor>
    bool cross(const WriterPairIndex& other, Visitor visit) const
    {
        if (participants.empty() && other.participants.empty()) {
            return true;
        }
        for (std::size_t i = 0; i < onlyRead.size(); ++i) {
            const auto* candidates = partners(i, other);
            if (candidates) {
                for (auto j : *candidates) {
                    if (!visit(i, j)) {
                        return false;
                    }
                }
            } else {
                for (std::size_t j = 0; j < other.onlyRead.size(); ++j) {
                    if (!visit(i, j)) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

private:
    // At least one endpoint must need a residual query, and at least one must
    // potentially write. Null means all opposite endpoints qualify.
    const std::vector<std::size_t>* partners(std::size_t i, const WriterPairIndex& other) const
    {
        if (onlyRead[i]) {
            return relevant[i] ? &other.writers : &other.participatingWriters;
        }
        return relevant[i] ? nullptr : &other.participants;
    }
    std::vector<bool> onlyRead, relevant;
    std::vector<std::size_t> writers, participants, participatingWriters;
};
} // namespace mlir::pto::frontiersynch
#endif
