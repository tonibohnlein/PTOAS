// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Shared numerical repetition: definitions retain child references and offset
// boundary identities. They never materialize copies or flatten child demands.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_NUMERICALREPEATEDSQUARING_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_NUMERICALREPEATEDSQUARING_H
#include "PTO/Transforms/FrontierSynch/ChainInterface.h"
namespace mlir::pto::frontiersynch {
struct NumericalRepeatedLink {
    uint32_t source = 0, target = 0;
    bool native = false;
};
struct NumericalRepeatedPort { uint64_t copy = 0; uint32_t port = 0; };
struct NumericalRepeatedSeam {
    uint32_t record = 0; // Canonical input link, including native priority.
    NumericalRepeatedPort source, target;
    bool native = false;
};
struct NumericalSquaringCost {
    uint64_t definitions = 0, merges = 0, boundaryPorts = 0;
    uint64_t indexOperations = 0, candidateLinks = 0, retainedLinks = 0;
};
struct NumericalSquaringQueryCost { uint64_t nodes = 0, indexOperations = 0; };
struct NumericalRepeatedNode {
    uint64_t copies = 0;
    std::array<std::shared_ptr<const NumericalRepeatedNode>, 2> children;
    std::shared_ptr<const NumericalChainInterface> index;
    std::shared_ptr<const NumericalChainMerge> merge;
    // Dense index IDs: every leaf port in the first and last copy, deduplicated
    // when copies==1. Empty repeated bodies may have positive copies but no ports.
    std::vector<NumericalRepeatedPort> ports;
    std::vector<uint32_t> first, last;
    // Only retained seam demands plus fixed native links. Internal demands are
    // referenced through children; a leaf's demand owner stays with its caller.
    std::vector<NumericalRepeatedSeam> seams;
};
struct NumericalRepeatedSquaring {
    std::string error;
    std::shared_ptr<const NumericalRepeatedNode> root;
    std::shared_ptr<const NumericalChainInterface> leaf;
    std::vector<NumericalRepeatedLink> links;
    std::vector<uint32_t> originalToCanonical, representatives;
    NumericalSquaringCost cost;
    std::optional<bool> query(uint64_t sourceCopy, uint32_t source,
        uint64_t targetCopy, uint32_t target, NumericalSquaringQueryCost& cost) const;
    // Lift exact leaf threshold vectors through the shared hierarchy. The
    // caller obtains them by leaf queries/binary searches, charged separately.
    std::optional<std::vector<uint32_t>> thresholds(uint64_t copy,
        const std::vector<uint32_t>& leafThresholds, bool reverse, NumericalSquaringQueryCost& cost) const;
    // Different copies only. Arbitrary same-copy queries remain the leaf's
    // responsibility: endpoint threshold vectors do not determine that order.
    std::optional<bool> across(uint64_t sourceCopy, const std::vector<uint32_t>& sourceForward,
        uint64_t targetCopy, const std::vector<uint32_t>& targetReverse, NumericalSquaringQueryCost& cost) const;
};
// The supplier certifies an exact reflexive numerical leaf order, distinct
// present port identities on native chains, and a complete invariant list of
// adjacent-copy storage/native crossings. Each nonempty chain requires its
// explicit native last->first link. Any other inter-copy prerequisite needs an
// additional qualified construction; no coverage is inferred from the list.
// Links are deduplicated with native priority. Each use preserves the leaf's
// physical cells, effects, pipes and prerequisites under occurrence translation.
// Copies/counts/offsets are uint64; retained boundary directories require 2P to
// fit uint32. The immutable snapshot copies the supplied leaf once. An external
// leaf owner can itself describe a shared fragment, allowing nested use without
// expanding that fragment. Zero copies return an empty root, with no occurrences.
//
// At most 2*floor(log2(T))+1 merge definitions, O(kP+k^2*r+1) work per merge,
// O(log(T+1)*(kP+r+1)) retained words, plus input validation/canonical sorting.
// Seams are retained for provenance; if r is not bounded by P this explicitly
// charges their storage. Queries use hierarchy propagation, not recursive child
// semantic calls. All work counts and offsets are checked; no visit expansion.
NumericalRepeatedSquaring buildNumericalRepeatedSquaring(
    std::shared_ptr<const NumericalChainInterface> leaf,
    const std::vector<NumericalRepeatedLink>& links, uint64_t copies);
} // namespace mlir::pto::frontiersynch
#endif
