// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_PERIODICCORRESPONDENCE_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_PERIODICCORRESPONDENCE_H

#include "PTO/Transforms/FrontierSynch/OriginalValueQueries.h"
#include "PTO/Transforms/FrontierSynch/PhysicalPermutation.h"

namespace mlir::pto::frontiersynch {

// One directional ORIGINAL endpoint domain. Its executable predicate is
//   !empty && (!testsSelector || selector == selectorEquals) && boundaryTest.
// A missing boundaryTest means true. The scalar is the original address SSA
// value, not a replay of a changed selector, a new visit counter, or event state.
// The finite ordinal domain is the meaning of that predicate, not emitted code.
struct PeriodicEndpointDomain {
    OriginalCut cut;
    periodic_uses::Domain domain;
    bool empty = false, testsSelector = false;
    Value selector;
    uint64_t selectorEquals = 0;
    std::optional<ObservationAtom> boundaryTest;
    OriginalValueQualification selectorQualification, boundaryQualification;
    bool qualified() const
    {
        return selectorQualification.executableAfterPrerequisites() &&
               boundaryQualification.executableAfterPrerequisites();
    }
};
struct PeriodicBankLink {
    periodic_uses::Link occurrence;
    // prev is interpreted at the target; next at the source. Initial/final
    // are their complements on the same bank's participating body visits.
    PeriodicEndpointDomain predecessor, successor, initial, final;
};
struct PeriodicRoleWitness {
    std::size_t operation = NoControlId, relation = NoControlId;
    const BaseMemInfo* effect = nullptr;
    bool read = false, write = false;
};
struct PeriodicUseCorrespondence {
    bool exact = false;
    OriginalInterval interval;
    std::size_t owner = NoControlId, source = NoControlId, target = NoControlId;
    bool sourceWrites = false, targetWrites = false;
    std::vector<periodic_uses::Bank> banks;
    std::vector<PeriodicBankLink> links;
    std::vector<PeriodicRoleWitness> witnesses;
    OriginalCut entry, exit;
    ObservationAtom bypassTest;
    OriginalValueQualification bypassQualification;
    // An initial case says "no local predecessor", NEVER "fresh/completed".
    // The caller retains the invocation/enclosing incoming obligation. Likewise
    // a final case says only that this owner has no next local target.
    bool initialNeedsIncomingInterface = true;
    bool finalKeepsEnclosingContinuation = true;
    std::string reason;
};

} // namespace mlir::pto::frontiersynch
#endif
