// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Exact periodic generators from supplied, atomized rotating footprints. This
// analysis boundary borrows shared phases; it does not import MLIR effects or
// establish physical disjointness from conservative StorageAnalysis facts.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_ROTATINGFOOTPRINTANALYSIS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_ROTATINGFOOTPRINTANALYSIS_H

#include "PTO/Transforms/FrontierSynch/DemandAnalysis.h"
#include "llvm/ADT/DynamicAPInt.h"

namespace mlir::pto::frontiersynch {
struct RotatingFamily {
    llvm::DynamicAPInt slots;
    llvm::DynamicAPInt stride;
    std::size_t atoms = 0;
};
// Owned geometry and reusable exact modular arithmetic for every family.
struct RotatingFamilyFacts {
    RotatingFamily geometry;
    llvm::DynamicAPInt divisor;
    llvm::DynamicAPInt refresh;
    llvm::DynamicAPInt inverseStride;
};
enum class RotatingAccessMode { Read = 1, Write = 2, ReadWrite = 3 };
struct RotatingFragment {
    std::size_t site = 0;
    std::size_t family = 0;
    std::size_t atom = 0;
    llvm::DynamicAPInt offset;
    RotatingAccessMode mode = RotatingAccessMode::Read;
};
struct RotatingWitness {
    Hazard hazard = Hazard::RAW;
    // Indices into fragments(), including the family, atom, normalized offset
    // and borrowed site identity that describe the physical overlap.
    std::size_t sourceFragment = 0;
    std::size_t consumerFragment = 0;
};
struct PeriodicStorageDemand {
    std::size_t source = 0;
    std::size_t consumer = 0;
    // Instances are source(i) -> consumer(i + distance), restricted to those
    // with both endpoints in the invocation. Period is one body iteration.
    llvm::DynamicAPInt distance;
    SmallVector<RotatingWitness> witnesses;
};
struct RotatingFamilyCertificate {
    std::size_t family = 0;
    llvm::DynamicAPInt gcd;
    llvm::DynamicAPInt refresh;
};

class RotatingFootprintAnalysis {
public:
    // Preconditions established by the supplier: sequence is the fixed executed
    // order of period positions borrowing shared phases; families and their within-slot atoms
    // are exact disjoint partitions of the selected requirement model; fragments
    // list ALL internal modeled effects. Physical leastness additionally needs
    // exact original effects; bounding-cell adapters must establish modeled
    // hazard equivalence and retain SharedModeled provenance independently.
    // Fragment (a,l,x,c) accesses (l,(stride[l]*i+c) mod slots[l],x).
    // Family and atom IDs are dense indices, not alias-equivalence guesses.
    // The shared phases and their source IR must remain alive and unchanged.
    // Repeated phase pointers are distinct positions (for grouped periods).
    // Geometry and modular arithmetic are owned; witness IDs index families().
    // Counts, IDs, modes and phase identities are validated. Offsets are reduced
    // modulo slots and duplicate modes merged. Invalid input clears all results.
    // For A normalized fragments, extraction uses O(A log(A+1)) arithmetic/
    // comparison operations after family preprocessing. Normalization costs
    // O(F log(F+1)) for F supplied fragments; validation also visits the supplied
    // sites/families. No slots/trips/LCM enumeration occurs.
    LogicalResult build(
        ArrayRef<const CompoundInstanceElement*> sequence, ArrayRef<RotatingFamily> families,
        ArrayRef<RotatingFragment> fragments);
    bool valid() const { return initialized; }
    ArrayRef<RotatingFamilyFacts> families() const { return familyFacts; }
    ArrayRef<const CompoundInstanceElement*> sites() const { return phaseSites; }
    ArrayRef<RotatingFragment> fragments() const { return normalized; }
    ArrayRef<PeriodicStorageDemand> generators() const { return edges; }
    ArrayRef<RotatingFamilyCertificate> certificates() const { return familyCertificates; }
    // Maximum refresh over accessed families, including read-only families;
    // zero if there are no fragments. It bounds every generator distance.
    const llvm::DynamicAPInt& distanceBound() const { return bound; }

private:
    bool initialized = false;
    SmallVector<RotatingFamilyFacts> familyFacts;
    SmallVector<const CompoundInstanceElement*> phaseSites;
    SmallVector<RotatingFragment> normalized;
    SmallVector<PeriodicStorageDemand> edges;
    SmallVector<RotatingFamilyCertificate> familyCertificates;
    llvm::DynamicAPInt bound;
};
} // namespace mlir::pto::frontiersynch
#endif
