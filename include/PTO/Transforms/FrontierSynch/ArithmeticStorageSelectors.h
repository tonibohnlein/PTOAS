// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_ARITHMETICSTORAGESELECTORS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_ARITHMETICSTORAGESELECTORS_H
#include "PTO/Transforms/FrontierSynch/GeneralArithmeticSelectors.h"
namespace mlir::pto::frontiersynch {
struct ArithmeticIntegerPiece {
    const PrimitiveRelation* schema = nullptr; // Borrows the input program.
    std::vector<uint64_t> residues; // Schema dimensions, canonical parameter order.
    IntegerSystem system;
};
// Imports every non-context primitive, restricted to the shared context.
// Parameter columns follow program.primitives.parameters even when primitive
// symbol declarations use another order. Empty unions remain empty.
FailureOr<std::vector<ArithmeticIntegerPiece>> importArithmeticIntegerPieces(const ArithmeticProgram& program);

enum class ArithmeticBoundaryKind {
    FirstWriter, LastWriter, FirstReaderBeforeWrite, LastReaderAfterWrite,
    FirstPayload, LastPayload, FirstSite, LastSite
};
struct ArithmeticBoundarySelector {
    ArithmeticBoundaryKind kind;
    std::optional<AddressSpace> storageSpace; // Absent for native payload selectors.
    Value storageBase;
    std::optional<uint32_t> pipe; // Writers range over all pipes.
    // Storage input is one original signed byte coordinate; native input is
    // empty. Parameters are the program's shared entry bindings. Piece outputs
    // are original occurrence coordinates, through period*q+residue.
    GeneralArithmeticEndpointSelector selector;
    std::optional<std::size_t> site; // Only for per-site occurrence extrema.
};
struct ArithmeticStorageSupportPiece {
    AddressSpace space;
    Value base;
    uint64_t byteResidue = 0;
    std::vector<uint64_t> parameterResidues;
    IntegerSystem domain; // Byte quotient, shared parameter quotients.
};
struct ArithmeticStorageSelectors {
    std::string error;
    uint64_t period = 1;
    unsigned parameterCount = 0;
    std::vector<ArithmeticBoundarySelector> boundaries;
    std::vector<ArithmeticStorageSupportPiece> support;
};
// Requires complete accepted primitives and their reference order. Extrema are
// unique by that total occurrence order, independently of F* functionality.
// A read at an occurrence also writing the queried byte is excluded from the
// before-first/after-last reader interfaces. Empty executions/access sets yield
// absent selectors. No physical bytes, trips or parameter valuations are
// enumerated. Exact integer joins, projections and differences are charged to
// the produced relation pieces; this does not claim a linear-time construction.
ArithmeticStorageSelectors buildArithmeticStorageSelectors(
    const ArithmeticProgram& program, llvm::ArrayRef<uint32_t> sitePipes);
} // namespace mlir::pto::frontiersynch
#endif
