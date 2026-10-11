// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Private ownership boundary for contextualized arithmetic primitive imports.
// The source program must remain alive, at its original address and unchanged,
// while this handle exists. Selector construction borrows its immutable pieces
// synchronously; every returned selector/support value owns its mathematics.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_ARITHMETICIMPORTEDPRIMITIVES_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_ARITHMETICIMPORTEDPRIMITIVES_H
#include "PTO/Transforms/FrontierSynch/Arithmetic/ArithmeticStorageSelectors.h"
#include <utility>
namespace mlir::pto::frontiersynch {
class ImportedArithmeticPrimitives {
public:
    ImportedArithmeticPrimitives(const ImportedArithmeticPrimitives&) = delete;
    ImportedArithmeticPrimitives& operator=(const ImportedArithmeticPrimitives&) = delete;
    ImportedArithmeticPrimitives(ImportedArithmeticPrimitives&& other) noexcept
        : source(std::exchange(other.source, nullptr)), complete(other.complete), values(std::move(other.values)) {}
    ImportedArithmeticPrimitives& operator=(ImportedArithmeticPrimitives&& other) noexcept
    {
        if (this != &other) {
            source = std::exchange(other.source, nullptr);
            complete = other.complete;
            values = std::move(other.values);
        }
        return *this;
    }
    llvm::ArrayRef<ArithmeticIntegerPiece> pieces() const { return values; }
    bool isCompleteFor(const ArithmeticProgram& program) const { return source == &program && complete; }
private:
    ImportedArithmeticPrimitives(const ArithmeticProgram& program, bool occurrencesOnly,
                                std::vector<ArithmeticIntegerPiece> pieces)
        : source(&program), complete(!occurrencesOnly), values(std::move(pieces)) {}
    const ArithmeticProgram* source;
    bool complete;
    std::vector<ArithmeticIntegerPiece> values;
    friend FailureOr<ImportedArithmeticPrimitives> importOwnedArithmeticIntegerPieces(
        const ArithmeticProgram&, bool);
    friend FailureOr<std::vector<ArithmeticIntegerPiece>> importArithmeticIntegerPieces(
        const ArithmeticProgram&, bool);
};
FailureOr<ImportedArithmeticPrimitives> importOwnedArithmeticIntegerPieces(
    const ArithmeticProgram& program, bool occurrencesOnly = false);
ArithmeticStorageSelectors buildArithmeticStorageSelectorsFromImported(
    const ArithmeticProgram& program, const ImportedArithmeticPrimitives& imported,
    llvm::ArrayRef<uint32_t> sitePipes);
} // namespace mlir::pto::frontiersynch
#endif
