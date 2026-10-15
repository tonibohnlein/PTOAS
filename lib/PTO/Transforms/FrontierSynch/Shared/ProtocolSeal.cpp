// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "ProtocolSeal.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/SHA256.h"
#include "llvm/ADT/StringExtras.h"
namespace mlir::pto::frontiersynch {
namespace {
class DigestStream : public llvm::raw_ostream {
public:
    DigestStream() : llvm::raw_ostream(true) {}
    std::string result() { return llvm::toHex(hash.final()); }

private:
    llvm::SHA256 hash;
    uint64_t position = 0;
    void write_impl(const char* data, std::size_t size) override
    {
        hash.update(StringRef(data, size));
        position += size;
    }
    uint64_t current_pos() const override { return position; }
};
std::string seal(func::FuncOp function, bool context)
{
    if (!function || function.isDeclaration() || !function.getBody().hasOneBlock()) {
        return {};
    }
    DigestStream stream;
    if (context) {
        function.getFunctionType().print(stream);
        NamedAttrList attributes(function->getAttrDictionary());
        if (auto contract = dyn_cast_or_null<DictionaryAttr>(attributes.get("pto.closed_completion"))) {
            NamedAttrList fields(contract);
            fields.erase("seal");
            attributes.set("pto.closed_completion", fields.getDictionary(function.getContext()));
        }
        attributes.getDictionary(function.getContext()).print(stream);
        if (auto module = function->getParentOfType<ModuleOp>()) {
            module->getAttrDictionary().print(stream);
        }
    }
    AsmState state(function, OpPrintingFlags().useLocalScope().printGenericOpForm());
    function.front().print(stream, state);
    return stream.result();
}
} // namespace
std::string protocolBodySeal(func::FuncOp function) { return seal(function, false); }
std::string closedCompletionSeal(func::FuncOp function) { return seal(function, true); }
} // namespace mlir::pto::frontiersynch
