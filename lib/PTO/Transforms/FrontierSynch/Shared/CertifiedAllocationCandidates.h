// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Owned completed proof inventory; enumeration never requests a producer.
#ifndef PTO_FRONTIERSYNCH_CERTIFIEDALLOCATIONCANDIDATES_H
#define PTO_FRONTIERSYNCH_CERTIFIEDALLOCATIONCANDIDATES_H
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
namespace mlir::pto::frontiersynch {
inline DictionaryAttr withoutAllocationInventory(DictionaryAttr certificate)
{
    if (!certificate || !certificate.get("certified_candidates")) { return certificate; }
    NamedAttrList values(certificate);
    values.erase("certified_candidates");
    return values.getDictionary(certificate.getContext());
}
inline FailureOr<SmallVector<DictionaryAttr>> certifiedAllocationCandidates(DictionaryAttr certificate)
{
    SmallVector<DictionaryAttr> result;
    if (!certificate) { return result; }
    auto append = [&](Attribute value) {
        auto entry = dyn_cast<DictionaryAttr>(value);
        if (!entry || entry.get("certified_candidates")) { return failure(); }
        if (!llvm::is_contained(result, entry)) { result.push_back(entry); }
        return success();
    };
    if (auto regular = certificate.get("regular_candidate")) {
        if (failed(append(regular))) { return failure(); }
    }
    if (auto raw = certificate.get("certified_candidates")) {
        auto values = dyn_cast<ArrayAttr>(raw);
        if (!values) { return failure(); }
        for (auto value : values) { if (failed(append(value))) { return failure(); } }
    }
    return result;
}
inline DictionaryAttr withAllocationInventory(DictionaryAttr incumbent, ArrayRef<DictionaryAttr> completed)
{
    auto root = withoutAllocationInventory(incumbent);
    SmallVector<Attribute> alternatives;
    for (auto candidate : completed) {
        auto entry = withoutAllocationInventory(candidate);
        if (entry && entry != root && !llvm::is_contained(alternatives, entry)) { alternatives.push_back(entry); }
    }
    if (alternatives.empty()) { return root; }
    NamedAttrList values(root);
    values.set("certified_candidates", ArrayAttr::get(root.getContext(), alternatives));
    return values.getDictionary(root.getContext());
}
} // namespace mlir::pto::frontiersynch
#endif
