// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/Shared/CompletionLoss.h"
#include "PTO/Transforms/FrontierSynch/Structured/PeriodicExcessCertificate.h"
#include "mlir/IR/Builders.h"
namespace mlir::pto::frontiersynch {
CompletionLossCertificate::CompletionLossCertificate(std::vector<CompletionLossAtom> input,
    std::vector<std::vector<std::size_t>> edges, std::size_t begin, std::size_t end)
    : atoms(std::move(input)), successors(std::move(edges)), entry(begin), exit(end)
{
    if (atoms.size() != successors.size() || entry >= atoms.size() || exit >= atoms.size()) { return; }
    for (auto [id, atom] : llvm::enumerate(atoms)) {
        if (atom.vertex != id || atom.firstOperation.has_value() != atom.lastOperation.has_value()) { return; }
        if (atom.firstOperation && *atom.firstOperation > *atom.lastOperation) { return; }
        for (auto next : successors[id]) { if (next >= atoms.size()) { return; } }
    }
    shapeValid = successors[exit].empty();
}
bool CompletionLossValue::append(const llvm::APInt& count, const llvm::APInt& internal)
{
    if (!error.empty()) { return false; }
    if (count.getActiveBits() > 256 || internal.getActiveBits() > 256) {
        error = "regional loss count exceeds 256 bits"; return false;
    }
    bool overflow = false;
    auto u = count.zextOrTrunc(256);
    auto cross = payloads.umul_ov(u, overflow);
    if (overflow) { error = "regional loss product overflow"; return false; }
    auto next = bound.uadd_ov(internal.zextOrTrunc(256), overflow);
    if (overflow) { error = "regional internal loss overflow"; return false; }
    next = next.uadd_ov(cross, overflow);
    if (overflow) { error = "regional cross loss overflow"; return false; }
    auto total = payloads.uadd_ov(u, overflow);
    if (overflow) { error = "regional payload count overflow"; return false; }
    payloads = std::move(total); bound = std::move(next);
    return true;
}
bool CompletionLossValue::appendPeriodic(const PeriodicExcessCertificate& certificate,
                                         bool realizedUpperEquality)
{
    if (!error.empty()) { return false; }
    if (!realizedUpperEquality || !certificate.error.empty() || !certificate.upper || !certificate.lower) {
        error = "periodic realized-order certificate unavailable"; return false;
    }
    auto count = llvm::APInt(256, certificate.upper->analysis().payloads.size()) * certificate.trips;
    return append(count, certificate.excess);
}
CompletionLossValue CompletionLossCertificate::evaluate(llvm::ArrayRef<CompletionLossVisit> visits) const
{
    CompletionLossValue result;
    if (!shapeValid || visits.empty() || visits.front().atom != entry ||
        visits.back().atom != exit) { result.error = "loss walk has invalid entry/exit"; return result; }
    for (auto [position, visit] : llvm::enumerate(visits)) {
        if (visit.atom >= atoms.size()) { result.error = "unknown loss atom"; return result; }
        if (position && !llvm::is_contained(successors[visits[position - 1].atom], visit.atom)) {
            result.error = "loss walk has an invalid CFG edge"; return result;
        }
        const auto& atom = atoms[visit.atom];
        if (atom.payloads && *atom.payloads != visit.payloads) {
            result.error = "loss visit changed its certified payload count"; return result;
        }
        llvm::APInt n(256, visit.payloads), internal(256, 0);
        if (!atom.exactRealizedOrder && visit.payloads) { internal = (n * (n - 1)).lshr(1); }
        if (!result.append(n, internal)) { return result; }
    }
    return result;
}
DictionaryAttr CompletionLossCertificate::encode(MLIRContext* context) const
{
    if (!shapeValid) { return {}; }
    Builder b(context);
    SmallVector<Attribute> entries;
    for (const auto& atom : atoms) {
        NamedAttrList entry;
        entry.set("vertex", b.getI64IntegerAttr(atom.vertex));
        if (atom.firstOperation) {
            entry.set("first_operation", b.getI64IntegerAttr(*atom.firstOperation));
            entry.set("last_operation", b.getI64IntegerAttr(*atom.lastOperation));
        }
        entry.set("payloads", atom.payloads ? Attribute(b.getI64IntegerAttr(*atom.payloads)) :
                                            Attribute(b.getStringAttr("actual-visit-count")));
        entry.set("internal_bound", b.getStringAttr(atom.exactRealizedOrder ? "zero" : "u*(u-1)/2"));
        SmallVector<int64_t> next(successors[atom.vertex].begin(), successors[atom.vertex].end());
        entry.set("successors", b.getDenseI64ArrayAttr(next));
        entries.push_back(entry.getDictionary(context));
    }
    return b.getDictionaryAttr({
        b.getNamedAttr("kind", b.getStringAttr("regional-ci-ceiling")),
        b.getNamedAttr("source_numbering", b.getStringAttr("original-function-postorder-operations")),
        b.getNamedAttr("scope", b.getStringAttr("each-admitted-finite-execution")),
        b.getNamedAttr("append", b.getStringAttr("(n,U)+(u,Ui)=(n+u,U+Ui+n*u)")),
        b.getNamedAttr("cross_lower", b.getI64IntegerAttr(0)),
        b.getNamedAttr("entry", b.getI64IntegerAttr(entry)),
        b.getNamedAttr("exit", b.getI64IntegerAttr(exit)),
        b.getNamedAttr("count_evidence", b.getStringAttr("supplied-actual-visit-counts-and-branch-outcomes")),
        b.getNamedAttr("atoms", b.getArrayAttr(entries))});
}
} // namespace mlir::pto::frontiersynch
