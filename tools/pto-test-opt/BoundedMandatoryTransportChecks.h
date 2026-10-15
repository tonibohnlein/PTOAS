// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_BOUNDEDMANDATORYTRANSPORTCHECKS_H
#define PTO_BOUNDEDMANDATORYTRANSPORTCHECKS_H
#include "../../lib/PTO/Transforms/FrontierSynch/Structured/BoundedMandatoryTransport.h"
#include <array>
namespace {
bool checkMandatoryTransport()
{
    using Certificate = mlir::pto::frontiersynch::detail::BoundedMandatoryTransport;
    using Closure = std::array<std::array<bool, 4>, 4>;
    auto graph = [](unsigned mask) {
        Certificate::Graph result(4);
        unsigned bit = 0;
        for (unsigned a = 0; a < 4; ++a) {
            for (unsigned b = a + 1; b < 4; ++b) {
                if (mask & (1U << bit)) { result[a].push_back(b); }
                ++bit;
            }
        }
        return result;
    };
    auto closure = [](unsigned mask) {
        Closure result{};
        unsigned bit = 0;
        for (unsigned a = 0; a < 4; ++a) {
            result[a][a] = true;
            for (unsigned b = a + 1; b < 4; ++b) { result[a][b] = bool(mask & (1U << bit++)); }
        }
        for (unsigned middle = 0; middle < 4; ++middle) {
            for (unsigned a = 0; a < 4; ++a) {
                for (unsigned b = 0; b < 4; ++b) {
                    result[a][b] = result[a][b] || (result[a][middle] && result[middle][b]);
                }
            }
        }
        return result;
    };
    // Exhaust all ordered four-event graphs in both physical residues. The
    // independent dense closure catches both union-of-residues unsoundness
    // and the loss of valid paths whose intermediate events differ by residue.
    for (unsigned first = 0; first < 64; ++first) {
        for (unsigned second = 0; second < 64; ++second) {
            Certificate certificate({graph(first), graph(second)});
            const auto a = closure(first), b = closure(second);
            for (unsigned source = 0; source < 4; ++source) {
                for (unsigned target = 0; target < 4; ++target) {
                    if (certificate.alwaysReaches(source, target) != (a[source][target] && b[source][target])) {
                        return false;
                    }
                }
                if (certificate.retainedRows() != source + 1) { return false; }
            }
            for (unsigned retry = 0; retry < 16; ++retry) {
                if (certificate.alwaysReaches(0, 3) != (a[0][3] && b[0][3]) ||
                    certificate.retainedRows() != 4) { return false; }
            }
            if (certificate.alwaysReaches(4, 0) || certificate.alwaysReaches(0, 4) ||
                certificate.retainedRows() != 4) { return false; }
        }
    }
    for (auto invalid : std::vector<std::vector<Certificate::Graph>>{
             {}, {{}}, {graph(63), Certificate::Graph(3)}, {{{4}, {}, {}, {}}}}) {
        Certificate certificate(std::move(invalid));
        if (certificate.alwaysReaches(0, 0) || certificate.retainedRows()) { return false; }
    }
    // A different certificate cannot inherit a previously cached positive.
    Certificate positive({graph(63)}), negative({graph(0)});
    return positive.alwaysReaches(0, 3) && !negative.alwaysReaches(0, 3);
}
} // namespace
#endif
