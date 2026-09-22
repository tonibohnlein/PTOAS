// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Bounded trace generation for portable oracle tests. Native tests supply traces.
#ifndef PTO_TEST_OAHS_TRACE_ORACLE_H
#define PTO_TEST_OAHS_TRACE_ORACLE_H
#include "GraphOracle.h"
namespace oahs_oracle {
using Trace = std::vector<unsigned>;
using Traces = std::vector<Trace>;
inline Traces product(const Traces& a, const Traces& b, std::size_t limit)
{
    Traces out;
    for (const auto& x : a) {
        for (const auto& y : b) {
            if (out.size() >= limit) {
                throw std::runtime_error("oracle trace budget exhausted");
            }
            Trace z = x;
            z.insert(z.end(), y.begin(), y.end());
            out.push_back(std::move(z));
        }
    }
    return out;
}
// Each loop visit can choose either branch independently. Production checking
// does not enumerate trips. Exhausting a budget still fails explicitly.
inline Traces expand(const o::Region& r, unsigned trips, std::size_t limit)
{
    if (r.kind == o::Region::Operation) {
        return {{unsigned(r.operation)}};
    }
    if (r.kind == o::Region::Sequence) {
        Traces result{{}};
        for (const auto& child : r.children) {
            result = product(result, expand(child, trips, limit), limit);
        }
        return result;
    }
    if (r.kind == o::Region::Choice) {
        auto a = expand(r.children[0], trips, limit), b = expand(r.children[1], trips, limit);
        if (a.size() + b.size() > limit) {
            throw std::runtime_error("oracle choice budget exhausted");
        }
        a.insert(a.end(), b.begin(), b.end());
        return a;
    }
    Traces prefix{{}}, body, result;
    if (r.kind == o::Region::While) {
        prefix = expand(r.children[0], trips, limit);
        body = product(expand(r.children[1], trips, limit), prefix, limit);
    } else {
        body = expand(r.children[0], trips, limit);
    }
    for (unsigned i = 0; i <= trips; ++i) {
        if (result.size() + prefix.size() > limit) {
            throw std::runtime_error("oracle loop budget exhausted");
        }
        result.insert(result.end(), prefix.begin(), prefix.end());
        if (i < trips) {
            prefix = product(prefix, body, limit);
        }
    }
    return result;
}
inline Traces traces(const o::Program& p, unsigned trips = 3, std::size_t limit = 20000)
{
    if (p.body.kind == o::Region::Sequence && p.body.children.empty()) {
        Trace t(p.operations.size());
        std::iota(t.begin(), t.end(), 0);
        return {t};
    }
    return expand(p.body, trips, limit);
}
} // namespace oahs_oracle
#endif
