// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/Shared/PrefixRecipe.h"
#include <algorithm>
#include <map>
#include <set>
namespace mlir::pto::frontiersynch {
std::vector<PrefixRole> prefixRoles(const std::vector<SourcePipeSite>& sites, const ServiceMask& masks)
{
    std::map<std::pair<uint32_t, uint32_t>, std::set<bool>> roles;
    if (sites.size() != masks.size()) {
        return {};
    }
    for (std::size_t site = 0; site < sites.size(); ++site) {
        auto target = sites[site].pipe;
        for (auto source : masks[site]) {
            if (source != target) {
                roles[{source, target}].insert(false);
                roles[{target, source}].insert(true);
            }
        }
    }
    std::vector<PrefixRole> result;
    for (const auto& [direction, names] : roles) {
        uint64_t ordinal = 0;
        for (bool reverse : names) {
            result.push_back({direction.first, direction.second, reverse, ordinal++});
        }
    }
    return result;
}
std::vector<PrefixCommand> prefixCommands(uint32_t target, const std::vector<uint32_t>& sources)
{
    std::vector<PrefixCommand> result;
    for (auto source : sources) {
        if (source != target) {
            result.push_back({PrefixCommandKind::Set, source, target, false});
            result.push_back({PrefixCommandKind::Wait, source, target, false});
            result.push_back({PrefixCommandKind::Set, target, source, true});
            result.push_back({PrefixCommandKind::Wait, target, source, true});
        }
    }
    if (std::find(sources.begin(), sources.end(), target) != sources.end()) {
        result.push_back({PrefixCommandKind::Barrier, target, target, false});
    }
    return result;
}
} // namespace mlir::pto::frontiersynch
