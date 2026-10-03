// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Mathematical result acceptance. These tags never certify hardware premises.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_ANALYSISCONTRACT_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_ANALYSISCONTRACT_H
#include <cstdint>
#include <string>
namespace mlir::pto::frontiersynch {
enum class SelectedReduction { SelectedOrderCovers, ClosureGenerators };
enum class EffectDomain { SharedModeled, SuppliedExactPremises };
// A cover certificate belongs to the selected closure. SoundUpper may add
// prerequisites beyond the input's modeled requirements.
enum class SelectedClosure { ModeledRequirements, SoundUpper };
enum class DemandInterface : unsigned {
    MinimumRepresentation, UniformMembership,
    PeriodicThresholds, CompletionFrontiers, CellBoundaries, RegionQueries, RegionBoundaryPorts, ExecutableEndpoints
};
constexpr std::uint32_t interfaceBit(DemandInterface interface)
{
    return static_cast<unsigned>(interface) <= static_cast<unsigned>(DemandInterface::ExecutableEndpoints) ?
        std::uint32_t(1) << static_cast<unsigned>(interface) : 0;
}
inline const char* interfaceName(DemandInterface interface)
{
    switch (interface) {
        case DemandInterface::MinimumRepresentation: return "minimum representation";
        case DemandInterface::UniformMembership: return "uniform membership";
        case DemandInterface::PeriodicThresholds: return "periodic thresholds";
        case DemandInterface::CompletionFrontiers: return "completion frontiers";
        case DemandInterface::CellBoundaries: return "cell boundaries";
        case DemandInterface::RegionQueries: return "region queries";
        case DemandInterface::RegionBoundaryPorts: return "guarded region boundary ports";
        case DemandInterface::ExecutableEndpoints: return "executable logical endpoints";
    }
    return "unknown interface";
}
struct AnalysisNeeds {
    SelectedReduction reduction = SelectedReduction::SelectedOrderCovers;
    // Exact input premises supplied to a mathematical API are not a recovered
    // physical-effects certificate. Native/target qualification remains separate.
    bool suppliedExactEffects = false;
    bool allowSoundUpper = false;
    std::uint32_t interfaces = interfaceBit(DemandInterface::MinimumRepresentation);
    static AnalysisNeeds modeledCovers() { return {}; }
    static AnalysisNeeds defaultPolicy()
    {
        AnalysisNeeds needs;
        needs.allowSoundUpper = true;
        return needs;
    }
    // Minimum-exact is the conjunction of exact original-effect premises and
    // selected-order covers, never a synonym for reducing a conservative graph.
    static AnalysisNeeds minimumExact()
    {
        AnalysisNeeds needs;
        needs.suppliedExactEffects = true;
        return needs;
    }
};
struct AnalysisContract {
    SelectedReduction reduction = SelectedReduction::SelectedOrderCovers;
    EffectDomain effects = EffectDomain::SharedModeled;
    SelectedClosure closure = SelectedClosure::ModeledRequirements;
    std::uint32_t interfaces = 0;
    // Effect precision and selected-order reduction are independent axes.
    // Covers satisfy a closure-generator request, including modeled covers.
    bool accepts(const AnalysisNeeds& needs, std::string& reason) const
    {
        constexpr auto known = (interfaceBit(DemandInterface::ExecutableEndpoints) << 1) - 1;
        if (needs.interfaces & ~known) {
            reason = "unknown requested analysis interface";
            return false;
        }
        if (needs.reduction == SelectedReduction::SelectedOrderCovers &&
            reduction != SelectedReduction::SelectedOrderCovers) {
            reason = "selected-order covers are not provided";
            return false;
        }
        if (needs.suppliedExactEffects && effects != EffectDomain::SuppliedExactPremises) {
            reason = "result uses shared modeled effects, not supplied exact-effect premises";
            return false;
        }
        if (closure == SelectedClosure::SoundUpper && (!needs.allowSoundUpper || needs.suppliedExactEffects)) {
            reason = "sound upper closure does not establish the original modeled requirements exactly";
            return false;
        }
        for (unsigned index = 0; index <= static_cast<unsigned>(DemandInterface::ExecutableEndpoints); ++index) {
            auto interface = static_cast<DemandInterface>(index);
            if ((needs.interfaces & interfaceBit(interface)) && !(interfaces & interfaceBit(interface))) {
                reason = std::string("requested interface not provided: ") + interfaceName(interface);
                return false;
            }
        }
        reason.clear();
        return true;
    }
};
} // namespace mlir::pto::frontiersynch
#endif
