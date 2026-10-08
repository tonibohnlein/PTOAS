// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/ArithmeticPeriodicConversion.h"
#include "mlir/IR/Matchers.h"
namespace mlir::pto::frontiersynch {
namespace {
bool constantIs(Value value, int64_t expected)
{
    APInt integer;
    return matchPattern(value, m_ConstantInt(&integer)) && integer.isSignedIntN(64) &&
           integer.getSExtValue() == expected;
}
bool bindSkeleton(const ArithmeticProgram& program, ArithmeticPeriodicProgram& out)
{
    if (program.sites.empty() || !out.period || out.period > INT64_MAX ||
        program.sites.size() > UINT32_MAX / out.period) {
        out.conversion.diagnostic = "periodic site/residue skeleton is unavailable or exceeds its index representation";
        return false;
    }
    for (const auto& site : program.sites) {
        if (!site.phase || !site.phase->elementOp || site.loops.size() != 1 || !site.guards.empty()) {
            out.conversion.diagnostic = "periodic binding needs one common loop and mandatory unguarded original sites";
            return false;
        }
        if (!out.loop) { out.loop = site.loops.front(); }
        if (site.loops.front() != out.loop || site.phase->elementOp->getBlock() != out.loop.getBody()) {
            out.conversion.diagnostic = "periodic binding needs one fixed ordered body of original cuts";
            return false;
        }
    }
    if (!constantIs(out.loop.getLowerBound(), 0) || !constantIs(out.loop.getStep(), 1)) {
        out.conversion.diagnostic = "periodic binding needs normalized zero-origin unit-step occurrence coordinates";
        return false;
    }
    for (uint64_t residue = 0; residue < out.period; ++residue) {
        for (uint32_t site = 0; site < program.sites.size(); ++site) {
            out.sites.push_back({site, residue});
            out.phases.push_back(program.sites[site].phase);
        }
    }
    return true;
}
std::optional<uint32_t> typeOf(const ArithmeticProgram& program, uint64_t period, const ArithmeticEventKey& event)
{
    if (event.site >= program.sites.size() || event.residues.size() != 1 || event.residues.front() >= period) {
        return {};
    }
    return event.residues.front() * program.sites.size() + event.site;
}
std::vector<IntegerSystem> domainsFor(const GeneralArithmeticGeneratorStage& stage, const ArithmeticEventKey& event,
                                     const std::vector<uint64_t>& parameters)
{
    std::vector<IntegerSystem> domains;
    for (const auto& domain : stage.occurrences()) {
        if (domain.site == event.site && domain.residues == event.residues && domain.parameterResidues == parameters) {
            domains.push_back(domain.system);
        }
    }
    return domains;
}
bool appendRelations(const ArithmeticProgram& program, const GeneralArithmeticGeneratorStage& stage,
                     const GeneralArithmeticRelation& relations, bool native, ArithmeticPeriodicInput& input)
{
    for (const auto& [key, pieces] : relations) {
        auto source = typeOf(program, input.parameterPeriod, key.source);
        auto target = typeOf(program, input.parameterPeriod, key.target);
        if (!source || !target) { return false; }
        const bool completionStart = key.source.event == ArithmeticEvent::Completion &&
                                     key.target.event == ArithmeticEvent::Start;
        if (!completionStart) {
            // These are already supplied by the canonical native pipe chains.
            // No other non-core native relation may silently disappear.
            const bool coreKinds = (key.source.event == ArithmeticEvent::Start &&
                (key.target.event == ArithmeticEvent::Start || key.target.event == ArithmeticEvent::Completion)) ||
                (key.source.event == ArithmeticEvent::Completion && key.target.event == ArithmeticEvent::Completion);
            if (!native || !coreKinds || input.payloads[*source].pipe != input.payloads[*target].pipe) { return false; }
            continue;
        }
        auto sourceDomains = domainsFor(stage, key.source, key.parameterResidues);
        auto targetDomains = domainsFor(stage, key.target, key.parameterResidues);
        if (sourceDomains.empty() || targetDomains.empty()) { return false; }
        for (const auto& piece : pieces) {
            input.pieces.push_back({*source, *target, piece, key.parameterResidues,
                sourceDomains, targetDomains, native, input.pieces.size()});
        }
    }
    return true;
}
} // namespace
ArithmeticPeriodicProgram convertArithmeticPeriodicProgram(const ArithmeticProgram& program,
    const GeneralArithmeticGeneratorStage& stage, std::shared_ptr<RegionExpressions> expressions)
{
    ArithmeticPeriodicProgram out;
    out.period = stage.analysis().period;
    out.conversion.status = ArithmeticPeriodicStatus::AdapterUnavailable;
    if (!stage.belongsTo(program) || !stage.analysis().error.empty()) {
        out.conversion.diagnostic = "arithmetic generator stage does not belong to this successful original program";
        return out;
    }
    if (!bindSkeleton(program, out)) { return out; }
    ArithmeticPeriodicInput input;
    input.expressions = expressions ? std::move(expressions) : std::make_shared<RegionExpressions>();
    input.parameterPeriod = out.period;
    if (program.parameters.size() != stage.analysis().parameterCount) {
        out.conversion.diagnostic = "original arithmetic parameter bindings are unavailable";
        return out;
    }
    for (auto parameter : program.parameters) { input.parameters.push_back(input.expressions->input(parameter)); }
    for (auto* phase : out.phases) {
        input.payloads.push_back({static_cast<uint32_t>(phase->kPipeValue), input.expressions->boolean(true)});
    }
    if (!appendRelations(program, stage, stage.analysis().generators, false, input) ||
        !appendRelations(program, stage, stage.analysis().nativeOrder, true, input)) {
        out.conversion.diagnostic =
            "arithmetic event/domain bindings or non-core native relations lack periodic export";
        return out;
    }
    out.conversion = convertArithmeticPeriodicIntervals(input);
    return out;
}
} // namespace mlir::pto::frontiersynch
