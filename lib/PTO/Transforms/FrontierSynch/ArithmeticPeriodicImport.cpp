// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/ArithmeticPeriodicConversion.h"
#include "mlir/IR/Matchers.h"
#include "CountedLoop.h"
namespace mlir::pto::frontiersynch {
namespace {
// Import the already normalized source expression. Only affine combinations
// of the existing parameter bindings are needed for this coordinate change.
// No new execution parameters or unchecked machine arithmetic are introduced.
std::optional<IntegerAffine> lowerForm(const ArithmeticProgram& program, scf::ForOp loop)
{
    mlir::pto::detail::ScalarEvolution scalar(loop.getContext(), loop);
    auto expression = scalar.value(loop.getLowerBound(), [&](Value value) -> AffineExpr {
        for (unsigned i = 0; i < program.parameters.size(); ++i) {
            if (program.parameters[i] == value) { return getAffineSymbolExpr(i, loop.getContext()); }
        }
        return {};
    });
    IntegerAffine form;
    form.coefficients.resize(program.parameters.size());
    SmallVector<std::pair<AffineExpr, BoundInteger>> pending{{expression, BoundInteger(1)}};
    while (!pending.empty()) {
        auto [term, scale] = pending.pop_back_val();
        if (!term) { return {}; }
        if (auto constant = dyn_cast<AffineConstantExpr>(term)) {
            form.constant += scale * BoundInteger(constant.getValue());
        } else if (auto parameter = dyn_cast<AffineSymbolExpr>(term)) {
            if (parameter.getPosition() >= form.coefficients.size()) { return {}; }
            form.coefficients[parameter.getPosition()] += scale;
        } else if (auto binary = dyn_cast<AffineBinaryOpExpr>(term)) {
            if (binary.getKind() == AffineExprKind::Add) {
                pending.push_back({binary.getLHS(), scale});
                pending.push_back({binary.getRHS(), scale});
            } else if (binary.getKind() == AffineExprKind::Mul) {
                auto constant = dyn_cast<AffineConstantExpr>(binary.getRHS());
                auto operand = binary.getLHS();
                if (!constant) {
                    constant = dyn_cast<AffineConstantExpr>(binary.getLHS());
                    operand = binary.getRHS();
                }
                if (!constant) { return {}; }
                pending.push_back({operand, scale * BoundInteger(constant.getValue())});
            } else { return {}; }
        } else { return {}; }
    }
    return form;
}
bool checkSkeleton(const ArithmeticProgram& program, ArithmeticPeriodicProgram& out)
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
    if (!CountedLoop::get(out.loop)) {
        out.conversion.diagnostic =
            "periodic binding needs a proved 64-bit counted-loop ordinal and positive constant step";
        return false;
    }
    if (!lowerForm(program, out.loop)) {
        out.conversion.diagnostic = "periodic binding needs an affine origin in the existing arithmetic parameters";
        return false;
    }
    return true;
}
bool bindSkeleton(const ArithmeticProgram& program, ArithmeticPeriodicProgram& out)
{
    if (!checkSkeleton(program, out)) { return false; }
    for (uint64_t residue = 0; residue < out.period; ++residue) {
        for (uint32_t site = 0; site < program.sites.size(); ++site) {
            out.sites.push_back({site, residue});
            out.phases.push_back(program.sites[site].phase);
        }
    }
    return true;
}
struct OrdinalPhase {
    uint32_t type = 0;
    // old IV quotient = step * ordinal quotient + lower parameter quotient
    // form + offset, for this ordinal phase and parameter-residue tuple.
    BoundInteger offset;
};
std::vector<OrdinalPhase> phasesFor(const ArithmeticProgram& program, uint64_t period,
    int64_t step, const IntegerAffine& lower, const ArithmeticEventKey& event,
    const std::vector<uint64_t>& parameters)
{
    if (event.site >= program.sites.size() || event.residues.size() != 1 ||
        event.residues.front() >= period || parameters.size() != lower.coefficients.size()) { return {}; }
    BoundInteger fixed = lower.constant;
    for (unsigned i = 0; i < parameters.size(); ++i) {
        if (parameters[i] >= period) { return {}; }
        fixed += lower.coefficients[i] * BoundInteger(static_cast<int64_t>(parameters[i]));
    }
    const BoundInteger modulus(static_cast<int64_t>(period));
    std::vector<OrdinalPhase> result;
    for (uint64_t phase = 0; phase < period; ++phase) {
        auto raw = fixed + BoundInteger(step) * BoundInteger(static_cast<int64_t>(phase));
        auto offset = floorDiv(raw, modulus);
        auto residue = raw - offset * modulus;
        if (residue == BoundInteger(static_cast<int64_t>(event.residues.front()))) {
            result.push_back({static_cast<uint32_t>(phase * program.sites.size() + event.site), offset});
        }
    }
    return result;
}
// Exact substitution preserves every inequality and congruence. There is no
// projection or rounding of access distances; floor occurs only in the fixed
// origin/residue conversion above (including negative origins).
FailureOr<IntegerSystem> ordinalSystem(const IntegerSystem& system, const IntegerAffine& lower,
    int64_t step, ArrayRef<BoundInteger> offsets)
{
    const auto coordinates = offsets.size();
    if (system.dimensions() != coordinates + lower.coefficients.size()) { return failure(); }
    auto substitute = [&](std::vector<BoundInteger>& row) {
        BoundInteger constant;
        for (unsigned coordinate = 0; coordinate < coordinates; ++coordinate) {
            auto coefficient = row[coordinate];
            constant += coefficient * offsets[coordinate];
            row[coordinate] *= BoundInteger(step);
            for (unsigned parameter = 0; parameter < lower.coefficients.size(); ++parameter) {
                row[coordinates + parameter] += coefficient * lower.coefficients[parameter];
            }
        }
        return constant;
    };
    auto constraints = system.constraints();
    auto congruences = system.congruences();
    for (auto& row : constraints) { row.bound -= substitute(row.coefficients); }
    for (auto& row : congruences) { row.residue -= substitute(row.coefficients); }
    return IntegerSystem::create(system.dimensions(), constraints, congruences);
}
FailureOr<IntegerSystem> integerRows(unsigned dimensions, ArrayRef<DifferenceBoundConstraint> atoms)
{
    std::vector<IntegerConstraint> rows;
    for (const auto& atom : atoms) {
        IntegerConstraint row{std::vector<BoundInteger>(dimensions), atom.bound};
        if (atom.lhs) { row.coefficients[atom.lhs - 1] += BoundInteger(1); }
        if (atom.rhs) { row.coefficients[atom.rhs - 1] -= BoundInteger(1); }
        rows.push_back(std::move(row));
    }
    return IntegerSystem::create(dimensions, rows, {});
}
FailureOr<IntegerSystem> ordinalSystem(const DifferenceBoundSystem& system, const IntegerAffine& lower,
    int64_t step, ArrayRef<BoundInteger> offsets)
{
    auto adapted = integerRows(system.dimensions(), system.constraints());
    if (failed(adapted)) { return failure(); }
    return ordinalSystem(*adapted, lower, step, offsets);
}
FailureOr<IntegerSystem> ordinalRelation(const IntegerSystem& system,
    const std::vector<IntegerSystem>& sources, const std::vector<IntegerSystem>& targets,
    const IntegerAffine& lower, int64_t step, ArrayRef<BoundInteger> offsets)
{
    (void)sources; (void)targets;
    return ordinalSystem(system, lower, step, offsets);
}
FailureOr<IntegerSystem> ordinalRelation(const DifferenceBoundSystem& system,
    const std::vector<DifferenceBoundSystem>& sources, const std::vector<DifferenceBoundSystem>& targets,
    const IntegerAffine& lower, int64_t step, ArrayRef<BoundInteger> offsets)
{
    // Closing a DBM derives absolute endpoint cutoffs from distance bounds and
    // the common finite domains. They are not new restrictions. Keep the
    // supported rows only when one exact closure proves their equivalence to
    // the original piece; otherwise preserve every atom and let the existing
    // periodic form check report the unsupported cutoff.
    auto common = [](const DifferenceBoundConstraint& atom, unsigned coordinate,
                     const std::vector<DifferenceBoundSystem>& domains) {
        const auto other = coordinate == 1 ? 2U : 1U;
        const bool foreignCoordinate = domains.empty() || atom.lhs == other || atom.rhs == other;
        if (foreignCoordinate) { return false; }
        auto project = [coordinate](unsigned value) {
            return value == coordinate ? 1U : (value > 2 ? value - 1 : 0U);
        };
        for (const auto& domain : domains) {
            if (domain.isEmpty()) { continue; }
            auto bound = domain.bound(project(atom.lhs), project(atom.rhs));
            if (!bound || *bound > atom.bound) { return false; }
        }
        return true;
    };
    const auto atoms = system.constraints();
    std::vector<DifferenceBoundConstraint> supported;
    for (const auto& atom : atoms) {
        const bool distance = (atom.lhs == 1 && atom.rhs == 2) || (atom.lhs == 2 && atom.rhs == 1);
        const bool parameters = atom.lhs != 1 && atom.lhs != 2 && atom.rhs != 1 && atom.rhs != 2;
        const bool accepted = distance || parameters || common(atom, 1, sources) || common(atom, 2, targets);
        if (accepted) {
            supported.push_back(atom);
        }
    }
    auto closure = DifferenceBoundSystem::create(system.dimensions(), supported);
    const bool equivalent = succeeded(closure) && closure->isSubsetOf(system);
    auto adapted = integerRows(system.dimensions(), equivalent ? supported : atoms);
    if (failed(adapted)) { return failure(); }
    return ordinalSystem(*adapted, lower, step, offsets);
}
template<class Stage>
std::vector<typename Stage::System> domainsFor(const Stage& stage, const ArithmeticEventKey& event,
                                     const std::vector<uint64_t>& parameters)
{
    std::vector<typename Stage::System> domains;
    for (const auto& domain : stage.occurrences()) {
        if (domain.site == event.site && domain.residues == event.residues && domain.parameterResidues == parameters) {
            domains.push_back(domain.system);
        }
    }
    return domains;
}
template<class Stage>
bool appendRelations(const ArithmeticProgram& program, const Stage& stage,
                     const TypedArithmeticRelation<typename Stage::System>& relations, bool native,
                     ArithmeticPeriodicInput& input, int64_t step, const IntegerAffine& lower)
{
    for (const auto& [key, pieces] : relations) {
        if (key.source.site >= program.sites.size() || key.target.site >= program.sites.size() ||
            key.source.residues.size() != 1 || key.target.residues.size() != 1 ||
            key.source.residues.front() >= input.parameterPeriod ||
            key.target.residues.front() >= input.parameterPeriod ||
            key.parameterResidues.size() != lower.coefficients.size() ||
            llvm::any_of(key.parameterResidues, [&](uint64_t r) { return r >= input.parameterPeriod; })) {
            return false;
        }
        auto sources = phasesFor(program, input.parameterPeriod, step, lower, key.source, key.parameterResidues);
        auto targets = phasesFor(program, input.parameterPeriod, step, lower, key.target, key.parameterResidues);
        // A residue not visited by a non-coprime stride has no occurrence.
        if (sources.empty() || targets.empty()) { continue; }
        const bool completionStart = key.source.event == ArithmeticEvent::Completion &&
                                     key.target.event == ArithmeticEvent::Start;
        if (!completionStart) {
            const bool coreKinds = (key.source.event == ArithmeticEvent::Start &&
                (key.target.event == ArithmeticEvent::Start || key.target.event == ArithmeticEvent::Completion)) ||
                (key.source.event == ArithmeticEvent::Completion && key.target.event == ArithmeticEvent::Completion);
            if (!native || !coreKinds || input.payloads[sources.front().type].pipe !=
                input.payloads[targets.front().type].pipe) { return false; }
            continue;
        }
        auto sourceDomains = domainsFor(stage, key.source, key.parameterResidues);
        auto targetDomains = domainsFor(stage, key.target, key.parameterResidues);
        if (sourceDomains.empty() || targetDomains.empty()) { return false; }
        for (auto source : sources) {
            for (auto target : targets) {
                std::vector<IntegerSystem> from, to;
                for (const auto& domain : sourceDomains) {
                    auto converted = ordinalSystem(domain, lower, step, {source.offset});
                    if (failed(converted)) { return false; }
                    from.push_back(std::move(*converted));
                }
                for (const auto& domain : targetDomains) {
                    auto converted = ordinalSystem(domain, lower, step, {target.offset});
                    if (failed(converted)) { return false; }
                    to.push_back(std::move(*converted));
                }
                for (const auto& piece : pieces) {
                    auto converted = ordinalRelation(piece, sourceDomains, targetDomains,
                        lower, step, {source.offset, target.offset});
                    if (failed(converted)) { return false; }
                    input.pieces.push_back({source.type, target.type, std::move(*converted), key.parameterResidues,
                        from, to, native, input.pieces.size()});
                }
            }
        }
    }
    return true;
}
} // namespace
bool checkArithmeticPeriodicSkeleton(const ArithmeticProgram& program, std::string& diagnostic)
{
    // Check only source shape. Residue cardinality is checked again when the
    // generator stage supplies its period; no relations or type word are built.
    ArithmeticPeriodicProgram out;
    const bool valid = checkSkeleton(program, out);
    diagnostic = std::move(out.conversion.diagnostic);
    return valid;
}
template<class Stage>
static ArithmeticPeriodicProgram convertGeneratorStage(const ArithmeticProgram& program,
    const Stage& stage, std::shared_ptr<RegionExpressions> expressions, bool buildGuarded)
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
    const auto domain = CountedLoop::get(out.loop);
    const auto lower = lowerForm(program, out.loop);
    if (!domain || !lower ||
        !appendRelations(program, stage, stage.analysis().generators, false, input, domain->step, *lower) ||
        !appendRelations(program, stage, stage.analysis().nativeOrder, true, input, domain->step, *lower)) {
        out.conversion.diagnostic =
            "arithmetic event/domain bindings or non-core native relations lack periodic export";
        return out;
    }
    out.conversion = convertArithmeticPeriodicIntervals(input, buildGuarded);
    return out;
}
ArithmeticPeriodicProgram convertArithmeticPeriodicProgram(const ArithmeticProgram& program,
    const GeneralArithmeticGeneratorStage& stage, std::shared_ptr<RegionExpressions> expressions, bool buildGuarded)
{
    return convertGeneratorStage(program, stage, std::move(expressions), buildGuarded);
}
ArithmeticPeriodicProgram convertArithmeticPeriodicProgram(const ArithmeticProgram& program,
    const DifferenceArithmeticGeneratorStage& stage, std::shared_ptr<RegionExpressions> expressions, bool buildGuarded)
{
    return convertGeneratorStage(program, stage, std::move(expressions), buildGuarded);
}
} // namespace mlir::pto::frontiersynch
