// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Explicit adapter from machine scalar circuits to exact integer primitives.
// Affine rows retain mathematical (checked i128) semantics; only the original
// scalar definitions use bit-vector arithmetic. Projection is charged separately.
#include "ArithmeticProgramInternal.h"
#include "PTO/Transforms/FrontierSynch/Shared/ModularScalar.h"
#include "PTO/Transforms/FrontierSynch/AnalysisCost.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "../../InsertSync/SyncScalarEvolution.h"
namespace mlir::pto::frontiersynch::detail {
namespace {
template <class Entry>
std::optional<RegionExpressions::Id> normalizeMachine(
    const ProgramBuilder& builder, RegionExpressions& arena, ModularScalar& scalar, Value input,
    const ArithmeticSite& site, Entry entry)
{
    return scalar.value(input, [&](Value value) -> std::optional<RegionExpressions::Id> {
        for (auto coordinate : site.fixedCoordinates) {
            if (value == coordinate.loop.getInductionVar()) {
                return arena.constant(coordinate.induction);
            }
        }
        for (auto loop : site.loops) {
            if (value == loop.getInductionVar()) {
                return arena.input(value);
            }
        }
        if (auto fixed = builder.constant(value)) {
            return arena.constant(*fixed);
        }
        if (builder.expandEntryDefinition(value)) { return std::nullopt; }
        return entry(value);
    });
}
class MachinePrimitiveBuilder {
public:
    MachinePrimitiveBuilder(ProgramBuilder& builder, const PrimitiveRelation& target, ArrayRef<AffineExpr> rows)
        : builder(builder),
          target(target),
          rows(rows),
          count(target.coordinates.size()),
          output(builder.output),
          limits(builder.limits),
          context(builder.context),
          machineBindings(builder.machineBindings),
          parameterIds(builder.parameterIds),
          referenced(machineBindings.size(), false)
    {}
    std::optional<std::vector<ResiduePiece>> lower()
    {
        if (!bindInputs() || !bindMachineRecipes()) {
            return std::nullopt;
        }
        auto predicate = buildPredicate();
        auto domain = buildDomain();
        if (!predicate || failed(domain)) {
            return std::nullopt;
        }
        std::string diagnostic;
        RegionExpressions::RelationCost cost;
        accumulateCost(output.machinePrimitiveConversions, 1);
        accumulateCost(output.machineExpressionNodes, e.size());
        auto pieces = e.integerRelation(*predicate, variables, *domain, diagnostic, &cost);
        accumulateCost(output.machineRelationPieces, cost.pieces);
        accumulateCost(output.machineProjections, cost.projections);
        if (failed(pieces)) {
            return std::nullopt;
        }
        std::vector<ResiduePiece> retained;
        for (auto& piece : *pieces) {
            ResiduePiece result;
            result.residues.assign(count, 0);
            result.machineSystem = std::make_shared<const IntegerSystem>(std::move(piece));
            retained.push_back(std::move(result));
        }
        return retained;
    }

private:
    ProgramBuilder& builder;
    const PrimitiveRelation& target;
    ArrayRef<AffineExpr> rows;
    const unsigned count;
    ArithmeticProgram& output;
    const ArithmeticLimits& limits;
    MLIRContext* context;
    const std::vector<ProgramBuilder::MachineBinding>& machineBindings;
    const DenseMap<Value, unsigned>& parameterIds;
    RegionExpressions e;
    Block inputs;
    SmallVector<RegionExpressions::Id> variables, mapped;
    SmallVector<bool> available, referenced;
    SmallVector<AffineExpr> dims, symbols;
    bool bindInputs()
    {
        bool validReferences = true;
        for (auto row : rows) {
            if (!row) {
                return false;
            }
            row.walk([&](AffineExpr expression) {
                if (auto symbol = dyn_cast<AffineSymbolExpr>(expression)) {
                    if (symbol.getPosition() >= limits.dimensions) {
                        const auto id = symbol.getPosition() - limits.dimensions;
                        if (id >= referenced.size()) {
                            validReferences = false;
                        } else {
                            referenced[id] = true;
                        }
                    }
                }
            });
        }
        if (!validReferences) {
            return false;
        }
        const auto location = output.context.root->getLoc();
        for (unsigned i = 0; i < count; ++i) {
            auto type =
                i < target.dimensions ? IndexType::get(context) : output.parameters[i - target.dimensions].getType();
            variables.push_back(e.input(inputs.addArgument(type, location)));
            auto coordinate = getAffineDimExpr(i, context);
            (i < target.dimensions ? dims : symbols).push_back(coordinate);
        }
        return true;
    }
    bool bindMachineRecipes()
    {
        mapped = variables;
        // The reserved gap cannot alias a true parameter. Unused gap columns are
        // constants, and every referenced machine placeholder gets an exact recipe.
        symbols.resize(limits.dimensions, getAffineConstantExpr(0, context));
        for (unsigned bindingId = 0; bindingId < machineBindings.size(); ++bindingId) {
            const auto& binding = machineBindings[bindingId];
            if (!referenced[bindingId]) {
                symbols.push_back(getAffineConstantExpr(0, context));
                available.push_back(false);
                continue;
            }
            ModularScalar scalar(e, binding.site.phase ? binding.site.phase->elementOp : output.context.root);
            auto value = scalar.signedValue(binding.value, [&](Value leaf) -> std::optional<RegionExpressions::Id> {
                for (auto fixed : binding.site.fixedCoordinates) {
                    if (leaf == fixed.loop.getInductionVar()) {
                        return e.constant(fixed.induction);
                    }
                }
                for (auto [id, storedLoop] : llvm::enumerate(binding.site.loops)) {
                    auto loop = storedLoop;
                    if (leaf == loop.getInductionVar()) {
                        auto coordinate = binding.offset + id;
                        return coordinate < target.dimensions ?
                                   std::optional<RegionExpressions::Id>(variables[coordinate]) :
                                   std::nullopt;
                    }
                }
                if (auto fixed = builder.constant(leaf)) {
                    return e.constant(*fixed);
                }
                if (builder.expandEntryDefinition(leaf)) { return std::nullopt; }
                auto found = parameterIds.find(leaf);
                return found != parameterIds.end() && found->second < count - target.dimensions ?
                           std::optional<RegionExpressions::Id>(variables[target.dimensions + found->second]) :
                           std::nullopt;
            });
            // Placeholders belonging to another endpoint/primitive may be unused.
            symbols.push_back(getAffineDimExpr(mapped.size(), context));
            mapped.push_back(value.value_or(e.constant(0)));
            available.push_back(value.has_value());
        }
        return true;
    }
    std::optional<RegionExpressions::Id> buildPredicate()
    {
        SmallVector<AffineExpr> coordinates;
        std::vector<IntegerConstraint> nonnegative;
        for (auto [id, row] : llvm::enumerate(rows)) {
            if (!row) {
                return std::nullopt;
            }
            bool valid = true;
            row.walk([&](AffineExpr expression) {
                if (auto symbol = dyn_cast<AffineSymbolExpr>(expression)) {
                    if (symbol.getPosition() >= limits.dimensions) {
                        auto bindingId = symbol.getPosition() - limits.dimensions;
                        if (bindingId >= machineBindings.size()) {
                            valid = false;
                        } else {
                            // A missing recipe must never silently become zero.
                            valid &= available[bindingId];
                        }
                    }
                }
            });
            if (!valid) {
                return std::nullopt;
            }
            auto coordinate = row.replaceDimsAndSymbols(dims, symbols);
            coordinates.push_back(coordinate);
            IntegerConstraint constraint;
            constraint.coefficients.resize(rows.size());
            constraint.coefficients[id] = BoundInteger(-1);
            constraint.bound = BoundInteger(0);
            nonnegative.push_back(std::move(constraint));
        }
        auto system = IntegerSystem::create(rows.size(), nonnegative);
        if (failed(system)) {
            return std::nullopt;
        }
        std::string diagnostic;
        auto predicate = e.integerMappedPredicate(
            *system, mapped, coordinates, 1, SmallVector<uint64_t>(rows.size(), 0), diagnostic);
        return predicate;
    }
    void occurrenceBounds(
        std::vector<IntegerConstraint>& entryRows, mlir::pto::detail::ScalarEvolution& ranges,
        std::optional<std::size_t> siteId, unsigned offset)
    {
        if (!siteId || *siteId >= output.sites.size()) {
            return;
        }
        for (auto [id, storedLoop] : llvm::enumerate(output.sites[*siteId].loops)) {
            auto loop = storedLoop;
            auto range = ranges.signedRange(loop.getInductionVar());
            if (!range || offset + id >= target.dimensions) {
                continue;
            }
            IntegerConstraint row;
            row.coefficients.resize(count);
            row.coefficients[offset + id] = BoundInteger(1);
            row.bound = BoundInteger(range->second);
            entryRows.push_back(row);
            row.coefficients[offset + id] = BoundInteger(-1);
            row.bound = -BoundInteger(range->first);
            entryRows.push_back(std::move(row));
        }
    }
    FailureOr<IntegerSystem> buildDomain()
    {
        // Preserve facts already certified by the scalar model at regional entry.
        // In particular a widened i32 load remains bounded by its source width;
        // it is not an unconstrained 64-bit input to the relation adapter.
        mlir::pto::detail::ScalarEvolution ranges(context, output.context.root);
        std::vector<IntegerConstraint> entryRows;
        for (auto [id, parameter] : llvm::enumerate(output.parameters)) {
            auto range = ranges.signedRange(parameter);
            if (!range || parameter.getType().isInteger(1)) {
                continue;
            }
            ModularScalar representation(e, output.context.root);
            range = representation.inputRange(parameter, *range);
            if (!range) {
                continue;
            }
            IntegerConstraint row;
            row.coefficients.resize(count);
            row.coefficients[target.dimensions + id] = BoundInteger(1);
            row.bound = BoundInteger(range->second);
            entryRows.push_back(row);
            row.coefficients[target.dimensions + id] = BoundInteger(-1);
            row.bound = -BoundInteger(range->first);
            entryRows.push_back(std::move(row));
        }
        occurrenceBounds(entryRows, ranges, target.sourceSite, 0);
        occurrenceBounds(entryRows, ranges, target.targetSite, target.sourceDimensions);
        auto domain = IntegerSystem::create(count, entryRows);
        return domain;
    }
};
} // namespace
bool ProgramBuilder::prepareMachine(Value input, const ArithmeticSite& site)
{
    RegionExpressions arena;
    ModularScalar scalar(arena, site.phase ? site.phase->elementOp : output.context.root);
    auto normalized =
        normalizeMachine(*this, arena, scalar, input, site, [&](Value value) -> std::optional<RegionExpressions::Id> {
            if (!entryParameter(value)) {
                return std::nullopt;
            }
            registerParameter(value);
            return arena.input(value);
        });
    return normalized.has_value() && arena.constructionError().empty();
}
AffineExpr ProgramBuilder::machineValue(Value input, const ArithmeticSite& site, unsigned offset) const
{
    // Admission registered every true leaf before the common parameter schema
    // was frozen. Never create a leaf here or bind a dynamic SSA result to an
    // independent parameter. Recheck the immutable context with existing IDs.
    RegionExpressions arena;
    ModularScalar scalar(arena, site.phase ? site.phase->elementOp : output.context.root);
    auto normalized =
        normalizeMachine(*this, arena, scalar, input, site, [&](Value value) -> std::optional<RegionExpressions::Id> {
            return parameterIds.count(value) ? std::optional<RegionExpressions::Id>(arena.input(value)) : std::nullopt;
        });
    if (!normalized || !arena.constructionError().empty()) {
        return {};
    }
    for (auto [id, binding] : llvm::enumerate(machineBindings)) {
        if (binding.value == input && binding.indexBits == scalar.width(IndexType::get(context)) &&
            binding.offset == offset && binding.site.loops == site.loops &&
            binding.site.fixedCoordinates.size() == site.fixedCoordinates.size() &&
            llvm::all_of(llvm::zip(binding.site.fixedCoordinates, site.fixedCoordinates), [](auto pair) {
                return std::get<0>(pair).loop == std::get<1>(pair).loop &&
                       std::get<0>(pair).induction == std::get<1>(pair).induction;
            })) {
            return getAffineSymbolExpr(limits.dimensions + id, context);
        }
    }
    machineBindings.push_back({input, site, offset, scalar.width(IndexType::get(context))});
    return getAffineSymbolExpr(limits.dimensions + machineBindings.size() - 1, context);
}
bool ProgramBuilder::emitMachine(PrimitiveRelation& target, ArrayRef<AffineExpr> rows)
{
    // This adapter works in original integer coordinates. Other residue
    // profiles retain their existing producer and can request this adapter
    // after an explicit residue conversion, never by relabeling coordinates.
    const unsigned count = target.coordinates.size();
    if (limits.period != 1 || count > limits.dimensions || target.dimensions > count) {
        return false;
    }
    std::vector<const void*> rowIdentity;
    for (auto row : rows) {
        if (!row) {
            return false;
        }
        rowIdentity.push_back(row.getAsOpaquePointer());
    }
    std::vector<Operation*> loopIdentity;
    for (auto site : {target.sourceSite, target.targetSite}) {
        if (site && *site < output.sites.size()) {
            for (auto loop : output.sites[*site].loops) {
                loopIdentity.push_back(loop);
            }
        }
        loopIdentity.push_back(nullptr);
    }
    // Collection froze the entry parameter schema and root assumptions. The
    // key also names each endpoint's loop range context. Phase-local range
    // assumptions, if introduced later, must extend this identity.
    const MachineKey key{count, target.dimensions, target.sourceDimensions, rowIdentity, loopIdentity};
    if (auto found = machinePrimitives.find(key); found != machinePrimitives.end()) {
        llvm::append_range(target.pieces, found->second);
        accumulateCost(output.machinePrimitiveCacheHits, 1);
        return true;
    }
    auto retained = MachinePrimitiveBuilder(*this, target, rows).lower();
    if (!retained) {
        return false;
    }
    llvm::append_range(target.pieces, *retained);
    machinePrimitives.emplace(key, std::move(*retained));
    return true;
}
} // namespace mlir::pto::frontiersynch::detail
