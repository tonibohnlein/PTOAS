// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Canonical tuple roles and SSA symbol meanings survive Presburger operations.
#include "PTO/Transforms/FrontierSynch/SymbolicDemandAnalysis.h"
#include "llvm/ADT/DenseSet.h"
#include <limits>
namespace mlir::pto::frontiersynch {
namespace {
bool integerType(Type type) { return type && isa<IntegerType, IndexType>(type); }
bool agrees(const presburger::PresburgerSpace& actual, const presburger::PresburgerSpace& expected)
{
    if (!actual.isCompatible(expected)) {
        return false;
    }
    for (auto kind : {presburger::VarKind::Domain, presburger::VarKind::Range, presburger::VarKind::Symbol}) {
        for (unsigned position = 0; position < actual.getNumVarKind(kind); ++position) {
            auto id = actual.getId(kind, position);
            if (id.hasValue() && id != expected.getId(kind, position)) {
                return false;
            }
        }
    }
    return true;
}
} // namespace

FailureOr<SymbolicSchemaHandle> SymbolicSchema::create(
    ArrayRef<SymbolicSite> sites, ArrayRef<Value> parameters, ArrayRef<Type> cellTypes, std::uint64_t partition)
{
    constexpr auto limit = std::numeric_limits<unsigned>::max();
    // Leave space for paired tuple axes, symbols and the constant column in all
    // generated IntegerRelations. These are representation limits, not value caps.
    if (sites.size() > limit || parameters.size() > limit / 4 || cellTypes.size() > limit / 4) {
        return failure();
    }
    auto result = std::shared_ptr<SymbolicSchema>(new SymbolicSchema());
    llvm::DenseSet<Value> symbols;
    llvm::DenseSet<PipelineType> pipes;
    MLIRContext* context = nullptr;
    auto checkType = [&](Type type) {
        if (!integerType(type) || (context && type.getContext() != context)) {
            return false;
        }
        context = type.getContext();
        return true;
    };
    for (auto value : parameters) {
        if (!value || !checkType(value.getType()) || !symbols.insert(value).second) {
            return failure();
        }
    }
    for (const auto& site : sites) {
        if (!site.phase || site.coordinates.size() > limit / 4 - 2) {
            return failure();
        }
        llvm::DenseSet<Value> coordinates;
        for (auto value : site.coordinates) {
            if (!value || !checkType(value.getType()) || symbols.count(value) || !coordinates.insert(value).second) {
                return failure();
            }
        }
        result->depth = std::max(result->depth, static_cast<unsigned>(site.coordinates.size()));
        pipes.insert(site.phase->kPipeValue);
    }
    for (auto type : cellTypes) {
        if (!checkType(type)) {
            return failure();
        }
    }
    result->phaseSites.assign(sites.begin(), sites.end());
    result->symbols.assign(parameters.begin(), parameters.end());
    result->cellAxes.assign(cellTypes.begin(), cellTypes.end());
    result->physicalPartition = partition;
    result->pipes = pipes.size();
    for (auto tuple : {SymbolicTuple::Unit, SymbolicTuple::Occurrence, SymbolicTuple::Event, SymbolicTuple::Cell}) {
        const auto index = static_cast<unsigned>(tuple);
        result->domainAxes[index].resize(*result->arity(tuple));
        result->rangeAxes[index].resize(*result->arity(tuple));
    }
    return SymbolicSchemaHandle(result);
}
FailureOr<unsigned> SymbolicSchema::arity(SymbolicTuple tuple) const
{
    switch (tuple) {
        case SymbolicTuple::Unit:
            return 0;
        case SymbolicTuple::Occurrence:
            return depth + 1;
        case SymbolicTuple::Event:
            return depth + 2;
        case SymbolicTuple::Cell:
            return static_cast<unsigned>(cellAxes.size());
    }
    return failure();
}
FailureOr<presburger::PresburgerSpace> SymbolicSchema::space(
    SymbolicTuple domain, SymbolicTuple range, bool bound) const
{
    auto source = arity(domain), target = arity(range);
    if (failed(source) || failed(target)) {
        return failure();
    }
    auto result = presburger::PresburgerSpace::getRelationSpace(*source, *target, bound ? 0 : symbols.size());
    for (unsigned i = 0; i < *source; ++i) {
        result.setId(
            presburger::VarKind::Domain, i, presburger::Identifier(&domainAxes[static_cast<unsigned>(domain)][i]));
    }
    for (unsigned i = 0; i < *target; ++i) {
        result.setId(
            presburger::VarKind::Range, i, presburger::Identifier(&rangeAxes[static_cast<unsigned>(range)][i]));
    }
    if (!bound) {
        for (auto [i, symbol] : llvm::enumerate(symbols)) {
            result.setId(presburger::VarKind::Symbol, i, presburger::Identifier(symbol));
        }
    }
    return result;
}

FailureOr<SymbolicPrimitive> SymbolicPrimitive::import(
    SymbolicSchemaHandle schema, SymbolicTuple domain, SymbolicTuple range, ArrayRef<Value> orderedSymbols,
    const presburger::PresburgerRelation& relation)
{
    if (!schema || schema->parameters() != orderedSymbols) {
        return failure();
    }
    auto space = schema->space(domain, range);
    if (failed(space) || !agrees(relation.getSpace(), *space)) {
        return failure();
    }
    for (const auto& disjunct : relation.getAllDisjuncts()) {
        if (!agrees(disjunct.getSpace(), *space)) {
            return failure();
        }
    }
    SymbolicPrimitive result;
    result.owner = std::move(schema);
    result.source = domain;
    result.target = range;
    result.value = relation;
    result.value.setSpace(*space);
    return result;
}
} // namespace mlir::pto::frontiersynch
