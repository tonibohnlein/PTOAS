// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "StructuredInternal.h"
#include "llvm/ADT/DenseSet.h"
#include <limits>
namespace mlir::pto::frontiersynch::structured {
LogicalResult Builder::discover(Region& region, SmallVector<Operation*> loops, SmallVector<Region*> path)
{
    if (options.scope && options.scope != region.getParentOp() &&
        region.getParentOp()->isProperAncestor(options.scope) &&
        !region.isAncestor(options.scope->getParentRegion())) { return success(); }
    if (region.empty()) {
        return success();
    }
    if (!region.hasOneBlock()) {
        return fail(StructuredImportStatus::UnsupportedControl, region.getParentOp(), "multi-block control");
    }
    path.push_back(&region);
    SmallVector<Operation*> operations;
    if (options.scope && options.scope != region.getParentOp() &&
        region.getParentOp()->isProperAncestor(options.scope)) {
        Operation* entry = options.scope;
        while (entry->getParentRegion() != &region) { entry = entry->getParentOp(); }
        operations.push_back(entry);
    } else {
        for (Operation& operation : region.front()) { operations.push_back(&operation); }
    }
    for (Operation* selected : operations) {
        Operation& operation = *selected;
        if (operation.hasTrait<OpTrait::IsTerminator>()) {
            Operation* owner = region.getParentOp();
            bool valid =
                (isa<func::ReturnOp>(&operation) && isa<func::FuncOp>(owner)) ||
                (isa<scf::YieldOp>(&operation) && isa<scf::ForOp, scf::IfOp>(owner)) ||
                (isa<affine::AffineYieldOp>(&operation) && isa<affine::AffineForOp, affine::AffineIfOp>(owner));
            if (!valid) {
                return fail(StructuredImportStatus::UnsupportedControl, &operation, "unsupported control terminator");
            }
        }
        auto phases = (options.context ? options.context->phases() : index).phasesFor(&operation);
        SmallVector<const CompoundInstanceElement*> ordered(phases.begin(), phases.end());
        if (phases.size() > 1) {
            if (!options.phaseOrder) {
                return fail(
                    StructuredImportStatus::MissingQualification, &operation,
                    "multi-phase anchor needs semantic phase order");
            }
            auto supplied = options.phaseOrder(&operation, phases);
            if (failed(supplied)) {
                return fail(StructuredImportStatus::InvalidInput, &operation, "phase-order callback failed");
            }
            ordered = std::move(*supplied);
            llvm::DenseSet<const CompoundInstanceElement*> unique;
            for (auto* phase : ordered) {
                if (!llvm::is_contained(phases, phase) || !unique.insert(phase).second) {
                    return fail(
                        StructuredImportStatus::InvalidInput, &operation, "phase-order result is not a permutation");
                }
            }
            if (unique.size() != phases.size()) {
                return fail(StructuredImportStatus::InvalidInput, &operation, "phase-order result omits phases");
            }
        }
        if (operation.getNumRegions() && !phases.empty()) {
            return fail(
                StructuredImportStatus::UnsupportedControl, &operation,
                "phase anchored on structured control operation");
        }
        for (std::size_t ordinal = 0; ordinal < ordered.size(); ++ordinal) {
            StructuredSite site;
            site.phase = ordered[ordinal];
            site.anchor = &operation;
            site.loops = loops;
            site.regions = path;
            site.localPhase = ordinal;
            for (Operation* loop : loops) {
                if (auto scfLoop = dyn_cast<scf::ForOp>(loop)) {
                    site.inductionVariables.push_back(scfLoop.getInductionVar());
                } else {
                    site.inductionVariables.push_back(cast<affine::AffineForOp>(loop).getInductionVar());
                }
            }
            sites.push_back(std::move(site));
        }
        if (operation.getNumRegions()) {
            if (!isa<scf::ForOp, affine::AffineForOp, scf::IfOp, affine::AffineIfOp>(&operation)) {
                return fail(
                    StructuredImportStatus::UnsupportedControl, &operation, "unsupported region/control operation");
            }
            auto childLoops = loops;
            if (isa<scf::ForOp, affine::AffineForOp>(&operation)) {
                childLoops.push_back(&operation);
                allLoops.push_back(&operation);
            }
            for (Region& child : operation.getRegions()) {
                if (failed(discover(child, childLoops, path))) {
                    return failure();
                }
            }
        }
    }
    return success();
}
FailureOr<SymbolicPrimitive> Builder::import(SymbolicTuple source, SymbolicTuple target, const Set& relation)
{
    auto result = SymbolicPrimitive::import(schema, source, target, schema->parameters(), relation);
    if (failed(result)) {
        return fail(StructuredImportStatus::InvalidInput, function, "generated relation schema mismatch");
    }
    return result;
}
namespace {
// Append constraints using explicitly shared visible axes and fresh existential
// locals. No Presburger intersection/compose API (and hence no solver) is used.
LogicalResult conjoin(Poly& target, const Relation& source, ArrayRef<unsigned> visible)
{
    if (source.getNumLocalVars() >= std::numeric_limits<unsigned>::max() - target.relation.getNumVars()) {
        return failure();
    }
    SmallVector<unsigned> columns(visible.begin(), visible.end());
    for (unsigned i = 0; i < source.getNumLocalVars(); ++i) {
        columns.push_back(target.relation.getNumVars());
        if (failed(target.local())) {
            return failure();
        }
    }
    auto row = [&](ArrayRef<Int> coefficients, bool equality) {
        Form form;
        form.constant = coefficients.back();
        for (unsigned i = 0; i < columns.size(); ++i) {
            form.terms[columns[i]] += coefficients[i];
        }
        target.constrain(form, equality);
    };
    for (unsigned i = 0; i < source.getNumEqualities(); ++i) {
        row(source.getEquality(i), true);
    }
    for (unsigned i = 0; i < source.getNumInequalities(); ++i) {
        row(source.getInequality(i), false);
    }
    return success();
}
bool role(const SymbolicPrimitive& value, SymbolicSchemaHandle schema, SymbolicTuple source, SymbolicTuple target)
{
    return value.valid() && value.schema() == schema && value.domain() == source && value.range() == target;
}
LogicalResult validateParameters(Builder& builder)
{
    for (Value parameter : builder.schema->parameters()) {
        auto argument = dyn_cast<BlockArgument>(parameter);
        if (argument && argument.getOwner() == &builder.function.getBody().front()) {
            continue;
        }
        auto* definition = parameter.getDefiningOp();
        if (!definition || !builder.function->isProperAncestor(definition)) {
            return builder.fail(
                StructuredImportStatus::InvalidInput, definition, "parameter belongs to another invocation", parameter);
        }
        // Being loop invariant alone is insufficient: the caller qualifies the
        // actual result as one invocation-immutable binding on the given context.
        for (auto* parent = definition->getParentOp(); parent != builder.function; parent = parent->getParentOp()) {
            if (isa<scf::ForOp, affine::AffineForOp>(parent)) {
                return builder.fail(
                    StructuredImportStatus::UnsupportedArithmetic, definition,
                    "loop-local SSA cannot be invocation parameter", parameter);
            }
        }
        if (!builder.options.immutableParameter ||
            !builder.options.immutableParameter(parameter, builder.schema, builder.effects.context)) {
            return builder.fail(
                StructuredImportStatus::MissingQualification, definition,
                "non-argument parameter needs immutable-result certificate", parameter);
        }
    }
    return success();
}
FailureOr<SymbolicInputs> construct(Builder& builder)
{
    using Tuple = SymbolicTuple;
    auto schema = builder.schema;
    auto present = Set::getEmpty(*schema->space(Tuple::Unit, Tuple::Occurrence));
    SmallVector<SmallVector<Poly, 0>> sitePresence;
    unsigned arity = *schema->arity(Tuple::Occurrence);
    SmallVector<unsigned> contextColumns;
    for (unsigned i = 0; i < schema->parameters().size(); ++i) {
        contextColumns.push_back(arity + i);
    }
    for (std::size_t site = 0; site < builder.sites.size(); ++site) {
        auto raw = builder.presence(site);
        if (failed(raw)) {
            return failure();
        }
        SmallVector<Poly, 0> guarded;
        for (const auto& branch : *raw) {
            for (const auto& context : builder.effects.context.relation().getAllDisjuncts()) {
                Poly piece = branch;
                if (failed(conjoin(piece, context, contextColumns))) {
                    return builder.fail(
                        StructuredImportStatus::RepresentationLimit, builder.function, "context column count overflow");
                }
                present.unionInPlace(piece.relation);
                guarded.push_back(std::move(piece));
            }
        }
        sitePresence.push_back(std::move(guarded));
    }
    auto reference = Set::getEmpty(*schema->space(Tuple::Occurrence, Tuple::Occurrence));
    SmallVector<unsigned> leftColumns, rightColumns;
    for (unsigned i = 0; i < arity; ++i) {
        leftColumns.push_back(i);
        rightColumns.push_back(arity + i);
    }
    for (unsigned i = 0; i < schema->parameters().size(); ++i) {
        leftColumns.push_back(2 * arity + i);
        rightColumns.push_back(2 * arity + i);
    }
    for (std::size_t a = 0; a < builder.sites.size(); ++a) {
        for (std::size_t b = 0; b < builder.sites.size(); ++b) {
            const auto& left = builder.sites[a];
            const auto& right = builder.sites[b];
            unsigned common = 0;
            while (common < left.loops.size() && common < right.loops.size() &&
                   left.loops[common] == right.loops[common]) {
                ++common;
            }
            for (const auto& source : sitePresence[a]) {
                for (const auto& target : sitePresence[b]) {
                    Poly prefix(*schema->space(Tuple::Occurrence, Tuple::Occurrence));
                    if (failed(conjoin(prefix, source.relation, leftColumns)) ||
                        failed(conjoin(prefix, target.relation, rightColumns))) {
                        return builder.fail(
                            StructuredImportStatus::RepresentationLimit, builder.function,
                            "reference column count overflow");
                    }
                    // Shared LOOP IDENTITIES, not equal depth/IV numbers. After
                    // their coordinates tie, lexical divergence in the original
                    // tree orders sibling loops, before/after sites, and phases.
                    for (unsigned coordinate = 0; coordinate < common; ++coordinate) {
                        Form difference = Form::axis(arity + 1 + coordinate).minus(Form::axis(1 + coordinate));
                        Poly less = prefix;
                        less.constrain(difference.plus(Form::number(Int(-1))));
                        reference.unionInPlace(less.relation);
                        prefix.constrain(difference, true);
                    }
                    if (a < b) {
                        reference.unionInPlace(prefix.relation);
                    }
                }
            }
        }
    }
    auto importedPresent = builder.import(Tuple::Unit, Tuple::Occurrence, present);
    auto importedReference = builder.import(Tuple::Occurrence, Tuple::Occurrence, reference);
    if (failed(importedPresent) || failed(importedReference)) {
        return failure();
    }
    return SymbolicInputs{builder.effects.context,    *importedPresent,       *importedReference,
                          builder.effects.reads,      builder.effects.writes, builder.effects.extras,
                          builder.effects.generators, builder.options.model};
}
} // namespace
} // namespace mlir::pto::frontiersynch::structured
namespace mlir::pto::frontiersynch {
FailureOr<std::shared_ptr<const StructuredImportContext>> StructuredImportContext::create(
    func::FuncOp function, const SyncInput& input, SymbolicSchemaHandle schema,
    std::shared_ptr<const PhaseIndex> phases)
{
    if (!function || !schema || schema->sites().size() != input.instructions().size()) { return failure(); }
    auto result = std::shared_ptr<StructuredImportContext>(new StructuredImportContext());
    result->function = function;
    result->input = &input;
    result->schema = std::move(schema);
    if (!phases) {
        auto created = std::make_shared<PhaseIndex>();
        if (failed(created->build(function, input))) { return failure(); }
        phases = std::move(created);
    }
    result->index = std::move(phases);
    llvm::DenseSet<const CompoundInstanceElement*> shared;
    shared.insert(input.instructions().begin(), input.instructions().end());
    for (auto [tag, site] : llvm::enumerate(result->schema->sites())) {
        if (!shared.contains(site.phase) ||
            !llvm::is_contained(result->index->phasesFor(site.phase->elementOp), site.phase) ||
            !result->phaseTags.try_emplace(site.phase, tag).second) {
            return failure();
        }
        SmallVector<Value> coordinates;
        for (Region* region : result->index->controlPath(*site.phase)) {
            if (auto loop = dyn_cast<scf::ForOp>(region->getParentOp())) {
                coordinates.push_back(loop.getInductionVar());
            } else if (auto loop = dyn_cast<affine::AffineForOp>(region->getParentOp())) {
                coordinates.push_back(loop.getInductionVar());
            }
        }
        if (coordinates != site.coordinates) { return failure(); }
        for (Operation* owner = site.phase->elementOp; owner; owner = owner->getParentOp()) {
            ++result->scopeCounts[owner];
            if (owner == function) { break; }
        }
    }
    return std::shared_ptr<const StructuredImportContext>(std::move(result));
}
bool StructuredImportContext::matches(
    func::FuncOp original, const SyncInput& shared, SymbolicSchemaHandle common) const
{
    return function == original && input == &shared && schema == common;
}
StructuredImportResult StructuredInputAdapter::build(
    func::FuncOp function, const SyncInput& input, const StructuredImportOptions& options,
    const ExactStructuredEffectsProvider& provider)
{
    using namespace structured;
    Builder builder(function, options);
    auto failedResult = [&]() { return StructuredImportResult{nullptr, builder.issue}; };
    if (!function || function.isExternal() || !provider || !function.getBody().hasOneBlock()) {
        (void)builder.fail(
            StructuredImportStatus::InvalidInput, function,
            "expected defined single-block function and effects provider");
        return failedResult();
    }
    if (options.scope && options.scope != function && !function->isProperAncestor(options.scope)) {
        (void)builder.fail(StructuredImportStatus::InvalidInput, options.scope, "scope belongs to another invocation");
        return failedResult();
    }
    if (options.context && !options.context->matches(function, input, options.schema)) {
        (void)builder.fail(StructuredImportStatus::InvalidInput, function, "common import context identity mismatch");
        return failedResult();
    }
    if ((!options.context && failed(builder.index.build(function, input))) ||
        failed(builder.discover(function.getBody(), {}, {}))) {
        if (builder.issue.status == StructuredImportStatus::Success) {
            (void)builder.fail(StructuredImportStatus::InvalidInput, function, "invalid shared phase index");
        }
        return failedResult();
    }
    auto expected = options.context ? options.context->phaseCount(options.scope ? options.scope : function) :
        llvm::count_if(input.instructions(), [&](const auto* phase) {
        return !options.scope || phase->elementOp == options.scope ||
               options.scope->isProperAncestor(phase->elementOp);
    });
    if (builder.sites.size() != static_cast<std::size_t>(expected)) {
        (void)builder.fail(StructuredImportStatus::InvalidInput, function, "not every shared phase was imported");
        return failedResult();
    }
    SmallVector<SymbolicSite> sites;
    for (const auto& site : builder.sites) {
        sites.push_back({site.phase, site.inductionVariables});
    }
    auto schema = options.schema ? FailureOr<SymbolicSchemaHandle>(options.schema) :
        SymbolicSchema::create(sites, options.parameters, options.cellAxes, options.physicalPartition);
    if (succeeded(schema)) {
        if ((*schema)->parameters() != ArrayRef<Value>(options.parameters) ||
            (*schema)->partition() != options.physicalPartition ||
            (*schema)->cellTypes() != ArrayRef<Type>(options.cellAxes)) {
            (void)builder.fail(StructuredImportStatus::InvalidInput, function, "scoped schema context mismatch");
            return failedResult();
        }
        if (options.context) {
            builder.sharedTags = &options.context->tags();
        } else {
            llvm::DenseSet<const CompoundInstanceElement*> sharedPhases;
            sharedPhases.insert(input.instructions().begin(), input.instructions().end());
            for (auto [tag, original] : llvm::enumerate((*schema)->sites())) {
                if (!sharedPhases.contains(original.phase) || !original.phase->elementOp ||
                    !function->isProperAncestor(original.phase->elementOp) ||
                    !builder.siteTags.try_emplace(original.phase, tag).second) {
                    (void)builder.fail(
                        StructuredImportStatus::InvalidInput, function, "schema site ownership mismatch");
                    return failedResult();
                }
            }
            for (const auto& site : sites) {
                auto found = builder.siteTags.find(site.phase);
                if (found == builder.siteTags.end() ||
                    (*schema)->sites()[found->second].coordinates != site.coordinates) {
                    (void)builder.fail(StructuredImportStatus::InvalidInput, function, "scoped schema site mismatch");
                    return failedResult();
                }
            }
        }
    }
    if (failed(schema)) {
        (void)builder.fail(StructuredImportStatus::InvalidInput, function, "invalid occurrence schema");
        return failedResult();
    }
    builder.schema = *schema;
    auto effects = provider(*schema, builder.sites);
    using Tuple = SymbolicTuple;
    if (failed(effects)) {
        (void)builder.fail(
            StructuredImportStatus::UnsupportedEffects, function, "exact effects provider declined input");
        return failedResult();
    }
    if (!role(effects->context, *schema, Tuple::Unit, Tuple::Unit) ||
        !role(effects->reads, *schema, Tuple::Occurrence, Tuple::Cell) ||
        !role(effects->writes, *schema, Tuple::Occurrence, Tuple::Cell) ||
        !role(effects->extras, *schema, Tuple::Occurrence, Tuple::Occurrence) ||
        (effects->generators && !role(*effects->generators, *schema, Tuple::Occurrence, Tuple::Occurrence))) {
        (void)builder.fail(
            StructuredImportStatus::InvalidInput, function, "provider relation schema/partition/tuple mismatch");
        return failedResult();
    }
    builder.effects = std::move(*effects);
    if (failed(validateParameters(builder))) {
        return failedResult();
    }
    // Phase-free loops still precede later events. Certify their terminating
    // progression too; absence of payload phases does not erase control effects.
    for (Operation* operation : builder.allLoops) {
        Int step(0);
        if (auto loop = dyn_cast<scf::ForOp>(operation)) {
            auto constant = loop.getStep().getDefiningOp<arith::ConstantOp>();
            auto attr = constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr();
            if (attr) {
                step = integer(attr.getValue());
            }
        } else {
            step = integer(cast<affine::AffineForOp>(operation).getStep());
        }
        if (step <= 0) {
            (void)builder.fail(
                StructuredImportStatus::UnsupportedControl, operation, "all loops require positive constant steps");
            return failedResult();
        }
        if (!builder.qualify(operation)) {
            return failedResult();
        }
    }
    auto relations = construct(builder);
    if (failed(relations)) {
        return failedResult();
    }
    auto result = std::shared_ptr<StructuredInput>(new StructuredInput());
    result->owner = *schema;
    result->inputs = std::move(*relations);
    result->phaseSites = std::move(builder.sites);
    return {StructuredInputHandle(result), {}};
}
} // namespace mlir::pto::frontiersynch
