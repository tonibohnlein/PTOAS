// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "DirectEmissionInternal.h"
#include "PTO/Transforms/InsertSync/SyncInput.h"
#include "StructuredInternal.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

namespace mlir::pto::frontiersynch {
namespace {
using Int = llvm::DynamicAPInt;
using Tuple = SymbolicTuple;
using Set = presburger::PresburgerRelation;
using Row = SmallVector<Int>;
struct Interval {
    Int lower, upper;
};
Interval typeRange(Type type, const DataLayout& layout)
{
    if (auto integer = dyn_cast<IntegerType>(type)) {
        if (integer.getWidth() == 1 || integer.isUnsigned()) {
            return {Int(0), structured::integer(APInt::getMaxValue(integer.getWidth()), true)};
        }
    }
    unsigned width = layout.getTypeSizeInBits(type).getFixedValue();
    return {structured::integer(APInt::getSignedMinValue(width)), structured::integer(APInt::getSignedMaxValue(width))};
}
bool fits(const Interval& values, const Interval& allowed)
{
    return allowed.lower <= values.lower && values.upper <= allowed.upper;
}
// Qualify only this optional executable-endpoint route, never the F* dump.
// MLIR's fixed-point analysis supplies sound ranges; unknown/wrapped ranges do
// not certify mathematical interpretation. No range is used to kill a conflict.
class ArithmeticQualification {
public:
    ArithmeticQualification(DataFlowSolver& solver, const DataLayout& layout) : solver(solver), layout(layout) {}
    std::optional<Interval> range(Value value) const
    {
        const auto* state = solver.lookupState<dataflow::IntegerValueRangeLattice>(value);
        if (!state || state->getValue().isUninitialized()) {
            return std::nullopt;
        }
        const auto& values = state->getValue().getValue();
        auto integer = dyn_cast<IntegerType>(value.getType());
        bool unsignedValue = integer && (integer.isUnsigned() || integer.getWidth() == 1);
        Interval result{
            structured::integer(unsignedValue ? values.umin() : values.smin(), unsignedValue),
            structured::integer(unsignedValue ? values.umax() : values.smax(), unsignedValue)};
        return fits(result, typeRange(value.getType(), layout)) ? std::optional<Interval>(result) : std::nullopt;
    }
    bool operator()(Operation* operation) const
    {
        if (auto loop = dyn_cast<scf::ForOp>(operation)) {
            if (loop.getInductionVar().getType().isInteger(1) || operation->hasAttr("unsignedCmp")) {
                return false;
            }
            auto lower = range(loop.getLowerBound()), upper = range(loop.getUpperBound()), step = range(loop.getStep());
            if (!lower || !upper || !step || step->lower != step->upper || step->lower <= 0) {
                return false;
            }
            auto allowed = typeRange(loop.getInductionVar().getType(), layout);
            return upper->upper + step->upper - 1 <= allowed.upper;
        }
        if (isa<arith::MinSIOp, arith::MaxSIOp>(operation)) {
            return bool(range(operation->getOperand(0))) && bool(range(operation->getOperand(1)));
        }
        if (isa<arith::IndexCastOp, arith::ExtSIOp, arith::TruncIOp>(operation)) {
            auto source = range(operation->getOperand(0));
            return source && fits(*source, typeRange(operation->getResult(0).getType(), layout));
        }
        if (!isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(operation)) {
            return false;
        }
        auto a = range(operation->getOperand(0)), b = range(operation->getOperand(1));
        if (!a || !b) {
            return false;
        }
        Interval result{Int(0), Int(0)};
        if (isa<arith::AddIOp>(operation)) {
            result = {a->lower + b->lower, a->upper + b->upper};
        } else if (isa<arith::SubIOp>(operation)) {
            result = {a->lower - b->upper, a->upper - b->lower};
        } else {
            for (const auto& x : {a->lower, a->upper}) {
                for (const auto& y : {b->lower, b->upper}) {
                    Int product = x * y;
                    // Including zero only enlarges the proof interval.
                    result.lower = std::min(result.lower, product);
                    result.upper = std::max(result.upper, product);
                }
            }
        }
        return fits(result, typeRange(operation->getResult(0).getType(), layout));
    }

private:
    DataFlowSolver& solver;
    const DataLayout& layout;
};
FailureOr<ExactStructuredEffects> modeledRequirements(
    const TraceDemandAnalysis& trace, const DataLayout& layout, SymbolicSchemaHandle schema,
    ArrayRef<StructuredSite> sites,
    const DenseMap<const CompoundInstanceElement*, std::size_t>& traceSites,
    const DenseMap<const CompoundInstanceElement*, std::size_t>& endpointSites)
{
    auto import = [&](Tuple a, Tuple b, const Set& relation) {
        return SymbolicPrimitive::import(schema, a, b, schema->parameters(), relation);
    };
    presburger::IntegerRelation admitted(*schema->space(Tuple::Unit, Tuple::Unit));
    for (auto [i, parameter] : llvm::enumerate(schema->parameters())) {
        auto bounds = typeRange(parameter.getType(), layout);
        Row lower(admitted.getNumCols(), Int(0)), upper = lower;
        lower[i] = Int(1);
        lower.back() = -bounds.lower;
        upper[i] = Int(-1);
        upper.back() = bounds.upper;
        admitted.addInequality(lower);
        admitted.addInequality(upper);
    }
    auto contextSet = Set::getEmpty(admitted.getSpace());
    contextSet.unionInPlace(admitted);
    auto context = import(Tuple::Unit, Tuple::Unit, contextSet);
    auto emptyEffects =
        import(Tuple::Occurrence, Tuple::Cell, Set::getEmpty(*schema->space(Tuple::Occurrence, Tuple::Cell)));
    auto emptyExtras = import(
        Tuple::Occurrence, Tuple::Occurrence, Set::getEmpty(*schema->space(Tuple::Occurrence, Tuple::Occurrence)));
    auto generators = Set::getEmpty(*schema->space(Tuple::Occurrence, Tuple::Occurrence));
    unsigned arity = *schema->arity(Tuple::Occurrence);
    // Include self-site and reverse static pairs: strict source occurrence order
    // is supplied by MLIR control, not by static instruction indices. Presence
    // and reference restrict this G before lifetime/reachability reduction.
    llvm::SmallPtrSet<const CompoundInstanceElement*, 16> active;
    for (const auto& site : sites) { active.insert(site.phase); }
    for (auto [b, consumer] : llvm::enumerate(sites)) {
        auto found = traceSites.find(consumer.phase);
        if (found == traceSites.end()) {
            return failure();
        }
        // Reuse G from the shared demand analysis: do not rescan dependencies
        // or independently reinterpret the memory analyzer's qualifications.
        for (std::size_t source : trace.conflicts()[found->second]) {
            auto endpoint = endpointSites.find(trace.sites()[source].phase);
            if (endpoint == endpointSites.end()) { return failure(); }
            if (!active.contains(trace.sites()[source].phase)) { continue; }
            presburger::IntegerRelation piece(generators.getSpace());
            Row left(piece.getNumCols(), Int(0)), right = left;
            left[0] = Int(1);
            left.back() = -Int(static_cast<int64_t>(endpoint->second));
            right[arity] = Int(1);
            right.back() = -Int(static_cast<int64_t>(endpointSites.lookup(consumer.phase)));
            piece.addEquality(left);
            piece.addEquality(right);
            generators.unionInPlace(piece);
        }
    }
    auto importedG = import(Tuple::Occurrence, Tuple::Occurrence, generators);
    if (failed(context) || failed(emptyEffects) || failed(emptyExtras) || failed(importedG)) {
        return failure();
    }
    return ExactStructuredEffects{*context, *emptyEffects, *emptyEffects, *emptyExtras, *importedG};
}
} // namespace
struct EndpointRelationImporter::Impl {
    func::FuncOp function;
    const SyncInput& input;
    const TraceDemandAnalysis& trace;
    DataFlowSolver solver;
    SymbolicSchemaHandle schema;
    std::shared_ptr<const StructuredImportContext> context;
    DenseMap<const CompoundInstanceElement*, std::size_t> traceSites;
    bool initialized = false, valid = false;
    Impl(func::FuncOp function, const SyncInput& input, const TraceDemandAnalysis& trace,
         std::shared_ptr<const PhaseIndex> phases)
        : function(function), input(input), trace(trace)
    {
        SmallVector<SymbolicSite> sites;
        SmallVector<Value> parameters;
        for (auto [id, site] : llvm::enumerate(trace.sites())) {
            SmallVector<Value> coordinates;
            for (Operation* operation : site.loops) {
                if (auto loop = dyn_cast<scf::ForOp>(operation)) {
                    coordinates.push_back(loop.getInductionVar());
                } else if (auto loop = dyn_cast<affine::AffineForOp>(operation)) {
                    coordinates.push_back(loop.getInductionVar());
                }
            }
            sites.push_back({site.phase, std::move(coordinates)});
            traceSites[site.phase] = id;
        }
        for (Value argument : function.getArguments()) {
            if (isa<IntegerType, IndexType>(argument.getType())) { parameters.push_back(argument); }
        }
        auto created = SymbolicSchema::create(sites, parameters, {}, 0);
        if (succeeded(created)) {
            schema = *created;
            auto prepared = StructuredImportContext::create(function, input, schema, std::move(phases));
            if (succeeded(prepared)) { context = *prepared; }
        }
    }
};
EndpointRelationImporter::EndpointRelationImporter(
    func::FuncOp function, const SyncInput& input, const TraceDemandAnalysis& trace,
    std::shared_ptr<const PhaseIndex> phases)
    : implementation(std::make_unique<Impl>(function, input, trace, std::move(phases))) {}
EndpointRelationImporter::~EndpointRelationImporter() = default;
FailureOr<StructuredInputHandle> EndpointRelationImporter::build(Operation* scope, std::string& reason)
{
    auto& state = *implementation;
    auto function = state.function;
    const auto& input = state.input;
    const auto& trace = state.trace;
    auto& solver = state.solver;
    if (!state.initialized) {
        state.initialized = true;
        solver.load<dataflow::DeadCodeAnalysis>();
        solver.load<dataflow::SparseConstantPropagation>();
        solver.load<dataflow::IntegerRangeAnalysis>();
        state.valid = state.context && succeeded(solver.initializeAndRun(function));
    }
    if (!state.valid) {
        reason = "integer range analysis or source schema did not establish endpoint arithmetic";
        return failure();
    }
    DataLayout layout = DataLayout::closest(function);
    ArithmeticQualification qualify(solver, layout);
    StructuredImportOptions options;
    options.model = RequirementModel::SharedModeled;
    options.scope = scope == function ? nullptr : scope;
    options.schema = state.schema;
    options.context = state.context;
    for (Value argument : function.getArguments()) {
        if (isa<IntegerType, IndexType>(argument.getType())) {
            options.parameters.push_back(argument);
        }
    }
    options.phaseOrder =
        [](Operation*,
           ArrayRef<const CompoundInstanceElement*> phases) -> FailureOr<SmallVector<const CompoundInstanceElement*>> {
        return SmallVector<const CompoundInstanceElement*>(phases.begin(), phases.end());
    };
    options.mathematicalArithmetic = [&](Operation* operation, SymbolicSchemaHandle, const SymbolicPrimitive&) {
        return qualify(operation);
    };
    auto imported = StructuredInputAdapter::build(
        function, input, options, [&](SymbolicSchemaHandle schema, ArrayRef<StructuredSite> sites) {
            return modeledRequirements(
                trace, layout, std::move(schema), sites, state.traceSites, state.context->tags());
        });
    if (!imported.succeeded()) {
        reason = imported.issue.reason;
        return failure();
    }
    // Logical coordinate keys use the target index representation. Narrower
    // signed induction values embed injectively; wider values require a future
    // key representation rather than silently truncating occurrence identities.
    unsigned indexBits = layout.getTypeSizeInBits(IndexType::get(function.getContext())).getFixedValue();
    for (const auto& site : imported.input->sites()) {
        for (Value iv : site.inductionVariables) {
            if (auto integer = dyn_cast<IntegerType>(iv.getType())) {
                if (integer.getWidth() > indexBits) {
                    reason = "loop coordinates exceed the logical index-key representation";
                    return failure();
                }
            }
        }
    }
    return imported.input;
}
} // namespace mlir::pto::frontiersynch
