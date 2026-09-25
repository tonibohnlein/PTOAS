// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_ORIGINALVALUEQUERIES_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_ORIGINALVALUEQUERIES_H

#include "PTO/Transforms/FrontierSynch/OriginalValueArithmetic.h"
#include "PTO/Transforms/FrontierSynch/ReaderFrontiers.h"
#include "PTO/Transforms/FrontierSynch/SyncSlotMapping.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <tuple>

namespace mlir::pto::frontiersynch {

// These cuts use originalSites, NOT translated phase IDs. After an scf.if or
// scf.for denotes its join/exit. A phase adapter must first check the original
// instruction's beforeExecutable/afterExecutable flags. No selected-word gap
// or selected completion enters this service.
struct OriginalValueCut {
    std::size_t site = NoControlId;
    bool after = false;
    bool operator==(const OriginalValueCut& other) const { return site == other.site && after == other.after; }
};
struct OriginalValueIdentity {
    Value value;
    // Innermost original repeat that defines this value; NoControlId is the
    // invocation. Capturing a value in another loop does not redefine it there.
    std::size_t occurrenceScope = NoControlId;
};
struct OriginalValueReference {
    Value value;
    std::size_t occurrenceScope = NoControlId;
    int64_t visitOffset = 0;
};
struct OriginalValueCondition {
    OriginalValueIdentity identity;
    bool positive = true;
};
struct OriginalCompletionPrerequisite {
    Value producedValue;
    std::size_t sourcePhase = NoControlId;
    std::size_t executableSourcePhase = NoControlId;
    OriginalValueCut source, deadline;
    PipelineType sourceEngine = PipelineType::PIPE_UNASSIGNED;
    // Original lexical control, not the unavailable endpoint predicate. The
    // prerequisite must be realized inside this control before its exact use.
    std::vector<OriginalValueCondition> applicability;
    OriginalValueReference sourceValue;
};
struct OriginalValueRecipe {
    enum Kind {
        OriginalSSA,
        LoopNonEmpty,
        LoopHasPrevious,
        LoopHasNext,
        LoopResidue,
        Minimum,
        Maximum,
        GuardedLast,
        // Composed I.2 recipes are interpreted by ExactFrontiers.h. Their
        // bounds/conditions remain original SSA references, not new counters.
        IntervalNonEmpty,
        IntervalFirst,
        IntervalLast
    } kind = OriginalSSA;
    std::vector<Value> operands;
    unsigned width = 0;
    bool unsignedComparison = false;
    uint64_t parameter = 0, expected = 1;
    // GuardedLast is if(lower < upper) upper-1, not select(lower < upper,
    // eagerly-computed upper-1, ...). Its empty arm evaluates no subtraction.
};
struct OriginalValueQualification {
    enum class Status { Available, NeedsCompletion, NotObservableHere, Unresolved };
    Status status = Status::Unresolved;
    std::vector<OriginalCompletionPrerequisite> prerequisites;
    std::vector<OriginalValueIdentity> values;
    std::vector<std::string> obstructions;
    OriginalValueRecipe recipe;
    OriginalValueReference reference;
    // An original SSA representative, never a recomputation with changed inputs.
    Value representative;
    bool available() const { return status == Status::Available; }
    bool executableAfterPrerequisites() const { return available() || status == Status::NeedsCompletion; }
    std::string reason() const { return obstructions.empty() ? std::string{} : obstructions.front(); }
};

// One immutable original-program service. ProgramAnalysis shares its instance
// with occurrence, reader-boundary and factored-guard clients. Standalone
// clients may own an instance of the same service. No IR is folded or rewritten.
// Index width is explicit; 64 is the existing SyncSlotMapping target contract.
class OriginalValueQueries {
public:
    using Result = OriginalValueQualification;
    using Status = Result::Status;
    explicit OriginalValueQueries(
        const OriginalStructure& structure, unsigned indexWidth = 64, uint64_t originalVersion = 1)
        : original(structure),
          dominance(structure.function),
          indexWidth(indexWidth),
          originalVersion(originalVersion),
          snapshot(structure.version)
    {
        for (auto [site, operation] : llvm::enumerate(original.originalSites)) {
            if (operation) {
                sites.try_emplace(operation, site);
            }
        }
        for (auto [phase, operation] : llvm::enumerate(original.operations)) {
            if (operation.instruction && operation.instruction->elementOp) {
                phases[operation.instruction->elementOp].push_back(phase);
            }
        }
        for (auto [site, operation] : llvm::enumerate(original.originalSites)) {
            auto choice = dyn_cast_or_null<scf::IfOp>(operation);
            if (!choice) {
                continue;
            }
            const auto id = identity(choice.getCondition());
            const auto key = std::make_pair(valueKey(id.value), id.occurrenceScope);
            auto inserted = guardOwners.emplace(key, site);
            canonicalGuards[site] = inserted.first->second;
        }
    }
    OriginalValueQueries(const OriginalValueQueries&) = delete;
    OriginalValueQueries& operator=(const OriginalValueQueries&) = delete;

    bool current() const { return original.version == snapshot; }

    OriginalValueIdentity identity(Value value) const
    {
        if (!current()) {
            return {};
        }
        // Only an actual unchanged carried value is an alias of its initial value.
        // No arithmetic folder is allowed to replace old values by new operands.
        llvm::DenseSet<Value> seen;
        while (value && seen.insert(value).second) {
            auto argument = dyn_cast<BlockArgument>(value);
            auto loop = argument ? dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp()) : scf::ForOp{};
            if (!loop || argument.getOwner() != loop.getBody() || !argument.getArgNumber()) {
                break;
            }
            const auto position = argument.getArgNumber() - 1;
            auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
            if (!yield || yield.getOperand(position) != value) {
                break;
            }
            value = loop.getInitArgs()[position];
        }
        mlir::Operation* scope = nullptr;
        if (auto argument = dyn_cast_if_present<BlockArgument>(value)) {
            scope = argument.getOwner()->getParentOp();
        } else if (value && value.getDefiningOp()) {
            scope = value.getDefiningOp()->getParentOp();
        }
        while (scope && !isa<scf::ForOp, scf::WhileOp>(scope)) {
            scope = scope->getParentOp();
        }
        return {value, siteOf(scope)};
    }
    std::size_t canonicalGuardOwner(std::size_t owner) const
    {
        const auto found = canonicalGuards.find(owner);
        return found == canonicalGuards.end() ? owner : found->second;
    }
    std::size_t siteOf(mlir::Operation* operation) const
    {
        const auto found = sites.find(operation);
        return found == sites.end() ? NoControlId : found->second;
    }
    OriginalValueCut before(mlir::Operation* operation) const { return {siteOf(operation), false}; }
    OriginalValueCut after(mlir::Operation* operation) const { return {siteOf(operation), true}; }
    OriginalValueCut phaseCut(std::size_t phase, bool after) const
    {
        if (phase >= original.operations.size()) {
            return {};
        }
        const auto& operation = original.operations[phase];
        if (after ? !operation.afterExecutable : !operation.beforeExecutable) {
            return {};
        }
        return {operation.original, after};
    }
    bool legal(OriginalValueCut cut) const
    {
        if (!current()) {
            return false;
        }
        if (cut.site >= original.originalSites.size() || !original.originalSites[cut.site]) {
            return false;
        }
        return !cut.after || !original.originalSites[cut.site]->hasTrait<OpTrait::IsTerminator>();
    }

    Result qualify(Value value, OriginalValueCut cut) const
    {
        const auto id = identity(value);
        return qualify({value, id.occurrenceScope, 0}, cut);
    }
    Result qualify(OriginalValueReference reference, OriginalValueCut cut) const
    {
        if (!current()) {
            return obstruction(Status::Unresolved, "original program changed; rebuild value queries");
        }
        const auto id = identity(reference.value);
        if (reference.occurrenceScope != id.occurrenceScope) {
            return obstruction(Status::Unresolved, "value occurrence scope does not match its original definition");
        }
        const auto key = std::make_tuple(
            valueKey(reference.value), reference.occurrenceScope, reference.visitOffset, cut.site, cut.after,
            originalVersion);
        const auto found = cache.find(key);
        if (found != cache.end()) {
            return found->second;
        }
        if (!active.insert(key).second) {
            return obstruction(Status::Unresolved, "scalar recurrence needs an original carried-value interface");
        }
        ++evaluations;
        auto result = reference.visitOffset ? previousValue(reference, cut) : qualifyImpl(reference.value, cut);
        result.reference = reference;
        if (!result.representative && !reference.visitOffset) {
            result.representative = reference.value;
        }
        active.erase(key);
        normalize(result);
        cache.emplace(key, result);
        return result;
    }

    // Check explicitly supplied enabling guards on a completion prerequisite.
    // In particular qualifyEnabled(g, cut, {g}) cannot manufacture a legal
    // guard-enabling transfer. This check is independent of resource binding.
    Result qualifyEnabled(Value value, OriginalValueCut cut, ArrayRef<Value> enablingGuards) const
    {
        auto result = qualify(value, cut);
        for (const auto& prerequisite : result.prerequisites) {
            for (Value guard : enablingGuards) {
                if (dependsOn(guard, prerequisite.producedValue)) {
                    return combine(
                        std::move(result),
                        obstruction(
                            Status::Unresolved,
                            "completion prerequisite is circularly guarded by the value it enables"));
                }
            }
        }
        for (Value guard : enablingGuards) {
            // Requiring Available is deliberate. A chain of guard-enabling packets
            // needs its own already ordered support, not mutually assumed receipts.
            auto guardResult = qualify(guard, cut);
            if (!guardResult.available()) {
                guardResult = combine(
                    std::move(guardResult),
                    obstruction(
                        Status::Unresolved, "enabling guard is not independently available before this prerequisite"));
            }
            result = combine(std::move(result), guardResult);
        }
        return result;
    }

    // This checks endpoint availability separately from original factored meaning.
    Result atom(const ObservationAtom& observation, OriginalValueCut cut) const
    {
        if (observation.owner >= original.originalSites.size()) {
            return obstruction(Status::Unresolved, "observation has no original owner");
        }
        auto* owner = original.originalSites[observation.owner];
        if (observation.kind == ObservationAtom::OriginalBoolean) {
            auto choice = dyn_cast_or_null<scf::IfOp>(owner);
            if (!choice || !choice.getCondition().getType().isInteger(1) || observation.value > 1) {
                return obstruction(Status::Unresolved, "observation is not a qualified original Boolean");
            }
            auto result = qualify(identity(choice.getCondition()).value, cut);
            result.recipe.expected = observation.value;
            return result;
        }
        auto loop = dyn_cast_or_null<scf::ForOp>(owner);
        if (!loop) {
            return obstruction(Status::Unresolved, "observation does not name a counted original loop");
        }
        auto result = counted(loop);
        for (Value bound : {loop.getLowerBound(), loop.getUpperBound(), loop.getStep()}) {
            result = combine(std::move(result), qualify(bound, cut));
        }
        const auto integer = loopInteger(loop);
        result.recipe.width = integer.width;
        result.recipe.unsignedComparison = integer.isUnsigned;
        result.recipe.operands = {loop.getLowerBound(), loop.getUpperBound(), loop.getStep()};
        result.recipe.expected = observation.value;
        if (observation.kind == ObservationAtom::LoopNonEmpty) {
            result.recipe.kind = OriginalValueRecipe::LoopNonEmpty;
            if (observation.value > 1) {
                return combine(
                    std::move(result),
                    obstruction(Status::Unresolved, "loop participation observation is not Boolean"));
            }
            return result;
        }
        if (!legal(cut) || !loop->isProperAncestor(original.originalSites[cut.site])) {
            return combine(
                std::move(result),
                obstruction(
                    Status::NotObservableHere, "original induction value is not observable outside this loop body"));
        }
        result = combine(std::move(result), qualify(loop.getInductionVar(), cut));
        result.recipe.operands.push_back(loop.getInductionVar());
        result.recipe.parameter = observation.parameter;
        if (!observation.parameter || observation.parameter > integer.mask()) {
            return combine(
                std::move(result),
                obstruction(
                    Status::Unresolved, "observation distance/modulus does not fit the original integer width"));
        }
        if (observation.kind == ObservationAtom::LoopHasPrevious) {
            result.recipe.kind = OriginalValueRecipe::LoopHasPrevious;
        } else if (observation.kind == ObservationAtom::LoopHasNext) {
            result.recipe.kind = OriginalValueRecipe::LoopHasNext;
        } else {
            return combine(std::move(result), obstruction(Status::Unresolved, "unsupported original loop observation"));
        }
        if (observation.kind != ObservationAtom::LoopResidue && observation.value > 1) {
            return combine(
                std::move(result), obstruction(Status::Unresolved, "loop participation observation is not Boolean"));
        }
        return result;
    }

    Result predicate(
        std::size_t root, const std::function<ParticipationExpression(std::size_t)>& get, OriginalValueCut cut) const
    {
        std::map<std::size_t, Result> memo;
        std::set<std::size_t> inProgress;
        std::function<Result(std::size_t)> visit = [&](std::size_t id) -> Result {
            auto found = memo.find(id);
            if (found != memo.end()) {
                return found->second;
            }
            if (!inProgress.insert(id).second) {
                return obstruction(Status::Unresolved, "cyclic endpoint predicate expression");
            }
            auto expression = get(id);
            Result result = obstruction(Status::Unresolved, "invalid endpoint predicate expression");
            if (expression.kind == ParticipationExpression::False || expression.kind == ParticipationExpression::True) {
                result = legal(cut) ? availableResult() :
                                      obstruction(Status::NotObservableHere, "endpoint is not a legal original cut");
            } else if (expression.kind == ParticipationExpression::Atom) {
                result = atom(expression.atom, cut);
            } else if (expression.kind == ParticipationExpression::Not) {
                result = visit(expression.left);
            } else if (
                expression.kind == ParticipationExpression::And || expression.kind == ParticipationExpression::Or) {
                // And/Or are eager Boolean combinations, NOT control-flow Choose.
                // Existing constant folding can remove a dead operand before here.
                result = combine(visit(expression.left), visit(expression.right));
            }
            inProgress.erase(id);
            memo.emplace(id, result);
            return result;
        };
        return visit(root);
    }

    // Arithmetic portion of I.2, usable before the interval/frontier query is
    // implemented. It supplies an explicitly conditional recipe, not an endpoint
    // or a claim that an arbitrary lower/upper expression denotes participation.
    Result guardedLast(Value lower, Value upper, OriginalValueCut cut, bool unsignedComparison) const
    {
        auto result = combine(qualify(lower, cut), qualify(upper, cut));
        if (!lower || !upper || lower.getType() != upper.getType() || !width(lower.getType())) {
            return combine(
                std::move(result),
                obstruction(Status::Unresolved, "interval bounds have different or unsupported integer types"));
        }
        result.recipe = {
            OriginalValueRecipe::GuardedLast, {lower, upper}, width(lower.getType()), unsignedComparison, 0, 1};
        return result;
    }
    Result extremum(ArrayRef<Value> bounds, OriginalValueCut cut, bool minimum, bool unsignedComparison) const
    {
        if (bounds.empty() || !bounds.front()) {
            return obstruction(Status::Unresolved, "empty scalar extremum");
        }
        auto result = availableResult();
        for (Value bound : bounds) {
            result = combine(std::move(result), qualify(bound, cut));
            if (!bound || bound.getType() != bounds.front().getType()) {
                result = combine(
                    std::move(result),
                    obstruction(Status::Unresolved, "extremum operands use different integer types"));
            }
        }
        result.recipe.kind = minimum ? OriginalValueRecipe::Minimum : OriginalValueRecipe::Maximum;
        result.recipe.operands.assign(bounds.begin(), bounds.end());
        result.recipe.width = width(bounds.front().getType());
        result.recipe.unsignedComparison = unsignedComparison;
        return result;
    }

    // A proof of the counted arithmetic, not proof of invariant participation,
    // physical permutation, event matching, or selected completion.
    Result counted(scf::ForOp loop) const
    {
        if (!current()) {
            return obstruction(Status::Unresolved, "original program changed; rebuild value queries");
        }
        const auto found = loops.find(loop.getOperation());
        if (found != loops.end()) {
            return found->second;
        }
        auto result = availableResult();
        const auto cut = before(loop.getOperation());
        for (Value value : {loop.getLowerBound(), loop.getUpperBound(), loop.getStep()}) {
            result = combine(std::move(result), qualify(value, cut));
        }
        const auto type = loopInteger(loop);
        const auto step = constantBits(loop.getStep());
        bool safe = false;
        if (type.valid() && step && *step && *step <= ((uint64_t(1) << (type.width - 1)) - 1)) {
            const auto lower = constantBits(loop.getLowerBound());
            const auto upper = constantBits(loop.getUpperBound());
            if (lower && upper) {
                safe = value_arithmetic::countedLoop(type, *lower, *upper, *step);
            } else if (*step == 1) {
                // On a nonempty visit the final IV becomes UB, which is representable.
                // On a zero-trip visit there is no arithmetic update.
                safe = true;
            } else {
                const auto bounds = integerRange(loop.getUpperBound());
                const auto maximumUpper = type.isUnsigned ? bounds.umax().getZExtValue() : bounds.smax().getZExtValue();
                const auto room =
                    value_arithmetic::checked(value_arithmetic::Binary::Subtract, type, type.maximum(), *step - 1);
                safe = room && !type.less(*room, maximumUpper);
            }
        }
        if (!safe) {
            result = combine(
                std::move(result),
                obstruction(
                    Status::Unresolved,
                    "counted bounds/positive step/final increment lack a width-correct no-overflow proof"));
        }
        loops.emplace(loop.getOperation(), result);
        return result;
    }

    static Result combine(Result a, const Result& b)
    {
        auto rank = [](Status status) {
            return status == Status::NotObservableHere ? 3 :
                   status == Status::Unresolved        ? 2 :
                   status == Status::NeedsCompletion   ? 1 :
                                                         0;
        };
        if (rank(b.status) > rank(a.status)) {
            a.status = b.status;
        }
        a.prerequisites.insert(a.prerequisites.end(), b.prerequisites.begin(), b.prerequisites.end());
        a.values.insert(a.values.end(), b.values.begin(), b.values.end());
        a.obstructions.insert(a.obstructions.end(), b.obstructions.begin(), b.obstructions.end());
        normalize(a);
        return a;
    }
    uint64_t evaluationCount() const { return evaluations; }

private:
    using Key = std::tuple<uintptr_t, std::size_t, int64_t, std::size_t, bool, uint64_t>;
    static uintptr_t valueKey(Value value) { return reinterpret_cast<uintptr_t>(value.getAsOpaquePointer()); }
    static Result availableResult()
    {
        Result result;
        result.status = Status::Available;
        return result;
    }
    static Result obstruction(Status status, std::string message)
    {
        Result result;
        result.status = status;
        result.obstructions.push_back(std::move(message));
        return result;
    }
    static void normalize(Result& result)
    {
        std::set<std::tuple<std::size_t, std::size_t, bool, uintptr_t>> seenPrerequisites;
        result.prerequisites.erase(
            std::remove_if(
                result.prerequisites.begin(), result.prerequisites.end(),
                [&](const OriginalCompletionPrerequisite& p) {
                    return !seenPrerequisites
                                .emplace(p.sourcePhase, p.deadline.site, p.deadline.after, valueKey(p.producedValue))
                                .second;
                }),
            result.prerequisites.end());
        std::set<std::pair<uintptr_t, std::size_t>> seenValues;
        result.values.erase(
            std::remove_if(
                result.values.begin(), result.values.end(),
                [&](const OriginalValueIdentity& v) {
                    return !seenValues.emplace(valueKey(v.value), v.occurrenceScope).second;
                }),
            result.values.end());
        std::set<std::string> seenObstructions;
        result.obstructions.erase(
            std::remove_if(
                result.obstructions.begin(), result.obstructions.end(),
                [&](const std::string& s) { return !seenObstructions.insert(s).second; }),
            result.obstructions.end());
    }
    unsigned width(Type type) const
    {
        if (isa<IndexType>(type)) {
            return indexWidth == 64 ? 64 : 0; // same contract as imported address facts
        }
        auto integer = dyn_cast<IntegerType>(type);
        return integer && integer.getWidth() <= 64 ? integer.getWidth() : 0;
    }
    value_arithmetic::Integer loopInteger(scf::ForOp loop) const
    {
        return {width(loop.getInductionVar().getType()), loop->hasAttr("unsignedCmp") || loop->hasAttr("unsigned_cmp")};
    }
    std::optional<uint64_t> constantBits(Value value) const
    {
        auto constant = value.getDefiningOp<arith::ConstantOp>();
        auto attribute = constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr{};
        if (!attribute || !width(value.getType()) || attribute.getValue().getBitWidth() > 64) {
            return {};
        }
        return attribute.getValue().getZExtValue();
    }
    ConstantIntRanges integerRange(Value value) const { return SyncSlotMapping::range(value, ranges); }
    bool observable(Value value, OriginalValueCut cut) const
    {
        if (!value || !legal(cut)) {
            return false;
        }
        auto* at = original.originalSites[cut.site];
        auto* definition = value.getDefiningOp();
        if (!definition) {
            definition = cast<BlockArgument>(value).getOwner()->getParentOp();
        }
        mlir::Operation* function = original.function;
        if (!function || !definition || (definition != function && !function->isProperAncestor(definition))) {
            return false;
        }
        if (value.getDefiningOp() == at) {
            return cut.after;
        }
        if (value.getDefiningOp() && definition->isProperAncestor(at)) {
            return false;
        }
        return dominance.dominates(value, at);
    }
    std::vector<OriginalValueCondition> controlAt(OriginalValueCut cut) const
    {
        std::vector<OriginalValueCondition> result;
        if (!legal(cut)) {
            return result;
        }
        auto* child = original.originalSites[cut.site];
        for (auto* parent = child->getParentOp(); parent; child = parent, parent = parent->getParentOp()) {
            if (auto choice = dyn_cast<scf::IfOp>(parent)) {
                result.push_back(
                    {identity(choice.getCondition()), child->getParentRegion() == &choice.getThenRegion()});
            }
        }
        std::reverse(result.begin(), result.end());
        return result;
    }
    bool dependsOn(Value value, Value target) const
    {
        SmallVector<Value> pending{value};
        llvm::DenseSet<Value> seen;
        while (!pending.empty()) {
            Value current = pending.pop_back_val();
            if (current == target) {
                return true;
            }
            if (!current || !seen.insert(current).second) {
                continue;
            }
            if (auto* operation = current.getDefiningOp()) {
                pending.append(operation->operand_begin(), operation->operand_end());
                if (auto choice = dyn_cast<scf::IfOp>(operation)) {
                    const auto result = cast<OpResult>(current).getResultNumber();
                    for (auto* region : {&choice.getThenRegion(), &choice.getElseRegion()}) {
                        if (!region->empty()) {
                            auto yield = dyn_cast<scf::YieldOp>(region->front().getTerminator());
                            if (yield && result < yield.getNumOperands()) {
                                pending.push_back(yield.getOperand(result));
                            }
                        }
                    }
                }
            } else if (auto argument = dyn_cast<BlockArgument>(current)) {
                auto loop = dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp());
                if (loop && argument.getOwner() == loop.getBody() && argument.getArgNumber()) {
                    const auto position = argument.getArgNumber() - 1;
                    pending.push_back(loop.getInitArgs()[position]);
                    pending.push_back(cast<scf::YieldOp>(loop.getBody()->getTerminator()).getOperand(position));
                }
            }
        }
        return false;
    }

    bool excludes(Value value, uint64_t bits, mlir::Operation* at) const
    {
        auto* child = at;
        for (auto* parent = child->getParentOp(); parent; child = parent, parent = parent->getParentOp()) {
            auto choice = dyn_cast<scf::IfOp>(parent);
            auto comparison = choice ? choice.getCondition().getDefiningOp<arith::CmpIOp>() : arith::CmpIOp{};
            if (!comparison) {
                continue;
            }
            const bool positive = child->getParentRegion() == &choice.getThenRegion();
            const auto lhs = comparison.getLhs(), rhs = comparison.getRhs();
            const auto constant = constantBits(lhs == value ? rhs : lhs);
            if ((lhs != value && rhs != value) || !constant || *constant != bits) {
                continue;
            }
            const auto predicate = comparison.getPredicate();
            const bool unequal = predicate == arith::CmpIPredicate::ne;
            // Equality with this constant is false for every strict comparison and
            // true for every non-strict comparison, regardless of operand order.
            const bool strict = predicate == arith::CmpIPredicate::slt || predicate == arith::CmpIPredicate::sgt ||
                                predicate == arith::CmpIPredicate::ult || predicate == arith::CmpIPredicate::ugt;
            if ((positive && (unequal || strict)) || (!positive && !unequal && !strict)) {
                return true;
            }
        }
        return false;
    }

    bool arithmeticTotal(mlir::Operation* operation) const
    {
        // This is a scalar fragment, not a second payload-effect registry. Payloads
        // are always recognized through the shared translated instruction list.
        if (!operation || operation->getNumRegions() || operation->getNumResults() != 1 ||
            !width(operation->getResult(0).getType())) {
            return false;
        }
        for (Value operand : operation->getOperands()) {
            if (!width(operand.getType())) {
                return false;
            }
        }
        if (isa<arith::ConstantOp, arith::CmpIOp, arith::AndIOp, arith::OrIOp, arith::XOrIOp, arith::MinSIOp,
                arith::MaxSIOp, arith::MinUIOp, arith::MaxUIOp, arith::IndexCastOp, arith::IndexCastUIOp,
                arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp, arith::SelectOp>(operation)) {
            return true;
        }
        if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(operation)) {
            auto flags = dyn_cast<arith::ArithIntegerOverflowFlagsInterface>(operation);
            if (!flags || (!flags.hasNoSignedWrap() && !flags.hasNoUnsignedWrap())) {
                return true; // original bitvector value, not a no-wrap mathematical rewrite
            }
            const auto left = integerRange(operation->getOperand(0));
            const auto right = integerRange(operation->getOperand(1));
            const auto opcode = isa<arith::AddIOp>(operation) ? value_arithmetic::Binary::Add :
                                isa<arith::SubIOp>(operation) ? value_arithmetic::Binary::Subtract :
                                                                value_arithmetic::Binary::Multiply;
            const auto bits = width(operation->getResult(0).getType());
            for (bool unsignedMode : {false, true}) {
                if (unsignedMode ? !flags.hasNoUnsignedWrap() : !flags.hasNoSignedWrap()) {
                    continue;
                }
                const uint64_t a[] = {
                    unsignedMode ? left.umin().getZExtValue() : left.smin().getZExtValue(),
                    unsignedMode ? left.umax().getZExtValue() : left.smax().getZExtValue()};
                const uint64_t b[] = {
                    unsignedMode ? right.umin().getZExtValue() : right.smin().getZExtValue(),
                    unsignedMode ? right.umax().getZExtValue() : right.smax().getZExtValue()};
                for (auto x : a) {
                    for (auto y : b) {
                        if (!value_arithmetic::checked(opcode, {bits, unsignedMode}, x, y)) {
                            return false;
                        }
                    }
                }
            }
            return true;
        }
        if (isa<arith::DivUIOp, arith::RemUIOp, arith::DivSIOp, arith::RemSIOp>(operation)) {
            const auto left = integerRange(operation->getOperand(0));
            const auto right = integerRange(operation->getOperand(1));
            const bool unsignedMode = isa<arith::DivUIOp, arith::RemUIOp>(operation);
            if (unsignedMode) {
                return !right.umin().isZero() || excludes(operation->getOperand(1), 0, operation);
            }
            const bool nonzero = right.smin().isStrictlyPositive() || right.smax().isNegative() ||
                                 excludes(operation->getOperand(1), 0, operation);
            const bool canMinusOne =
                right.smin().isNegative() && (!right.smax().isNegative() || right.smax().isAllOnes());
            const value_arithmetic::Integer type{width(operation->getOperand(1).getType()), false};
            return nonzero && !(left.smin().isMinSignedValue() && canMinusOne &&
                                !excludes(operation->getOperand(1), type.mask(), operation) &&
                                !excludes(operation->getOperand(0), type.minimum(), operation));
        }
        return false;
    }

    // Changed carried values keep their own identity. This limited SCC test
    // admits pure, total scalar recurrences only; it does not invent an async
    // incoming completion or substitute the initial value on later iterations.
    bool pureCarried(Value root) const
    {
        SmallVector<Value> pending{root};
        llvm::DenseSet<Value> seen;
        while (!pending.empty()) {
            Value value = pending.pop_back_val();
            if (!seen.insert(value).second) {
                continue;
            }
            if (auto argument = dyn_cast<BlockArgument>(value)) {
                auto* parent = argument.getOwner()->getParentOp();
                if (isa<func::FuncOp>(parent)) {
                    continue;
                }
                auto loop = dyn_cast<scf::ForOp>(parent);
                if (!loop || argument.getOwner() != loop.getBody()) {
                    return false;
                }
                if (argument.getArgNumber()) {
                    const auto index = argument.getArgNumber() - 1;
                    pending.push_back(loop.getInitArgs()[index]);
                    pending.push_back(cast<scf::YieldOp>(loop.getBody()->getTerminator()).getOperand(index));
                }
                continue;
            }
            auto* operation = value.getDefiningOp();
            if (!operation || phases.count(operation) || !arithmeticTotal(operation)) {
                return false;
            }
            pending.append(operation->operand_begin(), operation->operand_end());
        }
        return true;
    }

    bool previousVisitPath(scf::ForOp loop, OriginalValueCut cut) const
    {
        if (!legal(cut)) {
            return false;
        }
        auto* child = original.originalSites[cut.site];
        const bool unsignedLoop = loopInteger(loop).isUnsigned;
        for (auto* parent = child->getParentOp(); parent; child = parent, parent = parent->getParentOp()) {
            auto choice = dyn_cast<scf::IfOp>(parent);
            auto comparison = choice ? choice.getCondition().getDefiningOp<arith::CmpIOp>() : arith::CmpIOp{};
            if (!comparison) {
                continue;
            }
            const bool positive = child->getParentRegion() == &choice.getThenRegion();
            const auto lhs = comparison.getLhs(), rhs = comparison.getRhs();
            const bool forward = lhs == loop.getInductionVar() && rhs == loop.getLowerBound();
            const bool reverse = rhs == loop.getInductionVar() && lhs == loop.getLowerBound();
            if (!forward && !reverse) {
                continue;
            }
            const auto predicate = comparison.getPredicate();
            if ((positive && predicate == arith::CmpIPredicate::ne) ||
                (!positive && predicate == arith::CmpIPredicate::eq)) {
                return true;
            }
            const auto greater = unsignedLoop ? arith::CmpIPredicate::ugt : arith::CmpIPredicate::sgt;
            const auto less = unsignedLoop ? arith::CmpIPredicate::ult : arith::CmpIPredicate::slt;
            if (positive && (forward ? predicate == greater : predicate == less)) {
                return true;
            }
        }
        return false;
    }
    Result previousValue(OriginalValueReference reference, OriginalValueCut cut) const
    {
        if (reference.visitOffset != -1 || reference.occurrenceScope >= original.originalSites.size()) {
            return obstruction(
                Status::NotObservableHere, "earlier value occurrence has no qualified retained SSA representative");
        }
        auto loop = dyn_cast_or_null<scf::ForOp>(original.originalSites[reference.occurrenceScope]);
        if (!loop || !legal(cut) || !loop->isProperAncestor(original.originalSites[cut.site]) ||
            !previousVisitPath(loop, cut)) {
            return obstruction(
                Status::NotObservableHere,
                "previous value needs an original noninitial-visit path and retained carried SSA value");
        }
        auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
        if (!yield) {
            return obstruction(Status::Unresolved, "previous value has no original yield interface");
        }
        for (auto [position, value] : llvm::enumerate(yield.getOperands())) {
            if (value != reference.value) {
                continue;
            }
            const auto representative = loop.getRegionIterArgs()[position];
            if (!pureCarried(representative)) {
                return obstruction(
                    Status::Unresolved,
                    "previous asynchronous/partial scalar value needs qualified completion transport");
            }
            auto result = qualify(representative, cut);
            result.values.push_back(identity(reference.value));
            result.representative = representative;
            result.recipe = {
                OriginalValueRecipe::OriginalSSA, {representative}, width(representative.getType()), false, 0, 1};
            return result;
        }
        return obstruction(Status::NotObservableHere, "original loop does not retain the requested previous value");
    }

    Result qualifyImpl(Value value, OriginalValueCut cut) const
    {
        if (!value) {
            return obstruction(Status::Unresolved, "missing original scalar value");
        }
        if (!observable(value, cut)) {
            return obstruction(Status::NotObservableHere, "original SSA value is not available at this original cut");
        }
        auto result = availableResult();
        result.values.push_back(identity(value));
        result.recipe = {OriginalValueRecipe::OriginalSSA, {value}, width(value.getType()), false, 0, 1};
        if (!width(value.getType())) {
            // Unknown scalar qualification must not discard an identifiable async
            // producer (for example a float value feeding an unsupported comparison).
            result = combine(
                std::move(result), obstruction(Status::Unresolved, "unsupported original scalar width or type"));
        }
        if (auto argument = dyn_cast<BlockArgument>(value)) {
            auto* parent = argument.getOwner()->getParentOp();
            if (isa<func::FuncOp>(parent)) {
                return result; // original invocation inputs, not an unknown async producer
            }
            auto loop = dyn_cast<scf::ForOp>(parent);
            if (!loop || argument.getOwner() != loop.getBody()) {
                return combine(
                    std::move(result),
                    obstruction(
                        Status::Unresolved, "region argument lacks a supported original scalar incoming interface"));
            }
            result = combine(std::move(result), counted(loop));
            if (!argument.getArgNumber()) {
                return result;
            }
            const auto canonical = identity(value);
            if (canonical.value != value) {
                return combine(std::move(result), qualify(canonical.value, before(loop.getOperation())));
            }
            if (pureCarried(value)) {
                return result;
            }
            return combine(
                std::move(result),
                obstruction(
                    Status::Unresolved, "changed carried value needs a qualified arithmetic/completion transport"));
        }
        auto* operation = value.getDefiningOp();
        const auto translated = phases.find(operation);
        if (translated != phases.end()) {
            const auto conditions = controlAt(cut);
            for (const auto& condition : conditions) {
                if (dependsOn(condition.identity.value, value)) {
                    return combine(
                        std::move(result),
                        obstruction(
                            Status::Unresolved,
                            "original control would circularly enable its own completion prerequisite"));
                }
            }
            auto* child = original.originalSites[cut.site];
            for (auto* parent = child->getParentOp(); parent; parent = parent->getParentOp()) {
                if (auto choice = dyn_cast<scf::IfOp>(parent)) {
                    result = combine(std::move(result), qualify(choice.getCondition(), before(parent)));
                }
            }
            for (auto phase : translated->second) {
                const auto& source = original.operations[phase];
                const auto endpoint = phaseCut(source.enclosingAfter, true);
                if (!legal(endpoint)) {
                    result = combine(
                        std::move(result),
                        obstruction(
                            Status::Unresolved, "asynchronous value has no executable whole-instruction source cut"));
                }
                if (source.instruction->kPipeValue == PipelineType::PIPE_UNASSIGNED) {
                    result = combine(
                        std::move(result),
                        obstruction(Status::Unresolved, "asynchronous scalar producer has an unqualified engine"));
                }
                result.prerequisites.push_back(
                    {value,
                     phase,
                     source.enclosingAfter,
                     endpoint,
                     cut,
                     source.instruction->kPipeValue,
                     conditions,
                     {value, identity(value).occurrenceScope, 0}});
            }
            if (result.status == Status::Available) {
                result.status = Status::NeedsCompletion;
            }
            return result;
        }
        if (auto choice = dyn_cast<scf::IfOp>(operation)) {
            result = combine(std::move(result), qualify(choice.getCondition(), before(operation)));
            const auto number = cast<OpResult>(value).getResultNumber();
            const auto constant = constantBits(choice.getCondition());
            unsigned arm = 0;
            for (auto* region : {&choice.getThenRegion(), &choice.getElseRegion()}) {
                const bool selected = !constant || (*constant != 0) == (arm == 0);
                ++arm;
                if (!selected) {
                    continue;
                }
                if (region->empty()) {
                    return combine(std::move(result), obstruction(Status::Unresolved, "missing original value arm"));
                }
                auto yield = dyn_cast<scf::YieldOp>(region->front().getTerminator());
                if (!yield || number >= yield.getNumOperands()) {
                    return combine(
                        std::move(result), obstruction(Status::Unresolved, "missing original yielded value"));
                }
                result = combine(std::move(result), qualify(yield.getOperand(number), before(yield)));
            }
            return result;
        }
        if (auto loop = dyn_cast<scf::ForOp>(operation)) {
            result = combine(std::move(result), counted(loop));
            const auto number = cast<OpResult>(value).getResultNumber();
            auto carried = loop.getRegionIterArgs()[number];
            if (identity(carried).value != carried) {
                return combine(std::move(result), qualify(loop.getInitArgs()[number], before(operation)));
            }
            if (pureCarried(carried)) {
                return result;
            }
            return combine(
                std::move(result),
                obstruction(Status::Unresolved, "loop result needs a qualified original carried-value exit interface"));
        }
        // Dependencies are checked at their actual ORIGINAL evaluation point.
        // In async -> cmp -> if, the enabling deadline is before cmp, not before if.
        const auto use = before(operation);
        if (!arithmeticTotal(operation)) {
            result = combine(
                std::move(result),
                obstruction(
                    Status::Unresolved, "scalar operation lacks a total width/signedness/overflow qualification"));
        }
        if (auto select = dyn_cast<arith::SelectOp>(operation)) {
            result = combine(std::move(result), qualify(select.getCondition(), use));
            const auto condition = constantBits(select.getCondition());
            if (condition) {
                return combine(
                    std::move(result), qualify(*condition ? select.getTrueValue() : select.getFalseValue(), use));
            }
        }
        for (Value operand : operation->getOperands()) {
            result = combine(std::move(result), qualify(operand, use));
        }
        return result;
    }

    const OriginalStructure& original;
    mutable mlir::DominanceInfo dominance;
    unsigned indexWidth;
    uint64_t originalVersion;
    OriginalProgramVersion snapshot;
    DenseMap<mlir::Operation*, std::size_t> sites;
    DenseMap<mlir::Operation*, SmallVector<std::size_t>> phases;
    std::map<std::pair<uintptr_t, std::size_t>, std::size_t> guardOwners;
    std::map<std::size_t, std::size_t> canonicalGuards;
    mutable std::map<Key, Result> cache;
    mutable std::set<Key> active;
    mutable std::map<mlir::Operation*, Result> loops;
    mutable SyncSlotMapping::RangeCache ranges;
    mutable uint64_t evaluations = 0;
};

} // namespace mlir::pto::frontiersynch
#endif
