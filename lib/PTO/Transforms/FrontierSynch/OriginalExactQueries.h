// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#ifndef PTO_FRONTIERSYNCH_ORIGINALEXACTQUERIES_H
#define PTO_FRONTIERSYNCH_ORIGINALEXACTQUERIES_H

#include "OriginalReadQueries.h"
#include "PTO/Transforms/FrontierSynch/ExactFrontiers.h"
#include <algorithm>
#include <functional>
#include <map>
#include <set>

namespace mlir::pto::frontiersynch {

// One fixed-selector structural procedure for read frontiers and conflicting
// accesses. Its expression IDs belong to the existing shared reader arena.
// It neither enumerates original obligations nor changes their identities.
class OriginalExactQueries {
public:
    OriginalExactQueries(const OriginalStructure& original, const OriginalValueQueries& values,
                         OriginalReadQueries& expressions)
        : original(original), values(values), expressions(expressions), slices(original.body),
          algebra(expressions.boundaryAlgebra()) {}

    const OriginalExactFrontiers& query(const OriginalInterval& interval, bool readOnly)
    {
        const auto key = std::make_pair(interval, readOnly);
        const auto old = queries.find(key);
        if (old != queries.end() && values.current() && interval.query.version == original.version) {
            return old->second;
        }
        OriginalExactFrontiers answer;
        answer.interval = interval;
        answer.readOnly = readOnly;
        const auto& selector = interval.query.selector;
        if (!values.current() || interval.query.version != original.version) {
            answer.reason = "exact frontier has a stale original-program version";
        } else if (selector.cell >= original.cells.size() || (!selector.read && !selector.write) ||
                   (selector.engine && *selector.engine >= unsigned(PipelineType::PIPE_NUM))) {
            answer.reason = "invalid fixed physical access selector";
        } else if (readOnly && (!selector.read || selector.write || !selector.engine)) {
            answer.reason = "D3 requires a read-only selector for one reader engine";
        } else if (selector.qualification != NoControlId) {
            answer.reason = "selector qualification has no supplied exact interpretation";
        } else {
            const auto slice = slices.slice(interval.query);
            if (!slice.complete) {
                answer.reason = slice.reason;
            } else {
                std::vector<exact_frontier::Summary> parts;
                bool anyRead = false, anyWrite = false, completeEffects = true;
                for (auto i = slice.begin; i < slice.end; ++i) {
                    const auto part = all(*(*slice.parts)[i], selector);
                    anyRead |= part.may;
                    anyWrite |= part.writes;
                    completeEffects &= part.valid;
                }
                if (!completeEffects) {
                    parts.push_back(exact_frontier::unknown("All lacks a complete interval effect record"));
                } else if (readOnly && anyRead && anyWrite) {
                    // A write-only sibling must not disappear through the read
                    // selector's All exclusion and join two different generations.
                    parts.push_back(exact_frontier::unknown("overlapping write splits the read-only interval"));
                } else {
                    for (auto i = slice.begin; i < slice.end; ++i) {
                        parts.push_back(summary(*(*slice.parts)[i], selector, readOnly));
                    }
                }
                stats.compositionParts += parts.size();
                const auto result = exact_frontier::sequence(algebra, parts);
                answer.nonempty = result.nonempty;
                answer.first = result.first;
                answer.last = result.last;
                answer.reason = result.reason;
                answer.status = !result.complete ? OriginalExactFrontiers::Status::Unknown :
                                result.nonempty ? OriginalExactFrontiers::Status::Exact :
                                                  OriginalExactFrontiers::Status::NoHit;
                if (result.complete) {
                    answer.noHit = algebra.negate(result.nonempty);
                }
            }
        }
        // Updating a stale request's result cannot turn it into a cached success.
        return queries.insert_or_assign(key, std::move(answer)).first->second;
    }
    const OriginalExactFrontierStats& statistics() const { return stats; }

private:
    using Summary = exact_frontier::Summary;
    struct All {
        bool valid = true, may = false, writes = false;
    };
    static std::optional<uint64_t> constant(Value value)
    {
        auto op = value ? value.getDefiningOp<arith::ConstantOp>() : arith::ConstantOp{};
        auto attribute = op ? dyn_cast<IntegerAttr>(op.getValue()) : IntegerAttr{};
        if (!attribute || attribute.getValue().getBitWidth() > 64) {
            return {};
        }
        return attribute.getValue().getZExtValue();
    }
    static value_arithmetic::Integer integer(scf::ForOp loop)
    {
        auto type = loop.getInductionVar().getType();
        const auto width = type.isIndex() ? 64u : cast<IntegerType>(type).getWidth();
        return {width, loop->hasAttr("unsignedCmp") || loop->hasAttr("unsigned_cmp")};
    }
    scf::ForOp countedLoop(const Region& region) const
    {
        return region.originalOwner < original.originalSites.size() ?
                   dyn_cast_or_null<scf::ForOp>(original.originalSites[region.originalOwner]) : scf::ForOp{};
    }
    std::optional<bool> nonemptyLoop(scf::ForOp loop) const
    {
        const auto lo = constant(loop.getLowerBound()), hi = constant(loop.getUpperBound());
        const auto type = integer(loop);
        return lo && hi && type.valid() ? std::optional<bool>(type.less(*lo, *hi)) : std::nullopt;
    }
    bool selected(const Access& access, const PhysicalOperation& op, const OriginalAccessSelector& selector) const
    {
        return access.cell == selector.cell && ((selector.read && access.read) || (selector.write && access.write)) &&
               (!selector.engine || (op.instruction && unsigned(op.instruction->kPipeValue) == *selector.engine)) &&
               (selector.physicalRelation == NoControlId || access.physicalRelation == selector.physicalRelation ||
                access.physicalRelation == NoControlId);
    }
    All all(const Region& region, const OriginalAccessSelector& selector)
    {
        const auto key = std::make_pair(&region, selector);
        auto found = allCache.find(key);
        if (found != allCache.end()) {
            return found->second;
        }
        ++stats.allQueries;
        All out;
        if (region.kind == Region::For) {
            auto loop = countedLoop(region);
            if (loop && nonemptyLoop(loop) == std::optional<bool>(false) &&
                values.counted(loop).executableAfterPrerequisites()) {
                allCache.emplace(key, out);
                return out;
            }
        }
        if (region.kind == Region::Choice && region.children.size() == 2 &&
            region.originalOwner < original.originalSites.size()) {
            auto branch = dyn_cast_or_null<scf::IfOp>(original.originalSites[region.originalOwner]);
            const auto fixed = branch ? constant(branch.getCondition()) : std::optional<uint64_t>{};
            if (fixed) {
                out = all(region.children[*fixed ? 0 : 1], selector);
                allCache.emplace(key, out);
                return out;
            }
        }
        if (region.kind == Region::Operation) {
            if (region.operation >= original.operations.size() || !original.operations[region.operation].instruction) {
                out.valid = false;
            } else {
                const auto& op = original.operations[region.operation];
                if (op.instruction->kPipeValue == PipelineType::PIPE_UNASSIGNED ||
                    unsigned(op.instruction->kPipeValue) >= unsigned(PipelineType::PIPE_NUM)) {
                    out.valid = false; // An unknown engine is not an engine-filter exclusion.
                }
                for (const auto& access : op.accesses) {
                    out.may |= selected(access, op, selector);
                    // A write by ANOTHER reader engine still interrupts this cell's episode.
                    out.writes |= access.cell == selector.cell && access.write;
                }
            }
        } else {
            for (const auto& child : region.children) {
                const auto part = all(child, selector);
                out.valid &= part.valid;
                out.may |= part.may;
                out.writes |= part.writes;
            }
        }
        allCache.emplace(key, out);
        return out;
    }
    bool exactLeaf(const Region& region, const OriginalAccessSelector& selector) const
    {
        const auto& cell = original.cells[selector.cell];
        if (cell.unknownRange || cell.storage == Cell::Storage::OverlapWitness) {
            return false;
        }
        const auto& op = original.operations[region.operation];
        if (!op.instruction || op.instruction->kPipeValue == PipelineType::PIPE_UNASSIGNED ||
            unsigned(op.instruction->kPipeValue) >= unsigned(PipelineType::PIPE_NUM)) {
            return false;
        }
        for (const auto& access : op.accesses) {
            if (!selected(access, op, selector)) {
                continue;
            }
            if (selector.physicalRelation != NoControlId && access.physicalRelation != selector.physicalRelation) {
                continue;
            }
            if (access.physicalRelation != NoControlId &&
                (access.physicalRelation >= original.physicalAddresses.size() ||
                 original.physicalAddresses[access.physicalRelation].addresses.size() != 1 ||
                 original.physicalAddresses[access.physicalRelation].addresses.front().size() != 1)) {
                continue; // a may footprint is not fixed-cell participation
            }
            bool varying = false;
            for (const auto& address : original.physicalAddresses) {
                varying |= access.memory && address.memory == access.memory &&
                           (address.addresses.size() != 1 || address.addresses.front().size() != 1);
            }
            if (varying || (access.memory && access.memory->aliasesUnknownRange)) {
                continue;
            }
            // Keep the existing exact-cell read-incidence premise. For a
            // write-only selector, an allocation/may-write interval alone is
            // insufficient: the imported full-cell witness proves a hit.
            // One proved incidence suffices for this operation's participation;
            // other conservative incidences remain original obligations.
            if ((selector.read && access.read) || (selector.write && access.write && access.definiteWrite)) {
                return true;
            }
        }
        return false;
    }
    Summary summary(const Region& region, const OriginalAccessSelector& selector, bool readOnly)
    {
        const auto key = std::make_tuple(&region, selector, readOnly);
        const auto found = summaries.find(key);
        if (found != summaries.end()) {
            return found->second;
        }
        ++stats.summaryQueries;
        auto out = derive(region, selector, readOnly);
        summaries.emplace(key, out);
        return out;
    }
    Summary derive(const Region& region, const OriginalAccessSelector& selector, bool readOnly)
    {
        const auto footprint = all(region, selector);
        if (!footprint.valid) {
            return exact_frontier::unknown("All lacks a complete original effect/structure record");
        }
        if (!footprint.may) {
            return {}; // Only a proved All exclusion can skip this subtree.
        }
        auto loop = region.kind == Region::For ? countedLoop(region) : scf::ForOp{};
        if (loop && nonemptyLoop(loop) == std::optional<bool>(false)) {
            // Preserve the original counted arithmetic premise, but do not query
            // an unexecuted body's participation or its hypothetical endpoints.
            if (values.counted(loop).executableAfterPrerequisites()) {
                return {};
            }
        }
        if (readOnly && footprint.writes) {
            return exact_frontier::unknown("overlapping write interrupts the selected read-only interval");
        }
        if (region.kind == Region::Operation) {
            if (!exactLeaf(region, selector)) {
                return exact_frontier::unknown("matching may footprint lacks exact fixed-cell participation");
            }
            const auto leaf = expressions.boundaryLeaf(region.operation);
            return {true, 1, leaf, leaf, {}};
        }
        if (region.kind == Region::While) {
            return exact_frontier::unknown("uncounted participation has no specified exact frontier rule");
        }
        if (region.kind == Region::Choice) {
            if (region.children.size() != 2 || region.originalOwner >= original.originalSites.size()) {
                return exact_frontier::unknown("choice has no qualified original arms/guard");
            }
            auto choice = dyn_cast_or_null<scf::IfOp>(original.originalSites[region.originalOwner]);
            if (!choice) {
                return exact_frontier::unknown("choice has no original Boolean observation");
            }
            const auto fixed = constant(choice.getCondition());
            if (fixed) {
                return summary(region.children[*fixed ? 0 : 1], selector, readOnly);
            }
            const auto condition = expressions.boundaryAtom({ObservationAtom::OriginalBoolean, region.originalOwner, 0, 1});
            return exact_frontier::choice(algebra, condition,
                summary(region.children[0], selector, readOnly), summary(region.children[1], selector, readOnly));
        }
        if (region.kind == Region::For) {
            if (!loop || region.children.size() != 1 || !values.counted(loop).executableAfterPrerequisites()) {
                return exact_frontier::unknown("counted frontier lacks qualified original bounds and final increment");
            }
            const auto body = summary(region.children.front(), selector, readOnly);
            const auto invariant = body.complete &&
                (body.nonempty <= 1 || expressions.qualificationAtCut(body.nonempty, values.before(loop)).executableAfterPrerequisites());
            if (!invariant) {
                return intervalSummary(region, selector); // I.2 or an explicit missing premise
            }
            const auto known = nonemptyLoop(loop);
            const auto visits = known ? std::size_t(*known) :
                expressions.boundaryAtom({ObservationAtom::LoopNonEmpty, region.originalOwner, 0, 1});
            return exact_frontier::counted(algebra, body, visits,
                expressions.boundaryAtom({ObservationAtom::LoopHasPrevious, region.originalOwner, 1, 0}),
                expressions.boundaryAtom({ObservationAtom::LoopHasNext, region.originalOwner, 1, 0}), true);
        }
        if (region.kind != Region::Sequence) {
            return exact_frontier::unknown("unsupported original structured-control kind");
        }
        std::vector<Summary> parts;
        for (const auto& child : region.children) {
            parts.push_back(summary(child, selector, readOnly));
        }
        stats.compositionParts += parts.size();
        return exact_frontier::sequence(algebra, parts);
    }

    // I.2: one fixed physical read site, selected by a conjunction of invariant
    // interval bounds and optional invariant original conditions. Other sites,
    // varying disjunctions and nested repeats are not fitted to this fragment.
    Summary intervalSummary(const Region& region, const OriginalAccessSelector& selector)
    {
        auto loop = countedLoop(region);
        const auto type = integer(loop);
        if (!selector.read || selector.write || all(region, selector).writes || constant(loop.getStep()) != 1 ||
            constant(loop.getLowerBound()) != 0 || !type.valid()) {
            return exact_frontier::unknown("I.2 requires a unit-step zero-based read-only interval");
        }
        std::size_t operation = NoControlId;
        std::vector<std::pair<Value, bool>> path;
        std::string reason;
        std::function<bool(const Region&)> locate = [&](const Region& node) {
            if (!all(node, selector).may) {
                return true;
            }
            if (node.kind == Region::Operation) {
                if (operation != NoControlId || !exactLeaf(node, selector)) {
                    reason = "I.2 requires one exact selected read site, not several varying readers";
                    return false;
                }
                operation = node.operation;
                return true;
            }
            if (node.kind == Region::Sequence) {
                for (const auto& child : node.children) {
                    if (!locate(child)) {
                        return false;
                    }
                }
                return true;
            }
            if (node.kind != Region::Choice || node.children.size() != 2 ||
                node.originalOwner >= original.originalSites.size()) {
                reason = "I.2 path contains an unqualified nested repetition";
                return false;
            }
            auto branch = dyn_cast_or_null<scf::IfOp>(original.originalSites[node.originalOwner]);
            if (!branch) {
                reason = "I.2 path has no original branch condition";
                return false;
            }
            if (const auto fixed = constant(branch.getCondition())) {
                return locate(node.children[*fixed ? 0 : 1]);
            }
            const bool yes = all(node.children[0], selector).may, no = all(node.children[1], selector).may;
            if (yes && no) {
                reason = "I.2 has independently varying selected sites in both arms";
                return false;
            }
            path.emplace_back(branch.getCondition(), yes);
            return locate(node.children[yes ? 0 : 1]);
        };
        if (!locate(region.children.front()) || operation == NoControlId) {
            return exact_frontier::unknown(reason.empty() ? "I.2 has no exact selected read" : reason);
        }
        OriginalIntervalParticipation record;
        record.version = original.version;
        record.owner = region.originalOwner;
        record.operation = operation;
        record.induction = loop.getInductionVar();
        record.recipe.integer = type;
        DenseMap<Value, std::size_t> referenceIds;
        auto reference = [&](Value value) {
            const auto found = referenceIds.find(value);
            if (found != referenceIds.end()) {
                return found->second;
            }
            const auto id = record.references.size();
            record.references.push_back(value);
            referenceIds[value] = id;
            return id;
        };
        record.recipe.lower.push_back({reference(loop.getLowerBound()), false});
        record.recipe.upper.push_back({reference(loop.getUpperBound()), false});
        bool empty = false;
        auto addBound = [&](Value value, bool lower, bool successor) {
            if (!value || value.getType() != record.induction.getType() ||
                !values.qualify(value, values.before(loop)).executableAfterPrerequisites()) {
                reason = "I.2 bound is not an invariant available original value";
                return false;
            }
            if (successor) {
                const auto range = SyncSlotMapping::range(value, ranges);
                const auto maximum = type.isUnsigned ? range.umax().getZExtValue() : range.smax().getZExtValue();
                if (!value_arithmetic::checked(value_arithmetic::Binary::Add, type, maximum, 1)) {
                    reason = "I.2 inclusive/equality bound lacks an independent +1 overflow proof";
                    return false;
                }
            }
            (lower ? record.recipe.lower : record.recipe.upper).push_back({reference(value), successor});
            return true;
        };
        std::set<std::pair<std::uintptr_t, bool>> constrained;
        std::function<bool(Value, bool)> constrain = [&](Value condition, bool positive) {
            if (!constrained.emplace(reinterpret_cast<std::uintptr_t>(condition.getAsOpaquePointer()), positive).second) {
                return true; // shared Boolean subexpression, not its repeated tree expansion
            }
            if (const auto fixed = constant(condition)) {
                empty |= bool(*fixed) != positive;
                return true;
            }
            if (positive) {
                if (auto both = condition.getDefiningOp<arith::AndIOp>()) {
                    return constrain(both.getLhs(), true) && constrain(both.getRhs(), true);
                }
            } else if (auto either = condition.getDefiningOp<arith::OrIOp>()) {
                return constrain(either.getLhs(), false) && constrain(either.getRhs(), false);
            }
            if (auto inverted = condition.getDefiningOp<arith::XOrIOp>()) {
                if (auto rhs = constant(inverted.getRhs())) {
                    return constrain(inverted.getLhs(), positive != bool(*rhs));
                }
                if (auto lhs = constant(inverted.getLhs())) {
                    return constrain(inverted.getRhs(), positive != bool(*lhs));
                }
            }
            auto cmp = condition.getDefiningOp<arith::CmpIOp>();
            Value bound;
            arith::CmpIPredicate p;
            if (cmp && (cmp.getLhs() == record.induction || cmp.getRhs() == record.induction)) {
                p = cmp.getPredicate();
                const bool swapped = cmp.getRhs() == record.induction;
                bound = swapped ? cmp.getLhs() : cmp.getRhs();
                // Compare against mathematical order in the original loop's
                // signedness, never an implicit signed host reinterpretation.
                using P = arith::CmpIPredicate;
                const bool equality = p == P::eq || p == P::ne;
                const bool unsignedPredicate = p == P::ult || p == P::ule || p == P::ugt || p == P::uge;
                if (!equality && unsignedPredicate != type.isUnsigned) {
                    reason = "I.2 bound comparison differs from the original loop signedness";
                    return false;
                }
                using C = exact_frontier::Comparison;
                C normalized;
                switch (p) {
                case P::eq: normalized = C::Equal; break;
                case P::ne: normalized = C::NotEqual; break;
                case P::slt: case P::ult: normalized = C::Less; break;
                case P::sle: case P::ule: normalized = C::LessEqual; break;
                case P::sgt: case P::ugt: normalized = C::Greater; break;
                case P::sge: case P::uge: normalized = C::GreaterEqual; break;
                default: reason = "unsupported integer interval comparison"; return false;
                }
                normalized = exact_frontier::normalizeComparison(normalized, swapped, positive);
                switch (normalized) {
                case C::GreaterEqual: return addBound(bound, true, false);
                case C::Less: return addBound(bound, false, false);
                case C::Greater: return addBound(bound, true, true);
                case C::LessEqual: return addBound(bound, false, true);
                case C::Equal: return addBound(bound, true, false) && addBound(bound, false, true);
                default: reason = "I.2 excludes a varying non-interval participation condition"; return false;
                }
            }
            // Keep the original invariant SSA value, including an operand of
            // an original andi that has no separate if site of its own.
            if (!condition.getType().isInteger(1) ||
                !values.qualify(condition, values.before(loop)).executableAfterPrerequisites()) {
                reason = "varying or later unavailable condition has no admitted interval derivation";
                return false;
            }
            record.invariantConditions.emplace_back(values.identity(condition).value, positive);
            return true;
        };
        for (const auto& condition : path) {
            if (!constrain(condition.first, condition.second)) {
                return exact_frontier::unknown(reason);
            }
        }
        if (empty) {
            return {};
        }
        // Constant folding does not unroll the loop or inspect guard valuations.
        const auto domain = exact_frontier::intersection(record.recipe, [&](std::size_t id) {
            return constant(record.references[id]);
        });
        if (domain.valid && !domain.nonempty) {
            return {};
        }
        stats.boundReferences += record.recipe.lower.size() + record.recipe.upper.size();
        ++stats.intervalRecipes;
        const bool always = domain.valid && record.invariantConditions.empty();
        const auto id = expressions.registerIntervalParticipation(std::move(record));
        const auto nonempty = always ? 1 :
            expressions.boundaryAtom({ObservationAtom::IntervalNonEmpty, region.originalOwner, id, 1});
        const auto first = expressions.boundaryAtom({ObservationAtom::IntervalFirst, region.originalOwner, id, 1});
        const auto last = expressions.boundaryAtom({ObservationAtom::IntervalLast, region.originalOwner, id, 1});
        const auto leaf = expressions.boundaryLeaf(operation);
        return {true, nonempty, algebra.guard(leaf, first), algebra.guard(leaf, last), {}};
    }

    const OriginalStructure& original;
    const OriginalValueQueries& values;
    OriginalReadQueries& expressions;
    exact_frontier::SliceIndex slices;
    exact_frontier::Algebra algebra;
    OriginalExactFrontierStats stats;
    SyncSlotMapping::RangeCache ranges;
    std::map<std::pair<const Region*, OriginalAccessSelector>, All> allCache;
    std::map<std::tuple<const Region*, OriginalAccessSelector, bool>, Summary> summaries;
    std::map<std::pair<OriginalInterval, bool>, OriginalExactFrontiers> queries;
};
} // namespace mlir::pto::frontiersynch
#endif
