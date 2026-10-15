// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent semantic and structural controls for affected-path substitution.
#ifndef PTO_TEST_REGION_SUBSTITUTION_CHECKS_H
#define PTO_TEST_REGION_SUBSTITUTION_CHECKS_H
#include "PTO/Transforms/FrontierSynch/Shared/RegionExpressions.h"
#include "PTO/Transforms/FrontierSynch/Arithmetic/IntegerRelations.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <type_traits>
namespace {
bool checkAffectedSubstitution(mlir::Value index, mlir::Value predicate)
{
    using namespace mlir;
    namespace fs = mlir::pto::frontiersynch;
    using Arena = fs::RegionExpressions;
    // This predates the mutable structural index: the dominance owner is
    // unique. A move transfers nodes and index together; copies are unavailable.
    static_assert(!std::is_copy_constructible_v<Arena> && !std::is_copy_assignable_v<Arena>);
    static_assert(std::is_move_constructible_v<Arena>);
    // Independent modular and signed arithmetic values, with overlapping
    // non-input bindings whose replacement must remain atomic.
    {
        Arena e;
        Block inputs;
        const auto x = e.input(index), p = e.input(predicate);
        const auto y = e.input(inputs.addArgument(index.getType(), index.getLoc()));
        const auto three = e.constant(3), seven = e.constant(7);
        const auto sum = e.add(x, three), difference = e.sub(y, seven);
        const auto root = e.select(p, e.add(sum, difference), e.div(e.add(x, y), three));
        const auto signedOrder = e.slt(sum, difference);
        const uint64_t values[] = {0, 1, 7, uint64_t(INT64_MAX), uint64_t(1) << 63, UINT64_MAX};
        for (auto a : values) { for (auto b : values) { for (bool flag : {false, true}) {
            Arena::Substitution binding({{x, e.constant(a)}, {y, e.constant(b)}, {p, e.boolean(flag)}});
            const auto expected = flag ? (a + 3) + (b - 7) : (a + b) / 3;
            if (e.constantValue(e.substitute(root, binding)) != expected ||
                e.constantValue(e.substitute(signedOrder, binding)) !=
                    uint64_t(APInt(64, a + 3).slt(APInt(64, b - 7)))) { return false; }
        } } }
        Arena::Substitution overlap({{x, e.constant(11)}, {sum, difference}, {y, e.constant(17)}});
        const auto rewritten = e.substitute(root, overlap);
        for (uint64_t b : {2U, 19U}) { for (bool flag : {false, true}) {
            Arena::Substitution remaining({{y, e.constant(b)}, {p, e.boolean(flag)}});
            // sum's atomic replacement still refers to original y, whereas
            // the original difference is independently rewritten to 17-7.
            const auto expected = flag ? (b - 7) + 10 : uint64_t(28 / 3);
            if (e.constantValue(e.substitute(rewritten, remaining)) != expected) { return false; }
        } }
        if (!e.error().empty()) { return false; }
    }
    // All bindings validate even when the root is explicitly replaced or when
    // the malformed source cannot occur in the root's reachable topology.
    for (bool boundRoot : {false, true}) {
        Arena e;
        const auto x = e.input(index), p = e.input(predicate), zero = e.constant(0);
        SmallVector<std::pair<Arena::Id, Arena::Id>> bindings{{p, zero}};
        if (boundRoot) { bindings.insert(bindings.begin(), {x, zero}); }
        Arena::Substitution invalid(bindings);
        if (e.substitute(x, invalid) != Arena::invalid || e.error().empty()) { return false; }
    }
    // Ordinary substitution remains eager in BOTH affected Select arms.
    // Selecting the safe arm cannot conceal an invalid rewritten divisor.
    {
        Arena e;
        const auto x = e.input(index), p = e.input(predicate), three = e.constant(3);
        const auto quotient = e.div(x, three), root = e.select(p, x, quotient);
        Arena::Substitution invalid({{p, e.boolean(true)}, {three, e.constant(0)}});
        if (e.substitute(root, invalid) != Arena::invalid || e.error().empty()) { return false; }
    }
    // An explicit ancestor binding still cuts traversal before that invalid
    // descendant; replacements are not recursively subjected to the bindings.
    {
        Arena e;
        const auto x = e.input(index), p = e.input(predicate), three = e.constant(3);
        const auto quotient = e.div(x, three), root = e.select(p, x, quotient);
        Arena::Substitution cutoff({{p, e.boolean(false)}, {three, e.constant(0)}, {quotient, x}});
        if (e.substitute(root, cutoff) != x || !e.error().empty()) { return false; }
    }
    // Integer recipes carry operands outside the ordinary a/b/c node fields.
    // Partial rebinding must preserve every untouched operand's identity.
    {
        Arena e;
        Block inputs;
        const auto x = e.input(index), y = e.input(inputs.addArgument(index.getType(), index.getLoc()));
        auto domain = fs::IntegerSystem::create(2,
            {{{fs::BoundInteger(1), fs::BoundInteger(1)}, fs::BoundInteger(3)}});
        if (failed(domain)) { return false; }
        const auto root = e.integerPredicate(*domain, {x, y}, 1, {0, 0});
        for (uint64_t a = 0; a < 5; ++a) {
            Arena::Substitution first({{x, e.constant(a)}});
            const auto partial = e.substitute(root, first);
            for (uint64_t b = 0; b < 5; ++b) {
                Arena::Substitution second({{y, e.constant(b)}});
                if (e.constantValue(e.substitute(partial, second)) != uint64_t(a + b <= 3)) { return false; }
            }
        }
    }
    // Boolean Input cofactors scale with the affected path after indexing.
    // Numeric substitutions keep the original full reachable DFS and memo reuse.
    // Initial index preparation is charged, never hidden.
    uint64_t previousWarm = 0;
    for (unsigned length : {32U, 512U, 4096U}) {
        Arena e;
        Block inputs;
        const auto x = e.input(index), p = e.input(predicate);
        auto branch = e.input(inputs.addArgument(index.getType(), index.getLoc()));
        for (unsigned i = 0; i < length; ++i) {
            branch = e.add(branch, e.input(inputs.addArgument(index.getType(), index.getLoc())));
        }
        const auto root = e.add(e.select(p, x, e.constant(7)), branch);
        uint64_t cold = 0, warm = 0, novel = 0, repeated = 0;
        Arena::Substitution first({{p, e.boolean(true)}});
        if (e.substitute(root, first, &cold) != e.add(x, branch) || cold <= length) { return false; }
        Arena::Substitution second({{p, e.boolean(false)}});
        if (e.substitute(root, second, &warm) != e.add(e.constant(7), branch) || warm > 32) { return false; }
        Arena::Substitution anotherSource({{x, e.constant(11)}});
        const auto result = e.substitute(root, anotherSource, &novel);
        if (result != e.add(e.select(p, e.constant(11), e.constant(7)), branch) || novel <= length ||
            e.substitute(root, anotherSource, &repeated) != result || repeated != 1) { return false; }
        if (previousWarm && warm != previousWarm) { return false; }
        previousWarm = warm;
    }
    // Many overlapping roots share one structural index for Boolean cofactors.
    // An absent Boolean Input must not retain a closure per larger root.
    // This control does not assert general numeric substitution scalability.
    for (unsigned length : {64U, 512U, 4096U}) {
        Arena e;
        const auto x = e.input(index), absent = e.input(predicate);
        const auto increment = e.constant(3), replacement = e.constant(11), no = e.boolean(false);
        SmallVector<Arena::Id> roots;
        auto root = x;
        for (unsigned i = 0; i < length; ++i) {
            root = e.add(root, increment); roots.push_back(root);
        }
        uint64_t visits = 0;
        for (auto current : roots) {
            Arena::Substitution binding({{absent, no}});
            if (e.substitute(current, binding, &visits) != current) { return false; }
        }
        if (visits > 20 * e.size()) { return false; }
        // Later roots must not force a linear parent-list scan for an old root.
        for (unsigned i = 0; i < length; ++i) { (void)e.sub(x, e.constant(i + 20)); }
        Arena::Substitution latest({{absent, no}});
        if (e.substitute(e.sub(x, replacement), latest) == Arena::invalid) { return false; }
        uint64_t old = 0;
        Arena::Substitution changed({{x, replacement}});
        if (e.constantValue(e.substitute(roots.front(), changed, &old)) != 14 || old > 128) { return false; }
    }
    // A late tiny numeric/mixed/nonliteral root must not scan the reverse
    // ancestry of thousands of unrelated roots sharing its input. The exact
    // old root-local DFS visits only that root's reachable operands.
    for (unsigned length : {64U, 512U, 4096U}) {
        Arena e;
        Block inputs;
        const auto x = e.input(index), p = e.input(predicate);
        const auto q = e.input(inputs.addArgument(predicate.getType(), predicate.getLoc()));
        const auto eleven = e.constant(11), seven = e.constant(7);
        for (unsigned i = 0; i < length; ++i) {
            (void)e.add(x, e.constant(i + 20));
            (void)e.select(p, x, e.constant(i + 30));
        }
        const auto root = e.sub(x, seven), choice = e.select(p, root, x), internal = e.lt(x, seven);
        const auto internalChoice = e.select(internal, root, x);
        SmallVector<SmallVector<std::pair<Arena::Id, Arena::Id>>> cases{
            {{x, eleven}}, {{x, eleven}, {p, e.boolean(true)}},
            {{p, q}}, {{internal, e.boolean(true)}},
            {{p, e.boolean(true)}, {p, e.boolean(true)}}};
        const Arena::Id roots[] = {root, choice, choice, internalChoice, choice};
        const Arena::Id expected[] = {e.constant(4), e.constant(4), e.select(q, root, x), root, root};
        for (unsigned i = 0; i < cases.size(); ++i) {
            Arena::Substitution binding(cases[i]);
            uint64_t visits = 0;
            if (e.substitute(roots[i], binding, &visits) != expected[i] || visits > 32) { return false; }
        }
    }
    // Force index preparation without changing a numeric expression. Each
    // lifecycle fixture creates this unrelated Boolean input before its root.
    auto indexRoot = [](Arena& arena, Arena::Id root, Arena::Id flag, Arena::Id no) {
        Arena::Substitution cofactor({{flag, no}});
        return arena.substitute(root, cofactor) == root;
    };
    // The arena index survives inner speculative work. A root created
    // and committed inside an outer rollback must not survive ID reuse.
    {
        Arena e;
        Block inputs;
        const auto x = e.input(index), y = e.input(inputs.addArgument(index.getType(), index.getLoc()));
        const auto seven = e.constant(7), zero = e.constant(0);
        const auto flag = e.input(predicate), no = e.boolean(false);
        const auto old = e.add(x, seven);
        Arena::Id removed;
        {
            Arena::Transaction outer(e);
            if (!indexRoot(e, old, flag, no)) { return false; }
            Arena::Substitution binding({{x, zero}});
            if (e.constantValue(e.substitute(old, binding)) != 7) { return false; }
            {
                Arena::Transaction inner(e);
                removed = e.sub(x, seven);
                if (!indexRoot(e, removed, flag, no)) { return false; }
                Arena::Substitution local({{x, zero}});
                if (e.constantValue(e.substitute(removed, local)) != uint64_t(0) - 7) { return false; }
                inner.commit();
            }
        }
        uint64_t warm = 0;
        Arena::Substitution again({{x, zero}});
        if (e.constantValue(e.substitute(old, again, &warm)) != 7 || warm > 32) { return false; }
        const auto reused = e.sub(y, seven);
        if (!indexRoot(e, reused, flag, no)) { return false; }
        Arena::Substitution absent({{x, zero}});
        if (reused != removed || e.substitute(reused, absent) != reused) { return false; }
    }
    // Duplicate structural operands occupy two reverse edges; nested rollback
    // removes both before reusing the original node ID for another operation.
    {
        Arena e;
        const auto x = e.input(index), seven = e.constant(7);
        const auto flag = e.input(predicate), no = e.boolean(false);
        Arena::Id removed;
        {
            Arena::Transaction outer(e);
            Arena::Transaction inner(e);
            removed = e.add(x, x);
            if (!indexRoot(e, removed, flag, no)) { return false; }
            Arena::Substitution bind({{x, seven}});
            if (e.constantValue(e.substitute(removed, bind)) != 14) { return false; }
            inner.commit();
        }
        const auto reused = e.sub(x, seven);
        if (!indexRoot(e, reused, flag, no)) { return false; }
        Arena::Substitution bind({{x, seven}});
        if (reused != removed || e.constantValue(e.substitute(reused, bind)) != 0) { return false; }
    }
    // Moving an indexed arena keeps structural ownership exclusive. Repeated
    // speculative fan-outs on different original inputs are removed completely.
    {
        Arena source;
        Block inputs;
        SmallVector<Arena::Id> originals;
        for (unsigned i = 0; i < 32; ++i) {
            originals.push_back(source.input(inputs.addArgument(index.getType(), index.getLoc())));
        }
        const auto three = source.constant(3), seven = source.constant(7);
        const auto flag = source.input(predicate), no = source.boolean(false);
        const auto root = source.add(originals.front(), three);
        if (!indexRoot(source, root, flag, no)) { return false; }
        Arena::Substitution initial({{originals.front(), seven}});
        if (source.constantValue(source.substitute(root, initial)) != 10) { return false; }
        Arena moved(std::move(source));
        for (auto input : originals) {
            {
                Arena::Transaction rollback(moved);
                auto last = root;
                for (unsigned i = 0; i < 128; ++i) {
                    last = moved.sub(input, moved.constant(i + 20));
                }
                if (!indexRoot(moved, last, flag, no)) { return false; }
                Arena::Substitution bind({{input, seven}});
                if (moved.constantValue(moved.substitute(last, bind)) != uint64_t(7) - 147) { return false; }
            }
            const auto reused = moved.add(input, three);
            if (!indexRoot(moved, reused, flag, no)) { return false; }
            Arena::Substitution bind({{input, seven}});
            if (moved.constantValue(moved.substitute(reused, bind)) != 10) { return false; }
        }
        Arena::Substitution old({{originals.front(), seven}});
        if (moved.constantValue(moved.substitute(root, old)) != 10) { return false; }
    }
    // Generic indexing never populates or invalidates threshold-only caches.
    // Both services retain their independent ownership and accounting.
    {
        Arena e;
        const auto x = e.input(index);
        const auto unrelated = e.input(predicate), no = e.boolean(false);
        const auto root = e.lt(x, e.constant(7));
        Arena::ThresholdProofCost cold, warm;
        if (e.inactiveByThresholds(root, &cold) || !cold.dagVisits || !cold.dependencyVisits) { return false; }
        if (!indexRoot(e, root, unrelated, no)) { return false; }
        uint64_t work = 0;
        Arena::Substitution binding({{x, e.constant(3)}});
        if (e.constantValue(e.substitute(root, binding, &work)) != 1 || !work ||
            e.inactiveByThresholds(root, &warm) || warm.dagVisits || warm.dependencyVisits ||
            warm.preparationHits != 1) { return false; }
        Arena::Substitution absent({{unrelated, no}});
        if (e.substitute(root, absent) != root) { return false; }
        uint64_t saturated = UINT64_MAX;
        Arena::Substitution again({{x, e.constant(1)}});
        if (e.constantValue(e.substitute(root, again, &saturated)) != 1 || saturated != UINT64_MAX) { return false; }
    }
    // Opaque evaluation inputs are atomic despite their ordinal provenance.
    {
        OpBuilder builder(index.getContext());
        const auto loc = index.getLoc();
        OwningOpRef<func::FuncOp> function(func::FuncOp::create(loc, "substitution_opaque",
            builder.getFunctionType({builder.getIndexType()}, {})));
        auto* body = function->addEntryBlock(); builder.setInsertionPointToEnd(body);
        auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
        auto one = builder.create<arith::ConstantIndexOp>(loc, 1);
        auto loop = builder.create<scf::ForOp>(loc, zero, body->getArgument(0), one);
        builder.create<func::ReturnOp>(loc);
        builder.setInsertionPointToStart(loop.getBody());
        auto comparison = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
            loop.getInductionVar(), body->getArgument(0));
        Block placeholders;
        Arena e;
        const auto ordinal = e.input(loop.getInductionVar());
        const auto opaque = e.evaluatedInput(placeholders.addArgument(builder.getI1Type(), loc),
            comparison, loop, ordinal);
        Arena::Substitution shifted({{ordinal, e.add(ordinal, e.constant(1))}});
        if (opaque == Arena::invalid || e.substitute(opaque, shifted) != opaque ||
            !e.evaluation(opaque) || e.evaluation(opaque)->ordinal != ordinal) { return false; }
    }
    return true;
}
} // namespace
#endif
