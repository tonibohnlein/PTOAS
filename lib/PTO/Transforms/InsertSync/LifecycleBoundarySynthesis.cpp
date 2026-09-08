// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.
#include "PTO/Transforms/InsertSync/LifecycleSynthesis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <limits>
#include <array>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::insert_sync_frontier;

namespace {
std::optional<int64_t> literal(Value value)
{
    IntegerAttr attr;
    if (
        !value || !matchPattern(value, m_Constant(&attr)) || !attr.getValue().isSignedIntN(64)) {
        return std::nullopt;
    }
    return canonicalGuardConstant(attr.getValue().getSExtValue(), attr.getValue().getBitWidth());
}
bool boundaryLoop(scf::ForOp loop)
{
    auto lo = literal(loop.getLowerBound()), step = literal(loop.getStep());
    return lo && *lo >= 0 && *lo <= std::numeric_limits<int32_t>::max() && step && *step > 0 &&
           *step <= std::numeric_limits<int32_t>::max() && loop.getInductionVar().getType().isIndex() &&
           !loop->hasAttr("unsignedCmp");
}
int8_t equalFact(const GuardEnvironment& env, unsigned variable, int64_t value)
{
    auto it = env.find(variable);
    if (it == env.end()) {
        return -1;
    }
    if (it->second.equal) {
        return *it->second.equal == value;
    }
    if (it->second.excluded.count(value)) {
        return 0;
    }
    return -1;
}
int8_t truth(const InsertSyncLifecycleStructure& s, unsigned node, const LifecycleGuardTest& test)
{
    if (test.kind == LifecycleGuardTest::Kind::First || test.kind == LifecycleGuardTest::Kind::Last) {
        for (const auto& which : s.iterations[node]) {
            if (which.loop == test.loop) {
                return test.kind == LifecycleGuardTest::Kind::First ? which.first : which.last;
            }
        }
        return -1;
    }
    return equalFact(s.guards[node], test.variable, test.kind == LifecycleGuardTest::Kind::Empty ? 0 : test.value);
}
bool performs(const LifecycleRole& role, LifecyclePlacement::Role action)
{
    switch (action) {
        case LifecyclePlacement::Role::PublishBefore:
            return role.publishBefore;
        case LifecyclePlacement::Role::ReleaseBefore:
            return role.releaseBefore;
        case LifecyclePlacement::Role::AcquireFree:
            return role.acquireFree;
        case LifecyclePlacement::Role::PublishReady:
            return role.publishReady;
        case LifecyclePlacement::Role::AcquireReady:
            return role.acquireReady;
        case LifecyclePlacement::Role::PublishFree:
            return role.publishFree;
        case LifecyclePlacement::Role::BypassReady:
            return role.bypassReady;
    }
    return false;
}
struct Point {
    Operation* anchor = nullptr;
    Block* blockEnd = nullptr;
    std::vector<unsigned> nodes;
};
std::vector<LifecycleGuardTest> testsFor(
    const InsertSyncLifecycleStructure& s, const Point& point, func::FuncOp function, DominanceInfo& dominance)
{
    std::vector<LifecycleGuardTest> tests;
    Operation* anchor = point.anchor;
    bool atEnd = point.blockEnd != nullptr;
    if (atEnd && !point.blockEnd->empty()) {
        anchor = &point.blockEnd->back();
    }
    auto available = [&](Value value) {
        if (!value || !anchor) {
            return false;
        }
        return dominance.dominates(value, anchor) || (atEnd && value.getDefiningOp() == anchor);
    };
    // Walk order is deterministic; never use pointer order to choose guards.
    function.walk([&](scf::ForOp loop) {
        if (!boundaryLoop(loop)) {
            return;
        }
        if (
            anchor && loop->isAncestor(anchor) && available(loop.getInductionVar())) {
            tests.push_back({LifecycleGuardTest::Kind::First, loop.getOperation(), {}, 0, kInvalid});
            tests.push_back({LifecycleGuardTest::Kind::Last, loop.getOperation(), {}, 0, kInvalid});
        }
    });
    for (const auto& domain : s.guardDomains) {
        if (domain.tripShapeOf) {
            auto loop = dyn_cast<scf::ForOp>(domain.tripShapeOf);
            if (
                loop && boundaryLoop(loop) && available(loop.getLowerBound()) && available(loop.getUpperBound())) {
                tests.push_back({LifecycleGuardTest::Kind::Empty, loop.getOperation(), {}, 0, domain.identity});
            }
            continue;
        }
        Value value = domain.expression;
        if (
            !available(value) || !isa<IndexType, IntegerType>(value.getType())) {
            continue;
        }
        if (auto integer = dyn_cast<IntegerType>(value.getType()); integer && integer.getWidth() > 64) {
            continue;
        }
        std::set<int64_t> constants;
        for (unsigned node : point.nodes) {
            auto found = s.guards[node].find(domain.identity);
            if (found == s.guards[node].end()) {
                continue;
            }
            if (found->second.equal) {
                constants.insert(*found->second.equal);
            }
            constants.insert(found->second.excluded.begin(), found->second.excluded.end());
        }
        if (
            constants.size() > 16) {
            continue;
        }
        for (int64_t c : constants) {
            tests.push_back({LifecycleGuardTest::Kind::ValueEquals, nullptr, value, c, domain.identity});
        }
    }
    return tests;
}
} // namespace

bool mlir::pto::qualifyInsertSyncLifecycleBoundaries(
    const InsertSyncLifecycleStructure& s, InsertSyncLifecyclePlan::Channel& channel, func::FuncOp function,
    Budget& budget, std::string& reason)
{
    const unsigned size = s.program.nodes.size();
    if (
        s.guards.size() != size || s.iterations.size() != size || s.loopExits.size() != size ||
        s.blockExits.size() != size) {
        channel.logical.certificate.status = LifecycleCertificate::Status::InvalidInput;
        reason = "incomplete native guarded-occurrence bindings";
        return false;
    }
    const auto& spec = channel.logical.spec;
    llvm::SmallPtrSet<Operation*, 16> readerLoops;
    for (unsigned p = 0; p < spec.phases.size(); ++p) {
        if (!spec.phases[p].reads) {
            continue;
        }
        Operation* op = s.phases[p]->elementOp;
        auto loop = dyn_cast<scf::ForOp>(op->getParentOp());
        if (!loop || !boundaryLoop(loop)) {
            continue;
        }
        bool closed = true;
        for (unsigned q = 0; q < spec.phases.size(); ++q) {
            Operation* other = s.phases[q]->elementOp;
            if (
                !loop->isAncestor(other) || !(spec.phases[q].reads || spec.phases[q].writes)) {
                continue;
            }
            // No partial writer or optional nested reader is hidden by the empty
            // bridge. Those require another complete recipe, not this exception.
            if (
                spec.phases[q].writes || other->getParentOp() != loop.getOperation()) {
                closed = false;
            }
        }
        if (closed) {
            readerLoops.insert(loop.getOperation());
        }
    }
    LifecycleBoundaryFacts facts{Bits(size), Bits(size)};
    for (unsigned n = 0; n < size; ++n) {
        Operation* loop = s.loopExits[n];
        if (loop && readerLoops.contains(loop)) {
            for (const auto& domain : s.guardDomains) {
                if (
                    domain.tripShapeOf == loop && equalFact(s.guards[n], domain.identity, 0) == 1) {
                    facts.emptyConsumers.set(n);
                }
            }
        }
        for (Operation* readerLoop : readerLoops) {
            if (s.blockExits[n] && s.blockExits[n] == readerLoop->getBlock()) {
                facts.flushReaderless.set(n);
            }
        }
    }
    LifecycleCertificate cert;
    if (channel.bufferGenerations) {
        Bits boundaries(size);
        for (unsigned n = 0; n < size; ++n)
            if (s.blockExits[n] || s.program.nodes[n].kind == Node::Kind::Exit)
                boundaries.set(n);
        channel.generations = analyzeBufferGenerations(s.program, spec, boundaries, budget);
        cert = channel.generations.certificate;
    } else {
        cert = recognizeBoundaryLifecycle(s.program, spec, facts, budget);
    }
    channel.logical.certificate = cert;
    if (cert.status != LifecycleCertificate::Status::Complete) {
        reason = cert.reason;
        return false;
    }
    if (!cert.mayReuse) {
        reason = "one-shot storage remains with general insertion";
        return false;
    }
    channel.consumerRegions = readerLoops.size();
    std::vector<Point> points;
    // Group all represented occurrences of each physical phase. False cases are
    // retained; they are as important as true cases for guarded participation.
    for (unsigned p = 0; p < spec.phases.size(); ++p) {
        Point point;
        point.anchor = s.phases[p]->elementOp;
        for (unsigned n = 0; n < size; ++n) {
            if (s.program.nodes[n].kind == Node::Kind::Issue && s.program.nodes[n].phase == p) {
                point.nodes.push_back(n);
            }
        }
        if (!point.nodes.empty()) {
            points.push_back(std::move(point));
        }
    }
    for (unsigned n = 0; n < size; ++n) {
        if (s.program.nodes[n].kind == Node::Kind::Issue) {
            continue;
        }
        if (s.program.nodes[n].kind != Node::Kind::Exit && !s.blockExits[n]) {
            continue;
        }
        Operation* anchor = s.program.nodes[n].kind == Node::Kind::Exit ? s.anchors[n] : nullptr;
        Block* end = anchor ? nullptr : s.blockExits[n];
        auto it = std::find_if(
            points.begin(), points.end(), [&](const Point& p) { return p.anchor == anchor && p.blockEnd == end; });
        if (it == points.end()) {
            points.push_back({anchor, end, {n}});
        } else {
            it->nodes.push_back(n);
        }
    }
    DominanceInfo dominance(function);
    for (const Point& point : points) {
        for (auto role :
             {LifecyclePlacement::Role::PublishBefore, LifecyclePlacement::Role::ReleaseBefore,
              LifecyclePlacement::Role::BypassReady, LifecyclePlacement::Role::AcquireFree,
              LifecyclePlacement::Role::AcquireReady, LifecyclePlacement::Role::PublishReady,
              LifecyclePlacement::Role::PublishFree}) {
            bool any = false;
            for (unsigned n : point.nodes) {
                any |= performs(cert.nodeRoles[n], role);
            }
            if (!any) {
                continue;
            }
            Operation* owner = point.anchor ? point.anchor : point.blockEnd->getParentOp();
            if (owner != s.lifetimeScope && !s.lifetimeScope->isAncestor(owner)) {
                channel.logical.certificate.status = LifecycleCertificate::Status::Unsupported;
                reason = "required cleanup escapes the selected physical context";
                channel.placements.clear();
                return false;
            }
            auto tests = testsFor(s, point, function, dominance);
            std::vector<RoleCase> cases;
            for (unsigned n : point.nodes) {
                RoleCase item;
                item.performs = performs(cert.nodeRoles[n], role);
                for (const auto& test : tests) {
                    item.facts.push_back(truth(s, n, test));
                }
                cases.push_back(std::move(item));
            }
            auto guard = synthesizeRoleGuard(cases, budget);
            if (guard.status != GuardedRole::Status::Complete) {
                channel.logical.certificate.status = guard.status == GuardedRole::Status::InvalidInput ?
                                                         LifecycleCertificate::Status::InvalidInput :
                                                         (guard.status == GuardedRole::Status::AnalysisLimit ?
                                                              LifecycleCertificate::Status::AnalysisLimit :
                                                              LifecycleCertificate::Status::Unsupported);
                reason = "first/final/bypass predicate unavailable or not proved for all occurrences";
                channel.placements.clear();
                return false;
            }
            bool conditional = !(guard.alternatives.size() == 1 && guard.alternatives.front().empty());
            channel.guardedActions += conditional;
            channel.placements.push_back(
                {role, point.anchor, point.blockEnd,
                 role == LifecyclePlacement::Role::PublishReady || role == LifecyclePlacement::Role::PublishFree,
                 std::move(tests), std::move(guard)});
        }
    }
    reason = cert.reason;
    return true;
}

namespace {
struct IntegerBounds {
    int64_t lo, hi;
};
std::optional<IntegerBounds> scalarBounds(Value value, unsigned depth = 0)
{
    if (!value || depth > 12) {
        return std::nullopt;
    }
    if (auto c = literal(value)) {
        return IntegerBounds{*c, *c};
    }
    if (auto integer = dyn_cast<IntegerType>(value.getType())) {
        unsigned width = integer.getWidth();
        // Only scalar integer SSA inputs, not an arbitrary tile-produced value.
        if (
            width > 1 && width <= 32 && isa<BlockArgument>(value) &&
            isa<func::FuncOp>(cast<BlockArgument>(value).getOwner()->getParentOp())) {
            int64_t limit = int64_t(1) << (width - 1);
            return IntegerBounds{-limit, limit - 1};
        }
    }
    if (auto cast = value.getDefiningOp<arith::IndexCastOp>()) {
        auto sourceType = dyn_cast<IntegerType>(cast.getIn().getType());
        if (
            sourceType && sourceType.getWidth() <= 32 && cast.getOut().getType().isIndex()) {
            return scalarBounds(cast.getIn(), depth + 1);
        }
        return std::nullopt;
    }
    if (auto div = value.getDefiningOp<arith::DivSIOp>()) {
        auto divisor = literal(div.getRhs());
        auto input = scalarBounds(div.getLhs(), depth + 1);
        if (divisor && *divisor > 0 && input) {
            return IntegerBounds{input->lo / *divisor, input->hi / *divisor};
        }
        return std::nullopt;
    }
    Value left, right;
    unsigned kind = 0;
    if (auto add = value.getDefiningOp<arith::AddIOp>()) {
        left = add.getLhs();
        right = add.getRhs();
        kind = 1;
    } else if (auto sub = value.getDefiningOp<arith::SubIOp>()) {
        left = sub.getLhs();
        right = sub.getRhs();
        kind = 2;
    } else if (auto mul = value.getDefiningOp<arith::MulIOp>()) {
        left = mul.getLhs();
        right = mul.getRhs();
        kind = 3;
    }
    if (!kind) {
        return std::nullopt;
    }
    auto a = scalarBounds(left, depth + 1), b = scalarBounds(right, depth + 1);
    if (!a || !b) {
        return std::nullopt;
    }
    __int128 lo = 0, hi = 0;
    if (kind == 1) {
        lo = __int128(a->lo) + b->lo;
        hi = __int128(a->hi) + b->hi;
    } else if (kind == 2) {
        lo = __int128(a->lo) - b->hi;
        hi = __int128(a->hi) - b->lo;
    } else {
        std::array<__int128, 4> products{
            __int128(a->lo) * b->lo, __int128(a->lo) * b->hi, __int128(a->hi) * b->lo, __int128(a->hi) * b->hi};
        lo = *std::min_element(products.begin(), products.end());
        hi = *std::max_element(products.begin(), products.end());
    }
    // Do not assert nsw/nuw or infer non-wrapping mathematical arithmetic from
    // a benchmark shape. Bound EVERY intermediate using the existing SSA type.
    if (
        lo < std::numeric_limits<int32_t>::min() || hi > std::numeric_limits<int32_t>::max()) {
        return std::nullopt;
    }
    return IntegerBounds{int64_t(lo), int64_t(hi)};
}
bool sameScalar(Value a, Value b)
{
    if (a == b) {
        return true;
    }
    auto x = literal(a), y = literal(b);
    return x && y && *x == *y;
}
} // namespace

std::optional<bool> mlir::pto::classifyInsertSyncLifecycleContinuation(
    Value condition, scf::ForOp loop, bool first, bool last)
{
    if (!boundaryLoop(loop)) {
        return std::nullopt;
    }
    auto cmp = condition.getDefiningOp<arith::CmpIOp>();
    if (!cmp) {
        return std::nullopt;
    }
    if (
        (cmp.getLhs() == loop.getInductionVar() && sameScalar(cmp.getRhs(), loop.getLowerBound())) ||
        (cmp.getRhs() == loop.getInductionVar() && sameScalar(cmp.getLhs(), loop.getLowerBound()))) {
        if (
            cmp.getPredicate() == arith::CmpIPredicate::eq) {
            return first;
        }
        if (
            cmp.getPredicate() == arith::CmpIPredicate::ne) {
            return !first;
        }
    }
    auto add = cmp.getLhs().getDefiningOp<arith::AddIOp>();
    if (
        !add || cmp.getRhs() != loop.getUpperBound()) {
        return std::nullopt;
    }
    bool next = (add.getLhs() == loop.getInductionVar() && sameScalar(add.getRhs(), loop.getStep())) ||
                (add.getRhs() == loop.getInductionVar() && sameScalar(add.getLhs(), loop.getStep()));
    if (!next) {
        return std::nullopt;
    }
    auto bound = scalarBounds(loop.getUpperBound());
    auto step = literal(loop.getStep());
    if (!bound || !step || !safeNextIterationPredicate(bound->hi, *step)) {
        return std::nullopt;
    }
    if (
        cmp.getPredicate() == arith::CmpIPredicate::slt) {
        return !last;
    }
    if (
        cmp.getPredicate() == arith::CmpIPredicate::sge) {
        return last;
    }
    return std::nullopt;
}
