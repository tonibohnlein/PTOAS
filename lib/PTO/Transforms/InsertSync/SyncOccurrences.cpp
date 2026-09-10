// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/InsertSync/SyncOccurrences.h"
#include "PTO/IR/PTO.h"
#include "mlir/Analysis/FlatLinearValueConstraints.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <limits>
#include <map>
#include <set>
#include <chrono>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::logical_sync;
using namespace mlir::presburger;
namespace {
struct Constraint { AffineExpr expression; bool equality = false; };
using Conjunction = SmallVector<Constraint>;
using Alternatives = SmallVector<Conjunction>;
std::optional<int64_t> literal(Value value) {
    APInt integer;
    if (!matchPattern(value, m_ConstantInt(&integer)) || !integer.isSignedIntN(64)) return {};
    return integer.getBitWidth() == 1 ? int64_t(integer.getZExtValue()) : integer.getSExtValue();
}
using Range = std::pair<__int128, __int128>;
constexpr __int128 coefficientLimit = INT64_MAX / 16;
// Bound int64 AffineExpr/flattening coefficients before invoking unchecked MLIR
// arithmetic. Runtime range safety does not imply analysis coefficient safety.
__int128 coefficientMass(AffineExpr expr, unsigned depth = 0) {
    if (depth == 64) return coefficientLimit + 1;
    if (auto c = dyn_cast<AffineConstantExpr>(expr))
        return c.getValue() < 0 ? -__int128(c.getValue()) : __int128(c.getValue());
    if (isa<AffineDimExpr, AffineSymbolExpr>(expr)) return 1;
    auto binary = cast<AffineBinaryOpExpr>(expr);
    auto a = coefficientMass(binary.getLHS(), depth + 1);
    auto b = coefficientMass(binary.getRHS(), depth + 1);
    if (a > coefficientLimit || b > coefficientLimit) return coefficientLimit + 1;
    switch (expr.getKind()) {
    case AffineExprKind::Add: return a + b;
    case AffineExprKind::Mul: return a * b;
    case AffineExprKind::Mod:
    case AffineExprKind::FloorDiv:
    case AffineExprKind::CeilDiv: return a + 2*b + 1;
    default: return coefficientLimit + 1;
    }
}
bool supportedInteger(Type type) {
    auto integer = dyn_cast<IntegerType>(type);
    return isa<IndexType>(type) || (integer && integer.getWidth() <= 64);
}
Range typeRange(Type type) {
    unsigned width = isa<IndexType>(type) ? 64 : cast<IntegerType>(type).getWidth();
    if (width == 1) return {0,1};
    return {-(__int128(1) << (width-1)), (__int128(1) << (width-1))-1};
}
std::optional<IntegerRelation> flatten(ArrayRef<Constraint> constraints, unsigned dims, unsigned symbols,
                                     MLIRContext* context) {
    SmallVector<AffineExpr> expressions;
    SmallVector<bool> equalities;
    for (const auto& c : constraints) { expressions.push_back(c.expression); equalities.push_back(c.equality); }
    if (expressions.empty()) { expressions.push_back(getAffineConstantExpr(0,context)); equalities.push_back(false); }
    auto set = IntegerSet::get(dims, symbols, expressions, equalities);
    FlatLinearConstraints flat(dims,symbols);
    std::vector<SmallVector<int64_t,8>> rows;
    if (failed(getFlattenedAffineExprs(set,&rows,&flat))) return {};
    for (unsigned i=0; i<rows.size(); ++i)
        if (equalities[i]) flat.addEquality(rows[i]); else flat.addInequality(rows[i]);
    return IntegerRelation(flat);
}
// Equality queries may receive two expressions sharing a DAG. Preserve the
// importer's coefficient/depth limits without revisiting every unfolded path.
// This cache belongs to ONE query and one MLIR context, never a global process.
class EqualityCoefficients {
public:
    struct Receipt { __int128 mass; unsigned height; uint64_t expandedNodes; };
private:
    RelationQueries& queries;
    llvm::DenseMap<AffineExpr, Receipt> memo;
    QueryStatus outcome = QueryStatus::Proved;
    static uint64_t addSaturated(uint64_t a, uint64_t b) {
        return a > std::numeric_limits<uint64_t>::max() - b ? std::numeric_limits<uint64_t>::max() : a + b;
    }
public:
    explicit EqualityCoefficients(RelationQueries& q) : queries(q) {}
    QueryStatus status() const { return outcome; }
    std::optional<Receipt> analyze(AffineExpr expr, unsigned depth = 0) {
        // Charge every actual call, including an edge into an already visited
        // node, before recursion or a cache lookup can perform additional work.
        if (!queries.spend(1)) { outcome = QueryStatus::BudgetExhausted; return {}; }
        if (depth >= 64) { outcome = QueryStatus::Unsupported; return {}; }
        if (auto found = memo.find(expr); found != memo.end()) {
            if (depth + found->second.height > 64) { outcome = QueryStatus::Unsupported; return {}; }
            return found->second;
        }
        Receipt result{1, 1, 1};
        if (auto c = dyn_cast<AffineConstantExpr>(expr)) {
            result.mass = c.getValue() < 0 ? -__int128(c.getValue()) : __int128(c.getValue());
        } else if (auto binary = dyn_cast<AffineBinaryOpExpr>(expr)) {
            auto a = analyze(binary.getLHS(), depth + 1);
            if (!a) return {};
            auto b = analyze(binary.getRHS(), depth + 1);
            if (!b) return {};
            result.height = 1 + std::max(a->height, b->height);
            result.expandedNodes = addSaturated(1, addSaturated(a->expandedNodes, b->expandedNodes));
            if (a->mass > coefficientLimit || b->mass > coefficientLimit) result.mass = coefficientLimit + 1;
            else switch (expr.getKind()) {
            case AffineExprKind::Add: result.mass = a->mass + b->mass; break;
            case AffineExprKind::Mul: result.mass = a->mass * b->mass; break;
            case AffineExprKind::Mod:
            case AffineExprKind::FloorDiv:
            case AffineExprKind::CeilDiv: result.mass = a->mass + 2 * b->mass + 1; break;
            default: result.mass = coefficientLimit + 1; break;
            }
        }
        result.mass = std::min(result.mass, coefficientLimit + 1);
        memo.try_emplace(expr, result);
        return result;
    }
};
class Importer {
    func::FuncOp function;
    MLIRContext* context;
    SyncOccurrences result;
    llvm::DenseMap<Value,AffineExpr> expressions;
    llvm::DenseSet<Value> unavailableExpressions;
    bool definitionDomainsComplete = false;
    llvm::DenseMap<Value,Range> ranges;
    llvm::DenseMap<Operation*, Alternatives> definitionDomains;
    struct Pending { Operation* op; Alternatives domain; SmallVector<AffineExpr> time; SmallVector<unsigned> enclosing; };
    SmallVector<Pending, 0> pending;
    SmallVector<std::pair<Value, Alternatives>, 0> predicates;
    SmallVector<Operation*> requested;
    SmallVector<Value> requestedScalars;
    llvm::SmallPtrSet<Operation*,32> physical;
    unsigned depth = 0, predicateDepth = 0, regionDepth = 0;
    uint64_t workLeft = 100000;
    AffineExpr constant(int64_t n) const { return getAffineConstantExpr(n,context); }
    bool fail(StringRef text) { if (result.reason.empty()) result.reason=text.str(); return false; }
    bool limit(StringRef text)
    {
        result.limitExceeded = true;
        return fail(text);
    }
    SyncOccurrences finish()
    {
        result.work = 100000 - workLeft;
        return std::move(result);
    }
    std::optional<AffineExpr> parameter(Value value) {
        if (!supportedInteger(value.getType())) return {};
        auto range=typeRange(value.getType());
        auto expr=getAffineSymbolExpr(result.parameters.size(),context);
        result.parameters.push_back(value); expressions[value]=expr; ranges[value]=range; return expr;
    }
    // Interval arithmetic forgets correlations such as lower <= iv < upper.
    // Recover a sufficient rational enclosure on the scalar's ORIGINAL
    // definition domain, never its later use domain or the predicate it defines.
    // A successful enclosure qualifies only this value; operand ranges remain
    // unchanged. Fresh emitted import repeats this same query.
    std::optional<Range> definitionRange(Operation* op, AffineExpr expr) {
        auto found = definitionDomains.find(op);
        if (found == definitionDomains.end() || coefficientMass(expr) > coefficientLimit)
            return {};
        const unsigned n = result.loops.size(), symbols = result.parameters.size();
        std::optional<Range> total;
        for (auto constraints : found->second) {
            constraints.push_back({expr - getAffineDimExpr(n, context), true});
            auto flat = flatten(constraints, n + 1, symbols, context);
            if (!flat) return {};
            for (unsigned s = 0; s < symbols; ++s) {
                auto r = ranges[result.parameters[s]];
                SmallVector<llvm::DynamicAPInt> row(flat->getNumCols());
                row[n + 1 + s] = 1; row.back() = -llvm::DynamicAPInt(int64_t(r.first));
                flat->addInequality(row);
                row[n + 1 + s] = -1; row.back() = llvm::DynamicAPInt(int64_t(r.second));
                flat->addInequality(row);
            }
            uint64_t cost = uint64_t(flat->getNumConstraints() + 1) * flat->getNumCols() * 2;
            if (cost > workLeft) { limit("definition range query budget"); return {}; }
            workLeft -= cost;
            Simplex simplex(*flat);
            if (simplex.isEmpty()) continue;
            SmallVector<llvm::DynamicAPInt> objective(flat->getNumCols());
            objective[n] = 1;
            auto lo = simplex.computeOptimum(Simplex::Direction::Down, objective);
            auto hi = simplex.computeOptimum(Simplex::Direction::Up, objective);
            if (!lo.isBounded() || !hi.isBounded()) return {};
            auto lower = presburger::ceil(*lo), upper = presburger::floor(*hi);
            if (lower < INT64_MIN || upper > INT64_MAX || lower > upper) return {};
            Range r{int64_t(lower), int64_t(upper)};
            total = total ? Range{std::min(total->first, r.first), std::max(total->second, r.second)} : r;
        }
        return total;
    }
    std::optional<AffineExpr> expression(Value value) {
        // Charge recursive visits, including cache hits. An unsupported shared
        // SSA DAG must not expand exponentially without spending import work.
        if (result.limitExceeded) return {};
        if (!workLeft) { limit("occurrence expression work limit"); return {}; }
        --workLeft;
        if (definitionDomainsComplete && unavailableExpressions.contains(value)) return {};
        auto normalized = expressionImpl(value);
        // During the control walk, an original definition domain may not yet
        // have been recorded. Only cache definite unknowns after that walk;
        // depth/work failures remain explicit limits, never cached unknowns.
        if (definitionDomainsComplete && !normalized && !result.limitExceeded)
            unavailableExpressions.insert(value);
        return normalized;
    }
    std::optional<AffineExpr> expressionImpl(Value value) {
        if (!supportedInteger(value.getType())) return {};
        if (auto found=expressions.find(value); found!=expressions.end()) return found->second;
        if (depth >= 64) {
            limit("occurrence expression nesting limit");
            return {};
        }
        llvm::SaveAndRestore<unsigned> recurse(depth,depth+1);
        if (auto n=literal(value)) {
            // AffineExpr coefficients are int64. Keep extreme constants as
            // actual SSA parameters with exact APInt bounds, avoiding overflow
            // when comparisons negate/subtract their affine constants.
            if (__int128(*n) < -coefficientLimit / 4 || __int128(*n) > coefficientLimit / 4) {
                auto result = parameter(value); ranges[value]={*n,*n}; return result;
            }
            ranges[value]={*n,*n}; return constant(*n);
        }
        if (auto argument=dyn_cast<BlockArgument>(value)) {
            if (argument.getOwner()==&function.getBody().front()) return parameter(value);
            return {};
        }
        Operation* op=value.getDefiningOp();
        if (!op) return {};
        // A pure root-block scalar is one actual runtime value. Treating it as
        // a parameter avoids inventing mathematical semantics for its wrapping
        // arithmetic. No loop-local expression becomes an invariant parameter.
        if (op->getBlock()==&function.getBody().front() && !op->getNumRegions() && isMemoryEffectFree(op))
            return parameter(value);
        if (isa<arith::IndexCastOp,arith::ExtSIOp,arith::ExtUIOp>(op)) {
            auto input=expression(op->getOperand(0));
            if (!input || !ranges.count(op->getOperand(0))) return {};
            auto r=ranges[op->getOperand(0)], bound=typeRange(value.getType());
            if (op->getOperand(0).getType().isInteger(1) && isa<arith::ExtSIOp,arith::IndexCastOp>(op)) {
                *input = -*input; r={-r.second,-r.first};
            }
            if (isa<arith::ExtUIOp>(op) && r.first<0) return {};
            if (r.first<bound.first || r.second>bound.second) return {};
            ranges[value]=r; expressions[value]=*input; return input;
        }
        if (op->getNumOperands()!=2 || value.getType().isInteger(1)) return {};
        auto a=expression(op->getOperand(0)), b=expression(op->getOperand(1));
        if (!a || !b) return {};
        auto ra=ranges.find(op->getOperand(0)), rb=ranges.find(op->getOperand(1));
        if (ra==ranges.end() || rb==ranges.end()) return {};
        auto x=ra->second, y=rb->second;
        auto massA=coefficientMass(*a), massB=coefficientMass(*b);
        if (massA + massB > coefficientLimit) return {};
        AffineExpr expr;
        Range r;
        auto bound=typeRange(value.getType());
        if (isa<arith::AddIOp>(op)) {
            r={x.first+y.first,x.second+y.second};
            expr=*a+*b;
        } else if (isa<arith::SubIOp>(op)) {
            r={x.first-y.second,x.second-y.first};
            expr=*a-*b;
        }
        else if (isa<arith::MulIOp>(op)) {
            auto k=literal(op->getOperand(1));
            if (!k || x.first*__int128(*k)<INT64_MIN || x.first*__int128(*k)>INT64_MAX ||
                x.second*__int128(*k)<INT64_MIN || x.second*__int128(*k)>INT64_MAX) return {};
            if (massA * (*k < 0 ? -__int128(*k) : __int128(*k)) > coefficientLimit) return {};
            expr=*a * *k; r={std::min(x.first* *k,x.second* *k),std::max(x.first* *k,x.second* *k)};
        } else if (isa<arith::RemUIOp,arith::RemSIOp,arith::DivUIOp,arith::DivSIOp>(op)) {
            auto k=literal(op->getOperand(1));
            if (!k || *k<=0 || massA + 2*__int128(*k) + 1 > coefficientLimit) return {};
            if (x.first < 0) {
                auto qualified = definitionRange(op, *a);
                if (!qualified || qualified->first < 0) return {};
                x = *qualified;
            }
            bool rem=isa<arith::RemUIOp,arith::RemSIOp>(op);
            expr=rem ? *a % *k : a->floorDiv(*k);
            r=rem ? Range{0,*k-1} : Range{x.first/ *k,x.second/ *k};
        } else if (isa<arith::AndIOp>(op)) {
            auto k=literal(op->getOperand(1));
            if (!k || *k<0 || *k==INT64_MAX || ((*k+1)&*k) || x.first<0) return {};
            if (massA + 2*(__int128(*k)+1) + 1 > coefficientLimit) return {};
            expr=*a % (*k+1); r={0,*k};
        } else return {};
        if (r.first<bound.first || r.second>bound.second) {
            auto qualified = definitionRange(op, expr);
            if (!qualified || qualified->first<bound.first || qualified->second>bound.second) return {};
            r = *qualified;
        }
        expressions[value]=expr; ranges[value]=r; return expr;
    }
    Alternatives product(const Alternatives& a,const Alternatives& b) {
        if (a.size() * b.size() > 256) {
            limit("occurrence guard product limit");
            return {};
        }
        Alternatives out;
        for (const auto& x:a) for (const auto& y:b) { Conjunction c=x; llvm::append_range(c,y); out.push_back(std::move(c)); }
        return out;
    }
    std::optional<Alternatives> predicate(Value value,bool truth) {
        if (predicateDepth >= 64 || workLeft == 0) {
            limit("occurrence predicate work/nesting limit");
            return {};
        }
        --workLeft;
        llvm::SaveAndRestore<unsigned> recurse(predicateDepth, predicateDepth+1);
        if (auto n=literal(value)) return bool(*n)==truth ? Alternatives{Conjunction{}} : Alternatives{};
        if (auto cmp=value.getDefiningOp<arith::CmpIOp>()) {
            auto a=expression(cmp.getLhs()), b=expression(cmp.getRhs());
            if (!a || !b) return {};
            auto comparison=cmp.getPredicate();
            if (cmp.getLhs().getType().isInteger(1) &&
                comparison!=arith::CmpIPredicate::eq && comparison!=arith::CmpIPredicate::ne) {
                // Boolean storage is 0/1; signed i1 comparisons see 0/-1.
                *a=-*a; *b=-*b;
            }
            if (!truth) comparison=arith::invertPredicate(comparison);
            switch (comparison) {
            case arith::CmpIPredicate::eq: return Alternatives{Conjunction{{*a-*b,true}}};
            case arith::CmpIPredicate::ne: return Alternatives{Conjunction{{*a-*b-1,false}},Conjunction{{*b-*a-1,false}}};
            case arith::CmpIPredicate::slt: return Alternatives{Conjunction{{*b-*a-1,false}}};
            case arith::CmpIPredicate::sle: return Alternatives{Conjunction{{*b-*a,false}}};
            case arith::CmpIPredicate::sgt: return Alternatives{Conjunction{{*a-*b-1,false}}};
            case arith::CmpIPredicate::sge: return Alternatives{Conjunction{{*a-*b,false}}};
            default: return {};
            }
        }
        Operation* op=value.getDefiningOp();
        if (op && value.getType().isInteger(1) && isa<arith::XOrIOp>(op)) {
            for (unsigned i=0;i<2;++i) if (auto c=literal(op->getOperand(i)))
                return predicate(op->getOperand(1-i),truth != bool(*c));
            return {};
        }
        if (op && value.getType().isInteger(1) && isa<arith::AndIOp,arith::OrIOp>(op)) {
            auto a=predicate(op->getOperand(0),truth), b=predicate(op->getOperand(1),truth);
            if (!a || !b) return {};
            if (isa<arith::AndIOp>(op)==truth) return product(*a,*b);
            if (a->size() + b->size() > 256) {
                limit("occurrence guard union limit");
                return {};
            }
            llvm::append_range(*a,*b); return a;
        }
        if (value.getType().isInteger(1)) if (auto expr=expression(value))
            return Alternatives{Conjunction{{*expr-constant(truth?1:0),true}}};
        return {};
    }
    bool visit(Region& region,SmallVector<AffineExpr> time,Alternatives domain,SmallVector<unsigned> enclosing) {
        if (regionDepth >= 64)
            return limit("occurrence region nesting limit");
        llvm::SaveAndRestore<unsigned> recurse(regionDepth,regionDepth+1);
        if (!llvm::hasSingleElement(region)) return fail("multi-block occurrence region");
        unsigned rank=0;
        for (Operation& op:region.front()) {
            if (workLeft == 0)
                return limit("occurrence import work limit");
            --workLeft;
            if (isa<arith::AddIOp, arith::SubIOp, arith::RemUIOp, arith::RemSIOp,
                    arith::DivUIOp, arith::DivSIOp>(op))
                definitionDomains.try_emplace(&op, domain);
            auto here=time; here.push_back(constant(rank++));
            if (auto loop=dyn_cast<scf::ForOp>(op)) {
                if (!supportedInteger(loop.getInductionVar().getType()) ||
                    loop.getInductionVar().getType().isInteger(1))
                    return fail("unsupported induction integer width");
                auto low=expression(loop.getLowerBound()), high=expression(loop.getUpperBound());
                auto step=literal(loop.getStep());
                if (!low || !high || !step || *step<=0 || __int128(*step)>coefficientLimit/4 || loop->hasAttr("unsignedCmp"))
                    return fail("unqualified occurrence loop bounds/step");
                unsigned dim=llvm::find(result.loops,loop)-result.loops.begin();
                AffineExpr iv=getAffineDimExpr(dim,context);
                result.loopDomains[dim]={*low,*high,*step};
                expressions[loop.getInductionVar()]=iv;
                auto lowerRange=ranges[loop.getLowerBound()], upperRange=ranges[loop.getUpperBound()];
                if (upperRange.second+*step-1>typeRange(loop.getInductionVar().getType()).second)
                    return fail("loop increment can overflow its occurrence domain");
                ranges[loop.getInductionVar()]={lowerRange.first,upperRange.second-1};
                Conjunction bounds{{iv-*low,false},{*high-iv-1,false},{(iv-*low)% *step,true}};
                auto body=product(domain,Alternatives{bounds});
                auto nested=enclosing; nested.push_back(dim); here.push_back(iv);
                if (!visit(loop.getRegion(),here,body,nested)) return false;
                continue;
            }
            if (auto branch=dyn_cast<scf::IfOp>(op)) {
                auto yes=predicate(branch.getCondition(),true), no=predicate(branch.getCondition(),false);
                if (!yes || !no) return fail("unqualified occurrence predicate");
                if (llvm::none_of(predicates, [&](const auto& item) { return item.first == branch.getCondition(); }))
                    predicates.push_back({branch.getCondition(), *yes});
                auto branchTime=here; branchTime.push_back(constant(0));
                if (!visit(branch.getThenRegion(),branchTime,product(domain,*yes),enclosing)) return false;
                if (!branch.getElseRegion().empty()) {
                    branchTime.back()=constant(1);
                    if (!visit(branch.getElseRegion(),branchTime,product(domain,*no),enclosing)) return false;
                }
                continue;
            }
            if (isa<SectionCubeOp,SectionVectorOp>(op)) {
                if (!visit(op.getRegion(0),here,domain,enclosing)) return false;
                continue;
            }
            if (op.getNumRegions()) return fail("unmodeled occurrence region");
            if (physical.contains(&op)) pending.push_back({&op,domain,here,enclosing});
        }
        return true;
    }
public:
    Importer(func::FuncOp f,ArrayRef<Operation*> points, ArrayRef<Value> scalars)
        : function(f),context(f.getContext()),requested(points),requestedScalars(scalars) {
        for (Operation* op:points) if (!physical.insert(op).second)
            fail("multiple physical phases share one operation");
        f.walk<WalkOrder::PreOrder>([&](scf::ForOp loop){result.loops.push_back(loop);});
        result.loopDomains.resize(result.loops.size());
    }
    SyncOccurrences run() {
        if (!result.reason.empty())
            return finish();
        if (result.loops.size()>64 || physical.size()>1024 || requestedScalars.size()>256) {
            limit("occurrence projection size limit");
            return finish();
        }
        if (!visit(function.getBody(), {}, Alternatives{Conjunction{}}, {}))
            return finish();
        if (pending.size() != physical.size()) {
            fail("missing physical occurrence");
            return finish();
        }
        // Reuse the scalar interpreter only after visiting every original
        // definition domain, and before freezing the common parameter space.
        // Unknown syntax/ranges leave no scalar projection but preserve core
        // control import. Exhausted analysis limits remain explicit failures.
        definitionDomainsComplete = true;
        for (Value value : requestedScalars) {
            if (!value || result.scalarExpressions.count(value)) continue;
            if (!workLeft) { limit("optional scalar import work limit"); return finish(); }
            --workLeft;
            auto projected = expression(value);
            if (result.limitExceeded) return finish();
            if (projected) result.scalarExpressions.try_emplace(value, *projected);
        }
        unsigned dims=result.loops.size(), symbols=result.parameters.size();
        // The caller's phase identities survive insertion of synchronization
        // points. Schedule order is represented separately, never by this ID.
        llvm::DenseMap<Operation*,unsigned> positions;
        for (unsigned i=0;i<requested.size();++i) positions[requested[i]]=i;
        llvm::sort(pending,[&](const Pending& a,const Pending& b) { return positions[a.op]<positions[b.op]; });
        for (const auto& point:pending) {
            auto domain=PresburgerSet::getEmpty(PresburgerSpace::getSetSpace(dims,symbols));
            for (auto conjunction:point.domain) {
                for (unsigned d=0;d<dims;++d) if (!llvm::is_contained(point.enclosing,d))
                    conjunction.push_back({getAffineDimExpr(d,context),true});
                auto flat=flatten(conjunction,dims,symbols,context);
                if (!flat) {
                    fail("non-affine occurrence constraint");
                    return finish();
                }
                for (unsigned s=0;s<symbols;++s) {
                    Range bound=ranges[result.parameters[s]];
                    SmallVector<llvm::DynamicAPInt> row(flat->getNumCols());
                    row[dims+s]=1; row.back()=-llvm::DynamicAPInt(int64_t(bound.first)); flat->addInequality(row);
                    row[dims+s]=-1; row.back()=llvm::DynamicAPInt(int64_t(bound.second)); flat->addInequality(row);
                }
                domain.unionInPlace(*flat);
            }
            result.ids[point.op]=result.points.size();
            result.scheduleDimensions=std::max(result.scheduleDimensions,unsigned(point.time.size()));
            result.points.push_back({point.op,std::move(domain),point.time});
        }
        for (auto& point:result.points) while (point.schedule.size()<result.scheduleDimensions)
            point.schedule.push_back(constant(0));
        // Reuse the same normalization for boundary lowering. These predicates
        // carry no enclosing path assumption: a client must intersect with its
        // actual insertion domain and check SSA dominance before materializing.
        for (const auto& [value, alternatives] : predicates) {
            auto domain = PresburgerSet::getEmpty(PresburgerSpace::getSetSpace(dims, symbols));
            for (const auto& conjunction : alternatives) {
                auto piece = flatten(conjunction, dims, symbols, context);
                if (!piece) {
                    fail("non-affine boundary predicate");
                    return finish();
                }
                domain.unionInPlace(*piece);
            }
            result.predicates.push_back({value, std::move(domain)});
        }
        result.complete = result.reason.empty();
        return finish();
    }
};
}

SyncOccurrences SyncOccurrences::build(func::FuncOp f,ArrayRef<Operation*> points, ArrayRef<Value> scalars) {
    return Importer(f,points,scalars).run();
}

Relation SyncOccurrences::domain(unsigned p) const {
    auto set=points[p].domain;
    set.insertVarInPlace(VarKind::SetDim,0);
    auto result=Relation::getEmpty(PresburgerSpace::getRelationSpace(0,dimensions(),parameters.size()));
    for (auto piece:set.getAllDisjuncts()) { piece.addBound(BoundType::EQ,0,p); result.unionInPlace(piece); }
    return result;
}

Relation SyncOccurrences::identity(unsigned p) const {
    auto result=domain(p);
    result.insertVarInPlace(VarKind::Domain,0,dimensions());
    auto out=Relation::getEmpty(result.getSpace());
    for (auto piece:result.getAllDisjuncts()) {
        for (unsigned d=0;d<dimensions();++d) {
            SmallVector<int64_t> row(piece.getNumCols()); row[d]=1; row[dimensions()+d]=-1; piece.addEquality(row);
        }
        out.unionInPlace(piece);
    }
    return out;
}

Relation SyncOccurrences::predicateDomain(unsigned predicate, unsigned point) const
{
    auto set = predicates[predicate].whenTrue;
    set.insertVarInPlace(VarKind::SetDim, 0);
    return domain(point).intersect(set);
}

RelationResult SyncOccurrences::scalarDomain(Value value, unsigned point, int64_t lower,
                                             int64_t upper, RelationQueries& queries) const {
    if (!complete || point >= points.size())
        return {QueryStatus::Unsupported, {}, "scalar query requires a complete occurrence point"};
    auto found = scalarExpressions.find(value);
    if (found == scalarExpressions.end())
        return {QueryStatus::Unsupported, {}, "requested scalar normalization unavailable"};
    DominanceInfo dominance;
    if (!dominance.dominates(value, points[point].operation))
        return {QueryStatus::Unsupported, {}, "scalar does not dominate the queried occurrence"};
    if (!queries.spend(dimensions() + parameters.size() + 1))
        return {QueryStatus::BudgetExhausted, {}, "scalar domain setup budget"};
    auto ambient = domain(point);
    if (lower > upper) return queries.normalize(Relation::getEmpty(ambient.getSpace()));
    auto* context = points[point].operation->getContext();
    SmallVector<AffineExpr> dims, symbols;
    for (unsigned d = 0; d < loops.size(); ++d) dims.push_back(getAffineDimExpr(d + 1, context));
    for (unsigned s = 0; s < parameters.size(); ++s) symbols.push_back(getAffineSymbolExpr(s, context));
    auto expr = found->second.replaceDimsAndSymbols(dims, symbols);
    if (coefficientMass(expr) > coefficientLimit)
        return {QueryStatus::Unsupported, {}, "scalar coefficient bound unavailable"};
    auto set = IntegerSet::get(dimensions(), parameters.size(), {expr}, {false});
    FlatLinearConstraints flat(dimensions(), parameters.size());
    std::vector<SmallVector<int64_t, 8>> flattened;
    if (failed(getFlattenedAffineExprs(set, &flattened, &flat)))
        return {QueryStatus::Unsupported, {}, "scalar domain flattening unavailable"};
    // Add query constants as APInts after flattening; subtracting INT64_MIN
    // inside an AffineExpr would overflow before Presburger could check it.
    SmallVector<llvm::DynamicAPInt> low, high;
    for (int64_t coefficient : flattened.front()) {
        low.emplace_back(coefficient);
        high.emplace_back(-llvm::DynamicAPInt(coefficient));
    }
    low.back() -= llvm::DynamicAPInt(lower);
    high.back() += llvm::DynamicAPInt(upper);
    flat.addInequality(low);
    flat.addInequality(high);
    uint64_t cells = uint64_t(flat.getNumCols() + 1) * (flat.getNumConstraints() + 1) *
                     (ambient.getNumDisjuncts() + 1);
    if (!queries.spend(cells))
        return {QueryStatus::BudgetExhausted, {}, "scalar domain intersection budget"};
    return queries.normalize(ambient.intersect(Relation(flat)));
}

RelationResult SyncOccurrences::scalarEqualityConstraint(Value sourceValue, unsigned sourcePoint,
                                                        Value targetValue, unsigned targetPoint,
                                                        RelationQueries& queries) const {
    if (!complete)
        return {limitExceeded ? QueryStatus::BudgetExhausted : QueryStatus::Unsupported, {},
                "scalar equality requires complete occurrence import"};
    if (sourcePoint >= points.size() || targetPoint >= points.size() ||
        !points[sourcePoint].operation || !points[targetPoint].operation)
        return {QueryStatus::Unsupported, {}, "scalar equality requires valid occurrence points"};
    auto source = scalarExpressions.find(sourceValue), target = scalarExpressions.find(targetValue);
    if (source == scalarExpressions.end() || target == scalarExpressions.end())
        return {QueryStatus::Unsupported, {}, "scalar equality normalization unavailable"};
    DominanceInfo dominance;
    if (!dominance.dominates(sourceValue, points[sourcePoint].operation) ||
        !dominance.dominates(targetValue, points[targetPoint].operation))
        return {QueryStatus::Unsupported, {}, "scalar equality operand does not dominate its occurrence"};
    const unsigned n = dimensions(), symbols = parameters.size();
    if (!queries.spend(2 * n + symbols + 1))
        return {QueryStatus::BudgetExhausted, {}, "scalar equality setup budget"};
    // Check before forming the difference: mathematical range qualification of
    // the original SSA values does not bound AffineExpr's int64 coefficients.
    EqualityCoefficients coefficients(queries);
    auto sourceSize = coefficients.analyze(source->second);
    if (!sourceSize)
        return {coefficients.status(), {}, "scalar equality coefficient traversal unavailable"};
    auto targetSize = coefficients.analyze(target->second);
    if (!targetSize)
        return {coefficients.status(), {}, "scalar equality coefficient traversal unavailable"};
    if (sourceSize->mass > coefficientLimit || targetSize->mass > coefficientLimit ||
        sourceSize->mass + targetSize->mass > coefficientLimit)
        return {QueryStatus::Unsupported, {}, "scalar equality coefficient bound unavailable"};
    // Dimension replacement and MLIR flattening may unfold a shared expression.
    // Charge their conservative tree-visit bound before entering those APIs;
    // a small DAG cannot conceal exponential work after our memoized walk.
    for (unsigned pass = 0; pass < 2; ++pass)
        for (uint64_t amount : {sourceSize->expandedNodes, targetSize->expandedNodes})
            if (!queries.spend(amount))
                return {QueryStatus::BudgetExhausted, {}, "scalar equality affine traversal budget"};
    auto* context = points[sourcePoint].operation->getContext();
    SmallVector<AffineExpr> sourceDims, targetDims, syms;
    for (unsigned d = 0; d < loops.size(); ++d) {
        sourceDims.push_back(getAffineDimExpr(1 + d, context));
        targetDims.push_back(getAffineDimExpr(n + 1 + d, context));
    }
    for (unsigned s = 0; s < symbols; ++s) syms.push_back(getAffineSymbolExpr(s, context));
    auto a = source->second.replaceDimsAndSymbols(sourceDims, syms);
    auto b = target->second.replaceDimsAndSymbols(targetDims, syms);
    // Flatten one equality in the full source/range space. Total floor/modulo
    // witness definitions remain attached; there is no integer projection or
    // complement, and no identification of the two loop-coordinate tuples.
    auto flat = flatten({Constraint{a - b, true}}, 2 * n, symbols, context);
    if (!flat)
        return {QueryStatus::Unsupported, {}, "scalar equality flattening unavailable"};
    flat->setSpace(PresburgerSpace::getRelationSpace(n, n, symbols, flat->getNumLocalVars()));
    if (!queries.spend(uint64_t(flat->getNumCols()) * (flat->getNumConstraints() + 1)))
        return {QueryStatus::BudgetExhausted, {}, "scalar equality constraint budget"};
    return {QueryStatus::Proved, Relation(*flat), {}};
}

RelationResult SyncOccurrences::equalScalars(Value sourceValue, unsigned sourcePoint,
                                            Value targetValue, unsigned targetPoint,
                                            RelationQueries& queries) const {
    auto equality = scalarEqualityConstraint(sourceValue, sourcePoint, targetValue, targetPoint, queries);
    if (!equality) return equality;
    const unsigned n = dimensions(), symbols = parameters.size();
    const auto& flat = equality.relation->getAllDisjuncts().front();
    // Charge the potentially multiplicative ambient-domain product BEFORE
    // constructing it. Overflow is exhaustion, never a small wrapped charge.
    uint64_t rows = flat.getNumConstraints() + 3, locals = flat.getNumLocalVars();
    for (unsigned point : {sourcePoint, targetPoint}) {
        unsigned maxRows = 0, maxLocals = 0;
        for (const auto& piece : points[point].domain.getAllDisjuncts()) {
            if (!queries.spend(1))
                return {QueryStatus::BudgetExhausted, {}, "scalar equality domain indexing budget"};
            maxRows = std::max(maxRows, piece.getNumConstraints());
            maxLocals = std::max(maxLocals, piece.getNumLocalVars());
        }
        rows += maxRows;
        locals += maxLocals;
    }
    uint64_t cells = rows;
    for (uint64_t factor : {uint64_t(2 * n + symbols + 1) + locals,
                            uint64_t(points[sourcePoint].domain.getNumDisjuncts()) + 1,
                            uint64_t(points[targetPoint].domain.getNumDisjuncts()) + 1}) {
        if (factor && cells > std::numeric_limits<uint64_t>::max() / factor)
            return {QueryStatus::BudgetExhausted, {}, "scalar equality domain product overflow"};
        cells *= factor;
    }
    if (!queries.spend(cells))
        return {QueryStatus::BudgetExhausted, {}, "scalar equality intersection budget"};
    auto left = domain(sourcePoint);
    left.inverse();
    left.insertVarInPlace(VarKind::Range, 0, n);
    auto right = domain(targetPoint);
    right.insertVarInPlace(VarKind::Domain, 0, n);
    return queries.normalize(left.intersect(right).intersect(*equality.relation));
}

RelationResult SyncOccurrences::filterEqualScalars(const Relation& originalOccurrences,
                                                  Value sourceValue, unsigned sourcePoint,
                                                  Value targetValue, unsigned targetPoint,
                                                  RelationQueries& queries) const {
    auto equality = scalarEqualityConstraint(sourceValue, sourcePoint, targetValue, targetPoint, queries);
    if (!equality) return equality;
    if (!originalOccurrences.getSpace().isCompatible(equality.relation->getSpace()))
        return {QueryStatus::Unsupported, {}, "scalar equality filter occurrence space differs"};
    const auto& atom = equality.relation->getAllDisjuncts().front();
    // The caller retains the original qualified point domains. This operation
    // only adds a necessary-and-sufficient equality atom with its floor locals;
    // it does not reconstruct, relax, or reinterpret the caller's conditions.
    for (const auto& piece : originalOccurrences.getAllDisjuncts()) {
        uint64_t rows = uint64_t(piece.getNumConstraints()) + atom.getNumConstraints() + 1;
        uint64_t cols = uint64_t(piece.getNumCols()) + atom.getNumLocalVars();
        if (cols && rows > std::numeric_limits<uint64_t>::max() / cols)
            return {QueryStatus::BudgetExhausted, {}, "scalar equality filter product overflow"};
        if (!queries.spend(rows * cols))
            return {QueryStatus::BudgetExhausted, {}, "scalar equality filter intersection budget"};
    }
    return queries.normalize(originalOccurrences.intersect(*equality.relation));
}

// Exact optional cell normalization; deliberately does not call projectOut.
namespace {
struct PeriodicCellResult {
    QueryStatus status;
    std::optional<IntegerRelation> cell;
};

class PeriodicCellElimination {
    using Int = llvm::DynamicAPInt;
    using Row = std::vector<Int>;
    using Rows = std::vector<Row>;
    RelationQueries& queries;
    QueryStatus state = QueryStatus::Proved;
    unsigned variables;
    Rows equations, inequalities;
    bool empty = false;
    static constexpr unsigned maxRows = 256, maxColumns = 64, maxLocals = 16;

    bool spend(uint64_t work) {
        if (state != QueryStatus::Proved) return false;
        if (!queries.spend(work)) { state = QueryStatus::BudgetExhausted; return false; }
        return true;
    }
    bool bounded(const Int& value) {
        // Input and every stored intermediate have bounded numerical width.
        // A multiply of two qualified cells is therefore itself bounded before
        // this check, independently of DynamicAPInt's storage representation.
        const Int bound = Int(INT64_MAX) * Int(INT64_MAX);
        if (value < -bound || value > bound) { state = QueryStatus::Unsupported; return false; }
        return true;
    }
    Int canonical(const Int& value) {
        // Rebuild at most 126 numerical bits in a fixed-size representation;
        // a numerically small DynamicAPInt can otherwise retain wide storage.
        const Int base = Int(INT64_MAX) + Int(1);
        return Int(int64_t(value / base)) * base + Int(int64_t(value % base));
    }
    Int abs(Int value) { return value < 0 ? -value : value; }
    std::optional<Int> gcd(Int a, Int b) {
        a = abs(a); b = abs(b);
        while (b != 0) {
            if (!spend(1)) return {};
            Int remainder = a % b; a = b; b = remainder;
        }
        return a;
    }
    bool normalizeRow(Row& row, bool equality) {
        if (!spend(row.size())) return false;
        Int divisor(0);
        for (unsigned c = 0; c < variables; ++c) {
            if (!bounded(row[c])) return false;
            row[c] = canonical(row[c]);
            auto next = gcd(divisor, row[c]);
            if (!next) return false;
            divisor = *next;
        }
        if (!bounded(row.back())) return false;
        row.back() = canonical(row.back());
        if (divisor == 0) {
            if (equality ? row.back() != 0 : row.back() < 0) empty = true;
            row.clear(); return true;
        }
        if (equality && row.back() % divisor != 0) {
            empty = true; row.clear(); return true;
        }
        for (unsigned c = 0; c < variables; ++c) row[c] /= divisor;
        auto constant = row.back() / divisor;
        // C++/DynamicAPInt signed division truncates; integer inequality
        // normalization requires floor(constant / positive divisor).
        if (!equality && row.back() < 0 && row.back() % divisor != 0) --constant;
        row.back() = constant;
        if (equality) {
            for (unsigned c = 0; c < variables; ++c) if (row[c] != 0) {
                if (row[c] < 0) for (auto& value : row) value = -value;
                break;
            }
        }
        return true;
    }
    bool normalize() {
        if (equations.size() + inequalities.size() > maxRows) {
            state = QueryStatus::Unsupported; return false;
        }
        unsigned logarithm = 1;
        for (size_t count = equations.size() + inequalities.size(); count; count /= 2) ++logarithm;
        if (!spend(uint64_t(equations.size() + inequalities.size()) * (variables + 1) * logarithm)) return false;
        std::set<Row> uniqueEquations;
        // Equal coefficient vectors share the strongest (smallest) constant.
        std::map<Row, Int> strongest;
        for (auto& row : equations) {
            if (!normalizeRow(row, true)) return false;
            if (!row.empty()) uniqueEquations.insert(std::move(row));
        }
        for (auto& row : inequalities) {
            if (!normalizeRow(row, false)) return false;
            if (row.empty()) continue;
            Int constant = row.back(); row.pop_back();
            auto [found, inserted] = strongest.emplace(std::move(row), constant);
            if (!inserted && constant < found->second) found->second = constant;
        }
        // Opposite inequalities with opposite constants give an equality.
        // This is an exact row identity, not rational redundancy removal.
        inequalities.clear();
        for (const auto& [coefficients, constant] : strongest) {
            if (!spend(variables + 1)) return false;
            Row opposite = coefficients;
            for (auto& value : opposite) value = -value;
            auto other = strongest.find(opposite);
            Row row = coefficients; row.push_back(constant);
            if (other != strongest.end() && other->second + constant < 0) empty = true;
            if (other != strongest.end() && other->second == -constant) {
                if (!normalizeRow(row, true)) return false;
                if (!row.empty()) uniqueEquations.insert(std::move(row));
            } else inequalities.push_back(std::move(row));
        }
        equations.assign(uniqueEquations.begin(), uniqueEquations.end());
        return true;
    }
    bool eliminateEquality(unsigned column, unsigned pivot) {
        if (!spend(uint64_t(equations.size() + inequalities.size()) * (variables + 1))) return false;
        const Row source = equations[pivot];
        if (source[column] != 1 && source[column] != -1) {
            state = QueryStatus::Unsupported; return false;
        }
        auto substitute = [&](Row& row) {
            Int multiplier = row[column] * source[column];
            for (unsigned c = 0; c <= variables; ++c) {
                row[c] -= multiplier * source[c];
                if (!bounded(row[c])) return false;
            }
            return true;
        };
        for (unsigned r = 0; r < equations.size(); ++r)
            if (r != pivot && !substitute(equations[r])) return false;
        for (auto& row : inequalities) if (!substitute(row)) return false;
        equations.erase(equations.begin() + pivot);
        for (auto& row : equations) row.erase(row.begin() + column);
        for (auto& row : inequalities) row.erase(row.begin() + column);
        --variables;
        return normalize();
    }
    bool eliminateInequalities(unsigned column) {
        // Caller has established that no equality mentions this column.
        std::vector<unsigned> lower, upper;
        bool lowerUnit = true, upperUnit = true;
        for (unsigned r = 0; r < inequalities.size(); ++r) {
            const auto& coefficient = inequalities[r][column];
            if (coefficient > 0) { lower.push_back(r); lowerUnit &= coefficient == 1; }
            if (coefficient < 0) { upper.push_back(r); upperUnit &= coefficient == -1; }
        }
        // With an integral lower (or upper) frontier, rational nonemptiness
        // of this interval admits an integer endpoint. Two nonunit frontiers
        // would need divisibility reasoning, and are deliberately refused.
        if (!lower.empty() && !upper.empty() && !lowerUnit && !upperUnit) return false;
        uint64_t combinations = uint64_t(lower.size()) * upper.size();
        uint64_t outputRows = inequalities.size() - lower.size() - upper.size() + combinations;
        if (outputRows + equations.size() > maxRows) return false;
        if (!spend((outputRows + equations.size() + lower.size() + upper.size()) * (variables + 1))) return false;
        Rows next;
        next.reserve(outputRows);
        for (const auto& row : inequalities) if (row[column] == 0) next.push_back(row);
        for (unsigned a : lower) for (unsigned b : upper) {
            Row row(variables + 1);
            Int lowerMultiplier = -inequalities[b][column];
            Int upperMultiplier = inequalities[a][column];
            for (unsigned c = 0; c <= variables; ++c) {
                row[c] = lowerMultiplier * inequalities[a][c] + upperMultiplier * inequalities[b][c];
                if (!bounded(row[c])) return false;
            }
            next.push_back(std::move(row));
        }
        inequalities = std::move(next);
        for (auto& row : equations) row.erase(row.begin() + column);
        for (auto& row : inequalities) row.erase(row.begin() + column);
        --variables;
        return normalize();
    }
public:
    explicit PeriodicCellElimination(RelationQueries& queries) : queries(queries), variables(0) {}
    PeriodicCellResult run(const IntegerRelation& original, unsigned iteration) {
        auto refuse = [&]() -> PeriodicCellResult {
            return {state == QueryStatus::BudgetExhausted ? state : QueryStatus::Unsupported, {}};
        };
        if (original.getNumCols() + 1 > maxColumns || original.getNumLocalVars() > maxLocals ||
            original.getNumConstraints() > maxRows || iteration >= original.getNumDimAndSymbolVars()) return refuse();
        if (!spend(uint64_t(original.getNumCols()) * (original.getNumConstraints() + 1))) return refuse();
        variables = original.getNumVars();
        for (bool equality : {true, false}) {
            unsigned count = equality ? original.getNumEqualities() : original.getNumInequalities();
            for (unsigned r = 0; r < count; ++r) {
                auto input = equality ? original.getEquality(r) : original.getInequality(r);
                Row row(variables + 1);
                for (unsigned c = 0; c <= variables; ++c) {
                    if (!bounded(input[c])) return refuse();
                    row[c] = canonical(input[c]);
                }
                (equality ? equations : inequalities).push_back(std::move(row));
            }
        }
        // Recover exact implicit equalities BEFORE choosing the quotient.
        // For example i-2*q>=1 and i-2*q<=1 are the same congruence
        // as i-2*q-1=0; treating this cell as period one loses that structure.
        if (!normalize()) return refuse();
        if (empty) {
            IntegerRelation result(original.getSpaceWithoutLocals());
            SmallVector<Int> contradiction(result.getNumCols()); contradiction.back() = -1;
            result.addInequality(contradiction);
            return {QueryStatus::Proved, std::move(result)};
        }
        Int period(1), residue(0);
        // Pick one actual congruence. The new quotient is a distinct protected
        // coordinate, so signs, unnormalized residues and aliases of original
        // locals cannot accidentally change its meaning.
        if (!spend(uint64_t(equations.size()) * (variables + 1))) return refuse();
        for (const auto& row : equations) {
            if (row[iteration] != 1 && row[iteration] != -1) continue;
            std::optional<unsigned> local;
            bool pure = true;
            for (unsigned c = 0; c < variables; ++c) if (c != iteration && row[c] != 0) {
                if (c < original.getNumDimAndSymbolVars() || local) pure = false;
                else local = c;
            }
            if (!pure || !local) continue;
            period = abs(row[*local]);
            if (period > INT64_MAX) return refuse();
            residue = (-row.back() * row[iteration]) % period;
            if (residue < 0) residue += period;
            break;
        }
        if (!spend(uint64_t(equations.size() + inequalities.size()) * (variables + 2))) return refuse();
        const unsigned protectedQuotient = variables++;
        for (bool equality : {true, false}) {
            for (auto& row : equality ? equations : inequalities) {
                row.insert(row.end() - 1, Int(0));
                row.back() += row[iteration] * residue;
                row[protectedQuotient] = row[iteration] * period;
                row[iteration] = 0;
                if (!bounded(row.back()) || !bounded(row[protectedQuotient])) return refuse();
            }
        }
        if (!normalize()) return refuse();
        const unsigned firstLocal = original.getNumDimAndSymbolVars();
        // Every successful iteration eliminates exactly one ORIGINAL local.
        // The last column is the protected quotient and is never a candidate.
        for (unsigned remaining = original.getNumLocalVars(); remaining && !empty; --remaining) {
            bool changed = false;
            if (!spend(uint64_t(variables - firstLocal) * (equations.size() + inequalities.size() + 1))) return refuse();
            for (unsigned c = firstLocal; c + 1 < variables && !changed; ++c)
                for (unsigned r = 0; r < equations.size(); ++r)
                    if (equations[r][c] == 1 || equations[r][c] == -1) {
                        if (!eliminateEquality(c, r)) return refuse();
                        changed = true; break;
                    }
            for (unsigned c = firstLocal; c + 1 < variables && !changed; ++c) {
                bool mentioned = false;
                for (const auto& row : equations) mentioned |= row[c] != 0;
                if (mentioned) continue;
                if (eliminateInequalities(c)) changed = true;
                else if (state != QueryStatus::Proved) return refuse();
            }
            if (!changed) return refuse();
        }
        IntegerRelation result(original.getSpaceWithoutLocals());
        if (empty) {
            SmallVector<Int> contradiction(result.getNumCols()); contradiction.back() = -1;
            result.addInequality(contradiction);
            return {QueryStatus::Proved, std::move(result)};
        }
        if (variables != firstLocal + 1) return refuse();
        result.appendVar(VarKind::Local);
        if (!spend(uint64_t(equations.size() + inequalities.size() + 1) * result.getNumCols())) return refuse();
        // Replace q by (i-r)/p in each remaining row using positive scaling.
        // The separate anchor retains exact divisibility; no rational
        // projection or floor constraint is silently removed.
        for (bool equality : {true, false}) {
            for (const auto& source : equality ? equations : inequalities) {
                Row row = source;
                Int quotient = row[firstLocal];
                for (auto& coefficient : row) {
                    coefficient *= period;
                    if (!bounded(coefficient)) return refuse();
                }
                row[firstLocal] = 0;
                row[iteration] += quotient;
                row.back() -= quotient * residue;
                if (!bounded(row[iteration]) || !bounded(row.back()) || !normalizeRow(row, equality)) return refuse();
                if (row.empty()) continue;
                if (equality) result.addEquality(row); else result.addInequality(row);
            }
        }
        if (empty) {
            IntegerRelation contradiction(result.getSpaceWithoutLocals());
            SmallVector<Int> row(contradiction.getNumCols()); row.back() = -1;
            contradiction.addInequality(row);
            return {QueryStatus::Proved, std::move(contradiction)};
        }
        SmallVector<Int> anchor(result.getNumCols());
        anchor[iteration] = 1; anchor[firstLocal] = -period; anchor.back() = -residue;
        result.addEquality(anchor);
        return {QueryStatus::Proved, std::move(result)};
    }
};
}

RelationResult mlir::pto::logical_sync::normalizeOccurrenceCell(
    const IntegerRelation& cell, unsigned iterationCoordinate, RelationQueries& queries) {
    auto simplified = PeriodicCellElimination(queries).run(cell, iterationCoordinate);
    if (simplified.cell) return {QueryStatus::Proved, Relation(*simplified.cell), {}};
    return {simplified.status, {}, simplified.status == QueryStatus::BudgetExhausted ?
        "periodic cell elimination budget" : "periodic cell requires unsupported integer elimination"};
}

RelationResult mlir::pto::logical_sync::testing::simplifyPeriodicCell(
    const IntegerRelation& cell, unsigned iterationCoordinate, RelationQueries& queries) {
    return normalizeOccurrenceCell(cell, iterationCoordinate, queries);
}

RelationResult SyncOccurrences::periodicSuccessors(const PresburgerSet& publications,
                                                   RelationQueries& queries) const {
    using llvm::DynamicAPInt;
    auto unsupported = [](StringRef reason) -> RelationResult {
        return {QueryStatus::Unsupported, {}, reason.str()};
    };
    auto exhausted = []() -> RelationResult {
        return {QueryStatus::BudgetExhausted, {}, "native periodic population qualification budget"};
    };
    const unsigned n = dimensions(), symbols = parameters.size();
    if (!complete || publications.getNumDomainVars() || publications.getNumRangeVars() != n ||
        publications.getNumSymbolVars() != symbols)
        return unsupported("native periodic occurrence universe unavailable");
    auto profileTime = std::chrono::steady_clock::now();
    uint64_t profileWork = queries.work();
    auto profile = [&](StringRef stage) {
        auto now = std::chrono::steady_clock::now();
        if (queries.profilingEnabled())
            llvm::errs() << "logical periodic_stage " << stage << " pieces " << publications.getNumDisjuncts()
                         << " work " << queries.work() - profileWork << " seconds "
                         << std::chrono::duration<double>(now - profileTime).count() << "\n";
        profileTime = now; profileWork = queries.work();
    };
    auto normalized = queries.normalize(publications);
    profile("normalize");
    if (!normalized) return normalized;
    if (!normalized.relation->getNumDisjuncts())
        return {QueryStatus::Proved, Relation::getEmpty(PresburgerSpace::getRelationSpace(n, n, symbols)), {}};
    // Exact cell elimination supplies another representation of the WHOLE
    // original population. Unsupported cells stay unchanged in this union.
    // Candidate extraction below may forget constraints, but its result is
    // compared in both directions with this exact union before construction.
    auto exactPopulation = PresburgerSet::getEmpty(normalized.relation->getSpace());
    using Row = std::vector<int64_t>;
    using Rows = std::set<std::pair<bool, Row>>;
    std::optional<Rows> commonRows;
    std::optional<unsigned> activeLoop, scheduleIV;
    std::optional<int64_t> period;
    std::vector<PeriodicPublication> atoms;
    std::map<unsigned, std::vector<int64_t>> phaseSchedules;
    std::vector<int64_t> schedulePrefix;
    for (const auto& originalPiece : normalized.relation->getAllDisjuncts()) {
        const uint64_t cells = uint64_t(originalPiece.getNumCols()) * (originalPiece.getNumConstraints() + 1);
        uint64_t treeCharge = 1;
        for (unsigned k = originalPiece.getNumConstraints(); k; k /= 2) treeCharge += 2;
        if (cells > UINT64_MAX / treeCharge || !queries.spend(cells * treeCharge)) return exhausted();
        auto piece = originalPiece;
        std::optional<unsigned> phase;
        for (unsigned r = 0; r < piece.getNumEqualities(); ++r) {
            auto row = piece.getEquality(r);
            if (row[0] != 1 && row[0] != -1) continue;
            bool isolated = true;
            for (unsigned c = 1; c < piece.getNumVars(); ++c) isolated &= row[c] == 0;
            if (!isolated) continue;
            DynamicAPInt value = -row.back() * row[0];
            if (value < 0 || value >= int64_t(points.size())) return unsupported("native periodic phase unavailable");
            unsigned p = int64_t(value);
            if (phase && *phase != p) return unsupported("native periodic contradictory phase");
            phase = p;
        }
        if (!phase) return unsupported("native periodic phase not fixed");
        if (!phaseSchedules.count(*phase)) {
            const auto& schedule = points[*phase].schedule;
            if (!queries.spend(schedule.size() + 1)) return exhausted();
            std::optional<unsigned> loop, ivPosition;
            std::vector<int64_t> constants;
            for (unsigned s = 0; s < schedule.size(); ++s) {
                if (auto iv = dyn_cast<AffineDimExpr>(schedule[s])) {
                    if (loop) return unsupported("native periodic nested invocation");
                    loop = iv.getPosition(); ivPosition = s; constants.push_back(0);
                } else if (auto c = dyn_cast<AffineConstantExpr>(schedule[s])) constants.push_back(c.getValue());
                else return unsupported("native periodic schedule is not constant/iteration");
            }
            if (!loop || *loop >= loopDomains.size() || loopDomains[*loop].step != 1)
                return unsupported("native periodic schedule needs one unit-step loop");
            if (activeLoop && (*activeLoop != *loop || *scheduleIV != *ivPosition))
                return unsupported("native periodic publications span different loop invocations");
            std::vector<int64_t> prefix(constants.begin(), constants.begin() + *ivPosition);
            if (activeLoop && prefix != schedulePrefix)
                return unsupported("native periodic publications have different schedule prefixes");
            activeLoop = loop; scheduleIV = ivPosition; schedulePrefix = std::move(prefix);
            phaseSchedules.emplace(*phase, std::move(constants));
        }
        unsigned iv = 1 + *activeLoop;
        if (piece.getNumLocalVars()) {
            auto simplified = PeriodicCellElimination(queries).run(piece, iv);
            if (simplified.status == QueryStatus::BudgetExhausted) return exhausted();
            if (simplified.cell) piece = std::move(*simplified.cell);
            // Unsupported local elimination keeps the previous proposal path;
            // its whole-population equality gate still decides qualification.
        }
        if (!queries.spend(uint64_t(piece.getNumCols()) * (piece.getNumConstraints() + 1) + 1)) return exhausted();
        // Only a proved empty conjunct can disappear from the population.
        // Empty cells need no residue/interval proposal of their own.
        if (piece.isObviouslyEmpty()) continue;
        exactPopulation.unionInPlace(piece);
        std::optional<std::pair<int64_t, int64_t>> congruence;
        Rows rows;
        for (bool equality : {true, false}) {
            unsigned count = equality ? piece.getNumEqualities() : piece.getNumInequalities();
            for (unsigned r = 0; r < count; ++r) {
                auto row = equality ? piece.getEquality(r) : piece.getInequality(r);
                bool small = true;
                for (const auto& coefficient : row)
                    small &= coefficient >= INT64_MIN && coefficient <= INT64_MAX;
                // Large range guards can be redundant under the selected loop
                // bounds (e.g. n >= INT64_MIN in a nonempty nonnegative loop).
                // Omit them only from the proposal; whole-domain equality must
                // still prove every original constraint before using it.
                if (!small) continue;
                if (row[0] != 0) continue;
                std::optional<unsigned> local;
                bool multipleLocals = false;
                for (unsigned c = n + symbols; c < piece.getNumVars(); ++c) if (row[c] != 0) {
                    if (local) multipleLocals = true;
                    local = c;
                }
                if (local) {
                    bool onlyIV = equality && !multipleLocals && (row[iv] == 1 || row[iv] == -1);
                    for (unsigned c = 0; c < n + symbols; ++c) if (c != iv && row[c] != 0) onlyIV = false;
                    if (onlyIV) {
                        auto modulus = row[*local] < 0 ? -row[*local] : row[*local];
                        if (modulus > INT64_MAX) return unsupported("native periodic modulus exceeds int64");
                        auto residue = (-row.back() * row[iv]) % modulus;
                        if (residue < 0) residue += modulus;
                        std::pair<int64_t, int64_t> value{int64_t(modulus), int64_t(residue)};
                        if (congruence && *congruence != value)
                            return unsupported("native periodic needs one common congruence");
                        congruence = value;
                    }
                    continue;
                }
                Row compact(n + symbols + 1);
                for (unsigned c = 0; c < n + symbols; ++c) compact[c] = int64_t(row[c]);
                compact.back() = int64_t(row.back());
                rows.emplace(equality, std::move(compact));
            }
        }
        auto [p, residue] = congruence.value_or(std::make_pair(int64_t(1), int64_t(0)));
        if (period && *period != p) return unsupported("native periodic population uses different periods");
        period = p;
        if (!commonRows) commonRows = std::move(rows);
        else {
            // One relation-derived envelope, not a list of guessed bounds:
            // retain coefficient patterns common to every cell and weaken an
            // inequality only to the weakest actual constant. Equalities must
            // be identical. The final bidirectional population proof rejects
            // any holes or extra occurrences introduced by this proposal.
            Rows envelope;
            for (const auto& [equality, row] : *commonRows) {
                unsigned lookupCharge = 1;
                for (size_t count = rows.size(); count; count /= 2) ++lookupCharge;
                if (!queries.spend(uint64_t(row.size()) * lookupCharge)) return exhausted();
                if (equality) {
                    if (rows.count({true, row})) envelope.emplace(true, row);
                    continue;
                }
                Row probe = row; probe.back() = INT64_MIN;
                auto other = rows.lower_bound({false, probe});
                if (other != rows.end() && !other->first && other->second.size() == row.size() &&
                    std::equal(row.begin(), row.end() - 1, other->second.begin())) {
                    auto combined = row;
                    combined.back() = std::max(combined.back(), other->second.back());
                    envelope.emplace(false, std::move(combined));
                }
            }
            commonRows = std::move(envelope);
        }
        atoms.push_back({int64_t(*phase), 0, residue});
    }
    if (atoms.empty())
        return {QueryStatus::Proved, Relation::getEmpty(PresburgerSpace::getRelationSpace(n, n, symbols)), {}};
    // Rank by the actual imported lexicographic schedule, independently of the
    // physical-point enumeration. One phase can contribute multiple residues.
    uint64_t sortCharge = 1;
    for (size_t k = phaseSchedules.size(); k; k /= 2) sortCharge += 2;
    if (!queries.spend(sortCharge * phaseSchedules.size() * (scheduleDimensions + 1))) return exhausted();
    std::map<std::vector<int64_t>, unsigned> scheduled;
    for (const auto& [phase, schedule] : phaseSchedules)
        if (!scheduled.emplace(schedule, phase).second) return unsupported("native periodic schedule ranks tied");
    std::map<unsigned, int64_t> ranks;
    for (const auto& [schedule, phase] : scheduled) ranks[phase] = ranks.size();
    for (auto& atom : atoms) atom.rank = ranks.at(unsigned(atom.phase));
    IntegerRelation common(PresburgerSpace::getSetSpace(n, symbols));
    for (const auto& [equality, row] : *commonRows)
        if (equality) common.addEquality(row); else common.addInequality(row);
    profile("extract");
    // Reject an unsupported model grammar before expensive population
    // equality. This result remains private until native qualification below.
    auto candidate = queries.commonPeriodSuccessors(PresburgerSet(Relation(common)), 0, 1 + *activeLoop, *period, atoms);
    profile("helper");
    if (!candidate) return candidate;
    auto proposed = PresburgerSet::getEmpty(common.getSpace());
    for (const auto& atom : atoms) {
        if (!queries.spend(uint64_t(common.getNumCols() + 1) * (common.getNumConstraints() + 3))) return exhausted();
        auto part = common;
        part.appendVar(VarKind::Local);
        part.addBound(BoundType::EQ, 0, atom.phase);
        SmallVector<DynamicAPInt> row(part.getNumCols());
        row[1 + *activeLoop] = 1; row[part.getNumVars() - 1] = -DynamicAPInt(*period);
        row.back() = -DynamicAPInt(atom.residue);
        part.addEquality(row);
        proposed.unionInPlace(part);
    }
    for (auto direction : {true, false}) {
        // The exact cell rewrite relates this representation back to the
        // untouched selected population. Re-expanding its original existential
        // witnesses here would repeat the eliminated integer problem.
        auto status = direction ? queries.contains(proposed, exactPopulation) : queries.contains(exactPopulation, proposed);
        if (status == QueryStatus::BudgetExhausted) return exhausted();
        if (status != QueryStatus::Proved) {
            profile("equality-unavailable");
            return unsupported("native periodic whole-population equality not established");
        }
    }
    profile("equality");
    return candidate;
}

RelationResult SyncOccurrences::ordered(unsigned source,unsigned target,bool inclusive) const {
    const unsigned n=dimensions(), symbols=parameters.size();
    auto left=domain(source); left.inverse(); left.insertVarInPlace(VarKind::Range,0,n);
    auto right=domain(target); right.insertVarInPlace(VarKind::Domain,0,n);
    auto base=left.intersect(right);
    SmallVector<AffineExpr> srcDims,tgtDims,syms;
    auto* context=points[source].operation->getContext();
    for (unsigned d=0;d<loops.size();++d) {
        srcDims.push_back(getAffineDimExpr(1+d,context)); tgtDims.push_back(getAffineDimExpr(n+1+d,context));
    }
    for (unsigned s=0;s<symbols;++s) syms.push_back(getAffineSymbolExpr(s,context));
    auto result=Relation::getEmpty(base.getSpace());
    Conjunction equalPrefix;
    for (unsigned t=0;t<scheduleDimensions;++t) {
        auto a=points[source].schedule[t].replaceDimsAndSymbols(srcDims,syms);
        auto b=points[target].schedule[t].replaceDimsAndSymbols(tgtDims,syms);
        auto difference = b-a;
        if (auto literal=dyn_cast<AffineConstantExpr>(difference)) {
            if (literal.getValue()==0) continue;
            if (literal.getValue()<0) return {QueryStatus::Proved,std::move(result),{}};
            auto flat=flatten(equalPrefix,2*n,symbols,context);
            if (!flat) return {QueryStatus::Unsupported,std::nullopt,"non-affine occurrence order"};
            flat->setSpace(PresburgerSpace::getRelationSpace(n,n,symbols,flat->getNumLocalVars()));
            result.unionInPlace(base.intersect(Relation(*flat)));
            return {QueryStatus::Proved,std::move(result),{}};
        }
        auto candidate=equalPrefix; candidate.push_back({difference-1,false});
        auto flat=flatten(candidate,2*n,symbols,context);
        if (!flat) return {QueryStatus::Unsupported,std::nullopt,"non-affine occurrence order"};
        flat->setSpace(PresburgerSpace::getRelationSpace(n,n,symbols,flat->getNumLocalVars()));
        result.unionInPlace(base.intersect(Relation(*flat)));
        equalPrefix.push_back({b-a,true});
    }
    if (inclusive) {
        auto flat=flatten(equalPrefix,2*n,symbols,context);
        if (!flat) return {QueryStatus::Unsupported,std::nullopt,"non-affine inclusive occurrence order"};
        flat->setSpace(PresburgerSpace::getRelationSpace(n,n,symbols,flat->getNumLocalVars()));
        result.unionInPlace(base.intersect(Relation(*flat)));
    }
    return {QueryStatus::Proved,std::move(result),{}};
}
