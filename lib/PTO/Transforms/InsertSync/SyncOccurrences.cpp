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
