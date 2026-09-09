// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
// See LICENSE for details. Provided AS IS, WITHOUT WARRANTIES OF ANY KIND.
#include "PTO/Transforms/InsertSync/SyncOccurrences.h"
#include "PTO/IR/PTO.h"
#include "mlir/Analysis/FlatLinearValueConstraints.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/SaveAndRestore.h"
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
class Importer {
    func::FuncOp function;
    MLIRContext* context;
    SyncOccurrences result;
    llvm::DenseMap<Value,AffineExpr> expressions;
    llvm::DenseMap<Value,Range> ranges;
    struct Pending { Operation* op; Alternatives domain; SmallVector<AffineExpr> time; SmallVector<unsigned> enclosing; };
    SmallVector<Pending, 0> pending;
    SmallVector<std::pair<Value, Alternatives>, 0> predicates;
    SmallVector<Operation*> requested;
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
    std::optional<AffineExpr> expression(Value value) {
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
            if (r.first<bound.first || r.second>bound.second) return {};
            expr=*a+*b;
        } else if (isa<arith::SubIOp>(op)) {
            r={x.first-y.second,x.second-y.first};
            if (r.first<bound.first || r.second>bound.second) return {};
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
            if (!k || *k<=0 || x.first<0 || massA + 2*__int128(*k) + 1 > coefficientLimit) return {};
            bool rem=isa<arith::RemUIOp,arith::RemSIOp>(op);
            expr=rem ? *a % *k : a->floorDiv(*k);
            r=rem ? Range{0,*k-1} : Range{x.first/ *k,x.second/ *k};
        } else if (isa<arith::AndIOp>(op)) {
            auto k=literal(op->getOperand(1));
            if (!k || *k<0 || *k==INT64_MAX || ((*k+1)&*k) || x.first<0) return {};
            if (massA + 2*(__int128(*k)+1) + 1 > coefficientLimit) return {};
            expr=*a % (*k+1); r={0,*k};
        } else return {};
        if (r.first<bound.first || r.second>bound.second) return {};
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
    Importer(func::FuncOp f,ArrayRef<Operation*> points):function(f),context(f.getContext()),requested(points) {
        for (Operation* op:points) if (!physical.insert(op).second)
            fail("multiple physical phases share one operation");
        f.walk<WalkOrder::PreOrder>([&](scf::ForOp loop){result.loops.push_back(loop);});
        result.loopDomains.resize(result.loops.size());
    }
    SyncOccurrences run() {
        if (!result.reason.empty())
            return finish();
        if (result.loops.size()>64 || physical.size()>1024) {
            limit("occurrence projection size limit");
            return finish();
        }
        if (!visit(function.getBody(), {}, Alternatives{Conjunction{}}, {}))
            return finish();
        if (pending.size() != physical.size()) {
            fail("missing physical occurrence");
            return finish();
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

SyncOccurrences SyncOccurrences::build(func::FuncOp f,ArrayRef<Operation*> points) { return Importer(f,points).run(); }

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
