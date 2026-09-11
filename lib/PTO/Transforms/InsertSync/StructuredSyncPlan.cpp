// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.


#include "PTO/Transforms/InsertSync/StructuredSyncPlan.h"
#include "PTO/Transforms/InsertSync/StructuredSyncCore.h"
#include "PTO/Transforms/InsertSync/SyncPhysicalFacts.h"
#include "PTO/Transforms/InsertSync/SyncAddressAnalysis.h"
#include "PTO/Transforms/InsertSync/SyncPayloadSnapshot.h"
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"
#include "PTO/Transforms/InsertSync/MemoryDependentAnalyzer.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <tuple>

using namespace mlir;
using namespace mlir::pto;
namespace ss = mlir::pto::structured_sync;
using Outcome = mlir::pto::logical_sync::ConstructionResult;
namespace {
std::optional<ss::Pipe> pipe(PIPE p) {
    switch(p) {
    case PIPE::PIPE_S: return ss::Pipe::S;
    case PIPE::PIPE_V: return ss::Pipe::V;
    case PIPE::PIPE_M: return ss::Pipe::M;
    case PIPE::PIPE_MTE1: return ss::Pipe::MTE1;
    case PIPE::PIPE_MTE2: return ss::Pipe::MTE2;
    case PIPE::PIPE_MTE3: return ss::Pipe::MTE3;
    case PIPE::PIPE_FIX: return ss::Pipe::FIX;
    default: return {};
    }
}
PIPE pipe(ss::Pipe p) {
    switch(p) {
    case ss::Pipe::S: return PIPE::PIPE_S;
    case ss::Pipe::V: return PIPE::PIPE_V;
    case ss::Pipe::M: return PIPE::PIPE_M;
    case ss::Pipe::MTE1: return PIPE::PIPE_MTE1;
    case ss::Pipe::MTE2: return PIPE::PIPE_MTE2;
    case ss::Pipe::MTE3: return PIPE::PIPE_MTE3;
    case ss::Pipe::FIX: return PIPE::PIPE_FIX;
    }
    llvm_unreachable("invalid structured pipe");
}
std::optional<int64_t> literal(Value value) {
    IntegerAttr a;
    if (!value || !matchPattern(value,m_Constant(&a)) || !a.getValue().isSignedIntN(64)) return {};
    if (value.getType().isInteger(1)) return int64_t(a.getValue().getZExtValue());
    return a.getValue().getSExtValue();
}

// Exact syntax-directed periodic scalar fragment. Only a remainder of the
// original nonnegative IV (or equivalent low-bit mask) represents that IV.
// A bare IV, narrowing cast, mutable parameter or arbitrary expression is not
// accidentally evaluated using its residue as if that were its full value.
class PeriodicScalar {
    Value iv;
    llvm::DenseMap<Value,std::optional<uint64_t>> periods;
public:
    std::set<uint64_t> cuts{0};
    explicit PeriodicScalar(Value iv) : iv(iv) {}
    std::optional<uint64_t> period(Value value) {
        auto found=periods.find(value);
        if (found!=periods.end()) return found->second;
        auto result=computePeriod(value);
        periods[value]=result; return result;
    }
    std::optional<uint64_t> computePeriod(Value value) {
        if (auto c=literal(value)) {
            if (*c>=0) { cuts.insert(uint64_t(*c)); if (*c<INT64_MAX) cuts.insert(uint64_t(*c)+1); }
            return 1;
        }
        Operation *op=value.getDefiningOp();
        if (!op) return {};
        if (isa<arith::RemSIOp,arith::RemUIOp>(op) && op->getOperand(0)==iv) {
            auto d=literal(op->getOperand(1));
            return d && *d>0 ? std::optional<uint64_t>(uint64_t(*d)) : std::nullopt;
        }
        if (auto mask=dyn_cast<arith::AndIOp>(op)) {
            Value other=mask.getLhs()==iv?mask.getRhs():(mask.getRhs()==iv?mask.getLhs():Value());
            if (other) {
                auto c=literal(other);
                if (!c || *c<0 || *c==INT64_MAX) return {};
                uint64_t d=uint64_t(*c)+1;
                if ((d&(d-1))==0) return d;
                return {};
            }
        }
        if (!isa<arith::CmpIOp>(op) &&
            !(value.getType().isInteger(1) && isa<arith::AndIOp,arith::OrIOp,arith::XOrIOp>(op))) return {};
        uint64_t p=1;
        for (Value operand:op->getOperands()) {
            auto q=period(operand);
            if (!q || (p!=1 && *q!=1 && p!=*q)) return {};
            p=std::max(p,*q);
        }
        return p;
    }
    std::optional<int64_t> evaluate(Value value,uint64_t residue) const {
        llvm::DenseMap<Value,std::optional<int64_t>> cache;
        return eval(value,residue,cache);
    }
private:
    std::optional<int64_t> eval(Value value,uint64_t residue,
        llvm::DenseMap<Value,std::optional<int64_t>> &cache) const {
        auto found=cache.find(value);
        if (found!=cache.end()) return found->second;
        auto compute=[&]() -> std::optional<int64_t> {
            if (auto c=literal(value)) return c;
            auto *op=value.getDefiningOp(); if (!op) return {};
            if (isa<arith::RemSIOp,arith::RemUIOp>(op) && op->getOperand(0)==iv) {
                auto d=literal(op->getOperand(1));
                if (!d || *d<=0) return {};
                return int64_t(residue%uint64_t(*d));
            }
            if (auto mask=dyn_cast<arith::AndIOp>(op)) {
                Value other=mask.getLhs()==iv?mask.getRhs():(mask.getRhs()==iv?mask.getLhs():Value());
                if (other) {
                    auto c=literal(other);
                    if (!c || *c<0 || *c==INT64_MAX || ((uint64_t(*c)+1)&uint64_t(*c))) return {};
                    return int64_t(residue&uint64_t(*c));
                }
            }
            if (op->getNumOperands()!=2) return {};
            auto a=eval(op->getOperand(0),residue,cache), b=eval(op->getOperand(1),residue,cache);
            if (!a || !b) return {};
            if (auto cmp=dyn_cast<arith::CmpIOp>(op)) {
                switch(cmp.getPredicate()) {
                case arith::CmpIPredicate::eq: return *a==*b;
                case arith::CmpIPredicate::ne: return *a!=*b;
                case arith::CmpIPredicate::slt: return *a<*b;
                case arith::CmpIPredicate::sle: return *a<=*b;
                case arith::CmpIPredicate::sgt: return *a>*b;
                case arith::CmpIPredicate::sge: return *a>=*b;
                case arith::CmpIPredicate::ult: return uint64_t(*a)<uint64_t(*b);
                case arith::CmpIPredicate::ule: return uint64_t(*a)<=uint64_t(*b);
                case arith::CmpIPredicate::ugt: return uint64_t(*a)>uint64_t(*b);
                case arith::CmpIPredicate::uge: return uint64_t(*a)>=uint64_t(*b);
                }
            }
            if (!value.getType().isInteger(1)) return {};
            if (isa<arith::AndIOp>(op)) return bool(*a)&&bool(*b);
            if (isa<arith::OrIOp>(op)) return bool(*a)||bool(*b);
            if (isa<arith::XOrIOp>(op)) return bool(*a)!=bool(*b);
            return {};
        };
        auto answer=compute(); cache[value]=answer; return answer;
    }
};

// One shared physical import for every mutually exclusive region arm.
// Choice composition follows ORIGINAL region nesting, never a Cartesian
// product of independently enumerated Boolean assignments.
bool belongsTo(Region &region,Operation *op) {
    for(Region *r=op->getParentRegion();r;r=r->getParentOp()->getParentRegion())
        if(r==&region)return true;
    return false;
}
struct NativeInventory {
    func::FuncOp function;
    MemoryDependentAnalyzer memory;
    SyncIRs ir;
    Buffer2MemInfoMap buffers;
    SyncPhysicalFacts physical;
    std::string reason;
    explicit NativeInventory(func::FuncOp f):function(f) {}
    bool build() {
        if(!supportsLogicalSyncTranslation(function)){reason="unsupported loop forwarding or while transfer";return false;}
        if(qualifySyncPhysicalAddresses(function).status==SyncAddressAdmission::Rejected) {
            reason="unqualified physical address";return false;
        }
        PTOIRTranslator translator(ir,memory,buffers,function,SyncAnalysisMode::NORMALSYNC);
        if(failed(translator.Build())){reason="physical translation failed";return false;}
        physical=importStructuredSyncPhysicalFacts(function,ir);
        if(physical.status!=SyncPhysicalFacts::Status::Complete){reason=physical.reason;return false;}
        if(physical.lifetimeScope!=function.getOperation()) {
            reason="physical-section retirement transfer is not yet supported";return false;
        }
        return true;
    }
};
struct InvocationScope {
    Region *region=nullptr;
    SmallVector<scf::ForOp> wrappers;
};
class RegionLayout {
    NativeInventory &inventory;
    llvm::SmallPtrSet<Operation*,32> containsPayload;
    DominanceInfo dominance;
    bool fail(StringRef s){reason=s.str();return false;}
    bool normalized(scf::ForOp loop) {
        if(literal(loop.getLowerBound())!=std::optional<int64_t>(0)||
           literal(loop.getStep())!=std::optional<int64_t>(1)||!isa<IndexType>(loop.getInductionVar().getType()))
            return fail("invocation wrappers require zero-based unit-step index loops");
        SyncAddressEvaluator integers(inventory.function);auto value=integers.evaluate(loop.getLowerBound());
        if(!value||value->getBitWidth()!=64)return fail("unqualified invocation index layout");
        return true;
    }
    SmallVector<Operation*> active(Region &r) {
        SmallVector<Operation*> result;
        if(llvm::hasSingleElement(r))for(Operation &op:r.front())if(containsPayload.contains(&op))result.push_back(&op);
        return result;
    }
    bool collect(Region &r,SmallVector<scf::ForOp> wrappers) {
        if(!llvm::hasSingleElement(r))return fail("multi-block invocation region");
        auto roots=active(r);
        if(roots.empty())return true;
        if(roots.size()==1) {
            if(auto branch=dyn_cast<scf::IfOp>(roots.front())) {
                // Whole-region choice: one arm is selected for the entire
                // enclosing invocation nest. Values varying with an outer IV
                // or a loop-carried scalar are NOT treated as stable choices.
                if(!wrappers.empty()&&!dominance.properlyDominates(branch.getCondition(),wrappers.front().getOperation()))
                    return fail("region choice varies across enclosing invocations");
                if(!collect(branch.getThenRegion(),wrappers))return false;
                if(!branch.getElseRegion().empty()&&!collect(branch.getElseRegion(),wrappers))return false;
                return true;
            }
            if(auto outer=dyn_cast<scf::ForOp>(roots.front())) {
                unsigned loops=0;
                r.walk([&](scf::ForOp x){if(containsPayload.contains(x.getOperation()))++loops;});
                auto bodyRoots=active(*outer.getBody()->getParent());
                bool uniformWholeChoice=bodyRoots.size()==1 && isa<scf::IfOp>(bodyRoots.front()) &&
                    dominance.properlyDominates(cast<scf::IfOp>(bodyRoots.front()).getCondition(),outer.getOperation());
                if(loops>1||uniformWholeChoice) {
                    if(!normalized(outer))return false;
                    if(!wrappers.empty()&&!dominance.properlyDominates(outer.getUpperBound(),wrappers.front().getOperation()))
                        return fail("nonrectangular invocation bound is not invariant at the outer entry");
                    wrappers.push_back(outer);return collect(*outer.getBody()->getParent(),std::move(wrappers));
                }
            }
        }
        unsigned loops=0;r.walk([&](scf::ForOp x){if(containsPayload.contains(x.getOperation()))++loops;});
        if(loops>1)return fail("mixed sequential/nested child transfers are outside S3's re-entrant region fragment");
        scopes.push_back({&r,std::move(wrappers)});return true;
    }
public:
    std::vector<InvocationScope> scopes;
    std::string reason;
    explicit RegionLayout(NativeInventory &i):inventory(i),dominance(i.function) {
        for(auto *phase:i.physical.phases)for(auto *op=phase->elementOp;op&&op!=i.function.getOperation();op=op->getParentOp())
            containsPayload.insert(op);
    }
    bool build(){return collect(inventory.function.getBody(),{});}
};

struct NativeFacts {
    func::FuncOp function;
    InsertSyncGMAliasMode gm;
    NativeInventory &inventory;
    Region *scope;
    SmallVector<scf::ForOp> wrappers;
    Buffer2MemInfoMap &buffers;
    SyncPhysicalFacts physical;
    scf::ForOp loop;
    ss::Model model;
    std::vector<ss::InvocationRequirement> carried;
    std::vector<unsigned> origin;
    std::vector<ss::Segment> segments;
    llvm::DenseMap<Operation*,unsigned> phaseId;
    std::vector<std::vector<std::pair<Value,bool>>> guards;
    std::vector<std::vector<uint64_t>> residues;
    std::map<const BaseMemInfo*,std::optional<SyncPhysicalSlotMapping>> mappings;
    std::string reason;
    NativeFacts(NativeInventory &i,const InvocationScope &unit,InsertSyncGMAliasMode contract)
        :function(i.function),gm(contract),inventory(i),scope(unit.region),wrappers(unit.wrappers),buffers(i.buffers) {}
    ss::InvocationModel invocationModel() const {return {model,carried};}
    bool fail(StringRef message) { reason=message.str(); return false; }

    bool build() {
        physical=inventory.physical;
        physical.phases.erase(std::remove_if(physical.phases.begin(),physical.phases.end(),
            [&](const auto *p){return !belongsTo(*scope,p->elementOp);}),physical.phases.end());
        bool multiple=false;
        scope->walk([&](scf::ForOp candidate) { if (loop) multiple=true; else loop=candidate; });
        if (multiple) return fail("local region still contains mixed child-loop transfers");
        if (physical.phases.size()>std::numeric_limits<unsigned>::max())
            return fail("physical phase identity width overflow");
        model.recurring=bool(loop);
        if (loop) {
            if (literal(loop.getLowerBound())!=std::optional<int64_t>(0) ||
                literal(loop.getStep())!=std::optional<int64_t>(1) ||
                !isa<IndexType>(loop.getInductionVar().getType()))
                return fail("structured loop requires normalized zero-based unit-step index induction");
            SyncAddressEvaluator integers(function);
            auto lower=integers.evaluate(loop.getLowerBound());
            if (!lower || lower->getBitWidth()!=64)
                return fail("structured loop requires the qualified signed-64-bit PTO index layout");
        }
        if(!wrappers.empty()&&loop) {
            DominanceInfo dominance(function);
            if(!dominance.properlyDominates(loop.getUpperBound(),wrappers.front().getOperation()))
                return fail("inner bound varies across enclosing invocations");
        }
        PeriodicScalar scalar(loop?loop.getInductionVar():Value());
        guards.resize(physical.phases.size()); residues.resize(physical.phases.size());
        segments.assign(physical.phases.size(),ss::Segment::Body);
        llvm::DenseMap<Operation*,bool> beforeLoop;
        bool passedLoop=false;
        for (Operation &op:scope->front()) {
            if (&op==loop.getOperation()) passedLoop=true;
            beforeLoop[&op]=!passedLoop;
        }
        uint64_t period=1;
        auto mergePeriod=[&](uint64_t p) {
            if (p!=1 && period!=1 && p!=period) return false;
            period=std::max(period,p); return true;
        };
        for (unsigned p=0;p<physical.phases.size();++p) {
            auto *op=physical.phases[p]->elementOp; phaseId[op]=p;
            if (loop && !loop->isAncestor(op)) {
                // A single invocation with unconditional prefix/suffix effects.
                // Do not flatten a conditional or nested loop into this model.
                if (loop->getBlock()!=&scope->front() || op->getBlock()!=&scope->front())
                    return fail("boundary payload requires a direct, unconditional loop invocation");
                segments[p]=beforeLoop.lookup(op)?ss::Segment::Prelude:ss::Segment::Epilogue;
            }
            for (Operation *parent=op->getParentOp(); parent && parent!=scope->getParentOp();
                 parent=parent->getParentOp()) {
                if (parent==loop.getOperation()) continue;
                auto branch=dyn_cast<scf::IfOp>(parent);
                if (!branch) return fail("unrepresented physical control region");
                Region *r=op->getParentRegion();
                while (r && r->getParentOp()!=parent) r=r->getParentOp()->getParentRegion();
                if (!r) return fail("invalid control ancestry");
                guards[p].push_back({branch.getCondition(),r==&branch.getThenRegion()});
                auto d=scalar.period(branch.getCondition());
                if (!d || !mergePeriod(*d)) return fail("payload guard is outside the common-period scalar fragment");
            }
            auto visit=[&](const auto &accesses) {
                for (auto *a:accesses) {
                    auto mapping=qualifySyncPhysicalSlots(a,buffers);
                    if (mapping && mapping->selector) {
                        auto d=scalar.period(mapping->selector);
                        // Unknown selectors retain their entire physical may
                        // footprint; they do not acquire a fixed slot identity.
                        if (d && !mergePeriod(*d)) return false;
                    }
                    mappings.emplace(a,std::move(mapping));
                }
                return true;
            };
            if (!visit(physical.phases[p]->useVec) || !visit(physical.phases[p]->defVec))
                return fail("incompatible slot periods require a more general transfer");
        }
        model.period=period;
        scalar.cuts.insert(period);
        std::vector<uint64_t> breaks;
        for (uint64_t c:scalar.cuts) if (c<=period) breaks.push_back(c);
        for (unsigned p=0;p<physical.phases.size();++p) {
            bool explicitSlots=false;
            auto inspect=[&](const auto &v) {
                for (auto *a:v) {
                    const auto &map=mappings.at(a);
                    if (map && map->selector && map->bases.size()==period && scalar.period(map->selector))
                        explicitSlots=true;
                }
            };
            inspect(physical.phases[p]->useVec); inspect(physical.phases[p]->defVec);
            if (segments[p]!=ss::Segment::Body) residues[p].push_back(0);
            for (std::size_t i=1;segments[p]==ss::Segment::Body && i<breaks.size();++i) {
                uint64_t lo=breaks[i-1],hi=breaks[i]; bool active=true;
                for (auto [condition,take]:guards[p]) {
                    auto v=scalar.evaluate(condition,lo);
                    if (!v) return fail("periodic guard evaluation unavailable");
                    active &= bool(*v)==take;
                }
                if (!active || lo==hi) continue;
                // Do not enumerate a numeric modulus. A multi-residue payload
                // is expanded only when those slots already exist explicitly
                // in its qualified physical table. Otherwise decline this
                // representation, not run a solver under a larger allowance.
                if (hi-lo>1 && !explicitSlots)
                    return fail("multi-residue payload needs an interval-phase transfer");
                for (uint64_t r=lo;r<hi;++r) residues[p].push_back(r);
            }
            auto lane=pipe(static_cast<PIPE>(physical.phases[p]->kPipeValue));
            if (!lane) return fail("unqualified physical pipe");
            for (uint64_t r:residues[p]) {
                origin.push_back(p);
                model.atoms.push_back({r,p,{physical.cube?ss::Core::AIC:ss::Core::AIV,*lane},segments[p]});
            }
        }
        auto alias=[&](const BaseMemInfo *a,uint64_t ra,const BaseMemInfo *b,uint64_t rb) {
            if (!logicalSyncMayAlias(a,b,function,gm)) return false;
            const auto &ma=mappings.at(a), &mb=mappings.at(b);
            if (!ma || !mb || ma->scope!=mb->scope) return true;
            auto selected=[&](const SyncPhysicalSlotMapping &m,uint64_t r) -> std::optional<uint64_t> {
                if (!m.selector) return m.bases.size()==1?std::optional<uint64_t>(m.bases.front()):std::nullopt;
                auto d=scalar.period(m.selector);
                if (!d || model.period%*d) return {};
                auto s=scalar.evaluate(m.selector,r);
                if (!s || *s<0 || uint64_t(*s)>=m.bases.size()) return {};
                return m.bases[uint64_t(*s)];
            };
            auto x=selected(*ma,ra),y=selected(*mb,rb);
            if (!x || !y) return true;
            return *x<*y+mb->bytes && *y<*x+ma->bytes;
        };
        for (std::size_t p=0;p<origin.size();++p) for (std::size_t q=0;q<origin.size();++q) {
            auto distance=ss::priorDistance(model,p,q); if (!distance && wrappers.empty()) continue;
            auto *a=physical.phases[origin[p]], *b=physical.phases[origin[q]];
            bool conflict=false,acc=false,visibility=false;
            auto pair=[&](const auto &left,bool writeA,const auto &right,bool writeB) {
                for (auto *x:left) for (auto *y:right) {
                    bool resource=!writeA&&!writeB && x->scope==AddressSpace::ACC &&
                        y->scope==AddressSpace::ACC && model.atoms[p].lane!=model.atoms[q].lane;
                    if (!writeA&&!writeB&&!resource) continue;
                    if (!alias(x,model.atoms[p].residue,y,model.atoms[q].residue)) continue;
                    conflict=true; acc|=resource;
                    // Do not conflate a legal direction with same-address GM
                    // publication or scalar-cache coherence. This fragment has
                    // no qualified visibility realization.
                    if (x->scope==AddressSpace::GM && y->scope==AddressSpace::GM && writeA&&!writeB &&
                        ((model.atoms[p].lane.pipe==ss::Pipe::MTE3 && model.atoms[q].lane.pipe==ss::Pipe::MTE2) ||
                         model.atoms[p].lane.pipe==ss::Pipe::S || model.atoms[q].lane.pipe==ss::Pipe::S)) visibility=true;
                }
            };
            pair(a->defVec,true,b->useVec,false); pair(a->useVec,false,b->defVec,true);
            pair(a->defVec,true,b->defVec,true); pair(a->useVec,false,b->useVec,false);
            if (conflict) {
                auto property=visibility?ss::Property::Visibility:acc?ss::Property::AccResource:ss::Property::Completion;
                if(distance)model.requirements.push_back({p,q,*distance,property});
                if(!wrappers.empty())carried.push_back({p,q,property});
            }
        }
        return true;
    }
};

Value indexConstant(OpBuilder &builder,Location loc,uint64_t value) {
    return builder.create<arith::ConstantIndexOp>(loc,int64_t(value));
}
bool legalBoundaryOperands(NativeFacts &facts,const std::vector<ss::Action> &actions,std::string &why) {
    DominanceInfo dominance(facts.function);
    for (const auto &a:actions) {
        if (a.distanceInIterations>uint64_t(INT64_MAX) || a.guardResidue>uint64_t(INT64_MAX) ||
            facts.model.period>uint64_t(INT64_MAX)) {
            why="unrepresentable synchronization guard constant";return false;
        }
        if(a.invocation!=ss::Action::Local&&facts.wrappers.empty()) {
            why="invocation endpoint has no enclosing frame";return false;
        }
        if(a.invocationGuardResidue>uint64_t(INT64_MAX)) {
            why="unrepresentable invocation existence condition";return false;
        }
        if (a.participation!=ss::Action::IfBody && !a.invocationBodyGuard) continue;
        auto *anchor=facts.physical.phases[facts.origin[a.anchor]]->elementOp;
        if (!facts.loop || !dominance.properlyDominates(facts.loop.getUpperBound(),anchor)) {
            why="loop bound is unavailable at the selected early publication";return false;
        }
    }
    return true;
}
void emitNative(NativeFacts &facts,const std::vector<ss::Action> &actions) {
    std::map<std::pair<Operation*,bool>,std::vector<ss::Action>> points;
    for (auto a:actions) points[{facts.physical.phases[facts.origin[a.anchor]]->elementOp,a.after}].push_back(a);
    // Iterate original phases, never pointer ordering, for deterministic IR.
    for (unsigned p=0;p<facts.physical.phases.size();++p) for (bool after:{false,true}) {
        auto *anchor=facts.physical.phases[p]->elementOp;
        auto found=points.find({anchor,after}); if (found==points.end()) continue;
        OpBuilder outer(anchor);
        if (after) outer.setInsertionPointAfter(anchor);
        for (const auto &a:found->second) {
            OpBuilder::InsertionGuard restore(outer);
            OpBuilder builder=outer;
            auto guard=[&](Value c) {
                auto branch=builder.create<scf::IfOp>(anchor->getLoc(),c,false);
                builder.setInsertionPointToStart(&branch.getThenRegion().front());
            };
            if(a.invocation!=ss::Action::Local) {
                Value predicate;
                for(auto frame:facts.wrappers) {
                    Value part;
                    if(a.invocation==ss::Action::ToNextInvocation) {
                        auto remaining=builder.create<arith::SubIOp>(anchor->getLoc(),frame.getUpperBound(),frame.getInductionVar());
                        auto one=indexConstant(builder,anchor->getLoc(),1);
                        part=builder.create<arith::CmpIOp>(anchor->getLoc(),arith::CmpIPredicate::sgt,remaining,one);
                    } else {
                        auto zero=indexConstant(builder,anchor->getLoc(),0);
                        part=builder.create<arith::CmpIOp>(anchor->getLoc(),arith::CmpIPredicate::ne,frame.getInductionVar(),zero);
                    }
                    if(predicate)predicate=builder.create<arith::OrIOp>(anchor->getLoc(),predicate,part);
                    else predicate=part;
                }
                guard(predicate);
            }
            if(a.invocationBodyGuard) {
                auto r=indexConstant(builder,anchor->getLoc(),a.invocationGuardResidue);
                guard(builder.create<arith::CmpIOp>(anchor->getLoc(),arith::CmpIPredicate::sgt,
                    facts.loop.getUpperBound(),r));
            }
            if (facts.residues[p].size()>1) {
                auto d=indexConstant(builder,anchor->getLoc(),facts.model.period);
                auto r=builder.create<arith::RemUIOp>(anchor->getLoc(),facts.loop.getInductionVar(),d);
                auto c=indexConstant(builder,anchor->getLoc(),facts.model.atoms[a.anchor].residue);
                guard(builder.create<arith::CmpIOp>(anchor->getLoc(),arith::CmpIPredicate::eq,r,c));
            }
            if (a.participation==ss::Action::IfBody) {
                auto r=indexConstant(builder,anchor->getLoc(),a.guardResidue);
                guard(builder.create<arith::CmpIOp>(anchor->getLoc(),arith::CmpIPredicate::sgt,
                    facts.loop.getUpperBound(),r));
            } else if (a.participation==ss::Action::First) {
                auto r=indexConstant(builder,anchor->getLoc(),facts.model.atoms[a.anchor].residue);
                guard(builder.create<arith::CmpIOp>(anchor->getLoc(),arith::CmpIPredicate::eq,
                    facts.loop.getInductionVar(),r));
            } else if (a.participation==ss::Action::Last) {
                // In an executing body: 0 <= iv < upper <= INDEX_MAX.
                // This subtraction is representable even near INDEX_MAX.
                auto left=builder.create<arith::SubIOp>(anchor->getLoc(),facts.loop.getUpperBound(),facts.loop.getInductionVar());
                auto period=indexConstant(builder,anchor->getLoc(),facts.model.period);
                guard(builder.create<arith::CmpIOp>(anchor->getLoc(),arith::CmpIPredicate::sle,left,period));
            }
            if (a.distanceInIterations) {
                auto c=indexConstant(builder,anchor->getLoc(),a.distanceInIterations);
                if (a.kind==ss::Action::Set) {
                    // Inside the original loop: 0 <= iv < upper <= INDEX_MAX.
                    // upper-iv is representable. iv+distance need not be.
                    auto left=builder.create<arith::SubIOp>(anchor->getLoc(),facts.loop.getUpperBound(),facts.loop.getInductionVar());
                    guard(builder.create<arith::CmpIOp>(anchor->getLoc(),arith::CmpIPredicate::sgt,left,c));
                } else guard(builder.create<arith::CmpIOp>(anchor->getLoc(),arith::CmpIPredicate::sge,
                    facts.loop.getInductionVar(),c));
            }
            auto source=PipeAttr::get(facts.function.getContext(),pipe(a.source.pipe));
            auto target=PipeAttr::get(facts.function.getContext(),pipe(a.target.pipe));
            auto key=EventAttr::get(facts.function.getContext(),static_cast<EVENT>(a.key));
            if (a.kind==ss::Action::Set) builder.create<SetFlagOp>(anchor->getLoc(),source,target,key);
            else if (a.kind==ss::Action::Wait) builder.create<WaitFlagOp>(anchor->getLoc(),source,target,key);
            else builder.create<BarrierOp>(anchor->getLoc(),source);
            // Keep the next action after the complete newly inserted root,
            // not inside its guard and not ahead of an earlier same-point set.
            // outer's iterator was the original successor and remains valid.
        }
    }

}

class Reconstruct {
    NativeFacts &facts;
    const llvm::SmallPtrSetImpl<Operation*> &original;
    llvm::SmallPtrSet<Operation*,32> allowed;
    std::vector<ss::Action> actions;
    uint64_t ordinal=0;
    unsigned retirement=0;
    std::string why;
    enum GuardKind { Residue, Previous, Next, First, Last, IfBody, InvocationPrevious, InvocationNext };
    struct Condition { GuardKind kind; Value value; uint64_t distance=0; };

    bool fail(StringRef s) { why=s.str(); return false; }
    void allowExpression(Value v) {
        auto *op=v.getDefiningOp();
        if (!op || original.contains(op) || !allowed.insert(op).second) return;
        for (Value x:op->getOperands()) allowExpression(x);
    }
    bool invocationCondition(Value v,bool next) {
        if(facts.wrappers.empty())return false;
        SmallVector<Value> pending{v};llvm::SmallPtrSet<Operation*,8> frames;
        while(!pending.empty()) {
            Value x=pending.pop_back_val();
            if(auto either=x.getDefiningOp<arith::OrIOp>()) {
                pending.push_back(either.getLhs());pending.push_back(either.getRhs());continue;
            }
            auto cmp=x.getDefiningOp<arith::CmpIOp>();if(!cmp)return false;
            bool matched=false;
            for(auto loop:facts.wrappers) {
                bool yes=false;
                if(next) {
                    auto sub=cmp.getLhs().getDefiningOp<arith::SubIOp>();
                    yes=cmp.getPredicate()==arith::CmpIPredicate::sgt && literal(cmp.getRhs())==std::optional<int64_t>(1) &&
                        sub && sub.getLhs()==loop.getUpperBound() && sub.getRhs()==loop.getInductionVar();
                } else yes=cmp.getPredicate()==arith::CmpIPredicate::ne &&
                    literal(cmp.getRhs())==std::optional<int64_t>(0) && cmp.getLhs()==loop.getInductionVar();
                if(yes) {
                    if(!frames.insert(loop.getOperation()).second)return false;
                    matched=true;break;
                }
            }
            if(!matched)return false;
        }
        return frames.size()==facts.wrappers.size();
    }
    std::optional<Condition> condition(Value v) {
        if(invocationCondition(v,true)){allowExpression(v);return Condition{InvocationNext,v,0};}
        if(invocationCondition(v,false)){allowExpression(v);return Condition{InvocationPrevious,v,0};}
        if (!facts.loop) {
            if (literal(v)) { allowExpression(v); return Condition{Residue,v,0}; }
            return {};
        }
        if (auto cmp=v.getDefiningOp<arith::CmpIOp>()) {
            auto c=literal(cmp.getRhs());
            if (c && *c>=0) {
                if (cmp.getPredicate()==arith::CmpIPredicate::eq && cmp.getLhs()==facts.loop.getInductionVar()) {
                    allowExpression(v); return Condition{First,v,uint64_t(*c)};
                }
                if (cmp.getPredicate()==arith::CmpIPredicate::sgt && cmp.getLhs()==facts.loop.getUpperBound()) {
                    allowExpression(v); return Condition{IfBody,v,uint64_t(*c)};
                }
                if (auto sub=cmp.getLhs().getDefiningOp<arith::SubIOp>()) {
                    if (cmp.getPredicate()==arith::CmpIPredicate::sle &&
                        sub.getLhs()==facts.loop.getUpperBound() && sub.getRhs()==facts.loop.getInductionVar() &&
                        uint64_t(*c)==facts.model.period) {
                        allowExpression(v); return Condition{Last,v,uint64_t(*c)};
                    }
                }
            }
            if (c && *c>0) {
                if (cmp.getPredicate()==arith::CmpIPredicate::sge && cmp.getLhs()==facts.loop.getInductionVar()) {
                    allowExpression(v); return Condition{Previous,v,uint64_t(*c)};
                }
                if (auto sub=cmp.getLhs().getDefiningOp<arith::SubIOp>()) {
                    if (cmp.getPredicate()==arith::CmpIPredicate::sgt && sub.getLhs()==facts.loop.getUpperBound() &&
                        sub.getRhs()==facts.loop.getInductionVar()) {
                        allowExpression(v); return Condition{Next,v,uint64_t(*c)};
                    }
                }
            }
        }
        PeriodicScalar scalar(facts.loop.getInductionVar());
        auto p=scalar.period(v);
        if (!p || facts.model.period%*p) return {};
        allowExpression(v); return Condition{Residue,v,0};
    }
    bool generated(Operation *op,Operation *previous,Operation *next,std::vector<Condition> guards) {
        if (auto branch=dyn_cast<scf::IfOp>(op)) {
            if (branch.getNumResults() || !llvm::hasSingleElement(branch.getThenRegion()))
                return fail("generated guard has an unsupported region/result");
            if (!branch.getElseRegion().empty()) for (Operation &x:branch.getElseRegion().front())
                if (!isa<scf::YieldOp>(x) || x.getNumOperands()) return fail("generated else path has actions");
            auto c=condition(branch.getCondition());
            if (!c) return fail("generated predicate is not a qualified original-bound/residue expression");
            guards.push_back(*c); allowed.insert(op);
            for (Region &region:op->getRegions()) if (!region.empty()) {
                if (!llvm::hasSingleElement(region)) return fail("generated multiblock guard");
                for (Operation &x:region.front()) {
                    if (isa<scf::YieldOp>(x)) {
                        if (x.getNumOperands()) return fail("generated yield has values");
                        allowed.insert(&x); continue;
                    }
                    if (&region!=&branch.getThenRegion()) continue;
                    if (!generated(&x,previous,next,guards)) return false;
                }
            }
            return true;
        }
        if (isa<arith::ConstantOp,arith::CmpIOp,arith::SubIOp,arith::RemUIOp,arith::OrIOp>(op)) return true;
        ss::Action a;
        if (auto set=dyn_cast<SetFlagOp>(op)) {
            a.kind=ss::Action::Set; a.after=true; a.key=unsigned(set.getEventId().getEvent());
            auto s=pipe(set.getSrcPipe().getPipe()),t=pipe(set.getDstPipe().getPipe());
            if (!s || !t) return fail("unsupported reconstructed event direction");
            a.source={facts.physical.cube?ss::Core::AIC:ss::Core::AIV,*s};
            a.target={a.source.core,*t};
        } else if (auto wait=dyn_cast<WaitFlagOp>(op)) {
            a.kind=ss::Action::Wait; a.after=false; a.key=unsigned(wait.getEventId().getEvent());
            auto s=pipe(wait.getSrcPipe().getPipe()),t=pipe(wait.getDstPipe().getPipe());
            if (!s || !t) return fail("unsupported reconstructed event direction");
            a.source={facts.physical.cube?ss::Core::AIC:ss::Core::AIV,*s};
            a.target={a.source.core,*t};
        } else if (auto barrier=dyn_cast<BarrierOp>(op)) {
            if (barrier.getPipe().getPipe()==PIPE::PIPE_ALL) {
                auto *ret=facts.function.getBody().front().getTerminator();
                if (!guards.empty() || op->getBlock()!=ret->getBlock() || op->getNextNode()!=ret ||
                    op->hasAttr("pto.auto_sync_tail_barrier") || op->hasAttr("pto.auto_sync_tail_hint"))
                    return fail("unqualified retirement placement");
                ++retirement; allowed.insert(op); return true;
            }
            auto p=pipe(barrier.getPipe().getPipe()); if (!p) return fail("unqualified barrier pipe");
            a.kind=ss::Action::Barrier; a.after=false;
            a.source=a.target={facts.physical.cube?ss::Core::AIC:ss::Core::AIV,*p};
        } else return fail("unexpected operation added by structured emission");
        auto id=facts.phaseId.find(a.after?previous:next);
        if (id==facts.phaseId.end()) return fail("event/barrier lacks its actual original payload cut");
        std::optional<Condition> participation,invocation,existence;
        for(const auto &g:guards) {
            if(g.kind==Residue)continue;
            if(g.kind==InvocationNext||g.kind==InvocationPrevious) {
                if(invocation)return fail("duplicate invocation guard");
                invocation=g;
            } else if(g.kind==IfBody) {
                if(existence)return fail("duplicate body-existence guard");
                existence=g;
            } else {
                if(participation)return fail("repeated endpoint participation guards");
                participation=g;
            }
        }
        if(invocation) {
            a.invocation=invocation->kind==InvocationNext?ss::Action::ToNextInvocation:ss::Action::FromPreviousInvocation;
            if(existence){a.invocationBodyGuard=true;a.invocationGuardResidue=existence->distance;}
        } else if(existence) {
            if(participation)return fail("multiple local participation domains");
            participation=existence;
        }
        if (participation) {
            const auto &g=*participation;
            if (g.kind==Previous || g.kind==Next) {
                if (a.invocation!=ss::Action::Local || a.kind==ss::Action::Barrier ||
                    (a.kind==ss::Action::Set?g.kind!=Next:g.kind!=Previous))
                    return fail("wrong periodic participation direction");
                a.distanceInIterations=g.distance;
            } else if (g.kind==First) a.participation=ss::Action::First;
            else if (g.kind==Last) a.participation=ss::Action::Last;
            else if (g.kind==IfBody) {a.participation=ss::Action::IfBody;a.guardResidue=g.distance;}
        }
        a.order=ordinal++;
        PeriodicScalar scalar(facts.loop?facts.loop.getInductionVar():Value());
        for (std::size_t atom=0;atom<facts.origin.size();++atom) if (facts.origin[atom]==id->second) {
            const auto &represented=facts.model.atoms[atom];
            if (participation && (participation->kind==First || participation->kind==Last ||
                                 participation->kind==Previous || participation->kind==Next) &&
                represented.segment!=ss::Segment::Body)
                return fail("loop-local endpoint guard outside the body");
            bool active=true;
            for (const auto &g:guards) if (g.kind==Residue) {
                auto v=scalar.evaluate(g.value,represented.residue);
                if (!v) return fail("reconstructed residue predicate is unavailable");
                active &= bool(*v);
            }
            if (active) {
                if (participation && participation->kind==First && participation->distance!=represented.residue)
                    return fail("first-acquisition predicate selects the wrong original occurrence");
                a.anchor=atom; actions.push_back(a);
            }
        }
        allowed.insert(op); return true;
    }
    bool block(Block &b) {
        SmallVector<Operation*> originals;
        for (Operation &op:b) if (original.contains(&op)) originals.push_back(&op);
        std::size_t at=0; Operation *previous=nullptr;
        for (Operation &op:b) {
            if (original.contains(&op)) {
                previous=&op; ++at;
                for (Region &r:op.getRegions()) for (Block &child:r) if (!block(child)) return false;
            } else if (!generated(&op,previous,at<originals.size()?originals[at]:nullptr,{})) return false;
        }
        return true;
    }
public:
    Reconstruct(NativeFacts &f,const llvm::SmallPtrSetImpl<Operation*> &o):facts(f),original(o) {}
    Outcome run(const std::vector<ss::Action> &selected,llvm::SmallPtrSetImpl<Operation*> &allAllowed) {
        Outcome out;out.status=Outcome::InternalError;
        const unsigned expectedRetirement=scopeIsFunction()?1:0;
        if(!block(facts.scope->front())||retirement!=expectedRetirement) {
            out.reason=why.empty()?"retirement missing or duplicated":why;return out;
        }
        if(facts.wrappers.empty()) {
            auto result=ss::verify(facts.model,actions);
            if(result.status!=ss::Status::Applied){out.reason=result.reason;return out;}
            out.handoffs=result.plan.handoffs.size();out.barriers=result.plan.barriers.size()+result.plan.firstBarriers.size();
            out.work=result.completionRelaxations+result.eventRelaxations;
        } else {
            auto result=ss::verifyInvocations(facts.invocationModel(),actions);
            if(result.status!=ss::Status::Applied){out.reason=result.reason;return out;}
            out.handoffs=result.plan.local.handoffs.size()+result.plan.handoffs.size();
            out.barriers=result.plan.local.barriers.size()+result.plan.local.firstBarriers.size()+result.plan.barriers.size();
            out.work=result.graphVisits;
        }
        // Actual action ordering, predicates and local/outer occurrence roles
        // must match the selected early boundaries, even if a broader plan
        // would also be safe. No insertion tags are consumed as proof.
        using Fingerprint=std::tuple<unsigned,bool,ss::Lane,ss::Lane,unsigned,uint64_t,unsigned,uint64_t,unsigned,bool,uint64_t>;
        auto fingerprints=[](const std::vector<ss::Action> &items) {
            std::map<std::pair<std::size_t,bool>,std::vector<Fingerprint>> at;
            auto sorted=items;std::stable_sort(sorted.begin(),sorted.end(),[](const auto &a,const auto &b){return a.order<b.order;});
            for(const auto &a:sorted)at[{a.anchor,a.after}].emplace_back(unsigned(a.kind),a.after,a.source,a.target,a.key,
                a.distanceInIterations,unsigned(a.participation),a.guardResidue,unsigned(a.invocation),
                a.invocationBodyGuard,a.invocationGuardResidue);
            return at;
        };
        if(fingerprints(actions)!=fingerprints(selected)) {
            out.reason="emission changed selected invocation/readiness/release boundaries";return out;
        }
        for(auto *op:allowed)allAllowed.insert(op);
        out.status=Outcome::Applied;out.reason="structured region and actual invocation actions verified";
        out.requirements=facts.model.requirements.size()+facts.carried.size();return out;
    }
    bool scopeIsFunction() const {return facts.scope==&facts.function.getBody();}

};

Outcome run(func::FuncOp function,InsertSyncGMAliasMode gm,llvm::function_ref<void(func::FuncOp)> mutate) {
    if (function.isDeclaration() || !llvm::hasSingleElement(function.getBody())) {
        Outcome out; out.reason="structured construction requires one function block"; return out;
    }
    const auto start=std::chrono::steady_clock::now();
    OwningOpRef<ModuleOp> stage(ModuleOp::create(function.getLoc()));
    SmallVector<ModuleOp> ancestors;
    for (Operation *op=function->getParentOp();op;op=op->getParentOp())
        if (auto module=dyn_cast<ModuleOp>(op)) ancestors.push_back(module);
    ModuleOp parent=*stage;
    for (auto module:llvm::reverse(ancestors)) {
        auto child=ModuleOp::create(module.getLoc()); child->setAttrs(module->getAttrs());
        parent.getBody()->push_back(child); parent=child;
    }
    IRMapping mapping;
    auto working=cast<func::FuncOp>(function->clone(mapping)); parent.getBody()->push_back(working);
    Outcome out;NativeInventory inventory(working);
    if(!inventory.build()){out.reason=inventory.reason;return out;}
    RegionLayout layout(inventory);
    if(!layout.build()){out.reason=layout.reason;return out;}
    std::vector<std::unique_ptr<NativeFacts>> units;
    std::vector<std::vector<ss::Action>> selected;
    uint64_t atoms=0,views=0;
    for(const auto &scope:layout.scopes) {
        auto facts=std::make_unique<NativeFacts>(inventory,scope,gm);
        if(!facts->build()){out.reason=facts->reason;return out;}
        std::vector<ss::Action> actions;
        ss::Status status;std::string reason;
        if(facts->wrappers.empty()) {
            auto result=ss::construct(facts->model);status=result.status;reason=result.reason;
            if(status==ss::Status::Applied)actions=ss::actionsForPlan(facts->model,result.plan);
        } else {
            auto result=ss::constructInvocations(facts->invocationModel());status=result.status;reason=result.reason;
            if(status==ss::Status::Applied)actions=ss::actionsForInvocations(facts->invocationModel(),result.plan);
            views+=result.proofViews;
        }
        if(status!=ss::Status::Applied) {
            out.status=status==ss::Status::AllocationFailure?Outcome::AllocationFailure:
                status==ss::Status::InvalidPlan?Outcome::InternalError:Outcome::Unsupported;
            out.reason=reason;return out;
        }
        if(!legalBoundaryOperands(*facts,actions,out.reason))return out;
        atoms+=facts->model.atoms.size();selected.push_back(std::move(actions));units.push_back(std::move(facts));
    }
    SyncPayloadSnapshot snapshot(working);llvm::SmallPtrSet<Operation*,32> original;
    working.walk([&](Operation *op){original.insert(op);});
    for(std::size_t i=0;i<units.size();++i)emitNative(*units[i],selected[i]);
    auto *ret=working.getBody().front().getTerminator();OpBuilder builder(ret);
    builder.create<BarrierOp>(ret->getLoc(),PipeAttr::get(working.getContext(),PIPE::PIPE_ALL));
    if(mutate)mutate(working);
    out.status=Outcome::InternalError;
    if(failed(mlir::verify(working))){out.reason="malformed structured emission";return out;}
    auto drain=dyn_cast_or_null<BarrierOp>(ret->getPrevNode());unsigned drains=0;
    working.walk([&](BarrierOp b){if(b.getPipe().getPipe()==PIPE::PIPE_ALL)++drains;});
    if(!drain||drain.getPipe().getPipe()!=PIPE::PIPE_ALL||drains!=1||
       drain->hasAttr("pto.auto_sync_tail_barrier")||drain->hasAttr("pto.auto_sync_tail_hint")) {
        out.reason="one explicit unconditional function-retirement drain is required";return out;
    }
    llvm::SmallPtrSet<Operation*,32> allowed;allowed.insert(drain.getOperation());
    for(std::size_t i=0;i<units.size();++i) {
        auto result=Reconstruct(*units[i],original).run(selected[i],allowed);
        if(result.status!=Outcome::Applied)return result;
        out.requirements+=result.requirements;out.handoffs+=result.handoffs;out.barriers+=result.barriers;out.work+=result.work;
    }
    if(!snapshot.preserved(working,[&](Operation *op){return allowed.contains(op);})) {
        out.reason="original payload/control/geometry changed or unconsumed generated scalar operation";return out;
    }
    // One fresh whole-function physical import after all mutually exclusive
    // arms were reconstructed. Original control is preserved by the snapshot.
    SyncIRs rebuilt;Buffer2MemInfoMap buffers;MemoryDependentAnalyzer memory;
    PTOIRTranslator translator(rebuilt,memory,buffers,working,SyncAnalysisMode::NORMALSYNC);
    if(failed(translator.Build())){out.reason="emitted physical translation failed";return out;}
    std::map<Operation*,const CompoundInstanceElement*> phases;
    for(const auto &element:rebuilt)if(auto *p=dyn_cast<CompoundInstanceElement>(element.get()))
        if(!phases.emplace(p->elementOp,p).second){out.reason="duplicate emitted physical phase";return out;}
    if(phases.size()!=inventory.physical.phases.size()){out.reason="emitted physical phase population changed";return out;}
    auto same=[](const auto &a,const auto &b) {
        if(a.size()!=b.size())return false;
        for(std::size_t i=0;i<a.size();++i)if(!(*a[i]==*b[i]))return false;
        return true;
    };
    for(auto *old:inventory.physical.phases) {
        auto it=phases.find(old->elementOp);
        if(it==phases.end()||it->second->kPipeValue!=old->kPipeValue||
           !same(old->useVec,it->second->useVec)||!same(old->defVec,it->second->defVec)) {
            out.reason="reconstructed physical access contract changed";return out;
        }
    }
    out.status=Outcome::Applied;out.reason="structured local and nested invocation transfers verified";
    function.getBody().takeBody(working.getBody());
    if(std::getenv("PTOAS_LOGICAL_TRACE"))llvm::errs()<<"structured OAHS units "<<units.size()<<" atoms "<<atoms
        <<" invocation_views "<<views<<" requirements "<<out.requirements<<" handoffs "<<out.handoffs
        <<" barriers "<<out.barriers<<" presburger_queries 0 seconds "
        <<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<"\n";
    return out;
}
} // namespace
Outcome mlir::pto::structured_sync::constructStructuredSync(func::FuncOp f,InsertSyncGMAliasMode gm) {
    return run(f,gm,{});
}
Outcome mlir::pto::structured_sync::testing::constructWithEmissionMutation(
    func::FuncOp f,InsertSyncGMAliasMode gm,llvm::function_ref<void(func::FuncOp)> mutate) {
    return run(f,gm,mutate);
}
