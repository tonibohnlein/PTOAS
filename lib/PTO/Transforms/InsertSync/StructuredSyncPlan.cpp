// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.


#include "PTO/Transforms/InsertSync/StructuredSyncPlan.h"
#include "PTO/Transforms/InsertSync/StructuredSyncCore.h"
#include "PTO/Transforms/InsertSync/StructuredSyncOrdinal.h"
#include "PTO/Transforms/InsertSync/SyncPhysicalFacts.h"
#include "PTO/Transforms/InsertSync/StructuredSyncCoverage.h"
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
#include "llvm/Support/JSON.h"
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

Value stripIndexCast(Value value) {
    while (value) {
        if (auto cast=value.getDefiningOp<arith::IndexCastOp>()) { value=cast.getIn(); continue; }
        if (auto cast=value.getDefiningOp<arith::IndexCastUIOp>()) { value=cast.getIn(); continue; }
        break;
    }
    return value;
}
bool producedBy(Value value,StringRef operation) {
    value=stripIndexCast(value);
    Operation *op=value?value.getDefiningOp():nullptr;
    return op && op->getName().getStringRef()==operation;
}
bool blockDistributedWrapper(scf::ForOp loop) {
    return producedBy(loop.getLowerBound(),"pto.get_block_idx") &&
           producedBy(loop.getStep(),"pto.get_block_num");
}

// S7's lowering witness is deliberately stricter than may-access extraction.
// An allocation's maximum shape alone does not establish mad's effective m/n/k.
// Only direct immutable descriptors with constant FULL valid dimensions enter
// this first rule. Unqualified geometry retains ordinary completion demands.
std::optional<std::pair<uint64_t,uint64_t>> exactMatrixValid(Value value) {
    auto type=dyn_cast<TileBufType>(value.getType());
    auto alloc=value.getDefiningOp<AllocTileOp>();
    if (!type || !alloc || type.getRank()!=2 || type.getCompactModeI32()!=0)
        return {};
    for (Operation *user:value.getUsers())
        if (isa<SetValidShapeOp>(user)) return {};
    auto shape=type.getShape(), valid=type.getValidShape();
    if (shape.size()!=2 || valid.size()!=2 || shape[0]<=0 || shape[1]<=0)
        return {};
    std::optional<int64_t> rows=valid[0]>=0?std::optional<int64_t>(valid[0]):literal(alloc.getValidRow());
    std::optional<int64_t> cols=valid[1]>=0?std::optional<int64_t>(valid[1]):literal(alloc.getValidCol());
    if (!rows || !cols || *rows!=shape[0] || *cols!=shape[1]) return {};
    // Explicit operands, when present, must agree as well; do not trust a
    // static type over a conflicting runtime descriptor argument.
    if ((alloc.getValidRow() && literal(alloc.getValidRow())!=rows) ||
        (alloc.getValidCol() && literal(alloc.getValidCol())!=cols)) return {};
    return std::make_pair(uint64_t(*rows),uint64_t(*cols));
}
std::optional<std::pair<uint64_t,uint64_t>> exactAccFootprint(
    Value value,const CompoundInstanceElement &phase,bool write) {
    const auto &entries=write?phase.defVec:phase.useVec;
    const BaseMemInfo *match=nullptr;
    for (auto *entry:entries) if (entry && entry->baseBuffer==value) {
        if (match) return {};
        match=entry;
    }
    if (!match || match->scope!=AddressSpace::ACC || !match->hasKnownPhysicalAddresses ||
        match->aliasesUnknownRange || match->baseAddresses.size()!=1 || !match->allocateSize || match->allocateSize>uint64_t(INT64_MAX) ||
        match->baseAddresses[0]>uint64_t(INT64_MAX)-match->allocateSize) return {};
    return std::make_pair(match->baseAddresses[0],match->allocateSize);
}
ss::MmadInfo matrixLoweringFacts(const CompoundInstanceElement &phase) {
    ss::MmadInfo result;
    Operation *op=phase.elementOp;
    Value a,b,dst,input;
    if (auto init=dyn_cast<TMatmulOp>(op)) {
        if (auto accPhase=init.getAccPhaseAttr())
            if (accPhase.getValue()!=AccPhase::Unspecified) return result;
        a=init.getLhs(); b=init.getRhs(); dst=init.getDst();
    } else if (auto update=dyn_cast<TMatmulAccOp>(op)) {
        if (auto accPhase=update.getAccPhaseAttr())
            if (accPhase.getValue()!=AccPhase::Unspecified) return result;
        a=update.getLhs(); b=update.getRhs(); dst=update.getDst(); input=update.getAccIn();
    } else return result; // no GEMV, bias, MX, partial implicit lowering or UnitFlag claim
    if (phase.kPipeValue!=static_cast<PipelineType>(PIPE::PIPE_M)) return result;
    auto at=dyn_cast<TileBufType>(a.getType()),bt=dyn_cast<TileBufType>(b.getType()),
         ct=dyn_cast<TileBufType>(dst.getType());
    if (!at || !bt || !ct || !ct.getElementType().isF32() ||
        at.getElementType()!=bt.getElementType() ||
        (!at.getElementType().isF16() && !at.getElementType().isBF16())) return result;
    auto space=[](TileBufType t,AddressSpace expected) {
        auto s=dyn_cast_or_null<AddressSpaceAttr>(t.getMemorySpace());
        return s && s.getAddressSpace()==expected;
    };
    if (!space(at,AddressSpace::LEFT) || !space(bt,AddressSpace::RIGHT) ||
        !space(ct,AddressSpace::ACC) ||
        at.getBLayoutValueI32()!=0 || at.getSLayoutValueI32()!=1 || at.getSFractalSizeI32()!=512 ||
        bt.getBLayoutValueI32()!=0 || bt.getSLayoutValueI32()!=2 || bt.getSFractalSizeI32()!=512 ||
        ct.getBLayoutValueI32()!=1 || ct.getSLayoutValueI32()!=1 || ct.getSFractalSizeI32()!=1024)
        return result;
    auto av=exactMatrixValid(a),bv=exactMatrixValid(b),cv=exactMatrixValid(dst);
    auto footprint=exactAccFootprint(dst,phase,true);
    if (!av || !bv || !cv || !footprint || av->second!=bv->first ||
        av->first!=cv->first || bv->second!=cv->second) return result;
    if (input && (input.getType()!=dst.getType() || exactMatrixValid(input)!=cv ||
                  exactAccFootprint(input,phase,false)!=footprint)) return result;
    // These plain PTO-ISA overloads derive m/k from LEFT valid shape and n from
    // RIGHT valid shape. No descriptor/source-independent m/n estimate is used.
    if (cv->first>4095 || cv->second>4095 || av->second>4095 ||
        footprint->second!=4*cv->first*cv->second) return result;
    result.kind=input?ss::MmadInfo::Accumulate:ss::MmadInfo::Initialize;
    result.accumulatorBase=footprint->first; result.accumulatorBytes=footprint->second;
    result.m=cv->first; result.n=cv->second; result.k=av->second;
    result.input=at.getElementType().isF16()?ss::MmadInfo::F16:ss::MmadInfo::BF16;
    return result;
}

// Exact scalar interpretation in ITERATION ORDINALS. A first-only predicate
// has a finite initial phase; it is never evaluated at residue zero and then
// extrapolated to every iteration. Arithmetic used to recognize that predicate
// must be representable throughout the original loop's admitted domain.
class PeriodicScalar {
    Value iv;
    ss::LoopOrdinal induction;
    uint64_t offset = 0;
    bool initial = false, split = false;
    uint64_t lastOrdinal = 0;
    llvm::DenseMap<Value,std::optional<uint64_t>> periods;
    struct Affine { int64_t coefficient, constant; };
    llvm::DenseMap<Value,std::optional<Affine>> affineCache;

    std::optional<Affine> affine(Value value) {
        auto found=affineCache.find(value);
        if(found!=affineCache.end())return found->second;
        auto compute=[&]() -> std::optional<Affine> {
            if(!isa<IndexType>(value.getType()))return {};
            if(value==iv)return Affine{induction.step,induction.lower};
            if(auto c=literal(value))return Affine{0,*c};
            auto *op=value.getDefiningOp();
            if(!op||op->getNumOperands()!=2||
               !isa<arith::AddIOp,arith::SubIOp,arith::MulIOp>(op))return {};
            auto x=affine(op->getOperand(0)),y=affine(op->getOperand(1));
            if(!x||!y)return {};
            std::optional<int64_t> a,b;
            if(isa<arith::MulIOp>(op)) {
                if(x->coefficient&&y->coefficient)return {};
                if(y->coefficient)std::swap(x,y);
                a=ss::ordinalMultiply(x->coefficient,y->constant);
                b=ss::ordinalMultiply(x->constant,y->constant);
            } else {
                if(isa<arith::SubIOp>(op)) {
                    auto ya=ss::ordinalNegate(y->coefficient),yb=ss::ordinalNegate(y->constant);
                    if(!ya||!yb)return {};
                    y=Affine{*ya,*yb};
                }
                a=ss::ordinalAdd(x->coefficient,y->coefficient);
                b=ss::ordinalAdd(x->constant,y->constant);
            }
            if(!a||!b||lastOrdinal>uint64_t(INT64_MAX))return {};
            auto product=ss::ordinalMultiply(*a,int64_t(lastOrdinal));
            if(!product||!ss::ordinalAdd(*product,*b))return {};
            return Affine{*a,*b};
        };
        auto result=compute();affineCache[value]=result;return result;
    }
    // Return the value at ordinal zero of an eq/ne that flips permanently
    // after ordinal zero. Difference is used only after width/range checks.
    std::optional<bool> firstTest(Value value) {
        auto cmp=value.getDefiningOp<arith::CmpIOp>();
        if(!cmp||(cmp.getPredicate()!=arith::CmpIPredicate::eq &&
                  cmp.getPredicate()!=arith::CmpIPredicate::ne))return {};
        auto x=affine(cmp.getLhs()),y=affine(cmp.getRhs());
        if(!x||!y)return {};
        auto ya=ss::ordinalNegate(y->coefficient),yb=ss::ordinalNegate(y->constant);
        if(!ya||!yb)return {};
        auto a=ss::ordinalAdd(x->coefficient,*ya),b=ss::ordinalAdd(x->constant,*yb);
        if(!a||!b||!*a||*b)return {};
        return cmp.getPredicate()==arith::CmpIPredicate::eq;
    }
    std::optional<ss::OrdinalResidue> cycle(Value value) const {
        auto *op=value.getDefiningOp();if(!op)return {};
        std::optional<int64_t> modulus;
        if(isa<arith::RemSIOp,arith::RemUIOp>(op)&&op->getOperand(0)==iv)
            modulus=literal(op->getOperand(1));
        if(auto mask=dyn_cast<arith::AndIOp>(op)) {
            Value other=mask.getLhs()==iv?mask.getRhs():(mask.getRhs()==iv?mask.getLhs():Value());
            if(other) {
                auto bits=literal(other);
                if(!bits||*bits<0||*bits==INT64_MAX)return {};
                const uint64_t m=uint64_t(*bits)+1;
                if(m&(m-1))return {};
                modulus=int64_t(m);
            }
        }
        if(!modulus||*modulus<=0)return {};
        return ss::OrdinalResidue::get(induction,uint64_t(*modulus),offset);
    }
    void cut(const ss::OrdinalResidue &c,uint64_t raw) {
        if(auto r=c.preimage(raw)) {
            cuts.insert(*r);
            if(*r<UINT64_MAX)cuts.insert(*r+1);
        }
    }
public:
    std::set<uint64_t> cuts{0};
    PeriodicScalar(Value originalIV,ss::LoopOrdinal loop={},uint64_t start=0,
                   bool first=false,bool startup=false,std::optional<int64_t> upper={})
        :iv(originalIV),induction(loop),offset(start),initial(first),split(startup) {
        auto count=induction.count(upper.value_or(INT64_MAX));
        lastOrdinal=count&&*count?*count-1:0;
    }
    bool hasInitialization(Value value) {
        if(firstTest(value))return true;
        auto *op=value.getDefiningOp();
        if(!op||!value.getType().isInteger(1)||
           !isa<arith::CmpIOp,arith::AndIOp,arith::OrIOp,arith::XOrIOp>(op))return false;
        for(Value operand:op->getOperands())if(hasInitialization(operand))return true;
        return false;
    }
    std::optional<uint64_t> selectorPeriod(Value value) const {
        if(auto c=cycle(value))return c->period;
        return {};
    }
    std::optional<uint64_t> period(Value value) {
        auto found=periods.find(value);
        if(found!=periods.end())return found->second;
        auto result=computePeriod(value);periods[value]=result;return result;
    }
    std::optional<uint64_t> computePeriod(Value value) {
        if(literal(value))return 1;
        if(firstTest(value))return split?std::optional<uint64_t>(1):std::nullopt;
        if(auto c=cycle(value))return initial?uint64_t(1):c->period;
        auto *op=value.getDefiningOp();if(!op)return {};
        if(!isa<arith::CmpIOp>(op)&&
           !(value.getType().isInteger(1)&&isa<arith::AndIOp,arith::OrIOp,arith::XOrIOp>(op)))return {};
        uint64_t p=1;
        for(Value operand:op->getOperands()) {
            auto q=period(operand);
            if(!q||(p!=1&&*q!=1&&p!=*q))return {};
            p=std::max(p,*q);
        }
        if(auto cmp=dyn_cast<arith::CmpIOp>(op)) {
            // The loop below partitions a residue VALUE against a literal.
            // Equal effective periods do not make two residue values constant
            // on those intervals: for i=6*k, (i%4)!=(i%12) is false at k=0
            // and true at k=1, although both periods are 2. With no cuts, the
            // old importer could erase the entire executing true arm.
            // Boolean operands already carry their own truth-change cuts;
            // period-one operands are constant in this represented phase.
            // Other value/value comparisons need a separate exact partition.
            if(p>1 && !cmp.getLhs().getType().isInteger(1) &&
               !literal(cmp.getLhs()) && !literal(cmp.getRhs())) return {};
            for(unsigned side=0;side<2;++side) {
                auto c=cycle(op->getOperand(side));auto bound=literal(op->getOperand(1-side));
                if(!c||!bound||initial)continue;
                bool equality=cmp.getPredicate()==arith::CmpIPredicate::eq||cmp.getPredicate()==arith::CmpIPredicate::ne;
                // General strided threshold permutations are not flattened to
                // intervals. Equality has one modular preimage; unit stride
                // also admits threshold and wrap cuts. Other shapes decline.
                if(!equality&&induction.step!=1&&c->period!=1)return {};
                cut(*c,0);
                if(*bound>=0) {
                    cut(*c,uint64_t(*bound));
                    if(*bound<INT64_MAX)cut(*c,uint64_t(*bound)+1);
                }
            }
        }
        return p;
    }
    std::optional<int64_t> evaluate(Value value,uint64_t residue) {
        llvm::DenseMap<Value,std::optional<int64_t>> cache;
        return eval(value,residue,cache);
    }
private:
    std::optional<int64_t> eval(Value value,uint64_t residue,
        llvm::DenseMap<Value,std::optional<int64_t>> &cache) {
        auto found=cache.find(value);if(found!=cache.end())return found->second;
        auto compute=[&]() -> std::optional<int64_t> {
            if(auto c=literal(value))return c;
            if(auto first=firstTest(value)) {
                if(!split)return {};
                return initial?*first:!*first;
            }
            if(auto c=cycle(value))return int64_t(c->at(initial?0:residue));
            auto *op=value.getDefiningOp();if(!op||op->getNumOperands()!=2)return {};
            auto a=eval(op->getOperand(0),residue,cache),b=eval(op->getOperand(1),residue,cache);
            if(!a||!b)return {};
            if(auto cmp=dyn_cast<arith::CmpIOp>(op)) {
                ss::OrdinalCompare kind;
                switch(cmp.getPredicate()) {
                case arith::CmpIPredicate::eq: kind=ss::OrdinalCompare::EQ;break;
                case arith::CmpIPredicate::ne: kind=ss::OrdinalCompare::NE;break;
                case arith::CmpIPredicate::slt: kind=ss::OrdinalCompare::SLT;break;
                case arith::CmpIPredicate::sle: kind=ss::OrdinalCompare::SLE;break;
                case arith::CmpIPredicate::sgt: kind=ss::OrdinalCompare::SGT;break;
                case arith::CmpIPredicate::sge: kind=ss::OrdinalCompare::SGE;break;
                case arith::CmpIPredicate::ult: kind=ss::OrdinalCompare::ULT;break;
                case arith::CmpIPredicate::ule: kind=ss::OrdinalCompare::ULE;break;
                case arith::CmpIPredicate::ugt: kind=ss::OrdinalCompare::UGT;break;
                case arith::CmpIPredicate::uge: kind=ss::OrdinalCompare::UGE;break;
                default:return {};
                }
                return ss::ordinalCompare(kind,*a,*b,cmp.getLhs().getType().isInteger(1));
            }
            if(!value.getType().isInteger(1))return {};
            if(isa<arith::AndIOp>(op))return bool(*a)&&bool(*b);
            if(isa<arith::OrIOp>(op))return bool(*a)||bool(*b);
            if(isa<arith::XOrIOp>(op))return bool(*a)!=bool(*b);
            return {};
        };
        auto answer=compute();cache[value]=answer;return answer;
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
        physical=importCoverageStructuredSyncPhysicalFacts(function,ir);
        if(physical.status!=SyncPhysicalFacts::Status::Complete){reason=physical.reason;return false;}
        if(!physical.lifetimeScope ||
           (physical.lifetimeScope!=function.getOperation() &&
            !isa<SectionCubeOp,SectionVectorOp>(physical.lifetimeScope))) {
            reason="unsupported physical lifetime scope";return false;
        }
        return true;
    }
};
struct InvocationScope {
    Region *region=nullptr;
    SmallVector<scf::ForOp> wrappers;
    scf::ForOp localLoop;
    Operation *sequenceRoot=nullptr;
    Block *sequenceBlock=nullptr;
    unsigned sequenceOrder=0;
    bool sequenced=false;
};
class RegionLayout {
    NativeInventory &inventory;
    llvm::SmallPtrSet<Operation*,32> containsPayload;
    DominanceInfo dominance;
    bool fail(StringRef s){reason=s.str();return false;}
    bool normalized(scf::ForOp loop) {
        if(!isa<IndexType>(loop.getInductionVar().getType())||loop->hasAttr("unsignedCmp"))
            return fail("invocation wrapper requires a signed PTO index induction");
        auto lower=literal(loop.getLowerBound()),step=literal(loop.getStep());
        if(lower&&step) {
            if(*lower<0||*step<=0)
                return fail("invocation wrapper requires nonnegative lower and positive step");
            SyncAddressEvaluator integers(inventory.function);
            auto value=integers.evaluate(loop.getLowerBound());
            if(!value||value->getBitWidth()!=64)
                return fail("unqualified invocation index layout");
            return true;
        }
        if(!blockDistributedWrapper(loop))
            return fail("invocation wrapper requires constant positive coordinates or get_block_idx/get_block_num");
        // get_block_num is a target/runtime contract: positive and invariant for
        // the kernel launch. Preserve the original SSA values in emitted guards.
        DominanceInfo dom(inventory.function);
        if(!dom.properlyDominates(loop.getLowerBound(),loop.getOperation())||
           !dom.properlyDominates(loop.getStep(),loop.getOperation()))
            return fail("runtime invocation coordinates are unavailable at wrapper entry");
        return true;
    }
    SmallVector<Operation*> active(Region &r) {
        SmallVector<Operation*> result;
        if(llvm::hasSingleElement(r))for(Operation &op:r.front())if(containsPayload.contains(&op))result.push_back(&op);
        return result;
    }
    bool collectSequence(Region &r,ArrayRef<Operation*> roots,SmallVector<scf::ForOp> wrappers) {
        unsigned order=0;
        for(Operation *root:roots) {
            if(auto loop=dyn_cast<scf::ForOp>(root)) {
                unsigned nested=0;loop.getBody()->walk([&](scf::ForOp){++nested;});
                if(nested) return fail("nested child inside a sequential sibling requires a region summary");
                scopes.push_back({&loop.getRegion(),wrappers,loop,root,&r.front(),order++,true});
                continue;
            }
            return fail("sequential composition currently requires direct sibling loops");
        }
        return true;
    }
    bool collect(Region &r,SmallVector<scf::ForOp> wrappers) {
        if(!llvm::hasSingleElement(r))return fail("multi-block invocation region");
        auto roots=active(r);
        if(roots.empty())return true;
        // Sequential composition is for genuine sibling LOOPS. A prelude/body/
        // epilogue region also has several roots, and must keep the existing
        // boundary handling instead of being refused as a non-loop sibling.
        unsigned loopRoots=0;
        for(Operation *root:roots) if(isa<scf::ForOp>(root)) ++loopRoots;
        if(roots.size()>1&&loopRoots>1)return collectSequence(r,roots,std::move(wrappers));
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
        if(loops>1)return fail("mixed nested child transfers require a qualified region summary");
        scopes.push_back({&r,std::move(wrappers)});return true;
    }
public:
    std::vector<InvocationScope> scopes;
    std::string reason;
    explicit RegionLayout(NativeInventory &i):inventory(i),dominance(i.function) {
        for(auto *phase:i.physical.phases)for(auto *op=phase->elementOp;op&&op!=i.function.getOperation();op=op->getParentOp())
            containsPayload.insert(op);
    }
    bool build(){
        Region &root=inventory.physical.lifetimeScope==inventory.function.getOperation()?
            inventory.function.getBody():inventory.physical.lifetimeScope->getRegion(0);
        return collect(root,{});
    }
};

enum class StartupCase { Whole, Nonempty, Empty };
enum class AtomRole { Outside, Ordinary, Initial, Steady };

struct NativeFacts {
    func::FuncOp function;
    InsertSyncGMAliasMode gm;
    NativeInventory &inventory;
    Region *scope;
    SmallVector<scf::ForOp> wrappers;
    Operation *sequenceRoot=nullptr;
    Block *sequenceBlock=nullptr;
    unsigned sequenceOrder=0;
    bool sequenced=false;
    Buffer2MemInfoMap &buffers;
    SyncPhysicalFacts physical;
    scf::ForOp loop;
    ss::LoopOrdinal induction;
    std::optional<int64_t> knownUpper;
    StartupCase startupCase=StartupCase::Whole;
    bool startup=false,caseGuard=false,needsEmptyCase=false;
    uint64_t ordinalOffset=0;
    std::vector<AtomRole> roles;
    ss::Model model;
    std::vector<ss::InvocationRequirement> carried;
    std::vector<unsigned> origin;
    std::vector<ss::Segment> segments;
    llvm::DenseMap<Operation*,unsigned> phaseId;
    std::vector<std::vector<std::pair<Value,bool>>> guards;
    std::vector<std::vector<uint64_t>> residues;
    std::map<const BaseMemInfo*,std::optional<SyncPhysicalSlotMapping>> mappings;
    std::string reason;
    NativeFacts(NativeInventory &i,const InvocationScope &unit,InsertSyncGMAliasMode contract,
                StartupCase selected=StartupCase::Whole)
        :function(i.function),gm(contract),inventory(i),scope(unit.region),wrappers(unit.wrappers),
         sequenceRoot(unit.sequenceRoot),sequenceBlock(unit.sequenceBlock),
         sequenceOrder(unit.sequenceOrder),sequenced(unit.sequenced),
         buffers(i.buffers),loop(unit.localLoop),startupCase(selected) {}
    int64_t bodyBase() const { return *induction.value(ordinalOffset); }
    PeriodicScalar scalar(bool first=false) const {
        // An op class is a value-semantic handle: const protects the handle,
        // not the IR, and the generated accessors are non-const. Copy it.
        scf::ForOp body=loop;
        return PeriodicScalar(body?body.getInductionVar():Value(),induction,
                              first?0:ordinalOffset,first,startup,knownUpper);
    }
    std::vector<ss::SiteOrigin> siteOrigins() const {
        std::vector<ss::SiteOrigin> result;
        for(std::size_t a=0;a<origin.size();++a) {
            // Coalescing currently admits period-one startup only. Other
            // represented atoms keep one command per existing generated site.
            auto phase=ss::SiteOrigin::Other;
            if(startup && wrappers.empty() && model.period==1) {
                if(roles[a]==AtomRole::Initial)phase=ss::SiteOrigin::Initial;
                if(roles[a]==AtomRole::Steady)phase=ss::SiteOrigin::Steady;
            }
            result.push_back({phase,origin[a]});
        }
        return result;
    }
    ss::InvocationModel invocationModel() const {return {model,carried};}
    bool fail(StringRef message) { reason=message.str(); return false; }

    bool build() {
        physical=inventory.physical;
        auto inUnit=[&](Operation *op) {
            if(sequenceRoot)return op==sequenceRoot||sequenceRoot->isAncestor(op);
            return belongsTo(*scope,op);
        };
        physical.phases.erase(std::remove_if(physical.phases.begin(),physical.phases.end(),
            [&](const auto *p){return !inUnit(p->elementOp);}),physical.phases.end());
        bool multiple=false;
        if(!loop)scope->walk([&](scf::ForOp candidate) {
            if(sequenceRoot&&!sequenceRoot->isAncestor(candidate))return;
            if (loop) multiple=true; else loop=candidate;
        });
        if (multiple) return fail("local region still contains mixed child-loop transfers");
        if (physical.phases.size()>std::numeric_limits<unsigned>::max())
            return fail("physical phase identity width overflow");
        model.recurring=bool(loop);
        if (loop) {
            auto lower=literal(loop.getLowerBound()),step=literal(loop.getStep());
            if(!lower||!step||*lower<0||*step<=0||!isa<IndexType>(loop.getInductionVar().getType())||
               loop->hasAttr("unsignedCmp"))
                return fail("structured ordinal loop requires constant nonnegative lower and positive step, signed index");
            induction={*lower,*step};knownUpper=literal(loop.getUpperBound());
            SyncAddressEvaluator integers(function);auto value=integers.evaluate(loop.getLowerBound());
            if(!value||value->getBitWidth()!=64)
                return fail("structured loop requires the qualified signed-64-bit PTO index layout");
        }
        if(!wrappers.empty()&&loop) {
            DominanceInfo dominance(function);
            if(!dominance.properlyDominates(loop.getUpperBound(),wrappers.front().getOperation()))
                return fail("inner bound varies across enclosing invocations");
        }
        guards.resize(physical.phases.size());
        residues.resize(physical.phases.size());
        segments.assign(physical.phases.size(),ss::Segment::Body);
        llvm::DenseMap<Operation*,bool> beforeLoop;bool passedLoop=false;
        for(Operation &op:scope->front()) {
            if(&op==loop.getOperation())passedLoop=true;
            beforeLoop[&op]=!passedLoop;
        }
        auto discover=scalar();
        for(unsigned p=0;p<physical.phases.size();++p) {
            auto *op=physical.phases[p]->elementOp;phaseId[op]=p;
            if(loop&&!loop->isAncestor(op)) {
                if(loop->getBlock()!=&scope->front()||op->getBlock()!=&scope->front())
                    return fail("boundary payload requires a direct, unconditional loop invocation");
                segments[p]=beforeLoop.lookup(op)?ss::Segment::Prelude:ss::Segment::Epilogue;
            }
            for(Operation *parent=op->getParentOp();parent&&parent!=scope->getParentOp();parent=parent->getParentOp()) {
                if(parent==loop.getOperation())continue;
                auto branch=dyn_cast<scf::IfOp>(parent);
                if(!branch)return fail("unrepresented physical control region");
                Region *r=op->getParentRegion();
                while(r&&r->getParentOp()!=parent)r=r->getParentOp()->getParentRegion();
                if(!r)return fail("invalid control ancestry");
                const bool thenArm=r==&branch.getThenRegion();
                guards[p].push_back({branch.getCondition(),thenArm});
                startup|=loop&&discover.hasInitialization(branch.getCondition());
            }
            for(const auto *accesses:{&physical.phases[p]->useVec,&physical.phases[p]->defVec})
                for(auto *access:*accesses)mappings.emplace(access,qualifySyncPhysicalSlots(access,buffers));
        }
        // Exactly two structural cases, not an unrolling count: empty, and
        // first iteration followed by the periodic tail. The first iteration
        // is a VIRTUAL prefix; all original operations remain in their blocks.
        if(startup) {
            auto trips=knownUpper?induction.count(*knownUpper):std::optional<uint64_t>();
            if(startupCase==StartupCase::Whole)
                startupCase=trips&&!*trips?StartupCase::Empty:StartupCase::Nonempty;
            needsEmptyCase=startupCase==StartupCase::Nonempty&&!trips;
            caseGuard=!trips;
            ordinalOffset=startupCase==StartupCase::Nonempty?1:0;
            if(!induction.value(ordinalOffset))return fail("unrepresentable startup-tail origin");
        } else if(startupCase!=StartupCase::Whole) {
            return fail("startup case requested without a qualified initialization predicate");
        }
        if(startupCase==StartupCase::Empty) {
            model.recurring=false;model.period=1;
            for(unsigned p=0;p<physical.phases.size();++p)if(segments[p]!=ss::Segment::Body) {
                auto lane=pipe(static_cast<PIPE>(physical.phases[p]->kPipeValue));
                if(!lane)return fail("unqualified physical pipe");
                origin.push_back(p);roles.push_back(AtomRole::Outside);
                model.atoms.push_back({0,p,{physical.cube?ss::Core::AIC:ss::Core::AIV,*lane},ss::Segment::Body});
            }
        } else {
            auto recurring=scalar(),first=scalar(true);
            uint64_t period=1;
            auto mergePeriod=[&](uint64_t p) {
                if(p!=1&&period!=1&&p!=period)return false;
                period=std::max(period,p);return true;
            };
            for(unsigned p=0;p<physical.phases.size();++p) {
                for(auto [condition,take]:guards[p]) {
                    (void)take;
                    auto d=recurring.period(condition);
                    if(!d||!mergePeriod(*d))
                        return fail("payload guard is outside the ordinal periodic/initialization fragment");
                    if(startup&&!first.period(condition))return fail("initial guard has no qualified scalar interpretation");
                }
                for(const auto *accesses:{&physical.phases[p]->useVec,&physical.phases[p]->defVec})
                    for(auto *access:*accesses) {
                        const auto &mapping=mappings.at(access);
                        if(mapping&&mapping->selector) {
                            auto d=recurring.period(mapping->selector);
                            if(d&&!mergePeriod(*d))return fail("incompatible ordinal slot periods");
                        }
                    }
            }
            model.period=period;recurring.cuts.insert(period);
            std::vector<uint64_t> breaks;
            for(uint64_t cut:recurring.cuts)if(cut<=period)breaks.push_back(cut);
            auto enabled=[&](unsigned p,PeriodicScalar &values,uint64_t r) -> std::optional<bool> {
                bool active=true;
                for(auto [condition,take]:guards[p]) {
                    auto value=values.evaluate(condition,r);
                    if(!value)return {};
                    active&=bool(*value)==take;
                }
                return active;
            };
            for(unsigned p=0;p<physical.phases.size();++p) {
                auto lane=pipe(static_cast<PIPE>(physical.phases[p]->kPipeValue));
                if(!lane)return fail("unqualified physical pipe");
                auto append=[&](uint64_t r,ss::Segment segment,AtomRole role) {
                    origin.push_back(p);roles.push_back(role);
                    model.atoms.push_back({r,p,{physical.cube?ss::Core::AIC:ss::Core::AIV,*lane},segment});
                };
                if(segments[p]!=ss::Segment::Body) {
                    append(0,segments[p],AtomRole::Outside);continue;
                }
                if(startup) {
                    auto active=enabled(p,first,0);
                    if(!active)return fail("initial guard evaluation unavailable");
                    if(*active)append(0,ss::Segment::Prelude,AtomRole::Initial);
                }
                bool explicitSlots=false;
                for(const auto *accesses:{&physical.phases[p]->useVec,&physical.phases[p]->defVec})
                    for(auto *access:*accesses) {
                        const auto &mapping=mappings.at(access);
                        if(mapping&&mapping->selector&&mapping->bases.size()>=period&&recurring.period(mapping->selector))
                            explicitSlots=true;
                    }
                for(std::size_t i=1;i<breaks.size();++i) {
                    uint64_t lo=breaks[i-1],hi=breaks[i];auto active=enabled(p,recurring,lo);
                    if(!active)return fail("ordinal periodic guard evaluation unavailable");
                    if(!*active||lo==hi)continue;
                    if(hi-lo>1&&!explicitSlots)return fail("multi-residue payload needs an interval-phase transfer");
                    for(uint64_t r=lo;r<hi;++r) {
                        residues[p].push_back(r);
                        append(r,ss::Segment::Body,startup?AtomRole::Steady:AtomRole::Ordinary);
                    }
                }
            }
        }
        model.allowBoundaryKeyReuse=startup&&startupCase==StartupCase::Nonempty;
        for (std::size_t id=0;id<origin.size();++id)
            model.atoms[id].matrix=matrixLoweringFacts(*physical.phases[origin[id]]);
        auto recurringScalar=scalar(),initialScalar=scalar(true);
        auto alias=[&](const BaseMemInfo *a,std::size_t pa,const BaseMemInfo *b,std::size_t pb) {
            if (!logicalSyncMayAlias(a,b,function,gm)) return false;
            const auto &ma=mappings.at(a), &mb=mappings.at(b);
            if (!ma || !mb || ma->scope!=mb->scope) return true;
            auto selected=[&](const SyncPhysicalSlotMapping &m,std::size_t atom) -> std::optional<uint64_t> {
                auto &values=roles[atom]==AtomRole::Initial?initialScalar:recurringScalar;
                uint64_t r=model.atoms[atom].residue;
                if (!m.selector) return m.bases.size()==1?std::optional<uint64_t>(m.bases.front()):std::nullopt;
                auto d=values.period(m.selector);
                if (!d || model.period%*d) return {};
                auto s=values.evaluate(m.selector,r);
                if (!s || *s<0 || uint64_t(*s)>=m.bases.size()) return {};
                return m.bases[uint64_t(*s)];
            };
            auto x=selected(*ma,pa),y=selected(*mb,pb);
            if (!x || !y) return true;
            return *x<*y+mb->bytes && *y<*x+ma->bytes;
        };
        for (std::size_t p=0;p<origin.size();++p) for (std::size_t q=0;q<origin.size();++q) {
            auto distance=ss::priorDistance(model,p,q); if (!distance && wrappers.empty()) continue;
            auto *a=physical.phases[origin[p]], *b=physical.phases[origin[q]];
            bool conflict=false,acc=false,visibility=false,accOnly=true,exactAccumulatorWitness=true;
            auto exactAccumulatorEffect=[&](const BaseMemInfo *info,std::size_t atom) {
                const auto &matrix=model.atoms[atom].matrix;
                return info && matrix.kind!=ss::MmadInfo::Unknown &&
                    info->scope==AddressSpace::ACC && info->hasKnownPhysicalAddresses &&
                    !info->aliasesUnknownRange && info->baseAddresses.size()==1 &&
                    info->baseAddresses.front()==matrix.accumulatorBase &&
                    info->allocateSize==matrix.accumulatorBytes;
            };
            auto pair=[&](const auto &left,bool writeA,const auto &right,bool writeB) {
                for (auto *x:left) for (auto *y:right) {
                    bool resource=!writeA&&!writeB && x->scope==AddressSpace::ACC &&
                        y->scope==AddressSpace::ACC && model.atoms[p].lane!=model.atoms[q].lane;
                    if (!writeA&&!writeB&&!resource) continue;
                    if (!alias(x,p,y,q)) continue;
                    conflict=true; acc|=resource;
                    const bool accPair=x->scope==AddressSpace::ACC && y->scope==AddressSpace::ACC;
                    accOnly &= accPair;
                    if (accPair) exactAccumulatorWitness &=
                        exactAccumulatorEffect(x,p) && exactAccumulatorEffect(y,q);
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
                // Keep the immutable obligation; only its REQUIRED property is
                // narrower. Invalid/unavailable hardware facts still require
                // ordinary completion, and cannot affect non-ACC conflicts.
                const bool accumulatorUpdate=accOnly && exactAccumulatorWitness && !visibility && !acc &&
                    model.atoms[p].lane==ss::Lane{ss::Core::AIC,ss::Pipe::M} &&
                    model.atoms[q].lane==ss::Lane{ss::Core::AIC,ss::Pipe::M};
                if (accumulatorUpdate) property=ss::Property::AccumulatorUpdate;
                if(distance) {
                    ss::Requirement requirement{p,q,*distance,property};
                    if (accumulatorUpdate) {
                        requirement.hasStorageWitness=true;
                        requirement.storageBase=model.atoms[p].matrix.accumulatorBase;
                        requirement.storageBytes=model.atoms[p].matrix.accumulatorBytes;
                    }
                    model.requirements.push_back(requirement);
                }
                if(!wrappers.empty())carried.push_back({p,q,property==ss::Property::AccumulatorUpdate?ss::Property::Completion:property});
            }
        }
        return true;
    }
};


// A sequence bridge summarizes only physical hazards that cross two direct
// sibling loop regions in the same block. Local loop protocols must be fully
// verified before this layer runs, so no local notification remains live at a
// loop exit. A bridge is placed at the scalar boundary: Set after the source
// loop and Wait (or a same-pipe barrier) before the target loop. No internal
// PIPE_ALL is introduced.
enum class SequenceHazard : uint8_t { ReadAfterWrite, WriteAfterRead,
                                      WriteAfterWrite, SharedResource };
struct SequenceRequirement {
    Block *block=nullptr;
    Operation *sourceRoot=nullptr,*targetRoot=nullptr;
    Operation *sourcePhase=nullptr,*targetPhase=nullptr;
    unsigned sourceOrder=0,targetOrder=0;
    ss::Lane source{},target{};
    const BaseMemInfo *sourceAccess=nullptr,*targetAccess=nullptr;
    SequenceHazard kind=SequenceHazard::ReadAfterWrite;
    bool gm=false;
};
struct SequenceRequirementLedger {
    ss::Status status=ss::Status::Applied;
    std::string reason;
    std::vector<SequenceRequirement> requirements;
};

// Enumerate every immutable cross-loop access requirement.  Bridge selection
// below may merge several requirements at one boundary, but it cannot remove
// them from this ledger.  The same population is re-extracted from the fresh
// post-emission physical translation before the clone can commit.
SequenceRequirementLedger deriveSequenceRequirements(
    func::FuncOp function,const SyncPhysicalFacts &physical,
    const RegionLayout &layout,InsertSyncGMAliasMode gm,
    ss::HardwareContract hardware) {
    SequenceRequirementLedger out;
    struct Group { Block *block=nullptr;std::vector<const InvocationScope*> scopes; };
    std::vector<Group> groups;
    for(const auto &scope:layout.scopes) if(scope.sequenced) {
        auto found=llvm::find_if(groups,[&](const Group &g){return g.block==scope.sequenceBlock;});
        if(found==groups.end())groups.push_back({scope.sequenceBlock,{&scope}});
        else found->scopes.push_back(&scope);
    }
    ss::Target target;target.hardware=hardware;
    auto laneOf=[&](const CompoundInstanceElement *phase)->std::optional<ss::Lane> {
        auto p=pipe(static_cast<PIPE>(phase->kPipeValue));
        if(!p)return {};
        ss::Lane lane{physical.cube?ss::Core::AIC:ss::Core::AIV,*p};
        return target.supports(lane)?std::optional<ss::Lane>(lane):std::nullopt;
    };
    auto phasesOf=[&](Operation *root) {
        SmallVector<const CompoundInstanceElement*> phases;
        for(auto *phase:physical.phases) {
            if(root==phase->elementOp||root->isAncestor(phase->elementOp))phases.push_back(phase);
        }
        return phases;
    };
    for(auto &group:groups) {
        llvm::sort(group.scopes,[](auto *a,auto *b){return a->sequenceOrder<b->sequenceOrder;});
        for(std::size_t j=0;j<group.scopes.size();++j) {
            auto *dstScope=group.scopes[j];auto dstPhases=phasesOf(dstScope->sequenceRoot);
            for(std::size_t i=0;i<j;++i) {
                auto *srcScope=group.scopes[i];auto srcPhases=phasesOf(srcScope->sequenceRoot);
                for(auto *a:srcPhases)for(auto *b:dstPhases) {
                    auto la=laneOf(a),lb=laneOf(b);
                    if(!la||!lb){out.status=ss::Status::Unsupported;out.reason="sequential sibling has an unsupported physical lane";return out;}
                    auto pairs=[&](const auto &left,bool writeA,const auto &right,bool writeB,
                                   SequenceHazard kind) {
                        for(auto *x:left)for(auto *y:right) {
                            bool resource=!writeA&&!writeB&&x&&y&&x->scope==AddressSpace::ACC&&
                                y->scope==AddressSpace::ACC&&*la!=*lb;
                            if(!writeA&&!writeB&&!resource)continue;
                            if(!logicalSyncMayAlias(x,y,function,gm))continue;
                            out.requirements.push_back({group.block,srcScope->sequenceRoot,
                                dstScope->sequenceRoot,a->elementOp,b->elementOp,
                                srcScope->sequenceOrder,dstScope->sequenceOrder,*la,*lb,x,y,
                                resource?SequenceHazard::SharedResource:kind,
                                x&&y&&x->scope==AddressSpace::GM&&y->scope==AddressSpace::GM});
                        }
                    };
                    pairs(a->defVec,true,b->useVec,false,SequenceHazard::ReadAfterWrite);
                    pairs(a->useVec,false,b->defVec,true,SequenceHazard::WriteAfterRead);
                    pairs(a->defVec,true,b->defVec,true,SequenceHazard::WriteAfterWrite);
                    pairs(a->useVec,false,b->useVec,false,SequenceHazard::SharedResource);
                }
            }
        }
    }
    return out;
}

struct SequenceBridge {
    Block *block=nullptr;
    Operation *sourceRoot=nullptr,*targetRoot=nullptr;
    unsigned sourceOrder=0,targetOrder=0;
    ss::Lane source{},target{};
    bool barrier=false;
    unsigned key=0;
};
struct SequenceBridgePlan {
    ss::Status status=ss::Status::Applied;
    std::string reason;
    std::vector<SequenceBridge> bridges;
};

SequenceBridgePlan buildSequenceBridges(const SequenceRequirementLedger &ledger,
    ss::HardwareContract hardware) {
    SequenceBridgePlan out;
    if(ledger.status!=ss::Status::Applied){out.status=ledger.status;out.reason=ledger.reason;return out;}
    ss::Target target;target.hardware=hardware;
    using NeedKey=std::tuple<Block*,unsigned,ss::Lane,ss::Lane>;
    struct Need { Operation *sourceRoot=nullptr,*targetRoot=nullptr;
                  unsigned sourceOrder=0,targetOrder=0;bool gm=false; };
    std::map<NeedKey,Need> needs;
    for(const auto &requirement:ledger.requirements) {
        NeedKey key{requirement.block,requirement.targetOrder,
                    requirement.source,requirement.target};
        auto &need=needs[key];
        if(!need.sourceRoot||need.sourceOrder<requirement.sourceOrder) {
            need.sourceRoot=requirement.sourceRoot;
            need.sourceOrder=requirement.sourceOrder;
        }
        need.targetRoot=requirement.targetRoot;
        need.targetOrder=requirement.targetOrder;
        need.gm|=requirement.gm;
    }
    for(const auto &[key,need]:needs) {
        auto [block,targetOrder,sourceLane,targetLane]=key;(void)targetOrder;
        if(need.gm) {
            out.status=ss::Status::Unsupported;
            out.reason="sequential sibling GM hazard requires a qualified visibility recipe";
            return out;
        }
        if(sourceLane==targetLane) {
            if(target.synchronous(sourceLane))continue;
            if(!target.barrier(sourceLane)) {
                out.status=ss::Status::Unsupported;out.reason="sequential sibling same-pipe hazard has no barrier";return out;
            }
            out.bridges.push_back({block,need.sourceRoot,need.targetRoot,
                need.sourceOrder,need.targetOrder,sourceLane,targetLane,true,0});
        } else {
            if(!target.event(sourceLane,targetLane)) {
                out.status=ss::Status::Unsupported;out.reason="sequential sibling hazard has no legal direct event";return out;
            }
            out.bridges.push_back({block,need.sourceRoot,need.targetRoot,
                need.sourceOrder,need.targetOrder,sourceLane,targetLane,false,0});
        }
    }
    // One interval coloring per direction. An interval begins after its source
    // loop and is consumed before its target loop. Touching intervals do not
    // overlap: a wait before root j precedes a set after root j.
    std::map<std::pair<ss::Lane,ss::Lane>,std::vector<std::size_t>> byDirection;
    for(std::size_t i=0;i<out.bridges.size();++i)if(!out.bridges[i].barrier)
        byDirection[{out.bridges[i].source,out.bridges[i].target}].push_back(i);
    for(auto &[direction,indices]:byDirection) {
        llvm::sort(indices,[&](auto a,auto b) {
            const auto &x=out.bridges[a],&y=out.bridges[b];
            return std::tie(x.sourceOrder,x.targetOrder)<std::tie(y.sourceOrder,y.targetOrder);
        });
        std::vector<std::pair<unsigned,unsigned>> active; // end position, key
        for(auto id:indices) {
            auto &bridge=out.bridges[id];unsigned start=2*bridge.sourceOrder+1,end=2*bridge.targetOrder;
            active.erase(std::remove_if(active.begin(),active.end(),[&](auto x){return x.first<=start;}),active.end());
            bool assigned=false;
            for(unsigned key:target.compilerKeys) {
                if(!target.available(direction.first,direction.second,key)||
                   llvm::any_of(active,[&](auto x){return x.second==key;}))continue;
                bridge.key=key;active.emplace_back(end,key);assigned=true;break;
            }
            if(!assigned) {out.status=ss::Status::AllocationFailure;out.reason="sequential sibling event intervals exceed the qualified key pool";return out;}
        }
    }
    llvm::sort(out.bridges,[](const SequenceBridge &a,const SequenceBridge &b) {
        return std::tie(a.targetOrder,a.barrier,a.source,a.target,a.key,a.sourceOrder)<
               std::tie(b.targetOrder,b.barrier,b.source,b.target,b.key,b.sourceOrder);
    });
    return out;
}

struct EmittedSequenceBridge {
    SequenceBridge bridge;
    Operation *set=nullptr,*wait=nullptr,*barrier=nullptr;
};
std::vector<EmittedSequenceBridge> emitSequenceBridges(
    MLIRContext *context,ArrayRef<SequenceBridge> bridges) {
    std::vector<EmittedSequenceBridge> emitted(bridges.size());
    for(std::size_t i=0;i<bridges.size();++i)emitted[i].bridge=bridges[i];
    std::map<Operation*,std::vector<std::size_t>> after,before;
    for(std::size_t i=0;i<bridges.size();++i) {
        if(!bridges[i].barrier)after[bridges[i].sourceRoot].push_back(i);
        before[bridges[i].targetRoot].push_back(i);
    }
    for(auto &[root,indices]:after) {
        llvm::sort(indices,[&](auto a,auto b){const auto &x=bridges[a],&y=bridges[b];return std::tie(x.source,x.target,x.key)<std::tie(y.source,y.target,y.key);});
        Operation *cursor=root;
        for(auto id:indices) {
            const auto &x=bridges[id];OpBuilder b(cursor);b.setInsertionPointAfter(cursor);
            auto op=b.create<SetFlagOp>(root->getLoc(),PipeAttr::get(context,pipe(x.source.pipe)),
                PipeAttr::get(context,pipe(x.target.pipe)),EventAttr::get(context,static_cast<EVENT>(x.key)));
            emitted[id].set=op.getOperation();cursor=op.getOperation();
        }
    }
    for(auto &[root,indices]:before) {
        llvm::sort(indices,[&](auto a,auto b){const auto &x=bridges[a],&y=bridges[b];return std::tie(x.barrier,x.source,x.target,x.key)<std::tie(y.barrier,y.source,y.target,y.key);});
        OpBuilder b(root);b.setInsertionPoint(root);
        for(auto id:indices) {
            const auto &x=bridges[id];
            if(x.barrier) {
                auto op=b.create<BarrierOp>(root->getLoc(),PipeAttr::get(context,pipe(x.target.pipe)));
                emitted[id].barrier=op.getOperation();
            } else {
                auto op=b.create<WaitFlagOp>(root->getLoc(),PipeAttr::get(context,pipe(x.source.pipe)),
                    PipeAttr::get(context,pipe(x.target.pipe)),EventAttr::get(context,static_cast<EVENT>(x.key)));
                emitted[id].wait=op.getOperation();
            }
        }
    }
    return emitted;
}

bool verifySequenceBridges(ArrayRef<EmittedSequenceBridge> emitted,std::string &why) {
    using IntervalKey=std::tuple<Block*,ss::Lane,ss::Lane,unsigned>;
    std::map<IntervalKey,std::vector<std::pair<unsigned,unsigned>>> intervals;
    for(const auto &e:emitted) {
        const auto &x=e.bridge;
        if(!x.block||!x.sourceRoot||!x.targetRoot||x.sourceRoot->getBlock()!=x.block||x.targetRoot->getBlock()!=x.block||
           !x.sourceRoot->isBeforeInBlock(x.targetRoot)) {why="invalid sequential sibling bridge roots";return false;}
        if(x.barrier) {
            auto op=dyn_cast_or_null<BarrierOp>(e.barrier);
            if(!op||op->getBlock()!=x.block||!x.sourceRoot->isBeforeInBlock(e.barrier)||
               !e.barrier->isBeforeInBlock(x.targetRoot)||
               op.getPipe().getPipe()!=pipe(x.target.pipe)) {why="emitted sequential barrier changed boundary or lane";return false;}
        } else {
            auto set=dyn_cast_or_null<SetFlagOp>(e.set);auto wait=dyn_cast_or_null<WaitFlagOp>(e.wait);
            if(!set||!wait||set->getBlock()!=x.block||wait->getBlock()!=x.block||
               !x.sourceRoot->isBeforeInBlock(e.set)||!e.set->isBeforeInBlock(e.wait)||!e.wait->isBeforeInBlock(x.targetRoot)||
               set.getSrcPipe().getPipe()!=pipe(x.source.pipe)||set.getDstPipe().getPipe()!=pipe(x.target.pipe)||
               wait.getSrcPipe().getPipe()!=pipe(x.source.pipe)||wait.getDstPipe().getPipe()!=pipe(x.target.pipe)||
               unsigned(set.getEventId().getEvent())!=x.key||unsigned(wait.getEventId().getEvent())!=x.key) {
                why="emitted sequential event changed endpoint, direction or key";return false;
            }
            intervals[{x.block,x.source,x.target,x.key}].push_back({2*x.sourceOrder+1,2*x.targetOrder});
        }
    }
    for(auto &[key,ranges]:intervals) {
        llvm::sort(ranges);
        for(std::size_t i=1;i<ranges.size();++i)if(ranges[i].first<ranges[i-1].second) {
            why="sequential event key is rearmed before consumption";return false;
        }
    }
    return true;
}

bool sameSequenceRequirement(const SequenceRequirement &a,
                             const SequenceRequirement &b) {
    return a.block==b.block&&a.sourceRoot==b.sourceRoot&&a.targetRoot==b.targetRoot&&
        a.sourcePhase==b.sourcePhase&&a.targetPhase==b.targetPhase&&
        a.sourceOrder==b.sourceOrder&&a.targetOrder==b.targetOrder&&
        a.source==b.source&&a.target==b.target&&a.kind==b.kind&&a.gm==b.gm&&
        a.sourceAccess&&b.sourceAccess&&a.targetAccess&&b.targetAccess&&
        *a.sourceAccess==*b.sourceAccess&&*a.targetAccess==*b.targetAccess;
}

// Verify the immutable requirement population against actual post-emission IR,
// independently of the selected SequenceBridge objects.  This catches both a
// missing selected bridge and a generated mechanism moved outside the physical
// source/target interval.
bool verifySequenceRequirements(ArrayRef<SequenceRequirement> expected,
                                ArrayRef<SequenceRequirement> actual,
                                ss::HardwareContract hardware,std::string &why) {
    if(expected.size()!=actual.size()) {
        why="post-emission sequential requirement population changed";return false;
    }
    std::vector<bool> matched(actual.size());
    for(const auto &requirement:expected) {
        auto found=std::find_if(actual.begin(),actual.end(),[&](const auto &candidate) {
            auto index=std::size_t(&candidate-actual.data());
            return !matched[index]&&sameSequenceRequirement(requirement,candidate);
        });
        if(found==actual.end()) {
            why="post-emission sequential physical hazard changed";return false;
        }
        matched[std::size_t(found-actual.begin())]=true;
    }

    ss::Target target;target.hardware=hardware;
    std::set<Block*> blocks;
    for(const auto &requirement:actual) {
        blocks.insert(requirement.block);
        if(requirement.gm) {
            why="post-emission sequential GM hazard lacks visibility";return false;
        }
        if(requirement.source==requirement.target) {
            if(target.synchronous(requirement.source))continue;
            bool covered=false;
            for(Operation &op:*requirement.block) {
                if(auto barrier=dyn_cast<BarrierOp>(op)) {
                    if(barrier.getPipe().getPipe()==pipe(requirement.target.pipe)&&
                       requirement.sourceRoot->isBeforeInBlock(&op)&&
                       op.isBeforeInBlock(requirement.targetRoot)) {covered=true;break;}
                }
            }
            if(!covered) {
                why="sequential same-pipeline hazard lacks an actual interval barrier";return false;
            }
            continue;
        }
        if(!target.event(requirement.source,requirement.target)) {
            why="post-emission sequential hazard uses an unqualified event direction";return false;
        }
        bool covered=false;
        for(Operation &candidate:*requirement.block) {
            auto set=dyn_cast<SetFlagOp>(candidate);
            if(!set||!requirement.sourceRoot->isBeforeInBlock(&candidate)||
               !candidate.isBeforeInBlock(requirement.targetRoot)||
               set.getSrcPipe().getPipe()!=pipe(requirement.source.pipe)||
               set.getDstPipe().getPipe()!=pipe(requirement.target.pipe))continue;
            for(Operation *next=candidate.getNextNode();next&&next!=requirement.targetRoot;
                next=next->getNextNode())if(auto wait=dyn_cast<WaitFlagOp>(next))
                if(wait.getSrcPipe()==set.getSrcPipe()&&wait.getDstPipe()==set.getDstPipe()&&
                   wait.getEventId()==set.getEventId()) {covered=true;break;}
            if(covered)break;
        }
        if(!covered) {
            why="sequential cross-pipeline hazard lacks an actual set/wait interval";return false;
        }
    }

    // Reconstruct consume-before-rearm from the actual direct-block protocol,
    // without consulting constructor-assigned interval metadata.
    using EventKey=std::tuple<Block*,PIPE,PIPE,EVENT>;
    std::map<EventKey,Operation*> open;
    for(Block *block:blocks)for(Operation &operation:*block) {
        if(auto set=dyn_cast<SetFlagOp>(operation)) {
            EventKey key{block,set.getSrcPipe().getPipe(),set.getDstPipe().getPipe(),
                         set.getEventId().getEvent()};
            if(open[key]) {why="actual sequential event is rearmed before consumption";return false;}
            open[key]=&operation;
        } else if(auto wait=dyn_cast<WaitFlagOp>(operation)) {
            EventKey key{block,wait.getSrcPipe().getPipe(),wait.getDstPipe().getPipe(),
                         wait.getEventId().getEvent()};
            auto found=open.find(key);
            if(found==open.end()||!found->second) {
                why="actual sequential wait has no preceding publication";return false;
            }
            found->second=nullptr;
        }
    }
    if(llvm::any_of(open,[](const auto &item){return item.second!=nullptr;})) {
        why="actual sequential publication has no consumption";return false;
    }
    return true;
}

Value indexConstant(OpBuilder &builder,Location loc,uint64_t value) {
    return builder.create<arith::ConstantIndexOp>(loc,int64_t(value));
}
bool legalBoundaryOperands(NativeFacts &facts,const std::vector<ss::Action> &actions,std::string &why) {
    DominanceInfo dominance(facts.function);
    const ss::LoopOrdinal body{facts.bodyBase(),facts.induction.step};
    for(const auto &a:actions) {
        if(a.anchor>=facts.roles.size()) {why="invalid native occurrence anchor";return false;}
        if(a.distanceInIterations>uint64_t(INT64_MAX)||
           !facts.induction.distance(a.distanceInIterations)||!body.value(a.distanceInIterations)||
           !body.value(a.guardResidue)||!body.value(a.invocationGuardResidue)||
           !body.value(facts.model.atoms[a.anchor].residue)||!facts.induction.distance(facts.model.period)) {
            why="unrepresentable ordinal endpoint constant";return false;
        }
        if(a.invocation!=ss::Action::Local&&facts.wrappers.empty()) {
            why="invocation endpoint has no enclosing frame";return false;
        }
        bool needsUpper=facts.caseGuard||a.participation==ss::Action::IfBody||a.invocationBodyGuard;
        if(!needsUpper)continue;
        auto *anchor=facts.physical.phases[facts.origin[a.anchor]]->elementOp;
        if(!facts.loop||!dominance.properlyDominates(facts.loop.getUpperBound(),anchor)) {
            why="loop bound is unavailable at the selected early publication";return false;
        }
    }
    return true;
}
Value emitOrdinal(OpBuilder &builder,Location loc,const NativeFacts &facts) {
    // Used only inside the executing ordinary/tail body (and below the tail
    // phase guard). Its numerator is therefore nonnegative and representable.
    scf::ForOp body=facts.loop; // non-const handle copy; see NativeFacts::scalar
    Value value=body.getInductionVar();
    if(facts.bodyBase())value=builder.create<arith::SubIOp>(loc,value,indexConstant(builder,loc,uint64_t(facts.bodyBase())));
    if(facts.induction.step!=1)value=builder.create<arith::DivUIOp>(loc,value,indexConstant(builder,loc,uint64_t(facts.induction.step)));
    return value;
}
void emitNative(NativeFacts &facts,const std::vector<ss::Action> &actions,
                const std::vector<ss::EmissionSite> &sites) {
    // All groups have been checked before the clone is changed. Each two-member
    // group is ONE original-cut command whose initial and steady domains cover
    // exactly the original operation. It does not wait for a larger prefix.
    struct Command { ss::Action action; bool phaseFree; };
    std::map<std::pair<Operation*,bool>,std::vector<Command>> points;
    for (const auto &site:sites) {
        const auto &a=actions[site.members.front()];
        points[{facts.physical.phases[facts.origin[a.anchor]]->elementOp,a.after}].push_back({a,site.members.size()==2});
    }
    // Iterate original phases, never pointer ordering, for deterministic IR.
    for (unsigned p=0;p<facts.physical.phases.size();++p) for (bool after:{false,true}) {
        auto *anchor=facts.physical.phases[p]->elementOp;
        auto found=points.find({anchor,after}); if (found==points.end()) continue;
        OpBuilder outer(anchor);
        if (after) outer.setInsertionPointAfter(anchor);
        for (const auto &command:found->second) {
            const auto &a=command.action;
            OpBuilder::InsertionGuard restore(outer);
            OpBuilder builder=outer;
            auto guard=[&](Value c) {
                auto branch=builder.create<scf::IfOp>(anchor->getLoc(),c,false);
                builder.setInsertionPointToStart(&branch.getThenRegion().front());
            };
            if(facts.caseGuard) {
                auto lower=indexConstant(builder,anchor->getLoc(),uint64_t(facts.induction.lower));
                auto pred=facts.startupCase==StartupCase::Empty?arith::CmpIPredicate::sle:arith::CmpIPredicate::sgt;
                guard(builder.create<arith::CmpIOp>(anchor->getLoc(),pred,facts.loop.getUpperBound(),lower));
            }
            const auto role=facts.roles[a.anchor];
            if(!command.phaseFree&&(role==AtomRole::Initial||role==AtomRole::Steady)) {
                auto lower=indexConstant(builder,anchor->getLoc(),uint64_t(facts.induction.lower));
                auto pred=role==AtomRole::Initial?arith::CmpIPredicate::eq:arith::CmpIPredicate::ne;
                guard(builder.create<arith::CmpIOp>(anchor->getLoc(),pred,facts.loop.getInductionVar(),lower));
            }
            const ss::LoopOrdinal body{facts.bodyBase(),facts.induction.step};
            if(a.invocation!=ss::Action::Local) {
                Value predicate;
                for(auto frame:facts.wrappers) {
                    Value part;
                    if(a.invocation==ss::Action::ToNextInvocation) {
                        auto remaining=builder.create<arith::SubIOp>(anchor->getLoc(),frame.getUpperBound(),frame.getInductionVar());
                        part=builder.create<arith::CmpIOp>(anchor->getLoc(),arith::CmpIPredicate::sgt,remaining,frame.getStep());
                    } else {
                        part=builder.create<arith::CmpIOp>(anchor->getLoc(),arith::CmpIPredicate::ne,
                            frame.getInductionVar(),frame.getLowerBound());
                    }
                    if(predicate)predicate=builder.create<arith::OrIOp>(anchor->getLoc(),predicate,part);
                    else predicate=part;
                }
                guard(predicate);
            }
            if(a.invocationBodyGuard) {
                auto r=indexConstant(builder,anchor->getLoc(),uint64_t(*body.value(a.invocationGuardResidue)));
                guard(builder.create<arith::CmpIOp>(anchor->getLoc(),arith::CmpIPredicate::sgt,
                    facts.loop.getUpperBound(),r));
            }
            if ((role==AtomRole::Ordinary||role==AtomRole::Steady) && facts.residues[p].size()>1) {
                auto d=indexConstant(builder,anchor->getLoc(),facts.model.period);
                auto r=builder.create<arith::RemUIOp>(anchor->getLoc(),emitOrdinal(builder,anchor->getLoc(),facts),d);
                auto c=indexConstant(builder,anchor->getLoc(),facts.model.atoms[a.anchor].residue);
                guard(builder.create<arith::CmpIOp>(anchor->getLoc(),arith::CmpIPredicate::eq,r,c));
            }
            if (a.participation==ss::Action::IfBody) {
                auto r=indexConstant(builder,anchor->getLoc(),uint64_t(*body.value(a.guardResidue)));
                guard(builder.create<arith::CmpIOp>(anchor->getLoc(),arith::CmpIPredicate::sgt,
                    facts.loop.getUpperBound(),r));
            } else if (a.participation==ss::Action::First) {
                auto r=indexConstant(builder,anchor->getLoc(),uint64_t(*body.value(facts.model.atoms[a.anchor].residue)));
                guard(builder.create<arith::CmpIOp>(anchor->getLoc(),arith::CmpIPredicate::eq,
                    facts.loop.getInductionVar(),r));
            } else if (a.participation==ss::Action::Last) {
                // In an executing body: 0 <= iv < upper <= INDEX_MAX.
                // This subtraction is representable even near INDEX_MAX.
                auto left=builder.create<arith::SubIOp>(anchor->getLoc(),facts.loop.getUpperBound(),facts.loop.getInductionVar());
                auto period=indexConstant(builder,anchor->getLoc(),uint64_t(*facts.induction.distance(facts.model.period)));
                guard(builder.create<arith::CmpIOp>(anchor->getLoc(),arith::CmpIPredicate::sle,left,period));
            }
            if (a.distanceInIterations) {
                auto amount=a.kind==ss::Action::Set?facts.induction.distance(a.distanceInIterations):body.value(a.distanceInIterations);
                auto c=indexConstant(builder,anchor->getLoc(),uint64_t(*amount));
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
    enum GuardKind { Residue, OrdinalResidue, Previous, Next, First, Last, IfBody,
                     InvocationPrevious, InvocationNext, InitialPhase, SteadyPhase, EmptyCase, NonemptyCase };
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
                    yes=cmp.getPredicate()==arith::CmpIPredicate::sgt && cmp.getRhs()==loop.getStep() &&
                        sub && sub.getLhs()==loop.getUpperBound() && sub.getRhs()==loop.getInductionVar();
                } else yes=cmp.getPredicate()==arith::CmpIPredicate::ne &&
                    cmp.getRhs()==loop.getLowerBound() && cmp.getLhs()==loop.getInductionVar();
                if(yes) {
                    if(!frames.insert(loop.getOperation()).second)return false;
                    matched=true;break;
                }
            }
            if(!matched)return false;
        }
        return frames.size()==facts.wrappers.size();
    }
    bool isOrdinal(Value value) const {
        if(facts.induction.step!=1) {
            auto div=value.getDefiningOp<arith::DivUIOp>();
            if(!div||literal(div.getRhs())!=std::optional<int64_t>(facts.induction.step))return false;
            value=div.getLhs();
        }
        if(facts.bodyBase()) {
            auto sub=value.getDefiningOp<arith::SubIOp>();
            if(!sub||literal(sub.getRhs())!=std::optional<int64_t>(facts.bodyBase()))return false;
            value=sub.getLhs();
        }
        return value==facts.loop.getInductionVar();
    }
    std::optional<Condition> condition(Value v) {
        if(invocationCondition(v,true)){allowExpression(v);return Condition{InvocationNext,v,0};}
        if(invocationCondition(v,false)){allowExpression(v);return Condition{InvocationPrevious,v,0};}
        if(!facts.loop) {
            if(literal(v)){allowExpression(v);return Condition{Residue,v,0};}
            return {};
        }
        const ss::LoopOrdinal body{facts.bodyBase(),facts.induction.step};
        if(auto cmp=v.getDefiningOp<arith::CmpIOp>()) {
            auto c=literal(cmp.getRhs());
            if(c) {
                if(facts.caseGuard&&cmp.getLhs()==facts.loop.getUpperBound()&&*c==facts.induction.lower) {
                    if(cmp.getPredicate()==arith::CmpIPredicate::sle) {
                        allowExpression(v);return Condition{EmptyCase,v,0};
                    }
                    if(cmp.getPredicate()==arith::CmpIPredicate::sgt) {
                        allowExpression(v);return Condition{NonemptyCase,v,0};
                    }
                }
                if(facts.startup&&cmp.getLhs()==facts.loop.getInductionVar()&&*c==facts.induction.lower) {
                    if(cmp.getPredicate()==arith::CmpIPredicate::eq) {
                        allowExpression(v);return Condition{InitialPhase,v,0};
                    }
                    if(cmp.getPredicate()==arith::CmpIPredicate::ne) {
                        allowExpression(v);return Condition{SteadyPhase,v,0};
                    }
                }
                auto ordinal=body.index(*c);
                if(ordinal) {
                    if(cmp.getPredicate()==arith::CmpIPredicate::eq&&cmp.getLhs()==facts.loop.getInductionVar()) {
                        allowExpression(v);return Condition{First,v,*ordinal};
                    }
                    if(cmp.getPredicate()==arith::CmpIPredicate::sgt&&cmp.getLhs()==facts.loop.getUpperBound()) {
                        allowExpression(v);return Condition{IfBody,v,*ordinal};
                    }
                    if(*ordinal&&cmp.getPredicate()==arith::CmpIPredicate::sge&&cmp.getLhs()==facts.loop.getInductionVar()) {
                        allowExpression(v);return Condition{Previous,v,*ordinal};
                    }
                }
                if(auto sub=cmp.getLhs().getDefiningOp<arith::SubIOp>()) {
                    if(sub.getLhs()==facts.loop.getUpperBound()&&sub.getRhs()==facts.loop.getInductionVar()&&
                       *c>0&&uint64_t(*c)%uint64_t(facts.induction.step)==0) {
                        uint64_t distance=uint64_t(*c)/uint64_t(facts.induction.step);
                        if(cmp.getPredicate()==arith::CmpIPredicate::sle&&distance==facts.model.period) {
                            allowExpression(v);return Condition{Last,v,distance};
                        }
                        if(cmp.getPredicate()==arith::CmpIPredicate::sgt) {
                            allowExpression(v);return Condition{Next,v,distance};
                        }
                    }
                }
                if(*c>=0&&uint64_t(*c)<facts.model.period&&cmp.getPredicate()==arith::CmpIPredicate::eq) {
                    auto rem=cmp.getLhs().getDefiningOp<arith::RemUIOp>();
                    if(rem&&literal(rem.getRhs())==std::optional<int64_t>(int64_t(facts.model.period))&&isOrdinal(rem.getLhs())) {
                        allowExpression(v);return Condition{OrdinalResidue,v,uint64_t(*c)};
                    }
                }
            }
        }
        auto scalar=facts.scalar();auto period=scalar.period(v);
        if(!period||facts.model.period%*period)return {};
        allowExpression(v);return Condition{Residue,v,0};
    }
    bool generated(Operation *op,Operation *previous,Operation *next,std::vector<Condition> guards) {
        if (auto branch=dyn_cast<scf::IfOp>(op)) {
            if (branch.getNumResults() || !llvm::hasSingleElement(branch.getThenRegion()))
                return fail("generated guard has an unsupported region/result");
            if (!branch.getElseRegion().empty()) for (Operation &x:branch.getElseRegion().front())
                if (!isa<scf::YieldOp>(x) || x.getNumOperands()) return fail("generated else path has actions");
            auto c=condition(branch.getCondition());
            if (!c) return fail("generated predicate is not a qualified original-bound/residue expression");
            if((c->kind==EmptyCase&&facts.startupCase!=StartupCase::Empty)||
               (c->kind==NonemptyCase&&facts.startupCase!=StartupCase::Nonempty))return true;
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
        if (isa<arith::ConstantOp,arith::CmpIOp,arith::SubIOp,arith::DivUIOp,arith::RemUIOp,arith::OrIOp>(op)) return true;
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
                Operation *lifetime=facts.inventory.physical.lifetimeScope;
                Block *retirementBlock=nullptr;Operation *next=nullptr;
                if(lifetime==facts.function.getOperation()) {
                    next=facts.function.getBody().front().getTerminator();
                    retirementBlock=next->getBlock();
                } else if(lifetime&&llvm::hasSingleElement(lifetime->getRegion(0))) {
                    retirementBlock=&lifetime->getRegion(0).front();
                }
                if (!retirementBlock||!guards.empty()||op->getBlock()!=retirementBlock||
                    op->getNextNode()!=next||
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
        std::optional<Condition> participation,invocation,existence,phase,caseDomain;
        for(const auto &g:guards) {
            if(g.kind==Residue||g.kind==OrdinalResidue)continue;
            if(g.kind==EmptyCase||g.kind==NonemptyCase) {
                if(caseDomain)return fail("duplicate startup case predicate");
                caseDomain=g;continue;
            }
            if(g.kind==InitialPhase||g.kind==SteadyPhase) {
                if(phase)return fail("duplicate initial/steady phase predicate");
                phase=g;continue;
            }
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
        if(facts.caseGuard&&!caseDomain)return fail("missing original-loop empty/nonempty case predicate");
        const bool originalBody=facts.segments[id->second]==ss::Segment::Body&&bool(facts.loop);
        if(facts.startup&&originalBody&&!phase) {
            unsigned initial=0,steady=0;
            for(std::size_t atom=0;atom<facts.origin.size();++atom)if(facts.origin[atom]==id->second) {
                initial+=facts.roles[atom]==AtomRole::Initial;
                steady+=facts.roles[atom]==AtomRole::Steady;
            }
            // Missing phase guards are legal ONLY for an exhaustive pair of
            // simple disjoint domains of this original operation. The expanded
            // actual actions still undergo full verification and exact cut/FIFO
            // comparison below. A dropped singleton/first/last guard is refused.
            if(!facts.wrappers.empty()||facts.model.period!=1||initial!=1||steady!=1||
               participation||invocation||existence||
               llvm::any_of(guards,[](const Condition &g){return g.kind==Residue||g.kind==OrdinalResidue;}))
                return fail("missing initial/steady occurrence predicate outside exact startup union");
        }
        if(phase&&(!facts.startup||!originalBody))return fail("initial/steady predicate has no body occurrence");
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
        auto recurring=facts.scalar(),initial=facts.scalar(true);
        for (std::size_t atom=0;atom<facts.origin.size();++atom) if (facts.origin[atom]==id->second) {
            const auto &represented=facts.model.atoms[atom];
            if(phase&&((phase->kind==InitialPhase)!=(facts.roles[atom]==AtomRole::Initial)))continue;
            auto &scalar=facts.roles[atom]==AtomRole::Initial?initial:recurring;
            if (participation && (participation->kind==First || participation->kind==Last ||
                                 participation->kind==Previous || participation->kind==Next) &&
                represented.segment!=ss::Segment::Body)
                return fail("loop-local endpoint guard outside the body");
            bool active=true;
            for (const auto &g:guards) {
                if(g.kind==OrdinalResidue) {
                    if(represented.segment!=ss::Segment::Body||facts.roles[atom]==AtomRole::Outside)
                        return fail("ordinal residue outside the periodic tail");
                    active&=represented.residue==g.distance;
                } else if(g.kind==Residue) {
                    auto v=scalar.evaluate(g.value,represented.residue);
                    if(!v)return fail("reconstructed residue predicate is unavailable");
                    active&=bool(*v);
                }
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
        const unsigned expectedRetirement=scopeIsLifetime()?1:0;
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
    bool scopeIsLifetime() const {
        Operation *lifetime=facts.inventory.physical.lifetimeScope;
        Region *region=lifetime==facts.function.getOperation()?&facts.function.getBody():
            (lifetime?&lifetime->getRegion(0):nullptr);
        return facts.scope==region;
    }

};

// Explicit diagnostic mode only. The JSON contains original physical facts,
// selected logical episodes and emitted site membership, never coverage receipts
// consumed by the verifier. uint64 geometry is written as decimal strings.
void describeStructuredPlan(const std::vector<std::unique_ptr<NativeFacts>> &units,
    const std::vector<std::vector<ss::Action>> &actions,
    const std::vector<std::vector<ss::EmissionSite>> &sites,
    ArrayRef<SequenceBridge> sequenceBridges) {
    if(!std::getenv("PTOAS_STRUCTURED_PLAN_JSON"))return;
    using llvm::json::Array;using llvm::json::Object;
    auto lane=[](ss::Lane l) {
        static const char *names[]={"S","V","M","MTE1","MTE2","MTE3","FIX"};
        return Object{{"core",l.core==ss::Core::AIC?"AIC":"AIV"},{"pipe",names[unsigned(l.pipe)]}};
    };
    auto memory=[](const auto &effects) {
        Array out;
        for(const auto *e:effects) {
            Array bases;for(auto b:e->baseAddresses)bases.emplace_back(std::to_string(b));
            out.emplace_back(Object{{"space",int64_t(e->scope)},{"bases",std::move(bases)},
                {"bytes",std::to_string(e->allocateSize)},{"known_physical",e->hasKnownPhysicalAddresses},
                {"unknown_range",e->aliasesUnknownRange},{"footprint_kind","conservative-may-access"}});
        }
        return out;
    };
    Array all;uint64_t staticSites=1,intrinsicProofs=0;
    for(std::size_t unit=0;unit<units.size();++unit) {
        const auto &f=*units[unit];Array atoms,requirements,commands,groups,carried,audit;
        static const char *roleNames[]={"outside","ordinary","initial","steady"};
        static const char *participationNames[]={"every","first","last","if-body"};
        static const char *propertyNames[]={"completion","acc-resource","visibility","accumulator-update"};
        for(std::size_t i=0;i<f.model.atoms.size();++i) {
            const auto &a=f.model.atoms[i];const auto *p=f.physical.phases[f.origin[i]];
            atoms.emplace_back(Object{{"atom",int64_t(i)},{"original_phase",int64_t(f.origin[i])},
                {"operation",p->elementOp->getName().getStringRef().str()},{"lane",lane(a.lane)},
                {"segment",int64_t(a.segment)},{"role",roleNames[unsigned(f.roles[i])]},
                {"residue",std::to_string(a.residue)},
                {"matrix_lowering",Object{{"kind",int64_t(a.matrix.kind)},
                    {"m",std::to_string(a.matrix.m)},{"n",std::to_string(a.matrix.n)},
                    {"k",std::to_string(a.matrix.k)},
                    {"accumulator_base",std::to_string(a.matrix.accumulatorBase)},
                    {"accumulator_bytes",std::to_string(a.matrix.accumulatorBytes)}}},
                {"reads",memory(p->useVec)},{"writes",memory(p->defVec)}});
        }
        for(std::size_t i=0;i<f.model.requirements.size();++i) {
            const auto &r=f.model.requirements[i];
            const bool intrinsic=ss::intrinsicAccumulatorOrder(f.model,r);
            intrinsicProofs+=intrinsic;
            requirements.emplace_back(Object{{"intrinsic_accumulator_order",intrinsic},{"requirement",int64_t(i)},{"source",int64_t(r.source)},
                {"target",int64_t(r.target)},{"epoch_distance",std::to_string(r.distance)},
                {"property",propertyNames[unsigned(r.property)]},
                {"storage_witness",r.hasStorageWitness?Object{{"base",std::to_string(r.storageBase)},
                    {"bytes",std::to_string(r.storageBytes)}}:Object{}}});
        }
        for(const auto &r:f.carried)carried.emplace_back(Object{{"source",int64_t(r.source)},
            {"target",int64_t(r.target)},{"property",propertyNames[unsigned(r.property)]}});
        for(std::size_t i=0;i<actions[unit].size();++i) {
            const auto &a=actions[unit][i];static const char *kinds[]={"set","wait","barrier"};
            commands.emplace_back(Object{{"action",int64_t(i)},{"kind",kinds[unsigned(a.kind)]},
                {"anchor",int64_t(a.anchor)},{"after",a.after},{"source_lane",lane(a.source)},
                {"target_lane",lane(a.target)},{"key",int64_t(a.key)},
                {"planner_order",std::to_string(a.order)},{"participation",participationNames[unsigned(a.participation)]},
                {"ordinal_distance",std::to_string(a.distanceInIterations)},
                {"existence_residue",std::to_string(a.guardResidue)},{"invocation",int64_t(a.invocation)},
                {"invocation_body_guard",a.invocationBodyGuard},{"invocation_residue",std::to_string(a.invocationGuardResidue)}});
        }
        for(std::size_t i=0;i<sites[unit].size();++i) {
            Array members;for(auto a:sites[unit][i].members)members.emplace_back(int64_t(a));
            groups.emplace_back(Object{{"site",int64_t(i)},{"actions",std::move(members)}});
        }
        // Optional and intentionally outside benchmark timing mode. Keep
        // payload-completion proof loss distinct from event-state proof loss.
        if(f.wrappers.empty()) {
            auto checked=ss::verify(f.model,actions[unit]);
            auto details=ss::auditRegionHandoffs(f.model,checked.plan);
            if(details)for(const auto &a:*details) {
                const auto &h=checked.plan.handoffs[a.handoff];Array lost;
                for(auto r:a.completionLost)lost.emplace_back(int64_t(r));
                audit.emplace_back(Object{{"source",int64_t(h.source)},{"target",int64_t(h.target)},
                    {"key",int64_t(h.key)},{"epoch_distance",std::to_string(h.distance)},
                    {"completion_unproved_without",std::move(lost)},
                    {"protocol_proved_without",a.protocolWithout},{"protocol_reason",a.protocolReason}});
            }
        }
        staticSites+=sites[unit].size();
        all.emplace_back(Object{{"unit",int64_t(unit)},{"period",std::to_string(f.model.period)},
            {"startup",f.startup},{"startup_case",int64_t(f.startupCase)},{"case_guard",f.caseGuard},
            {"lower",std::to_string(f.induction.lower)},{"step",std::to_string(f.induction.step)},
            {"ordinal_offset",std::to_string(f.ordinalOffset)},{"wrapper_count",int64_t(f.wrappers.size())},
            {"gm_contract_enum",int64_t(f.gm)},
            {"hardware_contract",f.model.target.hardware==ss::HardwareContract::Conservative?"conservative":"a2a3-mmad-acc-v1"},
            {"atoms",std::move(atoms)},
            {"requirements",std::move(requirements)},{"carried_requirements",std::move(carried)},
            {"actions",std::move(commands)},{"sites",std::move(groups)},{"deletion_audit",std::move(audit)},
            {"audit_scope",f.wrappers.empty()?"complete-local-region":"reentrant-audit-not-run"}});
    }
    Array sequence;
    for(const auto &b:sequenceBridges) {
        sequence.emplace_back(Object{{"source_order",int64_t(b.sourceOrder)},
            {"target_order",int64_t(b.targetOrder)},{"source_lane",lane(b.source)},
            {"target_lane",lane(b.target)},{"kind",b.barrier?"barrier":"event"},
            {"key",int64_t(b.key)}});
        staticSites+=b.barrier?1:2;
    }
    Object root{{"schema","oahs.s7.plan.v2"},{"producer","structured"},{"units",std::move(all)},
        {"sequence_bridges",std::move(sequence)},
        {"generated_static_sites_including_retirement",std::to_string(staticSites)},
        {"retirement_sites",int64_t(1)},{"new_hardware_elision",intrinsicProofs!=0},
        {"intrinsic_accumulator_requirements",std::to_string(intrinsicProofs)}};
    llvm::errs()<<"OAHS_PLAN "<<llvm::json::Value(std::move(root))<<"\n";
}

Outcome run(func::FuncOp function,InsertSyncGMAliasMode gm,
            llvm::function_ref<void(func::FuncOp)> mutate, ss::HardwareContract hardware) {
    if (function.isDeclaration() || !llvm::hasSingleElement(function.getBody())) {
        Outcome out; out.reason="structured construction requires one function block"; return out;
    }
    if (hardware!=ss::HardwareContract::Conservative && hardware!=ss::HardwareContract::A2A3MmadAccV1) {
        Outcome out;out.reason="unknown structured hardware contract";return out;
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
    std::vector<std::vector<ss::EmissionSite>> emissionSites;
    uint64_t atoms=0,views=0,refinementTrials=0,refinementRemoved=0,rekeyTrials=0,rekeyed=0,coalesced=0;
    for(const auto &scope:layout.scopes) {
      std::vector<std::unique_ptr<NativeFacts>> cases;
      auto first=std::make_unique<NativeFacts>(inventory,scope,gm);
      if(!first->build()){out.reason=first->reason;return out;}
      bool needsEmpty=first->needsEmptyCase;cases.push_back(std::move(first));
      if(needsEmpty) {
          auto empty=std::make_unique<NativeFacts>(inventory,scope,gm,StartupCase::Empty);
          if(!empty->build()){out.reason=empty->reason;return out;}
          cases.push_back(std::move(empty));
      }
      for(auto &facts:cases) {
        facts->model.target.hardware=hardware;
        std::vector<ss::Action> actions;
        ss::Status status;std::string reason;
        if(facts->wrappers.empty()) {
            auto result=ss::construct(facts->model);
            if(result.status==ss::Status::AllocationFailure&&!facts->model.allowBoundaryKeyReuse) {
                // One alternative realization policy, never different payload
                // requirements or endpoints. Successfully allocated S1-S3
                // plans keep their old assignment. A mixed family is accepted
                // only by the causal continuation check, also run after emit.
                facts->model.allowBoundaryKeyReuse=true;
                result=ss::construct(facts->model);
            }
            if(result.status==ss::Status::Applied) {
                // This is a COMPLETE unwrapped region, including every
                // startup/exit/empty-body obligation. Never refine a projection
                // under a still-live S3 interface. No endpoint motion is allowed.
                result=ss::refineRegionHandoffs(facts->model,result.plan);
                refinementTrials+=result.refinementAttempts;refinementRemoved+=result.removedHandoffs;
                if(result.status==ss::Status::Applied && facts->startup && facts->model.period==1 &&
                   facts->startupCase==StartupCase::Nonempty) {
                    // Sharing is an explicit, fully checked optimization here,
                    // not only an allocation fallback after exhausting the pool.
                    // The verifier must use the same mixed-family contract.
                    facts->model.allowBoundaryKeyReuse=true;
                    result=ss::coalesceStartupHandoffs(facts->model,facts->siteOrigins(),result.plan);
                    rekeyTrials+=result.rekeyAttempts;rekeyed+=result.rekeyedHandoffs;coalesced+=result.coalescedSites;
                }
            }
            status=result.status;reason=result.reason;
            if(status==ss::Status::Applied)actions=ss::actionsForPlan(facts->model,result.plan);
        } else {
            auto result=ss::constructInvocations(facts->invocationModel());
            if(result.status==ss::Status::AllocationFailure&&!facts->model.allowBoundaryKeyReuse) {
                facts->model.allowBoundaryKeyReuse=true;
                result=ss::constructInvocations(facts->invocationModel());
            }
            status=result.status;reason=result.reason;
            if(status==ss::Status::Applied)actions=ss::actionsForInvocations(facts->invocationModel(),result.plan);
            views+=result.proofViews;
        }
        if(status!=ss::Status::Applied) {
            out.status=status==ss::Status::AllocationFailure?Outcome::AllocationFailure:
                status==ss::Status::InvalidPlan?Outcome::InternalError:Outcome::Unsupported;
            out.reason=reason;return out;
        }
        if(!legalBoundaryOperands(*facts,actions,out.reason))return out;
        auto sites=ss::startupEmissionSites(facts->model,facts->siteOrigins(),actions);
        if(!sites){out.status=Outcome::InternalError;out.reason="invalid startup emission-site population";return out;}
        // Grouping changes static sites, not the dynamic commands. Rebuild the
        // proposed per-domain order before emitting and check it independently.
        std::vector<ss::Action> grouped;
        for(std::size_t order=0;order<sites->size();++order)for(auto member:(*sites)[order].members) {
            auto a=actions[member];a.order=order;grouped.push_back(a);
        }
        if(llvm::any_of(*sites,[](const ss::EmissionSite &s){return s.members.size()==2;})) {
            auto groupStatus=ss::verify(facts->model,grouped).status;
            if(!facts->wrappers.empty()||groupStatus!=ss::Status::Applied) {
                out.status=Outcome::InternalError;out.reason="grouped emission changes synchronization protocol";return out;
            }
        }
        emissionSites.push_back(std::move(*sites));
        atoms+=facts->model.atoms.size();selected.push_back(std::move(actions));units.push_back(std::move(facts));
      }
    }
    auto sequenceRequirements=deriveSequenceRequirements(
        inventory.function,inventory.physical,layout,gm,hardware);
    auto sequencePlan=buildSequenceBridges(sequenceRequirements,hardware);
    if(sequencePlan.status!=ss::Status::Applied) {
        out.status=sequencePlan.status==ss::Status::AllocationFailure?Outcome::AllocationFailure:Outcome::Unsupported;
        out.reason=sequencePlan.reason;return out;
    }
    SyncPayloadSnapshot snapshot(working);llvm::SmallPtrSet<Operation*,32> original;
    working.walk([&](Operation *op){original.insert(op);});
    for(std::size_t i=0;i<units.size();++i)emitNative(*units[i],selected[i],emissionSites[i]);
    auto emittedSequence=emitSequenceBridges(working.getContext(),sequencePlan.bridges);
    OpBuilder retirementBuilder(working.getContext());Location retirementLoc=working.getLoc();
    if(inventory.physical.lifetimeScope==working.getOperation()) {
        auto *ret=working.getBody().front().getTerminator();
        retirementBuilder.setInsertionPoint(ret);retirementLoc=ret->getLoc();
    } else {
        Region &body=inventory.physical.lifetimeScope->getRegion(0);
        if(!llvm::hasSingleElement(body)){out.status=Outcome::InternalError;out.reason="physical section became multi-block";return out;}
        retirementBuilder.setInsertionPointToEnd(&body.front());retirementLoc=inventory.physical.lifetimeScope->getLoc();
    }
    auto drain=retirementBuilder.create<BarrierOp>(retirementLoc,
        PipeAttr::get(working.getContext(),PIPE::PIPE_ALL));
    if(mutate)mutate(working);
    out.status=Outcome::InternalError;
    if(failed(mlir::verify(working))){out.reason="malformed structured emission";return out;}
    unsigned drains=0;working.walk([&](BarrierOp b){if(b.getPipe().getPipe()==PIPE::PIPE_ALL)++drains;});
    Block *retirementBlock=nullptr;Operation *retirementNext=nullptr;
    if(inventory.physical.lifetimeScope==working.getOperation()) {
        retirementNext=working.getBody().front().getTerminator();
        retirementBlock=retirementNext->getBlock();
    } else if(inventory.physical.lifetimeScope&&
              llvm::hasSingleElement(inventory.physical.lifetimeScope->getRegion(0))) {
        retirementBlock=&inventory.physical.lifetimeScope->getRegion(0).front();
    }
    if(!drain||drain.getPipe().getPipe()!=PIPE::PIPE_ALL||drains!=1||
       !retirementBlock||drain->getBlock()!=retirementBlock||
       drain->getNextNode()!=retirementNext||
       drain->hasAttr("pto.auto_sync_tail_barrier")||drain->hasAttr("pto.auto_sync_tail_hint")) {
        out.reason="one explicit unconditional physical-context retirement drain is required";return out;
    }
    llvm::SmallPtrSet<Operation*,32> allowed;allowed.insert(drain.getOperation());
    for(const auto &bridge:emittedSequence) {
        if(bridge.set)allowed.insert(bridge.set);
        if(bridge.wait)allowed.insert(bridge.wait);
        if(bridge.barrier)allowed.insert(bridge.barrier);
    }
    for(std::size_t i=0;i<units.size();++i) {
        auto result=Reconstruct(*units[i],original).run(selected[i],allowed);
        if(result.status!=Outcome::Applied)return result;
        out.requirements+=result.requirements;out.handoffs+=result.handoffs;out.barriers+=result.barriers;out.work+=result.work;
    }
    for(const auto &bridge:sequencePlan.bridges) {
        if(bridge.barrier)++out.barriers; else ++out.handoffs;
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
    SyncPhysicalFacts rebuiltPhysical=inventory.physical;
    rebuiltPhysical.phases.clear();
    for(auto *old:inventory.physical.phases) {
        rebuiltPhysical.phases.push_back(phases.at(old->elementOp));
    }
    auto rebuiltRequirements=deriveSequenceRequirements(
        working,rebuiltPhysical,layout,gm,hardware);
    if(rebuiltRequirements.status!=ss::Status::Applied) {
        out.reason=rebuiltRequirements.reason;return out;
    }
    if(!verifySequenceRequirements(sequenceRequirements.requirements,
                                   rebuiltRequirements.requirements,
                                   hardware,out.reason))return out;
    if(!verifySequenceBridges(emittedSequence,out.reason))return out;
    describeStructuredPlan(units,selected,emissionSites,sequencePlan.bridges);
    // S7: rederive the lowering premises from the ACTUAL emitted physical ops,
    // not a cached optimizer receipt. The snapshot separately preserves scalar
    // descriptor operands/attributes and controls. This is not a second hardware spec.
    for (const auto &unit:units) for (std::size_t id=0;id<unit->origin.size();++id) {
        auto *op=unit->physical.phases[unit->origin[id]]->elementOp;
        if (matrixLoweringFacts(*phases.at(op))!=unit->model.atoms[id].matrix) {
            out.reason="emitted matrix hardware premise changed";return out;
        }
    }
    out.status=Outcome::Applied;out.reason="structured local and nested invocation transfers verified";
    function.getBody().takeBody(working.getBody());
    if(std::getenv("PTOAS_LOGICAL_TRACE"))llvm::errs()<<"structured OAHS units "<<units.size()<<" atoms "<<atoms
        <<" hardware_contract "<<(hardware==ss::HardwareContract::Conservative?"conservative":"a2a3-mmad-acc-v1")
        <<" invocation_views "<<views<<" requirements "<<out.requirements<<" handoffs "<<out.handoffs
        <<" barriers "<<out.barriers<<" sequence_bridges "<<sequencePlan.bridges.size()<<" refinement_trials "<<refinementTrials
        <<" removed_handoffs "<<refinementRemoved<<" rekey_trials "<<rekeyTrials
        <<" rekeyed_handoffs "<<rekeyed<<" coalesced_sites "<<coalesced<<" seconds "
        <<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<"\n";
    return out;
}
} // namespace
Outcome mlir::pto::structured_sync::constructStructuredSync(
    func::FuncOp f,InsertSyncGMAliasMode gm,HardwareContract hardware) {
    return run(f,gm,{},hardware);
}
Outcome mlir::pto::structured_sync::testing::constructWithEmissionMutation(
    func::FuncOp f,InsertSyncGMAliasMode gm,llvm::function_ref<void(func::FuncOp)> mutate,
    HardwareContract hardware) {
    return run(f,gm,mutate,hardware);
}
