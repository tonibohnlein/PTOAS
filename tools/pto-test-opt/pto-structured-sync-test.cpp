// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.


#include "PTO/IR/PTO.h"
#include "PTO/Transforms/InsertSync/StructuredSyncPlan.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
using namespace mlir::pto;
static std::string printed(Operation *op) {
    std::string text; llvm::raw_string_ostream stream(text); op->print(stream); return text;
}
int main(int argc,char **argv) {
    if (argc!=4) { llvm::errs()<<"input.pto mutation output.pto\n"; return 2; }
    DialectRegistry registry;
    registry.insert<PTODialect,func::FuncDialect,scf::SCFDialect,arith::ArithDialect,DLTIDialect>();
    MLIRContext context(registry,MLIRContext::Threading::DISABLED);
    auto module=parseSourceFile<ModuleOp>(argv[1],&context);
    if (!module || !llvm::hasSingleElement(module->getOps<func::FuncOp>())) return 2;
    auto arch=module->getOperation()->getAttrOfType<StringAttr>("pto.target_arch");
    if (!arch || arch.getValue()=="a2a3") module->getOperation()->setAttr("pto.target_arch",StringAttr::get(&context,"a3"));
    auto function=*module->getOps<func::FuncOp>().begin();
    const auto before=printed(function); StringRef mode(argv[2]); bool changed=false;
    auto mutate=[&](func::FuncOp working) {
        if (mode=="none" || mode=="expect-unsupported") return;
        Operation *chosen=nullptr;
        auto syncOnly=[](scf::IfOp branch) {
            bool event=false,payload=false;
            branch.walk([&](Operation *x) {
                event|=isa<SetFlagOp,WaitFlagOp,BarrierOp>(x);
                payload|=isa<OpPipeInterface>(x)&&!isa<SetFlagOp,WaitFlagOp,BarrierOp>(x);
            });
            return event&&!payload;
        };
        working.walk([&](Operation *op) {
            if (chosen) return;
            if(mode=="narrow-coalesced"||mode=="duplicate-coalesced"||mode=="late-coalesced-set") {
                // A shared startup publication is directly in its original
                // loop body, not in an initial/steady generated guard. This
                // fixture's nonzero first/steady population is checked by S6.
                if(isa<SetFlagOp>(op)&&isa<scf::ForOp>(op->getParentOp()))chosen=op;
            }
            if (mode=="late-boundary-set" || mode=="late-nested-set") {
                if (auto branch=dyn_cast<scf::IfOp>(op)) {
                    if (mode=="late-boundary-set") {
                        if (op->getParentOp()==working.getOperation() &&
                            llvm::any_of(branch.getThenRegion().front(),[](Operation &x){return isa<SetFlagOp>(x);}))
                            chosen=op;
                    } else {
                        // The original next payload, not another generated
                        // action, determines an independently later cut.
                        bool hasSet=false;
                        branch.walk([&](SetFlagOp){hasSet=true;});
                        if (hasSet) for (Operation *next=op->getNextNode();next;next=next->getNextNode())
                            if (isa<TLoadOp>(next)) {chosen=op;break;}
                    }
                }
            }
            if (mode=="wrong-startup" || mode=="wrong-empty-case") {
                if (auto branch=dyn_cast<scf::IfOp>(op)) {
                    auto cmp=branch.getCondition().getDefiningOp<arith::CmpIOp>();
                    if(cmp&&syncOnly(branch)) {
                        auto iv=dyn_cast<BlockArgument>(cmp.getLhs());
                        const bool phase=iv&&isa<scf::ForOp>(iv.getOwner()->getParentOp())&&
                            (cmp.getPredicate()==arith::CmpIPredicate::eq||cmp.getPredicate()==arith::CmpIPredicate::ne);
                        const bool empty=cmp.getPredicate()==arith::CmpIPredicate::sle&&
                            !cmp.getLhs().getDefiningOp<arith::SubIOp>();
                        if((mode=="wrong-startup"&&phase)||(mode=="wrong-empty-case"&&empty))chosen=cmp.getOperation();
                    }
                }
            }
            if(mode=="wrong-ordinal") {
                if(auto branch=dyn_cast<scf::IfOp>(op)) if(syncOnly(branch)) {
                    // Follow only this generated condition's SSA ancestry.
                    SmallVector<Value> pending{branch.getCondition()};
                    llvm::SmallPtrSet<Operation*,16> visited;
                    while(!pending.empty()&&!chosen) {
                        Value v=pending.pop_back_val();auto *definition=v.getDefiningOp();
                        if(!definition||!visited.insert(definition).second)continue;
                        if(isa<arith::DivUIOp>(definition)){chosen=definition;break;}
                        if(definition->getName().getDialectNamespace()=="arith")
                            for(Value operand:definition->getOperands())pending.push_back(operand);
                    }
                }
            }
            if (mode=="wrong-invocation") {
                if (auto cmp=dyn_cast<arith::CmpIOp>(op)) {
                    auto iv=dyn_cast<BlockArgument>(cmp.getLhs());
                    if (iv && isa<scf::ForOp>(iv.getOwner()->getParentOp()) &&
                        cmp.getPredicate()==arith::CmpIPredicate::ne) chosen=op;
                }
            }
            if (mode=="wrong-invocation-frame") {
                if (auto either=dyn_cast<arith::OrIOp>(op)) {
                    // The multi-frame publication/consumption predicate uses
                    // OR, not an iteration-product counter. Remove one frame
                    // while retaining a well-formed scalar expression.
                    if (llvm::any_of(either.getResult().getUsers(),[](Operation *user){return isa<scf::IfOp>(user);}))
                        chosen=op;
                }
            }
            if (mode=="wrong-first" || mode=="wrong-last" || mode=="wrong-existence") {
                if (auto cmp=dyn_cast<arith::CmpIOp>(op)) {
                    auto iv=dyn_cast<BlockArgument>(cmp.getLhs());
                    bool first=cmp.getPredicate()==arith::CmpIPredicate::eq && iv &&
                        isa<scf::ForOp>(iv.getOwner()->getParentOp());
                    bool last=cmp.getPredicate()==arith::CmpIPredicate::sle &&
                        bool(cmp.getLhs().getDefiningOp<arith::SubIOp>());
                    bool existence=cmp.getPredicate()==arith::CmpIPredicate::sgt &&
                        !cmp.getLhs().getDefiningOp<arith::SubIOp>();
                    if ((mode=="wrong-first"&&first)||(mode=="wrong-last"&&last)||
                        (mode=="wrong-existence"&&existence)) chosen=op;
                }
            }
            if ((mode=="drop-wait" || mode=="wrong-key") && isa<WaitFlagOp>(op)) chosen=op;
            if ((mode=="drop-set" || mode=="duplicate-set" || mode=="late-set") && isa<SetFlagOp>(op)) chosen=op;
            if (mode=="drop-retirement")
                if (auto b=dyn_cast<BarrierOp>(op)) if(b.getPipe().getPipe()==PIPE::PIPE_ALL) chosen=op;
            if (mode=="wrong-participation") {
                // The ordinary fixtures' ORIGINAL guards are selector equality.
                // Mutate only a new IV>=distance or remaining>distance guard,
                // not original payload control (which the snapshot rejects).
                if (auto b=dyn_cast<scf::IfOp>(op)) {
                    auto c=b.getCondition().getDefiningOp<arith::CmpIOp>();
                    bool participation=c && (c.getPredicate()==arith::CmpIPredicate::sge ||
                        (c.getPredicate()==arith::CmpIPredicate::sgt &&
                         c.getLhs().getDefiningOp<arith::SubIOp>()));
                    if (participation && llvm::any_of(b.getThenRegion().front(),
                        [](Operation &x){return isa<SetFlagOp,WaitFlagOp>(x);})) chosen=c.getOperation();
                }
            }
        });
        if (!chosen) return;
        if(mode=="narrow-coalesced") {
            auto loop=cast<scf::ForOp>(chosen->getParentOp());OpBuilder b(chosen);
            auto first=b.create<arith::CmpIOp>(chosen->getLoc(),arith::CmpIPredicate::eq,
                loop.getInductionVar(),loop.getLowerBound());
            auto branch=b.create<scf::IfOp>(chosen->getLoc(),first,false);
            chosen->moveBefore(&branch.getThenRegion().front(),branch.getThenRegion().front().begin());changed=true;
        }
        else if (mode=="drop-wait" || mode=="drop-set" || mode=="drop-retirement") { chosen->erase(); changed=true; }
        else if (mode=="duplicate-set" || mode=="duplicate-coalesced") { OpBuilder b(chosen); b.setInsertionPointAfter(chosen); b.clone(*chosen); changed=true; }
        else if (mode=="wrong-key") {
            auto w=cast<WaitFlagOp>(chosen); OpBuilder b(w);
            b.create<WaitFlagOp>(w.getLoc(),w.getSrcPipe(),w.getDstPipe(),
                EventAttr::get(&context,static_cast<EVENT>((unsigned(w.getEventId().getEvent())+1)%6)));
            w.erase(); changed=true;
        } else if(mode=="wrong-startup"||mode=="wrong-empty-case"||mode=="wrong-ordinal") {
            IntegerAttr value;
            if(chosen->getNumOperands()!=2||!matchPattern(chosen->getOperand(1),m_Constant(&value))||
               !value.getValue().isSignedIntN(63))return;
            OpBuilder b(chosen);
            auto wrong=b.create<arith::ConstantIndexOp>(chosen->getLoc(),value.getValue().getSExtValue()+1);
            chosen->setOperand(1,wrong);changed=true;
        } else if (mode=="wrong-participation") {
            auto cmp=dyn_cast<arith::CmpIOp>(chosen);
            if (!cmp) return;
            OpBuilder b(cmp); auto zero=b.create<arith::ConstantIndexOp>(cmp.getLoc(),0);
            cmp->setOperand(1,zero); changed=true;
        } else if (mode=="wrong-invocation-frame") {
            auto either=cast<arith::OrIOp>(chosen);
            either.getResult().replaceAllUsesWith(either.getLhs());
            either.erase();changed=true;
        } else if (mode=="wrong-first" || mode=="wrong-last" || mode=="wrong-existence" || mode=="wrong-invocation") {
            auto cmp=cast<arith::CmpIOp>(chosen);IntegerAttr value;
            if (!matchPattern(cmp.getRhs(),m_Constant(&value)) || !value.getValue().isSignedIntN(63)) return;
            OpBuilder b(cmp);auto wrong=b.create<arith::ConstantIndexOp>(cmp.getLoc(),value.getValue().getSExtValue()+1);
            cmp->setOperand(1,wrong);changed=true;
        } else if (mode=="late-set" || mode=="late-boundary-set" || mode=="late-nested-set" || mode=="late-coalesced-set") {
            for(Operation *next=chosen->getNextNode();next;next=next->getNextNode())
                if(isa<TLoadOp>(next)) { chosen->moveAfter(next); changed=true; break; }
        }
    };
    auto result=structured_sync::testing::constructWithEmissionMutation(
        function,InsertSyncGMAliasMode::DisjointArguments,mutate);
    using Result=logical_sync::ConstructionResult;
    bool applied=result.status==Result::Applied;
    bool expected=(mode=="none")?applied:
        mode=="expect-unsupported"?(!applied && result.status!=Result::InternalError):
        (changed && !applied);
    bool preserved=applied || printed(function)==before;
    std::error_code error;
    llvm::raw_fd_ostream output(argv[3],error,llvm::sys::fs::OF_Text);
    if(error) { llvm::errs()<<error.message(); return 2; }
    module->print(output);
    llvm::outs()<<llvm::json::Value(llvm::json::Object{
        {"accepted",applied},{"mutation_applied",changed},{"expected",expected},{"atomic",preserved},
        {"status",unsigned(result.status)},{"reason",result.reason},
        {"requirements",result.requirements},{"handoffs",result.handoffs},{"barriers",result.barriers},
        {"work_statistic",result.work}})<<"\n";
    return expected&&preserved?0:1;
}
