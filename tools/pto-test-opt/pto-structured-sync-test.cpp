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
#include "llvm/ADT/STLExtras.h"
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
    if (!arch) module->getOperation()->setAttr("pto.target_arch",StringAttr::get(&context,"a3"));
    auto function=*module->getOps<func::FuncOp>().begin();
    const auto before=printed(function); StringRef mode(argv[2]); bool changed=false;
    auto mutate=[&](func::FuncOp working) {
        if (mode=="none" || mode=="expect-unsupported") return;
        Operation *chosen=nullptr;
        working.walk([&](Operation *op) {
            if (chosen) return;
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
        if (mode=="drop-wait" || mode=="drop-set" || mode=="drop-retirement") { chosen->erase(); changed=true; }
        else if (mode=="duplicate-set") { OpBuilder b(chosen); b.setInsertionPointAfter(chosen); b.clone(*chosen); changed=true; }
        else if (mode=="wrong-key") {
            auto w=cast<WaitFlagOp>(chosen); OpBuilder b(w);
            b.create<WaitFlagOp>(w.getLoc(),w.getSrcPipe(),w.getDstPipe(),
                EventAttr::get(&context,static_cast<EVENT>((unsigned(w.getEventId().getEvent())+1)%6)));
            w.erase(); changed=true;
        } else if (mode=="wrong-participation") {
            auto cmp=dyn_cast<arith::CmpIOp>(chosen);
            if (!cmp) return;
            OpBuilder b(cmp); auto zero=b.create<arith::ConstantIndexOp>(cmp.getLoc(),0);
            cmp->setOperand(1,zero); changed=true;
        } else if (mode=="late-set") {
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
