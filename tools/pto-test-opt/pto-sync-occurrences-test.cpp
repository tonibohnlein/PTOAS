// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
// See LICENSE for details. Provided AS IS, WITHOUT WARRANTIES OF ANY KIND.
#include "PTO/IR/PTO.h"
#include "LogicalSyncTestJson.h"
#include "PTO/Transforms/InsertSync/SyncOccurrences.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::pto::logical_sync;
using llvm::json::Array;
using llvm::json::Object;

int main(int argc,char** argv) {
    if (argc!=2) return 2;
    DialectRegistry registry;
    registry.insert<mlir::pto::PTODialect,func::FuncDialect,scf::SCFDialect,arith::ArithDialect>();
    MLIRContext context(registry, MLIRContext::Threading::DISABLED);
    context.allowUnregisteredDialects();
    auto module=parseSourceFile<ModuleOp>(argv[1],&context);
    if (!module) return 2;
    Array functions;
    for (auto f:module->getOps<func::FuncOp>()) {
        if (f.isDeclaration()) continue;
        SmallVector<Operation*> phases;
        f.walk([&](Operation* op) {
            if (op->getName().getStringRef()=="test.phase" ||
                op->getName().getStringRef().starts_with("pto.t")) phases.push_back(op);
        });
        if (f->hasAttr("test.reverse_phases")) std::reverse(phases.begin(),phases.end());
        auto facts=SyncOccurrences::build(f,phases);
        Object result{{"function",f.getSymName()},{"complete",facts.complete},{"reason",facts.reason}};
        if (facts.complete) {
            Array parameters, points, orders;
            for (Value value:facts.parameters) {
                Object binding;
                if (auto arg=dyn_cast<BlockArgument>(value)) binding["argument"]=arg.getArgNumber();
                else {
                    std::string text; llvm::raw_string_ostream stream(text); value.print(stream);
                    binding["ssa"]=text;
                }
                parameters.push_back(std::move(binding));
            }
            for (unsigned p=0;p<facts.points.size();++p) {
                points.push_back(testing::encode(facts.domain(p)));
                for (unsigned q=0;q<facts.points.size();++q) {
                    auto relation=facts.ordered(p,q);
                    if (!relation) return 3;
                    orders.push_back(
                        Object{{"source", p}, {"target", q}, {"relation", testing::encode(*relation.relation)}});
                }
            }
            result["parameters"]=std::move(parameters);
            result["points"]=std::move(points);
            result["orders"]=std::move(orders);
        }
        functions.push_back(std::move(result));
    }
    llvm::outs()<<llvm::formatv("{0:2}\n",llvm::json::Value(std::move(functions)));
    return 0;
}
