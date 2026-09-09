// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
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
        SmallVector<Operation*> phases, scalarQueries;
        SmallVector<Value> requestedScalars;
        f.walk([&](Operation* op) {
            if (op->getName().getStringRef()=="test.phase" ||
                op->getName().getStringRef().starts_with("pto.t")) phases.push_back(op);
            if (op->getName().getStringRef()=="test.scalar" && op->getNumOperands()==1) {
                scalarQueries.push_back(op);
                requestedScalars.push_back(op->getOperand(0));
            }
        });
        if (f->hasAttr("test.reverse_phases")) std::reverse(phases.begin(),phases.end());
        auto facts=SyncOccurrences::build(f,phases,requestedScalars);
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
            Array scalars;
            for (Operation* query:scalarQueries) {
                auto point=query->getAttrOfType<IntegerAttr>("point");
                auto lower=query->getAttrOfType<IntegerAttr>("lower");
                auto upper=query->getAttrOfType<IntegerAttr>("upper");
                auto allowance=query->getAttrOfType<IntegerAttr>("budget");
                if (!point || !lower || !upper || point.getInt()<0 ||
                    uint64_t(point.getInt())>=facts.points.size() ||
                    (allowance && allowance.getInt()<0)) return 2;
                RelationQueries queries(allowance ? uint64_t(allowance.getInt()) : 1000000);
                auto domain=facts.scalarDomain(query->getOperand(0),unsigned(point.getInt()),
                                               lower.getInt(),upper.getInt(),queries);
                const char* status=domain.status==QueryStatus::Proved ? "proved" :
                    domain.status==QueryStatus::NotEstablished ? "not-established" :
                    domain.status==QueryStatus::Unsupported ? "unsupported" : "budget-exhausted";
                Object scalar{{"point",point.getInt()},{"lower",lower.getInt()},{"upper",upper.getInt()},
                              {"status",status},{"reason",domain.reason},{"work",int64_t(queries.work())}};
                if (domain) scalar["relation"]=testing::encode(*domain.relation);
                scalars.push_back(std::move(scalar));
            }
            result["scalars"]=std::move(scalars);
        }
        functions.push_back(std::move(result));
    }
    llvm::outs()<<llvm::formatv("{0:2}\n",llvm::json::Value(std::move(functions)));
    return 0;
}
