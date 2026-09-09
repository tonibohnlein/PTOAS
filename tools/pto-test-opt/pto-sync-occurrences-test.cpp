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
#include <limits>

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
        SmallVector<Operation*> phases, scalarQueries, equalityQueries;
        SmallVector<Value> requestedScalars;
        f.walk([&](Operation* op) {
            if (op->getName().getStringRef()=="test.phase" ||
                op->getName().getStringRef().starts_with("pto.t")) phases.push_back(op);
            if (op->getName().getStringRef()=="test.scalar" && op->getNumOperands()==1) {
                scalarQueries.push_back(op);
                requestedScalars.push_back(op->getOperand(0));
            }
            if (op->getName().getStringRef()=="test.equal_scalars" && op->getNumOperands()==2) {
                equalityQueries.push_back(op);
                requestedScalars.append(op->getOperands().begin(),op->getOperands().end());
            }
        });
        if (f->hasAttr("test.reverse_phases")) std::reverse(phases.begin(),phases.end());
        auto facts=SyncOccurrences::build(f,phases,requestedScalars);
        Object result{{"function",f.getSymName()},{"complete",facts.complete},{"reason",facts.reason}};
        if (facts.complete) {
            if (auto depth=f->getAttrOfType<IntegerAttr>("test.equality_dag_depth")) {
                // Isolated representation-cost challenge, not scalar-import
                // coverage. On this checked singleton domain i=0, each floor
                // diamond still denotes the actual SSA value zero exactly.
                if (depth.getInt()<0 || depth.getInt()>40 || facts.loops.size()!=1 || requestedScalars.empty()) return 2;
                auto lo=dyn_cast<AffineConstantExpr>(facts.loopDomains[0].lower);
                auto hi=dyn_cast<AffineConstantExpr>(facts.loopDomains[0].upper);
                if (!lo || !hi || lo.getValue()!=0 || hi.getValue()!=1 || facts.loopDomains[0].step!=1) return 2;
                Value iv=facts.loops[0].getInductionVar();
                for (Value value:requestedScalars) if (value!=iv) return 2;
                AffineExpr expr=getAffineDimExpr(0,&context);
                const bool chain=f->hasAttr("test.equality_chain");
                for (int64_t i=0;i<depth.getInt();++i)
                    expr=chain ? (expr+1).floorDiv(2) : expr.floorDiv(2)+expr.floorDiv(3);
                facts.scalarExpressions[iv]=expr;
                result["synthetic_equality_dag_depth"]=depth.getInt();
            }
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
            if (f->hasAttr("test.periodic")) {
                auto selected=presburger::PresburgerSet::getEmpty(facts.domain(0).getRangeSet().getSpace());
                for (unsigned p=0;p<facts.points.size();++p)
                    selected.unionInPlace(facts.domain(p).getRangeSet());
                auto lower=f->getAttrOfType<IntegerAttr>("test.periodic_lower");
                auto upper=f->getAttrOfType<IntegerAttr>("test.periodic_upper");
                if ((lower || upper) && facts.dimensions()<2) return 2;
                if (lower || upper) {
                    auto narrowed=presburger::PresburgerSet::getEmpty(selected.getSpace());
                    for (auto piece:selected.getAllDisjuncts()) {
                        if (lower) piece.addBound(presburger::BoundType::LB,1,lower.getInt());
                        if (upper) piece.addBound(presburger::BoundType::UB,1,upper.getInt());
                        narrowed.unionInPlace(piece);
                    }
                    selected=std::move(narrowed);
                }
                if (auto locals=f->getAttrOfType<IntegerAttr>("test.periodic_cell_locals")) {
                    // Synthetic relation qualification, not scalar-import
                    // precision: conjoin caller-described existential rows to
                    // actual point domains, so every occurrence remains in
                    // the original immutable universe.
                    if (locals.getInt()<0 || locals.getInt()>32 || facts.dimensions()<2) return 2;
                    unsigned added=locals.getInt(), fixed=facts.dimensions()+facts.parameters.size();
                    auto cells=presburger::PresburgerSet::getEmpty(selected.getSpace());
                    for (auto piece:selected.getAllDisjuncts()) {
                        unsigned first=piece.getNumVars();
                        piece.appendVar(presburger::VarKind::Local,added);
                        for (bool equality:{true,false}) {
                            auto rows=f->getAttrOfType<ArrayAttr>(equality ? "test.periodic_cell_eq" : "test.periodic_cell_ge");
                            if (!rows) continue;
                            for (auto attribute:rows) {
                                auto coefficients=dyn_cast<ArrayAttr>(attribute);
                                if (!coefficients || coefficients.size()!=fixed+added+1) return 2;
                                SmallVector<llvm::DynamicAPInt> row(piece.getNumCols());
                                for (unsigned c=0;c<coefficients.size();++c) {
                                    auto value=dyn_cast<IntegerAttr>(coefficients[c]);
                                    if (!value || !value.getValue().isSignedIntN(64)) return 2;
                                    unsigned target=c<fixed ? c : c<fixed+added ? first+c-fixed : piece.getNumVars();
                                    row[target]=llvm::DynamicAPInt(value.getInt());
                                }
                                if (equality) piece.addEquality(row); else piece.addInequality(row);
                            }
                        }
                        cells.unionInPlace(piece);
                    }
                    selected=std::move(cells);
                    result["synthetic_periodic_refinement"]=true;
                }
                if (f->hasAttr("test.periodic_empty")) selected=presburger::PresburgerSet::getEmpty(selected.getSpace());
                auto allowance=f->getAttrOfType<IntegerAttr>("test.periodic_budget");
                if (allowance && allowance.getInt()<0) return 2;
                if (f->hasAttr("test.periodic_cell_probe")) {
                    Array probes;
                    for (const auto& piece:selected.getAllDisjuncts()) {
                        RelationQueries cellQueries(allowance ? uint64_t(allowance.getInt()) : 1000000);
                        auto simplified=testing::simplifyPeriodicCell(piece,1,cellQueries);
                        Object probe{{"original",testing::encode(Relation(piece))},
                            {"work",int64_t(cellQueries.work())},
                            {"status",simplified.status==QueryStatus::Proved ? "proved" :
                                simplified.status==QueryStatus::BudgetExhausted ? "budget-exhausted" : "unsupported"}};
                        if (simplified) probe["simplified"]=testing::encode(*simplified.relation);
                        probes.push_back(std::move(probe));
                    }
                    result["periodic_cells"]=std::move(probes);
                }
                RelationQueries queries(allowance ? uint64_t(allowance.getInt()) : 1000000);
                auto successors=f->hasAttr("test.periodic_cell_only") ?
                    RelationResult{QueryStatus::Unsupported,{},"cell-only test"} : facts.periodicSuccessors(selected,queries);
                Object periodic{{"domain",testing::encode(selected)},{"reason",successors.reason},
                    {"status",successors.status==QueryStatus::Proved ? "proved" :
                              successors.status==QueryStatus::BudgetExhausted ? "budget-exhausted" : "unsupported"},
                    {"work",int64_t(queries.work())}};
                if (successors) periodic["relation"]=testing::encode(*successors.relation);
                result["periodic"]=std::move(periodic);
            }
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
            Array equalities;
            for (Operation* query:equalityQueries) {
                auto source=query->getAttrOfType<IntegerAttr>("source");
                auto target=query->getAttrOfType<IntegerAttr>("target");
                auto allowance=query->getAttrOfType<IntegerAttr>("budget");
                if (!source || !target || source.getInt()<0 || target.getInt()<0 ||
                    uint64_t(source.getInt())>std::numeric_limits<unsigned>::max() ||
                    uint64_t(target.getInt())>std::numeric_limits<unsigned>::max() ||
                    (allowance && allowance.getInt()<0)) return 2;
                RelationQueries queries(allowance ? uint64_t(allowance.getInt()) : 1000000);
                auto relation=facts.equalScalars(query->getOperand(0),unsigned(source.getInt()),
                                                query->getOperand(1),unsigned(target.getInt()),queries);
                const char* status=relation.status==QueryStatus::Proved ? "proved" :
                    relation.status==QueryStatus::NotEstablished ? "not-established" :
                    relation.status==QueryStatus::Unsupported ? "unsupported" : "budget-exhausted";
                Object equality{{"source",source.getInt()},{"target",target.getInt()},
                                {"status",status},{"reason",relation.reason},{"work",int64_t(queries.work())}};
                if (relation) equality["relation"]=testing::encode(*relation.relation);
                if (uint64_t(source.getInt())<facts.points.size() && uint64_t(target.getInt())<facts.points.size()) {
                    auto original=facts.ordered(unsigned(source.getInt()),unsigned(target.getInt()));
                    if (!original) return 3;
                    auto fromLower=query->getAttrOfType<IntegerAttr>("source_lower");
                    auto toUpper=query->getAttrOfType<IntegerAttr>("target_upper");
                    auto parameterEquals=query->getAttrOfType<IntegerAttr>("parameter_equals");
                    if ((fromLower || toUpper) && facts.dimensions()<2) return 2;
                    if (parameterEquals && facts.parameters.empty()) return 2;
                    auto restricted=Relation::getEmpty(original.relation->getSpace());
                    for (auto piece:original.relation->getAllDisjuncts()) {
                        if (fromLower) piece.addBound(presburger::BoundType::LB,1,fromLower.getInt());
                        if (toUpper) piece.addBound(presburger::BoundType::UB,facts.dimensions()+1,toUpper.getInt());
                        if (parameterEquals) piece.addBound(presburger::BoundType::EQ,2*facts.dimensions(),parameterEquals.getInt());
                        restricted.unionInPlace(piece);
                    }
                    if (query->hasAttr("empty_filter")) restricted=Relation::getEmpty(restricted.getSpace());
                    if (query->hasAttr("wrong_filter_space")) restricted=facts.domain(unsigned(source.getInt()));
                    RelationQueries filterQueries(allowance ? uint64_t(allowance.getInt()) : 1000000);
                    auto filtered=facts.filterEqualScalars(restricted,query->getOperand(0),unsigned(source.getInt()),
                                                          query->getOperand(1),unsigned(target.getInt()),filterQueries);
                    equality["original_occurrences"]=testing::encode(restricted);
                    equality["filter_status"]=filtered.status==QueryStatus::Proved ? "proved" :
                        filtered.status==QueryStatus::NotEstablished ? "not-established" :
                        filtered.status==QueryStatus::Unsupported ? "unsupported" : "budget-exhausted";
                    equality["filter_reason"]=filtered.reason;
                    equality["filter_work"]=int64_t(filterQueries.work());
                    if (filtered) equality["filtered_relation"]=testing::encode(*filtered.relation);
                }
                equalities.push_back(std::move(equality));
            }
            result["equalities"]=std::move(equalities);
        }
        functions.push_back(std::move(result));
    }
    llvm::outs()<<llvm::formatv("{0:2}\n",llvm::json::Value(std::move(functions)));
    return 0;
}
