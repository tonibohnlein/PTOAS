// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/InsertSync/LogicalSyncRelations.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>

using namespace mlir::pto::logical_sync;
using namespace mlir::presburger;
using llvm::json::Object;
using llvm::json::Array;
static Relation read(const Object& object) {
    unsigned d = *object.getInteger("d"), r = *object.getInteger("r"), s = *object.getInteger("s");
    auto result = Relation::getEmpty(PresburgerSpace::getRelationSpace(d, r, s));
    for (const auto& value : *object.getArray("pieces")) {
        auto& p = *value.getAsObject();
        IntegerRelation piece(PresburgerSpace::getRelationSpace(d, r, s, *p.getInteger("locals")));
        for (auto kind : {"eq", "ge"}) for (const auto& row : *p.getArray(kind)) {
            llvm::SmallVector<int64_t> coefficients;
            for (const auto& x : *row.getAsArray()) coefficients.push_back(*x.getAsInteger());
            if (coefficients.size() != piece.getNumCols()) std::exit(2);
            if (llvm::StringRef(kind) == "eq") piece.addEquality(coefficients);
            else piece.addInequality(coefficients);
        }
        result.unionInPlace(piece);
    }
    return result;
}
static Object write(const Relation& relation) {
    Array pieces;
    for (const auto& p : relation.getAllDisjuncts()) {
        Array equalities, inequalities;
        for (bool eq : {true, false}) {
            unsigned count = eq ? p.getNumEqualities() : p.getNumInequalities();
            for (unsigned i = 0; i < count; ++i) {
                Array row;
                for (unsigned j = 0; j < p.getNumCols(); ++j) {
                    std::string coefficient;
                    llvm::raw_string_ostream out(coefficient);
                    out << (eq ? p.atEq(i,j) : p.atIneq(i,j));
                    row.push_back(coefficient);
                }
                (eq ? equalities : inequalities).push_back(std::move(row));
            }
        }
        pieces.push_back(Object{{"locals", p.getNumLocalVars()}, {"eq", std::move(equalities)}, {"ge", std::move(inequalities)}});
    }
    return Object{{"d", relation.getNumDomainVars()}, {"r", relation.getNumRangeVars()},
                  {"s", relation.getNumSymbolVars()}, {"pieces", std::move(pieces)}};
}
static const char* status(QueryStatus value) {
    switch (value) {
    case QueryStatus::Proved: return "proved";
    case QueryStatus::NotEstablished: return "not-established";
    case QueryStatus::Unsupported: return "unsupported";
    case QueryStatus::BudgetExhausted: return "budget-exhausted";
    }
    std::abort();
}
int main() {
    auto input = llvm::MemoryBuffer::getSTDIN();
    if (!input) return 2;
    auto parsed = llvm::json::parse((*input)->getBuffer());
    if (!parsed) { llvm::errs() << llvm::toString(parsed.takeError()); return 2; }
    const auto* root = parsed->getAsObject();
    if (!root) return 2;
    RelationQueries queries(root->getInteger("budget").value_or(8000000));
    auto a = read(*root->getObject("a"));
    std::string op = root->getString("op")->str();
    Object output;
    if (op == "completion") {
        CompletionQueries completion(a);
        if (auto* order = root->getObject("issue_order")) {
            std::optional<Relation> global;
            if (auto* value = root->getObject("global_order"))
                global = read(*value);
            completion = CompletionQueries(a, read(*order), std::move(global));
        }
        Array answers;
        for (const auto& value : *root->getArray("needs")) {
            const auto& need = *value.getAsObject();
            if (auto* handoffs = need.getObject("replace_handoffs"))
                completion = completion.withHandoffs(read(*handoffs));
            if (auto* additional = need.getObject("add_handoffs")) {
                auto added = completion.addHandoffs(read(*additional), queries);
                if (added != QueryStatus::Proved) {
                    answers.push_back(Object{{"status", status(added)}, {"fixed", completion.fixedPoint()}});
                    continue;
                }
            }
            auto answer = completion.prove(read(*need.getObject("relation")), queries,
                                          need.getInteger("rounds").value_or(8));
            answers.push_back(Object{{"status", status(answer)}, {"fixed", completion.fixedPoint()}});
        }
        output["answers"] = std::move(answers);
        output["relation"] = write(completion.supply());
    } else {
        auto b = read(*root->getObject("b"));
        RelationResult result;
        if (op == "compose") result = queries.compose(a, b);
        else if (op == "subtract") result = queries.subtract(a, b);
        else if (op == "latest") result = queries.latestSources(a, b);
        else if (op == "first") result = queries.firstTargets(a, b);
        else if (op == "staircase") result = queries.staircase(a, b, read(*root->getObject("c")));
        else if (op == "contains") result.status = queries.contains(a, b);
        else return 2;
        output["status"] = status(result.status);
        output["reason"] = result.reason;
        if (result.relation) output["relation"] = write(*result.relation);
    }
    output["work"] = int64_t(queries.work());
    llvm::outs() << llvm::json::Value(std::move(output)) << "\n";
}
