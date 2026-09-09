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
#include <set>

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
namespace {
using Coordinate = std::optional<llvm::DynamicAPInt>;
struct ProviderCounts {
    uint64_t calls = 0, globalCalls = 0, blockLookups = 0, returnedPieces = 0;
};
// Independent test provider: retain an immutable dictionary of supplied order
// pieces and materialize only candidate source/target buckets. The native
// consumer can generate those buckets directly from its occurrence facts.
struct TestOrderProvider {
    using Targets = std::map<Coordinate, Relation>;
    using Blocks = std::map<Coordinate, Targets>;
    PresburgerSpace space;
    Blocks lane, global;
    bool hasGlobal = false;
    static Coordinate coordinate(const IntegerRelation& piece, unsigned column, unsigned count) {
        if (!count) return {};
        for (unsigned i = 0; i < piece.getNumEqualities(); ++i) {
            auto row = piece.getEquality(i);
            if (row[column] != 1 && row[column] != -1) continue;
            bool isolated = true;
            for (unsigned j = 0; j < piece.getNumVars(); ++j)
                if (j != column && row[j] != 0) isolated = false;
            if (isolated) return row[column] == 1 ? -row.back() : row.back();
        }
        return {};
    }
    static Blocks index(const Relation& order) {
        Blocks result;
        for (const auto& piece : order.getAllDisjuncts()) {
            auto source = coordinate(piece, 0, piece.getNumDomainVars());
            auto target = coordinate(piece, piece.getNumDomainVars(), piece.getNumRangeVars());
            auto [it, inserted] = result[source].try_emplace(target, Relation::getEmpty(order.getSpace()));
            (void)inserted;
            it->second.unionInPlace(Relation(piece));
        }
        return result;
    }
    TestOrderProvider(const Relation& order, const std::optional<Relation>& all)
        : space(order.getSpace()), lane(index(order)), hasGlobal(bool(all)) {
        if (all) global = index(*all);
    }
    RelationResult select(bool all, const PresburgerSet& source, const PresburgerSet& target,
                          RelationQueries& queries, ProviderCounts& counts) const {
        ++counts.calls;
        counts.globalCalls += all;
        if (all && !hasGlobal) return {QueryStatus::Unsupported, {}, "test provider has no global order"};
        auto keys = [](const PresburgerSet& set) {
            std::set<Coordinate> result;
            for (const auto& piece : set.getAllDisjuncts())
                result.insert(coordinate(piece, 0, piece.getNumRangeVars()));
            return result;
        };
        auto from = keys(source), to = keys(target);
        bool anySource = from.count(std::nullopt), anyTarget = to.count(std::nullopt);
        from.insert(std::nullopt);
        to.insert(std::nullopt);
        auto result = Relation::getEmpty(space);
        auto takeGroup = [&](const Targets& group) {
            auto take = [&](const Relation& block) {
                result.unionInPlace(block);
                counts.returnedPieces += block.getNumDisjuncts();
            };
            if (anyTarget || group.size() < to.size()) {
                for (const auto& [key, block] : group) {
                    ++counts.blockLookups;
                    if (!queries.spend(1)) return false;
                    if (anyTarget || to.count(key)) take(block);
                }
            } else {
                for (const auto& key : to) {
                    ++counts.blockLookups;
                    if (!queries.spend(1)) return false;
                    if (auto found = group.find(key); found != group.end()) take(found->second);
                }
            }
            return true;
        };
        const auto& blocks = all ? global : lane;
        if (anySource) {
            for (const auto& [key, group] : blocks) {
                (void)key;
                if (!takeGroup(group)) return {QueryStatus::BudgetExhausted, {}, "test provider budget"};
            }
        } else {
            for (const auto& key : from) {
                if (!queries.spend(1)) return {QueryStatus::BudgetExhausted, {}, "test provider budget"};
                if (auto found = blocks.find(key); found != blocks.end())
                    if (!takeGroup(found->second)) return {QueryStatus::BudgetExhausted, {}, "test provider budget"};
            }
        }
        return queries.restrictEndpoints(result, source, target);
    }
};
} // namespace
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
        auto counts = std::make_shared<ProviderCounts>();
        if (auto* order = root->getObject("issue_order")) {
            std::optional<Relation> global;
            if (auto* value = root->getObject("global_order"))
                global = read(*value);
            if (root->getBoolean("lazy").value_or(false)) {
                auto provider = std::make_shared<const TestOrderProvider>(read(*order), global);
                std::optional<QueryStatus> fault;
                if (auto requested = root->getString("provider_fault"))
                    fault = *requested == "budget-exhausted" ? QueryStatus::BudgetExhausted : QueryStatus::Unsupported;
                completion = CompletionQueries(a,
                    [provider, counts, fault](bool all, const PresburgerSet& source,
                                              const PresburgerSet& target, RelationQueries& queries) {
                        if (fault) return RelationResult{*fault, {}, "injected test provider failure"};
                        return provider->select(all, source, target, queries, *counts);
                    }, bool(global));
            } else completion = CompletionQueries(a, read(*order), std::move(global));
        }
        Array answers;
        Array steps;
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
            if (root->getBoolean("record_steps").value_or(false))
                steps.push_back(Object{{"supply", write(completion.supply())}, {"work", int64_t(queries.work())},
                    {"index_lookups", int64_t(completion.orderIndexLookups())},
                    {"endpoint_comparisons", int64_t(queries.endpointComparisonCount())},
                    {"composition_index_builds", int64_t(queries.compositionIndexBuildCount())},
                    {"composition_index_pieces", int64_t(queries.compositionIndexPieceCount())},
                    {"endpoint_projections", int64_t(queries.endpointProjectionCount())},
                    {"endpoint_projection_pieces", int64_t(queries.endpointProjectionPieceCount())},
                    {"provider_calls", int64_t(counts->calls)}, {"provider_global_calls", int64_t(counts->globalCalls)},
                    {"provider_block_lookups", int64_t(counts->blockLookups)},
                    {"provider_returned_pieces", int64_t(counts->returnedPieces)}});
        }
        output["answers"] = std::move(answers);
        if (root->getBoolean("record_steps").value_or(false)) output["steps"] = std::move(steps);
        output["relation"] = write(completion.supply());
    } else {
        auto b = root->getObject("b") ? read(*root->getObject("b")) : a;
        RelationResult result;
        if (op == "normalize") result = queries.normalize(a);
        else if (op == "periodic_successors") {
            if (a.getNumDomainVars()) return 2;
            // Test-only exact positive row scaling exercises coefficients
            // wider than the JSON reader's int64 input without truncating them.
            if (auto scale = root->getInteger("template_scale")) {
                if (*scale <= 0) return 2;
                auto scaled = Relation::getEmpty(a.getSpace());
                for (const auto& original : a.getAllDisjuncts()) {
                    IntegerRelation piece(original);
                    for (unsigned i = 0; i < piece.getNumEqualities(); ++i)
                        for (unsigned j = 0; j < piece.getNumCols(); ++j) piece.atEq(i,j) *= *scale;
                    for (unsigned i = 0; i < piece.getNumInequalities(); ++i)
                        for (unsigned j = 0; j < piece.getNumCols(); ++j) piece.atIneq(i,j) *= *scale;
                    scaled.unionInPlace(piece);
                }
                a = std::move(scaled);
            }
            std::vector<PeriodicPublication> atoms;
            for (const auto& value : *root->getArray("atoms")) {
                const auto& atom = *value.getAsObject();
                atoms.push_back({*atom.getInteger("phase"), *atom.getInteger("rank"), *atom.getInteger("residue")});
            }
            result = queries.commonPeriodSuccessors(PresburgerSet(a),
                *root->getInteger("phase_coordinate"), *root->getInteger("iteration_coordinate"),
                *root->getInteger("period"), atoms);
        }
        else if (op == "restrict_endpoints")
            result = queries.restrictEndpoints(a, PresburgerSet(read(*root->getObject("sources"))),
                                               PresburgerSet(read(*root->getObject("targets"))));
        else if (op == "compose" || op == "compose_sequence") {
            RelationQueries::CompositionRHS rhs(b);
            bool prepared = root->getBoolean("prepared").value_or(false);
            if (op == "compose") result = prepared ? queries.compose(a, rhs) : queries.compose(a, b);
            else {
                Array steps;
                for (const auto& value : *root->getArray("inputs")) {
                    const auto& input = *value.getAsObject();
                    if (auto* replacement = input.getObject("rhs")) {
                        b = read(*replacement);
                        rhs = RelationQueries::CompositionRHS(b);
                    }
                    auto left = read(*input.getObject("left"));
                    result = prepared ? queries.compose(left, rhs) : queries.compose(left, b);
                    Object step{{"status", status(result.status)}, {"reason", result.reason},
                        {"work", int64_t(queries.work())},
                        {"composition_index_builds", int64_t(queries.compositionIndexBuildCount())},
                        {"composition_index_pieces", int64_t(queries.compositionIndexPieceCount())}};
                    if (result.relation) step["relation"] = write(*result.relation);
                    steps.push_back(std::move(step));
                }
                output["compose_steps"] = std::move(steps);
            }
        }
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
    output["endpoint_comparisons"] = int64_t(queries.endpointComparisonCount());
    output["composition_index_builds"] = int64_t(queries.compositionIndexBuildCount());
    output["composition_index_pieces"] = int64_t(queries.compositionIndexPieceCount());
    output["endpoint_projections"] = int64_t(queries.endpointProjectionCount());
    output["endpoint_projection_pieces"] = int64_t(queries.endpointProjectionPieceCount());
    output["difference_common_rows"] = int64_t(queries.differenceCommonRowCount());
    output["difference_partition_pieces"] = int64_t(queries.differencePartitionPieceCount());
    output["difference_implication_tests"] = int64_t(queries.differenceImplicationTestCount());
    output["boolean_partition_nodes"] = int64_t(queries.booleanPartitionNodeCount());
    output["boolean_witness_leaves"] = int64_t(queries.booleanWitnessLeafCount());
    output["boolean_max_depth"] = int64_t(queries.booleanMaximumDepth());
    output["relation_endpoint_index_pieces"] = int64_t(queries.relationEndpointIndexPieceCount());
    output["relation_endpoint_bucket_lookups"] = int64_t(queries.relationEndpointBucketLookupCount());
    output["difference_endpoint_comparisons"] = int64_t(queries.differenceEndpointComparisonCount());
    output["containment_endpoint_comparisons"] = int64_t(queries.containmentEndpointComparisonCount());
    output["periodic_atom_visits"] = int64_t(queries.periodicAtomVisitCount());
    output["periodic_sort_comparisons"] = int64_t(queries.periodicSortComparisonCount());
    output["periodic_output_pieces"] = int64_t(queries.periodicOutputPieceCount());
    llvm::outs() << llvm::json::Value(std::move(output)) << "\n";
}
