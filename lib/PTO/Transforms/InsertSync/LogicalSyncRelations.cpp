// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.


#include "PTO/Transforms/InsertSync/LogicalSyncRelations.h"
#include "mlir/Analysis/Presburger/Utils.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include <algorithm>
#include <map>
#include <set>
#include <cstdlib>
#include <chrono>
#include <limits>
#include <tuple>

using namespace mlir::pto::logical_sync;
namespace {
// Existing opt-in diagnostics include expensive primitive queries so a small
// checked-work count cannot be mistaken for a wall-clock bound.
struct QueryTimer {
    const char* name;
    RelationQueries::PrimitiveStats* stats;
    uint64_t left, right, other;
    std::chrono::steady_clock::time_point start;
    QueryTimer(const char* name, RelationQueries::PrimitiveStats* stats,
               uint64_t left, uint64_t right = 0, uint64_t other = 0)
        : name(name), stats(stats), left(left), right(right), other(other)
    {
        if (!stats) return;
        ++stats->calls;
        stats->maxInputPieces = std::max(stats->maxInputPieces, left + right + other);
        start = std::chrono::steady_clock::now();
    }
    QueryTimer(const QueryTimer&) = delete;
    QueryTimer& operator=(const QueryTimer&) = delete;
    ~QueryTimer() {
        if (!stats) return;
        auto elapsed = std::chrono::steady_clock::now() - start;
        stats->wallNanoseconds += std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
        double seconds = std::chrono::duration<double>(elapsed).count();
        if (seconds >= 0.25)
            llvm::errs() << "logical query " << name << " pieces " << left << "/" << right
                         << " seconds " << seconds << "\n";
    }
};
RelationResult failure(QueryStatus status, const char* reason)
{
    if (std::getenv("PTOAS_LOGICAL_TRACE"))
        llvm::errs() << "logical relation " << unsigned(status) << ": " << reason << "\n";
    return {status, std::nullopt, reason};
}
bool sameSpace(const Relation& a, const Relation& b) {
    return a.getSpace().isCompatible(b.getSpace());
}
// Substitute only unit-coefficient local definitions. In particular, a local
// witnessing x=2*k must survive: rational projection would lose slot parity.
struct CompactPiece : mlir::presburger::IntegerRelation {
    explicit CompactPiece(const IntegerRelation& piece) : IntegerRelation(piece) {}
    void compact()
    {
        removeRedundantLocalVars();
        removeDuplicateDivs();
        removeTrivialRedundancy();
        simplify();
    }
    // Only this Boolean feasibility query may forget endpoint/symbol identities.
    // Every coordinate is existential here. Unit equality substitution on a
    // disposable copy is integer-exact; nonunit divisibility remains intact.
    // Never return the reduced copy as an occurrence/completion relation.
    static bool emptyAfterUnitSubstitution(const IntegerRelation& piece)
    {
        CompactPiece reduced(piece);
        reduced.setSpace(mlir::presburger::PresburgerSpace::getRelationSpace(
            0, 0, 0, reduced.getNumVars()));
        reduced.removeRedundantLocalVars();
        return reduced.isIntegerEmpty();
    }
};
size_t pieceFingerprint(const mlir::presburger::IntegerRelation& piece)
{
    auto hash = llvm::hash_combine(
        piece.getNumEqualities(), piece.getNumInequalities(), piece.getNumVars());
    for (unsigned i = 0; i < piece.getNumEqualities(); ++i)
        hash = llvm::hash_combine(
            hash, llvm::hash_combine_range(piece.getEquality(i).begin(), piece.getEquality(i).end()));
    for (unsigned i = 0; i < piece.getNumInequalities(); ++i)
        hash = llvm::hash_combine(
            hash, llvm::hash_combine_range(piece.getInequality(i).begin(), piece.getInequality(i).end()));
    return size_t(hash);
}
using Constants = std::vector<std::optional<llvm::DynamicAPInt>>;
Constants fixedCoordinates(const mlir::presburger::IntegerRelation& piece, unsigned offset, unsigned count)
{
    Constants result(count);
    for (unsigned i = 0; i < piece.getNumEqualities(); ++i) {
        auto row = piece.getEquality(i);
        // A fixed coordinate requires exactly one nonzero variable term in
        // the WHOLE row, including symbols and local witnesses. Find it once;
        // testing isolation separately for every +/-1 term rescans long zero
        // prefixes quadratically on dense, late-coordinate equalities.
        unsigned nonzero = piece.getNumVars();
        bool isolated = true;
        for (unsigned k = 0; k < piece.getNumVars(); ++k) {
            if (row[k] == 0) continue;
            if (nonzero != piece.getNumVars()) { isolated = false; break; }
            nonzero = k;
        }
        if (isolated && nonzero >= offset && nonzero - offset < count &&
            (row[nonzero] == 1 || row[nonzero] == -1))
            result[nonzero - offset] = row[nonzero] == 1 ? -row.back() : row.back();
    }
    return result;
}
bool incompatible(const Constants& a, const Constants& b)
{
    for (unsigned i = 0; i < a.size(); ++i)
        if (a[i] && b[i] && *a[i] != *b[i])
            return true;
    return false;
}
// A cheap necessary endpoint filter, not an alias or completion proof. Bucket
// by the first source/range coordinates where fixed, retaining wildcard pieces
// and every remaining coordinate check. Return original disjunct order so
// callers' bounded implication attempts retain their existing priority.
struct EndpointCandidates {
    using Coordinate = std::optional<llvm::DynamicAPInt>;
    using Key = std::pair<Coordinate, Coordinate>;
    unsigned domain, range;
    std::vector<Constants> coordinates;
    std::map<Key, std::vector<unsigned>> both;
    std::map<Coordinate, std::vector<unsigned>> sources, targets;
    std::vector<unsigned> all;
    explicit EndpointCandidates(const Relation& relation)
        : domain(relation.getNumDomainVars()), range(relation.getNumRangeVars())
    {
        for (const auto& piece : relation.getAllDisjuncts()) {
            coordinates.push_back(fixedCoordinates(piece, 0, domain + range));
            unsigned ordinal = coordinates.size() - 1;
            auto [source, target] = key(coordinates.back());
            both[{source, target}].push_back(ordinal);
            sources[source].push_back(ordinal);
            targets[target].push_back(ordinal);
            all.push_back(ordinal);
        }
    }
    Key key(const Constants& values) const {
        return {domain ? values[0] : Coordinate{}, range ? values[domain] : Coordinate{}};
    }
    std::optional<std::vector<unsigned>> select(
        const Constants& values, RelationQueries& queries, uint64_t& lookups) const
    {
        auto [source, target] = key(values);
        std::vector<unsigned> result;
        auto append = [&](const auto& index, const auto& key) {
            if (!queries.spend(1)) return false;
            ++lookups;
            if (auto found = index.find(key); found != index.end())
                result.insert(result.end(), found->second.begin(), found->second.end());
            return true;
        };
        if (source && target) {
            if (!append(both, Key{source, target}) || !append(both, Key{source, {}}) ||
                !append(both, Key{{}, target}) || !append(both, Key{{}, {}})) return {};
        } else if (source) {
            if (!append(sources, source) || !append(sources, Coordinate{})) return {};
        } else if (target) {
            if (!append(targets, target) || !append(targets, Coordinate{})) return {};
        } else {
            if (!queries.spend(1)) return {};
            ++lookups;
            return all;
        }
        llvm::sort(result);
        return result;
    }
};
}

RelationQueries::RelationQueries(uint64_t budget)
    : remaining(budget), profiling(std::getenv("PTOAS_LOGICAL_TRACE") != nullptr)
{}

RelationQueries::ScopedBudget::ScopedBudget(RelationQueries& queries, uint64_t allowance)
    : owner(queries), previousRemaining(queries.scopedRemaining), previousExhausted(queries.scopedExhausted),
      initialRemaining(0)
{
    const uint64_t inherited = queries.scopedRemaining ? *queries.scopedRemaining : queries.remaining;
    queries.scopedRemaining = std::min(allowance, inherited);
    initialRemaining = *queries.scopedRemaining;
    queries.scopedExhausted = false;
}

RelationQueries::ScopedBudget::~ScopedBudget()
{
    // Nested scopes spend the enclosing scope's allowance as well. Restoring
    // the old value unconditionally would make a child scope's work free to
    // its parent and could let speculative work exceed the parent's cap.
    const uint64_t childRemaining = owner.scopedRemaining.value_or(0);
    const uint64_t consumed = initialRemaining - std::min(initialRemaining, childRemaining);
    if (previousRemaining) {
        const uint64_t parentRemaining = *previousRemaining;
        owner.scopedRemaining = consumed >= parentRemaining ? 0 : parentRemaining - consumed;
        owner.scopedExhausted = previousExhausted || owner.remaining == 0 ||
                                (consumed >= parentRemaining);
    } else {
        owner.scopedRemaining = std::nullopt;
        owner.scopedExhausted = previousExhausted;
    }
}

bool RelationQueries::exhaustAvailableBudget()
{
    const uint64_t limit = effectiveRemaining();
    return spend(limit == std::numeric_limits<uint64_t>::max() ? limit : limit + 1);
}

bool RelationQueries::charge(const Relation& relation, uint64_t* chargedCost) {
    const uint64_t limit = effectiveRemaining();
    uint64_t cost = 1;
    for (const auto& piece : relation.getAllDisjuncts()) {
        uint64_t rows = uint64_t(piece.getNumEqualities()) + piece.getNumInequalities() + 1;
        uint64_t cols = uint64_t(piece.getNumVars()) + 1;
        if (rows > limit || cols > limit || rows > (limit - std::min(cost, limit)) / cols) {
            exhaustAvailableBudget();
            return false;
        }
        cost += rows * cols;
    }
    if (cost > limit) { exhaustAvailableBudget(); return false; }
    if (!spend(cost)) return false;
    if (chargedCost) *chargedCost = cost;
    return true;
}

RelationResult RelationQueries::normalize(const Relation& relation)
{
    QueryTimer timer{"normalize", profiling ? &queryProfile.normalize : nullptr, relation.getNumDisjuncts()};
    if (!charge(relation))
        return failure(QueryStatus::BudgetExhausted, "normalization budget");
    auto result = Relation::getEmpty(relation.getSpace());
    std::map<size_t, std::vector<unsigned>> fingerprints;
    for (const auto& input : relation.getAllDisjuncts()) {
        CompactPiece piece(input);
        piece.compact();
        if (piece.isEmpty())
            continue;
        bool duplicate = false;
        auto& bucket = fingerprints[pieceFingerprint(piece)];
        for (unsigned index : bucket) {
            if (!spend(1))
                return failure(QueryStatus::BudgetExhausted, "normalization deduplication budget");
            if (piece.isObviouslyEqual(result.getAllDisjuncts()[index])) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            bucket.push_back(result.getNumDisjuncts());
            result.unionInPlace(Relation(piece));
        }
    }
    return {QueryStatus::Proved, std::move(result), {}};
}

RelationResult RelationQueries::compose(const Relation& first, const Relation& second) {
    std::optional<CompositionRHS::Index> index;
    return composeImpl(first, second, index);
}

RelationResult RelationQueries::compose(const Relation& first, CompositionRHS& second) {
    return composeImpl(first, second.relation, second.index);
}

RelationResult RelationQueries::composeImpl(
    const Relation& first, const Relation& second, std::optional<CompositionRHS::Index>& index) {
    QueryTimer timer{"compose", profiling ? &queryProfile.compose : nullptr,
                     first.getNumDisjuncts(), second.getNumDisjuncts()};
    if (!first.getSpace().getRangeSpace().isCompatible(second.getSpace().getDomainSpace()))
        return failure(QueryStatus::Unsupported, "incompatible composition spaces");
    if (!first.getNumDisjuncts() || !second.getNumDisjuncts()) {
        auto empty = Relation::getEmpty(
            mlir::presburger::PresburgerSpace::getRelationSpace(
                first.getNumDomainVars(), second.getNumRangeVars(), first.getNumSymbolVars()));
        if (!charge(empty))
            return failure(QueryStatus::BudgetExhausted, "empty composition budget");
        return {QueryStatus::Proved, std::move(empty), {}};
    }
    uint64_t rhsCharge = 0;
    if (!charge(first) || (index ? !spend(index->chargeCost) : !charge(second, &rhsCharge)))
        return failure(QueryStatus::BudgetExhausted, "composition budget");
    if (!index) {
        CompositionRHS::Index built;
        built.chargeCost = rhsCharge;
        ++compositionIndexBuilds;
        for (const auto& piece : second.getAllDisjuncts()) {
            ++compositionIndexPieces;
            built.joined.push_back(fixedCoordinates(piece, 0, piece.getNumDomainVars()));
            unsigned ordinal = built.joined.size() - 1;
            built.all.push_back(ordinal);
            if (!built.joined.back().empty() && built.joined.back().front())
                built.fixed[*built.joined.back().front()].push_back(ordinal);
            else
                built.wildcard.push_back(ordinal);
        }
        index = std::move(built);
    }
    const auto& joined = index->joined;
    const auto& indexed = index->fixed;
    const auto& wildcard = index->wildcard;
    const auto& all = index->all;
    auto result = Relation::getEmpty(
        mlir::presburger::PresburgerSpace::getRelationSpace(
            first.getSpace().getNumDomainVars(), second.getSpace().getNumRangeVars(),
            first.getSpace().getNumSymbolVars()));
    std::map<size_t, std::vector<unsigned>> fingerprints;
    for (const auto& left : first.getAllDisjuncts()) {
        auto constants = fixedCoordinates(left, left.getNumDomainVars(), left.getNumRangeVars());
        auto candidates = wildcard;
        if (constants.empty() || !constants.front())
            candidates = all;
        else if (auto found = indexed.find(*constants.front()); found != indexed.end())
            candidates.insert(candidates.end(), found->second.begin(), found->second.end());
        for (unsigned j : candidates) {
            const auto& right = second.getAllDisjuncts()[j];
            // Bound enumeration even when the joined coordinates cannot match.
            if (!spend(1))
                return failure(QueryStatus::BudgetExhausted, "composition enumeration budget");
            if (incompatible(constants, joined[j]))
                continue;
            if (!charge(Relation(left)) || !charge(Relation(right)))
                return failure(QueryStatus::BudgetExhausted, "composition product budget");
            CompactPiece piece(left);
            piece.compose(right); // Joined coordinates remain integer locals.
            if (!charge(Relation(piece)))
                return failure(QueryStatus::BudgetExhausted, "composition local budget");
            piece.compact();
            if (piece.isEmpty())
                continue;
            bool duplicate = false;
            auto& bucket = fingerprints[pieceFingerprint(piece)];
            for (unsigned index : bucket) {
                if (!spend(1))
                    return failure(QueryStatus::BudgetExhausted, "composition deduplication budget");
                if (piece.isObviouslyEqual(result.getAllDisjuncts()[index])) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) {
                bucket.push_back(result.getNumDisjuncts());
                result.unionInPlace(Relation(piece));
            }
        }
    }
    // Different witnesses often describe the same endpoint relation. Coalesce
    // only small local-free buckets; the pinned implementation does not merge
    // local-bearing pieces. Bucket separation is solely a work optimization.
    using EndpointKey = std::pair<std::optional<llvm::DynamicAPInt>, std::optional<llvm::DynamicAPInt>>;
    std::map<EndpointKey, Relation> groups;
    for (const auto& piece : result.getAllDisjuncts()) {
        auto a = fixedCoordinates(piece, 0, piece.getNumDomainVars());
        auto b = fixedCoordinates(piece, piece.getNumDomainVars(), piece.getNumRangeVars());
        EndpointKey key{a.empty() ? std::nullopt : a.front(), b.empty() ? std::nullopt : b.front()};
        auto [it, inserted] = groups.try_emplace(key, Relation::getEmpty(result.getSpace()));
        (void)inserted;
        it->second.unionInPlace(Relation(piece));
    }
    auto compact = Relation::getEmpty(result.getSpace());
    for (auto& [key, group] : groups) {
        (void)key;
        if (group.getNumDisjuncts() > 1 && group.getNumDisjuncts() <= 32 &&
            llvm::all_of(group.getAllDisjuncts(), [](const auto& p) { return p.getNumLocalVars() == 0; })) {
            for (unsigned i = 0; i < group.getNumDisjuncts(); ++i)
                if (!charge(group))
                    return failure(QueryStatus::BudgetExhausted, "composition coalescing budget");
            group = group.coalesce();
        }
        compact.unionInPlace(group);
    }
    result = std::move(compact);
    if (!charge(result)) return failure(QueryStatus::BudgetExhausted, "composition result budget");
    return {QueryStatus::Proved, std::move(result), {}};
}


// One exact partitioner serves both materialized subtraction and the Boolean
// containment fallback. A cursor emits first-violated-membership pieces lazily;
// only the Boolean consumer changes traversal order and stops at a witness.
class RelationQueries::DifferenceEngine {
    using IntegerRelation = mlir::presburger::IntegerRelation;
    using Row = llvm::SmallVector<llvm::DynamicAPInt>;
    struct PreparedRight {
        IntegerRelation membership;
        std::optional<IntegerRelation> total;
    };
    struct Child { IntegerRelation piece; bool knownNonempty; };
    RelationQueries& q;
    const Relation& from;
    const Relation& remove;
    QueryStatus outcome = QueryStatus::Proved;
    std::string reason;
    std::optional<EndpointCandidates> endpoints;
    std::map<std::vector<unsigned>, std::vector<PreparedRight>> qualifiedSubsets;
    bool fail(QueryStatus status, const char* text) {
        outcome = status; reason = text; return false;
    }
    bool propagate(const RelationResult& value) {
        outcome = value.status; reason = value.reason; return false;
    }
    RelationResult failed() const { return {outcome, {}, reason}; }
    bool chargeMatrix(uint64_t rows, uint64_t cols) {
        // Account matrix work without first copying it into a relation merely
        // to measure that copy. Use the ordinary one-piece charge formula.
        uint64_t remaining = q.remainingWork();
        if (rows > remaining || cols > remaining ||
            rows > (remaining - std::min<uint64_t>(1, remaining)) / cols) {
            q.exhaustAvailableBudget();
            return false;
        }
        return q.spend(1 + rows * cols);
    }
    bool chargePiece(const IntegerRelation& piece) {
        return chargeMatrix(uint64_t(piece.getNumConstraints()) + 1,
                            uint64_t(piece.getNumVars()) + 1);
    }
    bool prepareRight(PreparedRight& right) {
        if (right.total) return true;
        QueryTimer timer{"subtract_qualification", q.profiling ? &q.queryProfile.subtractQualification : nullptr, 1};
        if (!chargePiece(right.membership))
            return fail(QueryStatus::BudgetExhausted, "difference RHS preparation budget");
        auto divisions = right.membership.getLocalReprs();
        if (!divisions.hasAllReprs())
            return fail(QueryStatus::Unsupported, "difference requires defined RHS divisions");
        if (!chargeMatrix(uint64_t(right.membership.getNumConstraints()) +
                              2 * uint64_t(right.membership.getNumLocalVars()) + 1,
                          uint64_t(right.membership.getNumVars()) + 1))
            return fail(QueryStatus::BudgetExhausted, "difference RHS definitions budget");
        right.total.emplace(right.membership);
        unsigned offset = right.membership.getNumVars() - right.membership.getNumLocalVars();
        // Prepare canonical TOTAL definitions once per visited RHS. Keeping
        // preparation lazy preserves early coverage when later pieces need an
        // unsupported qualification. Membership rows remain separate.
        for (unsigned i = 0; i < right.membership.getNumLocalVars(); ++i) {
            right.total->addInequality(mlir::presburger::getDivUpperBound(
                divisions.getDividend(i), divisions.getDenom(i), offset + i));
            right.total->addInequality(mlir::presburger::getDivLowerBound(
                divisions.getDividend(i), divisions.getDenom(i), offset + i));
        }
        return true;
    }
    bool rights(const IntegerRelation& piece, std::vector<PreparedRight>*& result) {
        unsigned dimensions = from.getNumDomainVars() + from.getNumRangeVars();
        auto coordinates = fixedCoordinates(piece, 0, dimensions);
        auto possible = endpoints->select(coordinates, q, q.relationEndpointBucketLookups);
        if (!possible) return fail(QueryStatus::BudgetExhausted, "difference endpoint lookup budget");
        std::vector<unsigned> selected;
        for (unsigned i : *possible) {
            if (!q.spend(1)) return fail(QueryStatus::BudgetExhausted, "difference endpoint budget");
            ++q.differenceEndpointComparisons;
            if (!incompatible(coordinates, endpoints->coordinates[i])) selected.push_back(i);
        }
        if (selected.empty()) { result = nullptr; return true; }
        auto found = qualifiedSubsets.find(selected);
        if (found == qualifiedSubsets.end()) {
            QueryTimer timer{"subtract_qualification", q.profiling ? &q.queryProfile.subtractQualification : nullptr,
                             selected.size()};
            auto subset = Relation::getEmpty(remove.getSpace());
            for (unsigned i : selected)
                subset.unionInPlace(Relation(remove.getAllDisjuncts()[i]));
            // Exact conversion satisfies difference's division-local RHS
            // precondition; this is not rational projection or union lexopt.
            auto compact = q.normalize(subset);
            if (!compact.relation)
                return propagate(compact);
            auto qualified = compact.relation->computeReprWithOnlyDivLocals();
            if (!q.charge(qualified))
                return fail(QueryStatus::BudgetExhausted, "division normalization budget");
            // The pinned symbolic-domain conversion can expose another bounded
            // existential witness (e.g. 1 <= e < n <= 2) in its result. Qualify
            // those pieces again, without rewriting already-qualified divs.
            // Every pass is integer-exact; the bounded retry never assumes the
            // conversion's name certifies its output representation.
            for (unsigned attempt = 0; !qualified.hasOnlyDivLocals() && attempt < 8; ++attempt) {
                auto next = Relation::getEmpty(qualified.getSpace());
                for (const auto& p : qualified.getAllDisjuncts()) {
                    if (!q.charge(Relation(p)))
                        return fail(QueryStatus::BudgetExhausted, "division requalification budget");
                    next.unionInPlace(p.hasOnlyDivLocals() ? Relation(p) : p.computeReprWithOnlyDivLocals());
                }
                if (!q.charge(next))
                    return fail(QueryStatus::BudgetExhausted, "division requalification result budget");
                qualified = std::move(next);
            }

            std::vector<PreparedRight> prepared;
            for (const auto& right : qualified.getAllDisjuncts()) {
                if (!chargePiece(right))
                    return fail(QueryStatus::BudgetExhausted, "difference RHS cache budget");
                prepared.push_back({right, {}});
            }
            found = qualifiedSubsets.emplace(std::move(selected), std::move(prepared)).first;
        }
        result = &found->second;
        return true;
    }
    class Cursor {
        DifferenceEngine& owner;
        IntegerRelation original, prefix;
        std::vector<Row> equalities, rows;
        std::vector<bool> keep;
        unsigned equalityIndex = 0, inequalityIndex = 0;
        bool secondSign = false, unchanged = false, finished = false, inequalitiesReady = false;
    public:
        Cursor(DifferenceEngine& owner, const IntegerRelation& left)
            : owner(owner), original(left), prefix(left) {}
        bool prepare(const PreparedRight& prepared) {
            QueryTimer partitionTimer{"subtract_partition", owner.q.profiling ? &owner.q.queryProfile.subtractPartition : nullptr, 1};
            auto rhs = *prepared.total;
            unsigned originalInequalities = prepared.membership.getNumInequalities();
            prefix.mergeLocalVars(rhs);
            for (unsigned i = originalInequalities; i < rhs.getNumInequalities(); ++i)
                prefix.addInequality(rhs.getInequality(i));
            auto copyRow = [](llvm::ArrayRef<llvm::DynamicAPInt> row) {
                return llvm::SmallVector<llvm::DynamicAPInt>(row);
            };
            // After local witnesses are aligned, a literal prefix row is
            // already true throughout that prefix. Do not partition its
            // impossible complement. Equalities may differ only by sign;
            // inequalities must match exactly. Hashes select candidates,
            // while exact coefficient equality establishes membership.
            struct KnownRows {
                using Row = llvm::SmallVector<llvm::DynamicAPInt>;
                bool equality;
                std::map<size_t, std::vector<Row>> buckets;
                std::optional<bool> insert(llvm::ArrayRef<llvm::DynamicAPInt> input,
                                           RelationQueries& queries) {
                    if (!queries.spend(uint64_t(input.size()) + 1)) return {};
                    Row row(input);
                    if (equality)
                        for (const auto& coefficient : row) {
                            if (coefficient == 0) continue;
                            if (coefficient < 0)
                                for (auto& entry : row) entry = -entry;
                            break;
                        }
                    auto& bucket = buckets[size_t(llvm::hash_combine_range(row.begin(), row.end()))];
                    for (const auto& existing : bucket) {
                        if (!queries.spend(uint64_t(row.size()) + 1)) return {};
                        if (existing == row) return false;
                    }
                    bucket.push_back(std::move(row));
                    return true;
                }
            };
            KnownRows knownEqualities{true, {}}, knownInequalities{false, {}};
            for (unsigned i = 0; i < prefix.getNumEqualities(); ++i)
                if (!knownEqualities.insert(prefix.getEquality(i), owner.q).has_value())
                    return owner.fail(QueryStatus::BudgetExhausted, "difference common-row index budget");
            for (unsigned i = 0; i < prefix.getNumInequalities(); ++i)
                if (!knownInequalities.insert(prefix.getInequality(i), owner.q).has_value())
                    return owner.fail(QueryStatus::BudgetExhausted, "difference common-row index budget");

            for (unsigned i = 0; i < rhs.getNumEqualities(); ++i) {
                auto inserted = knownEqualities.insert(rhs.getEquality(i), owner.q);
                if (!inserted)
                    return owner.fail(QueryStatus::BudgetExhausted, "difference common equality budget");
                if (*inserted) equalities.push_back(copyRow(rhs.getEquality(i)));
                else ++owner.q.differenceCommonRows;
            }
            for (unsigned i = 0; i < originalInequalities; ++i) {
                auto inserted = knownInequalities.insert(rhs.getInequality(i), owner.q);
                if (!inserted)
                    return owner.fail(QueryStatus::BudgetExhausted, "difference common inequality budget");
                if (*inserted) rows.push_back(copyRow(rhs.getInequality(i)));
                else ++owner.q.differenceCommonRows;
            }

            // Total definitions extend every point. If all membership rows are
            // already true, the entire branch is covered by this RHS.
            if (equalities.empty() && rows.empty()) { finished = true; return true; }
            if (!owner.chargeMatrix(uint64_t(prefix.getNumConstraints()) + equalities.size() + rows.size() + 1,
                                    uint64_t(prefix.getNumVars()) + 1))
                return owner.fail(QueryStatus::BudgetExhausted, "difference intersection budget");
            auto intersection = prefix;
            for (const auto& row : equalities) intersection.addEquality(row);
            for (const auto& row : rows) intersection.addInequality(row);
            if (CompactPiece::emptyAfterUnitSubstitution(intersection)) {
                unchanged = true; return true;
            }
            return true;
        }
        bool prepareInequalities() {
            // Equality-violation children are requested first. A Boolean
            // witness there needs none of these optional redundancy queries.
            // At this point the retained prefix contains EVERY RHS equality.
            if (rows.empty()) return true;
            // Redundancy is only a partition-size optimization. This is the
            // same fresh-Simplex implication as materialized subtraction, with
            // all equalities retained in the intersection and no tableau undo.
            if (!owner.chargeMatrix(uint64_t(prefix.getNumConstraints()) + rows.size() + 1,
                                    uint64_t(prefix.getNumVars()) + 1))
                return owner.fail(QueryStatus::BudgetExhausted, "difference implication copy budget");
            auto intersection = prefix;
            for (const auto& row : rows) intersection.addInequality(row);
            keep.assign(rows.size(), true);
            unsigned base = prefix.getNumInequalities();
            QueryTimer implicationTimer{"subtract_implication", owner.q.profiling ? &owner.q.queryProfile.subtractImplication : nullptr, 1};
            for (unsigned i = rows.size(); i > 0; --i) {
                intersection.removeInequality(base + i - 1);
                if (!owner.chargePiece(intersection))
                    return owner.fail(QueryStatus::BudgetExhausted, "difference implication budget");
                ++owner.q.differenceImplicationTests;
                mlir::presburger::Simplex simplex(intersection);
                if (simplex.isRedundantInequality(rows[i - 1])) keep[i - 1] = false;
                else intersection.addInequality(rows[i - 1]);
            }
            return true;
        }
        std::optional<Child> next() {
            QueryTimer partitionTimer{"subtract_partition", owner.q.profiling ? &owner.q.queryProfile.subtractPartition : nullptr, 1};
            if (finished) return {};
            if (unchanged) { finished = true; return Child{std::move(original), false}; }
            while (equalityIndex < equalities.size()) {
                if (!owner.chargePiece(prefix)) {
                    owner.fail(QueryStatus::BudgetExhausted, "difference equality budget"); return {};
                }
                Row row = equalities[equalityIndex];
                if (secondSign) for (auto& coefficient : row) coefficient = -coefficient;
                --row.back();
                CompactPiece excluded(prefix);
                excluded.addInequality(row);
                if (secondSign) { prefix.addEquality(equalities[equalityIndex++]); secondSign = false; }
                else secondSign = true;
                excluded.compact();
                if (!CompactPiece::emptyAfterUnitSubstitution(excluded)) {
                    ++owner.q.differencePartitionPieces;
                    return Child{std::move(excluded), true};
                }
            }
            if (!inequalitiesReady) {
                if (!prepareInequalities()) return {};
                inequalitiesReady = true;
            }
            while (inequalityIndex < rows.size()) {
                unsigned i = inequalityIndex++;
                if (!keep[i]) continue;
                if (!owner.chargePiece(prefix)) {
                    owner.fail(QueryStatus::BudgetExhausted, "difference partition budget"); return {};
                }
                Row negated = rows[i];
                for (auto& coefficient : negated) coefficient = -coefficient;
                --negated.back();
                CompactPiece excluded(prefix);
                excluded.addInequality(negated);
                prefix.addInequality(rows[i]);
                excluded.compact();
                if (!CompactPiece::emptyAfterUnitSubstitution(excluded)) {
                    ++owner.q.differencePartitionPieces;
                    return Child{std::move(excluded), true};
                }
            }
            finished = true;
            return {};
        }
    };
    std::unique_ptr<Cursor> cursor(const IntegerRelation& left, PreparedRight& right) {
        if (!prepareRight(right)) return {};
        // Local alignment expands BOTH matrices to the union of witnesses.
        // Charge those possible dimensions before mergeLocalVars allocates
        // them. Sums of the original matrix sizes do not bound the aligned
        // RHS. This is the existing cell-work model, not a wall-time bound.
        uint64_t alignedColumns = uint64_t(left.getNumVars()) + right.total->getNumLocalVars() + 1;
        uint64_t prefixRows = uint64_t(left.getNumConstraints()) +
                              2 * uint64_t(right.membership.getNumLocalVars()) + 1;
        if (!q.spend(1) || !chargePiece(left) || !chargeMatrix(prefixRows, alignedColumns) ||
            !chargeMatrix(uint64_t(right.total->getNumConstraints()) + 1, alignedColumns)) {
            fail(QueryStatus::BudgetExhausted, "difference alignment budget"); return {};
        }
        auto value = std::make_unique<Cursor>(*this, left);
        if (!value->prepare(right)) return {};
        return value;
    }
    bool nonempty(const Child& child) {
        if (child.knownNonempty) return true;
        if (!chargePiece(child.piece)) { fail(QueryStatus::BudgetExhausted, "difference leaf budget"); return false; }
        return !CompactPiece::emptyAfterUnitSubstitution(child.piece);
    }
public:
    DifferenceEngine(RelationQueries& queries, const Relation& from, const Relation& remove)
        : q(queries), from(from), remove(remove) {}
    RelationResult run(bool stopAtWitness) {
        if (!sameSpace(from, remove)) return failure(QueryStatus::Unsupported, "incompatible difference spaces");
        if (!from.getNumDisjuncts()) {
            if (!q.charge(from)) return failure(QueryStatus::BudgetExhausted, "empty difference budget");
            return {QueryStatus::Proved, from, {}};
        }
        if (!q.charge(from) || !q.charge(remove)) return failure(QueryStatus::BudgetExhausted, "difference budget");
        endpoints.emplace(remove);
        q.relationEndpointIndexPieces += remove.getNumDisjuncts();
        auto result = Relation::getEmpty(from.getSpace());
        for (const auto& piece : from.getAllDisjuncts()) {
            std::vector<PreparedRight>* right = nullptr;
            if (!rights(piece, right)) return failed();
            if (!right || right->empty()) {
                if (stopAtWitness) {
                    if (!chargePiece(piece)) return failure(QueryStatus::BudgetExhausted, "difference leaf copy budget");
                    if (nonempty({piece, false})) {
                        ++q.booleanWitnessLeaves;
                        return {QueryStatus::Proved, Relation(piece), {}};
                    }
                    if (outcome != QueryStatus::Proved) return failed();
                } else result.unionInPlace(piece);
                continue;
            }
            auto pending = q.normalize(Relation(piece));
            if (!pending) return pending;
            if (!stopAtWitness) {
                for (auto& rhs : *right) {
                    auto next = Relation::getEmpty(from.getSpace());
                    for (const auto& left : pending.relation->getAllDisjuncts()) {
                        auto step = cursor(left, rhs);
                        if (!step) return failed();
                        while (auto child = step->next()) next.unionInPlace(child->piece);
                        if (outcome != QueryStatus::Proved) return failed();
                    }
                    pending = q.normalize(next);
                    if (!pending) return pending;
                    if (!pending.relation->getNumDisjuncts()) break;
                }
                result.unionInPlace(*pending.relation);
                continue;
            }
            // Depth-first traversal retains one active branch per RHS level,
            // never the full uncovered union. A feasible leaf is a witness
            // ONLY after every relevant RHS piece has been excluded.
            struct Frame { unsigned right; std::unique_ptr<Cursor> step; };
            for (const auto& left : pending.relation->getAllDisjuncts()) {
                std::vector<Frame> stack;
                auto initial = cursor(left, right->front());
                if (!initial) return failed();
                stack.push_back({0, std::move(initial)});
                ++q.booleanPartitionNodes;
                q.booleanMaxDepth = std::max<uint64_t>(q.booleanMaxDepth, stack.size());
                while (!stack.empty()) {
                    auto child = stack.back().step->next();
                    if (outcome != QueryStatus::Proved) return failed();
                    if (!child) { stack.pop_back(); continue; }
                    unsigned nextRight = stack.back().right + 1;
                    if (nextRight == right->size()) {
                        if (nonempty(*child)) {
                            if (!chargePiece(child->piece))
                                return failure(QueryStatus::BudgetExhausted, "difference witness copy budget");
                            ++q.booleanWitnessLeaves;
                            return {QueryStatus::Proved, Relation(child->piece), {}};
                        }
                        if (outcome != QueryStatus::Proved) return failed();
                    } else {
                        auto next = cursor(child->piece, (*right)[nextRight]);
                        if (!next) return failed();
                        stack.push_back({nextRight, std::move(next)});
                        ++q.booleanPartitionNodes;
                        q.booleanMaxDepth = std::max<uint64_t>(q.booleanMaxDepth, stack.size());
                    }
                }
            }
        }
        return stopAtWitness ? RelationResult{QueryStatus::Proved, std::move(result), {}} : q.normalize(result);
    }
};

RelationResult RelationQueries::subtract(const Relation& from, const Relation& remove) {
    QueryTimer timer{"subtract", profiling ? &queryProfile.subtract : nullptr,
                     from.getNumDisjuncts(), remove.getNumDisjuncts()};
    return DifferenceEngine(*this, from, remove).run(false);
}

QueryStatus RelationQueries::contains(const Relation& supply, const Relation& requirement) {
    QueryTimer timer{"contains", profiling ? &queryProfile.contains : nullptr,
                     supply.getNumDisjuncts(), requirement.getNumDisjuncts()};
    if (!sameSpace(supply, requirement)) return QueryStatus::Unsupported;
    if (!charge(supply) || !charge(requirement)) return QueryStatus::BudgetExhausted;
    // A sufficient piecewise implication avoids manufacturing an exact set
    // difference for simple domain/bound comparisons. Local witnesses are
    // aligned only by mergeLocalVars' proved division identities. Qualified
    // total floor definitions extend the antecedent with integer witnesses,
    // never with RHS membership restrictions. Unqualified supply locals remain
    // unconstrained, giving a stronger sufficient test. Failure of this rational
    // test supplies no negative answer.
    if (!requirement.getNumDisjuncts()) return QueryStatus::Proved;
    EndpointCandidates endpoints(supply);
    relationEndpointIndexPieces += supply.getNumDisjuncts();
    bool covered = true;
    unsigned testsLeft = 512;
    for (const auto& needed : requirement.getAllDisjuncts()) {
        bool found = false;
        unsigned candidatesLeft = 8;
        auto coordinates = fixedCoordinates(needed, 0, needed.getNumDomainVars() + needed.getNumRangeVars());
        auto possible = endpoints.select(coordinates, *this, relationEndpointBucketLookups);
        if (!possible) return QueryStatus::BudgetExhausted;
        for (unsigned ordinal : *possible) {
            const auto& available = supply.getAllDisjuncts()[ordinal];
            if (!spend(1)) return QueryStatus::BudgetExhausted;
            ++containmentEndpointComparisons;
            if (incompatible(coordinates, endpoints.coordinates[ordinal])) continue;
            if (needed.isObviouslyEqual(available)) { found = true; break; }
            if (!candidatesLeft || !testsLeft) break;
            --candidatesLeft;
            mlir::presburger::IntegerRelation lhs(needed), rhs(available);
            unsigned membershipRows = rhs.getNumInequalities();
            auto divisions = rhs.getLocalReprs();
            if (divisions.hasAllReprs()) {
                unsigned offset = rhs.getNumVars() - rhs.getNumLocalVars();
                for (unsigned i = 0; i < rhs.getNumLocalVars(); ++i) {
                    rhs.addInequality(mlir::presburger::getDivUpperBound(
                        divisions.getDividend(i), divisions.getDenom(i), offset + i));
                    rhs.addInequality(mlir::presburger::getDivLowerBound(
                        divisions.getDividend(i), divisions.getDenom(i), offset + i));
                }
            }
            lhs.mergeLocalVars(rhs);
            for (unsigned i = membershipRows; i < rhs.getNumInequalities(); ++i)
                lhs.addInequality(rhs.getInequality(i));
            unsigned rows = membershipRows + 2 * rhs.getNumEqualities();
            if (lhs.getNumCols() > 64 || lhs.getNumConstraints() > 128 || rows > testsLeft) continue;
            bool implied = true, exhausted = false;
            // Each objective query rolls back its temporary row in the pinned
            // Simplex implementation. Reuse the unchanged antecedent tableau;
            // never add a tested RHS membership constraint to that antecedent.
            std::optional<mlir::presburger::Simplex> simplex;
            auto proves = [&](llvm::SmallVector<llvm::DynamicAPInt> row) {
                --testsLeft;
                if (!charge(Relation(lhs))) { exhausted = true; return false; }
                if (!simplex) simplex.emplace(lhs);
                return simplex->isEmpty() || simplex->isRedundantInequality(row);
            };
            for (unsigned i = 0; implied && i < rhs.getNumEqualities(); ++i) {
                llvm::SmallVector<llvm::DynamicAPInt> row(rhs.getEquality(i));
                implied = proves(row);
                for (auto& coefficient : row) coefficient = -coefficient;
                if (implied) implied = proves(row);
            }
            for (unsigned i = 0; implied && i < membershipRows; ++i)
                implied = proves(llvm::SmallVector<llvm::DynamicAPInt>(rhs.getInequality(i)));
            if (exhausted) return QueryStatus::BudgetExhausted;
            if (implied) { found = true; break; }
        }
        if (!found) { covered = false; break; }
    }
    if (covered) return QueryStatus::Proved;
    // A concrete integer counterexample can refute an over-broad proposed
    // guard without constructing its complement. Sampling never establishes
    // containment: membership or an unavailable sample falls through to the
    // exact difference. Drop only local witness coordinates, preserving all
    // domain/range coordinates and symbols for the existential membership test.
    unsigned samplesLeft = 2;
    bool smallSupply = supply.getNumDisjuncts() <= 64 && llvm::all_of(
        supply.getAllDisjuncts(), [](const auto& p) { return p.getNumCols() <= 16 && p.getNumConstraints() <= 64; });
    for (const auto& needed : requirement.getAllDisjuncts()) {
        if (!samplesLeft) break;
        if (!smallSupply || needed.getNumCols() > 16 || needed.getNumConstraints() > 64) continue;
        --samplesLeft;
        if (!charge(Relation(needed)) || !charge(supply)) return QueryStatus::BudgetExhausted;
        auto sample = needed.findIntegerSample();
        if (!sample) continue;
        sample->resize(needed.getNumVars() - needed.getNumLocalVars());
        if (!supply.containsPoint(*sample)) return QueryStatus::NotEstablished;
    }
    // The shared exact partitioner searches lazily here. An uncovered witness
    // is accepted only after excluding every relevant supply disjunct. Empty
    // branches and covered subtrees are pruned without materializing a union.
    auto missing = DifferenceEngine(*this, requirement, supply).run(true);
    if (!missing) return missing.status;
    return missing.relation->getNumDisjuncts() == 0 ? QueryStatus::Proved : QueryStatus::NotEstablished;
}

RelationResult RelationQueries::latestSources(const Relation& requirements, const Relation& sourceBefore) {
    auto order = restrictCandidateOrder(sourceBefore, requirements, true);
    if (!order)
        return order;
    auto dominated = compose(*order.relation, requirements);
    if (!dominated) return dominated;
    auto latest = subtract(requirements, *dominated.relation);
    if (!latest) return latest;
    auto complete = contains(latest.relation->getRangeSet(), requirements.getRangeSet());
    if (complete != QueryStatus::Proved)
        return failure(complete, "not every sink has an established latest producer");
    return latest;
}

RelationResult RelationQueries::firstTargets(const Relation& requirements, const Relation& targetBefore) {
    auto order = restrictCandidateOrder(targetBefore, requirements, false);
    if (!order)
        return order;
    auto dominated = compose(requirements, *order.relation);
    if (!dominated) return dominated;
    auto first = subtract(requirements, *dominated.relation);
    if (!first) return first;
    auto complete = contains(first.relation->getDomainSet(), requirements.getDomainSet());
    if (complete != QueryStatus::Proved)
        return failure(complete, "not every source has an established next overwrite");
    return first;
}

RelationResult RelationQueries::commonPeriodSuccessors(
    const mlir::presburger::PresburgerSet& common, unsigned phaseCoordinate,
    unsigned iterationCoordinate, int64_t period, llvm::ArrayRef<PeriodicPublication> atoms)
{
    using namespace mlir::presburger;
    using llvm::DynamicAPInt;
    const unsigned n = common.getNumRangeVars(), symbols = common.getNumSymbolVars();
    if (common.getNumDomainVars() || phaseCoordinate >= n || iterationCoordinate >= n ||
        phaseCoordinate == iterationCoordinate || period <= 0)
        return ::failure(QueryStatus::Unsupported, "periodic population space or period unavailable");
    auto result = Relation::getEmpty(PresburgerSpace::getRelationSpace(n, n, symbols));
    if (!charge(common)) return ::failure(QueryStatus::BudgetExhausted, "periodic template budget");
    if (!common.getNumDisjuncts()) return {QueryStatus::Proved, std::move(result), {}};
    if (common.getNumDisjuncts() != 1)
        return ::failure(QueryStatus::Unsupported, "periodic population needs one common interval template");
    const auto& base = common.getAllDisjuncts().front();
    std::vector<bool> fixed(n, false);
    bool lower = false, upper = false;
    for (bool equality : {true, false}) {
        const unsigned count = equality ? base.getNumEqualities() : base.getNumInequalities();
        for (unsigned i = 0; i < count; ++i) {
            auto row = equality ? base.getEquality(i) : base.getInequality(i);
            // Keep the compact path's per-cell memory bound independent of
            // arbitrary-width coefficients supplied by other relation queries.
            // Larger exact coefficients remain supported by the general path.
            for (const auto& coefficient : row)
                if (coefficient < std::numeric_limits<int64_t>::min() ||
                    coefficient > std::numeric_limits<int64_t>::max())
                    return ::failure(QueryStatus::Unsupported, "periodic template coefficient exceeds int64");
            if (row[phaseCoordinate] != 0)
                return ::failure(QueryStatus::Unsupported, "periodic common phase must be unconstrained");
            unsigned coordinate = n;
            for (unsigned j = 0; j < n; ++j) if (row[j] != 0) {
                if (coordinate != n)
                    return ::failure(QueryStatus::Unsupported, "periodic occurrence coordinates are coupled");
                coordinate = j;
            }
            if (coordinate == n) continue; // Entirely parameter/local guard.
            if (coordinate != iterationCoordinate) {
                if (!equality || (row[coordinate] != 1 && row[coordinate] != -1))
                    return ::failure(QueryStatus::Unsupported, "periodic enclosing coordinate is not fixed");
                for (unsigned j = n; j < base.getNumVars(); ++j) if (row[j] != 0)
                    return ::failure(QueryStatus::Unsupported, "periodic enclosing coordinate is parameter dependent");
                fixed[coordinate] = true;
                continue;
            }
            if (row[coordinate] != 1 && row[coordinate] != -1)
                return ::failure(QueryStatus::Unsupported, "periodic interval has a nonunit coefficient");
            for (unsigned j = n + symbols; j < base.getNumVars(); ++j) if (row[j] != 0)
                return ::failure(QueryStatus::Unsupported, "periodic interval depends on a local witness");
            lower |= equality || row[coordinate] == 1;
            upper |= equality || row[coordinate] == -1;
        }
    }
    for (unsigned j = 0; j < n; ++j)
        if (j != phaseCoordinate && j != iterationCoordinate && !fixed[j])
            return ::failure(QueryStatus::Unsupported, "periodic enclosing invocation is not fixed");
    if (!lower || !upper)
        return ::failure(QueryStatus::Unsupported, "periodic interval needs finite lower and upper bounds");
    if (atoms.size() > std::numeric_limits<uint64_t>::max() / 4 || !spend(atoms.size() * 4))
        return ::failure(QueryStatus::BudgetExhausted, "periodic population setup budget");
    std::map<int64_t, int64_t> phaseRanks, rankPhases;
    std::vector<PeriodicPublication> ordered(atoms.begin(), atoms.end()), scratch(ordered.size());
    uint64_t mapWork = 1;
    // Two balanced-tree lookups/inserts, each bounded by twice the height.
    for (size_t count = atoms.size(); count; count /= 2) mapWork += 4;
    for (const auto& atom : atoms) {
        ++periodicAtomVisits;
        if (!spend(mapWork)) return ::failure(QueryStatus::BudgetExhausted, "periodic atom qualification budget");
        if (atom.residue < 0 || atom.residue >= period)
            return ::failure(QueryStatus::Unsupported, "periodic residue is outside its period");
        auto [phase, newPhase] = phaseRanks.emplace(atom.phase, atom.rank);
        auto [rank, newRank] = rankPhases.emplace(atom.rank, atom.phase);
        if ((!newPhase && phase->second != atom.rank) || (!newRank && rank->second != atom.phase))
            return ::failure(QueryStatus::Unsupported, "periodic phase ranks are inconsistent or tied");
    }
    // Bounded bottom-up merge sort: charge every comparison and move, and stop
    // immediately on exhaustion. No comparator continues after a failed budget
    // request, and no period-sized table or all-pairs candidate map is built.
    for (size_t width = 1; width < ordered.size();) {
        for (size_t begin = 0; begin < ordered.size();) {
            size_t middle = begin + std::min(width, ordered.size() - begin);
            size_t end = middle + std::min(width, ordered.size() - middle);
            size_t left = begin, right = middle;
            for (size_t out = begin; out < end; ++out) {
                if (!spend(1)) return ::failure(QueryStatus::BudgetExhausted, "periodic sort move budget");
                bool takeLeft = right == end;
                if (left != middle && right != end) {
                    if (!spend(1)) return ::failure(QueryStatus::BudgetExhausted, "periodic sort comparison budget");
                    ++periodicSortComparisons;
                    takeLeft = std::tie(ordered[left].residue, ordered[left].rank) <=
                               std::tie(ordered[right].residue, ordered[right].rank);
                }
                scratch[out] = takeLeft ? ordered[left++] : ordered[right++];
            }
            begin = end;
        }
        ordered.swap(scratch);
        if (width > ordered.size() / 2) break;
        width *= 2;
    }
    auto last = std::unique(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) {
        return a.phase == b.phase && a.residue == b.residue;
    });
    ordered.erase(last, ordered.end());
    const uint64_t locals = 2 * uint64_t(base.getNumLocalVars()) + 1;
    const uint64_t columns = 2 * uint64_t(n) + symbols + locals + 1;
    const uint64_t rows = 2 * uint64_t(base.getNumConstraints()) + 4;
    if (columns > std::numeric_limits<unsigned>::max() || rows > std::numeric_limits<unsigned>::max())
        return ::failure(QueryStatus::Unsupported, "periodic output representation is too large");
    for (size_t i = 0; i < ordered.size(); ++i) {
        if (!spend(columns * rows))
            return ::failure(QueryStatus::BudgetExhausted, "periodic successor output budget");
        const auto& source = ordered[i];
        const auto& target = ordered[(i + 1) % ordered.size()];
        DynamicAPInt shift = DynamicAPInt(target.residue) - DynamicAPInt(source.residue);
        if (i + 1 == ordered.size()) shift += DynamicAPInt(period);
        IntegerRelation piece(PresburgerSpace::getRelationSpace(n, n, symbols, locals));
        for (unsigned side = 0; side < 2; ++side) {
            for (bool equality : {true, false}) {
                unsigned count = equality ? base.getNumEqualities() : base.getNumInequalities();
                for (unsigned j = 0; j < count; ++j) {
                    auto original = equality ? base.getEquality(j) : base.getInequality(j);
                    llvm::SmallVector<DynamicAPInt> row(columns);
                    // Range qualification above permits exact narrowing here.
                    // Rebuild the small representation: a small numeric value
                    // can otherwise retain a wide-backed APInt after arithmetic.
                    for (unsigned k = 0; k < n; ++k)
                        row[side * n + k] = DynamicAPInt(int64_t(original[k]));
                    for (unsigned k = 0; k < symbols; ++k)
                        row[2 * n + k] = DynamicAPInt(int64_t(original[n + k]));
                    for (unsigned k = 0; k < base.getNumLocalVars(); ++k)
                        row[2 * n + symbols + side * base.getNumLocalVars() + k] =
                            DynamicAPInt(int64_t(original[n + symbols + k]));
                    row.back() = DynamicAPInt(int64_t(original.back()));
                    if (equality) piece.addEquality(row); else piece.addInequality(row);
                }
            }
        }
        llvm::SmallVector<DynamicAPInt> row(columns);
        row[phaseCoordinate] = 1; row.back() = -DynamicAPInt(source.phase);
        piece.addEquality(row);
        std::fill(row.begin(), row.end(), DynamicAPInt(0));
        row[n + phaseCoordinate] = 1; row.back() = -DynamicAPInt(target.phase);
        piece.addEquality(row);
        std::fill(row.begin(), row.end(), DynamicAPInt(0));
        row[iterationCoordinate] = -1; row[n + iterationCoordinate] = 1; row.back() = -shift;
        piece.addEquality(row);
        std::fill(row.begin(), row.end(), DynamicAPInt(0));
        row[iterationCoordinate] = 1; row[columns - 2] = -DynamicAPInt(period);
        row.back() = -DynamicAPInt(source.residue);
        piece.addEquality(row);
        result.unionInPlace(piece);
        ++periodicOutputPieces;
    }
    return {QueryStatus::Proved, std::move(result), {}};
}

RelationResult RelationQueries::staircase(const Relation& latest, const Relation& sourceThrough,
                                         const Relation& targetBefore) {
    auto sources = restrictCandidateOrder(sourceThrough, latest, true);
    if (!sources)
        return sources;
    auto targets = restrictCandidateOrder(targetBefore, latest, false);
    if (!targets)
        return targets;
    auto prefix = compose(*sources.relation, latest);
    if (!prefix) return prefix;
    auto dominated = compose(*prefix.relation, *targets.relation);
    if (!dominated) return dominated;
    return subtract(latest, *dominated.relation);
}

RelationResult RelationQueries::restrictCandidateOrder(const Relation& order, const Relation& candidates, bool sources)
{
    auto endpoints = sources ? candidates.getDomainSet() : candidates.getRangeSet();
    return restrictEndpoints(order, endpoints, endpoints);
}

RelationResult RelationQueries::restrictEndpoints(
    const Relation& order, const mlir::presburger::PresburgerSet& sources,
    const mlir::presburger::PresburgerSet& targets)
{
    QueryTimer timer{"restrictEndpoints", profiling ? &queryProfile.restrictEndpoints : nullptr,
                     order.getNumDisjuncts(), sources.getNumDisjuncts(), targets.getNumDisjuncts()};
    if (order.getNumDomainVars() != sources.getNumRangeVars() || order.getNumRangeVars() != targets.getNumRangeVars())
        return failure(QueryStatus::Unsupported, "candidate endpoint space mismatch");
    if (!sources.getNumDisjuncts() || !targets.getNumDisjuncts()) {
        auto empty = Relation::getEmpty(order.getSpace());
        if (!charge(empty))
            return failure(QueryStatus::BudgetExhausted, "empty endpoint budget");
        return {QueryStatus::Proved, std::move(empty), {}};
    }
    if (!charge(order) || !charge(sources) || !charge(targets))
        return failure(QueryStatus::BudgetExhausted, "candidate endpoint budget");
    struct EndpointIndex {
        std::vector<Constants> values;
        std::map<llvm::DynamicAPInt, std::vector<unsigned>> fixed;
        std::vector<unsigned> wildcard;
        explicit EndpointIndex(const Relation& set) {
            for (const auto& piece : set.getAllDisjuncts()) {
                values.push_back(fixedCoordinates(piece, 0, piece.getNumRangeVars()));
                unsigned index = values.size() - 1;
                if (!values.back().empty() && values.back().front())
                    fixed[*values.back().front()].push_back(index);
                else wildcard.push_back(index);
            }
        }
    };
    EndpointIndex sourceIndex(sources), targetIndex(targets);
    auto result = Relation::getEmpty(order.getSpace());
    auto possible = [&](const Constants& endpoint, const EndpointIndex& index) -> std::optional<bool> {
        auto matches = [&](unsigned candidate) -> std::optional<bool> {
            if (!spend(1)) return {};
            ++endpointComparisons;
            return !incompatible(endpoint, index.values[candidate]);
        };
        if (endpoint.empty() || !endpoint.front()) {
            // An unknown first coordinate can match every fixed bucket. Keep
            // the remaining-coordinate checks; no equality is inferred here.
            for (unsigned i = 0; i < index.values.size(); ++i) {
                auto answer = matches(i);
                if (!answer || *answer) return answer;
            }
        } else {
            if (auto found = index.fixed.find(*endpoint.front()); found != index.fixed.end())
                for (unsigned i : found->second) {
                    auto answer = matches(i);
                    if (!answer || *answer) return answer;
                }
            for (unsigned i : index.wildcard) {
                auto answer = matches(i);
                if (!answer || *answer) return answer;
            }
        }
        return false;
    };
    for (const auto& piece : order.getAllDisjuncts()) {
        if (!spend(1))
            return failure(QueryStatus::BudgetExhausted, "candidate endpoint lookup budget");
        auto left = possible(fixedCoordinates(piece, 0, piece.getNumDomainVars()), sourceIndex);
        if (!left)
            return failure(QueryStatus::BudgetExhausted, "candidate endpoint enumeration budget");
        if (!*left) continue;
        auto right = possible(fixedCoordinates(piece, piece.getNumDomainVars(), piece.getNumRangeVars()), targetIndex);
        if (!right)
            return failure(QueryStatus::BudgetExhausted, "candidate endpoint enumeration budget");
        if (*right) result.unionInPlace(Relation(piece));
    }

    return {QueryStatus::Proved, std::move(result), {}};
}

RelationResult CompletionQueries::selectOrder(
    bool global, const mlir::presburger::PresburgerSet& source, const mlir::presburger::PresburgerSet& target,
    RelationQueries& queries)
{
    const Relation& base = primitive.value();
    if (!source.getSpace().isCompatible(base.getSpace().getDomainSpace()) ||
        !target.getSpace().isCompatible(base.getSpace().getRangeSpace()) ||
        (global && !hasGlobalOrder()))
        return failure(QueryStatus::Unsupported, "issue-order selection space or provider mismatch");
    if (!queries.spend(uint64_t(source.getNumDisjuncts()) + target.getNumDisjuncts() + 1))
        return failure(QueryStatus::BudgetExhausted, "issue-order selection budget");
    if (!source.getNumDisjuncts() || !target.getNumDisjuncts())
        return {QueryStatus::Proved, Relation::getEmpty(base.getSpace()), {}};
    if (orderSelection) {
        if (!*orderSelection)
            return failure(QueryStatus::Unsupported, "missing issue-order selection callback");
        auto value = (*orderSelection)(global, source, target, queries);
        if (value.status == QueryStatus::Proved &&
            (!value.relation || !sameSpace(*value.relation, base)))
            return failure(QueryStatus::Unsupported, "incompatible issue-order selection result");
        return value;
    }
    auto& index = *(global ? indexedGlobal : indexedIssues);
    const auto& original = global ? *globalIssueOrder : *issueOrder;
    if (!index) {
        auto value = queries.normalize(original);
        if (!value)
            return value;
        OrderBlocks blocks;
        for (const auto& piece : value.relation->getAllDisjuncts()) {
            if (!queries.spend(1))
                return failure(QueryStatus::BudgetExhausted, "issue-order index construction budget");
            auto a = fixedCoordinates(piece, 0, piece.getNumDomainVars());
            auto b = fixedCoordinates(piece, piece.getNumDomainVars(), piece.getNumRangeVars());
            auto key = std::make_pair(a.empty() ? std::nullopt : a.front(), b.empty() ? std::nullopt : b.front());
            auto [it, inserted] = blocks.values.try_emplace(key, Relation::getEmpty(original.getSpace()));
            if (inserted) {
                blocks.bySource[key.first].push_back(key);
                blocks.byTarget[key.second].push_back(key);
            }
            it->second.unionInPlace(Relation(piece));
        }
        index = std::move(blocks);
    }
    auto coordinates = [](const Relation& set) {
        std::set<Coordinate> result;
        for (const auto& piece : set.getAllDisjuncts()) {
            auto constants = fixedCoordinates(piece, 0, piece.getNumRangeVars());
            result.insert(constants.empty() ? std::nullopt : constants.front());
        }
        return result;
    };
    auto from = coordinates(source), to = coordinates(target);
    const bool anySource = from.count(std::nullopt), anyTarget = to.count(std::nullopt);
    auto selected = Relation::getEmpty(original.getSpace());
    auto append = [&](const OrderKey& key) {
        ++indexedLookups;
        if (!queries.spend(1)) return false;
        if (auto found = index->values.find(key); found != index->values.end())
            selected.unionInPlace(found->second);
        return true;
    };
    // Unknown coordinates in the ORDER are always possible. Unknown query
    // coordinates select every corresponding group, not just a null-key block.
    from.insert(std::nullopt);
    to.insert(std::nullopt);
    if (!anySource && !anyTarget) {
        for (const auto& a : from) {
            if (!queries.spend(1))
                return failure(QueryStatus::BudgetExhausted, "indexed issue source lookup budget");
            auto group = index->bySource.find(a);
            if (group == index->bySource.end()) continue;
            // A broad finite candidate set can still touch a sparse order.
            // Visit the smaller actual row rather than its empty Cartesian
            // product, while narrow targets use direct block lookups.
            if (group->second.size() < to.size()) {
                for (const auto& key : group->second) {
                    if (!queries.spend(1))
                        return failure(QueryStatus::BudgetExhausted, "indexed issue target filter budget");
                    if (to.count(key.second) && !append(key))
                        return failure(QueryStatus::BudgetExhausted, "indexed issue order budget");
                }
            } else {
                for (const auto& b : to)
                    if (!append({a, b}))
                        return failure(QueryStatus::BudgetExhausted, "indexed issue order budget");
            }
        }
    } else if (anySource && !anyTarget) {
        for (const auto& b : to) {
            if (!queries.spend(1))
                return failure(QueryStatus::BudgetExhausted, "indexed issue target lookup budget");
            if (auto found = index->byTarget.find(b); found != index->byTarget.end())
                for (const auto& key : found->second)
                    if (!append(key))
                        return failure(QueryStatus::BudgetExhausted, "indexed issue order budget");
        }
    } else if (!anySource && anyTarget) {
        for (const auto& a : from) {
            if (!queries.spend(1))
                return failure(QueryStatus::BudgetExhausted, "indexed issue source lookup budget");
            if (auto found = index->bySource.find(a); found != index->bySource.end())
                for (const auto& key : found->second)
                    if (!append(key))
                        return failure(QueryStatus::BudgetExhausted, "indexed issue order budget");
        }
    } else {
        for (const auto& [key, block] : index->values) {
            (void)block;
            if (!append(key))
                return failure(QueryStatus::BudgetExhausted, "indexed issue order budget");
        }
    }
    return queries.restrictEndpoints(selected, source, target);
}

QueryStatus CompletionQueries::addHandoffs(const Relation& additional, RelationQueries& queries)
{
    if (!sparse() || !sameSpace(primitive.value(), additional))
        return QueryStatus::Unsupported;
    if (!additional.getNumDisjuncts())
        return QueryStatus::Proved;
    if (!queries.charge(additional)) return QueryStatus::BudgetExhausted;
    RelationQueries::CompositionRHS incoming(additional);
    auto incomingSources = additional.getDomainSet();
    ++queries.endpointProjections;
    queries.endpointProjectionPieces += additional.getNumDisjuncts();
    uint64_t deferredSourceRelease = 0, deferredCellRelease = 0;
    uint64_t incrementalUpdates = 0, replayUpdates = 0;
    // The update is transactional: work on a copy until the primitive
    // relation can be committed. Avoid copying large deferred receipts just
    // to replace them with the conservative d55a frontier below. Compact
    // materialized states retain their old delta for the exact new-entry
    // update; large/deferred states copy only their reached relation.
    decltype(sources) updated;
    std::set<std::optional<llvm::DynamicAPInt>> conservativeScopes;
    auto isLargeUpdate = [&](const SourceState& state) {
        const uint64_t reachedPieces = state.reached.getNumDisjuncts();
        const uint64_t addedPieces = additional.getNumDisjuncts();
        return reachedPieces > 16 ||
            (addedPieces && reachedPieces > UINT64_MAX / addedPieces) ||
            (reachedPieces * addedPieces > 256);
    };
    for (const auto& [scope, original] : sources) {
        const bool conservative = std::get_if<DeferredFrontier>(&original.frontier) ||
                                   isLargeUpdate(original);
        // Account for the relation state before copying it into the
        // transactional update. Piece count alone is not sufficient: a small
        // disjunct population can still carry a wide constraint matrix.
        if (!queries.charge(original.reached))
            return QueryStatus::BudgetExhausted;
        if (!conservative) {
            const auto* frontier = std::get_if<MaterializedFrontier>(&original.frontier);
            if (!frontier || !queries.charge(frontier->delta))
                return QueryStatus::BudgetExhausted;
            // reachedTargets is a cache of the old reached relation. The new
            // handoffs invalidate it immediately below, so do not copy a
            // potentially large cache into the transactional state only to
            // discard it after the update.
            updated.emplace(scope, SourceState{original.reached,
                MaterializedFrontier{frontier->delta}, original.saturated, std::nullopt});
            continue;
        }
        if (auto* deferred = std::get_if<DeferredFrontier>(&original.frontier)) {
            if (deferredCellRelease > deferredCells ||
                deferred->retainedCells > deferredCells - deferredCellRelease)
                return QueryStatus::BudgetExhausted;
            deferredCellRelease += deferred->retainedCells;
            ++deferredSourceRelease;
        }
        updated.emplace(scope, SourceState{original.reached,
            MaterializedFrontier{Relation::getEmpty(original.reached.getSpace())}, false});
        conservativeScopes.insert(scope);
    }
    for (auto& [scope, state] : updated) {
        mlir::presburger::IntegerRelation filter(primitive.value().getSpace().getDomainSpace());
        if (scope)
            filter.addBound(mlir::presburger::BoundType::EQ, 0, *scope);
        auto before =
            selectOrder(false, mlir::presburger::PresburgerSet(Relation(filter)), incomingSources, queries);
        if (!before)
            return before.status;
        auto initial = queries.compose(*before.relation, incoming);
        if (!initial)
            return initial.status;
        Relation seeds = std::move(*initial.relation);
        // The exact new-entry seed is useful while the frontier is compact.
        // For d55a deferred receipts, or once the reached relation is large,
        // use the old whole-frontier replay instead. This is a sound bounded
        // fallback: it may do more work, but never drops a path and avoids
        // expanding a large R;O;A composition during ordinary repair.
        // Keep the exact new-entry update for genuinely compact states. The
        // ordinary buffering path can have a modest reached-piece count but a
        // large cross product with the newly added handoff; replaying the
        // complete frontier is then the bounded, cheaper representation.
        if (conservativeScopes.count(scope)) {
            ++replayUpdates;
            state.reached.unionInPlace(seeds);
            auto normalized = queries.normalize(state.reached);
            if (!normalized)
                return normalized.status;
            state.reached = std::move(*normalized.relation);
            if (!queries.charge(state.reached))
                return QueryStatus::BudgetExhausted;
            state.frontier = MaterializedFrontier{state.reached};
            state.reachedTargets.reset();
            state.saturated = false;
            continue;
        }
        ++incrementalUpdates;
        // New paths either start through an added handoff, or first enter one
        // after an established completion path. Issue order never supplies
        // completion without a real handoff.
        if (state.reached.getNumDisjuncts()) {
            if (!state.reachedTargets) {
                state.reachedTargets = state.reached.getRangeSet();
                ++queries.endpointProjections;
                queries.endpointProjectionPieces += state.reached.getNumDisjuncts();
            }
            auto between = selectOrder(false, *state.reachedTargets, incomingSources, queries);
            if (!between)
                return between.status;
            auto reachedBefore = queries.compose(state.reached, *between.relation);
            if (!reachedBefore)
                return reachedBefore.status;
            auto throughAdded = queries.compose(*reachedBefore.relation, incoming);
            if (!throughAdded)
                return throughAdded.status;
            seeds.unionInPlace(*throughAdded.relation);
        }
        if (!queries.charge(seeds))
            return QueryStatus::BudgetExhausted;
        state.reached.unionInPlace(seeds);
        auto normalizedReached = queries.normalize(state.reached);
        if (!normalizedReached)
            return normalizedReached.status;
        state.reached = std::move(*normalizedReached.relation);
        state.reachedTargets.reset();
        auto* frontier = std::get_if<MaterializedFrontier>(&state.frontier);
        if (!frontier) {
            return QueryStatus::Unsupported;
        }
        frontier->delta.unionInPlace(seeds);
        auto normalizedPending = queries.normalize(frontier->delta);
        if (!normalizedPending)
            return normalizedPending.status;
        frontier->delta = std::move(*normalizedPending.relation);
        state.saturated = frontier->delta.getNumDisjuncts() == 0;
    }
    if (deferredSourceRelease > deferredSources || deferredCellRelease > deferredCells)
        return QueryStatus::BudgetExhausted;
    if (!queries.charge(primitive.value()) || !queries.charge(additional))
        return QueryStatus::BudgetExhausted;
    primitive.relation.unionInPlace(additional);
    primitive.index.reset();
    primitiveSources.reset();
    primitiveTargets.reset();
    known = primitive.value();
    sources = std::move(updated);
    deferredSources -= deferredSourceRelease;
    deferredCells -= deferredCellRelease;
    frontierResets += deferredSourceRelease;
    frontierIncrementalUpdates += incrementalUpdates;
    frontierReplayUpdates += replayUpdates;
    transitions.reset();
    expanded = Relation::getEmpty(primitive.value().getSpace());
    fixed = false;
    return QueryStatus::Proved;
}

const mlir::presburger::PresburgerSet& CompletionQueries::primitiveEndpoints(
    bool sources, RelationQueries& queries)
{
    auto& cached = sources ? primitiveSources : primitiveTargets;
    if (!cached) {
        cached = sources ? primitive.value().getDomainSet() : primitive.value().getRangeSet();
        ++queries.endpointProjections;
        queries.endpointProjectionPieces += primitive.value().getNumDisjuncts();
    }
    return *cached;
}

QueryStatus CompletionQueries::resolveFrontier(SourceState& state, RelationQueries& queries)
{
    auto* deferred = std::get_if<DeferredFrontier>(&state.frontier);
    if (!deferred) return QueryStatus::Proved;
    ++frontierDifferences;
    auto fresh = queries.subtract(deferred->generated, deferred->before);
    if (!fresh) return fresh.status;
    if (!queries.charge(*fresh.relation) || !queries.charge(deferred->before))
        return QueryStatus::BudgetExhausted;
    bool empty = fresh.relation->isIntegerEmpty();
    Relation reached = deferred->before;
    reached.unionInPlace(*fresh.relation);
    // Commit only after all fallible operations. The set is unchanged, but
    // R_before union delta restores the semi-naive representation instead of
    // indefinitely retaining overlapping generated waves.
    deferredCells -= deferred->retainedCells;
    --deferredSources;
    ++frontierResolutions;
    state.reached = std::move(reached);
    state.frontier = MaterializedFrontier{std::move(*fresh.relation)};
    state.reachedTargets.reset();
    state.saturated = empty;
    return QueryStatus::Proved;
}

QueryStatus CompletionQueries::proveSparse(const Relation& requirement, RelationQueries& queries, unsigned rounds)
{
    using Scope = std::optional<llvm::DynamicAPInt>;
    std::map<Scope, Relation> demands;
    for (const auto& piece : requirement.getAllDisjuncts()) {
        auto coordinates = fixedCoordinates(piece, 0, piece.getNumDomainVars());
        Scope key = coordinates.empty() ? std::nullopt : coordinates.front();
        auto [it, inserted] = demands.try_emplace(key, Relation::getEmpty(requirement.getSpace()));
        (void)inserted;
        it->second.unionInPlace(Relation(piece));
    }
    expanded = Relation::getEmpty(requirement.getSpace());
    QueryStatus overall = QueryStatus::Proved;
    for (const auto& [scope, required] : demands) {
        auto found = sources.find(scope);
        if (found == sources.end()) {
            // Cache the FULL issue-order source scope, not just this demand's
            // possibly narrower guard/invocation domain. O itself retains all
            // qualified execution conditions; this filter never invents edges.
            mlir::presburger::IntegerRelation filter(primitive.value().getSpace().getDomainSpace());
            if (scope)
                filter.addBound(mlir::presburger::BoundType::EQ, 0, *scope);
            auto before = selectOrder(
                false, mlir::presburger::PresburgerSet(Relation(filter)), primitiveEndpoints(true, queries), queries);
            if (!before)
                return before.status;
            auto initial = queries.compose(*before.relation, primitive);
            if (!initial)
                return initial.status;
            if (!queries.spend(1) || !queries.charge(*initial.relation)) return QueryStatus::BudgetExhausted;
            Relation pending = *initial.relation;
            found = sources.emplace(scope, SourceState{std::move(*initial.relation),
                MaterializedFrontier{std::move(pending)}, false}).first;
        }
        auto& state = found->second;
        // These sets describe this exact demand, whereas the cached source
        // reachability above retains the complete qualified occurrence scope.
        const auto requiredSources = required.getDomainSet();
        const auto requiredTargets = required.getRangeSet();
        Relation available = Relation::getEmpty(requirement.getSpace());
        auto coverage = [&](const Relation& reached,
                            std::optional<mlir::presburger::PresburgerSet>& reachedTargets,
                            Relation& supplied) -> QueryStatus {
            if (!reachedTargets) {
                reachedTargets = reached.getRangeSet();
                ++queries.endpointProjections;
                queries.endpointProjectionPieces += reached.getNumDisjuncts();
            }
            auto after = selectOrder(false, *reachedTargets, requiredTargets, queries);
            if (!after)
                return after.status;
            auto value = queries.compose(reached, *after.relation);
            if (!value)
                return value.status;
            supplied = std::move(*value.relation);
            return queries.contains(supplied, required);
        };
        auto covered = coverage(state.reached, state.reachedTargets, available);
        if (covered == QueryStatus::NotEstablished && !state.saturated && hasGlobalOrder()) {
            // Rejection-only upper bounds: every completion path must enter
            // and leave through a real handoff. Global issue order is never
            // inserted into reached completion or used to claim saturation.
            auto before = selectOrder(true, requiredSources, primitiveEndpoints(true, queries), queries);
            auto after = selectOrder(false, primitiveEndpoints(false, queries), requiredTargets, queries);
            if (!before || !after)
                return !before ? before.status : after.status;
            auto prefix = queries.compose(*before.relation, primitive);
            if (!prefix)
                return prefix.status;
            auto possible = queries.compose(*prefix.relation, *after.relation);
            if (!possible)
                return possible.status;
            auto necessary = queries.contains(*possible.relation, required);
            if (necessary == QueryStatus::Proved) {
                auto firstBefore = selectOrder(false, requiredSources, primitiveEndpoints(true, queries), queries);
                auto firstAfter = selectOrder(true, primitiveEndpoints(false, queries), requiredTargets, queries);
                if (!firstBefore || !firstAfter)
                    return !firstBefore ? firstBefore.status : firstAfter.status;
                auto first = queries.compose(*firstBefore.relation, primitive);
                if (!first)
                    return first.status;
                auto possibleFirst = queries.compose(*first.relation, *firstAfter.relation);
                if (!possibleFirst)
                    return possibleFirst.status;
                necessary = queries.contains(*possibleFirst.relation, required);
            }
            if (necessary == QueryStatus::Unsupported || necessary == QueryStatus::BudgetExhausted)
                return necessary;
            if (necessary == QueryStatus::NotEstablished) {
                expanded.unionInPlace(available);
                overall = QueryStatus::NotEstablished;
                continue;
            }
        }
        // Previous squaring allowed 2^rounds handoffs. Delta propagation uses
        // that many extensions per query without constructing unrelated roots.
        // Resumed queries retain earlier reachability and pending work.
        unsigned steps = (1u << std::min(rounds, 8u)) - 1;
        for (unsigned step = 0; covered == QueryStatus::NotEstablished && !state.saturated && step < steps; ++step) {
            // A zero-round query makes no extension request and retains this
            // receipt. Otherwise resolve it BEFORE composing, without consuming
            // a round. Exact empty novelty, never early coverage, saturates.
            auto resolved = resolveFrontier(state, queries);
            if (resolved != QueryStatus::Proved) return resolved;
            if (state.saturated) break;
            if (!transitions) {
                auto between = selectOrder(false, primitiveEndpoints(false, queries), primitiveEndpoints(true, queries), queries);
                if (!between)
                    return between.status;
                auto value = queries.compose(*between.relation, primitive);
                if (!value)
                    return value.status;
                transitions.emplace(std::move(*value.relation));
            }
            const auto& pending = std::get<MaterializedFrontier>(state.frontier).delta;
            auto next = queries.compose(pending, *transitions);
            if (!next) return next.status;
            ++frontierExtensions;
            uint64_t oldCost = 0, generatedCost = 0;
            if (!queries.spend(1) || !queries.charge(state.reached, &oldCost) ||
                !queries.charge(*next.relation, &generatedCost)) return QueryStatus::BudgetExhausted;
            Relation candidate = state.reached;
            candidate.unionInPlace(*next.relation);
            std::optional<mlir::presburger::PresburgerSet> candidateTargets;
            Relation candidateSupply = Relation::getEmpty(requirement.getSpace());
            auto candidateCovered = coverage(candidate, candidateTargets, candidateSupply);
            if (candidateCovered == QueryStatus::Unsupported || candidateCovered == QueryStatus::BudgetExhausted)
                return candidateCovered;
            if (candidateCovered == QueryStatus::Proved) {
                // Both relations are moved into the one owning receipt for
                // this source. Its pre-image is R, not the enlarged R union N.
                uint64_t cells = oldCost + generatedCost - 2;
                if (cells > UINT64_MAX - deferredCells) return QueryStatus::BudgetExhausted;
                DeferredFrontier receipt{std::move(*next.relation), std::move(state.reached), cells};
                state.reached = std::move(candidate);
                state.frontier = std::move(receipt);
                state.reachedTargets = std::move(candidateTargets);
                ++frontierDeferrals;
                ++deferredSources;
                deferredCells += cells;
                peakDeferredCells = std::max(peakDeferredCells, deferredCells);
                available = std::move(candidateSupply);
                covered = QueryStatus::Proved;
                break;
            }
            ++frontierDifferences;
            auto fresh = queries.subtract(*next.relation, state.reached);
            if (!fresh) return fresh.status;
            if (!queries.charge(*fresh.relation) || !queries.charge(state.reached))
                return QueryStatus::BudgetExhausted;
            bool empty = fresh.relation->isIntegerEmpty();
            Relation reached = state.reached;
            reached.unionInPlace(*fresh.relation);
            state.reached = std::move(reached);
            state.frontier = MaterializedFrontier{std::move(*fresh.relation)};
            state.reachedTargets.reset();
            state.saturated = empty;
            // R union (N \ R) equals the candidate already checked. Keep its
            // exact answer/supply, avoiding another equivalent coverage query.
            available = std::move(candidateSupply);
            covered = candidateCovered;
        }
        if (covered == QueryStatus::Unsupported || covered == QueryStatus::BudgetExhausted)
            return covered;
        expanded.unionInPlace(available);
        if (covered != QueryStatus::Proved)
            overall = covered;
    }
    // A saturated source query says nothing about unqueried source scopes.
    // The public whole-plan fixed-point bit is therefore never set here.
    return overall;
}

QueryStatus CompletionQueries::prove(const Relation& requirement, RelationQueries& queries, unsigned rounds) {
    if (sparse())
        return proveSparse(requirement, queries, rounds);
    auto covered = queries.contains(known, requirement);
    if (covered != QueryStatus::NotEstablished || fixed)
        return covered;
    for (unsigned round = 0; round < rounds; ++round) {
        auto paths = queries.compose(known, known);
        if (!paths)
            return paths.status;
        auto added = queries.subtract(*paths.relation, known);
        if (!added)
            return added.status;
        if (added.relation->isIntegerEmpty()) {
            fixed = true;
            return QueryStatus::NotEstablished;
        }
        known.unionInPlace(*added.relation);
        covered = queries.contains(known, requirement);
        if (covered != QueryStatus::NotEstablished)
            return covered;
    }
    return QueryStatus::NotEstablished;
}
