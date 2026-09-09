// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/InsertSync/LogicalSyncPlan.h"
#include "PTO/Transforms/InsertSync/SyncPhysicalFacts.h"
#include "PTO/Transforms/InsertSync/SyncGlobalOccurrences.h"
#include "PTO/Transforms/InsertSync/MemoryDependentAnalyzer.h"
#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"
#include "PTO/Transforms/InsertSync/SyncEffectCoverage.h"
#include "PTO/Transforms/InsertSync/SyncPayloadSnapshot.h"
#include "mlir/Analysis/FlatLinearValueConstraints.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <map>
#include <set>
#include <cstdlib>
#include <limits>
#include <chrono>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::logical_sync;
using namespace mlir::presburger;
namespace {
using Lane = PipelineType;
using Domain = std::pair<Lane, Lane>;
using Requirement = OrderingRequirement;
struct Stream {
    Domain pipes;
    Relation matching;
    unsigned source;
    int key = -1;
};
struct Barrier {
    unsigned point;
    Lane pipe;
    PresburgerSet domain;
    std::optional<Relation> logical;
};
struct RetirementSink {
    // Stable occurrence identity for the original return. Deliberately absent
    // from the physical lane vector and ordinary handoff completion relation.
    unsigned point = 0;
};
// One immutable occurrence universe, shared by all plan versions. Only the
// exact phase-pair cache is mutable; it contains issue order, never completion.
// Fixed phase coordinates index requests. Unknown coordinates retain the whole
// relevant universe, rather than becoming a false empty answer.
struct NativeOrder {
    SyncOccurrences facts;
    std::vector<Lane> lanes;
    std::map<Lane, SmallVector<unsigned>> byLane;
    std::map<Lane, PresburgerSet> domains;
    std::map<unsigned, unsigned> clearPredecessor;
    std::map<std::tuple<unsigned, unsigned, bool>, Relation> cache;
    uint64_t requests = 0, built = 0, selections = 0;
    uint64_t acquisitionProjections = 0, acquisitionCandidates = 0;
    struct AcquisitionTargets {
        PresburgerSet domain;
        std::set<unsigned> phases;
    };
    Relation empty() const {
        return Relation::getEmpty(PresburgerSpace::getRelationSpace(
            facts.dimensions(), facts.dimensions(), facts.parameters.size()));
    }
    RelationResult get(unsigned p, unsigned q, bool inclusive, RelationQueries& queries) {
        ++requests;
        if (!queries.spend(1))
            return {QueryStatus::BudgetExhausted, {}, "native order lookup budget"};
        auto key = std::make_tuple(p, q, inclusive);
        if (auto found = cache.find(key); found != cache.end())
            return {QueryStatus::Proved, found->second, {}};
        ++built;
        auto value = facts.ordered(p, q, inclusive);
        if (value) value = queries.normalize(*value.relation);
        if (value) cache.emplace(key, *value.relation);
        return value;
    }
    std::optional<std::set<unsigned>> phaseIds(const PresburgerSet& domain, RelationQueries& queries) const {
        std::set<unsigned> ids;
        for (const auto& piece : domain.getAllDisjuncts()) {
            if (!queries.spend(uint64_t(piece.getNumEqualities() + 1) * piece.getNumCols())) return {};
            std::optional<llvm::DynamicAPInt> fixed;
            for (unsigned r = 0; r < piece.getNumEqualities(); ++r) {
                auto row = piece.getEquality(r);
                if (row[0] != 1 && row[0] != -1) continue;
                bool isolated = true;
                for (unsigned c = 1; c < piece.getNumVars(); ++c)
                    if (row[c] != 0) { isolated = false; break; }
                if (isolated) { fixed = row[0] == 1 ? -row.back() : row.back(); break; }
            }
            if (!fixed) {
                // Unknown coordinates retain every candidate, but that dense
                // output still consumes work. Refuse before allocating it if
                // the remaining allowance cannot pay for all phase entries.
                if (!queries.spend(lanes.size())) return {};
                for (unsigned p = 0; p < lanes.size(); ++p) ids.insert(ids.end(), p);
                return ids;
            }
            if (*fixed >= 0 && *fixed < int64_t(lanes.size())) ids.insert(unsigned(int64_t(*fixed)));
        }
        return ids;
    }
    std::optional<AcquisitionTargets> acquisitionTargets(const Relation& matching, RelationQueries& queries)
    {
        ++acquisitionProjections;
        auto domain = matching.getRangeSet();
        auto ids = phaseIds(domain, queries);
        if (!ids) return {};
        return AcquisitionTargets{std::move(domain), std::move(*ids)};
    }
    RelationResult select(const PresburgerSet& sources, const PresburgerSet& targets,
                          bool sameLane, bool inclusive, RelationQueries& queries) {
        ++selections;
        auto from = phaseIds(sources, queries), to = phaseIds(targets, queries);
        if (!from || !to) return {QueryStatus::BudgetExhausted, {}, "native order endpoint budget"};
        auto result = empty();
        std::map<Lane, SmallVector<unsigned>> destinations;
        if (sameLane)
            for (unsigned q : *to) destinations[lanes[q]].push_back(q);
        auto append = [&](unsigned p, unsigned q) {
            auto edge = get(p, q, inclusive, queries);
            if (!edge) return edge;
            result.unionInPlace(*edge.relation);
            return RelationResult{QueryStatus::Proved, {}, {}};
        };
        for (unsigned p : *from) {
            if (sameLane) {
                auto found = destinations.find(lanes[p]);
                if (found == destinations.end()) continue;
                for (unsigned q : found->second) {
                    auto status = append(p, q);
                    if (status.status != QueryStatus::Proved) return status;
                }
            } else for (unsigned q : *to) {
                auto status = append(p, q);
                if (status.status != QueryStatus::Proved) return status;
            }
        }
        // Prune fixed invocation coordinates before returning to composition.
        // This is a conservative filter, not projection or completion supply.
        return queries.restrictEndpoints(result, sources, targets);
    }
};
// Timings are diagnostic only. Preserve independent final reconstruction;
// expose its cost separately from proposals and optional simplification.
class StageProfile {
    RelationQueries& queries;
    const bool enabled;
    const char* label;
    uint64_t work;
    std::chrono::steady_clock::time_point started;
public:
    StageProfile(RelationQueries& q, const char* label) : queries(q), enabled(q.profilingEnabled()),
        label(label), work(q.work()) { if (enabled) started = std::chrono::steady_clock::now(); }
    void next(const char* name) {
        if (enabled) {
            auto now = std::chrono::steady_clock::now();
            llvm::errs() << "logical substage " << label << " work " << queries.work() - work
                         << " seconds " << std::chrono::duration<double>(now - started).count() << "\n";
            started = now; work = queries.work();
        }
        label = name;
    }
    ~StageProfile() { next("finished"); }
};
class Constructor {
    func::FuncOp function;
    SyncPayloadSnapshot payload;
    llvm::function_ref<void(func::FuncOp)> emissionMutation;
    testing::RequirementObserver observeRequirements;
    InsertSyncGMAliasMode gm;
    SyncIRs ir;
    Buffer2MemInfoMap buffers;
    MemoryDependentAnalyzer memory;
    RelationQueries queries;
    std::shared_ptr<NativeOrder> orders = std::make_shared<NativeOrder>();
    SyncOccurrences& facts = orders->facts;
    std::vector<const CompoundInstanceElement*> phases;
    std::vector<Lane>& lanes = orders->lanes;
    std::vector<Requirement> requirements;
    std::vector<Stream> streams;
    std::vector<Barrier> barriers;
    std::vector<Relation> targetRequirements;
    std::map<std::pair<unsigned, unsigned>, Relation> pairRequirements;
    // Physical slot order and normalized selector domains are input facts,
    // independent of the handoffs eventually selected. Unknown mappings or
    // scalar ranges retain the original conservative access relation.
    std::map<const BaseMemInfo*, std::optional<SyncPhysicalSlotMapping>> slotMappings;
    SmallVector<Value> slotSelectors;
    using SelectorPoint = std::pair<Value, std::pair<unsigned, unsigned>>;
    llvm::DenseMap<SelectorPoint, bool> selectorRanges;
    llvm::DenseMap<SelectorPoint, std::optional<std::vector<Relation>>> selectorDomains;
    std::map<std::pair<unsigned, const BaseMemInfo*>, std::vector<Relation>> usedSlotDomains;
    llvm::SmallPtrSet<const BaseMemInfo*, 16> compactSlotMemories;
    llvm::DenseMap<std::pair<SelectorPoint, SelectorPoint>, std::optional<Relation>> compactSlotRelations;
    std::set<std::pair<unsigned, const BaseMemInfo*>> usedCompactSlots;
    uint64_t slotRangeBuilds = 0, slotEqualityBuilds = 0, slotDomainsBuilt = 0, slotGeometryBuilds = 0;
    uint64_t slotComponentNodes = 0, slotComponentEdges = 0, slotComponentFinds = 0, slotComponentJoins = 0;
    std::optional<CompletionQueries> completionOrder;
    ConstructionResult result;
    RetirementSink retirement;
    // Requirements are immutable after discovery. Receipts certify all of a
    // target's requirements for this selected logical plan version. Adding a
    // barrier preserves existing receipts; deletion needs a fresh full proof.
    uint64_t planVersion = 0, receiptVersion = 0;
    std::set<unsigned> provedTargets;

    bool fail(ConstructionResult::Status status, StringRef reason)
    {
        result.status = status;
        result.reason = reason.str();
        return false;
    }
    bool expect(QueryStatus status, StringRef reason)
    {
        if (status == QueryStatus::Proved)
            return true;
        return fail(
            status == QueryStatus::BudgetExhausted ? ConstructionResult::AnalysisLimit :
            status == QueryStatus::Unsupported     ? ConstructionResult::Unsupported :
                                                     ConstructionResult::Unproved,
            reason);
    }
    bool queryFailed(QueryStatus status, StringRef reason)
    {
        if (status == QueryStatus::Proved || status == QueryStatus::NotEstablished)
            return false;
        expect(status, reason);
        return true;
    }
    Relation empty() const
    {
        return Relation::getEmpty(
            PresburgerSpace::getRelationSpace(facts.dimensions(), facts.dimensions(), facts.parameters.size()));
    }
    std::optional<Relation> take(RelationResult answer)
    {
        if (!answer) {
            fail(
                answer.status == QueryStatus::BudgetExhausted ? ConstructionResult::AnalysisLimit :
                answer.status == QueryStatus::Unsupported     ? ConstructionResult::Unsupported :
                                                                ConstructionResult::Unproved,
                answer.reason);
            return {};
        }
        return std::move(answer.relation);
    }
    std::optional<Relation> compose(const Relation& a, const Relation& b) { return take(queries.compose(a, b)); }
    std::optional<Relation> order(unsigned p, unsigned q, bool inclusive = false)
    { return take(orders->get(p, q, inclusive, queries)); }
    std::optional<Relation> selectedOrder(const PresburgerSet& sources, const PresburgerSet& targets,
                                         bool inclusive = false)
    { return take(orders->select(sources, targets, true, inclusive, queries)); }
    const PresburgerSet& laneDomain(Lane pipe) const { return orders->domains.at(pipe); }
    Relation identity() const
    {
        auto value = empty();
        for (unsigned p = 0; p < lanes.size(); ++p)
            value.unionInPlace(facts.identity(p));
        return value;
    }
    std::optional<CompletionQueries> completion(Relation handoffs)
    {
        if (!completionOrder) {
            auto owner = orders;
            completionOrder.emplace(empty(),
                [owner](bool global, const PresburgerSet& source, const PresburgerSet& target,
                        RelationQueries& queries) {
                    return owner->select(source, target, !global, true, queries);
                }, true);
        }
        return completionOrder->withHandoffs(std::move(handoffs));
    }
    bool discover()
    {
        auto admission = qualifySyncPhysicalAddresses(function);
        if (admission.status == SyncAddressAdmission::Rejected)
            return fail(ConstructionResult::Unsupported, admission.reason);
        if (!supportsLogicalSyncTranslation(function))
            return fail(ConstructionResult::Unsupported, "memory loop forwarding requires qualified translation");
        PTOIRTranslator translator(ir, memory, buffers, function, SyncAnalysisMode::NORMALSYNC);
        if (failed(translator.Build()))
            return fail(ConstructionResult::Unsupported, "physical translation failed");
        auto coverage = inspectInsertSyncEffectCoverage(function, ir, false);
        if (failed(coverage))
            return fail(ConstructionResult::InternalError, "effect inspection failed");
        if (!*coverage)
            return fail(ConstructionResult::Unsupported, "unqualified physical effects");
        auto physical = importSyncPhysicalFacts(function, ir, queries.remainingWork());
        queries.spend(physical.work);
        if (physical.status != SyncPhysicalFacts::Status::Complete)
            return fail(
                physical.status == SyncPhysicalFacts::Status::InternalError ? ConstructionResult::InternalError :
                physical.status == SyncPhysicalFacts::Status::AnalysisLimit ? ConstructionResult::AnalysisLimit :
                                                                                    ConstructionResult::Unsupported,
                physical.reason);
        // Construction does not consume the exact-slice lifecycle projection:
        // partially overlapping translated accesses must retain their conflicts.
        phases = physical.phases;
        SmallVector<Operation*> points;
        for (auto* phase : phases) {
            if (!phase || !phase->elementOp || phase->macroOpInstanceId >= 0)
                return fail(ConstructionResult::Unsupported, "multi-phase operation occurrence summary");
            points.push_back(phase->elementOp);
            lanes.push_back(phase->kPipeValue);
        }
        if (physical.lifetimeScope != function.getOperation())
            return fail(ConstructionResult::Unsupported, "physical section exit lowering not established");
        if (!llvm::hasSingleElement(function.getBody()) ||
            !isa<func::ReturnOp>(function.getBody().front().getTerminator()))
            return fail(ConstructionResult::Unsupported, "logical function exit shape");
        retirement.point = points.size();
        points.push_back(function.getBody().front().getTerminator());
        llvm::SmallDenseSet<Value> seenSelectors;
        for (const auto* phase : phases) {
            auto collect = [&](const auto& entries) {
                for (const auto* mem : entries) {
                    if (slotMappings.count(mem)) continue;
                    // Literal gets still validate/sort their complete root
                    // slot table. Charge that work before qualification.
                    if (!queries.spend(estimateSyncPhysicalSlotQualificationWork(mem, buffers))) return false;
                    auto mapping = qualifySyncPhysicalSlots(mem, buffers);
                    if (mapping && mapping->selector && seenSelectors.insert(mapping->selector).second)
                        slotSelectors.push_back(mapping->selector);
                    slotMappings.emplace(mem, std::move(mapping));
                }
                return true;
            };
            if (!collect(phase->defVec) || !collect(phase->useVec))
                return fail(ConstructionResult::AnalysisLimit, "physical slot mapping budget");
        }
        facts = SyncOccurrences::build(function, points, slotSelectors);
        if (!queries.spend(facts.work))
            return fail(ConstructionResult::AnalysisLimit, "occurrence import work budget");
        if (!facts.complete)
            return fail(
                facts.limitExceeded ? ConstructionResult::AnalysisLimit : ConstructionResult::Unsupported,
                facts.reason);
        for (unsigned p = 0; p < lanes.size(); ++p) {
            orders->byLane[lanes[p]].push_back(p);
            auto [it, inserted] = orders->domains.try_emplace(lanes[p],
                PresburgerSet::getEmpty(facts.domain(p).getRangeSet().getSpace()));
            it->second.unionInPlace(facts.domain(p).getRangeSet());
        }
        // Original clear-interval neighbours are immutable input facts. Reuse
        // them for proposals and independently compare them with emitted cuts.
        std::set<Block*> originalBlocks;
        for (const auto& point : facts.points) originalBlocks.insert(point.operation->getBlock());
        for (Block* block : originalBlocks) {
            std::map<Lane, unsigned> previous;
            for (auto& op : *block) {
                if (op.getNumRegions()) { previous.clear(); continue; }
                auto found = facts.ids.find(&op);
                if (found == facts.ids.end() || found->second >= lanes.size()) continue;
                unsigned p = found->second;
                if (auto last = previous.find(lanes[p]); last != previous.end())
                    orders->clearPredecessor.emplace(p, last->second);
                previous[lanes[p]] = p;
            }
        }
        auto add = [&](unsigned p, unsigned q, const BaseMemInfo* a, const BaseMemInfo* b,
                       Requirement::Kind kind) -> bool {
            if (a && b && !logicalSyncMayAlias(a, b, function, gm))
                return true;
            if (a && b && disjointInsertSyncGlobalOccurrences(a, points[p], b, points[q], function))
                return true;
            auto relation = order(p, q);
            if (!relation)
                return false;
            if (a && b) {
                relation = slotConflicts(p, q, a, b, *relation);
                if (!relation) return false;
            }
            if (!relation->isIntegerEmpty())
                requirements.push_back({kind, p, q, a, b, std::move(*relation)});
            return true;
        };
        // Index qualified physical overlap once; retain the original alias
        // contract and occurrence query for every conservative candidate.
        std::vector<SyncPhysicalAccess> accesses;
        for (unsigned p = 0; p < phases.size(); ++p) {
            for (const auto* mem : phases[p]->defVec) accesses.push_back({p, mem, true, lanes[p]});
            for (const auto* mem : phases[p]->useVec) accesses.push_back({p, mem, false, lanes[p]});
        }
        auto candidates = enumerateSyncAccessCandidates(accesses, [&](uint64_t work) { return queries.spend(work); });
        if (candidates.status != SyncAccessCandidates::Status::Complete)
            return fail(candidates.status == SyncAccessCandidates::Status::AnalysisLimit ?
                ConstructionResult::AnalysisLimit : ConstructionResult::InternalError, "storage candidate enumeration");
        if (queries.profilingEnabled())
            llvm::errs() << "logical discovery intervals " << candidates.intervalVisits
                << " candidate_visits " << candidates.candidateVisits << " pairs " << candidates.pairs.size() << "\n";
        if (!chooseSlotRepresentations(accesses, candidates.pairs)) return false;
        for (const auto& [x, y] : candidates.pairs) {
            const auto& a = accesses[x]; const auto& b = accesses[y];
            auto kind = [](bool sourceWrite, bool targetWrite) {
                return sourceWrite ? (targetWrite ? Requirement::WAW : Requirement::RAW) :
                                     (targetWrite ? Requirement::WAR : Requirement::AccResource);
            };
            if (!queries.spend(1)) return fail(ConstructionResult::AnalysisLimit, "storage candidates budget");
            if (!add(a.phase, b.phase, a.memory, b.memory, kind(a.write, b.write))) return false;
            if (x != y && !add(b.phase, a.phase, b.memory, a.memory, kind(b.write, a.write))) return false;
        }
        for (unsigned p = 0; p < phases.size(); ++p)
            if (!add(p, retirement.point, nullptr, nullptr, Requirement::Retirement))
                return false;
        // Keep every typed obligation immutable. Index their union once for
        // planning/proof clients; never drop a distinct occurrence relation
        // merely because its static source/target pair has already appeared.
        for (const auto& r : requirements) if (r.kind != Requirement::Retirement) {
            auto [it, inserted] = pairRequirements.try_emplace({r.source, r.target}, empty());
            it->second.unionInPlace(r.occurrences);
        }
        targetRequirements.assign(lanes.size(), empty());
        for (auto& [pair, relation] : pairRequirements) {
            auto normalized = take(queries.normalize(relation));
            if (!normalized) return false;
            relation = std::move(*normalized);
            targetRequirements[pair.second].unionInPlace(relation);
        }
        result.requirements = requirements.size();
        if (observeRequirements)
            observeRequirements(facts, phases, requirements);
        return true;
    }
    // Range proof is separate from enumerating slot domains. An exact compact
    // equality must not pay for N residue domains just to qualify its inputs.
    // multi_tile_get uses slot zero on an out-of-range dynamic index, so this
    // proof covers the entire access domain, not only the conflict subset.
    bool proveSlotRange(
        const SyncOccurrences& occurrences, unsigned point, const SyncPhysicalSlotMapping& mapping)
    {
        ++slotRangeBuilds;
        auto range = occurrences.scalarDomain(mapping.selector, point, 0, mapping.bases.size() - 1, queries);
        if (!range) {
            if (range.status == QueryStatus::BudgetExhausted)
                fail(ConstructionResult::AnalysisLimit, "slot selector domain budget");
            return false;
        }
        auto coverage = queries.contains(*range.relation, occurrences.domain(point));
        if (coverage != QueryStatus::Proved) {
            if (coverage == QueryStatus::BudgetExhausted)
                fail(ConstructionResult::AnalysisLimit, "slot selector range budget");
            return false;
        }
        return true;
    }
    bool slotInRange(unsigned point, const SyncPhysicalSlotMapping& mapping)
    {
        if (!queries.spend(1)) return fail(ConstructionResult::AnalysisLimit, "slot range lookup budget");
        SelectorPoint key{mapping.selector, {point, unsigned(mapping.bases.size())}};
        if (auto found = selectorRanges.find(key); found != selectorRanges.end()) return found->second;
        bool proved = proveSlotRange(facts, point, mapping);
        if (result.status != ConstructionResult::AnalysisLimit) selectorRanges.try_emplace(key, proved);
        return proved;
    }
    // Caller has a complete-domain range receipt for this selector/point/count.
    std::optional<std::vector<Relation>> enumerateSlotDomains(
        const SyncOccurrences& occurrences, unsigned point, const SyncPhysicalSlotMapping& mapping)
    {
        std::vector<Relation> domains;
        for (unsigned slot = 0; slot < mapping.bases.size(); ++slot) {
            ++slotDomainsBuilt;
            auto domain = occurrences.scalarDomain(mapping.selector, point, slot, slot, queries);
            if (!domain) {
                if (domain.status == QueryStatus::BudgetExhausted)
                    fail(ConstructionResult::AnalysisLimit, "slot equality domain budget");
                return {};
            }
            domains.push_back(std::move(*domain.relation));
        }
        return domains;
    }
    const std::vector<Relation>* slots(unsigned point, const BaseMemInfo* memory)
    {
        const auto& mapping = slotMappings.at(memory);
        if (!mapping) return nullptr;
        auto key = std::make_pair(point, memory);
        if (auto found = usedSlotDomains.find(key); found != usedSlotDomains.end()) return &found->second;
        std::optional<std::vector<Relation>> domains;
        if (mapping->selector) {
            // A shared SSA value can select roots of different depths. Range
            // qualification is depth-specific; do not reuse the wrong count.
            auto [found, inserted] = selectorDomains.try_emplace(
                SelectorPoint{mapping->selector, {point, unsigned(mapping->bases.size())}}, std::nullopt);
            if (inserted && slotInRange(point, *mapping))
                found->second = enumerateSlotDomains(facts, point, *mapping);
            domains = found->second;
        } else domains = std::vector<Relation>{facts.domain(point)};
        if (!domains) return nullptr;
        return &usedSlotDomains.emplace(key, std::move(*domains)).first->second;
    }
    bool sameSlotGeometry(const BaseMemInfo* a, const BaseMemInfo* b)
    {
        ++slotGeometryBuilds;
        const auto& ma = *slotMappings.at(a); const auto& mb = *slotMappings.at(b);
        bool same = ma.scope == mb.scope && ma.bytes == mb.bytes && ma.bases.size() == mb.bases.size();
        if (same) {
            if (!queries.spend(ma.bases.size()))
                return fail(ConstructionResult::AnalysisLimit, "slot table comparison budget");
            same = ma.bases == mb.bases;
        }
        // Both selector mappings were independently qualified, including
        // disjoint intervals within each ORIGINAL ordered table. A permutation
        // or partial overlap cannot use ordinal equality and stays general.
        return same;
    }
    bool chooseSlotRepresentations(ArrayRef<SyncPhysicalAccess> accesses,
                                    ArrayRef<std::pair<unsigned, unsigned>> candidates)
    {
        if (slotSelectors.empty()) return true;
        // Reuse the existing conservative overlap graph, not another all-pairs
        // mapping scan. Freeze representation BEFORE forming any requirement:
        // mixing equality and enumerated forms in one overlapping component
        // can make exact downstream difference substantially more expensive.
        if (!queries.spend(uint64_t(accesses.size()) * 4))
            return fail(ConstructionResult::AnalysisLimit, "slot component setup budget");
        struct Node { unsigned parent, rank; const BaseMemInfo* memory; bool homogeneous; };
        std::vector<Node> nodes;
        SmallVector<unsigned> accessNodes;
        llvm::DenseMap<const BaseMemInfo*, unsigned> ids;
        for (const auto& access : accesses) {
            auto [it, inserted] = ids.try_emplace(access.memory, nodes.size());
            if (inserted) nodes.push_back({it->second, 0, access.memory, bool(slotMappings.at(access.memory))});
            accessNodes.push_back(it->second);
        }
        slotComponentNodes = nodes.size();
        auto find = [&](unsigned id) -> std::optional<unsigned> {
            for (;;) {
                if (!queries.spend(1)) { fail(ConstructionResult::AnalysisLimit, "slot component find budget"); return {}; }
                ++slotComponentFinds;
                if (nodes[id].parent == id) return id;
                nodes[id].parent = nodes[nodes[id].parent].parent;
                id = nodes[id].parent;
            }
        };
        for (auto [x, y] : candidates) {
            if (!queries.spend(1)) return fail(ConstructionResult::AnalysisLimit, "slot component edge budget");
            ++slotComponentEdges;
            auto a = find(accessNodes[x]), b = find(accessNodes[y]);
            if (!a || !b) return false;
            if (*a == *b) continue;
            bool homogeneous = nodes[*a].homogeneous && nodes[*b].homogeneous &&
                               sameSlotGeometry(nodes[*a].memory, nodes[*b].memory);
            if (result.status == ConstructionResult::AnalysisLimit) return false;
            if (nodes[*a].rank < nodes[*b].rank) std::swap(a, b);
            nodes[*b].parent = *a;
            nodes[*a].rank += nodes[*a].rank == nodes[*b].rank;
            nodes[*a].homogeneous = homogeneous;
            ++slotComponentJoins;
        }
        for (unsigned id = 0; id < nodes.size(); ++id) {
            auto root = find(id);
            if (!root) return false;
            if (nodes[*root].homogeneous) compactSlotMemories.insert(nodes[id].memory);
            const auto& mapping = slotMappings.at(nodes[id].memory);
            if (std::getenv("PTOAS_LOGICAL_TRACE") && mapping && mapping->selector)
                llvm::errs() << "logical slot_component_member index " << id << " scope " << unsigned(mapping->scope)
                             << " base " << mapping->bases.front() << " slots " << mapping->bases.size()
                             << " compact " << nodes[*root].homogeneous << "\n";
        }
        // A heterogeneous/unknown component retains the SAME precise general
        // slot calculation. Representation selection never removes a conflict.
        return true;
    }
    std::optional<Relation> slotConflicts(unsigned p, unsigned q, const BaseMemInfo* a,
                                          const BaseMemInfo* b, const Relation& original)
    {
        const auto& ma = slotMappings.at(a); const auto& mb = slotMappings.at(b);
        if (!ma || !mb || (!ma->selector && !mb->selector)) return original;
        // Discovery only calls this for an edge of the frozen candidate graph.
        // Thus eligible endpoints belong to one qualified homogeneous component.
        bool sameGeometry = ma->selector && mb->selector && compactSlotMemories.contains(a) && compactSlotMemories.contains(b);
        if (sameGeometry) {
            bool leftRange = slotInRange(p, *ma);
            if (result.status == ConstructionResult::AnalysisLimit) return {};
            bool rightRange = slotInRange(q, *mb);
            if (result.status == ConstructionResult::AnalysisLimit) return {};
            if (!leftRange || !rightRange) return original;
            // The only caller (discovery) supplies FULL immutable NativeOrder
            // p->q here, never a previously filtered relation. The cache key
            // relies on that invariant: it cannot identify arbitrary subsets.
            // This order's ambient-domain qualification
            // is the filter API's precondition; do not multiply those domains
            // into the equality again. Cache across RAW/WAR/WAW consumers.
            if (!queries.spend(1)) {
                fail(ConstructionResult::AnalysisLimit, "slot equality lookup budget"); return {};
            }
            auto key = std::make_pair(SelectorPoint{ma->selector, {p, unsigned(ma->bases.size())}},
                                      SelectorPoint{mb->selector, {q, unsigned(mb->bases.size())}});
            auto found = compactSlotRelations.find(key);
            if (found == compactSlotRelations.end()) {
                ++slotEqualityBuilds;
                auto equal = facts.filterEqualScalars(original, ma->selector, p, mb->selector, q, queries);
                if (equal.status == QueryStatus::BudgetExhausted) {
                    fail(ConstructionResult::AnalysisLimit, "slot equality filter budget"); return {};
                }
                found = compactSlotRelations.try_emplace(key, equal ? std::move(equal.relation) : std::nullopt).first;
            }
            if (found->second) {
                usedCompactSlots.insert({p, a});
                usedCompactSlots.insert({q, b});
                return *found->second;
            }
        }
        const auto* left = slots(p, a); const auto* right = slots(q, b);
        if (result.status == ConstructionResult::AnalysisLimit) return {};
        if (!left || !right) return original;
        auto overlaps = overlappingSyncPhysicalSlots(*ma, *mb, [&](uint64_t work) { return queries.spend(work); });
        if (!overlaps) {
            fail(ConstructionResult::AnalysisLimit, "physical slot overlap budget"); return {};
        }
        auto result = empty();
        for (auto [x, y] : *overlaps) {
            const auto& source = (*left)[ma->selector ? x : 0];
            const auto& target = (*right)[mb->selector ? y : 0];
            // restrictEndpoints() only prunes incompatible fixed coordinates;
            // physical slot equality needs the EXACT guarded intersection.
            uint64_t estimate = 1;
            for (uint64_t factor : {uint64_t(original.getNumDisjuncts()) + 1,
                                   uint64_t(source.getNumDisjuncts()) + 1,
                                   uint64_t(target.getNumDisjuncts()) + 1,
                                   uint64_t(facts.dimensions() + facts.parameters.size() + 1) * 4}) {
                if (factor > queries.remainingWork() / estimate) {
                    fail(ConstructionResult::AnalysisLimit, "slot intersection expansion budget"); return {};
                }
                estimate *= factor;
            }
            if (!queries.spend(estimate)) {
                fail(ConstructionResult::AnalysisLimit, "slot intersection budget"); return {};
            }
            auto restricted = take(queries.normalize(
                original.intersectDomain(source.getRangeSet()).intersectRange(target.getRangeSet())));
            if (!restricted) return {};
            result.unionInPlace(*restricted);
        }
        return take(queries.normalize(result));
    }
    QueryStatus functional(const Relation& handoff)
    {
        auto inverse = handoff;
        inverse.inverse();
        auto consumers = queries.compose(inverse, handoff);
        if (!consumers)
            return consumers.status;
        auto producers = queries.compose(handoff, inverse);
        if (!producers)
            return producers.status;
        auto diagonal = identity();
        auto status = queries.contains(diagonal, *consumers.relation);
        return status == QueryStatus::Proved ? queries.contains(diagonal, *producers.relation) : status;
    }
    // Exact common case of the same prefix-staircase rule: all payload
    // occurrences execute once in the original function block. Scanning its
    // demands needs no Cartesian target-order relation. Guarded, recurring or
    // access-qualified alternatives remain with the general exact query path.
    std::optional<bool> constructLinearHandoffs() {
        auto* block = &function.getBody().front();
        if (!facts.loops.empty() || llvm::any_of(phases, [&](const auto* p) {
                return p->elementOp->getBlock() != block; })) return {};
        std::map<unsigned, unsigned> rank;
        unsigned next = 0;
        for (auto& op : *block) {
            auto found = facts.ids.find(&op);
            if (found != facts.ids.end()) rank.emplace(found->second, next++);
        }
        // Check the precise domain, not merely a convenient static phase ID.
        for (const auto& [pair, relation] : pairRequirements) {
            auto full = order(pair.first, pair.second);
            if (!full) return false;
            auto covered = queries.contains(relation, *full);
            if (queryFailed(covered, "linear handoff domain qualification")) return false;
            if (covered != QueryStatus::Proved) return {};
        }
        std::map<Domain, std::map<unsigned, std::pair<unsigned, unsigned>>> deadlines;
        for (const auto& [pair, relation] : pairRequirements) {
            auto [source, target] = pair;
            if (lanes[source] == lanes[target]) continue;
            auto& demands = deadlines[{lanes[source], lanes[target]}];
            auto [it, inserted] = demands.try_emplace(rank.at(target), source, target);
            if (!inserted && rank.at(source) > rank.at(it->second.first)) it->second.first = source;
        }
        for (const auto& [pipes, demands] : deadlines) {
            std::optional<unsigned> acquired;
            for (const auto& [position, demand] : demands) {
                auto [source, target] = demand;
                if (acquired && rank.at(source) <= *acquired) continue;
                auto matching = order(source, target);
                if (!matching) return false;
                streams.push_back({pipes, std::move(*matching), source, -1});
                acquired = rank.at(source);
            }
        }
        return true;
    }
    bool constructHandoffs()
    {
        if (auto linear = constructLinearHandoffs()) return *linear;
        std::map<Domain, Relation> missing;
        for (const auto& [pair, relation] : pairRequirements)
            if (lanes[pair.first] != lanes[pair.second]) {
                auto key = Domain{lanes[pair.first], lanes[pair.second]};
                auto [it, inserted] = missing.try_emplace(key, empty());
                it->second.unionInPlace(relation);
            }
        for (const auto& [pipes, needed] : missing) {
            auto sources = needed.getDomainSet(), targets = needed.getRangeSet();
            auto before = selectedOrder(sources, sources), through = selectedOrder(sources, sources, true);
            auto destinations = selectedOrder(targets, targets);
            if (!before || !through || !destinations) return false;
            auto latest = take(queries.latestSources(needed, *before));
            if (!latest)
                return false;
            auto staircase = take(queries.staircase(*latest, *through, *destinations));
            if (!staircase)
                return false;
            if (!expect(functional(*staircase), "handoff participation is not one-to-one"))
                return false;
            // Static publication families are logical streams, not numeric keys.
            // Alternatives for first/empty/later use remain in the same family.
            for (unsigned p = 0; p < phases.size(); ++p)
                if (lanes[p] == pipes.first) {
                    auto family = take(queries.normalize(staircase->intersectDomain(facts.domain(p).getRangeSet())));
                    if (!family)
                        return false;
                    if (!family->isIntegerEmpty())
                        streams.push_back({pipes, std::move(*family), p, -1});
                }
        }
        return true;
    }
    std::optional<Relation> primitive(
        std::optional<unsigned> omittedBarrier = {}, std::optional<unsigned> omittedStream = {})
    {
        auto supply = empty();
        for (unsigned i = 0; i < streams.size(); ++i)
            if (omittedStream != i) {
                auto& stream = streams[i];
                supply.unionInPlace(stream.matching);
            }
        for (unsigned i = 0; i < barriers.size(); ++i)
            if (omittedBarrier != i) {
                auto& barrier = barriers[i];
                if (barrier.logical) {
                    supply.unionInPlace(*barrier.logical);
                    continue;
                }
                // In a clear original block interval, the previous lane operation
                // is the exact predecessor in this invocation. Reuse the imported
                // IV identities rather than rediscovering it through every pair.
                if (auto previous = orders->clearPredecessor.find(barrier.point);
                    previous != orders->clearPredecessor.end()) {
                    auto edge = order(previous->second, barrier.point);
                    if (!edge) return {};
                    IntegerRelation invocation(empty().getSpace());
                    for (unsigned iv = 1; iv < facts.dimensions(); ++iv) {
                        SmallVector<int64_t> row(invocation.getNumCols());
                        row[iv] = 1;
                        row[facts.dimensions() + iv] = -1;
                        invocation.addEquality(row);
                    }
                    barrier.logical =
                        take(queries.normalize(edge->intersect(Relation(invocation)).intersectRange(barrier.domain)));
                    if (!barrier.logical) return {};
                }
                if (barrier.logical) {
                    supply.unionInPlace(*barrier.logical);
                    continue;
                }
                auto candidates = empty();
                for (unsigned source = 0; source < lanes.size(); ++source)
                    if (lanes[source] == barrier.pipe) {
                        auto edge = order(source, barrier.point);
                        if (!edge)
                            return {};
                        candidates.unionInPlace(*edge);
                    }
                auto cut = take(queries.normalize(candidates.intersectRange(barrier.domain)));
                if (!cut)
                    return {};
                auto before = selectedOrder(cut->getDomainSet(), cut->getDomainSet());
                if (!before) return {};
                auto edge = take(queries.latestSources(*cut, *before));
                if (!edge)
                    return {};
                barrier.logical = *edge;
                supply.unionInPlace(*edge);
            }
        return supply;
    }
    const Relation& requirementsAt(unsigned point) const { return targetRequirements[point]; }
    QueryStatus proveRequirements(CompletionQueries& completed)
    {
        for (unsigned p = 0; p < lanes.size(); ++p) {
            auto status = completed.prove(requirementsAt(p), queries, 2);
            if (status != QueryStatus::Proved)
                return status;
        }
        return QueryStatus::Proved;
    }
    bool repairBarriers()
    {
        auto supply = primitive();
        if (!supply)
            return false;
        auto completed = completion(*supply);
        if (!completed)
            return false;
        for (unsigned p = 0; p < lanes.size(); ++p) {
            auto required = requirementsAt(p);
            auto status = completed->prove(required, queries, 2);
            if (std::getenv("PTOAS_LOGICAL_TRACE"))
                llvm::errs() << "logical phase " << p << " work " << queries.work() << " status " << unsigned(status)
                             << " required pieces " << required.getNumDisjuncts() << " logical edges "
                             << supply->getNumDisjuncts() << " known pieces " << completed->supply().getNumDisjuncts()
                             << "\n";
            if (status == QueryStatus::Proved) {
                provedTargets.insert(p);
                continue;
            }
            if (queryFailed(
                    status, "combined completion query unavailable at phase " + std::to_string(p) + "; streams=" +
                                std::to_string(streams.size()) + "; barriers=" + std::to_string(barriers.size())))
                return false;
            auto missing = take(queries.subtract(required, completed->supply()));
            if (!missing)
                return false;
            // Cross-lane construction must already supply its own guarantees.
            for (const auto& r : requirements)
                if (r.kind != Requirement::Retirement && r.target == p && lanes[r.source] != lanes[p])
                    if (!expect(
                            queries.contains(completed->supply(), r.occurrences),
                            "cross-lane occurrence requirement remains unordered"))
                        return false;
            auto domain = take(queries.normalize(missing->getRangeSet()));
            if (!domain)
                return false;
            auto ambient = facts.domain(p).getRangeSet();
            auto covers = queries.contains(*domain, ambient);
            if (queryFailed(covers, "barrier execution domain query"))
                return false;
            barriers.push_back({p, lanes[p], covers == QueryStatus::Proved ? ambient : PresburgerSet(*domain), {}});
            auto laneBefore = selectedOrder(missing->getDomainSet(), barriers.back().domain);
            if (!laneBefore)
                return false;
            auto barrierSupply = take(queries.normalize(laneBefore->intersectRange(barriers.back().domain)));
            if (!barrierSupply)
                return false;
            // The new barrier supplies the exact missing subset; the prior
            // proved supply covers its complement. This is a constructive
            // proof, not a receipt inferred from merely selecting a barrier.
            if (!expect(queries.contains(*barrierSupply, *missing), "new barrier does not supply missing requirements"))
                return false;
            ++planVersion;
            receiptVersion = planVersion;
            provedTargets.insert(p);
            supply = primitive();
            if (!supply)
                return false;
            if (!barriers.back().logical)
                return fail(ConstructionResult::InternalError, "new barrier has no logical completion cut");
            if (!expect(completed->addHandoffs(*barriers.back().logical, queries), "new barrier completion update"))
                return false;
        }
        // Feedback is part of initial construction. Candidate barriers never
        // supply the proof of their own deletion.
        for (unsigned i = 0; i < barriers.size();) {
            auto remaining = primitive(i);
            if (!remaining)
                return false;
            auto trial = completion(*remaining);
            if (!trial)
                return false;
            // First challenge the demand at this cut. A failed local query
            // keeps the barrier; a successful one is followed by every other
            // requirement before deletion can be accepted.
            auto status = trial->prove(requirementsAt(barriers[i].point), queries, 2);
            if (status == QueryStatus::Proved)
                status = proveRequirements(*trial);
            if (status == QueryStatus::Proved) {
                barriers.erase(barriers.begin() + i);
                ++planVersion;
                receiptVersion = planVersion;
                provedTargets.clear();
                for (unsigned p = 0; p < lanes.size(); ++p)
                    provedTargets.insert(p);
            } else if (queryFailed(status, "barrier deletion completion query"))
                return false;
            else
                ++i;
        }
        return true;
    }
    // Lowering and independent reconstruction are provided below. They consume
    // this plan and never ask ordinary InsertSync to manufacture a seed.
    bool realize();
    bool reconstruct();
    QueryStatus reuseSafe(const Relation& matching, Lane pipe, CompletionQueries& completed)
    {
        auto domain = matching.getDomainSet();
        auto periodicStart = std::chrono::steady_clock::now();
        auto periodicWork = queries.work();
        auto next = facts.periodicSuccessors(domain, queries);
        if (std::getenv("PTOAS_LOGICAL_TRACE"))
            llvm::errs() << "logical periodic_successor applied " << bool(next)
                         << " work " << queries.work() - periodicWork << " seconds "
                         << std::chrono::duration<double>(std::chrono::steady_clock::now() - periodicStart).count()
                         << " reason " << next.reason << "\n";
        if (!next && next.status == QueryStatus::Unsupported) {
            auto before = selectedOrder(domain, domain);
            if (!before)
                return result.status == ConstructionResult::AnalysisLimit ? QueryStatus::BudgetExhausted :
                                                                            QueryStatus::Unsupported;
            auto following = before->intersectDomain(domain).intersectRange(domain);
            next = queries.firstTargets(following, *before);
        }
        if (!next)
            return next.status;
        auto inverse = matching;
        inverse.inverse();
        auto required = queries.compose(inverse, *next.relation);
        if (!required)
            return required.status;
        return completed.prove(*required.relation, queries, 2);
    }

public:
    Constructor(
        func::FuncOp f, InsertSyncGMAliasMode gm, uint64_t budget, llvm::function_ref<void(func::FuncOp)> mutate = {},
        testing::RequirementObserver observe = {})
        : function(f), payload(f), emissionMutation(mutate), observeRequirements(observe), gm(gm), queries(budget)
    {}
    ConstructionResult run()
    {
        StringRef stage = "discovery";
        auto run = [&](StringRef name, auto method) {
            stage = name;
            auto start = queries.work();
            auto time = std::chrono::steady_clock::now();
            bool ok = (this->*method)();
            if (std::getenv("PTOAS_LOGICAL_TRACE"))
                llvm::errs() << "logical stage " << name << " work " << queries.work() - start << " complete " << ok
                             << " seconds " << std::chrono::duration<double>(std::chrono::steady_clock::now() - time).count()
                             << " order_requests " << (orders ? orders->requests : 0)
                             << " order_built " << (orders ? orders->built : 0)
                             << " acquisition_projections " << (orders ? orders->acquisitionProjections : 0)
                             << " acquisition_candidates " << (orders ? orders->acquisitionCandidates : 0)
                             << " endpoint_comparisons " << queries.endpointComparisonCount() << "\n";
            if (!ok)
                result.reason += " (stage work " + std::to_string(queries.work() - start) + ")";
            return ok;
        };
        if (run("discovery", &Constructor::discover) && run("handoffs", &Constructor::constructHandoffs) &&
            run("barriers", &Constructor::repairBarriers) && run("realization", &Constructor::realize)) {
            result.status = ConstructionResult::Applied;
            result.reason = "constructed and reconstructed occurrence handoffs";
            result.handoffs = streams.size();
            result.barriers = barriers.size() + 1; // Includes the explicit retirement drain.
        } else
            result.reason = stage.str() + ": " + result.reason;
        result.work = queries.work();
        if (std::getenv("PTOAS_LOGICAL_TRACE"))
            llvm::errs() << "logical slot_queries range_builds " << slotRangeBuilds
                         << " equality_builds " << slotEqualityBuilds << " domain_builds " << slotDomainsBuilt
                         << " geometry_builds " << slotGeometryBuilds << "\n";
        if (std::getenv("PTOAS_LOGICAL_TRACE"))
            llvm::errs() << "logical slot_components nodes " << slotComponentNodes << " edges " << slotComponentEdges
                         << " find_steps " << slotComponentFinds << " joins " << slotComponentJoins
                         << " compact_members " << compactSlotMemories.size() << "\n";
        if (queries.profilingEnabled()) {
            auto report = [](StringRef name, const RelationQueries::PrimitiveStats& stats) {
                llvm::errs() << "logical primitive " << name << " calls " << stats.calls
                    << " nanoseconds " << stats.wallNanoseconds << " max_input_pieces " << stats.maxInputPieces << "\n";
            };
            const auto& profile = queries.profile();
            report("normalize", profile.normalize); report("compose", profile.compose);
            report("subtract", profile.subtract); report("contains", profile.contains);
            report("subtractQualification", profile.subtractQualification);
            report("subtractPartition", profile.subtractPartition);
            report("subtractImplication", profile.subtractImplication);
            report("restrictEndpoints", profile.restrictEndpoints);
            llvm::errs() << "logical cache composition_builds " << queries.compositionIndexBuildCount()
                << " composition_pieces " << queries.compositionIndexPieceCount()
                << " endpoint_projections " << queries.endpointProjectionCount()
                << " endpoint_pieces " << queries.endpointProjectionPieceCount() << "\n";
            llvm::errs() << "logical difference common_rows " << queries.differenceCommonRowCount() << "\n";
        }
        return result;
    }
};
} // namespace

namespace {
// These are lowering conditions, not program patterns. A proposed mathematical
// domain must equal their interpretation at the actual insertion point. No
// guessed trip distance or unavailable future condition is emitted.
struct BoundaryTest {
    enum Kind { First, Last, Nonempty, Predicate, BooleanParameter, ConstantBound, DifferenceBound,
                ParameterBound, ParameterResidue } kind;
    unsigned loop;
    bool truth;
    int64_t bound = 0;
    arith::CmpIPredicate comparison = arith::CmpIPredicate::eq;
    bool fromUpper = false;
    int64_t modulus = 1;
};
using Clause = SmallVector<BoundaryTest, 2>;
using Guard = SmallVector<Clause, 2>;
// Short-circuit DNF can duplicate a suffix at each failed literal. Account for
// that actual tree before allocation/emission, rather than depending on the
// later occurrence importer to discover an already expanded function.
bool chargeGuardEmission(const Guard& guard, uint64_t& remaining)
{
    constexpr uint64_t operationsPerLiteral = 12; // includes scalar ops and both region yields
    constexpr unsigned maxDepth = 64;
    uint64_t nodes = 0;
    unsigned depth = 0;
    for (const auto& clause : guard) {
        if (clause.size() > maxDepth - depth) return false;
        depth += clause.size();
    }
    const bool shortCircuit = llvm::any_of(guard, [](const Clause& clause) {
        return llvm::any_of(clause, [](const BoundaryTest& test) {
            return test.kind == BoundaryTest::ParameterResidue;
        });
    });
    if (shortCircuit) {
        for (const auto& clause : llvm::reverse(guard)) {
            uint64_t suffix = nodes;
            nodes = 1; // the action after a successful clause
            for (unsigned i = 0; i < clause.size(); ++i) {
                if (suffix > remaining || nodes > remaining - suffix ||
                    operationsPerLiteral > remaining - suffix - nodes) return false;
                nodes += suffix + operationsPerLiteral;
            }
        }
    } else {
        nodes = 4; // the action and optional guarding if/yield
        for (const auto& clause : guard)
            nodes += operationsPerLiteral * (clause.size() + 1);
    }
    if (nodes > remaining) return false;
    remaining -= nodes;
    return true;
}
struct Endpoint {
    unsigned point;
    bool publication;
    unsigned stream;
    Guard guard;
    PresburgerSet domain;
};
class BoundaryLowering {
    const SyncOccurrences& facts;
    RelationQueries& queries;
    DominanceInfo dominance;
    QueryStatus outcome = QueryStatus::Unsupported;
    using TestKey = std::tuple<unsigned, unsigned, bool, int64_t, unsigned, bool, int64_t>;
    static TestKey keyFor(BoundaryTest test)
    {
        return {test.kind, test.loop, test.truth, test.bound, unsigned(test.comparison),
                test.fromUpper, test.modulus};
    }
    // One immutable occurrence universe, before emission. A point is part of
    // the key: identical scalar tests can have different ambient domains and
    // operand bindings at different cuts. Map nodes own stable, borrowed
    // candidate domains, so repeated endpoints do not copy large relations.
    std::map<std::pair<unsigned, TestKey>, Relation> conditionDomains;
    struct DifferenceRange {
        Relation definition;
        std::optional<std::pair<int64_t, int64_t>> full;
    };
    std::map<std::tuple<unsigned, unsigned, bool>, DifferenceRange> differenceRanges;
    uint64_t conditionRequests = 0, conditionBuilds = 0, rangeRequests = 0, rangeBuilds = 0;
    // Cache only within a block and check the actual insertion cursor. Endpoint
    // traversal is not necessarily lexical; a later definition cannot be reused
    // at an earlier cut. In particular, no remainder escapes its guarding block.
    mutable std::map<Block*, std::map<TestKey, Value>> emittedTests;
    bool queryFailed(QueryStatus status)
    {
        if (status == QueryStatus::Proved || status == QueryStatus::NotEstablished)
            return false;
        outcome = status;
        return true;
    }
    const Relation* condition(unsigned point, BoundaryTest test)
    {
        ++conditionRequests;
        if (!queries.spend(1)) {
            outcome = QueryStatus::BudgetExhausted;
            return nullptr;
        }
        auto key = std::make_pair(point, keyFor(test));
        if (auto found = conditionDomains.find(key); found != conditionDomains.end())
            return &found->second;
        ++conditionBuilds;
        auto domain = buildCondition(point, test);
        if (!domain) return nullptr;
        // Charge retained storage once; a failed/exhausted query never enters
        // the cache as an empty domain. Keep the existing normalization and
        // proof queries at the consumers, rather than doing extra solver work
        // merely to intern an immutable domain.
        for (const auto& piece : domain->getAllDisjuncts()) {
            if (!queries.spend(uint64_t(piece.getNumConstraints() + 1) * piece.getNumCols())) {
                outcome = QueryStatus::BudgetExhausted;
                return nullptr;
            }
        }
        return &conditionDomains.emplace(std::move(key), std::move(*domain)).first->second;
    }
    std::optional<Relation> buildCondition(unsigned point, BoundaryTest test)
    {
        if (test.kind == BoundaryTest::ConstantBound || test.kind == BoundaryTest::ParameterBound) {
            auto domain = facts.domain(point);
            auto restricted = Relation::getEmpty(domain.getSpace());
            for (auto piece : domain.getAllDisjuncts()) {
                auto type = test.comparison == arith::CmpIPredicate::eq  ? BoundType::EQ :
                            test.comparison == arith::CmpIPredicate::sle ? BoundType::UB :
                                                                           BoundType::LB;
                unsigned coordinate = test.kind == BoundaryTest::ConstantBound ? test.loop + 1 :
                                          facts.dimensions() + test.loop;
                SmallVector<llvm::DynamicAPInt> row(piece.getNumCols());
                row[coordinate] = type == BoundType::UB ? -1 : 1;
                row.back() = type == BoundType::UB ? llvm::DynamicAPInt(test.bound) :
                                                     -llvm::DynamicAPInt(test.bound);
                restricted.unionInPlace(membership(piece, row, type == BoundType::EQ, test.truth));
            }
            return restricted;
        }
        if (test.kind == BoundaryTest::Predicate) {
            auto positive = facts.predicateDomain(test.loop, point);
            if (test.truth)
                return positive;
            auto negative = queries.subtract(facts.domain(point), positive);
            if (!negative)
                outcome = negative.status;
            return negative ? std::move(negative.relation) : std::nullopt;
        }
        auto* context = facts.points[point].operation->getContext();
        AffineExpr expression;
        if (test.kind == BoundaryTest::ParameterResidue)
            expression = getAffineSymbolExpr(test.loop, context) % test.modulus - test.bound;
        else if (test.kind == BoundaryTest::BooleanParameter)
            expression = getAffineSymbolExpr(test.loop, context) - 1;
        else {
            auto loop = facts.loopDomains[test.loop];
            auto iv = getAffineDimExpr(test.loop, context);
            expression = test.kind == BoundaryTest::DifferenceBound ?
                             (test.fromUpper ? loop.upper - iv : iv - loop.lower) :
                         test.kind == BoundaryTest::First ? iv - loop.lower :
                         test.kind == BoundaryTest::Last  ? iv + loop.step - loop.upper :
                                                            loop.upper - loop.lower - 1;
        }
        SmallVector<AffineExpr> dims, symbols;
        for (unsigned i = 0; i < facts.loops.size(); ++i)
            dims.push_back(getAffineDimExpr(i + 1, context));
        for (unsigned i = 0; i < facts.parameters.size(); ++i)
            symbols.push_back(getAffineSymbolExpr(i, context));
        expression = expression.replaceDimsAndSymbols(dims, symbols);
        auto build = [&](AffineExpr expr, bool equality) -> std::optional<Relation> {
            auto set = IntegerSet::get(facts.dimensions(), symbols.size(), {expr}, {equality});
            FlatLinearConstraints flat(facts.dimensions(), symbols.size());
            std::vector<SmallVector<int64_t, 8>> rows;
            if (failed(getFlattenedAffineExprs(set, &rows, &flat)))
                return {};
            SmallVector<llvm::DynamicAPInt> row;
            for (int64_t coefficient : rows.front()) row.emplace_back(coefficient);
            if (test.kind == BoundaryTest::DifferenceBound)
                row.back() += test.comparison == arith::CmpIPredicate::sle ?
                                  llvm::DynamicAPInt(test.bound) : -llvm::DynamicAPInt(test.bound);
            // Flattening contributes total floor/mod definitions. Complement
            // only the final membership atom, never those existential witness
            // definitions. APInt arithmetic also handles INT64_MIN/MAX bounds.
            return facts.domain(point).intersect(membership(flat, row, equality, test.truth));
        };
        if (test.kind == BoundaryTest::DifferenceBound && test.comparison == arith::CmpIPredicate::sle)
            expression = -expression;
        return build(expression,
            test.kind == BoundaryTest::First || test.kind == BoundaryTest::BooleanParameter ||
            test.kind == BoundaryTest::ParameterResidue ||
            (test.kind == BoundaryTest::DifferenceBound && test.comparison == arith::CmpIPredicate::eq));
    }

    static Relation membership(const IntegerRelation& definitions, ArrayRef<llvm::DynamicAPInt> row,
                               bool equality, bool truth)
    {
        auto result = Relation::getEmpty(definitions.getSpaceWithoutLocals());
        auto append = [&](SmallVector<llvm::DynamicAPInt> atom, bool eq) {
            auto piece = definitions;
            if (eq) piece.addEquality(atom);
            else piece.addInequality(atom);
            result.unionInPlace(Relation(piece));
        };
        if (truth) {
            append(SmallVector<llvm::DynamicAPInt>(row), equality);
        } else {
            SmallVector<llvm::DynamicAPInt> negative;
            for (const auto& value : row) negative.push_back(-value);
            --negative.back();
            append(std::move(negative), false);
            if (equality) {
                SmallVector<llvm::DynamicAPInt> positive(row);
                --positive.back();
                append(std::move(positive), false);
            }
        }
        return result;
    }

    const DifferenceRange* differenceRange(unsigned point, unsigned loopIndex, bool fromUpper)
    {
        ++rangeRequests;
        if (!queries.spend(1)) { outcome = QueryStatus::BudgetExhausted; return nullptr; }
        auto key = std::make_tuple(point, loopIndex, fromUpper);
        if (auto found = differenceRanges.find(key); found != differenceRanges.end()) return &found->second;
        ++rangeBuilds;
        auto* context = facts.points[point].operation->getContext();
        const unsigned n = facts.dimensions(), ns = facts.parameters.size();
        SmallVector<AffineExpr> dims, syms;
        for (unsigned d = 0; d < facts.loops.size(); ++d) dims.push_back(getAffineDimExpr(d + 1, context));
        for (unsigned s = 0; s < ns; ++s) syms.push_back(getAffineSymbolExpr(s, context));
        auto ld = facts.loopDomains[loopIndex];
        auto iv = getAffineDimExpr(loopIndex, context);
        auto expr = (fromUpper ? ld.upper - iv : iv - ld.lower).replaceDimsAndSymbols(dims, syms);
        auto set = IntegerSet::get(n + 1, ns, {expr - getAffineDimExpr(n, context)}, {true});
        FlatLinearConstraints flat(n + 1, ns);
        std::vector<SmallVector<int64_t, 8>> rows;
        if (failed(getFlattenedAffineExprs(set, &rows, &flat))) return nullptr;
        flat.addEquality(rows.front());
        auto ambient = facts.domain(point);
        ambient.insertVarInPlace(VarKind::Range, n);
        auto full = takeRange(ambient.intersect(Relation(flat)), n);
        if (outcome == QueryStatus::BudgetExhausted) return nullptr;
        if (!queries.spend(uint64_t(flat.getNumConstraints() + 1) * flat.getNumCols())) {
            outcome = QueryStatus::BudgetExhausted;
            return nullptr;
        }
        return &differenceRanges.emplace(key, DifferenceRange{Relation(flat), full}).first->second;
    }

public:
    BoundaryLowering(const SyncOccurrences& f, RelationQueries& q) : facts(f), queries(q) {}
    ~BoundaryLowering()
    {
        if (std::getenv("PTOAS_LOGICAL_TRACE"))
            llvm::errs() << "logical guard domains " << conditionBuilds << "/" << conditionRequests
                         << " ranges " << rangeBuilds << "/" << rangeRequests << "\n";
    }
    QueryStatus status() const { return outcome; }
    bool checkConditions()
    {
        // Test-only differential challenge: compare the direct atom complement
        // with the existing general integer subtraction, including unbounded
        // parameters and extrema that cannot be negated in int64 arithmetic.
        auto check = [&](unsigned point, BoundaryTest test) {
            test.truth = true;
            const auto* positive = condition(point, test);
            test.truth = false;
            const auto* negative = condition(point, test);
            if (!positive || !negative) return false;
            auto builds = conditionBuilds;
            auto stored = conditionDomains.size();
            for (unsigned repetitions : {8, 32, 128}) {
                auto start = queries.work();
                for (unsigned i = 0; i < repetitions; ++i)
                    if (condition(point, test) != negative) return false;
                if (conditionBuilds != builds || conditionDomains.size() != stored ||
                    queries.work() - start != repetitions) return false;
            }
            auto expected = queries.subtract(facts.domain(point), *positive);
            return expected && queries.contains(*expected.relation, *negative) == QueryStatus::Proved &&
                   queries.contains(*negative, *expected.relation) == QueryStatus::Proved;
        };
        std::set<unsigned> checkedLoops;
        bool checkedParameters = false;
        for (unsigned point = 0; point < facts.points.size(); ++point) {
            auto* anchor = facts.points[point].operation;
            if (!anchor) continue;
            for (unsigned i = 0; i < facts.loops.size(); ++i) {
                auto loop = facts.loops[i];
                if (!loop->isProperAncestor(anchor)) continue;
                if (!checkedLoops.insert(i).second) {
                    // Same test, different phase/guard domain in ONE cache.
                    // A cache accidentally keyed by the atom alone must fail.
                    if (!check(point, BoundaryTest{BoundaryTest::First, i, true})) return false;
                    continue;
                }
                for (auto kind : {BoundaryTest::First, BoundaryTest::Last, BoundaryTest::Nonempty})
                    if (!check(point, BoundaryTest{kind, i, true})) return false;
                for (int64_t bound : {INT64_MIN, int64_t(-2), int64_t(0), int64_t(3), INT64_MAX})
                    for (auto cmp : {arith::CmpIPredicate::eq, arith::CmpIPredicate::sle,
                                     arith::CmpIPredicate::sge}) {
                        if (!check(point, BoundaryTest{BoundaryTest::ConstantBound, i, true, bound, cmp}))
                            return false;
                        for (bool upper : {false, true})
                            if (!check(point, BoundaryTest{BoundaryTest::DifferenceBound, i, true, bound,
                                                           cmp, upper})) return false;
                    }
                for (bool upper : {false, true}) {
                    const auto* cached = differenceRange(point, i, upper);
                    if (!cached) return false;
                    auto builds = rangeBuilds, stored = differenceRanges.size();
                    for (unsigned repetitions : {8, 32, 128}) {
                        auto start = queries.work();
                        for (unsigned r = 0; r < repetitions; ++r)
                            if (differenceRange(point, i, upper) != cached) return false;
                        if (rangeBuilds != builds || differenceRanges.size() != stored ||
                            queries.work() - start != repetitions) return false;
                    }
                    auto ambient = facts.domain(point);
                    ambient.insertVarInPlace(VarKind::Range, facts.dimensions());
                    if (takeRange(ambient.intersect(cached->definition), facts.dimensions()) != cached->full)
                        return false;
                }
            }
            if (checkedParameters) continue;
            checkedParameters = true;
            for (unsigned s = 0; s < facts.parameters.size(); ++s) {
                if (facts.parameters[s].getType().isInteger(1)) {
                    if (!check(point, BoundaryTest{BoundaryTest::BooleanParameter, s, true})) return false;
                    continue;
                }
                for (int64_t bound : {INT64_MIN, int64_t(-1), int64_t(0), INT64_MAX})
                    if (!check(point, BoundaryTest{BoundaryTest::ParameterBound, s, true, bound,
                                                  arith::CmpIPredicate::sge})) return false;
                for (int64_t modulus : {2, 3})
                    for (int64_t residue = 0; residue < modulus; ++residue)
                        if (!check(point, BoundaryTest{BoundaryTest::ParameterResidue, s, true, residue,
                                                       arith::CmpIPredicate::eq, false, modulus})) return false;
            }
        }
        return conditionBuilds != 0 && queries.remainingWork() != 0;
    }
    std::optional<Guard> prepare(unsigned point, const PresburgerSet& wanted)
    {
        outcome = QueryStatus::Unsupported;
        Relation target = wanted;
        auto ambient = facts.domain(point);
        auto status = queries.contains(target, ambient);
        if (status == QueryStatus::Proved)
            return Guard{Clause{}};
        if (queryFailed(status))
            return {};
        struct Candidate {
            BoundaryTest test;
            const Relation* domain;
        };
        std::vector<Candidate> candidates;
        Operation* anchor = facts.points[point].operation;
        for (unsigned i = 0; i < facts.loops.size(); ++i) {
            auto loop = facts.loops[i];
            if (!dominance.dominates(loop.getLowerBound(), anchor) ||
                !dominance.dominates(loop.getUpperBound(), anchor))
                continue;
            for (auto kind : {BoundaryTest::First, BoundaryTest::Last, BoundaryTest::Nonempty}) {
                if (kind != BoundaryTest::Nonempty && !loop->isProperAncestor(anchor))
                    continue;
                for (bool truth : {true, false}) {
                    BoundaryTest test{kind, i, truth};
                    auto domain = condition(point, test);
                    if (!domain)
                        return {};
                    candidates.push_back({test, domain});
                }
            }
        }
        for (unsigned i = 0; i < facts.predicates.size(); ++i) {
            if (!dominance.dominates(facts.predicates[i].value, anchor))
                continue;
            for (bool truth : {true, false}) {
                BoundaryTest test{BoundaryTest::Predicate, i, truth};
                auto domain = condition(point, test);
                if (!domain)
                    return {};
                candidates.push_back({test, domain});
            }
        }
        // A normalized branch can be an equivalent spelling of an already
        // available Boolean parameter even when that spelling is computed
        // later. Reuse the original parameter binding and prove the domain;
        // never hoist the later expression or assume its value is available.
        for (unsigned i = 0; i < facts.parameters.size(); ++i) {
            Value value = facts.parameters[i];
            if (!value.getType().isInteger(1) || !dominance.dominates(value, anchor))
                continue;
            for (bool truth : {true, false}) {
                BoundaryTest test{BoundaryTest::BooleanParameter, i, truth};
                auto domain = condition(point, test);
                if (!domain)
                    return {};
                candidates.push_back({test, domain});
            }
        }
        if (candidates.size() > 32)
            return {};
        Guard guard;
        Relation remaining = target;
        auto admit = [&](const Relation& domain, Clause clause) -> bool {
            if (domain.isIntegerEmpty())
                return true;
            // A candidate already covered by prior clauses cannot contribute.
            // Test the intersection directly rather than subtracting and then
            // proving that the difference still contains the entire remainder.
            auto contribution = queries.normalize(domain.intersect(remaining));
            if (!contribution) { outcome = contribution.status; return false; }
            if (contribution.relation->isIntegerEmpty()) return true;
            auto included = queries.contains(target, domain);
            if (queryFailed(included))
                return false;
            if (included != QueryStatus::Proved)
                return true;
            auto rest = queries.subtract(remaining, domain);
            if (!rest) {
                outcome = rest.status;
                return false;
            }
            guard.push_back(std::move(clause));
            remaining = std::move(*rest.relation);
            return true;
        };
        unsigned coveredCandidates = 0;
        auto cover = [&]() -> std::optional<bool> {
            for (unsigned i = coveredCandidates; i < candidates.size(); ++i) {
                const auto& a = candidates[i];
                if (!admit(*a.domain, Clause{a.test}))
                    return std::nullopt;
                if (remaining.isIntegerEmpty())
                    return true;
            }
            for (unsigned i = 0; i < candidates.size(); ++i)
                for (unsigned j = std::max(i + 1, coveredCandidates); j < candidates.size(); ++j) {
                    auto domain = candidates[i].domain->intersect(*candidates[j].domain);
                    if (!admit(domain, Clause{candidates[i].test, candidates[j].test}))
                        return std::nullopt;
                    if (remaining.isIntegerEmpty())
                        return true;
                }
            // Target membership is immutable and accepted clauses only shrink
            // remaining. Previously tried singles/pairs never become useful
            // when new proposal forms are appended.
            coveredCandidates = candidates.size();
            return false;
        };
        auto covered = cover();
        if (!covered)
            return {};
        if (*covered)
            return guard;
        // A first use carried from a different branch can occur at an interior
        // iteration. Recover simple constant bounds from the actual required
        // domain, rather than enumerating special first/second-use recipes.
        // Emission is only cmp(iv, constant), with no new overflow-prone math.
        auto normalized = queries.normalize(remaining);
        if (!normalized) {
            outcome = normalized.status;
            return {};
        }
        std::set<std::tuple<unsigned, arith::CmpIPredicate, int64_t>> bounds;
        for (const auto& piece : normalized.relation->getAllDisjuncts()) {
            Simplex simplex(piece);
            if (simplex.isEmpty())
                continue;
            for (unsigned i = 0; i < facts.loops.size(); ++i) {
                auto loop = facts.loops[i];
                if (!loop->isProperAncestor(anchor))
                    continue;
                SmallVector<llvm::DynamicAPInt> objective(piece.getNumCols(), llvm::DynamicAPInt(0));
                objective[i + 1] = 1;
                std::optional<int64_t> lower;
                for (auto direction : {Simplex::Direction::Down, Simplex::Direction::Up}) {
                    uint64_t cost =
                        uint64_t(piece.getNumEqualities() + piece.getNumInequalities() + 1) * piece.getNumCols();
                    if (!queries.spend(cost)) {
                        outcome = QueryStatus::BudgetExhausted;
                        return {};
                    }
                    auto optimum = simplex.computeOptimum(direction, objective);
                    if (!optimum.isBounded())
                        continue;
                    // Rational extrema only propose constants. The existing
                    // INTEGER domain-equality query must prove the resulting
                    // guards, including parity, gaps and all parameter values.
                    auto bound = direction == Simplex::Direction::Down ? presburger::ceil(*optimum) :
                                                                         presburger::floor(*optimum);
                    if (bound < std::numeric_limits<int64_t>::min() || bound > std::numeric_limits<int64_t>::max())
                        continue;
                    int64_t value = int64_t(bound);
                    if (auto integer = dyn_cast<IntegerType>(loop.getInductionVar().getType()))
                        if (!llvm::isIntN(integer.getWidth(), value))
                            continue;
                    auto comparison =
                        direction == Simplex::Direction::Down ? arith::CmpIPredicate::sge : arith::CmpIPredicate::sle;
                    bounds.emplace(i, comparison, value);
                    if (direction == Simplex::Direction::Down)
                        lower = value;
                    else if (lower == value)
                        bounds.emplace(i, arith::CmpIPredicate::eq, value);
                }
            }
        }
        for (auto [loop, comparison, bound] : bounds) {
            BoundaryTest test{BoundaryTest::ConstantBound, loop, true, bound, comparison};
            auto domain = condition(point, test);
            if (!domain)
                return {};
            if (candidates.size() == 32)
                break;
            candidates.push_back({test, domain});
        }
        covered = cover();
        if (covered && *covered)
            return guard;
        if (!covered)
            return {};
        // General differences of available loop bounds and IVs. Thresholds
        // come from the requested integer domain, not from a slot count or a
        // kernel-specific last-use recipe. Prove arithmetic on the FULL anchor
        // domain before considering any threshold's true subset.
        const unsigned n = facts.dimensions();
        for (unsigned i = 0; i < facts.loops.size(); ++i) {
            auto loop = facts.loops[i];
            if (!loop->isProperAncestor(anchor))
                continue;
            for (bool fromUpper : {false, true}) {
                const auto* range = differenceRange(point, i, fromUpper);
                if (!range || !range->full) {
                    if (outcome == QueryStatus::BudgetExhausted) return {};
                    continue;
                }
                unsigned width = isa<IndexType>(loop.getInductionVar().getType()) ? 64 :
                                     cast<IntegerType>(loop.getInductionVar().getType()).getWidth();
                if (!llvm::isIntN(width, range->full->first) || !llvm::isIntN(width, range->full->second))
                    continue;
                // Only the full-ambient arithmetic qualification is cached.
                // Thresholds and extrema still follow this endpoint's changing
                // uncovered domain and cannot be reused across preparations.
                auto lift = [&](Relation domain) {
                    domain.insertVarInPlace(VarKind::Range, n);
                    return domain.intersect(range->definition);
                };
                auto wantedRange = lift(remaining);
                auto extrema = takeRange(wantedRange, n);
                if (!extrema) {
                    if (outcome == QueryStatus::BudgetExhausted) return {};
                    continue;
                }
                // Relaxed floor witnesses can put the rational endpoint at an
                // integer hole. Skip only proved-empty levels, with a fixed
                // bound; exact guard-domain cover remains the acceptance gate.
                for (auto comparison : {arith::CmpIPredicate::sge, arith::CmpIPredicate::sle}) {
                    int64_t bound = comparison == arith::CmpIPredicate::sge ? extrema->first : extrema->second;
                    for (unsigned step = 0; step < 8; ++step) {
                        auto slice = Relation::getEmpty(wantedRange.getSpace());
                        for (auto piece : wantedRange.getAllDisjuncts()) {
                            piece.addBound(BoundType::EQ, n, bound);
                            slice.unionInPlace(Relation(piece));
                        }
                        auto present = queries.normalize(slice);
                        if (!present) { outcome = present.status; return {}; }
                        if (!present.relation->isIntegerEmpty()) break;
                        if ((comparison == arith::CmpIPredicate::sge && bound == INT64_MAX) ||
                            (comparison == arith::CmpIPredicate::sle && bound == INT64_MIN)) break;
                        bound += comparison == arith::CmpIPredicate::sge ? 1 : -1;
                    }
                    if (!llvm::isIntN(width, bound)) continue;
                    BoundaryTest test{BoundaryTest::DifferenceBound, i, true, bound, comparison, fromUpper};
                    auto domain = condition(point, test);
                    if (!domain) return {};
                    if (!admit(*domain, Clause{test})) return {};
                    if (remaining.isIntegerEmpty()) return guard;
                    // Reuse already qualified predicate alternatives where a
                    // difference is needed only on part of the original path.
                    for (const auto& candidate : candidates) {
                        if (!admit(domain->intersect(*candidate.domain), Clause{test, candidate.test})) return {};
                        if (remaining.isIntegerEmpty()) return guard;
                    }
                }
            }
        }
        // At an enclosing boundary, a final participating lane can depend on
        // a parameter's residue. Recover only small divisors actually present
        // in the selected relation. Neither loop counts nor buffer depth are
        // consulted. A nonnegative bound is evaluated FIRST, so signed
        // remainder is mathematical modulo wherever its definition executes.
        std::set<int64_t> moduli;
        for (const auto& piece : remaining.getAllDisjuncts()) {
            auto divisions = piece.getLocalReprs();
            for (auto denominator : divisions.getDenoms())
                if (denominator >= 2 && denominator <= 16)
                    moduli.insert(int64_t(denominator));
        }
        for (unsigned s = 0; s < facts.parameters.size(); ++s) {
            Value value = facts.parameters[s];
            if (!dominance.dominates(value, anchor) || value.getType().isInteger(1)) continue;
            auto range = takeRange(remaining, n + s);
            if (!range) {
                if (outcome == QueryStatus::BudgetExhausted) return {};
                continue;
            }
            int64_t lower = range->first;
            for (unsigned step = 0; step < 8 && lower < INT64_MAX; ++step) {
                auto slice = Relation::getEmpty(remaining.getSpace());
                for (auto piece : remaining.getAllDisjuncts()) {
                    piece.addBound(BoundType::EQ, n + s, lower);
                    slice.unionInPlace(Relation(piece));
                }
                auto check = queries.normalize(slice);
                if (!check) { outcome = check.status; return {}; }
                if (!check.relation->isIntegerEmpty()) break;
                ++lower;
            }
            if (lower < 0) continue;
            unsigned width = isa<IndexType>(value.getType()) ? 64 : cast<IntegerType>(value.getType()).getWidth();
            if (!llvm::isIntN(width, lower)) continue;
            BoundaryTest bound{BoundaryTest::ParameterBound, s, true, lower, arith::CmpIPredicate::sge};
            auto nonnegative = condition(point, bound);
            if (!nonnegative) return {};
            if (!admit(*nonnegative, Clause{bound})) return {};
            if (remaining.isIntegerEmpty()) return guard;
            for (int64_t modulus : moduli) {
                if (!llvm::isIntN(width, modulus)) continue;
                for (int64_t residue = 0; residue < modulus; ++residue) {
                    BoundaryTest test{BoundaryTest::ParameterResidue, s, true, residue,
                                      arith::CmpIPredicate::eq, false, modulus};
                    auto domain = condition(point, test);
                    if (!domain || !admit(domain->intersect(*nonnegative), Clause{bound, test})) return {};
                    if (remaining.isIntegerEmpty()) return guard;
                }
            }
        }
        return {};
    }
    std::optional<std::pair<int64_t, int64_t>> takeRange(const Relation& relation, unsigned coordinate)
    {
        std::optional<std::pair<int64_t, int64_t>> range;
        for (const auto& piece : relation.getAllDisjuncts()) {
            if (!queries.spend(uint64_t(piece.getNumConstraints() + 1) * piece.getNumCols() * 2)) {
                outcome = QueryStatus::BudgetExhausted; return {};
            }
            Simplex simplex(piece);
            if (simplex.isEmpty()) continue;
            SmallVector<llvm::DynamicAPInt> objective(piece.getNumCols());
            objective[coordinate] = 1;
            auto lo = simplex.computeOptimum(Simplex::Direction::Down, objective);
            auto hi = simplex.computeOptimum(Simplex::Direction::Up, objective);
            if (!lo.isBounded() || !hi.isBounded()) return {};
            auto low = presburger::ceil(*lo), high = presburger::floor(*hi);
            if (low < INT64_MIN || high > INT64_MAX || low > high) return {};
            int64_t a = int64_t(low), b = int64_t(high);
            range = range ? std::make_pair(std::min(range->first, a), std::max(range->second, b)) : std::make_pair(a, b);
        }
        return range;
    }
    Value emitTest(OpBuilder& builder, Location location, BoundaryTest test) const
    {
        auto key = keyFor(test);
        auto& cache = emittedTests[builder.getBlock()];
        if (auto found = cache.find(key); found != cache.end()) {
            auto* definition = found->second.getDefiningOp();
            auto cursor = builder.getInsertionPoint();
            if (!definition || (definition->getBlock() == builder.getBlock() &&
                (cursor == builder.getBlock()->end() ||
                 (definition != &*cursor && definition->isBeforeInBlock(&*cursor)))))
                return found->second;
        }
        if (!test.truth) {
            test.truth = true;
            auto value = emitTest(builder, location, test);
            auto one = builder.create<arith::ConstantIntOp>(location, 1, 1);
            return cache[key] = builder.create<arith::XOrIOp>(location, value, one);
        }
        Value value;
        if (test.kind == BoundaryTest::Predicate)
            value = facts.predicates[test.loop].value;
        else if (test.kind == BoundaryTest::BooleanParameter)
            value = facts.parameters[test.loop];
        else if (test.kind == BoundaryTest::ParameterBound || test.kind == BoundaryTest::ParameterResidue) {
            value = facts.parameters[test.loop];
            if (test.kind == BoundaryTest::ParameterResidue) {
                auto divisor = builder.create<arith::ConstantOp>(location, builder.getIntegerAttr(value.getType(), test.modulus));
                value = builder.create<arith::RemSIOp>(location, value, divisor);
            }
            auto bound = builder.create<arith::ConstantOp>(location, builder.getIntegerAttr(value.getType(), test.bound));
            value = builder.create<arith::CmpIOp>(location, test.comparison, value, bound);
        }
        else if (test.kind == BoundaryTest::ConstantBound || test.kind == BoundaryTest::DifferenceBound) {
            auto loop = facts.loops[test.loop];
            Value iv = loop.getInductionVar();
            if (test.kind == BoundaryTest::DifferenceBound)
                iv = test.fromUpper ? builder.create<arith::SubIOp>(location, loop.getUpperBound(), iv) :
                                      builder.create<arith::SubIOp>(location, iv, loop.getLowerBound());
            auto constant =
                builder.create<arith::ConstantOp>(location, builder.getIntegerAttr(iv.getType(), test.bound));
            value = builder.create<arith::CmpIOp>(location, test.comparison, iv, constant);
        } else {
            auto loop = facts.loops[test.loop];
            if (test.kind == BoundaryTest::Nonempty)
                value = builder.create<arith::CmpIOp>(
                    location, arith::CmpIPredicate::slt, loop.getLowerBound(), loop.getUpperBound());
            else if (test.kind == BoundaryTest::First)
                value = builder.create<arith::CmpIOp>(
                    location, arith::CmpIPredicate::eq, loop.getInductionVar(), loop.getLowerBound());
            else {
                // The occurrence importer proves this induction increment
                // representable in the actual induction type.
                auto next = builder.create<arith::AddIOp>(location, loop.getInductionVar(), loop.getStep());
                value = builder.create<arith::CmpIOp>(
                    location, arith::CmpIPredicate::sge, next, loop.getUpperBound());
            }
        }
        return cache[key] = value;
    }
    Value emitGuard(OpBuilder& builder, Location location, const Guard& guard) const
    {
        if (guard.size() == 1 && guard.front().empty())
            return {};
        Value disjunction;
        for (const auto& clause : guard) {
            Value conjunction;
            for (const auto& test : clause) {
                Value value = emitTest(builder, location, test);
                conjunction = conjunction ? builder.create<arith::AndIOp>(location, conjunction, value) : value;
            }
            disjunction = disjunction ? builder.create<arith::OrIOp>(location, disjunction, conjunction) : conjunction;
        }
        return disjunction;
    }
};
} // namespace

bool Constructor::realize()
{
    StageProfile profile(queries, "guard_preparation");
    BoundaryLowering lowering(facts, queries);
    std::vector<Endpoint> endpoints;
    std::vector<Guard> barrierGuards;
    // Participation and predicate legality precede assignment, including entry,
    // first use, skipped/empty readers and final-use occurrences.
    for (unsigned i = 0; i < streams.size(); ++i) {
        auto& stream = streams[i];
        if (std::getenv("PTOAS_LOGICAL_TRACE"))
            llvm::errs() << "logical prepare publication " << stream.source << " work " << queries.work() << "\n";
        auto publication = lowering.prepare(stream.source, stream.matching.getDomainSet());
        if (!publication) {
            if (std::getenv("PTOAS_LOGICAL_TRACE")) {
                llvm::errs() << "unlowered publication at " << stream.source << "\n";
                stream.matching.getDomainSet().print(llvm::errs());
            }
            return expect(lowering.status(), "publication domain has no qualified boundary lowering");
        }
        endpoints.push_back({stream.source, true, i, std::move(*publication), stream.matching.getDomainSet()});
        auto targets = orders->acquisitionTargets(stream.matching, queries);
        if (!targets)
            return fail(ConstructionResult::AnalysisLimit, "acquisition endpoint indexing budget");
        for (unsigned p : targets->phases) {
            ++orders->acquisitionCandidates;
            auto domain = targets->domain.intersect(facts.domain(p).getRangeSet());
            if (domain.isIntegerEmpty())
                continue;
            if (std::getenv("PTOAS_LOGICAL_TRACE"))
                llvm::errs() << "logical prepare acquisition " << p << " work " << queries.work() << "\n";
            auto acquisition = lowering.prepare(p, domain);
            if (!acquisition) {
                if (std::getenv("PTOAS_LOGICAL_TRACE")) {
                    llvm::errs() << "unlowered acquisition at " << p << " from " << stream.source << "\n";
                    domain.print(llvm::errs());
                }
                return expect(lowering.status(), "acquisition domain has no qualified boundary lowering");
            }
            endpoints.push_back({p, false, i, std::move(*acquisition), std::move(domain)});
        }
    }
    for (const auto& barrier : barriers) {
        auto guard = lowering.prepare(barrier.point, barrier.domain);
        if (!guard)
            return expect(lowering.status(), "barrier domain has no qualified boundary lowering");
        barrierGuards.push_back(std::move(*guard));
    }
    uint64_t emissionAllowance = 4096;
    for (const auto& endpoint : endpoints)
        if (!chargeGuardEmission(endpoint.guard, emissionAllowance))
            return fail(ConstructionResult::AnalysisLimit, "guard emission size or depth limit");
    for (const auto& guard : barrierGuards)
        if (!chargeGuardEmission(guard, emissionAllowance))
            return fail(ConstructionResult::AnalysisLimit, "guard emission size or depth limit");
    auto supply = primitive();
    if (!supply)
        return false;
    // Preparing legal endpoint guards changes no selected logical action. Reuse
    // the established requirement receipt; actual emitted IR is still freshly
    // reconstructed and proved below, including its keys and token ownership.
    if (receiptVersion != planVersion || provedTargets.size() != lanes.size())
        return fail(ConstructionResult::InternalError, "logical requirement receipt is incomplete or stale");
    profile.next("allocation");
    // A single assignment interface sees every selected stream. Sharing is
    // permitted only with occurrence-specific consumption-before-rearm proof;
    // assignment does not alter publication or acquisition boundaries.
    std::map<std::tuple<Lane, Lane, unsigned>, Relation> assignments;
    // All assignment trials query the same logical completion plan. Key
    // proposals change the reuse requirement, not the payload handoffs.
    auto assignmentCompletion = completion(*supply);
    if (!assignmentCompletion)
        return false;
    for (auto& stream : streams) {
        bool assigned = false;
        for (unsigned k = 0; k < 8; ++k) {
            auto key = std::make_tuple(stream.pipes.first, stream.pipes.second, k);
            auto matching = stream.matching;
            if (auto found = assignments.find(key); found != assignments.end())
                matching.unionInPlace(found->second);
            auto participation = functional(matching);
            if (queryFailed(participation, "assignment participation query"))
                return false;
            if (participation != QueryStatus::Proved)
                continue;
            auto reuse = reuseSafe(matching, stream.pipes.first, *assignmentCompletion);
            if (queryFailed(reuse, "assignment consumption-before-rearm query"))
                return false;
            if (reuse != QueryStatus::Proved)
                continue;
            assignments.insert_or_assign(key, std::move(matching));
            stream.key = k;
            assigned = true;
            break;
        }
        if (!assigned)
            return fail(
                ConstructionResult::AllocationFailure,
                "no proved assignment within the event domain; boundaries retained");
    }
    profile.next("endpoint_normalization");
    // Only adjacent identical commands at the exact same insertion boundary
    // may coalesce. Concrete-key sharing alone does not identify a token: the
    // occurrence domains must be disjoint, preserving one action per execution.
    // Index boundaries once; do not search every endpoint against every other.
    std::map<std::pair<unsigned, bool>, SmallVector<unsigned>> atBoundary;
    for (unsigned i = 0; i < endpoints.size(); ++i)
        atBoundary[{endpoints[i].point, endpoints[i].publication}].push_back(i);
    std::vector<bool> omitted(endpoints.size(), false);
    // Optional simplification cannot spend the mandatory reconstruction budget.
    // Its actual work is charged back below; unknown/limit retains originals.
    RelationQueries normalization(std::min<uint64_t>(100000, queries.remainingWork() / 32));
    auto sameCommand = [&](unsigned a, unsigned b) {
        const auto& x = streams[endpoints[a].stream];
        const auto& y = streams[endpoints[b].stream];
        return x.pipes == y.pipes && x.key == y.key;
    };
    for (const auto& [boundary, indices] : atBoundary) {
        for (unsigned begin = 0; begin < indices.size();) {
            unsigned end = begin + 1;
            while (end < indices.size() && sameCommand(indices[begin], indices[end])) ++end;
            if (end - begin > 1) {
                using Merged = std::pair<PresburgerSet, Guard>;
                // Balanced unions keep structural copying O(E log E), rather
                // than repeatedly copying the growing prefix. Exact overlap
                // checks still have relation-dependent cost and share the budget.
                std::function<std::optional<Merged>(unsigned, unsigned)> merge =
                    [&](unsigned a, unsigned b) -> std::optional<Merged> {
                    if (b - a == 1)
                        return Merged{endpoints[indices[a]].domain, endpoints[indices[a]].guard};
                    unsigned middle = a + (b - a) / 2;
                    auto left = merge(a, middle), right = merge(middle, b);
                    if (!left || !right) return {};
                    uint64_t aPieces = left->first.getNumDisjuncts(), bPieces = right->first.getNumDisjuncts();
                    if (aPieces > 128 || bPieces > 128 || aPieces * bPieces > 128) return {};
                    uint64_t rows = 1, columns = 1;
                    for (const auto* side : {&left->first, &right->first})
                        for (const auto& piece : side->getAllDisjuncts()) {
                            rows = std::max(rows, uint64_t(piece.getNumConstraints()));
                            columns = std::max(columns, uint64_t(piece.getNumCols()));
                        }
                    if (rows > 128 || columns > 64) return {};
                    uint64_t cells = aPieces * bPieces * (2 * rows + 1) * (2 * columns + 1);
                    if (cells > 32768 || !normalization.spend(cells + 1)) return {};
                    auto overlap = normalization.normalize(left->first.intersect(right->first));
                    if (!overlap || !overlap.relation->isIntegerEmpty()) return {};
                    left->first.unionInPlace(right->first);
                    left->second.append(right->second.begin(), right->second.end());
                    return left;
                };
                auto joined = merge(begin, end);
                if (joined) {
                    auto ambient = facts.domain(boundary.first).getRangeSet();
                    auto unconditional = normalization.contains(joined->first, ambient);
                    if (unconditional == QueryStatus::Unsupported || unconditional == QueryStatus::BudgetExhausted) {
                        begin = end;
                        continue;
                    }
                    if (unconditional == QueryStatus::Proved) joined->second = Guard{Clause{}};
                    uint64_t oldAllowance = 4096, newAllowance = 4096;
                    bool oldFits = true;
                    for (unsigned j = begin; j < end; ++j)
                        oldFits &= chargeGuardEmission(endpoints[indices[j]].guard, oldAllowance);
                    // Combining short-circuit clauses can grow a decision tree.
                    // Retain the original endpoints unless the emitted bound improves.
                    if (oldFits && chargeGuardEmission(joined->second, newAllowance) &&
                        newAllowance > oldAllowance) {
                        auto& first = endpoints[indices[begin]];
                        first.domain = std::move(joined->first);
                        first.guard = std::move(joined->second);
                        for (unsigned j = begin + 1; j < end; ++j) omitted[indices[j]] = true;
                    }
                }
            }
            begin = end;
        }
    }
    if (!queries.spend(normalization.work()))
        return fail(ConstructionResult::AnalysisLimit, "endpoint normalization accounting");
    profile.next("emission");
    auto at = [&](unsigned p, bool after, const Guard& guard, auto action) {
        auto* anchor = facts.points[p].operation;
        OpBuilder builder(anchor);
        if (after)
            builder.setInsertionPointAfter(anchor);
        bool shortCircuit = llvm::any_of(guard, [](const Clause& clause) {
            return llvm::any_of(clause, [](const BoundaryTest& test) {
                return test.kind == BoundaryTest::ParameterResidue;
            });
        });
        if (shortCircuit) {
            // An ordered DNF: failed clauses try the next one; the first true
            // clause emits exactly one action. Remainders execute only beneath
            // their nonnegative bound, which fresh definition-domain import
            // independently qualifies. No result-bearing scalar if is needed.
            std::function<void(OpBuilder&, unsigned, unsigned)> emit;
            emit = [&](OpBuilder& b, unsigned clause, unsigned literal) {
                if (clause == guard.size()) return;
                if (literal == guard[clause].size()) { action(b, anchor->getLoc()); return; }
                auto condition = lowering.emitGuard(b, anchor->getLoc(), Guard{Clause{guard[clause][literal]}});
                auto branch = b.create<scf::IfOp>(anchor->getLoc(), condition, true);
                OpBuilder yes = OpBuilder::atBlockBegin(&branch.getThenRegion().front());
                emit(yes, clause, literal + 1);
                OpBuilder no = OpBuilder::atBlockBegin(&branch.getElseRegion().front());
                emit(no, clause + 1, 0);
            };
            emit(builder, 0, 0);
            return;
        }
        Value condition = lowering.emitGuard(builder, anchor->getLoc(), guard);
        if (condition) {
            auto branch = builder.create<scf::IfOp>(anchor->getLoc(), condition, false);
            builder.setInsertionPointToStart(&branch.getThenRegion().front());
        }
        action(builder, anchor->getLoc());
    };
    for (unsigned i = 0; i < endpoints.size(); ++i) {
        if (omitted[i]) continue;
        const auto& endpoint = endpoints[i];
        auto& stream = streams[endpoint.stream];
        at(endpoint.point, endpoint.publication, endpoint.guard, [&](OpBuilder& builder, Location location) {
            auto source = PipeAttr::get(function.getContext(), static_cast<PIPE>(stream.pipes.first));
            auto target = PipeAttr::get(function.getContext(), static_cast<PIPE>(stream.pipes.second));
            auto key = EventAttr::get(function.getContext(), static_cast<EVENT>(stream.key));
            if (endpoint.publication)
                builder.create<SetFlagOp>(location, source, target, key);
            else
                builder.create<WaitFlagOp>(location, source, target, key);
        });
    }
    for (unsigned i = 0; i < barriers.size(); ++i) {
        auto& barrier = barriers[i];
        at(barrier.point, false, barrierGuards[i], [&](OpBuilder& builder, Location location) {
            builder.create<BarrierOp>(location, PipeAttr::get(function.getContext(), static_cast<PIPE>(barrier.pipe)));
        });
    }
    // The supported single physical function returns only after this explicit
    // drain. Do not use the auto-sync-tail marker: its optional MTE3->S hint is
    // not a qualified substitute for whole-kernel retirement. This policy also
    // covers zero-trip paths with a preload outside the loop. PIPE_ALL does not
    // consume event tokens; their participation remains independently checked.
    OpBuilder retirementBuilder(facts.points[retirement.point].operation);
    retirementBuilder.create<BarrierOp>(function.getLoc(), PipeAttr::get(function.getContext(), PIPE::PIPE_ALL));
    if (emissionMutation)
        emissionMutation(function);
    if (failed(verify(function)))
        return fail(ConstructionResult::InternalError, "malformed logical emission");
    profile.next("reconstruction");
    return reconstruct();
}

bool Constructor::reconstruct()
{
    StageProfile profile(queries, "reconstruct_effects");
    if (!payload.preserved(function, [](Operation* op) {
            // The constructor owns only events, barriers and their scalar
            // guards. Extra allocations/views/resources are payload changes,
            // even when the physical-effect translator would not count them.
            return isa<
                SetFlagOp, WaitFlagOp, BarrierOp, scf::IfOp, scf::YieldOp, arith::ConstantOp, arith::CmpIOp,
                arith::AddIOp, arith::SubIOp, arith::RemSIOp, arith::AndIOp, arith::OrIOp, arith::XOrIOp>(op);
        }))
        return fail(
            ConstructionResult::InternalError, "emission changed original payload, control or allocation contract");
    // Re-translate actual payload effects after emission. Insertion metadata and
    // the selected plan's claims are not semantic input to this reconstruction.
    if (qualifySyncPhysicalAddresses(function).status == SyncAddressAdmission::Rejected)
        return fail(ConstructionResult::Unproved, "reconstructed physical-address admission failed");
    SyncIRs rebuiltIR;
    Buffer2MemInfoMap rebuiltBuffers;
    PTOIRTranslator translator(rebuiltIR, memory, rebuiltBuffers, function, SyncAnalysisMode::NORMALSYNC);
    if (failed(translator.Build()))
        return fail(ConstructionResult::Unproved, "reconstructed physical translation failed");
    std::map<Operation*, const CompoundInstanceElement*> rebuiltPhases;
    for (const auto& element : rebuiltIR)
        if (auto* phase = dyn_cast<CompoundInstanceElement>(element.get())) {
            if (!rebuiltPhases.emplace(phase->elementOp, phase).second)
                return fail(ConstructionResult::Unsupported, "reconstructed multi-phase operation");
        }
    if (rebuiltPhases.size() != phases.size())
        return fail(ConstructionResult::InternalError, "emission changed represented physical phases");
    std::map<const BaseMemInfo*, const BaseMemInfo*> rebuiltAccesses;
    auto sameEffects = [&](const auto& a, const auto& b) {
        if (a.size() != b.size())
            return false;
        for (unsigned i = 0; i < a.size(); ++i) {
            if (!(*a[i] == *b[i]))
                return false;
            rebuiltAccesses.emplace(a[i], b[i]);
        }
        return true;
    };
    for (auto* old : phases) {
        auto found = rebuiltPhases.find(old->elementOp);
        if (found == rebuiltPhases.end() || old->kPipeValue != found->second->kPipeValue ||
            !sameEffects(old->useVec, found->second->useVec) || !sameEffects(old->defVec, found->second->defVec))
            return fail(ConstructionResult::InternalError, "emission changed physical access contract");
    }
    // Reconstruct the actual retirement mechanism, not a planner certificate.
    // The current policy requires an unconditional, unmarked ALL immediately
    // before the original return. This also excludes any later payload issue or
    // key operation and does not depend on ordinary MTE1/V wait semantics.
    auto* returnOp = facts.points[retirement.point].operation;
    auto terminalDrain = dyn_cast_or_null<BarrierOp>(returnOp->getPrevNode());
    if (!terminalDrain || terminalDrain.getPipe().getPipe() != PIPE::PIPE_ALL ||
        terminalDrain->hasAttr("pto.auto_sync_tail_barrier") || terminalDrain->hasAttr("pto.auto_sync_tail_hint"))
        return fail(ConstructionResult::Unproved, "explicit terminal retirement drain unavailable");
    SmallVector<Operation*> points;
    for (const auto& point : facts.points)
        points.push_back(point.operation);
    using Key = std::tuple<Lane, Lane, unsigned>;
    struct Events {
        SmallVector<unsigned> publications, acquisitions;
    };
    std::map<Key, Events> groups;
    SmallVector<std::pair<unsigned, Lane>> actualBarriers;
    unsigned actualRetirement = 0;
    function.walk([&](Operation* op) {
        if (auto set = dyn_cast<SetFlagOp>(op)) {
            auto key =
                Key{static_cast<Lane>(set.getSrcPipe().getPipe()), static_cast<Lane>(set.getDstPipe().getPipe()),
                    unsigned(set.getEventId().getEvent())};
            groups[key].publications.push_back(points.size());
            points.push_back(op);
        } else if (auto wait = dyn_cast<WaitFlagOp>(op)) {
            auto key =
                Key{static_cast<Lane>(wait.getSrcPipe().getPipe()), static_cast<Lane>(wait.getDstPipe().getPipe()),
                    unsigned(wait.getEventId().getEvent())};
            groups[key].acquisitions.push_back(points.size());
            points.push_back(op);
        } else if (auto barrier = dyn_cast<BarrierOp>(op)) {
            if (barrier == terminalDrain)
                actualRetirement = points.size();
            else
                actualBarriers.push_back({points.size(), static_cast<Lane>(barrier.getPipe().getPipe())});
            points.push_back(op);
        }
    });
    profile.next("reconstruct_occurrences");
    auto actual = SyncOccurrences::build(function, points, slotSelectors);
    if (!queries.spend(actual.work))
        return fail(ConstructionResult::AnalysisLimit, "emitted occurrence import work budget");
    if (!actual.complete)
        return fail(
            actual.limitExceeded ? ConstructionResult::AnalysisLimit : ConstructionResult::Unproved,
            "emitted occurrence import: " + actual.reason);
    if (actual.loops != facts.loops || actual.parameters != facts.parameters)
        return fail(ConstructionResult::Unproved, "emitted occurrence bindings differ from input");
    for (unsigned p = 0; p < facts.points.size(); ++p)
        if (!expect(
                queries.contains(actual.domain(p), facts.domain(p)),
                "emitted payload domain missing input occurrences") ||
            !expect(
                queries.contains(facts.domain(p), actual.domain(p)), "emitted payload domain adds input occurrences"))
            return false;
    // Re-establish every physical/scalar fact used to narrow a may-conflict
    // from freshly translated effects and actual emitted control. Equal event
    // counts or a selected protocol's certificate cannot justify slot omission.
    std::map<const BaseMemInfo*, SyncPhysicalSlotMapping> actualMappings;
    llvm::DenseSet<SelectorPoint> checkedSelectors, checkedCompactSelectors, checkedRanges;
    auto usedSlots = usedCompactSlots;
    for (const auto& [key, domains] : usedSlotDomains) usedSlots.insert(key);
    for (const auto& key : usedSlots) {
        auto [point, oldMemory] = key;
        auto oldMapping = slotMappings.find(oldMemory);
        auto rebuilt = rebuiltAccesses.find(oldMemory);
        if (oldMapping == slotMappings.end() || !oldMapping->second || rebuilt == rebuiltAccesses.end())
            return fail(ConstructionResult::InternalError, "missing original slot access identity");
        const auto& original = *oldMapping->second;
        auto mapping = actualMappings.find(oldMemory);
        if (mapping == actualMappings.end()) {
            if (!queries.spend(estimateSyncPhysicalSlotQualificationWork(rebuilt->second, rebuiltBuffers)))
                return fail(ConstructionResult::AnalysisLimit, "emitted slot mapping budget");
            auto fresh = qualifySyncPhysicalSlots(rebuilt->second, rebuiltBuffers);
            if (!fresh || fresh->selector != original.selector || fresh->bases != original.bases ||
                fresh->bytes != original.bytes || fresh->scope != original.scope)
                return fail(ConstructionResult::Unproved, "emitted physical slot mapping changed");
            mapping = actualMappings.emplace(oldMemory, std::move(*fresh)).first;
        }
        if (original.selector) {
            SelectorPoint selector{original.selector, {point, unsigned(original.bases.size())}};
            if (!checkedRanges.contains(selector)) {
                if (!proveSlotRange(actual, point, mapping->second)) {
                    if (result.status == ConstructionResult::AnalysisLimit) return false;
                    return fail(ConstructionResult::Unproved, "emitted selector range unavailable");
                }
                checkedRanges.insert(selector);
            }
            if (usedCompactSlots.count(key) && !checkedCompactSelectors.contains(selector)) {
                if (!queries.spend(1)) return fail(ConstructionResult::AnalysisLimit, "emitted scalar identity budget");
                auto before = facts.scalarExpressions.find(original.selector);
                auto after = actual.scalarExpressions.find(original.selector);
                // Point domains and the original SSA/loop/parameter bindings
                // were independently checked above. Under those bindings,
                // identical normalized expressions preserve EACH selector's
                // value graph. Merely preserving pairwise selector equality
                // could miss simultaneous permutations of both selectors.
                if (before == facts.scalarExpressions.end() || after == actual.scalarExpressions.end() ||
                    before->second != after->second)
                    return fail(ConstructionResult::Unproved, "emitted selector value changed");
                checkedCompactSelectors.insert(selector);
            }
        }
        auto oldDomains = usedSlotDomains.find(key);
        if (oldDomains == usedSlotDomains.end()) continue;
        const auto& domains = oldDomains->second;
        std::vector<Relation> freshDomains;
        if (original.selector) {
            SelectorPoint selector{original.selector, {point, unsigned(original.bases.size())}};
            // Each memory still has its own actual physical mapping checked
            // above. The exact scalar equivalence is shared only WITHIN this
            // fresh reconstruction, independent of any planner receipt.
            if (checkedSelectors.contains(selector)) continue;
            auto fresh = enumerateSlotDomains(actual, point, mapping->second);
            if (!fresh) {
                if (result.status == ConstructionResult::AnalysisLimit) return false;
                return fail(ConstructionResult::Unproved, "emitted selector domain unavailable");
            }
            freshDomains = std::move(*fresh);
        } else freshDomains.push_back(actual.domain(point));
        if (domains.size() != freshDomains.size())
            return fail(ConstructionResult::Unproved, "emitted slot domain count changed");
        for (unsigned slot = 0; slot < domains.size(); ++slot)
            if (!expect(queries.contains(freshDomains[slot], domains[slot]), "emitted slot domain lost occurrences") ||
                !expect(queries.contains(domains[slot], freshDomains[slot]), "emitted slot domain adds occurrences"))
                return false;
        if (original.selector)
            checkedSelectors.insert({original.selector, {point, unsigned(original.bases.size())}});
    }
    std::map<std::pair<unsigned, unsigned>, Relation> actualOrder;
    auto before = [&](unsigned a, unsigned b) -> std::optional<Relation> {
        auto key = std::make_pair(a, b);
        if (auto found = actualOrder.find(key); found != actualOrder.end())
            return found->second;
        auto relation = take(actual.ordered(a, b));
        if (!relation)
            return {};
        auto normalized = take(queries.normalize(*relation));
        if (normalized)
            actualOrder.emplace(key, *normalized);
        return normalized;
    };
    auto supply = empty();
    // Preserve and check each immutable retirement requirement under its actual
    // guards/invocations. The terminal drain's all-pipeline contract establishes
    // completion across this composed cut; it supplies no event-consumption
    // facts and is not added to the ordinary payload completion relation.
    auto drainToReturn = before(actualRetirement, retirement.point);
    if (!drainToReturn)
        return false;
    for (const auto& r : requirements)
        if (r.kind == Requirement::Retirement) {
            auto prefix = before(r.source, actualRetirement);
            if (!prefix)
                return false;
            auto retired = compose(*prefix, *drainToReturn);
            if (!retired || !expect(
                    queries.contains(*retired, r.occurrences), "reconstructed retirement misses payload occurrences"))
                return false;
        }
    // Index clear block intervals once in each direction. An original payload
    // on the same lane in this invocation is the exact adjacent cut; nested
    // control invalidates the shortcut and uses full occurrence queries below.
    std::map<std::pair<Operation*, Lane>, unsigned> previousPayload, nextPayload;
    std::set<Block*> blocks;
    for (const auto& point : actual.points) blocks.insert(point.operation->getBlock());
    for (Block* block : blocks) {
        std::map<Lane, unsigned> last;
        auto visit = [&](Operation& op, auto& index) {
            if (op.getNumRegions()) { last.clear(); return; }
            auto found = facts.ids.find(&op);
            if (found != facts.ids.end() && found->second < lanes.size())
                last[lanes[found->second]] = found->second;
            else if (isa<SetFlagOp, WaitFlagOp, BarrierOp>(&op))
                for (auto [lane, point] : last) index.emplace(std::make_pair(&op, lane), point);
        };
        for (auto& op : *block) visit(op, previousPayload);
        last.clear();
        for (auto& op : llvm::reverse(*block)) visit(op, nextPayload);
    }
    auto sameInvocation = [&](Relation relation) -> std::optional<Relation> {
        IntegerRelation equal(relation.getSpace());
        for (unsigned iv = 1; iv < facts.dimensions(); ++iv) {
            SmallVector<int64_t> row(equal.getNumCols());
            row[iv] = 1; row[facts.dimensions() + iv] = -1;
            equal.addEquality(row);
        }
        return take(queries.normalize(relation.intersect(Relation(equal))));
    };
    auto adjacentCut = [&](ArrayRef<unsigned> eventPoints, Lane lane, bool source) -> std::optional<Relation> {
        auto result = empty();
        const auto& index = source ? previousPayload : nextPayload;
        // Check all endpoints before issuing queries. A missing neighbour is
        // an unsupported shortcut, never an empty source prefix or target cut.
        for (unsigned event : eventPoints)
            if (!index.count({actual.points[event].operation, lane})) return {};
        for (unsigned event : eventPoints) {
            unsigned payload = index.at({actual.points[event].operation, lane});
            auto edge = source ? before(payload, event) : before(event, payload);
            if (!edge) return {};
            auto same = sameInvocation(std::move(*edge));
            if (!same) return {};
            result.unionInPlace(*same);
        }
        return result;
    };
    profile.next("reconstruct_events");
    std::vector<std::pair<Key, Relation>> actualMatching;
    for (const auto& [key, events] : groups) {
        auto [sourceLane, targetLane, keyNumber] = key;
        if (events.acquisitions.empty()) return fail(ConstructionResult::Unproved, "emitted publication lacks acquisition");
        if (events.publications.empty()) return fail(ConstructionResult::Unproved, "emitted acquisition lacks publication");
        if (keyNumber >= 8 || sourceLane == targetLane || !orders->byLane.count(sourceLane) ||
            !orders->byLane.count(targetLane))
            return fail(ConstructionResult::InternalError, "invalid emitted event domain");
        auto preceding = empty(), pubOrder = empty();
        auto pubDomain = PresburgerSet::getEmpty(actual.domain(0).getRangeSet().getSpace());
        auto waitDomain = pubDomain;
        for (unsigned p : events.publications) {
            pubDomain.unionInPlace(actual.domain(p).getRangeSet());
            for (unsigned w : events.acquisitions) {
                auto edge = before(p, w);
                if (!edge)
                    return false;
                preceding.unionInPlace(*edge);
            }
            for (unsigned q : events.publications) {
                auto edge = before(p, q);
                if (!edge)
                    return false;
                pubOrder.unionInPlace(*edge);
            }
        }
        for (unsigned w : events.acquisitions)
            waitDomain.unionInPlace(actual.domain(w).getRangeSet());
        auto matched = take(queries.latestSources(preceding, pubOrder));
        if (!matched)
            return false;
        auto inverse = *matched;
        inverse.inverse();
        auto consumers = compose(inverse, *matched), producers = compose(*matched, inverse);
        if (!consumers || !producers)
            return false;
        auto diagonal = empty();
        for (unsigned p : events.publications)
            diagonal.unionInPlace(actual.identity(p));
        for (unsigned w : events.acquisitions)
            diagonal.unionInPlace(actual.identity(w));
        if (!expect(queries.contains(diagonal, *consumers), "emitted publication consumed multiple times") ||
            !expect(queries.contains(diagonal, *producers), "emitted acquisition has multiple publications") ||
            !expect(queries.contains(matched->getDomainSet(), pubDomain), "emitted publication lacks acquisition") ||
            !expect(queries.contains(matched->getRangeSet(), waitDomain), "emitted acquisition lacks publication"))
            return false;
        auto lastSource = adjacentCut(events.publications, sourceLane, true);
        auto firstTarget = adjacentCut(events.acquisitions, targetLane, false);
        if (!lastSource) {
            auto prefix = empty();
            for (unsigned p : orders->byLane.at(sourceLane))
                for (unsigned publication : events.publications) {
                    auto edge = before(p, publication);
                    if (!edge) return false;
                    prefix.unionInPlace(*edge);
                }
            auto sourceBefore = selectedOrder(prefix.getDomainSet(), prefix.getDomainSet());
            if (!sourceBefore) return false;
            lastSource = take(queries.latestSources(prefix, *sourceBefore));
        }
        if (!firstTarget) {
            auto suffix = empty();
            for (unsigned p : orders->byLane.at(targetLane))
                for (unsigned acquisition : events.acquisitions) {
                    auto edge = before(acquisition, p);
                    if (!edge) return false;
                    suffix.unionInPlace(*edge);
                }
            auto targetBefore = selectedOrder(suffix.getRangeSet(), suffix.getRangeSet());
            if (!targetBefore) return false;
            firstTarget = take(queries.firstTargets(suffix, *targetBefore));
        }
        if (!lastSource || !firstTarget) return false;
        // Payload-cut relations must not erase actual notifications. Recover
        // every endpoint, and require distinct publications on this concrete
        // key to have distinct preceding payload cuts. Otherwise two balanced
        // episodes can collapse into one logical edge and evade the later
        // consumption-before-rearm challenge. Since lastSource is the actual
        // greatest preceding cut, this injectivity also preserves its order.
        if (!expect(queries.contains(lastSource->getRangeSet(), pubDomain),
                    "emitted publication has no recovered payload cut") ||
            !expect(queries.contains(firstTarget->getDomainSet(), waitDomain),
                    "emitted acquisition has no recovered payload cut")) return false;
        auto inverseCut = *lastSource;
        inverseCut.inverse();
        auto publicationsPerCut = compose(inverseCut, *lastSource);
        if (!publicationsPerCut ||
            !expect(queries.contains(diagonal, *publicationsPerCut),
                    "emitted publications collapse onto one payload cut")) return false;
        // Compare freshly recovered cuts with the selected boundaries below.
        // Same-block adjacency does not trust stream identities or certificates.
        auto a = compose(*lastSource, *matched);
        if (!a)
            return false;
        auto logical = compose(*a, *firstTarget);
        if (!logical)
            return false;
        auto intended = empty();
        for (const auto& stream : streams)
            if (stream.pipes == Domain{sourceLane, targetLane} && stream.key == int(keyNumber))
                intended.unionInPlace(stream.matching);
        if (!expect(queries.contains(intended, *logical), "realization broadened selected handoff boundary") ||
            !expect(queries.contains(*logical, intended), "realization lost selected handoff boundary"))
            return false;
        supply.unionInPlace(*logical);
        actualMatching.push_back({key, std::move(*logical)});
    }
    profile.next("reconstruct_barriers");
    if (actualBarriers.size() != barriers.size())
        return fail(ConstructionResult::InternalError, "emission changed selected barrier inventory");
    std::set<unsigned> checkedBarriers;
    std::map<unsigned, SmallVector<unsigned>> selectedBarriers;
    for (unsigned i = 0; i < barriers.size(); ++i) selectedBarriers[barriers[i].point].push_back(i);
    for (auto [barrier, pipe] : actualBarriers) {
        if (!orders->byLane.count(pipe)) return fail(ConstructionResult::Unproved, "unrepresented emitted barrier lane");
        // A complete clear interval permits a compact independent challenge.
        // Recover the emitted cut, then derive the original cut from physical
        // input neighbours and the retained domain. Never trust barrier.logical.
        auto emittedPrevious = previousPayload.find({actual.points[barrier].operation, pipe});
        auto emittedNext = nextPayload.find({actual.points[barrier].operation, pipe});
        if (emittedPrevious != previousPayload.end() && emittedNext != nextPayload.end()) {
            auto originalPrevious = orders->clearPredecessor.find(emittedNext->second);
            auto candidates = selectedBarriers.find(emittedNext->second);
            if (originalPrevious != orders->clearPredecessor.end() && candidates != selectedBarriers.end()) {
                SmallVector<unsigned> singleton{barrier};
                auto last = adjacentCut(singleton, pipe, true), first = adjacentCut(singleton, pipe, false);
                if (!last || !first) return false;
                auto cut = compose(*last, *first);
                auto original = order(originalPrevious->second, emittedNext->second);
                if (!cut || !original) return false;
                original = sameInvocation(std::move(*original));
                if (!original) return false;
                bool matched = false;
                for (unsigned i : candidates->second) {
                    if (checkedBarriers.count(i) || barriers[i].pipe != pipe) continue;
                    auto intended = original->intersectRange(barriers[i].domain);
                    auto same = queries.contains(intended, *cut);
                    if (same == QueryStatus::Proved) same = queries.contains(*cut, intended);
                    if (queryFailed(same, "reconstructed adjacent barrier cut query")) return false;
                    if (same != QueryStatus::Proved) continue;
                    checkedBarriers.insert(i); matched = true; break;
                }
                if (!matched) return fail(ConstructionResult::Unproved, "realization changed adjacent barrier cut");
                supply.unionInPlace(*cut);
                continue;
            }
        }
        auto prefix = empty(), suffix = empty();
        for (unsigned p : orders->byLane.at(pipe)) {
                auto a = before(p, barrier), b = before(barrier, p);
                if (!a || !b)
                    return false;
                prefix.unionInPlace(*a);
                suffix.unionInPlace(*b);
            }
        auto sourceOrder = selectedOrder(prefix.getDomainSet(), prefix.getDomainSet());
        auto targetOrder = selectedOrder(suffix.getRangeSet(), suffix.getRangeSet());
        if (!sourceOrder || !targetOrder) return false;
        auto last = take(queries.latestSources(prefix, *sourceOrder));
        if (!last)
            return false;
        auto first = take(queries.firstTargets(suffix, *targetOrder));
        if (!first)
            return false;
        auto actualPrefix = compose(prefix, *first);
        if (!actualPrefix)
            return false;
        bool matchedBarrier = false;
        auto candidatePoints = orders->phaseIds(first->getRangeSet(), queries);
        if (!candidatePoints) return fail(ConstructionResult::AnalysisLimit, "barrier candidate index budget");
        SmallVector<unsigned> candidates;
        for (unsigned point : *candidatePoints) {
            auto found = selectedBarriers.find(point);
            if (found != selectedBarriers.end()) candidates.append(found->second);
        }
        for (unsigned i : candidates) {
            if (checkedBarriers.count(i) || barriers[i].pipe != pipe) continue;
            auto domain = queries.contains(first->getRangeSet(), barriers[i].domain);
            if (queryFailed(domain, "reconstructed barrier domain query"))
                return false;
            if (domain != QueryStatus::Proved)
                continue;
            auto exact = queries.contains(barriers[i].domain, first->getRangeSet());
            if (queryFailed(exact, "reconstructed barrier domain query"))
                return false;
            if (exact != QueryStatus::Proved)
                continue;
            auto beforeLane = selectedOrder(laneDomain(pipe), barriers[i].domain);
            if (!beforeLane) return false;
            auto intendedPrefix = beforeLane->intersectRange(barriers[i].domain);
            if (!expect(
                    queries.contains(intendedPrefix, *actualPrefix), "realization broadened selected barrier cut") ||
                !expect(queries.contains(*actualPrefix, intendedPrefix), "realization lost selected barrier cut"))
                return false;
            checkedBarriers.insert(i);
            matchedBarrier = true;
            break;
        }
        if (!matchedBarrier)
            return fail(ConstructionResult::Unproved, "realization changed selected barrier domain");
        auto cut = compose(*last, *first);
        if (!cut)
            return false;
        supply.unionInPlace(*cut);
    }
    profile.next("reconstruct_completion_reuse");
    auto completed = completion(supply);
    if (!completed)
        return false;
    if (!expect(proveRequirements(*completed), "reconstructed payload ordering unavailable"))
        return false;
    for (const auto& [key, matching] : actualMatching)
        if (!expect(
                reuseSafe(matching, std::get<0>(key), *completed),
                "reconstructed consumption-before-rearm unavailable"))
            return false;
    return true;
}

namespace {
ConstructionResult construct(
    func::FuncOp function, InsertSyncGMAliasMode gm, uint64_t budget,
    llvm::function_ref<void(func::FuncOp)> mutate = {}, testing::RequirementObserver observe = {})
{
    OwningOpRef<ModuleOp> stage(ModuleOp::create(function.getLoc()));
    SmallVector<ModuleOp> ancestors;
    for (Operation* op = function->getParentOp(); op; op = op->getParentOp())
        if (auto module = dyn_cast<ModuleOp>(op))
            ancestors.push_back(module);
    ModuleOp parent = *stage;
    for (auto module : llvm::reverse(ancestors)) {
        auto child = ModuleOp::create(module.getLoc());
        child->setAttrs(module->getAttrs());
        parent.getBody()->push_back(child);
        parent = child;
    }
    IRMapping mapping;
    auto working = cast<func::FuncOp>(function->clone(mapping));
    parent.getBody()->push_back(working);
    auto result = Constructor(working, gm, budget, mutate, observe).run();
    if (result.status == ConstructionResult::Applied)
        function.getBody().takeBody(working.getBody());
    return result;
}
} // namespace

ConstructionResult mlir::pto::logical_sync::constructLogicalSync(
    func::FuncOp function, InsertSyncGMAliasMode gm, bool useMmad, uint64_t budget)
{
    (void)useMmad; // Qualified intrinsic ACC discharge is a later native milestone.
    return construct(function, gm, budget);
}

ConstructionResult mlir::pto::logical_sync::testing::constructWithEmissionMutation(
    func::FuncOp function, InsertSyncGMAliasMode gm, uint64_t budget, llvm::function_ref<void(func::FuncOp)> mutate,
    testing::RequirementObserver observe)
{
    return construct(function, gm, budget, mutate, observe);
}

bool mlir::pto::logical_sync::testing::guardEmissionFits(
    ArrayRef<unsigned> clauseSizes, bool shortCircuit, uint64_t allowance)
{
    Guard guard;
    for (unsigned size : clauseSizes) {
        if (size > 65 || guard.size() >= 65) return false;
        guard.emplace_back(size, BoundaryTest{shortCircuit ? BoundaryTest::ParameterResidue : BoundaryTest::First,
                                              0, true});
    }
    return chargeGuardEmission(guard, allowance);
}

bool mlir::pto::logical_sync::testing::checkBoundaryConditions(const SyncOccurrences& facts, uint64_t budget)
{
    RelationQueries queries(budget);
    // The production target-index helper must depend on represented endpoint
    // pieces, not the number of unrelated phases. An unknown phase coordinate
    // is deliberately the dense control and must retain the whole universe.
    for (unsigned streams : {1, 4, 8}) {
        std::optional<uint64_t> indexedWork;
        for (unsigned phases : {8, 32, 128}) {
            NativeOrder order;
            order.lanes.resize(phases);
            uint64_t start = queries.work(), candidates = 0;
            for (unsigned s = 0; s < streams; ++s) {
                IntegerRelation piece(PresburgerSpace::getRelationSpace(1, 1));
                piece.addBound(BoundType::EQ, 0, 0);
                piece.addBound(BoundType::EQ, 1, s);
                auto targets = order.acquisitionTargets(Relation(piece), queries);
                if (!targets || targets->phases != std::set<unsigned>{s}) return false;
                candidates += targets->phases.size();
            }
            auto work = queries.work() - start;
            if (order.acquisitionProjections != streams || candidates != streams ||
                (indexedWork && work != *indexedWork)) return false;
            indexedWork = work;
            auto wildcard = Relation::getUniverse(PresburgerSpace::getRelationSpace(1, 1));
            auto denseStart = queries.work();
            auto targets = order.acquisitionTargets(wildcard, queries);
            if (!targets || targets->phases.size() != phases) return false;
            if (queries.work() - denseStart < phases) return false;
            RelationQueries tooSmall(phases);
            if (order.acquisitionTargets(wildcard, tooSmall)) return false;
        }
    }
    return BoundaryLowering(facts, queries).checkConditions();
}
