// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_OAHS_ORIGINAL_READ_QUERIES_H
#define PTO_OAHS_ORIGINAL_READ_QUERIES_H
#include "PTO/Transforms/OAHS/StorageFrontiers.h"
#include <algorithm>
#include <map>
#include <set>
#include <tuple>

namespace mlir::pto::oahs::storage_detail {
// Immutable original syntax/effects supply the meaning; these indexes merely
// share derived expressions. No selected command, credit or event key enters.
class OriginalReadQueries {
public:
  explicit OriginalReadQueries(const Program& current)
      : original(hasOriginalIdentityMap(current) ? current.originalStructure : nullptr) {
    predicates.push_back({ParticipationExpression::False, NoAnalysisId, NoAnalysisId, {}});
    predicates.push_back({ParticipationExpression::True, NoAnalysisId, NoAnalysisId, {}});
    frontiers.push_back({GuardedReadFrontier::Empty});
    if (!original) { return; }
    for (unsigned cell = 0; cell < original->cells.size(); ++cell) {
      if (cell >= current.cells.size()) { changedCells.insert(cell); continue; }
      const auto& a = original->cells[cell];
      const auto& b = current.cells[cell];
      const bool changed =
          std::tie(a.ranges, a.unknownRange, a.storage, a.coordinateSpace, a.storageOrigins, a.addressSpace) !=
          std::tie(b.ranges, b.unknownRange, b.storage, b.coordinateSpace, b.storageOrigins, b.addressSpace);
      if (changed) {
        changedCells.insert(cell);
      }
    }
    // Sparse incidence counts avoid comparing every original access with every
    // current phase access. Missing roles are unknown, not invented NoHit facts.
    std::vector<std::map<unsigned, unsigned>> originalRoles(original->operations.size());
    for (std::size_t op = 0; op < original->operations.size(); ++op) {
      for (const auto& access : original->operations[op].accesses) {
        originalRoles[op][access.cell] |= unsigned(access.read) | (unsigned(access.write) << 1);
      }
    }
    std::vector<std::size_t> phases(original->operations.size());
    std::map<std::pair<std::size_t, unsigned>, std::pair<std::size_t, std::size_t>> roleCounts;
    for (std::size_t phase = 0; phase < current.operations.size(); ++phase) {
      const auto operation = current.originalOperations[phase];
      const auto& before = original->operations[operation];
      const auto& after = current.operations[phase];
      ++phases[operation];
      std::map<unsigned, unsigned> roles;
      for (const auto& access : after.accesses) {
        roles[access.cell] |= unsigned(access.read) | (unsigned(access.write) << 1);
      }
      for (const auto& [cell, role] : roles) {
        const auto old = originalRoles[operation].find(cell);
        const bool covered = before.pipe == after.pipe && old != originalRoles[operation].end() &&
                             (role & ~old->second) == 0;
        if (!covered) { changedEffects.emplace(operation, cell); continue; }
        auto& counts = roleCounts[{operation, cell}];
        counts.first += bool(role & 1u);
        counts.second += bool(role & 2u);
      }
    }
    for (std::size_t operation = 0; operation < originalRoles.size(); ++operation) {
      for (const auto& [cell, role] : originalRoles[operation]) {
        const auto counts = roleCounts.find({operation, cell});
        const bool retained = phases[operation] && counts != roleCounts.end() &&
            (!(role & 1u) || counts->second.first == phases[operation]) &&
            (!(role & 2u) || counts->second.second == phases[operation]);
        if (!retained) { changedEffects.emplace(operation, cell); }
      }
    }
    std::vector<const Region*> pending{&original->body};
    while (!pending.empty()) {
      const auto* region = pending.back();
      pending.pop_back();
      ids.emplace(region, regions.size());
      regions.push_back(region);
      if (region->kind == Region::Operation) { operationRegions.emplace(region->operation, region); }
      for (const auto& child : region->children) { parents.emplace(&child, region); }
      if (region->originalOwner != NoControlId) {
        const auto added = owners.emplace(region->originalOwner, region);
        if (!added.second) { added.first->second = nullptr; }
      }
      for (auto child = region->children.rbegin(); child != region->children.rend(); ++child) {
        pending.push_back(&*child);
      }
    }
    owners.emplace(NoControlId, &original->body);
    for (const auto& relation : original->physicalUses) {
      for (const auto& effect : relation.effects) {
        std::map<unsigned, std::size_t> readVisits;
        for (const auto& residue : effect.residues) {
          std::set<unsigned> reads;
          for (const auto& access : residue) { if (access.read) { reads.insert(access.cell); } }
          for (auto cell : reads) { ++readVisits[cell]; }
        }
        if (effect.operation >= original->operations.size()) { continue; }
        for (const auto& access : original->operations[effect.operation].accesses) {
          if (!access.read) { continue; }
          const auto count = readVisits.find(access.cell);
          const bool invariant = !effect.residues.empty() && count != readVisits.end() &&
              count->second == effect.residues.size();
          if (!invariant) { varyingReads.emplace(effect.operation, access.cell); }
        }
      }
    }
  }
  const OriginalReaderFrontiers& query(const ReaderIntervalQuery& request) {
    const auto key = std::make_tuple(request.owner, request.cell, unsigned(request.reader),
                                     unsigned(request.scope), request.begin, request.end);
    const auto prior = queries.find(key);
    if (prior != queries.end()) { return prior->second; }
    auto& result = queries[key];
    result.interval = request;
    const auto owner = owners.find(request.owner);
    const bool valid = original && request.cell < original->cells.size() &&
        unsigned(request.reader) < PipeCount && owner != owners.end() && owner->second &&
        !changedCells.count(request.cell);
    if (!valid) { result.reason = "missing validated original read interval"; return result; }
    const auto* region = owner->second;
    if (request.scope == ReaderIntervalQuery::WholeRegion) {
      if (request.begin != 0 || request.end != NoAnalysisId) {
        result.reason = "whole-region query carries an interval slice";
        return result;
      }
      assign(result, summarize(*region, request.cell, request.reader));
      result.separatesVisits = region->kind == Region::For &&
          result.status == OriginalReaderFrontiers::Status::Exact;
      return result;
    }
    if (request.scope != ReaderIntervalQuery::BodyInterval) {
      result.reason = "invalid original read interval scope";
      return result;
    }
    const bool countedBody = region->kind == Region::For && region->children.size() == 1;
    if (countedBody) { region = &region->children.front(); }
    if (region->kind != Region::Sequence) {
      result.reason = "original read interval does not name a sequence";
      return result;
    }
    const auto end = request.end == NoAnalysisId ? region->children.size() : request.end;
    if (request.begin > end || end > region->children.size()) {
      result.reason = "original read interval is out of bounds";
      return result;
    }
    std::vector<Summary> parts;
    for (std::size_t i = request.begin; i < end; ++i) {
      parts.push_back(summarize(region->children[i], request.cell, request.reader));
    }
    assign(result, sequence(parts));
    return result;
  }
  const OriginalReadSegment& segment(std::size_t ownerId, std::size_t operation, unsigned cell, Pipe reader) {
    const auto key = std::make_tuple(ownerId, operation, cell, unsigned(reader));
    const auto prior = segments.find(key);
    if (prior != segments.end()) { return prior->second; }
    auto& result = segments[key];
    result.reason = "original access has no qualified sequence read interval";
    const auto owner = owners.find(ownerId);
    const auto access = operationRegions.find(operation);
    const bool valid = original && owner != owners.end() && owner->second && access != operationRegions.end();
    if (!valid) { return result; }
    const auto* sequence = owner->second;
    const bool countedSequence = sequence->kind == Region::For && sequence->children.size() == 1;
    if (countedSequence) { sequence = &sequence->children.front(); }
    if (sequence->kind != Region::Sequence) { return result; }
    const auto positionKey = std::make_pair(ownerId, operation);
    auto positionFound = childPositions.find(positionKey);
    if (positionFound == childPositions.end()) {
      const auto* child = access->second;
      auto parent = parents.find(child);
      while (parent != parents.end()) {
        if (parent->second == sequence) { break; }
        ++compositionParts;
        child = parent->second;
        parent = parents.find(child);
      }
      const auto position = parent == parents.end() ? NoAnalysisId :
          std::size_t(child - sequence->children.data());
      positionFound = childPositions.emplace(positionKey, position).first;
    }
    const auto position = positionFound->second;
    if (position == NoAnalysisId) { return result; }
    const auto projection = std::make_pair(ids.at(sequence), cell);
    auto writes = delimiters.find(projection);
    if (writes == delimiters.end()) {
      std::vector<std::size_t> positions;
      for (std::size_t i = 0; i < sequence->children.size(); ++i) {
        if (containsWrite(sequence->children[i], cell)) { positions.push_back(i); }
      }
      writes = delimiters.emplace(projection, std::move(positions)).first;
    }
    const auto next = std::lower_bound(writes->second.begin(), writes->second.end(), position);
    const bool writingChild = next != writes->second.end() && *next == position;
    if (writingChild) {
      result.reason = "reader child contains an overlapping write";
      return result;
    }
    const auto end = next == writes->second.end() ? sequence->children.size() : *next;
    const auto begin = next == writes->second.begin() ? 0 : *std::prev(next) + 1;
    result.interval = {ownerId, cell, reader, ReaderIntervalQuery::BodyInterval, begin, end};
    result.startsAtOwnerEntry = begin == 0;
    result.endsAtOwnerExit = end == sequence->children.size();
    if (!result.startsAtOwnerEntry) { result.before = ids.at(&sequence->children[begin - 1]); }
    if (!result.endsAtOwnerExit) { result.after = ids.at(&sequence->children[end]); }
    result.complete = true;
    result.reason.clear();
    return result;
  }
  std::size_t condition(std::size_t root, std::size_t operation) {
    if (root >= frontiers.size()) { return NoAnalysisId; }
    auto found = conditions.find(root);
    if (found == conditions.end()) {
      std::map<std::size_t, std::size_t> incoming{{root, 1}}, leaves;
      // Interned nodes refer only to lower IDs. Descending propagation visits
      // each relevant node once, including shared frontier subexpressions.
      while (!incoming.empty()) {
        auto current = std::prev(incoming.end());
        ++compositionParts;
        const auto node = frontiers[current->first];
        const auto guard = current->second;
        incoming.erase(current);
        auto send = [&](std::size_t child, std::size_t predicate) {
          incoming[child] = combine(false, incoming[child], predicate);
        };
        if (node.kind == GuardedReadFrontier::Access) {
          leaves[node.operation] = combine(false, leaves[node.operation], guard);
        } else if (node.kind == GuardedReadFrontier::Union) {
          send(node.left, guard); send(node.right, guard);
        } else if (node.kind == GuardedReadFrontier::Guard) {
          send(node.left, combine(true, guard, node.predicate));
        }
      }
      found = conditions.emplace(root, std::move(leaves)).first;
    }
    const auto leaf = found->second.find(operation);
    return leaf == found->second.end() ? 0 : leaf->second;
  }
  ParticipationExpression predicate(std::size_t id) const {
    return id < predicates.size() ? predicates[id] : ParticipationExpression{};
  }
  GuardedReadFrontier frontier(std::size_t id) const {
    return id < frontiers.size() ? frontiers[id] : GuardedReadFrontier{};
  }
  std::vector<OriginalParticipationDemand> participationDemands() {
    std::vector<OriginalParticipationDemand> out;
    if (!original) { return out; }
    std::set<std::tuple<std::size_t, unsigned, unsigned, std::size_t, std::size_t>> visited;
    std::set<std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>> emitted;
    // Enumerate original incidences once, not every copied analytical visit.
    for (const auto& entry : operationRegions) {
      const auto operation = entry.first;
      const auto& op = original->operations[operation];
      for (auto parent = parents.find(entry.second); parent != parents.end(); parent = parents.find(parent->second)) {
        ++compositionParts;
        const auto* owner = parent->second;
        if (owner->kind != Region::For) { continue; }
        for (const auto& access : op.accesses) {
          ++compositionParts;
          if (!access.read || access.write) { continue; }
          const auto& found = segment(owner->originalOwner, operation, access.cell, op.pipe);
          if (!found.complete || found.startsAtOwnerEntry) { continue; }
          auto interval = found.interval;
          const auto key = std::make_tuple(interval.owner, interval.cell, unsigned(interval.reader),
                                             interval.begin, interval.end);
          if (!visited.insert(key).second) { continue; }
          const auto& answer = query(interval);
          if (!answer.complete()) { continue; }
          // The predicate is needed only from the first relevant original use.
          // Unrelated payloads before it do not change that frontier's meaning.
          while (interval.begin < interval.end) {
            auto prefix = interval; prefix.end = prefix.begin + 1;
            ++compositionParts;
            if (query(prefix).status != OriginalReaderFrontiers::Status::NoHit) { break; }
            ++interval.begin;
          }
          for (auto root : {answer.first, answer.last}) {
            for (auto subject : frontierDemands(root)) {
              const auto identity = std::make_tuple(interval.owner, interval.begin, interval.end, subject);
              if (emitted.insert(identity).second) {
                out.push_back({interval, {ObservationAtom::LoopNonEmpty, subject, 0, 1}});
              }
            }
          }
        }
      }
    }
    return out;
  }
  std::size_t evaluations = 0, compositionParts = 0;
  std::size_t predicateCount() const { return predicates.size(); }
  std::size_t frontierCount() const { return frontiers.size(); }
private:
  const std::set<std::size_t>& frontierDemands(std::size_t root) {
    const auto cached = frontierDemandCache.find(root);
    if (cached != frontierDemandCache.end()) { return cached->second; }
    auto& out = frontierDemandCache[root];
    // Cache only requested roots. Materializing each internal node's transitive
    // subject set makes a linear optional-child chain quadratic.
    std::set<std::size_t> visitedFrontiers, visitedPredicates;
    std::vector<std::size_t> pending{root}, conditions;
    while (!pending.empty()) {
      const auto id = pending.back(); pending.pop_back();
      if (!visitedFrontiers.insert(id).second) { continue; }
      ++compositionParts;
      const auto node = frontier(id);
      for (auto child : {node.left, node.right}) {
        if (child != NoAnalysisId) { pending.push_back(child); }
      }
      if (node.kind == GuardedReadFrontier::Guard) { conditions.push_back(node.predicate); }
    }
    while (!conditions.empty()) {
      const auto id = conditions.back(); conditions.pop_back();
      if (!visitedPredicates.insert(id).second) { continue; }
      ++compositionParts;
      const auto node = predicate(id);
      if (node.kind == ParticipationExpression::Atom && node.atom.kind == ObservationAtom::LoopNonEmpty) {
        out.insert(node.atom.owner);
      } else {
        for (auto child : {node.left, node.right}) {
          if (child != NoAnalysisId) { conditions.push_back(child); }
        }
      }
    }
    return out;
  }
  std::map<std::size_t, std::set<std::size_t>> frontierDemandCache;
  struct Summary {
    bool complete = true;
    std::size_t nonempty = 0, first = 0, last = 0;
    std::string reason;
  };
  static Summary unknown(std::string reason) { return {false, 0, 0, 0, std::move(reason)}; }
  static void assign(OriginalReaderFrontiers& result, const Summary& summary) {
    result.nonempty = summary.nonempty;
    result.first = summary.first;
    result.last = summary.last;
    result.reason = summary.reason;
    result.status = !summary.complete ? OriginalReaderFrontiers::Status::Unknown :
        summary.nonempty ? OriginalReaderFrontiers::Status::Exact : OriginalReaderFrontiers::Status::NoHit;
  }
  std::size_t intern(ParticipationExpression expression) {
    const auto& atom = expression.atom;
    const auto key = std::make_tuple(unsigned(expression.kind), expression.left, expression.right,
        unsigned(atom.kind), atom.owner, atom.parameter, atom.value);
    const auto added = predicateIds.emplace(key, predicates.size());
    if (added.second) { predicates.push_back(expression); }
    return added.first->second;
  }
  std::size_t atom(ObservationAtom value) {
    return intern({ParticipationExpression::Atom, NoAnalysisId, NoAnalysisId, value});
  }
  std::size_t negate(std::size_t value) {
    if (value < 2) { return 1 - value; }
    if (predicates[value].kind == ParticipationExpression::Not) { return predicates[value].left; }
    return intern({ParticipationExpression::Not, value, NoAnalysisId, {}});
  }
  std::size_t combine(bool conjunction, std::size_t a, std::size_t b) {
    if (a == b) { return a; }
    if (conjunction) {
      if (!a || !b) { return 0; }
      if (a == 1) { return b; }
      if (b == 1) { return a; }
    } else {
      if (a == 1 || b == 1) { return 1; }
      if (!a) { return b; }
      if (!b) { return a; }
    }
    const bool complements = (predicates[a].kind == ParticipationExpression::Not && predicates[a].left == b) ||
        (predicates[b].kind == ParticipationExpression::Not && predicates[b].left == a);
    if (complements) { return conjunction ? 0 : 1; }
    if (b < a) { std::swap(a, b); }
    return intern({conjunction ? ParticipationExpression::And : ParticipationExpression::Or, a, b, {}});
  }
  std::size_t addFrontier(GuardedReadFrontier value) {
    const auto key = std::make_tuple(unsigned(value.kind), value.left, value.right, value.predicate, value.operation);
    const auto added = frontierIds.emplace(key, frontiers.size());
    if (added.second) { frontiers.push_back(value); }
    return added.first->second;
  }
  std::size_t guarded(std::size_t root, std::size_t condition) {
    if (!root || !condition) { return 0; }
    if (condition == 1) { return root; }
    return addFrontier({GuardedReadFrontier::Guard, root, NoAnalysisId, condition, NoAnalysisId});
  }
  std::size_t unite(std::size_t a, std::size_t b) {
    if (!a || a == b) { return b; }
    if (!b) { return a; }
    return addFrontier({GuardedReadFrontier::Union, a, b, 1, NoAnalysisId});
  }
  Summary sequence(const std::vector<Summary>& parts) {
    compositionParts += parts.size();
    std::vector<std::size_t> suffix(parts.size() + 1);
    for (std::size_t i = parts.size(); i-- > 0;) {
      if (!parts[i].complete) { return parts[i]; }
      suffix[i] = combine(false, parts[i].nonempty, suffix[i + 1]);
    }
    Summary result;
    std::size_t prefix = 0;
    for (std::size_t i = 0; i < parts.size(); ++i) {
      result.first = unite(result.first, guarded(parts[i].first, negate(prefix)));
      result.last = unite(result.last, guarded(parts[i].last, negate(suffix[i + 1])));
      prefix = combine(false, prefix, parts[i].nonempty);
    }
    result.nonempty = prefix;
    return result;
  }
  Summary summarize(const Region& region, unsigned cell, Pipe reader) {
    const auto key = std::make_tuple(ids.at(&region), cell, unsigned(reader));
    const auto found = summaries.find(key);
    if (found != summaries.end()) { return found->second; }
    ++evaluations;
    const auto result = derive(region, cell, reader);
    summaries.emplace(key, result);
    return result;
  }
  bool containsWrite(const Region& region, unsigned cell) {
    const auto key = std::make_pair(ids.at(&region), cell);
    const auto prior = writeSummaries.find(key);
    if (prior != writeSummaries.end()) { return prior->second; }
    ++compositionParts;
    bool writes = false;
    if (region.kind == Region::Operation && region.operation < original->operations.size()) {
      for (const auto& access : original->operations[region.operation].accesses) {
        writes |= access.cell == cell && access.write;
      }
    }
    for (const auto& child : region.children) { writes |= containsWrite(child, cell); }
    writeSummaries.emplace(key, writes);
    return writes;
  }
  Summary derive(const Region& region, unsigned cell, Pipe reader) {
    if (region.kind == Region::Operation) {
      if (region.operation >= original->operations.size()) { return unknown("invalid original access identity"); }
      if (changedEffects.count({region.operation, cell})) {
        return unknown("refined access is not covered by its original physical use");
      }
      const auto& operation = original->operations[region.operation];
      bool reads = false;
      for (const auto& access : operation.accesses) {
        if (access.cell != cell) { continue; }
        if (access.write) { return unknown("overlapping write interrupts the original read interval"); }
        reads |= access.read && operation.pipe == reader;
      }
      if (!reads) { return {}; }
      const auto& physical = original->cells[cell];
      const bool uncertain = physical.unknownRange || physical.storage == Cell::Storage::OverlapWitness ||
          varyingReads.count({region.operation, cell});
      if (uncertain) { return unknown("may footprint does not establish invariant read participation"); }
      const auto leaf = addFrontier({GuardedReadFrontier::Access, NoAnalysisId, NoAnalysisId, 1, region.operation});
      return {true, 1, leaf, leaf, {}};
    }
    std::vector<Summary> children;
    for (const auto& child : region.children) { children.push_back(summarize(child, cell, reader)); }
    if (region.kind == Region::Sequence) { return sequence(children); }
    for (const auto& child : children) { if (!child.complete) { return child; } }
    const bool choice = region.kind == Region::Choice && children.size() == 2;
    if (choice) {
      if (!children[0].nonempty && !children[1].nonempty) { return {}; }
      const auto identity = owners.find(region.originalOwner);
      const bool unique = identity != owners.end() && identity->second == &region;
      if (!unique) {
        return unknown("missing or ambiguous original guard owner");
      }
      if (region.originalOwner == NoControlId) { return unknown("choice lacks original predicate identity"); }
      const auto yes = atom({ObservationAtom::OriginalBoolean, region.originalOwner, 0, 1});
      const auto no = negate(yes);
      return {true, combine(false, combine(true, yes, children[0].nonempty),
                            combine(true, no, children[1].nonempty)),
        unite(guarded(children[0].first, yes), guarded(children[1].first, no)),
        unite(guarded(children[0].last, yes), guarded(children[1].last, no)), {}};
    }
    const bool counted = region.kind == Region::For && children.size() == 1;
    if (counted) {
      const auto& body = children.front();
      if (!body.nonempty) { return {}; }
      const auto identity = owners.find(region.originalOwner);
      const bool unique = identity != owners.end() && identity->second == &region;
      if (!unique) {
        return unknown("missing or ambiguous original guard owner");
      }
      if (!region.qualifiedCounted || region.originalOwner == NoControlId) {
        return unknown("counted read interval lacks an original iteration-domain proof");
      }
      if (body.nonempty != 1) { return unknown("reader participation is not proved invariant across iterations"); }
      const auto nonempty = region.zeroTripPossible ?
          atom({ObservationAtom::LoopNonEmpty, region.originalOwner, 0, 1}) : 1;
      return {true, nonempty,
        guarded(body.first, atom({ObservationAtom::LoopHasPrevious, region.originalOwner, 1, 0})),
        guarded(body.last, atom({ObservationAtom::LoopHasNext, region.originalOwner, 1, 0})), {}};
    }
    if (std::all_of(children.begin(), children.end(), [](const auto& child) { return child.nonempty == 0; })) {
      return {};
    }
    return unknown("read participation in an uncounted region is unresolved");
  }
  std::shared_ptr<const OriginalStructure> original;
  std::vector<const Region*> regions;
  std::set<std::pair<std::size_t, unsigned>> varyingReads, changedEffects;
  std::set<unsigned> changedCells;
  std::map<const Region*, std::size_t> ids;
  std::map<std::size_t, const Region*> owners;
  std::vector<ParticipationExpression> predicates;
  std::vector<GuardedReadFrontier> frontiers;
  std::map<const Region*, const Region*> parents;
  std::map<std::pair<std::size_t, std::size_t>, std::size_t> childPositions;
  std::map<std::size_t, const Region*> operationRegions;
  std::map<std::pair<std::size_t, unsigned>, bool> writeSummaries;
  std::map<std::pair<std::size_t, unsigned>, std::vector<std::size_t>> delimiters;
  std::map<std::tuple<std::size_t, std::size_t, unsigned, unsigned>, OriginalReadSegment> segments;
  std::map<std::size_t, std::map<std::size_t, std::size_t>> conditions;
  using PredicateKey = std::tuple<unsigned, std::size_t, std::size_t, unsigned, std::size_t, uint64_t, uint64_t>;
  std::map<PredicateKey, std::size_t> predicateIds;
  std::map<std::tuple<unsigned, std::size_t, std::size_t, std::size_t, std::size_t>, std::size_t> frontierIds;
  std::map<std::tuple<std::size_t, unsigned, unsigned>, Summary> summaries;
  using QueryKey = std::tuple<std::size_t, unsigned, unsigned, unsigned, std::size_t, std::size_t>;
  std::map<QueryKey, OriginalReaderFrontiers> queries;
};
} // namespace mlir::pto::oahs::storage_detail
#endif
