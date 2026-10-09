// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Exact arithmetic child interfaces for sequence composition.
#include "PTO/Transforms/FrontierSynch/AnalysisCost.h"
#include "PTO/Transforms/FrontierSynch/HardwareProtection.h"
#include "CountedLoop.h"
#include "RegionalRelationsInternal.h"
#include "SequenceAnalysisInternal.h"
#include "PTO/Transforms/FrontierSynch/ArithmeticRegional.h"
#include "PTO/Transforms/FrontierSynch/ArithmeticRegionalComposition.h"
#include "PTO/Transforms/FrontierSynch/ArithmeticStorageSelectors.h"
#include "PTO/Transforms/FrontierSynch/ArithmeticInsertion.h"
#include "PTO/Transforms/FrontierSynch/ArithmeticHandoffAllocation.h"
#include "PTO/Transforms/FrontierSynch/RegionalAllocation.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <map>
namespace mlir::pto::frontiersynch {
namespace {
using Id = RegionExpressions::Id;
struct State : ArithmeticRegionalRelations {
    std::shared_ptr<const SyncInput> inputOwner;
    std::optional<ArithmeticHandoffAllocation> handoffAllocation;
    std::vector<ArithmeticIntegerPiece> primitives;
    using StorageKey = std::pair<AddressSpace, Value>;
    using Interval = std::pair<uint64_t, uint64_t>;
    llvm::MapVector<StorageKey, std::vector<Interval>> finiteStorage;
    bool finiteStorageComplete = false;
    std::shared_ptr<RegionExpressions> arena;
    std::vector<uint32_t> pipes;
    struct LoopGeometry { Id lower; int64_t step; uint64_t maximumOrdinal; };
    std::map<Operation*, LoopGeometry> geometry;
    std::string error, emissionError;
    Id no() { return arena->boolean(false); }
    Id yes() { return arena->boolean(true); }
    Id c(uint64_t value) { return arena->constant(value); }
    bool initialize()
    {
        for (auto* ancestor = program.context.root->getParentOp(); ancestor; ancestor = ancestor->getParentOp()) {
            if (auto loop = dyn_cast<scf::ForOp>(ancestor)) { enclosing.push_back(loop); }
        }
        std::reverse(enclosing.begin(), enclosing.end());
        for (const auto& site : program.sites) {
            for (auto loop : site.loops) {
                if (geometry.count(loop)) { continue; }
                auto domain = CountedLoop::get(loop);
                auto lower = loop.getLowerBound();
                // An entry-bound origin is reusable in every endpoint query.
                // Origins depending on another queried coordinate need a map.
                auto* definition = lower.getDefiningOp();
                auto argument = dyn_cast<BlockArgument>(lower);
                auto* owner = argument ? argument.getOwner()->getParentOp() : nullptr;
                APInt constant;
                const bool fixed = matchPattern(lower, m_ConstantInt(&constant)) && constant.isSignedIntN(64);
                const bool entry = fixed || (definition ?
                    (!program.context.root->isProperAncestor(definition) ||
                     definition->getBlock() == &program.context.function.front()) :
                    (owner == program.context.function.getOperation() ||
                     (owner && owner != program.context.root && !program.context.root->isProperAncestor(owner))));
                if (!domain || !entry) {
                    error = "arithmetic regional ordinals require a representable counted domain and entry origin";
                    return false;
                }
                geometry.emplace(loop, LoopGeometry{fixed ? c(constant.getSExtValue()) : arena->input(lower),
                                                     domain->step, domain->maximumOrdinal});
            }
        }
        return true;
    }
    std::optional<std::vector<Id>> ordinals(RegionalEvent event)
    {
        if (event.type >= program.sites.size() || (event.kind != PeriodicEventKind::Start &&
            event.kind != PeriodicEventKind::Completion)) { return std::nullopt; }
        const auto& loops = program.sites[event.type].loops;
        if (event.visits.size() != (loops.empty() ? 0 : loops.size()-1)) { return std::nullopt; }
        std::vector<Id> result(event.visits);
        if (!loops.empty()) { result.push_back(event.ordinal); }
        else if (arena->constantValue(event.ordinal) != 0) { return std::nullopt; }
        for (auto value : result) {
            if (value >= arena->size() || arena->isBoolean(value)) { return std::nullopt; }
        }
        return result;
    }
    Id coordinateDomain(RegionalEvent event)
    {
        auto values = ordinals(event);
        if (!values) { return RegionExpressions::invalid; }
        Id valid = yes();
        const auto& loops = program.sites[event.type].loops;
        for (unsigned i = 0; i < loops.size(); ++i) {
            // Integer witnesses truncate off-domain. Exclude such ordinals
            // before interpreting the reconstructed bits as an occurrence.
            const auto& domain = geometry.at(loops[i]);
            auto representable = IntegerSystem::create(2, {IntegerConstraint{
                {BoundInteger(domain.step), BoundInteger(1)}, BoundInteger(INT64_MAX)}});
            if (failed(representable)) { return RegionExpressions::invalid; }
            valid = arena->land(valid, arena->land(arena->le((*values)[i], c(domain.maximumOrdinal)),
                arena->integerPredicate(*representable, {(*values)[i], domain.lower}, 1, {0, 0})));
        }
        return valid;
    }
    std::optional<std::vector<Id>> coordinates(RegionalEvent event)
    {
        auto result = ordinals(event);
        if (!result) { return std::nullopt; }
        const auto& loops = program.sites[event.type].loops;
        for (unsigned i = 0; i < loops.size(); ++i) {
            const auto& domain = geometry.at(loops[i]);
            IntegerAffine affine{{BoundInteger(domain.step), BoundInteger(1)}, BoundInteger(0)};
            (*result)[i] = arena->integerWitness(affine, BoundInteger(1), {(*result)[i], domain.lower}, 1, {0, 0}, 0);
        }
        return result;
    }
    Id primitive(PrimitiveKind kind, RegionalEvent a, std::optional<RegionalEvent> b = std::nullopt)
    {
        auto left = coordinates(a);
        if (!left) { return RegionExpressions::invalid; }
        auto values = *left;
        if (b) {
            auto right = coordinates(*b);
            if (!right) { return RegionExpressions::invalid; }
            llvm::append_range(values, *right);
        }
        llvm::append_range(values, parameters);
        Id valid = coordinateDomain(a);
        if (b) { valid = arena->land(valid, coordinateDomain(*b)); }
        Id result = no();
        if (kind != PrimitiveKind::Occurrences || b) { return RegionExpressions::invalid; }
        for (const auto& piece : occurrences) {
            if (piece.site != a.type) { continue; }
            auto residues = piece.residues;
            llvm::append_range(residues, piece.parameterResidues);
            result = arena->lor(result, arena->integerPredicate(piece.system, values,
                program.primitives.period, residues));
        }
        return arena->land(valid, result);
    }
    Id referenceOrder(RegionalEvent a, RegionalEvent b, bool includeSame = false)
    {
        auto left = ordinals(a), right = ordinals(b);
        if (!left || !right) { return RegionExpressions::invalid; }
        const auto& first = program.sites[a.type];
        const auto& second = program.sites[b.type];
        Id before = no(), equalPrefix = yes();
        unsigned common = 0;
        // The shared loop prefix fixes lexicographic execution order. Once
        // its coordinates agree, original site order fixes sibling regions.
        // Positive steps make ordinal comparison equivalent to IV comparison.
        while (common < std::min(left->size(), right->size()) &&
               first.loops[common] == second.loops[common]) {
            before = arena->lor(before,
                arena->land(equalPrefix, arena->lt((*left)[common], (*right)[common])));
            equalPrefix = arena->land(equalPrefix, arena->eq((*left)[common], (*right)[common]));
            ++common;
        }
        if (a.type < b.type || (includeSame && a.type == b.type)) {
            before = arena->lor(before, equalPrefix);
        }
        // Presence includes exact loop/branch domains and the representable
        // ordinal check. Mutually exclusive arms cannot acquire an order just
        // because the static preorder numbers one arm before the other.
        auto present = arena->land(primitive(PrimitiveKind::Occurrences, a),
                                   primitive(PrimitiveKind::Occurrences, b));
        return arena->land(present, before);
    }
    Id reach(RegionalEvent a, RegionalEvent b)
    {
        if (a.type >= pipes.size() || b.type >= pipes.size()) { return RegionExpressions::invalid; }
        if (pipes[a.type] == pipes[b.type] &&
            (a.kind == b.kind || (a.kind == PeriodicEventKind::Start && b.kind == PeriodicEventKind::Completion))) {
            // Native order already contains every forward I->I, C->C and
            // I->C pair on one pipe (including reflexivity where appropriate).
            // All added requirements are reference-forward, so no such pair
            // can be added in the reverse direction. Only C->I needs the
            // required-order relation on this pipe.
            return referenceOrder(a, b, true);
        }
        auto left = coordinates(a), right = coordinates(b);
        if (!left || !right) { return RegionExpressions::invalid; }
        auto values = *left;
        llvm::append_range(values, *right); llvm::append_range(values, parameters);
        Id result = no();
        auto kind = [](PeriodicEventKind event) {
            return event == PeriodicEventKind::Start ? ArithmeticEvent::Start : ArithmeticEvent::Completion;
        };
        for (const auto& [key, pieces] : analysis.requiredOrder) {
            if (key.source.site != a.type || key.target.site != b.type ||
                key.source.event != kind(a.kind) || key.target.event != kind(b.kind)) { continue; }
            auto residues = key.source.residues;
            llvm::append_range(residues, key.target.residues); llvm::append_range(residues, key.parameterResidues);
            for (const auto& system : pieces) {
                result = arena->lor(result, arena->integerPredicate(system, values, analysis.period, residues));
            }
        }
        return arena->land(arena->land(coordinateDomain(a), coordinateDomain(b)), result);
    }
    std::vector<RegionalSelector> selected(const GeneralArithmeticEndpointSelector& selector,
                                           std::optional<Id> byte = std::nullopt)
    {
        std::vector<Id> values;
        if (byte) { values.push_back(*byte); }
        llvm::append_range(values, parameters);
        std::map<std::size_t, RegionalSelector> selected;
        // The exact extremum is unique. Overlapping pieces therefore select
        // the same tagged tuple; priority masks would only duplicate guards.
        for (const auto& piece : selector.pieces) {
            auto residues = piece.inputResidues;
            llvm::append_range(residues, piece.parameterResidues);
            auto guard = arena->integerPredicate(piece.domain, values, selectors.period, residues);
            if (arena->constantValue(guard) == 0) { continue; }
            std::vector<Id> tuple;
            if (piece.outputSite >= program.sites.size() ||
                piece.outputs.size() != program.sites[piece.outputSite].loops.size()) {
                error = "arithmetic boundary selector output has inconsistent coordinates"; return {};
            }
            const auto& loops = program.sites[piece.outputSite].loops;
            for (unsigned i = 0; i < piece.outputs.size(); ++i) {
                const auto& output = piece.outputs[i];
                auto value = arena->integerWitness(output.numerator, output.denominator, values,
                                                   selectors.period, residues, output.residue);
                const auto& domain = geometry.at(loops[i]);
                // Signed mathematical subtraction/division; unsigned modular
                // subtraction is not the inverse at negative off-domain values.
                tuple.push_back(arena->integerWitness({{BoundInteger(1), BoundInteger(-1)}, BoundInteger(0)},
                    BoundInteger(domain.step), {value, domain.lower}, 1, {0, 0}, 0));
            }
            RegionalEvent event{static_cast<uint32_t>(piece.outputSite), c(0), PeriodicEventKind::Start};
            if (!tuple.empty()) { event.ordinal = tuple.back(); tuple.pop_back(); event.visits = std::move(tuple); }
            auto [found, inserted] = selected.emplace(piece.outputSite, RegionalSelector{event, guard});
            if (!inserted) {
                auto& previous = found->second;
                previous.event.ordinal = arena->select(guard, event.ordinal, previous.event.ordinal);
                for (unsigned i = 0; i < event.visits.size(); ++i) {
                    previous.event.visits[i] = arena->select(guard, event.visits[i], previous.event.visits[i]);
                }
                previous.present = arena->lor(previous.present, guard);
            }
        }
        std::vector<RegionalSelector> result;
        for (const auto& [site, choice] : selected) { result.push_back(choice); }
        return result;
    }
    std::vector<RegionalSelector> selected(const ArithmeticBoundarySelector& boundary,
                                           std::optional<Id> byte = std::nullopt)
    {
        return selected(boundary.selector, byte);
    }
    std::shared_ptr<RegionalAllocationSummary> allocation(const PreparedLogicalPlan& plan)
    {
        RegionExpressions::Transaction transaction(*arena);
        const auto previousError = error;
        bool accepted = false;
        auto restore = llvm::make_scope_exit([&] { if (!accepted) { error = previousError; } });
        if (!handoffAllocation) { handoffAllocation = buildArithmeticHandoffAllocation(analysis, pipes, 1); }
        if (!handoffAllocation->error.empty()) { return {}; }
        std::map<std::pair<std::size_t, std::size_t>, uint32_t> records;
        for (const auto& family : plan.families) {
            for (const auto& member : family.members) {
                records[{member.source, member.target}] = member.record;
            }
        }
        auto result = std::make_shared<RegionalAllocationSummary>();
        for (const auto& family : handoffAllocation->families) {
            auto record = records.find({family.sourceSite, family.targetSite});
            auto first = selected(family.firstSource), last = selected(family.lastTarget);
            if (record == records.end() || first.size() != 1 || last.size() != 1) { return {}; }
            RegionalAllocationGroup group;
            group.sourcePipe = pipes[family.sourceSite]; group.targetPipe = pipes[family.targetSite]; group.budget = 1;
            RegionalAllocationMember member;
            member.record = record->second; member.firstSource = first.front().event;
            member.lastTarget = last.front().event; member.lastTarget.kind = PeriodicEventKind::Completion;
            member.active = arena->land(first.front().present, last.front().present);
            member.sourcePipe = group.sourcePipe; member.targetPipe = group.targetPipe;
            member.tupleRule = PhysicalTupleRule{1 + program.sites[family.sourceSite].loops.size(), 0, {}};
            last.front().event.kind = PeriodicEventKind::Completion;
            group.members.push_back(std::move(member)); group.lanes.push_back({std::move(first), std::move(last)});
            result->groups.push_back(std::move(group));
        }
        const bool failedExport = !error.empty() || !arena->constructionError().empty();
        if (failedExport) { return {}; }
        transaction.commit(); accepted = true;
        return result;
    }
    RegionalStorageSelectors storage(AddressSpace space, Value base, Id byte)
    {
        RegionalStorageSelectors result;
        for (const auto& boundary : selectors.boundaries) {
            if (boundary.storageSpace != space || boundary.storageBase != base) { continue; }
            auto choices = selected(boundary, byte);
            if (choices.empty()) { continue; }
            switch (boundary.kind) {
            case ArithmeticBoundaryKind::FirstWriter: llvm::append_range(result.firstWriters, choices); break;
            case ArithmeticBoundaryKind::LastWriter: llvm::append_range(result.lastWriters, choices); break;
            case ArithmeticBoundaryKind::FirstReaderBeforeWrite:
                llvm::append_range(result.firstReaders[*boundary.pipe], choices); break;
            case ArithmeticBoundaryKind::LastReaderAfterWrite:
                llvm::append_range(result.lastReaders[*boundary.pipe], choices); break;
            default: break;
            }
        }
        return result;
    }
};
bool sameSelectors(const std::vector<RegionalSelector>& a, const std::vector<RegionalSelector>& b)
{
    if (a.size() != b.size()) { return false; }
    for (unsigned i = 0; i < a.size(); ++i) {
        if (a[i].present != b[i].present || a[i].event.type != b[i].event.type ||
            a[i].event.ordinal != b[i].event.ordinal || a[i].event.visits != b[i].event.visits) { return false; }
    }
    return true;
}
bool sameReaders(const std::map<uint32_t, std::vector<RegionalSelector>>& a,
                 const std::map<uint32_t, std::vector<RegionalSelector>>& b)
{
    if (a.size() != b.size()) { return false; }
    for (const auto& [pipe, choices] : a) {
        auto found = b.find(pipe);
        if (found == b.end() || !sameSelectors(choices, found->second)) { return false; }
    }
    return true;
}
// The finite-port adapter needs a bounded set of physical bytes across every
// entry binding. Establish this from access primitives before computing F* or
// first/last selectors; an unsupported child can then leave its parent to try a
// whole-region route without paying for interfaces it cannot export.
bool collectFiniteStorage(State& state)
{
    llvm::DenseSet<State::StorageKey> symbolic;
    for (const auto& access : state.primitives) {
        const auto& schema = *access.schema;
        if (schema.kind != PrimitiveKind::Reads && schema.kind != PrimitiveKind::Writes) { continue; }
        const auto byteColumn = schema.sourceDimensions;
        if (!schema.storageSpace || byteColumn >= access.system.dimensions() || byteColumn >= access.residues.size()) {
            state.error = "arithmetic storage primitive lacks its physical byte coordinate"; return false;
        }
        const State::StorageKey key{*schema.storageSpace, schema.storageBase};
        if (symbolic.count(key)) { continue; }
        // Imported primitives already include the exact occurrence and shared
        // parameter context. Projection here is directly to the byte quotient.
        auto projected = access.system.project({byteColumn});
        if (failed(projected)) { state.error = "arithmetic storage support projection failed"; return false; }
        for (const auto& piece : *projected) {
            if (piece.isEmpty()) { continue; }
            std::optional<BoundInteger> lower, upper;
            for (const auto& row : piece.constraints()) {
                const auto& coefficient = row.coefficients.front();
                if (coefficient > 0) {
                    auto bound = floorDiv(row.bound, coefficient);
                    upper = upper ? std::min(*upper, bound) : bound;
                } else if (coefficient < 0) {
                    auto bound = -floorDiv(row.bound, -coefficient);
                    lower = lower ? std::max(*lower, bound) : bound;
                }
            }
            if (!lower || !upper) {
                symbolic.insert(key); state.finiteStorage.erase(key); break;
            }
            const BoundInteger period(state.program.primitives.period), residue(access.residues[byteColumn]);
            const auto begin = *lower * period + residue;
            const auto end = *upper * period + residue + 1;
            if (begin < 0 || end > BoundInteger(INT64_MAX)) {
                symbolic.insert(key); state.finiteStorage.erase(key); break;
            }
            if (begin >= end) { continue; }
            state.finiteStorage[{*schema.storageSpace, schema.storageBase}].push_back({
                static_cast<uint64_t>(static_cast<int64_t>(begin)), static_cast<uint64_t>(static_cast<int64_t>(end))});
        }
    }
    for (auto& entry : state.finiteStorage) {
        auto& intervals = entry.second;
        std::sort(intervals.begin(), intervals.end());
        std::vector<State::Interval> merged;
        for (const auto& interval : intervals) {
            if (!merged.empty() && interval.first <= merged.back().second) {
                merged.back().second = std::max(merged.back().second, interval.second);
            } else { merged.push_back(interval); }
        }
        intervals = std::move(merged);
    }
    state.finiteStorageComplete = symbolic.empty();
    return true;
}
// A byte predicate with coefficient +1 changes at bound+1 minus its
// remaining affine form; coefficient -1 changes at that form minus bound.
// When the remaining coefficients have gcd g, every change belongs to a
// fixed residue modulo g. Splitting there preserves every access predicate
// for every occurrence/parameter binding, hence also every lifetime selector.
// Zero gcd means all changes are fixed absolute byte positions.
struct ByteAtoms {
    bool certified = false;
    BoundInteger period{0};
    std::vector<BoundInteger> cuts;
    uint64_t next(uint64_t begin, uint64_t limit) const
    {
        if (!certified) { return begin + 1; }
        BoundInteger result(static_cast<int64_t>(limit)), current(static_cast<int64_t>(begin));
        for (const auto& cut : cuts) {
            auto candidate = period == 0 ? cut :
                (floorDiv(current - cut, period) + BoundInteger(1)) * period + cut;
            if (candidate > current && candidate < result) { result = std::move(candidate); }
        }
        // The selected endpoint lies in [begin+1, limit], already checked
        // representable by finite storage support. All cut arithmetic is wide.
        return static_cast<uint64_t>(static_cast<int64_t>(result));
    }
};
ByteAtoms byteAtoms(const State& state, const State::StorageKey& key)
{
    ByteAtoms result;
    // Original residue classes can change membership between adjacent bytes.
    // Keep the byte path unless their equivalence is separately established.
    if (state.program.primitives.period != 1) { return result; }
    for (const auto& access : state.primitives) {
        const auto& schema = *access.schema;
        if ((schema.kind != PrimitiveKind::Reads && schema.kind != PrimitiveKind::Writes) ||
            schema.storageSpace != key.first || schema.storageBase != key.second) { continue; }
        const auto byte = schema.sourceDimensions;
        if (byte >= access.system.dimensions()) { return {}; }
        for (const auto& row : access.system.congruences()) {
            if (row.coefficients[byte] != 0) { return {}; }
        }
        for (const auto& row : access.system.constraints()) {
            const auto& coefficient = row.coefficients[byte];
            if (coefficient == 0) { continue; }
            if (coefficient != 1 && coefficient != -1) { return {}; }
            result.cuts.push_back(coefficient == 1 ? row.bound + BoundInteger(1) : -row.bound);
            for (unsigned i = 0; i < row.coefficients.size(); ++i) {
                if (i == byte) { continue; }
                auto value = row.coefficients[i] < 0 ? -row.coefficients[i] : row.coefficients[i];
                while (value != 0) {
                    auto remainder = result.period % value;
                    result.period = std::move(value);
                    value = std::move(remainder);
                }
            }
        }
    }
    if (result.period != 0) {
        for (auto& cut : result.cuts) {
            cut = cut - floorDiv(cut, result.period) * result.period;
        }
    }
    std::sort(result.cuts.begin(), result.cuts.end());
    result.cuts.erase(std::unique(result.cuts.begin(), result.cuts.end()), result.cuts.end());
    result.certified = true;
    return result;
}
bool finiteBoundaries(State& state, RegionalAnalysis& out)
{
    // Reuse the eligibility projection; do not project selector support again.
    for (const auto& [key, intervals] : state.finiteStorage) {
        const auto atoms = byteAtoms(state, key);
        // Instantiate selectors once per certified access-equivalent atom.
        // Unsupported byte predicates retain the exact per-byte path. The
        // byte ledger counts represented storage; cells counts exported atoms.
        // This finite adapter charges its output size, independently of the
        // compact slot-table limit. It does not enumerate loop occurrences.
        for (const auto& interval : intervals) {
            const auto bytes = interval.second - interval.first;
            if (bytes > UINT64_MAX - out.cost.boundaryBytes) {
                state.error = "arithmetic boundary byte count exceeds representation";
                return false;
            }
            out.cost.boundaryBytes += bytes;
            uint64_t byte = interval.first;
            while (byte < interval.second) {
                const auto begin = byte;
                byte = atoms.next(begin, interval.second);
                auto boundary = state.storage(key.first, key.second, state.c(begin));
                if (boundary.firstWriters.empty() && boundary.lastWriters.empty() &&
                    boundary.firstReaders.empty() && boundary.lastReaders.empty()) { continue; }
                if (!out.storageBoundary.empty()) {
                    auto& previous = out.storageBoundary.back();
                    if (previous.cell.space == key.first && previous.cell.base == key.second &&
                        previous.cell.end == begin &&
                        sameSelectors(previous.firstWriters, boundary.firstWriters) &&
                        sameSelectors(previous.lastWriters, boundary.lastWriters) &&
                        sameReaders(previous.firstReaders, boundary.firstReaders) &&
                        sameReaders(previous.lastReaders, boundary.lastReaders)) {
                        previous.cell.end = byte; continue;
                    }
                }
                out.storageBoundary.push_back({{key.first, begin, byte, key.second}, std::move(boundary.firstWriters),
                    std::move(boundary.lastWriters), std::move(boundary.firstReaders),
                    std::move(boundary.lastReaders)});
            }
        }
    }
    out.cost.cells = out.storageBoundary.size();
    return true;
}
FailureOr<RegionalAnalysis> exportState(const std::shared_ptr<State>& state,
    const SyncInput& input, std::string& error, bool finite, bool selectors = true)
{
    RegionalAnalysis out; out.expressions = state->arena;
    out.accessModel = &input.accesses(); out.gmAliasPolicy = input.memory().gmPolicy();
    for (const auto& site : state->program.sites) {
        auto* operation = site.phase->elementOp;
        auto* next = operation->getNextNode();
        if (!next) { error = "arithmetic payload has no legal after cut"; return failure(); }
        out.anchors.push_back({site.phase, {}, {operation->getBlock(), operation}, {operation->getBlock(), next}});
        out.occurrenceLoops.push_back(site.loops.empty() ? scf::ForOp{} : site.loops.back());
        out.outerLoops.emplace_back(site.loops.begin(), site.loops.empty() ? site.loops.end() : site.loops.end()-1);
        state->arena->forbidRecomputation(site.phase->elementOp);
    }
    out.presence = [state](RegionalEvent a) -> std::optional<Id> {
        auto value = state->primitive(PrimitiveKind::Occurrences, a);
        return value == RegionExpressions::invalid ? std::nullopt : std::optional<Id>(value);
    };
    out.referenceBefore = [state](RegionalEvent a, RegionalEvent b) -> std::optional<Id> {
        auto value = state->referenceOrder(a, b);
        return value == RegionExpressions::invalid ? std::nullopt : std::optional<Id>(value);
    };
    out.reachability = [state](RegionalEvent a, RegionalEvent b) -> std::optional<Id> {
        auto value = state->reach(a,b);
        return value == RegionExpressions::invalid ? std::nullopt : std::optional<Id>(value);
    };
    // Query snapshots deliberately contain no storage projection, extrema or
    // endpoint recipe. A stronger request constructs its own callback state.
    if (!selectors) {
        out.arithmeticRelations = state;
        out.capabilities.exactQueries = true;
        out.cost.children = 1; out.cost.arithmeticRegions = 1;
        out.cost.expressionNodes = state->arena->size();
        if (!state->arena->constructionError().empty()) {
            error = state->arena->constructionError(); return failure();
        }
        return out;
    }
    out.storageSelectors = [state](RegionalByteAddress byte) -> std::optional<RegionalStorageSelectors> {
        auto selected = state->storage(byte.space, byte.base, byte.offset);
        if (!state->error.empty() || !state->arena->constructionError().empty()) { return std::nullopt; }
        return selected;
    };
    std::map<std::size_t, std::vector<RegionalSelector>> first, last;
    for (const auto& boundary : state->selectors.boundaries) {
        if (boundary.storageSpace) { continue; }
        auto selected = state->selected(boundary);
        switch (boundary.kind) {
        case ArithmeticBoundaryKind::FirstPayload:
            llvm::append_range(out.firstPayloads[*boundary.pipe], selected);
            break;
        case ArithmeticBoundaryKind::LastPayload: llvm::append_range(out.lastPayloads[*boundary.pipe], selected); break;
        case ArithmeticBoundaryKind::FirstSite: first[*boundary.site] = std::move(selected); break;
        case ArithmeticBoundaryKind::LastSite: last[*boundary.site] = std::move(selected); break;
        default: break;
        }
    }
    if (!state->finiteStorage.empty() && !finiteBoundaries(*state, out)) {
        error = state->error; return failure();
    }
    out.arithmeticRelations = state;
    for (unsigned type = 0; type < state->program.sites.size(); ++type) {
        out.firstSitePayloads[type] = first[type];
        if (first[type].empty() && last[type].empty()) { continue; }
        if (first[type].size() != 1 || last[type].size() != 1) {
            error = "arithmetic site extrema lack a single guarded tuple"; return failure();
        }
        for (auto effect : input.accesses().effectsFor(state->program.sites[type].phase)) {
            if (llvm::is_contained(state->program.extraction.dischargedEffects, effect)) {
                out.deferredAccessBoundary.push_back({effect, first[type].front(), last[type].front(), false});
                continue;
            }
            const auto& modeled = input.accesses().effects()[effect];
            const bool geometric = !modeled.regions.empty() && modeled.memory &&
                llvm::all_of(modeled.regions, [&](const auto& region) {
                    return state->finiteStorage.count({modeled.memory->scope, region.base});
                });
            if (!finite && !geometric) { out.symbolicStorageEffects.push_back(effect); }
            out.accessBoundary.push_back({effect, first[type].front(), last[type].front(), geometric});
        }
    }
    if (!out.symbolicStorageEffects.empty()) {
        auto certificate = std::make_shared<RegionalSymbolicStorageCertificate>();
        certificate->expressions = state->arena; certificate->accessModel = &input.accesses();
        certificate->gmAliasPolicy = out.gmAliasPolicy;
        llvm::MapVector<State::StorageKey, bool> families;
        for (const auto& piece : state->selectors.support) { families[{piece.space, piece.base}] = true; }
        for (const auto& entry : families) {
            const auto key = entry.first;
            RegionalStorageFamily family;
            family.space = key.first; family.base = key.second;
            // Effects retain their original shared identities. Membership is
            // defined by the complete exact support union for this family.
            for (auto id : out.symbolicStorageEffects) {
                const auto& effect = input.accesses().effects()[id];
                const bool belongs = llvm::any_of(effect.regions, [&](const auto& region) {
                    return region.base == key.second;
                }) || llvm::any_of(effect.ranges, [&](const auto& range) {
                    return range.base == key.second && range.space == key.first;
                });
                if (effect.memory && effect.memory->scope == key.first && belongs) { family.effects.push_back(id); }
            }
            family.membership = [state, key](RegionalByteAddress address) -> std::optional<Id> {
                if (address.offset >= state->arena->size() || state->arena->isBoolean(address.offset)) {
                    return std::nullopt;
                }
                if (address.space != key.first) { return state->no(); }
                if (address.base != key.second) {
                    SmallVector<SyncStorageCell> domains{{key.first, 0, 1, key.second},
                                                        {address.space, 0, 1, address.base}};
                    if (storageBasesAreComparable(domains, state->input->memory().gmPolicy())) { return state->no(); }
                    return std::nullopt;
                }
                std::vector<Id> values{address.offset}; llvm::append_range(values, state->parameters);
                auto present = state->no();
                for (const auto& piece : state->selectors.support) {
                    if (piece.space != key.first || piece.base != key.second) { continue; }
                    std::vector<uint64_t> residues{piece.byteResidue};
                    llvm::append_range(residues, piece.parameterResidues);
                    present = state->arena->lor(present, state->arena->integerPredicate(
                        piece.domain, values, state->selectors.period, residues));
                }
                return present;
            };
            certificate->families.push_back(std::move(family));
        }
        out.symbolicStorage = std::move(certificate);
    }
    out.prepareWithVisits = [state](ArrayRef<scf::ForOp> enclosing)
        -> FailureOr<std::unique_ptr<PreparedLogicalPlan>> {
        if (enclosing != ArrayRef<scf::ForOp>(state->enclosing)) {
            state->emissionError = "arithmetic regional endpoint requires its original enclosing visit context";
            return failure();
        }
        auto plan = prepareGeneralArithmeticRegionalInsertion(state->program.context.function, state->program,
                                                       state->analysis, state->emissionError);
        if (succeeded(plan)) {
            (*plan)->completeInvocation = false;
            (*plan)->allocationPreparation = [state](PreparedLogicalPlan& prepared) {
                prepared.regionalAllocation = state->allocation(prepared);
            };
        }
        return plan;
    };
    out.prepare = [prepare = out.prepareWithVisits]() { return prepare({}); };
    out.capabilities = {true, true, true, true, true};
    out.cost.children = 1; out.cost.arithmeticRegions = 1;
    out.cost.expressionNodes = state->arena->size();
    if (!state->error.empty()) { error = state->error; return failure(); }
    if (!state->arena->constructionError().empty()) { error = state->arena->constructionError(); return failure(); }
    return out;
}
} // namespace
namespace {
enum class StorageExportRequest { Demands, Finite, SymbolicAllowed };
FailureOr<RegionalAnalysis> initializeExports(const std::shared_ptr<State>& state,
    const SyncInput& input, std::string& error, bool selectors, StorageExportRequest request,
    const std::function<std::optional<Id>(Value)>& parameterBinding,
    const std::function<bool()>& reduce = {})
{
    if (!state->initialize()) { error = state->error; return failure(); }
    auto imported = importArithmeticIntegerPieces(state->program, !selectors);
    if (failed(imported)) { error = "regional arithmetic primitive import failed"; return failure(); }
    state->primitives = std::move(*imported);
    if (selectors && !collectFiniteStorage(*state)) { error = state->error; return failure(); }
    const bool finite = state->finiteStorageComplete;
    if (request == StorageExportRequest::Finite && !finite) {
        error = "finite regional export deferred: symbolic storage support remains available to other routes";
        return failure();
    }
    for (const auto& piece : state->primitives) {
        if (piece.schema->kind != PrimitiveKind::Occurrences || !piece.schema->sourceSite) { continue; }
        const auto dimensions = piece.schema->sourceDimensions;
        state->occurrences.push_back({*piece.schema->sourceSite,
            {piece.residues.begin(), piece.residues.begin() + dimensions},
            {piece.residues.begin() + dimensions, piece.residues.end()}, piece.system});
    }
    for (Value parameter : state->program.parameters) {
        auto binding = parameterBinding ? parameterBinding(parameter) :
            std::optional<Id>(state->arena->input(parameter));
        if (!binding || *binding >= state->arena->size()) {
            error = "regional arithmetic phase parameter has no exact binding";
            return failure();
        }
        state->parameters.push_back(*binding);
    }
    for (const auto& site : state->program.sites) {
        state->pipes.push_back(static_cast<uint32_t>(site.phase->kPipeValue));
    }

    if (reduce && !reduce()) { return failure(); }
    if (selectors) {
        state->selectors = buildArithmeticStorageSelectors(state->program, state->pipes);
        if (!state->selectors.error.empty()) { error = state->selectors.error; return failure(); }
    }
    return exportState(state, input, error, finite, selectors);

}
FailureOr<RegionalAnalysis> analyzeArithmeticRegionImpl(ArithmeticRegionContext context,
    const PhaseIndex& index, const SyncInput& input, std::shared_ptr<RegionExpressions> expressions,
    std::string& error, std::function<std::optional<RegionExpressions::Id>(Value)> parameterBinding,
    StorageExportRequest request, std::shared_ptr<const ArithmeticRegionalRelations>* retained = nullptr,
    const ArithmeticProgram* certified = nullptr, ArrayRef<ArithmeticLimits> profiles = {})
{
    if (!context.function || !context.root || !expressions || !expressions->constructionError().empty()) {
        error = "arithmetic region requires a valid original root and expression arena";
        return failure();
    }
    auto state = std::make_shared<State>(); state->arena = std::move(expressions);
    state->input = &input;
    // Specialize certified entry constants before constructing relations. A
    // large bank stride times a known phase becomes a physical constant, not
    // an artificial variable coefficient that fails the arithmetic class.
    // Original sites/cuts remain, and their enclosing provider enforces the
    // phase/interval context for both queries and prepared endpoint recipes.
    ArithmeticEntryConstant entryConstant;
    if (parameterBinding) {
        entryConstant = [&](Value value) -> std::optional<int64_t> {
            auto expression = parameterBinding(value);
            if (!expression) { return std::nullopt; }
            auto fixed = state->arena->constantValue(*expression);
            if (!fixed) { return std::nullopt; }
            return APInt(64, *fixed).getSExtValue();
        };
    }
    if (certified) {
        if (certified->finiteExpansion) {
            error = "finite occurrence representation requires its original-coordinate regional adapter";
            return failure();
        }
        if (parameterBinding || certified->context.function != context.function ||
            certified->context.root != context.root || certified->modeledInput != &input ||
            certified->phaseIndex != &index || certified->specializedEntry) {
            error = "arithmetic certificate belongs to a different entry context";
            return failure();
        }
        state->program = *certified;
    } else {
        const SmallVector<ArithmeticLimits> defaults{{8, 8, 1, 4096}, {8, 8, 2, 4096}};
        if (profiles.empty()) { profiles = defaults; }
        for (const auto& limits : profiles) {
            state->program = recognizeArithmeticProgram(context, index, input, input.accesses(), limits, entryConstant);
            if (state->program.extraction.state == RecognitionState::Applicable &&
                state->program.recognition.state == RecognitionState::Applicable) { break; }
        }
    }
    if (state->program.extraction.state != RecognitionState::Applicable ||
        state->program.recognition.state != RecognitionState::Applicable) {
        llvm::raw_string_ostream message(error);
        message << "regional arithmetic primitives do not satisfy the configured class";
        for (const auto& diagnostic : state->program.extraction.diagnostics) {
            message << "; " << recognitionName(diagnostic.issue);
            if (diagnostic.anchor) {
                message << " at " << diagnostic.anchor->getName() << ' ';
                diagnostic.anchor->getLoc().print(message);
            }
        }
        for (const auto& diagnostic : state->program.recognition.diagnostics) {
            message << "; " << recognitionName(diagnostic.issue) << " relation " << diagnostic.relation
                    << " piece " << diagnostic.piece;
        }
        return failure();
    }
    // Retain exact covers before any regional query or storage export. Those
    // adapters can fail independently of the mathematical reduction.
    auto reduce = [&]() {
        const auto protection = structuredProtection(input.accesses());
        state->analysis = analyzeGeneralArithmeticDemandsWithProtection(state->program, protection);
        if (!state->analysis.error.empty()) { error = state->analysis.error; return false; }
        if (!state->analysis.exactMinimum) { error = "regional exact reduction is unavailable"; return false; }
        if (retained) { *retained = state; }
        return true;
    };
    // The retained session route always computes demands first. The finite-only
    // compatibility probe can still defer symbolic exports before construction.
    if (retained && !reduce()) { return failure(); }
    if (request == StorageExportRequest::Demands) { return RegionalAnalysis{}; }
    return initializeExports(state, input, error, true, request, parameterBinding,
        retained ? std::function<bool()>{} : reduce);

}
} // namespace
std::shared_ptr<const ArithmeticRegionalRelations> analyzeArithmeticRegionDemands(
    ArithmeticRegionContext context, const PhaseIndex& index, const SyncInput& input,
    const ArithmeticProgram* certified, std::string& error)
{
    std::shared_ptr<const ArithmeticRegionalRelations> demands;
    (void)analyzeArithmeticRegionImpl(context, index, input, std::make_shared<RegionExpressions>(),
        error, {}, StorageExportRequest::Demands, &demands, certified);
    return demands;
}
FailureOr<RegionalAnalysis> exportArithmeticRegion(
    const ArithmeticRegionalRelations& demands, std::shared_ptr<RegionExpressions> expressions,
    bool selectors, std::string& error, std::shared_ptr<const SyncInput> inputOwner)
{
    const bool valid = demands.input && expressions && expressions->constructionError().empty() &&
        demands.analysis.exactMinimum && demands.analysis.error.empty() &&
        (!inputOwner || inputOwner.get() == demands.input);
    if (!valid) {
        error = "arithmetic export requires exact mathematics in its modeled input context"; return failure();
    }
    // Session requests are synchronous. No previously published callback runs
    // inside this transaction, and no rejected state or appended ID escapes.
    RegionExpressions::Transaction transaction(*expressions);
    auto state = std::make_shared<State>();
    state->input = demands.input; state->inputOwner = std::move(inputOwner);
    state->program = demands.program; state->analysis = demands.analysis;
    state->arena = expressions;
    auto exported = initializeExports(state, *demands.input, error, selectors,
        StorageExportRequest::SymbolicAllowed, {});
    if (failed(exported)) { return failure(); }
    transaction.commit();
    return exported;
}
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareArithmeticRegion(
    const ArithmeticRegionalRelations& demands, std::shared_ptr<RegionExpressions> expressions,
    ArrayRef<scf::ForOp> enclosing, std::string& error, std::shared_ptr<const SyncInput> inputOwner)
{
    const bool valid = demands.input && expressions && expressions->constructionError().empty() &&
        demands.analysis.exactMinimum && (!inputOwner || inputOwner.get() == demands.input);
    if (!valid) { error = "arithmetic preparation requires its retained original context"; return failure(); }
    RegionExpressions::Transaction transaction(*expressions);
    auto state = std::make_shared<State>();
    state->input = demands.input; state->inputOwner = std::move(inputOwner);
    state->program = demands.program; state->analysis = demands.analysis;
    state->arena = expressions; state->selectors.period = state->program.primitives.period;
    if (!state->initialize()) { error = state->error; return failure(); }
    if (enclosing != ArrayRef<scf::ForOp>(state->enclosing)) {
        error = "arithmetic regional endpoint requires its original enclosing visit context"; return failure();
    }
    for (Value parameter : state->program.parameters) { state->parameters.push_back(state->arena->input(parameter)); }
    for (const auto& site : state->program.sites) {
        state->pipes.push_back(static_cast<uint32_t>(site.phase->kPipeValue));
    }
    auto plan = prepareGeneralArithmeticRegionalInsertion(state->program.context.function,
        state->program, state->analysis, error);
    if (failed(plan)) { return failure(); }
    (*plan)->completeInvocation = false;
    (*plan)->allocationPreparation = [state](PreparedLogicalPlan& prepared) {
        prepared.regionalAllocation = state->allocation(prepared);
    };
    transaction.commit();
    return plan;
}
FailureOr<RegionalAnalysis> analyzeArithmeticRegion(ArithmeticRegionContext context,
    const PhaseIndex& index, const SyncInput& input, std::shared_ptr<RegionExpressions> expressions,
    std::string& error, std::function<std::optional<RegionExpressions::Id>(Value)> parameterBinding)
{
    return analyzeArithmeticRegionImpl(context, index, input, std::move(expressions), error,
                                      std::move(parameterBinding), StorageExportRequest::SymbolicAllowed);
}
FailureOr<RegionalAnalysis> analyzeArithmeticRegionWithProfiles(ArithmeticRegionContext context,
    const PhaseIndex& index, const SyncInput& input, std::shared_ptr<RegionExpressions> expressions,
    std::string& error, ArrayRef<ArithmeticLimits> profiles,
    std::function<std::optional<RegionExpressions::Id>(Value)> parameterBinding)
{
    if (profiles.empty()) { error = "regional arithmetic has no declared profiles"; return failure(); }
    return analyzeArithmeticRegionImpl(context, index, input, std::move(expressions), error,
        std::move(parameterBinding), StorageExportRequest::SymbolicAllowed, nullptr, nullptr, profiles);
}
FailureOr<RegionalAnalysis> analyzeArithmeticRegionRetained(ArithmeticRegionContext context,
    const PhaseIndex& index, const SyncInput& input, std::shared_ptr<RegionExpressions> expressions,
    std::shared_ptr<const ArithmeticRegionalRelations>& demands, std::string& error)
{
    return analyzeArithmeticRegionRetained(context, index, input, std::move(expressions), demands, error, nullptr);
}
FailureOr<RegionalAnalysis> analyzeArithmeticRegionRetained(ArithmeticRegionContext context,
    const PhaseIndex& index, const SyncInput& input, std::shared_ptr<RegionExpressions> expressions,
    std::shared_ptr<const ArithmeticRegionalRelations>& demands, std::string& error,
    const ArithmeticProgram* certified)
{
    demands.reset();
    return analyzeArithmeticRegionImpl(context, index, input, std::move(expressions), error, {},
                                      StorageExportRequest::SymbolicAllowed, &demands, certified);
}
FailureOr<RegionalAnalysis> analyzeFiniteArithmeticRegion(ArithmeticRegionContext context,
    const PhaseIndex& index, const SyncInput& input, std::shared_ptr<RegionExpressions> expressions,
    std::string& error)
{
    return analyzeArithmeticRegionImpl(context, index, input, std::move(expressions), error, {},
                                      StorageExportRequest::Finite);
}
FailureOr<RegionalAnalysis> analyzeFiniteArithmeticRegionWithProfiles(ArithmeticRegionContext context,
    const PhaseIndex& index, const SyncInput& input, std::shared_ptr<RegionExpressions> expressions,
    std::string& error, ArrayRef<ArithmeticLimits> profiles)
{
    if (profiles.empty()) { error = "regional arithmetic has no declared profiles"; return failure(); }
    return analyzeArithmeticRegionImpl(context, index, input, std::move(expressions), error, {},
        StorageExportRequest::Finite, nullptr, nullptr, profiles);
}
FailureOr<RegionalAnalysis> exportRegionalRelationData(std::shared_ptr<RegionalRelations> relation,
    std::shared_ptr<RegionExpressions> expressions, std::string& error)
{
    const auto& data = relation->data;
    if (!data.input || !data.context.function || !expressions) { return failure(); }
    auto demandOnly = [&]() {
        RegionalAnalysis out;
        out.expressions = expressions; out.accessModel = &data.input->accesses();
        out.gmAliasPolicy = data.input->memory().gmPolicy(); out.relations = relation;
        relation->exportObligation = error;
        for (const auto& child : relation->children) {
            llvm::append_range(out.anchors, child.anchors);
            llvm::append_range(out.occurrenceLoops, child.occurrenceLoops);
            for (unsigned i = 0; i < child.anchors.size(); ++i) {
                out.outerLoops.push_back(child.outerLoops.empty() ? std::vector<scf::ForOp>{} : child.outerLoops[i]);
            }
        }
        return out;
    };
    auto state = std::make_shared<State>();
    state->input = data.input; state->program.context = data.context; state->program.sites = data.sites;
    state->program.parameters = data.parameterValues; state->parameters = data.parameters;
    state->program.incomingPrerequisites = data.incomingPrerequisites;
    state->program.extraction.dischargedEffects = data.dischargedEffects;
    state->program.primitives.period = data.analysis.period;
    state->program.primitives.pipeCount = data.analysis.pipeCount;
    state->program.primitives.parameters.resize(data.analysis.parameterCount);
    state->analysis = data.analysis; state->selectors = data.selectors; state->occurrences = data.occurrences;
    state->arena = expressions;
    if (!state->initialize()) { error = state->error; return demandOnly(); }
    state->enclosing = data.enclosing;
    for (const auto& site : data.sites) { state->pipes.push_back(static_cast<uint32_t>(site.phase->kPipeValue)); }
    auto result = exportState(state, *data.input, error, false);
    if (failed(result)) { return demandOnly(); }
    // The common relation representation owns crossing demands. Producers keep
    // their internal minimum-demand representation and its original recipes.
    result->arithmeticRelations.reset(); result->relations = relation;
    std::vector<TemplateEndpointAnchor> originals;
    for (const auto& child : relation->children) { llvm::append_range(originals, child.anchors); }
    if (originals.size() != result->anchors.size()) {
        error = "symbolic parent original endpoint owner count differs"; return failure();
    }
    result->anchors = std::move(originals);
    std::vector<bool> hasLeaf;
    result->occurrenceLoops.clear(); result->outerLoops.clear(); result->outerDivisors.clear();
    for (const auto& child : relation->children) {
        for (unsigned i = 0; i < child.anchors.size(); ++i) {
            hasLeaf.push_back(bool(child.occurrenceLoops[i]));
            result->occurrenceLoops.push_back(child.occurrenceLoops[i]);
            result->outerLoops.push_back(child.outerLoops.empty() ? std::vector<scf::ForOp>{} : child.outerLoops[i]);
        }
    }
    auto canonical = [hasLeaf, arena = state->arena](RegionalEvent event) {
        if (event.type >= hasLeaf.size()) { event.type = UINT32_MAX; return event; }
        if (!hasLeaf[event.type] && !event.visits.empty()) {
            event.ordinal = event.visits.back(); event.visits.pop_back();
        }
        return event;
    };
    std::vector<bool> hasCoordinates;
    for (const auto& site : data.sites) { hasCoordinates.push_back(!site.loops.empty()); }
    auto original = [hasLeaf, hasCoordinates, arena = state->arena](RegionalEvent event) {
        if (!hasLeaf[event.type] && hasCoordinates[event.type]) {
            event.visits.push_back(event.ordinal); event.ordinal = arena->constant(0);
        }
        return event;
    };
    auto oldPresence = result->presence;
    auto oldReach = result->reachability;
    auto oldBefore = result->referenceBefore;
    result->presence = [oldPresence, canonical](RegionalEvent event) { return oldPresence(canonical(event)); };
    result->reachability = [oldReach, canonical](RegionalEvent a, RegionalEvent b) {
        return oldReach(canonical(a), canonical(b));
    };
    result->referenceBefore = [oldBefore, canonical](RegionalEvent a, RegionalEvent b) {
        return oldBefore(canonical(a), canonical(b));
    };
    auto mapSelected = [original](auto& selected) {
        for (auto& choice : selected) { choice.event = original(choice.event); }
    };
    for (auto* table : {&result->firstPayloads, &result->lastPayloads, &result->firstSitePayloads}) {
        for (auto& [key, selected] : *table) { (void)key; mapSelected(selected); }
    }
    for (auto* boundaries : {&result->accessBoundary, &result->deferredAccessBoundary}) {
        for (auto& boundary : *boundaries) {
            boundary.first.event = original(boundary.first.event); boundary.last.event = original(boundary.last.event);
        }
    }
    auto oldStorage = result->storageSelectors;
    result->storageSelectors = [oldStorage, mapSelected](RegionalByteAddress address)
        -> std::optional<RegionalStorageSelectors> {
        auto selected = oldStorage(address);
        if (!selected) { return std::nullopt; }
        mapSelected(selected->firstWriters); mapSelected(selected->lastWriters);
        for (auto* table : {&selected->firstReaders, &selected->lastReaders}) {
            for (auto& [pipe, readers] : *table) { (void)pipe; mapSelected(readers); }
        }
        return selected;
    };
    auto crossingState = std::make_shared<State>(*state);
    crossingState->analysis.minimumDemands = relation->newCrossings;
    // A subset of certified minimum covers retains the endpoint functionality
    // used by the selector constructor. Internal recipes are emitted separately.
    crossingState->analysis.exactMinimum = true;
    crossingState->analysis.generators = relation->newCrossings;
    crossingState->handoffAllocation.reset();
    result->prepareWithVisits =
        [relation, state, crossingState, original, parent = *result](ArrayRef<scf::ForOp> enclosing)
        -> FailureOr<std::unique_ptr<PreparedLogicalPlan>> {
        if (enclosing != ArrayRef<scf::ForOp>(relation->data.enclosing)) { return failure(); }
        // Use the existing detached sequence import for record identities,
        // original preparation blocks and allocation provenance. Its crossing
        // list is empty: relation lowering supplies only the new crossings.
        auto merge = std::make_shared<SequenceAnalysisState>(relation->data.context.function, state->arena);
        merge->completeInvocation = false; merge->requiredOuterLoops.assign(enclosing.begin(), enclosing.end());
        std::vector<uint32_t> typeBases;
        uint32_t nextType = 0;
        for (const auto& child : relation->children) {
            if (child.anchors.size() > UINT32_MAX - nextType) { return failure(); }
            typeBases.push_back(nextType); nextType += child.anchors.size();
            Child imported; imported.regional = child; imported.anchors = child.anchors;
            merge->children.push_back(std::move(imported));
        }
        if (!relation->newCrossings.empty()) {
            Child crossing;
            crossing.regional = parent; crossing.regional.relations.reset();
            crossing.regional.endpointSiteGuard.reset(); crossing.regional.endpointInvocationGuard.reset();
            crossing.regional.endpointEventGuard = {};
            crossing.anchors = parent.anchors;
            crossing.regional.prepareWithVisits = [crossingState, original](ArrayRef<scf::ForOp> frames)
                -> FailureOr<std::unique_ptr<PreparedLogicalPlan>> {
                if (frames != ArrayRef<scf::ForOp>(crossingState->enclosing)) { return failure(); }
                auto prepared = prepareGeneralArithmeticRegionalInsertion(crossingState->program.context.function,
                    crossingState->program, crossingState->analysis, crossingState->emissionError);
                if (succeeded(prepared)) {
                    (*prepared)->allocationPreparation = [crossingState, original](PreparedLogicalPlan& plan) {
                        plan.regionalAllocation = crossingState->allocation(plan);
                        if (!plan.regionalAllocation) { return; }
                        for (auto& group : plan.regionalAllocation->groups) {
                            for (auto& member : group.members) {
                                member.firstSource = original(member.firstSource);
                                member.lastTarget = original(member.lastTarget);
                            }
                            for (auto& lane : group.lanes) {
                                for (auto* choices : {&lane.firstSources, &lane.lastTargets}) {
                                    for (auto& choice : *choices) { choice.event = original(choice.event); }
                                }
                            }
                        }
                    };
                }
                return prepared;
            };
            crossing.regional.prepare = [prepare = crossing.regional.prepareWithVisits]() { return prepare({}); };
            merge->children.push_back(std::move(crossing));
            typeBases.push_back(0); // Crossings already name the parent's original site table.
        }
        return merge->prepareWithTypeBases(enclosing, typeBases);
    };
    result->prepare = [prepare = result->prepareWithVisits]() { return prepare({}); };
    result->capabilities.endpointRecipes = llvm::all_of(relation->children, [](const auto& child) {
        return child.capabilities.endpointRecipes && (child.prepare || child.prepareWithVisits);
    });
    result->cost = {};
    for (auto member : {&RegionalCost::repeatedRegions, &RegionalCost::phaseDescriptions,
        &RegionalCost::cells, &RegionalCost::ports, &RegionalCost::crossings, &RegionalCost::physicalFragments,
        &RegionalCost::rotatingResidues, &RegionalCost::numericVisits, &RegionalCost::arithmeticRegions,
        &RegionalCost::boundaryBytes, &RegionalCost::selectorComparisons, &RegionalCost::crossingCandidates,
        &RegionalCost::implicationChecks, &RegionalCost::numericalLeafQueries, &RegionalCost::numericalIndexOperations,
        &RegionalCost::numericalMerges, &RegionalCost::numericalReusedChildren,
        &RegionalCost::retainedExpressionNodes}) {
        for (const auto& child : relation->children) {
            accumulateCost(result->cost.*member, child.cost.*member);
        }
    }
    result->cost.children = relation->children.size(); result->cost.expressionNodes = state->arena->size();
    return result;
}
} // namespace mlir::pto::frontiersynch
