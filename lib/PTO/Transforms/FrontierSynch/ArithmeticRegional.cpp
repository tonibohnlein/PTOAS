// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Exact arithmetic child interfaces for sequence composition.
#include "PTO/Transforms/FrontierSynch/HardwareProtection.h"
#include "PTO/Transforms/FrontierSynch/ArithmeticRegional.h"
#include "PTO/Transforms/FrontierSynch/ArithmeticStorageSelectors.h"
#include "PTO/Transforms/FrontierSynch/ArithmeticInsertion.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <map>
namespace mlir::pto::frontiersynch {
namespace {
using Id = RegionExpressions::Id;
struct State {
    ArithmeticProgram program;
    GeneralArithmeticDemandAnalysis analysis;
    ArithmeticStorageSelectors selectors;
    std::vector<ArithmeticIntegerPiece> primitives;
    using StorageKey = std::pair<AddressSpace, Value>;
    using Interval = std::pair<uint64_t, uint64_t>;
    llvm::MapVector<StorageKey, std::vector<Interval>> finiteStorage;
    std::shared_ptr<RegionExpressions> arena;
    std::vector<Id> parameters;
    std::vector<uint32_t> pipes;
    std::vector<scf::ForOp> enclosing;
    struct LoopGeometry { int64_t lower = 0, step = 1; uint64_t maximumOrdinal = 0; };
    std::map<Operation*, LoopGeometry> geometry;
    std::string error;
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
                APInt lower, step;
                if (!matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) ||
                    !matchPattern(loop.getStep(), m_ConstantInt(&step)) || !lower.isSignedIntN(64) ||
                    !step.isSignedIntN(64) || lower.getSExtValue() < 0 || step.getSExtValue() <= 0) {
                    error = "arithmetic regional ordinals require the recognized "
                            "nonnegative-lower positive-step domain";
                    return false;
                }
                const auto start = lower.getSExtValue(), stride = step.getSExtValue();
                geometry.emplace(loop, LoopGeometry{start, stride, static_cast<uint64_t>((INT64_MAX-start)/stride)});
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
            valid = arena->land(valid, arena->le((*values)[i], c(geometry.at(loops[i]).maximumOrdinal)));
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
            IntegerAffine affine{{BoundInteger(domain.step)}, BoundInteger(domain.lower)};
            (*result)[i] = arena->integerWitness(affine, BoundInteger(1), {(*result)[i]}, 1, {0}, 0);
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
        for (const auto& piece : primitives) {
            const auto& schema = *piece.schema;
            if (schema.kind != kind || schema.sourceSite != a.type ||
                (b && schema.targetSite != b->type)) { continue; }
            result = arena->lor(result, arena->integerPredicate(piece.system, values,
                program.primitives.period, piece.residues));
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
    std::vector<RegionalSelector> selected(const ArithmeticBoundarySelector& boundary,
                                           std::optional<Id> byte = std::nullopt)
    {
        std::vector<Id> values;
        if (byte) { values.push_back(*byte); }
        llvm::append_range(values, parameters);
        std::map<std::size_t, RegionalSelector> selected;
        // The exact extremum is unique. Overlapping pieces therefore select
        // the same tagged tuple; priority masks would only duplicate guards.
        for (const auto& piece : boundary.selector.pieces) {
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
                tuple.push_back(arena->integerWitness({{BoundInteger(1)}, -BoundInteger(domain.lower)},
                    BoundInteger(domain.step), {value}, 1, {0}, 0));
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
    for (const auto& access : state.primitives) {
        const auto& schema = *access.schema;
        if (schema.kind != PrimitiveKind::Reads && schema.kind != PrimitiveKind::Writes) { continue; }
        const auto byteColumn = schema.sourceDimensions;
        if (!schema.storageSpace || byteColumn >= access.system.dimensions() || byteColumn >= access.residues.size()) {
            state.error = "arithmetic storage primitive lacks its physical byte coordinate"; return false;
        }
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
                state.error = "arithmetic region needs a finite storage boundary or a symbolic crossing adapter";
                return false;
            }
            const BoundInteger period(state.program.primitives.period), residue(access.residues[byteColumn]);
            const auto begin = *lower * period + residue;
            const auto end = *upper * period + residue + 1;
            if (begin < 0 || end > BoundInteger(INT64_MAX)) {
                state.error = "arithmetic storage boundary is outside representable physical offsets"; return false;
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
} // namespace
FailureOr<RegionalAnalysis> analyzeArithmeticRegion(ArithmeticRegionContext context,
    const PhaseIndex& index, const SyncInput& input, std::shared_ptr<RegionExpressions> expressions,
    std::string& error, std::function<std::optional<RegionExpressions::Id>(Value)> parameterBinding)
{
    if (!context.function || !context.root || !expressions || !expressions->constructionError().empty()) {
        error = "arithmetic region requires a valid original root and expression arena";
        return failure();
    }
    auto state = std::make_shared<State>(); state->arena = std::move(expressions);
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
    // Fixed compiler input class; the bounds are not inferred from a kernel.
    state->program = recognizeArithmeticProgram(
        context, index, input, input.accesses(), {8, 8, 1, 4096}, entryConstant);
    if (state->program.extraction.state != RecognitionState::Applicable ||
        state->program.recognition.state != RecognitionState::Applicable) {
        state->program = recognizeArithmeticProgram(
            context, index, input, input.accesses(), {8, 8, 2, 4096}, entryConstant);
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
    if (!state->initialize()) { error = state->error; return failure(); }
    auto imported = importArithmeticIntegerPieces(state->program);
    if (failed(imported)) { error = "regional arithmetic primitive import failed"; return failure(); }
    state->primitives = std::move(*imported);
    if (!collectFiniteStorage(*state)) { error = state->error; return failure(); }
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
    const auto protection = structuredProtection(input.accesses());
    state->analysis = analyzeGeneralArithmeticDemandsWithProtection(state->program, protection);
    if (!state->analysis.error.empty()) { error = state->analysis.error; return failure(); }
    state->selectors = buildArithmeticStorageSelectors(state->program, state->pipes);
    if (!state->selectors.error.empty()) { error = state->selectors.error; return failure(); }
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
    if (!finiteBoundaries(*state, out)) { error = state->error; return failure(); }
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
            bool geometric = !input.accesses().effects()[effect].regions.empty();
            out.accessBoundary.push_back({effect, first[type].front(), last[type].front(), geometric});
        }
    }
    out.prepareWithVisits = [state](ArrayRef<scf::ForOp> enclosing)
        -> FailureOr<std::unique_ptr<PreparedLogicalPlan>> {
        if (enclosing != ArrayRef<scf::ForOp>(state->enclosing)) {
            state->error = "arithmetic regional endpoint requires its original enclosing visit context";
            return failure();
        }
        auto plan = prepareGeneralArithmeticRegionalInsertion(state->program.context.function, state->program,
                                                       state->analysis, state->error);
        if (succeeded(plan)) { (*plan)->completeInvocation = false; }
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
} // namespace mlir::pto::frontiersynch
