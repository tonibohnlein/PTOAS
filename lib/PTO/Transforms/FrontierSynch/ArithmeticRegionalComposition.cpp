// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/ArithmeticRegionalComposition.h"
#include "PTO/Transforms/FrontierSynch/RegionalRelations.h"
#include "PTO/Transforms/FrontierSynch/HardwareProtection.h"
#include <numeric>
#include <set>
namespace mlir::pto::frontiersynch {
namespace {
using Relation = GeneralArithmeticRelation;
using Systems = std::vector<IntegerSystem>;
using Boundary = ArithmeticBoundarySelector;
using Kind = ArithmeticBoundaryKind;
using Piece = GeneralArithmeticSelectorPiece;
void append(Systems& out, const IntegerSystem& system)
{
    if (!system.isKnownEmpty() && !llvm::is_contained(out, system)) { out.push_back(system); }
}
void removeEmpty(Relation& relation)
{
    for (auto it = relation.begin(); it != relation.end();) {
        if (it->second.empty()) { it = relation.erase(it); } else { ++it; }
    }
}
void unite(Relation& out, const Relation& other)
{
    for (const auto& [key, pieces] : other) {
        for (const auto& piece : pieces) { append(out[key], piece); }
    }
}
Relation renamed(const Relation& input, std::size_t offset)
{
    Relation result;
    for (const auto& [key, pieces] : input) {
        auto mapped = key;
        mapped.source.site += offset; mapped.target.site += offset;
        result.emplace(std::move(mapped), pieces);
    }
    return result;
}
class Composer {
public:
    Composer(const RegionalRelationData& left, const RegionalRelationData& right)
        : left(left), right(right), split(left.sites.size()), p(left.analysis.parameterCount) {}
    FailureOr<RegionalRelationData> run(ArithmeticRegionContext context);
    std::string error;
private:
    const RegionalRelationData &left, &right;
    std::size_t split;
    unsigned p;
    RegionalRelationData out;
    bool failedOperation = false;
    FailureOr<Systems> join(const IntegerSystem& a, ArrayRef<unsigned> am,
                           const IntegerSystem& b, ArrayRef<unsigned> bm,
                           unsigned dimensions, ArrayRef<unsigned> keep)
    {
        auto x = a.remap(dimensions, am), y = b.remap(dimensions, bm);
        if (failed(x) || failed(y)) { return failure(); }
        auto both = x->intersect(*y);
        ++out.analysis.cost.pieceJoins;
        if (failed(both)) { return failure(); }
        ++out.analysis.cost.projections;
        return both->project(keep);
    }
    Relation compose(const Relation& a, const Relation& b)
    {
        ++out.analysis.cost.relationCompositions;
        Relation result;
        using Match = std::pair<ArithmeticEventKey, std::vector<uint64_t>>;
        std::map<Match, std::vector<const Relation::value_type*>> bySource;
        for (const auto& entry : b) { bySource[{entry.first.source, entry.first.parameterResidues}].push_back(&entry); }
        for (const auto& [ak, av] : a) {
            auto found = bySource.find({ak.target, ak.parameterResidues});
            if (found == bySource.end()) { continue; }
            for (auto* entry : found->second) {
                const auto& [bk, bv] = *entry;
                const unsigned x = ak.source.residues.size(), z = ak.target.residues.size();
                const unsigned y = bk.target.residues.size(), dims = x + z + y + p;
                std::vector<unsigned> am(x + z + p), bm(z + y + p), keep(x + y + p);
                std::iota(am.begin(), am.begin() + x + z, 0);
                std::iota(am.begin() + x + z, am.end(), x + z + y);
                std::iota(bm.begin(), bm.begin() + z + y, x);
                std::iota(bm.begin() + z + y, bm.end(), x + z + y);
                std::iota(keep.begin(), keep.begin() + x, 0);
                std::iota(keep.begin() + x, keep.end(), x + z);
                ArithmeticRelationKey key{ak.source, bk.target, ak.parameterResidues};
                for (const auto& aa : av) {
                    for (const auto& bb : bv) {
                        auto pieces = join(aa, am, bb, bm, dims, keep);
                        if (failed(pieces)) { failedOperation = true; return {}; }
                        for (auto& piece : *pieces) { append(result[key], piece); }
                    }
                }
            }
        }
        return result;
    }
    Relation subtract(const Relation& a, const Relation& b)
    {
        Relation result;
        for (const auto& [key, pieces] : a) {
            auto match = b.find(key);
            if (match == b.end()) { result.emplace(key, pieces); continue; }
            ++out.analysis.cost.differences;
            auto selected = subtractIntegerUnions(key.source.residues.size() + key.target.residues.size() + p,
                                                  pieces, match->second);
            if (failed(selected)) { failedOperation = true; return {}; }
            for (auto& piece : *selected) { append(result[key], piece); }
        }
        return result;
    }
    Relation identities(ArrayRef<ArithmeticOccurrenceDomain> domains)
    {
        Relation result;
        for (const auto& domain : domains) {
            const unsigned d = domain.residues.size();
            std::vector<unsigned> map(d + p);
            std::iota(map.begin(), map.begin() + d, 0);
            std::iota(map.begin() + d, map.end(), 2 * d);
            auto lifted = domain.system.remap(2 * d + p, map);
            std::vector<IntegerConstraint> rows;
            for (unsigned i = 0; i < d; ++i) {
                IntegerConstraint row;
                row.coefficients.resize(2 * d + p);
                row.coefficients[i] = BoundInteger(1); row.coefficients[d + i] = BoundInteger(-1);
                rows.push_back(row);
                for (auto& c : row.coefficients) { c = -c; }
                rows.push_back(std::move(row));
            }
            auto equal = IntegerSystem::create(2 * d + p, rows);
            if (failed(lifted) || failed(equal)) { failedOperation = true; return {}; }
            auto present = lifted->intersect(*equal);
            if (failed(present)) { failedOperation = true; return {}; }
            for (auto kind : {ArithmeticEvent::Start, ArithmeticEvent::Completion}) {
                ArithmeticEventKey event{domain.site, kind, domain.residues};
                append(result[{event, event, domain.parameterResidues}], *present);
            }
        }
        return result;
    }
    Relation nativeCrossings()
    {
        Relation result;
        for (const auto& a : left.occurrences) {
            for (const auto& b : right.occurrences) {
                if (a.parameterResidues != b.parameterResidues ||
                    left.sites[a.site].phase->kPipeValue != right.sites[b.site].phase->kPipeValue) {
                        continue; }
                const unsigned x = a.residues.size(), y = b.residues.size(), dims = x + y + p;
                std::vector<unsigned> am(x + p), bm(y + p), keep(dims);
                std::iota(am.begin(), am.begin() + x, 0);
                std::iota(am.begin() + x, am.end(), x + y);
                std::iota(bm.begin(), bm.end(), x);
                std::iota(keep.begin(), keep.end(), 0);
                auto products = join(a.system, am, b.system, bm, dims, keep);
                if (failed(products)) { failedOperation = true; return {}; }
                for (auto kinds : {std::make_pair(ArithmeticEvent::Start, ArithmeticEvent::Start),
                                   std::make_pair(ArithmeticEvent::Completion, ArithmeticEvent::Completion),
                                   std::make_pair(ArithmeticEvent::Start, ArithmeticEvent::Completion)}) {
                    ArithmeticRelationKey key{{a.site, kinds.first, a.residues},
                                              {split + b.site, kinds.second, b.residues}, a.parameterResidues};
                    for (auto& piece : *products) { append(result[key], piece); }
                }
            }
        }
        return result;
    }
    // Domain columns: shared byte, then shared parameters. Add exact output
    // quotient equalities from the selector witness; no functional resynthesis.
    FailureOr<IntegerSystem> selectorGraph(const Piece& piece, unsigned total, unsigned first, unsigned byte)
    {
        if (piece.inputResidues.size() != 1 || piece.parameterResidues.size() != p ||
            piece.domain.dimensions() != 1 + p) { return failure(); }
        std::vector<unsigned> map(1 + p);
        map[0] = byte;
        std::iota(map.begin() + 1, map.end(), total - p);
        auto domain = piece.domain.remap(total, map);
        if (failed(domain)) { return failure(); }
        std::vector<IntegerConstraint> rows;
        for (unsigned i = 0; i < piece.outputs.size(); ++i) {
            const auto& output = piece.outputs[i];
            if (output.denominator <= BoundInteger(0) ||
                output.numerator.coefficients.size() != 1 + p) { return failure(); }
            IntegerConstraint row;
            row.coefficients.resize(total);
            row.coefficients[first + i] = output.denominator;
            for (unsigned j = 0; j < map.size(); ++j) { row.coefficients[map[j]] -= output.numerator.coefficients[j]; }
            row.bound = output.numerator.constant;
            rows.push_back(row);
            for (auto& c : row.coefficients) { c = -c; }
            row.bound = -row.bound; rows.push_back(std::move(row));
        }
        auto equalities = IntegerSystem::create(total, rows);
        return failed(equalities) ? FailureOr<IntegerSystem>(failure()) : domain->intersect(*equalities);
    }
    Relation bridges()
    {
        Relation result;
        const auto protection = structuredProtection(left.input->accesses());
        for (const auto& a : left.selectors.boundaries) {
            for (const auto& b : right.selectors.boundaries) {
                const bool pair = (a.kind == Kind::LastWriter &&
                    (b.kind == Kind::FirstWriter || b.kind == Kind::FirstReaderBeforeWrite)) ||
                    (a.kind == Kind::LastReaderAfterWrite && b.kind == Kind::FirstWriter);
                if (!pair || !a.storageSpace || a.storageSpace != b.storageSpace ||
                    a.storageBase != b.storageBase) { continue; }
                for (const auto& x : a.selector.pieces) {
                    for (const auto& y : b.selector.pieces) {
                        if (x.inputResidues != y.inputResidues ||
                            x.parameterResidues != y.parameterResidues) { continue; }
                        auto* source = left.sites[x.outputSite].phase;
                        auto* target = right.sites[y.outputSite].phase;
                        auto sourcePipe = static_cast<uint32_t>(source->kPipeValue);
                        auto targetPipe = static_cast<uint32_t>(target->kPipeValue);
                        if (ptoStorageProtection().protectsScalar(sourcePipe, targetPipe)) { continue; }
                        if (*a.storageSpace == AddressSpace::ACC && hardwareProtectsConflict(
                                sourcePipe, protection.within(source, left.enclosing),
                                targetPipe, protection.within(target, right.enclosing))) { continue; }
                        unsigned nx = x.outputs.size(), ny = y.outputs.size(), total = nx + ny + 1 + p;
                        auto xx = selectorGraph(x, total, 0, nx + ny);
                        auto yy = selectorGraph(y, total, nx, nx + ny);
                        if (failed(xx) || failed(yy)) { failedOperation = true; return {}; }
                        auto both = xx->intersect(*yy);
                        if (failed(both)) { failedOperation = true; return {}; }
                        std::vector<unsigned> keep(nx + ny + p);
                        std::iota(keep.begin(), keep.begin() + nx + ny, 0);
                        std::iota(keep.begin() + nx + ny, keep.end(), nx + ny + 1);
                        auto projected = both->project(keep);
                        ++out.analysis.cost.pieceJoins; ++out.analysis.cost.projections;
                        if (failed(projected)) { failedOperation = true; return {}; }
                        ArithmeticRelationKey key{{x.outputSite, ArithmeticEvent::Completion, {}},
                                                  {split + y.outputSite, ArithmeticEvent::Start, {}},
                                                      x.parameterResidues};
                        for (auto& v : x.outputs) { key.source.residues.push_back(v.residue); }
                        for (auto& v : y.outputs) { key.target.residues.push_back(v.residue); }
                        for (auto& piece : *projected) { append(result[key], piece); }
                    }
                }
            }
        }
        return result;
    }
    bool selectors();
};
bool Composer::selectors()
{
    out.selectors.period = left.selectors.period; out.selectors.parameterCount = p;
    out.selectors.support = left.selectors.support;
    out.selectors.support.insert(out.selectors.support.end(), right.selectors.support.begin(),
        right.selectors.support.end());
    for (bool fromLeft : {true, false}) {
        const auto& own = fromLeft ? left.selectors : right.selectors;
        const auto& other = fromLeft ? right.selectors : left.selectors;
        for (const auto& original : own.boundaries) {
            auto boundary = original;
            if (!fromLeft && boundary.site) { *boundary.site += split; }
            boundary.selector.pieces.clear();
            for (const auto& candidate : original.selector.pieces) {
                Systems blocked;
                for (const auto& blocker : other.boundaries) {
                    bool blocks = false;
                    const bool sameStorage = original.storageSpace == blocker.storageSpace &&
                        original.storageBase == blocker.storageBase;
                    const bool samePipe = original.pipe == blocker.pipe;
                    switch (original.kind) {
                    case Kind::FirstWriter:
                        blocks = !fromLeft && sameStorage && blocker.kind == Kind::FirstWriter; break;
                    case Kind::LastWriter:
                        blocks = fromLeft && sameStorage && blocker.kind == Kind::LastWriter; break;
                    case Kind::FirstReaderBeforeWrite:
                        blocks = !fromLeft && sameStorage && (blocker.kind == Kind::FirstWriter ||
                            (samePipe && blocker.kind == Kind::FirstReaderBeforeWrite)); break;
                    case Kind::LastReaderAfterWrite:
                        blocks = fromLeft && sameStorage && (blocker.kind == Kind::LastWriter ||
                            (samePipe && blocker.kind == Kind::LastReaderAfterWrite)); break;
                    case Kind::FirstPayload:
                        blocks = !fromLeft && samePipe && blocker.kind == Kind::FirstPayload; break;
                    case Kind::LastPayload:
                        blocks = fromLeft && samePipe && blocker.kind == Kind::LastPayload; break;
                    case Kind::FirstSite: case Kind::LastSite: break;
                    }
                    if (!blocks) { continue; }
                    for (const auto& piece : blocker.selector.pieces) {
                        if (piece.inputResidues == candidate.inputResidues &&
                            piece.parameterResidues == candidate.parameterResidues) { append(blocked, piece.domain); }
                    }
                }
                auto domains = subtractIntegerUnions(candidate.domain.dimensions(), {candidate.domain}, blocked);
                ++out.analysis.cost.differences;
                if (failed(domains)) { return false; }
                for (auto& domain : *domains) {
                    auto piece = candidate; piece.domain = std::move(domain);
                    if (!fromLeft) { piece.outputSite += split; }
                    boundary.selector.pieces.push_back(std::move(piece));
                }
            }
            if (!boundary.selector.pieces.empty()) { out.selectors.boundaries.push_back(std::move(boundary)); }
        }
    }
    return true;
}
bool validExport(const RegionalRelationData& region)
{
    const auto period = region.analysis.period;
    const auto parameters = region.analysis.parameterCount;
    if (!period || region.parameters.size() != parameters ||
        llvm::any_of(region.sites, [](const auto& site) { return !site.phase; })) { return false; }
    auto residues = [&](ArrayRef<uint64_t> values) {
        return llvm::all_of(values, [&](uint64_t value) { return value < period; });
    };
    auto endpoint = [&](const ArithmeticEventKey& event) {
        return event.site < region.sites.size() &&
            region.sites[event.site].phase &&
            event.residues.size() == region.sites[event.site].loops.size() && residues(event.residues) &&
            (event.event == ArithmeticEvent::Start || event.event == ArithmeticEvent::Completion);
    };
    for (const auto* relation : {&region.analysis.nativeOrder, &region.analysis.requiredOrder,
                                 &region.analysis.minimumDemands}) {
        for (const auto& [key, pieces] : *relation) {
            if (!endpoint(key.source) || !endpoint(key.target) || key.parameterResidues.size() != parameters ||
                !residues(key.parameterResidues)) { return false; }
            for (const auto& piece : pieces) {
                if (piece.dimensions() != key.source.residues.size() + key.target.residues.size() + parameters) {
                    return false;
                }
            }
        }
    }
    for (const auto& domain : region.occurrences) {
        if (!endpoint({domain.site, ArithmeticEvent::Start, domain.residues}) ||
            domain.parameterResidues.size() != parameters || !residues(domain.parameterResidues) ||
            domain.system.dimensions() != domain.residues.size() + parameters) { return false; }
    }
    for (const auto& support : region.selectors.support) {
        if (support.byteResidue >= period || support.parameterResidues.size() != parameters ||
            !residues(support.parameterResidues) || support.domain.dimensions() != 1 + parameters) { return false; }
    }
    for (const auto& boundary : region.selectors.boundaries) {
        const auto inputs = boundary.storageSpace ? 1U : 0U;
        if (boundary.selector.inputDimensions != inputs || boundary.selector.parameterCount != parameters ||
            (boundary.site && *boundary.site >= region.sites.size())) { return false; }
        for (const auto& piece : boundary.selector.pieces) {
            if (piece.outputSite >= region.sites.size() ||
                piece.outputs.size() != region.sites[piece.outputSite].loops.size() ||
                piece.inputResidues.size() != inputs || piece.parameterResidues.size() != parameters ||
                !residues(piece.inputResidues) || !residues(piece.parameterResidues) ||
                piece.domain.dimensions() != inputs + parameters) { return false; }
            for (const auto& output : piece.outputs) {
                if (output.residue >= period || output.denominator <= BoundInteger(0) ||
                    output.numerator.coefficients.size() != inputs + parameters) { return false; }
            }
        }
    }
    return true;
}
FailureOr<RegionalRelationData> Composer::run(ArithmeticRegionContext context)
{
    if (!validExport(left) || !validExport(right) ||
        !left.completeRequiredOrder || !right.completeRequiredOrder ||
        !left.analysis.error.empty() || !right.analysis.error.empty() ||
        !left.selectors.error.empty() || !right.selectors.error.empty() ||
        left.analysis.period != right.analysis.period || left.analysis.period != left.selectors.period ||
        right.analysis.period != right.selectors.period || !left.analysis.period ||
        left.input != right.input || !left.input || left.enclosing != right.enclosing ||
        left.parameterValues != right.parameterValues || left.parameters != right.parameters ||
        left.analysis.parameterCount != right.analysis.parameterCount ||
        p != left.parameterValues.size() || p != left.selectors.parameterCount ||
            p != right.selectors.parameterCount) {
        error = "symbolic sibling composition requires compatible exact parameter and residue interfaces";
        return failure();
    }
    if (!left.incomingPrerequisites.empty() || !right.incomingPrerequisites.empty()) {
        error = "symbolic sibling scalar prerequisite relation adapter unavailable"; return failure();
    }
    SmallVector<SyncStorageCell> domains;
    for (const auto* child : {&left, &right}) {
        for (const auto& support : child->selectors.support) {
            SyncStorageCell domain{support.space, 0, 1, support.base};
            if (llvm::none_of(domains, [&](const auto& old) { return sameStorageDomain(old, domain); })) {
                domains.push_back(domain);
            }
        }
    }
    if (!storageBasesAreComparable(domains, left.input->memory().gmPolicy())) {
        error = "symbolic sibling physical-base relation adapter unavailable"; return failure();
    }
    if (left.context.function != right.context.function ||
        context.function != left.context.function) {
        error = "symbolic sibling invocation contexts differ"; return failure();
    }
    // Cumulative work is retained by the reusable parent export. outputPieces
    // below describes the current result, not a sum of intermediate snapshots.
    for (auto member : {&ArithmeticAnalysisCost::primitivePieces, &ArithmeticAnalysisCost::pieceJoins,
        &ArithmeticAnalysisCost::projections, &ArithmeticAnalysisCost::relationCompositions,
        &ArithmeticAnalysisCost::differences}) {
        if (left.analysis.cost.*member > UINT64_MAX - right.analysis.cost.*member) {
            error = "symbolic composition cost exceeds representation"; return failure();
        }
        out.analysis.cost.*member = left.analysis.cost.*member + right.analysis.cost.*member;
    }
    for (auto member : {&RegionExpressions::RelationCost::gates, &RegionExpressions::RelationCost::pieces,
        &RegionExpressions::RelationCost::projections, &RegionExpressions::RelationCost::formulaProducts}) {
        if (left.translationCost.*member > UINT64_MAX - right.translationCost.*member) {
            error = "symbolic relation translation cost exceeds representation"; return failure();
        }
        out.translationCost.*member = left.translationCost.*member + right.translationCost.*member;
    }
    out.translationCost.peakClauses = std::max(left.translationCost.peakClauses, right.translationCost.peakClauses);
    out.context = context;
    out.dischargedEffects = left.dischargedEffects;
    for (auto effect : right.dischargedEffects) {
        if (!llvm::is_contained(out.dischargedEffects, effect)) {
            out.dischargedEffects.push_back(effect);
        }
    }
    out.parameterValues = left.parameterValues;
    out.sites = left.sites;
    out.sites.append(right.sites.begin(), right.sites.end());
    std::set<uint32_t> pipes;
    for (const auto& site : out.sites) { pipes.insert(static_cast<uint32_t>(site.phase->kPipeValue)); }
    out.parameters = left.parameters; out.input = left.input; out.enclosing = left.enclosing;
    out.occurrences = left.occurrences;
    for (auto domain : right.occurrences) { domain.site += split; out.occurrences.push_back(std::move(domain)); }
    out.analysis.period = left.analysis.period; out.analysis.parameterCount = p;
    out.analysis.pipeCount = pipes.size();
    auto lh = left.analysis.requiredOrder, rh = renamed(right.analysis.requiredOrder, split);
    auto lid = identities(left.occurrences);
    auto rid = renamed(identities(right.occurrences), split);
    auto lref = lh, rref = rh;
    unite(lref, lid); unite(rref, rid);
    auto bridge = bridges(), nativeEdges = nativeCrossings();
    auto native = compose(compose(left.analysis.nativeOrder, nativeEdges),
                          renamed(right.analysis.nativeOrder, split));
    auto crossings = bridge;
    unite(crossings, native);
    auto allCross = compose(compose(lref, crossings), rref);
    auto alternate = compose(compose(lh, crossings), rref);
    unite(alternate, compose(compose(lref, crossings), rh));
    unite(alternate, native);
    auto retained = subtract(bridge, alternate);
    if (failedOperation || !selectors()) {
        error = "exact symbolic crossing composition or storage-selector projection failed"; return failure();
    }
    out.analysis.generators = left.analysis.minimumDemands;
    unite(out.analysis.generators, renamed(right.analysis.minimumDemands, split));
    unite(out.analysis.generators, bridge);
    out.analysis.minimumDemands = left.analysis.minimumDemands;
    unite(out.analysis.minimumDemands, renamed(right.analysis.minimumDemands, split));
    unite(out.analysis.minimumDemands, retained);
    out.analysis.nativeOrder = left.analysis.nativeOrder;
    unite(out.analysis.nativeOrder, renamed(right.analysis.nativeOrder, split));
    unite(out.analysis.nativeOrder, native);
    out.analysis.requiredOrder = lh; unite(out.analysis.requiredOrder, rh); unite(out.analysis.requiredOrder, allCross);
    for (auto* relation : {&out.analysis.generators, &out.analysis.minimumDemands,
                           &out.analysis.nativeOrder, &out.analysis.requiredOrder}) { removeEmpty(*relation); }
    out.completeRequiredOrder = true;
    out.analysis.exactMinimum = left.analysis.exactMinimum && right.analysis.exactMinimum;
    // This adapter does not certify adjacency; insertion may use consumer cuts.
    out.analysis.adjacentLocalDemands = false;
    for (const auto& [key, pieces] : out.analysis.minimumDemands) {
        (void)key; out.analysis.cost.outputPieces += pieces.size();
    }
    return std::move(out);
}
} // namespace
FailureOr<RegionalRelationData> composeRegionalRelationData(
    const RegionalRelationData& left, const RegionalRelationData& right,
    ArithmeticRegionContext context, std::string& error)
{
    Composer composer(left, right);
    auto result = composer.run(context);
    if (failed(result)) { error = std::move(composer.error); }
    return result;
}
RegionalRelationData arithmeticRelationData(const ArithmeticRegionalRelations& in)
{
    RegionalRelationData out;
    out.input = in.input; out.context = in.program.context; out.sites = in.program.sites;
    out.parameterValues = in.program.parameters; out.parameters = in.parameters; out.enclosing = in.enclosing;
    out.incomingPrerequisites = in.program.incomingPrerequisites;
    out.dischargedEffects = in.program.extraction.dischargedEffects;
    out.analysis = in.analysis; out.selectors = in.selectors; out.occurrences = in.occurrences;
    out.completeRequiredOrder = in.analysis.exactMinimum && in.analysis.error.empty();
    return out;
}
FailureOr<ArithmeticRegionalRelations> composeArithmeticRegionalRelations(
    const ArithmeticRegionalRelations& left, const ArithmeticRegionalRelations& right,
    ArithmeticRegionContext context, std::string& error)
{
    auto data = composeRegionalRelationData(
        arithmeticRelationData(left), arithmeticRelationData(right), context, error);
    if (failed(data)) { return failure(); }
    ArithmeticRegionalRelations out;
    out.input = data->input; out.program.context = data->context; out.program.sites = data->sites;
    out.program.parameters = data->parameterValues; out.parameters = data->parameters; out.enclosing = data->enclosing;
    out.program.extraction.dischargedEffects = data->dischargedEffects;
    out.program.primitives.parameters = left.program.primitives.parameters;
    out.program.primitives.period = data->analysis.period; out.program.primitives.pipeCount = data->analysis.pipeCount;
    out.analysis = std::move(data->analysis); out.selectors = std::move(data->selectors);
    out.occurrences = std::move(data->occurrences);
    return out;
}
} // namespace mlir::pto::frontiersynch
