// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "RegionalRelationsInternal.h"
#include "CountedLoop.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <set>
#include "llvm/ADT/ScopeExit.h"
namespace mlir::pto::frontiersynch {
namespace {
using Id = RegionExpressions::Id;
using Kind = ArithmeticBoundaryKind;
struct Formula {
    Id predicate;
    std::vector<Id> coordinates;
    std::optional<ArithmeticRelationKey> relation;
    std::optional<std::size_t> occurrence;
    bool native = false;
};
struct SelectorFormula {
    ArithmeticBoundarySelector boundary;
    std::size_t site;
    Id predicate;
    std::vector<Id> inputs, outputs;
};
class Request final : public RegionalRelationRequest {
public:
    Request(const RegionalAnalysis& region, func::FuncOp function, const SyncInput& input, const PhaseIndex& index)
        : region(region), function(function), input(input), index(index), e(*region.expressions)
    {
        variables = std::make_shared<Block>();
        e.retainInputOwner(variables);
    }
    ArrayRef<Value> parameters() const override { return parameterValues; }
    bool build(std::string& diagnostic);
    FailureOr<RegionalRelationData> lower(ArrayRef<Value> parameters, std::string& diagnostic) override;
private:
    RegionalAnalysis region;
    func::FuncOp function;
    const SyncInput& input;
    const PhaseIndex& index;
    RegionExpressions& e;
    RegionalRelationData data;
    SmallVector<Value> parameterValues;
    std::set<Id> coordinateIds;
    std::vector<Formula> formulas;
    std::vector<SelectorFormula> selectors;
    std::string error;
    Id c(uint64_t n) { return e.constant(n); }
    Id yes() { return e.boolean(true); }
    Id no() { return e.boolean(false); }
    bool fail(StringRef message) { error = message.str(); return false; }
    Id variable()
    {
        auto value = variables->addArgument(IndexType::get(function.getContext()), function.getLoc());
        auto id = e.input(value); coordinateIds.insert(id); return id;
    }
    bool invariant(Value value, DenseSet<Value>& visiting)
    {
        if (!visiting.insert(value).second) { return false; }
        auto remove = llvm::make_scope_exit([&] { visiting.erase(value); });
        if (auto argument = dyn_cast<BlockArgument>(value)) {
            auto* owner = argument.getOwner()->getParentOp();
            return owner == function.getOperation() || (owner && data.context.root &&
                owner != data.context.root && owner->isProperAncestor(data.context.root));
        }
        auto* operation = value.getDefiningOp();
        if (!operation || operation->getNumRegions() || !isMemoryEffectFree(operation) ||
            !index.phasesFor(operation).empty()) { return false; }
        return llvm::all_of(operation->getOperands(), [&](Value operand) { return invariant(operand, visiting); });
    }
    bool observe(Id predicate)
    {
        if (predicate == RegionExpressions::invalid) {
            return fail("regional relation callback returned an invalid DAG");
        }
        for (auto [id, value] : e.referencedInputs(predicate)) {
            if (coordinateIds.count(id) || llvm::is_contained(parameterValues, value)) { continue; }
            // A parameter is fixed throughout this region. Local induction or
            // payload values need a coordinate/guard adapter, not a free symbol.
            DenseSet<Value> visiting;
            if (!invariant(value, visiting)) {
                return fail("regional relation callback has an unbound local parameter");
            }
            parameterValues.push_back(value);
        }
        return true;
    }
    std::optional<RegionalEvent> event(std::size_t site, ArrayRef<Id> values, Id& valid)
    {
        RegionalEvent result{static_cast<uint32_t>(site), c(0), PeriodicEventKind::Start};
        std::vector<Id> ordinals;
        std::vector<std::pair<Id, Id>> bindings;
        const auto& loops = data.sites[site].loops;
        if (loops.size() != values.size()) { return std::nullopt; }
        for (unsigned i = 0; i < loops.size(); ++i) {
            auto loop = loops[i]; auto geometry = CountedLoop::get(loop);
            if (!geometry) { return std::nullopt; }
            RegionExpressions::Substitution context(bindings);
            auto lower = e.substitute(e.input(loop.getLowerBound()), context);
            auto upper = e.substitute(e.input(loop.getUpperBound()), context);
            auto difference = e.sub(values[i], lower);
            auto ordinal = e.div(difference, c(geometry->step));
            valid = e.land(valid, e.land(e.sle(lower, values[i]), e.slt(values[i], upper)));
            valid = e.land(valid, e.eq(e.rem(difference, c(geometry->step)), c(0)));
            valid = e.land(valid, e.le(ordinal, c(geometry->maximumOrdinal)));
            ordinals.push_back(ordinal); bindings.push_back({e.input(loop.getInductionVar()), values[i]});
        }
        if (region.occurrenceLoops[site] && !ordinals.empty()) {
            result.ordinal = ordinals.back(); ordinals.pop_back();
        }
        result.visits = std::move(ordinals);
        return result;
    }
    std::optional<std::vector<Id>> raw(RegionalEvent event)
    {
        if (!validRegionalEvent(region, event)) { return std::nullopt; }
        auto ordinals = event.visits;
        const auto& loops = data.sites[event.type].loops;
        if (region.occurrenceLoops[event.type]) { ordinals.push_back(event.ordinal); }
        if (ordinals.size() != loops.size()) { return std::nullopt; }
        std::vector<Id> values;
        std::vector<std::pair<Id, Id>> bindings;
        for (unsigned i = 0; i < loops.size(); ++i) {
            auto loop = loops[i]; auto geometry = CountedLoop::get(loop);
            if (!geometry) { return std::nullopt; }
            RegionExpressions::Substitution context(bindings);
            auto lower = e.substitute(e.input(loop.getLowerBound()), context);
            auto value = e.integerWitness({{BoundInteger(geometry->step), BoundInteger(1)}, BoundInteger(0)},
                BoundInteger(1), {ordinals[i], lower}, 1, {0, 0}, 0);
            values.push_back(value); bindings.push_back({e.input(loop.getInductionVar()), value});
        }
        return values;
    }
    bool addSelector(Kind kind, std::optional<AddressSpace> space, Value base, std::optional<uint32_t> pipe,
                     const RegionalSelector& selector, std::optional<Id> byte = {})
    {
        auto outputs = raw(selector.event);
        auto present = regionalPresence(region, selector.event);
        if (!outputs || !present) { return fail("regional storage selector has unavailable coordinate/presence maps"); }
        SelectorFormula result;
        result.boundary.kind = kind; result.boundary.storageSpace = space;
        result.boundary.storageBase = base; result.boundary.pipe = pipe;
        if (kind == Kind::FirstSite || kind == Kind::LastSite) { result.boundary.site = selector.event.type; }
        result.site = selector.event.type; result.predicate = e.land(selector.present, *present);
        if (byte) { result.inputs.push_back(*byte); }
        for (auto value : *outputs) {
            auto output = variable(); result.outputs.push_back(output);
            result.predicate = e.land(result.predicate, e.eq(output, value));
        }
        if (!observe(result.predicate)) { return false; }
        selectors.push_back(std::move(result)); return true;
    }
};
bool Request::build(std::string& diagnostic)
{
    auto finish = [&](bool success) { diagnostic = error; return success; };
    if (!region.capabilities.completeStorageModel || !region.capabilities.exactQueries ||
        !region.capabilities.exactSelectors || region.accessModel != &input.accesses() ||
        region.gmAliasPolicy != input.memory().gmPolicy() || region.occurrenceLoops.size() != region.anchors.size()) {
        return finish(fail("regional relations require compatible complete modeled queries and selectors"));
    }
    data.input = &input; data.context.function = function;
    for (unsigned i = 0; i < region.anchors.size(); ++i) {
        const auto& anchor = region.anchors[i];
        if (!anchor.phase || !anchor.coordinates.empty() ||
            (!region.outerDivisors.empty() && llvm::any_of(region.outerDivisors[i], [](auto n) { return n != 1; }))) {
            return finish(fail("regional relation original-coordinate adapter unavailable"));
        }
        ArithmeticSite site; site.phase = anchor.phase;
        if (!region.outerLoops.empty()) { llvm::append_range(site.loops, region.outerLoops[i]); }
        if (region.occurrenceLoops[i]) { site.loops.push_back(region.occurrenceLoops[i]); }
        Operation* root = site.loops.empty() ? anchor.phase->elementOp : site.loops.front().getOperation();
        if (!data.context.root) { data.context.root = root; }
        while (data.context.root && data.context.root != root && !data.context.root->isProperAncestor(root)) {
            data.context.root = data.context.root->getParentOp();
        }
        data.sites.push_back(std::move(site));
    }
    if (!data.context.root) { data.context.root = function; }
    for (auto* ancestor = data.context.root->getParentOp(); ancestor; ancestor = ancestor->getParentOp()) {
        if (auto loop = dyn_cast<scf::ForOp>(ancestor)) { data.enclosing.push_back(loop); }
    }
    std::reverse(data.enclosing.begin(), data.enclosing.end());
    for (const auto& site : data.sites) {
        for (const auto& prerequisite : index.prerequisitesFor(site.phase->elementOp)) {
            if (prerequisite.producer) {
                return finish(fail("regional relation additional scalar prerequisite adapter unavailable"));
            }
        }
    }
    std::vector<std::vector<Id>> first, second;
    for (unsigned site = 0; site < data.sites.size(); ++site) {
        std::vector<Id> a, b;
        for (unsigned i = 0; i < data.sites[site].loops.size(); ++i) {
            a.push_back(variable()); b.push_back(variable());
        }
        Id valid = yes(); auto occurrence = event(site, a, valid);
        auto present = occurrence ? regionalPresence(region, *occurrence) : std::nullopt;
        if (!present) { return finish(fail("regional relation occurrence callback unavailable")); }
        auto predicate = e.land(valid, *present);
        if (!observe(predicate)) { return finish(false); }
        formulas.push_back({predicate, a, std::nullopt, site, false});
        first.push_back(a); second.push_back(b);
    }
    for (unsigned source = 0; source < data.sites.size(); ++source) {
        for (unsigned target = 0; target < data.sites.size(); ++target) {
            Id valid = yes(); auto a = event(source, first[source], valid), b = event(target, second[target], valid);
            if (!a || !b) { return finish(fail("regional relation event coordinates unavailable")); }
            auto pa = regionalPresence(region, *a), pb = regionalPresence(region, *b);
            auto before = regionalReferenceBefore(region, *a, *b);
            if (!pa || !pb || !before) {
                return finish(fail("regional relation reference/presence query unavailable"));
            }
            auto present = e.land(valid, e.land(*pa, *pb));
            auto equal = source == target ? yes() : no();
            if (source == target) {
                for (unsigned i = 0; i < first[source].size(); ++i) {
                    equal = e.land(equal, e.eq(first[source][i], second[target][i]));
                }
            }
            auto coordinates = first[source]; llvm::append_range(coordinates, second[target]);
            for (auto ak : {PeriodicEventKind::Start, PeriodicEventKind::Completion}) {
                for (auto bk : {PeriodicEventKind::Start, PeriodicEventKind::Completion}) {
                    a->kind = ak; b->kind = bk;
                    auto required = regionalReachability(region, *a, *b);
                    if (!required) { return finish(fail("regional all-event relation callback unavailable")); }
                    auto kind = [](auto k) { return k == PeriodicEventKind::Start ?
                        ArithmeticEvent::Start : ArithmeticEvent::Completion; };
                    ArithmeticRelationKey key{{source, kind(ak), std::vector<uint64_t>(first[source].size(), 0)},
                        {target, kind(bk), std::vector<uint64_t>(second[target].size(), 0)}, {}};
                    auto strict = e.land(present, e.land(*required, ak == bk ? e.lnot(equal) : yes()));
                    if (!observe(strict)) { return finish(false); }
                    formulas.push_back({strict, coordinates, key, std::nullopt, false});
                    const bool samePipe = data.sites[source].phase->kPipeValue == data.sites[target].phase->kPipeValue;
                    auto native = no();
                    if (ak == bk || (ak == PeriodicEventKind::Start && bk == PeriodicEventKind::Completion)) {
                        native = e.lor(equal, samePipe ? *before : no());
                    }
                    native = e.land(present, native);
                    if (!observe(native)) { return finish(false); }
                    formulas.push_back({native, coordinates, key, std::nullopt, true});
                }
            }
        }
    }
    using Domain = std::pair<AddressSpace, Value>;
    std::vector<Domain> domains;
    auto addDomain = [&](AddressSpace space, Value base) {
        Domain domain{space, base};
        if (!llvm::is_contained(domains, domain)) { domains.push_back(domain); }
    };
    for (const auto& boundary : region.storageBoundary) { addDomain(boundary.cell.space, boundary.cell.base); }
    if (region.symbolicStorage) {
        if (region.symbolicStorage->accessModel != &input.accesses() ||
            region.symbolicStorage->gmAliasPolicy != region.gmAliasPolicy ||
            region.symbolicStorage->expressions != region.expressions) {
            return finish(fail("regional symbolic storage provenance differs"));
        }
        for (const auto& family : region.symbolicStorage->families) { addDomain(family.space, family.base); }
    } else if (!region.symbolicStorageEffects.empty()) {
        return finish(fail("regional symbolic storage lacks complete family domains"));
    }
    for (const auto& domain : domains) {
        const auto space = domain.first;
        const Value base = domain.second;
        auto byte = variable();
        std::optional<RegionalStorageSelectors> boundary;
        if (region.storageSelectors) { boundary = region.storageSelectors({space, base, byte}); }
        else if (region.symbolicStorageEffects.empty()) {
            boundary.emplace();
            for (const auto& finite : region.storageBoundary) {
                if (finite.cell.space != space || finite.cell.base != base) { continue; }
                const auto active = e.land(e.le(c(finite.cell.begin), byte), e.lt(byte, c(finite.cell.end)));
                auto copy = [&](auto& to, const auto& from) {
                    for (auto selected : from) {
                        selected.present = e.land(active, selected.present); to.push_back(std::move(selected));
                    }
                };
                copy(boundary->firstWriters, finite.firstWriters); copy(boundary->lastWriters, finite.lastWriters);
                for (const auto& [pipe, selected] : finite.firstReaders) {
                    copy(boundary->firstReaders[pipe], selected);
                }
                for (const auto& [pipe, selected] : finite.lastReaders) { copy(boundary->lastReaders[pipe], selected); }
            }
        }
        if (!boundary) { return finish(fail("regional byte-selector callback unavailable on symbolic coordinates")); }
        auto add = [&](Kind kind, std::optional<uint32_t> pipe, const auto& selected) {
            for (const auto& choice : selected) {
                if (!addSelector(kind, space, base, pipe, choice, byte)) { return false; }
            }
            return true;
        };
        if (!add(Kind::FirstWriter, {}, boundary->firstWriters) || !add(Kind::LastWriter, {}, boundary->lastWriters)) {
            return finish(false);
        }
        for (const auto& [pipe, selected] : boundary->firstReaders) {
            if (!add(Kind::FirstReaderBeforeWrite, pipe, selected)) { return finish(false); }
        }
        for (const auto& [pipe, selected] : boundary->lastReaders) {
            if (!add(Kind::LastReaderAfterWrite, pipe, selected)) { return finish(false); }
        }
    }
    for (const auto* list : {&region.firstPayloads, &region.lastPayloads}) {
        for (const auto& [pipe, selected] : *list) {
            for (const auto& choice : selected) {
                if (!addSelector(list == &region.firstPayloads ? Kind::FirstPayload : Kind::LastPayload,
                    {}, {}, pipe, choice)) { return finish(false); }
            }
        }
    }
    std::set<std::pair<std::size_t, Kind>> siteExtrema;
    for (const auto* list : {&region.accessBoundary, &region.deferredAccessBoundary}) {
        for (const auto& access : *list) {
            for (auto kind : {Kind::FirstSite, Kind::LastSite}) {
                const auto& choice = kind == Kind::FirstSite ? access.first : access.last;
                if (siteExtrema.insert({choice.event.type, kind}).second &&
                    !addSelector(kind, {}, {}, {}, choice)) { return finish(false); }
            }
            if (list == &region.deferredAccessBoundary) { data.dischargedEffects.push_back(access.effect); }
        }
    }
    return finish(true);
}
FailureOr<RegionalRelationData> Request::lower(ArrayRef<Value> parameters, std::string& diagnostic)
{
    const unsigned p = parameters.size();
    data.parameterValues.assign(parameters.begin(), parameters.end());
    for (auto value : parameters) { data.parameters.push_back(e.input(value)); }
    data.analysis.period = 1; data.analysis.parameterCount = p; data.completeRequiredOrder = true;
    data.selectors.period = 1; data.selectors.parameterCount = p;
    std::set<uint32_t> pipes;
    for (const auto& site : data.sites) { pipes.insert(static_cast<uint32_t>(site.phase->kPipeValue)); }
    data.analysis.pipeCount = pipes.size();
    auto translate = [&](Id predicate, std::vector<Id> values) -> FailureOr<std::vector<IntegerSystem>> {
        auto domain = IntegerSystem::create(values.size(), {});
        if (failed(domain)) { return failure(); }
        RegionExpressions::RelationCost cost;
        auto result = e.integerRelation(predicate, values, *domain, diagnostic, &cost);
        for (auto member : {&RegionExpressions::RelationCost::gates, &RegionExpressions::RelationCost::pieces,
            &RegionExpressions::RelationCost::projections, &RegionExpressions::RelationCost::formulaProducts}) {
            if (cost.*member > UINT64_MAX - data.translationCost.*member) {
                diagnostic = "regional relation translation cost exceeds representation"; return failure();
            }
            data.translationCost.*member += cost.*member;
        }
        data.translationCost.peakClauses = std::max(data.translationCost.peakClauses, cost.peakClauses);
        return result;
    };
    for (const auto& formula : formulas) {
        auto values = formula.coordinates; llvm::append_range(values, data.parameters);
        auto systems = translate(formula.predicate, values);
        if (failed(systems)) { return failure(); }
        for (auto& system : *systems) {
            if (formula.occurrence) {
                data.occurrences.push_back({*formula.occurrence, std::vector<uint64_t>(formula.coordinates.size(), 0),
                    std::vector<uint64_t>(p, 0), std::move(system)});
            } else {
                auto key = *formula.relation; key.parameterResidues.assign(p, 0);
                auto& relation = formula.native ? data.analysis.nativeOrder : data.analysis.requiredOrder;
                relation[key].push_back(std::move(system));
            }
        }
    }
    for (const auto& formula : selectors) {
        auto values = formula.inputs; llvm::append_range(values, data.parameters);
        const unsigned inputs = values.size(); llvm::append_range(values, formula.outputs);
        auto systems = translate(formula.predicate, values);
        if (failed(systems)) { return failure(); }
        auto boundary = formula.boundary;
        boundary.selector.inputDimensions = formula.inputs.size(); boundary.selector.parameterCount = p;
        for (const auto& system : *systems) {
            auto witnesses = buildIntegerTupleWitnesses(
                system, inputs, std::vector<uint64_t>(formula.outputs.size(), 0));
            if (failed(witnesses)) {
                diagnostic = "regional functional storage-selector witness export failed"; return failure();
            }
            for (auto& witness : *witnesses) {
                if (boundary.storageSpace) {
                    data.selectors.support.push_back({*boundary.storageSpace, boundary.storageBase, 0,
                        std::vector<uint64_t>(p, 0), witness.domain});
                }
                boundary.selector.pieces.push_back({std::move(witness.domain),
                    std::vector<uint64_t>(formula.inputs.size(), 0), std::vector<uint64_t>(p, 0),
                    formula.site, std::move(witness.outputs)});
            }
        }
        data.selectors.boundaries.push_back(std::move(boundary));
    }
    return std::move(data);
}
} // namespace
FailureOr<std::unique_ptr<RegionalRelationRequest>> requestCallbackRegionalRelations(
    const RegionalAnalysis& region, func::FuncOp function, const SyncInput& input,
    const PhaseIndex& index, std::string& error)
{
    auto request = std::make_unique<Request>(region, function, input, index);
    if (!request->build(error)) { return failure(); }
    return std::unique_ptr<RegionalRelationRequest>(std::move(request));
}
} // namespace mlir::pto::frontiersynch
