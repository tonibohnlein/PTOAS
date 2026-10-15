// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Included once by DeferredFamily.cpp. Version two retains composition and typed DAGs.
#ifndef PTO_FRONTIER_DEFERREDCONTEXTUALFAMILY_H
#define PTO_FRONTIER_DEFERREDCONTEXTUALFAMILY_H
namespace {
constexpr StringLiteral sourceAttr = "pto.deferred_family_source";
SmallVector<int64_t> deferredSourcePath(Operation* operation, func::FuncOp function)
{
    SmallVector<SmallVector<int64_t>> frames;
    while (operation != function.getOperation()) {
        auto* block = operation->getBlock();
        auto* region = block->getParent();
        int64_t blockIndex = 0, operationIndex = 0;
        for (auto& candidate : *region) { if (&candidate == block) { break; } ++blockIndex; }
        for (auto& candidate : *block) {
            if (&candidate == operation) { break; }
            if (!isa<LogicalFamilyOp>(candidate)) { ++operationIndex; }
        }
        frames.push_back({static_cast<int64_t>(region->getRegionNumber()), blockIndex, operationIndex});
        operation = region->getParentOp();
    }
    SmallVector<int64_t> path;
    for (const auto& frame : llvm::reverse(frames)) { llvm::append_range(path, frame); }
    return path;
}
// Original-operation paths exclude only the archive markers. Unlike attributes
// on implicit SCF terminators, this function-owned directory survives custom printing.
Operation* deferredSourceAt(ArrayRef<int64_t> path, func::FuncOp function)
{
    if (path.size() % 3) { return nullptr; }
    Operation* operation = function.getOperation();
    for (std::size_t position = 0; position < path.size(); position += 3) {
        const auto regionId = path[position], blockId = path[position + 1], operationId = path[position + 2];
        if (regionId < 0 || blockId < 0 || operationId < 0 ||
            static_cast<uint64_t>(regionId) >= operation->getNumRegions()) { return nullptr; }
        auto& region = operation->getRegion(regionId);
        if (static_cast<uint64_t>(blockId) >= region.getBlocks().size()) { return nullptr; }
        auto block = region.begin(); std::advance(block, blockId);
        Operation* next = nullptr;
        int64_t index = 0;
        for (auto& candidate : *block) {
            if (isa<LogicalFamilyOp>(candidate)) { continue; }
            if (index++ == operationId) { next = &candidate; break; }
        }
        if (!next) { return nullptr; }
        operation = next;
    }
    return operation;
}
StringAttr contextualProtocol(DictionaryAttr data)
{
    NamedAttrList fields(data);
    fields.erase("protocol");
    return StringAttr::get(data.getContext(), metadataSeal(fields.getDictionary(data.getContext())));
}
bool qualifiedDeferredCleanup(const EndpointFamily& family)
{
    if (family.sourceChoices) { return false; }
    const auto& choices = family.targetChoices;
    if (!choices) { return true; }
    return !family.local && family.members.size() == 1 && !choices->loop() &&
        choices->pendingMaterialization() && choices->cuts().size() == 2 &&
        choices->cuts().front().before == family.targetCut.before && choices->validPendingSelection();
}
LogicalResult validateDeferredComposition(func::FuncOp function, const PreparedLogicalPlan& plan,
    ArrayRef<DeferredFamilyNode> nodes)
{
    const auto width = DataLayout::closest(function).getTypeSizeInBits(IndexType::get(function.getContext()));
    if (function->hasAttr(ActiveContextAttr) || hasPhysicalSections(function) || width.isScalable() ||
        width.getFixedValue() != 64 || plan.planId < 0 || nodes.empty() || plan.families.empty() ||
        !plan.allocationCertificate) { return failure(); }
    struct Member {
        const EndpointFamily* family;
        const EndpointFamilyMember* member;
        std::array<std::set<Operation*>, 3> seen{};
    };
    std::map<uint32_t, Member> members;
    std::set<uint32_t> familyIds;
    for (const auto& family : plan.families) {
        if (!familyIds.insert(family.id).second || family.members.empty() || !qualifiedDeferredCleanup(family) ||
            family.sourcePipe > static_cast<uint32_t>(PIPE::PIPE_FIX) ||
            family.targetPipe > static_cast<uint32_t>(PIPE::PIPE_FIX) ||
            family.sourcePipe == static_cast<uint32_t>(PIPE::PIPE_ALL) ||
            family.targetPipe == static_cast<uint32_t>(PIPE::PIPE_ALL) ||
            family.local != (family.sourcePipe == family.targetPipe)) { return failure(); }
        for (const auto& member : family.members) {
            if (!members.emplace(member.record, Member{&family, &member}).second) { return failure(); }
        }
    }
    DominanceInfo dominance(function);
    SmallVector<std::pair<std::size_t, uint32_t>> pending{{nodes.size() - 1, 0}};
    while (!pending.empty()) {
        auto [position, base] = pending.pop_back_val();
        if (position >= nodes.size()) { return failure(); }
        const auto& node = nodes[position];
        if (node.kind != DeferredFamilyNode::Kind::Leaf) {
            if (node.child < 0 || static_cast<uint64_t>(node.child) >= position ||
                node.recordBase > UINT32_MAX - base) {
                return failure();
            }
            pending.push_back({static_cast<std::size_t>(node.child), base + node.recordBase});
            if (node.previous != -1) {
                if (node.previous < 0 || static_cast<uint64_t>(node.previous) >= position) { return failure(); }
                pending.push_back({static_cast<std::size_t>(node.previous), base});
            }
            continue;
        }
        if (node.pieces.size() != node.numerical.size()) { return failure(); }
        if (node.outer && (node.outer->getParentOfType<func::FuncOp>() != function || !boundedLoop(node.outer))) {
            return failure();
        }
        for (auto [i, piece] : llvm::enumerate(node.pieces)) {
            auto* cut = piece.cut.before;
            if (!cut || cut->getParentOfType<func::FuncOp>() != function || piece.cut.block != cut->getBlock() ||
                piece.members.empty() || piece.displacement > INT64_MAX || piece.order > INT64_MAX ||
                piece.namespaceId > UINT32_MAX - base ||
                (node.numerical[i] && ((!node.outer && piece.displacement) ||
                    (node.outer && !node.outer->isProperAncestor(cut))))) { return failure(); }
            for (const auto& member : piece.members) {
                if (member.record > UINT32_MAX - base) { return failure(); }
                auto found = members.find(member.record + base);
                if (found == members.end()) { return failure(); }
                auto& entry = found->second;
                const bool local = piece.kind == EndpointKind::Barrier, publish = piece.kind == EndpointKind::Set;
                const unsigned kind = publish ? 0 : local ? 1 : 2;
                const auto& family = *entry.family;
                const auto expected = publish ? family.sourceCut : family.targetCut;
                const auto& coordinates = publish ? entry.member->sourceCoordinates : entry.member->targetCoordinates;
                const bool cleanup = !publish && !local && family.targetChoices &&
                    cut == family.targetChoices->cuts().back().before;
                if (!entry.seen[kind].insert(cut).second || local != family.local ||
                    (!cleanup && cut != expected.before) || piece.sourcePipe != family.sourcePipe ||
                    piece.targetPipe != family.targetPipe || piece.displacement != family.displacement ||
                    (cleanup ? !member.coordinates.empty() : !sameCoordinates(coordinates, member.coordinates))) {
                    return failure();
                }
                for (const auto& coordinate : member.coordinates) {
                    auto loop = coordinate.loop;
                    if (!boundedLoop(loop) ||
                        !dominance.properlyDominates(loop.getInductionVar(), cut)) { return failure(); }
                }
            }
        }
    }
    for (const auto& [id, member] : members) {
        (void)id;
        const auto& family = *member.family;
        for (unsigned kind = 0; kind < 3; ++kind) {
            std::set<Operation*> expected;
            const bool required = family.local ? kind == 1 : kind != 1;
            if (required) {
                if (kind == 2 && family.targetChoices) {
                    for (auto cut : family.targetChoices->cuts()) { expected.insert(cut.before); }
                } else { expected.insert(kind == 0 ? family.sourceCut.before : family.targetCut.before); }
            }
            if (member.seen[kind] != expected) { return failure(); }
        }
    }
    return success();
}
struct ContextualEncoder {
    Encoder base;
    func::FuncOp function;
    std::map<Operation*, int64_t> sources;
    struct Arena {
        std::shared_ptr<const RegionExpressions> owner;
        std::map<RegionExpressions::Id, int64_t> roots;
    };
    std::vector<Arena> arenas;
    std::map<const RegionExpressions*, int64_t> arenaIds;
    explicit ContextualEncoder(func::FuncOp function) : base(function.getContext()), function(function) {}
    int64_t source(Operation* operation)
    {
        if (!operation || (operation != function.getOperation() &&
            operation->getParentOfType<func::FuncOp>() != function)) { return -1; }
        return sources.emplace(operation, sources.size()).first->second;
    }
    Attribute value(Value value)
    {
        SmallVector<int64_t> reference;
        if (auto result = dyn_cast<OpResult>(value)) {
            const auto id = source(result.getOwner());
            if (id < 0) { return {}; }
            reference = {id, -1, -1, result.getResultNumber()};
        } else if (auto argument = dyn_cast<BlockArgument>(value)) {
            auto* block = argument.getOwner(); auto* region = block->getParent();
            if (!region) { return {}; }
            const auto id = source(region->getParentOp());
            if (id < 0) { return {}; }
            int64_t blockIndex = 0;
            for (auto& candidate : *region) { if (&candidate == block) { break; } ++blockIndex; }
            reference = {id, static_cast<int64_t>(region->getRegionNumber()), blockIndex, argument.getArgNumber()};
        } else { return {}; }
        return base.builder.getDictionaryAttr({
            base.builder.getNamedAttr("reference", base.builder.getDenseI64ArrayAttr(reference)),
            base.builder.getNamedAttr("type", TypeAttr::get(value.getType()))});
    }
    int64_t arena(const std::shared_ptr<const RegionExpressions>& owner)
    {
        if (!owner) { return -1; }
        auto [found, added] = arenaIds.emplace(owner.get(), arenas.size());
        if (added) { arenas.push_back({owner, {}}); }
        return found->second;
    }
    int64_t root(int64_t arenaId, RegionExpressions::Id id)
    {
        if (id == RegionExpressions::invalid) { return -1; }
        if (arenaId < 0 || id >= arenas[arenaId].owner->size()) { return -2; }
        auto& roots = arenas[arenaId].roots;
        return roots.emplace(id, roots.size()).first->second;
    }
    DictionaryAttr node(const DeferredFamilyNode& value, std::size_t position,
        std::map<Operation*, SmallVector<Attribute>>& markers)
    {
        auto& b = base.builder;
        const auto owner = arena(value.arena);
        SmallVector<Attribute> pieces, bindings, shared, contexts;
        for (auto [i, piece] : llvm::enumerate(value.pieces)) {
            base.cuts.emplace(piece.cut.before, base.cuts.size());
            pieces.push_back(b.getDictionaryAttr({base.number("cut", base.cuts.at(piece.cut.before)),
                b.getNamedAttr("piece", base.piece(piece))}));
            markers[piece.cut.before].push_back(b.getDictionaryAttr({
                base.number("node", position), base.number("piece", i)}));
            const auto& binding = value.bindings[i];
            SmallVector<int64_t> coordinates;
            for (auto coordinate : binding.matchingCoordinates) { coordinates.push_back(root(owner, coordinate)); }
            bindings.push_back(b.getDictionaryAttr({base.number("activity", root(owner, binding.activity)),
                base.number("ordinal", root(owner, binding.ordinal)),
                base.number("coordinate", root(owner, binding.residueCoordinate)),
                base.number("modulus", binding.modulus), base.number("residue", binding.residue),
                b.getNamedAttr("matching", b.getDenseI64ArrayAttr(coordinates))}));
        }
        for (const auto& entry : value.shared) {
            shared.push_back(b.getDenseI64ArrayAttr({source(entry.cut), root(owner, entry.expression)}));
        }
        for (const auto& [cut, expression] : value.contextCuts) {
            contexts.push_back(b.getDenseI64ArrayAttr({source(cut), root(owner, expression)}));
        }
        SmallVector<int64_t> numerical(value.numerical.begin(), value.numerical.end());
        return b.getDictionaryAttr({base.number("kind", static_cast<int64_t>(value.kind)),
            base.number("previous", value.previous), base.number("child", value.child),
            base.number("base", value.recordBase),
            base.number("arena", owner), base.number("outer", base.loop(value.outer)),
            b.getNamedAttr("pieces", b.getArrayAttr(pieces)), b.getNamedAttr("bindings", b.getArrayAttr(bindings)),
            b.getNamedAttr("numerical", b.getDenseI64ArrayAttr(numerical)),
            b.getNamedAttr("shared", b.getArrayAttr(shared)), b.getNamedAttr("contexts", b.getArrayAttr(contexts))});
    }
    FailureOr<ArrayAttr> cleanupChoices(ArrayRef<EndpointFamily> families)
    {
        SmallVector<Attribute> result;
        auto& b = base.builder;
        for (const auto& family : families) {
            if (!qualifiedDeferredCleanup(family)) { return failure(); }
            if (!family.targetChoices) { continue; }
            auto selection = family.targetChoices->pendingSelection();
            if (!selection) { return failure(); }
            SmallVector<int64_t> cuts;
            for (auto cut : family.targetChoices->cuts()) {
                auto found = base.cuts.find(cut.before);
                if (found == base.cuts.end()) { return failure(); }
                cuts.push_back(found->second);
            }
            const auto owner = arena(selection->arena), predicate = root(owner, selection->expression);
            if (owner < 0 || predicate < 0) { return failure(); }
            result.push_back(b.getDictionaryAttr({base.number("family", family.id),
                base.number("arena", owner), base.number("predicate", predicate),
                b.getNamedAttr("cuts", b.getDenseI64ArrayAttr(cuts))}));
        }
        return b.getArrayAttr(result);
    }
    FailureOr<ArrayAttr> graphs()
    {
        SmallVector<Attribute> result;
        auto& b = base.builder;
        for (const auto& arena : arenas) {
            SmallVector<RegionExpressions::Id> roots(arena.roots.size());
            for (auto [id, position] : arena.roots) { roots[position] = id; }
            auto graph = arena.owner->archive(roots, function.getContext(), [&](Value input) { return value(input); });
            if (failed(graph)) { return failure(); }
            SmallVector<int64_t> prohibited;
            for (auto* operation : arena.owner->prohibitedRecomputation()) {
                const auto id = source(operation);
                if (id < 0) { return failure(); }
                prohibited.push_back(id);
            }
            llvm::sort(prohibited);
            result.push_back(b.getDictionaryAttr({b.getNamedAttr("dag", *graph),
                b.getNamedAttr("prohibited", b.getDenseI64ArrayAttr(prohibited))}));
        }
        return b.getArrayAttr(result);
    }
};
LogicalResult insertContextualDeferredFamily(func::FuncOp function, PreparedLogicalPlan& plan)
{
    auto nodes = deferredFamilyRecipe(plan);
    if (failed(nodes) || failed(validateDeferredComposition(function, plan, *nodes))) {
        return function.emitError("deferred contextual output requires owned family recipes, "
            "exact provenance and allocation "
            "evidence; concrete endpoints, private state and unowned or nonterminal choices are not serializable");
    }
    bool occupied = function->hasAttr(recipeAttr) || function->hasAttr("pto.endpoint_families") ||
        function->hasAttr(CyclicAllocationAttr) || function->hasAttr(FiniteAllocationAttr);
    function.walk([&](Operation* op) {
        occupied |= op->hasAttr(cutAttr) || op->hasAttr(loopAttr) || op->hasAttr(sourceAttr) ||
            isa<LogicalFamilyOp, LogicalSetOp, LogicalWaitOp>(op);
    });
    if (occupied) { return function.emitError("deferred output cannot replace existing synchronization bindings"); }
    ContextualEncoder encoder(function);
    auto& b = encoder.base.builder;
    SmallVector<Attribute> recipes, families;
    std::map<Operation*, SmallVector<Attribute>> markers;
    for (auto [i, node] : llvm::enumerate(*nodes)) { recipes.push_back(encoder.node(node, i, markers)); }
    for (const auto& family : plan.families) { families.push_back(encoder.base.family(family)); }
    SmallVector<Attribute> localSources;
    for (const auto& family : plan.families) {
        if (!family.local) { continue; }
        auto* source = family.sourceCut.before;
        if (!source || source->getParentOfType<func::FuncOp>() != function ||
            family.sourceCut.block != source->getBlock()) {
            return function.emitError("deferred local family lacks its original source provenance cut");
        }
        localSources.push_back(b.getDenseI64ArrayAttr({family.id, encoder.source(source)}));
    }
    auto choices = encoder.cleanupChoices(plan.families);
    if (failed(choices)) { return function.emitError("deferred cleanup lacks owned selection or exact original cuts"); }
    auto graphs = encoder.graphs();
    if (failed(graphs)) {
        return function.emitError("deferred expression has an unowned input or captured evaluation");
    }
    SmallVector<Attribute> sourceBindings(encoder.sources.size());
    for (const auto& [operation, id] : encoder.sources) {
        sourceBindings[id] = b.getDictionaryAttr({
            b.getNamedAttr("path", b.getDenseI64ArrayAttr(deferredSourcePath(operation, function))),
            b.getNamedAttr("name", b.getStringAttr(operation->getName().getStringRef()))});
    }
    SmallVector<Attribute> cutBindings(markers.size());
    for (const auto& [cut, pieces] : markers) {
        SmallVector<int64_t> axes;
        for (const auto& [operation, id] : encoder.base.loops) {
            if (operation->isProperAncestor(cut)) { axes.push_back(id); }
        }
        llvm::sort(axes);
        cutBindings[encoder.base.cuts.at(cut)] = b.getDictionaryAttr({
            b.getNamedAttr("path", b.getDenseI64ArrayAttr(deferredSourcePath(cut, function))),
            b.getNamedAttr("name", b.getStringAttr(cut->getName().getStringRef())),
            b.getNamedAttr("pieces", b.getArrayAttr(pieces)),
            b.getNamedAttr("axis_ids", b.getDenseI64ArrayAttr(axes))});
    }
    auto metadata = b.getDictionaryAttr({encoder.base.number("version", choices->empty() ? 2 : 3),
        encoder.base.number("plan", plan.planId),
        encoder.base.number("cut_count", markers.size()), encoder.base.number("source_count", encoder.sources.size()),
        b.getNamedAttr("complete", b.getBoolAttr(plan.completeInvocation)),
        b.getNamedAttr("nested", b.getBoolAttr(plan.nestedIdentities)),
        b.getNamedAttr("grouped", b.getBoolAttr(plan.groupedFamilies)),
        b.getNamedAttr("independent", b.getBoolAttr(plan.independentPieces)),
        b.getNamedAttr("families", b.getArrayAttr(families)), b.getNamedAttr("allocation", plan.allocationCertificate),
        b.getNamedAttr("nodes", b.getArrayAttr(recipes)), b.getNamedAttr("arenas", *graphs),
        b.getNamedAttr("sources", b.getArrayAttr(sourceBindings)),
        b.getNamedAttr("cuts", b.getArrayAttr(cutBindings)),
        b.getNamedAttr("local_sources", b.getArrayAttr(localSources))});
    if (!choices->empty()) {
        NamedAttrList extended(metadata);
        extended.set("cleanup_choices", *choices);
        metadata = extended.getDictionary(function.getContext());
    }
    NamedAttrList sealed(metadata);
    sealed.set("protocol", contextualProtocol(metadata));
    metadata = sealed.getDictionary(function.getContext());
    for (const auto& [operation, id] : encoder.base.loops) { operation->setAttr(loopAttr, b.getI64IntegerAttr(id)); }
    for (const auto& [cut, pieces] : markers) {
        SmallVector<std::pair<int64_t, Value>> loopValues;
        for (const auto& [operation, id] : encoder.base.loops) {
            if (operation->isProperAncestor(cut)) {
                loopValues.push_back({id, cast<scf::ForOp>(operation).getInductionVar()});
            }
        }
        llvm::sort(loopValues, [](const auto& a, const auto& c) { return a.first < c.first; });
        SmallVector<int64_t> ids; SmallVector<Value> values;
        for (auto [id, value] : loopValues) { ids.push_back(id); values.push_back(value); }
        auto encodedPieces = b.getArrayAttr(pieces); auto axes = b.getDenseI64ArrayAttr(ids);
        OpBuilder builder(cut);
        OperationState state(cut->getLoc(), LogicalFamilyOp::getOperationName());
        state.addOperands(values); state.addAttribute("plan_id", b.getI64IntegerAttr(plan.planId));
        state.addAttribute("cut_id", b.getI64IntegerAttr(encoder.base.cuts.at(cut)));
        state.addAttribute("axis_ids", axes); state.addAttribute("pieces", encodedPieces);
        builder.create(state);
    }
    function->setAttr(recipeAttr, metadata);
    if (plan.recognitionReport) { function->setAttr("pto.frontier_recognition", plan.recognitionReport); }
    return success();
}
struct ContextualDecoder {
    Decoder base;
    std::map<int64_t, Operation*> sources;
    struct Arena { std::shared_ptr<RegionExpressions> owner; SmallVector<RegionExpressions::Id> roots; };
    std::vector<Arena> arenas;
    explicit ContextualDecoder(func::FuncOp function) : base{function, {}, {}} {}
    LogicalResult cleanupChoices(ArrayAttr choices, PreparedLogicalPlan& plan)
    {
        if (!choices || choices.empty()) { return failure(); }
        std::map<uint32_t, EndpointFamily*> families;
        for (auto& family : plan.families) {
            if (!families.emplace(family.id, &family).second) { return failure(); }
        }
        std::set<uint32_t> seen;
        for (auto raw : choices) {
            auto data = dyn_cast<DictionaryAttr>(raw);
            const auto family = integer(data, "family"), owner = integer(data, "arena");
            const auto predicate = integer(data, "predicate");
            auto cuts = data ? data.getAs<DenseI64ArrayAttr>("cuts") : DenseI64ArrayAttr{};
            if (!fields(data, {"family", "arena", "predicate", "cuts"}) || !nonnegative(family, UINT32_MAX) ||
                !nonnegative(owner) || !nonnegative(predicate) || !cuts || cuts.size() != 2 ||
                !families.count(*family) || !seen.insert(*family).second ||
                static_cast<uint64_t>(*owner) >= arenas.size() ||
                static_cast<uint64_t>(*predicate) >= arenas[*owner].roots.size()) { return failure(); }
            SmallVector<TemplateEndpointCut> alternatives;
            for (auto id : cuts.asArrayRef()) {
                auto found = base.cuts.find(id);
                if (found == base.cuts.end()) { return failure(); }
                alternatives.push_back({found->second->getBlock(), found->second});
            }
            auto& target = *families.at(*family);
            target.targetChoices = qualifyTerminalCleanupCuts(alternatives,
                arenas[*owner].owner, arenas[*owner].roots[*predicate]);
            if (!target.targetChoices || !qualifiedDeferredCleanup(target)) { return failure(); }
        }
        return success();
    }
    Value value(Attribute raw)
    {
        auto data = dyn_cast_or_null<DictionaryAttr>(raw);
        auto reference = data ? data.getAs<DenseI64ArrayAttr>("reference") : DenseI64ArrayAttr{};
        auto type = data ? data.getAs<TypeAttr>("type") : TypeAttr{};
        if (!fields(data, {"reference", "type"}) || !reference || reference.size() != 4 || !type ||
            !sources.count(reference[0]) || reference[3] < 0) { return {}; }
        auto* operation = sources.at(reference[0]);
        Value result;
        if (reference[1] == -1 && reference[2] == -1) {
            if (static_cast<uint64_t>(reference[3]) >= operation->getNumResults()) { return {}; }
            result = operation->getResult(reference[3]);
        } else {
            if (reference[1] < 0 || reference[2] < 0 ||
                static_cast<uint64_t>(reference[1]) >= operation->getNumRegions()) { return {}; }
            auto& region = operation->getRegion(reference[1]);
            if (static_cast<uint64_t>(reference[2]) >= region.getBlocks().size()) { return {}; }
            auto block = region.begin(); std::advance(block, reference[2]);
            if (static_cast<uint64_t>(reference[3]) >= block->getNumArguments()) { return {}; }
            result = block->getArgument(reference[3]);
        }
        return result.getType() == type.getValue() ? result : Value{};
    }
    LogicalResult graphs(ArrayAttr encoded)
    {
        for (auto raw : encoded) {
            auto data = dyn_cast<DictionaryAttr>(raw);
            auto graph = data ? data.getAs<DictionaryAttr>("dag") : DictionaryAttr{};
            auto prohibited = data ? data.getAs<DenseI64ArrayAttr>("prohibited") : DenseI64ArrayAttr{};
            if (!fields(data, {"dag", "prohibited"}) || !graph || !prohibited) { return failure(); }
            auto arena = std::make_shared<RegionExpressions>();
            for (int64_t id : prohibited.asArrayRef()) {
                if (!sources.count(id)) { return failure(); }
                arena->forbidRecomputation(sources.at(id));
            }
            auto roots = arena->restoreArchive(graph, [&](Attribute input) { return value(input); });
            if (failed(roots)) { return failure(); }
            arenas.push_back({std::move(arena), std::move(*roots)});
        }
        return success();
    }
    FailureOr<DeferredFamilyNode> node(DictionaryAttr data)
    {
        auto kind = integer(data, "kind"), previous = integer(data, "previous"), child = integer(data, "child");
        auto recordBase = integer(data, "base"), arenaId = integer(data, "arena"), outer = integer(data, "outer");
        auto pieces = data ? data.getAs<ArrayAttr>("pieces") : ArrayAttr{};
        auto bindings = data ? data.getAs<ArrayAttr>("bindings") : ArrayAttr{};
        auto numerical = data ? data.getAs<DenseI64ArrayAttr>("numerical") : DenseI64ArrayAttr{};
        auto shared = data ? data.getAs<ArrayAttr>("shared") : ArrayAttr{};
        auto contexts = data ? data.getAs<ArrayAttr>("contexts") : ArrayAttr{};
        if (!fields(data, {"kind", "previous", "child", "base", "arena", "outer", "pieces", "bindings",
                "numerical", "shared", "contexts"}) || !nonnegative(kind, 3) || !previous || *previous < -1 ||
            !child || *child < -1 || !nonnegative(recordBase, UINT32_MAX) || !arenaId || *arenaId < -1 ||
            (*arenaId != -1 && static_cast<uint64_t>(*arenaId) >= arenas.size()) || !outer || *outer < -1 ||
            (*outer != -1 && !base.loops.count(*outer)) || !pieces || !bindings || !numerical || !shared || !contexts ||
            pieces.size() != bindings.size() ||
            pieces.size() != static_cast<std::size_t>(numerical.size())) { return failure(); }
        DeferredFamilyNode result;
        result.kind = static_cast<DeferredFamilyNode::Kind>(*kind); result.previous = *previous; result.child = *child;
        result.recordBase = *recordBase;
        if (*outer != -1) { result.outer = base.loops.at(*outer); }
        if (*arenaId != -1) { result.arena = arenas[*arenaId].owner; }
        auto root = [&](int64_t index) -> std::optional<RegionExpressions::Id> {
            if (index == -1) { return RegionExpressions::invalid; }
            if (index < 0 || *arenaId < 0 || static_cast<uint64_t>(index) >= arenas[*arenaId].roots.size()) {
                return {};
            }
            return arenas[*arenaId].roots[index];
        };
        for (auto [i, raw] : llvm::enumerate(pieces)) {
            auto entry = dyn_cast<DictionaryAttr>(raw); auto cut = integer(entry, "cut");
            auto pieceData = entry ? entry.getAs<DictionaryAttr>("piece") : DictionaryAttr{};
            if (!fields(entry, {"cut", "piece"}) || !cut || !base.cuts.count(*cut) || !pieceData ||
                numerical[i] < 0 || numerical[i] > 1) { return failure(); }
            auto piece = base.piece(pieceData, base.cuts.at(*cut));
            if (failed(piece)) { return failure(); }
            result.pieces.push_back(*piece); result.numerical.push_back(numerical[i]);
            auto binding = dyn_cast<DictionaryAttr>(bindings[i]);
            auto activity = integer(binding, "activity"), ordinal = integer(binding, "ordinal");
            auto coordinate = integer(binding, "coordinate"), modulus = integer(binding, "modulus");
            auto residue = integer(binding, "residue");
            auto matching = binding ? binding.getAs<DenseI64ArrayAttr>("matching") : DenseI64ArrayAttr{};
            if (!fields(binding, {"activity", "ordinal", "coordinate", "modulus", "residue", "matching"}) ||
                !activity || !ordinal || !coordinate || !modulus || !residue || !matching) { return failure(); }
            auto activeRoot = root(*activity), ordinalRoot = root(*ordinal), coordinateRoot = root(*coordinate);
            if (!activeRoot || !ordinalRoot || !coordinateRoot) { return failure(); }
            ContextualEndpointFamily family;
            family.piece = *piece; family.activity = *activeRoot; family.ordinal = *ordinalRoot;
            family.residueCoordinate = *coordinateRoot; family.modulus = static_cast<uint64_t>(*modulus);
            family.residue = static_cast<uint64_t>(*residue);
            for (auto index : matching.asArrayRef()) {
                auto id = root(index);
                if (!id || *id == RegionExpressions::invalid) { return failure(); }
                family.matchingCoordinates.push_back(*id);
            }
            if (numerical[i] && (family.activity != RegionExpressions::invalid ||
                family.ordinal != RegionExpressions::invalid ||
                family.residueCoordinate != RegionExpressions::invalid || !family.matchingCoordinates.empty() ||
                family.modulus || family.residue)) { return failure(); }
            result.bindings.push_back(std::move(family));
        }
        for (auto raw : shared) {
            auto pair = dyn_cast<DenseI64ArrayAttr>(raw);
            if (!pair || pair.size() != 2 || !sources.count(pair[0])) { return failure(); }
            auto id = root(pair[1]);
            if (!id || *id == RegionExpressions::invalid) { return failure(); }
            result.shared.push_back({*id, sources.at(pair[0])});
        }
        for (auto raw : contexts) {
            auto pair = dyn_cast<DenseI64ArrayAttr>(raw);
            if (!pair || pair.size() != 2 || !sources.count(pair[0])) { return failure(); }
            auto id = root(pair[1]);
            if (!id || *id == RegionExpressions::invalid ||
                !result.contextCuts.emplace(sources.at(pair[0]), *id).second) {
                return failure();
            }
        }
        return result;
    }
};
FailureOr<std::unique_ptr<PreparedLogicalPlan>> readContextualDeferredFamily(func::FuncOp function, DictionaryAttr data)
{
    auto planId = integer(data, "plan"), cutCount = integer(data, "cut_count");
    auto sourceCount = integer(data, "source_count");
    auto complete = data.getAs<BoolAttr>("complete"), nested = data.getAs<BoolAttr>("nested");
    auto grouped = data.getAs<BoolAttr>("grouped"), independent = data.getAs<BoolAttr>("independent");
    auto families = data.getAs<ArrayAttr>("families"), nodes = data.getAs<ArrayAttr>("nodes");
    auto arenas = data.getAs<ArrayAttr>("arenas");
    auto allocation = data.getAs<DictionaryAttr>("allocation");
    auto sources = data.getAs<ArrayAttr>("sources"), cuts = data.getAs<ArrayAttr>("cuts");
    auto localSources = data.getAs<ArrayAttr>("local_sources");
    const auto version = integer(data, "version");
    SmallVector<StringRef> schema{"version", "plan", "cut_count", "source_count", "complete", "nested", "grouped",
        "independent", "families", "allocation", "nodes", "arenas", "sources", "cuts", "local_sources", "protocol"};
    if (version == 3) { schema.push_back("cleanup_choices"); }
    if (!fields(data, schema) || (version != 2 && version != 3) ||
        !nonnegative(planId) || !nonnegative(cutCount) || !nonnegative(sourceCount) ||
        !complete || !nested || !grouped ||
        !independent || !families || !nodes || nodes.empty() || !arenas || !allocation || !sources || !cuts ||
        !localSources ||
        sources.size() != static_cast<uint64_t>(*sourceCount) || cuts.size() != static_cast<uint64_t>(*cutCount) ||
        data.getAs<StringAttr>("protocol") != contextualProtocol(data)) {
        function.emitError("malformed contextual family archive"); return failure();
    }
    ContextualDecoder decoder(function);
    bool invalid = false;
    std::map<int64_t, Operation*> markers;
    std::set<Operation*> sourceOperations, cutOperations;
    for (auto [id, raw] : llvm::enumerate(sources)) {
        auto source = dyn_cast<DictionaryAttr>(raw);
        auto path = source ? source.getAs<DenseI64ArrayAttr>("path") : DenseI64ArrayAttr{};
        auto name = source ? source.getAs<StringAttr>("name") : StringAttr{};
        auto* operation = path ? deferredSourceAt(path.asArrayRef(), function) : nullptr;
        if (!fields(source, {"path", "name"}) || !operation || !name ||
            name.getValue() != operation->getName().getStringRef() ||
            !sourceOperations.insert(operation).second) { invalid = true; }
        else { decoder.sources.emplace(id, operation); }
    }
    for (auto [id, raw] : llvm::enumerate(cuts)) {
        auto cut = dyn_cast<DictionaryAttr>(raw);
        auto path = cut ? cut.getAs<DenseI64ArrayAttr>("path") : DenseI64ArrayAttr{};
        auto name = cut ? cut.getAs<StringAttr>("name") : StringAttr{};
        auto* operation = path ? deferredSourceAt(path.asArrayRef(), function) : nullptr;
        if (!fields(cut, {"path", "name", "pieces", "axis_ids"}) || !operation || !name ||
            operation == function.getOperation() || name.getValue() != operation->getName().getStringRef() ||
            !cut.getAs<ArrayAttr>("pieces") || !cut.getAs<DenseI64ArrayAttr>("axis_ids") ||
            !cutOperations.insert(operation).second) { invalid = true; }
        else { decoder.base.cuts.emplace(id, operation); }
    }
    function.walk([&](Operation* op) {
        // Contextual archives own all bindings in the sealed directories. Old,
        // orphaned or injected per-operation bindings are never silently accepted.
        if (op->hasAttr(sourceAttr) || op->hasAttr(cutAttr)) { invalid = true; }
        if (auto raw = op->getAttr(loopAttr)) {
            auto id = dyn_cast<IntegerAttr>(raw); auto loop = dyn_cast<scf::ForOp>(op);
            if (!id || !id.getType().isInteger(64) || id.getInt() < 0 || !loop ||
                !decoder.base.loops.emplace(id.getInt(), loop).second) { invalid = true; }
        }
        if (isa<LogicalFamilyOp>(op)) {
            auto id = integer(op->getAttrDictionary(), "cut_id"), owner = integer(op->getAttrDictionary(), "plan_id");
            if (!fields(op->getAttrDictionary(), {"cut_id", "plan_id", "pieces", "axis_ids"}) || !nonnegative(id) ||
                owner != planId || !decoder.base.cuts.count(*id) || !markers.emplace(*id, op).second ||
                failed(verify(op)) || op->getNextNode() != decoder.base.cuts.at(*id)) {
                invalid = true;
            } else {
                auto cut = cast<DictionaryAttr>(cuts[*id]);
                if (op->getAttr("pieces") != cut.get("pieces") ||
                    op->getAttr("axis_ids") != cut.get("axis_ids")) { invalid = true; }
            }
        } else if (isa<LogicalSetOp, LogicalWaitOp>(op)) { invalid = true; }
    });
    if (invalid || decoder.sources.size() != static_cast<uint64_t>(*sourceCount) ||
        decoder.base.cuts.size() != static_cast<uint64_t>(*cutCount) || markers.size() != decoder.base.cuts.size() ||
        failed(decoder.graphs(arenas))) {
        function.emitError("invalid contextual source, cut or expression bindings"); return failure();
    }
    std::vector<DeferredFamilyNode> restored;
    for (auto raw : nodes) {
        auto node = decoder.node(dyn_cast<DictionaryAttr>(raw));
        if (failed(node)) { function.emitError("malformed contextual family node"); return failure(); }
        restored.push_back(std::move(*node));
    }
    std::set<std::pair<std::size_t, std::size_t>> seen;
    DominanceInfo dominance(function);
    for (auto [cutId, marker] : markers) {
        std::set<int64_t> axes;
        auto axisIds = marker->getAttrOfType<DenseI64ArrayAttr>("axis_ids");
        for (auto [id, axis] : llvm::zip(axisIds.asArrayRef(), marker->getOperands())) {
            if (!decoder.base.loops.count(id) || !axes.insert(id).second ||
                decoder.base.loops.at(id).getInductionVar() != axis || !dominance.properlyDominates(axis, marker)) {
                return failure();
            }
        }
        for (auto raw : marker->getAttrOfType<ArrayAttr>("pieces")) {
            auto entry = dyn_cast<DictionaryAttr>(raw);
            auto node = integer(entry, "node"), piece = integer(entry, "piece");
            if (!fields(entry, {"node", "piece"}) || !nonnegative(node) || !nonnegative(piece) ||
                static_cast<uint64_t>(*node) >= restored.size() ||
                static_cast<uint64_t>(*piece) >= restored[*node].pieces.size() ||
                restored[*node].pieces[*piece].cut.before != decoder.base.cuts.at(cutId) ||
                !seen.emplace(*node, *piece).second) {
                function.emitError("mismatched contextual cut marker"); return failure();
            }
            const auto& leaf = restored[*node];
            if (leaf.outer && !axes.count(leaf.outer->getAttrOfType<IntegerAttr>(loopAttr).getInt())) {
                return failure();
            }
            for (const auto& member : leaf.pieces[*piece].members) {
                for (const auto& coordinate : member.coordinates) {
                    if (!axes.count(coordinate.loop->getAttrOfType<IntegerAttr>(loopAttr).getInt())) {
                        return failure();
                    }
                }
            }
        }
    }
    std::size_t pieceCount = 0;
    for (const auto& node : restored) { pieceCount += node.pieces.size(); }
    if (seen.size() != pieceCount) { return failure(); }
    auto plan = std::make_unique<PreparedLogicalPlan>(*planId);
    plan->completeInvocation = complete.getValue(); plan->nestedIdentities = nested.getValue();
    plan->groupedFamilies = grouped.getValue(); plan->independentPieces = independent.getValue();
    plan->allocationCertificate = allocation;
    std::map<uint32_t, Operation*> localCuts;
    for (auto raw : localSources) {
        auto binding = dyn_cast<DenseI64ArrayAttr>(raw);
        if (!binding || binding.size() != 2 || binding[0] < 0 || binding[0] > UINT32_MAX ||
            !decoder.sources.count(binding[1]) ||
            !localCuts.emplace(static_cast<uint32_t>(binding[0]), decoder.sources.at(binding[1])).second) {
            function.emitError("invalid local-family source provenance bindings"); return failure();
        }
    }
    for (auto raw : families) {
        auto family = decoder.base.family(dyn_cast<DictionaryAttr>(raw));
        if (failed(family)) { return failure(); }
        auto original = localCuts.find(family->id);
        if (family->local) {
            if (original == localCuts.end() ||
                original->second->getParentOfType<func::FuncOp>() != function) {
                function.emitError("missing original local-family source cut"); return failure();
            }
            family->sourceCut = {original->second->getBlock(), original->second};
            localCuts.erase(original);
        } else if (original != localCuts.end()) { return failure(); }
        plan->families.push_back(std::move(*family));
    }
    if (!localCuts.empty()) {
        function.emitError("unused local-family source provenance bindings"); return failure();
    }
    if (version == 3 && failed(decoder.cleanupChoices(data.getAs<ArrayAttr>("cleanup_choices"), *plan))) {
        function.emitError("deferred cleanup fails exact cut or singleton-selection qualification"); return failure();
    }
    if (failed(validateDeferredComposition(function, *plan, restored))) {
        function.emitError("contextual recipe disagrees with matching provenance"); return failure();
    }
    auto recipe = restoreDeferredFamilyRecipe(restored);
    if (failed(recipe)) {
        function.emitError("contextual recipe is unavailable at its original cut"); return failure();
    }
    plan->familyRecipe = std::move(*recipe);
    if (failed(normalizeLogicalFamilies(*plan))) { return failure(); }
    return plan;
}
} // namespace
#endif
