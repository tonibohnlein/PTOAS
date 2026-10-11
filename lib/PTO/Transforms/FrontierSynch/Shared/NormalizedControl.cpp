// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// One immutable original control tree; regional views borrow selected roots.
#include "NormalizedControl.h"
namespace mlir::pto::frontiersynch {
namespace {
struct Size {
    uint64_t nodes = 1, payloads = 0, effects = 0, expansions = 0;
    bool loops = false;
};
struct LoopDomain {
    int64_t lower = 0, step = 1;
    uint64_t trips = 0;
};
uint64_t saturatedAdd(uint64_t a, uint64_t b) { return b > UINT64_MAX - a ? UINT64_MAX : a + b; }
uint64_t saturatedMultiply(uint64_t a, uint64_t b) { return b && a > UINT64_MAX / b ? UINT64_MAX : a * b; }
void appendSize(Size& total, const Size& child)
{
    total.nodes = saturatedAdd(total.nodes, child.nodes);
    total.payloads = saturatedAdd(total.payloads, child.payloads);
    total.effects = saturatedAdd(total.effects, child.effects);
    total.expansions = saturatedAdd(total.expansions, child.expansions);
    total.loops |= child.loops;
}
bool fits(const Size& size, const FiniteExpansionLimits& limits)
{
    return size.nodes <= limits.visits && size.payloads <= limits.payloads;
}
std::optional<LoopDomain> domain(scf::ForOp loop, const FiniteExpansionLimits& limits)
{
    auto lower = detail::normalizedInteger(loop.getLowerBound(), {}, loop.getContext());
    auto upper = detail::normalizedInteger(loop.getUpperBound(), {}, loop.getContext());
    auto step = detail::normalizedInteger(loop.getStep(), {}, loop.getContext());
    if (!lower || !upper || !step || *step <= 0) {
        return std::nullopt;
    }
    const __int128 distance = static_cast<__int128>(*upper) - *lower;
    const __int128 trips = distance <= 0 ? 0 : (distance + *step - 1) / *step;
    const __int128 final = static_cast<__int128>(*lower) + trips * *step;
    if (trips > limits.visits || final > INT64_MAX) {
        return std::nullopt;
    }
    return LoopDomain{*lower, *step, static_cast<uint64_t>(trips)};
}
} // namespace
struct NormalizedControlTree {
    func::FuncOp function;
    const PhaseIndex* index = nullptr;
    const SyncInput* input = nullptr;
    FiniteExpansionLimits limits;
    SmallVector<NormalizedControlNode, 0> nodes;
    DenseMap<Operation*, std::size_t> originals;
    SmallVector<Size> compactSizes, selectedSizes;
    SmallVector<std::optional<LoopDomain>> choices;
    SmallVector<std::size_t> preorder, subtreeEnd;
    uint64_t planningVisits = 0, sourceVisits = 0;
};
namespace {
// Synthetic branch arms have stable global IDs too; they need not occupy a
// contiguous slice with their descendants. All consumers follow child IDs.
void compact(NormalizedControlTree& output)
{
    struct Pending {
        Operation* operation;
        std::optional<std::size_t> parent;
    };
    SmallVector<Pending> pending{{output.function.getOperation(), {}}};
    while (!pending.empty()) {
        auto item = pending.pop_back_val();
        output.sourceVisits = saturatedAdd(output.sourceVisits, 1);
        const auto id = output.nodes.size();
        output.nodes.push_back({NormalizedControlKind::Leaf, item.operation, {}, {}});
        output.originals.try_emplace(item.operation, id);
        if (item.parent) {
            output.nodes[*item.parent].children.push_back(id);
        }
        SmallVector<std::pair<Operation*, std::size_t>> children;
        auto append = [&](Block& block, std::size_t parent) {
            for (auto& operation : block) {
                children.push_back({&operation, parent});
            }
        };
        if (auto loop = dyn_cast<scf::ForOp>(item.operation)) {
            output.nodes[id].kind = NormalizedControlKind::RetainedLoop;
            append(*loop.getBody(), id);
        } else if (auto branch = dyn_cast<scf::IfOp>(item.operation)) {
            output.nodes[id].kind = NormalizedControlKind::Conditional;
            for (auto& region : branch->getRegions()) {
                if (region.empty()) {
                    continue;
                }
                const auto arm = output.nodes.size();
                output.nodes.push_back({NormalizedControlKind::Sequence, nullptr, {}, {}});
                output.nodes[id].children.push_back(arm);
                append(region.front(), arm);
            }
        } else if (item.operation == output.function.getOperation()) {
            output.nodes[id].kind = NormalizedControlKind::Sequence;
            append(output.function.front(), id);
        } else {
            // An opaque operation remains a leaf in its parent's view. Index
            // its nested roots separately so standalone regional normalization
            // retains the historical context contract without traversing them
            // as part of that opaque leaf.
            for (auto& region : llvm::reverse(item.operation->getRegions())) {
                for (auto& block : llvm::reverse(region)) {
                    for (auto& operation : llvm::reverse(block)) {
                        pending.push_back({&operation, {}});
                    }
                }
            }
        }
        for (auto child : llvm::reverse(children)) {
            pending.push_back({child.first, child.second});
        }
    }
}
// One interval index for all regional membership checks. Vector IDs need not
// be contiguous for a subtree because conditional arms are inserted together.
void indexSubtrees(NormalizedControlTree& tree)
{
    tree.preorder.resize(tree.nodes.size());
    tree.subtreeEnd.resize(tree.nodes.size());
    SmallVector<bool> childNode(tree.nodes.size(), false);
    for (const auto& node : tree.nodes) {
        for (auto child : node.children) {
            childNode[child] = true;
        }
    }
    std::size_t next = 0;
    for (std::size_t root = 0; root < tree.nodes.size(); ++root) {
        if (childNode[root]) {
            continue;
        }
        SmallVector<std::pair<std::size_t, bool>> pending{{root, false}};
        while (!pending.empty()) {
            auto [id, leaving] = pending.pop_back_val();
            if (leaving) {
                tree.subtreeEnd[id] = next;
                continue;
            }
            tree.sourceVisits = saturatedAdd(tree.sourceVisits, 1);
            tree.preorder[id] = next++;
            pending.push_back({id, true});
            for (auto child : llvm::reverse(tree.nodes[id].children)) {
                pending.push_back({child, false});
            }
        }
    }
}
void summarize(NormalizedControlTree& tree)
{
    tree.compactSizes.resize(tree.nodes.size());
    tree.selectedSizes.resize(tree.nodes.size());
    tree.choices.resize(tree.nodes.size());
    for (std::size_t id = tree.nodes.size(); id-- > 0;) {
        ++tree.planningVisits;
        tree.sourceVisits = saturatedAdd(tree.sourceVisits, 1);
        const auto& node = tree.nodes[id];
        Size local;
        for (auto* phase : tree.index->phasesFor(node.original)) {
            local.payloads = saturatedAdd(local.payloads, 1);
            local.effects = saturatedAdd(local.effects, tree.input->accesses().effectsFor(phase).size());
        }
        auto raw = local, selected = local;
        for (auto child : node.children) {
            appendSize(raw, tree.compactSizes[child]);
            appendSize(selected, tree.selectedSizes[child]);
        }
        if (auto loop = dyn_cast_or_null<scf::ForOp>(node.original)) {
            auto choice = domain(loop, tree.limits);
            const bool topLevel = loop->getParentOp() == tree.function.getOperation();
            if (choice && (!choice->trips || (!topLevel && !selected.loops))) {
                Size expanded;
                expanded.nodes = saturatedAdd(1, saturatedMultiply(selected.nodes - 1, choice->trips));
                expanded.payloads = saturatedMultiply(selected.payloads, choice->trips);
                expanded.effects = saturatedMultiply(selected.effects, choice->trips);
                expanded.expansions = 1;
                if (fits(expanded, tree.limits)) {
                    tree.choices[id] = choice;
                    selected = expanded;
                }
            }
            raw.loops = selected.loops = true;
        }
        tree.compactSizes[id] = raw;
        tree.selectedSizes[id] = selected;
    }
}
// Nonzero expansions are innermost. Retaining a selected root therefore
// restores its compact size directly, with no new subtree planning pass.
Size regionalSize(const NormalizedControlTree& tree, std::size_t root)
{
    if (tree.choices[root] && tree.choices[root]->trips) {
        return tree.compactSizes[root];
    }
    return tree.selectedSizes[root];
}
void expand(NormalizedControlDescription& output, const Size& total)
{
    output.original = std::make_shared<const NormalizedControlDescription>(output);
    auto original = output.backing;
    auto expanded = std::make_shared<NormalizedControlTree>();
    expanded->nodes.reserve(total.nodes);
    auto roots = std::move(output.roots);
    struct Pending {
        std::size_t original;
        std::optional<std::size_t> parent;
        SmallVector<FixedLoopCoordinate> coordinates;
    };
    SmallVector<Pending> pending;
    for (auto root : llvm::reverse(roots)) {
        pending.push_back({root, {}, {}});
    }
    while (!pending.empty()) {
        auto item = pending.pop_back_val();
        const auto& source = original->nodes[item.original];
        const auto id = expanded->nodes.size();
        expanded->nodes.push_back({source.kind, source.original, item.coordinates, {}});
        if (source.original && item.coordinates.empty()) {
            expanded->originals.try_emplace(source.original, id);
        }
        if (item.parent) {
            expanded->nodes[*item.parent].children.push_back(id);
        } else {
            output.roots.push_back(id);
        }
        auto selected = original->choices[item.original];
        if (!item.parent && selected && selected->trips) {
            selected.reset();
        }
        if (selected) {
            expanded->nodes[id].kind = NormalizedControlKind::ExpandedLoop;
            ++output.expandedLoops;
            for (uint64_t visit = selected->trips; visit-- > 0;) {
                auto fixed = item.coordinates;
                const auto induction =
                    static_cast<__int128>(selected->lower) + static_cast<__int128>(visit) * selected->step;
                fixed.push_back({cast<scf::ForOp>(source.original), static_cast<int64_t>(induction)});
                for (auto child : llvm::reverse(source.children)) {
                    pending.push_back({child, id, fixed});
                }
            }
        } else {
            for (auto child : llvm::reverse(source.children)) {
                pending.push_back({child, id, item.coordinates});
            }
        }
    }
    indexSubtrees(*expanded);
    output.backing = std::move(expanded);
    output.nodes = output.backing->nodes;
    output.representedNodes = total.nodes;
    output.payloads = total.payloads;
    output.effects = total.effects;
}
bool validContext(const ArithmeticRegionContext& region)
{
    auto function = region.function;
    bool valid = function && region.root && !function.isDeclaration() && function.getBody().hasOneBlock() &&
                 (region.root == function.getOperation() || function->isProperAncestor(region.root));
    if (!region.roots.empty()) {
        auto* previous = region.roots.front();
        valid = valid && previous == region.root;
        for (auto* selected : llvm::drop_begin(region.roots)) {
            const bool inFunction = region.function && selected && region.function->isProperAncestor(selected);
            valid = valid && inFunction && previous && previous->getNextNode() == selected;
            if (!valid) {
                break;
            }
            previous = selected;
        }
    }
    return valid;
}
bool sameLimits(const FiniteExpansionLimits& a, const FiniteExpansionLimits& b)
{
    return a.visits == b.visits && a.payloads == b.payloads && a.pairs == b.pairs && a.fragments == b.fragments &&
           a.depth == b.depth;
}
} // namespace
std::shared_ptr<const NormalizedControlTree> buildNormalizedControlTree(
    func::FuncOp function, const PhaseIndex& index, const SyncInput& input, const FiniteExpansionLimits& limits)
{
    if (!function || function.isDeclaration() || !function.getBody().hasOneBlock()) {
        return {};
    }
    auto tree = std::make_shared<NormalizedControlTree>();
    tree->function = function;
    tree->index = &index;
    tree->input = &input;
    tree->limits = limits;
    compact(*tree);
    summarize(*tree);
    indexSubtrees(*tree);
    return tree;
}
uint64_t normalizedControlSourceVisits(const NormalizedControlTree& tree) { return tree.sourceVisits; }
std::optional<std::size_t> NormalizedControlDescription::originalNode(Operation* operation) const
{
    if (!backing) {
        return std::nullopt;
    }
    auto found = backing->originals.find(operation);
    if (found == backing->originals.end()) {
        return std::nullopt;
    }
    const auto id = found->second;
    for (auto root : roots) {
        if (backing->preorder[root] <= backing->preorder[id] && backing->preorder[id] < backing->subtreeEnd[root]) {
            return id;
        }
    }
    return std::nullopt;
}
std::shared_ptr<const NormalizedControlDescription> normalizeSmallCountControl(
    ArithmeticRegionContext context, const PhaseIndex& index, const SyncInput& input,
    const FiniteExpansionLimits& limits, std::shared_ptr<const NormalizedControlTree> backing)
{
    auto output = std::make_shared<NormalizedControlDescription>();
    output->context = std::move(context);
    output->index = &index;
    output->input = &input;
    output->limits = limits;
    const auto& region = output->context;
    if (!validContext(region)) {
        output->result.note(RecognitionIssue::UnsupportedControl, region.root);
        return output;
    }
    if (!backing) {
        backing = buildNormalizedControlTree(region.function, index, input, limits);
    }
    if (!backing || backing->function != region.function || backing->index != &index || backing->input != &input ||
        !sameLimits(backing->limits, limits)) {
        output->result.note(RecognitionIssue::TemplateContext, region.root);
        return output;
    }
    auto roots = region.roots;
    if (roots.empty()) {
        roots.push_back(region.root);
    }
    Size raw, selected;
    raw.nodes = selected.nodes = 0;
    for (auto* operation : roots) {
        auto found = backing->originals.find(operation);
        if (found == backing->originals.end()) {
            output->result.note(RecognitionIssue::UnsupportedControl, operation);
            return output;
        }
        output->roots.push_back(found->second);
        appendSize(raw, backing->compactSizes[found->second]);
        appendSize(selected, regionalSize(*backing, found->second));
    }
    output->backing = std::move(backing);
    output->nodes = output->backing->nodes;
    output->representedNodes = raw.nodes;
    output->payloads = raw.payloads;
    output->effects = raw.effects;
    output->planningVisits = output->backing->planningVisits;
    if (fits(selected, limits) && selected.expansions) {
        expand(*output, selected);
    }
    return output;
}
} // namespace mlir::pto::frontiersynch
