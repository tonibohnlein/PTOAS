// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Size planning precedes construction of one original-coordinate control view.
#include "NormalizedControl.h"
namespace mlir::pto::frontiersynch {
namespace {
struct Size {
    uint64_t nodes = 1, payloads = 0, effects = 0;
    bool loops = false;
};
struct LoopDomain {
    int64_t lower = 0, step = 1;
    uint64_t trips = 0;
};
uint64_t saturatedAdd(uint64_t a, uint64_t b)
{
    return b > UINT64_MAX - a ? UINT64_MAX : a + b;
}
uint64_t saturatedMultiply(uint64_t a, uint64_t b)
{
    return b && a > UINT64_MAX / b ? UINT64_MAX : a * b;
}
std::optional<LoopDomain> domain(scf::ForOp loop, const FiniteExpansionLimits& limits)
{
    auto lower = detail::normalizedInteger(loop.getLowerBound(), {}, loop.getContext());
    auto upper = detail::normalizedInteger(loop.getUpperBound(), {}, loop.getContext());
    auto step = detail::normalizedInteger(loop.getStep(), {}, loop.getContext());
    if (!lower || !upper || !step || *step <= 0) { return std::nullopt; }
    const __int128 distance = static_cast<__int128>(*upper) - *lower;
    const __int128 trips = distance <= 0 ? 0 : (distance + *step - 1) / *step;
    const __int128 final = static_cast<__int128>(*lower) + trips * *step;
    if (trips > limits.visits || final > INT64_MAX) { return std::nullopt; }
    return LoopDomain{*lower, *step, static_cast<uint64_t>(trips)};
}
// Construct the compact structural view iteratively, even if an unreachable
// arm is deeper or larger than a producer's enumeration limit. Those limits
// belong to the producer after it has pruned proven-dead control.
void compact(NormalizedControlDescription& output)
{
    struct Pending { Operation* operation; std::optional<std::size_t> parent; };
    SmallVector<Pending> pending;
    auto roots = output.context.roots;
    if (roots.empty()) { roots.push_back(output.context.root); }
    for (auto* root : llvm::reverse(roots)) { pending.push_back({root, {}}); }
    while (!pending.empty()) {
        auto item = pending.pop_back_val();
        const auto id = output.nodes.size();
        output.nodes.push_back({NormalizedControlKind::Leaf, item.operation, {}, {}});
        if (item.parent) { output.nodes[*item.parent].children.push_back(id); }
        else { output.roots.push_back(id); }
        if (!item.operation) { continue; }
        SmallVector<std::pair<Operation*, std::size_t>> children;
        auto append = [&](Block& block, std::size_t parent) {
            for (auto& operation : block) { children.push_back({&operation, parent}); }
        };
        if (auto loop = dyn_cast<scf::ForOp>(item.operation)) {
            output.nodes[id].kind = NormalizedControlKind::RetainedLoop;
            append(*loop.getBody(), id);
        } else if (auto branch = dyn_cast<scf::IfOp>(item.operation)) {
            output.nodes[id].kind = NormalizedControlKind::Conditional;
            for (auto& region : branch->getRegions()) {
                if (region.empty()) { continue; }
                const auto arm = output.nodes.size();
                output.nodes.push_back({NormalizedControlKind::Sequence, nullptr, {}, {}});
                output.nodes[id].children.push_back(arm);
                append(region.front(), arm);
            }
        } else if (item.operation == output.context.function.getOperation()) {
            output.nodes[id].kind = NormalizedControlKind::Sequence;
            append(output.context.function.front(), id);
        }
        for (auto child : llvm::reverse(children)) { pending.push_back({child.first, child.second}); }
    }
}
void select(NormalizedControlDescription& output)
{
    // Each original node is measured once. Only innermost loops are optional
    // expansion candidates: no speculative nested trial or compact retry.
    // Compact counts remain meaningful when no expanded alternative fits.
    for (const auto& node : output.nodes) {
        for (auto* phase : output.index->phasesFor(node.original)) {
            output.payloads = saturatedAdd(output.payloads, 1);
            output.effects = saturatedAdd(output.effects, output.input->accesses().effectsFor(phase).size());
        }
    }
    SmallVector<Size> sizes(output.nodes.size());
    SmallVector<std::optional<LoopDomain>> choices(output.nodes.size());
    for (std::size_t i = output.nodes.size(); i-- > 0;) {
        ++output.planningVisits;
        auto& size = sizes[i];
        const auto& node = output.nodes[i];
        for (auto* phase : output.index->phasesFor(node.original)) {
            size.payloads = saturatedAdd(size.payloads, 1);
            size.effects = saturatedAdd(size.effects, output.input->accesses().effectsFor(phase).size());
        }
        for (auto child : node.children) {
            size.nodes = saturatedAdd(size.nodes, sizes[child].nodes);
            size.payloads = saturatedAdd(size.payloads, sizes[child].payloads);
            size.effects = saturatedAdd(size.effects, sizes[child].effects);
            size.loops |= sizes[child].loops;
        }
        auto loop = dyn_cast_or_null<scf::ForOp>(node.original);
        if (!loop) { continue; }
        const bool root = llvm::is_contained(output.roots, i) ||
            loop->getParentOp() == output.context.function.getOperation();
        auto selected = domain(loop, output.limits);
        const bool eligible = selected && (!selected->trips || (!root && !size.loops));
        if (eligible) {
            Size expanded;
            expanded.nodes = saturatedAdd(1, saturatedMultiply(size.nodes - 1, selected->trips));
            expanded.payloads = saturatedMultiply(size.payloads, selected->trips);
            expanded.effects = saturatedMultiply(size.effects, selected->trips);
            const bool fits = expanded.nodes <= output.limits.visits &&
                expanded.payloads <= output.limits.payloads;
            if (fits) { choices[i] = selected; size = expanded; }
        }
        size.loops = true;
    }
    Size total; total.nodes = 0;
    for (auto root : output.roots) {
        total.nodes = saturatedAdd(total.nodes, sizes[root].nodes);
        total.payloads = saturatedAdd(total.payloads, sizes[root].payloads);
        total.effects = saturatedAdd(total.effects, sizes[root].effects);
    }
    const bool fits = total.nodes <= output.limits.visits && total.payloads <= output.limits.payloads;
    if (!fits) { return; } // Optional expansion never rejects the compact input.
    auto original = std::move(output.nodes);
    auto roots = std::move(output.roots);
    output.nodes.clear(); output.roots.clear(); output.nodes.reserve(total.nodes);
    struct Pending {
        std::size_t original;
        std::optional<std::size_t> parent;
        SmallVector<FixedLoopCoordinate> coordinates;
    };
    SmallVector<Pending> pending;
    for (auto root : llvm::reverse(roots)) { pending.push_back({root, {}, {}}); }
    while (!pending.empty()) {
        auto item = pending.pop_back_val();
        const auto& source = original[item.original];
        const auto id = output.nodes.size();
        output.nodes.push_back({source.kind, source.original, item.coordinates, {}});
        if (item.parent) { output.nodes[*item.parent].children.push_back(id); }
        else { output.roots.push_back(id); }
        auto selected = choices[item.original];
        if (selected) {
            output.nodes[id].kind = NormalizedControlKind::ExpandedLoop;
            ++output.expandedLoops;
            for (uint64_t visit = selected->trips; visit-- > 0;) {
                auto fixed = item.coordinates;
                const auto induction = static_cast<__int128>(selected->lower) +
                    static_cast<__int128>(visit) * selected->step;
                fixed.push_back({cast<scf::ForOp>(source.original), static_cast<int64_t>(induction)});
                for (auto child : llvm::reverse(source.children)) { pending.push_back({child, id, fixed}); }
            }
        } else {
            for (auto child : llvm::reverse(source.children)) {
                pending.push_back({child, id, item.coordinates});
            }
        }
    }
    output.payloads = total.payloads; output.effects = total.effects;
}
} // namespace
std::shared_ptr<const NormalizedControlDescription> normalizeSmallCountControl(
    ArithmeticRegionContext context, const PhaseIndex& index, const SyncInput& input,
    const FiniteExpansionLimits& limits)
{
    auto output = std::make_shared<NormalizedControlDescription>();
    output->context = std::move(context); output->index = &index; output->input = &input; output->limits = limits;
    auto region = output->context;
    bool valid = region.function && region.root && !region.function.isDeclaration() &&
        region.function.getBody().hasOneBlock() &&
        (region.root == region.function.getOperation() || region.function->isProperAncestor(region.root));
    if (!region.roots.empty()) {
        auto* previous = region.roots.front();
        valid = valid && previous == region.root;
        for (auto* selected : llvm::drop_begin(region.roots)) {
            const bool inFunction = region.function && selected && region.function->isProperAncestor(selected);
            valid = valid && inFunction && previous && previous->getNextNode() == selected;
            if (!valid) { break; }
            previous = selected;
        }
    }
    if (!valid) { output->result.note(RecognitionIssue::UnsupportedControl, region.root); return output; }
    compact(*output); select(*output);
    return output;
}
} // namespace mlir::pto::frontiersynch
