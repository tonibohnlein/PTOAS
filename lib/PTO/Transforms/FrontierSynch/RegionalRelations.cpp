// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "RegionalRelationsInternal.h"
#include <map>
namespace mlir::pto::frontiersynch {
FailureOr<RegionalAnalysis> composeSymbolicRegionalSequence(
    ArrayRef<RegionalAnalysis> children, func::FuncOp function, const SyncInput& input,
    const PhaseIndex& index, std::string& error)
{
    if (children.empty() || !children.front().expressions) {
        error = "symbolic sequence requires owned child expression interfaces"; return failure();
    }
    auto arena = children.front().expressions;
    std::vector<RegionalRelationData> exports(children.size());
    std::vector<std::unique_ptr<RegionalRelationRequest>> requests(children.size());
    SmallVector<Value> parameters;
    std::vector<RegionExpressions::Id> bindings;
    std::map<const CompoundInstanceElement*, std::vector<std::pair<unsigned, const TemplateEndpointAnchor*>>> phases;
    Operation* root = nullptr;
    auto addParameter = [&](Value value, RegionExpressions::Id binding) {
        auto found = llvm::find(parameters, value);
        if (found != parameters.end()) {
            if (bindings[found - parameters.begin()] != binding) {
                error = "symbolic children bind the same parameter differently"; return false;
            }
        } else { parameters.push_back(value); bindings.push_back(binding); }
        return true;
    };
    for (unsigned i = 0; i < children.size(); ++i) {
        const auto& child = children[i];
        if (child.expressions != arena || child.accessModel != &input.accesses() ||
            child.gmAliasPolicy != input.memory().gmPolicy()) {
            error = "symbolic children have different modeled input or expression context"; return failure();
        }
        for (const auto& anchor : child.anchors) {
            if (!anchor.phase) {
                error = "symbolic sibling occurrence has no original phase"; return failure();
            }
            auto& occurrences = phases[anchor.phase];
            for (const auto& previous : occurrences) {
                const bool disjoint = llvm::any_of(anchor.coordinates, [&](const auto& fixed) {
                    return llvm::any_of(previous.second->coordinates, [&](const auto& other) {
                        return fixed.loop == other.loop && fixed.induction != other.induction;
                    });
                });
                if (previous.first != i || !disjoint) {
                    error = "symbolic sibling occurrence identities need an explicit phase-coordinate adapter";
                    return failure();
                }
            }
            occurrences.push_back({i, &anchor});
            auto* operation = anchor.phase->elementOp;
            if (!root) { root = operation; }
            while (root && root != operation && !root->isProperAncestor(operation)) { root = root->getParentOp(); }
        }
        if (child.relations) { exports[i] = child.relations->data; }
        else if (child.arithmeticRelations) { exports[i] = arithmeticRelationData(*child.arithmeticRelations); }
        else {
            auto requested = requestCallbackRegionalRelations(child, function, input, index, error);
            if (failed(requested)) { return failure(); }
            requests[i] = std::move(*requested);
        }
        if (requests[i]) {
            for (auto parameter : requests[i]->parameters()) {
                if (!addParameter(parameter, arena->input(parameter))) { return failure(); }
            }
        } else {
            if (exports[i].input != &input || exports[i].context.function != function ||
                exports[i].parameterValues.size() != exports[i].parameters.size()) {
                error = "symbolic relation provenance or parameter schema differs"; return failure();
            }
            for (unsigned j = 0; j < exports[i].parameters.size(); ++j) {
                if (!addParameter(exports[i].parameterValues[j], exports[i].parameters[j])) { return failure(); }
            }
        }
    }
    // Deferred effects retain physical identities. An omitted effect may only
    // remain omitted if no sibling can conflict under the shared alias policy.
    for (unsigned i = 0; i < children.size(); ++i) {
        for (unsigned j = i + 1; j < children.size(); ++j) {
            for (const auto* left : {&children[i].accessBoundary, &children[i].deferredAccessBoundary}) {
                for (const auto* right : {&children[j].accessBoundary, &children[j].deferredAccessBoundary}) {
                    for (const auto& a : *left) {
                        for (const auto& b : *right) {
                            if ((left == &children[i].deferredAccessBoundary ||
                                 right == &children[j].deferredAccessBoundary) &&
                                input.accesses().mayConflict(a.effect, b.effect)) {
                                error = "symbolic sibling omitted-access relation adapter unavailable";
                                return failure();
                            }
                        }
                    }
                }
            }
        }
    }
    auto carrier = std::make_shared<RegionalRelations>();
    carrier->children.assign(children.begin(), children.end());
    for (unsigned i = 0; i < children.size(); ++i) {
        if (requests[i]) {
            auto exported = requests[i]->lower(parameters, error);
            if (failed(exported)) { return failure(); }
            exports[i] = std::move(*exported); carrier->auxiliaryOwners.push_back(requests[i]->variables);
        }
        if (failed(normalizeRegionalRelationData(exports[i], parameters, bindings, error))) { return failure(); }
    }
    carrier->data = std::move(exports.front());
    std::vector<unsigned> owner(carrier->data.sites.size(), 0);
    for (unsigned i = 1; i < exports.size(); ++i) {
        auto combined = composeRegionalRelationData(carrier->data, exports[i], {function, root}, error);
        if (failed(combined)) { return failure(); }
        carrier->data = std::move(*combined); owner.insert(owner.end(), exports[i].sites.size(), i);
    }
    for (const auto& [key, pieces] : carrier->data.analysis.minimumDemands) {
        if (owner[key.source.site] != owner[key.target.site]) { carrier->newCrossings.emplace(key, pieces); }
    }
    return exportRegionalRelationData(std::move(carrier), std::move(arena), error);
}
} // namespace mlir::pto::frontiersynch
