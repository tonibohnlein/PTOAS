// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SourcePipeServiceChecks.h"
#include "PTO/Transforms/FrontierSynch/Shared/ServicePruning.h"
#include "PTO/Transforms/FrontierSynch/Shared/SourcePipeServices.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <map>
#include <set>
namespace fs = mlir::pto::frontiersynch;
namespace {
struct Input {
    std::vector<fs::SourcePipeNode> graph{{std::nullopt, {}}};
    std::vector<fs::SourcePipeSite> sites;
    std::size_t classes = 0;
    std::string error;
    std::size_t node(std::optional<std::size_t> site, const std::vector<std::size_t>& before)
    {
        auto id = graph.size();
        graph.push_back({site, {}});
        for (auto previous : before) {
            graph[previous].successors.push_back(id);
        }
        return id;
    }
    std::vector<std::size_t> sequence(const llvm::json::Array& tree, std::vector<std::size_t> before)
    {
        for (const auto& value : tree) {
            auto* item = value.getAsObject();
            if (!item) {
                error = "invalid structured node";
                return {};
            }
            auto kind = item->getString("kind");
            if (kind == "payload") {
                auto site = item->getInteger("site");
                if (!site || *site < 0) {
                    error = "invalid payload site";
                    return {};
                }
                before = {node(*site, before)};
            } else if (kind == "loop") {
                auto* body = item->getArray("body");
                if (!body) {
                    error = "missing loop body";
                    return {};
                }
                auto header = node(std::nullopt, before);
                for (auto backedge : sequence(*body, {header})) {
                    graph[backedge].successors.push_back(header);
                }
                before = {header};
            } else if (kind == "if") {
                auto* left = item->getArray("then");
                auto* right = item->getArray("else");
                if (!left || !right) {
                    error = "missing branch arms";
                    return {};
                }
                auto gate = node(std::nullopt, before);
                auto a = sequence(*left, {gate});
                auto b = sequence(*right, {gate});
                a.insert(a.end(), b.begin(), b.end());
                // Empty arms share the gate; the CFG is a set of edges.
                std::sort(a.begin(), a.end());
                a.erase(std::unique(a.begin(), a.end()), a.end());
                before = {node(std::nullopt, a)};
            } else {
                error = "unsupported structured control";
                return {};
            }
        }
        return before;
    }
    bool read(const llvm::json::Object& input)
    {
        auto count = input.getInteger("classes");
        auto* inventory = input.getArray("sites");
        auto* tree = input.getArray("tree");
        if (!count || *count < 0 || !inventory || !tree) {
            error = "missing class/site/tree inventory";
            return false;
        }
        classes = *count;
        for (const auto& value : *inventory) {
            auto* item = value.getAsObject();
            auto pipe = item ? item->getInteger("pipe") : std::nullopt;
            auto* facts = item ? item->getArray("facts") : nullptr;
            if (!pipe || *pipe < 0 || uint64_t(*pipe) > UINT32_MAX || !facts) {
                error = "invalid source pipe or facts";
                return false;
            }
            fs::SourcePipeSite site;
            site.pipe = *pipe;
            for (const auto& fact : *facts) {
                auto mode = fact.getAsInteger();
                if (!mode || *mode < 0 || *mode > 3) {
                    error = "invalid access mode";
                    return false;
                }
                site.modes.push_back(*mode);
            }
            if (auto* refresh = item->getArray("refresh")) {
                for (const auto& value : *refresh) {
                    auto cell = value.getAsInteger();
                    if (!cell || *cell < 0) {
                        error = "invalid class-wide refresh";
                        return false;
                    }
                    site.refreshed.push_back(*cell);
                }
            }
            sites.push_back(std::move(site));
        }
        sequence(*tree, {0});
        return error.empty();
    }
};
llvm::json::Array states(const std::vector<fs::SourcePipeFacts>& source)
{
    llvm::json::Array result;
    for (const auto& state : source) {
        std::vector<std::array<uint64_t, 3>> sorted;
        for (const auto& fact : state) {
            sorted.push_back({fact.cell, fact.mode, fact.pipe});
        }
        std::sort(sorted.begin(), sorted.end());
        llvm::json::Array row;
        for (auto fact : sorted) {
            row.push_back(llvm::json::Array{fact[0], fact[1], fact[2]});
        }
        result.push_back(std::move(row));
    }
    return result;
}

llvm::json::Array commandPlan(
    const llvm::json::Array& tree, const Input& input, const fs::SourcePipeServices& result, bool physical)
{
    llvm::json::Array output;
    for (const auto& value : tree) {
        const auto& item = *value.getAsObject();
        auto kind = *item.getString("kind");
        llvm::json::Object node;
        if (auto line = item.getInteger("line")) {
            node["line"] = *line;
        }
        if (kind == "payload") {
            auto site = *item.getInteger("site");
            auto target = input.sites[site].pipe;
            node["kind"] = "service";
            node["site"] = site;
            llvm::json::Array commands;
            auto add = [&](const char* op, uint32_t pipe, uint32_t source, bool reverse) {
                llvm::json::Object command{{"op", op}, {"pipe", pipe}};
                if (physical) {
                    command["source"] = std::string(reverse ? "R:" : "F:") + std::to_string(reverse ? target : source) +
                                        ":" + std::to_string(reverse ? source : target);
                } else {
                    command["source"] = source;
                }
                commands.push_back(std::move(command));
            };
            for (auto source : result.possible[site]) {
                if (source == target) {
                    continue;
                }
                add("S", source, source, false);
                if (physical) {
                    add("W", target, source, false);
                    add("S", target, source, true);
                    add("W", source, source, true);
                }
            }
            if (std::binary_search(result.possible[site].begin(), result.possible[site].end(), target)) {
                commands.push_back(llvm::json::Object{{"op", "B"}, {"pipe", target}});
            }
            if (!physical) {
                for (auto source : result.possible[site]) {
                    if (source != target) {
                        add("W", target, source, false);
                    }
                }
            }
            node["commands"] = std::move(commands);
        } else {
            node["kind"] = kind;
            if (kind == "loop") {
                node["body"] = commandPlan(*item.getArray("body"), input, result, physical);
            } else {
                node["then"] = commandPlan(*item.getArray("then"), input, result, physical);
                node["else"] = commandPlan(*item.getArray("else"), input, result, physical);
            }
        }
        output.push_back(std::move(node));
    }
    return output;
}
llvm::json::Array roleIds(const Input& input, const fs::SourcePipeServices& result)
{
    std::map<std::pair<uint32_t, uint32_t>, std::set<std::string>> roles;
    for (std::size_t site = 0; site < input.sites.size(); ++site) {
        auto target = input.sites[site].pipe;
        for (auto source : result.possible[site]) {
            if (source == target) {
                continue;
            }
            roles[{source, target}].insert("F");
            roles[{target, source}].insert("R");
        }
    }
    llvm::json::Array output;
    for (const auto& [direction, names] : roles) {
        uint64_t id = 0;
        for (const auto& name : names) {
            output.push_back(
                llvm::json::Object{
                    {"role", name}, {"source", direction.first}, {"target", direction.second}, {"id", id++}});
        }
    }
    return output;
}
} // namespace
llvm::json::Object exportSourcePipeServices(const llvm::json::Object& input)
{
    const auto began = std::chrono::steady_clock::now();
    Input source;
    if (!source.read(input)) {
        return llvm::json::Object{{"error", source.error}};
    }
    const auto prepared = std::chrono::steady_clock::now();
    auto result = fs::solveSourcePipeServices(source.graph, source.sites, source.classes);
    if (!result.error.empty()) {
        return llvm::json::Object{{"error", result.error}};
    }
    const auto solved = std::chrono::steady_clock::now();
    llvm::json::Array possible;
    for (const auto& pipes : result.possible) {
        llvm::json::Array row;
        for (auto pipe : pipes) {
            row.push_back(pipe);
        }
        possible.push_back(std::move(row));
    }
    auto logical = commandPlan(*input.getArray("tree"), source, result, false);
    auto physical = commandPlan(*input.getArray("tree"), source, result, true);
    auto ids = roleIds(source, result);
    const auto planned = std::chrono::steady_clock::now();
    const auto& w = result.work;
    return llvm::json::Object{
        {"error", result.error},
        {"logical_plan", std::move(logical)},
        {"physical_plan", std::move(physical)},
        {"role_ids", std::move(ids)},
        {"incoming", states(result.incoming)},
        {"outgoing", states(result.outgoing)},
        {"possible", std::move(possible)},
        {"metrics", llvm::json::Object{
                        {"prepared_modes", w.preparedModes},
                        {"cfg_nodes", source.graph.size()},
                        {"cfg_preparation_seconds", std::chrono::duration<double>(prepared - began).count()},
                        {"solve_seconds", std::chrono::duration<double>(solved - prepared).count()},
                        {"plan_serialization_seconds", std::chrono::duration<double>(planned - solved).count()},
                        {"cfg_edges", w.preparedEdges},
                        {"generated_insertions", w.generated},
                        {"facts_retained", w.outgoing},
                        {"incoming_insertions", w.incoming},
                        {"edge_fact_transmissions", w.transmissions},
                        {"killed_arrivals", w.killed},
                        {"selection_fact_inspections", w.inspected}}}};
}
llvm::json::Object exportSourcePipePruning(const llvm::json::Object& input, const llvm::json::Object& baseline)
{
    const auto started = std::chrono::steady_clock::now();
    Input source;
    if (!source.read(input)) {
        return llvm::json::Object{{"accepted", false}, {"reason", source.error}};
    }
    fs::ServiceMask full, protectedPaths(source.sites.size());
    auto readMask = [&](const llvm::json::Array* rows, fs::ServiceMask& mask) {
        if (!rows || rows->size() != source.sites.size()) {
            return false;
        }
        mask.clear();
        for (const auto& value : *rows) {
            auto* row = value.getAsArray();
            if (!row) {
                return false;
            }
            mask.emplace_back();
            for (const auto& pipe : *row) {
                auto number = pipe.getAsInteger();
                if (!number || *number < 0 || uint64_t(*number) > UINT32_MAX) {
                    return false;
                }
                mask.back().push_back(*number);
            }
        }
        return true;
    };
    if (!readMask(baseline.getArray("possible"), full) ||
        (input.get("protected_services") && !readMask(input.getArray("protected_services"), protectedPaths))) {
        return llvm::json::Object{{"accepted", false}, {"reason", "invalid service masks"}};
    }
    if (input.get("baseline_services") && !readMask(input.getArray("baseline_services"), full)) {
        return llvm::json::Object{{"accepted", false}, {"reason", "invalid diagnostic baseline mask"}};
    }
    const auto began = std::chrono::steady_clock::now();
    auto proposal = fs::proposeLocalServices(source.graph, source.sites, full, protectedPaths);
    if (!proposal.error.empty()) {
        return llvm::json::Object{{"accepted", false}, {"reason", proposal.error}, {"certificate_runs", 0}};
    }
    // Diagnostic-only fixed candidates exercise joint rejection, including
    // mutually redundant proposals. The emitter never supplies this field.
    if (input.get("candidate_services") && !readMask(input.getArray("candidate_services"), proposal.kept)) {
        return llvm::json::Object{{"accepted", false}, {"reason", "invalid candidate mask"}};
    }
    const auto proposed = std::chrono::steady_clock::now();
    auto certificate = fs::certifyServiceDelivery(source.graph, source.sites, full, proposal.kept, protectedPaths);
    const auto certified = std::chrono::steady_clock::now();
    fs::SourcePipeServices selected;
    selected.possible = proposal.kept;
    llvm::json::Array kept;
    for (const auto& row : proposal.kept) {
        llvm::json::Array pipes;
        for (auto pipe : row) {
            pipes.push_back(pipe);
        }
        kept.push_back(std::move(pipes));
    }
    llvm::json::Object result{
        {"accepted", certificate.accepted},
        {"reason", certificate.error},
        {"certificate_runs", 1},
        {"input_preparation_seconds", std::chrono::duration<double>(began - started).count()},
        {"proposal_seconds", std::chrono::duration<double>(proposed - began).count()},
        {"verification_seconds", std::chrono::duration<double>(certified - proposed).count()},
        {"proposal_nodes", proposal.nodes},
        {"service_tests", proposal.tested},
        {"class_tests", proposal.classTests},
        {"class_updates", proposal.updates},
        {"clear_entries", proposal.cleared},
        {"proposed_deletions", proposal.deleted},
        {"facts_retained", certificate.retained},
        {"fact_transmissions", certificate.transmissions},
        {"fact_inspections", certificate.inspected},
        {"violations", certificate.violations},
        {"kept", std::move(kept)}};
    if (certificate.accepted) {
        result["physical_plan"] = commandPlan(*input.getArray("tree"), source, selected, true);
    }
    result["plan_construction_seconds"] =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - certified).count();
    return result;
}
int runSourcePipeServicesJson(const char* path)
{
    auto buffer = llvm::MemoryBuffer::getFile(path);
    if (!buffer) {
        llvm::errs() << "cannot read source-pipe input\n";
        return 1;
    }
    auto parsed = llvm::json::parse((*buffer)->getBuffer());
    if (!parsed) {
        llvm::errs() << llvm::toString(parsed.takeError()) << "\n";
        return 1;
    }
    auto* input = parsed->getAsObject();
    if (!input) {
        llvm::errs() << "source-pipe input must be an object\n";
        return 1;
    }
    auto result = exportSourcePipeServices(*input);
    if (input->getBoolean("prune").value_or(false) && result.getString("error").value_or("invalid").empty()) {
        result["pruning"] = exportSourcePipePruning(*input, result);
    }
    llvm::outs() << llvm::json::Value(std::move(result)) << "\n";
    return 0;
}
