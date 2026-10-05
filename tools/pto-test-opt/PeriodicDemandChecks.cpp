// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Bounded numerical test protocol, independent of MLIR payload instructions.
#include "PTO/Transforms/FrontierSynch/LifetimeScan.h"
#include "PTO/Transforms/FrontierSynch/PeriodicAnalysis.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>
namespace fs = mlir::pto::frontiersynch;
llvm::json::Object dumpPeriodicAnalysis(const fs::PeriodicAnalysis& result);
bool appendLogicalChecks(const llvm::json::Object&, const fs::PeriodicAnalysis&, llvm::json::Object&);
bool appendAllocationChecks(const llvm::json::Object&, const fs::PeriodicAnalysis&, llvm::json::Object&);
namespace {
std::optional<uint64_t> number(const llvm::json::Object& object, llvm::StringRef name)
{
    const auto* value = object.get(name);
    return value ? value->getAsUINT64() : std::nullopt;
}
bool accesses(const llvm::json::Array& array, std::vector<fs::CellAccess>& result)
{
    if (array.size() > 4096) {
        return false;
    }
    for (const auto& item : array) {
        const auto* object = item.getAsObject();
        if (!object) {
            return false;
        }
        auto atom = number(*object, "atom");
        auto read = object->getBoolean("read"), write = object->getBoolean("write");
        if (!atom || *atom > UINT32_MAX || !read || !write) {
            return false;
        }
        auto group = number(*object, "protection_group");
        if (object->get("protection_group") && !group) {
            return false;
        }
        result.push_back({static_cast<uint32_t>(*atom), *read, *write, group.value_or(0)});
    }
    return true;
}
bool occurrences(const llvm::json::Array& array, std::vector<fs::ExplicitEffects>& result)
{
    if (array.size() > 128) {
        return false;
    }
    for (const auto& item : array) {
        const auto* object = item.getAsObject();
        if (!object) {
            return false;
        }
        auto id = number(*object, "id"), pipe = number(*object, "pipe");
        const auto* modes = object->getArray("accesses");
        if (!id || *id > UINT32_MAX || !pipe || *pipe > UINT32_MAX || !modes) {
            return false;
        }
        fs::ExplicitEffects occurrence{static_cast<uint32_t>(*id), static_cast<uint32_t>(*pipe), {}};
        if (!accesses(*modes, occurrence.accesses)) {
            return false;
        }
        result.push_back(std::move(occurrence));
    }
    return true;
}
bool records(const llvm::json::Array& array, std::vector<fs::PeriodicRecord>& result)
{
    if (array.size() > 4096) {
        return false;
    }
    for (const auto& item : array) {
        const auto* tuple = item.getAsArray();
        if (!tuple || tuple->size() != 3) {
            return false;
        }
        auto source = (*tuple)[0].getAsUINT64(), target = (*tuple)[1].getAsUINT64();
        auto distance = (*tuple)[2].getAsUINT64();
        if (!source || *source > UINT32_MAX || !target || *target > UINT32_MAX || !distance) {
            return false;
        }
        result.push_back({static_cast<uint32_t>(*source), static_cast<uint32_t>(*target), *distance});
    }
    return true;
}
llvm::json::Object dumpScan(const fs::StorageScanResult& scan)
{
    llvm::json::Array generators, witnesses;
    for (const auto& edge : scan.generators) {
        generators.push_back(llvm::json::Array{edge.source, edge.target});
    }
    for (const auto& witness : scan.witnesses) {
        witnesses.push_back(llvm::json::Array{witness.generator, witness.atom,
                                            static_cast<unsigned>(witness.hazard)});
    }
    return llvm::json::Object{{"error", scan.error}, {"generators", std::move(generators)},
                              {"witnesses", std::move(witnesses)}, {"protected_hazards", scan.protectedHazards}};
}
bool scanCase(const llvm::json::Object& input, llvm::json::Object& output)
{
    std::vector<fs::ExplicitEffects> effects;
    const auto* values = input.getArray("scan");
    if (!values || !occurrences(*values, effects)) {
        return false;
    }
    std::vector<fs::StorageGenerator> extra;
    if (const auto* entries = input.getArray("prerequisites")) {
        std::vector<fs::PeriodicRecord> parsed;
        if (!records(*entries, parsed)) {
            return false;
        }
        for (const auto& record : parsed) {
            extra.push_back({record.source, record.target});
        }
    }
    output = dumpScan(fs::scanStorageLifetimes(effects, extra));
    return true;
}
bool queryTables(const fs::PeriodicAnalysis& analysis, uint64_t prefix, llvm::json::Object& output)
{
    const auto count = analysis.payloads.size();
    llvm::json::Array thresholds, reachable, strict, ranks;
    for (uint32_t a = 0; a < count; ++a) {
        llvm::json::Array row;
        for (uint32_t v = 0; v < 2 * count; ++v) {
            auto target = fs::PeriodicEvent{v / 2, static_cast<fs::PeriodicEventKind>(v % 2)};
            const auto value = analysis.completionThreshold(a, target);
            if (value.error != fs::PeriodicQueryError::None) {
                return false;
            }
            row.push_back(value.displacement ? llvm::json::Value(*value.displacement) : llvm::json::Value(nullptr));
        }
        thresholds.push_back(std::move(row));
    }
    for (uint64_t a = 0; a < prefix; ++a) {
        llvm::json::Array row, strictRow;
        for (uint64_t v = 0; v < 2 * prefix; ++v) {
            auto target = fs::PeriodicEvent{static_cast<uint32_t>((v / 2) % count),
                                           static_cast<fs::PeriodicEventKind>(v % 2)};
            auto query = [&](bool strictIdentity) {
                return analysis.completionPrecedes(a % count, a / count, target,
                                                   v / 2 / count, prefix, strictIdentity);
            };
            const auto reflexive = query(false), strictResult = query(true);
            if (reflexive.error != fs::PeriodicQueryError::None || strictResult.error != fs::PeriodicQueryError::None) {
                return false;
            }
            row.push_back(reflexive.value);
            strictRow.push_back(strictResult.value);
        }
        reachable.push_back(std::move(row));
        strict.push_back(std::move(strictRow));
    }
    for (uint64_t v = 0; v < 2 * prefix; ++v) {
        llvm::json::Array row;
        const auto target = fs::PeriodicEvent{static_cast<uint32_t>((v / 2) % count),
                                              static_cast<fs::PeriodicEventKind>(v % 2)};
        for (const auto& pipe : analysis.frontiers) {
            const auto rank = analysis.completionRank(pipe.pipe, target, v / 2 / count, prefix);
            if (rank.error != fs::PeriodicQueryError::None) {
                return false;
            }
            row.push_back(rank.value);
        }
        ranks.push_back(std::move(row));
    }
    output["thresholds"] = std::move(thresholds);
    output["reachable"] = std::move(reachable);
    output["strict"] = std::move(strict);
    output["ranks"] = std::move(ranks);
    // Query boundary cases are checked independently of the supplied prefix.
    const auto invalid = fs::PeriodicEvent{static_cast<uint32_t>(count), fs::PeriodicEventKind::Start};
    output["invalid_threshold"] = static_cast<unsigned>(analysis.completionThreshold(0, invalid).error);
    output["invalid_endpoint"] = static_cast<unsigned>(analysis.completionPrecedes(0, 0, {0}, 0, 0).error);
    output["invalid_kind"] = static_cast<unsigned>(analysis.completionThreshold(
        0, {0, static_cast<fs::PeriodicEventKind>(9)}).error);
    return true;
}
bool rankQueries(const llvm::json::Object& input, const fs::PeriodicAnalysis& analysis,
                 llvm::json::Object& output)
{
    const auto* queries = input.getArray("rank_queries");
    if (!queries) {
        return true;
    }
    if (queries->size() > 128) {
        return false;
    }
    llvm::json::Array answers;
    for (const auto& item : *queries) {
        const auto* tuple = item.getAsArray();
        if (!tuple || tuple->size() != 5) {
            return false;
        }
        uint64_t values[5] = {};
        for (unsigned i = 0; i < 5; ++i) {
            auto value = (*tuple)[i].getAsUINT64();
            if (!value || (i < 3 && *value > UINT32_MAX)) {
                return false;
            }
            values[i] = *value;
        }
        const auto rank = analysis.completionRank(values[0],
            {static_cast<uint32_t>(values[1]), static_cast<fs::PeriodicEventKind>(values[2])}, values[3], values[4]);
        answers.push_back(llvm::json::Object{{"error", static_cast<unsigned>(rank.error)}, {"value", rank.value}});
    }
    output["rank_answers"] = std::move(answers);
    return true;
}
bool graphCase(const llvm::json::Object& input, llvm::json::Object& output)
{
    const auto* pipes = input.getArray("pipes");
    if (!pipes || pipes->size() > 512) {
        return false;
    }
    std::vector<fs::PeriodicPayload> payloads;
    for (const auto& value : *pipes) {
        auto pipe = value.getAsUINT64();
        if (!pipe || *pipe > UINT32_MAX) {
            return false;
        }
        payloads.push_back({static_cast<uint32_t>(*pipe)});
    }
    std::vector<fs::PeriodicRecord> generators;
    if (const auto* word = input.getArray("word")) {
        if (word->size() != payloads.size()) {
            return false;
        }
        std::vector<fs::ExplicitEffects> visits;
        for (uint32_t i = 0; i < 2 * payloads.size(); ++i) {
            auto type = i % payloads.size();
            const auto* mode = (*word)[type].getAsArray();
            fs::ExplicitEffects occurrence{i, payloads[type].pipe, {}};
            if (!mode || !accesses(*mode, occurrence.accesses)) {
                return false;
            }
            visits.push_back(std::move(occurrence));
        }
        const auto scanned = fs::scanStorageLifetimes(visits);
        output["scan"] = dumpScan(scanned);
        if (!scanned.error.empty()) {
            output["error"] = scanned.error;
            return true;
        }
        for (const auto& edge : scanned.generators) {
            if (edge.source < payloads.size()) {
                generators.push_back({edge.source, static_cast<uint32_t>(edge.target % payloads.size()),
                                      edge.target / payloads.size()});
            }
        }
    }
    if (const auto* values = input.getArray("records")) {
        if (!records(*values, generators)) {
            return false;
        }
    }
    const auto analysis = fs::analyzePeriodicDemands(payloads, generators);
    auto summary = dumpPeriodicAnalysis(analysis);
    for (auto& entry : summary) {
        output[entry.first] = std::move(entry.second);
    }
    auto prefix = number(input, "prefix").value_or(0);
    if (prefix > 64 || (payloads.empty() && prefix)) {
        return false;
    }
    if (analysis.error.empty()) {
        return queryTables(analysis, prefix, output) && rankQueries(input, analysis, output) &&
            appendLogicalChecks(input, analysis, output) && appendAllocationChecks(input, analysis, output);
    }
    return true;
}
} // namespace
int runPeriodicDemandChecks(llvm::StringRef path)
{
    auto buffer = llvm::MemoryBuffer::getFile(path);
    if (!buffer || (*buffer)->getBufferSize() > 16 * 1024 * 1024) {
        llvm::errs() << "cannot read bounded periodic test request\n";
        return 1;
    }
    auto parsed = llvm::json::parse((*buffer)->getBuffer());
    if (!parsed) {
        llvm::consumeError(parsed.takeError());
        return 1;
    }
    auto* cases = parsed->getAsArray();
    if (!cases || cases->size() > 512) {
        return 1;
    }
    llvm::json::Array output;
    for (const auto& value : *cases) {
        const auto* input = value.getAsObject();
        if (!input) {
            return 1;
        }
        llvm::json::Object result;
        if (!(input->get("scan") ? scanCase(*input, result) : graphCase(*input, result))) {
            llvm::errs() << "malformed periodic test request\n";
            return 1;
        }
        output.push_back(std::move(result));
    }
    llvm::outs() << llvm::json::Value(std::move(output)) << "\n";
    return 0;
}
