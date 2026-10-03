// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Bounded test-only adapter for supplied exact rotating fragments. No compiler
// import or executable synchronization behavior is implemented by this driver.
#include "frontier-periodic-driver.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include <memory>

namespace frontier_test {
namespace fs = mlir::pto::frontiersynch;
using llvm::json::Array;
using llvm::json::Object;
constexpr std::size_t MaxCases = 4096;
constexpr std::size_t MaxSites = 64;
constexpr std::size_t MaxFragments = 128;
constexpr std::size_t MaxFamilies = 16;
constexpr std::size_t MaxDigits = 1024;
constexpr std::size_t MaxFileBytes = 8 * 1024 * 1024;
mlir::FailureOr<llvm::DynamicAPInt> integer(const llvm::json::Value& value)
{
    auto text = value.getAsString();
    if (!text || text->empty() || text->size() > MaxDigits) {
        return mlir::failure();
    }
    bool negative = text->starts_with("-");
    llvm::StringRef digits = negative ? text->drop_front() : *text;
    if (digits.empty()) {
        return mlir::failure();
    }
    llvm::DynamicAPInt result(0);
    for (char digit : digits) {
        if (digit < '0' || digit > '9') {
            return mlir::failure();
        }
        result = result * 10 + (digit - '0');
    }
    return negative ? -result : result;
}

mlir::FailureOr<std::size_t> index(const llvm::json::Value& value, std::size_t bound)
{
    auto number = value.getAsInteger();
    if (!number || *number < 0 || static_cast<std::uint64_t>(*number) > bound) {
        return mlir::failure();
    }
    return static_cast<std::size_t>(*number);
}

mlir::LogicalResult readFamilies(const Array& rows, Input& input)
{
    if (rows.size() > MaxFamilies) {
        return mlir::failure();
    }
    for (const auto& row : rows) {
        const auto* array = row.getAsArray();
        if (!array || array->size() != 3) {
            return mlir::failure();
        }
        auto slots = integer((*array)[0]);
        auto stride = integer((*array)[1]);
        auto atoms = index((*array)[2], MaxFragments);
        if (mlir::failed(slots) || mlir::failed(stride) || mlir::failed(atoms)) {
            return mlir::failure();
        }
        input.families.push_back({*slots, *stride, *atoms});
    }
    return mlir::success();
}

mlir::LogicalResult readFragments(const Array& rows, Input& input)
{
    if (rows.size() > MaxFragments) {
        return mlir::failure();
    }
    for (const auto& row : rows) {
        const auto* array = row.getAsArray();
        if (!array || array->size() != 5) {
            return mlir::failure();
        }
        auto site = index((*array)[0], MaxSites);
        auto family = index((*array)[1], MaxFamilies);
        auto atom = index((*array)[2], MaxFragments);
        auto offset = integer((*array)[3]);
        auto mode = index((*array)[4], 4);
        if (mlir::failed(site) || mlir::failed(family) || mlir::failed(atom) || mlir::failed(offset) ||
            mlir::failed(mode)) {
            return mlir::failure();
        }
        input.fragments.push_back({*site, *family, *atom, *offset, static_cast<fs::RotatingAccessMode>(*mode)});
    }
    return mlir::success();
}

mlir::FailureOr<Input> readInput(const Object& object, mlir::MLIRContext& context)
{
    const auto* siteValue = object.get("sites");
    const auto* families = object.getArray("families");
    const auto* fragments = object.getArray("fragments");
    if (!siteValue || !families || !fragments) {
        return mlir::failure();
    }
    auto count = index(*siteValue, MaxSites);
    if (mlir::failed(count)) {
        return mlir::failure();
    }
    Input input;
    const auto* pipes = object.getArray("pipes");
    if (pipes && pipes->size() != *count) {
        return mlir::failure();
    }
    for (std::size_t id = 0; id < *count; ++id) {
        auto pipe = pipes ? index((*pipes)[id], 9) : mlir::FailureOr<std::size_t>(1);
        if (mlir::failed(pipe)) {
            return mlir::failure();
        }
        auto phase = std::make_unique<mlir::pto::CompoundInstanceElement>(
            static_cast<unsigned>(id), llvm::SmallVector<const mlir::pto::BaseMemInfo*>{},
            llvm::SmallVector<const mlir::pto::BaseMemInfo*>{}, static_cast<mlir::pto::PipelineType>(*pipe),
            mlir::OperationName("test.phase", &context));
        input.sites.push_back(phase.get());
        input.owned.push_back(std::move(phase));
    }
    if (mlir::failed(readFamilies(*families, input)) || mlir::failed(readFragments(*fragments, input))) {
        return mlir::failure();
    }
    if (const auto* references = object.getArray("site_refs")) {
        if (references->size() != input.sites.size()) {
            return mlir::failure();
        }
        for (std::size_t id = 0; id < references->size(); ++id) {
            auto reference = index((*references)[id], input.owned.size());
            if (mlir::failed(reference) || *reference >= input.owned.size()) {
                return mlir::failure();
            }
            input.sites[id] = input.owned[*reference].get();
        }
    }
    return input;
}

std::string decimal(const llvm::DynamicAPInt& value)
{
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << value;
    return text;
}

Object dump(const fs::RotatingFootprintAnalysis& analysis)
{
    Array fragments, edges, certificates;
    for (const auto& fragment : analysis.fragments()) {
        fragments.push_back(
            Array{
                fragment.site, fragment.family, fragment.atom, decimal(fragment.offset),
                static_cast<unsigned>(fragment.mode)});
    }
    for (const auto& edge : analysis.generators()) {
        Array witnesses;
        for (const auto& witness : edge.witnesses) {
            witnesses.push_back(
                Array{static_cast<unsigned>(witness.hazard), witness.sourceFragment, witness.consumerFragment});
        }
        edges.push_back(
            Object{
                {"source", edge.source},
                {"consumer", edge.consumer},
                {"distance", decimal(edge.distance)},
                {"witnesses", std::move(witnesses)}});
    }
    for (const auto& certificate : analysis.certificates()) {
        certificates.push_back(Array{certificate.family, decimal(certificate.gcd), decimal(certificate.refresh)});
    }
    return Object{
        {"valid", true},
        {"fragments", std::move(fragments)},
        {"edges", std::move(edges)},
        {"certificates", std::move(certificates)},
        {"bound", decimal(analysis.distanceBound())}};
}

mlir::FailureOr<Object> analyze(const Object& object, mlir::MLIRContext& context)
{
    auto input = readInput(object, context);
    if (mlir::failed(input)) {
        return mlir::failure();
    }
    if (object.getBoolean("symbolic").value_or(false)) {
        return analyzeSymbolic(object, input->sites, context);
    }
    if (object.getBoolean("periodic").value_or(false)) {
        return analyzePeriodic(object, input->sites, input->families, input->fragments, context);
    }
    // Nonempty warm result makes invalid-input atomicity observable.
    mlir::pto::CompoundInstanceElement warmPhase(
        0, {}, {}, mlir::pto::PipelineType::PIPE_V, mlir::OperationName("test.phase", &context));
    const mlir::pto::CompoundInstanceElement* warmSite = &warmPhase;
    fs::RotatingFamily warmFamily{llvm::DynamicAPInt(2), llvm::DynamicAPInt(1), 1};
    fs::RotatingFragment warmFragment{0, 0, 0, llvm::DynamicAPInt(0), fs::RotatingAccessMode::ReadWrite};
    fs::RotatingFootprintAnalysis analysis;
    if (mlir::failed(analysis.build({warmSite}, {warmFamily}, {warmFragment}))) {
        return mlir::failure();
    }
    if (object.getBoolean("null_site").value_or(false) && !input->sites.empty()) {
        input->sites[0] = nullptr;
    }
    if (object.getBoolean("duplicate_site").value_or(false) && input->sites.size() > 1) {
        input->sites[1] = input->sites[0];
    }
    if (mlir::failed(analysis.build(input->sites, input->families, input->fragments))) {
        return Object{
            {"valid", false},
            {"empty", !analysis.valid() && analysis.families().empty() && analysis.sites().empty() &&
                          analysis.fragments().empty() && analysis.generators().empty() &&
                          analysis.certificates().empty() && analysis.distanceBound() == 0}};
    }
    // Exercise rebuilding from views of the previous result, including pointers.
    if (mlir::failed(analysis.build(analysis.sites(), input->families, analysis.fragments()))) {
        return mlir::failure();
    }
    return dump(analysis);
}
} // namespace frontier_test

using namespace frontier_test;

int main(int argc, char** argv)
{
    if (argc != 2) {
        llvm::errs() << "usage: pto-frontier-rotating-test cases.json\n";
        return 1;
    }
    auto buffer = llvm::MemoryBuffer::getFile(argv[1]);
    if (!buffer || (*buffer)->getBufferSize() > MaxFileBytes) {
        llvm::errs() << "cannot read bounded rotating test input\n";
        return 1;
    }
    auto parsed = llvm::json::parse((*buffer)->getBuffer());
    if (!parsed) {
        llvm::errs() << llvm::toString(parsed.takeError()) << "\n";
        return 1;
    }
    const auto* cases = parsed->getAsArray();
    if (!cases || cases->size() > MaxCases) {
        llvm::errs() << "invalid rotating test batch\n";
        return 1;
    }
    mlir::MLIRContext context;
    context.disableMultithreading();
    Array results;
    for (const auto& row : *cases) {
        const auto* object = row.getAsObject();
        auto result = object ? analyze(*object, context) : mlir::FailureOr<Object>(mlir::failure());
        if (mlir::failed(result)) {
            llvm::errs() << "invalid rotating test case encoding\n";
            return 1;
        }
        results.push_back(std::move(*result));
    }
    llvm::outs() << llvm::json::Value(std::move(results)) << "\n";
    return 0;
}
