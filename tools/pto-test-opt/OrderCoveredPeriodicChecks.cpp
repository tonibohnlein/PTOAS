// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independently check physical conflicts and ownership of retained-order proofs.
#include "PTO/Transforms/FrontierSynch/FrontierAnalysis.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Structured/OrderCoveredPeriodic.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
namespace fs = pto::frontiersynch;
namespace {
std::string render(Operation* operation)
{
    std::string text;
    llvm::raw_string_ostream stream(text);
    operation->print(stream);
    return text;
}
// The fixture's two GM bases are distinct under MayNotAlias. Each store writes
// 128 row intervals, while every load writes and every store reads the same UB.
bool globalOverlap(int64_t a, int64_t b, bool small, bool physical)
{
    auto offset = [small, physical](int64_t iv) {
        auto wrapped = llvm::APInt(64, static_cast<uint64_t>(iv)) * llvm::APInt(64, physical ? 64 : (small ? 1 : 256));
        return wrapped.isNegative() ? uint64_t(0) : wrapped.getZExtValue();
    };
    const auto x = offset(a), y = offset(b);
    const uint64_t rows = small || physical ? 1 : 128, stride = physical ? 64 : (small ? 128 : 5120);
    const uint64_t width = physical ? 16 : (small ? 8 : 256);
    for (uint64_t row = 0; row < rows; ++row) {
        for (uint64_t other = 0; other < rows; ++other) {
            const auto left = static_cast<unsigned __int128>(x) + row * stride;
            const auto right = static_cast<unsigned __int128>(y) + other * stride;
            if (left < right + width && right < left + width) { return true; }
        }
    }
    return false;
}
bool concreteOrder(const fs::PeriodicAnalysis& periodic, bool small, bool physical = false)
{
    const std::size_t sites = physical ? 4 : 2;
    if (periodic.payloads.size() != sites) { return false; }
    const uint32_t expectedPipes[] = {4, physical ? 1U : 5U, 1, 5};
    for (std::size_t i = 0; i < sites; ++i) {
        if (periodic.payloads[i].pipe != expectedPipes[i]) { return false; }
    }
    // Three independent 32-byte cells: A=[0,64), B=[32,96), C=[0,64).
    const uint32_t reads[] = {0, 3, 6, 3}, writes[] = {3, 6, 3, 0};
    for (int64_t lower : {INT64_C(-41), INT64_C(-1), INT64_C(0), INT64_C(3),
                          INT64_C(1) << 55, (INT64_C(1) << 56) - 21}) {
        for (uint64_t trips = 0; trips <= 6; ++trips) {
            const std::size_t count = trips * sites, events = count * 2;
            std::vector<std::vector<bool>> graph(events, std::vector<bool>(events));
            for (std::size_t a = 0; a < count; ++a) {
                graph[2 * a][2 * a] = graph[2 * a + 1][2 * a + 1] = true;
                graph[2 * a][2 * a + 1] = true;
                for (std::size_t b = a + 1; b < count; ++b) {
                    if (periodic.payloads[a % sites].pipe == periodic.payloads[b % sites].pipe) {
                        graph[2 * a][2 * b] = true;
                        graph[2 * a + 1][2 * b + 1] = true;
                    }
                    const bool ub = physical ?
                        ((writes[a % sites] & (reads[b % sites] | writes[b % sites])) ||
                         (reads[a % sites] & writes[b % sites])) : !(a % 2 && b % 2);
                    const bool gm = a % sites == sites - 1 && b % sites == sites - 1 &&
                        globalOverlap(lower + static_cast<int64_t>(a / sites) * 20,
                            lower + static_cast<int64_t>(b / sites) * 20, small, physical);
                    if (ub || gm) { graph[2 * a + 1][2 * b] = true; }
                }
            }
            for (std::size_t k = 0; k < events; ++k) {
                for (std::size_t a = 0; a < events; ++a) {
                    for (std::size_t b = 0; b < events; ++b) {
                        graph[a][b] = graph[a][b] || (graph[a][k] && graph[k][b]);
                    }
                }
            }
            for (std::size_t a = 0; a < events; ++a) {
                for (std::size_t b = 0; b < events; ++b) {
                    auto result = periodic.eventPrecedes(
                        {static_cast<uint32_t>((a / 2) % sites), a % 2 ? fs::PeriodicEventKind::Completion :
                            fs::PeriodicEventKind::Start}, a / (2 * sites),
                        {static_cast<uint32_t>((b / 2) % sites), b % 2 ? fs::PeriodicEventKind::Completion :
                            fs::PeriodicEventKind::Start}, b / (2 * sites), count);
                    if (result.error != fs::PeriodicQueryError::None || result.value != graph[a][b]) { return false; }
                }
            }
        }
    }
    return true;
}
bool sameProtectedOrder(const fs::OrderCoveredPeriodicResult& proof)
{
    auto loop = proof.retained.loop;
    auto recognized = fs::recognizeRotating(loop, *proof.index, *proof.input, proof.input->accesses());
    if (recognized.state != fs::RecognitionState::Applicable) {
        llvm::errs() << "ACC ordinary recognition: " << fs::recognitionName(recognized.state) << "\n";
        return false;
    }
    auto ordinary = fs::analyzeRotating(loop, *proof.index, *proof.input, recognized, false);
    if (!ordinary.error.empty() || !ordinary.extraction.protectedHazards ||
        proof.retained.extraction.protectedHazards != ordinary.extraction.protectedHazards ||
        ordinary.periodic.payloads.size() != proof.retained.periodic.payloads.size()) {
        llvm::errs() << "ACC protection counts " << ordinary.extraction.protectedHazards << " "
                     << proof.retained.extraction.protectedHazards << " error=" << ordinary.error << "\n";
        return false;
    }
    const auto sites = ordinary.periodic.payloads.size();
    for (uint64_t trips = 0; trips <= 4; ++trips) {
        const uint64_t count = sites * trips;
        for (uint64_t a = 0; a < 2 * count; ++a) {
            for (uint64_t b = 0; b < 2 * count; ++b) {
                fs::PeriodicEvent first{static_cast<uint32_t>((a / 2) % sites),
                    a % 2 ? fs::PeriodicEventKind::Completion : fs::PeriodicEventKind::Start};
                fs::PeriodicEvent last{static_cast<uint32_t>((b / 2) % sites),
                    b % 2 ? fs::PeriodicEventKind::Completion : fs::PeriodicEventKind::Start};
                const auto original = ordinary.periodic.eventPrecedes(first, a / (2 * sites),
                    last, b / (2 * sites), count);
                const auto materialized = proof.retained.periodic.eventPrecedes(first, a / (2 * sites),
                    last, b / (2 * sites), count);
                if (original.error != fs::PeriodicQueryError::None ||
                    materialized.error != fs::PeriodicQueryError::None || original.value != materialized.value) {
                    return false;
                }
            }
        }
    }
    return true;
}
bool freshPlans(const fs::PreparedLogicalPlan& a, const fs::PreparedLogicalPlan& b)
{
    llvm::DenseSet<Operation*> operations;
    for (const auto& stage : a.preparation) {
        if (stage.code->getParentOp()) { return false; }
        for (auto& operation : *stage.code) { operations.insert(&operation); }
    }
    for (const auto& stage : b.preparation) {
        if (stage.code->getParentOp()) { return false; }
        for (auto& operation : *stage.code) {
            if (operations.contains(&operation)) { return false; }
            for (auto value : operation.getOperands()) {
                if (operations.contains(value.getDefiningOp())) { return false; }
            }
        }
    }
    return true;
}
std::string renderPlan(const fs::PreparedLogicalPlan& plan)
{
    std::string text;
    llvm::raw_string_ostream stream(text);
    for (const auto& stage : plan.preparation) { stage.code->print(stream); }
    return text;
}
} // namespace
LogicalResult checkOrderCoveredPeriodic(func::FuncOp function, pto::GMAliasPolicy policy)
{
    auto fail = [&](StringRef detail) -> LogicalResult {
        function.emitError("order-covered periodic check: ") << detail;
        return failure();
    };
    const auto before = render(function);
    const bool physical = function->hasAttr("test.order_covered_static_cells");
    fs::FrontierAnalysis session(function);
    if (failed(session.initialize(policy))) { return failure(); }
    scf::ForOp loop;
    function.walk([&](scf::ForOp candidate) { loop = candidate; });
    if (!loop) { return fail("missing loop"); }
    std::size_t region = 0;
    for (const auto& node : session.result()->nodes) {
        if (node.anchor == loop.getOperation()) { region = &node - session.result()->nodes.data(); break; }
    }
    auto input = session.sharedInput();
    auto index = std::make_shared<fs::PhaseIndex>();
    if (failed(index->build(function, *input))) { return failure(); }
    // TGetVal itself is synchronous PIPE_S. Exercise the separate completion-
    // dependent control gate with a test-owned phase view, without mutating IR
    // or the shared access model. Only the outside scalar producer is replaced.
    std::unique_ptr<pto::CompoundInstanceElement> asynchronous;
    if (function.getSymName() == "payload_bound") {
        SmallVector<const pto::CompoundInstanceElement*> phases(input->instructions());
        for (auto& phase : phases) {
            if (phase->elementOp->getName().getStringRef() == "pto.tgetval") {
                asynchronous = std::make_unique<pto::CompoundInstanceElement>(*phase);
                asynchronous->kPipeValue = pto::PipelineType::PIPE_V;
                phase = asynchronous.get();
            }
        }
        if (!asynchronous || failed(index->build(function, phases)) || !index->needsValuePrerequisite(loop)) {
            return fail("completion-dependent bound premise was not exercised");
        }
    }
    if (function.getSymName() == "dynamic_local") {
        SmallVector<std::size_t> local;
        for (auto [id, effect] : llvm::enumerate(input->accesses().effects())) {
            if (effect.memory->scope != pto::AddressSpace::GM) { local.push_back(id); }
        }
        if (input->accesses().hasMaterializedCellPartition(local)) {
            return fail("dynamic local fixture unexpectedly has a materialized partition");
        }
    }
    uint64_t reductions = 0;
    std::string error;
    auto proof = fs::analyzeOrderCoveredPeriodic(loop, region, input, index, reductions, error);
    if (function->hasAttr("test.order_covered_acc")) {
        if (!proof || reductions != 1 || !sameProtectedOrder(*proof) || render(function) != before) {
            llvm::errs() << "ACC factory diagnostic: " << error << "\n";
            return fail("ACC materialized protection differs from existing producer");
        }
        llvm::outs() << "order-covered accepted ACC protection parity\n";
        return success();
    }
    const bool rejected = function->hasAttr("test.order_covered_rejected") || policy == pto::GMAliasPolicy::MayAlias;
    if (rejected) {
        if (proof || error.empty() || render(function) != before) { return fail("negative proof accepted"); }
        if (policy == pto::GMAliasPolicy::MayNotAlias) {
            StringRef expected = function.getSymName() == "missing_ub" ? "does not cover" :
                (function.getSymName() == "outside" ? "outside its original owner" : "independent outermost loop");
            if (auto requested = function->getAttrOfType<StringAttr>("test.order_covered_obligation")) {
                expected = requested.getValue();
            }
            if (!StringRef(error).contains(expected)) {
                return fail("negative fixture missed its intended obligation");
            }
        }
        llvm::outs() << "order-covered rejected " << function.getSymName() << ": " << error << "\n";
        return success();
    }
    if (!proof || reductions != 1 || proof->omittedEffects.size() != 2 || proof->coveredPairs.size() != 1 ||
        !proof->applies(input.get(), index.get(), loop) || proof->applies(input.get(), nullptr, loop) ||
        !concreteOrder(proof->retained.periodic, function->hasAttr("test.order_covered_strong"), physical)) {
        return fail("physical conflict oracle or context");
    }
    if (physical && proof->retained.fragments.size() != 10) { return fail("partial cell partition changed"); }
    auto child = session.minimumDemands(region, {true});
    auto root = session.minimumDemands(0, {true});
    if (child.status != fs::AnalysisStatus::Ready || root.status != fs::AnalysisStatus::Ready ||
        !child.mathematical || !root.mathematical || !child.mathematical->orderCoveredDemands ||
        root.mathematical->orderCoveredDemands != child.mathematical->orderCoveredDemands ||
        child.mathematical->backend != "order-covered-periodic" || child.available.selectors ||
        !child.available.queries || child.obligations.empty() || session.constructionCounts().rotatingReductions != 1 ||
        session.constructionCounts().logicalPreparations || session.constructionCounts().allocationExports) {
        return fail("session selection or ownership");
    }
    auto certificates = session.certifyRegions();
    if (certificates.empty() || certificates[0].selectedClass != "periodic-required-order" ||
        certificates[0].representation != "order-covered-periodic") { return fail("certificate taxonomy"); }
    auto retained = child.mathematical;
    if (function->hasAttr("test.order_covered_strong")) {
        auto stronger = session.minimumDemands(region, {true, true});
        if (!stronger.mathematical || session.constructionCounts().rotatingReductions != 1 ||
            session.minimumDemands(region).mathematical != retained || child.available.selectors) {
            return fail("strong export retry lost exact owner");
        }
    }
    auto first = session.prepareLogical(root), second = session.prepareLogical(root);
    if (failed(first) || failed(second) || !freshPlans(**first, **second)) { return fail("detached plan ownership"); }
    const auto surviving = renderPlan(**second);
    first->reset();
    if (renderPlan(**second) != surviving || failed(session.attachAllocation(root, **second)) ||
        !(*second)->allocationCertificate || session.constructionCounts().rotatingReductions != 1) {
        return fail("retained allocation or independent plan lifetime");
    }
    if (failed(session.initialize(pto::GMAliasPolicy::MayAlias)) ||
        succeeded(session.prepareLogical(child)) ||
        retained->orderCoveredDemands->applies(session.input(), retained->orderCoveredDemands->index.get(), loop)) {
        return fail("alias context isolation");
    }
    auto alias = retained->rotatingDemands;
    std::weak_ptr<const pto::SyncInput> weakInput = retained->input;
    std::weak_ptr<const fs::PhaseIndex> weakIndex = retained->orderCoveredDemands->index;
    child = {}; root = {}; certificates.clear(); proof.reset(); input.reset(); index.reset();
    second->reset(); retained.reset(); session.invalidate();
    if (weakInput.expired() || weakIndex.expired() ||
        !concreteOrder(alias->periodic, function->hasAttr("test.order_covered_strong"), physical) ||
        render(function) != before) {
        return fail("alias-only owner lifetime or source mutation");
    }
    alias.reset();
    if (!weakInput.expired() || !weakIndex.expired()) { return fail("unexpected retained owner cycle"); }
    llvm::outs() << "order-covered accepted " << function.getSymName() << "\n";
    return success();
}
