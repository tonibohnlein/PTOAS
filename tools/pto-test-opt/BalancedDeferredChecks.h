// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Included after the independent command-graph oracle in CompactBoundingPipelineChecks.cpp.
#ifndef PTO_BALANCED_DEFERRED_CHECKS_H
#define PTO_BALANCED_DEFERRED_CHECKS_H
bool balancedArchiveRejects(StringRef saved, MLIRContext* context, StringRef name)
{
    for (unsigned corruption = 0; corruption < 11; ++corruption) {
        auto module = parseSourceString<ModuleOp>(saved, context);
        if (!module) { return false; }
        auto function = module->lookupSymbol<func::FuncOp>(name);
        auto data = function->getAttrOfType<DictionaryAttr>("pto.deferred_family");
        auto rows = data.getAs<ArrayAttr>("partition_choices");
        if (!rows || rows.empty()) { return false; }
        SmallVector<Attribute> changed(rows.begin(), rows.end());
        NamedAttrList row(cast<DictionaryAttr>(changed.front()));
        auto cuts = cast<DictionaryAttr>(changed.front()).getAs<DenseI64ArrayAttr>("cuts");
        SmallVector<int64_t> altered(cuts.asArrayRef());
        Builder b(context);
        if (corruption == 0) { row.set("family", b.getI64IntegerAttr(UINT32_MAX)); }
        else if (corruption == 1) { row.set("side", b.getI64IntegerAttr(2)); }
        else if (corruption == 2) { row.set("loop", b.getI64IntegerAttr(INT64_MAX)); }
        else if (corruption == 3) { altered.back() = altered.front(); }
        else if (corruption == 4) { altered.pop_back(); }
        else if (corruption == 5) { std::swap(altered.front(), altered.back()); }
        else if (corruption == 9) {
            // A valid side value is still wrong ownership for these exact cuts.
            const auto side = cast<DictionaryAttr>(changed.front()).getAs<IntegerAttr>("side");
            if (!side || (side.getInt() != 0 && side.getInt() != 1)) { return false; }
            row.set("side", b.getI64IntegerAttr(1 - side.getInt()));
        } else if (corruption == 10) {
            // Keep a valid sealed source reference, but select an original
            // operation that cannot own an SCF loop partition.
            auto sources = data.getAs<ArrayAttr>("sources");
            if (!sources) { return false; }
            std::optional<int64_t> wrongOwner;
            for (auto [i, raw] : llvm::enumerate(sources)) {
                auto source = dyn_cast<DictionaryAttr>(raw);
                auto sourceName = source ? source.getAs<StringAttr>("name") : StringAttr{};
                if (sourceName && sourceName.getValue() != "scf.for") {
                    wrongOwner = static_cast<int64_t>(i); break;
                }
            }
            if (!wrongOwner) { return false; }
            row.set("loop", b.getI64IntegerAttr(*wrongOwner));
        }
        if (corruption >= 3 && corruption <= 5) {
            row.set("cuts", b.getDenseI64ArrayAttr(altered));
        }
        changed.front() = row.getDictionary(context);
        if (corruption == 6) { changed.push_back(changed.front()); }
        if (corruption == 7) { changed.erase(changed.begin()); }
        NamedAttrList fields(data);
        fields.set("partition_choices", b.getArrayAttr(changed));
        if (corruption == 8) { fields.erase("partition_choices"); }
        // Re-seal to exercise structural proof checks, not merely stale metadata.
        fields.erase("protocol");
        std::string text;
        llvm::raw_string_ostream out(text); fields.getDictionary(context).print(out); out.flush();
        llvm::SHA256 hash; hash.update(text);
        fields.set("protocol", b.getStringAttr(llvm::toHex(hash.final())));
        function->setAttr("pto.deferred_family", fields.getDictionary(context));
        const auto before = render(function);
        ScopedDiagnosticHandler suppress(context, [](Diagnostic&) { return success(); });
        if (succeeded(fs::allocatePhysicalEventIds(function, {0, 1, 2, 3, 4, 5})) ||
            render(function) != before) { return false; }
    }
    return true;
}
bool checkBalancedDeferred(func::FuncOp original)
{
    auto* context = original.getContext();
    const auto name = original.getSymName();
    auto source = original->getParentOfType<ModuleOp>();
    if (!source) { return false; }
    const int64_t bounds[] = {-3, -2, 0, 1, 2, 5, 8};
    for (bool nonunit : {false, true}) {
        std::vector<Execution> logical, physical;
        std::string combined, saved;
        for (unsigned route = 0; route < 4; ++route) {
            auto fail = [&](StringRef stage) {
                llvm::errs() << "balanced owned route=" << route << " nonunit=" << nonunit
                             << " stage=" << stage << "\n";
                return false;
            };
            OwningOpRef<ModuleOp> module(cast<ModuleOp>(source->clone()));
            auto function = module->lookupSymbol<func::FuncOp>(name);
            {
                auto loop = *function.getOps<scf::ForOp>().begin();
                if (nonunit) {
                    OpBuilder b(loop);
                    loop.setLowerBound(b.create<arith::ConstantIndexOp>(loop.getLoc(), -2));
                    loop.setStep(b.create<arith::ConstantIndexOp>(loop.getLoc(), 3));
                }
                pto::SyncInput input;
                if (failed(input.build(function))) { return fail("input"); }
                fs::PhaseIndex index;
                if (failed(index.build(function, input))) { return fail("phase index"); }
                auto body = fs::recognizeBalancedCompactBody(loop, input, index);
                if (!body.error.empty() || body.slots.size() != 2) { return fail("body"); }
                auto graph = fs::analyzePeriodicDemands({{body.slots[0].pipe}, {body.slots[1].pipe}},
                    {{0, 1, 0}, {1, 0, 1}});
                std::string error;
                auto plan = fs::prepareBalancedCompactInsertion(function, body, graph, 97, error);
                if (failed(plan) || !(*plan)->familyRecipe || !(*plan)->endpoints.empty() ||
                    !(*plan)->preparation.empty()) { return fail("lazy preparation"); }
                fs::RegionalAnalysis region;
                region.expressions = std::make_shared<fs::RegionExpressions>();
                region.anchors.resize(2); region.occurrenceLoops.assign(2, loop);
                auto domain = fs::CountedLoop::get(loop);
                if (!domain || !fs::attachCompactBoundingAllocation(region, graph,
                    domain->trips(*region.expressions), **plan, error) || !(*plan)->preparation.empty() ||
                    !(*plan)->endpoints.empty()) { return fail("allocation without SSA"); }
                if (route == 2 && failed(fs::planPhysicalFamilyOutput(function, **plan, {0, 1, 2, 3, 4, 5}))) {
                    return fail("combined mapping");
                }
                (*plan)->deferredOutput = route == 3;
                if (failed(fs::insertLogicalSynchronization(function, **plan)) || failed(verify(function))) {
                    return fail("insertion");
                }
            } // Destroy analysis, recipes and borrowed producer owners before saved decode.
            if (route == 3) {
                auto data = function->getAttrOfType<DictionaryAttr>("pto.deferred_family");
                if (!data || data.getAs<IntegerAttr>("version").getInt() != 4) { return fail("version four"); }
                llvm::raw_string_ostream text(saved); module->print(text); text.flush();
                if (!balancedArchiveRejects(saved, context, name)) { return fail("malformed archive"); }
                module = parseSourceString<ModuleOp>(saved, context);
                if (!module) { return fail("roundtrip parse"); }
                function = module->lookupSymbol<func::FuncOp>(name);
            }
            if ((route == 1 || route == 3) &&
                failed(fs::allocatePhysicalEventIds(function, {0, 1, 2, 3, 4, 5}))) {
                return fail("physical allocation");
            }
            if (failed(verify(function))) { return fail("verification"); }
            if (route == 2) { combined = render(function); }
            if (route == 3 && render(function) != combined) { return fail("saved/combined exact construction"); }
            pto::SyncInput executed;
            if (failed(executed.build(function))) { return fail("executed input"); }
            for (auto bound : bounds) {
                function->setAttr("test.trace_arguments", DenseI64ArrayAttr::get(context, {bound}));
                auto run = decode(traceStructuredLogicalInsertion(function, executed.instructions()));
                if (!commandGraph(run) || !storageCovered(run)) { return fail("independent command graph"); }
                const int64_t lower = nonunit ? -2 : 0, step = nonunit ? 3 : 1;
                const auto trips = bound <= lower ? 0 : (bound - lower + step - 1) / step;
                if (run.payloads.size() != static_cast<uint64_t>(2 * trips)) { return fail("visit coverage"); }
                if (route == 0) {
                    std::map<std::pair<int64_t, std::string>, uint64_t> visits;
                    for (const auto& event : run.events) {
                        if (event.kind != "set" && event.kind != "wait") { continue; }
                        if (run.payloads.empty()) { return fail("notification without visit"); }
                        const auto writerPipe = run.events[run.payloads.front()].pipe;
                        const int64_t record = event.source == writerPipe ? 0 : 1;
                        const auto ordinal = visits[{record, event.kind}]++;
                        const auto identity = "97:" + std::to_string(record) + ":" +
                            std::to_string(ordinal) + ":[" + std::to_string(record) + "]";
                        if (event.identity != identity) { return fail("original source identity"); }
                    }
                    for (int64_t record : {0, 1}) {
                        const auto count = static_cast<uint64_t>(std::max<int64_t>(trips - record, 0));
                        if (visits[{record, "set"}] != count || visits[{record, "wait"}] != count) {
                            return fail("notification coverage");
                        }
                    }
                    logical.push_back(std::move(run));
                }
                else {
                    const auto index = static_cast<std::size_t>(llvm::find(bounds, bound) - std::begin(bounds));
                    if (!sameOrder(logical[index], run)) { return fail("logical/physical order"); }
                    if (route == 1) { physical.push_back(std::move(run)); }
                    else {
                        const auto& expected = physical[index];
                        if (run.events.size() != expected.events.size()) { return fail("command count"); }
                        for (std::size_t i = 0; i < run.events.size(); ++i) {
                            const auto& a = run.events[i]; const auto& b = expected.events[i];
                            if (a.kind != b.kind || a.pipe != b.pipe || a.source != b.source ||
                                a.target != b.target || a.identity != b.identity || a.occurrence != b.occurrence) {
                                return fail("exact command sequence");
                            }
                        }
                    }
                }
            }
        }
    }
    return true;
}
#endif
