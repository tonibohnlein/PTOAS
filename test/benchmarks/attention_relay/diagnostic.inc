// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
namespace {
using namespace mlir;
using namespace mlir::pto;
using J = llvm::json::Object;
using A = llvm::json::Array;
const char* diagnosticPipeName(oahs::Pipe p)
{
    static const char* n[] = {"S", "V", "M", "MTE1", "MTE2", "MTE3", "FIX"};
    return n[unsigned(p)];
}
int64_t idn(std::size_t x) { return x == oahs::NoAnalysisId ? -1 : int64_t(x); }
template <class T>
A ids(const T& values)
{
    A a;
    for (auto x : values)
        a.push_back(idn(x));
    return a;
}
J diagnosticModel(const oahs::NativeAnalysis& in)
{
    J out;
    A ops, cells, cuts, loops;
    for (std::size_t i = 0; i < in.program.operations.size(); ++i) {
        const auto& o = in.program.operations[i];
        A accesses;
        for (const auto& a : o.accesses)
            accesses.push_back(
                J{{"cell", a.cell}, {"read", a.read}, {"write", a.write}, {"definite", a.definiteWrite}});
        std::string loc;
        llvm::raw_string_ostream stream(loc);
        in.phases[i]->getLoc().print(stream);
        ops.push_back(
            J{{"id", idn(i)},
              {"pipe", diagnosticPipeName(o.pipe)},
              {"name", in.phases[i]->getName().getStringRef()},
              {"loc", loc},
              {"accesses", std::move(accesses)}});
    }
    for (std::size_t i = 0; i < in.program.cells.size(); ++i) {
        const auto& c = in.program.cells[i];
        A ranges;
        for (auto r : c.ranges)
            ranges.push_back(A{int64_t(r.first), int64_t(r.second)});
        cells.push_back(
            J{{"id", idn(i)},
              {"space", c.addressSpace},
              {"unknown", c.unknownRange},
              {"storage", unsigned(c.storage)},
              {"ranges", std::move(ranges)},
              {"origins", ids(c.storageOrigins)},
              {"provenance", c.provenance}});
    }
    if (in.program.observed) {
        for (std::size_t i = 0; i < in.program.observed->sites.size(); ++i) {
            const auto& s = in.program.observed->sites[i];
            std::string loc, name;
            if (i < in.cuts.size() && in.cuts[i]) {
                llvm::raw_string_ostream st(loc);
                in.cuts[i]->getLoc().print(st);
                name = in.cuts[i]->getName().getStringRef().str();
            }
            cuts.push_back(
                J{{"id", idn(i)},
                  {"op", idn(s.operation)},
                  {"observation", idn(s.observation)},
                  {"next", ids(s.successors)},
                  {"loc", loc},
                  {"anchor", name}});
        }
        for (const auto& l : in.program.observed->loops)
            loops.push_back(
                J{{"owner", idn(l.owner)},
                  {"entry", idn(l.entry)},
                  {"exit", idn(l.exit)},
                  {"bodyEntry", idn(l.bodyEntry)},
                  {"nonempty", l.atLeastOnce},
                  {"sites", ids(l.sites)}});
    }
    A notes;
    for (const auto& n : in.observationNotes)
        notes.push_back(n);
    out["operations"] = std::move(ops);
    out["cells"] = std::move(cells);
    out["cuts"] = std::move(cuts);
    out["loops"] = std::move(loops);
    out["notes"] = std::move(notes);
    return out;
}
void diagnosticPlan(llvm::StringRef name, J model, const oahs::SelectedPlan& p)
{
    A decisions, endpoints, channels, updates;
    for (std::size_t i = 0; i < p.decisions.size(); ++i) {
        const auto& d = p.decisions[i];
        A requirements;
        for (const auto& r : d.required)
            requirements.push_back(
                J{{"cell", r.cell},
                  {"source", diagnosticPipeName(r.source)},
                  {"sourceWrite", r.sourceWrite},
                  {"consumer", idn(r.consumer)},
                  {"consumerRead", r.consumerRead},
                  {"consumerWrite", r.consumerWrite}});
        decisions.push_back(
            J{{"id", idn(i)},
              {"consumer", idn(d.consumer)},
              {"publication", idn(d.publication)},
              {"source", diagnosticPipeName(d.source)},
              {"observer", diagnosticPipeName(d.observer)},
              {"common", d.commonCut},
              {"stage", unsigned(d.stage)},
              {"required", std::move(requirements)},
              {"endpoints", ids(d.endpoints)}});
    }
    for (const auto& e : p.ledger)
        endpoints.push_back(
            J{{"id", idn(e.id)},
              {"cut", idn(e.cut)},
              {"kind", unsigned(e.command.kind)},
              {"source", diagnosticPipeName(e.command.source)},
              {"observer", diagnosticPipeName(e.command.observer)},
              {"key", e.command.key},
              {"purpose", unsigned(e.purpose)},
              {"request", idn(e.request)},
              {"acknowledges", idn(e.acknowledges)}});
    for (const auto& c : p.channels)
        channels.push_back(
            J{{"cells", ids(c.cells)},
              {"source", diagnosticPipeName(c.source)},
              {"observer", diagnosticPipeName(c.observer)},
              {"key", c.key},
              {"publications", ids(c.publications)},
              {"acquisitions", ids(c.acquisitions)},
              {"owner", idn(c.owner)}});
    for (const auto& u : p.updates)
        updates.push_back(
            J{{"evaluations", int64_t(u.siteEvaluations)},
              {"finalizedQueries", int64_t(u.finalizedQueries)},
              {"cuts", ids(u.changedCuts)},
              {"contextual", u.contextual}});
    A words;
    for (const auto& word : p.commands) {
        A cs;
        for (const auto& c : word)
            cs.push_back(
                J{{"kind", unsigned(c.kind)},
                  {"source", diagnosticPipeName(c.source)},
                  {"observer", diagnosticPipeName(c.observer)},
                  {"key", c.key}});
        words.push_back(std::move(cs));
    }
    model["commands"] = std::move(words);
    model["success"] = p.success;
    model["function"] = name;
    model["decisions"] = std::move(decisions);
    model["endpoints"] = std::move(endpoints);
    model["channels"] = std::move(channels);
    model["updates"] = std::move(updates);
    llvm::errs() << "DIAGNOSTIC_JSON " << llvm::json::Value(std::move(model)) << "\n";
}
} // namespace