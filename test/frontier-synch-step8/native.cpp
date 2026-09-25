// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
// Native import/public-API regressions. Build and run with the repository's
// LLVM/MLIR toolchain; the standalone core oracle is NOT a substitute for this.
#include "PTO/Transforms/FrontierSynch/ProgramAnalysis.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <sstream>

using namespace mlir;
using namespace mlir::pto;
namespace fs = mlir::pto::frontiersynch;
static uint64_t assertions = 0, cases = 0;
static void check(bool condition, int line = __builtin_LINE())
{
    ++assertions;
    if (!condition) {
        std::cerr << "D2 native assertion failed at " << line << '\n';
        std::exit(1);
    }
}
static std::string print(mlir::Operation* op)
{
    std::string text;
    llvm::raw_string_ostream out(text);
    op->print(out);
    return text;
}
struct Case {
    unsigned modulus = 2, lower = 0, step = 1, shift = 0, upper = 19;
    bool carried = false, noninjective = false, fixedInterference = false;
    bool interveningWriter = false, independentPool = false, optionalReader = false;
};
static std::string program(const Case& c)
{
    std::ostringstream s;
    s << R"pto(
!vec = !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>
module attributes {pto.target_arch = "a3"} {
  func.func @periodic(%input: !pto.ptr<f32>, %output: !pto.ptr<f32>, %g: i1) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %three = arith.constant 3 : index
    %sixteen = arith.constant 16 : index
    %stride = arith.constant 1024 : index
    %other_base = arith.constant 16384 : index
    %fixed_addr = arith.constant 0 : i64
    %fixed = pto.alloc_tile addr = %fixed_addr : !vec
    %in_view = pto.make_tensor_view %input, shape = [%sixteen, %sixteen], strides = [%sixteen, %one] : !pto.tensor_view<?x?xf32>
    %out_view = pto.make_tensor_view %output, shape = [%sixteen, %sixteen], strides = [%sixteen, %one] : !pto.tensor_view<?x?xf32>
    %in_part = pto.partition_view %in_view, offsets = [%zero, %zero], sizes = [%sixteen, %sixteen] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<16x16xf32>
    %out_part = pto.partition_view %out_view, offsets = [%zero, %zero], sizes = [%sixteen, %sixteen] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<16x16xf32>
)pto";
    s << "    %lower = arith.constant " << c.lower << " : index\n"
      << "    %step = arith.constant " << c.step << " : index\n"
      << "    %mod = arith.constant " << c.modulus << " : index\n"
      << "    %shift = arith.constant " << c.shift << " : index\n"
      << "    %initial = arith.constant " << c.lower % c.modulus << " : index\n"
      << "    %upper = arith.constant " << c.upper << " : index\n";
    if (c.carried) {
        s << "    %unused = scf.for %i = %lower to %upper step %step iter_args(%slot = %initial) -> (index) {\n";
    } else {
        s << "    scf.for %i = %lower to %upper step %step {\n"
          << "      %slot = arith.remui %i, %mod : index\n";
    }
    s << "      %physical = arith." << (c.noninjective ? "andi %slot, %one" : "addi %slot, %zero") << " : index\n"
      << R"pto(      %w_offset = arith.muli %physical, %stride : index
      %w_addr = arith.index_cast %w_offset : index to i64
      %writer_bank = pto.alloc_tile addr = %w_addr : !vec
)pto";
    if (c.shift) {
        s << R"pto(      %r_slot0 = arith.addi %physical, %shift : index
      %r_slot = arith.remui %r_slot0, %mod : index
      %r_offset = arith.muli %r_slot, %stride : index
      %r_addr = arith.index_cast %r_offset : index to i64
      %reader_bank = pto.alloc_tile addr = %r_addr : !vec
)pto";
    }
    s << "      pto.tload ins(%in_part : !pto.partition_tensor_view<16x16xf32>) outs(%writer_bank : !vec) loc(\"writer\")\n";
    if (c.interveningWriter) {
        s << "      pto.tload ins(%in_part : !pto.partition_tensor_view<16x16xf32>) outs(%writer_bank : !vec) loc(\"intervening\")\n";
    }
    if (c.fixedInterference) {
        s << "      pto.tload ins(%in_part : !pto.partition_tensor_view<16x16xf32>) outs(%fixed : !vec) loc(\"fixed\")\n";
    }
    const auto bank = c.shift ? "%reader_bank" : "%writer_bank";
    if (c.optionalReader) {
        s << "      scf.if %g {\n";
    }
    s << "      pto.tstore ins(" << bank << " : !vec) outs(%out_part : !pto.partition_tensor_view<16x16xf32>) loc(\"reader1\")\n";
    if (c.optionalReader) {
        s << "      }\n";
    }
    s << "      pto.tstore ins(" << bank << " : !vec) outs(%out_part : !pto.partition_tensor_view<16x16xf32>) loc(\"reader2\")\n";
    if (c.independentPool) {
        s << R"pto(      %other_slot = arith.remui %i, %three : index
      %other_offset = arith.muli %other_slot, %stride : index
      %other_index = arith.addi %other_offset, %other_base : index
      %other_addr = arith.index_cast %other_index : index to i64
      %other_bank = pto.alloc_tile addr = %other_addr : !vec
      pto.tload ins(%in_part : !pto.partition_tensor_view<16x16xf32>) outs(%other_bank : !vec) loc("other_writer")
      pto.tstore ins(%other_bank : !vec) outs(%out_part : !pto.partition_tensor_view<16x16xf32>) loc("other_reader")
)pto";
    }
    if (c.carried) {
        s << "      %next0 = arith.addi %slot, %step : index\n"
          << "      %next = arith.remui %next0, %mod : index\n"
          << "      scf.yield %next : index\n";
    }
    s << "    }\n    return\n  }\n}\n";
    return s.str();
}
static std::size_t operation(const fs::OriginalStructure& p, llvm::StringRef name)
{
    std::size_t found = fs::NoControlId;
    for (std::size_t i = 0; i < p.operations.size(); ++i) {
        auto loc = dyn_cast<NameLoc>(p.operations[i].instruction->elementOp->getLoc());
        if (loc && loc.getName().getValue() == name) {
            check(found == fs::NoControlId);
            found = i;
        }
    }
    check(found != fs::NoControlId);
    return found;
}
static std::vector<fs::OriginalObligationId> localIds(
    const fs::ProgramAnalysis& analysis, std::size_t source, std::size_t target, fs::OriginalObligationKey::Kind kind)
{
    std::vector<fs::OriginalObligationId> out;
    const auto& p = analysis.structure();
    for (auto id : analysis.obligationsAt(p.operations[target].original)) {
        const auto* family = analysis.obligations().get(id);
        if (!family || family->key.kind != kind) {
            continue;
        }
        bool local = false;
        for (const auto& access : p.operations[source].accesses) {
            local |= access.cell == family->key.cell && access.memory &&
                     access.memory->scope != AddressSpace::GM && access.memory->scope != AddressSpace::Zero &&
                     (kind == fs::OriginalObligationKey::Kind::WAR ? access.read : access.write);
        }
        const fs::OriginalObligationId obligation{id, {source, false}};
        if (local && analysis.obligations().membership(obligation).status != fs::ObligationMembership::Status::Excluded) {
            out.push_back(obligation);
        }
    }
    return out;
}
static bool guard(const fs::PeriodicEndpointDomain& endpoint, uint64_t address, uint64_t ordinal, uint64_t trips)
{
    if (endpoint.empty || ordinal >= trips || (endpoint.testsSelector && endpoint.selectorEquals != address)) {
        return false;
    }
    if (!endpoint.boundaryTest) {
        return true;
    }
    const auto& atom = *endpoint.boundaryTest;
    check(atom.kind == fs::ObservationAtom::LoopHasPrevious || atom.kind == fs::ObservationAtom::LoopHasNext);
    const bool test = atom.kind == fs::ObservationAtom::LoopHasPrevious ? ordinal >= atom.parameter :
                                                                              atom.parameter < trips - ordinal;
    return test == bool(atom.value);
}
static void compareFinite(const fs::PeriodicUseCorrespondence& result, const Case& c, bool war, bool self = false)
{
    check(result.exact && !result.links.empty());
    check(result.initialNeedsIncomingInterface && result.finalKeepsEnclosingContinuation);
    const auto period = c.modulus / std::gcd(c.modulus, c.step);
    check(result.links.size() == period);
    // The original IR's actual finite bound, including its bypass. The separate
    // core oracle exhausts the symbolic domain over many additional lengths.
    const uint64_t trips = c.upper <= c.lower ? 0 : (c.upper - c.lower - 1) / c.step + 1;
    {
        std::vector<uint64_t> writers, readers;
        for (uint64_t i = 0; i < trips; ++i) {
            writers.push_back(((c.lower + i * c.step) % c.modulus) * 1024);
            readers.push_back((((c.lower + i * c.step) % c.modulus + c.shift) % c.modulus) * 1024);
        }
        const auto& sources = war ? readers : writers;
        const auto& targets = war || self ? writers : readers;
        for (uint64_t i = 0; i < trips; ++i) {
            std::optional<uint64_t> previous;
            for (uint64_t j = 0; j < trips; ++j) {
                const bool before = j < i || (j == i && !war && !self);
                if (before && sources[j] == targets[i]) {
                    previous = j;
                }
            }
            const auto found = std::find_if(result.links.begin(), result.links.end(), [&](const auto& link) {
                return result.banks[link.occurrence.bank].begin == targets[i];
            });
            check(found != result.links.end());
            check(found->occurrence.previous(i, trips) == previous);
            check(guard(found->predecessor, targets[i], i, trips) == bool(previous));
            check(guard(found->initial, targets[i], i, trips) == !previous);
            std::optional<uint64_t> next;
            for (uint64_t j = 0; j < trips; ++j) {
                const bool after = j > i || (j == i && !war && !self);
                if (after && targets[j] == sources[i]) {
                    next = j;
                    break;
                }
            }
            const auto nextLink = std::find_if(result.links.begin(), result.links.end(), [&](const auto& link) {
                return result.banks[link.occurrence.bank].begin == sources[i];
            });
            check(nextLink != result.links.end());
            check(nextLink->occurrence.next(i, trips) == next);
            check(guard(nextLink->successor, sources[i], i, trips) == bool(next));
            check(guard(nextLink->final, sources[i], i, trips) == !next);
        }
    }
    for (const auto& link : result.links) {
        check(link.predecessor.qualified() && link.successor.qualified() && link.initial.qualified() && link.final.qualified());
    }
}
static void run(const Case& c, MLIRContext& context)
{
    ++cases;
    auto module = parseSourceString<ModuleOp>(program(c), &context);
    check(module && succeeded(verify(*module)));
    const auto before = print(module->getOperation());
    auto function = module->lookupSymbol<func::FuncOp>("periodic");
    SyncInput input;
    check(succeeded(input.build(function)));
    fs::OriginalStructure p;
    check(succeeded(fs::importOriginalStructure(function, input, p)));
    const auto writer = operation(p, "writer"), reader1 = operation(p, "reader1"), reader2 = operation(p, "reader2");
    fs::ProgramAnalysis analysis(input, std::move(p));
    check(analysis.complete());
    const bool positive = !c.noninjective && !c.fixedInterference && !c.interveningWriter && !c.optionalReader;
    const auto raw = localIds(analysis, writer, reader2, fs::OriginalObligationKey::Kind::RAW);
    check(!raw.empty());
    // One canonical family retains both local and prior-visit obligations.
    std::set<std::tuple<std::size_t, std::size_t, fs::OriginalObligationKey::Kind>> familiesByRole;
    for (auto id : analysis.obligationsAt(analysis.structure().operations[writer].original)) {
        const auto* family = analysis.obligations().get(id);
        if (family->key.kind != fs::OriginalObligationKey::Kind::Typed) {
            check(familiesByRole.emplace(family->key.consumerOperation, family->key.cell, family->key.kind).second);
        }
    }
    const auto local = analysis.fixedSourcesFor(raw.front().family);
    check(local->sources.complete);
    check(local->sources.frame.kind == fs::FactoredUseFrame::Kind::ForBody);
    const auto pairsBefore = analysis.lifetimes().stats().requirements;
    auto result = analysis.periodicUseFor(raw.front());
    check(result.exact == positive);
    check(analysis.lifetimes().stats().requirements == pairsBefore); // no legacy-pair expansion
    const auto families = analysis.obligations().stats().families;
    check(analysis.obligations().get(raw.front().family)->id == raw.front().family);
    if (positive) {
        compareFinite(result, c, false);
        const auto first = localIds(analysis, writer, reader1, fs::OriginalObligationKey::Kind::RAW);
        check(!first.empty());
        compareFinite(analysis.periodicUseFor(first.front()), c, false);
        check(!(first.front() == raw.front())); // separate deadlines/obligations
        const auto returns = localIds(analysis, reader1, writer, fs::OriginalObligationKey::Kind::WAR);
        check(!returns.empty());
        compareFinite(analysis.periodicUseFor(returns.front()), c, true);
        const auto waw = localIds(analysis, writer, writer, fs::OriginalObligationKey::Kind::WAW);
        check(!waw.empty());
        compareFinite(analysis.periodicUseFor(waw.front()), c, false, true);
        check(fs::resolveOriginalCut(analysis.structure(), result.entry).has_value());
        check(fs::resolveOriginalCut(analysis.structure(), result.exit).has_value());
        auto wrong = result.interval;
        wrong.query.includeStoppingAccess = true;
        check(!analysis.occurrences().periodic(wrong, true, false).exact);
        wrong = result.interval;
        wrong.query.occurrence.stopVisit = fs::OriginalOccurrenceContext::StopVisit::FirstReach;
        check(!analysis.occurrences().periodic(wrong, true, false).exact);
        wrong = result.interval;
        wrong.query.continuationOwner = fs::NoControlId;
        check(!analysis.occurrences().periodic(wrong, true, false).exact);
        wrong = result.interval;
        wrong.query.occurrence.incomingInterface = 17;
        check(!analysis.occurrences().periodic(wrong, true, false).exact);
        check(!analysis.periodicUseFor({raw.front().family, fs::ObligationOrigin::entry()}).exact);
        // The compatibility consumer uses precisely the same local relation.
        bool legacy = false;
        const auto& requests = analysis.requirementsAt(reader2);
        for (std::size_t i = 0; i < requests.size(); ++i) {
            if (requests[i].relationship.source.operation == writer &&
                requests[i].relationship.cell == result.interval.query.selector.cell &&
                requests[i].relationship.kind == fs::StorageRelationship::RAW) {
                const auto& interpreted = analysis.interpretAt(reader2, i);
                check(interpreted.occurrence.status == fs::OriginalOccurrenceInterpretation::Status::PeriodicRoles);
                check(interpreted.occurrence.periodic && interpreted.occurrence.periodic->exact);
                legacy = true;
            }
        }
        check(legacy);
        if (c.independentPool) {
            auto ow = operation(analysis.structure(), "other_writer"), or_ = operation(analysis.structure(), "other_reader");
            const auto ids = localIds(analysis, or_, ow, fs::OriginalObligationKey::Kind::WAR);
            check(!ids.empty());
            const auto other = analysis.periodicUseFor(ids.front());
            check(other.exact && other.links.size() == 3 && result.links.size() == 2);
            check(other.links.size() + result.links.size() == 5);
        }
    } else {
        check(!result.reason.empty());
        // Failed D2 qualification does not delete or rewrite may footprints.
        check(!analysis.structure().physicalAddresses.empty());
        bool retainedFootprint = false;
        for (const auto& relation : analysis.structure().physicalAddresses) {
            retainedFootprint |= !relation.addresses.empty();
        }
        check(retainedFootprint);
    }
    check(analysis.obligations().stats().families == families);
    check(before == print(module->getOperation()));
    // Explicit snapshot invalidation: no previously cached success survives.
    auto oldInterval = result.interval;
    ++const_cast<fs::OriginalStructure&>(analysis.structure()).version.revision;
    check(!analysis.occurrences().periodic(oldInterval, true, false).exact);
    check(!analysis.periodicUseFor(raw.front()).exact);
}
int main()
{
    DialectRegistry dialects;
    dialects.insert<PTODialect, func::FuncDialect, arith::ArithDialect, scf::SCFDialect, cf::ControlFlowDialect>();
    MLIRContext context(dialects);
    for (unsigned n : {1, 2, 3, 5}) {
        Case c;
        c.modulus = n;
        run(c, context);
        c.carried = true;
        run(c, context);
    }
    for (unsigned upper : {0, 1, 2, 3, 4, 6}) {
        Case boundary;
        boundary.modulus = 3;
        boundary.upper = upper;
        run(boundary, context);
    }
    Case shifted;
    shifted.modulus = 3;
    shifted.shift = 1;
    run(shifted, context);
    shifted.carried = true;
    run(shifted, context);
    Case strided;
    strided.modulus = 6;
    strided.lower = 2;
    strided.step = 2;
    run(strided, context);
    strided.carried = true;
    run(strided, context);
    Case noninjective;
    noninjective.modulus = 4;
    noninjective.noninjective = true;
    run(noninjective, context);
    Case fixed;
    fixed.fixedInterference = true;
    run(fixed, context);
    Case interfering;
    interfering.interveningWriter = true;
    run(interfering, context);
    Case independent;
    independent.independentPool = true;
    run(independent, context);
    Case optional;
    optional.optionalReader = true;
    run(optional, context);
    std::cout << "PASS D2 native: " << cases << " imported PTO cases, " << assertions
              << " semantic assertions; original IR unchanged\n";
}
