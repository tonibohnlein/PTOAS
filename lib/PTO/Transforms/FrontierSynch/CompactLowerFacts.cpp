// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/CompactLowerFacts.h"
#include <algorithm>
#include <utility>
namespace mlir::pto::frontiersynch {
namespace {
bool proved(CompactFactProof proof) { return proof == CompactFactProof::Certified; }
bool forward(const PeriodicRecord& edge) { return edge.displacement || edge.source < edge.target; }
struct FamilyState {
    CompactLowerReason reason = CompactLowerReason::Emitted;
    bool mandatoryWriter = false;
};
class Builder {
public:
    explicit Builder(const CompactLowerFactsInput& input) : input(input) {}
    CompactLowerFacts result;
    bool run();
private:
    bool fail(const char* error) { result.error = error; return false; }
    bool validate();
    void families();
    void read(uint32_t id);
    void fullClass(uint32_t id);
    void rotating(uint32_t id);
    void decide(CompactLowerProvenance provenance, CompactLowerReason reason,
                llvm::ArrayRef<PeriodicRecord> records = {});
    const CompactLowerFactsInput& input;
    std::vector<FamilyState> states;
};
bool Builder::validate()
{
    if (input.payloads.size() > UINT32_MAX || input.families.size() > UINT32_MAX ||
        input.writers.size() > UINT32_MAX || input.reads.size() > UINT32_MAX ||
        input.fullClasses.size() > UINT32_MAX || input.rotating.size() > UINT32_MAX ||
        uint64_t(input.reads.size()) + input.fullClasses.size() + 2 * uint64_t(input.rotating.size()) > UINT32_MAX) {
        return fail("compact lower descriptor count exceeds representation");
    }
    for (const auto& family : input.families) {
        if (family.producer >= input.payloads.size()) { return fail("invalid compact lower producer site"); }
    }
    for (const auto& writer : input.writers) {
        if (writer.family >= input.families.size() || writer.site >= input.payloads.size()) {
            return fail("invalid compact lower writer incidence");
        }
    }
    for (const auto& candidate : input.reads) {
        if (candidate.family >= input.families.size() || candidate.site >= input.payloads.size()) {
            return fail("invalid compact lower read incidence");
        }
        if (candidate.alternatives.size() > UINT64_MAX - result.cost.alternatives) {
            return fail("compact lower alternative count exceeds representation");
        }
        result.cost.alternatives += candidate.alternatives.size();
        for (const auto& alternative : candidate.alternatives) {
            if (alternative.maximum && alternative.minimum > *alternative.maximum) {
                return fail("invalid compact lower source distance interval");
            }
        }
    }
    for (const auto& fact : input.fullClasses) {
        if (fact.endpoints.source >= input.payloads.size() || fact.endpoints.target >= input.payloads.size() ||
            (fact.writer != CompactFullWriter::Source && fact.writer != CompactFullWriter::Target)) {
            return fail("invalid compact full-class endpoint binding");
        }
    }
    for (const auto& fact : input.rotating) {
        if (fact.writer >= input.payloads.size() || fact.reader >= input.payloads.size() || !fact.slots) {
            return fail("invalid compact rotating-slot binding");
        }
    }
    return true;
}
void Builder::families()
{
    states.resize(input.families.size());
    result.cost.families = input.families.size();
    for (uint32_t id = 0; id < input.families.size(); ++id) {
        const auto& family = input.families[id];
        auto& state = states[id];
        if (!proved(family.disjointCells)) { state.reason = CompactLowerReason::UnprovedFamily; }
        else if (!proved(family.completeWriters)) { state.reason = CompactLowerReason::IncompleteWriters; }
        else if (!proved(family.requiresStorageOrder)) { state.reason = CompactLowerReason::UnprovedStorageOrder; }
    }
    for (const auto& writer : input.writers) {
        ++result.cost.writerIncidences;
        auto& state = states[writer.family];
        if (state.reason != CompactLowerReason::Emitted) { continue; }
        if (writer.site != input.families[writer.family].producer) {
            state.reason = CompactLowerReason::CompetingWriter;
        } else if (writer.index != CompactWriteIndex::Identity) {
            state.reason = CompactLowerReason::UnprovedIdentity;
        } else {
            state.mandatoryWriter |= proved(writer.mandatoryCellWrite);
        }
    }
    for (auto& state : states) {
        if (state.reason == CompactLowerReason::Emitted && !state.mandatoryWriter) {
            state.reason = CompactLowerReason::MissingMandatoryWriter;
        }
    }
}
void Builder::decide(CompactLowerProvenance provenance, CompactLowerReason reason,
                     llvm::ArrayRef<PeriodicRecord> records)
{
    ++result.cost.candidates;
    result.decisions.push_back({provenance, reason, static_cast<uint32_t>(result.records.size()),
                                static_cast<uint32_t>(records.size())});
    for (auto record : records) {
        result.records.push_back(record);
        result.provenance.push_back(provenance);
    }
}
void Builder::read(uint32_t id)
{
    const auto& candidate = input.reads[id];
    const auto& family = input.families[candidate.family];
    CompactLowerProvenance provenance{CompactLowerFactKind::MustSource, id, family.group};
    auto reason = states[candidate.family].reason;
    if (reason == CompactLowerReason::Emitted &&
        (!proved(candidate.mandatory) || candidate.alternatives.empty())) {
        reason = CompactLowerReason::MissingMandatoryRead;
    }
    uint64_t maximum = 0;
    for (const auto& alternative : candidate.alternatives) {
        if (reason != CompactLowerReason::Emitted) { break; }
        if (!proved(alternative.mandatoryFamilyRead)) { reason = CompactLowerReason::MissingMandatoryRead; }
        else if (!proved(alternative.validSourceIndex) || !alternative.maximum) {
            reason = CompactLowerReason::UnprovedSourceRange;
        } else if (!alternative.minimum && family.producer >= candidate.site) {
            reason = CompactLowerReason::NotForward;
        } else { maximum = std::max(maximum, *alternative.maximum); }
    }
    if (reason != CompactLowerReason::Emitted) { decide(provenance, reason); return; }
    // For j>=maximum the selected true source f(j) is at least j-maximum.
    // Native completion order of this ONE producer site forwards its earlier
    // completion to that required source. No source-ordinal residue assumption.
    const PeriodicRecord record{family.producer, candidate.site, maximum};
    decide(provenance, reason, llvm::ArrayRef<PeriodicRecord>(record));
}
void Builder::fullClass(uint32_t id)
{
    const auto& fact = input.fullClasses[id];
    CompactLowerProvenance provenance{CompactLowerFactKind::FullClass, id, fact.group};
    auto reason = CompactLowerReason::Emitted;
    if (!forward(fact.endpoints)) { reason = CompactLowerReason::NotForward; }
    else if (!proved(fact.mandatoryWriter) || !proved(fact.fullClassWrite)) {
        reason = CompactLowerReason::UnprovedFullWrite;
    } else if (!proved(fact.mandatoryNonemptyAccess)) { reason = CompactLowerReason::MissingMandatoryRead; }
    else if (!proved(fact.samePhysicalClass)) { reason = CompactLowerReason::UnprovedPhysicalClass; }
    else if (!proved(fact.requiresStorageOrder)) { reason = CompactLowerReason::UnprovedStorageOrder; }
    if (reason != CompactLowerReason::Emitted) { decide(provenance, reason); return; }
    decide(provenance, reason, llvm::ArrayRef<PeriodicRecord>(fact.endpoints));
}
void Builder::rotating(uint32_t id)
{
    const auto& fact = input.rotating[id];
    CompactLowerProvenance provenance{CompactLowerFactKind::RotatingFullWrite, id, fact.group};
    auto reason = CompactLowerReason::Emitted;
    if (fact.writer >= fact.reader || input.payloads[fact.writer].pipe == input.payloads[fact.reader].pipe) {
        reason = CompactLowerReason::UnsupportedRotatingBody;
    } else if (!proved(fact.disjointSlots) || !proved(fact.sameSlotMap)) {
        reason = CompactLowerReason::UnprovedPhysicalClass;
    } else if (!proved(fact.mandatoryFullWrite)) { reason = CompactLowerReason::UnprovedFullWrite; }
    else if (!proved(fact.mandatoryNonemptyRead)) { reason = CompactLowerReason::MissingMandatoryRead; }
    else if (!proved(fact.requiresStorageOrder)) { reason = CompactLowerReason::UnprovedStorageOrder; }
    if (reason != CompactLowerReason::Emitted) { decide(provenance, reason); return; }
    uint64_t left = fact.slots, right = fact.stride;
    while (right) {
        const auto remainder = left % right;
        left = right;
        right = remainder;
        ++result.cost.gcdSteps;
    }
    // offset cancels algebraically; neither slot addresses nor stride*i are
    // evaluated. Intervening effects cannot remove these unprotected conflicts.
    const PeriodicRecord records[]{{fact.writer, fact.reader, 0}, {fact.reader, fact.writer, fact.slots / left}};
    decide(provenance, reason, records);
}
bool Builder::run()
{
    if (!validate()) { return false; }
    families();
    for (uint32_t id = 0; id < input.reads.size(); ++id) { read(id); }
    for (uint32_t id = 0; id < input.fullClasses.size(); ++id) { fullClass(id); }
    for (uint32_t id = 0; id < input.rotating.size(); ++id) { rotating(id); }
    return true;
}
} // namespace
CompactLowerFacts buildCompactLowerFacts(const CompactLowerFactsInput& input)
{
    Builder builder(input);
    builder.run(); // Validation completes before any lower record is published.
    return std::move(builder.result);
}
} // namespace mlir::pto::frontiersynch
