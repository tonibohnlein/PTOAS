// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Certain lower demands from supplied access facts on a fixed repeated body.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_COMPACTLOWERFACTS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_COMPACTLOWERFACTS_H
#include "PTO/Transforms/FrontierSynch/PeriodicAnalysis.h"
namespace mlir::pto::frontiersynch {
// Certified is a TRUSTED descriptor-producer assertion, not an inferred fact.
// Every assertion concerns the unchanged shared modeled effects, alias and
// protection policy, occurrence identities and ALL admitted finite prefixes.
// Unknown facts never reject upper construction. Materialized enclosing ranges
// alone establish neither full writes nor mandatory nonempty actual accesses.
enum class CompactFactProof { Unknown, Certified };
enum class CompactWriteIndex { Unknown, Identity, Other };
struct CompactProducerFamily {
    uint64_t group = 0; // Original requirement-group namespace, not a record ID.
    uint32_t producer = 0;
    CompactFactProof disjointCells = CompactFactProof::Unknown;
    // All possible writes/aliases into x[s] appear in writers for this family.
    CompactFactProof completeWriters = CompactFactProof::Unknown;
    // The producer-to-reader dependency needs C->I ordering under the shared
    // storage-protection policy. A protected may-conflict is insufficient.
    CompactFactProof requiresStorageOrder = CompactFactProof::Unknown;
};
struct CompactFamilyWriter {
    uint32_t family = 0, site = 0;
    CompactWriteIndex index = CompactWriteIndex::Unknown;
    // This occurrence actually writes the whole cell x[i] on EVERY visit i.
    // Other possible incidences at this site must still have identity maps.
    CompactFactProof mandatoryCellWrite = CompactFactProof::Unknown;
};
struct CompactReadAlternative {
    // For every j>=maximum, at least one actual read selects x[f(j)] from the
    // named family, 0<=f(j)<=j, with minimum<=j-f(j)<=maximum. The selector need
    // not be affine. These facts describe every choice in this alternative.
    CompactFactProof mandatoryFamilyRead = CompactFactProof::Unknown;
    CompactFactProof validSourceIndex = CompactFactProof::Unknown;
    uint64_t minimum = 0;
    std::optional<uint64_t> maximum;
};
struct CompactFamilyRead {
    uint32_t family = 0, site = 0;
    CompactFactProof mandatory = CompactFactProof::Unknown;
    // Complete, nonempty list of alternatives. Every alternative must pass.
    std::vector<CompactReadAlternative> alternatives;
};
enum class CompactFullWriter { Source, Target };
struct CompactFullClassFact {
    uint64_t group = 0;
    PeriodicRecord endpoints;
    CompactFullWriter writer = CompactFullWriter::Source;
    CompactFactProof mandatoryWriter = CompactFactProof::Unknown;
    CompactFactProof fullClassWrite = CompactFactProof::Unknown;
    CompactFactProof mandatoryNonemptyAccess = CompactFactProof::Unknown;
    // Both named endpoint INSTANCES use the same nonempty physical class.
    CompactFactProof samePhysicalClass = CompactFactProof::Unknown;
    CompactFactProof requiresStorageOrder = CompactFactProof::Unknown;
};
struct CompactRotatingFullWrite {
    uint64_t group = 0;
    uint32_t writer = 0, reader = 0;
    uint64_t slots = 0, stride = 0, offset = 0;
    CompactFactProof disjointSlots = CompactFactProof::Unknown;
    // Both endpoints use mathematical (stride*i+offset) mod slots. The supplier
    // proves agreement with machine semantics; no wrapped multiplication is
    // inferred here. All slots are nonempty and retain identity across visits.
    CompactFactProof sameSlotMap = CompactFactProof::Unknown;
    CompactFactProof mandatoryFullWrite = CompactFactProof::Unknown;
    CompactFactProof mandatoryNonemptyRead = CompactFactProof::Unknown;
    CompactFactProof requiresStorageOrder = CompactFactProof::Unknown;
};
struct CompactLowerFactsInput {
    std::vector<PeriodicPayload> payloads;
    std::vector<CompactProducerFamily> families;
    std::vector<CompactFamilyWriter> writers;
    std::vector<CompactFamilyRead> reads;
    std::vector<CompactFullClassFact> fullClasses;
    std::vector<CompactRotatingFullWrite> rotating;
};
enum class CompactLowerFactKind { MustSource, FullClass, RotatingFullWrite };
enum class CompactLowerReason {
    Emitted, UnprovedFamily, IncompleteWriters, CompetingWriter, UnprovedIdentity,
    MissingMandatoryWriter, MissingMandatoryRead, UnprovedSourceRange, NotForward,
    UnprovedPhysicalClass, UnprovedFullWrite, UnprovedStorageOrder, UnsupportedRotatingBody
};
struct CompactLowerProvenance {
    CompactLowerFactKind kind = CompactLowerFactKind::MustSource;
    uint32_t descriptor = 0; // Original index in reads/fullClasses/rotating.
    uint64_t group = 0;
};
struct CompactLowerDecision {
    CompactLowerProvenance provenance;
    CompactLowerReason reason = CompactLowerReason::UnprovedFamily;
    uint32_t firstRecord = 0, recordCount = 0;
};
struct CompactLowerCost {
    uint64_t families = 0, writerIncidences = 0, candidates = 0, alternatives = 0;
    // Euclidean remainder steps, independent of runtime trips or slot count.
    uint64_t gcdSteps = 0;
};
struct CompactLowerFacts {
    std::string error; // Malformed representation; no records published.
    std::vector<PeriodicRecord> records;
    std::vector<CompactLowerProvenance> provenance; // Parallel, before dedup.
    std::vector<CompactLowerDecision> decisions;
    CompactLowerCost cost;
};
// Expected O(L+s) descriptor work (L includes all alternatives) plus logarithmic
// gcd work per rotating candidate. This excludes proving descriptor assertions.
// No trip/distance/slot unrolling, upper mutation, endpoint export or global
// deduplication. Missing lower proofs retain other candidates and their groups.
// Native prerequisites belong to the later lower quotient construction unchanged.
CompactLowerFacts buildCompactLowerFacts(const CompactLowerFactsInput& input);
} // namespace mlir::pto::frontiersynch
#endif
