# MTE3 → MTE2 GM publication qualification — unresolved

Status: **OPEN**. Compiler acceptance and device qualification are separate.
No publication capability is enabled by this record.

The [CANN 8.5 SetFlag/WaitFlag reference](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0270.html)
was inspected on 2026-09-14. It lists MTE3_MTE2 and describes intra-core
pipeline synchronization. It does not establish a recipe for the exact PTO
native instruction/cache combinations below. Event direction support is
therefore insufficient to admit overlapping GM store/reload in composition.

Local lowering premises to audit:

- `lib/PTO/Transforms/PTOToEmitC/LoadStore/TLoad.cpp`: default `TLOAD`, and
  `cache_policy=L2Bypass` lowering to `TLoadL2Hint::NotAllocKeep`.
- `lib/PTO/Transforms/PTOToEmitC/LoadStore/TStore.cpp`: normal store,
  phase/atomic/ReLU/prequantization overloads, and the distinct FP path.
- `StructuredSyncCore.cpp`: event availability is independent of a publication
  guarantee. MMAD accumulator and UnitFlag ownership profiles supply none.
- Record any lowering or consistency pass that changes the instructions,
  cache hints, fences, or event placement in the tested binary.

Task for the device qualification agent:

1. Pin device model/stepping, firmware, driver, CANN version, PTO headers and
   compiler revision. Record hashes of original PTO, emitted C++, compiled
   binary, relevant native instruction disassembly, and launch/alias contracts.
2. Audit the authoritative instruction/cache specification for each proposed
   same-core store/reload recipe. Define exactly which write generation and
   consumer context it publishes, and how later writes invalidate that credit.
3. Construct overlapping GM MTE3 store → MTE2 reload tests with cold and warm
   caches, repeated generations, alternating buffers, multiple consumers,
   branches and loop backedges. Include a branch where publication is absent;
   the join must retain only universally established publication facts.
4. Compare unsynchronized, directed event only, and each specification-backed
   cache/fence/event recipe. Delete or reorder each claimed necessary mechanism
   in negative controls. Record every run, iterations, seeds, and mismatches.
   A stress run without observed failures alone is not a contract proof.
5. Test default and L2-bypass loads separately. Explicitly exclude or separately
   qualify peer, atomic, FP, partial-phase and private-resource variants.
6. Return a narrow capability table with specification citations, exact native
   variants, required mechanisms, generation invalidation and join rules,
   negative controls, device results, and remaining uncertainty. If the evidence
   is insufficient, retain this OPEN status and the compiler's explicit refusal.

Acceptance requires both a supported hardware contract and independent native
reconstruction of the emitted recipe. Device timing is reported separately from
compiler timing and host reconstruction; no timing ratio enables publication.
