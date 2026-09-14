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

The local source audit began on `7584d9dc5` and was rechecked in the current
task tree based on `6d57749a6`:

| PTO operation or option | Observed EmitC lowering | Qualification boundary |
| --- | --- | --- |
| Default `pto.tload` | `TLOAD(dst, src)` | Native instruction and cache behavior still need the pinned PTO headers and device specification. |
| `pto.tload` with `L2Bypass` | `TLOAD<pto::TLoadL2Hint::NotAllocKeep>(dst, src)` | Separate cache premise; no credit inherited from the default load. |
| Default `pto.tstore` | `TSTORE(dst, src)` | Audit independently from phase, atomic, and conversion overloads. |
| Store phase / atomic / ReLU / prequantization | Distinct `TSTORE` template and operand combinations | No blanket store publication capability. |
| Store with FP operand | `TSTORE_FP(dst, src, fp)` with optional template arguments | Separate instruction family. |
| Prefetch and asynchronous prefetch | `TPREFETCH` and `TPREFETCH_ASYNC` | Not ordinary MTE2 reload witnesses. |

`tools/ptoas/ptoas_pipeline.cpp::appendAutoSyncPasses` selects InsertSync,
BufidSync, or barrier-all as alternatives. The barrier-all alternative is not
an additional consistency pass on the InsertSync path. Synchronization precedes
buffer-select resolution. This source inspection does not establish what the
event or load/store intrinsics expand to in a device binary; that remains part
of the device audit below.

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

## 2026-09-15 evidence correction

The isolated evidence package
`gm-publication-isolated-a5a5ad928.tar.gz` has SHA-256
`761abd54c42d952a13d97e21874cac8d3e2fef8b1785b345789b4439a65182a6`.
It pins compiler revision `a5a5ad928`, Ascend 910B2 architecture 2201,
CANN 9.0.0, driver 25.5.0, pto-isa `5a4f74cb`, ccec clang 15.0.5 and
`dav-c220-vec`. The corrections below supersede the bounded-population
isolation claim in that package; the files remain immutable evidence.

The single-generation experiment remains the clean observational baseline. It
has one generation per launch, distinct producer and reload UB buffers, no
loop, no buffer reuse and the same d1/d3 local-buffer handoffs in all arms.
Across five input seeds it ran 2,000 launches per arm per seed, for 10,000
launches per arm. All no-event, directed-event and `PIPE_ALL` arms reported zero
destination mismatches. The seed changes the source values in this harness.
This passing comparison is useful compatibility evidence; it is not a contract
proof and a failing negative control is not required.

The bounded-population experiment does not yet isolate one dependency. For
each adjacent generation, its shared scratch range has
`TSTORE(scratch, g) -> TSTORE(scratch, g+1)` WAW ordering:

| Arm | Completion path before the next scratch store |
| --- | --- |
| A, no event | None established. MTE3 issue order alone does not prove completion of the earlier GM store. |
| B, directed event | The varied d2_g SetFlag/WaitFlag is the only documented source-pipeline completion point after the store. It may therefore supply both the tested RAW and the cross-generation WAW completion. |
| C, `PIPE_ALL` | The substituted barrier is after the store; the synchronization-control documentation says following instructions cannot issue until preceding instructions are committed. |

Consequently, removing d2_g has not been shown to remove exactly one
dependency. A corrected bounded design needs the same completion mechanism in
all arms after d3_g and before `TSTORE(scratch, g+1)`, positioned so it cannot
supply `TSTORE(scratch, g) -> TLOAD(scratch, g)`. That corrected design has not
been run and contributes no device evidence yet.

The bounded driver invokes each arm 1,500 times in each of five runs, with four
unrolled generations per launch: 7,500 launches and 30,000 generations per
arm. Its `seed` argument is passed to `srand` but no random value is consumed;
these are repeated samples, not five distinct input seeds. Both the single and
bounded harnesses check the prefix and suffix guards of the destination buffer.
They initialize guards around source and scratch, but do not copy those buffers
back and therefore do not check their guards.

Event IDs 0 through 7 are available for Atlas A2/A3 in the
[SetFlag/WaitFlag product table](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0270.html).
The bounded emitted program uses IDs 0--7 for the retained MTE2-to-MTE3 d1/d3
handoffs, including IDs 6 and 7 in generation 3. Its tested MTE3-to-MTE2 events
use IDs 0--3. The API defines an event by both HardEvent direction and numeric
ID, so those are distinct keys in this test context.

The exact ordinary DMA contract remains unresolved. The inspected lowering is
default `TSTORE` through the `pto_copy_ubuf_to_gm` family and default `TLOAD`
through `tload_common`/`pto_copy_gm_to_ubuf`, on the pinned A2/A3 target. The
[synchronization-control documentation](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0179.html)
states that SetFlag follows completion of preceding source-pipeline reads and
writes and that WaitFlag blocks later target-pipeline instructions. The
[Fill API](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0891.html)
requires MTE3-to-MTE2 synchronization after its GM initialization before later
UB use. Neither source states that completion of the ordinary
`pto_copy_ubuf_to_gm` variant on architecture 2201 is a coherence point
observable by the subsequent ordinary `pto_copy_gm_to_ubuf` load of the same
address. Fill has its own implementation and premises, so its rule is not
silently generalized to default TSTORE/TLOAD.

The package preserves the PTO and generated C++ identities, including
`generated/single.existing.cpp` SHA-256
`e72292f6071db9add776d7d86db55aa391d0b678ccfaaaeeb733068e766e20af`
and `generated/bounded.existing.cpp` SHA-256
`da9e6da2157c0c78edcd57f88ff5bbf6c01d95bf6e320a4e0f82ec500b13d485`.
It does not contain the compiled device image, native disassembly or the pinned
PTO-ISA header snapshot. A different local PTO-ISA checkout cannot substitute
for revision `5a4f74cb`. The locally reproducible evidence therefore ends at
the generated intrinsic family and target defines; it cannot establish the
exact native instruction or the point at which its completion becomes
observable to the following MTE2 load.

The scalar/atomic `dcci` plus `dsb` sequence in `syncall_soft.hpp` remains
separate evidence for a scalar DCache path. The package's NDDMA-cache and
L2-control observations constrain possible DMA cache mechanisms but do not
establish visibility on architecture 2201. The precise open contract question
is therefore:

> On architecture 2201, does completion of default `pto_copy_ubuf_to_gm`
> before an MTE3-to-MTE2 SetFlag, followed by the matching WaitFlag, guarantee
> that default `pto_copy_gm_to_ubuf` on the same core observes that write; and,
> if not, what DMA-specific maintenance or load/store variant is required?

Until an authoritative guarantee answers that question for these exact
variants, the compiler's same-range publication refusal remains enabled.
