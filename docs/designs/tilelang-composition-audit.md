# TileLang composition audit and implementation milestones

This audit concerns the nine prepared ports in
`test/npu_validation/benchmarks/auto_sync`. Their pinned source correspondence,
helper expansion and admitted configurations are recorded in `AUDIT.md` there.
Expert and automatic variants retain the same payloads, buffers and control.
The expert protocol is structural evidence, not a proof of minimum demands,
hardware legality or six-ID feasibility. Device qualification of these nine
ports remains separate. In particular, authored IDs including 6 remain in some
expert ports.

## Storage and protocol structure

| Port | Expert protocol | Required generic analysis |
|---|---|---|
| persistent_gemm | Initial availability notifications; two rotating L1/L0 transfer protocols; first/last inner-use guards; FIX releases the accumulator across task visits; terminal consumption | Composed phase views, finite first/last cases and storage retained across re-entry |
| group_norm | Conditional subblock arm contains accumulation, scalar extraction and a two-bank load/vector/store pipeline | Recursive conditional composition, scalar prerequisites, boundary slicing |
| causal_conv1d_prefill | Loaded sequence metadata precedes a runtime-length banked pipeline; conditional final stores; V barriers inside arithmetic helpers | Region-entry values with actual definitions, multiple affine boundary guards, paired endpoints |
| causal_conv1d_decode | Finite load/vector/store handoffs and conditional state accesses | Existing logical analysis succeeds; shared lifetime allocation remains separate |
| gdn_chunk_cumsum | Bulk load, V-to-S handoff per chunk, inner scalar recurrence, bulk store | Per-byte selectors for visit-owned chunks, scalar prerequisite maps, bulk-boundary composition |
| mhc_head_mix | Bulk reads, scalar replication nest, vector computation and output | Same-allocation read-only prefix plus owned suffix, nested selectors |
| elementwise_pipeline | Initial free-slot notifications, prologue load, next-bank prefetch, add/store/release, final consumption | Existing rotating and sequence route; regression anchor |
| gated_delta_rule | Prefetch with scalar-derived floating predicates and recurrent state | Tractable subregions only until a whole-region Section 5 certificate exists |
| lossless_block_cast | Conditional scalar exponent reductions and scale expansion between transfer/vector stages; several full barriers | Conditional composition and affine selectors, including persistent reduction cells |

## Concrete scalar patterns

GDN has eight chunks of 128 elements. In chunk c, element zero is initialized
before the inner loop. Inner j reads input[128*c+j] and output[128*c+j-1], then
writes output[128*c+j]. For FP32, each outer visit owns a 512-byte output slice.
A later full-buffer store queries writers across all slices. The first element
must select the initializer, not an absent j=0 recurrence occurrence.

mHC reads base[j], j in [0,4), and writes base[4*i+j], i in [1,4). Its first
16 bytes are read-only inside the nest; the subsequent 48 bytes are owned by
three visits. These classes share an allocation but not bytes. Classifying an
entire allocation as either owned or read-only loses this structure.

Lossless block cast's reduction updates two persistent scalar cells indexed by
j/2 across rows. Those cells are not owned by individual rows. Later row
exponents and expanded scales have different affine footprints. Floating-point
values and bitcasts in the computation need not enter an access formula unless
they actually influence control or addresses.

The manual scalar handoffs do not establish a new native hardware rule. GDN,
for example, uses V-to-S before scalar work and V-to-MTE3 after it. The shared
model must independently establish how scalar writes precede the output store.
Compare payload prerequisite profiles only after accounting for those semantics;
never delete a modeled demand merely because it is absent from the expert plan.

## Implemented blockers at the baseline

Baseline `fad0ec0c6`: Existing and expert variants generate C++ for all nine.
Frontier generates C++ for elementwise_pipeline and a logical plan for decode;
decode's sufficient allocation fails within six supplied IDs. Seven ports fail
logical analysis. No new version-four symbolic repeat activates.

1. Regional numerical templates nevertheless require an immediate function
   parent. Finite guarded roots likewise require the function entry block.
2. Sequence collection sends conditionals containing loops to the loop-free
   guarded analyzer instead of recursively analyzing their arms.
3. Storage ownership requires local footprints independent of inner coordinates;
   symbolic storage children cannot compose with siblings.
4. Phase specialization accepts only one guarded rotating child without outer
   payloads. Boundary slicing accepts only one common transition.
5. Arithmetic extraction is whole-function only. Relaxing its function-argument
   parameter check alone would incorrectly forget enclosing-visit correlations.
6. Version-four allocation is missing. The draft only provides the periodic
   construction when inner handoffs form a finite list; general symbolic inner
   allocation remains outside that result.

## Milestones and gates

M1: original-block regional adapters and precise diagnostics, followed by an
explicit regional context for arithmetic. M2: conditional regional composition.
M3: affine per-byte selectors and surrounding bulk accesses. M4: composed phase
specialization and multiple boundary cases. M5: allocation of supported logical
plans without scarcity repair. All remain based on the shared access model;
no precision admission gate or kernel/instruction whitelist is introduced.

Each milestone requires architecture, correctness and performance review.
Small independent unfoldings check closure and emitted matching; corpus runs
record logical and allocation results separately. Track query/port counts,
expression and emitted-code size, numerical visits and phase descriptions.
Runs are serial. The campaign runner no longer uses the old fixed timeout or
1.5-GiB virtual-memory cap; original campaign metadata remains historical.

RegionExpressions already interns nodes and repeated queries already memoize
full event pairs. Further factoring should share boundary-entry/exit vectors
and preparation at the same original cut; cross-cut hoisting needs availability
proofs. The existing three-level phase stress emits about 338,000 scalar
preparation operations and must not be treated as a satisfactory size target.

## First completed change: conditional composition

Regional numeric and finite-guarded adapters now accept roots in their original
nested block. Conditional arms containing compact loops are analyzed recursively
and exported with presence-masked selectors, queries and allocation summaries.
Original cuts are retained. This completes M2 and the original-block part of M1;
it does not implement the regional arithmetic context or affine selectors.

Architecture, correctness and performance reviewers accepted this scope. The
conditional regression checks 15 unfolded closures, empty and nonempty arms,
zero trips, nested guards and unavailable-control transactional rejection.
Existing finite-guarded (64 valuations), sequence (32 closures), mixed-regional
(101 closures) and regional-allocation (15 traces) checks passed. Depths two
through four increase the conditional fixture from 5,924 to 6,284 expressions
and 7,860 to 7,998 emitted operations. This small test does not establish a
general growth bound.

The changed-code prefilter flags four braced compound conditions under
G.FMT.11-CPP because its regular expression matches an inner closing parenthesis
before a Boolean operator. These are inspected false positives; every flagged
body has braces. The actual long-line findings were corrected.
