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

## Follow-up implementation and validation

Three further fixes preserve the shared model:

- Finite strided streaming is proved from the union of physical intervals when
  their enclosing interval is too coarse. The additional check is quadratic in
  the interval count and does not enumerate loop iterations. Its independent
  test checks 45 byte-footprint cases and a zero-trip case.
- An unresolved selected footprint retains its descriptor's canonical GM base
  for the existing alias-policy comparison. Descriptor extents are not treated
  as accessed bytes. Tests cover distinct, equal and selected pointer origins
  under both GM policies.
- Explicit runs inside an original conditional block can insert handoffs at
  that block's cuts. The conditional test additionally checks active/inactive
  prefix execution and the prefix's load-to-compute readiness.

Finite boundary slicing now accepts multiple supported `iv+c` comparisons,
including equality and inequality. The loop has zero lower bound and unit
step, and scalar evolution must prove the affine expression without wrap.
Sorted clipped cut values define the slices; certified slice predicates become
immutable guards in each periodic analysis. This is an incremental part of M4,
not the general affine-boundary or composed-phase construction.

Crossing reduction now asks for the last edge entering the consumer's child.
The prefix ends in an earlier child and the suffix uses the child's internal
reachability. All original native, value and storage links participate, and
links with the candidate's actual endpoint pair are excluded together.
Deduplication follows reduction. Full reachability remains available for
exports without forcing all its intermediate-event circuits into endpoint code.

Preparation reuses available loop-entry values and structurally identical
detached arithmetic at the same original cut. The common insertion regression
checks remapping of endpoint guards, identities, member coordinates and later
SSA uses. Original payload code and cuts do not change.

Architecture and correctness reviewers accepted this subset. Performance review
accepted it as an incremental improvement, explicitly not completion of M4:
the small multi-cut probe still emits 11,257 arithmetic operations before its
loop and 9,085 inside/after it. Same-cut duplication is almost exhausted; the
remaining cost needs better boundary-query and guard factoring. In contrast,
the conditional-depth tests decreased from about 7,900 preparation operations
to about 1,450. These are different fixtures and should not be conflated.

Independent checks passed for 64 multi-cut closures (zero trips, empty/equal
cuts, signed comparisons and first/last singleton slices), 56 prior boundary
cases, 32 sequence closures, 101 mixed-regional closures, 15 allocation traces,
15 conditional closures, 104 guarded-quotient valuations and 86 guarded storage
traces. Billion-trip descriptions have unchanged emitted size. These finite
tests do not establish a uniform proof or device performance.

## Current benchmark outcome and outstanding work

The serial rerun in `.local/nested-regions/tilelang-nine-reviewed` accounts for
all nine ports plus the two prior GEMMs. All expert and Existing variants still
generate C++. Input/tool hashes were unchanged throughout the run. It used no
fixed timeout or virtual-memory cap. This is compiler validation, not device
execution.

| Cases | Frontier outcome |
|---|---|
| Earlier PyPTO GEMM, earlier TileLang GEMM, elementwise_pipeline | Logical insertion, six-ID allocation and C++ generation succeed |
| group_norm | Newly succeeds through logical insertion; sufficient regional allocation fails with six supplied IDs |
| causal_conv1d_decode | Logical insertion succeeds; sufficient regional allocation still fails |
| persistent_gemm | Composed phase interfaces and finite first/last cases remain missing |
| causal_conv1d_prefill | Entry metadata plus affine `L-4*i` boundary predicates need further treatment |
| gdn_chunk_cumsum, mhc_head_mix, lossless_block_cast | Regional arithmetic and per-byte selectors/composition remain missing |
| gated_delta_rule | Whole-kernel tractable-class membership remains unestablished |

Allocation failure is not a proof that six IDs are insufficient. No scarcity
repair was added. M3's regional arithmetic query adapter, affine selectors and
crossing families, M4's composed phases and economical general boundary guards,
and M5's supported allocation extensions remain unfinished. M1 and M2 are complete.
Do not present the current branch as completing all five milestones.

For regional arithmetic, enclosing iteration coordinates must remain shared
context parameters across both endpoints; making each occurrence's outer
coordinates independent would lose correlations. The existing signed arithmetic
emitter and the unsigned modular regional-expression arena also need an exact
adapter. A per-byte selector alone does not construct a finite family of
crossing demands against a bulk access. These are implementation obligations,
not reasons to invent operator-specific recognition or reinterpret unknown
geometry as disjoint storage.

The 52 previously successful logical plans in the selected PTO/PyPTO corpus
also remain successful; 51 still allocate, with the same one allocation failure.
This regression subset is not a new coverage census of the full corpus.

The changed-code prefilter reports two brace-rule matches in guarded recognition
and guard-binding validation. Both statements have braced bodies; the regex
stops at an inner closing parenthesis. They are inspected false positives.

## Regional arithmetic input context

The arithmetic extractor now accepts an original subtree root. Its occurrence
coordinates contain only loops inside that subtree (including a loop root).
Enclosing indices and values available at entry share one SSA parameter tuple
between both endpoints. Proved affine definitions are normalized first;
otherwise an available index/i1 value retains its original SSA binding.
Unsupported values produced inside the region remain unsupported. Enclosing
branch choices belong to the parent; a conditional root retains its own guards.

The result includes incoming scalar prerequisites with their original producer,
consumer and native-order classification. The parent must discharge them. This
contract describes one visit at its actual entry bindings; repeating a summary
still requires substitution/certification, and independent parameter assignments
need not all correspond to realizable executions.

This completes M1 under its original definition, not a composable arithmetic route.
No production dispatcher returns Ready on the strength of this API. Per-byte
first/last selectors, finite crossing families and an exact signed-arithmetic
adapter to regional expressions remain necessary. The first consumer should
analyze a whole scalar nest and compose it with adjacent bulk transfers.

Architecture, correctness and performance reviewers accepted this input layer.
The serial rebuild passed. Its independent test passed 28,038 checks covering
four accepted regional fixtures and two unsupported local-expression fixtures.
Existing tests passed 457,242 basic, 280,500 guarded and 43,008 extended arithmetic
checks, plus 12 arithmetic insertion traces. The brace-rule prefilter also
flags two already-braced compound conditions in ArithmeticResidues.cpp; these
are inspected regex false positives. No device behavior or additional benchmark
acceptance is claimed by this change.

## Milestone acceptance gates

At `840344a46`, the architecture, correctness and performance reviewers each
accepted the full original M1 and M2 contracts, not merely individual patches.
M1 comprises original-block adapters, precise diagnostics and explicit regional
arithmetic input context. M2 comprises recursive compact-arm conditional
composition. The query/selector integration needed to make arithmetic children
composable belongs to M3. Earlier status wording incorrectly assigned that
integration to M1; this correction does not change the original milestone list.

Remaining milestones must each pass all three reviews as a complete milestone
and be committed before proceeding to the next. Successful extraction alone is
not M3 acceptance, finite slices alone are not M4 acceptance, and a sufficient
allocation failure is neither a capacity impossibility proof nor M5 completion.
