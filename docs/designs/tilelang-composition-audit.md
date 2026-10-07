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

## M3 implementation and review evidence

The arithmetic regional adapter now constructs exact per-byte first/last writer
and reader selectors, occurrence selectors, and all-event queries from the
shared physical access relations. Enclosing coordinates remain shared entry
bindings. Bounded physical support is partitioned into access-equivalent atoms
when unit byte coefficients certify common cut residues; otherwise the adapter
retains individual bytes. This cost depends on physical support, not dynamic
trip count. Unbounded support still needs a symbolic boundary adapter.

The signed integer recipe adapter retains exact projection and constant-division
quotient locals. It does not enumerate a joint modulus just to represent a
constant division. Original cuts and matched endpoint identities are preserved.
Crossings are consolidated to guarded extrema and reduced through memoized
last-crossing queries. Preparation is shared only at an original dominating cut
where all required values are available; unavailable sharing leaves the ordinary
endpoint preparation intact.

Independent tests now compare twenty emitted command closures with authored
byte conflicts, including overlapping f32 reads and upper-half f16 writes,
zero/negative bounds, persistent cells, recurrence and owned suffixes. A
billion-trip parameter does not change emitted code. Existing integer relation,
arithmetic demand/selector, expression, first-site prerequisite, 32 sequence,
101 mixed, 15 conditional and ten arithmetic-insertion checks pass.

The parameterized recurrence and prefix fixtures now require 1,496 and 1,417
preparation operations (initially 42,429 and 104,919); persistent and overlapping
width fixtures require 192 and 258. These are preparation counts, not hardware
synchronization counts. GDN produces a logical plan in 8.02 seconds at 121 MB
peak RSS; mHC takes 0.15 seconds at 112 MB. The arithmetic route is still slower
than simple structural routes; these measurements establish no near-linear
complexity claim. Allocation of these new plans belongs to M5.

An allocation-interface regression was fixed by retaining the detached sequence
plan while trying the existing whole-function difference route, and replacing it
only when that route supplies an allocation certificate. Plans already exporting
regional allocation skip this retry. The regression subset again has 52 logical
successes and 51 allocation successes, matching its baseline.

Lossless block cast now passes sequence analysis and endpoint preparation through
six arithmetic regions: 38,923 represented bytes, 8,521 cells, 8,723 ports and 98
preparation operations. Analysis takes 122.18 seconds and peaks at 168 MB. The
later scalar GM access reuses the existing global-independence and disjoint-visit
proof, with deferred access selectors retained for enclosing re-entry. Six
focused fixtures pass under both alias policies, including overlapping peers,
repeated writes, read-only effects and original index-cast bindings.

Architecture, correctness and performance reviewers accepted the full original
M3 contract. Physical allocation remains M5; arbitrary unbounded symbolic storage
interfaces are not claimed. The changed-code checker reports brace-rule false
positives for already-braced compound conditions; actual width findings were
fixed. Serial builds, source review and independent tests pass. Production
logical insertion also succeeds for lossless block cast; all three M3 target
ports now produce logical plans. The milestone is accepted for commit.

## M4 implementation and review evidence — 2026-10-07

The earlier tables are historical checkpoints. Architecture, correctness and
performance reviewers have accepted M4 under its original contract. These
changes do not establish M5 allocation coverage or device correctness.

### Constructed interfaces and route order

Composed phase analysis now retains explicit outer payloads, recursively selected
conditional arms and compact inner regions. It first proves periodicity of the
complete physical maps, including offsets, extents and buffer selections. Each
phase exports the original occurrence identities, storage extrema, all-event
queries and endpoint recipes. A child is reused across visits only after the
required control and storage invariance checks.

Compact body composition is tried first. Original explicit runs with concrete
phase geometry reuse the shared access normalizer, storage scan and rank reducer;
this expands no loop. If compact composition fails, the existing finite-template
machinery can expand numeric inner visits into one finite body word. Geometry
uses a certified phase representative; control uses separate certified interval
facts and actual inner coordinates. Geometry substitution cannot determine loop
bounds or branch outcomes. Original cuts and inner coordinate tuples survive
both routes, and numeric expansion is charged in the result. A discharged writer
still needs an exported re-entry interface. Reader discharge requires the shared
whole-input proof, so a writer in a sibling region cannot disappear.

Signed affine boundary predicates can define several cuts. Cuts are sorted and
clipped to the actual iteration domain. Certified intervals containing at most
one visit avoid building an inter-visit quotient. Their selector coordinates are
canonicalized only under their original nonempty guard; zero trips, swapped
cuts and partial periods retain their original semantics. No nonnegative input
length is inferred from a kernel's intended use.

Speculative compact construction owns its expression suffix and query caches.
Rejected attempts destroy those caches before rolling back the arena; phase
normalizers with cached expression IDs are local to the attempt. A successful
finite fallback does not inherit the rejected route's diagnostics.

### Composition and cost accounting

Storage and native boundary lists are normalized before bridge products. Each
list denotes guarded alternatives for one extremum; simultaneous alternatives
must name the same occurrence. Selected coordinates retain their original-port
provenance, so queries distribute through existing boundary occurrences instead
of asking children to rediscover arbitrary synthetic coordinates. Immutable
consolidated generators answer reachability while cover guards are reduced.
Native same-pipe queries, absent endpoints and proved incompatible guards are
resolved before requesting more expensive child queries. Shared min-plus
expressions preserve exact distance and guard semantics.

The quotient contracts complete within-body paths and retains only crossing
targets as vertices. Every across-visit query charges its first crossing and
retains its final body path; guards remain correlated. This avoids closing the
already-transitive zero-distance body relation again. The lazy quotient is not
an O(P²)-memory algorithm. For P retained target ports, its memoized
Floyd states `(source, target, level)` can occupy O(P³) entries. The current
ordered-map implementation adds logarithmic lookup cost, giving O(P³ log P)
map work in the worst case, apart from child-query and circuit costs. The base
matrices themselves occupy O(P²). Laziness avoids unused queries; it does not
improve this worst-case bound. No bound independent of nesting depth is claimed.

### Recorded validation

These are compiler and independent graph-oracle checks, not device runs. The
final artifacts under `.local/nested-regions/` record the tested binary and
pinned inputs; earlier measurement logs retain their own intermediate binaries.

- `m4-regressions/results.json`: all eleven regression groups pass, covering
  arithmetic relations, demands, selectors, expressions, incoming prerequisites,
  composition, sequences, mixed and conditional regions, and insertion.
- `m4-validation/affine-final.log`: sixty independently unfolded affine-boundary
  closures pass.
- `m4-validation/composed-final.log`: composed byte closures, zero trips,
  partial periods, scalar inputs and three nesting levels pass. Finite moving
  partial footprints and an external scalar producer are included.
- `m4-validation/expressions-minplus-fixed.log`: 27,264 phased event queries
  with storage selectors, 123,552 repeated-region event pairs, ownership checks
  and expression checks pass. These counts include the added symbolic singleton
  cases; earlier logs contain smaller totals.
- `corpus-m4-final/results.json`: the selected regression corpus retains
  52 logical successes and 51 allocation successes. This is not a full corpus
  coverage census.

`tilelang-nine-m4-final/results.json` covers all nine benchmark ports and both earlier
GEMMs. All eleven expert and Existing variants generate C++. Frontier outcomes
in that snapshot are:

| Cases | Logical insertion | Physical allocation / C++ |
|---|---|---|
| Earlier PyPTO GEMM, earlier TileLang GEMM, elementwise_pipeline | Success | Success |
| persistent_gemm | Newly succeeds | Allocation export remains M5 |
| gdn_chunk_cumsum, mhc_head_mix, lossless_block_cast | Success | Allocation export remains M5 |
| group_norm, causal_conv1d_decode | Success | Current sufficient allocation fails within supplied capacity |
| causal_conv1d_prefill, gated_delta_rule | Not accepted by a complete current route | Not attempted |

The sufficient allocation failures do not prove ID scarcity. Prefill still lacks
a range/no-wrap certificate for its signed-length and derived unsigned-trip
arithmetic; assuming a nonnegative length would change the supplied contract.
Gated delta rule still has unsupported control/domain obligations. Neither is
silently reclassified as a successful tractable instance.

### Performance scope and acceptance

Persistent GEMM now has a prepared logical result through generic phase and finite
body interfaces. `m4-validation/persistent-finite.json` records 459 expression
nodes, 10,245 preparation operations, 3,288 charged numeric visits and unchanged
input. Its accompanying timing is 0.12 seconds with 83,744 KiB peak RSS. The
production logical pass also succeeds in the eleven-case snapshot. These are
preparation counts, not SET/WAIT counts or a performance measurement on hardware.

The original three-level stress now emits 10,934 preparation operations, versus
89,768 at the M3 baseline. `m4-validation/triple-final.json` and `.time` record
0.07 seconds and 68,460 KiB peak RSS. This is the same fixture; the older plan's
338,000 count predates intermediate improvements.

The larger parameterized first/last fixture remains costly.
`m4-validation/composed-native-order.json` records 312,281 expression nodes and
318,333 preparation operations, with zero numeric expansion. Its accompanying
log records 3.97 seconds and 427,268 KiB peak RSS. Earlier M4 attempts exceeded
three million preparation operations. This improvement does not imply uniformly
small emitted code, a near-linear algorithm, or measured device performance.
The performance reviewer accepted the original milestone contract with this
limitation explicit. Further cross-region query factoring remains future work.

All three reviewers accepted the complete M4 scope. Temporary profiling code
was removed. The changed-code prefilter's brace findings were inspected: each
flagged control body is braced, and the matcher stops at an inner parenthesis.
Actual line-width findings were fixed. M5 still needs allocation exports for the
new logical plans; no scarcity repair or new whole-kernel class is claimed here.
