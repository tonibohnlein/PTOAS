# Compact demand construction: Persistent GEMM checkpoint

This checkpoint qualifies Section 5 construction and exposes its first concrete
representation gaps. It does not change demand generation, allocation policy,
matching, or command placement. The proposed contract remains
`frontier-paper-sections54-55-design.md` (user-owned, not edited here).

## Pinned experiment

Base source: `e54d19e91cdf7c54a6b6738f0f0bd1b67cdb9f44`, plus the instrumentation
and test changes in this checkpoint. The companion JSON records source-file and
binary hashes; raw artifacts are in `.local/compact-demand-experiment/`.

Input: `test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto`,
SHA256 `cc890b426ae89da77223b93a5ba4ad0b6f03d418f22f9bde65aea05fcde6324e`.
The target is A3 cube; GM alias policy is `may-not-alias`, assuming the three GM
arguments are disjoint. The requested physical pool is IDs 0–5.

The earlier baseline receipt separates logical insertion (1.719 s), split
allocation (0.216 s), C++ generation (0.620 s), and combined output (1.519 s).
These are single production runs, not minimum-demand-generation times. This
checkpoint reuses those baseline stage outcomes. No device compilation or
execution was performed. The earlier physical trace uses IDs 0–3; this proves
feasibility for that protocol, not minimum capacity.

```bash
OPT=<candidate-build>/tools/pto-test-opt/pto-test-opt
PROBE=<candidate-build>/tools/pto-test-opt/pto-sync-input-test
INPUT=test/npu_validation/benchmarks/auto_sync/prepared/persistent_gemm/input.pto
"$OPT" --mlir-disable-threading \
  --pto-frontier-analysis="gm-alias=may-not-alias allocation-ids=0,1,2,3,4,5 profile-stages=true" \
  "$INPUT" -o persistent-logical.pto 2>persistent-profile.log
"$OPT" --mlir-disable-threading \
  --pto-frontier-allocate="eligible-ids=0,1,2,3,4,5" \
  persistent-logical.pto -o persistent-physical.pto
python test/lit/pto/Inputs/check_compact_persistent_contract.py \
  --optimizer "$OPT" --trace-tool "$PROBE" --input "$INPUT" --output section5-checks
python test/lit/pto/Inputs/profile_compact_persistent.py \
  --optimizer "$OPT" --input "$INPUT" --output section5-scaling
```

`profile-stages=true` now emits `frontier-retained-inventory`. It reads existing
owners and metadata only: no recognition, semantic callbacks, query construction,
selector export, or endpoint preparation. Missing counts remain unavailable.
Tests snapshot all construction counters and retained arena/cache sizes before
inspection. Inclusive cost fields must not be summed across ancestors. The
numeric visit preflight bound is explicitly not an executed-iteration count.

## What production actually constructs

The retained owner path is:

```text
original physical access descriptors
  → identity sequence → guarded regional sequence
  → phased repetition (four outer phases, plus a separate empty suffix path)
  → numerical chain composition of four specialized finite bodies
  → numeric-atom lifetime scan → explicit per-pipe rank reduction
  → numerical boundaries and regional query callbacks
  → cross-visit bridges / repeated boundary reduction
  → owned numerical/contextual endpoint families
  → allocation support → common materialization → SET/WAIT/local barrier/terminal ALL
```

For each of the four nonempty finite bodies, the inventory reports 449 payload
sites, 141 cells, 1,317 generators, and 379 reduced records. The enclosing repeated
boundary owner reports 38 crossing generators and 15 retained crossing records.
These are representation counts, not dynamic handshake counts. The root includes
both nonempty and empty/partial paths; repeated/shared owners are marked.

The selected leaf producer here is **finite numerical scanning**, not the
common-stride rotating extractor or the core-chain symbolic periodic reducer.
A phase/repetition wrapper does not mean every nested loop was kept periodic.

Responsible code:

- `Shared/PhasedRecognition.cpp::phasedChild` qualifies phases and requests body mathematics.
- `SessionProducers.cpp::specializedNumericDemands` retains that mathematics.
- `Structured/NumericTemplateRegional.cpp::analyzeNumericBody` scans physical atoms and reduces ranks.
- `exportNumericBody` retains numerical answers/boundaries; its sequence endpoint adapter is reported separately.
- `Shared/SequenceAdapters.cpp` and `RepeatedRegionExports.cpp` expose composition interfaces.
- `Shared/RetainedInventory.cpp` observes those owners without requesting new capabilities.
- `Shared/RegionalPalette.cpp` and `RegionalConstantPalette.cpp` contain the unchanged coalescer.

An inner seam reuses LEFT/RIGHT slot 0 after matmul `(wave=0,k=0,kk=2)` and before
extract `(0,1,0)`; slot 1 similarly crosses `(0,0,3) → (0,1,1)`. At an outer reset,
final MAT reads `(0,30,3)` precede initial loads in wave 1, and final slot-1 reads
`(0,31,3)` precede its next prefetch. FIX store in wave 0 precedes the next
initializing matmul. These are communication witnesses from the existing full
traces, not independent evidence of access extraction.

### Access-model qualification

The independent checker derives byte intervals from the supplied layout:

| Effect | Model |
|---|---|
| A/B prefetch | Exact GM rectangles and full selected MAT slots under supplied dimensions/layout |
| A extraction | Contiguous 16 KiB NZ slice at `kk * 16384` within its MAT slot |
| B extraction | Sixteen disjoint 2 KiB NZ strips, spaced 8192 bytes, offset `kk * 2048` |
| LEFT/RIGHT | Two alternating 16/32 KiB slots |
| ACC | 128 KiB; supplied matrix accumulation protection within one wave |
| Output | Exact GM rectangle under the supplied view, task guard and argument-disjointness assumption |

All checked concrete descriptors agree with these interval sets, rather than
only a conservative enclosing box. The original access descriptions are declared
regions. Native instruction footprints, out-of-bounds behavior, and the physical
validity of the declared layout/protection contract are **not** independently
qualified here. Therefore the conclusion is exactness for the supplied model,
not unconditional exact physical demand generation.

## Independent semantic checks

Eight small original nested variants pass: zero/short K and wave counts, skipped
last prefetches, an inactive outer visit, reused buffers, and counter resets.
Together they compare 730 recovered access descriptions, 203,192 potential-event
query entries, and 602 local storage-selector probes.

The oracle constructs conflicts directly from interval intersections, adds native
start/completion edges, and independently computes closure/covers. A separately
flattened copy validates recovered accesses and 226 retained flat explicit covers.
The original nested input is used for production regional queries and emission.
All four start/completion combinations and endpoint absence agree. Boundary checks
compare actual first/last writers and per-pipe boundary readers. GM selector
exports are not included in the local selector probes.

This is stronger than a checker consuming the compiler's generators. However,
it does **not** enumerate the original-identity expansion of every retained
regional crossing/internal record. Required-closure equality is checked; the
minimality of that complete raw regional record inventory remains a separate
qualification gap. Flat-reducer cover checks do not silently close it.

Matching, placement, occupancy and consumption-before-republication use the
independent command-event checker on emitted traces. Those checks validate
realization/resources, not generator extraction. The full main-kernel logical
command arrays for cores 0 and 19 agree with the pre-instrumentation arrays
(15,860 and 14,639 events). Byte-identical IR is **not** claimed: source paths and
associative AND construction order differ.

The seven other checked variants have equal selected and emitted payload order.
The structural zero-inner-body derivative has two extra completion-to-start
prerequisites. Its selected covers are `A_load[0] → A_load[1]` and
`B_load[0] → B_load[1]`. Local MTE2 barriers before the two second-wave loads also
order `B_load[0] → A_load[1]` and `A_load[1] → B_load[1]`. This is the expected
nonadjacent same-pipe placement limitation. This derivative tests synchronization
semantics, not numerical correctness of storing an uninitialized accumulator.
The retained guarantee remains `realization_order=covers-selected-order`.

An added outer replay loop selects a completion owner without an exact root query
export. The recorded obligations include GM discharge and unavailable regional
numerical interfaces. That ninth case is **unresolved**, not a passed exact check,
and does not imply that its mathematical class is intractable.

## Construction growth and allocation observation

Trip counts vary with wave count, inner body, storage geometry and nesting fixed:

| K trips | Sites per phase | Generators | Reduced records | Logical-only launcher median | With allocation support |
|---:|---:|---:|---:|---:|---:|
| 2 | 29 | 57 | 19 | 0.216 s | 0.215 s |
| 4 | 57 | 141 | 43 | 0.215 s | 0.265 s |
| 8 | 113 | 309 | 91 | 0.266 s | 0.316 s |
| 16 | 225 | 645 | 187 | 0.265 s | 0.566 s |
| 32 | 449 | 1317 | 379 | 0.366 s | 1.918 s |

These are three serial production runs per route, no audit requests. They are
launcher wall times including process startup and timeout polling; no speedup is
claimed. Instrumented runs are separate. The healthy trip sweep ran serially;
some later diagnostic runs overlapped a separately pinned worker and are not
exclusive performance comparisons.

The inner trip count enlarges the retained finite body. This is the first clear
Section 5 compactness gap: downstream recipe compaction does not eliminate that
upstream enumeration. The bounds must be stated in the supplied expanded
occurrence description, not source loop size.

Static stages are varied by unrolling only the inner body, keeping K=4 and
waves=2. Stage counts 1 and 2 exceed 120 seconds **with allocation disabled**;
stage count 4 completes. A five-second debugger sample of stage 1 reaches
`ArithmeticRegional::finiteBoundaries → State::storage/selected → integerPredicate`
and arbitrary-precision arithmetic. This attributes a reached construction
boundary, not its exclusive share of the complete timeout. The reason the
cheaper interface was unavailable still needs a targeted certificate trace.

Adding zero/one/two outer replay levels independently gives production completion
or exact outcomes; success alone is not an exact-order claim. The first added
level lacks the exact query export described above. Construction does not remain
constant merely because a syntactic loop count is small.

The existing stage ledger separates shared input, structural indexing, endpoint
preparation and allocation support. Recognition/extraction/reduction/interface
work inside a selected producer remains partly bundled; this checkpoint does
not relabel that bundle as minimum-demand-generation time. Leaf representation
counts locate growth even where exclusive substage timers are unavailable.

The coalescer's new counters show 1,537 input members, 91,609 fallback member-pair
visits, 98,265 handoff-query requests, 1,290 cache hits and 96,975 unique handoff
query constructions; envelope qualification makes 3,064 separate event-order
requests. Implication-cache requests are 201,372 with 201,370 hits. The measured
cost is primarily query construction, not hard implication proofs or summary
copying. Detailed times are in the JSON: coalescer spans, envelope spans and
nested query/proof timers overlap and must not be summed. No allocation
algorithm or protocol was changed in response.

## Prioritized gaps and next repair scope

1. **Section 5 representation/interface:** explain failed compact qualification
   at the inner K boundary and the 1/2-stage arithmetic-selector fallback. Preserve
   an already qualified numerical/repeated representation through that boundary
   where its exact premises hold. Do not add another backend or speculative full search.
2. **Validation export:** expose existing retained regional generators/covers in
   original occurrence coordinates for the final raw-record minimum-cover check.
3. **Unsupported exact export:** qualify GM effects across the replay boundary,
   or keep the exact-query capability unavailable while preserving child results.
4. **Realization scope:** preserve the covers/realization distinction. Any stronger
   same-pipe realization needs its own placement proof, not fewer guards or barriers by fiat.
5. **Unqualified physical assumptions/manuscript scope:** distinguish declared
   layout exactness from native footprints, and source compactness from bounds
   in supplied cells/occurrences. Six-ID feasibility is not a minimum budget.

No new demand-coverage, matching or physical-reuse defect was demonstrated on the
passing admitted instances. These findings justify a focused Section 5 repair,
not a Section 7 optimization milestone.

## Qualification receipt

The final incremental build passed for `pto-test-opt`, `pto-sync-input-test`,
`PTOASCompiler` and `PTOASPythonCore`. The focused gate is a fresh **7/7 pass**,
including the new contract checker, repeated allocation profiling, sequence
analysis/bridge work, repetition/storage, and finite allocation. A separate
source reviewer accepted this instrumentation/test checkpoint. No full suite
or device campaign was rerun.

The changed-code checker has no remaining real line-width finding. Its 16
`G.FMT.11-CPP` matches are false positives: the regex mistakes an inner closing
parenthesis in a condition for the end of the condition; each actual body has
braces. Six `G.CON.02-CPP` matches are `weak_ptr::lock()`, not mutex acquisition.
These were inspected rather than suppressed. The raw checker output remains
in the receipt; this is not a claim of full static-analyzer compliance.
