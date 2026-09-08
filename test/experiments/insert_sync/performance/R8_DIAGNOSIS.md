# Why R8 does not improve the benchmark

Analysis date: 2026-09-08. R8 source: `4c19cc1cba444f28ab9af2a022aae1e3c80079eb`.
Covering-performance source: `5eab65e919ca053cc93a96ab09efd3ffa434a70b`.

Six of the eleven fixtures leave the optional lifecycle constructor before it
can select anything. The four controls select protocols but match the current
lifecycle-disabled baseline's mechanism counts. GEMM selects useful partial
protocols, then discards them during combined event checking and allocation.
The absence of emitted improvement therefore has several concrete causes.

These findings concern the optional revision's admission and proof machinery.
They do **not** establish missing synchronization or incorrect execution in
production InsertSync. No device performance claim follows from this analysis.

## Native evidence and reproduction

Existing compiler binaries were reused. No compiler source was edited and no
full build was run. The R8 native library SHA-256 was
`d9d42296133e7c3dfaaac71ebe181b2fa8b6f5566bed780b495452d84e7c7158`.

The local evidence directory is:

```text
/home/toni/work/pypto3_sync_more/insertsync-builds/campaign/r8-diagnosis-4c19cc1c
```

`diagnostics.json` records commands and remarks for all eleven fixtures. Each
fixture directory contains the actual IR dumped immediately before
`pto-insert-sync`, the pass output, and diagnostic remarks. The normal CLI
suppresses useful remarks; replaying that captured input with `pto-test-opt`
exposes them:

```sh
"$PTO_TEST_OPT" before.pto --mlir-disable-threading \
  --mlir-print-op-on-diagnostic=false \
  '--pto-insert-sync=gm-alias=assume-disjoint-arguments effect-coverage=report defer-same-pipe=true mmad-chains=true lifecycle-synthesis=true' \
  -o pass.pto 2> remarks.txt
```

The option in that command is `mmad-chains=true`. FlashAttention required the
existing benchmark printer bridge, `measure.normalize_gm_pipe_assembly`, to
round-trip its pipe assembly; semantic attributes were preserved and the
normalization count was recorded.

`lifecycle_probe.cpp` is a diagnostic copy of the constructor translation unit,
linked against the existing native library. It adds logging and dumps rejected
clones, without changing selection or checking. `probe.log` and `attempt-*.pto`
retain that evidence. `event_probe.cpp` applies the same structured event oracle
to an already synchronized function. These standalone probes were compiled
serially outside the source checkout.

## Where each fixture stops

| Fixture | Observed optional lifecycle result |
| --- | --- |
| One buffer | Commits 2 channels; 11 supplied access pairs; first attempt |
| Two buffers | Commits 4 channels; 21 supplied access pairs; first attempt |
| Three buffers | Commits 6 channels; 31 supplied access pairs; first attempt |
| Four-use producer | Commits 4 channels; 246 supplied access pairs; first attempt |
| GEMM | Recognizes 5 channels initially; combined event checking and allocation fail; exhausts 8 attempts |
| TopK | Structural import rejects the first unsupported physical phase, `pto.tsort32` |
| Conv2D | Whole-function helper/macro veto before candidate selection |
| FlashAttention | Optional effect gate stops at `pto.initialize_l2g2l_pipe` |
| Triangular inverse | Optional effect gate stops at `pto.taxpy` |
| GDN | Optional effect gate stops at `pto.set_ffts` in both peers |
| KDA | Optional effect gate stops at `pto.set_ffts` in both peers |

The physical-phase whitelist in `StorageFrontierAnalysis.cpp:1066` admits only
load, store, abs, add, extract, move, matmul and accumulating matmul. TopK's sort
operations therefore cannot reach protocol recognition, even where legacy
translation has effects. Conv2D is rejected explicitly by the helper/macro
screen in `LifecycleSynthesis.cpp:831`. The other rows hit the revision's
effect gate. Removing the first gate alone would not establish support for all
subsequent operations or for complete cross-core queue protocols.

The controls demonstrate that construction runs and commits; equal counts are
not evidence that it was disconnected. R8 restores the extra barriers observed
in R7, but does not beat the current residual baseline's inventory. For the
remaining MTE3 barriers in the multi-slot controls, the source suggests a
different missing fact: separate iterations write disjoint rows through the
same GM output argument. Distinct-argument aliasing does not express that
occurrence-dependent disjointness. This is a source-based diagnosis to isolate
with a dependency witness, not a proven explanation of every retained barrier.

## GEMM: partial protocols are constructed and discarded

The physical Cube section imports successfully at the actual pass point. The
first attempt selects both LEFT/RIGHT operand bundles and three L1 panel slots.
The initial A-panel slot, MAT `[0,131072)`, fails with:

```text
overwrite before generation final use or qualified bypass
```

Its preload lies outside the inner K loop; consumers and subsequent prefetches
lie inside parity branches. The empty-reader bridge currently examines readers
whose **immediate parent** is a qualified `scf.for` and excludes nested optional
readers/writers (`LifecycleBoundarySynthesis.cpp:163–193`). That is insufficient
for the complete alternating-prefetch recipe, including a potentially empty
consumer loop and outer-loop reuse. The diagnostic establishes rejection; this
source structure explains a missing boundary capability, without claiming a
newly measured failing dynamic witness.

The first plan supplies 511 same-pipe access pairs through selected protocols,
but imports zero residual handoffs. Its reconstructed lifecycle checks pass.
The combined event proof fails on a **residual** event:

```text
pto.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID1>]
```

At projected node 198 the event marking is empty/consumed, but consumption is
not proved causally visible back at the source before rearm. This is not one
of the five newly assigned lifecycle streams. The residual scheme primes both
directions, waits at inner-loop entry, sets at inner-loop exit, and drains
afterward. The current projection cannot prove the required event transfer.

Applying the same event oracle independently also rejects:

- The ordinary emitted fallback, at the same directed residual event's set
  (projected node 178).
- The hand-tuned GEMM, at a guarded `FIX → M` wait (projected node 18).

See `event-attempt1.txt`, `event-fallback.txt`, and `event-manual.txt`. These
failures expose the oracle's limited proof coverage on the reference schedules;
they are **not evidence that those schedules are incorrect**. Bypassing the
checker would also not establish that the new composition is correct.

Retries first remove the B1 and B0 lifecycles while the residual ID1 failure
persists. Later attempts mix resource-assignment failures with the same event
proof failure. Excluding a bundle can rediscover its individual members as
separate channels. Thus candidate removal can increase key pressure without
addressing the original proof obstruction.

Allocation reserves every residual-used key and gives each new stream a
dedicated key for its whole scope. `allocateLifecycles` defaults to six IDs per
directed domain (`LifecycleProtocol.h:281`). It does not reuse keys over disjoint
lifetimes or jointly reassign residual and lifecycle streams. Failure here is
a limitation of this allocation strategy, not proof of hardware scarcity.

## Direct comparison with covering-performance

The existing covering-performance executable was run both on its archived lit
fixture and on **R8's captured GEMM input before InsertSync**. Both inputs give
the following covering counts. Commands, binary hash, outputs and return codes
are in `covering-comparison.json` and `covering-current-*.pto`.

All counts below are static sites. Pairs mean equal set and wait inventories,
not dynamic handshakes. Mechanisms are deliberately not summed into a score.

| GEMM plan | Sets | Waits | MTE2 | MTE1 | M | FIX | PIPE_ALL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hand-tuned reference | 53 | 53 | 0 | 0 | 0 | 0 | 0 |
| R8 emitted fallback, staged + MMAD | 44 | 44 | 3 | 2 | 4 | 1 | 1 |
| R8 first rejected candidate | 58 | 58 | 1 | 0 | 4 | 1 | 1 |
| Covering, distinct GM arguments | 56 | 56 | 0 | 0 | 0 | 0 | 1 |
| Covering, all GM accesses nonaliasing | 53 | 53 | 0 | 0 | 0 | 0 | 1 |

The first rejected R8 candidate trades more flag pairs for fewer named barriers;
its runtime and correctness have not been established. It is not an emitted
optimization. The R8 fallback is the **current staged + MMAD implementation**,
not the historical original InsertSync compiler.

Covering was invoked with `event-id-num-max=8` and either
`assume-distinct-gm-args-noalias=true` or
`assume-all-gm-accesses-noalias=true`. The latter is a stronger assumption and
must not silently replace the benchmark's contract. Under the comparable
distinct-arguments contract, covering already eliminates all named barriers
with only three more pairs than the manual reference. Its retained exit
`PIPE_ALL` means even the 53-pair result is not a complete manual match.

The relevant implemented capabilities in the covering branch are concrete:

| Missing or limited in R8 | Covering implementation |
| --- | --- |
| Initial preload and alternating A-panel generations | `buildAlternatingPrefetchProtocols` |
| Persistent L1 ownership across inner-loop invocations | `buildHierarchicalStableProtocols`, `buildHierarchicalAlternatingProtocols` |
| Empty inner loop and return of unused readiness credit | Hierarchical alternating protocol's explicit empty-loop actions |
| Stationary accumulator ownership through FIX | `buildBoundaryGuardedRoundTripProtocols` |
| Coherent nested L1 + accumulator selection | `buildCompositeOwnershipEventBundle` plus composition verification |

These live in `lib/PTO/Transforms/CanonicalSync/CanonicalSyncOwnership.cpp` at
the pinned covering commit. Recognition is in
`CanonicalSyncOwnershipAnalysis.cpp`, verification in
`CanonicalSyncOwnershipVerification.cpp`, and selection in
`CanonicalSyncCoveringSelection.cpp`. The historical GEMM test is
`test/lit/pto/canonical_sync_historical_gemm_noalias.pto`.

R8 explicitly excludes ACC from candidate storage and supports only the two
adjacent bidirectional domains, MTE2/MTE1 and MTE1/M on Cube
(`LifecycleSynthesis.cpp:120–131,217–227`). MMAD dependency elision is separate;
it does not implement the accumulator's M/FIX ownership protocol. Consequently,
fixing the current event rejection alone will not recover the manual plan.

## Next implementation should have a concrete acceptance target

1. Minimize the GEMM residual event rejection and teach the structured proof
   the required guard/loop/consumption relationships. Check the existing
   fallback and manual schedule as reference cases; do not weaken event reuse
   rules to admit an unproved new composition.
2. Reuse the demonstrated complete GEMM recipes from covering: alternating
   preload, nested panel lifetime, empty-loop handling, and accumulator release.
   Select their compatible composition before general repairs fix event shape.
3. Allocate the combined plan and make retry decisions address the actual
   failed resource or proof dependency, rather than dropping unrelated slots.
4. Require the unchanged GEMM under the distinct-arguments contract to commit
   the selected plan and reach **zero named body barriers**, retaining explicit
   pair and exit-PIPE_ALL accounting. Covering's 56-pair result is the concrete
   comparison; 53 requires explaining or proving the additional GM facts.
5. Expand native operation/effect adapters against TopK and triangular inverse,
   then address helper/queue kernels with their actual semantics. Each addition
   should demonstrate admission and a useful emitted change on its target input.

Passing generic protocol-core tests and unchanged-output regressions is not an
optimization acceptance test. The next patch needs an unchanged representative
kernel that selects, commits, and measurably changes the intended schedule;
device correctness and wall time remain the final performance checks.

## Follow-up: how general is the covering GEMM construction?

The working GEMM result establishes feasibility, but not broad generality.
The implementation contains reusable physical-slot reasoning and explicitly
specialized control-flow recipes. Inspection found no kernel-name or GEMM
tile-dimension checks in the ownership recognizers. Their structural constraints
are nevertheless significant:

- L0 operand recognition bundles exactly one LEFT and one RIGHT slot per
  consumer and requires at least two disjoint bundles. Both branch paths must
  use the same bundle set. A grouped conditional consumer is exactly one
  consumer in each arm of an if/else.
- The L1 alternating recognizer requires a zero-based, unit-step loop, signed
  remainder by two, comparison equal to zero, and exactly two alternating slots.
  Prefetch must be under the specific next-iteration continuation guard. Its
  stable L1-cycle recognition is coupled to discovery of an alternating pair.
- The full nested composite explicitly searches for one stable L1 cycle, one
  alternating L1 cycle in the same inner loop, and an accumulator cycle in the
  outer loop. Its verifier requires exactly three cycle groups and six logical
  events. It also rejects ambiguity if more than one such composite is found.

Source locations at the pinned covering commit:
`CanonicalSyncOwnershipAnalysis.cpp:646,802,1042`,
`CanonicalSyncOwnership.cpp:809`, and
`CanonicalSyncOwnershipVerification.cpp:694`.

The separate general slot-lifecycle subsystem discovers RAW readiness and
loop-carried WAR reclamation by physical extent and pipeline. Its native
emission adapter is narrower: unit/hierarchical release recipes require
distance one (`CanonicalSyncCoveringSlotRecipe.cpp:137`). A general covering
selector cannot select a good protocol that its candidate constructors never
generate.

There are positive native tests for L0 bundles, branched consumers, nested L1
readers, accumulator ownership, parity prefetch, and independent loops; negative
tests cover overlap, unknown addresses, missing consumers, and unsupported
control. Host protocol tests also exercise alternating and accumulator recipes.
These provide useful evidence for the individual recipes. They do not establish
robustness of the full historical-GEMM composition across equivalent IR shapes.

### Equivalent-expression experiment

Using the existing covering executable and the same captured R8 input, change
only the inner-loop parity expression. The loop starts at zero with step one,
so these rewrites preserve its condition on all executing iterations. Use the
same distinct-GM-arguments contract and eight-ID budget throughout.

| Parity expression at analysis entry | Recognized ownership cycles | Set/wait pairs | Named barriers | Body PIPE_ALL | Exit PIPE_ALL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Signed `i % 2 == 0` | 4 | 56 | 0 | 0 | 1 |
| Signed `i % 2 != 1` | 2 | 39 | 0 | 3 | 1 |
| Unsigned `i % 2 == 0` | 2 | 39 | 0 | 3 | 1 |

Both altered forms lose the stable and alternating L1 cycles; L0 operands and
the accumulator remain recognized. All six pass/plan invocations succeeded.
No payload operation, physical allocation, or alias assumption was changed.
This is a direct pass-entry experiment, without a preceding canonicalization
pipeline; it does not claim every frontend would deliver these forms unchanged.
Evidence is in the local diagnosis directory under
`covering-generality/results.json`, with each input, output, and printed plan.

### A simpler semantic target

For this already scheduled GEMM, recovering tile sizes, buffer placement, or
the compute schedule is unnecessary. The missing representation is the ordered
sequence of generations of each physical slot, with reaching producers,
relevant readers, and the next overwrite. At the conceptual level:

```text
produce generation g -> consume generation g -> overwrite for generation g+1
```

Readiness protects the first relation; reclamation protects the second. The
same rule describes L1 load/extract, L0 extract/matmul, and ACC update/store
boundaries. Accumulating read-modify-write chains retain their required ordering
and hardware-specific MMAD rules. They are not independent fresh writes.

A compositional branch/loop analysis could summarize these slot transitions,
including initial generations, zero trips, carried generations, and final
consumers. Equivalent predicates should normalize to the same slot selection.
Bundling should follow shared producer/consumer boundaries and compatible
lifetimes rather than a named GEMM-family combination. Existing pipeline and
event completion contracts still matter: source order alone does not prove
asynchronous completion or safe event reuse.

This is a design recommendation, not an implemented replacement or a claim that
arbitrary dynamic indexing and control are simple. It refines the earlier
recommendation to reuse covering's recipes: use their proven boundary behavior
as requirements and tests, while avoiding a copy of the hard-coded three-family
composition as the general architecture. First require unchanged results under
equivalent parity expressions, then vary slot count, reader count, preload,
loop shape, and accumulator presence independently.
