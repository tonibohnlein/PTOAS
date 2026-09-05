# ProtocolSync structured-repair continuation

## Checkpoints and scope

The straight-line local-frontier baseline is committed as `27d2c2c4e`.
Its frozen-input host results are 30/394 native rows with may-alias GM and
37/394 with disjoint GM arguments, including six zero-physical-work rows.
These are not results for the structured implementation described below.

The next patch starts step 1 of the approved sequence. It does not yet enable
ordinary loop repair, branch-crossing emission, static-view recovery, or
`set_validshape` extraction. No target contract is broadened.

The subsequent [hardware-grounded continuation](ptoas-protocol-sync-hardware-grounded-continuation.md)
first tightens named-barrier/MTE2 assumptions and adds an independent local
concrete scoreboard. Its new validation is separate from the earlier host
results below. Ordinary-loop and cross-region admission remain the next
semantic milestone, not an outcome of that target-hardening patch.

## Step 1: compositional memory transfer

`LocalMemoryAnalysis.cpp` recovers one shared physical UB atom partition.
`LocalMemoryRegionFlow.cpp` retains the production sparse straight-line transfer
and offers an explicit diagnostic expansion via `SyncLocalFlowOptions`:

```text
analyzeRegion(region, incoming)
    -> canonical requirements, outgoing, boundary summary
```

Sequential children consume their predecessor's outgoing state. Choice arms
each receive the same incoming state; their outgoing may-definitions and
outstanding accesses are unioned, while must-definitions are intersected.
Access IDs retain the original guard and loop ownership through the immutable
schedule. There is no lexical last-writer substitution at a join.

Each summary records incoming/outgoing definitions and outstanding readers /
writes, first exposed accesses, canonical requirement IDs and whether every
path issues a write. That last property is only about execution: it is not a
definite byte-set kill. Current instruction footprints remain conservative;
must-definitions remain empty and possible incoming contents are preserved.

Outstanding frontiers and semantic reaching definitions remain separate. A
write can replace the discovery frontier only because mandatory completion
requirements retain the earlier effects. It cannot erase a may-definition or
establish physical completion by lexical order.

Canonical IDs are built before candidate selection and are not replaced by
coverage results. The selected-world interpreter still decides which of those
requirements have certified completion supply.

When expansion is requested, cross-region state is retained for analysis and testing, but its accesses do
not enter `coveredAccesses`: the old timeline safeguards remain until both
placement and concrete verification understand that boundary. The previously
supported straight-line/common-guard subset uses sparse predecessor and
outstanding-access links without expanded history snapshots or region recursion.
Unknown footprints and recurring accesses remain outside the admitted domain.
Diagnostic recursion and explicit state-copy work are bounded. Expansion status
is `NotRequested`, `Complete`, or `LimitExceeded`; absent snapshots are not
evidence of empty reaching sets. Budget exhaustion discards partial expansion,
preserves the complete production requirement store, and does not become an
internal compiler failure. Cross-region expansion is never run implicitly by
production interpretation. The branch oracle requests expansion explicitly.

### Independent oracle

The local-memory unit executable adds branch programs with partially overlapping
512-byte footprints, reads, writes, in-place effects, and empty alternatives.
The reference enumerates both paths and all hazardous physical access pairs,
then checks reachability through the generated requirements. It does not use
production atom construction or region summaries to define expected hazards.
It also checks that no requirement orders mutually exclusive arms, conservative
writes preserve incoming may-definitions, and unverified boundaries retain old
protection. Each access's may-definition set is compared with all prior writes
on its feasible paths; the byte-set oracle also checks the shared branch atoms.
The matrix has 162 generated rows / 324 path executions, including
repeated empty-arm shapes; those are test executions, not independent topologies.

The existing 243-program straight-line byte/hazard oracle and concrete
synchronization mutations remain required. This branch oracle is not a loop
invariant or hardware test.

### Validation record

The initial follow-up was measured over `27d2c2c4e`; it is not part of that
frozen milestone's source revision. In that pre-review run, the configured build and all 40 focused
ProtocolSync lit tests pass. The unit driver passes its 243 straight-line
programs, 162 branch rows / 324 paths, missing/moved synchronization mutations,
and unknown/overflow footprint tests. The full host suite passes 1,897 tests,
with one unsupported and zero failures (1,898 discovered, 810.82 seconds,
two workers). The machine-readable result is
`build/protocol-sync-compositional-lit-results.json`; it is a local build
artifact, not a committed evidence archive. The 25 evidence-accounting tests
also pass. The changed-code checker reports six code files, zero errors and
zero warnings.

These are historical pre-review results. The hardware-grounded continuation
records the publication corrections and focused revalidation, including a
1,600-write sparse-baseline regression and explicit branch-expansion exhaustion.

No native corpus admission increase, device validation, or performance change
is claimed. C++17 bounds, explicit failure, scoped CMake and compiler-test rules
apply; changed-code checking is a prefilter, not a substitute for semantic
review or a target-specific static analyzer.

### Laptop iteration policy

Keep the two-worker aggregate limit and batch related source edits. Use
`cmake --build build --target pto-protocol-sync-local-memory-test --parallel 2`
for the analysis/oracle loop. Build `PTOASCompiler` only when CLI checks need a
new compiler; request other test or binding targets only when affected. The
existing targets already provide this separation, so changing the global
link/dependency graph is unnecessary. Run the smallest relevant test first,
then the focused suite at a semantic checkpoint. Do not repeat a full build or
the full host suite for small assertion, formatting, or documentation changes.

## Step 2: unconditional single-loop repair (not implemented)

The [single-loop occurrence foundation](ptoas-protocol-sync-single-loop-occurrences.md)
implements the initial requirement-discovery slice as an opt-in analysis with
an independent bounded oracle. Native recurring repair and its concrete token
verification remain unimplemented; this does not change loop admission.

Add explicit same-iteration and carried occurrence relations, including
same-static-phase backedges. Compose live-in, body, zero-trip bypass and exit
states, retaining outstanding readers as well as writes. Canonical obligations
must describe cross-boundary effects before old protection is removed.

The initial native strategy should be an atomic, conservative recurring control
handoff over understood phases, independent of storage-channel recognition.
Its proof must cover event consumption before rearming, priming, arbitrary-trip
induction and draining. Reuse E/F's qualified token and allocation machinery;
do not infer storage capacity from control-token capacity and do not introduce
an unqualified body `PIPE_ALL` repair.

The concrete verifier must reconstruct the recurrence from actual sets,
waits, barriers and loop boundaries without planner tags. General recurring
repair cannot pass by falling through the current strict-E reconstruction.

## Steps 3 and 4: participation and frontend recovery (not implemented)

Introduce explicit participation relations and guarded reaching states before
placing synchronization across joins. Use balanced branch-local mechanisms or
certified common frontiers; never emit an unconditional wait for an optional
signal. Compose accepted choice and loop summaries only after each is verified.

Alongside those changes, model descriptor-state updates from `set_validshape`,
recover bounded views and forwarding, and preserve authoritative physical-core
facts. Retain both GM alias contracts and distinguish local completion from GM
publication. Queues, communication, ACC/proxy and unknown domains remain separate
contract work.

## Step 5 and acceptance gates

After the structured baseline works, broaden motion, nested summaries, event-ID
reuse, modulo slots and optional protocol certificates. Recognition or optional
allocation failure must not reduce admission on the declared direct subset.

Each semantic extension needs its independent verifier extension, bounded
path/iteration differential tests, zero/one/odd/even/varying trip counts,
partial-overwrite and skipped-producer cases, and mutations of guards, backedges
and live event IDs. Bounded execution is an oracle, not arbitrary-trip proof.
Report per-row native admission in both GM modes with fallback disabled; do
not infer coverage gains from overlapping blocker counts.
