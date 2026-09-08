# Coherent handoff planning: native review and experimental implementation

This is an annotated copy of the supplied `COHERENT_SYNC_ALGORITHM.md`, reviewed
against `862ab811124d2e67a7f2cde992461983163f6a7a` and the native experiment in
this change. The supplied document reviewed `16d727c74`; its original proposal
is preserved below. The local Downloads copy is unchanged.

## Subsequent integrated milestone

The linear-only experiment below is retained as historical design evidence.
The later [shared requirements and publication implementation](ptoas-insertsync-shared-publications.md)
runs inside InsertSync on real loops. It demonstrates an earlier QK readiness
cut with correct empty-path retirement and a Q-projection readiness-stream
replacement. Its [results](../../test/experiments/insert_sync/performance/SHARED_PUBLICATIONS_RESULTS.md)
separate those gains from scalar overhead and remaining production-pass gaps.
It does not claim to implement every part of the general algorithm below.

## What the actual examples change about the proposal

The organizing idea is useful: retain storage requirements, propagate actual
completion forward, and propagate first-use needs backward to choose handoff
cuts. The next milestone should **not require finding a worse GEMM boundary**.

The current GEMM is 56/56 sets/waits, without named barriers or PIPE_ALL.
The eight recorded scalar scenarios show the same cross-lane completion
prefixes as manual at every physical operation. That is finite-trace evidence,
not symbolic equivalence or device timing. The extra static sites account for
first-matrix release, unused-preload return and accumulator priming. Nonempty
launches execute two more pairs overall than manual. A forced 56-to-53 target
would optimize an inventory rather than a demonstrated scheduling problem.

The remaining control MTE3 sites have also changed since the reviewed commit.
Two/three-buffer, four-use and TopK now preserve them only as arithmetic-overflow
fallbacks. At 16 trips/groups they execute 0 instead of 16/16/16/32. No complete
TopK lifecycle is needed for this improvement: shared GM partition facts already
reach residual placement.

See [the current eleven-fixture evidence](../../test/experiments/insert_sync/performance/SHARED_REQUIREMENTS_RESULTS.md).
Older R8 device timing does not qualify this emitted GEMM.

## Refinements to the algorithm's contracts

1. **A need must retain its original deadline when transferred through another
   lane.** At an acquisition, assign supplied needs to that handoff, then follow
   their provenance back through its publication. A source-prefix publication
   supplies its own lane; an acquired fact must retain its earlier supplier.
   Simply deleting supplied needs loses the obligations of upstream handoffs.
2. **Do not infer event causality from the memory demand set.** Return completion,
   publication/acquisition matching and consumption-before-rearm remain separate
   checks. A memory-redundant handoff can still enable another key's reuse.
3. **The no-additional-order test needs a stated domain.** The existing phase
   bits are must facts. For overlapping loop occurrences, comparing two such
   approximations does not prove inclusion of their actual ordering relations.
   The first experiment admits only one finite linear execution, with each
   physical phase and each key's publication/acquisition occurring once.
   Here the existing completion domain can compare matching occurrences.
4. **A rewrite is a complete transaction.** Requirements are retained from the
   seed independently of freshly discovered trial requirements. Reject a trial
   unless payload identities, original obligations, event validity, return
   completion and the ordering comparison all pass. Unknown preserves the seed.
5. **Finite realization precedes accepting a split.** The experiment uses an
   unused key from the existing directed eight-key domain. Existing keys remain
   reserved across the whole function. Exhaustion declines the split; it does
   not move a boundary or silently share another stream's key.
6. **A read-only plan view after emission is an experimental boundary.** It can
   test whether the planning decision is useful without first migrating every
   production allocator. It is not the proposal's final common plan before ID
   assignment, and should not be presented as completing that consolidation.

## Implemented experiment

Invoke the native development pass explicitly:

```sh
pto-test-opt --mlir-disable-threading \
  --pto-experiment-handoff-planning input.pto -o experiment.pto
```

Its optional `mmad-chains=true` uses the existing qualified MMAD rule. No new
production CLI flag, frontend annotation, buffer promise or kernel matcher is
introduced. Existing InsertSync output is the seed. The pass is not called by
the normal production pipeline.

The experiment uses the existing translator, shared storage/frontier snapshot,
and completion engine. A backward query retains requirement IDs and original
consumer deadlines, assigns them to actual supplying acquisitions/publications,
and follows transitive completion to earlier suppliers. This is a query over
the imported plan, not another storage-generation engine.

The native client tries three complete changes: advance a publication, delay
an acquisition, or split independent readiness deadlines. It does not invoke
the legacy frontier mover as a second owner of the same transaction. Private
point IDs map original payload and handoff actions into disposable clones;
freshly created events receive distinct identities after acceptance.

Each trial is freshly imported and checked. It must strictly remove at least
one modeled cross-lane completion obligation at a payload point, add none, keep
the seed's return completion, preserve original requirements and pass the
event proof. These facts are not a cost score. Extra flag instructions may
outweigh the removed ordering on a device.

The first positive is the proposal's generic broad readiness example:

```text
seed:       load A; load B; publish AB
            acquire AB; consume A; consume B

experiment: load A; publish A; load B; publish B
            acquire A; consume A; acquire B; consume B
```

It uses 2 pairs instead of 1 and removes the requirement that consuming A wait
for B's load completion. A consumer requiring A and B together must retain the
combined readiness. Existing generic release/readiness fixtures exercise
earlier publication and later acquisition; an extra relevant reader protects
the later release.

## Explicit limits and next migration

This first pass declines branches, loops, repeated key occurrences, dynamic
keys, unsupported effect summaries, and an unproved seed. It preserves these
functions. It does not certify the whole production pass, nor does its refusal
establish a bug in production InsertSync.

The unchanged eleven-fixture ladder must therefore be reported separately from
the linear positive/negative controls. The experiment is useful only as evidence
for its actual cut decisions, not as a claimed new GEMM optimization.

Remaining work from the proposal is still explicit:

- Transfer backward needs through guards and loop invocations, retaining
  matching generations and first/final participation.
- Connect protocol-owned and residual actions before allocation; retire the
  superseded decision for each migrated family.
- Share immutable control facts and compatible per-slice results rather than
  repeatedly recomputing them inside candidate discovery.
- Generalize finite assignment to eligible combined streams without widening.
- Add effect/region precision only when an unchanged fixture identifies the
  failing query; do not reopen the already solved MTE3 partition problem.

The earlier non-GEMM blockers still matter: helper/configuration summaries for
Conv2D, queue/peer resources for FlashAttention and GDN/KDA, and scalar publication
plus in-place updates for triangular inverse. A stronger cut planner cannot
replace those contracts. TopK's overlapping views are a different limitation
from its now-admitted sort phases. The Conv2D launch fault and FlashAttention
layout assertion from the device campaign are separate launcher/layout issues.

Native tests and the bounded campaign are recorded in
[HANDOFF_EXPERIMENT_RESULTS.md](../../test/experiments/insert_sync/performance/HANDOFF_EXPERIMENT_RESULTS.md).
Device correctness and timing for this experiment have not been run.

---

# Supplied proposal (historical review boundary)

# InsertSync: a coherent algorithm for guarded handoff planning

## Status and review boundary

Reviewed source: `tonibohnlein/PTOAS`, branch `codex/insertsync-revision-r1`,
commit `16d727c741565376ecc7b0c8cd933a492dc53b84`, parent R8
`4c19cc1cba444f28ab9af2a022aae1e3c80079eb`.

This document contains an algorithm proposal and an incremental integration plan.
It is not an implemented compiler patch, a new native benchmark, or a proof of
global scheduling optimality. The source review covered the shared flow engine,
its native consumers, lifecycle construction, MMAD discharge, concrete completion
cleanup, residual-key compaction, and recorded regression/test definitions.
No native build or device execution was performed for this review. The native
results below are the engineering agent's checked-in results.

## 1. What changed, and should be kept

The recorded current GEMM is 56 sets, 56 waits, zero named barriers, zero PIPE_ALL.
R8 was 44/44, with MTE2=3, MTE1=2, M=4, FIX=1 and one exit PIPE_ALL. The current
construction commits four L1 slots and two L0 operand bundles, not an ACC channel.
Guarded predecessor reasoning strengthens the existing qualified MMAD rule;
concrete completion cleanup removes remaining redundant boundaries. Equivalent
signed-equality, signed-inequality and unsigned-equality parity inputs have the
same inventory. This is compiler-output evidence, not a device performance result.

The new shared flow is genuinely used outside successful protocol matching.
However, sharing an engine is not sharing one canonical result: makeChannel()
still builds candidate masks, and analyzeBufferGenerations() invokes the flow
engine again, sometimes once more for a bundle. The engine repeats control-only
ordered reachability and immediate-lane-predecessor computation too.

Region transfers currently substitute reaching definitions. They are not full
transfers of outstanding reader obligations, completion or event ownership.
Successful generation construction now composes with completion cleanup and
identity-preserving residual barrier deletion, but not arbitrary event motion.
Residual key sharing checks both event causality and logical token ownership;
selected lifecycle keys are still assigned separately from remaining keys.

These are useful improvements, not reasons to restart the pass. They identify
where the next algorithm must connect existing results rather than add a rival
storage or completion engine.

## 2. Objective and fixed inputs

Fix the payload instructions, their original structured control, allocations,
physical phases, and existing alias/target contracts. Do not retile, reorder
payload, assume disjoint GM access instances, or introduce frontend promises.

The normal optimization policy is:

1. Preserve all required memory/resource/visibility and return-time properties.
2. Preserve correct event matching, participation, generation reuse and progress.
3. Do not introduce additional payload ordering compared with the accepted seed.
4. Prefer a strict removal of unnecessary ordering. Among equivalent-order
   alternatives, report mechanism costs separately and use measured costs only
   where available. There is no sum of pairs, named barriers and PIPE_ALL.

Finite-key scarcity recovery may add ordering only as a separately classified
operation. Failure of a bounded allocator does not prove that serialization is
unavoidable. This policy does not claim global maximum parallelism or minimum
runtime. Queue/dispatch overhead and instruction latency remain device questions.

## 3. Keep one semantic basis and one logical plan

Reuse current code through these logical responsibilities:

- ProgramFacts: stable physical phase/point IDs, structured control, normalized
  guards/selectors, target contracts, and plan-independent lane sequencing.
- StorageFlow: cached by physical slice and effect qualification, with reaching
  versions, reader frontiers, next replacement and supported occurrence maps.
- RequirementLedger: persistent, typed obligations over those identities.
- LogicalPlan: direct handoffs, recurring recipes and understood fixed actions
  in one inspectable representation, before physical event assignment.

Do not require a monolithic all-buffer product. A cache of compatible per-slice
results counts as a shared semantic result. Conversely, candidate-private
reinterpretation of the same payload does not.

Control-only reachability/predecessor facts should be calculated once per
immutable program version, not once per candidate storage projection. Definite
write qualifications must be explicit cache inputs or justified facts; they
must not silently depend on which recipe happened to be selected.

A minimal logical record is:

```
Requirement {
    id;
    source_effects;
    target_point;
    physical_slice;
    property;              // completion, access-order(rule), visibility, ...
    guard;
    occurrence_relation;   // source and target invocation/iteration mapping
}

Handoff {
    id;
    publication_frontier;  // guarded original program points
    acquisition_frontier;
    occurrence_matching;
    logical_streams;
    entry_body_exit_transfer;
    owned_actions;
    requirements_served;
}
```

A handoff is not identified by its physical event number. Before/after positions
refer to stable original points, not raw Operation pointers surviving cloning.
Materialization explicitly maps those points into each trial/output function.
No external IR annotations are required. Derived synchronization-control code
is permitted only when its conditions are justified by the existing input.

## 4. Three analyses with distinct contracts

### 4.1 Forward storage flow: what is required?

For each relevant physical slice and guarded occurrence:

- Read: associate the access with all possible reaching versions and record
  availability obligations for the applicable guarded alternatives.
- Definite replacement: retain ordering obligations against preceding relevant
  readers/writers, then create a new version for the definitely written bytes.
- May/partial write: retain old content alternatives for bytes not proved
  replaced. Preserve conservative ordering obligations.
- Read-modify-write: consume the old value and produce its successor. A chain
  can be one external publication episode without erasing its internal order.

A reaching-definition phi is a guarded choice, not a requirement to wait for
both mutually exclusive producers. A read combining different physical subsets
may instead need a conjunction of productions.

Sequence composes region transfers by substitution. Choice preserves guarded
alternatives. Loop closure includes entry, backedge and zero-trip identity;
regular generations are renamed symbolically at backedges. Unknown nested
invocation relationships remain unknown. Do not flatten all loops into one
iteration counter.

Value-flow changes do not establish asynchronous completion. Sparse ordering
frontiers may be retired only with their obligations preserved and with valid
property-specific composition, not because a may-write magically completed
older readers or definitely replaced their contents.

### 4.2 Forward supply: what does THIS logical plan establish?

At each destination lane/point maintain guaranteed facts about effect occurrences
or matching generations, plus event-causal state separately.

For a supported straight-line instance:

```
issue(e):
    create/advance the appropriate occurrence/version
    do not mark e complete

publish(h):
    message[h] describes completion of the actual preceding source prefix
    plus acquired facts that causally reach this publication

acquire(h):
    select the uniquely matching publication occurrence
    add its guaranteed facts to the destination's continuation
```

These are semantics of event firing and acquisition, not a claim that enqueueing
a signal on the scalar instruction stream has completed the payload.

A prefix publication includes earlier relevant physical effects on its source
lane, not only effects on the motivating buffer. The same store's local read
and GM write remain connected by its physical phase identity.

At an unconditional join, completion is a must fact across feasible incoming
paths. Where guards/phi versions are supported, retain conditional facts rather
than erasing a useful selected-generation relationship. Unknown alternatives
never become a convenient proof. Loop invariants require an entry/base case
and a backedge step; the loop cannot establish its own initial readiness.

Keep all-prior-static-phase completion as a strong conservative query. Add
matching-generation facts only when the correspondence is proved; do not weaken
the old bit's meaning or infer correspondence from slot count alone.

### 4.3 Backward need: what does a handoff actually have to protect?

Propagate required properties backward through the current logical plan. In a
straight-line case, ignoring guards/renaming in this notation:

```
NeedBefore(payload q) = NeedAfter(q) union RequirementsAt(q)
NeedBefore(acquire h) = NeedAfter(h) minus GuaranteedBy(h)
```

Subtraction uses proved implication between typed facts, not untyped phase IDs.
At a branch, preserve the may-needed properties with their guards. Across a
loop boundary, substitute the generation and invocation mapping. Include
logical-event participation and causal requirements as separate needs.

The need assigned to h determines which producers really must precede its
publication and which consumers really need its acquisition. Alternative
suppliers can be chosen deterministically, but removing one requires recomputing
the combined proof. No cached singleton-coverage union is authoritative.

These are coupled analyses of one plan. Forward supply tells backward need which
requirements are already established; backward need proposes weaker/earlier/later
handoffs; every changed plan gets a fresh or correctly invalidated supply result.

## 5. Concrete placement rule: source prefixes and destination cuts

On one fixed linear execution, a P->Q handoff placed after source prefix a and
before destination cut b imposes a rectangle of order:

```
completion of every relevant P operation in prefix(a)
    precedes
execution of Q operations following cut(b)
```

This is a useful algorithmic view. Combining two demands by taking the latest
producer and earliest consumer enlarges that rectangle and can add blocking.
Do not perform that combination merely to save a flag pair.

For a known missing need, choose the earliest materializable publication after
all its required producers and the latest acquisition that precedes its first
relevant consumer on each participating path. If these are conditional sets,
retain guarded frontiers rather than collapsing them to a common region entry
or exit. For different consumer lanes, do not model one consumable hardware
flag as a broadcast.

For each proposed new/widened acquire, a useful sufficient no-extra-blocking
check is that its entire actual publication-prefix guarantee was already
available at that point under the old plan, with matching guards/occurrences.
A corner requirement alone is not sufficient: a real publication covers more
than one operation. Complete boundary and token effects must also be compared.

### Example: split, do not widen

```
old P: A; B; publish(AB)
old Q: acquire(AB); useA; useB

new P: A; publish(A); B; publish(B)
new Q: acquire(A); useA; acquire(B); useB
```

The new plan may use more pairs while allowing useA to proceed before B
completes. Need analysis discovers the separate consumers; prefix analysis
checks what each concrete publication carries. The opposite merge is rejected
unless B's completion before useA was already established independently.

## 6. Rewrite a complete accepted plan, not disconnected mechanisms

The production migration should seed the logical plan from today's supported
lifecycle-plus-residual construction. It does not require a new universal
constructor before improving any kernel.

Supported rewrite families are:

- delete a residual barrier or a complete redundant handoff;
- advance publication to its required producer/final-reader frontier;
- delay acquisition to the first real consumer/overwrite frontier;
- split a broad handoff into independent readiness/release boundaries;
- factor repeated demands into an existing complete recurring recipe;
- merge only semantically compatible handoffs that introduce no extra ordering.

A recurring rewrite owns initialization, body, guarded alternatives, bypasses,
termination, and matching streams as one unit. It must not move one repeated
wait independently of its once-only publication.

No transformation is justified only by being able to prove the resulting
program safe. A safe but more serialized plan is not the ordinary optimization
we intend. Compare the modeled payload ordering before and after the rewrite.

```
plan_sync(function):
    facts = import_and_analyze_existing_IR(function)
    requirements = derive_and_retain_requirements(facts)
    plan = build_current_supported_logical_plan(facts, requirements)
    check_required_order_and_logical_participation(plan)

    work = changed_or_blocking_handoffs(plan)
    while work and within_budget():
        unit = work.pop_deterministically()
        supply = solve_or_update_forward_supply(plan)
        needs = solve_or_update_backward_needs(plan, requirements, supply)
        proposals = frontier_rewrites(unit, needs, facts)

        for rewrite in deterministic_order(proposals):
            trial = plan.replace_complete_unit(rewrite)
            trial_supply = recompute_affected_supply(trial)
            if not satisfies(requirements, trial_supply): continue
            if not valid_logical_participation(trial): continue
            if not proves_no_extra_blocking(trial, plan): continue
            if not strictly_improves_order_or_equivalent_cost(trial, plan): continue
            plan = trial
            invalidate_and_requeue_affected_users()
            break

    assignment = assign_all_logical_streams_without_moving_frontiers(plan)
    if unresolved(assignment):
        restore_a_previously_feasible_candidate_or_report_resource_failure()
        # Any serialization recovery is explicit and separately evaluated.
    output = materialize_on_clone(plan, assignment)
    reconstruct_and_check_output(output, requirements, facts)
    commit(output)
```

The seed and proposal catalogue may be incomplete. Unknown proofs preserve the
accepted plan. A bounded worklist of accepted improvements terminates through
an explicit attempt limit and a canonical plan fingerprint; do not assume a
simple instruction-count potential orders all schedule improvements.

## 7. Loop lowering and boundary cases

For a generation g with first consumers F and final readers L:

- Publish readiness from its applicable producer frontier.
- Acquire once per relevant consumer-lane episode, before its first actual use.
- Reuse that acquired generation fact for later reads of g on that lane.
- Publish reclamation after the final relevant read on each required lane.
- Acquire all required reclamation before the next conflicting overwrite.
- Preserve generation state across regions until the actual next reuse; do not
  reset it merely because an inner-loop invocation ended.

For `0,1,0 | empty | 0,1`, slot 0's reuse correspondence follows the accesses,
not an assumed iteration distance of two across an outer boundary.

If a production has no consumer, the producer may still be outstanding.
An unused ready token must either not be published under an available proved
guard or be consumed by a complete checked bypass before reuse/exit. A future
condition not available at the proposed publication cannot be used there.

A semantic final reader can depend on history not expressible by available SSA
at an early point. Then early publication is not yet a legal construction.
Retain a later boundary and explain the representability limitation rather than
pretend semantic last use always yields an executable guard.

For multiple consumer lanes, use a conjunction of releases or an existing proved
transitive completion path. Do not serialize those consumers merely to make
one lexical last reader convenient.

## 8. The GEMM relationships the algorithm must preserve

```
L1:
    panel production -> relevant extracts
    final relevant extracts -> next same-slot panel overwrite

L0:
    required LEFT/RIGHT productions -> matrix consumer
    final matrix readers -> next same-slot operand overwrite

ACC:
    initialization/update values -> final published value -> FIX
    FIX read completion -> next conflicting accumulator replacement
```

An extract participates in two structures: it reads an L1 generation and
produces L0 contents. A matrix operation consumes L0 and updates ACC. Physical
phase identity lets full completion feed all relevant structures. It does not
make all their first/final frontiers identical.

The initial optimization target should be a handoff where the current emitted
56-pair GEMM adds order that the manual reference does not, or delays a release.
If no such difference is found, do not force a different abstract plan merely
because the manual uses 53 pairs. Isolate device/runtime and exit differences.

## 9. Combined finite-ID assignment

Keep IDs absent from semantic handoff identities. Partition assignment by actual
physical context and directed target event domain, with authoritative fixed
reservations.

A quick interference approximation can use guarded live ranges. Sharing requires
more: every possible rearm must follow consumption of the preceding generation,
and each wait must still consume its own intended logical stream. Overlapping
or unknown lifetimes interfere. Lexical interval separation alone is insufficient.

Use deterministic greedy coloring with bounded backtracking or recoloring on
failed domains, followed by whole concrete-event reconstruction. Pairwise checks
are filters, not replacements for validating the entire assigned event program.

This generalizes current residual compaction to all eligible streams. It must not
change publications/acquisitions to make a coloring fit. Failure should identify
streams, reservations and the unresolved relation, not delete an unrelated
buffer. Removing a supplying recipe requires rebuilding/restoring its residual
requirements before another attempt.

## 10. Correctness and trust invariants

1. Every original effect and relevant condition remains represented or is left
   under the established conservative path.
2. Every accepted omission has a property- and occurrence-qualified derivation.
3. No candidate proves itself with synchronization that it is deleting.
4. Generation changes do not retire asynchronous work; intrinsic MMAD order is
   not full matrix/operand completion; event legality is not GM visibility.
5. Matched event participation and reuse are checked across entry/body/exit.
6. Normal optimization introduces no proved additional payload ordering.
7. Materialization preserves original payload and maps stable points explicitly.
8. Fresh reconstruction distrusts planner acceptance bits and physical-ID wishes.
9. Unsupported optimization does not become a new production admission gate.
10. Host/compiler checks, scalar replay and device results are reported separately.

## 11. Small next patch, not a replacement framework

First migrate one supported handoff family through the entire logical path:

1. Introduce a plan view with stable point/action/requirement identities over
   existing residual SyncOperations and lifecycle actions. A no-op adapter must
   preserve output. Import identities before residual allocation, not afterward.
2. Compute backward handoff needs using the existing forward combined supply.
3. Report extra prefix/early-cut ordering for current GEMM versus manual.
4. Replace one demonstrated broad or delayed handoff using a complete rewrite.
5. For that migrated family, remove its old independent motion/representation;
   retain established handling elsewhere.
6. Recheck both required order and absence of new blocking, then lower with
   unchanged or jointly assigned event resources.

Keep current source-flow improvements. Hoist repeated control-only work and
make candidate masks views over shared accesses before expanding the pattern
catalogue. Broader partial-write, reader-transfer or nested occurrence precision
should be added for an explicit native failing query, not as prerequisites to
this entire migration.

## 12. Acceptance and evidence

Native positives: unchanged GEMM and three equivalent parity forms; original
controls; one chosen demonstrated boundary improvement. Negatives: independent
readiness A/B; extra reader on another lane; history-dependent final use; stale
same-site generation; mismatched invocation; partial overwrite; missing return
acknowledgement; shared-key token theft; zero-trip unread production; MMAD rule
incorrectly used for operand release.

Retain sets, waits, barriers by pipe, body/exit PIPE_ALL, key footprint and proven
extra-order witnesses separately. An unchanged no-op refactor and a changed
optimization have different acceptance gates. Device correctness/progress and
isolated timing are still required for performance claims.

## Source anchors checked

At the reviewed commit:

- `include/PTO/Transforms/InsertSync/BufferGenerationAnalysis.h`
- `include/PTO/Transforms/InsertSync/StorageFrontierAnalysis.h`
- `include/PTO/Transforms/InsertSync/StorageFrontierQueries.h`
- `include/PTO/Transforms/InsertSync/SyncSlotMapping.h`
- `lib/PTO/Transforms/InsertSync/InsertSyncAnalysis.cpp`
- `lib/PTO/Transforms/InsertSync/MmadChainAnalysis.cpp`
- `lib/PTO/Transforms/InsertSync/StorageFrontierAnalysis.cpp`
- `lib/PTO/Transforms/InsertSync/LifecycleSynthesis.cpp`
- `lib/PTO/Transforms/InsertSync/PTOInsertSync.cpp`
- `test/lit/pto/insert_sync_buffer_generations.pto`
- `test/experiments/insert_sync/performance/BUFFER_GENERATION_RESULTS.md`
- `docs/designs/ptoas-insertsync-buffer-generations.md`
- `docs/designs/ptoas-insertsync-next-improvements-plan.md`

External conceptual references: LLVM MemorySSA documentation for cached
memory-use/definition queries; MLIR dense forward/backward data-flow analyses;
Huawei's synchronization-control documentation for queue-local waits and
pipeline-prefix publication/barrier effects. These motivate components, not a
claim that an existing LLVM algorithm already solves this target problem.
