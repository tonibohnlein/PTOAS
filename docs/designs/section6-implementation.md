# Compact demand analysis implementation

## Contract

The shared modeled accesses are authoritative. This work adds no address-precision
admission gate, instruction whitelist, or second footprint-recovery implementation.
Demand reduction preserves the selected modeled order; a completed covers
reduction returns its minimum generators. Section 6 additionally
constructs a sound bounding order when the compact exact routes do not apply.
Missing endpoint or allocation exports do not justify replacing already computed
exact demands with a different order.

Physical allocation uses one pool of at most six numeric IDs. Scarcity repair,
device experiments, and changes to the paper are outside this plan.

## Milestones

1. Finish the previous plan's milestone 7: reusable numerical chain interfaces
   and hierarchy queries, sparse guarded periodic frontiers, query-driven finite
   allocation, and a pinned corpus campaign.
2. Add Section 6 interfaces, requirement-group provenance, and certified partial
   reduction. Keep unresolved generators; delete only with an actual alternative
   path certificate.
3. Implement compact writer/reader generation: circular overwrite indexing,
   qualified two-sweep extraction, and the weighted control-origin engine.
   Consolidate latest sources, reduce, replace surviving local demands by the
   consumer-adjacent barrier policy, and reduce again.
4. Generate definite lower facts and excess certificates, including periodic
   frontier sums, support differences, and uniform exactness promotion.
5. Implement executable conditional matching for balanced slots and skipped
   sites. Prove source presence and guard availability; retain guarded demands
   when an unconditional reduction would be unsound.
6. Compose exact and bounding regions, preserving storage selectors and upper
   and lower reachability. Count cross-region excess as well as internal excess.
   Replace exact requirement groups before global identity merging.
7. Repeat certified bounding region graphs without unfolding visits. Export
   nested endpoints and queries, and compute excess by threshold sweeps.
8. Integrate dispatch, insertion, and allocation for successful logical plans;
   rerun the pinned corpus and classify remaining gaps explicitly.

Each milestone needs focused validation and acceptance by two independent
reviewers, followed by its own commit before the next milestone begins.
Validation commands obey the aggregate two-worker limit. Corpus cancellation,
if needed for machine safety, is recorded separately from class rejection.

## Status

Milestone 1 has passed two independent reviews, focused validation, and the
complete paired corpus comparison. This milestone is complete; Section 6
implementation continues with milestone 7. Milestone 1 is committed as
`3890123bb`; milestone 2 has passed validation and two independent reviews.
Milestone 2 is committed as `b0d71f7a5`.
Milestone 3 is committed as `ad83e6c2d`.
Milestone 4 is committed as `a5d5af0db`.
Milestone 5 is committed as `a213bf6ae`.
The previous six completed milestones are recorded in
`section5-recognizer-update.md`.

## Milestone 1 implementation

### Numerical composition

Sequence results retain immutable native-chain indices. A parent projects the
child rows and activates crossing records through numerical sweeps, rather than
repeating the child's semantic queries. Arbitrary-event queries descend only
the two endpoint paths and propagate threshold vectors through the hierarchy.
Graph-changing conditional, overlay, and repetition adapters discard the old
index. Missing requested ports or symbolic contexts retain the existing query
path; they do not justify reinterpreting a stale index.

### Guarded periodic frontiers

The quotient retains start-origin and completion-origin frontier rows per pipe:
`O(kV)` output entries and `O(g+kVE)` circuit gates beyond the supplied expression
DAG and endpoint grouping. Scalar event thresholds are constructed on demand.
Potential native positions are latent wires; intrinsic and demand transitions,
source seeds, and queried endpoints retain their presence guards. This preserves
native order when intermediate payloads are absent, including singleton wraps.
All scaled-score arithmetic is checked against the representation's bounds.

### Finite allocation

Explicit plans export `O(kh)` rank profiles in source order. A successful greedy
assignment checks at most `E` lane tails per handoff, without building the full
reuse relation. Only an exhausted pool materializes that relation and invokes
the existing exact matching fallback. The validated matrix API remains available.
The decoder uses one supplied subset of IDs 0 through 5. Existing hidden-macro
reservation obligations remain unchanged. No scarcity repair is introduced.

### Verification

- Serial dependency rebuild and both local tool links passed with project
  warnings enabled. The missed phased-frontier rebinding adapter was corrected
  and independently reviewed.
- The hierarchy oracle checked 96,768 event pairs against independent dense
  graphs. The four-leaf check verifies that a parent reuses child indices without
  repeating their semantic queries.
- Guarded frontiers passed 139 valuations, independent dense and unfolded graph
  checks, omission/wrap cases, arithmetic boundaries, and sparse operation counts.
- Finite allocation passed 2,048 brute-force order comparisons, compact profile
  checks, a 4,096-handoff query-count check, and 20 command traces.
- Existing regional checks passed 123,552 event queries, 27,264 phased queries,
  and 1,583 allocation envelopes. Sequence, regional allocation, nested allocation,
  and varying-loop traces passed (32, 15, 21, and 20 respectively).

The corpus runner pins input and executable hashes, records command options, and
separates analysis, allocation, and emission outcomes. This campaign uses
`gm-alias=may-not-alias`, one CPU per runner, and an inherited 6 GiB address-space
limit. Watchdog cancellations and process signals are distinct from compiler
failures. The C++ emitter is a separately pinned unchanged build; the analysis
and allocation tool contains this milestone's production objects.

### Pinned corpus comparison

All 786 prepared inputs were run through both binaries. Input and executable
hashes remained unchanged, and there were no outcome differences:

| Stage | Baseline successes | Updated successes | Other outcomes in each run |
|---|---:|---:|---|
| Analysis | 513 | 513 | 250 compiler failures; 23 watchdog cancellations |
| Six-ID allocation | 441 | 441 | 72 compiler failures; 273 not run |
| C++ emission | 439 | 439 | 2 compiler failures; 345 not run |

These are pass outcomes, including modules left unchanged because they already
contain synchronization; they are not counts of newly recognized classes.
Allocation failures include unavailable certificates as well as capacity
failures. No minimum-capacity claim follows from an unavailable certificate.

Paired successful analysis runs had median 0.115 s in both versions and p95
3.069 s / 3.119 s. Allocation medians were 0.115 s and emission medians 0.165 s
in both versions. These single-run wall timings include subprocess polling and
concurrent machine effects; they establish no measured speedup. The focused
operation-count checks establish the avoided work.

Campaign artifacts are under `.local/section6-implementation/`, in
`m1-baseline-final/`, `m1-updated-final/`, and `m1-comparison.json`. Both runs
used manifest hash
`84df1e2ae9d8616458f7281e8e271f686f8da39e07b0a018eb6fab961f2ed76f`.
Earlier partial campaigns in that directory are superseded and excluded.

## Milestone 2 implementation

The bounding interface binds lower and upper selected-order queries to one
immutable occurrence context and the shared access model. It clears stale
query and numerical exports, validates fresh numerical index shapes, and
permits mathematical results without endpoint or allocation exports.

Requirement provenance retains original, unreduced cell and prerequisite
ownership. Replacement evidence binds a producer's proof to the complete
original group, context, domain and records. The binding API is a trusted
producer boundary, not an equivalence checker; concrete proof-producing
composition adapters belong to milestone 6.

Partial reduction deletes only on a native path or an actual path through a
strict intermediate event in the unchanged selected graph. Unknown queries or
forwardness leave the original generator in place. Simultaneous removals keep
the same closure and the original provenance.

Both independent static reviews accepted the interfaces, provenance, partial
reducer and integration. Serial compilation and tool links passed. Focused
checks passed immutable-context and malformed-index cases, guarded provenance
binding and ownership cases, and 2,144 independent partial-reduction graph
valuations. The compliance check's 92 brace findings were regex false positives;
balanced-parenthesis inspection confirmed braced bodies. Whitespace checks
passed. These components are not yet a dispatched Section 6 fallback.

## Milestone 3 implementation

The fixed-body constructor indexes the next definite overwrite in expected
`O(m+z)` work. It intersects explicit source-specific distance bounds, keeps
the latest selected source per consumer and source pipe, reduces those records,
replaces only surviving local sources by the consumer's immediate pipe
predecessor, and reduces again. It exports the final selected-order quotient,
constant-shift logical endpoint recipes, original candidate ownership and local
replacement mappings. The separate two-sweep entry point is qualified only
when no source-specific filtering applies.

The control-origin backend computes exact interval bounds in its finite
abstraction using 0/1 shortest paths, reachable SCCs and condensation longest
paths. Definite overwrites remove through edges while preserving input queries.
Positive-cycle descendants have infinite upper distance. This backend does not
by itself prove source presence or executable matching for skipped payloads.

The fixed-body input adapter consumes shared modeled effects and existing
protection/prerequisite analysis. Uniform numeric ranges preserve disjointness;
unknown geometry and unresolved cross-visit relationships join wider classes.
No range is promoted to a definite overwrite. Native scalar prerequisites stay
native, and crossing prerequisites remain separate boundary obligations.
Effect-pair comparisons are charged explicitly; shared protection and original
syntax traversal are separate prepasses.

Serial compilation and both tool links passed. Independent oracles passed
667 control-distance cases and 2,056 finite graph envelopes, including zero
trips, kills, filtered sources, delayed local replacement, startup loss,
permanent native prerequisites and binary distance `2^40`. Shared-input checks
passed five fixtures covering numeric disjointness, cross-visit aliasing,
unresolved geometry, scalar prerequisites and structural rejection. Two static
reviewers accepted production and integration. The compliance checker's 73
brace findings were confirmed to have braced bodies; whitespace checks passed.
These constructors are not yet connected to the production fallback dispatcher.

## Milestone 4 implementation

Lower-fact generation checks supplied completeness and identity-writer evidence and every
mandatory read alternative, then emits the oldest certainly required source.
Full-class and rotating full-write facts retain their explicit physical and
protection premises. Missing proofs skip that lower candidate; they do not
reject the upper construction. Original group and descriptor ownership survives.

The numerical excess engine uses `O(mk)` frontier entries and `O(k^3)` work for
positive-length pipe support. It computes exact wide-integer finite-prefix counts, the quadratic
support coefficient, and Gamma/linear bounds when support agrees. It checks
lower containment and common native prerequisites. No loop grows with a
numerical trip count or distance, and no all-type-pair query table is built.

The fixed-body adapter owns both graph snapshots and their query views. Uniform
frontier equality can promote the selected upper covers to original-model
minimum demands. A zero count on a short or empty execution cannot promote
unequal frontiers. Unavailable exports preserve mathematical records.
Contradicted lower certificates retain an already available upper view and
withhold the invalid lower view.

Two independent reviewers accepted production, tests and integration. Serial
compilation and tool links passed. Lower-fact oracles, 972 independent finite
excess/profile cases, huge trip counts, support/diagonal and malformed-index
checks passed. Bounds queries and immutable-snapshot tests passed all five
shared-input fixtures. This milestone exports quality certificates and queries;
production fallback dispatch remains milestone 8.

## Milestone 5 implementation

Balanced nested branches compare their executed pipe words and retain one slot
with the original mutually exclusive alternatives. The shared extraction core
unions alternative effects and source bounds; it keeps a native prerequisite
only when every alternative pair supports it. Arm-specific requirements remain
upper software demands. No definite overwrite is inferred from a union.
Ordinary fixed bodies use this same core.

Detached endpoint preparation distributes each side of a logical record to its
original alternative cuts. Every copy keeps the same record and source ordinal;
source and target alternatives are never paired by a Cartesian product. The
original branch chooses exactly one side per visit. Finite-prefix guards handle
zero trips, short visits and the final unmatched publication.

Skipped-site matching separately establishes source presence and guard
availability. It accepts a mandatory source, identical shifted predicates, or
original enclosing-branch evidence. Payload-produced future predicates are not
recomputed. Unknown presence, unavailable placement and missing executed local
adjacency retain the mathematical records without a plan. Local skipped-demand
placement remains the existing bounded-lifetime route's separate obligation;
this adapter does not extend that route to arbitrary skipped control.

Balanced query exports use the common all-slots-present occurrence frame, with
no arbitrary branch chosen as the anchor. Enclosing invocation coordinates and
multiple-cut family allocation metadata require their own adapters; missing
exports do not discard the computed order. The optional-reader excess helper
implements only the bounded-indirect-read pattern and does not apply fixed-body
frontier sums to arbitrary skips.

Two independent reviewers accepted each production subset and the integration.
Serial compilation and both tool links passed. Actual inserted-command traces
passed zero/short-trip and changing-arm checks; shared input coverage passed
958 alternative words. Six guarded placement cases passed, along with the
fixed-body input/bounds regressions and 972 excess/profile cases. The initial
regression helper used a stale record layout; rebuilding all affected dependents
resolved it. The changed-code check's 68 brace findings were independently
confirmed as braced bodies; whitespace checks passed. Production fallback
registration and multiple-cut physical allocation remain milestone 8.

## Milestone 6 implementation

Mixed composition builds separate lower and upper sequence graphs from the same
occurrence context, retaining sign-specific access specifications, original
requirement owners and typed child records. Supplied crossings are reduced
jointly with the generated ones. Reclosure reconstructs original native value
links before clearing their temporary numerical representation. Empty children
retain the native links that skip them. Missing exports preserve original
mathematical snapshots.

Compact upper class summaries retain every access site's first and last
occurrences and complete original effect unions. They do not treat a may-write
as a kill. A generic collector compares classes through their shared effects
and generates last-source to first-target crossings for every interacting
ordered child pair. Finite siblings and already composed schemas use the same
consumer; classes remain local to their immutable owners. Native-only lower
crossings and exact finite child facts remain separate. Public physical-byte
storage capabilities are not fabricated by this class-schema adapter.
The cost charges child-pair/access-pair comparisons, tested effect-union
products, and the shared structured-protection prepass separately.

The finite replacement producer captures the original explicit scan itself,
proves equality with all original requirements of one group, and applies the
qualified revision before merging identities. It retains new typed record
definitions, other groups, fixed prerequisites, and original ownership. Unknown
proofs retain the old selection. A revised exact group's original closure
queries remain valid by the concrete equality certificate.

The numerical boundary counter counts prefix/suffix rectangle unions, including
uncertainty propagated through exact crossings. It also supplies the compressed
binary chain-index route and supports hierarchical accumulation. Qualified
fixed-body graph snapshots now produce exact pipe counts and rank selectors
without unfolding iterations; their construction cost is charged separately.
Arbitrary regional rank synthesis remains a supplied-interface obligation.

Selected-order excess does not include additional order introduced by local
barrier placement. Composition records when placement may strengthen its order;
weaker selected-order reuse proofs remain sound, but the selected excess cannot
be reported as actual-plan excess in that case.

All subsets and integration passed two independent static reviews. Serial
compilation and tool links passed. Independent checks passed 69,984 composition
event pairs, 128 finite boundary graph bounds, 256 finite replacement cases,
rectangle and hierarchy cases, large-count arithmetic, compact rank oracles,
and mixed finite/compact alias, unknown, disjoint and scalar-protected fixtures.
The common fixed/conditional input regressions passed. The changed-code check's
101 brace findings were confirmed as braced bodies; whitespace checks passed.
