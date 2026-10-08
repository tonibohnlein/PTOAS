# Compact demand analysis implementation

## Contract

The shared modeled accesses are authoritative. This work adds no address-precision
admission gate, instruction whitelist, or second footprint-recovery implementation.
Demand reduction is exact for the selected modeled order. Section 6 additionally
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
implementation starts with milestone 2. The previous six completed milestones are recorded in
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
