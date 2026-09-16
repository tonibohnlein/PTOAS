# M4: storage-guided construction and original observations

Base: `50e9541978c23f946c760dbe9002eb19e592431d`, applied M3.
Normative design: `main(20260915-212314).pdf`, 43 pages, especially section 10
(reference succession), sections 11--12 (completion and joint packets), and the
original-observation/recurrence contracts. This is an implementation of M4a,
M4b, and M4c for the supported interfaces below, not an unrestricted arithmetic
frontend or a new synchronization planner. `algorithm=existing` is unchanged.

## M4a: immutable reference-storage frontiers

`StorageFrontierAnalysis` owns the original Program. It accepts no candidate
synchronization plan and cannot confer acquired completion. Per-cell compressed
origin bit matrices propagate previous writers/readers and next writers/readers
to a fixed point on the existing static control graph. Positive-length witness
queries retain original site, operation, scope and backedge information. A source
equal to a target denotes a possible later visit, never an intra-occurrence
completion-to-issue self edge.

`Access::definiteWrite` defaults to false. A true value is a frontend assertion
of definite whole-cell overwrite, used ONLY to kill reference-succession origins.
A partial or may-overlap write adds an origin without killing prior origins.
The native importer does not infer this flag from its alias-witness cells. Those
witnesses are not necessarily exact atoms, so native origin tables initially
remain conservative. The causal pending histories, receipts and generations do
not consult this flag. An RMW queries incoming origins before its update.

Reader-boundary queries keep all possible preceding/following readers of the
specified engine, plus a boundary marker. A repeated reader can be both a first
and an interior reader; ambiguity is reported, not converted into a ghost guard.
Source/target corridors can pass other-engine work but stop at the relevant
same-engine payload. They are candidate locations, not motion certificates.

The `(preservesIncoming, writers, readers)` summary algebra implements sequence,
choice and unconstrained finite repetition of marginal reference origins. It does
not replace causal loop analysis. Existing mode correlations need their own
control representation; no product of all cells' histories is formed here.

## M4b: staged proposals with whole-plan rollback

The one constructor has four finite proposal tiers: nearest storage frontiers,
full same-engine corridors, relay expansion, and the original broader policy.
The final tier retains the prior M3 construction choices; it is not an exhaustive
solver or an exact-order completeness claim. `ConstructionOptions` permits a
storage-guidance ablation using the same effects, semantics and verifier.

Every tier starts from the original plan. Failed restricted endpoint/key choices
are discarded before widening. A restricted tier cannot turn failure into ALL;
only the final ordinary policy can select the existing checked conservative
realization. No stage drops effects, changes keys or relaxes a required checker
property. Existing reverse helpers and payload-free relay engines remain in the
broader tiers. Source discovery is cached on original storage/control; completion
and packet credit are recomputed after candidate changes.

`StageReport` records failed as well as successful trials, selections, protocol
repairs, elapsed host time and reason. Result counters are cumulative, including
work discarded for a conservative result. Final commands/handoffs describe only
the final plan. Same-engine fences remain ordinary residual repairs, not free
completion or event resets.

Finite repair slots and one-way relocation/acknowledgment changes still establish
construction progress. There is no arbitrary fixed-point work counter. Candidate
and state populations, repeated full replay, and distinct-footprint aliasing can
still be expensive. The guided policy is not guaranteed to improve every plan or
every compilation time; quality remains separately measurable by the paired
reference monitor.

## M4c: original cuts, uniform words, and recurring correspondence

### One qualified control interface

`Program::observed`, when present, is a finite original-control quotient. Its
sites name original physical phase classes or control-only positions. There are
no dummy payload operations. Legal cut observations name original anchors and
available original-value features. All sites with one observation execute the
same ordered command word; verification rejects nonuniform copies. Different
observations at the same anchor use a common feature vocabulary, preventing an
unconditional word and an overlapping partial predicate from masquerading as
mutually exclusive alternatives.

The low-level quotient's execution coverage is a frontend contract. Structural
validation does not prove arbitrary user-supplied predicates equivalent to an
original program. The provided normalized frontends and the native importer
supply the concrete qualifications described below. Event occupancy, reference
writer state, receipt validity and compiler histories are not observation atoms.

The existing Transfer/worklist, PrefixQuery and BundleQuery operate on this
same graph. They still track finite static phase classes, not unrolled dynamic
occurrences. `CompletionRequirement::consumerCut` separates an original phase
identity from its observed before-issue context. Prefix seeds aggregate all
reachable representatives of the same word; a canonical representative may
itself be unreachable. Unavailable or wholly unreachable emitted words are
rejected. Fresh effects still update all old remainders.

Packets can have several publication and acquisition observations. The same
complete replay establishes matching; the constructor does not assume each
static wait has exactly one static publisher. `ObservedSchema` exports a unique
word per original observation and the finite marginal publication-origin relation
for each acquisition. Reconstruction rechecks original observation identities,
actual words, complete safety, and the replayed relation. That relation is not
an affine generation-distance certificate or a substitute for causal rearming.

### Normalized portable frontends

`addStructuredBoundaryCuts` exposes original declared region entries/exits,
preheaders, branch continuations and body exits. A synthetic loop header remains
unavailable for emission. While before-regions execute on final false exits.

`makePeriodicLoop` accepts an arbitrary flat body with exact affine-modular cell
bindings and one explicitly selected period that preserves each binding. It
builds finite residue/capped-elapsed/capped-remaining control, not a bounded
number of loop visits. Modular arithmetic avoids intermediate overflow. Static
phase specializations retain a mapping to the original body operation. No
runtime counter, payload clone, new storage, or mode inference is emitted.

`refineCountedLoop` instead refines a supplied normalized region of an existing
original CFG, preserving the physical-phase population. It checks the structural
preheader/header/body/exit shape, external-entry and escape restrictions, and
qualified residue decisions. It adds finite original-value observations of
`i mod B`, `i >= B`, and `N-i > B`. Replaced unrefined sites are unavailable,
effect-free tombstones; they cannot accidentally emit unguarded commands at the
same original anchor. Disjoint scopes can be refined independently.

For every concrete nonnegative N, `(i mod B, min(i,B), min(N-i,B+1))` maps execution
to the finite quotient. A saturated remaining value either stays saturated or
enters the exact tail. The quotient includes zero trips and arbitrary finite
repetition; source N is not an iteration budget. Extra correlations with unrelated
control may be forgotten conservatively. This finite-control argument does not
complete the compact causal-domain simulation proof.

### Native supported fragment and exact read-back

Native import retains each original instruction as a potential anchor, including
scalar/control instructions and region terminators. It automatically refines
multiengine leaf `scf.for` loops with index IV, constant lower zero, unit step,
no iter_args and signed comparison semantics. It recognizes direct original
`iv % constant ==/!= literal` conditions when a single compatible existing period
fits the selected finite observation vocabulary. The optional period bound is
based on the actual target key-pool population, not an analysis-work budget.
Single-engine loops, unsupported bounds, incompatible periods and unsupported
relations retain the original conservative SCF graph. Diagnostic observation
notes record declined refinements where applicable. No generic affine solver,
full mixed-period nest product, or arbitrary first-active predicate is claimed.

Inserted guards contain only synchronization. They use original IV/upper values
available inside the normalized loop; `N-i > B` avoids overflowing `i+B`. No
payload is moved into a generated guard. Each selected word is emitted through
shared SyncCodegen, read back in its actual containing block, and compared in
order with the selected word. An independent syntax decoder checks the exact
original operands, literals, remainder/subtraction operations and comparisons
against the observation descriptor. Generated predicates/regions are removed for
original-IR identity checking and restored at their exact locations. Mutation
hooks cannot repair a wrong guard or endpoint by canonical relocation.

The NativeAnalysis API now exports native anchors for all cuts, representative
reachable `phaseCuts`, and observation notes. A phase can have several observed
contexts; the representative is not a full correspondence map. NoControlId means
that the original phase is unreachable in the qualified quotient. Stored MLIR
pointers become invalid after body replacement, as before.

Native code and new native regressions are SOURCE ONLY in this package: a full
MLIR/PTOAS build and device testing were unavailable. Their inclusion is not
qualification that every native API call, codegen word or hardware execution
has passed. The native gate in the package README is mandatory before landing.

## Reference and test boundary

The original M3 v0.8 source/certificates and bridge are unchanged. A separate
observed-control test bridge sends actual C++-constructed plans and original
finite graphs to the pinned causal and paired-order interpreters. It collects
whole reference states to convergence, not a selected trip count. Exactness is
checked on original issue/completion observables under matched issue-only core
contracts; it is not inferred from command counts or local collateral credit.

The observed bridge has an explicit test-only state-space interruption limit.
Exhaustion raises an inconclusive failure, never a pass, and is not used in the
compiler. It does not add synchronous-lane, ALL, native retirement, visibility,
macro or phase/resource adapters. These require separately matching contracts.

Tests cover all existing suites, storage write-free relations/summary algebra,
sparse versus complete finite conflict closure, weak-write retention, staged
rollback, matching/freshness, original-observation uniformity, nonexistent emitted
words, normalized zero/first/tail paths, affine stride, nested reader regions,
mixed writers and independent readers. Reference tests use unsynchronized
fixtures and the actual constructor; they do not hand it readiness/release
schemas. They corroborate the algorithms, not a complete mathematical or native
hardware simulation proof.

## Remaining work

Unrestricted predicate/arithmetic inference, correlated mixed-period nests,
a complete native affine-modular storage frontend, exact native atom refinement,
broader operation/macro/authored/visibility coverage, phase/resource protocols,
incremental replay and global scarcity/quality optimization remain separate
work. M4 introduces no UnitFlag, KEEP, reset, accumulator or phase credit.
