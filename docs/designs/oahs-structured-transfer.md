# OAHS: obligation repair and finite structured transfer

Patch base: `b0c241f12fbb3d9532746bc47e966d3978ccc9f9`.
This series keeps `algorithm=existing|handoff`. It does not port the old planner
stack or add a persistent-lifetime recognizer.

## Obligation import

`closeStructuredSyncOrigins` is shared, read-only analysis called after the
production translator and before OAHS trusts its phases. It closes SSA storage
origins over for initial/backedge/result edges, while initial/condition/backedge/
result edges, and forwarding operations. A loop-derived view covers its reaching
whole roots: this is deliberate widening, not an exact offset recurrence solver.
The records are append-only and original phase effects are then refreshed.
Unknown origins or exhausted root propagation gain unknown-range coverage.
This hook is not enabled for the legacy planner by this series.

Each native physical effect receives an access record even when it has no other
static conflict partner. Pairwise alias witnesses supplement these records; they
are not their sole source. This preserves a lone operation's dynamic recurrence.
Pairwise witness construction remains potentially quadratic. Its existing bound
selects a stronger realization: ALL before every phase, INCLUDING phase zero,
and retirement if required. The corresponding checker demands this placement;
an absent first-phase barrier cannot pass by suppressing its missing demands.

## Actual placement

Native extraction identifies the original successor instruction and containing
block of each new command. A branch terminator is not an arbitrary substitute
for a payload cut. Commands are detached only for payload/contract comparison and
restored at their actual saved block/successor, not at a reinterpreted global cut.
The immutable original control tree determines which commands execute. Mutation
checks move a required command into the other branch and move final retirement
into an optional arm; failure must leave the input function unchanged.

## One state transfer

`OAHS/Transfer.h` is the common candidate-analysis and final-checking interpreter.
It is not a symbolic occurrence expansion. Static operation classes stand for
all dynamic visits of that operation and all its imported conservative accesses.
The state contains per-observer possibly pending classes, live publication
remainders, possible empty/full balance, validity of publication receipts, and
must-knowledge of the latest consumption of each actual directional key.

Issue adds its class to all pending sets and all live remainders. Publication
captures the source's outstanding remainder minus source-engine classes; it does
not gate later source issue. Acquisition intersects the target's pending set with
that remainder. A new consumption replaces the old consumption knowledge and
invalidates that key in every saved acknowledgment. Barriers do not consume flags.

Sequences pass state. Choices union possible histories/balances and intersect
must facts. Loops compute an ascending finite invariant containing entry and
backedge state. Preconditions are checked in the stabilized invariant, not in an
optimistic discovery iteration. For includes the zero-trip edge; while exits from
the before-region. A live publication can cross a child region when the original
control and receipt preconditions justify it; no region-entry reset is inserted.
The full proof is relative to the documented prefix/event contract, not a claim
of new hardware validation. Finite tests are supporting evidence, not a
machine-checked simulation proof.

## Construction

One deterministic repair loop analyzes the same state, supplies missing memory
completion, and then checks the complete protocol. The first placement policy
retains early producer cuts in one-visit words; general structured cases use
matched commands at the consumer's actual cut. This is intentionally conservative
placement, not a second barrier-only semantic planner.

Key choice tries unused eligible keys, then proposes reuse. Neither choice is an
acceptance fact. The complete memory plan is present before recurrence is tested,
so real storage-release handoffs can acknowledge readiness without extra replies.
Only a failed causal-consumption check can introduce a reply. Overlapping logical
balances can force an earlier publication to its consumer cut. Unresolved repair
or an optional construction limit selects a fully checked ALL realization when
available; essential final-check exhaustion still rejects. There is no legacy
fallback and no claim of minimum command count or globally optimal placement.

## Shared target and operation contracts

`PTO/IR/SyncTargetProfile.h` exposes separate A3 AIV/AIC lane/direction tables.
The source-qualified directions and static-library-safe pool 0..5 are taken from
`c73c04fb3b7a76d10ab55b48d4dcbe0270e36da2`,
`StructuredSyncCore.cpp`, `Target::supports/event/barrier`, and its header.
In particular, the AIC scalar directions and MTE2/MTE3--FIX directions excluded by
that profile are not invented. IDs 6/7 are not asserted absent from hardware.
This is a deliberately selected conservative source contract, not a newly
measured device guarantee. Source classification is distinct from hardware
qualification; new targets must supply their own reviewed profile.

The adapter resolves the core from the original kernel-kind attribute and actual
pipeline/local-storage facts and rejects contradictory mixed contexts. It keeps
the pre-existing production alias/ordinary-GM compatibility premise; it does not
claim a new general visibility theorem.

`SyncOrdinaryExternalModels.h` adds explicit, IR-owned interface models for A3
TAbs, TMul, TSub, and plain MTE3 TStore. A dialect extension registers them before
parallel pass execution; the planner does not infer completeness from absence in
an opcode exception list. The store model excludes atomic, phased, quantized,
ReLU-preprocessing and tensor-result variants. Plain TLoad/TAdd retain their
existing declarations. Direct users of the native API must register the same
extension, as the native test driver does.

## Remaining coverage work (not claimed complete)

This is not the full production-population semantic-accounting milestone.
Macro import and its may-effects/must-completion boundary, authored event
composition, explicit visibility recipes, queue/resource ownership protocols,
additional operation variants, mixed physical sections, and additional targets
remain unsupported and are rejected. Reservations and resource-exclusion cells
are handled; generic resource/visibility fields are structurally validated but
are not silently treated as byte completion. No 100% corpus claim follows.
The pairwise alias-witness representation also remains a scalability task.

## Tests and required native gate

The standalone CMake target compiles the exact production Plan.cpp and its
finite transfer header. The graph oracle independently builds launch/finish
vertices, checks all original conflicts and command-graph acyclicity, and proves
consume-before-republication by reachability WITHOUT inserting those edges as
assumptions. It expands bounded for/while/choice traces only in tests.

The native tests cover carried roots, self-recurrence, while forwarding, actual
endpoint placement, atomic rollback, ordinary external models, and real alias
budget widening (730 loads). They must be built with the repository's configured
MLIR/PTOAS toolchain. Adding their source does not mean they were executed.
See the patch package validation record for commands actually run.
